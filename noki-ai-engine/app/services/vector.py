"""
Vector database service for embeddings and semantic search
"""
import logging
import asyncio
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime, timedelta
import uuid
from concurrent.futures import ThreadPoolExecutor
import threading

from langchain_openai import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_pinecone import PineconeVectorStore
from langchain.schema import Document
from pinecone import Pinecone, ServerlessSpec

from app.services.token_usage import TokenUsageService
from config import settings

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and semantic search"""
    
    def __init__(self):
        self.token_service = TokenUsageService()
        self._embeddings = None
        self._text_splitter = None
        self._vectorstore = None
        self._pc = None
        self._initialized = False
        
        # Performance optimizations
        self._embedding_cache = {}  # Simple in-memory cache
        self._cache_lock = threading.Lock()
        self._executor = ThreadPoolExecutor(max_workers=settings.max_concurrent_embeddings)
        self._semaphore = asyncio.Semaphore(settings.max_concurrent_embeddings)
    
    def _ensure_initialized(self):
        """Lazy initialization of services"""
        if self._initialized:
            return
            
        try:
            self._embeddings = OpenAIEmbeddings(
                openai_api_key=settings.openai_api_key,
                model="text-embedding-ada-002"
            )
            
            self._text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
                length_function=len,
            )
            
            self._initialize_pinecone()
            self._initialized = True
            
        except Exception as e:
            logger.error(f"Failed to initialize vector service: {e}")
            raise
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            # Create Pinecone client instance
            self._pc = Pinecone(api_key=settings.pinecone_api_key)
            
            # Check if index exists, create if it doesn't
            existing_indexes = self._pc.list_indexes().names()
            if settings.pinecone_index_name not in existing_indexes:
                self._pc.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.pinecone_dimension,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region="us-east-1"
                    )
                )
                logger.info(f"Created Pinecone index: {settings.pinecone_index_name}")
            
            # Set environment variable for PineconeVectorStore
            import os
            os.environ["PINECONE_API_KEY"] = settings.pinecone_api_key
            
            # Initialize vector store
            self._vectorstore = PineconeVectorStore.from_existing_index(
                index_name=settings.pinecone_index_name,
                embedding=self._embeddings
            )
            
            logger.info(f"Pinecone initialized with index: {settings.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise
    
    @property
    def embeddings(self):
        """Lazy access to embeddings"""
        self._ensure_initialized()
        return self._embeddings
    
    @property
    def text_splitter(self):
        """Lazy access to text splitter"""
        self._ensure_initialized()
        return self._text_splitter
    
    @property
    def vectorstore(self):
        """Lazy access to vector store"""
        self._ensure_initialized()
        return self._vectorstore
    
    def _get_content_hash(self, content: str) -> str:
        """Generate hash for content caching"""
        return hashlib.md5(content.encode()).hexdigest()
    
    def _is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        if not cache_entry:
            return False
        created_at = datetime.fromisoformat(cache_entry.get("created_at", ""))
        return datetime.utcnow() - created_at < timedelta(seconds=settings.embedding_cache_ttl)
    
    def _get_cached_embedding(self, content_hash: str) -> Optional[List[float]]:
        """Get cached embedding if available and valid"""
        with self._cache_lock:
            cache_entry = self._embedding_cache.get(content_hash)
            if cache_entry and self._is_cache_valid(cache_entry):
                logger.debug(f"Cache hit for content hash: {content_hash}")
                return cache_entry["embedding"]
        return None
    
    def _cache_embedding(self, content_hash: str, embedding: List[float]):
        """Cache embedding with timestamp"""
        with self._cache_lock:
            self._embedding_cache[content_hash] = {
                "embedding": embedding,
                "created_at": datetime.utcnow().isoformat()
            }
            logger.debug(f"Cached embedding for content hash: {content_hash}")
    
    async def _embed_text_async(self, text: str) -> List[float]:
        """Async wrapper for embedding generation"""
        async with self._semaphore:
            loop = asyncio.get_event_loop()
            return await loop.run_in_executor(
                self._executor,
                self.embeddings.embed_query,
                text
            )
    
    async def embed_resource_async(self, user_id: str, conversation_id: str, resource_id: str,
                                  resource_type: str, title: str, content: str,
                                  metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        """
        Async version of embed_resource with caching and batch processing
        """
        try:
            # Count embedding tokens for the content
            embedding_tokens = self.token_service.count_embedding_tokens(content)
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Process chunks in batches for better performance
            batch_size = settings.embedding_batch_size
            documents = []
            embedding_ids = []
            
            for batch_start in range(0, len(chunks), batch_size):
                batch_chunks = chunks[batch_start:batch_start + batch_size]
                batch_docs = []
                
                # Check cache for each chunk
                cached_embeddings = {}
                chunks_to_embed = []
                
                for i, chunk in enumerate(batch_chunks):
                    chunk_hash = self._get_content_hash(chunk)
                    cached_embedding = self._get_cached_embedding(chunk_hash)
                    
                    if cached_embedding:
                        cached_embeddings[i] = cached_embedding
                    else:
                        chunks_to_embed.append((i, chunk, chunk_hash))
                
                # Embed uncached chunks in parallel
                if chunks_to_embed:
                    embedding_tasks = [
                        self._embed_text_async(chunk) for _, chunk, _ in chunks_to_embed
                    ]
                    embeddings = await asyncio.gather(*embedding_tasks)
                    
                    # Cache the new embeddings
                    for (idx, chunk, chunk_hash), embedding in zip(chunks_to_embed, embeddings):
                        self._cache_embedding(chunk_hash, embedding)
                        cached_embeddings[idx] = embedding
                
                # Create documents with metadata
                for i, chunk in enumerate(batch_chunks):
                    chunk_index = batch_start + i
                    doc_metadata = {
                        "user_id": user_id,
                        "conversation_id": conversation_id,
                        "resource_id": resource_id,
                        "resource_type": resource_type,
                        "title": title,
                        "chunk_index": chunk_index,
                        "type": "resource",
                        "created_at": datetime.utcnow().isoformat(),
                        "embedding_tokens": self.token_service.count_embedding_tokens(chunk),
                        **(metadata or {})
                    }
                    
                    batch_docs.append(Document(
                        page_content=chunk,
                        metadata=doc_metadata
                    ))
                    embedding_ids.append(f"{resource_id}_chunk_{chunk_index}")
                
                documents.extend(batch_docs)
            
            # Add all documents to vector store in one batch
            if documents:
                await asyncio.get_event_loop().run_in_executor(
                    self._executor,
                    self.vectorstore.add_documents,
                    documents,
                    embedding_ids
                )
            
            logger.info(f"Async embedded resource {resource_id} with {len(chunks)} chunks, {embedding_tokens} tokens")
            return resource_id, embedding_tokens
            
        except Exception as e:
            logger.error(f"Failed to async embed resource {resource_id}: {e}")
            raise
    
    def embed_resource(self, user_id: str, conversation_id: str, resource_id: str,
                      resource_type: str, title: str, content: str,
                      metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        """
        Embed a resource (PDF, website, etc.) into the vector database
        
        Returns the embedding ID and total embedding tokens used
        """
        try:
            # Count embedding tokens for the content
            embedding_tokens = self.token_service.count_embedding_tokens(content)
            
            # Split content into chunks
            chunks = self.text_splitter.split_text(content)
            
            # Create documents with metadata
            documents = []
            for i, chunk in enumerate(chunks):
                doc_metadata = {
                    "user_id": user_id,
                    "conversation_id": conversation_id,
                    "resource_id": resource_id,
                    "resource_type": resource_type,
                    "title": title,
                    "chunk_index": i,
                    "type": "resource",
                    "created_at": datetime.utcnow().isoformat(),
                    "embedding_tokens": self.token_service.count_embedding_tokens(chunk),
                    **(metadata or {})
                }
                
                documents.append(Document(
                    page_content=chunk,
                    metadata=doc_metadata
                ))
            
            # Generate unique IDs for each chunk
            embedding_ids = [f"{resource_id}_chunk_{i}" for i in range(len(chunks))]
            
            # Add to vector store
            self.vectorstore.add_documents(documents, ids=embedding_ids)
            
            logger.info(f"Embedded resource {resource_id} with {len(chunks)} chunks, {embedding_tokens} tokens")
            return resource_id, embedding_tokens
            
        except Exception as e:
            logger.error(f"Failed to embed resource {resource_id}: {e}")
            raise
    
    async def embed_message_async(self, user_id: str, conversation_id: str, message_id: str,
                                 message_content: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        """
        Async version of embed_message with caching
        """
        try:
            # Count embedding tokens for the message
            embedding_tokens = self.token_service.count_embedding_tokens(message_content)
            
            # Check cache first
            content_hash = self._get_content_hash(message_content)
            cached_embedding = self._get_cached_embedding(content_hash)
            
            if cached_embedding:
                logger.debug(f"Using cached embedding for message {message_id}")
            else:
                # Generate embedding asynchronously
                cached_embedding = await self._embed_text_async(message_content)
                self._cache_embedding(content_hash, cached_embedding)
            
            doc_metadata = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "type": "chat",
                "created_at": datetime.utcnow().isoformat(),
                "embedding_tokens": embedding_tokens,
                **(metadata or {})
            }
            
            document = Document(
                page_content=message_content,
                metadata=doc_metadata
            )
            
            # Add to vector store asynchronously
            await asyncio.get_event_loop().run_in_executor(
                self._executor,
                self.vectorstore.add_documents,
                [document],
                [message_id]
            )
            
            logger.info(f"Async embedded message {message_id} with {embedding_tokens} tokens")
            return message_id, embedding_tokens
            
        except Exception as e:
            logger.error(f"Failed to async embed message {message_id}: {e}")
            raise
    
    def embed_message(self, user_id: str, conversation_id: str, message_id: str,
                     message_content: str, metadata: Optional[Dict[str, Any]] = None) -> Tuple[str, int]:
        """
        Embed a chat message into the vector database
        
        Returns the embedding ID and embedding tokens used
        """
        try:
            # Count embedding tokens for the message
            embedding_tokens = self.token_service.count_embedding_tokens(message_content)
            
            doc_metadata = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "message_id": message_id,
                "type": "chat",
                "created_at": datetime.utcnow().isoformat(),
                "embedding_tokens": embedding_tokens,
                **(metadata or {})
            }
            
            document = Document(
                page_content=message_content,
                metadata=doc_metadata
            )
            
            # Add to vector store
            self.vectorstore.add_documents([document], ids=[message_id])
            
            logger.info(f"Embedded message {message_id} with {embedding_tokens} tokens")
            return message_id, embedding_tokens
            
        except Exception as e:
            logger.error(f"Failed to embed message {message_id}: {e}")
            raise
    
    def search_semantic_context(self, user_id: str, conversation_id: str,
                              query: str, top_k: Optional[int] = None,
                              project_ids: Optional[List[str]] = None,
                              task_ids: Optional[List[str]] = None) -> List[Document]:
        """
        Search for semantically relevant context
        
        Returns list of relevant documents
        """
        try:
            top_k = top_k or settings.retrieval_top_k
            
            # Build filter for user and conversation
            filter_dict = {
                "user_id": user_id,
                "conversation_id": conversation_id
            }
            
            # Add optional filters
            if project_ids:
                filter_dict["project_id"] = {"$in": project_ids}
            if task_ids:
                filter_dict["task_id"] = {"$in": task_ids}
            
            # Perform similarity search
            results = self.vectorstore.similarity_search(
                query,
                k=top_k,
                filter=filter_dict
            )
            
            logger.info(f"Found {len(results)} relevant documents for query")
            return results
            
        except Exception as e:
            logger.error(f"Failed to search semantic context: {e}")
            return []
    
    def get_recent_chat_history(self, user_id: str, conversation_id: str,
                               limit: Optional[int] = None) -> List[Document]:
        """
        Get recent chat history for context
        
        Returns list of recent chat messages ordered by creation time
        """
        try:
            limit = limit or settings.max_chat_history
            
            # Build filter for user, conversation, and chat type
            filter_dict = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "type": "chat"
            }
            
            # Use conversation_id as query to get conversation-related documents
            # This works better than empty query for similarity search
            query_text = f"conversation {conversation_id}"
            
            try:
                # Get more results than needed to ensure we have enough after filtering by time
                results = self.vectorstore.similarity_search(
                    query=query_text,
                    k=limit * 5,  # Get more to filter by recency
                    filter=filter_dict
                )
                
            except Exception as e:
                logger.warning(f"Similarity search with query failed: {e}, trying without filter")
                # Fallback: try without strict filter
                try:
                    results = self.vectorstore.similarity_search(
                        query=query_text,
                        k=limit * 5
                    )
                    # Filter manually
                    results = [
                        doc for doc in results 
                        if (doc.metadata.get("user_id") == user_id and 
                            doc.metadata.get("conversation_id") == conversation_id and
                            doc.metadata.get("type") == "chat")
                    ]
                except Exception as e2:
                    logger.error(f"Fallback search also failed: {e2}")
                    return []
            
            # Sort by creation time (most recent first)
            # Filter to ensure all have created_at metadata
            valid_docs = [doc for doc in results if doc.metadata.get("created_at")]
            valid_docs.sort(
                key=lambda x: x.metadata.get("created_at", ""), 
                reverse=True
            )
            
            # Return the most recent messages
            return valid_docs[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent chat history: {e}", exc_info=True)
            return []
    
    def delete_user_embeddings(self, user_id: str) -> bool:
        """
        Delete all embeddings for a user (for data privacy)
        
        Returns success status
        """
        try:
            # This would require implementing deletion in Pinecone
            # For now, we'll log the request
            logger.info(f"Requested deletion of embeddings for user {user_id}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to delete embeddings for user {user_id}: {e}")
            return False
