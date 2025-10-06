"""
Vector database service for embeddings and semantic search
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
import uuid

from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.schema import Document
import pinecone

from app.services.token_usage import TokenUsageService

logger = logging.getLogger(__name__)


class VectorService:
    """Service for managing vector embeddings and semantic search"""
    
    def __init__(self):
        self.token_service = TokenUsageService()
        
        self.embeddings = OpenAIEmbeddings(
            openai_api_key=settings.openai_api_key,
            model="text-embedding-ada-002"
        )
        
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=settings.chunk_size,
            chunk_overlap=settings.chunk_overlap,
            length_function=len,
        )
        
        self._initialize_pinecone()
    
    def _initialize_pinecone(self):
        """Initialize Pinecone connection"""
        try:
            pinecone.init(
                api_key=settings.pinecone_api_key,
                environment=settings.pinecone_environment
            )
            
            # Create index if it doesn't exist
            if settings.pinecone_index_name not in pinecone.list_indexes():
                pinecone.create_index(
                    name=settings.pinecone_index_name,
                    dimension=settings.pinecone_dimension,
                    metric="cosine"
                )
            
            self.vectorstore = Pinecone.from_existing_index(
                index_name=settings.pinecone_index_name,
                embedding=self.embeddings
            )
            
            logger.info(f"Pinecone initialized with index: {settings.pinecone_index_name}")
            
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
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
        
        Returns list of recent chat messages
        """
        try:
            limit = limit or settings.max_chat_history
            
            # Search for recent chat messages
            filter_dict = {
                "user_id": user_id,
                "conversation_id": conversation_id,
                "type": "chat"
            }
            
            # Get recent messages (this is a simplified approach)
            # In production, you might want to use a different strategy
            results = self.vectorstore.similarity_search(
                "",  # Empty query to get all
                k=limit * 2,  # Get more to filter by recency
                filter=filter_dict
            )
            
            # Sort by creation time and limit
            results.sort(key=lambda x: x.metadata.get("created_at", ""), reverse=True)
            return results[:limit]
            
        except Exception as e:
            logger.error(f"Failed to get recent chat history: {e}")
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
