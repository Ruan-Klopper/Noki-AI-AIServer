"""
Embedding routes for resource and message embedding
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends, BackgroundTasks

from app.models.schemas import EmbedResourceInput, EmbedMessageInput
from app.services.vector import VectorService
from app.auth import verify_bearer_token, verify_backend_token

logger = logging.getLogger(__name__)
router = APIRouter()


def get_vector_service() -> VectorService:
    """Dependency to get vector service instance"""
    return VectorService()


@router.post("/embed_resource")
async def embed_resource(
    embed_input: EmbedResourceInput,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Embed a resource (PDF, website, YouTube video) into the vector database
    
    This endpoint is called by the backend when a new resource is added
    or updated. The AI will ingest the content and create embeddings
    for semantic search.
    """
    try:
        logger.info(f"Embedding resource {embed_input.resource_id} for user {embed_input.user_id}")
        
        # Use async embedding method for better performance
        embedding_id, embedding_tokens = await vector_service.embed_resource_async(
            user_id=embed_input.user_id,
            conversation_id=embed_input.conversation_id,
            resource_id=embed_input.resource_id,
            resource_type=embed_input.resource_type,
            title=embed_input.title,
            content=embed_input.content,
            metadata=embed_input.metadata
        )
        
        logger.info(f"Successfully embedded resource {embed_input.resource_id}")
        
        return {
            "status": "success",
            "resource_id": embed_input.resource_id,
            "embedding_id": embedding_id,
            "embedding_tokens": embedding_tokens,
            "message": "Resource successfully embedded"
        }
        
    except Exception as e:
        logger.error(f"Error embedding resource {embed_input.resource_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed resource: {str(e)}"
        )


@router.post("/embed_message")
async def embed_message(
    embed_input: EmbedMessageInput,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Embed a chat message into the vector database
    
    This endpoint allows the backend to push chat messages to the vector DB
    without triggering an AI response. Useful for storing conversation history
    or external messages.
    """
    try:
        logger.info(f"Embedding message {embed_input.message_id} for user {embed_input.user_id}")
        
        # Use async embedding method for better performance
        embedding_id, embedding_tokens = await vector_service.embed_message_async(
            user_id=embed_input.user_id,
            conversation_id=embed_input.conversation_id,
            message_id=embed_input.message_id,
            message_content=embed_input.message_content,
            metadata=embed_input.metadata
        )
        
        logger.info(f"Successfully embedded message {embed_input.message_id}")
        
        return {
            "status": "success",
            "message_id": embed_input.message_id,
            "embedding_id": embedding_id,
            "embedding_tokens": embedding_tokens,
            "message": "Message successfully embedded"
        }
        
    except Exception as e:
        logger.error(f"Error embedding message {embed_input.message_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to embed message: {str(e)}"
        )


@router.post("/embed_resource_async")
async def embed_resource_async(
    embed_input: EmbedResourceInput,
    background_tasks: BackgroundTasks,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Embed a resource asynchronously in the background
    
    This endpoint starts the embedding process in the background and returns immediately.
    Useful for large resources that would otherwise timeout.
    """
    try:
        logger.info(f"Starting async embedding for resource {embed_input.resource_id}")
        
        # Add the embedding task to background tasks
        background_tasks.add_task(
            vector_service.embed_resource_async,
            user_id=embed_input.user_id,
            conversation_id=embed_input.conversation_id,
            resource_id=embed_input.resource_id,
            resource_type=embed_input.resource_type,
            title=embed_input.title,
            content=embed_input.content,
            metadata=embed_input.metadata
        )
        
        return {
            "status": "accepted",
            "resource_id": embed_input.resource_id,
            "message": "Resource embedding started in background"
        }
        
    except Exception as e:
        logger.error(f"Error starting async embedding for resource {embed_input.resource_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to start resource embedding: {str(e)}"
        )


@router.delete("/embed_resource/{resource_id}")
async def delete_resource_embeddings(
    resource_id: str,
    user_id: str,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Delete embeddings for a specific resource
    
    This endpoint removes all embeddings associated with a resource
    when it's deleted from the backend.
    """
    try:
        logger.info(f"Deleting embeddings for resource {resource_id}")
        
        # Note: This would require implementing deletion in the vector service
        # For now, we'll return a success response
        success = vector_service.delete_resource_embeddings(resource_id)
        
        if success:
            return {
                "status": "success",
                "resource_id": resource_id,
                "message": "Resource embeddings deleted"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete resource embeddings"
            )
        
    except Exception as e:
        logger.error(f"Error deleting embeddings for resource {resource_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete resource embeddings: {str(e)}"
        )


@router.delete("/embed_user/{user_id}")
async def delete_user_embeddings(
    user_id: str,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Delete all embeddings for a user (for data privacy)
    
    This endpoint removes all embeddings associated with a user
    when they delete their account or request data deletion.
    """
    try:
        logger.info(f"Deleting all embeddings for user {user_id}")
        
        # Delete all user embeddings
        success = vector_service.delete_user_embeddings(user_id)
        
        if success:
            return {
                "status": "success",
                "user_id": user_id,
                "message": "All user embeddings deleted"
            }
        else:
            raise HTTPException(
                status_code=500,
                detail="Failed to delete user embeddings"
            )
        
    except Exception as e:
        logger.error(f"Error deleting embeddings for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to delete user embeddings: {str(e)}"
        )


@router.get("/embed_stats/{user_id}")
async def get_embedding_stats(
    user_id: str,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Get embedding statistics for a user
    
    Returns information about the user's embeddings including
    count, types, and storage usage.
    """
    try:
        logger.info(f"Getting embedding stats for user {user_id}")
        
        # This would require implementing stats collection in the vector service
        # For now, we'll return a placeholder response
        return {
            "user_id": user_id,
            "total_embeddings": 0,
            "resource_embeddings": 0,
            "message_embeddings": 0,
            "storage_size_mb": 0,
            "last_updated": "2025-01-01T00:00:00Z"
        }
        
    except Exception as e:
        logger.error(f"Error getting embedding stats for user {user_id}: {e}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to get embedding stats: {str(e)}"
        )
