"""
Chat routes for the main AI interaction endpoints
"""
import logging
from typing import Dict, Any
from fastapi import APIRouter, HTTPException, Depends

from app.models.schemas import ChatInput, AIResponse, ProjectManagementInput
from app.services.llm import LLMService
from app.services.vector import VectorService
from app.services.planner import PlannerService
from app.auth import verify_bearer_token, verify_backend_token

logger = logging.getLogger(__name__)
router = APIRouter()


def get_vector_service() -> VectorService:
    """Dependency to get vector service instance"""
    return VectorService()


def get_llm_service(vector_service: VectorService = Depends(get_vector_service)) -> LLMService:
    """Dependency to get LLM service instance"""
    return LLMService(vector_service)


def get_planner_service() -> PlannerService:
    """Dependency to get planner service instance"""
    return PlannerService()


@router.post("/chat", response_model=AIResponse)
async def chat(
    chat_input: ChatInput,
    llm_service: LLMService = Depends(get_llm_service),
    planner_service: PlannerService = Depends(get_planner_service),
    token: str = Depends(verify_bearer_token)
) -> AIResponse:
    """
    Main chat endpoint - processes user messages and returns structured AI responses
    
    This endpoint handles the complete chat flow:
    1. Processes user input
    2. Retrieves semantic context
    3. Determines if backend data is needed (intent)
    4. Generates structured response with UI blocks
    5. Saves message to vector store
    """
    try:
        logger.info(f"Processing chat request for user {chat_input.user_id}, conversation {chat_input.conversation_id}")
        
        # Process the chat request
        response = llm_service.process_chat_request(chat_input)
        
        logger.info(f"Chat response generated with stage: {response.stage}")
        return response
        
    except Exception as e:
        logger.error(f"Error processing chat request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing chat request"
        )


@router.post("/chat/stream")
async def chat_stream(
    chat_input: ChatInput,
    llm_service: LLMService = Depends(get_llm_service),
    token: str = Depends(verify_bearer_token)
):
    """
    Streaming chat endpoint for real-time responses
    
    This endpoint provides streaming responses for better UX.
    It emits multiple response chunks before completing.
    """
    try:
        from fastapi.responses import StreamingResponse
        import json
        
        async def generate_stream():
            # Emit thinking stage
            yield f"data: {json.dumps({'stage': 'thinking', 'conversation_id': chat_input.conversation_id, 'text': 'Processing your request...'})}\n\n"
            
            # Process the request
            response = llm_service.process_chat_request(chat_input)
            
            # Emit response chunks
            if response.blocks:
                for i, block in enumerate(response.blocks):
                    chunk_response = AIResponse(
                        stage="response",
                        conversation_id=response.conversation_id,
                        text=f"Generating response part {i+1} of {len(response.blocks)}...",
                        blocks=[block]
                    )
                    yield f"data: {json.dumps(chunk_response.dict())}\n\n"
            
            # Emit final complete response
            final_response = AIResponse(
                stage="complete",
                conversation_id=response.conversation_id,
                text=response.text,
                blocks=response.blocks
            )
            yield f"data: {json.dumps(final_response.dict())}\n\n"
        
        return StreamingResponse(
            generate_stream(),
            media_type="text/plain",
            headers={"Cache-Control": "no-cache", "Connection": "keep-alive"}
        )
        
    except Exception as e:
        logger.error(f"Error in streaming chat: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error in streaming chat"
        )


@router.post("/projects/manage")
async def manage_projects(
    project_input: ProjectManagementInput,
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Project management endpoint for creating, updating, and managing projects, tasks, and todos
    
    This endpoint handles CRUD operations for project management entities.
    """
    try:
        logger.info(f"Processing project management request for user {project_input.user_id}")
        
        # For now, return a simple response
        # In a real implementation, you'd integrate with a database
        return {
            "status": "success",
            "action": project_input.action,
            "user_id": project_input.user_id,
            "message": f"Project management action '{project_input.action}' processed successfully",
            "data": project_input.data
        }
        
    except Exception as e:
        logger.error(f"Error processing project management request: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error processing project management request"
        )


@router.get("/chat/history/{conversation_id}")
async def get_chat_history(
    conversation_id: str,
    user_id: str,
    vector_service: VectorService = Depends(get_vector_service),
    token: str = Depends(verify_bearer_token)
) -> Dict[str, Any]:
    """
    Get chat history for a conversation
    
    Returns recent messages and context for a conversation.
    """
    try:
        # Get recent chat history
        history = vector_service.get_recent_chat_history(
            user_id=user_id,
            conversation_id=conversation_id,
            limit=20  # Get more history for this endpoint
        )
        
        # Format history for response
        formatted_history = []
        for doc in history:
            formatted_history.append({
                "content": doc.page_content,
                "metadata": doc.metadata,
                "timestamp": doc.metadata.get("created_at")
            })
        
        return {
            "conversation_id": conversation_id,
            "user_id": user_id,
            "history": formatted_history,
            "count": len(formatted_history)
        }
        
    except Exception as e:
        logger.error(f"Error getting chat history: {e}")
        raise HTTPException(
            status_code=500,
            detail="Internal server error getting chat history"
        )
