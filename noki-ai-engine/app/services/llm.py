"""
LLM service with LangChain integration for prompt pipeline and RAG
"""
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, SystemMessagePromptTemplate, HumanMessagePromptTemplate
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

from config import settings
from app.services.token_usage import TokenUsageService
from app.models.schemas import ChatInput, AIResponse, AIIntent, IntentType, Stage, TokenUsage
from app.services.vector import VectorService

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations with LangChain"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.token_service = TokenUsageService()
        
        # Initialize OpenAI model
        self.llm = ChatOpenAI(
            openai_api_key=settings.openai_api_key,
            model_name=settings.openai_model,
            temperature=settings.openai_temperature,
            max_tokens=settings.openai_max_tokens
        )
        
        # Initialize memory
        self.memory = ConversationBufferWindowMemory(
            k=settings.max_chat_history,
            memory_key="chat_history",
            return_messages=True
        )
        
        # Initialize prompt templates
        self._initialize_prompts()
    
    def _initialize_prompts(self):
        """Initialize prompt templates for different AI roles"""
        
        # System prompt for general AI assistant
        self.system_prompt = """You are Noki AI, an intelligent academic assistant designed to help students with their coursework, assignments, and study planning.

Your capabilities:
- Analyze academic content and provide explanations
- Create study schedules and task lists
- Answer questions about course materials
- Help with project planning and time management
- Provide tutoring and learning support

Guidelines:
- Always provide structured, actionable responses
- Use the provided context (projects, tasks, resources) to give relevant advice
- When you need more information from the backend, emit an intent
- Format responses as structured blocks for the UI
- Be concise but comprehensive
- Focus on academic productivity and learning

Current context:
- User ID: {user_id}
- Conversation ID: {conversation_id}
- Available projects: {projects}
- Available tasks: {tasks}
- Relevant resources: {resources}"""

        # Planner-specific prompt
        self.planner_prompt = """You are Noki AI in planning mode. Your job is to create structured study plans, schedules, and task lists.

When creating plans:
- Use ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SSZ)
- Consider due dates and priorities
- Break down large tasks into manageable sessions
- Suggest realistic time allocations
- Include breaks and buffer time

Output format:
- Create todo_list blocks for task lists
- Use explanation_block for study strategies
- Include confirmation blocks for completed actions
- Always provide accept_decline options for proposed schedules"""

        # Tutor-specific prompt
        self.tutor_prompt = """You are Noki AI in tutoring mode. Your job is to explain concepts, provide learning guidance, and help with academic questions.

When tutoring:
- Explain concepts clearly and step-by-step
- Use examples and analogies
- Reference specific resources when available
- Encourage active learning
- Provide practice suggestions

Output format:
- Use explanation_block for detailed explanations
- Create resource_item blocks for relevant materials
- Include todo_list blocks for practice exercises
- Reference sources with proper citations"""

        # Research-specific prompt
        self.research_prompt = """You are Noki AI in research mode. Your job is to analyze resources, summarize information, and provide research insights.

When researching:
- Synthesize information from multiple sources
- Identify key concepts and themes
- Highlight important details and connections
- Provide balanced perspectives
- Suggest further research directions

Output format:
- Use explanation_block for summaries and analysis
- Create resource_item blocks for source materials
- Include todo_list blocks for research tasks
- Provide clear citations and references"""
    
    def process_chat_request(self, chat_input: ChatInput) -> AIResponse:
        """
        Process a chat request and return AI response
        
        This is the main entry point for chat processing
        """
        try:
            # Step 1: Retrieve semantic context
            semantic_context = self._retrieve_context(chat_input)
            
            # Step 2: Determine if backend data is needed
            intent = self._determine_intent(chat_input, semantic_context)
            
            if intent:
                # Return intent response
                return AIResponse(
                    stage=Stage.INTENT,
                    conversation_id=chat_input.conversation_id,
                    text="Let me gather some information to help you better.",
                    intent=intent
                )
            
            # Step 3: Generate response
            response = self._generate_response(chat_input, semantic_context)
            
            # Step 4: Save message to vector store
            self._save_message(chat_input)
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to process chat request: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text="I apologize, but I encountered an error processing your request. Please try again.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Error occurred. Please try again."
                }]
            )
    
    def continue_with_context(self, conversation_id: str, user_id: str,
                            context_data: Dict[str, Any]) -> AIResponse:
        """
        Continue processing after backend provides context data
        """
        try:
            # Generate response with the provided context
            response = self._generate_response_with_context(
                conversation_id, user_id, context_data
            )
            
            return response
            
        except Exception as e:
            logger.error(f"Failed to continue with context: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=conversation_id,
                text="I apologize, but I encountered an error processing the context data.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Error occurred. Please try again."
                }]
            )
    
    def _retrieve_context(self, chat_input: ChatInput) -> List[Document]:
        """Retrieve semantic context from vector database"""
        try:
            # Get project and task IDs for filtering
            project_ids = [p.project_id for p in (chat_input.projects or [])]
            task_ids = [t.task_id for t in (chat_input.tasks or [])]
            
            # Search for relevant context
            context = self.vector_service.search_semantic_context(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                query=chat_input.prompt,
                project_ids=project_ids,
                task_ids=task_ids
            )
            
            # Add recent chat history
            recent_history = self.vector_service.get_recent_chat_history(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id
            )
            
            return context + recent_history
            
        except Exception as e:
            logger.error(f"Failed to retrieve context: {e}")
            return []
    
    def _determine_intent(self, chat_input: ChatInput, context: List[Document]) -> Optional[AIIntent]:
        """Determine if backend data is needed"""
        try:
            # Simple heuristic: if user asks about assignments, schedule, or specific data
            prompt_lower = chat_input.prompt.lower()
            
            if any(keyword in prompt_lower for keyword in [
                "assignments", "homework", "due dates", "schedule", "calendar",
                "upcoming", "deadlines", "projects", "tasks"
            ]):
                return AIIntent(
                    type=IntentType.BACKEND_QUERY,
                    targets=["assignments", "schedule"],
                    filters={
                        "project_ids": [p.project_id for p in (chat_input.projects or [])],
                        "task_ids": [t.task_id for t in (chat_input.tasks or [])]
                    }
                )
            
            return None
            
        except Exception as e:
            logger.error(f"Failed to determine intent: {e}")
            return None
    
    def _generate_response(self, chat_input: ChatInput, context: List[Document]) -> AIResponse:
        """Generate AI response with structured blocks"""
        try:
            # Format context for prompt
            context_text = self._format_context(context)
            projects_text = self._format_projects(chat_input.projects or [])
            tasks_text = self._format_tasks(chat_input.tasks or [])
            
            # Create system message
            system_message = SystemMessage(content=self.system_prompt.format(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                projects=projects_text,
                tasks=tasks_text,
                resources=context_text
            ))
            
            # Create human message
            human_message = HumanMessage(content=chat_input.prompt)
            
            # Estimate prompt tokens before making the call
            prompt_tokens = self.token_service.estimate_prompt_tokens(
                chat_input.dict(), context
            )
            
            # Get response from LLM
            messages = [system_message, human_message]
            response = self.llm(messages)
            
            # Count completion tokens
            completion_tokens = self.token_service.count_tokens(response.content)
            
            # Create token usage object
            token_usage_data = self.token_service.create_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=settings.openai_model
            )
            token_usage = TokenUsage(**token_usage_data)
            
            # Parse response and create blocks
            blocks = self._parse_response_to_blocks(response.content)
            
            return AIResponse(
                stage=Stage.RESPONSE,
                conversation_id=chat_input.conversation_id,
                text=response.content,
                blocks=blocks,
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text="I apologize, but I couldn't generate a proper response.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Response generation failed."
                }]
            )
    
    def _generate_response_with_context(self, conversation_id: str, user_id: str,
                                      context_data: Dict[str, Any]) -> AIResponse:
        """Generate response with backend-provided context"""
        try:
            # Process the context data to create a meaningful response
            assignments = context_data.get("assignments", [])
            schedule = context_data.get("schedule", {})
            
            # Create a comprehensive response based on the context
            response_text = "Perfect! I've analyzed your assignments and schedule. "
            
            if assignments:
                response_text += f"I found {len(assignments)} assignments that need attention. "
            
            if schedule.get("available_slots"):
                response_text += "I can see your available time slots and will optimize your todo list accordingly. "
            
            response_text += "Let me create a comprehensive todo list that aligns with your academic goals and schedule."
            
            # Estimate tokens for context processing
            context_text = str(context_data)
            prompt_tokens = self.token_service.count_tokens(context_text)
            completion_tokens = self.token_service.count_tokens(response_text)
            
            # Create token usage object
            token_usage_data = self.token_service.create_token_usage(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                model=settings.openai_model
            )
            token_usage = TokenUsage(**token_usage_data)
            
            return AIResponse(
                stage=Stage.RESPONSE,
                conversation_id=conversation_id,
                text=response_text,
                blocks=[{
                    "type": "explanation_block",
                    "title": "Context Analysis Complete",
                    "description": "I've processed your assignments and schedule data.",
                    "blocks": [
                        {
                            "title": "Assignments Found",
                            "description": f"Found {len(assignments)} assignments to work with",
                            "list": [assignment.get("title", "Untitled") for assignment in assignments[:3]]
                        },
                        {
                            "title": "Schedule Analysis",
                            "description": "Analyzed your available time slots",
                            "list": [f"Available slots: {len(schedule.get('available_slots', []))}"]
                        }
                    ]
                }],
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response with context: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=conversation_id,
                text="I encountered an error processing the context data.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Context processing failed."
                }]
            )
    
    def _format_context(self, context: List[Document]) -> str:
        """Format context documents for prompt"""
        if not context:
            return "No relevant context found."
        
        formatted = []
        for doc in context[:5]:  # Limit to top 5
            formatted.append(f"- {doc.page_content[:200]}...")
        
        return "\n".join(formatted)
    
    def _format_projects(self, projects: List) -> str:
        """Format projects for prompt"""
        if not projects:
            return "No projects available."
        
        formatted = []
        for project in projects:
            formatted.append(f"- {project.title}: {project.description or 'No description'}")
        
        return "\n".join(formatted)
    
    def _format_tasks(self, tasks: List) -> str:
        """Format tasks for prompt"""
        if not tasks:
            return "No tasks available."
        
        formatted = []
        for task in tasks:
            status = f" ({task.status})" if task.status else ""
            due = f" - Due: {task.due_datetime}" if task.due_datetime else ""
            formatted.append(f"- {task.title}{status}{due}")
        
        return "\n".join(formatted)
    
    def _parse_response_to_blocks(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse LLM response into structured blocks"""
        # This is a simplified parser - in production, you'd want more sophisticated parsing
        blocks = []
        
        # For now, create a simple explanation block
        blocks.append({
            "type": "explanation_block",
            "title": "AI Response",
            "description": response_text,
            "blocks": []
        })
        
        return blocks
    
    def _save_message(self, chat_input: ChatInput):
        """Save chat message to vector store"""
        try:
            message_id, embedding_tokens = self.vector_service.embed_message(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                message_id=f"msg_{datetime.utcnow().timestamp()}",
                message_content=chat_input.prompt,
                metadata={
                    "stage": chat_input.stage,
                    "projects": [p.project_id for p in (chat_input.projects or [])],
                    "tasks": [t.task_id for t in (chat_input.tasks or [])],
                    "embedding_tokens": embedding_tokens
                }
            )
            logger.info(f"Saved message {message_id} with {embedding_tokens} embedding tokens")
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
