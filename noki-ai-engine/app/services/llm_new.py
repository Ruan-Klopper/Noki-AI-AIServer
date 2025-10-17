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
from app.models.schemas import ChatInput, AIResponse, Stage, TokenUsage
from app.models.ui_blocks import BlockFactory
from app.services.vector import VectorService

logger = logging.getLogger(__name__)


class LLMService:
    """Service for LLM operations with LangChain"""
    
    def __init__(self, vector_service: VectorService):
        self.vector_service = vector_service
        self.token_service = TokenUsageService()
        
        # Initialize OpenAI model
        if not settings.openai_api_key:
            raise ValueError(
                "OpenAI API key is not configured. "
                "Please set OPENAI_API_KEY in your .env file or environment variables."
            )
        
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
        self.system_prompt = """You are Noki AI, an intelligent project management assistant designed to help users organize their projects, tasks, and todos.

Your capabilities:
- Analyze and organize project information
- Create and manage task lists and todo items
- Help with project planning and time management
- Provide insights and recommendations for productivity
- Assist with project tracking and progress monitoring

Guidelines:
- Always provide structured, actionable responses
- Use the provided context (projects, tasks, todos) to give relevant advice
- Format responses as structured blocks for the UI
- Be concise but comprehensive
- Focus on productivity and organization
- ALWAYS reference conversation history when relevant
- Help users create, update, and manage their projects, tasks, and todos

Conversation Context:
- User ID: {user_id}
- Conversation ID: {conversation_id}
- Available projects: {projects}
- Available tasks: {tasks}
- Available todos: {todos}
- Relevant resources: {resources}
- Recent conversation history: {conversation_history}

IMPORTANT: When referencing conversation history, acknowledge what was discussed earlier and build upon it. Help users manage their projects, tasks, and todos effectively."""

        # Planner-specific prompt
        self.planner_prompt = """You are Noki AI in planning mode. Your job is to create structured project plans, task lists, and todo management systems.

When creating plans:
- Use ISO 8601 datetime format (YYYY-MM-DDTHH:MM:SSZ)
- Consider due dates and priorities
- Break down large projects into manageable tasks
- Break down tasks into actionable todos
- Suggest realistic time allocations
- Include buffer time for unexpected delays

Output format:
- Use todo_list blocks for task and todo management
- Create explanation_block for project insights
- Include confirmation blocks for user actions
- Provide clear, actionable next steps"""

        # Tutor-specific prompt
        self.tutor_prompt = """You are Noki AI in analysis mode. Your job is to analyze project information, provide insights, and help with productivity questions.

When analyzing:
- Break down complex projects into understandable components
- Identify patterns and dependencies in tasks and todos
- Provide actionable recommendations
- Explain project management concepts clearly
- Suggest improvements and optimizations

Output format:
- Use explanation_block for detailed analysis
- Create todo_list blocks for actionable recommendations
- Include confirmation blocks for user decisions
- Provide clear, structured insights"""

        # Research-specific prompt
        self.research_prompt = """You are Noki AI in research mode. Your job is to analyze project resources, summarize information, and provide project insights.

When researching:
- Synthesize information from multiple project sources
- Identify key concepts and themes in project documentation
- Highlight important details and connections between tasks
- Provide balanced perspectives on project approaches
- Suggest further research directions for project success

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
            
            # Step 2: Generate response directly
            response = self._generate_response(chat_input, semantic_context)
            
            # Step 3: Save message to vector store
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
    
    def _retrieve_context(self, chat_input: ChatInput) -> List[Document]:
        """Retrieve semantic context from vector database"""
        try:
            # Get project, task, and todo IDs for filtering
            project_ids = [p.project_id for p in (chat_input.projects or [])]
            task_ids = [t.task_id for t in (chat_input.tasks or [])]
            todo_ids = [t.todo_id for t in (chat_input.todos or [])]
            
            # Search for relevant context
            context = self.vector_service.search_semantic_context(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                query=chat_input.prompt,
                project_ids=project_ids,
                task_ids=task_ids,
                todo_ids=todo_ids
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
    
    def _generate_response(self, chat_input: ChatInput, context: List[Document]) -> AIResponse:
        """Generate AI response with structured blocks"""
        try:
            # Format context for prompt
            context_text = self._format_context(context)
            projects_text = self._format_projects(chat_input.projects or [])
            tasks_text = self._format_tasks(chat_input.tasks or [])
            todos_text = self._format_todos(chat_input.todos or [])
            conversation_history = self._format_conversation_history(context)
            
            # Create the prompt
            prompt = self.system_prompt.format(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                projects=projects_text,
                tasks=tasks_text,
                todos=todos_text,
                resources=context_text,
                conversation_history=conversation_history
            )
            
            # Generate response
            messages = [
                SystemMessage(content=prompt),
                HumanMessage(content=chat_input.prompt)
            ]
            
            response = self.llm(messages)
            response_text = response.content
            
            # Parse response into structured blocks
            blocks = self._parse_response_to_blocks(response_text)
            
            # Calculate token usage
            token_usage = self.token_service.calculate_usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(response_text.split())
            )
            
            return AIResponse(
                stage=Stage.RESPONSE,
                conversation_id=chat_input.conversation_id,
                text=response_text,
                blocks=blocks,
                token_usage=token_usage
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}")
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text="I apologize, but I encountered an error generating a response. Please try again.",
                blocks=[{
                    "type": "confirmation",
                    "message": "Error occurred. Please try again."
                }]
            )
    
    def _format_context(self, context: List[Document]) -> str:
        """Format context documents for prompt"""
        if not context:
            return "No relevant context found."
        
        formatted_context = []
        for doc in context:
            formatted_context.append(f"- {doc.page_content}")
        
        return "\n".join(formatted_context)
    
    def _format_projects(self, projects: List) -> str:
        """Format projects for prompt"""
        if not projects:
            return "No projects provided."
        
        formatted_projects = []
        for project in projects:
            formatted_projects.append(f"- {project.title}: {project.description or 'No description'}")
        
        return "\n".join(formatted_projects)
    
    def _format_tasks(self, tasks: List) -> str:
        """Format tasks for prompt"""
        if not tasks:
            return "No tasks provided."
        
        formatted_tasks = []
        for task in tasks:
            formatted_tasks.append(f"- {task.title}: {task.description or 'No description'}")
        
        return "\n".join(formatted_tasks)
    
    def _format_todos(self, todos: List) -> str:
        """Format todos for prompt"""
        if not todos:
            return "No todos provided."
        
        formatted_todos = []
        for todo in todos:
            formatted_todos.append(f"- {todo.title}: {todo.description or 'No description'}")
        
        return "\n".join(formatted_todos)
    
    def _format_conversation_history(self, context: List[Document]) -> str:
        """Format conversation history for prompt"""
        history_docs = [doc for doc in context if doc.metadata.get("type") == "chat_message"]
        
        if not history_docs:
            return "No previous conversation history."
        
        formatted_history = []
        for doc in history_docs[-5:]:  # Last 5 messages
            formatted_history.append(f"- {doc.page_content}")
        
        return "\n".join(formatted_history)
    
    def _parse_response_to_blocks(self, response_text: str) -> List[Dict[str, Any]]:
        """Parse AI response into structured blocks"""
        try:
            # Simple parsing - in a real implementation, you'd want more sophisticated parsing
            # For now, return a basic explanation block
            return [{
                "type": "explanation_block",
                "title": "AI Response",
                "description": response_text,
                "footer": "How can I help you further with your projects, tasks, or todos?"
            }]
        except Exception as e:
            logger.error(f"Failed to parse response to blocks: {e}")
            return [{
                "type": "confirmation",
                "message": "Response generated successfully."
            }]
    
    def _save_message(self, chat_input: ChatInput):
        """Save user message to vector store"""
        try:
            self.vector_service.save_chat_message(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                message_content=chat_input.prompt,
                metadata={
                    "type": "chat_message",
                    "stage": chat_input.stage.value,
                    "projects": [p.project_id for p in (chat_input.projects or [])],
                    "tasks": [t.task_id for t in (chat_input.tasks or [])],
                    "todos": [t.todo_id for t in (chat_input.todos or [])],
                    "created_at": datetime.utcnow().isoformat()
                }
            )
        except Exception as e:
            logger.error(f"Failed to save message: {e}")
