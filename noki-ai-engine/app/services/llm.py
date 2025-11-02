"""
LLM service with LangChain integration for prompt pipeline and RAG
"""
import logging
import json
import re
from typing import List, Dict, Any, Optional
from datetime import datetime

from langchain_community.chat_models import ChatOpenAI
from langchain.schema import Document, HumanMessage, SystemMessage
from langchain.memory import ConversationBufferWindowMemory

from config import settings
from app.services.token_usage import TokenUsageService
from app.models.schemas import ChatInput, AIResponse, Stage, TokenUsage
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
        
        # System prompt - completely rewritten from scratch
        self.system_prompt = """You are Noki AI, a project management assistant that helps users organize their projects, tasks, and todos.

CRITICAL CONTEXT REQUIREMENT:
- You MUST have context to respond. Context comes from:
  1. Vector database search results (semantic context)
  2. Projects, tasks, or todos provided in the request parameters
- If you receive NO context from vector database AND NO projects/tasks/todos in the parameters, you MUST respond asking the user to add context by selecting either a project or a task (depending on the question context).

YOUR CAPABILITIES:
You can perform TWO types of actions, each with its own UI block:

1. EXPLAIN a project or task:
   - Use this ONLY when the user asks to EXPLAIN something
   - ONLY used for explaining PROJECTS or TASKS, nothing else
   - Output format: ExplanationBlock

2. PLAN/CREATE TODO LIST:
   - Use this when the user asks to "plan", "create a todo list", "create a schedule", "organize", etc.
   - Should mainly propose TODOs, not tasks (tasks as last resort only)
   - Output format: Proposed_List

AVAILABLE CONTEXT:
- Projects: {projects}
- Tasks: {tasks}
- Todos: {todos}
- Vector DB Context: {resources}
- Conversation History: {conversation_history}

RESPONSE FORMAT - CRITICAL INSTRUCTIONS:

You MUST respond with a JSON object in this EXACT format:
{{
  "text": "string",  // This is the HEADER text only (e.g., "Okay here is an explanation of your assignment" or "Okay here is a potential todo list for your task")
  "blocks": [  // Array of UI blocks - can contain one or more blocks
    // Block content here
  ],
  "timestamp": "ISO 8601 datetime format"
}}

BLOCK TYPE 1: ExplanationBlock
Use ONLY when user asks to EXPLAIN a project or task.
Structure in the response:
{{
  "text": "string",  // Header ONLY (e.g., "Okay here is an explanation of your assignment")
  "blocks": [
    {{
      "type": "explanation_block",
      "explanation_content": "string"  // The full explanation of the project or task goes HERE, NOT in the "text" field
    }}
  ]
}}

CRITICAL REQUIREMENTS FOR ExplanationBlock:
- The explanation_content MUST ALWAYS be in Markdown format, NOT HTML
- If the context provides HTML content, you MUST convert it to Markdown format
- Use standard Markdown syntax: **bold**, *italic*, - for lists, # for headers, etc.
- Remove all HTML tags like <p>, <strong>, <ul>, <li>, <span>, etc.
- Convert HTML formatting to Markdown equivalents:
  * <p>text</p> → text (with appropriate line breaks)
  * <strong>text</strong> → **text**
  * <i>text</i> or <em>text</em> → *text*
  * <ul><li>item</li></ul> → - item (or numbered list if ordered)
  * <h1>text</h1> → # text
  * <h2>text</h2> → ## text
  * Remove all HTML attributes and tags
- The explanation_content should be clean, readable Markdown text
- NEVER include HTML tags in the explanation_content field

IMPORTANT: The explanation should NOT happen in the top-level "text" field. The "text" field is ONLY the header. The actual explanation goes in the explanation_block's "explanation_content" field within the blocks array.

BLOCK TYPE 2: Proposed_List
Use when user asks to "plan", "create todo list", "schedule", "organize", etc.
Structure:
{{
  "type": "proposed_list",
  "title": "string",  // Title of this todo group
  "proposed_list_for_task_id": "string",  // CRITICAL: MUST use the EXACT task_id or todo_id from the context provided above
  "items": [
    {{
      "title": "string",  // Todo item title
      "due_date": "ISO 8601 format (YYYY-MM-DDTHH:MM:SS.sssZ)",  // CRITICAL: Must look at today's date and schedule accordingly (day/week/month after, etc.)
      "is_all_day": true/false  // Whether this is an all-day todo
    }}
  ]
}}

CRITICAL REQUIREMENTS FOR Proposed_List:
- MUST use the EXACT task_id from the tasks provided in the context (listed above under "Tasks:") OR the EXACT todo_id from todos (listed above under "Todos:")
- DO NOT create or invent IDs - ONLY use the task_ids or todo_ids that are explicitly provided in the context
- If user asks about a task, use that task's task_id
- If user asks about a todo, use that todo's todo_id
- If multiple tasks/todos are provided, use the ID that corresponds to what the user is asking about
- If only one task/todo is provided, use that task's task_id or todo's todo_id
- MUST schedule todos based on TODAY'S DATE ({current_date})
- due_date must be in correct ISO format: "2025-11-02T20:04:26.518Z"
- Prefer creating TODOs over tasks (tasks only as last resort)
- Can create multiple Proposed_List blocks if there are multiple tasks/projects in context (one per task_id or todo_id)

EXAMPLES:

Example 1 - User asks to explain a task:
User: "Explain my assignment"
Response: {{
  "text": "Okay here is an explanation of your assignment",
  "blocks": [
    {{
      "type": "explanation_block",
      "explanation_content": "This assignment requires you to create a comprehensive project that demonstrates your understanding of the key concepts.\n\nYou will need to submit three components:\n- A written report\n- A presentation\n- Source code\n\nThe report should be at least 10 pages and include proper citations."
    }}
  ],
  "timestamp": "2025-11-02T20:04:26.519Z"
}}

Example 1b - If context has HTML:
Context contains HTML: "<p><strong>Comprehensive Analysis</strong></p><p>Present a well-researched paper.</p><ul><li>A 1,500 word paper</li><li>In APA format</li></ul>"
Response should convert to Markdown:
{{
  "text": "Okay here is an explanation of your assignment",
  "blocks": [
    {{
      "type": "explanation_block",
      "explanation_content": "**Comprehensive Analysis**\n\nPresent a well-researched paper.\n\n- A 1,500 word paper\n- In APA format"
    }}
  ],
  "timestamp": "2025-11-02T20:04:26.519Z"
}}

NOTE: Always convert HTML to Markdown - remove all HTML tags and use Markdown syntax instead.

Example 2 - User asks to plan/create todo list:
Context shows: "Task ID: task_abc123 | Title: Complete Project Report | Description: Write comprehensive report"
User: "Create a todo list for my task"
Response: {{
  "text": "Okay here is a potential todo list for your task",
  "blocks": [
    {{
      "type": "proposed_list",
      "title": "Todo List for Complete Project Report",
      "proposed_list_for_task_id": "task_abc123",  // CRITICAL: This must match the exact task_id from context
      "items": [
        {{
          "title": "Research and gather sources",
          "due_date": "2025-11-03T09:00:00.000Z",
          "is_all_day": false
        }},
        {{
          "title": "Write first draft",
          "due_date": "2025-11-04T14:00:00.000Z",
          "is_all_day": true
        }},
        {{
          "title": "Review and edit",
          "due_date": "2025-11-05T10:00:00.000Z",
          "is_all_day": false
        }}
      ]
    }}
  ],
  "timestamp": "2025-11-02T20:04:26.519Z"
}}

NOTE: In the example above, "task_abc123" must be the EXACT task_id from the context. Never invent or create your own task_id.

CRITICAL RULES:
1. ONLY use ExplanationBlock when user asks to EXPLAIN a project or task
2. ONLY use Proposed_List when user asks to plan/create todo list/schedule/organize
3. If no context available, respond with: "Please add context by selecting either a project or a task" (adjust based on question context)
4. The "text" field at the top level is ALWAYS just a header
5. Actual content goes in the "blocks" array
6. For ExplanationBlock, explanation_content MUST ALWAYS be in Markdown format - NEVER HTML, even if context provides HTML
7. For Proposed_List, MUST schedule based on today's date: {current_date}
8. For Proposed_List, the "proposed_list_for_task_id" field MUST use the EXACT task_id or todo_id from the context - NEVER invent or create IDs
9. Always output valid JSON - no markdown code blocks, no extra text, just JSON"""
    
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
    
    def _generate_response(self, chat_input: ChatInput, context: List[Document]) -> AIResponse:
        """Generate AI response with structured blocks"""
        try:
            # Check if we have context
            has_vector_context = len(context) > 0
            has_param_context = (
                (chat_input.projects and len(chat_input.projects) > 0) or
                (chat_input.tasks and len(chat_input.tasks) > 0) or
                (chat_input.todos and len(chat_input.todos) > 0)
            )
            
            # Format context for prompt
            context_text = self._format_context(context)
            projects_text = self._format_projects(chat_input.projects or [])
            tasks_text = self._format_tasks(chat_input.tasks or [])
            todos_text = self._format_todos(chat_input.todos or [])
            conversation_history = self._format_conversation_history(context)
            
            # Get current date for scheduling
            current_date = datetime.utcnow().isoformat() + "Z"
            
            # Create the prompt
            prompt = self.system_prompt.format(
                projects=projects_text,
                tasks=tasks_text,
                todos=todos_text,
                resources=context_text,
                conversation_history=conversation_history,
                current_date=current_date
            )
            
            # If no context, return a prompt asking user to add context
            if not has_vector_context and not has_param_context:
                return AIResponse(
                    stage=Stage.COMPLETE,
                    conversation_id=chat_input.conversation_id,
                    text="Please add context by selecting either a project or a task to continue.",
                    blocks=None,
                    timestamp=datetime.utcnow()
                )
            
            # Generate response with JSON format instruction
            system_message_content = prompt + "\n\nIMPORTANT: You MUST respond with ONLY valid JSON. Do not include any markdown formatting, code blocks, or explanatory text. Just return the raw JSON object."
            user_message_content = chat_input.prompt + "\n\nRemember: Respond with ONLY the JSON object, no other text."
            
            messages = [
                SystemMessage(content=system_message_content),
                HumanMessage(content=user_message_content)
            ]
            
            response = self.llm(messages)
            response_text = response.content.strip()
            
            # Parse JSON response
            parsed_response = self._parse_json_response(response_text)
            
            # Extract text and blocks from parsed response
            response_header = parsed_response.get("text", "")
            blocks = parsed_response.get("blocks", [])
            timestamp_str = parsed_response.get("timestamp")
            
            # Calculate token usage
            token_usage_data = self.token_service.create_token_usage(
                prompt_tokens=len(prompt.split()),
                completion_tokens=len(response_text.split()),
                model=settings.openai_model
            )
            token_usage = TokenUsage(**token_usage_data)
            
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text=response_header,
                blocks=blocks,
                token_usage=token_usage,
                timestamp=datetime.utcnow() if not timestamp_str else datetime.fromisoformat(timestamp_str.replace('Z', '+00:00'))
            )
            
        except Exception as e:
            logger.error(f"Failed to generate response: {e}", exc_info=True)
            return AIResponse(
                stage=Stage.COMPLETE,
                conversation_id=chat_input.conversation_id,
                text="I apologize, but I encountered an error generating a response. Please try again.",
                blocks=None,
                timestamp=datetime.utcnow()
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
            formatted_tasks.append(f"- Task ID: {task.task_id} | Title: {task.title} | Description: {task.description or 'No description'}")
        
        return "\n".join(formatted_tasks)
    
    def _format_todos(self, todos: List) -> str:
        """Format todos for prompt"""
        if not todos:
            return "No todos provided."
        
        formatted_todos = []
        for todo in todos:
            formatted_todos.append(f"- Todo ID: {todo.todo_id} | Title: {todo.title} | Description: {todo.description or 'No description'}")
        
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
    
    def _parse_json_response(self, response_text: str) -> Dict[str, Any]:
        """Parse JSON response from LLM"""
        try:
            # Try to extract JSON from response (in case LLM wraps it in markdown)
            # Remove markdown code blocks if present
            cleaned_text = response_text.strip()
            
            # Remove markdown code blocks (```json ... ```)
            if cleaned_text.startswith("```"):
                # Extract content between first ``` and last ```
                match = re.search(r'```(?:json)?\s*(.*?)\s*```', cleaned_text, re.DOTALL)
                if match:
                    cleaned_text = match.group(1).strip()
            
            # Try to parse as JSON
            parsed = json.loads(cleaned_text)
            
            # Validate structure
            if not isinstance(parsed, dict):
                raise ValueError("Response is not a JSON object")
            
            # Ensure required fields exist
            if "text" not in parsed:
                parsed["text"] = ""
            if "blocks" not in parsed:
                parsed["blocks"] = []
            
            return parsed
            
        except json.JSONDecodeError as e:
            logger.error(f"Failed to parse JSON response: {e}")
            logger.error(f"Response text was: {response_text[:500]}")
            # Return a fallback structure
            return {
                "text": "I apologize, but I had trouble parsing the response. Please try again.",
                "blocks": [],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
        except Exception as e:
            logger.error(f"Unexpected error parsing response: {e}")
            return {
                "text": "I apologize, but I encountered an error. Please try again.",
                "blocks": [],
                "timestamp": datetime.utcnow().isoformat() + "Z"
            }
    
    def _save_message(self, chat_input: ChatInput):
        """Save user message to vector store"""
        try:
            message_id = f"msg_{datetime.utcnow().timestamp()}"
            self.vector_service.embed_message(
                user_id=chat_input.user_id,
                conversation_id=chat_input.conversation_id,
                message_id=message_id,
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
