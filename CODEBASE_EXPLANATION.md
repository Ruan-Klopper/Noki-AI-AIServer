# Noki AI Engine - Complete Codebase Explanation

## 📋 Table of Contents

1. [System Overview](#system-overview)
2. [Architecture & Data Flow](#architecture--data-flow)
3. [Vector Database (Pinecone)](#vector-database-pinecone)
4. [Context System](#context-system)
5. [Project Management System](#project-management-system)
6. [Core Services](#core-services)
7. [API Endpoints](#api-endpoints)
8. [Data Models](#data-models)
9. [UI Blocks System](#ui-blocks-system)
10. [Token Usage & Cost Tracking](#token-usage--cost-tracking)
11. [Configuration](#configuration)
12. [Request-Response Flow Examples](#request-response-flow-examples)

---

## System Overview

**Noki AI Engine** is an intelligent academic assistant built with:

- **FastAPI**: REST API framework
- **LangChain**: LLM orchestration and prompt management
- **Pinecone**: Vector database for semantic search
- **OpenAI GPT-4o**: Language model for chat and reasoning

### Purpose

The system helps students with:

- Study planning and scheduling
- Assignment management
- Academic tutoring and explanations
- Resource organization (PDFs, websites, videos)
- Personalized learning assistance

---

## Architecture & Data Flow

### High-Level Architecture

```
┌─────────────┐
│   Frontend  │
│  (React/RN) │
└──────┬──────┘
       │
       ↓
┌─────────────┐
│   Backend   │
│  (Django)   │
└──────┬──────┘
       │
       ↓ HTTP REST
┌─────────────────────────────────────┐
│        Noki AI Engine (FastAPI)      │
│                                       │
│  ┌─────────┐  ┌──────────┐          │
│  │   LLM   │←→│  Vector  │          │
│  │ Service │  │  Service │          │
│  └─────────┘  └────┬─────┘          │
│                     ↓                │
│              ┌──────────┐           │
│              │ Planner  │           │
│              │ Service  │           │
│              └──────────┘           │
└─────────────────────────────────────┘
       ↓                    ↓
┌──────────┐         ┌─────────┐
│  OpenAI  │         │Pinecone │
│  API     │         │Vector DB│
└──────────┘         └─────────┘
```

### Request Flow

1. **User sends message** → Frontend → Backend → AI Engine
2. **AI Engine retrieves context** from Pinecone (semantic search)
3. **AI processes request** - Analyzes user input and context
4. **AI generates response** with structured UI blocks
5. **Response sent back** → Backend → Frontend → User sees UI

---

## Vector Database (Pinecone)

### What is the Vector Database?

The vector database stores **semantic embeddings** - mathematical representations of text that capture meaning. Similar concepts are close together in vector space, enabling semantic search.

### How It Works

#### 1. **Initialization** (`vector.py` lines 66-100)

```python
# When the service starts:
- Creates Pinecone client
- Checks if index exists (creates if not)
- Sets up vector store with OpenAI embeddings (text-embedding-ada-002)
- Index uses cosine similarity metric
- Dimension: 1536 (OpenAI's embedding size)
```

#### 2. **Embedding Resources** (`vector.py` lines 246-293)

When a PDF, website, or video is added:

```python
def embed_resource(user_id, conversation_id, resource_id, resource_type, title, content):
    1. Split content into chunks (1000 chars, 100 overlap)
    2. For each chunk:
       - Create metadata (user_id, conversation_id, resource_id, type, chunk_index)
       - Count embedding tokens
       - Create Document object
    3. Generate embeddings via OpenAI
    4. Store in Pinecone with unique IDs (resource_id_chunk_0, resource_id_chunk_1, etc.)
    5. Return total embedding tokens used
```

**Example Metadata Stored:**

```json
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "resource_id": "pdf_abc",
  "resource_type": "PDF",
  "title": "Machine Learning Textbook",
  "chunk_index": 0,
  "type": "resource",
  "created_at": "2025-10-10T10:00:00Z",
  "embedding_tokens": 250
}
```

#### 3. **Embedding Messages** (`vector.py` lines 345-379)

When chat messages are saved:

```python
def embed_message(user_id, conversation_id, message_id, message_content):
    1. Count embedding tokens
    2. Create metadata (user_id, conversation_id, message_id, type="chat")
    3. Generate embedding
    4. Store in Pinecone with message_id
    5. Return embedding tokens
```

#### 4. **Semantic Search** (`vector.py` lines 381-417)

When the AI needs context:

```python
def search_semantic_context(user_id, conversation_id, query, top_k=6):
    1. Build filter: user_id + conversation_id (scoped to user's data)
    2. Optional filters: project_ids, task_ids
    3. Perform similarity search in Pinecone
    4. Return top K most relevant documents
    5. Documents include page_content + metadata
```

**How Similarity Search Works:**

- Query is converted to embedding vector
- Pinecone compares with all stored vectors using cosine similarity
- Returns closest matches (highest similarity scores)
- Only searches within user's conversation scope (privacy)

#### 5. **Performance Optimizations**

**Caching** (`vector.py` lines 120-147):

- In-memory cache for embeddings (1 hour TTL)
- Avoids re-embedding same content
- Uses MD5 hash of content as key

**Async Processing** (`vector.py` lines 159-244):

- Parallel embedding generation
- Batch processing (10 chunks at a time)
- Semaphore limits concurrent operations (5 max)

---

## Context System

### What is Context?

Context is the **relevant information** retrieved from the vector database and conversation history that helps the AI generate better responses.

### Types of Context

#### 1. **Semantic Context** (from vector database)

Retrieved based on meaning similarity:

- Previous conversation messages
- Embedded resources (PDFs, websites)
- Related project/task information
- Study materials and notes

#### 2. **Conversation History** (`vector.py` lines 419-450)

Recent chat messages in the conversation:

```python
def get_recent_chat_history(user_id, conversation_id, limit=5):
    - Filters for type="chat"
    - Returns last N messages
    - Sorted by created_at timestamp
```

#### 3. **Structured Context** (from backend)

Provided by backend after intent fulfillment:

- Assignments data
- Schedule information
- Project details
- Task lists

### How Context is Used

#### Context Retrieval Flow (`llm.py` lines 208-234)

```python
def _retrieve_context(chat_input):
    1. Extract project_ids and task_ids from input
    2. Search vector DB for semantic matches
    3. Get recent chat history
    4. Combine both sources
    5. Return list of Document objects
```

#### Context Formatting (`llm.py` lines 453-509)

Before sending to LLM, context is formatted:

```python
def _format_context(context):
    - Takes top 5 documents
    - For context responses: full content
    - For resources: first 200 characters
    - Returns formatted string
```

**Example Formatted Context:**

```
[Context Data]: User previously asked about assignments. 5 assignments found.
- Chapter 3 reading - Due 2025-10-15...
- Design Methods Essay - Due 2025-10-08...
- [Previous message]: Can you help me with my schedule?
```

#### Context in System Prompt (`llm.py` lines 50-79)

The system prompt includes:

```python
system_prompt = """
You are Noki AI...

Conversation Context:
- User ID: {user_id}
- Conversation ID: {conversation_id}
- Available projects: {projects}
- Available tasks: {tasks}
- Relevant resources: {resources}
- Recent conversation history: {conversation_history}

IMPORTANT: Reference conversation history when relevant...
"""
```

### Context Response Saving (`llm.py` lines 544-586)

When backend provides context data, it's saved to vector DB:

```python
def _save_context_response(conversation_id, user_id, context_data, response):
    1. Create summary of context response
    2. Include assignment/schedule details
    3. Save to vector DB as chat message
    4. Metadata includes: stage="context_response", context_data_keys
    5. This becomes part of future conversation history
```

---

## Project Management System

### What is Project Management?

The Project Management System allows users to organize and manage their projects, tasks, and todos through structured data models and AI assistance.

### Intent Flow

```
User: "What are my upcoming assignments?"
   ↓
AI: Checks vector DB for context
   ↓
AI: Not enough data → Generates INTENT
   ↓
Response: {stage: "intent", intent: {type: "backend_query", targets: ["assignments"]}}
   ↓
Backend: Receives intent → Fetches assignments from database
   ↓
Backend: Calls /chat/context with assignment data
   ↓
AI: Processes context data → Generates structured response
   ↓
Response: {stage: "response", blocks: [todo_list with assignments]}
```

### Intent Types (`schemas.py` lines 18-22)

```python
class IntentType(str, Enum):
    BACKEND_QUERY = "backend_query"          # Request data from backend
    PROPOSED_SCHEDULE = "proposed_schedule"   # AI proposes schedule
    PROPOSED_TASKS = "proposed_tasks"         # AI proposes tasks
```

### Intent Structure (`schemas.py` lines 62-67)

```python
class AIIntent(BaseModel):
    type: IntentType                    # Type of intent
    targets: Optional[List[str]]        # What data to fetch ["assignments", "schedule"]
    filters: Optional[Dict[str, Any]]   # Filtering criteria
    payload: Optional[Dict[str, Any]]   # Additional data
```

### Intent Determination (`llm.py` lines 236-311)

The AI determines if an intent is needed:

```python
def _determine_intent(chat_input, context):
    1. Check what context already exists
       - has_assignments_context?
       - has_schedule_context?

    2. Analyze user's prompt for keywords:
       - Comprehensive: "all my", "show me all", "complete list"
       - Assignments: "homework", "due dates", "deadlines"
       - Schedule: "calendar", "time", "availability"

    3. Determine if fresh data needed:
       - No existing context OR
       - User wants comprehensive data OR
       - User asks for "updated/latest/current"

    4. Create intent with targets and filters

    5. Return intent OR None
```

**Example Intent Decision:**

```python
User prompt: "Show me all my assignments"
Context: Has some old assignment data

Decision: NEED_INTENT
- "all my" = comprehensive keyword
- "assignments" = assignment keyword
- wants_comprehensive = True
→ Generate backend_query intent
```

### Intent Response Processing (`llm.py` lines 174-206)

When backend provides context data:

```python
def continue_with_context(conversation_id, user_id, context_data):
    1. Get recent chat history for context
    2. Generate response with provided context
    3. Save context response to vector DB
    4. Return AIResponse with structured blocks
```

### Intent in Planner Service (`planner.py` lines 193-223)

The planner creates UI blocks based on intents:

```python
def create_intent_response(intent, context_data):
    if intent.type == BACKEND_QUERY:
        if "assignments" in targets:
            → Create todo_list from assignments
        if "schedule" in targets:
            → Create schedule block

    elif intent.type == PROPOSED_SCHEDULE:
        → Create schedule proposal block (with accept/decline)

    elif intent.type == PROPOSED_TASKS:
        → Create task proposal block (with accept/decline)
```

---

## Core Services

### 1. VectorService (`services/vector.py`)

**Responsibilities:**

- Manage Pinecone vector database connection
- Embed resources and messages
- Perform semantic search
- Handle caching and optimization

**Key Methods:**

- `embed_resource()`: Store document chunks with embeddings
- `embed_message()`: Store chat messages with embeddings
- `search_semantic_context()`: Find relevant context
- `get_recent_chat_history()`: Retrieve conversation history
- `delete_user_embeddings()`: Privacy/GDPR compliance

### 2. LLMService (`services/llm.py`)

**Responsibilities:**

- Orchestrate LLM operations
- Manage conversation flow
- Determine intents
- Generate structured responses

**Key Methods:**

- `process_chat_request()`: Main entry point
- `continue_with_context()`: Process after intent fulfillment
- `_retrieve_context()`: Get relevant context
- `_determine_intent()`: Check if backend data needed
- `_generate_response()`: Create AI response
- `_save_message()`: Store to vector DB

**Prompt Templates:**

- `system_prompt`: General AI assistant
- `planner_prompt`: Study planning mode
- `tutor_prompt`: Tutoring mode
- `research_prompt`: Research mode

### 3. PlannerService (`services/planner.py`)

**Responsibilities:**

- Transform LLM output into structured UI blocks
- Create todo lists, schedules, explanations
- Generate proposals with accept/decline

**Key Methods:**

- `create_todo_list_from_assignments()`: Assignment → todo list
- `create_study_schedule_proposal()`: Generate study plan
- `create_explanation_from_resources()`: Resource analysis
- `create_task_breakdown()`: Break tasks into subtasks
- `create_learning_plan()`: Structured learning path
- `create_intent_response()`: Process intent fulfillment

### 4. TokenUsageService (`services/token_usage.py`)

**Responsibilities:**

- Track token consumption
- Calculate API costs
- Provide usage analytics

**Key Methods:**

- `count_tokens()`: Count tokens in text
- `count_embedding_tokens()`: Count embedding tokens
- `calculate_cost()`: Estimate USD cost
- `estimate_prompt_tokens()`: Calculate total prompt size
- `get_model_info()`: Model limits and pricing

**Pricing (2024):**

- GPT-4o: $5/$15 per 1M tokens (prompt/completion)
- GPT-4: $30/$60 per 1M tokens
- Embeddings: $0.10 per 1M tokens

---

## API Endpoints

### Chat Endpoints (`routes/chat.py`)

#### `POST /chat/chat`

Main chat endpoint - processes user messages.

**Request:**

```json
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "prompt": "What are my upcoming assignments?",
  "projects": [
    {
      "project_id": "proj_dm101",
      "title": "Design Methods",
      "description": "UI design course"
    }
  ],
  "tasks": [
    {
      "task_id": "task_essay",
      "title": "Design Methods Essay",
      "due_datetime": "2025-10-08T16:00:00Z",
      "status": "not_started"
    }
  ],
  "stage": "thinking"
}
```

**Response (Intent):**

```json
{
  "stage": "intent",
  "conversation_id": "conv_045",
  "text": "Let me gather some information to help you better.",
  "intent": {
    "type": "backend_query",
    "targets": ["assignments"],
    "filters": {
      "project_ids": ["proj_dm101"]
    }
  },
  "timestamp": "2025-10-10T10:00:00Z"
}
```

**Response (Direct Answer):**

```json
{
  "stage": "response",
  "conversation_id": "conv_045",
  "text": "Here are your upcoming assignments...",
  "blocks": [
    {
      "type": "todo_list",
      "list_title": "Upcoming Assignments",
      "items": [
        {
          "title": "Design Methods Essay",
          "description": "Write 2000 word essay",
          "due_datetime": "2025-10-08T16:00:00Z"
        }
      ]
    }
  ],
  "token_usage": {
    "prompt_tokens": 450,
    "completion_tokens": 120,
    "total_tokens": 570,
    "cost_estimate_usd": 0.003
  }
}
```

#### `POST /chat/context`

Continue processing with backend-provided context.

**Request:**

```json
{
  "conversation_id": "conv_045",
  "user_id": "user_123",
  "context_data": {
    "assignments": [
      {
        "title": "Design Methods Essay",
        "due_date": "2025-10-08",
        "status": "not_started",
        "project_id": "proj_dm101"
      }
    ],
    "schedule": {
      "available_slots": [
        {
          "date": "2025-10-11",
          "time_slots": [{ "start": "09:00", "end": "11:00", "type": "study" }]
        }
      ]
    }
  }
}
```

**Response:**

```json
{
  "stage": "response",
  "conversation_id": "conv_045",
  "text": "Perfect! I've analyzed your assignments and schedule...",
  "blocks": [
    {
      "type": "todo_list",
      "list_title": "Upcoming Assignments",
      "items": [...]
    }
  ]
}
```

#### `POST /chat/stream`

Streaming chat for real-time responses.

**Response Format (SSE):**

```
data: {"stage": "thinking", "conversation_id": "conv_045", "text": "Processing..."}

data: {"stage": "response", "conversation_id": "conv_045", "blocks": [...]}

data: {"stage": "complete", "conversation_id": "conv_045", "text": "Done"}
```

#### `GET /chat/history/{conversation_id}`

Get chat history for a conversation.

**Parameters:**

- `conversation_id`: Conversation ID
- `user_id`: User ID (query param)

**Response:**

```json
{
  "conversation_id": "conv_045",
  "user_id": "user_123",
  "history": [
    {
      "content": "What are my assignments?",
      "metadata": {
        "stage": "thinking",
        "created_at": "2025-10-10T10:00:00Z"
      }
    }
  ],
  "count": 5
}
```

### Embedding Endpoints (`routes/embed.py`)

#### `POST /embed/embed_resource`

Embed resources into vector database.

**Request:**

```json
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "resource_id": "pdf_abc",
  "resource_type": "PDF",
  "title": "Machine Learning Textbook",
  "content": "Chapter 1: Introduction to ML...",
  "metadata": {
    "author": "John Doe",
    "page_count": 450
  }
}
```

**Response:**

```json
{
  "status": "success",
  "resource_id": "pdf_abc",
  "embedding_id": "pdf_abc",
  "embedding_tokens": 12500,
  "message": "Resource successfully embedded"
}
```

#### `POST /embed/embed_message`

Embed chat messages.

**Request:**

```json
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "message_id": "msg_001",
  "message_content": "I need help with my essay",
  "metadata": {
    "role": "user"
  }
}
```

#### `POST /embed/embed_resource_async`

Background embedding for large resources.

#### `DELETE /embed/embed_resource/{resource_id}`

Delete resource embeddings.

#### `DELETE /embed/embed_user/{user_id}`

Delete all user embeddings (GDPR compliance).

#### `GET /embed/embed_stats/{user_id}`

Get embedding statistics.

### Health Endpoints

#### `GET /`

API root with information.

#### `GET /health`

Health check endpoint.

---

## Data Models

### Core Models (`models/schemas.py`)

#### Stage Enum

```python
class Stage(str, Enum):
    THINKING = "thinking"    # AI is processing
    INTENT = "intent"        # AI needs backend data
    RESPONSE = "response"    # AI has answer
    COMPLETE = "complete"    # AI is done
```

#### IntentType Enum

```python
class IntentType(str, Enum):
    BACKEND_QUERY = "backend_query"
    PROPOSED_SCHEDULE = "proposed_schedule"
    PROPOSED_TASKS = "proposed_tasks"
```

#### ChatInput

```python
class ChatInput(BaseModel):
    user_id: str
    conversation_id: str
    prompt: str
    projects: Optional[List[Project]] = None
    tasks: Optional[List[Task]] = None
    stage: Stage = Stage.THINKING
    metadata: Optional[Dict[str, Any]] = None
```

#### AIResponse

```python
class AIResponse(BaseModel):
    stage: Stage
    conversation_id: str
    text: Optional[str] = None
    blocks: Optional[List[Dict[str, Any]]] = None
    intent: Optional[AIIntent] = None
    timestamp: datetime
    token_usage: Optional[TokenUsage] = None
```

#### AIIntent

```python
class AIIntent(BaseModel):
    type: IntentType
    targets: Optional[List[str]] = None
    filters: Optional[Dict[str, Any]] = None
    payload: Optional[Dict[str, Any]] = None
```

#### TokenUsage

```python
class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    cost_estimate_usd: float = 0.0
```

---

## UI Blocks System

### Block Types (`models/ui_blocks.py`)

#### 1. resource_item

```python
{
  "type": "resource_item",
  "item_type": "PDF" | "Website" | "YouTube",
  "title": "Machine Learning Textbook",
  "link": "https://example.com/ml-book.pdf",
  "footer": "Chapter 3 relevant to your question"
}
```

#### 2. todo_list

```python
{
  "type": "todo_list",
  "list_title": "Upcoming Assignments",
  "items": [
    {
      "title": "Design Methods Essay",
      "description": "Write 2000 word essay on UX principles",
      "project_id": "proj_dm101",
      "task_id": "task_essay",
      "due_datetime": "2025-10-08T16:00:00Z"
    }
  ],
  "footer": "Would you like me to create a study schedule?",
  "accept_decline": false
}
```

#### 3. explanation_block

```python
{
  "type": "explanation_block",
  "title": "Study Plan for Next Week",
  "description": "Here's a comprehensive study plan:",
  "blocks": [
    {
      "title": "Monday - Essay Research",
      "description": "Focus on gathering sources",
      "list": [
        "Read chapters 3-5",
        "Find 3 academic papers",
        "Take notes"
      ]
    }
  ],
  "footer": "Need any clarifications?"
}
```

#### 4. confirmation

```python
{
  "type": "confirmation",
  "message": "Study schedule created successfully!"
}
```

### BlockFactory (`models/ui_blocks.py`)

Creates blocks programmatically:

```python
factory = BlockFactory()

# Create todo list
block = factory.create_todo_list(
    list_title="Tasks",
    items=[...],
    footer="Accept?"
)

# Create explanation
block = factory.create_explanation_block(
    title="Analysis",
    blocks=[...],
    description="Here's what I found"
)
```

---

## Token Usage & Cost Tracking

### Token Counting

Uses `tiktoken` library with GPT-4 encoding:

```python
tokenizer = tiktoken.encoding_for_model("gpt-4")
tokens = len(tokenizer.encode(text))
```

### Cost Calculation

```python
def calculate_cost(prompt_tokens, completion_tokens, embedding_tokens, model):
    costs = {
        "gpt-4o": {"prompt": 0.005, "completion": 0.015, "embedding": 0.0001}
    }

    prompt_cost = (prompt_tokens / 1000) * costs["prompt"]
    completion_cost = (completion_tokens / 1000) * costs["completion"]
    embedding_cost = (embedding_tokens / 1000) * costs["embedding"]

    return prompt_cost + completion_cost + embedding_cost
```

### Token Estimation

For full prompts including context:

```python
def estimate_prompt_tokens(chat_input, context):
    system_prompt_tokens = 200  # Base
    user_prompt_tokens = count_tokens(prompt)
    context_tokens = count_context_tokens(context)
    projects_tokens = count_projects_tokens(projects)
    tasks_tokens = count_tasks_tokens(tasks)

    return system + user + context + projects + tasks
```

### Token Usage Response

Every AI response includes:

```json
{
  "token_usage": {
    "prompt_tokens": 450,
    "completion_tokens": 120,
    "total_tokens": 570,
    "embedding_tokens": 0,
    "cost_estimate_usd": 0.003
  }
}
```

---

## Configuration

### Environment Variables (`config.py`)

#### FastAPI Configuration

```python
app_name: str = "Noki AI Engine"
app_version: str = "1.0.0"
debug: bool = True
host: str = "0.0.0.0"
port: int = 8000
```

#### OpenAI Configuration

```python
openai_api_key: str
openai_model: str = "gpt-4o"
openai_temperature: float = 0.7
openai_max_tokens: int = 2000
```

#### Pinecone Configuration

```python
pinecone_api_key: str
pinecone_environment: str
pinecone_index_name: str = "noki-ai-rd41mlf"
pinecone_dimension: int = 1536
```

#### RAG Configuration

```python
retrieval_top_k: int = 6           # Number of context docs to retrieve
max_chat_history: int = 5          # Recent messages to include
chunk_size: int = 1000             # Document chunk size
chunk_overlap: int = 100           # Overlap between chunks
```

#### Embedding Optimization

```python
embedding_batch_size: int = 10     # Batch processing size
embedding_cache_ttl: int = 3600    # Cache duration (1 hour)
max_concurrent_embeddings: int = 5 # Parallel limit
```

#### Security

```python
secret_key: str
algorithm: str = "HS256"
backend_service_token: str         # Backend authentication
bearer_token: str                  # General authentication
```

---

## Request-Response Flow Examples

### Example 1: Simple Question (No Intent)

**Request:**

```json
POST /chat/chat
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "prompt": "What is the difference between supervised and unsupervised learning?",
  "projects": [],
  "tasks": []
}
```

**Internal Flow:**

1. `LLMService.process_chat_request()` called
2. `_retrieve_context()` searches vector DB
   - Finds ML textbook chunks embedded earlier
   - Gets recent conversation history
3. `_determine_intent()` checks if backend data needed
   - No assignment/schedule keywords → No intent
4. `_generate_response()` creates response
   - Formats context (textbook chunks)
   - Sends to GPT-4o with system prompt
   - Receives explanation
5. `_parse_response_to_blocks()` creates explanation block
6. `_save_message()` embeds user question to vector DB
7. Return response

**Response:**

```json
{
  "stage": "response",
  "conversation_id": "conv_045",
  "text": "Supervised learning uses labeled data where the algorithm learns from examples...",
  "blocks": [
    {
      "type": "explanation_block",
      "title": "AI Response",
      "description": "Supervised learning uses labeled data...",
      "blocks": []
    }
  ],
  "token_usage": {
    "prompt_tokens": 320,
    "completion_tokens": 85,
    "total_tokens": 405,
    "cost_estimate_usd": 0.002
  }
}
```

### Example 2: Assignment Query (With Intent)

**Request:**

```json
POST /chat/chat
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "prompt": "Show me all my upcoming assignments",
  "projects": [
    {"project_id": "proj_dm101", "title": "Design Methods"}
  ]
}
```

**Internal Flow:**

1. `LLMService.process_chat_request()` called
2. `_retrieve_context()` searches vector DB
   - Finds some old assignment mentions
3. `_determine_intent()` analyzes prompt
   - Detects "all my" (comprehensive keyword)
   - Detects "assignments" (assignment keyword)
   - wants_comprehensive = True
   - needs_assignments = True
   - Creates BACKEND_QUERY intent
4. `_save_message()` embeds question
5. Return intent response

**Response:**

```json
{
  "stage": "intent",
  "conversation_id": "conv_045",
  "text": "Let me gather some information to help you better.",
  "intent": {
    "type": "backend_query",
    "targets": ["assignments"],
    "filters": {
      "project_ids": ["proj_dm101"]
    }
  }
}
```

**Backend fulfills intent:**

```json
POST /chat/context
{
  "conversation_id": "conv_045",
  "user_id": "user_123",
  "context_data": {
    "assignments": [
      {
        "title": "Design Methods Essay",
        "due_date": "2025-10-08",
        "status": "not_started",
        "project_id": "proj_dm101",
        "description": "Write essay on UX principles"
      },
      {
        "title": "Wireframe Assignment",
        "due_date": "2025-10-15",
        "status": "in_progress",
        "project_id": "proj_dm101"
      }
    ]
  }
}
```

**Internal Flow (Context Processing):**

1. `LLMService.continue_with_context()` called
2. `_generate_response_with_context()` processes
   - Formats assignment data
   - Creates descriptive response
3. `PlannerService.create_intent_response()` called
   - Detects "assignments" in targets
   - Calls `create_todo_list_from_assignments()`
   - Creates structured todo_list block
4. `_save_context_response()` embeds summary
5. Return final response

**Final Response:**

```json
{
  "stage": "response",
  "conversation_id": "conv_045",
  "text": "Perfect! I found 2 assignments that need attention...",
  "blocks": [
    {
      "type": "todo_list",
      "list_title": "Upcoming Assignments",
      "items": [
        {
          "title": "Design Methods Essay",
          "description": "Write essay on UX principles",
          "project_id": "proj_dm101",
          "task_id": "task_essay",
          "due_datetime": "2025-10-08T00:00:00Z"
        },
        {
          "title": "Wireframe Assignment",
          "description": null,
          "project_id": "proj_dm101",
          "task_id": "task_wireframe",
          "due_datetime": "2025-10-15T00:00:00Z"
        }
      ],
      "footer": "Would you like me to create a study schedule for these?"
    }
  ],
  "token_usage": {
    "prompt_tokens": 180,
    "completion_tokens": 65,
    "total_tokens": 245,
    "cost_estimate_usd": 0.001
  }
}
```

### Example 3: Resource Embedding

**Request:**

```json
POST /embed/embed_resource
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "resource_id": "pdf_ml_textbook",
  "resource_type": "PDF",
  "title": "Introduction to Machine Learning",
  "content": "Chapter 1: Introduction\n\nMachine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed. There are three main types of machine learning: supervised learning, unsupervised learning, and reinforcement learning...\n\nChapter 2: Supervised Learning\n\nSupervised learning involves training a model on labeled data...",
  "metadata": {
    "author": "Jane Smith",
    "pages": 350
  }
}
```

**Internal Flow:**

1. `VectorService.embed_resource_async()` called
2. **Chunking:**
   ```
   Chunk 0: "Chapter 1: Introduction\n\nMachine learning is a subset..."
   Chunk 1: "...experience without being explicitly programmed. There are..."
   Chunk 2: "...supervised learning, unsupervised learning..."
   ...
   ```
3. **Batch Processing:**
   - Process 10 chunks at a time
   - Check cache for each chunk (MD5 hash)
   - Generate embeddings for uncached chunks (parallel)
4. **Store in Pinecone:**
   ```python
   IDs: ["pdf_ml_textbook_chunk_0", "pdf_ml_textbook_chunk_1", ...]
   Metadata: {
       "user_id": "user_123",
       "conversation_id": "conv_045",
       "resource_id": "pdf_ml_textbook",
       "resource_type": "PDF",
       "title": "Introduction to Machine Learning",
       "chunk_index": 0,
       "type": "resource",
       "created_at": "2025-10-10T10:00:00Z",
       "embedding_tokens": 256
   }
   ```
5. **Cache embeddings:**
   ```python
   cache["hash_abc123"] = {
       "embedding": [0.123, -0.456, ...],  # 1536 dimensions
       "created_at": "2025-10-10T10:00:00Z"
   }
   ```

**Response:**

```json
{
  "status": "success",
  "resource_id": "pdf_ml_textbook",
  "embedding_id": "pdf_ml_textbook",
  "embedding_tokens": 15420,
  "message": "Resource successfully embedded"
}
```

**Later, when user asks:**

```json
POST /chat/chat
{
  "user_id": "user_123",
  "conversation_id": "conv_045",
  "prompt": "What is supervised learning?"
}
```

**Semantic Search:**

1. Convert query to embedding: [0.234, -0.567, ...]
2. Search Pinecone with filter: `user_id="user_123", conversation_id="conv_045"`
3. Find top 6 matches:
   ```
   - pdf_ml_textbook_chunk_5 (similarity: 0.89)
   - pdf_ml_textbook_chunk_8 (similarity: 0.85)
   - pdf_ml_textbook_chunk_1 (similarity: 0.82)
   ...
   ```
4. Return chunks with content and metadata
5. AI uses these chunks as context to answer

---

## Summary

### Key Concepts

1. **Vector Database (Pinecone)**

   - Stores semantic embeddings of resources and messages
   - Enables meaning-based search (not keyword matching)
   - Scoped to user + conversation for privacy
   - Chunked storage with metadata

2. **Context System**

   - Semantic context from vector DB
   - Recent conversation history
   - Structured backend data
   - Formatted and injected into prompts

3. **Intent System**

   - AI requests backend data when needed
   - Backend fulfills intent and continues
   - Enables separation of concerns
   - Allows backend to control data access

4. **Structured Responses**

   - UI blocks instead of plain text
   - Rendered beautifully in frontend
   - Actionable (accept/decline, links)
   - Type-safe with Pydantic models

5. **Token Tracking**
   - All operations counted
   - Cost estimates provided
   - Optimizations (caching, batching)
   - Transparency for users

### Data Flow Summary

```
User Question
    ↓
Vector DB Search (semantic context)
    ↓
Intent Check (need backend data?)
    ↓
    ├─→ NO: Generate Response
    │          ↓
    │      Structured Blocks
    │          ↓
    │      Return to User
    │
    └─→ YES: Return Intent
               ↓
           Backend Fulfills
               ↓
           Continue with Context
               ↓
           Generate Response
               ↓
           Structured Blocks
               ↓
           Return to User
```

### File Structure

```
noki-ai-engine/
├── config.py              # Configuration and settings
├── app/
│   ├── main.py            # FastAPI app initialization
│   ├── auth.py            # Authentication
│   ├── models/
│   │   ├── schemas.py     # Pydantic data models
│   │   └── ui_blocks.py   # UI block definitions
│   ├── routes/
│   │   ├── chat.py        # Chat endpoints
│   │   ├── embed.py       # Embedding endpoints
│   │   └── health.py      # Health/metrics
│   └── services/
│       ├── vector.py      # Vector DB (Pinecone)
│       ├── llm.py         # LLM orchestration
│       ├── planner.py     # UI block generation
│       └── token_usage.py # Token tracking
└── requirements.txt       # Dependencies
```

---

## Questions & Clarifications

This document explains the complete Noki AI Engine codebase. For specific questions:

1. **Vector DB**: How embeddings work, semantic search, chunking
2. **Context**: How context is retrieved, formatted, and used
3. **Intents**: How AI determines needs, intent fulfillment flow
4. **Services**: LLM, Vector, Planner, Token services
5. **API**: All endpoints and their purposes
6. **Data Models**: Request/response structures
7. **UI Blocks**: Structured response system

The system is designed to be modular, scalable, and maintainable, with clear separation of concerns and strong typing throughout.
