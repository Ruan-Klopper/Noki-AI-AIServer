# Noki AI Engine

<div align="center">

**An intelligent academic assistant built with FastAPI, LangChain, and Pinecone for semantic search and AI-powered tutoring.**
</br>
</br>
**_By Ruan Klopper_**
</br>
**_Student no: 231280_**
</br>
</br>
[![Live Demo](https://img.shields.io/badge/Live%20Demo-app.noki.co.za-blue)](https://app.noki.co.za)
[![GitHub](https://img.shields.io/badge/GitHub-Repository-black?logo=github)](https://github.com/Ruan-Klopper/Noki-AI-frontend)
[![License](https://img.shields.io/badge/License-Private-red)](LICENSE)

</div>

## Tech Stack

**Core Framework**
<br>
![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-0.117-009688?style=for-the-badge&logo=fastapi&logoColor=white)
![Uvicorn](https://img.shields.io/badge/Uvicorn-0.37-499848?style=for-the-badge&logo=uvicorn&logoColor=white)

**AI & LLM**
<br>
![LangChain](https://img.shields.io/badge/LangChain-0.3-1C3C3C?style=for-the-badge&logo=langchain&logoColor=white)
![OpenAI](https://img.shields.io/badge/OpenAI-GPT--4o-412991?style=for-the-badge&logo=openai&logoColor=white)
![TikToken](https://img.shields.io/badge/TikToken-0.8-FF6B6B?style=for-the-badge)

**Vector Database**
<br>
![Pinecone](https://img.shields.io/badge/Pinecone-Vector_DB-430098?style=for-the-badge&logo=pinecone&logoColor=white)
![LangChain Pinecone](https://img.shields.io/badge/LangChain_Pinecone-0.2-1C3C3C?style=for-the-badge)

**Data Validation & Serialization**
<br>
![Pydantic](https://img.shields.io/badge/Pydantic-2.11-E92063?style=for-the-badge&logo=pydantic&logoColor=white)
![Pydantic Settings](https://img.shields.io/badge/Pydantic_Settings-2.11-E92063?style=for-the-badge)

**HTTP & Networking**
<br>
![HTTPX](https://img.shields.io/badge/HTTPX-0.28-98330A?style=for-the-badge)
![Requests](https://img.shields.io/badge/Requests-2.32-000000?style=for-the-badge&logo=python&logoColor=white)

**Security & Authentication**
<br>
![Python-JOSE](https://img.shields.io/badge/Python--JOSE-3.3-FF6B6B?style=for-the-badge)
![Passlib](https://img.shields.io/badge/Passlib-1.7-FFD700?style=for-the-badge)

**Database & ORM**
<br>
![SQLAlchemy](https://img.shields.io/badge/SQLAlchemy-2.0-D71F00?style=for-the-badge&logo=sqlalchemy&logoColor=white)
![Alembic](https://img.shields.io/badge/Alembic-1.14-00A9CE?style=for-the-badge)

**Monitoring & Logging**
<br>
![Prometheus](https://img.shields.io/badge/Prometheus-Client-0.21-E6522C?style=for-the-badge&logo=prometheus&logoColor=white)
![Structlog](https://img.shields.io/badge/Structlog-24.4-000000?style=for-the-badge)

**Development Tools**
<br>
![Pytest](https://img.shields.io/badge/Pytest-8.3-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white)
![Black](https://img.shields.io/badge/Black-24.10-000000?style=for-the-badge&logo=black&logoColor=white)
![isort](https://img.shields.io/badge/isort-5.13-EF8336?style=for-the-badge)
![Flake8](https://img.shields.io/badge/Flake8-7.1-000000?style=for-the-badge&logo=flake8&logoColor=white)

**Deployment**
<br>
![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)
![Railway](https://img.shields.io/badge/Railway-Deployed-0B0D0E?style=for-the-badge&logo=railway&logoColor=white)

## Architecture

This AI server implements the complete Noki AI Engine specification with:

- **FastAPI** REST API with OpenAPI/Swagger documentation
- **LangChain** for LLM orchestration, prompt pipelines, and RAG
- **Pinecone** vector database for semantic embeddings
- **OpenAI GPT-4o** for chat, tutoring, and planning
- **Structured UI responses** with cards/blocks instead of wall-of-text

## 🎯 Key Features

### State Machine Flow

- `thinking` → AI is processing (frontend shows loader)
- `intent` → AI requests backend data (targets + filters)
- `response` → AI returns structured answer with UI blocks
- `complete` → AI signals done (after user accept/decline + backend persist)

### AI Capabilities

- **Planner**: Create study schedules and task lists with ISO datetimes
- **Tutor**: Explain concepts with citations and resource references
- **Research**: Summarize retrieved chunks into structured blocks
- **Memory**: Semantic recall using vector embeddings per user + conversation

### Structured UI Blocks

- `resource_item`: PDF, Website, YouTube resources
- `todo_list`: Task lists with accept/decline options
- `explanation_block`: Multi-section explanations
- `confirmation`: Success/error messages

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd noki-ai-engine
pip install -r requirements.txt
```

### 2. Environment Setup

Create a `.env` file with:

```env
# OpenAI Configuration
OPENAI_API_KEY="your-openai-api-key"
OPENAI_MODEL="gpt-4o"

# Pinecone Configuration
PINECONE_API_KEY="your-pinecone-api-key"
PINECONE_ENVIRONMENT="your-pinecone-environment"
PINECONE_INDEX_NAME="noki-ai-embeddings"

# Security
SECRET_KEY="your-secret-key"
BACKEND_SERVICE_TOKEN="your-backend-service-token"
```

### 3. Run the Server

```bash
python -m app.main
```

The API will be available at `http://localhost:8000` with Swagger docs at `/docs`.

## 📡 API Endpoints

### Chat Endpoints

#### `POST /chat`

Main chat entrypoint - processes user messages and returns structured AI responses.

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
      "description": "User interface design course"
    }
  ],
  "tasks": [
    {
      "task_id": "task_essay",
      "title": "Design Methods Essay",
      "due_datetime": "2025-10-08T16:00:00Z",
      "status": "not_started"
    }
  ]
}
```

**Response:**

```json
{
  "stage": "intent",
  "conversation_id": "conv_045",
  "text": "Let me fetch your upcoming assignments first.",
  "intent": {
    "type": "backend_query",
    "targets": ["assignments"],
    "filters": { "project_ids": ["proj_dm101"] }
  }
}
```

#### `POST /chat/context`

Continue processing after backend provides context data.

#### `POST /chat/stream`

Streaming chat for real-time responses.

### Embedding Endpoints

#### `POST /embed_resource`

Embed resources (PDF, website, YouTube) into vector database.

#### `POST /embed_message`

Embed chat messages for semantic recall.

### Health & Metrics

#### `GET /health`

Health check endpoint.

#### `GET /metrics`

Performance metrics for monitoring.

## 🧩 Data Models

### ChatInput

- `user_id`: string (required)
- `conversation_id`: string (required)
- `prompt`: string (required)
- `projects`: Project[] (optional)
- `tasks`: Task[] (optional)
- `stage`: "thinking" (default)
- `metadata`: object (optional)

### AIResponse

- `stage`: "thinking" | "intent" | "response" | "complete"
- `conversation_id`: string
- `text`: string (optional)
- `blocks`: Block[] (optional)
- `intent`: AIIntent (optional)
- `timestamp`: ISO8601

### AIIntent

- `type`: "backend_query" | "proposed_schedule" | "proposed_tasks"
- `targets`: string[] (optional)
- `filters`: object (optional)
- `payload`: object (optional)

## 🔧 Services

### VectorService

- Manages Pinecone vector database
- Handles embeddings for resources and messages
- Performs semantic search with user/conversation scoping
- Implements chunking and retrieval policies

### LLMService

- LangChain integration with OpenAI GPT-4o
- Prompt templates for Planner/Tutor/Research modes
- RAG pipeline with semantic context retrieval
- Memory management with conversation history

### PlannerService

- Transforms LLM output into structured UI blocks
- Creates todo lists, explanations, and confirmations
- Handles schedule proposals and task breakdowns
- Generates learning plans and progress summaries

## 🔒 Security

- **Authentication**: Backend service token verification
- **Authorization**: User ID + conversation ID scoping
- **Rate Limiting**: Per user and per conversation limits
- **PII Minimization**: Only necessary fields in AI prompts
- **Data Retention**: Configurable embedding retention policies

## 📊 Monitoring

- **Metrics**: Request counts, error rates, latency
- **Health Checks**: Service connectivity and configuration
- **Prometheus**: Metrics export for dashboards
- **Logging**: Structured logging with correlation IDs

## 🧪 Testing

```bash
# Run tests
pytest

# Test specific endpoint
curl -X POST "http://localhost:8000/chat" \
  -H "Content-Type: application/json" \
  -d '{"user_id": "test_user", "conversation_id": "test_conv", "prompt": "Hello"}'
```

## 🚀 Deployment

### Docker

```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY . .
CMD ["python", "-m", "app.main"]
```

### Environment Variables

- `OPENAI_API_KEY`: OpenAI API key
- `PINECONE_API_KEY`: Pinecone API key
- `PINECONE_ENVIRONMENT`: Pinecone environment
- `BACKEND_SERVICE_TOKEN`: Service authentication token

## 📚 Integration

### Backend Integration

1. Backend calls `/chat` with user context
2. AI responds with `intent` if backend data needed
3. Backend fulfills intent and calls `/chat/context`
4. AI generates final response with structured blocks
5. Frontend renders UI blocks appropriately

### Frontend Integration

- Handle different response stages (`thinking`, `intent`, `response`, `complete`)
- Render UI blocks based on type (`todo_list`, `explanation_block`, etc.)
- Show accept/decline options for proposals
- Implement streaming for real-time responses

## 🔄 Development

### Project Structure

```
noki-ai-engine/
├── app/
│   ├── main.py              # FastAPI app and middleware
│   ├── routes/
│   │   ├── chat.py          # Chat endpoints
│   │   ├── embed.py         # Embedding endpoints
│   │   └── health.py        # Health and metrics
│   ├── services/
│   │   ├── llm.py           # LLM service with LangChain
│   │   ├── vector.py        # Vector database service
│   │   └── planner.py       # UI block generation
│   └── models/
│       ├── schemas.py       # Pydantic models
│       └── ui_blocks.py     # UI block definitions
├── config.py                # Configuration settings
└── requirements.txt         # Dependencies
```

### Adding New Features

1. Define data models in `models/schemas.py`
2. Implement business logic in `services/`
3. Create API endpoints in `routes/`
4. Update configuration in `config.py`
5. Add tests and documentation

## 📝 License

Private - Ruan Klopper
This project is private and proprietary. All rights reserved.
