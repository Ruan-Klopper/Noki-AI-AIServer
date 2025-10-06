# Token Usage Tracking - Implementation Complete

## ✅ What's Been Added

### 1. **TokenUsage Model**

```python
class TokenUsage(BaseModel):
    prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0
    embedding_tokens: int = 0
    cost_estimate_usd: float = 0.0
```

### 2. **AIResponse Now Includes Token Usage**

```python
class AIResponse(BaseModel):
    stage: Stage
    conversation_id: str
    text: Optional[str] = None
    blocks: Optional[List[Dict[str, Any]]] = None
    intent: Optional[AIIntent] = None
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    token_usage: Optional[TokenUsage] = None  # NEW!
```

### 3. **TokenUsageService**

- **Accurate token counting** using tiktoken
- **Cost calculation** for GPT-4o, GPT-4, GPT-3.5-turbo
- **Embedding token tracking** for vector operations
- **Context token estimation** including system prompts

### 4. **Updated API Responses**

#### Chat Response Example:

```json
{
  "stage": "response",
  "conversation_id": "conv_045",
  "text": "Here's your study schedule for this week...",
  "blocks": [...],
  "token_usage": {
    "prompt_tokens": 1250,
    "completion_tokens": 450,
    "total_tokens": 1700,
    "embedding_tokens": 0,
    "cost_estimate_usd": 0.0085
  },
  "timestamp": "2025-01-27T10:30:00Z"
}
```

#### Embedding Response Example:

```json
{
  "status": "success",
  "resource_id": "pdf_123",
  "embedding_id": "pdf_123",
  "embedding_tokens": 2500,
  "message": "Resource successfully embedded"
}
```

### 5. **Metrics Integration**

- **Token usage tracking** in metrics store
- **Cost monitoring** for budget management
- **Prometheus metrics** for dashboards
- **Per-user token limits** (configurable)

## 🔧 How It Works

### Token Counting Process:

1. **Prompt Tokens**: System prompt + user input + context + projects/tasks
2. **Completion Tokens**: AI response text
3. **Embedding Tokens**: Resource content + chat messages
4. **Cost Calculation**: Based on current OpenAI pricing

### Cost Tracking:

- **GPT-4o**: $5/1M prompt, $15/1M completion, $0.10/1M embedding
- **GPT-4**: $30/1M prompt, $60/1M completion, $0.10/1M embedding
- **GPT-3.5-turbo**: $1.5/1M prompt, $2/1M completion, $0.10/1M embedding

### Metrics Collection:

```python
# Automatic tracking in all responses
update_metrics(
    stage="response",
    latency=1.2,
    token_usage={
        "prompt_tokens": 1250,
        "completion_tokens": 450,
        "embedding_tokens": 0,
        "cost_estimate_usd": 0.0085
    }
)
```

## 📊 Monitoring & Analytics

### Metrics Endpoint (`/metrics`):

```json
{
  "requests_total": 150,
  "errors_total": 2,
  "avg_latency_ms": 1200.5,
  "stage_distribution": {
    "thinking": 10,
    "intent": 25,
    "response": 100,
    "complete": 15
  },
  "intent_frequency": {
    "backend_query": 20,
    "proposed_schedule": 5,
    "proposed_tasks": 0
  },
  "token_usage": {
    "total_prompt_tokens": 187500,
    "total_completion_tokens": 67500,
    "total_embedding_tokens": 125000,
    "total_cost_usd": 1.25
  }
}
```

### Prometheus Metrics:

```
# Token usage metrics
noki_ai_tokens_total{type="prompt"} 187500
noki_ai_tokens_total{type="completion"} 67500
noki_ai_tokens_total{type="embedding"} 125000

# Cost metrics
noki_ai_cost_total 1.25
```

## 🎯 Benefits

1. **Cost Control**: Track spending per user/conversation
2. **Usage Analytics**: Understand token consumption patterns
3. **Performance Monitoring**: Correlate tokens with response quality
4. **Budget Management**: Set limits and alerts
5. **Optimization**: Identify high-token operations

## 🚀 Usage Examples

### Frontend Integration:

```javascript
// Handle token usage in responses
const response = await fetch("/chat", {
  method: "POST",
  body: JSON.stringify(chatInput),
});

const data = await response.json();

if (data.token_usage) {
  console.log(`Used ${data.token_usage.total_tokens} tokens`);
  console.log(`Cost: $${data.token_usage.cost_estimate_usd}`);

  // Update UI with usage info
  updateTokenCounter(data.token_usage.total_tokens);
  updateCostDisplay(data.token_usage.cost_estimate_usd);
}
```

### Backend Integration:

```python
# Track token usage for billing
def process_chat_response(response: AIResponse):
    if response.token_usage:
        # Log for billing
        log_token_usage(
            user_id=response.conversation_id.split('_')[0],
            tokens=response.token_usage.total_tokens,
            cost=response.token_usage.cost_estimate_usd
        )

        # Check limits
        if exceeds_token_limit(response.token_usage.total_tokens):
            return rate_limit_response()
```

## 🔒 Security & Privacy

- **User-scoped tracking**: Tokens tracked per user/conversation
- **No PII in metrics**: Only aggregate usage data
- **Configurable retention**: Token data can be purged
- **Rate limiting**: Per-user token limits prevent abuse

The token usage tracking is now fully integrated into the Noki AI Engine, providing comprehensive monitoring and cost control for all AI operations!
