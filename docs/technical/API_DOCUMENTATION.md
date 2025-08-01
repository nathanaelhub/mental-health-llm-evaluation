# API Documentation

## Chat Server API

### Base URL
```
http://localhost:8000
```

### Health Check
```http
GET /api/health
```

**Response**:
```json
{
  "status": "healthy",
  "timestamp": "2025-07-31T12:00:00Z",
  "components": {
    "model_selector": true,
    "session_manager": true,
    "server_uptime": 123.45
  },
  "version": "1.0.0-fixed"
}
```

### Chat Endpoint
```http
POST /api/chat
Content-Type: application/json
```

**Request**:
```json
{
  "message": "I'm feeling anxious about work",
  "session_id": "optional-session-id"
}
```

**Response**:
```json
{
  "response": "I understand you're feeling anxious...",
  "selected_model": "openai",
  "session_id": "abc123",
  "confidence_score": 0.75,
  "reasoning": "Selected OPENAI for anxiety scenario...",
  "is_new_session": false,
  "turn_count": 2,
  "conversation_mode": "continuation",
  "model_scores": {
    "openai": 8.5,
    "deepseek": 7.2,
    "claude": 6.8,
    "gemma": 5.9
  },
  "prompt_type": "anxiety"
}
```

### Session Management
```http
POST /api/chat/reset
```

**Request**:
```json
{
  "session_id": "abc123"
}
```

**Response**:
```json
{
  "status": "reset",
  "new_session_id": "def456"
}
```

### Model Analytics
```http
GET /api/analytics
```

**Response**:
```json
{
  "total_selections": 150,
  "model_distribution": {
    "openai": 45,
    "deepseek": 60,
    "claude": 30,
    "gemma": 15
  },
  "avg_confidence_score": 0.68,
  "cache_hit_rate": 0.23
}
```

## Model Client Interfaces

### Base Model Interface
```python
from abc import ABC, abstractmethod
from typing import Optional, Dict, List

class BaseModel(ABC):
    @abstractmethod
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              conversation_history: Optional[List[Dict]] = None,
                              **kwargs) -> ModelResponse:
        pass
```

### OpenAI Client
```python
from src.models.openai_client import OpenAIClient

client = OpenAIClient()
response = await client.generate_response(
    prompt="I feel anxious",
    temperature=0.7,
    max_tokens=2048
)
```

### Local Model Clients
```python
from src.models.deepseek_client import DeepSeekClient
from src.models.gemma_client import GemmaClient

# DeepSeek (Local LLM)
deepseek = DeepSeekClient()
response = await deepseek.generate_response(prompt="I feel sad")

# Gemma (Local LLM)  
gemma = GemmaClient()
response = await gemma.generate_response(prompt="Tell me about therapy")
```

## Evaluation Framework API

### Therapeutic Evaluator
```python
from src.evaluation.evaluation_metrics import TherapeuticEvaluator

evaluator = TherapeuticEvaluator()
result = evaluator.evaluate_response(
    prompt="I'm feeling anxious",
    response="I understand your anxiety...",
    response_time_ms=1500,
    input_tokens=10,
    output_tokens=50
)

print(f"Empathy: {result.empathy_score}/10")
print(f"Safety: {result.safety_score}/10") 
print(f"Therapeutic: {result.therapeutic_value_score}/10")
print(f"Clarity: {result.clarity_score}/10")
print(f"Composite: {result.composite_score}/10")
```

### Research Pipeline
```python
from src.evaluation.mental_health_evaluator import MentalHealthEvaluator

# Create evaluator for multiple models
evaluator = MentalHealthEvaluator(models=['openai', 'deepseek'])

# Run evaluation
results = evaluator.run_evaluation(limit=10)

# Display results
evaluator.display_results()

# Save results
file_paths = evaluator.save_results('results/my_evaluation')
```

## Dynamic Model Selector API

### Initialize Selector
```python
from src.chat.dynamic_model_selector import DynamicModelSelector

models_config = {
    'models': {
        'openai': {'enabled': True},
        'deepseek': {'enabled': True}
    },
    'model_timeouts': {
        'openai': 30.0,
        'deepseek': 120.0
    },
    'selection_timeout': 150.0
}

selector = DynamicModelSelector(models_config)
```

### Select Best Model
```python
selection = await selector.select_best_model(
    prompt="I'm feeling anxious about work",
    context="Previous conversation about stress"
)

print(f"Selected: {selection.selected_model_id}")
print(f"Confidence: {selection.confidence_score:.2f}")
print(f"Reasoning: {selection.selection_reasoning}")
```

## Error Handling

### API Error Responses
```json
{
  "error": {
    "code": "MODEL_TIMEOUT",
    "message": "Model evaluation timed out",
    "details": {
      "timeout_duration": 120,
      "models_attempted": ["openai", "deepseek"]
    }
  }
}
```

### Common Error Codes
- `MODEL_TIMEOUT`: Model response timeout
- `EVALUATION_FAILED`: Scoring system error
- `SESSION_NOT_FOUND`: Invalid session ID
- `INVALID_REQUEST`: Malformed request
- `MODEL_UNAVAILABLE`: Model service offline

## Configuration

### Environment Variables
```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Local Model Server
LOCAL_LLM_SERVER=192.168.86.23:1234

# Model Configuration
DEEPSEEK_MODEL=deepseek/deepseek-r1-0528-qwen3-8b
GEMMA_MODEL=google/gemma-3-12b

# Database
DATABASE_URL=sqlite:///chat_sessions.db

# Demo Mode
DEMO_MODE=true
```

### Model Timeouts
```python
MODEL_TIMEOUTS = {
    'openai': 30.0,      # 30 seconds
    'claude': 30.0,      # 30 seconds
    'deepseek': 120.0,   # 2 minutes (local model)
    'gemma': 120.0       # 2 minutes (local model)
}
```

## Rate Limits

### OpenAI API
- **Requests**: 3,500 requests/minute
- **Tokens**: 90,000 tokens/minute

### Anthropic Claude API  
- **Requests**: 1,000 requests/minute
- **Tokens**: 40,000 tokens/minute

### Local Models
- **Concurrent Requests**: 1 (sequential processing)
- **Memory Requirements**: 24GB+ VRAM recommended

## Monitoring

### Health Monitoring
```bash
# Check system health
curl http://localhost:8000/api/health

# Check model availability
python scripts/development/test_all_four_models.py

# Monitor response times
python scripts/test_local_response_times.py
```

### Performance Metrics
```python
# Get selector analytics
analytics = selector.get_analytics()
print(f"Total selections: {analytics['total_selections']}")
print(f"Cache hit rate: {analytics['cache_hit_rate']:.2%}")
print(f"Avg confidence: {analytics['avg_confidence_score']:.2f}")
```

---

*For system architecture details, see [SYSTEM_ARCHITECTURE.md](SYSTEM_ARCHITECTURE.md)*  
*For research methodology, see [../methodology.md](../methodology.md)*