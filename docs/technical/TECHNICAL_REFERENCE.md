# 🔧 Technical Reference

## API Endpoints

### Core Chat Endpoints

#### POST `/api/chat`
Main chat endpoint for message processing.

**Request:**
```json
{
  "message": "string",
  "session_id": "string | null",
  "user_id": "string",
  "force_reselection": false
}
```

**Response:**
```json
{
  "response": "AI response text",
  "selected_model": "openai | claude | deepseek | gemma",
  "session_id": "string",
  "confidence_score": 0.85,
  "reasoning": "Model selection reasoning",
  "is_new_session": true,
  "turn_count": 1,
  "conversation_mode": "selection | continuation",
  "turn_number": 1
}
```

#### GET `/api/status`
System health check endpoint.

**Response:**
```json
{
  "status": "healthy",
  "version": "1.0.0",
  "available_models": ["openai", "claude", "deepseek", "gemma"],
  "uptime_seconds": 1234.56
}
```

#### GET `/api/models/status`
Detailed model availability status.

**Response:**
```json
{
  "models": {
    "openai": {
      "enabled": true,
      "status": "available",
      "cost_per_token": 0.0001,
      "model_name": "gpt-4",
      "specialties": ["general_support", "crisis", "anxiety", "depression"]
    }
    // ... other models
  },
  "total_available": 4
}
```

#### WebSocket `/api/chat/stream`
Real-time chat streaming endpoint.

**Message Types:**
- `connection`: Initial connection established
- `typing`: AI is processing
- `status`: Progress updates
- `model_selected`: Model chosen with confidence
- `response`: Final AI response
- `error`: Error occurred

## System Architecture

### Component Overview

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│   Chat UI       │────▶│  FastAPI Server  │────▶│ Model Selector  │
│  (Frontend)     │     │   (Backend)      │     │   (Core Logic)  │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                               │                           │
                               ▼                           ▼
                        ┌──────────────┐           ┌──────────────┐
                        │Session Manager│           │ Model Clients│
                        └──────────────┘           └──────────────┘
```

### Key Components

1. **Dynamic Model Selector** (`src/chat/dynamic_model_selector.py`)
   - Intelligent model selection based on prompt type
   - Health checks and fallback logic
   - Model-specific timeouts and retry logic
   - Availability caching (5-minute TTL)

2. **Session Manager** (`src/chat/conversation_session_manager.py`)
   - Conversation state persistence
   - Message history tracking
   - Model selection caching
   - Session cleanup (30-minute timeout)

3. **Model Clients** (`src/models/`)
   - OpenAI API integration
   - Claude/Anthropic integration
   - Local model support (DeepSeek, Gemma)
   - Unified response format

## Configuration

### Environment Variables

```bash
# API Keys
OPENAI_API_KEY=sk-...
ANTHROPIC_API_KEY=sk-ant-...

# Local Model Configuration
DEEPSEEK_BASE_URL=http://192.168.86.23:1234/v1
GEMMA_BASE_URL=http://192.168.86.23:1234/v1

# System Configuration
MODEL_SELECTION_TIMEOUT=15.0
ENABLE_CACHING=true
LOG_LEVEL=INFO
```

### Model Timeouts

```python
MODEL_TIMEOUTS = {
    'openai': 5.0,      # Fast cloud API
    'claude': 5.0,      # Fast cloud API  
    'deepseek': 10.0,   # Local model, may be slower
    'gemma': 10.0       # Local model, may be slower
}
```

## Debugging

### Common Issues

#### 422 Unprocessable Entity
- **Cause**: Field mismatch between frontend and backend
- **Solution**: Ensure `session_id` can be null: `Optional[str] = None`

#### Model Selection Timeout
- **Cause**: Models not responding within timeout
- **Solution**: Check API keys, local model servers, network connectivity

#### WebSocket Connection Failed
- **Cause**: Missing WebSocket dependencies
- **Solution**: `pip install 'uvicorn[standard]' websockets`

### Debug Scripts

```bash
# Test API connectivity
python scripts/test_chat_api.py

# Test model health
python scripts/chat_server_development/test_local_models.py

# Debug chat interface
python scripts/debug_chat_interface.py --comprehensive

# Test specific payload
python scripts/test_current_issue.py
```

### Logging

Enable debug logging:
```bash
export LOG_LEVEL=DEBUG
python chat_server.py
```

View logs:
```bash
tail -f server_debug.log
```

## Model Selection Logic

### Prompt Classification

The system classifies prompts into categories:
- **CRISIS**: Safety-focused selection
- **ANXIETY**: Empathy + therapeutic balance
- **DEPRESSION**: High empathy weighting
- **INFORMATION_SEEKING**: Clarity-focused
- **GENERAL_SUPPORT**: Balanced selection

### Selection Process

1. **Health Check** (2s timeout per model)
2. **Parallel Evaluation** (15s total timeout)
3. **Scoring** based on:
   - Empathy score (0-1)
   - Therapeutic quality (0-1)
   - Safety compliance (0-1)
   - Response clarity (0-1)
4. **Fallback Logic**:
   - No models available → Generic response
   - Single model → Skip selection
   - Timeout → Use first available

### Caching

- **Selection Cache**: 1-hour TTL for identical prompts
- **Availability Cache**: 5-minute TTL for model health
- **Session Cache**: 30-minute inactive timeout

## Performance Optimization

### Response Times

- **First Message**: 15-20s (includes model selection)
- **Follow-up**: 1-3s (single model response)
- **Health Check**: <100ms
- **Status Check**: <50ms

### Concurrency

- **Async/await** throughout the stack
- **Parallel model evaluation**
- **Connection pooling** for API calls
- **WebSocket** for real-time updates

### Resource Usage

- **Memory**: ~200MB base + 50MB per active session
- **CPU**: Minimal (mostly I/O bound)
- **Network**: 1-5KB per message

## Security Considerations

1. **API Key Protection**: Never log or expose API keys
2. **Input Validation**: Pydantic models for all inputs
3. **Rate Limiting**: Consider implementing for production
4. **Session Security**: UUID-based session IDs
5. **CORS**: Configure for production domains only

## Extending the System

### Adding New Models

1. Create client in `src/models/new_model_client.py`
2. Register in `model_config.yaml`
3. Add to `MODEL_TIMEOUTS` configuration
4. Update health check logic if needed

### Custom Prompt Types

1. Add to `PromptType` enum
2. Define `SelectionCriteria` weights
3. Update classification logic
4. Test with sample prompts

### UI Customization

- Templates: `src/ui/templates/`
- Styles: `src/ui/static/css/`
- JavaScript: `src/ui/static/js/`
- Dark theme variables in CSS

---

For more details on specific components, refer to the source code documentation.