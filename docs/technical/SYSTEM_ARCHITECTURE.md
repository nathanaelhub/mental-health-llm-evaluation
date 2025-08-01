# System Architecture

## Overview

The Mental Health LLM Evaluation System implements a modular, microservice-inspired architecture with intelligent model routing, session management, and comprehensive evaluation frameworks.

## High-Level Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Client Interface                         │
│  ┌─────────────────┐  ┌─────────────────┐  ┌──────────────┐ │
│  │   Web UI        │  │   REST API      │  │  CLI Tools   │ │
│  │ (Chat Interface)│  │ (Programmatic)  │  │ (Research)   │ │
│  └─────────────────┘  └─────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                    Core Services                            │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │            Dynamic Model Selector                      │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │ │
│  │  │   Prompt    │ │  Parallel   │ │   Selection Logic   │ │ │
│  │  │Classifier   │ │ Evaluator   │ │  & Confidence      │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │          Session Management Service                     │ │
│  │  ┌─────────────┐ ┌─────────────┐ ┌─────────────────────┐ │ │
│  │  │  Session    │ │Conversation │ │    SQLite Store     │ │ │
│  │  │  Manager    │ │  History    │ │   (Persistence)     │ │ │
│  │  └─────────────┘ └─────────────┘ └─────────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                   Model Abstraction Layer                  │
│  ┌─────────────┐ ┌─────────────┐ ┌─────────────┐ ┌────────┐ │
│  │   OpenAI    │ │   Claude    │ │  DeepSeek   │ │ Gemma  │ │
│  │   Client    │ │   Client    │ │   Client    │ │ Client │ │
│  │  (Cloud)    │ │  (Cloud)    │ │  (Local)    │ │(Local) │ │
│  └─────────────┘ └─────────────┘ └─────────────┘ └────────┘ │
└─────────────────────────────────────────────────────────────┘
                              │
┌─────────────────────────────────────────────────────────────┐
│                 Evaluation Framework                        │
│  ┌─────────────────────────────────────────────────────────┐ │
│  │           Therapeutic Evaluator                         │ │
│  │  ┌─────────┐ ┌──────────┐ ┌─────────┐ ┌────────────────┐ │ │
│  │  │Empathy  │ │ Safety   │ │Therapeutic│ │    Clarity     │ │ │
│  │  │Scoring  │ │Detection │ │  Value    │ │   Assessment   │ │ │
│  │  │ (30%)   │ │  (35%)   │ │  (25%)    │ │     (10%)      │ │ │
│  │  └─────────┘ └──────────┘ └─────────┘ └────────────────┘ │ │
│  └─────────────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## Core Components

### 1. Dynamic Model Selector (`src/chat/dynamic_model_selector.py`)

**Responsibilities**:
- Parallel model evaluation
- Context-aware selection criteria
- Confidence scoring
- Performance monitoring

**Key Classes**:
```python
class DynamicModelSelector:
    - select_best_model() -> ModelSelection
    - parallel_evaluate() -> List[ModelEvaluation]  
    - apply_selection_logic() -> ModelSelection
    - prompt_classification() -> PromptType

class ModelSelection:
    - selected_model_id: str
    - confidence_score: float
    - selection_reasoning: str
    - model_scores: Dict[str, float]
```

**Selection Algorithm**:
1. **Prompt Classification**: Categorize input (anxiety, depression, crisis, etc.)
2. **Parallel Evaluation**: Async evaluation of all available models
3. **Weighted Scoring**: Apply context-specific weights to evaluation dimensions
4. **Confidence Calculation**: Statistical confidence in selection decision
5. **Result Caching**: Cache selections for similar prompts

### 2. Session Management (`src/chat/conversation_session_manager.py`)

**Responsibilities**:
- Session lifecycle management
- Conversation history persistence
- Model selection caching per session
- SQLite database operations

**Key Classes**:
```python
class ConversationSessionManager:
    - create_session() -> str
    - get_session() -> ConversationSession
    - update_session() -> None
    - reset_session() -> str

class ConversationSession:
    - session_id: str
    - selected_model: str
    - conversation_history: List[Dict]
    - created_at: datetime
    - last_updated: datetime
```

**Database Schema**:
```sql
CREATE TABLE sessions (
    session_id TEXT PRIMARY KEY,
    selected_model TEXT,
    conversation_history TEXT, -- JSON
    created_at TIMESTAMP,
    last_updated TIMESTAMP
);
```

### 3. Model Abstraction Layer (`src/models/`)

**Base Interface**:
```python
class BaseModel(ABC):
    @abstractmethod
    async def generate_response(self, 
                              prompt: str,
                              system_prompt: Optional[str] = None,
                              conversation_history: Optional[List[Dict]] = None,
                              **kwargs) -> ModelResponse
```

**Model Implementations**:

#### Cloud Models
- **OpenAI Client** (`openai_client.py`): GPT-4 via OpenAI API
- **Claude Client** (`claude_client.py`): Claude-3 via Anthropic API

#### Local Models  
- **DeepSeek Client** (`deepseek_client.py`): Local LLM via LM Studio
- **Gemma Client** (`gemma_client.py`): Local LLM via LM Studio

**Configuration**:
```python
MODEL_CONFIG = {
    'openai': {
        'api_key': os.getenv('OPENAI_API_KEY'),
        'model': 'gpt-4',
        'timeout': 30.0
    },
    'deepseek': {
        'base_url': 'http://192.168.86.23:1234/v1',
        'model': 'deepseek/deepseek-r1-0528-qwen3-8b',
        'timeout': 120.0
    }
}
```

### 4. Evaluation Framework (`src/evaluation/`)

**Therapeutic Evaluator** (`evaluation_metrics.py`):
```python
class TherapeuticEvaluator:
    def evaluate_response(self, prompt: str, response: str) -> EvaluationResult:
        # Multi-dimensional scoring
        empathy_score = self.evaluate_empathy(response)
        safety_score = self.evaluate_safety(response)  
        therapeutic_score = self.evaluate_therapeutic_value(response)
        clarity_score = self.evaluate_clarity(response)
        
        # Weighted composite score
        composite = self.calculate_composite_score(
            empathy_score, safety_score, therapeutic_score, clarity_score
        )
        
        return EvaluationResult(...)
```

**Evaluation Dimensions**:

1. **Empathy (30% weight)**:
   - Emotional validation patterns
   - Feeling acknowledgment
   - Supportive language detection

2. **Safety (35% weight - Highest Priority)**:
   - Crisis content detection
   - Harmful advice prevention
   - Professional boundary maintenance

3. **Therapeutic Value (25% weight)**:
   - Coping strategy recommendations
   - Professional referral suggestions
   - Evidence-based guidance

4. **Clarity (10% weight)**:
   - Response length optimization
   - Readability assessment
   - Structure and organization

### 5. Web Interface (`templates/`, `static/`)

**FastAPI Server** (`chat_server.py`):
```python
@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    # Get or create session
    session = await session_manager.get_session(request.session_id)
    
    if session.turn_count == 0:
        # First message: Dynamic model selection
        selection = await model_selector.select_best_model(request.message)
        session.selected_model = selection.selected_model_id
    else:
        # Continuation: Use stored model
        selection = await use_stored_model(session, request.message)
    
    # Update session and return response
    await session_manager.update_session(session)
    return ChatResponse(...)
```

**UI Components**:
- **Chat Interface**: Real-time conversation with loading states
- **Model Display**: Shows selected model and confidence
- **Conversation History**: Persistent chat bubbles
- **Reset Functionality**: New conversation management

## Data Flow

### Initial Message Flow
```
User Input → Prompt Classification → Parallel Model Evaluation → 
Scoring & Selection → Response Generation → Session Storage → UI Update
```

### Continuation Message Flow  
```
User Input → Session Retrieval → Stored Model Usage → 
Response Generation → Session Update → UI Update
```

### Evaluation Research Flow
```
Scenario Loading → Multi-Model Generation → Parallel Evaluation → 
Statistical Analysis → Visualization Generation → Results Storage
```

## Performance Optimizations

### 1. Asynchronous Processing
- **Parallel Model Evaluation**: All models evaluated simultaneously
- **Non-blocking I/O**: FastAPI async endpoints
- **Connection Pooling**: Reuse HTTP connections

### 2. Caching Strategy
- **Response Cache**: Similar prompts use cached evaluations
- **Session Storage**: Avoid re-evaluation for continuations
- **Model Availability Cache**: Reduce health check overhead

### 3. Timeout Management
- **Model-Specific Timeouts**: Different limits for cloud vs local
- **Graceful Degradation**: Fallback models for failures
- **Demo Mode**: Extended timeouts for presentations

### 4. Database Optimization
- **SQLite Performance**: Indexed sessions table
- **JSON Storage**: Efficient conversation history storage
- **Connection Management**: Pool connections for concurrency

## Security Considerations

### 1. API Security
- **Input Validation**: Sanitize all user inputs
- **Rate Limiting**: Prevent API abuse
- **CORS Configuration**: Restrict cross-origin requests

### 2. Data Privacy
- **Local Processing**: Option to avoid cloud APIs
- **Session Isolation**: No cross-session data leakage
- **Minimal Storage**: Only store necessary conversation data

### 3. Model Safety
- **Crisis Detection**: Automatic identification of high-risk content
- **Professional Boundaries**: Prevent inappropriate therapeutic claims
- **Content Filtering**: Remove harmful or inappropriate responses

## Scalability Architecture

### Horizontal Scaling Options
```
Load Balancer → Multiple Chat Server Instances → Shared Database
```

### Microservice Decomposition
```
API Gateway → Model Selector Service → Evaluation Service → Session Service
```

### Cloud Deployment
```
Container Orchestration (Docker/K8s) → Auto-scaling → Database Cluster
```

## Monitoring and Observability

### Health Checks
- **System Health**: `/api/health` endpoint
- **Model Availability**: Periodic model health checks  
- **Database Connectivity**: SQLite connection monitoring

### Performance Metrics
- **Response Times**: Model-specific latency tracking
- **Selection Analytics**: Model usage statistics
- **Error Rates**: Failure monitoring and alerting

### Logging Strategy
- **Structured Logging**: JSON format for analysis
- **Level-based Filtering**: DEBUG/INFO/WARN/ERROR
- **Privacy-Safe**: No sensitive data in logs

## Development Architecture

### Local Development
```bash
# Single-machine setup
python chat_server.py          # Main application
LM Studio                      # Local model server  
SQLite                        # Local database
```

### Production Deployment
```bash
# Multi-container setup
docker-compose up              # Orchestrated deployment
nginx                         # Reverse proxy
PostgreSQL/MySQL              # Production database
Redis                         # Caching layer
```

---

*For API details, see [API_DOCUMENTATION.md](API_DOCUMENTATION.md)*  
*For deployment guide, see [../guides/DEPLOYMENT_GUIDE.md](../guides/DEPLOYMENT_GUIDE.md)*