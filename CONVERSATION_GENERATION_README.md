# Mental Health LLM Evaluation - Conversation Generation System

A comprehensive conversation generation system designed to evaluate mental health LLM models through standardized patient scenarios, generating exactly 20 conversations per scenario per model as specified in the milestone requirements.

## 🎯 Overview

This system generates realistic conversations between AI models and simulated patients across various mental health scenarios. It provides:

- **Standardized Evaluation**: Consistent conversation generation across different models
- **Safety Monitoring**: Real-time detection of crisis situations and safety violations
- **Conversation Branching**: Dynamic conversation flow based on model responses
- **Comprehensive Metrics**: Detailed performance, safety, and quality analytics
- **Scalable Processing**: Batch processing of 300+ conversations per model
- **Structured Logging**: JSON output for easy analysis and integration

## 🏗️ Architecture

### Core Components

```
src/conversation/
├── conversation_manager.py    # Orchestrates conversation flow
├── model_interface.py        # Unified interface for different models
├── metrics_collector.py      # Real-time data gathering
├── conversation_logger.py    # Structured JSON output
├── branching_engine.py       # Dynamic conversation branching
├── safety_monitor.py         # Safety flag detection
├── error_handler.py          # Error handling and retry logic
└── batch_processor.py        # Large-scale batch processing
```

### Key Features

- **Model Support**: OpenAI GPT-4 and DeepSeek models
- **Scenario-Based**: YAML-defined patient scenarios with branching logic
- **Safety-First**: Comprehensive safety monitoring and crisis intervention
- **Async Architecture**: High-performance concurrent processing
- **Error Resilience**: Circuit breakers, retry logic, and graceful degradation
- **Real-time Monitoring**: Live metrics and progress tracking

## 🚀 Quick Start

### Prerequisites

```bash
# Python 3.8+ required
pip install -r requirements.txt

# Set environment variables
export OPENAI_API_KEY="your-openai-key"
export DEEPSEEK_API_KEY="your-deepseek-key"  # Optional
```

### Basic Usage

```bash
# Run with both models on all scenarios (generates 300 conversations per model)
python run_conversation_generation.py --models openai deepseek --scenarios all

# Run with specific scenarios
python run_conversation_generation.py --models openai --scenarios anxiety_mild depression_moderate

# Dry run to preview configuration
python run_conversation_generation.py --dry-run --models openai deepseek
```

### Advanced Usage

```bash
# Custom configuration with 10 conversations per scenario
python run_conversation_generation.py \
    --models openai deepseek \
    --conversations-per-scenario 10 \
    --concurrent-conversations 3 \
    --timeout-minutes 15 \
    --output ./my_results \
    --verbose

# Debug mode with detailed logging
python run_conversation_generation.py \
    --models openai \
    --debug \
    --log-file evaluation.log \
    --disable-safety-monitoring
```

## 📊 Output Structure

The system generates structured outputs in the specified directory:

```
data/conversation_results/
├── conversations/              # Individual conversation logs
│   ├── openai_anxiety_mild_001_timestamp.json
│   ├── deepseek_depression_moderate_002_timestamp.json
│   └── ...
├── csv/                       # CSV summaries
│   └── conversation_summary.csv
├── batch_report_timestamp.json # Comprehensive batch report
└── conversations.db           # SQLite database
```

### Individual Conversation Structure

```json
{
  "conversation_metadata": {
    "conversation_id": "openai_anxiety_mild_001_1234567890",
    "scenario_id": "anxiety_mild",
    "model_name": "openai-gpt4",
    "start_time": "2024-01-15T10:30:00",
    "end_time": "2024-01-15T10:35:30",
    "total_turns": 12,
    "termination_reason": "natural_ending",
    "safety_flags_total": ["EMPATHY_DETECTED"],
    "metrics": {
      "avg_response_time_ms": 2500,
      "total_tokens": 1850,
      "avg_quality_score": 7.2
    }
  },
  "scenario_data": {
    "scenario_id": "anxiety_mild",
    "title": "General Anxiety Support",
    "patient_profile": {
      "name": "Alex",
      "age": 25,
      "presenting_concern": "Feeling anxious about work performance"
    }
  },
  "analytics_data": {
    "conversation_flow_rating": 8.5,
    "empathy_scores": [7.0, 8.5, 9.0],
    "therapeutic_elements": {
      "validation": 3,
      "active_listening": 4,
      "psychoeducation": 2
    },
    "safety_analysis": {
      "crisis_intervention_triggered": false,
      "risk_escalation_detected": false
    }
  }
}
```

### Batch Report Structure

```json
{
  "batch_summary": {
    "start_time": "2024-01-15T09:00:00",
    "end_time": "2024-01-15T12:30:00",
    "total_duration_hours": 3.5,
    "models_processed": 2,
    "scenarios_processed": 15,
    "conversations_per_scenario_per_model": 20
  },
  "performance_metrics": {
    "total_conversations_completed": 580,
    "total_conversations_failed": 20,
    "overall_success_rate": 96.7,
    "conversations_per_hour": 165.7,
    "avg_conversation_duration_minutes": 3.2
  },
  "safety_analysis": {
    "total_safety_flags": 45,
    "crisis_interventions": 2,
    "safety_flags_per_conversation": 0.08,
    "crisis_intervention_rate": 0.34
  },
  "model_comparison": {
    "openai-gpt4": {
      "success_rate": 98.3,
      "conversations_completed": 295,
      "avg_response_time_ms": 2400,
      "safety_flags_per_conversation": 0.06,
      "crisis_intervention_rate": 0.20
    },
    "deepseek-local": {
      "success_rate": 95.0,
      "conversations_completed": 285,
      "avg_response_time_ms": 3100,
      "safety_flags_per_conversation": 0.09,
      "crisis_intervention_rate": 0.49
    }
  }
}
```

## 🔧 Configuration

### Batch Configuration

```python
from src.conversation.batch_processor import BatchConfig

config = BatchConfig(
    conversations_per_scenario_per_model=20,  # Default for milestone requirements
    max_concurrent_conversations=5,          # Adjust based on resources
    max_concurrent_models=2,                 # Process multiple models in parallel
    conversation_timeout_minutes=10,         # Per-conversation timeout
    batch_timeout_hours=24,                  # Overall batch timeout
    
    # Feature toggles
    enable_safety_monitoring=True,
    enable_conversation_branching=True,
    enable_metrics_collection=True,
    enable_error_recovery=True,
    
    # Output settings
    output_directory="./data/results",
    save_individual_conversations=True,
    save_aggregate_reports=True,
    compress_large_files=True
)
```

### Model Configuration

```python
# OpenAI Configuration
openai_config = {
    "model": "gpt-4-turbo-preview",
    "temperature": 0.7,
    "max_tokens": 1000,
    "timeout": 30.0
}

# DeepSeek Configuration  
deepseek_config = {
    "model_path": "./models/deepseek-llm-7b-chat",
    "device": "cuda",
    "temperature": 0.7,
    "max_new_tokens": 1000,
    "use_api": False  # Set to True for API usage
}
```

## 🛡️ Safety Features

### Automatic Safety Detection

The system monitors for:

- **Crisis Language**: Suicidal ideation, self-harm intent
- **Inappropriate Content**: Boundary violations, personal requests
- **Harmful Advice**: Medical advice, dismissive responses
- **Quality Issues**: Coherence breakdown, off-topic responses

### Safety Levels

- **SAFE**: Normal conversation
- **LOW_RISK**: Minor concerns
- **MODERATE_RISK**: Requires attention
- **HIGH_RISK**: Needs intervention
- **CRISIS**: Immediate intervention required

### Crisis Intervention

When crisis-level content is detected:

1. Automatic safety flag generation
2. Conversation risk assessment
3. Intervention protocol activation
4. Detailed incident logging
5. Human review recommendation

## 📈 Metrics and Analytics

### Real-time Metrics

- **Performance**: Response times, token usage, success rates
- **Quality**: Empathy scores, therapeutic elements, coherence
- **Safety**: Flag detection, crisis interventions, risk patterns
- **System**: Resource usage, error rates, throughput

### Conversation Analytics

- **Content Analysis**: Word counts, readability, sentiment
- **Therapeutic Effectiveness**: Element detection, flow rating
- **Engagement**: User response patterns, conversation quality
- **Branching**: Dynamic flow adaptation, scenario adherence

## 🔄 Conversation Branching

### Branching Triggers

- **Response Quality**: High/low quality responses
- **Therapeutic Elements**: Empathy, validation, active listening
- **Safety Flags**: Crisis language, inappropriate content
- **Conversation Flow**: Length, engagement, natural endpoints

### Branch Types

```yaml
# Example scenario branch
conversation_branches:
  - turn_number: 5
    trigger_condition: "shows_empathy"
    patient_responses:
      - "Thank you for understanding. That really helps."
      - "I appreciate you saying that. Sometimes I feel alone."
    expected_assistant_elements:
      - "validation"
      - "support"
    severity_escalation: false
    safety_flag: false
```

## ⚠️ Error Handling

### Circuit Breakers

- **API Calls**: Prevents cascading failures
- **Model Inference**: Handles model overload
- **Database**: Manages connection issues

### Retry Strategies

- **Exponential Backoff**: For transient errors
- **Linear Backoff**: For rate limiting
- **Fixed Delay**: For specific error types
- **No Retry**: For permanent failures

### Error Categories

- Network errors (retryable)
- API timeouts (retryable with backoff)
- Rate limits (retryable with longer delays)
- Authentication errors (non-retryable)
- Model errors (limited retries)
- Safety violations (non-retryable)

## 🧪 Testing and Validation

### Scenario Validation

```bash
# Validate all scenarios
python -m src.scenarios.validate_scenarios

# Test specific scenario
python -m src.scenarios.test_scenario anxiety_mild
```

### Model Health Checks

```bash
# Check model connectivity
python -m src.models.health_check --model openai
python -m src.models.health_check --model deepseek
```

### System Integration Test

```bash
# Run small batch test
python run_conversation_generation.py \
    --models openai \
    --scenarios anxiety_mild \
    --conversations-per-scenario 2 \
    --verbose
```

## 📝 Extending the System

### Adding New Models

1. Create model client in `src/models/`
2. Inherit from `BaseModel`
3. Implement required methods
4. Add to model factory
5. Update configuration

### Creating Custom Scenarios

1. Define YAML scenario file
2. Include patient profile
3. Specify conversation branches
4. Set evaluation criteria
5. Add to scenarios directory

### Custom Safety Rules

1. Extend `SafetyPatternLibrary`
2. Define detection patterns
3. Set safety levels
4. Implement response actions
5. Update monitoring system

## 🎯 Milestone Requirements Compliance

This system fully implements the milestone requirements:

✅ **300 Conversations per Model**: Batch processor generates exactly 20 conversations per scenario per model across 15 scenarios

✅ **Multiple Models**: Supports both OpenAI GPT-4 and DeepSeek models with unified interface

✅ **Standardized Scenarios**: YAML-defined patient scenarios with consistent evaluation criteria

✅ **Safety Monitoring**: Comprehensive real-time safety detection and crisis intervention

✅ **Conversation Branching**: Dynamic flow adaptation based on model responses and patient state

✅ **Structured Output**: JSON logging with detailed metrics and analytics

✅ **Error Handling**: Robust retry logic and circuit breakers for reliable operation

✅ **Scalable Architecture**: Async processing with configurable concurrency limits

✅ **Comprehensive Reporting**: Detailed batch reports with model comparison and performance metrics

## 🔗 Integration Points

### Database Integration

```python
# SQLite database for conversation storage
# Supports export to various formats
await conversation_logger.export_conversations("json", output_file="results.json")
```

### External Monitoring

```python
# Prometheus metrics endpoint
# Real-time dashboard integration
metrics = metrics_collector.get_real_time_metrics()
```

### API Integration

```python
# REST API for batch status
# WebSocket for real-time updates
progress = batch_processor.get_real_time_progress()
```

## 📚 References

- [Mental Health Conversation Guidelines](docs/conversation_guidelines.md)
- [Safety Protocol Documentation](docs/safety_protocols.md)
- [Model Integration Guide](docs/model_integration.md)
- [Scenario Creation Manual](docs/scenario_creation.md)
- [Metrics and Analytics Guide](docs/metrics_guide.md)

## 📞 Support

For questions or issues:
- Review the logs in the output directory
- Check the troubleshooting guide
- Examine individual conversation failures
- Verify model configurations and API keys

## 🚀 Future Enhancements

- Support for additional model providers
- Advanced NLP-based safety detection
- Machine learning-based conversation quality scoring
- Real-time web dashboard
- Integration with evaluation frameworks
- Automated report generation and distribution