# Getting Started

This guide will help you set up and run your first mental health LLM evaluation.

## Quick Start

1. **Clone and Setup**
```bash
git clone https://github.com/your-username/mental-health-llm-evaluation.git
cd mental-health-llm-evaluation
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
pip install -r requirements.txt
```

2. **Configure Environment**
```bash
cp .env.example .env
# Edit .env with your API keys and configuration
```

3. **Run Basic Evaluation**
```python
from src.models.openai_client import OpenAIClient
from src.models.deepseek_client import DeepSeekClient
from src.evaluation.composite_scorer import CompositeScorer
from src.scenarios.scenario_loader import ScenarioLoader

# Initialize models
openai_model = OpenAIClient()
deepseek_model = DeepSeekClient()

# Load evaluation scenarios
loader = ScenarioLoader()
scenarios = loader.get_evaluation_suite("basic")

# Run evaluation
scorer = CompositeScorer()
results = scorer.compare_models([openai_model, deepseek_model], {
    "therapeutic_scenarios": scenarios,
    "technical_prompts": ["Hello", "How are you?", "Can you help me?"],
    "conversation_data": []
})

print("Evaluation complete!")
for model_name, result in results.items():
    print(f"{model_name}: {result.overall_score:.1f}/100")
```

## Next Steps

- Read the [Architecture Guide](architecture.md) to understand the system design
- See [Configuration Guide](configuration.md) for detailed setup options
- Check [API Reference](api_reference.md) for complete documentation
- Review [Examples](examples/) for common use cases