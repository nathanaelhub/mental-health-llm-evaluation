# Adding New Models to the Mental Health LLM Evaluation Framework

This guide explains how to easily add new LLM models to the evaluation framework using our flexible registry and factory system.

## Quick Start

Adding a new model is as simple as:

1. **Copy template file** (claude_client.py or llama_client.py)
2. **Implement model-specific API calls**
3. **Update configuration**
4. **Run tests**

## Table of Contents

- [Overview](#overview)
- [Cloud Model Template (Anthropic Claude)](#cloud-model-template)
- [Local Model Template (Meta Llama)](#local-model-template)
- [Step-by-Step Instructions](#step-by-step-instructions)
- [Configuration Updates](#configuration-updates)
- [Testing Requirements](#testing-requirements)
- [Advanced Features](#advanced-features)
- [Troubleshooting](#troubleshooting)

## Overview

The framework uses a registry-based architecture that automatically discovers and manages model implementations. Each model inherits from `BaseModel` and implements standard methods for conversation generation.

### Architecture Components

- **BaseModel**: Abstract interface all models must implement
- **ModelRegistry**: Automatic discovery and registration system
- **ModelFactory**: Centralized model creation and configuration
- **Auto-registration**: Decorator-based model registration

## Cloud Model Template (Anthropic Claude)

For cloud-based models, use `src/models/claude_client.py` as a template.

### Key Features
- Async API calls with proper error handling
- Token counting and cost calculation
- Health check implementation
- Configuration validation
- Auto-registration via decorator

### Template Structure
```python
@register_model_decorator(
    name="your-model-name",
    provider=ModelProvider.YOUR_PROVIDER,
    model_type=ModelType.CLOUD,
    description="Description of your model",
    requirements=["required-packages"],
    default_config={
        "model": "model-id",
        "temperature": 0.7,
        "max_tokens": 1000
    }
)
class YourModelClient(BaseModel):
    # Implementation here
```

## Local Model Template (Meta Llama)

For local models, use `src/models/llama_client.py` as a template.

### Key Features
- Local model loading and management
- GPU/CPU device handling
- Memory management utilities
- Quantization support (8-bit, 4-bit)
- Model unloading for memory cleanup

### Template Structure
```python
@register_model_decorator(
    name="your-local-model",
    provider=ModelProvider.YOUR_PROVIDER,
    model_type=ModelType.LOCAL,
    description="Description of your local model",
    requirements=["transformers", "torch"],
    default_config={
        "model_path": "./models/your-model",
        "device": "auto",
        "precision": "fp16"
    }
)
class YourLocalModelClient(BaseModel):
    # Implementation here
```

## Step-by-Step Instructions

### 1. Choose Template

**For Cloud Models (APIs):**
```bash
cp src/models/claude_client.py src/models/your_model_client.py
```

**For Local Models:**
```bash
cp src/models/llama_client.py src/models/your_model_client.py
```

### 2. Implement Required Methods

Every model must implement these abstract methods:

#### `generate_response()`
```python
async def generate_response(
    self,
    prompt: str,
    conversation_history: Optional[List[Dict[str, str]]] = None,
    **kwargs
) -> ModelResponse:
    """Generate response to user prompt."""
    # Your implementation here
```

#### `validate_configuration()`
```python
def validate_configuration(self) -> bool:
    """Validate model configuration."""
    # Check API keys, model paths, parameters, etc.
    return True
```

#### `health_check()`
```python
async def health_check(self) -> bool:
    """Test if model is working."""
    # Make simple API call or model inference
    return True
```

#### `get_model_info()`
```python
def get_model_info(self) -> Dict[str, Any]:
    """Return model capabilities and info."""
    return {
        "provider": "Your Provider",
        "model_name": self.model_name,
        "type": "cloud" or "local",
        "max_context_length": 4096,
        "supports_streaming": True,
        "config": self.config
    }
```

### 3. Update Model Registry

The registry decorator automatically registers your model:

```python
@register_model_decorator(
    name="your-model",           # Unique identifier
    provider=ModelProvider.YOUR_PROVIDER,  # Add to enum if needed
    model_type=ModelType.CLOUD,  # or ModelType.LOCAL
    description="Your model description",
    requirements=["package1", "package2"],
    default_config={
        # Default configuration parameters
    }
)
```

### 4. Add Provider to Enum (if new)

If your provider isn't in the enum, add it to `src/models/base_model.py`:

```python
class ModelProvider(Enum):
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"
    META = "meta"
    DEEPSEEK = "deepseek"
    MISTRAL = "mistral"
    YOUR_PROVIDER = "your_provider"  # Add this
    CUSTOM = "custom"
```

## Configuration Updates

### 1. Update Experiment Configuration

Add your model to `config/experiment_template.yaml`:

```yaml
models:
  cloud:  # or local
    - name: "your-model"
      provider: "your_provider"
      enabled: false  # Set to true when ready
      model: "model-id"
      temperature: 0.7
      max_tokens: 1000
      # Add model-specific parameters
```

### 2. Environment Variables

For cloud models, document required environment variables:

```bash
# Add to .env or set in environment
YOUR_PROVIDER_API_KEY=your-api-key-here
YOUR_PROVIDER_ORG_ID=optional-org-id
```

For local models, document model download/setup:

```bash
# Download model to models directory
mkdir -p models/your-model
# Copy or download model files here
```

## Testing Requirements

### 1. Unit Tests

Create test file `tests/test_your_model_client.py`:

```python
import pytest
from src.models.your_model_client import YourModelClient

@pytest.mark.asyncio
async def test_model_creation():
    """Test model can be created."""
    config = {"temperature": 0.5}
    client = YourModelClient(config=config)
    assert client.model_name == "your-model"

@pytest.mark.asyncio
async def test_health_check():
    """Test health check passes."""
    client = YourModelClient()
    # Mock API calls if needed
    is_healthy = await client.health_check()
    assert is_healthy == True

@pytest.mark.asyncio
async def test_generate_response():
    """Test response generation."""
    client = YourModelClient()
    response = await client.generate_response("Hello")
    assert response.is_successful
    assert len(response.content) > 0
```

### 2. Integration Tests

Test with the registry system:

```python
def test_model_registration():
    """Test model is properly registered."""
    from src.models import get_model_registry
    
    registry = get_model_registry()
    model_info = registry.get_model("your-model")
    assert model_info is not None
    assert model_info.provider.value == "your_provider"

def test_factory_creation():
    """Test factory can create model."""
    from src.models import create_model
    
    model = create_model("your-model", config={"temperature": 0.7})
    assert model is not None
    assert model.model_name == "your-model"
```

### 3. End-to-End Tests

Test with pipeline scripts:

```bash
# Test model in pipeline
python scripts/setup_experiment.py --models your-model --dry-run
python scripts/run_conversations.py --experiment test_exp --models your-model --dry-run
```

## Advanced Features

### 1. Model Variants

Create multiple variants of the same model:

```python
@register_model_decorator(
    name="your-model-large",
    provider=ModelProvider.YOUR_PROVIDER,
    model_type=ModelType.LOCAL,
    default_config={"model_path": "./models/your-model-large"}
)
class YourModelLargeClient(YourModelClient):
    """Larger variant of your model."""
    pass
```

### 2. Custom Metrics

Add model-specific metrics:

```python
def get_model_info(self) -> Dict[str, Any]:
    info = super().get_model_info()
    info.update({
        "custom_metric": self.calculate_custom_metric(),
        "special_features": ["feature1", "feature2"]
    })
    return info
```

### 3. Streaming Support

For models that support streaming:

```python
async def generate_response_stream(
    self,
    prompt: str,
    **kwargs
) -> AsyncGenerator[str, None]:
    """Generate streaming response."""
    # Implement streaming logic
    async for chunk in self.stream_api_call(prompt):
        yield chunk
```

### 4. Function Calling

For models with function calling capabilities:

```python
def supports_function_calling(self) -> bool:
    """Check if model supports function calling."""
    return True

async def call_function(
    self,
    function_name: str,
    arguments: Dict[str, Any]
) -> Any:
    """Execute function call."""
    # Implement function calling logic
```

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```
   ImportError: No module named 'your_package'
   ```
   - Add missing packages to requirements in decorator
   - Install dependencies: `pip install your_package`

2. **Registry Not Finding Model**
   ```
   Model not found: your-model
   ```
   - Ensure model file is imported in `src/models/__init__.py`
   - Check decorator name matches config

3. **Health Check Failing**
   ```
   Health check failed for your-model
   ```
   - Verify API keys/model paths
   - Check network connectivity
   - Validate configuration parameters

4. **Memory Issues (Local Models)**
   ```
   CUDA out of memory
   ```
   - Enable quantization: `load_in_8bit: true`
   - Reduce batch size or max tokens
   - Use smaller model variant

### Debug Commands

Check model registration:
```python
from src.models import get_model_registry
registry = get_model_registry()
print(registry.list_models())
```

Test model creation:
```python
from src.models import create_model
model = create_model("your-model")
print(model.get_model_info())
```

Validate configuration:
```python
from src.models import get_model_factory
factory = get_model_factory()
config = {"models": {"your-model": {"enabled": True}}}
results = factory.validate_config(config)
print(results)
```

## Example: Adding Google Gemini

Here's a complete example of adding Google's Gemini model:

### 1. Create Model File

`src/models/gemini_client.py`:
```python
@register_model_decorator(
    name="gemini-pro",
    provider=ModelProvider.GOOGLE,
    model_type=ModelType.CLOUD,
    description="Google Gemini Pro model",
    requirements=["google-generativeai"],
    default_config={
        "model": "gemini-pro",
        "temperature": 0.7,
        "max_tokens": 1000
    }
)
class GeminiClient(BaseModel):
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # Implementation...
```

### 2. Update Configuration

`config/experiment_template.yaml`:
```yaml
models:
  cloud:
    - name: "gemini-pro"
      provider: "google"
      enabled: false
      model: "gemini-pro"
      temperature: 0.7
      max_tokens: 1000
```

### 3. Set Environment Variables

```bash
export GOOGLE_API_KEY=your-api-key
```

### 4. Test Integration

```bash
python -c "
from src.models import create_model
model = create_model('gemini-pro')
print('Gemini model created successfully!')
"
```

## Summary

The framework's flexible architecture makes adding new models straightforward:

1. **Copy appropriate template** (cloud or local)
2. **Implement required methods** for your model's API
3. **Update configuration** to include your model
4. **Test thoroughly** with provided test templates
5. **Your model is ready** for evaluation!

The registry system automatically discovers and manages your model, making it available throughout the entire evaluation pipeline without any additional integration work.

For questions or issues, refer to the troubleshooting section or check existing model implementations for examples.