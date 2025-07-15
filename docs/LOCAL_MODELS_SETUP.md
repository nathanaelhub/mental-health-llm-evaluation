# Local Models Setup Guide

## Unified Server Configuration

Both DeepSeek and Gemma models now run on the same local server with different API identifiers.

### Server Configuration
- **Server Address**: `192.168.86.23:1234`
- **Base URL**: `http://192.168.86.23:1234/v1`

### Model Identifiers
- **DeepSeek**: `deepseek-r1`
- **Gemma**: `google/gemma-3-12b`

## Environment Variables

In your `.env` file:
```bash
# Unified local server configuration
LOCAL_LLM_SERVER=192.168.86.23:1234

# Model identifiers on the server
DEEPSEEK_MODEL=deepseek-r1
GEMMA_MODEL=google/gemma-3-12b
```

## Model Endpoints

### DeepSeek
- **Full URL**: `http://192.168.86.23:1234/v1`
- **Model ID**: `deepseek-r1`
- **Usage**: Standard OpenAI-compatible API

### Gemma  
- **Full URL**: `http://192.168.86.23:1234/v1/chat/completions`
- **Model ID**: `google/gemma-3-12b`
- **Usage**: OpenAI-compatible chat completions

## Testing Connection

```bash
# Test server is accessible
curl http://192.168.86.23:1234/v1/models

# Test DeepSeek model
python -c "from src.models import DeepSeekClient; client = DeepSeekClient(); print('DeepSeek ready')"

# Test Gemma model
python -c "from src.models import GemmaClient; client = GemmaClient(); print('Gemma ready')"
```

## Usage in Research

Both models will automatically use the unified server configuration when running:

```bash
# Research pipeline with all models
python scripts/run_research.py

# Compare local models only
python tools/compare_models.py --models "deepseek,gemma"
```

## Configuration File

The `config/main.yaml` is updated to use the unified server:

```yaml
models:
  deepseek:
    api_url: "http://${LOCAL_LLM_SERVER:-192.168.86.23:1234}/v1"
    model_path: "deepseek-r1"
    
  gemma:
    endpoint: "http://${LOCAL_LLM_SERVER:-192.168.86.23:1234}/v1/chat/completions"
    model: "google/gemma-3-12b"
```

This setup simplifies configuration and ensures both local models connect to the same server instance with appropriate model identifiers.