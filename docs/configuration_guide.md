# Configuration System Guide

This guide explains how to use the comprehensive configuration system for the Mental Health LLM Evaluation project.

## Overview

The configuration system provides:
- **YAML-based configuration files** for different environments
- **Environment variable support** for sensitive data
- **Comprehensive validation** of all parameters
- **Easy runtime access** to configuration values
- **Development/production separation** with appropriate defaults

## Quick Start

1. **Copy environment template:**
```bash
cp .env.template .env
# Edit .env with your API keys and paths
```

2. **Validate configuration:**
```bash
python scripts/validate_config.py --environment development --check-env
```

3. **Run evaluation with configuration:**
```bash
python scripts/run_evaluation_with_config.py --environment development
```

## Configuration Files

### Directory Structure
```
config/
├── base.yaml          # Common settings for all environments
├── development.yaml   # Development-specific settings
├── production.yaml    # Production-specific settings
└── testing.yaml       # Testing environment settings (optional)
```

### Environment Files
- **`.env.template`** - Template with all available environment variables
- **`.env.development`** - Development defaults
- **`.env.production`** - Production defaults
- **`.env`** - Your local settings (not in git)

## Configuration Sections

### 1. Models Configuration

Configure OpenAI and DeepSeek models:

```yaml
models:
  # OpenAI Configuration
  openai:
    api_key: "${OPENAI_API_KEY}"
    model: "gpt-4-turbo-preview"
    temperature: 0.7
    max_tokens: 1000
    timeout: 30.0
  
  # DeepSeek Configuration
  deepseek:
    model_path: "${DEEPSEEK_MODEL_PATH}"
    device: "auto"
    temperature: 0.7
    max_new_tokens: 1000
  
  # Enabled models for evaluation
  enabled_models: ["openai", "deepseek"]
```

### 2. Evaluation Configuration

Configure evaluation metrics and weights:

```yaml
evaluation:
  # Technical metrics (must sum to 1.0)
  technical:
    response_time_weight: 0.3
    throughput_weight: 0.25
    reliability_weight: 0.3
    efficiency_weight: 0.15
  
  # Therapeutic metrics (must sum to 1.0)
  therapeutic:
    empathy_weight: 0.3
    coherence_weight: 0.25
    safety_weight: 0.35    # Highest priority
    boundaries_weight: 0.1
  
  # Patient experience (must sum to 1.0)
  patient:
    satisfaction_weight: 0.4
    engagement_weight: 0.25
    trust_weight: 0.25
    accessibility_weight: 0.1
  
  # Composite weights (must sum to 1.0)
  technical_weight: 0.3
  therapeutic_weight: 0.5  # Highest priority
  patient_weight: 0.2
  
  # Readiness thresholds
  production_ready_threshold: 80.0
  clinical_ready_threshold: 90.0
  research_acceptable_threshold: 70.0
  minimum_viable_threshold: 60.0
```

### 3. Experiment Configuration

Configure experiment parameters:

```yaml
experiment:
  conversation_count: 5
  scenario_suite: "comprehensive"
  conversations_per_scenario: 2
  parallel_evaluations: true
  max_parallel_workers: 4
  random_seed: 42
  
  # Data collection
  save_conversations: true
  save_intermediate_results: true
  export_formats: ["json", "csv"]
```

### 4. Logging Configuration

Configure logging behavior:

```yaml
logging:
  level: "INFO"
  format: "standard"  # standard, detailed, json
  file_path: "./logs/evaluation.log"
  enable_console: true
  enable_file: true
  enable_structured: false
  
  # External library levels
  external_loggers:
    urllib3: "WARNING"
    requests: "WARNING"
    transformers: "WARNING"
```

## Environment Variables

### Required Variables
- **`OPENAI_API_KEY`** - Your OpenAI API key
- **`DEEPSEEK_MODEL_PATH`** - Path to local DeepSeek model

### Optional Variables
- **`OPENAI_ORG_ID`** - OpenAI organization ID
- **`DEEPSEEK_API_KEY`** - DeepSeek API key (if using API)
- **`ENVIRONMENT`** - Environment name (development/production)
- **`LOG_LEVEL`** - Logging level override
- **`DATA_DIR`** - Base data directory
- **`OUTPUT_DIR`** - Output directory for results

### Environment Variable Syntax

In YAML files, use `${VAR_NAME}` syntax:

```yaml
# Simple substitution
api_key: "${OPENAI_API_KEY}"

# With default value
model_path: "${DEEPSEEK_MODEL_PATH:-./models/deepseek}"

# Alternative default syntax
device: "${MODEL_DEVICE:auto}"
```

## Using the Configuration System

### 1. Loading Configuration

```python
from config.config_loader import get_config

# Load default environment configuration
config = get_config()

# Load specific environment
config = get_config(environment="production")

# Load specific file
config = get_config(config_file="./config/custom.yaml")
```

### 2. Accessing Configuration Values

```python
# Access model configuration
openai_config = config.models.openai
deepseek_config = config.models.deepseek

# Access evaluation weights
tech_weights = config.evaluation.technical
therapeutic_weights = config.evaluation.therapeutic

# Access experiment settings
conv_count = config.experiment.conversation_count
scenario_suite = config.experiment.scenario_suite
```

### 3. Configuration Validation

```python
from config.config_loader import validate_environment

# Validate environment variables
if validate_environment():
    print("Environment is properly configured")
else:
    print("Environment validation failed")
```

### 4. Getting Model-Specific Configuration

```python
from config.config_loader import get_model_config

# Get OpenAI configuration
openai_config = get_model_config("openai")

# Get DeepSeek configuration  
deepseek_config = get_model_config("deepseek")
```

## Command Line Tools

### Configuration Validation

```bash
# Validate development configuration
python scripts/validate_config.py --environment development

# Check environment variables
python scripts/validate_config.py --check-env

# Show configuration details
python scripts/validate_config.py --show-config

# Save configuration summary
python scripts/validate_config.py --output-summary config_summary.json
```

### Running Evaluations

```bash
# Use development configuration
python scripts/run_evaluation_with_config.py --environment development

# Override specific settings
python scripts/run_evaluation_with_config.py \
  --environment production \
  --models openai \
  --scenario-suite basic \
  --conversation-count 10

# Validate configuration only
python scripts/run_evaluation_with_config.py --validate-only
```

## Configuration Examples

### Development Setup
```bash
# 1. Copy environment template
cp .env.template .env

# 2. Edit .env file
OPENAI_API_KEY=sk-your-key-here
DEEPSEEK_MODEL_PATH=./models/deepseek-llm-7b-chat
ENVIRONMENT=development
LOG_LEVEL=DEBUG

# 3. Validate configuration
python scripts/validate_config.py --environment development --check-env

# 4. Run evaluation
python scripts/run_evaluation_with_config.py --environment development
```

### Production Deployment
```bash
# 1. Set production environment variables
export OPENAI_API_KEY="sk-prod-key-here"
export DEEPSEEK_MODEL_PATH="/opt/models/deepseek-llm-7b-chat"
export ENVIRONMENT="production"
export DATA_DIR="/var/lib/mental-health-llm/data"
export LOG_DIR="/var/log/mental-health-llm"

# 2. Validate production configuration
python scripts/validate_config.py --environment production --check-env

# 3. Run production evaluation
python scripts/run_evaluation_with_config.py --environment production
```

## Best Practices

### 1. Environment Management
- **Development:** Use lower thresholds, verbose logging, fewer conversations
- **Production:** Use strict thresholds, structured logging, comprehensive evaluation
- **Testing:** Use mock settings, minimal data, fast execution

### 2. Security
- Never commit `.env` files to version control
- Use environment variables for all sensitive data
- Rotate API keys regularly
- Use different keys for development and production

### 3. Configuration Validation
- Always validate configuration before running evaluations
- Use the validation script in CI/CD pipelines
- Test configuration changes in development first

### 4. Customization
- Override specific settings via command line arguments
- Create custom configuration files for special experiments
- Use environment variable defaults for flexibility

## Troubleshooting

### Common Issues

1. **Missing API Keys**
   ```bash
   # Check if keys are set
   python scripts/validate_config.py --check-env
   ```

2. **Model Path Issues**
   ```bash
   # Verify model path exists
   ls -la "$DEEPSEEK_MODEL_PATH"
   ```

3. **Configuration Validation Errors**
   ```bash
   # Check detailed validation
   python scripts/validate_config.py --verbose --show-config
   ```

4. **Permission Issues**
   ```bash
   # Check directory permissions
   mkdir -p ./logs ./data ./output
   chmod 755 ./logs ./data ./output
   ```

### Debug Mode

Enable debug mode for detailed troubleshooting:

```yaml
# In configuration file
debug: true
logging:
  level: "DEBUG"
  format: "detailed"
```

Or via environment:
```bash
export DEBUG=true
export LOG_LEVEL=DEBUG
```

## Advanced Features

### 1. Configuration Merging
Base configuration + Environment configuration + Command line overrides

### 2. Environment Variable Interpolation
Reference other configuration values within the config

### 3. Sensitive Data Masking
Automatically mask sensitive data in logs and displays

### 4. Configuration Backup
Automatic backup of configuration before changes

### 5. Runtime Configuration Updates
Hot-reload configuration without restarting

## API Reference

See the source code documentation in:
- `src/config/config_schema.py` - Configuration schema and validation
- `src/config/config_loader.py` - Configuration loading and access
- `src/config/config_utils.py` - Utility functions

For detailed API documentation, run:
```bash
python -c "from config.config_loader import ConfigLoader; help(ConfigLoader)"
```