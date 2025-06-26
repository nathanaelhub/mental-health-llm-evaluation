# Local Deployment Guide

This guide provides detailed instructions for setting up and running the Mental Health LLM Evaluation framework in a local development environment.

## Table of Contents

- [System Requirements](#system-requirements)
- [Installation](#installation)
- [Configuration](#configuration)
- [Model Setup](#model-setup)
- [Running the Framework](#running-the-framework)
- [Development Workflow](#development-workflow)
- [Troubleshooting](#troubleshooting)
- [Performance Optimization](#performance-optimization)

## System Requirements

### Minimum Requirements

**Hardware**:
- CPU: 4-core processor (Intel i5 or AMD Ryzen 5 equivalent)
- RAM: 8GB (16GB recommended for local model inference)
- Storage: 50GB free space (additional space for model files)
- GPU: Optional but recommended for DeepSeek local inference

**Software**:
- Python 3.8 or higher
- Git
- Operating System: Windows 10/11, macOS 10.15+, or Ubuntu 18.04+

### Recommended Requirements

**Hardware**:
- CPU: 8-core processor (Intel i7/i9 or AMD Ryzen 7/9)
- RAM: 32GB for optimal performance with local models
- Storage: 200GB+ SSD for fast model loading
- GPU: NVIDIA RTX 3080 or better (8GB+ VRAM) for local inference

**Software**:
- Python 3.9 or 3.10
- CUDA 11.8+ (for GPU acceleration)
- Docker (optional but recommended)

## Installation

### 1. Clone the Repository

```bash
# Clone the repository
git clone https://github.com/your-username/mental-health-llm-evaluation.git
cd mental-health-llm-evaluation

# Verify you're on the main branch
git branch
```

### 2. Create Virtual Environment

#### Using venv (Recommended)

**macOS/Linux**:
```bash
# Create virtual environment
python3 -m venv venv

# Activate virtual environment
source venv/bin/activate

# Verify activation
which python
# Should show: /path/to/mental-health-llm-evaluation/venv/bin/python
```

**Windows**:
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation
where python
# Should show: C:\path\to\mental-health-llm-evaluation\venv\Scripts\python.exe
```

#### Using conda (Alternative)

```bash
# Create conda environment
conda create -n mental-health-llm python=3.9

# Activate environment
conda activate mental-health-llm

# Verify activation
which python
conda info --envs
```

### 3. Install Dependencies

#### Basic Installation

```bash
# Upgrade pip
pip install --upgrade pip

# Install core dependencies
pip install -r requirements.txt
```

#### GPU Support (Optional)

For NVIDIA GPU acceleration:

```bash
# Install PyTorch with CUDA support
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Verify GPU detection
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
```

#### Development Dependencies

For development and testing:

```bash
# Install development dependencies
pip install -r requirements-dev.txt

# Install pre-commit hooks
pre-commit install
```

### 4. Verify Installation

```bash
# Run installation verification script
python scripts/verify_installation.py

# Expected output:
# ✓ Python version: 3.9.x
# ✓ Core dependencies installed
# ✓ GPU support: Available/Not Available
# ✓ Installation complete
```

## Configuration

### 1. Environment Variables

```bash
# Copy example environment file
cp .env.example .env

# Edit environment file with your settings
nano .env  # or use your preferred editor
```

### 2. Basic Configuration

Edit your `.env` file with the following settings:

```env
# General Configuration
PROJECT_NAME="Mental Health LLM Evaluation"
ENVIRONMENT="development"
LOG_LEVEL="INFO"
DEBUG=true

# Data Directories
DATA_DIR="./data"
RESULTS_DIR="./results"
LOGS_DIR="./logs"
CACHE_DIR="./cache"

# Database Configuration
DATABASE_URL="sqlite:///data/conversations.db"
DATABASE_POOL_SIZE=5
DATABASE_TIMEOUT=30

# API Configuration
OPENAI_API_KEY="your_openai_api_key_here"
OPENAI_ORG_ID="your_organization_id"
OPENAI_MODEL="gpt-4"
OPENAI_MAX_TOKENS=2048
OPENAI_TEMPERATURE=0.7

# DeepSeek Configuration
DEEPSEEK_MODEL_PATH="./models/deepseek-llm-7b-chat"
DEEPSEEK_DEVICE="auto"  # auto, cuda, cpu
DEEPSEEK_PRECISION="fp16"  # fp16, fp32, int8
DEEPSEEK_MAX_LENGTH=2048
DEEPSEEK_BATCH_SIZE=1

# Evaluation Configuration
ENABLE_SAFETY_MONITORING=true
ENABLE_METRICS_COLLECTION=true
BATCH_SIZE=32
MAX_CONCURRENT_REQUESTS=10
EVALUATION_TIMEOUT_SECONDS=300
MIN_CONVERSATION_TURNS=6
MAX_CONVERSATION_TURNS=20

# Redis Configuration (Optional)
REDIS_URL="redis://localhost:6379/0"
REDIS_PASSWORD=""
REDIS_TIMEOUT=30

# Security Configuration
SECRET_KEY="your_secret_key_here_change_in_production"
ALLOWED_HOSTS="localhost,127.0.0.1"

# Performance Monitoring
ENABLE_PERFORMANCE_MONITORING=true
METRICS_PORT=8080
```

### 3. Advanced Configuration

Create `config/config.yaml` for detailed settings:

```yaml
# config/config.yaml
project:
  name: "Mental Health LLM Evaluation"
  version: "1.0.0"
  description: "Comprehensive LLM evaluation framework"

models:
  openai:
    enabled: true
    default_model: "gpt-4"
    timeout: 30
    retry_attempts: 3
    rate_limit: 60  # requests per minute
    
  deepseek:
    enabled: true
    model_path: "./models/deepseek-llm-7b-chat"
    device_preference: ["cuda", "cpu"]
    memory_optimization: true
    quantization: "fp16"

evaluation:
  empathy:
    enabled: true
    weight: 0.3
    threshold: 7.0
    
  safety:
    enabled: true
    weight: 0.4
    crisis_threshold: 0.8
    
  coherence:
    enabled: true
    weight: 0.3
    threshold: 7.5

data_processing:
  conversation_batch_size: 10
  parallel_workers: 4
  memory_limit: "4GB"
  temp_directory: "./temp"

monitoring:
  log_level: "INFO"
  metrics_enabled: true
  alert_thresholds:
    error_rate: 0.05
    response_time: 5.0
    memory_usage: 0.8
```

### 4. Directory Structure Setup

```bash
# Create necessary directories
mkdir -p data/{scenarios,conversations,evaluations,results}
mkdir -p logs
mkdir -p cache
mkdir -p temp
mkdir -p models

# Set permissions (Linux/macOS)
chmod 755 data logs cache temp models
```

## Model Setup

### 1. OpenAI Setup

#### Get API Key
1. Visit [OpenAI API Keys](https://platform.openai.com/api-keys)
2. Create a new API key
3. Add to your `.env` file

#### Test OpenAI Connection

```python
# scripts/test_openai.py
import os
from src.models.openai_client import OpenAIClient

def test_openai_connection():
    """Test OpenAI API connection."""
    try:
        client = OpenAIClient()
        
        # Test basic request
        response = client.generate_response(
            prompt="Hello, this is a test message.",
            context={"test": True}
        )
        
        print("✓ OpenAI connection successful")
        print(f"Response: {response['content'][:100]}...")
        return True
        
    except Exception as e:
        print(f"✗ OpenAI connection failed: {e}")
        return False

if __name__ == "__main__":
    test_openai_connection()
```

Run the test:
```bash
python scripts/test_openai.py
```

### 2. DeepSeek Local Model Setup

#### Download Model

**Option 1: Hugging Face Hub**
```bash
# Install git-lfs if not already installed
git lfs install

# Clone model repository
cd models
git clone https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat
cd ..
```

**Option 2: Manual Download**
```bash
# Create model directory
mkdir -p models/deepseek-llm-7b-chat

# Download model files (example URLs)
cd models/deepseek-llm-7b-chat
wget https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat/resolve/main/config.json
wget https://huggingface.co/deepseek-ai/deepseek-llm-7b-chat/resolve/main/tokenizer.json
# ... download other necessary files
cd ../..
```

#### Test DeepSeek Model

```python
# scripts/test_deepseek.py
import os
from src.models.deepseek_client import DeepSeekClient

def test_deepseek_model():
    """Test DeepSeek local model."""
    try:
        # Initialize client
        client = DeepSeekClient(
            model_path=os.getenv("DEEPSEEK_MODEL_PATH"),
            device=os.getenv("DEEPSEEK_DEVICE", "auto")
        )
        
        # Test generation
        response = client.generate_response(
            prompt="Hello, this is a test message.",
            context={"test": True}
        )
        
        print("✓ DeepSeek model loaded successfully")
        print(f"Response: {response['content'][:100]}...")
        
        # Check resource usage
        usage = client.get_resource_usage()
        print(f"Memory usage: {usage.get('memory_mb', 'Unknown')} MB")
        print(f"GPU usage: {usage.get('gpu_utilization', 'Unknown')}%")
        
        return True
        
    except Exception as e:
        print(f"✗ DeepSeek model failed: {e}")
        return False

if __name__ == "__main__":
    test_deepseek_model()
```

Run the test:
```bash
python scripts/test_deepseek.py
```

### 3. Model Performance Optimization

#### GPU Memory Optimization

```python
# config/model_optimization.yaml
deepseek_optimization:
  memory_management:
    gradient_checkpointing: true
    cpu_offload: true
    max_memory_allocation: "6GB"
    
  quantization:
    enabled: true
    method: "int8"  # int8, fp16, bf16
    
  inference_optimization:
    torch_compile: true
    batch_size: 1
    sequence_length: 2048
```

#### CPU Optimization

```bash
# Set CPU optimization environment variables
export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4
export TOKENIZERS_PARALLELISM=false
```

## Running the Framework

### 1. Basic Usage

#### Generate Sample Conversations

```bash
# Run basic conversation generation
python scripts/generate_sample_conversations.py \
  --models openai \
  --scenarios data/scenarios/anxiety_disorders.yaml \
  --output data/conversations/ \
  --num-conversations 5
```

#### Run Evaluation

```bash
# Evaluate generated conversations
python scripts/run_evaluation.py \
  --input data/conversations/ \
  --output results/evaluation_results.json \
  --metrics empathy,safety,coherence
```

#### Generate Analysis Report

```bash
# Create comprehensive analysis report
python scripts/generate_report.py \
  --evaluation-results results/evaluation_results.json \
  --output results/analysis_report.html \
  --include-visualizations
```

### 2. Interactive Usage

#### Start Jupyter Server

```bash
# Install Jupyter if not already installed
pip install jupyter

# Start Jupyter server
jupyter notebook

# Open the example notebook
# Navigate to: notebooks/01_getting_started.ipynb
```

#### Python REPL Example

```python
# Start Python REPL
python

# Interactive example
from src.models.openai_client import OpenAIClient
from src.scenarios.scenario_loader import ScenarioLoader
from src.evaluation.composite_scorer import CompositeScorer

# Initialize components
openai_client = OpenAIClient()
scenario_loader = ScenarioLoader()
scorer = CompositeScorer()

# Load a scenario
scenarios = scenario_loader.load_all_scenarios()
test_scenario = scenarios[0]

# Generate conversation
conversation = openai_client.generate_conversation(
    scenario=test_scenario,
    conversation_id="interactive_test"
)

# Evaluate conversation
score = scorer.calculate_composite_score(
    conversation, 
    test_scenario.scenario_id
)

print(f"Overall Score: {score.overall_score:.2f}")
```

### 3. Command-Line Interface

#### Main CLI Commands

```bash
# Show available commands
python -m src.cli --help

# Run complete evaluation pipeline
python -m src.cli evaluate \
  --config config/config.yaml \
  --models openai,deepseek \
  --scenarios data/scenarios/ \
  --output results/ \
  --parallel

# Generate comparison report
python -m src.cli compare \
  --results results/openai_results.json results/deepseek_results.json \
  --output results/comparison_report.html

# Run safety testing
python -m src.cli safety-test \
  --model openai \
  --test-cases data/safety_test_cases.json \
  --output results/safety_results.json
```

### 4. Web Interface (Optional)

#### Start Development Server

```bash
# Install web dependencies
pip install -r requirements-web.txt

# Start development server
python -m src.web.app --debug --port 8000

# Access web interface at: http://localhost:8000
```

#### Web Interface Features
- Interactive conversation generation
- Real-time evaluation monitoring
- Visualization dashboards
- Model comparison tools
- Safety monitoring alerts

## Development Workflow

### 1. Code Quality

#### Run Tests

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test categories
python -m pytest tests/unit/ -v
python -m pytest tests/integration/ -v
python -m pytest tests/safety/ -v

# Run tests with coverage
python -m pytest tests/ --cov=src --cov-report=html
```

#### Code Formatting

```bash
# Format code with black
black src/ tests/ scripts/

# Sort imports
isort src/ tests/ scripts/

# Check code style
flake8 src/ tests/ scripts/

# Type checking
mypy src/
```

#### Pre-commit Hooks

```bash
# Install pre-commit hooks
pre-commit install

# Run hooks manually
pre-commit run --all-files
```

### 2. Database Management

#### Initialize Database

```bash
# Create database tables
python scripts/init_database.py

# Verify database setup
python scripts/verify_database.py
```

#### Database Migrations

```bash
# Create migration
python scripts/create_migration.py --name "add_new_evaluation_metrics"

# Apply migrations
python scripts/migrate_database.py

# Rollback migration
python scripts/migrate_database.py --rollback
```

### 3. Data Management

#### Load Test Data

```bash
# Load sample scenarios
python scripts/load_sample_data.py --scenarios

# Load evaluation test cases
python scripts/load_sample_data.py --test-cases

# Generate synthetic conversations
python scripts/generate_synthetic_data.py --count 100
```

#### Data Validation

```bash
# Validate scenario files
python scripts/validate_scenarios.py data/scenarios/

# Check data integrity
python scripts/check_data_integrity.py

# Clean up temporary files
python scripts/cleanup_data.py
```

## Troubleshooting

### Common Issues

#### 1. Import Errors

**Problem**: `ModuleNotFoundError: No module named 'src'`

**Solution**:
```bash
# Add project root to Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)"

# Or install in development mode
pip install -e .
```

#### 2. GPU Memory Issues

**Problem**: CUDA out of memory errors

**Solutions**:
```bash
# Reduce batch size
export DEEPSEEK_BATCH_SIZE=1

# Enable memory optimization
export PYTORCH_CUDA_ALLOC_CONF=max_split_size_mb:128

# Clear GPU cache
python -c "import torch; torch.cuda.empty_cache()"
```

#### 3. Model Loading Issues

**Problem**: DeepSeek model fails to load

**Solutions**:
```bash
# Check model files
ls -la models/deepseek-llm-7b-chat/

# Verify model path in config
python -c "import os; print(os.getenv('DEEPSEEK_MODEL_PATH'))"

# Test with CPU only
export DEEPSEEK_DEVICE=cpu
```

#### 4. Database Connection Issues

**Problem**: Database connection errors

**Solutions**:
```bash
# Check database file permissions
ls -la data/conversations.db

# Recreate database
rm data/conversations.db
python scripts/init_database.py

# Check SQLite installation
python -c "import sqlite3; print(sqlite3.version)"
```

### Debug Mode

#### Enable Debug Logging

```bash
# Set debug environment variable
export DEBUG=true
export LOG_LEVEL=DEBUG

# Run with verbose output
python scripts/generate_conversations.py --verbose --debug
```

#### Debug Configuration

```python
# config/debug_config.yaml
debug:
  enabled: true
  log_level: "DEBUG"
  print_sql_queries: true
  save_intermediate_results: true
  detailed_error_messages: true
  
  model_debugging:
    log_prompts: true
    log_responses: true
    save_conversation_states: true
```

### Performance Monitoring

#### Monitor Resource Usage

```bash
# Install monitoring tools
pip install psutil nvidia-ml-py3

# Run resource monitor
python scripts/monitor_resources.py --interval 5 --duration 3600

# Generate performance report
python scripts/performance_report.py --input logs/resource_usage.log
```

#### Profile Performance

```bash
# Profile code execution
python -m cProfile -o profile_stats scripts/run_evaluation.py

# Analyze profile results
python scripts/analyze_profile.py --profile profile_stats
```

## Performance Optimization

### 1. System Optimization

#### Memory Management

```bash
# Increase swap space (Linux)
sudo fallocate -l 4G /swapfile
sudo chmod 600 /swapfile
sudo mkswap /swapfile
sudo swapon /swapfile

# Optimize Python memory usage
export PYTHONHASHSEED=0
export PYTHONMALLOC=malloc
```

#### CPU Optimization

```bash
# Set CPU governor to performance mode (Linux)
echo performance | sudo tee /sys/devices/system/cpu/cpu*/cpufreq/scaling_governor

# Optimize NumPy threading
export OPENBLAS_NUM_THREADS=4
export MKL_NUM_THREADS=4
```

### 2. Model Optimization

#### Quantization

```python
# scripts/optimize_model.py
from src.models.model_optimizer import ModelOptimizer

optimizer = ModelOptimizer()

# Quantize model to int8
optimized_model = optimizer.quantize_model(
    model_path="models/deepseek-llm-7b-chat",
    quantization_method="int8",
    output_path="models/deepseek-llm-7b-chat-int8"
)

# Benchmark performance
benchmark_results = optimizer.benchmark_model(optimized_model)
print(f"Inference speed improvement: {benchmark_results['speedup']:.2f}x")
print(f"Memory reduction: {benchmark_results['memory_reduction']:.1f}%")
```

#### Caching

```python
# Enable response caching
cache_config = {
    'enabled': True,
    'backend': 'redis',  # or 'memory', 'disk'
    'ttl': 3600,  # 1 hour
    'max_size': '1GB'
}
```

### 3. Parallel Processing

#### Multi-threading

```bash
# Run with multiple workers
python scripts/batch_evaluation.py --workers 4 --parallel-models
```

#### GPU Optimization

```bash
# Optimize GPU settings
export CUDA_LAUNCH_BLOCKING=0
export CUDA_CACHE_DISABLE=0
export TORCH_CUDNN_V8_API_ENABLED=1
```

## Next Steps

After successful local deployment:

1. **Explore Examples**: Check `notebooks/` for guided tutorials
2. **Run Tests**: Verify everything works with the test suite
3. **Generate Data**: Create sample conversations and evaluations
4. **Customize Configuration**: Adapt settings for your use case
5. **Scale Up**: Consider cloud deployment for larger experiments

For production deployment, see:
- [Cloud Deployment Guide](cloud.md)
- [Docker Deployment Guide](docker.md)
- [Performance Tuning Guide](performance.md)

---

**Need Help?**
- Check [Troubleshooting Guide](../troubleshooting.md)
- Review [FAQ](../faq.md)
- Open an issue on [GitHub](https://github.com/your-username/mental-health-llm-evaluation/issues)