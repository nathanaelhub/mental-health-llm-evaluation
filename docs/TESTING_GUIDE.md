# 🧪 Mental Health LLM Evaluation - Simple Testing Guide

This guide covers the three main scripts for testing and running the mental health LLM evaluation system.

## 📋 Prerequisites

```bash
# Navigate to project directory
cd /home/nathanael/mental-health-llm-evaluation

# Activate virtual environment
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env
# Edit .env file with your API keys and local server settings
```

## 🚀 Main Scripts

### 1. `scripts/setup_models.py` - Model Setup and Configuration

**Purpose:** Set up and configure model clients for first-time use

```bash
# Initial model setup
python scripts/setup_models.py

# Check model configuration
python scripts/setup_models.py --check
```

**What it does:**
- Verifies API keys and connection settings
- Tests local model server connections
- Validates model configurations
- Provides setup recommendations

---

### 2. `test_all_models.py` - Model Connection Testing

**Purpose:** Test all model clients to ensure they work correctly

```bash
# Test all configured models
python test_all_models.py

# Expected output:
# ✓ OpenAI client created successfully
# ✓ Claude client created successfully  
# ✓ DeepSeek client created successfully
# ✓ Gemma client created successfully
```

**What it tests:**
- Model client initialization
- Basic response generation
- Cost tracking functionality
- Error handling

---

### 3. `scripts/compare_models.py` - Model Comparison Tool

**Purpose:** Compare responses between different models side-by-side

#### Basic Usage:
```bash
# Compare two models with a simple prompt
python scripts/compare_models.py --models openai,claude "What are coping strategies for anxiety?"

# Compare local models only
python scripts/compare_models.py --models deepseek,gemma "How can I manage work stress?"

# Compare all available models
python scripts/compare_models.py --all "I'm feeling overwhelmed"
```

#### Advanced Usage:
```bash
# Verbose output with detailed metrics
python scripts/compare_models.py --models openai,deepseek --verbose "I'm feeling overwhelmed with work stress"

# Interactive mode for multiple queries
python scripts/compare_models.py --interactive

# Save results to file
python scripts/compare_models.py --models openai,claude --save results.json "How do I handle anxiety?"

# Batch testing with multiple prompts
python scripts/compare_models.py --batch prompts.txt --models openai,deepseek
```

#### Available Models:
- `openai` - OpenAI GPT-4 (requires API key)
- `claude` - Anthropic Claude (requires API key)
- `deepseek` - DeepSeek local model (requires local server)
- `gemma` - Gemma local model (requires local server)

---

### 4. `scripts/run_research.py` - Full Research Pipeline

**Purpose:** Run comprehensive mental health LLM evaluation research

#### Quick Test:
```bash
# Run evaluation with 3 scenarios for testing
python scripts/run_research.py --limit 3 --output test_results/

# Check generated files
ls -la test_results/
```

#### Full Research Run:
```bash
# Run complete evaluation with all scenarios
python scripts/run_research.py --output research_results/

# Run with specific models only
python scripts/run_research.py --models openai,claude --output research_results/

# Run with all available models
python scripts/run_research.py --models all --output research_results/
```

#### Output Files:
- `detailed_results.json` - Raw evaluation data
- `statistical_analysis.json` - Statistical comparisons
- `research_report.txt` - Human-readable summary
- `visualizations/` - Charts and graphs
- `presentation/` - Presentation slides

---

## 🔧 Configuration

### Environment Variables (`.env`)
```bash
# Cloud API Keys
OPENAI_API_KEY=your_openai_key_here
ANTHROPIC_API_KEY=your_anthropic_key_here

# Local LLM Configuration (LM Studio)
LOCAL_LLM_BASE_URL=http://192.168.86.23:1234/v1
LOCAL_LLM_MODEL_DEEPSEEK=deepseek/deepseek-r1-0528-qwen3-8b
LOCAL_LLM_MODEL_GEMMA=google/gemma-3-12b
LOCAL_LLM_TIMEOUT=60
```

### Model Configuration (`config/main.yaml`)
```yaml
# Enable/disable models
enabled_models: ["openai", "claude", "deepseek", "gemma"]

# Evaluation weights
evaluation:
  weights:
    empathy: 0.30
    therapeutic: 0.25
    safety: 0.35
    clarity: 0.10
```

---

## 📊 Quick Testing Workflow

### 1. System Health Check
```bash
# Test all models
python test_all_models.py

# Quick comparison test
python scripts/compare_models.py --models openai,deepseek "Hello, how are you?"
```

### 2. Sample Research Run
```bash
# Run mini research with 3 scenarios
python scripts/run_research.py --limit 3 --output sample_results/

# Check if files were generated
ls -la sample_results/
```

### 3. Full Research Pipeline
```bash
# Run complete evaluation
python scripts/run_research.py --output final_results/

# Generate additional visualizations if needed
python -c "
from src.analysis.visualization import create_all_visualizations
import json
with open('final_results/detailed_results.json') as f:
    data = json.load(f)
create_all_visualizations(data, 'final_results/visualizations/')
"
```

---

## 🛠️ Troubleshooting

### Common Issues:

#### 1. Model Connection Errors
```bash
# Check API keys
python -c "import os; print('OpenAI:', bool(os.getenv('OPENAI_API_KEY')))"
python -c "import os; print('Claude:', bool(os.getenv('ANTHROPIC_API_KEY')))"

# Test local server connection
curl http://192.168.86.23:1234/v1/models
```

#### 2. Import Errors
```bash
# Reinstall dependencies
pip install -r requirements.txt --force-reinstall

# Check Python path
python -c "import sys; print(sys.path)"
```

#### 3. Local Model Issues
```bash
# Check if local models are running
python -c "
from src.models.deepseek_client import DeepSeekClient
try:
    client = DeepSeekClient()
    print('DeepSeek: OK')
except Exception as e:
    print(f'DeepSeek: ERROR - {e}')
"
```

#### 4. Permission Issues
```bash
# Make scripts executable
chmod +x scripts/*.py
chmod +x test_all_models.py
```

---

## ✅ Success Criteria

After running the tests, you should have:

- [ ] All models in `test_all_models.py` pass (or fail gracefully with clear error messages)
- [ ] `compare_models.py` can compare available models successfully
- [ ] `run_research.py` generates all expected output files
- [ ] Visualizations are created in the output directory
- [ ] Research report is readable and contains meaningful results

---

## 🎯 Recommended Testing Sequence

1. **Initial setup:** `python scripts/setup_models.py`
2. **Test all models:** `python test_all_models.py`
3. **Test comparison tool:** `python scripts/compare_models.py --models openai,deepseek "Hello"`
4. **Run quick research:** `python scripts/run_research.py --limit 3 --output test/`
5. **Run full research:** `python scripts/run_research.py --output results/`

This simplified guide focuses on the core functionality you need for your mental health LLM evaluation research! 🧠✨