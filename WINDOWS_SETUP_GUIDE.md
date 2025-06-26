# Windows Setup Guide for Mental Health LLM Evaluation

## 🖥️ Pre-Setup Checklist
- [ ] Project downloaded/cloned from GitHub
- [ ] Python 3.8+ installed on Windows
- [ ] DeepSeek or other local LLM already set up
- [ ] VS Code or preferred IDE installed
- [ ] Git installed (if cloned)
- [ ] CUDA drivers installed (for GPU acceleration)

## 📁 Step 1: Project Structure Verification
After extracting/cloning, verify you have:
```
mental-health-llm-evaluation/
├── src/
│   ├── models/
│   ├── evaluation/
│   ├── scenarios/
│   ├── analysis/
│   └── utils/
├── data/
├── config/
├── scripts/
├── templates/
├── tests/
├── requirements.txt
├── .env.example
└── README.md
```

## 🔧 Step 2: Environment Setup

### A. Open Command Prompt as Administrator
```cmd
# Navigate to project directory
cd C:\path\to\mental-health-llm-evaluation
```

### B. Create Virtual Environment
```cmd
# Create virtual environment
python -m venv venv

# Activate virtual environment
venv\Scripts\activate

# Verify activation (should show (venv) in prompt)
where python
```

### C. Install Dependencies
```cmd
# Upgrade pip first
python -m pip install --upgrade pip

# Install requirements
pip install -r requirements.txt

# For GPU support (if you have CUDA-capable GPU)
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

### D. Create Environment File
```cmd
copy .env.example .env
```

## 🔑 Step 3: API Keys and Configuration

### A. Edit `.env` file (in project root):
Open `.env` in your text editor and configure:

```env
# OpenAI API Configuration
OPENAI_API_KEY=sk-your-openai-api-key-here
OPENAI_ORG_ID=your-org-id-here

# Local Model Paths (Windows-specific paths)
DEEPSEEK_MODEL_PATH=C:\Users\YourUsername\models\deepseek-llm-7b-chat
LLAMA_MODEL_PATH=C:\Users\YourUsername\models\llama-2-7b-chat-hf

# Windows-specific paths using forward slashes
DATA_DIR=./data
LOGS_DIR=./logs
RESULTS_DIR=./results
EXPERIMENTS_DIR=./experiments

# GPU Configuration
CUDA_VISIBLE_DEVICES=0
ENABLE_GPU=true

# Logging
LOG_LEVEL=INFO
LOG_FILE=./logs/evaluation.log

# Processing Configuration
BATCH_SIZE=8
MAX_CONCURRENT_REQUESTS=3
TIMEOUT_SECONDS=300
```

### B. Update `config/experiment_template.yaml`:
Modify local model paths for Windows:
```yaml
experiment:
  name: "Mental Health LLM Evaluation - Windows"
  description: "Windows deployment of LLM evaluation framework"
  
models:
  cloud:
    - name: "gpt-4"
      provider: "openai"
      enabled: true
      model: "gpt-4-turbo-preview"
      temperature: 0.7
      max_tokens: 1000
      timeout: 30.0
      
  local:
    - name: "deepseek"
      provider: "deepseek"
      enabled: true
      model_path: "C:/Users/YourUsername/models/deepseek-llm-7b-chat"
      device: "auto"  # Will use GPU if available
      precision: "fp16"
      temperature: 0.7
      max_tokens: 1000
      load_in_8bit: false
      load_in_4bit: false

evaluation:
  conversations_per_scenario: 10
  max_conversation_turns: 15
  enable_safety_monitoring: true
  timeout_seconds: 300

output:
  base_directory: "./experiments"
  conversations: "conversations"
  evaluations: "evaluations"
  results: "results"
  reports: "reports"
```

## 🛠️ Step 4: Critical Path Fixes for Windows

### A. Files That MUST Be Updated:

#### 1. **`src/utils/config_manager.py`** - Fix path handling:
```python
import os
from pathlib import Path

class ConfigManager:
    def __init__(self):
        self.project_root = Path(__file__).parent.parent.parent
        self.config_dir = self.project_root / "config"
        self.data_dir = self.project_root / "data"
        
    def get_data_path(self, *args):
        """Get platform-independent data path"""
        return self.data_dir.joinpath(*args)
        
    def get_results_path(self, *args):
        """Get platform-independent results path"""
        return self.project_root.joinpath("results", *args)
```

#### 2. **`src/utils/logging_config.py`** - Fix log file paths:
```python
import os
from pathlib import Path

def setup_logging():
    project_root = Path(__file__).parent.parent.parent
    log_dir = project_root / "logs"
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / "evaluation.log"
    
    # Rest of logging configuration...
```

#### 3. **`src/models/deepseek_client.py`** - Fix model loading:
```python
import os
from pathlib import Path

class DeepSeekClient(BaseModel):
    def __init__(self, config=None):
        # Fix model path for Windows
        if config and "model_path" in config:
            model_path = Path(config["model_path"])
            if not model_path.exists():
                # Try relative path from project root
                project_root = Path(__file__).parent.parent.parent
                model_path = project_root / "models" / "deepseek-llm-7b-chat"
            config["model_path"] = str(model_path)
```

#### 4. **`scripts/setup_experiment.py`** - Fix directory creation:
```python
import os
from pathlib import Path

def create_experiment_directories(base_path):
    """Create experiment directories with Windows-compatible paths"""
    base_path = Path(base_path)
    
    directories = [
        base_path / "conversations",
        base_path / "evaluations", 
        base_path / "results",
        base_path / "reports",
        base_path / "logs"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created: {directory}")
```

### B. Add to PYTHONPATH (Critical for imports):
Create `setup_windows.bat` in project root:
```batch
@echo off
set PYTHONPATH=%PYTHONPATH%;%CD%
set PYTHONPATH=%PYTHONPATH%;%CD%\src
echo Python path configured for Windows
echo PYTHONPATH=%PYTHONPATH%
```

Run this before using the project:
```cmd
setup_windows.bat
```

## 🧪 Step 5: Test Installations

### A. Test Python Environment:
```cmd
# Activate environment
venv\Scripts\activate

# Test basic imports
python -c "import torch; print(f'PyTorch: {torch.__version__}')"
python -c "import transformers; print(f'Transformers: {transformers.__version__}')"
python -c "import openai; print('OpenAI library installed')"
```

### B. Test Project Imports:
```cmd
# Run from project root
setup_windows.bat
python -c "from src.models.base_model import BaseModel; print('Base model imported successfully')"
python -c "from src.models.openai_client import OpenAIClient; print('OpenAI client imported successfully')"
```

### C. Test Model Connections:
```cmd
# Test OpenAI connection
python -c "
from src.models.openai_client import OpenAIClient
client = OpenAIClient()
print('OpenAI client created successfully')
print(f'Model validation: {client.validate_configuration()}')
"

# Test local model (if DeepSeek is installed)
python -c "
from src.models.deepseek_client import DeepSeekClient
client = DeepSeekClient()
print('DeepSeek client created successfully')
"
```

## ▶️ Step 6: Running the Project

### A. First-Time Setup:
```cmd
# Always run this first
setup_windows.bat
venv\Scripts\activate

# Verify model registry
python scripts/model_management.py list

# Test configuration
python scripts/setup_experiment.py --dry-run
```

### B. Quick Test Run (5 minutes):
```cmd
# Generate 1 conversation per model for testing
python scripts/run_conversations.py --models gpt-4 --scenarios 1 --conversations 1

# Check if files were created
dir data\conversations\
```

### C. Small Evaluation Run (15 minutes):
```cmd
# Generate 5 conversations with 2 models
python scripts/run_conversations.py --models gpt-4,deepseek --scenarios 3 --conversations 5

# Evaluate the conversations
python scripts/evaluate_conversations.py --experiment latest

# Generate analysis
python scripts/analyze_results.py --experiment latest

# Create report
python scripts/generate_report.py --experiment latest --format html
```

### D. Full Evaluation Run (2+ hours):
```cmd
# Step 1: Setup full experiment
python scripts/setup_experiment.py --config config/experiment_template.yaml

# Step 2: Generate all conversations
python scripts/run_conversations.py --config config/experiment_template.yaml

# Step 3: Run all evaluations
python scripts/evaluate_conversations.py --experiment latest

# Step 4: Statistical analysis
python scripts/analyze_results.py --experiment latest

# Step 5: Generate comprehensive report
python scripts/generate_report.py --experiment latest --format html,pdf
```

## 🚨 Troubleshooting Common Windows Issues

### Issue 1: "Module not found" Errors
```cmd
# Solution 1: Set PYTHONPATH
set PYTHONPATH=%PYTHONPATH%;%CD%
set PYTHONPATH=%PYTHONPATH%;%CD%\src

# Solution 2: Install in development mode
pip install -e .

# Solution 3: Run from project root
cd C:\path\to\mental-health-llm-evaluation
python scripts/your_script.py
```

### Issue 2: Permission Errors
```cmd
# Run CMD as Administrator
# Right-click Command Prompt -> "Run as administrator"

# Or set permissions for your user
icacls "C:\path\to\project" /grant %USERNAME%:F /T
```

### Issue 3: Path Length Limits (Windows 260 character limit)
```cmd
# Enable long paths in Windows (requires admin)
reg add "HKLM\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

# Or move project closer to root
move C:\very\long\path\mental-health-llm-evaluation C:\ml-eval
```

### Issue 4: CUDA/GPU Issues
```cmd
# Check CUDA installation
nvcc --version

# Check GPU memory
nvidia-smi

# Test PyTorch CUDA
python -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}')"
python -c "import torch; print(f'CUDA devices: {torch.cuda.device_count()}')"

# If CUDA issues, install CPU-only version
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### Issue 5: OpenAI API Issues
```cmd
# Test API key
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
print('API Key loaded:', openai.api_key[:8] + '...' if openai.api_key else 'None')
"

# Test API connection
python -c "
import openai
import os
openai.api_key = os.getenv('OPENAI_API_KEY')
try:
    models = openai.Model.list()
    print('API connection successful')
except Exception as e:
    print(f'API error: {e}')
"
```

### Issue 6: Memory Issues (Local Models)
```cmd
# Check available memory
wmic OS get TotalVisibleMemorySize /value

# For memory issues, use smaller models or quantization
# Edit config/experiment_template.yaml:
# load_in_8bit: true
# load_in_4bit: true
```

### Issue 7: Antivirus/Windows Defender Issues
```cmd
# Add exclusions in Windows Defender:
# 1. Go to Windows Security
# 2. Virus & threat protection
# 3. Exclusions
# 4. Add: C:\path\to\mental-health-llm-evaluation
# 5. Add: C:\path\to\venv
```

## 📊 Step 7: Verify Everything Works

### A. Check Data Generation:
```cmd
# Look for conversation files
dir data\conversations\
type data\conversations\gpt-4_scenario_001_conv_001.json

# Check file structure
python -c "
import json
with open('data/conversations/gpt-4_scenario_001_conv_001.json', 'r') as f:
    data = json.load(f)
    print('Conversation keys:', list(data.keys()))
"
```

### B. Check Evaluation:
```cmd
# Look for evaluation files
dir data\evaluations\
type data\evaluations\empathy_scores.json

# Check evaluation structure
python -c "
import json
import glob
files = glob.glob('data/evaluations/*.json')
print('Evaluation files:', files)
"
```

### C. Check Analysis:
```cmd
# Look for results
dir data\results\
dir experiments\

# Check report generation
# Should create HTML file you can open in browser
start experiments\latest\reports\evaluation_report.html
```

## 🔄 Step 8: Daily Development Workflow

### A. Starting Your Work Session:
```cmd
# 1. Navigate to project
cd C:\path\to\mental-health-llm-evaluation

# 2. Activate environment
venv\Scripts\activate

# 3. Set Python path
setup_windows.bat

# 4. Verify everything works
python scripts/model_management.py list
```

### B. Running Experiments:
```cmd
# Quick test (development)
python scripts/run_conversations.py --models gpt-4 --scenarios 2 --conversations 3

# Full run (production)
python scripts/run_conversations.py --config config/experiment_template.yaml
```

### C. Analyzing Results:
```cmd
# Generate analysis
python scripts/analyze_results.py --experiment latest

# Create report
python scripts/generate_report.py --experiment latest --format html

# Open results
start experiments\latest\reports\evaluation_report.html
```

## 📝 Windows-Specific Performance Tips

### 1. **File System Optimization**
```cmd
# Disable Windows indexing for project folder
# Right-click folder -> Properties -> Advanced -> Uncheck "Allow files in this folder to have contents indexed"
```

### 2. **Memory Management**
```cmd
# Monitor memory usage
tasklist /fi "imagename eq python.exe"

# For large models, close other applications
# Consider using swap file on SSD
```

### 3. **GPU Optimization**
```cmd
# Monitor GPU usage
nvidia-smi -l 1

# Set GPU memory growth (add to model config)
# gpu_memory_growth: true
```

### 4. **Network Optimization (for API calls)**
```cmd
# Flush DNS cache
ipconfig /flushdns

# Check network connectivity
ping api.openai.com
```

## 🗂️ File Locations After Setup

After successful setup, key files will be at:
```
C:\your-project\mental-health-llm-evaluation\
├── data\
│   ├── conversations\        # Generated conversations
│   ├── evaluations\         # Evaluation scores
│   └── results\            # Analysis results
├── experiments\
│   └── latest\
│       ├── conversations\
│       ├── evaluations\
│       ├── results\
│       └── reports\        # HTML/PDF reports here
├── logs\                   # Log files
└── venv\                   # Python virtual environment
```

## 🔍 Verification Checklist

- [ ] Virtual environment activates without errors
- [ ] All Python packages install successfully
- [ ] Project imports work (`from src.models...`)
- [ ] OpenAI API key validates
- [ ] Local model loads (if using DeepSeek)
- [ ] Can generate at least 1 conversation
- [ ] Evaluation scripts run without errors
- [ ] HTML report generates and opens
- [ ] All file paths work with Windows format

## 🚨 Emergency Fixes

### If Everything Breaks:
```cmd
# Nuclear option - start fresh
rmdir /s venv
python -m venv venv
venv\Scripts\activate
pip install -r requirements.txt
```

### If Models Won't Load:
```cmd
# Check model files exist
dir "C:\Users\YourUsername\models\"

# Test with CPU-only mode
# Edit .env: ENABLE_GPU=false
```

### If API Calls Fail:
```cmd
# Test internet connection
ping 8.8.8.8

# Test OpenAI specifically  
curl https://api.openai.com/v1/models -H "Authorization: Bearer %OPENAI_API_KEY%"
```

## 📞 Getting Help

If you encounter issues:

1. **Check the logs**: `type logs\evaluation.log`
2. **Run with verbose output**: `python scripts/your_script.py --verbose`
3. **Test individual components**: Use the test commands in Step 5
4. **Check GitHub issues**: Look for Windows-specific problems
5. **Contact support**: Include full error message and Windows version

---

## 🗑️ Cleanup After Setup

Once everything is working, you can delete this file:
```cmd
del WINDOWS_SETUP_GUIDE.md
```

---

**Windows Version Tested:** Windows 10/11  
**Python Version:** 3.8-3.11  
**Last Updated:** June 2024

*This guide assumes you have DeepSeek or another local LLM already installed and working on your Windows machine. If not, refer to the model-specific installation guides in the templates/ directory.*