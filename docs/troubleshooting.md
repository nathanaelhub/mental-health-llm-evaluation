# Troubleshooting Guide

This guide provides solutions to common issues encountered when using the Mental Health LLM Evaluation framework.

## Table of Contents

- [Installation Issues](#installation-issues)
- [Configuration Problems](#configuration-problems)
- [Model-Specific Issues](#model-specific-issues)
- [Evaluation Errors](#evaluation-errors)
- [Performance Issues](#performance-issues)
- [Database Problems](#database-problems)
- [Network and API Issues](#network-and-api-issues)
- [Memory and Resource Issues](#memory-and-resource-issues)
- [Platform-Specific Issues](#platform-specific-issues)
- [Advanced Debugging](#advanced-debugging)

## Installation Issues

### Issue: Python Version Compatibility

**Symptoms**:
- `SyntaxError` when importing modules
- Package installation failures
- Type hint errors

**Cause**: Using Python version < 3.8

**Solution**:
```bash
# Check Python version
python --version

# If using Python < 3.8, upgrade:
# macOS with Homebrew
brew install python@3.9

# Ubuntu/Debian
sudo apt update
sudo apt install python3.9 python3.9-venv

# Windows - Download from python.org

# Create new virtual environment with correct Python
python3.9 -m venv venv
source venv/bin/activate  # Linux/macOS
# or
venv\Scripts\activate  # Windows
```

### Issue: pip Installation Failures

**Symptoms**:
- `ERROR: Could not install packages due to an EnvironmentError`
- Permission denied errors
- Network timeout errors

**Solutions**:

```bash
# Update pip first
python -m pip install --upgrade pip

# Install with user flag if permission issues
pip install --user -r requirements.txt

# Use different index if network issues
pip install -r requirements.txt -i https://pypi.org/simple/

# Clear pip cache if corrupted
pip cache purge
```

### Issue: Missing System Dependencies

**Symptoms**:
- `ModuleNotFoundError` for system libraries
- Compilation errors during package installation

**Solutions**:

**Ubuntu/Debian**:
```bash
sudo apt update
sudo apt install python3-dev python3-pip python3-venv
sudo apt install build-essential libssl-dev libffi-dev
sudo apt install sqlite3 libsqlite3-dev
```

**macOS**:
```bash
# Install Xcode command line tools
xcode-select --install

# Install Homebrew if not installed
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Install dependencies
brew install python sqlite
```

**Windows**:
```cmd
# Install Microsoft C++ Build Tools
# Download from: https://visualstudio.microsoft.com/visual-cpp-build-tools/

# Or install Visual Studio Community with C++ workload
```

### Issue: Virtual Environment Problems

**Symptoms**:
- `command not found: python` in virtual environment
- Wrong Python version in venv
- Packages not found despite installation

**Solutions**:

```bash
# Recreate virtual environment
deactivate  # if currently activated
rm -rf venv
python3 -m venv venv
source venv/bin/activate

# Verify correct Python
which python
python --version

# Reinstall packages
pip install -r requirements.txt

# For conda environments
conda deactivate
conda remove --name mental-health-llm --all
conda create -n mental-health-llm python=3.9
conda activate mental-health-llm
```

## Configuration Problems

### Issue: Environment Variables Not Loading

**Symptoms**:
- `KeyError` for environment variables
- Configuration using default values
- API authentication failures

**Solutions**:

```bash
# Check if .env file exists
ls -la .env

# Copy from example if missing
cp .env.example .env

# Verify environment variables are loaded
python -c "import os; print(os.getenv('OPENAI_API_KEY', 'NOT_SET'))"

# Load environment manually if needed
export $(cat .env | xargs)

# Check for hidden characters in .env
cat -A .env
```

### Issue: Configuration File Validation Errors

**Symptoms**:
- YAML parsing errors
- JSON validation failures
- Schema validation errors

**Solutions**:

```bash
# Validate YAML syntax
python -c "import yaml; yaml.safe_load(open('config/config.yaml'))"

# Validate JSON syntax
python -c "import json; json.load(open('config/settings.json'))"

# Use online validators:
# YAML: https://yaml-online-parser.appspot.com/
# JSON: https://jsonlint.com/

# Check for common issues:
# - Incorrect indentation in YAML
# - Missing quotes in JSON strings
# - Trailing commas in JSON
# - Special characters not escaped
```

### Issue: Path Resolution Problems

**Symptoms**:
- `FileNotFoundError` for configuration files
- Relative path issues
- Model files not found

**Solutions**:

```python
# Debug path resolution
import os
from pathlib import Path

print(f"Current working directory: {os.getcwd()}")
print(f"Script directory: {Path(__file__).parent}")
print(f"Project root: {Path(__file__).parent.parent}")

# Use absolute paths in configuration
DATA_DIR = os.path.abspath("./data")
MODEL_PATH = os.path.abspath("./models/deepseek-llm-7b-chat")

# Verify paths exist
assert os.path.exists(DATA_DIR), f"Data directory not found: {DATA_DIR}"
assert os.path.exists(MODEL_PATH), f"Model path not found: {MODEL_PATH}"
```

## Model-Specific Issues

### Issue: OpenAI API Authentication

**Symptoms**:
- `openai.AuthenticationError`
- `Invalid API key` messages
- Rate limit errors

**Solutions**:

```bash
# Verify API key format (should start with 'sk-')
echo $OPENAI_API_KEY | head -c 10

# Test API key directly
curl https://api.openai.com/v1/models \
  -H "Authorization: Bearer $OPENAI_API_KEY"

# Check organization ID if using one
python -c "
import openai
openai.api_key = 'your-api-key'
openai.organization = 'your-org-id'
print(openai.Model.list())
"

# Handle rate limits
python -c "
import time
import openai
from openai import OpenAI

client = OpenAI()
try:
    response = client.chat.completions.create(
        model='gpt-3.5-turbo',
        messages=[{'role': 'user', 'content': 'test'}],
        max_tokens=10
    )
    print('Success!')
except openai.RateLimitError as e:
    print(f'Rate limit error: {e}')
    print('Wait and retry, or check your plan limits')
"
```

### Issue: DeepSeek Model Loading

**Symptoms**:
- `OSError: Unable to load model`
- CUDA out of memory
- Model file corruption

**Solutions**:

```python
# Verify model files exist
import os
from pathlib import Path

model_path = Path("models/deepseek-llm-7b-chat")
required_files = ["config.json", "pytorch_model.bin", "tokenizer.json"]

for file in required_files:
    file_path = model_path / file
    if file_path.exists():
        print(f"✓ {file}: {file_path.stat().st_size} bytes")
    else:
        print(f"✗ {file}: Missing")

# Test with minimal configuration
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    tokenizer = AutoTokenizer.from_pretrained(str(model_path))
    print("✓ Tokenizer loaded")
    
    model = AutoModelForCausalLM.from_pretrained(
        str(model_path),
        torch_dtype=torch.float16,
        device_map="auto",
        low_cpu_mem_usage=True
    )
    print("✓ Model loaded")
except Exception as e:
    print(f"✗ Model loading failed: {e}")

# Alternative: Load with specific device
model = AutoModelForCausalLM.from_pretrained(
    str(model_path),
    torch_dtype=torch.float16,
    device_map={"": "cpu"}  # Force CPU loading
)
```

### Issue: GPU Detection and Usage

**Symptoms**:
- CUDA not available
- GPU not being used
- Inconsistent GPU detection

**Solutions**:

```python
# Check CUDA installation
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU count: {torch.cuda.device_count()}")

if torch.cuda.is_available():
    for i in range(torch.cuda.device_count()):
        gpu = torch.cuda.get_device_properties(i)
        print(f"GPU {i}: {gpu.name}, Memory: {gpu.total_memory / 1e9:.1f} GB")

# Set specific GPU
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "0"  # Use first GPU only

# Force CPU if GPU issues
os.environ["CUDA_VISIBLE_DEVICES"] = ""  # Disable GPU

# Check GPU memory usage
if torch.cuda.is_available():
    print(f"Allocated: {torch.cuda.memory_allocated() / 1e9:.1f} GB")
    print(f"Reserved: {torch.cuda.memory_reserved() / 1e9:.1f} GB")
```

## Evaluation Errors

### Issue: Scoring Function Failures

**Symptoms**:
- `TypeError` in scoring functions
- Inconsistent evaluation results
- Missing evaluation metrics

**Solutions**:

```python
# Debug scoring step by step
from src.evaluation.empathy_scorer import EmpathyScorer

scorer = EmpathyScorer()

# Test with simple input
try:
    score = scorer.score_empathy(
        response="I understand how you feel.",
        patient_message="I'm feeling sad.",
        context="depression_support"
    )
    print(f"Empathy score: {score}")
except Exception as e:
    print(f"Scoring error: {e}")
    import traceback
    traceback.print_exc()

# Validate input types
def debug_inputs(response, patient_message, context):
    print(f"Response type: {type(response)}, length: {len(response) if response else 0}")
    print(f"Patient message type: {type(patient_message)}, length: {len(patient_message) if patient_message else 0}")
    print(f"Context type: {type(context)}")
    
    # Check for None values
    if response is None:
        raise ValueError("Response cannot be None")
    if patient_message is None:
        raise ValueError("Patient message cannot be None")
    if context is None:
        raise ValueError("Context cannot be None")

debug_inputs("test response", "test message", "test_context")
```

### Issue: Safety Detection Failures

**Symptoms**:
- False positives in crisis detection
- Missing crisis situations
- Inconsistent safety assessments

**Solutions**:

```python
# Debug safety detection
from src.evaluation.safety_detector import SafetyDetector

detector = SafetyDetector()

# Test with known crisis statements
test_cases = [
    ("I want to hurt myself", "Should detect self-harm"),
    ("I'm feeling sad today", "Should be safe"),
    ("Everyone would be better without me", "Should detect suicidal ideation"),
    ("This headache is killing me", "Should not trigger false positive")
]

for statement, expected in test_cases:
    flags, level = detector.detect_safety_issues(statement)
    print(f"Statement: '{statement}'")
    print(f"Expected: {expected}")
    print(f"Flags: {flags}, Level: {level}")
    print("---")

# Check detection thresholds
detector.debug_mode = True
flags, level = detector.detect_safety_issues("I want to die")
```

### Issue: Conversation Generation Failures

**Symptoms**:
- Empty or incomplete conversations
- Model response timeouts
- Conversation loop errors

**Solutions**:

```python
# Debug conversation generation
from src.conversation.conversation_manager import ConversationManager

manager = ConversationManager(config={
    'debug': True,
    'timeout': 60,
    'max_retries': 3
})

# Test with simple scenario
test_scenario = {
    'scenario_id': 'DEBUG_001',
    'opening_statement': 'I need help with anxiety.',
    'conversation_goals': ['assess_anxiety', 'provide_support'],
    'max_turns': 5
}

try:
    conversation = manager.generate_conversation(
        model_client=your_model_client,
        scenario=test_scenario,
        conversation_id="debug_test"
    )
    print(f"Generated {len(conversation['conversation_turns'])} turns")
except Exception as e:
    print(f"Generation failed: {e}")
    # Check conversation state
    print(f"Last state: {manager.get_last_state()}")
```

## Performance Issues

### Issue: Slow Evaluation Performance

**Symptoms**:
- Long evaluation times
- High CPU/memory usage
- System freezing during evaluation

**Solutions**:

```python
# Profile performance
import cProfile
import pstats

# Profile evaluation function
profiler = cProfile.Profile()
profiler.enable()

# Your evaluation code here
# evaluation_results = run_evaluation(conversations)

profiler.disable()

# Analyze results
stats = pstats.Stats(profiler)
stats.sort_stats('cumulative')
stats.print_stats(10)  # Top 10 functions

# Optimize batch processing
from src.evaluation.batch_evaluator import BatchEvaluator

evaluator = BatchEvaluator(
    batch_size=8,  # Reduce if memory issues
    num_workers=2,  # Reduce if CPU overload
    timeout=120     # Increase for complex evaluations
)

# Monitor memory usage
import psutil
import os

process = psutil.Process(os.getpid())
print(f"Memory usage: {process.memory_info().rss / 1e6:.1f} MB")
```

### Issue: Memory Leaks

**Symptoms**:
- Gradually increasing memory usage
- Out of memory errors after prolonged use
- System slowdown

**Solutions**:

```python
# Enable garbage collection debugging
import gc
gc.set_debug(gc.DEBUG_LEAK)

# Monitor object creation
import tracemalloc
tracemalloc.start()

# Your code here

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1e6:.1f} MB")
print(f"Peak memory usage: {peak / 1e6:.1f} MB")

# Find memory leaks
snapshot = tracemalloc.take_snapshot()
top_stats = snapshot.statistics('lineno')
for stat in top_stats[:10]:
    print(stat)

# Manual cleanup
del large_objects
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

## Database Problems

### Issue: Database Connection Errors

**Symptoms**:
- `sqlite3.OperationalError: database is locked`
- Connection timeout errors
- Database file corruption

**Solutions**:

```python
# Check database file
import sqlite3
import os

db_path = "data/conversations.db"

# Verify file exists and permissions
if os.path.exists(db_path):
    stat = os.stat(db_path)
    print(f"Database size: {stat.st_size} bytes")
    print(f"Permissions: {oct(stat.st_mode)[-3:]}")
else:
    print("Database file does not exist")

# Test connection
try:
    conn = sqlite3.connect(db_path, timeout=30.0)
    cursor = conn.cursor()
    cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
    tables = cursor.fetchall()
    print(f"Tables: {tables}")
    conn.close()
except sqlite3.Error as e:
    print(f"Database error: {e}")

# Fix locked database
# 1. Close all connections
# 2. Restart application
# 3. Check for zombie processes
import psutil
for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
    if 'python' in proc.info['name'] and 'mental-health' in str(proc.info['cmdline']):
        print(f"Found process: {proc.info}")
```

### Issue: Database Schema Issues

**Symptoms**:
- `sqlite3.OperationalError: no such table`
- Schema mismatch errors
- Migration failures

**Solutions**:

```bash
# Reinitialize database
python scripts/init_database.py --force

# Check schema
sqlite3 data/conversations.db ".schema"

# Manual schema creation
sqlite3 data/conversations.db << EOF
CREATE TABLE IF NOT EXISTS conversations (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    conversation_id TEXT UNIQUE NOT NULL,
    model_name TEXT NOT NULL,
    scenario_id TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    conversation_data TEXT NOT NULL
);
EOF

# Backup and restore if corrupted
cp data/conversations.db data/conversations.db.backup
sqlite3 data/conversations.db ".dump" | sqlite3 data/conversations_new.db
mv data/conversations_new.db data/conversations.db
```

## Network and API Issues

### Issue: API Rate Limiting

**Symptoms**:
- `openai.RateLimitError`
- HTTP 429 responses
- Sudden API failures

**Solutions**:

```python
# Implement rate limiting
import time
from functools import wraps

def rate_limit(calls_per_minute=60):
    min_interval = 60.0 / calls_per_minute
    last_called = [0.0]
    
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            elapsed = time.time() - last_called[0]
            left_to_wait = min_interval - elapsed
            if left_to_wait > 0:
                time.sleep(left_to_wait)
            ret = func(*args, **kwargs)
            last_called[0] = time.time()
            return ret
        return wrapper
    return decorator

# Use exponential backoff
import random

def exponential_backoff(max_retries=5):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    if attempt == max_retries - 1:
                        raise e
                    
                    wait_time = (2 ** attempt) + random.uniform(0, 1)
                    print(f"Attempt {attempt + 1} failed, waiting {wait_time:.2f}s")
                    time.sleep(wait_time)
            return None
        return wrapper
    return decorator

# Monitor API usage
class APIUsageMonitor:
    def __init__(self):
        self.requests_count = 0
        self.start_time = time.time()
    
    def log_request(self):
        self.requests_count += 1
        elapsed = time.time() - self.start_time
        rate = self.requests_count / elapsed * 60  # requests per minute
        print(f"API requests: {self.requests_count}, Rate: {rate:.1f}/min")
```

### Issue: Network Connectivity

**Symptoms**:
- Connection timeout errors
- DNS resolution failures
- Intermittent network issues

**Solutions**:

```bash
# Test basic connectivity
ping -c 4 api.openai.com
nslookup api.openai.com

# Test HTTPS connectivity
curl -I https://api.openai.com/v1/models

# Check proxy settings
echo $HTTP_PROXY
echo $HTTPS_PROXY

# Configure requests with proxy
export HTTP_PROXY=http://proxy.example.com:8080
export HTTPS_PROXY=http://proxy.example.com:8080

# Test with Python requests
python -c "
import requests
response = requests.get('https://api.openai.com/v1/models', timeout=10)
print(f'Status: {response.status_code}')
"
```

## Memory and Resource Issues

### Issue: Out of Memory Errors

**Symptoms**:
- `RuntimeError: CUDA out of memory`
- `MemoryError` in Python
- System becomes unresponsive

**Solutions**:

```python
# Monitor memory usage
import psutil
import torch

def print_memory_usage():
    # System memory
    memory = psutil.virtual_memory()
    print(f"System RAM: {memory.used / 1e9:.1f}GB / {memory.total / 1e9:.1f}GB ({memory.percent}%)")
    
    # GPU memory
    if torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            allocated = torch.cuda.memory_allocated(i) / 1e9
            reserved = torch.cuda.memory_reserved(i) / 1e9
            total = torch.cuda.get_device_properties(i).total_memory / 1e9
            print(f"GPU {i}: {allocated:.1f}GB allocated, {reserved:.1f}GB reserved / {total:.1f}GB total")

# Optimize memory usage
def optimize_memory():
    # Clear GPU cache
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    # Force garbage collection
    import gc
    gc.collect()
    
    # Reduce model precision
    model = model.half()  # Use float16 instead of float32
    
    # Use gradient checkpointing
    model.gradient_checkpointing_enable()

# Set memory limits
import resource

# Limit memory usage (Linux/macOS)
resource.setrlimit(resource.RLIMIT_AS, (8 * 1024**3, 8 * 1024**3))  # 8GB limit
```

### Issue: CPU Overutilization

**Symptoms**:
- 100% CPU usage
- System slowdown
- Process hanging

**Solutions**:

```python
# Limit CPU threads
import os
os.environ["OMP_NUM_THREADS"] = "4"
os.environ["MKL_NUM_THREADS"] = "4"
os.environ["OPENBLAS_NUM_THREADS"] = "4"

# Monitor CPU usage
import psutil
import time

def monitor_cpu():
    cpu_percent = psutil.cpu_percent(interval=1)
    print(f"CPU usage: {cpu_percent}%")
    
    # Per-core usage
    cpu_per_core = psutil.cpu_percent(interval=1, percpu=True)
    for i, usage in enumerate(cpu_per_core):
        print(f"Core {i}: {usage}%")

# Use process pools for CPU-intensive tasks
from multiprocessing import Pool
import concurrent.futures

def cpu_intensive_task(data):
    # Your processing code here
    return result

# Limit concurrent workers
with concurrent.futures.ProcessPoolExecutor(max_workers=2) as executor:
    futures = [executor.submit(cpu_intensive_task, item) for item in data]
    results = [future.result() for future in futures]
```

## Platform-Specific Issues

### Windows-Specific Issues

#### Issue: Path Length Limitations

**Symptoms**:
- `FileNotFoundError` with long paths
- Model loading failures on Windows

**Solutions**:

```cmd
# Enable long path support (requires admin)
reg add "HKEY_LOCAL_MACHINE\SYSTEM\CurrentControlSet\Control\FileSystem" /v LongPathsEnabled /t REG_DWORD /d 1

# Use shorter model paths
mkdir C:\models
# Move models to C:\models\ instead of deep nested paths

# Use UNC paths if needed
set DEEPSEEK_MODEL_PATH=\\?\C:\models\deepseek-llm-7b-chat
```

#### Issue: Windows Defender Interference

**Symptoms**:
- Slow file operations
- Random process termination
- Performance degradation

**Solutions**:

```cmd
# Add exclusions to Windows Defender
# 1. Open Windows Security
# 2. Go to Virus & threat protection
# 3. Add exclusions for:
#    - Project directory
#    - Python executable
#    - Virtual environment directory
```

### macOS-Specific Issues

#### Issue: Apple Silicon Compatibility

**Symptoms**:
- Architecture mismatch errors
- PyTorch installation issues
- Performance problems

**Solutions**:

```bash
# Check architecture
uname -m  # Should show arm64 for Apple Silicon

# Install Apple Silicon optimized packages
pip install torch torchvision torchaudio

# Set Metal Performance Shaders (MPS) backend
export PYTORCH_ENABLE_MPS_FALLBACK=1

# Verify MPS availability
python -c "
import torch
print(f'MPS available: {torch.backends.mps.is_available()}')
print(f'MPS built: {torch.backends.mps.is_built()}')
"
```

#### Issue: Xcode Command Line Tools

**Symptoms**:
- Compilation errors during package installation
- Missing header files

**Solutions**:

```bash
# Install Xcode Command Line Tools
xcode-select --install

# Verify installation
xcode-select -p
gcc --version

# Reset if corrupted
sudo xcode-select --reset
```

### Linux-Specific Issues

#### Issue: GLIBC Version Compatibility

**Symptoms**:
- `version 'GLIBC_X.XX' not found`
- Binary compatibility errors

**Solutions**:

```bash
# Check GLIBC version
ldd --version

# Install from source if needed
wget https://ftp.gnu.org/gnu/glibc/glibc-2.34.tar.gz
tar -xzf glibc-2.34.tar.gz
cd glibc-2.34
./configure --prefix=/usr/local/glibc-2.34
make && sudo make install

# Use conda for consistent environment
conda install glibc
```

## Advanced Debugging

### Debug Mode Configuration

```python
# config/debug.yaml
debug:
  enabled: true
  log_level: "DEBUG"
  components:
    model_loading: true
    conversation_generation: true
    evaluation: true
    database: true
  
  output:
    save_intermediate_results: true
    log_api_requests: true
    log_api_responses: true
    trace_function_calls: true
    
  performance:
    profile_memory: true
    profile_cpu: true
    benchmark_operations: true
```

### Logging Configuration

```python
# Enhanced logging setup
import logging
import sys
from pathlib import Path

def setup_debug_logging():
    # Create logs directory
    Path("logs").mkdir(exist_ok=True)
    
    # Configure root logger
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s',
        handlers=[
            logging.FileHandler('logs/debug.log'),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    # Configure specific loggers
    loggers = [
        'src.models',
        'src.evaluation',
        'src.conversation',
        'src.analysis'
    ]
    
    for logger_name in loggers:
        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.DEBUG)
        
        # Add file handler for component-specific logs
        handler = logging.FileHandler(f'logs/{logger_name.split(".")[-1]}.log')
        handler.setFormatter(logging.Formatter(
            '%(asctime)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s'
        ))
        logger.addHandler(handler)

# Enable debug logging
setup_debug_logging()
```

### Interactive Debugging

```python
# Debug with IPython/Jupyter
%load_ext autoreload
%autoreload 2

# Set breakpoints
import pdb; pdb.set_trace()

# Or use IPython debugger
from IPython.core.debugger import set_trace
set_trace()

# Debug specific functions
def debug_empathy_scoring():
    from src.evaluation.empathy_scorer import EmpathyScorer
    
    scorer = EmpathyScorer()
    
    # Test with problematic input
    response = "Your problematic response here"
    patient_message = "Patient message here"
    context = "test_context"
    
    # Step through scoring
    print("Starting empathy scoring...")
    
    try:
        # Add debug prints in your scorer
        score = scorer.score_empathy(response, patient_message, context)
        print(f"Final score: {score}")
    except Exception as e:
        print(f"Error occurred: {e}")
        import traceback
        traceback.print_exc()

debug_empathy_scoring()
```

### System Information Collection

```python
def collect_system_info():
    """Collect comprehensive system information for debugging."""
    import platform
    import sys
    import psutil
    import torch
    import transformers
    import numpy
    import pandas
    
    info = {
        'platform': {
            'system': platform.system(),
            'release': platform.release(),
            'version': platform.version(),
            'machine': platform.machine(),
            'processor': platform.processor(),
        },
        'python': {
            'version': sys.version,
            'executable': sys.executable,
            'path': sys.path[:3],  # First 3 entries
        },
        'memory': {
            'total': f"{psutil.virtual_memory().total / 1e9:.1f} GB",
            'available': f"{psutil.virtual_memory().available / 1e9:.1f} GB",
            'percent': f"{psutil.virtual_memory().percent}%",
        },
        'cpu': {
            'count': psutil.cpu_count(),
            'freq': f"{psutil.cpu_freq().current:.0f} MHz" if psutil.cpu_freq() else "Unknown",
        },
        'packages': {
            'torch': torch.__version__,
            'transformers': transformers.__version__,
            'numpy': numpy.__version__,
            'pandas': pandas.__version__,
        }
    }
    
    # GPU information
    if torch.cuda.is_available():
        info['gpu'] = {
            'count': torch.cuda.device_count(),
            'devices': []
        }
        for i in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(i)
            info['gpu']['devices'].append({
                'name': props.name,
                'memory': f"{props.total_memory / 1e9:.1f} GB",
                'compute_capability': f"{props.major}.{props.minor}"
            })
    else:
        info['gpu'] = {'available': False}
    
    return info

# Run system info collection
import json
system_info = collect_system_info()
print(json.dumps(system_info, indent=2))

# Save to file for support
with open('logs/system_info.json', 'w') as f:
    json.dump(system_info, f, indent=2)
```

## Getting Help

### Self-Diagnosis Checklist

Before seeking help, run through this checklist:

1. **Basic Environment**:
   - [ ] Python version 3.8+
   - [ ] Virtual environment activated
   - [ ] All dependencies installed
   - [ ] Environment variables set

2. **Configuration**:
   - [ ] `.env` file exists and populated
   - [ ] Model paths are correct
   - [ ] API keys are valid
   - [ ] Database is initialized

3. **System Resources**:
   - [ ] Sufficient RAM available
   - [ ] Disk space available
   - [ ] Network connectivity working
   - [ ] No resource conflicts

4. **Common Fixes Tried**:
   - [ ] Restarted application
   - [ ] Cleared cache/temporary files
   - [ ] Updated dependencies
   - [ ] Checked logs for errors

### Support Channels

**GitHub Issues**:
- Search existing issues first
- Use appropriate labels
- Include system information
- Provide minimal reproduction case

**Community Support**:
- Stack Overflow (tag: mental-health-llm-evaluation)
- Reddit: r/MachineLearning
- Discord: AI/ML communities

**Documentation**:
- [API Reference](api-reference.md)
- [Configuration Guide](configuration.md)
- [Deployment Guides](deployment/)

### Bug Report Template

When reporting issues, please include:

```markdown
## Bug Report

### Environment
- OS: [e.g., Ubuntu 20.04, macOS 12.0, Windows 11]
- Python version: [e.g., 3.9.7]
- Package version: [e.g., 1.0.0]

### Description
[Clear description of the issue]

### Steps to Reproduce
1. [First step]
2. [Second step]
3. [Third step]

### Expected Behavior
[What you expected to happen]

### Actual Behavior
[What actually happened]

### Error Messages
```
[Full error traceback]
```

### System Information
[Output of collect_system_info() function]

### Additional Context
[Any other relevant information]
```

---

This troubleshooting guide covers the most common issues. For specific problems not covered here, please check the GitHub issues or create a new issue with detailed information about your problem.