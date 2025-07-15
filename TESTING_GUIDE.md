# 🚀 From Clean Code to Research Results - Testing Guide

## Phase 1: System Validation & Testing (Day 1)

### 1.1 Run Full System Test

```bash
# Navigate to your project
cd /path/to/your/project

# Activate virtual environment
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install/update dependencies
pip install -r requirements.txt

# Run comprehensive validation
python scripts/validate_cleanup.py

# Test main entry points
python scripts/run_research.py --help
python scripts/compare_models.py --help
```

### 1.2 Environment Setup

```bash
# Copy environment template
cp .env.example .env

# Edit .env file with your API keys
nano .env  # or use your preferred editor

# Required API keys:
# OPENAI_API_KEY=your_openai_key_here
# ANTHROPIC_API_KEY=your_anthropic_key_here (optional)
# LOCAL_LLM_SERVER=192.168.86.23:1234 (shared by DeepSeek and Gemma)
# DEEPSEEK_MODEL=deepseek-r1 (API identifier)
# GEMMA_MODEL=google/gemma-3-12b (API identifier)
```

### 1.3 Test Individual Models

```bash
# Test OpenAI connection
python -c "from src.models import OpenAIClient; client = OpenAIClient(); print('OpenAI: OK')"

# Test Claude connection (if API key configured)
python -c "from src.models import ClaudeClient; client = ClaudeClient(); print('Claude: OK')"

# Test DeepSeek connection (if local server running)
python -c "from src.models import DeepSeekClient; client = DeepSeekClient(); print('DeepSeek: OK')"

# Test Gemma connection (if local server running) 
python -c "from src.models import GemmaClient; client = GemmaClient(); print('Gemma: OK')"
```

### 1.4 Generate Sample Outputs

```bash
# Test quick evaluation (3 scenarios)
python scripts/run_research.py --quick --output test_outputs/

# Verify all output files are created
ls -la test_outputs/
ls -la test_outputs/visualizations/
ls -la test_outputs/presentation/

# Expected files:
# - detailed_results.json
# - statistical_analysis.json
# - model_strengths.json
# - research_report.txt
# - visualizations/ (5 charts)
# - presentation/ (4 slides)
```

### 1.5 Test Model Comparison Tool

```bash
# Test model comparison with different scenarios
python scripts/compare_models.py --scenario "anxiety_management" --verbose
python scripts/compare_models.py --scenario "depression_support" --verbose
python scripts/compare_models.py --prompt "I'm feeling overwhelmed with work stress" --verbose
```

## Phase 2: Configuration & Optimization (Day 2-3)

### 2.1 Model Configuration

Edit `config/main.yaml` to:
- Enable/disable specific models
- Adjust model parameters (temperature, max_tokens)
- Configure evaluation weights
- Set research parameters

```yaml
# Enable only available models
enabled_models: ["openai", "claude"]  # Remove unavailable models

# Adjust evaluation weights if needed
evaluation:
  weights:
    empathy: 0.30
    therapeutic: 0.25
    safety: 0.35
    clarity: 0.10
```

### 2.2 Local LLM Setup (Optional)

If running DeepSeek locally:
```bash
# Example using LM Studio or similar
# 1. Download DeepSeek model
# 2. Start local server on localhost:1234
# 3. Update LOCAL_LLM_BASE_URL in .env
```

If running Gemma locally:
```bash
# Example using Ollama or similar
# 1. Download Gemma-3-12b model
# 2. Start local server on localhost:8080
# 3. Update GEMMA_ENDPOINT in .env
```

### 2.3 Scenario Customization

```bash
# View available scenarios
cat config/scenarios/main_scenarios.yaml

# Edit scenarios if needed
nano config/scenarios/anxiety_scenarios.yaml
```

## Phase 3: Research Execution (Day 4-5)

### 3.1 Full Research Run

```bash
# Run complete research pipeline
python scripts/run_research.py --scenarios 10 --output results/

# Monitor progress and check for errors
tail -f results/research_report.txt
```

### 3.2 Analysis & Visualization

```bash
# Generate additional visualizations if needed
python -c "
from src.analysis.visualization import create_all_visualizations
create_all_visualizations('results/detailed_results.json', 'results/visualizations/')
"

# Create presentation slides
python -c "
from src.analysis.visualization import create_presentation_slides
create_presentation_slides('results/statistical_analysis.json', 'results/presentation/')
"
```

### 3.3 Model Comparison Studies

```bash
# Compare specific model pairs
python scripts/compare_models.py --models "openai,claude" --scenario "crisis_support"
python scripts/compare_models.py --models "openai,deepseek" --scenario "anxiety_management"

# Batch comparison for multiple scenarios
for scenario in anxiety depression crisis general; do
    python scripts/compare_models.py --scenario "${scenario}_support" --save "results/comparison_${scenario}.json"
done
```

## Phase 4: Results Analysis & Reporting (Day 6-7)

### 4.1 Statistical Analysis

```bash
# Generate comprehensive statistical report
python -c "
from src.analysis.statistical_analysis import generate_comprehensive_report
generate_comprehensive_report('results/detailed_results.json', 'results/statistical_report.txt')
"

# Check for statistical significance
python -c "
from src.analysis.statistical_analysis import check_significance
results = check_significance('results/detailed_results.json')
print(f'Significant differences found: {results}')
"
```

### 4.2 Model Performance Analysis

```bash
# Identify model strengths and weaknesses
python -c "
from src.analysis.statistical_analysis import identify_model_strengths
strengths = identify_model_strengths('results/detailed_results.json')
print('Model strengths analysis complete')
"

# Generate model recommendations
python -c "
from src.analysis.statistical_analysis import generate_recommendations
recommendations = generate_recommendations('results/detailed_results.json')
print('Recommendations generated')
"
```

### 4.3 Quality Assurance

```bash
# Validate all output files
python -c "
import os, json
files = ['detailed_results.json', 'statistical_analysis.json', 'model_strengths.json']
for f in files:
    if os.path.exists(f'results/{f}'):
        with open(f'results/{f}') as file:
            data = json.load(file)
            print(f'{f}: {len(data)} entries')
    else:
        print(f'{f}: MISSING')
"

# Check visualization quality
ls -la results/visualizations/*.png
ls -la results/presentation/*.png
```

## Troubleshooting Guide

### Common Issues

1. **Import Errors**
   ```bash
   # Reinstall dependencies
   pip install -r requirements.txt --force-reinstall
   
   # Check Python path
   python -c "import sys; print(sys.path)"
   ```

2. **API Connection Issues**
   ```bash
   # Test API keys
   python -c "import os; print('OpenAI key:', bool(os.getenv('OPENAI_API_KEY')))"
   python -c "import os; print('Claude key:', bool(os.getenv('ANTHROPIC_API_KEY')))"
   ```

3. **Local Model Issues**
   ```bash
   # Check if local servers are running
   curl http://localhost:1234/v1/models  # DeepSeek
   curl http://localhost:8080/v1/models  # Gemma
   ```

4. **Visualization Issues**
   ```bash
   # Check matplotlib backend
   python -c "import matplotlib; print(matplotlib.get_backend())"
   
   # Install additional fonts if needed
   sudo apt-get install fonts-dejavu-core  # Linux
   ```

### Performance Optimization

```bash
# Monitor resource usage
top -p $(pgrep -f "python.*run_research")

# Reduce memory usage by processing fewer scenarios at once
python scripts/run_research.py --scenarios 5 --output results_batch1/
python scripts/run_research.py --scenarios 5 --output results_batch2/
```

## Success Criteria

✅ **System Ready**: All validation tests pass (90%+)
✅ **API Connectivity**: At least OpenAI working, others optional
✅ **Sample Generation**: Can generate test outputs without errors
✅ **Model Comparison**: Comparison tool works with available models
✅ **Statistical Analysis**: Can generate charts and statistical reports
✅ **Research Pipeline**: Full pipeline completes successfully

## Expected Timeline

- **Day 1**: System validation and setup
- **Day 2-3**: Configuration and optimization
- **Day 4-5**: Research execution and data collection
- **Day 6-7**: Analysis, reporting, and presentation preparation

Your capstone project is now ready for serious research! 🎓