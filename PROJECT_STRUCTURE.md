# Mental Health LLM Evaluation - Project Structure

## Clean, Organized Directory Structure

```
mental-health-llm-evaluation/
├── 📁 scripts/                    # All executable scripts
│   ├── run_research.py           # 🎯 Main research pipeline
│   ├── compare_models.py         # 🔄 Model comparison tool
│   ├── run_conversation_generation.py
│   └── validate_cleanup.py       # 🔍 System validation script
├── 📁 src/                       # Core source code
│   ├── 📁 models/                # Model implementations
│   │   ├── base_model.py
│   │   ├── openai_client.py      # OpenAI GPT-4
│   │   ├── claude_client.py      # Anthropic Claude
│   │   ├── deepseek_client.py    # DeepSeek (local)
│   │   └── gemma_client.py       # Google Gemma (local)
│   ├── 📁 evaluation/            # Evaluation engine
│   ├── 📁 analysis/              # Statistical analysis
│   ├── 📁 scenarios/             # Scenario processing
│   ├── 📁 config/                # Configuration management
│   └── 📁 utils/                 # Utilities
├── 📁 config/                    # Configuration files
│   ├── main.yaml                 # Main configuration
│   ├── 📁 models/                # Model-specific configs
│   └── 📁 scenarios/             # Mental health scenarios
├── 📁 docs/                      # 📚 All documentation
│   ├── README.md                 # Documentation index
│   ├── FILE_DESCRIPTIONS.md     # Informal file guide
│   ├── TESTING_GUIDE.md         # Complete testing workflow
│   ├── LOCAL_MODELS_SETUP.md    # Local model setup
│   ├── VALIDATION_REPORT.md     # System validation
│   ├── methodology.md           # Research methodology
│   ├── results_interpretation.md # Results analysis
│   └── dependency_analysis.md   # Technical dependencies
├── 📁 data/                      # Data storage
│   ├── 📁 conversations/         # Generated conversations
│   ├── 📁 results/               # Evaluation results
│   └── 📁 scenarios/             # Additional scenarios
├── 📁 output/                    # Results output
│   ├── 📁 visualizations/        # Generated charts
│   └── 📁 presentation/          # Presentation slides
├── .env.example                  # Environment template
├── requirements.txt              # Dependencies
└── README.md                     # Main project documentation
```

## Key Organization Principles

### 1. **Consolidated Scripts**
- All executable scripts in `scripts/` directory
- No more confusion between `tools/` and `scripts/`
- Clear naming: `run_research.py`, `compare_models.py`

### 2. **Centralized Documentation**
- All `.md` files moved to `docs/` folder
- Organized documentation index in `docs/README.md`
- Clear separation of guides vs reports

### 3. **Unified Local Models**
- Both DeepSeek and Gemma use same server: `192.168.86.23:1234`
- Different API identifiers: `deepseek-r1` and `google/gemma-3-12b`
- Simplified environment configuration

### 4. **Clean Root Directory**
- Only essential files in root: `README.md`, `requirements.txt`, `.env.example`
- No scattered markdown files
- Clear project structure

## Quick Commands

```bash
# System validation
python scripts/validate_cleanup.py

# Main research pipeline
python scripts/run_research.py --quick

# Model comparison
python scripts/compare_models.py --help

# View documentation
ls docs/
```

## Documentation Navigation

- **Quick Start**: `docs/TESTING_GUIDE.md`
- **File Overview**: `docs/FILE_DESCRIPTIONS.md`
- **Local Models**: `docs/LOCAL_MODELS_SETUP.md`
- **Validation**: `docs/VALIDATION_REPORT.md`

## Validation Results

- **✅ 97.0% system validation success**
- **✅ 100% imports working**
- **✅ 100% files found**
- **✅ 100% models initialized**
- **✅ All documentation organized**

This structure provides clear navigation, logical organization, and eliminates confusion about where files belong. Perfect for a clean, professional capstone project! 🎓