# Project Structure

This document explains the directory structure and organization of the Mental Health LLM Evaluation project.

## 📁 Directory Overview

```
mental-health-llm-evaluation/
├── src/                      # Source code
│   ├── analysis/            # Statistical analysis and visualization
│   ├── config/              # Configuration handling
│   ├── evaluation/          # Evaluation metrics and logic
│   ├── models/              # Model interfaces and clients
│   ├── scenarios/           # Scenario loading and management
│   └── utils/               # Utility functions and helpers
├── data/                    # Static input data
│   └── scenarios/           # Scenario JSON files
├── output/                  # Generated output files
│   ├── conversations/       # Generated conversation logs
│   ├── evaluations/         # Evaluation results (JSON)
│   ├── analysis/            # Statistical analysis results
│   ├── visualizations/      # Charts and graphs
│   └── logs/                # Application logs
├── config/                  # Configuration files
│   ├── models/              # Model-specific configurations
│   └── scenarios/           # Scenario YAML configurations
├── scripts/                 # Utility scripts
└── docs/                    # Documentation
```

## 🎯 Design Philosophy

The project follows a **clean architecture** approach with clear separation of concerns:

- **Static vs Dynamic**: Static data (scenarios, config) separate from generated output
- **Functionality-based**: Code organized by what it does, not where it's used
- **Scalability**: Easy to add new models, scenarios, or analysis types
- **Maintainability**: Clear boundaries and minimal dependencies

## 📂 Directory Details

### `src/` - Source Code

All Python source code organized by functionality:

#### `src/analysis/`
- **Purpose**: Statistical analysis and visualization
- **Key Files**:
  - `statistical_analysis.py` - Statistical tests and analysis
  - `visualization.py` - Chart generation and plotting
- **Dependencies**: Uses `output/` for results, `config/` for settings

#### `src/config/`
- **Purpose**: Configuration management
- **Key Files**:
  - `config_loader.py` - Configuration file loading
  - `config_schema.py` - Configuration validation
  - `config_utils.py` - Configuration utilities
- **Dependencies**: Reads from `config/` directory

#### `src/evaluation/`
- **Purpose**: Evaluation metrics and assessment logic
- **Key Files**:
  - `mental_health_evaluator.py` - Main evaluation orchestrator
  - `evaluation_metrics.py` - Scoring and metrics calculation
- **Dependencies**: Uses `data/scenarios/`, outputs to `output/evaluations/`

#### `src/models/`
- **Purpose**: Model interfaces and API clients
- **Key Files**:
  - `base_model.py` - Abstract base class for models
  - `openai_client.py` - OpenAI API client
  - `claude_client.py` - Anthropic Claude client
  - `deepseek_client.py` - DeepSeek API client
  - `gemma_client.py` - Gemma model client
  - `local_llm_client.py` - Local LLM server client
- **Dependencies**: Uses `config/models/` for configuration

#### `src/scenarios/`
- **Purpose**: Scenario loading and conversation generation
- **Key Files**:
  - `scenario_loader.py` - Loads scenarios from JSON/YAML
  - `conversation_generator.py` - Generates conversations
  - `scenario.py` - Scenario data structures
- **Dependencies**: Uses `data/scenarios/` and `config/scenarios/`

#### `src/utils/`
- **Purpose**: Shared utilities and helpers
- **Key Files**:
  - `paths.py` - Centralized path management
  - `data_storage.py` - Data persistence utilities
  - `logging_config.py` - Logging configuration
- **Dependencies**: Used by all other modules

### `data/` - Static Input Data

Contains unchanging data that feeds into the evaluation:

#### `data/scenarios/`
- **Purpose**: Scenario definitions for evaluation
- **Format**: JSON files with standardized structure
- **Examples**:
  - `anxiety_001.json` - Anxiety scenario
  - `depression_001.json` - Depression scenario
  - `trauma_001.json` - Trauma scenario
- **Structure**:
  ```json
  {
    "id": "anxiety_001",
    "title": "Work-related anxiety",
    "description": "User experiencing anxiety about work presentation",
    "category": "anxiety",
    "severity_level": "mild",
    "user_message": "I have a big presentation...",
    "expected_qualities": ["empathy", "validation", "practical_support"],
    "emotional_context": {...},
    "evaluation_criteria": {...}
  }
  ```

### `output/` - Generated Output

All files generated during evaluation and analysis:

#### `output/conversations/`
- **Purpose**: Generated conversation logs
- **Format**: JSON files with full conversation history
- **Naming**: `conversation_{id}_{timestamp}.json`
- **Contains**: User messages, model responses, metadata

#### `output/evaluations/`
- **Purpose**: Evaluation results and scores
- **Format**: JSON files with structured evaluation data
- **Key Files**:
  - `evaluation_results_{timestamp}.json` - Detailed results
  - `model_strengths_{timestamp}.json` - Model strength analysis
- **Contains**: Scores, metrics, comparisons, metadata

#### `output/analysis/`
- **Purpose**: Statistical analysis results and reports
- **Format**: JSON and text files
- **Key Files**:
  - `statistical_analysis_{timestamp}.json` - Statistical test results
  - `analysis_report_{timestamp}.txt` - Human-readable report
- **Contains**: Significance tests, effect sizes, recommendations

#### `output/visualizations/`
- **Purpose**: Charts, graphs, and visual presentations
- **Format**: PNG images and presentation slides
- **Structure**:
  - `charts/` - Individual charts and graphs
  - `presentation/` - Presentation slides
- **Examples**:
  - `1_overall_comparison.png` - Model comparison chart
  - `2_category_radar.png` - Category performance radar
  - `slide_1_executive_summary.png` - Executive summary slide

#### `output/logs/`
- **Purpose**: Application logs and debug information
- **Format**: Log files with timestamps
- **Key Files**:
  - `evaluation.log` - Main evaluation log
  - `model_requests.log` - Model API request log
- **Rotation**: Logs are rotated to prevent excessive disk usage

### `config/` - Configuration Files

System configuration and settings:

#### `config/main.yaml`
- **Purpose**: Main configuration file
- **Contains**: 
  - Model settings (API keys, parameters)
  - Evaluation weights and thresholds
  - Output preferences
  - Path configurations
  - Research parameters

#### `config/models/`
- **Purpose**: Model-specific configurations
- **Files**:
  - `model_settings.yaml` - Default model parameters
  - `local_experiment.yaml` - Local LLM experiment settings
- **Contains**: Temperature, max_tokens, timeout settings

#### `config/scenarios/`
- **Purpose**: Detailed scenario configurations
- **Files**:
  - `main_scenarios.yaml` - Main scenario definitions
  - `anxiety_scenarios.yaml` - Anxiety-specific scenarios
  - `depression_scenarios.yaml` - Depression-specific scenarios
  - `crisis_scenarios.yaml` - Crisis intervention scenarios
- **Contains**: Rich scenario data with conversation flows

### `scripts/` - Utility Scripts

Executable scripts for common tasks:

- `run_research.py` - Main research evaluation script
- `compare_models.py` - Model comparison utility
- `run_conversation_generation.py` - Conversation generation script
- `validate_cleanup.py` - Data validation and cleanup

### `docs/` - Documentation

Project documentation:

- `README.md` - Main project documentation
- `PROJECT_STRUCTURE.md` - This file
- `TESTING_GUIDE.md` - Testing procedures
- `VALIDATION_REPORT.md` - Validation methodology
- `methodology.md` - Research methodology

## 🔄 Data Flow

The typical data flow through the system:

1. **Configuration** → `config/` files loaded by `src/config/`
2. **Scenarios** → `data/scenarios/` loaded by `src/scenarios/`
3. **Models** → `src/models/` clients initialized with config
4. **Evaluation** → `src/evaluation/` orchestrates the process
5. **Results** → Saved to `output/evaluations/`
6. **Analysis** → `src/analysis/` processes results
7. **Visualizations** → Generated to `output/visualizations/`
8. **Reports** → Final reports saved to `output/analysis/`

## 📝 Path Management

The project uses centralized path management through `src/utils/paths.py`:

```python
from src.utils.paths import get_evaluations_dir, get_scenarios_dir

# Get standard paths
evaluations_dir = get_evaluations_dir()
scenarios_dir = get_scenarios_dir()

# Get specific file paths
results_file = paths.get_evaluation_results_file("20240101_120000")
scenario_file = paths.get_scenario_file("anxiety_001")
```

## 🎛️ Configuration Management

Configuration is hierarchical:

1. **Environment Variables** (`.env`) - Secrets and environment-specific settings
2. **Main Config** (`config/main.yaml`) - Core application settings
3. **Specialized Configs** (`config/models/`, `config/scenarios/`) - Component-specific settings
4. **Defaults** - Hard-coded fallbacks in source code

## 🔧 Development Guidelines

### Adding New Components

1. **New Model**: Add to `src/models/` with base class inheritance
2. **New Scenario**: Add JSON to `data/scenarios/` and YAML to `config/scenarios/`
3. **New Analysis**: Add to `src/analysis/` with output to `output/analysis/`
4. **New Visualization**: Add to `src/analysis/visualization.py` with output to `output/visualizations/`

### File Naming Conventions

- **Source files**: `snake_case.py`
- **Configuration files**: `snake_case.yaml`
- **Data files**: `category_number.json` (e.g., `anxiety_001.json`)
- **Output files**: `descriptive_name_timestamp.ext`
- **Log files**: `component_name.log`

### Import Conventions

- Use relative imports within modules: `from .submodule import function`
- Use absolute imports between modules: `from src.utils.paths import get_output_dir`
- Use path utilities: `from src.utils.paths import get_evaluations_dir`

## 🚨 Important Notes

### Version Control

- **Included**: Source code, configuration, documentation, static data
- **Excluded**: Generated output, logs, temporary files, environment files
- **Selective**: Some example outputs may be included for documentation

### Security

- **API Keys**: Never commit to version control, use `.env` files
- **Sensitive Data**: Keep in `output/` which is git-ignored
- **Configuration**: Separate public settings from private credentials

### Performance

- **Output Directory**: Can grow large, implement cleanup procedures
- **Logs**: Rotate logs to prevent disk space issues
- **Temporary Files**: Clean up `output/temp/` regularly

## 🔄 Migration Notes

This structure was established through careful migration from previous layouts:

- `results/` → `output/` (more intuitive naming)
- `generated/` → `output/` (consolidated output location)
- Path references updated throughout codebase
- Configuration files updated to reflect new structure

## 📊 Monitoring

The structure supports monitoring through:

- **Logs**: Centralized in `output/logs/`
- **Metrics**: Evaluation results in `output/evaluations/`
- **Status**: Analysis reports in `output/analysis/`
- **Artifacts**: Visualizations in `output/visualizations/`

This structure provides a solid foundation for the Mental Health LLM Evaluation project, supporting both development and research activities while maintaining clarity and organization.