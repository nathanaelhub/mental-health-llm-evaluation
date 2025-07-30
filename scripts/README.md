# Scripts Directory

This directory contains all project scripts organized for easy navigation and usage.

## Core Research Scripts

These are the main scripts for running the mental health LLM evaluation research:

- **`run_research.py`** - Main evaluation pipeline
  - Comprehensive research runner with modular architecture
  - Supports all models: OpenAI, Claude, DeepSeek, Gemma
  - Multiple output modes: clean, minimal, ultra-clean, demo
  - Usage: `python run_research.py [--models MODEL1,MODEL2] [--quick] [--output DIR]`

- **`compare_models.py`** - Direct model comparison
  - Head-to-head model evaluation
  - Quick testing and validation
  - Statistical analysis and reporting

- **`run_evaluation.py`** - Evaluation runner
  - Core evaluation system
  - Scenario processing
  - Results generation

## Development Tools

### Chat Server Development

The `chat_server_development/` subdirectory contains all scripts related to UI and chat server development:

- **Testing & Debugging**
  - `debug_422_error.py` - API validation debugging
  - `debug_chat_interface.py` - Frontend debugging
  - `debug_model_selection.py` - Model selection debugging
  - `test_chat_api.py` - API endpoint testing
  - `test_complete_system.py` - End-to-end system testing
  - `test_health_check.py` - Health check validation
  - `test_validation_only.py` - Input validation testing
  - `test_fallback_scenarios.py` - Fallback behavior testing
  - `test_local_models.py` - Local model connectivity testing
  - `simple_chat_test.py` - Basic chat functionality testing

- **Development Utilities**
  - `fix_chat_system.py` - System repair utilities
  - `run_chat_server.py` - Development server runner
  - `demo_dynamic_selector.py` - Model selector demonstrations

## Usage Guidelines

### For Research
```bash
# Quick test with 2 default models
python scripts/run_research.py --quick

# Full evaluation with all models
python scripts/run_research.py --all-models

# Clean output for presentations
python scripts/run_research.py --all-models --demo
```

### For Development
```bash
# Test chat API
python scripts/chat_server_development/test_chat_api.py

# Debug specific issues
python scripts/chat_server_development/debug_422_error.py

# Test local model connectivity
python scripts/chat_server_development/test_local_models.py
```

## Directory Organization

- **Root (`/scripts/`)**: Core research scripts only
- **`chat_server_development/`**: All UI/chat development tools
- **Keep clean**: No temporary or test files in root

This organization maintains clear separation between research and development tools while keeping the main scripts directory focused on core functionality.