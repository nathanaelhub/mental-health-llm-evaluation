# Development Scripts

This folder contains auxiliary scripts used for development, testing, and debugging the Mental Health LLM Evaluation project.

## Testing Scripts

- **`test_success_tracking.py`** - Tests the StatusTracker functionality
- **`test_evaluation_tracking.py`** - Tests evaluation pipeline with tracking
- **`test_winner_messages.py`** - Tests winner message display fixes
- **`test_all_models.py`** - Tests all model client connections
- **`quick_fix_demo.py`** - Quick demonstration of recent fixes

## Debug & Utilities

- **`debug_test.py`** - Enhanced debugging for NoneType errors
- **`validate_cleanup.py`** - Validates project cleanup and organization
- **`setup_models.py`** - Model configuration setup and checking

## Backup & Reliability

- **`reliable_research.py`** - Wrapper script with retry logic and error handling
- **`run_research_backup.py`** - Backup version of the main research script
- **`run_conversation_generation.py`** - Conversation generation utilities

## Usage

All scripts in this folder should be run from the project root directory:

```bash
# Run from project root
python scripts/development/test_success_tracking.py
python scripts/development/reliable_research.py --all-models
```

## Main Scripts

The main user-facing scripts are located in the parent `/scripts` directory:

- `run_research.py` - Main evaluation script
- `compare_models.py` - Quick model comparison tool  
- `demo_presentation.py` - Clean demo for presentations
- `clean_results.py` - Results maintenance and cleanup