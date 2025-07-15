# Path Management Guide

This guide explains how to use the centralized path management system (`src/utils/paths.py`) in the Mental Health LLM Evaluation project.

## 🎯 Overview

The path management system provides:
- **Single source of truth** for all file paths
- **Cross-platform compatibility** using pathlib
- **Automatic directory creation** for output paths
- **Consistent path handling** across the entire project
- **Easy maintenance** when paths need to change

## 📁 Basic Usage

### Quick Start

```python
from src.utils.paths import get_evaluations_dir, get_scenarios_dir, get_output_dir

# Get standard directories
evaluations_dir = get_evaluations_dir()
scenarios_dir = get_scenarios_dir()
output_dir = get_output_dir()

# All paths are pathlib.Path objects
print(f"Evaluations will be saved to: {evaluations_dir}")
print(f"Scenarios are loaded from: {scenarios_dir}")
```

### Using the ProjectPaths Class

```python
from src.utils.paths import ProjectPaths

# Initialize (auto-detects project root)
paths = ProjectPaths()

# Get base directories
project_root = paths.get_project_root()
src_dir = paths.get_src_dir()
data_dir = paths.get_data_dir()
output_dir = paths.get_output_dir()
config_dir = paths.get_config_dir()

# Get specific subdirectories
evaluations_dir = paths.get_evaluations_dir()
scenarios_dir = paths.get_scenarios_dir()
visualizations_dir = paths.get_visualizations_dir()
```

## 🔧 Common Use Cases

### 1. Saving Evaluation Results

```python
from src.utils.paths import get_paths
import json
from datetime import datetime

paths = get_paths()

# Get timestamped filename
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
results_file = paths.get_evaluation_results_file(timestamp)

# Save results
with open(results_file, 'w') as f:
    json.dump(results_data, f, indent=2)

print(f"Results saved to: {results_file}")
```

### 2. Loading Scenarios

```python
from src.utils.paths import get_paths
import json

paths = get_paths()

# Load a specific scenario
scenario_file = paths.get_scenario_file("anxiety_001")
with open(scenario_file, 'r') as f:
    scenario_data = json.load(f)

# Load all scenarios
scenarios_dir = paths.get_scenarios_dir()
for scenario_file in scenarios_dir.glob("*.json"):
    with open(scenario_file, 'r') as f:
        scenario = json.load(f)
    print(f"Loaded scenario: {scenario['id']}")
```

### 3. Creating Visualizations

```python
from src.utils.paths import get_paths
import matplotlib.pyplot as plt

paths = get_paths()

# Create and save a chart
fig, ax = plt.subplots()
ax.plot([1, 2, 3], [4, 5, 6])
ax.set_title("Model Comparison")

# Save to visualizations directory
chart_file = paths.get_visualization_file("model_comparison", "png")
fig.savefig(chart_file, dpi=300, bbox_inches='tight')
print(f"Chart saved to: {chart_file}")
```

### 4. Logging

```python
from src.utils.paths import get_paths
import logging

paths = get_paths()

# Set up logging to the logs directory
log_file = paths.get_log_file("evaluation")
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_file),
        logging.StreamHandler()
    ]
)

logger = logging.getLogger(__name__)
logger.info(f"Logging to: {log_file}")
```

## 📂 Available Path Functions

### Convenience Functions

```python
from src.utils.paths import (
    get_project_root, get_src_dir, get_data_dir, get_output_dir,
    get_config_dir, get_scenarios_dir, get_evaluations_dir,
    get_analysis_dir, get_visualizations_dir, get_conversations_dir,
    get_logs_dir
)

# All return pathlib.Path objects
project_root = get_project_root()
scenarios_dir = get_scenarios_dir()
evaluations_dir = get_evaluations_dir()
```

### ProjectPaths Methods

```python
from src.utils.paths import ProjectPaths

paths = ProjectPaths()

# Base directories
paths.get_project_root()       # Project root
paths.get_src_dir()            # src/
paths.get_data_dir()           # data/
paths.get_output_dir()         # output/
paths.get_config_dir()         # config/

# Data directories
paths.get_scenarios_dir()      # data/scenarios/
paths.get_scenario_file(name)  # data/scenarios/{name}.json

# Output directories
paths.get_conversations_dir()  # output/conversations/
paths.get_evaluations_dir()    # output/evaluations/
paths.get_analysis_dir()       # output/analysis/
paths.get_visualizations_dir() # output/visualizations/
paths.get_logs_dir()           # output/logs/
paths.get_temp_dir()           # output/temp/

# Configuration directories
paths.get_models_config_dir()  # config/models/
paths.get_scenarios_config_dir() # config/scenarios/

# Specific files
paths.get_main_config_file()   # config/main.yaml
paths.get_env_file()           # .env
paths.get_readme_file()        # README.md
```

### File Generators

```python
from src.utils.paths import ProjectPaths

paths = ProjectPaths()

# With timestamps
timestamp = "20240101_120000"
results_file = paths.get_evaluation_results_file(timestamp)
report_file = paths.get_analysis_report_file(timestamp)
stats_file = paths.get_statistical_analysis_file(timestamp)

# Without timestamps (uses default names)
results_file = paths.get_evaluation_results_file()
report_file = paths.get_analysis_report_file()

# Specific files
conversation_file = paths.get_conversation_file("conv_001")
chart_file = paths.get_visualization_file("comparison_chart")
log_file = paths.get_log_file("evaluation")
```

## 🛠️ Utility Methods

### Directory Management

```python
from src.utils.paths import ProjectPaths

paths = ProjectPaths()

# Ensure a directory exists
my_dir = paths.get_output_dir() / "custom_analysis"
paths.ensure_dir(my_dir)

# Get relative paths
abs_path = paths.get_evaluations_dir() / "results.json"
rel_path = paths.get_relative_path(abs_path)
print(f"Relative path: {rel_path}")

# Check if path is in output directory
is_output = paths.is_output_file(abs_path)
print(f"Is output file: {is_output}")
```

### File Operations

```python
from src.utils.paths import ProjectPaths

paths = ProjectPaths()

# Get file size
results_file = paths.get_evaluation_results_file()
if results_file.exists():
    size = paths.get_file_size(results_file)
    print(f"File size: {size} bytes")

# List all output files
output_files = paths.list_output_files()
for file_path in output_files:
    print(f"Output file: {file_path}")

# Clean up temporary files
paths.cleanup_temp_files()
```

## 🔄 Migration from Hardcoded Paths

### Before (Hardcoded)

```python
# ❌ Don't do this
results_dir = "results/evaluations"
scenarios_dir = "data/scenarios"
output_file = "results/evaluations/results.json"

# Problems:
# - Not cross-platform compatible
# - Hard to maintain
# - Inconsistent across files
# - No automatic directory creation
```

### After (Path Management)

```python
# ✅ Do this instead
from src.utils.paths import get_evaluations_dir, get_scenarios_dir, get_paths

results_dir = get_evaluations_dir()
scenarios_dir = get_scenarios_dir()
output_file = get_paths().get_evaluation_results_file()

# Benefits:
# - Cross-platform compatible
# - Single source of truth
# - Automatic directory creation
# - Consistent across project
```

## 🎯 Best Practices

### 1. Use Convenience Functions for Common Paths

```python
from src.utils.paths import get_evaluations_dir, get_scenarios_dir

# Good for frequently used paths
evaluations_dir = get_evaluations_dir()
scenarios_dir = get_scenarios_dir()
```

### 2. Use ProjectPaths Instance for Complex Operations

```python
from src.utils.paths import ProjectPaths

paths = ProjectPaths()

# Good for multiple operations or custom paths
results_file = paths.get_evaluation_results_file("20240101_120000")
paths.ensure_dir(results_file.parent)
```

### 3. Use Global Instance for Consistency

```python
from src.utils.paths import get_paths

# Gets the global instance - consistent across modules
paths = get_paths()
```

### 4. Handle Path Objects Properly

```python
from src.utils.paths import get_evaluations_dir
from pathlib import Path

# Paths are pathlib.Path objects
evaluations_dir = get_evaluations_dir()

# Convert to string if needed
evaluations_str = str(evaluations_dir)

# Join paths properly
results_file = evaluations_dir / "results.json"

# Check existence
if results_file.exists():
    print(f"File exists: {results_file}")
```

## 🚨 Common Pitfalls

### 1. Don't Mix String and Path Operations

```python
# ❌ Don't do this
from src.utils.paths import get_evaluations_dir
evaluations_dir = get_evaluations_dir()
bad_path = evaluations_dir + "/results.json"  # Wrong!

# ✅ Do this instead
good_path = evaluations_dir / "results.json"
```

### 2. Don't Hardcode Paths Anymore

```python
# ❌ Don't do this
output_dir = "output/evaluations"

# ✅ Do this instead
from src.utils.paths import get_evaluations_dir
output_dir = get_evaluations_dir()
```

### 3. Don't Forget to Handle Missing Directories

```python
# ❌ Don't do this - might fail if directory doesn't exist
from src.utils.paths import get_evaluations_dir
results_file = get_evaluations_dir() / "results.json"
with open(results_file, 'w') as f:  # Might fail!
    json.dump(data, f)

# ✅ Do this instead - path management handles it automatically
from src.utils.paths import get_paths
paths = get_paths()
results_file = paths.get_evaluation_results_file()
# Directory automatically created by path management
with open(results_file, 'w') as f:
    json.dump(data, f)
```

## 🔧 Customization

### Custom Project Root

```python
from src.utils.paths import ProjectPaths

# Use custom project root
paths = ProjectPaths("/path/to/custom/project")

# Or reset global instance
from src.utils.paths import reset_paths
reset_paths("/path/to/custom/project")
```

### Custom Paths

```python
from src.utils.paths import ProjectPaths

paths = ProjectPaths()

# Create custom paths relative to project structure
custom_dir = paths.get_output_dir() / "custom_analysis"
paths.ensure_dir(custom_dir)

custom_file = custom_dir / "custom_results.json"
```

## 📊 Example: Complete Evaluation Script

```python
from src.utils.paths import get_paths
import json
from datetime import datetime

def run_evaluation():
    paths = get_paths()
    
    # Load scenarios
    scenarios_dir = paths.get_scenarios_dir()
    scenarios = []
    for scenario_file in scenarios_dir.glob("*.json"):
        with open(scenario_file, 'r') as f:
            scenarios.append(json.load(f))
    
    # Run evaluation (mock)
    results = {
        "timestamp": datetime.now().isoformat(),
        "scenarios_evaluated": len(scenarios),
        "results": [...]
    }
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = paths.get_evaluation_results_file(timestamp)
    
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    # Log completion
    log_file = paths.get_log_file("evaluation")
    with open(log_file, 'a') as f:
        f.write(f"{datetime.now()}: Evaluation completed, saved to {results_file}\n")
    
    print(f"Evaluation completed! Results saved to: {results_file}")

if __name__ == "__main__":
    run_evaluation()
```

This path management system ensures consistent, maintainable, and cross-platform file handling throughout the Mental Health LLM Evaluation project.