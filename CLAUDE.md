# Mental Health LLM Evaluation Project Context

## Project Overview
This is a research project comparing local vs cloud-based LLMs for mental health telemedicine applications. The codebase is clean, organized, and research-ready.

## Critical Directory Structure (DO NOT ALTER)
mental-health-llm-evaluation/
├── config/                    # YAML/JSON configuration FILES only
├── data/                      # Static input data (scenarios)
├── results/                   # ALL outputs go here
│   ├── evaluations/          # Model evaluation results
│   ├── reports/              # Research reports
│   ├── statistics/           # Statistical analysis
│   ├── visualizations/       # Charts and graphs
│   ├── conversations/        # Generated conversations
│   └── development/          # Development/test outputs
├── src/
│   └── config/              # Configuration CODE (Python modules)
└── temp/                    # Temporary files only

## STRICT RULES - ALWAYS FOLLOW

### 1. File Creation Discipline
- ❌ NEVER create files in the root directory (except setup scripts)
- ❌ NEVER create new directories without explicit user request
- ❌ NEVER create markdown files for explanations - use terminal output
- ✅ Code changes go in appropriate src/ subdirectories
- ✅ Outputs go in results/ subdirectories
- ✅ Test outputs use --output results/development/

### 2. Project Hygiene
- NO "example_", "test_", "demo_" files in root
- NO creating multiple versions of files (file_v2.py, file_new.py)
- NO markdown documentation unless explicitly requested
- Use existing files - don't create new ones without clear need

### 3. When Working on Tasks
- First check if relevant code already exists before creating new files
- Modify existing files rather than creating alternatives
- If creating new functionality, ask WHERE it should go first
- Keep responses focused on the specific task

### 4. Directory Usage
- Development testing: Always use `--output results/development/`
- Never create output/, generated/, or test_results/ directories
- Temp files go in temp/ and should be cleaned up
- Configuration changes: Edit existing YAML files, don't create new ones

## Common Commands (Use These)
```bash
# Quick test
python scripts/compare_models.py --quick --output results/development/

# Full evaluation
python scripts/run_research.py

# System verification
python verify_system.py
Project State

Status: Clean, organized, and ready for research
Cleanup: Completed July 2025
Structure: Finalized - do not reorganize
Focus: Running evaluations and analyzing results

Response Guidelines

Be concise - avoid lengthy explanations in markdown
Show code/commands rather than describing them
Use existing scripts/structure rather than creating new ones
If unsure about file placement, ASK before creating
Respect the existing architecture - it's been carefully designed

What NOT to Do

Don't suggest reorganizing the structure (it's final)
Don't create "helper" or "utility" scripts without clear need
Don't generate example/demo files
Don't create visualization scripts (use existing ones)
Don't add new dependencies without discussing first

Current Research Focus
The project is ready for:

Running model comparisons
Generating evaluation data
Statistical analysis
Report generation

Keep all work aligned with these research goals.

### Create `.claude/project_rules.yaml`:
```yaml
# Claude CLI Project Rules
project: mental-health-llm-evaluation
version: 1.0

file_creation:
  forbidden_patterns:
    - "test_*.py"  # In root directory
    - "example_*.py"
    - "demo_*.py"
    - "*.md"  # Unless explicitly requested
    - "*_backup.*"
    - "*_old.*"
    - "*_v2.*"
  
  allowed_locations:
    code: "src/"
    tests: "tests/"
    scripts: "scripts/"
    outputs: "results/"
    temp: "temp/"

behavior:
  - ask_before_creating_files: true
  - prefer_modification_over_creation: true
  - use_existing_structure: true
  - avoid_markdown_explanations: true
  - keep_root_directory_clean: true

output_rules:
  development: "results/development/"
  production: "results/"
  never_create: ["output/", "generated/", "test_results/"]

response_style:
  - concise: true
  - code_over_explanation: true
  - respect_existing_architecture: true
Quick Setup Script for Future Sessions:
bash#!/bin/bash
# save as: check_project_health.sh

echo "🏥 Mental Health LLM Project Health Check"
echo "========================================"

# Check for unwanted files in root
echo "Checking root directory cleanliness..."
unwanted=$(ls -la | grep -E "(test_|example_|demo_|backup_|_old|_v2)" | wc -l)
if [ $unwanted -eq 0 ]; then
    echo "✅ Root directory is clean"
else
    echo "⚠️  Found $unwanted unwanted files in root"
fi

# Check directory structure
echo -e "\nVerifying directory structure..."
dirs=("config" "data" "results" "src" "scripts")
for dir in "${dirs[@]}"; do
    if [ -d "$dir" ]; then
        echo "✅ $dir/"
    else
        echo "❌ Missing: $dir/"
    fi
done

# Check for forbidden directories
echo -e "\nChecking for forbidden directories..."
forbidden=("output" "generated" "test_results")
for dir in "${forbidden[@]}"; do
    if [ -d "$dir" ]; then
        echo "❌ Found forbidden: $dir/ (should be removed)"
    fi
done

echo -e "\n✨ Health check complete!"