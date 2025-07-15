# Project Organization

This document explains the reorganized directory structure for the Mental Health LLM Evaluation project.

## 🎯 Goals

The restructuring eliminates confusion between different types of files and provides:
- **Clear separation** between static data, generated data, and results
- **Logical organization** that scales with project growth
- **Consistent naming** that makes the purpose of each directory obvious
- **Easy maintenance** and collaboration

## 📁 Directory Structure

```
mental-health-llm-evaluation/
├── 📊 data/                       # Static input data
│   └── scenarios/                 # Scenario definitions (JSON files)
├── 🔧 src/                        # Source code
│   ├── analysis/                  # Analysis and visualization code
│   ├── config/                    # Configuration handling
│   ├── evaluation/                # Evaluation logic
│   ├── models/                    # Model clients and interfaces
│   ├── scenarios/                 # Scenario loading and management
│   └── utils/                     # Utility functions
├── ⚙️ config/                     # Configuration files
│   ├── main.yaml                  # Main configuration
│   ├── models/                    # Model-specific configurations
│   └── scenarios/                 # Detailed scenario definitions (YAML)
├── 🏗️ generated/                  # Generated/temporary data
│   ├── conversations/             # Generated conversations
│   ├── temp/                      # Temporary files
│   └── logs/                      # Log files
├── 📈 results/                    # Analysis results and outputs
│   ├── evaluations/               # Evaluation results (JSON)
│   ├── reports/                   # Text reports
│   ├── statistics/                # Statistical analysis
│   └── visualizations/            # Charts and graphs
│       ├── charts/                # Individual charts
│       └── presentation/          # Presentation slides
├── 🔨 scripts/                    # Utility scripts
├── 📚 docs/                       # Documentation
└── 🧪 tests/                      # Test files
```

## 🔍 Directory Purposes

### 📊 `data/` - Static Input Data
Contains **unchanging input data** that feeds into the evaluation process:
- `scenarios/`: JSON files defining evaluation scenarios
- Future: Other static datasets, reference materials

**Characteristics:**
- Version controlled
- Rarely changes
- Input to the system

### 🏗️ `generated/` - Generated/Temporary Data  
Contains **data created during execution** that may change between runs:
- `conversations/`: Generated conversations between models and simulated patients
- `temp/`: Temporary files created during processing
- `logs/`: Application logs

**Characteristics:**
- Not version controlled (in .gitignore)
- Changes frequently
- Can be recreated by running the system

### 📈 `results/` - Analysis Results and Outputs
Contains **final outputs** from evaluation and analysis:
- `evaluations/`: JSON files with detailed evaluation results
- `reports/`: Human-readable text reports
- `statistics/`: Statistical analysis results
- `visualizations/`: Charts, graphs, and presentation materials

**Characteristics:**
- Important outputs to be shared
- May be version controlled (selectively)
- Represents the "deliverables" of the project

### 🔧 `src/` - Source Code
Contains **application logic** organized by functionality:
- `analysis/`: Data analysis and visualization code
- `config/`: Configuration loading and validation
- `evaluation/`: Core evaluation logic
- `models/`: Model clients and interfaces
- `scenarios/`: Scenario loading and management
- `utils/`: Shared utility functions

### ⚙️ `config/` - Configuration Files
Contains **configuration data** for different aspects of the system:
- `main.yaml`: Main configuration file
- `models/`: Model-specific configurations
- `scenarios/`: Detailed scenario definitions (YAML format)

## 🚀 Migration Guide

### Running the Migration

1. **Backup First** (automatic):
   ```bash
   python migrate_structure.py
   ```

2. **Dry Run** (optional):
   ```bash
   python migrate_structure.py --dry-run
   ```

3. **Full Migration**:
   ```bash
   python migrate_structure.py --project-root .
   ```

### What the Migration Does

1. **Creates backup** of your current structure
2. **Creates new directories** with proper structure
3. **Moves files** from old locations to new locations:
   - `output/` → `results/`
   - `data/results/` → `results/evaluations/` (if used)
   - `data/conversations/` → `generated/conversations/`
4. **Updates code references** in Python files
5. **Updates configuration files** to use new paths
6. **Creates .gitkeep files** to preserve empty directories
7. **Updates .gitignore** for new structure
8. **Cleans up** old directories

### File Mapping

| Old Location | New Location |
|--------------|--------------|
| `output/detailed_results.json` | `results/evaluations/detailed_results.json` |
| `output/statistical_analysis.json` | `results/statistics/statistical_analysis.json` |
| `output/research_report.txt` | `results/reports/research_report.txt` |
| `output/visualizations/` | `results/visualizations/charts/` |
| `output/presentation/` | `results/visualizations/presentation/` |
| `data/results/` | `results/evaluations/` |
| `data/conversations/` | `generated/conversations/` |

## 📝 Usage After Migration

### Running Evaluations

The main entry points remain the same but now use the new structure:

```bash
# Run full evaluation
python scripts/run_research.py

# Run conversation generation
python scripts/run_conversation_generation.py

# Compare models
python scripts/compare_models.py
```

### Output Locations

After migration, outputs will be organized as follows:

```bash
results/
├── evaluations/           # JSON evaluation results
│   ├── detailed_results.json
│   └── model_strengths.json
├── reports/               # Text reports
│   └── research_report.txt
├── statistics/            # Statistical analysis
│   └── statistical_analysis.json
└── visualizations/        # Charts and presentations
    ├── charts/            # Individual charts
    │   ├── 1_overall_comparison.png
    │   ├── 2_category_radar.png
    │   └── ...
    └── presentation/      # Presentation slides
        ├── slide_1_executive_summary.png
        └── ...
```

### Configuration Updates

The configuration schema has been updated to reflect the new structure:

```python
# Old
output_dir = "output"
results_dir = "data/results"

# New  
results_dir = "results"
evaluations_dir = "results/evaluations"
reports_dir = "results/reports"
statistics_dir = "results/statistics"
visualizations_dir = "results/visualizations"
```

## 🔧 Development

### Adding New Features

When adding new features, follow these guidelines:

1. **Source code** goes in `src/` under the appropriate module
2. **Configuration** goes in `config/` 
3. **Static data** goes in `data/`
4. **Generated data** goes in `generated/`
5. **Results** go in `results/` with appropriate subdirectory

### Testing

The `tests/` directory is available for unit tests and integration tests.

### Documentation

Update documentation in the `docs/` directory when adding new features.

## 🔄 Rollback

If you need to rollback the migration:

1. The migration script creates a timestamped backup in `backup_YYYYMMDD_HHMMSS/`
2. You can restore from this backup manually
3. Or create a rollback script if needed

## 📞 Support

If you encounter issues with the migration:

1. Check the backup directory for your original files
2. Review the migration logs for any errors
3. Test individual components to isolate issues
4. Update any custom scripts you may have written

## 🎉 Benefits

After migration, you'll have:

- ✅ **Clear separation** of concerns
- ✅ **Logical organization** that's easy to navigate
- ✅ **Consistent naming** across the project
- ✅ **Scalable structure** for future growth
- ✅ **Better collaboration** with clear file purposes
- ✅ **Easier maintenance** and debugging