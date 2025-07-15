# Clean Structure Migration Guide

This guide explains the clean structure implementation for the Mental Health LLM Evaluation project.

## 🎯 Clean Structure Overview

The project has been reorganized into a clean, logical structure:

```
mental-health-llm-evaluation/
├── src/                      # All source code
│   ├── scenarios/           # Scenario generation/loading code
│   ├── models/             # Model interfaces
│   ├── evaluation/         # Evaluation metrics
│   ├── analysis/           # Statistical analysis
│   ├── config/             # Configuration handling
│   └── utils/              # Utility functions
├── data/                    # Static data only
│   └── scenarios/          # Scenario JSON templates
├── output/                  # All generated output
│   ├── conversations/      # Generated conversations
│   ├── evaluations/        # Evaluation results
│   ├── analysis/           # Statistical analysis results
│   └── visualizations/     # Charts and graphs
├── config/                 # Configuration files
└── scripts/                # Utility scripts
```

## 🔄 Migration Process

### 1. Run Migration Script

```bash
# Preview what will be done
python migrate_to_clean_structure.py --dry-run

# Run the migration
python migrate_to_clean_structure.py

# Skip backup (not recommended)
python migrate_to_clean_structure.py --no-backup
```

### 2. Verify Migration

```bash
# Run comprehensive verification
python verify_migration.py
```

### 3. Test Functionality

```bash
# Test main research script
python scripts/run_research.py --quick

# Test model comparison
python scripts/compare_models.py --quick
```

## 📁 Directory Purposes

### `src/` - Source Code
- **Purpose**: All Python source code
- **Contents**: Modules organized by functionality
- **Characteristics**: Version controlled, contains business logic

### `data/` - Static Data
- **Purpose**: Unchanging input data
- **Contents**: Scenario templates, reference files
- **Characteristics**: Version controlled, input to the system

### `output/` - Generated Output
- **Purpose**: All generated files and results
- **Contents**: Conversations, evaluations, analysis, visualizations
- **Characteristics**: Not version controlled, can be recreated

### `config/` - Configuration
- **Purpose**: Configuration files
- **Contents**: YAML configs, settings
- **Characteristics**: Version controlled, system configuration

## 🚀 Key Benefits

1. **Clear Separation**: Static data vs generated output
2. **Logical Organization**: Related files grouped together
3. **Scalable Structure**: Easy to add new components
4. **Clean Paths**: Intuitive file locations
5. **Better Collaboration**: Clear file purposes

## 🔧 Path Changes

### Old → New Path Mappings

| Old Path | New Path |
|----------|----------|
| `results/evaluations/` | `output/evaluations/` |
| `results/reports/` | `output/analysis/` |
| `results/statistics/` | `output/analysis/` |
| `results/visualizations/` | `output/visualizations/` |
| `generated/conversations/` | `output/conversations/` |
| `generated/temp/` | `output/temp/` |
| `generated/logs/` | `output/logs/` |

### Code Updates

The migration automatically updates:
- Path references in Python files
- Configuration schema
- Import statements
- Default directory settings

## 📋 Usage After Migration

### Running Evaluations

```bash
# All scripts work the same way
python scripts/run_research.py
python scripts/compare_models.py
python scripts/run_conversation_generation.py
```

### Output Locations

Results are now organized in `output/`:

```
output/
├── conversations/          # Generated conversations
├── evaluations/           # JSON evaluation results
│   ├── detailed_results.json
│   └── model_strengths.json
├── analysis/              # Analysis results and reports
│   ├── statistical_analysis.json
│   └── research_report.txt
└── visualizations/        # Charts and presentations
    ├── 1_overall_comparison.png
    └── presentation/
```

### Configuration

Updated configuration uses clean paths:

```yaml
# config/clean_structure.yaml
paths:
  data_dir: "./data"
  scenarios_dir: "./data/scenarios"
  output_dir: "./output"
  evaluations_dir: "./output/evaluations"
  analysis_dir: "./output/analysis"
  visualizations_dir: "./output/visualizations"
```

## 🛠️ Development

### Adding New Features

1. **Source code** → `src/` under appropriate module
2. **Static data** → `data/`
3. **Generated output** → `output/`
4. **Configuration** → `config/`

### Project Structure Standards

- **Module organization**: Group related functionality
- **Clear naming**: Descriptive directory names
- **Separation of concerns**: Static vs generated vs code
- **Documentation**: Update this guide for major changes

## 🔍 Verification

The migration includes comprehensive verification:

1. **Directory structure** check
2. **Key files** existence
3. **Import statements** validation
4. **Path references** update verification
5. **Configuration** correctness
6. **Data integrity** check
7. **Basic functionality** test

## 🧹 Cleanup

After successful migration:

```bash
# Remove backup (only after testing)
rm -rf backup_clean_YYYYMMDD_HHMMSS

# Remove migration files
rm migrate_to_clean_structure.py
rm verify_migration.py
rm test_clean_structure.py
```

## 📝 Git Integration

The migration updates `.gitignore` to:
- Ignore generated output files
- Preserve directory structure with `.gitkeep`
- Track configuration and source code

## 🚨 Troubleshooting

### Common Issues

1. **Import errors**: Check that `src/` is in Python path
2. **Path not found**: Verify migration completed successfully
3. **Old paths in code**: Run verification to find missed references
4. **Configuration errors**: Check `config/clean_structure.yaml`

### Recovery

If migration fails:
1. Restore from backup directory
2. Check error messages
3. Fix issues and retry migration

## 📊 Migration Checklist

- [ ] Run migration script
- [ ] Verify directory structure
- [ ] Test imports
- [ ] Run functionality tests
- [ ] Check output paths
- [ ] Verify configuration
- [ ] Test main scripts
- [ ] Clean up temporary files

## 🎓 Best Practices

1. **Always backup** before migration
2. **Test thoroughly** after migration
3. **Update documentation** for changes
4. **Use verification script** to catch issues
5. **Clean up** temporary files after success

## 📞 Support

If you encounter issues:
1. Check verification script output
2. Review error messages
3. Restore from backup if needed
4. Consult this guide for troubleshooting

The clean structure provides a solid foundation for continued development and maintenance of the Mental Health LLM Evaluation project.