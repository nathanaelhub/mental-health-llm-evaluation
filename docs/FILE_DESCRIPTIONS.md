# File Descriptions - Mental Health LLM Evaluation

*Quick reference for what each file actually does (no corporate BS)*

## 🚀 Main Entry Points

### scripts/run_research.py
The BIG KAHUNA. This runs your entire research pipeline - generates conversations, evaluates them, does stats, makes pretty charts. This is what you run for your capstone demo.

### scripts/compare_models.py  
The head-to-head comparison tool. Feed it a prompt, it runs both models and tells you which one did better and why. Great for understanding specific differences.

### scripts/run_conversation_generation.py
Generates therapeutic conversations from your scenarios. Creates the raw data that everything else analyzes.

### scripts/validate_cleanup.py
System validation script. Tests all imports, API connectivity, and core functionality. Run this after any major changes to make sure everything still works.

## 📁 Source Code (src/)

### models/
- **base_model.py** - The parent class all models inherit from. Sets the rules.
- **openai_client.py** - Talks to GPT-4. Handles the OpenAI API stuff.
- **claude_client.py** - Talks to Anthropic's Claude. Supports Opus, Sonnet, Haiku models.
- **deepseek_client.py** - Talks to DeepSeek (your local model). Can do cloud or local.
- **gemma_client.py** - Talks to Google's Gemma-3-12b running locally.
- **local_llm_client.py** - Generic wrapper for any local LLM endpoint.

### evaluation/
- **mental_health_evaluator.py** - The judge. Scores conversations on therapeutic quality.
- **evaluation_metrics.py** - The scoring rubric (empathy, safety, clarity, therapeutic value).
- **composite_scorer.py** - Combines all the scores into final verdicts.

### analysis/
- **statistical_analysis.py** - Does the math. T-tests, p-values, effect sizes, all that jazz.
- **visualization.py** - Makes the pretty charts for your presentation. 
- **compare_models.py** - CLI tool for quick model comparisons (wait, this might be redundant with tools/?)

### config/
- **config_loader.py** - Reads YAML files and turns them into Python objects.
- **config_schema.py** - Defines what configs should look like (validation).
- **config_utils.py** - Helper functions for config stuff.

### scenarios/
- **scenario_loader.py** - Loads test scenarios from YAML files.
- **scenario.py** - The data structure for a scenario.
- **conversation_generator.py** - Takes scenarios and makes them into conversations.

### utils/
- **data_storage.py** - Saves and loads JSON files. That's it.
- **logging_config.py** - Sets up logging so you know what's happening.

## 📄 Configuration Files

### config/main.yaml
The master config. Everything starts here. Sets models, scenarios, evaluation params.

### config/models/
- **model_settings.yaml** - Model-specific settings (temperature, tokens, etc.)
- **local_experiment.yaml** - Config for running local model experiments.

### config/scenarios/
All the mental health scenarios organized by type:
- **anxiety_scenarios.yaml** - Anxiety-related conversations
- **depression_scenarios.yaml** - Depression scenarios  
- **crisis_scenarios.yaml** - High-risk situations (suicide, self-harm)
- **stress_scenarios.yaml** - General stress/overwhelm
- **substance_use_scenarios.yaml** - Addiction-related
- **general_mental_health_scenarios.yaml** - Catch-all
- **main_scenarios.yaml** - The main list used by default
- **scenario_index.yaml** - Index of all scenarios

## 📊 Data & Output

### data/
- **conversations/** - Generated conversation JSON files
- **results/** - Evaluation results JSON files  
- **scenarios/** - Additional scenario files (if any)

### output/
Where all your results go:
- **visualizations/** - The 5 main research charts
- **presentation/** - Ready-to-use slides
- **detailed_results.json** - Raw evaluation data
- **statistical_analysis.json** - Stats results
- **model_strengths.json** - What each model is good at
- **research_report.txt** - Human-readable summary

## 📚 Documentation

### README.md
The professional docs. Installation, methodology, the works.

### FILE_DESCRIPTIONS.md  
This file. The casual "what does this do?" guide.

### docs/
- **methodology.md** - How the evaluation works
- **results_interpretation.md** - How to read the results
- **dependency_analysis.md** - What libraries we use and why

## 🔧 Other Stuff

### requirements.txt
All the Python packages you need. Keep it simple.

### .env.example
Template for environment variables (mainly OPENAI_API_KEY).

### .gitignore
Tells git what to ignore (venv, cache, keys, etc.)

---

## Quick Start Commands

```bash
# Run full research pipeline
python scripts/run_research.py

# Quick test with 3 scenarios  
python scripts/run_research.py --quick

# Compare models on specific prompt
python scripts/compare_models.py --prompt "I'm feeling anxious"

# Generate conversations only
python scripts/run_conversation_generation.py

# Validate system after changes
python scripts/validate_cleanup.py
```

## File Count Summary
- **Total Python files**: ~28
- **Config files**: ~15  
- **Documentation**: ~6
- **Total project files**: ~48 (excluding venv/git/output)

That's it! No bloat, no mystery files, just what you need for your capstone.