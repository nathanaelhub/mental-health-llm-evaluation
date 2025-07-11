# Mental Health LLM Evaluation - src/ Dependency Graph

## Core Entry Points and Dependencies

```
scripts/run_research.py (MAIN ENTRY POINT)
├── src.evaluation.mental_health_evaluator.MentalHealthEvaluator
├── src.analysis.statistical_analysis.analyze_results
├── src.analysis.statistical_analysis.generate_summary_report
├── src.analysis.statistical_analysis.identify_model_strengths
├── src.analysis.visualization.create_all_visualizations
└── src.analysis.visualization.create_presentation_slides

src/analysis/compare_models.py (STANDALONE TOOL)
├── Direct imports from models/
├── Direct imports from evaluation/
└── Can run independently

src/analysis/quick_comparison.py (SIMPLE STANDALONE)
├── src.models.openai_client.OpenAIClient
├── src.models.deepseek_client.DeepSeekClient
└── src.evaluation.evaluation_metrics.EvaluationMetrics
```

## Module Dependencies (from __init__.py files)

```
src/__init__.py DECLARES:
├── .models: OpenAIClient, DeepSeekClient, BaseModel
├── .evaluation: CompositeScorer, TechnicalMetrics, TherapeuticMetrics
├── .scenarios: ConversationGenerator, ScenarioLoader
├── .analysis: StatisticalAnalyzer, ResultsVisualizer
└── .utils: setup_logging, DataStorage

src/models/__init__.py DECLARES:
├── .base_model: BaseModel, ModelResponse
└── .local_llm_client: LocalLLMClient
   (NOTE: Missing OpenAIClient, DeepSeekClient from main __init__.py)

src/analysis/__init__.py DECLARES:
├── .statistical_analysis: StatisticalAnalyzer, StatisticalResults
└── .visualization: ResultsVisualizer, VisualizationConfig
```

## Reference Count Analysis

**HIGH USAGE (Critical):**
- visualization.py: 125 references
- statistical_analysis.py: 29 references  
- compare_models.py: 17 references
- openai_client.py: 10 references
- deepseek_client.py: 10 references

**MEDIUM USAGE:**
- batch_processor.py: 5 references
- conversation_manager.py: 6 references
- error_handler.py: 7 references
- mental_health_evaluator.py: 5 references
- safety_monitor.py: 4 references

**LOW/NO USAGE:**
- branching_engine.py: 0 references
- conversation_logger.py: 0 references
- metrics_collector.py: 0 references
- advanced_visualization.py: 0 references
- data_loader.py: 0 references

## Standalone Scripts (with __main__)

```
✅ USEFUL STANDALONE TOOLS:
├── src/analysis/compare_models.py (CLI comparison tool)
├── src/analysis/quick_comparison.py (Simple comparison)
├── src/evaluation/mental_health_evaluator.py (Can run standalone)
└── src/models/claude_usage_monitor.py (Monitoring tool)

❓ QUESTIONABLE:
└── src/evaluation/evaluation_metrics.py (Unclear purpose)
```