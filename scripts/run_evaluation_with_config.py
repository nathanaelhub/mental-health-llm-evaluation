#!/usr/bin/env python3
"""
Enhanced evaluation script using the new configuration system.

This script demonstrates how to use the comprehensive configuration system
for running mental health LLM evaluations with proper configuration management.
"""

import argparse
import asyncio
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_loader import get_config, ConfigLoader, validate_environment
from models.openai_client import OpenAIClient
from models.deepseek_client import DeepSeekClient
from evaluation.composite_scorer import CompositeScorer
from scenarios.scenario_loader import ScenarioLoader
from scenarios.conversation_generator import ConversationGenerator
from analysis.statistical_analysis import StatisticalAnalyzer
from analysis.visualization import ResultsVisualizer
from utils.logging_config import setup_logging
from utils.data_storage import DataStorage, EvaluationDataManager


async def main():
    """Main evaluation function with configuration system."""
    parser = argparse.ArgumentParser(
        description="Mental Health LLM Evaluation with Configuration System"
    )
    parser.add_argument(
        "--environment",
        default="development",
        choices=["development", "production", "testing"],
        help="Environment configuration to use"
    )
    parser.add_argument(
        "--config-file",
        help="Specific configuration file to use"
    )
    parser.add_argument(
        "--models",
        nargs="+",
        help="Override models to evaluate (openai, deepseek)"
    )
    parser.add_argument(
        "--scenario-suite",
        help="Override scenario suite to use"
    )
    parser.add_argument(
        "--conversation-count",
        type=int,
        help="Override number of conversations"
    )
    parser.add_argument(
        "--output-dir",
        help="Override output directory"
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Only validate configuration, don't run evaluation"
    )
    parser.add_argument(
        "--show-config",
        action="store_true",
        help="Show configuration summary"
    )
    
    args = parser.parse_args()
    
    print("🧠 Mental Health LLM Evaluation (with Configuration System)")
    print("=" * 60)
    
    # Load and validate configuration
    print(f"📋 Loading configuration for environment: {args.environment}")
    
    try:
        # Create configuration overrides from command line arguments
        overrides = {}
        if args.models:
            overrides["models"] = {"enabled_models": args.models}
        if args.scenario_suite:
            overrides["experiment"] = {"scenario_suite": args.scenario_suite}
        if args.conversation_count:
            if "experiment" not in overrides:
                overrides["experiment"] = {}
            overrides["experiment"]["conversation_count"] = args.conversation_count
        if args.output_dir:
            overrides["output"] = {"base_dir": args.output_dir}
        
        # Load configuration
        loader = ConfigLoader()
        config = loader.load_config(
            environment=args.environment,
            config_file=args.config_file,
            override_config=overrides
        )
        
        print("✅ Configuration loaded successfully")
        
    except Exception as e:
        print(f"❌ Failed to load configuration: {e}")
        return 1
    
    # Show configuration summary if requested
    if args.show_config:
        print("\n📄 Configuration Summary")
        print("-" * 30)
        summary = loader.create_config_summary()
        for key, value in summary.items():
            print(f"  {key}: {value}")
    
    # Validate environment
    print("\n🔐 Validating environment...")
    if not validate_environment():
        print("❌ Environment validation failed")
        return 1
    
    print("✅ Environment validation passed")
    
    # Setup logging from configuration
    logging_config = loader.get_logging_config()
    setup_logging(
        log_level=logging_config["level"],
        log_format=logging_config["format"],
        log_file=logging_config["file_path"],
        enable_console=logging_config["enable_console"],
        enable_structured=logging_config["enable_structured"]
    )
    
    print("✅ Logging configured")
    
    # If validate-only mode, exit here
    if args.validate_only:
        print("\n🎉 Configuration validation complete")
        return 0
    
    # Initialize models based on configuration
    print(f"\n🤖 Initializing models: {[m.value for m in config.models.enabled_models]}")
    
    models = []
    
    for model_type in config.models.enabled_models:
        try:
            if model_type.value == "openai":
                # Get OpenAI configuration
                openai_config = config.models.openai
                model = OpenAIClient({
                    "model": openai_config.model,
                    "temperature": openai_config.temperature,
                    "max_tokens": openai_config.max_tokens,
                    "timeout": openai_config.timeout,
                    "max_retries": openai_config.max_retries
                })
                
                # Validate model
                if await model.health_check():
                    models.append(model)
                    print(f"✅ OpenAI model initialized and validated")
                else:
                    print(f"❌ OpenAI model health check failed")
                    
            elif model_type.value == "deepseek":
                # Get DeepSeek configuration
                deepseek_config = config.models.deepseek
                model = DeepSeekClient({
                    "use_api": deepseek_config.use_api,
                    "model_path": deepseek_config.model_path,
                    "device": deepseek_config.device,
                    "temperature": deepseek_config.temperature,
                    "max_new_tokens": deepseek_config.max_new_tokens,
                    "timeout": deepseek_config.timeout
                })
                
                # Validate model
                if await model.health_check():
                    models.append(model)
                    print(f"✅ DeepSeek model initialized and validated")
                else:
                    print(f"❌ DeepSeek model health check failed")
                    
        except Exception as e:
            print(f"❌ Failed to initialize {model_type.value}: {e}")
    
    if not models:
        print("❌ No models successfully initialized")
        return 1
    
    print(f"✅ Successfully initialized {len(models)} models")
    
    # Load scenarios based on configuration
    print(f"\n📚 Loading scenarios: {config.experiment.scenario_suite}")
    
    try:
        scenario_loader = ScenarioLoader(
            scenarios_dir=config.scenario.scenarios_dir
        )
        scenarios = scenario_loader.get_evaluation_suite(config.experiment.scenario_suite)
        
        if not scenarios:
            # Load all scenarios if suite is empty
            scenarios = scenario_loader.load_scenarios()
        
        print(f"✅ Loaded {len(scenarios)} scenarios")
        
    except Exception as e:
        print(f"❌ Failed to load scenarios: {e}")
        return 1
    
    # Generate conversations based on configuration
    print(f"\n💬 Generating conversations...")
    print(f"  Conversations per scenario: {config.experiment.conversations_per_scenario}")
    print(f"  Total conversations: {len(scenarios) * config.experiment.conversations_per_scenario * len(models)}")
    
    try:
        conversation_generator = ConversationGenerator({
            "max_turns": config.conversation.max_turns,
            "min_turns": config.conversation.min_turns,
            "conversation_timeout": config.conversation.conversation_timeout,
            "user_response_probability": config.conversation.user_response_probability
        })
        
        # Generate subset of conversations for testing
        test_scenarios = scenarios[:min(3, len(scenarios))]
        conversations = await conversation_generator.generate_batch_conversations(
            models, 
            test_scenarios, 
            config.experiment.conversations_per_scenario
        )
        
        conversation_data = [conv.to_dict() for conv in conversations]
        print(f"✅ Generated {len(conversations)} conversations")
        
    except Exception as e:
        print(f"❌ Failed to generate conversations: {e}")
        return 1
    
    # Prepare evaluation data
    evaluation_data = {
        "technical_prompts": [
            "Hello, how are you?",
            "Can you help me with anxiety?",
            "I'm feeling overwhelmed.",
            "What are some coping strategies?",
            "Thank you for your help."
        ],
        "therapeutic_scenarios": scenarios,
        "conversation_data": conversation_data,
        "user_feedback": None
    }
    
    # Run evaluation with configuration
    print(f"\n🔬 Running comprehensive evaluation...")
    
    try:
        # Create composite scorer with configuration
        evaluation_config = {
            "technical_config": {
                "max_response_time_ms": config.evaluation.technical.max_response_time_ms,
                "target_throughput_rps": config.evaluation.technical.target_throughput_rps,
                "concurrent_requests": config.evaluation.technical.concurrent_requests,
                "score_weights": {
                    "response_time": config.evaluation.technical.response_time_weight,
                    "throughput": config.evaluation.technical.throughput_weight,
                    "reliability": config.evaluation.technical.reliability_weight,
                    "efficiency": config.evaluation.technical.efficiency_weight
                }
            },
            "therapeutic_config": {
                "score_weights": {
                    "empathy": config.evaluation.therapeutic.empathy_weight,
                    "coherence": config.evaluation.therapeutic.coherence_weight,
                    "safety": config.evaluation.therapeutic.safety_weight,
                    "boundaries": config.evaluation.therapeutic.boundaries_weight
                }
            },
            "patient_config": {
                "score_weights": {
                    "satisfaction": config.evaluation.patient.satisfaction_weight,
                    "engagement": config.evaluation.patient.engagement_weight,
                    "trust": config.evaluation.patient.trust_weight,
                    "accessibility": config.evaluation.patient.accessibility_weight
                }
            },
            "composite_weights": {
                "technical": config.evaluation.technical_weight,
                "therapeutic": config.evaluation.therapeutic_weight,
                "patient": config.evaluation.patient_weight
            },
            "score_thresholds": {
                "production_ready": config.evaluation.production_ready_threshold,
                "clinical_ready": config.evaluation.clinical_ready_threshold,
                "research_acceptable": config.evaluation.research_acceptable_threshold,
                "minimum_viable": config.evaluation.minimum_viable_threshold
            }
        }
        
        scorer = CompositeScorer(evaluation_config)
        results = await scorer.compare_models(models, evaluation_data)
        
        print("✅ Evaluation completed successfully")
        
    except Exception as e:
        print(f"❌ Evaluation failed: {e}")
        return 1
    
    # Display results
    print(f"\n📊 Evaluation Results")
    print("=" * 40)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Overall Score: {result.overall_score:.1f}/100")
        print(f"  Technical: {result.technical_score:.1f}/100")
        print(f"  Therapeutic: {result.therapeutic_score:.1f}/100")
        print(f"  Patient Experience: {result.patient_score:.1f}/100")
        
        # Show readiness assessment
        readiness = scorer.get_readiness_assessment(result)
        print(f"  Readiness: {readiness['status']}")
    
    # Statistical analysis
    print(f"\n📈 Statistical Analysis")
    print("=" * 40)
    
    try:
        analyzer = StatisticalAnalyzer({
            "alpha": config.statistical.alpha,
            "confidence_level": config.statistical.confidence_level,
            "min_sample_size": config.statistical.min_sample_size
        })
        
        # Convert results for analysis
        analysis_results = {name: [result] for name, result in results.items()}
        statistical_results = analyzer.analyze_model_comparison(analysis_results)
        
        print("✅ Statistical analysis completed")
        for rec in statistical_results.recommendations[:3]:
            print(f"  • {rec}")
            
    except Exception as e:
        print(f"❌ Statistical analysis failed: {e}")
        statistical_results = None
    
    # Save results using configured storage
    print(f"\n💾 Saving results...")
    
    try:
        storage = DataStorage(
            storage_type=config.storage.type.value,
            config={"base_dir": config.storage.base_dir}
        )
        data_manager = EvaluationDataManager(storage)
        
        session_id = f"eval_{int(asyncio.get_event_loop().time())}"
        
        # Save evaluation results
        for model_name, result in results.items():
            data_manager.save_evaluation_results(
                result.to_dict(), session_id, model_name
            )
        
        # Save statistical analysis if available
        if statistical_results:
            data_manager.save_analysis_results(
                statistical_results.to_dict(),
                f"{session_id}_statistical",
                "statistical_analysis"
            )
        
        print(f"✅ Results saved to: {config.storage.base_dir}")
        
    except Exception as e:
        print(f"❌ Failed to save results: {e}")
    
    # Generate visualizations if configured
    if config.output.generate_visualizations:
        print(f"\n🎨 Generating visualizations...")
        
        try:
            visualizer = ResultsVisualizer()
            viz_dir = f"{config.output.visualizations_dir}"
            
            analysis_results = {name: [result] for name, result in results.items()}
            generated_files = visualizer.create_comprehensive_dashboard(
                analysis_results, statistical_results, viz_dir
            )
            
            print(f"✅ Generated {len(generated_files)} visualizations in: {viz_dir}")
            
        except Exception as e:
            print(f"❌ Visualization generation failed: {e}")
    
    # Show best performing model
    if results:
        best_model = max(results.items(), key=lambda x: x[1].overall_score)
        print(f"\n🏆 Best performing model: {best_model[0]} ({best_model[1].overall_score:.1f}/100)")
    
    print(f"\n🎉 Evaluation complete!")
    print(f"Configuration: {args.environment}")
    print(f"Models evaluated: {len(models)}")
    print(f"Scenarios tested: {len(scenarios)}")
    print(f"Total conversations: {len(conversation_data)}")
    
    return 0


if __name__ == "__main__":
    asyncio.run(main())