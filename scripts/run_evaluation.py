#!/usr/bin/env python3
"""
Main evaluation script for mental health LLM comparison.

This script provides a command-line interface for running comprehensive
evaluations of local vs cloud-based mental health LLM models.
"""

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

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
    """Main evaluation function."""
    parser = argparse.ArgumentParser(
        description="Mental Health LLM Evaluation Tool"
    )
    parser.add_argument(
        "--models", 
        nargs="+", 
        default=["openai", "deepseek"],
        help="Models to evaluate (openai, deepseek)"
    )
    parser.add_argument(
        "--scenarios", 
        default="comprehensive",
        help="Scenario suite to use (basic, comprehensive, safety, empathy)"
    )
    parser.add_argument(
        "--output-dir", 
        default="./output",
        help="Output directory for results"
    )
    parser.add_argument(
        "--config-file",
        help="Configuration file path"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        help="Logging level (DEBUG, INFO, WARNING, ERROR)"
    )
    parser.add_argument(
        "--generate-visualizations",
        action="store_true",
        help="Generate visualization outputs"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(log_level=args.log_level)
    
    # Load configuration
    config = {}
    if args.config_file and os.path.exists(args.config_file):
        with open(args.config_file, 'r') as f:
            config = json.load(f)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    print("🧠 Mental Health LLM Evaluation Tool")
    print("=" * 40)
    
    # Initialize models
    models = []
    print(f"Initializing models: {args.models}")
    
    for model_name in args.models:
        if model_name.lower() == "openai":
            try:
                model = OpenAIClient(config.get("openai_config", {}))
                if await model.health_check():
                    models.append(model)
                    print(f"✅ OpenAI GPT-4 initialized")
                else:
                    print(f"❌ OpenAI GPT-4 health check failed")
            except Exception as e:
                print(f"❌ Failed to initialize OpenAI: {e}")
        
        elif model_name.lower() == "deepseek":
            try:
                model = DeepSeekClient(config.get("deepseek_config", {}))
                if await model.health_check():
                    models.append(model)
                    print(f"✅ DeepSeek initialized")
                else:
                    print(f"❌ DeepSeek health check failed")
            except Exception as e:
                print(f"❌ Failed to initialize DeepSeek: {e}")
    
    if not models:
        print("❌ No models successfully initialized. Exiting.")
        return
    
    # Load scenarios
    print(f"\nLoading scenarios: {args.scenarios}")
    loader = ScenarioLoader()
    scenarios = loader.get_evaluation_suite(args.scenarios)
    print(f"✅ Loaded {len(scenarios)} scenarios")
    
    # Generate evaluation data
    print("\nGenerating evaluation data...")
    
    # Technical prompts
    technical_prompts = [
        "Hello, how are you?",
        "Can you help me with my mental health?",
        "I'm feeling anxious about work.",
        "What should I do when I feel overwhelmed?",
        "Can you explain mindfulness techniques?"
    ]
    
    # Generate conversations
    generator = ConversationGenerator()
    print("Generating sample conversations...")
    conversations = await generator.generate_batch_conversations(
        models, scenarios[:3], conversations_per_scenario=1
    )
    conversation_data = [conv.to_dict() for conv in conversations]
    
    # Prepare evaluation data
    evaluation_data = {
        "technical_prompts": technical_prompts,
        "therapeutic_scenarios": scenarios,
        "conversation_data": conversation_data,
        "user_feedback": None  # Could be loaded from file
    }
    
    # Run evaluation
    print("\n🔬 Running comprehensive evaluation...")
    scorer = CompositeScorer(config.get("evaluation_config", {}))
    
    results = await scorer.compare_models(models, evaluation_data)
    
    # Display results
    print("\n📊 Evaluation Results")
    print("=" * 40)
    
    for model_name, result in results.items():
        print(f"\n{model_name}:")
        print(f"  Overall Score: {result.overall_score:.1f}/100")
        print(f"  Technical: {result.technical_score:.1f}/100")
        print(f"  Therapeutic: {result.therapeutic_score:.1f}/100")
        print(f"  Patient Experience: {result.patient_score:.1f}/100")
    
    # Statistical analysis
    print("\n📈 Statistical Analysis")
    print("=" * 40)
    
    analyzer = StatisticalAnalyzer()
    # Convert results to format expected by analyzer
    analysis_results = {
        name: [result] for name, result in results.items()
    }
    statistical_results = analyzer.analyze_model_comparison(analysis_results)
    
    print("Statistical significance tests completed")
    for rec in statistical_results.recommendations[:3]:
        print(f"  • {rec}")
    
    # Save results
    print(f"\n💾 Saving results to {args.output_dir}")
    
    storage = DataStorage("file", {"base_dir": args.output_dir})
    data_manager = EvaluationDataManager(storage)
    
    session_id = f"eval_{int(asyncio.get_event_loop().time())}"
    
    for model_name, result in results.items():
        data_manager.save_evaluation_results(
            result.to_dict(), session_id, model_name
        )
    
    # Save statistical analysis
    data_manager.save_analysis_results(
        statistical_results.to_dict(),
        f"{session_id}_statistical",
        "statistical_analysis"
    )
    
    # Generate visualizations
    if args.generate_visualizations:
        print("\n🎨 Generating visualizations...")
        
        visualizer = ResultsVisualizer()
        viz_dir = os.path.join(args.output_dir, "visualizations")
        
        generated_files = visualizer.create_comprehensive_dashboard(
            analysis_results, statistical_results, viz_dir
        )
        
        print(f"Generated {len(generated_files)} visualizations:")
        for name, path in generated_files.items():
            print(f"  • {name}: {path}")
    
    # Export CSV summary
    data_manager.export_to_csv(
        os.path.join(args.output_dir, "evaluation_summary.csv"),
        "evaluations"
    )
    
    print(f"\n✅ Evaluation complete! Results saved to {args.output_dir}")
    
    # Show best performing model
    best_model = max(results.items(), key=lambda x: x[1].overall_score)
    print(f"\n🏆 Best performing model: {best_model[0]} ({best_model[1].overall_score:.1f}/100)")


if __name__ == "__main__":
    asyncio.run(main())