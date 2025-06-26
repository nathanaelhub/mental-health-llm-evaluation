#!/usr/bin/env python3
"""
Mental Health LLM Evaluation - Conversation Generation System

Main entry point for running the complete conversation generation system
with support for generating exactly 20 conversations per scenario per model
as specified in the milestone requirements.

Usage:
    python run_conversation_generation.py --help
    python run_conversation_generation.py --models openai deepseek --scenarios all
    python run_conversation_generation.py --models openai --scenarios anxiety_mild depression_moderate --output ./results
"""

import asyncio
import argparse
import logging
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent / "src"))

from src.conversation.batch_processor import BatchProcessor, BatchConfig, run_batch_evaluation
from src.models.openai_client import OpenAIClient
from src.models.deepseek_client import DeepSeekClient
from src.scenarios.scenario import ScenarioLoader
from src.utils.logging_config import setup_logging


def setup_argument_parser() -> argparse.ArgumentParser:
    """Setup command line argument parser."""
    
    parser = argparse.ArgumentParser(
        description="Mental Health LLM Evaluation - Conversation Generation System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with both OpenAI and DeepSeek models on all scenarios
  python run_conversation_generation.py --models openai deepseek --scenarios all
  
  # Run with OpenAI only on specific scenarios
  python run_conversation_generation.py --models openai --scenarios anxiety_mild depression_moderate
  
  # Run with custom settings
  python run_conversation_generation.py --models deepseek --conversations-per-scenario 10 --concurrent 3
  
  # Run with detailed logging and custom output directory
  python run_conversation_generation.py --models openai deepseek --verbose --output ./my_results
        """
    )
    
    # Model selection
    parser.add_argument(
        "--models",
        nargs="+",
        choices=["openai", "deepseek"],
        default=["openai", "deepseek"],
        help="Models to evaluate (default: openai deepseek)"
    )
    
    # Scenario selection
    parser.add_argument(
        "--scenarios",
        nargs="+",
        default=["all"],
        help="Scenarios to use. Use 'all' for all available scenarios, or specify scenario IDs (default: all)"
    )
    
    # Configuration options
    parser.add_argument(
        "--conversations-per-scenario",
        type=int,
        default=20,
        help="Number of conversations per scenario per model (default: 20)"
    )
    
    parser.add_argument(
        "--concurrent-conversations",
        type=int,
        default=5,
        help="Maximum concurrent conversations (default: 5)"
    )
    
    parser.add_argument(
        "--concurrent-models",
        type=int,
        default=2,
        help="Maximum concurrent models (default: 2)"
    )
    
    parser.add_argument(
        "--timeout-minutes",
        type=int,
        default=10,
        help="Conversation timeout in minutes (default: 10)"
    )
    
    # Output options
    parser.add_argument(
        "--output",
        type=str,
        default="./data/conversation_results",
        help="Output directory for results (default: ./data/conversation_results)"
    )
    
    parser.add_argument(
        "--no-individual-logs",
        action="store_true",
        help="Disable saving individual conversation logs"
    )
    
    parser.add_argument(
        "--no-aggregate-reports",
        action="store_true", 
        help="Disable saving aggregate reports"
    )
    
    # Feature toggles
    parser.add_argument(
        "--disable-safety-monitoring",
        action="store_true",
        help="Disable safety monitoring"
    )
    
    parser.add_argument(
        "--disable-conversation-branching",
        action="store_true",
        help="Disable conversation branching"
    )
    
    parser.add_argument(
        "--disable-metrics-collection",
        action="store_true",
        help="Disable metrics collection"
    )
    
    parser.add_argument(
        "--disable-error-recovery",
        action="store_true",
        help="Disable error recovery"
    )
    
    # Logging options
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug logging"
    )
    
    parser.add_argument(
        "--log-file",
        type=str,
        help="Log file path (default: logs to console)"
    )
    
    # Dry run option
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without actually running conversations"
    )
    
    return parser


def create_models(model_names: List[str]) -> List:
    """Create model instances based on names."""
    
    models = []
    
    for model_name in model_names:
        try:
            if model_name == "openai":
                models.append(OpenAIClient())
                print(f"✓ Initialized OpenAI client")
                
            elif model_name == "deepseek":
                models.append(DeepSeekClient())
                print(f"✓ Initialized DeepSeek client")
                
            else:
                print(f"⚠ Unknown model: {model_name}")
                
        except Exception as e:
            print(f"✗ Failed to initialize {model_name}: {e}")
            continue
    
    return models


def load_scenarios(scenario_names: List[str]) -> List:
    """Load scenarios based on names."""
    
    scenario_loader = ScenarioLoader()
    
    if "all" in scenario_names:
        scenarios = scenario_loader.load_all_scenarios()
        print(f"✓ Loaded all {len(scenarios)} available scenarios")
    else:
        scenarios = []
        for scenario_name in scenario_names:
            try:
                scenario = scenario_loader.load_scenario(scenario_name)
                scenarios.append(scenario)
                print(f"✓ Loaded scenario: {scenario_name}")
            except Exception as e:
                print(f"✗ Failed to load scenario {scenario_name}: {e}")
    
    return scenarios


def setup_batch_config(args) -> BatchConfig:
    """Create batch configuration from arguments."""
    
    return BatchConfig(
        conversations_per_scenario_per_model=args.conversations_per_scenario,
        max_concurrent_conversations=args.concurrent_conversations,
        max_concurrent_models=args.concurrent_models,
        conversation_timeout_minutes=args.timeout_minutes,
        output_directory=args.output,
        save_individual_conversations=not args.no_individual_logs,
        save_aggregate_reports=not args.no_aggregate_reports,
        enable_safety_monitoring=not args.disable_safety_monitoring,
        enable_conversation_branching=not args.disable_conversation_branching,
        enable_metrics_collection=not args.disable_metrics_collection,
        enable_error_recovery=not args.disable_error_recovery
    )


def print_dry_run_summary(models: List, scenarios: List, config: BatchConfig):
    """Print what would be done in dry run mode."""
    
    total_conversations = len(models) * len(scenarios) * config.conversations_per_scenario_per_model
    
    print("\n" + "="*60)
    print("DRY RUN SUMMARY")
    print("="*60)
    print(f"Models to evaluate: {len(models)}")
    for model in models:
        print(f"  - {model.model_name}")
    
    print(f"\nScenarios to use: {len(scenarios)}")
    for scenario in scenarios:
        print(f"  - {scenario.scenario_id}: {scenario.title}")
    
    print(f"\nConfiguration:")
    print(f"  - Conversations per scenario per model: {config.conversations_per_scenario_per_model}")
    print(f"  - Total conversations planned: {total_conversations}")
    print(f"  - Max concurrent conversations: {config.max_concurrent_conversations}")
    print(f"  - Max concurrent models: {config.max_concurrent_models}")
    print(f"  - Conversation timeout: {config.conversation_timeout_minutes} minutes")
    print(f"  - Output directory: {config.output_directory}")
    
    print(f"\nFeatures enabled:")
    print(f"  - Safety monitoring: {config.enable_safety_monitoring}")
    print(f"  - Conversation branching: {config.enable_conversation_branching}")
    print(f"  - Metrics collection: {config.enable_metrics_collection}")
    print(f"  - Error recovery: {config.enable_error_recovery}")
    
    print("\n" + "="*60)
    print("This was a dry run. No conversations were generated.")
    print("Remove --dry-run to actually run the evaluation.")
    print("="*60)


async def run_evaluation(models: List, scenarios: List, config: BatchConfig):
    """Run the actual evaluation."""
    
    total_conversations = len(models) * len(scenarios) * config.conversations_per_scenario_per_model
    
    print("\n" + "="*60)
    print("STARTING CONVERSATION GENERATION")
    print("="*60)
    print(f"Total conversations to generate: {total_conversations}")
    print(f"Models: {', '.join(model.model_name for model in models)}")
    print(f"Scenarios: {len(scenarios)}")
    print(f"Output directory: {config.output_directory}")
    print("="*60)
    
    try:
        # Run batch evaluation
        results = await run_batch_evaluation(models, config, scenarios)
        
        # Print summary
        print("\n" + "="*60)
        print("BATCH COMPLETION SUMMARY")
        print("="*60)
        
        performance = results["performance_metrics"]
        print(f"Conversations completed: {performance['total_conversations_completed']}")
        print(f"Conversations failed: {performance['total_conversations_failed']}")
        print(f"Overall success rate: {performance['overall_success_rate']:.1f}%")
        print(f"Total duration: {results['batch_summary']['total_duration_hours']:.2f} hours")
        print(f"Rate: {performance['conversations_per_hour']:.1f} conversations/hour")
        
        # Safety summary
        safety = results["safety_analysis"]
        print(f"\nSafety Analysis:")
        print(f"  - Total safety flags: {safety['total_safety_flags']}")
        print(f"  - Crisis interventions: {safety['crisis_interventions']}")
        print(f"  - Safety flags per conversation: {safety['safety_flags_per_conversation']:.2f}")
        
        # Model comparison
        print(f"\nModel Performance:")
        for model_name, model_data in results["model_comparison"].items():
            print(f"  - {model_name}:")
            print(f"    Success rate: {model_data['success_rate']:.1f}%")
            print(f"    Conversations: {model_data['conversations_completed']}")
            print(f"    Avg response time: {model_data['avg_response_time_ms']:.0f}ms")
            print(f"    Crisis rate: {model_data['crisis_intervention_rate']:.1f}%")
        
        print(f"\nResults saved to: {config.output_directory}")
        print("="*60)
        
        return results
        
    except Exception as e:
        print(f"\n✗ Batch evaluation failed: {e}")
        raise


async def main():
    """Main entry point."""
    
    parser = setup_argument_parser()
    args = parser.parse_args()
    
    # Setup logging
    log_level = logging.DEBUG if args.debug else (logging.INFO if args.verbose else logging.WARNING)
    setup_logging(
        log_level=log_level,
        log_file=args.log_file,
        include_timestamp=True
    )
    
    logger = logging.getLogger(__name__)
    logger.info("Starting Mental Health LLM Evaluation System")
    
    try:
        # Create models
        print("Initializing models...")
        models = create_models(args.models)
        if not models:
            print("✗ No valid models could be initialized")
            return 1
        
        # Load scenarios
        print("Loading scenarios...")
        scenarios = load_scenarios(args.scenarios)
        if not scenarios:
            print("✗ No valid scenarios could be loaded")
            return 1
        
        # Setup configuration
        config = setup_batch_config(args)
        
        # Dry run or actual run
        if args.dry_run:
            print_dry_run_summary(models, scenarios, config)
            return 0
        else:
            results = await run_evaluation(models, scenarios, config)
            return 0
            
    except KeyboardInterrupt:
        print("\n\nEvaluation interrupted by user")
        return 1
    except Exception as e:
        print(f"\n✗ Fatal error: {e}")
        logger.exception("Fatal error during execution")
        return 1


if __name__ == "__main__":
    """Entry point when run as script."""
    
    # Check Python version
    if sys.version_info < (3, 8):
        print("Error: Python 3.8 or higher is required")
        sys.exit(1)
    
    # Run main function
    exit_code = asyncio.run(main())
    sys.exit(exit_code)