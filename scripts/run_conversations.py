#!/usr/bin/env python3
"""
Run Conversations Script

Generates conversations across all models and scenarios with real-time metrics,
error handling, and progress reporting.

Usage:
    python scripts/run_conversations.py --experiment exp_20240101_12345678
    python scripts/run_conversations.py --experiment exp_20240101_12345678 --models openai
    python scripts/run_conversations.py --resume exp_20240101_12345678
    python scripts/run_conversations.py --dry-run --experiment exp_20240101_12345678
"""

import argparse
import asyncio
import json
import logging
import signal
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from conversation.conversation_manager import ConversationManager
from conversation.batch_processor import BatchProcessor, BatchConfig
from models.openai_client import OpenAIClient
from models.deepseek_client import DeepSeekClient
from scenarios.scenario_loader import ScenarioLoader
from utils.logging_config import setup_logging, get_logger


class ConversationRunner:
    """Manages conversation generation across models and scenarios."""
    
    def __init__(self, experiment_id: str, dry_run: bool = False):
        self.experiment_id = experiment_id
        self.dry_run = dry_run
        self.logger = get_logger(__name__)
        
        # Initialize state
        self.experiment_dir = None
        self.manifest = None
        self.models = {}
        self.scenarios = []
        self.conversation_manager = None
        self.batch_processor = None
        
        # Progress tracking
        self.total_conversations = 0
        self.completed_conversations = 0
        self.failed_conversations = 0
        self.start_time = None
        self.interrupted = False
        
        # Setup signal handlers for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle interrupt signals gracefully."""
        self.logger.info("Received interrupt signal, shutting down gracefully...")
        self.interrupted = True
    
    def load_experiment(self) -> bool:
        """Load experiment configuration and setup."""
        try:
            # Find experiment directory
            experiments_dir = PROJECT_ROOT / "experiments"
            self.experiment_dir = experiments_dir / self.experiment_id
            
            if not self.experiment_dir.exists():
                # Try finding by partial ID
                matching_dirs = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and self.experiment_id in d.name]
                if len(matching_dirs) == 1:
                    self.experiment_dir = matching_dirs[0]
                    self.experiment_id = matching_dirs[0].name
                elif len(matching_dirs) > 1:
                    self.logger.error(f"Multiple experiments match '{self.experiment_id}':")
                    for d in matching_dirs:
                        self.logger.error(f"  - {d.name}")
                    return False
                else:
                    self.logger.error(f"Experiment not found: {self.experiment_id}")
                    return False
            
            # Load manifest
            manifest_path = self.experiment_dir / "experiment_manifest.json"
            if not manifest_path.exists():
                self.logger.error(f"Experiment manifest not found: {manifest_path}")
                return False
            
            with open(manifest_path, 'r') as f:
                self.manifest = json.load(f)
            
            self.logger.info(f"Loaded experiment: {self.experiment_id}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment: {str(e)}")
            return False
    
    def initialize_models(self, model_filter: Optional[List[str]] = None) -> bool:
        """Initialize model clients based on configuration."""
        try:
            models_config = self.manifest["configuration"]["models"]
            
            for model_name, model_config in models_config.items():
                if not model_config.get("enabled", False):
                    continue
                
                if model_filter and model_name not in model_filter:
                    continue
                
                self.logger.info(f"Initializing {model_name} model...")
                
                if not self.dry_run:
                    if model_name == "openai":
                        client = OpenAIClient(
                            model=model_config.get("model", "gpt-4"),
                            temperature=model_config.get("temperature", 0.7),
                            max_tokens=model_config.get("max_tokens", 2048)
                        )
                    elif model_name == "deepseek":
                        client = DeepSeekClient(
                            model_path=model_config.get("model_path"),
                            device=model_config.get("device", "auto"),
                            precision=model_config.get("precision", "fp16")
                        )
                    else:
                        self.logger.warning(f"Unknown model type: {model_name}")
                        continue
                    
                    self.models[model_name] = client
                else:
                    self.logger.info(f"DRY RUN: Would initialize {model_name}")
                    self.models[model_name] = f"MockClient_{model_name}"
            
            self.logger.info(f"Initialized {len(self.models)} models: {list(self.models.keys())}")
            return len(self.models) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to initialize models: {str(e)}")
            return False
    
    def load_scenarios(self) -> bool:
        """Load conversation scenarios."""
        try:
            scenarios_config = self.manifest["configuration"]["scenarios"]
            scenarios_dir = scenarios_config.get("directory", "data/scenarios")
            
            if not self.dry_run:
                loader = ScenarioLoader(scenarios_directory=scenarios_dir)
                
                # Load scenarios from manifest
                scenario_ids = [s["scenario_id"] for s in self.manifest["scenarios"]]
                scenarios = []
                
                for scenario_id in scenario_ids:
                    try:
                        scenario = loader.load_scenario(scenario_id)
                        scenarios.append(scenario)
                    except Exception as e:
                        self.logger.warning(f"Failed to load scenario {scenario_id}: {e}")
                
                self.scenarios = scenarios
            else:
                # Use manifest data for dry run
                self.scenarios = self.manifest["scenarios"]
            
            self.logger.info(f"Loaded {len(self.scenarios)} scenarios")
            return len(self.scenarios) > 0
            
        except Exception as e:
            self.logger.error(f"Failed to load scenarios: {str(e)}")
            return False
    
    def calculate_total_conversations(self) -> int:
        """Calculate total number of conversations to generate."""
        eval_config = self.manifest["configuration"]["evaluation"]
        conversations_per_scenario = eval_config.get("conversations_per_scenario", 10)
        
        total = len(self.models) * len(self.scenarios) * conversations_per_scenario
        self.total_conversations = total
        return total
    
    def setup_conversation_manager(self) -> bool:
        """Setup conversation manager and batch processor."""
        try:
            eval_config = self.manifest["configuration"]["evaluation"]
            
            # Conversation manager config
            conv_config = {
                "output_directory": str(self.experiment_dir / "conversations"),
                "enable_safety_monitoring": eval_config.get("enable_safety_monitoring", True),
                "enable_metrics_collection": eval_config.get("enable_metrics_collection", True),
                "max_conversation_turns": eval_config.get("max_conversation_turns", 20),
                "timeout_seconds": eval_config.get("timeout_seconds", 300)
            }
            
            if not self.dry_run:
                self.conversation_manager = ConversationManager(conv_config)
            
            # Batch processor config
            batch_config = BatchConfig(
                conversations_per_scenario_per_model=eval_config.get("conversations_per_scenario", 10),
                max_concurrent_conversations=eval_config.get("max_concurrent", 3),
                output_directory=str(self.experiment_dir / "conversations"),
                enable_safety_monitoring=eval_config.get("enable_safety_monitoring", True),
                enable_metrics_collection=eval_config.get("enable_metrics_collection", True),
                timeout_seconds=eval_config.get("timeout_seconds", 300)
            )
            
            if not self.dry_run:
                self.batch_processor = BatchProcessor(batch_config)
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to setup conversation manager: {str(e)}")
            return False
    
    def save_checkpoint(self, conversation_results: List[Dict]) -> None:
        """Save progress checkpoint."""
        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "completed_conversations": self.completed_conversations,
            "failed_conversations": self.failed_conversations,
            "total_conversations": self.total_conversations,
            "conversation_results": conversation_results
        }
        
        checkpoint_path = self.experiment_dir / "checkpoints" / f"conversations_checkpoint.json"
        checkpoint_path.parent.mkdir(exist_ok=True)
        
        if not self.dry_run:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.debug(f"Saved checkpoint: {checkpoint_path}")
    
    def load_checkpoint(self) -> Optional[List[Dict]]:
        """Load previous checkpoint if exists."""
        checkpoint_path = self.experiment_dir / "checkpoints" / f"conversations_checkpoint.json"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.completed_conversations = checkpoint_data.get("completed_conversations", 0)
                self.failed_conversations = checkpoint_data.get("failed_conversations", 0)
                
                self.logger.info(f"Loaded checkpoint: {self.completed_conversations} completed, "
                               f"{self.failed_conversations} failed")
                
                return checkpoint_data.get("conversation_results", [])
                
            except Exception as e:
                self.logger.warning(f"Failed to load checkpoint: {e}")
        
        return None
    
    def estimate_completion_time(self, elapsed_time: float) -> str:
        """Estimate completion time based on current progress."""
        if self.completed_conversations == 0:
            return "Calculating..."
        
        avg_time_per_conversation = elapsed_time / self.completed_conversations
        remaining_conversations = self.total_conversations - self.completed_conversations
        estimated_remaining = avg_time_per_conversation * remaining_conversations
        
        eta = datetime.now() + timedelta(seconds=estimated_remaining)
        return eta.strftime("%Y-%m-%d %H:%M:%S")
    
    async def run_conversations(self, resume: bool = False) -> Tuple[bool, Dict]:
        """Run conversation generation."""
        try:
            self.start_time = time.time()
            conversation_results = []
            
            # Load checkpoint if resuming
            if resume:
                checkpoint_results = self.load_checkpoint()
                if checkpoint_results:
                    conversation_results = checkpoint_results
                    self.logger.info(f"Resuming from checkpoint with {len(conversation_results)} existing results")
            
            # Calculate remaining work
            remaining_work = []
            completed_ids = {r.get("conversation_id") for r in conversation_results if r}
            
            eval_config = self.manifest["configuration"]["evaluation"]
            conversations_per_scenario = eval_config.get("conversations_per_scenario", 10)
            
            for model_name in self.models.keys():
                for scenario in self.scenarios:
                    scenario_id = scenario.scenario_id if hasattr(scenario, 'scenario_id') else scenario["scenario_id"]
                    
                    for i in range(conversations_per_scenario):
                        conv_id = f"{model_name}_{scenario_id}_{i:03d}"
                        if conv_id not in completed_ids:
                            remaining_work.append({
                                "conversation_id": conv_id,
                                "model_name": model_name,
                                "scenario": scenario
                            })
            
            self.logger.info(f"Total conversations: {self.total_conversations}")
            self.logger.info(f"Completed: {len(conversation_results)}")
            self.logger.info(f"Remaining: {len(remaining_work)}")
            
            if self.dry_run:
                self.logger.info("DRY RUN: Would generate conversations")
                return True, {
                    "total_planned": self.total_conversations,
                    "already_completed": len(conversation_results),
                    "remaining": len(remaining_work)
                }
            
            # Progress bar for remaining work
            progress_bar = tqdm(
                total=len(remaining_work),
                desc="Generating conversations",
                unit="conv",
                initial=0
            )
            
            # Process conversations
            for work_item in remaining_work:
                if self.interrupted:
                    self.logger.info("Stopping due to interrupt signal")
                    break
                
                try:
                    # Generate conversation
                    model_client = self.models[work_item["model_name"]]
                    scenario = work_item["scenario"]
                    conversation_id = work_item["conversation_id"]
                    
                    conversation = await self.conversation_manager.generate_conversation(
                        model_client=model_client,
                        scenario=scenario,
                        conversation_id=conversation_id,
                        enable_evaluation=True
                    )
                    
                    # Add metadata
                    conversation_result = {
                        "conversation_id": conversation_id,
                        "model_name": work_item["model_name"],
                        "scenario_id": scenario.scenario_id if hasattr(scenario, 'scenario_id') else scenario["scenario_id"],
                        "status": "completed",
                        "timestamp": datetime.now().isoformat(),
                        "conversation_data": conversation
                    }
                    
                    conversation_results.append(conversation_result)
                    self.completed_conversations += 1
                    
                    # Update progress
                    elapsed_time = time.time() - self.start_time
                    eta = self.estimate_completion_time(elapsed_time)
                    
                    progress_bar.set_postfix({
                        "Model": work_item["model_name"][:8],
                        "Scenario": scenario.scenario_id[:10] if hasattr(scenario, 'scenario_id') else scenario["scenario_id"][:10],
                        "ETA": eta.split()[-1] if eta != "Calculating..." else "Calc..."
                    })
                    progress_bar.update(1)
                    
                    # Save checkpoint every 10 conversations
                    if self.completed_conversations % 10 == 0:
                        self.save_checkpoint(conversation_results)
                    
                except Exception as e:
                    self.logger.error(f"Failed to generate conversation {conversation_id}: {str(e)}")
                    self.failed_conversations += 1
                    
                    # Add failed result
                    failed_result = {
                        "conversation_id": conversation_id,
                        "model_name": work_item["model_name"],
                        "scenario_id": scenario.scenario_id if hasattr(scenario, 'scenario_id') else scenario["scenario_id"],
                        "status": "failed",
                        "error": str(e),
                        "timestamp": datetime.now().isoformat()
                    }
                    conversation_results.append(failed_result)
                    
                    progress_bar.update(1)
            
            progress_bar.close()
            
            # Final checkpoint
            self.save_checkpoint(conversation_results)
            
            # Save final results
            results_path = self.experiment_dir / "conversations" / "all_conversations.json"
            with open(results_path, 'w') as f:
                json.dump(conversation_results, f, indent=2)
            
            elapsed_time = time.time() - self.start_time
            
            self.logger.info("Conversation generation completed!")
            self.logger.info(f"Total time: {elapsed_time:.1f} seconds")
            self.logger.info(f"Completed: {self.completed_conversations}")
            self.logger.info(f"Failed: {self.failed_conversations}")
            self.logger.info(f"Success rate: {self.completed_conversations / (self.completed_conversations + self.failed_conversations) * 100:.1f}%")
            
            return True, {
                "total_conversations": self.total_conversations,
                "completed": self.completed_conversations,
                "failed": self.failed_conversations,
                "elapsed_time": elapsed_time,
                "results_path": str(results_path),
                "success_rate": self.completed_conversations / (self.completed_conversations + self.failed_conversations) if (self.completed_conversations + self.failed_conversations) > 0 else 0
            }
            
        except Exception as e:
            self.logger.error(f"Conversation generation failed: {str(e)}")
            return False, {"error": str(e)}
    
    async def run_batch_conversations(self) -> Tuple[bool, Dict]:
        """Run conversations using batch processor (alternative approach)."""
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Would run batch conversation generation")
                return True, {"dry_run": True}
            
            self.logger.info("Starting batch conversation generation...")
            
            # Convert scenarios to proper objects if needed
            scenario_objects = []
            if self.scenarios and isinstance(self.scenarios[0], dict):
                # Load actual scenario objects
                loader = ScenarioLoader()
                for scenario_dict in self.scenarios:
                    scenario = loader.load_scenario(scenario_dict["scenario_id"])
                    scenario_objects.append(scenario)
            else:
                scenario_objects = self.scenarios
            
            # Progress callback
            def progress_callback(progress_data):
                percent = progress_data.get("progress_percentage", 0)
                completed = progress_data.get("conversations_completed", 0)
                total = progress_data.get("conversations_total", self.total_conversations)
                
                self.logger.info(f"Progress: {completed}/{total} ({percent:.1f}%)")
            
            # Run batch processing
            batch_results = await self.batch_processor.process_batch(
                models=list(self.models.values()),
                scenarios=scenario_objects,
                progress_callback=progress_callback
            )
            
            self.logger.info("Batch conversation generation completed!")
            return True, batch_results
            
        except Exception as e:
            self.logger.error(f"Batch conversation generation failed: {str(e)}")
            return False, {"error": str(e)}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Generate conversations for Mental Health LLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run conversations for experiment
  python scripts/run_conversations.py --experiment exp_20240101_12345678
  
  # Run with specific models only
  python scripts/run_conversations.py --experiment exp_20240101_12345678 --models openai
  
  # Resume interrupted experiment
  python scripts/run_conversations.py --resume exp_20240101_12345678
  
  # Test run without generating conversations
  python scripts/run_conversations.py --dry-run --experiment exp_20240101_12345678
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment ID to run conversations for"
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        help="Comma-separated list of models to use (openai,deepseek)"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume from checkpoint if available"
    )
    
    parser.add_argument(
        "--batch-mode",
        action="store_true",
        help="Use batch processor instead of individual conversation generation"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without generating actual conversations"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    async def run_async():
        try:
            # Initialize runner
            runner = ConversationRunner(
                experiment_id=args.experiment,
                dry_run=args.dry_run
            )
            
            # Load experiment
            if not runner.load_experiment():
                return 1
            
            # Parse model filter
            model_filter = None
            if args.models:
                model_filter = [m.strip() for m in args.models.split(",")]
            
            # Initialize models
            if not runner.initialize_models(model_filter):
                return 1
            
            # Load scenarios
            if not runner.load_scenarios():
                return 1
            
            # Calculate total work
            total_conversations = runner.calculate_total_conversations()
            
            # Setup conversation manager
            if not runner.setup_conversation_manager():
                return 1
            
            # Display experiment info
            print(f"\n{'='*60}")
            print(f"Mental Health LLM Evaluation - Conversation Generation")
            print(f"{'='*60}")
            print(f"Experiment ID: {runner.experiment_id}")
            print(f"Models: {', '.join(runner.models.keys())}")
            print(f"Scenarios: {len(runner.scenarios)}")
            print(f"Total conversations: {total_conversations}")
            
            if args.dry_run:
                print("🧪 DRY RUN MODE - No conversations will be generated")
            
            if args.resume:
                print("🔄 RESUME MODE - Will continue from checkpoint")
            
            print()
            
            # Run conversations
            if args.batch_mode:
                success, result = await runner.run_batch_conversations()
            else:
                success, result = await runner.run_conversations(resume=args.resume)
            
            if success:
                print(f"\n✅ Conversation generation completed!")
                if not args.dry_run:
                    print(f"📊 Results:")
                    print(f"  - Completed: {result.get('completed', 0)}")
                    print(f"  - Failed: {result.get('failed', 0)}")
                    print(f"  - Success rate: {result.get('success_rate', 0)*100:.1f}%")
                    print(f"  - Total time: {result.get('elapsed_time', 0):.1f} seconds")
                    
                    if 'results_path' in result:
                        print(f"  - Results saved: {result['results_path']}")
                    
                    print(f"\nNext step:")
                    print(f"python scripts/evaluate_conversations.py --experiment {runner.experiment_id}")
                
                return 0
            else:
                print(f"\n❌ Conversation generation failed!")
                if "error" in result:
                    print(f"Error: {result['error']}")
                return 1
        
        except KeyboardInterrupt:
            logger.info("Conversation generation interrupted by user")
            return 130
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            if args.verbose:
                import traceback
                traceback.print_exc()
            return 1
    
    # Run async main
    return asyncio.run(run_async())


if __name__ == "__main__":
    exit(main())