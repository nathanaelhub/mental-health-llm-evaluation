#!/usr/bin/env python3
"""
Master Pipeline Script

Orchestrates the complete Mental Health LLM Evaluation pipeline from setup to final report.
Runs all scripts in sequence with proper error handling and checkpoint management.

Usage:
    python scripts/run_pipeline.py --config config/experiment.yaml
    python scripts/run_pipeline.py --experiment exp_20240101_12345678 --resume
    python scripts/run_pipeline.py --dry-run --config config/experiment.yaml
"""

import argparse
import json
import logging
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from utils.logging_config import setup_logging, get_logger


class PipelineOrchestrator:
    """Orchestrates the complete evaluation pipeline."""
    
    def __init__(self, config_path: Optional[str] = None, experiment_id: Optional[str] = None, dry_run: bool = False):
        self.config_path = config_path
        self.experiment_id = experiment_id
        self.dry_run = dry_run
        self.logger = get_logger(__name__)
        
        # Pipeline state
        self.pipeline_steps = [
            ("setup_experiment", "scripts/setup_experiment.py"),
            ("run_conversations", "scripts/run_conversations.py"),
            ("evaluate_conversations", "scripts/evaluate_conversations.py"),
            ("analyze_results", "scripts/analyze_results.py"),
            ("generate_report", "scripts/generate_report.py")
        ]
        
        self.completed_steps = []
        self.failed_steps = []
        self.pipeline_start_time = None
        self.step_results = {}
        
    def load_pipeline_checkpoint(self) -> Optional[Dict]:
        """Load pipeline checkpoint if exists."""
        if not self.experiment_id:
            return None
            
        checkpoint_path = PROJECT_ROOT / "experiments" / self.experiment_id / "checkpoints" / "pipeline_checkpoint.json"
        
        if checkpoint_path.exists():
            try:
                with open(checkpoint_path, 'r') as f:
                    checkpoint_data = json.load(f)
                
                self.completed_steps = checkpoint_data.get("completed_steps", [])
                self.step_results = checkpoint_data.get("step_results", {})
                
                self.logger.info(f"Loaded pipeline checkpoint: {len(self.completed_steps)} steps completed")
                return checkpoint_data
                
            except Exception as e:
                self.logger.warning(f"Failed to load pipeline checkpoint: {e}")
        
        return None
    
    def save_pipeline_checkpoint(self, current_step: str, step_result: Dict) -> None:
        """Save pipeline checkpoint."""
        if not self.experiment_id:
            return
            
        checkpoint_data = {
            "experiment_id": self.experiment_id,
            "timestamp": datetime.now().isoformat(),
            "pipeline_start_time": self.pipeline_start_time,
            "completed_steps": self.completed_steps,
            "failed_steps": self.failed_steps,
            "current_step": current_step,
            "step_results": self.step_results
        }
        
        checkpoint_path = PROJECT_ROOT / "experiments" / self.experiment_id / "checkpoints" / "pipeline_checkpoint.json"
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        
        if not self.dry_run:
            with open(checkpoint_path, 'w') as f:
                json.dump(checkpoint_data, f, indent=2)
            
            self.logger.debug(f"Saved pipeline checkpoint: {checkpoint_path}")
    
    def run_script(self, step_name: str, script_path: str, args: List[str]) -> Tuple[bool, Dict]:
        """Run a single pipeline script."""
        try:
            self.logger.info(f"Running {step_name}...")
            
            # Build command
            script_full_path = PROJECT_ROOT / script_path
            cmd = [sys.executable, str(script_full_path)] + args
            
            if self.dry_run:
                self.logger.info(f"DRY RUN: Would execute: {' '.join(cmd)}")
                return True, {"dry_run": True, "command": ' '.join(cmd)}
            
            # Execute script
            start_time = time.time()
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                cwd=PROJECT_ROOT
            )
            elapsed_time = time.time() - start_time
            
            # Process result
            success = result.returncode == 0
            
            step_result = {
                "step_name": step_name,
                "success": success,
                "return_code": result.returncode,
                "elapsed_time": elapsed_time,
                "command": ' '.join(cmd),
                "stdout": result.stdout,
                "stderr": result.stderr
            }
            
            if success:
                self.logger.info(f"✅ {step_name} completed successfully ({elapsed_time:.1f}s)")
                self.completed_steps.append(step_name)
            else:
                self.logger.error(f"❌ {step_name} failed (return code: {result.returncode})")
                self.logger.error(f"STDERR: {result.stderr}")
                self.failed_steps.append(step_name)
            
            self.step_results[step_name] = step_result
            return success, step_result
            
        except Exception as e:
            self.logger.error(f"Exception running {step_name}: {str(e)}")
            step_result = {
                "step_name": step_name,
                "success": False,
                "error": str(e),
                "elapsed_time": 0
            }
            self.step_results[step_name] = step_result
            self.failed_steps.append(step_name)
            return False, step_result
    
    def run_setup_experiment(self) -> Tuple[bool, Dict]:
        """Run experiment setup step."""
        args = []
        
        if self.config_path:
            args.extend(["--config", self.config_path])
        
        if self.dry_run:
            args.append("--dry-run")
        
        success, result = self.run_script("setup_experiment", "scripts/setup_experiment.py", args)
        
        # Extract experiment ID from output if setup was successful
        if success and not self.dry_run and not self.experiment_id:
            try:
                # Parse experiment ID from stdout
                stdout_lines = result.get("stdout", "").split("\n")
                for line in stdout_lines:
                    if "Experiment ID:" in line:
                        self.experiment_id = line.split(":")[-1].strip()
                        self.logger.info(f"Detected experiment ID: {self.experiment_id}")
                        break
            except Exception as e:
                self.logger.warning(f"Failed to extract experiment ID: {e}")
        
        return success, result
    
    def run_conversations(self) -> Tuple[bool, Dict]:
        """Run conversation generation step."""
        if not self.experiment_id:
            return False, {"error": "No experiment ID available"}
        
        args = ["--experiment", self.experiment_id]
        
        if self.dry_run:
            args.append("--dry-run")
        
        return self.run_script("run_conversations", "scripts/run_conversations.py", args)
    
    def run_evaluation(self) -> Tuple[bool, Dict]:
        """Run conversation evaluation step."""
        if not self.experiment_id:
            return False, {"error": "No experiment ID available"}
        
        args = ["--experiment", self.experiment_id]
        
        if self.dry_run:
            args.append("--dry-run")
        
        return self.run_script("evaluate_conversations", "scripts/evaluate_conversations.py", args)
    
    def run_analysis(self) -> Tuple[bool, Dict]:
        """Run results analysis step."""
        if not self.experiment_id:
            return False, {"error": "No experiment ID available"}
        
        args = ["--experiment", self.experiment_id]
        
        if self.dry_run:
            args.append("--dry-run")
        
        return self.run_script("analyze_results", "scripts/analyze_results.py", args)
    
    def run_report_generation(self) -> Tuple[bool, Dict]:
        """Run report generation step."""
        if not self.experiment_id:
            return False, {"error": "No experiment ID available"}
        
        args = ["--experiment", self.experiment_id]
        
        if self.dry_run:
            args.append("--dry-run")
        
        return self.run_script("generate_report", "scripts/generate_report.py", args)
    
    def run_pipeline(self, resume: bool = False) -> Tuple[bool, Dict]:
        """Run the complete pipeline."""
        try:
            self.pipeline_start_time = datetime.now().isoformat()
            
            # Load checkpoint if resuming
            if resume and self.experiment_id:
                checkpoint_data = self.load_pipeline_checkpoint()
                if checkpoint_data:
                    self.logger.info(f"Resuming pipeline from checkpoint")
            
            # Define pipeline steps with their execution functions
            pipeline_steps = [
                ("setup_experiment", self.run_setup_experiment),
                ("run_conversations", self.run_conversations),
                ("evaluate_conversations", self.run_evaluation),
                ("analyze_results", self.run_analysis),
                ("generate_report", self.run_report_generation)
            ]
            
            self.logger.info(f"Starting pipeline with {len(pipeline_steps)} steps")
            
            # Execute pipeline steps
            for step_name, step_function in pipeline_steps:
                # Skip if already completed (resume mode)
                if step_name in self.completed_steps:
                    self.logger.info(f"⏭️  Skipping {step_name} (already completed)")
                    continue
                
                # Execute step
                success, step_result = step_function()
                
                # Save checkpoint after each step
                self.save_pipeline_checkpoint(step_name, step_result)
                
                # Handle failure
                if not success:
                    self.logger.error(f"Pipeline failed at step: {step_name}")
                    if "error" in step_result:
                        self.logger.error(f"Error: {step_result['error']}")
                    
                    return False, {
                        "failed_step": step_name,
                        "completed_steps": self.completed_steps,
                        "failed_steps": self.failed_steps,
                        "step_results": self.step_results,
                        "error": step_result.get("error", "Step failed")
                    }
            
            # Pipeline completed successfully
            total_time = sum(result.get("elapsed_time", 0) for result in self.step_results.values())
            
            self.logger.info("🎉 Pipeline completed successfully!")
            self.logger.info(f"Total execution time: {total_time:.1f} seconds")
            
            return True, {
                "experiment_id": self.experiment_id,
                "completed_steps": self.completed_steps,
                "step_results": self.step_results,
                "total_execution_time": total_time,
                "pipeline_start_time": self.pipeline_start_time
            }
            
        except Exception as e:
            self.logger.error(f"Pipeline execution failed: {str(e)}")
            return False, {"error": str(e)}
    
    def get_pipeline_status(self) -> Dict:
        """Get current pipeline status."""
        if not self.experiment_id:
            return {"status": "no_experiment"}
        
        checkpoint_path = PROJECT_ROOT / "experiments" / self.experiment_id / "checkpoints" / "pipeline_checkpoint.json"
        
        if not checkpoint_path.exists():
            return {"status": "not_started"}
        
        try:
            with open(checkpoint_path, 'r') as f:
                checkpoint_data = json.load(f)
            
            total_steps = len(self.pipeline_steps)
            completed_steps = len(checkpoint_data.get("completed_steps", []))
            failed_steps = len(checkpoint_data.get("failed_steps", []))
            
            if failed_steps > 0:
                status = "failed"
            elif completed_steps == total_steps:
                status = "completed"
            else:
                status = "in_progress"
            
            return {
                "status": status,
                "experiment_id": self.experiment_id,
                "total_steps": total_steps,
                "completed_steps": completed_steps,
                "failed_steps": failed_steps,
                "progress_percentage": (completed_steps / total_steps) * 100,
                "last_updated": checkpoint_data.get("timestamp"),
                "step_results": checkpoint_data.get("step_results", {})
            }
            
        except Exception as e:
            return {"status": "error", "error": str(e)}


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Run complete Mental Health LLM Evaluation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline with new experiment
  python scripts/run_pipeline.py --config config/experiment.yaml
  
  # Resume existing experiment
  python scripts/run_pipeline.py --experiment exp_20240101_12345678 --resume
  
  # Check pipeline status
  python scripts/run_pipeline.py --experiment exp_20240101_12345678 --status
  
  # Test run without execution
  python scripts/run_pipeline.py --dry-run --config config/experiment.yaml
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        help="Experiment ID to resume or check status"
    )
    
    parser.add_argument(
        "--resume", "-r",
        action="store_true",
        help="Resume pipeline from checkpoint"
    )
    
    parser.add_argument(
        "--status", "-s",
        action="store_true",
        help="Check pipeline status and exit"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without executing pipeline steps"
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
    
    try:
        # Initialize orchestrator
        orchestrator = PipelineOrchestrator(
            config_path=args.config,
            experiment_id=args.experiment,
            dry_run=args.dry_run
        )
        
        # Handle status check
        if args.status:
            if not args.experiment:
                print("❌ --experiment required for status check")
                return 1
            
            status = orchestrator.get_pipeline_status()
            
            print(f"\n{'='*60}")
            print(f"Pipeline Status: {args.experiment}")
            print(f"{'='*60}")
            print(f"Status: {status['status'].upper()}")
            
            if status["status"] not in ["no_experiment", "not_started", "error"]:
                print(f"Progress: {status['completed_steps']}/{status['total_steps']} steps ({status['progress_percentage']:.1f}%)")
                print(f"Last Updated: {status.get('last_updated', 'Unknown')}")
                
                if status.get("step_results"):
                    print(f"\nStep Results:")
                    for step_name, result in status["step_results"].items():
                        status_emoji = "✅" if result.get("success", False) else "❌"
                        elapsed = result.get("elapsed_time", 0)
                        print(f"  {status_emoji} {step_name}: {elapsed:.1f}s")
            
            if status["status"] == "error":
                print(f"Error: {status.get('error', 'Unknown error')}")
            
            return 0
        
        # Validate arguments
        if not args.config and not args.experiment:
            print("❌ Either --config or --experiment is required")
            return 1
        
        if args.resume and not args.experiment:
            print("❌ --experiment required for resume mode")
            return 1
        
        # Display pipeline info
        print(f"\n{'='*60}")
        print(f"Mental Health LLM Evaluation - Complete Pipeline")
        print(f"{'='*60}")
        
        if args.experiment:
            print(f"Experiment ID: {args.experiment}")
        
        if args.config:
            print(f"Configuration: {args.config}")
        
        if args.dry_run:
            print("🧪 DRY RUN MODE - No actual execution")
        
        if args.resume:
            print("🔄 RESUME MODE - Continue from checkpoint")
        
        print(f"Pipeline Steps:")
        for i, (step_name, _) in enumerate(orchestrator.pipeline_steps, 1):
            print(f"  {i}. {step_name}")
        
        print()
        
        # Run pipeline
        success, result = orchestrator.run_pipeline(resume=args.resume)
        
        if success:
            print(f"\n🎉 Pipeline completed successfully!")
            if not args.dry_run:
                print(f"📊 Results:")
                print(f"  - Experiment ID: {result.get('experiment_id')}")
                print(f"  - Completed steps: {len(result.get('completed_steps', []))}")
                print(f"  - Total time: {result.get('total_execution_time', 0):.1f} seconds")
                
                if result.get('experiment_id'):
                    exp_dir = PROJECT_ROOT / "experiments" / result['experiment_id']
                    print(f"  - Results directory: {exp_dir}")
                    
                    # Check for report files
                    reports_dir = exp_dir / "reports"
                    if reports_dir.exists():
                        report_files = list(reports_dir.glob("*"))
                        if report_files:
                            print(f"  - Generated reports: {len(report_files)}")
                            for report_file in report_files:
                                print(f"    • {report_file.name}")
            
            return 0
        else:
            print(f"\n❌ Pipeline failed!")
            if "failed_step" in result:
                print(f"Failed at step: {result['failed_step']}")
            if "error" in result:
                print(f"Error: {result['error']}")
            
            print(f"\nCompleted steps: {', '.join(result.get('completed_steps', []))}")
            
            if result.get('failed_step') and args.experiment:
                print(f"\nTo resume from checkpoint:")
                print(f"python scripts/run_pipeline.py --experiment {args.experiment} --resume")
            
            return 1
    
    except KeyboardInterrupt:
        logger.info("Pipeline interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())