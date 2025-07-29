#!/usr/bin/env python3
"""
Reliable Research Runner
========================

A wrapper around run_research.py that handles intermittent failures gracefully.

Features:
- Automatic retry on NoneType and other errors (up to 3 attempts)
- 30-second delays between retries
- Saves partial results between attempts
- Always uses demo mode for clean output
- Validates results before finishing
- Provides clear status updates

Usage:
    python scripts/reliable_research.py --all-models
    python scripts/reliable_research.py --models openai,claude
    python scripts/reliable_research.py --quick
"""

import os
import sys
import time
import json
import subprocess
import argparse
from datetime import datetime
from pathlib import Path
import shutil

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))


class ReliableResearchRunner:
    def __init__(self, models=None, quick=False, scenarios=None):
        self.models = models
        self.quick = quick
        self.scenarios = scenarios
        self.max_retries = 3
        self.retry_delay = 30
        self.partial_results_dir = "results/development/partial"
        self.attempt_count = 0
        
    def ensure_directories(self):
        """Ensure required directories exist"""
        Path(self.partial_results_dir).mkdir(parents=True, exist_ok=True)
        
    def build_command(self):
        """Build the command to run run_research.py"""
        cmd = ["python", "scripts/run_research.py", "--demo"]
        
        if self.models:
            if self.models == ["all"]:
                cmd.append("--all-models")
            else:
                cmd.extend(["--models", ",".join(self.models)])
        else:
            cmd.append("--all-models")
            
        if self.quick:
            cmd.append("--quick")
            
        if self.scenarios:
            cmd.extend(["--scenarios", str(self.scenarios)])
            
        return cmd
        
    def save_partial_results(self, attempt_num):
        """Save partial results from current attempt"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        partial_dir = os.path.join(self.partial_results_dir, f"attempt_{attempt_num}_{timestamp}")
        
        # Create partial results directory
        Path(partial_dir).mkdir(parents=True, exist_ok=True)
        
        # Copy any results that exist
        results_files = [
            "results/detailed_results.json",
            "results/statistical_analysis.json",
            "results/research_report.txt"
        ]
        
        copied_files = []
        for file in results_files:
            if os.path.exists(file):
                dest = os.path.join(partial_dir, os.path.basename(file))
                shutil.copy2(file, dest)
                copied_files.append(os.path.basename(file))
                
        # Copy visualizations if they exist
        viz_dir = "results/visualizations"
        if os.path.exists(viz_dir):
            dest_viz = os.path.join(partial_dir, "visualizations")
            shutil.copytree(viz_dir, dest_viz, dirs_exist_ok=True)
            copied_files.append("visualizations/")
            
        if copied_files:
            print(f"💾 Saved partial results to {partial_dir}:")
            for file in copied_files:
                print(f"   - {file}")
                
        return partial_dir
        
    def run_evaluation(self, attempt_num):
        """Run the evaluation with error handling"""
        print(f"\n🚀 Attempt {attempt_num}/{self.max_retries}: Running evaluation...")
        
        cmd = self.build_command()
        print(f"📋 Command: {' '.join(cmd)}")
        
        try:
            # Run the command with real-time output
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                bufsize=1,  # Line buffered
                universal_newlines=True
            )
            
            # Print output in real-time
            stdout_lines = []
            stderr_lines = []
            
            while True:
                # Check if process is still running
                poll = process.poll()
                
                # Read available output
                line = process.stdout.readline()
                if line:
                    print(line.rstrip())
                    stdout_lines.append(line)
                    
                # Check if process has finished
                if poll is not None:
                    # Read any remaining output
                    remaining_stdout, remaining_stderr = process.communicate()
                    if remaining_stdout:
                        print(remaining_stdout)
                        stdout_lines.append(remaining_stdout)
                    if remaining_stderr:
                        stderr_lines.append(remaining_stderr)
                    break
                    
            # Get the return code
            result_returncode = process.returncode
            result_stdout = ''.join(stdout_lines)
            result_stderr = ''.join(stderr_lines)
            
            # Check for errors in output
            if result_returncode != 0:
                error_msg = result_stderr or "Unknown error"
                if "NoneType" in error_msg or "TypeError" in error_msg:
                    raise Exception(f"NoneType error detected: {error_msg}")
                else:
                    raise Exception(f"Command failed: {error_msg}")
                    
            # Success!
            return True
            
        except Exception as e:
            print(f"❌ Error during evaluation: {e}")
            raise
            
    def validate_results(self):
        """Validate that results are complete"""
        print("\n🔍 Validating results...")
        
        validations = {
            "detailed_results.json": self.validate_detailed_results,
            "statistical_analysis.json": self.validate_statistical_analysis,
            "visualizations": self.validate_visualizations
        }
        
        all_valid = True
        for name, validator in validations.items():
            try:
                if validator():
                    print(f"   ✅ {name}: Valid")
                else:
                    print(f"   ❌ {name}: Invalid or missing")
                    all_valid = False
            except Exception as e:
                print(f"   ❌ {name}: Error - {e}")
                all_valid = False
                
        return all_valid
        
    def validate_detailed_results(self):
        """Validate detailed results file"""
        file_path = "results/evaluations/detailed_results.json"
        if not os.path.exists(file_path):
            # Try alternate path
            file_path = "results/detailed_results.json"
            
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check that we have results
            if not data or len(data) == 0:
                return False
                
            # Check that all expected models have scores
            expected_models = self.models if self.models and self.models != ["all"] else ["openai", "claude", "deepseek", "gemma"]
            
            # For multi-model results
            if isinstance(data[0], dict) and 'model_evaluations' in data[0]:
                for result in data:
                    evaluations = result.get('model_evaluations', {})
                    for model in expected_models:
                        if model not in evaluations:
                            print(f"      Missing evaluation for {model}")
                            return False
                        if evaluations[model] is None:
                            print(f"      Null evaluation for {model}")
                            return False
                            
            return True
            
        except Exception as e:
            print(f"      Error reading detailed results: {e}")
            return False
            
    def validate_statistical_analysis(self):
        """Validate statistical analysis file"""
        file_path = "results/statistics/statistical_analysis.json"
        if not os.path.exists(file_path):
            # Try alternate path
            file_path = "results/statistical_analysis.json"
            
        if not os.path.exists(file_path):
            return False
            
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)
                
            # Check for required fields
            required_fields = ['overall_winner', 'confidence_level', 'model_scores']
            for field in required_fields:
                if field not in data:
                    print(f"      Missing field: {field}")
                    return False
                    
            # Check model scores
            if not data.get('model_scores'):
                print("      No model scores found")
                return False
                
            return True
            
        except Exception as e:
            print(f"      Error reading statistical analysis: {e}")
            return False
            
    def validate_visualizations(self):
        """Validate that visualizations were created"""
        viz_dir = "results/visualizations"
        if not os.path.exists(viz_dir):
            return False
            
        # Check for expected chart files
        expected_charts = [
            "1_overall_comparison.png",
            "2_category_radar.png",
            "3_cost_effectiveness.png",
            "4_safety_metrics.png",
            "5_statistical_summary.png"
        ]
        
        found_charts = []
        for root, dirs, files in os.walk(viz_dir):
            for file in files:
                if file.endswith('.png'):
                    found_charts.append(file)
                    
        if len(found_charts) < 3:  # At least 3 charts expected
            print(f"      Only found {len(found_charts)} charts")
            return False
            
        return True
        
    def run(self):
        """Main execution with retry logic"""
        self.ensure_directories()
        
        print("🧠 Reliable Mental Health LLM Research Runner")
        print("=" * 50)
        
        for attempt in range(1, self.max_retries + 1):
            self.attempt_count = attempt
            
            try:
                # Run the evaluation
                self.run_evaluation(attempt)
                
                # Validate results
                if self.validate_results():
                    print("\n✅ Evaluation completed successfully!")
                    return True
                else:
                    print("\n⚠️  Results validation failed")
                    raise Exception("Incomplete results")
                    
            except Exception as e:
                print(f"\n❌ Attempt {attempt} failed: {e}")
                
                # Save partial results
                if attempt < self.max_retries:
                    self.save_partial_results(attempt)
                    print(f"\n⏳ Waiting {self.retry_delay} seconds before retry...")
                    time.sleep(self.retry_delay)
                else:
                    print("\n❌ All attempts failed")
                    self.save_partial_results(attempt)
                    return False
                    
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Reliable wrapper for Mental Health LLM evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--models",
        help="Comma-separated list of models (default: all models)"
    )
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="Use all available models"
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run quick evaluation (3 scenarios)"
    )
    parser.add_argument(
        "--scenarios",
        type=int,
        help="Number of scenarios to run"
    )
    
    args = parser.parse_args()
    
    # Parse models
    if args.all_models or not args.models:
        models = ["all"]
    else:
        models = [m.strip() for m in args.models.split(",")]
        
    # Create and run the reliable runner
    runner = ReliableResearchRunner(
        models=models,
        quick=args.quick,
        scenarios=args.scenarios
    )
    
    success = runner.run()
    
    if success:
        print("\n🎉 Research completed successfully!")
        print("📊 Check your results in the results/ directory")
        sys.exit(0)
    else:
        print("\n😞 Research failed after all attempts")
        print("💾 Partial results saved in results/development/partial/")
        sys.exit(1)


if __name__ == "__main__":
    main()