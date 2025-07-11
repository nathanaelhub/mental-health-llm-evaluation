"""
Mental Health LLM Evaluator - Streamlined Interface

This module provides a simple, unified interface for comparing mental health LLM responses
across OpenAI GPT-4 and DeepSeek models using standardized therapeutic scenarios.

Usage:
    evaluator = MentalHealthEvaluator()
    results = evaluator.run_evaluation()  # Runs all 10 scenarios
    evaluator.display_results()           # Shows comparison table
    evaluator.save_results()              # Saves to JSON/CSV
"""

import os
import json
import yaml
import time
import csv
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import evaluation system
from .evaluation_metrics import TherapeuticEvaluator, EvaluationResult

# Try to import model interfaces (may not be available in test environments)
try:
    from ..models.openai_client import OpenAIClient
    HAS_OPENAI_CLIENT = True
except ImportError:
    OpenAIClient = None
    HAS_OPENAI_CLIENT = False

try:
    from ..models.deepseek_client import DeepSeekClient
    HAS_DEEPSEEK_CLIENT = True
except ImportError:
    DeepSeekClient = None
    HAS_DEEPSEEK_CLIENT = False


@dataclass
class ScenarioResult:
    """Results for a single scenario evaluation"""
    scenario_id: str
    scenario_name: str
    category: str
    severity: str
    prompt: str
    openai_response: str
    deepseek_response: str
    openai_evaluation: EvaluationResult
    deepseek_evaluation: EvaluationResult
    winner: str
    timestamp: str


@dataclass
class ComparisonSummary:
    """Overall comparison summary across all scenarios"""
    total_scenarios: int
    openai_wins: int
    deepseek_wins: int
    ties: int
    openai_avg_score: float
    deepseek_avg_score: float
    openai_total_cost: float
    deepseek_total_cost: float
    evaluation_time_seconds: float


class MentalHealthEvaluator:
    """
    Streamlined mental health LLM evaluation interface
    
    Provides simple methods to:
    - Load mental health scenarios
    - Generate responses from multiple models
    - Evaluate therapeutic quality
    - Display and save comparison results
    """
    
    def __init__(self, scenarios_file: str = "config/scenarios/main_scenarios.yaml"):
        """
        Initialize the evaluator
        
        Args:
            scenarios_file: Path to YAML file containing scenarios
        """
        self.scenarios_file = scenarios_file
        self.scenarios = []
        self.results = []
        self.summary = None
        
        # Initialize model clients
        self.openai_client = None
        self.deepseek_client = None
        self.evaluator = TherapeuticEvaluator()
        
        # Model settings
        self.temperature = 0.7
        self.max_tokens = 2048
        
        print("🧠 Mental Health LLM Evaluator initialized")
        print("📋 Loading scenarios...")
        self._load_scenarios()
        print(f"✅ Loaded {len(self.scenarios)} scenarios")
    
    def _load_scenarios(self):
        """Load mental health scenarios from YAML file"""
        try:
            with open(self.scenarios_file, 'r') as f:
                data = yaml.safe_load(f)
                self.scenarios = data.get('scenarios', [])
        except FileNotFoundError:
            print(f"❌ Error: {self.scenarios_file} not found")
            raise
        except yaml.YAMLError as e:
            print(f"❌ Error parsing YAML: {e}")
            raise
    
    def _initialize_clients(self):
        """Initialize model clients if not already done"""
        if self.openai_client is None:
            print("🔧 Initializing OpenAI client...")
            try:
                if not HAS_OPENAI_CLIENT or OpenAIClient is None:
                    raise ImportError("OpenAI client not available")
                self.openai_client = OpenAIClient()
                print("✅ OpenAI client ready")
            except Exception as e:
                print(f"❌ OpenAI client failed: {e}")
                raise
        
        if self.deepseek_client is None:
            print("🔧 Initializing DeepSeek client...")
            try:
                if not HAS_DEEPSEEK_CLIENT or DeepSeekClient is None:
                    raise ImportError("DeepSeek client not available")
                self.deepseek_client = DeepSeekClient()
                print("✅ DeepSeek client ready")
            except Exception as e:
                print(f"❌ DeepSeek client failed: {e}")
                raise
    
    def run_evaluation(self, limit: Optional[int] = None) -> List[ScenarioResult]:
        """
        Run evaluation on all scenarios
        
        Args:
            limit: Optional limit on number of scenarios to run
            
        Returns:
            List of scenario results
        """
        print("\n🚀 Starting mental health LLM evaluation...")
        start_time = time.time()
        
        # Initialize clients
        self._initialize_clients()
        
        # Determine scenarios to run
        scenarios_to_run = self.scenarios[:limit] if limit else self.scenarios
        total_scenarios = len(scenarios_to_run)
        
        print(f"📊 Evaluating {total_scenarios} scenarios")
        print("=" * 60)
        
        self.results = []
        openai_total_cost = 0.0
        deepseek_total_cost = 0.0
        
        for i, scenario in enumerate(scenarios_to_run, 1):
            print(f"\n[{i}/{total_scenarios}] {scenario['name']} ({scenario['category']})")
            print("-" * 40)
            
            # Generate responses
            print("🤖 Generating OpenAI response...")
            openai_response, openai_time, openai_cost = self._generate_response(
                self.openai_client, scenario['prompt']
            )
            
            print("🤖 Generating DeepSeek response...")
            deepseek_response, deepseek_time, deepseek_cost = self._generate_response(
                self.deepseek_client, scenario['prompt']
            )
            
            # Evaluate responses
            print("📏 Evaluating responses...")
            openai_eval = self.evaluator.evaluate_response(
                scenario['prompt'], openai_response, 
                response_time_ms=openai_time, 
                input_tokens=len(scenario['prompt'].split()) * 1.3,  # Rough estimate
                output_tokens=len(openai_response.split()) * 1.3
            )
            
            deepseek_eval = self.evaluator.evaluate_response(
                scenario['prompt'], deepseek_response,
                response_time_ms=deepseek_time
            )
            
            # Determine winner
            if openai_eval.composite_score > deepseek_eval.composite_score:
                winner = "OpenAI"
            elif deepseek_eval.composite_score > openai_eval.composite_score:
                winner = "DeepSeek"
            else:
                winner = "Tie"
            
            # Store result
            result = ScenarioResult(
                scenario_id=scenario['id'],
                scenario_name=scenario['name'],
                category=scenario['category'],
                severity=scenario['severity'],
                prompt=scenario['prompt'],
                openai_response=openai_response,
                deepseek_response=deepseek_response,
                openai_evaluation=openai_eval,
                deepseek_evaluation=deepseek_eval,
                winner=winner,
                timestamp=datetime.now().isoformat()
            )
            
            self.results.append(result)
            openai_total_cost += openai_cost
            deepseek_total_cost += deepseek_cost
            
            # Show quick result
            print(f"🏆 Winner: {winner}")
            print(f"   OpenAI: {openai_eval.composite_score:.2f}/10")
            print(f"   DeepSeek: {deepseek_eval.composite_score:.2f}/10")
        
        # Calculate summary
        end_time = time.time()
        self._calculate_summary(openai_total_cost, deepseek_total_cost, end_time - start_time)
        
        print(f"\n✅ Evaluation complete in {end_time - start_time:.1f} seconds")
        return self.results
    
    def _generate_response(self, client, prompt: str) -> tuple[str, float, float]:
        """
        Generate response from a model client
        
        Returns:
            (response_text, response_time_ms, cost_usd)
        """
        start_time = time.time()
        
        try:
            if hasattr(client, 'generate_response'):
                response = client.generate_response(
                    prompt, 
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
            else:
                # Fallback for different interface
                response = client.chat(prompt)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract response text and cost
            if isinstance(response, dict):
                response_text = response.get('content', str(response))
                cost = response.get('cost', 0.0)
            else:
                response_text = str(response)
                cost = 0.0
            
            return response_text, response_time_ms, cost
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}", 0.0, 0.0
    
    def _calculate_summary(self, openai_cost: float, deepseek_cost: float, eval_time: float):
        """Calculate overall comparison summary"""
        if not self.results:
            return
        
        openai_wins = sum(1 for r in self.results if r.winner == "OpenAI")
        deepseek_wins = sum(1 for r in self.results if r.winner == "DeepSeek")
        ties = sum(1 for r in self.results if r.winner == "Tie")
        
        openai_scores = [r.openai_evaluation.composite_score for r in self.results]
        deepseek_scores = [r.deepseek_evaluation.composite_score for r in self.results]
        
        self.summary = ComparisonSummary(
            total_scenarios=len(self.results),
            openai_wins=openai_wins,
            deepseek_wins=deepseek_wins,
            ties=ties,
            openai_avg_score=sum(openai_scores) / len(openai_scores),
            deepseek_avg_score=sum(deepseek_scores) / len(deepseek_scores),
            openai_total_cost=openai_cost,
            deepseek_total_cost=deepseek_cost,
            evaluation_time_seconds=eval_time
        )
    
    def display_results(self):
        """Display results in a clean comparison table"""
        if not self.results:
            print("❌ No results to display. Run evaluation first.")
            return
        
        print("\n" + "=" * 100)
        print("🏆 MENTAL HEALTH LLM EVALUATION RESULTS")
        print("=" * 100)
        
        # Header
        print(f"{'Scenario':<25} {'Category':<12} {'OpenAI':<8} {'DeepSeek':<8} {'Winner':<10}")
        print(f"{'Name':<25} {'Severity':<12} {'Score':<8} {'Score':<8} {'':<10}")
        print("-" * 100)
        
        # Scenario results
        for result in self.results:
            openai_score = f"{result.openai_evaluation.composite_score:.2f}"
            deepseek_score = f"{result.deepseek_evaluation.composite_score:.2f}"
            category_severity = f"{result.category}/{result.severity}"
            
            print(f"{result.scenario_name[:24]:<25} {category_severity[:11]:<12} "
                  f"{openai_score:<8} {deepseek_score:<8} {result.winner:<10}")
        
        print("-" * 100)
        
        # Summary
        if self.summary:
            print(f"\n📊 SUMMARY:")
            print(f"   Total Scenarios: {self.summary.total_scenarios}")
            print(f"   OpenAI Wins: {self.summary.openai_wins}")
            print(f"   DeepSeek Wins: {self.summary.deepseek_wins}")
            print(f"   Ties: {self.summary.ties}")
            print(f"   OpenAI Avg Score: {self.summary.openai_avg_score:.2f}/10")
            print(f"   DeepSeek Avg Score: {self.summary.deepseek_avg_score:.2f}/10")
            print(f"   OpenAI Total Cost: ${self.summary.openai_total_cost:.4f}")
            print(f"   DeepSeek Total Cost: ${self.summary.deepseek_total_cost:.4f}")
            print(f"   Evaluation Time: {self.summary.evaluation_time_seconds:.1f} seconds")
            
            # Overall winner
            if self.summary.openai_avg_score > self.summary.deepseek_avg_score:
                overall_winner = "OpenAI GPT-4"
                score_diff = self.summary.openai_avg_score - self.summary.deepseek_avg_score
            elif self.summary.deepseek_avg_score > self.summary.openai_avg_score:
                overall_winner = "DeepSeek"
                score_diff = self.summary.deepseek_avg_score - self.summary.openai_avg_score
            else:
                overall_winner = "Tie"
                score_diff = 0
            
            print(f"\n🏆 OVERALL WINNER: {overall_winner}")
            if score_diff > 0:
                print(f"   Margin: +{score_diff:.2f} points")
    
    def display_detailed_breakdown(self):
        """Display detailed breakdown by evaluation dimension"""
        if not self.results:
            print("❌ No results to display. Run evaluation first.")
            return
        
        print("\n" + "=" * 120)
        print("📋 DETAILED EVALUATION BREAKDOWN")
        print("=" * 120)
        
        # Header
        print(f"{'Scenario':<20} {'Model':<8} {'Empathy':<8} {'Therapy':<8} {'Safety':<8} {'Clarity':<8} {'Total':<8}")
        print("-" * 120)
        
        for result in self.results:
            # OpenAI row
            openai_eval = result.openai_evaluation
            print(f"{result.scenario_name[:19]:<20} {'OpenAI':<8} "
                  f"{openai_eval.empathy_score:.1f}<8 {openai_eval.therapeutic_value_score:.1f}<8 "
                  f"{openai_eval.safety_score:.1f}<8 {openai_eval.clarity_score:.1f}<8 "
                  f"{openai_eval.composite_score:.2f}<8")
            
            # DeepSeek row
            deepseek_eval = result.deepseek_evaluation
            print(f"{'':<20} {'DeepSeek':<8} "
                  f"{deepseek_eval.empathy_score:.1f}<8 {deepseek_eval.therapeutic_value_score:.1f}<8 "
                  f"{deepseek_eval.safety_score:.1f}<8 {deepseek_eval.clarity_score:.1f}<8 "
                  f"{deepseek_eval.composite_score:.2f}<8")
            print()
    
    def save_results(self, output_dir: str = "data/results") -> Dict[str, str]:
        """
        Save results to JSON and CSV files
        
        Args:
            output_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        if not self.results:
            print("❌ No results to save. Run evaluation first.")
            return {}
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_file = os.path.join(output_dir, f"evaluation_results_{timestamp}.json")
        json_data = {
            "summary": asdict(self.summary) if self.summary else None,
            "scenarios": []
        }
        
        for result in self.results:
            scenario_data = {
                "scenario_id": result.scenario_id,
                "scenario_name": result.scenario_name,
                "category": result.category,
                "severity": result.severity,
                "prompt": result.prompt,
                "openai_response": result.openai_response,
                "deepseek_response": result.deepseek_response,
                "openai_evaluation": result.openai_evaluation.to_dict(),
                "deepseek_evaluation": result.deepseek_evaluation.to_dict(),
                "winner": result.winner,
                "timestamp": result.timestamp
            }
            json_data["scenarios"].append(scenario_data)
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save CSV summary
        csv_file = os.path.join(output_dir, f"evaluation_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            writer.writerow([
                'Scenario ID', 'Scenario Name', 'Category', 'Severity',
                'OpenAI Score', 'DeepSeek Score', 'Winner',
                'OpenAI Empathy', 'OpenAI Therapy', 'OpenAI Safety', 'OpenAI Clarity',
                'DeepSeek Empathy', 'DeepSeek Therapy', 'DeepSeek Safety', 'DeepSeek Clarity'
            ])
            
            # Data rows
            for result in self.results:
                writer.writerow([
                    result.scenario_id,
                    result.scenario_name,
                    result.category,
                    result.severity,
                    f"{result.openai_evaluation.composite_score:.2f}",
                    f"{result.deepseek_evaluation.composite_score:.2f}",
                    result.winner,
                    f"{result.openai_evaluation.empathy_score:.1f}",
                    f"{result.openai_evaluation.therapeutic_value_score:.1f}",
                    f"{result.openai_evaluation.safety_score:.1f}",
                    f"{result.openai_evaluation.clarity_score:.1f}",
                    f"{result.deepseek_evaluation.empathy_score:.1f}",
                    f"{result.deepseek_evaluation.therapeutic_value_score:.1f}",
                    f"{result.deepseek_evaluation.safety_score:.1f}",
                    f"{result.deepseek_evaluation.clarity_score:.1f}"
                ])
        
        file_paths = {
            "json": json_file,
            "csv": csv_file
        }
        
        print(f"\n💾 Results saved:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")
        
        return file_paths


def main():
    """Simple example usage"""
    print("🧠 Mental Health LLM Evaluation Demo")
    
    # Initialize evaluator
    evaluator = MentalHealthEvaluator()
    
    # Run evaluation (limit to 3 scenarios for demo)
    results = evaluator.run_evaluation(limit=3)
    
    # Display results
    evaluator.display_results()
    evaluator.display_detailed_breakdown()
    
    # Save results
    evaluator.save_results()
    
    print("\n✅ Demo complete!")


if __name__ == "__main__":
    main()