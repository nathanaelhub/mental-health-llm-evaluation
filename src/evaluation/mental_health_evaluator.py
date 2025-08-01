"""
Mental Health LLM Evaluator - Multi-Model Interface

This module provides a unified interface for comparing mental health LLM responses
across OpenAI GPT-4, Claude, DeepSeek, and Gemma models using standardized therapeutic scenarios.

Usage:
    evaluator = MentalHealthEvaluator(models=['openai', 'claude', 'deepseek', 'gemma'])
    results = evaluator.run_evaluation()  # Runs all scenarios on selected models
    evaluator.display_results()           # Shows comparison table
    evaluator.save_results()              # Saves to JSON/CSV
"""

import os
import json
import yaml
import time
import csv
import asyncio
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict

# Import evaluation system
from .evaluation_metrics import TherapeuticEvaluator, EvaluationResult

# Import unified wrapper
try:
    from ..models.unified_client_wrapper import UnifiedModelClient
    HAS_UNIFIED_WRAPPER = True
except ImportError:
    UnifiedModelClient = None
    HAS_UNIFIED_WRAPPER = False

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

try:
    from ..models.claude_client import ClaudeClient
    HAS_CLAUDE_CLIENT = True
except ImportError:
    ClaudeClient = None
    HAS_CLAUDE_CLIENT = False

try:
    from ..models.gemma_client import GemmaClient
    HAS_GEMMA_CLIENT = True
except ImportError:
    GemmaClient = None
    HAS_GEMMA_CLIENT = False


@dataclass
class ScenarioResult:
    """Results for a single scenario evaluation"""
    scenario_id: str
    scenario_name: str
    category: str
    severity: str
    prompt: str
    model_responses: Dict[str, str]  # model_name -> response
    model_evaluations: Dict[str, EvaluationResult]  # model_name -> evaluation
    winner: str
    timestamp: str


@dataclass
class ComparisonSummary:
    """Overall comparison summary across all scenarios"""
    total_scenarios: int
    model_wins: Dict[str, int]  # model_name -> win_count
    ties: int
    model_avg_scores: Dict[str, float]  # model_name -> avg_score
    model_total_costs: Dict[str, float]  # model_name -> total_cost
    evaluation_time_seconds: float
    models_evaluated: List[str]


class MentalHealthEvaluator:
    """
    Multi-model mental health LLM evaluation interface
    
    Provides simple methods to:
    - Load mental health scenarios
    - Generate responses from multiple models (OpenAI, Claude, DeepSeek, Gemma)
    - Evaluate therapeutic quality
    - Display and save comparison results
    """
    
    def __init__(self, models: List[str] = None, scenarios_file: str = "config/scenarios/main_scenarios.yaml"):
        """
        Initialize the evaluator
        
        Args:
            models: List of model names to evaluate (default: ['openai', 'deepseek'])
            scenarios_file: Path to YAML file containing scenarios
        """
        self.scenarios_file = scenarios_file
        self.scenarios = []
        self.results = []
        self.summary = None
        
        # Set default models if none provided
        self.selected_models = models or ['openai', 'deepseek']
        
        # Initialize model clients dictionary
        self.model_clients = {}
        self.evaluator = TherapeuticEvaluator()
        
        # Model settings
        self.temperature = 0.7
        self.max_tokens = 2048
        
        print("🧠 Mental Health LLM Evaluator initialized")
        print(f"📋 Selected models: {', '.join(self.selected_models)}")
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
        """Initialize model clients for selected models"""
        for model_name in self.selected_models:
            if model_name in self.model_clients:
                continue  # Already initialized
            
            print(f"🔧 Initializing {model_name.upper()} client...")
            try:
                if model_name == 'openai':
                    if not HAS_OPENAI_CLIENT or OpenAIClient is None:
                        raise ImportError("OpenAI client not available")
                    client = OpenAIClient()
                elif model_name == 'deepseek':
                    if not HAS_DEEPSEEK_CLIENT or DeepSeekClient is None:
                        raise ImportError("DeepSeek client not available")
                    client = DeepSeekClient()
                elif model_name == 'claude':
                    if not HAS_CLAUDE_CLIENT or ClaudeClient is None:
                        raise ImportError("Claude client not available")
                    client = ClaudeClient()
                elif model_name == 'gemma':
                    if not HAS_GEMMA_CLIENT or GemmaClient is None:
                        raise ImportError("Gemma client not available")
                    client = GemmaClient()
                else:
                    raise ValueError(f"Unknown model: {model_name}")
                
                # Wrap with unified client if available
                if HAS_UNIFIED_WRAPPER and UnifiedModelClient is not None:
                    self.model_clients[model_name] = UnifiedModelClient(client)
                else:
                    self.model_clients[model_name] = client
                    
                print(f"✅ {model_name.upper()} client ready")
            except Exception as e:
                print(f"❌ {model_name.upper()} client failed: {e}")
                # Remove from selected models if initialization fails
                if model_name in self.selected_models:
                    self.selected_models.remove(model_name)
                    print(f"⚠️  Removed {model_name} from evaluation")
    
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
        model_total_costs = {model: 0.0 for model in self.selected_models}
        
        for i, scenario in enumerate(scenarios_to_run, 1):
            print(f"\n[{i}/{total_scenarios}] {scenario['name']} ({scenario['category']})")
            print("-" * 40)
            
            # Generate responses from all models
            model_responses = {}
            model_evaluations = {}
            model_times = {}
            
            for model_name in self.selected_models:
                print(f"🤖 Generating {model_name.upper()} response...")
                response, response_time, cost = self._generate_response(
                    self.model_clients[model_name], scenario['prompt']
                )
                model_responses[model_name] = response
                model_times[model_name] = response_time
                model_total_costs[model_name] += (cost or 0.0)
            
            # Evaluate all responses
            print("📏 Evaluating responses...")
            for model_name in self.selected_models:
                evaluation = self.evaluator.evaluate_response(
                    scenario['prompt'], model_responses[model_name],
                    response_time_ms=model_times[model_name],
                    input_tokens=len(scenario['prompt'].split()) * 1.3,  # Rough estimate
                    output_tokens=len(model_responses[model_name].split()) * 1.3
                )
                model_evaluations[model_name] = evaluation
            
            # Determine winner
            winner = self._determine_winner(model_evaluations)
            
            # Store result
            result = ScenarioResult(
                scenario_id=scenario['id'],
                scenario_name=scenario['name'],
                category=scenario['category'],
                severity=scenario['severity'],
                prompt=scenario['prompt'],
                model_responses=model_responses,
                model_evaluations=model_evaluations,
                winner=winner,
                timestamp=datetime.now().isoformat()
            )
            
            self.results.append(result)
            
            # Show quick result
            print(f"🏆 Winner: {winner}")
            for model_name in self.selected_models:
                score = model_evaluations[model_name].composite_score
                print(f"   {model_name.upper()}: {score:.2f}/10")
        
        # Calculate summary
        end_time = time.time()
        self._calculate_summary(model_total_costs, end_time - start_time)
        
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
                method = getattr(client, 'generate_response')
                
                # Check if the method is async
                if inspect.iscoroutinefunction(method):
                    # Run async method with asyncio
                    response = asyncio.run(method(
                        prompt, 
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    ))
                else:
                    # Call sync method normally
                    response = method(
                        prompt, 
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
            else:
                # Fallback for different interface
                response = client.chat(prompt)
            
            response_time_ms = (time.time() - start_time) * 1000
            
            # Extract response text and cost
            if hasattr(response, 'content'):
                response_text = response.content
                cost = getattr(response, 'cost_usd', 0.0)
            elif isinstance(response, dict):
                response_text = response.get('content', str(response))
                cost = response.get('cost_usd', response.get('cost', 0.0))
            else:
                response_text = str(response)
                cost = 0.0
            
            return response_text, response_time_ms, cost
            
        except Exception as e:
            print(f"❌ Error generating response: {e}")
            return f"Error: {str(e)}", 0.0, 0.0
    
    def _determine_winner(self, model_evaluations: Dict[str, EvaluationResult]) -> str:
        """Determine the winner from model evaluations"""
        if not model_evaluations:
            return "No models"
        
        # Find the highest score
        max_score = max(eval.composite_score for eval in model_evaluations.values())
        winners = [model for model, eval in model_evaluations.items() 
                  if eval.composite_score == max_score]
        
        if len(winners) == 1:
            return winners[0].upper()
        else:
            return "Tie"
    
    def _calculate_summary(self, model_total_costs: Dict[str, float], eval_time: float):
        """Calculate overall comparison summary"""
        if not self.results:
            return
        
        # Count wins for each model
        model_wins = {model: 0 for model in self.selected_models}
        ties = 0
        
        for result in self.results:
            if result.winner == "Tie":
                ties += 1
            else:
                winner_lower = result.winner.lower()
                if winner_lower in model_wins:
                    model_wins[winner_lower] += 1
        
        # Calculate average scores
        model_avg_scores = {}
        for model in self.selected_models:
            scores = [r.model_evaluations[model].composite_score for r in self.results]
            model_avg_scores[model] = sum(scores) / len(scores) if scores else 0.0
        
        self.summary = ComparisonSummary(
            total_scenarios=len(self.results),
            model_wins=model_wins,
            ties=ties,
            model_avg_scores=model_avg_scores,
            model_total_costs=model_total_costs,
            evaluation_time_seconds=eval_time,
            models_evaluated=self.selected_models.copy()
        )
    
    def display_results(self):
        """Display results in a clean comparison table"""
        if not self.results:
            print("❌ No results to display. Run evaluation first.")
            return
        
        models_count = len(self.selected_models)
        table_width = 40 + (models_count * 10) + 10  # Base + model columns + winner
        
        print("\n" + "=" * table_width)
        print("🏆 MENTAL HEALTH LLM EVALUATION RESULTS")
        print("=" * table_width)
        
        # Header
        header = f"{'Scenario':<25} {'Category':<12} "
        for model in self.selected_models:
            header += f"{model.upper():<8} "
        header += f"{'Winner':<10}"
        print(header)
        
        subheader = f"{'Name':<25} {'Severity':<12} "
        for _ in self.selected_models:
            subheader += f"{'Score':<8} "
        subheader += f"{'':<10}"
        print(subheader)
        print("-" * table_width)
        
        # Scenario results
        for result in self.results:
            category_severity = f"{result.category}/{result.severity}"
            row = f"{result.scenario_name[:24]:<25} {category_severity[:11]:<12} "
            
            for model in self.selected_models:
                score = f"{result.model_evaluations[model].composite_score:.2f}"
                row += f"{score:<8} "
            
            row += f"{result.winner:<10}"
            print(row)
        
        print("-" * table_width)
        
        # Summary
        if self.summary:
            print(f"\n📊 SUMMARY:")
            print(f"   Total Scenarios: {self.summary.total_scenarios}")
            
            # Model wins
            for model in self.selected_models:
                wins = self.summary.model_wins.get(model, 0)
                print(f"   {model.upper()} Wins: {wins}")
            print(f"   Ties: {self.summary.ties}")
            
            # Average scores
            for model in self.selected_models:
                avg_score = self.summary.model_avg_scores.get(model, 0.0)
                print(f"   {model.upper()} Avg Score: {avg_score:.2f}/10")
            
            # Costs
            for model in self.selected_models:
                cost = self.summary.model_total_costs.get(model, 0.0)
                print(f"   {model.upper()} Total Cost: ${cost:.4f}")
            
            print(f"   Evaluation Time: {self.summary.evaluation_time_seconds:.1f} seconds")
            
            # Overall winner
            if self.summary.model_avg_scores:
                best_model = max(self.summary.model_avg_scores.items(), key=lambda x: x[1])
                best_models = [m for m, s in self.summary.model_avg_scores.items() if s == best_model[1]]
                
                if len(best_models) == 1:
                    print(f"\n🏆 OVERALL WINNER: {best_models[0].upper()}")
                    print(f"   Best Score: {best_model[1]:.2f}/10")
                else:
                    print(f"\n🏆 OVERALL RESULT: Tie between {', '.join(m.upper() for m in best_models)}")
                    print(f"   Tied Score: {best_model[1]:.2f}/10")
    
    def display_detailed_breakdown(self):
        """Display detailed breakdown by evaluation dimension"""
        if not self.results:
            print("❌ No results to display. Run evaluation first.")
            return
        
        table_width = 80 + (len(self.selected_models) * 50)
        print("\n" + "=" * min(table_width, 150))
        print("📋 DETAILED EVALUATION BREAKDOWN")
        print("=" * min(table_width, 150))
        
        # Header
        print(f"{'Scenario':<20} {'Model':<10} {'Empathy':<8} {'Therapy':<8} {'Safety':<8} {'Clarity':<8} {'Total':<8}")
        print("-" * min(table_width, 150))
        
        for result in self.results:
            first_model = True
            for model in self.selected_models:
                evaluation = result.model_evaluations[model]
                scenario_name = result.scenario_name[:19] if first_model else ''
                
                print(f"{scenario_name:<20} {model.upper():<10} "
                      f"{evaluation.empathy_score:.1f}{'<8'} {evaluation.therapeutic_value_score:.1f}{'<8'} "
                      f"{evaluation.safety_score:.1f}{'<8'} {evaluation.clarity_score:.1f}{'<8'} "
                      f"{evaluation.composite_score:.2f}{'<8'}")
                first_model = False
            print()
    
    def save_results(self, results_dir: str = "results/evaluations") -> Dict[str, str]:
        """
        Save results to JSON and CSV files
        
        Args:
            results_dir: Directory to save results
            
        Returns:
            Dictionary with file paths
        """
        if not self.results:
            print("❌ No results to save. Run evaluation first.")
            return {}
        
        # Create output directory
        os.makedirs(results_dir, exist_ok=True)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_file = os.path.join(results_dir, f"evaluation_results_{timestamp}.json")
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
                "model_responses": result.model_responses,
                "model_evaluations": {model: eval.to_dict() for model, eval in result.model_evaluations.items()},
                "winner": result.winner,
                "timestamp": result.timestamp
            }
            json_data["scenarios"].append(scenario_data)
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save CSV summary
        csv_file = os.path.join(results_dir, f"evaluation_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Dynamic header based on selected models
            header = ['Scenario ID', 'Scenario Name', 'Category', 'Severity']
            
            # Add score columns for each model
            for model in self.selected_models:
                header.append(f'{model.upper()} Score')
            header.append('Winner')
            
            # Add detailed metric columns for each model
            for model in self.selected_models:
                header.extend([
                    f'{model.upper()} Empathy',
                    f'{model.upper()} Therapy',
                    f'{model.upper()} Safety',
                    f'{model.upper()} Clarity'
                ])
            
            writer.writerow(header)
            
            # Data rows
            for result in self.results:
                row = [
                    result.scenario_id,
                    result.scenario_name,
                    result.category,
                    result.severity
                ]
                
                # Add scores for each model
                for model in self.selected_models:
                    evaluation = result.model_evaluations[model]
                    row.append(f"{evaluation.composite_score:.2f}")
                row.append(result.winner)
                
                # Add detailed metrics for each model
                for model in self.selected_models:
                    evaluation = result.model_evaluations[model]
                    row.extend([
                        f"{evaluation.empathy_score:.1f}",
                        f"{evaluation.therapeutic_value_score:.1f}",
                        f"{evaluation.safety_score:.1f}",
                        f"{evaluation.clarity_score:.1f}"
                    ])
                
                writer.writerow(row)
        
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
    
    # Initialize evaluator with default models
    evaluator = MentalHealthEvaluator(models=['openai', 'deepseek'])
    
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