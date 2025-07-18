"""
Multi-Model Mental Health LLM Evaluator
======================================

This module provides a flexible evaluator that can handle any number of models
simultaneously, with proper statistical analysis and comparison capabilities.
"""

import os
import json
import yaml
import time
import csv
import asyncio
import inspect
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
from dataclasses import dataclass, asdict
import logging

# Import evaluation system
from .evaluation_metrics import TherapeuticEvaluator, EvaluationResult

logger = logging.getLogger(__name__)

@dataclass
class MultiModelScenarioResult:
    """Results for a single scenario evaluation across multiple models"""
    scenario_id: str
    scenario_name: str
    category: str
    severity: str
    prompt: str
    model_responses: Dict[str, str]
    model_evaluations: Dict[str, EvaluationResult]
    winner: str
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'scenario_id': self.scenario_id,
            'scenario_name': self.scenario_name,
            'category': self.category,
            'severity': self.severity,
            'prompt': self.prompt,
            'model_responses': self.model_responses,
            'model_evaluations': {k: v.to_dict() if hasattr(v, 'to_dict') else v 
                                 for k, v in self.model_evaluations.items()},
            'winner': self.winner,
            'timestamp': self.timestamp
        }


@dataclass
class MultiModelSummary:
    """Overall comparison summary across all models"""
    total_scenarios: int
    model_wins: Dict[str, int]
    model_avg_scores: Dict[str, float]
    model_total_costs: Dict[str, float]
    evaluation_time_seconds: float
    
    def get_overall_winner(self) -> str:
        """Get the model with the highest average score"""
        if not self.model_avg_scores:
            return "No Data"
        
        best_model = max(self.model_avg_scores.items(), key=lambda x: x[1])
        winner = best_model[0].title()
        if best_model[0] == 'openai':
            winner = 'OpenAI'
        return winner


class MultiModelEvaluator:
    """
    Multi-model mental health LLM evaluation interface
    
    Supports evaluating any combination of models:
    - OpenAI GPT-4
    - Claude (Anthropic)
    - DeepSeek
    - Gemma
    """
    
    def __init__(self, 
                 selected_models: List[str] = None,
                 scenarios_file: str = "config/scenarios/main_scenarios.yaml"):
        """
        Initialize the multi-model evaluator
        
        Args:
            selected_models: List of model names to evaluate
            scenarios_file: Path to YAML file containing scenarios
        """
        self.selected_models = selected_models or ['openai', 'deepseek']
        self.scenarios_file = scenarios_file
        self.scenarios = []
        self.results = []
        self.summary = None
        
        # Initialize model clients dictionary
        self.model_clients = {}
        self.evaluator = TherapeuticEvaluator()
        
        # Model settings
        self.temperature = 0.7
        self.max_tokens = 2048
        
        logger.info(f"Multi-Model Evaluator initialized for models: {self.selected_models}")
        print(f"🧠 Multi-Model Mental Health LLM Evaluator initialized")
        print(f"🎯 Selected models: {', '.join(self.selected_models)}")
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
        """Initialize all selected model clients"""
        for model_name in self.selected_models:
            if model_name not in self.model_clients:
                self.model_clients[model_name] = self._create_client(model_name)
    
    def _create_client(self, model_name: str):
        """Create a client for the specified model"""
        print(f"🔧 Initializing {model_name} client...")
        
        try:
            if model_name == 'openai':
                from ..models.openai_client import OpenAIClient
                client = OpenAIClient()
                print(f"✅ OpenAI client ready")
                return client
                
            elif model_name == 'claude':
                from ..models.claude_client import ClaudeClient
                client = ClaudeClient()
                print(f"✅ Claude client ready")
                return client
                
            elif model_name == 'deepseek':
                from ..models.deepseek_client import DeepSeekClient
                client = DeepSeekClient()
                print(f"✅ DeepSeek client ready")
                return client
                
            elif model_name == 'gemma':
                from ..models.gemma_client import GemmaClient
                client = GemmaClient()
                print(f"✅ Gemma client ready")
                return client
                
            else:
                raise ValueError(f"Unknown model: {model_name}")
                
        except Exception as e:
            print(f"❌ {model_name} client failed: {e}")
            logger.error(f"Failed to initialize {model_name} client: {e}")
            return None
    
    def run_evaluation(self, limit: Optional[int] = None) -> List[MultiModelScenarioResult]:
        """
        Run evaluation on all scenarios across all selected models
        
        Args:
            limit: Optional limit on number of scenarios to run
            
        Returns:
            List of multi-model scenario results
        """
        print(f"\n🚀 Starting multi-model mental health LLM evaluation...")
        print(f"🎯 Models: {', '.join(self.selected_models)}")
        start_time = time.time()
        
        # Initialize clients
        self._initialize_clients()
        
        # Determine scenarios to run
        scenarios_to_run = self.scenarios[:limit] if limit else self.scenarios
        total_scenarios = len(scenarios_to_run)
        
        print(f"📊 Evaluating {total_scenarios} scenarios across {len(self.selected_models)} models")
        print("=" * 80)
        
        self.results = []
        model_total_costs = {model: 0.0 for model in self.selected_models}
        
        for i, scenario in enumerate(scenarios_to_run, 1):
            print(f"• Scenario {i}/{total_scenarios}: {scenario['name']}")
            
            # Generate responses from all models
            model_responses = {}
            model_evaluations = {}
            
            for model_name in self.selected_models:
                client = self.model_clients.get(model_name)
                if client is None:
                    print(f"  ⚠️ {model_name}: not available")
                    continue
                
                print(f"  ✓ {model_name}: evaluating...")
                response, response_time, cost = self._generate_response(client, scenario['prompt'])
                model_responses[model_name] = response
                model_total_costs[model_name] += cost
                evaluation = self.evaluator.evaluate_response(
                    scenario['prompt'], response,
                    response_time_ms=response_time,
                    input_tokens=len(scenario['prompt'].split()) * 1.3,
                    output_tokens=len(response.split()) * 1.3
                )
                
                # CRITICAL FIX: Standardize evaluation format
                eval_dict = self._standardize_evaluation_format(evaluation)
                model_evaluations[model_name] = eval_dict
            
            # Determine winner
            winner = self._determine_scenario_winner(model_evaluations)
            
            # Store result
            result = MultiModelScenarioResult(
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
            
            # Show minimal result
            winner_score = 0
            if winner in model_evaluations:
                eval_data = model_evaluations[winner]
                if hasattr(eval_data, 'composite_score'):
                    winner_score = eval_data.composite_score
                else:
                    winner_score = eval_data.get('composite', eval_data.get('composite_score', 0))
            print(f"    Winner: {winner} ({winner_score:.2f})")
        
        # Calculate summary
        end_time = time.time()
        self._calculate_summary(model_total_costs, end_time - start_time)
        
        print(f"\n✅ Multi-model evaluation complete in {end_time - start_time:.1f} seconds")
        print(f"🏆 Overall winner: {self.summary.get_overall_winner()}")
        
        return self.results
    
    def _generate_response(self, client, prompt: str) -> tuple[str, float, float]:
        """Generate response from a model client"""
        start_time = time.time()
        
        try:
            if hasattr(client, 'generate_response'):
                method = getattr(client, 'generate_response')
                
                if inspect.iscoroutinefunction(method):
                    response = asyncio.run(method(
                        prompt, 
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    ))
                else:
                    response = method(
                        prompt, 
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
            else:
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
    
    def _standardize_evaluation_format(self, evaluation) -> Dict[str, Any]:
        """Standardize evaluation format to ensure consistent data structure"""
        if hasattr(evaluation, 'to_dict'):
            eval_dict = evaluation.to_dict()
        else:
            eval_dict = evaluation if isinstance(evaluation, dict) else {}
        
        # Ensure all required fields exist
        required_fields = {
            'empathy': ['empathy', 'empathy_score'],
            'therapeutic': ['therapeutic', 'therapeutic_value_score'],
            'safety': ['safety', 'safety_score'], 
            'clarity': ['clarity', 'clarity_score']
        }
        
        standardized = {}
        
        # Extract component scores
        for standard_key, possible_keys in required_fields.items():
            value = 0.0
            for key in possible_keys:
                if key in eval_dict and eval_dict[key] is not None:
                    value = float(eval_dict[key])
                    break
            standardized[standard_key] = value
            standardized[f"{standard_key}_score"] = value
        
        # Calculate composite score
        composite = (
            standardized['empathy'] * 0.3 +
            standardized['therapeutic'] * 0.25 +
            standardized['safety'] * 0.35 +
            standardized['clarity'] * 0.1
        )
        standardized['composite'] = composite
        standardized['composite_score'] = composite
        
        # Copy other fields
        for key, value in eval_dict.items():
            if key not in standardized:
                standardized[key] = value
                
        return standardized

    def _determine_scenario_winner(self, model_evaluations: Dict[str, EvaluationResult]) -> str:
        """Determine winner for a single scenario"""
        if not model_evaluations:
            return "No Data"
        
        best_score = -1
        winner = "Tie"
        
        for model_name, evaluation in model_evaluations.items():
            score = evaluation.composite_score if hasattr(evaluation, 'composite_score') else 0.0
            if score > best_score:
                best_score = score
                winner = model_name.title()
                if model_name == 'openai':
                    winner = 'OpenAI'
        
        return winner
    
    def _calculate_summary(self, model_total_costs: Dict[str, float], eval_time: float):
        """Calculate overall multi-model comparison summary"""
        if not self.results:
            return
        
        # Count wins for each model
        model_wins = {model: 0 for model in self.selected_models}
        for result in self.results:
            winner_key = result.winner.lower()
            if winner_key == 'openai':
                winner_key = 'openai'
            elif winner_key in model_wins:
                model_wins[winner_key] += 1
        
        # Calculate average scores
        model_avg_scores = {}
        for model_name in self.selected_models:
            scores = []
            for result in self.results:
                if model_name in result.model_evaluations:
                    eval_result = result.model_evaluations[model_name]
                    score = eval_result.composite_score if hasattr(eval_result, 'composite_score') else 0.0
                    scores.append(score)
            
            model_avg_scores[model_name] = sum(scores) / len(scores) if scores else 0.0
        
        self.summary = MultiModelSummary(
            total_scenarios=len(self.results),
            model_wins=model_wins,
            model_avg_scores=model_avg_scores,
            model_total_costs=model_total_costs,
            evaluation_time_seconds=eval_time
        )
    
    def display_results(self):
        """Display results in a comprehensive comparison table"""
        if not self.results:
            print("❌ No results to display. Run evaluation first.")
            return
        
        print("\n" + "=" * 120)
        print("🏆 MULTI-MODEL MENTAL HEALTH LLM EVALUATION RESULTS")
        print("=" * 120)
        
        # Header
        header = f"{'Scenario':<25} {'Category':<12}"
        for model in self.selected_models:
            header += f" {model.title():<8}"
        header += f" {'Winner':<10}"
        print(header)
        
        subheader = f"{'Name':<25} {'Severity':<12}"
        for model in self.selected_models:
            subheader += f" {'Score':<8}"
        subheader += f" {'':<10}"
        print(subheader)
        print("-" * 120)
        
        # Scenario results
        for result in self.results:
            line = f"{result.scenario_name[:24]:<25} {result.category[:11]:<12}"
            
            for model in self.selected_models:
                if model in result.model_evaluations:
                    eval_result = result.model_evaluations[model]
                    score = eval_result.composite_score if hasattr(eval_result, 'composite_score') else 0.0
                    line += f" {score:.2f:<8}"
                else:
                    line += f" {'N/A':<8}"
            
            line += f" {result.winner:<10}"
            print(line)
        
        print("-" * 120)
        
        # Summary
        if self.summary:
            print(f"\n📊 MULTI-MODEL SUMMARY:")
            print(f"   Total Scenarios: {self.summary.total_scenarios}")
            
            for model in self.selected_models:
                wins = self.summary.model_wins.get(model, 0)
                avg_score = self.summary.model_avg_scores.get(model, 0.0)
                total_cost = self.summary.model_total_costs.get(model, 0.0)
                
                print(f"   {model.title()} - Wins: {wins}, Avg Score: {avg_score:.2f}/10, Cost: ${total_cost:.4f}")
            
            print(f"   Evaluation Time: {self.summary.evaluation_time_seconds:.1f} seconds")
            print(f"\n🏆 OVERALL WINNER: {self.summary.get_overall_winner()}")
    
    def save_results(self, results_dir: str = "results/multi_model_evaluations") -> Dict[str, str]:
        """Save multi-model results to JSON and CSV files"""
        if not self.results:
            print("❌ No results to save. Run evaluation first.")
            return {}
        
        os.makedirs(results_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed JSON results
        json_file = os.path.join(results_dir, f"multi_model_results_{timestamp}.json")
        json_data = {
            "summary": asdict(self.summary) if self.summary else None,
            "models_evaluated": self.selected_models,
            "scenarios": [result.to_dict() for result in self.results]
        }
        
        with open(json_file, 'w') as f:
            json.dump(json_data, f, indent=2)
        
        # Save CSV summary
        csv_file = os.path.join(results_dir, f"multi_model_summary_{timestamp}.csv")
        with open(csv_file, 'w', newline='') as f:
            writer = csv.writer(f)
            
            # Header
            header = ['Scenario ID', 'Scenario Name', 'Category', 'Severity']
            for model in self.selected_models:
                header.extend([f'{model.title()} Score', f'{model.title()} Response'])
            header.append('Winner')
            writer.writerow(header)
            
            # Data rows
            for result in self.results:
                row = [result.scenario_id, result.scenario_name, result.category, result.severity]
                
                for model in self.selected_models:
                    if model in result.model_evaluations:
                        eval_result = result.model_evaluations[model]
                        score = eval_result.composite_score if hasattr(eval_result, 'composite_score') else 0.0
                        response = result.model_responses.get(model, "N/A")
                        row.extend([f"{score:.2f}", response[:100] + "..." if len(response) > 100 else response])
                    else:
                        row.extend(["N/A", "N/A"])
                
                row.append(result.winner)
                writer.writerow(row)
        
        file_paths = {"json": json_file, "csv": csv_file}
        
        print(f"\n💾 Multi-model results saved:")
        print(f"   JSON: {json_file}")
        print(f"   CSV: {csv_file}")
        
        return file_paths