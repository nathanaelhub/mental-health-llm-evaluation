"""
Professional Output Formatter for Mental Health LLM Evaluation Study
==================================================================

Clean, academic-style output formatting for research presentations and capstone projects.
Removes verbose progress indicators and focuses on essential research results.
"""

import os
import json
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass


@dataclass
class StudyConfiguration:
    """Configuration for the research study"""
    models: List[str]
    scenarios_count: int
    mode: str  # 'quick' or 'full'
    output_dir: str
    timestamp: str


@dataclass
class ScenarioResult:
    """Results for a single scenario"""
    scenario_name: str
    model_scores: Dict[str, float]  # model_name -> composite_score
    winner: str
    
    def get_sorted_scores(self) -> List[Tuple[str, float]]:
        """Get models sorted by score (highest first)"""
        return sorted(self.model_scores.items(), key=lambda x: x[1], reverse=True)


@dataclass
class StudyResults:
    """Complete study results"""
    config: StudyConfiguration
    scenario_results: List[ScenarioResult]
    overall_rankings: List[Tuple[str, float]]  # (model_name, avg_score)
    statistics: Dict[str, Any]
    export_path: str


class ProfessionalFormatter:
    """
    Professional output formatter for mental health LLM evaluation research.
    
    Provides clean, academic-style output suitable for:
    - Capstone presentations
    - Research reports  
    - Academic papers
    - Professional demonstrations
    """
    
    def __init__(self, quiet_mode: bool = False):
        """
        Initialize formatter.
        
        Args:
            quiet_mode: If True, suppress all non-essential output
        """
        self.quiet_mode = quiet_mode
        self.model_display_names = {
            'openai': 'OpenAI GPT-4',
            'claude': 'Claude 3',
            'deepseek': 'DeepSeek R1', 
            'gemma': 'Gemma 3'
        }
    
    def format_study_header(self, config: StudyConfiguration) -> str:
        """Format the study header"""
        models_display = [self.model_display_names.get(m, m) for m in config.models]
        
        header = [
            "🧠 Mental Health LLM Evaluation Study",
            "=" * 41,
            f"Models: {', '.join(models_display)}",
            f"Scenarios: {config.scenarios_count} ({config.mode.title()} Mode)",
            ""
        ]
        
        return "\n".join(header)
    
    def format_scenario_result(self, result: ScenarioResult) -> str:
        """Format results for a single scenario"""
        sorted_scores = result.get_sorted_scores()
        
        # Format: "Scenario Name: Model1 (8.83) > Model2 (8.36) > Model3 (5.15)"
        score_parts = []
        for model, score in sorted_scores:
            display_name = self.model_display_names.get(model, model)
            score_parts.append(f"{display_name} ({score:.2f})")
        
        score_line = " > ".join(score_parts)
        
        # Pad scenario name for alignment
        scenario_padded = f"{result.scenario_name}:".ljust(22)
        
        return f"{scenario_padded} {score_line}"
    
    def format_overall_rankings(self, rankings: List[Tuple[str, float]], 
                               statistics: Dict[str, Any]) -> str:
        """Format overall performance rankings"""
        lines = [
            "",
            "Overall Performance:",
            "-" * 20
        ]
        
        for i, (model, avg_score) in enumerate(rankings, 1):
            display_name = self.model_display_names.get(model, model)
            
            # Get brief strength description
            strength = self._get_model_strength(model, statistics)
            
            lines.append(f"{i}. {display_name:<12} {avg_score:.2f}/10 ({strength})")
        
        return "\n".join(lines)
    
    def _get_model_strength(self, model: str, statistics: Dict[str, Any]) -> str:
        """Get a brief description of model's strength"""
        # Default descriptions based on typical model characteristics
        strengths = {
            'openai': 'Most consistent therapeutic responses',
            'deepseek': 'Strong reasoning and detailed responses', 
            'claude': 'Balanced across all dimensions',
            'gemma': 'Efficient but needs improvement'
        }
        
        # Try to get actual strength from statistics if available
        if 'model_strengths' in statistics:
            model_strengths = statistics['model_strengths']
            if model in model_strengths:
                return model_strengths[model].get('primary_strength', strengths.get(model, 'Good performance'))
        
        return strengths.get(model, 'Good performance')
    
    def format_completion_summary(self, export_path: str) -> str:
        """Format study completion summary"""
        return "\n".join([
            "",
            "📊 Statistical analysis complete",
            f"💾 Results exported to: {export_path}",
            ""
        ])
    
    def format_complete_study(self, results: StudyResults) -> str:
        """Format complete study results"""
        lines = []
        
        # Header
        lines.append(self.format_study_header(results.config))
        
        # Results section
        lines.append("Results:")
        lines.append("-" * 8)
        
        for scenario_result in results.scenario_results:
            lines.append(self.format_scenario_result(scenario_result))
        
        # Overall rankings
        lines.append(self.format_overall_rankings(results.overall_rankings, results.statistics))
        
        # Completion summary
        lines.append(self.format_completion_summary(results.export_path))
        
        return "\n".join(lines)
    
    def print_study_start(self, config: StudyConfiguration):
        """Print study start information"""
        if not self.quiet_mode:
            print(self.format_study_header(config))
    
    def print_progress_minimal(self, message: str):
        """Print minimal progress information"""
        if not self.quiet_mode:
            print(f"• {message}")
    
    def print_scenario_complete(self, scenario_name: str, results: Dict[str, float]):
        """Print completion of a scenario (minimal)"""
        if not self.quiet_mode:
            winner = max(results.items(), key=lambda x: x[1])
            winner_display = self.model_display_names.get(winner[0], winner[0])
            print(f"  {scenario_name}: {winner_display} ({winner[1]:.2f})")
    
    def print_study_results(self, results: StudyResults):
        """Print complete study results"""
        print(self.format_complete_study(results))
    
    def save_professional_summary(self, results: StudyResults, filename: str):
        """Save professional summary to file"""
        content = self.format_complete_study(results)
        
        with open(filename, 'w') as f:
            f.write(content)
            f.write("\n" + "=" * 60 + "\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    def create_study_results(self, config: StudyConfiguration, raw_results: List[Dict],
                           statistics: Dict[str, Any], export_path: str) -> StudyResults:
        """Convert raw evaluation results to structured study results"""
        
        # Group results by scenario
        scenarios = {}
        for result in raw_results:
            scenario_id = result.get('scenario_id', 'Unknown')
            if scenario_id not in scenarios:
                scenarios[scenario_id] = {}
            
            model = result.get('model', 'unknown')
            evaluation = result.get('evaluation', {})
            composite_score = evaluation.get('composite', evaluation.get('composite_score', 0.0))
            
            scenarios[scenario_id][model] = composite_score
        
        # Create scenario results
        scenario_results = []
        for scenario_id, model_scores in scenarios.items():
            if model_scores:  # Only include scenarios with actual results
                winner = max(model_scores.items(), key=lambda x: x[1])[0]
                scenario_results.append(ScenarioResult(
                    scenario_name=scenario_id.replace('_', ' ').title(),
                    model_scores=model_scores,
                    winner=winner
                ))
        
        # Calculate overall rankings
        model_totals = {}
        model_counts = {}
        
        for scenario_result in scenario_results:
            for model, score in scenario_result.model_scores.items():
                if model not in model_totals:
                    model_totals[model] = 0.0
                    model_counts[model] = 0
                # Defensive programming: ensure no None values get through
                safe_score = score if score is not None else 0.0
                model_totals[model] += safe_score
                model_counts[model] += 1
        
        overall_rankings = []
        for model in model_totals:
            if model_counts[model] > 0:
                avg_score = model_totals[model] / model_counts[model]
                overall_rankings.append((model, avg_score))
        
        overall_rankings.sort(key=lambda x: x[1], reverse=True)
        
        return StudyResults(
            config=config,
            scenario_results=scenario_results,
            overall_rankings=overall_rankings,
            statistics=statistics,
            export_path=export_path
        )


# Convenience functions for backward compatibility
def print_clean_header(models: List[str], scenarios_count: int, mode: str = "full"):
    """Print clean study header"""
    config = StudyConfiguration(
        models=models,
        scenarios_count=scenarios_count,
        mode=mode,
        output_dir="",
        timestamp=""
    )
    formatter = ProfessionalFormatter()
    formatter.print_study_start(config)


def print_clean_results(results: List[Dict], statistics: Dict[str, Any], 
                       export_path: str, models: List[str], scenarios_count: int):
    """Print clean study results"""
    config = StudyConfiguration(
        models=models,
        scenarios_count=scenarios_count,
        mode="full",
        output_dir=os.path.dirname(export_path),
        timestamp=datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
    )
    
    formatter = ProfessionalFormatter()
    study_results = formatter.create_study_results(config, results, statistics, export_path)
    formatter.print_study_results(study_results)


if __name__ == "__main__":
    # Example usage
    config = StudyConfiguration(
        models=['openai', 'deepseek', 'claude', 'gemma'],
        scenarios_count=3,
        mode='quick',
        output_dir='results/',
        timestamp='2025-07-16_14-30'
    )
    
    # Mock scenario results
    scenario_results = [
        ScenarioResult(
            scenario_name="Workplace Anxiety",
            model_scores={'openai': 8.83, 'deepseek': 8.36, 'claude': 5.15, 'gemma': 4.10},
            winner='openai'
        ),
        ScenarioResult(
            scenario_name="Panic Attack Crisis", 
            model_scores={'openai': 8.38, 'deepseek': 7.83, 'claude': 5.72, 'gemma': 4.10},
            winner='openai'
        ),
        ScenarioResult(
            scenario_name="Recurrent Depression",
            model_scores={'deepseek': 9.72, 'openai': 5.86, 'claude': 5.86, 'gemma': 4.10},
            winner='deepseek'
        )
    ]
    
    overall_rankings = [
        ('deepseek', 8.64),
        ('openai', 7.69), 
        ('claude', 5.58),
        ('gemma', 4.10)
    ]
    
    results = StudyResults(
        config=config,
        scenario_results=scenario_results,
        overall_rankings=overall_rankings,
        statistics={},
        export_path='results/study_2025-07-16/'
    )
    
    formatter = ProfessionalFormatter()
    print(formatter.format_complete_study(results))