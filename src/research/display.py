"""
Display Module
==============

All UI/display functions for the Mental Health LLM Evaluation Research.
Handles progress bars, formatting, scenario results, and summary displays.

Key Components:
- Progress tracking and demo mode displays
- Scenario result formatting (standard, minimal, ultra-clean)
- Summary displays and research headers
- Model display name utilities
"""

import time
from datetime import datetime
from typing import Dict, Any, List, Optional
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeRemainingColumn, TimeElapsedColumn, MofNCompleteColumn
from rich.table import Table
from rich.panel import Panel
from rich.live import Live
from rich.layout import Layout
from rich.align import Align
from rich.text import Text
from rich import box

# Initialize rich console
console = Console()

# =============================================================================
# OUTPUT CONTROL FUNCTIONS
# =============================================================================

def conditional_print(message, quiet=False, ultra_clean=False):
    """Print message only if not in quiet mode"""
    if not quiet and not ultra_clean:
        print(message)

def ultra_clean_print(message):
    """Print message in ultra-clean mode (always prints)"""
    print(message)

def minimal_print(message):
    """Print message in minimal mode (always prints)"""
    print(message)

def demo_print(message, demo_mode=False):
    """Print message only in demo mode"""
    if demo_mode:
        print(message)

# =============================================================================
# DEMO MODE DISPLAYS
# =============================================================================

def print_demo_header(num_models, num_scenarios):
    """Print clean header for demo mode"""
    print("\n🧠 Mental Health LLM Evaluation Study")
    print(f"Comparing {num_models} models on {num_scenarios} scenarios...")

def print_demo_progress(current, total, start_time=None):
    """Print simple progress bar for demo mode with time"""
    progress = int((current / total) * 40)
    bar = "━" * progress + "╺" * (40 - progress)
    percentage = int((current / total) * 100)
    
    # Calculate time info
    time_str = ""
    if start_time:
        elapsed = time.time() - start_time
        if elapsed > 0 and current > 0:
            rate = current / elapsed
            if rate > 0:
                remaining = (total - current) / rate
                time_str = f" | ETA: {remaining:.0f}s"
    
    print(f"Progress: [{bar}] {percentage:3d}% ({current}/{total}){time_str}")

def print_demo_results(analysis, results_dir, chart_files):
    """Print clean results summary for demo mode"""
    print("\n" + "=" * 50)
    print("EVALUATION RESULTS")
    print("Model Rankings:")
    
    if hasattr(analysis, 'model_stats') and analysis.model_stats:
        # Multi-model display
        model_names = getattr(analysis, 'model_names', list(analysis.model_stats.keys()))
        
        # Sort models by average score for ranking
        model_scores = [(name, analysis.model_stats[name].get('composite', {}).get('mean', 0.0)) 
                       for name in model_names]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(model_scores, 1):
            display_name = get_model_display_name(model_name)
            print(f"{i}. {display_name:<10} {score:.2f}/10")
    else:
        # Default OpenAI/DeepSeek display
        openai_score = analysis.openai_stats.get('composite', {}).get('mean', 0.0)
        deepseek_score = analysis.deepseek_stats.get('composite', {}).get('mean', 0.0)
        
        scores = [("OpenAI", openai_score), ("DeepSeek", deepseek_score)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(scores, 1):
            print(f"{i}. {name:<10} {score:.2f}/10")
    
    print(f"\nWinner: {analysis.overall_winner}")
    print(f"Charts saved to: {results_dir}/visualizations/")

# =============================================================================
# MODEL UTILITIES
# =============================================================================

def get_model_display_name(model_key: str) -> str:
    """Get consistent display name for a model key"""
    model_display = {
        'openai': 'OpenAI',
        'claude': 'Claude', 
        'deepseek': 'DeepSeek',
        'gemma': 'Gemma'
    }
    return model_display.get(model_key.lower(), model_key.title())

def extract_composite_score(evaluation) -> float:
    """Extract composite score from evaluation dict or object"""
    if evaluation is None:
        return 0.0
    
    if isinstance(evaluation, dict):
        return evaluation.get('composite', 0.0)
    else:
        return getattr(evaluation, 'composite_score', 0.0)

def build_model_scores(evaluations: Dict[str, Any]) -> Dict[str, float]:
    """Build model scores dictionary from evaluations"""
    model_scores = {}
    for model_key, evaluation in evaluations.items():
        if model_key in ['openai', 'claude', 'deepseek', 'gemma']:
            score = extract_composite_score(evaluation)
            model_scores[model_key] = score
    return model_scores

# =============================================================================
# HEADER DISPLAYS
# =============================================================================

def print_header(models):
    """Print the research study header using rich formatting"""
    header_text = Text()
    header_text.append("🧠 Mental Health LLM Evaluation Research Study", style="bold magenta")
    
    header_panel = Panel(
        Align.center(header_text),
        title="Research Study",
        title_align="center",
        border_style="cyan",
        padding=(1, 2)
    )
    
    console.print()
    console.print(header_panel)
    
    # Study details table
    study_table = Table(show_header=False, box=box.ROUNDED, padding=(0, 1))
    study_table.add_column("Field", style="cyan", min_width=15)
    study_table.add_column("Value", style="white")
    
    study_table.add_row("📋 Study Type", "Comparing Multiple LLM Models for Therapeutic Conversations")
    study_table.add_row("🤖 Models", f"{len(models)} models selected: {', '.join(models)}")
    study_table.add_row("🎯 Purpose", "Academic Capstone Project - Statistical Analysis & Recommendations")
    study_table.add_row("📅 Started", datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    
    console.print(study_table)
    console.print()

def print_clean_header(models, num_scenarios):
    """Print clean, minimal header for professional output"""
    console.print(f"🧠 Mental Health LLM Research Study")
    console.print(f"📊 Evaluating {num_scenarios} scenarios across {len(models)} models...")

# =============================================================================
# SCENARIO RESULT DISPLAYS
# =============================================================================

class ScenarioResultDisplayer:
    """Handles different modes of scenario result display"""
    
    @staticmethod
    def print_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner):
        """Print clean scenario result in desired format"""
        # Format like: [1/3] General Workplace Anxiety
        console.print(f"[{scenario_num}/{total_scenarios}] {scenario_name}")
        
        # Show model scores in a clean line
        result_line = "  "
        model_info = {
            'openai': 'OpenAI GPT-4',
            'claude': 'Claude',
            'deepseek': 'DeepSeek',
            'gemma': 'Gemma'
        }
        
        for model_key, evaluation in evaluations.items():
            if model_key in model_info:
                model_name = model_info[model_key]
                score = extract_composite_score(evaluation)
                result_line += f"{model_name}: {score:.2f}/10    "
        
        console.print(result_line.rstrip())

    @staticmethod
    def print_minimal_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner):
        """Print minimal scenario result with ranking of all models"""
        model_scores = build_model_scores(evaluations)
        
        # Sort models by score (descending)
        sorted_models = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Build ranking string
        ranking_parts = []
        for model_key, score in sorted_models:
            display_name = get_model_display_name(model_key)
            ranking_parts.append(f"{display_name} ({score:.1f})")
        
        ranking_str = " > ".join(ranking_parts)
        
        # Print result
        print(f"[{scenario_num}/{total_scenarios}] {scenario_name}: {ranking_str}")

    @staticmethod
    def print_ultra_clean_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner, demo_mode=False):
        """Print ultra-clean scenario result - single line format"""
        # In demo mode, just show a simple checkmark
        if demo_mode:
            ultra_clean_print(f"[{scenario_num}/{total_scenarios}] {scenario_name}: ✓")
            return
            
        model_scores = build_model_scores(evaluations)
        
        # Check if all scores are 0.0 (indicates scores haven't been properly calculated yet)
        all_scores_zero = all(score == 0.0 for score in model_scores.values())
        
        if all_scores_zero:
            # Don't show misleading (0.00) scores - just show progress
            ultra_clean_print(f"[{scenario_num}/{total_scenarios}] {scenario_name}: ✓")
            return
        
        # Build comparison string with actual scores
        if len(model_scores) == 2:
            # Two model comparison format
            models = list(model_scores.keys())
            model1, model2 = models[0], models[1]
            score1, score2 = model_scores[model1], model_scores[model2]
            
            display1 = get_model_display_name(model1)
            display2 = get_model_display_name(model2)
            winner_display = get_model_display_name(winner.lower())
            
            ultra_clean_print(f"[{scenario_num}/{total_scenarios}] {scenario_name}: {display1} ({score1:.1f}) vs {display2} ({score2:.1f}) → {winner_display} wins")
        else:
            # Multi-model format
            score_strings = []
            for model_key, score in model_scores.items():
                display_name = get_model_display_name(model_key)
                score_strings.append(f"{display_name} ({score:.1f})")
            
            winner_display = get_model_display_name(winner.lower())
            ultra_clean_print(f"[{scenario_num}/{total_scenarios}] {scenario_name}: {' vs '.join(score_strings)} → {winner_display} wins")

# Convenience functions for backward compatibility
def print_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner):
    ScenarioResultDisplayer.print_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner)

def print_minimal_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner):
    ScenarioResultDisplayer.print_minimal_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner)

def print_ultra_clean_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner, demo_mode=False):
    ScenarioResultDisplayer.print_ultra_clean_scenario_result(scenario_num, total_scenarios, scenario_name, evaluations, winner, demo_mode)

# =============================================================================
# SUMMARY DISPLAYS
# =============================================================================

def display_minimal_summary(analysis, results, output_dir):
    """Display results summary with professional formatting."""
    if not analysis:
        print("Error: No analysis results available")
        return
    
    print()
    
    # Professional box format
    print("╔════════════════════════════════════════════════════════════╗")
    print("║                    EVALUATION RESULTS                       ║")
    print("╠════════════════════════════════════════════════════════════╣")
    
    # Winner section
    confidence_text = f"{analysis.confidence_level.title()} Confidence"
    winner_line = f"║ 🏆 Overall Winner: {analysis.overall_winner} ({confidence_text})"
    padding = 60 - len(winner_line) + 1
    print(f"{winner_line}{' ' * padding}║")
    
    print("╠════════════════════════════════════════════════════════════╣")
    print("║ Model Rankings:                                            ║")
    
    # Check if we have multi-model results
    if hasattr(analysis, 'model_stats') and analysis.model_stats:
        # Multi-model display
        model_names = getattr(analysis, 'model_names', list(analysis.model_stats.keys()))
        
        # Sort models by average score
        model_scores = [(name, analysis.model_stats[name].get('composite', {}).get('mean', 0.0)) 
                       for name in model_names]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (model_name, score) in enumerate(model_scores, 1):
            display_name = get_model_display_name(model_name)
            
            # Create progress bar (20 chars max)
            bar_length = int((score / 10.0) * 20)
            bar = "█" * bar_length + " " * (20 - bar_length)
            
            model_line = f"║   {i}. {display_name:<8} {bar} {score:.2f}/10"
            line_padding = 60 - len(model_line) + 1
            print(f"{model_line}{' ' * line_padding}║")
    else:
        # Original OpenAI/DeepSeek display
        openai_score = analysis.openai_stats.get('composite', {}).get('mean', 0.0)
        deepseek_score = analysis.deepseek_stats.get('composite', {}).get('mean', 0.0)
        
        scores = [("OpenAI", openai_score), ("DeepSeek", deepseek_score)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        for i, (name, score) in enumerate(scores, 1):
            # Create progress bar (20 chars max)
            bar_length = int((score / 10.0) * 20)
            bar = "█" * bar_length + " " * (20 - bar_length)
            
            model_line = f"║   {i}. {name:<8} {bar} {score:.2f}/10"
            line_padding = 60 - len(model_line) + 1
            print(f"{model_line}{' ' * line_padding}║")
    
    print("╠════════════════════════════════════════════════════════════╣")
    print("║ 📁 Results saved to:                                       ║")
    
    # List key files
    key_files = [
        "📊 detailed_results.json",
        "📈 statistical_analysis.json", 
        "📋 research_report.txt",
        "📊 visualizations/"
    ]
    
    for file in key_files:
        file_line = f"║   {file}"
        # Truncate if too long
        if len(file_line) > 58:
            file_line = file_line[:55] + "..."
        padding = 60 - len(file_line) - 1
        print(f"║{file_line}{' ' * padding}║")
    
    print("╚════════════════════════════════════════════════════════════╝")
    print()

def display_ultra_clean_summary(analysis, results, output_dir):
    """Display ultra-clean summary for demo/presentation mode"""
    if not analysis:
        ultra_clean_print("Error: No analysis results available")
        return
    
    ultra_clean_print("")
    ultra_clean_print("=" * 50)
    ultra_clean_print("FINAL RESULTS")
    ultra_clean_print("=" * 50)
    
    if hasattr(analysis, 'model_stats') and analysis.model_stats:
        # Multi-model display
        model_names = getattr(analysis, 'model_names', list(analysis.model_stats.keys()))
        
        # Sort models by average score for ranking
        model_scores = [(name, analysis.model_stats[name].get('composite', {}).get('mean', 0.0)) 
                       for name in model_names]
        model_scores.sort(key=lambda x: x[1], reverse=True)
        
        ultra_clean_print(f"Winner: {analysis.overall_winner}")
        ultra_clean_print("")
        
        # Simple ranking table
        ultra_clean_print("Model Rankings:")
        for i, (model_name, score) in enumerate(model_scores, 1):
            display_name = get_model_display_name(model_name)
            ultra_clean_print(f"{i}. {display_name:<8} {score:.2f}/10")
    else:
        # Original OpenAI/DeepSeek display
        openai_score = analysis.openai_stats.get('composite', {}).get('mean', 0.0)
        deepseek_score = analysis.deepseek_stats.get('composite', {}).get('mean', 0.0)
        
        scores = [("OpenAI", openai_score), ("DeepSeek", deepseek_score)]
        scores.sort(key=lambda x: x[1], reverse=True)
        
        ultra_clean_print(f"Winner: {analysis.overall_winner}")
        ultra_clean_print("")
        ultra_clean_print("Model Rankings:")
        for i, (name, score) in enumerate(scores, 1):
            ultra_clean_print(f"{i}. {name:<8} {score:.2f}/10")
    
    ultra_clean_print("")
    ultra_clean_print(f"📁 Full results: {output_dir}")

# =============================================================================
# PROGRESS TRACKING
# =============================================================================

class ScriptProgressTracker:
    """Enhanced progress tracker for the main script execution"""
    
    def __init__(self, total_scenarios):
        self.progress = None
        self.task = None
        self.total_scenarios = total_scenarios
        
    def start(self):
        """Start the progress bar"""
        self.progress = Progress(
            TextColumn("[bold blue]{task.description}"),
            BarColumn(),
            "[progress.percentage]{task.percentage:>3.0f}%",
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            console=console,
            transient=True
        )
        self.progress.start()
        self.task = self.progress.add_task("Starting research study...", total=100)
        
    def update(self, percentage, description):
        """Update progress"""
        if self.progress and self.task:
            self.progress.update(self.task, completed=percentage, description=description)
            
    def stop(self):
        """Stop the progress bar"""
        if self.progress:
            self.progress.stop()