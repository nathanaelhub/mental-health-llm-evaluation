#!/usr/bin/env python3
"""
Mental Health LLM Evaluation Research Runner (Refactored)
========================================================

Streamlined main entry point for the Mental Health LLM Evaluation Research.
This refactored version uses a modular architecture for better maintainability.

Usage:
    python run_research.py [--models MODEL1,MODEL2,...] [--quick] [--scenarios N] [--output DIR]

Options:
    --models        Comma-separated list of models: openai,claude,deepseek,gemma or 'all' (default: openai,deepseek)
    --all-models    Use all available models (openai,claude,deepseek,gemma)
    --quick         Run with 3 scenarios for fast testing
    --scenarios N   Run with N scenarios (default: all 10)
    --output DIR    Output directory (default: results/)
    --clean         Use clean, professional output format
    --ultra-clean   Ultra-minimal output - show only essential results
    --minimal       Minimal output mode - compact results display
    --debug         Enable debug output for troubleshooting
    --demo          Demo mode - ultra-clean output for presentations
"""

import os
import sys
import argparse
import traceback
from datetime import datetime
from typing import Optional, List

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Import refactored modules
from src.research.display import (
    print_header, print_demo_header, ultra_clean_print, 
    print_demo_results, display_ultra_clean_summary, display_minimal_summary,
    ScriptProgressTracker
)
from src.research.evaluation import (
    load_model_clients, create_model_client_instances, check_model_availability,
    run_evaluation_pipeline
)
from src.research.utils import (
    check_dependencies, show_startup_loading, StatusTracker
)

# Rich imports for console output
from rich.console import Console
from rich.panel import Panel
from rich.text import Text

# Optional imports
try:
    from src.utils.professional_formatter import ProfessionalFormatter, StudyConfiguration, print_clean_header, print_clean_results
    HAS_FORMATTER = True
except ImportError:
    print("Warning: Professional formatter not available")
    HAS_FORMATTER = False

console = Console()

# =============================================================================
# MODULE LOADING FUNCTIONS
# =============================================================================

def load_modules(clean_output=False, minimal=False):
    """Load all required analysis and visualization modules"""
    from src.research.display import conditional_print
    
    conditional_print("📦 Loading evaluation modules...", quiet=clean_output or minimal)
    
    modules = {}
    
    # Load core evaluation modules
    try:
        from src.evaluation.mental_health_evaluator import MentalHealthEvaluator
        from src.evaluation.multi_model_evaluator import MultiModelEvaluator
        modules['evaluator'] = MentalHealthEvaluator
        modules['multi_evaluator'] = MultiModelEvaluator
        conditional_print("   ✅ Mental health evaluator loaded", quiet=clean_output or minimal)
        conditional_print("   ✅ Multi-model evaluator loaded", quiet=clean_output or minimal)
    except ImportError as e:
        conditional_print(f"   ❌ Failed to load evaluators: {e}", quiet=clean_output or minimal)
        return None
    
    # Load analysis modules
    try:
        from src.analysis.statistical_analysis import analyze_results, generate_summary_report, identify_model_strengths
        modules['analyze_results'] = analyze_results
        modules['generate_report'] = generate_summary_report
        modules['identify_strengths'] = identify_model_strengths
        conditional_print("   ✅ Statistical analysis loaded", quiet=clean_output or minimal)
    except ImportError as e:
        conditional_print(f"   ❌ Failed to load analysis: {e}", quiet=clean_output or minimal)
        return None
    
    # Load visualization modules
    try:
        from src.analysis.visualization import create_all_visualizations, HAS_MATPLOTLIB
        modules['create_visualizations'] = create_all_visualizations
        modules['create_slides'] = None
        modules['has_matplotlib'] = HAS_MATPLOTLIB
        conditional_print(f"   ✅ Visualization loaded ({'with matplotlib' if HAS_MATPLOTLIB else 'fallback mode'})", quiet=clean_output or minimal)
    except ImportError as e:
        # Fallback for missing matplotlib
        modules['create_visualizations'] = lambda *args: ([], None, "Matplotlib not available")
        modules['create_slides'] = lambda *args: None
        modules['has_matplotlib'] = False
        conditional_print(f"   ⚠️ Visualization loaded (matplotlib not available): {e}", quiet=clean_output or minimal)
    
    return modules

def run_statistical_analysis(results, analyze_func, report_func, strengths_func, clean_output: bool = False) -> tuple:
    """Run statistical analysis with error handling"""
    try:
        if not clean_output:
            console.print("📊 [bold cyan]Running statistical analysis...[/bold cyan]")
        
        # Run the analysis
        analysis = analyze_func(results)
        report = report_func(analysis)
        strengths = strengths_func(analysis)
        
        if not clean_output:
            console.print("✅ [green]Statistical analysis completed[/green]")
        
        return analysis, report, strengths, None
        
    except Exception as e:
        error_message = f"Statistical analysis failed: {e}"
        return None, None, None, error_message

def create_visualizations(results, analysis, viz_func, slides_func, has_matplotlib: bool, results_dir: str) -> tuple:
    """Create visualizations with error handling"""
    try:
        if not has_matplotlib:
            return [], None, "Matplotlib not available for visualizations"
        
        console.print("📈 [bold cyan]Creating visualizations...[/bold cyan]")
        
        # Create charts
        chart_files = viz_func(results, analysis, results_dir)
        
        # Create presentation slides if requested
        slides = None
        if slides_func:
            try:
                slides = slides_func(results, analysis, results_dir)
            except Exception as e:
                console.print(f"⚠️ [yellow]Slides creation failed: {e}[/yellow]")
        
        console.print(f"✅ [green]Created {len(chart_files)} visualization files[/green]")
        
        return chart_files, slides, None
        
    except Exception as e:
        error_message = f"Visualization creation failed: {e}"
        return [], None, error_message

def save_results(results, analysis, report, strengths, results_dir: str, clean_output: bool = False, minimal: bool = False):
    """Save all results to files"""
    import json
    import os
    from src.research.display import conditional_print
    from src.research.utils import ensure_directory_exists
    
    conditional_print("💾 Saving results...", quiet=clean_output or minimal)
    
    # Ensure results directory exists
    ensure_directory_exists(results_dir)
    ensure_directory_exists(os.path.join(results_dir, "evaluations"))
    ensure_directory_exists(os.path.join(results_dir, "statistics"))
    ensure_directory_exists(os.path.join(results_dir, "reports"))
    
    file_count = 0
    
    # Save detailed results
    if results:
        results_data = []
        for result in results:
            if hasattr(result, '__dict__'):
                results_data.append(result.__dict__)
            else:
                results_data.append(result)
        
        with open(os.path.join(results_dir, "evaluations", "detailed_results.json"), 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        file_count += 1
        
        # Save analysis summary
        if analysis:
            analysis_data = {
                'timestamp': datetime.now().isoformat(),
                'overall_winner': analysis.overall_winner,
                'confidence_level': analysis.confidence_level,
                'key_findings': analysis.key_findings,
                'practical_significance': analysis.practical_significance,
                'clinical_significance': analysis.clinical_significance,
                'safety_analysis': analysis.safety_analysis,
                'cost_analysis': analysis.cost_analysis
            }
            
            with open(os.path.join(results_dir, "statistics", "statistical_analysis.json"), 'w') as f:
                json.dump(analysis_data, f, indent=2, default=str)
            file_count += 1
        
        # Save research report
        if report:
            with open(os.path.join(results_dir, "reports", "research_report.txt"), 'w') as f:
                f.write(report)
            file_count += 1
        
        # Save model strengths
        if strengths:
            with open(os.path.join(results_dir, "reports", "model_strengths.json"), 'w') as f:
                json.dump(strengths, f, indent=2)
            file_count += 1
    
    conditional_print(f"💾 Saved {file_count} result files to {results_dir}/", quiet=clean_output or minimal)

def display_research_summary(analysis, strengths):
    """Display comprehensive research summary with rich formatting"""
    if not analysis:
        console.print("❌ [bold red]No analysis results to display[/bold red]")
        return
    
    # Main results panel
    console.print()
    console.print(Panel(
        Text.assemble(
            ("🏆 Overall Winner: ", "bold cyan"),
            (f"{analysis.overall_winner}", "bold green"),
            (f" ({analysis.confidence_level.title()} Confidence)", "yellow")
        ),
        title="[bold green]Research Results[/bold green]",
        border_style="green"
    ))
    
    # Key findings
    if hasattr(analysis, 'key_findings') and analysis.key_findings:
        console.print("\n📋 [bold cyan]Key Findings:[/bold cyan]")
        for finding in analysis.key_findings:
            console.print(f"   • {finding}")
    
    # Model strengths
    if strengths:
        console.print("\n💪 [bold cyan]Model Strengths:[/bold cyan]")
        for model, strength_list in strengths.items():
            model_display = model.title()
            if model == 'openai':
                model_display = 'OpenAI'
            console.print(f"   🤖 [bold]{model_display}[/bold]:")
            for strength in strength_list:
                console.print(f"      • {strength}")

# =============================================================================
# MAIN FUNCTION
# =============================================================================

def main():
    """Main research pipeline with modular architecture"""
    
    # Show startup loading
    show_startup_loading()
    
    # Initialize status tracker
    status_tracker = StatusTracker()
    progress_tracker = None
    
    try:
        # Parse command line arguments
        parser = argparse.ArgumentParser(
            description="Mental Health LLM Evaluation Research (Refactored)",
            epilog="""
Examples:
  %(prog)s                                    # Default: openai,deepseek
  %(prog)s --models openai,claude            # Compare specific models
  %(prog)s --all-models                      # Use all 4 models
  %(prog)s --all-models --minimal            # All models, compact output
  %(prog)s --all-models --ultra-clean        # All models, ultra-minimal output
            """,
            formatter_class=argparse.RawDescriptionHelpFormatter
        )
        
        parser.add_argument("--models", help="Comma-separated list of models: openai,claude,deepseek,gemma (default: openai,deepseek)")
        parser.add_argument("--all-models", action="store_true", help="Use all available models")
        parser.add_argument("--quick", action="store_true", help="Run quick evaluation (3 scenarios)")
        parser.add_argument("--scenarios", type=int, help="Number of scenarios to run (default: all)")
        parser.add_argument("--output", default="results", help="Output directory (default: results/)")
        parser.add_argument("--clean", action="store_true", help="Use clean, professional output format")
        parser.add_argument("--ultra-clean", action="store_true", help="Ultra-minimal output")
        parser.add_argument("--minimal", action="store_true", help="Minimal output mode")
        parser.add_argument("--debug", action="store_true", help="Enable debug output")
        parser.add_argument("--demo", action="store_true", help="Demo mode - presentation output")
        
        args = parser.parse_args()
        
        # Handle output modes
        if args.ultra_clean or args.minimal:
            args.clean = True
        
        if args.demo:
            args.ultra_clean = True
            args.clean = True
            args.debug = False
        
        # Parse model selection
        if args.all_models:
            selected_models = ['openai', 'claude', 'deepseek', 'gemma']
        elif args.models:
            if args.models.lower() == 'all':
                selected_models = ['openai', 'claude', 'deepseek', 'gemma']
            else:
                selected_models = [model.strip() for model in args.models.split(',')]
        else:
            selected_models = ['openai', 'deepseek']
        
        # Determine number of scenarios
        if args.quick:
            num_scenarios = 3
        elif args.scenarios:
            num_scenarios = args.scenarios
        else:
            num_scenarios = 10
        
        # Initialize progress tracker
        if not args.ultra_clean:
            progress_tracker = ScriptProgressTracker(num_scenarios)
            progress_tracker.start()
            progress_tracker.update(10, "📦 Loading modules...")
        
        # Display startup header
        if args.demo:
            print_demo_header(len(selected_models), num_scenarios)
        elif args.ultra_clean:
            ultra_clean_print("🧠 Mental Health LLM Evaluation Study")
            ultra_clean_print(f"📊 Evaluating {len(selected_models)} models on {num_scenarios} scenarios...")
        elif args.minimal:
            print("🧠 Mental Health LLM Evaluation Study")
            print(f"📊 Evaluating {len(selected_models)} models on {num_scenarios} scenarios...")
        else:
            print_header(selected_models)
        
        # Check dependencies
        if not check_dependencies():
            console.print("❌ [bold red]Cannot proceed without required dependencies[/bold red]")
            sys.exit(1)
        
        if not args.clean and not args.ultra_clean and not args.demo:
            console.print("✅ [green]Dependencies check passed[/green]")
        
        # Load modules
        modules = load_modules(args.clean or args.ultra_clean or args.demo, args.minimal)
        if not modules:
            console.print("❌ [bold red]Failed to load required modules[/bold red]")
            sys.exit(1)
        
        # Load model clients and check availability
        model_clients = load_model_clients(args.clean or args.ultra_clean or args.demo, args.minimal)
        available_models = check_model_availability(selected_models, model_clients, args.clean or args.ultra_clean or args.demo, args.minimal)
        
        if len(available_models) < 2:
            console.print("❌ [bold red]Need at least 2 models for comparison[/bold red]")
            sys.exit(1)
        
        # Create client instances
        client_instances = create_model_client_instances(
            available_models, model_clients, 
            args.clean or args.ultra_clean or args.minimal or args.demo, 
            args.debug and not args.demo, args.minimal
        )
        
        # Set evaluation parameters
        limit = None
        if args.quick:
            limit = 3
        elif args.scenarios:
            limit = args.scenarios
        
        # 1. Run evaluation
        if progress_tracker:
            progress_tracker.update(30, "🧠 Running evaluation...")
        
        use_multi_model = len(available_models) > 2 and not args.demo
        evaluator_class = modules['multi_evaluator'] if use_multi_model else modules['evaluator']
        
        results, status_tracker, eval_error = run_evaluation_pipeline(
            evaluator_class, limit, available_models, use_multi_model, 
            args.clean, progress_tracker, client_instances, 
            args.ultra_clean, args.minimal, args.debug, args.demo, status_tracker
        )
        
        if eval_error:
            console.print(f"❌ [bold red]Evaluation failed: {eval_error}[/bold red]")
            sys.exit(1)
        
        # 2. Statistical analysis
        if progress_tracker:
            progress_tracker.update(70, "📊 Statistical analysis...")
        
        analysis, report, strengths, analysis_error = run_statistical_analysis(
            results, modules['analyze_results'], modules['generate_report'], 
            modules['identify_strengths'], args.clean or args.ultra_clean
        )
        
        if analysis_error:
            console.print(f"❌ [bold red]Analysis failed: {analysis_error}[/bold red]")
            sys.exit(1)
        
        # 3. Create visualizations
        if progress_tracker:
            progress_tracker.update(85, "📈 Creating visualizations...")
        
        chart_files, _, viz_error = create_visualizations(
            results, analysis, modules['create_visualizations'], 
            modules['create_slides'], modules['has_matplotlib'], args.output
        )
        
        if viz_error:
            console.print(f"⚠️ [yellow]Visualization warning: {viz_error}[/yellow]")
        
        # 4. Save results
        if progress_tracker:
            progress_tracker.update(95, "💾 Saving results...")
        
        save_results(results, analysis, report, strengths, args.output, args.clean, args.minimal)
        
        # 5. Display summary
        if args.demo:
            print_demo_results(analysis, args.output, chart_files)
        elif args.ultra_clean:
            display_ultra_clean_summary(analysis, results, args.output)
        elif args.minimal:
            display_minimal_summary(analysis, results, args.output)
        else:
            display_research_summary(analysis, strengths)
            
            # Final completion panel
            completion_panel = Panel(
                Text.assemble(
                    ("✅ Research study complete!\n", "bold green"),
                    (f"📁 Results saved to: {os.path.abspath(args.output)}/\n", "blue"),
                    (f"📊 Generated {len(chart_files)} visualizations\n" if chart_files else "", "cyan"),
                    ("\n🎓 Ready for academic use", "bold yellow")
                ),
                title="[bold green]Study Complete[/bold green]",
                border_style="green"
            )
            console.print(completion_panel)
        
        # Display final metrics if available
        if status_tracker and not args.ultra_clean and not args.minimal:
            console.print("\n📊 [bold cyan]Final Evaluation Metrics:[/bold cyan]")
            console.print(f"   API Calls Made: {status_tracker.api_calls}")
            console.print(f"   Success Rate: {status_tracker.get_success_rate():.1f}%")
            console.print(f"   Total Cost: ${status_tracker.total_cost:.4f}")
            
            # Show detailed metrics table if in verbose mode
            if not args.clean:
                console.print()
                metrics_table = status_tracker.create_metrics_table()
                console.print(metrics_table)
        
        # Final progress update
        if progress_tracker:
            progress_tracker.update(100, "✅ Complete!")
            progress_tracker.stop()
        
    except KeyboardInterrupt:
        if 'args' in locals() and args.clean:
            print("\nStudy interrupted by user")
        else:
            console.print("\n❌ [bold red]Research interrupted by user[/bold red]")
        sys.exit(1)
        
    except Exception as e:
        if 'args' in locals() and args.clean:
            print(f"\nError: {e}")
        else:
            console.print(f"\n❌ [bold red]Unexpected error: {e}[/bold red]")
            console.print("[dim]Use --debug for detailed error information[/dim]")
        
        if 'args' in locals() and args.debug:
            traceback.print_exc()
        
        sys.exit(1)
        
    finally:
        # Cleanup
        if progress_tracker:
            progress_tracker.stop()

# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n❌ Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        sys.exit(1)