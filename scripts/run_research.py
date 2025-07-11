#!/usr/bin/env python3
"""
Mental Health LLM Evaluation Research Runner
===========================================

Main entry point that orchestrates the complete evaluation study comparing
OpenAI GPT-4 vs DeepSeek for therapeutic conversations.

This script:
1. Loads scenarios and initializes models
2. Runs comprehensive evaluation across all mental health scenarios
3. Performs rigorous statistical analysis
4. Generates publication-quality visualizations
5. Creates detailed research report
6. Displays key findings and recommendations

Usage:
    python run_research.py [--quick] [--scenarios N] [--output DIR]

Options:
    --quick         Run with 3 scenarios for fast testing
    --scenarios N   Run with N scenarios (default: all 10)
    --output DIR    Output directory (default: output/)
"""

import os
import sys
import time
import argparse
from datetime import datetime
from typing import Optional, Dict, Any

# Add parent directory to path to import from src
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Progress bar implementation
def create_progress_bar(current: int, total: int, width: int = 50) -> str:
    """Create a visual progress bar"""
    if total == 0:
        return "[" + "=" * width + "]"
    
    progress = current / total
    filled = int(width * progress)
    bar = "=" * filled + "-" * (width - filled)
    percentage = progress * 100
    return f"[{bar}] {percentage:.1f}% ({current}/{total})"


def print_header():
    """Print the research study header"""
    print("🧠 Mental Health LLM Evaluation Research Study")
    print("=" * 60)
    print("📋 Comparing OpenAI GPT-4 vs DeepSeek for Therapeutic Conversations")
    print("🎯 Academic Capstone Project - Statistical Analysis & Recommendations")
    print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 60)
    print()


def check_dependencies():
    """Check if required modules are available"""
    missing_modules = []
    optional_modules = []
    
    # Check required modules
    try:
        import yaml
    except ImportError:
        missing_modules.append("pyyaml")
    
    # Check optional modules for full functionality
    try:
        import matplotlib
    except ImportError:
        optional_modules.append("matplotlib")
    
    try:
        import numpy
    except ImportError:
        optional_modules.append("numpy")
    
    try:
        import scipy
    except ImportError:
        optional_modules.append("scipy")
    
    if missing_modules:
        print("❌ Missing required dependencies:")
        for module in missing_modules:
            print(f"   • {module}")
        print("\nInstall with: pip install " + " ".join(missing_modules))
        return False
    
    if optional_modules:
        print("⚠️  Optional dependencies missing (reduced functionality):")
        for module in optional_modules:
            print(f"   • {module}")
        print("   Install with: pip install " + " ".join(optional_modules))
        print()
    
    return True


def load_modules():
    """Import and return all required modules"""
    modules = {}
    
    try:
        # Always try to import main modules
        print("📦 Loading evaluation modules...")
        
        # Import with error handling
        try:
            from src.evaluation.mental_health_evaluator import MentalHealthEvaluator
            modules['evaluator'] = MentalHealthEvaluator
            print("   ✅ Mental health evaluator loaded")
        except ImportError as e:
            print(f"   ❌ Failed to load evaluator: {e}")
            return None
        
        try:
            from src.analysis.statistical_analysis import analyze_results, generate_summary_report, identify_model_strengths
            modules['analyze_results'] = analyze_results
            modules['generate_report'] = generate_summary_report
            modules['identify_strengths'] = identify_model_strengths
            print("   ✅ Statistical analysis loaded")
        except ImportError as e:
            print(f"   ❌ Failed to load statistical analysis: {e}")
            return None
        
        try:
            from src.analysis.visualization import create_all_visualizations, create_presentation_slides, HAS_MATPLOTLIB
            modules['create_visualizations'] = create_all_visualizations
            modules['create_slides'] = create_presentation_slides
            modules['has_matplotlib'] = HAS_MATPLOTLIB
            print(f"   ✅ Visualization loaded ({'with matplotlib' if HAS_MATPLOTLIB else 'fallback mode'})")
        except ImportError as e:
            print(f"   ⚠️  Visualization unavailable: {e}")
            modules['create_visualizations'] = None
            modules['create_slides'] = None
            modules['has_matplotlib'] = False
        
        return modules
        
    except Exception as e:
        print(f"❌ Error loading modules: {e}")
        return None


def run_evaluation_pipeline(evaluator_class, limit: Optional[int] = None) -> tuple:
    """
    Run the complete evaluation pipeline
    
    Returns:
        (results, analysis, error_message)
    """
    try:
        # Initialize evaluator
        print("🔧 Initializing mental health evaluator...")
        evaluator = evaluator_class()
        total_scenarios = len(evaluator.scenarios)
        
        if limit:
            total_scenarios = min(limit, total_scenarios)
            print(f"📊 Running evaluation on {total_scenarios} scenarios (limited)")
        else:
            print(f"📊 Running evaluation on all {total_scenarios} scenarios")
        
        print()
        
        # Run evaluation with progress tracking
        print("🤖 Generating therapeutic conversations...")
        print("   This may take several minutes depending on model response times...")
        print()
        
        start_time = time.time()
        results = evaluator.run_evaluation(limit=limit)
        end_time = time.time()
        
        evaluation_time = end_time - start_time
        print(f"\n✅ Evaluation completed in {evaluation_time:.1f} seconds")
        print(f"📋 Generated {len(results)} conversation pairs")
        
        return results, None, None
        
    except Exception as e:
        error_msg = f"Evaluation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return None, None, error_msg


def run_statistical_analysis(results, analyze_func, report_func, strengths_func) -> tuple:
    """
    Run statistical analysis on evaluation results
    
    Returns:
        (analysis, report, strengths, error_message)
    """
    try:
        print("\n📊 Performing comprehensive statistical analysis...")
        print("   • Computing descriptive statistics")
        print("   • Running normality tests")
        print("   • Performing significance testing")
        print("   • Calculating effect sizes")
        print("   • Analyzing safety metrics")
        print("   • Computing cost-benefit analysis")
        
        # Convert results to expected format
        if hasattr(results[0], '__dict__'):
            # Convert ScenarioResult objects to dict format
            results_data = {'scenarios': []}
            for result in results:
                results_data['scenarios'].append({
                    'category': result.category,
                    'openai_evaluation': result.openai_evaluation,
                    'deepseek_evaluation': result.deepseek_evaluation
                })
        else:
            # Already in dict format
            results_data = {'scenarios': results}
        
        analysis = analyze_func(results_data)
        
        print("   • Generating summary report")
        report = report_func(analysis)
        
        print("   • Identifying model strengths")
        strengths = strengths_func(analysis)
        
        print("✅ Statistical analysis complete")
        
        return analysis, report, strengths, None
        
    except Exception as e:
        error_msg = f"Statistical analysis failed: {str(e)}"
        print(f"❌ {error_msg}")
        return None, None, None, error_msg


def create_visualizations(results, analysis, viz_func, slides_func, has_matplotlib: bool, output_dir: str) -> tuple:
    """
    Generate all visualizations
    
    Returns:
        (chart_files, slide_files, error_message)
    """
    try:
        if not has_matplotlib:
            print("\n📊 Skipping visualizations (matplotlib not available)")
            print("   Install with: pip install matplotlib seaborn numpy")
            return [], [], None
        
        print("\n🎨 Creating publication-quality visualizations...")
        
        # Create visualizations directory
        viz_dir = os.path.join(output_dir, "visualizations")
        presentation_dir = os.path.join(output_dir, "presentation")
        
        print("   • Overall comparison bar chart")
        print("   • Category performance radar chart")
        print("   • Cost-effectiveness scatter plot")
        print("   • Safety metrics analysis")
        print("   • Statistical summary table")
        
        chart_files = viz_func(results, analysis, viz_dir)
        
        print("   • Generating presentation slides")
        slide_files = slides_func(results, analysis, presentation_dir)
        
        print(f"✅ Generated {len(chart_files)} charts and {len(slide_files)} slides")
        
        return chart_files, slide_files, None
        
    except Exception as e:
        error_msg = f"Visualization generation failed: {str(e)}"
        print(f"❌ {error_msg}")
        return [], [], error_msg


def save_results(results, analysis, report, strengths, output_dir: str):
    """Save all results to files"""
    try:
        print(f"\n💾 Saving results to {output_dir}/...")
        
        # Create output directory
        os.makedirs(output_dir, exist_ok=True)
        
        # Save detailed results
        import json
        
        # Convert results to serializable format
        if hasattr(results[0], '__dict__'):
            serializable_results = []
            for result in results:
                result_dict = {
                    'scenario_id': result.scenario_id,
                    'scenario_name': result.scenario_name,
                    'category': result.category,
                    'severity': result.severity,
                    'prompt': result.prompt,
                    'openai_response': result.openai_response,
                    'deepseek_response': result.deepseek_response,
                    'openai_evaluation': result.openai_evaluation.to_dict() if hasattr(result.openai_evaluation, 'to_dict') else result.openai_evaluation,
                    'deepseek_evaluation': result.deepseek_evaluation.to_dict() if hasattr(result.deepseek_evaluation, 'to_dict') else result.deepseek_evaluation,
                    'winner': result.winner,
                    'timestamp': result.timestamp
                }
                serializable_results.append(result_dict)
            
            results_data = {
                'metadata': {
                    'evaluation_date': datetime.now().isoformat(),
                    'total_scenarios': len(serializable_results),
                    'models_compared': ['OpenAI GPT-4', 'DeepSeek']
                },
                'scenarios': serializable_results
            }
        else:
            results_data = {'scenarios': results}
        
        # Save detailed results as JSON
        with open(os.path.join(output_dir, "detailed_results.json"), 'w') as f:
            json.dump(results_data, f, indent=2, default=str)
        print("   ✅ Detailed results saved")
        
        # Save analysis summary
        if analysis:
            with open(os.path.join(output_dir, "statistical_analysis.json"), 'w') as f:
                json.dump({
                    'overall_winner': analysis.overall_winner,
                    'confidence_level': analysis.confidence_level,
                    'key_findings': analysis.key_findings,
                    'practical_significance': analysis.practical_significance,
                    'clinical_significance': analysis.clinical_significance,
                    'safety_analysis': analysis.safety_analysis.__dict__ if hasattr(analysis.safety_analysis, '__dict__') else analysis.safety_analysis,
                    'cost_analysis': analysis.cost_analysis
                }, f, indent=2, default=str)
            print("   ✅ Statistical analysis saved")
        
        # Save research report
        if report:
            with open(os.path.join(output_dir, "research_report.txt"), 'w') as f:
                f.write(report)
            print("   ✅ Research report saved")
        
        # Save model strengths
        if strengths:
            with open(os.path.join(output_dir, "model_strengths.json"), 'w') as f:
                json.dump(strengths, f, indent=2)
            print("   ✅ Model strengths saved")
        
        print(f"✅ All results saved to {output_dir}/")
        
    except Exception as e:
        print(f"❌ Error saving results: {e}")


def display_research_summary(analysis, strengths):
    """Display key research findings in the terminal"""
    print("\n" + "=" * 60)
    print("📊 RESEARCH FINDINGS SUMMARY")
    print("=" * 60)
    
    if not analysis:
        print("❌ No analysis results available")
        return
    
    # Overall winner
    print(f"\n🏆 OVERALL WINNER: {analysis.overall_winner}")
    print(f"   Confidence Level: {analysis.confidence_level.upper()}")
    
    # Key metrics comparison
    print(f"\n📈 KEY METRICS COMPARISON:")
    openai_composite = analysis.openai_stats['composite'].mean
    deepseek_composite = analysis.deepseek_stats['composite'].mean
    difference = openai_composite - deepseek_composite
    
    print(f"   OpenAI GPT-4:  {openai_composite:.2f}/10 (±{analysis.openai_stats['composite'].std_dev:.2f})")
    print(f"   DeepSeek:      {deepseek_composite:.2f}/10 (±{analysis.deepseek_stats['composite'].std_dev:.2f})")
    print(f"   Difference:    {difference:+.2f} points")
    
    # Statistical significance
    composite_test = analysis.comparison_tests['composite']
    print(f"\n🧮 STATISTICAL SIGNIFICANCE:")
    print(f"   p-value: {composite_test.p_value:.4f}")
    print(f"   Effect size (Cohen's d): {composite_test.effect_size:.2f} ({composite_test.effect_interpretation})")
    print(f"   Statistically significant: {'YES' if composite_test.is_significant else 'NO'}")
    
    # Practical significance
    print(f"\n📊 PRACTICAL SIGNIFICANCE:")
    if analysis.clinical_significance['composite']:
        print("   🏥 CLINICALLY SIGNIFICANT (>1.0 points difference)")
    elif analysis.practical_significance['composite']:
        print("   📊 PRACTICALLY SIGNIFICANT (>0.5 points difference)")
    else:
        print("   ⚪ MARGINAL DIFFERENCE (<0.5 points)")
    
    # Safety analysis
    safety = analysis.safety_analysis
    print(f"\n🛡️  SAFETY ANALYSIS (PRIORITY):")
    print(f"   Safety Violations:    OpenAI: {safety.openai_safety_violations}, DeepSeek: {safety.deepseek_safety_violations}")
    
    if safety.crisis_scenarios_total > 0:
        openai_crisis_rate = safety.openai_crisis_appropriate_responses / safety.crisis_scenarios_total * 100
        deepseek_crisis_rate = safety.deepseek_crisis_appropriate_responses / safety.crisis_scenarios_total * 100
        print(f"   Crisis Handling:      OpenAI: {openai_crisis_rate:.0f}%, DeepSeek: {deepseek_crisis_rate:.0f}%")
    
    print(f"   Professional Referrals: OpenAI: {safety.openai_professional_referral_rate:.1%}, DeepSeek: {safety.deepseek_professional_referral_rate:.1%}")
    
    # Cost analysis
    cost = analysis.cost_analysis
    print(f"\n💰 COST-BENEFIT ANALYSIS:")
    print(f"   OpenAI Cost per Conversation: ${cost['openai_avg_cost']:.4f}")
    print(f"   DeepSeek Cost per Conversation: ${cost['deepseek_avg_cost']:.4f}")
    if cost.get('cost_per_point_improvement', 0) > 0:
        print(f"   Cost per Quality Point Improvement: ${cost['cost_per_point_improvement']:.4f}")
    
    # Deployment recommendation
    print(f"\n🚀 DEPLOYMENT RECOMMENDATION:")
    if analysis.overall_winner == "OpenAI GPT-4":
        if analysis.clinical_significance['composite']:
            print("   🟢 STRONG RECOMMENDATION for OpenAI GPT-4")
            print("      → Clinically significant improvement in therapeutic quality")
        elif analysis.practical_significance['composite']:
            print("   🟡 MODERATE RECOMMENDATION for OpenAI GPT-4")
            print("      → Practically significant improvement, consider cost-benefit")
        else:
            print("   🟠 WEAK RECOMMENDATION for OpenAI GPT-4")
            print("      → Marginal improvement, cost sensitivity important")
    elif analysis.overall_winner == "DeepSeek":
        print("   🟢 RECOMMENDATION for DeepSeek")
        print("      → Superior performance with zero operational cost")
    else:
        print("   ⚪ NO CLEAR RECOMMENDATION")
        print("      → Models perform similarly, choose based on cost preference")
    
    # Safety considerations
    if safety.openai_safety_violations < safety.deepseek_safety_violations:
        print("   🛡️  OpenAI preferred for safety-critical applications")
    
    # Crisis handling
    if safety.crisis_scenarios_total > 0:
        openai_crisis_rate = safety.openai_crisis_appropriate_responses / safety.crisis_scenarios_total
        deepseek_crisis_rate = safety.deepseek_crisis_appropriate_responses / safety.crisis_scenarios_total
        
        if openai_crisis_rate > deepseek_crisis_rate:
            print("   🚨 OpenAI REQUIRED for crisis intervention scenarios")
    
    # Key findings
    print(f"\n🔍 KEY RESEARCH FINDINGS:")
    for finding in analysis.key_findings:
        print(f"   • {finding}")
    
    # Model strengths
    if strengths:
        print(f"\n🎯 MODEL STRENGTHS:")
        for model, strength_list in strengths.items():
            if strength_list:
                print(f"   {model}:")
                for strength in strength_list[:3]:  # Show top 3 strengths
                    print(f"      • {strength}")
    
    print("=" * 60)


def main():
    """Main research pipeline"""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Mental Health LLM Evaluation Research")
    parser.add_argument("--quick", action="store_true", help="Run quick evaluation (3 scenarios)")
    parser.add_argument("--scenarios", type=int, help="Number of scenarios to run (default: all)")
    parser.add_argument("--output", default="output", help="Output directory (default: output/)")
    
    args = parser.parse_args()
    
    # Print header
    print_header()
    
    # Check dependencies
    if not check_dependencies():
        print("❌ Cannot proceed without required dependencies")
        sys.exit(1)
    
    print("✅ Dependencies check passed")
    print()
    
    # Load modules
    modules = load_modules()
    if not modules:
        print("❌ Failed to load required modules")
        sys.exit(1)
    
    print()
    
    # Determine evaluation parameters
    limit = None
    if args.quick:
        limit = 3
        print("🚀 Quick mode: Running with 3 scenarios")
    elif args.scenarios:
        limit = args.scenarios
        print(f"🚀 Custom mode: Running with {limit} scenarios")
    else:
        print("🚀 Full mode: Running with all scenarios")
    
    print()
    
    # 1. Run evaluation
    results, _, eval_error = run_evaluation_pipeline(modules['evaluator'], limit)
    if eval_error:
        print(f"❌ Evaluation failed: {eval_error}")
        sys.exit(1)
    
    # 2. Statistical analysis
    analysis, report, strengths, analysis_error = run_statistical_analysis(
        results, modules['analyze_results'], modules['generate_report'], modules['identify_strengths']
    )
    if analysis_error:
        print(f"❌ Analysis failed: {analysis_error}")
        sys.exit(1)
    
    # 3. Generate visualizations
    chart_files, slide_files, viz_error = create_visualizations(
        results, analysis, 
        modules['create_visualizations'], modules['create_slides'], 
        modules['has_matplotlib'], args.output
    )
    if viz_error:
        print(f"⚠️  Visualization warning: {viz_error}")
    
    # 4. Save all results
    save_results(results, analysis, report, strengths, args.output)
    
    # 5. Display summary
    display_research_summary(analysis, strengths)
    
    # Final success message
    print(f"\n✅ Research study complete!")
    print(f"📁 Results saved to: {os.path.abspath(args.output)}/")
    
    if chart_files:
        print(f"📊 Generated {len(chart_files)} visualizations")
    if slide_files:
        print(f"📝 Generated {len(slide_files)} presentation slides")
    
    print("\n🎓 Ready for:")
    print("   • Academic paper submission")
    print("   • Capstone presentation")
    print("   • Healthcare deployment decisions")
    print("   • Further research expansion")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n\n❌ Research interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\n\n❌ Unexpected error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)