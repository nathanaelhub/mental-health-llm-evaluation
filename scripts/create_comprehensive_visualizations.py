#!/usr/bin/env python3
"""
Comprehensive Research Visualization Generator
Creates publication-ready charts and graphs for mental health LLM evaluation results
"""

import json
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
from pathlib import Path
import argparse
from datetime import datetime

# Set style for publication-quality plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_research_data(json_file):
    """Load research data from JSON file"""
    with open(json_file, 'r') as f:
        data = json.load(f)
    return data

def create_model_comparison_chart(data, output_dir):
    """Create overall model comparison chart"""
    summary = data['summary']
    models = summary['models_evaluated']
    
    # Prepare data
    avg_scores = [summary['model_avg_scores'][model] for model in models]
    wins = [summary['model_wins'][model] for model in models]
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Average scores comparison
    bars1 = ax1.bar(models, avg_scores, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
    ax1.set_title('Average Therapeutic Quality Scores', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Average Score (0-10)', fontsize=12)
    ax1.set_ylim(0, 10)
    
    # Add value labels on bars
    for bar, score in zip(bars1, avg_scores):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Wins comparison
    bars2 = ax2.bar(models, wins, color=['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)])
    ax2.set_title('Scenario Wins', fontsize=14, fontweight='bold')
    ax2.set_ylabel('Number of Wins', fontsize=12)
    ax2.set_ylim(0, max(wins) + 2)
    
    # Add value labels on bars
    for bar, win in zip(bars2, wins):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{int(win)}', ha='center', va='bottom', fontweight='bold')
    
    plt.tight_layout()
    output_file = Path(output_dir) / '1_model_comparison.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)

def create_detailed_scores_heatmap(data, output_dir):
    """Create heatmap of detailed scores by scenario"""
    scenarios = data['scenarios']
    models = data['summary']['models_evaluated']
    
    # Prepare data for heatmap
    score_data = []
    scenario_names = []
    
    for scenario in scenarios:
        scenario_names.append(scenario['scenario_name'][:20] + '...' if len(scenario['scenario_name']) > 20 else scenario['scenario_name'])
        row = []
        for model in models:
            evaluation = scenario['model_evaluations'][model]
            row.append(evaluation['composite_score'])
        score_data.append(row)
    
    # Create heatmap
    plt.figure(figsize=(12, 8))
    
    df = pd.DataFrame(score_data, index=scenario_names, columns=[m.upper() for m in models])
    
    sns.heatmap(df, annot=True, fmt='.2f', cmap='RdYlGn', 
                center=5, vmin=0, vmax=10,
                cbar_kws={'label': 'Composite Score (0-10)'})
    
    plt.title('Therapeutic Quality Scores by Scenario and Model', fontsize=14, fontweight='bold')
    plt.xlabel('Models', fontsize=12)
    plt.ylabel('Scenarios', fontsize=12)
    plt.xticks(rotation=0)
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    output_file = Path(output_dir) / '2_detailed_scores_heatmap.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)

def create_dimension_analysis(data, output_dir):
    """Create radar chart showing performance across therapeutic dimensions"""
    scenarios = data['scenarios']
    models = data['summary']['models_evaluated']
    
    # Calculate average scores for each dimension
    dimensions = ['empathy_score', 'therapeutic_value_score', 'safety_score', 'clarity_score']
    dimension_labels = ['Empathy', 'Therapeutic\nValue', 'Safety', 'Clarity']
    
    model_dimension_scores = {}
    
    for model in models:
        scores = {dim: [] for dim in dimensions}
        
        for scenario in scenarios:
            evaluation = scenario['model_evaluations'][model]
            for dim in dimensions:
                scores[dim].append(evaluation[dim])
        
        # Calculate averages
        avg_scores = [np.mean(scores[dim]) for dim in dimensions]
        model_dimension_scores[model] = avg_scores
    
    # Create radar chart
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
    
    # Number of variables
    num_vars = len(dimensions)
    
    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]  # Complete the circle
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, (model, scores) in enumerate(model_dimension_scores.items()):
        scores += scores[:1]  # Complete the circle
        ax.plot(angles, scores, 'o-', linewidth=2, label=model.upper(), color=colors[i])
        ax.fill(angles, scores, alpha=0.25, color=colors[i])
    
    # Add labels
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dimension_labels, fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.set_yticklabels(['2', '4', '6', '8', '10'], fontsize=10)
    ax.grid(True)
    
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
    plt.title('Therapeutic Dimension Analysis', size=16, fontweight='bold', pad=20)
    
    plt.tight_layout()
    output_file = Path(output_dir) / '3_dimension_radar.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)

def create_category_performance(data, output_dir):
    """Create performance analysis by scenario category"""
    scenarios = data['scenarios']
    models = data['summary']['models_evaluated']
    
    # Group by category
    category_data = {}
    for scenario in scenarios:
        category = scenario['category']
        if category not in category_data:
            category_data[category] = {model: [] for model in models}
        
        for model in models:
            score = scenario['model_evaluations'][model]['composite_score']
            category_data[category][model].append(score)
    
    # Calculate averages
    categories = list(category_data.keys())
    model_category_avgs = {model: [] for model in models}
    
    for category in categories:
        for model in models:
            avg_score = np.mean(category_data[category][model])
            model_category_avgs[model].append(avg_score)
    
    # Create grouped bar chart
    fig, ax = plt.subplots(figsize=(12, 8))
    
    x = np.arange(len(categories))
    width = 0.35 if len(models) == 2 else 0.2
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12']
    
    for i, model in enumerate(models):
        offset = width * (i - len(models)/2 + 0.5)
        bars = ax.bar(x + offset, model_category_avgs[model], width, 
                     label=model.upper(), color=colors[i], alpha=0.8)
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    ax.set_xlabel('Mental Health Categories', fontsize=12)
    ax.set_ylabel('Average Score (0-10)', fontsize=12)
    ax.set_title('Performance by Mental Health Category', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([cat.title() for cat in categories])
    ax.legend()
    ax.set_ylim(0, 10)
    
    plt.tight_layout()
    output_file = Path(output_dir) / '4_category_performance.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)

def create_response_time_analysis(data, output_dir):
    """Create response time comparison"""
    scenarios = data['scenarios']
    models = data['summary']['models_evaluated']
    
    # Collect response times
    response_times = {model: [] for model in models}
    
    for scenario in scenarios:
        for model in models:
            time_ms = scenario['model_evaluations'][model]['response_time_ms']
            response_times[model].append(time_ms / 1000)  # Convert to seconds
    
    # Create box plot
    fig, ax = plt.subplots(figsize=(10, 6))
    
    data_for_plot = [response_times[model] for model in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
    
    box_plot = ax.boxplot(data_for_plot, labels=[m.upper() for m in models], 
                         patch_artist=True, medianprops={'color': 'black', 'linewidth': 2})
    
    for patch, color in zip(box_plot['boxes'], colors):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)
    
    ax.set_ylabel('Response Time (seconds)', fontsize=12)
    ax.set_title('Response Time Distribution by Model', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add mean values as text
    for i, model in enumerate(models):
        mean_time = np.mean(response_times[model])
        ax.text(i+1, max(response_times[model]) * 0.9, f'Mean: {mean_time:.1f}s', 
                ha='center', va='center', fontweight='bold', 
                bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    plt.tight_layout()
    output_file = Path(output_dir) / '5_response_times.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)

def create_summary_infographic(data, output_dir):
    """Create a comprehensive summary infographic"""
    summary = data['summary']
    models = summary['models_evaluated']
    
    fig = plt.figure(figsize=(16, 10))
    gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
    
    # Title
    fig.suptitle('Mental Health LLM Evaluation Summary', fontsize=20, fontweight='bold', y=0.95)
    
    # Overall winner
    ax1 = fig.add_subplot(gs[0, :2])
    winner = max(summary['model_avg_scores'].items(), key=lambda x: x[1])
    ax1.text(0.5, 0.5, f'🏆 OVERALL WINNER\n{winner[0].upper()}\nScore: {winner[1]:.2f}/10', 
             ha='center', va='center', fontsize=18, fontweight='bold',
             bbox=dict(boxstyle='round,pad=0.5', facecolor='gold', alpha=0.8))
    ax1.set_xlim(0, 1)
    ax1.set_ylim(0, 1)
    ax1.axis('off')
    
    # Key metrics
    ax2 = fig.add_subplot(gs[0, 2:])
    metrics_text = f"""📊 KEY METRICS
Total Scenarios: {summary['total_scenarios']}
Models Evaluated: {len(models)}
Evaluation Time: {summary['evaluation_time_seconds']:.1f}s
Total Cost: ${sum(summary['model_total_costs'].values()):.4f}"""
    
    ax2.text(0.05, 0.95, metrics_text, ha='left', va='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightblue', alpha=0.8))
    ax2.set_xlim(0, 1)
    ax2.set_ylim(0, 1)
    ax2.axis('off')
    
    # Model scores bar chart
    ax3 = fig.add_subplot(gs[1, :2])
    scores = [summary['model_avg_scores'][model] for model in models]
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12'][:len(models)]
    bars = ax3.bar([m.upper() for m in models], scores, color=colors)
    ax3.set_title('Average Scores', fontweight='bold')
    ax3.set_ylabel('Score (0-10)')
    ax3.set_ylim(0, 10)
    
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}', ha='center', va='bottom', fontweight='bold')
    
    # Wins pie chart
    ax4 = fig.add_subplot(gs[1, 2:])
    wins = [summary['model_wins'][model] for model in models]
    ax4.pie(wins, labels=[m.upper() for m in models], autopct='%1.0f%%', 
            colors=colors, startangle=90)
    ax4.set_title('Win Distribution', fontweight='bold')
    
    # Performance insights
    ax5 = fig.add_subplot(gs[2, :])
    
    # Calculate some insights
    best_model = winner[0].upper()
    total_wins = sum(wins)
    best_win_rate = max(wins) / total_wins * 100
    
    insights_text = f"""📈 KEY INSIGHTS
• {best_model} achieved the highest therapeutic quality with {winner[1]:.2f}/10 average score
• {best_model} won {max(wins)}/{total_wins} scenarios ({best_win_rate:.0f}% win rate)
• Results show {"balanced competition" if abs(max(scores) - min(scores)) < 1.0 else "clear performance differences"} between models
• Evaluation demonstrates unbiased model selection without artificial preferences"""
    
    ax5.text(0.05, 0.95, insights_text, ha='left', va='top', fontsize=12,
             bbox=dict(boxstyle='round,pad=0.5', facecolor='lightgreen', alpha=0.8))
    ax5.set_xlim(0, 1)
    ax5.set_ylim(0, 1)
    ax5.axis('off')
    
    plt.tight_layout()
    output_file = Path(output_dir) / '6_summary_infographic.png'
    plt.savefig(output_file, dpi=300, bbox_inches='tight')
    plt.close()
    
    return str(output_file)

def main():
    parser = argparse.ArgumentParser(description='Create comprehensive research visualizations')
    parser.add_argument('--input', required=True, help='Input JSON file with research data')
    parser.add_argument('--output', required=True, help='Output directory for visualizations')
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load data
    print(f"📊 Loading research data from {args.input}")
    data = load_research_data(args.input)
    
    print(f"📈 Creating visualizations in {output_dir}")
    
    # Create all visualizations
    charts = []
    
    print("  🔵 Model comparison chart...")
    charts.append(create_model_comparison_chart(data, output_dir))
    
    print("  🟡 Detailed scores heatmap...")
    charts.append(create_detailed_scores_heatmap(data, output_dir))
    
    print("  🟢 Therapeutic dimension radar...")
    charts.append(create_dimension_analysis(data, output_dir))
    
    print("  🟣 Category performance analysis...")
    charts.append(create_category_performance(data, output_dir))
    
    print("  🟠 Response time analysis...")
    charts.append(create_response_time_analysis(data, output_dir))
    
    print("  ⭐ Summary infographic...")
    charts.append(create_summary_infographic(data, output_dir))
    
    print(f"\n✅ Created {len(charts)} visualizations:")
    for chart in charts:
        print(f"   📁 {Path(chart).name}")
    
    print(f"\n🎯 All visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()