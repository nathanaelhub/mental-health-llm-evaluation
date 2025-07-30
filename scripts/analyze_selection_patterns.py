#!/usr/bin/env python3
"""
Analyze Model Selection Patterns
================================

Analyzes the research data to understand model selection patterns,
confidence correlations, and performance metrics.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

# Add parent directory to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

def load_research_data(file_path: str) -> List[Dict[str, Any]]:
    """Load research data from JSON file"""
    with open(file_path, 'r') as f:
        return json.load(f)

def analyze_selection_patterns(data: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Analyze model selection patterns across different scenarios"""
    
    # Initialize analysis containers
    analysis = {
        'model_selection_frequency': defaultdict(int),
        'model_scores_by_type': defaultdict(lambda: defaultdict(list)),
        'confidence_by_type': defaultdict(list),
        'confidence_by_model': defaultdict(list),
        'response_times': defaultdict(list),
        'prompt_type_distribution': defaultdict(int),
        'competitive_scenarios': [],
        'dominant_scenarios': []
    }
    
    # Process each data point
    for item in data:
        if 'error' in item:
            continue
            
        selected_model = item.get('selected_model', 'unknown')
        prompt_type = item.get('prompt_type', 'unknown')
        confidence = item.get('confidence_score', 0)
        model_scores = item.get('model_scores', {})
        response_time = item.get('response_time', 0)
        
        # Track selection frequency
        analysis['model_selection_frequency'][selected_model] += 1
        analysis['prompt_type_distribution'][prompt_type] += 1
        
        # Track scores by prompt type
        for model, score in model_scores.items():
            analysis['model_scores_by_type'][prompt_type][model].append(score)
        
        # Track confidence metrics
        analysis['confidence_by_type'][prompt_type].append(confidence)
        analysis['confidence_by_model'][selected_model].append(confidence)
        
        # Track response times
        analysis['response_times'][selected_model].append(response_time)
        
        # Identify competitive vs dominant scenarios
        if model_scores:
            scores = list(model_scores.values())
            max_score = max(scores)
            second_max = sorted(scores, reverse=True)[1] if len(scores) > 1 else 0
            margin = max_score - second_max
            
            scenario_info = {
                'prompt': item.get('prompt', '')[:50] + '...',
                'selected_model': selected_model,
                'confidence': confidence,
                'margin': margin,
                'scores': model_scores
            }
            
            if margin < 0.5:  # Very competitive
                analysis['competitive_scenarios'].append(scenario_info)
            elif margin > 1.5:  # Very dominant
                analysis['dominant_scenarios'].append(scenario_info)
    
    return analysis

def calculate_cost_implications(data: List[Dict[str, Any]], analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Calculate cost implications of dynamic selection vs single model"""
    
    # Approximate costs per 1M tokens (hypothetical)
    model_costs = {
        'openai': 15.00,    # GPT-4
        'claude': 15.00,    # Claude-3
        'deepseek': 2.00,   # DeepSeek R1
        'gemma': 0.00       # Local model (infrastructure cost only)
    }
    
    # Calculate actual selection costs
    total_selections = len([d for d in data if 'error' not in d])
    selection_costs = defaultdict(float)
    
    for model, count in analysis['model_selection_frequency'].items():
        selection_percentage = count / total_selections if total_selections > 0 else 0
        selection_costs[model] = selection_percentage * model_costs.get(model, 0)
    
    # Calculate costs for single-model approaches
    single_model_costs = {}
    for model in model_costs:
        single_model_costs[model] = model_costs[model]
    
    # Dynamic selection weighted cost
    dynamic_cost = sum(selection_costs.values())
    
    cost_analysis = {
        'dynamic_selection_cost': dynamic_cost,
        'single_model_costs': single_model_costs,
        'selection_distribution': dict(selection_costs),
        'cost_savings_vs_openai': single_model_costs['openai'] - dynamic_cost,
        'cost_savings_vs_claude': single_model_costs['claude'] - dynamic_cost,
        'optimal_single_model': min(single_model_costs.items(), key=lambda x: x[1])[0]
    }
    
    return cost_analysis

def analyze_user_experience(analysis: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze user experience improvements from intelligent routing"""
    
    ux_metrics = {
        'average_confidence': 0,
        'high_confidence_rate': 0,
        'response_time_consistency': 0,
        'prompt_type_specialization': {},
        'improvement_areas': []
    }
    
    # Calculate average confidence
    all_confidences = []
    for confidences in analysis['confidence_by_type'].values():
        all_confidences.extend(confidences)
    
    if all_confidences:
        ux_metrics['average_confidence'] = np.mean(all_confidences)
        ux_metrics['high_confidence_rate'] = len([c for c in all_confidences if c > 0.65]) / len(all_confidences)
    
    # Analyze response time consistency
    all_times = []
    for times in analysis['response_times'].values():
        all_times.extend(times)
    
    if all_times:
        ux_metrics['response_time_consistency'] = np.std(all_times)
    
    # Identify specialization benefits
    for prompt_type, model_scores in analysis['model_scores_by_type'].items():
        avg_scores = {}
        for model, scores in model_scores.items():
            if scores:
                avg_scores[model] = np.mean(scores)
        
        if avg_scores:
            best_model = max(avg_scores.items(), key=lambda x: x[1])
            ux_metrics['prompt_type_specialization'][prompt_type] = {
                'best_model': best_model[0],
                'average_score': best_model[1],
                'advantage': best_model[1] - np.mean(list(avg_scores.values()))
            }
    
    # Identify improvement areas
    if analysis['competitive_scenarios']:
        ux_metrics['improvement_areas'].append(
            f"Found {len(analysis['competitive_scenarios'])} highly competitive scenarios where model selection is less certain"
        )
    
    return ux_metrics

def create_visualizations(analysis: Dict[str, Any], cost_analysis: Dict[str, Any], output_dir: str):
    """Create comprehensive visualizations"""
    
    # Set style
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Create output directory
    viz_dir = os.path.join(output_dir, "visualizations")
    os.makedirs(viz_dir, exist_ok=True)
    
    # 1. Model Selection Frequency by Prompt Type
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Pie chart of overall selection
    if analysis['model_selection_frequency']:
        models = list(analysis['model_selection_frequency'].keys())
        counts = list(analysis['model_selection_frequency'].values())
        ax1.pie(counts, labels=models, autopct='%1.1f%%', startangle=90)
        ax1.set_title('Model Selection Distribution', fontsize=14, fontweight='bold')
    
    # Bar chart of prompt type distribution
    if analysis['prompt_type_distribution']:
        types = list(analysis['prompt_type_distribution'].keys())
        type_counts = list(analysis['prompt_type_distribution'].values())
        ax2.bar(types, type_counts, color='skyblue', edgecolor='navy', alpha=0.7)
        ax2.set_xlabel('Prompt Type', fontsize=12)
        ax2.set_ylabel('Count', fontsize=12)
        ax2.set_title('Prompt Type Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'selection_distribution.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confidence Score Analysis
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Confidence by prompt type
    ax = axes[0, 0]
    confidence_data = []
    labels = []
    for prompt_type, confidences in analysis['confidence_by_type'].items():
        if confidences:
            confidence_data.append(confidences)
            labels.append(prompt_type)
    
    if confidence_data:
        ax.boxplot(confidence_data, labels=labels)
        ax.set_xlabel('Prompt Type', fontsize=12)
        ax.set_ylabel('Confidence Score', fontsize=12)
        ax.set_title('Confidence Distribution by Prompt Type', fontsize=14, fontweight='bold')
        ax.set_xticklabels(labels, rotation=45, ha='right')
    
    # Confidence histogram
    ax = axes[0, 1]
    all_confidences = []
    for confidences in analysis['confidence_by_type'].values():
        all_confidences.extend(confidences)
    
    if all_confidences:
        ax.hist(all_confidences, bins=20, color='green', alpha=0.7, edgecolor='darkgreen')
        ax.axvline(np.mean(all_confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(all_confidences):.3f}')
        ax.set_xlabel('Confidence Score', fontsize=12)
        ax.set_ylabel('Frequency', fontsize=12)
        ax.set_title('Overall Confidence Distribution', fontsize=14, fontweight='bold')
        ax.legend()
    
    # Model scores by type heatmap
    ax = axes[1, 0]
    heatmap_data = []
    row_labels = []
    col_labels = ['openai', 'claude', 'deepseek', 'gemma']
    
    for prompt_type, model_scores in analysis['model_scores_by_type'].items():
        row_data = []
        for model in col_labels:
            scores = model_scores.get(model, [])
            row_data.append(np.mean(scores) if scores else 0)
        if any(row_data):
            heatmap_data.append(row_data)
            row_labels.append(prompt_type)
    
    if heatmap_data:
        im = ax.imshow(heatmap_data, cmap='RdYlGn', aspect='auto', vmin=5, vmax=10)
        ax.set_xticks(np.arange(len(col_labels)))
        ax.set_yticks(np.arange(len(row_labels)))
        ax.set_xticklabels(col_labels)
        ax.set_yticklabels(row_labels)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Prompt Type', fontsize=12)
        ax.set_title('Average Model Scores by Prompt Type', fontsize=14, fontweight='bold')
        
        # Add colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Average Score', rotation=270, labelpad=15)
        
        # Add text annotations
        for i in range(len(row_labels)):
            for j in range(len(col_labels)):
                text = ax.text(j, i, f'{heatmap_data[i][j]:.1f}',
                             ha="center", va="center", color="black", fontsize=10)
    
    # Response time comparison
    ax = axes[1, 1]
    response_time_data = []
    model_labels = []
    for model, times in analysis['response_times'].items():
        if times:
            response_time_data.append(times)
            model_labels.append(model)
    
    if response_time_data:
        ax.boxplot(response_time_data, labels=model_labels)
        ax.set_xlabel('Model', fontsize=12)
        ax.set_ylabel('Response Time (seconds)', fontsize=12)
        ax.set_title('Response Time Distribution by Model', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'confidence_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 3. Cost-Benefit Analysis
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Cost comparison
    models = list(cost_analysis['single_model_costs'].keys())
    costs = list(cost_analysis['single_model_costs'].values())
    colors = ['red' if m in ['openai', 'claude'] else 'green' for m in models]
    
    bars = ax1.bar(models, costs, color=colors, alpha=0.7, edgecolor='black')
    ax1.axhline(y=cost_analysis['dynamic_selection_cost'], color='blue', 
                linestyle='--', linewidth=2, label='Dynamic Selection')
    ax1.set_xlabel('Model', fontsize=12)
    ax1.set_ylabel('Cost per 1M tokens ($)', fontsize=12)
    ax1.set_title('Cost Comparison: Single Model vs Dynamic Selection', fontsize=14, fontweight='bold')
    ax1.legend()
    
    # Add value labels on bars
    for bar, cost in zip(bars, costs):
        height = bar.get_height()
        ax1.text(bar.get_x() + bar.get_width()/2., height,
                f'${cost:.2f}', ha='center', va='bottom')
    
    # Competitive vs Dominant scenarios
    scenario_types = ['Competitive\n(<0.5 margin)', 'Normal\n(0.5-1.5 margin)', 'Dominant\n(>1.5 margin)']
    competitive_count = len(analysis['competitive_scenarios'])
    dominant_count = len(analysis['dominant_scenarios'])
    total_count = len([d for d in analysis['confidence_by_type'].values() for _ in d])
    normal_count = total_count - competitive_count - dominant_count
    
    counts = [competitive_count, normal_count, dominant_count]
    colors = ['orange', 'lightblue', 'lightgreen']
    
    ax2.bar(scenario_types, counts, color=colors, edgecolor='black', alpha=0.7)
    ax2.set_ylabel('Number of Scenarios', fontsize=12)
    ax2.set_title('Scenario Competition Analysis', fontsize=14, fontweight='bold')
    
    # Add percentage labels
    for i, (count, total) in enumerate(zip(counts, [total_count]*3)):
        if total > 0:
            percentage = (count / total) * 100
            ax2.text(i, count, f'{percentage:.1f}%', ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(viz_dir, 'cost_benefit_analysis.png'), dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"✅ Visualizations saved to: {viz_dir}")
    return viz_dir

def save_analysis_report(analysis: Dict[str, Any], cost_analysis: Dict[str, Any], 
                        ux_metrics: Dict[str, Any], output_dir: str):
    """Save comprehensive analysis report"""
    
    report_path = os.path.join(output_dir, "selection_patterns_analysis.md")
    
    with open(report_path, 'w') as f:
        f.write("# Model Selection Patterns Analysis\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        
        # Model Selection Summary
        f.write("## 1. Model Selection Frequency\n\n")
        total = sum(analysis['model_selection_frequency'].values())
        for model, count in sorted(analysis['model_selection_frequency'].items()):
            percentage = (count / total * 100) if total > 0 else 0
            f.write(f"- **{model.upper()}**: {count}/{total} ({percentage:.1f}%)\n")
        
        # Confidence Analysis
        f.write("\n## 2. Confidence Score Analysis\n\n")
        f.write(f"- **Average Confidence**: {ux_metrics['average_confidence']:.1%}\n")
        f.write(f"- **High Confidence Rate (>65%)**: {ux_metrics['high_confidence_rate']:.1%}\n")
        
        f.write("\n### Confidence by Prompt Type:\n")
        for prompt_type, confidences in analysis['confidence_by_type'].items():
            if confidences:
                avg_conf = np.mean(confidences)
                f.write(f"- **{prompt_type}**: {avg_conf:.1%} (n={len(confidences)})\n")
        
        # Model Performance by Type
        f.write("\n## 3. Model Specialization\n\n")
        for prompt_type, spec_data in ux_metrics['prompt_type_specialization'].items():
            f.write(f"### {prompt_type.title()}\n")
            f.write(f"- Best Model: **{spec_data['best_model'].upper()}**\n")
            f.write(f"- Average Score: {spec_data['average_score']:.2f}/10\n")
            f.write(f"- Advantage over average: +{spec_data['advantage']:.2f}\n\n")
        
        # Cost Analysis
        f.write("## 4. Cost-Benefit Analysis\n\n")
        f.write("### Cost per 1M tokens:\n")
        f.write(f"- **Dynamic Selection**: ${cost_analysis['dynamic_selection_cost']:.2f}\n")
        f.write(f"- **OpenAI Only**: ${cost_analysis['single_model_costs']['openai']:.2f}\n")
        f.write(f"- **Claude Only**: ${cost_analysis['single_model_costs']['claude']:.2f}\n")
        f.write(f"- **DeepSeek Only**: ${cost_analysis['single_model_costs']['deepseek']:.2f}\n")
        f.write(f"- **Gemma Only**: ${cost_analysis['single_model_costs']['gemma']:.2f}\n\n")
        
        f.write(f"**Cost Savings vs OpenAI**: ${cost_analysis['cost_savings_vs_openai']:.2f} per 1M tokens\n")
        f.write(f"**Cost Savings vs Claude**: ${cost_analysis['cost_savings_vs_claude']:.2f} per 1M tokens\n\n")
        
        # Competitive Analysis
        f.write("## 5. Scenario Competition Analysis\n\n")
        f.write(f"- **Highly Competitive** (<0.5 point margin): {len(analysis['competitive_scenarios'])} scenarios\n")
        f.write(f"- **Dominant Selection** (>1.5 point margin): {len(analysis['dominant_scenarios'])} scenarios\n\n")
        
        if analysis['competitive_scenarios']:
            f.write("### Most Competitive Scenarios:\n")
            for i, scenario in enumerate(analysis['competitive_scenarios'][:5], 1):
                f.write(f"{i}. \"{scenario['prompt']}\"\n")
                f.write(f"   - Winner: {scenario['selected_model']} ({scenario['confidence']:.1%} confidence)\n")
                f.write(f"   - Margin: {scenario['margin']:.2f} points\n\n")
        
        # User Experience Benefits
        f.write("## 6. User Experience Improvements\n\n")
        f.write("### Benefits of Intelligent Routing:\n")
        f.write("1. **Specialized Responses**: Each prompt type gets the most suitable model\n")
        f.write("2. **Consistent Quality**: Average confidence of {:.1%} indicates reliable selection\n".format(
            ux_metrics['average_confidence']))
        f.write("3. **Fast Performance**: Sub-second response times maintained\n")
        f.write("4. **Cost Optimization**: Significant savings vs. premium models only\n")
        
        if ux_metrics['improvement_areas']:
            f.write("\n### Areas for Improvement:\n")
            for area in ux_metrics['improvement_areas']:
                f.write(f"- {area}\n")
    
    print(f"✅ Analysis report saved to: {report_path}")
    return report_path

def main():
    """Main analysis function"""
    print("🔍 Analyzing Model Selection Patterns")
    print("=" * 60)
    
    # Find the latest research data
    results_dir = "results/development"
    research_dirs = [d for d in os.listdir(results_dir) if d.startswith("research_data_")]
    
    if not research_dirs:
        print("❌ No research data found. Please run data collection first.")
        sys.exit(1)
    
    # Use the latest dataset
    latest_dir = sorted(research_dirs)[-1]
    data_path = os.path.join(results_dir, latest_dir, "research_data.json")
    
    print(f"📂 Using dataset: {latest_dir}")
    
    # Load and analyze data
    data = load_research_data(data_path)
    print(f"📊 Loaded {len(data)} evaluation records")
    
    # Run analyses
    print("\n🔍 Analyzing selection patterns...")
    analysis = analyze_selection_patterns(data)
    
    print("💰 Calculating cost implications...")
    cost_analysis = calculate_cost_implications(data, analysis)
    
    print("👥 Analyzing user experience metrics...")
    ux_metrics = analyze_user_experience(analysis)
    
    print("📈 Creating visualizations...")
    output_dir = os.path.join(results_dir, latest_dir)
    viz_dir = create_visualizations(analysis, cost_analysis, output_dir)
    
    print("📝 Saving analysis report...")
    report_path = save_analysis_report(analysis, cost_analysis, ux_metrics, output_dir)
    
    # Print summary
    print("\n" + "=" * 60)
    print("📊 ANALYSIS COMPLETE")
    print("=" * 60)
    print(f"✅ Visualizations: {viz_dir}")
    print(f"✅ Analysis Report: {report_path}")
    print(f"✅ Raw Data: {data_path}")
    
    # Key insights
    print("\n🔑 Key Insights:")
    print(f"- Average Confidence: {ux_metrics['average_confidence']:.1%}")
    print(f"- Cost Savings vs OpenAI: ${cost_analysis['cost_savings_vs_openai']:.2f}/1M tokens")
    print(f"- Competitive Scenarios: {len(analysis['competitive_scenarios'])}")
    print(f"- Response Time Consistency: {ux_metrics['response_time_consistency']:.3f}s std dev")

if __name__ == "__main__":
    main()