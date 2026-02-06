#!/usr/bin/env python3
"""
Generate Research Visualizations
=================================

Creates publication-ready charts from research evaluation results.
"""

import json
import matplotlib.pyplot as plt
import numpy as np
import os
from pathlib import Path


def load_results(results_dir="results"):
    """Load evaluation results from JSON files."""

    detailed_path = Path(results_dir) / "evaluations" / "detailed_results.json"
    summary_path = Path(results_dir) / "evaluations" / "summary.json"

    if not detailed_path.exists():
        raise FileNotFoundError(f"Results not found at {detailed_path}")

    with open(detailed_path) as f:
        detailed = json.load(f)

    with open(summary_path) as f:
        summary = json.load(f)

    return detailed, summary


def calculate_dimension_averages(results):
    """Calculate average scores per dimension for each model."""

    model_dims = {}

    for result in results:
        if result.get('error'):
            continue

        model = result['model']
        scores = result['scores']

        if model not in model_dims:
            model_dims[model] = {
                'empathy': [], 'therapeutic': [], 'safety': [], 'clarity': []
            }

        for dim in ['empathy', 'therapeutic', 'safety', 'clarity']:
            if dim in scores:
                model_dims[model][dim].append(scores[dim])

    # Calculate averages
    averages = {}
    for model, dims in model_dims.items():
        averages[model] = {}
        for dim, values in dims.items():
            averages[model][dim] = np.mean(values) if values else 0

    return averages


def create_bar_chart(summary, output_dir):
    """Create model comparison bar chart."""

    models = list(summary['model_averages'].keys())
    scores = [summary['model_averages'][m] for m in models]

    # Sort by score descending
    sorted_pairs = sorted(zip(models, scores), key=lambda x: x[1], reverse=True)
    models, scores = zip(*sorted_pairs)

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 6))

    # Colors - highlight winner
    colors = ['#2ecc71' if m == summary['winner'] else '#3498db' for m in models]

    # Create bars
    bars = ax.bar(range(len(models)), scores, color=colors, edgecolor='white', linewidth=2)

    # Customize
    ax.set_xticks(range(len(models)))
    ax.set_xticklabels([m.upper() for m in models], fontsize=12, fontweight='bold')
    ax.set_ylabel('Composite Score (0-10)', fontsize=12)
    ax.set_ylim(0, 10)
    ax.set_title('Model Performance Comparison\nMental Health LLM Evaluation', fontsize=14, fontweight='bold')

    # Add score labels on bars
    for bar, score in zip(bars, scores):
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                f'{score:.2f}', ha='center', va='bottom', fontsize=11, fontweight='bold')

    # Add winner annotation
    ax.annotate('WINNER', xy=(0, scores[0]), xytext=(0.5, scores[0] + 0.8),
                fontsize=10, fontweight='bold', color='#27ae60',
                ha='center')

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / 'model_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_dimension_chart(dim_averages, output_dir):
    """Create dimension breakdown grouped bar chart."""

    models = list(dim_averages.keys())
    dimensions = ['empathy', 'therapeutic', 'safety', 'clarity']
    dim_labels = ['Empathy', 'Therapeutic', 'Safety', 'Clarity']

    # Prepare data
    x = np.arange(len(dimensions))
    width = 0.2

    fig, ax = plt.subplots(figsize=(12, 7))

    # Colors for each model
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # Create grouped bars
    for i, model in enumerate(models):
        values = [dim_averages[model][d] for d in dimensions]
        offset = (i - len(models)/2 + 0.5) * width
        bars = ax.bar(x + offset, values, width, label=model.upper(), color=colors[i], alpha=0.85)

    # Customize
    ax.set_xticks(x)
    ax.set_xticklabels(dim_labels, fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (0-10)', fontsize=12)
    ax.set_ylim(0, 11)
    ax.set_title('Performance by Evaluation Dimension\nMental Health LLM Evaluation', fontsize=14, fontweight='bold')
    ax.legend(loc='upper right', fontsize=10)

    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    output_path = Path(output_dir) / 'dimension_breakdown.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_radar_chart(dim_averages, winner, output_dir):
    """Create radar/spider chart comparing all models."""

    models = list(dim_averages.keys())
    dimensions = ['empathy', 'therapeutic', 'safety', 'clarity']
    dim_labels = ['Empathy', 'Therapeutic\nValue', 'Safety', 'Clarity']

    # Number of variables
    num_vars = len(dimensions)

    # Compute angle for each axis
    angles = [n / float(num_vars) * 2 * np.pi for n in range(num_vars)]
    angles += angles[:1]

    # Create figure
    fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))

    # Colors
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#9b59b6']

    # Plot each model
    for i, model in enumerate(models):
        values = [dim_averages[model][d] for d in dimensions]
        values += values[:1]

        linewidth = 4 if model == winner else 2
        alpha = 0.9 if model == winner else 0.7
        label = f'{model.upper()} (WINNER)' if model == winner else model.upper()

        ax.plot(angles, values, 'o-', linewidth=linewidth, label=label,
                color=colors[i], markersize=8, alpha=alpha)

        if model == winner:
            ax.fill(angles, values, alpha=0.15, color=colors[i])

    # Customize
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels(dim_labels, fontsize=11, fontweight='bold')
    ax.set_ylim(0, 11)
    ax.set_yticks([2, 4, 6, 8, 10])
    ax.grid(True, alpha=0.3)

    ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0), fontsize=10)

    plt.title('Multi-Dimensional Performance Analysis\nMental Health LLM Evaluation',
              fontsize=14, fontweight='bold', y=1.08)

    plt.tight_layout()

    output_path = Path(output_dir) / 'radar_comparison.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def create_scenario_heatmap(results, output_dir):
    """Create heatmap of scores across scenarios."""

    # Extract data
    models = sorted(set(r['model'] for r in results if not r.get('error')))
    scenarios = sorted(set(r['scenario'] for r in results if not r.get('error')))

    # Build matrix
    matrix = np.zeros((len(models), len(scenarios)))

    for result in results:
        if result.get('error'):
            continue
        m_idx = models.index(result['model'])
        s_idx = scenarios.index(result['scenario'])
        matrix[m_idx, s_idx] = result['composite_score']

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 6))

    # Create heatmap
    im = ax.imshow(matrix, cmap='RdYlGn', aspect='auto', vmin=0, vmax=10)

    # Labels
    ax.set_xticks(range(len(scenarios)))
    ax.set_xticklabels([s.replace('_', '\n') for s in scenarios], fontsize=9, rotation=45, ha='right')
    ax.set_yticks(range(len(models)))
    ax.set_yticklabels([m.upper() for m in models], fontsize=11, fontweight='bold')

    # Add score text
    for i in range(len(models)):
        for j in range(len(scenarios)):
            score = matrix[i, j]
            color = 'white' if score < 5 else 'black'
            ax.text(j, i, f'{score:.1f}', ha='center', va='center',
                   fontsize=9, color=color, fontweight='bold')

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label('Composite Score', fontsize=11)

    ax.set_title('Model Performance Across Mental Health Scenarios', fontsize=14, fontweight='bold')

    plt.tight_layout()

    output_path = Path(output_dir) / 'scenario_heatmap.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight', facecolor='white')
    plt.close()

    return output_path


def main():
    """Generate all visualizations."""

    print("Loading results...")
    results, summary = load_results()

    # Create output directory
    output_dir = Path("results/visualizations")
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Winner: {summary['winner'].upper()}")
    print(f"Generating visualizations to {output_dir}/\n")

    # Calculate dimension averages
    dim_averages = calculate_dimension_averages(results)

    # Generate charts
    charts = []

    print("  Creating bar chart...", end=" ")
    charts.append(create_bar_chart(summary, output_dir))
    print("done")

    print("  Creating dimension breakdown...", end=" ")
    charts.append(create_dimension_chart(dim_averages, output_dir))
    print("done")

    print("  Creating radar chart...", end=" ")
    charts.append(create_radar_chart(dim_averages, summary['winner'], output_dir))
    print("done")

    print("  Creating scenario heatmap...", end=" ")
    charts.append(create_scenario_heatmap(results, output_dir))
    print("done")

    print(f"\nGenerated {len(charts)} visualizations:")
    for chart in charts:
        print(f"  - {chart}")


if __name__ == "__main__":
    main()
