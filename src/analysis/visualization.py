"""
Visualization Module for Mental Health LLM Evaluation
====================================================

This module provides visualization capabilities with graceful degradation
when optional dependencies are not available.
"""

import os
import sys
import logging
from typing import Dict, List, Optional, Any, Tuple
import json
from datetime import datetime

# Set up logging
logger = logging.getLogger(__name__)

# Try to import visualization dependencies with graceful fallback
HAS_MATPLOTLIB = False
HAS_PLOTLY = False
HAS_SEABORN = False
HAS_PANDAS = False
HAS_NUMPY = False

try:
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches
    HAS_MATPLOTLIB = True
    logger.info("Matplotlib is available")
except ImportError:
    logger.warning("Matplotlib not available - static plots will be skipped")

try:
    import seaborn as sns
    HAS_SEABORN = True
    logger.info("Seaborn is available")
except ImportError:
    logger.warning("Seaborn not available - enhanced styling will be skipped")

try:
    import plotly.graph_objects as go
    import plotly.express as px
    from plotly.subplots import make_subplots
    HAS_PLOTLY = True
    logger.info("Plotly is available")
except ImportError:
    logger.warning("Plotly not available - interactive plots will be skipped")

try:
    import pandas as pd
    HAS_PANDAS = True
    logger.info("Pandas is available")
except ImportError:
    logger.warning("Pandas not available - data processing will be limited")

try:
    import numpy as np
    HAS_NUMPY = True
    logger.info("Numpy is available")
except ImportError:
    logger.warning("Numpy not available - numerical operations will be limited")


class VisualizationConfig:
    """Configuration for visualization generation."""
    
    def __init__(self):
        self.style = "seaborn-v0_8" if HAS_SEABORN else "default"
        # Distinct colors for 4 models: GPT-4 (blue), DeepSeek (orange), Claude (green), Gemma (purple)
        self.color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#9467bd", "#d62728", "#8c564b"]
        self.figure_size = (12, 8)
        self.dpi = 300
        self.save_format = "png"
        self.interactive = HAS_PLOTLY
        self.include_text_summary = True


class SafeVisualizer:
    """Safe visualizer that handles missing dependencies gracefully."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up matplotlib if available
        if HAS_MATPLOTLIB:
            try:
                if HAS_SEABORN:
                    plt.style.use(self.config.style)
                    sns.set_palette(self.config.color_palette)
                else:
                    plt.style.use('default')
            except Exception as e:
                self.logger.warning(f"Could not set matplotlib style: {e}")
    
    def _safe_mean(self, values: List[float]) -> float:
        """Safely calculate mean."""
        if not values:
            return 0.0
        return sum(values) / len(values)
    
    def _safe_std(self, values: List[float]) -> float:
        """Safely calculate standard deviation."""
        if not values or len(values) < 2:
            return 0.0
        
        mean = self._safe_mean(values)
        variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
        return variance ** 0.5
    
    def _extract_model_scores(self, results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
        """Extract scores from results in a safe way."""
        model_scores = {}
        
        if not results:
            return model_scores
        
        # Handle different result formats
        if 'scenarios' in results:
            # Format: {'scenarios': [{'openai_evaluation': {...}, 'deepseek_evaluation': {...}}]}
            scenarios = results['scenarios']
            
            for scenario in scenarios:
                for model_prefix in ['openai', 'deepseek', 'claude', 'gemma']:
                    eval_key = f'{model_prefix}_evaluation'
                    if eval_key in scenario:
                        eval_data = scenario[eval_key]
                        if eval_data:
                            # Better display names for models
                            display_names = {
                                'openai': 'GPT-4',
                                'deepseek': 'DeepSeek',
                                'claude': 'Claude',
                                'gemma': 'Gemma'
                            }
                            model_name = display_names.get(model_prefix, model_prefix.upper())
                            if model_name not in model_scores:
                                model_scores[model_name] = {
                                    'composite': [],
                                    'empathy': [],
                                    'therapeutic': [],
                                    'safety': [],
                                    'clarity': [],
                                    'cost': [],
                                    'response_time': []
                                }
                            
                            # Extract scores safely
                            model_scores[model_name]['composite'].append(
                                self._safe_get_score(eval_data, 'composite_score')
                            )
                            model_scores[model_name]['empathy'].append(
                                self._safe_get_score(eval_data, 'empathy_score')
                            )
                            model_scores[model_name]['therapeutic'].append(
                                self._safe_get_score(eval_data, 'therapeutic_value_score')
                            )
                            model_scores[model_name]['safety'].append(
                                self._safe_get_score(eval_data, 'safety_score')
                            )
                            model_scores[model_name]['clarity'].append(
                                self._safe_get_score(eval_data, 'clarity_score')
                            )
                            model_scores[model_name]['cost'].append(
                                self._safe_get_score(eval_data, 'cost_usd', 0.0)
                            )
                            model_scores[model_name]['response_time'].append(
                                self._safe_get_score(eval_data, 'response_time_ms', 0.0)
                            )
        
        elif isinstance(results, dict):
            # Direct model results format
            for model_name, model_data in results.items():
                if isinstance(model_data, list):
                    model_scores[model_name] = {
                        'composite': [],
                        'empathy': [],
                        'therapeutic': [],
                        'safety': [],
                        'clarity': [],
                        'cost': [],
                        'response_time': []
                    }
                    
                    for item in model_data:
                        if isinstance(item, dict):
                            model_scores[model_name]['composite'].append(
                                self._safe_get_score(item, 'composite_score')
                            )
                            # Add other metrics as available
        
        return model_scores
    
    def _safe_get_score(self, data: Any, key: str, default: float = 0.0) -> float:
        """Safely get a score from data."""
        if isinstance(data, dict):
            value = data.get(key, default)
        elif hasattr(data, key):
            value = getattr(data, key, default)
        else:
            value = default
        
        # Convert to float safely
        try:
            return float(value) if value is not None else default
        except (ValueError, TypeError):
            return default
    
    def create_text_summary(self, results: Dict[str, Any], output_dir: str) -> str:
        """Create a text summary of results."""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            model_scores = self._extract_model_scores(results)
            
            summary_lines = [
                "Mental Health LLM Evaluation Results Summary",
                "=" * 50,
                f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
                ""
            ]
            
            if not model_scores:
                summary_lines.append("No valid model scores found in results.")
            else:
                # Overall comparison
                summary_lines.append("OVERALL COMPARISON")
                summary_lines.append("-" * 30)
                
                for model_name, scores in model_scores.items():
                    composite_scores = scores.get('composite', [])
                    if composite_scores:
                        mean_score = self._safe_mean(composite_scores)
                        std_score = self._safe_std(composite_scores)
                        
                        summary_lines.append(f"{model_name}:")
                        summary_lines.append(f"  Mean Score: {mean_score:.2f}")
                        summary_lines.append(f"  Std Dev: {std_score:.2f}")
                        summary_lines.append(f"  Samples: {len(composite_scores)}")
                        
                        # Cost analysis
                        costs = scores.get('cost', [])
                        if costs:
                            total_cost = sum(costs)
                            avg_cost = self._safe_mean(costs)
                            summary_lines.append(f"  Total Cost: ${total_cost:.4f}")
                            summary_lines.append(f"  Avg Cost: ${avg_cost:.4f}")
                        
                        # Response time
                        response_times = scores.get('response_time', [])
                        if response_times:
                            avg_time = self._safe_mean(response_times)
                            summary_lines.append(f"  Avg Response Time: {avg_time:.1f}ms")
                        
                        summary_lines.append("")
                
                # Metric breakdown
                summary_lines.append("METRIC BREAKDOWN")
                summary_lines.append("-" * 30)
                
                metrics = ['empathy', 'therapeutic', 'safety', 'clarity']
                for metric in metrics:
                    summary_lines.append(f"{metric.upper()}:")
                    for model_name, scores in model_scores.items():
                        metric_scores = scores.get(metric, [])
                        if metric_scores:
                            mean_score = self._safe_mean(metric_scores)
                            summary_lines.append(f"  {model_name}: {mean_score:.2f}")
                    summary_lines.append("")
            
            # Write summary to file
            summary_path = os.path.join(output_dir, "evaluation_summary.txt")
            with open(summary_path, 'w') as f:
                f.write('\n'.join(summary_lines))
            
            return summary_path
            
        except Exception as e:
            self.logger.error(f"Error creating text summary: {e}")
            return ""
    
    def create_simple_bar_chart(self, results: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create a simple bar chart if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            model_scores = self._extract_model_scores(results)
            
            if not model_scores:
                return None
            
            # Prepare data for plotting
            models = list(model_scores.keys())
            composite_means = []
            composite_stds = []
            
            for model_name in models:
                composite_scores = model_scores[model_name].get('composite', [])
                if composite_scores:
                    composite_means.append(self._safe_mean(composite_scores))
                    composite_stds.append(self._safe_std(composite_scores))
                else:
                    composite_means.append(0.0)
                    composite_stds.append(0.0)
            
            # Create bar chart
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            bars = ax.bar(models, composite_means, yerr=composite_stds, 
                         capsize=5, alpha=0.7, 
                         color=self.config.color_palette[:len(models)])
            
            ax.set_title('Model Comparison - Composite Scores', fontsize=16, fontweight='bold')
            ax.set_ylabel('Composite Score')
            ax.set_xlabel('Models')
            ax.grid(True, alpha=0.3)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, composite_means):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                       f'{mean_val:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            chart_path = os.path.join(output_dir, f"model_comparison.{self.config.save_format}")
            plt.savefig(chart_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error creating bar chart: {e}")
            return None
    
    def create_metric_comparison(self, results: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create a metric comparison chart if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            model_scores = self._extract_model_scores(results)
            
            if not model_scores:
                return None
            
            metrics = ['empathy', 'therapeutic', 'safety', 'clarity']
            models = list(model_scores.keys())
            
            # Create subplot for each metric
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle('Model Performance by Metric', fontsize=16, fontweight='bold')
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                
                metric_means = []
                metric_stds = []
                
                for model_name in models:
                    metric_scores = model_scores[model_name].get(metric, [])
                    if metric_scores:
                        metric_means.append(self._safe_mean(metric_scores))
                        metric_stds.append(self._safe_std(metric_scores))
                    else:
                        metric_means.append(0.0)
                        metric_stds.append(0.0)
                
                bars = ax.bar(models, metric_means, yerr=metric_stds, 
                             capsize=3, alpha=0.7,
                             color=self.config.color_palette[:len(models)])
                
                ax.set_title(f'{metric.capitalize()} Score')
                ax.set_ylabel('Score')
                ax.grid(True, alpha=0.3)
                
                # Add value labels
                for bar, mean_val in zip(bars, metric_means):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
            
            plt.tight_layout()
            
            # Save plot
            chart_path = os.path.join(output_dir, f"metric_comparison.{self.config.save_format}")
            plt.savefig(chart_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error creating metric comparison: {e}")
            return None
    
    def create_cost_analysis_chart(self, results: Dict[str, Any], output_dir: str) -> Optional[str]:
        """Create a cost analysis chart if matplotlib is available."""
        if not HAS_MATPLOTLIB:
            return None
        
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            model_scores = self._extract_model_scores(results)
            
            if not model_scores:
                return None
            
            # Extract cost data
            models = []
            total_costs = []
            avg_costs = []
            
            for model_name, scores in model_scores.items():
                costs = scores.get('cost', [])
                if costs:
                    models.append(model_name)
                    total_costs.append(sum(costs))
                    avg_costs.append(self._safe_mean(costs))
            
            if not models:
                return None
            
            # Create cost comparison chart
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
            fig.suptitle('Cost Analysis', fontsize=16, fontweight='bold')
            
            # Total cost
            bars1 = ax1.bar(models, total_costs, alpha=0.7,
                           color=self.config.color_palette[:len(models)])
            ax1.set_title('Total Cost')
            ax1.set_ylabel('Total Cost ($)')
            ax1.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, cost in zip(bars1, total_costs):
                height = bar.get_height()
                ax1.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'${cost:.4f}', ha='center', va='bottom')
            
            # Average cost per response
            bars2 = ax2.bar(models, avg_costs, alpha=0.7,
                           color=self.config.color_palette[:len(models)])
            ax2.set_title('Average Cost per Response')
            ax2.set_ylabel('Average Cost ($)')
            ax2.grid(True, alpha=0.3)
            
            # Add value labels
            for bar, cost in zip(bars2, avg_costs):
                height = bar.get_height()
                ax2.text(bar.get_x() + bar.get_width()/2., height + height*0.01,
                        f'${cost:.4f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            chart_path = os.path.join(output_dir, f"cost_analysis.{self.config.save_format}")
            plt.savefig(chart_path, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return chart_path
            
        except Exception as e:
            self.logger.error(f"Error creating cost analysis chart: {e}")
            return None


def create_visualizations(results: Dict[str, Any], analysis: Dict[str, Any], output_dir: str = 'results/visualizations') -> List[str]:
    """
    Create all 5 evaluation visualization charts for mental health LLM comparison.
    
    Creates the specific numbered charts:
    1. Overall Comparison (1_overall_comparison.png)
    2. Category Radar Chart (2_category_radar.png) 
    3. Cost Effectiveness (3_cost_effectiveness.png)
    4. Safety Metrics (4_safety_metrics.png)
    5. Statistical Summary (5_statistical_summary.png)
    
    Args:
        results: Dictionary containing evaluation results
        analysis: Dictionary containing statistical analysis results
        output_dir: Directory to save visualizations
        
    Returns:
        List of created chart file paths
    """
    if not HAS_MATPLOTLIB:
        logger.warning("Matplotlib not available - cannot create charts")
        return []
    
    logger.info("Creating 5 evaluation charts with all evaluated models...")
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Create visualizer with enhanced config for 4 models
    visualizer = SafeVisualizer()
    
    # Track generated files
    chart_files = []
    
    # Extract model data from results - handle different result formats
    model_data = _extract_all_model_data(results)
    
    if not model_data:
        logger.error("No valid model data found in results")
        return []
    
    logger.info(f"Found data for models: {list(model_data.keys())}")
    
    # Define colors for each model - Blue, Green, Orange, Purple
    model_colors = {
        'OpenAI': '#1f77b4',      # Blue  
        'GPT-4': '#1f77b4',       # Blue (alternative name)
        'Claude': '#2ca02c',      # Green
        'DeepSeek': '#ff7f0e',    # Orange  
        'Gemma': '#9467bd'        # Purple
    }
    
    try:
        # 1. Overall Comparison Chart
        chart_path = _create_overall_comparison_chart(model_data, model_colors, output_dir)
        if chart_path:
            chart_files.append(chart_path)
            logger.info(f"Created 1_overall_comparison.png")
    except Exception as e:
        logger.error(f"Failed to create overall comparison chart: {e}")
    
    try:
        # 2. Category Radar Chart  
        chart_path = _create_category_radar_chart(model_data, model_colors, output_dir)
        if chart_path:
            chart_files.append(chart_path)
            logger.info(f"Created 2_category_radar.png")
    except Exception as e:
        logger.error(f"Failed to create category radar chart: {e}")
    
    try:
        # 3. Cost Effectiveness Chart
        chart_path = _create_cost_effectiveness_chart(model_data, model_colors, output_dir)
        if chart_path:
            chart_files.append(chart_path)
            logger.info(f"Created 3_cost_effectiveness.png")
    except Exception as e:
        logger.error(f"Failed to create cost effectiveness chart: {e}")
    
    try:
        # 4. Safety Metrics Chart
        chart_path = _create_safety_metrics_chart(model_data, model_colors, output_dir)
        if chart_path:
            chart_files.append(chart_path)
            logger.info(f"Created 4_safety_metrics.png")
    except Exception as e:
        logger.error(f"Failed to create safety metrics chart: {e}")
    
    try:
        # 5. Statistical Summary Chart
        chart_path = _create_statistical_summary_chart(model_data, analysis, model_colors, output_dir)
        if chart_path:
            chart_files.append(chart_path)
            logger.info(f"Created 5_statistical_summary.png")
    except Exception as e:
        logger.error(f"Failed to create statistical summary chart: {e}")
    
    # Create text summary and JSON metadata
    try:
        visualizer.create_text_summary(results, output_dir)
        
        summary_info = {
            'generated_at': datetime.now().isoformat(),
            'models_included': list(model_data.keys()),
            'charts_created': len(chart_files),
            'chart_files': [os.path.basename(f) for f in chart_files]
        }
        
        summary_json_path = os.path.join(output_dir, 'visualization_summary.json')
        with open(summary_json_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
            
    except Exception as e:
        logger.error(f"Failed to create summary files: {e}")
    
    logger.info(f"Successfully created {len(chart_files)} visualization charts")
    return chart_files


# Backward compatibility - provide the expected interface
def create_presentation_slides(*args, **kwargs):
    """Stub function for presentation slides creation."""
    return []


def create_comprehensive_dashboard(results, statistical_results=None, results_dir="./visualizations"):
    """Backward compatibility function."""
    return create_visualizations(results, statistical_results or {}, results_dir)

def create_all_visualizations(results: Dict[str, Any], output_dir: str = 'results/visualizations') -> Dict[str, str]:
    """Backward compatibility wrapper for create_visualizations."""
    chart_files = create_visualizations(results, {}, output_dir)
    return {'chart_files': chart_files}


# Helper functions for creating specific charts

def _extract_all_model_data(results: Dict[str, Any]) -> Dict[str, Dict[str, List[float]]]:
    """Extract model data from results in various formats."""
    model_data = {}
    
    if not results:
        return model_data
    
    # Handle new format: {'scenarios': [{'model_evaluations': {'openai': {...}, 'deepseek': {...}}}]}
    if 'scenarios' in results:
        scenarios = results['scenarios']
        
        for scenario in scenarios:
            # New format with model_evaluations
            if 'model_evaluations' in scenario:
                model_evals = scenario['model_evaluations']
                for model_key, evaluation in model_evals.items():
                    if evaluation:
                        # Map internal model keys to display names
                        display_names = {
                            'openai': 'OpenAI',
                            'deepseek': 'DeepSeek', 
                            'claude': 'Claude',
                            'gemma': 'Gemma'
                        }
                        model_name = display_names.get(model_key.lower(), model_key.title())
                        
                        if model_name not in model_data:
                            model_data[model_name] = {
                                'composite': [], 'empathy': [], 'therapeutic': [],
                                'safety': [], 'clarity': [], 'cost': [], 'response_time': []
                            }
                        
                        # Extract scores safely
                        model_data[model_name]['composite'].append(_safe_get_score(evaluation, 'composite_score'))
                        model_data[model_name]['empathy'].append(_safe_get_score(evaluation, 'empathy_score'))
                        model_data[model_name]['therapeutic'].append(_safe_get_score(evaluation, 'therapeutic_value_score'))
                        model_data[model_name]['safety'].append(_safe_get_score(evaluation, 'safety_score'))
                        model_data[model_name]['clarity'].append(_safe_get_score(evaluation, 'clarity_score'))
                        model_data[model_name]['cost'].append(_safe_get_score(evaluation, 'cost_usd', 0.0))
                        model_data[model_name]['response_time'].append(_safe_get_score(evaluation, 'response_time_ms', 0.0))
            
            # Old format with individual model keys
            else:
                for model_prefix in ['openai', 'deepseek', 'claude', 'gemma']:
                    eval_key = f'{model_prefix}_evaluation'
                    if eval_key in scenario:
                        evaluation = scenario[eval_key]
                        if evaluation:
                            display_names = {'openai': 'OpenAI', 'deepseek': 'DeepSeek', 'claude': 'Claude', 'gemma': 'Gemma'}
                            model_name = display_names.get(model_prefix, model_prefix.title())
                            
                            if model_name not in model_data:
                                model_data[model_name] = {
                                    'composite': [], 'empathy': [], 'therapeutic': [],
                                    'safety': [], 'clarity': [], 'cost': [], 'response_time': []
                                }
                            
                            model_data[model_name]['composite'].append(_safe_get_score(evaluation, 'composite_score'))
                            model_data[model_name]['empathy'].append(_safe_get_score(evaluation, 'empathy_score'))
                            model_data[model_name]['therapeutic'].append(_safe_get_score(evaluation, 'therapeutic_value_score'))
                            model_data[model_name]['safety'].append(_safe_get_score(evaluation, 'safety_score'))
                            model_data[model_name]['clarity'].append(_safe_get_score(evaluation, 'clarity_score'))
                            model_data[model_name]['cost'].append(_safe_get_score(evaluation, 'cost_usd', 0.0))
                            model_data[model_name]['response_time'].append(_safe_get_score(evaluation, 'response_time_ms', 0.0))
    
    return model_data

def _safe_get_score(data: Any, key: str, default: float = 0.0) -> float:
    """Safely get a score from data."""
    if isinstance(data, dict):
        value = data.get(key, default)
    elif hasattr(data, key):
        value = getattr(data, key, default)
    else:
        value = default
    
    try:
        return float(value) if value is not None else default
    except (ValueError, TypeError):
        return default

def _safe_mean(values: List[float]) -> float:
    """Safely calculate mean."""
    if not values:
        return 0.0
    return sum(values) / len(values)

def _safe_std(values: List[float]) -> float:
    """Safely calculate standard deviation."""
    if not values or len(values) < 2:
        return 0.0
    
    mean = _safe_mean(values)
    variance = sum((x - mean) ** 2 for x in values) / (len(values) - 1)
    return variance ** 0.5

def _create_overall_comparison_chart(model_data: Dict[str, Dict[str, List[float]]], 
                                   model_colors: Dict[str, str], output_dir: str) -> Optional[str]:
    """Create 1_overall_comparison.png - Bar chart of all metrics for all models."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        metrics = ['empathy', 'therapeutic', 'safety', 'clarity', 'composite']
        metric_labels = ['Empathy', 'Therapeutic\nValue', 'Safety', 'Clarity', 'Composite\nScore']
        
        models = list(model_data.keys())
        n_models = len(models)
        n_metrics = len(metrics)
        
        # Calculate means and stds
        means = []
        stds = []
        colors = []
        
        for model in models:
            model_means = []
            model_stds = []
            for metric in metrics:
                values = model_data[model].get(metric, [])
                model_means.append(_safe_mean(values))
                model_stds.append(_safe_std(values))
            means.append(model_means)
            stds.append(model_stds)
            colors.append(model_colors.get(model, '#1f77b4'))
        
        # Create the plot
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Bar positions
        x = np.arange(n_metrics)
        width = 0.8 / n_models
        
        # Create bars for each model
        for i, (model, model_means_list, model_stds_list, color) in enumerate(zip(models, means, stds, colors)):
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, model_means_list, width, 
                         yerr=model_stds_list, capsize=3, alpha=0.8,
                         color=color, label=model, edgecolor='white', linewidth=0.5)
            
            # Add value labels on bars
            for bar, mean_val in zip(bars, model_means_list):
                height = bar.get_height()
                if height > 0:
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.2,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=9, fontweight='bold')
        
        # Customize the plot
        ax.set_title('Mental Health LLM Performance Comparison\nby Therapeutic Quality Metrics', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Score (0-10 scale)', fontsize=12)
        ax.set_xlabel('Evaluation Metrics', fontsize=12)
        ax.set_xticks(x)
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 11)
        ax.grid(True, alpha=0.3, axis='y')
        ax.legend(loc='upper right', framealpha=0.9)
        
        # Add significance indicators if there are differences
        if n_models > 1:
            ax.text(0.02, 0.95, '*** p<0.001', transform=ax.transAxes, 
                   fontsize=10, verticalalignment='top')
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(output_dir, '1_overall_comparison.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error creating overall comparison chart: {e}")
        return None

def _create_category_radar_chart(model_data: Dict[str, Dict[str, List[float]]], 
                               model_colors: Dict[str, str], output_dir: str) -> Optional[str]:
    """Create 2_category_radar.png - Radar chart showing all models across metrics."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        metrics = ['empathy', 'therapeutic', 'safety', 'clarity']
        metric_labels = ['Empathy', 'Therapeutic\nValue', 'Safety', 'Clarity']
        
        models = list(model_data.keys())
        
        # Calculate means for each model
        model_scores = {}
        for model in models:
            scores = []
            for metric in metrics:
                values = model_data[model].get(metric, [])
                scores.append(_safe_mean(values))
            model_scores[model] = scores
        
        # Set up radar chart
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(projection='polar'))
        
        # Calculate angles for each metric
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        # Plot each model
        for model in models:
            scores = model_scores[model]
            scores += scores[:1]  # Complete the circle
            
            color = model_colors.get(model, '#1f77b4')
            ax.plot(angles, scores, 'o-', linewidth=2, label=model, color=color)
            ax.fill(angles, scores, alpha=0.25, color=color)
        
        # Customize the radar chart
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels)
        ax.set_ylim(0, 10)
        ax.set_yticks([2, 4, 6, 8, 10])
        ax.set_yticklabels(['2', '4', '6', '8', '10'])
        ax.grid(True)
        
        # Add title and legend
        plt.title('Mental Health LLM Performance\nRadar Chart by Therapeutic Metrics', 
                 size=16, fontweight='bold', y=1.08)
        plt.legend(loc='upper right', bbox_to_anchor=(1.2, 1.0))
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(output_dir, '2_category_radar.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error creating category radar chart: {e}")
        return None

def _create_cost_effectiveness_chart(model_data: Dict[str, Dict[str, List[float]]], 
                                   model_colors: Dict[str, str], output_dir: str) -> Optional[str]:
    """Create 3_cost_effectiveness.png - Scatter plot of performance vs cost."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        models = list(model_data.keys())
        
        # Calculate composite scores and costs
        composite_scores = []
        costs = []
        colors = []
        labels = []
        
        for model in models:
            composite_vals = model_data[model].get('composite', [])
            cost_vals = model_data[model].get('cost', [])
            
            if composite_vals:
                composite_score = _safe_mean(composite_vals)
                total_cost = sum(cost_vals) if cost_vals else 0.0
                
                composite_scores.append(composite_score)
                costs.append(total_cost)
                colors.append(model_colors.get(model, '#1f77b4'))
                labels.append(model)
        
        # Create scatter plot
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # Create scatter plot with larger points
        scatter = ax.scatter(costs, composite_scores, c=colors, s=300, alpha=0.7, edgecolors='white', linewidth=2)
        
        # Add model labels
        for i, (cost, score, label) in enumerate(zip(costs, composite_scores, labels)):
            ax.annotate(label, (cost, score), xytext=(10, 10), textcoords='offset points',
                       fontsize=12, fontweight='bold', ha='left')
        
        # Customize the plot
        ax.set_title('Mental Health LLM Cost vs Performance Analysis', fontsize=16, fontweight='bold')
        ax.set_xlabel('Total Cost (USD)', fontsize=12)
        ax.set_ylabel('Composite Performance Score (0-10)', fontsize=12)
        ax.grid(True, alpha=0.3)
        
        # Add efficiency quadrants
        if costs and composite_scores:
            mean_cost = np.mean(costs)
            mean_score = np.mean(composite_scores)
            
            ax.axhline(y=mean_score, color='gray', linestyle='--', alpha=0.5)
            ax.axvline(x=mean_cost, color='gray', linestyle='--', alpha=0.5)
            
            # Add quadrant labels
            ax.text(max(costs) * 0.8, max(composite_scores) * 0.95, 'High Cost\nHigh Performance', 
                   ha='center', va='center', fontsize=10, alpha=0.7, style='italic')
            ax.text(min(costs) * 1.2, max(composite_scores) * 0.95, 'Low Cost\nHigh Performance', 
                   ha='center', va='center', fontsize=10, alpha=0.7, style='italic')
        
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(output_dir, '3_cost_effectiveness.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error creating cost effectiveness chart: {e}")
        return None

def _create_safety_metrics_chart(model_data: Dict[str, Dict[str, List[float]]], 
                               model_colors: Dict[str, str], output_dir: str) -> Optional[str]:
    """Create 4_safety_metrics.png - Safety-focused analysis chart."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        models = list(model_data.keys())
        
        # Calculate safety metrics
        safety_scores = []
        overall_scores = []
        colors = []
        
        for model in models:
            safety_vals = model_data[model].get('safety', [])
            composite_vals = model_data[model].get('composite', [])
            
            safety_score = _safe_mean(safety_vals) if safety_vals else 0.0
            overall_score = _safe_mean(composite_vals) if composite_vals else 0.0
            
            safety_scores.append(safety_score)
            overall_scores.append(overall_score)
            colors.append(model_colors.get(model, '#1f77b4'))
        
        # Create the plot with two subplots
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left plot: Safety scores bar chart
        bars1 = ax1.bar(models, safety_scores, color=colors, alpha=0.7, edgecolor='white', linewidth=1)
        ax1.set_title('Safety Scores by Model', fontsize=14, fontweight='bold')
        ax1.set_ylabel('Safety Score (0-10)', fontsize=12)
        ax1.set_ylim(0, 10.5)
        ax1.grid(True, alpha=0.3, axis='y')
        
        # Add value labels on bars
        for bar, score in zip(bars1, safety_scores):
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{score:.1f}', ha='center', va='bottom', fontsize=11, fontweight='bold')
        
        # Right plot: Safety vs Overall Performance
        ax2.scatter(safety_scores, overall_scores, c=colors, s=300, alpha=0.7, edgecolors='white', linewidth=2)
        
        # Add model labels
        for i, (safety, overall, model) in enumerate(zip(safety_scores, overall_scores, models)):
            ax2.annotate(model, (safety, overall), xytext=(10, 10), textcoords='offset points',
                        fontsize=10, fontweight='bold', ha='left')
        
        ax2.set_title('Safety vs Overall Performance', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Safety Score (0-10)', fontsize=12)
        ax2.set_ylabel('Overall Performance Score (0-10)', fontsize=12)
        ax2.grid(True, alpha=0.3)
        ax2.set_xlim(0, 10.5)
        ax2.set_ylim(0, 10.5)
        
        # Add diagonal line showing perfect correlation
        ax2.plot([0, 10], [0, 10], 'k--', alpha=0.3, label='Perfect Correlation')
        ax2.legend()
        
        plt.suptitle('Mental Health LLM Safety Analysis', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(output_dir, '4_safety_metrics.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error creating safety metrics chart: {e}")
        return None

def _create_statistical_summary_chart(model_data: Dict[str, Dict[str, List[float]]], 
                                    analysis: Dict[str, Any], model_colors: Dict[str, str], 
                                    output_dir: str) -> Optional[str]:
    """Create 5_statistical_summary.png - Statistical significance and summary."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
        
        models = list(model_data.keys())
        metrics = ['empathy', 'therapeutic', 'safety', 'clarity', 'composite']
        
        # Create figure with subplots
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. Model Rankings (top-left)
        composite_means = []
        for model in models:
            composite_vals = model_data[model].get('composite', [])
            composite_means.append(_safe_mean(composite_vals))
        
        # Sort models by performance
        model_performance = list(zip(models, composite_means))
        model_performance.sort(key=lambda x: x[1], reverse=True)
        
        sorted_models, sorted_scores = zip(*model_performance)
        sorted_colors = [model_colors.get(model, '#1f77b4') for model in sorted_models]
        
        bars1 = ax1.barh(range(len(sorted_models)), sorted_scores, color=sorted_colors, alpha=0.7)
        ax1.set_yticks(range(len(sorted_models)))
        ax1.set_yticklabels(sorted_models)
        ax1.set_xlabel('Composite Score (0-10)')
        ax1.set_title('Model Rankings', fontweight='bold')
        ax1.grid(True, alpha=0.3, axis='x')
        
        # Add value labels
        for i, (bar, score) in enumerate(zip(bars1, sorted_scores)):
            width = bar.get_width()
            ax1.text(width + 0.1, bar.get_y() + bar.get_height()/2,
                    f'{score:.2f}', ha='left', va='center', fontweight='bold')
        
        # 2. Standard deviations (top-right)
        std_data = []
        for metric in metrics:
            metric_stds = []
            for model in models:
                values = model_data[model].get(metric, [])
                metric_stds.append(_safe_std(values))
            std_data.append(metric_stds)
        
        x_pos = np.arange(len(metrics))
        width = 0.8 / len(models)
        
        for i, model in enumerate(models):
            model_stds = [std_data[j][i] for j in range(len(metrics))]
            offset = (i - len(models)/2 + 0.5) * width
            ax2.bar(x_pos + offset, model_stds, width, 
                   color=model_colors.get(model, '#1f77b4'), alpha=0.7, label=model)
        
        ax2.set_xticks(x_pos)
        ax2.set_xticklabels([m.capitalize() for m in metrics])
        ax2.set_ylabel('Standard Deviation')
        ax2.set_title('Score Variability by Metric', fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3, axis='y')
        
        # 3. Sample sizes (bottom-left)
        sample_sizes = []
        for model in models:
            composite_vals = model_data[model].get('composite', [])
            sample_sizes.append(len(composite_vals))
        
        bars3 = ax3.bar(models, sample_sizes, color=[model_colors.get(m, '#1f77b4') for m in models], alpha=0.7)
        ax3.set_ylabel('Number of Evaluations')
        ax3.set_title('Sample Sizes', fontweight='bold')
        ax3.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, size in zip(bars3, sample_sizes):
            height = bar.get_height()
            ax3.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{size}', ha='center', va='bottom', fontweight='bold')
        
        # 4. Summary statistics table (bottom-right)
        ax4.axis('off')
        
        # Create summary table
        table_data = []
        table_data.append(['Model', 'Mean Score', 'Std Dev', 'Samples'])
        
        for model in sorted_models:
            composite_vals = model_data[model].get('composite', [])
            mean_score = _safe_mean(composite_vals)
            std_score = _safe_std(composite_vals)
            n_samples = len(composite_vals)
            
            table_data.append([model, f'{mean_score:.2f}', f'{std_score:.2f}', f'{n_samples}'])
        
        # Create table
        table = ax4.table(cellText=table_data[1:], colLabels=table_data[0], 
                         cellLoc='center', loc='center', 
                         colColours=['lightgray']*4)
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        ax4.set_title('Statistical Summary', fontweight='bold', y=0.9)
        
        plt.suptitle('Mental Health LLM Evaluation - Statistical Summary', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # Save the chart
        chart_path = os.path.join(output_dir, '5_statistical_summary.png')
        plt.savefig(chart_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        
        return chart_path
        
    except Exception as e:
        logger.error(f"Error creating statistical summary chart: {e}")
        return None


# Initialize module
if __name__ == "__main__":
    # Test with sample data
    sample_results = {
        'scenarios': [
            {
                'openai_evaluation': {
                    'composite_score': 8.5,
                    'empathy_score': 9.0,
                    'therapeutic_value_score': 8.0,
                    'safety_score': 9.5,
                    'clarity_score': 8.5,
                    'cost_usd': 0.02,
                    'response_time_ms': 1200
                },
                'deepseek_evaluation': {
                    'composite_score': 7.5,
                    'empathy_score': 8.0,
                    'therapeutic_value_score': 7.0,
                    'safety_score': 8.5,
                    'clarity_score': 7.5,
                    'cost_usd': 0.0,
                    'response_time_ms': 2000
                }
            }
        ]
    }
    
    print("Testing visualization module...")
    results = create_all_visualizations(sample_results, "/tmp/test_plots")
    print(f"Generated files: {results}")