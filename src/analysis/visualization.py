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
        self.color_palette = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]
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
                            model_name = model_prefix.upper()
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


def create_all_visualizations(results: Dict[str, Any], output_dir: str = 'results/plots') -> Dict[str, str]:
    """
    Create all available visualizations for the evaluation results.
    
    This function provides graceful degradation - it will create text summaries
    even if matplotlib is not available, and will create charts if it is.
    
    Args:
        results: Dictionary containing evaluation results
        output_dir: Directory to save visualizations
        
    Returns:
        Dictionary mapping visualization names to file paths
    """
    logger.info("Creating visualizations...")
    
    # Create visualizer
    visualizer = SafeVisualizer()
    
    # Track generated files
    generated_files = {}
    
    # Always create text summary
    try:
        summary_path = visualizer.create_text_summary(results, output_dir)
        if summary_path:
            generated_files['text_summary'] = summary_path
            logger.info(f"Created text summary: {summary_path}")
    except Exception as e:
        logger.error(f"Failed to create text summary: {e}")
    
    # Create charts if matplotlib is available
    if HAS_MATPLOTLIB:
        logger.info("Matplotlib available - creating charts...")
        
        # Simple bar chart
        try:
            chart_path = visualizer.create_simple_bar_chart(results, output_dir)
            if chart_path:
                generated_files['bar_chart'] = chart_path
                logger.info(f"Created bar chart: {chart_path}")
        except Exception as e:
            logger.error(f"Failed to create bar chart: {e}")
        
        # Metric comparison
        try:
            chart_path = visualizer.create_metric_comparison(results, output_dir)
            if chart_path:
                generated_files['metric_comparison'] = chart_path
                logger.info(f"Created metric comparison: {chart_path}")
        except Exception as e:
            logger.error(f"Failed to create metric comparison: {e}")
        
        # Cost analysis
        try:
            chart_path = visualizer.create_cost_analysis_chart(results, output_dir)
            if chart_path:
                generated_files['cost_analysis'] = chart_path
                logger.info(f"Created cost analysis: {chart_path}")
        except Exception as e:
            logger.error(f"Failed to create cost analysis: {e}")
    
    else:
        logger.warning("Matplotlib not available - skipping chart generation")
    
    # Create JSON summary of generated files
    try:
        summary_info = {
            'generated_at': datetime.now().isoformat(),
            'has_matplotlib': HAS_MATPLOTLIB,
            'has_plotly': HAS_PLOTLY,
            'has_seaborn': HAS_SEABORN,
            'has_pandas': HAS_PANDAS,
            'has_numpy': HAS_NUMPY,
            'files_generated': generated_files
        }
        
        summary_json_path = os.path.join(output_dir, 'visualization_summary.json')
        with open(summary_json_path, 'w') as f:
            json.dump(summary_info, f, indent=2)
        
        generated_files['summary_json'] = summary_json_path
        
    except Exception as e:
        logger.error(f"Failed to create JSON summary: {e}")
    
    logger.info(f"Generated {len(generated_files)} visualization files")
    
    return generated_files


# Backward compatibility - provide the expected interface
def create_presentation_slides(*args, **kwargs):
    """Stub function for presentation slides creation."""
    return []


def create_comprehensive_dashboard(results, statistical_results=None, results_dir="./visualizations"):
    """Backward compatibility function."""
    return create_all_visualizations(results, results_dir)


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