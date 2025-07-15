"""
Visualization tools for mental health LLM evaluation results.

This module provides comprehensive visualization capabilities for displaying
evaluation results, comparisons, and insights.
"""

from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
import logging

# from ..evaluation.composite_scorer import CompositeScore
from .statistical_analysis import StatisticalResults

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization generation."""
    
    style: str = "seaborn-v0_8"
    color_palette: str = "husl"
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    save_format: str = "png"
    interactive: bool = True
    show_confidence_intervals: bool = True
    include_statistical_annotations: bool = True


class ResultsVisualizer:
    """Comprehensive visualization tool for evaluation results."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize results visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set up matplotlib style
        plt.style.use(self.config.style)
        sns.set_palette(self.config.color_palette)
        
        # Default colors for models
        self.model_colors = [
            "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
            "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"
        ]
    
    def create_comprehensive_dashboard(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        statistical_results: Optional[StatisticalResults] = None,
        results_dir: str = "./visualizations"
    ) -> Dict[str, str]:
        """
        Create a comprehensive visualization dashboard.
        
        Args:
            results: Model evaluation results
            statistical_results: Optional statistical analysis results
            results_dir: Directory to save visualizations
            
        Returns:
            Dictionary mapping visualization names to file paths
        """
        import os
        os.makedirs(results_dir, exist_ok=True)
        
        self.logger.info("Creating comprehensive visualization dashboard")
        
        generated_files = {}
        
        # Overall comparison
        fig_path = self.create_overall_comparison(results, results_dir)
        if fig_path:
            generated_files["overall_comparison"] = fig_path
        
        # Score distributions
        fig_path = self.create_score_distributions(results, results_dir)
        if fig_path:
            generated_files["score_distributions"] = fig_path
        
        # Performance radar charts
        fig_path = self.create_radar_charts(results, results_dir)
        if fig_path:
            generated_files["radar_charts"] = fig_path
        
        # Technical metrics
        fig_path = self.create_technical_metrics_plot(results, results_dir)
        if fig_path:
            generated_files["technical_metrics"] = fig_path
        
        # Therapeutic quality heatmap
        fig_path = self.create_therapeutic_heatmap(results, results_dir)
        if fig_path:
            generated_files["therapeutic_heatmap"] = fig_path
        
        # Patient experience comparison
        fig_path = self.create_patient_experience_plot(results, results_dir)
        if fig_path:
            generated_files["patient_experience"] = fig_path
        
        # Statistical significance plot
        if statistical_results:
            fig_path = self.create_statistical_significance_plot(
                statistical_results, results_dir
            )
            if fig_path:
                generated_files["statistical_significance"] = fig_path
        
        # Interactive dashboard
        if self.config.interactive:
            fig_path = self.create_interactive_dashboard(results, results_dir)
            if fig_path:
                generated_files["interactive_dashboard"] = fig_path
        
        self.logger.info(f"Generated {len(generated_files)} visualizations")
        return generated_files
    
    def create_overall_comparison(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create overall model comparison chart."""
        try:
            # Prepare data
            data = []
            for model_name, scores in results.items():
                for score in scores:
                    data.append({
                        "Model": model_name,
                        "Overall": score.overall_score,
                        "Technical": score.technical_score,
                        "Therapeutic": score.therapeutic_score,
                        "Patient": score.patient_score
                    })
            
            df = pd.DataFrame(data)
            
            # Create comparison plot
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle("Model Performance Comparison", fontsize=16, fontweight='bold')
            
            metrics = ["Overall", "Technical", "Therapeutic", "Patient"]
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                
                # Box plot with individual points
                sns.boxplot(data=df, x="Model", y=metric, ax=ax)
                sns.stripplot(data=df, x="Model", y=metric, ax=ax, 
                            size=4, alpha=0.7, color='black')
                
                ax.set_title(f"{metric} Score")
                ax.set_ylabel("Score (0-100)")
                ax.tick_params(axis='x', rotation=45)
                
                # Add mean values as text
                for j, model in enumerate(df["Model"].unique()):
                    model_data = df[df["Model"] == model][metric]
                    mean_val = model_data.mean()
                    ax.text(j, mean_val + 2, f"{mean_val:.1f}", 
                           ha='center', va='bottom', fontweight='bold')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"overall_comparison.{self.config.save_format}"
            filepath = f"{results_dir}/{filename}"
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating overall comparison: {e}")
            return None
    
    def create_score_distributions(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create score distribution plots."""
        try:
            # Prepare data
            data = []
            for model_name, scores in results.items():
                overall_scores = [score.overall_score for score in scores]
                data.extend([(model_name, score) for score in overall_scores])
            
            df = pd.DataFrame(data, columns=["Model", "Overall_Score"])
            
            # Create distribution plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
            fig.suptitle("Score Distributions", fontsize=16, fontweight='bold')
            
            # Histogram
            for i, model in enumerate(df["Model"].unique()):
                model_scores = df[df["Model"] == model]["Overall_Score"]
                ax1.hist(model_scores, alpha=0.7, label=model, 
                        color=self.model_colors[i % len(self.model_colors)],
                        bins=15)
            
            ax1.set_xlabel("Overall Score")
            ax1.set_ylabel("Frequency")
            ax1.set_title("Score Histograms")
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Violin plot
            sns.violinplot(data=df, x="Model", y="Overall_Score", ax=ax2)
            ax2.set_title("Score Distributions (Violin Plot)")
            ax2.set_ylabel("Overall Score")
            ax2.tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"score_distributions.{self.config.save_format}"
            filepath = f"{results_dir}/{filename}"
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating score distributions: {e}")
            return None
    
    def create_radar_charts(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create radar charts for model comparison."""
        try:
            # Calculate mean scores for each model
            model_means = {}
            for model_name, scores in results.items():
                if scores:
                    mean_scores = {
                        "Technical": np.mean([s.technical_score for s in scores]),
                        "Therapeutic": np.mean([s.therapeutic_score for s in scores]),
                        "Patient": np.mean([s.patient_score for s in scores]),
                        "Response Time": np.mean([100 - min(100, s.technical_details.response_time_ms / 50) for s in scores]),
                        "Safety": np.mean([s.therapeutic_details.safety_score for s in scores]),
                        "Empathy": np.mean([s.therapeutic_details.empathy_score for s in scores]),
                        "Trust": np.mean([s.patient_details.trust_score for s in scores]),
                        "Engagement": np.mean([s.patient_details.engagement_score for s in scores])
                    }
                    model_means[model_name] = mean_scores
            
            # Create radar chart
            categories = list(next(iter(model_means.values())).keys())
            
            fig = go.Figure()
            
            for i, (model_name, scores) in enumerate(model_means.items()):
                values = list(scores.values())
                values.append(values[0])  # Close the radar chart
                
                fig.add_trace(go.Scatterpolar(
                    r=values,
                    theta=categories + [categories[0]],
                    fill='toself',
                    name=model_name,
                    line_color=self.model_colors[i % len(self.model_colors)]
                ))
            
            fig.update_layout(
                polar=dict(
                    radialaxis=dict(
                        visible=True,
                        range=[0, 100]
                    )),
                showlegend=True,
                title="Model Performance Radar Chart",
                title_x=0.5
            )
            
            # Save plot
            filename = f"radar_charts.html"
            filepath = f"{results_dir}/{filename}"
            fig.write_html(filepath)
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating radar charts: {e}")
            return None
    
    def create_technical_metrics_plot(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create technical metrics comparison plot."""
        try:
            # Prepare data
            data = []
            for model_name, scores in results.items():
                for score in scores:
                    tech = score.technical_details
                    data.append({
                        "Model": model_name,
                        "Response Time (ms)": tech.response_time_ms,
                        "Throughput (RPS)": tech.throughput_rps,
                        "Success Rate (%)": tech.success_rate * 100,
                        "CPU Usage (%)": tech.cpu_usage_percent
                    })
            
            df = pd.DataFrame(data)
            
            # Create subplots
            fig, axes = plt.subplots(2, 2, figsize=self.config.figure_size)
            fig.suptitle("Technical Performance Metrics", fontsize=16, fontweight='bold')
            
            metrics = ["Response Time (ms)", "Throughput (RPS)", "Success Rate (%)", "CPU Usage (%)"]
            
            for i, metric in enumerate(metrics):
                ax = axes[i // 2, i % 2]
                
                # Bar plot with error bars
                means = df.groupby("Model")[metric].mean()
                stds = df.groupby("Model")[metric].std()
                
                bars = ax.bar(means.index, means.values, 
                             yerr=stds.values, capsize=5,
                             color=[self.model_colors[j % len(self.model_colors)] 
                                   for j in range(len(means))])
                
                ax.set_title(metric)
                ax.set_ylabel(metric)
                ax.tick_params(axis='x', rotation=45)
                
                # Add value labels on bars
                for bar, mean_val in zip(bars, means.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.01 * height,
                           f'{mean_val:.1f}', ha='center', va='bottom')
            
            plt.tight_layout()
            
            # Save plot
            filename = f"technical_metrics.{self.config.save_format}"
            filepath = f"{results_dir}/{filename}"
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating technical metrics plot: {e}")
            return None
    
    def create_therapeutic_heatmap(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create therapeutic quality heatmap."""
        try:
            # Prepare data
            therapeutic_data = {}
            for model_name, scores in results.items():
                if scores:
                    therapeutic_data[model_name] = {
                        "Empathy": np.mean([s.therapeutic_details.empathy_score for s in scores]),
                        "Safety": np.mean([s.therapeutic_details.safety_score for s in scores]),
                        "Coherence": np.mean([s.therapeutic_details.coherence_score for s in scores]),
                        "Boundaries": np.mean([s.therapeutic_details.boundaries_score for s in scores])
                    }
            
            df = pd.DataFrame(therapeutic_data).T
            
            # Create heatmap
            plt.figure(figsize=self.config.figure_size)
            
            sns.heatmap(df, annot=True, fmt='.1f', cmap='RdYlGn',
                       center=75, vmin=0, vmax=100,
                       cbar_kws={'label': 'Score (0-100)'})
            
            plt.title("Therapeutic Quality Metrics Heatmap", fontsize=16, fontweight='bold')
            plt.ylabel("Models")
            plt.xlabel("Therapeutic Metrics")
            plt.xticks(rotation=45)
            plt.yticks(rotation=0)
            
            # Save plot
            filename = f"therapeutic_heatmap.{self.config.save_format}"
            filepath = f"{results_dir}/{filename}"
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating therapeutic heatmap: {e}")
            return None
    
    def create_patient_experience_plot(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create patient experience comparison plot."""
        try:
            # Prepare data
            data = []
            for model_name, scores in results.items():
                for score in scores:
                    patient = score.patient_details
                    data.append({
                        "Model": model_name,
                        "Satisfaction": patient.satisfaction_score,
                        "Engagement": patient.engagement_score,
                        "Trust": patient.trust_score,
                        "Accessibility": patient.accessibility_score
                    })
            
            df = pd.DataFrame(data)
            
            # Create grouped bar chart
            fig, ax = plt.subplots(figsize=self.config.figure_size)
            
            x = np.arange(len(df["Model"].unique()))
            width = 0.2
            
            metrics = ["Satisfaction", "Engagement", "Trust", "Accessibility"]
            
            for i, metric in enumerate(metrics):
                means = df.groupby("Model")[metric].mean()
                bars = ax.bar(x + i * width, means.values, width, 
                             label=metric, alpha=0.8,
                             color=self.model_colors[i % len(self.model_colors)])
                
                # Add value labels
                for bar, mean_val in zip(bars, means.values):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                           f'{mean_val:.1f}', ha='center', va='bottom', fontsize=8)
            
            ax.set_xlabel("Models")
            ax.set_ylabel("Score (0-100)")
            ax.set_title("Patient Experience Metrics", fontsize=16, fontweight='bold')
            ax.set_xticks(x + width * 1.5)
            ax.set_xticklabels(df["Model"].unique())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save plot
            filename = f"patient_experience.{self.config.save_format}"
            filepath = f"{results_dir}/{filename}"
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating patient experience plot: {e}")
            return None
    
    def create_statistical_significance_plot(
        self,
        statistical_results: StatisticalResults,
        results_dir: str
    ) -> Optional[str]:
        """Create statistical significance visualization."""
        try:
            # Prepare significance data
            significance_data = []
            for metric, test in statistical_results.significance_tests.items():
                significance_data.append({
                    "Metric": metric.replace("_", " ").title(),
                    "P-Value": test.get("p_value", 1.0),
                    "Significant": test.get("significant", False),
                    "Test": test.get("test_type", "unknown")
                })
            
            df = pd.DataFrame(significance_data)
            
            if df.empty:
                return None
            
            # Create significance plot
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=self.config.figure_size)
            fig.suptitle("Statistical Significance Analysis", fontsize=16, fontweight='bold')
            
            # P-values bar chart
            colors = ['red' if sig else 'blue' for sig in df["Significant"]]
            bars = ax1.bar(range(len(df)), df["P-Value"], color=colors, alpha=0.7)
            ax1.axhline(y=0.05, color='red', linestyle='--', label='α = 0.05')
            ax1.set_xlabel("Metrics")
            ax1.set_ylabel("P-Value")
            ax1.set_title("P-Values by Metric")
            ax1.set_xticks(range(len(df)))
            ax1.set_xticklabels(df["Metric"], rotation=45, ha='right')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Significance summary
            sig_count = df["Significant"].sum()
            total_count = len(df)
            
            labels = ['Significant', 'Not Significant']
            sizes = [sig_count, total_count - sig_count]
            colors = ['lightcoral', 'lightblue']
            
            ax2.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
            ax2.set_title("Significance Summary")
            
            plt.tight_layout()
            
            # Save plot
            filename = f"statistical_significance.{self.config.save_format}"
            filepath = f"{results_dir}/{filename}"
            plt.savefig(filepath, dpi=self.config.dpi, bbox_inches='tight')
            plt.close()
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating statistical significance plot: {e}")
            return None
    
    def create_interactive_dashboard(
        self,
        results: Dict[str, List[Dict[str, Any]]],
        results_dir: str
    ) -> Optional[str]:
        """Create interactive dashboard using Plotly."""
        try:
            # Prepare comprehensive data
            data = []
            for model_name, scores in results.items():
                for i, score in enumerate(scores):
                    data.append({
                        "Model": model_name,
                        "Sample": i,
                        "Overall": score.overall_score,
                        "Technical": score.technical_score,
                        "Therapeutic": score.therapeutic_score,
                        "Patient": score.patient_score,
                        "Response_Time": score.technical_details.response_time_ms,
                        "Throughput": score.technical_details.throughput_rps,
                        "Empathy": score.therapeutic_details.empathy_score,
                        "Safety": score.therapeutic_details.safety_score,
                        "Trust": score.patient_details.trust_score,
                        "Satisfaction": score.patient_details.satisfaction_score
                    })
            
            df = pd.DataFrame(data)
            
            # Create interactive dashboard
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=("Overall Scores", "Technical vs Therapeutic", 
                              "Patient Experience", "Performance Distribution"),
                specs=[[{"type": "scatter"}, {"type": "scatter"}],
                       [{"type": "bar"}, {"type": "box"}]]
            )
            
            # Plot 1: Overall scores scatter
            for model in df["Model"].unique():
                model_data = df[df["Model"] == model]
                fig.add_trace(
                    go.Scatter(
                        x=model_data["Sample"],
                        y=model_data["Overall"],
                        mode='markers+lines',
                        name=model,
                        showlegend=True
                    ),
                    row=1, col=1
                )
            
            # Plot 2: Technical vs Therapeutic scatter
            for model in df["Model"].unique():
                model_data = df[df["Model"] == model]
                fig.add_trace(
                    go.Scatter(
                        x=model_data["Technical"],
                        y=model_data["Therapeutic"],
                        mode='markers',
                        name=model,
                        showlegend=False
                    ),
                    row=1, col=2
                )
            
            # Plot 3: Patient experience bars
            patient_means = df.groupby("Model")[["Trust", "Satisfaction", "Patient"]].mean()
            for metric in ["Trust", "Satisfaction", "Patient"]:
                fig.add_trace(
                    go.Bar(
                        x=patient_means.index,
                        y=patient_means[metric],
                        name=metric,
                        showlegend=False
                    ),
                    row=2, col=1
                )
            
            # Plot 4: Overall score distribution
            for model in df["Model"].unique():
                model_data = df[df["Model"] == model]
                fig.add_trace(
                    go.Box(
                        y=model_data["Overall"],
                        name=model,
                        showlegend=False
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title="Mental Health LLM Evaluation Dashboard",
                height=800,
                showlegend=True
            )
            
            # Save interactive plot
            filename = f"interactive_dashboard.html"
            filepath = f"{results_dir}/{filename}"
            fig.write_html(filepath)
            
            return filepath
        
        except Exception as e:
            self.logger.error(f"Error creating interactive dashboard: {e}")
            return None