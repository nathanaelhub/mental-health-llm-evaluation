"""
Advanced Visualization Module for LLM Performance Analysis

This module provides comprehensive visualization capabilities for statistical analysis
including box plots, radar charts, heatmaps, time series analysis, and publication-quality figures
with statistical significance indicators.
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px
import plotly.figure_factory as ff
from plotly.subplots import make_subplots
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Any, Tuple, Union
from dataclasses import dataclass
import logging
from pathlib import Path
import warnings
from scipy import stats
from matplotlib.patches import Rectangle
import math

from .statistical_analysis import StatisticalResults, DescriptiveStatistics, StatisticalTestResult

logger = logging.getLogger(__name__)

# Set style defaults
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("husl")


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings."""
    
    # Figure settings
    figure_size: Tuple[int, int] = (12, 8)
    dpi: int = 300
    font_size: int = 12
    title_size: int = 14
    
    # Color settings
    color_palette: str = "Set2"
    significance_color: str = "red"
    non_significance_color: str = "gray"
    
    # Statistical settings
    alpha_level: float = 0.05
    show_confidence_intervals: bool = True
    show_effect_sizes: bool = True
    
    # Output settings
    save_format: str = "png"
    save_transparent: bool = False
    save_bbox_inches: str = "tight"


class AdvancedVisualizer:
    """
    Advanced visualization system for LLM performance analysis with publication-quality output.
    
    Provides comprehensive plotting capabilities including statistical significance indicators,
    effect size visualizations, and interactive plots.
    """
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Initialize advanced visualizer.
        
        Args:
            config: Visualization configuration
        """
        self.config = config or VisualizationConfig()
        self.logger = logging.getLogger(__name__)
        
        # Set matplotlib defaults
        plt.rcParams.update({
            'figure.figsize': self.config.figure_size,
            'figure.dpi': self.config.dpi,
            'font.size': self.config.font_size,
            'axes.titlesize': self.config.title_size,
            'axes.labelsize': self.config.font_size,
            'xtick.labelsize': self.config.font_size - 1,
            'ytick.labelsize': self.config.font_size - 1,
            'legend.fontsize': self.config.font_size - 1,
            'figure.titlesize': self.config.title_size + 2
        })
        
        # Set seaborn defaults
        sns.set_style("whitegrid")
        sns.set_palette(self.config.color_palette)
    
    def create_comparison_boxplots(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        group_col: str = "model",
        statistical_results: Optional[StatisticalResults] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive box plots for model comparisons with statistical annotations.
        
        Args:
            df: DataFrame with conversation data
            metrics: List of metrics to plot
            group_col: Column name for grouping (usually 'model')
            statistical_results: Optional statistical test results for annotations
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        n_metrics = len(metrics)
        n_cols = min(3, n_metrics)
        n_rows = math.ceil(n_metrics / n_cols)
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(6 * n_cols, 5 * n_rows))
        if n_metrics == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = axes.reshape(1, -1)
        
        fig.suptitle("Model Performance Comparison - Distribution Analysis", 
                    fontsize=self.config.title_size + 4, y=0.98)
        
        for i, metric in enumerate(metrics):
            row = i // n_cols
            col = i % n_cols
            ax = axes[row, col] if n_rows > 1 else axes[col]
            
            # Create box plot
            box_plot = sns.boxplot(
                data=df, x=group_col, y=metric, ax=ax,
                palette=self.config.color_palette
            )
            
            # Customize plot
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontweight='bold')
            ax.set_xlabel("Model" if group_col == "model" else group_col.title())
            ax.set_ylabel(metric.replace('_', ' ').title())
            
            # Add statistical annotations if available
            if statistical_results and hasattr(statistical_results, 'anova_results'):
                anova_result = statistical_results.anova_results.get(metric)
                if anova_result and anova_result.is_significant:
                    # Add significance indicator
                    ax.text(0.02, 0.98, f"p = {anova_result.p_value:.4f}*",
                           transform=ax.transAxes, fontsize=10,
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="yellow", alpha=0.7),
                           verticalalignment='top')
                    
                    # Add effect size if available
                    if anova_result.effect_size:
                        effect_text = f"η² = {anova_result.effect_size:.3f}"
                        ax.text(0.02, 0.88, effect_text,
                               transform=ax.transAxes, fontsize=9,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.7),
                               verticalalignment='top')
            
            # Add mean markers
            means = df.groupby(group_col)[metric].mean()
            for j, (group, mean_val) in enumerate(means.items()):
                ax.scatter(j, mean_val, marker='D', s=50, color='red', zorder=10, label='Mean' if j == 0 else "")
            
            # Rotate x-axis labels if needed
            if len(df[group_col].unique()) > 3:
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Remove empty subplots
        for i in range(n_metrics, n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            fig.delaxes(axes[row, col] if n_rows > 1 else axes[col])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_radar_chart(
        self,
        df: pd.DataFrame,
        metrics: List[str],
        group_col: str = "model",
        normalize: bool = True,
        save_path: Optional[Path] = None
    ) -> go.Figure:
        """
        Create interactive radar chart for multi-dimensional model comparison.
        
        Args:
            df: DataFrame with conversation data
            metrics: List of metrics to include in radar chart
            group_col: Column name for grouping
            normalize: Whether to normalize metrics to 0-1 scale
            save_path: Optional path to save the figure
            
        Returns:
            Plotly figure object
        """
        # Calculate means for each group
        group_means = df.groupby(group_col)[metrics].mean()
        
        # Normalize if requested
        if normalize:
            for metric in metrics:
                col_min = df[metric].min()
                col_max = df[metric].max()
                if col_max > col_min:
                    group_means[metric] = (group_means[metric] - col_min) / (col_max - col_min)
                else:
                    group_means[metric] = 0.5  # Neutral value if no variation
        
        # Create radar chart
        fig = go.Figure()
        
        colors = px.colors.qualitative.Set2
        
        for i, (group_name, group_data) in enumerate(group_means.iterrows()):
            values = group_data.values.tolist()
            values.append(values[0])  # Close the radar chart
            
            metric_labels = [metric.replace('_', ' ').title() for metric in metrics]
            metric_labels.append(metric_labels[0])  # Close the labels
            
            fig.add_trace(go.Scatterpolar(
                r=values,
                theta=metric_labels,
                fill='toself',
                name=str(group_name),
                line=dict(color=colors[i % len(colors)]),
                fillcolor=colors[i % len(colors)],
                opacity=0.6
            ))
        
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1] if normalize else [0, max(group_means.max())],
                    tickformat=".2f"
                )
            ),
            title={
                'text': "Multi-Dimensional Model Performance Comparison",
                'x': 0.5,
                'xanchor': 'center',
                'font': {'size': self.config.title_size + 4}
            },
            showlegend=True,
            legend=dict(
                orientation="v",
                yanchor="top",
                y=1,
                xanchor="left",
                x=1.05
            ),
            width=800,
            height=600
        )
        
        if save_path:
            self._save_plotly_figure(fig, save_path)
        
        return fig
    
    def create_correlation_heatmap(
        self,
        df: pd.DataFrame,
        metrics: Optional[List[str]] = None,
        method: str = "pearson",
        mask_non_significant: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create correlation heatmap with significance masking.
        
        Args:
            df: DataFrame with conversation data
            metrics: Optional list of metrics (uses all numeric if None)
            method: Correlation method ('pearson', 'spearman', 'kendall')
            mask_non_significant: Whether to mask non-significant correlations
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        if metrics is None:
            metrics = df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Calculate correlations
        corr_data = df[metrics]
        correlation_matrix = corr_data.corr(method=method)
        
        # Calculate p-values if masking requested
        p_values = None
        if mask_non_significant:
            p_values = pd.DataFrame(np.zeros_like(correlation_matrix), 
                                  index=correlation_matrix.index, 
                                  columns=correlation_matrix.columns)
            
            for i, col1 in enumerate(correlation_matrix.columns):
                for j, col2 in enumerate(correlation_matrix.columns):
                    if i != j:
                        if method == "pearson":
                            _, p_val = stats.pearsonr(corr_data[col1].dropna(), corr_data[col2].dropna())
                        elif method == "spearman":
                            _, p_val = stats.spearmanr(corr_data[col1].dropna(), corr_data[col2].dropna())
                        else:  # kendall
                            _, p_val = stats.kendalltau(corr_data[col1].dropna(), corr_data[col2].dropna())
                        p_values.iloc[i, j] = p_val
        
        # Create mask for non-significant correlations
        mask = None
        if mask_non_significant and p_values is not None:
            mask = p_values > self.config.alpha_level
        
        # Create the heatmap
        fig, ax = plt.subplots(figsize=(len(metrics) * 0.8 + 2, len(metrics) * 0.8 + 2))
        
        # Generate custom colormap
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        
        # Create heatmap
        sns.heatmap(
            correlation_matrix,
            mask=mask,
            annot=True,
            fmt='.3f',
            cmap=cmap,
            center=0,
            square=True,
            linewidths=0.5,
            cbar_kws={"shrink": 0.8},
            ax=ax
        )
        
        ax.set_title(f"{method.title()} Correlation Matrix\n" + 
                    ("(Non-significant correlations masked)" if mask_non_significant else ""),
                    fontsize=self.config.title_size, pad=20)
        
        # Improve label formatting
        ax.set_xticklabels([label.get_text().replace('_', ' ').title() 
                           for label in ax.get_xticklabels()], rotation=45, ha='right')
        ax.set_yticklabels([label.get_text().replace('_', ' ').title() 
                           for label in ax.get_yticklabels()], rotation=0)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_time_series_analysis(
        self,
        df: pd.DataFrame,
        time_col: str,
        metrics: List[str],
        group_col: str = "model",
        smoothing: bool = True,
        confidence_intervals: bool = True,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create time series analysis plots for conversation flow analysis.
        
        Args:
            df: DataFrame with time-indexed conversation data
            time_col: Column name for time variable
            metrics: List of metrics to plot over time
            group_col: Column name for grouping
            smoothing: Whether to apply smoothing
            confidence_intervals: Whether to show confidence intervals
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Ensure time column is datetime
        if not pd.api.types.is_datetime64_any_dtype(df[time_col]):
            df[time_col] = pd.to_datetime(df[time_col])
        
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=(14, 5 * n_metrics), sharex=True)
        if n_metrics == 1:
            axes = [axes]
        
        fig.suptitle("Conversation Flow Analysis Over Time", 
                    fontsize=self.config.title_size + 4, y=0.98)
        
        colors = plt.cm.Set2(np.linspace(0, 1, len(df[group_col].unique())))
        
        for i, metric in enumerate(metrics):
            ax = axes[i]
            
            for j, (group_name, group_data) in enumerate(df.groupby(group_col)):
                # Sort by time
                group_data = group_data.sort_values(time_col)
                
                if smoothing:
                    # Apply rolling mean smoothing
                    window_size = max(3, len(group_data) // 10)
                    smoothed_values = group_data[metric].rolling(window=window_size, center=True).mean()
                    ax.plot(group_data[time_col], smoothed_values, 
                           label=f"{group_name} (smoothed)", color=colors[j], linewidth=2)
                    
                    if confidence_intervals:
                        # Calculate rolling standard error
                        rolling_std = group_data[metric].rolling(window=window_size, center=True).std()
                        rolling_se = rolling_std / np.sqrt(window_size)
                        
                        ax.fill_between(
                            group_data[time_col],
                            smoothed_values - 1.96 * rolling_se,
                            smoothed_values + 1.96 * rolling_se,
                            alpha=0.2, color=colors[j]
                        )
                else:
                    ax.plot(group_data[time_col], group_data[metric], 
                           label=group_name, color=colors[j], alpha=0.7, marker='o', markersize=3)
            
            ax.set_ylabel(metric.replace('_', ' ').title())
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Format y-axis
            ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, p: f'{x:.2f}'))
        
        # Format x-axis
        axes[-1].set_xlabel("Time")
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_statistical_significance_plot(
        self,
        statistical_results: StatisticalResults,
        metrics: Optional[List[str]] = None,
        correction_method: str = "bonferroni",
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create comprehensive statistical significance visualization.
        
        Args:
            statistical_results: Results from statistical analysis
            metrics: Optional list of metrics to include
            correction_method: Multiple comparison correction method
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract significance data
        significance_data = []
        
        # Process ANOVA results
        if hasattr(statistical_results, 'anova_results'):
            for metric, result in statistical_results.anova_results.items():
                if metrics is None or metric in metrics:
                    significance_data.append({
                        'metric': metric,
                        'test_type': 'ANOVA',
                        'p_value': result.p_value,
                        'is_significant': result.is_significant,
                        'effect_size': result.effect_size or 0
                    })
        
        # Process pairwise comparisons
        if hasattr(statistical_results, 'pairwise_comparisons'):
            for metric, comparisons in statistical_results.pairwise_comparisons.items():
                if metrics is None or metric in metrics:
                    for comparison in comparisons:
                        significance_data.append({
                            'metric': metric,
                            'test_type': f"Pairwise: {comparison.groups[0]} vs {comparison.groups[1]}",
                            'p_value': comparison.p_value,
                            'is_significant': comparison.is_significant,
                            'effect_size': comparison.effect_size or 0
                        })
        
        if not significance_data:
            logger.warning("No significance data found for plotting")
            return plt.figure()
        
        # Create DataFrame
        sig_df = pd.DataFrame(significance_data)
        
        # Apply multiple comparison correction if specified
        if correction_method != "none" and len(sig_df) > 1:
            from statsmodels.stats.multitest import multipletests
            
            method_map = {
                "bonferroni": "bonferroni",
                "holm": "holm",
                "benjamini_hochberg": "fdr_bh",
                "sidak": "sidak"
            }
            
            if correction_method in method_map:
                rejected, pvals_corrected, _, _ = multipletests(
                    sig_df['p_value'], method=method_map[correction_method]
                )
                sig_df['p_value_corrected'] = pvals_corrected
                sig_df['is_significant_corrected'] = rejected
        
        # Create the plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 8))
        
        # Plot 1: P-values with significance threshold
        colors = ['red' if sig else 'gray' for sig in sig_df['is_significant']]
        
        y_positions = range(len(sig_df))
        ax1.barh(y_positions, -np.log10(sig_df['p_value']), color=colors, alpha=0.7)
        ax1.axvline(-np.log10(self.config.alpha_level), color='black', linestyle='--', 
                   label=f'α = {self.config.alpha_level}')
        ax1.set_xlabel('-log10(p-value)')
        ax1.set_ylabel('Statistical Tests')
        ax1.set_title('Statistical Significance Overview')
        ax1.set_yticks(y_positions)
        ax1.set_yticklabels([f"{row['metric']}\n{row['test_type']}" for _, row in sig_df.iterrows()],
                           fontsize=9)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # Plot 2: Effect sizes
        if 'effect_size' in sig_df.columns:
            effect_colors = ['darkred' if sig else 'lightgray' for sig in sig_df['is_significant']]
            ax2.barh(y_positions, sig_df['effect_size'], color=effect_colors, alpha=0.7)
            ax2.set_xlabel('Effect Size')
            ax2.set_ylabel('Statistical Tests')
            ax2.set_title('Effect Size Analysis')
            ax2.set_yticks(y_positions)
            ax2.set_yticklabels([f"{row['metric']}\n{row['test_type']}" for _, row in sig_df.iterrows()],
                               fontsize=9)
            ax2.grid(True, alpha=0.3)
            
            # Add effect size interpretation lines
            ax2.axvline(0.2, color='green', linestyle=':', alpha=0.7, label='Small effect')
            ax2.axvline(0.5, color='orange', linestyle=':', alpha=0.7, label='Medium effect')
            ax2.axvline(0.8, color='red', linestyle=':', alpha=0.7, label='Large effect')
            ax2.legend()
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_effect_size_forest_plot(
        self,
        statistical_results: StatisticalResults,
        metrics: Optional[List[str]] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create forest plot for effect sizes with confidence intervals.
        
        Args:
            statistical_results: Results from statistical analysis
            metrics: Optional list of metrics to include
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Extract effect size data
        effect_data = []
        
        if hasattr(statistical_results, 'pairwise_comparisons'):
            for metric, comparisons in statistical_results.pairwise_comparisons.items():
                if metrics is None or metric in metrics:
                    for comparison in comparisons:
                        if comparison.effect_size is not None and comparison.confidence_interval:
                            effect_data.append({
                                'comparison': f"{comparison.groups[0]} vs {comparison.groups[1]}",
                                'metric': metric,
                                'effect_size': comparison.effect_size,
                                'ci_lower': comparison.confidence_interval[0],
                                'ci_upper': comparison.confidence_interval[1],
                                'significant': comparison.is_significant
                            })
        
        if not effect_data:
            logger.warning("No effect size data with confidence intervals found")
            return plt.figure()
        
        effect_df = pd.DataFrame(effect_data)
        
        # Create forest plot
        fig, ax = plt.subplots(figsize=(12, len(effect_df) * 0.5 + 3))
        
        y_positions = range(len(effect_df))
        
        for i, (_, row) in enumerate(effect_df.iterrows()):
            color = 'red' if row['significant'] else 'gray'
            
            # Plot effect size point
            ax.scatter(row['effect_size'], i, color=color, s=100, zorder=3)
            
            # Plot confidence interval
            ax.plot([row['ci_lower'], row['ci_upper']], [i, i], color=color, linewidth=2, zorder=2)
            
            # Plot CI caps
            ax.plot([row['ci_lower'], row['ci_lower']], [i-0.1, i+0.1], color=color, linewidth=2, zorder=2)
            ax.plot([row['ci_upper'], row['ci_upper']], [i-0.1, i+0.1], color=color, linewidth=2, zorder=2)
        
        # Add vertical line at no effect
        ax.axvline(0, color='black', linestyle='-', alpha=0.5, zorder=1)
        
        # Customize plot
        ax.set_yticks(y_positions)
        ax.set_yticklabels([f"{row['metric']}\n{row['comparison']}" for _, row in effect_df.iterrows()],
                          fontsize=10)
        ax.set_xlabel('Effect Size (Cohen\'s d)')
        ax.set_title('Forest Plot: Effect Sizes with 95% Confidence Intervals')
        ax.grid(True, alpha=0.3)
        
        # Add legend
        significant_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', 
                                     markersize=10, label='Significant')
        non_significant_patch = plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='gray', 
                                         markersize=10, label='Non-significant')
        ax.legend(handles=[significant_patch, non_significant_patch])
        
        plt.tight_layout()
        
        if save_path:
            self._save_figure(fig, save_path)
        
        return fig
    
    def create_publication_ready_figure(
        self,
        df: pd.DataFrame,
        primary_metric: str,
        secondary_metrics: List[str],
        group_col: str = "model",
        statistical_results: Optional[StatisticalResults] = None,
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Create publication-ready multi-panel figure.
        
        Args:
            df: DataFrame with conversation data
            primary_metric: Main metric for large panel
            secondary_metrics: Additional metrics for smaller panels
            group_col: Column name for grouping
            statistical_results: Optional statistical results
            save_path: Optional path to save the figure
            
        Returns:
            Matplotlib figure object
        """
        # Create figure with custom layout
        fig = plt.figure(figsize=(16, 12))
        gs = fig.add_gridspec(3, 4, hspace=0.3, wspace=0.3)
        
        # Main panel: Primary metric box plot
        ax_main = fig.add_subplot(gs[0:2, 0:2])
        
        box_plot = sns.boxplot(
            data=df, x=group_col, y=primary_metric, ax=ax_main,
            palette=self.config.color_palette
        )
        
        # Add statistical annotations
        if statistical_results and hasattr(statistical_results, 'anova_results'):
            result = statistical_results.anova_results.get(primary_metric)
            if result and result.is_significant:
                ax_main.text(0.02, 0.98, f"F = {result.test_statistic:.2f}, p = {result.p_value:.4f}",
                           transform=ax_main.transAxes, fontsize=12, weight='bold',
                           bbox=dict(boxstyle="round,pad=0.5", facecolor="yellow", alpha=0.8),
                           verticalalignment='top')
                
                if result.effect_size:
                    ax_main.text(0.02, 0.88, f"η² = {result.effect_size:.3f}",
                               transform=ax_main.transAxes, fontsize=11,
                               bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue", alpha=0.8),
                               verticalalignment='top')
        
        ax_main.set_title(f"{primary_metric.replace('_', ' ').title()}", 
                         fontsize=self.config.title_size + 2, weight='bold')
        ax_main.set_xlabel("Model", fontsize=self.config.font_size + 1, weight='bold')
        ax_main.set_ylabel(primary_metric.replace('_', ' ').title(), 
                          fontsize=self.config.font_size + 1, weight='bold')
        
        # Secondary panels: Smaller metrics
        secondary_positions = [
            gs[0, 2], gs[0, 3], gs[1, 2], gs[1, 3]
        ]
        
        for i, metric in enumerate(secondary_metrics[:4]):
            if i < len(secondary_positions):
                ax = fig.add_subplot(secondary_positions[i])
                
                sns.boxplot(data=df, x=group_col, y=metric, ax=ax, 
                           palette=self.config.color_palette)
                
                ax.set_title(metric.replace('_', ' ').title(), fontsize=self.config.font_size + 1)
                ax.set_xlabel("")
                ax.set_ylabel("")
                ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=9)
                
                # Add significance indicator
                if statistical_results and hasattr(statistical_results, 'anova_results'):
                    result = statistical_results.anova_results.get(metric)
                    if result and result.is_significant:
                        ax.text(0.5, 0.95, "***", transform=ax.transAxes, 
                               ha='center', va='top', fontsize=16, color='red', weight='bold')
        
        # Bottom panel: Summary statistics table
        ax_table = fig.add_subplot(gs[2, :])
        ax_table.axis('off')
        
        # Create summary table
        summary_data = []
        all_metrics = [primary_metric] + secondary_metrics
        
        for metric in all_metrics:
            row = [metric.replace('_', ' ').title()]
            for group in df[group_col].unique():
                group_data = df[df[group_col] == group][metric].dropna()
                if len(group_data) > 0:
                    row.append(f"{group_data.mean():.3f} ± {group_data.std():.3f}")
                else:
                    row.append("N/A")
            summary_data.append(row)
        
        # Create table
        columns = ['Metric'] + [f"{group}\n(Mean ± SD)" for group in df[group_col].unique()]
        table = ax_table.table(cellText=summary_data, colLabels=columns,
                             cellLoc='center', loc='center',
                             bbox=[0, 0, 1, 1])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2)
        
        # Style table
        for i in range(len(columns)):
            table[(0, i)].set_facecolor('#4CAF50')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        # Add main title
        fig.suptitle("Comprehensive Model Performance Analysis", 
                    fontsize=self.config.title_size + 6, y=0.98, weight='bold')
        
        if save_path:
            self._save_figure(fig, save_path, dpi=300)
        
        return fig
    
    def _save_figure(self, fig: plt.Figure, save_path: Path, **kwargs):
        """Save matplotlib figure with standard settings."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        save_kwargs = {
            'format': self.config.save_format,
            'dpi': kwargs.get('dpi', self.config.dpi),
            'bbox_inches': self.config.save_bbox_inches,
            'transparent': self.config.save_transparent,
            'facecolor': 'white'
        }
        
        fig.savefig(save_path, **save_kwargs)
        self.logger.info(f"Saved figure to {save_path}")
    
    def _save_plotly_figure(self, fig: go.Figure, save_path: Path):
        """Save plotly figure with standard settings."""
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        if save_path.suffix.lower() == '.html':
            fig.write_html(save_path)
        else:
            fig.write_image(save_path, width=1200, height=800, scale=2)
        
        self.logger.info(f"Saved interactive figure to {save_path}")


# Convenience functions for quick plotting
def quick_boxplot_comparison(
    df: pd.DataFrame,
    metric: str,
    group_col: str = "model",
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Quick box plot comparison for a single metric."""
    visualizer = AdvancedVisualizer()
    return visualizer.create_comparison_boxplots(df, [metric], group_col, save_path=save_path)


def quick_correlation_heatmap(
    df: pd.DataFrame,
    save_path: Optional[Path] = None
) -> plt.Figure:
    """Quick correlation heatmap for all numeric columns."""
    visualizer = AdvancedVisualizer()
    return visualizer.create_correlation_heatmap(df, save_path=save_path)


def quick_radar_chart(
    df: pd.DataFrame,
    metrics: List[str],
    group_col: str = "model",
    save_path: Optional[Path] = None
) -> go.Figure:
    """Quick radar chart for multiple metrics."""
    visualizer = AdvancedVisualizer()
    return visualizer.create_radar_chart(df, metrics, group_col, save_path=save_path)