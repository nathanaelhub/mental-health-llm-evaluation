#!/usr/bin/env python3
"""
Analyze Results Script

Performs statistical analysis and generates comprehensive visualizations
for evaluation results, including model comparisons and publication-ready figures.

Usage:
    python scripts/analyze_results.py --experiment exp_20240101_12345678
    python scripts/analyze_results.py --experiment exp_20240101_12345678 --output-format pdf
    python scripts/analyze_results.py --experiment exp_20240101_12345678 --include-visualizations
    python scripts/analyze_results.py --dry-run --experiment exp_20240101_12345678
"""

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from scipy import stats
from scipy.stats import f_oneway, ttest_ind
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from analysis.statistical_analysis import StatisticalAnalyzer, StatisticalResults
from analysis.advanced_visualization import AdvancedVisualizer
from utils.logging_config import setup_logging, get_logger


class ResultsAnalyzer:
    """Comprehensive analysis of evaluation results."""
    
    def __init__(self, experiment_id: str, dry_run: bool = False):
        self.experiment_id = experiment_id
        self.dry_run = dry_run
        self.logger = get_logger(__name__)
        
        # Initialize state
        self.experiment_dir = None
        self.manifest = None
        self.evaluation_data = None
        self.results_df = None
        self.statistical_analyzer = None
        self.visualizer = None
        
        # Analysis results
        self.statistical_results = None
        self.visualizations = {}
        
    def load_experiment(self) -> bool:
        """Load experiment configuration and evaluation results."""
        try:
            # Find experiment directory
            experiments_dir = PROJECT_ROOT / "experiments"
            self.experiment_dir = experiments_dir / self.experiment_id
            
            if not self.experiment_dir.exists():
                # Try finding by partial ID
                matching_dirs = [d for d in experiments_dir.iterdir() 
                               if d.is_dir() and self.experiment_id in d.name]
                if len(matching_dirs) == 1:
                    self.experiment_dir = matching_dirs[0]
                    self.experiment_id = matching_dirs[0].name
                elif len(matching_dirs) > 1:
                    self.logger.error(f"Multiple experiments match '{self.experiment_id}':")
                    for d in matching_dirs:
                        self.logger.error(f"  - {d.name}")
                    return False
                else:
                    self.logger.error(f"Experiment not found: {self.experiment_id}")
                    return False
            
            # Load manifest
            manifest_path = self.experiment_dir / "experiment_manifest.json"
            if manifest_path.exists():
                with open(manifest_path, 'r') as f:
                    self.manifest = json.load(f)
            
            # Load evaluation results
            results_path = self.experiment_dir / "evaluations" / "evaluation_results.json"
            if not results_path.exists():
                self.logger.error(f"Evaluation results not found: {results_path}")
                return False
            
            with open(results_path, 'r') as f:
                self.evaluation_data = json.load(f)
            
            self.logger.info(f"Loaded experiment: {self.experiment_id}")
            self.logger.info(f"Evaluation results: {len(self.evaluation_data)} entries")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to load experiment: {str(e)}")
            return False
    
    def prepare_analysis_data(self) -> bool:
        """Prepare data for statistical analysis."""
        try:
            # Extract successful evaluations
            successful_results = [
                result for result in self.evaluation_data 
                if result.get("status") == "completed"
            ]
            
            if not successful_results:
                self.logger.error("No successful evaluations found")
                return False
            
            # Convert to DataFrame
            analysis_data = []
            
            for result in successful_results:
                row = {
                    "conversation_id": result["conversation_id"],
                    "model_name": result["model_name"],
                    "scenario_id": result["scenario_id"],
                    "timestamp": result["timestamp"]
                }
                
                # Extract scores
                scores = result.get("scores", {})
                
                # Empathy scores
                empathy_data = scores.get("empathy", {})
                if isinstance(empathy_data, dict):
                    row["empathy_score"] = empathy_data.get("average_score", 0)
                    row["empathy_min"] = empathy_data.get("min_score", 0)
                    row["empathy_max"] = empathy_data.get("max_score", 0)
                else:
                    row["empathy_score"] = empathy_data
                
                # Safety scores
                safety_data = scores.get("safety", {})
                if isinstance(safety_data, dict):
                    row["safety_score"] = safety_data.get("safety_score", 10)
                    row["safety_flags"] = safety_data.get("flags_count", 0)
                    row["crisis_flags"] = safety_data.get("crisis_flags_count", 0)
                else:
                    row["safety_score"] = safety_data
                    row["safety_flags"] = 0
                    row["crisis_flags"] = 0
                
                # Coherence scores
                coherence_data = scores.get("coherence", {})
                if isinstance(coherence_data, dict):
                    row["coherence_score"] = coherence_data.get("average_score", 0)
                else:
                    row["coherence_score"] = coherence_data
                
                # Therapeutic scores
                therapeutic_data = scores.get("therapeutic", {})
                if isinstance(therapeutic_data, dict):
                    row["therapeutic_score"] = therapeutic_data.get("average_score", 0)
                else:
                    row["therapeutic_score"] = therapeutic_data
                
                # Composite scores
                composite_data = scores.get("composite", {})
                if isinstance(composite_data, dict):
                    row["overall_score"] = composite_data.get("overall_score", 0)
                    row["technical_score"] = composite_data.get("technical_score", 0)
                    row["therapeutic_composite"] = composite_data.get("therapeutic_score", 0)
                    row["patient_score"] = composite_data.get("patient_score", 0)
                    
                    # Technical details
                    tech_details = composite_data.get("details", {}).get("technical", {})
                    row["response_time_ms"] = tech_details.get("response_time_ms", 0)
                    row["throughput_rps"] = tech_details.get("throughput_rps", 0)
                    row["success_rate"] = tech_details.get("success_rate", 1.0)
                
                # Metadata
                metadata = result.get("metadata", {})
                row["conversation_length"] = metadata.get("conversation_length", 0)
                row["total_tokens"] = metadata.get("total_tokens", 0)
                row["avg_response_time"] = metadata.get("avg_response_time", 0)
                row["flags_total"] = len(result.get("flags", []))
                
                analysis_data.append(row)
            
            self.results_df = pd.DataFrame(analysis_data)
            
            # Basic data validation
            if self.results_df.empty:
                self.logger.error("No data available for analysis")
                return False
            
            # Add derived metrics
            self.results_df["safety_risk_score"] = 10 - self.results_df["safety_flags"]
            self.results_df["total_quality_score"] = (
                self.results_df["empathy_score"] * 0.3 +
                self.results_df["safety_score"] * 0.4 +
                self.results_df["coherence_score"] * 0.3
            )
            
            self.logger.info(f"Prepared analysis data: {len(self.results_df)} conversations")
            self.logger.info(f"Models: {self.results_df['model_name'].unique().tolist()}")
            self.logger.info(f"Scenarios: {len(self.results_df['scenario_id'].unique())} unique scenarios")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to prepare analysis data: {str(e)}")
            return False
    
    def initialize_analyzers(self) -> bool:
        """Initialize statistical analyzer and visualizer."""
        try:
            if not self.dry_run:
                self.statistical_analyzer = StatisticalAnalyzer()
                self.visualizer = AdvancedVisualizer()
            else:
                self.logger.info("DRY RUN: Would initialize analyzers")
            
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to initialize analyzers: {str(e)}")
            return False
    
    def perform_statistical_analysis(self) -> bool:
        """Perform comprehensive statistical analysis."""
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Would perform statistical analysis")
                return True
            
            self.logger.info("Performing statistical analysis...")
            
            # Prepare data for statistical analysis
            models = self.results_df['model_name'].unique()
            analysis_metrics = [
                "overall_score", "empathy_score", "safety_score", 
                "coherence_score", "therapeutic_score", "response_time_ms"
            ]
            
            results_by_model = {}
            for model in models:
                model_data = self.results_df[self.results_df['model_name'] == model]
                model_scores = []
                
                for _, row in model_data.iterrows():
                    # Create mock composite score object for the analyzer
                    from types import SimpleNamespace
                    
                    score = SimpleNamespace()
                    score.overall_score = row.get("overall_score", 0)
                    score.technical_score = row.get("technical_score", 0)
                    score.therapeutic_score = row.get("therapeutic_composite", 0)
                    score.patient_score = row.get("patient_score", 0)
                    
                    # Technical details
                    score.technical_details = SimpleNamespace()
                    score.technical_details.response_time_ms = row.get("response_time_ms", 0)
                    score.technical_details.throughput_rps = row.get("throughput_rps", 0)
                    score.technical_details.success_rate = row.get("success_rate", 1.0)
                    
                    # Therapeutic details
                    score.therapeutic_details = SimpleNamespace()
                    score.therapeutic_details.empathy_score = row.get("empathy_score", 0)
                    score.therapeutic_details.safety_score = row.get("safety_score", 0)
                    score.therapeutic_details.coherence_score = row.get("coherence_score", 0)
                    
                    # Patient details
                    score.patient_details = SimpleNamespace()
                    score.patient_details.satisfaction_score = row.get("coherence_score", 0)
                    score.patient_details.trust_score = row.get("empathy_score", 0) * 0.9
                    score.patient_details.engagement_score = row.get("overall_score", 0) * 0.8
                    
                    model_scores.append(score)
                
                results_by_model[model] = model_scores
            
            # Run statistical analysis
            self.statistical_results = self.statistical_analyzer.analyze_model_comparison(results_by_model)
            
            # Additional statistical tests
            self.additional_statistical_tests()
            
            self.logger.info("Statistical analysis completed")
            return True
            
        except Exception as e:
            self.logger.error(f"Statistical analysis failed: {str(e)}")
            return False
    
    def additional_statistical_tests(self) -> Dict[str, Any]:
        """Perform additional statistical tests."""
        additional_results = {}
        
        try:
            models = self.results_df['model_name'].unique()
            metrics = ["overall_score", "empathy_score", "safety_score", "coherence_score"]
            
            # ANOVA tests for each metric
            for metric in metrics:
                groups = [
                    self.results_df[self.results_df['model_name'] == model][metric].values
                    for model in models
                ]
                
                # Remove any groups with insufficient data
                groups = [group for group in groups if len(group) >= 3]
                
                if len(groups) >= 2:
                    f_stat, p_value = f_oneway(*groups)
                    
                    additional_results[f"{metric}_anova"] = {
                        "f_statistic": f_stat,
                        "p_value": p_value,
                        "significant": p_value < 0.05
                    }
            
            # Pairwise t-tests between models
            pairwise_results = {}
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i < j:  # Avoid duplicate comparisons
                        for metric in metrics:
                            group1 = self.results_df[self.results_df['model_name'] == model1][metric]
                            group2 = self.results_df[self.results_df['model_name'] == model2][metric]
                            
                            if len(group1) >= 3 and len(group2) >= 3:
                                t_stat, p_value = ttest_ind(group1, group2)
                                effect_size = self._calculate_cohens_d(group1, group2)
                                
                                comparison_key = f"{model1}_vs_{model2}_{metric}"
                                pairwise_results[comparison_key] = {
                                    "t_statistic": t_stat,
                                    "p_value": p_value,
                                    "effect_size": effect_size,
                                    "significant": p_value < 0.05,
                                    "effect_magnitude": self._interpret_effect_size(effect_size)
                                }
            
            additional_results["pairwise_tests"] = pairwise_results
            
            # Correlation analysis
            correlation_matrix = self.results_df[metrics + ["response_time_ms", "total_tokens"]].corr()
            additional_results["correlations"] = correlation_matrix.to_dict()
            
            # Safety analysis
            safety_analysis = self._analyze_safety_patterns()
            additional_results["safety_analysis"] = safety_analysis
            
            return additional_results
            
        except Exception as e:
            self.logger.error(f"Additional statistical tests failed: {str(e)}")
            return additional_results
    
    def _calculate_cohens_d(self, group1, group2):
        """Calculate Cohen's d effect size."""
        n1, n2 = len(group1), len(group2)
        pooled_std = np.sqrt(((n1 - 1) * np.var(group1, ddof=1) + 
                             (n2 - 1) * np.var(group2, ddof=1)) / (n1 + n2 - 2))
        return (np.mean(group1) - np.mean(group2)) / pooled_std
    
    def _interpret_effect_size(self, effect_size):
        """Interpret Cohen's d effect size."""
        abs_effect = abs(effect_size)
        if abs_effect < 0.2:
            return "negligible"
        elif abs_effect < 0.5:
            return "small"
        elif abs_effect < 0.8:
            return "medium"
        else:
            return "large"
    
    def _analyze_safety_patterns(self) -> Dict[str, Any]:
        """Analyze safety-related patterns in the data."""
        safety_analysis = {}
        
        try:
            # Safety score distribution by model
            safety_by_model = self.results_df.groupby('model_name')['safety_score'].agg([
                'mean', 'std', 'min', 'max', 'count'
            ]).to_dict('index')
            safety_analysis["safety_by_model"] = safety_by_model
            
            # Crisis detection rates
            crisis_rates = self.results_df.groupby('model_name')['crisis_flags'].agg([
                'sum', 'mean', 'count'
            ]).to_dict('index')
            safety_analysis["crisis_rates"] = crisis_rates
            
            # Safety flags distribution
            flag_distribution = self.results_df.groupby('model_name')['safety_flags'].describe().to_dict('index')
            safety_analysis["flag_distribution"] = flag_distribution
            
            # Identify conversations with safety concerns
            safety_concerns = self.results_df[
                (self.results_df['safety_score'] < 7) | (self.results_df['crisis_flags'] > 0)
            ]
            
            safety_analysis["safety_concerns"] = {
                "total_count": len(safety_concerns),
                "percentage": len(safety_concerns) / len(self.results_df) * 100,
                "by_model": safety_concerns['model_name'].value_counts().to_dict()
            }
            
            return safety_analysis
            
        except Exception as e:
            self.logger.error(f"Safety analysis failed: {str(e)}")
            return safety_analysis
    
    def generate_visualizations(self, include_interactive: bool = True) -> Dict[str, Path]:
        """Generate comprehensive visualization suite."""
        try:
            if self.dry_run:
                self.logger.info("DRY RUN: Would generate visualizations")
                return {}
            
            self.logger.info("Generating visualizations...")
            
            # Create visualizations directory
            viz_dir = self.experiment_dir / "results" / "visualizations"
            viz_dir.mkdir(parents=True, exist_ok=True)
            
            generated_files = {}
            
            # Set style for matplotlib
            plt.style.use('seaborn-v0_8')
            sns.set_palette("husl")
            
            # 1. Model Comparison Box Plots
            self.logger.info("Creating comparison box plots...")
            metrics = ["overall_score", "empathy_score", "safety_score", "coherence_score"]
            
            fig, axes = plt.subplots(2, 2, figsize=(15, 12))
            axes = axes.ravel()
            
            for i, metric in enumerate(metrics):
                sns.boxplot(data=self.results_df, x='model_name', y=metric, ax=axes[i])
                axes[i].set_title(f'{metric.replace("_", " ").title()} by Model')
                axes[i].set_xlabel('Model')
                axes[i].set_ylabel('Score')
                axes[i].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            box_plot_path = viz_dir / "model_comparison_boxplots.png"
            plt.savefig(box_plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files["boxplots"] = box_plot_path
            
            # 2. Correlation Heatmap
            self.logger.info("Creating correlation heatmap...")
            correlation_metrics = [
                "overall_score", "empathy_score", "safety_score", 
                "coherence_score", "response_time_ms", "total_tokens"
            ]
            
            corr_matrix = self.results_df[correlation_metrics].corr()
            
            plt.figure(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0,
                       square=True, linewidths=0.5)
            plt.title('Metric Correlations')
            plt.tight_layout()
            
            heatmap_path = viz_dir / "correlation_heatmap.png"
            plt.savefig(heatmap_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files["heatmap"] = heatmap_path
            
            # 3. Safety Analysis Visualization
            self.logger.info("Creating safety analysis plots...")
            fig, axes = plt.subplots(1, 3, figsize=(18, 6))
            
            # Safety score distribution
            sns.histplot(data=self.results_df, x='safety_score', hue='model_name', 
                        alpha=0.7, kde=True, ax=axes[0])
            axes[0].set_title('Safety Score Distribution')
            axes[0].axvline(x=7, color='red', linestyle='--', label='Safety Threshold')
            axes[0].legend()
            
            # Safety flags by model
            flag_data = self.results_df.groupby('model_name')['safety_flags'].sum()
            axes[1].bar(flag_data.index, flag_data.values)
            axes[1].set_title('Total Safety Flags by Model')
            axes[1].set_xlabel('Model')
            axes[1].set_ylabel('Safety Flags Count')
            axes[1].tick_params(axis='x', rotation=45)
            
            # Crisis detection rates
            crisis_data = self.results_df.groupby('model_name')['crisis_flags'].sum()
            axes[2].bar(crisis_data.index, crisis_data.values, color='red', alpha=0.7)
            axes[2].set_title('Crisis Flags by Model')
            axes[2].set_xlabel('Model')
            axes[2].set_ylabel('Crisis Flags Count')
            axes[2].tick_params(axis='x', rotation=45)
            
            plt.tight_layout()
            safety_path = viz_dir / "safety_analysis.png"
            plt.savefig(safety_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files["safety"] = safety_path
            
            # 4. Performance vs Quality Scatter Plot
            self.logger.info("Creating performance vs quality scatter plot...")
            plt.figure(figsize=(12, 8))
            
            models = self.results_df['model_name'].unique()
            colors = plt.cm.Set1(np.linspace(0, 1, len(models)))
            
            for i, model in enumerate(models):
                model_data = self.results_df[self.results_df['model_name'] == model]
                plt.scatter(model_data['response_time_ms'], model_data['overall_score'],
                          label=model, alpha=0.6, s=60, color=colors[i])
            
            plt.xlabel('Response Time (ms)')
            plt.ylabel('Overall Quality Score')
            plt.title('Performance vs Quality Trade-off')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            scatter_path = viz_dir / "performance_vs_quality.png"
            plt.savefig(scatter_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files["scatter"] = scatter_path
            
            # 5. Interactive Plotly Visualizations (if requested)
            if include_interactive:
                self.logger.info("Creating interactive visualizations...")
                
                # Interactive box plot
                fig_box = px.box(self.results_df, x='model_name', y='overall_score',
                               title='Overall Score Distribution by Model',
                               labels={'model_name': 'Model', 'overall_score': 'Overall Score'})
                fig_box.update_layout(showlegend=False)
                
                interactive_box_path = viz_dir / "interactive_boxplot.html"
                fig_box.write_html(interactive_box_path)
                generated_files["interactive_box"] = interactive_box_path
                
                # Interactive radar chart
                radar_data = self.results_df.groupby('model_name')[
                    ['empathy_score', 'safety_score', 'coherence_score', 'therapeutic_score']
                ].mean()
                
                fig_radar = go.Figure()
                
                for model in radar_data.index:
                    fig_radar.add_trace(go.Scatterpolar(
                        r=radar_data.loc[model].values,
                        theta=['Empathy', 'Safety', 'Coherence', 'Therapeutic'],
                        fill='toself',
                        name=model
                    ))
                
                fig_radar.update_layout(
                    polar=dict(
                        radialaxis=dict(visible=True, range=[0, 10])
                    ),
                    showlegend=True,
                    title="Model Performance Radar Chart"
                )
                
                radar_path = viz_dir / "interactive_radar.html"
                fig_radar.write_html(radar_path)
                generated_files["interactive_radar"] = radar_path
            
            # 6. Summary Statistics Table
            self.logger.info("Creating summary statistics table...")
            summary_stats = self.results_df.groupby('model_name')[
                ['overall_score', 'empathy_score', 'safety_score', 'coherence_score', 'response_time_ms']
            ].agg(['mean', 'std', 'min', 'max']).round(2)
            
            # Create a formatted table
            fig, ax = plt.subplots(figsize=(14, 8))
            ax.axis('tight')
            ax.axis('off')
            
            # Flatten column names for the table
            columns = [f"{col[1]}_{col[0]}" for col in summary_stats.columns]
            table_data = summary_stats.values
            
            table = ax.table(cellText=table_data,
                           rowLabels=summary_stats.index,
                           colLabels=columns,
                           cellLoc='center',
                           loc='center')
            
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 1.5)
            
            plt.title('Summary Statistics by Model', fontsize=16, pad=20)
            
            stats_table_path = viz_dir / "summary_statistics_table.png"
            plt.savefig(stats_table_path, dpi=300, bbox_inches='tight')
            plt.close()
            generated_files["stats_table"] = stats_table_path
            
            self.logger.info(f"Generated {len(generated_files)} visualizations")
            return generated_files
            
        except Exception as e:
            self.logger.error(f"Visualization generation failed: {str(e)}")
            return {}
    
    def save_analysis_results(self, visualization_files: Dict[str, Path]) -> Dict[str, Path]:
        """Save complete analysis results."""
        try:
            results_dir = self.experiment_dir / "results"
            results_dir.mkdir(exist_ok=True)
            
            saved_files = {}
            
            if not self.dry_run:
                # Save statistical results
                if self.statistical_results:
                    stats_path = results_dir / "statistical_analysis.json"
                    
                    # Convert statistical results to serializable format
                    serializable_stats = {
                        "anova_results": getattr(self.statistical_results, 'anova_results', {}),
                        "descriptive_stats": getattr(self.statistical_results, 'descriptive_stats', {}),
                        "recommendations": getattr(self.statistical_results, 'recommendations', [])
                    }
                    
                    # Add pairwise comparisons if available
                    if hasattr(self.statistical_results, 'pairwise_comparisons'):
                        serializable_stats["pairwise_comparisons"] = self.statistical_results.pairwise_comparisons
                    
                    with open(stats_path, 'w') as f:
                        json.dump(serializable_stats, f, indent=2, default=str)
                    
                    saved_files["statistics"] = stats_path
                
                # Save summary DataFrame
                if self.results_df is not None:
                    summary_path = results_dir / "analysis_summary.csv"
                    self.results_df.to_csv(summary_path, index=False)
                    saved_files["summary_csv"] = summary_path
                
                # Save model comparison summary
                model_summary = self._create_model_summary()
                model_summary_path = results_dir / "model_comparison_summary.json"
                with open(model_summary_path, 'w') as f:
                    json.dump(model_summary, f, indent=2)
                saved_files["model_summary"] = model_summary_path
                
                # Save analysis metadata
                analysis_metadata = {
                    "experiment_id": self.experiment_id,
                    "analysis_timestamp": datetime.now().isoformat(),
                    "total_conversations": len(self.results_df) if self.results_df is not None else 0,
                    "models_analyzed": self.results_df['model_name'].unique().tolist() if self.results_df is not None else [],
                    "metrics_analyzed": [
                        "overall_score", "empathy_score", "safety_score", 
                        "coherence_score", "therapeutic_score"
                    ],
                    "visualizations_generated": list(visualization_files.keys()),
                    "statistical_tests_performed": [
                        "ANOVA", "pairwise_t_tests", "correlation_analysis", "safety_analysis"
                    ]
                }
                
                metadata_path = results_dir / "analysis_metadata.json"
                with open(metadata_path, 'w') as f:
                    json.dump(analysis_metadata, f, indent=2)
                saved_files["metadata"] = metadata_path
            
            return saved_files
            
        except Exception as e:
            self.logger.error(f"Failed to save analysis results: {str(e)}")
            return {}
    
    def _create_model_summary(self) -> Dict[str, Any]:
        """Create comprehensive model comparison summary."""
        if self.results_df is None:
            return {}
        
        summary = {}
        
        try:
            models = self.results_df['model_name'].unique()
            
            for model in models:
                model_data = self.results_df[self.results_df['model_name'] == model]
                
                summary[model] = {
                    "conversation_count": len(model_data),
                    "performance_metrics": {
                        "overall_score": {
                            "mean": float(model_data['overall_score'].mean()),
                            "std": float(model_data['overall_score'].std()),
                            "min": float(model_data['overall_score'].min()),
                            "max": float(model_data['overall_score'].max())
                        },
                        "empathy_score": {
                            "mean": float(model_data['empathy_score'].mean()),
                            "std": float(model_data['empathy_score'].std())
                        },
                        "safety_score": {
                            "mean": float(model_data['safety_score'].mean()),
                            "std": float(model_data['safety_score'].std())
                        },
                        "coherence_score": {
                            "mean": float(model_data['coherence_score'].mean()),
                            "std": float(model_data['coherence_score'].std())
                        },
                        "response_time_ms": {
                            "mean": float(model_data['response_time_ms'].mean()),
                            "median": float(model_data['response_time_ms'].median())
                        }
                    },
                    "safety_analysis": {
                        "total_safety_flags": int(model_data['safety_flags'].sum()),
                        "total_crisis_flags": int(model_data['crisis_flags'].sum()),
                        "conversations_with_safety_concerns": int(
                            len(model_data[model_data['safety_score'] < 7])
                        ),
                        "safety_concern_rate": float(
                            len(model_data[model_data['safety_score'] < 7]) / len(model_data) * 100
                        )
                    },
                    "quality_distribution": {
                        "excellent_quality_count": int(len(model_data[model_data['overall_score'] >= 8])),
                        "good_quality_count": int(len(model_data[
                            (model_data['overall_score'] >= 6) & (model_data['overall_score'] < 8)
                        ])),
                        "poor_quality_count": int(len(model_data[model_data['overall_score'] < 6]))
                    }
                }
            
            # Add comparative analysis
            if len(models) > 1:
                best_overall = self.results_df.groupby('model_name')['overall_score'].mean().idxmax()
                best_safety = self.results_df.groupby('model_name')['safety_score'].mean().idxmax()
                fastest_response = self.results_df.groupby('model_name')['response_time_ms'].mean().idxmin()
                
                summary["comparative_analysis"] = {
                    "best_overall_quality": best_overall,
                    "best_safety": best_safety,
                    "fastest_response": fastest_response,
                    "recommendation": self._generate_recommendation(models)
                }
            
            return summary
            
        except Exception as e:
            self.logger.error(f"Failed to create model summary: {str(e)}")
            return summary
    
    def _generate_recommendation(self, models: List[str]) -> str:
        """Generate recommendation based on analysis."""
        try:
            model_scores = self.results_df.groupby('model_name').agg({
                'overall_score': 'mean',
                'safety_score': 'mean',
                'response_time_ms': 'mean',
                'safety_flags': 'sum'
            })
            
            # Weight different factors
            weighted_scores = (
                model_scores['overall_score'] * 0.4 +
                model_scores['safety_score'] * 0.4 +
                (10 - model_scores['response_time_ms'] / 1000) * 0.1 -  # Penalty for slow response
                model_scores['safety_flags'] * 0.1  # Penalty for safety flags
            )
            
            best_model = weighted_scores.idxmax()
            
            recommendation = f"Based on the comprehensive analysis, {best_model} is recommended for mental health applications. "
            
            # Add specific reasoning
            best_safety = model_scores['safety_score'].idxmax()
            best_quality = model_scores['overall_score'].idxmax()
            
            if best_model == best_safety and best_model == best_quality:
                recommendation += f"It demonstrates the best overall quality and safety performance."
            elif best_model == best_safety:
                recommendation += f"While not the highest in overall quality, it excels in safety - a critical factor for mental health applications."
            elif best_model == best_quality:
                recommendation += f"It provides the highest overall quality, with acceptable safety performance."
            else:
                recommendation += f"It provides the best balance of quality, safety, and performance factors."
            
            return recommendation
            
        except Exception as e:
            self.logger.error(f"Failed to generate recommendation: {str(e)}")
            return "Unable to generate recommendation due to insufficient data."


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Analyze results for Mental Health LLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Full analysis with visualizations
  python scripts/analyze_results.py --experiment exp_20240101_12345678
  
  # Analysis without interactive plots
  python scripts/analyze_results.py --experiment exp_20240101_12345678 --no-interactive
  
  # Test run without generating files
  python scripts/analyze_results.py --dry-run --experiment exp_20240101_12345678
        """
    )
    
    parser.add_argument(
        "--experiment", "-e",
        type=str,
        required=True,
        help="Experiment ID to analyze results for"
    )
    
    parser.add_argument(
        "--output-format",
        type=str,
        choices=["png", "pdf", "svg"],
        default="png",
        help="Output format for static visualizations"
    )
    
    parser.add_argument(
        "--no-interactive",
        action="store_true",
        help="Skip interactive visualizations"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test run without generating actual analysis"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    try:
        # Initialize analyzer
        analyzer = ResultsAnalyzer(
            experiment_id=args.experiment,
            dry_run=args.dry_run
        )
        
        # Load experiment data
        if not analyzer.load_experiment():
            return 1
        
        # Prepare analysis data
        if not analyzer.prepare_analysis_data():
            return 1
        
        # Initialize analyzers
        if not analyzer.initialize_analyzers():
            return 1
        
        # Display experiment info
        print(f"\n{'='*60}")
        print(f"Mental Health LLM Evaluation - Results Analysis")
        print(f"{'='*60}")
        print(f"Experiment ID: {analyzer.experiment_id}")
        if analyzer.results_df is not None:
            print(f"Conversations: {len(analyzer.results_df)}")
            print(f"Models: {', '.join(analyzer.results_df['model_name'].unique())}")
            print(f"Scenarios: {len(analyzer.results_df['scenario_id'].unique())} unique")
        
        if args.dry_run:
            print("🧪 DRY RUN MODE - No analysis will be performed")
        
        print()
        
        # Perform statistical analysis
        if not analyzer.perform_statistical_analysis():
            return 1
        
        # Generate visualizations
        include_interactive = not args.no_interactive
        visualization_files = analyzer.generate_visualizations(include_interactive)
        
        # Save results
        saved_files = analyzer.save_analysis_results(visualization_files)
        
        print(f"\n✅ Results analysis completed!")
        if not args.dry_run:
            print(f"📊 Generated files:")
            for file_type, file_path in {**visualization_files, **saved_files}.items():
                print(f"  - {file_type}: {file_path}")
            
            print(f"\nNext step:")
            print(f"python scripts/generate_report.py --experiment {analyzer.experiment_id}")
        
        return 0
    
    except KeyboardInterrupt:
        logger.info("Analysis interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())