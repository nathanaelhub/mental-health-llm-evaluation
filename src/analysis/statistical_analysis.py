"""
Statistical Analysis Module for LLM Performance Comparison

This module provides comprehensive statistical analysis capabilities including
descriptive statistics, hypothesis testing, effect size calculations,
and multiple comparison corrections for mental health LLM evaluation.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Any, Tuple, Union
import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import mannwhitneyu, wilcoxon, kruskal, friedmanchisquare, f_oneway
from statsmodels.stats.multicomp import pairwise_tukeyhsd, MultiComparison
from statsmodels.stats.power import ttest_power
from statsmodels.stats.contingency_tables import mcnemar
from statsmodels.stats.multitest import multipletests
from statsmodels.stats.anova import anova_lm
from statsmodels.formula.api import ols
import logging
from enum import Enum
import warnings
import itertools
from math import sqrt

try:
    from ..evaluation.composite_scorer import CompositeScore
except ImportError:
    # Fallback for standalone usage
    CompositeScore = None

logger = logging.getLogger(__name__)


@dataclass
class DescriptiveStatistics:
    """Descriptive statistics for a metric."""
    
    count: int
    mean: float
    std: float
    median: float
    min_val: float
    max_val: float
    q25: float
    q75: float
    skewness: float
    kurtosis: float
    

@dataclass
class StatisticalTestResult:
    """Result of a statistical test."""
    
    test_name: str
    test_statistic: float
    p_value: float
    is_significant: bool
    alpha_level: float
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    degrees_of_freedom: Optional[int] = None
    power: Optional[float] = None
    

@dataclass
class PairwiseComparison:
    """Pairwise comparison result."""
    
    groups: Tuple[str, str]
    test_statistic: float
    p_value: float
    adjusted_p_value: float
    is_significant: bool
    effect_size: Optional[float] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    

@dataclass
class StatisticalResults:
    """Comprehensive results from statistical analysis of model comparisons."""
    
    # Basic descriptive statistics
    descriptive_stats: Dict[str, Dict[str, DescriptiveStatistics]]
    
    # ANOVA results
    anova_results: Dict[str, StatisticalTestResult]
    
    # Pairwise comparisons
    pairwise_comparisons: Dict[str, List[PairwiseComparison]]
    
    # Non-parametric tests
    nonparametric_tests: Dict[str, StatisticalTestResult]
    
    # Multiple comparison corrections
    multiple_comparison_corrections: Dict[str, Dict[str, Any]]
    
    # Effect sizes
    effect_sizes: Dict[str, Dict[str, float]]
    
    # Power analysis
    power_analysis: Dict[str, Dict[str, float]]
    
    # Recommendations
    recommendations: List[str]
    
    # Metadata
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert results to dictionary for serialization."""
        
        # Convert dataclass objects to dictionaries
        descriptive_dict = {}
        for model, stats in self.descriptive_stats.items():
            descriptive_dict[model] = {}
            for metric, desc_stats in stats.items():
                descriptive_dict[model][metric] = {
                    'count': desc_stats.count,
                    'mean': desc_stats.mean,
                    'std': desc_stats.std,
                    'median': desc_stats.median,
                    'min': desc_stats.min_val,
                    'max': desc_stats.max_val,
                    'q25': desc_stats.q25,
                    'q75': desc_stats.q75,
                    'skewness': desc_stats.skewness,
                    'kurtosis': desc_stats.kurtosis
                }
        
        anova_dict = {}
        for metric, result in self.anova_results.items():
            anova_dict[metric] = {
                'test_name': result.test_name,
                'test_statistic': result.test_statistic,
                'p_value': result.p_value,
                'is_significant': result.is_significant,
                'effect_size': result.effect_size,
                'degrees_of_freedom': result.degrees_of_freedom
            }
        
        pairwise_dict = {}
        for metric, comparisons in self.pairwise_comparisons.items():
            pairwise_dict[metric] = []
            for comp in comparisons:
                pairwise_dict[metric].append({
                    'groups': comp.groups,
                    'test_statistic': comp.test_statistic,
                    'p_value': comp.p_value,
                    'adjusted_p_value': comp.adjusted_p_value,
                    'is_significant': comp.is_significant,
                    'effect_size': comp.effect_size,
                    'confidence_interval': comp.confidence_interval
                })
        
        nonparam_dict = {}
        for metric, result in self.nonparametric_tests.items():
            nonparam_dict[metric] = {
                'test_name': result.test_name,
                'test_statistic': result.test_statistic,
                'p_value': result.p_value,
                'is_significant': result.is_significant,
                'effect_size': result.effect_size
            }
        
        return {
            "descriptive_statistics": descriptive_dict,
            "anova_results": anova_dict,
            "pairwise_comparisons": pairwise_dict,
            "nonparametric_tests": nonparam_dict,
            "multiple_comparison_corrections": self.multiple_comparison_corrections,
            "effect_sizes": self.effect_sizes,
            "power_analysis": self.power_analysis,
            "recommendations": self.recommendations,
            "metadata": self.metadata
        }


class StatisticalAnalyzer:
    """Statistical analyzer for model evaluation results."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize statistical analyzer.
        
        Args:
            config: Configuration for statistical analysis
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Statistical parameters
        self.alpha = self.config.get("alpha", 0.05)
        self.confidence_level = self.config.get("confidence_level", 0.95)
        self.min_sample_size = self.config.get("min_sample_size", 5)
        self.effect_size_thresholds = self.config.get("effect_size_thresholds", {
            "small": 0.2,
            "medium": 0.5,
            "large": 0.8
        })
    
    def analyze_model_comparison(
        self,
        results: Dict[str, List[CompositeScore]],
        **kwargs
    ) -> StatisticalResults:
        """
        Comprehensive statistical analysis of model comparison results.
        
        Args:
            results: Dictionary mapping model names to lists of composite scores
            **kwargs: Additional analysis parameters
            
        Returns:
            Statistical analysis results
        """
        self.logger.info(f"Starting statistical analysis of {len(results)} models")
        
        # Validate input data
        if not self._validate_input(results):
            raise ValueError("Invalid input data for statistical analysis")
        
        # Convert to DataFrame for easier analysis
        df = self._create_analysis_dataframe(results)
        
        # Perform comprehensive statistical analyses
        descriptive_stats = self._calculate_descriptive_statistics(df)
        anova_results = self._perform_comprehensive_anova(df)
        pairwise_comparisons = self._perform_pairwise_comparisons(df)
        nonparametric_tests = self._perform_nonparametric_tests(df)
        multiple_corrections = self._apply_multiple_comparison_corrections(pairwise_comparisons)
        effect_sizes = self._calculate_comprehensive_effect_sizes(df, anova_results, pairwise_comparisons)
        power_analysis = self._calculate_power_analysis(df)
        recommendations = self._generate_comprehensive_recommendations(
            descriptive_stats, anova_results, pairwise_comparisons, effect_sizes
        )
        
        statistical_results = StatisticalResults(
            descriptive_stats=descriptive_stats,
            anova_results=anova_results,
            pairwise_comparisons=pairwise_comparisons,
            nonparametric_tests=nonparametric_tests,
            multiple_comparison_corrections=multiple_corrections,
            effect_sizes=effect_sizes,
            power_analysis=power_analysis,
            recommendations=recommendations,
            metadata={
                "analysis_config": self.config,
                "sample_sizes": {model: len(scores) for model, scores in results.items()},
                "total_comparisons": len(results),
                "analysis_timestamp": pd.Timestamp.now().isoformat(),
                "alpha_level": self.alpha,
                "confidence_level": self.confidence_level
            }
        )
        
        self.logger.info("Statistical analysis complete")
        return statistical_results
    
    def _validate_input(self, results: Dict[str, List[CompositeScore]]) -> bool:
        """Validate input data for statistical analysis."""
        
        if len(results) < 2:
            self.logger.error("Need at least 2 models for comparison")
            return False
        
        for model_name, scores in results.items():
            if len(scores) < self.min_sample_size:
                self.logger.warning(
                    f"Model {model_name} has only {len(scores)} samples "
                    f"(minimum: {self.min_sample_size})"
                )
        
        return True
    
    def _create_analysis_dataframe(self, results: Dict[str, List[CompositeScore]]) -> pd.DataFrame:
        """Create DataFrame from results for analysis."""
        
        data = []
        
        for model_name, scores in results.items():
            for i, score in enumerate(scores):
                data.append({
                    "model": model_name,
                    "sample_id": i,
                    "overall_score": score.overall_score,
                    "technical_score": score.technical_score,
                    "therapeutic_score": score.therapeutic_score,
                    "patient_score": score.patient_score,
                    "response_time_ms": score.technical_details.response_time_ms,
                    "throughput_rps": score.technical_details.throughput_rps,
                    "success_rate": score.technical_details.success_rate,
                    "empathy_score": score.therapeutic_details.empathy_score,
                    "safety_score": score.therapeutic_details.safety_score,
                    "coherence_score": score.therapeutic_details.coherence_score,
                    "satisfaction_score": score.patient_details.satisfaction_score,
                    "trust_score": score.patient_details.trust_score,
                    "engagement_score": score.patient_details.engagement_score
                })
        
        return pd.DataFrame(data)
    
    def _calculate_descriptive_statistics(self, df: pd.DataFrame) -> Dict[str, Dict[str, DescriptiveStatistics]]:
        """Calculate summary statistics for each model and metric."""
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        descriptive_stats = {}
        
        for model in df['model'].unique():
            model_data = df[df['model'] == model]
            model_stats = {}
            
            for col in numeric_cols:
                values = model_data[col].dropna()
                if len(values) > 0:
                    model_stats[col] = DescriptiveStatistics(
                        count=len(values),
                        mean=float(values.mean()),
                        std=float(values.std()),
                        median=float(values.median()),
                        min_val=float(values.min()),
                        max_val=float(values.max()),
                        q25=float(values.quantile(0.25)),
                        q75=float(values.quantile(0.75)),
                        skewness=float(values.skew()),
                        kurtosis=float(values.kurtosis())
                    )
            
            descriptive_stats[model] = model_stats
        
        return descriptive_stats
    
    def _perform_comprehensive_anova(self, df: pd.DataFrame) -> Dict[str, StatisticalTestResult]:
        """Perform comprehensive one-way ANOVA analysis."""
        
        anova_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        models = df['model'].unique()
        
        if len(models) < 2:
            return anova_results
        
        for col in numeric_cols:
            # Get groups
            groups = [df[df['model'] == model][col].dropna() for model in models]
            groups = [group for group in groups if len(group) >= 3]
            
            if len(groups) < 2:
                continue
            
            try:
                # Perform one-way ANOVA
                f_statistic, p_value = f_oneway(*groups)
                
                # Calculate effect size (eta-squared)
                # eta^2 = SSbetween / SStotal
                grand_mean = np.concatenate(groups).mean()
                ss_total = sum((np.concatenate(groups) - grand_mean) ** 2)
                
                group_means = [group.mean() for group in groups]
                group_sizes = [len(group) for group in groups]
                ss_between = sum(n * (mean - grand_mean) ** 2 for n, mean in zip(group_sizes, group_means))
                
                eta_squared = ss_between / ss_total if ss_total > 0 else 0
                
                # Degrees of freedom
                df_between = len(groups) - 1
                df_within = sum(len(group) for group in groups) - len(groups)
                
                # Create result
                anova_results[col] = StatisticalTestResult(
                    test_name="One-way ANOVA",
                    test_statistic=float(f_statistic),
                    p_value=float(p_value),
                    is_significant=p_value < self.alpha,
                    alpha_level=self.alpha,
                    effect_size=float(eta_squared),
                    degrees_of_freedom=df_between
                )
                
            except Exception as e:
                self.logger.warning(f"ANOVA failed for {col}: {e}")
        
        return anova_results
    
    def _perform_pairwise_comparisons(self, df: pd.DataFrame) -> Dict[str, List[PairwiseComparison]]:
        """Perform pairwise comparisons with multiple correction methods."""
        
        pairwise_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        models = list(df['model'].unique())
        
        if len(models) < 2:
            return pairwise_results
        
        for col in numeric_cols:
            comparisons = []
            p_values = []
            
            # Perform all pairwise comparisons
            for i, model1 in enumerate(models):
                for j, model2 in enumerate(models):
                    if i >= j:  # Avoid duplicate comparisons
                        continue
                    
                    group1 = df[df['model'] == model1][col].dropna()
                    group2 = df[df['model'] == model2][col].dropna()
                    
                    if len(group1) < 3 or len(group2) < 3:
                        continue
                    
                    # Check normality
                    _, p_norm1 = stats.shapiro(group1) if len(group1) <= 5000 else (None, 0.05)
                    _, p_norm2 = stats.shapiro(group2) if len(group2) <= 5000 else (None, 0.05)
                    
                    if p_norm1 > self.alpha and p_norm2 > self.alpha:
                        # Use t-test for normal data
                        t_stat, p_val = stats.ttest_ind(group1, group2)
                        test_stat = t_stat
                        
                        # Calculate Cohen's d
                        pooled_std = sqrt(((len(group1) - 1) * group1.var() + 
                                         (len(group2) - 1) * group2.var()) / 
                                        (len(group1) + len(group2) - 2))
                        cohens_d = (group1.mean() - group2.mean()) / pooled_std if pooled_std > 0 else 0
                        
                        # Confidence interval for difference in means
                        diff_mean = group1.mean() - group2.mean()
                        se_diff = pooled_std * sqrt(1/len(group1) + 1/len(group2))
                        df_pooled = len(group1) + len(group2) - 2
                        t_crit = stats.t.ppf(1 - self.alpha/2, df_pooled)
                        ci_lower = diff_mean - t_crit * se_diff
                        ci_upper = diff_mean + t_crit * se_diff
                        
                    else:
                        # Use Mann-Whitney U for non-normal data
                        u_stat, p_val = mannwhitneyu(group1, group2, alternative='two-sided')
                        test_stat = u_stat
                        
                        # Effect size for Mann-Whitney (r = Z / sqrt(N))
                        n1, n2 = len(group1), len(group2)
                        z_score = stats.norm.ppf(p_val/2) if p_val > 0 else 0
                        cohens_d = abs(z_score) / sqrt(n1 + n2)
                        
                        ci_lower, ci_upper = None, None
                    
                    comparison = PairwiseComparison(
                        groups=(model1, model2),
                        test_statistic=float(test_stat),
                        p_value=float(p_val),
                        adjusted_p_value=float(p_val),  # Will be corrected later
                        is_significant=p_val < self.alpha,
                        effect_size=float(cohens_d),
                        confidence_interval=(ci_lower, ci_upper) if ci_lower is not None else None
                    )
                    
                    comparisons.append(comparison)
                    p_values.append(p_val)
            
            # Apply multiple comparison corrections
            if p_values:
                # Bonferroni correction
                bonferroni_rejected, bonferroni_pvals, _, _ = multipletests(p_values, method='bonferroni')
                
                # Holm correction
                holm_rejected, holm_pvals, _, _ = multipletests(p_values, method='holm')
                
                # Benjamini-Hochberg (FDR) correction
                fdr_rejected, fdr_pvals, _, _ = multipletests(p_values, method='fdr_bh')
                
                # Update comparisons with corrected p-values
                for i, comparison in enumerate(comparisons):
                    comparison.adjusted_p_value = float(bonferroni_pvals[i])
                    comparison.is_significant = bonferroni_rejected[i]
            
            pairwise_results[col] = comparisons
        
        return pairwise_results
    
    def _perform_nonparametric_tests(self, df: pd.DataFrame) -> Dict[str, StatisticalTestResult]:
        """Perform non-parametric tests for ordinal data."""
        
        nonparametric_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        models = df['model'].unique()
        
        if len(models) < 2:
            return nonparametric_results
        
        for col in numeric_cols:
            groups = [df[df['model'] == model][col].dropna() for model in models]
            groups = [group for group in groups if len(group) >= 3]
            
            if len(groups) < 2:
                continue
            
            try:
                if len(groups) == 2:
                    # Mann-Whitney U test for two groups
                    u_stat, p_value = mannwhitneyu(groups[0], groups[1], alternative='two-sided')
                    test_name = "Mann-Whitney U"
                    test_stat = u_stat
                    
                    # Effect size (r = Z / sqrt(N))
                    n_total = len(groups[0]) + len(groups[1])
                    z_score = abs(stats.norm.ppf(p_value/2)) if p_value > 0 else 0
                    effect_size = z_score / sqrt(n_total)
                    
                else:
                    # Kruskal-Wallis test for multiple groups
                    h_stat, p_value = kruskal(*groups)
                    test_name = "Kruskal-Wallis"
                    test_stat = h_stat
                    
                    # Effect size (eta-squared analog for Kruskal-Wallis)
                    n_total = sum(len(group) for group in groups)
                    effect_size = (h_stat - len(groups) + 1) / (n_total - len(groups))
                
                nonparametric_results[col] = StatisticalTestResult(
                    test_name=test_name,
                    test_statistic=float(test_stat),
                    p_value=float(p_value),
                    is_significant=p_value < self.alpha,
                    alpha_level=self.alpha,
                    effect_size=float(effect_size)
                )
                
            except Exception as e:
                self.logger.warning(f"Non-parametric test failed for {col}: {e}")
        
        return nonparametric_results
    
    def _calculate_power_analysis(self, df: pd.DataFrame) -> Dict[str, Dict[str, float]]:
        """Perform power analysis for the tests."""
        
        power_results = {}
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        models = df['model'].unique()
        
        if len(models) != 2:  # Power analysis primarily for two-group comparisons
            return power_results
        
        model1, model2 = models
        
        for col in numeric_cols:
            group1 = df[df['model'] == model1][col].dropna()
            group2 = df[df['model'] == model2][col].dropna()
            
            if len(group1) < 3 or len(group2) < 3:
                continue
            
            try:
                # Calculate effect size (Cohen's d)
                pooled_std = sqrt(((len(group1) - 1) * group1.var() + 
                                 (len(group2) - 1) * group2.var()) / 
                                (len(group1) + len(group2) - 2))
                
                if pooled_std > 0:
                    cohens_d = abs(group1.mean() - group2.mean()) / pooled_std
                    
                    # Calculate observed power
                    observed_power = ttest_power(cohens_d, len(group1), self.alpha, alternative='two-sided')
                    
                    # Calculate required sample size for 80% power
                    try:
                        from statsmodels.stats.power import ttest_power, tt_solve_power
                        required_n = tt_solve_power(cohens_d, power=0.8, alpha=self.alpha, alternative='two-sided')
                    except:
                        required_n = None
                    
                    power_results[col] = {
                        "effect_size": float(cohens_d),
                        "observed_power": float(observed_power),
                        "required_n_for_80_power": float(required_n) if required_n else None,
                        "current_n1": len(group1),
                        "current_n2": len(group2)
                    }
            
            except Exception as e:
                self.logger.warning(f"Power analysis failed for {col}: {e}")
        
        return power_results
    
    def _calculate_comprehensive_effect_sizes(
        self, 
        df: pd.DataFrame, 
        anova_results: Dict[str, StatisticalTestResult],
        pairwise_comparisons: Dict[str, List[PairwiseComparison]]
    ) -> Dict[str, Dict[str, float]]:
        """Calculate comprehensive effect sizes."""
        
        effect_sizes = {}
        
        # Effect sizes from ANOVA (eta-squared)
        for metric, result in anova_results.items():
            if metric not in effect_sizes:
                effect_sizes[metric] = {}
            effect_sizes[metric]['eta_squared'] = result.effect_size or 0.0
        
        # Effect sizes from pairwise comparisons (Cohen's d)
        for metric, comparisons in pairwise_comparisons.items():
            if metric not in effect_sizes:
                effect_sizes[metric] = {}
            
            # Average effect size across all pairwise comparisons
            effect_size_values = [comp.effect_size for comp in comparisons if comp.effect_size is not None]
            if effect_size_values:
                effect_sizes[metric]['average_cohens_d'] = np.mean(effect_size_values)
                effect_sizes[metric]['max_cohens_d'] = np.max(effect_size_values)
        
        return effect_sizes
    
    def _apply_multiple_comparison_corrections(
        self, 
        pairwise_comparisons: Dict[str, List[PairwiseComparison]]
    ) -> Dict[str, Dict[str, Any]]:
        """Apply multiple comparison corrections."""
        
        correction_results = {}
        
        for metric, comparisons in pairwise_comparisons.items():
            if not comparisons:
                continue
            
            p_values = [comp.p_value for comp in comparisons]
            
            # Apply different correction methods
            corrections = {}
            
            # Bonferroni correction
            bonf_rejected, bonf_pvals, _, bonf_alpha = multipletests(p_values, method='bonferroni')
            corrections['bonferroni'] = {
                'adjusted_p_values': bonf_pvals.tolist(),
                'rejected': bonf_rejected.tolist(),
                'alpha_corrected': bonf_alpha
            }
            
            # Holm correction
            holm_rejected, holm_pvals, _, holm_alpha = multipletests(p_values, method='holm')
            corrections['holm'] = {
                'adjusted_p_values': holm_pvals.tolist(),
                'rejected': holm_rejected.tolist(),
                'alpha_corrected': holm_alpha
            }
            
            # Benjamini-Hochberg (FDR) correction
            fdr_rejected, fdr_pvals, _, fdr_alpha = multipletests(p_values, method='fdr_bh')
            corrections['benjamini_hochberg'] = {
                'adjusted_p_values': fdr_pvals.tolist(),
                'rejected': fdr_rejected.tolist(),
                'alpha_corrected': fdr_alpha
            }
            
            correction_results[metric] = corrections
        
        return correction_results
    
    def _generate_comprehensive_recommendations(
        self,
        descriptive_stats: Dict[str, Dict[str, DescriptiveStatistics]],
        anova_results: Dict[str, StatisticalTestResult],
        pairwise_comparisons: Dict[str, List[PairwiseComparison]],
        effect_sizes: Dict[str, Dict[str, float]]
    ) -> List[str]:
        """Generate comprehensive recommendations based on statistical analysis."""
        
        recommendations = []
        
        # ANOVA findings
        significant_anova = [metric for metric, result in anova_results.items() if result.is_significant]
        if significant_anova:
            recommendations.append(
                f"ANOVA detected significant differences in {len(significant_anova)} metrics: "
                f"{', '.join(significant_anova[:3])}{'...' if len(significant_anova) > 3 else ''}"
            )
        else:
            recommendations.append("No significant differences detected by ANOVA across models")
        
        # Effect size findings
        large_effects = []
        for metric, effects in effect_sizes.items():
            eta_sq = effects.get('eta_squared', 0)
            cohens_d = effects.get('max_cohens_d', 0)
            
            if eta_sq > 0.14 or cohens_d > 0.8:  # Large effect thresholds
                large_effects.append(metric)
        
        if large_effects:
            recommendations.append(
                f"Large effect sizes found in: {', '.join(large_effects[:3])}{'...' if len(large_effects) > 3 else ''}"
            )
        
        # Pairwise comparison findings
        significant_pairs = 0
        for comparisons in pairwise_comparisons.values():
            significant_pairs += sum(1 for comp in comparisons if comp.is_significant)
        
        if significant_pairs > 0:
            recommendations.append(f"Found {significant_pairs} significant pairwise comparisons")
        
        # Sample size recommendations
        min_sample_size = float('inf')
        for model_stats in descriptive_stats.values():
            for metric_stats in model_stats.values():
                min_sample_size = min(min_sample_size, metric_stats.count)
        
        if min_sample_size < 30:
            recommendations.append(
                f"Small sample sizes detected (minimum: {int(min_sample_size)}). "
                "Consider increasing sample size for more robust statistical inference."
            )
        
        # Data quality recommendations
        high_variance_metrics = []
        for model, model_stats in descriptive_stats.items():
            for metric, stats in model_stats.items():
                cv = stats.std / stats.mean if stats.mean != 0 else float('inf')
                if cv > 0.5:  # High coefficient of variation
                    high_variance_metrics.append(f"{metric} ({model})")
        
        if high_variance_metrics:
            recommendations.append(
                f"High variability detected in: {', '.join(high_variance_metrics[:3])}{'...' if len(high_variance_metrics) > 3 else ''}. "
                "Consider investigating sources of variation."
            )
        
        return recommendations
    
    
    def create_statistical_report(self, results: StatisticalResults) -> str:
        """Create a comprehensive formatted statistical analysis report."""
        
        report = ["Comprehensive Statistical Analysis Report", "=" * 50, ""]
        
        # Summary
        report.append("## Analysis Summary")
        report.append(f"- Analysis Date: {results.metadata['analysis_timestamp']}")
        report.append(f"- Models Compared: {results.metadata['total_comparisons']}")
        report.append(f"- Significance Level (α): {results.metadata['alpha_level']}")
        report.append(f"- Confidence Level: {results.metadata['confidence_level']}")
        report.append("")
        
        # ANOVA Results
        report.append("## ANOVA Results")
        significant_anova = {metric: test for metric, test in results.anova_results.items() if test.is_significant}
        
        if significant_anova:
            for metric, test in significant_anova.items():
                report.append(f"- {metric}: F = {test.test_statistic:.3f}, p = {test.p_value:.4f}, η² = {test.effect_size:.3f}")
        else:
            report.append("- No statistically significant ANOVA results")
        
        report.append("")
        
        # Pairwise Comparisons
        report.append("## Pairwise Comparisons (Significant Results)")
        significant_pairs_found = False
        
        for metric, comparisons in results.pairwise_comparisons.items():
            significant_comparisons = [comp for comp in comparisons if comp.is_significant]
            if significant_comparisons:
                significant_pairs_found = True
                report.append(f"### {metric}")
                for comp in significant_comparisons:
                    report.append(
                        f"  - {comp.groups[0]} vs {comp.groups[1]}: "
                        f"p = {comp.p_value:.4f}, d = {comp.effect_size:.3f}"
                    )
        
        if not significant_pairs_found:
            report.append("- No significant pairwise differences found")
        
        report.append("")
        
        # Non-parametric Tests
        report.append("## Non-parametric Test Results")
        significant_nonparam = {metric: test for metric, test in results.nonparametric_tests.items() if test.is_significant}
        
        if significant_nonparam:
            for metric, test in significant_nonparam.items():
                report.append(f"- {metric}: {test.test_name}, statistic = {test.test_statistic:.3f}, p = {test.p_value:.4f}")
        else:
            report.append("- No significant non-parametric test results")
        
        report.append("")
        
        # Effect Sizes Summary
        report.append("## Effect Sizes Summary")
        if results.effect_sizes:
            for metric, effects in results.effect_sizes.items():
                eta_sq = effects.get('eta_squared', 0)
                avg_d = effects.get('average_cohens_d', 0)
                max_d = effects.get('max_cohens_d', 0)
                
                report.append(f"- {metric}:")
                report.append(f"  - ANOVA η² = {eta_sq:.3f}")
                if avg_d > 0:
                    report.append(f"  - Average Cohen's d = {avg_d:.3f}")
                if max_d > 0:
                    report.append(f"  - Maximum Cohen's d = {max_d:.3f}")
        else:
            report.append("- No effect sizes calculated")
        
        report.append("")
        
        # Power Analysis
        report.append("## Power Analysis")
        if results.power_analysis:
            for metric, power_data in results.power_analysis.items():
                report.append(f"- {metric}:")
                report.append(f"  - Observed Power: {power_data['observed_power']:.3f}")
                if power_data.get('required_n_for_80_power'):
                    report.append(f"  - Sample size for 80% power: {power_data['required_n_for_80_power']:.0f}")
        else:
            report.append("- No power analysis performed")
        
        report.append("")
        
        # Multiple Comparison Corrections
        report.append("## Multiple Comparison Corrections")
        if results.multiple_comparison_corrections:
            report.append("- Correction methods applied: Bonferroni, Holm, Benjamini-Hochberg")
            for metric, corrections in results.multiple_comparison_corrections.items():
                bonf_sig = sum(corrections['bonferroni']['rejected'])
                holm_sig = sum(corrections['holm']['rejected'])
                fdr_sig = sum(corrections['benjamini_hochberg']['rejected'])
                report.append(f"- {metric}: Bonferroni: {bonf_sig}, Holm: {holm_sig}, FDR: {fdr_sig} significant")
        else:
            report.append("- No multiple comparison corrections applied")
        
        report.append("")
        
        # Recommendations
        report.append("## Statistical Recommendations")
        for rec in results.recommendations:
            report.append(f"- {rec}")
        
        report.append("")
        report.append("=" * 50)
        report.append("End of Statistical Analysis Report")
        
        return "\n".join(report)
    
    def export_results_to_csv(self, df: pd.DataFrame, filename: str) -> None:
        """Export analysis DataFrame to CSV."""
        try:
            df.to_csv(filename, index=False)
            self.logger.info(f"Exported analysis results to {filename}")
        except Exception as e:
            self.logger.error(f"Error exporting results to CSV: {e}")
            raise