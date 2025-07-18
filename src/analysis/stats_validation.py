"""
Statistical Results Validation and Structure Ensuring
====================================================

This module provides validation and structure ensuring functions for statistical
analysis results to prevent KeyError when accessing nested stats.
"""

from typing import Dict, List, Any, Optional
from dataclasses import dataclass


# Expected metrics in the stats
EXPECTED_METRICS = [
    'composite',
    'empathy',
    'clarity',
    'therapeutic',  # Primary field name for therapeutic effectiveness
    'safety',
    'professionalism',
    'engagement',
    'helpfulness'  # Legacy field name for backward compatibility
]

# Default stat structure for a single metric
DEFAULT_METRIC_STATS = {
    'mean': 0.0,
    'std_dev': 0.0,
    'median': 0.0,
    'min': 0.0,
    'max': 0.0,
    'count': 0
}

# Default comparison test result
DEFAULT_COMPARISON_TEST = {
    'p_value': 1.0,
    'is_significant': False,
    'effect_size': 0.0,
    'effect_interpretation': 'negligible',
    'test_statistic': 0.0,
    'test_name': 'unknown'
}


def ensure_stats_structure(stats: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
    """
    Ensure stats dictionary has all expected metrics with proper structure.
    
    Args:
        stats: Stats dictionary that may be missing keys or structure
        
    Returns:
        Complete stats dictionary with all expected metrics
    """
    if not stats:
        stats = {}
    
    # Ensure all expected metrics exist
    for metric in EXPECTED_METRICS:
        if metric not in stats:
            stats[metric] = DEFAULT_METRIC_STATS.copy()
        else:
            # Ensure the metric has all required stat fields
            stats[metric] = ensure_metric_stats(stats[metric])
    
    return stats


def ensure_metric_stats(metric_stats: Any) -> Dict[str, float]:
    """
    Ensure a single metric's stats has all required fields.
    
    Args:
        metric_stats: Stats for a single metric
        
    Returns:
        Complete metric stats dictionary
    """
    if not isinstance(metric_stats, dict):
        return DEFAULT_METRIC_STATS.copy()
    
    # Create a new dict with all required fields
    complete_stats = DEFAULT_METRIC_STATS.copy()
    
    # Update with existing values
    for key, default_value in DEFAULT_METRIC_STATS.items():
        if key in metric_stats:
            try:
                complete_stats[key] = float(metric_stats[key])
            except (ValueError, TypeError):
                complete_stats[key] = default_value
    
    return complete_stats


def ensure_comparison_tests_structure(tests: Optional[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Ensure comparison tests dictionary has all expected metrics.
    
    Args:
        tests: Comparison tests dictionary
        
    Returns:
        Complete comparison tests dictionary
    """
    if not tests:
        tests = {}
    
    # Ensure all expected metrics have test results
    for metric in EXPECTED_METRICS:
        if metric not in tests:
            tests[metric] = DEFAULT_COMPARISON_TEST.copy()
        else:
            # Ensure test result has all required fields
            tests[metric] = ensure_test_result(tests[metric])
    
    return tests


def ensure_test_result(test_result: Any) -> Dict[str, Any]:
    """
    Ensure a single test result has all required fields.
    
    Args:
        test_result: Test result for a single metric
        
    Returns:
        Complete test result dictionary
    """
    if not isinstance(test_result, dict):
        return DEFAULT_COMPARISON_TEST.copy()
    
    # Create a new dict with all required fields
    complete_test = DEFAULT_COMPARISON_TEST.copy()
    
    # Update with existing values
    for key, default_value in DEFAULT_COMPARISON_TEST.items():
        if key in test_result:
            complete_test[key] = test_result[key]
    
    return complete_test


def validate_analysis_structure(analysis: Any) -> bool:
    """
    Validate that an analysis object has the expected structure.
    
    Args:
        analysis: Analysis results object
        
    Returns:
        True if structure is valid, False otherwise
    """
    if not analysis:
        return False
    
    # Check required top-level attributes
    required_attrs = [
        'overall_winner',
        'confidence_level',
        'openai_stats',
        'deepseek_stats',
        'comparison_tests'
    ]
    
    for attr in required_attrs:
        if not hasattr(analysis, attr):
            return False
    
    # Check stats structure
    if not isinstance(analysis.openai_stats, dict) or not isinstance(analysis.deepseek_stats, dict):
        return False
    
    # Check that composite metric exists in stats
    if 'composite' not in analysis.openai_stats or 'composite' not in analysis.deepseek_stats:
        return False
    
    return True


def safe_get_stat(stats: Dict[str, Any], metric: str, stat_name: str, default: float = 0.0) -> float:
    """
    Safely get a statistic value from nested stats dictionary.
    
    Args:
        stats: Stats dictionary
        metric: Metric name (e.g., 'composite')
        stat_name: Statistic name (e.g., 'mean')
        default: Default value if not found
        
    Returns:
        The statistic value or default
    """
    try:
        if metric in stats and isinstance(stats[metric], dict):
            if stat_name in stats[metric]:
                return float(stats[metric][stat_name])
    except (TypeError, ValueError):
        pass
    
    return default


def fix_analysis_structure(analysis: Any) -> Any:
    """
    Fix an analysis object to ensure it has all required structure.
    
    Args:
        analysis: Analysis results object that may be missing structure
        
    Returns:
        Analysis object with ensured structure
    """
    if not analysis:
        return None
    
    # Ensure stats have proper structure
    if hasattr(analysis, 'openai_stats'):
        analysis.openai_stats = ensure_stats_structure(analysis.openai_stats)
    else:
        analysis.openai_stats = ensure_stats_structure({})
    
    if hasattr(analysis, 'deepseek_stats'):
        analysis.deepseek_stats = ensure_stats_structure(analysis.deepseek_stats)
    else:
        analysis.deepseek_stats = ensure_stats_structure({})
    
    # Ensure comparison tests have proper structure
    if hasattr(analysis, 'comparison_tests'):
        analysis.comparison_tests = ensure_comparison_tests_structure(analysis.comparison_tests)
    else:
        analysis.comparison_tests = ensure_comparison_tests_structure({})
    
    return analysis


def create_empty_stats() -> Dict[str, Dict[str, float]]:
    """
    Create an empty but properly structured stats dictionary.
    
    Returns:
        Stats dictionary with all metrics initialized to default values
    """
    stats = {}
    for metric in EXPECTED_METRICS:
        stats[metric] = DEFAULT_METRIC_STATS.copy()
    return stats


def create_empty_comparison_tests() -> Dict[str, Dict[str, Any]]:
    """
    Create an empty but properly structured comparison tests dictionary.
    
    Returns:
        Comparison tests dictionary with all metrics initialized
    """
    tests = {}
    for metric in EXPECTED_METRICS:
        tests[metric] = DEFAULT_COMPARISON_TEST.copy()
    return tests


@dataclass
class SafeStatAccess:
    """Helper class for safe access to nested statistics."""
    
    stats: Dict[str, Any]
    
    def get(self, metric: str, stat_name: str = 'mean', default: float = 0.0) -> float:
        """
        Get a stat value safely.
        
        Args:
            metric: Metric name (e.g., 'composite')
            stat_name: Stat name (e.g., 'mean', 'std_dev')
            default: Default value if not found
            
        Returns:
            The stat value or default
        """
        return safe_get_stat(self.stats, metric, stat_name, default)
    
    def __getitem__(self, metric: str) -> Dict[str, float]:
        """Allow dict-style access with guaranteed structure."""
        if metric not in self.stats:
            return DEFAULT_METRIC_STATS.copy()
        return ensure_metric_stats(self.stats[metric])


# Example usage functions
def safe_display_stats(analysis: Any) -> None:
    """
    Safely display stats from analysis results.
    
    Args:
        analysis: Analysis results that may have missing structure
    """
    # Fix structure first
    analysis = fix_analysis_structure(analysis)
    
    if not analysis:
        print("No analysis results available")
        return
    
    # Now safe to access nested keys
    openai_composite_mean = safe_get_stat(analysis.openai_stats, 'composite', 'mean')
    deepseek_composite_mean = safe_get_stat(analysis.deepseek_stats, 'composite', 'mean')
    
    print(f"OpenAI Composite: {openai_composite_mean:.2f}")
    print(f"DeepSeek Composite: {deepseek_composite_mean:.2f}")
    
    # Or use SafeStatAccess wrapper
    openai_safe = SafeStatAccess(analysis.openai_stats)
    deepseek_safe = SafeStatAccess(analysis.deepseek_stats)
    
    print(f"OpenAI Empathy: {openai_safe.get('empathy'):.2f}")
    print(f"DeepSeek Empathy: {deepseek_safe.get('empathy'):.2f}")


if __name__ == "__main__":
    # Test the validation functions
    print("Testing stats validation functions...")
    
    # Test with empty stats
    empty_stats = {}
    fixed_stats = ensure_stats_structure(empty_stats)
    print(f"\nEmpty stats fixed: {list(fixed_stats.keys())}")
    
    # Test with partial stats
    partial_stats = {
        'composite': {'mean': 8.5, 'std_dev': 1.2},
        'empathy': {'mean': 7.0}  # Missing other fields
    }
    fixed_partial = ensure_stats_structure(partial_stats)
    print(f"\nPartial stats fixed:")
    print(f"  Composite: {fixed_partial['composite']}")
    print(f"  Empathy: {fixed_partial['empathy']}")
    print(f"  Safety (was missing): {fixed_partial['safety']}")
    
    # Test safe access
    safe_stats = SafeStatAccess(fixed_partial)
    print(f"\nSafe access tests:")
    print(f"  Composite mean: {safe_stats.get('composite', 'mean')}")
    print(f"  Missing metric: {safe_stats.get('nonexistent', 'mean')}")
    print(f"  Dict access: {safe_stats['composite']['mean']}")