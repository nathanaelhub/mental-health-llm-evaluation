"""
Analysis and visualization tools for mental health LLM evaluation.

This module provides statistical analysis and visualization capabilities
for comparing and presenting evaluation results.
"""

try:
    from .statistical_analysis import StatisticalAnalyzer, StatisticalResults
except ImportError:
    # Handle missing statistical analysis components
    StatisticalAnalyzer = None
    StatisticalResults = None

try:
    from .visualization import SafeVisualizer, VisualizationConfig, create_all_visualizations
    # For backward compatibility
    ResultsVisualizer = SafeVisualizer
except ImportError:
    # Handle missing visualization components
    SafeVisualizer = None
    VisualizationConfig = None
    ResultsVisualizer = None
    create_all_visualizations = None

__all__ = [
    "StatisticalAnalyzer",
    "StatisticalResults",
    "SafeVisualizer",
    "ResultsVisualizer",  # Backward compatibility
    "VisualizationConfig",
    "create_all_visualizations",
]