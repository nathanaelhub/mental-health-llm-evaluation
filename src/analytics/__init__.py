"""
Advanced Analytics and Research Package

Comprehensive analytics, A/B testing, and research tools for the mental health
LLM evaluation system to provide deep insights into model selection performance
and user satisfaction patterns.
"""

from .ab_testing import ModelSelectionExperiment, ExperimentManager, ExperimentResult
from .feedback_system import FeedbackCollector, UserFeedback, FeedbackAnalyzer
from .smart_switching import SmartModelSwitcher, SwitchingDecision, ConversationOptimizer
from .research_tools import ResearchExporter, StatisticalAnalyzer, DataMiner
from .dashboard import AnalyticsDashboard, VisualizationEngine, MetricsCollector

__version__ = "1.0.0"

__all__ = [
    # A/B Testing
    "ModelSelectionExperiment",
    "ExperimentManager", 
    "ExperimentResult",
    
    # Feedback System
    "FeedbackCollector",
    "UserFeedback",
    "FeedbackAnalyzer",
    
    # Smart Switching
    "SmartModelSwitcher",
    "SwitchingDecision",
    "ConversationOptimizer",
    
    # Research Tools
    "ResearchExporter",
    "StatisticalAnalyzer",
    "DataMiner",
    
    # Dashboard
    "AnalyticsDashboard",
    "VisualizationEngine",
    "MetricsCollector"
]