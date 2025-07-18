"""
Mental Health LLM Evaluation Package

A comprehensive framework for evaluating and comparing local vs cloud-based 
Large Language Models in mental health telemedicine applications.
"""

__version__ = "1.0.0"
__author__ = "Mental Health LLM Research Team"
__email__ = "contact@mental-health-llm.org"

# Use safer imports with error handling
try:
    from .models import OpenAIClient, ClaudeClient, DeepSeekClient, GemmaClient, BaseModel, LocalLLMClient
except ImportError:
    # Handle missing model components
    OpenAIClient = ClaudeClient = DeepSeekClient = GemmaClient = BaseModel = LocalLLMClient = None

try:
    from .evaluation import MentalHealthEvaluator, EvaluationMetrics
except ImportError:
    MentalHealthEvaluator = EvaluationMetrics = None

try:
    from .scenarios import ScenarioLoader, ConversationGenerator
except ImportError:
    ScenarioLoader = ConversationGenerator = None

try:
    from .analysis import StatisticalAnalyzer, ResultsVisualizer
except ImportError:
    StatisticalAnalyzer = ResultsVisualizer = None

try:
    from .utils import setup_logging, DataStorage
except ImportError:
    setup_logging = DataStorage = None

__all__ = [
    "OpenAIClient",
    "ClaudeClient",
    "DeepSeekClient",
    "GemmaClient",
    "BaseModel",
    "LocalLLMClient",
    "MentalHealthEvaluator",
    "EvaluationMetrics",
    "ScenarioLoader",
    "ConversationGenerator",
    "StatisticalAnalyzer",
    "ResultsVisualizer",
    "setup_logging",
    "DataStorage",
]
