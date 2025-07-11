"""
Mental Health LLM Evaluation Package

A comprehensive framework for evaluating and comparing local vs cloud-based 
Large Language Models in mental health telemedicine applications.
"""

__version__ = "1.0.0"
__author__ = "Mental Health LLM Research Team"
__email__ = "contact@mental-health-llm.org"

from .models import OpenAIClient, ClaudeClient, DeepSeekClient, GemmaClient, BaseModel, LocalLLMClient
from .evaluation import MentalHealthEvaluator, EvaluationMetrics
from .scenarios import ScenarioLoader, ConversationGenerator
from .analysis import StatisticalAnalyzer, ResultsVisualizer
from .utils import setup_logging, DataStorage

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
