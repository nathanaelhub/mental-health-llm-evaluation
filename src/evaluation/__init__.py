"""
Evaluation framework for mental health LLM comparison.

This module provides comprehensive evaluation metrics for comparing
local vs cloud-based LLMs in therapeutic conversation contexts.
"""

from .technical_metrics import TechnicalMetrics, TechnicalScore
from .therapeutic_metrics import TherapeuticMetrics, TherapeuticScore
from .patient_experience import PatientExperience, PatientScore
from .composite_scorer import CompositeScorer, CompositeScore

__all__ = [
    "TechnicalMetrics",
    "TechnicalScore", 
    "TherapeuticMetrics",
    "TherapeuticScore",
    "PatientExperience",
    "PatientScore",
    "CompositeScorer",
    "CompositeScore",
]