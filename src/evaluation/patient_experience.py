"""
Patient experience metrics for mental health LLM evaluation.

This module evaluates user satisfaction, engagement, trust, and accessibility
aspects of mental health conversations.
"""

import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from textstat import flesch_reading_ease, flesch_kincaid_grade
import logging

from ..models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class PatientScore:
    """Patient experience score for a model."""
    
    satisfaction_score: float   # 0-100, user-reported satisfaction
    engagement_score: float     # 0-100, conversation depth and length
    trust_score: float         # 0-100, confidence in responses
    accessibility_score: float # 0-100, ease of understanding
    overall_score: float       # 0-100, weighted average
    
    satisfaction_metrics: Dict[str, float]
    engagement_metrics: Dict[str, float]
    trust_metrics: Dict[str, float]
    accessibility_metrics: Dict[str, float]
    
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "satisfaction_score": self.satisfaction_score,
            "engagement_score": self.engagement_score,
            "trust_score": self.trust_score,
            "accessibility_score": self.accessibility_score,
            "overall_score": self.overall_score,
            "satisfaction_metrics": self.satisfaction_metrics,
            "engagement_metrics": self.engagement_metrics,
            "trust_metrics": self.trust_metrics,
            "accessibility_metrics": self.accessibility_metrics,
            "metadata": self.metadata
        }


class PatientExperience:
    """Patient experience evaluator for mental health LLM models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize patient experience evaluator.
        
        Args:
            config: Configuration for evaluation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Load evaluation criteria
        self._load_evaluation_criteria()
    
    def _load_evaluation_criteria(self) -> None:
        """Load criteria for patient experience evaluation."""
        
        # Trust indicators
        self.trust_indicators = {
            "confident_language": [
                "research shows", "studies indicate", "evidence suggests",
                "professionals recommend", "commonly used approach"
            ],
            "uncertainty_acknowledgment": [
                "I'm not sure", "I don't know", "I may be wrong",
                "please check with", "consider consulting"
            ],
            "qualified_statements": [
                "in my understanding", "from what I know", "generally speaking",
                "it's possible that", "one perspective is"
            ]
        }
        
        # Engagement quality indicators
        self.engagement_indicators = {
            "questions": [
                "how do you feel", "what do you think", "can you tell me more",
                "how has this been", "what would help", "how are you coping"
            ],
            "personalization": [
                "for you", "in your situation", "given what you've shared",
                "based on what you've told me", "your experience"
            ],
            "follow_up": [
                "let's explore", "tell me more about", "can we talk about",
                "how did that make you feel", "what happened next"
            ]
        }
        
        # Accessibility markers
        self.accessibility_markers = {
            "clear_explanations": [
                "in other words", "what this means is", "to put it simply",
                "for example", "let me explain", "in simpler terms"
            ],
            "jargon_avoidance": [
                # These are medical/psychological terms that should be explained
                "cognitive behavioral", "mindfulness", "coping mechanisms",
                "therapeutic", "intervention", "symptoms"
            ]
        }
    
    async def evaluate_model(
        self,
        model: BaseModel,
        conversation_data: List[Dict[str, Any]],
        user_feedback: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> PatientScore:
        """
        Comprehensive patient experience evaluation.
        
        Args:
            model: Model to evaluate
            conversation_data: Conversation transcripts and metadata
            user_feedback: Optional user satisfaction ratings and feedback
            **kwargs: Additional evaluation parameters
            
        Returns:
            PatientScore with all metrics
        """
        self.logger.info(f"Starting patient experience evaluation for {model.model_name}")
        
        # Evaluate different aspects
        satisfaction_metrics = await self._evaluate_satisfaction(conversation_data, user_feedback)
        engagement_metrics = await self._evaluate_engagement(conversation_data)
        trust_metrics = await self._evaluate_trust(conversation_data)
        accessibility_metrics = await self._evaluate_accessibility(conversation_data)
        
        # Calculate overall score
        score = self._calculate_patient_score(
            satisfaction_metrics,
            engagement_metrics,
            trust_metrics,
            accessibility_metrics
        )
        
        self.logger.info(
            f"Patient experience evaluation complete for {model.model_name}: "
            f"Overall Score: {score.overall_score:.2f}"
        )
        
        return score
    
    async def _evaluate_satisfaction(
        self,
        conversation_data: List[Dict[str, Any]],
        user_feedback: Optional[List[Dict[str, Any]]] = None
    ) -> Dict[str, float]:
        """Evaluate user satisfaction metrics."""
        self.logger.info("Evaluating satisfaction...")
        
        satisfaction_scores = []
        helpfulness_scores = []
        comfort_scores = []
        recommendation_scores = []
        
        # If direct user feedback is available, use it
        if user_feedback:
            for feedback in user_feedback:
                if feedback.get("satisfaction_rating"):
                    satisfaction_scores.append(feedback["satisfaction_rating"] * 20)  # Convert 1-5 to 0-100
                if feedback.get("helpfulness_rating"):
                    helpfulness_scores.append(feedback["helpfulness_rating"] * 20)
                if feedback.get("comfort_rating"):
                    comfort_scores.append(feedback["comfort_rating"] * 20)
                if feedback.get("would_recommend"):
                    recommendation_scores.append(100 if feedback["would_recommend"] else 0)
        
        # Infer satisfaction from conversation characteristics
        for conversation in conversation_data:
            inferred_satisfaction = self._infer_satisfaction_from_conversation(conversation)
            satisfaction_scores.append(inferred_satisfaction)
        
        return {
            "overall_satisfaction": np.mean(satisfaction_scores) if satisfaction_scores else 0,
            "helpfulness": np.mean(helpfulness_scores) if helpfulness_scores else 0,
            "comfort_level": np.mean(comfort_scores) if comfort_scores else 0,
            "recommendation_likelihood": np.mean(recommendation_scores) if recommendation_scores else 0,
            "feedback_count": len(user_feedback) if user_feedback else 0,
            "conversation_count": len(conversation_data)
        }
    
    def _infer_satisfaction_from_conversation(self, conversation: Dict[str, Any]) -> float:
        """Infer satisfaction level from conversation characteristics."""
        try:
            messages = conversation.get("messages", [])
            if not messages:
                return 50  # Neutral score
            
            score = 60  # Base score
            
            # Conversation length indicates engagement
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            if len(user_messages) > 5:
                score += 15  # Good engagement
            elif len(user_messages) > 10:
                score += 25  # Excellent engagement
            
            # Look for positive indicators in user messages
            user_text = " ".join([msg.get("content", "") for msg in user_messages]).lower()
            
            positive_indicators = [
                "thank you", "helpful", "better", "understand", "good",
                "appreciate", "makes sense", "feel better"
            ]
            
            negative_indicators = [
                "not helpful", "confused", "worse", "don't understand",
                "frustrated", "annoying", "wrong"
            ]
            
            positive_count = sum(1 for indicator in positive_indicators if indicator in user_text)
            negative_count = sum(1 for indicator in negative_indicators if indicator in user_text)
            
            score += positive_count * 5
            score -= negative_count * 10
            
            # Check if conversation ended abruptly (potential dissatisfaction)
            if len(user_messages) < 3:
                score -= 15
            
            return max(0, min(100, score))
        
        except Exception as e:
            self.logger.warning(f"Error inferring satisfaction: {e}")
            return 50
    
    async def _evaluate_engagement(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate engagement quality metrics."""
        self.logger.info("Evaluating engagement...")
        
        engagement_scores = []
        depth_scores = []
        length_scores = []
        interaction_quality_scores = []
        
        for conversation in conversation_data:
            messages = conversation.get("messages", [])
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            
            if not assistant_messages:
                continue
            
            # Conversation depth (number of exchanges)
            exchange_count = min(len(assistant_messages), len(user_messages))
            depth_score = min(100, exchange_count * 10)  # Max at 10 exchanges
            depth_scores.append(depth_score)
            
            # Response length appropriateness
            avg_response_length = np.mean([
                len(msg.get("content", "").split()) 
                for msg in assistant_messages
            ])
            
            # Ideal response length is 30-100 words
            if 30 <= avg_response_length <= 100:
                length_score = 100
            elif avg_response_length < 30:
                length_score = max(0, 100 - (30 - avg_response_length) * 2)
            else:
                length_score = max(0, 100 - (avg_response_length - 100) * 0.5)
            
            length_scores.append(length_score)
            
            # Interaction quality (questions, personalization, follow-ups)
            interaction_score = self._evaluate_interaction_quality(assistant_messages)
            interaction_quality_scores.append(interaction_score)
            
            # Overall engagement for this conversation
            conversation_engagement = np.mean([depth_score, length_score, interaction_score])
            engagement_scores.append(conversation_engagement)
        
        return {
            "overall_engagement": np.mean(engagement_scores) if engagement_scores else 0,
            "conversation_depth": np.mean(depth_scores) if depth_scores else 0,
            "response_length_quality": np.mean(length_scores) if length_scores else 0,
            "interaction_quality": np.mean(interaction_quality_scores) if interaction_quality_scores else 0,
            "conversation_count": len(conversation_data)
        }
    
    def _evaluate_interaction_quality(self, assistant_messages: List[Dict[str, str]]) -> float:
        """Evaluate quality of interactions in assistant messages."""
        try:
            all_text = " ".join([msg.get("content", "") for msg in assistant_messages]).lower()
            
            # Count engagement indicators
            question_count = sum(
                1 for phrase in self.engagement_indicators["questions"]
                if phrase in all_text
            )
            
            personalization_count = sum(
                1 for phrase in self.engagement_indicators["personalization"]
                if phrase in all_text
            )
            
            follow_up_count = sum(
                1 for phrase in self.engagement_indicators["follow_up"]
                if phrase in all_text
            )
            
            # Score based on presence of engagement elements
            question_score = min(30, question_count * 10)
            personalization_score = min(40, personalization_count * 8)
            follow_up_score = min(30, follow_up_count * 15)
            
            return question_score + personalization_score + follow_up_score
        
        except Exception as e:
            self.logger.warning(f"Error evaluating interaction quality: {e}")
            return 50
    
    async def _evaluate_trust(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate trust and confidence metrics."""
        self.logger.info("Evaluating trust...")
        
        trust_scores = []
        confidence_scores = []
        transparency_scores = []
        
        for conversation in conversation_data:
            messages = conversation.get("messages", [])
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            
            if not assistant_messages:
                continue
            
            all_text = " ".join([msg.get("content", "") for msg in assistant_messages]).lower()
            
            # Confidence indicators
            confident_count = sum(
                1 for phrase in self.trust_indicators["confident_language"]
                if phrase in all_text
            )
            
            # Appropriate uncertainty acknowledgment
            uncertainty_count = sum(
                1 for phrase in self.trust_indicators["uncertainty_acknowledgment"]
                if phrase in all_text
            )
            
            # Qualified statements (showing humility)
            qualified_count = sum(
                1 for phrase in self.trust_indicators["qualified_statements"]
                if phrase in all_text
            )
            
            # Balance of confidence and humility
            confidence_score = min(50, confident_count * 10)  # Max 50 points for confidence
            transparency_score = min(50, (uncertainty_count + qualified_count) * 8)  # Max 50 for transparency
            
            trust_score = confidence_score + transparency_score
            
            trust_scores.append(trust_score)
            confidence_scores.append(confidence_score)
            transparency_scores.append(transparency_score)
        
        return {
            "overall_trust": np.mean(trust_scores) if trust_scores else 0,
            "confidence_level": np.mean(confidence_scores) if confidence_scores else 0,
            "transparency": np.mean(transparency_scores) if transparency_scores else 0,
            "conversation_count": len(trust_scores)
        }
    
    async def _evaluate_accessibility(self, conversation_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate accessibility and ease of understanding."""
        self.logger.info("Evaluating accessibility...")
        
        accessibility_scores = []
        readability_scores = []
        clarity_scores = []
        
        for conversation in conversation_data:
            messages = conversation.get("messages", [])
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            
            if not assistant_messages:
                continue
            
            # Combine all assistant text
            all_text = " ".join([msg.get("content", "") for msg in assistant_messages])
            
            # Readability analysis
            readability_score = self._evaluate_readability(all_text)
            readability_scores.append(readability_score)
            
            # Clarity indicators
            clarity_score = self._evaluate_clarity(all_text)
            clarity_scores.append(clarity_score)
            
            # Overall accessibility
            conversation_accessibility = np.mean([readability_score, clarity_score])
            accessibility_scores.append(conversation_accessibility)
        
        return {
            "overall_accessibility": np.mean(accessibility_scores) if accessibility_scores else 0,
            "readability": np.mean(readability_scores) if readability_scores else 0,
            "clarity": np.mean(clarity_scores) if clarity_scores else 0,
            "conversation_count": len(accessibility_scores)
        }
    
    def _evaluate_readability(self, text: str) -> float:
        """Evaluate text readability using standard metrics."""
        try:
            if len(text.strip()) < 10:
                return 50  # Neutral score for very short text
            
            # Flesch Reading Ease (higher = easier)
            flesch_score = flesch_reading_ease(text)
            
            # Convert to 0-100 scale where higher is better
            if flesch_score >= 70:  # Easy to read
                readability_score = 100
            elif flesch_score >= 60:  # Standard
                readability_score = 80
            elif flesch_score >= 50:  # Fairly difficult
                readability_score = 60
            elif flesch_score >= 30:  # Difficult
                readability_score = 40
            else:  # Very difficult
                readability_score = 20
            
            return readability_score
        
        except Exception as e:
            self.logger.warning(f"Readability evaluation failed: {e}")
            return 50
    
    def _evaluate_clarity(self, text: str) -> float:
        """Evaluate clarity of explanations and language."""
        try:
            text_lower = text.lower()
            
            # Look for clear explanation markers
            explanation_count = sum(
                1 for phrase in self.accessibility_markers["clear_explanations"]
                if phrase in text_lower
            )
            
            # Check for unexplained jargon
            jargon_count = sum(
                1 for term in self.accessibility_markers["jargon_avoidance"]
                if term in text_lower
            )
            
            # Sentence length analysis
            sentences = text.split('.')
            if sentences:
                avg_sentence_length = np.mean([len(sentence.split()) for sentence in sentences if sentence.strip()])
                
                # Ideal sentence length is 15-20 words
                if avg_sentence_length <= 20:
                    sentence_score = 100
                else:
                    sentence_score = max(0, 100 - (avg_sentence_length - 20) * 2)
            else:
                sentence_score = 50
            
            # Calculate clarity score
            explanation_score = min(40, explanation_count * 20)  # Bonus for explanations
            jargon_penalty = min(30, jargon_count * 10)  # Penalty for unexplained jargon
            
            clarity_score = sentence_score + explanation_score - jargon_penalty
            
            return max(0, min(100, clarity_score))
        
        except Exception as e:
            self.logger.warning(f"Clarity evaluation failed: {e}")
            return 50
    
    def _calculate_patient_score(
        self,
        satisfaction_metrics: Dict[str, float],
        engagement_metrics: Dict[str, float],
        trust_metrics: Dict[str, float],
        accessibility_metrics: Dict[str, float]
    ) -> PatientScore:
        """Calculate overall patient experience score."""
        
        satisfaction_score = satisfaction_metrics["overall_satisfaction"]
        engagement_score = engagement_metrics["overall_engagement"]
        trust_score = trust_metrics["overall_trust"]
        accessibility_score = accessibility_metrics["overall_accessibility"]
        
        # Weighted average
        weights = self.config.get("score_weights", {
            "satisfaction": 0.4,
            "engagement": 0.25,
            "trust": 0.25,
            "accessibility": 0.1
        })
        
        overall_score = (
            satisfaction_score * weights["satisfaction"] +
            engagement_score * weights["engagement"] +
            trust_score * weights["trust"] +
            accessibility_score * weights["accessibility"]
        )
        
        return PatientScore(
            satisfaction_score=satisfaction_score,
            engagement_score=engagement_score,
            trust_score=trust_score,
            accessibility_score=accessibility_score,
            overall_score=overall_score,
            satisfaction_metrics=satisfaction_metrics,
            engagement_metrics=engagement_metrics,
            trust_metrics=trust_metrics,
            accessibility_metrics=accessibility_metrics,
            metadata={
                "evaluation_config": self.config,
                "evaluation_criteria": {
                    "trust_indicators": len(self.trust_indicators),
                    "engagement_indicators": len(self.engagement_indicators),
                    "accessibility_markers": len(self.accessibility_markers)
                }
            }
        )