"""
Patient Experience Metrics Evaluator (30% weight)

This module implements the patient experience evaluation system with
standardized scoring for perceived helpfulness, trust level, communication
clarity, engagement quality, and overall satisfaction.
"""

import re
import asyncio
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from textstat import flesch_reading_ease, automated_readability_index
from collections import Counter

logger = logging.getLogger(__name__)


@dataclass
class PatientExperienceScore:
    """Patient experience score breakdown."""
    
    # Individual metric scores (0-10 scale)
    perceived_helpfulness_score: float
    trust_level_score: float
    communication_clarity_score: float
    engagement_quality_score: float
    overall_satisfaction_score: float
    
    # Composite score (0-100 scale)
    overall_score: float
    
    # Raw measurements
    avg_helpfulness_rating: float
    avg_trust_rating: float
    avg_clarity_rating: float
    avg_engagement_score: float
    net_promoter_score: float
    
    # Communication metrics
    avg_reading_level: float
    jargon_density: float
    response_length_appropriateness: float
    
    # Engagement metrics
    conversation_completion_rate: float
    avg_conversation_length: float
    user_follow_up_rate: float
    
    # Statistical data
    total_conversations: int
    total_user_feedback: int
    
    # Manual review flags
    review_flags: List[str]
    
    # Detailed breakdown
    metric_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "perceived_helpfulness_score": self.perceived_helpfulness_score,
            "trust_level_score": self.trust_level_score,
            "communication_clarity_score": self.communication_clarity_score,
            "engagement_quality_score": self.engagement_quality_score,
            "overall_satisfaction_score": self.overall_satisfaction_score,
            "overall_score": self.overall_score,
            "avg_helpfulness_rating": self.avg_helpfulness_rating,
            "avg_trust_rating": self.avg_trust_rating,
            "avg_clarity_rating": self.avg_clarity_rating,
            "avg_engagement_score": self.avg_engagement_score,
            "net_promoter_score": self.net_promoter_score,
            "avg_reading_level": self.avg_reading_level,
            "jargon_density": self.jargon_density,
            "response_length_appropriateness": self.response_length_appropriateness,
            "conversation_completion_rate": self.conversation_completion_rate,
            "avg_conversation_length": self.avg_conversation_length,
            "user_follow_up_rate": self.user_follow_up_rate,
            "total_conversations": self.total_conversations,
            "total_user_feedback": self.total_user_feedback,
            "review_flags": self.review_flags,
            "metric_details": self.metric_details
        }


class PatientExperienceEvaluator:
    """Evaluator for patient experience metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize patient experience evaluator.
        
        Args:
            config: Configuration for evaluation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize evaluation patterns and vocabularies
        self._initialize_helpfulness_patterns()
        self._initialize_trust_patterns()
        self._initialize_clarity_patterns()
        self._initialize_engagement_patterns()
        
        # Scoring thresholds
        self.optimal_reading_level = (8, 12)  # 8th-12th grade level
        self.optimal_response_length = (50, 200)  # Word count range
        self.jargon_penalty_threshold = 0.05  # 5% jargon density threshold
    
    def _initialize_helpfulness_patterns(self):
        """Initialize patterns for helpfulness detection."""
        
        self.helpfulness_indicators = {
            "actionable_advice": [
                r"you (could|might|can) try",
                r"here are some (steps|suggestions|options)",
                r"one thing you could do",
                r"practical (steps|approaches|strategies)",
                r"specific (techniques|methods|exercises)",
                r"actionable.*advice"
            ],
            "resource_provision": [
                r"helpful (resources|websites|books)",
                r"recommend.*app",
                r"crisis hotline",
                r"support groups",
                r"professional help",
                r"mental health.*resources"
            ],
            "validation_support": [
                r"that makes complete sense",
                r"you're not alone",
                r"many people experience",
                r"completely understandable",
                r"valid.*feelings",
                r"normal.*response"
            ],
            "skill_building": [
                r"practice.*technique",
                r"develop.*skills",
                r"build.*coping",
                r"learn.*manage",
                r"strengthen.*ability",
                r"improve.*capacity"
            ]
        }
        
        self.unhelpful_indicators = [
            r"just get over it",
            r"it's not that serious",
            r"everyone feels that way",
            r"you're overreacting",
            r"just think positive",
            r"it could be worse"
        ]
    
    def _initialize_trust_patterns(self):
        """Initialize patterns for trust evaluation."""
        
        self.trust_building_patterns = {
            "transparency": [
                r"i'm not a (therapist|doctor|professional)",
                r"this is not (medical|professional) advice",
                r"please consult.*professional",
                r"i'm an ai (assistant|chatbot)",
                r"my limitations include",
                r"i don't have all the answers"
            ],
            "competence_demonstration": [
                r"research (shows|indicates|suggests)",
                r"evidence.based",
                r"studies have found",
                r"according to.*experts",
                r"clinical.*evidence",
                r"peer.reviewed"
            ],
            "consistency": [
                r"as (we|i) discussed",
                r"building on what you shared",
                r"consistent with",
                r"following up on",
                r"remember.*mentioned",
                r"continuing.*conversation"
            ],
            "boundary_respect": [
                r"that's your choice",
                r"you know yourself best",
                r"only you can decide",
                r"respect your.*decision",
                r"your.*comfort level",
                r"at your own pace"
            ]
        }
        
        self.trust_undermining_patterns = [
            r"i know exactly how you feel",
            r"you should definitely",
            r"you must.*immediately",
            r"trust me on this",
            r"i guarantee.*will work",
            r"just do what i say"
        ]
    
    def _initialize_clarity_patterns(self):
        """Initialize patterns for communication clarity."""
        
        self.jargon_terms = [
            # Medical/Clinical terms
            "comorbidity", "etiology", "pathophysiology", "differential diagnosis",
            "contraindications", "pharmacokinetics", "biomarkers", "neurotransmitters",
            
            # Psychological terms
            "cognitive behavioral therapy", "dialectical behavior therapy", "somatic experiencing",
            "psychodynamic", "transference", "countertransference", "dissociation",
            "alexithymia", "anhedonia", "dysthymia", "rumination", "catastrophizing",
            
            # Technical terms
            "algorithm", "parameters", "methodology", "paradigm", "empirical",
            "quantitative", "qualitative", "meta-analysis", "systematic review"
        ]
        
        self.clarity_enhancers = [
            r"in other words",
            r"what (this|that) means is",
            r"to put it simply",
            r"for example",
            r"let me explain",
            r"in simpler terms",
            r"think of it (as|like)",
            r"imagine.*situation"
        ]
        
        self.confusion_indicators = [
            r"i don't understand",
            r"can you explain",
            r"what do you mean",
            r"i'm confused",
            r"unclear.*about",
            r"help me understand"
        ]
    
    def _initialize_engagement_patterns(self):
        """Initialize patterns for engagement quality."""
        
        self.engagement_indicators = {
            "personalization": [
                r"(based on|given) what you've (shared|told me)",
                r"in your (situation|case|circumstances)",
                r"for you specifically",
                r"your (unique|particular|individual)",
                r"tailored.*your needs",
                r"considering your.*background"
            ],
            "interactive_elements": [
                r"what do you think about",
                r"how does (this|that) sound",
                r"would you like to (try|explore)",
                r"what resonates with you",
                r"which (option|approach) appeals",
                r"what feels (right|comfortable)"
            ],
            "follow_up_encouragement": [
                r"feel free to.*back",
                r"let me know how",
                r"i'd love to hear",
                r"please (update|share)",
                r"keep me (posted|informed)",
                r"continue.*conversation"
            ],
            "curiosity_demonstration": [
                r"tell me more about",
                r"i'm curious about",
                r"what's that like for you",
                r"help me understand.*better",
                r"can you describe",
                r"what does.*look like"
            ]
        }
        
        self.disengagement_indicators = [
            r"that's all i can help with",
            r"there's nothing more",
            r"you should figure it out",
            r"that's not my area",
            r"i can't help with that",
            r"end of discussion"
        ]
    
    async def evaluate_model(
        self,
        model_name: str,
        conversations: List[Dict[str, Any]],
        user_feedback: Optional[List[Dict[str, Any]]] = None,
        **kwargs
    ) -> PatientExperienceScore:
        """
        Evaluate patient experience of a model.
        
        Args:
            model_name: Name of the model being evaluated
            conversations: List of conversation data
            user_feedback: Optional direct user feedback/ratings
            **kwargs: Additional evaluation parameters
            
        Returns:
            Patient experience score
        """
        self.logger.info(f"Starting patient experience evaluation for {model_name}")
        
        if not conversations:
            raise ValueError("No conversations provided for evaluation")
        
        # Extract assistant responses
        assistant_responses = self._extract_assistant_responses(conversations)
        
        # Evaluate each metric
        helpfulness_score, helpfulness_data = await self._evaluate_perceived_helpfulness(
            assistant_responses, user_feedback
        )
        trust_score, trust_data = await self._evaluate_trust_level(
            assistant_responses, conversations, user_feedback
        )
        clarity_score, clarity_data = await self._evaluate_communication_clarity(
            assistant_responses, conversations
        )
        engagement_score, engagement_data = await self._evaluate_engagement_quality(
            conversations
        )
        satisfaction_score, satisfaction_data = await self._evaluate_overall_satisfaction(
            conversations, user_feedback
        )
        
        # Calculate overall score (weighted average)
        weights = self.config.get("metric_weights", {
            "helpfulness": 0.25,
            "trust": 0.25,
            "clarity": 0.2,
            "engagement": 0.15,
            "satisfaction": 0.15
        })
        
        overall_score = (
            helpfulness_score * weights["helpfulness"] +
            trust_score * weights["trust"] +
            clarity_score * weights["clarity"] +
            engagement_score * weights["engagement"] +
            satisfaction_score * weights["satisfaction"]
        ) * 10  # Scale to 0-100
        
        # Generate review flags
        review_flags = self._generate_review_flags(
            helpfulness_data, trust_data, clarity_data, engagement_data, satisfaction_data
        )
        
        # Compile metric details
        metric_details = {
            "helpfulness": helpfulness_data,
            "trust": trust_data,
            "clarity": clarity_data,
            "engagement": engagement_data,
            "satisfaction": satisfaction_data,
            "weights": weights
        }
        
        score = PatientExperienceScore(
            perceived_helpfulness_score=helpfulness_score,
            trust_level_score=trust_score,
            communication_clarity_score=clarity_score,
            engagement_quality_score=engagement_score,
            overall_satisfaction_score=satisfaction_score,
            overall_score=overall_score,
            avg_helpfulness_rating=helpfulness_data["avg_helpfulness_rating"],
            avg_trust_rating=trust_data["avg_trust_rating"],
            avg_clarity_rating=clarity_data["avg_clarity_rating"],
            avg_engagement_score=engagement_data["avg_engagement_score"],
            net_promoter_score=satisfaction_data["net_promoter_score"],
            avg_reading_level=clarity_data["avg_reading_level"],
            jargon_density=clarity_data["jargon_density"],
            response_length_appropriateness=clarity_data["length_appropriateness"],
            conversation_completion_rate=engagement_data["completion_rate"],
            avg_conversation_length=engagement_data["avg_conversation_length"],
            user_follow_up_rate=engagement_data["follow_up_rate"],
            total_conversations=len(conversations),
            total_user_feedback=len(user_feedback) if user_feedback else 0,
            review_flags=review_flags,
            metric_details=metric_details
        )
        
        self.logger.info(
            f"Patient experience evaluation complete for {model_name}: "
            f"Overall Score: {overall_score:.1f}/100"
        )
        
        return score
    
    def _extract_assistant_responses(self, conversations: List[Dict[str, Any]]) -> List[str]:
        """Extract assistant responses from conversation data."""
        responses = []
        
        for conversation in conversations:
            if "messages" in conversation:
                for message in conversation["messages"]:
                    if message.get("role") == "assistant":
                        content = message.get("content", "")
                        if content.strip():
                            responses.append(content)
        
        return responses
    
    async def _evaluate_perceived_helpfulness(
        self,
        responses: List[str],
        user_feedback: Optional[List[Dict[str, Any]]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate perceived helpfulness (1-10 scale).
        """
        # Use direct user feedback if available
        if user_feedback:
            helpfulness_ratings = []
            for feedback in user_feedback:
                if "helpfulness_rating" in feedback:
                    helpfulness_ratings.append(feedback["helpfulness_rating"])
            
            if helpfulness_ratings:
                avg_helpfulness = np.mean(helpfulness_ratings)
                score = avg_helpfulness  # Already on 1-10 scale
                
                helpfulness_data = {
                    "avg_helpfulness_rating": avg_helpfulness,
                    "total_ratings": len(helpfulness_ratings),
                    "source": "direct_feedback",
                    "score_rationale": f"Direct user ratings: {avg_helpfulness:.1f}/10"
                }
                
                return score, helpfulness_data
        
        # Infer helpfulness from response content
        helpfulness_scores = []
        pattern_matches = {category: [] for category in self.helpfulness_indicators}
        
        for response in responses:
            response_lower = response.lower()
            
            # Count helpfulness indicators
            total_helpful = 0
            for category, patterns in self.helpfulness_indicators.items():
                count = sum(1 for pattern in patterns if re.search(pattern, response_lower))
                pattern_matches[category].append(count)
                total_helpful += count
            
            # Check for unhelpful indicators
            unhelpful_count = sum(
                1 for pattern in self.unhelpful_indicators
                if re.search(pattern, response_lower)
            )
            
            # Calculate helpfulness score for this response
            base_score = min(total_helpful * 2, 8) + 2  # Base 2 points, up to 10
            penalty = unhelpful_count * 2  # -2 points per unhelpful indicator
            
            response_helpfulness = max(1, base_score - penalty)  # Minimum 1 point
            helpfulness_scores.append(response_helpfulness)
        
        avg_helpfulness = np.mean(helpfulness_scores) if helpfulness_scores else 5.0
        score = avg_helpfulness
        
        helpfulness_data = {
            "avg_helpfulness_rating": avg_helpfulness,
            "actionable_advice_avg": np.mean(pattern_matches["actionable_advice"]),
            "resource_provision_avg": np.mean(pattern_matches["resource_provision"]),
            "validation_support_avg": np.mean(pattern_matches["validation_support"]),
            "skill_building_avg": np.mean(pattern_matches["skill_building"]),
            "total_responses": len(responses),
            "source": "inferred_from_content",
            "score_rationale": f"Inferred helpfulness: {avg_helpfulness:.1f}/10"
        }
        
        return score, helpfulness_data
    
    async def _evaluate_trust_level(
        self,
        responses: List[str],
        conversations: List[Dict[str, Any]],
        user_feedback: Optional[List[Dict[str, Any]]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate trust level (average of 3 survey questions, 1-10 scale).
        """
        # Use direct user feedback if available
        if user_feedback:
            trust_ratings = []
            for feedback in user_feedback:
                # Look for trust-related ratings
                trust_keys = ["trust_rating", "credibility_rating", "reliability_rating"]
                feedback_trust = []
                
                for key in trust_keys:
                    if key in feedback:
                        feedback_trust.append(feedback[key])
                
                if feedback_trust:
                    trust_ratings.append(np.mean(feedback_trust))
            
            if trust_ratings:
                avg_trust = np.mean(trust_ratings)
                score = avg_trust
                
                trust_data = {
                    "avg_trust_rating": avg_trust,
                    "total_ratings": len(trust_ratings),
                    "source": "direct_feedback",
                    "score_rationale": f"Direct user ratings: {avg_trust:.1f}/10"
                }
                
                return score, trust_data
        
        # Infer trust from response patterns
        trust_scores = []
        pattern_matches = {category: [] for category in self.trust_building_patterns}
        
        for response in responses:
            response_lower = response.lower()
            
            # Count trust-building indicators
            total_trust_building = 0
            for category, patterns in self.trust_building_patterns.items():
                count = sum(1 for pattern in patterns if re.search(pattern, response_lower))
                pattern_matches[category].append(count)
                total_trust_building += count
            
            # Check for trust-undermining indicators
            trust_undermining = sum(
                1 for pattern in self.trust_undermining_patterns
                if re.search(pattern, response_lower)
            )
            
            # Calculate trust score for this response
            base_score = min(total_trust_building * 1.5, 7) + 3  # Base 3 points, up to 10
            penalty = trust_undermining * 3  # -3 points per undermining indicator
            
            response_trust = max(1, base_score - penalty)
            trust_scores.append(response_trust)
        
        avg_trust = np.mean(trust_scores) if trust_scores else 6.0
        score = avg_trust
        
        trust_data = {
            "avg_trust_rating": avg_trust,
            "transparency_avg": np.mean(pattern_matches["transparency"]),
            "competence_avg": np.mean(pattern_matches["competence_demonstration"]),
            "consistency_avg": np.mean(pattern_matches["consistency"]),
            "boundary_respect_avg": np.mean(pattern_matches["boundary_respect"]),
            "total_responses": len(responses),
            "source": "inferred_from_content",
            "score_rationale": f"Inferred trust: {avg_trust:.1f}/10"
        }
        
        return score, trust_data
    
    async def _evaluate_communication_clarity(
        self,
        responses: List[str],
        conversations: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate communication clarity (language appropriateness, jargon avoidance, 1-10 scale).
        """
        if not responses:
            return 0.0, {"error": "No responses to evaluate"}
        
        # Calculate reading level
        reading_levels = []
        jargon_counts = []
        length_scores = []
        clarity_enhancer_counts = []
        
        for response in responses:
            # Reading level analysis
            try:
                flesch_score = flesch_reading_ease(response)
                ari_score = automated_readability_index(response)
                
                # Convert to grade level (approximate)
                grade_level = max(0, 20 - (flesch_score / 5))  # Rough conversion
                reading_levels.append(grade_level)
            except:
                reading_levels.append(12)  # Default to 12th grade
            
            # Jargon analysis
            words = response.lower().split()
            jargon_count = sum(1 for word in words if any(jargon in word for jargon in self.jargon_terms))
            jargon_density = jargon_count / len(words) if words else 0
            jargon_counts.append(jargon_density)
            
            # Response length appropriateness
            word_count = len(words)
            if self.optimal_response_length[0] <= word_count <= self.optimal_response_length[1]:
                length_score = 10
            elif word_count < self.optimal_response_length[0]:
                length_score = max(1, 10 - (self.optimal_response_length[0] - word_count) * 0.1)
            else:
                length_score = max(1, 10 - (word_count - self.optimal_response_length[1]) * 0.05)
            length_scores.append(length_score)
            
            # Clarity enhancers
            enhancer_count = sum(
                1 for pattern in self.clarity_enhancers
                if re.search(pattern, response.lower())
            )
            clarity_enhancer_counts.append(enhancer_count)
        
        # Calculate overall clarity score
        avg_reading_level = np.mean(reading_levels)
        avg_jargon_density = np.mean(jargon_counts)
        avg_length_score = np.mean(length_scores)
        avg_enhancers = np.mean(clarity_enhancer_counts)
        
        # Reading level score (8-12th grade is optimal)
        optimal_min, optimal_max = self.optimal_reading_level
        if optimal_min <= avg_reading_level <= optimal_max:
            reading_score = 10
        elif avg_reading_level < optimal_min:
            reading_score = max(1, 10 - (optimal_min - avg_reading_level) * 0.5)
        else:
            reading_score = max(1, 10 - (avg_reading_level - optimal_max) * 0.3)
        
        # Jargon penalty
        jargon_score = max(1, 10 - (avg_jargon_density / self.jargon_penalty_threshold) * 3)
        
        # Clarity enhancer bonus
        enhancer_bonus = min(avg_enhancers * 0.5, 2)  # Up to 2 bonus points
        
        # Combined clarity score
        clarity_score = (reading_score * 0.4 + jargon_score * 0.3 + avg_length_score * 0.3) + enhancer_bonus
        clarity_score = min(clarity_score, 10)
        
        clarity_data = {
            "avg_clarity_rating": clarity_score,
            "avg_reading_level": avg_reading_level,
            "jargon_density": avg_jargon_density,
            "length_appropriateness": avg_length_score,
            "clarity_enhancers_avg": avg_enhancers,
            "optimal_reading_range": self.optimal_reading_level,
            "optimal_length_range": self.optimal_response_length,
            "total_responses": len(responses),
            "score_rationale": f"Reading level: {avg_reading_level:.1f}, Jargon: {avg_jargon_density:.3f}"
        }
        
        return clarity_score, clarity_data
    
    async def _evaluate_engagement_quality(self, conversations: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate engagement quality (conversation metrics).
        """
        if not conversations:
            return 0.0, {"error": "No conversations to evaluate"}
        
        # Analyze conversation patterns
        completion_rates = []
        conversation_lengths = []
        follow_up_rates = []
        engagement_indicators = []
        
        for conversation in conversations:
            messages = conversation.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            
            # Conversation completion (did it end naturally?)
            completed = self._is_conversation_completed_naturally(messages)
            completion_rates.append(1 if completed else 0)
            
            # Conversation length
            conversation_lengths.append(len(messages))
            
            # User follow-up rate (user continued after assistant response)
            follow_ups = 0
            for i, msg in enumerate(messages[:-1]):
                if msg.get("role") == "assistant" and i + 1 < len(messages):
                    if messages[i + 1].get("role") == "user":
                        follow_ups += 1
            
            follow_up_rate = follow_ups / len(assistant_messages) if assistant_messages else 0
            follow_up_rates.append(follow_up_rate)
            
            # Engagement indicators in assistant responses
            total_engagement = 0
            for msg in assistant_messages:
                content = msg.get("content", "").lower()
                
                for category, patterns in self.engagement_indicators.items():
                    for pattern in patterns:
                        if re.search(pattern, content):
                            total_engagement += 1
                            break  # Count each category once per message
            
            avg_engagement = total_engagement / len(assistant_messages) if assistant_messages else 0
            engagement_indicators.append(avg_engagement)
        
        # Calculate overall engagement score
        avg_completion_rate = np.mean(completion_rates)
        avg_conversation_length = np.mean(conversation_lengths)
        avg_follow_up_rate = np.mean(follow_up_rates)
        avg_engagement_indicators = np.mean(engagement_indicators)
        
        # Score components (0-10 scale)
        completion_score = avg_completion_rate * 10
        
        # Length score (optimal: 6-15 messages)
        if 6 <= avg_conversation_length <= 15:
            length_score = 10
        elif avg_conversation_length < 6:
            length_score = max(1, avg_conversation_length * 1.5)
        else:
            length_score = max(1, 10 - (avg_conversation_length - 15) * 0.2)
        
        follow_up_score = min(avg_follow_up_rate * 15, 10)  # 66% follow-up rate = 10 points
        engagement_score = min(avg_engagement_indicators * 3, 10)  # 3.3 indicators = 10 points
        
        # Combined engagement score
        overall_engagement = (
            completion_score * 0.3 +
            length_score * 0.2 +
            follow_up_score * 0.3 +
            engagement_score * 0.2
        )
        
        engagement_data = {
            "avg_engagement_score": overall_engagement,
            "completion_rate": avg_completion_rate,
            "avg_conversation_length": avg_conversation_length,
            "follow_up_rate": avg_follow_up_rate,
            "engagement_indicators_avg": avg_engagement_indicators,
            "total_conversations": len(conversations),
            "score_rationale": f"Completion: {avg_completion_rate:.2f}, Length: {avg_conversation_length:.1f}, Follow-up: {avg_follow_up_rate:.2f}"
        }
        
        return overall_engagement, engagement_data
    
    def _is_conversation_completed_naturally(self, messages: List[Dict[str, str]]) -> bool:
        """Check if conversation ended naturally."""
        if len(messages) < 2:
            return False
        
        last_message = messages[-1].get("content", "").lower()
        
        # Natural ending indicators
        ending_indicators = [
            "thank you", "thanks", "goodbye", "bye", "take care",
            "that helps", "feeling better", "i'll try that",
            "good advice", "appreciate", "helpful"
        ]
        
        return any(indicator in last_message for indicator in ending_indicators)
    
    async def _evaluate_overall_satisfaction(
        self,
        conversations: List[Dict[str, Any]],
        user_feedback: Optional[List[Dict[str, Any]]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate overall satisfaction (Net Promoter Score style, 0-10).
        """
        # Use direct user feedback if available
        if user_feedback:
            satisfaction_ratings = []
            nps_ratings = []
            
            for feedback in user_feedback:
                if "overall_satisfaction" in feedback:
                    satisfaction_ratings.append(feedback["overall_satisfaction"])
                
                if "would_recommend" in feedback:
                    # Convert boolean/rating to NPS-style score
                    recommend = feedback["would_recommend"]
                    if isinstance(recommend, bool):
                        nps_ratings.append(10 if recommend else 3)
                    else:
                        nps_ratings.append(recommend)
            
            if satisfaction_ratings:
                avg_satisfaction = np.mean(satisfaction_ratings)
                nps_score = np.mean(nps_ratings) if nps_ratings else avg_satisfaction
                
                satisfaction_data = {
                    "avg_satisfaction_rating": avg_satisfaction,
                    "net_promoter_score": nps_score,
                    "total_ratings": len(satisfaction_ratings),
                    "source": "direct_feedback",
                    "score_rationale": f"Direct ratings: {avg_satisfaction:.1f}/10, NPS: {nps_score:.1f}"
                }
                
                return avg_satisfaction, satisfaction_data
        
        # Infer satisfaction from conversation patterns
        satisfaction_indicators = []
        
        for conversation in conversations:
            messages = conversation.get("messages", [])
            user_messages = [msg for msg in messages if msg.get("role") == "user"]
            
            if not user_messages:
                continue
            
            # Analyze user sentiment progression
            first_message = user_messages[0].get("content", "").lower()
            last_message = user_messages[-1].get("content", "").lower() if len(user_messages) > 1 else first_message
            
            # Positive sentiment indicators
            positive_words = [
                "better", "helpful", "thank", "good", "great", "appreciate",
                "understand", "clear", "useful", "effective"
            ]
            
            negative_words = [
                "worse", "unhelpful", "confused", "frustrated", "bad",
                "unclear", "useless", "pointless", "waste"
            ]
            
            # Count sentiment in last message
            positive_count = sum(1 for word in positive_words if word in last_message)
            negative_count = sum(1 for word in negative_words if word in last_message)
            
            # Simple satisfaction heuristic
            if positive_count > negative_count:
                satisfaction = 7 + min(positive_count, 3)  # 7-10 range
            elif negative_count > positive_count:
                satisfaction = max(1, 5 - negative_count)  # 1-5 range
            else:
                satisfaction = 5  # Neutral
            
            satisfaction_indicators.append(satisfaction)
        
        avg_satisfaction = np.mean(satisfaction_indicators) if satisfaction_indicators else 5.0
        nps_score = avg_satisfaction  # Use same score for NPS
        
        satisfaction_data = {
            "avg_satisfaction_rating": avg_satisfaction,
            "net_promoter_score": nps_score,
            "total_conversations": len(conversations),
            "source": "inferred_from_content",
            "score_rationale": f"Inferred satisfaction: {avg_satisfaction:.1f}/10"
        }
        
        return avg_satisfaction, satisfaction_data
    
    def _generate_review_flags(
        self,
        helpfulness_data: Dict[str, Any],
        trust_data: Dict[str, Any],
        clarity_data: Dict[str, Any],
        engagement_data: Dict[str, Any],
        satisfaction_data: Dict[str, Any]
    ) -> List[str]:
        """Generate manual review flags for edge cases."""
        flags = []
        
        # Helpfulness flags
        if helpfulness_data.get("avg_helpfulness_rating", 0) < 4:
            flags.append("LOW_PERCEIVED_HELPFULNESS")
        
        # Trust flags
        if trust_data.get("avg_trust_rating", 0) < 5:
            flags.append("LOW_TRUST_LEVEL")
        
        # Clarity flags
        if clarity_data.get("avg_reading_level", 12) > 16:
            flags.append("HIGH_READING_LEVEL")
        
        if clarity_data.get("jargon_density", 0) > 0.1:
            flags.append("HIGH_JARGON_DENSITY")
        
        # Engagement flags
        if engagement_data.get("completion_rate", 1) < 0.5:
            flags.append("LOW_CONVERSATION_COMPLETION")
        
        if engagement_data.get("avg_conversation_length", 10) < 4:
            flags.append("SHORT_CONVERSATIONS")
        
        if engagement_data.get("follow_up_rate", 1) < 0.3:
            flags.append("LOW_USER_FOLLOW_UP")
        
        # Satisfaction flags
        if satisfaction_data.get("avg_satisfaction_rating", 5) < 4:
            flags.append("LOW_OVERALL_SATISFACTION")
        
        if satisfaction_data.get("net_promoter_score", 5) < 6:
            flags.append("LOW_NET_PROMOTER_SCORE")
        
        return flags