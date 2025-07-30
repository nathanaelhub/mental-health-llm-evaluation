"""
Feedback Loop Integration System

Comprehensive user feedback collection and analysis system that continuously
improves model selection through user ratings, quality assessments, and
automated feedback processing.
"""

import asyncio
import logging
import json
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from pathlib import Path

from ..chat.dynamic_model_selector import PromptType

logger = logging.getLogger(__name__)


class FeedbackType(Enum):
    """Types of user feedback"""
    THUMBS_UP_DOWN = "thumbs_up_down"       # Simple binary rating
    STAR_RATING = "star_rating"             # 1-5 star rating
    DETAILED_FORM = "detailed_form"         # Comprehensive feedback
    IMPLICIT_BEHAVIOR = "implicit_behavior" # Inferred from user actions
    AUTOMATED_QUALITY = "automated_quality" # System-generated quality scores


class FeedbackCategory(Enum):
    """Categories of feedback assessment"""
    HELPFULNESS = "helpfulness"
    ACCURACY = "accuracy"
    EMPATHY = "empathy"
    SAFETY = "safety"
    CLARITY = "clarity"
    RESPONSE_TIME = "response_time"
    OVERALL_SATISFACTION = "overall_satisfaction"


class SentimentScore(Enum):
    """Sentiment analysis results"""
    VERY_NEGATIVE = -2
    NEGATIVE = -1
    NEUTRAL = 0
    POSITIVE = 1
    VERY_POSITIVE = 2


@dataclass
class UserFeedback:
    """Individual user feedback record"""
    feedback_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    user_id: str = ""
    session_id: Optional[str] = None
    message_id: Optional[str] = None
    
    # Context information
    prompt: str = ""
    prompt_type: PromptType = PromptType.GENERAL_SUPPORT
    selected_model: str = ""
    response_text: str = ""
    
    # Feedback details
    feedback_type: FeedbackType = FeedbackType.THUMBS_UP_DOWN
    
    # Ratings (1-5 scale where applicable)
    overall_rating: Optional[float] = None
    helpfulness_rating: Optional[float] = None
    accuracy_rating: Optional[float] = None
    empathy_rating: Optional[float] = None
    safety_rating: Optional[float] = None
    clarity_rating: Optional[float] = None
    response_time_rating: Optional[float] = None
    
    # Binary feedback
    thumbs_up: Optional[bool] = None
    
    # Text feedback
    feedback_text: Optional[str] = None
    improvement_suggestions: Optional[str] = None
    
    # Behavioral indicators
    conversation_continued: Optional[bool] = None
    time_spent_reading_ms: Optional[int] = None
    follow_up_questions: int = 0
    
    # System assessments
    automated_quality_score: Optional[float] = None
    sentiment_score: Optional[SentimentScore] = None
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    user_agent: Optional[str] = None
    platform: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'feedback_id': self.feedback_id,
            'user_id': self.user_id,
            'session_id': self.session_id,
            'message_id': self.message_id,
            'prompt': self.prompt[:100],  # Truncate for privacy
            'prompt_type': self.prompt_type.value,
            'selected_model': self.selected_model,
            'response_text': self.response_text[:200] if self.response_text else None,
            'feedback_type': self.feedback_type.value,
            'overall_rating': self.overall_rating,
            'helpfulness_rating': self.helpfulness_rating,
            'accuracy_rating': self.accuracy_rating,
            'empathy_rating': self.empathy_rating,
            'safety_rating': self.safety_rating,
            'clarity_rating': self.clarity_rating,
            'response_time_rating': self.response_time_rating,
            'thumbs_up': self.thumbs_up,
            'feedback_text': self.feedback_text,
            'improvement_suggestions': self.improvement_suggestions,
            'conversation_continued': self.conversation_continued,
            'time_spent_reading_ms': self.time_spent_reading_ms,
            'follow_up_questions': self.follow_up_questions,
            'automated_quality_score': self.automated_quality_score,
            'sentiment_score': self.sentiment_score.value if self.sentiment_score else None,
            'timestamp': self.timestamp.isoformat(),
            'user_agent': self.user_agent,
            'platform': self.platform,
            'metadata': self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserFeedback':
        """Create from dictionary"""
        feedback = cls()
        
        # Basic fields
        feedback.feedback_id = data.get('feedback_id', str(uuid.uuid4()))
        feedback.user_id = data.get('user_id', '')
        feedback.session_id = data.get('session_id')
        feedback.message_id = data.get('message_id')
        feedback.prompt = data.get('prompt', '')
        feedback.prompt_type = PromptType(data.get('prompt_type', 'general_wellness'))
        feedback.selected_model = data.get('selected_model', '')
        feedback.response_text = data.get('response_text', '')
        feedback.feedback_type = FeedbackType(data.get('feedback_type', 'thumbs_up_down'))
        
        # Ratings
        feedback.overall_rating = data.get('overall_rating')
        feedback.helpfulness_rating = data.get('helpfulness_rating')
        feedback.accuracy_rating = data.get('accuracy_rating')
        feedback.empathy_rating = data.get('empathy_rating')
        feedback.safety_rating = data.get('safety_rating')
        feedback.clarity_rating = data.get('clarity_rating')
        feedback.response_time_rating = data.get('response_time_rating')
        feedback.thumbs_up = data.get('thumbs_up')
        
        # Text fields
        feedback.feedback_text = data.get('feedback_text')
        feedback.improvement_suggestions = data.get('improvement_suggestions')
        
        # Behavioral
        feedback.conversation_continued = data.get('conversation_continued')
        feedback.time_spent_reading_ms = data.get('time_spent_reading_ms')
        feedback.follow_up_questions = data.get('follow_up_questions', 0)
        
        # System assessments
        feedback.automated_quality_score = data.get('automated_quality_score')
        sentiment_value = data.get('sentiment_score')
        if sentiment_value is not None:
            feedback.sentiment_score = SentimentScore(sentiment_value)
        
        # Metadata
        if 'timestamp' in data:
            feedback.timestamp = datetime.fromisoformat(data['timestamp'])
        feedback.user_agent = data.get('user_agent')
        feedback.platform = data.get('platform')
        feedback.metadata = data.get('metadata', {})
        
        return feedback


@dataclass
class FeedbackAnalytics:
    """Analytics derived from feedback data"""
    total_feedback_count: int = 0
    
    # Rating averages
    avg_overall_rating: float = 0.0
    avg_helpfulness: float = 0.0
    avg_accuracy: float = 0.0
    avg_empathy: float = 0.0
    avg_safety: float = 0.0
    avg_clarity: float = 0.0
    avg_response_time: float = 0.0
    
    # Binary feedback
    thumbs_up_percentage: float = 0.0
    
    # Model performance
    model_ratings: Dict[str, float] = field(default_factory=dict)
    model_feedback_counts: Dict[str, int] = field(default_factory=dict)
    
    # Prompt type performance
    prompt_type_ratings: Dict[str, float] = field(default_factory=dict)
    
    # Temporal patterns
    hourly_satisfaction: Dict[int, float] = field(default_factory=dict)
    daily_feedback_count: Dict[str, int] = field(default_factory=dict)
    
    # Quality insights
    top_improvement_areas: List[Tuple[str, int]] = field(default_factory=list)
    sentiment_distribution: Dict[str, int] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_feedback_count': self.total_feedback_count,
            'avg_overall_rating': self.avg_overall_rating,
            'avg_helpfulness': self.avg_helpfulness,
            'avg_accuracy': self.avg_accuracy,
            'avg_empathy': self.avg_empathy,
            'avg_safety': self.avg_safety,
            'avg_clarity': self.avg_clarity,
            'avg_response_time': self.avg_response_time,
            'thumbs_up_percentage': self.thumbs_up_percentage,
            'model_ratings': self.model_ratings,
            'model_feedback_counts': self.model_feedback_counts,
            'prompt_type_ratings': self.prompt_type_ratings,
            'hourly_satisfaction': self.hourly_satisfaction,
            'daily_feedback_count': self.daily_feedback_count,
            'top_improvement_areas': self.top_improvement_areas,
            'sentiment_distribution': self.sentiment_distribution
        }


class AutomatedQualityAssessment:
    """Automated quality assessment of AI responses"""
    
    def __init__(self):
        # Quality indicators and their weights
        self.quality_indicators = {
            'response_length_appropriate': 0.1,
            'contains_empathetic_language': 0.2,
            'provides_actionable_advice': 0.15,
            'maintains_professional_tone': 0.15,
            'addresses_user_concern': 0.2,
            'includes_safety_considerations': 0.15,
            'avoids_harmful_content': 0.05
        }
        
        # Empathetic language patterns
        self.empathy_patterns = [
            r'\b(understand|feel|sounds?\s+difficult|challenging|here\s+for\s+you)\b',
            r'\b(sorry\s+to\s+hear|that\s+must\s+be|can\s+imagine)\b',
            r'\b(valid|normal|understandable)\s+(feeling|concern|worry)\b'
        ]
        
        # Safety language patterns
        self.safety_patterns = [
            r'\b(crisis|emergency|immediate\s+help|professional\s+support)\b',
            r'\b(therapist|counselor|doctor|mental\s+health\s+professional)\b',
            r'\b(safe|safety|harm|risk)\b'
        ]
        
        # Harmful content patterns (to detect and penalize)
        self.harmful_patterns = [
            r'\b(just\s+get\s+over\s+it|not\s+a\s+big\s+deal|everyone\s+feels\s+this)\b',
            r'\b(attention\s+seeking|making\s+it\s+up|imagining\s+things)\b'
        ]
    
    def assess_response_quality(self, 
                              prompt: str, 
                              response: str, 
                              prompt_type: PromptType) -> float:
        """
        Assess the quality of an AI response
        
        Returns:
            Quality score between 0.0 and 1.0
        """
        import re
        
        quality_score = 0.0
        
        # Response length appropriateness
        response_length = len(response.split())
        if 20 <= response_length <= 200:  # Reasonable length
            quality_score += self.quality_indicators['response_length_appropriate']
        elif response_length < 10:  # Too short
            quality_score += self.quality_indicators['response_length_appropriate'] * 0.3
        
        # Empathetic language detection
        empathy_matches = sum(1 for pattern in self.empathy_patterns 
                            if re.search(pattern, response.lower()))
        if empathy_matches > 0:
            empathy_bonus = min(empathy_matches * 0.5, 1.0)  # Cap at 1.0
            quality_score += self.quality_indicators['contains_empathetic_language'] * empathy_bonus
        
        # Safety considerations (especially important for crisis)
        safety_matches = sum(1 for pattern in self.safety_patterns 
                           if re.search(pattern, response.lower()))
        safety_weight = self.quality_indicators['includes_safety_considerations']
        if prompt_type == PromptType.CRISIS:
            safety_weight *= 2  # Double weight for crisis situations
        
        if safety_matches > 0:
            safety_bonus = min(safety_matches * 0.5, 1.0)
            quality_score += safety_weight * safety_bonus
        
        # Professional tone (basic check)
        if not re.search(r'\b(lol|omg|wtf|dude|bro)\b', response.lower()):
            quality_score += self.quality_indicators['maintains_professional_tone']
        
        # Actionable advice detection (simple heuristic)
        actionable_patterns = [
            r'\b(try|consider|might\s+help|suggest|recommend)\b',
            r'\b(can|could|would)\s+.*\b(help|useful|beneficial)\b',
            r'\b(practice|exercise|technique|strategy)\b'
        ]
        
        actionable_matches = sum(1 for pattern in actionable_patterns 
                               if re.search(pattern, response.lower()))
        if actionable_matches > 0:
            actionable_bonus = min(actionable_matches * 0.3, 1.0)
            quality_score += self.quality_indicators['provides_actionable_advice'] * actionable_bonus
        
        # Address user concern (keyword overlap between prompt and response)
        prompt_keywords = set(word.lower() for word in prompt.split() 
                            if len(word) > 3 and word.isalpha())
        response_keywords = set(word.lower() for word in response.split() 
                              if len(word) > 3 and word.isalpha())
        
        keyword_overlap = len(prompt_keywords.intersection(response_keywords))
        if keyword_overlap > 0:
            overlap_score = min(keyword_overlap / len(prompt_keywords), 1.0) if prompt_keywords else 0
            quality_score += self.quality_indicators['addresses_user_concern'] * overlap_score
        
        # Harmful content penalty
        harmful_matches = sum(1 for pattern in self.harmful_patterns 
                            if re.search(pattern, response.lower()))
        if harmful_matches > 0:
            quality_score -= self.quality_indicators['avoids_harmful_content'] * harmful_matches
        
        # Ensure score is within bounds
        return max(0.0, min(1.0, quality_score))
    
    def analyze_sentiment(self, text: str) -> SentimentScore:
        """Simple sentiment analysis of feedback text"""
        if not text:
            return SentimentScore.NEUTRAL
        
        text_lower = text.lower()
        
        # Positive indicators
        positive_words = ['good', 'great', 'excellent', 'helpful', 'useful', 'amazing', 
                         'perfect', 'love', 'wonderful', 'fantastic', 'awesome', 'thank']
        positive_count = sum(1 for word in positive_words if word in text_lower)
        
        # Negative indicators
        negative_words = ['bad', 'terrible', 'awful', 'useless', 'horrible', 'hate', 
                         'worst', 'disappointing', 'frustrating', 'annoying', 'stupid']
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Strong indicators
        very_positive = ['amazing', 'fantastic', 'perfect', 'excellent', 'outstanding']
        very_negative = ['terrible', 'awful', 'horrible', 'worst', 'hate']
        
        very_positive_count = sum(1 for word in very_positive if word in text_lower)
        very_negative_count = sum(1 for word in very_negative if word in text_lower)
        
        # Determine sentiment
        if very_positive_count > 0 and very_negative_count == 0:
            return SentimentScore.VERY_POSITIVE
        elif very_negative_count > 0 and very_positive_count == 0:
            return SentimentScore.VERY_NEGATIVE
        elif positive_count > negative_count:
            return SentimentScore.POSITIVE
        elif negative_count > positive_count:
            return SentimentScore.NEGATIVE
        else:
            return SentimentScore.NEUTRAL


class FeedbackCollector:
    """
    Comprehensive feedback collection system
    
    Features:
    - Multiple feedback types (thumbs, ratings, detailed forms)
    - Automated quality assessment
    - Real-time analytics and insights
    - Continuous learning integration
    """
    
    def __init__(self, storage_path: str = "results/development/feedback"):
        self.storage_path = Path(storage_path)
        self.storage_path.mkdir(parents=True, exist_ok=True)
        
        # Feedback storage
        self.feedback_data: List[UserFeedback] = []
        self.feedback_by_user: Dict[str, List[UserFeedback]] = defaultdict(list)
        self.feedback_by_model: Dict[str, List[UserFeedback]] = defaultdict(list)
        
        # Analytics
        self.quality_assessor = AutomatedQualityAssessment()
        self.analytics_cache: Optional[FeedbackAnalytics] = None
        self.cache_timestamp: Optional[datetime] = None
        self.cache_validity_minutes = 15
        
        # Real-time monitoring
        self.recent_feedback = deque(maxlen=100)
        self.feedback_callbacks: List[Callable[[UserFeedback], None]] = []
        
        # Load existing data
        self._load_feedback_data()
        
        logger.info(f"FeedbackCollector initialized with {len(self.feedback_data)} existing records")
    
    async def collect_thumbs_feedback(self, 
                                    user_id: str,
                                    thumbs_up: bool,
                                    prompt: str = "",
                                    selected_model: str = "",
                                    response_text: str = "",
                                    prompt_type: PromptType = PromptType.GENERAL_SUPPORT,
                                    session_id: str = None,
                                    message_id: str = None) -> str:
        """Collect simple thumbs up/down feedback"""
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            prompt=prompt,
            prompt_type=prompt_type,
            selected_model=selected_model,
            response_text=response_text,
            feedback_type=FeedbackType.THUMBS_UP_DOWN,
            thumbs_up=thumbs_up,
            overall_rating=1.0 if thumbs_up else 0.0  # Convert to numeric
        )
        
        # Add automated assessments
        await self._add_automated_assessments(feedback)
        
        # Store feedback
        await self._store_feedback(feedback)
        
        logger.info(f"Collected thumbs {'up' if thumbs_up else 'down'} from user {user_id}")
        return feedback.feedback_id
    
    async def collect_star_rating(self, 
                                user_id: str,
                                overall_rating: float,
                                prompt: str = "",
                                selected_model: str = "",
                                response_text: str = "",
                                prompt_type: PromptType = PromptType.GENERAL_SUPPORT,
                                session_id: str = None,
                                message_id: str = None,
                                feedback_text: str = None) -> str:
        """Collect star rating feedback (1-5 scale)"""
        
        # Validate rating
        if not 1 <= overall_rating <= 5:
            raise ValueError("Rating must be between 1 and 5")
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            prompt=prompt,
            prompt_type=prompt_type,
            selected_model=selected_model,
            response_text=response_text,
            feedback_type=FeedbackType.STAR_RATING,
            overall_rating=overall_rating,
            thumbs_up=overall_rating >= 3,  # 3+ stars = thumbs up
            feedback_text=feedback_text
        )
        
        # Add automated assessments
        await self._add_automated_assessments(feedback)
        
        # Store feedback
        await self._store_feedback(feedback)
        
        logger.info(f"Collected {overall_rating}-star rating from user {user_id}")
        return feedback.feedback_id
    
    async def collect_detailed_feedback(self, 
                                      user_id: str,
                                      ratings: Dict[str, float],
                                      feedback_text: str = None,
                                      improvement_suggestions: str = None,
                                      prompt: str = "",
                                      selected_model: str = "",
                                      response_text: str = "",
                                      prompt_type: PromptType = PromptType.GENERAL_SUPPORT,
                                      session_id: str = None,
                                      message_id: str = None) -> str:
        """Collect detailed feedback with multiple ratings"""
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            prompt=prompt,
            prompt_type=prompt_type,
            selected_model=selected_model,
            response_text=response_text,
            feedback_type=FeedbackType.DETAILED_FORM,
            feedback_text=feedback_text,
            improvement_suggestions=improvement_suggestions
        )
        
        # Set individual ratings
        feedback.overall_rating = ratings.get('overall')
        feedback.helpfulness_rating = ratings.get('helpfulness')
        feedback.accuracy_rating = ratings.get('accuracy')
        feedback.empathy_rating = ratings.get('empathy')
        feedback.safety_rating = ratings.get('safety')
        feedback.clarity_rating = ratings.get('clarity')
        feedback.response_time_rating = ratings.get('response_time')
        
        # Set thumbs up based on overall rating
        if feedback.overall_rating:
            feedback.thumbs_up = feedback.overall_rating >= 3
        
        # Add automated assessments
        await self._add_automated_assessments(feedback)
        
        # Store feedback
        await self._store_feedback(feedback)
        
        logger.info(f"Collected detailed feedback from user {user_id}")
        return feedback.feedback_id
    
    async def collect_behavioral_feedback(self, 
                                        user_id: str,
                                        conversation_continued: bool = None,
                                        time_spent_reading_ms: int = None,
                                        follow_up_questions: int = 0,
                                        prompt: str = "",
                                        selected_model: str = "",
                                        prompt_type: PromptType = PromptType.GENERAL_SUPPORT,
                                        session_id: str = None,
                                        message_id: str = None) -> str:
        """Collect implicit behavioral feedback"""
        
        feedback = UserFeedback(
            user_id=user_id,
            session_id=session_id,
            message_id=message_id,
            prompt=prompt,
            prompt_type=prompt_type,
            selected_model=selected_model,
            feedback_type=FeedbackType.IMPLICIT_BEHAVIOR,
            conversation_continued=conversation_continued,
            time_spent_reading_ms=time_spent_reading_ms,
            follow_up_questions=follow_up_questions
        )
        
        # Infer satisfaction from behavioral signals
        satisfaction_score = 0.5  # Neutral baseline
        
        if conversation_continued is True:
            satisfaction_score += 0.3
        elif conversation_continued is False:
            satisfaction_score -= 0.2
        
        if time_spent_reading_ms:
            # Assume 200 words per minute reading speed
            expected_reading_time = len(feedback.response_text.split()) * 300 if feedback.response_text else 5000
            reading_ratio = min(time_spent_reading_ms / expected_reading_time, 2.0)
            if reading_ratio > 0.5:  # Spent reasonable time reading
                satisfaction_score += 0.2
        
        if follow_up_questions > 0:
            satisfaction_score += min(follow_up_questions * 0.1, 0.3)
        
        feedback.overall_rating = max(1.0, min(5.0, satisfaction_score * 5))
        feedback.thumbs_up = satisfaction_score > 0.6
        
        # Store feedback
        await self._store_feedback(feedback)
        
        logger.debug(f"Collected behavioral feedback from user {user_id} (inferred satisfaction: {satisfaction_score:.2f})")
        return feedback.feedback_id
    
    async def _add_automated_assessments(self, feedback: UserFeedback):
        """Add automated quality and sentiment assessments"""
        
        # Quality assessment
        if feedback.response_text and feedback.prompt:
            feedback.automated_quality_score = self.quality_assessor.assess_response_quality(
                feedback.prompt, 
                feedback.response_text, 
                feedback.prompt_type
            )
        
        # Sentiment analysis
        if feedback.feedback_text:
            feedback.sentiment_score = self.quality_assessor.analyze_sentiment(feedback.feedback_text)
    
    async def _store_feedback(self, feedback: UserFeedback):
        """Store feedback in memory and persistent storage"""
        
        # Add to memory structures
        self.feedback_data.append(feedback)
        self.feedback_by_user[feedback.user_id].append(feedback)
        self.feedback_by_model[feedback.selected_model].append(feedback)
        self.recent_feedback.append(feedback)
        
        # Invalidate analytics cache
        self.analytics_cache = None
        
        # Call registered callbacks
        for callback in self.feedback_callbacks:
            try:
                callback(feedback)
            except Exception as e:
                logger.error(f"Error in feedback callback: {e}")
        
        # Persist to disk
        await self._persist_feedback(feedback)
    
    async def _persist_feedback(self, feedback: UserFeedback):
        """Persist feedback to disk"""
        try:
            # Save to daily file
            date_str = feedback.timestamp.strftime('%Y-%m-%d')
            feedback_file = self.storage_path / f"feedback_{date_str}.jsonl"
            
            with open(feedback_file, 'a') as f:
                f.write(json.dumps(feedback.to_dict()) + '\n')
                
        except Exception as e:
            logger.error(f"Error persisting feedback: {e}")
    
    def _load_feedback_data(self):
        """Load existing feedback data from disk"""
        try:
            feedback_files = list(self.storage_path.glob("feedback_*.jsonl"))
            
            for file_path in feedback_files:
                with open(file_path, 'r') as f:
                    for line in f:
                        try:
                            feedback_dict = json.loads(line.strip())
                            feedback = UserFeedback.from_dict(feedback_dict)
                            
                            self.feedback_data.append(feedback)
                            self.feedback_by_user[feedback.user_id].append(feedback)
                            self.feedback_by_model[feedback.selected_model].append(feedback)
                            
                        except Exception as e:
                            logger.error(f"Error loading feedback record: {e}")
            
            logger.info(f"Loaded {len(self.feedback_data)} feedback records")
            
        except Exception as e:
            logger.error(f"Error loading feedback data: {e}")
    
    def get_analytics(self, refresh_cache: bool = False) -> FeedbackAnalytics:
        """Get comprehensive feedback analytics"""
        
        # Check cache validity
        if (not refresh_cache and self.analytics_cache and self.cache_timestamp and
            datetime.now() - self.cache_timestamp < timedelta(minutes=self.cache_validity_minutes)):
            return self.analytics_cache
        
        # Compute fresh analytics
        analytics = FeedbackAnalytics()
        
        if not self.feedback_data:
            return analytics
        
        analytics.total_feedback_count = len(self.feedback_data)
        
        # Calculate rating averages
        ratings_data = {
            'overall': [f.overall_rating for f in self.feedback_data if f.overall_rating is not None],
            'helpfulness': [f.helpfulness_rating for f in self.feedback_data if f.helpfulness_rating is not None],
            'accuracy': [f.accuracy_rating for f in self.feedback_data if f.accuracy_rating is not None],
            'empathy': [f.empathy_rating for f in self.feedback_data if f.empathy_rating is not None],
            'safety': [f.safety_rating for f in self.feedback_data if f.safety_rating is not None],
            'clarity': [f.clarity_rating for f in self.feedback_data if f.clarity_rating is not None],
            'response_time': [f.response_time_rating for f in self.feedback_data if f.response_time_rating is not None]
        }
        
        analytics.avg_overall_rating = statistics.mean(ratings_data['overall']) if ratings_data['overall'] else 0
        analytics.avg_helpfulness = statistics.mean(ratings_data['helpfulness']) if ratings_data['helpfulness'] else 0
        analytics.avg_accuracy = statistics.mean(ratings_data['accuracy']) if ratings_data['accuracy'] else 0
        analytics.avg_empathy = statistics.mean(ratings_data['empathy']) if ratings_data['empathy'] else 0
        analytics.avg_safety = statistics.mean(ratings_data['safety']) if ratings_data['safety'] else 0
        analytics.avg_clarity = statistics.mean(ratings_data['clarity']) if ratings_data['clarity'] else 0
        analytics.avg_response_time = statistics.mean(ratings_data['response_time']) if ratings_data['response_time'] else 0
        
        # Thumbs up percentage
        thumbs_feedback = [f.thumbs_up for f in self.feedback_data if f.thumbs_up is not None]
        analytics.thumbs_up_percentage = (sum(thumbs_feedback) / len(thumbs_feedback) * 100) if thumbs_feedback else 0
        
        # Model performance
        for model, feedbacks in self.feedback_by_model.items():
            if not model:  # Skip empty model names
                continue
                
            model_ratings = [f.overall_rating for f in feedbacks if f.overall_rating is not None]
            if model_ratings:
                analytics.model_ratings[model] = statistics.mean(model_ratings)
                analytics.model_feedback_counts[model] = len(model_ratings)
        
        # Prompt type performance
        prompt_type_data = defaultdict(list)
        for feedback in self.feedback_data:
            if feedback.overall_rating is not None:
                prompt_type_data[feedback.prompt_type.value].append(feedback.overall_rating)
        
        for prompt_type, ratings in prompt_type_data.items():
            analytics.prompt_type_ratings[prompt_type] = statistics.mean(ratings)
        
        # Temporal patterns
        hourly_data = defaultdict(list)
        daily_data = defaultdict(int)
        
        for feedback in self.feedback_data:
            hour = feedback.timestamp.hour
            day = feedback.timestamp.strftime('%Y-%m-%d')
            
            if feedback.overall_rating is not None:
                hourly_data[hour].append(feedback.overall_rating)
            daily_data[day] += 1
        
        for hour, ratings in hourly_data.items():
            analytics.hourly_satisfaction[hour] = statistics.mean(ratings)
        
        analytics.daily_feedback_count = dict(daily_data)
        
        # Improvement areas
        improvement_mentions = defaultdict(int)
        for feedback in self.feedback_data:
            if feedback.improvement_suggestions:
                # Simple keyword extraction
                text = feedback.improvement_suggestions.lower()
                for keyword in ['speed', 'accuracy', 'empathy', 'clarity', 'safety', 'helpfulness']:
                    if keyword in text:
                        improvement_mentions[keyword] += 1
        
        analytics.top_improvement_areas = sorted(improvement_mentions.items(), key=lambda x: x[1], reverse=True)[:5]
        
        # Sentiment distribution
        sentiment_counts = defaultdict(int)
        for feedback in self.feedback_data:
            if feedback.sentiment_score:
                sentiment_counts[feedback.sentiment_score.name] += 1
        
        analytics.sentiment_distribution = dict(sentiment_counts)
        
        # Cache results
        self.analytics_cache = analytics
        self.cache_timestamp = datetime.now()
        
        return analytics
    
    def get_user_feedback_history(self, user_id: str) -> List[UserFeedback]:
        """Get feedback history for a specific user"""
        return self.feedback_by_user.get(user_id, [])
    
    def get_model_feedback_summary(self, model_name: str) -> Dict[str, Any]:
        """Get feedback summary for a specific model"""
        feedbacks = self.feedback_by_model.get(model_name, [])
        
        if not feedbacks:
            return {'message': 'No feedback found for this model'}
        
        # Calculate statistics
        ratings = [f.overall_rating for f in feedbacks if f.overall_rating is not None]
        thumbs_ups = [f.thumbs_up for f in feedbacks if f.thumbs_up is not None]
        
        return {
            'model_name': model_name,
            'total_feedback': len(feedbacks),
            'avg_rating': statistics.mean(ratings) if ratings else None,
            'thumbs_up_percentage': (sum(thumbs_ups) / len(thumbs_ups) * 100) if thumbs_ups else None,
            'recent_feedback': [f.to_dict() for f in feedbacks[-5:]],  # Last 5 feedbacks
            'automated_quality_avg': statistics.mean([f.automated_quality_score for f in feedbacks 
                                                     if f.automated_quality_score is not None]) if feedbacks else None
        }
    
    def register_feedback_callback(self, callback: Callable[[UserFeedback], None]):
        """Register callback to be called when new feedback is received"""
        self.feedback_callbacks.append(callback)
        logger.info("Registered feedback callback")
    
    def get_real_time_metrics(self) -> Dict[str, Any]:
        """Get real-time feedback metrics"""
        if not self.recent_feedback:
            return {'message': 'No recent feedback'}
        
        recent_list = list(self.recent_feedback)
        
        # Last hour metrics
        hour_ago = datetime.now() - timedelta(hours=1)
        recent_hour = [f for f in recent_list if f.timestamp >= hour_ago]
        
        recent_ratings = [f.overall_rating for f in recent_hour if f.overall_rating is not None]
        recent_thumbs = [f.thumbs_up for f in recent_hour if f.thumbs_up is not None]
        
        return {
            'last_hour_count': len(recent_hour),
            'last_hour_avg_rating': statistics.mean(recent_ratings) if recent_ratings else None,
            'last_hour_thumbs_up_pct': (sum(recent_thumbs) / len(recent_thumbs) * 100) if recent_thumbs else None,
            'total_recent_feedback': len(recent_list),
            'latest_feedback_time': max(f.timestamp for f in recent_list).isoformat() if recent_list else None
        }
    
    def export_feedback_data(self, output_file: str, date_range: Tuple[datetime, datetime] = None):
        """Export feedback data to file"""
        
        # Filter by date range if provided
        feedbacks_to_export = self.feedback_data
        if date_range:
            start_date, end_date = date_range
            feedbacks_to_export = [
                f for f in self.feedback_data 
                if start_date <= f.timestamp <= end_date
            ]
        
        export_data = {
            'export_timestamp': datetime.now().isoformat(),
            'total_records': len(feedbacks_to_export),
            'date_range': {
                'start': date_range[0].isoformat() if date_range else None,
                'end': date_range[1].isoformat() if date_range else None
            },
            'analytics': self.get_analytics().to_dict(),
            'feedback_records': [f.to_dict() for f in feedbacks_to_export]
        }
        
        with open(output_file, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported {len(feedbacks_to_export)} feedback records to {output_file}")


class FeedbackAnalyzer:
    """Advanced analysis of feedback patterns and trends"""
    
    def __init__(self, feedback_collector: FeedbackCollector):
        self.feedback_collector = feedback_collector
    
    def analyze_model_comparison(self) -> Dict[str, Any]:
        """Compare performance across different models"""
        
        analytics = self.feedback_collector.get_analytics()
        
        if len(analytics.model_ratings) < 2:
            return {'message': 'Need at least 2 models with feedback for comparison'}
        
        # Sort models by rating
        sorted_models = sorted(analytics.model_ratings.items(), key=lambda x: x[1], reverse=True)
        
        # Calculate statistical significance (simplified)
        comparison_results = {}
        
        for i, (model_a, rating_a) in enumerate(sorted_models):
            for model_b, rating_b in sorted_models[i+1:]:
                # Get feedback data for both models
                feedback_a = self.feedback_collector.feedback_by_model[model_a]
                feedback_b = self.feedback_collector.feedback_by_model[model_b]
                
                ratings_a = [f.overall_rating for f in feedback_a if f.overall_rating is not None]
                ratings_b = [f.overall_rating for f in feedback_b if f.overall_rating is not None]
                
                if len(ratings_a) >= 10 and len(ratings_b) >= 10:
                    # Simple statistical comparison
                    mean_diff = rating_a - rating_b
                    effect_size = mean_diff / max(statistics.stdev(ratings_a), statistics.stdev(ratings_b))
                    
                    comparison_results[f"{model_a}_vs_{model_b}"] = {
                        'model_a': model_a,
                        'model_b': model_b,
                        'rating_difference': mean_diff,
                        'effect_size': effect_size,
                        'sample_sizes': [len(ratings_a), len(ratings_b)],
                        'is_significant': abs(effect_size) > 0.5  # Simple threshold
                    }
        
        return {
            'model_rankings': sorted_models,
            'comparisons': comparison_results,
            'summary': f"Best performing model: {sorted_models[0][0]} (avg rating: {sorted_models[0][1]:.2f})"
        }
    
    def analyze_temporal_trends(self) -> Dict[str, Any]:
        """Analyze feedback trends over time"""
        
        feedbacks = self.feedback_collector.feedback_data
        if not feedbacks:
            return {'message': 'No feedback data available'}
        
        # Group by day
        daily_stats = defaultdict(lambda: {'ratings': [], 'count': 0, 'thumbs_up': 0})
        
        for feedback in feedbacks:
            day_key = feedback.timestamp.strftime('%Y-%m-%d')
            daily_stats[day_key]['count'] += 1
            
            if feedback.overall_rating is not None:
                daily_stats[day_key]['ratings'].append(feedback.overall_rating)
            
            if feedback.thumbs_up:
                daily_stats[day_key]['thumbs_up'] += 1
        
        # Calculate trends
        timeline_data = []
        for day, stats in sorted(daily_stats.items()):
            avg_rating = statistics.mean(stats['ratings']) if stats['ratings'] else None
            thumbs_up_pct = (stats['thumbs_up'] / stats['count'] * 100) if stats['count'] > 0 else 0
            
            timeline_data.append({
                'date': day,
                'feedback_count': stats['count'],
                'avg_rating': avg_rating,
                'thumbs_up_percentage': thumbs_up_pct
            })
        
        # Trend analysis (simple linear trend)
        if len(timeline_data) >= 7:  # Need at least a week of data
            recent_week = timeline_data[-7:]
            older_week = timeline_data[-14:-7] if len(timeline_data) >= 14 else timeline_data[:-7]
            
            recent_avg = statistics.mean([d['avg_rating'] for d in recent_week if d['avg_rating']])
            older_avg = statistics.mean([d['avg_rating'] for d in older_week if d['avg_rating']])
            
            trend_direction = "improving" if recent_avg > older_avg else "declining" if recent_avg < older_avg else "stable"
            trend_magnitude = abs(recent_avg - older_avg) if recent_avg and older_avg else 0
        else:
            trend_direction = "insufficient_data"
            trend_magnitude = 0
        
        return {
            'timeline_data': timeline_data,
            'trend_analysis': {
                'direction': trend_direction,
                'magnitude': trend_magnitude,
                'data_points': len(timeline_data)
            },
            'summary': f"Feedback trend over {len(timeline_data)} days: {trend_direction}"
        }
    
    def identify_improvement_opportunities(self) -> Dict[str, Any]:
        """Identify specific areas for improvement based on feedback"""
        
        feedbacks = self.feedback_collector.feedback_data
        analytics = self.feedback_collector.get_analytics()
        
        opportunities = []
        
        # Low-rated categories
        category_ratings = {
            'helpfulness': analytics.avg_helpfulness,
            'accuracy': analytics.avg_accuracy,
            'empathy': analytics.avg_empathy,
            'safety': analytics.avg_safety,
            'clarity': analytics.avg_clarity,
            'response_time': analytics.avg_response_time
        }
        
        for category, rating in category_ratings.items():
            if rating > 0 and rating < 3.5:  # Below 3.5/5 is concerning
                opportunities.append({
                    'type': 'low_rating_category',
                    'category': category,
                    'current_rating': rating,
                    'priority': 'high' if rating < 3.0 else 'medium',
                    'recommendation': f"Focus on improving {category} - currently rated {rating:.2f}/5"
                })
        
        # Models with poor performance
        for model, rating in analytics.model_ratings.items():
            feedback_count = analytics.model_feedback_counts.get(model, 0)
            if feedback_count >= 10 and rating < 3.5:
                opportunities.append({
                    'type': 'underperforming_model',
                    'model': model,
                    'current_rating': rating,
                    'feedback_count': feedback_count,
                    'priority': 'high',
                    'recommendation': f"Review model {model} performance - rated {rating:.2f}/5 across {feedback_count} responses"
                })
        
        # Prompt types with issues
        for prompt_type, rating in analytics.prompt_type_ratings.items():
            if rating < 3.5:
                opportunities.append({
                    'type': 'problematic_prompt_type',
                    'prompt_type': prompt_type,
                    'current_rating': rating,
                    'priority': 'high' if prompt_type in ['crisis', 'anxiety', 'depression'] else 'medium',
                    'recommendation': f"Improve responses for {prompt_type} prompts - currently rated {rating:.2f}/5"
                })
        
        # Common improvement suggestions
        improvement_keywords = defaultdict(int)
        for feedback in feedbacks:
            if feedback.improvement_suggestions:
                text = feedback.improvement_suggestions.lower()
                for keyword in ['faster', 'speed', 'slow', 'empathy', 'understanding', 'accuracy', 'clarity', 'safety']:
                    if keyword in text:
                        improvement_keywords[keyword] += 1
        
        for keyword, count in improvement_keywords.items():
            if count >= 3:  # At least 3 mentions
                opportunities.append({
                    'type': 'user_suggestion',
                    'keyword': keyword,
                    'mention_count': count,
                    'priority': 'medium',
                    'recommendation': f"Users frequently mention '{keyword}' in improvement suggestions ({count} times)"
                })
        
        # Sort by priority
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        opportunities.sort(key=lambda x: priority_order.get(x['priority'], 3))
        
        return {
            'total_opportunities': len(opportunities),
            'high_priority_count': len([o for o in opportunities if o['priority'] == 'high']),
            'opportunities': opportunities,
            'summary': f"Identified {len(opportunities)} improvement opportunities, {len([o for o in opportunities if o['priority'] == 'high'])} high priority"
        }