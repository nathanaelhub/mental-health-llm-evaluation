"""
Conversation Summarizer for Long Mental Health Conversations

Provides intelligent summarization capabilities to manage token limits
while preserving important therapeutic context and safety information.
"""

import logging
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass
from datetime import datetime, timedelta
from enum import Enum

from .conversation_session_manager import Message, MessageRole, ConversationSession
from ..evaluation.evaluation_metrics import TherapeuticEvaluator

logger = logging.getLogger(__name__)


class SummarizationType(Enum):
    """Types of conversation summarization"""
    EXTRACTIVE = "extractive"  # Select key messages
    ABSTRACTIVE = "abstractive"  # Generate new summary
    HYBRID = "hybrid"  # Combination approach
    CRISIS_PRESERVING = "crisis_preserving"  # Preserve safety-critical content


@dataclass
class SummarySegment:
    """A segment of conversation summary"""
    start_message_id: str
    end_message_id: str
    summary_text: str
    key_topics: List[str]
    safety_notes: List[str]
    preserved_messages: List[str]  # IDs of messages that must be kept
    timestamp_range: Tuple[datetime, datetime]
    token_count: int
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'start_message_id': self.start_message_id,
            'end_message_id': self.end_message_id,
            'summary_text': self.summary_text,
            'key_topics': self.key_topics,
            'safety_notes': self.safety_notes,
            'preserved_messages': self.preserved_messages,
            'timestamp_range': [ts.isoformat() for ts in self.timestamp_range],
            'token_count': self.token_count
        }


@dataclass
class ConversationSummary:
    """Complete conversation summary with metadata"""
    session_id: str
    summary_segments: List[SummarySegment]
    original_message_count: int
    summarized_message_count: int
    preserved_message_count: int
    total_token_reduction: int
    summary_type: SummarizationType
    created_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'session_id': self.session_id,
            'summary_segments': [seg.to_dict() for seg in self.summary_segments],
            'original_message_count': self.original_message_count,
            'summarized_message_count': self.summarized_message_count,
            'preserved_message_count': self.preserved_message_count,
            'total_token_reduction': self.total_token_reduction,
            'summary_type': self.summary_type.value,
            'created_at': self.created_at.isoformat()
        }


class ConversationSummarizer:
    """
    Intelligent conversation summarizer for mental health conversations
    
    Features:
    - Crisis-aware summarization that preserves safety-critical content
    - Therapeutic context preservation
    - Token-efficient compression
    - Multiple summarization strategies
    - Quality assessment of summaries
    """
    
    def __init__(self,
                 max_segment_messages: int = 10,
                 min_segment_messages: int = 4,
                 preserve_recent_messages: int = 5,
                 safety_threshold: float = 0.7):
        """
        Initialize conversation summarizer
        
        Args:
            max_segment_messages: Maximum messages per summary segment
            min_segment_messages: Minimum messages to consider for summarization
            preserve_recent_messages: Number of recent messages to always preserve
            safety_threshold: Safety score below which messages are preserved
        """
        self.max_segment_messages = max_segment_messages
        self.min_segment_messages = min_segment_messages
        self.preserve_recent_messages = preserve_recent_messages
        self.safety_threshold = safety_threshold
        
        # Initialize evaluator for safety assessment
        self.evaluator = TherapeuticEvaluator()
        
        # Crisis keywords that indicate messages should be preserved
        self.crisis_keywords = [
            'suicide', 'kill myself', 'end it all', 'hurt myself', 'self harm',
            'crisis', 'emergency', 'desperate', 'hopeless', 'can\'t go on'
        ]
        
        # Therapeutic keywords that indicate important content
        self.therapeutic_keywords = [
            'coping', 'strategy', 'technique', 'progress', 'goal', 'plan',
            'therapy', 'treatment', 'medication', 'diagnosis', 'breakthrough'
        ]
        
        logger.info("ConversationSummarizer initialized")
    
    async def should_summarize(self, session: ConversationSession) -> bool:
        """
        Determine if a conversation should be summarized
        
        Args:
            session: Conversation session to evaluate
            
        Returns:
            True if summarization is needed
        """
        # Check if we have enough messages to summarize
        if len(session.conversation_history) < self.min_segment_messages + self.preserve_recent_messages:
            return False
        
        # Check token count against model limits
        model_config = self._get_model_config(session.selected_model)
        max_tokens = model_config.get('max_tokens', 4000)
        
        # Consider summarization if we're at 70% of token limit
        return session.total_tokens >= (max_tokens * 0.7)
    
    async def summarize_conversation(self,
                                   session: ConversationSession,
                                   summary_type: SummarizationType = SummarizationType.CRISIS_PRESERVING) -> ConversationSummary:
        """
        Summarize a conversation while preserving important context
        
        Args:
            session: Conversation session to summarize
            summary_type: Type of summarization to perform
            
        Returns:
            ConversationSummary with summarized content
        """
        logger.info(f"Starting {summary_type.value} summarization for session {session.session_id}")
        
        # Analyze conversation for important content
        analysis = await self._analyze_conversation(session)
        
        # Identify messages to preserve
        preserve_indices = self._identify_messages_to_preserve(session, analysis)
        
        # Create summary segments
        segments = await self._create_summary_segments(session, preserve_indices, summary_type)
        
        # Calculate metrics
        original_count = len(session.conversation_history)
        summarized_count = sum(1 for seg in segments for _ in seg.preserved_messages)
        preserved_count = len(preserve_indices)
        
        # Calculate token reduction
        original_tokens = session.total_tokens
        new_tokens = sum(seg.token_count for seg in segments)
        token_reduction = original_tokens - new_tokens
        
        summary = ConversationSummary(
            session_id=session.session_id,
            summary_segments=segments,
            original_message_count=original_count,
            summarized_message_count=len(segments),
            preserved_message_count=preserved_count,
            total_token_reduction=token_reduction,
            summary_type=summary_type,
            created_at=datetime.now()
        )
        
        logger.info(f"Summarization complete: {original_count} messages -> {len(segments)} segments, {token_reduction} tokens saved")
        
        return summary
    
    async def apply_summary_to_session(self,
                                     session: ConversationSession,
                                     summary: ConversationSummary) -> ConversationSession:
        """
        Apply summary to session by replacing old messages with summary
        
        Args:
            session: Original session
            summary: Generated summary
            
        Returns:
            Updated session with summarized history
        """
        # Create new conversation history
        new_history = []
        
        # Add summary segments as system messages
        for segment in summary.summary_segments:
            summary_message = Message(
                message_id=f"summary_{segment.start_message_id}_{segment.end_message_id}",
                role=MessageRole.SYSTEM,
                content=f"[CONVERSATION SUMMARY: {segment.timestamp_range[0].strftime('%Y-%m-%d %H:%M')} - {segment.timestamp_range[1].strftime('%Y-%m-%d %H:%M')}]\n\n{segment.summary_text}",
                timestamp=segment.timestamp_range[0],
                token_count=segment.token_count,
                metadata={
                    'is_summary': True,
                    'original_messages': len(segment.preserved_messages),
                    'key_topics': segment.key_topics,
                    'safety_notes': segment.safety_notes
                }
            )
            new_history.append(summary_message)
            
            # Add any preserved messages from this segment
            for msg_id in segment.preserved_messages:
                original_msg = next((m for m in session.conversation_history if m.message_id == msg_id), None)
                if original_msg:
                    new_history.append(original_msg)
        
        # Always preserve the most recent messages
        recent_messages = session.conversation_history[-self.preserve_recent_messages:]
        for msg in recent_messages:
            if msg not in new_history:
                new_history.append(msg)
        
        # Sort by timestamp
        new_history.sort(key=lambda m: m.timestamp)
        
        # Update session
        session.conversation_history = new_history
        session.total_tokens = sum(m.token_count or 0 for m in new_history)
        
        # Add summary metadata
        session.metadata['last_summary'] = summary.created_at.isoformat()
        session.metadata['summary_count'] = session.metadata.get('summary_count', 0) + 1
        
        logger.info(f"Applied summary to session {session.session_id}: {len(new_history)} messages, {session.total_tokens} tokens")
        
        return session
    
    async def _analyze_conversation(self, session: ConversationSession) -> Dict[str, Any]:
        """Analyze conversation for important content and patterns"""
        
        analysis = {
            'crisis_messages': [],
            'therapeutic_messages': [],
            'safety_scores': [],
            'topic_distribution': {},
            'user_engagement_patterns': [],
            'model_responses_quality': []
        }
        
        for i, message in enumerate(session.conversation_history):
            # Crisis detection
            if self._contains_crisis_content(message.content):
                analysis['crisis_messages'].append(i)
            
            # Therapeutic content detection
            if self._contains_therapeutic_content(message.content):
                analysis['therapeutic_messages'].append(i)
            
            # Safety scoring for assistant messages
            if message.role == MessageRole.ASSISTANT:
                safety_score = await self._evaluate_message_safety(message, session)
                analysis['safety_scores'].append((i, safety_score))
                
                if safety_score < self.safety_threshold:
                    analysis['crisis_messages'].append(i)
            
            # Topic extraction (simplified)
            topics = self._extract_topics(message.content)
            for topic in topics:
                analysis['topic_distribution'][topic] = analysis['topic_distribution'].get(topic, 0) + 1
        
        return analysis
    
    def _identify_messages_to_preserve(self,
                                     session: ConversationSession,
                                     analysis: Dict[str, Any]) -> List[int]:
        """Identify message indices that must be preserved"""
        
        preserve_indices = set()
        
        # Always preserve crisis-related messages
        preserve_indices.update(analysis['crisis_messages'])
        
        # Preserve therapeutically important messages
        preserve_indices.update(analysis['therapeutic_messages'])
        
        # Preserve messages with low safety scores
        for msg_idx, safety_score in analysis['safety_scores']:
            if safety_score < self.safety_threshold:
                preserve_indices.add(msg_idx)
        
        # Always preserve recent messages
        total_messages = len(session.conversation_history)
        recent_start = max(0, total_messages - self.preserve_recent_messages)
        preserve_indices.update(range(recent_start, total_messages))
        
        # Preserve context around important messages (1 message before/after)
        extended_preserve = set(preserve_indices)
        for idx in preserve_indices:
            if idx > 0:
                extended_preserve.add(idx - 1)
            if idx < total_messages - 1:
                extended_preserve.add(idx + 1)
        
        return sorted(list(extended_preserve))
    
    async def _create_summary_segments(self,
                                     session: ConversationSession,
                                     preserve_indices: List[int],
                                     summary_type: SummarizationType) -> List[SummarySegment]:
        """Create summary segments for the conversation"""
        
        segments = []
        messages = session.conversation_history
        
        # Find ranges of messages that can be summarized
        summarizable_ranges = self._find_summarizable_ranges(len(messages), preserve_indices)
        
        for start_idx, end_idx in summarizable_ranges:
            if end_idx - start_idx < self.min_segment_messages:
                continue
            
            segment_messages = messages[start_idx:end_idx + 1]
            
            # Generate summary for this segment
            summary_text = await self._generate_segment_summary(segment_messages, summary_type)
            
            # Extract key information
            key_topics = self._extract_segment_topics(segment_messages)
            safety_notes = self._extract_safety_notes(segment_messages)
            
            # Count tokens in summary
            summary_tokens = self._estimate_tokens(summary_text, session.selected_model)
            
            segment = SummarySegment(
                start_message_id=segment_messages[0].message_id,
                end_message_id=segment_messages[-1].message_id,
                summary_text=summary_text,
                key_topics=key_topics,
                safety_notes=safety_notes,
                preserved_messages=[],  # No individual messages preserved in this segment
                timestamp_range=(segment_messages[0].timestamp, segment_messages[-1].timestamp),
                token_count=summary_tokens
            )
            
            segments.append(segment)
        
        return segments
    
    def _find_summarizable_ranges(self, total_messages: int, preserve_indices: List[int]) -> List[Tuple[int, int]]:
        """Find ranges of messages that can be summarized"""
        
        ranges = []
        preserve_set = set(preserve_indices)
        
        current_start = None
        
        for i in range(total_messages):
            if i in preserve_set:
                # End current range if we have one
                if current_start is not None and i - current_start >= self.min_segment_messages:
                    ranges.append((current_start, i - 1))
                current_start = None
            else:
                # Start new range if we don't have one
                if current_start is None:
                    current_start = i
        
        # Handle final range
        if current_start is not None and total_messages - current_start >= self.min_segment_messages:
            # Don't summarize if it goes to the very end (preserve recent messages)
            recent_boundary = total_messages - self.preserve_recent_messages
            if current_start < recent_boundary:
                ranges.append((current_start, min(recent_boundary - 1, total_messages - 1)))
        
        return ranges
    
    async def _generate_segment_summary(self,
                                      messages: List[Message],
                                      summary_type: SummarizationType) -> str:
        """Generate summary text for a segment of messages"""
        
        if summary_type == SummarizationType.EXTRACTIVE:
            return self._generate_extractive_summary(messages)
        elif summary_type == SummarizationType.ABSTRACTIVE:
            return await self._generate_abstractive_summary(messages)
        elif summary_type == SummarizationType.CRISIS_PRESERVING:
            return self._generate_crisis_preserving_summary(messages)
        else:  # HYBRID
            return await self._generate_hybrid_summary(messages)
    
    def _generate_extractive_summary(self, messages: List[Message]) -> str:
        """Generate extractive summary by selecting key messages"""
        
        # Score messages by importance
        scored_messages = []
        
        for msg in messages:
            score = 0
            
            # Higher score for longer messages (more content)
            score += min(len(msg.content) / 100, 3)
            
            # Higher score for messages with therapeutic keywords
            if self._contains_therapeutic_content(msg.content):
                score += 2
            
            # Higher score for questions (engagement)
            if '?' in msg.content:
                score += 1
            
            # Higher score for user messages (preserve user voice)
            if msg.role == MessageRole.USER:
                score += 1
            
            scored_messages.append((score, msg))
        
        # Select top messages
        scored_messages.sort(reverse=True, key=lambda x: x[0])
        selected_count = min(3, len(messages) // 2)  # Select up to 3 or half the messages
        
        selected_messages = [msg for _, msg in scored_messages[:selected_count]]
        selected_messages.sort(key=lambda m: m.timestamp)  # Restore chronological order
        
        # Create summary
        summary_parts = []
        for msg in selected_messages:
            role_prefix = "User" if msg.role == MessageRole.USER else "Assistant"
            summary_parts.append(f"{role_prefix}: {msg.content[:200]}{'...' if len(msg.content) > 200 else ''}")
        
        return f"Key exchanges:\n" + "\n\n".join(summary_parts)
    
    async def _generate_abstractive_summary(self, messages: List[Message]) -> str:
        """Generate abstractive summary (would use LLM in production)"""
        
        # In a production system, this would use an LLM to generate a new summary
        # For now, we'll create a structured summary
        
        user_messages = [m for m in messages if m.role == MessageRole.USER]
        assistant_messages = [m for m in messages if m.role == MessageRole.ASSISTANT]
        
        # Extract main themes
        themes = set()
        for msg in messages:
            themes.update(self._extract_topics(msg.content))
        
        # Create structured summary
        summary_parts = [
            f"Conversation covered {len(themes)} main topics: {', '.join(list(themes)[:5])}.",
            f"User shared {len(user_messages)} messages, assistant provided {len(assistant_messages)} responses.",
        ]
        
        # Add therapeutic context if present
        therapeutic_messages = [m for m in messages if self._contains_therapeutic_content(m.content)]
        if therapeutic_messages:
            summary_parts.append(f"Discussion included therapeutic strategies and coping techniques.")
        
        # Add safety context if relevant
        crisis_messages = [m for m in messages if self._contains_crisis_content(m.content)]
        if crisis_messages:
            summary_parts.append("⚠️ Conversation contained safety-sensitive content.")
        
        return " ".join(summary_parts)
    
    def _generate_crisis_preserving_summary(self, messages: List[Message]) -> str:
        """Generate summary that emphasizes safety-critical information"""
        
        crisis_messages = []
        therapeutic_messages = []
        general_messages = []
        
        for msg in messages:
            if self._contains_crisis_content(msg.content):
                crisis_messages.append(msg)
            elif self._contains_therapeutic_content(msg.content):
                therapeutic_messages.append(msg)
            else:
                general_messages.append(msg)
        
        summary_parts = []
        
        # Crisis content first
        if crisis_messages:
            summary_parts.append("⚠️ SAFETY-CRITICAL CONTENT:")
            for msg in crisis_messages[:2]:  # Limit to avoid too long
                role = "User" if msg.role == MessageRole.USER else "Assistant"
                summary_parts.append(f"  {role}: {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}")
        
        # Therapeutic content
        if therapeutic_messages:
            summary_parts.append("\n🔧 THERAPEUTIC CONTENT:")
            for msg in therapeutic_messages[:2]:
                role = "User" if msg.role == MessageRole.USER else "Assistant"
                summary_parts.append(f"  {role}: {msg.content[:150]}{'...' if len(msg.content) > 150 else ''}")
        
        # General summary
        if general_messages:
            topics = set()
            for msg in general_messages:
                topics.update(self._extract_topics(msg.content))
            
            if topics:
                summary_parts.append(f"\n💬 GENERAL DISCUSSION: {', '.join(list(topics)[:3])}")
        
        return "\n".join(summary_parts)
    
    async def _generate_hybrid_summary(self, messages: List[Message]) -> str:
        """Generate hybrid summary combining extractive and abstractive approaches"""
        
        # Get extractive summary for key exchanges
        extractive = self._generate_extractive_summary(messages)
        
        # Get abstractive summary for overview
        abstractive = await self._generate_abstractive_summary(messages)
        
        return f"OVERVIEW: {abstractive}\n\nKEY EXCHANGES:\n{extractive}"
    
    def _contains_crisis_content(self, content: str) -> bool:
        """Check if content contains crisis-related keywords"""
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self.crisis_keywords)
    
    def _contains_therapeutic_content(self, content: str) -> bool:
        """Check if content contains therapeutic keywords"""
        content_lower = content.lower()
        return any(keyword in content_lower for keyword in self.therapeutic_keywords)
    
    def _extract_topics(self, content: str) -> List[str]:
        """Extract main topics from content (simplified implementation)"""
        
        # Simple keyword-based topic extraction
        topics = []
        content_lower = content.lower()
        
        topic_keywords = {
            'anxiety': ['anxiety', 'anxious', 'worry', 'panic', 'stress'],
            'depression': ['depression', 'depressed', 'sad', 'hopeless'],
            'relationships': ['relationship', 'partner', 'family', 'friends'],
            'work': ['work', 'job', 'career', 'workplace', 'boss'],
            'therapy': ['therapy', 'counseling', 'treatment', 'session'],
            'coping': ['coping', 'cope', 'manage', 'deal with', 'handle'],
            'medication': ['medication', 'pills', 'prescription', 'medicine']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics
    
    def _extract_segment_topics(self, messages: List[Message]) -> List[str]:
        """Extract key topics from a segment of messages"""
        
        all_topics = set()
        for msg in messages:
            all_topics.update(self._extract_topics(msg.content))
        
        return list(all_topics)
    
    def _extract_safety_notes(self, messages: List[Message]) -> List[str]:
        """Extract safety-related notes from messages"""
        
        safety_notes = []
        
        for msg in messages:
            if self._contains_crisis_content(msg.content):
                safety_notes.append(f"Crisis content detected at {msg.timestamp.strftime('%H:%M')}")
            
            if msg.safety_score is not None and msg.safety_score < self.safety_threshold:
                safety_notes.append(f"Low safety score ({msg.safety_score:.2f}) at {msg.timestamp.strftime('%H:%M')}")
        
        return safety_notes
    
    async def _evaluate_message_safety(self, message: Message, session: ConversationSession) -> float:
        """Evaluate safety score of a message"""
        
        try:
            evaluation = self.evaluator.evaluate_response(
                prompt="",  # We don't have the original prompt context
                response=message.content,
                response_time_ms=0,
                input_tokens=0,
                output_tokens=message.token_count or 0
            )
            
            if hasattr(evaluation, 'safety_score'):
                return evaluation.safety_score / 10.0  # Normalize to 0-1
            elif isinstance(evaluation, dict):
                return evaluation.get('safety_score', 10.0) / 10.0
            
        except Exception as e:
            logger.error(f"Error evaluating message safety: {e}")
        
        return 1.0  # Default to safe if evaluation fails
    
    def _estimate_tokens(self, text: str, model: str) -> int:
        """Estimate token count for text"""
        # Simplified token estimation
        return int(len(text.split()) * 1.3)
    
    def _get_model_config(self, model: str) -> Dict[str, Any]:
        """Get configuration for a specific model"""
        
        configs = {
            'openai': {'max_tokens': 4096},
            'claude': {'max_tokens': 8192},
            'deepseek': {'max_tokens': 4096},
            'gemma': {'max_tokens': 2048}
        }
        
        return configs.get(model, {'max_tokens': 4000})