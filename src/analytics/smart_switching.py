"""
Smart Model Switching System

Intelligent mid-conversation model switching that dynamically adapts model selection
based on conversation context, performance degradation detection, and user needs.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics

from ..chat.dynamic_model_selector import PromptType, ModelSelection, DynamicModelSelector
from ..chat.conversation_session_manager import ConversationSession, SessionStatus

logger = logging.getLogger(__name__)


class SwitchReason(Enum):
    """Reasons for model switching"""
    PERFORMANCE_DEGRADATION = "performance_degradation"
    CONTEXT_ESCALATION = "context_escalation"           # Crisis escalation
    CONTEXT_DE_ESCALATION = "context_de_escalation"     # Crisis de-escalation
    USER_DISSATISFACTION = "user_dissatisfaction"       # Poor user feedback
    COST_OPTIMIZATION = "cost_optimization"             # Switch to cheaper model
    SPEED_OPTIMIZATION = "speed_optimization"           # Switch to faster model
    CAPABILITY_MISMATCH = "capability_mismatch"         # Current model inadequate
    CONVERSATION_FLOW = "conversation_flow"             # Natural conversation transition
    MANUAL_OVERRIDE = "manual_override"                 # Explicit user request


class SwitchStrategy(Enum):
    """Strategies for model switching"""
    IMMEDIATE = "immediate"           # Switch immediately
    GRADUAL = "gradual"              # Gradually transition
    CONTEXT_PRESERVED = "context_preserved"  # Maintain conversation context
    SEAMLESS = "seamless"            # Invisible to user
    TRANSPARENT = "transparent"      # Notify user of switch


@dataclass
class SwitchingDecision:
    """Decision to switch models"""
    decision_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    
    # Context
    session_id: str = ""
    current_model: str = ""
    recommended_model: str = ""
    
    # Decision factors
    switch_reason: SwitchReason = SwitchReason.PERFORMANCE_DEGRADATION
    confidence_score: float = 0.0
    urgency_level: int = 1  # 1=low, 5=critical
    
    # Metrics that led to decision
    current_model_score: float = 0.0
    recommended_model_score: float = 0.0
    performance_delta: float = 0.0
    
    # Strategy
    switch_strategy: SwitchStrategy = SwitchStrategy.SEAMLESS
    notification_message: Optional[str] = None
    
    # Execution
    should_switch: bool = True
    estimated_improvement: float = 0.0
    switch_cost: float = 0.0  # Context loss, latency, etc.
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'decision_id': self.decision_id,
            'session_id': self.session_id,
            'current_model': self.current_model,
            'recommended_model': self.recommended_model,
            'switch_reason': self.switch_reason.value,
            'confidence_score': self.confidence_score,
            'urgency_level': self.urgency_level,
            'current_model_score': self.current_model_score,
            'recommended_model_score': self.recommended_model_score,
            'performance_delta': self.performance_delta,
            'switch_strategy': self.switch_strategy.value,
            'notification_message': self.notification_message,
            'should_switch': self.should_switch,
            'estimated_improvement': self.estimated_improvement,
            'switch_cost': self.switch_cost,
            'timestamp': self.timestamp.isoformat(),
            'metadata': self.metadata
        }


@dataclass
class ConversationContext:
    """Current conversation context for switching decisions"""
    session_id: str
    user_id: str
    current_model: str
    
    # Conversation history
    message_count: int = 0
    conversation_duration_minutes: float = 0.0
    recent_messages: List[Dict[str, Any]] = field(default_factory=list)
    
    # Performance tracking
    recent_response_times: List[float] = field(default_factory=list)
    recent_user_ratings: List[float] = field(default_factory=list)
    model_performance_trend: float = 0.0  # Positive = improving, negative = degrading
    
    # Context analysis
    dominant_prompt_type: PromptType = PromptType.GENERAL_SUPPORT
    crisis_level: float = 0.0  # 0-1 scale
    emotional_intensity: float = 0.5  # 0-1 scale
    topic_complexity: float = 0.5  # 0-1 scale
    
    # User behavior
    user_engagement_score: float = 0.5  # Based on response times, follow-ups
    user_satisfaction_trend: float = 0.0  # Recent satisfaction trend
    
    # Technical factors
    current_context_length: int = 0
    model_capabilities_used: Set[str] = field(default_factory=set)
    
    def update_performance_metrics(self, response_time: float, user_rating: float = None):
        """Update performance tracking"""
        self.recent_response_times.append(response_time)
        if len(self.recent_response_times) > 10:
            self.recent_response_times.pop(0)
        
        if user_rating is not None:
            self.recent_user_ratings.append(user_rating)
            if len(self.recent_user_ratings) > 5:
                self.recent_user_ratings.pop(0)
        
        # Calculate trend (simple linear regression)
        if len(self.recent_user_ratings) >= 3:
            ratings = self.recent_user_ratings
            x_vals = list(range(len(ratings)))
            n = len(ratings)
            
            sum_x = sum(x_vals)
            sum_y = sum(ratings)
            sum_xy = sum(x * y for x, y in zip(x_vals, ratings))
            sum_x_sq = sum(x * x for x in x_vals)
            
            # Calculate slope (trend)
            denominator = n * sum_x_sq - sum_x * sum_x
            if denominator != 0:
                self.model_performance_trend = (n * sum_xy - sum_x * sum_y) / denominator


@dataclass
class SwitchingConfig:
    """Configuration for smart model switching"""
    
    # Performance thresholds
    min_performance_threshold: float = 3.0  # Minimum acceptable user rating
    performance_degradation_threshold: float = 0.5  # Trigger switch if performance drops by this much
    
    # Response time thresholds
    max_acceptable_response_time_ms: float = 10000  # 10 seconds
    response_time_degradation_factor: float = 2.0   # Switch if 2x slower than baseline
    
    # Context switching rules
    crisis_escalation_threshold: float = 0.7   # Switch to crisis model above this level
    crisis_de_escalation_threshold: float = 0.3 # Switch away from crisis model below this
    
    # User satisfaction rules
    consecutive_low_ratings_threshold: int = 2  # Switch after N consecutive low ratings
    satisfaction_improvement_threshold: float = 0.3  # Expected improvement to justify switch
    
    # Cost considerations
    enable_cost_optimization: bool = True
    max_cost_increase_factor: float = 1.5  # Don't switch if cost increases by more than 50%
    
    # Switching frequency limits
    min_time_between_switches_minutes: float = 5.0  # Prevent too frequent switching
    max_switches_per_session: int = 3  # Maximum switches in one conversation
    
    # Notification settings
    notify_user_of_switches: bool = False  # Whether to inform user of model switches
    transparency_level: str = "minimal"    # "none", "minimal", "detailed"


class ConversationOptimizer:
    """Analyzes conversation context and recommends optimizations"""
    
    def __init__(self):
        # Model capability profiles
        self.model_capabilities = {
            "claude-3-opus": {
                "crisis_handling": 0.95,
                "empathy": 0.90,
                "complex_reasoning": 0.95,
                "speed": 0.6,
                "cost": 0.2,
                "context_length": 0.95
            },
            "claude-3-sonnet": {
                "crisis_handling": 0.85,
                "empathy": 0.85,
                "complex_reasoning": 0.85,
                "speed": 0.8,
                "cost": 0.6,
                "context_length": 0.90
            },
            "gpt-4-turbo": {
                "crisis_handling": 0.80,
                "empathy": 0.75,
                "complex_reasoning": 0.90,
                "speed": 0.7,
                "cost": 0.4,
                "context_length": 0.85
            },
            "gpt-3.5-turbo": {
                "crisis_handling": 0.70,
                "empathy": 0.65,
                "complex_reasoning": 0.70,
                "speed": 0.95,
                "cost": 0.9,
                "context_length": 0.70
            },
            "claude-3-haiku": {
                "crisis_handling": 0.75,
                "empathy": 0.80,
                "complex_reasoning": 0.75,
                "speed": 0.95,
                "cost": 0.95,
                "context_length": 0.85
            }
        }
        
        # Capability requirements by prompt type
        self.prompt_type_requirements = {
            PromptType.CRISIS: {
                "crisis_handling": 0.9,
                "empathy": 0.8,
                "speed": 0.7,
                "complex_reasoning": 0.6
            },
            PromptType.ANXIETY: {
                "empathy": 0.8,
                "crisis_handling": 0.6,
                "complex_reasoning": 0.7,
                "speed": 0.6
            },
            PromptType.DEPRESSION: {
                "empathy": 0.85,
                "crisis_handling": 0.7,
                "complex_reasoning": 0.7,
                "speed": 0.6
            },
            PromptType.INFORMATION_SEEKING: {
                "complex_reasoning": 0.8,
                "speed": 0.8,
                "empathy": 0.4,
                "crisis_handling": 0.3
            },
            PromptType.GENERAL_WELLNESS: {
                "empathy": 0.6,
                "speed": 0.8,
                "cost": 0.8,
                "complex_reasoning": 0.5
            }
        }
    
    def calculate_model_suitability(self, 
                                  model_name: str, 
                                  context: ConversationContext,
                                  requirements_override: Dict[str, float] = None) -> float:
        """Calculate how suitable a model is for the current context"""
        
        if model_name not in self.model_capabilities:
            return 0.5  # Unknown model, neutral score
        
        capabilities = self.model_capabilities[model_name]
        requirements = requirements_override or self.prompt_type_requirements.get(
            context.dominant_prompt_type, {}
        )
        
        suitability_score = 0.0
        total_weight = 0.0
        
        # Calculate weighted suitability
        for capability, requirement in requirements.items():
            if capability in capabilities:
                capability_score = capabilities[capability]
                weight = requirement
                
                # Apply context-specific adjustments
                if capability == "crisis_handling":
                    weight *= (1 + context.crisis_level)
                elif capability == "empathy":
                    weight *= (1 + context.emotional_intensity)
                elif capability == "complex_reasoning":
                    weight *= (1 + context.topic_complexity)
                elif capability == "speed" and context.recent_response_times:
                    avg_response_time = statistics.mean(context.recent_response_times)
                    if avg_response_time > 5000:  # If responses are slow, prioritize speed
                        weight *= 1.5
                
                suitability_score += capability_score * weight
                total_weight += weight
        
        # Normalize to 0-1 range
        if total_weight > 0:
            suitability_score /= total_weight
        
        # Apply conversation-specific adjustments
        if context.current_context_length > 50000:  # Long conversation
            context_capability = capabilities.get("context_length", 0.8)
            suitability_score *= context_capability
        
        return min(1.0, max(0.0, suitability_score))
    
    def detect_context_changes(self, context: ConversationContext) -> List[str]:
        """Detect significant context changes that might warrant model switching"""
        
        changes = []
        
        # Crisis level changes
        if context.crisis_level > 0.7:
            changes.append("crisis_escalation")
        elif context.crisis_level < 0.3 and context.current_model == "claude-3-opus":
            changes.append("crisis_de_escalation")
        
        # Emotional intensity changes
        if context.emotional_intensity > 0.8:
            changes.append("high_emotional_intensity")
        elif context.emotional_intensity < 0.3:
            changes.append("low_emotional_intensity")
        
        # Topic complexity changes
        if context.topic_complexity > 0.8:
            changes.append("high_complexity")
        elif context.topic_complexity < 0.3:
            changes.append("low_complexity")
        
        # Performance degradation
        if context.model_performance_trend < -0.3:
            changes.append("performance_degradation")
        
        # User engagement changes
        if context.user_engagement_score < 0.3:
            changes.append("low_engagement")
        
        return changes
    
    def analyze_conversation_patterns(self, context: ConversationContext) -> Dict[str, Any]:
        """Analyze conversation patterns for optimization opportunities"""
        
        analysis = {
            'conversation_phase': self._identify_conversation_phase(context),
            'user_state': self._assess_user_state(context),
            'model_performance': self._evaluate_model_performance(context),
            'optimization_opportunities': []
        }
        
        # Identify optimization opportunities
        context_changes = self.detect_context_changes(context)
        
        for change in context_changes:
            if change in ["crisis_escalation", "high_emotional_intensity"]:
                analysis['optimization_opportunities'].append({
                    'type': 'capability_upgrade',
                    'reason': change,
                    'recommendation': 'Switch to model with higher empathy/crisis handling'
                })
            elif change in ["crisis_de_escalation", "low_complexity"]:
                analysis['optimization_opportunities'].append({
                    'type': 'cost_optimization',
                    'reason': change,
                    'recommendation': 'Switch to more cost-effective model'
                })
            elif change == "performance_degradation":
                analysis['optimization_opportunities'].append({
                    'type': 'performance_improvement',
                    'reason': change,
                    'recommendation': 'Switch to better performing model'
                })
        
        return analysis
    
    def _identify_conversation_phase(self, context: ConversationContext) -> str:
        """Identify the current phase of conversation"""
        
        if context.message_count <= 2:
            return "opening"
        elif context.conversation_duration_minutes < 5:
            return "early"
        elif context.conversation_duration_minutes < 15:
            return "middle"
        elif context.conversation_duration_minutes < 30:
            return "late"
        else:
            return "extended"
    
    def _assess_user_state(self, context: ConversationContext) -> Dict[str, float]:
        """Assess current user emotional/mental state"""
        
        return {
            'crisis_level': context.crisis_level,
            'emotional_intensity': context.emotional_intensity,
            'engagement': context.user_engagement_score,
            'satisfaction_trend': context.user_satisfaction_trend
        }
    
    def _evaluate_model_performance(self, context: ConversationContext) -> Dict[str, Any]:
        """Evaluate current model performance"""
        
        avg_rating = statistics.mean(context.recent_user_ratings) if context.recent_user_ratings else None
        avg_response_time = statistics.mean(context.recent_response_times) if context.recent_response_times else None
        
        return {
            'average_rating': avg_rating,
            'average_response_time_ms': avg_response_time,
            'performance_trend': context.model_performance_trend,
            'sample_size': len(context.recent_user_ratings)
        }


class SmartModelSwitcher:
    """
    Smart model switching system for mid-conversation optimization
    
    Features:
    - Performance degradation detection
    - Context-aware model recommendations
    - Seamless model transitions with context preservation
    - User notification management
    - Cost-benefit analysis of switches
    """
    
    def __init__(self, 
                 model_selector: DynamicModelSelector,
                 config: SwitchingConfig = None):
        
        self.model_selector = model_selector
        self.config = config or SwitchingConfig()
        self.optimizer = ConversationOptimizer()
        
        # Session tracking
        self.active_contexts: Dict[str, ConversationContext] = {}
        self.switch_history: List[SwitchingDecision] = []
        self.session_switch_counts: Dict[str, int] = defaultdict(int)
        
        # Performance baselines
        self.model_baselines: Dict[str, Dict[str, float]] = defaultdict(dict)
        
        logger.info("SmartModelSwitcher initialized")
    
    def register_session(self, 
                        session_id: str, 
                        user_id: str, 
                        initial_model: str,
                        prompt_type: PromptType = PromptType.GENERAL_SUPPORT):
        """Register a new conversation session for monitoring"""
        
        context = ConversationContext(
            session_id=session_id,
            user_id=user_id,
            current_model=initial_model,
            dominant_prompt_type=prompt_type
        )
        
        self.active_contexts[session_id] = context
        logger.info(f"Registered session {session_id} with model {initial_model}")
    
    def update_session_context(self, 
                             session_id: str,
                             message_text: str = None,
                             prompt_type: PromptType = None,
                             response_time_ms: float = None,
                             user_rating: float = None,
                             crisis_indicators: Dict[str, float] = None):
        """Update session context with new information"""
        
        if session_id not in self.active_contexts:
            logger.warning(f"Session {session_id} not registered")
            return
        
        context = self.active_contexts[session_id]
        context.message_count += 1
        
        # Update prompt type if provided
        if prompt_type:
            context.dominant_prompt_type = prompt_type
        
        # Update performance metrics
        if response_time_ms is not None:
            context.update_performance_metrics(response_time_ms, user_rating)
        
        # Update context analysis
        if crisis_indicators:
            context.crisis_level = crisis_indicators.get('crisis_level', context.crisis_level)
            context.emotional_intensity = crisis_indicators.get('emotional_intensity', context.emotional_intensity)
            context.topic_complexity = crisis_indicators.get('topic_complexity', context.topic_complexity)
        
        # Analyze message for context clues
        if message_text:
            self._analyze_message_context(context, message_text)
        
        logger.debug(f"Updated context for session {session_id}")
    
    async def evaluate_switching_opportunity(self, session_id: str) -> Optional[SwitchingDecision]:
        """Evaluate whether model switching would be beneficial"""
        
        if session_id not in self.active_contexts:
            return None
        
        context = self.active_contexts[session_id]
        
        # Check if switching is allowed
        if not self._can_switch_model(session_id):
            return None
        
        # Analyze current situation
        analysis = self.optimizer.analyze_conversation_patterns(context)
        
        # Find best alternative model
        best_alternative = await self._find_best_alternative_model(context)
        
        if not best_alternative:
            return None
        
        alternative_model, alternative_score = best_alternative
        current_score = self.optimizer.calculate_model_suitability(context.current_model, context)
        
        # Create switching decision
        decision = SwitchingDecision(
            session_id=session_id,
            current_model=context.current_model,
            recommended_model=alternative_model,
            current_model_score=current_score,
            recommended_model_score=alternative_score,
            performance_delta=alternative_score - current_score
        )
        
        # Determine switch reason and strategy
        decision.switch_reason, decision.urgency_level = self._determine_switch_reason(context, analysis)
        decision.switch_strategy = self._determine_switch_strategy(decision.switch_reason, decision.urgency_level)
        
        # Calculate confidence and benefits
        decision.confidence_score = self._calculate_switch_confidence(context, decision)
        decision.estimated_improvement = decision.performance_delta * decision.confidence_score
        decision.switch_cost = self._calculate_switch_cost(context, decision)
        
        # Final decision
        decision.should_switch = self._should_execute_switch(decision)
        
        if decision.should_switch:
            decision.notification_message = self._generate_notification_message(decision)
        
        logger.info(f"Evaluated switching for {session_id}: {'SWITCH' if decision.should_switch else 'NO SWITCH'} "
                   f"({decision.switch_reason.value}, confidence: {decision.confidence_score:.2f})")
        
        return decision
    
    async def execute_model_switch(self, 
                                 session_id: str, 
                                 decision: SwitchingDecision,
                                 preserve_context: bool = True) -> bool:
        """Execute model switch with context preservation"""
        
        try:
            context = self.active_contexts.get(session_id)
            if not context:
                logger.error(f"Session {session_id} not found for model switch")
                return False
            
            old_model = context.current_model
            new_model = decision.recommended_model
            
            # Update context tracking
            context.current_model = new_model
            self.session_switch_counts[session_id] += 1
            
            # Record switch in history
            decision.timestamp = datetime.now()
            self.switch_history.append(decision)
            
            # In a real implementation, you would:
            # 1. Preserve conversation context/memory
            # 2. Initialize new model with context
            # 3. Update model selector configuration
            # 4. Handle any API client switching
            
            logger.info(f"Executed model switch in session {session_id}: {old_model} -> {new_model} "
                       f"(reason: {decision.switch_reason.value})")
            
            return True
            
        except Exception as e:
            logger.error(f"Error executing model switch: {e}")
            return False
    
    def _can_switch_model(self, session_id: str) -> bool:
        """Check if model switching is allowed for this session"""
        
        # Check switch frequency limits
        if self.session_switch_counts[session_id] >= self.config.max_switches_per_session:
            return False
        
        # Check time since last switch
        recent_switches = [s for s in self.switch_history 
                          if s.session_id == session_id and 
                          datetime.now() - s.timestamp <= timedelta(minutes=self.config.min_time_between_switches_minutes)]
        
        if recent_switches:
            return False
        
        return True
    
    async def _find_best_alternative_model(self, context: ConversationContext) -> Optional[Tuple[str, float]]:
        """Find the best alternative model for current context"""
        
        available_models = list(self.optimizer.model_capabilities.keys())
        available_models = [m for m in available_models if m != context.current_model]
        
        if not available_models:
            return None
        
        best_model = None
        best_score = 0.0
        
        for model in available_models:
            score = self.optimizer.calculate_model_suitability(model, context)
            
            if score > best_score:
                best_score = score
                best_model = model
        
        # Only recommend if significantly better
        current_score = self.optimizer.calculate_model_suitability(context.current_model, context)
        
        if best_score > current_score + self.config.satisfaction_improvement_threshold:
            return best_model, best_score
        
        return None
    
    def _determine_switch_reason(self, context: ConversationContext, analysis: Dict[str, Any]) -> Tuple[SwitchReason, int]:
        """Determine the reason for switching and urgency level"""
        
        # Crisis escalation (highest priority)
        if context.crisis_level > self.config.crisis_escalation_threshold:
            return SwitchReason.CONTEXT_ESCALATION, 5
        
        # Performance degradation
        if (context.recent_user_ratings and 
            len(context.recent_user_ratings) >= self.config.consecutive_low_ratings_threshold):
            recent_avg = statistics.mean(context.recent_user_ratings[-self.config.consecutive_low_ratings_threshold:])
            if recent_avg < self.config.min_performance_threshold:
                return SwitchReason.USER_DISSATISFACTION, 4
        
        # Response time issues
        if context.recent_response_times:
            avg_response_time = statistics.mean(context.recent_response_times)
            if avg_response_time > self.config.max_acceptable_response_time_ms:
                return SwitchReason.SPEED_OPTIMIZATION, 3
        
        # Context de-escalation (cost optimization opportunity)
        if (context.crisis_level < self.config.crisis_de_escalation_threshold and 
            context.current_model == "claude-3-opus"):
            return SwitchReason.CONTEXT_DE_ESCALATION, 2
        
        # Capability mismatch
        current_suitability = self.optimizer.calculate_model_suitability(context.current_model, context)
        if current_suitability < 0.6:
            return SwitchReason.CAPABILITY_MISMATCH, 3
        
        # General optimization
        return SwitchReason.CONVERSATION_FLOW, 1
    
    def _determine_switch_strategy(self, reason: SwitchReason, urgency: int) -> SwitchStrategy:
        """Determine how to execute the switch"""
        
        if reason == SwitchReason.CONTEXT_ESCALATION:
            return SwitchStrategy.IMMEDIATE
        elif reason in [SwitchReason.USER_DISSATISFACTION, SwitchReason.PERFORMANCE_DEGRADATION]:
            return SwitchStrategy.CONTEXT_PRESERVED
        elif urgency >= 4:
            return SwitchStrategy.IMMEDIATE
        else:
            return SwitchStrategy.SEAMLESS
    
    def _calculate_switch_confidence(self, context: ConversationContext, decision: SwitchingDecision) -> float:
        """Calculate confidence in the switching decision"""
        
        confidence = 0.5  # Base confidence
        
        # Higher confidence for significant performance gaps
        if decision.performance_delta > 0.3:
            confidence += 0.3
        elif decision.performance_delta > 0.1:
            confidence += 0.1
        
        # Higher confidence with more data
        if len(context.recent_user_ratings) >= 3:
            confidence += 0.2
        
        # Higher confidence for urgent situations
        if decision.urgency_level >= 4:
            confidence += 0.2
        
        # Lower confidence for frequent switches
        if self.session_switch_counts[context.session_id] >= 2:
            confidence -= 0.2
        
        return max(0.0, min(1.0, confidence))
    
    def _calculate_switch_cost(self, context: ConversationContext, decision: SwitchingDecision) -> float:
        """Calculate the cost of switching models"""
        
        cost = 0.0
        
        # Context loss cost
        if context.current_context_length > 1000:
            cost += 0.1  # Some context may be lost
        
        # Switching overhead
        cost += 0.05  # Base switching cost
        
        # User experience cost (if notification required)
        if decision.switch_strategy == SwitchStrategy.TRANSPARENT:
            cost += 0.1
        
        # Frequent switching penalty
        cost += self.session_switch_counts[context.session_id] * 0.05
        
        return cost
    
    def _should_execute_switch(self, decision: SwitchingDecision) -> bool:
        """Final decision on whether to execute the switch"""
        
        # Don't switch if confidence is too low
        if decision.confidence_score < 0.3:
            return False
        
        # Always switch for critical situations
        if decision.urgency_level >= 5:
            return True
        
        # Switch if net benefit is positive
        net_benefit = decision.estimated_improvement - decision.switch_cost
        
        if net_benefit > 0.1:  # Minimum threshold
            return True
        
        # Switch for high urgency even with marginal benefit
        if decision.urgency_level >= 4 and net_benefit > 0:
            return True
        
        return False
    
    def _generate_notification_message(self, decision: SwitchingDecision) -> Optional[str]:
        """Generate user notification message for model switch"""
        
        if not self.config.notify_user_of_switches:
            return None
        
        if self.config.transparency_level == "none":
            return None
        elif self.config.transparency_level == "minimal":
            return "I'm optimizing my responses to better help you."
        elif self.config.transparency_level == "detailed":
            reason_messages = {
                SwitchReason.CONTEXT_ESCALATION: "I'm switching to a more specialized model to better support you through this difficult time.",
                SwitchReason.USER_DISSATISFACTION: "I'm trying a different approach to provide more helpful responses.",
                SwitchReason.SPEED_OPTIMIZATION: "I'm switching to a faster model to reduce response time.",
                SwitchReason.COST_OPTIMIZATION: "I'm optimizing to provide efficient responses.",
                SwitchReason.CAPABILITY_MISMATCH: "I'm switching to a model better suited for your needs."
            }
            return reason_messages.get(decision.switch_reason, "I'm optimizing my responses for you.")
        
        return None
    
    def _analyze_message_context(self, context: ConversationContext, message: str):
        """Analyze message content for context clues"""
        
        message_lower = message.lower()
        
        # Crisis indicators
        crisis_keywords = ["suicide", "kill myself", "end it all", "hurt myself", "emergency"]
        crisis_score = sum(1 for keyword in crisis_keywords if keyword in message_lower)
        context.crisis_level = min(1.0, crisis_score * 0.3 + context.crisis_level * 0.7)
        
        # Emotional intensity indicators
        high_emotion_keywords = ["extremely", "terrible", "awful", "devastating", "overwhelming"]
        emotion_score = sum(1 for keyword in high_emotion_keywords if keyword in message_lower)
        context.emotional_intensity = min(1.0, emotion_score * 0.2 + context.emotional_intensity * 0.8)
        
        # Topic complexity indicators
        complex_keywords = ["explain", "understand", "complex", "detailed", "analysis"]
        complexity_score = sum(1 for keyword in complex_keywords if keyword in message_lower)
        context.topic_complexity = min(1.0, complexity_score * 0.1 + context.topic_complexity * 0.9)
    
    def get_session_summary(self, session_id: str) -> Dict[str, Any]:
        """Get comprehensive session summary including switching history"""
        
        context = self.active_contexts.get(session_id)
        if not context:
            return {'error': 'Session not found'}
        
        session_switches = [s for s in self.switch_history if s.session_id == session_id]
        
        return {
            'session_id': session_id,
            'current_model': context.current_model,
            'message_count': context.message_count,
            'conversation_duration_minutes': context.conversation_duration_minutes,
            'switches_executed': len(session_switches),
            'dominant_prompt_type': context.dominant_prompt_type.value,
            'current_context': {
                'crisis_level': context.crisis_level,
                'emotional_intensity': context.emotional_intensity,
                'user_engagement': context.user_engagement_score,
                'performance_trend': context.model_performance_trend
            },
            'switch_history': [s.to_dict() for s in session_switches],
            'performance_metrics': {
                'avg_response_time': statistics.mean(context.recent_response_times) if context.recent_response_times else None,
                'avg_user_rating': statistics.mean(context.recent_user_ratings) if context.recent_user_ratings else None,
                'rating_samples': len(context.recent_user_ratings)
            }
        }
    
    def get_switching_analytics(self) -> Dict[str, Any]:
        """Get analytics on model switching patterns"""
        
        if not self.switch_history:
            return {'message': 'No switching history available'}
        
        # Switch reason distribution
        reason_counts = defaultdict(int)
        for switch in self.switch_history:
            reason_counts[switch.switch_reason.value] += 1
        
        # Model transition patterns
        transitions = defaultdict(int)
        for switch in self.switch_history:
            transition_key = f"{switch.current_model} -> {switch.recommended_model}"
            transitions[transition_key] += 1
        
        # Success metrics
        successful_switches = [s for s in self.switch_history if s.should_switch]
        avg_improvement = statistics.mean([s.estimated_improvement for s in successful_switches]) if successful_switches else 0
        
        # Temporal patterns
        recent_switches = [s for s in self.switch_history 
                          if datetime.now() - s.timestamp <= timedelta(days=7)]
        
        return {
            'total_switches': len(self.switch_history),
            'successful_switches': len(successful_switches),
            'avg_estimated_improvement': avg_improvement,
            'switch_reasons': dict(reason_counts),
            'common_transitions': dict(sorted(transitions.items(), key=lambda x: x[1], reverse=True)[:10]),
            'recent_activity': {
                'switches_last_7_days': len(recent_switches),
                'avg_switches_per_day': len(recent_switches) / 7
            },
            'active_sessions': len(self.active_contexts)
        }
    
    def cleanup_inactive_sessions(self, max_age_hours: int = 24):
        """Clean up old inactive sessions"""
        
        cutoff_time = datetime.now() - timedelta(hours=max_age_hours)
        
        inactive_sessions = []
        for session_id, context in self.active_contexts.items():
            # Check if session has been inactive (simple heuristic)
            if context.message_count == 0:  # No messages yet
                inactive_sessions.append(session_id)
        
        for session_id in inactive_sessions:
            del self.active_contexts[session_id]
            if session_id in self.session_switch_counts:
                del self.session_switch_counts[session_id]
        
        logger.info(f"Cleaned up {len(inactive_sessions)} inactive sessions")
        return len(inactive_sessions)