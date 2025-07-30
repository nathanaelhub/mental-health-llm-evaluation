"""
Progressive Enhancement System for Model Selection

Provides fast initial responses using cached/heuristic selections while running
full evaluation in the background to improve future performance.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque

from ..chat.dynamic_model_selector import PromptType, ModelSelection, DynamicModelSelector
from .smart_cache import SmartModelCache, CachedSelection
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class EnhancementStrategy(Enum):
    """Progressive enhancement strategies"""
    CACHE_FIRST = "cache_first"          # Use cache, then background evaluation
    HEURISTIC_FIRST = "heuristic_first"  # Fast heuristics, then full evaluation
    HYBRID = "hybrid"                    # Combination of cache and heuristics


@dataclass
class FastSelection:
    """Fast model selection result for immediate response"""
    selected_model: str
    confidence_score: float
    prompt_classification: PromptType
    reasoning: str
    selection_method: str  # "cache", "heuristic", "emergency"
    latency_ms: float
    background_evaluation_id: Optional[str] = None


@dataclass
class BackgroundEvaluation:
    """Background evaluation task"""
    evaluation_id: str
    prompt: str
    prompt_type: PromptType
    fast_selection: FastSelection
    start_time: datetime
    completion_callback: Optional[Callable[[ModelSelection], Awaitable[None]]] = None
    timeout_seconds: int = 30
    priority: int = 1  # 1=high, 2=medium, 3=low


@dataclass
class EnhancementMetrics:
    """Metrics for progressive enhancement performance"""
    total_requests: int = 0
    cache_hits: int = 0
    heuristic_hits: int = 0
    emergency_selections: int = 0
    
    # Timing metrics
    avg_fast_response_ms: float = 0.0
    avg_background_completion_ms: float = 0.0
    
    # Accuracy metrics
    fast_vs_full_agreement_rate: float = 0.0
    model_switch_rate: float = 0.0  # How often background evaluation changes selection
    
    # Queue metrics
    background_queue_size: int = 0
    max_queue_size: int = 0
    avg_queue_wait_time_ms: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'heuristic_hits': self.heuristic_hits,
            'emergency_selections': self.emergency_selections,
            'avg_fast_response_ms': self.avg_fast_response_ms,
            'avg_background_completion_ms': self.avg_background_completion_ms,
            'fast_vs_full_agreement_rate': self.fast_vs_full_agreement_rate,
            'model_switch_rate': self.model_switch_rate,
            'background_queue_size': self.background_queue_size,
            'max_queue_size': self.max_queue_size,
            'avg_queue_wait_time_ms': self.avg_queue_wait_time_ms
        }


class HeuristicSelector:
    """Fast heuristic-based model selection for immediate responses"""
    
    def __init__(self):
        # Simple rule-based selection for speed
        self.heuristic_rules = {
            PromptType.CRISIS: "claude-3-opus",      # Always use most capable for crisis
            PromptType.ANXIETY: "claude-3-sonnet",   # Good balance for anxiety
            PromptType.DEPRESSION: "claude-3-sonnet", # Good balance for depression
            PromptType.INFORMATION_SEEKING: "gpt-4-turbo", # Fast and accurate for info
            PromptType.GENERAL_WELLNESS: "gpt-3.5-turbo",  # Fast for general wellness
        }
        
        # Confidence scores for heuristic selections
        self.heuristic_confidence = {
            PromptType.CRISIS: 0.95,           # High confidence for safety-critical
            PromptType.ANXIETY: 0.80,          # Good confidence
            PromptType.DEPRESSION: 0.80,       # Good confidence
            PromptType.INFORMATION_SEEKING: 0.75, # Moderate confidence
            PromptType.GENERAL_WELLNESS: 0.70,  # Lower confidence, more flexible
        }
    
    def select_model(self, prompt: str, prompt_type: PromptType) -> FastSelection:
        """Select model using fast heuristics"""
        start_time = time.time()
        
        # Apply heuristic rules
        selected_model = self.heuristic_rules.get(prompt_type, "gpt-3.5-turbo")
        confidence = self.heuristic_confidence.get(prompt_type, 0.7)
        
        # Simple keyword-based adjustments
        prompt_lower = prompt.lower()
        
        # Crisis keywords override everything
        crisis_keywords = ["suicide", "kill myself", "end it all", "hurt myself", "emergency"]
        if any(keyword in prompt_lower for keyword in crisis_keywords):
            selected_model = "claude-3-opus"
            confidence = 0.98
            prompt_type = PromptType.CRISIS
        
        # Complex question indicators
        elif any(word in prompt_lower for word in ["explain", "analyze", "complex", "detailed"]):
            if prompt_type in [PromptType.INFORMATION_SEEKING, PromptType.GENERAL_WELLNESS]:
                selected_model = "gpt-4-turbo"
                confidence = min(confidence + 0.1, 0.95)
        
        # Simple question indicators
        elif any(word in prompt_lower for word in ["yes", "no", "quick", "simple", "hello"]):
            if prompt_type == PromptType.GENERAL_WELLNESS:
                selected_model = "gpt-3.5-turbo"
                confidence = max(confidence - 0.05, 0.6)
        
        latency_ms = (time.time() - start_time) * 1000
        
        reasoning = f"Heuristic selection for {prompt_type.value}: {selected_model} (confidence: {confidence:.2f})"
        
        return FastSelection(
            selected_model=selected_model,
            confidence_score=confidence,
            prompt_classification=prompt_type,
            reasoning=reasoning,
            selection_method="heuristic",
            latency_ms=latency_ms
        )


class ProgressiveEnhancer:
    """
    Progressive Enhancement System that provides fast responses while improving accuracy
    
    Features:
    - Immediate fast response using cache or heuristics
    - Background full evaluation for accuracy
    - Learning from mismatches to improve fast selection
    - Queue management for background tasks
    - Performance monitoring and optimization
    """
    
    def __init__(self, 
                 cache: SmartModelCache,
                 model_selector: DynamicModelSelector,
                 performance_monitor: PerformanceMonitor,
                 strategy: EnhancementStrategy = EnhancementStrategy.HYBRID,
                 max_background_workers: int = 3,
                 background_timeout_seconds: int = 30):
        
        self.cache = cache
        self.model_selector = model_selector
        self.performance_monitor = performance_monitor
        self.strategy = strategy
        self.max_background_workers = max_background_workers
        self.background_timeout = background_timeout_seconds
        
        # Fast selection components
        self.heuristic_selector = HeuristicSelector()
        
        # Background evaluation queue and workers
        self.background_queue: asyncio.Queue = asyncio.Queue()
        self.background_workers: List[asyncio.Task] = []
        self.active_evaluations: Dict[str, BackgroundEvaluation] = {}
        
        # Metrics and learning
        self.metrics = EnhancementMetrics()
        self.evaluation_history = deque(maxlen=1000)  # Store recent evaluations for learning
        
        # Learning system for improving fast selections
        self.mismatch_patterns = defaultdict(list)  # Track when fast != full selection
        
        # Emergency fallback model (when everything fails)
        self.emergency_model = "gpt-3.5-turbo"
        
        logger.info(f"ProgressiveEnhancer initialized with strategy: {strategy.value}")
    
    async def start_background_workers(self):
        """Start background evaluation workers"""
        for i in range(self.max_background_workers):
            worker = asyncio.create_task(self._background_worker(f"worker-{i}"))
            self.background_workers.append(worker)
        
        logger.info(f"Started {self.max_background_workers} background evaluation workers")
    
    async def stop_background_workers(self):
        """Stop background evaluation workers"""
        # Cancel all workers
        for worker in self.background_workers:
            worker.cancel()
        
        # Wait for workers to complete
        await asyncio.gather(*self.background_workers, return_exceptions=True)
        self.background_workers.clear()
        
        logger.info("Stopped all background evaluation workers")
    
    async def select_model_enhanced(self, 
                                  prompt: str, 
                                  prompt_type: Optional[PromptType] = None,
                                  completion_callback: Optional[Callable[[ModelSelection], Awaitable[None]]] = None) -> FastSelection:
        """
        Get fast model selection with background enhancement
        
        Args:
            prompt: User prompt
            prompt_type: Optional prompt classification hint
            completion_callback: Optional callback for when background evaluation completes
            
        Returns:
            FastSelection with immediate response
        """
        request_start = time.time()
        self.metrics.total_requests += 1
        
        try:
            # Step 1: Try to get fast selection
            fast_selection = await self._get_fast_selection(prompt, prompt_type)
            
            # Step 2: Queue background evaluation if needed
            if fast_selection.selection_method != "cache" or fast_selection.confidence_score < 0.9:
                evaluation_id = await self._queue_background_evaluation(
                    prompt, 
                    fast_selection.prompt_classification, 
                    fast_selection,
                    completion_callback
                )
                fast_selection.background_evaluation_id = evaluation_id
            
            # Update metrics
            request_latency = (time.time() - request_start) * 1000
            self.metrics.avg_fast_response_ms = (
                (self.metrics.avg_fast_response_ms * (self.metrics.total_requests - 1) + request_latency) /
                self.metrics.total_requests
            )
            
            logger.debug(f"Fast selection: {fast_selection.selected_model} ({fast_selection.selection_method}) in {request_latency:.2f}ms")
            return fast_selection
            
        except Exception as e:
            logger.error(f"Error in enhanced model selection: {e}")
            # Emergency fallback
            return self._emergency_selection(prompt, prompt_type)
    
    async def _get_fast_selection(self, prompt: str, prompt_type: Optional[PromptType]) -> FastSelection:
        """Get fast selection using cache or heuristics"""
        
        # Strategy: Cache First
        if self.strategy in [EnhancementStrategy.CACHE_FIRST, EnhancementStrategy.HYBRID]:
            cached_selection = await self.cache.get_cached_selection(prompt, prompt_type)
            if cached_selection:
                self.metrics.cache_hits += 1
                return FastSelection(
                    selected_model=cached_selection.selected_model,
                    confidence_score=cached_selection.confidence_score,
                    prompt_classification=cached_selection.prompt_classification,
                    reasoning=f"Cached selection: {cached_selection.reasoning}",
                    selection_method="cache",
                    latency_ms=5.0  # Assume very fast cache lookup
                )
        
        # Strategy: Heuristic selection
        if self.strategy in [EnhancementStrategy.HEURISTIC_FIRST, EnhancementStrategy.HYBRID]:
            # If no prompt type provided, use simple classification
            if not prompt_type:
                prompt_type = self._classify_prompt_fast(prompt)
            
            heuristic_selection = self.heuristic_selector.select_model(prompt, prompt_type)
            self.metrics.heuristic_hits += 1
            return heuristic_selection
        
        # Emergency fallback
        return self._emergency_selection(prompt, prompt_type)
    
    def _classify_prompt_fast(self, prompt: str) -> PromptType:
        """Fast prompt classification using simple rules"""
        prompt_lower = prompt.lower()
        
        # Crisis detection (highest priority)
        crisis_keywords = ["suicide", "kill myself", "end it all", "hurt myself", "emergency", "crisis"]
        if any(keyword in prompt_lower for keyword in crisis_keywords):
            return PromptType.CRISIS
        
        # Anxiety indicators
        anxiety_keywords = ["anxious", "anxiety", "worried", "panic", "stress", "nervous"]
        if any(keyword in prompt_lower for keyword in anxiety_keywords):
            return PromptType.ANXIETY
        
        # Depression indicators
        depression_keywords = ["depressed", "depression", "sad", "hopeless", "empty", "worthless"]
        if any(keyword in prompt_lower for keyword in depression_keywords):
            return PromptType.DEPRESSION
        
        # Information seeking
        info_keywords = ["how", "what", "why", "when", "where", "explain", "tell me about"]
        if any(keyword in prompt_lower for keyword in info_keywords):
            return PromptType.INFORMATION_SEEKING
        
        # Default to general wellness
        return PromptType.GENERAL_WELLNESS
    
    def _emergency_selection(self, prompt: str, prompt_type: Optional[PromptType]) -> FastSelection:
        """Emergency fallback selection"""
        self.metrics.emergency_selections += 1
        
        # Always use safe model for unknown situations
        return FastSelection(
            selected_model=self.emergency_model,
            confidence_score=0.5,
            prompt_classification=prompt_type or PromptType.GENERAL_WELLNESS,
            reasoning="Emergency fallback selection due to system unavailability",
            selection_method="emergency",
            latency_ms=1.0
        )
    
    async def _queue_background_evaluation(self, 
                                         prompt: str, 
                                         prompt_type: PromptType, 
                                         fast_selection: FastSelection,
                                         completion_callback: Optional[Callable[[ModelSelection], Awaitable[None]]]) -> str:
        """Queue background evaluation task"""
        
        evaluation_id = f"eval_{int(time.time() * 1000)}_{hash(prompt) % 10000}"
        
        evaluation = BackgroundEvaluation(
            evaluation_id=evaluation_id,
            prompt=prompt,
            prompt_type=prompt_type,
            fast_selection=fast_selection,
            start_time=datetime.now(),
            completion_callback=completion_callback,
            timeout_seconds=self.background_timeout,
            priority=self._calculate_priority(prompt_type, fast_selection)
        )
        
        # Add to queue
        await self.background_queue.put(evaluation)
        self.active_evaluations[evaluation_id] = evaluation
        
        # Update queue metrics
        self.metrics.background_queue_size = self.background_queue.qsize()
        self.metrics.max_queue_size = max(self.metrics.max_queue_size, self.metrics.background_queue_size)
        
        logger.debug(f"Queued background evaluation: {evaluation_id}")
        return evaluation_id
    
    def _calculate_priority(self, prompt_type: PromptType, fast_selection: FastSelection) -> int:
        """Calculate priority for background evaluation (1=high, 3=low)"""
        # Crisis always gets highest priority
        if prompt_type == PromptType.CRISIS:
            return 1
        
        # Low confidence selections get higher priority
        if fast_selection.confidence_score < 0.7:
            return 1
        elif fast_selection.confidence_score < 0.8:
            return 2
        else:
            return 3
    
    async def _background_worker(self, worker_id: str):
        """Background worker for full model evaluation"""
        logger.info(f"Background worker {worker_id} started")
        
        while True:
            try:
                # Get next evaluation task
                evaluation = await self.background_queue.get()
                
                if evaluation.evaluation_id not in self.active_evaluations:
                    continue  # Task was cancelled
                
                logger.debug(f"Worker {worker_id} processing {evaluation.evaluation_id}")
                
                # Run full model selection
                start_time = time.time()
                
                try:
                    # Use timeout to prevent hanging
                    full_selection = await asyncio.wait_for(
                        self.model_selector.select_model(evaluation.prompt, evaluation.prompt_type),
                        timeout=evaluation.timeout_seconds
                    )
                    
                    completion_time = (time.time() - start_time) * 1000
                    
                    # Update metrics
                    self.metrics.avg_background_completion_ms = (
                        (self.metrics.avg_background_completion_ms * len(self.evaluation_history) + completion_time) /
                        (len(self.evaluation_history) + 1)
                    )
                    
                    # Compare with fast selection
                    await self._analyze_selection_difference(evaluation, full_selection)
                    
                    # Store evaluation in cache for future use
                    await self.cache.store_selection(
                        evaluation.prompt, 
                        full_selection, 
                        success=True,
                        response_time_ms=completion_time
                    )
                    
                    # Call completion callback if provided
                    if evaluation.completion_callback:
                        try:
                            await evaluation.completion_callback(full_selection)
                        except Exception as e:
                            logger.error(f"Error in completion callback: {e}")
                    
                    logger.debug(f"Completed background evaluation {evaluation.evaluation_id} in {completion_time:.2f}ms")
                    
                except asyncio.TimeoutError:
                    logger.warning(f"Background evaluation {evaluation.evaluation_id} timed out after {evaluation.timeout_seconds}s")
                
                except Exception as e:
                    logger.error(f"Error in background evaluation {evaluation.evaluation_id}: {e}")
                
                finally:
                    # Clean up
                    if evaluation.evaluation_id in self.active_evaluations:
                        del self.active_evaluations[evaluation.evaluation_id]
                    
                    self.background_queue.task_done()
                    self.metrics.background_queue_size = self.background_queue.qsize()
                
            except asyncio.CancelledError:
                logger.info(f"Background worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Unexpected error in background worker {worker_id}: {e}")
                await asyncio.sleep(1)  # Brief pause before retrying
    
    async def _analyze_selection_difference(self, evaluation: BackgroundEvaluation, full_selection: ModelSelection):
        """Analyze difference between fast and full selection for learning"""
        
        fast_model = evaluation.fast_selection.selected_model
        full_model = full_selection.selected_model
        
        # Record evaluation history
        evaluation_record = {
            'timestamp': datetime.now().isoformat(),
            'prompt_type': evaluation.prompt_type.value,
            'fast_method': evaluation.fast_selection.selection_method,
            'fast_model': fast_model,
            'fast_confidence': evaluation.fast_selection.confidence_score,
            'full_model': full_model,
            'full_confidence': full_selection.confidence_score,
            'models_match': fast_model == full_model,
            'confidence_diff': abs(evaluation.fast_selection.confidence_score - full_selection.confidence_score)
        }
        
        self.evaluation_history.append(evaluation_record)
        
        # Update accuracy metrics
        total_evaluations = len(self.evaluation_history)
        matching_evaluations = sum(1 for e in self.evaluation_history if e['models_match'])
        self.metrics.fast_vs_full_agreement_rate = (matching_evaluations / total_evaluations) * 100
        
        # Track model switches
        if fast_model != full_model:
            self.metrics.model_switch_rate = ((total_evaluations - matching_evaluations) / total_evaluations) * 100
            
            # Record mismatch pattern for learning
            mismatch_key = f"{evaluation.prompt_type.value}_{evaluation.fast_selection.selection_method}"
            self.mismatch_patterns[mismatch_key].append({
                'fast_model': fast_model,
                'full_model': full_model,
                'timestamp': datetime.now(),
                'prompt_length': len(evaluation.prompt)
            })
            
            logger.debug(f"Model mismatch detected: {fast_model} -> {full_model} for {evaluation.prompt_type.value}")
    
    def get_metrics(self) -> EnhancementMetrics:
        """Get current enhancement metrics"""
        return self.metrics
    
    def get_learning_insights(self) -> Dict[str, Any]:
        """Get insights from learning patterns"""
        insights = {
            'mismatch_patterns': {},
            'recommendations': [],
            'accuracy_trends': []
        }
        
        # Analyze mismatch patterns
        for pattern_key, mismatches in self.mismatch_patterns.items():
            if len(mismatches) >= 3:  # Need minimum data
                common_switches = defaultdict(int)
                for mismatch in mismatches:
                    switch_key = f"{mismatch['fast_model']} -> {mismatch['full_model']}"
                    common_switches[switch_key] += 1
                
                most_common_switch = max(common_switches.items(), key=lambda x: x[1])
                
                insights['mismatch_patterns'][pattern_key] = {
                    'total_mismatches': len(mismatches),
                    'most_common_switch': most_common_switch[0],
                    'switch_frequency': most_common_switch[1]
                }
                
                # Generate recommendation
                if most_common_switch[1] >= 3:
                    insights['recommendations'].append({
                        'type': 'heuristic_adjustment',
                        'pattern': pattern_key,
                        'suggestion': f"Consider updating heuristic rule: {most_common_switch[0]}",
                        'confidence': min(most_common_switch[1] / len(mismatches), 1.0)
                    })
        
        return insights
    
    async def optimize_heuristics(self):
        """Optimize heuristic rules based on learning patterns"""
        logger.info("Optimizing heuristics based on learning patterns...")
        
        insights = self.get_learning_insights()
        adjustments_made = 0
        
        for pattern, data in insights['mismatch_patterns'].items():
            if data['switch_frequency'] >= 5 and data['switch_frequency'] / data['total_mismatches'] > 0.7:
                # Strong pattern detected - adjust heuristic
                prompt_type_str, method = pattern.split('_', 1)
                
                if method == 'heuristic':
                    try:
                        prompt_type = PromptType(prompt_type_str)
                        
                        # Extract target model from most common switch
                        switch_parts = data['most_common_switch'].split(' -> ')
                        if len(switch_parts) == 2:
                            target_model = switch_parts[1]
                            
                            # Update heuristic rule
                            old_model = self.heuristic_selector.heuristic_rules.get(prompt_type)
                            self.heuristic_selector.heuristic_rules[prompt_type] = target_model
                            
                            logger.info(f"Updated heuristic for {prompt_type.value}: {old_model} -> {target_model}")
                            adjustments_made += 1
                            
                    except ValueError:
                        logger.warning(f"Invalid prompt type in pattern: {prompt_type_str}")
        
        logger.info(f"Made {adjustments_made} heuristic adjustments based on learning patterns")
        return adjustments_made
    
    async def get_queue_status(self) -> Dict[str, Any]:
        """Get current background queue status"""
        return {
            'queue_size': self.background_queue.qsize(),
            'active_evaluations': len(self.active_evaluations),
            'active_workers': len([w for w in self.background_workers if not w.done()]),
            'total_workers': len(self.background_workers),
            'oldest_queued_task': min(
                (eval.start_time for eval in self.active_evaluations.values()),
                default=None
            ).isoformat() if self.active_evaluations else None
        }
    
    async def cleanup(self):
        """Clean up resources"""
        logger.info("Cleaning up ProgressiveEnhancer...")
        
        # Stop background workers
        await self.stop_background_workers()
        
        # Clear queues and active evaluations
        while not self.background_queue.empty():
            try:
                self.background_queue.get_nowait()
                self.background_queue.task_done()
            except asyncio.QueueEmpty:
                break
        
        self.active_evaluations.clear()
        
        logger.info("ProgressiveEnhancer cleanup complete")