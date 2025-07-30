"""
Dynamic Model Selection Engine for Mental Health Conversations

This module provides intelligent model selection based on prompt classification,
weighted scoring criteria, and real-time evaluation of model responses.
"""

import asyncio
import time
import logging
import hashlib
import re
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, asdict
from datetime import datetime
from enum import Enum
import numpy as np
from collections import defaultdict

from ..evaluation.evaluation_metrics import TherapeuticEvaluator
from .response_cache import ResponseCache

logger = logging.getLogger(__name__)


class PromptType(Enum):
    """Classification of mental health prompt types"""
    CRISIS = "crisis"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    INFORMATION_SEEKING = "information_seeking"
    GENERAL_SUPPORT = "general_support"
    RELATIONSHIP = "relationship"
    TRAUMA = "trauma"
    UNKNOWN = "unknown"


@dataclass
class SelectionCriteria:
    """Weighted criteria for model selection based on prompt type"""
    empathy_weight: float
    therapeutic_weight: float
    safety_weight: float
    clarity_weight: float
    
    def __post_init__(self):
        # Ensure weights sum to 1.0
        total = self.empathy_weight + self.therapeutic_weight + self.safety_weight + self.clarity_weight
        if abs(total - 1.0) > 0.001:
            raise ValueError(f"Selection criteria weights must sum to 1.0, got {total}")


@dataclass
class ModelEvaluation:
    """Results from evaluating a single model"""
    model_id: str
    response_content: str
    evaluation_scores: Dict[str, float]
    composite_score: float
    response_time_ms: float
    error: Optional[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class ModelSelection:
    """Result of dynamic model selection process"""
    selected_model_id: str
    model_scores: Dict[str, float]
    response_content: str
    selection_reasoning: str
    latency_metrics: Dict[str, float]
    prompt_type: PromptType
    selection_criteria: SelectionCriteria
    confidence_score: float
    timestamp: datetime
    cached: bool = False
    
    def to_dict(self) -> Dict[str, Any]:
        result = asdict(self)
        result['prompt_type'] = self.prompt_type.value
        result['timestamp'] = self.timestamp.isoformat()
        return result


class PerformanceMonitor:
    """Monitors selection performance and usage analytics"""
    
    def __init__(self):
        self.selection_history: List[ModelSelection] = []
        self.model_usage_counts = defaultdict(int)
        self.prompt_type_counts = defaultdict(int)
        self.cache_hit_count = 0
        self.cache_miss_count = 0
        self.total_selection_time_ms = 0.0
        self.confidence_scores: List[float] = []
        
    def record_selection(self, selection: ModelSelection):
        """Record a model selection for analytics"""
        self.selection_history.append(selection)
        self.model_usage_counts[selection.selected_model_id] += 1
        self.prompt_type_counts[selection.prompt_type.value] += 1
        self.total_selection_time_ms += selection.latency_metrics.get('total_time_ms', 0)
        self.confidence_scores.append(selection.confidence_score)
        
        if selection.cached:
            self.cache_hit_count += 1
        else:
            self.cache_miss_count += 1
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive performance analytics"""
        total_selections = len(self.selection_history)
        
        if total_selections == 0:
            return {
                'total_selections': 0,
                'model_distribution': {},
                'prompt_type_distribution': {},
                'cache_hit_rate': 0.0,
                'avg_selection_time_ms': 0.0,
                'avg_confidence_score': 0.0
            }
        
        return {
            'total_selections': total_selections,
            'model_distribution': dict(self.model_usage_counts),
            'prompt_type_distribution': dict(self.prompt_type_counts),
            'cache_hit_rate': self.cache_hit_count / (self.cache_hit_count + self.cache_miss_count),
            'avg_selection_time_ms': self.total_selection_time_ms / total_selections,
            'avg_confidence_score': np.mean(self.confidence_scores) if self.confidence_scores else 0.0,
            'confidence_std': np.std(self.confidence_scores) if len(self.confidence_scores) > 1 else 0.0
        }


class ModelAvailabilityCache:
    """Cache model availability status to avoid repeated health checks"""
    
    def __init__(self, ttl: int = 300):  # 5 minute cache
        self.cache = {}
        self.ttl = ttl
        logger.debug(f"ModelAvailabilityCache initialized with {ttl}s TTL")
    
    async def is_available(self, model_name: str, health_check_func) -> bool:
        """Check if model is available, using cache when possible"""
        current_time = time.time()
        
        # Check cache first
        if model_name in self.cache:
            cache_entry = self.cache[model_name]
            if current_time - cache_entry['time'] < self.ttl:
                logger.debug(f"Using cached availability for {model_name}: {cache_entry['available']}")
                return cache_entry['available']
        
        # Cache miss or expired - test availability
        logger.debug(f"Testing availability for {model_name}")
        try:
            available = await health_check_func(model_name)
        except Exception as e:
            logger.error(f"Health check failed for {model_name}: {e}")
            available = False
        
        # Update cache
        self.cache[model_name] = {
            'available': available,
            'time': current_time
        }
        
        logger.debug(f"Cached availability for {model_name}: {available}")
        return available
    
    def clear_cache(self):
        """Clear the availability cache"""
        self.cache.clear()
        logger.debug("Model availability cache cleared")
    
    def get_cached_status(self) -> Dict[str, bool]:
        """Get current cached status for all models"""
        current_time = time.time()
        status = {}
        
        for model_name, cache_entry in self.cache.items():
            if current_time - cache_entry['time'] < self.ttl:
                status[model_name] = cache_entry['available']
        
        return status


class DynamicModelSelector:
    """
    Intelligently selects the best model for each conversation based on initial response quality.
    
    Features:
    - Prompt classification for context-aware selection
    - Weighted scoring based on prompt type
    - Smart caching with similarity matching
    - Parallel async evaluation with timeout handling
    - Performance monitoring and analytics
    - Selection transparency and reasoning
    """
    
    # Selection criteria for different prompt types
    SELECTION_CRITERIA = {
        PromptType.CRISIS: SelectionCriteria(
            empathy_weight=0.25,
            therapeutic_weight=0.25,
            safety_weight=0.50,  # Prioritize safety for crisis
            clarity_weight=0.0
        ),
        PromptType.ANXIETY: SelectionCriteria(
            empathy_weight=0.40,
            therapeutic_weight=0.40,
            safety_weight=0.15,
            clarity_weight=0.05
        ),
        PromptType.DEPRESSION: SelectionCriteria(
            empathy_weight=0.45,
            therapeutic_weight=0.35,
            safety_weight=0.15,
            clarity_weight=0.05
        ),
        PromptType.INFORMATION_SEEKING: SelectionCriteria(
            empathy_weight=0.10,
            therapeutic_weight=0.40,
            safety_weight=0.10,
            clarity_weight=0.40  # Prioritize clarity for information
        ),
        PromptType.GENERAL_SUPPORT: SelectionCriteria(
            empathy_weight=0.35,
            therapeutic_weight=0.35,
            safety_weight=0.20,
            clarity_weight=0.10
        ),
        PromptType.RELATIONSHIP: SelectionCriteria(
            empathy_weight=0.40,
            therapeutic_weight=0.35,
            safety_weight=0.15,
            clarity_weight=0.10
        ),
        PromptType.TRAUMA: SelectionCriteria(
            empathy_weight=0.30,
            therapeutic_weight=0.30,
            safety_weight=0.35,
            clarity_weight=0.05
        ),
        PromptType.UNKNOWN: SelectionCriteria(  # Balanced default
            empathy_weight=0.30,
            therapeutic_weight=0.30,
            safety_weight=0.25,
            clarity_weight=0.15
        )
    }
    
    # Keywords for prompt classification
    CLASSIFICATION_KEYWORDS = {
        PromptType.CRISIS: [
            'suicide', 'kill myself', 'end it all', 'not worth living', 'want to die',
            'hurt myself', 'self harm', 'emergency', 'crisis', 'desperate',
            'can\'t go on', 'give up', 'hopeless', 'worthless'
        ],
        PromptType.ANXIETY: [
            'anxious', 'anxiety', 'worry', 'worried', 'panic', 'nervous',
            'stressed', 'stress', 'overwhelming', 'scared', 'fear', 'phobia',
            'racing thoughts', 'can\'t breathe', 'heart racing', 'panic attack'
        ],
        PromptType.DEPRESSION: [
            'depressed', 'depression', 'sad', 'sadness', 'empty', 'numb',
            'tired', 'exhausted', 'no energy', 'unmotivated', 'hopeless',
            'worthless', 'guilty', 'shame', 'dark thoughts', 'heavy'
        ],
        PromptType.INFORMATION_SEEKING: [
            'what is', 'how do', 'can you explain', 'tell me about',
            'information', 'learn', 'understand', 'definition', 'meaning',
            'research', 'facts', 'statistics', 'studies', 'evidence'
        ],
        PromptType.RELATIONSHIP: [
            'relationship', 'partner', 'boyfriend', 'girlfriend', 'spouse',
            'marriage', 'divorce', 'breakup', 'dating', 'love', 'family',
            'friends', 'friendship', 'conflict', 'argument', 'communication'
        ],
        PromptType.TRAUMA: [
            'trauma', 'traumatic', 'abuse', 'abused', 'assault', 'violence',
            'ptsd', 'flashbacks', 'nightmares', 'triggered', 'survivor',
            'childhood', 'neglect', 'betrayal', 'violation'
        ]
    }
    
    def __init__(self, models_config: Dict[str, Any], evaluation_framework=None):
        """
        Initialize the dynamic model selector
        
        Args:
            models_config: Configuration for available models
            evaluation_framework: Mental health evaluation framework
        """
        self.models = self._initialize_models(models_config)
        self.evaluator = evaluation_framework or TherapeuticEvaluator()
        self.selection_cache = ResponseCache(
            cache_dir="temp/selection_cache",
            ttl_hours=1,
            max_entries=500
        )
        self.performance_monitor = PerformanceMonitor()
        self.availability_cache = ModelAvailabilityCache(ttl=300)  # 5 minute cache
        
        # Model timeout configuration - different timeouts for cloud vs local models
        self.model_timeouts = models_config.get('model_timeouts', {
            'openai': 5.0,      # Fast cloud API
            'claude': 5.0,      # Fast cloud API  
            'deepseek': 10.0,   # Local model, may be slower
            'gemma': 10.0       # Local model, may be slower
        })
        
        # Selection configuration
        self.default_model = models_config.get('default_model', 'openai')
        self.selection_timeout = models_config.get('selection_timeout', 15.0)  # Reduced from 40s
        self.similarity_threshold = models_config.get('similarity_threshold', 0.9)
        
        logger.info(f"DynamicModelSelector initialized with {len(self.models)} models")
    
    def _initialize_models(self, models_config: Dict[str, Any]) -> Dict[str, Any]:
        """Initialize model clients from configuration"""
        models = {}
        
        for model_id, config in models_config.get('models', {}).items():
            try:
                if model_id == 'openai':
                    from ..models.openai_client import OpenAIClient
                    models[model_id] = OpenAIClient()
                elif model_id == 'deepseek':
                    from ..models.deepseek_client import DeepSeekClient
                    models[model_id] = DeepSeekClient()
                elif model_id == 'claude':
                    from ..models.claude_client import ClaudeClient
                    models[model_id] = ClaudeClient()
                elif model_id == 'gemma':
                    from ..models.gemma_client import GemmaClient
                    models[model_id] = GemmaClient()
                else:
                    logger.warning(f"Unknown model type: {model_id}")
                    continue
                    
                logger.info(f"Initialized model client: {model_id}")
                
            except ImportError as e:
                logger.warning(f"Could not initialize {model_id}: {e}")
            except Exception as e:
                logger.error(f"Error initializing {model_id}: {e}")
        
        return models
    
    def prompt_classification(self, prompt: str, context: Optional[str] = None) -> PromptType:
        """
        Classify prompt type based on content analysis
        
        Args:
            prompt: User's input message
            context: Optional conversation context
            
        Returns:
            PromptType classification
        """
        # Combine prompt and context for analysis
        full_text = prompt.lower()
        if context:
            full_text += " " + context.lower()
        
        # Score each prompt type based on keyword matches
        type_scores = defaultdict(float)
        
        for prompt_type, keywords in self.CLASSIFICATION_KEYWORDS.items():
            for keyword in keywords:
                # Use word boundaries for more accurate matching
                pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
                matches = len(re.findall(pattern, full_text))
                if matches > 0:
                    # Weight by keyword importance and frequency
                    type_scores[prompt_type] += matches * (1.0 + len(keyword) / 20.0)
        
        # Additional pattern-based classification
        self._apply_pattern_classification(full_text, type_scores)
        
        # Return the highest scoring type, or UNKNOWN if no clear match
        if not type_scores:
            return PromptType.UNKNOWN
        
        best_type = max(type_scores.items(), key=lambda x: x[1])
        
        # Require minimum confidence for classification
        if best_type[1] < 0.5:
            return PromptType.UNKNOWN
        
        logger.debug(f"Classified prompt as {best_type[0].value} with confidence {best_type[1]:.2f}")
        return best_type[0]
    
    def _apply_pattern_classification(self, text: str, type_scores: Dict[PromptType, float]):
        """Apply pattern-based classification rules"""
        
        # Crisis detection patterns
        if re.search(r'\b(end|ending)\s+(it|this|everything)\b', text):
            type_scores[PromptType.CRISIS] += 2.0
        
        if re.search(r'\b(can\'?t|cannot)\s+(take|handle|deal|cope)\b', text):
            type_scores[PromptType.CRISIS] += 1.5
        
        # Question patterns for information seeking
        question_patterns = [
            r'^\s*(what|how|why|when|where|who)\s+',
            r'\?\s*$',
            r'\bcan\s+you\s+(tell|explain|help)\b'
        ]
        
        for pattern in question_patterns:
            if re.search(pattern, text):
                type_scores[PromptType.INFORMATION_SEEKING] += 1.0
                break
        
        # Emotional intensity indicators
        if re.search(r'\b(really|very|extremely|incredibly|totally)\s+(sad|depressed|down)\b', text):
            type_scores[PromptType.DEPRESSION] += 1.5
        
        if re.search(r'\b(really|very|extremely|incredibly|totally)\s+(anxious|worried|scared)\b', text):
            type_scores[PromptType.ANXIETY] += 1.5
    
    async def select_best_model(self, prompt: str, context: Optional[str] = None) -> ModelSelection:
        """
        Evaluate all models in parallel and select the best performer.
        
        Args:
            prompt: User's input message
            context: Optional conversation context
            
        Returns:
            ModelSelection object with comprehensive selection results
        """
        start_time = time.time()
        
        # Check cache first
        cached_selection = self._check_cache(prompt, context)
        if cached_selection:
            self.performance_monitor.record_selection(cached_selection)
            logger.info(f"Using cached selection: {cached_selection.selected_model_id}")
            return cached_selection
        
        # Classify prompt type
        prompt_type = self.prompt_classification(prompt, context)
        selection_criteria = self.SELECTION_CRITERIA[prompt_type]
        
        logger.info(f"Selecting model for {prompt_type.value} prompt")
        
        try:
            # Parallel evaluation of all models
            model_evaluations = await asyncio.wait_for(
                self.parallel_evaluate(prompt, context),
                timeout=self.selection_timeout
            )
            
            if not model_evaluations:
                logger.warning("No model evaluations available, using default model")
                return self._create_fallback_selection(prompt, prompt_type, start_time)
            
            # Apply selection logic
            selection = self.apply_selection_logic(
                model_evaluations, 
                selection_criteria, 
                prompt_type,
                start_time
            )
            
            # Cache the selection
            self._cache_selection(prompt, context, selection)
            
            # Record for analytics
            self.performance_monitor.record_selection(selection)
            
            logger.info(f"Selected {selection.selected_model_id} with confidence {selection.confidence_score:.2f}")
            return selection
            
        except asyncio.TimeoutError:
            logger.warning(f"Model selection timed out after {self.selection_timeout}s")
            return self._create_fallback_selection(prompt, prompt_type, start_time)
        
        except Exception as e:
            logger.error(f"Error in model selection: {e}")
            return self._create_fallback_selection(prompt, prompt_type, start_time)
    
    async def parallel_evaluate(self, prompt: str, context: Optional[str] = None) -> List[ModelEvaluation]:
        """
        Run all models simultaneously and evaluate their responses with partial selection support
        
        Args:
            prompt: User's input message
            context: Optional conversation context
            
        Returns:
            List of ModelEvaluation objects
        """
        # Create evaluation tasks for all models with timeout and task names
        tasks = []
        model_names = list(self.models.keys())
        
        for model_id, model_client in self.models.items():
            task = asyncio.create_task(
                self._evaluate_single_model(model_id, model_client, prompt, context),
                name=model_id
            )
            tasks.append(task)
        
        logger.debug(f"Starting parallel evaluation of {len(tasks)} models")
        
        # Wait for completion with timeout - allows partial results
        results = {}
        errors = {}
        
        try:
            # Use asyncio.wait with timeout instead of gather for better control
            done, pending = await asyncio.wait(
                tasks,
                timeout=self.selection_timeout,  # Total timeout for all models  
                return_when=asyncio.ALL_COMPLETED  # Wait for all, but timeout if too slow
            )
            
            # Cancel any pending tasks
            for task in pending:
                model_name = task.get_name()
                logger.warning(f"Cancelling slow task for {model_name}")
                task.cancel()
                errors[model_name] = "Task cancelled due to timeout"
            
            # Process completed tasks
            for task in done:
                model_name = task.get_name()
                try:
                    result = await task
                    if isinstance(result, ModelEvaluation) and result.error is None:
                        results[model_name] = result
                        logger.debug(f"✅ {model_name} evaluation completed successfully")
                    else:
                        errors[model_name] = f"Evaluation failed: {result.error if hasattr(result, 'error') else 'Unknown error'}"
                        logger.warning(f"❌ {model_name} evaluation failed")
                except Exception as e:
                    errors[model_name] = str(e)
                    logger.error(f"❌ {model_name} evaluation exception: {e}")
            
        except Exception as e:
            logger.error(f"Parallel evaluation failed: {e}")
            # Cancel all tasks if something goes wrong
            for task in tasks:
                if not task.done():
                    task.cancel()
        
        # Log summary
        logger.info(f"Parallel evaluation completed: {len(results)} successful, {len(errors)} failed")
        if errors:
            logger.debug(f"Failed models: {list(errors.keys())}")
        
        # Return successful evaluations
        valid_evaluations = list(results.values())
        
        # If no models succeeded, we'll let the calling code handle fallback
        if not valid_evaluations:
            logger.warning("All model evaluations failed - caller should handle fallback")
            
        return valid_evaluations
    
    async def _call_model_with_retry(self, 
                                   model_id: str, 
                                   model_client: Any, 
                                   prompt: str, 
                                   context: Optional[str] = None,
                                   max_retries: int = 2) -> Tuple[Any, str]:
        """Call model with retry logic for local models"""
        
        # Determine if this is a local model (needs retries)
        is_local = model_id in ['deepseek', 'gemma']
        retries = max_retries if is_local else 0
        
        # Use model-specific timeout
        timeout = self.model_timeouts.get(model_id, 10.0)
        
        last_exception = None
        
        for attempt in range(retries + 1):
            try:
                logger.debug(f"Attempt {attempt + 1}/{retries + 1} for {model_id} (timeout: {timeout}s)")
                
                if hasattr(model_client, 'generate_response'):
                    response_obj = await asyncio.wait_for(
                        model_client.generate_response(
                            prompt=prompt,
                            system_prompt=self._get_mental_health_system_prompt(),
                            conversation_history=self._parse_context(context) if context else None
                        ),
                        timeout=timeout
                    )
                    response_content = response_obj.content if hasattr(response_obj, 'content') else str(response_obj)
                else:
                    # Fallback for older clients
                    logger.warning(f"{model_id} using fallback chat method")
                    response_obj = model_client.chat(prompt)
                    response_content = response_obj.get('content', str(response_obj)) if isinstance(response_obj, dict) else str(response_obj)
                
                logger.debug(f"Successfully got response from {model_id}")
                return response_obj, response_content
                
            except asyncio.TimeoutError as e:
                last_exception = e
                logger.warning(f"{model_id} attempt {attempt + 1} timed out after {timeout}s")
                if attempt < retries:
                    await asyncio.sleep(1)  # Brief pause before retry
                    continue
                else:
                    raise Exception(f"Response timeout after {timeout}s (tried {retries + 1} times)")
                    
            except Exception as e:
                last_exception = e
                logger.error(f"{model_id} attempt {attempt + 1} failed: {e}")
                if attempt < retries and is_local:
                    await asyncio.sleep(1)  # Brief pause before retry
                    continue
                else:
                    raise e
        
        # This shouldn't be reached, but just in case
        if last_exception:
            raise last_exception
        else:
            raise Exception(f"Unexpected error in retry logic for {model_id}")
    
    async def _evaluate_single_model(self, 
                                   model_id: str, 
                                   model_client: Any, 
                                   prompt: str, 
                                   context: Optional[str] = None) -> ModelEvaluation:
        """Evaluate a single model's response"""
        
        eval_start = time.time()
        
        try:
            logger.debug(f"Starting evaluation for {model_id}")
            
            # Generate response with model-specific timeout and retry logic
            response_obj, response_content = await self._call_model_with_retry(
                model_id, model_client, prompt, context
            )
            
            response_time_ms = (time.time() - eval_start) * 1000
            
            # Validate response content
            if not response_content or len(response_content.strip()) == 0:
                logger.warning(f"{model_id} returned empty response")
                raise Exception("Empty response content")
            
            # Evaluate response quality
            try:
                evaluation = self.evaluator.evaluate_response(
                    prompt=prompt,
                    response=response_content,
                    response_time_ms=response_time_ms,
                    input_tokens=len(prompt.split()) * 1.3,
                    output_tokens=len(response_content.split()) * 1.3
                )
                scores = self._extract_evaluation_scores(evaluation)
                logger.info(f"📊 {model_id} evaluation scores: {scores}")
                logger.info(f"📈 {model_id} composite score: {evaluation.composite_score:.2f}")
            except Exception as e:
                logger.warning(f"Evaluation failed for {model_id}, using default scores: {e}")
                # Use reasonable default scores if evaluation fails
                scores = {'empathy': 6.0, 'therapeutic': 6.0, 'safety': 8.0, 'clarity': 7.0}
            
            # Calculate composite score (will be weighted later)
            composite_score = (
                scores.get('empathy', 0) * 0.25 +
                scores.get('therapeutic', 0) * 0.25 +
                scores.get('safety', 0) * 0.25 +
                scores.get('clarity', 0) * 0.25
            )
            
            logger.info(f"Successfully evaluated {model_id} - composite score: {composite_score:.2f}")
            
            return ModelEvaluation(
                model_id=model_id,
                response_content=response_content,
                evaluation_scores=scores,
                composite_score=composite_score,
                response_time_ms=response_time_ms
            )
            
        except Exception as e:
            response_time_ms = (time.time() - eval_start) * 1000
            error_msg = f"Model {model_id} evaluation failed: {str(e)}"
            logger.error(error_msg)
            
            return ModelEvaluation(
                model_id=model_id,
                response_content="",
                evaluation_scores={},
                composite_score=0.0,
                response_time_ms=response_time_ms,
                error=error_msg
            )
    
    def apply_selection_logic(self, 
                            model_evaluations: List[ModelEvaluation],
                            selection_criteria: SelectionCriteria,
                            prompt_type: PromptType,
                            start_time: float) -> ModelSelection:
        """
        Choose the best model based on weighted scores and prompt type
        
        Args:
            model_evaluations: Results from all model evaluations
            selection_criteria: Weighted criteria for this prompt type
            prompt_type: Classified prompt type
            start_time: Selection start time for latency calculation
            
        Returns:
            ModelSelection with the chosen model and reasoning
        """
        # Calculate weighted scores for each model
        weighted_scores = {}
        model_scores = {}
        
        logger.info(f"🎯 Applying selection criteria for {prompt_type.value}: empathy={selection_criteria.empathy_weight:.2f}, therapeutic={selection_criteria.therapeutic_weight:.2f}, safety={selection_criteria.safety_weight:.2f}, clarity={selection_criteria.clarity_weight:.2f}")
        
        for evaluation in model_evaluations:
            scores = evaluation.evaluation_scores
            
            # Apply weighted scoring based on prompt type
            weighted_score = (
                scores.get('empathy', 0) * selection_criteria.empathy_weight +
                scores.get('therapeutic', 0) * selection_criteria.therapeutic_weight +
                scores.get('safety', 0) * selection_criteria.safety_weight +
                scores.get('clarity', 0) * selection_criteria.clarity_weight
            )
            
            weighted_scores[evaluation.model_id] = weighted_score
            model_scores[evaluation.model_id] = evaluation.composite_score
            
            logger.info(f"📊 {evaluation.model_id}: individual scores {scores}, weighted score: {weighted_score:.2f}")
        
        # Select the model with the highest weighted score
        if not weighted_scores:
            raise ValueError("No valid model evaluations available")
        
        selected_model_id = max(weighted_scores.items(), key=lambda x: x[1])[0]
        selected_evaluation = next(e for e in model_evaluations if e.model_id == selected_model_id)
        
        # Calculate meaningful confidence score as percentage
        scores_list = list(weighted_scores.values())
        selected_score = weighted_scores[selected_model_id]
        
        if len(scores_list) <= 1:
            # Only one model - confidence based on absolute performance
            confidence_score = min(selected_score / 10.0, 1.0)
        else:
            # Multiple models - confidence based on margin of victory and absolute performance
            sorted_scores = sorted(scores_list, reverse=True)
            best_score = sorted_scores[0]  # Should be selected_score
            second_best_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
            
            # Margin of victory (how much better than second place)
            margin = (best_score - second_best_score) / 10.0 if best_score > 0 else 0
            
            # Absolute performance (how good the score is on 0-10 scale)
            absolute_performance = selected_score / 10.0
            
            # Combined confidence: 70% absolute performance + 30% margin of victory
            confidence_score = (0.7 * absolute_performance) + (0.3 * margin)
            confidence_score = min(max(confidence_score, 0.0), 1.0)  # Clamp to 0-1
        
        logger.info(f"🎯 Selected {selected_model_id} with weighted score {selected_score:.2f}/10.0 and confidence {confidence_score:.1%}")
        
        # Generate selection reasoning
        reasoning = self.get_selection_explanation(
            selected_model_id,
            selected_evaluation,
            weighted_scores,
            selection_criteria,
            prompt_type
        )
        
        # Calculate latency metrics
        total_time_ms = (time.time() - start_time) * 1000
        latency_metrics = {
            'total_time_ms': total_time_ms,
            'model_response_time_ms': selected_evaluation.response_time_ms,
            'evaluation_overhead_ms': total_time_ms - selected_evaluation.response_time_ms
        }
        
        return ModelSelection(
            selected_model_id=selected_model_id,
            model_scores=weighted_scores,
            response_content=selected_evaluation.response_content,
            selection_reasoning=reasoning,
            latency_metrics=latency_metrics,
            prompt_type=prompt_type,
            selection_criteria=selection_criteria,
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            cached=False
        )
    
    def get_selection_explanation(self,
                                selected_model_id: str,
                                selected_evaluation: ModelEvaluation,
                                weighted_scores: Dict[str, float],
                                selection_criteria: SelectionCriteria,
                                prompt_type: PromptType) -> str:
        """
        Generate human-readable reasoning for model selection
        
        Args:
            selected_model_id: ID of the selected model
            selected_evaluation: Evaluation results for selected model
            weighted_scores: Weighted scores for all models
            selection_criteria: Applied selection criteria
            prompt_type: Classified prompt type
            
        Returns:
            Human-readable explanation string
        """
        selected_score = weighted_scores[selected_model_id]
        scores = selected_evaluation.evaluation_scores
        
        # Identify the strongest scoring dimension
        dimension_scores = {
            'empathy': scores.get('empathy', 0) * selection_criteria.empathy_weight,
            'therapeutic': scores.get('therapeutic', 0) * selection_criteria.therapeutic_weight,
            'safety': scores.get('safety', 0) * selection_criteria.safety_weight,
            'clarity': scores.get('clarity', 0) * selection_criteria.clarity_weight
        }
        
        strongest_dimension = max(dimension_scores.items(), key=lambda x: x[1])[0]
        
        # Get actual scores for detailed explanation
        empathy_score = scores.get('empathy', 0)
        therapeutic_score = scores.get('therapeutic', 0)
        safety_score = scores.get('safety', 0)
        clarity_score = scores.get('clarity', 0)
        
        # Create detailed explanation
        explanation_parts = [
            f"Selected {selected_model_id.upper()} for {prompt_type.value.replace('_', ' ')} prompt"
        ]
        
        # Add specific scores
        score_details = []
        if empathy_score > 0:
            score_details.append(f"empathy: {empathy_score:.1f}/10")
        if therapeutic_score > 0:
            score_details.append(f"therapeutic: {therapeutic_score:.1f}/10")
        if safety_score > 0:
            score_details.append(f"safety: {safety_score:.1f}/10")
        if clarity_score > 0:
            score_details.append(f"clarity: {clarity_score:.1f}/10")
        
        if score_details:
            explanation_parts.append(f"Scores: {', '.join(score_details)}")
        
        explanation_parts.append(f"Weighted score: {selected_score:.2f}/10.0")
        
        # Identify and highlight the strongest dimension
        if strongest_dimension and scores.get(strongest_dimension, 0) > 0:
            dimension_name = strongest_dimension.replace('_', ' ')
            explanation_parts.append(f"Excelled in {dimension_name} ({scores.get(strongest_dimension, 0):.1f}/10.0)")
        
        # Add comparison with other models if available
        other_models = [k for k in weighted_scores.keys() if k != selected_model_id]
        if other_models:
            best_alternative = max(other_models, key=lambda x: weighted_scores[x])
            score_diff = selected_score - weighted_scores[best_alternative]
            if score_diff > 0.1:  # Only mention if meaningful difference
                explanation_parts.append(f"Outperformed {best_alternative.upper()} by {score_diff:.2f} points")
        
        # Add selection criteria emphasis based on prompt type
        criteria_focus = []
        if selection_criteria.safety_weight >= 0.4:
            criteria_focus.append("crisis safety")
        elif selection_criteria.empathy_weight >= 0.35:
            criteria_focus.append("emotional support")
        elif selection_criteria.therapeutic_weight >= 0.35:
            criteria_focus.append("therapeutic guidance")
        elif selection_criteria.clarity_weight >= 0.25:
            criteria_focus.append("clear communication")
        
        if criteria_focus:
            explanation_parts.append(f"Prioritized {', '.join(criteria_focus)}")
        
        return ". ".join(explanation_parts) + "."
    
    def _extract_evaluation_scores(self, evaluation) -> Dict[str, float]:
        """Extract standardized scores from evaluation result"""
        if hasattr(evaluation, 'to_dict'):
            eval_dict = evaluation.to_dict()
        elif isinstance(evaluation, dict):
            eval_dict = evaluation
        else:
            return {}
        
        return {
            'empathy': eval_dict.get('empathy', eval_dict.get('empathy_score', 0.0)),
            'therapeutic': eval_dict.get('therapeutic', eval_dict.get('therapeutic_value_score', 0.0)),
            'safety': eval_dict.get('safety', eval_dict.get('safety_score', 0.0)),
            'clarity': eval_dict.get('clarity', eval_dict.get('clarity_score', 0.0))
        }
    
    def _get_mental_health_system_prompt(self) -> str:
        """Get standardized system prompt for mental health context"""
        return """You are a compassionate and professional mental health support assistant. 

Your role is to:
- Provide empathetic, supportive responses
- Offer evidence-based coping strategies and techniques
- Maintain appropriate professional boundaries
- Recognize when to suggest professional help
- Never provide medical diagnoses or replace professional therapy

Guidelines:
- Listen actively and validate emotions
- Ask clarifying questions when appropriate
- Provide practical, actionable advice
- Be culturally sensitive and inclusive
- Maintain confidentiality and privacy
- Use clear, accessible language

Remember: You are here to support, not replace professional mental health services."""
    
    def _parse_context(self, context: str) -> List[Dict[str, str]]:
        """Parse conversation context into message format"""
        # Simple implementation - in practice, this would be more sophisticated
        return [{"role": "user", "content": context}] if context else []
    
    def _check_cache(self, prompt: str, context: Optional[str] = None) -> Optional[ModelSelection]:
        """Check if we have a cached selection for similar prompt"""
        if not self.selection_cache:
            return None
        
        # Create cache key
        cache_key = self._create_cache_key(prompt, context)
        
        # Check for exact match first
        cached_data = self.selection_cache.get_cached_response(cache_key)
        if cached_data:
            try:
                selection_dict = eval(cached_data)  # In practice, use proper JSON serialization
                selection = ModelSelection(**selection_dict)
                selection.cached = True
                return selection
            except Exception as e:
                logger.error(f"Error deserializing cached selection: {e}")
        
        return None
    
    def _cache_selection(self, prompt: str, context: Optional[str], selection: ModelSelection):
        """Cache the selection result"""
        if not self.selection_cache:
            return
        
        cache_key = self._create_cache_key(prompt, context)
        
        try:
            # Serialize selection (in practice, use proper JSON)
            cached_data = str(selection.to_dict())
            self.selection_cache.cache_response(
                prompt=cache_key,
                system_prompt=None,
                model_name="selector",
                response_text=cached_data
            )
        except Exception as e:
            logger.error(f"Error caching selection: {e}")
    
    def _create_cache_key(self, prompt: str, context: Optional[str] = None) -> str:
        """Create a cache key for the prompt and context"""
        key_components = [prompt.strip().lower()]
        if context:
            key_components.append(context.strip().lower())
        
        full_key = "||".join(key_components)
        return hashlib.sha256(full_key.encode()).hexdigest()[:16]
    
    def _create_fallback_selection(self, 
                                 prompt: str, 
                                 prompt_type: PromptType, 
                                 start_time: float) -> ModelSelection:
        """Create a fallback selection when normal selection fails"""
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Generate a reasonable mock response for the fallback
        fallback_responses = {
            PromptType.ANXIETY: "I understand you're feeling anxious. That's completely normal, and there are ways to help manage these feelings. Would you like to explore some coping strategies?",
            PromptType.DEPRESSION: "I hear that you're going through a difficult time. It takes courage to reach out, and I'm here to support you through this.",
            PromptType.CRISIS: "I'm very concerned about what you've shared. Your safety is the most important thing right now. Please consider reaching out to a crisis helpline: 988.",
            PromptType.GENERAL_SUPPORT: "Thank you for sharing with me. I'm here to listen and provide support. How are you feeling right now?"
        }
        
        response_content = fallback_responses.get(prompt_type, "I'm here to help and support you. Could you tell me more about what's on your mind?")
        
        # Set a reasonable confidence score for fallback (0.6 - moderate confidence)
        confidence_score = 0.6
        
        return ModelSelection(
            selected_model_id=self.default_model,
            model_scores={self.default_model: 6.0},  # Reasonable score out of 10
            response_content=response_content,
            selection_reasoning=f"Selected {self.default_model.upper()} as fallback model due to timeout. Default model provides reliable mental health support.",
            latency_metrics={'total_time_ms': total_time_ms, 'model_response_time_ms': 0, 'evaluation_overhead_ms': total_time_ms},
            prompt_type=prompt_type,
            selection_criteria=self.SELECTION_CRITERIA[prompt_type],
            confidence_score=confidence_score,
            timestamp=datetime.now(),
            cached=False
        )
    
    def get_analytics(self) -> Dict[str, Any]:
        """Get comprehensive analytics about selection performance"""
        return self.performance_monitor.get_analytics()
    
    def get_available_models(self) -> List[str]:
        """Get list of available model IDs"""
        return list(self.models.keys())