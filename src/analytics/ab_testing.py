"""
A/B Testing Framework for Model Selection Experiments

Comprehensive experimentation system for testing different model selection strategies
with statistical rigor and real-time monitoring of experiment performance.
"""

import asyncio
import logging
import json
import random
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
from pathlib import Path

from ..chat.dynamic_model_selector import PromptType, ModelSelection, DynamicModelSelector

logger = logging.getLogger(__name__)


class ExperimentStatus(Enum):
    """Experiment lifecycle status"""
    DRAFT = "draft"           # Being configured
    ACTIVE = "active"         # Currently running
    PAUSED = "paused"         # Temporarily stopped
    COMPLETED = "completed"   # Finished successfully
    FAILED = "failed"         # Ended due to error
    ARCHIVED = "archived"     # Historical record


class SelectionStrategy(Enum):
    """Available model selection strategies for experiments"""
    BASELINE = "baseline"                    # Standard dynamic selection
    WEIGHTED_BY_TYPE = "weighted_by_type"   # Enhanced weights per prompt type
    ML_OPTIMIZED = "ml_optimized"           # ML-predicted optimal selection
    USER_HISTORY = "user_history"           # Based on user's past preferences
    COST_OPTIMIZED = "cost_optimized"       # Prioritize cost efficiency
    SPEED_OPTIMIZED = "speed_optimized"     # Prioritize response speed
    SAFETY_FIRST = "safety_first"           # Maximum safety prioritization


@dataclass
class ExperimentConfig:
    """Configuration for an A/B test experiment"""
    name: str
    description: str
    strategies: Dict[str, SelectionStrategy]  # variant_name -> strategy
    traffic_allocation: Dict[str, float]      # variant_name -> percentage
    
    # Experiment parameters
    min_sample_size: int = 100               # Minimum samples per variant
    max_duration_days: int = 30              # Maximum experiment duration
    confidence_level: float = 0.95           # Statistical confidence
    minimum_effect_size: float = 0.05        # Minimum detectable effect
    
    # Success metrics
    primary_metric: str = "user_satisfaction" # Primary success metric
    secondary_metrics: List[str] = field(default_factory=lambda: [
        "response_time", "model_cost", "safety_score", "task_completion"
    ])
    
    # Targeting criteria
    user_segments: Optional[List[str]] = None # Target specific user segments
    prompt_types: Optional[List[PromptType]] = None # Target specific prompt types
    
    # Safety controls
    early_stopping_enabled: bool = True      # Stop if variant performs poorly
    safety_threshold: float = 0.8            # Minimum safety score to continue
    
    def validate(self) -> List[str]:
        """Validate experiment configuration"""
        errors = []
        
        # Check traffic allocation sums to 1.0
        total_allocation = sum(self.traffic_allocation.values())
        if abs(total_allocation - 1.0) > 0.01:
            errors.append(f"Traffic allocation must sum to 1.0, got {total_allocation}")
        
        # Check strategies match variants
        if set(self.strategies.keys()) != set(self.traffic_allocation.keys()):
            errors.append("Strategy variants must match traffic allocation variants")
        
        # Check minimum sample size is reasonable
        if self.min_sample_size < 50:
            errors.append("Minimum sample size should be at least 50 per variant")
        
        return errors


@dataclass
class ExperimentResult:
    """Results from an A/B test experiment"""
    experiment_id: str
    variant_name: str
    user_id: str
    prompt: str
    prompt_type: PromptType
    
    # Selection details
    selected_model: str
    selection_strategy: SelectionStrategy
    confidence_score: float
    selection_time_ms: float
    
    # Outcome metrics
    user_satisfaction: Optional[float] = None    # 1-5 rating
    response_time_ms: Optional[float] = None     # Total response time
    model_cost_usd: Optional[float] = None       # Cost of model call
    safety_score: Optional[float] = None         # Safety assessment
    task_completion: Optional[bool] = None       # Did user complete task
    
    # Metadata
    timestamp: datetime = field(default_factory=datetime.now)
    session_id: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage/analysis"""
        return {
            'experiment_id': self.experiment_id,
            'variant_name': self.variant_name,
            'user_id': self.user_id,
            'prompt': self.prompt[:100],  # Truncate for privacy
            'prompt_type': self.prompt_type.value,
            'selected_model': self.selected_model,
            'selection_strategy': self.selection_strategy.value,
            'confidence_score': self.confidence_score,
            'selection_time_ms': self.selection_time_ms,
            'user_satisfaction': self.user_satisfaction,
            'response_time_ms': self.response_time_ms,
            'model_cost_usd': self.model_cost_usd,
            'safety_score': self.safety_score,
            'task_completion': self.task_completion,
            'timestamp': self.timestamp.isoformat(),
            'session_id': self.session_id,
            'metadata': self.metadata
        }


@dataclass
class ExperimentStats:
    """Statistical analysis of experiment results"""
    experiment_id: str
    variant_name: str
    
    # Sample statistics
    sample_size: int = 0
    
    # Primary metric statistics
    primary_mean: float = 0.0
    primary_std: float = 0.0
    primary_confidence_interval: Tuple[float, float] = (0.0, 0.0)
    
    # Secondary metrics
    secondary_stats: Dict[str, Dict[str, float]] = field(default_factory=dict)
    
    # Statistical significance
    is_significant: bool = False
    p_value: float = 1.0
    effect_size: float = 0.0
    
    # Performance metrics
    conversion_rate: float = 0.0
    retention_rate: float = 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'experiment_id': self.experiment_id,
            'variant_name': self.variant_name,
            'sample_size': self.sample_size,
            'primary_mean': self.primary_mean,
            'primary_std': self.primary_std,
            'primary_confidence_interval': self.primary_confidence_interval,
            'secondary_stats': self.secondary_stats,
            'is_significant': self.is_significant,
            'p_value': self.p_value,
            'effect_size': self.effect_size,
            'conversion_rate': self.conversion_rate,
            'retention_rate': self.retention_rate
        }


class BaseSelectionStrategy:
    """Base class for model selection strategies"""
    
    def __init__(self, name: str):
        self.name = name
    
    async def select_model(self, 
                          prompt: str, 
                          prompt_type: PromptType,
                          user_context: Dict[str, Any] = None) -> ModelSelection:
        """Select model using this strategy"""
        raise NotImplementedError


class StandardSelection(BaseSelectionStrategy):
    """Baseline standard dynamic selection"""
    
    def __init__(self, model_selector: DynamicModelSelector):
        super().__init__("standard")
        self.model_selector = model_selector
    
    async def select_model(self, prompt: str, prompt_type: PromptType, user_context: Dict[str, Any] = None) -> ModelSelection:
        return await self.model_selector.select_model(prompt, prompt_type)


class WeightedByPromptType(BaseSelectionStrategy):
    """Enhanced weights based on prompt type performance"""
    
    def __init__(self, model_selector: DynamicModelSelector):
        super().__init__("weighted_by_type")
        self.model_selector = model_selector
        
        # Enhanced weights based on empirical performance
        self.type_weight_adjustments = {
            PromptType.CRISIS: {
                'safety_weight': 0.6,      # Increased safety focus
                'empathy_weight': 0.25,
                'therapeutic_weight': 0.15
            },
            PromptType.ANXIETY: {
                'therapeutic_weight': 0.4,  # Increased therapeutic focus
                'empathy_weight': 0.35,
                'safety_weight': 0.25
            },
            PromptType.DEPRESSION: {
                'therapeutic_weight': 0.4,
                'empathy_weight': 0.35,
                'safety_weight': 0.25
            },
            PromptType.INFORMATION_SEEKING: {
                'clarity_weight': 0.6,     # Increased clarity focus
                'therapeutic_weight': 0.25,
                'empathy_weight': 0.15
            }
        }
    
    async def select_model(self, prompt: str, prompt_type: PromptType, user_context: Dict[str, Any] = None) -> ModelSelection:
        # Apply weight adjustments
        original_criteria = self.model_selector.SELECTION_CRITERIA.get(prompt_type)
        if original_criteria and prompt_type in self.type_weight_adjustments:
            adjustments = self.type_weight_adjustments[prompt_type]
            
            # Temporarily modify weights
            modified_criteria = original_criteria._replace(**adjustments)
            original_criteria_backup = self.model_selector.SELECTION_CRITERIA[prompt_type]
            self.model_selector.SELECTION_CRITERIA[prompt_type] = modified_criteria
            
            try:
                selection = await self.model_selector.select_model(prompt, prompt_type)
                selection.reasoning += f" [Enhanced weights for {prompt_type.value}]"
                return selection
            finally:
                # Restore original criteria
                self.model_selector.SELECTION_CRITERIA[prompt_type] = original_criteria_backup
        
        return await self.model_selector.select_model(prompt, prompt_type)


class CostOptimizedSelection(BaseSelectionStrategy):
    """Cost-optimized model selection"""
    
    def __init__(self, model_selector: DynamicModelSelector):
        super().__init__("cost_optimized")
        self.model_selector = model_selector
        
        # Model cost rankings (lower is cheaper)
        self.model_costs = {
            "gpt-3.5-turbo": 1,
            "claude-3-haiku": 2,
            "claude-3-sonnet": 3,
            "gpt-4-turbo": 4,
            "claude-3-opus": 5
        }
    
    async def select_model(self, prompt: str, prompt_type: PromptType, user_context: Dict[str, Any] = None) -> ModelSelection:
        # Get standard selection
        selection = await self.model_selector.select_model(prompt, prompt_type)
        
        # Apply cost optimization
        if selection.model_scores:
            # Re-rank considering cost
            cost_adjusted_scores = {}
            for model, score in selection.model_scores.items():
                cost_rank = self.model_costs.get(model, 3)  # Default to medium cost
                cost_penalty = (cost_rank - 1) * 0.1  # 10% penalty per cost tier
                cost_adjusted_scores[model] = max(0, score - cost_penalty)
            
            # Select model with best cost-adjusted score
            best_model = max(cost_adjusted_scores.items(), key=lambda x: x[1])
            
            if best_model[0] != selection.selected_model:
                selection.selected_model = best_model[0]
                selection.confidence_score = best_model[1]
                selection.reasoning += f" [Cost-optimized selection]"
        
        return selection


class SafetyFirstSelection(BaseSelectionStrategy):
    """Safety-prioritized model selection"""
    
    def __init__(self, model_selector: DynamicModelSelector):
        super().__init__("safety_first")
        self.model_selector = model_selector
        
        # Safety rankings for models
        self.model_safety_scores = {
            "claude-3-opus": 0.95,
            "claude-3-sonnet": 0.90,
            "gpt-4-turbo": 0.85,
            "claude-3-haiku": 0.80,
            "gpt-3.5-turbo": 0.75
        }
    
    async def select_model(self, prompt: str, prompt_type: PromptType, user_context: Dict[str, Any] = None) -> ModelSelection:
        # For crisis situations, always use highest safety model
        if prompt_type == PromptType.CRISIS:
            best_safety_model = max(self.model_safety_scores.items(), key=lambda x: x[1])
            
            return ModelSelection(
                selected_model=best_safety_model[0],
                confidence_score=0.95,
                prompt_classification=prompt_type,
                reasoning=f"Safety-first selection for crisis: {best_safety_model[0]} (safety: {best_safety_model[1]:.2f})",
                evaluation_time_ms=10.0,
                model_scores={best_safety_model[0]: best_safety_model[1]}
            )
        
        # For other types, apply safety weighting
        selection = await self.model_selector.select_model(prompt, prompt_type)
        
        if selection.model_scores:
            safety_adjusted_scores = {}
            for model, score in selection.model_scores.items():
                safety_score = self.model_safety_scores.get(model, 0.7)
                safety_adjusted_scores[model] = score * safety_score
            
            best_model = max(safety_adjusted_scores.items(), key=lambda x: x[1])
            selection.selected_model = best_model[0]
            selection.confidence_score = best_model[1]
            selection.reasoning += f" [Safety-weighted selection]"
        
        return selection


class ModelSelectionExperiment:
    """
    Individual A/B test experiment for model selection strategies
    
    Features:
    - Multiple variant support with custom traffic allocation
    - Real-time statistical monitoring
    - Early stopping for poor performing variants
    - Comprehensive result tracking and analysis
    """
    
    def __init__(self, config: ExperimentConfig):
        self.config = config
        self.experiment_id = str(uuid.uuid4())
        self.status = ExperimentStatus.DRAFT
        
        # Results storage
        self.results: List[ExperimentResult] = []
        self.variant_stats: Dict[str, ExperimentStats] = {}
        
        # Strategy implementations
        self.strategies: Dict[str, BaseSelectionStrategy] = {}
        
        # Experiment tracking
        self.start_time: Optional[datetime] = None
        self.end_time: Optional[datetime] = None
        self.participant_assignments: Dict[str, str] = {}  # user_id -> variant
        
        # Statistical monitoring
        self.last_analysis_time: Optional[datetime] = None
        self.analysis_frequency_minutes = 60  # Analyze hourly
        
        logger.info(f"Created experiment: {self.config.name} ({self.experiment_id})")
    
    def add_strategy(self, variant_name: str, strategy: BaseSelectionStrategy):
        """Add a strategy implementation for a variant"""
        if variant_name not in self.config.strategies:
            raise ValueError(f"Variant {variant_name} not defined in experiment config")
        
        self.strategies[variant_name] = strategy
        logger.info(f"Added strategy {strategy.name} for variant {variant_name}")
    
    def start_experiment(self) -> bool:
        """Start the experiment"""
        # Validate configuration
        config_errors = self.config.validate()
        if config_errors:
            logger.error(f"Cannot start experiment: {config_errors}")
            return False
        
        # Check strategies are implemented
        missing_strategies = set(self.config.strategies.keys()) - set(self.strategies.keys())
        if missing_strategies:
            logger.error(f"Missing strategy implementations: {missing_strategies}")
            return False
        
        self.status = ExperimentStatus.ACTIVE
        self.start_time = datetime.now()
        
        logger.info(f"Started experiment {self.config.name} at {self.start_time}")
        return True
    
    def assign_user_to_variant(self, user_id: str) -> str:
        """Assign user to experiment variant"""
        if user_id in self.participant_assignments:
            return self.participant_assignments[user_id]
        
        # Apply targeting criteria
        if not self._user_matches_criteria(user_id):
            return "control"  # Default to control if doesn't match
        
        # Random assignment based on traffic allocation
        random_value = random.random()
        cumulative_allocation = 0.0
        
        for variant_name, allocation in self.config.traffic_allocation.items():
            cumulative_allocation += allocation
            if random_value <= cumulative_allocation:
                self.participant_assignments[user_id] = variant_name
                logger.debug(f"Assigned user {user_id} to variant {variant_name}")
                return variant_name
        
        # Fallback to first variant
        first_variant = list(self.config.traffic_allocation.keys())[0]
        self.participant_assignments[user_id] = first_variant
        return first_variant
    
    async def select_model_for_user(self, 
                                  user_id: str, 
                                  prompt: str, 
                                  prompt_type: PromptType,
                                  session_id: str = None) -> Tuple[ModelSelection, str]:
        """
        Select model using assigned experiment variant
        
        Returns:
            Tuple of (ModelSelection, variant_name)
        """
        if self.status != ExperimentStatus.ACTIVE:
            raise RuntimeError(f"Experiment not active: {self.status}")
        
        # Get user's assigned variant
        variant_name = self.assign_user_to_variant(user_id)
        
        # Apply targeting filters
        if self.config.prompt_types and prompt_type not in self.config.prompt_types:
            # Use control strategy for non-targeted prompt types
            variant_name = "control" if "control" in self.strategies else list(self.strategies.keys())[0]
        
        # Get strategy for variant
        strategy = self.strategies.get(variant_name)
        if not strategy:
            raise ValueError(f"No strategy found for variant {variant_name}")
        
        # Perform model selection
        start_time = time.time()
        selection = await strategy.select_model(prompt, prompt_type)
        selection_time_ms = (time.time() - start_time) * 1000
        
        # Record result
        result = ExperimentResult(
            experiment_id=self.experiment_id,
            variant_name=variant_name,
            user_id=user_id,
            prompt=prompt,
            prompt_type=prompt_type,
            selected_model=selection.selected_model,
            selection_strategy=self.config.strategies[variant_name],
            confidence_score=selection.confidence_score,
            selection_time_ms=selection_time_ms,
            session_id=session_id
        )
        
        self.results.append(result)
        
        # Trigger analysis if needed
        await self._maybe_analyze_results()
        
        return selection, variant_name
    
    def record_outcome_metrics(self, 
                             result_id: str = None,
                             user_id: str = None,
                             user_satisfaction: float = None,
                             response_time_ms: float = None,
                             model_cost_usd: float = None,
                             safety_score: float = None,
                             task_completion: bool = None):
        """Record outcome metrics for a result"""
        # Find the result to update
        target_result = None
        if result_id:
            target_result = next((r for r in self.results if r.experiment_id == result_id), None)
        elif user_id:
            # Find most recent result for user
            user_results = [r for r in self.results if r.user_id == user_id]
            target_result = max(user_results, key=lambda r: r.timestamp) if user_results else None
        
        if not target_result:
            logger.warning(f"Could not find result to update metrics for")
            return
        
        # Update metrics
        if user_satisfaction is not None:
            target_result.user_satisfaction = user_satisfaction
        if response_time_ms is not None:
            target_result.response_time_ms = response_time_ms
        if model_cost_usd is not None:
            target_result.model_cost_usd = model_cost_usd
        if safety_score is not None:
            target_result.safety_score = safety_score
        if task_completion is not None:
            target_result.task_completion = task_completion
        
        logger.debug(f"Updated outcome metrics for result {target_result.experiment_id}")
    
    async def _maybe_analyze_results(self):
        """Analyze results if enough time has passed"""
        if (self.last_analysis_time is None or 
            datetime.now() - self.last_analysis_time >= timedelta(minutes=self.analysis_frequency_minutes)):
            
            await self._analyze_results()
            self.last_analysis_time = datetime.now()
    
    async def _analyze_results(self):
        """Perform statistical analysis of current results"""
        if len(self.results) < self.config.min_sample_size:
            return  # Not enough data yet
        
        # Group results by variant
        variant_results = defaultdict(list)
        for result in self.results:
            variant_results[result.variant_name].append(result)
        
        # Analyze each variant
        for variant_name, results in variant_results.items():
            if len(results) < 10:  # Need minimum data
                continue
            
            stats = self._calculate_variant_stats(variant_name, results)
            self.variant_stats[variant_name] = stats
            
            # Check for early stopping conditions
            if self.config.early_stopping_enabled:
                await self._check_early_stopping(variant_name, stats)
        
        # Check if experiment should complete
        await self._check_completion_criteria()
    
    def _calculate_variant_stats(self, variant_name: str, results: List[ExperimentResult]) -> ExperimentStats:
        """Calculate statistical metrics for a variant"""
        stats = ExperimentStats(
            experiment_id=self.experiment_id,
            variant_name=variant_name,
            sample_size=len(results)
        )
        
        # Primary metric analysis (user satisfaction)
        satisfaction_scores = [r.user_satisfaction for r in results if r.user_satisfaction is not None]
        if satisfaction_scores:
            stats.primary_mean = statistics.mean(satisfaction_scores)
            if len(satisfaction_scores) > 1:
                stats.primary_std = statistics.stdev(satisfaction_scores)
                
                # Calculate confidence interval (assuming normal distribution)
                margin_of_error = 1.96 * (stats.primary_std / (len(satisfaction_scores) ** 0.5))
                stats.primary_confidence_interval = (
                    stats.primary_mean - margin_of_error,
                    stats.primary_mean + margin_of_error
                )
        
        # Secondary metrics
        secondary_data = {
            'response_time': [r.response_time_ms for r in results if r.response_time_ms is not None],
            'model_cost': [r.model_cost_usd for r in results if r.model_cost_usd is not None],
            'safety_score': [r.safety_score for r in results if r.safety_score is not None],
            'selection_time': [r.selection_time_ms for r in results]
        }
        
        for metric_name, values in secondary_data.items():
            if values:
                stats.secondary_stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'std': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'count': len(values)
                }
        
        # Performance metrics
        completed_tasks = [r for r in results if r.task_completion is True]
        stats.conversion_rate = len(completed_tasks) / len(results) if results else 0.0
        
        return stats
    
    async def _check_early_stopping(self, variant_name: str, stats: ExperimentStats):
        """Check if variant should be stopped early"""
        # Safety check
        safety_scores = [r.safety_score for r in self.results 
                        if r.variant_name == variant_name and r.safety_score is not None]
        
        if safety_scores:
            avg_safety = statistics.mean(safety_scores)
            if avg_safety < self.config.safety_threshold:
                logger.warning(f"Variant {variant_name} below safety threshold: {avg_safety:.2f}")
                # In a real implementation, you might pause this variant
        
        # Performance check (if significantly worse than control)
        if "control" in self.variant_stats and variant_name != "control":
            control_stats = self.variant_stats["control"]
            
            # Simple check - if variant mean is significantly lower
            if (stats.primary_mean < control_stats.primary_mean * 0.8 and 
                stats.sample_size > 50):
                logger.warning(f"Variant {variant_name} performing poorly vs control")
    
    async def _check_completion_criteria(self):
        """Check if experiment should be completed"""
        # Time-based completion
        if self.start_time and datetime.now() - self.start_time >= timedelta(days=self.config.max_duration_days):
            await self.complete_experiment("max_duration_reached")
            return
        
        # Sample size completion
        min_samples_reached = all(
            len([r for r in self.results if r.variant_name == variant]) >= self.config.min_sample_size
            for variant in self.config.strategies.keys()
        )
        
        if min_samples_reached and len(self.results) >= self.config.min_sample_size * len(self.config.strategies):
            # Check for statistical significance
            if self._has_statistical_significance():
                await self.complete_experiment("statistical_significance")
    
    def _has_statistical_significance(self) -> bool:
        """Check if results show statistical significance"""
        if len(self.variant_stats) < 2:
            return False
        
        # Simple check - in practice you'd use proper statistical tests
        variants = list(self.variant_stats.values())
        means = [v.primary_mean for v in variants]
        
        if not means:
            return False
        
        # Check if difference is greater than minimum effect size
        max_mean = max(means)
        min_mean = min(means)
        effect_size = (max_mean - min_mean) / max_mean if max_mean > 0 else 0
        
        return effect_size >= self.config.minimum_effect_size
    
    async def complete_experiment(self, reason: str = "manual"):
        """Complete the experiment"""
        self.status = ExperimentStatus.COMPLETED
        self.end_time = datetime.now()
        
        # Final analysis
        await self._analyze_results()
        
        logger.info(f"Completed experiment {self.config.name}: {reason}")
    
    def pause_experiment(self):
        """Pause the experiment"""
        self.status = ExperimentStatus.PAUSED
        logger.info(f"Paused experiment {self.config.name}")
    
    def resume_experiment(self):
        """Resume the experiment"""
        if self.status == ExperimentStatus.PAUSED:
            self.status = ExperimentStatus.ACTIVE
            logger.info(f"Resumed experiment {self.config.name}")
    
    def _user_matches_criteria(self, user_id: str) -> bool:
        """Check if user matches experiment targeting criteria"""
        # In a real implementation, you'd check user segments, demographics, etc.
        # For now, include all users
        return True
    
    def get_results_summary(self) -> Dict[str, Any]:
        """Get comprehensive results summary"""
        return {
            'experiment_id': self.experiment_id,
            'name': self.config.name,
            'status': self.status.value,
            'start_time': self.start_time.isoformat() if self.start_time else None,
            'end_time': self.end_time.isoformat() if self.end_time else None,
            'total_participants': len(set(r.user_id for r in self.results)),
            'total_results': len(self.results),
            'variant_stats': {name: stats.to_dict() for name, stats in self.variant_stats.items()},
            'config': {
                'traffic_allocation': self.config.traffic_allocation,
                'primary_metric': self.config.primary_metric,
                'min_sample_size': self.config.min_sample_size
            }
        }
    
    def export_results(self, file_path: str):
        """Export results to file"""
        results_data = {
            'experiment': self.get_results_summary(),
            'detailed_results': [result.to_dict() for result in self.results]
        }
        
        with open(file_path, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Exported experiment results to {file_path}")


class ExperimentManager:
    """
    Manages multiple A/B test experiments
    
    Features:
    - Multiple concurrent experiments
    - Experiment lifecycle management
    - Cross-experiment analysis
    - Resource allocation and conflict resolution
    """
    
    def __init__(self, model_selector: DynamicModelSelector):
        self.model_selector = model_selector
        self.experiments: Dict[str, ModelSelectionExperiment] = {}
        self.active_experiments: Set[str] = set()
        
        # Pre-built strategies
        self.standard_strategy = StandardSelection(model_selector)
        self.weighted_strategy = WeightedByPromptType(model_selector)
        self.cost_optimized_strategy = CostOptimizedSelection(model_selector)
        self.safety_first_strategy = SafetyFirstSelection(model_selector)
        
        logger.info("ExperimentManager initialized")
    
    def create_experiment(self, config: ExperimentConfig) -> str:
        """Create a new experiment"""
        experiment = ModelSelectionExperiment(config)
        
        # Add strategy implementations
        for variant_name, strategy_type in config.strategies.items():
            if strategy_type == SelectionStrategy.BASELINE:
                experiment.add_strategy(variant_name, self.standard_strategy)
            elif strategy_type == SelectionStrategy.WEIGHTED_BY_TYPE:
                experiment.add_strategy(variant_name, self.weighted_strategy)
            elif strategy_type == SelectionStrategy.COST_OPTIMIZED:
                experiment.add_strategy(variant_name, self.cost_optimized_strategy)
            elif strategy_type == SelectionStrategy.SAFETY_FIRST:
                experiment.add_strategy(variant_name, self.safety_first_strategy)
            else:
                logger.warning(f"Unknown strategy type: {strategy_type}")
        
        self.experiments[experiment.experiment_id] = experiment
        logger.info(f"Created experiment: {config.name} ({experiment.experiment_id})")
        
        return experiment.experiment_id
    
    def start_experiment(self, experiment_id: str) -> bool:
        """Start an experiment"""
        experiment = self.experiments.get(experiment_id)
        if not experiment:
            logger.error(f"Experiment not found: {experiment_id}")
            return False
        
        success = experiment.start_experiment()
        if success:
            self.active_experiments.add(experiment_id)
        
        return success
    
    async def select_model_for_user(self, 
                                  user_id: str, 
                                  prompt: str, 
                                  prompt_type: PromptType,
                                  session_id: str = None) -> ModelSelection:
        """
        Select model using active experiments
        
        If user is in multiple experiments, prioritize by creation order
        """
        # Find active experiments that apply to this user/prompt
        applicable_experiments = []
        for exp_id in self.active_experiments:
            experiment = self.experiments[exp_id]
            
            # Check if prompt type matches experiment criteria
            if (experiment.config.prompt_types is None or 
                prompt_type in experiment.config.prompt_types):
                applicable_experiments.append(experiment)
        
        if not applicable_experiments:
            # No active experiments, use standard selection
            return await self.standard_strategy.select_model(prompt, prompt_type)
        
        # Use first applicable experiment
        experiment = applicable_experiments[0]
        selection, variant = await experiment.select_model_for_user(
            user_id, prompt, prompt_type, session_id
        )
        
        # Add experiment info to reasoning
        selection.reasoning += f" [Experiment: {experiment.config.name}, Variant: {variant}]"
        
        return selection
    
    def record_feedback(self, 
                       user_id: str,
                       user_satisfaction: float = None,
                       response_time_ms: float = None,
                       model_cost_usd: float = None,
                       safety_score: float = None,
                       task_completion: bool = None):
        """Record feedback for all applicable experiments"""
        for experiment in self.experiments.values():
            if experiment.status == ExperimentStatus.ACTIVE:
                experiment.record_outcome_metrics(
                    user_id=user_id,
                    user_satisfaction=user_satisfaction,
                    response_time_ms=response_time_ms,
                    model_cost_usd=model_cost_usd,
                    safety_score=safety_score,
                    task_completion=task_completion
                )
    
    def get_experiment_status(self) -> Dict[str, Any]:
        """Get status of all experiments"""
        return {
            'total_experiments': len(self.experiments),
            'active_experiments': len(self.active_experiments),
            'experiments': {
                exp_id: {
                    'name': exp.config.name,
                    'status': exp.status.value,
                    'participants': len(set(r.user_id for r in exp.results)),
                    'results': len(exp.results)
                }
                for exp_id, exp in self.experiments.items()
            }
        }
    
    def export_all_experiments(self, output_dir: str):
        """Export all experiment data"""
        output_path = Path(output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        for exp_id, experiment in self.experiments.items():
            file_path = output_path / f"experiment_{exp_id}.json"
            experiment.export_results(str(file_path))
        
        # Export summary
        summary_path = output_path / "experiments_summary.json"
        with open(summary_path, 'w') as f:
            json.dump(self.get_experiment_status(), f, indent=2)
        
        logger.info(f"Exported all experiments to {output_dir}")