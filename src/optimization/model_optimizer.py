"""
Model Optimization and Warm-up System

Advanced model management system that pre-loads models, manages warm-up strategies,
and optimizes model performance through intelligent resource management.
"""

import asyncio
import logging
import time
import json
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Set, Callable
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from pathlib import Path

from ..chat.dynamic_model_selector import PromptType

logger = logging.getLogger(__name__)


class WarmupStrategy(Enum):
    """Model warm-up strategies"""
    EAGER = "eager"           # Pre-load all models immediately
    LAZY = "lazy"             # Load models on first use
    PREDICTIVE = "predictive" # Load based on usage patterns
    SCHEDULED = "scheduled"   # Load at specific times


class ModelState(Enum):
    """Model loading states"""
    UNLOADED = "unloaded"
    LOADING = "loading"
    READY = "ready"
    ERROR = "error"
    COOLDOWN = "cooldown"     # Recently unloaded, cooling down


@dataclass
class ModelMetrics:
    """Performance metrics for individual models"""
    model_name: str
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    
    # Performance metrics
    avg_load_time_ms: float = 0.0
    avg_inference_time_ms: float = 0.0
    avg_tokens_per_second: float = 0.0
    
    # Usage patterns
    last_used: Optional[datetime] = None
    usage_frequency: float = 0.0  # Requests per hour
    peak_usage_hour: int = 12  # Hour with highest usage
    
    # Resource usage
    memory_usage_mb: float = 0.0
    gpu_usage_percent: float = 0.0
    
    # Cost metrics
    total_cost_usd: float = 0.0
    avg_cost_per_request: float = 0.0
    
    def success_rate(self) -> float:
        """Calculate success rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.successful_requests / self.total_requests) * 100
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_name': self.model_name,
            'total_requests': self.total_requests,
            'success_rate': self.success_rate(),
            'avg_load_time_ms': self.avg_load_time_ms,
            'avg_inference_time_ms': self.avg_inference_time_ms,
            'avg_tokens_per_second': self.avg_tokens_per_second,
            'last_used': self.last_used.isoformat() if self.last_used else None,
            'usage_frequency': self.usage_frequency,
            'memory_usage_mb': self.memory_usage_mb,
            'total_cost_usd': self.total_cost_usd
        }


@dataclass
class WarmupTask:
    """Model warm-up task"""
    model_name: str
    priority: int = 1  # 1=highest, 10=lowest
    scheduled_time: Optional[datetime] = None
    reason: str = "manual"
    max_attempts: int = 3
    attempt_count: int = 0
    
    def should_run_now(self) -> bool:
        """Check if task should run now"""
        if self.scheduled_time is None:
            return True
        return datetime.now() >= self.scheduled_time


@dataclass
class OptimizationConfig:
    """Configuration for model optimization"""
    # Warm-up settings
    warmup_strategy: WarmupStrategy = WarmupStrategy.PREDICTIVE
    max_concurrent_warmups: int = 2
    warmup_timeout_seconds: int = 60
    
    # Memory management
    max_loaded_models: int = 5
    memory_threshold_mb: float = 8000.0  # 8GB
    unload_unused_after_minutes: int = 30
    
    # Performance optimization
    enable_model_pooling: bool = True
    pool_size_per_model: int = 2
    enable_request_batching: bool = True
    max_batch_size: int = 10
    
    # Cost optimization
    prefer_cheaper_models: bool = False
    cost_weight_factor: float = 0.1  # How much to weight cost in selection
    
    # Monitoring
    enable_performance_tracking: bool = True
    metrics_retention_days: int = 30


class ModelPool:
    """Pool of pre-loaded model instances"""
    
    def __init__(self, model_name: str, pool_size: int = 2):
        self.model_name = model_name
        self.pool_size = pool_size
        self.available_instances: deque = deque()
        self.busy_instances: Set = set()
        self.total_instances = 0
        self.creation_lock = asyncio.Lock()
    
    async def get_instance(self):
        """Get an available model instance"""
        # Try to get from available pool
        if self.available_instances:
            instance = self.available_instances.popleft()
            self.busy_instances.add(instance)
            return instance
        
        # Create new instance if under limit
        async with self.creation_lock:
            if self.total_instances < self.pool_size:
                instance = await self._create_instance()
                if instance:
                    self.busy_instances.add(instance)
                    self.total_instances += 1
                    return instance
        
        # Wait for instance to become available
        return await self._wait_for_instance()
    
    async def return_instance(self, instance):
        """Return instance to pool"""
        if instance in self.busy_instances:
            self.busy_instances.remove(instance)
            self.available_instances.append(instance)
    
    async def _create_instance(self):
        """Create new model instance"""
        try:
            # This would be implemented by specific model clients
            # For now, return a placeholder
            logger.debug(f"Creating new instance for {self.model_name}")
            await asyncio.sleep(0.1)  # Simulate creation time
            return f"{self.model_name}_instance_{self.total_instances}"
        except Exception as e:
            logger.error(f"Failed to create instance for {self.model_name}: {e}")
            return None
    
    async def _wait_for_instance(self, timeout: float = 10.0):
        """Wait for an instance to become available"""
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.available_instances:
                instance = self.available_instances.popleft()
                self.busy_instances.add(instance)
                return instance
            await asyncio.sleep(0.1)
        
        raise TimeoutError(f"No instance available for {self.model_name} within {timeout}s")


class WarmupManager:
    """
    Manages model warm-up and pre-loading strategies
    
    Features:
    - Multiple warm-up strategies (eager, lazy, predictive, scheduled)
    - Intelligent model loading based on usage patterns
    - Memory management and resource optimization
    - Performance monitoring and cost tracking
    """
    
    def __init__(self, config: OptimizationConfig = None):
        self.config = config or OptimizationConfig()
        
        # Model state tracking
        self.model_states: Dict[str, ModelState] = {}
        self.model_metrics: Dict[str, ModelMetrics] = {}
        self.model_pools: Dict[str, ModelPool] = {}
        
        # Warm-up management
        self.warmup_queue: asyncio.Queue = asyncio.Queue()
        self.warmup_workers: List[asyncio.Task] = []
        self.active_warmups: Dict[str, WarmupTask] = {}
        
        # Usage pattern learning
        self.usage_history = deque(maxlen=10000)
        self.hourly_usage_patterns = defaultdict(lambda: defaultdict(int))
        
        # Resource monitoring
        self.total_memory_usage_mb = 0.0
        self.memory_check_interval = 60  # seconds
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._cleanup_task: Optional[asyncio.Task] = None
        
        logger.info(f"WarmupManager initialized with strategy: {self.config.warmup_strategy.value}")
    
    async def start(self):
        """Start the warm-up manager"""
        # Start warm-up workers
        for i in range(self.config.max_concurrent_warmups):
            worker = asyncio.create_task(self._warmup_worker(f"warmup-{i}"))
            self.warmup_workers.append(worker)
        
        # Start background tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        # Apply initial warm-up strategy
        await self._apply_warmup_strategy()
        
        logger.info("WarmupManager started successfully")
    
    async def stop(self):
        """Stop the warm-up manager"""
        # Cancel workers
        for worker in self.warmup_workers:
            worker.cancel()
        await asyncio.gather(*self.warmup_workers, return_exceptions=True)
        self.warmup_workers.clear()
        
        # Cancel background tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        logger.info("WarmupManager stopped")
    
    async def warmup_model(self, model_name: str, priority: int = 1, reason: str = "manual") -> bool:
        """
        Request model warm-up
        
        Args:
            model_name: Name of model to warm up
            priority: Priority level (1=highest)
            reason: Reason for warm-up
            
        Returns:
            True if warm-up was queued successfully
        """
        try:
            # Check if already loaded
            if self.model_states.get(model_name) == ModelState.READY:
                logger.debug(f"Model {model_name} already loaded")
                return True
            
            # Check if already queued
            if model_name in self.active_warmups:
                logger.debug(f"Model {model_name} already queued for warm-up")
                return True
            
            # Create warm-up task
            task = WarmupTask(
                model_name=model_name,
                priority=priority,
                reason=reason
            )
            
            # Queue task
            await self.warmup_queue.put(task)
            self.active_warmups[model_name] = task
            
            logger.info(f"Queued warm-up for {model_name} (priority: {priority}, reason: {reason})")
            return True
            
        except Exception as e:
            logger.error(f"Failed to queue warm-up for {model_name}: {e}")
            return False
    
    async def ensure_model_ready(self, model_name: str, timeout: float = 30.0) -> bool:
        """
        Ensure model is ready, warming up if necessary
        
        Args:
            model_name: Name of model
            timeout: Maximum time to wait
            
        Returns:
            True if model is ready
        """
        # Check if already ready
        if self.model_states.get(model_name) == ModelState.READY:
            return True
        
        # Start warm-up if needed
        await self.warmup_model(model_name, priority=1, reason="on_demand")
        
        # Wait for model to be ready
        start_time = time.time()
        while time.time() - start_time < timeout:
            if self.model_states.get(model_name) == ModelState.READY:
                return True
            await asyncio.sleep(0.5)
        
        logger.warning(f"Model {model_name} not ready within {timeout}s")
        return False
    
    async def get_model_instance(self, model_name: str):
        """Get a model instance from the pool"""
        if not self.config.enable_model_pooling:
            # Direct model access without pooling
            return model_name
        
        # Ensure model is ready
        if not await self.ensure_model_ready(model_name):
            raise RuntimeError(f"Model {model_name} not available")
        
        # Get from pool
        if model_name not in self.model_pools:
            self.model_pools[model_name] = ModelPool(model_name, self.config.pool_size_per_model)
        
        return await self.model_pools[model_name].get_instance()
    
    async def return_model_instance(self, model_name: str, instance):
        """Return model instance to pool"""
        if self.config.enable_model_pooling and model_name in self.model_pools:
            await self.model_pools[model_name].return_instance(instance)
    
    def record_model_usage(self, model_name: str, success: bool, 
                          inference_time_ms: float = 0.0, 
                          tokens_generated: int = 0,
                          cost_usd: float = 0.0):
        """Record model usage for optimization"""
        # Initialize metrics if needed
        if model_name not in self.model_metrics:
            self.model_metrics[model_name] = ModelMetrics(model_name=model_name)
        
        metrics = self.model_metrics[model_name]
        
        # Update basic counters
        metrics.total_requests += 1
        if success:
            metrics.successful_requests += 1
        else:
            metrics.failed_requests += 1
        
        # Update timing metrics
        if inference_time_ms > 0:
            metrics.avg_inference_time_ms = (
                (metrics.avg_inference_time_ms * (metrics.total_requests - 1) + inference_time_ms) /
                metrics.total_requests
            )
        
        # Update token metrics
        if tokens_generated > 0 and inference_time_ms > 0:
            tokens_per_second = (tokens_generated / inference_time_ms) * 1000
            metrics.avg_tokens_per_second = (
                (metrics.avg_tokens_per_second * (metrics.successful_requests - 1) + tokens_per_second) /
                metrics.successful_requests
            )
        
        # Update cost metrics
        if cost_usd > 0:
            metrics.total_cost_usd += cost_usd
            metrics.avg_cost_per_request = metrics.total_cost_usd / metrics.total_requests
        
        # Update usage patterns
        metrics.last_used = datetime.now()
        current_hour = datetime.now().hour
        self.hourly_usage_patterns[model_name][current_hour] += 1
        
        # Calculate usage frequency (requests per hour over last 24 hours)
        recent_usage = sum(1 for entry in self.usage_history 
                          if entry.get('model') == model_name and 
                          datetime.now() - datetime.fromisoformat(entry['timestamp']) <= timedelta(hours=24))
        metrics.usage_frequency = recent_usage / 24.0
        
        # Record in usage history
        self.usage_history.append({
            'timestamp': datetime.now().isoformat(),
            'model': model_name,
            'success': success,
            'inference_time_ms': inference_time_ms,
            'tokens_generated': tokens_generated,
            'cost_usd': cost_usd
        })
        
        logger.debug(f"Recorded usage for {model_name}: success={success}, time={inference_time_ms:.2f}ms")
    
    async def _apply_warmup_strategy(self):
        """Apply the configured warm-up strategy"""
        if self.config.warmup_strategy == WarmupStrategy.EAGER:
            await self._eager_warmup()
        elif self.config.warmup_strategy == WarmupStrategy.PREDICTIVE:
            await self._predictive_warmup()
        elif self.config.warmup_strategy == WarmupStrategy.SCHEDULED:
            await self._scheduled_warmup()
        # LAZY strategy doesn't pre-load anything
    
    async def _eager_warmup(self):
        """Eagerly warm up all known models"""
        # This would be configured with your actual model list
        common_models = [
            "gpt-3.5-turbo", "gpt-4-turbo", "claude-3-sonnet", 
            "claude-3-opus", "claude-3-haiku"
        ]
        
        for model in common_models:
            await self.warmup_model(model, priority=5, reason="eager_strategy")
    
    async def _predictive_warmup(self):
        """Warm up models based on usage patterns"""
        current_hour = datetime.now().hour
        
        # Analyze usage patterns
        for model_name, hourly_usage in self.hourly_usage_patterns.items():
            # Check if this model is typically used at this hour
            usage_at_hour = hourly_usage.get(current_hour, 0)
            total_usage = sum(hourly_usage.values())
            
            if total_usage > 10 and usage_at_hour / total_usage > 0.1:  # >10% of usage at this hour
                priority = min(10, max(1, 11 - usage_at_hour))  # Higher usage = higher priority
                await self.warmup_model(model_name, priority=priority, reason="predictive")
    
    async def _scheduled_warmup(self):
        """Scheduled warm-up based on time"""
        current_hour = datetime.now().hour
        
        # Business hours strategy
        if 8 <= current_hour <= 18:  # Business hours
            business_models = ["gpt-4-turbo", "claude-3-sonnet"]
            for model in business_models:
                await self.warmup_model(model, priority=2, reason="business_hours")
        else:  # Off hours - use faster models
            off_hours_models = ["gpt-3.5-turbo", "claude-3-haiku"]
            for model in off_hours_models:
                await self.warmup_model(model, priority=3, reason="off_hours")
    
    async def _warmup_worker(self, worker_id: str):
        """Background worker for model warm-up"""
        logger.info(f"Warmup worker {worker_id} started")
        
        while True:
            try:
                # Get next warm-up task
                task = await self.warmup_queue.get()
                
                if task.model_name not in self.active_warmups:
                    continue  # Task was cancelled
                
                logger.info(f"Worker {worker_id} warming up {task.model_name}")
                
                # Perform warm-up
                success = await self._perform_warmup(task)
                
                if success:
                    logger.info(f"Successfully warmed up {task.model_name}")
                else:
                    logger.warning(f"Failed to warm up {task.model_name}")
                    
                    # Retry if attempts remaining
                    task.attempt_count += 1
                    if task.attempt_count < task.max_attempts:
                        logger.info(f"Retrying warm-up for {task.model_name} (attempt {task.attempt_count + 1})")
                        await asyncio.sleep(5)  # Brief delay before retry
                        await self.warmup_queue.put(task)
                        continue
                
                # Clean up
                if task.model_name in self.active_warmups:
                    del self.active_warmups[task.model_name]
                
                self.warmup_queue.task_done()
                
            except asyncio.CancelledError:
                logger.info(f"Warmup worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in warmup worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    async def _perform_warmup(self, task: WarmupTask) -> bool:
        """Perform actual model warm-up"""
        model_name = task.model_name
        start_time = time.time()
        
        try:
            # Set state to loading
            self.model_states[model_name] = ModelState.LOADING
            
            # Initialize metrics if needed
            if model_name not in self.model_metrics:
                self.model_metrics[model_name] = ModelMetrics(model_name=model_name)
            
            # Simulate model loading (replace with actual model client loading)
            logger.debug(f"Loading model {model_name}...")
            await asyncio.sleep(2)  # Simulate loading time
            
            # Update load time metrics
            load_time_ms = (time.time() - start_time) * 1000
            metrics = self.model_metrics[model_name]
            
            if metrics.avg_load_time_ms == 0:
                metrics.avg_load_time_ms = load_time_ms
            else:
                # Running average
                metrics.avg_load_time_ms = (metrics.avg_load_time_ms + load_time_ms) / 2
            
            # Set state to ready
            self.model_states[model_name] = ModelState.READY
            
            # Create model pool if enabled
            if self.config.enable_model_pooling:
                self.model_pools[model_name] = ModelPool(model_name, self.config.pool_size_per_model)
            
            logger.debug(f"Model {model_name} loaded in {load_time_ms:.2f}ms")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            self.model_states[model_name] = ModelState.ERROR
            return False
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(self.memory_check_interval)
                
                # Monitor memory usage
                await self._check_memory_usage()
                
                # Update usage patterns
                await self._update_usage_patterns()
                
                # Apply optimization strategies
                if self.config.warmup_strategy == WarmupStrategy.PREDICTIVE:
                    await self._predictive_warmup()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(300)  # Run every 5 minutes
                
                # Unload unused models
                await self._unload_unused_models()
                
                # Clean up old metrics
                await self._cleanup_old_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    async def _check_memory_usage(self):
        """Check and manage memory usage"""
        # This would integrate with actual memory monitoring
        # For now, simulate based on loaded models
        
        loaded_models = [name for name, state in self.model_states.items() 
                        if state == ModelState.READY]
        
        # Estimate memory usage (simplified)
        estimated_memory = len(loaded_models) * 1500  # ~1.5GB per model
        self.total_memory_usage_mb = estimated_memory
        
        if estimated_memory > self.config.memory_threshold_mb:
            logger.warning(f"Memory usage ({estimated_memory:.0f}MB) exceeds threshold")
            await self._free_memory()
    
    async def _free_memory(self):
        """Free memory by unloading least used models"""
        # Find least recently used models
        model_usage = [
            (name, metrics.last_used or datetime.min, metrics.usage_frequency)
            for name, metrics in self.model_metrics.items()
            if self.model_states.get(name) == ModelState.READY
        ]
        
        # Sort by usage frequency and last used time
        model_usage.sort(key=lambda x: (x[2], x[1]))  # Lowest frequency first
        
        # Unload least used models until under threshold
        models_to_unload = min(2, len(model_usage))  # Unload at most 2 models
        
        for i in range(models_to_unload):
            model_name = model_usage[i][0]
            await self._unload_model(model_name, reason="memory_pressure")
    
    async def _unload_model(self, model_name: str, reason: str = "cleanup"):
        """Unload a model to free resources"""
        try:
            logger.info(f"Unloading model {model_name} ({reason})")
            
            # Remove from pools
            if model_name in self.model_pools:
                del self.model_pools[model_name]
            
            # Update state
            self.model_states[model_name] = ModelState.COOLDOWN
            
            # Schedule for potential reload later
            await asyncio.sleep(1)  # Brief delay
            self.model_states[model_name] = ModelState.UNLOADED
            
        except Exception as e:
            logger.error(f"Error unloading model {model_name}: {e}")
    
    async def _unload_unused_models(self):
        """Unload models that haven't been used recently"""
        cutoff_time = datetime.now() - timedelta(minutes=self.config.unload_unused_after_minutes)
        
        for model_name, metrics in self.model_metrics.items():
            if (self.model_states.get(model_name) == ModelState.READY and
                metrics.last_used and metrics.last_used < cutoff_time):
                
                await self._unload_model(model_name, reason="unused")
    
    async def _update_usage_patterns(self):
        """Update usage pattern analysis"""
        # Update peak usage hours for each model
        for model_name, hourly_usage in self.hourly_usage_patterns.items():
            if hourly_usage:
                peak_hour = max(hourly_usage.items(), key=lambda x: x[1])[0]
                if model_name in self.model_metrics:
                    self.model_metrics[model_name].peak_usage_hour = peak_hour
    
    async def _cleanup_old_metrics(self):
        """Clean up old usage history"""
        cutoff_time = datetime.now() - timedelta(days=self.config.metrics_retention_days)
        
        # Clean usage history
        cutoff_timestamp = cutoff_time.isoformat()
        self.usage_history = deque(
            [entry for entry in self.usage_history 
             if entry['timestamp'] >= cutoff_timestamp],
            maxlen=self.usage_history.maxlen
        )
    
    def get_model_metrics(self) -> Dict[str, ModelMetrics]:
        """Get current model metrics"""
        return dict(self.model_metrics)
    
    def get_optimization_insights(self) -> Dict[str, Any]:
        """Get optimization insights and recommendations"""
        insights = {
            'current_state': {
                'loaded_models': len([s for s in self.model_states.values() if s == ModelState.READY]),
                'total_memory_mb': self.total_memory_usage_mb,
                'active_warmups': len(self.active_warmups)
            },
            'recommendations': [],
            'model_performance': {}
        }
        
        # Analyze model performance
        for model_name, metrics in self.model_metrics.items():
            performance = {
                'efficiency_score': 0.0,
                'cost_effectiveness': 0.0,
                'reliability': metrics.success_rate(),
                'usage_trend': 'stable'
            }
            
            # Calculate efficiency score
            if metrics.avg_inference_time_ms > 0 and metrics.avg_tokens_per_second > 0:
                performance['efficiency_score'] = min(100, metrics.avg_tokens_per_second * 10)
            
            # Calculate cost effectiveness
            if metrics.avg_cost_per_request > 0:
                performance['cost_effectiveness'] = max(0, 100 - (metrics.avg_cost_per_request * 1000))
            
            insights['model_performance'][model_name] = performance
            
            # Generate recommendations
            if metrics.success_rate() < 90:
                insights['recommendations'].append({
                    'type': 'reliability',
                    'model': model_name,
                    'message': f"Model {model_name} has low success rate ({metrics.success_rate():.1f}%)"
                })
            
            if metrics.avg_inference_time_ms > 10000:  # >10 seconds
                insights['recommendations'].append({
                    'type': 'performance',
                    'model': model_name,
                    'message': f"Model {model_name} has high latency ({metrics.avg_inference_time_ms:.0f}ms)"
                })
        
        return insights
    
    def export_configuration(self) -> Dict[str, Any]:
        """Export current configuration and state"""
        return {
            'config': {
                'warmup_strategy': self.config.warmup_strategy.value,
                'max_concurrent_warmups': self.config.max_concurrent_warmups,
                'max_loaded_models': self.config.max_loaded_models,
                'memory_threshold_mb': self.config.memory_threshold_mb,
                'enable_model_pooling': self.config.enable_model_pooling
            },
            'state': {
                'model_states': {k: v.value for k, v in self.model_states.items()},
                'total_memory_usage_mb': self.total_memory_usage_mb
            },
            'metrics': {name: metrics.to_dict() for name, metrics in self.model_metrics.items()}
        }