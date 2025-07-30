"""
Batch Processing System for Multiple Users

Advanced batch processing system that efficiently handles multiple user requests
through intelligent batching, queuing, and resource optimization strategies.
"""

import asyncio
import logging
import time
import uuid
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Callable, Awaitable, Set
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor

from ..chat.dynamic_model_selector import PromptType, ModelSelection, DynamicModelSelector
from .smart_cache import SmartModelCache
from .performance_monitor import PerformanceMonitor

logger = logging.getLogger(__name__)


class BatchStrategy(Enum):
    """Batch processing strategies"""
    TIME_BASED = "time_based"           # Batch by time window
    SIZE_BASED = "size_based"           # Batch by request count
    ADAPTIVE = "adaptive"               # Dynamic based on load
    PRIORITY_BASED = "priority_based"   # Batch by priority groups


class RequestPriority(Enum):
    """Request priority levels"""
    CRITICAL = 1    # Crisis/emergency requests
    HIGH = 2        # Therapeutic content
    MEDIUM = 3      # Information seeking
    LOW = 4         # General wellness


@dataclass
class BatchRequest:
    """Individual request in a batch"""
    request_id: str
    user_id: str
    prompt: str
    prompt_type: Optional[PromptType] = None
    priority: RequestPriority = RequestPriority.MEDIUM
    timestamp: datetime = field(default_factory=datetime.now)
    timeout_seconds: int = 30
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    # Callback for result delivery
    result_callback: Optional[Callable[['BatchResult'], Awaitable[None]]] = None
    
    def __post_init__(self):
        if not self.request_id:
            self.request_id = str(uuid.uuid4())


@dataclass
class BatchResult:
    """Result for a batch request"""
    request_id: str
    user_id: str
    success: bool
    model_selection: Optional[ModelSelection] = None
    error_message: Optional[str] = None
    processing_time_ms: float = 0.0
    batch_id: Optional[str] = None
    cache_hit: bool = False


@dataclass
class ProcessingBatch:
    """A batch of requests being processed together"""
    batch_id: str
    requests: List[BatchRequest]
    strategy: BatchStrategy
    created_at: datetime
    priority_level: RequestPriority
    estimated_processing_time_ms: float = 0.0
    
    def __post_init__(self):
        # Calculate average priority
        if self.requests:
            avg_priority = sum(req.priority.value for req in self.requests) / len(self.requests)
            self.priority_level = RequestPriority(round(avg_priority))


@dataclass
class BatchingConfig:
    """Configuration for batch processing"""
    # Batching strategy
    strategy: BatchStrategy = BatchStrategy.ADAPTIVE
    
    # Size-based batching
    max_batch_size: int = 20
    min_batch_size: int = 2
    
    # Time-based batching
    max_wait_time_ms: int = 1000  # 1 second
    min_wait_time_ms: int = 100   # 100ms
    
    # Resource limits
    max_concurrent_batches: int = 5
    max_requests_per_second: float = 100.0
    
    # Priority handling
    priority_queue_enabled: bool = True
    critical_bypass_batching: bool = True  # Process critical requests immediately
    
    # Performance optimization
    enable_cache_batching: bool = True     # Group cache lookups
    enable_model_batching: bool = True     # Use same model for similar requests
    prefetch_similar_requests: bool = True
    
    # Timeout settings
    default_request_timeout_seconds: int = 30
    batch_processing_timeout_seconds: int = 60


@dataclass
class BatchingMetrics:
    """Metrics for batch processing performance"""
    total_requests: int = 0
    total_batches: int = 0
    avg_batch_size: float = 0.0
    avg_batch_processing_time_ms: float = 0.0
    
    # Throughput metrics
    requests_per_second: float = 0.0
    peak_requests_per_second: float = 0.0
    
    # Efficiency metrics
    cache_hit_rate: float = 0.0
    model_reuse_rate: float = 0.0
    
    # Priority distribution
    priority_distribution: Dict[RequestPriority, int] = field(default_factory=dict)
    
    # Queue metrics
    avg_queue_wait_time_ms: float = 0.0
    max_queue_size: int = 0
    current_queue_size: int = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'total_requests': self.total_requests,
            'total_batches': self.total_batches,
            'avg_batch_size': self.avg_batch_size,
            'avg_batch_processing_time_ms': self.avg_batch_processing_time_ms,
            'requests_per_second': self.requests_per_second,
            'cache_hit_rate': self.cache_hit_rate,
            'model_reuse_rate': self.model_reuse_rate,
            'avg_queue_wait_time_ms': self.avg_queue_wait_time_ms,
            'current_queue_size': self.current_queue_size,
            'priority_distribution': {p.name: count for p, count in self.priority_distribution.items()}
        }


class RequestQueue:
    """Priority-aware request queue"""
    
    def __init__(self, enable_priority: bool = True):
        self.enable_priority = enable_priority
        
        # Separate queues for each priority level
        self.priority_queues: Dict[RequestPriority, asyncio.Queue] = {
            priority: asyncio.Queue() for priority in RequestPriority
        }
        
        # Combined queue for non-priority mode
        self.combined_queue: asyncio.Queue = asyncio.Queue()
        
        self.total_size = 0
        self._lock = asyncio.Lock()
    
    async def put(self, request: BatchRequest):
        """Add request to appropriate queue"""
        async with self._lock:
            if self.enable_priority:
                await self.priority_queues[request.priority].put(request)
            else:
                await self.combined_queue.put(request)
            
            self.total_size += 1
    
    async def get_batch(self, max_size: int, max_wait_ms: int) -> List[BatchRequest]:
        """Get a batch of requests, respecting priority"""
        batch = []
        start_time = time.time()
        max_wait_seconds = max_wait_ms / 1000.0
        
        if not self.enable_priority:
            # Simple FIFO batching
            while len(batch) < max_size and (time.time() - start_time) < max_wait_seconds:
                try:
                    request = await asyncio.wait_for(
                        self.combined_queue.get(), 
                        timeout=max(0.1, max_wait_seconds - (time.time() - start_time))
                    )
                    batch.append(request)
                    async with self._lock:
                        self.total_size -= 1
                except asyncio.TimeoutError:
                    break
            
            return batch
        
        # Priority-based batching
        for priority in RequestPriority:
            queue = self.priority_queues[priority]
            
            # For critical requests, create immediate batch
            if priority == RequestPriority.CRITICAL and not queue.empty():
                while not queue.empty() and len(batch) < max_size:
                    try:
                        request = queue.get_nowait()
                        batch.append(request)
                        async with self._lock:
                            self.total_size -= 1
                    except asyncio.QueueEmpty:
                        break
                
                if batch:
                    return batch  # Immediate processing for critical
            
            # For other priorities, collect up to remaining batch size
            remaining_size = max_size - len(batch)
            remaining_time = max_wait_seconds - (time.time() - start_time)
            
            if remaining_size > 0 and remaining_time > 0:
                while len(batch) < max_size and remaining_time > 0:
                    try:
                        request = await asyncio.wait_for(queue.get(), timeout=remaining_time)
                        batch.append(request)
                        async with self._lock:
                            self.total_size -= 1
                        remaining_time = max_wait_seconds - (time.time() - start_time)
                    except asyncio.TimeoutError:
                        break
        
        return batch
    
    def size(self) -> int:
        """Get total queue size"""
        return self.total_size
    
    def size_by_priority(self) -> Dict[RequestPriority, int]:
        """Get queue size by priority"""
        if not self.enable_priority:
            return {RequestPriority.MEDIUM: self.combined_queue.qsize()}
        
        return {priority: queue.qsize() for priority, queue in self.priority_queues.items()}


class BatchProcessor:
    """
    Advanced batch processing system for multiple user requests
    
    Features:
    - Multiple batching strategies (time, size, adaptive, priority-based)
    - Priority-aware request queuing with immediate critical processing
    - Intelligent request grouping for cache and model efficiency
    - Resource optimization and load balancing
    - Comprehensive performance monitoring
    - Automatic scaling based on load patterns
    """
    
    def __init__(self, 
                 model_selector: DynamicModelSelector,
                 cache: SmartModelCache,
                 performance_monitor: PerformanceMonitor,
                 config: BatchingConfig = None):
        
        self.model_selector = model_selector
        self.cache = cache
        self.performance_monitor = performance_monitor
        self.config = config or BatchingConfig()
        
        # Request queuing
        self.request_queue = RequestQueue(self.config.priority_queue_enabled)
        
        # Batch processing
        self.active_batches: Dict[str, ProcessingBatch] = {}
        self.batch_workers: List[asyncio.Task] = []
        self.batch_counter = 0
        
        # Metrics and monitoring
        self.metrics = BatchingMetrics()
        self.processing_history = deque(maxlen=1000)
        
        # Adaptive batching
        self.load_monitor = deque(maxlen=60)  # Last 60 seconds
        self.adaptive_config = {
            'current_batch_size': self.config.min_batch_size,
            'current_wait_time': self.config.min_wait_time_ms,
            'last_adjustment': datetime.now()
        }
        
        # Background tasks
        self._monitoring_task: Optional[asyncio.Task] = None
        self._adaptive_task: Optional[asyncio.Task] = None
        
        logger.info(f"BatchProcessor initialized with strategy: {self.config.strategy.value}")
    
    async def start(self):
        """Start the batch processor"""
        # Start batch processing workers
        for i in range(self.config.max_concurrent_batches):
            worker = asyncio.create_task(self._batch_worker(f"batch-worker-{i}"))
            self.batch_workers.append(worker)
        
        # Start monitoring tasks
        self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        
        if self.config.strategy == BatchStrategy.ADAPTIVE:
            self._adaptive_task = asyncio.create_task(self._adaptive_loop())
        
        logger.info(f"Started {len(self.batch_workers)} batch processing workers")
    
    async def stop(self):
        """Stop the batch processor"""
        # Cancel workers
        for worker in self.batch_workers:
            worker.cancel()
        await asyncio.gather(*self.batch_workers, return_exceptions=True)
        self.batch_workers.clear()
        
        # Cancel monitoring tasks
        if self._monitoring_task:
            self._monitoring_task.cancel()
        if self._adaptive_task:
            self._adaptive_task.cancel()
        
        logger.info("BatchProcessor stopped")
    
    async def submit_request(self, 
                           user_id: str, 
                           prompt: str, 
                           prompt_type: Optional[PromptType] = None,
                           priority: RequestPriority = RequestPriority.MEDIUM,
                           timeout_seconds: int = None,
                           result_callback: Optional[Callable[[BatchResult], Awaitable[None]]] = None,
                           metadata: Dict[str, Any] = None) -> str:
        """
        Submit a request for batch processing
        
        Args:
            user_id: Unique user identifier
            prompt: User prompt
            prompt_type: Optional prompt classification
            priority: Request priority level
            timeout_seconds: Request timeout
            result_callback: Callback for result delivery
            metadata: Additional request metadata
            
        Returns:
            Request ID for tracking
        """
        request = BatchRequest(
            request_id=str(uuid.uuid4()),
            user_id=user_id,
            prompt=prompt,
            prompt_type=prompt_type,
            priority=priority,
            timeout_seconds=timeout_seconds or self.config.default_request_timeout_seconds,
            result_callback=result_callback,
            metadata=metadata or {}
        )
        
        # Update metrics
        self.metrics.total_requests += 1
        self.metrics.priority_distribution[priority] = self.metrics.priority_distribution.get(priority, 0) + 1
        
        # Queue request
        await self.request_queue.put(request)
        
        # Update queue metrics
        self.metrics.current_queue_size = self.request_queue.size()
        self.metrics.max_queue_size = max(self.metrics.max_queue_size, self.metrics.current_queue_size)
        
        logger.debug(f"Submitted request {request.request_id} for user {user_id} (priority: {priority.name})")
        return request.request_id
    
    async def _batch_worker(self, worker_id: str):
        """Background worker for batch processing"""
        logger.info(f"Batch worker {worker_id} started")
        
        while True:
            try:
                # Get batch configuration
                batch_size, wait_time = self._get_current_batch_config()
                
                # Get batch of requests
                batch_requests = await self.request_queue.get_batch(batch_size, wait_time)
                
                if not batch_requests:
                    await asyncio.sleep(0.1)  # Brief pause if no requests
                    continue
                
                # Create and process batch
                batch = ProcessingBatch(
                    batch_id=f"batch_{self.batch_counter}_{int(time.time())}",
                    requests=batch_requests,
                    strategy=self.config.strategy,
                    created_at=datetime.now()
                )
                
                self.batch_counter += 1
                self.active_batches[batch.batch_id] = batch
                
                logger.debug(f"Worker {worker_id} processing batch {batch.batch_id} with {len(batch_requests)} requests")
                
                # Process batch
                await self._process_batch(batch)
                
                # Clean up
                if batch.batch_id in self.active_batches:
                    del self.active_batches[batch.batch_id]
                
            except asyncio.CancelledError:
                logger.info(f"Batch worker {worker_id} cancelled")
                break
            except Exception as e:
                logger.error(f"Error in batch worker {worker_id}: {e}")
                await asyncio.sleep(1)
    
    def _get_current_batch_config(self) -> tuple[int, int]:
        """Get current batch size and wait time based on strategy"""
        if self.config.strategy == BatchStrategy.SIZE_BASED:
            return self.config.max_batch_size, self.config.max_wait_time_ms
        
        elif self.config.strategy == BatchStrategy.TIME_BASED:
            return self.config.max_batch_size, self.config.max_wait_time_ms
        
        elif self.config.strategy == BatchStrategy.ADAPTIVE:
            return (
                self.adaptive_config['current_batch_size'],
                self.adaptive_config['current_wait_time']
            )
        
        elif self.config.strategy == BatchStrategy.PRIORITY_BASED:
            # Smaller batches for faster priority processing
            return min(10, self.config.max_batch_size), self.config.min_wait_time_ms
        
        else:
            return self.config.max_batch_size, self.config.max_wait_time_ms
    
    async def _process_batch(self, batch: ProcessingBatch):
        """Process a batch of requests"""
        start_time = time.time()
        results = []
        
        try:
            # Group requests for optimization
            request_groups = self._group_requests_for_optimization(batch.requests)
            
            # Process each group
            for group_name, group_requests in request_groups.items():
                group_results = await self._process_request_group(group_requests, batch.batch_id)
                results.extend(group_results)
            
            # Update metrics
            processing_time_ms = (time.time() - start_time) * 1000
            self.metrics.total_batches += 1
            self.metrics.avg_batch_size = (
                (self.metrics.avg_batch_size * (self.metrics.total_batches - 1) + len(batch.requests)) /
                self.metrics.total_batches
            )
            self.metrics.avg_batch_processing_time_ms = (
                (self.metrics.avg_batch_processing_time_ms * (self.metrics.total_batches - 1) + processing_time_ms) /
                self.metrics.total_batches
            )
            
            # Record processing history
            self.processing_history.append({
                'timestamp': datetime.now().isoformat(),
                'batch_id': batch.batch_id,
                'request_count': len(batch.requests),
                'processing_time_ms': processing_time_ms,
                'success_rate': sum(1 for r in results if r.success) / len(results) if results else 0
            })
            
            logger.debug(f"Processed batch {batch.batch_id} in {processing_time_ms:.2f}ms")
            
        except Exception as e:
            logger.error(f"Error processing batch {batch.batch_id}: {e}")
            
            # Create error results for all requests
            for request in batch.requests:
                results.append(BatchResult(
                    request_id=request.request_id,
                    user_id=request.user_id,
                    success=False,
                    error_message=f"Batch processing error: {str(e)}",
                    batch_id=batch.batch_id
                ))
        
        # Deliver results
        await self._deliver_results(results)
    
    def _group_requests_for_optimization(self, requests: List[BatchRequest]) -> Dict[str, List[BatchRequest]]:
        """Group requests for optimal processing"""
        groups = defaultdict(list)
        
        if not self.config.enable_model_batching:
            # No grouping - process individually
            for i, request in enumerate(requests):
                groups[f"individual_{i}"] = [request]
            return dict(groups)
        
        # Group by prompt type for model efficiency
        for request in requests:
            prompt_type = request.prompt_type or PromptType.GENERAL_WELLNESS
            group_key = f"type_{prompt_type.value}"
            
            # For critical requests, create separate groups for immediate processing
            if request.priority == RequestPriority.CRITICAL:
                group_key = f"critical_{request.request_id}"
            
            groups[group_key].append(request)
        
        return dict(groups)
    
    async def _process_request_group(self, requests: List[BatchRequest], batch_id: str) -> List[BatchResult]:
        """Process a group of similar requests"""
        results = []
        
        # Check cache for all requests first
        if self.config.enable_cache_batching:
            cache_results = await self._batch_cache_lookup(requests)
            
            # Separate cached and non-cached requests
            cached_requests = []
            uncached_requests = []
            
            for request, cache_result in zip(requests, cache_results):
                if cache_result:
                    results.append(BatchResult(
                        request_id=request.request_id,
                        user_id=request.user_id,
                        success=True,
                        model_selection=self._cached_to_model_selection(cache_result),
                        processing_time_ms=5.0,  # Assume fast cache lookup
                        batch_id=batch_id,
                        cache_hit=True
                    ))
                    cached_requests.append(request)
                else:
                    uncached_requests.append(request)
            
            # Update cache hit rate
            if requests:
                hit_rate = len(cached_requests) / len(requests)
                self.metrics.cache_hit_rate = (
                    (self.metrics.cache_hit_rate * (self.metrics.total_batches - 1) + hit_rate) /
                    self.metrics.total_batches
                )
            
            # Process uncached requests
            requests = uncached_requests
        
        # Process remaining requests through model selection
        for request in requests:
            result = await self._process_single_request(request, batch_id)
            results.append(result)
        
        return results
    
    async def _batch_cache_lookup(self, requests: List[BatchRequest]) -> List[Optional[Any]]:
        """Perform batch cache lookups"""
        tasks = []
        for request in requests:
            task = self.cache.get_cached_selection(request.prompt, request.prompt_type)
            tasks.append(task)
        
        return await asyncio.gather(*tasks, return_exceptions=True)
    
    def _cached_to_model_selection(self, cached_result) -> ModelSelection:
        """Convert cached result to ModelSelection"""
        # This would be implemented based on your cache structure
        return ModelSelection(
            selected_model=cached_result.selected_model,
            confidence_score=cached_result.confidence_score,
            prompt_classification=cached_result.prompt_classification,
            reasoning=cached_result.reasoning,
            evaluation_time_ms=1.0,
            model_scores={}
        )
    
    async def _process_single_request(self, request: BatchRequest, batch_id: str) -> BatchResult:
        """Process a single request"""
        start_time = time.time()
        
        try:
            # Perform model selection
            selection = await asyncio.wait_for(
                self.model_selector.select_model(request.prompt, request.prompt_type),
                timeout=request.timeout_seconds
            )
            
            processing_time_ms = (time.time() - start_time) * 1000
            
            # Store in cache for future use
            await self.cache.store_selection(
                request.prompt, 
                selection, 
                success=True,
                response_time_ms=processing_time_ms
            )
            
            return BatchResult(
                request_id=request.request_id,
                user_id=request.user_id,
                success=True,
                model_selection=selection,
                processing_time_ms=processing_time_ms,
                batch_id=batch_id,
                cache_hit=False
            )
            
        except asyncio.TimeoutError:
            return BatchResult(
                request_id=request.request_id,
                user_id=request.user_id,
                success=False,
                error_message="Request timeout",
                processing_time_ms=(time.time() - start_time) * 1000,
                batch_id=batch_id
            )
            
        except Exception as e:
            return BatchResult(
                request_id=request.request_id,
                user_id=request.user_id,
                success=False,
                error_message=str(e),
                processing_time_ms=(time.time() - start_time) * 1000,
                batch_id=batch_id
            )
    
    async def _deliver_results(self, results: List[BatchResult]):
        """Deliver results to requesting clients"""
        for result in results:
            try:
                # Find original request for callback
                original_request = None
                for batch in self.active_batches.values():
                    for req in batch.requests:
                        if req.request_id == result.request_id:
                            original_request = req
                            break
                    if original_request:
                        break
                
                # Call result callback if provided
                if original_request and original_request.result_callback:
                    await original_request.result_callback(result)
                
                # Record performance metrics
                if result.success and result.model_selection:
                    self.performance_monitor.record_cache_hit(
                        result.processing_time_ms,
                        result.model_selection.selected_model
                    )
                
            except Exception as e:
                logger.error(f"Error delivering result for {result.request_id}: {e}")
    
    async def _monitoring_loop(self):
        """Background monitoring and metrics collection"""
        while True:
            try:
                await asyncio.sleep(1)  # Check every second
                
                # Update load monitoring
                current_load = self.request_queue.size()
                self.load_monitor.append(current_load)
                
                # Update throughput metrics
                recent_requests = sum(1 for entry in self.processing_history 
                                    if datetime.now() - datetime.fromisoformat(entry['timestamp']) <= timedelta(seconds=60))
                self.metrics.requests_per_second = recent_requests / 60.0
                self.metrics.peak_requests_per_second = max(
                    self.metrics.peak_requests_per_second, 
                    self.metrics.requests_per_second
                )
                
                # Update queue wait time
                if self.processing_history:
                    recent_processing_times = [
                        entry['processing_time_ms'] for entry in self.processing_history
                        if datetime.now() - datetime.fromisoformat(entry['timestamp']) <= timedelta(minutes=5)
                    ]
                    
                    if recent_processing_times:
                        self.metrics.avg_queue_wait_time_ms = sum(recent_processing_times) / len(recent_processing_times)
                
                # Update queue size
                self.metrics.current_queue_size = self.request_queue.size()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _adaptive_loop(self):
        """Adaptive batching optimization loop"""
        while True:
            try:
                await asyncio.sleep(10)  # Adjust every 10 seconds
                
                # Analyze recent performance
                if len(self.processing_history) < 5:
                    continue  # Need more data
                
                recent_entries = [
                    entry for entry in self.processing_history
                    if datetime.now() - datetime.fromisoformat(entry['timestamp']) <= timedelta(minutes=2)
                ]
                
                if not recent_entries:
                    continue
                
                # Calculate performance metrics
                avg_processing_time = sum(entry['processing_time_ms'] for entry in recent_entries) / len(recent_entries)
                avg_batch_size = sum(entry['request_count'] for entry in recent_entries) / len(recent_entries)
                avg_success_rate = sum(entry['success_rate'] for entry in recent_entries) / len(recent_entries)
                
                # Adjust batch configuration
                await self._adjust_adaptive_config(avg_processing_time, avg_batch_size, avg_success_rate)
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in adaptive loop: {e}")
    
    async def _adjust_adaptive_config(self, avg_processing_time: float, avg_batch_size: float, avg_success_rate: float):
        """Adjust adaptive configuration based on performance"""
        current_load = sum(self.load_monitor) / len(self.load_monitor) if self.load_monitor else 0
        
        # Adjust batch size based on load and performance
        if current_load > 20 and avg_processing_time < 5000:  # High load, good performance
            # Increase batch size
            new_batch_size = min(
                self.config.max_batch_size,
                self.adaptive_config['current_batch_size'] + 2
            )
        elif current_load < 5 or avg_processing_time > 10000:  # Low load or poor performance
            # Decrease batch size
            new_batch_size = max(
                self.config.min_batch_size,
                self.adaptive_config['current_batch_size'] - 1
            )
        else:
            new_batch_size = self.adaptive_config['current_batch_size']
        
        # Adjust wait time based on load
        if current_load > 10:
            # Reduce wait time for faster processing
            new_wait_time = max(
                self.config.min_wait_time_ms,
                self.adaptive_config['current_wait_time'] - 100
            )
        elif current_load < 3:
            # Increase wait time to allow larger batches
            new_wait_time = min(
                self.config.max_wait_time_ms,
                self.adaptive_config['current_wait_time'] + 200
            )
        else:
            new_wait_time = self.adaptive_config['current_wait_time']
        
        # Apply changes if significant
        if (new_batch_size != self.adaptive_config['current_batch_size'] or
            abs(new_wait_time - self.adaptive_config['current_wait_time']) > 50):
            
            logger.info(f"Adaptive adjustment: batch_size {self.adaptive_config['current_batch_size']} -> {new_batch_size}, "
                       f"wait_time {self.adaptive_config['current_wait_time']} -> {new_wait_time}")
            
            self.adaptive_config['current_batch_size'] = new_batch_size
            self.adaptive_config['current_wait_time'] = new_wait_time
            self.adaptive_config['last_adjustment'] = datetime.now()
    
    def get_metrics(self) -> BatchingMetrics:
        """Get current batching metrics"""
        return self.metrics
    
    def get_queue_status(self) -> Dict[str, Any]:
        """Get current queue status"""
        return {
            'total_size': self.request_queue.size(),
            'priority_breakdown': self.request_queue.size_by_priority(),
            'active_batches': len(self.active_batches),
            'processing_workers': len([w for w in self.batch_workers if not w.done()]),
            'adaptive_config': self.adaptive_config if self.config.strategy == BatchStrategy.ADAPTIVE else None
        }
    
    def get_performance_insights(self) -> Dict[str, Any]:
        """Get performance insights and recommendations"""
        if len(self.processing_history) < 10:
            return {'message': 'Insufficient data for insights'}
        
        recent_entries = [
            entry for entry in self.processing_history
            if datetime.now() - datetime.fromisoformat(entry['timestamp']) <= timedelta(minutes=10)
        ]
        
        if not recent_entries:
            return {'message': 'No recent activity'}
        
        insights = {
            'current_performance': {
                'avg_batch_size': sum(e['request_count'] for e in recent_entries) / len(recent_entries),
                'avg_processing_time_ms': sum(e['processing_time_ms'] for e in recent_entries) / len(recent_entries),
                'success_rate': sum(e['success_rate'] for e in recent_entries) / len(recent_entries),
            },
            'recommendations': []
        }
        
        # Generate recommendations
        avg_processing_time = insights['current_performance']['avg_processing_time_ms']
        
        if avg_processing_time > 8000:
            insights['recommendations'].append({
                'type': 'performance',
                'message': 'High processing time detected. Consider reducing batch size or increasing workers.'
            })
        
        if self.metrics.cache_hit_rate < 50:
            insights['recommendations'].append({
                'type': 'cache',
                'message': 'Low cache hit rate. Consider cache warming or optimization.'
            })
        
        if self.metrics.current_queue_size > 50:
            insights['recommendations'].append({
                'type': 'capacity',
                'message': 'High queue size. Consider scaling up processing capacity.'
            })
        
        return insights