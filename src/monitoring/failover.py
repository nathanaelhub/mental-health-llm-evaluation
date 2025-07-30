"""
Failover and Circuit Breaker System

Advanced failover mechanisms including circuit breakers, fallback strategies,
and automatic recovery for the mental health LLM system components.
"""

import asyncio
import logging
import time
import random
from datetime import datetime, timedelta
from typing import Dict, List, Any, Optional, Callable, Union
from dataclasses import dataclass, field
from enum import Enum
import aiohttp
from contextlib import asynccontextmanager

logger = logging.getLogger(__name__)


class CircuitState(Enum):
    """Circuit breaker states"""
    CLOSED = "closed"      # Normal operation
    OPEN = "open"          # Failing, calls blocked
    HALF_OPEN = "half_open"  # Testing recovery


class FailoverStrategy(Enum):
    """Failover strategies"""
    ROUND_ROBIN = "round_robin"
    RANDOM = "random"
    WEIGHTED = "weighted"
    LEAST_LATENCY = "least_latency"
    LEAST_ERRORS = "least_errors"
    PRIORITY = "priority"


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker"""
    failure_threshold: int = 5  # Failures before opening
    recovery_timeout: int = 60  # Seconds before trying half-open
    success_threshold: int = 3   # Successes to close from half-open
    timeout: float = 30.0       # Request timeout in seconds
    
    # Sliding window configuration
    window_size: int = 100      # Number of requests in window
    failure_rate_threshold: float = 0.5  # 50% failure rate to open


@dataclass
class EndpointConfig:
    """Configuration for a service endpoint"""
    name: str
    url: str
    weight: int = 1
    priority: int = 1  # Lower numbers = higher priority
    timeout: float = 30.0
    max_concurrent: int = 100
    
    # Health check configuration
    health_check_path: str = "/health"
    health_check_interval: int = 30
    
    # Retry configuration
    max_retries: int = 3
    retry_delay: float = 1.0
    retry_backoff: float = 2.0


@dataclass
class RequestMetrics:
    """Metrics for tracking endpoint performance"""
    total_requests: int = 0
    successful_requests: int = 0
    failed_requests: int = 0
    total_latency: float = 0.0
    avg_latency: float = 0.0
    last_success: Optional[datetime] = None
    last_failure: Optional[datetime] = None
    
    # Sliding window for recent performance
    recent_requests: List[bool] = field(default_factory=lambda: [])  # True for success
    recent_latencies: List[float] = field(default_factory=list)
    
    def add_result(self, success: bool, latency: float, window_size: int = 100):
        """Add a request result to metrics"""
        self.total_requests += 1
        self.total_latency += latency
        
        if success:
            self.successful_requests += 1
            self.last_success = datetime.now()
        else:
            self.failed_requests += 1
            self.last_failure = datetime.now()
        
        # Update averages
        self.avg_latency = self.total_latency / self.total_requests
        
        # Update sliding windows
        self.recent_requests.append(success)
        self.recent_latencies.append(latency)
        
        # Keep windows to specified size
        if len(self.recent_requests) > window_size:
            self.recent_requests.pop(0)
        if len(self.recent_latencies) > window_size:
            self.recent_latencies.pop(0)
    
    def get_recent_failure_rate(self) -> float:
        """Get failure rate in recent window"""
        if not self.recent_requests:
            return 0.0
        
        failures = sum(1 for success in self.recent_requests if not success)
        return failures / len(self.recent_requests)
    
    def get_recent_avg_latency(self) -> float:
        """Get average latency in recent window"""
        if not self.recent_latencies:
            return 0.0
        
        return sum(self.recent_latencies) / len(self.recent_latencies)


class CircuitBreaker:
    """Circuit breaker implementation"""
    
    def __init__(self, name: str, config: CircuitBreakerConfig):
        self.name = name
        self.config = config
        self.state = CircuitState.CLOSED
        self.failure_count = 0
        self.success_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.metrics = RequestMetrics()
        
        logger.info(f"Circuit breaker created: {name}")
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function through circuit breaker"""
        
        # Check if circuit is open and can transition to half-open
        if self.state == CircuitState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitState.HALF_OPEN
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} transitioning to HALF_OPEN")
            else:
                raise CircuitBreakerOpenError(f"Circuit breaker {self.name} is OPEN")
        
        start_time = time.time()
        
        try:
            # Execute the function with timeout
            result = await asyncio.wait_for(func(*args, **kwargs), timeout=self.config.timeout)
            
            # Record success
            latency = time.time() - start_time
            self.metrics.add_result(True, latency, self.config.window_size)
            self._on_success()
            
            return result
        
        except Exception as e:
            # Record failure
            latency = time.time() - start_time
            self.metrics.add_result(False, latency, self.config.window_size)
            self._on_failure()
            
            raise e
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit should attempt to reset from OPEN to HALF_OPEN"""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful request"""
        self.failure_count = 0
        
        if self.state == CircuitState.HALF_OPEN:
            self.success_count += 1
            
            if self.success_count >= self.config.success_threshold:
                self.state = CircuitState.CLOSED
                self.success_count = 0
                logger.info(f"Circuit breaker {self.name} closed after recovery")
    
    def _on_failure(self):
        """Handle failed request"""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitState.HALF_OPEN:
            # Failed during recovery, go back to OPEN
            self.state = CircuitState.OPEN
            logger.warning(f"Circuit breaker {self.name} reopened after failed recovery attempt")
        
        elif self.state == CircuitState.CLOSED:
            # Check if we should open the circuit
            if self._should_open_circuit():
                self.state = CircuitState.OPEN
                logger.error(f"Circuit breaker {self.name} opened due to failures")
    
    def _should_open_circuit(self) -> bool:
        """Determine if circuit should be opened"""
        
        # Simple failure count threshold
        if self.failure_count >= self.config.failure_threshold:
            return True
        
        # Failure rate threshold in sliding window
        if len(self.metrics.recent_requests) >= self.config.window_size:
            failure_rate = self.metrics.get_recent_failure_rate()
            if failure_rate >= self.config.failure_rate_threshold:
                return True
        
        return False
    
    def get_state(self) -> Dict[str, Any]:
        """Get current circuit breaker state"""
        return {
            'name': self.name,
            'state': self.state.value,
            'failure_count': self.failure_count,
            'success_count': self.success_count,
            'last_failure_time': self.last_failure_time.isoformat() if self.last_failure_time else None,
            'metrics': {
                'total_requests': self.metrics.total_requests,
                'success_rate': self.metrics.successful_requests / max(1, self.metrics.total_requests),
                'failure_rate': self.metrics.failed_requests / max(1, self.metrics.total_requests),
                'avg_latency': self.metrics.avg_latency,
                'recent_failure_rate': self.metrics.get_recent_failure_rate(),
                'recent_avg_latency': self.metrics.get_recent_avg_latency()
            }
        }


class CircuitBreakerOpenError(Exception):
    """Exception raised when circuit breaker is open"""
    pass


class LoadBalancer:
    """Load balancer with multiple failover strategies"""
    
    def __init__(self, 
                 name: str,
                 endpoints: List[EndpointConfig],
                 strategy: FailoverStrategy = FailoverStrategy.ROUND_ROBIN,
                 circuit_breaker_config: CircuitBreakerConfig = None):
        
        self.name = name
        self.endpoints = {ep.name: ep for ep in endpoints}
        self.strategy = strategy
        self.circuit_breaker_config = circuit_breaker_config or CircuitBreakerConfig()
        
        # Circuit breakers for each endpoint
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        for endpoint in endpoints:
            self.circuit_breakers[endpoint.name] = CircuitBreaker(
                f"{name}_{endpoint.name}",
                self.circuit_breaker_config
            )
        
        # Strategy state
        self._round_robin_index = 0
        self._endpoint_semaphores: Dict[str, asyncio.Semaphore] = {}
        
        # Initialize semaphores for concurrency control
        for endpoint in endpoints:
            self._endpoint_semaphores[endpoint.name] = asyncio.Semaphore(endpoint.max_concurrent)
        
        # Health checking
        self._health_check_task: Optional[asyncio.Task] = None
        self._healthy_endpoints: Set[str] = set(ep.name for ep in endpoints)
        self._running = False
        
        logger.info(f"Load balancer created: {name} with {len(endpoints)} endpoints")
    
    async def start(self):
        """Start the load balancer and health checking"""
        if self._running:
            return
        
        self._running = True
        self._health_check_task = asyncio.create_task(self._health_check_loop())
        
        logger.info(f"Load balancer {self.name} started")
    
    async def stop(self):
        """Stop the load balancer"""
        self._running = False
        
        if self._health_check_task:
            self._health_check_task.cancel()
            try:
                await self._health_check_task
            except asyncio.CancelledError:
                pass
        
        logger.info(f"Load balancer {self.name} stopped")
    
    async def make_request(self, 
                          method: str,
                          path: str,
                          **kwargs) -> aiohttp.ClientResponse:
        """Make a request through the load balancer"""
        
        available_endpoints = self._get_available_endpoints()
        
        if not available_endpoints:
            raise NoHealthyEndpointsError(f"No healthy endpoints available for {self.name}")
        
        # Try endpoints in order based on strategy
        last_exception = None
        
        for endpoint_name in self._select_endpoints(available_endpoints):
            endpoint = self.endpoints[endpoint_name]
            circuit_breaker = self.circuit_breakers[endpoint_name]
            
            try:
                # Check concurrency limit
                semaphore = self._endpoint_semaphores[endpoint_name]
                async with semaphore:
                    
                    # Make request through circuit breaker
                    result = await circuit_breaker.call(
                        self._make_http_request,
                        endpoint,
                        method,
                        path,
                        **kwargs
                    )
                    
                    return result
            
            except CircuitBreakerOpenError:
                logger.warning(f"Circuit breaker open for {endpoint_name}, trying next endpoint")
                continue
            
            except Exception as e:
                logger.warning(f"Request failed to {endpoint_name}: {e}")
                last_exception = e
                
                # Try next endpoint for certain types of errors
                if isinstance(e, (aiohttp.ClientError, asyncio.TimeoutError)):
                    continue
                else:
                    # For other errors, don't retry on other endpoints
                    raise e
        
        # All endpoints failed
        if last_exception:
            raise last_exception
        else:
            raise NoHealthyEndpointsError(f"All endpoints failed for {self.name}")
    
    async def _make_http_request(self,
                               endpoint: EndpointConfig,
                               method: str,
                               path: str,
                               **kwargs) -> aiohttp.ClientResponse:
        """Make HTTP request to specific endpoint"""
        
        url = f"{endpoint.url.rstrip('/')}/{path.lstrip('/')}"
        timeout = aiohttp.ClientTimeout(total=endpoint.timeout)
        
        # Retry logic
        max_retries = endpoint.max_retries
        retry_delay = endpoint.retry_delay
        
        for attempt in range(max_retries + 1):
            try:
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.request(method, url, **kwargs) as response:
                        # Check if response indicates success
                        if response.status < 500:  # Don't retry on client errors
                            return response
                        else:
                            raise aiohttp.ClientResponseError(
                                request_info=response.request_info,
                                history=response.history,
                                status=response.status
                            )
            
            except Exception as e:
                if attempt < max_retries:
                    logger.debug(f"Request attempt {attempt + 1} failed to {endpoint.name}: {e}")
                    await asyncio.sleep(retry_delay)
                    retry_delay *= endpoint.retry_backoff
                else:
                    raise e
    
    def _get_available_endpoints(self) -> List[str]:
        """Get list of healthy endpoints"""
        
        available = []
        
        for endpoint_name in self.endpoints:
            # Check if endpoint is healthy
            if endpoint_name not in self._healthy_endpoints:
                continue
            
            # Check if circuit breaker allows requests
            circuit_breaker = self.circuit_breakers[endpoint_name]
            if circuit_breaker.state == CircuitState.OPEN:
                continue
            
            available.append(endpoint_name)
        
        return available
    
    def _select_endpoints(self, available_endpoints: List[str]) -> List[str]:
        """Select endpoints based on load balancing strategy"""
        
        if not available_endpoints:
            return []
        
        if self.strategy == FailoverStrategy.ROUND_ROBIN:
            # Round robin selection
            ordered = []
            start_index = self._round_robin_index % len(available_endpoints)
            
            for i in range(len(available_endpoints)):
                index = (start_index + i) % len(available_endpoints)
                ordered.append(available_endpoints[index])
            
            self._round_robin_index += 1
            return ordered
        
        elif self.strategy == FailoverStrategy.RANDOM:
            # Random selection, but try all
            shuffled = available_endpoints.copy()
            random.shuffle(shuffled)
            return shuffled
        
        elif self.strategy == FailoverStrategy.WEIGHTED:
            # Weighted random selection
            weighted_endpoints = []
            for endpoint_name in available_endpoints:
                endpoint = self.endpoints[endpoint_name]
                weighted_endpoints.extend([endpoint_name] * endpoint.weight)
            
            if weighted_endpoints:
                selected = random.choice(weighted_endpoints)
                # Put selected first, then others randomly
                others = [ep for ep in available_endpoints if ep != selected]
                random.shuffle(others)
                return [selected] + others
            
            return available_endpoints
        
        elif self.strategy == FailoverStrategy.LEAST_LATENCY:
            # Sort by average latency
            sorted_endpoints = sorted(
                available_endpoints,
                key=lambda ep: self.circuit_breakers[ep].metrics.get_recent_avg_latency()
            )
            return sorted_endpoints
        
        elif self.strategy == FailoverStrategy.LEAST_ERRORS:
            # Sort by failure rate
            sorted_endpoints = sorted(
                available_endpoints,
                key=lambda ep: self.circuit_breakers[ep].metrics.get_recent_failure_rate()
            )
            return sorted_endpoints
        
        elif self.strategy == FailoverStrategy.PRIORITY:
            # Sort by priority (lower number = higher priority)
            sorted_endpoints = sorted(
                available_endpoints,
                key=lambda ep: self.endpoints[ep].priority
            )
            return sorted_endpoints
        
        else:
            return available_endpoints
    
    async def _health_check_loop(self):
        """Background health checking loop"""
        
        while self._running:
            try:
                # Check health of all endpoints
                health_tasks = []
                
                for endpoint_name, endpoint in self.endpoints.items():
                    task = asyncio.create_task(
                        self._check_endpoint_health(endpoint_name, endpoint)
                    )
                    health_tasks.append(task)
                
                # Wait for all health checks
                results = await asyncio.gather(*health_tasks, return_exceptions=True)
                
                # Update healthy endpoints
                new_healthy = set()
                
                for i, (endpoint_name, result) in enumerate(zip(self.endpoints.keys(), results)):
                    if isinstance(result, Exception):
                        logger.debug(f"Health check failed for {endpoint_name}: {result}")
                    elif result:
                        new_healthy.add(endpoint_name)
                
                # Log changes in health status
                newly_healthy = new_healthy - self._healthy_endpoints
                newly_unhealthy = self._healthy_endpoints - new_healthy
                
                for endpoint_name in newly_healthy:
                    logger.info(f"Endpoint {endpoint_name} is now healthy")
                
                for endpoint_name in newly_unhealthy:
                    logger.warning(f"Endpoint {endpoint_name} is now unhealthy")
                
                self._healthy_endpoints = new_healthy
                
                # Sleep until next check
                min_interval = min(ep.health_check_interval for ep in self.endpoints.values())
                await asyncio.sleep(min_interval)
            
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health check loop: {e}")
                await asyncio.sleep(30)
    
    async def _check_endpoint_health(self, endpoint_name: str, endpoint: EndpointConfig) -> bool:
        """Check health of a specific endpoint"""
        
        try:
            health_url = f"{endpoint.url.rstrip('/')}/{endpoint.health_check_path.lstrip('/')}"
            timeout = aiohttp.ClientTimeout(total=10)  # Short timeout for health checks
            
            async with aiohttp.ClientSession(timeout=timeout) as session:
                async with session.get(health_url) as response:
                    return response.status == 200
        
        except Exception:
            return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get load balancer status"""
        
        endpoint_status = {}
        
        for endpoint_name, endpoint in self.endpoints.items():
            circuit_breaker = self.circuit_breakers[endpoint_name]
            
            endpoint_status[endpoint_name] = {
                'url': endpoint.url,
                'healthy': endpoint_name in self._healthy_endpoints,
                'circuit_breaker': circuit_breaker.get_state(),
                'priority': endpoint.priority,
                'weight': endpoint.weight
            }
        
        return {
            'name': self.name,
            'strategy': self.strategy.value,
            'healthy_endpoints': len(self._healthy_endpoints),
            'total_endpoints': len(self.endpoints),
            'endpoints': endpoint_status
        }


class NoHealthyEndpointsError(Exception):
    """Exception raised when no healthy endpoints are available"""
    pass


class FailoverManager:
    """Central failover management system"""
    
    def __init__(self):
        self.load_balancers: Dict[str, LoadBalancer] = {}
        self._running = False
        
        logger.info("FailoverManager initialized")
    
    def add_load_balancer(self, load_balancer: LoadBalancer):
        """Add a load balancer to the manager"""
        self.load_balancers[load_balancer.name] = load_balancer
        logger.info(f"Added load balancer: {load_balancer.name}")
    
    def remove_load_balancer(self, name: str):
        """Remove a load balancer"""
        if name in self.load_balancers:
            del self.load_balancers[name]
            logger.info(f"Removed load balancer: {name}")
    
    async def start(self):
        """Start all load balancers"""
        if self._running:
            return
        
        self._running = True
        
        # Start all load balancers
        start_tasks = []
        for lb in self.load_balancers.values():
            start_tasks.append(lb.start())
        
        await asyncio.gather(*start_tasks)
        
        logger.info("FailoverManager started")
    
    async def stop(self):
        """Stop all load balancers"""
        self._running = False
        
        # Stop all load balancers
        stop_tasks = []
        for lb in self.load_balancers.values():
            stop_tasks.append(lb.stop())
        
        await asyncio.gather(*stop_tasks, return_exceptions=True)
        
        logger.info("FailoverManager stopped")
    
    async def make_request(self,
                          service_name: str,
                          method: str,
                          path: str,
                          **kwargs) -> aiohttp.ClientResponse:
        """Make a request through a specific load balancer"""
        
        if service_name not in self.load_balancers:
            raise ValueError(f"Unknown service: {service_name}")
        
        load_balancer = self.load_balancers[service_name]
        return await load_balancer.make_request(method, path, **kwargs)
    
    def get_overall_status(self) -> Dict[str, Any]:
        """Get status of all load balancers"""
        
        status = {
            'total_services': len(self.load_balancers),
            'healthy_services': 0,
            'services': {}
        }
        
        for name, lb in self.load_balancers.items():
            lb_status = lb.get_status()
            status['services'][name] = lb_status
            
            if lb_status['healthy_endpoints'] > 0:
                status['healthy_services'] += 1
        
        return status
    
    @asynccontextmanager
    async def managed_lifecycle(self):
        """Context manager for automatic startup/shutdown"""
        try:
            await self.start()
            yield self
        finally:
            await self.stop()


# Example usage and configuration
async def setup_model_service_failover() -> FailoverManager:
    """Setup failover for model services"""
    
    failover_manager = FailoverManager()
    
    # OpenAI service endpoints
    openai_endpoints = [
        EndpointConfig(
            name="openai_primary",
            url="https://api.openai.com",
            priority=1,
            weight=3,
            timeout=30.0
        ),
        EndpointConfig(
            name="openai_backup",
            url="https://api.openai.com",  # Could be different region
            priority=2,
            weight=1,
            timeout=45.0
        )
    ]
    
    openai_lb = LoadBalancer(
        name="openai_service",
        endpoints=openai_endpoints,
        strategy=FailoverStrategy.PRIORITY,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=60,
            success_threshold=2
        )
    )
    
    # Anthropic service endpoints
    anthropic_endpoints = [
        EndpointConfig(
            name="anthropic_primary",
            url="https://api.anthropic.com",
            priority=1,
            weight=3,
            timeout=30.0
        ),
        EndpointConfig(
            name="anthropic_backup", 
            url="https://api.anthropic.com",
            priority=2,
            weight=1,
            timeout=45.0
        )
    ]
    
    anthropic_lb = LoadBalancer(
        name="anthropic_service",
        endpoints=anthropic_endpoints,
        strategy=FailoverStrategy.LEAST_LATENCY,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=5,
            recovery_timeout=120,
            success_threshold=3
        )
    )
    
    # Local model endpoints (fallback)
    local_endpoints = [
        EndpointConfig(
            name="local_model_1",
            url="http://localhost:8001",
            priority=1,
            weight=2,
            timeout=60.0,
            health_check_interval=15
        ),
        EndpointConfig(
            name="local_model_2",
            url="http://localhost:8002",
            priority=2,
            weight=1,
            timeout=60.0,
            health_check_interval=15
        )
    ]
    
    local_lb = LoadBalancer(
        name="local_models",
        endpoints=local_endpoints,
        strategy=FailoverStrategy.ROUND_ROBIN,
        circuit_breaker_config=CircuitBreakerConfig(
            failure_threshold=3,
            recovery_timeout=30,
            success_threshold=2
        )
    )
    
    # Add load balancers to manager
    failover_manager.add_load_balancer(openai_lb)
    failover_manager.add_load_balancer(anthropic_lb)
    failover_manager.add_load_balancer(local_lb)
    
    return failover_manager


if __name__ == "__main__":
    async def main():
        # Setup failover system
        failover_manager = await setup_model_service_failover()
        
        async with failover_manager.managed_lifecycle():
            
            # Test requests
            try:
                # Make test request to OpenAI service
                response = await failover_manager.make_request(
                    "openai_service",
                    "POST",
                    "/v1/chat/completions",
                    json={
                        "model": "gpt-3.5-turbo",
                        "messages": [{"role": "user", "content": "Hello"}]
                    }
                )
                
                print(f"OpenAI request successful: {response.status}")
                
            except Exception as e:
                print(f"OpenAI request failed: {e}")
            
            # Print status
            status = failover_manager.get_overall_status()
            print(f"\nOverall Status:")
            print(f"Healthy Services: {status['healthy_services']}/{status['total_services']}")
            
            for service_name, service_status in status['services'].items():
                print(f"\n{service_name}:")
                print(f"  Strategy: {service_status['strategy']}")
                print(f"  Healthy Endpoints: {service_status['healthy_endpoints']}/{service_status['total_endpoints']}")
                
                for endpoint_name, endpoint_status in service_status['endpoints'].items():
                    cb_state = endpoint_status['circuit_breaker']['state']
                    healthy = endpoint_status['healthy']
                    print(f"    {endpoint_name}: {'✓' if healthy else '✗'} ({cb_state})")
            
            # Keep running for a while to see health checks
            await asyncio.sleep(60)
    
    asyncio.run(main())