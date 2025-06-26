"""
Error Handling and Retry System for Mental Health LLM Evaluation

This module provides comprehensive error handling, retry logic, circuit breakers,
and resilience patterns for conversation generation systems.
"""

import asyncio
import logging
import time
from typing import Dict, List, Optional, Any, Callable, Type, Union
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import json
import random
from functools import wraps

logger = logging.getLogger(__name__)


class ErrorCategory(Enum):
    """Categories of errors that can occur during conversation generation."""
    
    # Network and API errors
    NETWORK_ERROR = "network_error"
    API_TIMEOUT = "api_timeout"
    API_RATE_LIMIT = "api_rate_limit"
    API_AUTHENTICATION = "api_authentication"
    API_QUOTA_EXCEEDED = "api_quota_exceeded"
    
    # Model and generation errors
    MODEL_ERROR = "model_error"
    GENERATION_FAILURE = "generation_failure"
    CONTEXT_LENGTH_EXCEEDED = "context_length_exceeded"
    CONTENT_FILTERING = "content_filtering"
    
    # System and resource errors
    MEMORY_ERROR = "memory_error"
    DISK_SPACE_ERROR = "disk_space_error"
    CPU_OVERLOAD = "cpu_overload"
    DEPENDENCY_ERROR = "dependency_error"
    
    # Data and validation errors
    INVALID_INPUT = "invalid_input"
    VALIDATION_ERROR = "validation_error"
    SERIALIZATION_ERROR = "serialization_error"
    
    # Application logic errors
    CONFIGURATION_ERROR = "configuration_error"
    STATE_ERROR = "state_error"
    SAFETY_VIOLATION = "safety_violation"
    
    # Unknown errors
    UNKNOWN_ERROR = "unknown_error"


class RetryStrategy(Enum):
    """Different retry strategies for handling errors."""
    
    EXPONENTIAL_BACKOFF = "exponential_backoff"
    LINEAR_BACKOFF = "linear_backoff"
    FIXED_DELAY = "fixed_delay"
    IMMEDIATE = "immediate"
    NO_RETRY = "no_retry"


@dataclass
class ErrorInfo:
    """Information about an error occurrence."""
    
    error_id: str
    category: ErrorCategory
    exception: Exception
    timestamp: datetime
    context: Dict[str, Any]
    
    # Error details
    error_message: str
    stack_trace: Optional[str] = None
    retry_count: int = 0
    recovery_attempted: bool = False
    
    # Classification
    is_recoverable: bool = True
    is_transient: bool = True
    severity: str = "medium"  # low, medium, high, critical
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "error_id": self.error_id,
            "category": self.category.value,
            "error_type": type(self.exception).__name__,
            "error_message": self.error_message,
            "timestamp": self.timestamp.isoformat(),
            "context": self.context,
            "stack_trace": self.stack_trace,
            "retry_count": self.retry_count,
            "recovery_attempted": self.recovery_attempted,
            "is_recoverable": self.is_recoverable,
            "is_transient": self.is_transient,
            "severity": self.severity
        }


@dataclass
class RetryConfig:
    """Configuration for retry behavior."""
    
    strategy: RetryStrategy = RetryStrategy.EXPONENTIAL_BACKOFF
    max_attempts: int = 3
    base_delay: float = 1.0
    max_delay: float = 60.0
    backoff_multiplier: float = 2.0
    jitter: bool = True
    
    # Conditional retry settings
    retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    non_retryable_exceptions: List[Type[Exception]] = field(default_factory=list)
    retryable_error_categories: List[ErrorCategory] = field(default_factory=list)


@dataclass
class CircuitBreakerConfig:
    """Configuration for circuit breaker pattern."""
    
    failure_threshold: int = 5
    recovery_timeout: float = 60.0
    expected_exception: Type[Exception] = Exception
    
    # State tracking
    name: str = "default"
    half_open_max_calls: int = 3


class CircuitBreakerState(Enum):
    """Circuit breaker states."""
    
    CLOSED = "closed"      # Normal operation
    OPEN = "open"         # Failing, calls rejected
    HALF_OPEN = "half_open"  # Testing if service recovered


class CircuitBreaker:
    """Circuit breaker implementation for preventing cascade failures."""
    
    def __init__(self, config: CircuitBreakerConfig):
        """Initialize circuit breaker."""
        self.config = config
        self.state = CircuitBreakerState.CLOSED
        self.failure_count = 0
        self.last_failure_time: Optional[datetime] = None
        self.half_open_calls = 0
        self.logger = logging.getLogger(__name__)
    
    async def call(self, func: Callable, *args, **kwargs):
        """Execute function with circuit breaker protection."""
        
        if self.state == CircuitBreakerState.OPEN:
            if self._should_attempt_reset():
                self.state = CircuitBreakerState.HALF_OPEN
                self.half_open_calls = 0
                self.logger.info(f"Circuit breaker {self.config.name} entering HALF_OPEN state")
            else:
                raise Exception(f"Circuit breaker {self.config.name} is OPEN")
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            if self.half_open_calls >= self.config.half_open_max_calls:
                raise Exception(f"Circuit breaker {self.config.name} HALF_OPEN call limit exceeded")
        
        try:
            result = await func(*args, **kwargs)
            self._on_success()
            return result
        
        except self.config.expected_exception as e:
            self._on_failure()
            raise
    
    def _should_attempt_reset(self) -> bool:
        """Check if circuit breaker should attempt to reset."""
        if self.last_failure_time is None:
            return True
        
        time_since_failure = datetime.now() - self.last_failure_time
        return time_since_failure.total_seconds() >= self.config.recovery_timeout
    
    def _on_success(self):
        """Handle successful call."""
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.half_open_calls += 1
            if self.half_open_calls >= self.config.half_open_max_calls:
                self.state = CircuitBreakerState.CLOSED
                self.failure_count = 0
                self.logger.info(f"Circuit breaker {self.config.name} CLOSED (recovered)")
        
        if self.state == CircuitBreakerState.CLOSED:
            self.failure_count = 0
    
    def _on_failure(self):
        """Handle failed call."""
        self.failure_count += 1
        self.last_failure_time = datetime.now()
        
        if self.state == CircuitBreakerState.HALF_OPEN:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.config.name} OPEN (half-open test failed)")
        
        elif self.failure_count >= self.config.failure_threshold:
            self.state = CircuitBreakerState.OPEN
            self.logger.warning(f"Circuit breaker {self.config.name} OPEN (threshold exceeded)")


class ErrorHandler:
    """
    Comprehensive error handling system with retry logic, circuit breakers,
    and recovery mechanisms for conversation generation.
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize error handler.
        
        Args:
            config: Configuration for error handling behavior
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Error tracking
        self.error_history: List[ErrorInfo] = []
        self.error_counts: Dict[ErrorCategory, int] = {cat: 0 for cat in ErrorCategory}
        self.recovery_stats: Dict[str, int] = {
            "successful_retries": 0,
            "failed_retries": 0,
            "circuit_breaker_trips": 0,
            "manual_recoveries": 0
        }
        
        # Circuit breakers
        self.circuit_breakers: Dict[str, CircuitBreaker] = {}
        
        # Default retry configurations
        self.default_retry_configs = {
            ErrorCategory.NETWORK_ERROR: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=1.0,
                retryable_error_categories=[ErrorCategory.NETWORK_ERROR]
            ),
            ErrorCategory.API_TIMEOUT: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=3,
                base_delay=2.0,
                retryable_error_categories=[ErrorCategory.API_TIMEOUT]
            ),
            ErrorCategory.API_RATE_LIMIT: RetryConfig(
                strategy=RetryStrategy.LINEAR_BACKOFF,
                max_attempts=5,
                base_delay=60.0,  # Wait longer for rate limits
                retryable_error_categories=[ErrorCategory.API_RATE_LIMIT]
            ),
            ErrorCategory.MODEL_ERROR: RetryConfig(
                strategy=RetryStrategy.EXPONENTIAL_BACKOFF,
                max_attempts=2,
                base_delay=1.0,
                retryable_error_categories=[ErrorCategory.MODEL_ERROR]
            ),
            ErrorCategory.MEMORY_ERROR: RetryConfig(
                strategy=RetryStrategy.NO_RETRY,  # Memory errors typically need intervention
                max_attempts=0,
                retryable_error_categories=[]
            )
        }
        
        # Initialize default circuit breakers
        self._setup_default_circuit_breakers()
        
        self.logger.info("ErrorHandler initialized")
    
    def _setup_default_circuit_breakers(self):
        """Setup default circuit breakers for common failure points."""
        
        # API circuit breaker
        api_config = CircuitBreakerConfig(
            name="api_calls",
            failure_threshold=10,
            recovery_timeout=120.0,
            expected_exception=Exception
        )
        self.circuit_breakers["api_calls"] = CircuitBreaker(api_config)
        
        # Model inference circuit breaker
        model_config = CircuitBreakerConfig(
            name="model_inference",
            failure_threshold=5,
            recovery_timeout=60.0,
            expected_exception=Exception
        )
        self.circuit_breakers["model_inference"] = CircuitBreaker(model_config)
        
        # Database circuit breaker
        db_config = CircuitBreakerConfig(
            name="database",
            failure_threshold=3,
            recovery_timeout=30.0,
            expected_exception=Exception
        )
        self.circuit_breakers["database"] = CircuitBreaker(db_config)
    
    async def execute_with_retry(
        self,
        func: Callable,
        *args,
        retry_config: Optional[RetryConfig] = None,
        circuit_breaker_name: Optional[str] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Any:
        """
        Execute function with comprehensive error handling and retry logic.
        
        Args:
            func: Function to execute
            *args: Function arguments
            retry_config: Retry configuration (optional)
            circuit_breaker_name: Circuit breaker to use (optional)
            context: Additional context for error tracking
            **kwargs: Function keyword arguments
            
        Returns:
            Function result
            
        Raises:
            Exception: After all retry attempts exhausted
        """
        context = context or {}
        retry_config = retry_config or RetryConfig()
        
        last_exception = None
        
        for attempt in range(retry_config.max_attempts + 1):
            try:
                # Use circuit breaker if specified
                if circuit_breaker_name and circuit_breaker_name in self.circuit_breakers:
                    circuit_breaker = self.circuit_breakers[circuit_breaker_name]
                    return await circuit_breaker.call(func, *args, **kwargs)
                else:
                    # Direct execution
                    if asyncio.iscoroutinefunction(func):
                        return await func(*args, **kwargs)
                    else:
                        return func(*args, **kwargs)
            
            except Exception as e:
                last_exception = e
                
                # Classify error
                error_category = self.classify_error(e)
                
                # Create error info
                error_info = ErrorInfo(
                    error_id=f"err_{int(time.time())}_{attempt}",
                    category=error_category,
                    exception=e,
                    timestamp=datetime.now(),
                    context=context,
                    error_message=str(e),
                    retry_count=attempt,
                    is_recoverable=self.is_recoverable_error(e, error_category),
                    is_transient=self.is_transient_error(e, error_category),
                    severity=self.get_error_severity(e, error_category)
                )
                
                # Track error
                self.track_error(error_info)
                
                # Check if should retry
                if not self.should_retry(error_info, retry_config, attempt):
                    break
                
                # Calculate delay
                delay = self.calculate_retry_delay(retry_config, attempt)
                
                self.logger.warning(
                    f"Attempt {attempt + 1} failed: {error_category.value} - {str(e)}. "
                    f"Retrying in {delay:.2f}s"
                )
                
                # Wait before retry
                if delay > 0:
                    await asyncio.sleep(delay)
        
        # All retries exhausted
        self.recovery_stats["failed_retries"] += 1
        
        if last_exception:
            self.logger.error(f"All retry attempts exhausted. Final error: {last_exception}")
            raise last_exception
        else:
            raise Exception("Unknown error during retry execution")
    
    def classify_error(self, exception: Exception) -> ErrorCategory:
        """Classify an exception into an error category."""
        
        error_type = type(exception).__name__
        error_message = str(exception).lower()
        
        # Network and connectivity errors
        if any(term in error_message for term in ["connection", "network", "unreachable", "dns"]):
            return ErrorCategory.NETWORK_ERROR
        
        # Timeout errors
        if any(term in error_message for term in ["timeout", "timed out", "deadline"]):
            return ErrorCategory.API_TIMEOUT
        
        # Rate limiting
        if any(term in error_message for term in ["rate limit", "quota", "throttle", "429"]):
            return ErrorCategory.API_RATE_LIMIT
        
        # Authentication
        if any(term in error_message for term in ["auth", "unauthorized", "401", "403", "api key"]):
            return ErrorCategory.API_AUTHENTICATION
        
        # Memory errors
        if any(term in error_message for term in ["memory", "out of memory", "oom"]):
            return ErrorCategory.MEMORY_ERROR
        
        # Model-specific errors
        if any(term in error_message for term in ["model", "generation", "inference", "token"]):
            return ErrorCategory.MODEL_ERROR
        
        # Content filtering
        if any(term in error_message for term in ["content", "filter", "policy", "violation"]):
            return ErrorCategory.CONTENT_FILTERING
        
        # Validation errors
        if any(term in error_message for term in ["validation", "invalid", "malformed"]):
            return ErrorCategory.VALIDATION_ERROR
        
        # Serialization errors
        if any(term in error_message for term in ["json", "serialize", "encode", "decode"]):
            return ErrorCategory.SERIALIZATION_ERROR
        
        # Configuration errors
        if any(term in error_message for term in ["config", "setting", "parameter"]):
            return ErrorCategory.CONFIGURATION_ERROR
        
        # Common exception types
        if error_type in ["ConnectionError", "TimeoutError", "HTTPError"]:
            return ErrorCategory.NETWORK_ERROR
        
        if error_type in ["ValueError", "TypeError", "KeyError"]:
            return ErrorCategory.VALIDATION_ERROR
        
        if error_type in ["MemoryError", "ResourceWarning"]:
            return ErrorCategory.MEMORY_ERROR
        
        # Default to unknown
        return ErrorCategory.UNKNOWN_ERROR
    
    def is_recoverable_error(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is recoverable."""
        
        # Non-recoverable categories
        non_recoverable = [
            ErrorCategory.API_AUTHENTICATION,
            ErrorCategory.CONFIGURATION_ERROR,
            ErrorCategory.VALIDATION_ERROR,
            ErrorCategory.SAFETY_VIOLATION
        ]
        
        if category in non_recoverable:
            return False
        
        # Memory errors are typically not recoverable without intervention
        if category == ErrorCategory.MEMORY_ERROR:
            return False
        
        # Check exception type
        non_recoverable_types = [
            "SyntaxError",
            "ImportError",
            "ModuleNotFoundError",
            "AttributeError"
        ]
        
        if type(exception).__name__ in non_recoverable_types:
            return False
        
        return True
    
    def is_transient_error(self, exception: Exception, category: ErrorCategory) -> bool:
        """Determine if an error is transient (likely to resolve on retry)."""
        
        # Typically transient categories
        transient = [
            ErrorCategory.NETWORK_ERROR,
            ErrorCategory.API_TIMEOUT,
            ErrorCategory.API_RATE_LIMIT,
            ErrorCategory.CPU_OVERLOAD
        ]
        
        return category in transient
    
    def get_error_severity(self, exception: Exception, category: ErrorCategory) -> str:
        """Determine error severity level."""
        
        critical_categories = [
            ErrorCategory.SAFETY_VIOLATION,
            ErrorCategory.MEMORY_ERROR,
            ErrorCategory.API_AUTHENTICATION
        ]
        
        high_categories = [
            ErrorCategory.MODEL_ERROR,
            ErrorCategory.CONFIGURATION_ERROR,
            ErrorCategory.DEPENDENCY_ERROR
        ]
        
        if category in critical_categories:
            return "critical"
        elif category in high_categories:
            return "high"
        elif category in [ErrorCategory.NETWORK_ERROR, ErrorCategory.API_TIMEOUT]:
            return "medium"
        else:
            return "low"
    
    def should_retry(self, error_info: ErrorInfo, retry_config: RetryConfig, attempt: int) -> bool:
        """Determine if should retry based on error and configuration."""
        
        # Check if attempts exhausted
        if attempt >= retry_config.max_attempts:
            return False
        
        # Check if error is recoverable
        if not error_info.is_recoverable:
            return False
        
        # Check retry strategy
        if retry_config.strategy == RetryStrategy.NO_RETRY:
            return False
        
        # Check retryable categories
        if (retry_config.retryable_error_categories and 
            error_info.category not in retry_config.retryable_error_categories):
            return False
        
        # Check non-retryable exceptions
        if (retry_config.non_retryable_exceptions and 
            type(error_info.exception) in retry_config.non_retryable_exceptions):
            return False
        
        # Check retryable exceptions (if specified, must match)
        if (retry_config.retryable_exceptions and 
            type(error_info.exception) not in retry_config.retryable_exceptions):
            return False
        
        return True
    
    def calculate_retry_delay(self, retry_config: RetryConfig, attempt: int) -> float:
        """Calculate delay before retry based on strategy."""
        
        if retry_config.strategy == RetryStrategy.IMMEDIATE:
            return 0.0
        
        elif retry_config.strategy == RetryStrategy.FIXED_DELAY:
            delay = retry_config.base_delay
        
        elif retry_config.strategy == RetryStrategy.LINEAR_BACKOFF:
            delay = retry_config.base_delay * (attempt + 1)
        
        elif retry_config.strategy == RetryStrategy.EXPONENTIAL_BACKOFF:
            delay = retry_config.base_delay * (retry_config.backoff_multiplier ** attempt)
        
        else:
            delay = retry_config.base_delay
        
        # Apply maximum delay limit
        delay = min(delay, retry_config.max_delay)
        
        # Add jitter if enabled
        if retry_config.jitter:
            jitter_range = delay * 0.1  # 10% jitter
            delay += random.uniform(-jitter_range, jitter_range)
        
        return max(0.0, delay)
    
    def track_error(self, error_info: ErrorInfo):
        """Track error occurrence for analysis."""
        
        self.error_history.append(error_info)
        self.error_counts[error_info.category] += 1
        
        # Limit history size
        max_history = 1000
        if len(self.error_history) > max_history:
            self.error_history = self.error_history[-max_history:]
        
        # Log based on severity
        if error_info.severity == "critical":
            self.logger.critical(f"Critical error: {error_info.error_message}")
        elif error_info.severity == "high":
            self.logger.error(f"High severity error: {error_info.error_message}")
        elif error_info.severity == "medium":
            self.logger.warning(f"Medium severity error: {error_info.error_message}")
        else:
            self.logger.debug(f"Low severity error: {error_info.error_message}")
    
    def get_error_statistics(self) -> Dict[str, Any]:
        """Get comprehensive error statistics."""
        
        total_errors = len(self.error_history)
        
        # Error distribution by category
        category_distribution = {
            category.value: count for category, count in self.error_counts.items()
        }
        
        # Recent error trends (last hour)
        one_hour_ago = datetime.now() - timedelta(hours=1)
        recent_errors = [
            error for error in self.error_history 
            if error.timestamp >= one_hour_ago
        ]
        
        # Severity distribution
        severity_counts = {"low": 0, "medium": 0, "high": 0, "critical": 0}
        for error in self.error_history:
            severity_counts[error.severity] += 1
        
        # Circuit breaker states
        circuit_breaker_states = {
            name: breaker.state.value 
            for name, breaker in self.circuit_breakers.items()
        }
        
        return {
            "total_errors": total_errors,
            "recent_errors_1h": len(recent_errors),
            "error_rate_per_hour": len(recent_errors),
            "category_distribution": category_distribution,
            "severity_distribution": severity_counts,
            "recovery_statistics": self.recovery_stats,
            "circuit_breaker_states": circuit_breaker_states,
            "most_common_error": max(category_distribution.items(), key=lambda x: x[1])[0] if category_distribution else None
        }
    
    def get_retry_config_for_category(self, category: ErrorCategory) -> RetryConfig:
        """Get retry configuration for specific error category."""
        return self.default_retry_configs.get(category, RetryConfig())
    
    def add_circuit_breaker(self, name: str, config: CircuitBreakerConfig):
        """Add a custom circuit breaker."""
        self.circuit_breakers[name] = CircuitBreaker(config)
        self.logger.info(f"Added circuit breaker: {name}")
    
    def reset_circuit_breaker(self, name: str):
        """Manually reset a circuit breaker."""
        if name in self.circuit_breakers:
            breaker = self.circuit_breakers[name]
            breaker.state = CircuitBreakerState.CLOSED
            breaker.failure_count = 0
            breaker.last_failure_time = None
            self.recovery_stats["manual_recoveries"] += 1
            self.logger.info(f"Circuit breaker {name} manually reset")
    
    def export_error_report(self, output_path: str):
        """Export comprehensive error report."""
        
        report = {
            "report_timestamp": datetime.now().isoformat(),
            "error_statistics": self.get_error_statistics(),
            "error_history": [error.to_dict() for error in self.error_history[-100:]],  # Last 100 errors
            "circuit_breaker_configs": {
                name: {
                    "state": breaker.state.value,
                    "failure_count": breaker.failure_count,
                    "config": {
                        "failure_threshold": breaker.config.failure_threshold,
                        "recovery_timeout": breaker.config.recovery_timeout
                    }
                }
                for name, breaker in self.circuit_breakers.items()
            },
            "configuration": self.config
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Error report exported to {output_path}")


def with_error_handling(
    retry_config: Optional[RetryConfig] = None,
    circuit_breaker_name: Optional[str] = None,
    error_handler: Optional[ErrorHandler] = None
):
    """
    Decorator for adding error handling and retry logic to functions.
    
    Args:
        retry_config: Retry configuration
        circuit_breaker_name: Circuit breaker to use
        error_handler: Error handler instance
    """
    def decorator(func):
        @wraps(func)
        async def wrapper(*args, **kwargs):
            handler = error_handler or ErrorHandler()
            
            return await handler.execute_with_retry(
                func,
                *args,
                retry_config=retry_config,
                circuit_breaker_name=circuit_breaker_name,
                **kwargs
            )
        return wrapper
    return decorator