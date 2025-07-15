"""
Logging configuration for mental health LLM evaluation system.

This module provides centralized logging configuration with support for
different log levels, formats, and output destinations.
"""

import logging
import logging.handlers
import os
import sys
from datetime import datetime
from typing import Optional, Dict, Any
import json
import structlog


def setup_logging(
    log_level: str = "INFO",
    log_format: str = "standard",
    log_file: Optional[str] = None,
    enable_console: bool = True,
    enable_structured: bool = False
) -> None:
    """
    Configure logging for the application.
    
    Args:
        log_level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_format: Format type ("standard", "detailed", "json")
        log_file: Optional file path for log output
        enable_console: Whether to enable console logging
        enable_structured: Whether to use structured logging
    """
    
    # Convert string level to logging constant
    numeric_level = getattr(logging, log_level.upper(), logging.INFO)
    
    # Create logs directory if needed
    if log_file:
        log_dir = os.path.dirname(log_file)
        if log_dir:
            os.makedirs(log_dir, exist_ok=True)
    
    # Configure root logger
    logging.root.handlers = []
    logging.root.setLevel(numeric_level)
    
    # Set up formatters
    formatters = _get_formatters()
    formatter = formatters.get(log_format, formatters["standard"])
    
    handlers = []
    
    # Console handler
    if enable_console:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(numeric_level)
        console_handler.setFormatter(formatter)
        handlers.append(console_handler)
    
    # File handler
    if log_file:
        file_handler = logging.handlers.RotatingFileHandler(
            log_file,
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5
        )
        file_handler.setLevel(numeric_level)
        file_handler.setFormatter(formatter)
        handlers.append(file_handler)
    
    # Add handlers to root logger
    for handler in handlers:
        logging.root.addHandler(handler)
    
    # Configure structured logging if enabled
    if enable_structured:
        _setup_structured_logging()
    
    # Set specific logger levels
    _configure_external_loggers()
    
    # Log configuration
    logger = logging.getLogger(__name__)
    logger.info(f"Logging configured: level={log_level}, format={log_format}")


def _get_formatters() -> Dict[str, logging.Formatter]:
    """Get available log formatters."""
    
    formatters = {
        "standard": logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        ),
        "detailed": logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(filename)s:%(lineno)d - %(funcName)s - %(message)s"
        ),
        "json": JsonFormatter()
    }
    
    return formatters


def _setup_structured_logging() -> None:
    """Configure structured logging with structlog."""
    
    structlog.configure(
        processors=[
            structlog.stdlib.filter_by_level,
            structlog.stdlib.add_logger_name,
            structlog.stdlib.add_log_level,
            structlog.stdlib.PositionalArgumentsFormatter(),
            structlog.processors.TimeStamper(fmt="ISO"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer()
        ],
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )


def _configure_external_loggers() -> None:
    """Configure logging levels for external libraries."""
    
    # Suppress verbose logs from external libraries
    external_loggers = {
        "urllib3.connectionpool": logging.WARNING,
        "requests.packages.urllib3": logging.WARNING,
        "matplotlib": logging.WARNING,
        "PIL": logging.WARNING,
        "transformers": logging.WARNING,
        "torch": logging.WARNING,
        "tensorflow": logging.ERROR,
    }
    
    for logger_name, level in external_loggers.items():
        logging.getLogger(logger_name).setLevel(level)


class JsonFormatter(logging.Formatter):
    """JSON formatter for structured logging."""
    
    def format(self, record: logging.LogRecord) -> str:
        """Format log record as JSON."""
        
        log_data = {
            "timestamp": datetime.fromtimestamp(record.created).isoformat(),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add exception info if present
        if record.exc_info:
            log_data["exception"] = self.formatException(record.exc_info)
        
        # Add extra fields
        for key, value in record.__dict__.items():
            if key not in ["name", "msg", "args", "levelname", "levelno", "pathname",
                          "filename", "module", "lineno", "funcName", "created",
                          "msecs", "relativeCreated", "thread", "threadName",
                          "processName", "process", "exc_info", "exc_text", "stack_info"]:
                log_data[key] = value
        
        return json.dumps(log_data, default=str)


def get_logger(name: str, **kwargs) -> logging.Logger:
    """
    Get a logger with the specified name and optional context.
    
    Args:
        name: Logger name
        **kwargs: Additional context fields
        
    Returns:
        Configured logger instance
    """
    
    logger = logging.getLogger(name)
    
    # Add context fields if structured logging is enabled
    if structlog.is_configured():
        struct_logger = structlog.get_logger(name)
        if kwargs:
            struct_logger = struct_logger.bind(**kwargs)
        return struct_logger
    
    return logger


class ContextLogger:
    """Logger wrapper that maintains context across log calls."""
    
    def __init__(self, logger: logging.Logger, context: Dict[str, Any]):
        """
        Initialize context logger.
        
        Args:
            logger: Base logger instance
            context: Context dictionary to include in all log messages
        """
        self.logger = logger
        self.context = context
    
    def _log_with_context(self, level: int, message: str, *args, **kwargs):
        """Log message with context."""
        extra = kwargs.get("extra", {})
        extra.update(self.context)
        kwargs["extra"] = extra
        self.logger.log(level, message, *args, **kwargs)
    
    def debug(self, message: str, *args, **kwargs):
        """Log debug message with context."""
        self._log_with_context(logging.DEBUG, message, *args, **kwargs)
    
    def info(self, message: str, *args, **kwargs):
        """Log info message with context."""
        self._log_with_context(logging.INFO, message, *args, **kwargs)
    
    def warning(self, message: str, *args, **kwargs):
        """Log warning message with context."""
        self._log_with_context(logging.WARNING, message, *args, **kwargs)
    
    def error(self, message: str, *args, **kwargs):
        """Log error message with context."""
        self._log_with_context(logging.ERROR, message, *args, **kwargs)
    
    def critical(self, message: str, *args, **kwargs):
        """Log critical message with context."""
        self._log_with_context(logging.CRITICAL, message, *args, **kwargs)


def create_evaluation_logger(
    model_name: str,
    scenario_id: str,
    session_id: str
) -> ContextLogger:
    """
    Create a logger with evaluation context.
    
    Args:
        model_name: Name of the model being evaluated
        scenario_id: ID of the evaluation scenario
        session_id: Unique session identifier
        
    Returns:
        Context logger with evaluation metadata
    """
    
    base_logger = get_logger("evaluation")
    context = {
        "model_name": model_name,
        "scenario_id": scenario_id,
        "session_id": session_id,
        "timestamp": datetime.now().isoformat()
    }
    
    return ContextLogger(base_logger, context)


def log_performance_metrics(
    logger: logging.Logger,
    operation: str,
    duration_ms: float,
    success: bool,
    **metrics
) -> None:
    """
    Log performance metrics in a standardized format.
    
    Args:
        logger: Logger instance
        operation: Operation name
        duration_ms: Operation duration in milliseconds
        success: Whether operation was successful
        **metrics: Additional metrics to log
    """
    
    metrics_data = {
        "operation": operation,
        "duration_ms": duration_ms,
        "success": success,
        **metrics
    }
    
    logger.info(f"Performance metrics for {operation}", extra={"metrics": metrics_data})


# Environment-based configuration
def configure_from_environment() -> None:
    """Configure logging based on environment variables."""
    
    log_level = os.getenv("LOG_LEVEL", "INFO")
    log_format = os.getenv("LOG_FORMAT", "standard")
    log_file = os.getenv("LOG_FILE")
    enable_structured = os.getenv("ENABLE_STRUCTURED_LOGGING", "false").lower() == "true"
    
    setup_logging(
        log_level=log_level,
        log_format=log_format,
        log_file=log_file,
        enable_structured=enable_structured
    )