"""
Logging Configuration for Mental Health AI Chat API

Comprehensive logging setup with structured logging, correlation IDs,
performance monitoring, and security event tracking.
"""

import logging
import logging.handlers
import json
import time
import uuid
from datetime import datetime
from typing import Dict, Any, Optional
from contextvars import ContextVar
import os
from pathlib import Path

# Context variables for request tracking
request_id_context: ContextVar[Optional[str]] = ContextVar('request_id', default=None)
user_id_context: ContextVar[Optional[str]] = ContextVar('user_id', default=None)
session_id_context: ContextVar[Optional[str]] = ContextVar('session_id', default=None)


class StructuredFormatter(logging.Formatter):
    """
    Custom formatter for structured JSON logging
    
    Includes request correlation IDs, user context, and performance metrics
    """
    
    def format(self, record):
        """Format log record as structured JSON"""
        
        # Base log data
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
            "module": record.module,
            "function": record.funcName,
            "line": record.lineno
        }
        
        # Add context information
        request_id = request_id_context.get()
        if request_id:
            log_data["request_id"] = request_id
            
        user_id = user_id_context.get()
        if user_id:
            log_data["user_id"] = user_id
            
        session_id = session_id_context.get()
        if session_id:
            log_data["session_id"] = session_id
        
        # Add exception information
        if record.exc_info:
            log_data["exception"] = {
                "type": record.exc_info[0].__name__,
                "message": str(record.exc_info[1]),
                "traceback": self.formatException(record.exc_info)
            }
        
        # Add extra fields
        if hasattr(record, 'extra_fields'):
            log_data.update(record.extra_fields)
        
        # Performance metrics
        if hasattr(record, 'duration'):
            log_data["duration_ms"] = record.duration
            
        if hasattr(record, 'response_size'):
            log_data["response_size_bytes"] = record.response_size
        
        # Security context
        if hasattr(record, 'security_event'):
            log_data["security"] = record.security_event
        
        return json.dumps(log_data, ensure_ascii=False)


class SecurityLogger:
    """
    Specialized logger for security events
    
    Tracks authentication, authorization, safety alerts, and suspicious activity
    """
    
    def __init__(self):
        self.logger = logging.getLogger("security")
        
    def log_authentication_attempt(self, user_id: str, success: bool, ip_address: str, user_agent: str):
        """Log authentication attempt"""
        self.logger.info(
            f"Authentication {'succeeded' if success else 'failed'} for user {user_id}",
            extra={
                "extra_fields": {
                    "event_type": "authentication",
                    "user_id": user_id,
                    "success": success,
                    "ip_address": ip_address,
                    "user_agent": user_agent
                }
            }
        )
    
    def log_rate_limit_exceeded(self, identifier: str, endpoint: str, ip_address: str):
        """Log rate limit violations"""
        self.logger.warning(
            f"Rate limit exceeded for {identifier} on {endpoint}",
            extra={
                "extra_fields": {
                    "event_type": "rate_limit_exceeded",
                    "identifier": identifier,
                    "endpoint": endpoint,
                    "ip_address": ip_address
                }
            }
        )
    
    def log_safety_alert(self, session_id: str, user_id: str, alert_level: str, message: str):
        """Log safety alerts"""
        self.logger.warning(
            f"Safety alert ({alert_level}): {message}",
            extra={
                "extra_fields": {
                    "event_type": "safety_alert",
                    "session_id": session_id,
                    "user_id": user_id,
                    "alert_level": alert_level,
                    "alert_message": message
                }
            }
        )
    
    def log_suspicious_activity(self, description: str, context: Dict[str, Any]):
        """Log suspicious activity"""
        self.logger.warning(
            f"Suspicious activity detected: {description}",
            extra={
                "extra_fields": {
                    "event_type": "suspicious_activity",
                    "description": description,
                    **context
                }
            }
        )


class PerformanceLogger:
    """
    Logger for performance metrics and monitoring
    
    Tracks response times, model selection performance, and system resource usage
    """
    
    def __init__(self):
        self.logger = logging.getLogger("performance")
    
    def log_request_performance(self, 
                               endpoint: str, 
                               method: str, 
                               duration_ms: float,
                               status_code: int,
                               response_size: int = None):
        """Log HTTP request performance"""
        self.logger.info(
            f"{method} {endpoint} completed in {duration_ms:.2f}ms (status: {status_code})",
            extra={
                "duration": duration_ms,
                "response_size": response_size,
                "extra_fields": {
                    "event_type": "http_request",
                    "endpoint": endpoint,
                    "method": method,
                    "status_code": status_code,
                    "duration_ms": duration_ms,
                    "response_size_bytes": response_size
                }
            }
        )
    
    def log_model_selection_performance(self,
                                      selected_model: str,
                                      duration_ms: float,
                                      confidence: float,
                                      prompt_type: str):
        """Log model selection performance"""
        self.logger.info(
            f"Model selection completed: {selected_model} (confidence: {confidence:.3f}, {duration_ms:.2f}ms)",
            extra={
                "duration": duration_ms,
                "extra_fields": {
                    "event_type": "model_selection",
                    "selected_model": selected_model,
                    "duration_ms": duration_ms,
                    "confidence": confidence,
                    "prompt_type": prompt_type
                }
            }
        )
    
    def log_websocket_performance(self, event_type: str, connection_count: int, message_count: int):
        """Log WebSocket performance metrics"""
        self.logger.info(
            f"WebSocket {event_type}: {connection_count} connections, {message_count} messages",
            extra={
                "extra_fields": {
                    "event_type": "websocket_performance",
                    "websocket_event": event_type,
                    "connection_count": connection_count,
                    "message_count": message_count
                }
            }
        )


class AuditLogger:
    """
    Logger for audit trail and compliance
    
    Tracks all significant system events for compliance and forensic analysis
    """
    
    def __init__(self):
        self.logger = logging.getLogger("audit")
    
    def log_session_created(self, session_id: str, user_id: str, metadata: Dict[str, Any]):
        """Log session creation"""
        self.logger.info(
            f"Session created: {session_id}",
            extra={
                "extra_fields": {
                    "event_type": "session_created",
                    "session_id": session_id,
                    "user_id": user_id,
                    "metadata": metadata
                }
            }
        )
    
    def log_message_exchanged(self, session_id: str, user_id: str, role: str, model_used: str = None):
        """Log message exchange"""
        self.logger.info(
            f"Message exchanged in session {session_id}: {role}",
            extra={
                "extra_fields": {
                    "event_type": "message_exchanged",
                    "session_id": session_id,
                    "user_id": user_id,
                    "role": role,
                    "model_used": model_used
                }
            }
        )
    
    def log_model_switched(self, session_id: str, from_model: str, to_model: str, reason: str):
        """Log model switches"""
        self.logger.info(
            f"Model switched in session {session_id}: {from_model} -> {to_model}",
            extra={
                "extra_fields": {
                    "event_type": "model_switched",
                    "session_id": session_id,
                    "from_model": from_model,
                    "to_model": to_model,
                    "reason": reason
                }
            }
        )
    
    def log_data_export(self, user_id: str, export_type: str, record_count: int):
        """Log data export operations"""
        self.logger.info(
            f"Data export requested: {export_type} ({record_count} records)",
            extra={
                "extra_fields": {
                    "event_type": "data_export",
                    "user_id": user_id,
                    "export_type": export_type,
                    "record_count": record_count
                }
            }
        )


class LoggingConfig:
    """
    Central logging configuration manager
    
    Sets up structured logging with multiple outputs, log rotation,
    and specialized loggers for different event types
    """
    
    def __init__(self, 
                 log_level: str = "INFO",
                 log_dir: str = "logs",
                 enable_file_logging: bool = True,
                 enable_console_logging: bool = True,
                 max_file_size: int = 10 * 1024 * 1024,  # 10MB
                 backup_count: int = 5):
        
        self.log_level = getattr(logging, log_level.upper())
        self.log_dir = Path(log_dir)
        self.enable_file_logging = enable_file_logging
        self.enable_console_logging = enable_console_logging
        self.max_file_size = max_file_size
        self.backup_count = backup_count
        
        # Create log directory
        if self.enable_file_logging:
            self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize specialized loggers
        self.security_logger = SecurityLogger()
        self.performance_logger = PerformanceLogger()
        self.audit_logger = AuditLogger()
    
    def setup_logging(self):
        """Configure all loggers and handlers"""
        
        # Create structured formatter
        formatter = StructuredFormatter()
        
        # Configure root logger
        root_logger = logging.getLogger()
        root_logger.setLevel(self.log_level)
        
        # Clear existing handlers
        root_logger.handlers.clear()
        
        # Console handler
        if self.enable_console_logging:
            console_handler = logging.StreamHandler()
            
            # Use simple format for console in development
            if os.getenv("ENVIRONMENT") == "development":
                console_formatter = logging.Formatter(
                    '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
                )
                console_handler.setFormatter(console_formatter)
            else:
                console_handler.setFormatter(formatter)
            
            console_handler.setLevel(self.log_level)
            root_logger.addHandler(console_handler)
        
        # File handlers
        if self.enable_file_logging:
            self._setup_file_handlers(formatter)
        
        # Set levels for specific loggers
        logging.getLogger("uvicorn").setLevel(logging.INFO)
        logging.getLogger("uvicorn.access").setLevel(logging.INFO)
        logging.getLogger("fastapi").setLevel(logging.INFO)
        
        # Suppress overly verbose loggers
        logging.getLogger("asyncio").setLevel(logging.WARNING)
        logging.getLogger("urllib3").setLevel(logging.WARNING)
        
        logging.info("Logging configuration initialized")
    
    def _setup_file_handlers(self, formatter):
        """Setup rotating file handlers for different log types"""
        
        # Main application log
        app_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "app.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        app_handler.setFormatter(formatter)
        app_handler.setLevel(self.log_level)
        logging.getLogger().addHandler(app_handler)
        
        # Error log (errors and above only)
        error_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "error.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        error_handler.setFormatter(formatter)
        error_handler.setLevel(logging.ERROR)
        logging.getLogger().addHandler(error_handler)
        
        # Security log
        security_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "security.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        security_handler.setFormatter(formatter)
        security_handler.setLevel(logging.INFO)
        logging.getLogger("security").addHandler(security_handler)
        
        # Performance log
        performance_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "performance.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        performance_handler.setFormatter(formatter)
        performance_handler.setLevel(logging.INFO)
        logging.getLogger("performance").addHandler(performance_handler)
        
        # Audit log
        audit_handler = logging.handlers.RotatingFileHandler(
            self.log_dir / "audit.log",
            maxBytes=self.max_file_size,
            backupCount=self.backup_count
        )
        audit_handler.setFormatter(formatter)
        audit_handler.setLevel(logging.INFO)
        logging.getLogger("audit").addHandler(audit_handler)
    
    def set_request_context(self, request_id: str = None, user_id: str = None, session_id: str = None):
        """Set context variables for request tracking"""
        if request_id:
            request_id_context.set(request_id)
        if user_id:
            user_id_context.set(user_id)
        if session_id:
            session_id_context.set(session_id)
    
    def clear_request_context(self):
        """Clear all context variables"""
        request_id_context.set(None)
        user_id_context.set(None)
        session_id_context.set(None)
    
    def get_logger(self, name: str) -> logging.Logger:
        """Get a logger with the specified name"""
        return logging.getLogger(name)


# Context manager for request logging
class RequestLoggingContext:
    """Context manager for request-scoped logging"""
    
    def __init__(self, request_id: str = None, user_id: str = None, session_id: str = None):
        self.request_id = request_id or str(uuid.uuid4())
        self.user_id = user_id
        self.session_id = session_id
        self.start_time = time.time()
    
    def __enter__(self):
        request_id_context.set(self.request_id)
        if self.user_id:
            user_id_context.set(self.user_id)
        if self.session_id:
            session_id_context.set(self.session_id)
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        request_id_context.set(None)
        user_id_context.set(None)
        session_id_context.set(None)
        
        # Log request completion
        duration = (time.time() - self.start_time) * 1000
        logger = logging.getLogger("request")
        
        if exc_type:
            logger.error(
                f"Request {self.request_id} failed after {duration:.2f}ms",
                exc_info=(exc_type, exc_val, exc_tb),
                extra={"duration": duration}
            )
        else:
            logger.info(
                f"Request {self.request_id} completed in {duration:.2f}ms",
                extra={"duration": duration}
            )


# Global logging configuration instance
logging_config = LoggingConfig(
    log_level=os.getenv("LOG_LEVEL", "INFO"),
    log_dir=os.getenv("LOG_DIR", "logs"),
    enable_file_logging=os.getenv("ENABLE_FILE_LOGGING", "true").lower() == "true",
    enable_console_logging=os.getenv("ENABLE_CONSOLE_LOGGING", "true").lower() == "true"
)

# Initialize logging on module import
logging_config.setup_logging()