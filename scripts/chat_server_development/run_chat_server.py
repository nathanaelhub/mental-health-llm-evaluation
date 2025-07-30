#!/usr/bin/env python3
"""
Mental Health AI Chat Server Launcher

Production-ready server launcher with comprehensive configuration,
environment detection, and graceful shutdown handling.
"""

import os
import sys
import signal
import asyncio
import logging
from pathlib import Path
from typing import Optional, Dict, Any
import argparse
import json

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import uvicorn
    from src.api.main import app
    from src.api.logging_config import logging_config, RequestLoggingContext
except ImportError as e:
    print(f"❌ Import error: {e}")
    print("💡 Make sure to install dependencies: pip install -r requirements_api.txt")
    sys.exit(1)


class ChatServerConfig:
    """Server configuration management"""
    
    def __init__(self):
        self.host = os.getenv("HOST", "0.0.0.0")
        self.port = int(os.getenv("PORT", "8000"))
        self.workers = int(os.getenv("WORKERS", "1"))
        self.reload = os.getenv("RELOAD", "false").lower() == "true"
        self.log_level = os.getenv("LOG_LEVEL", "info").lower()
        self.environment = os.getenv("ENVIRONMENT", "development")
        
        # SSL configuration
        self.ssl_keyfile = os.getenv("SSL_KEYFILE")
        self.ssl_certfile = os.getenv("SSL_CERTFILE")
        
        # Advanced configuration
        self.access_log = os.getenv("ACCESS_LOG", "true").lower() == "true"
        self.proxy_headers = os.getenv("PROXY_HEADERS", "false").lower() == "true"
        self.forwarded_allow_ips = os.getenv("FORWARDED_ALLOW_IPS", "*")
        
        # Performance tuning
        self.max_requests = int(os.getenv("MAX_REQUESTS", "0"))
        self.max_requests_jitter = int(os.getenv("MAX_REQUESTS_JITTER", "0"))
        self.timeout_keep_alive = int(os.getenv("TIMEOUT_KEEP_ALIVE", "5"))
        
        # Database configuration
        self.database_url = os.getenv("DATABASE_URL", "sqlite:///data/sessions.db")
        
        # Redis configuration (optional)
        self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379")
        
        # Model configuration
        self.enable_model_caching = os.getenv("ENABLE_MODEL_CACHING", "true").lower() == "true"
        self.model_cache_size = int(os.getenv("MODEL_CACHE_SIZE", "1000"))
    
    def validate(self) -> bool:
        """Validate configuration"""
        try:
            # Check port range
            if not (1 <= self.port <= 65535):
                raise ValueError(f"Invalid port: {self.port}")
            
            # Check SSL configuration
            if self.ssl_certfile and not Path(self.ssl_certfile).exists():
                raise ValueError(f"SSL certificate file not found: {self.ssl_certfile}")
            
            if self.ssl_keyfile and not Path(self.ssl_keyfile).exists():
                raise ValueError(f"SSL key file not found: {self.ssl_keyfile}")
            
            # Check log level
            valid_log_levels = ["critical", "error", "warning", "info", "debug", "trace"]
            if self.log_level not in valid_log_levels:
                raise ValueError(f"Invalid log level: {self.log_level}")
            
            return True
            
        except Exception as e:
            print(f"❌ Configuration validation failed: {e}")
            return False
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary for uvicorn"""
        config = {
            "app": app,
            "host": self.host,
            "port": self.port,
            "reload": self.reload,
            "log_level": self.log_level,
            "access_log": self.access_log,
            "proxy_headers": self.proxy_headers,
            "forwarded_allow_ips": self.forwarded_allow_ips,
            "timeout_keep_alive": self.timeout_keep_alive,
        }
        
        # Add SSL configuration if provided
        if self.ssl_certfile and self.ssl_keyfile:
            config.update({
                "ssl_keyfile": self.ssl_keyfile,
                "ssl_certfile": self.ssl_certfile
            })
        
        # Add performance configuration
        if self.max_requests > 0:
            config["max_requests"] = self.max_requests
            
        if self.max_requests_jitter > 0:
            config["max_requests_jitter"] = self.max_requests_jitter
        
        return config
    
    def print_config(self):
        """Print current configuration"""
        print("🔧 Server Configuration:")
        print(f"   Host: {self.host}")
        print(f"   Port: {self.port}")
        print(f"   Workers: {self.workers}")
        print(f"   Environment: {self.environment}")
        print(f"   Log Level: {self.log_level}")
        print(f"   Reload: {self.reload}")
        print(f"   SSL: {'Enabled' if self.ssl_certfile else 'Disabled'}")
        print(f"   Database: {self.database_url}")
        print(f"   Redis: {self.redis_url}")


class ChatServer:
    """Mental Health AI Chat Server"""
    
    def __init__(self, config: ChatServerConfig):
        self.config = config
        self.server = None
        self.logger = logging.getLogger(__name__)
        
    async def startup(self):
        """Server startup tasks"""
        try:
            self.logger.info("🚀 Starting Mental Health AI Chat Server...")
            
            # Create necessary directories
            os.makedirs("data", exist_ok=True)
            os.makedirs("logs", exist_ok=True)
            os.makedirs("results/development", exist_ok=True)
            
            # Validate configuration
            if not self.config.validate():
                raise Exception("Configuration validation failed")
            
            # Print configuration
            self.config.print_config()
            
            self.logger.info("✅ Server startup completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Server startup failed: {e}")
            raise
    
    async def shutdown(self):
        """Server shutdown tasks"""
        try:
            self.logger.info("🛑 Shutting down Mental Health AI Chat Server...")
            
            # Graceful shutdown tasks would go here
            # (closing database connections, saving state, etc.)
            
            self.logger.info("✅ Server shutdown completed successfully")
            
        except Exception as e:
            self.logger.error(f"❌ Error during server shutdown: {e}")
    
    def run(self):
        """Run the server"""
        try:
            # Setup logging
            logging_config.setup_logging()
            
            # Startup tasks
            asyncio.run(self.startup())
            
            # Setup signal handlers
            self._setup_signal_handlers()
            
            # Start server
            uvicorn_config = self.config.to_dict()
            
            if self.config.workers > 1:
                # Multi-worker mode
                uvicorn_config["workers"] = self.config.workers
                self.logger.info(f"🔄 Starting server with {self.config.workers} workers...")
            else:
                # Single worker mode
                self.logger.info("🔄 Starting server in single worker mode...")
            
            # Run server
            uvicorn.run(**uvicorn_config)
            
        except KeyboardInterrupt:
            self.logger.info("🛑 Server stopped by user")
        except Exception as e:
            self.logger.error(f"❌ Server error: {e}", exc_info=True)
            sys.exit(1)
        finally:
            # Cleanup
            asyncio.run(self.shutdown())
    
    def _setup_signal_handlers(self):
        """Setup graceful shutdown signal handlers"""
        def signal_handler(signum, frame):
            self.logger.info(f"🛑 Received signal {signum}, initiating graceful shutdown...")
            # Uvicorn will handle the graceful shutdown
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)


def validate_environment():
    """Validate that required dependencies and configurations are available"""
    errors = []
    warnings = []
    
    # Check for required directories
    required_dirs = [
        "src/models",
        "src/evaluation", 
        "config/scenarios",
        "temp"
    ]
    
    for dir_path in required_dirs:
        if not Path(dir_path).exists():
            errors.append(f"Required directory missing: {dir_path}")
    
    # Check for model client implementations
    model_files = [
        "src/models/openai_client.py",
        "src/models/deepseek_client.py"
    ]
    
    available_models = []
    for model_file in model_files:
        if Path(model_file).exists():
            model_name = Path(model_file).stem.replace('_client', '')
            available_models.append(model_name)
        else:
            warnings.append(f"Model client not found: {model_file}")
    
    if not available_models:
        errors.append("No model clients available. At least one model client is required.")
    
    # Check for scenarios file
    scenarios_file = Path("config/scenarios/main_scenarios.yaml")
    if not scenarios_file.exists():
        warnings.append("Main scenarios file not found. Some evaluation features may not work.")
    
    # Print validation results
    if errors:
        print("❌ Environment validation failed:")
        for error in errors:
            print(f"   • {error}")
        return False, available_models
    
    if warnings:
        print("⚠️  Environment warnings:")
        for warning in warnings:
            print(f"   • {warning}")
    
    print("✅ Environment validation passed")
    print(f"📋 Available models: {', '.join(available_models)}")
    return True, available_models


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Mental Health AI Chat Server",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/run_chat_server.py                          # Development mode
  python scripts/run_chat_server.py --port 8080              # Custom port
  python scripts/run_chat_server.py --environment production # Production mode
  python scripts/run_chat_server.py --workers 4              # Multi-worker mode
  python scripts/run_chat_server.py --ssl-cert cert.pem --ssl-key key.pem  # HTTPS
        """
    )
    
    parser.add_argument(
        "--host", 
        default=os.getenv("HOST", "0.0.0.0"),
        help="Host to bind to (default: 0.0.0.0)"
    )
    
    parser.add_argument(
        "--port", 
        type=int,
        default=int(os.getenv("PORT", "8000")),
        help="Port to bind to (default: 8000)"
    )
    
    parser.add_argument(
        "--workers",
        type=int,
        default=int(os.getenv("WORKERS", "1")),
        help="Number of worker processes (default: 1)"
    )
    
    parser.add_argument(
        "--environment",
        choices=["development", "staging", "production"],
        default=os.getenv("ENVIRONMENT", "development"),
        help="Environment mode (default: development)"
    )
    
    parser.add_argument(
        "--log-level",
        choices=["critical", "error", "warning", "info", "debug", "trace"],
        default=os.getenv("LOG_LEVEL", "info"),
        help="Log level (default: info)"
    )
    
    parser.add_argument(
        "--reload",
        action="store_true",
        help="Enable auto-reload (development only)"
    )
    
    parser.add_argument(
        "--ssl-cert",
        help="SSL certificate file path"
    )
    
    parser.add_argument(
        "--ssl-key",
        help="SSL private key file path"
    )
    
    parser.add_argument(
        "--check-config",
        action="store_true",
        help="Check configuration and exit"
    )
    
    args = parser.parse_args()
    
    # Override environment variables with command line arguments
    if args.host:
        os.environ["HOST"] = args.host
    if args.port:
        os.environ["PORT"] = str(args.port)
    if args.workers:
        os.environ["WORKERS"] = str(args.workers)
    if args.environment:
        os.environ["ENVIRONMENT"] = args.environment
    if args.log_level:
        os.environ["LOG_LEVEL"] = args.log_level
    if args.reload:
        os.environ["RELOAD"] = "true"
    if args.ssl_cert:
        os.environ["SSL_CERTFILE"] = args.ssl_cert
    if args.ssl_key:
        os.environ["SSL_KEYFILE"] = args.ssl_key
    
    # Create configuration
    config = ChatServerConfig()
    
    if args.check_config:
        print("🔍 Checking configuration...")
        config.print_config()
        
        if config.validate():
            print("✅ Configuration is valid")
        else:
            print("❌ Configuration validation failed")
            sys.exit(1)
        return
    
    # Start server
    server = ChatServer(config)
    server.run()


if __name__ == "__main__":
    sys.exit(main())