"""
Configuration loader with validation and environment variable support.

This module provides a centralized configuration loading system that handles
YAML files, environment variables, validation, and runtime configuration access.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import logging
from functools import lru_cache

from .config_schema import ConfigSchema, validate_config
from .config_utils import resolve_env_vars, merge_configs

logger = logging.getLogger(__name__)


class ConfigLoader:
    """
    Configuration loader with validation and environment variable support.
    
    This class handles loading configuration from YAML files, resolving
    environment variables, validating the configuration, and providing
    easy access to configuration values throughout the application.
    """
    
    def __init__(self, config_dir: Optional[str] = None):
        """
        Initialize configuration loader.
        
        Args:
            config_dir: Directory containing configuration files
        """
        self.config_dir = Path(config_dir or "./config")
        self._config: Optional[ConfigSchema] = None
        self._raw_config: Optional[Dict[str, Any]] = None
        self.logger = logging.getLogger(__name__)
        
        # Ensure config directory exists
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(
        self,
        environment: Optional[str] = None,
        config_file: Optional[str] = None,
        override_config: Optional[Dict[str, Any]] = None
    ) -> ConfigSchema:
        """
        Load and validate configuration.
        
        Args:
            environment: Environment name (development/production/testing)
            config_file: Specific config file path
            override_config: Additional configuration to override defaults
            
        Returns:
            Validated configuration schema
            
        Raises:
            ValueError: If configuration is invalid
            FileNotFoundError: If configuration file is not found
        """
        try:
            # Determine environment
            if environment is None:
                environment = os.getenv("ENVIRONMENT", "development")
            
            self.logger.info(f"Loading configuration for environment: {environment}")
            
            # Load base configuration
            if config_file:
                config_path = Path(config_file)
            else:
                # Look for environment-specific config in environments subfolder
                config_path = self.config_dir / "environments" / f"{environment}.yaml"
                if not config_path.exists():
                    # Fallback to old location
                    config_path = self.config_dir / f"{environment}.yaml"
            
            if not config_path.exists():
                raise FileNotFoundError(f"Configuration file not found: {config_path}")
            
            # Load YAML configuration
            raw_config = self._load_yaml_file(config_path)
            
            # Load base configuration if it exists
            base_config_path = self.config_dir / "base.yaml"
            if base_config_path.exists():
                base_config = self._load_yaml_file(base_config_path)
                raw_config = merge_configs(base_config, raw_config)
            
            # Apply overrides
            if override_config:
                raw_config = merge_configs(raw_config, override_config)
            
            # Resolve environment variables
            raw_config = resolve_env_vars(raw_config)
            
            # Store raw config for debugging
            self._raw_config = raw_config
            
            # Validate and create configuration schema
            self._config = validate_config(raw_config)
            
            # Set environment in config if not set
            if hasattr(self._config, 'environment'):
                self._config.environment = environment
            
            self.logger.info("Configuration loaded and validated successfully")
            return self._config
            
        except Exception as e:
            self.logger.error(f"Failed to load configuration: {e}")
            raise
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """
        Load YAML configuration file.
        
        Args:
            file_path: Path to YAML file
            
        Returns:
            Configuration dictionary
        """
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                config = yaml.safe_load(f)
            
            if config is None:
                config = {}
            
            self.logger.debug(f"Loaded YAML config from {file_path}")
            return config
            
        except yaml.YAMLError as e:
            raise ValueError(f"Invalid YAML in {file_path}: {e}")
        except Exception as e:
            raise ValueError(f"Error reading config file {file_path}: {e}")
    
    def get_config(self) -> ConfigSchema:
        """
        Get the current configuration.
        
        Returns:
            Current configuration schema
            
        Raises:
            RuntimeError: If configuration has not been loaded
        """
        if self._config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
        
        return self._config
    
    def get_raw_config(self) -> Dict[str, Any]:
        """
        Get the raw configuration dictionary (for debugging).
        
        Returns:
            Raw configuration dictionary
        """
        if self._raw_config is None:
            raise RuntimeError("Configuration has not been loaded. Call load_config() first.")
        
        return self._raw_config
    
    def reload_config(self) -> ConfigSchema:
        """
        Reload configuration from the same source.
        
        Returns:
            Reloaded configuration schema
        """
        if self._config is None:
            raise RuntimeError("No configuration to reload. Call load_config() first.")
        
        environment = getattr(self._config, 'environment', 'development')
        return self.load_config(environment=environment)
    
    def validate_environment_variables(self) -> Dict[str, bool]:
        """
        Validate that required environment variables are set.
        
        Returns:
            Dictionary mapping variable names to whether they are set
        """
        required_vars = [
            "OPENAI_API_KEY",  # Required for OpenAI
            "DEEPSEEK_MODEL_PATH",  # Required for local DeepSeek
        ]
        
        optional_vars = [
            "OPENAI_ORG_ID",
            "DEEPSEEK_API_KEY",
            "LOG_LEVEL",
            "ENVIRONMENT",
            "DATA_DIR",
            "OUTPUT_DIR",
        ]
        
        validation_results = {}
        
        # Check required variables
        for var in required_vars:
            is_set = bool(os.getenv(var))
            validation_results[var] = is_set
            if not is_set:
                self.logger.warning(f"Required environment variable not set: {var}")
        
        # Check optional variables
        for var in optional_vars:
            is_set = bool(os.getenv(var))
            validation_results[var] = is_set
            if is_set:
                self.logger.debug(f"Optional environment variable set: {var}")
        
        return validation_results
    
    def get_model_config(self, model_name: str) -> Union[Dict[str, Any], None]:
        """
        Get configuration for a specific model.
        
        Args:
            model_name: Name of the model (openai/deepseek)
            
        Returns:
            Model configuration or None if not found
        """
        config = self.get_config()
        
        if model_name.lower() == "openai":
            return config.models.openai
        elif model_name.lower() == "deepseek":
            return config.models.deepseek
        else:
            self.logger.warning(f"Unknown model: {model_name}")
            return None
    
    def get_evaluation_config(self, metric_type: str) -> Union[Dict[str, Any], None]:
        """
        Get configuration for a specific evaluation metric type.
        
        Args:
            metric_type: Type of metric (technical/therapeutic/patient)
            
        Returns:
            Metric configuration or None if not found
        """
        config = self.get_config()
        
        if metric_type.lower() == "technical":
            return config.evaluation.technical
        elif metric_type.lower() == "therapeutic":
            return config.evaluation.therapeutic
        elif metric_type.lower() == "patient":
            return config.evaluation.patient
        else:
            self.logger.warning(f"Unknown metric type: {metric_type}")
            return None
    
    def is_development(self) -> bool:
        """Check if running in development environment."""
        config = self.get_config()
        return getattr(config, 'environment', 'development') == 'development'
    
    def is_production(self) -> bool:
        """Check if running in production environment."""
        config = self.get_config()
        return getattr(config, 'environment', 'development') == 'production'
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration as dictionary."""
        config = self.get_config()
        return {
            "level": config.logging.level.value,
            "format": config.logging.format.value,
            "file_path": config.logging.file_path,
            "max_file_size_mb": config.logging.max_file_size_mb,
            "backup_count": config.logging.backup_count,
            "enable_console": config.logging.enable_console,
            "enable_file": config.logging.enable_file,
            "enable_structured": config.logging.enable_structured,
            "external_loggers": {
                name: level.value for name, level in config.logging.external_loggers.items()
            }
        }
    
    def save_config(self, output_path: str, include_defaults: bool = True) -> None:
        """
        Save current configuration to YAML file.
        
        Args:
            output_path: Path to save configuration
            include_defaults: Whether to include default values
        """
        try:
            config_dict = self._raw_config if self._raw_config else {}
            
            with open(output_path, 'w', encoding='utf-8') as f:
                yaml.dump(
                    config_dict,
                    f,
                    default_flow_style=False,
                    allow_unicode=True,
                    sort_keys=False
                )
            
            self.logger.info(f"Configuration saved to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Failed to save configuration: {e}")
            raise
    
    def create_config_summary(self) -> Dict[str, Any]:
        """
        Create a summary of the current configuration.
        
        Returns:
            Configuration summary
        """
        if self._config is None:
            return {"error": "No configuration loaded"}
        
        config = self._config
        
        return {
            "environment": getattr(config, 'environment', 'unknown'),
            "debug": getattr(config, 'debug', False),
            "version": getattr(config, 'version', 'unknown'),
            "enabled_models": [model.value for model in config.models.enabled_models],
            "evaluation_weights": {
                "technical": config.evaluation.technical_weight,
                "therapeutic": config.evaluation.therapeutic_weight,
                "patient": config.evaluation.patient_weight,
            },
            "experiment": {
                "conversation_count": config.experiment.conversation_count,
                "scenario_suite": config.experiment.scenario_suite,
                "parallel_evaluations": config.experiment.parallel_evaluations,
            },
            "logging": {
                "level": config.logging.level.value,
                "format": config.logging.format.value,
                "file_enabled": config.logging.enable_file,
            },
            "storage": {
                "type": config.storage.type.value,
                "base_dir": config.storage.base_dir,
            }
        }


# Global configuration loader instance
_config_loader: Optional[ConfigLoader] = None


def get_config_loader(config_dir: Optional[str] = None) -> ConfigLoader:
    """
    Get global configuration loader instance.
    
    Args:
        config_dir: Configuration directory (only used on first call)
        
    Returns:
        Global configuration loader instance
    """
    global _config_loader
    
    if _config_loader is None:
        _config_loader = ConfigLoader(config_dir)
    
    return _config_loader


@lru_cache(maxsize=1)
def get_config(
    environment: Optional[str] = None,
    config_file: Optional[str] = None,
    config_dir: Optional[str] = None
) -> ConfigSchema:
    """
    Get configuration with caching.
    
    Args:
        environment: Environment name
        config_file: Specific config file path
        config_dir: Configuration directory
        
    Returns:
        Validated configuration schema
    """
    loader = get_config_loader(config_dir)
    return loader.load_config(environment=environment, config_file=config_file)


def reload_config() -> ConfigSchema:
    """
    Reload configuration and clear cache.
    
    Returns:
        Reloaded configuration schema
    """
    # Clear the cache
    get_config.cache_clear()
    
    # Get fresh config
    loader = get_config_loader()
    return loader.reload_config()


def validate_environment() -> bool:
    """
    Validate that the current environment is properly configured.
    
    Returns:
        True if environment is valid
    """
    try:
        loader = get_config_loader()
        env_vars = loader.validate_environment_variables()
        
        # Check if any required variables are missing
        required_vars = ["OPENAI_API_KEY", "DEEPSEEK_MODEL_PATH"]
        missing_required = [var for var in required_vars if not env_vars.get(var, False)]
        
        if missing_required:
            logger.error(f"Missing required environment variables: {missing_required}")
            return False
        
        # Try to load configuration
        config = get_config()
        logger.info(f"Environment validation successful for: {config.environment}")
        return True
        
    except Exception as e:
        logger.error(f"Environment validation failed: {e}")
        return False


def get_model_config(model_name: str) -> Union[Dict[str, Any], None]:
    """
    Get model configuration by name.
    
    Args:
        model_name: Model name (openai/deepseek)
        
    Returns:
        Model configuration or None
    """
    loader = get_config_loader()
    return loader.get_model_config(model_name)


def is_development() -> bool:
    """Check if running in development mode."""
    try:
        config = get_config()
        return config.environment == "development"
    except:
        return True  # Default to development if config fails


def is_production() -> bool:
    """Check if running in production mode."""
    try:
        config = get_config()
        return config.environment == "production"
    except:
        return False