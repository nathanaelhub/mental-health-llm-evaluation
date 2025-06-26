"""
Configuration utilities and helper functions.

This module provides utility functions for configuration management including
environment variable resolution, configuration merging, and validation helpers.
"""

import os
import re
import copy
from typing import Dict, Any, Union, Optional, List
import logging

logger = logging.getLogger(__name__)


def get_env_value(
    key: str,
    default: Any = None,
    var_type: type = str,
    required: bool = False
) -> Any:
    """
    Get environment variable value with type conversion and validation.
    
    Args:
        key: Environment variable name
        default: Default value if not found
        var_type: Type to convert value to (str, int, float, bool)
        required: Whether the variable is required
        
    Returns:
        Environment variable value converted to specified type
        
    Raises:
        ValueError: If required variable is missing or conversion fails
    """
    value = os.getenv(key)
    
    if value is None:
        if required:
            raise ValueError(f"Required environment variable '{key}' is not set")
        return default
    
    # Handle boolean conversion
    if var_type == bool:
        return value.lower() in ('true', '1', 'yes', 'on', 'enabled')
    
    # Handle list conversion (comma-separated)
    if var_type == list:
        if not value.strip():
            return default if default is not None else []
        return [item.strip() for item in value.split(',') if item.strip()]
    
    # Handle other type conversions
    try:
        if var_type == str:
            return value
        return var_type(value)
    except (ValueError, TypeError) as e:
        logger.error(f"Failed to convert environment variable '{key}' to {var_type}: {e}")
        if required:
            raise ValueError(f"Invalid value for required environment variable '{key}': {value}")
        return default


def resolve_env_vars(config: Dict[str, Any], prefix: str = '') -> Dict[str, Any]:
    """
    Recursively resolve environment variables in configuration.
    
    Supports the following patterns:
    - ${VAR_NAME} - Replace with environment variable value
    - ${VAR_NAME:-default} - Replace with env var or default if not set
    - ${VAR_NAME:default} - Same as above
    
    Args:
        config: Configuration dictionary
        prefix: Prefix for nested keys (for logging)
        
    Returns:
        Configuration with resolved environment variables
    """
    if not isinstance(config, dict):
        return config
    
    resolved_config = {}
    
    for key, value in config.items():
        current_key = f"{prefix}.{key}" if prefix else key
        
        if isinstance(value, dict):
            # Recursively resolve nested dictionaries
            resolved_config[key] = resolve_env_vars(value, current_key)
        elif isinstance(value, list):
            # Handle lists (resolve string items)
            resolved_config[key] = [
                resolve_env_string(item) if isinstance(item, str) else item
                for item in value
            ]
        elif isinstance(value, str):
            # Resolve environment variables in strings
            resolved_config[key] = resolve_env_string(value)
        else:
            # Keep other types as-is
            resolved_config[key] = value
    
    return resolved_config


def resolve_env_string(value: str) -> str:
    """
    Resolve environment variables in a string.
    
    Args:
        value: String that may contain environment variable references
        
    Returns:
        String with environment variables resolved
    """
    if not isinstance(value, str):
        return value
    
    # Pattern to match ${VAR_NAME} or ${VAR_NAME:-default} or ${VAR_NAME:default}
    pattern = r'\$\{([^}]+)\}'
    
    def replace_env_var(match):
        var_expr = match.group(1)
        
        # Check for default value syntax
        if ':-' in var_expr:
            var_name, default_value = var_expr.split(':-', 1)
        elif ':' in var_expr and not var_expr.startswith(':'):
            var_name, default_value = var_expr.split(':', 1)
        else:
            var_name = var_expr
            default_value = ''
        
        # Get environment variable value
        env_value = os.getenv(var_name.strip())
        
        if env_value is not None:
            return env_value
        elif default_value:
            return default_value
        else:
            # Variable not found and no default
            logger.warning(f"Environment variable '{var_name}' not found, keeping placeholder")
            return match.group(0)  # Return original placeholder
    
    return re.sub(pattern, replace_env_var, value)


def merge_configs(base_config: Dict[str, Any], override_config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Recursively merge two configuration dictionaries.
    
    Args:
        base_config: Base configuration dictionary
        override_config: Configuration to merge/override base
        
    Returns:
        Merged configuration dictionary
    """
    if not isinstance(base_config, dict):
        return override_config
    
    if not isinstance(override_config, dict):
        return override_config
    
    # Deep copy base config to avoid modifying original
    merged = copy.deepcopy(base_config)
    
    for key, value in override_config.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            # Recursively merge nested dictionaries
            merged[key] = merge_configs(merged[key], value)
        else:
            # Override or add new key
            merged[key] = copy.deepcopy(value)
    
    return merged


def validate_required_env_vars(required_vars: List[str]) -> Dict[str, bool]:
    """
    Validate that required environment variables are set.
    
    Args:
        required_vars: List of required environment variable names
        
    Returns:
        Dictionary mapping variable names to whether they are set
    """
    validation_results = {}
    
    for var_name in required_vars:
        is_set = bool(os.getenv(var_name))
        validation_results[var_name] = is_set
        
        if not is_set:
            logger.error(f"Required environment variable not set: {var_name}")
    
    return validation_results


def get_config_from_env(config_schema: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
    """
    Build configuration from environment variables based on schema.
    
    Args:
        config_schema: Schema defining environment variable mappings
        
    Returns:
        Configuration built from environment variables
    """
    config = {}
    
    for section_name, section_schema in config_schema.items():
        section_config = {}
        
        for key, key_schema in section_schema.items():
            env_var = key_schema.get('env_var')
            default = key_schema.get('default')
            var_type = key_schema.get('type', str)
            required = key_schema.get('required', False)
            
            if env_var:
                try:
                    value = get_env_value(env_var, default, var_type, required)
                    section_config[key] = value
                except ValueError as e:
                    logger.error(f"Error getting {section_name}.{key}: {e}")
                    if required:
                        raise
        
        if section_config:
            config[section_name] = section_config
    
    return config


def flatten_config(config: Dict[str, Any], separator: str = '.', prefix: str = '') -> Dict[str, Any]:
    """
    Flatten nested configuration dictionary.
    
    Args:
        config: Configuration dictionary to flatten
        separator: Separator for nested keys
        prefix: Prefix for all keys
        
    Returns:
        Flattened configuration dictionary
    """
    flattened = {}
    
    for key, value in config.items():
        new_key = f"{prefix}{separator}{key}" if prefix else key
        
        if isinstance(value, dict):
            flattened.update(flatten_config(value, separator, new_key))
        else:
            flattened[new_key] = value
    
    return flattened


def unflatten_config(flattened_config: Dict[str, Any], separator: str = '.') -> Dict[str, Any]:
    """
    Unflatten configuration dictionary.
    
    Args:
        flattened_config: Flattened configuration dictionary
        separator: Separator used for nested keys
        
    Returns:
        Nested configuration dictionary
    """
    config = {}
    
    for key, value in flattened_config.items():
        parts = key.split(separator)
        current = config
        
        for part in parts[:-1]:
            if part not in current:
                current[part] = {}
            current = current[part]
        
        current[parts[-1]] = value
    
    return config


def mask_sensitive_values(config: Dict[str, Any], sensitive_keys: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Mask sensitive values in configuration for logging/display.
    
    Args:
        config: Configuration dictionary
        sensitive_keys: List of keys to mask (defaults to common sensitive keys)
        
    Returns:
        Configuration with sensitive values masked
    """
    if sensitive_keys is None:
        sensitive_keys = [
            'api_key', 'password', 'secret', 'token', 'key',
            'credential', 'auth', 'private'
        ]
    
    def mask_value(key: str, value: Any) -> Any:
        if isinstance(value, str) and any(sensitive in key.lower() for sensitive in sensitive_keys):
            if len(value) <= 8:
                return '*' * len(value)
            else:
                return value[:4] + '*' * (len(value) - 8) + value[-4:]
        return value
    
    def mask_dict(d: Dict[str, Any]) -> Dict[str, Any]:
        masked = {}
        for k, v in d.items():
            if isinstance(v, dict):
                masked[k] = mask_dict(v)
            else:
                masked[k] = mask_value(k, v)
        return masked
    
    return mask_dict(config)


def validate_config_completeness(config: Dict[str, Any], required_sections: List[str]) -> List[str]:
    """
    Validate that configuration contains all required sections.
    
    Args:
        config: Configuration dictionary
        required_sections: List of required section names
        
    Returns:
        List of missing sections
    """
    missing_sections = []
    
    for section in required_sections:
        if section not in config:
            missing_sections.append(section)
            logger.error(f"Required configuration section missing: {section}")
    
    return missing_sections


def get_config_diff(config1: Dict[str, Any], config2: Dict[str, Any]) -> Dict[str, Any]:
    """
    Get differences between two configuration dictionaries.
    
    Args:
        config1: First configuration
        config2: Second configuration
        
    Returns:
        Dictionary showing differences
    """
    diff = {
        'added': {},
        'removed': {},
        'changed': {}
    }
    
    # Flatten both configs for easier comparison
    flat1 = flatten_config(config1)
    flat2 = flatten_config(config2)
    
    all_keys = set(flat1.keys()) | set(flat2.keys())
    
    for key in all_keys:
        if key in flat1 and key in flat2:
            if flat1[key] != flat2[key]:
                diff['changed'][key] = {'old': flat1[key], 'new': flat2[key]}
        elif key in flat1:
            diff['removed'][key] = flat1[key]
        else:
            diff['added'][key] = flat2[key]
    
    return diff


def create_config_backup(config: Dict[str, Any], backup_path: str) -> bool:
    """
    Create a backup of configuration.
    
    Args:
        config: Configuration to backup
        backup_path: Path to save backup
        
    Returns:
        True if backup was successful
    """
    try:
        import yaml
        import os
        
        # Create backup directory if it doesn't exist
        os.makedirs(os.path.dirname(backup_path), exist_ok=True)
        
        with open(backup_path, 'w', encoding='utf-8') as f:
            yaml.dump(config, f, default_flow_style=False, allow_unicode=True)
        
        logger.info(f"Configuration backup created: {backup_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to create configuration backup: {e}")
        return False


def load_config_from_file(file_path: str) -> Dict[str, Any]:
    """
    Load configuration from YAML or JSON file.
    
    Args:
        file_path: Path to configuration file
        
    Returns:
        Configuration dictionary
        
    Raises:
        ValueError: If file format is unsupported or invalid
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                import yaml
                return yaml.safe_load(f) or {}
            elif file_path.endswith('.json'):
                import json
                return json.load(f)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path}")
    
    except Exception as e:
        logger.error(f"Failed to load configuration from {file_path}: {e}")
        raise


def save_config_to_file(config: Dict[str, Any], file_path: str) -> bool:
    """
    Save configuration to YAML or JSON file.
    
    Args:
        config: Configuration dictionary to save
        file_path: Path to save configuration
        
    Returns:
        True if save was successful
    """
    try:
        import os
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        
        with open(file_path, 'w', encoding='utf-8') as f:
            if file_path.endswith('.yaml') or file_path.endswith('.yml'):
                import yaml
                yaml.dump(config, f, default_flow_style=False, allow_unicode=True, sort_keys=False)
            elif file_path.endswith('.json'):
                import json
                json.dump(config, f, indent=2, ensure_ascii=False)
            else:
                raise ValueError(f"Unsupported configuration file format: {file_path}")
        
        logger.info(f"Configuration saved to: {file_path}")
        return True
        
    except Exception as e:
        logger.error(f"Failed to save configuration to {file_path}: {e}")
        return False


def interpolate_config_values(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Interpolate configuration values within the configuration itself.
    
    Supports ${section.key} syntax for referencing other config values.
    
    Args:
        config: Configuration dictionary
        
    Returns:
        Configuration with interpolated values
    """
    flattened = flatten_config(config)
    
    def interpolate_string(value: str) -> str:
        if not isinstance(value, str):
            return value
        
        pattern = r'\$\{([^}]+)\}'
        
        def replace_config_ref(match):
            ref_key = match.group(1)
            if ref_key in flattened:
                return str(flattened[ref_key])
            else:
                logger.warning(f"Configuration reference not found: {ref_key}")
                return match.group(0)
        
        return re.sub(pattern, replace_config_ref, value)
    
    # Interpolate values
    interpolated_flat = {}
    for key, value in flattened.items():
        if isinstance(value, str):
            interpolated_flat[key] = interpolate_string(value)
        else:
            interpolated_flat[key] = value
    
    return unflatten_config(interpolated_flat)