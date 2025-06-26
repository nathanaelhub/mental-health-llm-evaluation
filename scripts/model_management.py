#!/usr/bin/env python3
"""
Model Management Script

Utility script for managing models in the Mental Health LLM Evaluation framework.
Provides commands for listing, testing, and managing model implementations.

Usage:
    python scripts/model_management.py list                    # List all models
    python scripts/model_management.py test gpt-4              # Test specific model
    python scripts/model_management.py test-all                # Test all available models
    python scripts/model_management.py info claude-3           # Get model info
    python scripts/model_management.py validate-config config.yaml  # Validate config
"""

import argparse
import asyncio
import json
import sys
from pathlib import Path
from typing import Dict, Any, Optional

import yaml
from tabulate import tabulate

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import (
    get_model_registry, 
    get_model_factory, 
    ModelType, 
    ModelProvider,
    create_model
)
from utils.logging_config import setup_logging, get_logger


def list_models(show_unavailable: bool = False) -> None:
    """List all registered models."""
    registry = get_model_registry()
    models_info = registry.list_models()
    
    table_data = []
    for name, info in models_info.items():
        if not show_unavailable and not info["available"]:
            continue
            
        status = "✅ Available" if info["available"] else "❌ Unavailable"
        provider = info["provider"]
        model_type = info["type"]
        description = info["description"]
        
        table_data.append([name, provider, model_type, status, description])
    
    headers = ["Model Name", "Provider", "Type", "Status", "Description"]
    print("\n📋 Registered Models:")
    print(tabulate(table_data, headers=headers, tablefmt="grid"))
    
    if not show_unavailable:
        unavailable_count = len([m for m in models_info.values() if not m["available"]])
        if unavailable_count > 0:
            print(f"\n💡 Tip: Use --show-unavailable to see {unavailable_count} unavailable models")


async def test_model(model_name: str, config: Optional[Dict[str, Any]] = None) -> bool:
    """Test a specific model."""
    logger = get_logger(__name__)
    
    print(f"\n🧪 Testing model: {model_name}")
    
    try:
        # Create model instance
        model = create_model(model_name, config)
        if not model:
            print(f"❌ Failed to create model: {model_name}")
            return False
        
        print(f"✅ Model created successfully")
        
        # Validate configuration
        print("🔧 Validating configuration...")
        if model.validate_configuration():
            print("✅ Configuration valid")
        else:
            print("❌ Configuration validation failed")
            return False
        
        # Run health check
        print("🏥 Running health check...")
        is_healthy = await model.health_check()
        
        if is_healthy:
            print("✅ Health check passed")
            
            # Get model info
            info = model.get_model_info()
            print(f"\n📊 Model Information:")
            print(f"  Provider: {info.get('provider', 'Unknown')}")
            print(f"  Type: {info.get('type', 'Unknown')}")
            print(f"  Max Context: {info.get('max_context_length', 'Unknown')} tokens")
            print(f"  Streaming: {'Yes' if info.get('supports_streaming', False) else 'No'}")
            print(f"  Function Calling: {'Yes' if info.get('supports_function_calling', False) else 'No'}")
            
            return True
        else:
            print("❌ Health check failed")
            return False
            
    except Exception as e:
        print(f"❌ Error testing model: {e}")
        logger.error(f"Model test failed: {e}")
        return False


async def test_all_models() -> Dict[str, bool]:
    """Test all available models."""
    registry = get_model_registry()
    available_models = registry.get_available_models()
    
    print(f"\n🧪 Testing {len(available_models)} available models...")
    
    results = {}
    for model_reg in available_models:
        print(f"\n{'='*50}")
        success = await test_model(model_reg.name)
        results[model_reg.name] = success
    
    # Summary
    print(f"\n{'='*50}")
    print("📊 Test Summary:")
    
    passed = len([r for r in results.values() if r])
    total = len(results)
    
    for model_name, success in results.items():
        status = "✅ PASS" if success else "❌ FAIL"
        print(f"  {model_name}: {status}")
    
    print(f"\nOverall: {passed}/{total} models passed")
    
    return results


def get_model_info(model_name: str) -> None:
    """Get detailed information about a model."""
    registry = get_model_registry()
    factory = get_model_factory()
    
    # Get registration info
    registration = registry.get_model(model_name)
    if not registration:
        print(f"❌ Model not found: {model_name}")
        return
    
    print(f"\n📋 Model Information: {model_name}")
    print(f"{'='*50}")
    
    print(f"Provider: {registration.provider.value}")
    print(f"Type: {registration.model_type.value}")
    print(f"Description: {registration.description}")
    print(f"Module: {registration.module_path}")
    print(f"Available: {'Yes' if registration.is_available else 'No'}")
    print(f"Requirements: {', '.join(registration.requirements) if registration.requirements else 'None'}")
    
    print(f"\nDefault Configuration:")
    if registration.default_config:
        for key, value in registration.default_config.items():
            print(f"  {key}: {value}")
    else:
        print("  No default configuration")
    
    # Get capabilities if available
    capabilities = factory.get_model_info(model_name)
    if capabilities:
        print(f"\nCapabilities:")
        for key, value in capabilities.items():
            if key != "config":  # Skip config as we already showed it
                print(f"  {key}: {value}")


def validate_config(config_path: str) -> bool:
    """Validate model configurations in a config file."""
    factory = get_model_factory()
    
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        print(f"\n🔧 Validating configuration: {config_path}")
        
        # Validate models configuration
        validation_results = factory.validate_config(config)
        
        print(f"\n📊 Validation Results:")
        
        all_valid = True
        for model_name, is_valid in validation_results.items():
            status = "✅ Valid" if is_valid else "❌ Invalid"
            print(f"  {model_name}: {status}")
            if not is_valid:
                all_valid = False
        
        if all_valid:
            print(f"\n✅ All model configurations are valid!")
        else:
            print(f"\n❌ Some model configurations are invalid")
        
        return all_valid
        
    except Exception as e:
        print(f"❌ Error validating config: {e}")
        return False


def show_providers_and_types() -> None:
    """Show available providers and model types."""
    print("\n📋 Available Providers:")
    for provider in ModelProvider:
        print(f"  - {provider.value}")
    
    print("\n📋 Model Types:")
    for model_type in ModelType:
        print(f"  - {model_type.value}")


def create_sample_config() -> None:
    """Create a sample configuration file."""
    registry = get_model_registry()
    available_models = registry.get_available_models()
    
    config = {
        "experiment": {
            "name": "Sample Mental Health LLM Evaluation",
            "description": "Sample configuration with all available models",
            "version": "2.0.0",
            "allow_partial_models": True
        },
        "models": {
            "cloud": [],
            "local": []
        }
    }
    
    # Add available models to config
    for model_reg in available_models:
        model_config = {
            "name": model_reg.name,
            "provider": model_reg.provider.value,
            "enabled": False,  # Disabled by default
            **model_reg.default_config
        }
        
        if model_reg.model_type == ModelType.CLOUD:
            config["models"]["cloud"].append(model_config)
        else:
            config["models"]["local"].append(model_config)
    
    # Add other required sections
    config.update({
        "scenarios": {
            "directory": "data/scenarios",
            "include": [],
            "category": [],
            "severity": []
        },
        "evaluation": {
            "conversations_per_scenario": 10,
            "max_conversation_turns": 20,
            "enable_safety_monitoring": True,
            "enable_metrics_collection": True
        },
        "output": {
            "base_directory": "./experiments",
            "conversations": "conversations",
            "evaluations": "evaluations",
            "results": "results"
        }
    })
    
    output_file = PROJECT_ROOT / "config" / "sample_experiment.yaml"
    output_file.parent.mkdir(exist_ok=True)
    
    with open(output_file, 'w') as f:
        yaml.dump(config, f, default_flow_style=False, sort_keys=False)
    
    print(f"\n✅ Created sample configuration: {output_file}")
    print("Edit the file to enable desired models and customize settings.")


async def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Model Management for Mental Health LLM Evaluation",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all available models
  python scripts/model_management.py list
  
  # Test a specific model
  python scripts/model_management.py test gpt-4
  
  # Test all models
  python scripts/model_management.py test-all
  
  # Get model information
  python scripts/model_management.py info claude-3
  
  # Validate configuration
  python scripts/model_management.py validate-config config/experiment.yaml
  
  # Show providers and types
  python scripts/model_management.py providers
  
  # Create sample configuration
  python scripts/model_management.py create-config
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all registered models')
    list_parser.add_argument('--show-unavailable', action='store_true', 
                           help='Show unavailable models')
    
    # Test command
    test_parser = subparsers.add_parser('test', help='Test a specific model')
    test_parser.add_argument('model_name', help='Name of the model to test')
    test_parser.add_argument('--config', help='JSON config overrides')
    
    # Test all command
    subparsers.add_parser('test-all', help='Test all available models')
    
    # Info command
    info_parser = subparsers.add_parser('info', help='Get model information')
    info_parser.add_argument('model_name', help='Name of the model')
    
    # Validate config command
    validate_parser = subparsers.add_parser('validate-config', help='Validate configuration file')
    validate_parser.add_argument('config_path', help='Path to configuration file')
    
    # Providers command
    subparsers.add_parser('providers', help='Show available providers and types')
    
    # Create config command
    subparsers.add_parser('create-config', help='Create sample configuration file')
    
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    
    if not args.command:
        parser.print_help()
        return 1
    
    try:
        if args.command == 'list':
            list_models(args.show_unavailable)
            
        elif args.command == 'test':
            config = None
            if args.config:
                config = json.loads(args.config)
            success = await test_model(args.model_name, config)
            return 0 if success else 1
            
        elif args.command == 'test-all':
            results = await test_all_models()
            failed = [name for name, success in results.items() if not success]
            return 0 if not failed else 1
            
        elif args.command == 'info':
            get_model_info(args.model_name)
            
        elif args.command == 'validate-config':
            success = validate_config(args.config_path)
            return 0 if success else 1
            
        elif args.command == 'providers':
            show_providers_and_types()
            
        elif args.command == 'create-config':
            create_sample_config()
            
        return 0
        
    except KeyboardInterrupt:
        print("\n❌ Interrupted by user")
        return 130
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(asyncio.run(main()))