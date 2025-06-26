#!/usr/bin/env python3
"""
Configuration validation script for mental health LLM evaluation.

This script validates configuration files, checks environment variables,
and provides detailed feedback on configuration issues.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, Any

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from config.config_loader import ConfigLoader, validate_environment
from config.config_schema import validate_config
from config.config_utils import mask_sensitive_values, get_env_value


def main():
    """Main validation function."""
    parser = argparse.ArgumentParser(
        description="Validate Mental Health LLM Evaluation Configuration"
    )
    parser.add_argument(
        "--environment",
        default="development",
        choices=["development", "production", "testing"],
        help="Environment to validate"
    )
    parser.add_argument(
        "--config-file",
        help="Specific config file to validate"
    )
    parser.add_argument(
        "--check-env",
        action="store_true",
        help="Check environment variables"
    )
    parser.add_argument(
        "--show-config",
        action="store_true", 
        help="Show loaded configuration (sensitive values masked)"
    )
    parser.add_argument(
        "--output-summary",
        help="Output configuration summary to file"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )
    
    args = parser.parse_args()
    
    print("🔍 Mental Health LLM Configuration Validator")
    print("=" * 50)
    
    success = True
    
    # Initialize configuration loader
    try:
        loader = ConfigLoader()
        
        if args.verbose:
            print(f"📂 Config directory: {loader.config_dir}")
            print(f"🌍 Environment: {args.environment}")
    
    except Exception as e:
        print(f"❌ Failed to initialize config loader: {e}")
        return 1
    
    # Validate environment variables
    if args.check_env:
        print("\n🔐 Checking Environment Variables...")
        print("-" * 30)
        
        try:
            env_valid = validate_environment()
            
            # Check specific variables
            env_vars = loader.validate_environment_variables()
            
            required_vars = ["OPENAI_API_KEY", "DEEPSEEK_MODEL_PATH"]
            optional_vars = ["OPENAI_ORG_ID", "DEEPSEEK_API_KEY", "LOG_LEVEL"]
            
            print("Required variables:")
            for var in required_vars:
                status = "✅" if env_vars.get(var, False) else "❌"
                print(f"  {status} {var}")
                if not env_vars.get(var, False):
                    success = False
            
            print("\nOptional variables:")
            for var in optional_vars:
                status = "✅" if env_vars.get(var, False) else "⚠️"
                print(f"  {status} {var}")
            
            if not env_valid:
                success = False
                
        except Exception as e:
            print(f"❌ Environment validation failed: {e}")
            success = False
    
    # Load and validate configuration
    print(f"\n📋 Loading Configuration ({args.environment})...")
    print("-" * 30)
    
    try:
        config = loader.load_config(
            environment=args.environment,
            config_file=args.config_file
        )
        
        print("✅ Configuration loaded successfully")
        
        # Show configuration summary
        summary = loader.create_config_summary()
        print(f"✅ Environment: {summary['environment']}")
        print(f"✅ Debug mode: {summary['debug']}")
        print(f"✅ Enabled models: {', '.join(summary['enabled_models'])}")
        print(f"✅ Scenario suite: {summary['experiment']['scenario_suite']}")
        print(f"✅ Storage type: {summary['storage']['type']}")
        
    except Exception as e:
        print(f"❌ Configuration validation failed: {e}")
        success = False
        config = None
    
    # Show detailed configuration if requested
    if args.show_config and config is not None:
        print("\n📄 Configuration Details...")
        print("-" * 30)
        
        try:
            raw_config = loader.get_raw_config()
            masked_config = mask_sensitive_values(raw_config)
            
            # Show key sections
            sections_to_show = ["models", "evaluation", "experiment", "logging"]
            
            for section in sections_to_show:
                if section in masked_config:
                    print(f"\n{section.upper()}:")
                    _print_config_section(masked_config[section], indent=2)
                    
        except Exception as e:
            print(f"⚠️ Could not display configuration: {e}")
    
    # Validate specific configuration requirements
    if config is not None:
        print("\n🧪 Validating Configuration Requirements...")
        print("-" * 30)
        
        validation_issues = []
        
        # Check model configuration
        try:
            for model_type in config.models.enabled_models:
                model_config = loader.get_model_config(model_type.value)
                if model_config is None:
                    validation_issues.append(f"Model configuration missing: {model_type.value}")
                else:
                    print(f"✅ Model config valid: {model_type.value}")
        except Exception as e:
            validation_issues.append(f"Model validation error: {e}")
        
        # Check evaluation thresholds
        try:
            eval_config = config.evaluation
            thresholds = [
                eval_config.minimum_viable_threshold,
                eval_config.research_acceptable_threshold,
                eval_config.production_ready_threshold,
                eval_config.clinical_ready_threshold
            ]
            
            if thresholds == sorted(thresholds):
                print("✅ Evaluation thresholds properly ordered")
            else:
                validation_issues.append("Evaluation thresholds not in ascending order")
                
        except Exception as e:
            validation_issues.append(f"Threshold validation error: {e}")
        
        # Check directory paths
        try:
            paths_to_check = [
                ("data", config.storage.base_dir),
                ("scenarios", config.scenario.scenarios_dir),
                ("output", config.output.base_dir)
            ]
            
            for name, path in paths_to_check:
                path_obj = Path(path)
                if path_obj.exists():
                    print(f"✅ Directory exists: {name} ({path})")
                else:
                    print(f"⚠️ Directory missing: {name} ({path})")
                    
        except Exception as e:
            validation_issues.append(f"Directory validation error: {e}")
        
        # Report validation issues
        if validation_issues:
            print(f"\n❌ Found {len(validation_issues)} validation issues:")
            for issue in validation_issues:
                print(f"  • {issue}")
            success = False
        else:
            print("\n✅ All validation checks passed")
    
    # Save configuration summary if requested
    if args.output_summary and config is not None:
        try:
            summary = loader.create_config_summary()
            
            import json
            with open(args.output_summary, 'w') as f:
                json.dump(summary, f, indent=2, default=str)
            
            print(f"\n💾 Configuration summary saved to: {args.output_summary}")
            
        except Exception as e:
            print(f"⚠️ Failed to save summary: {e}")
    
    # Final result
    print("\n" + "=" * 50)
    if success:
        print("🎉 Configuration validation PASSED")
        return 0
    else:
        print("💥 Configuration validation FAILED")
        return 1


def _print_config_section(section: Dict[str, Any], indent: int = 0):
    """Print configuration section with indentation."""
    spaces = " " * indent
    
    for key, value in section.items():
        if isinstance(value, dict):
            print(f"{spaces}{key}:")
            _print_config_section(value, indent + 2)
        elif isinstance(value, list):
            print(f"{spaces}{key}: [{', '.join(str(v) for v in value)}]")
        else:
            print(f"{spaces}{key}: {value}")


if __name__ == "__main__":
    sys.exit(main())