#!/usr/bin/env python3
"""
Setup Experiment Script

Initializes a new experiment run with proper configuration validation,
directory setup, and model connection testing.

Usage:
    python scripts/setup_experiment.py --config config/experiment.yaml
    python scripts/setup_experiment.py --name "my_experiment" --models openai,deepseek
    python scripts/setup_experiment.py --dry-run  # Test without creating files
"""

import argparse
import asyncio
import json
import logging
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import yaml
from tqdm import tqdm

# Add project root to Python path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models import get_model_factory, get_model_registry, ModelFactory, ModelRegistry
from scenarios.scenario_loader import ScenarioLoader
from utils.logging_config import setup_logging, get_logger


class ExperimentSetup:
    """Handles experiment initialization and validation."""
    
    def __init__(self, config_path: Optional[str] = None, dry_run: bool = False):
        self.config_path = config_path
        self.dry_run = dry_run
        self.logger = get_logger(__name__)
        self.experiment_id = self._generate_experiment_id()
        self.config = {}
        self.models = {}
        self.scenarios = []
        self.model_factory = get_model_factory()
        self.model_registry = get_model_registry()
        
    def _generate_experiment_id(self) -> str:
        """Generate unique experiment ID."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        unique_id = str(uuid.uuid4())[:8]
        return f"exp_{timestamp}_{unique_id}"
    
    def load_configuration(self, config_path: Optional[str] = None) -> Dict:
        """Load and validate experiment configuration."""
        self.logger.info("Loading experiment configuration...")
        
        if config_path:
            config_file = Path(config_path)
        else:
            # Try default locations
            config_file = None
            for path in [
                PROJECT_ROOT / "config" / "experiment.yaml",
                PROJECT_ROOT / "config" / "config.yaml",
                PROJECT_ROOT / "config.yaml"
            ]:
                if path.exists():
                    config_file = path
                    break
        
        if not config_file or not config_file.exists():
            raise FileNotFoundError(
                f"Configuration file not found. Please provide --config argument or "
                f"create config/experiment.yaml"
            )
        
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.logger.info(f"Loaded configuration from {config_file}")
        return self.config
    
    def validate_configuration(self) -> List[str]:
        """Validate configuration and return list of issues."""
        issues = []
        
        # Required sections
        required_sections = ["experiment", "models", "evaluation", "output"]
        for section in required_sections:
            if section not in self.config:
                issues.append(f"Missing required section: {section}")
        
        # Experiment validation
        if "experiment" in self.config:
            exp_config = self.config["experiment"]
            if "name" not in exp_config:
                issues.append("Missing experiment.name")
            if "description" not in exp_config:
                issues.append("Missing experiment.description")
        
        # Models validation
        if "models" in self.config:
            models_config = self.config["models"]
            # Check for both old and new config formats
            enabled_models = []
            
            if "cloud" in models_config or "local" in models_config:
                # New format
                for model_type in ["cloud", "local"]:
                    if model_type in models_config:
                        type_models = models_config[model_type]
                        if isinstance(type_models, list):
                            enabled_models.extend([m.get("name") for m in type_models if m.get("enabled", False)])
                        elif isinstance(type_models, dict):
                            enabled_models.extend([name for name, config in type_models.items() if config.get("enabled", False)])
            else:
                # Legacy format
                enabled_models = [model for model, config in models_config.items() if config.get("enabled", False)]
            
            if not enabled_models:
                issues.append("No models enabled in configuration")
            else:
                # Validate that enabled models exist in registry
                available_models = [reg.name for reg in self.model_registry.get_available_models()]
                invalid_models = [name for name in enabled_models if name not in available_models]
                if invalid_models:
                    issues.append(f"Unknown models in configuration: {', '.join(invalid_models)}")
                    issues.append(f"Available models: {', '.join(available_models)}")
        
        # Output validation
        if "output" in self.config:
            output_config = self.config["output"]
            required_outputs = ["base_directory", "conversations", "evaluations", "results"]
            for output in required_outputs:
                if output not in output_config:
                    issues.append(f"Missing output.{output}")
        
        return issues
    
    def create_output_directories(self) -> Dict[str, Path]:
        """Create output directory structure."""
        self.logger.info("Creating output directories...")
        
        base_dir = Path(self.config["output"]["base_directory"]) / self.experiment_id
        
        directories = {
            "base": base_dir,
            "conversations": base_dir / self.config["output"]["conversations"],
            "evaluations": base_dir / self.config["output"]["evaluations"],
            "results": base_dir / self.config["output"]["results"],
            "logs": base_dir / "logs",
            "temp": base_dir / "temp",
            "checkpoints": base_dir / "checkpoints"
        }
        
        if not self.dry_run:
            for name, path in directories.items():
                path.mkdir(parents=True, exist_ok=True)
                self.logger.debug(f"Created directory: {path}")
        else:
            self.logger.info("DRY RUN: Would create directories:")
            for name, path in directories.items():
                self.logger.info(f"  {name}: {path}")
        
        return directories
    
    async def test_model_connections(self) -> Dict[str, Tuple[bool, str]]:
        """Test connections to all enabled models."""
        self.logger.info("Testing model connections...")
        
        results = {}
        
        if self.dry_run:
            # In dry run mode, just validate configurations
            validation_results = self.model_factory.validate_config(self.config)
            for model_name, is_valid in validation_results.items():
                if is_valid:
                    results[model_name] = (True, "DRY RUN: Configuration valid")
                else:
                    results[model_name] = (False, "DRY RUN: Configuration invalid")
            return results
        
        # Create models from configuration
        try:
            models = self.model_factory.create_models_from_config(self.config)
            
            # Test each model
            for model_name, model in models.items():
                try:
                    self.logger.info(f"Testing {model_name} connection...")
                    
                    # Validate configuration first
                    if not model.validate_configuration():
                        results[model_name] = (False, "Configuration validation failed")
                        continue
                    
                    # Run health check
                    is_healthy = await model.health_check()
                    
                    if is_healthy:
                        self.models[model_name] = model
                        results[model_name] = (True, "Connection successful")
                        self.logger.info(f"✅ {model_name} connection successful")
                    else:
                        results[model_name] = (False, "Health check failed")
                        self.logger.warning(f"⚠️ {model_name} health check failed")
                        
                except Exception as e:
                    results[model_name] = (False, f"Connection failed: {str(e)}")
                    self.logger.error(f"❌ {model_name} connection failed: {e}")
            
            # Log summary
            successful = len([r for r in results.values() if r[0]])
            total = len(results)
            self.logger.info(f"Model connection tests: {successful}/{total} successful")
            
        except Exception as e:
            self.logger.error(f"Failed to create models from configuration: {e}")
            results["error"] = (False, f"Model creation failed: {str(e)}")
        
        return results
    
    def load_scenarios(self) -> List:
        """Load and validate conversation scenarios."""
        self.logger.info("Loading conversation scenarios...")
        
        scenarios_config = self.config.get("scenarios", {})
        scenarios_dir = scenarios_config.get("directory", "data/scenarios")
        
        if not self.dry_run:
            loader = ScenarioLoader(scenarios_directory=scenarios_dir)
            
            # Load specific scenarios if specified
            if "include" in scenarios_config:
                scenarios = []
                for scenario_id in scenarios_config["include"]:
                    try:
                        scenario = loader.load_scenario(scenario_id)
                        scenarios.append(scenario)
                    except Exception as e:
                        self.logger.warning(f"Failed to load scenario {scenario_id}: {e}")
            else:
                # Load all scenarios
                scenarios = loader.load_all_scenarios()
            
            # Apply filters
            if "category" in scenarios_config:
                scenarios = [s for s in scenarios if s.category in scenarios_config["category"]]
            
            if "severity" in scenarios_config:
                scenarios = [s for s in scenarios if s.severity in scenarios_config["severity"]]
            
            self.scenarios = scenarios
            self.logger.info(f"Loaded {len(scenarios)} scenarios")
        else:
            self.logger.info("DRY RUN: Would load scenarios from configuration")
            self.scenarios = []  # Empty for dry run
        
        return self.scenarios
    
    def generate_experiment_manifest(self, directories: Dict[str, Path]) -> Dict:
        """Generate experiment manifest with all configuration."""
        manifest = {
            "experiment_id": self.experiment_id,
            "created_at": datetime.now().isoformat(),
            "configuration": self.config,
            "directories": {k: str(v) for k, v in directories.items()},
            "models": {
                name: {
                    "enabled": True,
                    "type": type(client).__name__
                } for name, client in self.models.items()
            },
            "scenarios": [
                {
                    "scenario_id": scenario.scenario_id,
                    "title": scenario.title,
                    "category": scenario.category,
                    "severity": scenario.severity
                } for scenario in self.scenarios
            ],
            "status": "initialized"
        }
        
        return manifest
    
    def save_experiment_manifest(self, manifest: Dict, directories: Dict[str, Path]) -> Path:
        """Save experiment manifest to file."""
        manifest_path = directories["base"] / "experiment_manifest.json"
        
        if not self.dry_run:
            with open(manifest_path, 'w') as f:
                json.dump(manifest, f, indent=2)
            self.logger.info(f"Saved experiment manifest: {manifest_path}")
        else:
            self.logger.info(f"DRY RUN: Would save manifest to {manifest_path}")
        
        return manifest_path
    
    def setup_logging(self, directories: Dict[str, Path]) -> None:
        """Setup experiment-specific logging."""
        if not self.dry_run:
            log_file = directories["logs"] / f"{self.experiment_id}.log"
            setup_logging(
                level="INFO",
                log_file=str(log_file),
                enable_file_logging=True
            )
            self.logger.info(f"Experiment logging configured: {log_file}")
        else:
            self.logger.info("DRY RUN: Would setup experiment logging")
    
    async def run_setup(self) -> Tuple[bool, Dict]:
        """Run complete experiment setup."""
        try:
            self.logger.info(f"Setting up experiment: {self.experiment_id}")
            
            # Load configuration
            config = self.load_configuration(self.config_path)
            
            # Validate configuration
            self.logger.info("Validating configuration...")
            issues = self.validate_configuration()
            if issues:
                self.logger.error("Configuration validation failed:")
                for issue in issues:
                    self.logger.error(f"  - {issue}")
                return False, {"errors": issues}
            
            # Create directories
            directories = self.create_output_directories()
            
            # Setup logging
            self.setup_logging(directories)
            
            # Test model connections
            connection_results = await self.test_model_connections()
            failed_connections = {k: v for k, v in connection_results.items() if not v[0]}
            
            if failed_connections:
                self.logger.error("Model connection tests failed:")
                for model, (success, message) in failed_connections.items():
                    self.logger.error(f"  - {model}: {message}")
                
                if not self.config.get("experiment", {}).get("allow_partial_models", False):
                    return False, {"connection_errors": failed_connections}
            
            # Load scenarios
            scenarios = self.load_scenarios()
            
            # Generate and save manifest
            manifest = self.generate_experiment_manifest(directories)
            manifest_path = self.save_experiment_manifest(manifest, directories)
            
            self.logger.info("Experiment setup completed successfully!")
            self.logger.info(f"Experiment ID: {self.experiment_id}")
            self.logger.info(f"Base directory: {directories['base']}")
            self.logger.info(f"Models enabled: {list(self.models.keys())}")
            self.logger.info(f"Scenarios loaded: {len(scenarios)}")
            
            return True, {
                "experiment_id": self.experiment_id,
                "directories": directories,
                "manifest_path": manifest_path,
                "models": list(self.models.keys()),
                "scenarios_count": len(scenarios),
                "connection_results": connection_results
            }
            
        except Exception as e:
            self.logger.error(f"Experiment setup failed: {str(e)}")
            return False, {"error": str(e)}


def create_default_config() -> Dict:
    """Create default experiment configuration."""
    return {
        "experiment": {
            "name": "Mental Health LLM Evaluation",
            "description": "Comprehensive evaluation of LLMs in mental health applications",
            "version": "1.0.0",
            "allow_partial_models": False
        },
        "models": {
            "openai": {
                "enabled": True,
                "model": "gpt-4",
                "temperature": 0.7,
                "max_tokens": 2048
            },
            "deepseek": {
                "enabled": True,
                "model_path": "./models/deepseek-llm-7b-chat",
                "device": "auto",
                "precision": "fp16"
            }
        },
        "scenarios": {
            "directory": "data/scenarios",
            "include": [],  # Empty means all scenarios
            "category": [],  # Empty means all categories
            "severity": []   # Empty means all severities
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
    }


def main():
    """Main function for command-line usage."""
    parser = argparse.ArgumentParser(
        description="Setup Mental Health LLM Evaluation Experiment",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Setup with custom config
  python scripts/setup_experiment.py --config config/my_experiment.yaml
  
  # Setup with inline parameters
  python scripts/setup_experiment.py --name "My Experiment" --models openai,deepseek
  
  # Dry run to test configuration
  python scripts/setup_experiment.py --dry-run
  
  # Create default config file
  python scripts/setup_experiment.py --create-default-config
        """
    )
    
    parser.add_argument(
        "--config", "-c",
        type=str,
        help="Path to experiment configuration file"
    )
    
    parser.add_argument(
        "--name", "-n",
        type=str,
        help="Experiment name (overrides config)"
    )
    
    parser.add_argument(
        "--models", "-m",
        type=str,
        help="Comma-separated list of models to enable (openai,deepseek)"
    )
    
    parser.add_argument(
        "--scenarios-dir",
        type=str,
        help="Directory containing scenario files"
    )
    
    parser.add_argument(
        "--output-dir", "-o",
        type=str,
        help="Base output directory for experiment"
    )
    
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test configuration without creating files or connections"
    )
    
    parser.add_argument(
        "--create-default-config",
        action="store_true",
        help="Create default configuration file and exit"
    )
    
    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Setup basic logging
    setup_logging(level="DEBUG" if args.verbose else "INFO")
    logger = get_logger(__name__)
    
    # Create default config if requested
    if args.create_default_config:
        config_path = PROJECT_ROOT / "config" / "experiment.yaml"
        config_path.parent.mkdir(exist_ok=True)
        
        with open(config_path, 'w') as f:
            yaml.dump(create_default_config(), f, default_flow_style=False, sort_keys=False)
        
        print(f"Created default configuration: {config_path}")
        print("Edit the configuration file and run setup again.")
        return 0
    
    try:
        # Initialize experiment setup
        setup = ExperimentSetup(config_path=args.config, dry_run=args.dry_run)
        
        # Load configuration
        if args.config:
            config = setup.load_configuration(args.config)
        else:
            # Try to load default or create minimal config
            try:
                config = setup.load_configuration()
            except FileNotFoundError:
                logger.info("No configuration file found, using defaults")
                config = create_default_config()
                setup.config = config
        
        # Apply command-line overrides
        if args.name:
            config["experiment"]["name"] = args.name
        
        if args.models:
            models = args.models.split(",")
            for model in ["openai", "deepseek"]:
                config["models"][model]["enabled"] = model in models
        
        if args.scenarios_dir:
            config["scenarios"]["directory"] = args.scenarios_dir
        
        if args.output_dir:
            config["output"]["base_directory"] = args.output_dir
        
        setup.config = config
        
        # Run setup
        print(f"\n{'='*60}")
        print(f"Mental Health LLM Evaluation - Experiment Setup")
        print(f"{'='*60}")
        
        if args.dry_run:
            print("🧪 DRY RUN MODE - No files will be created")
        
        success, result = asyncio.run(setup.run_setup())
        
        if success:
            print(f"\n✅ Experiment setup completed successfully!")
            print(f"📊 Experiment ID: {result['experiment_id']}")
            print(f"📁 Base Directory: {result['directories']['base']}")
            print(f"🤖 Models: {', '.join(result['models'])}")
            print(f"📝 Scenarios: {result['scenarios_count']}")
            
            if not args.dry_run:
                print(f"\nNext steps:")
                print(f"1. Review the experiment manifest: {result['manifest_path']}")
                print(f"2. Run conversations: python scripts/run_conversations.py --experiment {result['experiment_id']}")
            
            return 0
        else:
            print(f"\n❌ Experiment setup failed!")
            if "errors" in result:
                print("Configuration errors:")
                for error in result["errors"]:
                    print(f"  - {error}")
            if "connection_errors" in result:
                print("Connection errors:")
                for model, (success, message) in result["connection_errors"].items():
                    print(f"  - {model}: {message}")
            if "error" in result:
                print(f"Error: {result['error']}")
            
            return 1
    
    except KeyboardInterrupt:
        logger.info("Setup interrupted by user")
        return 130
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        if args.verbose:
            import traceback
            traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit(main())