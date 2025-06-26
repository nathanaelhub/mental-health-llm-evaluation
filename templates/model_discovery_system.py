"""
Model Discovery System

Automatically detects and manages available models in the evaluation framework.
This system extends the base model registry with advanced discovery and management capabilities.

FEATURES:
- Automatic model detection and registration
- Dynamic model loading based on availability
- Health monitoring and status tracking
- Dependency validation and management
- Performance profiling and benchmarking
"""

import asyncio
import importlib
import inspect
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from models.base_model import BaseModel, ModelProvider, ModelType
from models.model_registry import get_model_registry, ModelRegistration
from models.model_factory import get_model_factory

logger = logging.getLogger(__name__)


class ModelStatus(Enum):
    """Model availability status."""
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"
    LOADING = "loading"
    FAILED = "failed"
    UNKNOWN = "unknown"


@dataclass
class ModelDiscoveryResult:
    """Result of model discovery process."""
    name: str
    registration: ModelRegistration
    status: ModelStatus
    health_check_passed: bool = False
    discovery_time: datetime = field(default_factory=datetime.now)
    last_health_check: Optional[datetime] = None
    error_message: Optional[str] = None
    performance_metrics: Dict[str, Any] = field(default_factory=dict)
    dependencies_satisfied: bool = False
    config_validated: bool = False


class ModelDiscoverySystem:
    """Advanced model discovery and management system."""
    
    def __init__(self):
        self.registry = get_model_registry()
        self.factory = get_model_factory()
        self.logger = logging.getLogger(__name__)
        
        # Discovery state
        self.discovered_models: Dict[str, ModelDiscoveryResult] = {}
        self.last_discovery: Optional[datetime] = None
        self.discovery_in_progress = False
        
        # Performance tracking
        self.performance_history: Dict[str, List[Dict[str, Any]]] = {}
        
        # Health check scheduling
        self.health_check_interval = timedelta(minutes=30)
        self.last_health_checks: Dict[str, datetime] = {}
    
    async def discover_models(
        self, 
        force_refresh: bool = False,
        include_unavailable: bool = True
    ) -> Dict[str, ModelDiscoveryResult]:
        """
        Discover all available models in the system.
        
        Args:
            force_refresh: Force re-discovery even if recently completed
            include_unavailable: Include models that failed dependency checks
            
        Returns:
            Dictionary mapping model names to discovery results
        """
        if self.discovery_in_progress:
            self.logger.warning("Discovery already in progress, waiting for completion...")
            while self.discovery_in_progress:
                await asyncio.sleep(0.1)
            return self.discovered_models
        
        # Check if recent discovery exists
        if (not force_refresh and 
            self.last_discovery and 
            datetime.now() - self.last_discovery < timedelta(minutes=5)):
            self.logger.info("Using recent discovery results")
            return self.discovered_models
        
        self.discovery_in_progress = True
        self.logger.info("Starting model discovery process...")
        
        try:
            # Refresh registry to pick up any new models
            self.registry.refresh_availability()
            
            # Get all registered models
            registered_models = self.registry.list_models()
            
            # Discover each model
            discovery_tasks = []
            for model_name, model_info in registered_models.items():
                task = self._discover_single_model(model_name, model_info)
                discovery_tasks.append(task)
            
            # Execute discovery in parallel
            discovery_results = await asyncio.gather(*discovery_tasks, return_exceptions=True)
            
            # Process results
            for result in discovery_results:
                if isinstance(result, Exception):
                    self.logger.error(f"Discovery task failed: {result}")
                    continue
                
                if result:
                    self.discovered_models[result.name] = result
            
            # Filter results if requested
            if not include_unavailable:
                self.discovered_models = {
                    name: result for name, result in self.discovered_models.items()
                    if result.status == ModelStatus.AVAILABLE
                }
            
            self.last_discovery = datetime.now()
            self.logger.info(f"Discovery completed: {len(self.discovered_models)} models found")
            
            return self.discovered_models
            
        finally:
            self.discovery_in_progress = False
    
    async def _discover_single_model(
        self, 
        model_name: str, 
        model_info: Dict[str, Any]
    ) -> Optional[ModelDiscoveryResult]:
        """Discover and analyze a single model."""
        try:
            # Get registration info
            registration = self.registry.get_model(model_name)
            if not registration:
                return None
            
            result = ModelDiscoveryResult(
                name=model_name,
                registration=registration,
                status=ModelStatus.UNKNOWN
            )
            
            # Check dependencies
            result.dependencies_satisfied = self._check_dependencies(registration.requirements)
            if not result.dependencies_satisfied:
                result.status = ModelStatus.UNAVAILABLE
                result.error_message = f"Missing dependencies: {', '.join(registration.requirements)}"
                return result
            
            # Validate configuration
            result.config_validated = self._validate_model_config(model_name)
            if not result.config_validated:
                result.status = ModelStatus.UNAVAILABLE
                result.error_message = "Configuration validation failed"
                return result
            
            # Test model creation
            try:
                result.status = ModelStatus.LOADING
                model = self.factory.create_model(model_name, cache=False)
                if not model:
                    result.status = ModelStatus.FAILED
                    result.error_message = "Model creation failed"
                    return result
                
                # Run health check
                result.health_check_passed = await model.health_check()
                result.last_health_check = datetime.now()
                
                if result.health_check_passed:
                    result.status = ModelStatus.AVAILABLE
                    
                    # Collect performance metrics
                    result.performance_metrics = await self._collect_performance_metrics(model)
                else:
                    result.status = ModelStatus.FAILED
                    result.error_message = "Health check failed"
                
            except Exception as e:
                result.status = ModelStatus.FAILED
                result.error_message = f"Model testing failed: {str(e)}"
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error discovering model {model_name}: {e}")
            return ModelDiscoveryResult(
                name=model_name,
                registration=registration,
                status=ModelStatus.FAILED,
                error_message=str(e)
            )
    
    def _check_dependencies(self, requirements: List[str]) -> bool:
        """Check if all required dependencies are available."""
        for requirement in requirements:
            try:
                importlib.import_module(requirement)
            except ImportError:
                return False
        return True
    
    def _validate_model_config(self, model_name: str) -> bool:
        """Validate model configuration."""
        try:
            return self.factory.validate_config({"models": {model_name: {"enabled": True}}}).get(model_name, False)
        except Exception:
            return False
    
    async def _collect_performance_metrics(self, model: BaseModel) -> Dict[str, Any]:
        """Collect performance metrics for a model."""
        metrics = {}
        
        try:
            # Test response time
            start_time = time.time()
            test_response = await model.generate_response(
                "Hello, this is a performance test.",
                max_tokens=10,
                temperature=0.1
            )
            response_time = time.time() - start_time
            
            metrics.update({
                "response_time_ms": response_time * 1000,
                "test_successful": test_response.is_successful,
                "token_count": test_response.token_count or 0,
                "cost_usd": test_response.cost_usd or 0.0
            })
            
            # Get model info
            model_info = model.get_model_info()
            metrics.update({
                "model_type": model_info.get("type"),
                "max_context_length": model_info.get("max_context_length"),
                "supports_streaming": model_info.get("supports_streaming", False),
                "supports_function_calling": model_info.get("supports_function_calling", False)
            })
            
            # Store performance history
            if model.model_name not in self.performance_history:
                self.performance_history[model.model_name] = []
            
            self.performance_history[model.model_name].append({
                "timestamp": datetime.now().isoformat(),
                "response_time_ms": response_time * 1000,
                "successful": test_response.is_successful
            })
            
            # Keep only last 100 entries
            if len(self.performance_history[model.model_name]) > 100:
                self.performance_history[model.model_name] = self.performance_history[model.model_name][-100:]
                
        except Exception as e:
            self.logger.warning(f"Failed to collect performance metrics for {model.model_name}: {e}")
            metrics["error"] = str(e)
        
        return metrics
    
    async def run_health_checks(self, model_names: Optional[List[str]] = None) -> Dict[str, bool]:
        """Run health checks on specified models or all available models."""
        if model_names is None:
            model_names = list(self.discovered_models.keys())
        
        results = {}
        
        for model_name in model_names:
            try:
                # Check if health check is needed
                last_check = self.last_health_checks.get(model_name)
                if (last_check and 
                    datetime.now() - last_check < self.health_check_interval):
                    # Use cached result
                    if model_name in self.discovered_models:
                        results[model_name] = self.discovered_models[model_name].health_check_passed
                    continue
                
                # Run fresh health check
                model = self.factory.create_model(model_name, cache=False)
                if model:
                    is_healthy = await model.health_check()
                    results[model_name] = is_healthy
                    self.last_health_checks[model_name] = datetime.now()
                    
                    # Update discovery result
                    if model_name in self.discovered_models:
                        self.discovered_models[model_name].health_check_passed = is_healthy
                        self.discovered_models[model_name].last_health_check = datetime.now()
                        if not is_healthy:
                            self.discovered_models[model_name].status = ModelStatus.FAILED
                else:
                    results[model_name] = False
                    
            except Exception as e:
                self.logger.error(f"Health check failed for {model_name}: {e}")
                results[model_name] = False
        
        return results
    
    def get_available_models(self, provider: Optional[ModelProvider] = None, model_type: Optional[ModelType] = None) -> List[str]:
        """Get list of available model names, optionally filtered."""
        available = []
        
        for name, result in self.discovered_models.items():
            if result.status != ModelStatus.AVAILABLE:
                continue
            
            if provider and result.registration.provider != provider:
                continue
                
            if model_type and result.registration.model_type != model_type:
                continue
            
            available.append(name)
        
        return available
    
    def get_model_status(self, model_name: str) -> Optional[ModelDiscoveryResult]:
        """Get detailed status for a specific model."""
        return self.discovered_models.get(model_name)
    
    def get_performance_summary(self) -> Dict[str, Dict[str, Any]]:
        """Get performance summary for all models."""
        summary = {}
        
        for model_name, result in self.discovered_models.items():
            if result.status == ModelStatus.AVAILABLE and result.performance_metrics:
                metrics = result.performance_metrics
                history = self.performance_history.get(model_name, [])
                
                summary[model_name] = {
                    "current_response_time_ms": metrics.get("response_time_ms", 0),
                    "average_response_time_ms": sum(h.get("response_time_ms", 0) for h in history) / len(history) if history else 0,
                    "success_rate": sum(1 for h in history if h.get("successful", False)) / len(history) if history else 0,
                    "total_tests": len(history),
                    "last_test": history[-1]["timestamp"] if history else None,
                    "model_type": metrics.get("model_type"),
                    "supports_streaming": metrics.get("supports_streaming", False)
                }
        
        return summary
    
    async def benchmark_models(self, test_prompts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
        """Run comprehensive benchmarks on available models."""
        if test_prompts is None:
            test_prompts = [
                "Hello, how are you?",
                "I'm feeling anxious about my upcoming presentation. Can you help?",
                "What are some healthy coping strategies for stress?"
            ]
        
        available_models = self.get_available_models()
        benchmark_results = {}
        
        for model_name in available_models:
            self.logger.info(f"Benchmarking {model_name}...")
            
            try:
                model = self.factory.create_model(model_name)
                if not model:
                    continue
                
                model_results = {
                    "response_times": [],
                    "token_counts": [],
                    "costs": [],
                    "success_rate": 0,
                    "average_response_time": 0,
                    "total_cost": 0
                }
                
                successful_tests = 0
                
                for prompt in test_prompts:
                    try:
                        start_time = time.time()
                        response = await model.generate_response(prompt, max_tokens=100)
                        end_time = time.time()
                        
                        if response.is_successful:
                            successful_tests += 1
                            model_results["response_times"].append((end_time - start_time) * 1000)
                            model_results["token_counts"].append(response.token_count or 0)
                            model_results["costs"].append(response.cost_usd or 0.0)
                        
                    except Exception as e:
                        self.logger.warning(f"Benchmark test failed for {model_name}: {e}")
                
                # Calculate aggregated metrics
                if model_results["response_times"]:
                    model_results["average_response_time"] = sum(model_results["response_times"]) / len(model_results["response_times"])
                    model_results["total_cost"] = sum(model_results["costs"])
                    model_results["success_rate"] = successful_tests / len(test_prompts)
                
                benchmark_results[model_name] = model_results
                
            except Exception as e:
                self.logger.error(f"Benchmark failed for {model_name}: {e}")
                benchmark_results[model_name] = {"error": str(e)}
        
        return benchmark_results
    
    def generate_discovery_report(self) -> str:
        """Generate a comprehensive discovery report."""
        report_lines = []
        report_lines.append("# Model Discovery Report")
        report_lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report_lines.append(f"Last Discovery: {self.last_discovery.strftime('%Y-%m-%d %H:%M:%S') if self.last_discovery else 'Never'}")
        report_lines.append("")
        
        # Summary statistics
        total_models = len(self.discovered_models)
        available_models = len([r for r in self.discovered_models.values() if r.status == ModelStatus.AVAILABLE])
        cloud_models = len([r for r in self.discovered_models.values() if r.registration.model_type == ModelType.CLOUD])
        local_models = len([r for r in self.discovered_models.values() if r.registration.model_type == ModelType.LOCAL])
        
        report_lines.append("## Summary")
        report_lines.append(f"- Total Models: {total_models}")
        report_lines.append(f"- Available Models: {available_models}")
        report_lines.append(f"- Cloud Models: {cloud_models}")
        report_lines.append(f"- Local Models: {local_models}")
        report_lines.append("")
        
        # Model details
        report_lines.append("## Model Details")
        report_lines.append("")
        
        for name, result in sorted(self.discovered_models.items()):
            status_emoji = {
                ModelStatus.AVAILABLE: "✅",
                ModelStatus.UNAVAILABLE: "❌",
                ModelStatus.FAILED: "💥",
                ModelStatus.LOADING: "⏳",
                ModelStatus.UNKNOWN: "❓"
            }.get(result.status, "❓")
            
            report_lines.append(f"### {name} {status_emoji}")
            report_lines.append(f"- **Provider:** {result.registration.provider.value}")
            report_lines.append(f"- **Type:** {result.registration.model_type.value}")
            report_lines.append(f"- **Status:** {result.status.value}")
            report_lines.append(f"- **Health Check:** {'✅ Passed' if result.health_check_passed else '❌ Failed'}")
            
            if result.error_message:
                report_lines.append(f"- **Error:** {result.error_message}")
            
            if result.performance_metrics:
                metrics = result.performance_metrics
                if "response_time_ms" in metrics:
                    report_lines.append(f"- **Response Time:** {metrics['response_time_ms']:.2f}ms")
                if "cost_usd" in metrics:
                    report_lines.append(f"- **Test Cost:** ${metrics['cost_usd']:.6f}")
            
            report_lines.append("")
        
        # Performance summary
        performance_summary = self.get_performance_summary()
        if performance_summary:
            report_lines.append("## Performance Summary")
            report_lines.append("")
            
            for model_name, metrics in performance_summary.items():
                report_lines.append(f"### {model_name}")
                report_lines.append(f"- **Average Response Time:** {metrics['average_response_time_ms']:.2f}ms")
                report_lines.append(f"- **Success Rate:** {metrics['success_rate']*100:.1f}%")
                report_lines.append(f"- **Total Tests:** {metrics['total_tests']}")
                report_lines.append("")
        
        return "\n".join(report_lines)


# Global discovery system instance
_global_discovery_system = None


def get_discovery_system() -> ModelDiscoverySystem:
    """Get the global model discovery system instance."""
    global _global_discovery_system
    if _global_discovery_system is None:
        _global_discovery_system = ModelDiscoverySystem()
    return _global_discovery_system


# Convenience functions
async def discover_all_models(force_refresh: bool = False) -> Dict[str, ModelDiscoveryResult]:
    """Discover all available models."""
    discovery_system = get_discovery_system()
    return await discovery_system.discover_models(force_refresh=force_refresh)


async def get_available_models(provider: Optional[ModelProvider] = None, model_type: Optional[ModelType] = None) -> List[str]:
    """Get list of available model names."""
    discovery_system = get_discovery_system()
    await discovery_system.discover_models()
    return discovery_system.get_available_models(provider=provider, model_type=model_type)


async def run_model_health_checks(model_names: Optional[List[str]] = None) -> Dict[str, bool]:
    """Run health checks on models."""
    discovery_system = get_discovery_system()
    return await discovery_system.run_health_checks(model_names=model_names)


async def benchmark_all_models(test_prompts: Optional[List[str]] = None) -> Dict[str, Dict[str, Any]]:
    """Run benchmarks on all available models."""
    discovery_system = get_discovery_system()
    return await discovery_system.benchmark_models(test_prompts=test_prompts)


def generate_model_report() -> str:
    """Generate a comprehensive model discovery report."""
    discovery_system = get_discovery_system()
    return discovery_system.generate_discovery_report()