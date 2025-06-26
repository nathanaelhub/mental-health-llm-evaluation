"""
Technical performance metrics for LLM evaluation.

This module evaluates technical aspects like response time, throughput,
reliability, and resource usage for mental health LLM applications.
"""

import time
import asyncio
import psutil
import statistics
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import logging

from ..models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class TechnicalScore:
    """Technical performance score for a model."""
    
    response_time_score: float  # 0-100, lower time = higher score
    throughput_score: float     # 0-100, higher throughput = higher score
    reliability_score: float    # 0-100, success rate based
    efficiency_score: float     # 0-100, resource usage based
    overall_score: float        # 0-100, weighted average
    
    response_time_ms: float
    throughput_rps: float
    success_rate: float
    cpu_usage_percent: float
    memory_usage_mb: float
    
    metadata: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "response_time_score": self.response_time_score,
            "throughput_score": self.throughput_score,
            "reliability_score": self.reliability_score,
            "efficiency_score": self.efficiency_score,
            "overall_score": self.overall_score,
            "response_time_ms": self.response_time_ms,
            "throughput_rps": self.throughput_rps,
            "success_rate": self.success_rate,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "metadata": self.metadata
        }


class TechnicalMetrics:
    """Technical performance evaluator for LLM models."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize technical metrics evaluator.
        
        Args:
            config: Configuration for evaluation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Default configuration
        self.max_response_time_ms = self.config.get("max_response_time_ms", 5000)
        self.target_throughput_rps = self.config.get("target_throughput_rps", 10)
        self.concurrent_requests = self.config.get("concurrent_requests", 5)
        self.test_duration_seconds = self.config.get("test_duration_seconds", 60)
        self.warmup_requests = self.config.get("warmup_requests", 5)
    
    async def evaluate_model(
        self,
        model: BaseModel,
        test_prompts: List[str],
        **kwargs
    ) -> TechnicalScore:
        """
        Comprehensive technical evaluation of a model.
        
        Args:
            model: Model to evaluate
            test_prompts: List of prompts for testing
            **kwargs: Additional evaluation parameters
            
        Returns:
            TechnicalScore with all metrics
        """
        self.logger.info(f"Starting technical evaluation for {model.model_name}")
        
        # Warmup
        await self._warmup_model(model, test_prompts[:self.warmup_requests])
        
        # Reset metrics before evaluation
        model.reset_metrics()
        
        # Run evaluation tests
        response_time_data = await self._measure_response_time(model, test_prompts)
        throughput_data = await self._measure_throughput(model, test_prompts)
        reliability_data = await self._measure_reliability(model, test_prompts)
        efficiency_data = await self._measure_efficiency(model, test_prompts)
        
        # Calculate scores
        score = self._calculate_technical_score(
            response_time_data,
            throughput_data,
            reliability_data,
            efficiency_data
        )
        
        self.logger.info(
            f"Technical evaluation complete for {model.model_name}: "
            f"Overall Score: {score.overall_score:.2f}"
        )
        
        return score
    
    async def _warmup_model(self, model: BaseModel, prompts: List[str]) -> None:
        """Warmup model with initial requests."""
        self.logger.info("Warming up model...")
        
        for prompt in prompts:
            try:
                await model.generate_response(prompt, max_tokens=50)
            except Exception as e:
                self.logger.warning(f"Warmup request failed: {e}")
        
        self.logger.info("Model warmup complete")
    
    async def _measure_response_time(
        self,
        model: BaseModel,
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Measure response time statistics."""
        self.logger.info("Measuring response times...")
        
        response_times = []
        successful_requests = 0
        
        for prompt in prompts:
            try:
                response = await model.generate_response(prompt)
                if response.is_successful:
                    response_times.append(response.response_time_ms)
                    successful_requests += 1
            except Exception as e:
                self.logger.warning(f"Response time measurement failed: {e}")
        
        if not response_times:
            return {
                "mean": float('inf'),
                "median": float('inf'),
                "p95": float('inf'),
                "p99": float('inf'),
                "min": float('inf'),
                "max": float('inf'),
                "std": 0,
                "successful_requests": 0
            }
        
        return {
            "mean": statistics.mean(response_times),
            "median": statistics.median(response_times),
            "p95": np.percentile(response_times, 95),
            "p99": np.percentile(response_times, 99),
            "min": min(response_times),
            "max": max(response_times),
            "std": statistics.stdev(response_times) if len(response_times) > 1 else 0,
            "successful_requests": successful_requests
        }
    
    async def _measure_throughput(
        self,
        model: BaseModel,
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Measure throughput under concurrent load."""
        self.logger.info("Measuring throughput...")
        
        start_time = time.time()
        completed_requests = 0
        successful_requests = 0
        
        # Create concurrent tasks
        semaphore = asyncio.Semaphore(self.concurrent_requests)
        
        async def make_request(prompt: str) -> bool:
            async with semaphore:
                try:
                    response = await model.generate_response(prompt)
                    return response.is_successful
                except Exception:
                    return False
        
        # Run concurrent requests
        tasks = [make_request(prompt) for prompt in prompts]
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        end_time = time.time()
        duration = end_time - start_time
        
        # Count results
        for result in results:
            completed_requests += 1
            if isinstance(result, bool) and result:
                successful_requests += 1
        
        throughput_rps = completed_requests / duration if duration > 0 else 0
        successful_throughput_rps = successful_requests / duration if duration > 0 else 0
        
        return {
            "total_requests": len(prompts),
            "completed_requests": completed_requests,
            "successful_requests": successful_requests,
            "duration_seconds": duration,
            "throughput_rps": throughput_rps,
            "successful_throughput_rps": successful_throughput_rps,
            "success_rate": successful_requests / completed_requests if completed_requests > 0 else 0
        }
    
    async def _measure_reliability(
        self,
        model: BaseModel,
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Measure model reliability and error rates."""
        self.logger.info("Measuring reliability...")
        
        total_requests = 0
        successful_requests = 0
        error_types = {}
        
        for prompt in prompts:
            total_requests += 1
            try:
                response = await model.generate_response(prompt)
                if response.is_successful:
                    successful_requests += 1
                else:
                    error_type = response.error or "unknown_error"
                    error_types[error_type] = error_types.get(error_type, 0) + 1
            except Exception as e:
                error_type = type(e).__name__
                error_types[error_type] = error_types.get(error_type, 0) + 1
        
        success_rate = successful_requests / total_requests if total_requests > 0 else 0
        error_rate = 1 - success_rate
        
        return {
            "total_requests": total_requests,
            "successful_requests": successful_requests,
            "failed_requests": total_requests - successful_requests,
            "success_rate": success_rate,
            "error_rate": error_rate,
            "error_types": error_types
        }
    
    async def _measure_efficiency(
        self,
        model: BaseModel,
        prompts: List[str]
    ) -> Dict[str, Any]:
        """Measure resource efficiency during operation."""
        self.logger.info("Measuring efficiency...")
        
        # Initial resource snapshot
        initial_cpu = psutil.cpu_percent(interval=1)
        initial_memory = psutil.virtual_memory()
        
        process = psutil.Process()
        initial_process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        cpu_samples = []
        memory_samples = []
        
        # Monitor resources during requests
        async def monitor_resources():
            while True:
                cpu_samples.append(psutil.cpu_percent())
                memory_samples.append(psutil.virtual_memory().percent)
                await asyncio.sleep(0.5)
        
        # Start monitoring
        monitor_task = asyncio.create_task(monitor_resources())
        
        try:
            # Run sample requests
            sample_prompts = prompts[:min(10, len(prompts))]
            for prompt in sample_prompts:
                try:
                    await model.generate_response(prompt)
                except Exception:
                    pass
        finally:
            monitor_task.cancel()
        
        # Final resource snapshot
        final_process_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Calculate efficiency metrics
        avg_cpu = statistics.mean(cpu_samples) if cpu_samples else initial_cpu
        max_cpu = max(cpu_samples) if cpu_samples else initial_cpu
        avg_memory = statistics.mean(memory_samples) if memory_samples else initial_memory.percent
        memory_increase = final_process_memory - initial_process_memory
        
        return {
            "initial_cpu_percent": initial_cpu,
            "average_cpu_percent": avg_cpu,
            "max_cpu_percent": max_cpu,
            "initial_memory_percent": initial_memory.percent,
            "average_memory_percent": avg_memory,
            "process_memory_mb": final_process_memory,
            "memory_increase_mb": memory_increase,
            "cpu_samples": len(cpu_samples),
            "memory_samples": len(memory_samples)
        }
    
    def _calculate_technical_score(
        self,
        response_time_data: Dict[str, Any],
        throughput_data: Dict[str, Any],
        reliability_data: Dict[str, Any],
        efficiency_data: Dict[str, Any]
    ) -> TechnicalScore:
        """Calculate overall technical score from metrics."""
        
        # Response time score (0-100, lower is better)
        mean_response_time = response_time_data["mean"]
        if mean_response_time == float('inf'):
            response_time_score = 0.0
        else:
            # Score decreases exponentially with response time
            response_time_score = max(0, 100 * np.exp(-mean_response_time / self.max_response_time_ms))
        
        # Throughput score (0-100, higher is better)
        actual_throughput = throughput_data["successful_throughput_rps"]
        throughput_score = min(100, (actual_throughput / self.target_throughput_rps) * 100)
        
        # Reliability score (0-100, based on success rate)
        reliability_score = reliability_data["success_rate"] * 100
        
        # Efficiency score (0-100, lower resource usage is better)
        cpu_penalty = min(50, efficiency_data["average_cpu_percent"] / 2)  # Max 50 point penalty
        memory_penalty = min(25, efficiency_data["average_memory_percent"] / 4)  # Max 25 point penalty
        efficiency_score = max(0, 100 - cpu_penalty - memory_penalty)
        
        # Overall score (weighted average)
        weights = self.config.get("score_weights", {
            "response_time": 0.3,
            "throughput": 0.25,
            "reliability": 0.3,
            "efficiency": 0.15
        })
        
        overall_score = (
            response_time_score * weights["response_time"] +
            throughput_score * weights["throughput"] +
            reliability_score * weights["reliability"] +
            efficiency_score * weights["efficiency"]
        )
        
        return TechnicalScore(
            response_time_score=response_time_score,
            throughput_score=throughput_score,
            reliability_score=reliability_score,
            efficiency_score=efficiency_score,
            overall_score=overall_score,
            response_time_ms=mean_response_time,
            throughput_rps=actual_throughput,
            success_rate=reliability_data["success_rate"],
            cpu_usage_percent=efficiency_data["average_cpu_percent"],
            memory_usage_mb=efficiency_data["process_memory_mb"],
            metadata={
                "response_time_data": response_time_data,
                "throughput_data": throughput_data,
                "reliability_data": reliability_data,
                "efficiency_data": efficiency_data,
                "evaluation_config": self.config
            }
        )
    
    async def benchmark_models(
        self,
        models: List[BaseModel],
        test_prompts: List[str]
    ) -> Dict[str, TechnicalScore]:
        """
        Benchmark multiple models and return comparative results.
        
        Args:
            models: List of models to benchmark
            test_prompts: Test prompts for evaluation
            
        Returns:
            Dictionary mapping model names to their technical scores
        """
        self.logger.info(f"Starting benchmark of {len(models)} models")
        
        results = {}
        
        for model in models:
            try:
                score = await self.evaluate_model(model, test_prompts)
                results[model.model_name] = score
                
                self.logger.info(
                    f"Benchmarked {model.model_name}: "
                    f"Score: {score.overall_score:.2f}, "
                    f"Response Time: {score.response_time_ms:.2f}ms, "
                    f"Throughput: {score.throughput_rps:.2f} RPS"
                )
                
            except Exception as e:
                self.logger.error(f"Failed to benchmark {model.model_name}: {e}")
                # Create failed score
                results[model.model_name] = TechnicalScore(
                    response_time_score=0.0,
                    throughput_score=0.0,
                    reliability_score=0.0,
                    efficiency_score=0.0,
                    overall_score=0.0,
                    response_time_ms=float('inf'),
                    throughput_rps=0.0,
                    success_rate=0.0,
                    cpu_usage_percent=0.0,
                    memory_usage_mb=0.0,
                    metadata={"error": str(e)}
                )
        
        self.logger.info("Benchmark complete")
        return results