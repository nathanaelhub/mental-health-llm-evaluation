"""
Technical Performance Metrics Evaluator (25% weight)

This module implements the technical performance evaluation system with
standardized scoring methods for response latency, context retention,
token efficiency, and resource usage.
"""

import time
import psutil
import asyncio
import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Any, Tuple
import numpy as np
import logging
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from ..models.base_model import BaseModel, ModelResponse

logger = logging.getLogger(__name__)


@dataclass
class TechnicalPerformanceScore:
    """Technical performance score breakdown."""
    
    # Individual metric scores (0-10 scale)
    response_latency_score: float
    context_retention_score: float
    token_efficiency_score: float
    resource_usage_score: float
    
    # Composite score (0-100 scale)
    overall_score: float
    
    # Raw measurements
    avg_response_time_ms: float
    context_retention_percentage: float
    avg_token_count: float
    cpu_usage_percent: Optional[float]
    memory_usage_mb: Optional[float]
    gpu_usage_percent: Optional[float]
    
    # Statistical data
    response_time_std: float
    token_count_std: float
    total_conversations: int
    
    # Manual review flags
    review_flags: List[str]
    
    # Detailed breakdown
    metric_details: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "response_latency_score": self.response_latency_score,
            "context_retention_score": self.context_retention_score,
            "token_efficiency_score": self.token_efficiency_score,
            "resource_usage_score": self.resource_usage_score,
            "overall_score": self.overall_score,
            "avg_response_time_ms": self.avg_response_time_ms,
            "context_retention_percentage": self.context_retention_percentage,
            "avg_token_count": self.avg_token_count,
            "cpu_usage_percent": self.cpu_usage_percent,
            "memory_usage_mb": self.memory_usage_mb,
            "gpu_usage_percent": self.gpu_usage_percent,
            "response_time_std": self.response_time_std,
            "token_count_std": self.token_count_std,
            "total_conversations": self.total_conversations,
            "review_flags": self.review_flags,
            "metric_details": self.metric_details
        }


class TechnicalPerformanceEvaluator:
    """Evaluator for technical performance metrics."""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize technical performance evaluator.
        
        Args:
            config: Configuration for evaluation parameters
        """
        self.config = config or {}
        self.logger = logging.getLogger(__name__)
        
        # Initialize sentence transformer for context analysis
        try:
            self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        except Exception as e:
            self.logger.warning(f"Failed to load sentence transformer: {e}")
            self.sentence_model = None
        
        # Scoring thresholds
        self.latency_thresholds = {
            "excellent": 500,    # <500ms = 10pts
            "good": 1000,        # 500-1000ms = 8pts
            "fair": 2000,        # 1000-2000ms = 6pts
            "poor": float('inf') # >2000ms = 4pts
        }
        
        self.optimal_token_range = (150, 300)  # Optimal token range
        self.token_deviation_penalty = 0.1    # Penalty per token outside range
        
        # Resource monitoring
        self.monitor_resources = self.config.get("monitor_resources", True)
        self.resource_sample_interval = self.config.get("resource_sample_interval", 1.0)
    
    async def evaluate_model(
        self,
        model: BaseModel,
        conversations: List[Dict[str, Any]],
        **kwargs
    ) -> TechnicalPerformanceScore:
        """
        Evaluate technical performance of a model.
        
        Args:
            model: Model to evaluate
            conversations: List of conversation data with responses
            **kwargs: Additional evaluation parameters
            
        Returns:
            Technical performance score
        """
        self.logger.info(f"Starting technical performance evaluation for {model.model_name}")
        
        if not conversations:
            raise ValueError("No conversations provided for evaluation")
        
        # Extract responses from conversations
        responses = self._extract_responses(conversations)
        
        # Evaluate each metric
        latency_score, latency_data = self._evaluate_response_latency(responses)
        context_score, context_data = await self._evaluate_context_retention(conversations)
        token_score, token_data = self._evaluate_token_efficiency(responses)
        resource_score, resource_data = await self._evaluate_resource_usage(model, conversations)
        
        # Calculate overall score (weighted average)
        weights = self.config.get("metric_weights", {
            "latency": 0.3,
            "context": 0.3,
            "tokens": 0.2,
            "resources": 0.2
        })
        
        overall_score = (
            latency_score * weights["latency"] +
            context_score * weights["context"] +
            token_score * weights["tokens"] +
            resource_score * weights["resources"]
        ) * 10  # Scale to 0-100
        
        # Generate review flags
        review_flags = self._generate_review_flags(
            latency_data, context_data, token_data, resource_data
        )
        
        # Compile metric details
        metric_details = {
            "latency": latency_data,
            "context": context_data,
            "tokens": token_data,
            "resources": resource_data,
            "weights": weights
        }
        
        score = TechnicalPerformanceScore(
            response_latency_score=latency_score,
            context_retention_score=context_score,
            token_efficiency_score=token_score,
            resource_usage_score=resource_score,
            overall_score=overall_score,
            avg_response_time_ms=latency_data["avg_response_time"],
            context_retention_percentage=context_data["avg_retention_percentage"],
            avg_token_count=token_data["avg_token_count"],
            cpu_usage_percent=resource_data.get("avg_cpu_percent"),
            memory_usage_mb=resource_data.get("avg_memory_mb"),
            gpu_usage_percent=resource_data.get("avg_gpu_percent"),
            response_time_std=latency_data["std_response_time"],
            token_count_std=token_data["std_token_count"],
            total_conversations=len(conversations),
            review_flags=review_flags,
            metric_details=metric_details
        )
        
        self.logger.info(
            f"Technical evaluation complete for {model.model_name}: "
            f"Overall Score: {overall_score:.1f}/100"
        )
        
        return score
    
    def _extract_responses(self, conversations: List[Dict[str, Any]]) -> List[ModelResponse]:
        """Extract model responses from conversation data."""
        responses = []
        
        for conversation in conversations:
            if "model_responses" in conversation:
                responses.extend(conversation["model_responses"])
            elif "messages" in conversation:
                # Extract assistant messages
                for message in conversation["messages"]:
                    if message.get("role") == "assistant":
                        # Create mock ModelResponse
                        response = ModelResponse(
                            content=message.get("content", ""),
                            model_name=conversation.get("model_name", "unknown"),
                            timestamp=message.get("timestamp", ""),
                            response_time_ms=message.get("response_time_ms", 0),
                            token_count=len(message.get("content", "").split()) * 1.3  # Rough estimate
                        )
                        responses.append(response)
        
        return responses
    
    def _evaluate_response_latency(self, responses: List[ModelResponse]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate response latency metric.
        
        Response Latency Scoring:
        - <500ms = 10pts
        - 500-1000ms = 8pts
        - 1000-2000ms = 6pts
        - >2000ms = 4pts
        """
        if not responses:
            return 0.0, {"error": "No responses to evaluate"}
        
        response_times = [r.response_time_ms for r in responses if r.response_time_ms > 0]
        
        if not response_times:
            return 0.0, {"error": "No valid response times found"}
        
        avg_response_time = np.mean(response_times)
        std_response_time = np.std(response_times)
        
        # Score based on average response time
        if avg_response_time < self.latency_thresholds["excellent"]:
            score = 10.0
        elif avg_response_time < self.latency_thresholds["good"]:
            score = 8.0
        elif avg_response_time < self.latency_thresholds["fair"]:
            score = 6.0
        else:
            score = 4.0
        
        # Apply penalty for high variance (inconsistent performance)
        cv = std_response_time / avg_response_time if avg_response_time > 0 else 0
        if cv > 0.5:  # High coefficient of variation
            score *= 0.9  # 10% penalty
        
        latency_data = {
            "avg_response_time": avg_response_time,
            "std_response_time": std_response_time,
            "min_response_time": min(response_times),
            "max_response_time": max(response_times),
            "median_response_time": np.median(response_times),
            "coefficient_variation": cv,
            "total_responses": len(response_times),
            "score_rationale": f"Average: {avg_response_time:.0f}ms, CV: {cv:.2f}"
        }
        
        return score, latency_data
    
    async def _evaluate_context_retention(self, conversations: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate context retention metric.
        
        Context Retention: Percentage of correctly referenced conversation elements (0-100%)
        """
        if not self.sentence_model:
            self.logger.warning("Sentence model not available, using simplified context evaluation")
            return await self._evaluate_context_retention_simple(conversations)
        
        retention_scores = []
        detailed_analysis = []
        
        for conversation in conversations:
            messages = conversation.get("messages", [])
            if len(messages) < 3:  # Need at least user-assistant-user to evaluate
                continue
            
            conversation_score = await self._analyze_conversation_context(messages)
            retention_scores.append(conversation_score)
            
            detailed_analysis.append({
                "conversation_id": conversation.get("conversation_id", "unknown"),
                "retention_score": conversation_score,
                "message_count": len(messages)
            })
        
        if not retention_scores:
            return 0.0, {"error": "No conversations suitable for context evaluation"}
        
        avg_retention = np.mean(retention_scores)
        score = avg_retention / 10  # Convert percentage to 0-10 scale
        
        context_data = {
            "avg_retention_percentage": avg_retention,
            "std_retention": np.std(retention_scores),
            "min_retention": min(retention_scores),
            "max_retention": max(retention_scores),
            "total_conversations": len(retention_scores),
            "detailed_analysis": detailed_analysis[:10],  # First 10 for brevity
            "score_rationale": f"Average retention: {avg_retention:.1f}%"
        }
        
        return score, context_data
    
    async def _analyze_conversation_context(self, messages: List[Dict[str, str]]) -> float:
        """Analyze context retention in a single conversation."""
        try:
            # Extract user messages and following assistant responses
            context_scores = []
            
            for i in range(len(messages) - 1):
                if messages[i].get("role") == "user" and messages[i + 1].get("role") == "assistant":
                    user_content = messages[i].get("content", "")
                    assistant_content = messages[i + 1].get("content", "")
                    
                    # Get conversation history up to this point
                    history = messages[:i]
                    
                    if history:
                        # Check if assistant response references previous context
                        context_score = await self._calculate_context_reference_score(
                            user_content, assistant_content, history
                        )
                        context_scores.append(context_score)
            
            return np.mean(context_scores) if context_scores else 0.0
            
        except Exception as e:
            self.logger.warning(f"Error analyzing conversation context: {e}")
            return 0.0
    
    async def _calculate_context_reference_score(
        self,
        user_message: str,
        assistant_response: str,
        history: List[Dict[str, str]]
    ) -> float:
        """Calculate how well the assistant response references previous context."""
        try:
            # Get embeddings for current user message and assistant response
            current_embedding = self.sentence_model.encode([user_message + " " + assistant_response])
            
            # Get embeddings for conversation history
            history_texts = [msg.get("content", "") for msg in history if msg.get("content")]
            if not history_texts:
                return 50.0  # Neutral score if no history
            
            history_embeddings = self.sentence_model.encode(history_texts)
            
            # Calculate semantic similarity between current conversation and history
            similarities = cosine_similarity(current_embedding, history_embeddings)[0]
            max_similarity = np.max(similarities)
            
            # Check for explicit references (pronouns, demonstratives)
            reference_indicators = [
                "you mentioned", "as you said", "like you", "your", "that", "this",
                "earlier", "before", "previously", "we discussed", "you told me"
            ]
            
            reference_count = sum(
                1 for indicator in reference_indicators
                if indicator.lower() in assistant_response.lower()
            )
            
            # Combine semantic similarity and explicit references
            semantic_score = min(max_similarity * 100, 100)  # Scale to 0-100
            reference_bonus = min(reference_count * 10, 20)   # Up to 20 bonus points
            
            total_score = min(semantic_score + reference_bonus, 100)
            
            return total_score
            
        except Exception as e:
            self.logger.warning(f"Error calculating context reference score: {e}")
            return 50.0  # Neutral score on error
    
    async def _evaluate_context_retention_simple(self, conversations: List[Dict[str, Any]]) -> Tuple[float, Dict[str, Any]]:
        """Simplified context retention evaluation without sentence transformers."""
        retention_scores = []
        
        for conversation in conversations:
            messages = conversation.get("messages", [])
            if len(messages) < 3:
                continue
            
            # Simple heuristic: check for pronouns and references in assistant responses
            assistant_messages = [msg for msg in messages if msg.get("role") == "assistant"]
            
            reference_count = 0
            total_responses = len(assistant_messages)
            
            for response in assistant_messages:
                content = response.get("content", "").lower()
                
                # Count reference indicators
                references = [
                    "you mentioned", "you said", "your", "that", "this",
                    "earlier", "before", "we discussed", "as you"
                ]
                
                if any(ref in content for ref in references):
                    reference_count += 1
            
            retention_percentage = (reference_count / total_responses * 100) if total_responses > 0 else 0
            retention_scores.append(retention_percentage)
        
        avg_retention = np.mean(retention_scores) if retention_scores else 0
        score = min(avg_retention / 10, 10)  # Scale to 0-10
        
        context_data = {
            "avg_retention_percentage": avg_retention,
            "total_conversations": len(retention_scores),
            "method": "simplified_heuristic",
            "score_rationale": f"Simple reference analysis: {avg_retention:.1f}%"
        }
        
        return score, context_data
    
    def _evaluate_token_efficiency(self, responses: List[ModelResponse]) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate token efficiency metric.
        
        Token Efficiency: Deviation from optimal 150-300 token range
        """
        if not responses:
            return 0.0, {"error": "No responses to evaluate"}
        
        token_counts = []
        for response in responses:
            if response.token_count:
                token_counts.append(response.token_count)
            else:
                # Estimate token count from content length
                estimated_tokens = len(response.content.split()) * 1.3
                token_counts.append(estimated_tokens)
        
        if not token_counts:
            return 0.0, {"error": "No token counts available"}
        
        avg_tokens = np.mean(token_counts)
        std_tokens = np.std(token_counts)
        
        # Calculate efficiency score
        optimal_min, optimal_max = self.optimal_token_range
        
        if optimal_min <= avg_tokens <= optimal_max:
            # Within optimal range
            base_score = 10.0
        else:
            # Calculate deviation penalty
            if avg_tokens < optimal_min:
                deviation = optimal_min - avg_tokens
            else:
                deviation = avg_tokens - optimal_max
            
            penalty = deviation * self.token_deviation_penalty
            base_score = max(0, 10.0 - penalty)
        
        # Apply consistency bonus/penalty
        cv = std_tokens / avg_tokens if avg_tokens > 0 else 0
        if cv < 0.3:  # Consistent token usage
            base_score *= 1.1  # 10% bonus
        elif cv > 0.7:  # Inconsistent token usage
            base_score *= 0.9  # 10% penalty
        
        score = min(base_score, 10.0)
        
        token_data = {
            "avg_token_count": avg_tokens,
            "std_token_count": std_tokens,
            "min_tokens": min(token_counts),
            "max_tokens": max(token_counts),
            "median_tokens": np.median(token_counts),
            "optimal_range": self.optimal_token_range,
            "within_optimal": optimal_min <= avg_tokens <= optimal_max,
            "coefficient_variation": cv,
            "total_responses": len(token_counts),
            "score_rationale": f"Average: {avg_tokens:.0f} tokens (optimal: {optimal_min}-{optimal_max})"
        }
        
        return score, token_data
    
    async def _evaluate_resource_usage(
        self,
        model: BaseModel,
        conversations: List[Dict[str, Any]]
    ) -> Tuple[float, Dict[str, Any]]:
        """
        Evaluate resource usage metric (local models only).
        
        Resource Usage: CPU/RAM/GPU usage during inference
        """
        if not self.monitor_resources:
            return 8.0, {"monitoring_disabled": True}  # Default good score
        
        try:
            # Monitor resources during a sample conversation
            resource_samples = await self._monitor_resources_during_inference(model)
            
            if not resource_samples:
                return 5.0, {"error": "No resource samples collected"}
            
            # Calculate average usage
            cpu_usage = [sample["cpu_percent"] for sample in resource_samples]
            memory_usage = [sample["memory_mb"] for sample in resource_samples]
            
            avg_cpu = np.mean(cpu_usage)
            avg_memory = np.mean(memory_usage)
            
            # Get GPU usage if available
            gpu_usage = None
            if hasattr(model, 'get_memory_usage'):
                try:
                    gpu_info = model.get_memory_usage()
                    if "gpu_memory_allocated_mb" in gpu_info:
                        gpu_usage = gpu_info["gpu_memory_allocated_mb"]
                except:
                    pass
            
            # Score based on resource efficiency
            score = self._calculate_resource_efficiency_score(avg_cpu, avg_memory, gpu_usage)
            
            resource_data = {
                "avg_cpu_percent": avg_cpu,
                "max_cpu_percent": max(cpu_usage),
                "avg_memory_mb": avg_memory,
                "max_memory_mb": max(memory_usage),
                "avg_gpu_percent": gpu_usage,
                "sample_count": len(resource_samples),
                "monitoring_duration": len(resource_samples) * self.resource_sample_interval,
                "score_rationale": f"CPU: {avg_cpu:.1f}%, Memory: {avg_memory:.0f}MB"
            }
            
            return score, resource_data
            
        except Exception as e:
            self.logger.warning(f"Resource monitoring failed: {e}")
            return 5.0, {"error": str(e), "default_score_applied": True}
    
    async def _monitor_resources_during_inference(self, model: BaseModel) -> List[Dict[str, float]]:
        """Monitor resource usage during model inference."""
        samples = []
        
        try:
            # Run a short inference session while monitoring
            async def monitor_task():
                for _ in range(10):  # Monitor for 10 samples
                    cpu_percent = psutil.cpu_percent(interval=None)
                    memory_info = psutil.virtual_memory()
                    
                    samples.append({
                        "cpu_percent": cpu_percent,
                        "memory_mb": memory_info.used / 1024 / 1024,
                        "memory_percent": memory_info.percent
                    })
                    
                    await asyncio.sleep(self.resource_sample_interval)
            
            # Run inference and monitoring concurrently
            inference_task = model.generate_response("How are you feeling today?")
            monitor_task_coro = monitor_task()
            
            await asyncio.gather(inference_task, monitor_task_coro)
            
        except Exception as e:
            self.logger.warning(f"Error during resource monitoring: {e}")
        
        return samples
    
    def _calculate_resource_efficiency_score(
        self,
        cpu_percent: float,
        memory_mb: float,
        gpu_usage: Optional[float] = None
    ) -> float:
        """Calculate resource efficiency score."""
        score = 10.0
        
        # CPU usage scoring (lower is better)
        if cpu_percent > 90:
            score -= 3
        elif cpu_percent > 70:
            score -= 2
        elif cpu_percent > 50:
            score -= 1
        
        # Memory usage scoring (reasonable usage expected)
        if memory_mb > 8000:  # >8GB
            score -= 2
        elif memory_mb > 4000:  # >4GB
            score -= 1
        
        # GPU usage (if available)
        if gpu_usage is not None:
            if gpu_usage > 16000:  # >16GB GPU memory
                score -= 1
        
        return max(score, 0)
    
    def _generate_review_flags(
        self,
        latency_data: Dict[str, Any],
        context_data: Dict[str, Any],
        token_data: Dict[str, Any],
        resource_data: Dict[str, Any]
    ) -> List[str]:
        """Generate manual review flags for edge cases."""
        flags = []
        
        # Latency flags
        if latency_data.get("coefficient_variation", 0) > 1.0:
            flags.append("HIGH_LATENCY_VARIANCE")
        
        if latency_data.get("max_response_time", 0) > 10000:  # >10 seconds
            flags.append("EXTREME_LATENCY_DETECTED")
        
        # Context flags
        if context_data.get("avg_retention_percentage", 0) < 20:
            flags.append("LOW_CONTEXT_RETENTION")
        
        # Token flags
        if token_data.get("coefficient_variation", 0) > 1.0:
            flags.append("INCONSISTENT_TOKEN_USAGE")
        
        avg_tokens = token_data.get("avg_token_count", 0)
        if avg_tokens < 50:
            flags.append("RESPONSES_TOO_SHORT")
        elif avg_tokens > 500:
            flags.append("RESPONSES_TOO_LONG")
        
        # Resource flags
        if resource_data.get("avg_cpu_percent", 0) > 95:
            flags.append("HIGH_CPU_USAGE")
        
        if resource_data.get("avg_memory_mb", 0) > 16000:  # >16GB
            flags.append("HIGH_MEMORY_USAGE")
        
        return flags