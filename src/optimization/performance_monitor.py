"""
Performance Monitor for Model Selection and Chat System

Comprehensive monitoring system that tracks latency, throughput, and performance
metrics across the entire mental health chat system.
"""

import asyncio
import logging
import time
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, NamedTuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict, deque
import statistics
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class MetricType(Enum):
    """Types of performance metrics"""
    LATENCY = "latency"
    THROUGHPUT = "throughput"
    ERROR_RATE = "error_rate"
    CACHE_HIT_RATE = "cache_hit_rate"
    MODEL_SELECTION_TIME = "model_selection_time"
    RESPONSE_GENERATION_TIME = "response_generation_time"


@dataclass
class LatencyMetrics:
    """Latency performance metrics"""
    p50: float = 0.0  # Median
    p95: float = 0.0  # 95th percentile
    p99: float = 0.0  # 99th percentile
    p999: float = 0.0  # 99.9th percentile
    mean: float = 0.0
    min: float = float('inf')
    max: float = 0.0
    count: int = 0
    
    def update(self, value: float):
        """Update metrics with new latency value"""
        self.count += 1
        self.min = min(self.min, value)
        self.max = max(self.max, value)
        
        # For more accurate percentiles, we'd need to store all values
        # This is a simplified update for mean
        self.mean = ((self.mean * (self.count - 1)) + value) / self.count
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for serialization"""
        return {
            'p50': self.p50,
            'p95': self.p95,
            'p99': self.p99,
            'p999': self.p999,
            'mean': self.mean,
            'min': self.min if self.min != float('inf') else 0.0,
            'max': self.max,
            'count': self.count
        }


@dataclass
class ThroughputMetrics:
    """Throughput performance metrics"""
    requests_per_second: float = 0.0
    requests_per_minute: float = 0.0
    peak_rps: float = 0.0
    total_requests: int = 0
    window_start: datetime = field(default_factory=datetime.now)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'requests_per_second': self.requests_per_second,
            'requests_per_minute': self.requests_per_minute,
            'peak_rps': self.peak_rps,
            'total_requests': self.total_requests,
            'window_start': self.window_start.isoformat()
        }


@dataclass
class ModelPerformanceMetrics:
    """Performance metrics for individual models"""
    model_name: str
    selection_count: int = 0
    avg_selection_time_ms: float = 0.0
    avg_response_time_ms: float = 0.0
    success_rate: float = 1.0
    error_count: int = 0
    cache_hit_rate: float = 0.0
    latency_metrics: LatencyMetrics = field(default_factory=LatencyMetrics)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'model_name': self.model_name,
            'selection_count': self.selection_count,
            'avg_selection_time_ms': self.avg_selection_time_ms,
            'avg_response_time_ms': self.avg_response_time_ms,
            'success_rate': self.success_rate,
            'error_count': self.error_count,
            'cache_hit_rate': self.cache_hit_rate,
            'latency_metrics': self.latency_metrics.to_dict()
        }


class PerformanceMonitor:
    """
    Comprehensive performance monitoring system
    
    Features:
    - Real-time latency tracking with percentiles
    - Throughput monitoring and alerting
    - Model-specific performance metrics
    - Historical trend analysis
    - Performance alerts and anomaly detection
    - Cost optimization insights
    """
    
    def __init__(self, 
                 history_window_hours: int = 24,
                 metrics_retention_days: int = 30,
                 alert_thresholds: Dict[str, float] = None):
        
        self.history_window = timedelta(hours=history_window_hours)
        self.metrics_retention = timedelta(days=metrics_retention_days)
        
        # Alert thresholds
        self.alert_thresholds = alert_thresholds or {
            'p99_latency_ms': 5000,      # 5 second P99 threshold
            'error_rate_percent': 5.0,    # 5% error rate threshold
            'cache_hit_rate_min': 60.0,   # Minimum 60% cache hit rate
            'throughput_min_rps': 1.0     # Minimum 1 RPS
        }
        
        # Metrics storage
        self.latency_history: deque = deque(maxlen=10000)
        self.throughput_history: deque = deque(maxlen=1000)
        self.error_history: deque = deque(maxlen=1000)
        
        # Model-specific metrics
        self.model_metrics: Dict[str, ModelPerformanceMetrics] = {}
        
        # Global metrics
        self.global_latency = LatencyMetrics()
        self.global_throughput = ThroughputMetrics()
        
        # Time-based metrics
        self.hourly_metrics = defaultdict(lambda: defaultdict(list))
        self.daily_metrics = defaultdict(lambda: defaultdict(list))
        
        # Request tracking
        self.active_requests: Dict[str, float] = {}  # request_id -> start_time
        self.request_counter = 0
        
        # Performance alerts
        self.alert_history: deque = deque(maxlen=100)
        
        # Background tasks
        self._monitoring_task = None
        self._cleanup_task = None
        
        logger.info("PerformanceMonitor initialized")
    
    def start_monitoring(self):
        """Start background monitoring tasks"""
        if not self._monitoring_task:
            self._monitoring_task = asyncio.create_task(self._monitoring_loop())
        if not self._cleanup_task:
            self._cleanup_task = asyncio.create_task(self._cleanup_loop())
        
        logger.info("Started performance monitoring")
    
    async def stop_monitoring(self):
        """Stop background monitoring tasks"""
        if self._monitoring_task:
            self._monitoring_task.cancel()
            self._monitoring_task = None
        if self._cleanup_task:
            self._cleanup_task.cancel()
            self._cleanup_task = None
        
        logger.info("Stopped performance monitoring")
    
    def start_request(self, request_id: str = None) -> str:
        """Start tracking a request"""
        if not request_id:
            request_id = f"req_{self.request_counter}"
            self.request_counter += 1
        
        self.active_requests[request_id] = time.time()
        return request_id
    
    def end_request(self, request_id: str, 
                   success: bool = True, 
                   model_used: str = None,
                   operation_type: str = "chat") -> float:
        """End tracking a request and record metrics"""
        if request_id not in self.active_requests:
            logger.warning(f"Request {request_id} not found in active requests")
            return 0.0
        
        start_time = self.active_requests.pop(request_id)
        latency = (time.time() - start_time) * 1000  # Convert to milliseconds
        
        # Record global latency
        self.global_latency.update(latency)
        self.latency_history.append({
            'timestamp': datetime.now(),
            'latency_ms': latency,
            'success': success,
            'model': model_used,
            'operation': operation_type
        })
        
        # Record model-specific metrics
        if model_used:
            if model_used not in self.model_metrics:
                self.model_metrics[model_used] = ModelPerformanceMetrics(model_name=model_used)
            
            model_metric = self.model_metrics[model_used]
            model_metric.latency_metrics.update(latency)
            
            if operation_type == "model_selection":
                model_metric.selection_count += 1
                model_metric.avg_selection_time_ms = (
                    (model_metric.avg_selection_time_ms * (model_metric.selection_count - 1) + latency) /
                    model_metric.selection_count
                )
            elif operation_type == "response_generation":
                model_metric.avg_response_time_ms = (
                    (model_metric.avg_response_time_ms * model_metric.latency_metrics.count + latency) /
                    (model_metric.latency_metrics.count + 1)
                )
            
            if not success:
                model_metric.error_count += 1
                model_metric.success_rate = (
                    (model_metric.selection_count - model_metric.error_count) / 
                    model_metric.selection_count
                )
        
        # Record error if applicable
        if not success:
            self.error_history.append({
                'timestamp': datetime.now(),
                'request_id': request_id,
                'model': model_used,
                'operation': operation_type,
                'latency_ms': latency
            })
        
        # Update hourly metrics
        current_hour = datetime.now().hour
        self.hourly_metrics[current_hour]['latencies'].append(latency)
        self.hourly_metrics[current_hour]['successes'].append(success)
        
        return latency
    
    def record_cache_hit(self, latency_ms: float, model: str = None):
        """Record cache hit metrics"""
        cache_entry = {
            'timestamp': datetime.now(),
            'latency_ms': latency_ms,
            'model': model,
            'hit': True
        }
        
        # Update model-specific cache metrics
        if model and model in self.model_metrics:
            # This would be calculated more accurately in a real implementation
            self.model_metrics[model].cache_hit_rate += 0.01  # Simplified update
    
    def record_cache_miss(self, latency_ms: float, model: str = None):
        """Record cache miss metrics"""
        cache_entry = {
            'timestamp': datetime.now(),
            'latency_ms': latency_ms,
            'model': model,
            'hit': False
        }
        
        # Update model-specific cache metrics
        if model and model in self.model_metrics:
            # This would be calculated more accurately in a real implementation
            pass
    
    def calculate_percentiles(self, values: List[float]) -> Dict[str, float]:
        """Calculate latency percentiles"""
        if not values:
            return {'p50': 0, 'p95': 0, 'p99': 0, 'p999': 0}
        
        sorted_values = sorted(values)
        n = len(sorted_values)
        
        return {
            'p50': sorted_values[int(n * 0.50)] if n > 0 else 0,
            'p95': sorted_values[int(n * 0.95)] if n > 0 else 0,
            'p99': sorted_values[int(n * 0.99)] if n > 0 else 0,
            'p999': sorted_values[int(n * 0.999)] if n > 0 else 0
        }
    
    def get_current_metrics(self) -> Dict[str, Any]:
        """Get current performance metrics snapshot"""
        now = datetime.now()
        
        # Calculate recent latencies for percentiles
        recent_latencies = [
            entry['latency_ms'] for entry in self.latency_history
            if now - entry['timestamp'] <= timedelta(minutes=5)
        ]
        
        percentiles = self.calculate_percentiles(recent_latencies)
        
        # Calculate current throughput
        recent_requests = [
            entry for entry in self.latency_history
            if now - entry['timestamp'] <= timedelta(minutes=1)
        ]
        current_rps = len(recent_requests) / 60.0
        
        # Calculate error rate
        recent_errors = [
            entry for entry in self.error_history
            if now - entry['timestamp'] <= timedelta(minutes=5)
        ]
        error_rate = (len(recent_errors) / len(recent_latencies)) * 100 if recent_latencies else 0
        
        # Calculate cache hit rate
        cache_hits = sum(1 for entry in self.latency_history 
                        if entry.get('from_cache', False) and 
                        now - entry['timestamp'] <= timedelta(minutes=5))
        cache_hit_rate = (cache_hits / len(recent_latencies)) * 100 if recent_latencies else 0
        
        return {
            'timestamp': now.isoformat(),
            'latency': {
                **percentiles,
                'mean': statistics.mean(recent_latencies) if recent_latencies else 0,
                'count': len(recent_latencies)
            },
            'throughput': {
                'requests_per_second': current_rps,
                'requests_per_minute': len(recent_requests),
                'active_requests': len(self.active_requests)
            },
            'error_rate': error_rate,
            'cache_hit_rate': cache_hit_rate,
            'model_metrics': {
                name: metrics.to_dict() 
                for name, metrics in self.model_metrics.items()
            }
        }
    
    def get_historical_metrics(self, hours: int = 24) -> Dict[str, Any]:
        """Get historical performance metrics"""
        cutoff_time = datetime.now() - timedelta(hours=hours)
        
        # Filter historical data
        historical_latencies = [
            entry for entry in self.latency_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        historical_errors = [
            entry for entry in self.error_history
            if entry['timestamp'] >= cutoff_time
        ]
        
        # Group by hour
        hourly_data = defaultdict(lambda: {'latencies': [], 'errors': 0, 'requests': 0})
        
        for entry in historical_latencies:
            hour_key = entry['timestamp'].strftime('%Y-%m-%d %H:00')
            hourly_data[hour_key]['latencies'].append(entry['latency_ms'])
            hourly_data[hour_key]['requests'] += 1
        
        for entry in historical_errors:
            hour_key = entry['timestamp'].strftime('%Y-%m-%d %H:00')
            hourly_data[hour_key]['errors'] += 1
        
        # Calculate hourly metrics
        hourly_metrics = {}
        for hour, data in hourly_data.items():
            latencies = data['latencies']
            percentiles = self.calculate_percentiles(latencies)
            
            hourly_metrics[hour] = {
                'latency': percentiles,
                'throughput': data['requests'] / 3600,  # Per second
                'error_rate': (data['errors'] / data['requests']) * 100 if data['requests'] > 0 else 0,
                'total_requests': data['requests']
            }
        
        return {
            'period': f"{hours} hours",
            'hourly_metrics': hourly_metrics,
            'summary': {
                'total_requests': len(historical_latencies),
                'total_errors': len(historical_errors),
                'avg_latency': statistics.mean([e['latency_ms'] for e in historical_latencies]) if historical_latencies else 0,
                'overall_error_rate': (len(historical_errors) / len(historical_latencies)) * 100 if historical_latencies else 0
            }
        }
    
    def check_alert_conditions(self) -> List[Dict[str, Any]]:
        """Check for performance alert conditions"""
        alerts = []
        current_metrics = self.get_current_metrics()
        
        # Check P99 latency
        p99_latency = current_metrics['latency']['p99']
        if p99_latency > self.alert_thresholds['p99_latency_ms']:
            alerts.append({
                'type': 'high_latency',
                'severity': 'warning',
                'message': f"P99 latency ({p99_latency:.2f}ms) exceeds threshold ({self.alert_thresholds['p99_latency_ms']}ms)",
                'metric': p99_latency,
                'threshold': self.alert_thresholds['p99_latency_ms']
            })
        
        # Check error rate
        error_rate = current_metrics['error_rate']
        if error_rate > self.alert_thresholds['error_rate_percent']:
            alerts.append({
                'type': 'high_error_rate',
                'severity': 'critical',
                'message': f"Error rate ({error_rate:.2f}%) exceeds threshold ({self.alert_thresholds['error_rate_percent']}%)",
                'metric': error_rate,
                'threshold': self.alert_thresholds['error_rate_percent']
            })
        
        # Check cache hit rate
        cache_hit_rate = current_metrics['cache_hit_rate']
        if cache_hit_rate < self.alert_thresholds['cache_hit_rate_min']:
            alerts.append({
                'type': 'low_cache_hit_rate',
                'severity': 'warning',
                'message': f"Cache hit rate ({cache_hit_rate:.2f}%) below threshold ({self.alert_thresholds['cache_hit_rate_min']}%)",
                'metric': cache_hit_rate,
                'threshold': self.alert_thresholds['cache_hit_rate_min']
            })
        
        # Check throughput
        rps = current_metrics['throughput']['requests_per_second']
        if rps < self.alert_thresholds['throughput_min_rps']:
            alerts.append({
                'type': 'low_throughput',
                'severity': 'warning',
                'message': f"Throughput ({rps:.2f} RPS) below threshold ({self.alert_thresholds['throughput_min_rps']} RPS)",
                'metric': rps,
                'threshold': self.alert_thresholds['throughput_min_rps']
            })
        
        # Store alerts
        for alert in alerts:
            alert['timestamp'] = datetime.now().isoformat()
            self.alert_history.append(alert)
        
        return alerts
    
    def get_cost_optimization_insights(self) -> Dict[str, Any]:
        """Analyze performance data to provide cost optimization insights"""
        insights = {
            'recommendations': [],
            'potential_savings': 0.0,
            'efficiency_metrics': {}
        }
        
        # Analyze cache performance
        current_metrics = self.get_current_metrics()
        cache_hit_rate = current_metrics['cache_hit_rate']
        
        if cache_hit_rate < 70:
            insights['recommendations'].append({
                'type': 'cache_optimization',
                'priority': 'high',
                'description': f"Cache hit rate is {cache_hit_rate:.1f}%. Optimizing cache could reduce model selection latency.",
                'potential_impact': 'High - could reduce average latency by 20-50%'
            })
        
        # Analyze model usage patterns
        model_usage = {}
        for entry in self.latency_history:
            model = entry.get('model')
            if model:
                if model not in model_usage:
                    model_usage[model] = {'count': 0, 'total_latency': 0}
                model_usage[model]['count'] += 1
                model_usage[model]['total_latency'] += entry['latency_ms']
        
        # Find underperforming models
        for model, usage in model_usage.items():
            avg_latency = usage['total_latency'] / usage['count']
            if avg_latency > 3000:  # 3 second threshold
                insights['recommendations'].append({
                    'type': 'model_optimization',
                    'priority': 'medium',
                    'description': f"Model {model} has high average latency ({avg_latency:.0f}ms). Consider optimization or replacement.",
                    'potential_impact': f'Medium - affects {usage["count"]} requests'
                })
        
        # Calculate efficiency metrics
        total_requests = len(self.latency_history)
        if total_requests > 0:
            avg_latency = sum(entry['latency_ms'] for entry in self.latency_history) / total_requests
            cache_saves = cache_hit_rate * total_requests / 100
            
            insights['efficiency_metrics'] = {
                'average_latency_ms': avg_latency,
                'total_requests': total_requests,
                'cache_saves': cache_saves,
                'estimated_time_saved_seconds': cache_saves * (avg_latency / 1000) * 0.7  # Assume 70% latency reduction from cache
            }
        
        return insights
    
    async def _monitoring_loop(self):
        """Background monitoring loop"""
        while True:
            try:
                await asyncio.sleep(60)  # Check every minute
                
                # Check for alerts
                alerts = self.check_alert_conditions()
                
                if alerts:
                    for alert in alerts:
                        logger.warning(f"Performance alert: {alert['message']}")
                
                # Update global metrics
                self._update_global_metrics()
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in monitoring loop: {e}")
    
    async def _cleanup_loop(self):
        """Background cleanup loop"""
        while True:
            try:
                await asyncio.sleep(3600)  # Run every hour
                
                # Clean up old metrics
                cutoff_time = datetime.now() - self.metrics_retention
                
                # Clean latency history
                self.latency_history = deque(
                    [entry for entry in self.latency_history if entry['timestamp'] >= cutoff_time],
                    maxlen=self.latency_history.maxlen
                )
                
                # Clean error history
                self.error_history = deque(
                    [entry for entry in self.error_history if entry['timestamp'] >= cutoff_time],
                    maxlen=self.error_history.maxlen
                )
                
                logger.debug("Cleaned up old performance metrics")
                
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in cleanup loop: {e}")
    
    def _update_global_metrics(self):
        """Update global performance metrics"""
        now = datetime.now()
        
        # Calculate recent metrics for global updates
        recent_window = timedelta(minutes=5)
        recent_entries = [
            entry for entry in self.latency_history
            if now - entry['timestamp'] <= recent_window
        ]
        
        if recent_entries:
            latencies = [entry['latency_ms'] for entry in recent_entries]
            percentiles = self.calculate_percentiles(latencies)
            
            # Update global latency metrics
            self.global_latency.p50 = percentiles['p50']
            self.global_latency.p95 = percentiles['p95']
            self.global_latency.p99 = percentiles['p99']
            self.global_latency.p999 = percentiles['p999']
        
        # Update throughput
        minute_window = timedelta(minutes=1)
        recent_minute = [
            entry for entry in self.latency_history
            if now - entry['timestamp'] <= minute_window
        ]
        
        self.global_throughput.requests_per_second = len(recent_minute) / 60.0
        self.global_throughput.requests_per_minute = len(recent_minute)
        self.global_throughput.total_requests = len(self.latency_history)
        
        # Update peak RPS
        if self.global_throughput.requests_per_second > self.global_throughput.peak_rps:
            self.global_throughput.peak_rps = self.global_throughput.requests_per_second
    
    def export_metrics(self, file_path: str):
        """Export metrics to file"""
        try:
            metrics_data = {
                'export_timestamp': datetime.now().isoformat(),
                'current_metrics': self.get_current_metrics(),
                'historical_metrics': self.get_historical_metrics(24),
                'cost_insights': self.get_cost_optimization_insights(),
                'alert_history': list(self.alert_history)
            }
            
            with open(file_path, 'w') as f:
                json.dump(metrics_data, f, indent=2)
            
            logger.info(f"Exported metrics to {file_path}")
            
        except Exception as e:
            logger.error(f"Error exporting metrics: {e}")
    
    def reset_metrics(self):
        """Reset all metrics (useful for testing)"""
        self.latency_history.clear()
        self.throughput_history.clear()
        self.error_history.clear()
        self.model_metrics.clear()
        self.active_requests.clear()
        self.alert_history.clear()
        
        self.global_latency = LatencyMetrics()
        self.global_throughput = ThroughputMetrics()
        self.request_counter = 0
        
        logger.info("Reset all performance metrics")