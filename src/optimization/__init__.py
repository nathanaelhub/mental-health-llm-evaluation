"""
Performance Optimization and Caching Package

Advanced caching and optimization strategies for the mental health chat system
to minimize latency in model selection and improve overall performance.
"""

from .smart_cache import SmartModelCache, CachedSelection, CacheStatistics
from .performance_monitor import PerformanceMonitor, LatencyMetrics
from .model_optimizer import WarmupManager, ModelMetrics, OptimizationConfig
from .batch_processor import BatchProcessor, BatchRequest, BatchResult
from .progressive_enhancement import ProgressiveEnhancer, FastSelection, EnhancementMetrics
from .prompt_shortcuts import PromptShortcuts, ShortcutResult, ShortcutMetrics

__version__ = "1.0.0"

__all__ = [
    # Smart caching
    "SmartModelCache",
    "CachedSelection", 
    "CacheStatistics",
    
    # Performance monitoring
    "PerformanceMonitor",
    "LatencyMetrics",
    
    # Model optimization
    "WarmupManager",
    "ModelMetrics",
    "OptimizationConfig",
    
    # Batch processing
    "BatchProcessor",
    "BatchRequest",
    "BatchResult",
    
    # Progressive enhancement
    "ProgressiveEnhancer",
    "FastSelection",
    "EnhancementMetrics",
    
    # Prompt shortcuts
    "PromptShortcuts",
    "ShortcutResult",
    "ShortcutMetrics"
]