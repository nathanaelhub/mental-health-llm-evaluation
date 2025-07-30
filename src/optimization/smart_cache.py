"""
Smart Model Cache with Semantic Similarity

Intelligent caching system that learns from usage patterns to optimize model selection
using embedding-based semantic similarity and statistical analysis.
"""

import asyncio
import json
import logging
import time
import hashlib
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any, Tuple, NamedTuple
from dataclasses import dataclass, asdict
from enum import Enum
import numpy as np
from collections import defaultdict, deque
import pickle
import sqlite3
from pathlib import Path

try:
    from sentence_transformers import SentenceTransformer
    import faiss
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    SentenceTransformer = None
    faiss = None

from ..chat.dynamic_model_selector import PromptType, ModelSelection

logger = logging.getLogger(__name__)


class CacheStrategy(Enum):
    """Cache utilization strategies"""
    CONSERVATIVE = "conservative"  # High similarity threshold
    BALANCED = "balanced"         # Moderate threshold
    AGGRESSIVE = "aggressive"     # Low threshold for maximum speed


@dataclass
class SimilarPrompt:
    """Similar prompt found in cache"""
    prompt: str
    embedding: np.ndarray
    model_selection: str
    confidence: float
    prompt_type: PromptType
    timestamp: datetime
    usage_count: int
    success_rate: float
    similarity_score: float


@dataclass
class CachedSelection:
    """Cached model selection result"""
    selected_model: str
    confidence_score: float
    prompt_classification: PromptType
    reasoning: str
    cache_hit_score: float
    similar_prompts_count: int
    timestamp: datetime
    expires_at: datetime
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'selected_model': self.selected_model,
            'confidence_score': self.confidence_score,
            'prompt_classification': self.prompt_classification.value,
            'reasoning': self.reasoning,
            'cache_hit_score': self.cache_hit_score,
            'similar_prompts_count': self.similar_prompts_count,
            'timestamp': self.timestamp.isoformat(),
            'expires_at': self.expires_at.isoformat()
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'CachedSelection':
        """Create from dictionary"""
        return cls(
            selected_model=data['selected_model'],
            confidence_score=data['confidence_score'],
            prompt_classification=PromptType(data['prompt_classification']),
            reasoning=data['reasoning'],
            cache_hit_score=data['cache_hit_score'],
            similar_prompts_count=data['similar_prompts_count'],
            timestamp=datetime.fromisoformat(data['timestamp']),
            expires_at=datetime.fromisoformat(data['expires_at'])
        )


@dataclass
class CacheStatistics:
    """Comprehensive cache performance statistics"""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    
    # Performance metrics
    avg_hit_latency_ms: float = 0.0
    avg_miss_latency_ms: float = 0.0
    
    # Hit rates by category
    hit_rates_by_prompt_type: Dict[PromptType, float] = None
    hit_rates_by_model: Dict[str, float] = None
    
    # Temporal statistics
    hourly_hit_rates: List[float] = None
    recent_performance: deque = None
    
    # Cost savings
    estimated_cost_savings: float = 0.0
    time_saved_seconds: float = 0.0
    
    def __post_init__(self):
        if self.hit_rates_by_prompt_type is None:
            self.hit_rates_by_prompt_type = {}
        if self.hit_rates_by_model is None:
            self.hit_rates_by_model = {}
        if self.hourly_hit_rates is None:
            self.hourly_hit_rates = [0.0] * 24
        if self.recent_performance is None:
            self.recent_performance = deque(maxlen=1000)
    
    @property
    def hit_rate(self) -> float:
        """Calculate overall cache hit rate"""
        if self.total_requests == 0:
            return 0.0
        return (self.cache_hits / self.total_requests) * 100
    
    @property
    def miss_rate(self) -> float:
        """Calculate cache miss rate"""
        return 100.0 - self.hit_rate
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization"""
        return {
            'total_requests': self.total_requests,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'hit_rate': self.hit_rate,
            'miss_rate': self.miss_rate,
            'avg_hit_latency_ms': self.avg_hit_latency_ms,
            'avg_miss_latency_ms': self.avg_miss_latency_ms,
            'hit_rates_by_prompt_type': {
                pt.value: rate for pt, rate in self.hit_rates_by_prompt_type.items()
            },
            'hit_rates_by_model': self.hit_rates_by_model,
            'estimated_cost_savings': self.estimated_cost_savings,
            'time_saved_seconds': self.time_saved_seconds
        }


class EmbeddingStore:
    """Vector store for semantic similarity search"""
    
    def __init__(self, dimension: int = 384, cache_dir: str = "data/cache"):
        self.dimension = dimension
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Initialize FAISS index
        self.index = None
        self.prompts_metadata = []
        self.index_file = self.cache_dir / "faiss_index.bin"
        self.metadata_file = self.cache_dir / "metadata.json"
        
        self._initialize_index()
    
    def _initialize_index(self):
        """Initialize or load existing FAISS index"""
        if not EMBEDDINGS_AVAILABLE:
            logger.warning("FAISS not available - semantic similarity disabled")
            return
        
        try:
            if self.index_file.exists() and self.metadata_file.exists():
                # Load existing index
                self.index = faiss.read_index(str(self.index_file))
                with open(self.metadata_file, 'r') as f:
                    self.prompts_metadata = json.load(f)
                logger.info(f"Loaded FAISS index with {self.index.ntotal} vectors")
            else:
                # Create new index
                self.index = faiss.IndexFlatIP(self.dimension)  # Inner product for cosine similarity
                self.prompts_metadata = []
                logger.info("Created new FAISS index")
        except Exception as e:
            logger.error(f"Error initializing FAISS index: {e}")
            self.index = None
    
    def add_embedding(self, prompt: str, embedding: np.ndarray, metadata: Dict[str, Any]):
        """Add embedding to the vector store"""
        if not self.index:
            return
        
        try:
            # Normalize embedding for cosine similarity
            embedding = embedding / np.linalg.norm(embedding)
            
            # Add to FAISS index
            self.index.add(embedding.reshape(1, -1).astype('float32'))
            
            # Store metadata
            metadata_entry = {
                'prompt': prompt,
                'timestamp': datetime.now().isoformat(),
                **metadata
            }
            self.prompts_metadata.append(metadata_entry)
            
        except Exception as e:
            logger.error(f"Error adding embedding: {e}")
    
    def search_similar(self, query_embedding: np.ndarray, k: int = 5) -> List[Tuple[float, Dict[str, Any]]]:
        """Search for similar embeddings"""
        if not self.index or self.index.ntotal == 0:
            return []
        
        try:
            # Normalize query embedding
            query_embedding = query_embedding / np.linalg.norm(query_embedding)
            
            # Search FAISS index
            scores, indices = self.index.search(
                query_embedding.reshape(1, -1).astype('float32'), 
                min(k, self.index.ntotal)
            )
            
            # Return results with metadata
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < len(self.prompts_metadata):
                    results.append((float(score), self.prompts_metadata[idx]))
            
            return results
            
        except Exception as e:
            logger.error(f"Error searching embeddings: {e}")
            return []
    
    def save_index(self):
        """Save FAISS index and metadata to disk"""
        if not self.index:
            return
        
        try:
            faiss.write_index(self.index, str(self.index_file))
            with open(self.metadata_file, 'w') as f:
                json.dump(self.prompts_metadata, f, indent=2)
            logger.debug("Saved FAISS index to disk")
        except Exception as e:
            logger.error(f"Error saving FAISS index: {e}")
    
    def cleanup_old_entries(self, max_age_days: int = 30):
        """Remove old entries from the index"""
        # This is a simplified version - in production, you'd rebuild the index
        cutoff_date = datetime.now() - timedelta(days=max_age_days)
        
        old_entries = 0
        for metadata in self.prompts_metadata:
            entry_date = datetime.fromisoformat(metadata['timestamp'])
            if entry_date < cutoff_date:
                old_entries += 1
        
        if old_entries > 0:
            logger.info(f"Found {old_entries} old cache entries (not removed in this implementation)")


class SmartModelCache:
    """
    Intelligent caching system that learns from usage patterns to optimize model selection
    
    Features:
    - Semantic similarity matching using sentence embeddings
    - Progressive confidence scoring based on historical performance
    - Adaptive thresholds based on prompt classification
    - Statistical analysis for cache effectiveness
    - Background cache warming and optimization
    """
    
    def __init__(self, 
                 cache_dir: str = "data/cache",
                 embedding_model_name: str = "all-MiniLM-L6-v2",
                 similarity_threshold: float = 0.85,
                 cache_strategy: CacheStrategy = CacheStrategy.BALANCED,
                 max_cache_size: int = 10000,
                 cache_ttl_hours: int = 24):
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.similarity_threshold = similarity_threshold
        self.cache_strategy = cache_strategy
        self.max_cache_size = max_cache_size
        self.cache_ttl = timedelta(hours=cache_ttl_hours)
        
        # Initialize embedding model
        self.embedding_model = None
        if EMBEDDINGS_AVAILABLE:
            try:
                self.embedding_model = SentenceTransformer(embedding_model_name)
                logger.info(f"Loaded embedding model: {embedding_model_name}")
            except Exception as e:
                logger.error(f"Failed to load embedding model: {e}")
        
        # Initialize vector store
        self.vector_store = EmbeddingStore(cache_dir=str(self.cache_dir))
        
        # Cache statistics
        self.stats = CacheStatistics()
        
        # In-memory cache for recent selections
        self.memory_cache: Dict[str, CachedSelection] = {}
        
        # Performance tracking
        self.latency_history = deque(maxlen=1000)
        
        # Adaptive thresholds by prompt type
        self.adaptive_thresholds = {
            PromptType.CRISIS: 0.95,          # Very high confidence needed for crisis
            PromptType.INFORMATION_SEEKING: 0.75,  # Lower threshold for info requests
            PromptType.GENERAL_WELLNESS: 0.80,     # Moderate threshold
            PromptType.ANXIETY: 0.85,              # High threshold for therapeutic content
            PromptType.DEPRESSION: 0.85,           # High threshold for therapeutic content
        }
        
        # Background tasks
        self._optimization_task = None
        self._cleanup_task = None
        
        logger.info("SmartModelCache initialized successfully")
    
    async def get_cached_selection(self, prompt: str, prompt_type: Optional[PromptType] = None) -> Optional[CachedSelection]:
        """
        Find similar prompts and return cached model selection if confidence > threshold
        
        Args:
            prompt: User prompt to find similar cached selections for
            prompt_type: Optional prompt classification to improve matching
            
        Returns:
            CachedSelection if found with sufficient confidence, None otherwise
        """
        start_time = time.time()
        
        try:
            # Update request statistics
            self.stats.total_requests += 1
            
            # Quick hash-based lookup for exact matches
            prompt_hash = self._hash_prompt(prompt)
            if prompt_hash in self.memory_cache:
                cached = self.memory_cache[prompt_hash]
                if cached.expires_at > datetime.now():
                    latency = (time.time() - start_time) * 1000
                    await self._record_cache_hit(latency, cached)
                    return cached
                else:
                    # Remove expired entry
                    del self.memory_cache[prompt_hash]
            
            # Semantic similarity search
            if not self.embedding_model:
                await self._record_cache_miss(time.time() - start_time)
                return None
            
            # Generate embedding for prompt
            prompt_embedding = self.embedding_model.encode(prompt)
            
            # Search for similar prompts
            similar_results = self.vector_store.search_similar(prompt_embedding, k=10)
            
            if not similar_results:
                await self._record_cache_miss(time.time() - start_time)
                return None
            
            # Convert to SimilarPrompt objects
            similar_prompts = []
            for score, metadata in similar_results:
                similar_prompts.append(SimilarPrompt(
                    prompt=metadata['prompt'],
                    embedding=prompt_embedding,  # Simplified
                    model_selection=metadata.get('selected_model', ''),
                    confidence=metadata.get('confidence', 0.0),
                    prompt_type=PromptType(metadata.get('prompt_type', 'general_wellness')),
                    timestamp=datetime.fromisoformat(metadata['timestamp']),
                    usage_count=metadata.get('usage_count', 1),
                    success_rate=metadata.get('success_rate', 1.0),
                    similarity_score=score
                ))
            
            # Determine if cache should be used
            if await self._should_use_cache(similar_prompts, prompt_type):
                cached_selection = await self._aggregate_selections(similar_prompts, prompt)
                
                # Store in memory cache
                self.memory_cache[prompt_hash] = cached_selection
                
                latency = (time.time() - start_time) * 1000
                await self._record_cache_hit(latency, cached_selection)
                
                return cached_selection
            
            await self._record_cache_miss(time.time() - start_time)
            return None
            
        except Exception as e:
            logger.error(f"Error in get_cached_selection: {e}")
            await self._record_cache_miss(time.time() - start_time)
            return None
    
    async def store_selection(self, 
                            prompt: str, 
                            selection: ModelSelection, 
                            success: bool = True,
                            response_time_ms: float = 0.0):
        """
        Store a new model selection in the cache
        
        Args:
            prompt: Original user prompt
            selection: Model selection result
            success: Whether the selection was successful
            response_time_ms: Time taken for model inference
        """
        try:
            if not self.embedding_model:
                return
            
            # Generate embedding
            prompt_embedding = self.embedding_model.encode(prompt)
            
            # Create metadata
            metadata = {
                'selected_model': selection.selected_model,
                'confidence': selection.confidence_score,
                'prompt_type': selection.prompt_classification.value,
                'reasoning': selection.reasoning,
                'success': success,
                'response_time_ms': response_time_ms,
                'usage_count': 1,
                'success_rate': 1.0 if success else 0.0
            }
            
            # Add to vector store
            self.vector_store.add_embedding(prompt, prompt_embedding, metadata)
            
            # Create cached selection for memory cache
            cached_selection = CachedSelection(
                selected_model=selection.selected_model,
                confidence_score=selection.confidence_score,
                prompt_classification=selection.prompt_classification,
                reasoning=f"Cached from recent selection: {selection.reasoning}",
                cache_hit_score=1.0,
                similar_prompts_count=1,
                timestamp=datetime.now(),
                expires_at=datetime.now() + self.cache_ttl
            )
            
            # Store in memory cache
            prompt_hash = self._hash_prompt(prompt)
            self.memory_cache[prompt_hash] = cached_selection
            
            # Cleanup if cache is getting too large
            if len(self.memory_cache) > self.max_cache_size:
                await self._cleanup_memory_cache()
            
            logger.debug(f"Stored selection for prompt type {selection.prompt_classification.value}")
            
        except Exception as e:
            logger.error(f"Error storing selection: {e}")
    
    async def _should_use_cache(self, similar_prompts: List[SimilarPrompt], prompt_type: Optional[PromptType] = None) -> bool:
        """
        Determine if cache should be used based on similarity and consistency
        
        Args:
            similar_prompts: List of similar prompts found
            prompt_type: Optional prompt classification
            
        Returns:
            True if cache should be used, False otherwise
        """
        if not similar_prompts:
            return False
        
        # Get the most similar prompt
        best_match = max(similar_prompts, key=lambda p: p.similarity_score)
        
        # Check minimum similarity threshold
        threshold = self.adaptive_thresholds.get(prompt_type, self.similarity_threshold)
        if best_match.similarity_score < threshold:
            return False
        
        # Check recency - don't use very old cache entries
        max_age = timedelta(hours=48)  # Configurable
        if datetime.now() - best_match.timestamp > max_age:
            return False
        
        # Check consistency of model selection among similar prompts
        model_votes = defaultdict(float)
        total_weight = 0
        
        for prompt in similar_prompts:
            if prompt.similarity_score >= 0.7:  # Minimum threshold for voting
                weight = prompt.similarity_score * prompt.success_rate
                model_votes[prompt.model_selection] += weight
                total_weight += weight
        
        if not model_votes:
            return False
        
        # Check if there's a clear winner
        best_model = max(model_votes.items(), key=lambda x: x[1])
        confidence = best_model[1] / total_weight if total_weight > 0 else 0
        
        # Strategy-based decision
        if self.cache_strategy == CacheStrategy.CONSERVATIVE:
            return confidence > 0.8 and best_match.similarity_score > 0.9
        elif self.cache_strategy == CacheStrategy.AGGRESSIVE:
            return confidence > 0.6 and best_match.similarity_score > 0.75
        else:  # BALANCED
            return confidence > 0.7 and best_match.similarity_score >= threshold
    
    async def _aggregate_selections(self, similar_prompts: List[SimilarPrompt], original_prompt: str) -> CachedSelection:
        """
        Aggregate multiple similar selections into a single cache result
        
        Args:
            similar_prompts: List of similar prompts
            original_prompt: Original user prompt
            
        Returns:
            Aggregated CachedSelection
        """
        # Weighted voting based on similarity and success rate
        model_votes = defaultdict(float)
        confidence_scores = defaultdict(list)
        reasoning_parts = defaultdict(list)
        
        total_weight = 0
        for prompt in similar_prompts:
            if prompt.similarity_score >= 0.7:
                weight = prompt.similarity_score * prompt.success_rate
                model_votes[prompt.model_selection] += weight
                confidence_scores[prompt.model_selection].append(prompt.confidence)
                reasoning_parts[prompt.model_selection].append(
                    f"Similar to: '{prompt.prompt[:50]}...' (similarity: {prompt.similarity_score:.2f})"
                )
                total_weight += weight
        
        # Select best model
        best_model = max(model_votes.items(), key=lambda x: x[1])[0]
        
        # Calculate aggregated confidence
        model_confidences = confidence_scores[best_model]
        avg_confidence = sum(model_confidences) / len(model_confidences) if model_confidences else 0
        
        # Calculate cache hit score
        cache_hit_score = model_votes[best_model] / total_weight if total_weight > 0 else 0
        
        # Create reasoning
        reasoning = f"Cached selection based on {len(similar_prompts)} similar prompts:\n" + \
                   "\n".join(reasoning_parts[best_model][:3])  # Top 3 reasons
        
        # Determine prompt classification from most similar
        best_match = max(similar_prompts, key=lambda p: p.similarity_score)
        
        return CachedSelection(
            selected_model=best_model,
            confidence_score=avg_confidence * cache_hit_score,  # Adjust confidence by cache quality
            prompt_classification=best_match.prompt_type,
            reasoning=reasoning,
            cache_hit_score=cache_hit_score,
            similar_prompts_count=len(similar_prompts),
            timestamp=datetime.now(),
            expires_at=datetime.now() + self.cache_ttl
        )
    
    async def _record_cache_hit(self, latency_ms: float, cached_selection: CachedSelection):
        """Record cache hit statistics"""
        self.stats.cache_hits += 1
        
        # Update latency statistics
        self.stats.avg_hit_latency_ms = (
            (self.stats.avg_hit_latency_ms * (self.stats.cache_hits - 1) + latency_ms) /
            self.stats.cache_hits
        )
        
        # Update hit rates by category
        prompt_type = cached_selection.prompt_classification
        if prompt_type not in self.stats.hit_rates_by_prompt_type:
            self.stats.hit_rates_by_prompt_type[prompt_type] = 0
        
        # Update model-specific statistics
        model = cached_selection.selected_model
        if model not in self.stats.hit_rates_by_model:
            self.stats.hit_rates_by_model[model] = 0
        
        # Record performance
        self.stats.recent_performance.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'hit',
            'latency_ms': latency_ms,
            'model': model,
            'prompt_type': prompt_type.value
        })
        
        logger.debug(f"Cache hit: {model} in {latency_ms:.2f}ms")
    
    async def _record_cache_miss(self, latency_ms: float):
        """Record cache miss statistics"""
        self.stats.cache_misses += 1
        
        # Update latency statistics
        self.stats.avg_miss_latency_ms = (
            (self.stats.avg_miss_latency_ms * (self.stats.cache_misses - 1) + latency_ms) /
            self.stats.cache_misses
        )
        
        # Record performance
        self.stats.recent_performance.append({
            'timestamp': datetime.now().isoformat(),
            'type': 'miss',
            'latency_ms': latency_ms
        })
        
        logger.debug(f"Cache miss in {latency_ms:.2f}ms")
    
    def _hash_prompt(self, prompt: str) -> str:
        """Generate hash for prompt-based lookup"""
        return hashlib.md5(prompt.encode()).hexdigest()
    
    async def _cleanup_memory_cache(self):
        """Clean up old entries from memory cache"""
        now = datetime.now()
        expired_keys = [
            key for key, cached in self.memory_cache.items()
            if cached.expires_at <= now
        ]
        
        for key in expired_keys:
            del self.memory_cache[key]
        
        # If still too large, remove oldest entries
        if len(self.memory_cache) > self.max_cache_size:
            sorted_items = sorted(
                self.memory_cache.items(),
                key=lambda x: x[1].timestamp
            )
            
            items_to_remove = len(sorted_items) - self.max_cache_size + 100  # Remove some extra
            for key, _ in sorted_items[:items_to_remove]:
                del self.memory_cache[key]
        
        logger.debug(f"Cleaned up memory cache, now has {len(self.memory_cache)} entries")
    
    def get_statistics(self) -> CacheStatistics:
        """Get current cache statistics"""
        # Update hit rates by category
        for prompt_type in PromptType:
            type_hits = sum(1 for perf in self.stats.recent_performance 
                          if perf.get('type') == 'hit' and perf.get('prompt_type') == prompt_type.value)
            type_total = sum(1 for perf in self.stats.recent_performance 
                           if perf.get('prompt_type') == prompt_type.value)
            
            if type_total > 0:
                self.stats.hit_rates_by_prompt_type[prompt_type] = (type_hits / type_total) * 100
        
        return self.stats
    
    async def warm_cache(self, common_prompts: List[Tuple[str, PromptType]]):
        """Pre-warm cache with common prompts"""
        logger.info(f"Warming cache with {len(common_prompts)} common prompts")
        
        for prompt, prompt_type in common_prompts:
            try:
                # This would trigger the full model selection process
                # and store the result in cache for future use
                logger.debug(f"Warming cache for prompt type: {prompt_type.value}")
                
                # In a real implementation, you'd call the model selector here
                # For now, we'll just add a placeholder entry
                
            except Exception as e:
                logger.error(f"Error warming cache for prompt '{prompt[:50]}...': {e}")
    
    async def optimize_thresholds(self):
        """Optimize similarity thresholds based on performance history"""
        logger.info("Optimizing cache thresholds based on performance history")
        
        # Analyze recent performance by prompt type
        for prompt_type in PromptType:
            type_performances = [
                perf for perf in self.stats.recent_performance
                if perf.get('prompt_type') == prompt_type.value
            ]
            
            if len(type_performances) < 10:  # Need minimum data
                continue
            
            hit_rate = sum(1 for p in type_performances if p['type'] == 'hit') / len(type_performances)
            
            # Adjust threshold based on hit rate
            current_threshold = self.adaptive_thresholds.get(prompt_type, self.similarity_threshold)
            
            if hit_rate < 0.3:  # Too many misses, lower threshold
                new_threshold = max(0.65, current_threshold - 0.05)
            elif hit_rate > 0.8:  # Very high hit rate, could be more selective
                new_threshold = min(0.95, current_threshold + 0.02)
            else:
                new_threshold = current_threshold
            
            if new_threshold != current_threshold:
                self.adaptive_thresholds[prompt_type] = new_threshold
                logger.info(f"Adjusted threshold for {prompt_type.value}: {current_threshold:.2f} -> {new_threshold:.2f}")
    
    async def invalidate_cache(self, pattern: str = None, prompt_type: PromptType = None):
        """Invalidate cache entries matching criteria"""
        invalidated = 0
        
        # Clear memory cache
        if pattern or prompt_type:
            keys_to_remove = []
            for key, cached in self.memory_cache.items():
                should_remove = False
                
                if prompt_type and cached.prompt_classification == prompt_type:
                    should_remove = True
                
                # Pattern matching would need to store original prompts
                # This is a simplified implementation
                
                if should_remove:
                    keys_to_remove.append(key)
            
            for key in keys_to_remove:
                del self.memory_cache[key]
                invalidated += 1
        else:
            # Clear all
            invalidated = len(self.memory_cache)
            self.memory_cache.clear()
        
        logger.info(f"Invalidated {invalidated} cache entries")
    
    async def save_cache(self):
        """Save cache state to disk"""
        try:
            # Save vector store
            self.vector_store.save_index()
            
            # Save statistics
            stats_file = self.cache_dir / "statistics.json"
            with open(stats_file, 'w') as f:
                json.dump(self.stats.to_dict(), f, indent=2)
            
            logger.debug("Saved cache state to disk")
            
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    async def cleanup(self):
        """Cleanup resources"""
        logger.info("Cleaning up SmartModelCache...")
        
        # Cancel background tasks
        if self._optimization_task:
            self._optimization_task.cancel()
        if self._cleanup_task:
            self._cleanup_task.cancel()
        
        # Save cache state
        await self.save_cache()
        
        # Clean up old vector store entries
        self.vector_store.cleanup_old_entries()
        
        logger.info("SmartModelCache cleanup complete")