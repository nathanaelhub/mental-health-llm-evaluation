"""
Response Cache for Dynamic Model Selection

Caches model selections and responses for similar prompts to reduce latency
and improve performance in repeated scenarios.
"""

import json
import hashlib
import time
import logging
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path

from .model_selector import ModelSelectionResult

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Single cache entry for model selection or response"""
    key: str
    prompt_hash: str
    selected_model: str
    selection_result: ModelSelectionResult
    response_text: Optional[str]
    created_at: datetime
    access_count: int
    last_accessed: datetime
    cache_type: str  # 'selection' or 'response'
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            'key': self.key,
            'prompt_hash': self.prompt_hash,
            'selected_model': self.selected_model,
            'selection_result': self.selection_result.to_dict(),
            'response_text': self.response_text,
            'created_at': self.created_at.isoformat(),
            'access_count': self.access_count,
            'last_accessed': self.last_accessed.isoformat(),
            'cache_type': self.cache_type
        }


class ResponseCache:
    """
    Intelligent caching system for model selections and responses
    
    Features:
    - Caches model selection results for similar prompts
    - Caches full responses for identical prompts
    - Similarity-based matching using prompt hashing
    - LRU eviction with configurable TTL
    - Cache hit analytics and performance metrics
    """
    
    def __init__(self,
                 cache_dir: str = "temp/response_cache",
                 max_entries: int = 1000,
                 ttl_hours: int = 24,
                 similarity_threshold: float = 0.9,
                 enable_response_caching: bool = True):
        """
        Initialize response cache
        
        Args:
            cache_dir: Directory to store cache data
            max_entries: Maximum number of cache entries
            ttl_hours: Time-to-live for cache entries in hours
            similarity_threshold: Threshold for prompt similarity matching
            enable_response_caching: Whether to cache full responses
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        self.max_entries = max_entries
        self.ttl = timedelta(hours=ttl_hours)
        self.similarity_threshold = similarity_threshold
        self.enable_response_caching = enable_response_caching
        
        # In-memory cache
        self.cache: Dict[str, CacheEntry] = {}
        self.prompt_hashes: Dict[str, Set[str]] = {}  # hash -> set of cache keys
        
        # Performance metrics
        self.cache_hits = 0
        self.cache_misses = 0
        self.total_requests = 0
        
        # Load existing cache
        self._load_cache()
        
        logger.info(f"ResponseCache initialized with {len(self.cache)} entries")
    
    def get_cached_selection(self, prompt: str, system_prompt: Optional[str] = None) -> Optional[ModelSelectionResult]:
        """
        Get cached model selection for a prompt
        
        Args:
            prompt: User prompt to check
            system_prompt: Optional system prompt
            
        Returns:
            Cached ModelSelectionResult or None if not found
        """
        self.total_requests += 1
        
        # Create prompt key
        full_prompt = self._create_prompt_key(prompt, system_prompt)
        prompt_hash = self._hash_prompt(full_prompt)
        
        # Look for exact match first
        exact_key = f"selection_{prompt_hash}"
        if exact_key in self.cache:
            entry = self.cache[exact_key]
            if not self._is_expired(entry):
                self._update_access(entry)
                self.cache_hits += 1
                logger.debug(f"Cache hit (exact) for prompt hash {prompt_hash[:8]}")
                return entry.selection_result
        
        # Look for similar prompts
        similar_key = self._find_similar_prompt(prompt_hash, 'selection')
        if similar_key:
            entry = self.cache[similar_key]
            if not self._is_expired(entry):
                self._update_access(entry)
                self.cache_hits += 1
                logger.debug(f"Cache hit (similar) for prompt hash {prompt_hash[:8]}")
                return entry.selection_result
        
        self.cache_misses += 1
        logger.debug(f"Cache miss for prompt hash {prompt_hash[:8]}")
        return None
    
    def cache_selection(self,
                       prompt: str,
                       system_prompt: Optional[str],
                       selection_result: ModelSelectionResult):
        """
        Cache a model selection result
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            selection_result: Result to cache
        """
        full_prompt = self._create_prompt_key(prompt, system_prompt)
        prompt_hash = self._hash_prompt(full_prompt)
        cache_key = f"selection_{prompt_hash}"
        
        entry = CacheEntry(
            key=cache_key,
            prompt_hash=prompt_hash,
            selected_model=selection_result.selected_model,
            selection_result=selection_result,
            response_text=None,
            created_at=datetime.now(),
            access_count=0,
            last_accessed=datetime.now(),
            cache_type='selection'
        )
        
        self._add_to_cache(entry)
        logger.debug(f"Cached selection result for prompt hash {prompt_hash[:8]}")
    
    def get_cached_response(self,
                          prompt: str,
                          system_prompt: Optional[str] = None,
                          model_name: Optional[str] = None) -> Optional[str]:
        """
        Get cached response for a prompt
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model_name: Optional specific model name
            
        Returns:
            Cached response text or None if not found
        """
        if not self.enable_response_caching:
            return None
        
        self.total_requests += 1
        
        full_prompt = self._create_prompt_key(prompt, system_prompt, model_name)
        prompt_hash = self._hash_prompt(full_prompt)
        
        # Look for exact match
        exact_key = f"response_{prompt_hash}"
        if exact_key in self.cache:
            entry = self.cache[exact_key]
            if not self._is_expired(entry):
                self._update_access(entry)
                self.cache_hits += 1
                logger.debug(f"Response cache hit for prompt hash {prompt_hash[:8]}")
                return entry.response_text
        
        self.cache_misses += 1
        return None
    
    def cache_response(self,
                      prompt: str,
                      system_prompt: Optional[str],
                      model_name: str,
                      response_text: str,
                      selection_result: Optional[ModelSelectionResult] = None):
        """
        Cache a response
        
        Args:
            prompt: User prompt
            system_prompt: Optional system prompt
            model_name: Model that generated the response
            response_text: The response to cache
            selection_result: Optional selection result
        """
        if not self.enable_response_caching:
            return
        
        full_prompt = self._create_prompt_key(prompt, system_prompt, model_name)
        prompt_hash = self._hash_prompt(full_prompt)
        cache_key = f"response_{prompt_hash}"
        
        # Create dummy selection result if not provided
        if not selection_result:
            selection_result = ModelSelectionResult(
                selected_model=model_name,
                selection_score=0.0,
                selection_time_ms=0.0,
                all_scores={model_name: 0.0},
                response_preview=response_text[:100],
                timestamp=datetime.now()
            )
        
        entry = CacheEntry(
            key=cache_key,
            prompt_hash=prompt_hash,
            selected_model=model_name,
            selection_result=selection_result,
            response_text=response_text,
            created_at=datetime.now(),
            access_count=0,
            last_accessed=datetime.now(),
            cache_type='response'
        )
        
        self._add_to_cache(entry)
        logger.debug(f"Cached response for prompt hash {prompt_hash[:8]}")
    
    def clear_cache(self):
        """Clear all cache entries"""
        self.cache.clear()
        self.prompt_hashes.clear()
        
        # Clear storage
        for cache_file in self.cache_dir.glob("*.json"):
            cache_file.unlink()
        
        logger.info("Cache cleared")
    
    def cleanup_expired(self) -> int:
        """Remove expired cache entries and return count removed"""
        expired_keys = []
        
        for key, entry in self.cache.items():
            if self._is_expired(entry):
                expired_keys.append(key)
        
        for key in expired_keys:
            self._remove_from_cache(key)
        
        logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
        return len(expired_keys)
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics"""
        hit_rate = (self.cache_hits / self.total_requests) if self.total_requests > 0 else 0
        
        # Calculate cache size by type
        selection_count = sum(1 for entry in self.cache.values() if entry.cache_type == 'selection')
        response_count = sum(1 for entry in self.cache.values() if entry.cache_type == 'response')
        
        # Calculate average access count
        avg_access = sum(entry.access_count for entry in self.cache.values()) / len(self.cache) if self.cache else 0
        
        return {
            'total_entries': len(self.cache),
            'selection_entries': selection_count,
            'response_entries': response_count,
            'hit_rate': hit_rate,
            'cache_hits': self.cache_hits,
            'cache_misses': self.cache_misses,
            'total_requests': self.total_requests,
            'avg_access_count': avg_access,
            'max_entries': self.max_entries,
            'ttl_hours': self.ttl.total_seconds() / 3600
        }
    
    def get_top_cached_prompts(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get most frequently accessed cached prompts"""
        sorted_entries = sorted(
            self.cache.values(),
            key=lambda e: e.access_count,
            reverse=True
        )
        
        return [
            {
                'prompt_hash': entry.prompt_hash[:8],
                'selected_model': entry.selected_model,
                'access_count': entry.access_count,
                'cache_type': entry.cache_type,
                'created_at': entry.created_at.isoformat(),
                'last_accessed': entry.last_accessed.isoformat()
            }
            for entry in sorted_entries[:limit]
        ]
    
    def _create_prompt_key(self,
                          prompt: str,
                          system_prompt: Optional[str] = None,
                          model_name: Optional[str] = None) -> str:
        """Create a standardized prompt key"""
        parts = [prompt.strip().lower()]
        
        if system_prompt:
            parts.append(f"SYSTEM:{system_prompt.strip().lower()}")
        
        if model_name:
            parts.append(f"MODEL:{model_name}")
        
        return "||".join(parts)
    
    def _hash_prompt(self, prompt: str) -> str:
        """Create hash of prompt for caching"""
        return hashlib.sha256(prompt.encode('utf-8')).hexdigest()
    
    def _find_similar_prompt(self, prompt_hash: str, cache_type: str) -> Optional[str]:
        """Find similar cached prompt using hash comparison"""
        # For now, use exact hash matching
        # In a more sophisticated implementation, you could use
        # semantic similarity, fuzzy matching, or embedding similarity
        
        for cached_hash, cache_keys in self.prompt_hashes.items():
            if cached_hash == prompt_hash:
                # Find valid entry of the right type
                for key in cache_keys:
                    if key in self.cache and self.cache[key].cache_type == cache_type:
                        return key
        
        return None
    
    def _add_to_cache(self, entry: CacheEntry):
        """Add entry to cache with eviction if necessary"""
        # Remove old entry if exists
        if entry.key in self.cache:
            self._remove_from_cache(entry.key)
        
        # Evict if at capacity
        if len(self.cache) >= self.max_entries:
            self._evict_lru()
        
        # Add new entry
        self.cache[entry.key] = entry
        
        # Update hash index
        if entry.prompt_hash not in self.prompt_hashes:
            self.prompt_hashes[entry.prompt_hash] = set()
        self.prompt_hashes[entry.prompt_hash].add(entry.key)
        
        # Save to disk
        self._save_entry(entry)
    
    def _remove_from_cache(self, key: str):
        """Remove entry from cache"""
        if key not in self.cache:
            return
        
        entry = self.cache[key]
        del self.cache[key]
        
        # Update hash index
        if entry.prompt_hash in self.prompt_hashes:
            self.prompt_hashes[entry.prompt_hash].discard(key)
            if not self.prompt_hashes[entry.prompt_hash]:
                del self.prompt_hashes[entry.prompt_hash]
        
        # Remove from disk
        cache_file = self.cache_dir / f"{key}.json"
        if cache_file.exists():
            cache_file.unlink()
    
    def _evict_lru(self):
        """Evict least recently used entry"""
        if not self.cache:
            return
        
        lru_entry = min(self.cache.values(), key=lambda e: e.last_accessed)
        self._remove_from_cache(lru_entry.key)
        logger.debug(f"Evicted LRU entry {lru_entry.key}")
    
    def _update_access(self, entry: CacheEntry):
        """Update access statistics for cache entry"""
        entry.access_count += 1
        entry.last_accessed = datetime.now()
        self._save_entry(entry)
    
    def _is_expired(self, entry: CacheEntry) -> bool:
        """Check if cache entry has expired"""
        return datetime.now() - entry.created_at > self.ttl
    
    def _save_entry(self, entry: CacheEntry):
        """Save cache entry to disk"""
        cache_file = self.cache_dir / f"{entry.key}.json"
        
        try:
            with open(cache_file, 'w') as f:
                json.dump(entry.to_dict(), f, indent=2)
        except Exception as e:
            logger.error(f"Error saving cache entry {entry.key}: {e}")
    
    def _load_cache(self):
        """Load cache entries from disk"""
        if not self.cache_dir.exists():
            return
        
        for cache_file in self.cache_dir.glob("*.json"):
            try:
                with open(cache_file, 'r') as f:
                    data = json.load(f)
                
                # Reconstruct selection result
                selection_data = data['selection_result']
                selection_result = ModelSelectionResult(
                    selected_model=selection_data['selected_model'],
                    selection_score=selection_data['selection_score'],
                    selection_time_ms=selection_data['selection_time_ms'],
                    all_scores=selection_data['all_scores'],
                    response_preview=selection_data['response_preview'],
                    timestamp=datetime.fromisoformat(selection_data['timestamp'])
                )
                
                entry = CacheEntry(
                    key=data['key'],
                    prompt_hash=data['prompt_hash'],
                    selected_model=data['selected_model'],
                    selection_result=selection_result,
                    response_text=data['response_text'],
                    created_at=datetime.fromisoformat(data['created_at']),
                    access_count=data['access_count'],
                    last_accessed=datetime.fromisoformat(data['last_accessed']),
                    cache_type=data['cache_type']
                )
                
                # Skip expired entries
                if self._is_expired(entry):
                    cache_file.unlink()
                    continue
                
                self.cache[entry.key] = entry
                
                # Update hash index
                if entry.prompt_hash not in self.prompt_hashes:
                    self.prompt_hashes[entry.prompt_hash] = set()
                self.prompt_hashes[entry.prompt_hash].add(entry.key)
                
            except Exception as e:
                logger.error(f"Error loading cache entry {cache_file}: {e}")
                # Remove corrupted file
                try:
                    cache_file.unlink()
                except:
                    pass