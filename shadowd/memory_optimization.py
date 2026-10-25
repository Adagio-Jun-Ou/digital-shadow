#!/usr/bin/env python3
"""
Memory Optimization - Efficient retrieval and caching

Implements:
- Smart caching (LRU + TTL)
- Query optimization
- Memory compression
- Predictive prefetch
"""

import asyncio
import copy
import re
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any
from collections import OrderedDict
from functools import lru_cache
import hashlib
import json
from pathlib import Path
import logging

logger = logging.getLogger("shadowd.memory_optimization")


_SENTINEL = object()


class SmartCache:
    """
    LRU Cache with TTL (Time To Live).
    
    Caches:
    - Query results
    - Context calculations
    - Ability responses
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self._lock = threading.Lock()
        
        self.cache: OrderedDict = OrderedDict()
        self.timestamps: Dict[str, datetime] = {}
        self.ttls: Dict[str, int] = {}
        
        # Stats
        self.hits = 0
        self.misses = 0
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache if valid. Returns _SENTINEL for cache misses to distinguish from cached None."""
        with self._lock:
            if key not in self.cache:
                self.misses += 1
                return _SENTINEL
            
            # Check TTL
            timestamp = self.timestamps.get(key)
            ttl = self.ttls.get(key, self.default_ttl)
            
            if timestamp and (datetime.now(timezone.utc) - timestamp).total_seconds() > ttl:
                # Expired
                self._remove(key)
                self.misses += 1
                return _SENTINEL
            
            # Cache hit - move to end (most recently used)
            self.cache.move_to_end(key)
            self.hits += 1
            
            return self.cache[key]
    
    def set(self, key: str, value: Any, ttl: int = None):
        """Set cache value with TTL."""
        with self._lock:
            if key in self.cache:
                self._remove(key)
            
            # Evict oldest if at capacity
            while len(self.cache) >= self.max_size:
                self.cache.popitem(last=False)
            
            self.cache[key] = value
            self.timestamps[key] = datetime.now(timezone.utc)
            self.ttls[key] = ttl or self.default_ttl
    
    def _remove(self, key: str):
        """Remove from cache (caller must hold _lock)."""
        self.cache.pop(key, None)
        self.timestamps.pop(key, None)
        self.ttls.pop(key, None)
    
    def memoize(self, func):
        """Decorator to memoize function results."""
        def wrapper(*args, **kwargs):
            key = self._make_key(func.__name__, args, kwargs)
            
            cached = self.get(key)
            if cached is not _SENTINEL:
                return cached
            
            result = func(*args, **kwargs)
            self.set(key, result)
            
            return result
        
        return wrapper
    
    async def amemoize(self, func):
        """Async decorator for memoization."""
        async def wrapper(*args, **kwargs):
            key = self._make_key(func.__name__, args, kwargs)
            
            cached = self.get(key)
            if cached is not _SENTINEL:
                return cached
            
            result = await func(*args, **kwargs)
            self.set(key, result)
            
            return result
        
        return wrapper
    
    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        """Create cache key from function call."""
        # Hash args and kwargs
        args_str = json.dumps(args, sort_keys=True, default=str)
        kwargs_str = json.dumps(kwargs, sort_keys=True, default=str)
        
        combined = f"{func_name}:{args_str}:{kwargs_str}"
        return hashlib.sha256(combined.encode()).hexdigest()
    
    def invalidate(self, pattern: str = None):
        """Invalidate cache entries matching pattern."""
        with self._lock:
            if pattern is None:
                self.cache.clear()
                self.timestamps.clear()
                self.ttls.clear()
                return
            
            keys_to_remove = [
                k for k in self.cache.keys()
                if pattern in k
            ]
            
            for key in keys_to_remove:
                self._remove(key)
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        with self._lock:
            total_requests = self.hits + self.misses
            hit_rate = self.hits / total_requests if total_requests > 0 else 0
            
            return {
                "size": len(self.cache),
                "max_size": self.max_size,
                "hits": self.hits,
                "misses": self.misses,
                "hit_rate": f"{hit_rate:.2%}"
            }


class QueryOptimizer:
    _MAX_QUERY_PATTERNS = 5000
    
    """
    Optimizes frequent queries.
    
    Provides:
    - Query pattern detection
    - Result caching
    - Predictive prefetch
    """
    
    def __init__(self, cache: SmartCache = None):
        self.cache = cache or SmartCache()
        self.query_patterns: Dict[str, int] = {}
        self.recent_queries: List[str] = []
        self.prefetch_queue: List[str] = []
    
    def optimize_query(self, query: str) -> str:
        """Optimize query for retrieval."""
        if len(query) > 10000:
            query = query[:10000]
        # Normalize query
        query_normalized = query.lower().strip()
        
        # Track query frequency
        self.query_patterns[query_normalized] = self.query_patterns.get(query_normalized, 0) + 1
        if len(self.query_patterns) > self._MAX_QUERY_PATTERNS:
            oldest = list(self.query_patterns.keys())[:len(self.query_patterns) - self._MAX_QUERY_PATTERNS]
            for k in oldest:
                del self.query_patterns[k]
        
        # Track recent queries
        self.recent_queries.append(query_normalized)
        if len(self.recent_queries) > 100:
            self.recent_queries = self.recent_queries[-100:]
        
        # Check if query should be prefetched
        if self.query_patterns[query_normalized] > 3:
            # Frequent query - add to prefetch
            if query_normalized not in self.prefetch_queue:
                self.prefetch_queue.append(query_normalized)
        
        return query_normalized
    
    def should_prefetch(self) -> List[str]:
        """Get queries to prefetch."""
        # Clear prefetch queue
        to_prefetch = self.prefetch_queue[:5]  # Top 5
        self.prefetch_queue = []
        
        return to_prefetch
    
    def get_similar_queries(self, query: str) -> List[str]:
        """Find similar queries (not cached — query_patterns is mutable)."""
        similar = []
        query_words = set(query.split())
        
        for historic_query in self.query_patterns.keys():
            historic_words = set(historic_query.split())
            
            # Calculate Jaccard similarity
            intersection = query_words & historic_words
            union = query_words | historic_words
            
            if union:
                similarity = len(intersection) / len(union)
                if similarity > 0.5:
                    similar.append(historic_query)
        
        return similar[:10]
    
    def get_patterns(self) -> Dict:
        """Get query patterns."""
        sorted_patterns = sorted(
            self.query_patterns.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]
        
        return {
            "top_queries": sorted_patterns,
            "unique_queries": len(self.query_patterns),
            "total_queries": sum(self.query_patterns.values())
        }


class MemoryCompressor:
    """
    Compresses memory for efficient storage.
    
    Features:
    - Summary generation
    - Deduplication
    - Importance weighting
    """
    
    def __init__(self):
        self.importance_keywords = {
            "commitment": ["call", "promise", "tomorrow", "deadline", "meeting"],
            "critical": ["urgent", "important", "asap", "critical"],
            "preference": ["like", "prefer", "always", "never"]
        }
    
    def compress_memories(self, memories: List[Dict], target_size: int = 100) -> List[Dict]:
        """
        Compress memories to target size.
        
        Strategy:
        1. Keep all high-importance memories
        2. Summarize similar memories
        3. Discard low-importance duplicates
        """
        if len(memories) <= target_size:
            return memories
        
        # Score each memory
        scored_memories = [
            (self._score_importance(m), m)
            for m in memories
        ]
        
        # Sort by importance
        scored_memories.sort(key=lambda x: x[0], reverse=True)
        
        # Take top target_size
        compressed = [m for score, m in scored_memories[:target_size]]
        
        # Group remaining by similarity
        remaining = [m for score, m in scored_memories[target_size:]]
        
        # Create summaries for similar groups
        summaries = self._create_summaries(remaining)
        
        return compressed + summaries
    
    def _score_importance(self, memory: Dict) -> float:
        """Score memory importance (0-1)."""
        content = str(memory).lower()
        score = 0.0
        
        for category, keywords in self.importance_keywords.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', content) for kw in keywords):
                score += 0.3
        
        # Boost recent memories
        if "timestamp" in memory:
            try:
                ts = datetime.fromisoformat(memory["timestamp"])
                days_old = (datetime.now(timezone.utc) - ts).days
                recency_score = max(0.0, 1.0 - (days_old / 30))
                score += recency_score * 0.3
            except (ValueError, TypeError, KeyError):
                pass
        
        return min(1.0, score)
    
    def _create_summaries(self, memories: List[Dict]) -> List[Dict]:
        """Create summary memories."""
        if not memories:
            return []
        
        # Group by type
        by_type = {}
        for m in memories:
            m_type = m.get("type", "unknown")
            by_type.setdefault(m_type, []).append(m)
        
        summaries = []
        for m_type, group in by_type.items():
            if len(group) > 1:
                summary = {
                    "type": f"{m_type}_summary",
                    "content": f"Summary of {len(group)} {m_type} events",
                    "count": len(group),
                    "first_seen": group[0].get("timestamp"),
                    "last_seen": group[-1].get("timestamp"),
                    "summarized": True
                }
                summaries.append(summary)
        
        return summaries


class PredictivePrefetch:
    _MAX_CONTEXTS = 500
    _MAX_QUERIES_PER_CONTEXT = 100

    """
    Predictive prefetch for memory retrieval.
    
    Predicts what will be needed next and prefetches.
    """
    
    def __init__(self):
        self.context_prefetch: Dict[str, List[str]] = {}
    
    def record_access(self, context: str, query: str):
        """Record an access pattern."""
        if context not in self.context_prefetch:
            if len(self.context_prefetch) >= self._MAX_CONTEXTS:
                oldest = next(iter(self.context_prefetch))
                del self.context_prefetch[oldest]
            self.context_prefetch[context] = []
        self.context_prefetch[context].append(query)
        if len(self.context_prefetch[context]) > self._MAX_QUERIES_PER_CONTEXT:
            self.context_prefetch[context] = self.context_prefetch[context][-self._MAX_QUERIES_PER_CONTEXT:]
    
    def predict_next(self, context: str, current_query: str) -> List[str]:
        """Predict next queries."""
        predictions = []
        
        # Get queries from same context
        context_queries = self.context_prefetch.get(context, [])
        
        if context_queries:
            # Find patterns after similar queries
            for i, query in enumerate(context_queries[:-1]):
                if query == current_query:
                    # Next query might be similar to what followed
                    predictions.append(context_queries[i + 1])
        
        # Most common queries for this context
        from collections import Counter
        query_counts = Counter(context_queries)
        common = [q for q, count in query_counts.most_common(5) if q != current_query]
        
        predictions.extend(common[:3])
        
        return list(set(predictions))[:5]
    
    def get_prefetch_candidates(self) -> List[str]:
        """Get queries to prefetch based on patterns."""
        candidates = []
        
        # Find contexts with frequent access
        for context, queries in self.context_prefetch.items():
            if len(queries) > 5:
                # Get most common queries for this context
                from collections import Counter
                common = Counter(queries).most_common(3)
                candidates.extend([q for q, _ in common])
        
        return list(set(candidates))[:10]


class MemoryOptimizer:
    """
    Main memory optimization coordinator.
    
    Integrates:
    - Smart caching
    - Query optimization
    - Memory compression
    - Predictive prefetch
    """
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "optimized")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Components
        self.cache = SmartCache(max_size=2000)
        self.query_optimizer = QueryOptimizer(self.cache)
        self.compressor = MemoryCompressor()
        self.prefetch = PredictivePrefetch()
        
        # Statistics
        self.optimizations = 0
    
    def optimize_retrieval(self, query: str, context: str = None) -> Dict:
        """
        Optimize a retrieval request.
        
        Returns optimized query and prefetch suggestions.
        """
        self.optimizations += 1
        
        # Optimize query
        optimized_query = self.query_optimizer.optimize_query(query)
        
        # Check cache
        cache_key = f"{context}:{optimized_query}"
        cached = self.cache.get(cache_key)
        if cached is _SENTINEL:
            cached = None
        
        # Record access for prefetch
        if context:
            self.prefetch.record_access(context, optimized_query)
        
        # Get prefetch candidates
        prefetch_candidates = []
        if context:
            prefetch_candidates = self.prefetch.predict_next(context, optimized_query)
        
        return {
            "optimized_query": optimized_query,
            "cached_result": cached,
            "prefetch_suggestions": prefetch_candidates,
            "cache_stats": self.cache.get_stats()
        }
    
    def store_result(self, query: str, result: Any, context: str = None, ttl: int = 3600):
        """Store result in cache."""
        cache_key = f"{context}:{query}"
        self.cache.set(cache_key, result, ttl)
    
    def compress_storage(self, memories: List[Dict], target_size: int = 100) -> List[Dict]:
        """Compress memories for storage."""
        return self.compressor.compress_memories(memories, target_size)
    
    def get_optimization_stats(self) -> Dict:
        """Get optimization statistics."""
        return {
            "optimizations": self.optimizations,
            "cache_stats": self.cache.get_stats(),
            "query_patterns": self.query_optimizer.get_patterns(),
            "prefetch_patterns": len(self.prefetch.context_prefetch)
        }