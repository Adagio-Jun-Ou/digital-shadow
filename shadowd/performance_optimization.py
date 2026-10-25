#!/usr/bin/env python3
"""
Performance Optimization - Based on Research

Implements:
- Multi-level caching (Redis, LRU, TTL)
- Lazy evaluation patterns
- Async optimization with task pools
- Memory pooling
- Query optimization
- Performance profiling
"""

import asyncio
import copy
import time
import functools
import weakref
import gc
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Any, Callable, TypeVar, Generic
from collections import OrderedDict, deque
import collections
from dataclasses import dataclass, field
from pathlib import Path
import json
import logging
import threading
from concurrent.futures import ThreadPoolExecutor
import hashlib

logger = logging.getLogger("shadowd.performance")


# 1. Multi-Level Caching

class MultiLevelCache:
    """
    Three-level cache (L1 memory → L2 disk → L3 would be Redis).
    
    From Redis research:
    - L1: In-memory LRU cache (fastest)
    - L2: Persistent disk cache
    - L3: Would be Redis (not implemented, but pattern ready)
    """
    
    def __init__(self, l1_size: int = 1000, l2_dir: Path = None, ttl: int = 3600):
        self.l1_size = l1_size
        self.l2_dir = l2_dir or (Path.home() / "digital-shadow" / "cache")
        self.l2_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(self.l2_dir), 0o700)
        self.ttl = ttl
        
        # L1 cache (in-memory LRU)
        self.l1_cache: OrderedDict = OrderedDict()
        self.l1_metadata: Dict[str, Dict] = {}
        
        # Statistics
        self.hits = {"l1": 0, "l2": 0, "miss": 0}
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get from cache (checks L1 → L2)."""
        cache_key = hashlib.sha256(key.encode()).hexdigest()
        
        # Check L1
        with self.lock:
            if cache_key in self.l1_cache:
                # Check TTL
                metadata = self.l1_metadata.get(cache_key, {})
                if self._is_valid(metadata):
                    # Move to end (most recently used)
                    self.l1_cache.move_to_end(cache_key)
                    self.hits["l1"] += 1
                    return copy.deepcopy(self.l1_cache[cache_key])
                else:
                    # Expired, remove
                    del self.l1_cache[cache_key]
                    self.l1_metadata.pop(cache_key, None)
        
        # Check L2 (disk cache)
        l2_file = self.l2_dir / f"{cache_key}.json"
        if l2_file.exists():
            try:
                data = json.loads(l2_file.read_text())
                # Check TTL
                if self._is_valid(data.get("metadata", {})):
                    cached_value = data["value"]
                    # Promote to L1
                    self._set_l1(cache_key, cached_value)
                    self.hits["l2"] += 1
                    return copy.deepcopy(cached_value)
            except Exception as e:
                logger.warning(f"Error reading L2 cache: {type(e).__name__}")
        
        self.hits["miss"] += 1
        return None
    
    def set(self, key: str, value: Any, ttl: int = None) -> None:
        """Set cache value in L1 and L2."""
        cache_key = hashlib.sha256(key.encode()).hexdigest()
        ttl = ttl or self.ttl
        
        # Set L1
        self._set_l1(cache_key, value, ttl)
        
        # Set L2 (disk cache) - atomic write with fsync
        try:
            l2_file = self.l2_dir / f"{cache_key}.json"
            tmp_fd, tmp_path = os.open(str(l2_file.with_suffix('.json.tmp')),
                os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    json.dump({
                        "value": value,
                        "metadata": {
                            "timestamp": datetime.now(timezone.utc).isoformat(),
                            "ttl": ttl,
                            "key": key
                        }
                    }, f)
                    f.flush()
                    os.fsync(f.fileno())
            except (ValueError, TypeError, RuntimeError):
                os.close(tmp_fd)
                raise
            os.replace(str(l2_file.with_suffix('.json.tmp')), str(l2_file))
        except Exception as e:
            logger.warning(f"Error writing L2 cache: {type(e).__name__}")
    
    def _set_l1(self, cache_key: str, value: Any, ttl: int = None) -> None:
        """Set L1 cache with eviction."""
        with self.lock:
            # Evict if at capacity
            while len(self.l1_cache) >= self.l1_size:
                evicted_key, _ = self.l1_cache.popitem(last=False)
                self.l1_metadata.pop(evicted_key, None)
            
            self.l1_cache[cache_key] = value
            self.l1_metadata[cache_key] = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "ttl": ttl or self.ttl
            }
    
    def _is_valid(self, metadata: Dict) -> bool:
        """Check if cache entry is valid (not expired)."""
        if "timestamp" not in metadata or "ttl" not in metadata:
            return False
        
        try:
            timestamp = datetime.fromisoformat(metadata["timestamp"])
            ttl = metadata["ttl"]
            now = datetime.now(timezone.utc)
            if timestamp.tzinfo is None:
                now = datetime.now(timezone.utc)
            age = (now - timestamp).total_seconds()
            return age < ttl
        except (ValueError, TypeError):
            return False
    
    def invalidate(self, key: str) -> None:
        """Invalidate cache entry."""
        cache_key = hashlib.sha256(key.encode()).hexdigest()
        
        with self.lock:
            self.l1_cache.pop(cache_key, None)
            self.l1_metadata.pop(cache_key, None)
        
        # Remove L2
        l2_file = self.l2_dir / f"{cache_key}.json"
        try:
            l2_file.unlink()
        except FileNotFoundError:
            pass
    
    def clear(self) -> None:
        """Clear all caches."""
        with self.lock:
            self.l1_cache.clear()
            self.l1_metadata.clear()
        
        # Clear L2
        for cache_file in list(self.l2_dir.glob("*.json")):
            try:
                cache_file.unlink()
            except FileNotFoundError:
                pass
    
    def get_stats(self) -> Dict:
        """Get cache statistics."""
        total_requests = sum(self.hits.values())
        hit_rate = (self.hits["l1"] + self.hits["l2"]) / total_requests if total_requests > 0 else 0
        
        return {
            "l1_size": len(self.l1_cache),
            "l1_capacity": self.l1_size,
            "hits": dict(self.hits),
            "hit_rate": f"{hit_rate:.2%}",
            "l1_hit_rate": f"{self.hits['l1'] / total_requests:.2%}" if total_requests > 0 else "0%",
            "l2_hit_rate": f"{self.hits['l2'] / total_requests:.2%}" if total_requests > 0 else "0%"
        }


# 2. Lazy Evaluation

def lazy(model_name: str, loader: Callable):
    """
    Lazy loading decorator (from research).
    
    Loads model only when first accessed.
    """
    attr_name = f"_lazy_{model_name}"
    
    @property
    def prop(self):
        if not hasattr(self, attr_name):
            setattr(self, attr_name, loader(self))
        return getattr(self, attr_name)
    
    return prop


class LazyBuffer:
    """
    Lazy evaluation buffer for large datasets.
    
    From research: generator-based processing.
    """
    _MAX_BUFFER_SIZE = 10000
    
    def __init__(self, data_source: Callable, chunk_size: int = 100):
        self.data_source = data_source
        self.chunk_size = chunk_size
        self._buffer: List = []
        self._exhausted = False
        self._generator = None
    
    def __iter__(self):
        """Iterate through data lazily."""
        if self._generator is None:
            self._generator = self._generate()
        return self
    
    def __next__(self):
        """Get next item."""
        if self._generator is None:
            self._generator = self._generate()
        
        return next(self._generator)
    
    def _generate(self):
        """Generate data in chunks."""
        for chunk in self.data_source():
            for item in chunk:
                yield item
    
    def prefetch(self, n: int):
        """Prefetch n items into buffer."""
        n = min(n, self._MAX_BUFFER_SIZE - len(self._buffer))
        for _ in range(n):
            try:
                self._buffer.append(next(self))
            except StopIteration:
                break


# 3. Async Optimization

class AsyncTaskPool:
    """
    Async task pool with rate limiting.
    
    From asyncio research: concurrency control.
    """
    
    def __init__(self, max_concurrent: int = 10, rate_limit: int = 100):
        self.max_concurrent = max_concurrent
        self.rate_limit = rate_limit  # requests per second
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.request_times: List[float] = []
        self.lock = asyncio.Lock()
    
    async def submit(self, coro: Callable, *args, **kwargs) -> Any:
        """Submit task with rate limiting."""
        # Rate limiting
        async with self.lock:
            now = time.time()
            # Remove requests older than 1 second
            self.request_times = [t for t in self.request_times if now - t < 1.0]
            
            # Wait if at rate limit
            if len(self.request_times) >= self.rate_limit:
                sleep_time = 1.0 - (now - self.request_times[0])
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
            
            self.request_times.append(now)
        
        # Concurrency limiting
        async with self.semaphore:
            return await coro(*args, **kwargs)
    
    async def submit_batch(self, tasks: List[Callable]) -> List[Any]:
        """Submit batch of tasks concurrently."""
        results = await asyncio.gather(*[
            self.submit(task) for task in tasks
        ], return_exceptions=True)
        
        return results


class BackgroundWorker:
    """
    Background worker for async tasks.
    
    From research: background processing patterns.
    """
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.tasks: List[asyncio.Task] = []
    
    async def run_in_background(self, func: Callable, *args, **kwargs) -> asyncio.Future:
        """Run function in background (returns Future, not Task)."""
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = asyncio.get_event_loop()
        future = loop.run_in_executor(self.executor, func, *args, **kwargs)
        self.tasks.append(future)
        return future
    
    async def wait_for_all(self):
        """Wait for all background tasks to complete."""
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
            self.tasks.clear()
    
    def shutdown(self):
        """Shutdown worker pool."""
        self.executor.shutdown(wait=False)


# 4. Memory Optimization

class ObjectPool:
    """
    Object pool for memory optimization.
    
    From research: object reuse patterns.
    """
    
    def __init__(self, factory: Callable, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self.pool: List[Any] = []
        self.lock = threading.Lock()
    
    def acquire(self) -> Any:
        """Acquire object from pool."""
        with self.lock:
            if self.pool:
                return self.pool.pop()
        
        # Create new object
        return self.factory()
    
    def release(self, obj: Any) -> None:
        """Release object back to pool."""
        with self.lock:
            if len(self.pool) < self.max_size:
                self.pool.append(obj)
    
    def clear(self) -> None:
        """Clear pool."""
        with self.lock:
            self.pool.clear()


class WeakRefCache:
    """
    Weak reference cache (from research).
    
    Allows garbage collection of cached objects.
    """
    
    def __init__(self):
        self.cache: Dict[str, weakref.ref] = {}
        self._lock = threading.Lock()
    
    def get(self, key: str) -> Optional[Any]:
        """Get object from weak reference."""
        with self._lock:
            ref = self.cache.get(key)
            if ref is None:
                return None
            
            obj = ref()
            if obj is None:
                self.cache.pop(key, None)
                return None
            
            return obj
    
    def set(self, key: str, obj: Any) -> None:
        """Set object with weak reference."""
        with self._lock:
            self.cache[key] = weakref.ref(obj)
    
    def __contains__(self, key: str) -> bool:
        return self.get(key) is not None


# 5. Query Optimization

class QueryOptimizer:
    """
    Query optimization utilities.
    
    From research: batch processing and indexing.
    """
    _MAX_INDEXES = 100
    
    def __init__(self, batch_size: int = 100):
        self.batch_size = batch_size
        self.indexes: Dict[str, Dict] = {}
        self._lock = threading.Lock()
    
    def create_index(self, collection: str, key: str, items: List[Dict]) -> None:
        """Create index on collection."""
        with self._lock:
            if len(self.indexes) >= self._MAX_INDEXES:
                oldest = next(iter(self.indexes))
                del self.indexes[oldest]
            self.indexes[f"{collection}:{key}"] = {
                item[key]: copy.deepcopy(item)
                for item in items
                if key in item
            }
    
    def query_index(self, collection: str, key: str, value: Any) -> Optional[Dict]:
        """Query using index."""
        with self._lock:
            index = self.indexes.get(f"{collection}:{key}", {})
            result = index.get(value)
            return copy.deepcopy(result) if result else None
    
    async def batch_query(self, query_func: Callable, items: List[Any]) -> List[Any]:
        """Execute queries in batches."""
        results = []
        
        for i in range(0, len(items), self.batch_size):
            batch = items[i:i + self.batch_size]
            batch_results = await asyncio.gather(*[
                query_func(item) for item in batch
            ])
            results.extend(batch_results)
        
        return results
    
    def stream_results(self, items: List[Any], chunk_size: int = 10):
        """Stream results in chunks."""
        for i in range(0, len(items), chunk_size):
            yield items[i:i + chunk_size]


# 6. Performance Profiling

class PerformanceProfiler:
    """
    Performance profiling utilities.
    
    From research: cProfile, timing, memory tracking.
    """
    
    _MAX_TIMINGS_PER_NAME = 1000
    _MAX_MEMORY_SNAPSHOTS = 1000
    
    def __init__(self):
        self._lock = threading.Lock()
        self.timings: Dict[str, collections.deque] = {}
        self.call_counts: Dict[str, int] = {}
        self.memory_snapshots: List[Dict] = []
    
    def time_it(self, name: str):
        """Decorator to time function execution."""
        def decorator(func):
            @functools.wraps(func)
            async def async_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return await func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    self._record_timing(name, elapsed)
            
            @functools.wraps(func)
            def sync_wrapper(*args, **kwargs):
                start = time.time()
                try:
                    return func(*args, **kwargs)
                finally:
                    elapsed = time.time() - start
                    self._record_timing(name, elapsed)
            
            if asyncio.iscoroutinefunction(func):
                return async_wrapper
            else:
                return sync_wrapper
        return decorator
    
    def _record_timing(self, name: str, elapsed: float) -> None:
        """Record timing."""
        with self._lock:
            if name not in self.timings:
                self.timings[name] = collections.deque(maxlen=self._MAX_TIMINGS_PER_NAME)
            self.timings[name].append(elapsed)
            self.call_counts[name] = self.call_counts.get(name, 0) + 1
    
    def record_memory(self, name: str = "snapshot") -> None:
        """Record memory usage."""
        import tracemalloc
        
        if not tracemalloc.is_tracing():
            tracemalloc.start()
        
        current, peak = tracemalloc.get_traced_memory()
        
        with self._lock:
            self.memory_snapshots.append({
                "name": name,
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "current_mb": current / 1024 / 1024,
                "peak_mb": peak / 1024 / 1024
            })
            if len(self.memory_snapshots) > self._MAX_MEMORY_SNAPSHOTS:
                self.memory_snapshots = self.memory_snapshots[-self._MAX_MEMORY_SNAPSHOTS:]
    
    def get_stats(self) -> Dict:
        """Get performance statistics."""
        with self._lock:
            stats = {
                "timings": {},
                "total_calls": sum(self.call_counts.values()),
                "memory_snapshots": len(self.memory_snapshots)
            }
            
            for name, times in self.timings.items():
                if times:
                    stats["timings"][name] = {
                        "calls": self.call_counts.get(name, 0),
                        "total_time": sum(times),
                        "avg_time": sum(times) / len(times),
                        "min_time": min(times),
                        "max_time": max(times)
                    }
            
            if self.memory_snapshots:
                last_snapshot = self.memory_snapshots[-1]
                stats["memory"] = {
                    "current_mb": last_snapshot["current_mb"],
                    "peak_mb": last_snapshot["peak_mb"]
                }
            
            return stats


# Integration: Performance-Optimized Shadow AI

class PerformanceOptimizedShadowAI:
    """
    Shadow AI with all performance optimizations integrated.
    """
    
    def __init__(self, shadow_ai):
        self.shadow_ai = shadow_ai
        
        # Caching
        self.cache = MultiLevelCache(l1_size=500, ttl=3600)
        self.embedding_cache = WeakRefCache()
        
        # Async
        self.task_pool = AsyncTaskPool(max_concurrent=20, rate_limit=100)
        self.background_worker = BackgroundWorker(max_workers=4)
        
        # Query optimization
        self.query_optimizer = QueryOptimizer(batch_size=50)
        
        # Profiling
        self.profiler = PerformanceProfiler()
        
        # Object pools
        self.context_pool = ObjectPool(
            factory=lambda: {},
            max_size=100
        )
    
    @functools.lru_cache(maxsize=128)
    def cached_classify_domain(self, query: str) -> str:
        """Classify domain (cached with LRU)."""
        return self.shadow_ai.domains.classify_domain(query)
    
    @functools.lru_cache(maxsize=256)
    def cached_ability_selection(self, query: str) -> tuple:
        """Select abilities (cached with LRU)."""
        # Would call shadow_ai.coordinator.analyze_query
        return ("omni_sense",)
    
    async def optimized_ask(self, question: str, domain: str = None) -> Dict:
        """Optimized ask with caching."""
        # Check cache first
        cache_key = f"ask:{question}:{domain}"
        cached = self.cache.get(cache_key)
        if cached:
            logger.info("Cache hit for: %s", question[:50].replace(chr(10), ' ').replace(chr(13), ''))
            return cached
        
        # Execute with profiling
        @self.profiler.time_it("ask")
        async def _ask():
            return await self.shadow_ai.ask(question, domain)
        
        result = await _ask()
        
        # Cache result
        self.cache.set(cache_key, result)
        
        return result
    
    def get_performance_stats(self) -> Dict:
        """Get performance statistics."""
        return {
            "cache": self.cache.get_stats(),
            "profiler": self.profiler.get_stats(),
            "task_pool": {
                "semaphore": self.task_pool.semaphore._value,
                "request_times": len(self.task_pool.request_times)
            }
        }
    
    async def cleanup(self):
        """Cleanup resources."""
        await self.background_worker.wait_for_all()
        self.background_worker.shutdown()
        self.cache.clear()
        gc.collect()