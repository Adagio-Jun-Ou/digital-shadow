# AI System Performance Optimization Research

## Executive Summary

This research document provides comprehensive performance optimization techniques for AI systems, specifically tailored for Shadow AI. It covers caching strategies, lazy evaluation, async optimization, memory management, query optimization, and profiling approaches.

---

## 1. Caching Strategies

### 1.1 Redis Caching Patterns

#### Key Caching Strategies

**1. Cache-Aside (Lazy Loading)**
```python
# Application manages cache
async def get_user(user_id: str):
    # Check cache first
    cached = await redis.get(f"user:{user_id}")
    if cached:
        return json.loads(cached)
    
    # Cache miss - load from database
    user = await db.get_user(user_id)
    await redis.setex(f"user:{user_id}", 3600, json.dumps(user))
    return user
```

**2. Write-Through Cache**
```python
# Update cache synchronously with database
async def update_user(user_id: str, data: dict):
    await db.update_user(user_id, data)
    await redis.setex(f"user:{user_id}", 3600, json.dumps(data))
```

**3. Write-Behind (Write-Back)**
```python
# Async cache updates
write_queue = asyncio.Queue()

async def cache_writer():
    while True:
        key, value = await write_queue.get()
        await redis.setex(key, 3600, json.dumps(value))

async def update_user_async(user_id: str, data: dict):
    await db.update_user(user_id, data)
    await write_queue.put((f"user:{user_id}", data))
```

**4. Refresh-Ahead**
```python
# Proactively refresh hot data
async def refresh_popular_items():
    popular_keys = await redis.zrevrange("popular:items", 0, 100)
    for key in popular_keys:
        if await redis.ttl(key) < 300:  # Less than 5 min TTL
            data = await db.load(key)
            await redis.setex(key, 3600, json.dumps(data))
```

#### Redis Distributed Locks (Redlock Algorithm)

```python
import uuid
import time
from redis.asyncio import Redis

class RedisLock:
    """Distributed lock implementation using Redlock algorithm."""
    
    def __init__(self, redis_clients: list[Redis], lock_name: str, 
                 ttl: int = 10000):
        self.clients = redis_clients
        self.lock_name = lock_name
        self.ttl = ttl
        self.identifier = str(uuid.uuid4())
        self.quorum = len(redis_clients) // 2 + 1
    
    async def acquire(self) -> bool:
        start_time = time.time()
        acquired = 0
        
        for client in self.clients:
            try:
                if await client.set(
                    self.lock_name, 
                    self.identifier, 
                    nx=True, 
                    px=self.ttl
                ):
                    acquired += 1
            except Exception:
                continue
        
        elapsed = (time.time() - start_time) * 1000
        validity = self.ttl - elapsed - 2  # Drift compensation
        
        if acquired >= self.quorum and validity > 0:
            return True
        
        await self.release()
        return False
    
    async def release(self):
        script = """
        if redis.call("get", KEYS[1]) == ARGV[1] then
            return redis.call("del", KEYS[1])
        else
            return 0
        end
        """
        for client in self.clients:
            try:
                await client.eval(script, [self.lock_name], [self.identifier])
            except Exception:
                continue
```

### 1.2 LRU/LFU/ARC Cache Implementations

#### LRU Cache (Built-in Python)

```python
from functools import lru_cache
from typing import Optional
import hashlib

# Basic LRU cache with size limit
@lru_cache(maxsize=1000)
def compute_embedding(text: str) -> list[float]:
    # Expensive embedding computation
    return model.encode(text)

# Cache with TTL (Time-To-Live)
from datetime import datetime, timedelta
from functools import wraps

def timed_lru_cache(seconds: int, maxsize: int = 128):
    """LRU cache with TTL expiration."""
    def decorator(func):
        func = lru_cache(maxsize=maxsize)(func)
        func.lifetime = timedelta(seconds=seconds)
        func.expiration = datetime.utcnow() + func.lifetime
        
        @wraps(func)
        def wrapped(*args, **kwargs):
            if datetime.utcnow() >= func.expiration:
                func.cache_clear()
                func.expiration = datetime.utcnow() + func.lifetime
            return func(*args, **kwargs)
        
        wrapped.cache_info = func.cache_info
        wrapped.cache_clear = func.cache_clear
        return wrapped
    return decorator

@timed_lru_cache(seconds=300, maxsize=100)
async def get_cached_response(query: str) -> Optional[str]:
    return await database.fetch_response(query)
```

#### LFU Cache Implementation

```python
from collections import defaultdict
from typing import Hashable, TypeVar
import heapq

T = TypeVar('T')

class LFUCache:
    """Least Frequently Used cache implementation."""
    
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.cache: dict[Hashable, T] = {}
        self.frequency: dict[Hashable, int] = defaultdict(int)
        self.min_heap: list[tuple[int, Hashable]] = []
    
    def get(self, key: Hashable) -> tuple[bool, T]:
        if key not in self.cache:
            return False, None
        
        # Increment frequency
        self.frequency[key] += 1
        heapq.heappush(self.min_heap, (self.frequency[key], key))
        return True, self.cache[key]
    
    def put(self, key: Hashable, value: T) -> None:
        if key in self.cache:
            self.cache[key] = value
            self.frequency[key] += 1
            return
        
        if len(self.cache) >= self.capacity:
            self._evict()
        
        self.cache[key] = value
        self.frequency[key] = 1
        heapq.heappush(self.min_heap, (1, key))
    
    def _evict(self):
        while self.min_heap:
            freq, key = heapq.heappop(self.min_heap)
            if freq == self.frequency[key]:
                del self.cache[key]
                del self.frequency[key]
                break
```

### 1.3 Multi-Level Caching

```python
from dataclasses import dataclass
from typing import Optional, Any
import time

@dataclass
class CacheLayer:
    name: str
    cache: dict[str, Any]
    ttl: dict[str, float]
    max_size: int
    default_ttl: int

class MultiLevelCache:
    """L1 (in-memory) -> L2 (Redis) -> L3 (Database) cache hierarchy."""
    
    def __init__(self, redis_client, db_client):
        self.l1 = CacheLayer(
            name="L1_memory",
            cache={},
            ttl={},
            max_size=1000,
            default_ttl=60
        )
        self.l2 = CacheLayer(
            name="L2_redis",
            cache=redis_client,
            ttl={},
            max_size=100000,
            default_ttl=3600
        )
        self.db = db_client
        self.stats = {"l1_hits": 0, "l2_hits": 0, "misses": 0}
    
    async def get(self, key: str) -> tuple[Optional[Any], str]:
        # Check L1 cache
        if key in self.l1.cache:
            if time.time() < self.l1.ttl.get(key, float('inf')):
                self.stats["l1_hits"] += 1
                return self.l1.cache[key], "L1"
            else:
                del self.l1.cache[key]
                del self.l1.ttl[key]
        
        # Check L2 cache (Redis)
        redis_value = await self.l2.cache.get(key)
        if redis_value:
            self.stats["l2_hits"] += 1
            # Promote to L1
            self._add_to_l1(key, redis_value)
            return redis_value, "L2"
        
        # Cache miss - fetch from database
        self.stats["misses"] += 1
        db_value = await self.db.fetch(key)
        if db_value:
            # Populate both cache levels
            await self._add_to_l2(key, db_value)
            self._add_to_l1(key, db_value)
        
        return db_value, "DB"
    
    def _add_to_l1(self, key: str, value: Any):
        if len(self.l1.cache) >= self.l1.max_size:
            # Simple LRU: remove oldest
            oldest_key = next(iter(self.l1.cache))
            del self.l1.cache[oldest_key]
            del self.l1.ttl[oldest_key]
        
        self.l1.cache[key] = value
        self.l1.ttl[key] = time.time() + self.l1.default_ttl
    
    async def _add_to_l2(self, key: str, value: Any):
        await self.l2.cache.setex(
            key, 
            self.l2.default_ttl, 
            json.dumps(value)
        )
    
    def get_stats(self) -> dict:
        total = sum(self.stats.values())
        return {
            **self.stats,
            "l1_hit_rate": self.stats["l1_hits"] / total if total else 0,
            "l2_hit_rate": self.stats["l2_hits"] / total if total else 0,
        }
```

---

## 2. Lazy Evaluation Patterns

### 2.1 Python Generator-Based Processing

```python
from typing import Iterator, AsyncIterator, TypeVar
import asyncio

T = TypeVar('T')

def lazy_chunk_processor(data: Iterator[T], chunk_size: int) -> Iterator[list[T]]:
    """Process data in chunks using generators - memory efficient."""
    chunk = []
    for item in data:
        chunk.append(item)
        if len(chunk) == chunk_size:
            yield chunk
            chunk = []
    if chunk:
        yield chunk

async def async_stream_processor(
    data_stream: AsyncIterator[T],
    process_fn: callable
) -> AsyncIterator[T]:
    """Async generator for streaming data processing."""
    async for item in data_stream:
        result = await process_fn(item)
        if result is not None:
            yield result

class LazyEmbeddingLoader:
    """Lazy loading of embeddings - only compute when needed."""
    
    def __init__(self, texts: list[str], model):
        self.texts = texts
        self.model = model
        self._cache: dict[int, list[float]] = {}
    
    def __iter__(self):
        return self
    
    def __next__(self) -> tuple[int, list[float]]:
        for idx, text in enumerate(self.texts):
            if idx not in self._cache:
                self._cache[idx] = self.model.encode(text)
            yield idx, self._cache[idx]
        raise StopIteration
    
    def get_embedding(self, idx: int) -> list[float]:
        if idx not in self._cache:
            self._cache[idx] = self.model.encode(self.texts[idx])
        return self._cache[idx]
```

### 2.2 Just-in-Time Computation

```python
from functools import cached_property
from typing import Callable, TypeVar, Generic

T = TypeVar('T')

class LazyProperty(Generic[T]):
    """Descriptor for lazy property evaluation."""
    
    def __init__(self, factory: Callable[[], T]):
        self.factory = factory
        self._value: T = None
        self._computed = False
    
    def __get__(self, obj, objtype=None) -> T:
        if not self._computed:
            self._value = self.factory()
            self._computed = True
        return self._value

class LazyAIModel:
    """Lazy model initialization - only load when accessed."""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self._model = None
    
    @cached_property
    def model(self):
        """Load model only on first access."""
        print("Loading model...")
        from transformers import AutoModel
        return AutoModel.from_pretrained(self.model_path)
    
    @cached_property
    def tokenizer(self):
        """Load tokenizer only on first access."""
        print("Loading tokenizer...")
        from transformers import AutoTokenizer
        return AutoTokenizer.from_pretrained(self.model_path)

class DeferredComputation:
    """Defer expensive computations until required."""
    
    def __init__(self):
        self._computations = []
        self._results = {}
    
    def defer(self, name: str, computation: Callable):
        """Register a deferred computation."""
        self._computations.append((name, computation))
    
    def compute(self, name: str) -> Any:
        """Execute a specific deferred computation."""
        if name not in self._results:
            for comp_name, comp_fn in self._computations:
                if comp_name == name:
                    self._results[name] = comp_fn()
                    break
        return self._results.get(name)
    
    def compute_all(self):
        """Execute all deferred computations."""
        for name, comp_fn in self._computations:
            if name not in self._results:
                self._results[name] = comp_fn()
```

---

## 3. Async Optimization

### 3.1 AsyncIO Patterns for AI

```python
import asyncio
from typing import AsyncGenerator, AsyncContextManager
from contextlib import asynccontextmanager

class AsyncTaskPool:
    """Manage concurrent AI tasks with rate limiting."""
    
    def __init__(self, max_concurrent: int = 10, rate_limit: int = 100):
        self.semaphore = asyncio.Semaphore(max_concurrent)
        self.rate_limit = rate_limit
        self.last_request_time = 0
        self.request_count = 0
        self.lock = asyncio.Lock()
    
    async def execute(self, coro, *args, **kwargs):
        """Execute coroutine with concurrency control."""
        async with self.semaphore:
            # Rate limiting
            async with self.lock:
                now = time.time()
                if now - self.last_request_time < 1.0:
                    if self.request_count >= self.rate_limit:
                        await asyncio.sleep(1.0 - (now - self.last_request_time))
                        self.request_count = 0
                else:
                    self.request_count = 0
                self.last_request_time = time.time()
                self.request_count += 1
            
            return await coro(*args, **kwargs)

class AsyncModelBatch:
    """Batch multiple inference requests for efficiency."""
    
    def __init__(self, model, batch_size: int = 32, timeout: float = 0.01):
        self.model = model
        self.batch_size = batch_size
        self.timeout = timeout
        self.queue: asyncio.Queue = asyncio.Queue()
        self.results: dict[str, asyncio.Future] = {}
        self._worker_task = None
    
    async def start(self):
        """Start the batch processing worker."""
        self._worker_task = asyncio.create_task(self._process_batches())
    
    async def stop(self):
        """Stop the worker gracefully."""
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass
    
    async def infer(self, request_id: str, input_data: Any) -> Any:
        """Submit inference request and wait for result."""
        future = asyncio.Future()
        self.results[request_id] = future
        await self.queue.put((request_id, input_data))
        return await future
    
    async def _process_batches(self):
        """Process queued requests in batches."""
        while True:
            batch = []
            request_ids = []
            
            # Collect batch
            try:
                async with asyncio.timeout(self.timeout):
                    while len(batch) < self.batch_size:
                        request_id, data = await self.queue.get()
                        batch.append(data)
                        request_ids.append(request_id)
            except asyncio.TimeoutError:
                pass
            
            if not batch:
                continue
            
            # Process batch
            try:
                results = await self.model.batch_infer(batch)
                for req_id, result in zip(request_ids, results):
                    if req_id in self.results:
                        self.results[req_id].set_result(result)
                        del self.results[req_id]
            except Exception as e:
                for req_id in request_ids:
                    if req_id in self.results:
                        self.results[req_id].set_exception(e)
                        del self.results[req_id]

# Priority Queue for Task Scheduling
from dataclasses import dataclass, field
from typing import Any
import heapq

@dataclass(order=True)
class PrioritizedTask:
    priority: int
    task_id: str = field(compare=False)
    coro: Any = field(compare=False)

class AsyncTaskScheduler:
    """Priority-based async task scheduler."""
    
    def __init__(self, max_workers: int = 5):
        self.tasks: list[PrioritizedTask] = []
        self.max_workers = max_workers
        self.active_workers = 0
        self.condition = asyncio.Condition()
    
    async def schedule(self, coro, priority: int = 0, task_id: str = None):
        """Schedule a task with priority (lower = higher priority)."""
        task_id = task_id or str(uuid.uuid4())
        prioritized = PrioritizedTask(priority, task_id, coro)
        heapq.heappush(self.tasks, prioritized)
        async with self.condition:
            self.condition.notify()
        return task_id
    
    async def run(self):
        """Execute tasks based on priority."""
        while True:
            async with self.condition:
                while not self.tasks or self.active_workers >= self.max_workers:
                    await self.condition.wait()
                
                if self.tasks:
                    prioritized = heapq.heappop(self.tasks)
                    self.active_workers += 1
            
            asyncio.create_task(self._execute_task(prioritized))
    
    async def _execute_task(self, prioritized: PrioritizedTask):
        try:
            await prioritized.coro
        finally:
            async with self.condition:
                self.active_workers -= 1
                self.condition.notify()
```

### 3.2 Concurrent Execution Patterns

```python
import asyncio
from typing import Sequence, TypeVar, Callable
from concurrent.futures import ThreadPoolExecutor

T = TypeVar('T')
R = TypeVar('R')

async def gather_with_concurrency(
    n: int, 
    *tasks: Sequence[asyncio.coroutine]
) -> list[Any]:
    """Execute coroutines with limited concurrency."""
    semaphore = asyncio.Semaphore(n)
    
    async def bound_task(coro):
        async with semaphore:
            return await coro
    
    return await asyncio.gather(*[bound_task(t) for t in tasks])

async def map_concurrent(
    func: Callable[[T], R],
    items: Sequence[T],
    max_concurrency: int = 10
) -> list[R]:
    """Apply async function to items with concurrency limit."""
    semaphore = asyncio.Semaphore(max_concurrency)
    
    async def bounded_call(item: T) -> R:
        async with semaphore:
            return await func(item)
    
    tasks = [bounded_call(item) for item in items]
    return await asyncio.gather(*tasks)

async def process_with_timeout(
    coro,
    timeout: float,
    default: Any = None
) -> Any:
    """Execute coroutine with timeout."""
    try:
        async with asyncio.timeout(timeout):
            return await coro
    except asyncio.TimeoutError:
        return default

async def retry_with_backoff(
    coro,
    max_retries: int = 3,
    initial_delay: float = 1.0,
    max_delay: float = 60.0,
    backoff_factor: float = 2.0
) -> Any:
    """Execute with exponential backoff retry."""
    delay = initial_delay
    
    for attempt in range(max_retries):
        try:
            return await coro
        except Exception as e:
            if attempt == max_retries - 1:
                raise
            
            await asyncio.sleep(delay)
            delay = min(delay * backoff_factor, max_delay)
    
    raise RuntimeError("Max retries exceeded")

# Background Workers
class BackgroundWorker:
    """Manage background async tasks."""
    
    def __init__(self, max_workers: int = 5):
        self.max_workers = max_workers
        self.tasks: set[asyncio.Task] = set()
        self.semaphore = asyncio.Semaphore(max_workers)
    
    async def submit(self, coro, name: str = None) -> asyncio.Task:
        """Submit task to background worker pool."""
        await self.semaphore.acquire()
        
        task = asyncio.create_task(coro, name=name)
        self.tasks.add(task)
        task.add_done_callback(lambda t: self._task_done(t))
        
        return task
    
    def _task_done(self, task: asyncio.Task):
        self.tasks.discard(task)
        self.semaphore.release()
    
    async def wait_all(self):
        """Wait for all background tasks."""
        if self.tasks:
            await asyncio.gather(*self.tasks, return_exceptions=True)
    
    async def cancel_all(self):
        """Cancel all background tasks."""
        for task in self.tasks:
            task.cancel()
        await asyncio.gather(*self.tasks, return_exceptions=True)
```

---

## 4. Memory Optimization

### 4.1 Memory Pooling and Object Reuse

```python
from typing import TypeVar, Generic, Optional
from dataclasses import dataclass
import threading

T = TypeVar('T')

class ObjectPool(Generic[T]):
    """Thread-safe object pool for reuse."""
    
    def __init__(self, factory: callable, max_size: int = 100):
        self.factory = factory
        self.max_size = max_size
        self._pool: list[T] = []
        self._lock = threading.Lock()
        self._stats = {"created": 0, "reused": 0}
    
    def acquire(self) -> T:
        """Get object from pool or create new."""
        with self._lock:
            if self._pool:
                obj = self._pool.pop()
                self._stats["reused"] += 1
                return obj
        
        obj = self.factory()
        self._stats["created"] += 1
        return obj
    
    def release(self, obj: T):
        """Return object to pool."""
        with self._lock:
            if len(self._pool) < self.max_size:
                self._pool.append(obj)
    
    def get_stats(self) -> dict:
        return self._stats.copy()

@dataclass
class BufferInfo:
    size: int
    ref_count: int

class BufferPool:
    """Pre-allocated buffer pool for large data."""
    
    def __init__(self, buffer_sizes: list[int] = None):
        self.buffer_sizes = buffer_sizes or [1024, 4096, 16384, 65536]
        self.pools: dict[int, list[bytearray]] = {
            size: [] for size in self.buffer_sizes
        }
        self.lock = threading.Lock()
    
    def acquire(self, needed_size: int) -> bytearray:
        """Get buffer of appropriate size."""
        size = next((s for s in self.buffer_sizes if s >= needed_size), None)
        if not size:
            return bytearray(needed_size)
        
        with self.lock:
            if self.pools[size]:
                return self.pools[size].pop()
        
        return bytearray(size)
    
    def release(self, buffer: bytearray):
        """Return buffer to pool."""
        size = len(buffer)
        if size in self.pools:
            with self.lock:
                if len(self.pools[size]) < 10:  # Limit pool size
                    self.pools[size].append(buffer)

class EmbeddingCache:
    """Memory-efficient embedding cache with eviction."""
    
    def __init__(self, max_memory_mb: int = 512):
        self.max_memory = max_memory_mb * 1024 * 1024
        self.cache: dict[str, list[float]] = {}
        self.access_counts: dict[str, int] = {}
        self.current_size = 0
        self.lock = threading.Lock()
    
    def get(self, key: str) -> Optional[list[float]]:
        with self.lock:
            if key in self.cache:
                self.access_counts[key] += 1
                return self.cache[key]
        return None
    
    def put(self, key: str, embedding: list[float]):
        size = len(embedding) * 8  # float64 = 8 bytes
        
        with self.lock:
            # Evict if necessary
            while self.current_size + size > self.max_memory:
                self._evict_one()
            
            if key in self.cache:
                self.current_size -= len(self.cache[key]) * 8
            
            self.cache[key] = embedding
            self.access_counts[key] = 1
            self.current_size += size
    
    def _evict_one(self):
        if not self.cache:
            return
        
        # Evict least frequently used
        lfu_key = min(self.access_counts, key=self.access_counts.get)
        self.current_size -= len(self.cache[lfu_key]) * 8
        del self.cache[lfu_key]
        del self.access_counts[lfu_key]
```

### 4.2 Garbage Collection Tuning

```python
import gc
from typing import Optional
import weakref

class GCOptimizer:
    """Optimize garbage collection for Python applications."""
    
    def __init__(self):
        self.original_thresholds = None
        self.gc_stats = {
            "collections": 0,
            "collected": 0,
            "uncollectable": 0
        }
    
    def tune_for_throughput(self):
        """Optimize GC for high throughput - fewer collections."""
        # Get current thresholds
        self.original_thresholds = gc.get_threshold()
        
        # Set less frequent collections
        # (threshold0, threshold1, threshold2) for generations 0, 1, 2
        gc.set_threshold(700, 10, 10)
        
        # Disable automatic GC and collect manually
        gc.disable()
    
    def tune_for_latency(self):
        """Optimize for low latency - more frequent, smaller collections."""
        gc.set_threshold(100, 5, 5)
        gc.enable()
    
    def manual_collect(self, generation: Optional[int] = None):
        """Manually trigger garbage collection."""
        collected = gc.collect(generation)
        self.gc_stats["collections"] += 1
        self.gc_stats["collected"] += collected
        return collected
    
    def restore_default(self):
        """Restore original GC settings."""
        if self.original_thresholds:
            gc.set_threshold(*self.original_thresholds)
        gc.enable()
    
    def get_stats(self) -> dict:
        return {
            **self.gc_stats,
            "current_thresholds": gc.get_threshold(),
            "garbage_count": len(gc.garbage),
        }

# Weak references for caching large objects
class WeakValueCache:
    """Cache that uses weak references - allows GC of cached items."""
    
    def __init__(self):
        import weakref
        self.cache = weakref.WeakValueDictionary()
    
    def get_or_compute(self, key: str, factory: callable) -> Any:
        """Get cached value or compute and cache it."""
        try:
            return self.cache[key]
        except KeyError:
            value = factory()
            self.cache[key] = value
            return value
```

### 4.3 Memory Profiling

```python
import tracemalloc
import linecache
from typing import Optional
import asyncio

class MemoryProfiler:
    """Profile memory usage during execution."""
    
    def __init__(self):
        self.snapshots = []
        tracemalloc.start()
    
    def take_snapshot(self, label: str = ""):
        """Take memory snapshot."""
        snapshot = tracemalloc.take_snapshot()
        self.snapshots.append((label, snapshot))
        return snapshot
    
    def compare_snapshots(self, idx1: int, idx2: int) -> list:
        """Compare two snapshots."""
        if idx1 >= len(self.snapshots) or idx2 >= len(self.snapshots):
            return []
        
        label1, snap1 = self.snapshots[idx1]
        label2, snap2 = self.snapshots[idx2]
        
        top_stats = snap2.compare_to(snap1, 'lineno')
        return top_stats[:10]
    
    @staticmethod
    def format_bytes(bytes_count: int) -> str:
        for unit in ['B', 'KB', 'MB', 'GB']:
            if bytes_count < 1024:
                return f"{bytes_count:.2f} {unit}"
            bytes_count /= 1024
        return f"{bytes_count:.2f} TB"
    
    def get_top_allocations(self, snapshot_idx: int = -1, limit: int = 10) -> list:
        """Get top memory allocations."""
        if not self.snapshots:
            return []
        
        label, snapshot = self.snapshots[snapshot_idx]
        top_stats = snapshot.statistics('lineno')
        
        return [
            {
                "file": stat.traceback[0].filename,
                "line": stat.traceback[0].lineno,
                "size": self.format_bytes(stat.size),
                "count": stat.count,
            }
            for stat in top_stats[:limit]
        ]
    
    def print_report(self):
        """Print memory usage report."""
        print(f"\n{'='*60}")
        print("Memory Profiler Report")
        print(f"{'='*60}\n")
        
        if len(self.snapshots) < 2:
            print("Need at least 2 snapshots for comparison")
            return
        
        # Compare first and last snapshot
        diffs = self.compare_snapshots(0, -1)
        
        print("Top memory increases:")
        for stat in diffs:
            print(f"{stat}\n")
        
        # Current allocations
        print("\nCurrent top allocations:")
        current_stats = self.snapshots[-1][1].statistics('lineno')
        for stat in current_stats[:5]:
            print(f"{stat}\n")

# Context manager for profiling
@asynccontextmanager
async def profile_memory(label: str = "block"):
    """Profile memory usage of a code block."""
    profiler = MemoryProfiler()
    profiler.take_snapshot(f"before_{label}")
    
    yield profiler
    
    profiler.take_snapshot(f"after_{label}")
    print(f"\nMemory for {label}:")
    diffs = profiler.compare_snapshots(0, -1)
    for stat in diffs[:5]:
        print(f"  {stat}")
```

---

## 5. Query Optimization

### 5.1 Database Index Strategies

```python
from typing import Any
import time

class QueryOptimizer:
    """Optimize database queries for AI systems."""
    
    @staticmethod
    def create_optimal_indexes(conn, table: str, columns: list[str]):
        """Create indexes based on query patterns."""
        for column in columns:
            conn.execute(f"""
                CREATE INDEX IF NOT EXISTS idx_{table}_{column}
                ON {table}({column})
            """)
    
    @staticmethod
    def analyze_query_plan(conn, query: str) -> dict:
        """Analyze query execution plan."""
        result = conn.execute(f"EXPLAIN QUERY PLAN {query}")
        return {
            "plan": result.fetchall(),
            "has_index_scan": "INDEX" in str(result.fetchall()),
            "has_table_scan": "SCAN TABLE" in str(result.fetchall())
        }

class BatchQueryExecutor:
    """Execute queries in batches for efficiency."""
    
    def __init__(self, conn, batch_size: int = 100):
        self.conn = conn
        self.batch_size = batch_size
        self.batch = []
        self.results_buffer = []
    
    async def add_query(self, query: str, params: tuple) -> Any:
        """Add query to batch."""
        self.batch.append((query, params))
        
        if len(self.batch) >= self.batch_size:
            return await self.flush()
        
        return None
    
    async def flush(self) -> list[Any]:
        """Execute all queued queries."""
        if not self.batch:
            return []
        
        results = []
        for query, params in self.batch:
            try:
                cursor = self.conn.execute(query, params)
                results.append(cursor.fetchall())
            except Exception as e:
                results.append({"error": str(e)})
        
        self.batch = []
        return results

class PreparedStatements:
    """Cache prepared statements for repeated queries."""
    
    def __init__(self, conn):
        self.conn = conn
        self.statements: dict[str, Any] = {}
    
    def prepare(self, name: str, query: str):
        """Prepare and cache statement."""
        self.statements[name] = self.conn.prepare(query)
    
    def execute(self, name: str, params: tuple = ()) -> Any:
        """Execute prepared statement."""
        if name not in self.statements:
            raise ValueError(f"Statement '{name}' not prepared")
        
        return self.statements[name].execute(params)
    
    def close_all(self):
        """Close all prepared statements."""
        for stmt in self.statements.values():
            stmt.close()
        self.statements.clear()

# Query result streaming
async def stream_query_results(
    conn,
    query: str,
    params: tuple = (),
    chunk_size: int = 1000
) -> AsyncGenerator[list, None]:
    """Stream large query results in chunks."""
    cursor = conn.execute(query, params)
    
    while True:
        chunk = cursor.fetchmany(chunk_size)
        if not chunk:
            break
        yield chunk
```

### 5.2 Streaming Results

```python
import asyncio
from typing import AsyncGenerator, Any
import sqlite3

class StreamingQueryProcessor:
    """Process large datasets via streaming."""
    
    def __init__(self, db_path: str, chunk_size: int = 1000):
        self.db_path = db_path
        self.chunk_size = chunk_size
    
    async def stream_embeddings(
        self,
        query: str,
        params: tuple = ()
    ) -> AsyncGenerator[dict, None]:
        """Stream embeddings from database."""
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        
        try:
            cursor = conn.execute(query, params)
            
            while True:
                rows = cursor.fetchmany(self.chunk_size)
                if not rows:
                    break
                
                for row in rows:
                    yield dict(row)
                
                # Allow other tasks to run
                await asyncio.sleep(0)
        
        finally:
            conn.close()
    
    async def process_stream(
        self,
        query: str,
        processor: callable,
        params: tuple = ()
    ):
        """Process stream with custom processor."""
        async for batch in self.stream_embeddings(query, params):
            await processor(batch)

# Pagination helper
class PaginatedQuery:
    """Efficiently paginate through large results."""
    
    def __init__(self, conn, query: str, page_size: int = 100):
        self.conn = conn
        self.query = query
        self.page_size = page_size
        self.offset = 0
    
    def get_page(self, page: int) -> list[tuple]:
        """Get a specific page of results."""
        offset = (page - 1) * self.page_size
        paginated_query = f"{self.query} LIMIT {self.page_size} OFFSET {offset}"
        return self.conn.execute(paginated_query).fetchall()
    
    async def iterate_all(self) -> AsyncGenerator[list, None]:
        """Iterate through all pages."""
        page = 1
        while True:
            results = self.get_page(page)
            if not results:
                break
            yield results
            page += 1
            await asyncio.sleep(0)  # Yield control
```

---

## 6. Profiling Tools

### 6.1 cProfile Usage

```python
import cProfile
import pstats
from io import StringIO
from typing import Optional
import functools

class Profiler:
    """Comprehensive profiler wrapper."""
    
    def __init__(self):
        self.profiler = cProfile.Profile()
    
    def profile_function(self, func):
        """Decorator to profile a function."""
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            self.profiler.enable()
            try:
                result = func(*args, **kwargs)
                return result
            finally:
                self.profiler.disable()
        return wrapper
    
    def profile_async_function(self, func):
        """Decorator for async functions."""
        @functools.wraps(func)
        async def wrapper(*args, **kwargs):
            self.profiler.enable()
            try:
                result = await func(*args, **kwargs)
                return result
            finally:
                self.profiler.disable()
        return wrapper
    
    def print_stats(self, sortby: str = 'cumulative', limit: int = 20):
        """Print profiling statistics."""
        s = StringIO()
        stats = pstats.Stats(self.profiler, stream=s)
        stats.sort_stats(sortby)
        stats.print_stats(limit)
        print(s.getvalue())
    
    def get_hotspots(self, threshold: float = 0.1) -> list[dict]:
        """Get functions consuming most time."""
        stats = pstats.Stats(self.profiler)
        stats.sort_stats('cumulative')
        
        total_time = stats.total_tt
        hotspots = []
        
        for func, (cc, nc, tt, ct, callers) in stats.stats.items():
            if ct / total_time > threshold:
                hotspots.append({
                    "function": func,
                    "calls": nc,
                    "total_time": tt,
                    "cumulative_time": ct,
                    "percentage": (ct / total_time) * 100
                })
        
        return sorted(hotspots, key=lambda x: x["cumulative_time"], reverse=True)
    
    def save_to_file(self, filename: str):
        """Save profiling data to file."""
        self.profiler.dump_stats(filename)

# Context manager for profiling
class ProfileContext:
    """Profile a code block."""
    
    def __init__(self, sortby: str = 'cumulative', limit: int = 20):
        self.profiler = cProfile.Profile()
        self.sortby = sortby
        self.limit = limit
    
    def __enter__(self):
        self.profiler.enable()
        return self
    
    def __exit__(self, *args):
        self.profiler.disable()
        stats = pstats.Stats(self.profiler)
        stats.sort_stats(self.sortby)
        stats.print_stats(self.limit)
```

### 6.2 Line Profiling

```python
import line_profiler
from functools import wraps

class LineProfilerWrapper:
    """Convenient line profiling wrapper."""
    
    def __init__(self):
        self.profiler = line_profiler.LineProfiler()
    
    def profile(self, func):
        """Decorate function for line profiling."""
        return self.profiler(func)
    
    def print_stats(self):
        """Print line-by-line profiling stats."""
        self.profiler.print_stats()
    
    def get_stats_by_line(self, func_name: str) -> dict:
        """Get per-line timing for specific function."""
        stats = self.profiler.get_stats()
        
        for func_stat in stats.timings:
            if func_name in str(func_stat.code):
                return {
                    "function": func_stat.code,
                    "unit": stats.unit,
                    "lines": [
                        {
                            "line_no": line_no,
                            "hits": hits,
                            "time": time_,
                            "per_hit": time_ / hits if hits else 0
                        }
                        for line_no, hits, time_ in func_stat.timeline
                    ]
                }
        
        return None

# Usage examples:
# profiler = LineProfilerWrapper()
# 
# @profiler.profile
# def expensive_function():
#     ...
#
# profiler.print_stats()
```

### 6.3 Memory Profiling

```python
import tracemalloc
import sys
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime

@dataclass
class MemorySnapshot:
    timestamp: datetime
    size: int
    peak: int
    count: int

class MemoryTracker:
    """Track memory usage over time."""
    
    def __init__(self, interval: float = 1.0):
        self.interval = interval
        self.snapshots: list[MemorySnapshot] = []
        self._running = False
        tracemalloc.start()
    
    def snapshot(self) -> MemorySnapshot:
        """Take current memory snapshot."""
        current, peak = tracemalloc.get_traced_memory()
        count = tracemalloc.get_tracemalloc_memory()
        
        return MemorySnapshot(
            timestamp=datetime.now(),
            size=current,
            peak=peak,
            count=count
        )
    
    def start_monitoring(self):
        """Start periodic monitoring."""
        import asyncio
        
        async def monitor():
            self._running = True
            while self._running:
                self.snapshots.append(self.snapshot())
                await asyncio.sleep(self.interval)
        
        asyncio.create_task(monitor())
    
    def stop_monitoring(self):
        """Stop monitoring."""
        self._running = False
    
    def get_peak_memory(self) -> int:
        """Get peak memory usage."""
        if not self.snapshots:
            return 0
        return max(s.peak for s in self.snapshots)
    
    def get_average_memory(self) -> float:
        """Get average memory usage."""
        if not self.snapshots:
            return 0
        return sum(s.size for s in self.snapshots) / len(self.snapshots)
    
    def get_memory_growth(self) -> list[tuple]:
        """Track memory growth over time."""
        return [
            (s.timestamp, s.size / (1024 * 1024))  # MB
            for s in self.snapshots
        ]
    
    def detect_leaks(self, threshold_mb: float = 10.0) -> bool:
        """Detect potential memory leaks."""
        if len(self.snapshots) < 2:
            return False
        
        initial = self.snapshots[0].size / (1024 * 1024)
        final = self.snapshots[-1].size / (1024 * 1024)
        
        return (final - initial) > threshold_mb

# Object size profiler
class ObjectProfiler:
    """Profile individual objects."""
    
    @staticmethod
    def get_size(obj, seen: Optional[set] = None) -> int:
        """Recursively calculate object size."""
        size = sys.getsizeof(obj)
        if seen is None:
            seen = set()
        
        obj_id = id(obj)
        if obj_id in seen:
            return 0
        
        seen.add(obj_id)
        
        if isinstance(obj, dict):
            size += sum(
                ObjectProfiler.get_size(k, seen) + ObjectProfiler.get_size(v, seen)
                for k, v in obj.items()
            )
        elif hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes, bytearray)):
            try:
                size += sum(ObjectProfiler.get_size(i, seen) for i in obj)
            except TypeError:
                pass
        
        return size
    
    @staticmethod
    def profile_object(obj, name: str = "object"):
        """Get detailed object profile."""
        size = ObjectProfiler.get_size(obj)
        
        return {
            "name": name,
            "type": type(obj).__name__,
            "size_bytes": size,
            "size_kb": size / 1024,
            "size_mb": size / (1024 * 1024),
        }
```

---

## 7. Benchmarking Approach

```python
import time
import asyncio
from typing import Callable, Any
from dataclasses import dataclass
from statistics import mean, stdev
import json

@dataclass
class BenchmarkResult:
    function: str
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    iterations: int
    memory_peak: int
    throughput: float  # ops/sec

class PerformanceBenchmark:
    """Comprehensive benchmarking framework."""
    
    def __init__(self, warmup_iterations: int = 5):
        self.warmup_iterations = warmup_iterations
        self.results: dict[str, BenchmarkResult] = {}
    
    def bench_sync(
        self,
        func: Callable,
        *args,
        iterations: int = 100,
        name: str = None,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark synchronous function."""
        name = name or func.__name__
        
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)
        
        result = BenchmarkResult(
            function=name,
            mean_time=mean(times),
            std_time=stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            iterations=iterations,
            memory_peak=0,
            throughput=iterations / sum(times)
        )
        
        self.results[name] = result
        return result
    
    async def bench_async(
        self,
        func: Callable,
        *args,
        iterations: int = 100,
        name: str = None,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark async function."""
        name = name or func.__name__
        
        # Warmup
        for _ in range(self.warmup_iterations):
            await func(*args, **kwargs)
        
        # Measure
        times = []
        for _ in range(iterations):
            start = time.perf_counter()
            await func(*args, **kwargs)
            times.append(time.perf_counter() - start)
        
        result = BenchmarkResult(
            function=name,
            mean_time=mean(times),
            std_time=stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            iterations=iterations,
            memory_peak=0,
            throughput=iterations / sum(times)
        )
        
        self.results[name] = result
        return result
    
    def bench_with_memory(
        self,
        func: Callable,
        *args,
        iterations: int = 100,
        **kwargs
    ) -> BenchmarkResult:
        """Benchmark with memory tracking."""
        import tracemalloc
        
        tracemalloc.start()
        
        # Warmup
        for _ in range(self.warmup_iterations):
            func(*args, **kwargs)
        
        # Measure time and memory
        times = []
        peak_memory = 0
        
        for _ in range(iterations):
            tracemalloc.reset_peak()
            start = time.perf_counter()
            func(*args, **kwargs)
            times.append(time.perf_counter() - start)
            current, peak = tracemalloc.get_traced_memory()
            peak_memory = max(peak_memory, peak)
        
        tracemalloc.stop()
        
        result = BenchmarkResult(
            function=func.__name__,
            mean_time=mean(times),
            std_time=stdev(times) if len(times) > 1 else 0,
            min_time=min(times),
            max_time=max(times),
            iterations=iterations,
            memory_peak=peak_memory,
            throughput=iterations / sum(times)
        )
        
        self.results[func.__name__] = result
        return result
    
    def compare(self, *names: str) -> str:
        """Compare benchmarked functions."""
        if not names:
            names = list(self.results.keys())
        
        lines = [
            f"\n{'Function':<30} {'Mean (ms)':<12} {'Std (ms)':<12} "
            f"{'Min (ms)':<12} {'Max (ms)':<12} {'Throughput':<12}",
            "=" * 96
        ]
        
        for name in names:
            r = self.results[name]
            lines.append(
                f"{r.function:<30} "
                f"{r.mean_time*1000:<12.3f} "
                f"{r.std_time*1000:<12.3f} "
                f"{r.min_time*1000:<12.3f} "
                f"{r.max_time*1000:<12.3f} "
                f"{r.throughput:<12.1f}"
            )
        
        return "\n".join(lines)
    
    def to_json(self) -> str:
        """Export results as JSON."""
        return json.dumps({
            name: {
                "mean_time": r.mean_time,
                "std_time": r.std_time,
                "min_time": r.min_time,
                "max_time": r.max_time,
                "iterations": r.iterations,
                "memory_peak": r.memory_peak,
                "throughput": r.throughput,
            }
            for name, r in self.results.items()
        }, indent=2)
```

---

## 8. Integration Patterns for Shadow AI

### 8.1 Complete Optimization Pipeline

```python
import asyncio
from typing import Optional, Any
from dataclasses import dataclass
import time

@dataclass
class OptimizationConfig:
    cache_enabled: bool = True
    cache_ttl: int = 3600
    cache_max_size: int = 10000
    
    batch_size: int = 32
    batch_timeout: float = 0.01
    
    max_concurrent_requests: int = 100
    rate_limit_per_second: int = 1000
    
    memory_pool_size: int = 100
    max_memory_mb: int = 512
    
    profiling_enabled: bool = False

class OptimizedAIPipeline:
    """Main optimized pipeline for AI inference."""
    
    def __init__(self, config: OptimizationConfig, model, redis_client, db):
        self.config = config
        self.model = model
        self.redis = redis_client
        self.db = db
        
        # Initialize components
        self.cache = MultiLevelCache(redis_client, db)
        self.task_pool = AsyncTaskPool(
            max_concurrent=config.max_concurrent_requests,
            rate_limit=config.rate_limit_per_second
        )
        self.memory_pool = ObjectPool(
            factory=lambda: bytearray(1024 * 1024),
            max_size=config.memory_pool_size
        )
        self.batch_processor = AsyncModelBatch(
            model=model,
            batch_size=config.batch_size,
            timeout=config.batch_timeout
        )
        
        if config.profiling_enabled:
            self.profiler = Profiler()
        else:
            self.profiler = None
    
    async def initialize(self):
        """Start background workers."""
        await self.batch_processor.start()
    
    async def process(self, input_data: Any) -> Any:
        """Process input with full optimization stack."""
        # Generate cache key
        cache_key = self._generate_cache_key(input_data)
        
        # Check cache first
        if self.config.cache_enabled:
            cached_result, level = await self.cache.get(cache_key)
            if cached_result is not None:
                return cached_result
        
        # Apply rate limiting
        async with self.task_pool:
            # Batch processing
            result = await self.batch_processor.infer(
                request_id=str(uuid.uuid4()),
                input_data=input_data
            )
        
        # Cache result
        if self.config.cache_enabled:
            await self.cache._add_to_l2(cache_key, result)
        
        return result
    
    def _generate_cache_key(self, input_data: Any) -> str:
        """Generate deterministic cache key."""
        import hashlib
        import json
        
        data_str = json.dumps(input_data, sort_keys=True)
        return hashlib.sha256(data_str.encode()).hexdigest()
    
    async def shutdown(self):
        """Cleanup resources."""
        await self.batch_processor.stop()
        if hasattr(self.cache, 'close'):
            await self.cache.close()
```

---

## 9. Best Practices Summary

### Performance Checklist

1. **Caching**
   - Use multi-level caching (L1 in-memory, L2 Redis, L3 database)
   - Implement TTL-based cache invalidation
   - Consider cache warming for hot data
   - Monitor cache hit rates

2. **Lazy Evaluation**
   - Use generators for large data processing
   - Implement lazy loading with `@cached_property`
   - Defer expensive computations until needed
   - Use async generators for streaming

3. **Async Optimization**
   - Use `asyncio.gather()` for concurrent operations
   - Implement rate limiting for API calls
   - Use semaphores to limit concurrency
   - Batch requests when possible

4. **Memory**
   - Use object pools for frequently created objects
   - Monitor memory with `tracemalloc`
   - Implement LFU/LRU eviction for caches
   - Consider weak references for caches

5. **Query**
   - Create appropriate indexes
   - Use prepared statements
   - Stream large results
   - Batch database operations

6. **Profiling**
   - Use `cProfile` for CPU profiling
   - Use `tracemalloc` for memory tracking
   - Profile before optimizing
   - Benchmark after optimizations

---

## References

1. Redis Documentation - Caching Patterns: https://redis.io/docs/latest/develop/clients/patterns/
2. Redis Distributed Locks (Redlock): https://redis.io/docs/latest/develop/clients/patterns/distributed-locks/
3. Python LRU Cache Implementation: https://realpython.com/lru-cache-python/
4. Python Generators and Lazy Evaluation: https://realpython.com/python-generators/
5. Python AsyncIO Documentation: https://docs.python.org/3/library/asyncio.html
6. Python AsyncIO Tasks: https://docs.python.org/3/library/asyncio-task.html
7. Python AsyncIO Queues: https://docs.python.org/3/library/asyncio-queue.html
8. Python Multiprocessing: https://docs.python.org/3/library/multiprocessing.html
9. Python Profiling (cProfile): https://docs.python.org/3/library/profile.html
10. Python Trace Module: https://docs.python.org/3/library/trace.html
11. PostgreSQL Performance Tips: https://postgresql.org/docs/current/performance-tips.html

---

**Document Version**: 1.0  
**Last Updated**: 2026  
**Author**: Research Compilation for Shadow AI