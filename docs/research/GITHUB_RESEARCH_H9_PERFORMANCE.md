# Hour 9 Research: Performance Optimization

## Overview
Research on performance optimization patterns for Shadow AI, based on GitHub projects.

---

## Redis (73.7k stars)
**GitHub**: https://github.com/redis/redis

### Key Concepts
- In-memory data store
- Multiple data structures (strings, lists, sets, hashes, sorted sets)
- Pub/sub messaging
- Streams for event sourcing
- Caching with TTL
- Rate limiting
- Session management

### Pattern
```python
import redis.asyncio as redis
import json
from datetime import timedelta

class RedisCache:
    def __init__(self, url: str = "redis://localhost:6379"):
        self.redis = redis.from_url(url)
        self.default_ttl = 3600  # 1 hour
    
    async def get(self, key: str) -> Optional[str]:
        value = await self.redis.get(key)
        return json.loads(value) if value else None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self.default_ttl
        await self.redis.setex(key, ttl, json.dumps(value))
    
    async def delete(self, key: str):
        await self.redis.delete(key)
    
    async def exists(self, key: str) -> bool:
        return await self.redis.exists(key) > 0
    
    async def get_or_set(self, key: str, factory: Callable, ttl: int = None):
        cached = await self.get(key)
        if cached:
            return cached
        
        value = await factory()
        await self.set(key, value, ttl)
        return value
```

---

## HTTPX (15.1k stars)
**GitHub**: https://github.com/encode/httpx

### Key Concepts
- Sync and async HTTP client
- HTTP/2 support
- Connection pooling
- Timeouts
- Async context managers

### Pattern
```python
import httpx
from typing import Optional

class HTTPClient:
    def __init__(self, base_url: str = None, timeout: float = 30.0):
        self.base_url = base_url
        self.timeout = timeout
        self._client = None
    
    async def __aenter__(self):
        self._client = httpx.AsyncClient(
            base_url=self.base_url,
            timeout=self.timeout,
            http2=True,
            follow_redirects=True
        )
        return self
    
    async def __aexit__(self, *args):
        if self._client:
            await self._client.aclose()
    
    async def get(self, url: str, **kwargs) -> httpx.Response:
        return await self._client.get(url, **kwargs)
    
    async def post(self, url: str, **kwargs) -> httpx.Response:
        return await self._client.post(url, **kwargs)
```

---

## Key Patterns for Shadow AI

### 1. Multi-Level Caching
```python
from dataclasses import dataclass
from typing import Any, Optional, Callable
from datetime import timedelta
import hashlib
import json
from pathlib import Path

@dataclass
class CacheEntry:
    data: Any
    expires_at: float
    created_at: float
    metadata: dict

class MultiLevelCache:
    """
    Multi-level caching system for Shadow AI.
    
    Levels:
    1. In-memory (L1) - Fastest, smallest capacity
    2. Redis (L2) - Fast, shared across instances
    3. Disk (L3) - Persistent, slower
    """
    
    def __init__(
        self,
        redis_url: str = None,
        disk_cache_dir: Path = None,
        l1_max_size: int = 1000,
        default_ttl: int = 3600
    ):
        self.l1_cache: dict[str, CacheEntry] = {}
        self.l1_max_size = l1_max_size
        self.default_ttl = default_ttl
        
        self.redis = None
        if redis_url:
            import redis.asyncio as redis
            self.redis = redis.from_url(redis_url)
        
        self.disk_cache_dir = disk_cache_dir or (
            Path.home() / "digital-shadow" / "cache"
        )
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        
        self._l1_order: list[str] = []
    
    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        cache_key = self._hash_key(key)
        
        # L1 Cache
        if cache_key in self.l1_cache:
            entry = self.l1_cache[cache_key]
            if time.time() < entry.expires_at:
                return entry.data
            del self.l1_cache[cache_key]
        
        # L2 Cache (Redis)
        if self.redis:
            data = await self.redis.get(cache_key)
            if data:
                value = json.loads(data)
                self._set_l1(cache_key, value)
                return value
        
        # L3 Cache (Disk)
        disk_path = self.disk_cache_dir / f"{cache_key}.json"
        if disk_path.exists():
            try:
                entry_data = json.loads(disk_path.read_text())
                if time.time() < entry_data["expires_at"]:
                    value = entry_data["data"]
                    self._set_l1(cache_key, value)
                    if self.redis:
                        await self.redis.setex(
                            cache_key,
                            int(entry_data["expires_at"] - time.time()),
                            json.dumps(value)
                        )
                    return value
                disk_path.unlink()
            except:
                pass
        
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        ttl = ttl or self.default_ttl
        cache_key = self._hash_key(key)
        expires_at = time.time() + ttl
        
        # L1
        self._set_l1(cache_key, value)
        
        # L2 (Redis)
        if self.redis:
            await self.redis.setex(cache_key, ttl, json.dumps(value))
        
        # L3 (Disk)
        disk_path = self.disk_cache_dir / f"{cache_key}.json"
        disk_path.write_text(json.dumps({
            "data": value,
            "expires_at": expires_at,
            "created_at": time.time()
        }))
    
    def _set_l1(self, key: str, value: Any):
        if len(self.l1_cache) >= self.l1_max_size:
            oldest = self._l1_order.pop(0)
            if oldest in self.l1_cache:
                del self.l1_cache[oldest]
        
        self.l1_cache[key] = CacheEntry(
            data=value,
            expires_at=time.time() + self.default_ttl,
            created_at=time.time(),
            metadata={}
        )
        self._l1_order.append(key)
    
    async def delete(self, key: str):
        cache_key = self._hash_key(key)
        
        if cache_key in self.l1_cache:
            del self.l1_cache[cache_key]
        
        if self.redis:
            await self.redis.delete(cache_key)
        
        disk_path = self.disk_cache_dir / f"{cache_key}.json"
        if disk_path.exists():
            disk_path.unlink()
    
    async def get_or_set(
        self,
        key: str,
        factory: Callable[[], Any],
        ttl: int = None
    ) -> Any:
        value = await self.get(key)
        if value is not None:
            return value
        
        value = await factory() if asyncio.iscoroutinefunction(factory) else factory()
        await self.set(key, value, ttl)
        return value
```

### 2. Rate Limiting with Redis
```python
import time
from typing import Optional

class RedisRateLimiter:
    """Distributed rate limiting using Redis."""
    
    def __init__(self, redis_client, prefix: str = "ratelimit:"):
        self.redis = redis_client
        self.prefix = prefix
    
    def _key(self, identifier: str) -> str:
        return f"{self.prefix}{identifier}"
    
    async def is_allowed(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple[bool, int]:
        """
        Check if request is allowed.
        
        Returns:
            (is_allowed, remaining_requests)
        """
        key = self._key(identifier)
        now = time.time()
        window_start = now - window_seconds
        
        # Remove old entries
        await self.redis.zremrangebyscore(key, 0, window_start)
        
        # Count current requests
        count = await self.redis.zcard(key)
        
        if count >= max_requests:
            return False, 0
        
        # Add new request
        await self.redis.zadd(key, {str(now): now})
        await self.redis.expire(key, window_seconds)
        
        remaining = max_requests - count - 1
        return True, remaining
    
    async def get_remaining(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> int:
        """Get remaining requests."""
        key = self._key(identifier)
        count = await self.redis.zcard(key)
        return max(0, max_requests - count)
```

### 3. Connection Pooling
```python
import httpx
from contextlib import asynccontextmanager

class ConnectionPool:
    """Managed connection pool for HTTP requests."""
    
    _instance = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._clients = {}
        return cls._instance
    
    async def get_client(
        self,
        base_url: str = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive: int = 20
    ) -> httpx.AsyncClient:
        key = f"{base_url}:{timeout}"
        
        if key not in self._clients:
            self._clients[key] = httpx.AsyncClient(
                base_url=base_url,
                timeout=timeout,
                limits=httpx.Limits(
                    max_connections=max_connections,
                    max_keepalive_connections=max_keepalive
                ),
                http2=True
            )
        
        return self._clients[key]
    
    async def close_all(self):
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()
```

### 4. Async Task Queue
```python
import asyncio
from typing import Callable, Any
from dataclasses import dataclass
from datetime import datetime

@dataclass
class Task:
    id: str
    func: Callable
    args: tuple
    kwargs: dict
    priority: int = 0
    created_at: datetime = None

class AsyncTaskQueue:
    """Priority-based async task queue."""
    
    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: set = set()
        self.results: dict = {}
        self._running = False
    
    async def add_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple = (),
        kwargs: dict = None,
        priority: int = 0
    ) -> str:
        """Add task to queue."""
        task = Task(
            id=task_id,
            func=func,
            args=args,
            kwargs=kwargs or {},
            priority=priority,
            created_at=datetime.now()
        )
        await self.queue.put((priority, task))
        return task_id
    
    async def process_tasks(self):
        """Process tasks from queue."""
        self._running = True
        
        while self._running:
            if len(self.active_tasks) >= self.max_concurrent:
                await asyncio.sleep(0.1)
                continue
            
            try:
                priority, task = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=0.5
                )
                
                coro = self._execute_task(task)
                self.active_tasks.add(asyncio.create_task(coro))
                
            except asyncio.TimeoutError:
                continue
    
    async def _execute_task(self, task: Task):
        """Execute a single task."""
        try:
            if asyncio.iscoroutinefunction(task.func):
                result = await task.func(*task.args, **task.kwargs)
            else:
                result = task.func(*task.args, **task.kwargs)
            
            self.results[task.id] = {
                "status": "completed",
                "result": result,
                "completed_at": datetime.now()
            }
        except Exception as e:
            self.results[task.id] = {
                "status": "failed",
                "error": str(e),
                "completed_at": datetime.now()
            }
        finally:
            self.queue.task_done()
    
    async def get_result(self, task_id: str, timeout: float = None) -> Any:
        """Wait for and return task result."""
        start = time.time()
        
        while True:
            if task_id in self.results:
                return self.results[task_id]
            
            if timeout and (time.time() - start) > timeout:
                raise TimeoutError(f"Task {task_id} timed out")
            
            await asyncio.sleep(0.1)
    
    def stop(self):
        """Stop processing tasks."""
        self._running = False
```

---

## Summary

- **Redis**: In-memory caching, rate limiting, session management
- **HTTPX**: Async HTTP client with connection pooling
- **Multi-level caching**: L1 (memory) → L2 (Redis) → L3 (disk)
- **Rate limiting**: Sliding window with Redis sorted sets
- **Connection pooling**: Reuse HTTP connections
- **Async task queue**: Priority-based task processing