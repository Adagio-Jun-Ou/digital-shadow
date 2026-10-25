#!/usr/bin/env python3
"""
Caching and Performance Optimization Module for Shadow AI

Implements:
- Multi-level caching (L1: memory, L2: Redis, L3: disk)
- Rate limiting with sliding window
- Connection pooling
- Async task queue
"""

import copy
import time
import json
import hashlib
import asyncio
import os
import threading
import uuid
from typing import Any, Optional, Callable, Dict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from collections import OrderedDict
import logging

logger = logging.getLogger(__name__)


@dataclass
class CacheEntry:
    """Cache entry with expiration."""
    data: Any
    expires_at: float
    created_at: float
    metadata: Dict = field(default_factory=dict)
    
    def is_expired(self) -> bool:
        return time.time() > self.expires_at
    
    def to_dict(self) -> Dict:
        return {
            "data": copy.deepcopy(self.data) if isinstance(self.data, (dict, list)) else self.data,
            "expires_at": round(self.expires_at, 3),
            "created_at": round(self.created_at, 3),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {}
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CacheEntry':
        return cls(
            data=data["data"],
            expires_at=data["expires_at"],
            created_at=data["created_at"],
            metadata=data.get("metadata", {})
        )


class LRUCache:
    """
    LRU (Least Recently Used) cache implementation.
    
    Thread-safe in-memory cache with automatic eviction.
    """
    
    def __init__(self, max_size: int = 1000, default_ttl: int = 3600):
        self.max_size = max_size
        self.default_ttl = default_ttl
        self.cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._lock = asyncio.Lock()
    
    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        async with self._lock:
            cache_key = self._hash_key(key)
            
            if cache_key not in self.cache:
                return None
            
            entry = self.cache[cache_key]
            
            if entry.is_expired():
                del self.cache[cache_key]
                return None
            
            self.cache.move_to_end(cache_key)
            return entry.data
    
    async def set(self, key: str, value: Any, ttl: int = None):
        async with self._lock:
            cache_key = self._hash_key(key)
            ttl = ttl or self.default_ttl
            
            if cache_key in self.cache:
                self.cache.move_to_end(cache_key)
            
            self.cache[cache_key] = CacheEntry(
                data=value,
                expires_at=time.time() + ttl,
                created_at=time.time()
            )
            
            while len(self.cache) > self.max_size:
                self.cache.popitem(last=False)
    
    async def delete(self, key: str) -> bool:
        async with self._lock:
            cache_key = self._hash_key(key)
            
            if cache_key in self.cache:
                del self.cache[cache_key]
                return True
            return False
    
    async def clear(self):
        async with self._lock:
            self.cache.clear()
    
    async def cleanup_expired(self):
        async with self._lock:
            expired = [
                key for key, entry in self.cache.items()
                if entry.is_expired()
            ]
            for key in expired:
                del self.cache[key]
    
    def stats(self) -> Dict:
        return {
            "size": len(self.cache),
            "max_size": self.max_size,
            "entries": [
                {"key": k[:16], "expires_in": round(max(0, v.expires_at - time.time()), 3)}
                for k, v in list(self.cache.items())[:10]
            ]
        }


class MultiLevelCache:
    """
    Multi-level caching system.
    
    L1: In-memory (LRU cache) - Fastest
    L2: Redis (if available) - Distributed
    L3: Disk - Persistent
    """
    
    def __init__(
        self,
        l1_max_size: int = 1000,
        default_ttl: int = 3600,
        disk_cache_dir: Path = None,
        redis_client = None
    ):
        self.default_ttl = default_ttl
        self.l1 = LRUCache(max_size=l1_max_size, default_ttl=default_ttl)
        self.redis = redis_client
        self.disk_cache_dir = disk_cache_dir or (
            Path.home() / "digital-shadow" / "cache"
        )
        self.disk_cache_dir.mkdir(parents=True, exist_ok=True)
        os.chmod(str(self.disk_cache_dir), 0o700)
        
        self._l1_hits = 0
        self._l2_hits = 0
        self._l3_hits = 0
        self._misses = 0
        self._counter_lock = threading.Lock()
        self._factory_locks: Dict[str, asyncio.Lock] = {}
    
    def _hash_key(self, key: str) -> str:
        return hashlib.sha256(key.encode()).hexdigest()
    
    async def get(self, key: str) -> Optional[Any]:
        """Get value from cache (L1 -> L2 -> L3)."""
        cache_key = self._hash_key(key)
        
        # L1 Cache
        value = await self.l1.get(cache_key)
        if value is not None:
            with self._counter_lock:
                self._l1_hits += 1
            return copy.deepcopy(value) if isinstance(value, (dict, list)) else value
        
        # L2 Cache (Redis)
        if self.redis:
            try:
                data = await self.redis.get(cache_key)
                if data:
                    value = json.loads(data)
                    await self.l1.set(cache_key, value)
                    with self._counter_lock:
                        self._l2_hits += 1
                    return copy.deepcopy(value) if isinstance(value, (dict, list)) else value
            except (ConnectionError, OSError, json.JSONDecodeError, TypeError, ValueError) as e:
                logger.warning("Redis get error: %s", type(e).__name__)
        
        # L3 Cache (Disk)
        disk_path = self.disk_cache_dir / f"{cache_key}.json"
        try:
            entry_data = json.loads(disk_path.read_text())
            entry = CacheEntry.from_dict(entry_data)
            
            if entry.is_expired():
                try:
                    disk_path.unlink()
                except FileNotFoundError:
                    pass
            else:
                await self.l1.set(cache_key, entry.data)
                
                if self.redis:
                    try:
                        ttl = int(entry.expires_at - time.time())
                        await self.redis.setex(
                            cache_key,
                            ttl,
                            json.dumps(entry.data)
                        )
                    except (ConnectionError, OSError, json.JSONDecodeError):
                        pass
                
                with self._counter_lock:
                    self._l3_hits += 1
                return copy.deepcopy(entry.data) if isinstance(entry.data, (dict, list)) else entry.data
        except FileNotFoundError:
            pass
        except (json.JSONDecodeError, ValueError, OSError) as e:
            logger.warning("Disk cache read error: %s", type(e).__name__)
        
        with self._counter_lock:
            self._misses += 1
        return None
    
    async def set(self, key: str, value: Any, ttl: int = None):
        """Set value in all cache levels."""
        cache_key = self._hash_key(key)
        ttl = ttl or self.default_ttl
        expires_at = time.time() + ttl
        
        # L1
        await self.l1.set(cache_key, value, ttl)
        
        # L2 (Redis)
        if self.redis:
            try:
                await self.redis.setex(cache_key, ttl, json.dumps(value))
            except (ConnectionError, OSError, TypeError, ValueError) as e:
                logger.warning("Redis set error: %s", type(e).__name__)
        # L3 (Disk)
        disk_path = self.disk_cache_dir / f"{cache_key}.json"
        try:
            entry = CacheEntry(
                data=value,
                expires_at=expires_at,
                created_at=time.time()
            )
            import tempfile
            import os
            tmp_fd, tmp_path = tempfile.mkstemp(dir=str(self.disk_cache_dir), suffix=".json.tmp")
            try:
                with os.fdopen(tmp_fd, 'w') as f:
                    f.write(json.dumps(entry.to_dict()))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(tmp_path, str(disk_path))
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Disk cache write error: %s", type(e).__name__)
    
    async def delete(self, key: str):
        """Delete from all cache levels."""
        cache_key = self._hash_key(key)
        
        await self.l1.delete(cache_key)
        
        if self.redis:
            try:
                await self.redis.delete(cache_key)
            except (ConnectionError, OSError, json.JSONDecodeError):
                pass
        
        disk_path = self.disk_cache_dir / f"{cache_key}.json"
        try:
            disk_path.unlink()
        except FileNotFoundError:
            pass
    
    _MAX_FACTORY_LOCKS = 1000

    async def get_or_set(
        self,
        key: str,
        factory: Callable,
        ttl: int = None
    ) -> Any:
        """Get value or compute and cache it (stampede-safe with lock eviction)."""
        value = await self.get(key)
        if value is not None:
            return value

        cache_key = self._hash_key(key)
        if cache_key not in self._factory_locks:
            if len(self._factory_locks) >= self._MAX_FACTORY_LOCKS:
                oldest = next(iter(self._factory_locks))
                del self._factory_locks[oldest]
            self._factory_locks[cache_key] = asyncio.Lock()

        async with self._factory_locks[cache_key]:
            value = await self.get(key)
            if value is not None:
                return value

            if asyncio.iscoroutinefunction(factory):
                value = await factory()
            else:
                value = factory()

            await self.set(key, value, ttl)
            return value
    
    async def cleanup_expired(self):
        """Clean up expired entries from disk cache."""
        await self.l1.cleanup_expired()
        
        for cache_file in self.disk_cache_dir.glob("*.json"):
            try:
                entry_data = json.loads(cache_file.read_text())
                entry = CacheEntry.from_dict(entry_data)
                if entry.is_expired():
                    cache_file.unlink()
            except (OSError, json.JSONDecodeError, ValueError):
                pass
    
    def stats(self) -> Dict:
        with self._counter_lock:
            total = self._l1_hits + self._l2_hits + self._l3_hits + self._misses
            return {
                "l1_hits": self._l1_hits,
                "l2_hits": self._l2_hits,
                "l3_hits": self._l3_hits,
                "misses": self._misses,
                "hit_rate": round((self._l1_hits + self._l2_hits + self._l3_hits) / max(1, total), 4),
                "l1_stats": self.l1.stats()
            }


class RateLimiter:
    """
    Rate limiter with sliding window.
    
    Supports both in-memory and Redis-backed rate limiting.
    """
    
    MAX_LOCAL_CLIENTS = 10000

    def __init__(
        self,
        redis_client=None,
        prefix: str = "ratelimit:",
        default_limit: int = 60,
        default_window: int = 60
    ):
        self.redis = redis_client
        self.prefix = prefix
        self.default_limit = default_limit
        self.default_window = default_window
        
        self._local_limits: Dict[str, list] = {}
        self._lock = asyncio.Lock()
    
    def _key(self, identifier: str) -> str:
        return f"{self.prefix}{identifier}"
    
    async def is_allowed(
        self,
        identifier: str,
        max_requests: int = None,
        window_seconds: int = None
    ) -> tuple:
        """
        Check if request is allowed.
        
        Returns:
            (is_allowed, remaining_requests, reset_time)
        """
        max_requests = max_requests or self.default_limit
        window_seconds = window_seconds or self.default_window
        
        if self.redis:
            return await self._check_redis(identifier, max_requests, window_seconds)
        else:
            return await self._check_local(identifier, max_requests, window_seconds)
    
    async def _check_redis(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple:
        """Redis-backed rate limiting."""
        key = self._key(identifier)
        now = time.time()
        window_start = now - window_seconds
        
        try:
            # Remove old entries
            await self.redis.zremrangebyscore(key, 0, window_start)
            
            # Count current requests
            count = await self.redis.zcard(key)
            
            if count >= max_requests:
                # Get reset time
                earliest = await self.redis.zrange(key, 0, 0, withscores=True)
                if earliest:
                    reset_time = earliest[0][1] + window_seconds
                else:
                    reset_time = now + window_seconds
                
                return False, 0, reset_time
            
            # Add new request
            await self.redis.zadd(key, {str(now): now})
            await self.redis.expire(key, window_seconds)
            
            remaining = max_requests - count - 1
            return True, remaining, now + window_seconds
            
        except Exception as e:
            # Fail open on *any* Redis-side failure — a broken cache must not
            # take down the whole API. Catching base Exception is intentional:
            # `redis-py` raises many different exception classes (redis.exceptions,
            # asyncio.TimeoutError, custom Connection*Error, etc.) and an
            # over-narrow filter has bitten this codebase before.
            logger.warning("Redis rate limit error (failing open): %s", type(e).__name__)
            return True, max_requests - 1, now + window_seconds
    
    async def _check_local(
        self,
        identifier: str,
        max_requests: int,
        window_seconds: int
    ) -> tuple:
        """Local in-memory rate limiting."""
        async with self._lock:
            key = self._key(identifier)
            now = time.time()
            window_start = now - window_seconds
            
            if key not in self._local_limits:
                if len(self._local_limits) >= self.MAX_LOCAL_CLIENTS:
                    oldest_key = min(
                        self._local_limits,
                        key=lambda k: self._local_limits[k][0] if self._local_limits[k] else float('inf')
                    )
                    del self._local_limits[oldest_key]
                self._local_limits[key] = []
            
            # Remove old entries
            self._local_limits[key] = [
                t for t in self._local_limits[key]
                if t > window_start
            ]
            if len(self._local_limits[key]) > max_requests * 2:
                self._local_limits[key] = self._local_limits[key][-max_requests:]
            
            count = len(self._local_limits[key])
            
            if count >= max_requests:
                reset_time = self._local_limits[key][0] + window_seconds
                return False, 0, reset_time
            
            self._local_limits[key].append(now)
            remaining = max_requests - count - 1
            return True, remaining, now + window_seconds
    
    async def get_remaining(
        self,
        identifier: str,
        max_requests: int = None,
        window_seconds: int = None
    ) -> int:
        """Get remaining requests without incrementing."""
        max_requests = max_requests or self.default_limit
        window_seconds = window_seconds or self.default_window
        
        if self.redis:
            key = self._key(identifier)
            count = await self.redis.zcard(key)
            return max(0, max_requests - count)
        else:
            async with self._lock:
                key = self._key(identifier)
                if key not in self._local_limits:
                    return max_requests
                
                window_start = time.time() - window_seconds
                count = sum(1 for t in self._local_limits[key] if t > window_start)
                return max(0, max_requests - count)


class AsyncTaskQueue:
    """
    Priority-based async task queue with concurrency limits.
    """
    
    MAX_RESULTS = 1000

    def __init__(self, max_concurrent: int = 10):
        self.max_concurrent = max_concurrent
        self.queue: asyncio.PriorityQueue = asyncio.PriorityQueue()
        self.active_tasks: set = set()
        self.results: Dict[str, Dict] = {}
        self._running = False
        self._task_counter = 0
        self._slot_available = asyncio.Event()
        self._slot_available.set()
        self._result_events: Dict[str, asyncio.Event] = {}
        self._MAX_RESULT_EVENTS = 2000
    
    async def add_task(
        self,
        func: Callable,
        args: tuple = (),
        kwargs: Dict = None,
        priority: int = 0,
        task_id: str = None
    ) -> str:
        """Add task to queue."""
        if not task_id:
            task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        await self.queue.put((
            priority,
            datetime.now(timezone.utc),
            task_id,
            func,
            args or (),
            kwargs or {}
        ))
        
        return task_id
    
    async def process_tasks(self):
        """Process tasks from queue with event-driven slot waiting."""
        self._running = True

        while self._running:
            try:
                priority, created_at, task_id, func, args, kwargs = await asyncio.wait_for(
                    self.queue.get(),
                    timeout=1.0
                )
            except asyncio.TimeoutError:
                continue

            if len(self.active_tasks) >= self.max_concurrent:
                self._slot_available.clear()
                await self._slot_available.wait()
                await self.queue.put((priority, created_at, task_id, func, args, kwargs))
                continue

            task = asyncio.create_task(
                self._execute_task(task_id, func, args, kwargs)
            )
            self.active_tasks.add(task)
            task.add_done_callback(self._on_task_done)
    
    def _on_task_done(self, task):
        """Callback when a task completes — frees a slot."""
        self.active_tasks.discard(task)
        if len(self.active_tasks) < self.max_concurrent:
            self._slot_available.set()

    async def _execute_task(
        self,
        task_id: str,
        func: Callable,
        args: tuple,
        kwargs: Dict
    ):
        """Execute a single task."""
        try:
            if asyncio.iscoroutinefunction(func):
                result = await func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            self.results[task_id] = {
                "status": "completed",
                "result": result,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
        except Exception as e:
            self.results[task_id] = {
                "status": "failed",
                "error": type(e).__name__,
                "completed_at": datetime.now(timezone.utc).isoformat()
            }
        finally:
            if len(self.results) > self.MAX_RESULTS:
                oldest_keys = list(self.results.keys())[:len(self.results) - self.MAX_RESULTS]
                for k in oldest_keys:
                    del self.results[k]
            event = self._result_events.pop(task_id, None)
            if event:
                event.set()
    
    async def get_result(
        self,
        task_id: str,
        timeout: float = None
    ) -> Dict:
        """Wait for and return task result."""
        if task_id in self.results:
            return self.results[task_id]

        event = self._result_events.get(task_id)
        if event is None:
            event = asyncio.Event()
            if len(self._result_events) >= self._MAX_RESULT_EVENTS:
                oldest = next(iter(self._result_events))
                del self._result_events[oldest]
            self._result_events[task_id] = event

        if timeout:
            try:
                await asyncio.wait_for(event.wait(), timeout=timeout)
            except asyncio.TimeoutError:
                return {
                    "status": "timeout",
                    "error": f"Task {task_id} timed out"
                }
        else:
            await event.wait()

        return self.results.get(task_id, {
            "status": "unknown",
            "error": f"Task {task_id} result not found"
        })
    
    def stop(self):
        """Stop processing tasks."""
        self._running = False
    
    def stats(self) -> Dict:
        return {
            "queue_size": self.queue.qsize(),
            "active_tasks": len(self.active_tasks),
            "completed_tasks": len(self.results),
            "max_concurrent": self.max_concurrent
        }


class ConnectionPool:
    """
    HTTP connection pool manager.
    
    Reuses connections for efficient HTTP requests.
    """
    
    _instance = None
    _creation_lock = threading.Lock()
    _MAX_CLIENTS = 50
    
    def __new__(cls):
        with cls._creation_lock:
            if cls._instance is None:
                cls._instance = super().__new__(cls)
                cls._instance._clients = {}
                cls._instance._lock_instance = None
            return cls._instance
    
    @property
    def _lock(self):
        if self._lock_instance is None:
            self._lock_instance = asyncio.Lock()
        return self._lock_instance
    
    async def get_client(
        self,
        base_url: str = None,
        timeout: float = 30.0,
        max_connections: int = 100,
        max_keepalive: int = 20
    ):
        """Get or create HTTP client."""
        try:
            import httpx
        except ImportError:
            raise ImportError("httpx required: pip install httpx")
        
        key = f"{base_url}:{timeout}"
        
        async with self._lock:
            if len(self._clients) >= self._MAX_CLIENTS:
                oldest_key = next(iter(self._clients))
                try:
                    await self._clients[oldest_key].aclose()
                except (OSError, RuntimeError):
                    pass
                del self._clients[oldest_key]
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
        """Close all clients."""
        async with self._lock:
            for client in self._clients.values():
                await client.aclose()
            self._clients.clear()


if __name__ == "__main__":
    print("Performance Module")
    print("  - LRUCache: In-memory LRU cache with TTL")
    print("  - MultiLevelCache: L1 (memory) -> L2 (Redis) -> L3 (disk)")
    print("  - RateLimiter: Sliding window rate limiting")
    print("  - AsyncTaskQueue: Priority-based task queue")
    print("  - ConnectionPool: HTTP connection pooling")