"""
Cognitive Caching Layer — LRU and TTL-based caching for cognitive engines.

Provides:
1. LRU cache with TTL-based expiration
2. Cache statistics (hit/miss ratios, evictions)
3. Thread-safe operations
4. Memory-bounded storage with automatic eviction
5. Batch invalidation for cache warming
"""

import copy
import logging
import re
import threading
import time as time_module
import uuid
from collections import OrderedDict, deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]

_MAX_SIZE = 10000
_DEFAULT_TTL_SECONDS = 300


@dataclass
class CacheEntry:
    key: str
    value: Any
    created_at: float = field(default_factory=time_module.monotonic)
    ttl_seconds: int = _DEFAULT_TTL_SECONDS
    access_count: int = 0
    last_accessed: float = field(default_factory=time_module.monotonic)

    def is_expired(self) -> bool:
        return time_module.monotonic() - self.created_at > self.ttl_seconds

    def access(self) -> Any:
        self.access_count += 1
        self.last_accessed = time_module.monotonic
        return self.value


class LRUCache:
    _MAX_SIZE = _MAX_SIZE

    def __init__(self, max_size: int = _MAX_SIZE, default_ttl: int = _DEFAULT_TTL_SECONDS):
        self._lock = threading.Lock()
        self._max_size = max(1, min(_MAX_SIZE, max_size))
        self._default_ttl = max(1, default_ttl)
        self._cache: OrderedDict[str, CacheEntry] = OrderedDict()
        self._access_list: deque = deque(maxlen=1000)

        self._hits: int = 0
        self._misses: int = 0
        self._evictions: int = 0
        self._expirations: int = 0

    def get(self, key: str) -> Optional[Any]:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None:
                self._misses += 1
                return None
            if entry.is_expired():
                self._cache.pop(key, None)
                self._expirations += 1
                self._misses += 1
                return None
            self._cache.move_to_end(key)
            self._hits += 1
            return copy.deepcopy(entry.value)

    def set(self, key: str, value: Any, ttl: Optional[int] = None) -> None:
        ttl = ttl if ttl is not None else self._default_ttl
        ttl = max(1, ttl)
        with self._lock:
            if key in self._cache:
                self._cache.move_to_end(key)
            entry = CacheEntry(key=key, value=copy.deepcopy(value), ttl_seconds=ttl)
            self._cache[key] = entry
            self._cache.move_to_end(key)
            self._evict_if_needed()

    def delete(self, key: str) -> bool:
        with self._lock:
            removed = self._cache.pop(key, None)
            return removed is not None

    def clear(self) -> None:
        with self._lock:
            self._cache.clear()
            self._hits = 0
            self._misses = 0

    def contains(self, key: str) -> bool:
        with self._lock:
            entry = self._cache.get(key)
            if entry is None or entry.is_expired():
                return False
            return True

    def get_or_compute(self, key: str, compute_fn, ttl: Optional[int] = None) -> Any:
        with self._lock:
            result = self.get(key)
            if result is not None:
                return result

        value = compute_fn()
        self.set(key, value, ttl)
        return value

    def invalidate_pattern(self, prefix: str) -> int:
        count = 0
        with self._lock:
            keys_to_remove = [k for k in self._cache if k.startswith(prefix)]
            for k in keys_to_remove:
                self._cache.pop(k, None)
                count += 1
        return count

    def get_size(self) -> int:
        with self._lock:
            return len(self._cache)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            hit_rate = self._hits / max(1, self._hits + self._misses)
            return {
                "size": len(self._cache),
                "max_size": self._max_size,
                "hits": self._hits,
                "misses": self._misses,
                "hit_rate": round(hit_rate, 4),
                "evictions": self._evictions,
                "expirations": self._expirations,
                "default_ttl": self._default_ttl,
            }

    def _evict_if_needed(self) -> None:
        while len(self._cache) > self._max_size:
            self._cache.popitem(last=False)
            self._evictions += 1


class MultiLevelCache:
    def __init__(self, l1_size: int = 1000, l2_size: int = 5000):
        self._l1 = LRUCache(max_size=l1_size, default_ttl=60)
        self._l2 = LRUCache(max_size=l2_size, default_ttl=600)
        self._lock = threading.Lock()

    def get(self, key: str) -> Optional[Any]:
        result = self._l1.get(key)
        if result is not None:
            return result
        result = self._l2.get(key)
        if result is not None:
            self._l1.set(key, result, ttl=60)
        return result

    def set(self, key: str, value: Any, ttl: int = 300) -> None:
        self._l2.set(key, value, ttl=ttl)
        self._l1.set(key, value, ttl=60)

    def delete(self, key: str) -> bool:
        d1 = self._l1.delete(key)
        d2 = self._l2.delete(key)
        return d1 or d2

    def get_stats(self) -> Dict[str, Any]:
        return {
            "l1": self._l1.get_stats(),
            "l2": self._l2.get_stats(),
        }
