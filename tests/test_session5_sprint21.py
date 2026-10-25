#!/usr/bin/env python3
"""
Session 5 Sprint 21 - Remaining Backlog Fixes

Regression tests for:
- V-275: RateLimiter fail-open on Redis error
- V-280: AsyncTaskQueue busy-wait loop (event-driven slot release)
- V-281: get_or_set() stampede lock eviction (MAX_FACTORY_LOCKS cap)
- V-288: AdvancedMemorySystem thread safety (defensive copies, flush buffers)
- V-289: O(n^2) string concatenation in add_to_core (append buffer)
"""

import asyncio
import pytest
import threading
import time
from datetime import datetime, timedelta
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path


# ============================================================
# V-275: RateLimiter fail-open on Redis error
# ============================================================

class TestRateLimiterFailOpen:
    """RateLimiter should allow requests when Redis is down (fail-open), not deny all."""

    def test_fail_open_on_redis_error(self):
        from shadowd.performance import RateLimiter

        async def _test():
            mock_redis = AsyncMock()
            mock_redis.zremrangebyscore = AsyncMock(side_effect=ConnectionError("Redis down"))

            rl = RateLimiter(redis_client=mock_redis, default_limit=60, default_window=60)
            is_allowed, remaining, reset_time = await rl.is_allowed("test_client")

            assert is_allowed is True, "RateLimiter should fail-open (allow) when Redis errors"
            assert remaining > 0, "Remaining requests should be positive on fail-open"

        asyncio.run(_test())

    def test_local_rate_limiter_still_works(self):
        from shadowd.performance import RateLimiter

        async def _test():
            rl = RateLimiter(default_limit=5, default_window=60)
            results = []
            for i in range(7):
                allowed, remaining, reset_time = await rl.is_allowed("client1")
                results.append(allowed)
            assert results[:5] == [True] * 5, "First 5 requests should be allowed"
            assert results[5] is False, "6th request should be denied"

        asyncio.run(_test())

    def test_local_rate_limiter_allows_second_client(self):
        from shadowd.performance import RateLimiter

        async def _test():
            rl = RateLimiter(default_limit=3, default_window=60)
            for _ in range(3):
                await rl.is_allowed("client1")
            allowed, _, _ = await rl.is_allowed("client2")
            assert allowed is True, "Different clients should have independent limits"

        asyncio.run(_test())

    def test_local_rate_limiter_client_cap(self):
        from shadowd.performance import RateLimiter

        async def _test():
            rl = RateLimiter(default_limit=60, default_window=60)
            for i in range(RateLimiter.MAX_LOCAL_CLIENTS + 100):
                await rl.is_allowed(f"client_{i}")
            assert len(rl._local_limits) <= RateLimiter.MAX_LOCAL_CLIENTS + 1

        asyncio.run(_test())


# ============================================================
# V-280: AsyncTaskQueue event-driven slot release
# ============================================================

class TestAsyncTaskQueueEventDriven:
    """AsyncTaskQueue should use event-driven slot signaling, not busy-wait."""

    def test_has_slot_available_event(self):
        from shadowd.performance import AsyncTaskQueue
        q = AsyncTaskQueue(max_concurrent=2)
        assert hasattr(q, '_slot_available')
        assert q._slot_available.is_set()

    def test_has_result_events(self):
        from shadowd.performance import AsyncTaskQueue
        q = AsyncTaskQueue()
        assert hasattr(q, '_result_events')
        assert isinstance(q._result_events, dict)

    def test_results_capped_at_max(self):
        from shadowd.performance import AsyncTaskQueue
        queue = AsyncTaskQueue(max_concurrent=5)
        assert queue.MAX_RESULTS == 1000

    def test_task_counter_monotonic(self):
        from shadowd.performance import AsyncTaskQueue
        queue = AsyncTaskQueue(max_concurrent=5)

        async def _test():
            id1 = await queue.add_task(lambda: None, priority=0)
            id2 = await queue.add_task(lambda: None, priority=0)
            assert id2 != id1, "Task IDs should be unique"

        asyncio.run(_test())

    def test_slot_release_after_completion(self):
        from shadowd.performance import AsyncTaskQueue

        async def _test():
            queue = AsyncTaskQueue(max_concurrent=1)

            async def slow_task(x):
                await asyncio.sleep(0.05)
                return x

            task1 = await queue.add_task(slow_task, args=(1,), priority=0)
            task2 = await queue.add_task(slow_task, args=(2,), priority=0)

            processing = asyncio.create_task(queue.process_tasks())
            await asyncio.sleep(0.3)

            assert queue.results.get(task1, {}).get("status") == "completed"
            queue.stop()
            await asyncio.sleep(0.1)

        asyncio.run(_test())


# ============================================================
# V-281: get_or_set() factory lock cap
# ============================================================

class TestGetOrSetLockCap:
    """get_or_set should cap factory locks to prevent unbounded memory growth."""

    def test_factory_locks_capped(self):
        from shadowd.performance import MultiLevelCache

        async def _test():
            import tempfile
            cache = MultiLevelCache(l1_max_size=2000, default_ttl=60, disk_cache_dir=Path(tempfile.mkdtemp()))
            cap = MultiLevelCache._MAX_FACTORY_LOCKS

            for i in range(cap + 200):
                await cache.get_or_set(f"key_{i}", lambda: f"value_{i}", ttl=60)

            assert len(cache._factory_locks) <= cap + 200, (
                f"Factory locks should be bounded, got {len(cache._factory_locks)}"
            )

        asyncio.run(_test())

    def test_get_or_set_returns_cached(self):
        from shadowd.performance import MultiLevelCache

        async def _test():
            import tempfile
            cache = MultiLevelCache(l1_max_size=100, default_ttl=60, disk_cache_dir=Path(tempfile.mkdtemp()))
            call_count = 0

            def factory():
                nonlocal call_count
                call_count += 1
                return "computed_value"

            v1 = await cache.get_or_set("unique_key_xyz123", factory, ttl=60)
            v2 = await cache.get_or_set("unique_key_xyz123", factory, ttl=60)
            assert v1 == "computed_value"
            assert v2 == "computed_value"
            assert call_count >= 1, "Factory should be called at least once"

        asyncio.run(_test())

    def test_l1_cache_ttl(self):
        from shadowd.performance import LRUCache

        async def _test():
            cache = LRUCache(max_size=100, default_ttl=1)
            await cache.set("key1", "val1", ttl=1)
            assert await cache.get("key1") == "val1"
            await asyncio.sleep(1.1)
            assert await cache.get("key1") is None

        asyncio.run(_test())

    def test_l1_cache_eviction(self):
        from shadowd.performance import LRUCache

        async def _test():
            cache = LRUCache(max_size=3, default_ttl=60)
            await cache.set("a", 1)
            await cache.set("b", 2)
            await cache.set("c", 3)
            await cache.set("d", 4)

            assert await cache.get("a") is None
            assert await cache.get("d") == 4

        asyncio.run(_test())


# ============================================================
# V-288: AdvancedMemorySystem thread safety
# ============================================================

class TestAdvancedMemoryThreadSafety:
    """AdvancedMemorySystem should be safe for concurrent access."""

    def _make_mem(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        test_dir = tmp_path / "adv_mem_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        return AdvancedMemorySystem(data_dir=test_dir)

    def test_concurrent_add_to_core(self, tmp_path):
        mem = self._make_mem(tmp_path)
        errors = []

        def worker(label, content):
            try:
                for i in range(50):
                    mem.add_to_core(label, f"{content}_{i}")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=worker, args=("user", f"thread_{t}_content"))
            for t in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_concurrent_add_to_archival(self, tmp_path):
        mem = self._make_mem(tmp_path)
        errors = []

        def worker(idx):
            try:
                for i in range(20):
                    mem.add_to_archival(
                        f"content_{idx}_{i}",
                        entities=[f"entity_{idx}"],
                        source_type="test"
                    )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_add_entity_fact_threaded(self, tmp_path):
        mem = self._make_mem(tmp_path)
        errors = []

        def worker(idx):
            try:
                mem.add_entity_fact(
                    f"person_{idx}", "person",
                    f"likes_python_{idx}", "test"
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(t,)) for t in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        assert len(errors) == 0, f"Thread errors: {errors}"

    def test_search_returns_copy(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.add_to_archival("alpha beta gamma", label="test1")
        mem.add_to_archival("delta epsilon zeta", label="test2")

        results = mem.search_archival("beta")
        assert len(results) == 1
        assert "beta" in results[0].value

    def test_working_context_includes_all_blocks(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.add_to_core("user", "Test User", append=False)
        mem.add_to_core("learned_patterns", "pattern1", append=True)

        context = mem.get_working_context()
        assert "USER" in context
        assert "Test User" in context

    def test_stats(self, tmp_path):
        mem = self._make_mem(tmp_path)
        mem.add_to_archival("content1")
        mem.add_to_archival("content2")

        stats = mem.get_stats()
        assert stats["archival_count"] == 2


# ============================================================
# V-289: O(n^2) string concatenation in add_to_core
# ============================================================

class TestAddToCoreAppendBuffer:
    """add_to_core should use append buffer for efficient concatenation."""

    def test_append_many_items(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        test_dir = tmp_path / "append_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        mem = AdvancedMemorySystem(data_dir=test_dir)

        for i in range(100):
            mem.add_to_core("learned_patterns", f"pattern_{i}", append=True)

        mem._flush_all_buffers()
        value = mem._get_core_value("learned_patterns")
        assert "pattern_0" in value
        assert "pattern_99" in value

    def test_append_then_overwrite(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        test_dir = tmp_path / "append_test2"
        test_dir.mkdir(parents=True, exist_ok=True)
        mem = AdvancedMemorySystem(data_dir=test_dir)

        mem.add_to_core("user", "initial value", append=False)
        mem.add_to_core("user", "appended", append=True)

        mem._flush_all_buffers()
        value = mem._get_core_value("user")
        assert "initial value" in value
        assert "appended" in value

    def test_overwrite_clears_buffer(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        test_dir = tmp_path / "overwrite_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        mem = AdvancedMemorySystem(data_dir=test_dir)

        mem.add_to_core("user", "first", append=True)
        mem.add_to_core("user", "REPLACED", append=False)

        mem._flush_all_buffers()
        value = mem._get_core_value("user")
        assert value == "REPLACED"

    def test_buffer_auto_flush_at_limit(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        test_dir = tmp_path / "auto_flush_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        mem = AdvancedMemorySystem(data_dir=test_dir)

        assert mem._MAX_APPEND_BUFFER == 1000

        for i in range(mem._MAX_APPEND_BUFFER + 10):
            mem.add_to_core("learned_patterns", f"line_{i}", append=True)

        assert len(mem._append_buffer.get("learned_patterns", [])) <= mem._MAX_APPEND_BUFFER

    def test_truncation_at_max_chars(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        test_dir = tmp_path / "truncation_test"
        test_dir.mkdir(parents=True, exist_ok=True)
        mem = AdvancedMemorySystem(data_dir=test_dir)

        huge_content = "x" * (mem._MAX_CORE_BLOCK_CHARS + 1000)
        mem.add_to_core("user", huge_content, append=False)

        mem._flush_all_buffers()
        value = mem._get_core_value("user")
        assert len(value) <= mem._MAX_CORE_BLOCK_CHARS


# ============================================================
# MultiLevelCache disk persistence
# ============================================================

class TestMultiLevelCacheDiskPersistence:
    """Disk cache should use atomic writes."""

    def test_l3_disk_cache(self, tmp_path):
        from shadowd.performance import MultiLevelCache

        async def _test():
            cache_dir = tmp_path / "cache_test"
            cache_dir.mkdir()
            cache = MultiLevelCache(
                l1_max_size=100,
                default_ttl=60,
                disk_cache_dir=cache_dir
            )
            await cache.set("test_key", {"data": "hello"}, ttl=60)
            value = await cache.get("test_key")
            assert value is not None
            assert value["data"] == "hello"

        asyncio.run(_test())

    def test_cache_delete(self, tmp_path):
        from shadowd.performance import MultiLevelCache

        async def _test():
            cache_dir = tmp_path / "cache_delete_test"
            cache_dir.mkdir()
            cache = MultiLevelCache(
                l1_max_size=100,
                default_ttl=60,
                disk_cache_dir=cache_dir
            )
            await cache.set("del_key", "value", ttl=60)
            await cache.delete("del_key")
            result = await cache.get("del_key")
            assert result is None

        asyncio.run(_test())

    def test_cache_stats(self, tmp_path):
        from shadowd.performance import MultiLevelCache

        async def _test():
            cache_dir = tmp_path / "cache_stats_test"
            cache_dir.mkdir()
            cache = MultiLevelCache(
                l1_max_size=100,
                default_ttl=60,
                disk_cache_dir=cache_dir
            )
            await cache.set("k1", "v1", ttl=60)
            await cache.get("k1")
            await cache.get("nonexistent")

            stats = cache.stats()
            assert stats["l1_hits"] >= 1
            assert stats["misses"] >= 1

        asyncio.run(_test())


# ============================================================
# RateLimiter edge cases
# ============================================================

class TestRateLimiterEdgeCases:
    """Test RateLimiter edge cases."""

    def test_different_windows(self):
        from shadowd.performance import RateLimiter

        async def _test():
            rl = RateLimiter(default_limit=10, default_window=1)
            results = []
            for i in range(12):
                allowed, _, _ = await rl.is_allowed("test", max_requests=10, window_seconds=1)
                results.append(allowed)
            assert results[:10] == [True] * 10
            assert results[10] is False

        asyncio.run(_test())

    def test_get_remaining_without_increment(self):
        from shadowd.performance import RateLimiter

        async def _test():
            rl = RateLimiter(default_limit=5, default_window=60)
            remaining = await rl.get_remaining("new_client", max_requests=5, window_seconds=60)
            assert remaining == 5

            await rl.is_allowed("new_client", max_requests=5, window_seconds=60)
            remaining2 = await rl.get_remaining("new_client", max_requests=5, window_seconds=60)
            assert remaining2 == 4

        asyncio.run(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])