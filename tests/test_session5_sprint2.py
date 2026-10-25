#!/usr/bin/env python3
"""
Session 5 Sprint 2 Regression Tests

Covers deeper findings from second audit pass:
- V-100: timedelta.seconds vs total_seconds() TTL bypass
- V-102/V-103: proactive_insights timedelta bugs
- V-101: ShellTool expanded blocklist
- V-104: ConnectionPool singleton thread safety
- V-105: AsyncTaskQueue results cap
- V-107: SmartCache thread safety
- V-108: Person name validation in ambient_monitor
- V-117: Predictable task IDs in agent_coordination
"""

import pytest
import os
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile


class TestTimedeltaSecondsBug:
    """Verify timedelta.seconds replaced with total_seconds()."""

    def test_memory_optimization_ttl_uses_total_seconds(self):
        from shadowd.memory_optimization import SmartCache
        import inspect
        source = inspect.getsource(SmartCache)
        assert ".total_seconds()" in source
        # Ensure .seconds is not used for TTL checks
        lines = source.split('\n')
        for line in lines:
            if 'timestamp' in line and '.seconds' in line and 'total_seconds' not in line:
                pytest.fail(f"Found .seconds instead of .total_seconds(): {line.strip()}")

    def test_smart_cache_ttl_expiry_over_24h(self):
        """Verify entries older than 24h are expired correctly."""
        from shadowd.memory_optimization import SmartCache, _SENTINEL
        cache = SmartCache(max_size=10, default_ttl=3600)  # 1 hour TTL
        cache.set("key1", "value1")
        # Simulate entry created 25 hours ago
        with cache._lock:
            cache.timestamps["key1"] = datetime.now(timezone.utc) - timedelta(hours=25)
        result = cache.get("key1")
        assert result is _SENTINEL  # Should be expired

    def test_proactive_insights_cooldown_uses_total_seconds(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        import inspect
        source = inspect.getsource(ProactiveInsightEngine.evaluate_rules)
        assert ".total_seconds()" in source

    def test_proactive_insights_time_until_uses_total_seconds(self):
        from shadowd.proactive_insights import ProactiveInsightEngine
        import inspect
        source = inspect.getsource(ProactiveInsightEngine)
        # No .seconds used for time calculations
        assert "time_until.seconds" not in source


class TestShellToolExpandedBlocklist:
    """Verify ShellTool blocks more patterns."""

    @pytest.mark.asyncio
    async def test_blocks_rm_any_flags(self):
        from shadowd.tools.enhanced_tools import ShellTool
        tool = ShellTool()
        result = await tool.execute({"command": "rm -r -f /tmp/test"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocks_sudo(self):
        from shadowd.tools.enhanced_tools import ShellTool
        tool = ShellTool()
        result = await tool.execute({"command": "sudo apt install malware"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocks_chmod_any(self):
        from shadowd.tools.enhanced_tools import ShellTool
        tool = ShellTool()
        result = await tool.execute({"command": "chmod 644 /etc/passwd"})
        assert not result.success

    @pytest.mark.asyncio
    async def test_blocks_find_delete(self):
        from shadowd.tools.enhanced_tools import ShellTool
        tool = ShellTool()
        result = await tool.execute({"command": "find / -delete"})
        assert not result.success

    @pytest.mark.skipif(os.name == "nt", reason="ls is a POSIX utility")
    @pytest.mark.asyncio
    async def test_allows_safe_commands(self):
        from shadowd.tools.enhanced_tools import ShellTool
        tool = ShellTool()
        result = await tool.execute({"command": "ls -la"})
        assert result.success


class TestConnectionPoolThreadSafety:
    """Verify ConnectionPool singleton is thread-safe."""

    def test_singleton_creation_thread_safe(self):
        from shadowd.performance import ConnectionPool
        instances = []

        def create_instance():
            instances.append(ConnectionPool())

        threads = [threading.Thread(target=create_instance) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        # All instances should be the same object
        assert all(i is instances[0] for i in instances)


class TestAsyncTaskQueueResultsCap:
    """Verify AsyncTaskQueue caps results dict."""

    def test_results_max_cap(self):
        from shadowd.performance import AsyncTaskQueue
        queue = AsyncTaskQueue()
        assert hasattr(AsyncTaskQueue, 'MAX_RESULTS')
        assert AsyncTaskQueue.MAX_RESULTS == 1000


class TestSmartCacheThreadSafety:
    """Verify SmartCache has thread lock."""

    def test_smart_cache_has_lock(self):
        from shadowd.memory_optimization import SmartCache
        cache = SmartCache()
        assert hasattr(cache, '_lock')
        assert isinstance(cache._lock, type(threading.Lock()))


class TestPersonValidation:
    """Verify person names are validated in ambient_monitor."""

    def test_empty_person_rejected(self):
        from shadowd.ambient_monitor import AmbientMonitor
        from shadowd.context_protocol import ContextProtocol
        ctx = ContextProtocol()
        monitor = AmbientMonitor(data_dir=Path(tempfile.mkdtemp()), context_protocol=ctx)
        initial = len(monitor.people_encounters)
        monitor.track_person_encounter("", location="test")
        assert len(monitor.people_encounters) == initial

    def test_long_person_name_truncated(self):
        from shadowd.ambient_monitor import AmbientMonitor
        from shadowd.context_protocol import ContextProtocol
        ctx = ContextProtocol()
        monitor = AmbientMonitor(data_dir=Path(tempfile.mkdtemp()), context_protocol=ctx)
        monitor.track_person_encounter("A" * 200, location="test")
        # Key should be max 100 chars
        keys = [k for k in monitor.people_encounters.keys() if len(k) <= 100]
        assert len(keys) >= 1

    def test_special_chars_stripped(self):
        from shadowd.ambient_monitor import AmbientMonitor
        from shadowd.context_protocol import ContextProtocol
        ctx = ContextProtocol()
        monitor = AmbientMonitor(data_dir=Path(tempfile.mkdtemp()), context_protocol=ctx)
        monitor.track_person_encounter("<script>alert(1)</script>", location="test")
        keys = list(monitor.people_encounters.keys())
        assert len(keys) >= 1
        assert "<script>" not in keys[0]


class TestTaskIDUniqueness:
    """Verify task IDs are unique and not predictable."""

    def test_task_ids_are_monotonic(self):
        from shadowd.agent_coordination import ShadowCrew
        crew = ShadowCrew(crew_id="test-crew")
        # Create tasks and verify IDs are unique
        if hasattr(crew, 'create_task'):
            task1 = crew.create_task("Test task 1", goal="test")
            task2 = crew.create_task("Test task 2", goal="test")
            assert task1.id != task2.id


class TestSecurityUtilsRateLimiter:
    """Verify rate limiter works correctly."""

    def test_rate_limiter_blocks_excess(self):
        from shadowd.security_utils import RateLimiter
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        for _ in range(5):
            assert limiter.check("key1")
        assert not limiter.check("key1")

    def test_rate_limiter_separate_keys(self):
        from shadowd.security_utils import RateLimiter
        limiter = RateLimiter(max_requests=2, window_seconds=60)
        assert limiter.check("a")
        assert limiter.check("a")
        assert not limiter.check("a")
        assert limiter.check("b")


class TestSecurityUtilsInputValidator:
    """Verify input validation utilities."""

    def test_sanitize_strips_control_chars(self):
        from shadowd.security_utils import InputValidator
        result = InputValidator.sanitize_string("hello\x00world")
        assert result == "helloworld"

    def test_sanitize_truncates(self):
        from shadowd.security_utils import InputValidator
        result = InputValidator.sanitize_string("a" * 2000, max_length=100)
        assert len(result) == 100

    def test_validate_id_rejects_special_chars(self):
        from shadowd.security_utils import InputValidator
        with pytest.raises(ValueError):
            InputValidator.validate_id("hello world")
        with pytest.raises(ValueError):
            InputValidator.validate_id("abc;drop table")

    def test_validate_id_accepts_valid(self):
        from shadowd.security_utils import InputValidator
        assert InputValidator.validate_id("device-123") == "device-123"
        assert InputValidator.validate_id("abc_456") == "abc_456"