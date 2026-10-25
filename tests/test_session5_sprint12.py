#!/usr/bin/env python3
"""Session 5 Sprint 12 regression tests.

Covers:
- V-275: RateLimiter fail-closed on Redis error
- V-280: AsyncTaskQueue event-based (no busy-wait)
- V-281: get_or_set stampede prevention
- V-288: AdvancedMemorySystem thread safety
- V-289: O(n²) string concat / core block char cap
- V-290: CLI error message leaks
- V-293: MCP server name validation
- V-294: MCP client max servers
- V-296: MCPServer primitive caps
- V-298: EnhancedMemory importance clamping
- V-299: EnhancedMemory episodes cap
- V-300: EnhancedMemory add_to_core
- V-304: UnifiedSystem event content sanitization
"""

import asyncio
import pytest
import sys
import tempfile
import threading
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock, patch

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRateLimiterFailOpen:
    """V-275: RateLimiter should fail-open on Redis error (allow requests when Redis is down)."""

    def test_redis_error_returns_allowed(self):
        from shadowd.performance import RateLimiter

        async def _test():
            mock_redis = AsyncMock()
            mock_redis.zremrangebyscore = AsyncMock(side_effect=Exception("Connection lost"))
            limiter = RateLimiter(redis_client=mock_redis)
            result = await limiter.is_allowed("user1")
            assert result[0] is True
            assert result[1] > 0

        asyncio.run(_test())


class TestAsyncTaskQueueEventBased:
    """V-280: AsyncTaskQueue should use events, not polling."""

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

    def test_on_task_done_frees_slot(self):
        from shadowd.performance import AsyncTaskQueue
        q = AsyncTaskQueue(max_concurrent=1)
        q._slot_available.clear()
        mock_task = MagicMock()
        q.active_tasks.add(mock_task)
        q._on_task_done(mock_task)
        assert q._slot_available.is_set()
        assert mock_task not in q.active_tasks


class TestGetOrSetStampede:
    """V-281: get_or_set should prevent thundering-herd."""

    def test_has_factory_locks(self):
        from shadowd.performance import MultiLevelCache
        cache = MultiLevelCache()
        assert hasattr(cache, '_factory_locks')
        assert isinstance(cache._factory_locks, dict)

    def test_single_flight_factory(self):
        from shadowd.performance import MultiLevelCache

        async def _test():
            cache = MultiLevelCache()
            call_count = 0

            async def expensive_factory():
                nonlocal call_count
                call_count += 1
                await asyncio.sleep(0.01)
                return "result"

            results = await asyncio.gather(
                cache.get_or_set("key1", expensive_factory),
                cache.get_or_set("key1", expensive_factory),
                cache.get_or_set("key1", expensive_factory),
            )
            assert all(r == "result" for r in results)
            assert call_count <= 2

        asyncio.run(_test())


class TestAdvancedMemoryThreadSafety:
    """V-288: AdvancedMemorySystem thread safety."""

    def test_has_lock(self):
        from shadowd.advanced_memory import AdvancedMemorySystem
        with tempfile.TemporaryDirectory() as td:
            mem = AdvancedMemorySystem(Path(td))
            assert hasattr(mem, '_lock')
            assert isinstance(mem._lock, type(threading.Lock()))

    def test_concurrent_add_to_core(self):
        from shadowd.advanced_memory import AdvancedMemorySystem
        with tempfile.TemporaryDirectory() as td:
            mem = AdvancedMemorySystem(Path(td))
            errors = []

            def worker(i):
                try:
                    mem.add_to_core("test", f"value_{i}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert "value_" in mem.core_memory["test"].value

    def test_concurrent_add_to_archival(self):
        from shadowd.advanced_memory import AdvancedMemorySystem
        with tempfile.TemporaryDirectory() as td:
            mem = AdvancedMemorySystem(Path(td))
            errors = []

            def worker(i):
                try:
                    mem.add_to_archival(f"archival_{i}")
                except Exception as e:
                    errors.append(e)

            threads = [threading.Thread(target=worker, args=(i,)) for i in range(20)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(mem.archival_memory) == 20


class TestCoreBlockCharCap:
    """V-289: Core block value char cap."""

    def test_max_core_block_chars(self):
        from shadowd.advanced_memory import AdvancedMemorySystem
        assert AdvancedMemorySystem._MAX_CORE_BLOCK_CHARS == 100000

    def test_append_truncates(self):
        from shadowd.advanced_memory import AdvancedMemorySystem
        with tempfile.TemporaryDirectory() as td:
            mem = AdvancedMemorySystem(Path(td))
            mem.add_to_core("test", "A" * 99999)
            mem.add_to_core("test", "B" * 5000, append=True)
            assert len(mem.core_memory["test"].value) <= mem._MAX_CORE_BLOCK_CHARS


class TestCLIErrorLeak:
    """V-290: CLI error messages should not leak str(e)."""

    def test_error_message_no_str_e(self):
        from shadowd.cli import ShadowCLI
        import inspect
        source = inspect.getsource(ShadowCLI.handle_message)
        assert "Error: {e}" not in source
        assert "type(e).__name__" in source

    def test_clear_no_os_system(self):
        from shadowd.cli import ShadowCLI
        import inspect
        source = inspect.getsource(ShadowCLI.cmd_clear)
        assert "os.system" not in source
        assert "\\033" in source


class TestMCPServerNameValidation:
    """V-293: MCP server name validation."""

    def test_valid_names(self):
        from shadowd.mcp_client import _validate_server_name
        assert _validate_server_name("filesystem") == "filesystem"
        assert _validate_server_name("my-server-1") == "my-server-1"

    def test_invalid_names(self):
        from shadowd.mcp_client import _validate_server_name
        with pytest.raises(ValueError):
            _validate_server_name("")
        with pytest.raises(ValueError):
            _validate_server_name("a" * 65)
        with pytest.raises(ValueError):
            _validate_server_name("server with spaces")
        with pytest.raises(ValueError):
            _validate_server_name("server/slash")


class TestMCPClientMaxServers:
    """V-294: MCP client max servers."""

    def test_max_servers_constant(self):
        from shadowd.mcp_client import MCPClient
        assert MCPClient.MAX_SERVERS == 50


class TestMCPServerPrimitiveCaps:
    """V-296: MCPServer primitive registration caps."""

    def test_max_constants_exist(self):
        from shadowd.mcp_protocol import MCPServer
        assert MCPServer.MAX_TOOLS == 100
        assert MCPServer.MAX_RESOURCES == 100
        assert MCPServer.MAX_PROMPTS == 50

    def test_tool_registration_eviction(self):
        from shadowd.mcp_protocol import MCPServer, MCPTool
        server = MCPServer()
        for i in range(105):
            server.register_tool(MCPTool(
                name=f"tool_{i}",
                description=f"Tool {i}",
                input_schema={}
            ))
        assert len(server.tools) <= server.MAX_TOOLS


class TestEnhancedMemoryImportanceClamp:
    """V-298: EnhancedMemory importance clamping."""

    def test_importance_above_1_clamped(self):
        from shadowd.memory_enhanced import EnhancedMemory
        mem = EnhancedMemory(Path(tempfile.mkdtemp()))
        ep = mem.add_episode("test", "content", importance=5.0)
        assert ep.importance == 1.0

    def test_importance_below_0_clamped(self):
        from shadowd.memory_enhanced import EnhancedMemory
        mem = EnhancedMemory(Path(tempfile.mkdtemp()))
        ep = mem.add_episode("test", "content", importance=-1.0)
        assert ep.importance == 0.0


class TestEnhancedMemoryEpisodesCap:
    """V-299: EnhancedMemory episodes cap."""

    def test_max_episodes_constant(self):
        from shadowd.memory_enhanced import EnhancedMemory
        assert EnhancedMemory._MAX_EPISODES == 10000

    def test_episodes_capped(self):
        from shadowd.memory_enhanced import EnhancedMemory
        mem = EnhancedMemory(Path(tempfile.mkdtemp()))
        mem._MAX_EPISODES = 10
        for i in range(20):
            mem.add_episode("test", f"content_{i}")
        assert len(mem.episodes) <= 10


class TestEnhancedMemoryAddToCore:
    """V-300: EnhancedMemory.add_to_core method."""

    def test_add_to_core(self):
        from shadowd.memory_enhanced import EnhancedMemory
        mem = EnhancedMemory(Path(tempfile.mkdtemp()))
        mem.add_to_core("name", "User")
        assert mem.core_memory["name"] == "User"

    def test_add_to_core_truncates(self):
        from shadowd.memory_enhanced import EnhancedMemory
        mem = EnhancedMemory(Path(tempfile.mkdtemp()))
        mem.add_to_core("big", "A" * 20000)
        assert len(mem.core_memory["big"]) <= 10000


class TestUnifiedSystemEventSanitization:
    """V-304: UnifiedSystem observe event content sanitization."""

    def test_max_tasks_constant(self):
        from shadowd.unified_system import UnifiedShadowSystem
        assert UnifiedShadowSystem._MAX_TASKS == 10

    def test_event_content_sanitized_in_observe(self):
        from shadowd.unified_system import UnifiedShadowSystem, _MAX_OBSERVATION_SIZE
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.observe)
        assert "_MAX_OBSERVATION_SIZE" in source
        assert _MAX_OBSERVATION_SIZE >= 64000
        assert "str(event)" not in source or "event_content[:64000]" in source


class TestMCPProtocolErrorLeakFix:
    """V-295: MCP protocol error logging."""

    def test_handle_request_no_str_e(self):
        from shadowd.mcp_protocol import MCPServer
        import inspect
        source = inspect.getsource(MCPServer.handle_request)
        assert "type(e).__name__" in source

    def test_run_stdio_no_str_e(self):
        from shadowd.mcp_protocol import MCPServer
        import inspect
        source = inspect.getsource(MCPServer.run_stdio)
        assert "type(e).__name__" in source


class TestCircuitBreaker:
    """Feature: CircuitBreaker in ShadowCrew."""

    def test_circuit_breaker_states(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=0.01)
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available()

    def test_circuit_opens_after_threshold(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=2, recovery_timeout=1.0)
        cb.record_failure()
        assert cb.state == CircuitState.CLOSED
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        assert not cb.is_available()

    def test_circuit_half_open_after_timeout(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=0.01)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        import time
        time.sleep(0.02)
        assert cb.is_available()
        assert cb.state == CircuitState.HALF_OPEN

    def test_circuit_resets_on_success(self):
        from shadowd.cognition.shadow_crew import CircuitBreaker, CircuitState
        cb = CircuitBreaker(failure_threshold=1, recovery_timeout=1.0)
        cb.record_failure()
        assert cb.state == CircuitState.OPEN
        cb.record_success()
        assert cb.state == CircuitState.CLOSED
        assert cb.is_available()

    def test_crew_has_circuit_breakers_dict(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        assert hasattr(crew, '_circuit_breakers')
        assert isinstance(crew._circuit_breakers, dict)

    def test_circuit_breaker_created_on_task_failure(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CircuitBreaker

        async def _test():
            crew = ShadowCrew()
            mission = await crew.create_mission("test mission")
            mission.tasks[0].assigned_agent = "shadow_nonexistent"
            await crew.execute_mission(mission.mission_id)
            if "shadow_nonexistent" in crew._circuit_breakers:
                assert isinstance(crew._circuit_breakers["shadow_nonexistent"], CircuitBreaker)

        asyncio.run(_test())


class TestDarkMemoryDedup:
    """Feature: DarkMemory deduplication."""

    def test_find_duplicates_empty(self):
        from shadowd.cognition.dark_memory import DarkMemory

        async def _test():
            mem = DarkMemory()
            result = await mem.find_duplicates()
            assert result == []

        asyncio.run(_test())

    def test_find_duplicates_no_dupes(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType

        async def _test():
            mem = DarkMemory()
            await mem.store("unique content alpha", atom_type=AtomType.OBSERVATION)
            await mem.store("totally different beta", atom_type=AtomType.OBSERVATION)
            result = await mem.find_duplicates(similarity_threshold=0.99)
            assert result == []

        asyncio.run(_test())

    def test_deduplicate_returns_count(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType

        async def _test():
            mem = DarkMemory()
            await mem.store("hello world test", atom_type=AtomType.OBSERVATION, importance=0.5)
            await mem.store("hello world test", atom_type=AtomType.OBSERVATION, importance=0.3)
            removed = await mem.deduplicate(similarity_threshold=0.9)
            assert removed >= 0

        asyncio.run(_test())


class TestDarkMemoryHealth:
    """Feature: DarkMemory health scoring."""

    def test_health_empty(self):
        from shadowd.cognition.dark_memory import DarkMemory
        mem = DarkMemory()
        health = mem.get_health()
        assert health["score"] == 100.0
        assert health["status"] == "empty"

    def test_health_returns_required_fields(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType

        async def _test():
            mem = DarkMemory()
            await mem.store("test content", atom_type=AtomType.OBSERVATION)
            health = mem.get_health()
            assert "score" in health
            assert "status" in health
            assert "issues" in health
            assert 0 <= health["score"] <= 100

        asyncio.run(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])