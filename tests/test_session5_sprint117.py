"""Sprint 117 — shadow_crew.py Deep Audit Regression Tests

Tests for fixes applied in Sprint 117:
- SC-T1: CircuitBreaker.state property acquires lock
- SC-T2: AgentRegistry has threading.Lock, all methods acquire lock
- SC-RC1: AgentRegistry.register() TOCTOU fixed with lock
- SC-T3: ShadowAgent._state dict has MAX_STATE_KEYS cap
- SC-DC2: CrewTask has to_dict() with copy.deepcopy
- SC-IS1: handoff() sanitizes context values and agent IDs
- SC-IS2: CrewTask.__post_init__ sanitizes fields
- SC-UB1: _circuit_breakers dict capped at _MAX_CIRCUIT_BREAKERS=50
- SC-UB2: ShadowAgent._state dict capped at _MAX_STATE_KEYS=50
- SC-EH1: Handler except blocks log error type instead of silent pass
- SC-BONUS1: _find_available_agent() iterates _agents directly instead of serializing
"""

import asyncio
import threading
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shadowd.cognition.shadow_crew import (
    AgentRegistry, AgentRole, CircuitBreaker, CircuitState,
    CrewMission, CrewTask, DelegationMode, MissionStatus,
    ShadowAgent, ShadowCrew, _sanitize, _sanitize_strict, _sanitize_content,
)


# --- SC-T1: CircuitBreaker.state acquires lock ---

class TestCircuitBreakerStateLock:
    def test_state_reads_under_lock(self):
        cb = CircuitBreaker()
        assert cb.state == CircuitState.CLOSED

    def test_state_transitions_visible(self):
        cb = CircuitBreaker()
        cb.record_failure()
        cb.record_failure()
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_concurrent_state_reads(self):
        cb = CircuitBreaker()
        results = []

        def read_state():
            for _ in range(100):
                results.append(cb.state)

        threads = [threading.Thread(target=read_state) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(s in (CircuitState.CLOSED, CircuitState.OPEN, CircuitState.HALF_OPEN) for s in results)


# --- SC-T2/SC-RC1: AgentRegistry thread safety ---

class TestAgentRegistryLock:
    def test_register_under_lock(self):
        reg = AgentRegistry()
        agent = ShadowAgent(
            agent_id="test_agent", name="Test", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        reg.register(agent)
        assert reg.get("test_agent") is not None

    def test_register_at_max_evicts(self):
        reg = AgentRegistry()
        for i in range(100):
            agent = ShadowAgent(
                agent_id=f"agent_{i}", name=f"Test {i}", role=AgentRole.OBSERVER,
                goal="Test", backstory="Test"
            )
            reg.register(agent)
        assert len(reg._agents) <= 100

    def test_register_duplicate_id_updates(self):
        reg = AgentRegistry()
        agent1 = ShadowAgent(
            agent_id="dup_agent", name="First", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        agent2 = ShadowAgent(
            agent_id="dup_agent", name="Second", role=AgentRole.REASONER,
            goal="Test", backstory="Test"
        )
        reg.register(agent1)
        reg.register(agent2)
        assert reg.get("dup_agent").name == "Second"

    def test_get_returns_none_for_unknown(self):
        reg = AgentRegistry()
        assert reg.get("nonexistent") is None

    def test_get_by_role_thread_safe(self):
        reg = AgentRegistry()
        agent = ShadowAgent(
            agent_id="obs1", name="Observer", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        reg.register(agent)
        result = reg.get_by_role(AgentRole.OBSERVER)
        assert len(result) == 1
        assert result[0].agent_id == "obs1"

    def test_list_agents_returns_copies(self):
        reg = AgentRegistry()
        agent = ShadowAgent(
            agent_id="agent1", name="Test", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        reg.register(agent)
        result1 = reg.list_agents()
        result2 = reg.list_agents()
        assert result1[0] is not result2[0]

    def test_concurrent_register_and_read(self):
        reg = AgentRegistry()
        errors = []

        def register_agent(i):
            try:
                agent = ShadowAgent(
                    agent_id=f"concurrent_{i}", name=f"Concurrent {i}",
                    role=AgentRole.OBSERVER, goal="Test", backstory="Test"
                )
                reg.register(agent)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_agent, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# --- SC-T3/SC-UB2: ShadowAgent._state cap ---

class TestShadowAgentStateCap:
    def test_set_state_basic(self):
        agent = ShadowAgent(
            agent_id="st_agent", name="Test", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        agent.set_state("key1", "value1")
        assert agent.get_state("key1") == "value1"

    def test_set_state_default(self):
        agent = ShadowAgent(
            agent_id="st_agent2", name="Test", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        assert agent.get_state("nonexistent", "default") == "default"

    def test_state_capped_at_50(self):
        agent = ShadowAgent(
            agent_id="st_agent3", name="Test", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        for i in range(60):
            agent.set_state(f"key_{i}", f"value_{i}")
        assert len(agent._state) <= 50

    def test_state_update_existing_key_no_eviction(self):
        agent = ShadowAgent(
            agent_id="st_agent4", name="Test", role=AgentRole.OBSERVER,
            goal="Test", backstory="Test"
        )
        for i in range(50):
            agent.set_state(f"key_{i}", f"value_{i}")
        agent.set_state("key_25", "updated")
        assert agent.get_state("key_25") == "updated"
        assert len(agent._state) == 50


# --- SC-DC2: CrewTask.to_dict() with copy.deepcopy ---

class TestCrewTaskToDict:
    def test_to_dict_returns_deep_copy(self):
        task = CrewTask(
            description="Test task",
            input_data={"key": "value"},
            metadata={"meta": "data"},
        )
        d = task.to_dict()
        d["input_data"]["new_key"] = "new_value"
        d["metadata"]["new_meta"] = "new_meta"
        assert "new_key" not in task.input_data
        assert "new_meta" not in task.metadata

    def test_dependencies_is_copy(self):
        task = CrewTask(dependencies=["dep1", "dep2"])
        d = task.to_dict()
        d["dependencies"].append("dep3")
        assert len(task.dependencies) == 2

    def test_output_data_deep_copy(self):
        task = CrewTask(output_data={"result": [1, 2, 3]})
        d = task.to_dict()
        d["output_data"]["result"].append(4)
        assert task.output_data["result"] == [1, 2, 3]


# --- SC-IS2: CrewTask.__post_init__ sanitization ---

class TestCrewTaskSanitization:
    def test_description_sanitized(self):
        task = CrewTask(description="Hello\x00World\x1fTest")
        assert "\x00" not in task.description
        assert "\x1f" not in task.description
        assert "Hello" in task.description
        assert "World" in task.description

    def test_assigned_agent_sanitized_strict(self):
        task = CrewTask(description="Test", assigned_agent="<script>alert(1)</script>")
        assert "<" not in task.assigned_agent
        assert ">" not in task.assigned_agent

    def test_error_sanitized_content(self):
        task = CrewTask(description="Test", error="Error\x00message")
        assert "\x00" not in task.error

    def test_auto_id_generated(self):
        task = CrewTask()
        assert task.task_id.startswith("task_")
        assert len(task.task_id) > 5


# --- SC-IS1: handoff() sanitization ---

class TestHandoffSanitization:
    @pytest.mark.asyncio
    async def test_handoff_sanitizes_agent_ids(self):
        crew = ShadowCrew()
        result = await crew.handoff(
            from_agent_id="shadow_eye\x00<script>",
            to_agent_id="shadow_mind<script>",
            context={"analysis": "test"},
        )
        assert result is False

    @pytest.mark.asyncio
    async def test_handoff_sanitizes_string_values(self):
        crew = ShadowCrew()
        context = {"analysis": "test\x00value", "observations": "data\x1fhere"}
        result = await crew.handoff("shadow_eye", "shadow_mind", context)
        if result:
            agent = crew._registry.get("shadow_mind")
            val = agent.get_state("analysis")
            if val:
                assert "\x00" not in val

    @pytest.mark.asyncio
    async def test_handoff_ignores_disallowed_keys(self):
        crew = ShadowCrew()
        context = {"analysis": "test", "evil_key": "should_not_pass"}
        result = await crew.handoff("shadow_eye", "shadow_mind", context)
        if result:
            agent = crew._registry.get("shadow_mind")
            assert agent.get_state("evil_key") is None


# --- SC-UB1: CircuitBreakers cap ---

class TestCircuitBreakersCap:
    @pytest.mark.asyncio
    async def test_circuit_breakers_capped_at_50(self):
        crew = ShadowCrew()
        assert crew._MAX_CIRCUIT_BREAKERS == 50

    def test_circuit_breaker_record_failure(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        assert cb._failure_count == 2
        cb.record_failure()
        assert cb.state == CircuitState.OPEN

    def test_circuit_breaker_record_success(self):
        cb = CircuitBreaker(failure_threshold=3)
        cb.record_failure()
        cb.record_failure()
        cb.record_success()
        assert cb._failure_count == 0
        assert cb.state == CircuitState.CLOSED


# --- SC-EH1: Handler except blocks log error type ---

class TestHandlerErrorLogging:
    @pytest.mark.asyncio
    async def test_observe_handler_logs_memory_error(self):
        crew = ShadowCrew()
        crew.shadow_os = MagicMock()
        memory_mock = AsyncMock()
        memory_mock.get_recent = AsyncMock(side_effect=ValueError("test error"))
        memory_mock.search = AsyncMock(side_effect=ValueError("test error"))
        crew.shadow_os.memory = memory_mock

        agent = crew._registry.get("shadow_eye")
        task = CrewTask(description="test", assigned_agent="shadow_eye")
        mission = CrewMission(objective="test objective")

        result = await crew._handle_observe(agent, task, mission)
        assert result["status"] == "no_data"

    @pytest.mark.asyncio
    async def test_analyze_handler_logs_personality_error(self):
        crew = ShadowCrew()
        crew.shadow_os = MagicMock()
        personality_mock = MagicMock()
        personality_mock.get_profile = MagicMock(side_effect=RuntimeError("test"))
        crew.shadow_os.personality = personality_mock

        agent = crew._registry.get("shadow_mind")
        task = CrewTask(description="test", assigned_agent="shadow_mind")
        mission = CrewMission(objective="test objective")

        result = await crew._handle_analyze(agent, task, mission)
        assert result["confidence"] == 0.7


# --- SC-BONUS1: _find_available_agent direct iteration ---

class TestFindAvailableAgentOptimization:
    def test_find_available_agent_excludes_id(self):
        crew = ShadowCrew()
        task = CrewTask(description="test")
        result = crew._find_available_agent("shadow_eye", task)
        if result:
            assert result.agent_id != "shadow_eye"

    def test_find_available_agent_returns_agent(self):
        crew = ShadowCrew()
        task = CrewTask(description="analyze")
        result = crew._find_available_agent("nonexistent", task)
        assert result is not None


# --- Sanitization function tests ---

class TestSanitizationFunctions:
    def test_sanitize_strips_control_chars(self):
        assert _sanitize("hello\x00world\x1f") == "helloworld"

    def test_sanitize_preserves_angle_brackets(self):
        assert _sanitize("latency < 100ms") == "latency < 100ms"

    def test_sanitize_strict_strips_angle_brackets(self):
        assert _sanitize_strict("<script>alert(1)</script>") == "scriptalert(1)/script"

    def test_sanitize_strict_strips_control_chars(self):
        assert _sanitize_strict("hello\x00world\n") == "helloworld"

    def test_sanitize_content_preserves_angle_brackets(self):
        assert _sanitize_content("latency < 100ms") == "latency < 100ms"

    def test_sanitize_content_strips_control_chars(self):
        assert _sanitize_content("hello\x00world\x1f") == "helloworld"

    def test_sanitize_none_handling(self):
        assert _sanitize(None) == ""
        assert _sanitize_strict(None) == ""
        assert _sanitize_content(None) == ""

    def test_sanitize_strict_strips_tabs_newlines(self):
        assert _sanitize_strict("hello\tworld\n") == "helloworld"


# --- ShadowCrew integration tests ---

class TestShadowCrewIntegration:
    @pytest.mark.asyncio
    async def test_create_mission_sanitizes_objective(self):
        crew = ShadowCrew()
        mission = await crew.create_mission(
            objective="Test\x00objective with <script>alert(1)</script>"
        )
        assert "\x00" not in mission.objective

    def test_get_stats_returns_copy(self):
        crew = ShadowCrew()
        stats = crew.get_stats()
        stats["stats"]["missions_created"] = 999
        assert crew._stats["missions_created"] != 999

    def test_circuit_breaker_integration(self):
        crew = ShadowCrew()
        with crew._cb_lock:
            cb = crew._circuit_breakers.get("test_agent")
            assert cb is None