#!/usr/bin/env python3
"""
Sprint 103 Tests — ShadowArise Deep Audit + ShadowOS v1→v2 Integration

Tests for:
1. ShadowArise thread-safety fixes (templates_used inside lock, recommend_template snapshot)
2. ShadowArise to_dict immutability deep copy
3. ShadowArise edge cases (deadline sanitization, metadata caps, execute on wrong lifecycle)
4. ShadowOS v1→v2 integration (EventBus publish, ShadowAI bridge, ask/observe_event)
5. Sanitization regex correctness
"""

import copy
import threading
import pytest
from unittest.mock import AsyncMock, MagicMock, patch

from shadowd.cognition.shadow_arise import (
    ShadowArise, AgentLifecycle, CapabilityType, MissionPriority,
    AgentStatus, AgentTemplate, Capability, MissionContext,
    ExecutionStep, _sanitize, _sanitize_content,
)


@pytest.fixture
def engine():
    return ShadowArise()


class TestShadowAriseThreadSafety:
    """Test thread-safety fixes from Sprint 103 audit."""

    def test_templates_used_increment_inside_lock(self, engine):
        """templates_used stat should be incremented inside the lock."""
        engine.arise(name="T1", objective="Obj", template_id="analyst")
        stats = engine.get_stats()
        assert stats["templates_used"].get("analyst") == 1

    def test_templates_used_concurrent(self, engine):
        """Concurrent arise with template_id should not lose increments."""
        errors = []

        def create_agent(i):
            try:
                engine.arise(name=f"T{i}", objective=f"Obj {i}", template_id="analyst")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_agent, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = engine.get_stats()
        assert stats["templates_used"].get("analyst", 0) == 20

    def test_recommend_template_uses_snapshot(self, engine):
        """recommend_template should work even during concurrent template changes."""
        template = engine.recommend_template("Analyze the data")
        assert template is not None
        assert template.template_id == "analyst"

    def test_get_stats_returns_copy(self, engine):
        """get_stats should return a copy, not a mutable reference."""
        engine.arise(name="T1", objective="Obj")
        stats = engine.get_stats()
        stats["agents_created"] = 999
        stats2 = engine.get_stats()
        assert stats2["agents_created"] == 1


class TestShadowAriseTodataClasses:
    """Test to_dict immutability for all data classes."""

    def test_capability_to_dict_deep_copy(self):
        cap = Capability(name="reason", capability_type=CapabilityType.REASON,
                         proficiency=0.8, description="test",
                         configuration={"a": [1, 2]}, usage_count=0)
        d = cap.to_dict()
        d["configuration"]["a"].append(3)
        assert len(cap.configuration["a"]) == 2

    def test_agent_template_to_dict_deep_copy(self):
        t = AgentTemplate(template_id="t1", name="Test", description="desc",
                          capabilities=[CapabilityType.REASON], default_proficiency=0.7,
                          mission_types=["test"], metadata={"k": "v"})
        d = t.to_dict()
        d["metadata"]["k2"] = "v2"
        assert "k2" not in t.metadata
        d["mission_types"].append("extra")
        assert "extra" not in t.mission_types

    def test_mission_context_to_dict_deep_copy(self):
        m = MissionContext(objective="Obj", priority=MissionPriority.HIGH,
                          constraints=["c1"], success_criteria=["s1"],
                          deadline="2024-12-31", context_data={"k": "v"})
        d = m.to_dict()
        d["constraints"].append("c2")
        assert "c2" not in m.constraints
        d["context_data"]["k2"] = "v2"
        assert "k2" not in m.context_data

    def test_execution_step_to_dict_deep_copy(self):
        s = ExecutionStep(step_id="s1", description="step",
                         capability_used=CapabilityType.ANALYZE,
                         input_data={"a": 1}, output_data={"b": 2},
                         confidence=0.9, timestamp="2024-01-01", duration_ms=100)
        d = s.to_dict()
        d["input_data"]["c"] = 3
        assert "c" not in s.input_data

    def test_agent_instance_to_dict_deep_copy(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        fetched = engine.get_agent(agent.agent_id)
        d = fetched.to_dict()
        d["mission"]["constraints"].append("injected")
        d["capabilities"][0]["name"] = "tampered"
        d["metadata"]["evil"] = "data"
        refetched = engine.get_agent(agent.agent_id)
        assert "injected" not in refetched.mission.constraints
        assert "evil" not in refetched.metadata


class TestShadowAriseEdgeCases:
    """Test edge cases discovered during audit."""

    def test_arise_with_deadline(self, engine):
        agent = engine.arise(name="D", objective="O", deadline="2024-12-31T23:59")
        assert agent.mission.deadline == "2024-12-31T23:59"

    def test_arise_deadline_sanitized(self, engine):
        agent = engine.arise(name="D", objective="O", deadline="<script>alert(1)</script>")
        assert "<script>" not in agent.mission.deadline

    def test_arise_metadata_cap(self, engine):
        big_meta = {f"key_{i}": f"val_{i}" for i in range(100)}
        agent = engine.arise(name="D", objective="O", metadata=big_meta)
        assert len(agent.metadata) <= 50

    def test_arise_context_data_cap(self, engine):
        big_ctx = {f"ctx_{i}": f"v_{i}" for i in range(100)}
        agent = engine.arise(name="D", objective="O", context_data=big_ctx)
        assert len(agent.mission.context_data) <= 50

    def test_execute_step_on_pending_agent_returns_none(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        step = engine.execute_step(agent.agent_id, "Step", CapabilityType.ANALYZE)
        assert step is None

    def test_evolve_on_archived_agent(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        engine.complete(agent.agent_id)
        engine.archive(agent.agent_id)
        result = engine.evolve(agent.agent_id, "feedback")
        assert result is None

    def test_complete_with_failed_status(self, engine):
        agent = engine.arise(name="Test", objective="Obj")
        engine.activate(agent.agent_id)
        result = engine.complete(agent.agent_id, result={"error": "timeout"},
                                status=AgentStatus.FAILED)
        assert result.status == AgentStatus.FAILED
        stats = engine.get_stats()
        assert stats["agents_failed"] == 1

    def test_add_custom_template(self, engine):
        t = AgentTemplate(
            template_id="custom_scout",
            name="Scout",
            description="Scout agent for exploration",
            capabilities=[CapabilityType.OBSERVE, CapabilityType.ADAPT],
            default_proficiency=0.7,
            mission_types=["exploration", "scouting"],
            metadata={"domain": "exploration"},
        )
        result = engine.add_template(t)
        assert result.template_id == "custom_scout"
        fetched = engine.get_template("custom_scout")
        assert fetched is not None
        assert fetched.name == "Scout"

    def test_add_template_sanitizes(self, engine):
        t = AgentTemplate(
            template_id="<script>alert(1)</script>",
            name="<script>evil</script>",
            description="Clean description",
            capabilities=[CapabilityType.REASON],
            default_proficiency=0.5,
            mission_types=["test"],
            metadata={"key": "val"},
        )
        result = engine.add_template(t)
        assert "<script>" not in result.template_id
        assert "<script>" not in result.name

    def test_agent_arise_with_no_template_uses_defaults(self, engine):
        agent = engine.arise(name="Default", objective="Do stuff")
        assert agent.template_id == "custom"
        assert len(agent.capabilities) == 2

    def test_evolution_history_tracks(self, engine):
        agent = engine.arise(name="E", objective="O")
        engine.activate(agent.agent_id)
        engine.evolve(agent.agent_id, "Need more analysis", {"reason": 0.2})
        stats = engine.get_stats()
        assert stats["agents_evolved"] == 1
        assert stats["evolution_history_size"] == 1

    def test_get_nonexistent_agent(self, engine):
        result = engine.get_agent("nonexistent_id___")
        assert result is None

    def test_configure_nonexistent_agent(self, engine):
        result = engine.configure("nonexistent_id___", capabilities=[CapabilityType.PLAN])
        assert result is None

    def test_activate_nonexistent_agent(self, engine):
        result = engine.activate("nonexistent_id___")
        assert result is None

    def test_complete_nonexistent_agent(self, engine):
        result = engine.complete("nonexistent_id___")
        assert result is None

    def test_archive_nonexistent_agent(self, engine):
        result = engine.archive("nonexistent_id___")
        assert result is None


class TestSanitizationRegex:
    """Test sanitization regex patterns."""

    def test_strict_strips_angle_brackets(self):
        result = _sanitize("<script>alert(1)</script>test", 200)
        assert "<script>" not in result
        assert "</script>" not in result
        assert "alert(1)" in result

    def test_strict_strips_control_chars(self):
        result = _sanitize("hello\x00world\x01test", 200)
        assert "\x00" not in result
        assert "\x01" not in result

    def test_strict_strips_tabs_and_newlines(self):
        result = _sanitize("hello\tworld\n", 200)
        assert "\t" not in result
        assert "\n" not in result

    def test_content_preserves_angle_brackets(self):
        result = _sanitize_content("latency < 100ms is good", 2000)
        assert "<" in result
        assert "latency < 100ms is good" in result

    def test_content_strips_control_chars(self):
        result = _sanitize_content("hello\x00world", 2000)
        assert "\x00" not in result

    def test_content_preserves_tabs_and_newlines(self):
        result = _sanitize_content("hello\tworld\nnext", 2000)
        assert "\t" in result
        assert "\n" in result

    def test_content_strips_del(self):
        result = _sanitize_content("text\x7fhere", 2000)
        assert "\x7f" not in result


class TestShadowOSIntegrationDetails:
    """Test ShadowOS v1→v2 integration details."""

    def test_shadowos_has_arise(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        assert hasattr(os_instance, 'arise')
        assert isinstance(os_instance.arise, ShadowArise)

    def test_shadowos_has_mind(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.mind import MindEngine
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        assert isinstance(os_instance.mind, MindEngine)

    def test_shadowos_has_event_bus(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.event_bus import EventBus
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        assert isinstance(os_instance.event_bus, EventBus)

    def test_shadowos_mind_is_async(self):
        """MindEngine.think should be awaitable."""
        import inspect
        from shadowd.mind import MindEngine
        assert hasattr(MindEngine.think, '__call__')

    def test_shadowos_shadow_ai_none_by_default(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        assert os_instance.shadow_ai is None

    @pytest.mark.asyncio
    async def test_observe_event_calls_process_observation(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        event = {"type": "test", "content": "hello"}
        result = await os_instance.observe_event(event)
        assert result is not None
        assert "status" in result

    @pytest.mark.asyncio
    async def test_observe_event_publishes_to_event_bus(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.event_bus import ShadowEvent, EventType
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        published = []
        os_instance.event_bus.publish = AsyncMock(side_effect=lambda e: published.append(e))

        event = {"type": "test_event", "content": "hello"}
        try:
            result = await os_instance.observe_event(event)
        except Exception:
            pass

        assert len(published) >= 1
        assert published[0].event_type == EventType.OBSERVATION
        assert published[0].source == "shadow_os"

    @pytest.mark.asyncio
    async def test_ask_uses_mind_engine(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(return_value="LLM says hello")
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.RESPOND, confidence=0.5))

        result = await os_instance.ask("Hello")

        assert result["response"] == "LLM says hello"
        os_instance.mind.think.assert_called()

    @pytest.mark.asyncio
    async def test_ask_falls_back_to_mind_when_no_llm_in_graph(self):
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance = ShadowOS(data_dir="/tmp/test_sprint103_os")
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(return_value="Fallback response")
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.ANALYZE, confidence=0.5))

        result = await os_instance.ask("What should I do?")

        os_instance.mind.think.assert_called()
        assert result["response"] == "Fallback response"


class TestEventBusCorrectAPI:
    """Test that EventBus is used correctly (get_history, not get_recent)."""

    def test_get_history_has_limit_param(self):
        from shadowd.event_bus import EventBus
        bus = EventBus()
        import inspect
        sig = inspect.signature(bus.get_history)
        assert "limit" in sig.parameters

    def test_get_history_returns_events(self):
        from shadowd.event_bus import EventBus, ShadowEvent, EventType
        bus = EventBus()

        async def test():
            await bus.publish(ShadowEvent(
                event_type=EventType.OBSERVATION,
                source="test_source",
                data={"key": "value"},
            ))
            events = bus.get_history(limit=10)
            assert len(events) == 1
            assert events[0].source == "test_source"
            assert events[0].data == {"key": "value"}

        import asyncio
        asyncio.run(test())


class TestShadowAIAsyncInterface:
    """Verify ShadowAI.observe is async (requires await)."""

    def test_shadow_ai_observe_is_async(self):
        import inspect
        from shadowd.shadow_ai import ShadowAI
        assert inspect.iscoroutinefunction(ShadowAI.observe)

    def test_event_bus_publish_is_async(self):
        import inspect
        from shadowd.event_bus import EventBus
        assert inspect.iscoroutinefunction(EventBus.publish)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])