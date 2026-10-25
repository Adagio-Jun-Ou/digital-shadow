#!/usr/bin/env python3
"""
Sprint 101 Tests — ShadowAI v1 into ShadowOS v2 Unification

Tests for:
1. ShadowOS.__init__ with shikigami_registry parameter
2. MindEngine integration in _handle_reason
3. ShadowAI bridge in _handle_respond
4. ask() method — unified v1+v2 query interface
5. observe_event() method — unified v1+v2 observation pipeline
6. stream_ask() method — SSE streaming stub
7. Bug fixes: await on async calls, record_observation→observe, event_bus API
"""

import asyncio
import pytest
from unittest.mock import AsyncMock, MagicMock, patch
from datetime import datetime, timezone


@pytest.fixture
def os_instance(tmp_path):
    """Create a ShadowOS instance for testing."""
    from shadowd.cognition.shadow_os import ShadowOS
    return ShadowOS(data_dir=str(tmp_path / "shadow_os"))


@pytest.fixture
def os_with_mocked_mind(os_instance):
    """ShadowOS with mocked MindEngine."""
    os_instance.mind = AsyncMock()
    os_instance.mind.think = AsyncMock(return_value="Mock LLM response")
    return os_instance


@pytest.fixture
def os_with_mocked_graph(os_instance):
    """ShadowOS with mocked MindEngine and ShadowGraph."""
    from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
    os_instance.mind = AsyncMock()
    os_instance.mind.think = AsyncMock(return_value="Mock LLM response")
    os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
        current_query="", mode=CognitiveMode.RESPOND, confidence=0.75
    ))
    return os_instance


class TestShikigamiRegistry:
    """Test ShadowOS init with and without shikigami_registry."""

    def test_init_without_registry(self, tmp_path):
        """ShadowOS should init without shikigami_registry (v2-only mode)."""
        from shadowd.cognition.shadow_os import ShadowOS
        os_instance = ShadowOS(data_dir=str(tmp_path / "os1"))
        assert os_instance.shadow_ai is None
        assert os_instance._shikigami_registry is None
        assert os_instance.mind is not None
        assert os_instance.event_bus is not None

    def test_init_with_registry(self, tmp_path):
        """ShadowOS should create ShadowAI when shikigami_registry provided."""
        from shadowd.cognition.shadow_os import ShadowOS
        mock_registry = MagicMock()
        with patch('shadowd.cognition.shadow_os.ShadowAI') as MockShadowAI:
            MockShadowAI.return_value = MagicMock()
            os_instance = ShadowOS(data_dir=str(tmp_path / "os2"), shikigami_registry=mock_registry)
            assert os_instance.shadow_ai is not None
            assert os_instance._shikigami_registry == mock_registry
            MockShadowAI.assert_called_once_with(mock_registry, os_instance.mind, data_dir=os_instance.data_dir)

    def test_mind_engine_created(self, tmp_path):
        """ShadowOS should always create a MindEngine."""
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.mind import MindEngine
        os_instance = ShadowOS(data_dir=str(tmp_path / "os3"))
        assert isinstance(os_instance.mind, MindEngine)

    def test_event_bus_created(self, tmp_path):
        """ShadowOS should always have an event_bus."""
        from shadowd.cognition.shadow_os import ShadowOS
        from shadowd.event_bus import EventBus
        os_instance = ShadowOS(data_dir=str(tmp_path / "os4"))
        assert isinstance(os_instance.event_bus, EventBus)


class TestHandleReasonMindEngine:
    """Test MindEngine integration in _handle_reason."""

    @pytest.mark.asyncio
    async def test_stores_llm_response(self, os_instance):
        """_handle_reason should call MindEngine.think and store llm_response."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(return_value="Focus on coding today.")

        state = CognitiveState(current_query="What should I do?", mode=CognitiveMode.ANALYZE, confidence=0.5)
        result = await os_instance._handle_reason(state)

        os_instance.mind.think.assert_called_once()
        assert result.context.get("llm_response") == "Focus on coding today."

    @pytest.mark.asyncio
    async def test_mind_failure_graceful(self, os_instance):
        """_handle_reason should handle MindEngine failure gracefully."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(side_effect=ConnectionError("API timeout"))

        state = CognitiveState(current_query="test", mode=CognitiveMode.ANALYZE, confidence=0.5)
        result = await os_instance._handle_reason(state)

        os_instance.mind.think.assert_called_once()
        assert "llm_response" not in result.context

    @pytest.mark.asyncio
    async def test_uses_event_history(self, os_instance):
        """_handle_reason should use event_bus history for LLM context."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        from shadowd.event_bus import ShadowEvent, EventType
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(return_value="Response")

        await os_instance.event_bus.publish(ShadowEvent(
            event_type=EventType.OBSERVATION,
            source="test",
            data={"content": "test observation"},
        ))

        state = CognitiveState(current_query="What happened?", mode=CognitiveMode.REASON, confidence=0.5)
        result = await os_instance._handle_reason(state)

        call_args = os_instance.mind.think.call_args
        observations = call_args[1].get("observations", None)
        assert observations is not None
        assert isinstance(observations, list)
        assert len(observations) > 0
        assert observations[0]["type"] == "observation"


class TestHandleRespondBridge:
    """Test ShadowAI bridge in _handle_respond."""

    @pytest.mark.asyncio
    async def test_uses_llm_response(self, os_instance):
        """_handle_respond should use llm_response from context when available."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(current_query="test query", mode=CognitiveMode.RESPOND, confidence=0.8)
        state.context["llm_response"] = "LLM generated response"

        result = await os_instance._handle_respond(state)
        assert os_instance._stats.get("queries_handled", 0) >= 1

    @pytest.mark.asyncio
    async def test_without_llm_response(self, os_instance):
        """_handle_respond should work without llm_response in context."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(current_query="test query", mode=CognitiveMode.RESPOND, confidence=0.8)

        result = await os_instance._handle_respond(state)
        assert os_instance._stats.get("queries_handled", 0) >= 1

    @pytest.mark.asyncio
    async def test_shadow_ai_bridge(self, os_instance):
        """_handle_respond should call shadow_ai.observe when shadow_ai exists."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        mock_ai = AsyncMock()
        os_instance.shadow_ai = mock_ai

        state = CognitiveState(current_query="test query", mode=CognitiveMode.RESPOND, confidence=0.8)
        state.context["llm_response"] = "test response"

        result = await os_instance._handle_respond(state)

        mock_ai.observe.assert_called_once()
        call_data = mock_ai.observe.call_args[0][0]
        assert call_data["type"] == "response"
        assert call_data["content"] == "test query"
        assert call_data["response"] == "test response"

    @pytest.mark.asyncio
    async def test_shadow_ai_failure_graceful(self, os_instance):
        """_handle_respond should handle shadow_ai.observe failure gracefully."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        mock_ai = AsyncMock()
        mock_ai.observe = AsyncMock(side_effect=ConnectionError("Connection error"))
        os_instance.shadow_ai = mock_ai

        state = CognitiveState(current_query="test query", mode=CognitiveMode.RESPOND, confidence=0.8)

        result = await os_instance._handle_respond(state)
        assert result is not None


class TestAskMethod:
    """Test unified ask() method."""

    @pytest.mark.asyncio
    async def test_sanitizes_input(self, os_with_mocked_graph):
        """ask() should sanitize the question input."""
        result = await os_with_mocked_graph.ask("What is <script>alert('xss')</script> the weather?")
        assert "<script>" not in result["query"]

    @pytest.mark.asyncio
    async def test_returns_cognitive_mode(self, os_with_mocked_graph):
        """ask() should return the cognitive mode from graph processing."""
        result = await os_with_mocked_graph.ask("Should I refactor this code?")
        assert "cognitive_mode" in result
        assert "confidence" in result

    @pytest.mark.asyncio
    async def test_uses_graph_for_cognitive_context(self, os_with_mocked_graph):
        """ask() should route through ShadowGraph for cognitive processing."""
        result = await os_with_mocked_graph.ask("Analyze this data")
        os_with_mocked_graph.graph.process_event.assert_called_once()
        call_kwargs = os_with_mocked_graph.graph.process_event.call_args[1]
        assert call_kwargs["trigger"].value == "human_request"

    @pytest.mark.asyncio
    async def test_fallback_llm_when_no_graph_response(self, os_instance):
        """ask() should call MindEngine.think directly when graph doesn't provide LLM response."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(return_value="Direct LLM response")
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.RESPOND, confidence=0.5
        ))

        result = await os_instance.ask("Tell me something")

        os_instance.mind.think.assert_called()
        assert result["response"] == "Direct LLM response"

    @pytest.mark.asyncio
    async def test_records_personality_behavior(self, os_with_mocked_graph):
        """ask() should record behavior in PersonalityEngine."""
        os_with_mocked_graph.personality.record_behavior = MagicMock()
        await os_with_mocked_graph.ask("Short Q")
        os_with_mocked_graph.personality.record_behavior.assert_called_once()
        call_args = os_with_mocked_graph.personality.record_behavior.call_args
        assert call_args[0][0] == "question_asked"

    @pytest.mark.asyncio
    async def test_short_question_simple(self, os_with_mocked_graph):
        """ask() should classify short questions as simple complexity."""
        os_with_mocked_graph.personality.record_behavior = MagicMock()
        await os_with_mocked_graph.ask("Hi")
        call_args = os_with_mocked_graph.personality.record_behavior.call_args
        assert call_args[0][1]["complexity"] == "simple"

    @pytest.mark.asyncio
    async def test_long_question_moderate(self, os_with_mocked_graph):
        """ask() should classify long questions as moderate complexity."""
        os_with_mocked_graph.personality.record_behavior = MagicMock()
        await os_with_mocked_graph.ask("This is a moderately long question about the system architecture")
        call_args = os_with_mocked_graph.personality.record_behavior.call_args
        assert call_args[0][1]["complexity"] == "moderate"

    @pytest.mark.asyncio
    async def test_shadow_ai_bridge(self, os_with_mocked_graph):
        """ask() should bridge to ShadowAI when available."""
        mock_ai = AsyncMock()
        os_with_mocked_graph.shadow_ai = mock_ai

        result = await os_with_mocked_graph.ask("Hello")

        mock_ai.observe.assert_called_once()
        observe_data = mock_ai.observe.call_args[0][0]
        assert observe_data["type"] == "ask"

    @pytest.mark.asyncio
    async def test_shadow_ai_failure_graceful(self, os_with_mocked_graph):
        """ask() should handle ShadowAI.observe failure gracefully."""
        mock_ai = AsyncMock()
        mock_ai.observe = AsyncMock(side_effect=ConnectionError("API down"))
        os_with_mocked_graph.shadow_ai = mock_ai

        result = await os_with_mocked_graph.ask("Hello")
        assert result is not None
        assert "response" in result

    @pytest.mark.asyncio
    async def test_no_shadow_ai(self, os_with_mocked_graph):
        """ask() should work without ShadowAI (v2-only mode)."""
        result = await os_with_mocked_graph.ask("Hello")
        assert result is not None
        assert "response" in result
        assert os_with_mocked_graph.shadow_ai is None

    @pytest.mark.asyncio
    async def test_stores_observation_in_memory(self, os_with_mocked_graph):
        """ask() should store the question in DarkMemory."""
        os_with_mocked_graph.memory.store = AsyncMock()
        await os_with_mocked_graph.ask("Remember this")
        os_with_mocked_graph.memory.store.assert_called_once()

    @pytest.mark.asyncio
    async def test_truncates_shadow_ai_bridge_content(self, os_with_mocked_graph):
        """ask() should truncate content to 500 chars when bridging to ShadowAI."""
        mock_ai = AsyncMock()
        os_with_mocked_graph.shadow_ai = mock_ai
        os_with_mocked_graph.mind.think = AsyncMock(return_value="A" * 1000)

        result = await os_with_mocked_graph.ask("Q" * 600)
        observe_data = mock_ai.observe.call_args[0][0]
        assert len(observe_data["content"]) <= 500
        assert len(observe_data["response"]) <= 500


class TestObserveEvent:
    """Test unified observe_event() method."""

    @pytest.mark.asyncio
    async def test_calls_process_observation(self, os_instance):
        """observe_event() should call process_observation() internally."""
        event = {"type": "test", "content": "hello"}
        result = await os_instance.observe_event(event)
        assert result is not None

    @pytest.mark.asyncio
    async def test_bridges_to_shadow_ai(self, os_instance):
        """observe_event() should bridge observation to ShadowAI when available."""
        mock_ai = AsyncMock()
        os_instance.shadow_ai = mock_ai

        event = {"type": "code_change", "content": "refactored module.py"}
        result = await os_instance.observe_event(event)

        mock_ai.observe.assert_called_once_with(event)

    @pytest.mark.asyncio
    async def test_publishes_to_event_bus(self, os_instance):
        """observe_event() should publish observation to EventBus."""
        from shadowd.event_bus import ShadowEvent, EventType

        event = {"type": "test_event", "content": "hello"}
        published = []
        original_publish = os_instance.event_bus.publish
        os_instance.event_bus.publish = AsyncMock(side_effect=lambda e: published.append(e))

        try:
            result = await os_instance.observe_event(event)
        except Exception:
            pass

        assert len(published) >= 1
        assert published[0].event_type == EventType.OBSERVATION
        assert published[0].source == "shadow_os"

    @pytest.mark.asyncio
    async def test_shadow_ai_failure_graceful(self, os_instance):
        """observe_event() should handle ShadowAI.observe failure gracefully."""
        mock_ai = AsyncMock()
        mock_ai.observe = AsyncMock(side_effect=ConnectionError("down"))
        os_instance.shadow_ai = mock_ai

        event = {"type": "test", "content": "hello"}
        result = await os_instance.observe_event(event)
        assert result is not None

    @pytest.mark.asyncio
    async def test_event_bus_failure_graceful(self, os_instance):
        """observe_event() should handle EventBus.publish failure gracefully."""
        os_instance.event_bus.publish = AsyncMock(side_effect=RuntimeError("Bus error"))

        event = {"type": "test", "content": "hello"}
        result = await os_instance.observe_event(event)
        assert result is not None

    @pytest.mark.asyncio
    async def test_no_shadow_ai(self, os_instance):
        """observe_event() should work without ShadowAI (v2-only mode)."""
        os_instance.shadow_ai = None
        event = {"type": "test", "content": "hello"}
        result = await os_instance.observe_event(event)
        assert result is not None

    @pytest.mark.asyncio
    async def test_returns_v2_result(self, os_instance):
        """observe_event() should return the result from process_observation."""
        event = {"type": "test", "content": "hello"}
        result = await os_instance.observe_event(event)
        assert "status" in result


class TestStreamAsk:
    """Test stream_ask() method."""

    @pytest.mark.asyncio
    async def test_yields_cognitive_state(self, os_instance):
        """stream_ask() should yield cognitive_state first."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.ANALYZE, confidence=0.7
        ))

        chunks = []
        async for chunk in os_instance.stream_ask("test question"):
            chunks.append(chunk)

        assert len(chunks) >= 1
        assert chunks[0]["type"] == "cognitive_state"
        assert chunks[0]["mode"] == "analyze"
        assert chunks[0]["confidence"] == 0.7

    @pytest.mark.asyncio
    async def test_yields_response(self, os_instance):
        """stream_ask() should yield a response chunk."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.RESPOND, confidence=0.5
        ))

        chunks = []
        async for chunk in os_instance.stream_ask("test"):
            chunks.append(chunk)

        assert len(chunks) >= 2
        assert chunks[1]["type"] == "response"

    @pytest.mark.asyncio
    async def test_sanitizes_input(self, os_instance):
        """stream_ask() should sanitize the question."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.RESPOND, confidence=0.5
        ))

        chunks = []
        async for chunk in os_instance.stream_ask("<script>alert(1)</script>test"):
            chunks.append(chunk)

        call_kwargs = os_instance.graph.process_event.call_args[1]
        assert "<script>" not in str(call_kwargs["event"].get("query", ""))


class TestEventBusIntegration:
    """Test EventBus integration fixes."""

    def test_get_history_exists(self):
        """EventBus should have get_history method."""
        from shadowd.event_bus import EventBus
        bus = EventBus()
        assert hasattr(bus, 'get_history')

    def test_get_recent_does_not_exist(self):
        """EventBus should NOT have get_recent method (bug fix verification)."""
        from shadowd.event_bus import EventBus
        bus = EventBus()
        assert not hasattr(bus, 'get_recent')

    @pytest.mark.asyncio
    async def test_get_history_returns_events(self):
        """EventBus.get_history should return recently published events."""
        from shadowd.event_bus import EventBus, ShadowEvent, EventType
        bus = EventBus()
        await bus.publish(ShadowEvent(
            event_type=EventType.OBSERVATION,
            source="test",
            data={"key": "value"},
        ))
        history = bus.get_history(limit=10)
        assert len(history) == 1
        assert history[0].source == "test"


class TestShadowAIObserveIsAsync:
    """Verify ShadowAI.observe is properly awaited."""

    def test_observe_is_async(self):
        """ShadowAI.observe should be an async method."""
        import inspect
        from shadowd.shadow_ai import ShadowAI
        assert inspect.iscoroutinefunction(ShadowAI.observe), \
            "ShadowAI.observe must be async for proper awaiting"

    def test_publish_is_async(self):
        """EventBus.publish should be an async method."""
        import inspect
        from shadowd.event_bus import EventBus
        assert inspect.iscoroutinefunction(EventBus.publish), \
            "EventBus.publish must be async for proper awaiting"


class TestUnifiedAPIBackwardCompatibility:
    """Test that v2-only mode still works without any v1 components."""

    @pytest.mark.asyncio
    async def test_process_observation_without_v1(self, os_instance):
        """process_observation should work without any v1 components."""
        event = {"type": "observation", "content": "test content"}
        result = await os_instance.process_observation(event)
        assert "status" in result

    @pytest.mark.asyncio
    async def test_ask_without_shadow_ai(self, os_instance):
        """ask() should work in v2-only mode (no ShadowAI)."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        os_instance.mind = AsyncMock()
        os_instance.mind.think = AsyncMock(return_value="V2 response only")
        os_instance.graph.process_event = AsyncMock(return_value=CognitiveState(
            current_query="", mode=CognitiveMode.RESPOND, confidence=0.5
        ))

        result = await os_instance.ask("Hello from v2")
        assert result is not None
        assert result["response"] == "V2 response only"
        assert os_instance.shadow_ai is None

    @pytest.mark.asyncio
    async def test_observe_event_without_shadow_ai(self, os_instance):
        """observe_event() should work in v2-only mode."""
        event = {"type": "test", "content": "v2 observation"}
        result = await os_instance.observe_event(event)
        assert result is not None
        assert os_instance.shadow_ai is None

    def test_get_status_structure(self, os_instance):
        """get_status() should return status dict with expected keys."""
        status = os_instance.get_status()
        assert isinstance(status, dict)
        assert "stats" in status

    def test_get_health_structure(self, os_instance):
        """get_health() should return health dict with modules."""
        health = os_instance.get_health()
        assert isinstance(health, dict)
        assert "modules" in health
        assert "status" in health


class TestInputSanitization:
    """Test input sanitization in ask/observe_event methods."""

    @pytest.mark.asyncio
    async def test_ask_strips_xss(self, os_with_mocked_graph):
        """ask() should strip XSS from question."""
        result = await os_with_mocked_graph.ask("<script>alert('xss')</script>hello")
        assert "<script>" not in result["query"]

    @pytest.mark.asyncio
    async def test_ask_strips_control_chars(self, os_with_mocked_graph):
        """ask() should strip control characters from question."""
        result = await os_with_mocked_graph.ask("hello\x00world\x01test")
        assert "\x00" not in result["query"]
        assert "\x01" not in result["query"]

    @pytest.mark.asyncio
    async def test_ask_truncates_long_input(self, os_with_mocked_graph):
        """ask() should truncate inputs exceeding MAX_QUERY_LENGTH."""
        result = await os_with_mocked_graph.ask("A" * 5000)
        assert len(result["query"]) <= 2000


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])