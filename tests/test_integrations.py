"""Tests for Session 3 integrations: event_bus, stuck_detector, memory_tools,
ContextWindowBudget, config, and superhuman algorithm wiring."""

import asyncio
import json
import pytest
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch


# ══════════════════════════════════════════════════════════════════
# Event Bus Tests
# ══════════════════════════════════════════════════════════════════

class TestEventBus:
    def test_import(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent, get_event_bus
        bus = get_event_bus()
        assert bus is not None
        assert isinstance(bus, EventBus)

    def test_event_types(self):
        from shadowd.event_bus import EventType
        assert EventType.OBSERVATION.value == "observation"
        assert EventType.INSIGHT.value == "insight"
        assert EventType.ACTION.value == "action"
        assert EventType.STUCK.value == "stuck"
        assert EventType.RECOVERY.value == "recovery"

    def test_shadow_event_defaults(self):
        from shadowd.event_bus import ShadowEvent, EventType
        event = ShadowEvent(
            event_type=EventType.OBSERVATION,
            source="test",
            data={"key": "value"},
        )
        assert event.event_id.startswith("evt_")
        assert event.priority == 5
        assert event.timestamp

    @pytest.mark.asyncio
    async def test_publish_and_subscribe(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe(EventType.INSIGHT, handler)
        event = ShadowEvent(
            event_type=EventType.INSIGHT,
            source="test",
            data={"message": "hello"},
        )
        await bus.publish(event)
        assert len(received) == 1
        assert received[0].data["message"] == "hello"

    @pytest.mark.asyncio
    async def test_wildcard_subscriber(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent
        bus = EventBus()
        received = []

        async def handler(event):
            received.append(event)

        bus.subscribe_all(handler)
        await bus.publish(ShadowEvent(event_type=EventType.OBSERVATION, source="a", data={}))
        await bus.publish(ShadowEvent(event_type=EventType.INSIGHT, source="b", data={}))
        assert len(received) == 2

    @pytest.mark.asyncio
    async def test_stats(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent
        bus = EventBus()

        async def handler(event):
            pass

        bus.subscribe(EventType.OBSERVATION, handler)
        await bus.publish(ShadowEvent(event_type=EventType.OBSERVATION, source="test", data={}))
        stats = bus.get_stats()
        assert stats["events_published"] == 1
        assert stats["events_processed"] == 1

    @pytest.mark.asyncio
    async def test_handler_error_doesnt_crash(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent
        bus = EventBus()

        async def bad_handler(event):
            raise ValueError("test error")

        bus.subscribe(EventType.OBSERVATION, bad_handler)
        await bus.publish(ShadowEvent(event_type=EventType.OBSERVATION, source="test", data={}))
        stats = bus.get_stats()
        assert stats["errors"] == 1

    @pytest.mark.asyncio
    async def test_history(self):
        from shadowd.event_bus import EventBus, EventType, ShadowEvent
        bus = EventBus()
        for i in range(5):
            await bus.publish(ShadowEvent(event_type=EventType.OBSERVATION, source=f"src{i}", data={"i": i}))
        history = bus.get_history(limit=3)
        assert len(history) == 3
        filtered = bus.get_history(event_type=EventType.OBSERVATION)
        assert len(filtered) == 5

    def test_global_singleton(self):
        from shadowd.event_bus import get_event_bus
        bus1 = get_event_bus()
        bus2 = get_event_bus()
        assert bus1 is bus2


# ══════════════════════════════════════════════════════════════════
# StuckDetector Tests
# ══════════════════════════════════════════════════════════════════

class TestStuckDetector:
    def test_import(self):
        from shadowd.reasoning.stuck_detector import StuckDetector, LoopType, StuckAnalysis
        sd = StuckDetector()
        assert sd is not None

    def test_no_stuck_when_empty(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        result = sd.analyze()
        assert not result.is_stuck

    def test_record_action(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        entry = sd.record_action("read_file", tool_name="file_read")
        assert entry.action_type.value == "tool_call"
        assert entry.tool_name == "file_read"

    def test_record_thought(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        entry = sd.record_thought("I should check the logs")
        assert entry.action_type.value == "thought"

    def test_repeated_action_detection(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        for _ in range(4):
            sd.record_action("read_file", tool_name="file_read")
        result = sd.analyze()
        assert result.is_stuck
        assert result.loop_type.value == "repeated_action"

    def test_monologue_detection(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        for i in range(5):
            sd.record_thought(f"thinking step {i}")
        result = sd.analyze()
        assert result.is_stuck
        assert result.loop_type.value == "monologue"

    def test_context_window_loop_detection(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        for _ in range(3):
            sd.record_error("context window exceeded", "tried to send message")
        result = sd.analyze()
        assert result.is_stuck
        assert result.loop_type.value == "context_window_loop"

    def test_reset(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        sd.record_action("test")
        sd.record_action("test")
        sd.record_action("test")
        sd.reset()
        assert len(sd.history) == 0

    def test_stats(self):
        from shadowd.reasoning.stuck_detector import StuckDetector
        sd = StuckDetector()
        sd.record_action("test", tool_name="test_tool")
        stats = sd.get_stats()
        assert stats["total_entries"] == 1
        assert "test_tool" in stats["tool_counts"]


# ══════════════════════════════════════════════════════════════════
# MemoryTools Tests
# ══════════════════════════════════════════════════════════════════

class TestMemoryTools:
    def test_import(self):
        from shadowd.memory_tools import MemoryTools, ContextWindowBudget, MemoryEditResult, render_memory_xml

    def test_replace_success(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory, MemoryBlock
        memory = CoreMemory()
        memory.set_block("human", "John likes Python")
        tools = MemoryTools(memory)
        result = tools.memory_replace("human", "Python", "JavaScript")
        assert result.success
        assert "JavaScript" in memory.get_block("human").value

    def test_replace_not_found(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        memory.set_block("human", "Hello world")
        tools = MemoryTools(memory)
        result = tools.memory_replace("human", "nonexistent", "replacement")
        assert not result.success
        assert "not found" in result.error.lower()

    def test_replace_protected_block(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        result = MemoryTools(memory).memory_replace("persona", "I am", "You are")
        assert not result.success

    def test_insert(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        memory.set_block("human", "line1\nline2")
        tools = MemoryTools(memory)
        result = tools.memory_insert("human", "inserted", insert_line=1)
        assert result.success
        lines = memory.get_block("human").value.split("\n")
        assert "inserted" in lines

    def test_rethink(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        memory.set_block("human", "old content")
        tools = MemoryTools(memory)
        result = tools.memory_rethink("human", "brand new content")
        assert result.success
        assert memory.get_block("human").value == "brand new content"

    def test_finish_edits(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        tools = MemoryTools(memory)
        result = tools.memory_finish_edits()
        assert result["status"] == "editing_complete"
        assert "persona" in result["blocks"]

    def test_xml_rendering(self):
        from shadowd.memory_tools import render_memory_xml
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        memory.set_block("human", "test data")
        xml = render_memory_xml(memory.blocks)
        assert "<memory_blocks>" in xml
        assert "<human>" in xml
        assert "test data" in xml


class TestContextWindowBudget:
    def test_budget_overview(self):
        from shadowd.memory_tools import ContextWindowBudget
        budget = ContextWindowBudget(max_context_tokens=8192)
        overview = budget.get_budget_overview(
            system_prompt="You are a helpful assistant.",
            messages=[{"role": "user", "content": "Hello"}],
        )
        assert "total_tokens_used" in overview
        assert "utilization" in overview
        assert "sections" in overview

    def test_trim_messages(self):
        from shadowd.memory_tools import ContextWindowBudget
        budget = ContextWindowBudget(max_context_tokens=8192)
        messages = [{"role": "user", "content": f"Message {i}" * 50} for i in range(100)]
        trimmed = budget.trim_messages(messages, target_tokens=500)
        assert len(trimmed) < len(messages)

    def test_estimate_tokens(self):
        from shadowd.memory_tools import ContextWindowBudget
        budget = ContextWindowBudget()
        tokens = budget.estimate_tokens("Hello world")
        assert tokens > 0
        assert tokens < len("Hello world")


# ══════════════════════════════════════════════════════════════════
# Config Tests
# ══════════════════════════════════════════════════════════════════

class TestConfig:
    def test_import(self):
        from shadowd.config import ShadowConfig, get_config
        assert ShadowConfig is not None

    def test_default_config(self):
        from shadowd.config import ShadowConfig
        config = ShadowConfig()
        assert config.llm.model == "glm-5"
        assert config.llm.api_base == "https://ollama.com/v1"
        assert config.server.port == 8765
        assert config.memory.core_limit == 2000

    def test_from_env(self):
        from shadowd.config import ShadowConfig
        import os
        old = os.environ.get("OLLAMA_MODEL")
        os.environ["OLLAMA_MODEL"] = "test-model"
        try:
            config = ShadowConfig.from_env()
            assert config.llm.model == "test-model"
        finally:
            if old:
                os.environ["OLLAMA_MODEL"] = old
            else:
                os.environ.pop("OLLAMA_MODEL", None)

    def test_config_sections(self):
        from shadowd.config import ShadowConfig
        config = ShadowConfig()
        assert hasattr(config, 'llm')
        assert hasattr(config, 'memory')
        assert hasattr(config, 'shikigami')
        assert hasattr(config, 'observer')
        assert hasattr(config, 'security')
        assert hasattr(config, 'server')


# ══════════════════════════════════════════════════════════════════
# Superhuman Algorithm Integration Tests
# ══════════════════════════════════════════════════════════════════

class TestSuperhumanAlgorithmsWired:
    def test_bayesian_predictor_exists(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert hasattr(registry, 'bayesian')
        from shikigami.superhuman import BayesianPredictor
        assert isinstance(registry.bayesian, BayesianPredictor)

    def test_signal_triangulator_exists(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert hasattr(registry, 'signal_triangulator')
        from shikigami.superhuman import SignalTriangulator
        assert isinstance(registry.signal_triangulator, SignalTriangulator)

    def test_knowledge_graph_exists(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert hasattr(registry, 'knowledge_graph')
        from shikigami.superhuman import KnowledgeGraph
        assert isinstance(registry.knowledge_graph, KnowledgeGraph)

    def test_markov_exists(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert hasattr(registry, 'markov')
        from shikigami.superhuman import MarkovBehaviorModel
        assert isinstance(registry.markov, MarkovBehaviorModel)

    def test_rl_learner_exists(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert hasattr(registry, 'rl_learner')
        from shikigami.superhuman import ReinforcementLearner
        assert isinstance(registry.rl_learner, ReinforcementLearner)

    def test_env_model_exists(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert hasattr(registry, 'env_model')
        from shikigami.superhuman import EnvironmentModel
        assert isinstance(registry.env_model, EnvironmentModel)

    def test_bayesian_hypotheses_initialized(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        assert "user_active" in registry.bayesian.hypotheses
        assert "user_idle" in registry.bayesian.hypotheses

    def test_markov_trained_from_observations(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        registry = EnhancedShikigamiRegistry()
        # Should have been trained even with empty observations
        assert hasattr(registry, 'markov')
        assert isinstance(registry.markov.state_counts, dict)


# ══════════════════════════════════════════════════════════════════
# ShadowAI Integration Tests
# ══════════════════════════════════════════════════════════════════

class TestShadowAIIntegration:
    @pytest.mark.asyncio
    async def test_event_bus_wired(self):
        from shadowd.shadow_ai import ShadowAI
        from shadowd.event_bus import EventBus, EventType
        mock_registry = MagicMock()
        mock_registry._scan_bt = AsyncMock(return_value=[])
        mock_registry._scan_wifi = AsyncMock(return_value=[])
        mock_registry.close = AsyncMock()
        mock_mind = MagicMock()
        mock_mind.close = AsyncMock()

        ai = ShadowAI(mock_registry, mock_mind, data_dir="/tmp/shadow_test")
        assert hasattr(ai, 'event_bus')
        assert isinstance(ai.event_bus, EventBus)

    @pytest.mark.asyncio
    async def test_stuck_detector_wired(self):
        from shadowd.shadow_ai import ShadowAI
        from shadowd.reasoning.stuck_detector import StuckDetector

        mock_registry = MagicMock()
        mock_registry._scan_bt = AsyncMock(return_value=[])
        mock_registry._scan_wifi = AsyncMock(return_value=[])
        mock_registry.close = AsyncMock()
        mock_mind = MagicMock()
        mock_mind.close = AsyncMock()

        ai = ShadowAI(mock_registry, mock_mind, data_dir="/tmp/shadow_test")
        assert hasattr(ai, 'stuck_detector')
        assert isinstance(ai.stuck_detector, StuckDetector)

    @pytest.mark.asyncio
    async def test_observe_publishes_event(self):
        from shadowd.shadow_ai import ShadowAI
        from shadowd.event_bus import EventType

        mock_registry = MagicMock()
        mock_registry._scan_bt = AsyncMock(return_value=[])
        mock_registry._scan_wifi = AsyncMock(return_value=[])
        mock_registry.close = AsyncMock()
        mock_mind = MagicMock()
        mock_mind.close = AsyncMock()

        ai = ShadowAI(mock_registry, mock_mind, data_dir="/tmp/shadow_test")
        await ai.observe({"type": "test", "content": {"msg": "hello"}})
        assert len(ai.observations) == 1

        history = ai.event_bus.get_history(event_type=EventType.OBSERVATION)
        assert len(history) == 1
        assert history[0].data["type"] == "test"

    @pytest.mark.asyncio
    async def test_stuck_detection_creates_insight(self):
        from shadowd.shadow_ai import ShadowAI
        from shadowd.reasoning.stuck_detector import StuckDetector, ActionType

        mock_registry = MagicMock()
        mock_registry._scan_bt = AsyncMock(return_value=[])
        mock_registry._scan_wifi = AsyncMock(return_value=[])
        mock_registry.close = AsyncMock()
        mock_mind = MagicMock()
        mock_mind.close = AsyncMock()

        ai = ShadowAI(mock_registry, mock_mind, data_dir="/tmp/shadow_test")
        # Simulate stuck: 5 repeated identical actions
        for _ in range(5):
            ai.stuck_detector.record_action("same_action", tool_name="same_tool")
            await ai.observe({"type": "test", "content": {"msg": "stuck"}})

        # Should have a stuck-related insight
        stuck_insights = [i for i in ai.insights if "stuck" in i.message.lower() or "loop" in i.message.lower()]
        assert len(stuck_insights) > 0


# ══════════════════════════════════════════════════════════════════
# MindEngine ContextBudget Tests
# ══════════════════════════════════════════════════════════════════

class TestMindEngineContextBudget:
    def test_context_budget_injected(self):
        from shadowd.mind import MindEngine
        from shadowd.memory_tools import ContextWindowBudget
        engine = MindEngine()
        assert hasattr(engine, 'context_budget')
        assert isinstance(engine.context_budget, ContextWindowBudget)

    def test_context_budget_used_in_messages(self):
        from shadowd.mind import MindEngine
        engine = MindEngine()
        messages = engine._build_messages("test query", None, [{"type": "test", "content": {"msg": "hi"}, "timestamp": "2025-01-01T00:00:00"}])
        assert len(messages) >= 2
        assert messages[0]["role"] == "system"


# ══════════════════════════════════════════════════════════════════
# MemoryTools with real CoreMemory
# ══════════════════════════════════════════════════════════════════

class TestMemoryToolsWithCoreMemory:
    def test_replace_with_core_memory(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        memory.set_block("human", "User prefers Python over JavaScript")
        tools = MemoryTools(memory)
        result = tools.memory_replace("human", "Python", "Rust")
        assert result.success
        assert "Rust" in memory.get_block("human").value

    def test_insert_with_core_memory(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        tools = MemoryTools(memory)
        result = tools.memory_insert("human", "New fact about user")
        assert result.success

    def test_rethink_with_core_memory(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        memory.set_block("human", "Old info")
        tools = MemoryTools(memory)
        result = tools.memory_rethink("human", "Completely new info about user")
        assert result.success
        assert memory.get_block("human").value == "Completely new info about user"

    def test_nonexistent_block(self):
        from shadowd.memory_tools import MemoryTools
        from shadowd.block_memory import CoreMemory
        memory = CoreMemory()
        tools = MemoryTools(memory)
        result = tools.memory_replace("nonexistent", "old", "new")
        assert not result.success
        assert "not found" in result.error.lower()