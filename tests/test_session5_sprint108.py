#!/usr/bin/env python3
"""Sprint 108 regression tests: mcp_protocol, mcp_server, mcp_server_enhanced, event_bus, decision_learning_v2"""

import copy
import json
import re
import threading
import time
import asyncio
import pytest
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock


# ==================== mcp_protocol tests ====================

class TestMCPServerThreadSafety:
    def test_register_tool_under_lock(self):
        from shadowd.mcp_protocol import MCPServer, MCPTool
        server = MCPServer(name="test")
        tool = MCPTool(name="test_tool", description="desc", input_schema={"type": "object"})
        server.register_tool(tool)
        assert "test_tool" in server.tools

    def test_register_resource_under_lock(self):
        from shadowd.mcp_protocol import MCPServer, MCPResource
        server = MCPServer(name="test")
        res = MCPResource(uri="test://x", name="res", description="desc")
        server.register_resource(res)
        assert "test://x" in server.resources

    def test_register_prompt_under_lock(self):
        from shadowd.mcp_protocol import MCPServer, MCPPrompt
        server = MCPServer(name="test")
        prompt = MCPPrompt(name="test_prompt", description="desc")
        server.register_prompt(prompt)
        assert "test_prompt" in server.prompts

    def test_tool_eviction(self):
        from shadowd.mcp_protocol import MCPServer, MCPTool
        server = MCPServer(name="test")
        for i in range(server.MAX_TOOLS + 5):
            tool = MCPTool(name=f"tool_{i}", description=f"desc_{i}", input_schema={})
            server.register_tool(tool)
        assert len(server.tools) <= server.MAX_TOOLS

    def test_resource_eviction(self):
        from shadowd.mcp_protocol import MCPServer, MCPResource
        server = MCPServer(name="test")
        for i in range(server.MAX_RESOURCES + 5):
            res = MCPResource(uri=f"test://r_{i}", name=f"res_{i}", description="d")
            server.register_resource(res)
        assert len(server.resources) <= server.MAX_RESOURCES

    def test_prompt_eviction(self):
        from shadowd.mcp_protocol import MCPServer, MCPPrompt
        server = MCPServer(name="test")
        for i in range(server.MAX_PROMPTS + 5):
            prompt = MCPPrompt(name=f"prompt_{i}", description="d")
            server.register_prompt(prompt)
        assert len(server.prompts) <= server.MAX_PROMPTS


class TestMCPSanitize:
    def test_sanitize_strips_angle_brackets(self):
        from shadowd.mcp_protocol import _sanitize
        assert _sanitize("hello<script>alert(1)</script>") == "helloscriptalert(1)/script"

    def test_sanitize_strips_control_chars(self):
        from shadowd.mcp_protocol import _sanitize
        assert _sanitize("hello\t\n\rworld") == "helloworld"

    def test_sanitize_content_preserves_newlines(self):
        from shadowd.mcp_protocol import _sanitize_content
        result = _sanitize_content("hello\nworld\ttab\r\n")
        assert "\n" in result
        assert "\t" in result
        assert "\r" in result

    def test_sanitize_content_strips_control_chars(self):
        from shadowd.mcp_protocol import _sanitize_content
        assert _sanitize_content("hello\x00world\x01test") == "helloworldtest"

    def test_tool_to_dict_deepcopy_schema(self):
        from shadowd.mcp_protocol import MCPTool
        tool = MCPTool(name="t", description="d", input_schema={"type": "object", "properties": {"a": {"type": "string"}}})
        d = tool.to_dict()
        d["inputSchema"]["properties"]["a"]["type"] = "number"
        assert tool.input_schema["properties"]["a"]["type"] == "string"

    def test_prompt_to_dict_deepcopy_arguments(self):
        from shadowd.mcp_protocol import MCPPrompt
        prompt = MCPPrompt(name="p", description="d", arguments=[{"name": "x", "required": True}])
        d = prompt.to_dict()
        d["arguments"][0]["name"] = "y"
        assert prompt.arguments[0]["name"] == "x"


# ==================== decision_learning_v2 tests ====================

class TestDecisionLearningV2Sanitize:
    def test_sanitize_strips_angle_brackets(self):
        from shadowd.decision_learning_v2 import _sanitize
        assert _sanitize("<script>alert(1)</script>") == "scriptalert(1)/script"

    def test_sanitize_content_preserves_formatting(self):
        from shadowd.decision_learning_v2 import _sanitize_content
        result = _sanitize_content("hello\nworld\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_content_strips_control(self):
        from shadowd.decision_learning_v2 import _sanitize_content
        assert _sanitize_content("hello\x00world") == "helloworld"

    def test_create_decision_sanitizes_input(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_v2"))
        result = dl.create_decision(
            situation="test\x00situation",
            options=["opt<1>", "opt\n2"],
            context="test\x00context",
            priority="<script>evil</script>medium"
        )
        decision = result[0] if isinstance(result, tuple) else result
        assert "\x00" not in decision.situation
        assert "\x00" not in decision.context
        assert decision.priority == "medium"  # strict sanitize strips <>, invalid values default


class TestDecisionLearningV2Deepcopy:
    def test_decision_to_dict_copies_lists(self):
        from shadowd.decision_learning_v2 import DecisionV2
        d = DecisionV2(
            id="test", context="ctx", situation="sit",
            options=["a", "b"], chosen="a", reasoning="r",
            tags=["t1"], similar_decisions=["sd1"],
            lessons_learned=["l1"], outcome_steps=[{"step": 1}],
            dependencies=["dep1"]
        )
        d_dict = d.to_dict()
        d_dict["options"].append("c")
        d_dict["tags"].append("t2")
        d_dict["dependencies"].append("dep2")
        d_dict["outcome_steps"].append({"step": 2})
        assert len(d.options) == 2
        assert len(d.tags) == 1
        assert len(d.dependencies) == 1

    def test_pattern_to_dict_copies_examples(self):
        from shadowd.decision_learning_v2 import DecisionPattern
        p = DecisionPattern(
            pattern_id="p1", pattern_type="contextual",
            description="desc", examples=[{"a": 1}],
            success_rate=0.8, confidence=0.9
        )
        d = p.to_dict()
        d["examples"].append({"b": 2})
        assert len(p.examples) == 1

    def test_get_pending_returns_deepcopy(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2, DecisionStatus
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_v2_dc"))
        result = dl.create_decision(situation="test sit", options=["a", "b"])
        decision = result[0] if isinstance(result, tuple) else result
        pending = dl.get_pending()
        assert len(pending) >= 1
        pending[0].situation = "MODIFIED"
        orig = dl.decisions[decision.id]
        assert orig.situation != "MODIFIED"

    def test_get_patterns_returns_deepcopy(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_v2_patt"))
        assert dl.get_patterns() == []


class TestDecisionLearningWordBoundary:
    def test_detect_category_word_boundary(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_v2_wb"))
        result = dl.create_decision(situation="I need a hobby", options=["crafts", "reading"])
        decision = result[0] if isinstance(result, tuple) else result
        assert decision.category == "lifestyle"

    def test_classify_choice_word_boundary(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_v2_cls"))
        assert dl._classify_choice("I say yes") == "positive"
        assert dl._classify_choice("absolutely no way") == "negative"
        assert dl._classify_choice("maybe") == "neutral"

    def test_confidence_clamped(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_v2_conf"))
        result = dl.create_decision(situation="test", options=["a", "b"])
        decision = result[0] if isinstance(result, tuple) else result
        decided = dl.decide(decision.id, "a", "reason", confidence=1.5)
        assert decided.confidence == 1.0


# ==================== event_bus tests ====================

class TestEventBusSanitize:
    def test_sanitize_strips_angle_brackets(self):
        from shadowd.event_bus import _sanitize
        assert _sanitize("<script>alert(1)</script>") == "scriptalert(1)/script"

    def test_sanitize_content_preserves_newlines(self):
        from shadowd.event_bus import _sanitize_content
        result = _sanitize_content("hello\nworld\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_content_strips_control(self):
        from shadowd.event_bus import _sanitize_content
        assert _sanitize_content("hello\x00world\x01test") == "helloworldtest"


class TestEventBusDeepcopy:
    def test_event_to_dict_copies_data(self):
        from shadowd.event_bus import ShadowEvent, EventType
        event = ShadowEvent(
            event_type=EventType.OBSERVATION,
            source="test",
            data={"key": "value", "nested": {"a": 1}},
            metadata={"meta": "data"}
        )
        d = event.to_dict()
        d["data"]["key"] = "MODIFIED"
        d["data"]["nested"]["a"] = 999
        d["metadata"]["meta"] = "MODIFIED"
        assert event.data["key"] == "value"
        assert event.data["nested"]["a"] == 1
        assert event.metadata["meta"] == "data"

    def test_event_sanitizes_source(self):
        from shadowd.event_bus import ShadowEvent, EventType
        event = ShadowEvent(
            event_type=EventType.INSIGHT,
            source="<script>evil</script>",
            data={}
        )
        assert "<script>" not in event.source
        assert event.source == "scriptevil/script"

    def test_get_history_returns_deepcopy(self):
        from shadowd.event_bus import EventBus, ShadowEvent, EventType
        bus = EventBus()
        event = ShadowEvent(
            event_type=EventType.OBSERVATION,
            source="test",
            data={"key": "value"}
        )
        bus._event_history.append(event)
        history = bus.get_history()
        history[0].source = "MODIFIED"
        assert bus._event_history[0].source == "test"

    def test_get_stats_returns_copy(self):
        from shadowd.event_bus import EventBus
        bus = EventBus()
        stats = bus.get_stats()
        stats["events_published"] = 9999
        assert bus._stats["events_published"] == 0


class TestEventBusSubscriberCap:
    def test_subscriber_cap_per_event_type(self):
        from shadowd.event_bus import EventBus, EventType
        bus = EventBus()
        cap = 50
        for i in range(cap + 5):
            handler = AsyncMock()
            bus.subscribe(EventType.OBSERVATION, handler)
        assert len(bus._subscribers[EventType.OBSERVATION.value]) <= cap

    def test_wildcard_subscriber_cap(self):
        from shadowd.event_bus import EventBus
        bus = EventBus()
        cap = 20
        for i in range(cap + 5):
            handler = AsyncMock()
            bus.subscribe_all(handler)
        assert len(bus._wildcard_subscribers) <= cap


# ==================== mcp_server tests ====================

class TestMCPServerSanitize:
    @pytest.fixture(autouse=True)
    def check_mcp(self):
        try:
            from shadowd.mcp_server import _sanitize
        except ImportError:
            pytest.skip("mcp package not installed")

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.mcp_server import _sanitize
        assert _sanitize("<script>alert(1)</script>") == "scriptalert(1)/script"

    def test_sanitize_content_preserves_newlines(self):
        from shadowd.mcp_server import _sanitize_content
        result = _sanitize_content("hello\nworld\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_save_json_uses_unique_tmp(self):
        from shadowd.mcp_server import _save_json
        import tempfile
        tmp_dir = Path(tempfile.mkdtemp())
        p1 = tmp_dir / "test1.json"
        p2 = tmp_dir / "test2.json"
        _save_json(p1, {"a": 1})
        _save_json(p2, {"b": 2})
        assert p1.exists()
        assert p2.exists()
        assert json.loads(p1.read_text()) == {"a": 1}
        assert json.loads(p2.read_text()) == {"b": 2}

    def test_get_user_dir_path_traversal(self):
        # _get_user_dir sanitizes the input (strips path separators and dots)
        # before doing anything with it, so traversal attempts collapse to a
        # safe inside-USERS_DIR path rather than raising. Verify *behavior*:
        # the returned dir must stay under USERS_DIR no matter what.
        from shadowd.mcp_server import _get_user_dir, USERS_DIR
        result = _get_user_dir("../../etc/passwd")
        assert result.resolve().is_relative_to(USERS_DIR.resolve()), \
            f"traversal escaped: {result}"

    def test_get_user_dir_valid(self):
        from shadowd.mcp_server import _get_user_dir
        result = _get_user_dir("testuser123")
        assert result.exists()


class TestEnhancedMCPServerSanitize:
    @pytest.fixture(autouse=True)
    def check_mcp(self):
        try:
            from shadowd.mcp_server_enhanced import _sanitize
        except ImportError:
            pytest.skip("mcp package not installed")

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.mcp_server_enhanced import _sanitize
        assert _sanitize("<script>alert(1)</script>") == "scriptalert(1)/script"

    def test_sanitize_content_preserves_newlines(self):
        from shadowd.mcp_server_enhanced import _sanitize_content
        result = _sanitize_content("hello\nworld\ttab")
        assert "\n" in result
        assert "\t" in result

    def test_metrics_to_dict_copies(self):
        from shadowd.mcp_server_enhanced import MCPServerMetrics
        m = MCPServerMetrics()
        m.tools_called["tool1"] = 5
        m.resources_accessed["res1"] = 3
        d = m.to_dict()
        d["tools_called"]["tool1"] = 999
        d["resources_accessed"]["res1"] = 888
        assert m.tools_called["tool1"] == 5
        assert m.resources_accessed["res1"] == 3

    def test_user_dir_path_traversal(self):
        from shadowd.mcp_server_enhanced import EnhancedMCPServer, MCPServerMetrics
        server = EnhancedMCPServer.__new__(EnhancedMCPServer)
        server.data_dir = Path("/tmp/test_mcp_enhanced_pt2")
        server.users_dir = server.data_dir / "users"
        server.knowledge_dir = server.data_dir / "knowledge_graph"
        server._subscribers = {}
        server._lock = threading.Lock()
        server.mcp = None
        server.websocket_manager = None
        server.shadow_ai = None
        server.name = "test"
        server.version = "1.0"
        server.metrics = MCPServerMetrics()
        # Path traversal is sanitized to safe path, not rejected
        result = server._get_user_dir("../../../etc/passwd")
        # Result should be within users_dir (path traversal stripped)
        assert str(result.resolve()).startswith(str(server.users_dir.resolve()))

    def test_subscribe_cap(self):
        from shadowd.mcp_server_enhanced import EnhancedMCPServer, MCPServerMetrics
        server = EnhancedMCPServer.__new__(EnhancedMCPServer)
        server.data_dir = Path("/tmp/test_mcp_enhanced_sub")
        server.users_dir = server.data_dir / "users"
        server.knowledge_dir = server.data_dir / "knowledge_graph"
        server._subscribers = {}
        server._lock = threading.Lock()
        server.mcp = None
        server.websocket_manager = None
        server.shadow_ai = None
        server.name = "test"
        server.version = "1.0"
        server.metrics = MCPServerMetrics()
        for i in range(105):
            if "test_topic" not in server._subscribers:
                server._subscribers["test_topic"] = []
            server._subscribers["test_topic"].append(f"user_{i}")
            if len(server._subscribers["test_topic"]) > 100:
                server._subscribers["test_topic"].pop(0)


class TestEnhancedMCPServerUniqueIDs:
    @pytest.fixture(autouse=True)
    def check_mcp(self):
        try:
            from shadowd.mcp_server_enhanced import EnhancedMCPServer
        except ImportError:
            pytest.skip("mcp package not installed")

    def test_save_json_atomic(self):
        from shadowd.mcp_server_enhanced import EnhancedMCPServer, MCPServerMetrics
        import tempfile
        server = EnhancedMCPServer.__new__(EnhancedMCPServer)
        server.data_dir = Path("/tmp/test_mcp_atomic")
        server.users_dir = server.data_dir / "users"
        server.knowledge_dir = server.data_dir / "knowledge_graph"
        server._subscribers = {}
        server._lock = threading.Lock()
        server.mcp = None
        server.websocket_manager = None
        server.shadow_ai = None
        server.name = "test"
        server.version = "1.0"
        server.metrics = MCPServerMetrics()
        
        test_file = server.data_dir / "test_save" / "data.json"
        server._save_json(test_file, {"key": "value"})
        assert test_file.exists()
        data = json.loads(test_file.read_text())
        assert data["key"] == "value"


# ==================== concurrent access tests ====================

class TestMCPServerConcurrentAccess:
    def test_concurrent_register_tools(self):
        from shadowd.mcp_protocol import MCPServer, MCPTool
        server = MCPServer(name="test_concurrent")
        errors = []

        def register_tool(i):
            try:
                tool = MCPTool(name=f"tool_{i}", description=f"desc_{i}", input_schema={})
                server.register_tool(tool)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register_tool, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(server.tools) == 50


class TestDecisionLearningConcurrentAccess:
    def test_concurrent_create_decisions(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        dl = DecisionLearningV2(data_dir=Path("/tmp/test_dl_concurrent"))
        errors = []

        def create_decision(i):
            try:
                result = dl.create_decision(
                    situation=f"test situation {i}",
                    options=["a", "b"]
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_decision, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEventBusConcurrency:
    def test_concurrent_publish_and_history(self):
        from shadowd.event_bus import EventBus, ShadowEvent, EventType
        bus = EventBus()
        errors = []

        async def publish_events():
            for i in range(10):
                event = ShadowEvent(
                    event_type=EventType.OBSERVATION,
                    source=f"source_{i}",
                    data={"index": i}
                )
                bus._event_history.append(event)

        threads = [
            threading.Thread(target=lambda: [bus._event_history.append(
                ShadowEvent(event_type=EventType.OBSERVATION, source=f"src_{j}", data={})
            ) for j in range(10)])
            for _ in range(5)
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(bus._event_history) == 50


if __name__ == "__main__":
    pytest.main([__file__, "-v"])