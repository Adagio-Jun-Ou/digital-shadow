"""Sprint 198: Deep second-pass audit — reasoning/state_graph, reasoning/context_protocol, 
super_memory, memory_enhanced fixes."""
import pytest
import copy
import threading
import sys
import os
import inspect
from datetime import datetime, timezone


class TestKnowledgeGraphDeadlockFix:
    """SM-198-4/SM-198-8: KG.save() no longer deadlocks on nested lock acquisition."""

    def test_kg_to_dict_unlocked_exists(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        assert hasattr(KnowledgeGraph, '_to_dict_unlocked')

    def test_kg_save_uses_unlocked(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        source = inspect.getsource(KnowledgeGraph.save)
        assert '_to_dict_unlocked' in source
        assert 'self.to_dict()' not in source

    def test_kg_to_dict_still_works(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        kg = KnowledgeGraph()
        result = kg.to_dict()
        assert isinstance(result, dict)
        assert "entities" in result
        assert "relations" in result


class TestMemoryEnhancedTimezone:
    """ME-198-2: Episode.timestamp uses datetime.now(timezone.utc)."""

    def test_episode_timestamp_timezone_aware(self):
        from shadowd.memory_enhanced import Episode
        ep = Episode(id="test-1", timestamp=datetime.now(timezone.utc), type="test", content="hello")
        assert ep.timestamp.tzinfo is not None

    def test_enhanced_memory_add_episode_timezone(self):
        from shadowd.memory_enhanced import EnhancedMemory
        import tempfile
        from pathlib import Path
        em = EnhancedMemory(Path(tempfile.mkdtemp()))
        ep = em.add_episode("test", "hello world")
        assert ep.timestamp.tzinfo is not None


class TestMemoryEnhancedStripBeforeTruncate:
    """ME-198-1/ME-198-8: .strip() before truncation in sanitize and add_to_core."""

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.memory_enhanced import _sanitize_content
        assert _sanitize_content("  hello  ") == "hello"

    def test_sanitize_strips_whitespace(self):
        from shadowd.memory_enhanced import _sanitize
        assert _sanitize("  hello  ") == "hello"

    def test_add_to_core_strips_before_truncate(self):
        from shadowd.memory_enhanced import EnhancedMemory
        import tempfile
        from pathlib import Path
        em = EnhancedMemory(Path(tempfile.mkdtemp()))
        em.add_to_core("test", "  hello  ")
        assert em.core_memory["test"] == "hello"

    def test_get_context_strips_before_truncate(self):
        from shadowd.memory_enhanced import EnhancedMemory
        import tempfile
        from pathlib import Path
        em = EnhancedMemory(Path(tempfile.mkdtemp()))
        em.add_episode("test", "  hello world  ")
        ctx = em.get_context()
        assert "- hello world" in ctx


class TestSuperMemorySanitizeStrip:
    """SM-198-1/SM-198-2/SM-198-3: .strip() in sanitize functions and Entity/Relation truncation."""

    def test_sanitize_strips_whitespace(self):
        from shadowd.memory.super_memory import _sanitize
        assert _sanitize("  hello  ") == "hello"

    def test_sanitize_strict_strips_whitespace(self):
        from shadowd.memory.super_memory import _sanitize_strict
        assert _sanitize_strict("  hello  ") == "hello"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.memory.super_memory import _sanitize_content
        assert _sanitize_content("  hello  ") == "hello"

    def test_entity_name_strip_before_truncate(self):
        from shadowd.memory.super_memory import Entity
        e = Entity(id="test", entity_type="person", name="  Hello World  ")
        assert e.name == "Hello World"

    def test_relation_context_strip_before_truncate(self):
        from shadowd.memory.super_memory import Relation
        r = Relation(id="r1", subject_id="a", predicate="knows", object_id="b",
                      context={"key": "  value  "})
        assert r.context["key"] == "value"


class TestStateGraphExceptNarrowing:
    """SG-198-6: except Exception narrowed in state_graph.py execute()."""

    def test_execute_except_narrowed(self):
        from shadowd.reasoning.state_graph import StateGraph
        source = inspect.getsource(StateGraph.execute)
        assert "except (RuntimeError, ValueError, TypeError, AttributeError, KeyError, OSError)" in source
        assert "except Exception" not in source.replace("except (RuntimeError", "")


class TestContextProtocolDeepCopy:
    """RCP-198-2/RCP-198-4: Capability/ContextMessage to_dict uses copy.deepcopy for metadata/payload."""

    def test_capability_metadata_deepcopy(self):
        from shadowd.reasoning.context_protocol import Capability, Permission
        cap = Capability(name="test", version="1.0",
                        permissions=[Permission.READ],
                        metadata={"nested": {"key": "value"}})
        d = cap.to_dict()
        d["metadata"]["nested"]["key"] = "changed"
        assert cap.metadata["nested"]["key"] == "value"

    def test_context_message_payload_deepcopy(self):
        from shadowd.reasoning.context_protocol import ContextMessage
        msg = ContextMessage(id="m1", sender_id="a", receiver_id="b",
                            message_type="test", payload={"data": "secret"})
        d = msg.to_dict()
        d["payload"]["data"] = "changed"
        assert msg.payload["data"] == "secret"

    def test_context_message_metadata_deepcopy(self):
        from shadowd.reasoning.context_protocol import ContextMessage
        msg = ContextMessage(id="m1", sender_id="a", receiver_id="b",
                            message_type="test", payload={},
                            metadata={"key": "val"})
        d = msg.to_dict()
        d["metadata"]["key"] = "changed"
        assert msg.metadata["key"] == "val"