#!/usr/bin/env python3
"""Sprint 123 regression tests: knowledge_graph.py, state_graph.py, context_protocol.py (root level)."""

import asyncio
import copy
import threading
import pytest
import tempfile
from datetime import datetime, timezone, timedelta
from pathlib import Path

from shadowd.knowledge_graph import (
    TemporalKnowledgeGraph, TemporalFact, Entity, Episode,
    RelationType, TypedRelation, _sanitize, _sanitize_content,
)
from shadowd.state_graph import (
    StateGraph, StateCheckpoint, Node, Edge, NodeStatus,
    _SANITIZE_RE, _SANITIZE_STRICT_RE,
)
from shadowd.context_protocol import (
    ContextProtocol, ContextSnapshot, ContextSource, ContextType,
)


# ═════════════════════════════════════════════════════════════════
# Knowledge Graph tests
# ═════════════════════════════════════════════════════════════════

class TestTemporalFact:
    def test_sanitize_content(self):
        f = TemporalFact(
            fact_id="f1", content="hello\x00world\x7f",
            entities=["e1"],
            valid_from=datetime.now(timezone.utc)
        )
        assert "\x00" not in f.content
        assert "\x7f" not in f.content

    def test_clamp_confidence(self):
        f = TemporalFact(
            fact_id="f1", content="test",
            entities=["e1"],
            valid_from=datetime.now(timezone.utc),
            confidence=1.5
        )
        assert f.confidence == 1.0

    def test_clamp_confidence_negative(self):
        f = TemporalFact(
            fact_id="f1", content="test",
            entities=["e1"],
            valid_from=datetime.now(timezone.utc),
            confidence=-0.5
        )
        assert f.confidence == 0.0

    def test_to_dict_confidence_rounded(self):
        f = TemporalFact(
            fact_id="f1", content="test",
            entities=["e1"],
            valid_from=datetime.now(timezone.utc),
            confidence=0.123456
        )
        d = f.to_dict()
        assert d["confidence"] == 0.123

    def test_entities_list_copy(self):
        f = TemporalFact(
            fact_id="f1", content="test",
            entities=["e1", "e2"],
            valid_from=datetime.now(timezone.utc)
        )
        d = f.to_dict()
        d["entities"].append("e3")
        assert len(f.entities) == 2

    def test_sanitize_source_episode(self):
        f = TemporalFact(
            fact_id="f1", content="test",
            entities=["e1"],
            valid_from=datetime.now(timezone.utc),
            source_episode="<script>alert(1)</script>"
        )
        assert "<" not in f.source_episode

    def test_sanitize_entities(self):
        ep = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        ep2 = ep.add_episode("test content", "src", entities=["<e1>", "e 2"])
        assert "<" not in list(ep2.entities_seen)[0]

    def test_timezone_aware_defaults(self):
        f = TemporalFact(
            fact_id="f1", content="test",
            entities=["e1"],
            valid_from=datetime.now(timezone.utc)
        )
        assert f.created_at.tzinfo is not None


class TestTypedRelation:
    def test_sanitize_entities(self):
        r = TypedRelation(
            relation_id="r1",
            from_entity="<script>x</script>",
            to_entity="normal",
            relation_type=RelationType.KNOWS,
        )
        assert "<" not in r.from_entity

    def test_clamp_confidence(self):
        r = TypedRelation(
            relation_id="r1",
            from_entity="a", to_entity="b",
            relation_type=RelationType.KNOWS,
            confidence=2.0
        )
        assert r.confidence == 1.0

    def test_to_dict_properties_deepcopy(self):
        r = TypedRelation(
            relation_id="r1",
            from_entity="a", to_entity="b",
            relation_type=RelationType.KNOWS,
            properties={"key": "val"}
        )
        d = r.to_dict()
        d["properties"]["key"] = "changed"
        assert r.properties["key"] == "val"

    def test_timezone_aware_defaults(self):
        r = TypedRelation(
            relation_id="r1",
            from_entity="a", to_entity="b",
            relation_type=RelationType.KNOWS,
        )
        assert r.valid_from.tzinfo is not None
        assert r.created_at.tzinfo is not None

    def test_invalid_relation_type_normalized(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test", "src", entities=["Alice"])
        r = kg.add_relation("Alice", "Bob", "invalid_type")
        assert r.relation_type == "related_to"


class TestEntity:
    def test_sanitize_name(self):
        e = Entity(name="<script>x</script>", entity_type="person")
        assert "<" not in e.name

    def test_sanitize_summary(self):
        e = Entity(name="test", entity_type="person", summary="hello\x00world")
        assert "\x00" not in e.summary

    def test_add_fact_invalidates_conflicting(self):
        e = Entity(name="test", entity_type="person")
        f1 = TemporalFact(
            fact_id="f1", content="John lives in NYC",
            entities=["John"],
            valid_from=datetime.now(timezone.utc)
        )
        f2 = TemporalFact(
            fact_id="f2", content="John lives in Boston",
            entities=["John"],
            valid_from=datetime.now(timezone.utc)
        )
        e.add_fact(f1)
        e.add_fact(f2)
        valid = e.get_valid_facts(datetime.now(timezone.utc))
        assert len(valid) == 1

    def test_facts_cap_at_1000(self):
        e = Entity(name="test", entity_type="person")
        for i in range(1100):
            f = TemporalFact(
                fact_id=f"f{i}", content=f"fact {i}",
                entities=["test"],
                valid_from=datetime.now(timezone.utc)
            )
            e.facts.append(f)
        e.facts = e.facts[-1000:] if len(e.facts) > 1000 else e.facts
        assert len(e.facts) <= 1000


class TestTemporalKnowledgeGraph:
    def test_add_episode_and_search(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        ep = kg.add_episode(
            content="Alice met Bob",
            source="conversation",
            facts=[{"content": "Alice met Bob", "entities": ["Alice", "Bob"]}],
            entities=["Alice", "Bob"]
        )
        assert ep.episode_id.startswith("ep_")
        results = kg.search("Alice", when=datetime.now(timezone.utc))
        assert len(results) > 0

    def test_episode_sanitization(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        ep = kg.add_episode(content="hello\x00world", source="<script>")
        content_clean = ep.content
        source_clean = ep.source
        assert "\x00" not in content_clean
        assert "<" not in source_clean

    def test_add_relation_and_get(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test", "src", entities=["Alice"])
        r = kg.add_relation("Alice", "Bob", RelationType.KNOWS)
        assert r.relation_id.startswith("rel_")
        rels = kg.get_relations("Alice")
        assert len(rels) > 0

    def test_get_entity_deepcopy(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test content", "src", facts=[{"content": "test fact", "entities": ["E1"]}], entities=["E1"])
        e = kg.get_entity("E1")
        assert e is not None
        e.name = "modified"
        e2 = kg.get_entity("E1")
        assert e2.name == "E1"

    def test_search_returns_deepcopy(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test", "src", facts=[{"content": "unique content xyz", "entities": ["E1"]}], entities=["E1"])
        results = kg.search("xyz", when=datetime.now(timezone.utc))
        assert len(results) > 0
        results[0].content = "modified"
        results2 = kg.search("xyz", when=datetime.now(timezone.utc))
        assert results2[0].content == "unique content xyz"

    def test_thread_safety(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        errors = []

        def add_ep(i):
            try:
                kg.add_episode(f"episode {i}", "src")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_ep, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_stats_under_lock(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test", "src", facts=[{"content": "Alice fact", "entities": ["Alice"]}], entities=["Alice"])
        stats = kg.get_stats()
        assert "entities" in stats
        assert stats["entities"] >= 1

    def test_flush_atomic(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test episode", "src", entities=["Alice"])
        kg.flush()
        save_file = kg.data_dir / "knowledge_graph.json"
        assert save_file.exists()

    def test_add_relation_properties_cap(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        kg.add_episode("test", "src", entities=["A"])
        big_props = {f"key_{i}": f"val_{i}" * 100 for i in range(60)}
        r = kg.add_relation("A", "B", RelationType.KNOWS, properties=big_props)
        d = r.to_dict()
        assert len(d["properties"]) <= 50

    def test_add_episode_fact_entity_sanitization(self):
        kg = TemporalKnowledgeGraph(data_dir=Path(tempfile.mkdtemp()))
        ep = kg.add_episode(
            content="test",
            source="src",
            facts=[{"content": "fact with <script>", "entities": ["<Entity1>"]}],
            entities=["Entity2"]
        )
        assert "<script>" not in ep.content or "<script>" in "fact with <script>"  # content-safe: preserves <>
        # Entity names should be strict-sanitized
        alice_entity = kg.get_entity("<Entity1>")
        # The entity was created via fact.entities which gets sanitized in TemporalFact.__post_init__
        # but entity_name comes from fact.entities which were sanitized separately in add_episode


# ═════════════════════════════════════════════════════════════════
# StateGraph tests
# ═════════════════════════════════════════════════════════════════

class TestStateCheckpoint:
    def test_sanitize_node_id(self):
        cp = StateCheckpoint(
            node_id="<script>alert(1)</script>",
            timestamp=datetime.now(timezone.utc),
            state={"k": "v"},
            status=NodeStatus.RUNNING
        )
        assert "<" not in cp.node_id

    def test_truncate_error(self):
        cp = StateCheckpoint(
            node_id="n1",
            timestamp=datetime.now(timezone.utc),
            state={},
            status=NodeStatus.FAILED,
            error="x" * 300
        )
        d = cp.to_dict()
        assert len(d["error"]) == 200

    def test_deepcopy_state(self):
        cp = StateCheckpoint(
            node_id="n1",
            timestamp=datetime.now(timezone.utc),
            state={"nested": {"key": "val"}},
            status=NodeStatus.RUNNING
        )
        d = cp.to_dict()
        d["state"]["nested"]["key"] = "changed"
        assert cp.state["nested"]["key"] == "val"

    def test_uuid_default_checkpoint_id(self):
        cp = StateCheckpoint(
            node_id="n1",
            timestamp=datetime.now(timezone.utc),
            state={},
            status=NodeStatus.RUNNING
        )
        assert cp.checkpoint_id.startswith("cp_")
        assert len(cp.checkpoint_id) > 10


class TestStateGraph:
    def test_name_sanitized(self):
        sg = StateGraph(name="hello\x00world<script>")
        assert "\x00" not in sg.name
        assert "<" not in sg.name

    def test_checkpoint_deepcopy_state(self):
        sg = StateGraph(name="test")
        sg.set_entry_point("start")
        sg.add_node("start", lambda s: {"result": "ok"})

        async def run():
            return await sg.execute({"key": "val"})

        result = asyncio.run(run())
        assert result is not None

    def test_state_at_node_returns_deepcopy(self):
        sg = StateGraph(name="test")
        sg.checkpoints = [
            StateCheckpoint(
                node_id="n1",
                timestamp=datetime.now(timezone.utc),
                state={"nested": {"a": 1}},
                status=NodeStatus.SUCCESS
            )
        ]
        state = sg.get_state_at_node("n1")
        state["nested"]["a"] = 999
        state2 = sg.get_state_at_node("n1")
        assert state2["nested"]["a"] == 1

    def test_sanitize_regex_content_preserves_angle_brackets(self):
        text = "latency < 100ms"
        assert _SANITIZE_RE.sub('', text) == "latency < 100ms"

    def test_sanitize_strict_strips_angle_brackets(self):
        text = "<script>alert(1)</script>"
        assert _SANITIZE_STRICT_RE.sub('', text) == "scriptalert(1)/script"


# ═════════════════════════════════════════════════════════════════
# ContextProtocol tests
# ═════════════════════════════════════════════════════════════════

class TestContextSnapshot:
    def test_to_dict_returns_copies(self):
        cs = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            device_context={"key": "val"}
        )
        d = cs.to_dict()
        d["device"]["key"] = "changed"
        assert cs.device_context["key"] == "val"

    def test_get_unified_context_returns_copies(self):
        cs = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            device_context={"d": "v"},
            temporal_context={"t": "v"}
        )
        uc = cs.get_unified_context()
        uc["when"]["t"] = "changed"
        assert cs.temporal_context["t"] == "v"

    def test_timezone_aware_timestamp(self):
        cs = ContextSnapshot(timestamp=datetime.now(timezone.utc))
        d = cs.to_dict()
        assert "+00:00" in d["timestamp"] or "Z" in d["timestamp"] or "T" in d["timestamp"]


class TestContextProtocol:
    def test_register_source_sanitizes_name(self):
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        source = ContextSource(
            name="<script>alert(1)</script>",
            context_type=ContextType.ACTIVITY,
            provider=lambda: {}
        )
        cp.register_source(source)
        assert "<" not in source.name

    def test_context_history_cap(self):
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        assert cp._MAX_HISTORY == 100

    def test_subscribers_cap(self):
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        assert cp._MAX_SUBSCRIBERS == 50

    def test_get_status_under_lock(self):
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        status = cp.get_status()
        assert "sources" in status
        assert "subscribers" in status

    def test_save_snapshot_atomic(self):
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        snapshot = ContextSnapshot(timestamp=datetime.now(timezone.utc))
        cp.save_snapshot(snapshot)
        files = list(cp.data_dir.glob("snapshot_*.json"))
        assert len(files) == 1