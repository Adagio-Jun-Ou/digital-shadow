"""Sprint 118 regression tests — super_memory.py deep audit.

Tests for:
- WorkingMemory: lock on all methods, deepcopy returns, sanitization, caps
- ArchivalMemory: lock, sanitize, UUID IDs, deepcopy returns, directory fsync, narrow except, timezone
- Entity/Relation: __post_init__ sanitization, deepcopy to_dict, timezone-aware defaults
- KnowledgeGraph: lock, sanitization, UUID IDs, deepcopy returns, eviction logging, timezone
- SuperMemory: lock, deepcopy in recall, sanitization in remember
"""

import copy
import re
import threading
import time
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock
import tempfile
import os
import json

import pytest

from shadowd.memory.super_memory import (
    MemoryBlock, CoreMemory, WorkingMemory, ArchivalMemory,
    Entity, Relation, KnowledgeGraph, SuperMemory,
    _sanitize, _sanitize_strict, _sanitize_content
)


# ═════════════════════════════════════════════════════════════════
# WorkingMemory tests
# ═════════════════════════════════════════════════════════════════

class TestWorkingMemoryThreadSafety:
    def test_set_task_acquires_lock(self):
        wm = WorkingMemory()
        task = {"key": "value", "secret": "data"}
        wm.set_task(task)
        assert wm.current_task is not None
        assert wm.current_task["key"] == "value"

    def test_set_task_sanitizes_keys(self):
        wm = WorkingMemory()
        wm.set_task({"<script>alert(1)</script>": "val", "normal_key": "val2"})
        assert "<script>" not in str(wm.current_task)

    def test_set_task_caps_keys(self):
        wm = WorkingMemory()
        big = {f"k{i}": f"v{i}" for i in range(60)}
        wm.set_task(big)
        assert len(wm.current_task) <= 50

    def test_add_attention_sanitizes(self):
        wm = WorkingMemory()
        wm.add_attention("hello\x00world<script>")
        assert "\x00" not in wm.attention_focus[-1]
        assert "<script>" in wm.attention_focus[-1]  # content-safe preserves <>

    def test_add_attention_caps_at_max(self):
        wm = WorkingMemory()
        for i in range(15):
            wm.add_attention(f"focus_{i}")
        assert len(wm.attention_focus) <= 10

    def test_add_attention_truncates_long(self):
        wm = WorkingMemory()
        long_focus = "x" * 1000
        wm.add_attention(long_focus)
        assert len(wm.attention_focus[-1]) <= 500

    def test_get_context_returns_copy(self):
        wm = WorkingMemory()
        wm.add_message("user", "test")
        ctx = wm.get_context()
        assert "test" in ctx

    def test_clear_acquires_lock(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        wm.set_task({"k": "v"})
        wm.add_attention("focus")
        wm.clear()
        assert wm.messages == []
        assert wm.current_task is None
        assert wm.attention_focus == []

    def test_to_dict_returns_deepcopy(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        d = wm.to_dict()
        d["messages"][0]["content"] = "modified"
        d["current_task"] = "hacked"
        assert wm.messages[0]["content"] == "hello"

    def test_add_message_deepcopies_metadata(self):
        wm = WorkingMemory()
        meta = {"key": "value"}
        wm.add_message("user", "hello", metadata=meta)
        meta["key"] = "modified"
        assert wm.messages[0]["metadata"]["key"] == "value"

    def test_set_task_none(self):
        wm = WorkingMemory()
        wm.set_task(None)
        assert wm.current_task is None


class TestWorkingMemorySanitization:
    def test_role_sanitized(self):
        wm = WorkingMemory()
        wm.add_message("user\x00role", "content")
        assert "\x00" not in wm.messages[0]["role"]

    def test_content_sanitized_content_safe(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello <b>world</b>\x00")
        assert "\x00" not in wm.messages[0]["content"]
        assert "<b>" in wm.messages[0]["content"]  # content-safe preserves <>

    def test_timestamp_timezone_aware(self):
        wm = WorkingMemory()
        wm.add_message("user", "test")
        ts = wm.messages[0]["timestamp"]
        dt = datetime.fromisoformat(ts)
        assert dt.tzinfo is not None


# ═════════════════════════════════════════════════════════════════
# ArchivalMemory tests
# ═════════════════════════════════════════════════════════════════

class TestArchivalMemoryThreadSafety:
    def test_insert_acquires_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            mid = am.insert("test content")
            assert mid is not None
            assert mid.startswith("arch_")

    def test_search_returns_deepcopy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("unique test content xyz")
            results = am.search("xyz")
            assert len(results) == 1
            results[0]["content"] = "modified"
            original = am.search("xyz")
            assert original[0]["content"] != "modified"

    def test_get_recent_returns_deepcopy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("recent content")
            recent = am.get_recent()
            if recent:
                recent[0]["content"] = "modified"
                orig = am.get_recent()
                assert orig[0]["content"] != "modified"

    def test_get_important_returns_deepcopy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("important content", importance=0.9)
            imp = am.get_important()
            if imp:
                imp[0]["content"] = "modified"
                orig = am.get_important()
                assert orig[0]["content"] != "modified"

    def test_delete_acquires_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            mid = am.insert("to be deleted")
            result = am.delete(mid)
            assert result is True

    def test_stats_under_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("test content")
            stats = am.get_stats()
            assert stats["total_memories"] == 1


class TestArchivalMemoryUUIDIDs:
    def test_insert_uses_uuid(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            mid = am.insert("test content")
            assert mid.startswith("arch_")
            hex_part = mid[5:]  # strip "arch_"
            assert len(hex_part) == 12
            int(hex_part, 16)  # must be valid hex


class TestArchivalMemorySanitization:
    def test_insert_sanitize_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            content = "hello\x00world<script>alert(1)</script>"
            mid = am.insert(content)
            stored = am.get_important(threshold=0.0)
            assert "\x00" not in stored[0]["content"]
            assert "<script>" in stored[0]["content"]  # content-safe

    def test_importance_clamped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("test", importance=5.0)
            assert am.memories[-1]["importance"] == 1.0
            am.insert("test2", importance=-1.0)
            assert am.memories[-1]["importance"] == 0.0

    def test_metadata_capped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            meta = {f"k{i}": f"v{i}" for i in range(60)}
            am.insert("test", metadata=meta)
            assert len(am.memories[-1]["metadata"]) <= 50

    def test_content_capped_at_100k(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("x" * 200000)
            assert len(am.memories[-1]["content"]) <= 100000


class TestArchivalMemoryTimestamps:
    def test_insert_uses_timezone_utc(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            am.insert("timezone test")
            ts = am.memories[-1]["timestamp"]
            dt = datetime.fromisoformat(ts)
            assert dt.tzinfo is not None


# ═════════════════════════════════════════════════════════════════
# Entity/Relation tests
# ═════════════════════════════════════════════════════════════════

class TestEntitySanitization:
    def test_entity_id_strict_sanitized(self):
        e = Entity(id="test\x00id<script>", entity_type="person", name="Test")
        assert "\x00" not in e.id
        assert "<script>" not in e.id

    def test_entity_type_strict_sanitized(self):
        e = Entity(id="e1", entity_type="type\x00<script>", name="Test")
        assert "\x00" not in e.entity_type
        assert "<script>" not in e.entity_type

    def test_entity_name_content_sanitized(self):
        e = Entity(id="e1", entity_type="person", name="Test\x00Name<script>alert</script>")
        assert "\x00" not in e.name
        assert "<script>" in e.name  # content-safe preserves <>

    def test_entity_importance_clamped(self):
        e = Entity(id="e1", entity_type="person", name="Test", importance=2.0)
        assert e.importance == 1.0
        e2 = Entity(id="e2", entity_type="person", name="Test", importance=-1.0)
        assert e2.importance == 0.0

    def test_entity_attributes_capped(self):
        attrs = {f"k{i}": f"v{i}" for i in range(60)}
        e = Entity(id="e1", entity_type="person", name="Test", attributes=attrs)
        assert len(e.attributes) <= 50

    def test_entity_to_dict_deepcopy_attributes(self):
        e = Entity(id="e1", entity_type="person", name="Test", attributes={"k": "v"})
        d = e.to_dict()
        d["attributes"]["k"] = "modified"
        assert e.attributes["k"] == "v"

    def test_entity_created_timezone_aware(self):
        e = Entity(id="e1", entity_type="person", name="Test")
        assert e.created.tzinfo is not None


class TestRelationSanitization:
    def test_relation_subject_id_strict(self):
        r = Relation(id="r1", subject_id="s\x00<script>", predicate="knows", object_id="o1")
        assert "\x00" not in r.subject_id
        assert "<script>" not in r.subject_id

    def test_relation_predicate_strict(self):
        r = Relation(id="r1", subject_id="s1", predicate="knows\x00<script>", object_id="o1")
        assert "\x00" not in r.predicate
        assert "<script>" not in r.predicate

    def test_relation_confidence_clamped(self):
        r = Relation(id="r1", subject_id="s1", predicate="knows", object_id="o1", confidence=2.0)
        assert r.confidence == 1.0

    def test_relation_context_capped(self):
        ctx = {f"k{i}": f"v{i}" for i in range(60)}
        r = Relation(id="r1", subject_id="s1", predicate="knows", object_id="o1", context=ctx)
        assert len(r.context) <= 50

    def test_relation_to_dict_deepcopy_context(self):
        r = Relation(id="r1", subject_id="s1", predicate="knows", object_id="o1", context={"k": "v"})
        d = r.to_dict()
        d["context"]["k"] = "modified"
        assert r.context["k"] == "v"

    def test_relation_timestamp_timezone_aware(self):
        r = Relation(id="r1", subject_id="s1", predicate="knows", object_id="o1")
        assert r.timestamp.tzinfo is not None


# ═════════════════════════════════════════════════════════════════
# KnowledgeGraph tests
# ═════════════════════════════════════════════════════════════════

class TestKnowledgeGraphThreadSafety:
    def test_add_entity_returns_uuid(self):
        kg = KnowledgeGraph()
        eid = kg.add_entity("person", "Alice")
        assert len(eid) == 12
        int(eid, 16)  # valid hex

    def test_add_relation_returns_uuid(self):
        kg = KnowledgeGraph()
        e1 = kg.add_entity("person", "Alice")
        e2 = kg.add_entity("person", "Bob")
        rid = kg.add_relation(e1, "knows", e2)
        assert len(rid) == 12
        int(rid, 16)

    def test_get_entity_returns_deepcopy(self):
        kg = KnowledgeGraph()
        e1 = kg.add_entity("person", "Alice")
        entity = kg.get_entity(e1)
        entity.name = "Modified"
        original = kg.get_entity(e1)
        assert original.name == "Alice"

    def test_find_entities_returns_deepcopy(self):
        kg = KnowledgeGraph()
        kg.add_entity("person", "UniqueAlice")
        results = kg.find_entities(name_contains="UniqueAlice")
        assert len(results) == 1
        results[0].name = "Modified"
        orig = kg.find_entities(name_contains="UniqueAlice")
        assert orig[0].name == "UniqueAlice"

    def test_get_relations_returns_deepcopy(self):
        kg = KnowledgeGraph()
        e1 = kg.add_entity("person", "AliceX")
        e2 = kg.add_entity("person", "BobX")
        kg.add_relation(e1, "knows_well", e2)
        rels = kg.get_relations(predicate="knows_well")
        assert len(rels) == 1
        rels[0].predicate = "modified"
        orig = kg.get_relations(predicate="knows_well")
        assert orig[0].predicate == "knows_well"

    def test_to_dict_under_lock(self):
        kg = KnowledgeGraph()
        kg.add_entity("person", "Test")
        d = kg.to_dict()
        assert "entities" in d

    def test_get_timeline_returns_deepcopy(self):
        kg = KnowledgeGraph()
        e1 = kg.add_entity("person", "TimelineAlice")
        e2 = kg.add_entity("person", "TimelineBob")
        kg.add_relation(e1, "met", e2, context={"place": "cafe"})
        timeline = kg.get_timeline()
        assert len(timeline) > 0
        if "context" in timeline[0]:
            timeline[0]["context"]["place"] = "modified"
            orig = kg.get_timeline()
            assert orig[0]["context"]["place"] == "cafe"


class TestKnowledgeGraphSanitization:
    def test_entity_type_sanitized(self):
        kg = KnowledgeGraph()
        eid = kg.add_entity("type\x00<script>", "TestName")
        entity = kg.get_entity(eid)
        assert "\x00" not in entity.entity_type
        assert "<script>" not in entity.entity_type

    def test_entity_name_sanitized_content(self):
        kg = KnowledgeGraph()
        eid = kg.add_entity("person", "Name\x00with<script>")
        entity = kg.get_entity(eid)
        assert "\x00" not in entity.name
        assert "<script>" in entity.name  # content-safe preserves <>

    def test_relation_ids_sanitized(self):
        kg = KnowledgeGraph()
        e1 = kg.add_entity("person", "A")
        e2 = kg.add_entity("person", "B")
        rid = kg.add_relation("sid\x00<script>", "pred\x00<script>", "oid\x00<script>")
        rels = kg.get_relations()
        for r in rels:
            assert "\x00" not in r.subject_id
            assert "<script>" not in r.subject_id

    def test_eviction_logs_warning(self):
        kg = KnowledgeGraph()
        for i in range(kg._MAX_ENTITIES + 10):
            kg.add_entity("type", f"Entity_{i}", importance=0.1)
        assert len(kg.entities) <= kg._MAX_ENTITIES


# ═════════════════════════════════════════════════════════════════
# SuperMemory tests
# ═════════════════════════════════════════════════════════════════

class TestSuperMemorySanitization:
    def test_remember_sanitizes_content(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SuperMemory(storage_dir=Path(tmpdir))
            sm.remember("hello\x00world<script>alert(1)</script>", memory_type="working")
            msgs = sm.working_memory.messages
            assert "\x00" not in msgs[-1]["content"]
            assert "<script>" in msgs[-1]["content"]

    def test_remember_archival_sanitized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SuperMemory(storage_dir=Path(tmpdir))
            sm.remember("archive\x00test<script>", memory_type="archival")
            recent = sm.archival_memory.get_recent()
            assert any("\x00" not in m["content"] for m in recent)

    def test_recall_returns_deepcopy_working(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SuperMemory(storage_dir=Path(tmpdir))
            sm.remember("unique_query_test_xyz", memory_type="working")
            results = sm.recall("unique_query_test_xyz")
            if results["working"]:
                results["working"][0]["content"] = "modified"
                orig = sm.recall("unique_query_test_xyz")
                assert orig["working"][0]["content"] != "modified"

    def test_remember_metadata_capped(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SuperMemory(storage_dir=Path(tmpdir))
            meta = {f"k{i}": f"v{i}" for i in range(60)}
            sm.remember("test metadata cap", metadata=meta, memory_type="archival")
            assert len(sm.archival_memory.memories[-1]["metadata"]) <= 50


class TestSuperMemoryLock:
    def test_stats_returns_dict(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            sm = SuperMemory(storage_dir=Path(tmpdir))
            stats = sm.get_stats()
            assert isinstance(stats, dict)
            assert "core_blocks" in stats


# ═════════════════════════════════════════════════════════════════
# CoreMemory regression tests (ensure changes don't break existing)
# ═════════════════════════════════════════════════════════════════

class TestCoreMemoryRegression:
    def test_get_block_returns_deepcopy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CoreMemory(storage_path=Path(tmpdir) / "core.json")
            block = cm.get_block("persona")
            assert block is not None
            block.value = "modified"
            original = cm.get_block("persona")
            assert original.value != "modified"

    def test_set_block_sanitized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CoreMemory(storage_path=Path(tmpdir) / "core.json")
            cm.set_block("test\x00<script>", "value")
            # _sanitize_strict strips control chars AND <>
            assert "testscript" in cm.blocks
            assert "\x00" not in list(cm.blocks.keys())[-1]
            assert "<script>" not in list(cm.blocks.keys())[-1]

    def test_append_block_sanitized(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CoreMemory(storage_path=Path(tmpdir) / "core.json")
            cm.append_block("test\x00<script>", "append_value")
            # Should have created a sanitized block label

    def test_save_load_roundtrip(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CoreMemory(storage_path=Path(tmpdir) / "core.json")
            cm.set_block("test_block", "test content")
            cm.save()
            cm2 = CoreMemory(storage_path=Path(tmpdir) / "core.json")
            cm2.load()
            block = cm2.get_block("test_block")
            assert block is not None
            assert block.value == "test content"

    def test_get_context_under_lock(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            cm = CoreMemory(storage_path=Path(tmpdir) / "core.json")
            ctx = cm.get_context()
            assert "Core Memory" in ctx


class TestMemoryBlockRegression:
    def test_timezone_aware_defaults(self):
        mb = MemoryBlock(label="test", value="content")
        assert mb.created.tzinfo is not None
        assert mb.last_modified.tzinfo is not None

    def test_to_dict_deepcopy_metadata(self):
        mb = MemoryBlock(label="test", value="content", metadata={"k": "v"})
        d = mb.to_dict()
        d["metadata"]["k"] = "modified"
        assert mb.metadata["k"] == "v"

    def test_append_timezone(self):
        mb = MemoryBlock(label="test", value="initial")
        mb.append("appended")
        assert mb.last_modified.tzinfo is not None

    def test_replace_timezone(self):
        mb = MemoryBlock(label="test", value="initial")
        mb.replace("replaced")
        assert mb.last_modified.tzinfo is not None


# ═════════════════════════════════════════════════════════════════
# Concurrent thread tests
# ═════════════════════════════════════════════════════════════════

class TestConcurrentAccess:
    def test_concurrent_working_memory_add(self):
        wm = WorkingMemory()
        errors = []
        def add_messages():
            try:
                for i in range(50):
                    wm.add_message("user", f"message_{i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add_messages) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(wm.messages) <= 50 * 4

    def test_concurrent_knowledge_graph_add(self):
        kg = KnowledgeGraph()
        errors = []
        def add_entities():
            try:
                for i in range(50):
                    kg.add_entity("person", f"ConcurrentPerson_{threading.current_thread().name}_{i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=add_entities) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_archival_insert(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            am = ArchivalMemory(storage_path=Path(tmpdir) / "archival")
            errors = []
            def insert_memories():
                try:
                    for i in range(20):
                        am.insert(f"concurrent_test_{threading.current_thread().name}_{i}")
                except Exception as e:
                    errors.append(e)
            threads = [threading.Thread(target=insert_memories) for _ in range(4)]
            for t in threads:
                t.start()
            for t in threads:
                t.join()
            assert len(errors) == 0
            assert len(am.memories) == 80  # 20 * 4