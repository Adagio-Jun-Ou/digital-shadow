#!/usr/bin/env python3
"""Sprint 107 Regression Tests — block_memory.py, streaming.py, websocket_manager.py Audit"""

import sys
import os
import copy
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch
from datetime import datetime

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from shadowd.block_memory import (
    MemoryBlock, CoreMemory, ArchivalMemory, WorkingMemory, BlockBasedMemory,
    _sanitize_content, _SANITIZE_STRICT_RE
)


class TestMemoryBlockDeepCopy:
    def test_to_dict_metadata_deepcopy(self):
        mb = MemoryBlock(label="test", value="hello", metadata={"k": [1, 2]})
        d = mb.to_dict()
        d["metadata"]["k"].append(3)
        assert mb.metadata["k"] == [1, 2]

    def test_label_sanitized_on_init(self):
        mb = MemoryBlock(label="my<script>block", value="hello")
        assert "<script>" not in mb.label
        assert "script" in mb.label


class TestCoreMemoryDeepCopy:
    def test_get_block_returns_deepcopy(self):
        cm = CoreMemory()
        block = cm.get_block("human")
        assert block is not None
        block.value = "MODIFIED"
        original = cm.get_block("human")
        assert original.value != "MODIFIED"

    def test_set_block_sanitizes_label(self):
        cm = CoreMemory()
        cm.set_block("test<script>", "hello")
        assert "testscript" in cm.blocks

    def test_search_returns_copies(self):
        cm = CoreMemory()
        cm.set_block("human", "test content")
        results = cm.search("test")
        assert len(results) > 0
        # Modifying result should not affect internal state
        results[0]["value"] = "MODIFIED"
        results2 = cm.search("test")
        assert results2[0]["value"] != "MODIFIED"

    def test_get_all_returns_copy(self):
        cm = CoreMemory()
        labels = cm.get_all()
        labels.append("nonexistent")
        assert "nonexistent" not in cm.get_all()


class TestArchivalMemoryDeepCopy:
    def test_search_returns_deepcopy(self):
        tmpdir = tempfile.mkdtemp()
        am = ArchivalMemory(Path(tmpdir))
        am.insert("hello world")
        results = am.search("hello")
        assert len(results) > 0
        original_content = results[0]["content"]
        results[0]["content"] = "MODIFIED"
        results2 = am.search("hello")
        assert results2[0]["content"] == original_content

    def test_get_all_returns_deepcopy(self):
        tmpdir = tempfile.mkdtemp()
        am = ArchivalMemory(Path(tmpdir))
        am.insert("test")
        all_mem = am.get_all()
        if all_mem:
            all_mem[0]["content"] = "MODIFIED"
            all_mem2 = am.get_all()
            assert all_mem2[0]["content"] == "test"

    def test_max_long_term(self):
        assert ArchivalMemory._MAX_LONG_TERM == 50000


class TestWorkingMemoryDeepCopy:
    def test_get_messages_returns_copy(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        msgs = wm.get_messages()
        msgs.append({"role": "attacker", "content": "injected"})
        assert len(wm.get_messages()) == 1

    def test_to_dict_deepcopy_messages(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        d = wm.to_dict()
        d["messages"].append({"role": "attacker", "content": "injected"})
        assert len(wm.get_messages()) == 1

    def test_get_recent_returns_copy(self):
        wm = WorkingMemory()
        wm.add_message("user", "hello")
        recent = wm.get_recent()
        recent.append({"role": "attacker", "content": "injected"})
        assert len(wm.get_recent()) == 1


class TestBlockBasedMemory:
    def test_sanitize_content_strips_control(self):
        assert _sanitize_content("hello\x00world") == "helloworld"

    def test_sanitize_content_preserves_angle_brackets(self):
        assert _sanitize_content("x < 5") == "x < 5"

    def test_get_stats_returns_dict(self):
        tmpdir = tempfile.mkdtemp()
        bm = BlockBasedMemory(Path(tmpdir))
        stats = bm.get_stats()
        assert "core_blocks" in stats
        assert isinstance(stats, dict)

    def test_archival_insert_and_recall(self):
        tmpdir = tempfile.mkdtemp()
        bm = BlockBasedMemory(Path(tmpdir))
        bm.archival_insert("test observation")
        results = bm.archival_search("test")
        assert len(results) > 0

    def test_working_add_and_get(self):
        tmpdir = tempfile.mkdtemp()
        bm = BlockBasedMemory(Path(tmpdir))
        bm.working_add("user", "hello")
        msgs = bm.working_get()
        assert len(msgs) == 1
        assert msgs[0]["content"] == "hello"


class TestSanitizeRegex:
    def test_strict_regex_strips_angle_brackets(self):
        result = _SANITIZE_STRICT_RE.sub('', "hello<script>alert(1)</script>")
        assert "<" not in result
        assert ">" not in result

    def test_strict_regex_strips_tabs(self):
        result = _SANITIZE_STRICT_RE.sub('', "hello\tworld")
        assert "\t" not in result

    def test_strict_regex_strips_newlines(self):
        result = _SANITIZE_STRICT_RE.sub('', "hello\nworld")
        assert "\n" not in result

    def test_content_regex_preserves_newlines(self):
        from shadowd.block_memory import _SANITIZE_CONTENT_RE
        result = _SANITIZE_CONTENT_RE.sub('', "hello\nworld\ttest")
        assert "\n" in result
        assert "\t" in result

    def test_content_regex_strips_null(self):
        from shadowd.block_memory import _SANITIZE_CONTENT_RE
        result = _SANITIZE_CONTENT_RE.sub('', "hello\x00world")
        assert "\x00" not in result


# ─── Streaming.py Tests ───

from shadowd.streaming import _mask_key


class TestStreamingMaskKey:
    def test_mask_short_key(self):
        assert _mask_key("abc") == "***"

    def test_mask_long_key(self):
        result = _mask_key("sk-1234567890abcdef")
        assert result.startswith("sk-1")
        assert result.endswith("cdef")
        assert "..." in result

    def test_mask_empty_key(self):
        assert _mask_key("") == "***"

    def test_mask_none(self):
        assert _mask_key(None) == "***"


# ─── WebSocket Manager Tests ───

from shadowd.websocket_manager import ConnectionManager, _TOPIC_RE


class TestWebSocketManagerSanitization:
    def test_topic_validation_valid(self):
        assert _TOPIC_RE.match("my-topic_123") is not None

    def test_topic_rejects_special_chars(self):
        assert _TOPIC_RE.match("my<script>topic") is None

    def test_topic_max_length(self):
        long_topic = "a" * 200
        assert _TOPIC_RE.match(long_topic) is None

    def test_topic_min_length(self):
        assert _TOPIC_RE.match("") is None

    def test_connection_manager_init(self):
        cm = ConnectionManager()
        assert cm is not None


# ─── memory_enhanced.py Tests ───

from shadowd.memory_enhanced import EnhancedMemory, Episode, _sanitize, _sanitize_content


class TestEnhancedMemorySanitization:
    def test_episode_type_sanitized(self):
        ep = Episode(id="e1", timestamp=datetime.now(), type="test<script>", content="hello")
        assert ep.type == "testscript"

    def test_episode_content_sanitized(self):
        ep = Episode(id="e1", timestamp=datetime.now(), type="test", content="hello\x00world")
        assert ep.content == "helloworld"

    def test_episode_importance_clamped(self):
        ep = Episode(id="e1", timestamp=datetime.now(), type="test", content="hello", importance=5.0)
        assert ep.importance == 1.0
        ep2 = Episode(id="e2", timestamp=datetime.now(), type="test", content="hello", importance=-1.0)
        assert ep2.importance == 0.0


class TestEnhancedMemoryThreadSafety:
    def test_lock_exists(self):
        em = EnhancedMemory()
        assert hasattr(em, '_lock')

    def test_add_episode_stores_correctly(self):
        em = EnhancedMemory()
        ep = em.add_episode("commitment", "Call John tomorrow")
        assert ep.type == "commitment"
        assert ep.importance == 0.9

    def test_add_to_core_sanitizes_key(self):
        em = EnhancedMemory()
        em.add_to_core("my<script>key", "value")
        assert "myscriptkey" in em.core_memory

    def test_search_returns_deepcopy(self):
        em = EnhancedMemory()
        em.add_episode("observation", "Python programming")
        results = em.search("Python")
        assert len(results) > 0
        results[0].content = "MODIFIED"
        results2 = em.search("Python")
        assert results2[0].content != "MODIFIED"

    def test_get_stats_under_lock(self):
        em = EnhancedMemory()
        em.add_episode("test", "hello")
        stats = em.get_stats()
        assert "total_episodes" in stats

    def test_episode_to_dict_metadata_deepcopy(self):
        ep = Episode(id="e1", timestamp=datetime.now(), type="test", content="hello", metadata={"k": [1, 2]})
        d = ep.to_dict()
        d["metadata"]["k"].append(3)
        assert ep.metadata["k"] == [1, 2]


if __name__ == "__main__":
    import unittest
    unittest.main()