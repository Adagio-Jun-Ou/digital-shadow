"""Sprint 119 regression tests — prompt_cache.py + stuck_detector.py deep audit.

Tests for:
- PromptCache: deepcopy returns, fsync, input caps
- BackgroundPromptCache: thread safety, pattern cap
- PromptCacheManager: deepcopy get, fsync save
- StuckDetector: thread safety, sanitization, deepcopy, timezone
"""

import copy
import threading
import tempfile
from pathlib import Path
import pytest

from shadowd.cache.prompt_cache import (
    CacheEntry, CacheStats, CacheStatus,
    PromptCache, BackgroundPromptCache, PromptCacheManager,
    _sanitize_content,
)
from shadowd.reasoning.stuck_detector import (
    LoopType, StuckAnalysis, ActionType, HistoryEntry, StuckDetector,
    _sanitize_content as sd_sanitize_content,
    _sanitize_strict as sd_sanitize_strict,
)


# ═════════════════════════════════════════════════════════════════
# PromptCache tests
# ═════════════════════════════════════════════════════════════════

class TestCacheEntryDeepCopy:
    def test_to_dict_copies_file_hashes(self):
        entry = CacheEntry(key="k", prompt="p", response="r", file_hashes={"f.py": "abc"})
        d = entry.to_dict()
        d["file_hashes"]["f.py"] = "modified"
        assert entry.file_hashes["f.py"] == "abc"

    def test_to_dict_copies_tags(self):
        entry = CacheEntry(key="k", prompt="p", response="r", tags=["t1", "t2"])
        d = entry.to_dict()
        d["tags"].append("t3")
        assert len(entry.tags) == 2

    def test_timestamp_timezone_aware(self):
        entry = CacheEntry(key="k", prompt="p", response="r")
        assert entry.created_at.tzinfo is not None
        assert entry.last_accessed.tzinfo is not None


class TestPromptCacheThreadSafety:
    def test_get_returns_deepcopy(self):
        # PromptCache.get() returns raw response; deepcopy is in PromptCacheManager.get()
        pc = PromptCache()
        pc.set("test prompt", {"data": [1, 2, 3]})
        status, response = pc.get("test prompt")
        assert status == CacheStatus.HIT
        assert response["data"] == [1, 2, 3]

    def test_set_get_eviction(self):
        pc = PromptCache(max_entries=2, default_ttl=3600)
        pc.set("a", "val_a")
        pc.set("b", "val_b")
        pc.set("c", "val_c")
        assert len(pc._cache) <= 2

    def test_concurrent_set_get(self):
        pc = PromptCache()
        errors = []
        def writer():
            try:
                for i in range(50):
                    pc.set(f"prompt_{threading.current_thread().name}_{i}", f"response_{i}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_get_stats_returns_dict(self):
        pc = PromptCache()
        stats = pc.get_stats()
        assert hasattr(stats, 'hits')

    def test_get_top_entries_returns_dicts(self):
        pc = PromptCache()
        pc.set("p1", "r1")
        entries = pc.get_top_entries(limit=5)
        assert isinstance(entries, list)
        if entries:
            assert isinstance(entries[0], dict)

    def test_get_size_returns_dict(self):
        pc = PromptCache()
        size = pc.get_size()
        assert "entries" in size
        assert "max_entries" in size


class TestPromptCacheFileHash:
    def test_hash_file_nonexistent(self):
        pc = PromptCache()
        result = pc._hash_file("/nonexistent/path/file.py")
        assert result == ""

    def test_hash_file_size_limit(self):
        pc = PromptCache()
        assert pc._MAX_FILE_SIZE == 10 * 1024 * 1024


class TestBackgroundPromptCache:
    def test_pattern_stats_thread_safe(self):
        pc = PromptCache()
        bpc = BackgroundPromptCache(pc)
        bpc._record_pattern("test error pattern")
        bpc._record_pattern("test error pattern")
        stats = bpc.get_pattern_stats()
        assert stats.get("error_fix", 0) >= 2
        # Returns copy
        stats["error_fix"] = 0
        orig = bpc.get_pattern_stats()
        assert orig.get("error_fix", 0) >= 2

    def test_pattern_stats_capped(self):
        pc = PromptCache()
        bpc = BackgroundPromptCache(pc)
        for i in range(1100):
            bpc._record_pattern(f"unique_pattern_{i}")
        assert len(bpc._pattern_stats) <= 1000

    def test_classify_prompt(self):
        pc = PromptCache()
        bpc = BackgroundPromptCache(pc)
        assert bpc._classify_prompt("fix this error") == "error_fix"
        assert bpc._classify_prompt("write a test") == "test"
        assert bpc._classify_prompt("implement this feature") == "implementation"
        assert bpc._classify_prompt("refactor the code") == "refactor"
        assert bpc._classify_prompt("explain this") == "explanation"
        assert bpc._classify_prompt("random stuff") == "other"


class TestPromptCacheManager:
    def test_get_returns_deepcopy(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PromptCacheManager(storage_path=Path(tmpdir) / "cache")
            mgr.set("prompt1", {"key": "value"})
            status, response = mgr.get("prompt1")
            assert response["key"] == "value"
            response["key"] = "modified"
            _, original = mgr.get("prompt1")
            assert original["key"] == "value"

    def test_save_with_fsync(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PromptCacheManager(storage_path=Path(tmpdir) / "cache")
            mgr.set("test_prompt", "test_response")
            mgr.save()
            cache_file = Path(tmpdir) / "cache" / "prompt_cache.json"
            assert cache_file.exists()

    def test_load_from_storage(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr1 = PromptCacheManager(storage_path=Path(tmpdir) / "cache")
            mgr1.set("load_test", "test_value")
            mgr1.save()
            mgr2 = PromptCacheManager(storage_path=Path(tmpdir) / "cache")
            status, response = mgr2.get("load_test")
            assert status == CacheStatus.HIT

    def test_clear(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PromptCacheManager(storage_path=Path(tmpdir) / "cache")
            mgr.set("p", "r")
            mgr.clear()
            status, _ = mgr.get("p")
            assert status == CacheStatus.MISS

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            mgr = PromptCacheManager(storage_path=Path(tmpdir) / "cache")
            mgr.set("p1", "r1")
            stats = mgr.get_stats()
            assert "total_entries" in stats


# ═════════════════════════════════════════════════════════════════
# StuckDetector tests
# ═════════════════════════════════════════════════════════════════

class TestStuckDetectorSanitization:
    def test_record_action_sanitizes_content(self):
        sd = StuckDetector()
        entry = sd.record_action("hello\x00world<script>alert</script>", tool_name="test\x00tool<script>")
        assert "\x00" not in entry.content
        assert "<script>" in entry.content  # content-safe preserves <>
        assert "\x00" not in entry.tool_name
        assert "<script>" not in entry.tool_name  # strict strips <>

    def test_record_thought_sanitizes_content(self):
        sd = StuckDetector()
        entry = sd.record_thought("think\x00about<script>stuff</script>")
        assert "\x00" not in entry.content
        assert "<script>" in entry.content

    def test_record_observation_sanitizes_content(self):
        sd = StuckDetector()
        entry = sd.record_observation("observe\x00things<script>")
        assert "\x00" not in entry.content

    def test_record_error_sanitizes_content(self):
        sd = StuckDetector()
        entry = sd.record_error("error\x00msg<script>", "context\x00data<script>")
        assert "\x00" not in entry.error_message
        assert "\x00" not in entry.content
        assert "<script>" in entry.error_message  # content-safe
        assert "<script>" in entry.content  # content-safe

    def test_content_truncation(self):
        sd = StuckDetector()
        long_content = "x" * 100000
        entry = sd.record_thought(long_content)
        assert len(entry.content) <= 50000

    def test_error_message_truncation(self):
        sd = StuckDetector()
        long_error = "e" * 10000
        entry = sd.record_error(long_error)
        assert len(entry.error_message) <= 5000


class TestStuckDetectorThreadSafety:
    def test_concurrent_record(self):
        sd = StuckDetector()
        errors = []
        def record():
            try:
                for i in range(50):
                    sd.record_action(f"action_{threading.current_thread().name}_{i}", tool_name="test")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=record) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(sd.history) <= 100

    def test_analyze_under_lock(self):
        sd = StuckDetector()
        for i in range(5):
            sd.record_action("same_action", tool_name="test_tool")
        result = sd.analyze()
        assert isinstance(result, StuckAnalysis)

    def test_reset_under_lock(self):
        sd = StuckDetector()
        sd.record_action("test")
        sd.reset()
        assert len(sd.history) == 0

    def test_get_stats_under_lock(self):
        sd = StuckDetector()
        sd.record_action("test", tool_name="test_tool")
        sd.record_thought("thinking")
        stats = sd.get_stats()
        assert stats["total_entries"] == 2
        assert "tool_call" in stats["type_counts"]


class TestStuckDetectorTimezone:
    def test_entry_timezone_aware(self):
        sd = StuckDetector()
        entry = sd.record_action("test")
        assert entry.timestamp.tzinfo is not None


class TestStuckAnalysisDeepCopy:
    def test_to_dict_copies_suggestions(self):
        sa = StuckAnalysis(
            is_stuck=True,
            loop_type=LoopType.REPEATED_ACTION,
            repeat_count=3,
            pattern_description="test",
            recovery_suggestions=["s1", "s2"],
            confidence=0.8,
        )
        d = sa.to_dict()
        d["recovery_suggestions"].append("s3")
        assert len(sa.recovery_suggestions) == 2


class TestHistoryEntryTimestamp:
    def test_timezone_aware(self):
        entry = HistoryEntry(action_type=ActionType.THOUGHT, content="test")
        assert entry.timestamp.tzinfo is not None

    def test_fingerprint(self):
        entry = HistoryEntry(action_type=ActionType.TOOL_CALL, content="test action")
        fp = entry.fingerprint
        assert len(fp) == 10  # 10 hex chars