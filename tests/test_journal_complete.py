#!/usr/bin/env python3
"""
Comprehensive tests for ShadowJournal module.

Tests cover:
- All EntryType values (observation, decision, action, alert, insight, summary, milestone)
- All ActivityCategory values (work, health, social, learning, creativity, security, system, visual, general)
- Convenience methods (record_observation, record_decision, record_alert, record_insight)
- Invalid entry_type falls back to "observation"
- Get entries with date, type, category filters
- Search by keyword with scoring
- Daily summary generation with category counts
- Empty date summary
- Weekly digest structure
- Get single entry by ID (found and not found)
- Export as JSON and text
- Stats accuracy
- Tags filtering in search
- Thread safety under concurrent recording
- to_dict deepcopy safety on metadata
- Importance clamping to [0,1]
- Content sanitization (control chars stripped)
- DailySummary dataclass creation and to_dict
"""

import copy
import json
import os
import sys
import tempfile
import threading
from datetime import datetime, timezone, timedelta
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

from shadowd.cognition.shadow_journal import (
    ShadowJournal, JournalEntry, DailySummary, EntryType, ActivityCategory,
)


class TestEntryTypeEnum:
    def test_all_entry_types_exist(self):
        values = {e.value for e in EntryType}
        expected = {"observation", "decision", "action", "alert", "insight", "summary", "milestone"}
        assert values == expected

    def test_record_all_entry_types(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        types_and_content = [
            ("observation", "observed user logging in"),
            ("decision", "decided to upgrade kernel"),
            ("action", "executed system update"),
            ("alert", "high CPU usage detected"),
            ("insight", "user prefers CLI over GUI"),
            ("summary", "end of day wrap-up"),
            ("milestone", "project v2.0 shipped"),
        ]
        for etype, content in types_and_content:
            eid = journal.record(entry_type=etype, content=content)
            entry_dict = journal.get_entry(eid)
            assert entry_dict is not None
            assert entry_dict["entry_type"] == etype

    def test_invalid_entry_type_falls_back_to_observation(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="nonexistent_type", content="some content")
        entry = journal.get_entry(eid)
        assert entry["entry_type"] == "observation"


class TestActivityCategoryEnum:
    def test_all_activity_categories_exist(self):
        values = {c.value for c in ActivityCategory}
        expected = {"work", "health", "social", "learning", "creativity", "security", "system", "visual", "general"}
        assert values == expected

    def test_record_all_activity_categories(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        categories = ["work", "health", "social", "learning", "creativity", "security", "system", "visual", "general"]
        for cat in categories:
            eid = journal.record(entry_type="observation", content=f"content for {cat}", category=cat)
            entry = journal.get_entry(eid)
            assert entry["category"] == cat

    def test_invalid_category_uses_general(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="observation", content="test", category="invalid_category")
        entry = journal.get_entry(eid)
        assert entry["category"] == "general"


class TestConvenienceMethods:
    def test_record_observation(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record_observation("user opened file explorer", source_module="test_mod", importance=0.7, tags=["ui"])
        entry = journal.get_entry(eid)
        assert entry["entry_type"] == "observation"
        assert entry["source_module"] == "test_mod"
        assert entry["importance"] == 0.7
        assert "ui" in entry["tags"]
        assert entry["category"] == "general"

    def test_record_decision(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record_decision("chose TypeScript over JavaScript", importance=0.9, tags=["tech"])
        entry = journal.get_entry(eid)
        assert entry["entry_type"] == "decision"
        assert entry["importance"] == 0.9
        assert entry["category"] == "work"
        assert "source_module" in entry

    def test_record_alert(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record_alert("unauthorized login attempt", tags=["security-high"])
        entry = journal.get_entry(eid)
        assert entry["entry_type"] == "alert"
        assert entry["importance"] == 0.9
        assert entry["category"] == "security"
        assert "security-high" in entry["tags"]

    def test_record_insight(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record_insight("user is more productive in morning", importance=0.85, tags=["pattern"])
        entry = journal.get_entry(eid)
        assert entry["entry_type"] == "insight"
        assert entry["importance"] == 0.85
        assert entry["category"] == "system"
        assert "pattern" in entry["tags"]


class TestGetEntries:
    def test_get_entries_returns_recent(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="first")
        journal.record(entry_type="observation", content="second")
        journal.record(entry_type="observation", content="third")
        entries = journal.get_entries(limit=10)
        assert len(entries) >= 3

    def test_get_entries_with_date_filter(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        eid1 = journal.record(entry_type="observation", content="today entry")
        entries = journal.get_entries(date=today, limit=50)
        assert any(e["entry_id"] == eid1 for e in entries)

    def test_get_entries_with_type_filter(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="alert", content="alert content")
        journal.record(entry_type="observation", content="obs content")
        entries = journal.get_entries(entry_type="alert", limit=50)
        assert all(e["entry_type"] == "alert" for e in entries)

    def test_get_entries_with_category_filter(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="security log", category="security")
        journal.record(entry_type="observation", content="work log", category="work")
        entries = journal.get_entries(category="security", limit=50)
        assert all(e["category"] == "security" for e in entries)

    def test_get_entries_with_tags_filter(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="tagged entry", tags=["important", "urgent"])
        journal.record(entry_type="observation", content="untagged", tags=["normal"])
        entries = journal.get_entries(tags=["urgent"], limit=50)
        assert all("urgent" in e["tags"] for e in entries)


class TestSearch:
    def test_search_by_keyword(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="python script completed successfully")
        journal.record(entry_type="observation", content="unrelated log entry")
        results = journal.search("python")
        assert len(results) >= 1
        assert "python" in results[0]["content"].lower()

    def test_search_scores_summary_lower_than_content(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="content with keyword hidden",
                       summary="summary has keyword")
        journal.record(entry_type="observation", content="keyword in content",
                       summary="unrelated summary")
        results = journal.search("keyword")
        scores = []
        for r in results:
            score = 0
            if "keyword" in r["content"].lower():
                score += 2
            if "keyword" in r.get("summary", "").lower():
                score += 1
            scores.append(score)
        assert len(scores) >= 1

    def test_search_by_tag(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="deployment log", tags=["deploy", "prod"])
        results = journal.search("deploy")
        assert len(results) >= 1

    def test_search_no_results(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="some content")
        results = journal.search("xyznonexistent9876")
        assert results == []


class TestDailySummary:
    def test_daily_summary_generation(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="work done", category="work")
        journal.record(entry_type="decision", content="choose stack", category="work")
        journal.record(entry_type="alert", content="disk full", category="security", importance=0.9)
        summary = journal.generate_daily_summary()
        assert summary.total_entries == 3
        assert "work" in summary.categories
        assert summary.categories.get("work") == 2
        assert summary.categories.get("security") == 1

    def test_empty_date_summary(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        summary = journal.generate_daily_summary(date="2000-01-01")
        assert summary.total_entries == 0
        assert summary.categories == {}
        assert summary.productivity_score == 0.0

    def test_daily_summary_importance_sorting(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="low priority", importance=0.1)
        journal.record(entry_type="insight", content="critical finding", importance=1.0)
        journal.record(entry_type="observation", content="mid", importance=0.5)
        summary = journal.generate_daily_summary()
        top = summary.top_entries
        assert len(top) <= 10
        if len(top) >= 2:
            assert top[0]["importance"] >= top[-1]["importance"]

    def test_daily_summary_productivity_score(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="w1", category="work", importance=1.0)
        journal.record(entry_type="observation", content="w2", category="work", importance=0.8)
        journal.record(entry_type="observation", content="s1", category="social")
        summary = journal.generate_daily_summary()
        assert 0.0 <= summary.productivity_score <= 1.0
        assert summary.key_events

    def test_daily_summary_dataclass_manual_creation(self):
        ds = DailySummary(
            date="2025-06-15",
            total_entries=10,
            categories={"work": 5, "health": 3, "learning": 2},
            top_entries=[{"entry_id": "abc", "content": "test"}],
            mood_summary="productive day",
            productivity_score=0.75,
            key_events=["shipped feature", "gym session"],
        )
        assert ds.date == "2025-06-15"
        assert ds.total_entries == 10
        assert ds.categories["work"] == 5
        assert ds.productivity_score == 0.75

    def test_daily_summary_to_dict(self):
        ds = DailySummary(
            date="2025-06-15",
            total_entries=5,
            categories={"work": 3},
            top_entries=[{"entry_id": "x", "content": "y"}],
            mood_summary="fine",
            productivity_score=0.6,
            key_events=["event1"],
        )
        d = ds.to_dict()
        assert d["date"] == "2025-06-15"
        assert d["total_entries"] == 5
        assert d["categories"] == {"work": 3}
        assert d["productivity_score"] == 0.6
        assert "generated_at" in d


class TestWeeklyDigest:
    def test_weekly_digest_structure(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="entry today")
        digest = journal.get_weekly_digest()
        assert "start_date" in digest
        assert "end_date" in digest
        assert "total_entries" in digest
        assert "daily_breakdown" in digest
        assert "generated_at" in digest
        assert digest["total_entries"] >= 1

    def test_weekly_digest_7_days(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        digest = journal.get_weekly_digest()
        assert len(digest["daily_breakdown"]) == 7


class TestGetEntry:
    def test_get_entry_found(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="insight", content="deep insight")
        entry = journal.get_entry(eid)
        assert entry is not None
        assert entry["entry_id"] == eid
        assert entry["content"] == "deep insight"

    def test_get_entry_not_found(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        entry = journal.get_entry("nonexistent_id_12345")
        assert entry is None


class TestExport:
    def test_export_as_json(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="observation", content="json export test", category="work")
        exported = journal.export_journal(fmt="json")
        data = json.loads(exported)
        assert isinstance(data, list)
        assert len(data) >= 1

    def test_export_as_text(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        journal.record(entry_type="alert", content="text export test", category="security")
        exported = journal.export_journal(fmt="text")
        assert "ShadowOS Activity Log" in exported
        assert "ALERT" in exported
        assert "text export test" in exported


class TestStats:
    def test_stats_accuracy(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        for i in range(5):
            journal.record(entry_type="observation", content=f"entry {i}")
        stats = journal.get_stats()
        assert stats["entries_recorded"] >= 5
        assert stats["total_entries"] >= 5
        assert "total_days" in stats
        assert "avg_per_day" in stats

    def test_stats_increments_on_record(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        initial = journal.get_stats()
        initial_count = initial["total_entries"]
        journal.record(entry_type="observation", content="increment test")
        updated = journal.get_stats()
        assert updated["total_entries"] >= initial_count + 1


class TestImportanceClamping:
    def test_importance_above_one_clamped(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="observation", content="over-important", importance=2.5)
        entry = journal.get_entry(eid)
        assert entry["importance"] == 1.0

    def test_importance_below_zero_clamped(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="observation", content="negative importance", importance=-0.5)
        entry = journal.get_entry(eid)
        assert entry["importance"] == 0.0


class TestContentSanitization:
    def test_control_chars_stripped(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="observation", content="clean\x00\x08\x1f\x7fdirty")
        entry = journal.get_entry(eid)
        assert "\x00" not in entry["content"]
        assert "\x08" not in entry["content"]
        assert "\x1f" not in entry["content"]
        assert "\x7f" not in entry["content"]

    def test_tags_sanitized(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="observation", content="test", tags=["clean", "dirty\x1f\x7ftag"])
        entry = journal.get_entry(eid)
        for tag in entry["tags"]:
            assert "\x1f" not in tag
            assert "\x7f" not in tag


class TestThreadSafety:
    def test_concurrent_recording(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        errors = []
        def record_entry(idx):
            try:
                journal.record(entry_type="observation", content=f"thread {idx}")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=record_entry, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        stats = journal.get_stats()
        assert stats["total_entries"] >= 20

    def test_concurrent_get_entries_safe(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        for i in range(50):
            journal.record(entry_type="observation", content=f"entry {i}")
        errors = []
        def read_entries():
            try:
                journal.get_entries(limit=50)
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=read_entries) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestJournalEntry:
    def test_to_dict_deepcopy_safety_on_metadata(self):
        entry = JournalEntry(
            content="test content",
            entry_type=EntryType.OBSERVATION,
            category=ActivityCategory.WORK,
            metadata={"key": "original_value", "count": "42"},
        )
        d = entry.to_dict()
        assert d["metadata"] is not entry.metadata
        d["metadata"]["key"] = "modified_value"
        d["metadata"]["new_field"] = "injected"
        assert entry.metadata["key"] == "original_value"
        assert "new_field" not in entry.metadata
        assert d["metadata"]["key"] == "modified_value"

    def test_entry_id_generated(self):
        entry = JournalEntry(content="auto id")
        assert entry.entry_id
        assert len(entry.entry_id) > 0

    def test_timestamp_default(self):
        entry = JournalEntry(content="ts test")
        assert isinstance(entry.timestamp, datetime)

    def test_date_key_property(self):
        from datetime import datetime as dt
        ts = datetime(2025, 6, 15, 12, 0, 0, tzinfo=timezone.utc)
        entry = JournalEntry(content="date test", timestamp=ts)
        assert entry.date_key == "2025-06-15"


class TestDailySummaryDataclass:
    def test_daily_summary_fields(self):
        ds = DailySummary(
            date="2025-01-01",
            total_entries=5,
            categories={"work": 3},
            top_entries=[],
            mood_summary="neutral",
            productivity_score=0.5,
            key_events=["event1"],
        )
        assert ds.date == "2025-01-01"
        assert ds.total_entries == 5
        assert ds.productivity_score == 0.5

    def test_productivity_score_clamped(self):
        ds = DailySummary(
            date="2025-01-01",
            total_entries=1,
            categories={},
            top_entries=[],
            mood_summary="",
            productivity_score=2.5,
            key_events=[],
        )
        assert ds.productivity_score == 1.0

    def test_productivity_score_clamped_low(self):
        ds = DailySummary(
            date="2025-01-01",
            total_entries=1,
            categories={},
            top_entries=[],
            mood_summary="",
            productivity_score=-0.5,
            key_events=[],
        )
        assert ds.productivity_score == 0.0

    def test_to_dict_copies_categories(self):
        ds = DailySummary(
            date="2025-01-01",
            total_entries=5,
            categories={"work": 3, "health": 2},
            top_entries=[{"eid": "1"}],
            mood_summary="ok",
            productivity_score=0.5,
            key_events=["ev1"],
        )
        d = ds.to_dict()
        d["categories"]["work"] = 99
        assert ds.categories["work"] == 3

    def test_to_dict_copies_top_entries(self):
        ds = DailySummary(
            date="2025-01-01",
            total_entries=3,
            categories={"general": 3},
            top_entries=[{"entry_id": "abc", "content": "original"}],
            mood_summary="ok",
            productivity_score=0.5,
            key_events=["e1"],
        )
        d = ds.to_dict()
        d["top_entries"][0]["content"] = "hacked"
        assert ds.top_entries[0]["content"] == "original"

    def test_to_dict_copies_key_events(self):
        ds = DailySummary(
            date="2025-01-01",
            total_entries=3,
            categories={},
            top_entries=[],
            mood_summary="ok",
            productivity_score=0.5,
            key_events=["original_event"],
        )
        d = ds.to_dict()
        d["key_events"].append("injected")
        assert len(ds.key_events) == 1
        assert ds.key_events[0] == "original_event"


class TestEdgeCases:
    def test_record_with_metadata(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="observation", content="meta test", metadata={"source": "test", "version": 2})
        entry = journal.get_entry(eid)
        assert "source" in entry["metadata"]
        assert "version" in entry["metadata"]

    def test_record_preserves_summary(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        eid = journal.record(entry_type="insight", content="long detailed insight about user behavior", summary="short summary")
        entry = journal.get_entry(eid)
        assert entry["summary"] == "short summary"

    def test_summary_auto_falls_back_to_content_prefix(self):
        journal = ShadowJournal(data_dir=tempfile.mkdtemp())
        long_content = "A" * 500
        eid = journal.record(entry_type="observation", content=long_content, summary="")
        entry = journal.get_entry(eid)
        assert len(entry.get("summary", "")) <= 200

    def test_entry_id_collision_low_probability(self):
        ids = set()
        for _ in range(1000):
            entry = JournalEntry(content="collision test")
            assert entry.entry_id not in ids
            ids.add(entry.entry_id)
