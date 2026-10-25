"""Sprint 47: NarrativeMemory — Life story construction engine"""

import pytest
import threading

from shadowd.cognition.narrative_memory import (
    NarrativeMemory, EmotionTone, ThemeCategory,
    NarrativeEvent, NarrativeChapter, NarrativeArc, NarrativeSummary,
    _MAX_CHAPTERS, _MAX_ARCS, _MAX_EVENTS_PER_CHAPTER,
)


class TestNarrativeEvent:
    def test_add_event(self):
        nm = NarrativeMemory()
        evt = nm.add_event("I got promoted at work today!", importance=0.8)
        assert evt.event_id
        assert evt.emotion == "positive"
        assert "career" in evt.themes
        assert evt.importance == 0.8

    def test_add_negative_event(self):
        nm = NarrativeMemory()
        evt = nm.add_event("Failed the exam and felt sad about it")
        assert evt.emotion == "negative"
        assert "adversity" in evt.themes or "learning" in evt.themes

    def test_add_mixed_event(self):
        nm = NarrativeMemory()
        evt = nm.add_event("Struggled with the project but learned and grew from it")
        assert evt.emotion in ("growth", "resilience", "mixed")

    def test_add_neutral_event(self):
        nm = NarrativeMemory()
        evt = nm.add_event("Went to the store to buy groceries")
        assert evt.emotion == "neutral"

    def test_event_to_dict(self):
        from shadowd.cognition.narrative_memory import NarrativeEvent
        evt = NarrativeEvent(
            event_id="abc123", content="Test event", timestamp="2026-01-01T00:00:00",
            themes=["career"], emotion="positive", importance=0.7
        )
        d = evt.to_dict()
        assert d["event_id"] == "abc123"
        assert d["emotion"] == "positive"
        assert len(d["content"]) <= 200

    def test_importance_clamped(self):
        nm = NarrativeMemory()
        evt = nm.add_event("test", importance=2.0)
        assert evt.importance == 1.0
        evt2 = nm.add_event("test2", importance=-1.0)
        assert evt2.importance == 0.0

    def test_content_sanitized(self):
        nm = NarrativeMemory()
        evt = nm.add_event("Test\x00event\x01here")
        assert "\x00" not in evt.content
        assert "\x01" not in evt.content


class TestNarrativeChapter:
    def test_create_chapter(self):
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Career Growth", "career", "My journey")
        assert chapter.chapter_id
        assert chapter.theme == "career"
        assert chapter.title == "Career Growth"

    def test_add_event_to_chapter(self):
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Career", "career")
        evt = nm.add_event("Got promoted!", importance=0.9)
        result = nm.add_event_to_chapter(chapter.chapter_id, evt)
        assert result is True
        updated = nm.get_chapter(chapter.chapter_id)
        assert len(updated.events) == 1

    def test_add_event_to_nonexistent_chapter(self):
        nm = NarrativeMemory()
        evt = nm.add_event("test")
        result = nm.add_event_to_chapter("nonexistent", evt)
        assert result is False

    def test_chapter_emotion_arc(self):
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Growth Story", "growth")
        nm.add_event_to_chapter(chapter.chapter_id, nm.add_event("Learned something great!"))
        nm.add_event_to_chapter(chapter.chapter_id, nm.add_event("Achieved my goal!"))
        nm.add_event_to_chapter(chapter.chapter_id, nm.add_event("More success"))
        updated = nm.get_chapter(chapter.chapter_id)
        assert updated.emotion_arc in ("ascending", "steady", "mixed")

    def test_chapter_to_dict(self):
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Test Chapter", "learning", "Summary")
        d = chapter.to_dict()
        assert d["chapter_id"]
        assert d["theme"] == "learning"

    def test_chapter_event_cap(self):
        from shadowd.cognition.narrative_memory import _MAX_EVENTS_PER_CHAPTER
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Big Chapter", "career")
        for i in range(_MAX_EVENTS_PER_CHAPTER + 5):
            evt = nm.add_event(f"Event {i}")
            nm.add_event_to_chapter(chapter.chapter_id, evt)
        updated = nm.get_chapter(chapter.chapter_id)
        assert len(updated.events) <= _MAX_EVENTS_PER_CHAPTER


class TestNarrativeArc:
    def test_create_arc(self):
        nm = NarrativeMemory()
        arc = nm.create_arc("Career Journey", themes=["career", "growth"])
        assert arc.arc_id
        assert len(arc.themes) == 2

    def test_add_chapter_to_arc(self):
        nm = NarrativeMemory()
        chapter = nm.create_chapter("Starting Out", "career")
        arc = nm.create_arc("Career Arc")
        result = nm.add_chapter_to_arc(arc.arc_id, chapter.chapter_id)
        assert result is True
        fresh_arc = nm._arcs[arc.arc_id]
        assert chapter.chapter_id in fresh_arc.chapters

    def test_resolve_arc(self):
        nm = NarrativeMemory()
        arc = nm.create_arc("Test Arc")
        result = nm.resolve_arc(arc.arc_id, "Successfully completed the journey")
        assert result is True
        fresh_arc = nm._arcs[arc.arc_id]
        assert fresh_arc.resolution == "Successfully completed the journey"
        assert fresh_arc.progression == 1.0

    def test_arc_to_dict(self):
        nm = NarrativeMemory()
        arc = nm.create_arc("Test Arc")
        d = arc.to_dict()
        assert d["arc_id"]
        assert d["chapter_count"] == 0

    def test_arc_cap(self):
        from shadowd.cognition.narrative_memory import _MAX_ARCS
        nm = NarrativeMemory()
        for i in range(_MAX_ARCS + 5):
            nm.create_arc(f"Arc {i}")
        arcs = nm.list_arcs()
        assert len(arcs) <= _MAX_ARCS


class TestNarrativeThemes:
    def test_dominant_themes(self):
        nm = NarrativeMemory()
        nm.add_event("Got promoted at work!")
        nm.add_event("Started a new career project")
        nm.add_event("Went for a run and felt great")
        themes = nm.get_dominant_themes(limit=3)
        assert len(themes) > 0
        theme_names = [t for t, _ in themes]
        assert "career" in theme_names

    def test_theme_detection(self):
        nm = NarrativeMemory()
        evt = nm.add_event("I achieved my fitness goal and feel proud!")
        assert "achievement" in evt.themes or "health" in evt.themes

    def test_multiple_themes(self):
        nm = NarrativeMemory()
        evt = nm.add_event("Learned a new skill at work and grew as a professional")
        assert len(evt.themes) >= 1


class TestNarrativeSummary:
    def test_generate_summary(self):
        nm = NarrativeMemory()
        for i in range(10):
            nm.add_event(f"Event {i}", importance=i / 10)
        nm.create_chapter("Test", "career")
        summary = nm.generate_summary()
        assert summary.total_events == 10
        assert summary.total_chapters == 1
        assert len(summary.dominant_themes) >= 0

    def test_summary_to_dict(self):
        from shadowd.cognition.narrative_memory import NarrativeSummary
        s = NarrativeSummary(
            summary_id="abc", total_chapters=1, total_events=10,
            dominant_themes=["career"], emotional_trajectory="ascending",
            arcs=["arc1"], key_events=["e1"], timestamp="2026-01-01"
        )
        d = s.to_dict()
        assert d["total_chapters"] == 1


class TestNarrativeSearch:
    def test_search_events(self):
        nm = NarrativeMemory()
        nm.add_event("Started a new career project at work")
        nm.add_event("Went for a morning run")
        nm.add_event("Met with my career mentor")
        results = nm.search_events("career", limit=5)
        assert len(results) >= 2

    def test_search_events_empty(self):
        nm = NarrativeMemory()
        results = nm.search_events("nonexistent")
        assert results == []


class TestNarrativeSecurity:
    def test_sanitize_control_chars(self):
        nm = NarrativeMemory()
        evt = nm.add_event("Test\x00\x01content")
        assert "\x00" not in evt.content

    def test_chapter_title_truncation(self):
        from shadowd.cognition.narrative_memory import _MAX_EVENT_LENGTH
        nm = NarrativeMemory()
        title = "A" * 300
        chapter = nm.create_chapter(title, "career")
        assert len(chapter.title) <= 200

    def test_arc_title_truncation(self):
        nm = NarrativeMemory()
        title = "B" * 300
        arc = nm.create_arc(title)
        assert len(arc.title) <= 200

    def test_thread_safety(self):
        nm = NarrativeMemory()
        errors = []

        def worker():
            try:
                for i in range(25):
                    nm.add_event(f"Event from thread {i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        stats = nm.get_stats()
        assert stats["total_events"] == 100


class TestNarrativeStats:
    def test_get_stats(self):
        nm = NarrativeMemory()
        nm.add_event("Test event 1")
        nm.add_event("Test event 2")
        nm.create_chapter("Chapter 1", "career")
        stats = nm.get_stats()
        assert stats["total_events"] == 2
        assert stats["total_chapters"] == 1
        assert stats["total_arcs"] == 0

    def test_list_chapters(self):
        nm = NarrativeMemory()
        nm.create_chapter("Career", "career")
        nm.create_chapter("Health", "health")
        chapters = nm.list_chapters()
        assert len(chapters) == 2

    def test_list_chapters_by_theme(self):
        nm = NarrativeMemory()
        nm.create_chapter("Career 1", "career")
        nm.create_chapter("Health 1", "health")
        nm.create_chapter("Career 2", "career")
        career = nm.list_chapters(theme="career")
        assert len(career) == 2

    def test_list_arcs(self):
        nm = NarrativeMemory()
        nm.create_arc("Arc 1")
        nm.create_arc("Arc 2")
        arcs = nm.list_arcs()
        assert len(arcs) == 2

    def test_get_chapters_for_arc(self):
        nm = NarrativeMemory()
        ch1 = nm.create_chapter("Ch1", "career")
        ch2 = nm.create_chapter("Ch2", "career")
        arc = nm.create_arc("Arc")
        nm.add_chapter_to_arc(arc.arc_id, ch1.chapter_id)
        nm.add_chapter_to_arc(arc.arc_id, ch2.chapter_id)
        chapters = nm.get_chapters_for_arc(arc.arc_id)
        assert len(chapters) == 2

    def test_get_chapters_for_nonexistent_arc(self):
        nm = NarrativeMemory()
        chapters = nm.get_chapters_for_arc("nonexistent")
        assert chapters == []


class TestNarrativeFullFlow:
    def test_complete_story_flow(self):
        nm = NarrativeMemory()
        ch1 = nm.create_chapter("Early Career", "career", "Starting out")
        evt1 = nm.add_event("Got my first job!", importance=0.9)
        nm.add_event_to_chapter(ch1.chapter_id, evt1)
        evt2 = nm.add_event("Learned Python programming", importance=0.7)
        nm.add_event_to_chapter(ch1.chapter_id, evt2)

        ch2 = nm.create_chapter("Growth", "growth", "Developing skills")
        evt3 = nm.add_event("Promoted to senior developer", importance=0.95)
        nm.add_event_to_chapter(ch2.chapter_id, evt3)

        arc = nm.create_arc("Career Arc", themes=["career", "growth"])
        nm.add_chapter_to_arc(arc.arc_id, ch1.chapter_id)
        nm.add_chapter_to_arc(arc.arc_id, ch2.chapter_id)

        nm.resolve_arc(arc.arc_id, "Successfully built a career through learning")

        summary = nm.generate_summary()
        assert summary.total_events == 3
        assert summary.total_chapters == 2
        assert len(summary.arcs) == 1
        assert len(summary.dominant_themes) > 0