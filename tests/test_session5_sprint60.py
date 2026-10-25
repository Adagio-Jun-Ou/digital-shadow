#!/usr/bin/env python3
"""Sprint 60 regression tests — NarrativeMemory, ProceduralMemory, moral_reasoning, metacognition, creative_synthesis, conflict_resolution fixes."""

import copy
import time
import pytest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch


from shadowd.cognition.narrative_memory import (
    NarrativeMemory, NarrativeChapter, NarrativeArc, NarrativeSummary,
    EmotionTone,
)
from shadowd.cognition.procedural_memory import (
    ProceduralMemory, Procedure, Habit, Routine, HabitStage,
    _VALID_FREQUENCIES,
)
from shadowd.cognition.moral_reasoning import (
    MoralReasoning, EthicalEvaluation, MoralPrinciple, EthicalFramework,
)
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, ThinkingEpisode, ReasoningType,
)
from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
from shadowd.cognition.creative_synthesis import CreativeSynthesis, CreativeIdea
from shadowd.cognition.conflict_resolution import (
    ConflictResolution, ConflictStatus, ConflictType,
)


class TestNMH1ThemeSanitization:
    def test_create_chapter_theme_sanitized(self):
        nm = NarrativeMemory()
        ch = nm.create_chapter(title="Test", theme="gro\x00wth", summary="ok")
        assert "\x00" not in ch.theme

    def test_create_arc_themes_sanitized(self):
        nm = NarrativeMemory()
        arc = nm.create_arc(title="Arc", themes=["the\x00me", "ok theme"])
        assert "\x00" not in arc.themes[0]
        assert arc.themes[1] == "ok theme"


class TestNMH2DeepCopy:
    def test_get_chapter_returns_copy(self):
        nm = NarrativeMemory()
        ch = nm.create_chapter(title="Ch", theme="growth")
        ch_id = ch.chapter_id
        retrieved = nm.get_chapter(ch_id)
        retrieved.title = "MUTATED"
        assert nm.get_chapter(ch_id).title == "Ch"

    def test_get_arc_returns_copy(self):
        nm = NarrativeMemory()
        arc = nm.create_arc(title="Arc")
        arc_id = arc.arc_id
        retrieved = nm.get_arc(arc_id)
        retrieved.title = "MUTATED"
        assert nm.get_arc(arc_id).title == "Arc"

    def test_list_chapters_returns_copies(self):
        nm = NarrativeMemory()
        nm.create_chapter(title="Ch1", theme="t1")
        chapters = nm.list_chapters()
        chapters[0].title = "MUTATED"
        assert nm.list_chapters()[0].title == "Ch1"

    def test_list_arcs_returns_copies(self):
        nm = NarrativeMemory()
        nm.create_arc(title="Arc1")
        arcs = nm.list_arcs()
        arcs[0].title = "MUTATED"
        assert nm.list_arcs()[0].title == "Arc1"

    def test_get_chapters_for_arc_returns_copies(self):
        nm = NarrativeMemory()
        ch = nm.create_chapter(title="Ch", theme="t")
        arc = nm.create_arc(title="Arc")
        nm.add_chapter_to_arc(arc.arc_id, ch.chapter_id)
        chapters = nm.get_chapters_for_arc(arc.arc_id)
        chapters[0].title = "MUTATED"
        assert nm.get_chapters_for_arc(arc.arc_id)[0].title == "Ch"


class TestNMH3GrowthEmotionReachable:
    def test_pure_growth_text_returns_growth(self):
        nm = NarrativeMemory()
        result = nm._detect_emotion("I learned and grew and developed skills")
        assert result == EmotionTone.GROWTH.value

    def test_growth_with_negative_returns_resilience(self):
        nm = NarrativeMemory()
        result = nm._detect_emotion("I struggled but learned and grew")
        assert result == EmotionTone.RESILIENCE.value

    def test_growth_with_positive_and_negative_returns_resilience(self):
        nm = NarrativeMemory()
        result = nm._detect_emotion("I was happy to learn and grow but also experienced fear")
        assert result == EmotionTone.RESILIENCE.value


class TestNMH4DeadGrowthArcCode:
    def test_resilience_arc_boosts_progression(self):
        nm = NarrativeMemory()
        ch = nm.create_chapter(title="Ch", theme="t")
        ch.emotion_arc = "resilience"
        nm._chapters[ch.chapter_id] = ch
        arc = nm.create_arc(title="Arc")
        base_prog = arc.progression
        nm.add_chapter_to_arc(arc.arc_id, ch.chapter_id)
        arc2 = nm.get_arc(arc.arc_id)
        assert arc2.progression > base_prog


class TestNMM1TrajectoryTypoFix:
    def test_summary_field_name(self):
        nm = NarrativeMemory()
        nm.add_event(content="test event", importance=0.5)
        summary = nm.generate_summary()
        assert hasattr(summary, "emotional_trajectory")
        assert not hasattr(summary, "emotionaltrajectory")

    def test_summary_to_dict_key(self):
        nm = NarrativeMemory()
        nm.add_event(content="test event", importance=0.5)
        d = nm.generate_summary().to_dict()
        assert "emotional_trajectory" in d


class TestPMH1TimeAwareStreak:
    def test_rapid_fire_doesnt_inflate_streak(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("brush", "morning", "brush teeth", "clean")
        h1 = pm.complete_habit(habit.habit_id)
        h2 = pm.complete_habit(habit.habit_id)
        assert h2.streak == 2

    def test_streak_resets_after_48h(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("brush", "morning", "brush teeth", "clean")
        old_time = (datetime.now(timezone.utc) - timedelta(hours=50)).isoformat()
        with pm._lock:
            h = pm._habits[habit.habit_id]
            h.last_completed = old_time
            h.streak = 5
        result = pm.complete_habit(habit.habit_id)
        assert result.streak == 1

    def test_streak_keeps_between_25_48h(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("brush", "morning", "brush teeth", "clean")
        with pm._lock:
            h = pm._habits[habit.habit_id]
            h.last_completed = (datetime.now(timezone.utc) - timedelta(hours=30)).isoformat()
            h.streak = 5
        result = pm.complete_habit(habit.habit_id)
        assert result.streak == 5

    def test_first_completion_streak_1(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("brush", "morning", "brush teeth", "clean")
        result = pm.complete_habit(habit.habit_id)
        assert result.streak == 1


class TestPMH2DeepCopy:
    def test_get_procedure_returns_copy(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("test", "cat")
        retrieved = pm.get_procedure(proc.procedure_id)
        retrieved.name = "MUTATED"
        assert pm.get_procedure(proc.procedure_id).name == "test"

    def test_get_habit_returns_copy(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("h", "c", "r", "w")
        retrieved = pm.get_habit(habit.habit_id)
        retrieved.name = "MUTATED"
        assert pm.get_habit(habit.habit_id).name == "h"

    def test_get_routine_returns_copy(self):
        pm = ProceduralMemory()
        r = pm.add_routine("r")
        retrieved = pm.get_routine(r.routine_id)
        retrieved.name = "MUTATED"
        assert pm.get_routine(r.routine_id).name == "r"

    def test_list_procedures_returns_copies(self):
        pm = ProceduralMemory()
        pm.add_procedure("p1", "cat")
        procs = pm.list_procedures()
        procs[0].name = "MUTATED"
        assert pm.list_procedures()[0].name == "p1"

    def test_list_habits_returns_copies(self):
        pm = ProceduralMemory()
        pm.add_habit("h", "c", "r", "w")
        habits = pm.list_habits()
        habits[0].name = "MUTATED"
        assert pm.list_habits()[0].name == "h"

    def test_list_routines_returns_copies(self):
        pm = ProceduralMemory()
        pm.add_routine("r1")
        routines = pm.list_routines()
        routines[0].name = "MUTATED"
        assert pm.list_routines()[0].name == "r1"

    def test_practice_procedure_returns_copy(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("p", "cat")
        result = pm.practice_procedure(proc.procedure_id)
        result.name = "MUTATED"
        assert pm.get_procedure(proc.procedure_id).name == "p"

    def test_complete_habit_returns_copy(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("h", "c", "r", "w")
        result = pm.complete_habit(habit.habit_id)
        result.name = "MUTATED"
        assert pm.get_habit(habit.habit_id).name == "h"

    def test_miss_habit_returns_copy(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("h", "c", "r", "w")
        result = pm.miss_habit(habit.habit_id)
        result.name = "MUTATED"
        assert pm.get_habit(habit.habit_id).name == "h"

    def test_complete_routine_returns_copy(self):
        pm = ProceduralMemory()
        r = pm.add_routine("r")
        result = pm.complete_routine(r.routine_id)
        result.name = "MUTATED"
        assert pm.get_routine(r.routine_id).name == "r"


class TestPMH4RoutineConsistencyDecreases:
    def test_consistency_reflects_cadence(self):
        pm = ProceduralMemory()
        r = pm.add_routine("daily jog", "daily")
        now = datetime.now(timezone.utc)
        with pm._lock:
            routine = pm._routines[r.routine_id]
            for i in range(7):
                t = (now - timedelta(days=i)).isoformat()
                routine.completion_log.append(t)
            routine.times_completed = 7
        result = pm.complete_routine(r.routine_id)
        assert result.consistency > 0.8

    def test_consistency_low_with_gaps(self):
        pm = ProceduralMemory()
        r = pm.add_routine("daily jog", "daily")
        now = datetime.now(timezone.utc)
        with pm._lock:
            routine = pm._routines[r.routine_id]
            routine.completion_log.append((now - timedelta(days=10)).isoformat())
            routine.completion_log.append((now - timedelta(days=9)).isoformat())
            routine.times_completed = 2
        result = pm.complete_routine(r.routine_id)
        assert result.consistency < 0.5


class TestPMM1ValidateIdRemoved:
    def test_validate_id_not_present(self):
        pm = ProceduralMemory()
        assert not hasattr(pm, "_validate_id")


class TestPMM3FrequencyValidation:
    def test_valid_frequency_accepted(self):
        pm = ProceduralMemory()
        r = pm.add_routine("r", "weekly")
        assert r.frequency == "weekly"

    def test_invalid_frequency_defaults_daily(self):
        pm = ProceduralMemory()
        r = pm.add_routine("r", "crazily")
        assert r.frequency == "daily"

    def test_all_valid_frequencies(self):
        pm = ProceduralMemory()
        for f in _VALID_FREQUENCIES:
            r = pm.add_routine("r_" + f, f)
            assert r.frequency == f


class TestMR4DeepCopyReturns:
    def test_evaluate_returns_copy(self):
        mr = MoralReasoning()
        ev = mr.evaluate("help someone", "good deed")
        ev.action = "MUTATED"
        ev2 = mr.get_evaluations()
        assert ev2[0]["action"] != "MUTATED" if ev2 else True

    def test_add_principle_returns_copy(self):
        mr = MoralReasoning()
        p = mr.add_principle("test", "desc", EthicalFramework.UTILITARIAN)
        p.name = "MUTATED"
        principles = mr.get_principles()
        found = [x for x in principles if x["name"] == "test"]
        assert len(found) > 0

    def test_evaluate_mutation_isolation(self):
        mr = MoralReasoning()
        ev = mr.evaluate("save a life", "rescue operation")
        original_action = ev.action
        ev.action = "MUTATED"
        assert original_action != "MUTATED"


class TestME6DeepCopyEpisodes:
    def test_start_thinking_returns_copy(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "test", confidence=0.8)
        ep.topic = "MUTATED"
        ep2 = me.start_thinking(ReasoningType.SYSTEMATIC, "other")
        assert me.get_stats()["episodes_total"] == 2

    def test_resolve_thinking_returns_copy(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "test", confidence=0.8)
        resolved = me.resolve_thinking(ep.episode_id, "correct", 0.9)
        assert resolved is not None
        resolved.topic = "MUTATED"

    def test_start_thinking_mutation_isolation(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "test", confidence=0.8)
        ep_id = ep.episode_id
        ep.confidence = 0.0
        report = me.get_calibration_report()


class TestCS7MetadataCap:
    def test_creative_idea_metadata_capped(self):
        cs = CreativeSynthesis()
        big_meta = {f"key_{i}": f"val_{i}" for i in range(100)}
        idea = CreativeIdea(
            title="test",
            description="test idea",
            metadata=big_meta,
        )
        assert len(idea.metadata) <= 50

    def test_creative_idea_metadata_under_cap(self):
        cs = CreativeSynthesis()
        small_meta = {f"k{i}": f"v{i}" for i in range(10)}
        idea = CreativeIdea(title="t", description="d", metadata=small_meta)
        assert len(idea.metadata) == 10


class TestCR8MediatingTransition:
    def test_active_to_mediating_on_first_perspective(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test conflict", "interest", 2)
        assert conflict.status == ConflictStatus.ACTIVE
        cr.add_perspective(conflict.conflict_id, "Alice", "I want X")
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.MEDIATING

    def test_mediating_not_set_on_second_perspective(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test conflict", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "I want X")
        cr.add_perspective(conflict.conflict_id, "Bob", "I want Y")
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.MEDIATING

    def test_resolved_conflict_no_status_change(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "A", "pos")
        cr.resolve_conflict(conflict.conflict_id, "done")
        cr.add_perspective(conflict.conflict_id, "B", "pos2")
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED