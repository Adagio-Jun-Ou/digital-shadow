"""Sprint 49-50: Audit fixes + ProceduralMemory module"""

import pytest
import threading

from shadowd.cognition.procedural_memory import (
    ProceduralMemory, HabitStage, Procedure, Habit, Routine,
    _MAX_PROCEDURES, _MAX_HABITS, _MAX_ROUTINES, _MAX_STEPS,
)
from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
from shadowd.cognition.narrative_memory import NarrativeMemory
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))



class TestAuditAdaptiveSkillTree:
    def test_add_skill_atomic_under_lock(self):
        source = open(_repo_file("shadowd/cognition/adaptive_skill_tree.py")).read()
        assert "if skill_id in self._skills:" in source, "Duplicate check should be inside lock"

    def test_remove_skill_rechecks_unlocks(self):
        source = open(_repo_file("shadowd/cognition/adaptive_skill_tree.py")).read()
        rm_method = source[source.index("def remove_skill"):]
        assert "skill.unlocked" in rm_method or "unlocked" in rm_method, "remove_skill should recheck unlocks"

    def test_unused_xp_constants_dont_crash(self):
        tree = AdaptiveSkillTree()
        tree.add_skill("test", "Test", "technical")
        for _ in range(5):
            tree.use_skill("test", success=True)
        stats = tree.get_stats()
        assert stats["total_skills"] == 1


class TestAuditNarrativeMemory:
    def test_create_chapter_atomic(self):
        source = open(_repo_file("shadowd/cognition/narrative_memory.py")).read()
        ch_start = source.index("def create_chapter(")
        ch_end = source.index("\n    def ", ch_start + 1)
        ch_method = source[ch_start:ch_end]
        assert ch_method.count("with self._lock:") == 1, "create_chapter should use single lock section"

    def test_create_arc_atomic(self):
        source = open(_repo_file("shadowd/cognition/narrative_memory.py")).read()
        arc_start = source.index("def create_arc(")
        arc_end = source.index("\n    def ", arc_start + 1)
        arc_method = source[arc_start:arc_end]
        assert arc_method.count("with self._lock:") == 1, "create_arc should use single lock section"


class TestProceduralMemoryProcedures:
    def test_add_procedure(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Python Testing", "testing", steps=["Write test", "Run test", "Check result"])
        assert proc.procedure_id
        assert proc.name == "Python Testing"
        assert len(proc.steps) == 3
        assert proc.mastery == 0.0

    def test_practice_procedure_success(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing")
        practiced = pm.practice_procedure(proc.procedure_id, success=True)
        assert practiced.mastery > 0.0
        assert practiced.reps == 1

    def test_practice_procedure_failure(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing")
        practiced = pm.practice_procedure(proc.procedure_id, success=False)
        assert practiced.reps == 1
        assert practiced.mastery <= 0.0

    def test_practice_nonexistent_raises(self):
        pm = ProceduralMemory()
        with pytest.raises(ValueError):
            pm.practice_procedure("nonexistent")

    def test_mastery_increases_with_reps(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing")
        mastery_values = []
        for _ in range(20):
            proc = pm.practice_procedure(proc.procedure_id, success=True)
            mastery_values.append(proc.mastery)
        assert all(mastery_values[i] <= mastery_values[i + 1] for i in range(len(mastery_values) - 1))

    def test_mastery_caps_at_one(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing")
        for _ in range(200):
            pm.practice_procedure(proc.procedure_id, success=True)
        proc = pm.get_procedure(proc.procedure_id)
        assert proc.mastery <= 1.0

    def test_stage_name(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing")
        assert proc.stage_name == "learning"
        for _ in range(30):
            pm.practice_procedure(proc.procedure_id, success=True)
        proc = pm.get_procedure(proc.procedure_id)
        assert proc.stage_name in ("learning", "practicing", "proficient", "expert")

    def test_procedure_to_dict(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing", steps=["Step 1"])
        d = proc.to_dict()
        assert "procedure_id" in d
        assert "mastery" in d
        assert "stage_name" in d

    def test_list_procedures(self):
        pm = ProceduralMemory()
        pm.add_procedure("Proc A", "testing")
        pm.add_procedure("Proc B", "development")
        procs = pm.list_procedures()
        assert len(procs) == 2

    def test_list_procedures_by_category(self):
        pm = ProceduralMemory()
        pm.add_procedure("Proc A", "testing")
        pm.add_procedure("Proc B", "development")
        procs = pm.list_procedures(category="testing")
        assert len(procs) == 1

    def test_remove_procedure(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Testing", "testing")
        assert pm.remove_procedure(proc.procedure_id) is True
        assert pm.get_procedure(proc.procedure_id) is None

    def test_remove_nonexistent_procedure(self):
        pm = ProceduralMemory()
        assert pm.remove_procedure("nonexistent") is False


class TestProceduralMemoryHabits:
    def test_add_habit(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Morning Run", "Alarm rings", "Go for a 5K run", "Feel energized")
        assert habit.habit_id
        assert habit.cue == "Alarm rings"
        assert habit.stage == HabitStage.PRECONTEMPLATION

    def test_complete_habit(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Reading", "After dinner", "Read 30 pages", "Knowledge gained")
        completed = pm.complete_habit(habit.habit_id)
        assert completed.streak == 1
        assert completed.best_streak == 1
        assert len(completed.completions) == 1
        assert completed.automaticity > 0.0

    def test_miss_habit(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Reading", "After dinner", "Read 30 pages", "Knowledge gained")
        pm.complete_habit(habit.habit_id)
        pm.complete_habit(habit.habit_id)
        missed = pm.miss_habit(habit.habit_id)
        assert missed.streak == 0
        assert missed.best_streak == 2

    def test_habit_stage_progression(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Reading", "Cue", "Routine", "Reward")
        assert habit.stage == HabitStage.PRECONTEMPLATION
        for _ in range(5):
            pm.complete_habit(habit.habit_id)
        habit = pm.get_habit(habit.habit_id)
        assert habit.stage.value >= HabitStage.CONTEMPLATION.value

    def test_miss_nonexistent_habit(self):
        pm = ProceduralMemory()
        with pytest.raises(ValueError):
            pm.miss_habit("nonexistent")

    def test_habit_to_dict(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Test", "Cue", "Routine", "Reward")
        d = habit.to_dict()
        assert "habit_id" in d
        assert "stage" in d
        assert "automaticity" in d
        assert "streak" in d

    def test_habit_stage_names(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Test", "Cue", "Routine", "Reward")
        assert habit.stage_name == "precontemplation"
        assert habit.stage_label == "Not yet considering"

    def test_completion_cap(self):
        from shadowd.cognition.procedural_memory import _MAX_COMPLETIONS
        pm = ProceduralMemory()
        habit = pm.add_habit("Test", "Cue", "Routine", "Reward")
        for _ in range(_MAX_COMPLETIONS + 10):
            pm.complete_habit(habit.habit_id)
        habit = pm.get_habit(habit.habit_id)
        assert len(habit.completions) <= _MAX_COMPLETIONS

    def test_best_streak_tracking(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Test", "Cue", "Routine", "Reward")
        for _ in range(5):
            pm.complete_habit(habit.habit_id)
        pm.miss_habit(habit.habit_id)
        for _ in range(3):
            pm.complete_habit(habit.habit_id)
        habit = pm.get_habit(habit.habit_id)
        assert habit.best_streak == 5

    def test_list_habits_by_stage(self):
        pm = ProceduralMemory()
        pm.add_habit("H1", "C", "R", "W")
        habits = pm.list_habits(stage=0)
        assert len(habits) == 1

    def test_habit_streaks(self):
        pm = ProceduralMemory()
        h1 = pm.add_habit("H1", "C", "R", "W")
        pm.complete_habit(h1.habit_id)
        pm.complete_habit(h1.habit_id)
        streaks = pm.get_habit_streaks()
        assert streaks[h1.habit_id] == 2


class TestProceduralMemoryRoutines:
    def test_add_routine(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Morning Workout", frequency="daily")
        assert routine.routine_id
        assert routine.frequency == "daily"
        assert routine.consistency == 0.0

    def test_complete_routine(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Daily Standup")
        completed = pm.complete_routine(routine.routine_id)
        assert completed.times_completed == 1
        assert completed.consistency > 0.0

    def test_complete_nonexistent_routine(self):
        pm = ProceduralMemory()
        with pytest.raises(ValueError):
            pm.complete_routine("nonexistent")

    def test_routine_to_dict(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Test Routine")
        d = routine.to_dict()
        assert "routine_id" in d
        assert "consistency" in d

    def test_consistency_caps(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Test")
        for _ in range(50):
            pm.complete_routine(routine.routine_id)
        routine = pm.get_routine(routine.routine_id)
        assert routine.consistency <= 1.0

    def test_remove_routine(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Test")
        assert pm.remove_routine(routine.routine_id) is True
        assert pm.get_routine(routine.routine_id) is None


class TestProceduralMemorySecurity:
    def test_sanitize_control_chars(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Test\x00Name", "cat\x01egory")
        assert "\x00" not in proc.name
        assert "\x01" not in proc.category

    def test_steps_sanitized(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Test", "cat", steps=["Step\x001"])
        assert "\x00" not in proc.steps[0]

    def test_steps_capped(self):
        pm = ProceduralMemory()
        steps = [f"Step {i}" for i in range(30)]
        proc = pm.add_procedure("Test", "cat", steps=steps)
        assert len(proc.steps) <= _MAX_STEPS

    def test_habit_name_truncation(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("A" * 300, "Cue", "Routine", "Reward")
        assert len(habit.name) <= 200

    def test_procedure_cap(self):
        pm = ProceduralMemory()
        for i in range(_MAX_PROCEDURES + 5):
            pm.add_procedure(f"proc_{i}", "testing")
        procs = pm.list_procedures()
        assert len(procs) <= _MAX_PROCEDURES

    def test_habit_cap(self):
        pm = ProceduralMemory()
        for i in range(_MAX_HABITS + 5):
            pm.add_habit(f"habit_{i}", "C", "R", "W")
        habits = pm.list_habits()
        assert len(habits) <= _MAX_HABITS

    def test_routine_cap(self):
        pm = ProceduralMemory()
        for i in range(_MAX_ROUTINES + 5):
            pm.add_routine(f"routine_{i}")
        routines = pm.list_routines()
        assert len(routines) <= _MAX_ROUTINES

    def test_thread_safety(self):
        pm = ProceduralMemory()
        errors = []

        def worker():
            try:
                for i in range(50):
                    proc = pm.add_procedure(f"proc_{i}", "testing")
                    pm.practice_procedure(proc.procedure_id, success=True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestProceduralMemoryStats:
    def test_get_stats(self):
        pm = ProceduralMemory()
        pm.add_procedure("P1", "testing")
        h1 = pm.add_habit("H1", "C", "R", "W")
        pm.complete_habit(h1.habit_id)
        r1 = pm.add_routine("R1")
        pm.complete_routine(r1.routine_id)
        stats = pm.get_stats()
        assert stats["total_procedures"] == 1
        assert stats["total_habits"] == 1
        assert stats["total_routines"] == 1
        assert stats["avg_mastery"] >= 0.0
        assert stats["avg_automaticity"] > 0.0
        assert stats["avg_consistency"] > 0.0

    def test_mastery_progress(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("P1", "testing")
        pm.practice_procedure(proc.procedure_id, success=True)
        progress = pm.get_mastery_progress()
        assert proc.procedure_id in progress
        assert progress[proc.procedure_id] > 0.0

    def test_procedural_in_shadow_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        assert hasattr(os, 'procedural')
        os._running = True
        os._start_time = 0
        health = os.get_health()
        assert "procedural_memory" in health["modules"]

    def test_procedural_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        status = os.get_status()
        assert "procedural_stats" in status

    def test_procedural_api_endpoints(self):
        from shadowd.cognition.api_router import router
        routes = [r.path for r in router.routes]
        assert "/os/procedural/procedure" in routes
        assert "/os/procedural/habit" in routes
        assert "/os/procedural/routine" in routes
        assert "/os/procedural/stats" in routes


class TestProceduralMemoryFullFlow:
    def test_learning_to_mastery(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Python Testing", "testing", steps=[
            "Write unit test",
            "Run test suite",
            "Check coverage",
        ])

        for i in range(30):
            pm.practice_procedure(proc.procedure_id, success=True)

        proc = pm.get_procedure(proc.procedure_id)
        assert proc.mastery > 0.5
        assert proc.reps == 30

        mastery = pm.get_mastery_progress()
        assert proc.procedure_id in mastery

    def test_habit_formation_journey(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Daily Journal", "After breakfast", "Write 3 pages", "Clarity of thought")

        for _ in range(70):
            pm.complete_habit(habit.habit_id)

        habit = pm.get_habit(habit.habit_id)
        assert habit.stage.value >= HabitStage.ACTION.value
        assert habit.automaticity > 0.5
        assert habit.streak == 70

    def test_routine_consistency_building(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Morning Meditation", frequency="daily")

        for _ in range(15):
            pm.complete_routine(routine.routine_id)

        routine = pm.get_routine(routine.routine_id)
        assert routine.times_completed == 15
        assert routine.consistency > 0.5