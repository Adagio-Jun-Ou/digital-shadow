#!/usr/bin/env python3
"""
ProceduralMemory - Procedural/Habit Learning Engine

Tracks learned procedures, habits, and routines. Estimates mastery through
repetition, tracks habit formation stages (cue-routine-reward), and monitors
consistency of routine execution.

Architecture:
  - Procedure: learned sequence with step-by-step instructions
  - Habit: cue-routine-reward pattern with automaticity tracking
  - Routine: recurring scheduled/unscheduled activity with consistency metrics
  - ProceduralMemory: main engine managing all three
  - Thread-safe, bounded data structures, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.procedural_memory")


class HabitStage(int, Enum):
    PRECONTEMPLATION = 0
    CONTEMPLATION = 1
    PREPARATION = 2
    ACTION = 3
    MAINTENANCE = 4
    AUTOMATIC = 5


_STAGE_NAMES = ["precontemplation", "contemplation", "preparation", "action", "maintenance", "automatic"]

_STAGE_LABELS = {
    HabitStage.PRECONTEMPLATION: "Not yet considering",
    HabitStage.CONTEMPLATION: "Considering change",
    HabitStage.PREPARATION: "Preparing to act",
    HabitStage.ACTION: "Actively building",
    HabitStage.MAINTENANCE: "Sustaining habit",
    HabitStage.AUTOMATIC: "Fully automatic",
}

_REPS_PER_STAGE = [0, 3, 7, 21, 66, 100]


@dataclass
class Procedure:
    procedure_id: str
    name: str
    category: str
    steps: List[str] = field(default_factory=list)
    mastery: float = 0.0
    reps: int = 0
    last_practiced: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def stage_name(self) -> str:
        if self.mastery >= 0.9:
            return "expert"
        elif self.mastery >= 0.7:
            return "proficient"
        elif self.mastery >= 0.4:
            return "practicing"
        else:
            return "learning"

    def to_dict(self) -> Dict:
        return {
            "procedure_id": self.procedure_id,
            "name": self.name,
            "category": self.category,
            "steps": list(self.steps)[:20],
            "mastery": round(self.mastery, 3),
            "stage_name": self.stage_name,
            "reps": self.reps,
            "last_practiced": self.last_practiced,
            "created_at": self.created_at,
        }


@dataclass
class Habit:
    habit_id: str
    name: str
    cue: str
    routine: str
    reward: str
    stage: HabitStage = HabitStage.PRECONTEMPLATION
    automaticity: float = 0.0
    streak: int = 0
    best_streak: int = 0
    completions: List[str] = field(default_factory=list)
    last_completed: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    @property
    def stage_label(self) -> str:
        return _STAGE_LABELS.get(self.stage, "unknown")

    def to_dict(self) -> Dict:
        return {
            "habit_id": self.habit_id,
            "name": self.name[:200],
            "cue": self.cue[:200],
            "routine": self.routine[:200],
            "reward": self.reward[:200],
            "stage": self.stage.value,
            "stage_name": self.stage_name,
            "stage_label": self.stage_label,
            "automaticity": round(self.automaticity, 3),
            "streak": self.streak,
            "best_streak": self.best_streak,
            "completion_count": len(self.completions),
            "completions": list(self.completions[-10:]),
            "last_completed": self.last_completed,
            "created_at": self.created_at,
        }

    @property
    def stage_name(self) -> str:
        if 0 <= self.stage.value < len(_STAGE_NAMES):
            return _STAGE_NAMES[self.stage.value]
        return "unknown"


@dataclass
class Routine:
    routine_id: str
    name: str
    frequency: str = "daily"
    times_completed: int = 0
    consistency: float = 0.0
    last_completed: Optional[str] = None
    completion_log: List[str] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "routine_id": self.routine_id,
            "name": self.name[:200],
            "frequency": self.frequency,
            "times_completed": self.times_completed,
            "consistency": round(self.consistency, 3),
            "last_completed": self.last_completed,
            "completion_count": len(self.completion_log),
            "completion_log": list(self.completion_log[-10:]),
            "created_at": self.created_at,
        }


_MAX_PROCEDURES = 500
_MAX_HABITS = 200
_MAX_ROUTINES = 200
_MAX_NAME_LENGTH = 200
_MAX_STEPS = 20
_MAX_COMPLETIONS = 1000
_VALID_FREQUENCIES = {"daily", "weekly", "biweekly", "monthly", "quarterly", "yearly"}


class ProceduralMemory:
    _MAX_PROCEDURES = _MAX_PROCEDURES
    _MAX_HABITS = _MAX_HABITS
    _MAX_ROUTINES = _MAX_ROUTINES

    def __init__(self):
        self._lock = threading.Lock()
        self._procedures: Dict[str, Procedure] = {}
        self._habits: Dict[str, Habit] = {}
        self._routines: Dict[str, Routine] = {}

    def _sanitize(self, text: str) -> str:
        text = re.sub(r"[<>\x00-\x1f\x7f]", "", text).strip()
        return text[:_MAX_NAME_LENGTH]

    @staticmethod
    def _sanitize_content(text: str) -> str:
        text = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]", "", text).strip()
        return text[:_MAX_NAME_LENGTH]

    def add_procedure(
        self,
        name: str,
        category: str,
        steps: Optional[List[str]] = None,
    ) -> Procedure:
        name = self._sanitize(name)
        steps = [self._sanitize_content(s) for s in (steps or [])[:_MAX_STEPS]]
        category = self._sanitize(category)[:50]

        proc = Procedure(
            procedure_id=uuid.uuid4().hex[:12],
            name=name,
            category=category,
            steps=steps,
        )

        with self._lock:
            if len(self._procedures) >= self._MAX_PROCEDURES:
                oldest_id = next(iter(self._procedures))
                logger.warning("Evicting oldest procedure %s", oldest_id)
                del self._procedures[oldest_id]
            self._procedures[proc.procedure_id] = proc

        return copy.deepcopy(proc)

    def practice_procedure(self, procedure_id: str, success: bool = True) -> Procedure:
        with self._lock:
            proc = self._procedures.get(procedure_id)
            if proc is None:
                raise ValueError(f"Procedure not found: {procedure_id}")

            proc.reps += 1
            proc.last_practiced = datetime.now(timezone.utc).isoformat()

            if success:
                proc.mastery = min(1.0, proc.mastery + 0.05 + 0.02 * min(proc.reps / 10, 1.0))
            else:
                proc.mastery = max(0.0, proc.mastery - 0.02)

            return copy.deepcopy(proc)

    def add_habit(
        self,
        name: str,
        cue: str,
        routine: str,
        reward: str,
    ) -> Habit:
        name = self._sanitize(name)
        cue = self._sanitize_content(cue)
        routine = self._sanitize_content(routine)
        reward = self._sanitize_content(reward)

        habit = Habit(
            habit_id=uuid.uuid4().hex[:12],
            name=name,
            cue=cue,
            routine=routine,
            reward=reward,
        )

        with self._lock:
            if len(self._habits) >= self._MAX_HABITS:
                oldest_id = next(iter(self._habits))
                logger.warning("Evicting oldest habit %s", oldest_id)
                del self._habits[oldest_id]
            self._habits[habit.habit_id] = habit

        return copy.deepcopy(habit)

    def complete_habit(self, habit_id: str) -> Habit:
        with self._lock:
            habit = self._habits.get(habit_id)
            if habit is None:
                raise ValueError(f"Habit not found: {habit_id}")

            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()
            habit.completions.append(now_iso)
            if len(habit.completions) > _MAX_COMPLETIONS:
                habit.completions = habit.completions[-_MAX_COMPLETIONS:]

            if habit.last_completed is not None:
                try:
                    last = datetime.fromisoformat(habit.last_completed)
                    hours_since = (now - last).total_seconds() / 3600
                    if hours_since > 48:
                        habit.streak = 1
                    elif hours_since <= 25:
                        habit.streak += 1
                except (ValueError, TypeError):
                    habit.streak = 1
            else:
                habit.streak = 1

            habit.last_completed = now_iso
            habit.best_streak = max(habit.best_streak, habit.streak)
            habit.automaticity = min(1.0, habit.automaticity + 0.03 + 0.01 * min(habit.streak / 30, 1.0))

            habit.stage = self._calculate_habit_stage(habit)

            return copy.deepcopy(habit)

    def miss_habit(self, habit_id: str) -> Habit:
        with self._lock:
            habit = self._habits.get(habit_id)
            if habit is None:
                raise ValueError(f"Habit not found: {habit_id}")

            habit.streak = 0
            habit.automaticity = max(0.0, habit.automaticity - 0.05)
            habit.stage = self._calculate_habit_stage(habit)

            return copy.deepcopy(habit)

    def _calculate_habit_stage(self, habit: Habit) -> HabitStage:
        reps = len(habit.completions)
        auto = habit.automaticity
        if auto >= 0.95 and reps >= _REPS_PER_STAGE[5]:
            return HabitStage.AUTOMATIC
        elif auto >= 0.8 and reps >= _REPS_PER_STAGE[4]:
            return HabitStage.MAINTENANCE
        elif auto >= 0.5 and reps >= _REPS_PER_STAGE[3]:
            return HabitStage.ACTION
        elif auto >= 0.2 and reps >= _REPS_PER_STAGE[2]:
            return HabitStage.PREPARATION
        elif reps >= _REPS_PER_STAGE[1]:
            return HabitStage.CONTEMPLATION
        else:
            return HabitStage.PRECONTEMPLATION

    def add_routine(
        self,
        name: str,
        frequency: str = "daily",
    ) -> Routine:
        name = self._sanitize(name)
        if frequency not in _VALID_FREQUENCIES:
            frequency = "daily"

        routine = Routine(
            routine_id=uuid.uuid4().hex[:12],
            name=name,
            frequency=frequency,
        )

        with self._lock:
            if len(self._routines) >= self._MAX_ROUTINES:
                oldest_id = next(iter(self._routines))
                logger.warning("Evicting oldest routine %s", oldest_id)
                del self._routines[oldest_id]
            self._routines[routine.routine_id] = routine

        return copy.deepcopy(routine)

    def complete_routine(self, routine_id: str) -> Routine:
        with self._lock:
            routine = self._routines.get(routine_id)
            if routine is None:
                raise ValueError(f"Routine not found: {routine_id}")

            now = datetime.now(timezone.utc)
            now_iso = now.isoformat()

            freq = routine.frequency
            if freq == "daily":
                window_days = 7
                expected = 7
                gap_days = 1
            elif freq == "weekly":
                window_days = 30
                expected = 4
                gap_days = 7
            elif freq == "biweekly":
                window_days = 60
                expected = 4
                gap_days = 14
            elif freq == "monthly":
                window_days = 90
                expected = 3
                gap_days = 30
            elif freq == "quarterly":
                window_days = 365
                expected = 4
                gap_days = 90
            elif freq == "yearly":
                window_days = 365
                expected = 1
                gap_days = 365
            else:
                window_days = 7
                expected = 7
                gap_days = 1

            if routine.last_completed is not None:
                try:
                    last_dt = datetime.fromisoformat(routine.last_completed)
                    gap = (now - last_dt).total_seconds() / 86400
                    missed_periods = max(0, int(gap / gap_days) - 1)
                    if missed_periods > 0:
                        penalty = min(0.5, missed_periods * 0.1)
                        routine.consistency = max(0.0, routine.consistency - penalty)
                except (ValueError, TypeError):
                    pass

            routine.times_completed += 1
            routine.last_completed = now_iso
            routine.completion_log.append(now_iso)
            if len(routine.completion_log) > _MAX_COMPLETIONS:
                routine.completion_log = routine.completion_log[-_MAX_COMPLETIONS:]

            cutoff = (now - timedelta(days=window_days)).isoformat()
            recent = [t for t in routine.completion_log if t >= cutoff]
            window_consistency = min(1.0, len(recent) / max(expected, 1))
            routine.consistency = min(1.0, max(routine.consistency, window_consistency))

            return copy.deepcopy(routine)

    def get_procedure(self, procedure_id: str) -> Optional[Procedure]:
        with self._lock:
            proc = self._procedures.get(procedure_id)
            return copy.deepcopy(proc) if proc else None

    def get_habit(self, habit_id: str) -> Optional[Habit]:
        with self._lock:
            habit = self._habits.get(habit_id)
            return copy.deepcopy(habit) if habit else None

    def get_routine(self, routine_id: str) -> Optional[Routine]:
        with self._lock:
            routine = self._routines.get(routine_id)
            return copy.deepcopy(routine) if routine else None

    def list_procedures(self, category: Optional[str] = None) -> List[Procedure]:
        with self._lock:
            procs = [copy.deepcopy(p) for p in self._procedures.values()]
        if category:
            procs = [p for p in procs if p.category == category]
        return sorted(procs, key=lambda p: p.mastery, reverse=True)

    def list_habits(self, stage: Optional[int] = None) -> List[Habit]:
        with self._lock:
            habits = [copy.deepcopy(h) for h in self._habits.values()]
        if stage is not None:
            habits = [h for h in habits if h.stage.value == stage]
        return sorted(habits, key=lambda h: h.automaticity, reverse=True)

    def list_routines(self) -> List[Routine]:
        with self._lock:
            return [copy.deepcopy(r) for r in self._routines.values()]

    def remove_procedure(self, procedure_id: str) -> bool:
        with self._lock:
            if procedure_id not in self._procedures:
                return False
            del self._procedures[procedure_id]
            return True

    def remove_habit(self, habit_id: str) -> bool:
        with self._lock:
            if habit_id not in self._habits:
                return False
            del self._habits[habit_id]
            return True

    def remove_routine(self, routine_id: str) -> bool:
        with self._lock:
            if routine_id not in self._routines:
                return False
            del self._routines[routine_id]
            return True

    def get_habit_streaks(self) -> Dict[str, int]:
        with self._lock:
            return {hid: h.streak for hid, h in self._habits.items()}

    def get_mastery_progress(self) -> Dict[str, float]:
        with self._lock:
            return {p_id: round(p.mastery, 3) for p_id, p in self._procedures.items()}

    def get_stats(self) -> Dict:
        with self._lock:
            procs = list(self._procedures.values())
            habits = list(self._habits.values())
            routines = list(self._routines.values())
        avg_mastery = sum(p.mastery for p in procs) / max(len(procs), 1) if procs else 0.0
        avg_auto = sum(h.automaticity for h in habits) / max(len(habits), 1) if habits else 0.0
        avg_consistency = sum(r.consistency for r in routines) / max(len(routines), 1) if routines else 0.0
        return {
            "total_procedures": len(procs),
            "total_habits": len(habits),
            "total_routines": len(routines),
            "avg_mastery": round(avg_mastery, 3),
            "avg_automaticity": round(avg_auto, 3),
            "avg_consistency": round(avg_consistency, 3),
            "best_streak": max((h.best_streak for h in habits), default=0),
        }