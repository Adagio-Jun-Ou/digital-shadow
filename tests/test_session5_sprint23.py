"""
Session 6 Sprint 23 — Deep Cognitive Module Audit (Second Pass) Regression Tests

Fixes:
  C6: shadow_crew dead code `if False` → proper dependency chain
  C7: shadow_crew `_find_available_agent` bypasses encapsulation → uses registry methods
  C8: shadow_crew `missions_failed` double-counting per task → only count at mission level
  C9: shadow_crew mission dict thread safety → `threading.Lock`
  C10: shadow_crew dependency validation → `_resolve_dependencies` drops invalid deps
  C11: shadow_crew per-task timeout → task.metadata["timeout"] override
  C12: shadow_voice intent matching `startswith` too broad → regex word boundary
  C13: shadow_voice session cleanup dict mutation → threading.Lock
  C14: shadow_voice `start_session` enforces max_sessions
  C15: personality_engine `_save_profile` missing fsync → atomic write with fsync
  C16: personality_engine `_load_profile` exception narrowing → catches specific types
  C17: personality_engine `PersonaDistiller._behavior_signals` → deque(maxlen=5000)
  C18: personality_engine `_signal_timestamps` rate limit → deque with O(1) cleanup
  C19: personality_engine `distill()` version only increments on actual changes
  C20: personality_engine `record_behavior` sanitizes details values to 200 chars
  C21: daily_briefing overnight timestamp comparison → `datetime.fromisoformat` not string
  C22: daily_briefing wellness score penalizes urgency not item count
  C23: daily_briefing `_briefing_schedule_loop` early-wake via `asyncio.Event`
  C24: shadow_graph `_MAX_OBSERVATIONS_PER_EVENT` placed as class attribute
"""

import asyncio
import inspect
import json
import os
import threading
import time
import uuid
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ─── C6: shadow_crew hierarchical planning uses proper dependency chain ───

class TestCrewHierarchicalDeps:
    @pytest.mark.asyncio
    async def test_hierarchical_observe_task_dependency(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, DelegationMode
        crew = ShadowCrew()
        mission = await crew.create_mission("Test objective", mode=DelegationMode.HIERARCHICAL)
        assert len(mission.tasks) >= 2
        observe_task = mission.tasks[0]
        analyze_task = mission.tasks[1]
        assert observe_task.assigned_agent == "shadow_eye"
        assert analyze_task.assigned_agent == "shadow_mind"
        assert observe_task.task_id in analyze_task.dependencies, \
            "Analyze task should depend on observe task"


# ─── C7: _find_available_agent uses registry methods, not _agents ───

class TestCrewFindAgentEncapsulation:
    def test_find_available_agent_uses_registry(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CrewTask
        crew = ShadowCrew()
        task = CrewTask(description="observe", assigned_agent="shadow_hands")
        result = crew._find_available_agent("shadow_hands", task)
        assert result is not None, "Should find another agent"
        assert result.agent_id != "shadow_hands", "Should exclude specified agent"


# ─── C8: missions_failed not double-counted per task ───

class TestCrewMissionsFailedCount:
    def test_no_per_task_failure_incr(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        assert source.count('"missions_failed"') == 0, \
            "missions_failed should not be incremented per task failure"

    def test_failure_counted_only_at_mission_level(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        mission_level_checks = source.count("MissionStatus.FAILED")
        assert mission_level_checks >= 1, "Should have mission-level failure status"


# ─── C9: mission dict thread safety ───

class TestCrewThreadSafety:
    def test_missions_lock_exists(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        crew = ShadowCrew()
        assert hasattr(crew, "_missions_lock"), "Should have _missions_lock"
        assert isinstance(crew._missions_lock, type(threading.Lock())), \
            "Should be threading.Lock"

    def test_create_mission_uses_lock(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.create_mission)
        assert "_missions_lock" in source, "Should use _missions_lock"


# ─── C10: dependency validation ───

class TestCrewDependencyValidation:
    def test_resolve_dependencies_drops_invalid(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CrewTask
        crew = ShadowCrew()
        task = CrewTask(description="test", dependencies=["nonexistent_id"])
        resolved = crew._resolve_dependencies([task])
        assert resolved[0].dependencies == [], "Invalid dep should be dropped"

    def test_resolve_dependencies_keeps_valid(self):
        from shadowd.cognition.shadow_crew import ShadowCrew, CrewTask
        crew = ShadowCrew()
        task1 = CrewTask(description="first")
        task2 = CrewTask(description="second", dependencies=[task1.task_id])
        resolved = crew._resolve_dependencies([task1, task2])
        assert task1.task_id in resolved[1].dependencies, "Valid dep should be kept"


# ─── C11: per-task timeout ───

class TestCrewPerTaskTimeout:
    def test_per_task_timeout_metadata(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        assert "task.metadata.get" in source, "Should use task metadata for timeout"
        assert '"timeout"' in source, "Should check 'timeout' key in metadata"


# ─── C12: shadow_voice intent matching uses word boundaries ───

class TestVoiceIntentMatching:
    def test_intent_parser_does_not_match_substring(self):
        from shadowd.cognition.shadow_voice import IntentParser, VoiceIntent
        parser = IntentParser()
        intent, conf = parser.parse("whatever you say")
        assert intent != VoiceIntent.QUERY, \
            "'whatever' should not match QUERY intent (what)"

    def test_intent_parser_matches_word_boundary(self):
        from shadowd.cognition.shadow_voice import IntentParser, VoiceIntent
        parser = IntentParser()
        intent, conf = parser.parse("what is the weather")
        assert intent == VoiceIntent.QUERY, \
            "'what' at start should match QUERY"

    def test_intent_parser_matches_mid_sentence(self):
        from shadowd.cognition.shadow_voice import IntentParser, VoiceIntent
        parser = IntentParser()
        intent, conf = parser.parse("I need help with this")
        assert intent == VoiceIntent.EMERGENCY, \
            "'help' should match EMERGENCY"

    def test_intent_parser_uses_regex(self):
        from shadowd.cognition.shadow_voice import IntentParser
        source = inspect.getsource(IntentParser)
        assert "re.search" in source or "re.match" in source, \
            "Should use regex matching"


# ─── C13: shadow_voice session cleanup uses threading lock ───

class TestVoiceSessionLock:
    def test_sessions_lock_exists(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        voice = ShadowVoice()
        assert hasattr(voice, "_sessions_lock"), "Should have _sessions_lock"
        assert isinstance(voice._sessions_lock, type(threading.Lock())), \
            "Should be threading.Lock"

    def test_cleanup_uses_lock(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        source = inspect.getsource(ShadowVoice._cleanup_expired_sessions)
        assert "_sessions" in source, "Cleanup should access sessions dict"

        start_source = inspect.getsource(ShadowVoice.start_session)
        assert "_sessions_lock" in start_source, "start_session should hold lock when calling cleanup"


# ─── C14: start_session enforces max_sessions ───

class TestVoiceStartSessionEnforcesLimit:
    def test_start_session_checks_max(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        source = inspect.getsource(ShadowVoice._cleanup_expired_sessions)
        assert "_max_sessions" in source, "Cleanup should enforce max_sessions"


# ─── C15: personality_engine _save_profile uses fsync ───

class TestPersonalitySaveFsync:
    def test_save_profile_uses_fsync(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._save_profile)
        assert "fsync" in source, "Should use fsync for atomic write"


# ─── C16: _load_profile narrows exception types ───

class TestPersonalityLoadExceptionNarrowing:
    def test_load_catches_specific_exceptions(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._load_profile)
        assert "json.JSONDecodeError" in source, "Should catch JSONDecodeError"
        assert "KeyError" in source, "Should catch KeyError"
        assert "TypeError" in source, "Should catch TypeError"


# ─── C17: PersonaDistiller uses deque ───

class TestPersonaDistillerDeque:
    def test_behavior_signals_is_deque(self):
        from shadowd.cognition.personality_engine import PersonaDistiller
        from collections import deque
        d = PersonaDistiller()
        assert isinstance(d._behavior_signals, deque), "Should be deque"
        assert d._behavior_signals.maxlen == 5000, "Should have maxlen"

    def test_signal_timestamps_is_deque(self):
        from shadowd.cognition.personality_engine import PersonaDistiller
        from collections import deque
        d = PersonaDistiller()
        assert isinstance(d._signal_timestamps, deque), "Should be deque"


# ─── C18: signal timestamps rate limit ───

class TestPersonaDistillerRateLimit:
    def test_rate_limit_with_deque(self):
        from shadowd.cognition.personality_engine import PersonaDistiller
        source = inspect.getsource(PersonaDistiller.record_behavior)
        assert "deque" in source, "Should reconstruct deque for rate limit cleanup"


# ─── C19: distill version only increments on changes ───

class TestPersonalityVersionIncrement:
    @pytest.mark.asyncio
    async def test_version_no_increment_without_changes(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        engine = PersonalityEngine(data_dir=f"/tmp/test_personality_v_{uuid.uuid4().hex[:8]}")
        initial_version = engine.profile.version
        result = await engine.distill()
        assert result["trait_updates"] == 0, "No trait updates expected"
        assert result["style_updates"] == 0, "No style updates expected"
        assert engine.profile.version == initial_version, \
            "Version should not increment without changes"

    @pytest.mark.asyncio
    async def test_version_increments_on_trait_changes(self):
        from shadowd.cognition.personality_engine import PersonalityEngine, TraitDimension
        engine = PersonalityEngine(data_dir=f"/tmp/test_personality_v2_{uuid.uuid4().hex[:8]}")
        engine.record_behavior("question_asked", {"complexity": "complex"})
        initial_version = engine.profile.version
        result = await engine.distill()
        assert engine.profile.version == initial_version + 1, \
            "Version should increment with trait changes"


# ─── C20: record_behavior sanitizes details ───

class TestPersonaBehaviorSanitization:
    def test_details_values_truncated(self):
        from shadowd.cognition.personality_engine import PersonaDistiller
        d = PersonaDistiller()
        long_value = "x" * 500
        d.record_behavior("test", {"key": long_value})
        signal = list(d._behavior_signals)[-1]
        assert len(signal["details"]["key"]) == 200, \
            "Details values should be truncated to 200 chars"


# ─── C21: daily_briefing uses datetime comparison ───

class TestBriefingDatetimeComparison:
    def test_overnight_uses_datetime(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        source = inspect.getsource(DailyBriefingEngine._add_overnight_events)
        assert "fromisoformat" in source, "Should use datetime.fromisoformat"
        assert ".isoformat()" not in source or "overnight_cutoff" not in source.split("fromisoformat")[0], \
            "Should not compare ISO strings directly"

    def test_overnight_uses_class_constant(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        assert hasattr(DailyBriefingEngine, "_OVERNIGHT_HOURS"), "Should have _OVERNIGHT_HOURS"
        assert DailyBriefingEngine._OVERNIGHT_HOURS == 8
        assert hasattr(DailyBriefingEngine, "_RECENT_MEMORY_LIMIT"), "Should have _RECENT_MEMORY_LIMIT"


# ─── C22: wellness score penalizes urgency not count ───

class TestBriefingWellnessUrgency:
    def test_wellness_penalizes_medium(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine, DailyBriefing, BriefingItem, BriefingSection, UrgencyLevel
        engine = DailyBriefingEngine()
        briefing = DailyBriefing()
        briefing.items.append(BriefingItem(
            section=BriefingSection.TODAY,
            title="test",
            content="test",
            urgency=UrgencyLevel.MEDIUM,
        ))
        score = engine._calculate_wellness(briefing)
        assert score < 0.7, "Medium urgency should lower wellness"

    def test_wellness_does_not_penalize_info(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine, DailyBriefing, BriefingItem, BriefingSection, UrgencyLevel
        engine = DailyBriefingEngine()
        briefing = DailyBriefing()
        for _ in range(10):
            briefing.items.append(BriefingItem(
                section=BriefingSection.TODAY,
                title="test",
                content="test",
                urgency=UrgencyLevel.INFO,
            ))
        score = engine._calculate_wellness(briefing)
        assert score == 0.7, "INFO items should not affect wellness"


# ─── C23: briefing schedule loop uses asyncio.Event for early wake ───

class TestBriefingEarlyWake:
    def test_wake_event_exists(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        engine = DailyBriefingEngine()
        assert hasattr(engine, "_wake_event"), "Should have _wake_event"
        assert isinstance(engine._wake_event, asyncio.Event), \
            "Should be asyncio.Event"

    def test_schedule_loop_uses_event(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        source = inspect.getsource(DailyBriefingEngine._briefing_schedule_loop)
        assert "_wake_event" in source, "Should use _wake_event"

    def test_trigger_briefing_method(self):
        from shadowd.cognition.daily_briefing import DailyBriefingEngine
        engine = DailyBriefingEngine()
        assert hasattr(engine, "trigger_briefing"), "Should have trigger_briefing method"
        engine.trigger_briefing()
        assert engine._wake_event.is_set(), "Should set wake event"


# ─── C24: shadow_graph _MAX_OBSERVATIONS_PER_EVENT is class attribute ───

class TestShadowGraphMaxObservations:
    def test_max_observations_is_class_attr(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph()
        assert hasattr(g, "_MAX_OBSERVATIONS_PER_EVENT"), "Should have _MAX_OBSERVATIONS_PER_EVENT"
        assert g._MAX_OBSERVATIONS_PER_EVENT == 50

    def test_max_observations_not_module_level(self):
        import shadowd.cognition.shadow_graph as mod
        assert not hasattr(mod, "_MAX_OBSERVATIONS_PER_EVENT"), \
            "_MAX_OBSERVATIONS_PER_EVENT should not be module-level"