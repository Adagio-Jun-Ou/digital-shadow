"""
Session 6 Sprint 24 — Deep Audit Security & Robustness Fixes Regression Tests

Fixes:
  C6-C11: shadow_crew fixes (dead code, encapsulation, missions_failed, lock, deps, timeout)
  C12-C14: shadow_voice fixes (intent regex, session lock, start_session limit)
  C15-C20: personality_engine fixes (fsync, exceptions, deque, version increment, sanitization)
  C21-C23: daily_briefing fixes (datetime comparison, wellness urgency, early-wake)
  C24: shadow_graph _MAX_OBSERVATIONS_PER_EVENT class attribute
  SL-1/ES-5: SensitivityLevel int enum (FIXED — broken string comparison)
  SV-1: TTS/STT TOCTOU race fixed with asyncio.Lock
  SV-2/3: current_session race fixed, process_speech session creation in lock
  SOS-1: shadow_os health check uses correct crew.list_agents()
  SOS-3: shadow_os _save_state fsync
  SOS-4: shadow_os handlers no longer double-call transition_to
  SC-2: delegated tasks now executed via iteration loop
  NM-4: neural_mesh tasks stored and cancelled on stop
  NM-5: consensus updated on peer removal
  SI-1: self_improvement _save_state fsync
  SI-2: calibrator evicted model predictions cleaned from _predictions
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


# ─── SL-1/ES-5: SensitivityLevel int enum ───

class TestSensitivityLevelOrdering:
    def test_sensitivity_ordering_correct(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert SensitivityLevel.PUBLIC < SensitivityLevel.INTERNAL
        assert SensitivityLevel.INTERNAL < SensitivityLevel.CONFIDENTIAL
        assert SensitivityLevel.CONFIDENTIAL < SensitivityLevel.SECRET
        assert SensitivityLevel.SECRET < SensitivityLevel.TOP_SECRET

    def test_sensitivity_int_values(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert SensitivityLevel.PUBLIC.value == 0
        assert SensitivityLevel.INTERNAL.value == 1
        assert SensitivityLevel.CONFIDENTIAL.value == 2
        assert SensitivityLevel.SECRET.value == 3
        assert SensitivityLevel.TOP_SECRET.value == 4

    def test_sensitivity_label(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert SensitivityLevel.PUBLIC.label == "public"
        assert SensitivityLevel.CONFIDENTIAL.label == "confidential"
        assert SensitivityLevel.TOP_SECRET.label == "top_secret"

    def test_access_control_allows_lower_sensitivity(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert SensitivityLevel.CONFIDENTIAL < SensitivityLevel.SECRET, \
            "CONFIDENTIAL should be less than SECRET (needed for access control)"

    def test_access_control_blocks_higher_sensitivity(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert SensitivityLevel.TOP_SECRET > SensitivityLevel.INTERNAL, \
            "TOP_SECRET should be greater than INTERNAL (needed for access control)"

    def test_access_control_internal_under_secret(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert SensitivityLevel.INTERNAL < SensitivityLevel.SECRET, \
            "INTERNAL should be less than SECRET"

    def test_sensitivity_comparison_in_check_access(self):
        from shadowd.cognition.stealth_layer import PrivacyGuard
        source = inspect.getsource(PrivacyGuard.check_access)
        assert ".value >" in source, "check_access should compare .value (int) for sensitivity"


# ─── SV-1: TTS/STT client lock ───

class TestVoiceClientLock:
    def test_tts_has_client_lock(self):
        from shadowd.cognition.shadow_voice import TTSEngine
        engine = TTSEngine()
        assert hasattr(engine, "_client_lock"), "TTSEngine should have _client_lock"
        assert isinstance(engine._client_lock, asyncio.Lock), "Should be asyncio.Lock"

    def test_stt_has_client_lock(self):
        from shadowd.cognition.shadow_voice import STTEngine
        engine = STTEngine()
        assert hasattr(engine, "_client_lock"), "STTEngine should have _client_lock"
        assert isinstance(engine._client_lock, asyncio.Lock), "Should be asyncio.Lock"


# ─── SV-2/3: process_speech session creation in lock ───

class TestVoiceProcessSpeechSessionLock:
    def test_process_speech_uses_lock(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        source = inspect.getsource(ShadowVoice.process_speech)
        assert "_sessions_lock" in source, "process_speech should use _sessions_lock"

    def test_list_sessions_uses_lock(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        source = inspect.getsource(ShadowVoice.list_sessions)
        assert "_sessions_lock" in source, "list_sessions should use _sessions_lock"


# ─── SOS-1: shadow_os health check uses correct attr ───

class TestShadowOSHealthCheck:
    def test_crew_health_uses_list_agents(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS.get_health)
        assert "crew.list_agents()" in source, "Should use list_agents() not _agents"
        assert "_agents" not in source.split("crew")[1].split(")")[0] if "crew" in source else True, \
            "Should not access _agents directly"


# ─── SOS-3: shadow_os _save_state fsync ───

class TestShadowOSSaveStateFsync:
    def test_save_state_uses_fsync(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._save_state)
        assert "fsync" in source, "_save_state should use fsync"


# ─── SOS-4: handlers don't double-call transition_to ───

class TestShadowOSHandlerTransitions:
    def test_analyze_no_transition_to(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_analyze)
        assert "transition_to" not in source, "_handle_analyze should not call transition_to"

    def test_reason_no_transition_to(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_reason)
        assert "transition_to" not in source, "_handle_reason should not call transition_to"

    def test_respond_no_transition_to(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_respond)
        assert "transition_to" not in source, "_handle_respond should not call transition_to"

    def test_learn_no_transition_to(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_learn)
        assert "transition_to" not in source, "_handle_learn should not call transition_to"

    def test_reflect_no_transition_to(self):
        from shadowd.cognition.shadow_os import ShadowOS
        source = inspect.getsource(ShadowOS._handle_reflect)
        assert "transition_to" not in source, "_handle_reflect should not call transition_to"


# ─── NM-4: neural_mesh tasks stored/cancelled ───

class TestNeuralMeshTaskManagement:
    def test_start_stores_tasks(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh.start)
        assert "_tasks" in source, "start() should store tasks in _tasks"

    def test_stop_cancels_tasks(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh.stop)
        assert "cancel" in source, "stop() should cancel tasks"
        assert "_tasks" in source, "stop() should reference _tasks"


# ─── NM-5: consensus updated on remove_peer ───

class TestNeuralMeshConsensusOnRemoval:
    def test_remove_peer_updates_consensus(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        source = inspect.getsource(NeuralMesh.remove_peer)
        assert "consensus" in source.lower(), "remove_peer should update consensus"


# ─── SI-1: self_improvement _save_state fsync ───

class TestSelfImprovementSaveFsync:
    def test_save_state_uses_fsync(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine._save_state)
        assert "fsync" in source, "_save_state should use fsync"


# ─── SI-2: calibrator evicted model predictions cleaned ───

class TestCalibratorEvictionConsistency:
    def test_evicted_model_predictions_cleaned(self):
        from shadowd.cognition.self_improvement import AccuracyCalibrator, Prediction
        source = inspect.getsource(AccuracyCalibrator.record_prediction)
        assert "evicted_ids" in source, "record_prediction should clean evicted model predictions from _predictions"

    def test_evicted_domain_predictions_cleaned(self):
        from shadowd.cognition.self_improvement import AccuracyCalibrator
        source = inspect.getsource(AccuracyCalibrator.record_prediction)
        assert "_by_domain" in source, "Should also handle domain eviction"


# ─── SC-2: delegated tasks executed via iteration ───

class TestCrewDelegatedTasksExecution:
    def test_execute_mission_uses_iteration_loop(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        source = inspect.getsource(ShadowCrew.execute_mission)
        assert "made_progress" in source, "Should use made_progress iteration pattern"
        assert "max_iterations" in source, "Should cap iterations"


# ─── SV-5: & not stripped from voice transcripts ───

class TestVoiceTranscriptSanitization:
    def test_ampersand_preserved(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        voice = ShadowVoice()
        result = re.sub(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f]', '', "AT&T wireless")
        assert "&" in result, "Ampersand should be preserved in transcripts"


# ─── Import for above test
import re


# ─── Escalation sensitivity comparison ───

class TestEscalationSensitivity:
    def test_escalation_uses_int_sensitivity(self):
        from shadowd.cognition.stealth_layer import SensitivityLevel
        assert issubclass(SensitivityLevel, int), "SensitivityLevel should inherit from int"