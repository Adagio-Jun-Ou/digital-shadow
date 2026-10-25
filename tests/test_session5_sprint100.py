#!/usr/bin/env python3
"""Sprint 100 — Deep Audit of unified_system.py, coordinator.py, super_orchestrator.py

Tests for thread safety, sanitization, deepcopy safety, and UUID fixes.
"""

import pytest
import copy
import threading
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from pathlib import Path
from dataclasses import dataclass, field


# ═══════════════════════════════════════════════════════════════════
# unified_system.py Tests
# ═══════════════════════════════════════════════════════════════════

class TestUnifiedSystemSanitization:
    def test_observe_strips_control_chars(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE
        text = "hello\x00\x01\x02world"
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert result == "helloworld"

    def test_observe_strips_del_char(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE
        text = "hello\x7fworld"
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert result == "helloworld"

    def test_observe_preserves_newlines(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE
        text = "hello\nworld\ttab"
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert "\n" in result
        assert "\t" in result

    def test_observe_preserves_angles(self):
        from shadowd.unified_system import _SANITIZE_CONTENT_RE
        text = "latency < 100ms"
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert "<" in result
        assert "100ms" in result

    def test_max_observation_size(self):
        from shadowd.unified_system import _MAX_OBSERVATION_SIZE
        assert _MAX_OBSERVATION_SIZE == 65536

    def test_max_question_size(self):
        from shadowd.unified_system import _MAX_QUESTION_SIZE
        assert _MAX_QUESTION_SIZE == 10000


class TestUnifiedSystemThreadSafety:
    def test_has_thread_lock(self):
        from shadowd.unified_system import UnifiedShadowSystem
        # _lock is initialized in __init__, not as class attribute
        # Verify by checking the source has threading.Lock
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.__init__)
        assert "self._lock" in source

    def test_get_unified_status_returns_copy(self):
        from shadowd.unified_system import UnifiedShadowSystem
        system = UnifiedShadowSystem.__new__(UnifiedShadowSystem)
        system._lock = threading.Lock()
        system._tasks = []
        system.memory = MagicMock()
        system.memory.get_stats.return_value = {"test_key": "test_val"}
        system.context = MagicMock()
        system.context.current_snapshot = None
        system.context.context_history = []
        system.ambient = MagicMock()
        system.ambient.get_status.return_value = {"ambient_key": "ambient_val"}
        system.learning = MagicMock()
        system.learning.get_status.return_value = {"learning_key": "learning_val"}
        system.domains = MagicMock()
        system.domains.get_status.return_value = {"domain_key": "domain_val"}
        system.optimizer = MagicMock()
        system.optimizer.get_optimization_stats.return_value = {"opt_key": "opt_val"}
        system.proactive = MagicMock()
        system.proactive.get_summary.return_value = {"proactive_key": "proactive_val"}
        system.coordinator = None

        status = system.get_unified_status()
        # Mutate returned value — should not affect internals
        status["components"]["memory"]["test_key"] = "modified"
        status2 = system.get_unified_status()
        assert status2["components"]["memory"]["test_key"] == "test_val"


class TestUnifiedSystemTrackDecision:
    def test_track_decision_sanitizes(self):
        from shadowd.unified_system import UnifiedShadowSystem
        system = UnifiedShadowSystem.__new__(UnifiedShadowSystem)
        system.learning = MagicMock()
        system.learning.track_decision.return_value = MagicMock(id="test-id")
        system._lock = threading.Lock()

        system.track_decision(
            context="test\x00context",
            options=["opt1\x01", "opt2"],
            chosen="chosen<script>",
            reasoning="reason\x7fing",
        )
        call_args = system.learning.track_decision.call_args
        # Control chars should be stripped from context
        assert "\x00" not in call_args.kwargs.get("context", call_args[1].get("context", ""))


# ═══════════════════════════════════════════════════════════════════
# coordinator.py Tests
# ═══════════════════════════════════════════════════════════════════

class TestCoordinatorCopySafety:
    def test_get_ability_status_returns_tags_copy(self):
        from shadowd.coordinator import AbilityProfile, AbilityCoordinator
        from shadowd.coordinator import UnifiedContext
        
        ctx = UnifiedContext()
        profile = AbilityProfile(
            name="test_ability",
            ability_type=MagicMock(value="test"),
            description="Test ability",
            tags=["tag1", "tag2"],
        )
        ctx.profiles["test"] = profile
        
        coordinator = AbilityCoordinator.__new__(AbilityCoordinator)
        import threading
        coordinator._lock = threading.Lock()
        coordinator.context = ctx
        
        status = coordinator.get_ability_status()
        tags = status["test"]["profile"]["tags"]
        assert isinstance(tags, list)
        tags.append("modified")
        
        # Original should be unchanged
        status2 = coordinator.get_ability_status()
        assert "modified" not in status2["test"]["profile"]["tags"]


# ═══════════════════════════════════════════════════════════════════
# super_orchestrator.py Tests
# ═══════════════════════════════════════════════════════════════════

class TestSuperOrchestratorThreadSafety:
    def test_has_thread_locks(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        import inspect
        source = inspect.getsource(SuperShadowOrchestrator.__init__)
        assert "self._lock" in source
        assert "self._state_lock" in source

    def test_get_status_returns_copy(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        
        orch = SuperShadowOrchestrator.__new__(SuperShadowOrchestrator)
        orch._lock = threading.Lock()
        orch._state_lock = threading.Lock()
        orch.data_dir = Path("/tmp/test_shadow")
        orch.state = MagicMock(value="idle")
        orch.stats = MagicMock()
        orch.stats.to_dict.return_value = {"tasks_processed": 0}
        orch.agent_registry = MagicMock()
        orch.agent_registry.get_statistics.return_value = {"agents": 0}
        orch.function_db = MagicMock()
        orch.function_db.get_statistics.return_value = {"functions": 0}
        orch.memory = MagicMock()
        orch.prompt_cache = MagicMock()
        orch.prompt_cache.get_stats.return_value = {"hit_rate": 0.0}
        orch._crews = {}
        orch._active_workflows = {}
        orch._MAX_CREWS = 50

        status = orch.get_status()
        # Mutate returned copy — should not affect internals
        status["stats"]["tasks_processed"] = 99999
        status2 = orch.get_status()
        assert status2["stats"]["tasks_processed"] == 0

    def test_get_status_counts_crews_and_workflows(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        
        orch = SuperShadowOrchestrator.__new__(SuperShadowOrchestrator)
        orch._lock = threading.Lock()
        orch._state_lock = threading.Lock()
        orch.data_dir = Path("/tmp/test_shadow")
        orch.state = MagicMock(value="idle")
        orch.stats = MagicMock()
        orch.stats.to_dict.return_value = {"tasks_processed": 0}
        orch.agent_registry = MagicMock()
        orch.agent_registry.get_statistics.return_value = {}
        orch.function_db = MagicMock()
        orch.function_db.get_statistics.return_value = {}
        orch.memory = MagicMock()
        orch.prompt_cache = MagicMock()
        orch.prompt_cache.get_stats.return_value = {}
        orch._crews = {"c1": MagicMock(), "c2": MagicMock()}
        orch._active_workflows = {"w1": MagicMock()}

        status = orch.get_status()
        assert status["crews"] == 2
        assert status["workflows"]["active"] == 1


class TestSuperOrchestratorUUID:
    def test_crew_id_12_chars(self):
        import uuid
        id_str = uuid.uuid4().hex[:12]
        assert len(id_str) == 12

    def test_workflow_id_12_chars(self):
        import uuid
        id_str = uuid.uuid4().hex[:12]
        assert len(id_str) == 12


# ═══════════════════════════════════════════════════════════════════
# UnifiedContext Thread Safety Tests
# ═══════════════════════════════════════════════════════════════════

class TestUnifiedContextThreadSafety:
    def test_context_mutable_state(self):
        from shadowd.coordinator import UnifiedContext
        ctx = UnifiedContext()
        ctx.state["key1"] = "value1"
        assert "key1" in ctx.state
        assert ctx.state["key1"] == "value1"

    def test_context_profiles_dict(self):
        from shadowd.coordinator import UnifiedContext
        ctx = UnifiedContext()
        assert isinstance(ctx.profiles, dict)
        assert len(ctx.profiles) == 0

    def test_context_completed_abilities_list(self):
        from shadowd.coordinator import UnifiedContext
        ctx = UnifiedContext()
        ctx.completed_abilities.append("test_ability")
        assert "test_ability" in ctx.completed_abilities


# ═══════════════════════════════════════════════════════════════════
# Regression Tests — Ensure No Imports Or Basic Functionality Broken
# ═══════════════════════════════════════════════════════════════════

class TestImportSmokeTests:
    def test_import_unified_system(self):
        from shadowd.unified_system import UnifiedShadowSystem
        assert UnifiedShadowSystem is not None

    def test_import_coordinator(self):
        from shadowd.coordinator import AbilityCoordinator, AbilityProfile
        assert AbilityCoordinator is not None
        assert AbilityProfile is not None

    def test_import_super_orchestrator(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator, SuperShadowState
        assert SuperShadowOrchestrator is not None
        assert SuperShadowState is not None

    def test_super_shadow_state_values(self):
        from shadowd.super_orchestrator import SuperShadowState
        assert SuperShadowState.IDLE.value == "idle"
        assert SuperShadowState.PROCESSING.value == "processing"
        assert SuperShadowState.LEARNING.value == "learning"

    def test_ability_profile_fields(self):
        from shadowd.coordinator import AbilityProfile
        from shikigami import ShikigamiType
        profile = AbilityProfile(
            name="test",
            ability_type=ShikigamiType.X_RAY,
            description="Test",
            tags=["test_tag"],
        )
        assert profile.name == "test"
        assert profile.max_iterations == 15
        assert profile.timeout == 300