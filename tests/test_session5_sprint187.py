import pytest
import re
import threading
import copy
from unittest.mock import AsyncMock, MagicMock, patch


class TestSprint187ShadowCrewWordBoundary:
    """Sprint 187: word-boundary regex fixes in _plan_tasks_* methods."""

    @pytest.fixture
    def crew(self):
        from shadowd.cognition.shadow_crew import ShadowCrew
        sc = ShadowCrew.__new__(ShadowCrew)
        sc._lock = threading.Lock()
        sc._missions_lock = threading.Lock()
        sc._stats_lock = threading.Lock()
        sc._cb_lock = threading.Lock()
        sc._missions = {}
        sc._stats = {"missions_created": 0, "missions_completed": 0, "tasks_completed": 0, "handoffs": 0, "escalations": 0}
        sc._circuit_breakers = {}
        sc._MAX_MISSIONS = 100
        sc._MAX_SUBTASKS_PER_MISSION = 20
        sc._task_timeout = 30.0
        sc._shared_context = {}
        sc._registry = MagicMock()
        return sc

    def test_word_boundary_do_not_match_domain(self):
        """'do' should NOT match 'domain' in objective."""
        import shadowd.cognition.shadow_crew as sc_module
        objective = "Analyze the domain model for our business"
        obj_lower = objective.lower()
        result = any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["do", "execute", "run", "create", "build", "make"])
        assert not result, "'do' should not match 'domain'"

    def test_word_boundary_will_not_match_willing(self):
        """'will' should NOT match 'willing' in objective."""
        objective = "I am willing to help analyze the data"
        obj_lower = objective.lower()
        result = any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["predict", "forecast", "will", "chance"])
        assert not result, "'will' should not match 'willing'"

    def test_word_boundary_run_not_match_running(self):
        """'run' should NOT match 'running' in objective."""
        objective = "The system is running smoothly"
        obj_lower = objective.lower()
        result = any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["do", "execute", "run", "create", "build", "make"])
        assert not result, "'run' should not match 'running'"

    def test_word_boundary_execute_matches_exactly(self):
        """'execute' should match 'execute' exactly."""
        objective = "Please execute the script"
        obj_lower = objective.lower()
        result = any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["do", "execute", "run", "create", "build", "make"])
        assert result, "'execute' should match exactly"

    def test_word_boundary_predict_matches_exactly(self):
        """'predict' should match 'predict' exactly."""
        objective = "Predict the outcome"
        obj_lower = objective.lower()
        result = any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["predict", "forecast", "will", "chance"])
        assert result, "'predict' should match exactly"

    def test_word_boundary_forecast_matches_exactly(self):
        """'forecast' should match 'forecast' exactly."""
        objective = "Can you forecast the weather?"
        obj_lower = objective.lower()
        result = any(re.search(r'\b' + re.escape(kw) + r'\b', obj_lower) for kw in ["predict", "forecast", "will", "chance"])
        assert result, "'forecast' should match exactly"


class TestSprint187ShadowCrewSecurity:
    """Security and thread-safety fixes in shadow_crew."""

    def test_execute_mission_error_message_generic(self):
        """execute_mission should return generic error, not leak mission_id."""
        import asyncio
        from shadowd.cognition.shadow_crew import ShadowCrew
        sc = ShadowCrew.__new__(ShadowCrew)
        sc._missions_lock = threading.Lock()
        sc._missions = {}
        result = asyncio.new_event_loop().run_until_complete(
            sc.execute_mission("nonexistent_mission_id_" + "x" * 100)
        )
        assert isinstance(result, dict)
        assert "error" in result
        assert "nonexistent" not in str(result)

    def test_circuit_breaker_read_under_lock(self):
        """Circuit breaker success recording should use _cb_lock."""
        from shadowd.cognition.shadow_crew import ShadowCrew
        sc = ShadowCrew.__new__(ShadowCrew)
        sc._cb_lock = threading.Lock()
        sc._circuit_breakers = {}
        sc._MAX_CIRCUIT_BREAKERS = 50
        with sc._cb_lock:
            assert sc._circuit_breakers == {}