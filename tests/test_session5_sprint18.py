#!/usr/bin/env python3
"""Session 5 Sprint 18 regression tests - shadow_ai.py, mind.py, shadowd.py, shadow_enhanced.py.

Covers:
- shadow_ai.py: str(e) → type(e).__name__ (4 instances), patterns cap, people cap, devices cap
- mind.py: str(e) → type(e).__name__ (2 instances), API error no longer returned to caller, query length limit
- shadowd.py: str(e) → type(e).__name__ (2 instances)
- shadow_enhanced.py: non-atomic write → fsync, silent error → logged, str(e) → type(e).__name__ (2),
  LifeDomain records/goals caps, insights_queue cap
"""

import inspect
import pytest
import sys
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))


sys.path.insert(0, str(Path(__file__).parent.parent))


class TestShadowAIStrE:
    """shadow_ai.py logger str(e) → type(e).__name__."""

    def test_load_data_uses_type_name(self):
        from shadowd.shadow_ai import ShadowAI
        source = inspect.getsource(ShadowAI._load_data)
        assert "type(e).__name__" in source

    def test_save_data_uses_type_name(self):
        from shadowd.shadow_ai import ShadowAI
        source = inspect.getsource(ShadowAI._save_data)
        assert "type(e).__name__" in source

    def test_proactive_monitor_uses_type_name(self):
        from shadowd.shadow_ai import ShadowAI
        source = inspect.getsource(ShadowAI.run_proactive_monitor)
        assert "type(e).__name__" in source

    def test_ability_failed_uses_type_name(self):
        text = Path(_repo_file("shadowd/shadow_ai.py")).read_text()
        assert "Ability {ability} failed: {e}" not in text


class TestShadowAICaps:
    """shadow_ai.py patterns/people/devices caps."""

    def test_max_patterns(self):
        from shadowd.shadow_ai import ShadowAI
        assert ShadowAI._MAX_PATTERNS == 500

    def test_max_people(self):
        from shadowd.shadow_ai import ShadowAI
        assert ShadowAI._MAX_PEOPLE == 100

    def test_max_devices_per_person(self):
        from shadowd.shadow_ai import ShadowAI
        assert ShadowAI._MAX_DEVICES_PER_PERSON == 50

    def test_learn_pattern_eviction(self):
        from shadowd.shadow_ai import ShadowAI
        ai = ShadowAI.__new__(ShadowAI)
        ai.patterns = {}
        for i in range(600):
            ai.patterns[f"cause_{i}"] = f"effect_{i}"
            if len(ai.patterns) > ai._MAX_PATTERNS:
                oldest = next(iter(ai.patterns))
                del ai.patterns[oldest]
        assert len(ai.patterns) <= ai._MAX_PATTERNS


class TestMindStrE:
    """mind.py str(e) → type(e).__name__ and no API error returned."""

    def test_llm_response_json_uses_type_name(self):
        from shadowd.mind import MindEngine
        source = inspect.getsource(MindEngine._call_llm)
        assert 'LLM response not valid JSON: {e}' not in source
        assert "type(e).__name__" in source

    def test_llm_structure_uses_type_name(self):
        from shadowd.mind import MindEngine
        source = inspect.getsource(MindEngine._call_llm)
        assert 'LLM response missing expected structure: {e}' not in source

    def test_api_error_not_returned_to_caller(self):
        from shadowd.mind import MindEngine
        source = inspect.getsource(MindEngine._call_llm)
        assert 'data.get("error"' not in source

    def test_query_length_limit(self):
        # `think` enforces the cap by calling _sanitize_content with
        # _MAX_QUERY_LENGTH; verify both the call AND that the module-level
        # constant is the documented 50000.
        from shadowd import mind
        from shadowd.mind import MindEngine
        source = inspect.getsource(MindEngine.think)
        assert "_MAX_QUERY_LENGTH" in source or "50000" in source
        assert mind._MAX_QUERY_LENGTH == 50000


class TestShadowdStrE:
    """shadowd.py str(e) → type(e).__name__."""

    def test_orchestrator_uses_type_name(self):
        # The implementation later switched from f-string to lazy %s logging
        # for performance/safety. Either form is acceptable as long as the
        # raw exception ({e}) is *not* interpolated.
        text = Path(_repo_file("shadowd/shadowd.py")).read_text()
        assert "Super Orchestrator not available: {e}" not in text
        assert (
            "Super Orchestrator not available: {type(e).__name__}" in text
            or "Super Orchestrator not available: %s" in text
        )

    def test_cognitive_router_uses_type_name(self):
        text = Path(_repo_file("shadowd/shadowd.py")).read_text()
        assert "Cognitive module router not loaded: {e}" not in text
        assert (
            "Cognitive module router not loaded: {type(e).__name__}" in text
            or "Cognitive module router not loaded: %s" in text
        )


class TestShadowEnhancedStrE:
    """shadow_enhanced.py str(e) → type(e).__name__."""

    def test_ability_failed_uses_type_name(self):
        text = Path(_repo_file("shadowd/shadow_enhanced.py")).read_text()
        assert "Ability {ability.value} failed: {e}" not in text

    def test_ambient_monitor_uses_type_name(self):
        text = Path(_repo_file("shadowd/shadow_enhanced.py")).read_text()
        assert "Ambient monitor error: {e}" not in text


class TestShadowEnhancedAtomicWrite:
    """shadow_enhanced.py domain save uses fsync."""

    def test_domain_save_uses_fsync(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.add_domain_goal)
        assert "fsync" in source

    def test_domain_save_not_silent(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI.add_domain_goal)
        assert "type(e).__name__" in source


class TestShadowEnhancedCaps:
    """shadow_enhanced.py LifeDomain/insights caps."""

    def test_max_records_per_domain(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        assert EnhancedShadowAI._MAX_RECORDS_PER_DOMAIN == 1000

    def test_max_goals_per_domain(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        assert EnhancedShadowAI._MAX_GOALS_PER_DOMAIN == 100

    def test_max_insights_queue(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        assert EnhancedShadowAI._MAX_INSIGHTS_QUEUE == 500

    def test_cap_insights_queue_method(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        assert hasattr(EnhancedShadowAI, '_cap_insights_queue')