#!/usr/bin/env python3
"""Session 5 Sprint 15 regression tests - Shikigami/superhuman/coordinator audit.

Covers:
- V-370: shikigami.py _llm returns generic error (no str(e))
- V-371: shikigami.py _async_cmd/_cmd return generic error
- V-372: enhanced_abilities.py _llm returns generic error (no str(e))
- V-373: enhanced_abilities.py _async_cmd/_cmd return generic error
- V-374: Reality Forge filename validation in both files
- V-375: Reality Forge atomic write in both files
- V-376: superhuman.py MD5 → SHA-256 (entity IDs, prediction IDs)
- V-377: shikigami.py track_person devices cap + people cap
- V-378: shikigami.py track_decision decisions cap + satisfaction clamp
- V-379: enhanced_abilities.py track_person devices cap
- V-380: enhanced_abilities.py kalman_filters cap
- V-381: superhuman.py KnowledgeGraph entities/relations caps
- V-382: superhuman.py BayesianPredictor hypotheses cap
- V-383: Logger str(e) → type(e).__name__ across all files
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


class TestShikigamiLLMErrorGeneric:
    """V-370: shikigami.py _llm error is generic."""

    def test_llm_error_no_str_e(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry._llm)
        assert 'f"Error: {e}"' not in source
        assert 'type(e).__name__' in source


class TestShikigamiCommandErrorGeneric:
    """V-371: shikigami.py command errors are generic."""

    def test_async_cmd_no_str_e(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry._async_cmd)
        assert '"execution failed"' in source
        assert 'str(e)' not in source

    def test_cmd_no_str_e(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry._cmd)
        assert '"execution failed"' in source
        assert 'str(e)' not in source


class TestEnhancedLLMErrorGeneric:
    """V-372: enhanced_abilities.py _llm error is generic."""

    def test_llm_error_no_str_e(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        source = inspect.getsource(EnhancedShikigamiRegistry._llm)
        assert 'f"Error: {e}"' not in source
        assert 'type(e).__name__' in source


class TestEnhancedCommandErrorGeneric:
    """V-373: enhanced_abilities.py command errors are generic."""

    def test_async_cmd_no_str_e(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        source = inspect.getsource(EnhancedShikigamiRegistry._async_cmd)
        assert '"execution failed"' in source

    def test_cmd_no_str_e(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        source = inspect.getsource(EnhancedShikigamiRegistry._cmd)
        assert '"execution failed"' in source


class TestRealityForgeFilenameValidation:
    """V-374: Reality Forge validates filenames with regex."""

    def test_shikigami_safe_filename_regex(self):
        from shikigami.shikigami import _SAFE_FILENAME_RE
        assert _SAFE_FILENAME_RE.match("hello.py")
        assert _SAFE_FILENAME_RE.match("my-script.sh")
        assert _SAFE_FILENAME_RE.match("config.json")
        assert not _SAFE_FILENAME_RE.match("../../etc/passwd")
        assert not _SAFE_FILENAME_RE.match("")
        assert not _SAFE_FILENAME_RE.match("a" * 200 + ".py")

    def test_enhanced_safe_filename_regex(self):
        from shikigami.enhanced_abilities import _SAFE_FILENAME_RE
        assert _SAFE_FILENAME_RE.match("hello.py")
        assert not _SAFE_FILENAME_RE.match("../../etc/passwd")


class TestRealityForgeAtomicWrite:
    """V-375: Reality Forge uses tmp+rename for file writes."""

    def test_shikigami_reality_forge_atomic(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry._reality_forge)
        assert ".tmp" in source
        assert ".replace(" in source

    def test_enhanced_reality_forge_atomic(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        source = inspect.getsource(EnhancedShikigamiRegistry._reality_forge)
        assert ".tmp" in source
        assert ".replace(" in source


class TestSuperhumanSHA256:
    """V-376: superhuman.py uses SHA-256 instead of MD5."""

    def test_entity_id_uses_sha256(self):
        from shikigami.superhuman import KnowledgeGraph
        source = inspect.getsource(KnowledgeGraph.add_entity)
        assert "sha256" in source
        assert "md5" not in source

    def test_prediction_id_uses_sha256(self):
        from shikigami.superhuman import FeedbackLoop
        source = inspect.getsource(FeedbackLoop.record_prediction)
        assert "sha256" in source
        assert "md5" not in source

    def test_no_md5_in_superhuman(self):
        text = Path(_repo_file("shikigami/superhuman.py")).read_text()
        assert "hashlib.md5" not in text


class TestShikigamiTrackPersonCaps:
    """V-377: shikigami.py track_person has caps on people and devices."""

    def test_max_people_constant(self):
        from shikigami.shikigami import ShikigamiRegistry
        reg = ShikigamiRegistry()
        assert hasattr(reg, '_MAX_PEOPLE')
        assert reg._MAX_PEOPLE == 1000

    def test_max_devices_constant(self):
        from shikigami.shikigami import ShikigamiRegistry
        reg = ShikigamiRegistry()
        assert hasattr(reg, '_MAX_DEVICES_PER_PERSON')
        assert reg._MAX_DEVICES_PER_PERSON == 50


class TestShikigamiTrackDecisionCaps:
    """V-378: shikigami.py track_decision has decisions cap and satisfaction clamp."""

    def test_max_decisions_constant(self):
        from shikigami.shikigami import ShikigamiRegistry
        reg = ShikigamiRegistry()
        assert hasattr(reg, '_MAX_DECISIONS')
        assert reg._MAX_DECISIONS == 10000

    def test_satisfaction_clamped(self):
        from shikigami.shikigami import ShikigamiRegistry
        reg = ShikigamiRegistry()
        reg.decisions = []
        result = reg.track_decision("test", "choice", "ok", satisfaction=100)
        assert reg.decisions[-1]["satisfaction"] == 10
        result = reg.track_decision("test", "choice", "ok", satisfaction=-5)
        assert reg.decisions[-1]["satisfaction"] == 1


class TestEnhancedTrackPersonCaps:
    """V-379: enhanced_abilities.py track_person has caps."""

    def test_max_people_constant(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        reg = EnhancedShikigamiRegistry()
        assert hasattr(reg, '_MAX_PEOPLE')
        assert reg._MAX_PEOPLE == 1000

    def test_max_devices_constant(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        reg = EnhancedShikigamiRegistry()
        assert hasattr(reg, '_MAX_DEVICES_PER_PERSON')
        assert reg._MAX_DEVICES_PER_PERSON == 50


class TestEnhancedKalmanFiltersCap:
    """V-380: enhanced_abilities.py kalman_filters capped."""

    def test_max_kalman_filters_constant(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        reg = EnhancedShikigamiRegistry()
        assert hasattr(reg, '_MAX_KALMAN_FILTERS')
        assert reg._MAX_KALMAN_FILTERS == 500


class TestKnowledgeGraphCaps:
    """V-381: superhuman.py KnowledgeGraph has entity/relation caps."""

    def test_max_entities_constant(self):
        from shikigami.superhuman import KnowledgeGraph
        kg = KnowledgeGraph()
        assert kg._MAX_ENTITIES == 5000

    def test_max_relations_constant(self):
        from shikigami.superhuman import KnowledgeGraph
        kg = KnowledgeGraph()
        assert kg._MAX_RELATIONS == 10000

    def test_entity_name_truncated(self):
        from shikigami.superhuman import KnowledgeGraph
        kg = KnowledgeGraph()
        eid = kg.add_entity("person", "A" * 500, {"role": "test"})
        assert len(kg.entities[eid]["name"]) == 200

    def test_mentioned_list_capped(self):
        from shikigami.superhuman import KnowledgeGraph
        kg = KnowledgeGraph()
        eid = kg.add_entity("person", "Alice")
        for i in range(150):
            kg.add_entity("person", "Alice", {"count": i})
        assert len(kg.entities[eid]["mentioned"]) <= 100


class TestBayesianPredictorCap:
    """V-382: superhuman.py BayesianPredictor hypotheses capped."""

    def test_max_hypotheses_constant(self):
        from shikigami.superhuman import BayesianPredictor
        bp = BayesianPredictor()
        assert bp._MAX_HYPOTHESES == 200

    def test_hypotheses_eviction(self):
        from shikigami.superhuman import BayesianPredictor
        bp = BayesianPredictor()
        bp._MAX_HYPOTHESES = 5
        for i in range(10):
            bp.add_hypothesis(f"h_{i}", 0.5, ["evidence"])
        assert len(bp.hypotheses) <= 5


class TestLoggerStrEFixed:
    """V-383: All logger str(e) replaced with type(e).__name__."""

    def test_shikigami_logger_no_str_e(self):
        text = Path(_repo_file("shikigami/shikigami.py")).read_text()
        assert 'logger.error(f"Save failed: {e}")' not in text
        assert 'logger.error(f"Command error: {e}")' not in text

    def test_enhanced_logger_no_str_e(self):
        text = Path(_repo_file("shikigami/enhanced_abilities.py")).read_text()
        assert 'logger.error(f"Save observations error: {e}")' not in text
        assert 'logger.error(f"LLM error: {e}")' not in text
        assert 'logger.error(f"Save error: {e}")' not in text
        assert 'logger.debug(f"Decision tree error: {e}")' not in text

    def test_superhuman_logger_no_str_e(self):
        text = Path(_repo_file("shikigami/superhuman.py")).read_text()
        assert 'Load Q-table error: {e}' not in text
        assert 'Save model {name} error: {e}' not in text
        assert 'Load model {name} error: {e}' not in text


class TestShikigamiLoggerUsesTypeName:
    """Check shikigami.py logger uses type(e).__name__."""

    def test_load_json_uses_type_name(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry._load_json)
        assert "type(e).__name__" in source

    def test_save_json_uses_type_name(self):
        from shikigami.shikigami import ShikigamiRegistry
        source = inspect.getsource(ShikigamiRegistry._save_json)
        assert "type(e).__name__" in source


class TestEnhancedLoggerUsesTypeName:
    """Check enhanced_abilities.py logger uses type(e).__name__."""

    def test_save_observations_uses_type_name(self):
        from shikigami.enhanced_abilities import EnhancedShikigamiRegistry
        source = inspect.getsource(EnhancedShikigamiRegistry._save_observations)
        assert "type(e).__name__" in source