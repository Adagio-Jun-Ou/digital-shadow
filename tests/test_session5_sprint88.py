"""
Sprint 88 Regression Tests — Content-safe vs Strict Sanitize Distinction

Modules updated to use _sanitize_content() (preserve <>) for content fields
and keep _sanitize() (strip <>) for identifier fields:

- conflict_resolution: _sanitize() strips <>, _sanitize_content() preserves <>
- narrative_memory: _sanitize() strips <>, _sanitize_content() preserves <>
- procedural_memory: _sanitize() strips <>, _sanitize_content() preserves <>
- cognitive_trinity: _sanitize() strips <>, _sanitize_content() preserves <>; situation uses content-safe
- attention_engine: _sanitize() strips <>, _sanitize_content() preserves <>; topic/description use content-safe
- decision_journal: _sanitize() strips <>, _sanitize_content() preserves <>; description/reasoning/outcome use content-safe
- cognitive_reflection: _sanitize() strips <>, _sanitize_content() preserves <>; description/lessons use content-safe
- goal_reasoning: _sanitize() strips <>, _sanitize_content() preserves <>; description/mitigation/reason use content-safe
- anomaly_detector: _sanitize() strips <>, _sanitize_content() preserves <>; description uses content-safe
- cognitive_pipeline: _sanitize_RE strips <>, _sanitize_content_RE preserves <>; text/context use content-safe
- shadow_voice: speaker strips <>, transcript/text preserve <>
- dream_state: content/query preserve <>, domain/tags strip <>
- goal_planner: _SANITIZE_RE preserves <>, _SANITIZE_STRICT_RE strips <>
"""

import re
import pytest


CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
STRICT_RE = re.compile(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
TEST_CONTENT = "latency < 100ms > 50ms\x00\x7f"
TEST_ID = "id<script>\x00\x7f"


class TestConflictResolutionDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        cr = ConflictResolution()
        result = cr._sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        result = ConflictResolution._sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        result = ConflictResolution._sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestNarrativeMemoryDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        nm = NarrativeMemory()
        result = nm._sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        result = NarrativeMemory._sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.narrative_memory import NarrativeMemory
        result = NarrativeMemory._sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestProceduralMemoryDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.procedural_memory import ProceduralMemory
        pm = ProceduralMemory()
        result = pm._sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.procedural_memory import ProceduralMemory
        result = ProceduralMemory._sanitize_content("latency < 100ms")
        assert "<" in result


class TestCognitiveTrinityDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct._sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        result = CognitiveTrinity._sanitize_content("latency < 100ms")
        assert "<" in result

    def test_analyze_uses_content_safe(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct.analyze("latency < 100ms\x00test")
        assert "<" in result.situation
        assert "\x00" not in result.situation


class TestAttentionEngineDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        ae = AttentionEngine()
        result = ae._sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        result = AttentionEngine._sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        result = AttentionEngine._sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestDecisionJournalDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.decision_journal import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.decision_journal import _sanitize_content
        result = _sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.decision_journal import _sanitize_content
        result = _sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestCognitiveReflectionDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.cognitive_reflection import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.cognitive_reflection import _sanitize_content
        result = _sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.cognitive_reflection import _sanitize_content
        result = _sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestGoalReasoningDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.goal_reasoning import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.goal_reasoning import _sanitize_content
        result = _sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.goal_reasoning import _sanitize_content
        result = _sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestAnomalyDetectorDual:
    def test_strict_strips_angles(self):
        from shadowd.cognition.anomaly_detector import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result

    def test_content_preserves_angles(self):
        from shadowd.cognition.anomaly_detector import _sanitize_content
        result = _sanitize_content("latency < 100ms")
        assert "<" in result

    def test_content_strips_control(self):
        from shadowd.cognition.anomaly_detector import _sanitize_content
        result = _sanitize_content("test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result


class TestCognitivePipelineDual:
    def test_content_re_preserves_angles(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_CONTENT_RE
        result = _SANITIZE_CONTENT_RE.sub('', "latency < 100ms > 50ms")
        assert "<" in result
        assert ">" in result

    def test_content_re_strips_control(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_CONTENT_RE
        result = _SANITIZE_CONTENT_RE.sub('', "test\x00data\x7f")
        assert "\x00" not in result
        assert "\x7f" not in result

    def test_strict_re_strips_angles(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', "latency < 100ms > 50ms")
        assert "<" not in result
        assert ">" not in result


class TestShadowVoiceContentSafe:
    def test_speaker_strips_angles(self):
        pattern = re.compile(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        result = pattern.sub('', "user<script>")
        assert "<" not in result

    def test_transcript_preserves_angles(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        result = pattern.sub('', "latency < 100ms > 50ms")
        assert "<" in result
        assert ">" in result


class TestGoalPlannerDual:
    def test_content_safe_preserves_angles(self):
        from shadowd.cognition.goal_planner import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', "latency < 100ms > 50ms")
        assert "<" in result

    def test_strict_strips_angles(self):
        from shadowd.cognition.goal_planner import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', "latency < 100ms > 50ms")
        assert "<" not in result


class TestAllPreservesWhitespace:
    """All sanitize variants preserve \\t, \\n, \\r"""

    def test_strict_preserves_whitespace(self):
        text = "hello\tworld\nnew\rline"
        result = STRICT_RE.sub('', text)
        assert "\t" in result
        assert "\n" in result
        assert "\r" in result

    def test_content_preserves_whitespace(self):
        text = "hello\tworld\nnew\rline"
        result = CONTENT_RE.sub('', text)
        assert "\t" in result
        assert "\n" in result
        assert "\r" in result

    def test_conflict_resolution_strict_strips_whitespace_content_preserves(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        cr = ConflictResolution()
        text = "hello\tworld\nnew\rline"
        strict = cr._sanitize(text)
        content = ConflictResolution._sanitize_content(text)
        assert "\t" not in strict
        assert "\n" not in strict
        assert "\r" not in strict
        assert "\t" in content
        assert "\n" in content