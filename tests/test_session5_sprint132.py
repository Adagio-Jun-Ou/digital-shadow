#!/usr/bin/env python3
"""Sprint 132 — Global .strip() before [:max_length] audit regression tests.

Verifies that all sanitize functions call .strip() BEFORE [:max_length]
truncation, not after. Also verifies content-safe vs strict sanitize
distinction is preserved after the strip-order fix.
"""

import pytest
import re


class TestCognitivePipelineStrip:
    def test_sanitize_pipeline_input_strips_before_truncate(self):
        from shadowd.cognition.cognitive_pipeline import _sanitize_pipeline_input
        result = _sanitize_pipeline_input("  hello world  ", max_len=10)
        assert result == "hello worl"
        assert not result.startswith(" ")

    def test_sanitize_pipeline_input_strips_whitespace(self):
        from shadowd.cognition.cognitive_pipeline import _sanitize_pipeline_input
        result = _sanitize_pipeline_input("  test  ", max_len=10)
        assert result == "test"


class TestShadowOSDStrip:
    def test_sanitize_strips_before_truncate(self):
        from shadowd.cognition.shadowosd import _sanitize
        result = _sanitize("  hello  ", max_length=5)
        assert result == "hello"

    def test_sanitize_content_strips_before_truncate(self):
        from shadowd.cognition.shadowosd import _sanitize_content
        result = _sanitize_content("  hello  ", max_length=5)
        assert result == "hello"

    def test_sanitize_strips_strict_chars(self):
        from shadowd.cognition.shadowosd import _sanitize
        result = _sanitize("<script>alert(1)</script>", max_length=200)
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.shadowosd import _sanitize_content
        result = _sanitize_content("latency < 100ms", max_length=200)
        assert "<" in result


class TestEscalationStrip:
    def test_sanitize_strips_before_truncate(self):
        from shadowd.cognition.escalation import _sanitize
        result = _sanitize("  test  ", max_length=5)
        assert result == "test"

    def test_sanitize_content_strips_before_truncate(self):
        from shadowd.cognition.escalation import _sanitize_content
        result = _sanitize_content("  test  ", max_length=5)
        assert result == "test"

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.escalation import _sanitize
        result = _sanitize("<b>bold</b>", max_length=200)
        assert "<" not in result

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.escalation import _sanitize_content
        result = _sanitize_content("value < 5", max_length=200)
        assert "<" in result


class TestConflictResolutionStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        cr = ConflictResolution()
        result = cr._sanitize("  hello  ", max_len=5)
        assert result == "hello"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        result = ConflictResolution._sanitize_content("  hello  ", max_len=5)
        assert result == "hello"

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        cr = ConflictResolution()
        result = cr._sanitize("<test>", max_len=200)
        assert "<" not in result

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.conflict_resolution import ConflictResolution
        result = ConflictResolution._sanitize_content("x < 5", max_len=200)
        assert "<" in result


class TestAdaptiveSkillTreeStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        ast = AdaptiveSkillTree()
        result = ast._sanitize("  test  ")
        assert result == "test"

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree
        ast = AdaptiveSkillTree()
        result = ast._sanitize("<script>")
        assert "<" not in result


class TestCognitiveTrinityStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()
        result = ct._sanitize("  hello  ")
        assert result == "hello"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        result = CognitiveTrinity._sanitize_content("  hello  ")
        assert result == "hello"


class TestShadowVoiceStrip:
    def test_add_transcript_strips_speaker(self):
        from shadowd.cognition.shadow_voice import VoiceSession
        session = VoiceSession(session_id="test")
        session.add_transcript("  alice  ", "hello world")
        assert session.transcript[-1]["speaker"] == "alice"

    def test_add_transcript_strips_text(self):
        from shadowd.cognition.shadow_voice import VoiceSession
        session = VoiceSession(session_id="test")
        session.add_transcript("alice", "  hello  ")
        assert session.transcript[-1]["text"] == "hello"


class TestNeuralMeshStrip:
    def test_sanitize_message_strips_keys(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        result = NeuralMesh._sanitize_message({"  key  ": "value"})
        keys = list(result.keys())
        assert keys[0] == "key"

    def test_sanitize_message_strips_values(self):
        from shadowd.cognition.neural_mesh import NeuralMesh
        result = NeuralMesh._sanitize_message({"k": "  value  "})
        assert result["k"] == "value"


class TestSelfImprovementStrip:
    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.self_improvement import _sanitize_content
        result = _sanitize_content("  test  ", max_length=10)
        assert result == "test"

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.self_improvement import _sanitize_content
        result = _sanitize_content("x < 5", max_length=200)
        assert "<" in result


class TestDarkMemoryStrip:
    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.dark_memory import _sanitize_content
        result = _sanitize_content("  test  ", max_length=10)
        assert result == "test"

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.dark_memory import _sanitize_content
        result = _sanitize_content("value < 10", max_length=200)
        assert "<" in result


class TestDeviceContextStrip:
    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.device_context import _sanitize
        result = _sanitize("  test  ", max_len=5)
        assert result == "test"

    def test_sanitize_strict_strips_whitespace(self):
        from shadowd.cognition.device_context import _sanitize_strict
        result = _sanitize_strict("  test  ", max_len=5)
        assert result == "test"

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.device_context import _sanitize
        result = _sanitize("x < 5", max_len=200)
        assert "<" in result

    def test_sanitize_strict_strips_angles(self):
        from shadowd.cognition.device_context import _sanitize_strict
        result = _sanitize_strict("<script>", max_len=200)
        assert "<" not in result


class TestAutonomousDaemonStrip:
    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.autonomous_daemon import _sanitize
        result = _sanitize("  test  ", max_len=5)
        assert result == "test"

    def test_sanitize_content_strips_whitespace(self):
        from shadowd.cognition.autonomous_daemon import _sanitize_content
        result = _sanitize_content("  test  ", max_len=5)
        assert result == "test"

    def test_sanitize_strict_strips_whitespace(self):
        from shadowd.cognition.autonomous_daemon import _sanitize_strict
        result = _sanitize_strict("  test  ", max_len=5)
        assert result == "test"

    def test_no_duplicate_sanitize_strict(self):
        import inspect
        from shadowd.cognition import autonomous_daemon
        source = inspect.getsource(autonomous_daemon)
        count = source.count("def _sanitize_strict(")
        assert count == 1, f"Found {count} definitions of _sanitize_strict, expected 1"

    def test_sanitize_strict_strips_angles(self):
        from shadowd.cognition.autonomous_daemon import _sanitize_strict
        result = _sanitize_strict("<script>", max_len=200)
        assert "<" not in result

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.autonomous_daemon import _sanitize_content
        result = _sanitize_content("value < 5", max_len=200)
        assert "<" in result