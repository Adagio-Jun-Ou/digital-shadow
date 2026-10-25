"""
Sprint 89 Regression Tests — Sanitization added to dark_memory, self_improvement, shadow_graph

These 3 modules previously had NO input sanitization at all. Now they have both
_sanitize() (strict, strips <>) and _sanitize_content() (preserves <>) functions.

- dark_memory: store() content uses content-safe, tags/domain use strict;
  add_entity()/add_relation() IDs and types use strict
- self_improvement: record_prediction content/predicted_outcome use content-safe,
  domain/model use strict; record_decision action/reasoning/predicted_outcome use
  content-safe, domain uses strict; add_feedback comment uses content-safe
- shadow_graph: process_event query/observations use content-safe, domain uses
  strict; add_node name uses strict; add_edge source/target use strict
"""

import pytest


class TestDarkMemorySanitize:
    def test_strict_strips_angles(self):
        from shadowd.cognition.dark_memory import _sanitize
        assert "<" not in _sanitize("test<script>")
        assert ">" not in _sanitize("test</script>")

    def test_content_preserves_angles(self):
        from shadowd.cognition.dark_memory import _sanitize_content
        result = _sanitize_content("latency < 100ms")
        assert "<" in result
        assert "latency < 100ms" in result

    def test_both_strips_control_chars(self):
        from shadowd.cognition.dark_memory import _sanitize, _sanitize_content
        assert "\x00" not in _sanitize("test\x00data")
        assert "\x1f" not in _sanitize("test\x1fdata")
        assert "\x7f" not in _sanitize("test\x7fdata")
        assert "\x00" not in _sanitize_content("test\x00data")
        assert "\x7f" not in _sanitize_content("test\x7fdata")

    def test_strict_strips_whitespace_content_preserves(self):
        from shadowd.cognition.dark_memory import _sanitize, _sanitize_content
        text = "hello\tworld\nnew\rline"
        strict = _sanitize(text, 100)
        content = _sanitize_content(text, 100)
        assert "\t" not in strict and "\n" not in strict and "\r" not in strict
        assert "\t" in content and "\n" in content and "\r" in content

    def test_sanitize_none(self):
        from shadowd.cognition.dark_memory import _sanitize, _sanitize_content
        assert _sanitize(None) == ""
        assert _sanitize_content(None) == ""


class TestSelfImprovementSanitize:
    def test_strict_strips_angles(self):
        from shadowd.cognition.self_improvement import _sanitize
        assert "<" not in _sanitize("test<script>")

    def test_content_preserves_angles(self):
        from shadowd.cognition.self_improvement import _sanitize_content
        result = _sanitize_content("latency < 100ms > 50ms")
        assert "<" in result
        assert ">" in result

    def test_both_strips_control(self):
        from shadowd.cognition.self_improvement import _sanitize, _sanitize_content
        assert "\x00" not in _sanitize("test\x00data")
        assert "\x7f" not in _sanitize("test\x7fdata")
        assert "\x00" not in _sanitize_content("test\x00data")
        assert "\x7f" not in _sanitize_content("test\x7fdata")

    def test_record_prediction_sanitizes(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        si = SelfImprovementEngine()
        pred = si.record_prediction(
            content="score < 0.5\x00test",
            predicted_outcome="result > 0.8\x7f",
            domain="work<script>",
            model="default<script>"
        )
        assert "<" in pred.content
        assert "\x00" not in pred.content
        assert ">" in pred.predicted_outcome
        assert "\x7f" not in pred.predicted_outcome
        assert "<" not in pred.domain
        assert "<" not in pred.model

    def test_record_decision_sanitizes(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        si = SelfImprovementEngine()
        dec = si.record_decision(
            action="deploy because latency < threshold",
            reasoning="score > 0.9\x00test",
            domain="prod<script>"
        )
        assert "<" in dec.action
        assert ">" in dec.reasoning
        assert "\x00" not in dec.reasoning
        assert "<" not in dec.domain

    def test_add_feedback_sanitizes(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine, FeedbackSource
        si = SelfImprovementEngine()
        dec = si.record_decision("test")
        dec.add_feedback(FeedbackSource.EXPLICIT_USER, 0.8, "score < threshold\x00test")
        fb = dec.feedback[-1]
        assert "<" in fb["comment"]
        assert "\x00" not in fb["comment"]


class TestShadowGraphSanitize:
    def test_strict_strips_angles(self):
        from shadowd.cognition.shadow_graph import _sanitize
        assert "<" not in _sanitize("node<script>")

    def test_content_preserves_angles(self):
        from shadowd.cognition.shadow_graph import _sanitize_content
        result = _sanitize_content("latency < 100ms")
        assert "<" in result

    def test_both_strips_control(self):
        from shadowd.cognition.shadow_graph import _sanitize, _sanitize_content
        assert "\x00" not in _sanitize("test\x00data")
        assert "\x7f" not in _sanitize("test\x7fdata")
        assert "\x00" not in _sanitize_content("test\x00data")
        assert "\x7f" not in _sanitize_content("test\x7fdata")

    def test_sanitize_none(self):
        from shadowd.cognition.shadow_graph import _sanitize, _sanitize_content
        assert _sanitize(None) == ""
        assert _sanitize_content(None) == ""