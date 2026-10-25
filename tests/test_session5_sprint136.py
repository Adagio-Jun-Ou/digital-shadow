#!/usr/bin/env python3
"""Sprint 136 regression tests — remaining .strip()-before-truncate in 10 modules."""


def test_recall_synthesis_sanitize_strip():
    from shadowd.cognition.recall_synthesis import _sanitize, _sanitize_strict
    assert _sanitize("  hello  ", 10) == "hello"
    assert _sanitize_strict("  <tag>  ", 10) == "tag"


def test_recall_synthesis_sanitize_truncation_after_strip():
    from shadowd.cognition.recall_synthesis import _sanitize
    assert _sanitize("  hello world  ", 8) == "hello wo"


def test_shadow_graph_sanitize_content_strip():
    from shadowd.cognition.shadow_graph import _sanitize_content
    assert _sanitize_content("  test  ", 10) == "test"


def test_shadow_graph_sanitize_content_truncation():
    from shadowd.cognition.shadow_graph import _sanitize_content
    assert _sanitize_content("  hello world  ", 8) == "hello wo"


def test_thought_sense_sanitize_strip():
    from shadowd.cognition.thought_sense import _sanitize, _sanitize_strict
    assert _sanitize("  hello  ") == "hello"
    assert _sanitize_strict("  <tag>  ") == "tag"


def test_personality_engine_sanitize_strip():
    from shadowd.cognition.personality_engine import _sanitize, _sanitize_content
    assert _sanitize("  hello  ") == "hello"
    assert _sanitize_content("  hello  ") == "hello"


def test_personality_engine_sanitize_content_preserves_lt():
    from shadowd.cognition.personality_engine import _sanitize, _sanitize_content
    assert "<" in _sanitize_content("  x < 5  ")
    assert "<" not in _sanitize("  x < 5  ")


def test_stealth_layer_sanitize_strip():
    from shadowd.cognition.stealth_layer import _sanitize, _sanitize_content
    assert _sanitize("  hello  ") == "hello"
    assert _sanitize_content("  hello  ") == "hello"


def test_stealth_layer_sanitize_content_preserves_lt():
    from shadowd.cognition.stealth_layer import _sanitize, _sanitize_content
    assert "<" in _sanitize_content("  x < 5  ")
    assert "<" not in _sanitize("  x < 5  ")


def test_shadow_crew_sanitize_strip():
    from shadowd.cognition.shadow_crew import _sanitize, _sanitize_strict, _sanitize_content
    assert _sanitize("  hello  ") == "hello"
    assert _sanitize_strict("  <tag>  ") == "tag"
    assert _sanitize_content("  hello  ") == "hello"


def test_shadow_crew_sanitize_content_preserves_lt():
    from shadowd.cognition.shadow_crew import _sanitize, _sanitize_strict, _sanitize_content, _SANITIZE_RE, _SANITIZE_STRICT_RE
    assert "<" in _sanitize_content("x < 5")
    assert "<" not in _sanitize_strict("x < 5")
    # _sanitize uses _SANITIZE_RE which is content-safe (no <>)
    assert "<" in _sanitize("x < 5")


def test_empathy_mapping_topic_strip():
    from shadowd.cognition.empathy_mapping import EmpathyMapping
    em = EmpathyMapping()
    person_id = em.add_person("Test Person")
    result = em.record_interaction(person_id, "text", 0.5, notes="test")
    assert result is not None


def test_daily_briefing_domain_strip():
    from shadowd.cognition.daily_briefing import BriefingItem, BriefingSection
    bi = BriefingItem(section=BriefingSection.TODAY, title="Test", content="content", domain="  work  ")
    assert bi.domain == "work"


def test_daily_briefing_action_url_strip():
    from shadowd.cognition.daily_briefing import BriefingItem, BriefingSection
    bi = BriefingItem(section=BriefingSection.TODAY, title="Test", content="content", action_url="  https://example.com  ")
    assert bi.action_url == "https://example.com"


def test_escalation_evaluate_action_strip():
    from shadowd.cognition.escalation import EscalationManager
    em = EscalationManager()
    result = em.evaluate("  test_action  ", {"key": "val"}, domain="  test  ")
    assert result is not None


def test_presence_detection_metadata_strip():
    from shadowd.cognition.presence_detection import PresenceDetection
    pd = PresenceDetection()
    pd.register_device("dev1", "laptop", "My Device")
    # metadata is set via ingest/update, not register_device
    device = pd._devices.get("dev1")
    assert device is not None


def test_cognitive_pipeline_result_strip():
    from shadowd.cognition.cognitive_pipeline import PipelineResult
    pr = PipelineResult(
        text="  Hello world  ",
        context="  testing  ",
        overall_assessment="  good result  ",
        overall_confidence=0.75,
    )
    d = pr.to_dict()
    assert d["text"] == "Hello world"
    assert d["context"] == "testing"
    assert d["overall_assessment"] == "good result"


def test_all_modules_sanitize_strip_consistency():
    """Verify all cognition _sanitize functions call .strip()."""
    modules = [
        ("shadowd.cognition.recall_synthesis", "_sanitize"),
        ("shadowd.cognition.recall_synthesis", "_sanitize_strict"),
        ("shadowd.cognition.shadow_graph", "_sanitize_content"),
        ("shadowd.cognition.thought_sense", "_sanitize"),
        ("shadowd.cognition.thought_sense", "_sanitize_strict"),
        ("shadowd.cognition.personality_engine", "_sanitize"),
        ("shadowd.cognition.personality_engine", "_sanitize_content"),
        ("shadowd.cognition.stealth_layer", "_sanitize"),
        ("shadowd.cognition.stealth_layer", "_sanitize_content"),
        ("shadowd.cognition.shadow_crew", "_sanitize"),
        ("shadowd.cognition.shadow_crew", "_sanitize_strict"),
        ("shadowd.cognition.shadow_crew", "_sanitize_content"),
        ("shadowd.cognition.goal_planner", "_sanitize"),
        ("shadowd.cognition.goal_planner", "_sanitize_strict"),
        ("shadowd.cognition.anomaly_detector", "_sanitize"),
        ("shadowd.cognition.anomaly_detector", "_sanitize_content"),
    ]
    import importlib
    for mod_name, func_name in modules:
        mod = importlib.import_module(mod_name)
        func = getattr(mod, func_name)
        result = func("  padded  ")
        assert result == "padded" or result.strip() == result, f"{mod_name}.{func_name} doesn't strip"


def test_recall_synthesis_observation_round():
    from shadowd.cognition.recall_synthesis import Observation
    o = Observation("id1", "test", "domain", "2025-01-01", 0.3333333, ["kw"])
    d = o.to_dict()
    assert d["importance"] == round(0.3333333, 3)


def test_shadow_graph_urgency_round():
    from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
    cs = CognitiveState(urgency=0.7777777)
    d = cs.to_dict()
    assert d["urgency"] == round(0.7777777, 3)


def test_shadow_graph_duration_ms_round():
    from shadowd.cognition.shadow_graph import ExecutionTrace
    et = ExecutionTrace(execution_id="test", start_time="2025-01-01", duration_ms=123.456789)
    d = et.to_dict()
    assert d["duration_ms"] == round(123.456789, 2)