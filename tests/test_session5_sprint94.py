import copy
import pytest
from shadowd.cognition.shadow_os import ShadowOS, sanitize_input, validate_observation


class TestShadowOSStatsCopy:
    def test_get_health_returns_stats_copy(self):
        os_ = ShadowOS(data_dir="/tmp/test_sos_health")
        health = os_.get_health()
        stats = health["stats"]
        original_queries = stats.get("queries_handled", 0)
        stats["queries_handled"] = 9999
        assert os_._stats["queries_handled"] == original_queries

    def test_get_status_returns_stats_copy(self):
        os_ = ShadowOS(data_dir="/tmp/test_sos_status")
        status = os_.get_status()
        stats = status["stats"]
        original = stats.get("queries_handled", 0)
        stats["queries_handled"] = 9999
        assert os_._stats["queries_handled"] == original


class TestShadowOSSanitizeInput:
    def test_strips_script_tags(self):
        result = sanitize_input("<script>alert(1)</script> hello")
        assert "<script>" not in result
        assert "hello" in result

    def test_strips_javascript_uri(self):
        result = sanitize_input("javascript:alert(1)")
        assert "javascript:" not in result.lower()

    def test_strips_control_chars(self):
        result = sanitize_input("hello\x00world\x01test")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_max_length(self):
        result = sanitize_input("a" * 5000, max_length=100)
        assert len(result) <= 100


class TestShadowOSValidateObservation:
    def test_limits_keys(self):
        obs = {f"key_{i}": i for i in range(100)}
        result = validate_observation(obs)
        assert len(result) <= 50

    def test_truncates_string_values(self):
        obs = {"data": "x" * 20000}
        result = validate_observation(obs)
        assert len(result["data"]) <= 10000


class TestSelfImprovementDeepCopy:
    def test_prediction_to_dict_deepcopy(self):
        from shadowd.cognition.self_improvement import Prediction
        p = Prediction(content="test", metadata={"k": {"nested": 1}})
        d = p.to_dict()
        d["metadata"]["k"]["nested"] = 999
        assert p.metadata["k"]["nested"] == 1

    def test_decision_record_feedback_deepcopy(self):
        from shadowd.cognition.self_improvement import DecisionRecord, FeedbackSource
        d = DecisionRecord(action="test")
        d.add_feedback(FeedbackSource.EXPLICIT_USER, 0.8, "great")
        r = d.to_dict()
        r["feedback"].append({"fake": True})
        assert len(d.feedback) == 1


class TestDailyBriefingItemSanitization:
    def test_domain_strips_angle_brackets(self):
        from shadowd.cognition.daily_briefing import BriefingItem, BriefingSection
        item = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="content",
            domain="<script>evil</script>tech",
        )
        assert "<script>" not in item.domain
        assert "</script>" not in item.domain

    def test_action_url_strips_control_chars(self):
        from shadowd.cognition.daily_briefing import BriefingItem, BriefingSection
        item = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="content",
            action_url="http://example.com/\x00path",
        )
        assert "\x00" not in (item.action_url or "")


class TestAttentionEngineNoMutation:
    def test_get_priorities_does_not_mutate_items(self):
        from shadowd.cognition.attention_engine import AttentionEngine
        engine = AttentionEngine()
        item_id = engine.add_attention_item("test topic", level="high", domain="test")
        priorities = engine.get_priorities()
        assert len(priorities) == 1
        assert priorities[0].item_id == item_id
        with engine._lock:
            original = engine._attention_items[item_id]
            assert not hasattr(original, "_priority_score")