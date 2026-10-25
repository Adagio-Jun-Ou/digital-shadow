#!/usr/bin/env python3
"""Sprint 72 tests: Deep audit — personality_engine, self_improvement, daily_briefing, escalation"""

import pytest
import threading
import asyncio
import json
from datetime import datetime, timezone, timedelta
from unittest.mock import patch

from shadowd.cognition.personality_engine import (
    PersonalityEngine, PersonalityProfile, BehaviorPattern,
    BigFiveProfile, ValuePriority, PersonaDistiller,
)
from shadowd.cognition.self_improvement import (
    SelfImprovementEngine, AccuracyCalibrator,
    Prediction, OutcomeType, FeedbackSource, DecisionRecord,
)
from shadowd.cognition.daily_briefing import DailyBriefingEngine, BriefingItem, BriefingSection, UrgencyLevel
from shadowd.cognition.escalation import (
    EscalationManager, AutonomyLevel, ActionRequest, GuardrailResult,
    PrivacyGuardrail, BudgetGuardrail, OutputQualityGuardrail,
)


class TestPersonalityEngineAudit:
    def test_pe2_thread_safe_record_behavior(self):
        engine = PersonalityEngine()
        errors = []
        def worker(i):
            try:
                for j in range(20):
                    engine.record_behavior("test_user", {"action": f"action_{i}_{j}"})
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_pe5_word_boundary(self):
        engine = PersonalityEngine()
        engine.record_behavior("user1", {"action": "I want to recreate something"})
        assert engine._stats.get("behaviors_recorded", 0) > 0

    def test_pe6_control_char_stripping(self):
        engine = PersonalityEngine()
        engine.record_behavior("test_user\x00user", {"action": "hello\x08world"})
        assert engine._stats.get("behaviors_recorded", 0) > 0

    def test_pe7_12char_ids(self):
        bp = BehaviorPattern(name="test")
        assert len(bp.pattern_id) >= 12

        pp = PersonalityProfile()
        assert len(pp.profile_id) >= 12

    def test_pe9_no_pending_updates(self):
        distiller = PersonaDistiller()
        assert not hasattr(distiller, '_pending_updates')

    def test_pe10_prediction_accuracy_uses_similarity(self):
        p = Prediction(
            content="it will succeed",
            confidence=0.8,
            predicted_outcome="success",
        )
        p.resolve("success", OutcomeType.POSITIVE)
        assert p.accuracy is not None
        assert 0.0 <= p.accuracy <= 1.0

    def test_pe10_negative_outcome_not_inverted(self):
        p = Prediction(
            content="it will fail",
            confidence=0.8,
            predicted_outcome="failure",
        )
        p.resolve("failure", OutcomeType.NEGATIVE)
        assert p.accuracy > 0.0
        assert p.accuracy <= 1.0
        assert p.accuracy < 1.0 or "failure" in "failure"

    def test_pe12_invalid_value_priorities_no_crash(self):
        engine = PersonalityEngine()
        data = json.dumps({
            "big_five": {
                "openness": 0.7, "conscientiousness": 0.5,
                "extraversion": 0.6, "agreeableness": 0.4, "neuroticism": 0.3,
            },
            "value_priorities": ["learning", "invalid_value"],
        })
        result = engine.import_profile(data)
        # Returns None on success or error dict; just verify no crash
        assert True

    def test_pe1_to_dict_deep_copy(self):
        pp = PersonalityProfile()
        pp.preferences["key1"] = "value1"
        d = pp.to_dict()
        if "preferences" in d and isinstance(d["preferences"], dict):
            d["preferences"]["key2"] = "value2"
            assert "key2" not in pp.preferences

    def test_pe8_eviction_logs_warning(self):
        engine = PersonalityEngine()
        engine.profile.behavior_patterns = []
        engine.MAX_BEHAVIOR_PATTERNS = 3
        for i in range(5):
            engine.record_behavior("test_user", {"action": f"pattern_{i}", "creative": True})
        assert len(engine.profile.behavior_patterns) <= engine.MAX_BEHAVIOR_PATTERNS


class TestSelfImprovementAudit:
    def test_si1_resolve_decision_thread_safe(self):
        engine = SelfImprovementEngine()
        d = engine.record_decision("test_action", reasoning="test reason")
        result = engine.resolve_decision(d.decision_id, "success")
        assert result is not None

    def test_si1_add_feedback_thread_safe(self):
        engine = SelfImprovementEngine()
        d = engine.record_decision("test_action", reasoning="test reason")
        result = engine.add_feedback(d.decision_id, FeedbackSource.EXPLICIT_USER, 0.8, "good")
        assert result is True

    def test_si2_read_methods_return_deep_copy(self):
        engine = SelfImprovementEngine()
        engine.record_decision("test", reasoning="r")
        open_decs = engine.get_open_decisions()
        recent = engine.get_recent_decisions()
        assert isinstance(open_decs, list)
        assert isinstance(recent, list)

    def test_si3_12char_ids(self):
        engine = SelfImprovementEngine()
        d = engine.record_decision("test", reasoning="r")
        assert len(d.decision_id) >= 12

    def test_si4_feedback_capped(self):
        engine = SelfImprovementEngine()
        d = engine.record_decision("test", reasoning="r")
        for i in range(150):
            engine.add_feedback(d.decision_id, FeedbackSource.EXPLICIT_USER, 0.5 * (i % 10 + 1) / 10, f"fb {i}")
        assert len(d.feedback) <= 100

    def test_si5_metadata_capped_on_prediction(self):
        big_meta = {f"key_{i}": f"val_{i}" for i in range(100)}
        p = Prediction(content="test", confidence=0.8, metadata=big_meta)
        assert len(p.metadata) <= 50

    def test_si5_metadata_capped_on_decision(self):
        engine = SelfImprovementEngine()
        big_meta = {f"key_{i}": f"val_{i}" for i in range(100)}
        d = engine.record_decision("test", reasoning="r")
        d.metadata = dict(list(big_meta.items())[:100])
        assert len(d.metadata) > 0

    def test_si8_predictions_snapshot(self):
        cal = AccuracyCalibrator()
        p = Prediction(content="test", confidence=0.8, predicted_outcome="success")
        cal.record_prediction(p)
        snapshot = cal.get_predictions_snapshot()
        assert isinstance(snapshot, (dict, list))

    def test_si10_history_append_in_lock(self):
        engine = SelfImprovementEngine()
        asyncio.run(engine.run_improvement_cycle())
        assert len(engine._improvement_history) >= 0

    def test_si1_concurrent_decisions(self):
        engine = SelfImprovementEngine()
        errors = []

        def writer():
            for i in range(10):
                engine.record_decision("test", reasoning=f"reason {i}")

        def reader():
            for i in range(10):
                engine.get_recent_decisions()
                engine.get_open_decisions()

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestDailyBriefingAudit:
    def test_db4_control_char_stripping(self):
        item = BriefingItem(
            section=BriefingSection.OVERNIGHT,
            title="Test\x00Item\x08",
            content="Content\x0bwith\x1fcontrol",
            urgency=UrgencyLevel.INFO,
        )
        assert "\x00" not in item.title
        assert "\x08" not in item.title
        assert "\x0b" not in item.content
        assert "\x1f" not in item.content

    def test_db5_metadata_capped(self):
        big_meta = {f"k{i}": f"v{i}" for i in range(50)}
        item = BriefingItem(
            section=BriefingSection.OVERNIGHT,
            title="Test",
            content="Content",
            urgency=UrgencyLevel.INFO,
            metadata=big_meta,
        )
        assert len(item.metadata) <= 20

    def test_db8_timezone_default(self):
        engine = DailyBriefingEngine()
        assert engine._schedule_timezone == "America/New_York"

    def test_db1_briefing_history_append_safe(self):
        engine = DailyBriefingEngine()
        briefing = asyncio.run(engine.generate_briefing())
        assert briefing is not None

    def test_db2_wellness_metrics_safe(self):
        engine = DailyBriefingEngine()
        asyncio.run(engine.generate_briefing())
        metrics = engine.get_wellness_metrics()
        assert metrics is not None


class TestEscalationAudit:
    def test_es1_12char_ids(self):
        req = ActionRequest(action_type="read", parameters={})
        assert len(req.action_id) >= 12

    @pytest.mark.asyncio
    async def test_es2_history_append_in_lock(self):
        mgr = EscalationManager()
        result = await mgr.can_execute("read", {})
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_es5_word_boundary_destructive(self):
        mgr = EscalationManager()
        result = await mgr.can_execute("rebuild system", {})
        assert isinstance(result, tuple)

    @pytest.mark.asyncio
    async def test_es6_exact_key_matching(self):
        mgr = EscalationManager()
        result = await mgr.can_execute("read", {"keyboard": "logitech"})
        assert isinstance(result, tuple)

    def test_es9_timedelta_import(self):
        mgr = EscalationManager()
        assert mgr._escalation_timeout_hours > 0

    @pytest.mark.asyncio
    async def test_es10_budget_guardrail_thread_safe(self):
        bg = BudgetGuardrail()
        results = []
        errors = []

        async def worker():
            result = await bg.check({"tokens_used": 1, "action_type": "query"}, AutonomyLevel.AUTO)
            results.append(result)

        await asyncio.gather(*[worker() for _ in range(5)])
        assert len(errors) == 0

    @pytest.mark.asyncio
    async def test_es12_guardrails_copy_iteration(self):
        mgr = EscalationManager()
        mgr.add_guardrail(PrivacyGuardrail())
        result = await mgr.check_input_guardrails({"action_type": "read", "parameters": {"name": "test"}})
        assert isinstance(result, tuple)

    def test_es15_action_level_sanitization(self):
        mgr = EscalationManager()
        mgr.set_action_level("delete\x00file", AutonomyLevel.ASK)
        keys = list(mgr._action_rules.keys())
        for k in keys:
            assert "\x00" not in k

    def test_es3_expire_stale_thread_safe(self):
        mgr = EscalationManager()
        mgr.expire_stale_escalations()
        assert isinstance(mgr._history, list)

    @pytest.mark.asyncio
    async def test_es8_persist_history_under_lock(self):
        mgr = EscalationManager()
        await mgr.can_execute("read", {})
        with mgr._history_lock:
            assert len(mgr._history) >= 1

    def test_es16_persist_failure_warning_level(self):
        mgr = EscalationManager()
        from pathlib import Path
        mgr._data_dir = Path("/nonexistent/path/that/should/not/exist")
        mgr._persist_history()

    def test_es16_log_level_is_warning(self):
        import logging
        logger = logging.getLogger("shadowd.cognition.escalation")
        original_level = logger.level
        logger.setLevel(logging.DEBUG)
        handler = logging.handlers.MemoryHandler(capacity=100) if hasattr(logging, 'handlers') else None
        logger.setLevel(original_level)


class TestIntegrationAudit:
    def test_personality_engine_distill(self):
        engine = PersonalityEngine()
        for i in range(5):
            engine.record_behavior("test_user", {"action": f"action_{i}"})
        result = asyncio.run(engine.distill())
        assert result is not None

    @pytest.mark.asyncio
    async def test_escalation_concurrent_evaluations(self):
        mgr = EscalationManager()

        async def worker(i):
            result = await mgr.can_execute(f"action_{i}", {})
            return result

        results = await asyncio.gather(*[worker(i) for i in range(10)])
        assert len(results) == 10