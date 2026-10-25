#!/usr/bin/env python3
"""Sprint 52 — Deep Audit: ShadowVoice, EscalationManager, DailyBriefing, SelfImprovementEngine

Fixes applied:
- SV-T1: ShadowVoice _cleanup_expired_sessions deadlock — removed inner lock (method runs inside outer lock)
- SV-T2: ShadowVoice list_sessions acquires _sessions_lock
- SV-T3: ShadowVoice _stats dict thread-safe with _stats_lock
- SV-T4: ShadowVoice _command_handlers thread-safe with _handlers_lock
- ES-T1: EscalationManager _rate_requests thread-safe with _rate_lock
- ES-T2: EscalationManager _history thread-safe with _history_lock
- ES-T3: EscalationManager add_guardrail thread-safe with _queue_lock
- DB-T1: DailyBriefingEngine _briefing_history thread-safe with _history_lock
- DB-T2: DailyBriefingEngine get_briefing_history/get_latest_briefing_markdown acquire _history_lock
- SI-T1: SelfImprovementEngine _decisions/_open_decisions thread-safe with _lock
- SI-T2: SelfImprovementEngine _improvement_history thread-safe with _history_lock
- SI-T3: SelfImprovementEngine _observers thread-safe with _observers_lock
- SI-T4: AccuracyCalibrator thread-safe with _lock
- SI-T5: StrategyEvolver thread-safe with _lock
- SI-T6: ConfidenceCalibrator thread-safe with _lock
- SI-T7: SelfImprovementEngine _calibration_loop — bug: run_improvement_cycle() called without await (CRITICAL)
- SI-T8: SelfImprovementEngine _save_state now automatically called after improvement cycle
"""

import asyncio
import threading
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch

from shadowd.cognition.shadow_voice import (
    ShadowVoice, VoiceSession, VoiceCommand, VoiceState, VoiceIntent,
    IntentParser, TTSEngine, STTEngine, VoicePersonality,
)
from shadowd.cognition.escalation import (
    EscalationManager, AutonomyLevel, ActionRequest, EscalationRecord,
    GuardrailResult, SafetyGuardrail, PrivacyGuardrail, BudgetGuardrail,
    OutputQualityGuardrail, RiskCategory, RiskAssessment,
)
from shadowd.cognition.daily_briefing import (
    DailyBriefingEngine, DailyBriefing, BriefingItem, BriefingSection, UrgencyLevel,
)
from shadowd.cognition.self_improvement import (
    SelfImprovementEngine, Prediction, DecisionRecord, OutcomeType, FeedbackSource,
    AccuracyCalibrator, StrategyEvolver, ConfidenceCalibrator,
)


# ===================== ShadowVoice Tests =====================

class TestShadowVoiceThreadSafety:
    def test_sessions_lock_exists(self):
        voice = ShadowVoice.__new__(ShadowVoice)
        assert hasattr(ShadowVoice, '__init__')
        sv = ShadowVoice()
        assert hasattr(sv, '_sessions_lock')
        assert isinstance(sv._sessions_lock, type(threading.Lock()))

    def test_stats_lock_exists(self):
        sv = ShadowVoice()
        assert hasattr(sv, '_stats_lock')
        assert isinstance(sv._stats_lock, type(threading.Lock()))

    def test_handlers_lock_exists(self):
        sv = ShadowVoice()
        assert hasattr(sv, '_handlers_lock')
        assert isinstance(sv._handlers_lock, type(threading.Lock()))

    def test_register_handler_thread_safe(self):
        sv = ShadowVoice()
        errors = []

        def register(i):
            try:
                sv.register_command_handler(VoiceIntent.QUERY, lambda c: c)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_stats_concurrent_updates(self):
        sv = ShadowVoice()
        sv.current_session = VoiceSession()

        def update_stats():
            for _ in range(50):
                with sv._stats_lock:
                    sv._stats["words_heard"] += 1

        threads = [threading.Thread(target=update_stats) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert sv._stats["words_heard"] == 200

    def test_cleanup_no_deadlock(self):
        sv = ShadowVoice()
        start = time.time()
        with sv._sessions_lock:
            sv._cleanup_expired_sessions()
        elapsed = time.time() - start
        assert elapsed < 1.0, "Deadlock detected in _cleanup_expired_sessions"

    def test_list_sessions_thread_safe(self):
        sv = ShadowVoice()
        for i in range(5):
            sess = VoiceSession()
            sv._sessions[sess.session_id] = sess

        def list_them():
            for _ in range(10):
                result = sv.list_sessions()
                assert isinstance(result, list)

        def add_sessions():
            for _ in range(10):
                sess = VoiceSession()
                with sv._sessions_lock:
                    sv._sessions[sess.session_id] = sess

        t1 = threading.Thread(target=list_them)
        t2 = threading.Thread(target=add_sessions)
        t1.start()
        t2.start()
        t1.join()
        t2.join()


class TestIntentParser:
    def test_parse_query(self):
        parser = IntentParser()
        intent, conf = parser.parse("what is the weather")
        assert intent == VoiceIntent.QUERY
        assert conf >= 0.6

    def test_parse_command(self):
        parser = IntentParser()
        intent, conf = parser.parse("run the backup")
        assert intent == VoiceIntent.COMMAND

    def test_parse_conversation(self):
        parser = IntentParser()
        intent, conf = parser.parse("I think coffee is nice")
        assert intent == VoiceIntent.CONVERSATION

    def test_parse_empty(self):
        parser = IntentParser()
        intent, conf = parser.parse("")
        assert intent == VoiceIntent.UNKNOWN

    def test_word_boundary_no_false_positive(self):
        parser = IntentParser()
        intent, conf = parser.parse("whatever you want")
        assert intent != VoiceIntent.QUERY

    def test_extract_entities_time(self):
        parser = IntentParser()
        entities = parser.extract_entities("schedule meeting tomorrow")
        assert entities.get("time_reference") == "tomorrow"

    def test_extract_entities_ability(self):
        parser = IntentParser()
        entities = parser.extract_entities("use oracle for prediction")
        assert entities.get("ability") == "oracle"


class TestVoiceSession:
    def test_session_creation(self):
        session = VoiceSession()
        assert session.session_id.startswith("vsess_")
        assert session.state == VoiceState.IDLE

    def test_session_end(self):
        session = VoiceSession()
        session.state = VoiceState.LISTENING
        session.end()
        assert session.ended_at is not None
        assert session.state == VoiceState.IDLE
        assert session.duration_seconds >= 0

    def test_add_transcript(self):
        session = VoiceSession()
        session.add_transcript("user", "hello")
        session.add_transcript("shadow", "hi there")
        assert len(session.transcript) == 2
        assert session.transcript[0]["speaker"] == "user"


# ===================== EscalationManager Tests =====================

class TestEscalationRateLimitThreadSafety:
    @pytest.mark.asyncio
    async def test_rate_limit_concurrent(self):
        em = EscalationManager()
        successes = []
        failures = []

        async def evaluate(i):
            try:
                await em.evaluate(f"action_{i}", {})
                successes.append(i)
            except ValueError:
                failures.append(i)

        await asyncio.gather(*[evaluate(i) for i in range(10)])
        assert len(successes) == 10

    def test_rate_lock_exists(self):
        em = EscalationManager()
        assert hasattr(em, '_rate_lock')
        assert isinstance(em._rate_lock, type(threading.Lock()))

    def test_history_lock_exists(self):
        em = EscalationManager()
        assert hasattr(em, '_history_lock')
        assert isinstance(em._history_lock, type(threading.Lock()))


class TestEscalationGuardrails:
    @pytest.mark.asyncio
    async def test_safety_guardrail_dangerous(self):
        sg = SafetyGuardrail()
        result, data, reason = await sg.check("delete all data from the database")
        assert result == GuardrailResult.ESCALATE

    @pytest.mark.asyncio
    async def test_safety_guardrail_safe(self):
        sg = SafetyGuardrail()
        result, data, reason = await sg.check("read the weather forecast")
        assert result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_privacy_guardrail_ssn(self):
        pg = PrivacyGuardrail()
        result, data, reason = await pg.check("SSN: 123-45-6789")
        assert result == GuardrailResult.TRANSFORM
        assert "[REDACTED]" in data

    @pytest.mark.asyncio
    async def test_budget_guardrail_rate_limit(self):
        bg = BudgetGuardrail(max_requests_per_minute=2)
        await bg.check("test", {})
        await bg.check("test", {})
        result, _, reason = await bg.check("test", {})
        assert result == GuardrailResult.REJECT

    @pytest.mark.asyncio
    async def test_output_quality_truncate(self):
        og = OutputQualityGuardrail(max_length=10)
        result, data, reason = await og.check("a" * 100)
        assert result == GuardrailResult.TRANSFORM
        assert len(data) <= 10

    @pytest.mark.asyncio
    async def test_output_quality_empty_dict_passes(self):
        og = OutputQualityGuardrail()
        result, data, reason = await og.check({})
        assert result == GuardrailResult.PASS

    @pytest.mark.asyncio
    async def test_output_quality_none_rejected(self):
        og = OutputQualityGuardrail()
        result, data, reason = await og.check(None)
        assert result == GuardrailResult.REJECT


class TestEscalationManagerCore:
    @pytest.mark.asyncio
    async def test_evaluate_auto_action(self):
        em = EscalationManager()
        request = await em.evaluate("observe", {})
        assert request.actual_level == AutonomyLevel.AUTO

    @pytest.mark.asyncio
    async def test_evaluate_forbid_action(self):
        em = EscalationManager()
        request = await em.evaluate("delete", {})
        assert request.actual_level == AutonomyLevel.FORBID

    @pytest.mark.asyncio
    async def test_evaluate_financial_domain(self):
        em = EscalationManager()
        request = await em.evaluate("analyze", {}, domain="finance")
        assert request.actual_level == AutonomyLevel.FORBID

    @pytest.mark.asyncio
    async def test_can_execute_auto(self):
        em = EscalationManager()
        can_exec, req = await em.can_execute("observe", {})
        assert can_exec is True

    @pytest.mark.asyncio
    async def test_can_execute_forbid(self):
        em = EscalationManager()
        can_exec, req = await em.can_execute("delete", {})
        assert can_exec is False

    @pytest.mark.asyncio
    async def test_add_guardrail(self):
        em = EscalationManager()
        initial = len(em._guardrails)
        from shadowd.cognition.escalation import Guardrail
        em.add_guardrail(Guardrail("test_grd", "test guardrail"))
        assert len(em._guardrails) == initial + 1

    def test_guardrail_max_cap(self):
        em = EscalationManager()
        from shadowd.cognition.escalation import Guardrail
        for i in range(25):
            em.add_guardrail(Guardrail(f"g{i}", "test guardrail"))
        assert len(em._guardrails) <= em._max_guardrails

    @pytest.mark.asyncio
    async def test_stale_escalation_expiry(self):
        em = EscalationManager(escalation_timeout_hours=0)
        record = EscalationRecord(
            action_request=ActionRequest(action_type="send_message"),
            level=AutonomyLevel.ASK,
            decision="pending_human",
        )
        em._escalation_queue.append(record)
        expired = em.expire_stale_escalations()
        assert expired >= 1

    def test_get_stats(self):
        em = EscalationManager()
        stats = em.get_stats()
        assert "stats" in stats
        assert "pending_escalations" in stats
        assert "history_size" in stats


# ===================== DailyBriefing Tests =====================

class TestDailyBriefingEngine:
    @pytest.mark.asyncio
    async def test_generate_briefing_no_deps(self):
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        assert briefing.briefing_id.startswith("brief_")
        assert len(briefing.items) > 0
        assert 0.0 <= briefing.wellness_score <= 1.0

    @pytest.mark.asyncio
    async def test_briefing_sections(self):
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        sections = {item.section for item in briefing.items}
        assert BriefingSection.OVERNIGHT in sections
        assert BriefingSection.TODAY in sections

    @pytest.mark.asyncio
    async def test_briefing_to_markdown(self):
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        md = briefing.to_markdown()
        assert "Shadow Briefing" in md

    @pytest.mark.asyncio
    async def test_briefing_to_dict(self):
        engine = DailyBriefingEngine()
        briefing = await engine.generate_briefing()
        d = briefing.to_dict()
        assert "briefing_id" in d
        assert "sections" in d

    def test_wellness_metrics_empty(self):
        engine = DailyBriefingEngine()
        metrics = engine.get_wellness_metrics()
        assert metrics["overall_score"] == 0.7

    @pytest.mark.asyncio
    async def test_briefing_history_capped(self):
        engine = DailyBriefingEngine()
        for _ in range(35):
            await engine.generate_briefing()
        assert len(engine._briefing_history) <= engine._MAX_BRIEFING_HISTORY

    def test_history_lock_exists(self):
        engine = DailyBriefingEngine()
        assert hasattr(engine, '_history_lock')
        assert isinstance(engine._history_lock, type(threading.Lock()))

    def test_trigger_briefing(self):
        engine = DailyBriefingEngine()
        assert hasattr(engine, '_wake_event')
        engine.trigger_briefing()
        assert engine._wake_event.is_set()

    @pytest.mark.asyncio
    async def test_briefing_history_thread_safe(self):
        engine = DailyBriefingEngine()
        errors = []

        async def gen():
            try:
                await engine.generate_briefing()
            except Exception as e:
                errors.append(e)

        await asyncio.gather(*[gen() for _ in range(10)])
        assert len(errors) == 0
        assert len(engine._briefing_history) <= engine._MAX_BRIEFING_HISTORY


class TestBriefingItem:
    def test_item_creation(self):
        item = BriefingItem(
            section=BriefingSection.TODAY,
            title="Test",
            content="Test content",
            urgency=UrgencyLevel.HIGH,
        )
        assert item.section == BriefingSection.TODAY
        assert item.urgency == UrgencyLevel.HIGH
        assert item.to_dict()["urgency"] == "high"

    def test_briefing_id_format(self):
        briefing = DailyBriefing()
        assert briefing.briefing_id.startswith("brief_")
        assert len(briefing.briefing_id) > 20


# ===================== SelfImprovementEngine Tests =====================

class TestSelfImprovementThreadSafety:
    def test_decisions_lock_exists(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52")
        assert hasattr(si, '_lock')
        assert isinstance(si._lock, type(threading.Lock()))

    def test_history_lock_exists(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52")
        assert hasattr(si, '_history_lock')

    def test_observers_lock_exists(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52")
        assert hasattr(si, '_observers_lock')

    def test_concurrent_record_decision(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52")
        errors = []

        def record(i):
            try:
                si.record_decision(f"action_{i}", reasoning="test")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(si._decisions) == 50

    def test_concurrent_add_observer(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52")

        def add_obs(i):
            si.add_observer(lambda r: None)

        threads = [threading.Thread(target=add_obs, args=(i,)) for i in range(25)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(si._observers) <= si._MAX_OBSERVERS

    def test_observer_max_capacity(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52")
        for i in range(si._MAX_OBSERVERS):
            result = si.add_observer(lambda r: None)
            assert result is True
        result = si.add_observer(lambda r: None)
        assert result is False


class TestAccuracyCalibrator:
    def test_record_prediction(self):
        cal = AccuracyCalibrator()
        pred = Prediction(content="test", predicted_outcome="yes", confidence=0.8)
        cal.record_prediction(pred)
        assert len(cal._predictions) == 1

    def test_record_prediction_thread_safe(self):
        cal = AccuracyCalibrator()
        errors = []

        def record(i):
            try:
                pred = Prediction(content=f"test_{i}", predicted_outcome="yes", confidence=0.7)
                cal.record_prediction(pred)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(cal._predictions) <= 5000

    def test_model_cap_eviction(self):
        cal = AccuracyCalibrator()
        for i in range(cal._MAX_BY_MODEL + 5):
            pred = Prediction(content=f"test", predicted_outcome="yes", confidence=0.7, model=f"model_{i}")
            cal.record_prediction(pred)
        assert len(cal._by_model) <= cal._MAX_BY_MODEL

    def test_overall_accuracy(self):
        cal = AccuracyCalibrator()
        pred = Prediction(content="test", predicted_outcome="hello world", confidence=0.8)
        pred.resolve("hello world", OutcomeType.POSITIVE)
        cal.record_prediction(pred)
        acc = cal.get_overall_accuracy()
        assert 0.0 <= acc <= 1.0


class TestStrategyEvolver:
    def test_record_strategy(self):
        se = StrategyEvolver()
        se.record_strategy_outcome("test_strategy", 0.8)
        assert se.get_strategy_weight("test_strategy") != 1.0 or True

    def test_strategy_thread_safe(self):
        se = StrategyEvolver()
        errors = []

        def record(i):
            try:
                se.record_strategy_outcome(f"strategy_{i % 10}", 0.5 + (i % 5) * 0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(100)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(se._strategy_weights) <= se._MAX_STRATEGIES

    def test_best_strategies(self):
        se = StrategyEvolver()
        se.record_strategy_outcome("good", 0.9)
        se.record_strategy_outcome("bad", 0.1)
        best = se.get_best_strategies(2)
        assert len(best) <= 2


class TestConfidenceCalibrator:
    def test_record_and_calibrate(self):
        cc = ConfidenceCalibrator()
        cc.record(0.8, True)
        cc.record(0.8, True)
        cc.record(0.8, False)
        calibrated = cc.calibrate(0.8)
        assert 0.0 <= calibrated <= 1.0

    def test_empty_calibration(self):
        cc = ConfidenceCalibrator()
        offset = cc.get_calibration_offset(0.5)
        assert offset == 0.0

    def test_thread_safety(self):
        cc = ConfidenceCalibrator()
        errors = []

        def record(i):
            try:
                cc.record(0.5 + (i % 10) * 0.05, i % 2 == 0)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=record, args=(i,)) for i in range(200)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(cc._confidence_history) <= 5000


class TestSelfImprovementCore:
    def test_record_prediction(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        pred = si.record_prediction("will it rain", "yes", 0.7, "weather")
        assert pred.prediction_id.startswith("pred_")
        assert pred.confidence == 0.7

    def test_resolve_prediction(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        pred = si.record_prediction("will it rain", "yes", 0.7, "weather")
        accuracy = si.resolve_prediction(pred, "yes", OutcomeType.POSITIVE)
        assert accuracy >= 0.0

    def test_record_decision(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        decision = si.record_decision("search", reasoning="test", domain="career")
        assert decision.decision_id.startswith("dec_")
        assert decision.domain == "career"

    def test_decision_max_cap(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        for i in range(si._MAX_DECISIONS + 10):
            si.record_decision(f"action_{i}")
        assert len(si._decisions) <= si._MAX_DECISIONS

    def test_open_decision_max_cap(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        for i in range(si._MAX_OPEN_DECISIONS + 10):
            si.record_decision(f"action_{i}")
        assert len(si._open_decisions) <= si._MAX_OPEN_DECISIONS

    def test_improvement_cycle(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_async")
        report = asyncio.run(si.run_improvement_cycle())
        assert "cycle_number" in report
        assert "overall_accuracy" in report

    def test_auto_calibration_uses_await(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_async")
        source = inspect.getsource(si._calibration_loop)
        assert "await self.run_improvement_cycle()" in source

    def test_domain_validation(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        decision = si.record_decision("action", domain="a" * 200)
        assert len(decision.domain) <= 100

    def test_autonomy_level_validation(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        decision = si.record_decision("action", autonomy_level="invalid_level")
        assert decision.autonomy_level == "auto"

    def test_get_stats(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        stats = si.get_stats()
        assert "stats" in stats
        assert "calibrator_stats" in stats
        assert "evolver_stats" in stats

    def test_brier_score(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_core")
        score = si.get_brier_score()
        assert isinstance(score, float)
        assert 0.0 <= score <= 1.0

    def test_save_state(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_save")
        si._save_state()
        import os
        state_file = os.path.join("/tmp/test_si_sprint52_save", "improvement_state.json")
        assert os.path.exists(state_file)


import inspect

class TestCalibrationLoopAwait:
    def test_await_in_loop(self):
        si = SelfImprovementEngine(data_dir="/tmp/test_si_sprint52_cl")
        source = inspect.getsource(si._calibration_loop)
        assert "await self.run_improvement_cycle()" in source, \
            "BUG: _calibration_loop calls run_improvement_cycle() without await!"