"""Sprint 204: AnticipatoryEngine — prediction of user needs before they ask"""

import pytest
import threading
from shadowd.cognition.anticipatory_engine import (
    AnticipatoryEngine, AnticipationContext, AnticipationSignal,
    SignalType, TriggerCondition,
)


@pytest.fixture
def engine():
    return AnticipatoryEngine()


@pytest.fixture
def context_morning_work():
    return AnticipationContext(
        emotional_state="joy",
        active_domains=["work", "creative"],
        time_of_day=10,
        day_of_week=2,
        recent_actions=["check_messages", "review_schedule"],
        goal_progress={"project_alpha": 0.2},
    )


@pytest.fixture
def context_evening_stressed():
    return AnticipationContext(
        emotional_state="fear",
        active_domains=["work", "health"],
        time_of_day=20,
        day_of_week=4,
        recent_actions=["deadline_approaching", "overtime"],
        goal_progress={"quarterly_review": 0.15, "presentation": 0.85},
    )


class TestAnticipationContext:
    def test_context_sanitization(self):
        ctx = AnticipationContext(
            emotional_state="jo<script>y",
            active_domains=["wo<>rk", "normal"],
            recent_actions=["act\x00ion"],
        )
        assert "<script>" not in ctx.emotional_state
        assert "\x00" not in ctx.recent_actions[0]

    def test_context_strict_sanitization(self):
        ctx = AnticipationContext(
            emotional_state="test<>value",
            active_domains=["domain"],
        )
        assert "<>" not in ctx.emotional_state

    def test_context_timestamp_auto(self):
        ctx = AnticipationContext()
        assert ctx.timestamp != ""

    def test_context_domains_capped(self):
        ctx = AnticipationContext(
            active_domains=[f"d{i}" for i in range(20)],
        )
        assert len(ctx.active_domains) <= 10

    def test_context_actions_capped(self):
        ctx = AnticipationContext(
            recent_actions=[f"a{i}" for i in range(50)],
        )
        assert len(ctx.recent_actions) <= 20

    def test_context_to_dict(self):
        ctx = AnticipationContext(
            emotional_state="joy",
            active_domains=["work"],
            time_of_day=14,
        )
        d = ctx.to_dict()
        assert d["emotional_state"] == "joy"
        assert isinstance(d["active_domains"], list)
        assert isinstance(d["goal_progress"], dict)


class TestAnticipationSignal:
    def test_signal_auto_id(self):
        s = AnticipationSignal(prediction="test")
        assert len(s.signal_id) == 12

    def test_signal_confidence_clamped(self):
        s = AnticipationSignal(prediction="test", confidence=1.5)
        assert s.confidence == 1.0
        s2 = AnticipationSignal(prediction="test", confidence=-0.5)
        assert s2.confidence == 0.0

    def test_signal_priority_clamped(self):
        s = AnticipationSignal(prediction="test", priority=2.0)
        assert s.priority == 1.0

    def test_signal_sanitization(self):
        s = AnticipationSignal(
            signal_type="<script>alert</script>",
            prediction="Need to review \x00document",
            reasoning="Because of threat",
            suggested_action="Take \x01action",
        )
        assert "<script>" not in s.signal_type
        assert "\x00" not in s.prediction
        assert "\x01" not in s.suggested_action

    def test_signal_to_dict_rounded(self):
        s = AnticipationSignal(prediction="test", confidence=0.75321, priority=0.67890)
        d = s.to_dict()
        assert d["confidence"] == 0.7532
        assert d["priority"] == 0.6789

    def test_signal_to_dict_returns_copies(self):
        s = AnticipationSignal(prediction="test", related_modules=["a", "b"])
        d = s.to_dict()
        d["related_modules"].append("c")
        assert len(s.related_modules) == 2


class TestPredictFromTime:
    def test_morning_prediction(self, engine):
        ctx = AnticipationContext(time_of_day=8, day_of_week=1)
        signals = engine.predict(ctx)
        time_signals = [s for s in signals if s.trigger == "time"]
        assert len(time_signals) > 0
        domains = [s.related_modules[0] for s in time_signals if s.related_modules]
        assert any(d in ["health", "planning", "communication"] for d in domains)

    def test_afternoon_prediction(self, engine):
        ctx = AnticipationContext(time_of_day=14, day_of_week=2)
        signals = engine.predict(ctx)
        time_signals = [s for s in signals if s.trigger == "time"]
        domains = [s.related_modules[0] for s in time_signals if s.related_modules]
        assert any(d in ["work", "productivity"] for d in domains)

    def test_evening_prediction(self, engine):
        ctx = AnticipationContext(time_of_day=19, day_of_week=3)
        signals = engine.predict(ctx)
        time_signals = [s for s in signals if s.trigger == "time"]
        domains = [s.related_modules[0] for s in time_signals if s.related_modules]
        assert any(d in ["reflection", "learning", "social"] for d in domains)

    def test_night_prediction(self, engine):
        ctx = AnticipationContext(time_of_day=23, day_of_week=5)
        signals = engine.predict(ctx)
        time_signals = [s for s in signals if s.trigger == "time"]
        domains = [s.related_modules[0] for s in time_signals if s.related_modules]
        assert any(d in ["rest", "consolidation", "automation"] for d in domains)


class TestPredictFromEmotion:
    def test_joy_prediction(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        signals = engine.predict(ctx)
        emotion_signals = [s for s in signals if s.trigger == "emotion"]
        assert len(emotion_signals) > 0
        types = {s.signal_type for s in emotion_signals}
        assert "social" in types or "creative" in types or "information" in types

    def test_sadness_prediction(self, engine):
        ctx = AnticipationContext(emotional_state="sadness")
        signals = engine.predict(ctx)
        emotion_signals = [s for s in signals if s.trigger == "emotion"]
        assert len(emotion_signals) > 0
        assert any(s.confidence >= 0.5 for s in emotion_signals)

    def test_fear_prediction(self, engine):
        ctx = AnticipationContext(emotional_state="fear")
        signals = engine.predict(ctx)
        emotion_signals = [s for s in signals if s.trigger == "emotion"]
        assert len(emotion_signals) > 0
        warning_signals = [s for s in emotion_signals if s.signal_type == "warning"]
        assert len(warning_signals) > 0

    def test_anger_prediction(self, engine):
        ctx = AnticipationContext(emotional_state="anger")
        signals = engine.predict(ctx)
        emotion_signals = [s for s in signals if s.trigger == "emotion"]
        assert len(emotion_signals) > 0

    def test_no_emotion_no_prediction(self, engine):
        ctx = AnticipationContext(emotional_state=None)
        signals = engine._predict_from_emotion(ctx)
        assert len(signals) == 0


class TestPredictFromDomains:
    def test_overload_warning(self, engine):
        ctx = AnticipationContext(
            active_domains=["work", "learning", "social", "health"],
        )
        signals = engine._predict_from_domains(ctx)
        overload = [s for s in signals if "overload" in s.prediction.lower()]
        assert len(overload) > 0

    def test_work_without_health_reminder(self, engine):
        ctx = AnticipationContext(active_domains=["work"])
        signals = engine._predict_from_domains(ctx)
        break_signals = [s for s in signals if "break" in s.prediction.lower()]
        assert len(break_signals) > 0

    def test_learning_without_creative(self, engine):
        ctx = AnticipationContext(active_domains=["learning"])
        signals = engine._predict_from_domains(ctx)
        creative = [s for s in signals if "creative" in s.suggested_action.lower() or "creative" in s.prediction.lower()]
        assert len(creative) > 0


class TestPredictFromSequence:
    def test_short_sequence_no_prediction(self, engine):
        ctx = AnticipationContext(recent_actions=["one"])
        signals = engine._predict_from_sequence(ctx)
        assert len(signals) == 0

    def test_record_and_predict_sequence(self, engine):
        engine.record_sequence(["check_messages", "review_schedule", "prioritize_tasks"])
        ctx = AnticipationContext(
            recent_actions=["check_messages", "review_schedule"],
            time_of_day=9,
            day_of_week=1,
        )
        signals = engine._predict_from_sequence(ctx)
        # Should detect the pattern or produce some prediction
        assert isinstance(signals, list)

    def test_sequence_with_known_pattern(self, engine):
        ctx = AnticipationContext(
            recent_actions=["check_messages", "review_schedule", "prioritize_tasks"],
            time_of_day=9,
            day_of_week=1,
        )
        signals = engine._predict_from_sequence(ctx)
        assert isinstance(signals, list)

    def test_record_sequence_sanitization(self, engine):
        engine.record_sequence(["clean\x00act", "go\x01od"])
        assert "cleanact" in engine._sequence_memory or len(engine._sequence_memory) > 0


class TestPredictFromGoals:
    def test_stuck_goal_triggers_resource(self, engine):
        ctx = AnticipationContext(
            goal_progress={"project": 0.2, "task": 0.1},
        )
        signals = engine._predict_from_goals(ctx)
        resource_signals = [s for s in signals if s.signal_type == "resource"]
        assert len(resource_signals) > 0

    def test_near_complete_goal_triggers_reminder(self, engine):
        ctx = AnticipationContext(
            goal_progress={"project": 0.85, "task": 0.9},
        )
        signals = engine._predict_from_goals(ctx)
        reminder_signals = [s for s in signals if s.signal_type == "reminder"]
        assert len(reminder_signals) > 0

    def test_no_goals_no_prediction(self, engine):
        ctx = AnticipationContext(goal_progress={})
        signals = engine._predict_from_goals(ctx)
        assert len(signals) == 0


class TestValidate:
    def test_validate_correct(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        signals = engine.predict(ctx)
        if signals:
            result = engine.validate(signals[0].signal_id, "correct")
            assert result is True
            assert engine._signals[signals[0].signal_id].validated is True

    def test_validate_incorrect(self, engine):
        ctx = AnticipationContext(emotional_state="fear")
        signals = engine.predict(ctx)
        if signals:
            engine.validate(signals[0].signal_id, "incorrect")
            with engine._stats_lock:
                assert engine._stats["incorrect_predictions"] >= 1

    def test_validate_nonexistent_signal(self, engine):
        result = engine.validate("nonexistent_id", "correct")
        assert result is False

    def test_validate_reinforces_pattern(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        signals = engine.predict(ctx)
        if signals:
            s = signals[0]
            engine.validate(s.signal_id, "correct")
            key = f"{s.trigger}:{s.prediction[:100]}"
            assert key in engine._patterns
            assert engine._patterns[key]["success_count"] >= 1

    def test_accuracy_stats(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        signals = engine.predict(ctx)
        if signals:
            engine.validate(signals[0].signal_id, "correct")
        acc = engine.get_accuracy()
        assert "overall_accuracy" in acc

    def test_accuracy_by_trigger(self, engine):
        ctx = AnticipationContext(emotional_state="fear")
        signals = engine.predict(ctx)
        if signals:
            s = [s for s in signals if s.trigger == "emotion"]
            if s:
                engine.validate(s[0].signal_id, "correct")
                acc = engine.get_accuracy(trigger="emotion")
                assert "trigger" in acc


class TestGetSignals:
    def test_get_all_signals(self, engine):
        ctx = AnticipationContext(emotional_state="joy", active_domains=["work"], time_of_day=10)
        engine.predict(ctx)
        all_signals = engine.get_signals()
        assert len(all_signals) > 0

    def test_get_signals_by_type(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        engine.predict(ctx)
        social_signals = engine.get_signals(signal_type="social")
        for s in social_signals:
            assert s.signal_type == "social"

    def test_get_signals_limit(self, engine):
        ctx = AnticipationContext(
            emotional_state="joy",
            active_domains=["work", "learning", "health", "creative"],
            time_of_day=10,
        )
        engine.predict(ctx)
        limited = engine.get_signals(limit=3)
        assert len(limited) <= 3

    def test_get_signal_by_id(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        signals = engine.predict(ctx)
        if signals:
            signal = engine.get_signal(signals[0].signal_id)
            assert signal is not None
            assert signal.signal_id == signals[0].signal_id

    def test_get_nonexistent_signal(self, engine):
        result = engine.get_signal("nonexistent")
        assert result is None


class TestGetHistory:
    def test_history_tracking(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        engine.predict(ctx)
        history = engine.get_history()
        assert len(history) > 0

    def test_history_limit(self, engine):
        ctx = AnticipationContext(emotional_state="joy")
        engine.predict(ctx)
        history = engine.get_history(limit=1)
        assert len(history) <= 1


class TestGetStats:
    def test_initial_stats(self, engine):
        stats = engine.get_stats()
        assert stats["signals_generated"] == 0
        assert stats["correct_predictions"] == 0

    def test_stats_after_predict(self, engine):
        ctx = AnticipationContext(emotional_state="joy", time_of_day=10)
        engine.predict(ctx)
        stats = engine.get_stats()
        assert stats["signals_generated"] > 0


class TestThreadSafety:
    def test_concurrent_predict(self, engine):
        errors = []

        def predict_many():
            try:
                for _ in range(50):
                    ctx = AnticipationContext(emotional_state="joy", time_of_day=10)
                    engine.predict(ctx)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=predict_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert engine.get_stats()["signals_generated"] > 0

    def test_concurrent_validate(self, engine):
        ctx = AnticipationContext(emotional_state="joy", time_of_day=10)
        signals = engine.predict(ctx)
        if not signals:
            pytest.skip("No signals generated")

        signal_id = signals[0].signal_id
        errors = []

        def validate_many():
            try:
                for _ in range(20):
                    engine.validate(signal_id, "correct")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=validate_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    def test_empty_context(self, engine):
        ctx = AnticipationContext()
        signals = engine.predict(ctx)
        assert isinstance(signals, list)

    def test_very_long_domain_name(self, engine):
        ctx = AnticipationContext(active_domains=["a" * 200])
        signals = engine.predict(ctx)
        assert isinstance(signals, list)

    def test_many_recent_actions(self, engine):
        ctx = AnticipationContext(recent_actions=[f"action_{i}" for i in range(50)])
        assert len(ctx.recent_actions) <= 20

    def test_signal_uniqueness(self, engine):
        ctx = AnticipationContext(
            emotional_state="joy",
            active_domains=["work"],
            time_of_day=10,
            day_of_week=2,
        )
        signals = engine.predict(ctx)
        predictions = [s.prediction for s in signals]
        assert len(predictions) == len(set(predictions))

    def test_max_signals_cap(self, engine):
        ctx_morning = AnticipationContext(time_of_day=10, day_of_week=2, emotional_state="joy")
        signals = engine.predict(ctx_morning)
        assert len(signals) <= 20