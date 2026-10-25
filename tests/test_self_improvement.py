#!/usr/bin/env python3
"""
Tests for Self-Improvement Engine
"""

import asyncio
import tempfile
import pytest
from pathlib import Path

from shadowd.cognition.self_improvement import (
    SelfImprovementEngine, Prediction, DecisionRecord,
    AccuracyCalibrator, StrategyEvolver, ConfidenceCalibrator,
    OutcomeType, FeedbackSource,
)


class TestPrediction:
    def test_creation(self):
        pred = Prediction(content="Test", predicted_outcome="Positive", confidence=0.8)
        assert pred.prediction_id.startswith("pred_")
        assert pred.confidence == 0.8
        assert pred.outcome_type == OutcomeType.PENDING

    def test_resolve(self):
        pred = Prediction(content="Test", predicted_outcome="positive result")
        pred.resolve("positive result", OutcomeType.POSITIVE)
        assert pred.outcome_type == OutcomeType.POSITIVE
        assert pred.actual_outcome == "positive result"
        assert pred.accuracy is not None
        assert pred.resolved_at is not None

    def test_to_dict(self):
        pred = Prediction(content="Test prediction", predicted_outcome="Good")
        d = pred.to_dict()
        assert d["prediction_id"].startswith("pred_")
        assert d["confidence"] == 0.5

    def test_text_similarity(self):
        pred = Prediction()
        assert pred._text_similarity("hello world", "hello world") == 1.0
        assert pred._text_similarity("hello", "world") == 0.0
        assert 0 < pred._text_similarity("hello world", "hello there") < 1.0

    def test_empty_similarity(self):
        pred = Prediction()
        assert pred._text_similarity("", "test") == 0.0
        assert pred._text_similarity("test", "") == 0.0


class TestDecisionRecord:
    def test_creation(self):
        dec = DecisionRecord(action="test_action", reasoning="because", confidence=0.9)
        assert dec.decision_id.startswith("dec_")
        assert dec.confidence == 0.9

    def test_resolve(self):
        dec = DecisionRecord(action="test")
        dec.resolve("Good outcome", OutcomeType.POSITIVE)
        assert dec.actual_outcome == "Good outcome"
        assert dec.outcome_type == OutcomeType.POSITIVE
        assert dec.resolved_at is not None

    def test_feedback(self):
        dec = DecisionRecord(action="test")
        dec.add_feedback(FeedbackSource.EXPLICIT_USER, 0.8, "Good job")
        assert len(dec.feedback) == 1
        assert dec.feedback[0]["rating"] == 0.8
        assert dec.get_average_rating() == 0.8

    def test_multiple_feedback(self):
        dec = DecisionRecord(action="test")
        dec.add_feedback(FeedbackSource.EXPLICIT_USER, 1.0, "Perfect")
        dec.add_feedback(FeedbackSource.IMPLICIT_BEHAVIOR, 0.6, "Decent")
        assert abs(dec.get_average_rating() - 0.8) < 0.01

    def test_to_dict(self):
        dec = DecisionRecord(action="test_action", confidence=0.7)
        d = dec.to_dict()
        assert d["decision_id"].startswith("dec_")
        assert d["confidence"] == 0.7


class TestAccuracyCalibrator:
    def test_empty(self):
        cal = AccuracyCalibrator()
        assert cal.get_overall_accuracy() == 0.5

    def test_record_and_accuracy(self):
        cal = AccuracyCalibrator()
        pred = Prediction(content="Test", confidence=0.8, model="test_model", domain="career")
        pred.accuracy = 0.9
        cal.record_prediction(pred)
        assert cal.get_overall_accuracy() == 0.9

    def test_accuracy_by_model(self):
        cal = AccuracyCalibrator()
        p1 = Prediction(content="T1", confidence=0.7, model="model_a")
        p1.accuracy = 0.9
        cal.record_prediction(p1)
        p2 = Prediction(content="T2", confidence=0.7, model="model_b")
        p2.accuracy = 0.5
        cal.record_prediction(p2)
        assert cal.get_accuracy_by_model("model_a") == 0.9
        assert cal.get_accuracy_by_model("model_b") == 0.5

    def test_accuracy_by_domain(self):
        cal = AccuracyCalibrator()
        p = Prediction(content="T", domain="finance")
        p.accuracy = 0.7
        cal.record_prediction(p)
        assert cal.get_accuracy_by_domain("finance") == 0.7

    def test_calibration(self):
        cal = AccuracyCalibrator()
        for i in range(10):
            p = Prediction(content=f"T{i}", confidence=0.8)
            p.accuracy = 0.8
            cal.record_prediction(p)
        calibration = cal.get_confidence_calibration()
        assert isinstance(calibration, dict)

    def test_calibration_error(self):
        cal = AccuracyCalibrator()
        error = cal.get_calibration_error()
        assert isinstance(error, float)

    def test_stats(self):
        cal = AccuracyCalibrator()
        stats = cal.get_stats()
        assert "total_predictions" in stats
        assert "overall_accuracy" in stats


class TestStrategyEvolver:
    def test_record_and_get(self):
        evolver = StrategyEvolver(learning_rate=0.1)
        for _ in range(5):
            evolver.record_strategy_outcome("strategy_a", 1.0)
        weight = evolver.get_strategy_weight("strategy_a")
        assert weight >= 1.0

    def test_negative_outcome(self):
        evolver = StrategyEvolver(learning_rate=0.1)
        evolver.record_strategy_outcome("strategy_b", 0.0)
        weight = evolver.get_strategy_weight("strategy_b")
        assert weight < 1.0

    def test_best_strategies(self):
        evolver = StrategyEvolver()
        for _ in range(5):
            evolver.record_strategy_outcome("good", 1.0)
            evolver.record_strategy_outcome("bad", 0.0)
        best = evolver.get_best_strategies(2)
        assert len(best) <= 2
        if best:
            assert best[0][0] == "good"

    def test_default_weight(self):
        evolver = StrategyEvolver()
        assert evolver.get_strategy_weight("nonexistent") == 1.0

    def test_stats(self):
        evolver = StrategyEvolver()
        stats = evolver.get_stats()
        assert "strategies_count" in stats
        assert "learning_rate" in stats


class TestConfidenceCalibrator:
    def test_empty(self):
        cal = ConfidenceCalibrator()
        calibrated = cal.calibrate(0.8)
        assert calibrated == 0.8

    def test_record_and_calibrate(self):
        cal = ConfidenceCalibrator()
        cal.record(0.8, True)
        cal.record(0.8, True)
        cal.record(0.8, False)
        calibrated = cal.calibrate(0.8)
        assert 0.0 <= calibrated <= 1.0

    def test_stats(self):
        cal = ConfidenceCalibrator()
        cal.record(0.7, True)
        stats = cal.get_stats()
        assert "total_records" in stats
        assert stats["total_records"] == 1


class TestSelfImprovementEngine:
    def test_init(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            stats = engine.get_stats()
            assert "stats" in stats

    def test_record_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            pred = engine.record_prediction("Project will succeed", "Project succeeds", confidence=0.7, domain="career")
            assert pred.prediction_id.startswith("pred_")
            assert engine._stats["predictions_made"] == 1

    def test_resolve_prediction(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            pred = engine.record_prediction("Test", "positive", confidence=0.8)
            accuracy = engine.resolve_prediction(pred, "positive result", OutcomeType.POSITIVE)
            assert isinstance(accuracy, float)
            assert engine._stats["outcomes_resolved"] == 1

    def test_decision_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            dec = engine.record_decision("Schedule meeting", confidence=0.8, domain="career")
            assert dec.decision_id.startswith("dec_")
            assert len(engine._open_decisions) == 1

            result = engine.resolve_decision(dec.decision_id, "Meeting was productive", OutcomeType.POSITIVE)
            assert result is not None
            assert result.outcome_type == OutcomeType.POSITIVE
            assert len(engine._open_decisions) == 0

    def test_feedback(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            dec = engine.record_decision("Test", confidence=0.7)
            engine.add_feedback(dec.decision_id, FeedbackSource.EXPLICIT_USER, 0.9, "Great")
            assert engine._stats["feedback_received"] == 1
            assert len(dec.feedback) == 1

    def test_feedback_nonexistent(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            result = engine.add_feedback("nonexistent", FeedbackSource.EXPLICIT_USER, 0.5)
            assert result is False

    def test_confidence_calibration(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            calibrated = engine.calibrate_confidence(0.8)
            assert 0.0 <= calibrated <= 1.0

    @pytest.mark.asyncio
    async def test_improvement_cycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            report = await engine.run_improvement_cycle()
            assert "cycle_number" in report
            assert "overall_accuracy" in report
            assert "improvements" in report

    def test_get_open_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            engine.record_decision("Test 1")
            engine.record_decision("Test 2")
            open_decs = engine.get_open_decisions()
            assert len(open_decs) == 2

    def test_get_recent_decisions(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            for i in range(5):
                engine.record_decision(f"Test {i}")
            recent = engine.get_recent_decisions(limit=3)
            assert len(recent) == 3

    def test_get_improvement_history(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            history = engine.get_improvement_history()
            assert isinstance(history, list)

    def test_resolve_nonexistent_decision(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            result = engine.resolve_decision("nonexistent", "outcome")
            assert result is None

    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)

            pred = engine.record_prediction("Will finish project", "Done by Friday", confidence=0.75, domain="career", model="oracle")
            dec = engine.record_decision("Start coding sprint", reasoning="Need to ship", confidence=0.8, domain="career", autonomy_level="auto")
            engine.resolve_prediction(pred, "Done by Thursday", OutcomeType.POSITIVE)
            engine.resolve_decision(dec.decision_id, "Shipped on time", OutcomeType.POSITIVE)
            engine.add_feedback(dec.decision_id, FeedbackSource.EXPLICIT_USER, 0.9, "Right call")

            stats = engine.get_stats()
            assert stats["stats"]["predictions_made"] == 1
            assert stats["stats"]["decisions_recorded"] == 1
            assert stats["stats"]["outcomes_resolved"] == 2
            assert stats["stats"]["feedback_received"] == 1

    def test_negative_outcome_triggers_learning(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            engine = SelfImprovementEngine(data_dir=tmpdir)
            dec = engine.record_decision("Skip testing", confidence=0.6, domain="career", autonomy_level="auto")
            engine.resolve_decision(dec.decision_id, "Bugs found in production", OutcomeType.NEGATIVE)
            engine.add_feedback(dec.decision_id, FeedbackSource.EXPLICIT_USER, 0.2, "Bad idea")
            weight = engine.evolver.get_strategy_weight("auto_career")
            assert weight < 1.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])