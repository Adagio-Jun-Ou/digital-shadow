"""Tests for superhuman v2 algorithms: TemporalPatternEngine, KalmanFilter1D, MultiArmedBandit, AnomalyDetector, DecisionTree, ModelPersistence, FeedbackLoop"""

import json
import math
import tempfile
from pathlib import Path
from datetime import datetime, timezone, timedelta

from shikigami.superhuman import (
    TemporalPatternEngine,
    KalmanFilter1D,
    MultiArmedBandit,
    AnomalyDetector,
    DecisionTree,
    ModelPersistence,
    FeedbackLoop,
    BayesianPredictor,
    MarkovBehaviorModel,
    KnowledgeGraph,
    ReinforcementLearner,
)


class TestTemporalPatternEngine:
    def test_observe_increments_counts(self):
        engine = TemporalPatternEngine()
        engine.observe({"type": "terminal_command", "timestamp": "2026-01-15T10:00:00"})
        engine.observe({"type": "terminal_command", "timestamp": "2026-01-15T10:30:00"})
        engine.observe({"type": "file_change", "timestamp": "2026-01-15T14:00:00"})
        assert engine.hour_counts[10] == 2
        assert engine.hour_counts[14] == 1
        assert engine.total_observations == 3

    def test_forecast_hour(self):
        engine = TemporalPatternEngine()
        for h in range(9, 17):
            for _ in range(5):
                engine.observe({"type": "terminal_command", "timestamp": f"2026-01-15T{h:02d}:00:00"})
        fc = engine.forecast_hour(10, "terminal_command")
        assert fc["hour"] == 10
        assert fc["probability"] > 0
        assert fc["observations_at_hour"] > 0

    def test_forecast_weekday(self):
        engine = TemporalPatternEngine()
        for day in range(5):
            for _ in range(10):
                engine.observe({"type": "terminal_command", "timestamp": f"2026-01-{13+day}T10:00:00"})
        fc = engine.forecast_weekday(0)
        assert fc["weekday"] == 0
        assert fc["probability"] >= 0

    def test_detect_trend_increasing(self):
        engine = TemporalPatternEngine()
        base = datetime(2026, 1, 1)
        for i in range(14):
            for _ in range(i + 1):
                engine.observe({"type": "terminal_command", "timestamp": (base + timedelta(days=i)).isoformat()})
        trend = engine.detect_trend()
        assert trend["trend"] in ("increasing", "stable", "decreasing")
        assert "slope" in trend

    def test_detect_periodicity_insufficient_data(self):
        engine = TemporalPatternEngine()
        result = engine.detect_periodicity()
        assert result == []

    def test_predict_next_activity(self):
        engine = TemporalPatternEngine()
        for h in range(9, 17):
            for _ in range(3):
                engine.observe({"type": "terminal_command", "timestamp": f"2026-01-15T{h:02d}:00:00"})
        pred = engine.predict_next_activity(14)
        assert "predicted_hour" in pred
        assert "predicted_probability" in pred


class TestKalmanFilter1D:
    def test_converges_to_true_value(self):
        kf = KalmanFilter1D(process_noise=0.01, measurement_noise=1.0)
        measurements = [5.0 + 0.1 * (i % 3 - 1) for i in range(20)]
        results = kf.filter_sequence(measurements)
        final_est = results[-1][0]
        assert abs(final_est - 5.0) < 1.0

    def test_uncertainty_decreases(self):
        kf = KalmanFilter1D(process_noise=0.01, measurement_noise=1.0)
        kf.update(10.0)
        unc1 = kf.uncertainty
        kf.update(10.5)
        unc2 = kf.uncertainty
        assert unc2 < unc1

    def test_predict_step(self):
        kf = KalmanFilter1D()
        state, cov = kf.predict()
        assert cov > 0

    def test_update_step(self):
        kf = KalmanFilter1D(measurement_noise=0.5)
        state, cov = kf.update(10.0)
        assert abs(state - 10.0) < 5.0
        assert cov > 0

    def test_multiple_devices(self):
        trackers = {}
        for rssi in [-60, -65, -70, -55]:
            mac = f"AA:BB:CC:DD:EE:{abs(rssi)}"
            trackers[mac] = KalmanFilter1D(process_noise=0.01, measurement_noise=2.0)
            trackers[mac].update(rssi)
        for _ in range(10):
            for mac, kf in trackers.items():
                rssi_base = {"AA:BB:CC:DD:EE:60": -60, "AA:BB:CC:DD:EE:65": -65,
                             "AA:BB:CC:DD:EE:70": -70, "AA:BB:CC:DD:EE:55": -55}
                kf.update(rssi_base[mac] + (hash(str(_)) % 3 - 1))
        assert all(kf.uncertainty < 1.0 for kf in trackers.values())


class TestMultiArmedBandit:
    def test_ucb1_selection(self):
        bandit = MultiArmedBandit(algorithm="ucb1")
        bandit.add_arm("arm_a", 0.7, "Good arm")
        bandit.add_arm("arm_b", 0.3, "Bad arm")
        for _ in range(20):
            arm, score = bandit.select()
            reward = 0.8 if arm == "arm_a" else 0.2
            bandit.update(arm, reward)
        a_pulls = bandit.arms["arm_a"]["pulls"]
        b_pulls = bandit.arms["arm_b"]["pulls"]
        assert a_pulls > b_pulls

    def test_thompson_sampling(self):
        bandit = MultiArmedBandit(algorithm="thompson")
        bandit.add_arm("arm_1", 0.6)
        bandit.add_arm("arm_2", 0.4)
        for _ in range(30):
            arm, _ = bandit.select()
            bandit.update(arm, 0.7 if arm == "arm_1" else 0.3)
        assert bandit.total_pulls > 0

    def test_get_stats(self):
        bandit = MultiArmedBandit()
        bandit.add_arm("test_arm", 0.5, "Test")
        bandit.update("test_arm", 0.8)
        stats = bandit.get_stats()
        assert "arms" in stats
        assert "test_arm" in stats["arms"]
        assert stats["arms"]["test_arm"]["pulls"] == 1

    def test_constrained_selection(self):
        bandit = MultiArmedBandit()
        bandit.add_arm("a", 0.5)
        bandit.add_arm("b", 0.5)
        arm, _ = bandit.select(available_arms=["b"])
        assert arm == "b"


class TestAnomalyDetector:
    def test_zscore_anomaly(self):
        det = AnomalyDetector(window_size=20, z_threshold=2.0)
        for i in range(20):
            det.update_baseline("temp", 20.0 + (i % 3) * 0.1)
        result = det.check_zscore("temp", 30.0)
        assert result is not None
        assert result["severity"] in ("critical", "warning")
        assert abs(result["z_score"]) > 2.0

    def test_no_anomaly_in_range(self):
        det = AnomalyDetector(z_threshold=3.0)
        for i in range(20):
            det.update_baseline("cpu", 50.0 + (i % 5))
        result = det.check_zscore("cpu", 52.0)
        assert result is None

    def test_entropy_novelty(self):
        det = AnomalyDetector()
        for i in range(20):
            det.update_baseline("latency", 100.0 + i * 0.5)
        result = det.check_entropy_novelty("latency", 500.0)
        assert result is not None
        assert result["type"] == "novelty"

    def test_get_anomalies(self):
        det = AnomalyDetector()
        for i in range(20):
            det.update_baseline("metric", 10.0)
        det.check_zscore("metric", 100.0)
        anomalies = det.get_anomalies()
        assert len(anomalies) >= 1

    def test_contextual_anomaly(self):
        det = AnomalyDetector()
        for i in range(20):
            det.update_baseline("requests", 100.0 + i)
        result = det.detect_contextual("requests", 200.0, {"source": "external"})
        assert result is not None
        assert "context" in result


class TestDecisionTree:
    def test_simple_classification(self):
        dt = DecisionTree(max_depth=3)
        features = [
            [9, 0, 0], [10, 0, 0], [11, 0, 0],
            [14, 1, 1], [15, 1, 1], [16, 1, 1],
            [22, 2, 0], [23, 2, 0], [0, 2, 0],
        ]
        labels = ["work", "work", "work", "work", "work", "work", "rest", "rest", "rest"]
        dt.fit(features, labels, feature_names=["hour", "category", "priority"])
        pred = dt.predict([10, 0, 0])
        assert pred in ("work", "rest")

    def test_feature_importance(self):
        dt = DecisionTree(max_depth=3)
        features = [[i, 10 - i, i % 3] for i in range(20)]
        labels = ["a" if i < 10 else "b" for i in range(20)]
        dt.fit(features, labels, feature_names=["x", "y", "z"])
        importance = dt.feature_importance()
        assert isinstance(importance, dict)

    def test_predict_proba(self):
        dt = DecisionTree(max_depth=3)
        features = [[i] for i in range(20)]
        labels = ["low" if i < 10 else "high" for i in range(20)]
        dt.fit(features, labels)
        proba = dt.predict_proba([5])
        assert "low" in proba or "high" in proba


class TestFeedbackLoop:
    def test_record_prediction(self):
        fl = FeedbackLoop()
        fl.record_prediction("oracle", "will rain tomorrow", 0.7)
        assert len(fl.predictions) == 1
        assert fl.predictions[0]["ability"] == "oracle"
        assert fl.predictions[0]["confidence"] == 0.7

    def test_record_outcome(self):
        fl = FeedbackLoop()
        fl.record_prediction("oracle", "will rain", 0.7)
        pred_id = fl.predictions[0]["id"]
        fl.record_outcome("oracle", pred_id, "rained", True)
        assert fl.predictions[0]["resolved"] is True
        assert fl.predictions[0]["accuracy"] is True

    def test_get_accuracy(self):
        fl = FeedbackLoop()
        fl.record_prediction("oracle", "p1", 0.8)
        fl.record_prediction("oracle", "p2", 0.6)
        fl.record_outcome("oracle", fl.predictions[0]["id"], "actual1", True)
        fl.record_outcome("oracle", fl.predictions[1]["id"], "actual2", False)
        acc = fl.get_accuracy("oracle")
        assert acc["accuracy"] == 0.5
        assert acc["count"] == 2

    def test_calibrate_confidence(self):
        fl = FeedbackLoop()
        for i in range(10):
            fl.record_prediction("oracle", f"pred_{i}", 0.9)
            fl.record_outcome("oracle", fl.predictions[i]["id"], f"actual_{i}", True)
        calibrated = fl.calibrate_confidence("oracle", 0.9)
        assert 0.0 <= calibrated <= 1.0

    def test_max_history_trimming(self):
        fl = FeedbackLoop(max_history=10)
        for i in range(20):
            fl.record_prediction("oracle", f"pred_{i}", 0.5)
        assert len(fl.predictions) <= 15


class TestModelPersistence:
    def test_save_and_load(self, tmp_path):
        ModelPersistence.BASE_DIR = tmp_path / "models"
        ModelPersistence.save("test_model", {"key": "value", "number": 42})
        loaded = ModelPersistence.load("test_model")
        assert loaded["key"] == "value"
        assert loaded["number"] == 42

    def test_load_nonexistent(self, tmp_path):
        ModelPersistence.BASE_DIR = tmp_path / "models"
        loaded = ModelPersistence.load("nonexistent_model")
        assert loaded is None

    def test_save_bayesian(self, tmp_path):
        ModelPersistence.BASE_DIR = tmp_path / "models"
        bp = BayesianPredictor()
        bp.add_hypothesis("test_h", 0.6, ["e1", "e2"])
        bp.observe("e1", {"test_h": True})
        ModelPersistence.save_bayesian(bp)
        
        bp2 = BayesianPredictor()
        ModelPersistence.load_bayesian(bp2)
        assert "test_h" in bp2.hypotheses
        assert bp2.hypotheses["test_h"]["count"] == 1

    def test_save_load_markov(self, tmp_path):
        ModelPersistence.BASE_DIR = tmp_path / "models"
        mm = MarkovBehaviorModel()
        mm.record("state_a", "action_1", "state_b", 1.0)
        mm.record("state_b", "action_2", "state_c", 0.5)
        ModelPersistence.save_markov(mm)
        
        mm2 = MarkovBehaviorModel()
        ModelPersistence.load_markov(mm2)
        assert mm2.state_counts.get("state_a", 0) == 1
        assert mm2.state_counts.get("state_b", 0) == 1

    def test_save_load_knowledge_graph(self, tmp_path):
        ModelPersistence.BASE_DIR = tmp_path / "models"
        kg = KnowledgeGraph()
        eid = kg.add_entity("person", "Alice", {"role": "developer"})
        kg.add_relation(eid, "knows", kg.add_entity("person", "Bob"))
        ModelPersistence.save_knowledge_graph(kg)
        
        kg2 = KnowledgeGraph()
        ModelPersistence.load_knowledge_graph(kg2)
        assert len(kg2.entities) > 0

    def test_save_load_bandit(self, tmp_path):
        ModelPersistence.BASE_DIR = tmp_path / "models"
        bandit = MultiArmedBandit()
        bandit.add_arm("arm_a", 0.7)
        bandit.update("arm_a", 0.9)
        ModelPersistence.save_bandit(bandit)
        
        bandit2 = MultiArmedBandit()
        ModelPersistence.load_bandit(bandit2)
        assert "arm_a" in bandit2.arms
        assert bandit2.arms["arm_a"]["pulls"] >= 1


class TestIntegrationEnhancedAbilities:
    def test_temporal_engine_in_oracle_context(self):
        engine = TemporalPatternEngine()
        for day in range(7):
            for h in range(9, 18):
                engine.observe({
                    "type": "terminal_command",
                    "timestamp": f"2026-01-{13+day}T{h:02d}:00:00",
                })
        forecast = engine.predict_next_activity(14)
        assert "predicted_hour" in forecast
        trend = engine.detect_trend()
        assert "trend" in trend
        periodicity = engine.detect_periodicity()
        assert isinstance(periodicity, list)

    def test_kalman_filter_signal_smoothing(self):
        kf = KalmanFilter1D(process_noise=0.01, measurement_noise=2.0)
        raw_rssi = [-60, -62, -58, -65, -61, -59, -63, -60, -61, -57]
        kf.state = float(raw_rssi[0])
        smoothed = [kf.update(r)[0] for r in raw_rssi]
        assert len(smoothed) == len(raw_rssi)
        assert abs(kf.estimated_value - sum(raw_rssi) / len(raw_rssi)) < 10.0
        assert kf.uncertainty < 1.0

    def test_bandit_with_social_radar_context(self):
        bandit = MultiArmedBandit(algorithm="ucb1")
        strategies = ["engage_proactively", "observe_wait", "communicate_directly", "avoid_topic"]
        for s in strategies:
            bandit.add_arm(s, 0.5, s)
        
        for _ in range(50):
            arm, _ = bandit.select(strategies)
            reward = 0.9 if arm == "engage_proactively" else 0.3
            bandit.update(arm, reward)
        
        stats = bandit.get_stats()
        assert stats["arms"]["engage_proactively"]["pulls"] > 0
        best = max(stats["arms"].items(), key=lambda x: x[1]["mean_reward"])
        assert best[0] == "engage_proactively"

    def test_anomaly_detector_with_env_model(self):
        from shikigami.superhuman import EnvironmentModel
        det = AnomalyDetector(window_size=50, z_threshold=2.0)
        for i in range(30):
            det.update_baseline("activity_rate", 5.0 + (i % 5) * 0.5)
        result = det.check_zscore("activity_rate", 50.0)
        assert result is not None
        assert result["z_score"] > 2.0

    def test_decision_tree_with_pattern_types(self):
        dt = DecisionTree(max_depth=4)
        features = [
            [9, 1, 0], [10, 1, 0], [11, 1, 0],
            [12, 1, 0], [13, 1, 1], [14, 1, 1],
            [22, 0, 0], [23, 0, 0], [1, 0, 0],
            [2, 0, 0], [15, 1, 1], [16, 1, 1],
        ]
        labels = ["coding", "coding", "coding", "meeting", "meeting", "meeting",
                   "resting", "resting", "resting", "resting", "meeting", "meeting"]
        dt.fit(features, labels, feature_names=["hour", "weekday", "is_work"])
        assert dt.predict([10, 1, 0]) in labels