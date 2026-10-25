#!/usr/bin/env python3
"""
Sprint 144 regression tests:
- OracleBridge module: state management, prediction, diagnosis, heuristic fallback
- ShadowOS wiring
- Thread safety, bounded collections, input sanitization, deepcopy
"""

import copy
import threading
import pytest

from shadowd.cognition.oracle_bridge import (
    OracleBridge, PredictionDomain, ConfidenceLevel,
    PredictionResult, ForecastSummary, DomainStateSnapshot,
    _sanitize, _sanitize_content,
)


class TestDomainStateSnapshot:
    def test_auto_timestamp(self):
        s = DomainStateSnapshot(domain=PredictionDomain.ECONOMICS, state_values={"gdp": 0.025})
        assert s.timestamp

    def test_domain_validation(self):
        s = DomainStateSnapshot(domain="invalid", state_values={})
        assert s.domain == PredictionDomain.ECONOMICS

    def test_state_values_converted(self):
        s = DomainStateSnapshot(domain=PredictionDomain.ECONOMICS, state_values={"gdp": 0.025, "name": "us"})
        assert isinstance(s.state_values["gdp"], float)
        assert isinstance(s.state_values["name"], str)

    def test_source_sanitized(self):
        s = DomainStateSnapshot(domain=PredictionDomain.ECONOMICS, state_values={}, source="<script>alert(1)</script>")
        assert "<" not in s.source

    def test_to_dict(self):
        s = DomainStateSnapshot(domain=PredictionDomain.CLIMATE, state_values={"temp": 1.5})
        d = s.to_dict()
        assert d["domain"] == "climate"
        assert d["state_values"]["temp"] == 1.5


class TestPredictionResult:
    def test_auto_id(self):
        p = PredictionResult()
        assert len(p.prediction_id) == 12

    def test_domain_validation(self):
        p = PredictionResult(domain="invalid")
        assert p.domain == PredictionDomain.ECONOMICS

    def test_variable_sanitized(self):
        p = PredictionResult(variable="<script>x</script>value")
        assert "<" not in p.variable

    def test_model_name_strict(self):
        p = PredictionResult(model_name="<b>bold</b>")
        assert "<" not in p.model_name

    def test_confidence_level_validation(self):
        p = PredictionResult(confidence_level="invalid")
        assert p.confidence_level == ConfidenceLevel.MEDIUM

    def test_to_dict_round(self):
        p = PredictionResult(predicted_value=0.123456, confidence_interval=(0.1, 0.2))
        d = p.to_dict()
        assert isinstance(d["predicted_value"], float)


class TestForecastSummary:
    def test_auto_id(self):
        f = ForecastSummary()
        assert len(f.forecast_id) == 12

    def test_domains_sanitized(self):
        f = ForecastSummary(domains_predicted=["<evil>", "economics"])
        assert "<" not in f.domains_predicted[0]

    def test_to_dict(self):
        f = ForecastSummary(target_date="2026-01-01")
        d = f.to_dict()
        assert d["target_date"] == "2026-01-01"
        assert "prediction_count" in d


class TestOracleBridgeCore:
    def test_set_domain_state(self):
        ob = OracleBridge(oracle_engine=None)
        snap = ob.set_domain_state("economics", {"gdp_growth_rate": 0.025})
        assert snap.domain == PredictionDomain.ECONOMICS
        assert ob._stats["states_set"] == 1

    def test_set_multiple_domains(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.set_domain_state("climate", {"co2_ppm": 420})
        states = ob.get_domain_states()
        assert "economics" in states
        assert "climate" in states
        assert ob._stats["domains_active"] == 2

    def test_heuristic_prediction_fallback(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        fs = ob.predict(target_date="2026-01-01", confidence="medium")
        assert len(fs.predictions) >= 1
        assert fs.confidence_level == ConfidenceLevel.MEDIUM

    def test_predict_with_domain_filter(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.set_domain_state("climate", {"temp": 1.5})
        fs = ob.predict(target_date="2026-12-31", domains=["economics"])
        for p in fs.predictions:
            assert p.domain == PredictionDomain.ECONOMICS

    def test_predict_default_target_date(self):
        ob = OracleBridge(oracle_engine=None)
        fs = ob.predict(confidence="high")
        assert fs.target_date

    def test_get_predictions(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.predict(target_date="2026-01-01")
        preds = ob.get_predictions()
        assert len(preds) >= 1

    def test_get_predictions_by_domain(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.set_domain_state("climate", {"temp": 1.5})
        ob.predict(target_date="2026-01-01")
        preds = ob.get_predictions(domain="economics")
        for p in preds:
            assert p.domain == PredictionDomain.ECONOMICS

    def test_get_forecasts(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.predict(target_date="2026-01-01")
        fcs = ob.get_forecasts()
        assert len(fcs) >= 1

    def test_diagnose_without_engine(self):
        ob = OracleBridge(oracle_engine=None)
        result = ob.diagnose()
        assert isinstance(result, dict)
        assert "status" in result or "error" in result or "anomalies" in result or "summary" in result

    def test_get_available_domains(self):
        ob = OracleBridge(oracle_engine=None)
        domains = ob.get_available_domains()
        assert len(domains) >= 27
        assert any(d["domain"] == "economics" for d in domains)

    def test_get_stats(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        stats = ob.get_stats()
        assert stats["states_set"] == 1
        assert stats["domains_active"] == 1

    def test_get_diagnoses(self):
        ob = OracleBridge(oracle_engine=None)
        result = ob.get_diagnoses()
        assert isinstance(result, list)


class TestOracleBridgeWithEngine:
    def test_predict_with_oracle_engine(self):
        try:
            from oracle.prediction.engine import OracleEngine
            engine = OracleEngine()
        except ImportError:
            pytest.skip("Oracle engine not installed")

        ob = OracleBridge(oracle_engine=engine)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.025, "inflation_rate": 0.03})
        fs = ob.predict(target_date="2026-01-01", confidence="high")
        assert len(fs.predictions) >= 1
        assert fs.confidence_level == ConfidenceLevel.HIGH

    def test_diagnose_with_oracle_engine(self):
        try:
            from oracle.prediction.engine import OracleEngine
            engine = OracleEngine()
        except ImportError:
            pytest.skip("Oracle engine not installed")

        ob = OracleBridge(oracle_engine=engine)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.025})
        result = ob.diagnose()
        assert isinstance(result, dict)


class TestOracleBridgeThreadSafety:
    def test_concurrent_state_setting(self):
        ob = OracleBridge(oracle_engine=None)
        errors = []

        def set_state():
            try:
                for i in range(20):
                    ob.set_domain_state("economics", {"val": float(i)})
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=set_state) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_predict(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.025})
        errors = []

        def predict():
            try:
                for _ in range(5):
                    ob.predict(target_date="2026-01-01")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=predict) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_read_write(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        errors = []

        def write():
            try:
                for i in range(10):
                    ob.set_domain_state("climate", {"temp": float(i)})
            except Exception as e:
                errors.append(e)

        def read():
            try:
                for _ in range(10):
                    ob.get_domain_states()
                    ob.get_predictions()
                    ob.get_stats()
                    ob.get_available_domains()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=write), threading.Thread(target=read)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestGetStatsReturnsCopy:
    def test_stats_copy(self):
        ob = OracleBridge(oracle_engine=None)
        s = ob.get_stats()
        s["states_set"] = 999
        assert ob._stats["states_set"] != 999


class TestDeepcopyReturns:
    def test_predictions_deepcopy(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.predict(target_date="2026-01-01")
        preds = ob.get_predictions()
        if preds:
            preds[0].predicted_value = 9999
        original_preds = ob.get_predictions()
        for p in original_preds:
            assert p.predicted_value != 9999

    def test_forecasts_deepcopy(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.predict(target_date="2026-01-01")
        fcs = ob.get_forecasts()
        if fcs:
            fcs[0].domains_predicted.append("hacked")
        original = ob.get_forecasts()
        if original:
            assert "hacked" not in original[0].domains_predicted


class TestPredictionDomainEnum:
    def test_all_domains_known(self):
        assert len(PredictionDomain) >= 27

    def test_domain_values(self):
        assert PredictionDomain.ECONOMICS.value == "economics"
        assert PredictionDomain.CYBER.value == "cyber"
        assert PredictionDomain.SPACE.value == "space"


class TestConfidenceLevel:
    def test_all_levels(self):
        assert len(ConfidenceLevel) == 4
        assert ConfidenceLevel.LOW.value == "low"
        assert ConfidenceLevel.VERY_HIGH.value == "very_high"


class TestSanitize:
    def test_strict_strips_lt_gt(self):
        assert "<" not in _sanitize("hello<world>test")

    def test_content_preserves_lt_gt(self):
        assert "<" in _sanitize_content("latency < 100ms")

    def test_strip_before_truncate(self):
        assert _sanitize("  hello  ", 7) == "hello"