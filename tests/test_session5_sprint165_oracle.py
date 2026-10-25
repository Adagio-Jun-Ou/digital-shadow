#!/usr/bin/env python3
"""Tests for OracleBridge deeper integration methods."""

import pytest
from unittest.mock import MagicMock, patch
from shadowd.cognition.oracle_bridge import (
    OracleBridge, PredictionDomain, ConfidenceLevel,
    DomainStateSnapshot, PredictionResult, ForecastSummary,
)


@pytest.fixture
def ob():
    return OracleBridge(oracle_engine=None)


class TestPredictionDomainExtended:
    def test_new_domains_exist(self):
        new_domains = [
            PredictionDomain.ECOLOGICAL,
            PredictionDomain.PALEOCLIMATE,
            PredictionDomain.COSMIC_RADIATION,
            PredictionDomain.GRAVITATIONAL,
            PredictionDomain.COMPOUND_HAZARD,
            PredictionDomain.WORLD_STATE,
            PredictionDomain.FATE,
            PredictionDomain.PERSONAL_LIFE,
        ]
        for d in new_domains:
            assert d.value is not None

    def test_domain_count_increased(self):
        assert len(PredictionDomain) >= 35


class TestHTTPFallback:
    def test_http_post_no_server(self, ob):
        result = ob._http_post("/predict", {"target_date": "2026-01-01"})
        assert result is None

    def test_http_get_no_server(self, ob):
        result = ob._http_get("/daily-oracle")
        assert result is None

    def test_http_post_custom_url(self):
        ob = OracleBridge(oracle_engine=None, oracle_url="http://localhost:9999")
        assert ob._oracle_url == "http://localhost:9999"

    def test_predict_http_fallback_no_server(self, ob):
        fs = ob.predict(target_date="2026-01-01", confidence="high")
        assert isinstance(fs, ForecastSummary)
        assert len(fs.predictions) > 0


class TestDailyOracle:
    def test_daily_oracle_no_data(self, ob):
        result = ob.get_daily_oracle()
        assert isinstance(result, dict)
        assert "overall_trend" in result
        assert result["source"] == "heuristic"

    def test_daily_oracle_with_data(self, ob):
        ob.set_domain_state("economics", {"gdp_growth": 75, "inflation": 3.2})
        result = ob.get_daily_oracle()
        assert result["domains_active"] >= 1

    def test_daily_oracle_stats_increment(self, ob):
        ob.get_daily_oracle()
        stats = ob.get_stats()
        assert stats.get("daily_oracle_calls", 0) >= 1


class TestCompoundHazard:
    def test_compound_hazard_default(self, ob):
        result = ob.assess_compound_hazard()
        assert isinstance(result, dict)
        assert "overall_risk" in result
        assert "domain_risks" in result
        assert "cascade_risks" in result

    def test_compound_hazard_specific_domains(self, ob):
        ob.set_domain_state("climate", {"temperature_anomaly": 75})
        ob.set_domain_state("food", {"crop_yield": 30})
        result = ob.assess_compound_hazard(domains=["climate", "food"])
        assert "climate" in result["domain_risks"]
        assert "food" in result["domain_risks"]

    def test_compound_hazard_invalid_domain(self, ob):
        result = ob.assess_compound_hazard(domains=["invalid_domain_xyz"])
        assert isinstance(result["overall_risk"], float)

    def test_compound_hazard_stats(self, ob):
        ob.assess_compound_hazard()
        stats = ob.get_stats()
        assert stats.get("compound_hazard_assessments", 0) >= 1

    def test_compound_hazard_risk_bounded(self, ob):
        ob.set_domain_state("economics", {"gdp": 90})
        result = ob.assess_compound_hazard(domains=["economics"])
        assert 0.0 <= result["overall_risk"] <= 1.0


class TestRegimeShift:
    def test_regime_shift_no_data(self, ob):
        result = ob.get_regime_shift_probability("economics")
        assert result["probability"] == 0.0
        assert result["source"] in ("no_data", "insufficient_data")

    def test_regime_shift_with_data(self, ob):
        ob.set_domain_state("geopolitics", {"tension_index": 80, "conflict_risk": 15})
        result = ob.get_regime_shift_probability("geopolitics")
        assert "probability" in result
        assert "mean" in result
        assert "std_dev" in result

    def test_regime_shift_invalid_domain(self, ob):
        result = ob.get_regime_shift_probability("invalid_xyz")
        assert result["domain"] == "economics"

    def test_regime_shift_stats(self, ob):
        ob.get_regime_shift_probability("economics")
        stats = ob.get_stats()
        assert stats.get("regime_shift_assessments", 0) >= 1


class TestProphecy:
    def test_prophecy_default(self, ob):
        result = ob.get_prophecy()
        assert isinstance(result, dict)
        assert "horizon_days" in result
        assert "target_date" in result
        assert "confidence" in result
        assert "narrative_tone" in result
        assert "key_signals" in result
        assert result["horizon_days"] == 90

    def test_prophecy_custom_horizon(self, ob):
        result = ob.get_prophecy(horizon_days=30)
        assert result["horizon_days"] == 30

    def test_prophecy_invalid_horizon_rounded(self, ob):
        result = ob.get_prophecy(horizon_days=50)
        assert result["horizon_days"] in (1, 7, 30, 90, 365)

    def test_prophecy_stats(self, ob):
        ob.get_prophecy()
        stats = ob.get_stats()
        assert stats.get("prophecies_generated", 0) >= 1


class TestSetterMapCompleteness:
    def test_all_predicate_domains_have_setters(self):
        for domain in PredictionDomain:
            setter = _SETTER_MAP.get(domain)
            assert setter is not None, f"No setter for {domain.value}"

    def test_setter_names_valid(self):
        for domain, setter in _SETTER_MAP.items():
            assert setter.startswith("set_"), f"Invalid setter name: {setter}"


from shadowd.cognition.oracle_bridge import _SETTER_MAP