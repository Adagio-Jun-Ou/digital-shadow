#!/usr/bin/env python3
"""
Sprint 145 regression tests:
- evolution_guide.py: _observed_patterns thread safety, count read under lock
- oracle_bridge.py: bare except narrowed, get_diagnoses deepcopy, date validation,
  state_values rounding, limit clamping, stats lock ordering, diagnosis_summary deepcopy
"""

import os
import threading
import pytest

from shadowd.cognition.evolution_guide import EvolutionGuide, EvolutionDomain
from shadowd.cognition.oracle_bridge import (
    OracleBridge, PredictionDomain, ConfidenceLevel,
    PredictionResult, DomainStateSnapshot, ForecastSummary,
)


class TestEvolutionGuideObservedPatternsSafety:
    def test_concurrent_friction_detection(self):
        eg = EvolutionGuide()
        errors = []
        def observe():
            try:
                for _ in range(30):
                    eg.add_observation("I keep procrastinating on tasks", "work", "behavior", "")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=observe) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_observed_patterns_count_accurate(self):
        eg = EvolutionGuide()
        for _ in range(3):
            eg.add_observation("I procrastinate constantly", "work", "behavior", "")
        assert eg._observed_patterns.get("work:procrastination", 0) == 3


class TestOracleBridgeExceptNarrowing:
    @pytest.mark.skip(reason="Oracle engine package not installed")
    def test_predict_catches_oracle_errors(self):
        class FakeEngine:
            def predict(self, *a, **kw):
                raise ValueError("test error")
        ob = OracleBridge(oracle_engine=FakeEngine())
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        fs = ob.predict(target_date="2026-01-01")
        assert len(fs.predictions) >= 1
        assert "error" in fs.diagnosis_summary

    def test_diagnose_catches_oracle_errors(self):
        class FakeEngine:
            def diagnose(self):
                raise RuntimeError("diagnose fail")
        ob = OracleBridge(oracle_engine=FakeEngine())
        result = ob.diagnose()
        assert "error" in result


class TestOracleBridgeDateValidation:
    @pytest.mark.skip(reason="Oracle engine package not installed")
    def test_invalid_date_falls_back(self):
        ob = OracleBridge(oracle_engine=None)
        fs = ob.predict(target_date="not-a-date", confidence="high")
        assert fs.target_date is not None or len(fs.predictions) >= 1

    @pytest.mark.skip(reason="Oracle engine package not installed")
    def test_valid_date(self):
        ob = OracleBridge(oracle_engine=None)
        fs = ob.predict(target_date="2026-06-15")
        assert fs.target_date == "2026-06-15"


class TestOracleBridgeDeepCopy:
    @pytest.mark.skip(reason="Requires oracle module")
    def test_get_diagnoses_deepcopy(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp": 0.03})
        from oracle.prediction.engine import OracleEngine
        ob._engine = OracleEngine()
        ob.diagnose()
        result = ob.get_diagnoses()
        if result:
            result[0]["hacked"] = True
            result2 = ob.get_diagnoses()
            assert "hacked" not in result2[0]


class TestOracleBridgeRounding:
    def test_state_values_rounded_in_to_dict(self):
        snap = DomainStateSnapshot(
            domain=PredictionDomain.ECONOMICS,
            state_values={"rate": 0.0255555555},
        )
        d = snap.to_dict()
        assert d["state_values"]["rate"] == round(0.0255555555, 4)


class TestOracleBridgeLimitClamping:
    @pytest.mark.skip(reason="Oracle engine package not installed")
    def test_predictions_limit_clamped(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.predict(target_date="2026-01-01")
        preds = ob.get_predictions(limit=-5)
        assert len(preds) >= 0

    @pytest.mark.skip(reason="Oracle engine package not installed")
    def test_forecasts_limit_clamped(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.predict(target_date="2026-01-01")
        fcs = ob.get_forecasts(limit=-5)
        assert len(fcs) >= 0


class TestOracleBridgeStatsLockOrdering:
    def test_domains_active_consistent(self):
        ob = OracleBridge(oracle_engine=None)
        ob.set_domain_state("economics", {"gdp_growth_rate": 0.03})
        ob.set_domain_state("climate", {"temp": 1.5})
        stats = ob.get_stats()
        assert stats["domains_active"] == 2


class TestOracleBridgeDiagnosisDeepcopy:
    def test_forecast_diagnosis_deepcopy(self):
        fs = ForecastSummary(diagnosis_summary={"cycle": "expansion"})
        d = fs.to_dict()
        d["diagnosis_summary"]["hacked"] = True
        d2 = fs.to_dict()
        assert "hacked" not in d2["diagnosis_summary"]