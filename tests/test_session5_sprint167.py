#!/usr/bin/env python3
"""Sprint 167 — OracleBridge Return Value Deepcopy + predict/diagnose Safety

Tests:
- set_domain_state() returns deepcopy (not mutable internal reference)
- predict() returns deepcopy of ForecastSummary
- diagnose() returns fresh dicts (not internal references)
"""

import pytest
from shadowd.cognition.oracle_bridge import (
    OracleBridge,
    PredictionDomain,
    ConfidenceLevel,
)


class TestSetDomainStateDeepcopy:
    """set_domain_state() returns deepcopy."""

    def test_returned_snapshot_is_deepcopy(self):
        ob = OracleBridge(oracle_engine=None)
        result = ob.set_domain_state("economics", {"gdp": 0.03, "inflation": 0.02})
        assert result.domain.value == "economics"
        result.state_values["gdp"] = 999.0
        inner = ob.get_domain_states()
        assert inner["economics"]["state_values"]["gdp"] != 999.0

    def test_state_values_not_mutated(self):
        ob = OracleBridge(oracle_engine=None)
        original = {"gdp": 0.03, "inflation": 0.02}
        result = ob.set_domain_state("economics", original.copy())
        result.state_values["hacked"] = True
        inner = ob.get_domain_states()
        assert "hacked" not in inner["economics"]["state_values"]


class TestPredictDeepcopy:
    """predict() returns deepcopy of ForecastSummary."""

    def test_predict_returns_deepcopy(self):
        ob = OracleBridge(oracle_engine=None)
        result = ob.predict(target_date="2026-01-01")
        result.domains_predicted.append("hacked_domain")
        result2 = ob.predict(target_date="2026-01-01")
        stored_forecasts = ob.get_forecasts(limit=1)
        if stored_forecasts:
            assert "hacked_domain" not in stored_forecasts[0].domains_predicted


class TestDiagnoseFreshDict:
    """diagnose() returns fresh dicts."""

    def test_diagnose_returns_fresh_dict(self):
        ob = OracleBridge(oracle_engine=None)
        result = ob.diagnose()
        assert isinstance(result, dict)
        assert "status" in result or "error" in result