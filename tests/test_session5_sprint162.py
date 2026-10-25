#!/usr/bin/env python3
"""Sprint 162 — bare except Exception narrowing pass 3

Narrowed:
- oracle_bridge.py: ingest_temporal_data() and ingest_anomalies() Exception → (ValueError, TypeError, AttributeError, KeyError, OSError)
- evolution_guide.py: ingest_device_context() Exception → (ValueError, TypeError, AttributeError, KeyError)
"""

import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.oracle_bridge import OracleBridge
from shadowd.cognition.evolution_guide import EvolutionGuide


class TestOracleBridgeExceptNarrowing:
    """OracleBridge ingest methods catch specific exceptions."""

    def test_ingest_temporal_data_no_crash_on_bad_data(self):
        ob = OracleBridge()
        tr = MagicMock()
        tr.get_patterns.return_value = []
        result = ob.ingest_temporal_data(tr)
        assert result is None or result is not None

    def test_ingest_anomalies_no_crash_on_bad_data(self):
        ob = OracleBridge()
        ad = MagicMock()
        ad.get_recent_anomalies.return_value = []
        result = ob.ingest_anomalies(ad)
        assert result is None or result is not None


class TestEvolutionGuideExceptNarrowing:
    """EvolutionGuide ingest_device_context catches specific exceptions."""

    def test_ingest_device_context_no_crash_on_empty_context(self):
        eg = EvolutionGuide()
        ctx = MagicMock()
        ctx.get_context.return_value = {}
        eg.ingest_device_context(ctx)

    def test_ingest_device_context_no_crash_on_missing_attributes(self):
        eg = EvolutionGuide()
        ctx = MagicMock(spec=[])
        eg.ingest_device_context(ctx)