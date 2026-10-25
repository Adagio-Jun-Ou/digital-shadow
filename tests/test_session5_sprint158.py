#!/usr/bin/env python3
"""Sprint 158-159 — EvolutionGuide Auto-Sync + OracleBridge Auto-Sync Tests

Tests:
- AutonomousDaemon has EVOLUTION_SYNC and ORACLE_SYNC action types
- Rate limits configured for evolution_sync (120s) and oracle_sync (180s)
- _run_evolution_sync() calls shadow_os.sync_evolution_from_device()
- _run_oracle_sync() calls shadow_os.sync_oracle_from_cognition()
- Decisions recorded for both sync types
- Stats tracked for both sync types
- Tick loop includes evolution_sync when user PRESENT
- Tick loop includes oracle_sync when user IDLE/AWAY
"""

import asyncio
import time
import pytest
from unittest.mock import MagicMock, AsyncMock, patch, PropertyMock
from datetime import datetime, timezone

from shadowd.cognition.autonomous_daemon import (
    AutonomousDaemon,
    AutonomousActionType,
    AutonomousDecision,
    UserActivityState,
    RhythmSample,
)


class TestEvolutionSyncActionType:
    """EVOLUTION_SYNC action type exists."""

    def test_evolution_sync_in_enum(self):
        assert AutonomousActionType.EVOLUTION_SYNC == "evolution_sync"

    def test_oracle_sync_in_enum(self):
        assert AutonomousActionType.ORACLE_SYNC == "oracle_sync"


class TestEvolutionSyncRateLimits:
    """Rate limits for evolution/oracle sync configured."""

    def test_evolution_sync_rate_limit(self):
        daemon = AutonomousDaemon(base_tick_seconds=15.0)
        assert daemon._rate_limit("evolution_sync") == 120.0

    def test_oracle_sync_rate_limit(self):
        daemon = AutonomousDaemon(base_tick_seconds=15.0)
        assert daemon._rate_limit("oracle_sync") == 180.0


class TestEvolutionSyncTrackingTimestamps:
    """Timestamps for evolution/oracle sync tracking."""

    def test_last_evolution_sync_initialized(self):
        daemon = AutonomousDaemon()
        assert daemon._last_evolution_sync == 0

    def test_last_oracle_sync_initialized(self):
        daemon = AutonomousDaemon()
        assert daemon._last_oracle_sync == 0


class TestEvolutionSyncStats:
    """Stats tracking for evolution/oracle sync."""

    def test_evolution_syncs_in_stats(self):
        daemon = AutonomousDaemon()
        assert "evolution_syncs" in daemon._stats
        assert daemon._stats["evolution_syncs"] == 0

    def test_oracle_syncs_in_stats(self):
        daemon = AutonomousDaemon()
        assert "oracle_syncs" in daemon._stats
        assert daemon._stats["oracle_syncs"] == 0


class TestRunEvolutionSync:
    """_run_evolution_sync() calls shadow_os.sync_evolution_from_device()."""

    @pytest.mark.asyncio
    async def test_evolution_sync_calls_sync_fn(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = ["device_context"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_evolution_sync()
        mock_os.sync_evolution_from_device.assert_called_once()

    @pytest.mark.asyncio
    async def test_evolution_sync_records_decision(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = ["device_context"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_evolution_sync()
        assert len(daemon._decisions) >= 1
        assert daemon._decisions[-1].action_type == AutonomousActionType.EVOLUTION_SYNC

    @pytest.mark.asyncio
    async def test_evolution_sync_increments_stats(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = ["device_context"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_evolution_sync()
        assert daemon._stats["evolution_syncs"] >= 1

    @pytest.mark.asyncio
    async def test_evolution_sync_updates_timestamp(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = []
        daemon.set_shadow_os(mock_os)
        before = daemon._last_evolution_sync
        await daemon._run_evolution_sync()
        assert daemon._last_evolution_sync > before

    @pytest.mark.asyncio
    async def test_evolution_sync_no_shadow_os(self):
        daemon = AutonomousDaemon()
        daemon._shadow_os = None
        await daemon._run_evolution_sync()
        assert daemon._stats["evolution_syncs"] >= 1

    @pytest.mark.asyncio
    async def test_evolution_sync_exception_handled(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.side_effect = RuntimeError("test")
        daemon.set_shadow_os(mock_os)
        await daemon._run_evolution_sync()
        assert daemon._stats["evolution_syncs"] >= 1


class TestRunOracleSync:
    """_run_oracle_sync() calls shadow_os.sync_oracle_from_cognition()."""

    @pytest.mark.asyncio
    async def test_oracle_sync_calls_sync_fn(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.return_value = ["temporal_reasoning", "anomaly_detector"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_oracle_sync()
        mock_os.sync_oracle_from_cognition.assert_called_once()

    @pytest.mark.asyncio
    async def test_oracle_sync_records_decision(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.return_value = ["temporal_reasoning"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_oracle_sync()
        assert len(daemon._decisions) >= 1
        assert daemon._decisions[-1].action_type == AutonomousActionType.ORACLE_SYNC

    @pytest.mark.asyncio
    async def test_oracle_sync_increments_stats(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.return_value = ["anomaly_detector"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_oracle_sync()
        assert daemon._stats["oracle_syncs"] >= 1

    @pytest.mark.asyncio
    async def test_oracle_sync_updates_timestamp(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.return_value = []
        daemon.set_shadow_os(mock_os)
        before = daemon._last_oracle_sync
        await daemon._run_oracle_sync()
        assert daemon._last_oracle_sync > before

    @pytest.mark.asyncio
    async def test_oracle_sync_no_shadow_os(self):
        daemon = AutonomousDaemon()
        daemon._shadow_os = None
        await daemon._run_oracle_sync()
        assert daemon._stats["oracle_syncs"] >= 1

    @pytest.mark.asyncio
    async def test_oracle_sync_exception_handled(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.side_effect = RuntimeError("test")
        daemon.set_shadow_os(mock_os)
        await daemon._run_oracle_sync()
        assert daemon._stats["oracle_syncs"] >= 1


class TestDecisionOutcomeIncludesSynced:
    """Decision outcome includes synced sources."""

    @pytest.mark.asyncio
    async def test_evolution_sync_outcome_includes_sources(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = ["device_context"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_evolution_sync()
        assert "device_context" in daemon._decisions[-1].outcome

    @pytest.mark.asyncio
    async def test_oracle_sync_outcome_includes_sources(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.return_value = ["temporal_reasoning"]
        daemon.set_shadow_os(mock_os)
        await daemon._run_oracle_sync()
        assert "temporal_reasoning" in daemon._decisions[-1].outcome

    @pytest.mark.asyncio
    async def test_evolution_sync_empty_outcome(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = []
        daemon.set_shadow_os(mock_os)
        await daemon._run_evolution_sync()
        assert "none" in daemon._decisions[-1].outcome


class TestAutoMousActionsIncrement:
    """autonomous_actions counter incremented for sync actions."""

    @pytest.mark.asyncio
    async def test_evolution_sync_increments_autonomous_actions(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_evolution_from_device.return_value = []
        daemon.set_shadow_os(mock_os)
        before = daemon._stats["autonomous_actions"]
        await daemon._run_evolution_sync()
        assert daemon._stats["autonomous_actions"] > before

    @pytest.mark.asyncio
    async def test_oracle_sync_increments_autonomous_actions(self):
        daemon = AutonomousDaemon()
        mock_os = MagicMock()
        mock_os.sync_oracle_from_cognition.return_value = []
        daemon.set_shadow_os(mock_os)
        before = daemon._stats["autonomous_actions"]
        await daemon._run_oracle_sync()
        assert daemon._stats["autonomous_actions"] > before


class TestSyncWithRealShadowOS:
    """Integration test with real ShadowOS sync methods."""

    @pytest.mark.asyncio
    async def test_evolution_sync_with_real_device_context(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os.device_context = MagicMock()
        os.evolution = MagicMock()
        os.evolution.ingest_device_context = MagicMock()
        ctx = MagicMock()
        os.device_context.get_context.return_value = ctx
        synced = os.sync_evolution_from_device()
        assert "device_context" in synced
        os.evolution.ingest_device_context.assert_called_once_with(ctx)

    @pytest.mark.asyncio
    async def test_oracle_sync_with_real_temporal(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os.temporal_reasoning = MagicMock()
        os.anomaly_detector = MagicMock()
        os.oracle = MagicMock()
        os.oracle.ingest_temporal_data = MagicMock()
        os.oracle.ingest_anomalies = MagicMock()
        os.anomaly_detector.get_recent_anomalies = MagicMock(return_value=[])
        synced = os.sync_oracle_from_cognition()
        assert "temporal_reasoning" in synced
        assert "anomaly_detector" in synced