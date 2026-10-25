#!/usr/bin/env python3
"""Regression tests for Session 5 Sprint 30 — 4 feature improvements.

Feature 1: EscalationManager pending escalation timeout
Feature 2: NeuralMesh peer health scoring
Feature 3: PersonalityEngine behavior pattern detection expansion
Feature 4: ShadowGraph checkpointing improvements
"""

import asyncio
import json
import os
import tempfile
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# ---- Feature 1: EscalationManager pending escalation timeout ----

from shadowd.cognition.escalation import (
    EscalationManager,
    AutonomyLevel,
    ActionRequest,
    EscalationRecord,
    RiskCategory,
    RiskAssessment,
)


class TestEscalationTimeout:
    """Tests for pending escalation auto-expiry."""

    def setup_method(self):
        self.em = EscalationManager(data_dir=None)

    def test_default_timeout_hours(self):
        assert self.em._escalation_timeout_hours == 24

    def test_custom_timeout_hours(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=0.001)
        assert em._escalation_timeout_hours == 0.001

    @pytest.mark.asyncio
    async def test_expire_stale_escalations_auto_denies(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=0)
        record = EscalationRecord(
            action_request=ActionRequest(action_type="send_message"),
            level=AutonomyLevel.ASK,
            decision="pending_human",
            created_at=(datetime.now(timezone.utc) - timedelta(hours=1)).isoformat(),
        )
        em._escalation_queue.append(record)
        em._history = []
        expired = em.expire_stale_escalations()
        assert expired == 1
        assert record.decision == "denied"
        assert record.decided_by == "system_timeout"

    @pytest.mark.asyncio
    async def test_stale_escalation_count(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=1)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=2)).isoformat()
        recent_ts = datetime.now(timezone.utc).isoformat()
        r1 = EscalationRecord(level=AutonomyLevel.ASK, decision="pending_human", created_at=old_ts)
        r2 = EscalationRecord(level=AutonomyLevel.ASK, decision="pending_human", created_at=recent_ts)
        em._escalation_queue.extend([r1, r2])
        assert em.stale_escalation_count == 1

    @pytest.mark.asyncio
    async def test_get_pending_escalations_calls_expire(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=0)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        record = EscalationRecord(
            action_request=ActionRequest(action_type="send_message"),
            level=AutonomyLevel.ASK,
            decision="pending_human",
            created_at=old_ts,
        )
        em._escalation_queue.append(record)
        pending = em.get_pending_escalations()
        assert len(pending) == 0

    @pytest.mark.asyncio
    async def test_can_execute_calls_expire(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=0)
        old_ts = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
        record = EscalationRecord(
            level=AutonomyLevel.ASK,
            decision="pending_human",
            created_at=old_ts,
        )
        em._escalation_queue.append(record)
        await em.can_execute("observe", {})
        assert record.decision == "denied"
        assert record.decided_by == "system_timeout"

    def test_recent_escalation_not_expired(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=1)
        record = EscalationRecord(
            level=AutonomyLevel.ASK,
            decision="pending_human",
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        em._escalation_queue.append(record)
        expired = em.expire_stale_escalations()
        assert expired == 0
        assert record.decision == "pending_human"

    def test_zero_stale_when_empty(self):
        em = EscalationManager(data_dir=None, escalation_timeout_hours=24)
        assert em.stale_escalation_count == 0


# ---- Feature 2: NeuralMesh peer health scoring ----

from shadowd.cognition.neural_mesh import (
    NeuralMesh,
    MeshNode,
    NodeStatus,
)


class TestPeerHealthScoring:
    """Tests for NeuralMesh peer health tracking."""

    def setup_method(self):
        self.mesh = NeuralMesh(node_id="local_1", node_name="test-local")

    def test_add_peer_initializes_health(self):
        peer = MeshNode(node_id="peer_1", name="test-peer", address="10.0.0.1", port=8765)
        self.mesh.add_peer(peer)
        assert "peer_1" in self.mesh._peer_health
        h = self.mesh._peer_health["peer_1"]
        assert h["successes"] == 0
        assert h["failures"] == 0
        assert h["latency_ms"] == 0.0
        assert h["last_seen"] > 0

    def test_update_peer_health_success(self):
        peer = MeshNode(node_id="peer_1", name="test-peer", address="10.0.0.1", port=8765)
        self.mesh.add_peer(peer)
        self.mesh.update_peer_health("peer_1", success=True, latency_ms=50.0)
        h = self.mesh._peer_health["peer_1"]
        assert h["successes"] == 1
        assert h["failures"] == 0
        assert h["latency_ms"] == 50.0

    def test_update_peer_health_failure(self):
        peer = MeshNode(node_id="peer_1", name="test-peer", address="10.0.0.1", port=8765)
        self.mesh.add_peer(peer)
        self.mesh.update_peer_health("peer_1", success=False, latency_ms=0.0)
        h = self.mesh._peer_health["peer_1"]
        assert h["successes"] == 0
        assert h["failures"] == 1

    def test_update_peer_health_unknown_node(self):
        from shadowd.cognition.neural_mesh import MeshNode
        peer = MeshNode(node_id="unknown_peer", address="10.0.0.1", port=8765)
        self.mesh.add_peer(peer)
        self.mesh.update_peer_health("unknown_peer", success=True, latency_ms=10.0)
        assert "unknown_peer" in self.mesh._peer_health
        assert self.mesh._peer_health["unknown_peer"]["successes"] == 1

    def test_get_mesh_health_scoring(self):
        peer = MeshNode(node_id="peer_1", name="test-peer", address="10.0.0.1", port=8765)
        self.mesh.add_peer(peer)
        self.mesh.update_peer_health("peer_1", success=True, latency_ms=10.0)
        self.mesh.update_peer_health("peer_1", success=True, latency_ms=20.0)
        self.mesh.update_peer_health("peer_1", success=False, latency_ms=0.0)
        health = self.mesh.get_mesh_health()
        assert "peer_1" in health
        h = health["peer_1"]
        assert h["successes"] == 2
        assert h["failures"] == 1
        expected_score = (2 / 3) * 100
        assert abs(h["health_score"] - round(expected_score, 2)) < 0.01

    def test_get_mesh_health_zero_total(self):
        peer = MeshNode(node_id="peer_new", name="new-peer", address="10.0.0.2", port=8765)
        self.mesh.add_peer(peer)
        health = self.mesh.get_mesh_health()
        assert health["peer_new"]["health_score"] == 0.0

    def test_remove_peer_cleans_health(self):
        peer = MeshNode(node_id="peer_rm", name="rm-peer", address="10.0.0.3", port=8765)
        self.mesh.add_peer(peer)
        assert "peer_rm" in self.mesh._peer_health
        self.mesh.remove_peer("peer_rm")
        assert "peer_rm" not in self.mesh._peer_health

    @pytest.mark.asyncio
    async def test_send_updates_health(self):
        peer = MeshNode(node_id="peer_send", name="send-peer", address="10.0.0.4", port=8765)
        self.mesh.add_peer(peer)
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(message_type=MeshMessageType.PING, sender_id=self.mesh.local_node.node_id)
        result = await self.mesh.send("peer_send", msg)
        assert result is True
        assert self.mesh._peer_health["peer_send"]["successes"] == 1

    @pytest.mark.asyncio
    async def test_send_failure_updates_health(self):
        from shadowd.cognition.neural_mesh import MeshMessage, MeshMessageType
        msg = MeshMessage(message_type=MeshMessageType.PING, sender_id=self.mesh.local_node.node_id)
        result = await self.mesh.send("nonexistent_peer", msg)
        assert result is False


# ---- Feature 3: PersonalityEngine behavior pattern detection ----

from shadowd.cognition.personality_engine import (
    PersonalityEngine,
    PersonalityProfile,
    BehaviorPattern,
)


class TestCreativePatternDetection:
    """Tests for detect_creative_patterns."""

    def setup_method(self):
        self.pe = PersonalityEngine(data_dir=tempfile.mkdtemp())

    def test_creative_keywords_detected(self):
        result = self.pe.detect_creative_patterns(
            "task_completed", {"description": "I want to create a novel design"}
        )
        assert result["pattern"] == "creative_behavior"
        assert result["confidence"] > 0
        assert any("create" in e for e in result["evidence"])
        assert any("novel" in e for e in result["evidence"])

    def test_creative_no_keywords(self):
        result = self.pe.detect_creative_patterns(
            "task_completed", {"description": "just a regular task"}
        )
        assert result["confidence"] == 0.0
        assert len(result["evidence"]) == 0

    def test_creative_reinforces_pattern(self):
        self.pe.detect_creative_patterns("communication", {"text": "build something cool"})
        p = self.pe.profile.find_pattern("creative_behavior")
        assert p is not None
        assert p.evidence_count >= 1

    def test_creative_event_type_match(self):
        result = self.pe.detect_creative_patterns("create", {})
        assert result["pattern"] == "creative_behavior"
        assert len(result["evidence"]) > 0


class TestCollaborationPatternDetection:
    """Tests for detect_collaboration_patterns."""

    def setup_method(self):
        self.pe = PersonalityEngine(data_dir=tempfile.mkdtemp())

    def test_collaboration_keywords(self):
        result = self.pe.detect_collaboration_patterns(
            "communication", {"text": "let's work together as a team"}
        )
        assert result["pattern"] == "collaborative_behavior"
        assert result["confidence"] > 0
        assert any("together" in e for e in result["evidence"])
        assert any("team" in e for e in result["evidence"])

    def test_collaboration_group_size(self):
        result = self.pe.detect_collaboration_patterns(
            "social_interaction", {"group_size": 5}
        )
        assert result["pattern"] == "collaborative_behavior"
        assert any("group_size" in e for e in result["evidence"])

    def test_collaboration_no_keywords(self):
        result = self.pe.detect_collaboration_patterns(
            "decision_made", {"speed": "slow", "risk": "low"}
        )
        assert result["confidence"] == 0.0

    def test_collaboration_reinforces_pattern(self):
        self.pe.detect_collaboration_patterns("communication", {"text": "share this with the team"})
        p = self.pe.profile.find_pattern("collaborative_behavior")
        assert p is not None


class TestUrgencyPatternDetection:
    """Tests for detect_urgency_patterns."""

    def setup_method(self):
        self.pe = PersonalityEngine(data_dir=tempfile.mkdtemp())

    def test_urgency_keywords(self):
        result = self.pe.detect_urgency_patterns(
            "communication", {"text": "this is urgent, need it asap!"}
        )
        assert result["pattern"] == "urgency_driven"
        assert result["confidence"] > 0
        assert any("urgent" in e for e in result["evidence"])

    def test_urgency_high_urgency_level(self):
        result = self.pe.detect_urgency_patterns(
            "task_completed", {"urgency": 0.9}
        )
        assert any("urgency_level" in e for e in result["evidence"])

    def test_urgency_priority_keyword(self):
        result = self.pe.detect_urgency_patterns(
            "communication", {"priority": "critical", "text": "deadline approaching"}
        )
        assert len(result["evidence"]) >= 2

    def test_urgency_no_keywords(self):
        result = self.pe.detect_urgency_patterns(
            "communication", {"text": "regular update"}
        )
        assert result["confidence"] == 0.0

    def test_urgency_reinforces_pattern(self):
        self.pe.detect_urgency_patterns("communication", {"text": "rush this NOW"})
        p = self.pe.profile.find_pattern("urgency_driven")
        assert p is not None
        assert p.evidence_count >= 1


class TestPatternDetectionIntegration:
    """Integration tests: pattern detectors called from record_behavior."""

    def setup_method(self):
        self.pe = PersonalityEngine(data_dir=tempfile.mkdtemp())

    def test_record_behavior_triggers_creative(self):
        self.pe.record_behavior("creative", {"text": "I want to design a novel system"})
        p = self.pe.profile.find_pattern("creative_behavior")
        assert p is not None

    def test_record_behavior_triggers_collaboration(self):
        self.pe.record_behavior("communication", {"text": "Let's help each other out"})
        p = self.pe.profile.find_pattern("collaborative_behavior")
        assert p is not None

    def test_record_behavior_triggers_urgency(self):
        self.pe.record_behavior("communication", {"text": "urgent deadline approaching"})
        p = self.pe.profile.find_pattern("urgency_driven")
        assert p is not None


# ---- Feature 4: ShadowGraph checkpointing improvements ----

from shadowd.cognition.shadow_graph import (
    ShadowGraph,
    CognitiveState,
    CognitiveMode,
    ShadowNode,
    ShadowEdge,
    TransitionTrigger,
    ApprovalLevel,
    Checkpoint,
    ExecutionTrace,
)


class TestExportGraphState:
    """Tests for ShadowGraph.export_graph_state."""

    def setup_method(self):
        self.sg = ShadowGraph(graph_id="test_export", data_dir=tempfile.mkdtemp())

    def test_export_contains_state(self):
        state = self.sg.export_graph_state()
        assert "state" in state
        assert state["state"]["shadow_id"] == "test_export"
        assert state["state"]["mode"] == "listen"

    def test_export_contains_checkpoints(self):
        state = self.sg.export_graph_state()
        assert "checkpoints" in state
        assert isinstance(state["checkpoints"], list)

    def test_export_contains_traces(self):
        state = self.sg.export_graph_state()
        assert "last_traces" in state

    def test_export_contains_node_visit_counts(self):
        state = self.sg.export_graph_state()
        assert "node_visit_counts" in state
        assert isinstance(state["node_visit_counts"], dict)
        assert "listen" in state["node_visit_counts"]

    def test_export_contains_statistics(self):
        state = self.sg.export_graph_state()
        assert "statistics" in state
        assert "transitions" in state["statistics"]

    def test_export_contains_edges(self):
        state = self.sg.export_graph_state()
        assert "edges" in state
        assert len(state["edges"]) > 0

    def test_export_after_processing(self):
        original = self.sg.export_graph_state()
        assert original["statistics"]["transitions"] == 0


class TestImportGraphState:
    """Tests for ShadowGraph.import_graph_state."""

    def setup_method(self):
        self.sg = ShadowGraph(graph_id="test_import", data_dir=tempfile.mkdtemp())

    def test_roundtrip_export_import(self):
        self.sg.state.confidence = 0.75
        self.sg.state.urgency = 0.3
        exported = self.sg.export_graph_state()
        sg2 = ShadowGraph(graph_id="test_import2", data_dir=tempfile.mkdtemp())
        sg2.import_graph_state(exported)
        assert sg2.state.confidence == 0.75
        assert sg2.state.urgency == 0.3

    def test_import_restores_mode(self):
        self.sg.state.mode = CognitiveMode.ANALYZE
        exported = self.sg.export_graph_state()
        sg2 = ShadowGraph(graph_id="test_import3", data_dir=tempfile.mkdtemp())
        sg2.import_graph_state(exported)
        assert sg2.state.mode == CognitiveMode.ANALYZE

    def test_import_restores_statistics(self):
        self.sg._stats["transitions"] = 42
        self.sg._stats["by_mode"]["listen"] = 10
        exported = self.sg.export_graph_state()
        sg2 = ShadowGraph(graph_id="test_import4", data_dir=tempfile.mkdtemp())
        sg2.import_graph_state(exported)
        assert sg2._stats["transitions"] == 42
        assert sg2._stats["by_mode"]["listen"] == 10

    def test_import_with_empty_data(self):
        sg2 = ShadowGraph(graph_id="empty_import", data_dir=tempfile.mkdtemp())
        sg2.import_graph_state({})
        assert sg2.state.shadow_id == "empty_import"

    def test_import_restores_previous_mode(self):
        self.sg.state.mode = CognitiveMode.RESPOND
        self.sg.state.previous_mode = CognitiveMode.ANALYZE
        exported = self.sg.export_graph_state()
        sg2 = ShadowGraph(graph_id="test_import5", data_dir=tempfile.mkdtemp())
        sg2.import_graph_state(exported)
        assert sg2.state.previous_mode == CognitiveMode.ANALYZE


class TestGetNodeStatistics:
    """Tests for ShadowGraph.get_node_statistics."""

    def setup_method(self):
        self.sg = ShadowGraph(graph_id="test_stats", data_dir=tempfile.mkdtemp())

    def test_returns_all_nodes(self):
        stats = self.sg.get_node_statistics()
        assert len(stats) == len(self.sg.nodes)
        for name in self.sg.nodes:
            assert name in stats

    def test_visit_counts_initially_zero(self):
        stats = self.sg.get_node_statistics()
        for node_name, node_stats in stats.items():
            assert node_stats["visit_count"] == 0

    def test_visit_counts_updated(self):
        self.sg._stats["by_mode"]["listen"] = 5
        self.sg._stats["by_mode"]["analyze"] = 3
        stats = self.sg.get_node_statistics()
        assert stats["listen"]["visit_count"] == 5
        assert stats["analyze"]["visit_count"] == 3

    def test_statistics_structure(self):
        stats = self.sg.get_node_statistics()
        for node_name, node_stats in stats.items():
            assert "mode" in node_stats
            assert "visit_count" in node_stats
            assert "transitions_from" in node_stats
            assert "transitions_to" in node_stats

    def test_transitions_tracking(self):
        trace = ExecutionTrace(
            execution_id="test_exec",
            start_time=datetime.now(timezone.utc).isoformat(),
            nodes_visited=["listen", "analyze", "reason"],
        )
        self.sg.traces.append(trace)
        stats = self.sg.get_node_statistics()
        assert "listen->analyze" in stats["listen"]["transitions_from"]
        assert stats["listen"]["transitions_from"]["listen->analyze"] == 1
        assert "analyze->reason" in stats["analyze"]["transitions_from"]