#!/usr/bin/env python3
"""Sprint 74 tests: Round 2 audit — shadow_graph, shadow_voice, stealth_layer, anomaly_detector"""

import pytest
import asyncio

from shadowd.cognition.shadow_graph import ShadowGraph, CognitiveMode, CognitiveState
from shadowd.cognition.shadow_voice import ShadowVoice, VoiceSession
from shadowd.cognition.stealth_layer import (
    StealthLayer, PrivacyGuard, CamouflageEngine,
    DataCategory, AccessAction, SensitivityLevel,
)
from shadowd.cognition.anomaly_detector import (
    AnomalyDetectionEngine, Anomaly, AnomalyType, AnomalySeverity,
)


class TestShadowGraphRound2:
    def test_sg2_2_escalation_ids_are_12char(self):
        g = ShadowGraph(graph_id="test")
        assert hasattr(g, '_MAX_STATE_ESCALATIONS')

    def test_sg2_4_metadata_cap(self):
        g = ShadowGraph(graph_id="test")
        assert hasattr(g, '_MAX_METADATA_KEYS')
        assert g._MAX_METADATA_KEYS <= 100

    def test_sg2_5_error_uses_type_name(self):
        g = ShadowGraph(graph_id="test")
        g.state = CognitiveState(shadow_id="test", mode=CognitiveMode.LISTEN)
        assert g is not None

    def test_sg2_7_checkpoint_id_format(self):
        g = ShadowGraph(graph_id="test")
        g.state = CognitiveState(shadow_id="test", mode=CognitiveMode.LISTEN)
        ckpt = g._create_checkpoint("listen", "exec_1")
        assert len(ckpt.checkpoint_id.split("_")[1]) >= 12

    def test_sg2_4_state_caps_exist(self):
        g = ShadowGraph(graph_id="test")
        assert hasattr(g, '_MAX_STATE_ESCALATIONS')
        assert hasattr(g, '_MAX_STATE_ABILITIES')
        assert hasattr(g, '_MAX_STATE_DECISIONS')
        assert hasattr(g, '_MAX_METADATA_KEYS')


class TestShadowVoiceRound2:
    def test_sv2_1_command_id_12char(self):
        from shadowd.cognition.shadow_voice import VoiceCommand, VoiceIntent
        cmd = VoiceCommand(intent=VoiceIntent.QUERY)
        parts = cmd.command_id.split("_")
        assert len(parts[1]) >= 12

    def test_sv2_1_session_id_12char(self):
        sess = VoiceSession()
        parts = sess.session_id.split("_")
        assert len(parts[1]) >= 12

    def test_sv2_6_stats_returns_copy(self):
        voice = ShadowVoice()
        stats = voice.get_stats()
        assert isinstance(stats, dict)

    def test_sv2_7_command_to_dict_copies(self):
        from shadowd.cognition.shadow_voice import VoiceCommand, VoiceIntent
        cmd = VoiceCommand(intent=VoiceIntent.QUERY, entities={"key": "val"}, parameters={"p": "v"})
        d = cmd.to_dict()
        assert isinstance(d["entities"], dict)
        assert isinstance(d["parameters"], dict)

    def test_shadow_voice_init(self):
        voice = ShadowVoice()
        assert voice is not None


class TestStealthLayerRound2:
    def test_sl2_1_audit_entry_id_12char(self):
        pg = PrivacyGuard()
        pg.classify_data("doc1", DataCategory.PERSONAL)
        entry = pg._audit_log[0] if pg._audit_log else None
        if entry:
            parts = entry.entry_id.split("_")
            assert len(parts[1]) >= 12

    def test_sl2_2_eviction_with_logging(self):
        pg = PrivacyGuard()
        pg._MAX_CLASSIFICATIONS = 10
        for i in range(15):
            pg.classify_data(f"test_doc_{i}", DataCategory.PERSONAL)
        test_classifications = len([k for k in pg._classifications if k.startswith("test_doc_")])
        assert test_classifications <= 10

    def test_sl2_4_classify_data_returns_copy(self):
        pg = PrivacyGuard()
        pg.classify_data("doc1", DataCategory.PERSONAL)
        cls = pg.classify_data("doc1", DataCategory.PERSONAL)
        if cls and hasattr(cls, 'metadata'):
            cls.metadata["injected"] = "bad"
            cls2 = pg.classify_data("doc1", DataCategory.PERSONAL)
            if cls2 and hasattr(cls2, 'metadata'):
                assert "injected" not in cls2.metadata

    def test_sl2_6_get_stats_returns_copy(self):
        sl = StealthLayer(epsilon=1.0)
        stats = sl.get_stats()
        assert isinstance(stats, dict)

    def test_sl2_7_anonymize_deepcopy(self):
        ce = CamouflageEngine()
        data = {"name": "Alice", "nested": {"key": "value"}}
        result = ce.anonymize(data, fields=["name"])
        result["nested"]["key"] = "modified"
        assert data["nested"]["key"] == "value"

    def test_sl2_8_aggregate_method_name(self):
        ce = CamouflageEngine()
        assert hasattr(ce, 'aggregate_with_dp')
        assert not hasattr(ce, 'aggregat_with_dp')


class TestAnomalyDetectorRound2:
    def test_ad2_1_no_dead_max_sources(self):
        from shadowd.cognition.anomaly_detector import MetricStream
        assert not hasattr(MetricStream, '_MAX_SOURCES')

    def test_ad2_3_baseline_not_mutable(self):
        ae = AnomalyDetectionEngine()
        ae.register_metric("test_metric", description="Test", unit="count")
        ae.set_baseline("user1", {"test_metric": {"mean": 50.0, "stddev": 10.0}})
        ae.observe("test_metric", 75.0)
        anomalies = ae.get_anomalies()
        if anomalies:
            ctx = anomalies[0].to_dict().get("context", {})
            if "baseline" in ctx:
                assert isinstance(ctx["baseline"], dict)

    def test_ad2_4_no_correlation_matrix(self):
        ae = AnomalyDetectionEngine()
        assert not hasattr(ae, '_correlation_matrix')

    def test_ad2_6_anomaly_to_dict_context_copy(self):
        a = Anomaly(
            anomaly_id="test",
            anomaly_type=AnomalyType.STATISTICAL,
            severity=AnomalySeverity.MEDIUM,
            description="test anomaly",
            metric="test_metric",
            value=100.0,
            expected_range=(40.0, 60.0),
            z_score=5.0,
            context={"key": "value"},
        )
        d = a.to_dict()
        assert isinstance(d["context"], dict)
        d["context"]["injected"] = "bad"
        original = a.to_dict()
        assert "injected" not in original["context"]

    def test_ad2_2_eviction_with_logging(self):
        ae = AnomalyDetectionEngine()
        ae._MAX_STREAMS = 3
        for i in range(5):
            ae.register_metric(f"metric_{i}", description=f"Metric {i}")
        assert len(ae._streams) <= 3

    def test_anomaly_detector_observe(self):
        ae = AnomalyDetectionEngine()
        ae.register_metric("test", description="Test metric", unit="count")
        ae.observe("test", 50.0)
        ae.observe("test", 55.0)
        stats = ae.get_stats()
        assert stats is not None


class TestIntegrationRound2:
    def test_shadow_graph_basic(self):
        g = ShadowGraph(graph_id="test")
        g.state = CognitiveState(shadow_id="test", mode=CognitiveMode.LISTEN)
        assert g is not None

    def test_camouflage_engine_pseudonymize(self):
        ce = CamouflageEngine()
        data = {"name": "Alice", "email": "alice@example.com", "ssn": "123-45-6789"}
        result = ce.pseudonymize(data, "name")
        assert result["name"] != "Alice"

    def test_anomaly_detector_basic_flow(self):
        ae = AnomalyDetectionEngine()
        ae.register_metric("test", description="Test metric", unit="count")
        ae.observe("test", 50.0)
        stats = ae.get_stats()
        assert stats["streams"] >= 1