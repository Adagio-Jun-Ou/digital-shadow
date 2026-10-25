import pytest
import copy
import time
import threading
from datetime import datetime, timedelta, timezone


class TestShadowEnhancedDeepAudit:
    def test_get_domain_summary_returns_deepcopy(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.data_dir = None
        ai.memory = None
        ai.insights_queue = []
        ai.current_context = {}
        ai.people = {}
        ai.decisions = {}
        ai.patterns = {}
        from shadowd.shadow_enhanced import LifeDomain
        ai.domains = {
            "career": LifeDomain(
                name="Career",
                description="Career domain",
                records=[{"content": "test record", "timestamp": datetime.now(timezone.utc).isoformat()}],
                insights=["insight1"],
                goals=[{"goal": "test", "timestamp": datetime.now(timezone.utc).isoformat()}],
            ),
        }
        result = ai.get_domain_summary("career")
        assert result["domain"] == "career"
        assert result["records_count"] == 1
        result["recent_records"].append({"content": "mutated"})
        result["insights"].append("mutated_insight")
        result["goals"].append({"mutated": True})
        assert len(ai.domains["career"].records) == 1
        assert len(ai.domains["career"].insights) == 1
        assert len(ai.domains["career"].goals) == 1

    def test_get_status_returns_context_copy(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.data_dir = None
        ai.current_context = {"domain": "career", "location": "home"}
        ai.insights_queue = []
        ai.people = {}
        ai.decisions = {}
        ai.patterns = {}
        ai.domains = {}
        ai.memory = type("M", (), {"get_stats": lambda self: {}})()
        status = ai.get_status()
        assert status["current_context"]["domain"] == "career"
        status["current_context"]["domain"] = "mutated"
        assert ai.current_context["domain"] == "career"

    def test_get_proactive_insights_under_lock(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.data_dir = None
        ai.current_context = {}
        ai.people = {}
        ai.decisions = {}
        ai.patterns = {}
        ai.domains = {}
        ai.memory = type("M", (), {"get_stats": lambda self: {}})()
        ai.insights_queue = [
            {"priority": "high", "type": "test", "message": "important", "timestamp": datetime.now(timezone.utc)},
            {"priority": "low", "type": "test", "message": "unimportant", "timestamp": datetime.now(timezone.utc)},
            {"priority": "medium", "type": "test", "message": "somewhat", "timestamp": datetime.now(timezone.utc)},
        ]
        result = ai.get_proactive_insights()
        assert len(result) == 2
        assert all(i["priority"] in ("high", "medium") for i in result)

    def test_get_domain_summary_unknown_domain(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.domains = {}
        result = ai.get_domain_summary("nonexistent")
        assert "error" in result


class TestShadowGraphSyncLock:
    def test_sync_lock_exists(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        graph = ShadowGraph(graph_id="test_sync_lock")
        assert hasattr(graph, '_sync_lock')
        assert isinstance(graph._sync_lock, type(threading.Lock()))

    def test_add_node_thread_safe(self):
        from shadowd.cognition.shadow_graph import ShadowGraph, ShadowNode, CognitiveMode
        graph = ShadowGraph(graph_id="test_add_node_ts")
        errors = []
        def add_nodes():
            try:
                for i in range(10):
                    graph.add_node(ShadowNode(name=f"node_{threading.current_thread().name}_{i}", mode=CognitiveMode.DORMANT))
            except Exception as e:
                errors.append(type(e).__name__)
        threads = [threading.Thread(target=add_nodes) for _ in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestStealthLayerStatsLock:
    def test_stats_lock_exists(self):
        from shadowd.cognition.stealth_layer import StealthLayer
        sl = StealthLayer()
        assert hasattr(sl, '_stats_lock')

    def test_get_stats_returns_copy(self):
        from shadowd.cognition.stealth_layer import StealthLayer
        sl = StealthLayer()
        stats = sl.get_stats()
        stats["stealth_stats"]["access_checks"] = 9999
        assert sl._stats["access_checks"] != 9999


class TestShadowVoiceStateLock:
    def test_get_state_returns_copy(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        v = ShadowVoice()
        state = v.get_state()
        assert isinstance(state, dict)
        assert "state" in state
        assert "stats" in state

    def test_get_stats_returns_copy(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        v = ShadowVoice()
        stats = v.get_stats()
        assert isinstance(stats, dict)
        assert "sessions_total" in stats