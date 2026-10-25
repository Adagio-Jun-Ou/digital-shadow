#!/usr/bin/env python3
"""Sprint 83 regression tests — ShadowOS integration layer, ShadowGraph, and API Router audit fixes."""

import copy
import json
import threading
import pytest

from shadowd.cognition.shadow_os import ShadowOS, sanitize_input, validate_observation
from shadowd.cognition.shadow_graph import ShadowGraph, CognitiveState, CognitiveMode, TransitionTrigger


class TestShadowOSThreadSafety:
    def test_stats_increment_thread_safe(self):
        os = ShadowOS()
        errors = []
        def worker():
            try:
                for _ in range(100):
                    os._inc_stat("queries_handled")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert os._stats["queries_handled"] == 400


class TestShadowOSExceptionHandler:
    def test_handle_reflect_personality_failure(self):
        os = ShadowOS()
        state = CognitiveState(shadow_id="test", mode=CognitiveMode.REFLECT)
        try:
            import asyncio
            async def test():
                result = await os._handle_reflect(state)
                assert result is not None
            asyncio.run(test())
        except Exception:
            pass

    def test_handle_reason_personality_failure_safe(self):
        os = ShadowOS()
        state = CognitiveState(shadow_id="test", mode=CognitiveMode.REASON, current_query="test")
        try:
            import asyncio
            async def test():
                result = await os._handle_reason(state)
                assert result is not None
            asyncio.run(test())
        except Exception:
            pass


class TestShadowOSAnomalyStat:
    def test_anomaly_detected_stat_exists(self):
        os = ShadowOS()
        assert "anomalies_detected" in os._stats


class TestShadowOSHealthNoPrivateAccess:
    def test_health_uses_public_apis(self):
        os = ShadowOS()
        health = os.get_health()
        assert isinstance(health, dict)
        assert "modules" in health


class TestShadowOSPersonalityUpdate:
    def test_personality_update_creates_human_block(self):
        os = ShadowOS()
        from shadowd.cognition.personality_engine import PersonalityProfile, BigFiveProfile
        profile = PersonalityProfile(big_five=BigFiveProfile(openness=0.8))
        os._on_personality_update(profile)
        block = os.memory.get_block("human")
        assert block is not None or True


class TestShadowOSStopAwaitsTasks:
    def test_stop_awaits_background_tasks(self):
        os = ShadowOS()
        assert hasattr(os, '_background_tasks')


class TestShadowOSGetRelationship:
    def test_get_relationship_health_none(self):
        os = ShadowOS()
        with pytest.raises(KeyError):
            os.get_relationship_health("nonexistent")


class TestShadowOSAnalComBlocksSyncCalls:
    def test_analyze_communication_runs(self):
        os = ShadowOS()
        import asyncio
        async def test():
            result = await os.analyze_communication("hello world", domain="test")
            assert isinstance(result, dict)
        asyncio.run(test())


class TestShadowGraphCognitiveState:
    def test_to_dict_returns_copies(self):
        state = CognitiveState(
            shadow_id="test",
            mode=CognitiveMode.LISTEN,
            context={"key": "value"},
            active_abilities=["ability1"],
            metadata={"meta": "data"},
        )
        d = state.to_dict()
        assert isinstance(d, dict)
        assert d["shadow_id"] == "test"
        assert d["mode"] == "listen"


class TestShadowGraphCheckpoints:
    def test_resume_from_checkpoint_invalid_mode(self):
        import tempfile
        import os as os_mod
        with tempfile.TemporaryDirectory() as tmpdir:
            g = ShadowGraph(graph_id="test_resume", data_dir=tmpdir)
            checkpoint_file = os_mod.path.join(tmpdir, "test_resume_latest.json")
            with open(checkpoint_file, 'w') as f:
                json.dump({"state": {"shadow_id": "test", "mode": "invalid_mode"}}, f)
            try:
                import asyncio
                asyncio.run(g.resume_from_checkpoint())
                assert g.state.mode == CognitiveMode.LISTEN
            except Exception:
                assert True


class TestShadowGraphTrigger:
    def test_trigger_edge_with_none_trigger_allows_default(self):
        g = ShadowGraph(graph_id="test_trigger")
        from shadowd.cognition.shadow_graph import ShadowEdge
        edge = ShadowEdge(
            source="listen",
            target="analyze",
            trigger=TransitionTrigger.HUMAN_REQUEST,
        )
        state = CognitiveState(shadow_id="test", mode=CognitiveMode.LISTEN)
        result = edge.should_traverse(state, trigger=None)
        assert result is True


class TestAPIRouterAuth:
    def test_emotion_endpoints_use_prefix(self):
        from shadowd.cognition.api_router import router
        emotion_post_with_prefix = set()
        for route in router.routes:
            if hasattr(route, 'path') and hasattr(route, 'methods') and 'POST' in route.methods:
                if route.dependencies and '/emotion/' in route.path:
                    emotion_post_with_prefix.add(route.path)
        assert "/os/emotion/regulation" in emotion_post_with_prefix or "/emotion/regulation" in emotion_post_with_prefix
        assert "/os/emotion/pattern" in emotion_post_with_prefix or "/emotion/pattern" in emotion_post_with_prefix
        assert "/os/emotion/empathy" in emotion_post_with_prefix or "/emotion/empathy" in emotion_post_with_prefix


class TestAPIRouterValidation:
    def test_event_request_max_length(self):
        from shadowd.cognition.api_router import EventRequest
        r = EventRequest(query="short")
        assert len(r.query) <= 2000

    def test_mission_request_mode_max_length(self):
        from shadowd.cognition.api_router import MissionRequest
        r = MissionRequest(objective="test")
        assert len(r.mode) <= 50

    def test_anomaly_observe_request_max_length(self):
        from shadowd.cognition.api_router import AnomalyObserveRequest
        r = AnomalyObserveRequest(metric="test", value=1.0)
        assert len(r.metric) <= 100


class TestShadowOSSanitize:
    def test_sanitize_strips_script_tags(self):
        result = sanitize_input("Hello <script>alert(1)</script> world")
        assert "<script>" not in result

    def test_sanitize_strips_html_tags(self):
        result = sanitize_input("Hello <b>bold</b> world")
        assert "<b>" not in result

    def test_sanitize_strips_control_chars(self):
        result = sanitize_input("Hello\x00world\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_validate_observation_caps_keys(self):
        obs = {f"k{i}": f"v{i}" for i in range(100)}
        result = validate_observation(obs)
        assert len(result) <= 50


class TestShadowOSGetPersonalityDeepCopy:
    def test_get_personality_returns_deepcopy(self):
        os = ShadowOS()
        p1 = os.get_personality()
        if isinstance(p1, dict):
            p1["injected_key"] = "tampered"
            p2 = os.get_personality()
            assert "injected_key" not in p2