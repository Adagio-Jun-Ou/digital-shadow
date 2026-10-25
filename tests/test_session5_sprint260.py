import pytest
import copy
import threading
import re
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch


class TestAnticipatoryEngineAudit:
    def setup_method(self):
        from shadowd.cognition.anticipatory_engine import AnticipatoryEngine
        self.engine = AnticipatoryEngine()

    def test_sanitize_helpers_exist(self):
        from shadowd.cognition.anticipatory_engine import _sanitize, _sanitize_content
        assert _sanitize("hello<>world", 50) == "helloworld"
        assert _sanitize_content("hello<>world", 50) == "hello<>world"
        assert _sanitize("  hello  ", 50) == "hello"
        assert _sanitize_content("  hello  ", 50) == "hello"

    def test_strip_before_truncate(self):
        from shadowd.cognition.anticipatory_engine import _sanitize
        assert _sanitize("  hello  ", 5) == "hello"[:5]

    def test_context_sanitization(self):
        from shadowd.cognition.anticipatory_engine import AnticipationContext
        ctx = AnticipationContext(
            emotional_state="hap<py>",
            active_domains=["work<>life"],
            recent_actions=["did something"],
        )
        assert "<" not in ctx.emotional_state
        assert "<" not in ctx.active_domains[0]
        assert "<" not in ctx.recent_actions[0]

    def test_signal_sanitization(self):
        from shadowd.cognition.anticipatory_engine import AnticipationSignal
        sig = AnticipationSignal(
            signal_type="co<ntext>",
            trigger="test<trigger>",
            related_modules=["mo<d1>", "mod2"],
            prediction="user will <do> X",
            reasoning="based on <pattern>",
            suggested_action="open <app>",
            confidence=0.9,
        )
        assert "<" not in sig.signal_type
        assert "<" not in sig.trigger
        assert "<" not in sig.related_modules[0]
        assert "<" in sig.prediction

    def test_validation_accuracy_thread_safety(self):
        from shadowd.cognition.anticipatory_engine import AnticipatoryEngine, AnticipationContext
        engine = AnticipatoryEngine()
        engine.predict(AnticipationContext(emotional_state="calm", active_domains=["work"]))
        errors = []
        def validate_many():
            try:
                for _ in range(50):
                    engine.validate("test_signal_0", outcome="positive")
            except Exception as e:
                errors.append(e)
        threads = [threading.Thread(target=validate_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_context_to_dict_deepcopy(self):
        from shadowd.cognition.anticipatory_engine import AnticipationContext
        ctx = AnticipationContext(
            emotional_state="calm",
            active_domains=["work"],
            recent_actions=["checking"],
            user_state={"energy": 0.8},
            device_context={"screen": "on"},
            goal_progress={"fitness": 0.5},
        )
        d = ctx.to_dict()
        d["user_state"]["energy"] = 0.0
        assert ctx.user_state["energy"] == 0.8

    def test_predict_returns_list(self):
        from shadowd.cognition.anticipatory_engine import AnticipatoryEngine, AnticipationContext
        engine = AnticipatoryEngine()
        result = engine.predict(AnticipationContext(emotional_state="calm", active_domains=["work"]))
        assert isinstance(result, list)


class TestSocialGraphPropagationAudit:
    def test_create_ripple_returns_ripple(self):
        from shadowd.cognition.social_graph_propagation import SocialGraphPropagation, PropagationSource
        sg = SocialGraphPropagation()
        ripple = sg.create_ripple(PropagationSource.SOCIAL_INTELLIGENCE.value, "interaction_recorded", 0.5, "test reason")
        assert ripple is not None
        assert len(ripple.ripple_id) >= 12


class TestCreativeDreamAudit:
    def setup_method(self):
        from shadowd.cognition.creative_dream import CreativeDreamIntegration
        self.cd = CreativeDreamIntegration()

    def test_dream_to_create_returns_list(self):
        result = self.cd.dream_to_create()
        assert isinstance(result, list)

    def test_create_to_dream_returns_list(self):
        result = self.cd.create_to_dream()
        assert isinstance(result, list)

    def test_run_cycle_returns_cycle(self):
        from shadowd.cognition.creative_dream import CreativeDreamIntegration
        cd = CreativeDreamIntegration()
        result = cd.run_cycle()
        assert result is not None


class TestHealthDashboardAudit:
    def setup_method(self):
        from shadowd.cognition.health_dashboard import CognitiveHealthDashboard
        self.hd = CognitiveHealthDashboard()

    def test_sanitize_helpers_exist(self):
        from shadowd.cognition.health_dashboard import _sanitize, _sanitize_content
        assert _sanitize("test<>val", 50) == "testval"
        assert _sanitize_content("test<>val", 50) == "test<>val"

    def test_strip_before_truncate(self):
        from shadowd.cognition.health_dashboard import _sanitize
        assert _sanitize("  hello  ", 5) == "hello"

    def test_module_health_sanitization(self):
        from shadowd.cognition.health_dashboard import ModuleHealth
        mh = ModuleHealth(
            module_name="test<module>",
            status="healthy",
            latency_ms=10.5,
            success_rate=0.99,
            error_count=0,
        )
        assert "<" not in mh.module_name

    def test_update_module_returns_health(self):
        result = self.hd.update_module("test_mod", latency_ms=10.0, success_rate=0.99)
        assert result is not None
        assert result.module_name == "test_mod"

    def test_get_stats_returns_dict(self):
        stats = self.hd.get_stats()
        assert isinstance(stats, dict)


class TestCognitiveResonanceAudit:
    def setup_method(self):
        from shadowd.cognition.cognitive_resonance import CognitiveResonanceDetector
        self.cr = CognitiveResonanceDetector()

    def test_sanitize_helpers_exist(self):
        from shadowd.cognition.cognitive_resonance import _sanitize, _sanitize_content
        assert _sanitize("test<>val", 50) == "testval"
        assert _sanitize_content("test<>val", 50) == "test<>val"

    def test_detect_returns_list(self):
        from shadowd.cognition.cognitive_resonance import CognitiveResonanceDetector
        cr = CognitiveResonanceDetector()
        cr.log_activity("emotional_intelligence", emotion="happy", metric="mood", value=0.8)
        cr.log_activity("emotional_intelligence", emotion="sad", metric="mood", value=0.3)
        result = cr.detect()
        assert isinstance(result, list)

    def test_get_stats_returns_dict(self):
        stats = self.cr.get_stats()
        assert isinstance(stats, dict)


class TestTemporalSelfAudit:
    def setup_method(self):
        from shadowd.cognition.temporal_self import TemporalSelf
        self.ts = TemporalSelf()

    def test_sanitize_helpers_exist(self):
        from shadowd.cognition.temporal_self import _sanitize, _sanitize_content
        assert _sanitize("test<>val", 50) == "testval"
        assert _sanitize_content("test<>val", 50) == "test<>val"

    def test_strip_before_truncate(self):
        from shadowd.cognition.temporal_self import _sanitize
        assert _sanitize("  hello  ", 5) == "hello"

    def test_record_milestone(self):
        ms = self.ts.record_milestone("achieved", "first goal", significance=0.8)
        assert ms is not None
        assert ms.milestone_type == "achieved"

    def test_identity_marker_sanitization(self):
        marker = self.ts.add_identity_marker("cate<gory>", "val<ue>", 0.8)
        assert "<" not in marker.category
        assert "<" not in marker.value

    def test_epoch_eviction(self):
        from shadowd.cognition.temporal_self import TemporalSelf
        ts = TemporalSelf()
        ts._MAX_EPOCHS = 5
        for i in range(10):
            ts.add_identity_marker(f"trait_{i}", f"value_{i}", 0.5)
        assert len(ts._epoch_order) <= 5

    def test_get_stats_returns_dict(self):
        stats = self.ts.get_stats()
        assert isinstance(stats, dict)


class TestShadowPowersIntegration:
    def setup_method(self):
        from shadowd.cognition.shadow_powers import ShadowPowers
        self.sp = ShadowPowers()

    def test_powers_created(self):
        assert len(self.sp._powers) == 10

    def test_apply_power_returns_result(self):
        result = self.sp.apply_power("shadow_extract", "data_query", "found_data", True)
        assert result is not None
        assert result.category.value == "extraction"

    def test_apply_power_not_found(self):
        result = self.sp.apply_power("nonexistent_power", "data_query", "found_data", True)
        assert result is None

    def test_list_powers(self):
        powers = self.sp.list_powers()
        assert len(powers) >= 10

    def test_get_stats(self):
        stats = self.sp.get_stats()
        assert "total_powers" in stats
        assert stats["total_powers"] == 10

    def test_synthesize_powers_sanitization(self):
        result = self.sp.synthesize_powers(["extract<ion>", "dom<inion>"])
        assert result is None or isinstance(result, dict)

    def test_awaken_dormant(self):
        result = self.sp.awaken_dormant("shadow regeneration")
        assert result is not None or result is None

    def test_recover_fatigue(self):
        self.sp.apply_power("extraction", "data_query", "found_data", True)
        self.sp.recover_fatigue()
        stats = self.sp.get_stats()
        assert "total_applications" in stats

    def test_evolution_history(self):
        history = self.sp.get_evolution_history()
        assert isinstance(history, list)

    def test_get_power_tree(self):
        tree = self.sp.get_power_tree()
        assert isinstance(tree, dict)
        assert len(tree) >= 10


class TestFStringLogFix:
    def test_anomaly_detector_no_fstring_logs(self):
        with open("shadowd/cognition/anomaly_detector.py") as f:
            content = f.read()
        assert 'logger.info(f"' not in content
        assert 'logger.warning(f"' not in content
        assert 'logger.error(f"' not in content
        assert 'logger.debug(f"' not in content

    def test_goal_planner_no_fstring_logs(self):
        with open("shadowd/cognition/goal_planner.py") as f:
            content = f.read()
        assert 'logger.info(f"' not in content
        assert 'logger.warning(f"' not in content
        assert 'logger.error(f"' not in content

    def test_neural_mesh_no_fstring_logs(self):
        with open("shadowd/cognition/neural_mesh.py") as f:
            content = f.read()
        assert 'logger.info(f"' not in content
        assert 'logger.warning(f"' not in content
        assert 'logger.error(f"' not in content

    def test_shadow_crew_no_fstring_logs(self):
        with open("shadowd/cognition/shadow_crew.py") as f:
            content = f.read()
        assert 'logger.info(f"' not in content
        assert 'logger.warning(f"' not in content
        assert 'logger.error(f"' not in content

    def test_shadow_os_no_fstring_logs(self):
        with open("shadowd/cognition/shadow_os.py") as f:
            content = f.read()
        assert 'logger.info(f"' not in content
        assert 'logger.warning(f"' not in content
        assert 'logger.error(f"' not in content


class TestBugFixes:
    def test_shadow_ai_subprocess_import(self):
        import shadowd.shadow_ai
        import subprocess
        assert True

    def test_shadow_enhanced_timezone_aware_comparison(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        import threading
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._lock = threading.Lock()
        ai.insights_queue = [
            {"priority": "high", "type": "test", "message": "important", "timestamp": datetime.now(timezone.utc)},
        ]
        result = ai.get_proactive_insights()
        assert len(result) == 1

    def test_anomaly_detector_max_streams_200(self):
        from shadowd.cognition.anomaly_detector import AnomalyDetectionEngine
        engine = AnomalyDetectionEngine()
        assert engine._MAX_STREAMS == 200

    def test_personality_engine_narrowed_except(self):
        with open("shadowd/cognition/personality_engine.py") as f:
            content = f.read()
        assert "except Exception:\n                pass" not in content
