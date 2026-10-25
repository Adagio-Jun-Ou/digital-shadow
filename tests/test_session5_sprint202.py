"""Sprint 202: FeedLoop engine + deep integration regression tests."""

import pytest
from shadowd.cognition.feed_loop import (
    FeedLoop, FeedbackSignal, ResonanceEvent,
    _FEEDBACK_ROUTES, _REVERSE_ROUTES,
    _SANITIZE_RE, _SANITIZE_STRICT_RE,
)


class TestFeedLoopCore:
    """Core FeedLoop functionality tests."""

    def test_emit_creates_signals(self):
        fl = FeedLoop()
        ids = fl.emit("emotional_intelligence", "emotion", 0.8, {"primary": "stress"})
        assert len(ids) > 0
        stats = fl.get_stats()
        assert stats["signals_emitted"] == len(ids)

    def test_emit_with_broadcast_mode(self):
        fl = FeedLoop()
        ids = fl.emit("thought_sense", "cognitive", 0.7, {"intent": "question"})
        assert len(ids) > 0

    def test_emit_threshold_mode_filters_low_intensity(self):
        fl = FeedLoop()
        fl.set_contagion_threshold("cognitive", 0.5)
        ids = fl.emit("thought_sense", "cognitive", 0.1, {"low": True}, mode="threshold")
        routes = _FEEDBACK_ROUTES.get("thought_sense", [])
        high_routes = [r for r in routes if r[2] * 0.1 >= 0.5]
        assert len(ids) == len(high_routes)

    def test_poll_receives_targeted_signals(self):
        fl = FeedLoop()
        fl.emit("emotional_intelligence", "emotion", 0.8, {"primary": "stress"})
        signals = fl.poll("cognitive_trinity")
        assert len(signals) > 0
        assert signals[0].source_module == "emotional_intelligence"

    def test_poll_filters_by_signal_type(self):
        fl = FeedLoop()
        fl.emit("emotional_intelligence", "emotion", 0.8, {"primary": "stress"})
        signals = fl.poll("cognitive_trinity", signal_type="emotion")
        assert len(signals) > 0
        signals_wrong = fl.poll("cognitive_trinity", signal_type="anomaly")
        assert len(signals_wrong) == 0

    def test_poll_returns_deep_copies(self):
        fl = FeedLoop()
        fl.emit("anomaly_detector", "anomaly", 0.9, {"count": 3})
        sigs1 = fl.poll("attention_engine")
        sigs2 = fl.poll("attention_engine")
        if sigs1 and sigs2:
            assert sigs1[0].signal_id == sigs2[0].signal_id
            sigs1[0].content["modified"] = True
            assert "modified" not in sigs2[0].content

    def test_amplify_mode(self):
        fl = FeedLoop()
        ids = fl.emit("anomaly_detector", "anomaly", 0.5, {"type": "spike"}, mode="amplify")
        assert len(ids) > 0
        signals = fl.poll("attention_engine")
        assert len(signals) > 0
        assert signals[0].intensity >= 0.5

    def test_dampen_mode(self):
        fl = FeedLoop()
        ids = fl.emit("anomaly_detector", "anomaly", 0.8, {"type": "spike"}, mode="dampen")
        assert len(ids) > 0
        signals = fl.poll("attention_engine")
        assert len(signals) > 0
        assert signals[0].intensity <= 0.8

    def test_create_loop_and_complete(self):
        fl = FeedLoop()
        loop_id = fl.create_loop("thought_sense", "cognitive_trinity", "cognitive", 0.7, {"type": "question"})
        assert loop_id is not None
        assert len(fl.get_active_loops()) == 1
        outcome = fl.complete_loop(loop_id, 0.9, {"resolved": True})
        assert outcome is not None
        assert outcome.source_module == "cognitive_trinity"
        assert len(fl.get_active_loops()) == 0

    def test_complete_nonexistent_loop_returns_none(self):
        fl = FeedLoop()
        result = fl.complete_loop("nonexistent", 0.5, {})
        assert result is None

    def test_detect_resonance(self):
        fl = FeedLoop()
        for i in range(5):
            fl.emit("emotional_intelligence", "emotion", 0.9, {"primary": "anger", "intensity": f"high_{i}"})
        resonance = fl.detect_resonance(min_signals=3, min_intensity=0.3)
        assert len(resonance) > 0
        assert resonance[0].resonance_score > 0

    def test_detect_resonance_returns_empty_for_few_signals(self):
        fl = FeedLoop()
        fl.emit("goal_reasoning", "goal", 0.3, {"goal": "test"})
        resonance = fl.detect_resonance(min_signals=5, min_intensity=0.9)
        assert len(resonance) == 0

    def test_routing_table_has_all_sources(self):
        assert "emotional_intelligence" in _FEEDBACK_ROUTES
        assert "thought_sense" in _FEEDBACK_ROUTES
        assert "anomaly_detector" in _FEEDBACK_ROUTES
        assert "device_context" in _FEEDBACK_ROUTES
        assert "personality" in _FEEDBACK_ROUTES
        assert "dream_state" in _FEEDBACK_ROUTES

    def test_reverse_routes_populated(self):
        assert "cognitive_trinity" in _REVERSE_ROUTES
        assert "attention_engine" in _REVERSE_ROUTES
        assert "personality" in _REVERSE_ROUTES

    def test_get_routes(self):
        fl = FeedLoop()
        routes = fl.get_routes()
        assert "emotional_intelligence" in routes
        assert len(routes) > 10

    def test_get_reverse_routes(self):
        fl = FeedLoop()
        routes = fl.get_reverse_routes()
        assert "cognitive_trinity" in routes

    def test_signal_counts(self):
        fl = FeedLoop()
        fl.emit("emotional_intelligence", "emotion", 0.8, {"primary": "stress"})
        fl.emit("anomaly_detector", "anomaly", 0.9, {"type": "spike"})
        counts = fl.get_signal_counts()
        assert "emotion" in counts
        assert "anomaly" in counts

    def test_set_contagion_threshold(self):
        fl = FeedLoop()
        fl.set_contagion_threshold("emotion", 0.5)
        assert fl._contagion_thresholds["emotion"] == 0.5

    def test_clear(self):
        fl = FeedLoop()
        fl.emit("emotional_intelligence", "emotion", 0.8, {"primary": "stress"})
        fl.clear()
        assert len(fl._signals) == 0
        assert len(fl._active_loops) == 0

    def test_resonance_history(self):
        fl = FeedLoop()
        for i in range(5):
            fl.emit("emotional_intelligence", "emotion", 0.9, {"primary": "stress"})
        fl.detect_resonance(min_signals=3)
        history = fl.get_resonance_history()
        assert len(history) > 0

    def test_stats_thread_safe(self):
        fl = FeedLoop()
        import threading
        threads = []
        for i in range(10):
            t = threading.Thread(target=lambda: fl.emit("thought_sense", "cognitive", 0.5, {"i": 0}))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        stats = fl.get_stats()
        assert stats["signals_emitted"] > 0


class TestFeedbackSignal:
    """FeedbackSignal dataclass tests."""

    def test_signal_id_generated(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["target"], intensity=0.5, content={})
        assert len(s.signal_id) == 12

    def test_intensity_clamped(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["target"], intensity=1.5, content={})
        assert s.intensity == 1.0
        s2 = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["target"], intensity=-0.5, content={})
        assert s2.intensity == 0.0

    def test_sanitization(self):
        s = FeedbackSignal(signal_type="emotion", source_module="<script>alert(1)</script>", target_modules=["target"], intensity=0.5, content={"key<script>": "val<script>ue"})
        assert "<script>" not in s.source_module
        assert "<" not in s.source_module

    def test_to_dict(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test_mod", target_modules=["target1", "target2"], intensity=0.75, content={"key": "value"})
        d = s.to_dict()
        assert d["signal_type"] == "emotion"
        assert d["intensity"] == 0.75
        assert isinstance(d["target_modules"], list)
        assert isinstance(d["content"], dict)

    def test_to_dict_returns_deepcopy(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["t"], intensity=0.5, content={"k": "v"})
        d = s.to_dict()
        d["content"]["modified"] = True
        assert "modified" not in s.content


class TestResonanceEvent:
    """ResonanceEvent dataclass tests."""

    def test_resonance_score_clamped(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["t"], intensity=0.5, content={})
        r = ResonanceEvent(event_id="abc123", signals=[s], resonance_score=1.5, description="test")
        assert r.resonance_score == 1.0

    def test_description_sanitized(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["t"], intensity=0.5, content={})
        r = ResonanceEvent(event_id="abc", signals=[s], resonance_score=0.5, description="test\x00bad<script>")
        assert "\x00" not in r.description
        assert "<script>" in r.description  # content-safe preserves <>

    def test_to_dict(self):
        s = FeedbackSignal(signal_type="emotion", source_module="test", target_modules=["t"], intensity=0.5, content={})
        r = ResonanceEvent(event_id="abc", signals=[s], resonance_score=0.8, description="test resonance")
        d = r.to_dict()
        assert d["event_id"] == "abc"
        assert d["signal_count"] == 1
        assert d["resonance_score"] == 0.8


class TestFeedLoopMaxCaps:
    """Boundary condition tests."""

    def test_max_feedback_history(self):
        fl = FeedLoop()
        for i in range(2500):
            fl.emit("thought_sense", "cognitive", 0.5, {"i": i})
        assert len(fl._signals) <= 2000

    def test_max_active_loops(self):
        from shadowd.cognition.feed_loop import _MAX_ACTIVE_LOOPS
        fl = FeedLoop()
        for i in range(_MAX_ACTIVE_LOOPS + 10):
            fl.create_loop(f"source_{i % 5}", f"target_{i % 5}", "cognitive", 0.5, {})
        assert len(fl._active_loops) <= _MAX_ACTIVE_LOOPS

    def test_poll_limit(self):
        fl = FeedLoop()
        for i in range(50):
            fl.emit("emotional_intelligence", "emotion", 0.5 + i * 0.01, {})
        signals = fl.poll("cognitive_trinity", limit=5)
        assert len(signals) <= 5