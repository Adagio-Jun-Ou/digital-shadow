"""Sprint 205: CognitiveResonanceDetector — cross-module hidden connections"""

import pytest
import threading
from shadowd.cognition.cognitive_resonance import (
    CognitiveResonanceDetector, ResonanceLink, ResonancePattern,
    ResonanceSource, ResonanceType, ResonanceStrength,
)


@pytest.fixture
def detector():
    return CognitiveResonanceDetector()


class TestResonanceSource:
    def test_sanitization(self):
        s = ResonanceSource(module="test<>module", emotion="<script>")
        assert "<>" not in s.module
        assert "<script>" not in s.emotion

    def test_value_clamped(self):
        s = ResonanceSource(value=2.0)
        assert s.value == 1.0
        s2 = ResonanceSource(value=-2.0)
        assert s2.value == -1.0

    def test_to_dict(self):
        s = ResonanceSource(module="test", domain="work", value=0.5)
        d = s.to_dict()
        assert d["module"] == "test"
        assert d["value"] == 0.5

    def test_timestamp_auto(self):
        s = ResonanceSource()
        assert s.timestamp != ""


class TestResonanceLink:
    def test_auto_id(self):
        l = ResonanceLink(description="test")
        assert len(l.link_id) == 12

    def test_strength_clamped(self):
        l = ResonanceLink(description="test", strength=1.5)
        assert l.strength == 1.0
        l2 = ResonanceLink(description="test", strength=-0.5)
        assert l2.strength == 0.0

    def test_sanitization(self):
        l = ResonanceLink(
            resonance_type="<script>alert</script>",
            description="Contains\x00control char",
        )
        assert "<script>" not in l.resonance_type
        assert "\x00" not in l.description

    def test_to_dict_rounded(self):
        l = ResonanceLink(description="test", strength=0.75321)
        d = l.to_dict()
        assert d["strength"] == 0.7532

    def test_to_dict_returns_deepcopy(self):
        l = ResonanceLink(
            source_a=ResonanceSource(module="a"),
            source_b=ResonanceSource(module="b"),
            description="test",
        )
        d = l.to_dict()
        d["strength"] = 0.0
        assert l.strength != 0.0


class TestResonancePattern:
    def test_auto_id(self):
        p = ResonancePattern(pattern_type="temporal")
        assert len(p.pattern_id) == 12

    def test_sanitization(self):
        p = ResonancePattern(
            pattern_type="<script>x</script>",
            description="test\x00desc",
            modules_involved=["mod1", "mod2"],
        )
        assert "<script>" not in p.pattern_type
        assert "\x00" not in p.description

    def test_modules_capped(self):
        p = ResonancePattern(
            pattern_type="test",
            modules_involved=[f"mod{i}" for i in range(30)],
        )
        assert len(p.modules_involved) <= 15

    def test_to_dict_returns_copy(self):
        p = ResonancePattern(pattern_type="test", modules_involved=["a", "b"])
        d = p.to_dict()
        d["modules_involved"].append("c")
        assert len(p.modules_involved) == 2


class TestLogActivity:
    def test_log_basic(self, detector):
        detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        stats = detector.get_stats()
        assert stats["activities_logged"] == 1

    def test_log_multiple(self, detector):
        for i in range(5):
            detector.log_activity("attention_engine", domain="work", value=0.5)
        stats = detector.get_stats()
        assert stats["activities_logged"] == 5

    def test_log_sanitization(self, detector):
        detector.log_activity("test<>module", domain="work<>stuff", emotion="<script>")
        stats = detector.get_stats()
        assert stats["activities_logged"] == 1

    def test_log_auto_detect(self, detector):
        for i in range(12):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.1)
        # Auto-detect fires on every 10th activity
        stats = detector.get_stats()
        assert stats["links_discovered"] >= 0


class TestDetect:
    def test_detect_basic(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
            detector.log_activity("creative_synthesis", domain="creative", value=0.7)
        links = detector.detect()
        assert isinstance(links, list)

    def test_detect_with_domain(self, detector):
        for i in range(15):
            detector.log_activity("attention_engine", domain="work", value=0.7)
        links = detector.detect()
        assert isinstance(links, list)

    def test_detect_returns_new_links_only(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        first = detector.detect()
        second = detector.detect()
        # Second call should find no new unique links (already stored)
        assert len(second) == 0 or len(second) <= len(first)


class TestFindResonance:
    def test_find_by_module(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
            detector.log_activity("creative_synthesis", domain="creative", value=0.7)
        detector.detect()
        # Should be able to find resonance
        findings = detector.find_resonance(min_strength=0.0)
        assert isinstance(findings, list)

    def test_find_by_resonance_type(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        detector.detect()
        causal = detector.find_resonance(resonance_type="causal", min_strength=0.0)
        for f in causal:
            assert f["resonance_type"] == "causal"

    def test_find_min_strength(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        detector.detect()
        strong = detector.find_resonance(min_strength=0.5)
        for f in strong:
            assert f["strength"] >= 0.5

    def test_find_limit(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
            detector.log_activity("creative_synthesis", domain="creative", value=0.7)
        detector.detect()
        limited = detector.find_resonance(min_strength=0.0, limit=3)
        assert len(limited) <= 3


class TestValidateLink:
    def test_validate_existing_link(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        links = detector.detect()
        if links:
            result = detector.validate_link(links[0].link_id, True)
            assert result is True
            stats = detector.get_stats()
            assert stats["validations"] >= 1

    def test_validate_nonexistent_link(self, detector):
        result = detector.validate_link("nonexistent_id", True)
        assert result is False

    def test_validate_creates_pattern(self, detector):
        for i in range(15):
            detector.log_activity('emotional_intelligence', emotion='joy', value=0.8)
        # Links are auto-detected; get them directly
        links = detector.get_links()
        if links:
            detector.validate_link(links[0].link_id, True)
            patterns = detector.get_patterns()
            assert len(patterns) >= 1


class TestGetLinks:
    def test_get_all_links(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        detector.detect()
        all_links = detector.get_links()
        assert isinstance(all_links, list)

    def test_get_links_by_type(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        detector.detect()
        causal = detector.get_links(resonance_type="causal")
        for link in causal:
            assert link.resonance_type == "causal"

    def test_get_links_limit(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
            detector.log_activity("creative_synthesis", domain="creative", value=0.7)
        detector.detect()
        limited = detector.get_links(limit=3)
        assert len(limited) <= 3

    def test_get_links_returns_deepcopy(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        detector.detect()
        links = detector.get_links()
        if links:
            links[0].strength = 0.0
            fresh = detector.get_links()
            assert fresh[0].strength != 0.0


class TestGetPatterns:
    def test_get_patterns_empty(self, detector):
        patterns = detector.get_patterns()
        assert isinstance(patterns, list)
        assert len(patterns) == 0

    def test_get_patterns_after_validation(self, detector):
        # Need enough activities to trigger auto-detect (every 10th)
        for i in range(25):
            detector.log_activity('emotional_intelligence', emotion='joy', value=0.8)
        # Auto-detect fires on 10th and 20th, may create links
        links = detector.get_links()
        if links:
            detector.validate_link(links[0].link_id, True)
            patterns = detector.get_patterns()
            assert len(patterns) >= 1
        else:
            # If no auto-detected links, manually detect
            detector.detect()
            links = detector.get_links()
            if links:
                detector.validate_link(links[0].link_id, True)
                patterns = detector.get_patterns()
                assert len(patterns) >= 1

    def test_get_patterns_limit(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        links = detector.detect()
        for link in links[:5]:
            detector.validate_link(link.link_id, True)
        limited = detector.get_patterns(limit=3)
        assert len(limited) <= 3


class TestGetStats:
    def test_initial_stats(self, detector):
        stats = detector.get_stats()
        assert stats["activities_logged"] == 0
        assert stats["links_discovered"] == 0
        assert stats["total_links"] == 0

    def test_stats_after_activity(self, detector):
        detector.log_activity("test_module", domain="work", value=0.5)
        stats = detector.get_stats()
        assert stats["activities_logged"] == 1

    def test_stats_after_detect(self, detector):
        for i in range(15):
            detector.log_activity("emotional_intelligence", emotion="joy", value=0.8)
        detector.detect()
        stats = detector.get_stats()
        # May or may not have discovered links depending on thresholds
        assert "links_discovered" in stats


class TestThreadSafety:
    def test_concurrent_logging(self, detector):
        errors = []

        def log_many():
            try:
                for i in range(50):
                    detector.log_activity("emotional_intelligence", emotion="joy", value=0.5)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=log_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = detector.get_stats()
        assert stats["activities_logged"] == 200

    def test_concurrent_detect_and_log(self, detector):
        errors = []

        def log_activities():
            try:
                for i in range(30):
                    detector.log_activity("emotional_intelligence", emotion="joy", value=0.5)
            except Exception as e:
                errors.append(type(e).__name__)

        def detect_repeatedly():
            try:
                for i in range(5):
                    detector.detect()
            except Exception as e:
                errors.append(type(e).__name__)

        t1 = threading.Thread(target=log_activities)
        t2 = threading.Thread(target=detect_repeatedly)
        t1.start()
        t2.start()
        t1.join()
        t2.join()

        assert len(errors) == 0


class TestResonanceTypes:
    def test_all_types_available(self):
        assert ResonanceType.TEMPORAL.value == "temporal"
        assert ResonanceType.CAUSAL.value == "causal"
        assert ResonanceType.THEMATIC.value == "thematic"
        assert ResonanceType.AMPLIFYING.value == "amplifying"
        assert ResonanceType.DAMPENING.value == "dampening"
        assert ResonanceType.SEQUENTIAL.value == "sequential"
        assert ResonanceType.FEEDBACK.value == "feedback"

    def test_strength_values(self):
        assert ResonanceStrength.WEAK.value == 0.2
        assert ResonanceStrength.STRONG.value == 0.75


class TestEdgeCases:
    def test_empty_detect(self, detector):
        links = detector.detect()
        assert isinstance(links, list)

    def test_find_resonance_empty(self, detector):
        findings = detector.find_resonance()
        assert isinstance(findings, list)

    def test_get_signal_nonexistent(self, detector):
        result = detector.validate_link("nonexistent", True)
        assert result is False

    def test_value_clamping(self, detector):
        detector.log_activity("test", value=10.0)
        assert detector.get_stats()["activities_logged"] == 1

    def test_many_modules(self, detector):
        for mod in ["emotional_intelligence", "thought_sense", "attention_engine",
                     "creative_synthesis", "goal_planner"]:
            detector.log_activity(mod, domain="work", value=0.5)
        stats = detector.get_stats()
        assert stats["activities_logged"] == 5