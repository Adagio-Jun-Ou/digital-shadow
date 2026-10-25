"""Sprint 206: TemporalSelf — self-history and continuity"""

import pytest
import threading
from shadowd.cognition.temporal_self import (
    TemporalSelf, SelfMilestone, IdentityMarker, SelfEpoch,
    MilestoneType, EpochTransition,
)


@pytest.fixture
def ts():
    return TemporalSelf()


class TestSelfMilestone:
    def test_auto_id(self):
        m = SelfMilestone(title="test")
        assert len(m.milestone_id) == 12

    def test_significance_clamped(self):
        m = SelfMilestone(title="test", significance=2.0)
        assert m.significance == 1.0
        m2 = SelfMilestone(title="test", significance=-0.5)
        assert m2.significance == 0.0

    def test_sanitization(self):
        m = SelfMilestone(
            milestone_type="<script>alert</script>",
            title="Contains\x00control",
            description="Description\nwith\nnewlines",
            tags=["tag1", "tag<>2"],
        )
        assert "<script>" not in m.milestone_type
        assert "\x00" not in m.title
        assert "\n" in m.description  # content-safe preserves newlines
        assert "<>" not in m.tags[1]  # tags use strict sanitization, strips <>

    def test_to_dict_rounded(self):
        m = SelfMilestone(title="test", significance=0.75321)
        d = m.to_dict()
        assert d["significance"] == 0.7532

    def test_to_dict_deepcopy(self):
        m = SelfMilestone(title="test", context={"key": "val"}, tags=["a", "b"])
        d = m.to_dict()
        d["context"]["key"] = "modified"
        assert m.context["key"] == "val"
        d["tags"].append("c")
        assert len(m.tags) == 2


class TestIdentityMarker:
    def test_auto_id(self):
        im = IdentityMarker(trait="test")
        assert im.marker_id

    def test_confidence_clamped(self):
        im = IdentityMarker(trait="test", confidence=2.0)
        assert im.confidence == 1.0

    def test_sanitization(self):
        im = IdentityMarker(
            category="<script>x</script>",
            trait="test<>trait",
            value="val\x00ue",
        )
        assert "<script>" not in im.category
        assert "<>" not in im.trait
        assert "\x00" not in im.value

    def test_to_dict(self):
        im = IdentityMarker(category="value", trait="honesty", value="high")
        d = im.to_dict()
        assert d["category"] == "value"
        assert d["confidence"] == 0.5


class TestSelfEpoch:
    def test_auto_id(self):
        e = SelfEpoch(name="test")
        assert len(e.epoch_id) == 12

    def test_sanitization(self):
        e = SelfEpoch(name="My Epoch<script>", description="Good\x00desc")
        assert "<script>" not in e.name

    def test_to_dict(self):
        e = SelfEpoch(name="test_epoch", key_traits=["curious", "adaptable"])
        d = e.to_dict()
        assert isinstance(d["key_traits"], list)
        assert d["name"] == "test_epoch"


class TestRecordMilestone:
    def test_record_basic(self, ts):
        m = ts.record_milestone("creation", "First Boot", significance=0.9)
        assert m.milestone_id
        assert m.milestone_type == "creation"
        assert m.title == "First Boot"

    def test_record_with_context(self, ts):
        m = ts.record_milestone("skill_gained", "Learned Python",
                                context={"module": "recall_synthesis"},
                                tags=["learning", "technical"])
        assert m.context.get("module") == "recall_synthesis"

    def test_milestone_capped(self, ts):
        for i in range(600):
            ts.record_milestone("creation", f"Milestone {i}")
        stats = ts.get_stats()
        assert stats["total_milestones"] <= 500

    def test_get_milestones(self, ts):
        ts.record_milestone("creation", "Boot", significance=0.5)
        ts.record_milestone("skill_gained", "Learned", significance=0.8)
        milestones = ts.get_milestones()
        assert len(milestones) >= 2
        assert milestones[0].significance >= milestones[1].significance

    def test_get_milestones_by_type(self, ts):
        ts.record_milestone("creation", "Boot")
        ts.record_milestone("skill_gained", "Learned")
        skill = ts.get_milestones(milestone_type="skill_gained")
        assert len(skill) == 1
        assert skill[0].milestone_type == "skill_gained"

    def test_get_milestone_by_id(self, ts):
        m = ts.record_milestone("creation", "Boot")
        found = ts.get_milestone(m.milestone_id)
        assert found is not None
        assert found.milestone_id == m.milestone_id

    def test_get_nonexistent_milestone(self, ts):
        assert ts.get_milestone("nonexistent") is None


class TestIdentityMarkers:
    def test_add_marker(self, ts):
        im = ts.add_identity_marker("communication_style", "tone", "helpful")
        assert im.trait == "tone"
        assert im.value == "helpful"

    def test_reinforce_marker(self, ts):
        ts.add_identity_marker("value", "honesty", "high", confidence=0.5)
        im = ts.add_identity_marker("value", "honesty", "very high", confidence=0.3)
        assert im.evidence_count == 2
        assert im.confidence > 0.5
        assert im.value == "very high"

    def test_identity_returns_categories(self, ts):
        ts.add_identity_marker("communication_style", "tone", "direct")
        ts.add_identity_marker("value", "honesty", "high")
        identity = ts.get_identity()
        assert "categories" in identity
        assert len(identity["top_traits"]) >= 1

    def test_marker_capped(self, ts):
        for i in range(250):
            ts.add_identity_marker("value", f"trait_{i}", f"val_{i}")
        stats = ts.get_stats()
        assert stats["total_identity_markers"] <= 200

    def test_invalid_category_defaults(self, ts):
        im = ts.add_identity_marker("invalid_category", "trait", "value")
        assert im.category == "value"


class TestTrackDecision:
    def test_track_basic(self, ts):
        decision_id = ts.track_decision("User asked question", "Provided answer")
        assert len(decision_id) == 12

    def test_track_with_outcome(self, ts):
        did = ts.track_decision("Complex task", "Used heuristic", outcome="success", confidence=0.8)
        log = ts.get_decision_log()
        assert len(log) >= 1
        assert log[0]["outcome"] == "success"

    def test_decision_log_capped(self, ts):
        for i in range(3000):
            ts.track_decision(f"context {i}", f"decision {i}")
        stats = ts.get_stats()
        assert stats["total_decisions"] <= 2000

    def test_decision_sanitization(self, ts):
        did = ts.track_decision("context\x00evil", "decision<script>", tags=["tag1"])
        log = ts.get_decision_log()
        assert "\x00" not in log[0]["context"]
        assert "decision<script>" in log[0]["decision"]


class TestSelfNarrative:
    def test_generate_narrative(self, ts):
        ts.record_milestone("creation", "First Boot", significance=0.9)
        ts.add_identity_marker("value", "curiosity", "high")
        ts.track_decision("Test context", "Test decision")
        narrative = ts.generate_self_narrative()
        assert "identity_summary" in narrative
        assert "recent_milestones" in narrative
        assert "recent_decisions" in narrative
        assert "coherence_score" in narrative
        assert narrative["coherence_score"] > 0

    def test_coherence_score(self, ts):
        ts.record_milestone("creation", "Boot", significance=0.9)
        ts.add_identity_marker("value", "honesty", "high", confidence=0.8)
        score = ts.get_coherence_score()
        assert 0.0 <= score <= 1.0

    def test_coherence_increases(self, ts):
        initial = ts.get_coherence_score()
        for i in range(20):
            ts.add_identity_marker("value", f"trait_{i}", f"val_{i}", confidence=0.7)
        ts.record_milestone("learning_breakthrough", "Major insight", significance=0.9)
        after = ts.get_coherence_score()
        assert after >= initial


class TestEpochs:
    def test_initial_epoch(self, ts):
        epochs = ts.get_epochs()
        assert len(epochs) >= 1
        assert epochs[0].name == "Initial Awakening"

    def test_high_significance_creates_epoch(self, ts):
        ts.record_milestone("value_realignment", "Core Shift", significance=0.9)
        epochs = ts.get_epochs()
        assert len(epochs) >= 2

    def test_abrupt_transition_for_very_significant(self, ts):
        ts.record_milestone("learning_breakthrough", "Paradigm Shift", significance=0.95)
        epochs = ts.get_epochs()
        latest = epochs[-1]
        assert latest.transition == "abrupt"


class TestGetStats:
    def test_initial_stats(self, ts):
        stats = ts.get_stats()
        assert stats["milestones_recorded"] == 0
        assert stats["identity_markers"] == 0
        assert stats["epochs_created"] >= 0

    def test_stats_after_operations(self, ts):
        ts.record_milestone("creation", "Boot")
        ts.add_identity_marker("value", "curiosity", "high")
        ts.track_decision("test", "test")
        stats = ts.get_stats()
        assert stats["milestones_recorded"] == 1
        assert stats["identity_markers"] >= 1
        assert stats["decisions_tracked"] >= 1


class TestThreadSafety:
    def test_concurrent_milestones(self, ts):
        errors = []

        def add_milestones():
            try:
                for i in range(50):
                    ts.record_milestone("creation", f"Milestone {i}")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=add_milestones) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert ts.get_stats()["milestones_recorded"] == 200

    def test_concurrent_identity_markers(self, ts):
        errors = []

        def add_markers():
            try:
                for i in range(30):
                    ts.add_identity_marker("value", f"trait_{i}", f"val_{i}")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=add_markers) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    def test_empty_identity(self, ts):
        identity = ts.get_identity()
        assert identity["total_markers"] == 0
        assert isinstance(identity["top_traits"], list)

    def test_empty_narrative(self, ts):
        narrative = ts.generate_self_narrative()
        assert narrative["total_milestones"] == 0
        assert narrative["total_decisions"] == 0

    def test_milestone_type_sanitization(self, ts):
        m = ts.record_milestone("<script>test</script>", "Title")
        assert "<script>" not in m.milestone_type

    def test_get_milestones_returns_deepcopy(self, ts):
        ts.record_milestone("creation", "Boot")
        m1 = ts.get_milestones()
        m2 = ts.get_milestones()
        assert m1[0].milestone_id == m2[0].milestone_id
        m1[0].title = "MODIFIED"
        assert m2[0].title != "MODIFIED"

    def test_epoch_key_traits_truncate(self, ts):
        ts.add_identity_marker("value", "honesty", "high", confidence=0.9)
        ts.record_milestone("learning_breakthrough", "Shift", significance=0.8)
        epochs = ts.get_epochs()
        for e in epochs:
            assert len(e.key_traits) <= 10