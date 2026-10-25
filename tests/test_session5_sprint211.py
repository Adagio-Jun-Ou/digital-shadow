#!/usr/bin/env python3
"""Sprint 211: Cross-module integration stress tests for sprints 203-210 modules."""

import copy
import threading
import time
import unittest
from unittest.mock import MagicMock

from shadowd.cognition.anticipatory_engine import AnticipatoryEngine, AnticipationContext
from shadowd.cognition.cognitive_resonance import CognitiveResonanceDetector
from shadowd.cognition.temporal_self import TemporalSelf
from shadowd.cognition.health_dashboard import CognitiveHealthDashboard
from shadowd.cognition.creative_dream import CreativeDreamIntegration
from shadowd.cognition.social_graph_propagation import (
    SocialGraphPropagation, PropagationSource, RippleType,
)
from shadowd.cognition.dream_state import DreamState
from shadowd.cognition.creative_synthesis import CreativeSynthesis
from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, InfluenceType
from shadowd.cognition.empathy_mapping import EmpathyMapping
from shadowd.cognition.conflict_resolution import ConflictResolution
from shadowd.cognition.feed_loop import FeedLoop, SignalType


class TestAnticipatoryHealthIntegration(unittest.TestCase):
    def test_anticipatory_stats_feed_dashboard(self):
        ae = AnticipatoryEngine()
        hd = CognitiveHealthDashboard()
        ctx = AnticipationContext(time_of_day=9, emotional_state="neutral")
        signals = ae.predict(ctx)
        if signals:
            ae.validate(signals[0].signal_id, True)
        hd.auto_update({"anticipatory": ae})
        health = hd.get_module_health("anticipatory")
        self.assertIsNotNone(health)
        self.assertEqual(health.status, "healthy")

    def test_anticipatory_becomes_degraded(self):
        ae = AnticipatoryEngine()
        hd = CognitiveHealthDashboard()
        for i in range(20):
            ctx = AnticipationContext(time_of_day=9, emotional_state="neutral")
            signals = ae.predict(ctx)
            if signals:
                ae.validate(signals[0].signal_id, False)
        hd.auto_update({"anticipatory": ae})
        health = hd.get_module_health("anticipatory")
        self.assertIsNotNone(health)


class TestResonanceAnticipatoryIntegration(unittest.TestCase):
    def test_resonance_feeds_anticipatory(self):
        crd = CognitiveResonanceDetector()
        ae = AnticipatoryEngine()

        for i in range(10):
            crd.log_activity("module_a", "action", {"keyword": "deploy"})
            crd.log_activity("module_b", "action", {"keyword": "deploy"})

        results = crd.detect()
        ctx = AnticipationContext(time_of_day=14, active_domains=["work"],
                                 recent_actions=["action"])
        signals = ae.predict(ctx)
        self.assertIsNotNone(signals)


class TestTemporalSelfHealthIntegration(unittest.TestCase):
    def test_temporal_self_auto_update(self):
        ts = TemporalSelf()
        hd = CognitiveHealthDashboard()
        ts.record_milestone("career", "Promoted to senior")
        ts.record_milestone("health", "Ran first marathon")
        hd.auto_update({"temporal_self": ts})
        health = hd.get_module_health("temporal_self")
        self.assertIsNotNone(health)

    def test_temporal_self_identity_in_dashboard(self):
        ts = TemporalSelf()
        ts.add_identity_marker("learner", "curiosity", "Always studying new things", 0.8)
        identity = ts.get_identity()
        self.assertIn("identity_markers", identity)


class TestCreativeDreamHealthIntegration(unittest.TestCase):
    def test_creative_dream_feeds_dashboard(self):
        dream = DreamState()
        creative = CreativeSynthesis()
        cdi = CreativeDreamIntegration(dream, creative)
        hd = CognitiveHealthDashboard()

        for _ in range(3):
            cdi.run_cycle()

        hd.auto_update({"creative_dream": cdi})
        health = hd.get_module_health("creative_dream")
        self.assertIsNotNone(health)

    def test_creative_dream_cycle_increments_stats(self):
        dream = DreamState()
        creative = CreativeSynthesis()
        cdi = CreativeDreamIntegration(dream, creative)
        cdi.run_cycle()
        stats = cdi.get_stats()
        self.assertGreater(stats["cycles_run"], 0)


class TestSocialPropagationIntegration(unittest.TestCase):
    def test_ripple_from_interaction_to_empathy(self):
        sgp = SocialGraphPropagation()
        empathy = EmpathyMapping()
        p1 = empathy.add_person("Alice", "colleague")
        r = sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8,
                              "Alice interacted", person_id=p1)
        effects = sgp.propagate(r, empathy=empathy)
        self.assertGreater(len(effects), 0)

    def test_ripple_from_conflict_to_empathy(self):
        sgp = SocialGraphPropagation()
        empathy = EmpathyMapping()
        conflict = ConflictResolution()
        p1 = empathy.add_person("Bob", "friend")
        c = conflict.create_conflict("Dispute", "interest", 3)
        r = sgp.create_ripple("conflict_resolution", "conflict_created", 0.7,
                              "Bob in conflict", person_id=p1)
        effects = sgp.propagate(r, empathy=empathy)
        self.assertGreater(len(effects), 0)

    def test_ripple_from_empathy_to_conflict(self):
        sgp = SocialGraphPropagation()
        empathy = EmpathyMapping()
        conflict = ConflictResolution()
        p1 = empathy.add_person("Carol", "friend")
        empathy.record_interaction(p1, "text", -0.7, "very upset about the project")
        r = sgp.create_ripple("empathy_mapping", "trigger_detected", 0.7,
                              "Sensitive topic for Carol", person_id=p1)
        effects = sgp.propagate(r, conflict=conflict)
        self.assertGreater(len(effects), 0)

    def test_full_social_graph_build(self):
        sgp = SocialGraphPropagation()
        si = SocialIntelligence()
        empathy = EmpathyMapping()
        conflict = ConflictResolution()

        p1 = empathy.add_person("Alice", "colleague")
        p2 = empathy.add_person("Bob", "friend")
        empathy.record_interaction(p1, "meeting", 0.5, "productive")
        empathy.record_interaction(p2, "text", -0.3, "frustrated")

        g1 = si.create_group("Team", GroupType.PROFESSIONAL, members=["alice", "bob"])
        si.add_influence("alice", "bob", InfluenceType.PEER, 0.6)

        c = conflict.create_conflict("Budget dispute", "resource", 2)
        conflict.add_perspective(c.conflict_id, "Alice", "We need more budget")

        graph = sgp.build_graph(social_intel=si, empathy=empathy, conflict=conflict)
        self.assertGreater(graph["total_nodes"], 0)
        self.assertGreater(graph["total_edges"], 0)


class TestHealthDashboardAllModules(unittest.TestCase):
    def test_auto_update_all_new_modules(self):
        hd = CognitiveHealthDashboard()
        ae = AnticipatoryEngine()
        ae.predict(AnticipationContext(time_of_day=9))

        crd = CognitiveResonanceDetector()
        crd.log_activity("test", "action", {"key": "val"})

        ts = TemporalSelf()
        ts.record_milestone("career", "test milestone")

        dream = DreamState()
        creative = CreativeSynthesis()
        cdi = CreativeDreamIntegration(dream, creative)

        sgp = SocialGraphPropagation()
        sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test")

        modules_map = {
            "anticipatory": ae,
            "resonance": crd,
            "temporal_self": ts,
            "creative_dream": cdi,
            "social_propagation": sgp,
        }
        result = hd.auto_update(modules_map)
        self.assertEqual(result["modules_updated"], 5)

        report = hd.generate_report()
        self.assertEqual(report.module_count, 5)

    def test_concurrent_auto_update_many_modules(self):
        hd = CognitiveHealthDashboard()
        ae = AnticipatoryEngine()
        crd = CognitiveResonanceDetector()
        ts = TemporalSelf()
        dream = DreamState()
        creative = CreativeSynthesis()
        cdi = CreativeDreamIntegration(dream, creative)
        sgp = SocialGraphPropagation()

        modules_map = {
            "anticipatory": ae,
            "resonance": crd,
            "temporal_self": ts,
            "creative_dream": cdi,
            "social_propagation": sgp,
        }
        errors = []

        def do_update():
            try:
                hd.auto_update(modules_map)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=do_update) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestFeedLoopIntegrationWithNewModules(unittest.TestCase):
    def test_feed_loop_emits_signals(self):
        fl = FeedLoop()
        fl.emit("social_intelligence", SignalType.SOCIAL, 0.7,
                {"event": "interaction"}, "broadcast")
        fl.emit("anticipatory", SignalType.COGNITIVE, 0.5,
                {"event": "prediction"}, "targeted")

        resonances = fl.detect_resonance()
        stats = fl.get_stats()
        self.assertGreater(stats["signals_emitted"], 0)


class TestAnticipatoryTemporalSelfIntegration(unittest.TestCase):
    def test_anticipatory_sees_temporal_patterns(self):
        ae = AnticipatoryEngine()
        ts = TemporalSelf()

        ts.record_milestone("career", "Started new role")
        ts.track_decision("career", "accept_job", "long_term")
        ts.record_milestone("learning", "Completed certification")

        context = AnticipationContext(
            time_of_day=9,
            active_domains=["career", "learning"],
            recent_actions=["review", "plan"],
        )
        signals = ae.predict(context)
        self.assertIsNotNone(signals)


class TestCreativeDreamSocialIntegration(unittest.TestCase):
    def test_creative_dream_produces_content(self):
        dream = DreamState()
        creative = CreativeSynthesis()
        cdi = CreativeDreamIntegration(dream, creative)

        cdi.run_cycle()
        stats = cdi.get_stats()
        self.assertIn("cycles_run", stats)

        dream_insights = dream.get_insights()
        self.assertGreaterEqual(len(dream_insights), 0)


class TestEndToEndAllNewModules(unittest.TestCase):
    def test_full_chain(self):
        ae = AnticipatoryEngine()
        crd = CognitiveResonanceDetector()
        ts = TemporalSelf()
        dream = DreamState()
        creative = CreativeSynthesis()
        cdi = CreativeDreamIntegration(dream, creative)
        sgp = SocialGraphPropagation()
        hd = CognitiveHealthDashboard()
        empathy = EmpathyMapping()

        # Step 1: Log activities
        for i in range(10):
            crd.log_activity("module_%d" % (i % 3), "observe", {"keyword": "deploy"})

        # Step 2: Predict
        signals = ae.predict(AnticipationContext(time_of_day=14, active_domains=["work"],
                                                 recent_actions=["review", "deploy"]))
        self.assertIsNotNone(signals)

        # Step 3: Milestones
        ts.record_milestone("career", "Deployed v2")

        # Step 4: Creative dream
        cdi.run_cycle()

        # Step 5: Social ripple
        p1 = empathy.add_person("Dave", "colleague")
        r = sgp.create_ripple("social_intelligence", "interaction_recorded", 0.6,
                              "Dave interacted", person_id=p1)
        effects = sgp.propagate(r, empathy=empathy)
        self.assertGreater(len(effects), 0)

        # Step 6: Health dashboard
        result = hd.auto_update({
            "anticipatory": ae, "resonance": crd, "temporal_self": ts,
            "creative_dream": cdi, "social_propagation": sgp, "empathy": empathy,
        })
        self.assertEqual(result["modules_updated"], 6)

        # Step 7: Generate report
        report = hd.generate_report()
        self.assertEqual(report.module_count, 6)
        self.assertIn("overall_score", report.to_dict())


if __name__ == "__main__":
    unittest.main()