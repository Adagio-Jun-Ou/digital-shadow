#!/usr/bin/env python3
"""Sprint 79 — Round 2 audit: cognitive_reflection, decision_journal,
social_intelligence, attention_engine (sanitize deepcopy fixes)."""

import copy
import threading
import unittest
from datetime import datetime, timezone, timedelta

from shadowd.cognition.cognitive_reflection import (
    CognitiveReflection, ReflectionType,
)
from shadowd.cognition.decision_journal import (
    DecisionJournal, DecisionType, DecisionStatus,
)
from shadowd.cognition.social_intelligence import (
    SocialIntelligence, GroupType, InfluenceType, RoleType,
)
from shadowd.cognition.attention_engine import (
    AttentionEngine, AttentionLevel, FocusState, DistractionType,
)
import shadowd.cognition.cognitive_reflection as cr_mod
import shadowd.cognition.decision_journal as dj_mod
import shadowd.cognition.social_intelligence as si_mod
import shadowd.cognition.attention_engine as ae_mod


class TestCognitiveReflectionSprint79(unittest.TestCase):
    """CR-1: sanitize strips <>; CR-2: deepcopy returns; CR-4: deepcopy insights"""

    def test_cr1_sanitize_strips_angle_brackets(self):
        self.assertNotIn("<script>", cr_mod._sanitize("hello<script>world\x00test"))
        self.assertNotIn(">", cr_mod._sanitize("a>b"))

    def test_cr2_add_reflection_returns_deepcopy(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection("Test Topic", description="test desc",
                                  what_went_well=["thing1"])
        entry.topic = "MODIFIED"
        entry.what_went_well.append("extra")
        retrieved = cr.get_reflection(entry.entry_id)
        self.assertNotEqual(retrieved.topic, "MODIFIED")
        self.assertEqual(len(retrieved.what_went_well), 1)

    def test_cr3_get_reflection_returns_deepcopy(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection("Topic")
        r1 = cr.get_reflection(entry.entry_id)
        r2 = cr.get_reflection(entry.entry_id)
        r1.topic = "MOD1"
        r2.topic = "MOD2"
        r3 = cr.get_reflection(entry.entry_id)
        self.assertNotEqual(r3.topic, "MOD1")
        self.assertNotEqual(r3.topic, "MOD2")

    def test_cr4_extract_insights_returns_deepcopy(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            "Topic", what_went_well=["improved significantly"],
            what_could_improve=["should have tested better"],
            lessons_learned=["learned to iterate"]
        )
        insights = cr.extract_insights(entry.entry_id)
        if insights:
            insights[0].title = "MODIFIED"
            insights2 = cr.get_learning_insights()
            for i in insights2:
                self.assertNotEqual(i.title, "MODIFIED")

    def test_cr5_get_learning_insights_returns_deepcopy(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            "Topic", what_went_well=["discovered new approach"],
        )
        cr.extract_insights(entry.entry_id)
        insights = cr.get_learning_insights()
        if insights:
            insights[0].title = "MODIFIED"
            insights2 = cr.get_learning_insights()
            self.assertNotEqual(insights2[0].title, "MODIFIED")

    def test_cr6_list_reflections_returns_deepcopy(self):
        cr = CognitiveReflection()
        cr.add_reflection("Topic1")
        cr.add_reflection("Topic2")
        results = cr.list_reflections()
        results[0].topic = "MODIFIED"
        results2 = cr.list_reflections()
        self.assertNotEqual(results2[0].topic, "MODIFIED")

    def test_cr7_eviction_logging(self):
        cr = CognitiveReflection()
        max_ref = cr._MAX_REFLECTIONS
        for i in range(max_ref + 5):
            cr.add_reflection(f"Topic {i}")
        with cr._lock:
            self.assertLessEqual(len(cr._reflections), max_ref)

    def test_cr8_insight_eviction_logging(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            "Topic", what_went_well=["discovered something " + str(i) for i in range(15)],
        )
        cr.extract_insights(entry.entry_id)
        max_ins = cr._MAX_INSIGHTS
        entry2 = cr.add_reflection(
            "Topic2", what_went_well=["discovered item " + str(i) for i in range(max_ins + 5)],
        )
        cr.extract_insights(entry2.entry_id)
        with cr._lock:
            self.assertLessEqual(len(cr._insights), max_ins)

    def test_cr9_pattern_dedup(self):
        cr = CognitiveReflection()
        for _ in range(5):
            cr.add_reflection("Test", domain="coding",
                              what_went_well=["good code"],
                              what_could_improve=["bad tests"], sentiment=0.5)
        patterns = cr.detect_patterns(min_frequency=2)
        self.assertIsInstance(patterns, list)

    def test_cr10_confidence_trend(self):
        cr = CognitiveReflection()
        cr.add_reflection("T1", confidence_before=0.3, confidence_after=0.5)
        cr.add_reflection("T2", confidence_before=0.4, confidence_after=0.7)
        trend = cr.get_confidence_trend()
        self.assertEqual(trend["direction"], "learning")

    def test_cr11_invalid_reflection_type(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection("Topic", reflection_type="nonexistent_type")
        self.assertEqual(entry.reflection_type, ReflectionType.DAILY)


class TestDecisionJournalSprint79(unittest.TestCase):
    """DJ-1: sanitize strips <>; DJ-2: deepcopy returns; DJ-5: metadata values sanitized"""

    def test_dj1_sanitize_strips_angle_brackets(self):
        self.assertNotIn("<script>", dj_mod._sanitize("hello<script>world\x00test"))

    def test_dj2_record_decision_returns_deepcopy(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test Decision", chosen="Option A",
                                 alternatives=["Option B"])
        rec.title = "MODIFIED"
        rec.alternatives.append("extra")
        retrieved = dj.get_decision(rec.decision_id)
        self.assertNotEqual(retrieved.title, "MODIFIED")
        self.assertEqual(len(retrieved.alternatives), 1)

    def test_dj3_get_decision_returns_deepcopy(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Decision", chosen="opt1")
        r1 = dj.get_decision(rec.decision_id)
        r2 = dj.get_decision(rec.decision_id)
        r1.title = "MOD1"
        r2.title = "MOD2"
        r3 = dj.get_decision(rec.decision_id)
        self.assertNotEqual(r3.title, "MOD1")

    def test_dj4_list_decisions_returns_deepcopy(self):
        dj = DecisionJournal()
        dj.record_decision("Decision 1", chosen="a")
        dj.record_decision("Decision 2", chosen="b")
        results = dj.list_decisions()
        results[0].title = "MODIFIED"
        results2 = dj.list_decisions()
        self.assertNotEqual(results2[0].title, "MODIFIED")

    def test_dj5_metadata_values_sanitized(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt",
                                 metadata={"key": "a" * 600, "num": 42, "obj": {"nested": True}})
        retrieved = dj.get_decision(rec.decision_id)
        self.assertLessEqual(len(str(retrieved.metadata.get("key", ""))), 500)
        self.assertEqual(retrieved.metadata.get("num"), 42)

    def test_dj6_eviction_logging(self):
        dj = DecisionJournal()
        max_dec = dj._MAX_DECISIONS
        for i in range(max_dec + 5):
            dj.record_decision(f"Decision {i}", chosen="opt")
        with dj._lock:
            self.assertLessEqual(len(dj._decisions), max_dec)

    def test_dj7_record_outcome(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt", confidence=0.8)
        result = dj.record_outcome(rec.decision_id, "Success", 0.9)
        self.assertTrue(result)
        retrieved = dj.get_decision(rec.decision_id)
        self.assertEqual(retrieved.status, DecisionStatus.COMPLETED)

    def test_dj8_record_outcome_failed(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt", confidence=0.8)
        dj.record_outcome(rec.decision_id, "Failed", 0.2)
        retrieved = dj.get_decision(rec.decision_id)
        self.assertEqual(retrieved.status, DecisionStatus.FAILED)

    def test_dj9_record_outcome_nonexistent(self):
        dj = DecisionJournal()
        result = dj.record_outcome("nonexistent", "x", 0.5)
        self.assertFalse(result)

    def test_dj10_update_status(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt")
        result = dj.update_status(rec.decision_id, "executing")
        self.assertTrue(result)
        result = dj.update_status("nonexistent", "executing")
        self.assertFalse(result)

    def test_dj11_update_status_invalid(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt")
        result = dj.update_status(rec.decision_id, "invalid_status")
        self.assertFalse(result)

    def test_dj12_overconfidence_report(self):
        dj = DecisionJournal()
        for i in range(5):
            rec = dj.record_decision(f"Dec {i}", chosen="opt", confidence=0.9)
            dj.record_outcome(rec.decision_id, "result", 0.3)
        report = dj.get_overconfidence_report()
        self.assertGreater(report.get("overconfident_count", 0), 0)

    def test_dj13_decision_accuracy(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Dec", chosen="opt", confidence=0.7, domain="tech")
        dj.record_outcome(rec.decision_id, "ok", 0.8)
        acc = dj.get_decision_accuracy(domain="tech")
        self.assertEqual(acc["total_resolved"], 1)

    def test_dj14_run_retrospective(self):
        dj = DecisionJournal()
        for i in range(3):
            rec = dj.record_decision(f"Dec {i}", chosen="opt", confidence=0.8, domain="tech",
                                     biases_flagged=["confirmation"])
            dj.record_outcome(rec.decision_id, "ok", 0.6)
        retro = dj.run_retrospective(period_days=30)
        self.assertGreater(retro.decisions_reviewed, 0)

    def test_dj15_detect_patterns_overconfident(self):
        dj = DecisionJournal()
        for i in range(5):
            rec = dj.record_decision(f"Dec {i}", chosen="opt", confidence=0.9, domain="tech")
            dj.record_outcome(rec.decision_id, "fail", 0.3)
        patterns = dj.detect_patterns(min_frequency=3)
        self.assertIsInstance(patterns, list)
        pattern_types = [p.pattern_type for p in patterns]
        if pattern_types:
            self.assertIn("consistently_overconfident", pattern_types)

    def test_dj16_decision_id_12_chars(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt")
        self.assertEqual(len(rec.decision_id.split("_", 1)[1]), 12)

    def test_dj17_invalid_decision_type(self):
        dj = DecisionJournal()
        rec = dj.record_decision("Test", chosen="opt", decision_type="nonexistent")
        self.assertEqual(rec.decision_type, DecisionType.ROUTINE)


class TestSocialIntelligenceSprint79(unittest.TestCase):
    """SI-1: sanitize strips <>; SI-2/3: deepcopy returns"""

    def test_si1_sanitize_strips_angle_brackets(self):
        self.assertNotIn("<script>", si_mod._sanitize("hello<script>world\x00test", 100))

    def test_si2_create_group_returns_deepcopy(self):
        si = SocialIntelligence()
        group = si.create_group("Test Group", GroupType.PROJECT, members=["alice", "bob"])
        group.name = "MODIFIED"
        group.members.append("eve")
        retrieved = si.get_group(group.group_id)
        self.assertNotEqual(retrieved.name, "MODIFIED")
        self.assertEqual(len(retrieved.members), 2)

    def test_si3_add_influence_returns_deepcopy(self):
        si = SocialIntelligence()
        rel = si.add_influence("alice", "bob", InfluenceType.MENTOR, strength=0.8)
        rel.source = "MODIFIED"
        network = si.get_influence_network("alice")
        self.assertEqual(network["outgoing"][0]["source"], "alice")

    def test_si4_assign_role_returns_deepcopy(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROFESSIONAL)
        role = si.assign_role("alice", group.group_id, RoleType.LEADER)
        role.person_id = "MODIFIED"
        with si._lock:
            stored = si._roles.get(role.role_id)
        self.assertEqual(stored.person_id, "alice")

    def test_si5_get_group_returns_deepcopy(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT)
        g1 = si.get_group(group.group_id)
        g2 = si.get_group(group.group_id)
        g1.name = "MOD1"
        self.assertNotEqual(g2.name, "MOD1")

    def test_si6_list_groups_returns_deepcopy(self):
        si = SocialIntelligence()
        si.create_group("Team1", GroupType.PROJECT)
        si.create_group("Team2", GroupType.PROJECT)
        groups = si.list_groups()
        groups[0].name = "MODIFIED"
        groups2 = si.list_groups()
        self.assertNotEqual(groups2[0].name, "MODIFIED")

    def test_si7_add_member(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT)
        result = si.add_member(group.group_id, "charlie")
        self.assertTrue(result)

    def test_si7b_add_member_duplicate(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT, members=["alice"])
        result = si.add_member(group.group_id, "alice")
        self.assertTrue(result)

    def test_si7c_add_member_full(self):
        si = SocialIntelligence()
        members = [f"person_{i}" for i in range(si._MAX_MEMBERS)]
        group = si.create_group("Big Team", GroupType.PROJECT, members=members)
        result = si.add_member(group.group_id, "overflow")
        self.assertFalse(result)

    def test_si8_remove_member(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT, members=["alice"])
        result = si.remove_member(group.group_id, "alice")
        self.assertTrue(result)

    def test_si8b_remove_member_not_in_group(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT)
        result = si.remove_member(group.group_id, "nobody")
        self.assertFalse(result)

    def test_si9_group_eviction(self):
        si = SocialIntelligence()
        max_groups = si._MAX_GROUPS
        for i in range(max_groups + 5):
            si.create_group(f"Group {i}", GroupType.PROJECT)
        with si._lock:
            self.assertLessEqual(len(si._groups), max_groups)

    def test_si10_detect_communication_patterns(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT, members=["alice", "bob", "carol"])
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.8)
        si.add_influence("carol", "bob", InfluenceType.PEER, 0.6)
        patterns = si.detect_communication_patterns(group.group_id)
        self.assertGreater(len(patterns), 0)

    def test_si11_get_group_dynamics(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT, members=["alice", "bob"])
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.7)
        si.assign_role("alice", group.group_id, RoleType.LEADER)
        dynamics = si.get_group_dynamics(group.group_id)
        self.assertIn("cohesion_score", dynamics)
        self.assertIn("member_count", dynamics)

    def test_si12_find_influencers(self):
        si = SocialIntelligence()
        si.add_influence("alice", "bob", InfluenceType.AUTHORITY, 0.9)
        si.add_influence("alice", "carol", InfluenceType.MENTOR, 0.8)
        si.add_influence("dave", "eve", InfluenceType.PEER, 0.3)
        influencers = si.find_influencers(min_influence=0.7)
        self.assertGreater(len(influencers), 0)
        self.assertEqual(influencers[0]["person_id"], "alice")

    def test_si13_influence_network(self):
        si = SocialIntelligence()
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.7)
        si.add_influence("carol", "alice", InfluenceType.PEER, 0.5)
        network = si.get_influence_network("alice")
        self.assertEqual(network["outgoing_count"], 1)
        self.assertEqual(network["incoming_count"], 1)

    def test_si14_stats(self):
        si = SocialIntelligence()
        si.create_group("Team", GroupType.PROJECT, members=["a", "b"])
        si.add_influence("a", "b", InfluenceType.PEER)
        stats = si.get_stats()
        self.assertEqual(stats["groups_tracked"], 1)
        self.assertEqual(stats["influence_relations"], 1)

    def test_si15_id_12_chars(self):
        si = SocialIntelligence()
        group = si.create_group("Team", GroupType.PROJECT)
        self.assertEqual(len(group.group_id.split("_", 1)[1]), 12)
        rel = si.add_influence("a", "b", InfluenceType.PEER)
        self.assertEqual(len(rel.relation_id.split("_", 1)[1]), 12)
        role = si.assign_role("a", group.group_id, RoleType.CONTRIBUTOR)
        self.assertEqual(len(role.role_id.split("_", 1)[1]), 12)


class TestAttentionEngineSprint79(unittest.TestCase):
    """AE-1: sanitize strips <>"""

    def test_ae1_sanitize_strips_angle_brackets(self):
        ae = AttentionEngine()
        result = ae._sanitize("hello<script>world\x00test")
        self.assertNotIn("<script>", result)

    def test_ae2_add_attention_item(self):
        ae = AttentionEngine()
        item_id = ae.add_attention_item("Urgent task", level="critical", urgency=0.9)
        self.assertEqual(len(item_id), 12)

    def test_ae2b_add_attention_item_invalid_level(self):
        ae = AttentionEngine()
        with self.assertRaises(ValueError):
            ae.add_attention_item("Task", level="nonexistent")

    def test_ae2c_add_attention_item_empty_topic(self):
        ae = AttentionEngine()
        with self.assertRaises(ValueError):
            ae.add_attention_item("   ")

    def test_ae3_start_end_focus(self):
        ae = AttentionEngine()
        session_id = ae.start_focus("Deep work")
        self.assertIsNotNone(session_id)
        result = ae.end_focus()
        self.assertIsNotNone(result)

    def test_ae4_start_focus_ends_previous(self):
        ae = AttentionEngine()
        ae.start_focus("First task")
        ae.start_focus("Second task")
        stats = ae.get_stats()
        self.assertEqual(stats["focus_sessions"], 1)

    def test_ae5_record_distraction(self):
        ae = AttentionEngine()
        event_id = ae.record_distraction("notification", "Phone buzzed", 30.0)
        self.assertEqual(len(event_id), 12)

    def test_ae5b_record_distraction_invalid(self):
        ae = AttentionEngine()
        with self.assertRaises(ValueError):
            ae.record_distraction("invalid_type")

    def test_ae6_get_priorities(self):
        ae = AttentionEngine()
        ae.add_attention_item("Low task", level="low", urgency=0.1, importance=0.1)
        ae.add_attention_item("High task", level="high", urgency=0.9, importance=0.9)
        priorities = ae.get_priorities(limit=2)
        self.assertEqual(len(priorities), 2)
        self.assertEqual(priorities[0].topic, "High task")

    def test_ae7_detect_focus_state(self):
        ae = AttentionEngine()
        ae.add_attention_item("Task", level="low", urgency=0.1)
        state = ae.detect_focus_state()
        self.assertIn(state.value, ["idle", "scanning", "distracted", "active", "deep_work", "fatigued"])

    def test_ae7b_focus_state_active(self):
        ae = AttentionEngine()
        ae.add_attention_item("Task", level="high", urgency=0.9)
        ae.start_focus("Working")
        state = ae.detect_focus_state()
        self.assertIn(state.value, ["active", "deep_work", "scanning"])

    def test_ae8_attention_profile(self):
        ae = AttentionEngine()
        ae.start_focus("Work")
        ae.end_focus()
        profile = ae.get_attention_profile()
        self.assertIsInstance(profile.avg_focus_span_minutes, float)

    def test_ae9_suggest_attention_action(self):
        ae = AttentionEngine()
        ae.add_attention_item("Task", level="medium")
        action = ae.suggest_attention_action()
        self.assertIn("current_state", action)
        self.assertIn("suggestions", action)

    def test_ae10_recover_focus(self):
        ae = AttentionEngine()
        ae.start_focus("Deep work")
        ae._fatigue_level = 0.5
        ae._cognitive_load = 0.8
        ae.recover_focus(10.0)
        self.assertLess(ae._fatigue_level, 0.5)
        self.assertLess(ae._cognitive_load, 0.8)

    def test_ae11_eviction(self):
        ae = AttentionEngine()
        max_items = ae._MAX_ATTENTION_ITEMS
        for i in range(max_items + 5):
            ae.add_attention_item(f"Task {i}", level="low")
        with ae._lock:
            self.assertLessEqual(len(ae._attention_items), max_items)

    def test_ae12_item_sanitization(self):
        ae = AttentionEngine()
        item_id = ae.add_attention_item("Task<script>alert(1)</script>", level="medium",
                                        domain="test<script>")
        with ae._lock:
            item = ae._attention_items[item_id]
        self.assertIn("<script>", item.topic)
        self.assertNotIn("<script>", item.domain)

    def test_ae13_metadata_cap(self):
        ae = AttentionEngine()
        big_meta = {f"key_{i}": f"val_{i}" for i in range(50)}
        item_id = ae.add_attention_item("Task", level="medium", metadata=big_meta)
        with ae._lock:
            item = ae._attention_items[item_id]
        self.assertLessEqual(len(item.metadata), 20)

    def test_ae14_deep_work_detection(self):
        ae = AttentionEngine()
        ae.start_focus("Important work")
        session = ae.end_focus()
        if session and session.duration_minutes >= ae._DEEP_WORK_THRESHOLD_MINUTES:
            self.assertIn("deep_work_sessions", ae.get_stats())


class TestConcurrentAccessSprint79(unittest.TestCase):
    def test_decision_journal_concurrent(self):
        dj = DecisionJournal()
        errors = []

        def writer():
            try:
                for i in range(50):
                    dj.record_decision(f"Dec {i}", chosen="opt")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    dj.list_decisions()
                    dj.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer),
                   threading.Thread(target=writer),
                   threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_cognitive_reflection_concurrent(self):
        cr = CognitiveReflection()
        errors = []

        def writer():
            try:
                for i in range(50):
                    cr.add_reflection(f"Topic {i}")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    cr.list_reflections()
                    cr.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer),
                   threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)


if __name__ == "__main__":
    unittest.main()