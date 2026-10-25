#!/usr/bin/env python3
"""Sprint 209 tests: SocialGraphPropagation — influence ripple across social modules."""

import copy
import threading
import time
import unittest
from unittest.mock import MagicMock, patch

from shadowd.cognition.social_graph_propagation import (
    SocialGraphPropagation, PropagationSource, RippleType, RippleStrength,
    PropagationRule, InfluenceRipple, RippleEffect,
    SocialGraphNode, SocialGraphEdge, GraphNodeType,
    _sanitize, _sanitize_content, _MAX_RIPPLES, _MAX_EFFECTS,
    _MAX_RULES, _MAX_GRAPH_NODES, _MAX_GRAPH_EDGES,
)


class TestSanitization(unittest.TestCase):
    def test_strict_strips_lt_gt(self):
        self.assertNotIn("<", _sanitize("hello <world>", 50))
        self.assertNotIn(">", _sanitize("hello <world>", 50))

    def test_content_preserves_lt_gt(self):
        self.assertIn("<", _sanitize_content("latency < 100ms"))
        self.assertIn(">", _sanitize_content("x > 5"))

    def test_strict_strips_control_chars(self):
        self.assertNotIn("\t", _sanitize("has\ttab", 50))
        self.assertNotIn("\n", _sanitize("has\nnewline", 50))

    def test_content_preserves_formatting(self):
        self.assertIn("\t", _sanitize_content("has\ttab"))
        self.assertIn("\n", _sanitize_content("has\nnewline"))

    def test_strips_before_truncation(self):
        self.assertEqual(_sanitize("  hello  ", 5), "hello")

    def test_max_length(self):
        self.assertEqual(len(_sanitize("a" * 300, 200)), 200)


class TestPropagationRule(unittest.TestCase):
    def test_to_dict(self):
        r = PropagationRule("r1", PropagationSource.EMPATHY_MAPPING, RippleType.TRIGGER_DETECTED,
                            PropagationSource.CONFLICT_RESOLUTION, 0.7, 0.3)
        d = r.to_dict()
        self.assertEqual(d["source"], "empathy_mapping")
        self.assertEqual(d["ripple_type"], "trigger_detected")
        self.assertEqual(d["target"], "conflict_resolution")
        self.assertEqual(d["weight"], 0.7)

    def test_weight_clamped(self):
        r = PropagationRule("r1", PropagationSource.SOCIAL_INTELLIGENCE,
                            RippleType.INTERACTION_RECORDED,
                            PropagationSource.EMPATHY_MAPPING, 5.0, 0.0)
        self.assertEqual(r.to_dict()["weight"], 2.0)

    def test_threshold_clamped(self):
        r = PropagationRule("r1", PropagationSource.SOCIAL_INTELLIGENCE,
                            RippleType.INTERACTION_RECORDED,
                            PropagationSource.EMPATHY_MAPPING, 0.5, -0.5)
        self.assertEqual(r.to_dict()["strength_threshold"], 0.0)


class TestInfluenceRipple(unittest.TestCase):
    def test_to_dict(self):
        r = InfluenceRipple("rip_1", PropagationSource.SOCIAL_INTELLIGENCE,
                            RippleType.INTERACTION_RECORDED, 0.8,
                            "p1", "g1", None, "test reason")
        d = r.to_dict()
        self.assertEqual(d["ripple_id"], "rip_1")
        self.assertEqual(d["source"], "social_intelligence")
        self.assertEqual(d["strength"], 0.8)
        self.assertFalse(d["propagated"])

    def test_strength_clamped(self):
        r = InfluenceRipple("rip_1", PropagationSource.CONFLICT_RESOLUTION,
                            RippleType.CONFLICT_ESCALATED, 1.5,
                            None, None, "c1", "test")
        self.assertEqual(r.to_dict()["strength"], 1.0)

    def test_metadata_deepcopy(self):
        meta = {"key": [1, 2, 3]}
        r = InfluenceRipple("rip_1", PropagationSource.EMPATHY_MAPPING,
                            RippleType.SENTIMENT_SHIFT, 0.5,
                            "p1", None, None, "test", metadata=meta)
        d = r.to_dict()
        d["metadata"]["key"].append(4)
        self.assertEqual(len(r.metadata["key"]), 3)

    def test_reason_truncated(self):
        r = InfluenceRipple("rip_1", PropagationSource.SOCIAL_INTELLIGENCE,
                            RippleType.INTERACTION_RECORDED, 0.5,
                            None, None, None, "x" * 1000)
        self.assertLessEqual(len(r.to_dict()["reason"]), 500)


class TestRippleEffect(unittest.TestCase):
    def test_to_dict(self):
        e = RippleEffect("eff_1", "rip_1", PropagationSource.EMPATHY_MAPPING,
                         "noted_person", 0.6, "test notes")
        d = e.to_dict()
        self.assertEqual(d["effect_id"], "eff_1")
        self.assertEqual(d["target_module"], "empathy_mapping")
        self.assertEqual(d["impact_score"], 0.6)

    def test_impact_clamped(self):
        e = RippleEffect("eff_1", "rip_1", PropagationSource.CONFLICT_RESOLUTION,
                         "action", 2.0, "notes")
        self.assertEqual(e.to_dict()["impact_score"], 1.0)


class TestSocialGraphNode(unittest.TestCase):
    def test_to_dict(self):
        n = SocialGraphNode("p1", GraphNodeType.PERSON, "Alice", 75.0, 0.8, 2, 1)
        d = n.to_dict()
        self.assertEqual(d["node_type"], "person")
        self.assertEqual(d["empathy_score"], 75.0)
        self.assertEqual(d["label"], "Alice")

    def test_scores_clamped(self):
        n = SocialGraphNode("p1", GraphNodeType.PERSON, "Bob", 200.0, -0.5, 0, 0)
        d = n.to_dict()
        self.assertEqual(d["empathy_score"], 100.0)
        self.assertEqual(d["influence_score"], 0.0)

    def test_metadata_deepcopy(self):
        n = SocialGraphNode("p1", GraphNodeType.PERSON, "Eve", 50.0, 0.5, 0, 0,
                            metadata={"k": [1]})
        d = n.to_dict()
        d["metadata"]["k"].append(2)
        self.assertEqual(len(n.metadata["k"]), 1)


class TestSocialGraphEdge(unittest.TestCase):
    def test_to_dict(self):
        e = SocialGraphEdge("e1", "p1", "p2", "empathy_affinity", 0.7, True)
        d = e.to_dict()
        self.assertEqual(d["edge_type"], "empathy_affinity")
        self.assertEqual(d["weight"], 0.7)
        self.assertTrue(d["bidirectional"])

    def test_weight_clamped(self):
        e = SocialGraphEdge("e1", "p1", "p2", "test", 2.0)
        self.assertEqual(e.to_dict()["weight"], 1.0)


class TestSocialGraphPropagationInit(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_default_rules_loaded(self):
        rules = self.sgp.get_rules()
        self.assertEqual(len(rules), 15)

    def test_initial_stats(self):
        stats = self.sgp.get_stats()
        self.assertEqual(stats["ripples_created"], 0)
        self.assertEqual(stats["rules_active"], 15)

    def test_empty_ripples(self):
        self.assertEqual(len(self.sgp.get_ripples()), 0)


class TestCreateRipple(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_basic_ripple(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test")
        self.assertEqual(r.source, PropagationSource.SOCIAL_INTELLIGENCE)
        self.assertEqual(r.ripple_type, RippleType.INTERACTION_RECORDED)
        self.assertEqual(r.strength, 0.8)
        self.assertFalse(r.propagated)
        self.assertEqual(self.sgp.get_stats()["ripples_created"], 1)

    def test_invalid_source(self):
        with self.assertRaises(ValueError):
            self.sgp.create_ripple("invalid", "interaction_recorded", 0.5, "test")

    def test_invalid_ripple_type(self):
        with self.assertRaises(ValueError):
            self.sgp.create_ripple("social_intelligence", "invalid", 0.5, "test")

    def test_strength_clamped_high(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 1.5, "test")
        self.assertEqual(r.strength, 1.0)

    def test_strength_clamped_low(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", -0.5, "test")
        self.assertEqual(r.strength, 0.0)

    def test_person_id_sanitized(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test",
                                    person_id="p<script>")
        self.assertNotIn("<", r.person_id)

    def test_metadata_capped(self):
        big_meta = {"k%d" % i: "v" for i in range(50)}
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test",
                                    metadata=big_meta)
        self.assertLessEqual(len(r.metadata), 20)

    def test_all_ripple_types(self):
        for rt in RippleType:
            r = self.sgp.create_ripple("social_intelligence", rt.value, 0.5, "test")
            self.assertIsInstance(r, InfluenceRipple)

    def test_max_ripples_bounded(self):
        for i in range(_MAX_RIPPLES + 100):
            self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test %d" % i)
        self.assertLessEqual(len(self.sgp.get_ripples(limit=9999)), _MAX_RIPPLES)


class TestPropagation(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def _make_mock_empathy(self, person_id="p1"):
        m = MagicMock()
        m.get_person_profile.return_value = MagicMock(person_id=person_id, name="Alice")
        return m

    def _make_mock_conflict(self):
        return MagicMock()

    def _make_mock_social(self):
        return MagicMock()

    def test_propagate_interaction_recorded(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.7, "test",
                                    person_id="p1")
        effects = self.sgp.propagate(r, empathy=self._make_mock_empathy(),
                                     conflict=self._make_mock_conflict())
        self.assertGreater(len(effects), 0)
        self.assertTrue(r.propagated)
        self.assertEqual(self.sgp.get_stats()["ripples_propagated"], 1)

    def test_propagate_conflict_escalated(self):
        r = self.sgp.create_ripple("conflict_resolution", "conflict_escalated", 0.8, "escalated!",
                                    person_id="p1")
        effects = self.sgp.propagate(r, empathy=self._make_mock_empathy("p1"),
                                     social_intel=self._make_mock_social())
        self.assertGreater(len(effects), 0)
        target_modules = {e.target_module.value for e in effects}
        self.assertIn("empathy_mapping", target_modules)

    def test_below_threshold_no_propagation(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.05, "weak")
        effects = self.sgp.propagate(r, empathy=self._make_mock_empathy())
        self.assertEqual(len(effects), 0)

    def test_no_target_modules(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test")
        effects = self.sgp.propagate(r)
        self.assertEqual(len(effects), 0)

    def test_disabled_rule(self):
        self.sgp.toggle_rule("r01", False)
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test",
                                    person_id="p1")
        effects = self.sgp.propagate(r, empathy=self._make_mock_empathy())
        self.assertEqual(len(effects), 0)

    def test_propagate_conflict_created(self):
        r = self.sgp.create_ripple("conflict_resolution", "conflict_created", 0.7, "new conflict",
                                    person_id="p1")
        effects = self.sgp.propagate(r, empathy=self._make_mock_empathy("p1"))
        self.assertGreater(len(effects), 0)

    def test_propagate_trigger_detected(self):
        r = self.sgp.create_ripple("empathy_mapping", "trigger_detected", 0.7, "sensitive topic",
                                    person_id="p1")
        effects = self.sgp.propagate(r, conflict=self._make_mock_conflict())
        self.assertGreater(len(effects), 0)

    def test_propagate_sentiment_shift(self):
        r = self.sgp.create_ripple("empathy_mapping", "sentiment_shift", 0.6, "mood change",
                                    person_id="p1")
        effects = self.sgp.propagate(r, social_intel=self._make_mock_social())
        self.assertGreater(len(effects), 0)

    def test_propagate_cohesion_shift(self):
        r = self.sgp.create_ripple("social_intelligence", "cohesion_shift", 0.5, "group unstable",
                                    group_id="g1")
        effects = self.sgp.propagate(r, conflict=self._make_mock_conflict())
        self.assertGreater(len(effects), 0)

    def test_propagate_relationship_health_shift(self):
        r = self.sgp.create_ripple("empathy_mapping", "relationship_health_shift", 0.6, "declining",
                                    person_id="p1")
        effects = self.sgp.propagate(r, conflict=self._make_mock_conflict())
        self.assertGreater(len(effects), 0)

    def test_effect_impact_is_weighted(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test",
                                    person_id="p1")
        effects = self.sgp.propagate(r, empathy=self._make_mock_empathy())
        for e in effects:
            self.assertLess(e.impact_score, 0.8)
            self.assertGreater(e.impact_score, 0.0)


class TestCustomRules(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_add_rule(self):
        r = self.sgp.add_rule("social_intelligence", "group_membership_changed",
                              "conflict_resolution", 0.6, 0.3)
        self.assertEqual(r.source, PropagationSource.SOCIAL_INTELLIGENCE)
        self.assertEqual(r.target, PropagationSource.CONFLICT_RESOLUTION)
        self.assertTrue(r.enabled)

    def test_add_invalid_source(self):
        with self.assertRaises(ValueError):
            self.sgp.add_rule("invalid", "interaction_recorded", "empathy_mapping", 0.5, 0.3)

    def test_add_invalid_target(self):
        with self.assertRaises(ValueError):
            self.sgp.add_rule("social_intelligence", "interaction_recorded", "invalid", 0.5, 0.3)

    def test_toggle_rule(self):
        r = self.sgp.add_rule("social_intelligence", "influence_added",
                              "empathy_mapping", 0.5, 0.3)
        self.assertTrue(self.sgp.toggle_rule(r.rule_id, False))
        rules = self.sgp.get_rules()
        found = [x for x in rules if x.rule_id == r.rule_id]
        self.assertFalse(found[0].enabled)

    def test_toggle_nonexistent(self):
        self.assertFalse(self.sgp.toggle_rule("bad_id", True))

    def test_rule_cap(self):
        for i in range(_MAX_RULES + 10):
            self.sgp.add_rule("social_intelligence", "interaction_recorded",
                              "empathy_mapping", 0.5, 0.3)
        self.assertLessEqual(len(self.sgp.get_rules()), _MAX_RULES)


class TestBuildGraph(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_empty_graph(self):
        g = self.sgp.build_graph()
        self.assertEqual(g["total_nodes"], 0)
        self.assertEqual(g["total_edges"], 0)

    def test_build_with_empathy(self):
        empathy = MagicMock()
        empathy.get_social_graph.return_value = {
            "nodes": [
                {"id": "p1", "name": "Alice", "empathy_score": 75.0, "interaction_count": 10, "avg_sentiment": 0.5},
                {"id": "p2", "name": "Bob", "empathy_score": 60.0, "interaction_count": 5, "avg_sentiment": 0.3},
            ],
            "edges": [
                {"source": "p1", "target": "p2", "affinity": 0.8},
            ],
            "total_persons": 2,
        }
        g = self.sgp.build_graph(empathy=empathy)
        self.assertGreater(g["total_nodes"], 0)
        self.assertGreater(g["total_edges"], 0)

    def test_build_with_social_intel(self):
        si = MagicMock()
        group = MagicMock()
        group.group_id = "g1"
        group.name = "Team"
        group.group_type = MagicMock(value="professional")
        group.cohesion_score = 0.7
        group.members = ["p1", "p2"]
        si.list_groups.return_value = [group]
        g = self.sgp.build_graph(social_intel=si)
        self.assertGreater(g["total_nodes"], 0)

    def test_build_with_conflict(self):
        cr = MagicMock()
        conflict_obj = MagicMock()
        conflict_obj.conflict_id = "c1"
        conflict_obj.title = "Dispute"
        conflict_obj.conflict_type = MagicMock(value="interest")
        conflict_obj.intensity = MagicMock(value=3)
        conflict_obj.status = MagicMock(value="active")
        p = MagicMock()
        p.party = "p1"
        conflict_obj.perspectives = [p]
        cr.list_conflicts.return_value = [conflict_obj]
        g = self.sgp.build_graph(conflict=cr)
        self.assertGreater(g["total_nodes"], 0)

    def test_build_all_modules(self):
        empathy = MagicMock()
        empathy.get_social_graph.return_value = {
            "nodes": [{"id": "p1", "name": "Alice", "empathy_score": 70.0,
                        "interaction_count": 8, "avg_sentiment": 0.4}],
            "edges": [],
            "total_persons": 1,
        }
        si = MagicMock()
        group = MagicMock()
        group.group_id = "g1"
        group.name = "Team"
        group.group_type = MagicMock(value="professional")
        group.cohesion_score = 0.6
        group.members = ["p1"]
        si.list_groups.return_value = [group]
        cr = MagicMock()
        cr.list_conflicts.return_value = []
        g = self.sgp.build_graph(social_intel=si, empathy=empathy, conflict=cr)
        self.assertGreater(g["total_nodes"], 0)

    def test_graph_nodes_capped(self):
        empathy = MagicMock()
        nodes = [{"id": "p%d" % i, "name": "P%d" % i, "empathy_score": 50.0,
                   "interaction_count": 1, "avg_sentiment": 0.0} for i in range(600)]
        empathy.get_social_graph.return_value = {"nodes": nodes, "edges": [], "total_persons": 600}
        g = self.sgp.build_graph(empathy=empathy)
        self.assertLessEqual(g["total_nodes"], _MAX_GRAPH_NODES)

    def test_build_handles_errors(self):
        empathy = MagicMock()
        empathy.get_social_graph.side_effect = ValueError("broken")
        g = self.sgp.build_graph(empathy=empathy)
        self.assertEqual(g["total_nodes"], 0)


class TestGetters(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_get_ripples_returns_copy(self):
        self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test")
        r1 = self.sgp.get_ripples()
        r2 = self.sgp.get_ripples()
        self.assertIsNot(r1[0], r2[0])

    def test_get_ripples_filter_by_source(self):
        self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test")
        self.sgp.create_ripple("empathy_mapping", "trigger_detected", 0.5, "test")
        si_ripples = self.sgp.get_ripples(source="social_intelligence")
        self.assertTrue(all(r.source == PropagationSource.SOCIAL_INTELLIGENCE for r in si_ripples))

    def test_get_effects_returns_copy(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test", person_id="p1")
        self.sgp.propagate(r, empathy=MagicMock())
        e1 = self.sgp.get_effects()
        e2 = self.sgp.get_effects()
        self.assertIsNot(e1[0], e2[0])

    def test_get_rules_returns_copy(self):
        r1 = self.sgp.get_rules()
        r2 = self.sgp.get_rules()
        self.assertIsNot(r1[0], r2[0])

    def test_get_ripples_limit(self):
        for _ in range(20):
            self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test")
        self.assertEqual(len(self.sgp.get_ripples(limit=5)), 5)

    def test_get_effects_limit(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test", person_id="p1")
        self.sgp.propagate(r, empathy=MagicMock())
        effects = self.sgp.get_effects(limit=1)
        self.assertLessEqual(len(effects), 1)


class TestPropagationSummary(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_empty_summary(self):
        s = self.sgp.get_propagation_summary()
        self.assertEqual(s["total_ripples"], 0)
        self.assertEqual(s["propagation_rate"], 0.0)

    def test_summary_after_ripples(self):
        self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test", person_id="p1")
        self.sgp.create_ripple("empathy_mapping", "trigger_detected", 0.7, "test", person_id="p1")
        s = self.sgp.get_propagation_summary()
        self.assertEqual(s["total_ripples"], 2)
        self.assertIn("social_intelligence", s["ripples_by_source"])


class TestDeepcopyReturns(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_ripple_to_dict_mutable_safe(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test",
                                    metadata={"k": "v1"})
        d = r.to_dict()
        d["metadata"]["k"] = "modified"
        self.assertEqual(r.metadata["k"], "v1")

    def test_node_to_dict_mutable_safe(self):
        n = SocialGraphNode("p1", GraphNodeType.PERSON, "Alice", 50.0, 0.5, 0, 0,
                            metadata={"k": [1]})
        d = n.to_dict()
        d["metadata"]["k"].append(2)
        self.assertEqual(len(n.metadata["k"]), 1)

    def test_effect_to_dict_safe(self):
        e = RippleEffect("e1", "r1", PropagationSource.EMPATHY_MAPPING, "action", 0.5, "notes")
        d = e.to_dict()
        self.assertIsInstance(d, dict)


class TestThreadSafety(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_concurrent_ripple_creation(self):
        errors = []

        def create_ripples():
            try:
                for i in range(100):
                    self.sgp.create_ripple("social_intelligence", "interaction_recorded",
                                          0.5, "concurrent %d" % i, person_id="p1")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=create_ripples) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)
        stats = self.sgp.get_stats()
        self.assertEqual(stats["ripples_created"], 500)

    def test_concurrent_propagate(self):
        errors = []

        def do_propagate():
            try:
                r = self.sgp.create_ripple("social_intelligence", "interaction_recorded",
                                          0.8, "test", person_id="p1")
                empathy = MagicMock()
                empathy.get_person_profile.return_value = MagicMock(person_id="p1")
                self.sgp.propagate(r, empathy=empathy)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=do_propagate) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)


class TestEdgeCases(unittest.TestCase):
    def setUp(self):
        self.sgp = SocialGraphPropagation()

    def test_person_id_sanitized_in_ripple(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.5, "test",
                                    person_id="p\t<script>id")
        self.assertNotIn("<", r.person_id)
        self.assertNotIn("\t", r.person_id)

    def test_group_id_sanitized(self):
        r = self.sgp.create_ripple("social_intelligence", "group_membership_changed", 0.5, "test",
                                    group_id="g\n<script>")
        self.assertNotIn("<", r.group_id)
        self.assertNotIn("\n", r.group_id)

    def test_conflict_id_sanitized(self):
        r = self.sgp.create_ripple("conflict_resolution", "conflict_escalated", 0.8, "test",
                                    conflict_id="c\tbad")
        self.assertNotIn("\t", r.conflict_id)

    def test_get_graph_returns_dict(self):
        g = self.sgp.get_graph()
        self.assertIn("nodes", g)
        self.assertIn("edges", g)

    def test_stats_returns_copy(self):
        s1 = self.sgp.get_stats()
        s2 = self.sgp.get_stats()
        self.assertIsNot(s1, s2)

    def test_propagation_with_person_not_found(self):
        r = self.sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test",
                                    person_id="missing")
        empathy = MagicMock()
        empathy.get_person_profile.side_effect = KeyError("not found")
        effects = self.sgp.propagate(r, empathy=empathy)
        self.assertGreater(len(effects), 0)

    def test_build_graph_with_empty_ids(self):
        empathy = MagicMock()
        empathy.get_social_graph.return_value = {
            "nodes": [{"id": "", "name": "Empty"}],
            "edges": [],
            "total_persons": 0,
        }
        g = self.sgp.build_graph(empathy=empathy)
        self.assertEqual(g["total_nodes"], 0)


class TestAPIEndpoints(unittest.TestCase):
    def test_ripple_request_model(self):
        from shadowd.cognition.api_router import SocialRippleRequest
        req = SocialRippleRequest(
            source="social_intelligence",
            ripple_type="interaction_recorded",
            strength=0.8,
            reason="test",
        )
        self.assertEqual(req.source, "social_intelligence")

    def test_ripple_request_strength_validation(self):
        from shadowd.cognition.api_router import SocialRippleRequest
        with self.assertRaises(Exception):
            SocialRippleRequest(source="x", ripple_type="y", strength=5.0, reason="t")

    def test_rule_request_model(self):
        from shadowd.cognition.api_router import PropagationRuleRequest
        req = PropagationRuleRequest(
            source="social_intelligence",
            ripple_type="interaction_recorded",
            target="empathy_mapping",
            weight=0.7,
            strength_threshold=0.3,
        )
        self.assertEqual(req.target, "empathy_mapping")

    def test_toggle_request_model(self):
        from shadowd.cognition.api_router import RuleToggleRequest
        req = RuleToggleRequest(enabled=False)
        self.assertFalse(req.enabled)


class TestEndToEndFlow(unittest.TestCase):
    def test_full_ripple_lifecycle(self):
        sgp = SocialGraphPropagation()

        empathy = MagicMock()
        empathy.get_person_profile.return_value = MagicMock(person_id="p1", name="Alice")

        r = sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "Alice called",
                              person_id="p1")
        self.assertFalse(r.propagated)

        effects = sgp.propagate(r, empathy=empathy)
        self.assertTrue(r.propagated)
        self.assertGreater(len(effects), 0)

        summary = sgp.get_propagation_summary()
        self.assertEqual(summary["total_ripples"], 1)
        self.assertEqual(summary["propagated_ripples"], 1)

        ripples = sgp.get_ripples()
        self.assertEqual(len(ripples), 1)

        effects_list = sgp.get_effects()
        self.assertEqual(len(effects_list), len(effects))

    def test_multiple_sources(self):
        sgp = SocialGraphPropagation()
        empathy = MagicMock()
        empathy.get_person_profile.return_value = MagicMock(person_id="p1")
        conflict = MagicMock()
        si = MagicMock()

        sgp.create_ripple("social_intelligence", "interaction_recorded", 0.8, "test", person_id="p1")
        r2 = sgp.create_ripple("empathy_mapping", "trigger_detected", 0.7, "sensitive", person_id="p1")
        sgp.create_ripple("conflict_resolution", "conflict_escalated", 0.9, "bad!", person_id="p1")

        for ripple in list(sgp._ripples):
            if not ripple.propagated:
                sgp.propagate(ripple, empathy=empathy, conflict=conflict, social_intel=si)

        summary = sgp.get_propagation_summary()
        self.assertEqual(summary["total_ripples"], 3)
        self.assertEqual(summary["propagated_ripples"], 3)


if __name__ == "__main__":
    unittest.main()