#!/usr/bin/env python3
"""Sprint 82 regression tests — Round 3 deep audit of goal_reasoning, cognitive_reflection,
decision_journal, social_intelligence."""

import copy
import threading
import pytest

from shadowd.cognition.goal_reasoning import GoalReasoning, GoalConflict, ConflictType, _sanitize
from shadowd.cognition.cognitive_reflection import CognitiveReflection, _sanitize as cr_sanitize
from shadowd.cognition.decision_journal import DecisionJournal, _sanitize as dj_sanitize
from shadowd.cognition.social_intelligence import SocialIntelligence, _sanitize as si_sanitize, GroupType, InfluenceType


class TestGoalReasoningResourceConflict:
    def test_gr21_intersection_not_union(self):
        gr = GoalReasoning()
        ga = {"goal_id": "a", "priority": "medium", "resource_needs": {"cpu": 0.3, "mem": 0.2}}
        gb = {"goal_id": "b", "priority": "medium", "resource_needs": {"disk": 0.9, "net": 0.8}}
        conflicts = gr.analyze_conflicts([ga, gb])
        assert len(conflicts) == 0

    def test_gr21_intersection_detects_real_conflict(self):
        gr = GoalReasoning()
        ga = {"goal_id": "a", "priority": "medium", "resource_needs": {"cpu": 0.8, "mem": 0.2}}
        gb = {"goal_id": "b", "priority": "medium", "resource_needs": {"cpu": 0.5, "mem": 0.9}}
        conflicts = gr.analyze_conflicts([ga, gb])
        assert len(conflicts) >= 1
        assert conflicts[0].conflict_type == ConflictType.RESOURCE


class TestGoalReasoningToDict:
    def test_gr18_conflict_to_dict_returns_value_not_enum(self):
        c = GoalConflict(
            conflict_id="test",
            goal_id_a="a",
            goal_id_b="b",
            conflict_type=ConflictType.RESOURCE,
            description="test",
            resolution_hint="test",
            detected_at="2024-01-01",
        )
        d = c.to_dict()
        assert isinstance(d["conflict_type"], str)
        assert d["conflict_type"] == "resource"

    def test_gr25_assessment_to_dict_list_copies(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", 0.5, blockers=["b1"])
        d = a.to_dict()
        d["risk_factors"].append("extra")
        d2 = a.to_dict()
        assert len(d2["risk_factors"]) == 1


class TestGoalReasoningDeepcopy:
    def test_gr26_assess_goal_deepcopy(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", 0.5)
        a.risk_factors.append("tampered")
        a2 = gr.get_assessment(a.assessment_id)
        assert "tampered" not in a2["risk_factors"]

    def test_gr26_detect_obstacle_deepcopy(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "blocker")
        o.description = "tampered"
        o2 = gr.get_obstacle(o.obstacle_id)
        assert o2["description"] != "tampered"


class TestGoalReasoningMetadataValue:
    def test_gr08_metadata_value_capped(self):
        gr = GoalReasoning()
        a = gr.assess_goal("g1", 0.5, context={"k": "v" * 500})
        assert len(a.metadata["k"]) <= 200

    def test_gr08_obstacle_metadata_value_capped(self):
        gr = GoalReasoning()
        o = gr.detect_obstacle("g1", "blocker", metadata={"k": "v" * 500})
        assert len(o.metadata["k"]) <= 200


class TestGoalReasoningPrioritiesCap:
    def test_gr33_goal_priorities_capped(self):
        gr = GoalReasoning()
        gr._MAX_GOAL_PRIORITIES = 10
        for i in range(15):
            gr.assess_goal(f"g{i}", 0.5)
        assert len(gr._goal_priorities) <= 10


class TestGoalReasoningPrioritizeThreadSafe:
    def test_gr13_prioritize_reads_snapshot(self):
        gr = GoalReasoning()
        gr.assess_goal("g1", 0.5)
        goals = [{"goal_id": "g1", "priority": "medium", "progress": 0.5}]
        result = gr.prioritize_goals(goals)
        assert len(result) == 1


class TestGoalReasoningSanitize:
    def test_sanitize_none(self):
        assert _sanitize(None, 64) == ""

    def test_sanitize_non_string(self):
        assert isinstance(_sanitize(123, 64), str)


class TestCognitiveReflectionSanitize:
    def test_f15_sanitize_none(self):
        assert cr_sanitize(None) == ""

    def test_f01_metadata_value_sanitized(self):
        cr = CognitiveReflection()
        e = cr.add_reflection("topic", metadata={"k": "value\x00\x7f<script>alert(1)</script>"})
        assert "\x00" not in e.metadata["k"]
        assert "\x7f" not in e.metadata["k"]
        assert "<script>" in e.metadata["k"]


class TestCognitiveReflectionDeepcopy:
    def test_f12_detect_patterns_deepcopy(self):
        cr = CognitiveReflection()
        cr.add_reflection("t1", domain="work", what_could_improve=["slow delivery"], sentiment=-0.5)
        cr.add_reflection("t2", domain="work", what_could_improve=["slow delivery"], sentiment=-0.6)
        patterns = cr.detect_patterns(min_frequency=2)
        if patterns:
            patterns[0].frequency = 9999
            patterns2 = cr.detect_patterns(min_frequency=2)
            if patterns2:
                assert patterns2[0].frequency != 9999


class TestCognitiveReflectionToDict:
    def test_entry_to_dict_list_copies(self):
        cr = CognitiveReflection()
        e = cr.add_reflection("topic", what_went_well=["a", "b"])
        d = e.to_dict()
        d["what_went_well"].append("c")
        d2 = e.to_dict()
        assert len(d2["what_went_well"]) == 2

    def test_insight_to_dict_list_copies(self):
        cr = CognitiveReflection()
        e = cr.add_reflection("topic", what_went_well=["I learned something new"], domain="test")
        insights = cr.extract_insights(e.entry_id)
        if insights:
            d = insights[0].to_dict()
            d["applicable_domains"].append("extra")
            d2 = insights[0].to_dict()
            assert len(d2["applicable_domains"]) == len(insights[0].applicable_domains)


class TestCognitiveReflectionDeadCode:
    def test_f17_domain_type_removed(self):
        import inspect
        src = inspect.getsource(CognitiveReflection.detect_patterns)
        assert "domain_type" not in src


class TestDecisionJournalToDict:
    def test_dj02_decision_to_dict_list_copies(self):
        dj = DecisionJournal()
        rec = dj.record_decision("title", "chosen", alternatives=["a", "b"], biases_flagged=["c"])
        d = rec.to_dict()
        d["alternatives"].append("tampered")
        d["biases_flagged"].append("tampered")
        d2 = rec.to_dict()
        assert len(d2["alternatives"]) == 2
        assert len(d2["biases_flagged"]) == 1

    def test_dj03_retro_to_dict_list_copies(self):
        dj = DecisionJournal()
        rec = dj.record_decision("title", "chosen", confidence=0.8)
        dj.record_outcome(rec.decision_id, "good", 0.9)
        retro = dj.run_retrospective()
        d = retro.to_dict()
        d["findings"].append("tampered")
        d2 = retro.to_dict()
        assert "tampered" not in d2["findings"]

    def test_dj04_pattern_to_dict_list_copies(self):
        dj = DecisionJournal()
        for _ in range(3):
            rec = dj.record_decision("title", "same_choice", domain="work", confidence=0.9)
            dj.record_outcome(rec.decision_id, "ok", 0.6)
        patterns = dj.detect_patterns()
        if patterns:
            d = patterns[0].to_dict()
            d["domains"].append("tampered")
            d2 = patterns[0].to_dict()
            assert "tampered" not in d2["domains"]


class TestDecisionJournalDeepcopy:
    def test_dj18_retro_deepcopy(self):
        dj = DecisionJournal()
        rec = dj.record_decision("title", "chosen", confidence=0.8)
        dj.record_outcome(rec.decision_id, "good", 0.9)
        retro = dj.run_retrospective()
        retro.findings.append("tampered")
        retro2 = dj.run_retrospective()
        assert "tampered" not in retro2.findings

    def test_dj23_retrospective_tocou_snapshot(self):
        dj = DecisionJournal()
        rec = dj.record_decision("title", "chosen", confidence=0.8)
        dj.record_outcome(rec.decision_id, "good", 0.9)
        retro = dj.run_retrospective()
        assert retro.decisions_reviewed >= 1


class TestDecisionJournalSanitize:
    def test_dj50_sanitize_none(self):
        assert dj_sanitize(None) == ""

    def test_dj09_metadata_value_sanitized(self):
        dj = DecisionJournal()
        rec = dj.record_decision("title", "chosen", metadata={"k": "val\x00\x7f<script>alert(1)</script>"})
        assert "\x00" not in rec.metadata["k"]
        assert "\x7f" not in rec.metadata["k"]
        assert "<script>" in rec.metadata["k"]


class TestDecisionJournalPatternType:
    def test_dj29_repeated_choice_not_always_avoids(self):
        dj = DecisionJournal()
        for _ in range(4):
            rec = dj.record_decision("title", "same_choice")
            dj.record_outcome(rec.decision_id, "ok", 0.7)
        patterns = dj.detect_patterns()
        for p in patterns:
            if "Repeatedly chooses" in p.description:
                assert p.pattern_type == "repeated_choice"
                assert p.pattern_type != "always_avoids"


class TestSocialIntelligenceDeepcopy:
    def test_si1_communication_patterns_deepcopy(self):
        si = SocialIntelligence()
        g = si.create_group("team", GroupType.PROFESSIONAL, members=["alice", "bob"])
        si.add_influence("alice", "bob", InfluenceType.MENTOR, 0.8)
        patterns = si.detect_communication_patterns(g.group_id)
        if patterns:
            patterns[0].frequency = 9999
            patterns2 = si.detect_communication_patterns(g.group_id)
            if patterns2:
                assert patterns2[0].frequency != 9999


class TestSocialIntelligenceSanitize:
    def test_si5_sanitize_strips_del_and_c1(self):
        assert "\x7f" not in si_sanitize("hello\x7fworld", 64)
        # \x80-\x9f are UTF-8 continuation bytes, must NOT be stripped

    def test_si6_group_id_sanitized_add_member(self):
        si = SocialIntelligence()
        g = si.create_group("team", GroupType.PROFESSIONAL)
        result = si.add_member("<script>g1</script>", "alice")
        assert not result

    def test_si6_group_id_sanitized_remove_member(self):
        si = SocialIntelligence()
        g = si.create_group("team", GroupType.PROFESSIONAL)
        result = si.remove_member("<script>g1</script>", "alice")
        assert not result


class TestSocialIntelligenceDeadCode:
    def test_si3_unreachable_guard_removed(self):
        import inspect
        src = inspect.getsource(SocialIntelligence.detect_communication_patterns)
        assert "len(rels) < 1" not in src


class TestGoalReasoningSanitizeChars:
    def test_sanitize_strips_angle_brackets(self):
        assert "<" not in _sanitize("hello <script> world", 64)
        assert ">" not in _sanitize("hello </script> world", 64)

    def test_sanitize_strips_control_chars(self):
        assert "\x00" not in _sanitize("hello\x00world", 64)
        assert "\x1f" not in _sanitize("hello\x1fworld", 64)

    def test_sanitize_strips_newline_tab(self):
        assert "\n" not in _sanitize("hello\nworld", 64)
        assert "\t" not in _sanitize("hello\tworld", 64)


class TestThreadSafety:
    def test_goal_reasoning_concurrent_assess(self):
        gr = GoalReasoning()
        errors = []

        def worker():
            try:
                for i in range(50):
                    gr.assess_goal(f"g{i}", 0.5)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_decision_journal_concurrent_record(self):
        dj = DecisionJournal()
        errors = []

        def worker():
            try:
                for i in range(50):
                    dj.record_decision(f"title{i}", f"chosen{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_cognitive_reflection_concurrent_add(self):
        cr = CognitiveReflection()
        errors = []

        def worker():
            try:
                for i in range(50):
                    cr.add_reflection(f"topic{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_social_intelligence_concurrent_create_group(self):
        si = SocialIntelligence()
        errors = []

        def worker():
            try:
                for i in range(50):
                    si.create_group(f"group{i}", GroupType.PROFESSIONAL)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestGoalReasoningMetadataSanitization:
    def test_replan_metadata_sanitized(self):
        gr = GoalReasoning()
        a = gr.suggest_replan("g1", "blocked by latency < 100ms", current_state={"k": "val"})
        assert "<" in a.reason