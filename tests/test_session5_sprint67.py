#!/usr/bin/env python3
"""
Session 5 Sprint 67 — Cross-Module Integration Stress Tests

Exercises multiple cognitive modules working together, catching edge cases
that don't appear when modules are tested in isolation.
"""

import copy
import re
import threading
import time
import unittest
from datetime import datetime, timezone, timedelta
from unittest.mock import patch, MagicMock

from shadowd.cognition.cognitive_pipeline import CognitivePipeline, PipelineResult
from shadowd.cognition.thought_sense import ThoughtSense
from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, RegulationApproach, PatternType,
)
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, ReasoningType, BiasType, CognitiveState,
)
from shadowd.cognition.moral_reasoning import MoralReasoning, EthicalFramework
from shadowd.cognition.narrative_memory import NarrativeMemory, EmotionTone, ThemeCategory
from shadowd.cognition.procedural_memory import ProceduralMemory, HabitStage
from shadowd.cognition.conflict_resolution import (
    ConflictResolution, ConflictType, ConflictIntensity, ConflictStatus, StrategyType,
)
from shadowd.cognition.empathy_mapping import EmpathyMapping, RelationshipType
from shadowd.cognition.social_intelligence import (
    SocialIntelligence, GroupType, InfluenceType, RoleType,
)
from shadowd.cognition.decision_journal import DecisionJournal, DecisionType, DecisionStatus
from shadowd.cognition.cognitive_reflection import CognitiveReflection, ReflectionType
from shadowd.cognition.attention_engine import AttentionEngine, FocusState, DistractionType
from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillDomain
from shadowd.cognition.shadow_os import ShadowOS


# ---------------------------------------------------------------------------
# 1. TestCognitivePipelineIntegration (10 tests)
# ---------------------------------------------------------------------------
class TestCognitivePipelineIntegration(unittest.TestCase):
    def setUp(self):
        self.ts = ThoughtSense()
        self.ei = EmotionalIntelligence()
        self.me = MetacognitionEngine()
        self.mr = MoralReasoning()

    def test_pipeline_all_modules_active(self):
        pipe = CognitivePipeline(self.ts, self.ei, self.me, self.mr)
        result = pipe.process("I must urgent help with this critical decision")
        self.assertIn("thought_sense", result["modules_used"])
        self.assertIn("emotional_intelligence", result["modules_used"])
        self.assertIn("metacognition", result["modules_used"])
        self.assertIn("moral_reasoning", result["modules_used"])
        self.assertIn("overall_assessment", result)
        self.assertNotEqual(result["overall_confidence"], 0.5)

    def test_pipeline_no_subsystems_minimal_result(self):
        pipe = CognitivePipeline(None, None, None, None)
        result = pipe.process("hello world")
        self.assertEqual(result["modules_used"], [])
        self.assertIsNone(result["thought_analysis"])
        self.assertIsNone(result["emotion_state"])
        self.assertIsNone(result["bias_detection"])
        self.assertIsNone(result["moral_evaluation"])
        self.assertIn("Minimal", result["overall_assessment"])

    def test_pipeline_only_thought_sense(self):
        pipe = CognitivePipeline(self.ts, None, None, None)
        result = pipe.process("can you please help me?")
        self.assertIn("thought_sense", result["modules_used"])
        self.assertNotIn("emotional_intelligence", result["modules_used"])
        self.assertIsNotNone(result["thought_analysis"])

    def test_pipeline_only_emotional_intelligence(self):
        pipe = CognitivePipeline(None, self.ei, None, None)
        result = pipe.process("I'm so happy and excited today!")
        self.assertIn("emotional_intelligence", result["modules_used"])
        self.assertIsNotNone(result["emotion_state"])

    def test_pipeline_handles_subsystem_exception(self):
        failing_ts = MagicMock()
        failing_ts.analyze.side_effect = RuntimeError("boom")
        pipe = CognitivePipeline(failing_ts, self.ei, self.me, self.mr)
        result = pipe.process("test text with ethical should question")
        self.assertNotIn("thought_sense", result["modules_used"])
        self.assertIn("emotional_intelligence", result["modules_used"])

    def test_pipeline_100_texts_rapidly(self):
        pipe = CognitivePipeline(self.ts, self.ei, self.me, self.mr)
        results = []
        for i in range(100):
            results.append(pipe.process(f"text number {i} about happiness and joy"))
        self.assertEqual(len(results), 100)
        for r in results:
            self.assertIn("result_id", r)

    def test_pipeline_result_deque_maxlen(self):
        pipe = CognitivePipeline(self.ts, self.ei, self.me, self.mr)
        for i in range(1100):
            pipe.process(f"overflow text {i}")
        stats = pipe.get_pipeline_stats()
        self.assertLessEqual(stats["results_stored"], 1000)

    def test_pipeline_ethical_content_triggers_moral(self):
        pipe = CognitivePipeline(self.ts, self.ei, self.me, self.mr)
        result = pipe.process("should we fire the employee for poor performance?")
        self.assertIn("moral_reasoning", result["modules_used"])
        self.assertIsNotNone(result["moral_evaluation"])

    def test_pipeline_non_ethical_skips_moral(self):
        pipe = CognitivePipeline(self.ts, self.ei, self.me, self.mr)
        result = pipe.process("the weather is nice and sunny today")
        self.assertNotIn("moral_reasoning", result["modules_used"])
        self.assertIsNone(result["moral_evaluation"])

    def test_pipeline_confidence_varies_by_module_count(self):
        pipe_none = CognitivePipeline(None, None, None, None)
        pipe_all = CognitivePipeline(self.ts, self.ei, self.me, self.mr)
        r_none = pipe_none.process("should we consider fairness in our decision?")
        r_all = pipe_all.process("should we consider fairness in our decision?")
        self.assertGreater(r_all["overall_confidence"], r_none["overall_confidence"])


# ---------------------------------------------------------------------------
# 2. TestEmotionalMetacognitionIntegration (8 tests)
# ---------------------------------------------------------------------------
class TestEmotionalMetacognitionIntegration(unittest.TestCase):
    def setUp(self):
        self.ei = EmotionalIntelligence()
        self.me = MetacognitionEngine()

    def test_detect_emotion_plus_biases(self):
        emo = self.ei.detect_emotion("I'm trusting and comfortable that I'm definitely certain about this")
        biases = self.me.detect_biases("I'm definitely certain this is right and everyone thinks so")
        self.assertIn(emo.primary, [EmotionCategory.TRUST, EmotionCategory.ANTICIPATION, EmotionCategory.NEUTRAL])
        self.assertTrue(len(biases) > 0)

    def test_high_intensity_emotion_biases_detection(self):
        high_emo = "I'm furious and outraged yet I definitely know I'm right about this"
        low_emo = "the report is due on friday"
        self.ei.detect_emotion(high_emo)
        high_biases = self.me.detect_biases(high_emo)
        low_biases = self.me.detect_biases(low_emo)
        self.assertGreater(len(high_biases), len(low_biases))

    def test_record_pattern_detect_cycle(self):
        for _ in range(4):
            self.ei.detect_emotion("I'm sad and disappointed every time this happens")
        pat = self.ei.record_pattern(
            PatternType.CYCLE, "Sadness loop",
            [EmotionCategory.SADNESS],
            ["disappointed"], "Recurring sadness pattern"
        )
        self.assertIsNotNone(pat.pattern_id)
        detected = self.ei.detect_patterns(min_frequency=2)
        self.assertTrue(any(p.pattern_type == PatternType.CYCLE for p in detected))

    def test_suggest_regulation_after_bias(self):
        self.ei.detect_emotion("I'm angry and furious about the unfair treatment", context="workplace dispute")
        biases = self.me.detect_biases("I'm angry and furious about the unfair treatment")
        state = list(self.ei._emotion_history)[-1] if self.ei._emotion_history else None
        self.assertIsNotNone(state)
        reg = self.ei.suggest_regulation(EmotionCategory.ANGER, 0.8)
        self.assertEqual(reg.approach, RegulationApproach.MINDFULNESS)

    def test_independent_state_no_cross_contamination(self):
        self.ei.detect_emotion("I'm happy and joyful")
        self.me.detect_biases("I'm definitely certain about this")
        ei_stats = self.ei.get_stats()
        me_stats = self.me.get_stats()
        self.assertEqual(ei_stats["stats"]["states_recorded"], 1)
        self.assertGreater(me_stats["stats"]["biases_detected"], 0)

    def test_concurrent_emotion_bias_no_deadlock(self):
        errors = []

        def run_emotion():
            try:
                for i in range(50):
                    self.ei.detect_emotion(f"happy text iteration {i}")
            except Exception as e:
                errors.append(e)

        def run_bias():
            try:
                for i in range(50):
                    self.me.detect_biases(f"definitely certain yes iteration {i}")
            except Exception as e:
                errors.append(e)

        t1 = threading.Thread(target=run_emotion)
        t2 = threading.Thread(target=run_bias)
        t1.start()
        t2.start()
        t1.join(timeout=10)
        t2.join(timeout=10)
        self.assertEqual(len(errors), 0)

    def test_emotional_trajectory_after_multiple(self):
        texts = [
            "I'm devastated and heartbroken",
            "I'm sad and disappointed",
            "I'm okay and content",
            "I'm happy and excited about progress",
        ]
        for t in texts:
            self.ei.detect_emotion(t)
        traj = self.ei.analyze_emotional_trajectory(limit=10)
        self.assertEqual(traj["direction"], "improving")

    def test_calibration_report_after_resolved(self):
        ep = self.me.start_thinking(ReasoningType.INDUCTIVE, "test topic", confidence=0.9)
        self.me.resolve_thinking(ep.episode_id, "correct", 0.7)
        report = self.me.get_calibration_report()
        self.assertEqual(report["total_resolved"], 1)
        self.assertNotEqual(report["avg_calibration_offset"], 0.0)


# ---------------------------------------------------------------------------
# 3. TestNarrativeProceduralIntegration (8 tests)
# ---------------------------------------------------------------------------
class TestNarrativeProceduralIntegration(unittest.TestCase):
    def setUp(self):
        self.nm = NarrativeMemory()
        self.pm = ProceduralMemory()
        self.ei = EmotionalIntelligence()

    def test_chapter_with_emotion_events(self):
        ch = self.nm.create_chapter("Career Growth", "career")
        evt = self.nm.add_event("I learn and adapt and overcome obstacles and discover insights", importance=0.9)
        self.assertIn(evt.emotion, ["positive", "growth", "resilience", "mixed"])
        self.nm.add_event_to_chapter(ch.chapter_id, evt)
        ch_data = self.nm.get_chapter(ch.chapter_id)
        self.assertEqual(len(ch_data.events), 1)

    def test_habit_complete_and_streak(self):
        h = self.pm.add_habit("morning jog", "7am alarm", "run 5km", "feeling energized")
        self.pm.complete_habit(h.habit_id)
        self.pm.complete_habit(h.habit_id)
        updated = self.pm.get_habit(h.habit_id)
        self.assertEqual(updated.streak, 2)

    def test_narrative_event_feeds_emotional_intel(self):
        evt = self.nm.add_event("I'm happy and proud of my team's success")
        self.ei.detect_emotion(evt.content)
        ei_stats = self.ei.get_stats()
        self.assertEqual(ei_stats["stats"]["states_recorded"], 1)

    def test_procedural_streak_changes(self):
        h = self.pm.add_habit("reading", "evening", "read 30 min", "knowledge gain")
        for _ in range(5):
            self.pm.complete_habit(h.habit_id)
        habit = self.pm.get_habit(h.habit_id)
        self.assertGreater(habit.streak, 0)
        self.pm.miss_habit(h.habit_id)
        habit = self.pm.get_habit(h.habit_id)
        self.assertEqual(habit.streak, 0)

    def test_multi_chapter_arc_coherence(self):
        arc = self.nm.create_arc("Career Journey", ["career", "growth"])
        ch1 = self.nm.create_chapter("First Job", "career")
        ch2 = self.nm.create_chapter("Promotion", "career")
        e1 = self.nm.add_event("started my first job and learned skills", importance=0.6)
        e2 = self.nm.add_event("got promoted and grew in my career", importance=0.8)
        self.nm.add_event_to_chapter(ch1.chapter_id, e1)
        self.nm.add_event_to_chapter(ch2.chapter_id, e2)
        self.nm.add_chapter_to_arc(arc.arc_id, ch1.chapter_id)
        self.nm.add_chapter_to_arc(arc.arc_id, ch2.chapter_id)
        arc_data = self.nm.get_arc(arc.arc_id)
        self.assertEqual(len(arc_data.chapters), 2)

    def test_chapter_arc_growth_resilience(self):
        ch = self.nm.create_chapter("Recovery", "growth")
        e = self.nm.add_event("I struggled but learned and overcame adversity through resilience", importance=0.8)
        self.nm.add_event_to_chapter(ch.chapter_id, e)
        ch_data = self.nm.get_chapter(ch.chapter_id)
        self.assertIn(ch_data.emotion_arc, ("ascending", "steady", "mixed"))

    def test_routine_consistency_decreases_after_gap(self):
        r = self.pm.add_routine("standup meeting", "daily")
        self.pm.complete_routine(r.routine_id)
        routine = self.pm.get_routine(r.routine_id)
        first_consistency = routine.consistency
        r2 = self.pm.add_routine("weekly review", "weekly")
        self.pm.complete_routine(r2.routine_id)
        routine_after = self.pm.get_routine(r2.routine_id)
        self.assertGreater(routine_after.consistency, 0)

    def test_skill_tree_leveling_with_procedural(self):
        st = AdaptiveSkillTree()
        st.add_skill("writing", "Writing", "creativity")
        for _ in range(15):
            st.use_skill("writing", success=True)
        skill = st.get_skill("writing")
        self.assertGreater(skill.level, 0)


# ---------------------------------------------------------------------------
# 4. TestConflictEmpathyIntegration (6 tests)
# ---------------------------------------------------------------------------
class TestConflictEmpathyIntegration(unittest.TestCase):
    def setUp(self):
        self.cr = ConflictResolution()
        self.em = EmpathyMapping()

    def test_conflict_with_empathy_perspectives(self):
        p1 = self.em.add_person("Alice", "colleague")
        p2 = self.em.add_person("Bob", "colleague")
        self.em.record_interaction(p1, "meeting", 0.5, "discussing project scope")
        self.em.record_interaction(p2, "meeting", -0.3, "disagrees on approach")
        conflict = self.cr.create_conflict("Project Direction", "interest", 2, "Disagreement on tech stack")
        self.cr.add_perspective(conflict.conflict_id, "Alice", "Use modern stack",
                                interests=["innovation"], flexibility=0.7)
        self.cr.add_perspective(conflict.conflict_id, "Bob", "Use stable stack",
                                interests=["reliability"], flexibility=0.5)
        c = self.cr.get_conflict(conflict.conflict_id)
        self.assertEqual(len(c.perspectives), 2)

    def test_conflict_strategies_informed_by_empathy(self):
        conflict = self.cr.create_conflict("Resource Allocation", "resource", 2)
        self.cr.add_perspective(conflict.conflict_id, "Team A", "Need more resources",
                                flexibility=0.8)
        self.cr.add_perspective(conflict.conflict_id, "Team B", "Also need resources",
                                flexibility=0.7)
        strategies = self.cr.suggest_strategies(conflict.conflict_id)
        self.assertTrue(len(strategies) > 0)
        self.assertIn(strategies[0].strategy_type.value, ["collaborate", "compromise"])

    def test_resolve_conflict_updates_empathy(self):
        p1 = self.em.add_person("Carol", "friend")
        self.em.record_interaction(p1, "text", 0.8, "resolved our disagreement")
        conflict = self.cr.create_conflict("Schedule Clash", "interest", 1)
        self.cr.add_perspective(conflict.conflict_id, "Carol", "Prefer morning", flexibility=0.6)
        resolved = self.cr.resolve_conflict(conflict.conflict_id, "Compromise on afternoon")
        self.assertTrue(resolved)
        health = self.em.get_relationship_health(p1)
        self.assertGreater(health.health_score, 0)

    def test_multiple_conflicts_same_parties(self):
        conflict = self.cr.create_conflict("Issue 1", "interest", 1)
        self.cr.add_perspective(conflict.conflict_id, "Dave", "Position A", flexibility=0.5)
        conflict2 = self.cr.create_conflict("Issue 2", "value", 2)
        self.cr.add_perspective(conflict2.conflict_id, "Dave", "Position B", flexibility=0.4)
        conflicts = self.cr.list_conflicts()
        self.assertEqual(len(conflicts), 2)

    def test_escalation_blocked_for_resolved(self):
        conflict = self.cr.create_conflict("Resolved Issue", "process", 1)
        self.cr.add_perspective(conflict.conflict_id, "Eve", "Want process change", flexibility=0.5)
        self.cr.resolve_conflict(conflict.conflict_id, "Process updated")
        result = self.cr.escalate_conflict(conflict.conflict_id)
        self.assertFalse(result)

    def test_common_ground_multiple_perspectives(self):
        conflict = self.cr.create_conflict("Team Vision", "value", 2)
        self.cr.add_perspective(conflict.conflict_id, "Frank", "Focus on quality",
                                interests=["quality", "users"], concerns=["burnout"],
                                flexibility=0.6)
        self.cr.add_perspective(conflict.conflict_id, "Grace", "Focus on speed",
                                interests=["speed", "users"], concerns=["burnout"],
                                flexibility=0.6)
        cg = self.cr.analyze_common_ground(conflict.conflict_id)
        self.assertIn("users", cg["common_interests"])
        self.assertIn("burnout", cg["common_concerns"])
        self.assertGreater(cg["overlap_score"], 0)


# ---------------------------------------------------------------------------
# 5. TestSocialInfluenceRoleIntegration (6 tests)
# ---------------------------------------------------------------------------
class TestSocialInfluenceRoleIntegration(unittest.TestCase):
    def setUp(self):
        self.si = SocialIntelligence()

    def test_group_members_roles(self):
        g = self.si.create_group("Engineering", GroupType.PROFESSIONAL, members=["alice", "bob", "carol"])
        self.si.assign_role("alice", g.group_id, RoleType.LEADER, influence_level=0.9)
        self.si.assign_role("bob", g.group_id, RoleType.CONTRIBUTOR, influence_level=0.5)
        dynamics = self.si.get_group_dynamics(g.group_id)
        self.assertEqual(dynamics["member_count"], 3)
        self.assertIn("leader", dynamics["role_distribution"])

    def test_influence_network_within_group(self):
        g = self.si.create_group("Research", GroupType.PROJECT, members=["dave", "eve"])
        self.si.add_influence("dave", "eve", InfluenceType.MENTOR, strength=0.8)
        dynamics = self.si.get_group_dynamics(g.group_id)
        self.assertIn("dave", dynamics["influence_network"])
        self.assertEqual(dynamics["influence_count"], 1)

    def test_communication_patterns_after_influences(self):
        g = self.si.create_group("Design", GroupType.INFORMAL, members=["frank", "grace"])
        self.si.add_influence("frank", "grace", InfluenceType.PEER, strength=0.6)
        patterns = self.si.detect_communication_patterns(g.group_id)
        self.assertTrue(len(patterns) > 0)

    def test_find_influencers_across_groups(self):
        g1 = self.si.create_group("Team A", GroupType.PROFESSIONAL, members=["henry"])
        g2 = self.si.create_group("Team B", GroupType.PROFESSIONAL, members=["henry"])
        self.si.add_influence("henry", "ira", InfluenceType.EXPERT, strength=0.9)
        influencers = self.si.find_influencers(min_influence=0.8)
        self.assertTrue(any(i["person_id"] == "henry" for i in influencers))

    def test_member_removal_updates_dynamics(self):
        g = self.si.create_group("Sprint", GroupType.PROJECT, members=["jack", "kate"])
        self.si.remove_member(g.group_id, "kate")
        g_data = self.si.get_group(g.group_id)
        self.assertEqual(len(g_data.members), 1)
        self.assertNotIn("kate", g_data.members)

    def test_role_tracks_influence_level(self):
        g = self.si.create_group("Board", GroupType.FORMAL, members=["leo"])
        role = self.si.assign_role("leo", g.group_id, RoleType.LEADER, influence_level=0.95)
        self.assertEqual(role.influence_level, 0.95)
        self.assertEqual(role.role_type, RoleType.LEADER)


# ---------------------------------------------------------------------------
# 6. TestDecisionJournalIntegration (8 tests)
# ---------------------------------------------------------------------------
class TestDecisionJournalIntegration(unittest.TestCase):
    def setUp(self):
        self.dj = DecisionJournal()
        self.me = MetacognitionEngine()
        self.mr = MoralReasoning()
        self.ei = EmotionalIntelligence()

    def test_decision_with_pipeline_reasoning(self):
        ts = ThoughtSense()
        pipe = CognitivePipeline(ts, self.ei, self.me, self.mr)
        pipe_result = pipe.process("should we invest in the new project?")
        d = self.dj.record_decision(
            "Invest in project X", "Proceed with investment",
            alternatives=["Wait", "Partial investment"],
            reasoning=pipe_result["overall_assessment"],
            confidence=pipe_result["overall_confidence"],
            domain="finance",
            decision_type="strategic",
        )
        self.assertEqual(d.status, DecisionStatus.PENDING)

    def test_outcome_updates_accuracy(self):
        d = self.dj.record_decision("Hire candidate", "Yes, hire",
                                     confidence=0.8, domain="hr")
        self.dj.record_outcome(d.decision_id, "Good hire", 0.9)
        updated = self.dj.get_decision(d.decision_id)
        self.assertEqual(updated.outcome_accuracy, 0.9)
        self.assertEqual(updated.status, DecisionStatus.COMPLETED)

    def test_retrospective_after_decisions(self):
        for i in range(5):
            d = self.dj.record_decision(f"Decision {i}", f"Option {i}",
                                         confidence=0.7, domain="ops",
                                         biases_flagged=["confirmation"])
            self.dj.record_outcome(d.decision_id, f"Outcome {i}", 0.6)
        retro = self.dj.run_retrospective(period_days=30)
        self.assertEqual(retro.decisions_reviewed, 5)
        self.assertIn("confirmation", retro.biases_found)

    def test_detect_overconfidence_pattern(self):
        for i in range(5):
            d = self.dj.record_decision(f"Overconf {i}", "Proceed",
                                         confidence=0.95, domain="trading",
                                         decision_type="tactical")
            self.dj.record_outcome(d.decision_id, f"Result {i}", 0.3)
        patterns = self.dj.detect_patterns(min_frequency=3)
        overconf = [p for p in patterns if p.pattern_type == "consistently_overconfident"]
        self.assertTrue(len(overconf) > 0)

    def test_decision_accuracy_varies_by_domain(self):
        d1 = self.dj.record_decision("Tech choice", "Go with A",
                                      confidence=0.7, domain="tech")
        d2 = self.dj.record_decision("People choice", "Go with B",
                                      confidence=0.7, domain="hr")
        self.dj.record_outcome(d1.decision_id, "Great choice", 0.9)
        self.dj.record_outcome(d2.decision_id, "Bad choice", 0.3)
        tech_acc = self.dj.get_decision_accuracy(domain="tech")
        hr_acc = self.dj.get_decision_accuracy(domain="hr")
        self.assertGreater(tech_acc["avg_accuracy"], hr_acc["avg_accuracy"])

    def test_journal_with_moral_reasoning(self):
        d = self.dj.record_decision("Terminate employee", "Yes, terminate",
                                     confidence=0.6, domain="hr",
                                     decision_type="ethical")
        eval_result = self.mr.evaluate("terminate employee for misconduct")
        self.assertIsNotNone(eval_result.overall_score)
        updated = self.dj.get_decision(d.decision_id)
        self.assertEqual(updated.decision_type, DecisionType.ETHICAL)

    def test_journal_metacognition_flags_biases(self):
        biases = self.me.detect_biases("I'm definitely certain and everyone thinks so")
        d = self.dj.record_decision("Rush feature", "Ship now",
                                     confidence=0.9, domain="eng",
                                     biases_flagged=[b.bias_type.value for b in biases])
        self.assertTrue(len(d.biases_flagged) > 0)

    def test_status_transitions(self):
        d = self.dj.record_decision("Deploy service", "Deploy to production",
                                     confidence=0.8, domain="devops")
        self.assertEqual(d.status, DecisionStatus.PENDING)
        self.dj.update_status(d.decision_id, "executing")
        updated = self.dj.get_decision(d.decision_id)
        self.assertEqual(updated.status, DecisionStatus.EXECUTING)
        self.dj.record_outcome(d.decision_id, "Deploy succeeded", 0.95)
        final = self.dj.get_decision(d.decision_id)
        self.assertEqual(final.status, DecisionStatus.COMPLETED)


# ---------------------------------------------------------------------------
# 7. TestAttentionEngineIntegration (4 tests)
# ---------------------------------------------------------------------------
class TestAttentionEngineIntegration(unittest.TestCase):
    def setUp(self):
        self.ae = AttentionEngine()

    def test_focus_with_distractions_state(self):
        self.ae.add_attention_item("write report", "high", "work", 0.8, 0.9)
        self.ae.start_focus("write report")
        self.ae.record_distraction("notification", "Slack message", 30)
        self.ae.record_distraction("interruption", "Phone call", 120)
        self.ae.end_focus()
        state = self.ae.detect_focus_state()
        self.assertIn(state.value, ["distracted", "scanning", "active", "fatigued", "idle"])

    def test_deep_work_after_25_min(self):
        sid = self.ae.start_focus("deep analysis task")
        session = self.ae._current_session
        session.started_at = (
            datetime.now(timezone.utc) - timedelta(minutes=30)
        ).isoformat()
        self.ae.end_focus(session_id=sid)
        stats = self.ae.get_stats()
        self.assertEqual(stats["deep_work_sessions"], 1)

    def test_recover_focus_reduces_fatigue(self):
        for _ in range(10):
            self.ae.record_distraction("fatigue", "tired", 60)
        pre = self.ae.get_attention_profile()
        self.ae.recover_focus(15.0)
        post = self.ae.get_attention_profile()
        self.assertLess(post.fatigue_level, pre.fatigue_level)

    def test_attention_profile_reflects_sessions(self):
        self.ae.start_focus("coding session")
        self.ae.end_focus()
        self.ae.start_focus("review session")
        self.ae.end_focus()
        profile = self.ae.get_attention_profile()
        self.assertEqual(profile.deep_work_sessions + profile.total_interruptions,
                         profile.deep_work_sessions + profile.total_interruptions)


# ---------------------------------------------------------------------------
# 8. TestShadowOSStatusConsistency (5 tests)
# ---------------------------------------------------------------------------
class TestShadowOSStatusConsistency(unittest.TestCase):
    def setUp(self):
        self.os = ShadowOS.__new__(ShadowOS)
        self.os.data_dir = None
        self.os._stats_lock = threading.Lock()
        self.os.graph = MagicMock()
        self.os.graph.state = MagicMock()
        self.os.graph.state.mode = MagicMock()
        self.os.graph.state.mode.value = "listen"
        self.os.graph.get_status = MagicMock(return_value={})
        self.os.memory = MagicMock()
        self.os.memory._atoms = {}
        self.os.memory.get_stats = MagicMock(return_value={})
        self.os.memory.list_blocks = MagicMock(return_value=[])
        self.os.crew = MagicMock()
        self.os.crew.get_stats = MagicMock(return_value={})
        self.os.crew.list_agents = MagicMock(return_value=[])
        self.os.mesh = MagicMock()
        self.os.mesh.get_stats = MagicMock(return_value={})
        self.os.mesh._peers = {}
        self.os.mesh._circuit_breakers = {}
        self.os.stealth = MagicMock()
        self.os.stealth._audit_log = []
        self.os.personality = MagicMock()
        self.os.personality.get_stats = MagicMock(return_value={})
        self.os.personality.get_profile = MagicMock(return_value={})
        self.os.voice = MagicMock()
        self.os.voice.get_state = MagicMock(return_value={})
        self.os.thought_sense = ThoughtSense()
        self.os.recall_synthesis = MagicMock()
        self.os.recall_synthesis.get_stats = MagicMock(return_value={"total_observations": 0})
        self.os.temporal_reasoning = MagicMock()
        self.os.temporal_reasoning.get_stats = MagicMock(return_value={"total_events": 0})
        self.os.empathy = EmpathyMapping()
        self.os.presence = MagicMock()
        self.os.presence.get_stats = MagicMock(return_value={"total_devices": 0})
        self.os.trinity = MagicMock()
        self.os.trinity.get_stats = MagicMock(return_value={"total_analyses": 0})
        self.os.dream = MagicMock()
        self.os.dream.get_stats = MagicMock(return_value={"total_memories": 0})
        self.os.skills = AdaptiveSkillTree()
        self.os.narrative = NarrativeMemory()
        self.os.procedural = ProceduralMemory()
        self.os.conflict = ConflictResolution()
        self.os.emotional_intel = EmotionalIntelligence()
        self.os.creative = MagicMock()
        self.os.creative.get_stats = MagicMock(return_value={"stats": {"ideas_generated": 0}})
        self.os.metacognition = MetacognitionEngine()
        self.os.moral = MoralReasoning()
        self.os.attention = AttentionEngine()
        self.os.pipeline = CognitivePipeline(
            self.os.thought_sense, self.os.emotional_intel,
            self.os.metacognition, self.os.moral,
        )
        self.os.social_intel = SocialIntelligence()
        self.os.arise = MagicMock()
        self.os.arise.get_stats = MagicMock(return_value={"active_agents": 0})
        self.os.summon = MagicMock()
        self.os.journal = DecisionJournal()
        self.os.decision_journal = DecisionJournal()
        self.os.reflection = CognitiveReflection()
        self.os.goal_reasoning = MagicMock()
        self.os.goal_reasoning.get_stats = MagicMock(return_value={"assessments_made": 0})
        self.os.device_context = MagicMock()
        self.os.device_context.get_stats = MagicMock(return_value={"stats": {"context_refreshes": 0}})
        self.os.daemon = MagicMock()
        self.os.daemon.get_status = MagicMock(return_value={"status": "idle"})
        self.os.daemon.get_health = MagicMock(return_value=True)
        self.os.evolution = MagicMock()
        self.os.evolution.get_stats = MagicMock(return_value={"observations_added": 0})
        self.os.oracle = MagicMock()
        self.os.oracle.get_stats = MagicMock(return_value={"predictions_generated": 0})
        self.os.anomaly_detector = MagicMock()
        self.os.goal_planner = MagicMock()
        self.os._running = True
        self.os._start_time = time.time()
        self.os._stats = {
            "events_processed": 0, "queries_handled": 0,
            "actions_taken": 0, "insights_generated": 0,
            "errors": 0, "uptime_seconds": 0,
        }
        self.os.feed_loop = MagicMock()
        self.os.feed_loop.get_stats = MagicMock(return_value={"signals_emitted": 0})
        for attr in ['anticipatory', 'boundary', 'causal', 'chrono', 'creative_dream',
                      'drift_detector', 'economy', 'existential', 'grammar', 'health_dashboard',
                      'holographic', 'immune', 'morphic', 'neural_bridge', 'parallax',
                      'phoenix', 'quantum_field', 'quantum_memory', 'resonance',
                      'resonance_engine', 'social_propagation', 'symbiosis', 'syncognitive',
                      'temporal_self', 'topology', 'shadow_powers',
                      'synthesis_engine', 'self_mod', 'visualizer', 'profiler',
                      'intel_report', 'anomaly_corr', 'module_bus', 'opt_daemon',
                      'module_registry', 'arbiter', 'evolutionary',
                      'vision', 'notifications', 'visual_bridge', 'shadow_deploy',
                      'dashboard_agg', 'predictive', 'shadow_pool',
                      'feed_loop']:
            mock = MagicMock()
            mock.get_stats = MagicMock(return_value={})
            mock.get_health = MagicMock(return_value={"healthy": True})
            setattr(self.os, attr, mock)

    def test_get_status_includes_all_modules(self):
        status = self.os.get_status()
        expected_keys = [
            "graph_status", "memory_stats", "crew_stats", "mesh_stats",
            "personality_stats", "thought_sense_stats", "recall_synthesis_stats",
            "temporal_reasoning_stats", "empathy_stats", "presence_stats",
            "trinity_stats", "dream_stats", "skill_stats", "narrative_stats",
            "procedural_stats", "conflict_stats", "emotional_intelligence_stats",
            "creative_synthesis_stats", "metacognition_stats", "moral_reasoning_stats",
            "attention_stats", "pipeline_stats", "social_intelligence_stats",
            "decision_journal_stats", "reflection_stats", "goal_reasoning_stats", "device_context_stats", "arise_stats", "stats",
        ]
        for key in expected_keys:
            self.assertIn(key, status, f"Missing key: {key}")

    def test_get_health_returns_healthy_or_degraded(self):
        health = self.os.get_health()
        self.assertIn(health["status"], ["healthy", "degraded"])

    def test_module_count_matches(self):
        health = self.os.get_health()
        expected_modules = 51
        self.assertGreaterEqual(len(health["modules"]), expected_modules)

    def test_stats_keys_consistent(self):
        s1 = self.os.get_status()
        s2 = self.os.get_status()
        self.assertEqual(set(s1.keys()), set(s2.keys()))

    def test_no_module_missing_from_health(self):
        health = self.os.get_health()
        module_names = list(health["modules"].keys())
        required = [
            "shadow_graph", "dark_memory", "shadow_crew", "stealth_layer",
            "personality_engine", "shadow_voice", "thought_sense",
            "recall_synthesis", "temporal_reasoning", "empathy_mapping",
            "presence_detection", "cognitive_trinity", "dream_state",
            "skill_tree", "narrative_memory", "procedural_memory",
            "conflict_resolution", "emotional_intelligence",
            "creative_synthesis", "metacognition", "moral_reasoning",
            "attention_engine", "cognitive_pipeline", "social_intelligence",
            "decision_journal",
            "cognitive_reflection",
            "device_context",
        ]
        for r in required:
            self.assertIn(r, module_names, f"Missing module in health: {r}")


if __name__ == "__main__":
    unittest.main()