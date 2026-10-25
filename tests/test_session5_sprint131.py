#!/usr/bin/env python3
"""Sprint 131 regression tests — second-pass deep audit round 4.

Verifies fixes in:
- cognitive_reflection.py: round() for confidence_before/after, sentiment in to_dict()
- decision_journal.py: round() for outcome_accuracy in to_dict()
- goal_reasoning.py: context dict values sanitized with _sanitize_content, dead severity_map[0] removed
- social_intelligence.py: .strip() before [:max_length], description/context use _sanitize_content
- shadow_arise.py: .strip() before [:max_length], word boundary regex in recommend_template()
- shadow_summon.py: .strip() before [:max_length], word boundary regex in classify_intent()
"""

import pytest

from shadowd.cognition.cognitive_reflection import CognitiveReflection
from shadowd.cognition.decision_journal import DecisionJournal
from shadowd.cognition.goal_reasoning import GoalReasoning
from shadowd.cognition.social_intelligence import SocialIntelligence, GroupType, InfluenceType
from shadowd.cognition.shadow_arise import ShadowArise
from shadowd.cognition.shadow_summon import ShadowSummon, SummonIntent


class TestCognitiveReflectionRound4:
    def test_confidence_before_rounded(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            topic="test",
            confidence_before=0.7777777,
            confidence_after=0.8888888,
        )
        d = cr.get_reflection(entry.entry_id).to_dict()
        assert d["confidence_before"] == round(0.7777777, 3)
        assert d["confidence_after"] == round(0.8888888, 3)

    def test_sentiment_rounded(self):
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            topic="test",
            confidence_before=0.5,
            confidence_after=0.6,
            sentiment=0.3333333,
        )
        d = cr.get_reflection(entry.entry_id).to_dict()
        assert d["sentiment"] == round(0.3333333, 3)


class TestDecisionJournalRound4:
    def test_outcome_accuracy_rounded(self):
        dj = DecisionJournal()
        rec = dj.record_decision("test", "chosen_option", confidence=0.7)
        dj.update_status(rec.decision_id, "executing")
        dj.record_outcome(rec.decision_id, "good result", 0.7777777)
        d = dj.get_decision(rec.decision_id).to_dict()
        assert d["outcome_accuracy"] == round(0.7777777, 4)


class TestGoalReasoningRound4:
    def test_context_values_sanitized(self):
        gr = GoalReasoning()
        assessment = gr.assess_goal(
            "test_goal", progress=0.5, context={"note": "latency < 100ms\x00bad"},
        )
        assert "\x00" not in str(assessment.metadata)


class TestSocialIntelligenceRound4:
    def test_description_preserves_angles(self):
        si = SocialIntelligence()
        group = si.create_group("Test", GroupType.PROJECT, description="latency < 100ms")
        fetched = si.get_group(group.group_id)
        assert "<" in fetched.description

    def test_context_preserves_angles(self):
        si = SocialIntelligence()
        rel = si.add_influence("alice", "bob", InfluenceType.EXPERT, context="usage < 5ms")
        inf = si._influences[rel.relation_id]
        assert "<" in inf.context

    def test_sanitize_strips_before_truncation(self):
        si = SocialIntelligence()
        group = si.create_group("  test  ", GroupType.PROJECT)
        fetched = si.get_group(group.group_id)
        assert fetched.name == "test"


class TestShadowAriseRound4:
    def test_sanitize_strips_before_truncation(self):
        sa = ShadowArise()
        agent = sa.arise("  test agent  ", objective="  do stuff  ")
        fetched = sa.get_agent(agent.agent_id)
        assert fetched.name == "test agent"

    def test_recommend_template_no_false_match(self):
        sa = ShadowArise()
        template = sa.recommend_template("planet exploration and plant growth")
        assert template is None or "plant" not in (template.name.lower() if template else "")


class TestShadowSummonRound4:
    def test_sanitize_strips_before_truncation(self):
        ss = ShadowSummon()
        result = ss.classify_intent("  explain this  ")
        assert result == SummonIntent.ASK_QUESTION

    def test_explain_matches_question(self):
        ss = ShadowSummon()
        assert ss.classify_intent("explain quantum computing") == SummonIntent.ASK_QUESTION

    def test_why_matches_question(self):
        ss = ShadowSummon()
        assert ss.classify_intent("why is the sky blue") == SummonIntent.ASK_QUESTION

    def test_describe_matches_question(self):
        ss = ShadowSummon()
        assert ss.classify_intent("describe the system") == SummonIntent.ASK_QUESTION

    def test_jigsaw_no_false_match_saw(self):
        ss = ShadowSummon()
        result = ss.classify_intent("i solved the jigsaw puzzle")
        assert result != SummonIntent.OBSERVE_EVENT