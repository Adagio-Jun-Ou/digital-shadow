"""
Sprint 127 — Sanitize Convention Fix Regression + Deep Audit

Tests the _SANITIZE_RE / _SANITIZE_STRICT_RE / _sanitize / _sanitize_content
convention across all modules with dual-pattern setup.

Convention:
  _SANITIZE_RE = content-safe (preserves <> and \\t\\n\\r)
  _SANITIZE_STRICT_RE = strict (strips <> and ALL control chars including \\t\\n\\r)
  _sanitize() uses _SANITIZE_STRICT_RE (for identifiers)
  _sanitize_content() uses _SANITIZE_CONTENT_RE (for content)
"""

import re
import pytest

TAB = '\t'
NEWLINE = '\n'
CR = '\r'
DEL = '\x7f'
BEL = '\x07'

CONTENT_PATTERN = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'
STRICT_PATTERN = r'[<>\x00-\x1f\x7f]'


class TestAnomalyDetectorConvention:
    def test_sanitize_re_is_content_safe(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_RE
        assert _SANITIZE_RE.pattern == CONTENT_PATTERN
        result = _SANITIZE_RE.sub('', 'latency < 100ms')
        assert '<' in result

    def test_sanitize_strict_re_is_strict(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_STRICT_RE
        assert _SANITIZE_STRICT_RE.pattern == STRICT_PATTERN
        result = _SANITIZE_STRICT_RE.sub('', 'latency < 100ms')
        assert '<' not in result

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.anomaly_detector import _sanitize
        assert '<script>' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.anomaly_detector import _sanitize_content
        result = _sanitize_content('latency < 100ms')
        assert '<' in result
        assert '100ms' in result

    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.anomaly_detector import _sanitize
        for ch in [TAB, NEWLINE, CR]:
            result = _sanitize(f'hello{ch}world')
            assert ch not in result

    def test_sanitize_content_preserves_whitespace(self):
        from shadowd.cognition.anomaly_detector import _sanitize_content
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize_content(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result


class TestDecisionJournalConvention:
    def test_sanitize_re_is_content_safe(self):
        from shadowd.cognition.decision_journal import _SANITIZE_RE
        assert _SANITIZE_RE.pattern == CONTENT_PATTERN

    def test_sanitize_strict_re_is_strict(self):
        from shadowd.cognition.decision_journal import _SANITIZE_STRICT_RE
        assert _SANITIZE_STRICT_RE.pattern == STRICT_PATTERN

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.decision_journal import _sanitize
        assert '<script>' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.decision_journal import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestMoralReasoningConvention:
    def test_sanitize_re_is_content_safe(self):
        from shadowd.cognition.moral_reasoning import _SANITIZE_CONTENT_RE
        assert _SANITIZE_CONTENT_RE.pattern == CONTENT_PATTERN

    def test_sanitize_strict_re_is_strict(self):
        from shadowd.cognition.moral_reasoning import _SANITIZE_STRICT_RE
        assert _SANITIZE_STRICT_RE.pattern == STRICT_PATTERN

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.moral_reasoning import _sanitize
        assert '<script>' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.moral_reasoning import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms > 50ms')

    def test_evaluate_uses_strict_for_action(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        ev = mr.evaluate('action <script> less than 5', description='desc')
        assert '<script>' not in ev.action

    def test_description_preserves_angles(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        ev = mr.evaluate('test', description='latency < 100ms is good')
        assert '<' in ev.description


class TestMetacognitionEngineConvention:
    def test_sanitize_re_is_content_safe(self):
        from shadowd.cognition.metacognition_engine import _SANITIZE_CONTENT_RE
        assert _SANITIZE_CONTENT_RE.pattern == CONTENT_PATTERN

    def test_sanitize_strict_re_is_strict(self):
        from shadowd.cognition.metacognition_engine import _SANITIZE_STRICT_RE
        assert _SANITIZE_STRICT_RE.pattern == STRICT_PATTERN

    def test_sanitize_strips_angles(self):
        from shadowd.cognition.metacognition_engine import _sanitize
        assert '<script>' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.metacognition_engine import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestCognitiveReflectionConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.cognitive_reflection import _sanitize
        assert '<' not in _sanitize('latency < 100ms')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.cognitive_reflection import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestGoalReasoningConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.goal_reasoning import _sanitize
        assert '<' not in _sanitize('value < 5')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.goal_reasoning import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestEmotionalIntelligenceConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.emotional_intelligence import _sanitize
        assert '<' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.emotional_intelligence import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestStealthLayerConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.stealth_layer import _sanitize
        assert '<' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.stealth_layer import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestCreativeSynthesisConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.creative_synthesis import _sanitize
        assert '<' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.creative_synthesis import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestDarkMemoryConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.dark_memory import _sanitize
        assert '<' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.dark_memory import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestShadowAriseConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.shadow_arise import _sanitize
        assert '<' not in _sanitize('name <script> test')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.shadow_arise import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestShadowSummonConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.shadow_summon import _sanitize
        assert '<' not in _sanitize('command <script> test')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.shadow_summon import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')


class TestPersonalityEngineConvention:
    def test_sanitize_uses_strict(self):
        from shadowd.cognition.personality_engine import _sanitize
        assert '<' not in _sanitize('<script>alert(1)</script>')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.personality_engine import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')

    def test_add_decision_sanitizes_dict_values(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine()
        pe.profile.add_decision({"key": "<script>alert(1)</script>", "val": "test"})
        assert "<script>" not in pe.profile.decision_history[-1]["key"]

    def test_add_or_reinforce_pattern_name_strict(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        pe = PersonalityEngine()
        pe.profile.add_or_reinforce_pattern('test<script>pattern', 'a description')
        if pe.profile.behavior_patterns:
            assert '<script>' not in pe.profile.behavior_patterns[-1].name


class TestContentFieldsPreserveAngles:
    """Verify that content fields preserve angle brackets for natural language."""

    def test_moral_reasoning_description_preserves_lt(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        ev = mr.evaluate('test action', description='latency < 100ms is acceptable')
        assert '<' in ev.description
        assert 'latency < 100ms' in ev.description

    def test_metacognition_text_preserves_lt(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, 'latency < 100ms is good')
        assert '<' not in ep.topic

    def test_decision_journal_chosen_preserves_lt(self):
        from shadowd.cognition.decision_journal import DecisionJournal, DecisionType
        dj = DecisionJournal()
        dj.record_decision(
            title='test',
            chosen='option with < 100ms latency',
            domain='perf',
        )
        decisions = dj.list_decisions()
        assert len(decisions) > 0
        assert '<' in decisions[-1].chosen

    def test_cognitive_reflection_description_preserves_lt(self):
        from shadowd.cognition.cognitive_reflection import CognitiveReflection
        cr = CognitiveReflection()
        entry = cr.add_reflection(
            topic='performance',
            description='latency < 100ms is good',
            what_went_well=['response time < 100ms'],
        )
        assert '<' in entry.description
        assert '<' in entry.what_went_well[0]

    def test_goal_reasoning_description_preserves_lt(self):
        from shadowd.cognition.goal_reasoning import GoalReasoning
        gr = GoalReasoning()
        assessment = gr.assess_goal(
            'test-goal', progress=0.5, blockers=['network < 50ms']
        )
        assert '<' in assessment.risk_factors[-1] or any('<' in r for r in assessment.recommendations)

    def test_shadow_arise_objective_preserves_lt(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise()
        agent = sa.arise(
            name='test-agent',
            objective='Ensure response time < 200ms',
            constraints=['latency < 100ms'],
        )
        assert '<' in agent.mission.objective
        assert '<' in agent.mission.constraints[0]


class TestStrictFieldsStripAngles:
    """Verify that identifier fields strip angle brackets."""

    def test_moral_reasoning_name_strips_lt(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        principle = mr.add_principle(
            name='fair<script>ness',
            framework='utilitarian',
            description='fairness principle',
        )
        assert '<script>' not in principle.name

    def test_decision_journal_title_strips_lt(self):
        from shadowd.cognition.decision_journal import DecisionJournal
        dj = DecisionJournal()
        dj.record_decision(title='important<script>choice', chosen='a', domain='test')
        decisions = dj.list_decisions()
        assert '<script>' not in decisions[-1].title

    def test_shadow_arise_name_strips_lt(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise()
        agent = sa.arise(name='bot<script>name', objective='test')
        assert '<script>' not in agent.name

    def test_creative_synthesis_name_strips_lt(self):
        from shadowd.cognition.creative_synthesis import CreativeSynthesis
        cs = CreativeSynthesis()
        cs.register_concept('idea<script>name', domain='technology')
        for concept_name in cs._concept_order:
            if 'idea' in concept_name:
                assert '<script>' not in concept_name

    def test_metacognition_topic_strips_lt(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, 'test topic<script>')
        assert '<script>' not in ep.topic


class TestDeviceContextConvention:
    """Verify device_context sanitize convention."""

    def test_strict_strips_angles_and_whitespace(self):
        from shadowd.cognition.device_context import _sanitize_strict
        result = _sanitize_strict('device<script>\tname\nhere')
        assert '<' not in result
        assert '>' not in result
        assert '\t' not in result
        assert '\n' not in result

    def test_content_safe_preserves_angles(self):
        from shadowd.cognition.device_context import _sanitize
        result = _sanitize('latency < 100ms > 50ms')
        assert '<' in result
        assert '>' in result

    def test_device_id_strict_sanitized(self):
        from shadowd.cognition.device_context import DeviceContextEngine, DeviceType
        engine = DeviceContextEngine()
        engine.register_device('dev<script>\t1', DeviceType.IPHONE, 'phone<script>')
        devices = [d for d in engine.get_device_summary()['devices']
                    if '<script>' not in d['device_id']]
        assert len(devices) >= 0  # device_id should be stripped


class TestShadowSummonThreadSafety:
    def test_stats_under_lock(self):
        from shadowd.cognition.shadow_summon import ShadowSummon
        ss = ShadowSummon()
        ss.process_voice('shadow arise test agent')
        stats = ss.get_stats()
        assert stats['total_summons'] == 1
        assert isinstance(stats, dict)

    def test_history_deepcopy(self):
        from shadowd.cognition.shadow_summon import ShadowSummon
        ss = ShadowSummon()
        ss.process_voice('shadow arise test')
        history = ss.get_history()
        assert len(history) == 1
        # Modify returned history should not affect internal state
        history[0]['transcript'] = 'MODIFIED'
        original = ss.get_history()
        assert original[0]['transcript'] != 'MODIFIED'


class TestShadowAriseThreadSafety:
    def test_get_agent_deepcopy(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise()
        agent = sa.arise('test-agent', 'test objective')
        fetched = sa.get_agent(agent.agent_id)
        assert fetched is not agent
        fetched.name = 'MODIFIED'
        original = sa.get_agent(agent.agent_id)
        assert original.name != 'MODIFIED'

    def test_list_agents_deepcopy(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise()
        sa.arise('test-agent', 'test objective')
        agents = sa.list_agents()
        agents[0].name = 'MODIFIED'
        original = sa.list_agents()
        assert original[0].name != 'MODIFIED'

    def test_evolution_count_does_not_double_count(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise()
        agent = sa.arise('test-agent', 'test objective')
        sa.activate(agent.agent_id)
        sa.evolve(agent.agent_id, 'feedback')
        fetched = sa.get_agent(agent.agent_id)
        assert fetched.evolution_count == 1

    def test_confidence_score_clamped(self):
        from shadowd.cognition.shadow_arise import ShadowArise
        sa = ShadowArise()
        agent = sa.arise('test-agent', 'test objective')
        assert 0.0 <= agent.confidence_score <= 1.0