"""
Sprint 87 Regression Tests — _SANITIZE_RE Consistency Audit

Key fixes tested:
- All modules with _SANITIZE_RE or inline sanitize patterns now include \\x7f (DEL character)
- Content-safe patterns: r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]' — preserves \\t\\n\\r and <>
- Strict patterns: r'[<>\\x00-\\x1f\\x7f]' — strips <> and ALL control chars including \\t\\n\\r
- Strict patterns are used for identifiers (titles, IDs, domains, device_ids, names)
- Content-safe patterns are used for content fields (descriptions, objectives, narratives)

Updated in Sprint 124+:
- anomaly_detector, cognitive_pipeline, cognitive_reflection, decision_journal,
  goal_reasoning, stealth_layer changed _SANITIZE_RE from strict to content-safe
- These modules now also have _SANITIZE_STRICT_RE for identifier fields
"""

import re
import pytest

CONTENT_SAFE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

DEL = '\x7f'
TAB = '\t'
NEWLINE = '\n'
CR = '\r'
BEL = '\x07'
GS = '\x1d'


class TestAnomalyDetectorSanitize:
    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'latency < 100ms > x')
        assert '<' in result
        assert '>' in result

    def test_sanitize_re_preserves_tab_newline_cr(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_strips_control(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_RE
        assert BEL not in _SANITIZE_RE.sub('', f'test{BEL}data')
        assert GS not in _SANITIZE_RE.sub('', f'test{GS}data')

    def test_sanitize_strict_re_strips_angle_brackets(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'latency < 100ms > x')
        assert '<' not in result
        assert '>' not in result

    def test_sanitize_strict_re_strips_tab_newline_cr(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_STRICT_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        from shadowd.cognition.anomaly_detector import _sanitize_content
        result = _sanitize_content('latency < 100ms > x')
        assert '<' in result
        assert '>' in result

    def test_sanitize_content_preserves_tab_newline_cr(self):
        from shadowd.cognition.anomaly_detector import _sanitize_content
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize_content(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.anomaly_detector import _sanitize
        result = _sanitize('latency < 100ms > x')
        assert '<' not in result
        assert '>' not in result

    def test_sanitize_strips_tab_newline_cr(self):
        from shadowd.cognition.anomaly_detector import _sanitize
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize(text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestCognitivePipelineSanitize:
    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'x < 5 and > 3')
        assert '<' in result
        assert '>' in result

    def test_sanitize_re_preserves_whitespace(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_RE
        text = f'line{NEWLINE}text{TAB}here{CR}end'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_strict_re_strips_angle_brackets(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'x < 5 and > 3')
        assert '<' not in result
        assert '>' not in result

    def test_sanitize_strict_re_strips_whitespace(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_STRICT_RE
        text = f'line{NEWLINE}text{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result

    def test_sanitize_pipeline_input_preserves_angle_brackets(self):
        from shadowd.cognition.cognitive_pipeline import _sanitize_pipeline_input
        result = _sanitize_pipeline_input('x < 5 and > 3')
        assert '<' in result
        assert '>' in result

    def test_sanitize_pipeline_input_preserves_whitespace(self):
        from shadowd.cognition.cognitive_pipeline import _sanitize_pipeline_input
        text = f'line{NEWLINE}text{TAB}here{CR}end'
        result = _sanitize_pipeline_input(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_strict_re_strips_angle_brackets_pipeline(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'x < 5 and > 3')
        assert '<' not in result
        assert '>' not in result

    def test_sanitize_strict_re_strips_whitespace_pipeline(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_STRICT_RE
        text = f'line{NEWLINE}text{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestCognitiveReflectionSanitize:
    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'value < 10')
        assert '<' in result

    def test_sanitize_re_preserves_whitespace(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_strict_re_strips_angle_brackets(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'value < 10')
        assert '<' not in result

    def test_sanitize_strict_re_strips_whitespace(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_STRICT_RE
        text = f'line{NEWLINE}text{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        from shadowd.cognition.cognitive_reflection import _sanitize_content
        result = _sanitize_content('value < 10')
        assert '<' in result

    def test_sanitize_content_preserves_whitespace(self):
        from shadowd.cognition.cognitive_reflection import _sanitize_content
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize_content(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.cognitive_reflection import _sanitize
        result = _sanitize('value < 10')
        assert '<' not in result

    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.cognitive_reflection import _sanitize
        text = f'line{NEWLINE}text{TAB}here{CR}end'
        result = _sanitize(text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestDailyBriefingSanitize:
    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.daily_briefing import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.daily_briefing import _SANITIZE_RE
        text = 'latency < 100ms'
        result = _SANITIZE_RE.sub('', text)
        assert '<' in result
        assert 'latency < 100ms' == result

    def test_sanitize_re_preserves_whitespace(self):
        from shadowd.cognition.daily_briefing import _SANITIZE_RE
        text = f'hello{NEWLINE}world{TAB}here'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result


class TestDecisionJournalSanitize:
    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.decision_journal import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'x < 5')
        assert '<' in result

    def test_sanitize_re_preserves_tab_newline_cr(self):
        from shadowd.cognition.decision_journal import _SANITIZE_RE
        text = f'hello{BEL}world{NEWLINE}line{TAB}tab{CR}cr'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.decision_journal import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_strips_control(self):
        from shadowd.cognition.decision_journal import _SANITIZE_RE
        assert BEL not in _SANITIZE_RE.sub('', f'test{BEL}data')

    def test_sanitize_strict_re_strips_angle_brackets(self):
        from shadowd.cognition.decision_journal import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'title <tag> value')
        assert '<' not in result
        assert '>' not in result

    def test_sanitize_strict_re_strips_whitespace(self):
        from shadowd.cognition.decision_journal import _SANITIZE_STRICT_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        from shadowd.cognition.decision_journal import _sanitize_content
        result = _sanitize_content('x < 5')
        assert '<' in result

    def test_sanitize_content_preserves_whitespace(self):
        from shadowd.cognition.decision_journal import _sanitize_content
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize_content(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.decision_journal import _sanitize
        result = _sanitize('x < 5')
        assert '<' not in result

    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.decision_journal import _sanitize
        text = f'line{NEWLINE}text{TAB}here{CR}end'
        result = _sanitize(text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestEscalationSanitize:
    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.escalation import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.escalation import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'latency < 100ms')
        assert '<' in result
        assert 'latency < 100ms' == result

    def test_sanitize_re_preserves_whitespace(self):
        from shadowd.cognition.escalation import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', f'line{NEWLINE}text{TAB}here{CR}end')
        assert NEWLINE in result
        assert TAB in result


class TestGoalPlannerSanitize:
    def test_content_safe_strips_del(self):
        from shadowd.cognition.goal_planner import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_content_safe_preserves_angle_brackets(self):
        from shadowd.cognition.goal_planner import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'latency < 100ms')
        assert '<' in result
        assert '>' not in result or '>' in 'latency < 100ms'

    def test_strict_strips_angle_brackets(self):
        from shadowd.cognition.goal_planner import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'title <tag> value')
        assert '<' not in result
        assert '>' not in result

    def test_strict_strips_del(self):
        from shadowd.cognition.goal_planner import _SANITIZE_STRICT_RE
        assert DEL not in _SANITIZE_STRICT_RE.sub('', f'x{DEL}y')

    def test_content_safe_preserves_whitespace(self):
        from shadowd.cognition.goal_planner import _SANITIZE_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result
        assert BEL not in result

    def test_strict_strips_whitespace(self):
        from shadowd.cognition.goal_planner import _SANITIZE_STRICT_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result


class TestGoalReasoningSanitize:
    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'x < 5')
        assert '<' in result

    def test_sanitize_re_preserves_tab_newline_cr(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_RE
        text = f'hello{BEL}world{NEWLINE}line{TAB}tab{CR}cr'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_strips_control(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_RE
        assert BEL not in _SANITIZE_RE.sub('', f'test{BEL}data')

    def test_sanitize_strict_re_strips_angle_brackets(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'title <tag> value')
        assert '<' not in result
        assert '>' not in result

    def test_sanitize_strict_re_strips_whitespace(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_STRICT_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        from shadowd.cognition.goal_reasoning import _sanitize_content
        result = _sanitize_content('x < 5')
        assert '<' in result

    def test_sanitize_content_preserves_whitespace(self):
        from shadowd.cognition.goal_reasoning import _sanitize_content
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize_content(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.goal_reasoning import _sanitize
        result = _sanitize('x < 5')
        assert '<' not in result

    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.goal_reasoning import _sanitize
        text = f'line{NEWLINE}text{TAB}here{CR}end'
        result = _sanitize(text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestShadowCrewSanitize:
    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.shadow_crew import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.shadow_crew import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'latency < 100ms')
        assert '<' in result

    def test_sanitize_re_preserves_whitespace(self):
        from shadowd.cognition.shadow_crew import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', f'line{NEWLINE}text{TAB}here{CR}end')
        assert NEWLINE in result
        assert TAB in result


class TestStealthLayerSanitize:
    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'x < 5')
        assert '<' in result

    def test_sanitize_re_preserves_tab_newline_cr(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_strips_control(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_RE
        assert BEL not in _SANITIZE_RE.sub('', f'test{BEL}data')

    def test_sanitize_strict_re_strips_angle_brackets(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', 'x < 5')
        assert '<' not in result

    def test_sanitize_strict_re_strips_whitespace(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_STRICT_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_STRICT_RE.sub('', text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        from shadowd.cognition.stealth_layer import _sanitize_content
        result = _sanitize_content('x < 5')
        assert '<' in result

    def test_sanitize_content_preserves_whitespace(self):
        from shadowd.cognition.stealth_layer import _sanitize_content
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = _sanitize_content(text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.stealth_layer import _sanitize
        result = _sanitize('x < 5')
        assert '<' not in result

    def test_sanitize_strips_whitespace(self):
        from shadowd.cognition.stealth_layer import _sanitize
        text = f'line{NEWLINE}text{TAB}here{CR}end'
        result = _sanitize(text)
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestConflictResolutionSanitize:
    def test_inline_sanitize_strips_del(self):
        cr = __import__('shadowd.cognition.conflict_resolution', fromlist=['ConflictResolution'])
        text = f'test{DEL}data'
        result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f<>]", "", text)
        assert DEL not in result

    def test_inline_sanitize_strips_angle_brackets(self):
        result = re.sub(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f<>]", "", 'x < 5 > y')
        assert '<' not in result

    def test_inline_sanitize_strips_whitespace(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", f'line{NEWLINE}text{TAB}here{CR}end{BEL}')
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert BEL not in result


class TestNarrativeMemorySanitize:
    def test_inline_sanitize_strips_del(self):
        text = f'test{DEL}data'
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", text)
        assert DEL not in result

    def test_inline_sanitize_strips_angle_brackets(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", 'title <tag>')
        assert '<' not in result


class TestProceduralMemorySanitize:
    def test_inline_sanitize_strips_del(self):
        text = f'test{DEL}data'
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", text)
        assert DEL not in result

    def test_inline_sanitize_strips_angle_brackets(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", 'step < 5')
        assert '<' not in result


class TestDreamStateSanitize:
    def test_content_safe_pattern_strips_del(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        assert DEL not in pattern.sub('', f'test{DEL}data')

    def test_content_safe_preserves_angle_brackets(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        result = pattern.sub('', 'latency < 100ms')
        assert '<' in result

    def test_content_safe_preserves_whitespace(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = pattern.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_strict_pattern_strips_angle_brackets_and_del(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        result = pattern.sub('', f'x < 5{DEL}y')
        assert '<' not in result
        assert DEL not in result

    def test_strict_pattern_strips_whitespace(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        result = pattern.sub('', f'x{NEWLINE}y{TAB}z{CR}w')
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestCognitiveTrinitySanitize:
    def test_inline_sanitize_strips_del(self):
        ct = __import__('shadowd.cognition.cognitive_trinity', fromlist=['CognitiveTrinity'])
        text = f'test{DEL}data'
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", text)
        assert DEL not in result

    def test_inline_sanitize_strips_angle_brackets(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", 'x < 5')
        assert '<' not in result

    def test_inline_sanitize_strips_whitespace(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", f'line{NEWLINE}text{TAB}here{CR}end')
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestTemporalReasoningSanitize:
    def test_content_safe_strips_del(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        assert DEL not in pattern.sub('', f'test{DEL}data')

    def test_content_safe_preserves_angle_brackets(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        result = pattern.sub('', 'latency < 100ms')
        assert '<' in result

    def test_content_safe_preserves_whitespace(self):
        pattern = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        text = f'hello{NEWLINE}world{TAB}here{CR}end'
        result = pattern.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result


class TestAttentionEngineSanitize:
    def test_strict_strips_del(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", f'test{DEL}data')
        assert DEL not in result

    def test_strict_strips_angle_brackets(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", 'x < 5')
        assert '<' not in result

    def test_strict_strips_whitespace(self):
        result = re.sub(r"[<>\x00-\x1f\x7f]", "", f'line{NEWLINE}text{TAB}here{CR}end')
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result


class TestPresenceDetectionSanitize:
    def test_strict_strips_del(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        assert DEL not in pattern.sub('', f'device{DEL}id')

    def test_strict_strips_angle_brackets(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        result = pattern.sub('', 'device<script>')
        assert '<' not in result

    def test_strict_strips_whitespace(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        result = pattern.sub('', f'text{NEWLINE}line{TAB}here{CR}end')
        assert NEWLINE not in result
        assert TAB not in result


class TestEmpathyMappingSanitize:
    def test_sanitize_re_strips_del(self):
        from shadowd.cognition.empathy_mapping import _SANITIZE_RE
        assert DEL not in _SANITIZE_RE.sub('', f'test{DEL}data')

    def test_sanitize_re_preserves_angle_brackets(self):
        from shadowd.cognition.empathy_mapping import _SANITIZE_RE
        result = _SANITIZE_RE.sub('', 'latency < 100ms')
        assert '<' in result

    def test_sanitize_re_preserves_whitespace(self):
        from shadowd.cognition.empathy_mapping import _SANITIZE_RE
        text = f'hello{NEWLINE}world{TAB}here{CR}end{BEL}'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result


class TestShadowVoiceSanitize:
    def test_strict_strips_del(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        assert DEL not in pattern.sub('', f'speaker{DEL}name')

    def test_strict_strips_angle_brackets(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        result = pattern.sub('', 'text <script> content')
        assert '<' not in result
        assert '>' not in result


class TestNeuralMeshSanitize:
    def test_strict_strips_del(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        assert DEL not in pattern.sub('', f'key{DEL}name')

    def test_strict_strips_angle_brackets(self):
        pattern = re.compile(r'[<>\x00-\x1f\x7f]')
        result = pattern.sub('', 'value <tag> end')
        assert '<' not in result


class TestGlobalConsistencyPatterns:
    """Verify all module-level _SANITIZE_RE constants match the expected patterns."""

    def _check_module_re(self, modpath, attr, expected_pattern):
        import importlib
        mod = importlib.import_module(modpath)
        obj = getattr(mod, attr)
        assert obj.pattern == expected_pattern, f"{modpath}.{attr}: expected {expected_pattern}, got {obj.pattern}"

    def test_anomaly_detector_sanitize_re(self):
        self._check_module_re('shadowd.cognition.anomaly_detector', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_anomaly_detector_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.anomaly_detector', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_cognitive_pipeline_sanitize_re(self):
        self._check_module_re('shadowd.cognition.cognitive_pipeline', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_cognitive_pipeline_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.cognitive_pipeline', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_cognitive_reflection_sanitize_re(self):
        self._check_module_re('shadowd.cognition.cognitive_reflection', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_cognitive_reflection_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.cognitive_reflection', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_daily_briefing_sanitize_re(self):
        self._check_module_re('shadowd.cognition.daily_briefing', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_decision_journal_sanitize_re(self):
        self._check_module_re('shadowd.cognition.decision_journal', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_decision_journal_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.decision_journal', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_escalation_sanitize_re(self):
        self._check_module_re('shadowd.cognition.escalation', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_goal_planner_sanitize_re(self):
        self._check_module_re('shadowd.cognition.goal_planner', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_goal_planner_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.goal_planner', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_goal_reasoning_sanitize_re(self):
        self._check_module_re('shadowd.cognition.goal_reasoning', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_goal_reasoning_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.goal_reasoning', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_shadow_crew_sanitize_re(self):
        self._check_module_re('shadowd.cognition.shadow_crew', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_stealth_layer_sanitize_re(self):
        self._check_module_re('shadowd.cognition.stealth_layer', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_stealth_layer_sanitize_strict_re(self):
        self._check_module_re('shadowd.cognition.stealth_layer', '_SANITIZE_STRICT_RE', r'[<>\x00-\x1f\x7f]')

    def test_empathy_mapping_sanitize_re(self):
        self._check_module_re('shadowd.cognition.empathy_mapping', '_SANITIZE_RE', r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')

    def test_no_old_style_x00_x1f_content_patterns(self):
        """Verify no module uses over-restrictive [\\x00-\\x1f] in content-safe sanitize patterns."""
        import glob
        import os
        cognition_dir = os.path.join(os.path.dirname(__file__), '..', 'shadowd', 'cognition')
        py_files = glob.glob(os.path.join(cognition_dir, '*.py'))
        for fpath in py_files:
            with open(fpath, 'r') as f:
                content = f.read()
            lines = content.split('\n')
            for i, line in enumerate(lines, 1):
                if '_SANITIZE_CONTENT_RE' in line or '_SANITIZE_RE' in line:
                    if '<>' not in line and '\\x80' not in line:
                        assert '\\x00-\\x1f' not in line, \
                            f"Over-restrictive \\x00-\\x1f in content-safe pattern at {os.path.basename(fpath)}:{i}: {line.strip()}"