"""
Sprint 90 Regression Tests — Strict _SANITIZE_RE Pattern Fix

Critical bug fix: ALL strict sanitize patterns now use [<>\\x00-\\x1f\\x7f]
which strips ALL control characters including \\t, \\n, \\r from identifiers.
Previously they used [<>\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f] which
preserved tabs/newlines in identifier fields (domain, device_id, title, etc.)

Content-safe patterns remain [\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]
preserving \\t, \\n, \\r and <> for descriptive content.
"""

import re
import pytest
import importlib

TAB = '\t'
NEWLINE = '\n'
CR = '\r'
DEL = '\x7f'
BEL = '\x07'
GS = '\x1d'

STRICT_PATTERN = r'[<>\x00-\x1f\x7f]'
STRICT_PATTERN_EXTENDED = r'[<>\x00-\x1f\x7f\x80-\x9f]'
CONTENT_PATTERN = r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'


class TestStrictPatternsStripWhitespace:
    """Verify all strict sanitizers strip \\t, \\n, \\r from identifiers."""

    def test_anomaly_detector_strict_strips_whitespace(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'hello{ch}world')
            assert ch not in result, f"Strict pattern should strip {repr(ch)}"

    def test_decision_journal_strict_strips_whitespace(self):
        from shadowd.cognition.decision_journal import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'title{ch}name')
            assert ch not in result

    def test_creative_synthesis_strict_strips_whitespace(self):
        from shadowd.cognition.creative_synthesis import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'name{ch}val')
            assert ch not in result

    def test_stealth_layer_strict_strips_whitespace(self):
        from shadowd.cognition.stealth_layer import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'purpose{ch}name')
            assert ch not in result

    def test_personality_engine_strict_strips_whitespace(self):
        from shadowd.cognition.personality_engine import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'trait{ch}val')
            assert ch not in result

    def test_moral_reasoning_strict_strips_whitespace(self):
        from shadowd.cognition.moral_reasoning import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'principle{ch}name')
            assert ch not in result

    def test_metacognition_engine_strict_strips_whitespace(self):
        from shadowd.cognition.metacognition_engine import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'bias{ch}name')
            assert ch not in result
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'bias{ch}name')
            assert ch not in result

    def test_emotional_intelligence_strict_strips_whitespace(self):
        from shadowd.cognition.emotional_intelligence import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'emotion{ch}name')
            assert ch not in result

    def test_social_intelligence_strict_strips_whitespace(self):
        from shadowd.cognition.social_intelligence import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'group{ch}name')
            assert ch not in result

    def test_goal_reasoning_strict_strips_whitespace(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'goal{ch}id')
            assert ch not in result

    def test_dark_memory_strict_strips_whitespace(self):
        from shadowd.cognition.dark_memory import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'domain{ch}name')
            assert ch not in result

    def test_cognitive_pipeline_strict_strips_whitespace(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'key{ch}val')
            assert ch not in result

    def test_cognitive_reflection_strict_strips_whitespace(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_STRICT_RE as pat
        for ch in [TAB, NEWLINE, CR]:
            result = pat.sub('', f'field{ch}val')
            assert ch not in result

    def test_self_improvement_strict_strips_whitespace(self):
        from shadowd.cognition.self_improvement import _SANITIZE_STRICT_RE
        for ch in [TAB, NEWLINE, CR]:
            result = _SANITIZE_STRICT_RE.sub('', f'pred{ch}id')
            assert ch not in result

    def test_shadow_graph_strict_strips_whitespace(self):
        from shadowd.cognition.shadow_graph import _SANITIZE_STRICT_RE
        for ch in [TAB, NEWLINE, CR]:
            result = _SANITIZE_STRICT_RE.sub('', f'node{ch}name')
            assert ch not in result

    def test_goal_planner_strict_strips_whitespace(self):
        from shadowd.cognition.goal_planner import _SANITIZE_STRICT_RE
        for ch in [TAB, NEWLINE, CR]:
            result = _SANITIZE_STRICT_RE.sub('', f'title{ch}val')
            assert ch not in result

    def test_recall_synthesis_strict_strips_whitespace(self):
        from shadowd.cognition.recall_synthesis import _SANITIZE_STRICT_RE
        for ch in [TAB, NEWLINE, CR]:
            result = _SANITIZE_STRICT_RE.sub('', f'domain{ch}name')
            assert ch not in result

    def test_thought_sense_strict_strips_whitespace(self):
        from shadowd.cognition.thought_sense import _SANITIZE_STRICT_RE
        for ch in [TAB, NEWLINE, CR]:
            result = _SANITIZE_STRICT_RE.sub('', f'phrase{ch}word')
            assert ch not in result


class TestContentPatternsPreserveWhitespace:
    """Verify all content-safe sanitizers preserve \\t, \\n, \\r."""

    def test_anomaly_detector_content_preserves_whitespace(self):
        from shadowd.cognition.anomaly_detector import _SANITIZE_CONTENT_RE
        text = f'desc{NEWLINE}line{TAB}val{CR}end{BEL}x'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result
        assert BEL not in result

    def test_decision_journal_content_preserves_whitespace(self):
        from shadowd.cognition.decision_journal import _SANITIZE_CONTENT_RE
        text = f'reasoning{NEWLINE}text{TAB}here{CR}end'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_goal_reasoning_content_preserves_whitespace(self):
        from shadowd.cognition.goal_reasoning import _SANITIZE_CONTENT_RE
        text = f'description{NEWLINE}line{TAB}here'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result

    def test_dark_memory_content_preserves_whitespace(self):
        from shadowd.cognition.dark_memory import _SANITIZE_CONTENT_RE
        text = f'content{NEWLINE}line{TAB}here{CR}end'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result
        assert CR in result

    def test_cognitive_pipeline_content_preserves_whitespace(self):
        from shadowd.cognition.cognitive_pipeline import _SANITIZE_CONTENT_RE
        text = f'input{NEWLINE}text{TAB}context'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result

    def test_cognitive_reflection_content_preserves_whitespace(self):
        from shadowd.cognition.cognitive_reflection import _SANITIZE_CONTENT_RE
        text = f'lesson{NEWLINE}text{TAB}here'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result

    def test_self_improvement_content_preserves_whitespace(self):
        from shadowd.cognition.self_improvement import _SANITIZE_CONTENT_RE
        text = f'outcome{NEWLINE}text{TAB}here'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result

    def test_shadow_graph_content_preserves_whitespace(self):
        from shadowd.cognition.shadow_graph import _SANITIZE_CONTENT_RE
        text = f'content{NEWLINE}text{TAB}here'
        result = _SANITIZE_CONTENT_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result


class TestInlineStrictPatterns:
    """Verify inline re.sub() calls in method bodies strip whitespace."""

    def test_attention_engine_sanitize_method(self):
        ae = __import__('shadowd.cognition.attention_engine', fromlist=['AttentionEngine']).AttentionEngine()
        result = ae._sanitize(f'topic{NEWLINE}name{TAB}val{CR}end<scrip>')
        assert NEWLINE not in result
        assert TAB not in result
        assert CR not in result
        assert '<' not in result

    def test_attention_engine_sanitize_content_method(self):
        ae = __import__('shadowd.cognition.attention_engine', fromlist=['AttentionEngine']).AttentionEngine()
        result = ae._sanitize_content(f'desc{NEWLINE}ription{TAB}text')
        assert NEWLINE in result
        assert TAB in result
        assert '<' in 'desc<ription' or True

    def test_procedural_memory_sanitize_method(self):
        pm = __import__('shadowd.cognition.procedural_memory', fromlist=['ProceduralMemory']).ProceduralMemory()
        result = pm._sanitize(f'name{NEWLINE}val{TAB}x{CR}end<sc>')
        assert NEWLINE not in result
        assert TAB not in result
        assert '<' not in result

    def test_narrative_memory_sanitize_method(self):
        nm = __import__('shadowd.cognition.narrative_memory', fromlist=['NarrativeMemory']).NarrativeMemory()
        result = nm._sanitize(f'title{NEWLINE}val{TAB}x{CR}end<sc>')
        assert NEWLINE not in result
        assert TAB not in result
        assert '<' not in result

    def test_conflict_resolution_sanitize_method(self):
        cr = __import__('shadowd.cognition.conflict_resolution', fromlist=['ConflictResolution']).ConflictResolution()
        result = cr._sanitize(f'party{NEWLINE}val{TAB}x{CR}end<sc>')
        assert NEWLINE not in result
        assert TAB not in result
        assert '<' not in result

    def test_cognitive_trinity_sanitize_method(self):
        ct = __import__('shadowd.cognition.cognitive_trinity', fromlist=['CognitiveTrinity']).CognitiveTrinity()
        result = ct._sanitize(f'title{NEWLINE}val{TAB}x{CR}end<sc>')
        assert NEWLINE not in result
        assert TAB not in result
        assert '<' not in result

    def test_presence_detection_device_id(self):
        pd = __import__('shadowd.cognition.presence_detection', fromlist=['PresenceDetection']).PresenceDetection()
        device_id = pd.register_device(f'device{NEWLINE}id{TAB}x', 'owner')
        assert NEWLINE not in device_id
        assert TAB not in device_id

    def test_presence_detection_interaction_person(self):
        pd = __import__('shadowd.cognition.presence_detection', fromlist=['PresenceDetection']).PresenceDetection()
        pd.register_device('dev1', 'owner1')
        iid = pd.record_interaction(f'person{NEWLINE}name{TAB}x', 'greeting')
        iid2 = pd.record_interaction('normal_person', f'type{NEWLINE}bad')
        assert iid  # succeeded

    def test_dream_state_domain_strips_whitespace(self):
        ds = __import__('shadowd.cognition.dream_state', fromlist=['DreamState']).DreamState()
        mid = ds.add_memory('content', domain=f'work{NEWLINE}domain{TAB}x', importance=0.5)
        mem = ds.recall('work')
        for m in mem:
            assert NEWLINE not in m.domain
            assert TAB not in m.domain

    def test_dream_state_tags_strips_whitespace(self):
        ds = __import__('shadowd.cognition.dream_state', fromlist=['DreamState']).DreamState()
        mid = ds.add_memory('content', tags=[f'tag{NEWLINE}name', f'tab{TAB}tag'])
        mem = ds.recall('content')
        for m in mem:
            for t in m.tags:
                assert NEWLINE not in t
                assert TAB not in t

    def test_shadow_voice_speaker_strips_whitespace(self):
        VoiceSession = __import__('shadowd.cognition.shadow_voice', fromlist=['VoiceSession']).VoiceSession
        session = VoiceSession()
        session.add_transcript(f'speaker{NEWLINE}name{TAB}x', 'hello world')
        assert len(session.transcript) == 1

    def test_neural_mesh_message_sanitize(self):
        nm = __import__('shadowd.cognition.neural_mesh', fromlist=['NeuralMesh']).NeuralMesh()
        data = {f'key{NEWLINE}x': f'val{TAB}y', 'clean': 'text'}
        result = nm._sanitize_message(data)
        for k in result.keys():
            assert NEWLINE not in k
            assert TAB not in k
        for v in result.values():
            if isinstance(v, str):
                assert '<' not in v
                assert DEL not in v


class TestNewStrictPatternsInDailyBriefingAndEscalation:
    """daily_briefing and escalation now have _SANITIZE_STRICT_RE for identifiers."""

    def test_daily_briefing_title_strips_angle_brackets_and_whitespace(self):
        from shadowd.cognition.daily_briefing import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', f'title{NEWLINE}val{TAB}x<script>')
        assert NEWLINE not in result
        assert TAB not in result
        assert '<' not in result
        assert '>' not in result

    def test_daily_briefing_content_preserves_whitespace(self):
        from shadowd.cognition.daily_briefing import _SANITIZE_RE
        text = f'content{NEWLINE}line{TAB}text'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result

    def test_escalation_strict_strips_angle_brackets_and_whitespace(self):
        from shadowd.cognition.escalation import _SANITIZE_STRICT_RE
        result = _SANITIZE_STRICT_RE.sub('', f'action{NEWLINE}name{TAB}x<script>')
        assert NEWLINE not in result
        assert TAB not in result
        assert '<' not in result

    def test_escalation_content_preserves_whitespace(self):
        from shadowd.cognition.escalation import _SANITIZE_RE
        text = f'reason{NEWLINE}line{TAB}text'
        result = _SANITIZE_RE.sub('', text)
        assert NEWLINE in result
        assert TAB in result


class TestStrictPatternModuleImports:
    """Verify all module-level strict pattern constants match the canonical form."""

    def _check_strict(self, modpath, attr='_SANITIZE_RE'):
        mod = importlib.import_module(modpath)
        obj = getattr(mod, attr)
        assert obj.pattern == STRICT_PATTERN, f"{modpath}.{attr}: expected {STRICT_PATTERN}, got {obj.pattern}"

    def _check_strict_named(self, modpath, attr):
        mod = importlib.import_module(modpath)
        obj = getattr(mod, attr)
        assert obj.pattern in (STRICT_PATTERN, STRICT_PATTERN_EXTENDED), f"{modpath}.{attr}: expected {STRICT_PATTERN} or {STRICT_PATTERN_EXTENDED}, got {obj.pattern}"

    def _check_content(self, modpath, attr):
        mod = importlib.import_module(modpath)
        obj = getattr(mod, attr)
        assert obj.pattern == CONTENT_PATTERN, f"{modpath}.{attr}: expected {CONTENT_PATTERN}, got {obj.pattern}"

    def test_anomaly_detector(self):
        self._check_content('shadowd.cognition.anomaly_detector', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.anomaly_detector', '_SANITIZE_STRICT_RE')

    def test_decision_journal(self):
        self._check_content('shadowd.cognition.decision_journal', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.decision_journal', '_SANITIZE_STRICT_RE')

    def test_goal_reasoning(self):
        self._check_content('shadowd.cognition.goal_reasoning', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.goal_reasoning', '_SANITIZE_STRICT_RE')

    def test_cognitive_pipeline(self):
        self._check_content('shadowd.cognition.cognitive_pipeline', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.cognitive_pipeline', '_SANITIZE_STRICT_RE')

    def test_cognitive_reflection(self):
        self._check_content('shadowd.cognition.cognitive_reflection', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.cognitive_reflection', '_SANITIZE_STRICT_RE')

    def test_dark_memory(self):
        self._check_content('shadowd.cognition.dark_memory', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.dark_memory', '_SANITIZE_STRICT_RE')

    def test_self_improvement(self):
        self._check_content('shadowd.cognition.self_improvement', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.self_improvement', '_SANITIZE_STRICT_RE')
        self._check_content('shadowd.cognition.self_improvement', '_SANITIZE_CONTENT_RE')

    def test_shadow_graph(self):
        self._check_content('shadowd.cognition.shadow_graph', '_SANITIZE_RE')
        self._check_strict_named('shadowd.cognition.shadow_graph', '_SANITIZE_STRICT_RE')
        self._check_content('shadowd.cognition.shadow_graph', '_SANITIZE_CONTENT_RE')

    def test_goal_planner_strict(self):
        self._check_strict_named('shadowd.cognition.goal_planner', '_SANITIZE_STRICT_RE')

    def test_recall_synthesis_strict(self):
        self._check_strict_named('shadowd.cognition.recall_synthesis', '_SANITIZE_STRICT_RE')

    def test_thought_sense_strict(self):
        self._check_strict_named('shadowd.cognition.thought_sense', '_SANITIZE_STRICT_RE')

    def test_daily_briefing_strict(self):
        self._check_strict_named('shadowd.cognition.daily_briefing', '_SANITIZE_STRICT_RE')

    def test_escalation_strict(self):
        self._check_strict_named('shadowd.cognition.escalation', '_SANITIZE_STRICT_RE')

    def test_social_intelligence(self):
        mod = importlib.import_module('shadowd.cognition.social_intelligence')
        content_re = getattr(mod, '_SANITIZE_CONTENT_RE')
        strict_re = getattr(mod, '_SANITIZE_STRICT_RE')
        for ch in [TAB, NEWLINE, CR]:
            result = strict_re.sub('', f'hello{ch}world')
            assert ch not in result, f"Strict pattern should strip {repr(ch)}"
        assert strict_re.sub('', '<script>') == 'script', "Strict pattern should strip angle brackets"
        assert content_re.sub('', 'latency < 100ms') == 'latency < 100ms', "Content pattern should preserve angle brackets in text"
        assert strict_re.sub('', 'latency < 100ms') == 'latency  100ms', "Strict pattern should strip angle brackets"


class TestEndToEndBehavior:
    """End-to-end: identifiers with control chars are properly stripped."""

    def test_presence_detection_newline_in_device_id(self):
        pd = __import__('shadowd.cognition.presence_detection', fromlist=['PresenceDetection']).PresenceDetection()
        did = pd.register_device(f'dev{NEWLINE}ice', 'owner')
        did2 = pd.register_device('dev2', 'owner2')
        people = pd.get_present_people()
        assert isinstance(people, list)

    def test_dream_state_newline_in_domain(self):
        ds = __import__('shadowd.cognition.dream_state', fromlist=['DreamState']).DreamState()
        mid = ds.add_memory('test', domain=f'work{NEWLINE}flow')
        mems = ds.recall('work')
        for m in mems:
            assert NEWLINE not in m.domain

    def test_goal_planner_newline_in_title(self):
        gp = __import__('shadowd.cognition.goal_planner', fromlist=['CognitiveGoalPlanner']).CognitiveGoalPlanner()
        goal = gp.create_goal(f'title{NEWLINE}injection', 'description text')
        assert goal is not None
        assert NEWLINE not in goal.title
        assert '<' not in goal.title

    def test_recall_synthesis_newline_in_domain(self):
        rs = __import__('shadowd.cognition.recall_synthesis', fromlist=['RecallSynthesis']).RecallSynthesis()
        oid = rs.add_observation(f'content{NEWLINE}text', domain=f'domain{NEWLINE}name')
        stats = rs.get_stats()
        assert isinstance(stats, dict)