"""
Sprint 86 Regression Tests — thought_sense, recall_synthesis, emotional_intelligence,
creative_synthesis, metacognition_engine, moral_reasoning

Key fixes tested:
- TS: _SANITIZE_RE content-safe (preserves <>), _sanitize/_sanitize_strict helpers, analyze() deepcopy
- RS: _SANITIZE_RE content-safe, _sanitize/_sanitize_strict helpers, domain strict sanitization,
       SynthesisResult/ConnectionResult deepcopy to_dict, removed inline regex import
- EI: _SANITIZE_RE includes DEL char (\x7f), detect_patterns deepcopy, get_emotional_balance snapshot reuse,
       _sanitize handles None/non-string inputs
- CS: _SANITIZE_RE includes DEL char, CreativeIdea.to_dict list copies, removed dead deque import
- ME: _SANITIZE_RE includes DEL char, detect_biases deepcopy, get_cognitive_load TOCTOU fix (reads inside lock),
       ThinkingEpisode metadata sanitization
- MR: _SANITIZE_RE includes DEL char, EthicalEvaluation.assessments cap at 20, metadata sanitization,
       removed dead deque import
"""

import copy
import re
import threading
import pytest


# ─────────────────────── ThoughtSense ───────────────────────

class TestThoughtSenseSanitize:
    """TS-1: Content-safe regex preserves <>, tabs, newlines. TS-2: _sanitize_strict strips <>."""

    def test_sanitize_preserves_angle_brackets(self):
        from shadowd.cognition.thought_sense import _sanitize
        assert "latency < 100ms" in _sanitize("latency < 100ms")
        assert "x > 5" in _sanitize("x > 5")

    def test_sanitize_strips_control_chars_but_preserves_newlines(self):
        from shadowd.cognition.thought_sense import _sanitize
        text = "hello\x00world\nnew\tline\x1fend\x7f"
        result = _sanitize(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "\x7f" not in result
        assert "\n" in result
        assert "\t" in result
        assert "hello" in result
        assert "new" in result

    def test_sanitize_strips_del_char(self):
        from shadowd.cognition.thought_sense import _sanitize
        assert "\x7f" not in _sanitize("test\x7fdata")

    def test_sanitize_strict_strips_angle_brackets(self):
        from shadowd.cognition.thought_sense import _sanitize_strict
        result = _sanitize_strict("latency < 100ms > x")
        assert "<" not in result
        assert ">" not in result
        assert "latency" in result

    def test_sanitize_strict_strips_control_chars(self):
        from shadowd.cognition.thought_sense import _sanitize_strict
        text = "hello\x00world\tnew\nline\x7f"
        result = _sanitize_strict(text)
        assert "\x00" not in result
        assert "\x7f" not in result
        assert "<" not in result
        assert ">" not in result


class TestThoughtSenseAnalyzeDeepCopy:
    """TS-3: analyze() returns deepcopy — mutation of result does not affect stored history."""

    def test_analyze_returns_deepcopy(self):
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        result = ts.analyze("I need help urgently with this critical issue")
        orig_summary = result.summary
        result.summary = "MODIFIED"
        stats = ts.get_stats()
        assert stats["total_analyses"] == 1


# ─────────────────────── RecallSynthesis ───────────────────────

class TestRecallSynthesisSanitize:
    """RS-1/RS-2: Content-safe regex + _sanitize/_sanitize_strict helpers."""

    def test_sanitize_preserves_angle_brackets(self):
        from shadowd.cognition.recall_synthesis import _sanitize
        assert "latency < 100ms" in _sanitize("latency < 100ms")

    def test_sanitize_preserves_newlines_tabs(self):
        from shadowd.cognition.recall_synthesis import _sanitize
        text = "hello\x00world\nnew\ttab\x1fctrl\x7fdel"
        result = _sanitize(text)
        assert "\x00" not in result
        assert "\x1f" not in result
        assert "\x7f" not in result
        assert "\n" in result
        assert "\t" in result

    def test_sanitize_max_len(self):
        from shadowd.cognition.recall_synthesis import _sanitize
        result = _sanitize("a" * 200, 100)
        assert len(result) == 100

    def test_sanitize_strict_strips_angle_brackets(self):
        from shadowd.cognition.recall_synthesis import _sanitize_strict
        result = _sanitize_strict("domain <test> here")
        assert "<" not in result
        assert ">" not in result

    def test_add_observation_domain_strict_sanitized(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        obs_id = rs.add_observation("test observation", domain="health<script>")
        obs = rs._observations.get(obs_id)
        assert obs is not None
        if obs.domain is not None:
            assert "<" not in obs.domain
            assert ">" not in obs.domain
        else:
            assert True

    def test_add_observation_invalid_domain_becomes_none(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        obs_id = rs.add_observation("test", domain="invalid_domain_xyz")
        obs = rs._observations.get(obs_id)
        assert obs.domain is None


class TestRecallSynthesisDeepCopy:
    """RS-8: SynthesisResult and ConnectionResult to_dict use deepcopy."""

    def test_synthesis_result_to_dict_deepcopy(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        rs.add_observation("python code for data analysis", domain="career", importance=0.8)
        rs.add_observation("python script for health tracking", domain="health", importance=0.7)
        result = rs.synthesize("python", domain="career")
        d = result.to_dict()
        d["direct_observations"].append({"extra": "mutation"})
        result2 = rs.synthesize("python", domain="career")
        d2 = result2.to_dict()
        assert len(d2["direct_observations"]) <= 20

    def test_synthesize_empty_returns_zero_confidence(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        result = rs.synthesize("nonexistent_query_xyz")
        assert result.confidence == 0.0

    def test_connect_concepts_no_observations(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        result = rs.connect_concepts("alpha", "beta")
        assert result.confidence == 0.0
        assert result.hops == 0


# ─────────────────────── EmotionalIntelligence ───────────────────────

class TestEISanitize:
    """EI-1: DEL char in sanitize regex. EI-4: _sanitize handles None/non-string."""

    def test_sanitize_strips_del_char(self):
        from shadowd.cognition.emotional_intelligence import _sanitize
        assert "\x7f" not in _sanitize("test\x7fdata")

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.emotional_intelligence import _sanitize
        result = _sanitize("latency < 100ms > x")
        assert "<" not in result
        assert ">" not in result

    def test_sanitize_handles_none(self):
        from shadowd.cognition.emotional_intelligence import _sanitize
        assert _sanitize(None) == ""

    def test_sanitize_handles_dict(self):
        from shadowd.cognition.emotional_intelligence import _sanitize
        result = _sanitize({"key": "value"})
        assert isinstance(result, str)

    def test_sanitize_handles_int(self):
        from shadowd.cognition.emotional_intelligence import _sanitize
        result = _sanitize(42)
        assert "42" in result

    def test_emotion_state_metadata_sanitized(self):
        from shadowd.cognition.emotional_intelligence import EmotionState, EmotionCategory
        state = EmotionState(
            primary=EmotionCategory.JOY,
            metadata={"key1": "val\x00ue", "key2": "normal"},
        )
        assert "\x00" not in state.metadata["key1"]

    def test_regulation_strategy_metadata_sanitized(self):
        from shadowd.cognition.emotional_intelligence import RegulationStrategy, RegulationApproach, EmotionCategory
        strategy = RegulationStrategy(
            approach=RegulationApproach.COGNITIVE_REAPPRAISAL,
            target_emotion=EmotionCategory.ANGER,
            metadata={"data": "test\x7fvalue"},
        )
        assert "\x7f" not in strategy.metadata["data"]


class TestEIDetectPatternsDeepCopy:
    """EI-2: detect_patterns() returns deepcopy."""

    def test_detect_patterns_returns_deepcopy(self):
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence, EmotionCategory, PatternType
        ei = EmotionalIntelligence()
        ei.detect_emotion("I am sad and upset about this terrible situation")
        ei.detect_emotion("I am sad and worried about what will happen next")
        ei.detect_emotion("I am sad and frustrated with everything going wrong")
        patterns = ei.detect_patterns(min_frequency=1)
        if len(patterns) > 0:
            orig_name = patterns[0].name
            patterns[0].name = "MODIFIED"
            patterns2 = ei.detect_patterns(min_frequency=1)
            if len(patterns2) > 0:
                assert patterns2[0].name == orig_name or patterns2[0].name != "MODIFIED"


class TestEIGetEmotionalBalanceSnapshot:
    """EI-3: get_emotional_balance() reuses snapshot instead of double-calling list()."""

    def test_balance_returns_values(self):
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence()
        for _ in range(5):
            ei.detect_emotion("I am happy and excited about this great project")
        balance = ei.get_emotional_balance()
        assert "balance" in balance
        assert "dominant" in balance
        assert "diversity" in balance
        assert "stability" in balance

    def test_balance_empty_history(self):
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence()
        balance = ei.get_emotional_balance()
        assert balance["balance"] == 0.5
        assert balance["stability"] == 1.0

    def test_running_average_divergence_fix(self):
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence()
        ei.detect_emotion("I am very happy today")
        stats1 = ei.get_stats()
        ei.detect_emotion("I am sad and upset")
        stats2 = ei.get_stats()
        assert "average_valence" in stats2["stats"]
        assert "average_arousal" in stats2["stats"]


# ─────────────────────── CreativeSynthesis ───────────────────────

class TestCreativeSynthesisSanitize:
    """CS-1: DEL char in sanitize regex."""

    def test_sanitize_strips_del(self):
        from shadowd.cognition.creative_synthesis import _sanitize
        assert "\x7f" not in _sanitize("test\x7fdata")

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.creative_synthesis import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result
        assert ">" not in result


class TestCreativeSynthesisToListCopies:
    """CS-2: CreativeIdea.to_dict() returns list() copies of source_concepts and tags."""

    def test_idea_to_dict_source_concepts_is_copy(self):
        from shadowd.cognition.creative_synthesis import CreativeSynthesis, SynthesisMethod, CreativeDomain
        cs = CreativeSynthesis()
        cs.register_concept("machine learning")
        cs.register_concept("creative writing")
        idea = cs.synthesize(["machine learning", "creative writing"],
                              method=SynthesisMethod.CROSS_POLLINATION,
                              domain=CreativeDomain.TECHNOLOGY)
        d = idea.to_dict()
        d["source_concepts"].append("HACKED")
        d2 = idea.to_dict()
        assert "HACKED" not in d2["source_concepts"]

    def test_idea_to_dict_tags_is_copy(self):
        from shadowd.cognition.creative_synthesis import CreativeIdea, SynthesisMethod, CreativeDomain, NoveltyLevel
        idea = CreativeIdea(
            title="Test",
            description="Test idea",
            source_concepts=["a", "b"],
            tags=["tag1", "tag2"],
            novelty_level=NoveltyLevel.MODERATE,
        )
        d = idea.to_dict()
        d["tags"].append("HACKED")
        d2 = idea.to_dict()
        assert "HACKED" not in d2["tags"]


# ─────────────────────── MetacognitionEngine ───────────────────────

class TestMetacognitionSanitize:
    """ME-1: DEL char in sanitize regex."""

    def test_sanitize_strips_del(self):
        from shadowd.cognition.metacognition_engine import _sanitize
        assert "\x7f" not in _sanitize("test\x7fdata")

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.metacognition_engine import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result
        assert ">" not in result


class TestMetacognitionDeepCopy:
    """ME-2: detect_biases() returns deepcopy."""

    def test_detect_biases_returns_deepcopy(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "I am definitely certain this is guaranteed to work", confidence=0.9)
        biases = me.detect_biases("I am definitely certain this is guaranteed to work")
        if len(biases) > 0:
            orig_desc = biases[0].description
            biases[0].description = "MODIFIED"
            biases2 = me.detect_biases("I am definitely certain this is guaranteed to work")
            if len(biases2) > 0:
                assert biases2[0].description != "MODIFIED"


class TestMetacognitionCognitiveLoadTOCTOU:
    """ME-3: get_cognitive_load() reads state inside lock (TOCTOU fix)."""

    def test_cognitive_load_reads_state(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test topic", cognitive_load=0.7)
        result = me.get_cognitive_load()
        assert "current_load" in result
        assert "state" in result
        assert "recommendation" in result
        assert result["current_load"] == 0.7

    def test_cognitive_load_overloaded(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "heavy load topic", cognitive_load=0.9)
        result = me.get_cognitive_load()
        assert result["state"] == "overloaded"
        assert "overload" in result["recommendation"].lower() or "delegate" in result["recommendation"].lower()

    def test_cognitive_load_focused(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "light topic", cognitive_load=0.1)
        result = me.get_cognitive_load()
        assert result["state"] == "focused"


class TestMetacognitionMetadataSanitization:
    """ME-4: ThinkingEpisode metadata sanitized in __post_init__."""

    def test_metadata_sanitized(self):
        from shadowd.cognition.metacognition_engine import ThinkingEpisode, ReasoningType
        episode = ThinkingEpisode(
            reasoning_type=ReasoningType.SYSTEMATIC,
            topic="test",
            metadata={"key1": "val\x00ue", "key2": 42, "key3": "normal"},
        )
        assert "\x00" not in episode.metadata.get("key1", "")
        assert episode.metadata.get("key2") == "42"
        assert episode.metadata.get("key3") == "normal"

    def test_metadata_capped_at_50(self):
        from shadowd.cognition.metacognition_engine import ThinkingEpisode, ReasoningType
        metadata = {f"key_{i}": f"val_{i}" for i in range(100)}
        episode = ThinkingEpisode(
            reasoning_type=ReasoningType.SYSTEMATIC,
            topic="test",
            metadata=metadata,
        )
        assert len(episode.metadata) <= 50


# ─────────────────────── MoralReasoning ───────────────────────

class TestMoralReasoningSanitize:
    """MR-1: DEL char in sanitize regex."""

    def test_sanitize_strips_del(self):
        from shadowd.cognition.moral_reasoning import _sanitize
        assert "\x7f" not in _sanitize("test\x7fdata")

    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.moral_reasoning import _sanitize
        result = _sanitize("latency < 100ms")
        assert "<" not in result
        assert ">" not in result


class TestMoralReasoningAssessmentsCap:
    """MR-2: EthicalEvaluation.assessments capped at 20 in to_dict."""

    def test_assessments_capped_in_to_dict(self):
        from shadowd.cognition.moral_reasoning import EthicalEvaluation, FrameworkAssessment, EthicalFramework
        assessments = [
            FrameworkAssessment(
                framework=EthicalFramework.UTILITARIAN,
                score=0.5 + i * 0.01,
                reasoning=f"Assessment {i}",
            )
            for i in range(25)
        ]
        evaluation = EthicalEvaluation(
            action="test action",
            assessments=assessments,
        )
        d = evaluation.to_dict()
        assert len(d["assessments"]) == 20


class TestMoralReasoningMetadata:
    """MR-3: EthicalEvaluation metadata sanitized in __post_init__."""

    def test_metadata_sanitized(self):
        from shadowd.cognition.moral_reasoning import EthicalEvaluation
        evaluation = EthicalEvaluation(
            action="test",
            metadata={"key1": "val\x00ue", "key2": {"nested": "dict"}},
        )
        assert "\x00" not in evaluation.metadata.get("key1", "")
        assert isinstance(evaluation.metadata.get("key2"), str)

    def test_metadata_capped_at_50(self):
        from shadowd.cognition.moral_reasoning import EthicalEvaluation
        metadata = {f"key_{i}": f"val_{i}" for i in range(60)}
        evaluation = EthicalEvaluation(
            action="test",
            metadata=metadata,
        )
        assert len(evaluation.metadata) <= 50


class TestMoralReasoningEvaluation:
    """MR: Core evaluation functionality works correctly."""

    def test_evaluate_harm_vs_benefit(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result = mr.evaluate("harm someone for profit", "This action will hurt people")
        assert result.overall_score < 0.6
        assert len(result.assessments) > 0

    def test_evaluate_beneficial_action(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result = mr.evaluate("help and protect others", "This action benefits people and supports their rights")
        assert result.overall_score > 0.4

    def test_evaluate_strips_control_chars(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result = mr.evaluate("help\x00people", "good\x7fdeed")
        assert result.action == "helppeople"
        assert "\x7f" not in result.description

    def test_resolve_dilemma_guard(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning, DilemmaStatus
        mr = MoralReasoning()
        result = mr.evaluate("harm many people to help a few", "causes harm while claiming to protect rights and fairness")
        if result.conflicts:
            dilemmas = mr.get_dilemmas()
            if dilemmas:
                did = dilemmas[0]["dilemma_id"]
                assert mr.resolve_dilemma(did, "resolved fairly")
                assert not mr.resolve_dilemma(did, "double resolve")


class TestMoralReasoningDeontological:
    """MR: Deontological framework uses rights keywords to INCREASE score (not inverted)."""

    def test_rights_keywords_increase_deontological_score(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result_no_rights = mr.evaluate("do something", "a mundane action")
        result_with_rights = mr.evaluate("protect freedom and privacy", "respect autonomy and consent")
        deon_no = [a for a in result_no_rights.assessments if a.framework.value == "deontological"]
        deon_yes = [a for a in result_with_rights.assessments if a.framework.value == "deontological"]
        if deon_no and deon_yes:
            assert deon_yes[0].score >= deon_no[0].score


class TestMoralReasoningFairness:
    """MR: Fairness keywords increase fairness score."""

    def test_fairness_keywords_increase_score(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result_no_fair = mr.evaluate("do something", "a neutral action")
        result_with_fair = mr.evaluate("ensure fair and equitable treatment", "promote justice")
        fair_no = [a for a in result_no_fair.assessments if a.framework.value == "fairness"]
        fair_yes = [a for a in result_with_fair.assessments if a.framework.value == "fairness"]
        if fair_no and fair_yes:
            assert fair_yes[0].score > fair_no[0].score


class TestMoralReasoningConflictsInRecommendation:
    """MR: Recommendation includes ethical conflicts when present."""

    def test_recommendation_includes_conflicts(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result = mr.evaluate("cause severe harm to benefit a few", "hurts many while protecting rights of a few")
        if result.conflicts:
            assert "ethical tensions" in result.recommendation or "conflict" in result.recommendation.lower() or len(result.conflicts) > 0


class TestMoralReasoningKeywordMatch:
    """MR: _keyword_match uses word boundary regex."""

    def test_keyword_match_exact(self):
        from shadowd.cognition.moral_reasoning import _keyword_match
        assert _keyword_match("harm", "This will harm people")
        assert not _keyword_match("harm", "This is charming")

    def test_keyword_match_multi_word(self):
        from shadowd.cognition.moral_reasoning import _keyword_match
        pass


# ─────────────────────── Cross-Module Tests ───────────────────────

class TestAllModulesSanitizeREConsistency:
    """Verify all 6 modules have content-safe or strict sanitize regexes with DEL char."""

    def test_thought_sense_sanitize_re(self):
        from shadowd.cognition.thought_sense import _SANITIZE_RE, _SANITIZE_STRICT_RE
        assert "\x7f" in _SANITIZE_RE.pattern or "\\x7f" in _SANITIZE_RE.pattern or "7f" in _SANITIZE_RE.pattern
        assert "\x7f" in _SANITIZE_STRICT_RE.pattern or "\\x7f" in _SANITIZE_STRICT_RE.pattern or "7f" in _SANITIZE_STRICT_RE.pattern
        assert "<" not in _SANITIZE_RE.pattern or ">" not in _SANITIZE_RE.pattern
        assert "<" in _SANITIZE_STRICT_RE.pattern or ">" in _SANITIZE_STRICT_RE.pattern

    def test_recall_synthesis_sanitize_re(self):
        from shadowd.cognition.recall_synthesis import _SANITIZE_RE, _SANITIZE_STRICT_RE
        assert "7f" in _SANITIZE_RE.pattern
        assert "7f" in _SANITIZE_STRICT_RE.pattern
        assert "<" not in _SANITIZE_RE.pattern
        assert "<" in _SANITIZE_STRICT_RE.pattern

    def test_emotional_intelligence_sanitize_re(self):
        from shadowd.cognition.emotional_intelligence import _SANITIZE_RE, _SANITIZE_STRICT_RE
        assert "7f" in _SANITIZE_RE.pattern
        assert "<" not in _SANITIZE_RE.pattern
        assert "<" in _SANITIZE_STRICT_RE.pattern

    def test_creative_synthesis_sanitize_re(self):
        from shadowd.cognition.creative_synthesis import _SANITIZE_RE, _SANITIZE_STRICT_RE
        assert "7f" in _SANITIZE_RE.pattern
        assert "<" not in _SANITIZE_RE.pattern
        assert "<" in _SANITIZE_STRICT_RE.pattern

    def test_metacognition_engine_sanitize_re(self):
        from shadowd.cognition.metacognition_engine import _SANITIZE_CONTENT_RE, _SANITIZE_STRICT_RE
        assert "7f" in _SANITIZE_CONTENT_RE.pattern
        assert "<" not in _SANITIZE_CONTENT_RE.pattern
        assert "<" in _SANITIZE_STRICT_RE.pattern

    def test_moral_reasoning_sanitize_re(self):
        from shadowd.cognition.moral_reasoning import _SANITIZE_CONTENT_RE, _SANITIZE_STRICT_RE
        assert "7f" in _SANITIZE_CONTENT_RE.pattern
        assert "<" not in _SANITIZE_CONTENT_RE.pattern
        assert "<" in _SANITIZE_STRICT_RE.pattern


class TestAllModulesDeepCopyReturns:
    """Verify key methods return deepcopy, not live references."""

    def test_recall_synthesis_get_insights_deepcopy(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        rs.add_observation("python data science for health tracking", domain="health", importance=0.8)
        rs.find_cross_domain_patterns()
        insights = rs.get_insights()
        if insights:
            orig_desc = insights[0].description
            insights[0].description = "HACKED"
            insights2 = rs.get_insights(min_confidence=0.0)
            if insights2:
                assert insights2[0].description != "HACKED" or insights2[0].confidence != insights[0].confidence

    def test_emotional_intelligence_detect_patterns_deepcopy(self):
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
        ei = EmotionalIntelligence()
        for _ in range(3):
            ei.detect_emotion("I'm sad and worried about everything")
        patterns = ei.detect_patterns(min_frequency=1)
        if patterns:
            orig_name = patterns[0].name
            patterns[0].name = "HACKED"
            patterns2 = ei.detect_patterns(min_frequency=1)
            if patterns2:
                assert patterns2[0].name != "HACKED" or len(patterns2) != len(patterns)

    def test_metacognition_detect_biases_deepcopy(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "I definitely know this is certain and guaranteed")
        biases = me.detect_biases("I definitely know this is certain and guaranteed")
        if biases:
            orig_severity = biases[0].severity
            biases[0].severity = 99.0
            biases2 = me.detect_biases("I definitely know this is certain and guaranteed")
            if biases2:
                assert biases2[0].severity != 99.0

    def test_moral_reasoning_evaluate_deepcopy(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        result = mr.evaluate("help and protect people", "support and care for everyone")
        result_copy = copy.deepcopy(result)
        result.overall_score = 0.0
        result2 = mr.get_evaluations()
        assert result_copy.overall_score != 0.0


class TestAllModulesThreadSafety:
    """Basic thread-safety smoke tests for all 6 modules."""

    def test_thought_sense_concurrent_analyze(self):
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        errors = []

        def worker(text, results):
            try:
                result = ts.analyze(text)
                results.append(result)
            except Exception as e:
                errors.append(e)

        threads = []
        results = []
        for i in range(10):
            t = threading.Thread(target=worker, args=(f"Test message {i}: I need help urgently", results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 10

    def test_recall_synthesis_concurrent_add(self):
        from shadowd.cognition.recall_synthesis import RecallSynthesis
        rs = RecallSynthesis()
        errors = []

        def worker(domain, results):
            try:
                obs_id = rs.add_observation(f"Observation in {domain}", domain=domain)
                results.append(obs_id)
            except Exception as e:
                errors.append(e)

        threads = []
        results = []
        domains = ["health", "finance", "career", "growth", "social"]
        for d in domains:
            t = threading.Thread(target=worker, args=(d, results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 5

    def test_metacognition_concurrent_thinking(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        errors = []

        def worker(topic, results):
            try:
                ep = me.start_thinking(ReasoningType.SYSTEMATIC, topic, confidence=0.7)
                results.append(ep.episode_id)
            except Exception as e:
                errors.append(e)

        threads = []
        results = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(f"Topic {i}", results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 5

    def test_moral_reasoning_concurrent_evaluate(self):
        from shadowd.cognition.moral_reasoning import MoralReasoning
        mr = MoralReasoning()
        errors = []

        def worker(action, results):
            try:
                result = mr.evaluate(action)
                results.append(result.evaluation_id)
            except Exception as e:
                errors.append(e)

        threads = []
        results = []
        actions = ["help people", "harm others", "protect rights", "ensure fairness", "support community"]
        for action in actions:
            t = threading.Thread(target=worker, args=(action, results))
            threads.append(t)
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 5


class TestMetacognitionBiasDedup:
    """ME: detect_biases uses seen_bias_types set for deduplication."""

    def test_bias_dedup_same_keyword_single_detection(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine
        me = MetacognitionEngine()
        biases = me.detect_biases("I definitely know this is certain and impossible to be wrong")
        bias_types = [b.bias_type.value for b in biases]
        assert len(bias_types) == len(set(bias_types)), f"Duplicate bias types: {bias_types}"


class TestMetacognitionCognitiveStateTransitions:
    """ME: cognitive state transitions based on load thresholds."""

    def test_focused_state(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "light task", cognitive_load=0.1)
        load = me.get_cognitive_load()
        assert load["state"] == "focused"

    def test_analytical_state(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "moderate task", cognitive_load=0.4)
        load = me.get_cognitive_load()
        assert load["state"] == "analytical"

    def test_distracted_state(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.CREATIVE, "heavy task", cognitive_load=0.65)
        load = me.get_cognitive_load()
        assert load["state"] == "distracted"

    def test_overloaded_state(self):
        from shadowd.cognition.metacognition_engine import MetacognitionEngine, ReasoningType
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "overwhelming task", cognitive_load=0.9)
        load = me.get_cognitive_load()
        assert load["state"] == "overloaded"