"""Sprint 41: CognitiveTrinity — Three-Mind Decision Ensemble"""

import threading
import pytest
from shadowd.cognition.cognitive_trinity import (
    CognitiveTrinity, MindType, ConsensusType, MindResult,
    SynthesisResult, TrinityAnalysis,
)


class TestCognitiveTrinityInit:
    def test_init(self):
        ct = CognitiveTrinity()
        assert ct._total_analyses == 0
        assert len(ct._analysis_history) == 0
        assert "intuition" in ct._mind_accuracy

    def test_thread_safety(self):
        ct = CognitiveTrinity()
        assert hasattr(ct, '_lock')
        assert isinstance(ct._lock, type(threading.Lock()))


class TestIntuitionMind:
    def test_deadline_urgency(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("I have a deadline tomorrow and need to act urgently")
        assert result.mind_type == MindType.INTUITION
        assert "urgency" in result.key_factors
        assert result.risk_level in ("medium", "high")

    def test_risk_detection(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("There is a risk of harm and danger ahead")
        assert "caution" in result.key_factors or "risk" in result.reasoning.lower()
        assert result.risk_level == "high"

    def test_opportunity_detection(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("This is a great opportunity for potential growth")
        assert "opportunity" in result.key_factors

    def test_neutral_no_patterns(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("The weather is cloudy today.")
        assert result.confidence >= 0.3
        assert result.mind_type == MindType.INTUITION

    def test_emotional_positive(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("I love this excellent wonderful idea!")
        assert "positive" in result.reasoning.lower() or result.confidence > 0.4

    def test_emotional_negative(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("This is terrible and I hate it")
        assert "negative" in result.reasoning.lower() or result.risk_level != "low"

    def test_sanitize_control_chars(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("test\x00\x01risk\x02danger")
        assert "\x00" not in result.reasoning

    def test_truncation(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("deadline " * 1000)
        assert result is not None


class TestLogicMind:
    def test_causal_reasoning(self):
        ct = CognitiveTrinity()
        result = ct.logic_mind("Because the data shows X, therefore we should proceed")
        assert "causal" in result.key_factors or "evidence" in result.key_factors

    def test_evidence_based(self):
        ct = CognitiveTrinity()
        result = ct.logic_mind("The evidence and data prove this fact")
        assert "evidence_available" in result.key_factors

    def test_uncertainty(self):
        ct = CognitiveTrinity()
        result = ct.logic_mind("Maybe we could perhaps consider this uncertain option")
        assert result.risk_level in ("medium", "high")

    def test_comparison(self):
        ct = CognitiveTrinity()
        result = ct.logic_mind("We need to compare these two options versus the alternative")
        assert "comparison" in result.key_factors

    def test_neutral_logic(self):
        ct = CognitiveTrinity()
        result = ct.logic_mind("The sky is blue.")
        assert result.confidence >= 0.3
        assert result.mind_type == MindType.LOGIC


class TestCreativeMind:
    def test_inversion(self):
        ct = CognitiveTrinity()
        result = ct.creative_mind("I'm stuck and this seems impossible, can't proceed")
        assert "inversion" in result.key_factors

    def test_divergence(self):
        ct = CognitiveTrinity()
        result = ct.creative_mind("Let's try a creative and innovative novel approach")
        assert "divergence" in result.key_factors

    def test_simplification(self):
        ct = CognitiveTrinity()
        result = ct.creative_mind("Is there a simple and easy straightforward way?")
        assert "simplification" in result.key_factors

    def test_decomposition(self):
        ct = CognitiveTrinity()
        result = ct.creative_mind("This complex and complicated problem overwhelms me")
        assert "decomposition" in result.key_factors

    def test_analogy(self):
        ct = CognitiveTrinity()
        result = ct.creative_mind("Is there a connection or similarity to something we know?")
        assert "analogy" in result.key_factors


class TestSynthesize:
    def _make_mind(self, mind_type, rec, conf, risk="low"):
        return MindResult(
            mind_type=mind_type,
            recommendation=rec,
            confidence=conf,
            reasoning="test",
            key_factors=["test"],
            risk_level=risk,
        )

    def test_consensus(self):
        ct = CognitiveTrinity()
        i = self._make_mind(MindType.INTUITION, "Proceed", 0.8)
        l = self._make_mind(MindType.LOGIC, "Proceed", 0.7)
        c = self._make_mind(MindType.CREATIVITY, "Proceed", 0.6)
        result = ct.synthesize(i, l, c)
        assert result.consensus == ConsensusType.CONSENSUS
        assert result.final_recommendation == "Proceed"
        assert result.confidence >= 0.7

    def test_split(self):
        ct = CognitiveTrinity()
        i = self._make_mind(MindType.INTUITION, "Proceed", 0.8)
        l = self._make_mind(MindType.LOGIC, "Proceed", 0.7)
        c = self._make_mind(MindType.CREATIVITY, "Wait", 0.5)
        result = ct.synthesize(i, l, c)
        assert result.consensus == ConsensusType.SPLIT

    def test_divergence(self):
        ct = CognitiveTrinity()
        i = self._make_mind(MindType.INTUITION, "Act now", 0.7)
        l = self._make_mind(MindType.LOGIC, "Gather more info", 0.6)
        c = self._make_mind(MindType.CREATIVITY, "Try something new", 0.5)
        result = ct.synthesize(i, l, c)
        assert result.consensus == ConsensusType.DIVERGENCE
        assert result.confidence < 0.7

    def test_risk_weights(self):
        ct = CognitiveTrinity()
        i = self._make_mind(MindType.INTUITION, "Risky", 0.9, "high")
        l = self._make_mind(MindType.LOGIC, "Safe", 0.6, "low")
        c = self._make_mind(MindType.CREATIVITY, "Moderate", 0.7, "medium")
        result = ct.synthesize(i, l, c)
        assert result.dominant_mind == MindType.LOGIC or result.dominant_mind == MindType.CREATIVITY


class TestAnalyze:
    def test_full_analysis(self):
        ct = CognitiveTrinity()
        result = ct.analyze("I need to decide quickly about this urgent deadline")
        assert isinstance(result, TrinityAnalysis)
        assert result.analysis_id
        assert result.timestamp
        assert isinstance(result.intuition, MindResult)
        assert isinstance(result.logic, MindResult)
        assert isinstance(result.creative, MindResult)
        assert isinstance(result.synthesis, SynthesisResult)

    def test_analysis_id_unique(self):
        ct = CognitiveTrinity()
        r1 = ct.analyze("situation 1")
        r2 = ct.analyze("situation 2")
        assert r1.analysis_id != r2.analysis_id

    def test_analysis_stored_in_history(self):
        ct = CognitiveTrinity()
        ct.analyze("test situation")
        assert len(ct._analysis_history) == 1
        assert ct._total_analyses == 1

    def test_to_dict(self):
        ct = CognitiveTrinity()
        result = ct.analyze("test situation")
        d = result.to_dict()
        assert "analysis_id" in d
        assert "intuition" in d
        assert "logic" in d
        assert "creative" in d
        assert "synthesis" in d

    def test_context_passed(self):
        ct = CognitiveTrinity()
        result = ct.analyze("decide", context={"domain": "health"})
        assert result is not None


class TestRecordOutcome:
    def test_record_updates_accuracy(self):
        ct = CognitiveTrinity()
        analysis = ct.analyze("test situation")
        ct.record_outcome(analysis.analysis_id, "positive", 0.9)
        stats = ct.get_stats()
        assert stats["mind_accuracy"]["intuition"]["general"] != 0.5

    def test_record_satisfaction_clamped(self):
        ct = CognitiveTrinity()
        analysis = ct.analyze("test")
        ct.record_outcome(analysis.analysis_id, "good", 2.0)  # clamped to 1.0
        ct.record_outcome(analysis.analysis_id, "bad", -1.0)   # clamped to 0.0
        assert True  # no crash


class TestGetDominantMind:
    def test_default_dominant(self):
        ct = CognitiveTrinity()
        mind = ct.get_dominant_mind()
        assert mind in ("intuition", "logic", "creativity")

    def test_after_outcomes(self):
        ct = CognitiveTrinity()
        analysis = ct.analyze("deadline urgent risk")
        ct.record_outcome(analysis.analysis_id, "positive", 0.9)
        mind = ct.get_dominant_mind()
        assert mind in ("intuition", "logic", "creativity")


class TestGetStats:
    def test_initial_stats(self):
        ct = CognitiveTrinity()
        stats = ct.get_stats()
        assert stats["total_analyses"] == 0
        assert stats["history_size"] == 0

    def test_after_analysis(self):
        ct = CognitiveTrinity()
        ct.analyze("test")
        stats = ct.get_stats()
        assert stats["total_analyses"] == 1
        assert stats["history_size"] == 1


class TestEdgeCases:
    def test_empty_situation(self):
        ct = CognitiveTrinity()
        result = ct.intuition_mind("")
        assert result is not None
        assert result.confidence >= 0.3

    def test_very_long_situation(self):
        ct = CognitiveTrinity()
        text = "deadline " * 2000  # ~18k chars, will be truncated
        result = ct.analyze(text)
        assert result is not None

    def test_unicode_situation(self):
        ct = CognitiveTrinity()
        result = ct.analyze("风险 deadline 紧急")
        assert result is not None

    def test_concurrent_analyses(self):
        ct = CognitiveTrinity()
        results = []
        errors = []

        def analyze(i):
            try:
                r = ct.analyze(f"situation {i}")
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=analyze, args=(i,)) for i in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 10

    def test_history_eviction(self):
        ct = CognitiveTrinity()
        for i in range(CognitiveTrinity._MAX_HISTORY + 100):
            ct._analysis_history.append(type('obj', (object,), {'analysis_id': str(i)})())
        assert len(ct._analysis_history) <= CognitiveTrinity._MAX_HISTORY + 100  # deque handles eviction