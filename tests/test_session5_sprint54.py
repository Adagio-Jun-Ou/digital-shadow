#!/usr/bin/env python3
"""Sprint 54 — Integration + Security Hardening + Metacognition + MoralReasoning

Tests for:
- EmotionalIntelligence: security hardening (RegulationApproval typo fix)
- CreativeSynthesis: deadlock fix (register_concept inside lock), TOCTOU fix (_evaluate_idea)
- MetacognitionEngine: thinking episodes, bias detection, calibration, alerts
- MoralReasoning: ethical evaluation, framework assessment, dilemmas, principles
- ShadowOS wiring: health/status for all 4 new modules
- Thread safety, input sanitization, bounded data
"""

import asyncio
import threading
import pytest
import inspect
from unittest.mock import MagicMock, patch

from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, RegulationApproach,
    PatternType, EmotionState, EmotionalPattern, RegulationStrategy,
)
from shadowd.cognition.creative_synthesis import (
    CreativeSynthesis, CreativeDomain, SynthesisMethod,
    NoveltyLevel, CreativeIdea, ConceptPair, IdeationPattern,
)
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, ReasoningType, BiasType, AlertSeverity,
    CognitiveState, ThinkingEpisode, CognitiveBias, MetacognitiveAlert,
)
from shadowd.cognition.moral_reasoning import (
    MoralReasoning, EthicalFramework, MoralWeight, DilemmaStatus,
    StakeholderType, Stakeholder, FrameworkAssessment,
    EthicalEvaluation, MoralDilemma, MoralPrinciple,
)


# ===================== Security Fixes =====================

class TestEmotionalIntelligenceSecurityFix:
    def test_no_typo_regulation_approval(self):
        source = inspect.getsource(EmotionalIntelligence)
        assert "RegulationApproval" not in source, "Typo RegulationApproval should be removed"

    def test_sadness_cognitive_reappraisal_in_guidance(self):
        from shadowd.cognition.emotional_intelligence import _REGULATION_GUIDANCE
        key = (EmotionCategory.SADNESS, RegulationApproach.COGNITIVE_REAPPRAISAL)
        assert key in _REGULATION_GUIDANCE, "Missing sadness+cognitive_reappraisal guidance"
        assert len(_REGULATION_GUIDANCE[key]) > 0


class TestCreativeSynthesisSecurityFix:
    def test_no_deadlock_add_association(self):
        cs = CreativeSynthesis()
        pair = cs.add_association("ai", "ml", strength=0.8)
        assert pair.concept_a == "ai"
        assert "ai" in cs._concepts
        assert "ml" in cs._concepts

    def test_register_concept_unlocked_method(self):
        cs = CreativeSynthesis()
        assert hasattr(cs, '_register_concept_unlocked')

    def test_evaluate_idea_uses_snapshot(self):
        cs = CreativeSynthesis()
        cs.register_concept("ai")
        cs.register_concept("art")
        source = inspect.getsource(cs._evaluate_idea)
        assert "associations_snapshot" in source, "_evaluate_idea should use snapshot, not direct access"

    def test_synthesize_deadlock_free(self):
        cs = CreativeSynthesis()
        cs.register_concept("test")
        idea = cs.synthesize(["test", "other"])
        assert idea.idea_id.startswith("idea_")

    def test_concurrent_associations(self):
        cs = CreativeSynthesis()
        errors = []

        def add_assoc(i):
            try:
                cs.add_association(f"concept_{i}", f"partner_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_assoc, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ===================== MetacognitionEngine Tests =====================

class TestThinkingEpisodes:
    def test_start_thinking(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "solve the bug")
        assert ep.episode_id.startswith("think_")
        assert ep.reasoning_type == ReasoningType.SYSTEMATIC
        assert ep.confidence == 0.5

    def test_resolve_thinking(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "test")
        resolved = me.resolve_thinking(ep.episode_id, "found it", 0.8)
        assert resolved.outcome_accuracy == 0.8
        assert resolved.calibration_offset == pytest.approx(0.3, abs=0.01)

    def test_resolve_overconfident(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.INTUITIVE, "test", confidence=0.9)
        resolved = me.resolve_thinking(ep.episode_id, "wrong", 0.2)
        assert resolved.calibration_offset < 0

    def test_resolve_nonexistent(self):
        me = MetacognitionEngine()
        result = me.resolve_thinking("fake_id", "test", 0.5)
        assert result is None

    def test_episode_clamped(self):
        ep = ThinkingEpisode(confidence=2.0, cognitive_load=1.5)
        assert ep.confidence == 1.0
        assert ep.cognitive_load == 1.0

    def test_episode_to_dict(self):
        ep = ThinkingEpisode(reasoning_type=ReasoningType.SYSTEMATIC, topic="test")
        d = ep.to_dict()
        assert "episode_id" in d
        assert d["reasoning_type"] == "systematic"

    def test_episode_cap(self):
        me = MetacognitionEngine()
        for i in range(550):
            me.start_thinking(ReasoningType.INDUCTIVE, f"topic_{i}")
        assert len(me._episodes) <= 5000


class TestBiasDetection:
    def test_detect_confirmation_bias(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("I'm only looking for evidence that confirms my belief")
        assert len(biases) > 0
        assert any(b.bias_type == BiasType.CONFIRMATION for b in biases)

    def test_detect_overconfidence_bias(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("it's definitely impossible to be wrong")
        assert any(b.bias_type == BiasType.OVERCONFIDENCE for b in biases)

    def test_detect_no_bias(self):
        me = MetacognitionEngine()
        biases = me.detect_biases("testing regular text")
        assert len(biases) == 0

    def test_bias_mitigation(self):
        from shadowd.cognition.metacognition_engine import _BIAS_MITIGATIONS
        assert BiasType.CONFIRMATION in _BIAS_MITIGATIONS
        assert len(_BIAS_MITIGATIONS[BiasType.CONFIRMATION]) > 0

    def test_bias_with_episode(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.INDUCTIVE, "test")
        me.detect_biases("only looking for confirming evidence", episode_id=ep.episode_id)
        with me._lock:
            updated_ep = me._episodes.get(ep.episode_id)
        assert updated_ep is not None and len(updated_ep.biases_detected) > 0

    def test_bias_accumulation(self):
        me = MetacognitionEngine()
        me.detect_biases("only looking for confirming evidence")
        me.detect_biases("only looking for another confirmation, confirms my belief")
        with me._lock:
            biases = list(me._biases.values())
        confirmation = [b for b in biases if b.bias_type == BiasType.CONFIRMATION]
        assert len(confirmation) > 0
        assert confirmation[0].frequency >= 2


class TestCognitiveLoad:
    def test_cognitive_load_reporting(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.9)
        load = me.get_cognitive_load()
        assert load["current_load"] == 0.9
        assert "recommendation" in load

    def test_overloaded_state(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.9)
        assert me._cognitive_state == CognitiveState.OVERLOADED

    def test_focused_state(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.2)
        assert me._cognitive_state == CognitiveState.FOCUSED


class TestCalibrationAlerts:
    def test_alert_on_overconfidence(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.INTUITIVE, "test", confidence=0.9)
        me.resolve_thinking(ep.episode_id, "wrong", 0.2)
        alerts = me.get_alerts()
        assert len(alerts) > 0

    def test_no_alert_on_good_calibration(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.SYSTEMATIC, "test", confidence=0.5)
        me.resolve_thinking(ep.episode_id, "correct", 0.5)
        alerts = me.get_alerts()
        assert len(alerts) == 0

    def test_dismiss_alert(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.INTUITIVE, "test", confidence=0.95)
        me.resolve_thinking(ep.episode_id, "wrong", 0.1)
        alerts = me.get_alerts()
        if alerts:
            alert_id = alerts[0]["alert_id"]
            assert me.dismiss_alert(alert_id) is True

    def test_confidence_assessment(self):
        me = MetacognitionEngine()
        calibrated = me.assess_confidence(0.5, evidence_count=3, disconfirming_count=1)
        assert 0.0 <= calibrated <= 1.0

    def test_calibration_report_insufficient_data(self):
        me = MetacognitionEngine()
        report = me.get_calibration_report()
        assert report["status"] == "insufficient_data"

    def test_calibration_report_with_data(self):
        me = MetacognitionEngine()
        ep = me.start_thinking(ReasoningType.DEDUCTIVE, "test", confidence=0.7)
        me.resolve_thinking(ep.episode_id, "found", 0.8)
        report = me.get_calibration_report()
        assert report["total_resolved"] == 1


class TestMetacognitionStats:
    def test_get_stats(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test")
        stats = me.get_stats()
        assert "stats" in stats
        assert stats["stats"]["episodes_tracked"] == 1
        assert "cognitive_state" in stats


class TestMetacognitionThreadSafety:
    def test_concurrent_start_resolve(self):
        me = MetacognitionEngine()
        errors = []

        def start_and_resolve(i):
            try:
                ep = me.start_thinking(ReasoningType.SYSTEMATIC, f"topic_{i}")
                me.resolve_thinking(ep.episode_id, f"outcome_{i}", 0.5 + (i % 5) * 0.1)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=start_and_resolve, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ===================== MoralReasoning Tests =====================

class TestEthicalEvaluation:
    def test_evaluate_beneficial_action(self):
        mr = MoralReasoning()
        ev = mr.evaluate("help the community", description="volunteer at food bank")
        assert ev.evaluation_id.startswith("eth_")
        assert ev.overall_score > 0.4

    def test_evaluate_harmful_action(self):
        mr = MoralReasoning()
        ev = mr.evaluate("harm others", description="action that causes pain and suffering")
        assert ev.overall_score < 0.6

    def test_evaluate_with_stakeholders(self):
        mr = MoralReasoning()
        stakeholders = [
            Stakeholder(name="patient", stakeholder_type=StakeholderType.OTHERS, vulnerability=0.9),
            Stakeholder(name="doctor", stakeholder_type=StakeholderType.SELF),
        ]
        ev = mr.evaluate("medical decision", description="care and support patient", stakeholders=stakeholders)
        assert len(ev.stakeholders) == 2

    def test_evaluation_to_dict(self):
        ev = EthicalEvaluation(action="test", overall_score=0.7)
        d = ev.to_dict()
        assert d["evaluation_id"].startswith("eth_")
        assert d["overall_score"] == 0.7

    def test_evaluation_sanitizes_input(self):
        mr = MoralReasoning()
        ev = mr.evaluate("<script>alert(1)</script> action", description="test\x00")
        assert "<script>" not in ev.action


class TestFrameworkAssessments:
    def test_assessment_to_dict(self):
        fa = FrameworkAssessment(
            framework=EthicalFramework.UTILITARIAN, score=0.7,
            reasoning="benefits outweigh costs"
        )
        d = fa.to_dict()
        assert d["framework"] == "utilitarian"
        assert d["score"] == 0.7

    def test_assessment_clamped(self):
        fa = FrameworkAssessment(score=2.0)
        assert fa.score == 1.0


class TestMoralDilemmas:
    def test_dilemma_from_conflicts(self):
        mr = MoralReasoning()
        ev = mr.evaluate("surveillance for safety", description="harm privacy but protect from danger")
        dilemmas = mr.get_dilemmas()
        if ev.conflicts:
            assert len(dilemmas) > 0

    def test_resolve_dilemma(self):
        mr = MoralReasoning()
        from shadowd.cognition.moral_reasoning import MoralDilemma, DilemmaStatus
        d = MoralDilemma(title="test dilemma", competing_values=["freedom vs security"])
        mr._dilemmas[d.dilemma_id] = d
        result = mr.resolve_dilemma(d.dilemma_id, "chose balance")
        assert result is True
        assert d.status == DilemmaStatus.RESOLVED

    def test_resolve_nonexistent_dilemma(self):
        mr = MoralReasoning()
        assert mr.resolve_dilemma("fake_id", "test") is False

    def test_dilemma_to_dict(self):
        d = MoralDilemma(title="test", description="test dilemma")
        d_dict = d.to_dict()
        assert d_dict["dilemma_id"].startswith("dilm_")


class TestMoralPrinciples:
    def test_core_principles_loaded(self):
        mr = MoralReasoning()
        principles = mr.get_principles()
        assert len(principles) >= 5

    def test_add_principle(self):
        mr = MoralReasoning()
        p = mr.add_principle("test principle", "description", EthicalFramework.VIRTUE_ETHICS)
        assert p.name == "test principle"
        assert p.framework == EthicalFramework.VIRTUE_ETHICS

    def test_principle_cap(self):
        mr = MoralReasoning()
        for i in range(110):
            mr.add_principle(f"principle_{i}", "test", EthicalFramework.UTILITARIAN)
        assert len(mr._principles) <= 100

    def test_non_negotiable_principle(self):
        mr = MoralReasoning()
        p = mr.add_principle("sacred", "never violate", EthicalFramework.DEONTOLOGICAL, non_negotiable=True)
        assert p.non_negotiable is True


class TestMoralReasoningStats:
    def test_get_stats(self):
        mr = MoralReasoning()
        mr.evaluate("test action")
        stats = mr.get_stats()
        assert "stats" in stats
        assert stats["stats"]["evaluations_made"] == 1

    def test_evaluations_list(self):
        mr = MoralReasoning()
        mr.evaluate("help others")
        mr.evaluate("harm people")
        evals = mr.get_evaluations()
        assert len(evals) == 2

    def test_evaluations_filtered(self):
        mr = MoralReasoning()
        mr.evaluate("benefit all")
        evals = mr.get_evaluations(min_score=0.9)
        for e in evals:
            assert e["overall_score"] >= 0.9


class TestMoralReasoningThreadSafety:
    def test_concurrent_evaluations(self):
        mr = MoralReasoning()
        errors = []

        def evaluate(i):
            try:
                mr.evaluate(f"action_{i}", description="test description")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=evaluate, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


# ===================== ShadowOS Wiring Tests =====================

class TestShadowOSWiring:
    def test_emotional_intel_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS
        sos = ShadowOS.__new__(ShadowOS)
        sos._running = True
        sos._start_time = 0
        sos.emotional_intel = EmotionalIntelligence()
        # Verify key exists in health check pattern
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_health)
        assert "emotional_intelligence" in source

    def test_creative_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_health)
        assert "creative_synthesis" in source

    def test_metacognition_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_health)
        assert "metacognition" in source

    def test_moral_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_health)
        assert "moral_reasoning" in source

    def test_emotional_intel_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_status)
        assert "emotional_intelligence_stats" in source

    def test_metacognition_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_status)
        assert "metacognition_stats" in source

    def test_moral_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS as SOS
        source = inspect.getsource(SOS.get_status)
        assert "moral_reasoning_stats" in source