import copy
import re
import threading
import time
from datetime import datetime, timezone, timedelta
from unittest.mock import MagicMock, patch

import pytest


from shadowd.cognition.emotional_intelligence import (
    EmotionalIntelligence, EmotionCategory, _emotion_keyword_match,
)
from shadowd.cognition.procedural_memory import ProceduralMemory, Routine
from shadowd.cognition.metacognition_engine import (
    MetacognitionEngine, CognitiveState, ReasoningType,
)
from shadowd.cognition.conflict_resolution import ConflictResolution
from shadowd.cognition.cognitive_pipeline import CognitivePipeline, PipelineResult
from shadowd.cognition.shadow_os import ShadowOS


# ═══════════════════════════════════════════════════════════════════
# 1. EmotionalIntelligence — multi-word keyword fix
# ═══════════════════════════════════════════════════════════════════

class TestEmotionKeywordMatch:
    def test_single_word_boundary(self):
        assert _emotion_keyword_match("happy", "I am happy today")

    def test_single_word_no_substring(self):
        assert not _emotion_keyword_match("happy", "I am unhappy today")

    def test_multi_word_match(self):
        assert _emotion_keyword_match("looking forward", "I am looking forward to it")

    def test_multi_word_no_match(self):
        assert not _emotion_keyword_match("looking forward", "looking at the sky")

    def test_single_word_at_start(self):
        assert _emotion_keyword_match("sad", "sad about it")

    def test_single_word_at_end(self):
        assert _emotion_keyword_match("sad", "feeling sad")

    def test_single_word_with_punctuation(self):
        assert _emotion_keyword_match("sad", "I am sad, really")

    def test_multi_word_partial_no_match(self):
        assert not _emotion_keyword_match("looking forward", "forward looking")


class TestDetectEmotionMultiWord:
    def test_looking_forward_detects_anticipation(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am looking forward to the trip")
        assert state.primary == EmotionCategory.ANTICIPATION

    def test_single_word_emotion_still_works(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am happy today")
        assert state.primary == EmotionCategory.JOY

    def test_no_false_substring_match(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am unhappy about this")
        assert state.primary != EmotionCategory.JOY

    def test_eager_keyword_matches(self):
        ei = EmotionalIntelligence()
        state = ei.detect_emotion("I am eager for the results")
        assert state.primary == EmotionCategory.ANTICIPATION


# ═══════════════════════════════════════════════════════════════════
# 2. ProceduralMemory — consistency decay on time gaps
# ═══════════════════════════════════════════════════════════════════

class TestRoutineConsistencyDecay:
    def test_daily_routine_consistency_no_decay(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("morning jog", "daily")
        r = pm.complete_routine(routine.routine_id)
        assert r.consistency >= 0.0

    def test_daily_routine_decay_after_gap(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("morning jog", "daily")
        routine_id = routine.routine_id

        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=5)).isoformat()
        with pm._lock:
            rt = pm._routines[routine_id]
            rt.last_completed = old_ts
            rt.completion_log = [old_ts]
            rt.consistency = 1.0

        r = pm.complete_routine(routine_id)
        assert r.consistency < 1.0

    def test_weekly_routine_no_decay_within_gap(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("weekly review", "weekly")
        routine_id = routine.routine_id

        r = pm.complete_routine(routine_id)
        first_consistency = r.consistency

        now = datetime.now(timezone.utc)
        within_gap_ts = (now - timedelta(days=3)).isoformat()
        with pm._lock:
            rt = pm._routines[routine_id]
            rt.last_completed = within_gap_ts
            rt.completion_log = [within_gap_ts]
            rt.consistency = first_consistency

        r = pm.complete_routine(routine_id)
        assert r.consistency >= first_consistency

    def test_weekly_routine_decay_after_long_gap(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("weekly review", "weekly")
        routine_id = routine.routine_id

        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=30)).isoformat()
        with pm._lock:
            rt = pm._routines[routine_id]
            rt.last_completed = old_ts
            rt.completion_log = [old_ts]
            rt.consistency = 1.0

        r = pm.complete_routine(routine_id)
        assert r.consistency < 1.0

    def test_monthly_routine_within_gap(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("monthly report", "monthly")
        routine_id = routine.routine_id

        r = pm.complete_routine(routine_id)
        first_consistency = r.consistency

        now = datetime.now(timezone.utc)
        within_gap_ts = (now - timedelta(days=20)).isoformat()
        with pm._lock:
            rt = pm._routines[routine_id]
            rt.last_completed = within_gap_ts
            rt.completion_log = [within_gap_ts]
            rt.consistency = first_consistency

        r = pm.complete_routine(routine_id)
        assert r.consistency >= first_consistency

    def test_consistency_capped_at_1(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("daily task", "daily")
        routine_id = routine.routine_id
        now = datetime.now(timezone.utc)
        with pm._lock:
            rt = pm._routines[routine_id]
            for i in range(10):
                ts = (now - timedelta(days=9 - i)).isoformat()
                rt.completion_log.append(ts)
        r = pm.complete_routine(routine_id)
        assert r.consistency <= 1.0

    def test_consistency_never_negative(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("daily task", "daily")
        routine_id = routine.routine_id
        now = datetime.now(timezone.utc)
        old_ts = (now - timedelta(days=100)).isoformat()
        with pm._lock:
            rt = pm._routines[routine_id]
            rt.last_completed = old_ts
            rt.consistency = 0.05
        r = pm.complete_routine(routine_id)
        assert r.consistency >= 0.0


# ═══════════════════════════════════════════════════════════════════
# 3. MetacognitionEngine — _update_cognitive_state stuck states
# ═══════════════════════════════════════════════════════════════════

class TestCognitiveStateTransitions:
    def test_focused_at_low_load(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.1)
        assert me._cognitive_state == CognitiveState.FOCUSED

    def test_analytical_at_medium_load(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.5)
        assert me._cognitive_state == CognitiveState.ANALYTICAL

    def test_distracted_at_high_load(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.7)
        assert me._cognitive_state == CognitiveState.DISTRACTED

    def test_overloaded_at_very_high_load(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.9)
        assert me._cognitive_state == CognitiveState.OVERLOADED

    def test_distracted_recovers_to_focused(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.7)
        assert me._cognitive_state == CognitiveState.DISTRACTED
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.1)
        assert me._cognitive_state == CognitiveState.FOCUSED

    def test_overloaded_recovers_to_focused(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.9)
        assert me._cognitive_state == CognitiveState.OVERLOADED
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.1)
        assert me._cognitive_state == CognitiveState.FOCUSED

    def test_overloaded_to_analytical(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.9)
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.4)
        assert me._cognitive_state == CognitiveState.ANALYTICAL

    def test_distracted_to_analytical(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.7)
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.45)
        assert me._cognitive_state == CognitiveState.ANALYTICAL

    def test_boundary_03(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.3)
        assert me._cognitive_state == CognitiveState.ANALYTICAL

    def test_boundary_06(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.6)
        assert me._cognitive_state == CognitiveState.DISTRACTED

    def test_load_report_reflects_state(self):
        me = MetacognitionEngine()
        me.start_thinking(ReasoningType.SYSTEMATIC, "test", cognitive_load=0.1)
        report = me.get_cognitive_load()
        assert report["state"] == "focused"


# ═══════════════════════════════════════════════════════════════════
# 4. ConflictResolution — single perspective edge case (verify)
# ═══════════════════════════════════════════════════════════════════

class TestConflictResolutionSinglePerspective:
    def test_single_perspective_returns_zero_overlap(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("test", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "position A",
                           interests=["x"], concerns=["y"], flexibility=0.7)
        result = cr.analyze_common_ground(conflict.conflict_id)
        assert result["overlap_score"] == 0.0
        assert result["perspective_count"] == 1

    def test_no_perspectives_returns_zero(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("test", "interest", 2)
        result = cr.analyze_common_ground(conflict.conflict_id)
        assert result["overlap_score"] == 0.0
        assert result["perspective_count"] == 0

    def test_two_perspectives_with_overlap(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("test", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "pos A",
                           interests=["peace", "profit"], concerns=["time"])
        cr.add_perspective(conflict.conflict_id, "Bob", "pos B",
                           interests=["peace", "growth"], concerns=["time", "cost"])
        result = cr.analyze_common_ground(conflict.conflict_id)
        assert result["overlap_score"] > 0.0
        assert "peace" in result["common_interests"]
        assert "time" in result["common_concerns"]
        assert result["perspective_count"] == 2


# ═══════════════════════════════════════════════════════════════════
# 5. CognitivePipeline orchestrator
# ═══════════════════════════════════════════════════════════════════

class TestCognitivePipeline:
    def _make_pipeline(self):
        from shadowd.cognition.thought_sense import ThoughtSense
        from shadowd.cognition.emotional_intelligence import EmotionalIntelligence
        from shadowd.cognition.metacognition_engine import MetacognitionEngine
        from shadowd.cognition.moral_reasoning import MoralReasoning
        return CognitivePipeline(
            thought_sense=ThoughtSense(),
            emotional_intelligence=EmotionalIntelligence(),
            metacognition=MetacognitionEngine(),
            moral_reasoning=MoralReasoning(),
        )

    def test_process_returns_dict(self):
        p = self._make_pipeline()
        result = p.process("I am happy today")
        assert isinstance(result, dict)
        assert "result_id" in result
        assert "modules_used" in result
        assert "overall_confidence" in result

    def test_thought_analysis_in_result(self):
        p = self._make_pipeline()
        result = p.process("Can you help me with this?")
        assert "thought_analysis" in result
        assert result["thought_analysis"] is not None

    def test_emotion_state_in_result(self):
        p = self._make_pipeline()
        result = p.process("I am feeling sad and lonely")
        assert "emotion_state" in result
        assert result["emotion_state"] is not None

    def test_bias_detection_in_result(self):
        p = self._make_pipeline()
        result = p.process("I definitely knew it all along, it's obvious")
        assert "bias_detection" in result
        assert result["bias_detection"] is not None

    def test_moral_evaluation_for_ethical_content(self):
        p = self._make_pipeline()
        result = p.process("Is it fair to take that action?")
        assert "moral_evaluation" in result
        if result["moral_evaluation"] is not None:
            assert "moral_reasoning" in result["modules_used"]

    def test_no_moral_evaluation_for_non_ethical(self):
        p = self._make_pipeline()
        result = p.process("The weather is nice today")
        assert result["moral_evaluation"] is None

    def test_modules_used_populated(self):
        p = self._make_pipeline()
        result = p.process("I am happy")
        assert len(result["modules_used"]) >= 2

    def test_overall_assessment_not_empty(self):
        p = self._make_pipeline()
        result = p.process("I need help with something urgent")
        assert len(result["overall_assessment"]) > 0

    def test_pipeline_stats(self):
        p = self._make_pipeline()
        p.process("hello")
        p.process("world")
        stats = p.get_pipeline_stats()
        assert stats["stats"]["pipelines_run"] == 2

    def test_pipeline_stats_avg_confidence(self):
        p = self._make_pipeline()
        p.process("hello")
        stats = p.get_pipeline_stats()
        assert 0.0 <= stats["stats"]["avg_confidence"] <= 1.0

    def test_pipeline_results_bounded(self):
        p = self._make_pipeline()
        for i in range(1010):
            p.process(f"text {i}")
        stats = p.get_pipeline_stats()
        assert stats["results_stored"] <= 1000

    def test_pipeline_thread_safety(self):
        p = self._make_pipeline()
        results = []
        errors = []

        def worker(text):
            try:
                r = p.process(text)
                results.append(r)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(f"thread text {i}",)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        assert len(results) == 20

    def test_pipeline_with_no_modules(self):
        p = CognitivePipeline()
        result = p.process("hello")
        assert result["modules_used"] == []
        assert result["overall_confidence"] > 0

    def test_input_sanitization(self):
        p = self._make_pipeline()
        result = p.process("<script>alert('xss')</script>Hello")
        assert "\x00" not in result.get("text", "")
        assert "\x7f" not in result.get("text", "")
        assert "Hello" in result.get("text", "")

    def test_context_passed(self):
        p = self._make_pipeline()
        result = p.process("I am sad", context="user context")
        assert result["context"] == "user context"

    def test_ethical_keywords_trigger_moral(self):
        p = self._make_pipeline()
        result = p.process("We should do the right thing")
        assert "moral_reasoning" in result["modules_used"]

    def test_pipeline_result_to_dict(self):
        pr = PipelineResult(text="test", overall_assessment="test assessment")
        d = pr.to_dict()
        assert "result_id" in d
        assert d["text"] == "test"

    def test_pipeline_with_failing_module(self):
        failing = MagicMock()
        failing.analyze.side_effect = RuntimeError("boom")
        p = CognitivePipeline(thought_sense=failing)
        result = p.process("hello")
        assert result["thought_analysis"] is None


# ═══════════════════════════════════════════════════════════════════
# 6. ShadowOS wiring
# ═══════════════════════════════════════════════════════════════════

class TestShadowOSPipelineWiring:
    def test_shadowos_has_pipeline(self):
        os = ShadowOS()
        assert hasattr(os, 'pipeline')
        assert isinstance(os.pipeline, CognitivePipeline)

    def test_pipeline_modules_wired(self):
        os = ShadowOS()
        assert os.pipeline._thought_sense is os.thought_sense
        assert os.pipeline._emotional_intelligence is os.emotional_intel
        assert os.pipeline._metacognition is os.metacognition
        assert os.pipeline._moral_reasoning is os.moral

    def test_pipeline_in_status(self):
        os = ShadowOS()
        status = os.get_status()
        assert "pipeline_stats" in status

    def test_pipeline_in_health(self):
        os = ShadowOS()
        os._running = True
        health = os.get_health()
        assert "cognitive_pipeline" in health["modules"]
        os._running = False


# ═══════════════════════════════════════════════════════════════════
# 7. Integration — full pipeline through ShadowOS
# ═══════════════════════════════════════════════════════════════════

class TestPipelineIntegration:
    def test_full_pipeline_process(self):
        os = ShadowOS()
        result = os.pipeline.process(
            "I am looking forward to the project, but I should consider if it's fair to everyone"
        )
        assert result["thought_analysis"] is not None
        assert result["emotion_state"] is not None
        assert result["bias_detection"] is not None
        assert "moral_reasoning" in result["modules_used"]

    def test_pipeline_after_multiple_runs(self):
        os = ShadowOS()
        os.pipeline.process("hello")
        os.pipeline.process("I am sad")
        stats = os.pipeline.get_pipeline_stats()
        assert stats["stats"]["pipelines_run"] == 2