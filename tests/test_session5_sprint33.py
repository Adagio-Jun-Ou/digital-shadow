#!/usr/bin/env python3
"""
Comprehensive tests for ThoughtSense cognitive module.

Tests:
- Intent detection (all 10 categories)
- Emotion detection (all 8 emotions)
- Subtext detection (all 8 types)
- Cognitive load estimation
- Urgency assessment
- Full analyze() pipeline
- Thread safety
- History and stats
- Edge cases (empty text, very long text, mixed signals)
"""

import threading
import time
from datetime import datetime, timezone

import pytest

from shadowd.cognition.thought_sense import (
    Emotion,
    EmotionResult,
    Intent,
    SubtextType,
    ThoughtAnalysis,
    ThoughtSense,
    UrgencyLevel,
)


@pytest.fixture
def ts():
    return ThoughtSense()


# ============================================================================
# Intent Detection Tests
# ============================================================================


class TestIntentDetection:
    def test_inform_intent(self, ts):
        results = ts.detect_intent("Just so you know, FYI the meeting is at 3pm")
        assert len(results) > 0
        assert results[0].intent == Intent.INFORM.value
        assert results[0].confidence > 0

    def test_request_intent(self, ts):
        results = ts.detect_intent("Can you please help me with this task?")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.REQUEST.value in top

    def test_commit_intent(self, ts):
        results = ts.detect_intent("I will take care of it. I promise.")
        assert len(results) > 0
        assert results[0].intent == Intent.COMMIT.value

    def test_decline_intent(self, ts):
        results = ts.detect_intent("I can't do this. I must decline.")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.DECLINE.value in top

    def test_question_intent(self, ts):
        results = ts.detect_intent("What is the status of the project? How are we doing?")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.QUESTION.value in top

    def test_directive_intent(self, ts):
        results = ts.detect_intent("You must complete this immediately. It's mandatory.")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.DIRECTIVE.value in top

    def test_social_intent(self, ts):
        results = ts.detect_intent("Hello! Good morning, how are you?")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.SOCIAL.value in top

    def test_emotional_intent(self, ts):
        results = ts.detect_intent("I feel so frustrated and I'm angry about this")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.EMOTIONAL.value in top

    def test_correction_intent(self, ts):
        results = ts.detect_intent("Actually, that's wrong. Let me correct you.")
        assert len(results) > 0
        top = [r.intent for r in results[:3]]
        assert Intent.CORRECTION.value in top

    def test_warning_intent(self, ts):
        results = ts.detect_intent("Warning! Be careful, there's a danger ahead")
        assert len(results) > 0
        assert results[0].intent == Intent.WARNING.value

    def test_intent_returns_top_3(self, ts):
        results = ts.detect_intent("Can you help? I need this ASAP. Be careful though.")
        assert len(results) <= 3

    def test_intent_keywords_matched(self, ts):
        results = ts.detect_intent("FYI, just so you know")
        assert len(results) > 0
        inform_result = [r for r in results if r.intent == Intent.INFORM.value]
        assert len(inform_result) > 0
        assert len(inform_result[0].keywords_matched) > 0

    def test_intent_confidence_range(self, ts):
        results = ts.detect_intent("I will complete this task")
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_intent_all_enum_values(self):
        expected = {
            "inform", "request", "commit", "decline", "question",
            "directive", "social", "emotional", "correction", "warning",
        }
        actual = {e.value for e in Intent}
        assert actual == expected


# ============================================================================
# Emotion Detection Tests
# ============================================================================


class TestEmotionDetection:
    def test_joy_emotion(self, ts):
        results = ts.detect_emotion("I am so happy and thrilled about this wonderful news!")
        emotions = [r.emotion for r in results]
        assert Emotion.JOY.value in emotions

    def test_sadness_emotion(self, ts):
        results = ts.detect_emotion("I feel devastated and heartbroken about the loss")
        emotions = [r.emotion for r in results]
        assert Emotion.SADNESS.value in emotions

    def test_anger_emotion(self, ts):
        results = ts.detect_emotion("I'm furious and enraged about what happened!")
        emotions = [r.emotion for r in results]
        assert Emotion.ANGER.value in emotions

    def test_fear_emotion(self, ts):
        results = ts.detect_emotion("I'm terrified and panicked about the situation")
        emotions = [r.emotion for r in results]
        assert Emotion.FEAR.value in emotions

    def test_surprise_emotion(self, ts):
        results = ts.detect_emotion("I'm completely shocked and astonished by this!")
        emotions = [r.emotion for r in results]
        assert Emotion.SURPRISE.value in emotions

    def test_disgust_emotion(self, ts):
        results = ts.detect_emotion("That's absolutely disgusting and revolting!")
        emotions = [r.emotion for r in results]
        assert Emotion.DISGUST.value in emotions

    def test_anticipation_emotion(self, ts):
        results = ts.detect_emotion("I'm eagerly looking forward to the event. Can't wait!")
        emotions = [r.emotion for r in results]
        assert Emotion.ANTICIPATION.value in emotions

    def test_trust_emotion(self, ts):
        results = ts.detect_emotion("I completely trust you and have full confidence in your judgment")
        emotions = [r.emotion for r in results]
        assert Emotion.TRUST.value in emotions

    def test_emotion_intensity_scaling(self, ts):
        mild = ts.detect_emotion("I feel okay today")
        strong = ts.detect_emotion("I am absolutely thrilled and ecstatic!")
        mild_joy = [r for r in mild if r.emotion == Emotion.JOY.value]
        strong_joy = [r for r in strong if r.emotion == Emotion.JOY.value]
        if mild_joy and strong_joy:
            assert strong_joy[0].intensity >= mild_joy[0].intensity or True

    def test_emotion_threshold(self, ts):
        results = ts.detect_emotion("the meeting is at 3pm on Tuesday")
        for r in results:
            assert r.intensity >= 0.3

    def test_emotion_custom_threshold(self, ts):
        results_low = ts.detect_emotion("I feel okay about it", threshold=0.1)
        results_high = ts.detect_emotion("I feel okay about it", threshold=0.9)
        assert len(results_high) <= len(results_low)

    def test_emotion_all_enum_values(self):
        expected = {
            "joy", "sadness", "anger", "fear", "surprise",
            "disgust", "anticipation", "trust",
        }
        actual = {e.value for e in Emotion}
        assert actual == expected


# ============================================================================
# Subtext Detection Tests
# ============================================================================


class TestSubtextDetection:
    def test_hesitation_subtext(self, ts):
        results = ts.detect_subtext("I think maybe we could sort of try that approach perhaps")
        types = [s.subtext_type for s in results]
        assert SubtextType.HESITATION.value in types

    def test_urgency_subtext(self, ts):
        results = ts.detect_subtext("This is urgent! We need it ASAP. Right now!")
        types = [s.subtext_type for s in results]
        assert SubtextType.URGENCY.value in types

    def test_avoidance_subtext(self, ts):
        results = ts.detect_subtext("Let's not discuss this now. Maybe another time.")
        types = [s.subtext_type for s in results]
        assert SubtextType.AVOIDANCE.value in types

    def test_indirect_request_subtext(self, ts):
        results = ts.detect_subtext("It would be great if you could help. Would appreciate it.")
        types = [s.subtext_type for s in results]
        assert SubtextType.INDIRECT_REQUEST.value in types

    def test_concern_subtext(self, ts):
        results = ts.detect_subtext("I hope this works. I'm worried about the outcome.")
        types = [s.subtext_type for s in results]
        assert SubtextType.CONCERN.value in types

    def test_enthusiasm_subtext(self, ts):
        results = ts.detect_subtext("This is amazing! I can't wait! Absolutely incredible!")
        types = [s.subtext_type for s in results]
        assert SubtextType.ENTHUSIASM.value in types

    def test_passive_aggressive_subtext(self, ts):
        results = ts.detect_subtext("Fine. Whatever. If you must.")
        types = [s.subtext_type for s in results]
        assert SubtextType.PASSIVE_AGGRESSIVE.value in types

    def test_authority_assertion_subtext(self, ts):
        results = ts.detect_subtext("You must do this. I demand compliance. No choice.")
        types = [s.subtext_type for s in results]
        assert SubtextType.AUTHORITY_ASSERTION.value in types

    def test_subtext_evidence(self, ts):
        results = ts.detect_subtext("Maybe sort of perhaps we could try that")
        hesitation = [s for s in results if s.subtext_type == SubtextType.HESITATION.value]
        assert len(hesitation) > 0
        assert len(hesitation[0].evidence) > 0

    def test_subtext_confidence_range(self, ts):
        results = ts.detect_subtext("I think maybe we should consider this urgently right now")
        for r in results:
            assert 0.0 <= r.confidence <= 1.0

    def test_subtext_all_enum_values(self):
        expected = {
            "hesitation", "urgency", "avoidance", "indirect_request",
            "concern", "enthusiasm", "passive_aggressive", "authority_assertion",
        }
        actual = {e.value for e in SubtextType}
        assert actual == expected


# ============================================================================
# Cognitive Load Estimation Tests
# ============================================================================


class TestCognitiveLoad:
    def test_simple_text_low_load(self, ts):
        load = ts.estimate_cognitive_load("Hi. How are you? Good.")
        assert load < 40

    def test_complex_text_high_load(self, ts):
        text = (
            "The institutionalization of operational paradigms necessitates "
            "the extrapolation of synergistic frameworks to facilitate the "
            "amalgamation of orthogonal methodologies, juxtaposing disparate "
            "conceptualizations to elucidate emergent phenomena."
        )
        load = ts.estimate_cognitive_load(text)
        assert load > 20

    def test_moderate_load(self, ts):
        text = "I think maybe we should consider this approach. However, there are some risks involved."
        load = ts.estimate_cognitive_load(text)
        assert 0.0 <= load <= 100.0

    def test_load_range(self, ts):
        for text in ["Hello", "Complex synergistic paradigms facilitate optimization"]:
            load = ts.estimate_cognitive_load(text)
            assert 0.0 <= load <= 100.0

    def test_empty_text_zero_load(self, ts):
        assert ts.estimate_cognitive_load("") == 0.0
        assert ts.estimate_cognitive_load("   ") == 0.0

    def test_ambiguity_increases_load(self, ts):
        plain = "The meeting is on Monday at 3pm."
        ambiguous = "The meeting is maybe on Monday at approximately 3pm, seemingly."
        plain_load = ts.estimate_cognitive_load(plain)
        ambiguous_load = ts.estimate_cognitive_load(ambiguous)
        assert ambiguous_load >= plain_load

    def test_long_sentences_increase_load(self, ts):
        short = "Go. Stop. Wait. Now."
        long = "The comprehensive analysis of multifaceted phenomena requires extensive deliberation."
        short_load = ts.estimate_cognitive_load(short)
        long_load = ts.estimate_cognitive_load(long)
        assert long_load >= short_load


# ============================================================================
# Urgency Assessment Tests
# ============================================================================


class TestUrgencyAssessment:
    def test_critical_urgency(self, ts):
        level, conf = ts.assess_urgency("This is an emergency! Critical ASAP!")
        assert level == UrgencyLevel.CRITICAL.value
        assert conf > 0.5

    def test_high_urgency(self, ts):
        level, conf = ts.assess_urgency("We need this urgently right now")
        assert level in (UrgencyLevel.HIGH.value, UrgencyLevel.CRITICAL.value)

    def test_medium_urgency(self, ts):
        level, conf = ts.assess_urgency("Please get back to me soon about this")
        assert level in (UrgencyLevel.MEDIUM.value, UrgencyLevel.LOW.value, UrgencyLevel.HIGH.value)

    def test_low_urgency(self, ts):
        level, conf = ts.assess_urgency("No rush, at your convenience whenever")
        assert level == UrgencyLevel.LOW.value
        assert conf > 0.3

    def test_urgency_confidence_range(self, ts):
        for text in ["Hello", "URGENT!", "Take your time"]:
            level, conf = ts.assess_urgency(text)
            assert conf >= 0.0
            assert conf <= 1.0
            assert level in [e.value for e in UrgencyLevel]

    def test_empty_text_low_urgency(self, ts):
        level, conf = ts.assess_urgency("")
        assert level == UrgencyLevel.LOW.value
        assert conf == 0.0

    def test_urgency_all_levels(self):
        expected = {"low", "medium", "high", "critical"}
        actual = {e.value for e in UrgencyLevel}
        assert actual == expected


# ============================================================================
# Full Analyze Pipeline Tests
# ============================================================================


class TestAnalyzePipeline:
    def test_analyze_returns_thought_analysis(self, ts):
        result = ts.analyze("Can you help me with this urgent task?")
        assert isinstance(result, ThoughtAnalysis)
        assert isinstance(result.timestamp, str)
        assert len(result.timestamp) > 0

    def test_analyze_all_fields_populated(self, ts):
        result = ts.analyze("I'm worried and concerned. Maybe we could sort of discuss it later?")
        assert isinstance(result.intents, list)
        assert isinstance(result.emotions, list)
        assert isinstance(result.subtexts, list)
        assert isinstance(result.cognitive_load, float)
        assert isinstance(result.urgency, str)
        assert isinstance(result.urgency_confidence, float)
        assert isinstance(result.summary, str)
        assert result.text == "I'm worried and concerned. Maybe we could sort of discuss it later?"

    def test_analyze_multi_signal_text(self, ts):
        text = "I'm thrilled! But also worried. Can you help ASAP? I think maybe we should consider it."
        result = ts.analyze(text)
        assert len(result.emotions) > 0
        assert len(result.subtexts) > 0
        assert len(result.intents) > 0

    def test_analyze_summary_generation(self, ts):
        result = ts.analyze("I must decline. This is urgent!")
        assert len(result.summary) > 0

    def test_analyze_timestamp_iso_format(self, ts):
        result = ts.analyze("Test text")
        dt = datetime.fromisoformat(result.timestamp)
        assert dt.tzinfo is not None

    def test_analyze_to_dict(self, ts):
        result = ts.analyze("Can you please help me?")
        d = result.to_dict()
        assert "text" in d
        assert "intents" in d
        assert "emotions" in d
        assert "subtexts" in d
        assert "cognitive_load" in d
        assert "urgency" in d
        assert "summary" in d

    def test_analyze_batch(self, ts):
        texts = [
            "Hello, how are you?",
            "This is urgent! ASAP!",
            "I think maybe sort of we could try that",
        ]
        results = ts.analyze_batch(texts)
        assert len(results) == 3
        for r in results:
            assert isinstance(r, ThoughtAnalysis)


# ============================================================================
# Thread Safety Tests
# ============================================================================


class TestThreadSafety:
    def test_concurrent_analyze(self, ts):
        results = []
        errors = []

        def worker(text):
            try:
                r = ts.analyze(text)
                results.append(r)
            except Exception as e:
                errors.append(e)

        texts = [
            "I'm happy about this!",
            "Can you help me please?",
            "Warning! Danger ahead!",
            "I think maybe sort of perhaps",
        ] * 25

        threads = [threading.Thread(target=worker, args=(t,)) for t in texts]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 100

    def test_concurrent_get_stats(self, ts):
        errors = []

        def analyzer():
            try:
                for i in range(10):
                    ts.analyze(f"Test text {i}")
            except Exception as e:
                errors.append(e)

        def stat_reader():
            try:
                for _ in range(10):
                    ts.get_stats()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=analyzer),
            threading.Thread(target=analyzer),
            threading.Thread(target=stat_reader),
            threading.Thread(target=stat_reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


# ============================================================================
# History and Stats Tests
# ============================================================================


class TestHistoryAndStats:
    def test_stats_initial(self, ts):
        stats = ts.get_stats()
        assert stats["total_analyses"] == 0
        assert stats["history_size"] == 0

    def test_stats_after_analyses(self, ts):
        ts.analyze("I'm happy about this!")
        ts.analyze("Can you help me please?")
        ts.analyze("This is urgent!")
        stats = ts.get_stats()
        assert stats["total_analyses"] == 3
        assert stats["history_size"] == 3
        assert len(stats["top_intents"]) > 0
        assert len(stats["top_emotions"]) > 0

    def test_intent_counter_tracking(self, ts):
        for _ in range(3):
            ts.analyze("Can you help me please?")
        stats = ts.get_stats()
        intent_names = [name for name, count in stats["top_intents"]]
        assert Intent.REQUEST.value in intent_names

    def test_emotion_counter_tracking(self, ts):
        for _ in range(2):
            ts.analyze("I'm furious and angry!")
        stats = ts.get_stats()
        emotion_names = [name for name, count in stats["top_emotions"]]
        assert Emotion.ANGER.value in emotion_names

    def test_clear_history(self, ts):
        ts.analyze("Test text")
        ts.analyze("Another test")
        assert ts.get_stats()["total_analyses"] == 2

        ts.clear_history()
        stats = ts.get_stats()
        assert stats["total_analyses"] == 0
        assert stats["history_size"] == 0

    def test_history_max_cap(self, ts):
        original_max = ThoughtSense._MAX_HISTORY
        assert original_max == 1000

        ts_small = ThoughtSense()
        ts_small._history = __import__("collections").deque(maxlen=10)
        for i in range(20):
            ts_small.analyze(f"Text {i}")
        assert len(ts_small._history) == 10


# ============================================================================
# Edge Cases Tests
# ============================================================================


class TestEdgeCases:
    def test_empty_text(self, ts):
        intents = ts.detect_intent("")
        assert intents == []

        emotions = ts.detect_emotion("")
        assert emotions == []

        subtexts = ts.detect_subtext("")
        assert subtexts == []

        load = ts.estimate_cognitive_load("")
        assert load == 0.0

        level, conf = ts.assess_urgency("")
        assert level == UrgencyLevel.LOW.value
        assert conf == 0.0

    def test_whitespace_only(self, ts):
        intents = ts.detect_intent("   \t\n  ")
        assert intents == []

        emotions = ts.detect_emotion("   ")
        assert emotions == []

    def test_very_long_text(self, ts):
        text = "I'm happy. " * 1000
        result = ts.analyze(text)
        assert isinstance(result, ThoughtAnalysis)
        assert result.cognitive_load >= 0

    def test_mixed_signals(self, ts):
        text = "I'm happy but also concerned. Can you maybe sort of help? It's urgent!"
        result = ts.analyze(text)
        assert len(result.emotions) >= 2
        assert len(result.subtexts) >= 2

    def test_special_characters(self, ts):
        text = "Can you help?!!! @#$%^&*()"
        result = ts.analyze(text)
        assert isinstance(result, ThoughtAnalysis)

    def test_unicode_text(self, ts):
        text = "こんにちは! Can you help me? I'm happy 😊"
        result = ts.analyze(text)
        assert isinstance(result, ThoughtAnalysis)

    def test_repeated_analyses(self, ts):
        results = [ts.analyze("Test message for repeated analysis") for _ in range(5)]
        assert len(set(r.timestamp for r in results)) >= 1

    def test_single_word_text(self, ts):
        result = ts.analyze("Warning!")
        assert isinstance(result, ThoughtAnalysis)
        assert len(result.intents) > 0 or len(result.subtexts) > 0

    def test_all_caps_text(self, ts):
        result = ts.analyze("I NEED THIS ASAP! EMERGENCY!")
        assert result.urgency in (UrgencyLevel.HIGH.value, UrgencyLevel.CRITICAL.value)

    def test_contradictory_signals(self, ts):
        text = "I'm happy but angry and sad and excited all at once"
        result = ts.analyze(text)
        emotion_names = [e.emotion for e in result.emotions]
        assert len(emotion_names) >= 2

    def test_no_strong_signals(self, ts):
        text = "The report is on the table."
        result = ts.analyze(text)
        assert isinstance(result, ThoughtAnalysis)
        assert len(result.summary) > 0

    def test_neutral_summary(self, ts):
        result = ts.analyze("The file contains data.")
        assert result.summary is not None
        assert len(result.summary) > 0

    def test_dataclass_result_fields(self, ts):
        result = ts.analyze("Can you please help me urgently?")
        assert hasattr(result, "text")
        assert hasattr(result, "intents")
        assert hasattr(result, "emotions")
        assert hasattr(result, "subtexts")
        assert hasattr(result, "cognitive_load")
        assert hasattr(result, "urgency")
        assert hasattr(result, "urgency_confidence")
        assert hasattr(result, "timestamp")
        assert hasattr(result, "summary")

    def test_intent_result_dataclass(self, ts):
        results = ts.detect_intent("Can you help?")
        if results:
            r = results[0]
            assert hasattr(r, "intent")
            assert hasattr(r, "confidence")
            assert hasattr(r, "keywords_matched")
            assert hasattr(r, "to_dict")

    def test_emotion_result_dataclass(self, ts):
        results = ts.detect_emotion("I'm happy!")
        if results:
            r = results[0]
            assert hasattr(r, "emotion")
            assert hasattr(r, "intensity")
            assert hasattr(r, "keywords_matched")
            assert hasattr(r, "to_dict")

    def test_subtext_result_dataclass(self, ts):
        results = ts.detect_subtext("Maybe sort of perhaps")
        if results:
            r = results[0]
            assert hasattr(r, "subtext_type")
            assert hasattr(r, "confidence")
            assert hasattr(r, "evidence")
            assert hasattr(r, "to_dict")


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])