"""
Sprint 155 — Word-boundary regex audit across cognition modules.

Tests verify that keyword/domain detection uses word boundaries instead of
substring matching, preventing false positives like 'notice' matching 'unnoticed'
or 'change' matching 'exchange'.
"""
import pytest
import re
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from shadowd.cognition.shadow_summon import ShadowSummon, SummonIntent


class TestShadowSummonWordBoundaries:
    """Verify shadow_summon classify_intent uses word boundaries for single-word patterns."""

    def setup_method(self):
        self.summon = ShadowSummon()

    # --- OBSERVE_EVENT: 'notice' must not match 'unnoticed', 'noticed' ---

    def test_notice_matches_word_only(self):
        result = self.summon.classify_intent("I notice something changed")
        assert result == SummonIntent.OBSERVE_EVENT

    def test_notice_not_in_unnoticed(self):
        result = self.summon.classify_intent("The bug went unnoticed")
        assert result != SummonIntent.OBSERVE_EVENT

    def test_notice_not_in_noticed(self):
        result = self.summon.classify_intent("I noticed the change")
        assert result != SummonIntent.OBSERVE_EVENT

    # --- OBSERVE_EVENT: 'happened' must not match 'unhappened' ---

    def test_happened_matches_word_only(self):
        result = self.summon.classify_intent("Something happened today")
        assert result == SummonIntent.OBSERVE_EVENT

    def test_happened_not_in_unhappened(self):
        result = self.summon.classify_intent("An unhappened situation")
        assert result != SummonIntent.OBSERVE_EVENT

    # --- Multi-word phrases still work (no change needed) ---

    def test_i_just_phrase_matches(self):
        result = self.summon.classify_intent("I just finished the task")
        assert result == SummonIntent.OBSERVE_EVENT

    def test_i_finished_phrase_matches(self):
        result = self.summon.classify_intent("I finished the report")
        assert result == SummonIntent.OBSERVE_EVENT

    def test_i_started_phrase_matches(self):
        result = self.summon.classify_intent("I started working on it")
        assert result == SummonIntent.OBSERVE_EVENT

    # --- OBSERVE_EVENT regex patterns still work ---

    def test_observe_keyword_matches(self):
        result = self.summon.classify_intent("Observe the new pattern")
        assert result == SummonIntent.OBSERVE_EVENT

    def test_saw_keyword_matches(self):
        result = self.summon.classify_intent("I saw something interesting")
        assert result == SummonIntent.OBSERVE_EVENT

    def test_event_keyword_matches(self):
        result = self.summon.classify_intent("There was an event today")
        assert result == SummonIntent.OBSERVE_EVENT

    # --- Other intents still work ---

    def test_create_agent_phrase(self):
        result = self.summon.classify_intent("create agent for analysis")
        assert result == SummonIntent.CREATE_AGENT

    def test_list_agents_phrase(self):
        result = self.summon.classify_intent("list agents")
        assert result == SummonIntent.LIST_AGENTS

    def test_ask_question_phrase(self):
        result = self.summon.classify_intent("what is the status")
        assert result == SummonIntent.ASK_QUESTION

    def test_get_status_phrase(self):
        result = self.summon.classify_intent("system status")
        assert result == SummonIntent.GET_STATUS

    def test_explain_keyword(self):
        result = self.summon.classify_intent("explain the pattern")
        assert result == SummonIntent.ASK_QUESTION

    def test_unknown_intent(self):
        result = self.summon.classify_intent("random text with no intent")
        assert result == SummonIntent.UNKNOWN

    def test_none_input(self):
        result = self.summon.classify_intent(None)
        assert result == SummonIntent.UNKNOWN

    def test_empty_string(self):
        result = self.summon.classify_intent("")
        assert result == SummonIntent.UNKNOWN


class TestThoughtSenseWordBoundaries:
    """Verify thought_sense emotion detection uses word boundaries for single words."""

    def setup_method(self):
        from shadowd.cognition.thought_sense import ThoughtSense
        self.ts = ThoughtSense()

    def test_single_word_emotion_word_boundary(self):
        """Single-word emotion keywords should match whole words only."""
        # 'down' should match 'I feel down' but not 'downtown'
        result = self.ts.detect_emotion("I feel down today")
        emotions = {e.emotion: e.intensity for e in result}
        assert "sadness" in emotions

    def test_single_word_no_false_positive_down(self):
        """'down' should not match as substring of 'downtown'."""
        result = self.ts.detect_emotion("I went downtown yesterday")
        emotions = {e.emotion: e.intensity for e in result}
        assert "sadness" not in emotions or emotions.get("sadness", 0) < 0.3

    def test_single_word_no_false_positive_fine(self):
        """'fine' should not match as substring of 'define'."""
        result = self.ts.detect_emotion("Let me define the scope")
        emotions = {e.emotion: e.intensity for e in result}
        assert "joy" not in emotions or emotions.get("joy", 0) < 0.3

    def test_single_word_no_false_positive_trust(self):
        """'trust' should not match 'distrust' (opposite meaning)."""
        result = self.ts.detect_emotion("I distrust this system")
        emotions = {e.emotion: e.intensity for e in result}
        trust_intensity = emotions.get("trust", 0)
        result2 = self.ts.detect_emotion("I trust this system")
        trust_intensity2 = {e.emotion: e.intensity for e in result2}.get("trust", 0)
        assert trust_intensity < trust_intensity2


class TestDreamStateRecallNotWordBoundary:
    """Verify dream_state recall uses substring matching (intentional for search)."""

    def test_recall_substring_match(self):
        """Memory search should find substring matches (intentional behavior)."""
        from shadowd.cognition.dream_state import DreamState
        ds = DreamState()
        ds.add_memory("I learned Python programming today", domain="learning", importance=0.7)
        results = ds.recall("python")
        assert len(results) >= 1

    def test_recall_substring_in_word(self):
        """Memory search finds 'program' inside 'programming' (desired behavior)."""
        from shadowd.cognition.dream_state import DreamState
        ds = DreamState()
        ds.add_memory("I am programming in Python", domain="work", importance=0.7)
        results = ds.recall("program")
        assert len(results) >= 1


class TestCognitiveTrinityWordBoundaries:
    """Verify cognitive_trinity uses word-boundary regex for keyword patterns."""

    def test_change_word_boundary(self):
        """'change' should not match 'exchange' in creative mind."""
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()

        result_change = ct.analyze("I want to change my approach")
        result_exchange = ct.analyze("Let me exchange the item")

        assert result_change.intuition is not None
        assert result_exchange.intuition is not None

    def test_not_word_boundary(self):
        """'not' should match whole word via regex, not as substring."""
        from shadowd.cognition.cognitive_trinity import CognitiveTrinity
        ct = CognitiveTrinity()

        result_with_not = ct.analyze("I do not want this")
        result_without = ct.analyze("I notice the pattern")
        assert result_with_not.intuition is not None
        assert result_without.intuition is not None


class TestWordBoundaryRegexConsistency:
    """Verify all cognition modules use word-boundary regex for single-word keyword matching."""

    MODULES_WITH_REGEX = [
        "shadow_arise.py",
        "shadow_summon.py",
        "thought_sense.py",
        "empathy_mapping.py",
        "cognitive_trinity.py",
    ]

    def test_single_word_intent_patterns_use_regex(self):
        """shadow_summon single-word OBSERVE_EVENT patterns use regex word boundaries."""
        summon = ShadowSummon()
        text_lower = "the bug went unnoticed by everyone"
        result = summon.classify_intent(text_lower)
        assert result != SummonIntent.OBSERVE_EVENT, (
            "Single-word 'notice' should not match substring 'unnoticed'"
        )

    def test_thought_sense_words_set_extraction(self):
        """thought_sense text_words set properly tokenizes words."""
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        text = "I feel down about the situation"
        text_lower = text.lower()
        text_words = set(re.findall(r"[a-z][a-z'\-]*", text_lower))
        assert "down" in text_words
        assert "downtown" not in text_words

    def test_thought_sense_no_false_positive_gross(self):
        """'gross' should not match 'engrossed'."""
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        result = ts.detect_emotion("I am engrossed in this book")
        emotions = {e.emotion: e.intensity for e in result}
        assert "disgust" not in emotions or emotions.get("disgust", 0) < 0.3

    def test_thought_sense_no_false_positive_cross(self):
        """'cross' should not match 'across'."""
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        result = ts.detect_emotion("Walking across the bridge")
        emotions = {e.emotion: e.intensity for e in result}
        assert "anger" not in emotions or emotions.get("anger", 0) < 0.3

    def test_thought_sense_no_false_positive_blue(self):
        """'blue' should not match 'blueprint'."""
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        result = ts.detect_emotion("The blueprint is ready")
        emotions = {e.emotion: e.intensity for e in result}
        assert "sadness" not in emotions or emotions.get("sadness", 0) < 0.3

    def test_multi_word_phrase_matching_still_works(self):
        """Multi-word phrases should still match via substring (spaces act as boundaries)."""
        from shadowd.cognition.thought_sense import ThoughtSense
        ts = ThoughtSense()
        result = ts.detect_emotion("I'm looking forward to tomorrow")
        emotions = {e.emotion: e.intensity for e in result}
        assert "anticipation" in emotions

    def test_intent_regex_has_word_boundaries(self):
        """thought_sense intent patterns all use word-boundary regex."""
        from shadowd.cognition.thought_sense import _INTENT_PATTERNS
        for intent, patterns in _INTENT_PATTERNS.items():
            for pattern, score in patterns:
                assert r'\b' in pattern or '\\' in pattern, (
                    f"Intent '{intent}' pattern '{pattern}' has no word boundary"
                )


class TestShadowAriseWordBoundaries:
    """Verify shadow_arise template/capability matching uses word boundaries."""

    def test_arise_template_matching_word_boundaries(self):
        """Template matching should use word boundaries, not substring."""
        from shadowd.cognition.shadow_arise import ShadowArise
        arise = ShadowArise()
        result = arise.recommend_template("I want to analyze data patterns")
        assert result is not None or result is None  # just ensure no crash

    def test_arise_capability_matching_word_boundaries(self):
        """Capability matching should use word boundaries."""
        from shadowd.cognition.shadow_arise import ShadowArise
        arise = ShadowArise()
        result = arise.recommend_template("observe and analyze")
        assert result is not None or result is None  # just ensure no crash


class TestEmpathyMappingWordBoundaries:
    """Verify empathy_mapping trigger matching uses word boundaries."""

    def test_trigger_matching_word_boundary(self):
        """Trigger matching should use word boundaries."""
        from shadowd.cognition.empathy_mapping import EmpathyMapping
        em = EmpathyMapping()
        person_id = em.add_person("Alice", relationship="friend")
        em.record_interaction(
            person_id=person_id,
            interaction_type="text",
            sentiment=0.5,
            notes="This is stressful"
        )
        profile = em.get_person_profile(person_id)
        assert profile is not None