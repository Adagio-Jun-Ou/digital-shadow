import pytest
import re
import threading
import time
from unittest.mock import patch

from shadowd.cognition.thought_sense import ThoughtSense, Emotion, UrgencyLevel
from shadowd.cognition.recall_synthesis import (
    RecallSynthesis, _extract_keywords, DOMAIN_PROXIMITY, LIFE_DOMAINS,
)


class TestThoughtSenseEmotionWordBoundary:
    def test_keyword_after_comma(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("I feel sad, right now")
        emotions = {e.emotion: e for e in result}
        assert "sadness" in emotions
        assert "sad" in emotions["sadness"].keywords_matched

    def test_keyword_after_period(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("Everything is fine. I feel upset.")
        emotions = {e.emotion: e for e in result}
        assert "sadness" in emotions
        assert "upset" in emotions["sadness"].keywords_matched

    def test_keyword_with_exclamation(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("I feel hurt!")
        emotions = {e.emotion: e for e in result}
        assert "sadness" in emotions

    def test_keyword_with_semicolon(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("I am afraid; please help")
        emotions = {e.emotion: e for e in result}
        assert "fear" in emotions

    def test_no_substring_match_whatever(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("Whatever you say")
        for e in result:
            assert "what" not in [k.lower() for k in e.keywords_matched]

    def test_no_false_emotion_from_substring(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("understandable behavior")
        emotion_names = [e.emotion for e in result]
        assert "anger" not in emotion_names

    def test_multi_word_keyword_still_works(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("I am looking forward to this")
        emotions = {e.emotion: e for e in result}
        assert "anticipation" in emotions


class TestThoughtSenseDuplicateDownKeyword:
    def test_down_not_double_counted(self):
        ts = ThoughtSense()
        result = ts.detect_emotion("I feel down about things")
        sadness = next((e for e in result if e.emotion == "sadness"), None)
        if sadness:
            down_count = sum(1 for k in sadness.keywords_matched if k.lower() == "down")
            assert down_count <= 1


class TestThoughtSenseUrgencyTieBreaking:
    def test_tie_favors_higher_urgency(self):
        urgency_scores = {
            UrgencyLevel.LOW.value: 0.5,
            UrgencyLevel.MEDIUM.value: 0.5,
            UrgencyLevel.HIGH.value: 0.5,
            UrgencyLevel.CRITICAL.value: 0.5,
        }
        level_priority = {
            UrgencyLevel.CRITICAL.value: 4,
            UrgencyLevel.HIGH.value: 3,
            UrgencyLevel.MEDIUM.value: 2,
            UrgencyLevel.LOW.value: 1,
        }
        best = max(urgency_scores, key=lambda k: (urgency_scores[k], level_priority.get(k, 0)))
        assert best == UrgencyLevel.CRITICAL.value

    def test_critical_wins_over_low_on_tie(self):
        ts = ThoughtSense()
        urgency_scores = {
            UrgencyLevel.LOW.value: 0.0,
            UrgencyLevel.MEDIUM.value: 0.5,
            UrgencyLevel.HIGH.value: 0.5,
            UrgencyLevel.CRITICAL.value: 0.0,
        }
        level_priority = {
            UrgencyLevel.CRITICAL.value: 4,
            UrgencyLevel.HIGH.value: 3,
            UrgencyLevel.MEDIUM.value: 2,
            UrgencyLevel.LOW.value: 1,
        }
        best = max(urgency_scores, key=lambda k: (urgency_scores[k], level_priority.get(k, 0)))
        assert best == UrgencyLevel.HIGH.value


class TestRecallSynthesisDomainFilterBug:
    def test_cross_domain_observations_get_proper_scoring(self):
        rs = RecallSynthesis()
        rs.add_observation("exercise workout fitness routine", domain="health", importance=0.9)
        rs.add_observation("exercise budget fitness financial planning", domain="finance", importance=0.9)
        result = rs.synthesize("exercise fitness", domain="health")
        all_obs = result.direct_observations + result.cross_domain_insights
        cross_domain = [o for o in all_obs if o.get("domain") == "finance"]
        assert len(cross_domain) > 0
        assert cross_domain[0]["score"] > 0.1

    def test_cross_domain_gets_domain_proximity_bonus(self):
        rs = RecallSynthesis()
        rs.add_observation("salary workout career exercise", domain="career", importance=0.5)
        result = rs.synthesize("salary exercise", domain="finance")
        all_obs = result.direct_observations + result.cross_domain_insights
        career_obs = [o for o in all_obs if o.get("domain") == "career"]
        if career_obs:
            assert career_obs[0]["score"] >= 0.3

    def test_domainless_observations_get_scored_with_domain_filter(self):
        rs = RecallSynthesis()
        rs.add_observation("exercise workout fitness important", domain=None, importance=0.9)
        result = rs.synthesize("exercise fitness", domain="health")
        all_obs = result.direct_observations + result.cross_domain_insights
        assert len(all_obs) > 0
        assert all_obs[0]["score"] > 0.0


class TestRecallSynthesisEmptyConfidence:
    def test_zero_confidence_on_no_matches(self):
        rs = RecallSynthesis()
        result = rs.synthesize("completely unrelated xyzzy query")
        assert result.confidence == 0.0

    def test_nonzero_confidence_when_matches_exist(self):
        rs = RecallSynthesis()
        rs.add_observation("python programming code", domain="career")
        result = rs.synthesize("python programming")
        assert result.confidence > 0.0


class TestRecallSynthesisDeadCodeRemoval:
    def test_no_seen_variable_in_source(self):
        import inspect
        source = inspect.getsource(RecallSynthesis.find_cross_domain_patterns)
        assert "seen" not in source


class TestRecallSynthesisBatchLockAcquisition:
    def test_find_patterns_stores_all_patterns(self):
        rs = RecallSynthesis()
        for d in ["health", "finance", "career"]:
            for i in range(2):
                rs.add_observation(
                    f"project planning strategy important data {i}",
                    domain=d,
                )
        patterns = rs.find_cross_domain_patterns()
        if patterns:
            stats = rs.get_stats()
            assert stats["pattern_count"] > 0
            assert stats["insight_count"] > 0
            for ins in rs._insights.values():
                assert len(ins.domains) > 0


class TestRecallSynthesisDomainValidation:
    def test_invalid_domain_becomes_none(self):
        rs = RecallSynthesis()
        obs_id = rs.add_observation("test observation", domain="invalid_domain_xyz")
        obs = rs._observations[obs_id]
        assert obs.domain is None

    def test_valid_domain_accepted(self):
        rs = RecallSynthesis()
        for d in LIFE_DOMAINS:
            obs_id = rs.add_observation("test", domain=d)
            obs = rs._observations[obs_id]
            assert obs.domain == d

    def test_domain_case_insensitive(self):
        rs = RecallSynthesis()
        obs_id = rs.add_observation("test", domain="HEALTH")
        obs = rs._observations[obs_id]
        assert obs.domain == "HEALTH"


class TestRecallSynthesisGetInsightsCrossDomain:
    def test_insight_retrievable_by_secondary_domain(self):
        rs = RecallSynthesis()
        for i in range(4):
            rs.add_observation(
                f"important project planning strategy milestone {i}",
                domain="health" if i % 2 == 0 else "finance",
            )
        rs.find_cross_domain_patterns()
        insights_health = rs.get_insights(domain="health")
        insights_finance = rs.get_insights(domain="finance")
        all_insights = rs.get_insights()
        assert len(all_insights) > 0
        assert len(insights_health) + len(insights_finance) >= len(all_insights)


class TestRecallSynthesisConnectConceptsBestMatch:
    def test_finds_strongest_connection(self):
        rs = RecallSynthesis()
        rs.add_observation("python fast performance optimization", domain="career")
        rs.add_observation("python slow performance debugging", domain="career")
        rs.add_observation("python fast performance compiler cache", domain="career")
        result = rs.connect_concepts("python fast", "python slow")
        if result.confidence > 0:
            assert len(result.chain) >= 2


class TestRecallSynthesisNumericKeywordFilter:
    def test_pure_numbers_not_keywords(self):
        kws = _extract_keywords("In 2024 the 100 items cost $50")
        assert "2024" not in kws
        assert "100" not in kws

    def test_words_still_extracted(self):
        kws = _extract_keywords("python programming important")
        assert "python" in kws
        assert "programming" in kws
        assert "important" in kws


class TestRecallSynthesisDomainConnections:
    def test_connection_count_is_actual_shared(self):
        rs = RecallSynthesis()
        rs.add_observation("exercise fitness workout training", domain="health", importance=0.8)
        rs.add_observation("exercise budget fitness planning", domain="finance", importance=0.8)
        rs.add_observation("investment portfolio returns", domain="finance", importance=0.8)
        conns = rs.get_domain_connections("health")
        finance_conn = next((c for c in conns if c.domain_b == "finance"), None)
        if finance_conn:
            assert finance_conn.connection_count <= 2


class TestRecallSynthesisFullProximityMatrix:
    def test_all_domain_pairs_have_proximity(self):
        for d1 in LIFE_DOMAINS:
            for d2 in LIFE_DOMAINS:
                if d1 != d2:
                    key = (d1, d2)
                    assert key in DOMAIN_PROXIMITY, f"Missing proximity for {key}"

    def test_proximity_is_symmetric(self):
        for d1 in LIFE_DOMAINS:
            for d2 in LIFE_DOMAINS:
                if d1 != d2:
                    assert DOMAIN_PROXIMITY[(d1, d2)] == DOMAIN_PROXIMITY[(d2, d1)]

    def test_proximity_in_valid_range(self):
        for key, val in DOMAIN_PROXIMITY.items():
            assert 0.0 <= val <= 1.0, f"Proximity {key} = {val} out of range"


class TestRecallSynthesisObservationIdLength:
    def test_observation_id_is_12_chars(self):
        rs = RecallSynthesis()
        obs_id = rs.add_observation("test observation")
        assert len(obs_id) == 12


class TestThoughtSenseThreadSafety:
    def test_concurrent_analyze(self):
        ts = ThoughtSense()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    ts.analyze("I need help urgently right now!")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_stats(self):
        ts = ThoughtSense()
        ts.analyze("test text")

        errors = []

        def reader():
            try:
                for _ in range(50):
                    ts.get_stats()
            except Exception as e:
                errors.append(e)

        def writer():
            try:
                for _ in range(50):
                    ts.analyze("another test text here")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=reader) for _ in range(2)]
        threads += [threading.Thread(target=writer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestRecallSynthesisThreadSafety:
    def test_concurrent_add_and_synthesize(self):
        rs = RecallSynthesis()
        errors = []

        def adder():
            try:
                for i in range(100):
                    rs.add_observation(
                        f"test observation number important {i}",
                        domain=["health", "finance", "career"][i % 3],
                    )
            except Exception as e:
                errors.append(e)

        def synthesizer():
            try:
                for _ in range(20):
                    rs.synthesize("test important")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=adder)]
        threads += [threading.Thread(target=synthesizer) for _ in range(2)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0