"""
Tests for RecallSynthesis - Cross-Domain Pattern Synthesis Engine
"""

import threading
import time
from datetime import datetime, timezone, timedelta

import pytest

from shadowd.cognition.recall_synthesis import (
    RecallSynthesis,
    Observation,
    CrossDomainPattern,
    SynthesisResult,
    Insight,
    ConnectionResult,
    DomainConnection,
    _extract_keywords,
    LIFE_DOMAINS,
    DOMAIN_PROXIMITY,
    STOP_WORDS,
)


class TestKeywordExtraction:
    def test_basic_keywords(self):
        kws = _extract_keywords("Running every morning improved my cardiovascular health")
        assert "running" in kws
        assert "morning" in kws
        assert "improved" in kws
        assert "cardiovascular" in kws
        assert "health" in kws

    def test_stops_out_short_words(self):
        kws = _extract_keywords("This is a test of the system")
        assert "this" not in kws
        assert "test" in kws
        assert "system" in kws

    def test_capitalized_short_words_kept(self):
        kws = _extract_keywords("NASA launched a new satellite")
        assert "nasa" in kws

    def test_empty_string(self):
        assert _extract_keywords("") == []

    def test_all_stop_words(self):
        kws = _extract_keywords("the a an is was are")
        assert kws == []

    def test_punctuation_stripped(self):
        kws = _extract_keywords("Sleep, diet, and exercise!")
        assert "sleep" in kws
        assert "diet" in kws
        assert "exercise" in kws

    def test_mixed_case(self):
        kws = _extract_keywords("Python python PYTHON")
        assert kws.count("python") >= 1

    def test_long_text(self):
        text = " ".join(["investment"] * 20)
        kws = _extract_keywords(text)
        assert "investment" in kws

    def test_numbers_included_via_length(self):
        kws = _extract_keywords("portfolio2024 grew substantially")
        assert "portfolio2024" in kws
        assert "substantially" in kws

    def test_four_char_words_kept(self):
        kws = _extract_keywords("work life goal plan")
        assert "work" in kws
        assert "life" in kws
        assert "goal" in kws
        assert "plan" in kws


class TestAddObservation:
    def test_basic_add(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Started meditating every morning")
        assert isinstance(oid, str)
        assert len(oid) == 12

    def test_add_with_domain(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Got a promotion", domain="career")
        stats = rs.get_stats()
        assert stats["observation_count"] == 1
        assert stats["domain_distribution"]["career"] == 1

    def test_add_with_timestamp(self):
        rs = RecallSynthesis()
        ts = "2024-06-15T10:00:00+00:00"
        oid = rs.add_observation("Invested in index funds", domain="finance", timestamp=ts)
        obs = rs._observations[oid]
        assert obs.timestamp == ts

    def test_add_with_importance(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Major health scare", domain="health", importance=0.9)
        obs = rs._observations[oid]
        assert obs.importance == 0.9

    def test_importance_clamped_high(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Test", importance=1.5)
        obs = rs._observations[oid]
        assert obs.importance == 1.0

    def test_importance_clamped_low(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Test", importance=-0.5)
        obs = rs._observations[oid]
        assert obs.importance == 0.0

    def test_auto_timestamp(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Auto timestamp test")
        obs = rs._observations[oid]
        assert obs.timestamp is not None
        assert "T" in obs.timestamp

    def test_default_importance(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Default importance")
        obs = rs._observations[oid]
        assert obs.importance == 0.5

    def test_keywords_extracted(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Running marathon training schedule")
        obs = rs._observations[oid]
        assert "running" in obs.keywords
        assert "marathon" in obs.keywords

    def test_fifo_eviction(self):
        rs = RecallSynthesis()
        rs._MAX_OBSERVATIONS = 5
        ids = []
        for i in range(7):
            ids.append(rs.add_observation(f"Observation {i}", domain="health"))
        stats = rs.get_stats()
        assert stats["observation_count"] == 5
        assert ids[0] not in rs._observations
        assert ids[1] not in rs._observations
        assert ids[6] in rs._observations

    def test_none_domain(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("No domain observation")
        obs = rs._observations[oid]
        assert obs.domain is None


class TestFindCrossDomainPatterns:
    def test_no_patterns_with_few_observations(self):
        rs = RecallSynthesis()
        rs.add_observation("Just one observation", domain="health")
        patterns = rs.find_cross_domain_patterns()
        assert patterns == []

    def test_no_patterns_single_domain(self):
        rs = RecallSynthesis()
        for i in range(5):
            rs.add_observation(f"Health exercise fitness running training {i}", domain="health")
        patterns = rs.find_cross_domain_patterns()
        assert patterns == []

    def test_detects_cross_domain_pattern(self):
        rs = RecallSynthesis()
        for domain in ["health", "career", "finance"]:
            for i in range(3):
                rs.add_observation(
                    f"Stress management strategy planning approach {i}",
                    domain=domain,
                )
        patterns = rs.find_cross_domain_patterns()
        assert len(patterns) > 0
        pattern = patterns[0]
        assert len(pattern.domains_involved) >= 2
        assert len(pattern.observations_linked) >= 3
        assert pattern.confidence > 0

    def test_pattern_fields(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "career"]:
            for i in range(4):
                rs.add_observation(
                    f"Sleep productivity performance stress {i}",
                    domain=domain,
                )
        patterns = rs.find_cross_domain_patterns()
        assert len(patterns) > 0
        p = patterns[0]
        assert p.pattern_id is not None
        assert isinstance(p.keywords, list)
        assert isinstance(p.domains_involved, list)
        assert isinstance(p.observations_linked, list)
        assert 0 <= p.confidence <= 1
        assert isinstance(p.description, str)

    def test_pattern_to_dict(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "career"]:
            for i in range(3):
                rs.add_observation(
                    f"Investment strategy growth planning {i}",
                    domain=domain,
                )
        patterns = rs.find_cross_domain_patterns()
        d = patterns[0].to_dict()
        assert "pattern_id" in d
        assert "keywords" in d
        assert "confidence" in d

    def test_pattern_cap(self):
        rs = RecallSynthesis()
        rs._MAX_PATTERNS = 2
        for batch in range(5):
            for domain in ["health", "finance", "career"]:
                for i in range(3):
                    rs.add_observation(
                        f"Pattern topic {batch} keyword{batch} connection {i}",
                        domain=domain,
                    )
        rs.find_cross_domain_patterns()
        assert len(rs._patterns) <= 2


class TestSynthesize:
    def test_basic_synthesize(self):
        rs = RecallSynthesis()
        rs.add_observation("Running improves cardiovascular health", domain="health")
        rs.add_observation("Morning exercise routine", domain="health")
        rs.add_observation("Invested in fitness company stock", domain="finance")
        result = rs.synthesize("exercise health running")
        assert isinstance(result, SynthesisResult)
        assert result.query == "exercise health running"
        assert isinstance(result.direct_observations, list)
        assert isinstance(result.cross_domain_insights, list)

    def test_synthesize_with_domain_filter(self):
        rs = RecallSynthesis()
        rs.add_observation("Running improves cardiovascular health", domain="health")
        rs.add_observation("Fitness tracker data analysis", domain="career")
        result = rs.synthesize("running health", domain="health")
        direct_domains = [o.get("domain") for o in result.direct_observations if o.get("domain")]
        assert all(d == "health" for d in direct_domains)

    def test_synthesize_confidence_range(self):
        rs = RecallSynthesis()
        rs.add_observation("Some observation about Python programming", domain="career")
        result = rs.synthesize("Python programming")
        assert 0 <= result.confidence <= 1

    def test_synthesize_empty_query(self):
        rs = RecallSynthesis()
        rs.add_observation("Random observation")
        result = rs.synthesize("")
        assert isinstance(result, SynthesisResult)

    def test_synthesize_no_matches(self):
        rs = RecallSynthesis()
        rs.add_observation("Cooking dinner recipe pasta", domain="home", importance=0.0)
        result = rs.synthesize("quantum physics string theory")
        if result.direct_observations:
            assert result.direct_observations[0]["from_keywords"] if "from_keywords" in result.direct_observations[0] else True
            assert result.confidence < 0.15

    def test_synthesize_to_dict(self):
        rs = RecallSynthesis()
        rs.add_observation("Test observation about algorithms", domain="career")
        result = rs.synthesize("algorithms")
        d = result.to_dict()
        assert "query" in d
        assert "confidence" in d


class TestGetInsights:
    def test_insights_from_patterns(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "social"]:
            for i in range(4):
                rs.add_observation(
                    f"Sleep productivity performance improvement {i}",
                    domain=domain,
                )
        rs.find_cross_domain_patterns()
        insights = rs.get_insights()
        assert len(insights) > 0

    def test_filter_by_domain(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "social"]:
            for i in range(3):
                rs.add_observation(
                    f"Connection pattern stress management {i}",
                    domain=domain,
                )
        rs.find_cross_domain_patterns()
        health_insights = rs.get_insights(domain="health")
        for ins in health_insights:
            assert ins.domain == "health" or "health" in ins.domains

    def test_filter_by_confidence(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "career"]:
            for i in range(4):
                rs.add_observation(
                    f"Investment strategy planning growth {i}",
                    domain=domain,
                    importance=0.9,
                )
        rs.find_cross_domain_patterns()
        high_conf = rs.get_insights(min_confidence=0.8)
        for ins in high_conf:
            assert ins.confidence >= 0.8

    def test_no_insights_initially(self):
        rs = RecallSynthesis()
        insights = rs.get_insights()
        assert insights == []

    def test_insight_to_dict(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "career"]:
            for i in range(3):
                rs.add_observation(
                    f"Budget planning strategy growth {i}",
                    domain=domain,
                )
        rs.find_cross_domain_patterns()
        insights = rs.get_insights()
        if insights:
            d = insights[0].to_dict()
            assert "domain" in d
            assert "confidence" in d


class TestConnectConcepts:
    def test_direct_connection(self):
        rs = RecallSynthesis()
        rs.add_observation("Sleep deprivation causes health problems", domain="health")
        rs.add_observation("Sleep quality affects work performance", domain="career")
        result = rs.connect_concepts("sleep health", "sleep work")
        assert isinstance(result, ConnectionResult)
        assert result.hops >= 0
        assert 0 <= result.confidence <= 1

    def test_no_connection(self):
        rs = RecallSynthesis()
        rs.add_observation("Quantum computing research", domain="career")
        rs.add_observation("Grocery shopping list", domain="home")
        result = rs.connect_concepts("quantum computing", "grocery shopping")
        assert result.confidence == 0.0

    def test_bridge_connection(self):
        rs = RecallSynthesis()
        rs.add_observation("Running marathon training", domain="health")
        rs.add_observation("Marathon registration costs money", domain="finance")
        rs.add_observation("Training schedule affects work productivity", domain="career")
        result = rs.connect_concepts("running training", "work productivity")
        assert isinstance(result, ConnectionResult)

    def test_connection_result_to_dict(self):
        rs = RecallSynthesis()
        rs.add_observation("Exercise health fitness", domain="health")
        rs.add_observation("Exercise career productivity", domain="career")
        result = rs.connect_concepts("exercise", "career")
        d = result.to_dict()
        assert "concept_a" in d
        assert "concept_b" in d
        assert "chain" in d


class TestGetDomainConnections:
    def test_domain_connections(self):
        rs = RecallSynthesis()
        for i in range(5):
            rs.add_observation(f"Exercise running fitness health {i}", domain="health")
            rs.add_observation(f"Exercise running fitness career {i}", domain="career")
        connections = rs.get_domain_connections("health")
        assert len(connections) > 0
        assert isinstance(connections[0], DomainConnection)

    def test_unknown_domain(self):
        rs = RecallSynthesis()
        connections = rs.get_domain_connections("unknown")
        assert connections == []

    def test_connection_strength(self):
        rs = RecallSynthesis()
        shared_kw = "productivity"
        for i in range(5):
            rs.add_observation(f"Morning {shared_kw} exercise {i}", domain="health")
            rs.add_observation(f"Work {shared_kw} focus {i}", domain="career")
        connections = rs.get_domain_connections("health")
        if connections:
            assert connections[0].strength > 0

    def test_connection_fields(self):
        rs = RecallSynthesis()
        for i in range(5):
            rs.add_observation(f"Investment strategy finance {i}", domain="finance")
            rs.add_observation(f"Investment strategy career {i}", domain="career")
        connections = rs.get_domain_connections("finance")
        if connections:
            c = connections[0]
            assert c.domain_a == "finance"
            assert isinstance(c.domain_b, str)
            assert isinstance(c.strength, float)
            assert isinstance(c.shared_keywords, list)
            assert isinstance(c.connection_count, int)

    def test_domain_connection_to_dict(self):
        rs = RecallSynthesis()
        for i in range(5):
            rs.add_observation(f"Growth learning development {i}", domain="growth")
            rs.add_observation(f"Growth career development {i}", domain="career")
        connections = rs.get_domain_connections("growth")
        if connections:
            d = connections[0].to_dict()
            assert "domain_a" in d
            assert "strength" in d


class TestGetStats:
    def test_empty_stats(self):
        rs = RecallSynthesis()
        stats = rs.get_stats()
        assert stats["observation_count"] == 0
        assert stats["pattern_count"] == 0
        assert stats["insight_count"] == 0
        assert stats["domain_distribution"] == {}

    def test_stats_after_observations(self):
        rs = RecallSynthesis()
        rs.add_observation("Obs 1", domain="health")
        rs.add_observation("Obs 2", domain="finance")
        rs.add_observation("Obs 3", domain="health")
        stats = rs.get_stats()
        assert stats["observation_count"] == 3
        assert stats["domain_distribution"]["health"] == 2
        assert stats["domain_distribution"]["finance"] == 1

    def test_stats_after_patterns(self):
        rs = RecallSynthesis()
        for domain in ["health", "finance", "career"]:
            for i in range(4):
                rs.add_observation(
                    f"Pattern discovery connection analysis {i}",
                    domain=domain,
                )
        rs.find_cross_domain_patterns()
        stats = rs.get_stats()
        assert stats["pattern_count"] > 0
        assert stats["insight_count"] > 0

    def test_stats_max_values(self):
        rs = RecallSynthesis()
        stats = rs.get_stats()
        assert stats["max_observations"] == 5000
        assert stats["max_patterns"] == 1000
        assert stats["max_insights"] == 500

    def test_avg_importance(self):
        rs = RecallSynthesis()
        rs.add_observation("High importance", importance=0.9)
        rs.add_observation("Low importance", importance=0.1)
        stats = rs.get_stats()
        assert abs(stats["avg_importance"] - 0.5) < 0.01


class TestThreadSafety:
    def test_concurrent_add_observations(self):
        rs = RecallSynthesis()
        errors = []

        def add_obs(n):
            try:
                for i in range(50):
                    rs.add_observation(f"Obs {n}-{i}", domain="health")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=add_obs, args=(t,)) for t in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert rs.get_stats()["observation_count"] == 250

    def test_concurrent_read_write(self):
        rs = RecallSynthesis()
        for d in LIFE_DOMAINS:
            rs.add_observation(f"Init {d}", domain=d)

        errors = []

        def writer():
            try:
                for i in range(20):
                    rs.add_observation(f"Writer {i}", domain="health")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(20):
                    rs.get_stats()
                    rs.synthesize("health")
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestDataClasses:
    def test_observation_dataclass(self):
        obs = Observation(
            observation_id="abc123",
            text="Test observation",
            domain="health",
            timestamp="2024-01-01T00:00:00+00:00",
            importance=0.7,
            keywords=["test", "observation"],
        )
        d = obs.to_dict()
        assert d["observation_id"] == "abc123"
        assert d["text"] == "Test observation"
        assert d["domain"] == "health"

    def test_cross_domain_pattern_dataclass(self):
        p = CrossDomainPattern(
            pattern_id="p1",
            keywords=["sleep", "stress"],
            domains_involved=["health", "career"],
            observations_linked=["o1", "o2", "o3"],
            confidence=0.8,
            description="Test pattern",
        )
        d = p.to_dict()
        assert d["pattern_id"] == "p1"
        assert d["confidence"] == 0.8

    def test_synthesis_result_dataclass(self):
        sr = SynthesisResult(
            query="test",
            direct_observations=[],
            cross_domain_insights=[],
            connections=[],
            confidence=0.5,
        )
        d = sr.to_dict()
        assert d["query"] == "test"

    def test_insight_dataclass(self):
        ins = Insight(
            domain="health",
            description="Test insight",
            confidence=0.7,
            source_observations=["o1"],
            created_at="2024-01-01T00:00:00+00:00",
        )
        d = ins.to_dict()
        assert d["domain"] == "health"
        assert "source_observations" in d

    def test_connection_result_dataclass(self):
        cr = ConnectionResult(
            concept_a="sleep",
            concept_b="productivity",
            chain=[],
            confidence=0.6,
            hops=0,
        )
        d = cr.to_dict()
        assert d["concept_a"] == "sleep"

    def test_domain_connection_dataclass(self):
        dc = DomainConnection(
            domain_a="health",
            domain_b="career",
            strength=0.7,
            shared_keywords=["stress"],
            connection_count=5,
        )
        d = dc.to_dict()
        assert d["domain_a"] == "health"


class TestEdgeCases:
    def test_zero_observations(self):
        rs = RecallSynthesis()
        assert rs.find_cross_domain_patterns() == []
        assert rs.get_insights() == []
        assert rs.get_stats()["observation_count"] == 0

    def test_single_observation_synthesize(self):
        rs = RecallSynthesis()
        rs.add_observation("Lone observation about Python programming")
        result = rs.synthesize("Python")
        assert isinstance(result, SynthesisResult)

    def test_long_text(self):
        rs = RecallSynthesis()
        text = " ".join(["important"] * 200)
        oid = rs.add_observation(text, domain="career")
        obs = rs._observations[oid]
        assert len(obs.keywords) > 0

    def test_unicode_text(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Meditation and 禅 practice improves focus")
        assert oid is not None

    def test_special_characters(self):
        rs = RecallSynthesis()
        oid = rs.add_observation("Cost: $500, ROI: 12% — good investment!")
        assert oid is not None

    def test_insight_cap_enforcement(self):
        rs = RecallSynthesis()
        rs._MAX_INSIGHTS = 2
        for batch in range(10):
            for domain in ["health", "finance", "career"]:
                for i in range(3):
                    rs.add_observation(
                        f"Cap test topic{batch} word{batch} connection {i}",
                        domain=domain,
                    )
        rs.find_cross_domain_patterns()
        assert len(rs._insights) <= 2

    def test_domain_proximity_symmetry(self):
        for (a, b), val in DOMAIN_PROXIMITY.items():
            reverse_val = DOMAIN_PROXIMITY.get((b, a), None)
            if reverse_val is not None:
                assert abs(val - reverse_val) < 0.01

    def test_life_domains(self):
        assert "health" in LIFE_DOMAINS
        assert "finance" in LIFE_DOMAINS
        assert "career" in LIFE_DOMAINS
        assert "growth" in LIFE_DOMAINS
        assert "home" in LIFE_DOMAINS
        assert "admin" in LIFE_DOMAINS
        assert "legal" in LIFE_DOMAINS
        assert "social" in LIFE_DOMAINS
        assert len(LIFE_DOMAINS) == 8

    def test_stop_words_count(self):
        assert len(STOP_WORDS) >= 30

    def test_connect_concepts_no_observations(self):
        rs = RecallSynthesis()
        result = rs.connect_concepts("sleep", "productivity")
        assert result.confidence == 0.0
        assert result.chain == []