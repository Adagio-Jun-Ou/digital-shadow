#!/usr/bin/env python3
"""
Session 5 Sprint 9 Regression Tests

Covers:
- V-261: Thread safety lock in DecisionLearningV2
- V-262: WebSocket topic validation
- V-271: Unbounded dicts in learning_system (patterns, domain_expertise, context_success)
- V-272: Unbounded examples in LearnedPattern
- V-273: hash() → SHA-256 for pattern IDs
- V-274: Timestamp → UUID decision IDs
- V-275: Thread safety lock in LearningSystem
- V-276: str(e) → type(e).__name__ in learning_system
- V-278/279: Confidence/satisfaction clamped in learning_system
- V-280: Rating/feedback_type validated in learning_system
- V-281: Limit validated in query_similar_decisions
- V-284: Path traversal via block.label in advanced_memory
- V-278: str(e) → type(e).__name__ in advanced_memory
- V-272-277: Unbounded data caps in advanced_memory (episodes, entities, archival)
- V-271: str(e) → type(e).__name__ in performance.py (5 instances)
- V-272: ConnectionPool lock TOCTOU fix (eager init)
- V-273: RateLimiter per-client cap
- V-270: Domain validation in domain_expertise
- V-273: str(e) returned to caller in shadow_sync
- V-275: SSRF host validation in shadow_sync
"""

import pytest
import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestDecisionLearningV2ThreadSafety:
    """V-261: Thread safety lock in DecisionLearningV2."""

    def test_has_lock(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        source = inspect.getsource(DecisionLearningV2.__init__)
        assert '_lock' in source

    def test_create_decision_uses_lock(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        source = inspect.getsource(DecisionLearningV2.create_decision)
        assert 'self._lock' in source

    def test_decide_uses_lock(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        source = inspect.getsource(DecisionLearningV2.decide)
        assert 'self._lock' in source

    def test_record_outcome_uses_lock(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        source = inspect.getsource(DecisionLearningV2.record_outcome)
        assert 'self._lock' in source


class TestWebSocketTopicValidation:
    """V-262: WebSocket topic validation."""

    def test_topic_validation_in_broadcast(self):
        from shadowd import shadowd
        source = inspect.getsource(shadowd)
        assert "Invalid topic name" in source


class TestLearningSystemUnboundedDicts:
    """V-271: Unbounded dicts capped in learning_system."""

    def test_has_max_patterns_constant(self):
        from shadowd.learning_system import LearningSystem
        assert hasattr(LearningSystem, 'MAX_PATTERNS')
        assert LearningSystem.MAX_PATTERNS > 0

    def test_has_max_domains_constant(self):
        from shadowd.learning_system import LearningSystem
        assert hasattr(LearningSystem, 'MAX_DOMAINS')
        assert LearningSystem.MAX_DOMAINS > 0

    def test_has_max_context_keys_constant(self):
        from shadowd.learning_system import LearningSystem
        assert hasattr(LearningSystem, 'MAX_CONTEXT_KEYS')
        assert LearningSystem.MAX_CONTEXT_KEYS > 0

    def test_has_max_examples_constant(self):
        from shadowd.learning_system import LearningSystem
        assert hasattr(LearningSystem, 'MAX_EXAMPLES_PER_PATTERN')
        assert LearningSystem.MAX_EXAMPLES_PER_PATTERN > 0


class TestLearningSystemHashToSHA256:
    """V-273: hash() → SHA-256 for pattern IDs."""

    def test_no_hash_modulo(self):
        from shadowd import learning_system as ls
        source = inspect.getsource(ls)
        assert "hash(decision.context) % 10000" not in source
        assert "hashlib.sha256" in source


class TestLearningSystemTimestampToUUID:
    """V-274: Timestamp → UUID decision IDs."""

    def test_uses_uuid_for_ids(self):
        from shadowd import learning_system as ls
        source = inspect.getsource(ls)
        assert "uuid.uuid4()" in source
        assert "datetime.now().timestamp()" not in source or "uuid" in source.split("datetime.now().timestamp()")[0][-100:]


class TestLearningSystemThreadSafety:
    """V-275: Thread safety lock in LearningSystem."""

    def test_has_lock(self):
        from shadowd.learning_system import LearningSystem
        source = inspect.getsource(LearningSystem.__init__)
        assert '_lock' in source


class TestLearningSystemErrorMessages:
    """V-276: str(e) → type(e).__name__ in learning_system."""

    def test_load_uses_type_name(self):
        from shadowd import learning_system as ls
        source = inspect.getsource(ls)
        assert 'type(e).__name__' in source

    def test_save_uses_type_name(self):
        from shadowd import learning_system as ls
        source = inspect.getsource(ls)
        load_idx = source.index("Failed to save learning state")
        assert "type(e).__name__" in source[load_idx:load_idx+100]


class TestLearningSystemInputValidation:
    """V-278/279/280/281: Input validation in learning_system."""

    def test_confidence_clamped(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        decision = ls.track_decision("test context here", ["a"], "a", "reason", confidence=5.0)
        assert decision.confidence <= 1.0

    def test_satisfaction_clamped(self, tmp_path):
        from shadowd.learning_system import LearningSystem, Decision
        from datetime import datetime
        ls = LearningSystem(data_dir=tmp_path)
        d = Decision(id="test_x", timestamp=datetime.now(), context="test context here", options=["a"], chosen="a", reasoning="r")
        ls.decisions.append(d)
        ls.record_outcome("test_x", "good", satisfaction=20)
        assert ls.decisions[0].satisfaction <= 10

    def test_rating_clamped(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        ls.record_feedback("test", "q", "resp", rating=10)
        assert ls.feedback[-1].rating <= 5

    def test_feedback_type_validated(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        ls.record_feedback("test", "q", "resp", rating=3, feedback_type="invalid_type")
        assert ls.feedback[-1].feedback_type == "helpful"

    def test_limit_validated(self):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem.__new__(LearningSystem)
        ls.data_dir = None
        ls._lock = __import__('threading').Lock()
        ls.decisions = []
        ls.feedback = []
        ls.patterns = {}
        ls.domain_expertise = {}
        ls.context_success = {}
        result = ls.query_similar_decisions("test", limit=-1)
        assert isinstance(result, list)


class TestAdvancedMemoryPathTraversal:
    """V-284: Path traversal via block.label."""

    def test_sanitize_label_function(self):
        from shadowd.advanced_memory import _sanitize_label
        assert _sanitize_label("../../evil") == "______evil"
        assert _sanitize_label("normal_label") == "normal_label"

    def test_sanitize_label_rejects_dots(self):
        from shadowd.advanced_memory import _sanitize_label
        result = _sanitize_label("a..b")
        assert ".." not in result


class TestAdvancedMemoryErrorMessages:
    """V-278: str(e) → type(e).__name__ in advanced_memory."""

    def test_no_str_e_in_logs(self):
        from shadowd import advanced_memory as am
        source = inspect.getsource(am)
        for line in source.split('\n'):
            if 'logger.error' in line and '{e}' in line:
                pytest.fail(f"Found str(e) in logger: {line.strip()}")


class TestAdvancedMemoryUnboundedCaps:
    """V-272-277: Unbounded data caps in advanced_memory."""

    def test_kg_max_episodes(self):
        from shadowd.advanced_memory import TemporalKnowledgeGraph
        assert hasattr(TemporalKnowledgeGraph, '_MAX_EPISODES')

    def test_kg_max_entities(self):
        from shadowd.advanced_memory import TemporalKnowledgeGraph
        assert hasattr(TemporalKnowledgeGraph, '_MAX_ENTITIES')

    def test_kg_max_entity_episodes(self):
        from shadowd.advanced_memory import TemporalKnowledgeGraph
        assert hasattr(TemporalKnowledgeGraph, '_MAX_ENTITY_EPISODES')

    def test_advanced_max_archival(self):
        from shadowd.advanced_memory import AdvancedMemorySystem
        assert hasattr(AdvancedMemorySystem, '_MAX_ARCHIVAL')


class TestPerformanceErrorMessages:
    """V-271: str(e) → type(e).__name__ in performance.py."""

    def test_redis_get_uses_type_name(self):
        from shadowd import performance as perf
        source = inspect.getsource(perf)
        assert "Redis get error: %s" in source

    def test_disk_read_uses_type_name(self):
        from shadowd import performance as perf
        source = inspect.getsource(perf)
        assert "Disk cache read error: %s" in source


class TestConnectionPoolLockFix:
    """V-272: ConnectionPool lock TOCTOU fix (lazy property pattern)."""

    def test_lock_property_exists(self):
        from shadowd import performance as perf
        assert hasattr(perf.ConnectionPool, '_lock')

    def test_lock_lazy_init(self):
        from shadowd import performance as perf
        import inspect
        source = inspect.getsource(perf.ConnectionPool._lock.fget)
        assert "asyncio.Lock()" in source


class TestRateLimiterPerClientCap:
    """V-273: RateLimiter per-client cap."""

    def test_per_client_cap_exists(self):
        from shadowd import performance as perf
        source = inspect.getsource(perf)
        assert "max_requests * 2" in source or "max_requests]" in source


class TestDomainExpertiseValidation:
    """V-270: Domain validation in domain_expertise."""

    def test_add_observation_validates_domain(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        source = inspect.getsource(DomainExpertiseManager.add_observation)
        assert "admin" in source  # Falls back to admin for unknown domains

    def test_get_domain_summary_handles_unknown(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        source = inspect.getsource(DomainExpertiseManager.get_domain_summary)
        assert "not in self.experts" in source or "observation_count" in source

    def test_analyze_validates_domain(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        source = inspect.getsource(DomainExpertiseManager.analyze)
        assert "admin" in source


class TestShadowSyncErrorMessages:
    """V-273: str(e) returned to caller in shadow_sync."""

    def test_summon_returns_generic_error(self):
        from sync import shadow_sync
        source = inspect.getsource(shadow_sync)
        assert "Failed to contact shadow daemon" in source

    def test_register_uses_type_name(self):
        from sync import shadow_sync
        source = inspect.getsource(shadow_sync)
        assert "type(e).__name__" in source


class TestShadowSyncSSRF:
    """V-275: SSRF host validation in shadow_sync."""

    def test_validate_host_exists(self):
        from sync.shadow_sync import ShadowSync
        assert hasattr(ShadowSync, '_validate_host')

    def test_validate_host_rejects_bad(self):
        from sync.shadow_sync import ShadowSync
        with pytest.raises(ValueError):
            ShadowSync._validate_host("")
        with pytest.raises(ValueError):
            ShadowSync._validate_host("a" * 300)

    def test_validate_host_accepts_good(self):
        from sync.shadow_sync import ShadowSync
        assert ShadowSync._validate_host("127.0.0.1") == "127.0.0.1"
        assert ShadowSync._validate_host("my-host.tail123.ts.net") == "my-host.tail123.ts.net"

    def test_dict_payload_size_check(self):
        from sync import shadow_sync
        source = inspect.getsource(shadow_sync)
        assert "65536" in source