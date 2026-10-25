#!/usr/bin/env python3
"""
Session 5 Sprint 10-11 Regression Tests

Covers:
Sprint 10 (memory_optimization.py fixes):
- V-270: SmartCache.invalidate() acquires lock
- V-271: QueryOptimizer.query_patterns capped
- V-272: PredictivePrefetch.context_prefetch capped
- V-273: Bare except → specific in MemoryCompressor

Sprint 11 (features):
- Feature: Super orchestrator retry/fallback logic
- Feature: Knowledge graph typed relations
- Feature: Multi-label domain classifier
- Feature: Expanded personality pattern detection
"""

import pytest
import inspect
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


class TestSmartCacheInvalidateLock:
    """V-270: SmartCache.invalidate() acquires lock."""

    def test_invalidate_uses_lock(self):
        from shadowd.memory_optimization import SmartCache
        source = inspect.getsource(SmartCache.invalidate)
        assert 'self._lock' in source

    def test_invalidate_clear_uses_lock(self):
        from shadowd.memory_optimization import SmartCache
        source = inspect.getsource(SmartCache.invalidate)
        assert 'with self._lock' in source


class TestQueryOptimizerCaps:
    """V-271: QueryOptimizer.query_patterns capped."""

    def test_has_max_query_patterns(self):
        from shadowd.memory_optimization import QueryOptimizer
        assert hasattr(QueryOptimizer, '_MAX_QUERY_PATTERNS')
        assert QueryOptimizer._MAX_QUERY_PATTERNS > 0

    def test_query_length_limited(self):
        from shadowd.memory_optimization import QueryOptimizer
        source = inspect.getsource(QueryOptimizer.optimize_query)
        assert '10000' in source or 'len(query)' in source

    def test_patterns_evicted(self):
        from shadowd.memory_optimization import QueryOptimizer
        qo = QueryOptimizer()
        qo._MAX_QUERY_PATTERNS = 5
        for i in range(10):
            qo.optimize_query(f"query_{i}")
        assert len(qo.query_patterns) <= 5


class TestPredictivePrefetchCaps:
    """V-272: PredictivePrefetch.context_prefetch capped."""

    def test_has_max_contexts(self):
        from shadowd.memory_optimization import PredictivePrefetch
        assert hasattr(PredictivePrefetch, '_MAX_CONTEXTS')
        assert PredictivePrefetch._MAX_CONTEXTS > 0

    def test_has_max_queries_per_context(self):
        from shadowd.memory_optimization import PredictivePrefetch
        assert hasattr(PredictivePrefetch, '_MAX_QUERIES_PER_CONTEXT')
        assert PredictivePrefetch._MAX_QUERIES_PER_CONTEXT > 0

    def test_contexts_capped(self):
        from shadowd.memory_optimization import PredictivePrefetch
        pp = PredictivePrefetch()
        pp._MAX_CONTEXTS = 5
        for i in range(10):
            pp.record_access(f"ctx_{i}", f"query_{i}")
        assert len(pp.context_prefetch) <= 5

    def test_no_sequence_patterns_dead_code(self):
        from shadowd.memory_optimization import PredictivePrefetch
        source = inspect.getsource(PredictivePrefetch)
        assert 'sequence_patterns' not in source

    def test_queries_per_context_capped(self):
        from shadowd.memory_optimization import PredictivePrefetch
        pp = PredictivePrefetch()
        pp._MAX_QUERIES_PER_CONTEXT = 5
        for i in range(10):
            pp.record_access("ctx", f"query_{i}")
        assert len(pp.context_prefetch["ctx"]) <= 5


class TestMemoryCompressorBareExcept:
    """V-273: Bare except → specific."""

    def test_no_bare_except(self):
        from shadowd import memory_optimization as mo
        source = inspect.getsource(mo)
        lines = source.split('\n')
        for i, line in enumerate(lines):
            if 'except:' in line and 'ValueError' not in line and 'TypeError' not in line and 'OSError' not in line and 'KeyError' not in line:
                stripped = line.strip()
                if stripped == 'except:' or stripped.startswith('except:'):
                    pytest.fail(f"Found bare except at line {i+1}: {line.strip()}")


class TestSuperOrchestratorRetry:
    """Feature: Super orchestrator retry/fallback logic."""

    def test_has_retry_method(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        assert hasattr(SuperShadowOrchestrator, '_retry_execute')

    def test_has_max_retries_constant(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        assert hasattr(SuperShadowOrchestrator, '_MAX_RETRIES')
        assert SuperShadowOrchestrator._MAX_RETRIES > 0

    def test_process_uses_fallback(self):
        from shadowd.super_orchestrator import SuperShadowOrchestrator
        source = inspect.getsource(SuperShadowOrchestrator.process)
        assert 'falling back' in source.lower() or 'errors' in source.lower()


class TestKnowledgeGraphTypedRelations:
    """Feature: Knowledge graph typed relations."""

    def test_typed_relation_class(self):
        from shadowd.knowledge_graph import TypedRelation
        assert TypedRelation is not None

    def test_relation_type_class(self):
        from shadowd.knowledge_graph import RelationType
        assert hasattr(RelationType, 'KNOWS')
        assert hasattr(RelationType, 'WORKS_FOR')
        assert hasattr(RelationType, 'LIVES_IN')
        assert hasattr(RelationType, 'REPORTS_TO')

    def test_typed_relation_valid_types(self):
        from shadowd.knowledge_graph import TypedRelation, RelationType
        assert RelationType.KNOWS in TypedRelation.VALID_TYPES
        assert len(TypedRelation.VALID_TYPES) >= 7

    def test_add_relation_method(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=None)
        assert hasattr(kg, 'add_relation')

    def test_get_relations_method(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=None)
        assert hasattr(kg, 'get_relations')

    def test_get_relation_between_method(self):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph
        kg = TemporalKnowledgeGraph(data_dir=None)
        assert hasattr(kg, 'get_relation_between')

    def test_add_and_retrieve_relation(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        rel = kg.add_relation("Alice", "Acme Corp", RelationType.WORKS_FOR)
        assert rel.from_entity == "Alice"
        assert rel.to_entity == "Acme Corp"
        assert rel.relation_type == RelationType.WORKS_FOR

    def test_relation_confidence_clamped(self):
        from shadowd.knowledge_graph import TypedRelation, RelationType
        rel = TypedRelation(
            relation_id="test",
            from_entity="A",
            to_entity="B",
            relation_type=RelationType.KNOWS,
            confidence=2.0,
        )
        d = rel.to_dict()
        assert d["confidence"] <= 1.0

    def test_invalid_relation_type_defaults(self, tmp_path):
        from shadowd.knowledge_graph import TemporalKnowledgeGraph, RelationType
        kg = TemporalKnowledgeGraph(data_dir=tmp_path)
        rel = kg.add_relation("A", "B", "invalid_type")
        assert rel.relation_type == RelationType.RELATED_TO


class TestMultiLabelDomainClassifier:
    """Feature: Multi-label domain classifier in shadow_enhanced."""

    def test_has_classify_domains_multi(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        assert hasattr(EnhancedShadowAI, '_classify_domains_multi')

    def test_multi_label_returns_list(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        ai = EnhancedShadowAI.__new__(EnhancedShadowAI)
        ai._classify_domains_multi  # Just verify method exists
        source = inspect.getsource(EnhancedShadowAI._classify_domains_multi)
        assert 'List[str]' in source or 'results' in source

    def test_classify_domain_uses_scored_dict(self):
        from shadowd.shadow_enhanced import EnhancedShadowAI
        source = inspect.getsource(EnhancedShadowAI._classify_domain)
        assert 'scored' in source


class TestExpandedPatternDetection:
    """Feature: Expanded personality pattern detection."""

    def test_has_risk_patterns(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._auto_detect_pattern)
        assert 'risk_taker' in source or 'risk_averse' in source

    def test_has_social_patterns(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._auto_detect_pattern)
        assert 'social_interaction' in source or 'group_socializer' in source

    def test_has_learning_patterns(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._auto_detect_pattern)
        assert 'learning' in source

    def test_has_perfectionist_pattern(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._auto_detect_pattern)
        assert 'perfectionist' in source

    def test_has_communication_tone(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        source = inspect.getsource(PersonalityEngine._auto_detect_pattern)
        assert 'formal_communicator' in source or 'tone' in source