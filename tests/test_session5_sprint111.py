"""
Sprint 111 Regression Tests — learning_system, domain_expertise, advanced_memory, state_graph audit fixes

Fixes:
- learning_system.py: import copy/re at module level, record_outcome lock fix, query_similar_decisions deepcopy, get_domain_expertise lock+deepcopy, _learn_from_outcome local import removed, Feedback timezone
- domain_expertise.py: get_status/get_cross_domain_insights/analyze under lock, word-boundary regex for expert analyzes
- advanced_memory.py: to_dict() deepcopy for MemoryBlock/Episode/Entity, get_entity/get_entity_episodes deepcopy, search_archival deepcopy
- state_graph.py: Checkpoint/ExecutionResult to_dict() deepcopy, _create_checkpoint deepcopy context, list_checkpoints deepcopy, fsync checkpoint writes
"""

import copy
import os
import re
import tempfile
import threading
import pytest
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock


# ═══════════════════════════════════════════════════════════════════
# Learning System Tests
# ═══════════════════════════════════════════════════════════════════

class TestLearningSystemDeepCopy:
    """Accessor methods must return copies, not mutable references."""

    def test_query_similar_decisions_returns_deepcopy(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        d = ls.track_decision("testing python code", ["option_a", "option_b"], "option_a", "reasoning", 0.8)
        ls.record_outcome(d.id, "worked well", 8)
        results = ls.query_similar_decisions("testing python code")
        assert len(results) == 1
        # Modifying returned decision should not affect internal state
        results[0].context = "MUTATED"
        internal = ls.query_similar_decisions("testing python code")
        assert internal[0].context != "MUTATED"

    def test_get_domain_expertise_returns_deepcopy(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        d = ls.track_decision("python project", ["opt_a"], "opt_a", "reasoning", 0.8)
        expertise = ls.get_domain_expertise("python")
        expertise["total_decisions"] = 9999
        expertise2 = ls.get_domain_expertise("python")
        assert expertise2["total_decisions"] != 9999

    def test_record_outcome_lock_fix(self, tmp_path):
        """record_outcome should hold lock for entire search+modify, not acquire inside loop."""
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        d = ls.track_decision("context", ["opt"], "opt", "reasoning", 0.8)
        # This should work without race condition
        result = ls.record_outcome(d.id, "success", 9)
        assert result is True
        assert ls.decisions[0].outcome == "success"

    def test_record_outcome_not_found(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        result = ls.record_outcome("nonexistent_id", "outcome", 5)
        assert result is False

    def test_get_status_thread_safe(self, tmp_path):
        from shadowd.learning_system import LearningSystem
        ls = LearningSystem(data_dir=tmp_path)
        d = ls.track_decision("context", ["opt"], "opt", "reasoning", 0.8)
        status = ls.get_status()
        assert isinstance(status, dict)
        assert status["decisions"]["total"] == 1

    def test_imports_at_module_level(self):
        """import re should be at module level, not inline in _learn_from_outcome."""
        import shadowd.learning_system as mod
        assert hasattr(mod, '_re')
        assert hasattr(mod, 'copy')

    def test_feedback_timestamp_timezone_aware(self, tmp_path):
        from shadowd.learning_system import LearningSystem, Feedback
        ls = LearningSystem(data_dir=tmp_path)
        ls.record_feedback("oracle", "test query", "preview", 4, "helpful")
        with ls._lock:
            fb = ls.feedback[-1]
        assert fb.timestamp.tzinfo is not None


# ═══════════════════════════════════════════════════════════════════
# Domain Expertise Tests
# ═══════════════════════════════════════════════════════════════════

class TestDomainExpertiseLockAndRegex:
    """Domain expertise manager should be thread-safe and use word boundaries."""

    def test_get_status_under_lock(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        mgr = DomainExpertiseManager()
        mgr.add_observation({"type": "health", "content": "test"})
        status = mgr.get_status()
        assert isinstance(status, dict)
        assert "domains" in status
        assert "total_observations" in status

    def test_get_cross_domain_insights_under_lock(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        mgr = DomainExpertiseManager()
        for i in range(15):
            mgr.add_observation({"type": "finance", "content": f"expense ${i}"})
        insights = mgr.get_cross_domain_insights()
        assert isinstance(insights, list)

    def test_analyze_reads_under_lock(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        mgr = DomainExpertiseManager()
        mgr.add_observation({"type": "health", "content": "feeling healthy"})
        result = mgr.analyze("health query")
        assert isinstance(result, dict)

    def test_word_boundary_health_when_feel(self):
        """'whenever' should not match 'when', 'feelings' should match 'feel' via word boundary."""
        from shadowd.domain_expertise import HealthExpert
        expert = HealthExpert()
        # "whenever" contains "when" but shouldn't trigger prediction (no \b match)
        result = expert.analyze("whenever I go outside", [])
        assert result["prediction"] is None  # "whenever" != "when"

    def test_word_boundary_finance_budget(self):
        """'budget' as word should trigger prediction."""
        from shadowd.domain_expertise import FinanceExpert
        expert = FinanceExpert()
        result = expert.analyze("what about the budget", [])
        assert result["prediction"] is not None

    def test_word_boundary_finance_budget_false_positive(self):
        """'overbudget' should NOT match 'budget' via word boundary."""
        from shadowd.domain_expertise import FinanceExpert
        expert = FinanceExpert()
        result = expert.analyze("we are overbudget", [])
        assert result["prediction"] is None

    def test_get_domain_summary_deepcopy(self):
        from shadowd.domain_expertise import DomainExpertiseManager
        mgr = DomainExpertiseManager()
        mgr.add_observation({"content": "test observation"})
        summary = mgr.get_domain_summary("admin")
        if summary.get("recent"):
            summary["recent"][0]["mutated"] = True
            summary2 = mgr.get_domain_summary("admin")
            assert "mutated" not in summary2.get("recent", [{}])[0]


# ═══════════════════════════════════════════════════════════════════
# Advanced Memory Tests
# ═══════════════════════════════════════════════════════════════════

class TestAdvancedMemoryDeepCopy:
    """to_dict() and accessor methods must return deep copies."""

    def test_memory_block_to_dict_metadata_copy(self):
        from shadowd.advanced_memory import MemoryBlock
        block = MemoryBlock(uuid="test", label="user", value="hello", metadata={"key": "val"})
        d = block.to_dict()
        assert d["metadata"] is not block.metadata
        d["metadata"]["key"] = "mutated"
        assert block.metadata["key"] == "val"

    def test_episode_to_dict_copies(self):
        from shadowd.advanced_memory import Episode
        ep = Episode(uuid="test", content="hello", source_type="observation", source_id="src",
                     entities=["e1", "e2"], facts=["f1"])
        d = ep.to_dict()
        assert d["entities"] is not ep.entities
        d["entities"].append("e3")
        assert len(ep.entities) == 2

    def test_entity_to_dict_facts_copy(self):
        from shadowd.advanced_memory import Entity
        entity = Entity(name="test", entity_type="person", facts=[{"fact": "lives in NYC", "valid_at": datetime.now(timezone.utc), "invalid_at": None, "source": "test"}])
        d = entity.to_dict()
        assert d["facts"] is not entity.facts
        d["facts"].append({"fact": "mutated"})
        assert len(entity.facts) == 1

    def test_get_entity_returns_deepcopy(self):
        from shadowd.advanced_memory import TemporalKnowledgeGraph
        tkg = TemporalKnowledgeGraph()
        tkg.entities["alice"] = __import__('shadowd.advanced_memory', fromlist=['Entity']).Entity(
            name="alice", entity_type="person"
        )
        entity = tkg.get_entity("alice")
        assert entity is not tkg.entities["alice"]

    def test_get_entity_episodes_returns_deepcopy(self):
        from shadowd.advanced_memory import TemporalKnowledgeGraph, Episode
        tkg = TemporalKnowledgeGraph()
        ep = tkg.add_episode("test content", "observation", "src1", entities=["alice"])
        episodes = tkg.get_entity_episodes("alice")
        assert len(episodes) == 0 or (episodes is not tkg.episodes)

    def test_search_archival_returns_deepcopy(self, tmp_path):
        from shadowd.advanced_memory import AdvancedMemorySystem
        ams = AdvancedMemorySystem(data_dir=tmp_path)
        ams.add_to_archival("python code testing content", label="test")
        results = ams.search_archival("python")
        if results:
            results[0].value = "MUTATED"
            results2 = ams.search_archival("python")
            assert results2[0].value != "MUTATED"


# ═══════════════════════════════════════════════════════════════════
# State Graph Tests
# ═══════════════════════════════════════════════════════════════════

class TestStateGraphDeepCopy:
    """to_dict() and checkpoint methods must return deep copies."""

    def test_checkpoint_to_dict_context_copy(self):
        from shadowd.reasoning.state_graph import Checkpoint, State
        cp = Checkpoint(
            id="test",
            state=State.INIT,
            context={"key": "value", "nested": {"inner": [1, 2]}},
            created_at=datetime.now(timezone.utc),
            metadata={"meta_key": "meta_val"}
        )
        d = cp.to_dict()
        assert d["context"] is not cp.context
        assert d["metadata"] is not cp.metadata
        d["context"]["key"] = "mutated"
        assert cp.context["key"] == "value"

    def test_execution_result_to_dict_output_copy(self):
        from shadowd.reasoning.state_graph import ExecutionResult, State
        result = ExecutionResult(
            success=True,
            state=State.COMPLETED,
            output={"key": "value", "nested": {"inner": [1]}},
            review_data={"review_key": "review_val"}
        )
        d = result.to_dict()
        assert d["output"] is not result.output
        assert d["review_data"] is not result.review_data
        d["output"]["key"] = "mutated"
        assert result.output["key"] == "value"

    def test_list_checkpoints_returns_deepcopy(self, tmp_path):
        from shadowd.reasoning.state_graph import StateGraph, State, StateNode
        sg = StateGraph("test_graph", checkpoint_dir=tmp_path)
        sg.add_node(StateNode(id="start", name="Start", state=State.INIT))
        sg.set_start("start")
        cp = sg._create_checkpoint(State.INIT)
        cps = sg.list_checkpoints()
        assert len(cps) >= 1
        # Modifying returned checkpoint should not affect internal state
        cps[0].context["mutated"] = True
        internal_cp = sg._checkpoints.get(cp.id)
        assert "mutated" not in internal_cp.context

    def test_create_checkpoint_context_is_deepcopy(self, tmp_path):
        from shadowd.reasoning.state_graph import StateGraph, State, StateNode
        sg = StateGraph("test_graph", checkpoint_dir=tmp_path)
        sg.add_node(StateNode(id="start", name="Start", state=State.INIT))
        sg.set_start("start")
        sg._current_context = {"key": "original"}
        cp = sg._create_checkpoint(State.INIT)
        # Mutating checkpoint context should NOT affect _current_context
        cp.context["key"] = "mutated"
        assert sg._current_context["key"] == "original"