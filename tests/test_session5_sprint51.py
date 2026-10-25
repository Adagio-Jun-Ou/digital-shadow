"""Sprint 51: ConflictResolution module + module audit fixes"""

import os
from pathlib import Path

import pytest
import threading

from shadowd.cognition.conflict_resolution import (
    ConflictResolution, ConflictType, ConflictIntensity,
    ConflictStatus, StrategyType, Perspective, Conflict, ResolutionStrategy,
    _MAX_CONFLICTS, _MAX_PERSPECTIVES, _MAX_STRATEGIES,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))


class TestConflictBasics:
    def test_create_conflict(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Resource Allocation", "resource", 2, "Team needs more budget")
        assert conflict.conflict_id
        assert conflict.conflict_type == ConflictType.RESOURCE
        assert conflict.intensity == ConflictIntensity.MEDIUM
        assert conflict.status == ConflictStatus.ACTIVE

    def test_create_conflict_all_types(self):
        cr = ConflictResolution()
        for ct in ConflictType:
            conflict = cr.create_conflict(f"Test {ct.value}", ct.value, 1)
            assert conflict.conflict_type == ct

    def test_create_conflict_invalid_type(self):
        cr = ConflictResolution()
        with pytest.raises(ValueError):
            cr.create_conflict("Test", "invalid_type", 2)

    def test_create_conflict_invalid_intensity(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 5)
        assert conflict.intensity == ConflictIntensity.MEDIUM

    def test_conflict_to_dict(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 3, "Description")
        d = conflict.to_dict()
        assert "conflict_id" in d
        assert "conflict_type" in d
        assert "intensity" in d
        assert "status" in d

    def test_add_perspective(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        result = cr.add_perspective(
            conflict.conflict_id, "Alice", "Need more resources",
            interests=["growth", "autonomy"], concerns=["overwork"],
            flexibility=0.7,
        )
        assert result is True
        updated = cr.get_conflict(conflict.conflict_id)
        assert len(updated.perspectives) == 1

    def test_add_perspective_nonexistent_conflict(self):
        cr = ConflictResolution()
        result = cr.add_perspective("nonexistent", "Alice", "Position")
        assert result is False

    def test_perspective_to_dict(self):
        p = Perspective(party="Alice", position="I need more time", flexibility=0.8)
        d = p.to_dict()
        assert d["party"] == "Alice"
        assert d["flexibility"] == 0.8


class TestConflictStrategies:
    def test_suggest_strategies_high_flexibility(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Team Direction", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "Go fast", flexibility=0.8)
        cr.add_perspective(conflict.conflict_id, "Bob", "Go carefully", flexibility=0.7)
        strategies = cr.suggest_strategies(conflict.conflict_id)
        assert len(strategies) >= 2
        types = [s.strategy_type for s in strategies]
        assert StrategyType.COLLABORATE in types
        assert StrategyType.COMPROMISE in types

    def test_suggest_strategies_low_flexibility(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Budget Dispute", "resource", 3)
        cr.add_perspective(conflict.conflict_id, "Alice", "Must have budget", flexibility=0.2)
        cr.add_perspective(conflict.conflict_id, "Bob", "No budget available", flexibility=0.1)
        strategies = cr.suggest_strategies(conflict.conflict_id)
        assert len(strategies) >= 2

    def test_suggest_strategies_high_intensity(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Critical Dispute", "value", 4)
        cr.add_perspective(conflict.conflict_id, "Alice", "Position A", flexibility=0.5)
        cr.add_perspective(conflict.conflict_id, "Bob", "Position B", flexibility=0.4)
        strategies = cr.suggest_strategies(conflict.conflict_id)
        types = [s.strategy_type for s in strategies]
        assert StrategyType.AVOID in types

    def test_suggest_strategies_no_perspectives(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 1)
        strategies = cr.suggest_strategies(conflict.conflict_id)
        assert len(strategies) >= 1

    def test_suggest_strategies_nonexistent(self):
        cr = ConflictResolution()
        strategies = cr.suggest_strategies("nonexistent")
        assert strategies == []


class TestConflictResolution:
    def test_resolve_conflict(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        result = cr.resolve_conflict(conflict.conflict_id, "Agreed to compromise")
        assert result is True
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED
        assert updated.resolution == "Agreed to compromise"
        assert updated.resolved_at is not None

    def test_resolve_nonexistent_conflict(self):
        cr = ConflictResolution()
        result = cr.resolve_conflict("nonexistent", "Resolution")
        assert result is False

    def test_escalate_conflict(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 1)
        result = cr.escalate_conflict(conflict.conflict_id)
        assert result is True
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.ESCALATED
        assert updated.intensity.value == 2

    def test_escalate_already_critical(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 4)
        cr.escalate_conflict(conflict.conflict_id)
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.intensity.value == 4

    def test_escalate_nonexistent(self):
        cr = ConflictResolution()
        result = cr.escalate_conflict("nonexistent")
        assert result is False


class TestConflictAnalysis:
    def test_analyze_common_ground(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "Position A",
                           interests=["growth", "learning"], concerns=["overwork"])
        cr.add_perspective(conflict.conflict_id, "Bob", "Position B",
                           interests=["growth", "stability"], concerns=["overwork", "burnout"])
        analysis = cr.analyze_common_ground(conflict.conflict_id)
        assert "common_interests" in analysis
        assert "growth" in analysis["common_interests"]
        assert "common_concerns" in analysis
        assert "overwork" in analysis["common_concerns"]
        assert analysis["overlap_score"] > 0

    def test_analyze_common_ground_single_perspective(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "Position A")
        analysis = cr.analyze_common_ground(conflict.conflict_id)
        assert analysis["overlap_score"] == 0.0

    def test_analyze_nonexistent_conflict(self):
        cr = ConflictResolution()
        analysis = cr.analyze_common_ground("nonexistent")
        assert "common_interests" in analysis
        assert "common_concerns" in analysis
        assert "overlap_score" in analysis
        assert "perspective_count" in analysis


class TestConflictListAndStats:
    def test_list_conflicts(self):
        cr = ConflictResolution()
        cr.create_conflict("Test 1", "interest", 2)
        cr.create_conflict("Test 2", "resource", 3)
        conflicts = cr.list_conflicts()
        assert len(conflicts) == 2
        assert conflicts[0].intensity.value >= conflicts[1].intensity.value

    def test_list_conflicts_by_status(self):
        cr = ConflictResolution()
        c1 = cr.create_conflict("Test 1", "interest", 2)
        cr.create_conflict("Test 2", "resource", 3)
        cr.resolve_conflict(c1.conflict_id, "Done")
        active = cr.list_conflicts(status="active")
        assert len(active) == 1

    def test_get_stats(self):
        cr = ConflictResolution()
        cr.create_conflict("Test 1", "interest", 2)
        c2 = cr.create_conflict("Test 2", "resource", 3)
        cr.resolve_conflict(c2.conflict_id, "Resolved")
        stats = cr.get_stats()
        assert stats["total_conflicts"] == 2
        assert stats["active"] == 1
        assert stats["resolved"] == 1
        assert stats["resolution_rate"] == 0.5

    def test_resolution_history(self):
        cr = ConflictResolution()
        c1 = cr.create_conflict("Test", "interest", 2)
        cr.resolve_conflict(c1.conflict_id, "Agreed")
        history = cr.get_resolution_history()
        assert len(history) == 1
        assert history[0]["resolution"] == "Agreed"

    def test_remove_conflict(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        assert cr.remove_conflict(conflict.conflict_id) is True
        assert cr.get_conflict(conflict.conflict_id) is None

    def test_remove_nonexistent(self):
        cr = ConflictResolution()
        assert cr.remove_conflict("nonexistent") is False


class TestConflictSecurity:
    def test_sanitize_title(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test\x00Title", "interest", 1)
        assert "\x00" not in conflict.title

    def test_title_truncation(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("A" * 300, "interest", 1)
        assert len(conflict.title) <= 200

    def test_perspective_capping(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        for i in range(_MAX_PERSPECTIVES + 5):
            cr.add_perspective(conflict.conflict_id, f"Party_{i}", f"Position_{i}")
        updated = cr.get_conflict(conflict.conflict_id)
        assert len(updated.perspectives) <= _MAX_PERSPECTIVES

    def test_conflict_cap(self):
        cr = ConflictResolution()
        for i in range(_MAX_CONFLICTS + 5):
            cr.create_conflict(f"Conflict_{i}", "interest", 1)
        conflicts = cr.list_conflicts()
        assert len(conflicts) <= _MAX_CONFLICTS

    def test_flexibility_clamped(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test", "interest", 2)
        cr.add_perspective(conflict.conflict_id, "Alice", "Pos", flexibility=2.0)
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.perspectives[0].flexibility <= 1.0

    def test_thread_safety(self):
        cr = ConflictResolution()
        errors = []

        def worker():
            try:
                for i in range(25):
                    conflict = cr.create_conflict(f"C_{i}", "interest", 1)
                    cr.add_perspective(conflict.conflict_id, "P1", "Pos")
                    cr.suggest_strategies(conflict.conflict_id)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0
        stats = cr.get_stats()
        assert stats["total_conflicts"] > 0


class TestConflictIntegration:
    def test_full_conflict_lifecycle(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Team Direction", "interest", 2, "Disagreement on project priorities")
        cr.add_perspective(conflict.conflict_id, "Alice", "Focus on speed", interests=["delivery", "growth"], flexibility=0.6)
        cr.add_perspective(conflict.conflict_id, "Bob", "Focus on quality", interests=["quality", "growth"], flexibility=0.7)

        strategies = cr.suggest_strategies(conflict.conflict_id)
        assert len(strategies) >= 2

        analysis = cr.analyze_common_ground(conflict.conflict_id)
        assert "growth" in analysis["common_interests"]

        cr.resolve_conflict(conflict.conflict_id, "Agreed to balance speed and quality")
        updated = cr.get_conflict(conflict.conflict_id)
        assert updated.status == ConflictStatus.RESOLVED
        assert updated.resolution is not None

    def test_conflict_in_shadow_os(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        assert hasattr(os, 'conflict')
        os._running = True
        os._start_time = 0
        health = os.get_health()
        assert "conflict_resolution" in health["modules"]

    def test_conflict_api_endpoints(self):
        from shadowd.cognition.api_router import router
        routes = [r.path for r in router.routes]
        assert "/os/conflict/create" in routes
        assert "/os/conflict/perspective" in routes
        assert "/os/conflict/{conflict_id}" in routes
        assert "/os/conflict/{conflict_id}/strategies" in routes
        assert "/os/conflict/{conflict_id}/common-ground" in routes
        assert "/os/conflicts" in routes
        assert "/os/conflict/stats" in routes


class TestSprint49AuditFixes:
    def test_adaptive_skill_tree_add_skill_atomic(self):
        source = open(_repo_file("shadowd/cognition/adaptive_skill_tree.py")).read()
        add_method_start = source.index("def add_skill(")
        add_method_end = source.index("\n    def ", add_method_start + 1)
        add_method = source[add_method_start:add_method_end]
        assert add_method.count("with self._lock:") == 1, "add_skill should use single lock section"

    def test_adaptive_skill_tree_remove_rechecks_unlocks(self):
        source = open(_repo_file("shadowd/cognition/adaptive_skill_tree.py")).read()
        rm_start = source.index("def remove_skill(")
        rm_end = source.index("\n    def ", rm_start + 1)
        rm_method = source[rm_start:rm_end]
        assert "unlocked" in rm_method, "remove_skill should recheck unlock status"

    def test_narrative_memory_create_chapter_atomic(self):
        source = open(_repo_file("shadowd/cognition/narrative_memory.py")).read()
        ch_start = source.index("def create_chapter(")
        ch_end = source.index("\n    def ", ch_start + 1)
        ch_method = source[ch_start:ch_end]
        assert ch_method.count("with self._lock:") == 1, "create_chapter should use single lock section"

    def test_narrative_memory_create_arc_atomic(self):
        source = open(_repo_file("shadowd/cognition/narrative_memory.py")).read()
        arc_start = source.index("def create_arc(")
        arc_end = source.index("\n    def ", arc_start + 1)
        arc_method = source[arc_start:arc_end]
        assert arc_method.count("with self._lock:") == 1, "create_arc should use single lock section"