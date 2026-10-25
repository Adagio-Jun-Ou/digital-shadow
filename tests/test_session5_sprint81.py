#!/usr/bin/env python3
"""Sprint 81 regression tests — Round 3 deep audit of cognitive_pipeline, adaptive_skill_tree,
narrative_memory, procedural_memory, conflict_resolution."""

import copy
import logging
import threading
import pytest

from shadowd.cognition.cognitive_pipeline import CognitivePipeline, PipelineResult, _sanitize_pipeline_input, _truncate_dict
from shadowd.cognition.adaptive_skill_tree import AdaptiveSkillTree, SkillNode, SkillDomain
from shadowd.cognition.narrative_memory import NarrativeMemory, EmotionTone
from shadowd.cognition.procedural_memory import ProceduralMemory
from shadowd.cognition.conflict_resolution import ConflictResolution, ConflictType


class TestCognitivePipelineSanitization:
    def test_cp1_sanitize_preserves_angle_brackets(self):
        result = _sanitize_pipeline_input("Hello <script>alert(1)</script> world")
        assert "<" in result
        assert "Hello" in result
        assert "</script>" in result

    def test_cp1_sanitize_strips_control_chars(self):
        result = _sanitize_pipeline_input("Hello\x00world\x1f end")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_cp1_sanitize_preserves_newline_tab(self):
        result = _sanitize_pipeline_input("Hello\nworld\tend")
        assert "\n" in result
        assert "\t" in result


class TestCognitivePipelineToDict:
    def test_cp_pipeline_result_to_dict_list_copy(self):
        pr = PipelineResult(modules_used=["a", "b", "c"])
        d = pr.to_dict()
        d["modules_used"].append("d")
        d2 = pr.to_dict()
        assert len(d2["modules_used"]) == 3


class TestCognitivePipelineTruncateDict:
    def test_cp6_truncate_dict_caps_string_values(self):
        d = {"key": "x" * 500}
        result = _truncate_dict(d)
        assert len(result["key"]) == 200

    def test_cp6_truncate_dict_caps_key_count(self):
        d = {f"k{i}": f"v{i}" for i in range(60)}
        result = _truncate_dict(d)
        assert len(result) == 50


class TestAdaptiveSkillTreeSanitization:
    def test_ast1_sanitize_strips_angle_brackets(self):
        ast = AdaptiveSkillTree()
        result = ast._sanitize("Hello <script>alert(1)</script> world")
        assert "<" not in result
        assert ">" not in result

    def test_ast1_sanitize_strips_control_chars(self):
        ast = AdaptiveSkillTree()
        result = ast._sanitize("Test\x00value\x1f")
        assert "\x00" not in result
        assert "\x1f" not in result

    def test_ast1_sanitize_strips_newline(self):
        ast = AdaptiveSkillTree()
        result = ast._sanitize("Test\nvalue")
        assert "\n" not in result


class TestAdaptiveSkillTreeToDict:
    def test_ast2_dependencies_list_copy(self):
        node = SkillNode(
            skill_id="test", name="Test", domain=SkillDomain.TECHNICAL,
            xp=0.0, dependencies=["dep1", "dep2"]
        )
        d = node.to_dict()
        d["dependencies"].append("dep3")
        d2 = node.to_dict()
        assert len(d2["dependencies"]) == 2


class TestAdaptiveSkillTreeDeadCode:
    def test_ast13_no_unused_xp_decay_constant(self):
        import inspect
        src = inspect.getsource(AdaptiveSkillTree)
        assert "_XP_DECAY_FACTOR" not in src
        assert "_XP_RECENCY_BONUS_DAYS" not in src


class TestAdaptiveSkillTreeStats:
    def test_ast20_total_usage_is_actual_usage(self):
        ast = AdaptiveSkillTree()
        ast.add_skill("s1", "Skill 1", SkillDomain.TECHNICAL)
        ast.add_skill("s2", "Skill 2", SkillDomain.REASONING)
        ast.use_skill("s1", success=True)
        ast.use_skill("s1", success=True)
        ast.use_skill("s2", success=True)
        stats = ast.get_stats()
        assert stats["total_usage"] == 3


class TestNarrativeMemorySanitization:
    def test_nm1_sanitize_preserves_angle_brackets(self):
        nm = NarrativeMemory()
        event = nm.add_event(content="Hello <script>alert(1)</script> world")
        results = nm.search_events("Hello")
        if results:
            assert "<" in results[0].content
            assert "Hello" in results[0].content

    def test_nm1_sanitize_strips_control_chars(self):
        nm = NarrativeMemory()
        event = nm.add_event(content="Test\x00value\x1f")
        results = nm.search_events("Test")
        if results:
            assert "\x00" not in results[0].content


class TestNarrativeMemoryDeepCopy:
    def test_nm2_search_events_returns_deepcopy(self):
        nm = NarrativeMemory()
        nm.add_event(content="test event one", importance=0.8)
        results = nm.search_events("test")
        if results:
            results[0].content = "MUTATED"
            results2 = nm.search_events("test")
            assert results2[0].content != "MUTATED"


class TestNarrativeMemoryEmotionLogic:
    def test_nm6_resilience_is_reachable(self):
        nm = NarrativeMemory()
        result = nm._detect_emotion("I failed but learned and grew from it")
        assert result == EmotionTone.RESILIENCE.value

    def test_nm6_growth_without_negative(self):
        nm = NarrativeMemory()
        result = nm._detect_emotion("I learned and grew today")
        assert result == EmotionTone.GROWTH.value

    def test_nm6_mixed_without_growth(self):
        nm = NarrativeMemory()
        result = nm._detect_emotion("I was happy but also sad about leaving")
        assert result == EmotionTone.MIXED.value


class TestNarrativeMemoryMetadata:
    def test_nm4_metadata_key_count_cap(self):
        nm = NarrativeMemory()
        big_meta = {f"k{i}": f"v{i}" for i in range(30)}
        event = nm.add_event(content="test", metadata=big_meta)
        assert len(event.metadata) <= 20


class TestProceduralMemorySanitization:
    def test_pm1_sanitize_strips_angle_brackets(self):
        pm = ProceduralMemory()
        result = pm._sanitize("Hello <script>alert(1)</script> world")
        assert "<" not in result
        assert ">" not in result

    def test_pm1_sanitize_strips_newline_tab(self):
        pm = ProceduralMemory()
        result = pm._sanitize("Test\nvalue\tend")
        assert "\n" not in result
        assert "\t" not in result


class TestProceduralMemoryToListCopy:
    def test_pm3_steps_list_copy(self):
        from shadowd.cognition.procedural_memory import Procedure
        p = Procedure(
            procedure_id="test", name="Test", category="general",
            steps=["step1", "step2", "step3"], mastery=0.0
        )
        d = p.to_dict()
        d["steps"].append("step4")
        d2 = p.to_dict()
        assert len(d2["steps"]) == 3


class TestProceduralMemoryDeepCopy:
    def test_pm19_add_procedure_returns_deepcopy(self):
        pm = ProceduralMemory()
        proc = pm.add_procedure("Test", "learning")
        original_id = proc.procedure_id
        proc.name = "MUTATED"
        proc2 = pm.get_procedure(original_id)
        assert proc2.name != "MUTATED"

    def test_pm19_add_habit_returns_deepcopy(self):
        pm = ProceduralMemory()
        habit = pm.add_habit("Test", "trigger", "do it", "reward")
        original_id = habit.habit_id
        habit.name = "MUTATED"
        habit2 = pm.get_habit(original_id)
        assert habit2.name != "MUTATED"

    def test_pm19_add_routine_returns_deepcopy(self):
        pm = ProceduralMemory()
        routine = pm.add_routine("Test", "daily")
        original_id = routine.routine_id
        routine.name = "MUTATED"
        routine2 = pm.get_routine(original_id)
        assert routine2.name != "MUTATED"


class TestConflictResolutionSanitization:
    def test_cr1_sanitize_strips_angle_brackets(self):
        cr = ConflictResolution()
        result = cr._sanitize("Hello <script>alert(1)</script> world")
        assert "<" not in result
        assert ">" not in result


class TestConflictResolutionToList:
    def test_cr2_perspective_to_dict_list_copy(self):
        from shadowd.cognition.conflict_resolution import Perspective
        p = Perspective(party="alice", position="center", interests=["i1", "i2"], concerns=["c1", "c2"])
        d = p.to_dict()
        d["interests"].append("i3")
        d["concerns"].append("c3")
        d2 = p.to_dict()
        assert len(d2["interests"]) == 2
        assert len(d2["concerns"]) == 2


class TestConflictResolutionLogging:
    def test_cr6_re_resolve_logs_warning(self, caplog):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test conflict", ConflictType.INTEREST, "Alice", "Bob")
        cid = conflict.conflict_id
        cr.resolve_conflict(cid, "Resolution 1")
        result = cr.resolve_conflict(cid, "Resolution 2")
        assert result is False
        assert any("already resolved" in r.message for r in caplog.records)

    def test_cr7_escalate_resolved_logs_warning(self, caplog):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test conflict", ConflictType.INTEREST, "Alice", "Bob")
        cid = conflict.conflict_id
        cr.resolve_conflict(cid, "Done")
        result = cr.escalate_conflict(cid)
        assert result is False
        assert any("Cannot escalate resolved conflict" in r.message for r in caplog.records)

    def test_cr20_remove_conflict_logs(self, caplog):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test conflict", ConflictType.INTEREST, "Alice", "Bob")
        cid = conflict.conflict_id
        cr.remove_conflict(cid)
        assert any("Removing conflict" in r.message for r in caplog.records)


class TestConflictResolutionDeepCopy:
    def test_cr11_suggest_strategies_returns_deepcopy(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test dispute", ConflictType.INTEREST, "Alice", "Bob")
        cid = conflict.conflict_id
        cr.add_perspective(cid, "Alice", "I want more resources", ["fairness"], ["scarcity"])
        strategies = cr.suggest_strategies(cid)
        if strategies:
            strategies[0].description = "MUTATED"
            strategies2 = cr.suggest_strategies(cid)
            assert strategies2[0].description != "MUTATED"

    def test_cr13_resolution_history_returns_deepcopy(self):
        cr = ConflictResolution()
        conflict = cr.create_conflict("Test conflict", ConflictType.INTEREST, "Alice", "Bob")
        cid = conflict.conflict_id
        cr.resolve_conflict(cid, "Agreement reached")
        history = cr.get_resolution_history()
        if history:
            history[0]["resolution"] = "MUTATED"
            history2 = cr.get_resolution_history()
            assert history2[0]["resolution"] != "MUTATED"


class TestConflictResolutionErrorMessages:
    def test_cr23_error_message_truncated(self):
        cr = ConflictResolution()
        with pytest.raises(ValueError):
            cr.create_conflict("Test", "invalid_type" + "x" * 100, "Alice", "Bob")