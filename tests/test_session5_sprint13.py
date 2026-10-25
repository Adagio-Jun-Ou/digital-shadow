#!/usr/bin/env python3
"""Session 5 Sprint 13 regression tests - Cognition module deep audit.

Covers:
- V-310: ShadowGraph trace error → type(e).__name__
- V-311: ShadowGraph persist error → type(e).__name__
- V-312/V-313: ShadowGraph hooks capped at 20
- V-316: ShadowGraph edges capped at 100
- V-321: PersonalityEngine behavior_patterns capped
- V-323: PersonalityEngine pattern example sanitization
- V-330: Escalation SafetyGuardrail no pattern leak
- V-331: Escalation atomic history persist
- V-332: Escalation guardrails capped
- V-333: Escalation queue capped
- V-341: DarkMemory importance clamped on store/import
- V-343: DarkMemory search doesn't mutate metadata
"""

import asyncio
import json
import pytest
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, AsyncMock

sys.path.insert(0, str(Path(__file__).parent.parent))


class TestShadowGraphTraceError:
    """V-310: ShadowGraph trace error uses type name."""

    def test_trace_error_no_str_e(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        import inspect
        source = inspect.getsource(ShadowGraph.process_event)
        assert "trace.error = str(e)" not in source
        assert "type(e).__name__" in source


class TestShadowGraphPersistError:
    """V-311: ShadowGraph persist error uses type name."""

    def test_persist_error_no_str_e(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        import inspect
        source = inspect.getsource(ShadowGraph._persist_checkpoint)
        assert "type(e).__name__" in source


class TestShadowGraphHooksCap:
    """V-312/V-313: ShadowGraph hooks capped at 20."""

    def test_max_hooks_constant(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph(graph_id="test")
        assert g._MAX_HOOKS == 20

    def test_transition_hooks_eviction(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph(graph_id="test")
        for i in range(25):
            g.on_transition(lambda *a: None)
        assert len(g._transition_hooks) <= 20

    def test_error_hooks_eviction(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph(graph_id="test")
        for i in range(25):
            g.on_error(lambda *a: None)
        assert len(g._error_hooks) <= 20


class TestShadowGraphEdgesCap:
    """V-316: ShadowGraph edges capped at 100."""

    def test_edges_eviction(self):
        from shadowd.cognition.shadow_graph import ShadowGraph
        g = ShadowGraph(graph_id="test_edge_cap")
        base_count = len(g.edges)
        for i in range(105):
            g.add_edge("listen", "analyze")
        assert len(g.edges) <= max(base_count + 100, 100)


class TestPersonalityEnginePatternCap:
    """V-321: PersonalityEngine behavior_patterns capped."""

    def test_max_patterns_constant(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        assert PersonalityEngine.MAX_BEHAVIOR_PATTERNS == 200

    def test_patterns_eviction(self):
        from shadowd.cognition.personality_engine import PersonalityEngine, PersonalityProfile
        engine = PersonalityEngine(data_dir=tempfile.mkdtemp())
        for i in range(210):
            engine.profile.add_or_reinforce_pattern(f"pattern_{i}", f"Pattern {i}")
        assert len(engine.profile.behavior_patterns) <= PersonalityEngine.MAX_BEHAVIOR_PATTERNS


class TestPersonalityEngineProfileErrors:
    """V-323: PersonalityEngine profile errors use type(e).__name__."""

    def test_load_profile_error(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        import inspect
        source = inspect.getsource(PersonalityEngine._load_profile)
        assert "type(e).__name__" in source

    def test_save_profile_error(self):
        from shadowd.cognition.personality_engine import PersonalityEngine
        import inspect
        source = inspect.getsource(PersonalityEngine._save_profile)
        assert "type(e).__name__" in source


class TestEscalationSafetyPatternLeak:
    """V-330: Escalation SafetyGuardrail doesn't leak pattern."""

    def test_no_pattern_in_message(self):
        from shadowd.cognition.escalation import SafetyGuardrail
        import inspect
        source = inspect.getsource(SafetyGuardrail.check)
        assert "matched pattern" not in source

    def test_safety_escalate_no_pattern(self):
        from shadowd.cognition.escalation import SafetyGuardrail, GuardrailResult

        async def _test():
            g = SafetyGuardrail()
            result, data, reason = await g.check("delete all data and format disk")
            assert result == GuardrailResult.ESCALATE
            assert "matched pattern" not in reason

        asyncio.run(_test())


class TestEscalationAtomicPersist:
    """V-331: Escalation history uses atomic write."""

    def test_persist_uses_tmp_file(self):
        from shadowd.cognition.escalation import EscalationManager
        import inspect
        source = inspect.getsource(EscalationManager._persist_history)
        assert ".tmp" in source or "tmp_file" in source

    def test_persist_error_type_name(self):
        from shadowd.cognition.escalation import EscalationManager
        import inspect
        source = inspect.getsource(EscalationManager._persist_history)
        assert "type(e).__name__" in source


class TestEscalationGuardrailsCap:
    """V-332: Escalation guardrails capped at 20."""

    def test_max_guardrails(self):
        from shadowd.cognition.escalation import EscalationManager, Guardrail
        mgr = EscalationManager()
        assert mgr._max_guardrails == 20

    def test_guardrails_eviction(self):
        from shadowd.cognition.escalation import EscalationManager, Guardrail
        mgr = EscalationManager()
        for i in range(25):
            mgr.add_guardrail(Guardrail(f"test_{i}"))
        assert len(mgr._guardrails) <= mgr._max_guardrails


class TestEscalationQueueCap:
    """V-333: Escalation queue capped at 100."""

    def test_max_queue(self):
        from shadowd.cognition.escalation import EscalationManager
        mgr = EscalationManager()
        assert mgr._max_queue == 100


class TestDarkMemoryImportanceClamp:
    """V-341: DarkMemory importance clamped on store and import."""

    def test_store_clamps_importance(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType

        async def _test():
            mem = DarkMemory()
            atom = await mem.store("test", importance=5.0)
            assert atom.importance == 1.0
            atom2 = await mem.store("test2", importance=-1.0)
            assert atom2.importance == 0.0

        asyncio.run(_test())

    def test_import_clamps_importance(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType

        async def _test():
            mem = DarkMemory()
            data = json.dumps({
                "atoms": [{
                    "atom_id": "test_123",
                    "atom_type": "observation",
                    "tier": "working",
                    "content": "test",
                    "tags": [],
                    "importance": 5.0,
                    "metadata": {},
                }]
            })
            result = await mem.import_data(data)
            assert result["imported"] == 1
            for atom in mem._atoms.values():
                assert atom.importance <= 1.0

        asyncio.run(_test())


class TestDarkMemorySearchNoMutate:
    """V-343: DarkMemory search doesn't mutate atom metadata."""

    def test_search_does_not_add_search_score(self):
        from shadowd.cognition.dark_memory import DarkMemory, AtomType

        async def _test():
            mem = DarkMemory()
            await mem.store("alpha beta gamma", atom_type=AtomType.OBSERVATION)
            for atom in mem._atoms.values():
                assert "search_score" not in atom.metadata
            results = await mem.search("alpha beta", limit=5)
            for atom in mem._atoms.values():
                assert "search_score" not in atom.metadata

        asyncio.run(_test())


if __name__ == "__main__":
    pytest.main([__file__, "-v"])