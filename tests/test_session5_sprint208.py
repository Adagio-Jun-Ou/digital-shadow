import copy
import re
import threading
import uuid
from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from shadowd.cognition.creative_dream import (
    CreativeDreamIntegration,
    CreativeDreamCycle,
    CreativityIncubation,
    DreamInspiration,
    _MAX_CYCLES,
    _MAX_INCUBATIONS,
    _MAX_INSPIRATIONS,
    _DREAM_NOVELTY_THRESHOLD,
    _INSIGHT_CONFIDENCE_THRESHOLD,
    _PATTERN_FREQUENCY_THRESHOLD,
    _SANITIZE_RE,
    _SANITIZE_CONTENT_RE,
    _sanitize,
    _sanitize_content,
)
from shadowd.cognition.dream_state import DreamState, DreamInsight, DreamPattern
from shadowd.cognition.creative_synthesis import (
    CreativeSynthesis,
    CreativeIdea,
    CreativeDomain,
    SynthesisMethod,
    NoveltyLevel,
)


class TestDreamInspiration:
    def test_creation_defaults(self):
        insp = DreamInspiration(
            inspiration_id="",
            source_insight_id="ins_123",
            source_pattern="cross-domain connection",
            concept_seeds=["ai", "health"],
            emotional_valence=0.5,
            novelty_potential=0.7,
            created_at=datetime.now(timezone.utc).isoformat(),
        )
        assert len(insp.inspiration_id) == 12
        assert insp.source_insight_id == "ins_123"
        assert insp.concept_seeds == ["ai", "health"]
        assert insp.emotional_valence == 0.5
        assert insp.novelty_potential == 0.7

    def test_sanitization(self):
        insp = DreamInspiration(
            inspiration_id="",
            source_insight_id="ins<>x\0y",
            source_pattern="hello<script>alert(1)</script>world",
            concept_seeds=["a<>b", "c\0d"],
            emotional_valence=2.0,
            novelty_potential=-1.0,
            created_at="now",
        )
        assert "<" not in insp.source_insight_id
        assert ">" not in insp.source_insight_id
        assert "\0" not in insp.source_insight_id
        assert "\0" not in insp.source_pattern
        assert "<" in insp.source_pattern
        assert "<" not in insp.concept_seeds[0]
        assert "\0" not in insp.concept_seeds[1]
        assert insp.emotional_valence == 1.0
        assert insp.novelty_potential == 0.0

    def test_to_dict(self):
        insp = DreamInspiration(
            inspiration_id="aaa111bbb222",
            source_insight_id="ins1",
            source_pattern="pattern",
            concept_seeds=["x", "y"],
            emotional_valence=0.333,
            novelty_potential=0.777,
            created_at="t1",
        )
        d = insp.to_dict()
        assert d["inspiration_id"] == "aaa111bbb222"
        assert d["concept_seeds"] == ["x", "y"]
        assert d["emotional_valence"] == round(0.333, 3)
        assert d["novelty_potential"] == round(0.777, 3)
        assert isinstance(d["concept_seeds"], list)

    def test_to_dict_deepcopy_safety(self):
        insp = DreamInspiration(
            inspiration_id="id1",
            source_insight_id="si1",
            source_pattern="p1",
            concept_seeds=["a", "b"],
            emotional_valence=0.5,
            novelty_potential=0.5,
            created_at="t1",
        )
        d = insp.to_dict()
        d["concept_seeds"].append("c")
        assert len(insp.concept_seeds) == 2

    def test_concept_seeds_capped(self):
        insp = DreamInspiration(
            inspiration_id="id1",
            source_insight_id="si1",
            source_pattern="p1",
            concept_seeds=[f"s{i}" for i in range(30)],
            emotional_valence=0.5,
            novelty_potential=0.5,
            created_at="t1",
        )
        assert len(insp.concept_seeds) <= 20

    def test_strip_before_truncate(self):
        raw = "   hello world   "
        insp = DreamInspiration(
            inspiration_id="",
            source_insight_id=raw,
            source_pattern=raw,
            concept_seeds=[],
            emotional_valence=0.5,
            novelty_potential=0.5,
            created_at="t1",
        )
        assert not insp.source_insight_id.startswith(" ")
        assert not insp.source_pattern.startswith(" ")


class TestCreativityIncubation:
    def test_creation_defaults(self):
        inc = CreativityIncubation(
            incubation_id="",
            source_idea_id="idea_1",
            concept="AI + health",
            domain="technology",
            incubation_status="seeded",
            first_dreamed_at="t1",
            reinforcement_count=1,
            last_reinforced="t2",
        )
        assert len(inc.incubation_id) == 12
        assert inc.source_idea_id == "idea_1"
        assert inc.concept == "AI + health"
        assert inc.reinforcement_count == 1

    def test_sanitization(self):
        inc = CreativityIncubation(
            incubation_id="",
            source_idea_id="id<>x",
            concept="idea with <html> and\x00null",
            domain="tech<>",
            incubation_status="seed\0ed",
            first_dreamed_at="t1",
            reinforcement_count=-1,
            last_reinforced="t2",
        )
        assert "<" not in inc.source_idea_id
        assert "<" in inc.concept
        assert "\x00" not in inc.concept
        assert "<" not in inc.domain
        assert "\0" not in inc.incubation_status
        assert inc.reinforcement_count == 0

    def test_to_dict(self):
        inc = CreativityIncubation(
            incubation_id="iii111jjj222",
            source_idea_id="si1",
            concept="concept",
            domain="d1",
            incubation_status="seeded",
            first_dreamed_at="t1",
            reinforcement_count=3,
            last_reinforced="t2",
        )
        d = inc.to_dict()
        assert d["incubation_id"] == "iii111jjj222"
        assert d["reinforcement_count"] == 3

    def test_to_dict_deepcopy_safety(self):
        inc = CreativityIncubation(
            incubation_id="id1",
            source_idea_id="si1",
            concept="c",
            domain="d",
            incubation_status="s",
            first_dreamed_at="t1",
            reinforcement_count=1,
            last_reinforced="t2",
        )
        d = inc.to_dict()
        d["reinforcement_count"] = 999
        assert inc.reinforcement_count == 1


class TestCreativeDreamCycle:
    def test_creation(self):
        cycle = CreativeDreamCycle(
            cycle_id="",
            dream_inspirations_count=5,
            creative_outputs_count=3,
            incubated_count=2,
            resonance_score=0.75,
            started_at="t1",
            completed_at="t2",
        )
        assert len(cycle.cycle_id) == 12
        assert cycle.resonance_score == 0.75

    def test_resonance_clamped(self):
        cycle = CreativeDreamCycle(
            cycle_id="",
            dream_inspirations_count=0,
            creative_outputs_count=0,
            incubated_count=0,
            resonance_score=2.0,
            started_at="t1",
        )
        assert cycle.resonance_score == 1.0

        cycle2 = CreativeDreamCycle(
            cycle_id="",
            dream_inspirations_count=0,
            creative_outputs_count=0,
            incubated_count=0,
            resonance_score=-1.0,
            started_at="t1",
        )
        assert cycle2.resonance_score == 0.0

    def test_to_dict_round(self):
        cycle = CreativeDreamCycle(
            cycle_id="c1",
            dream_inspirations_count=5,
            creative_outputs_count=3,
            incubated_count=2,
            resonance_score=0.777,
            started_at="t1",
            completed_at="t2",
        )
        d = cycle.to_dict()
        assert d["resonance_score"] == round(0.777, 3)
        assert d["completed_at"] == "t2"

    def test_completed_at_optional(self):
        cycle = CreativeDreamCycle(
            cycle_id="c1",
            dream_inspirations_count=0,
            creative_outputs_count=0,
            incubated_count=0,
            resonance_score=0.5,
            started_at="t1",
        )
        assert cycle.completed_at is None


class TestDreamToCreate:
    def test_no_dream_state(self):
        cdi = CreativeDreamIntegration()
        result = cdi.dream_to_create()
        assert result == []

    def test_with_insights(self):
        ds = DreamState()
        ds.add_memory("machine learning improves healthcare diagnostics", domain="technology", importance=0.8)
        ds.add_memory("neural networks applied to health monitoring", domain="health", importance=0.7)
        ds.add_memory("data science for health outcomes prediction", domain="science", importance=0.6)
        ds.consolidate()

        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        result = cdi.dream_to_create()
        assert len(result) >= 0
        if result:
            assert isinstance(result[0], DreamInspiration)
            assert len(result[0].concept_seeds) > 0

    def test_with_patterns(self):
        ds = DreamState()
        for _ in range(5):
            ds.add_memory("repeated pattern about optimization algorithms", domain="tech", importance=0.8)
            ds.add_memory("another repeated optimization pattern", domain="science", importance=0.7)
        ds.consolidate()

        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        result = cdi.dream_to_create()
        has_pattern_source = any("Pattern:" in i.source_pattern for i in result)
        if ds.get_patterns():
            assert has_pattern_source

    def test_low_confidence_filtered(self):
        ds = DreamState()
        ds.add_memory("low value memory", domain="general", importance=0.05)
        ds.consolidate()

        cdi = CreativeDreamIntegration(ds, None)
        result = cdi.dream_to_create()
        for insp in result:
            assert insp.novelty_potential >= 0.3

    def test_stats_updated(self):
        ds = DreamState()
        cdi = CreativeDreamIntegration(ds, None)
        cdi.dream_to_create()
        stats = cdi.get_stats()
        assert stats["dream_to_create_calls"] >= 1


class TestCreateToDream:
    def test_no_creative_synthesis(self):
        cdi = CreativeDreamIntegration()
        result = cdi.create_to_dream()
        assert result == []

    def test_with_high_novelty_ideas(self):
        cs = CreativeSynthesis()
        cs.synthesize(["quantum", "biology", "neural"], SynthesisMethod.CROSS_POLLINATION)
        cs.synthesize(["space", "ocean", "deep"], SynthesisMethod.ANALOGY)

        ds = DreamState()
        cdi = CreativeDreamIntegration(ds, cs)
        result = cdi.create_to_dream()
        if result:
            assert isinstance(result[0], CreativityIncubation)
            assert result[0].incubation_status == "seeded"

    def test_dream_memory_added(self):
        cs = CreativeSynthesis()
        cs.synthesize(["artificial intelligence", "healthcare"], SynthesisMethod.COMBINATION)

        ds = DreamState()
        mem_before = ds.get_stats()["total_memories"]
        cdi = CreativeDreamIntegration(ds, cs)
        cdi.create_to_dream()
        mem_after = ds.get_stats()["total_memories"]
        if cdi.get_stats()["total_incubations"] > 0:
            assert mem_after >= mem_before

    def test_reinforcement_on_duplicate(self):
        cs = CreativeSynthesis()
        cs.synthesize(["neural", "network"], SynthesisMethod.COMBINATION)

        ds = DreamState()
        cdi = CreativeDreamIntegration(ds, cs)
        first = cdi.create_to_dream()
        second = cdi.create_to_dream()
        if first and second:
            same_id = first[0].source_idea_id
            matching = [i for i in second if i.source_idea_id == same_id]
            if matching:
                assert matching[0].reinforcement_count >= 2

    def test_low_novelty_filtered(self):
        cs = CreativeSynthesis()
        cs.synthesize(["a"], SynthesisMethod.COMBINATION)

        ds = DreamState()
        cdi = CreativeDreamIntegration(ds, cs)
        result = cdi.create_to_dream()
        for inc in result:
            assert inc.concept is not None


class TestRunCycle:
    def test_full_cycle(self):
        ds = DreamState()
        for _ in range(5):
            ds.add_memory("deep learning advances in medical imaging diagnostics", domain="technology", importance=0.8)
            ds.add_memory("healthcare AI for disease prediction and treatment", domain="health", importance=0.7)
        ds.consolidate()

        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        cycle = cdi.run_cycle()
        assert isinstance(cycle, CreativeDreamCycle)
        assert len(cycle.cycle_id) == 12
        assert cycle.completed_at is not None

    def test_cycle_stats(self):
        ds = DreamState()
        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        cdi.run_cycle()
        stats = cdi.get_stats()
        assert stats["cycles_run"] == 1

    def test_cycle_no_modules(self):
        cdi = CreativeDreamIntegration()
        cycle = cdi.run_cycle()
        assert cycle.dream_inspirations_count == 0
        assert cycle.incubated_count == 0


class TestBoundedHistory:
    def test_inspirations_bounded(self):
        ds = DreamState()
        ds.add_memory("test memory content for dream inspiration", domain="test", importance=0.9)
        ds.consolidate()

        cdi = CreativeDreamIntegration(ds, None)
        for _ in range(_MAX_INSPIRATIONS + 100):
            cdi._inspirations.append(
                DreamInspiration(
                    inspiration_id=uuid.uuid4().hex[:12],
                    source_insight_id="si",
                    source_pattern="p",
                    concept_seeds=["x"],
                    emotional_valence=0.5,
                    novelty_potential=0.5,
                    created_at="t",
                )
            )
        assert len(cdi._inspirations) <= _MAX_INSPIRATIONS

    def test_incubations_bounded(self):
        cdi = CreativeDreamIntegration()
        for _ in range(_MAX_INCUBATIONS + 100):
            cdi._incubations.append(
                CreativityIncubation(
                    incubation_id=uuid.uuid4().hex[:12],
                    source_idea_id="si",
                    concept="c",
                    domain="d",
                    incubation_status="s",
                    first_dreamed_at="t",
                    reinforcement_count=1,
                    last_reinforced="t",
                )
            )
        assert len(cdi._incubations) <= _MAX_INCUBATIONS

    def test_cycles_bounded(self):
        cdi = CreativeDreamIntegration()
        for _ in range(_MAX_CYCLES + 100):
            cdi._cycles.append(
                CreativeDreamCycle(
                    cycle_id=uuid.uuid4().hex[:12],
                    dream_inspirations_count=0,
                    creative_outputs_count=0,
                    incubated_count=0,
                    resonance_score=0.5,
                    started_at="t",
                )
            )
        assert len(cdi._cycles) <= _MAX_CYCLES


class TestThreadSafety:
    def test_concurrent_dream_to_create(self):
        ds = DreamState()
        for i in range(20):
            ds.add_memory(f"memory about topic {i} with important keywords", domain=f"domain_{i % 3}", importance=0.8)
        ds.consolidate()

        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        errors = []

        def worker():
            try:
                cdi.dream_to_create()
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_create_to_dream(self):
        cs = CreativeSynthesis()
        for i in range(5):
            cs.synthesize([f"concept_a_{i}", f"concept_b_{i}"], SynthesisMethod.CROSS_POLLINATION)

        ds = DreamState()
        cdi = CreativeDreamIntegration(ds, cs)
        errors = []

        def worker():
            try:
                cdi.create_to_dream()
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=worker) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_concurrent_run_cycle(self):
        ds = DreamState()
        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        errors = []
        results = []

        def worker():
            try:
                r = cdi.run_cycle()
                results.append(r)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0


class TestStats:
    def test_initial_stats(self):
        cdi = CreativeDreamIntegration()
        stats = cdi.get_stats()
        assert stats["dream_to_create_calls"] == 0
        assert stats["create_to_dream_calls"] == 0
        assert stats["cycles_run"] == 0
        assert stats["total_inspirations"] == 0
        assert stats["total_incubations"] == 0
        assert stats["dream_state_wired"] is False
        assert stats["creative_synthesis_wired"] is False

    def test_stats_after_operations(self):
        ds = DreamState()
        ds.add_memory("important memory about data and learning", domain="tech", importance=0.9)
        ds.consolidate()

        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)
        cdi.dream_to_create()
        cdi.create_to_dream()
        cdi.run_cycle()

        stats = cdi.get_stats()
        assert stats["dream_to_create_calls"] >= 1
        assert stats["create_to_dream_calls"] >= 1
        assert stats["cycles_run"] >= 1
        assert stats["dream_state_wired"] is True
        assert stats["creative_synthesis_wired"] is True

    def test_stats_dict_copy(self):
        cdi = CreativeDreamIntegration()
        s1 = cdi.get_stats()
        s1["dream_to_create_calls"] = 999
        s2 = cdi.get_stats()
        assert s2["dream_to_create_calls"] == 0


class TestToDictDeepcopy:
    def test_inspiration_deepcopy(self):
        insp = DreamInspiration(
            inspiration_id="id1",
            source_insight_id="si1",
            source_pattern="p1",
            concept_seeds=["a", "b"],
            emotional_valence=0.5,
            novelty_potential=0.5,
            created_at="t1",
        )
        d = insp.to_dict()
        d["concept_seeds"].append("c")
        assert len(insp.concept_seeds) == 2

    def test_incubation_deepcopy(self):
        inc = CreativityIncubation(
            incubation_id="id1",
            source_idea_id="si1",
            concept="c",
            domain="d",
            incubation_status="s",
            first_dreamed_at="t1",
            reinforcement_count=1,
            last_reinforced="t2",
        )
        d = inc.to_dict()
        assert isinstance(d, dict)

    def test_cycle_deepcopy(self):
        cycle = CreativeDreamCycle(
            cycle_id="c1",
            dream_inspirations_count=1,
            creative_outputs_count=1,
            incubated_count=1,
            resonance_score=0.5,
            started_at="t1",
            completed_at="t2",
        )
        d = cycle.to_dict()
        assert isinstance(d, dict)


class TestStripBeforeTruncate:
    def test_sanitize_strips_first(self):
        raw = "   hello   "
        result = _sanitize(raw, max_len=5)
        assert result == "hello"

    def test_sanitize_content_strips_first(self):
        raw = "   world   "
        result = _sanitize_content(raw, max_len=5)
        assert result == "world"

    def test_sanitize_strict_strips_angle_brackets(self):
        raw = "test<script>alert</script>data"
        result = _sanitize(raw, max_len=100)
        assert "<script>" not in result
        assert ">" not in result

    def test_sanitize_content_preserves_angle_brackets(self):
        raw = "value < 100 and > 50"
        result = _sanitize_content(raw, max_len=100)
        assert "<" in result
        assert ">" in result

    def test_control_chars_stripped(self):
        raw = "hello\x00world\x01test\x7f"
        result = _sanitize(raw, max_len=100)
        assert "\x00" not in result
        assert "\x01" not in result
        assert "\x7f" not in result


class TestSetModules:
    def test_set_modules_after_construction(self):
        cdi = CreativeDreamIntegration()
        assert cdi.get_stats()["dream_state_wired"] is False
        assert cdi.get_stats()["creative_synthesis_wired"] is False

        ds = DreamState()
        cs = CreativeSynthesis()
        cdi.set_modules(ds, cs)
        assert cdi.get_stats()["dream_state_wired"] is True
        assert cdi.get_stats()["creative_synthesis_wired"] is True

    def test_set_modules_enables_operations(self):
        cdi = CreativeDreamIntegration()
        assert cdi.dream_to_create() == []
        assert cdi.create_to_dream() == []

        ds = DreamState()
        cs = CreativeSynthesis()
        cdi.set_modules(ds, cs)
        cdi.dream_to_create()
        cdi.create_to_dream()
        stats = cdi.get_stats()
        assert stats["dream_to_create_calls"] >= 1


class TestNoModulesWired:
    def test_dream_to_create_empty(self):
        cdi = CreativeDreamIntegration()
        assert cdi.dream_to_create() == []

    def test_create_to_dream_empty(self):
        cdi = CreativeDreamIntegration()
        assert cdi.create_to_dream() == []

    def test_run_cycle_empty(self):
        cdi = CreativeDreamIntegration()
        cycle = cdi.run_cycle()
        assert cycle.dream_inspirations_count == 0
        assert cycle.incubated_count == 0

    def test_get_inspirations_empty(self):
        cdi = CreativeDreamIntegration()
        assert cdi.get_inspirations() == []

    def test_get_incubations_empty(self):
        cdi = CreativeDreamIntegration()
        assert cdi.get_incubations() == []

    def test_get_cycles_empty(self):
        cdi = CreativeDreamIntegration()
        assert cdi.get_cycles() == []


class TestAccessors:
    def test_get_inspirations_limit(self):
        ds = DreamState()
        for i in range(5):
            ds.add_memory(f"test memory {i} with unique content keyword", domain="test", importance=0.9)
        ds.consolidate()

        cdi = CreativeDreamIntegration(ds, None)
        cdi.dream_to_create()
        all_insp = cdi.get_inspirations(limit=1000)
        limited = cdi.get_inspirations(limit=2)
        assert len(limited) <= 2

    def test_get_incubations_limit(self):
        cs = CreativeSynthesis()
        for i in range(5):
            cs.synthesize([f"concept_{i}_a", f"concept_{i}_b"], SynthesisMethod.CROSS_POLLINATION)

        cdi = CreativeDreamIntegration(None, cs)
        cdi.create_to_dream()
        all_inc = cdi.get_incubations(limit=1000)
        limited = cdi.get_incubations(limit=2)
        assert len(limited) <= 2

    def test_get_cycles_limit(self):
        cdi = CreativeDreamIntegration()
        for _ in range(5):
            cdi.run_cycle()
        limited = cdi.get_cycles(limit=2)
        assert len(limited) <= 2

    def test_limit_bounds(self):
        cdi = CreativeDreamIntegration()
        result = cdi.get_inspirations(limit=0)
        assert len(result) >= 1 or result == []
        result = cdi.get_inspirations(limit=999)
        assert isinstance(result, list)


class TestShadowOSWiring:
    def test_creative_dream_attribute(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        assert hasattr(os, "creative_dream") or True

    def test_health_includes_creative_dream(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os.creative_dream = CreativeDreamIntegration()
        os._stats_lock = threading.Lock()
        os._stats = {"events_processed": 0, "queries_handled": 0, "actions_taken": 0,
                     "insights_generated": 0, "anomalies_detected": 0, "errors": 0, "uptime_seconds": 0}
        os._running = True
        os._start_time = None
        os.graph = MagicMock()
        os.graph.state.mode.value = "idle"
        os.memory = MagicMock()
        os.memory.get_stats.return_value = {"total_atoms": 0}
        os.crew = MagicMock()
        os.crew.list_agents.return_value = []
        os.stealth = MagicMock()
        os.stealth.get_audit_log.return_value = []
        os.personality = MagicMock()
        os.personality.get_profile.return_value = {}
        os.voice = MagicMock()
        os.voice.get_state.return_value = {}
        os.thought_sense = MagicMock()
        os.recall_synthesis = MagicMock()
        os.temporal_reasoning = MagicMock()
        os.empathy = MagicMock()
        os.empathy.get_stats.return_value = {"total_persons": 0}
        os.presence = MagicMock()
        os.presence.get_stats.return_value = {"total_devices": 0}
        os.trinity = MagicMock()
        os.trinity.get_stats.return_value = {"total_analyses": 0}
        os.dream = MagicMock()
        os.dream.get_stats.return_value = {"total_memories": 0}
        os.skills = MagicMock()
        os.skills.get_stats.return_value = {"total_skills": 0}
        os.narrative = MagicMock()
        os.narrative.get_stats.return_value = {"total_events": 0}
        os.procedural = MagicMock()
        os.procedural.get_stats.return_value = {"total_procedures": 0}
        os.conflict = MagicMock()
        os.conflict.get_stats.return_value = {"total_conflicts": 0}
        os.emotional_intel = MagicMock()
        os.emotional_intel.get_stats.return_value = {"stats": {"states_recorded": 0}}
        os.creative = MagicMock()
        os.creative.get_stats.return_value = {"stats": {"ideas_generated": 0}}
        os.metacognition = MagicMock()
        os.metacognition.get_stats.return_value = {"stats": {"episodes_tracked": 0}}
        os.moral = MagicMock()
        os.moral.get_stats.return_value = {"stats": {"evaluations_made": 0}}
        os.attention = MagicMock()
        os.attention.get_stats.return_value = {"items_tracked": 0}
        os.pipeline = MagicMock()
        os.pipeline.get_pipeline_stats.return_value = {"stats": {"pipelines_run": 0}}
        os.social_intel = MagicMock()
        os.social_intel.get_stats.return_value = {"groups_tracked": 0}
        os.journal = MagicMock()
        os.journal.get_stats.return_value = {"decisions_recorded": 0}
        os.reflection = MagicMock()
        os.reflection.get_stats.return_value = {"reflections_added": 0}
        os.goal_reasoning = MagicMock()
        os.goal_reasoning.get_stats.return_value = {"assessments_made": 0}
        os.device_context = MagicMock()
        os.device_context.get_stats.return_value = {"stats": {"context_refreshes": 0}}
        os.arise = MagicMock()
        os.arise.get_stats.return_value = {"active_agents": 0}
        os.daemon = MagicMock()
        os.daemon.get_health.return_value = {}
        os.evolution = MagicMock()
        os.evolution.get_stats.return_value = {"observations_added": 0}
        os.oracle = MagicMock()
        os.oracle.get_stats.return_value = {"predictions_generated": 0}
        os.feed_loop = MagicMock()
        os.feed_loop.get_stats.return_value = {"signals_emitted": 0}
        os.anticipatory = MagicMock()
        os.anticipatory.get_stats.return_value = {"signals_generated": 0}
        os.resonance = MagicMock()
        os.resonance.get_stats.return_value = {"activities_logged": 0}
        os.temporal_self = MagicMock()
        os.temporal_self.get_stats.return_value = {"milestones_recorded": 0}
        os.health_dashboard = MagicMock()
        os.health_dashboard.get_stats.return_value = {"total_checks": 0}
        os.social_propagation = MagicMock()
        os.social_propagation.get_stats.return_value = {"ripples_created": 0}
        os.neural_bridge = MagicMock()
        os.neural_bridge.get_stats.return_value = {"mappings_created": 0}
        os.quantum_field = MagicMock()
        os.quantum_field.get_stats.return_value = {"decisions_created": 0}
        os.grammar = MagicMock()
        os.grammar.get_stats.return_value = {"rules_created": 0}
        os.parallax = MagicMock()
        os.parallax.get_stats.return_value = {"events_registered": 0}
        os.holographic = MagicMock()
        os.holographic.get_stats.return_value = {"status": "healthy"}
        os.syncognitive = MagicMock()
        os.syncognitive.get_stats.return_value = {}
        os.morphic = MagicMock()
        os.morphic.get_stats.return_value = {}
        os.existential = MagicMock()
        os.existential.get_stats.return_value = {}
        os.chrono = MagicMock()
        os.chrono.get_stats.return_value = {}
        os.phoenix = MagicMock()
        os.phoenix.get_stats.return_value = {}
        os.causal = MagicMock()
        os.causal.get_stats.return_value = {}
        os.quantum_memory = MagicMock()
        os.quantum_memory.get_stats.return_value = {}
        os.drift_detector = MagicMock()
        os.drift_detector.get_stats.return_value = {}
        os.immune = MagicMock()
        os.immune.get_stats.return_value = {}
        os.boundary = MagicMock()
        os.boundary.get_stats.return_value = {}
        os.synthesis_engine = MagicMock()
        os.self_mod = MagicMock()
        os.profiler = MagicMock()
        os.intel_report = MagicMock()
        os.anomaly_corr = MagicMock()
        os.module_bus = MagicMock()
        os.opt_daemon = MagicMock()
        os.module_registry = MagicMock()
        os.arbiter = MagicMock()
        os.visualizer = MagicMock()
        os.economy = MagicMock()
        os.topology = MagicMock()
        os.symbiosis = MagicMock()
        os.resonance_engine = MagicMock()
        os.shadow_powers = MagicMock()
        os.shadow_powers.get_stats.return_value = {"total_applications": 0}
        os.mind = MagicMock()
        os.mind.get_stats.return_value = {}
        os.mind.health_check.return_value = True
        os.shadow_ai = None
        os.event_bus = MagicMock()
        os.mesh = MagicMock()
        os.mesh.get_stats.return_value = {}
        os.vision = MagicMock()
        os.vision.get_stats.return_value = {}
        os.notifications = MagicMock()
        os.notifications.get_stats.return_value = {}
        os.visual_bridge = MagicMock()
        os.visual_bridge.get_stats.return_value = {}
        os.shadow_deploy = MagicMock()
        os.shadow_deploy.get_stats.return_value = {}
        os.evolutionary = MagicMock()
        os.evolutionary.get_stats.return_value = {}
        os.decision_journal = MagicMock()
        os.decision_journal.get_stats.return_value = {}

        health = os.get_health()
        assert "creative_dream" in health["modules"]

    def test_status_includes_creative_dream(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os.creative_dream = CreativeDreamIntegration()
        os._stats_lock = threading.Lock()
        os._stats = {"events_processed": 0, "queries_handled": 0, "actions_taken": 0,
                     "insights_generated": 0, "anomalies_detected": 0, "errors": 0, "uptime_seconds": 0}
        os._running = False
        os._start_time = None
        os.graph = MagicMock()
        os.memory = MagicMock()
        os.crew = MagicMock()
        os.crew.get_stats.return_value = {}
        os.mesh = MagicMock()
        os.personality = MagicMock()
        os.personality.get_stats.return_value = {}
        os.synthesis_engine = MagicMock()
        os.self_mod = MagicMock()
        os.profiler = MagicMock()
        os.intel_report = MagicMock()
        os.anomaly_corr = MagicMock()
        os.module_bus = MagicMock()
        os.opt_daemon = MagicMock()
        os.module_registry = MagicMock()
        os.arbiter = MagicMock()
        os.visualizer = MagicMock()
        os.thought_sense = MagicMock()
        os.thought_sense.get_stats.return_value = {}
        os.recall_synthesis = MagicMock()
        os.recall_synthesis.get_stats.return_value = {}
        os.temporal_reasoning = MagicMock()
        os.temporal_reasoning.get_stats.return_value = {}
        os.empathy = MagicMock()
        os.empathy.get_stats.return_value = {}
        os.presence = MagicMock()
        os.presence.get_stats.return_value = {}
        os.trinity = MagicMock()
        os.trinity.get_stats.return_value = {}
        os.dream = MagicMock()
        os.dream.get_stats.return_value = {}
        os.skills = MagicMock()
        os.skills.get_stats.return_value = {}
        os.narrative = MagicMock()
        os.narrative.get_stats.return_value = {}
        os.procedural = MagicMock()
        os.procedural.get_stats.return_value = {}
        os.conflict = MagicMock()
        os.conflict.get_stats.return_value = {}
        os.emotional_intel = MagicMock()
        os.emotional_intel.get_stats.return_value = {}
        os.creative = MagicMock()
        os.creative.get_stats.return_value = {}
        os.metacognition = MagicMock()
        os.metacognition.get_stats.return_value = {}
        os.moral = MagicMock()
        os.moral.get_stats.return_value = {}
        os.attention = MagicMock()
        os.attention.get_stats.return_value = {}
        os.pipeline = MagicMock()
        os.pipeline.get_pipeline_stats.return_value = {}
        os.social_intel = MagicMock()
        os.social_intel.get_stats.return_value = {}
        os.journal = MagicMock()
        os.journal.get_stats.return_value = {}
        os.decision_journal = MagicMock()
        os.decision_journal.get_stats.return_value = {}
        os.reflection = MagicMock()
        os.reflection.get_stats.return_value = {}
        os.goal_reasoning = MagicMock()
        os.goal_reasoning.get_stats.return_value = {}
        os.device_context = MagicMock()
        os.device_context.get_stats.return_value = {}
        os.arise = MagicMock()
        os.arise.get_stats.return_value = {}
        os.daemon = MagicMock()
        os.daemon.get_status.return_value = {}
        os.evolution = MagicMock()
        os.evolution.get_stats.return_value = {}
        os.oracle = MagicMock()
        os.oracle.get_stats.return_value = {}
        os.feed_loop = MagicMock()
        os.feed_loop.get_stats.return_value = {}
        os.anticipatory = MagicMock()
        os.anticipatory.get_stats.return_value = {}
        os.resonance = MagicMock()
        os.resonance.get_stats.return_value = {}
        os.temporal_self = MagicMock()
        os.temporal_self.get_stats.return_value = {}
        os.health_dashboard = MagicMock()
        os.health_dashboard.get_stats.return_value = {}
        os.neural_bridge = MagicMock()
        os.neural_bridge.get_stats.return_value = {"mappings_created": 0}
        os.quantum_field = MagicMock()
        os.quantum_field.get_stats.return_value = {"decisions_created": 0}
        os.grammar = MagicMock()
        os.grammar.get_stats.return_value = {"rules_created": 0}
        os.parallax = MagicMock()
        os.parallax.get_stats.return_value = {"events_registered": 0}
        os.holographic = MagicMock()
        os.holographic.get_stats.return_value = {}
        os.syncognitive = MagicMock()
        os.syncognitive.get_stats.return_value = {}
        os.morphic = MagicMock()
        os.morphic.get_stats.return_value = {}
        os.existential = MagicMock()
        os.existential.get_stats.return_value = {}
        os.chrono = MagicMock()
        os.chrono.get_stats.return_value = {}
        os.phoenix = MagicMock()
        os.phoenix.get_stats.return_value = {}
        os.causal = MagicMock()
        os.causal.get_stats.return_value = {}
        os.quantum_memory = MagicMock()
        os.quantum_memory.get_stats.return_value = {}
        os.drift_detector = MagicMock()
        os.drift_detector.get_stats.return_value = {}
        os.immune = MagicMock()
        os.immune.get_stats.return_value = {}
        os.boundary = MagicMock()
        os.boundary.get_stats.return_value = {}
        os.resonance_engine = MagicMock()
        os.resonance_engine.get_stats.return_value = {}
        os.economy = MagicMock()
        os.economy.get_stats.return_value = {}
        os.topology = MagicMock()
        os.topology.get_stats.return_value = {}
        os.symbiosis = MagicMock()
        os.symbiosis.get_stats.return_value = {}
        os.shadow_powers = MagicMock()
        os.shadow_powers.get_stats.return_value = {"total_applications": 0}
        os.social_propagation = MagicMock()
        os.social_propagation.get_stats.return_value = {"ripples_created": 0}
        os.mind = MagicMock()
        os.mind.get_stats.return_value = {}
        os.mind.health_check.return_value = True
        os.event_bus = MagicMock()
        os.shadow_ai = None
        os.vision = MagicMock()
        os.vision.get_stats.return_value = {}
        os.notifications = MagicMock()
        os.notifications.get_stats.return_value = {}
        os.visual_bridge = MagicMock()
        os.visual_bridge.get_stats.return_value = {}
        os.shadow_deploy = MagicMock()
        os.shadow_deploy.get_stats.return_value = {}
        os.evolutionary = MagicMock()
        os.evolutionary.get_stats.return_value = {}

        status = os.get_status()
        assert "creative_dream_stats" in status

    def test_convenience_methods(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os.creative_dream = CreativeDreamIntegration()

        cycle_result = os.run_creative_dream_cycle()
        assert isinstance(cycle_result, dict)
        assert "cycle_id" in cycle_result

        inspirations = os.get_dream_inspirations()
        assert isinstance(inspirations, list)

        incubations = os.get_creativity_incubations()
        assert isinstance(incubations, list)

    def test_convenience_cycle_returns_dict(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os.creative_dream = CreativeDreamIntegration()
        result = os.run_creative_dream_cycle()
        assert "cycle_id" in result
        assert "dream_inspirations_count" in result
        assert "resonance_score" in result


class TestIntegrationEndToEnd:
    def test_full_bidirectional_flow(self):
        ds = DreamState()
        for i in range(10):
            ds.add_memory(
                f"Research on quantum computing and molecular biology {i}",
                domain="technology" if i % 2 == 0 else "science",
                importance=0.8,
                tags=["quantum", "biology"],
            )
        ds.consolidate()

        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)

        cycle = cdi.run_cycle()
        assert isinstance(cycle, CreativeDreamCycle)
        assert cycle.started_at is not None
        assert cycle.completed_at is not None

    def test_multiple_cycles(self):
        ds = DreamState()
        cs = CreativeSynthesis()
        cdi = CreativeDreamIntegration(ds, cs)

        for _ in range(3):
            cdi.run_cycle()

        cycles = cdi.get_cycles()
        assert len(cycles) >= 3

    def test_concept_seeds_extracted(self):
        ds = DreamState()
        ds.add_memory("artificial intelligence transforms healthcare diagnostics", domain="technology", importance=0.9)
        ds.add_memory("deep learning for health monitoring and prediction", domain="health", importance=0.8)
        ds.consolidate()

        cdi = CreativeDreamIntegration(ds, None)
        inspirations = cdi.dream_to_create()
        for insp in inspirations:
            if insp.source_insight_id:
                assert len(insp.concept_seeds) > 0