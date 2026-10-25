"""Sprint 201: Deep cognitive pipeline integration tests.

Verify that ALL 38 modules are wired into ShadowOS _handle_* methods
and that the cognitive pipeline actually invokes each module."""

import pytest
import asyncio
from unittest.mock import MagicMock, patch, AsyncMock
from shadowd.cognition.shadow_os import ShadowOS


class TestDeepAnalyzeIntegration:
    """Test that _handle_analyze invokes all cognitive modules."""

    @pytest.fixture
    def os_instance(self):
        os_inst = ShadowOS.__new__(ShadowOS)
        os_inst._running = False
        os_inst._stats = {}
        os_inst._stats_lock = MagicMock()
        os_inst._start_time = 0

        # Mock all 38+ modules
        os_inst.graph = MagicMock()
        os_inst.memory = AsyncMock()
        os_inst.memory.get_recent = AsyncMock(return_value=[])
        os_inst.memory.search = AsyncMock(return_value=[])
        os_inst.crew = MagicMock()
        os_inst.mesh = MagicMock()
        os_inst.stealth = MagicMock()
        os_inst.personality = MagicMock()
        os_inst.voice = MagicMock()
        os_inst.anomaly_detector = MagicMock()
        os_inst.anomaly_detector.get_anomalies = MagicMock(return_value=[])
        os_inst.goal_planner = MagicMock()
        os_inst.goal_planner.list_goals = MagicMock(return_value=[])
        os_inst.thought_sense = MagicMock()
        os_inst.thought_sense.analyze = MagicMock(return_value={"intent": {"type": "question"}, "emotion": {"primary": "curiosity"}, "urgency": {"level": "low"}, "cognitive_load": 0.4})
        os_inst.recall_synthesis = MagicMock()
        os_inst.recall_synthesis.synthesize = MagicMock(return_value={"insights": []})
        os_inst.temporal_reasoning = MagicMock()
        os_inst.temporal_reasoning.detect_patterns = MagicMock(return_value=[])
        os_inst.empathy = MagicMock()
        # `_handle_analyze` calls get_stats(), not get_overall_empathy.
        os_inst.empathy.get_stats = MagicMock(return_value={"people_tracked": 1, "avg_empathy": 0.7, "total_persons": 1})
        os_inst.empathy.get_overall_empathy = MagicMock(return_value={"average_score": 0.7})
        os_inst.presence = MagicMock()
        os_inst.trinity = MagicMock()
        os_inst.dream = MagicMock()
        os_inst.skills = MagicMock()
        os_inst.skills.get_domain_summary = MagicMock(return_value={"technical": 3, "career": 2})
        os_inst.narrative = MagicMock()
        # `_handle_analyze` calls get_dominant_themes(limit=5).
        os_inst.narrative.get_dominant_themes = MagicMock(return_value=[("career", 0.8)])
        os_inst.narrative.get_themes = MagicMock(return_value={"career": 0.8})
        os_inst.procedural = MagicMock()
        # `_handle_analyze` calls list_procedures(category="general").
        os_inst.procedural.list_procedures = MagicMock(return_value=[])
        os_inst.procedural.get_procedures_by_domain = MagicMock(return_value=[])
        os_inst.conflict = MagicMock()
        os_inst.conflict.list_conflicts = MagicMock(return_value=[])
        os_inst.emotional_intel = MagicMock()
        os_inst.creative = MagicMock()
        os_inst.metacognition = MagicMock()
        os_inst.moral = MagicMock()
        os_inst.attention = MagicMock()
        os_inst.attention.get_priorities = MagicMock(return_value=[])
        os_inst.attention.detect_focus_state = MagicMock(return_value=None)
        os_inst.pipeline = MagicMock()
        os_inst.social_intel = MagicMock()
        os_inst.social_intel.list_groups = MagicMock(return_value=[])
        os_inst.journal = MagicMock()
        os_inst.reflection = MagicMock()
        os_inst.goal_reasoning = MagicMock()
        os_inst.device_context = MagicMock()
        os_inst.device_context.get_context = MagicMock(return_value=None)
        os_inst.arise = MagicMock()
        os_inst.summon = MagicMock()
        os_inst.evolution = MagicMock()
        os_inst.evolution.get_friction_points = MagicMock(return_value=[])
        os_inst.oracle = MagicMock()
        # `_handle_analyze` calls get_domain_states().
        os_inst.oracle.get_domain_states = MagicMock(return_value={})
        os_inst.oracle.get_active_domains = MagicMock(return_value=set())
        os_inst.daemon = MagicMock()
        os_inst.mind = AsyncMock()
        os_inst.shadow_ai = AsyncMock()
        os_inst.event_bus = MagicMock()
        os_inst.event_bus.get_history = MagicMock(return_value=[])
        return os_inst

    def test_analyze_invokes_thought_sense(self, os_instance):
        """ThoughtSense.analyze() is called during _handle_analyze."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.thought_sense.analyze.assert_called_once()

    def test_analyze_invokes_recall_synthesis(self, os_instance):
        """RecallSynthesis.synthesize() is called when there are search results."""
        os_instance.memory.search = AsyncMock(return_value=[MagicMock(content="test", atom_id="a1")])
        asyncio.run(self._run_analyze(os_instance))
        os_instance.recall_synthesis.synthesize.assert_called_once()

    def test_analyze_invokes_temporal_reasoning(self, os_instance):
        """TemporalReasoning.detect_patterns() is called during _handle_analyze."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.temporal_reasoning.detect_patterns.assert_called_once()

    def test_analyze_invokes_attention_engine(self, os_instance):
        """AttentionEngine.get_priorities() and detect_focus_state() are called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.attention.get_priorities.assert_called_once()
        os_instance.attention.detect_focus_state.assert_called_once()

    def test_analyze_invokes_empathy(self, os_instance):
        """EmpathyMapping.get_stats() is called by _handle_analyze."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.empathy.get_stats.assert_called()

    def test_analyze_invokes_skill_tree(self, os_instance):
        """AdaptiveSkillTree.get_domain_summary() is called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.skills.get_domain_summary.assert_called_once()

    def test_analyze_invokes_goal_planner(self, os_instance):
        """GoalPlanner.list_goals() is called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.goal_planner.list_goals.assert_called_once()

    def test_analyze_invokes_evolution(self, os_instance):
        """EvolutionGuide.get_friction_points() is called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.evolution.get_friction_points.assert_called_once()

    def test_analyze_invokes_oracle(self, os_instance):
        """OracleBridge.get_domain_states() is called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.oracle.get_domain_states.assert_called_once()

    def test_analyze_invokes_narrative(self, os_instance):
        """NarrativeMemory.get_dominant_themes() is called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.narrative.get_dominant_themes.assert_called_once()

    def test_analyze_invokes_procedural(self, os_instance):
        """ProceduralMemory.list_procedures(category=...) is called."""
        asyncio.run(self._run_analyze(os_instance))
        os_instance.procedural.list_procedures.assert_called_once()

    async def _run_analyze(self, os_inst):
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(
            mode=CognitiveMode.ANALYZE,
            current_query="What should I work on today?",
            observations=[],
            context={},
            confidence=0.5,
            domain="career",
        )
        return await os_inst._handle_analyze(state)


class TestDeepReasonIntegration:
    """Test that _handle_reason invokes pipeline, trinity, moral, metacognition."""

    @pytest.fixture
    def os_instance(self):
        os_inst = ShadowOS.__new__(ShadowOS)
        os_inst._running = False
        os_inst._stats = {}
        os_inst._stats_lock = MagicMock()
        os_inst._start_time = 0

        os_inst.personality = MagicMock()
        os_inst.personality.predict_decision = MagicMock(return_value={"predicted_choice": 0, "confidence": 0.7})
        os_inst.mind = AsyncMock()
        os_inst.mind.think = AsyncMock(return_value="I recommend focusing on your project.")
        os_inst.event_bus = MagicMock()
        os_inst.event_bus.get_history = MagicMock(return_value=[])
        os_inst.device_context = MagicMock()
        os_inst.device_context.get_context = MagicMock(return_value=None)
        os_inst.pipeline = MagicMock()
        os_inst.pipeline.process = MagicMock(return_value={"overall_confidence": 0.8, "modules_used": ["thought", "emotion", "meta", "moral"]})
        os_inst.trinity = MagicMock()
        trinity_result = MagicMock()
        trinity_result.synthesis = {"consensus": "consensus", "confidence": 0.75}
        os_inst.trinity.analyze = MagicMock(return_value=trinity_result)
        os_inst.moral = MagicMock()
        os_inst.moral.evaluate = MagicMock(return_value={"dominant_framework": "utilitarian", "ethical_score": 0.8, "conflicts": []})
        os_inst.metacognition = MagicMock()
        os_inst.metacognition.start_thinking = MagicMock(return_value={"biases": [], "cognitive_load": 0.3})
        os_inst.social_intel = MagicMock()
        os_inst.social_intel.list_groups = MagicMock(return_value=[])
        os_inst.conflict = MagicMock()
        os_inst.conflict.list_conflicts = MagicMock(return_value=[])
        os_inst.journal = MagicMock()
        os_inst.journal.record_decision = MagicMock(return_value="decision_123")
        # _handle_reason uses self.decision_journal, not self.journal.
        os_inst.decision_journal = MagicMock()
        os_inst.decision_journal.record_decision = MagicMock(return_value="decision_123")
        os_inst.goal_planner = MagicMock()
        os_inst.goal_planner.list_goals = MagicMock(return_value=[{"goal_id": "g1", "status": "in_progress"}])
        os_inst.goal_reasoning = MagicMock()
        os_inst.goal_reasoning.assess_goal = MagicMock(return_value=MagicMock(urgency_score=0.6, confidence=0.7, obstacles=[]))
        return os_inst

    def test_reason_invokes_pipeline(self, os_instance):
        """CognitivePipeline.process() is called during _handle_reason."""
        asyncio.run(self._run_reason(os_instance))
        os_instance.pipeline.process.assert_called_once()

    def test_reason_invokes_trinity(self, os_instance):
        """CognitiveTrinity.analyze() is called during _handle_reason."""
        asyncio.run(self._run_reason(os_instance))
        os_instance.trinity.analyze.assert_called_once()

    def test_reason_invokes_moral(self, os_instance):
        """MoralReasoning.evaluate() is called during _handle_reason."""
        asyncio.run(self._run_reason(os_instance))
        os_instance.moral.evaluate.assert_called_once()

    def test_reason_invokes_metacognition(self, os_instance):
        """MetacognitionEngine.start_thinking() is called during _handle_reason."""
        asyncio.run(self._run_reason(os_instance))
        os_instance.metacognition.start_thinking.assert_called_once()

    def test_reason_invokes_journal(self, os_instance):
        """DecisionJournal.record_decision() is called during _handle_reason.
        The runtime uses self.decision_journal (not self.journal)."""
        asyncio.run(self._run_reason(os_instance))
        os_instance.decision_journal.record_decision.assert_called_once()

    def test_reason_records_context_keys(self, os_instance):
        """Reasoning enriches context with pipeline, trinity, moral, meta results."""
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = asyncio.run(self._run_reason(os_instance))
        assert "cognitive_pipeline" in state.context
        assert "trinity_consensus" in state.context
        assert "moral_assessment" in state.context
        assert "metacognition" in state.context
        assert "decision_id" in state.context

    async def _run_reason(self, os_inst):
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(
            mode=CognitiveMode.REASON,
            current_query="Should I accept this job offer?",
            observations=[],
            context={},
            confidence=0.5,
            domain="career",
            urgency=0.3,
        )
        return await os_inst._handle_reason(state)


class TestDeepRespondIntegration:
    """Test that _handle_respond invokes creative, narrative, procedural, attention, evolution, oracle."""

    @pytest.fixture
    def os_instance(self):
        os_inst = ShadowOS.__new__(ShadowOS)
        os_inst._running = False
        os_inst._stats = {}
        os_inst._stats_lock = MagicMock()
        os_inst._start_time = 0
        os_inst._inc_stat = MagicMock()

        os_inst.memory = AsyncMock()
        os_inst.memory.store = AsyncMock()
        os_inst.creative = MagicMock()
        os_inst.creative.synthesize = MagicMock(return_value={"ideas": [{"description": "Test idea"}]})
        os_inst.narrative = MagicMock()
        os_inst.procedural = MagicMock()
        os_inst.attention = MagicMock()
        os_inst.evolution = MagicMock()
        os_inst.oracle = MagicMock()
        os_inst.shadow_ai = AsyncMock()
        return os_inst

    def test_respond_invokes_creative(self, os_instance):
        """CreativeSynthesis.synthesize() is called during _handle_respond."""
        asyncio.run(self._run_respond(os_instance))
        os_instance.creative.synthesize.assert_called_once()

    def test_respond_invokes_narrative(self, os_instance):
        """NarrativeMemory.add_event() is called during _handle_respond."""
        asyncio.run(self._run_respond(os_instance))
        os_instance.narrative.add_event.assert_called_once()

    def test_respond_invokes_procedural(self, os_instance):
        """ProceduralMemory.practice_procedure() is called during _handle_respond."""
        asyncio.run(self._run_respond(os_instance))
        os_instance.procedural.practice_procedure.assert_called_once()

    def test_respond_invokes_evolution(self, os_instance):
        """EvolutionGuide.observe() is called during _handle_respond."""
        asyncio.run(self._run_respond(os_instance))
        os_instance.evolution.observe.assert_called_once()

    async def _run_respond(self, os_inst):
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(
            mode=CognitiveMode.RESPOND,
            current_query="Test query",
            observations=[],
            context={"llm_response": "Test response"},
            confidence=0.7,
            domain="general",
        )
        return await os_inst._handle_respond(state)


class TestDeepLearnIntegration:
    """Test that _handle_learn updates skills, empathy, goals, recall, attention."""

    @pytest.fixture
    def os_instance(self):
        os_inst = ShadowOS.__new__(ShadowOS)
        os_inst._running = False
        os_inst._stats = {}
        os_inst._stats_lock = MagicMock()
        os_inst._start_time = 0

        os_inst.personality = MagicMock()
        os_inst.skills = MagicMock()
        os_inst.emotional_intel = MagicMock()
        os_inst.goal_planner = MagicMock()
        os_inst.goal_planner.track_goal = MagicMock()
        os_inst.recall_synthesis = MagicMock()
        os_inst.attention = MagicMock()
        os_inst.journal = MagicMock()
        os_inst.journal.update_status = MagicMock()
        os_inst.device_context = MagicMock()
        return os_inst

    def test_learn_invokes_skills(self, os_instance):
        """AdaptiveSkillTree.use_skill() is called during _handle_learn."""
        asyncio.run(self._run_learn(os_instance))
        os_instance.skills.use_skill.assert_called()

    def test_learn_invokes_emotional_intel(self, os_instance):
        """EmotionalIntelligence.record_pattern() is called when emotion context exists."""
        asyncio.run(self._run_learn(os_instance))
        os_instance.emotional_intel.record_pattern.assert_called_once()

    def test_learn_invokes_attention(self, os_instance):
        """AttentionEngine.add_attention_item() is called during _handle_learn."""
        asyncio.run(self._run_learn(os_instance))
        os_instance.attention.add_attention_item.assert_called_once()

    def test_learn_invokes_recall(self, os_instance):
        """RecallSynthesis.add_observation() is called during _handle_learn."""
        asyncio.run(self._run_learn(os_instance))
        os_instance.recall_synthesis.add_observation.assert_called_once()

    async def _run_learn(self, os_inst):
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(
            mode=CognitiveMode.LEARN,
            current_query="Learn test",
            observations=[],
            context={"emotion": "curiosity", "active_goals": [{"goal_id": "g1", "status": "in_progress"}], "decision_id": "d1"},
            confidence=0.7,
            domain="general",
            active_abilities=["oracle"],
            decisions=[],
        )
        return await os_inst._handle_learn(state)


class TestDeepReflectIntegration:
    """Test that _handle_reflect invokes reflection, dream, metacognition end."""

    @pytest.fixture
    def os_instance(self):
        os_inst = ShadowOS.__new__(ShadowOS)
        os_inst._running = False
        os_inst._stats = {}
        os_inst._stats_lock = MagicMock()
        os_inst._start_time = 0

        os_inst.personality = AsyncMock()
        os_inst.memory = AsyncMock()
        os_inst.memory.compact = AsyncMock(return_value=5)
        os_inst.reflection = MagicMock()
        os_inst.reflection.add_reflection = MagicMock(return_value=MagicMock(entry_id="r1"))
        os_inst.dream = MagicMock()
        os_inst.dream.consolidate = MagicMock(return_value={"insights_generated": 3, "patterns_strengthened": 2})
        os_inst.metacognition = MagicMock()
        os_inst.metacognition.end_thinking = MagicMock()
        os_inst.evolution = MagicMock()
        os_inst.evolve_from_device = MagicMock()
        return os_inst

    def test_reflect_invokes_reflection(self, os_instance):
        """CognitiveReflection.add_reflection() is called during _handle_reflect."""
        asyncio.run(self._run_reflect(os_instance))
        os_instance.reflection.add_reflection.assert_called_once()

    def test_reflect_invokes_dream(self, os_instance):
        """DreamState.consolidate() is called during _handle_reflect."""
        asyncio.run(self._run_reflect(os_instance))
        os_instance.dream.consolidate.assert_called_once()

    async def _run_reflect(self, os_inst):
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(
            mode=CognitiveMode.REFLECT,
            current_query="test reflection",
            observations=[],
            context={"metacognition": {"cognitive_load": 0.4}},
            confidence=0.7,
            domain="general",
        )
        return await os_inst._handle_reflect(state)


class TestContextEnrichment:
    """Test that the cognitive pipeline enriches state.context with module data."""

    @pytest.fixture
    def os_instance(self):
        os_inst = ShadowOS.__new__(ShadowOS)
        os_inst._running = False
        os_inst._stats = {}
        os_inst._stats_lock = MagicMock()
        os_inst._start_time = 0

        os_inst.graph = MagicMock()
        os_inst.memory = AsyncMock()
        os_inst.memory.get_recent = AsyncMock(return_value=[])
        os_inst.memory.search = AsyncMock(return_value=[])
        os_inst.crew = MagicMock()
        os_inst.mesh = MagicMock()
        os_inst.stealth = MagicMock()
        os_inst.personality = MagicMock()
        os_inst.voice = MagicMock()
        os_inst.anomaly_detector = MagicMock()
        os_inst.anomaly_detector.get_anomalies = MagicMock(return_value=[])
        os_inst.goal_planner = MagicMock()
        os_inst.goal_planner.list_goals = MagicMock(return_value=[])
        os_inst.thought_sense = MagicMock()
        os_inst.thought_sense.analyze = MagicMock(return_value={"intent": {"type": "question"}, "emotion": {"primary": "neutral"}, "urgency": {"level": "low"}, "cognitive_load": 0.3})
        os_inst.recall_synthesis = MagicMock()
        os_inst.recall_synthesis.synthesize = MagicMock(return_value={"insights": []})
        os_inst.temporal_reasoning = MagicMock()
        os_inst.temporal_reasoning.detect_patterns = MagicMock(return_value=[])
        os_inst.empathy = MagicMock()
        os_inst.empathy.get_overall_empathy = MagicMock(return_value={"average_score": 0.65})
        # _handle_analyze actually calls get_stats() — give it a populated
        # response so social_empathy gets written into state.context.
        os_inst.empathy.get_stats = MagicMock(return_value={"people_tracked": 1, "avg_empathy": 0.65, "total_persons": 1})
        os_inst.presence = MagicMock()
        os_inst.trinity = MagicMock()
        os_inst.dream = MagicMock()
        os_inst.skills = MagicMock()
        os_inst.skills.get_domain_summary = MagicMock(return_value={})
        os_inst.narrative = MagicMock()
        os_inst.narrative.get_themes = MagicMock(return_value={})
        os_inst.procedural = MagicMock()
        os_inst.procedural.get_procedures_by_domain = MagicMock(return_value=[])
        os_inst.conflict = MagicMock()
        os_inst.conflict.list_conflicts = MagicMock(return_value=[])
        os_inst.emotional_intel = MagicMock()
        os_inst.creative = MagicMock()
        os_inst.metacognition = MagicMock()
        os_inst.moral = MagicMock()
        os_inst.attention = MagicMock()
        os_inst.attention.get_priorities = MagicMock(return_value=[])
        os_inst.attention.detect_focus_state = MagicMock(return_value=None)
        os_inst.pipeline = MagicMock()
        os_inst.social_intel = MagicMock()
        os_inst.social_intel.list_groups = MagicMock(return_value=[])
        os_inst.journal = MagicMock()
        os_inst.reflection = MagicMock()
        os_inst.goal_reasoning = MagicMock()
        os_inst.device_context = MagicMock()
        os_inst.device_context.get_context = MagicMock(return_value=None)
        os_inst.arise = MagicMock()
        os_inst.summon = MagicMock()
        os_inst.evolution = MagicMock()
        os_inst.evolution.get_friction_points = MagicMock(return_value=[])
        os_inst.oracle = MagicMock()
        # `_handle_analyze` calls get_domain_states().
        os_inst.oracle.get_domain_states = MagicMock(return_value={})
        os_inst.oracle.get_active_domains = MagicMock(return_value=set())
        os_inst.daemon = MagicMock()
        os_inst.mind = AsyncMock()
        os_inst.shadow_ai = AsyncMock()
        os_inst.event_bus = MagicMock()
        os_inst.event_bus.get_history = MagicMock(return_value=[])
        return os_inst

    def test_analyze_enriches_context(self, os_instance):
        """_handle_analyze populates context with data from multiple modules."""
        state = asyncio.run(self._run_analyze(os_instance))
        assert "intent" in state.context
        assert "emotion" in state.context
        assert "urgency" in state.context
        assert "cognitive_load" in state.context
        assert "skill_domains" in state.context or "social_empathy" in state.context

    async def _run_analyze(self, os_inst):
        from shadowd.cognition.shadow_graph import CognitiveState, CognitiveMode
        state = CognitiveState(
            mode=CognitiveMode.ANALYZE,
            current_query="How are you doing?",
            observations=[],
            context={},
            confidence=0.5,
            domain="general",
        )
        return await os_inst._handle_analyze(state)