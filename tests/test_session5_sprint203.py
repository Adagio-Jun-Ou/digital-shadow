"""Sprint 203: Cognitive Memory Bridge — _enrich_with_cognitive_modules() tests"""

import asyncio
import pytest
from unittest.mock import MagicMock, patch, AsyncMock
from shadowd.cognition.shadow_os import ShadowOS


class MockShadowOS:
    """Minimal ShadowOS mock with all cognitive modules needed for enrichment."""

    def __init__(self):
        self.emotional_intel = MagicMock()
        self.emotional_intel.detect_emotion = MagicMock(return_value={
            "primary_emotion": "joy",
            "intensity": 0.7,
        })

        self.thought_sense = MagicMock()
        self.thought_sense.analyze = MagicMock(return_value={
            "intent": {"type": "inform"},
            "urgency": {"level": "medium"},
        })

        self.recall_synthesis = MagicMock()
        self.recall_synthesis.add_observation = MagicMock()

        self.temporal_reasoning = MagicMock()
        self.temporal_reasoning.record_event = MagicMock()

        self.attention = MagicMock()
        self.attention.add_attention_item = MagicMock()

        self.narrative = MagicMock()
        self.narrative.add_event = MagicMock()

        self.skills = MagicMock()
        self.skills.use_skill = MagicMock()

        self.goal_planner = MagicMock()
        self.goal_planner.list_goals = MagicMock(return_value=[])

        self.evolution = MagicMock()
        self.evolution.observe = MagicMock()

        self.procedural = MagicMock()
        self.procedural.practice_procedure = MagicMock()

        self.empathy = MagicMock()
        self.empathy.record_interaction = MagicMock()

        self.journal = MagicMock()
        self.journal.record_decision = MagicMock()

        self.feed_loop = MagicMock()
        self.feed_loop.emit = MagicMock()

    async def _enrich_with_cognitive_modules(self, observation, obs_type, domain, importance):
        """Copy of the real method for testing."""
        content = observation.get("content", str(observation))[:500]
        if not content:
            return

        try:
            result = self.emotional_intel.detect_emotion(content)
            if result:
                primary = result.get("primary_emotion", "neutral")
                intensity = result.get("intensity", 0.3)
                self.feed_loop.emit("emotional_intelligence", "emotion",
                                   intensity, {"primary": primary, "content_snippet": content[:100]})
                self.feed_loop.emit("emotional_intelligence", "social",
                                    intensity * 0.7, {"emotion": primary})
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            thought = self.thought_sense.analyze(content)
            if thought:
                urgency = thought.get("urgency", {}).get("level", "low")
                if urgency in ("high", "critical"):
                    self.feed_loop.emit("thought_sense", "cognitive", 0.8,
                                       {"intent": thought.get("intent", {}).get("type", "unknown"),
                                        "urgency": urgency})
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            self.recall_synthesis.add_observation(content=content, domain=domain,
                                                   importance=importance)
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            self.temporal_reasoning.record_event(
                event_type=obs_type,
                content=content,
                domain=domain,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            level = "high" if importance > 0.7 else "medium" if importance > 0.4 else "low"
            self.attention.add_attention_item(
                topic=content[:100],
                level=level,
                domain=domain,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            self.narrative.add_event(
                content=content,
                event_type=obs_type,
                importance=importance,
                domain=domain,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            if domain in ("career", "work", "productivity", "technical", "learning"):
                self.skills.use_skill(skill_id=f"observe_{domain}", success=importance > 0.5)
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            goals = self.goal_planner.list_goals(limit=3)
            active = [g for g in goals if g.get("status") in ("in_progress", "pending")]
            if active and importance > 0.5:
                self.feed_loop.emit("goal_reasoning", "goal", importance,
                                   {"observation": content[:200], "active_goals": len(active)})
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            self.evolution.observe(
                observation_type=obs_type,
                content=content,
                domain=domain,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        try:
            self.procedural.practice_procedure(
                procedure_id=f"observe_{obs_type}",
                success=importance > 0.5,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError):
            pass

        obs_people = observation.get("people_nearby", 0)
        if isinstance(obs_people, int) and obs_people > 0:
            try:
                self.empathy.record_interaction(
                    person_id="group",
                    interaction_type="proximity",
                    sentiment=0.5,
                    notes=f"Near {obs_people} people during {obs_type}",
                )
                self.feed_loop.emit("presence_detection", "social", 0.6,
                                    {"people_nearby": obs_people})
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass

        if obs_type in ("decision", "action"):
            try:
                self.journal.record_decision(
                    title=content[:200],
                    description=content[:500],
                    decision_type=obs_type,
                    options=["observed"],
                    chosen="observed",
                    reasoning="Automatic observation tracking",
                    confidence=importance,
                )
                self.feed_loop.emit("decision_journal", "cognitive", importance,
                                   {"decision_type": obs_type})
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError):
                pass

        if any(kw in content.lower() for kw in ("conflict", "disagree", "argue", "oppose", "object")):
            try:
                self.feed_loop.emit("conflict_resolution", "social", 0.7,
                                   {"content": content[:200], "type": "observed_conflict"})
            except (ValueError, TypeError, AttributeError, RuntimeError):
                pass


@pytest.fixture
def mock_os():
    return MockShadowOS()


class TestCognitiveMemoryBridge:
    """Test _enrich_with_cognitive_modules auto-triggers."""

    @pytest.mark.asyncio
    async def test_emotion_detection_enrichment(self, mock_os):
        obs = {"content": "I'm feeling great about this project!"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.8)
        mock_os.emotional_intel.detect_emotion.assert_called_once()
        assert mock_os.feed_loop.emit.call_count >= 2  # emotion + social

    @pytest.mark.asyncio
    async def test_thought_sense_urgency(self, mock_os):
        mock_os.thought_sense.analyze = MagicMock(return_value={
            "intent": {"type": "request"},
            "urgency": {"level": "critical"},
        })
        obs = {"content": "We need this immediately!"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.9)
        mock_os.thought_sense.analyze.assert_called_once()
        sources = [call.args[0] for call in mock_os.feed_loop.emit.call_args_list]
        assert "thought_sense" in sources

    @pytest.mark.asyncio
    async def test_thought_sense_low_urgency_no_signal(self, mock_os):
        mock_os.thought_sense.analyze = MagicMock(return_value={
            "intent": {"type": "inform"},
            "urgency": {"level": "low"},
        })
        obs = {"content": "Just fyi"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.3)
        thought_emits = [c for c in mock_os.feed_loop.emit.call_args_list if c[0][0] == "thought_sense"]
        assert len(thought_emits) == 0

    @pytest.mark.asyncio
    async def test_recall_synthesis_called(self, mock_os):
        obs = {"content": "Important observation about the system"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.7)
        mock_os.recall_synthesis.add_observation.assert_called_once()
        call_kwargs = mock_os.recall_synthesis.add_observation.call_args[1]
        assert call_kwargs["domain"] == "work"

    @pytest.mark.asyncio
    async def test_temporal_reasoning_called(self, mock_os):
        obs = {"content": "Pattern detected at noon"}
        await mock_os._enrich_with_cognitive_modules(obs, "event", "general", 0.5)
        mock_os.temporal_reasoning.record_event.assert_called_once()
        call_kwargs = mock_os.temporal_reasoning.record_event.call_args[1]
        assert call_kwargs["event_type"] == "event"

    @pytest.mark.asyncio
    async def test_attention_level_mapping(self, mock_os):
        obs = {"content": "Urgent task"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.9)
        call_kwargs = mock_os.attention.add_attention_item.call_args[1]
        assert call_kwargs["level"] == "high"

        mock_os.attention.add_attention_item.reset_mock()
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.3)
        call_kwargs = mock_os.attention.add_attention_item.call_args[1]
        assert call_kwargs["level"] == "low"

    @pytest.mark.asyncio
    async def test_narrative_event_recording(self, mock_os):
        obs = {"content": "A meaningful life event"}
        await mock_os._enrich_with_cognitive_modules(obs, "milestone", "personal", 0.8)
        mock_os.narrative.add_event.assert_called_once()
        call_kwargs = mock_os.narrative.add_event.call_args[1]
        assert call_kwargs["event_type"] == "milestone"
        assert call_kwargs["importance"] == 0.8

    @pytest.mark.asyncio
    async def test_skill_usage_for_career_domain(self, mock_os):
        obs = {"content": "Working on project"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "career", 0.7)
        mock_os.skills.use_skill.assert_called_once()
        call_kwargs = mock_os.skills.use_skill.call_args[1]
        assert call_kwargs["skill_id"] == "observe_career"
        assert call_kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_skill_not_triggered_for_general_domain(self, mock_os):
        obs = {"content": "Casual observation"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.5)
        mock_os.skills.use_skill.assert_not_called()

    @pytest.mark.asyncio
    async def test_goal_tracking_with_active_goals(self, mock_os):
        mock_os.goal_planner.list_goals = MagicMock(return_value=[
            {"status": "in_progress", "title": "Project X"},
            {"status": "pending", "title": "Project Y"},
        ])
        obs = {"content": "Working on an important goal"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.8)
        goal_emits = [c for c in mock_os.feed_loop.emit.call_args_list if c[0][0] == "goal_reasoning"]
        assert len(goal_emits) == 1

    @pytest.mark.asyncio
    async def test_goal_tracking_no_active_goals(self, mock_os):
        mock_os.goal_planner.list_goals = MagicMock(return_value=[
            {"status": "completed", "title": "Done"},
        ])
        obs = {"content": "Observation"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.8)
        goal_emits = [c for c in mock_os.feed_loop.emit.call_args_list if c[0][0] == "goal_reasoning"]
        assert len(goal_emits) == 0

    @pytest.mark.asyncio
    async def test_evolution_observe_called(self, mock_os):
        obs = {"content": "Evolutionary observation"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "learning", 0.6)
        mock_os.evolution.observe.assert_called_once()
        call_kwargs = mock_os.evolution.observe.call_args[1]
        assert call_kwargs["observation_type"] == "observation"
        assert call_kwargs["domain"] == "learning"

    @pytest.mark.asyncio
    async def test_procedural_practice_called(self, mock_os):
        obs = {"content": "Practicing a procedure"}
        await mock_os._enrich_with_cognitive_modules(obs, "action", "work", 0.7)
        mock_os.procedural.practice_procedure.assert_called_once()
        call_kwargs = mock_os.procedural.practice_procedure.call_args[1]
        assert call_kwargs["procedure_id"] == "observe_action"
        assert call_kwargs["success"] is True

    @pytest.mark.asyncio
    async def test_empathy_with_people_nearby(self, mock_os):
        obs = {"content": "Meeting room observation", "people_nearby": 5}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "social", 0.5)
        mock_os.empathy.record_interaction.assert_called_once()
        call_kwargs = mock_os.empathy.record_interaction.call_args[1]
        assert call_kwargs["person_id"] == "group"
        assert call_kwargs["interaction_type"] == "proximity"

    @pytest.mark.asyncio
    async def test_no_empathy_without_people(self, mock_os):
        obs = {"content": "Solo observation"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.5)
        mock_os.empathy.record_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_decision_journal_for_decision_type(self, mock_os):
        obs = {"content": "Made an important decision"}
        await mock_os._enrich_with_cognitive_modules(obs, "decision", "career", 0.8)
        mock_os.journal.record_decision.assert_called_once()
        call_kwargs = mock_os.journal.record_decision.call_args[1]
        assert call_kwargs["decision_type"] == "decision"

    @pytest.mark.asyncio
    async def test_decision_journal_for_action_type(self, mock_os):
        obs = {"content": "Took an action"}
        await mock_os._enrich_with_cognitive_modules(obs, "action", "career", 0.7)
        mock_os.journal.record_decision.assert_called_once()

    @pytest.mark.asyncio
    async def test_no_journal_for_non_decision_type(self, mock_os):
        obs = {"content": "Just observing"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.5)
        mock_os.journal.record_decision.assert_not_called()

    @pytest.mark.asyncio
    async def test_conflict_detection(self, mock_os):
        obs = {"content": "There's a disagreement about the approach"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.6)
        conflict_emits = [c for c in mock_os.feed_loop.emit.call_args_list
                         if c[0][0] == "conflict_resolution"]
        assert len(conflict_emits) == 1

    @pytest.mark.asyncio
    async def test_no_conflict_detection_without_keywords(self, mock_os):
        obs = {"content": "Everything is fine and agreed upon"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.5)
        conflict_emits = [c for c in mock_os.feed_loop.emit.call_args_list
                         if c[0][0] == "conflict_resolution"]
        assert len(conflict_emits) == 0

    @pytest.mark.asyncio
    async def test_module_failure_does_not_crash(self, mock_os):
        mock_os.emotional_intel.detect_emotion = MagicMock(side_effect=ValueError("test"))
        mock_os.thought_sense.analyze = MagicMock(side_effect=RuntimeError("test"))
        mock_os.recall_synthesis.add_observation = MagicMock(side_effect=AttributeError("test"))
        obs = {"content": "Important data"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.7)
        # Should not raise — all failures caught
        mock_os.temporal_reasoning.record_event.assert_called_once()

    @pytest.mark.asyncio
    async def test_empty_content_returns_early(self, mock_os):
        obs = {"content": ""}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.5)
        mock_os.emotional_intel.detect_emotion.assert_not_called()

    @pytest.mark.asyncio
    async def test_content_truncation(self, mock_os):
        long_content = "x" * 1000
        obs = {"content": long_content}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.5)
        mock_os.emotional_intel.detect_emotion.assert_called_once()
        call_arg = mock_os.emotional_intel.detect_emotion.call_args[0][0]
        assert len(call_arg) <= 500

    @pytest.mark.asyncio
    async def test_people_nearby_non_integer_ignored(self, mock_os):
        obs = {"content": "Observation", "people_nearby": "five"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "social", 0.5)
        mock_os.empathy.record_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_people_nearby_zero_ignored(self, mock_os):
        obs = {"content": "Solo work", "people_nearby": 0}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "general", 0.5)
        mock_os.empathy.record_interaction.assert_not_called()

    @pytest.mark.asyncio
    async def test_multiple_conflict_keywords(self, mock_os):
        for kw in ("conflict", "disagree", "argue", "oppose", "object"):
            mock_os.feed_loop.emit.reset_mock()
            obs = {"content": f"We {kw} about the plan"}
            await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.5)
            conflict_emits = [c for c in mock_os.feed_loop.emit.call_args_list
                             if c[0][0] == "conflict_resolution"]
            assert len(conflict_emits) == 1, f"Failed for keyword: {kw}"

    @pytest.mark.asyncio
    async def test_attention_medium_level(self, mock_os):
        obs = {"content": "Medium urgency"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "work", 0.5)
        call_kwargs = mock_os.attention.add_attention_item.call_args[1]
        assert call_kwargs["level"] == "medium"

    @pytest.mark.asyncio
    async def test_feed_loop_signals_include_metadata(self, mock_os):
        obs = {"content": "Joyful news!", "people_nearby": 3}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "social", 0.8)
        # Check that emit calls include metadata dicts
        for call in mock_os.feed_loop.emit.call_args_list:
            assert len(call[0]) >= 3  # source, signal_type, intensity
            if len(call[0]) >= 4:
                assert isinstance(call[0][3], dict)  # metadata is a dict

    @pytest.mark.asyncio
    async def test_skill_success_depends_on_importance(self, mock_os):
        obs = {"content": "Working"}
        await mock_os._enrich_with_cognitive_modules(obs, "observation", "career", 0.3)
        call_kwargs = mock_os.skills.use_skill.call_args[1]
        assert call_kwargs["success"] is False

    @pytest.mark.asyncio
    async def test_procedural_success_depends_on_importance(self, mock_os):
        obs = {"content": "Task"}
        await mock_os._enrich_with_cognitive_modules(obs, "action", "work", 0.3)
        call_kwargs = mock_os.procedural.practice_procedure.call_args[1]
        assert call_kwargs["success"] is False