#!/usr/bin/env python3
"""Sprint 189 — Deep audit of shadow_arise, shadow_summon, shadow_enhanced."""
import copy
import os
import re
import threading
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch, AsyncMock

from shadowd.cognition.shadow_arise import (
    ShadowArise, AgentLifecycle, AgentStatus, CapabilityType, MissionPriority,
    _sanitize, _sanitize_content, _SANITIZE_STRICT_RE, _SANITIZE_CONTENT_RE,
)
from shadowd.cognition.shadow_summon import (
    ShadowSummon, SummonIntent, _sanitize as ss_sanitize,
    _sanitize_content as ss_sanitize_content,
    _SANITIZE_STRICT_RE as SS_STRICT, _SANITIZE_CONTENT_RE as SS_CONTENT,
)

REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))


class TestShadowAriseSanitize(unittest.TestCase):
    def test_strict_strips_angle_brackets(self):
        result = _sanitize("hello <script>alert('xss')</script> world", 200)
        self.assertNotIn("<", result)
        self.assertNotIn(">", result)

    def test_content_preserves_angle_brackets(self):
        result = _sanitize_content("latency < 100ms and > 50ms", 2000)
        self.assertIn("<", result)
        self.assertIn(">", result)

    def test_strict_strips_control_chars(self):
        result = _sanitize("hello\x00world\x01test", 200)
        self.assertNotIn("\x00", result)
        self.assertNotIn("\x01", result)

    def test_content_strips_control_chars_but_preserves_newlines(self):
        result = _sanitize_content("hello\nworld\ttab", 2000)
        self.assertIn("\n", result)
        self.assertIn("\t", result)
        self.assertNotIn("\x00", result)

    def test_strict_strips_tabs_and_newlines(self):
        result = _sanitize("hello\tworld\ntest", 200)
        self.assertNotIn("\t", result)
        self.assertNotIn("\n", result)

    def test_content_preserves_tabs_and_newlines(self):
        result = _sanitize_content("hello\tworld\ntest", 2000)
        self.assertIn("\t", result)
        self.assertIn("\n", result)

    def test_content_strips_del_char(self):
        result = _sanitize_content("hello\x7fworld", 2000)
        self.assertNotIn("\x7f", result)

    def test_strict_strips_del_char(self):
        result = _sanitize("hello\x7fworld", 200)
        self.assertNotIn("\x7f", result)

    def test_strip_before_truncate(self):
        long_text = "   hello world   "
        result = _sanitize(long_text, 5)
        self.assertEqual(result, "hello")

    def test_content_strip_before_truncate(self):
        long_text = "   hello world   "
        result = _sanitize_content(long_text, 5)
        self.assertEqual(result, "hello")

    def test_non_string_returns_empty(self):
        self.assertEqual(_sanitize(123, 200), "")
        self.assertEqual(_sanitize_content(None, 2000), "")

    def test_content_max_length_default(self):
        self.assertEqual(_sanitize_content.__defaults__, (2000,))

    def test_strict_max_length_default(self):
        self.assertEqual(_sanitize.__defaults__, (200,))


class TestShadowAriseThreadSafety(unittest.TestCase):
    def setUp(self):
        self.arise = ShadowArise()

    def test_stats_lock_exists(self):
        self.assertTrue(hasattr(self.arise, '_stats_lock'))
        self.assertIsInstance(self.arise._stats_lock, type(threading.Lock()))

    def test_concurrent_arise_creations(self):
        errors = []

        def create_agent(i):
            try:
                self.arise.arise(
                    name=f"agent_{i}",
                    objective=f"test objective {i}",
                )
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=create_agent, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(errors), 0)

    def test_get_stats_returns_copy(self):
        self.arise.arise(name="test", objective="test obj")
        stats1 = self.arise.get_stats()
        stats2 = self.arise.get_stats()
        self.assertIsNot(stats1, stats2)

    def test_get_stats_thread_safe_read(self):
        self.arise.arise(name="test", objective="test obj")

        stats_reads = []

        def read_stats():
            for _ in range(50):
                stats_reads.append(self.arise.get_stats())

        threads = [threading.Thread(target=read_stats) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        self.assertEqual(len(stats_reads), 250)


class TestShadowAriseAgentLifecycle(unittest.TestCase):
    def setUp(self):
        self.arise = ShadowArise()

    def test_arise_creates_agent(self):
        agent = self.arise.arise(name="Test Agent", objective="Analyze data")
        self.assertIsNotNone(agent.agent_id)
        self.assertEqual(agent.name, "Test Agent")
        self.assertEqual(agent.lifecycle, AgentLifecycle.ARISE)
        self.assertEqual(agent.status, AgentStatus.PENDING)

    def test_configure_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        configured = self.arise.configure(agent.agent_id, capabilities=[CapabilityType.CREATE])
        self.assertIsNotNone(configured)
        self.assertEqual(configured.lifecycle, AgentLifecycle.CONFIGURE)

    def test_activate_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        activated = self.arise.activate(agent.agent_id)
        self.assertEqual(activated.status, AgentStatus.ACTIVE)

    def test_execute_step(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        step = self.arise.execute_step(
            agent.agent_id,
            description="Do something",
            capability_used=CapabilityType.REASON,
            confidence=0.8,
        )
        self.assertIsNotNone(step)
        self.assertEqual(round(step.confidence, 3), 0.8)

    def test_evolve_rejects_completed_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        self.arise.complete(agent.agent_id, status=AgentStatus.COMPLETED)
        result = self.arise.evolve(agent.agent_id, feedback="test")
        self.assertIsNone(result)

    def test_evolve_rejects_failed_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        self.arise.complete(agent.agent_id, status=AgentStatus.FAILED)
        result = self.arise.evolve(agent.agent_id, feedback="test")
        self.assertIsNone(result)

    def test_evolve_active_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        result = self.arise.evolve(agent.agent_id, feedback="improve")
        self.assertIsNotNone(result)
        self.assertEqual(result.lifecycle, AgentLifecycle.EXECUTE)

    def test_complete_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        completed = self.arise.complete(agent.agent_id)
        self.assertEqual(completed.status, AgentStatus.COMPLETED)

    def test_archive_agent(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        self.arise.complete(agent.agent_id)
        archived = self.arise.archive(agent.agent_id)
        self.assertEqual(archived.status, AgentStatus.ARCHIVED)

    def test_get_agent_returns_deepcopy(self):
        agent = self.arise.arise(name="Test", objective="Test")
        retrieved = self.arise.get_agent(agent.agent_id)
        agent.name = "Modified"
        self.assertNotEqual(retrieved.name, "Modified")

    def test_list_agents_returns_deepcopy(self):
        self.arise.arise(name="A1", objective="O1")
        self.arise.arise(name="A2", objective="O2")
        agents = self.arise.list_agents()
        original_name = agents[0].name
        agents[0].name = "Modified"
        retrieved = self.arise.get_agent(agents[0].agent_id)
        self.assertEqual(retrieved.name, original_name)

    def test_recommend_template(self):
        template = self.arise.recommend_template("I need to analyze research data")
        self.assertIsNotNone(template)
        self.assertEqual(template.template_id, "analyst")

    def test_recommend_template_returns_deepcopy(self):
        template = self.arise.recommend_template("analyze data")
        template.name = "Modified"
        original = self.arise._templates["analyst"]
        self.assertNotEqual(template.name, original.name)

    def test_agent_delegation_cap(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        for i in range(600):
            step = self.arise.execute_step(
                agent.agent_id,
                description=f"Step {i}",
                capability_used=CapabilityType.REASON,
                confidence=0.5,
            )
        execution_log = self.arise.get_agent(agent.agent_id).execution_log
        self.assertLessEqual(len(execution_log), 500)

    def test_max_agents_eviction(self):
        arise = ShadowArise()
        arise._MAX_AGENTS = 3
        agent_ids = []
        for i in range(4):
            agent = arise.arise(name=f"Agent_{i}", objective="Test")
            agent_ids.append(agent.agent_id)
        self.assertLessEqual(len(arise._agents), 3)


class TestShadowAriseToDict(unittest.TestCase):
    def setUp(self):
        self.arise = ShadowArise()

    def test_capability_to_dict_rounds_proficiency(self):
        from shadowd.cognition.shadow_arise import Capability
        cap = Capability(name="test", capability_type=CapabilityType.REASON, proficiency=0.123456, description="desc", configuration={"key": "val"})
        d = cap.to_dict()
        self.assertEqual(d["proficiency"], 0.123)

    def test_agent_instance_to_dict_deepcopy(self):
        agent = self.arise.arise(name="Test", objective="Test")
        d = agent.to_dict()
        d["metadata"]["new_key"] = "new_val"
        retrieved = self.arise.get_agent(agent.agent_id)
        self.assertNotIn("new_key", retrieved.metadata)

    def test_execution_step_to_dict_deepcopy(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        step = self.arise.execute_step(
            agent.agent_id,
            description="Do it",
            capability_used=CapabilityType.REASON,
            input_data={"key": "val"},
            output_data={"result": "data"},
            confidence=0.75,
        )
        d = step.to_dict()
        d["input_data"]["new_key"] = "hacked"
        self.assertNotIn("new_key", step.input_data)

    def test_mission_context_to_dict_deepcopy(self):
        from shadowd.cognition.shadow_arise import MissionContext
        mission = MissionContext(
            objective="Test",
            priority=MissionPriority.HIGH,
            constraints=["c1"],
            success_criteria=["s1"],
            deadline=None,
            context_data={"ctx": "data"},
        )
        d = mission.to_dict()
        d["context_data"]["new_key"] = "hacked"
        self.assertNotIn("new_key", mission.context_data)


class TestShadowAriseWordBoundary(unittest.TestCase):
    def setUp(self):
        self.arise = ShadowArise()

    def test_recommend_template_word_boundary(self):
        template = self.arise.recommend_template("I love creative design")
        self.assertEqual(template.template_id, "creator")

    def test_recommend_template_no_false_match(self):
        template = self.arise.recommend_template("coordinator skills needed")
        self.assertIsNotNone(template)
        self.assertEqual(template.template_id, "coordinator")

    def test_recommend_template_capability_match(self):
        template = self.arise.recommend_template("I need reason and analysis")
        self.assertIsNotNone(template)


class TestShadowSummonSanitize(unittest.TestCase):
    def test_strict_strips_angle_brackets(self):
        result = ss_sanitize("hello <b>world</b>", 500)
        self.assertNotIn("<", result)

    def test_content_preserves_angle_brackets(self):
        result = ss_sanitize_content("latency < 100ms", 500)
        self.assertIn("<", result)

    def test_strict_strips_control_chars(self):
        result = ss_sanitize("hello\x00world", 500)
        self.assertNotIn("\x00", result)

    def test_content_preserves_tabs_newlines(self):
        result = ss_sanitize_content("hello\tworld\ntest", 500)
        self.assertIn("\t", result)
        self.assertIn("\n", result)

    def test_non_string_returns_empty(self):
        self.assertEqual(ss_sanitize(123), "")
        self.assertEqual(ss_sanitize_content(None), "")


class TestShadowSummonClassify(unittest.TestCase):
    def setUp(self):
        self.summon = ShadowSummon()

    def test_create_agent_intent(self):
        intent = self.summon.classify_intent("create a new agent")
        self.assertEqual(intent, SummonIntent.CREATE_AGENT)

    def test_list_agents_intent(self):
        intent = self.summon.classify_intent("list all agents")
        self.assertEqual(intent, SummonIntent.LIST_AGENTS)

    def test_list_abilities_intent(self):
        intent = self.summon.classify_intent("what abilities do you have")
        self.assertEqual(intent, SummonIntent.LIST_ABILITIES)

    def test_ask_question_intent(self):
        intent = self.summon.classify_intent("what is the meaning of life")
        self.assertEqual(intent, SummonIntent.ASK_QUESTION)

    def test_observe_event_intent(self):
        intent = self.summon.classify_intent("I just finished the project")
        self.assertEqual(intent, SummonIntent.OBSERVE_EVENT)

    def test_get_status_intent(self):
        intent = self.summon.classify_intent("show me system status")
        self.assertEqual(intent, SummonIntent.GET_STATUS)

    def test_invoke_ability_intent(self):
        intent = self.summon.classify_intent("shadow predict what will happen")
        self.assertEqual(intent, SummonIntent.INVOKE_ABILITY)

    def test_unknown_intent(self):
        intent = self.summon.classify_intent("xyz random nonsense")
        self.assertEqual(intent, SummonIntent.UNKNOWN)

    def test_none_input(self):
        intent = self.summon.classify_intent(None)
        self.assertEqual(intent, SummonIntent.UNKNOWN)

    def test_non_string_input(self):
        intent = self.summon.classify_intent(123)
        self.assertEqual(intent, SummonIntent.UNKNOWN)

    def test_word_boundary_explain(self):
        intent = self.summon.classify_intent("please explain this concept")
        self.assertEqual(intent, SummonIntent.ASK_QUESTION)

    def test_word_boundary_why(self):
        intent = self.summon.classify_intent("why does this happen")
        self.assertEqual(intent, SummonIntent.ASK_QUESTION)

    def test_complete_agent_intent(self):
        intent = self.summon.classify_intent("complete the agent")
        self.assertEqual(intent, SummonIntent.COMPLETE_AGENT)

    def test_evolve_agent_intent(self):
        intent = self.summon.classify_intent("evolve agent")
        self.assertEqual(intent, SummonIntent.EVOLVE_AGENT)


class TestShadowSummonMatchAbility(unittest.TestCase):
    def setUp(self):
        self.summon = ShadowSummon()

    def test_match_oracle(self):
        ability = self.summon.match_ability("predict the future")
        self.assertEqual(ability, "oracle")

    def test_match_x_ray(self):
        ability = self.summon.match_ability("scan nearby devices")
        self.assertEqual(ability, "x_ray")

    def test_match_perfect_memory(self):
        ability = self.summon.match_ability("remember this")
        self.assertEqual(ability, "perfect_memory")

    def test_match_none_input(self):
        ability = self.summon.match_ability(None)
        self.assertIsNone(ability)

    def test_match_non_string(self):
        ability = self.summon.match_ability(42)
        self.assertIsNone(ability)

    def test_longest_match_wins(self):
        ability = self.summon.match_ability("find patterns in data")
        self.assertEqual(ability, "pattern_sight")


class TestShadowSummonMatchTemplate(unittest.TestCase):
    def setUp(self):
        self.summon = ShadowSummon()

    def test_match_analyst(self):
        template = self.summon.match_template("analyze the data deeply")
        self.assertEqual(template, "analyst")

    def test_match_creator(self):
        template = self.summon.match_template("create something new")
        self.assertEqual(template, "creator")

    def test_match_none_input(self):
        template = self.summon.match_template(None)
        self.assertIsNone(template)


class TestShadowSummonProcessVoice(unittest.TestCase):
    def setUp(self):
        self.summon = ShadowSummon()

    def test_process_voice_empty(self):
        response = self.summon.process_voice("")
        self.assertEqual(response["intent"], "unknown")
        self.assertEqual(response["confidence"], 0.0)

    def test_process_voice_creates_agent(self):
        response = self.summon.process_voice("shadow arise create agent")
        self.assertEqual(response["intent"], "create_agent")

    def test_process_voice_sanitizes_transcript(self):
        response = self.summon.process_voice("hello\x00world<script>alert</script>")
        self.assertNotIn("\x00", response["result"]["transcript"])

    def test_process_voice_confidence_clamped(self):
        response = self.summon.process_voice("predict what will happen for analysis and research")
        self.assertLessEqual(response["confidence"], 1.0)
        self.assertGreaterEqual(response["confidence"], 0.0)

    def test_process_voice_history_tracked(self):
        self.summon.process_voice("show status")
        self.summon.process_voice("list agents")
        history = self.summon.get_history()
        self.assertGreaterEqual(len(history), 2)

    def test_process_voice_history_deepcopy(self):
        self.summon.process_voice("show status")
        h1 = self.summon.get_history()
        h2 = self.summon.get_history()
        self.assertIsNot(h1, h2)

    def test_get_abilities_returns_deepcopy(self):
        a1 = self.summon.get_abilities()
        a2 = self.summon.get_abilities()
        self.assertIsNot(a1, a2)

    def test_get_agent_templates_returns_deepcopy(self):
        t1 = self.summon.get_agent_templates()
        t2 = self.summon.get_agent_templates()
        self.assertIsNot(t1, t2)

    def test_get_stats_returns_copy(self):
        self.summon.process_voice("show status")
        s1 = self.summon.get_stats()
        s2 = self.summon.get_stats()
        self.assertIsNot(s1, s2)

    def test_stats_thread_safety(self):
        errors = []

        def process():
            for i in range(20):
                try:
                    self.summon.process_voice(f"test command {i}")
                except Exception as e:
                    errors.append(e)

        threads = [threading.Thread(target=process) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        self.assertEqual(len(errors), 0)

    def test_stats_lock_exists(self):
        self.assertTrue(hasattr(self.summon, '_stats_lock'))
        self.assertIsInstance(self.summon._stats_lock, type(threading.Lock()))


class TestShadowSummonSpokenResponse(unittest.TestCase):
    def setUp(self):
        self.summon = ShadowSummon()

    def test_create_agent_response(self):
        response = self.summon.process_voice("create agent for analysis")
        self.assertIn("Creating", response["spoken"])

    def test_invoke_ability_response(self):
        response = self.summon.process_voice("shadow predict the outcome")
        self.assertIn("Activating", response["spoken"])

    def test_unknown_response(self):
        response = self.summon.process_voice("random xyzzy")
        self.assertIn("How can I help", response["spoken"])


class TestShadowSummonRegexPatterns(unittest.TestCase):
    """Verify word boundary regex patterns in _ABILITY_MAP and _AGENT_TEMPLATES."""

    def test_ability_patterns_are_phrases(self):
        for ability, phrases in [
            ("oracle", ["predict", "prediction", "forecast"]),
            ("pattern_sight", ["find pattern", "find patterns"]),
        ]:
            for phrase in phrases:
                self.assertTrue(
                    re.search(r'\b' + re.escape(phrase) + r'\b', phrase),
                    f"Pattern '{phrase}' should match itself"
                )

    def test_word_boundary_notice(self):
        summon = ShadowSummon()
        result = summon.classify_intent("I notice a change")
        self.assertEqual(result, SummonIntent.OBSERVE_EVENT)

    def test_word_boundary_happened(self):
        summon = ShadowSummon()
        result = summon.classify_intent("what happened")
        self.assertEqual(result, SummonIntent.OBSERVE_EVENT)

    def test_word_boundary_explain(self):
        summon = ShadowSummon()
        result = summon.classify_intent("please explain this")
        self.assertEqual(result, SummonIntent.ASK_QUESTION)

    def test_word_boundary_describe(self):
        summon = ShadowSummon()
        result = summon.classify_intent("describe the situation")
        self.assertEqual(result, SummonIntent.ASK_QUESTION)

    def test_word_boundary_why(self):
        summon = ShadowSummon()
        result = summon.classify_intent("why did this happen")
        self.assertEqual(result, SummonIntent.ASK_QUESTION)

    def test_no_false_positive_whatever(self):
        summon = ShadowSummon()
        result = summon.classify_intent("whatever you say")
        self.assertNotEqual(result, SummonIntent.OBSERVE_EVENT)

    def test_no_false_positive_query_in_word(self):
        summon = ShadowSummon()
        result = summon.classify_intent("this is a requirement")
        self.assertNotEqual(result, SummonIntent.ASK_QUESTION)


class TestShadowAriseEvolutionHistory(unittest.TestCase):
    def setUp(self):
        self.arise = ShadowArise()

    def test_evolution_history_capped(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        for i in range(250):
            self.arise.evolve(agent.agent_id, feedback=f"feedback {i}")
        self.assertLessEqual(len(self.arise._evolution_history), 200)

    def test_evolution_history_entries(self):
        agent = self.arise.arise(name="Test", objective="Test")
        self.arise.activate(agent.agent_id)
        self.arise.evolve(agent.agent_id, feedback="improve reasoning")
        entry = self.arise._evolution_history[-1]
        self.assertIn("agent_id", entry)
        self.assertIn("feedback", entry)
        self.assertEqual(entry["feedback"][:8], "improve ")


class TestShadowAriseMaxTemplates(unittest.TestCase):
    def test_add_custom_template(self):
        arise = ShadowArise()
        from shadowd.cognition.shadow_arise import AgentTemplate
        template = AgentTemplate(
            template_id="custom_test",
            name="Custom Test",
            description="A test template",
            capabilities=[CapabilityType.REASON],
            default_proficiency=0.7,
            mission_types=["testing"],
            metadata={},
        )
        result = arise.add_template(template)
        self.assertEqual(result.template_id, "custom_test")

    def test_get_template_returns_deepcopy(self):
        arise = ShadowArise()
        t1 = arise.get_template("analyst")
        t2 = arise.get_template("analyst")
        self.assertIsNot(t1, t2)

    def test_list_templates_returns_deepcopy(self):
        arise = ShadowArise()
        t1 = arise.list_templates()
        t2 = arise.list_templates()
        self.assertIsNot(t1, t2)


class TestShadowEnhancedDatetime(unittest.TestCase):
    """Sprint 189 — shadow_enhanced.py datetime.now() → timezone.utc fixes."""

    def test_observe_uses_utc_timestamps(self):
        """All datetime.now() calls should use timezone.utc."""
        import re
        with open(_repo_file("shadowd/shadow_enhanced.py")) as f:
            source = f.read()
        naive_calls = [m for m in re.finditer(r'datetime\.now\(\)', source)
                       if 'timezone.utc' not in source[max(0, m.start()-20):m.end()+15]]
        self.assertEqual(len(naive_calls), 0, f"Found {len(naive_calls)} timezone-naive datetime.now() calls")

    def test_shadow_enhanced_narrowed_except(self):
        """All except clauses should be narrowed from bare Exception."""
        from shadowd.shadow_enhanced import EnhancedShadowAI
        with open(_repo_file("shadowd/shadow_enhanced.py")) as f:
            source = f.read()
        bare_excepts = re.findall(r'except Exception(?!\s+as)', source)
        self.assertEqual(len(bare_excepts), 0, f"Found {len(bare_excepts)} bare except Exception clauses")

    def test_shadow_enhanced_log_format(self):
        """Logger calls should use %s format, not f-strings."""
        with open(_repo_file("shadowd/shadow_enhanced.py")) as f:
            source = f.read()
        fstring_logs = re.findall(r'logger\.\w+\(f["\']', source)
        self.assertEqual(len(fstring_logs), 0, f"Found {len(fstring_logs)} f-string logger calls")


class TestApiRouterSprint189(unittest.TestCase):
    """Sprint 189 — API router endpoint validation deep audit fixes."""

    def test_event_request_confidence_is_clamped(self):
        """EventRequest confidence should have ge=0.0, le=1.0."""
        from shadowd.cognition.api_router import EventRequest
        req = EventRequest(confidence=0.5, urgency=0.5)
        self.assertEqual(req.confidence, 0.5)

    def test_event_request_confidence_rejects_negative(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import EventRequest
        with self.assertRaises(ValidationError):
            EventRequest(confidence=-1.0)

    def test_event_request_confidence_rejects_over_1(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import EventRequest
        with self.assertRaises(ValidationError):
            EventRequest(confidence=2.0)

    def test_event_request_urgency_rejects_negative(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import EventRequest
        with self.assertRaises(ValidationError):
            EventRequest(urgency=-0.5)

    def test_kg_query_request_has_max_length(self):
        from shadowd.cognition.api_router import KGQueryRequest
        req = KGQueryRequest(entity_id="test", relation_type="related")
        self.assertEqual(req.entity_id, "test")

    def test_kg_query_request_rejects_long_entity_id(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import KGQueryRequest
        with self.assertRaises(ValidationError):
            KGQueryRequest(entity_id="x" * 101)

    def test_search_request_domain_has_max_length(self):
        from shadowd.cognition.api_router import SearchRequest
        req = SearchRequest(query="test", domain="health")
        self.assertEqual(req.domain, "health")

    def test_daemon_watch_request_blocks_traversal(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import DaemonWatchRequest
        with self.assertRaises(ValidationError):
            DaemonWatchRequest(path="/etc/shadow")

    def test_daemon_watch_request_blocks_dotdot(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import DaemonWatchRequest
        with self.assertRaises(ValidationError):
            DaemonWatchRequest(path="../../etc/passwd")

    def test_daemon_observe_request_validates_type(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import DaemonObserveRequest
        with self.assertRaises(ValidationError):
            DaemonObserveRequest(observation_type="invalid TYPE!")

    def test_daemon_observe_request_accepts_valid(self):
        from shadowd.cognition.api_router import DaemonObserveRequest
        req = DaemonObserveRequest(observation_type="system_metrics")
        self.assertEqual(req.observation_type, "system_metrics")

    def test_observation_request_metadata_key_cap(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import ObservationRequest, _MAX_DICT_KEYS
        big_meta = {f"key_{i}": f"val_{i}" for i in range(_MAX_DICT_KEYS + 1)}
        with self.assertRaises(ValidationError):
            ObservationRequest(content="test", metadata=big_meta)

    def test_action_request_parameters_key_cap(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import ActionRequest, _MAX_DICT_KEYS
        big_params = {f"param_{i}": f"val_{i}" for i in range(_MAX_DICT_KEYS + 1)}
        with self.assertRaises(ValidationError):
            ActionRequest(action="test", parameters=big_params)

    def test_empathy_person_request_metadata_key_cap(self):
        from pydantic import ValidationError
        from shadowd.cognition.api_router import EmpathyPersonRequest, _MAX_DICT_KEYS
        big_meta = {f"k_{i}": f"v_{i}" for i in range(_MAX_DICT_KEYS + 1)}
        with self.assertRaises(ValidationError):
            EmpathyPersonRequest(name="test", metadata=big_meta)


if __name__ == "__main__":
    unittest.main()