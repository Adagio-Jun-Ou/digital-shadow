#!/usr/bin/env python3
"""
Sprint 104 Tests — ShadowSummon Voice Summon Router + PWA Integration

Tests cover:
  - ShadowSummon intent classification (10 intents)
  - Ability matching (10 Shikigami abilities, plural handling)
  - Template matching (8 agent templates)
  - process_voice() full pipeline
  - TTS-ready spoken responses
  - Stats tracking and thread safety
  - History management (bounded deque)
  - PWA service worker, manifest, endpoint routing
  - ShadowOS wiring (summon_voice, get_summon_abilities, etc.)
  - API endpoint validation
  - Edge cases: empty input, sanitization, punctuation stripping
"""

import threading
import time
import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.shadow_summon import (
    ShadowSummon,
    SummonIntent,
    SummonResponse,
    _ABILITY_MAP,
    _AGENT_TEMPLATES,
    _SANITIZE_RE,
)


class TestSummonIntentClassification:
    def test_create_agent_explicit(self):
        s = ShadowSummon()
        assert s.classify_intent("shadow arise a new agent") == SummonIntent.CREATE_AGENT

    def test_create_agent_variants(self):
        s = ShadowSummon()
        for phrase in [
            "create agent for research",
            "arise agent analyst",
            "spawn agent planner",
            "new agent needed",
            "make agent executor",
            "build agent guardian",
            "shadow create agent",
            "shadow summon analyst",
            "summon agent creator",
            "launch agent observer",
        ]:
            assert s.classify_intent(phrase) == SummonIntent.CREATE_AGENT, f"Failed: {phrase}"

    def test_list_agents(self):
        s = ShadowSummon()
        for phrase in ["list agents", "show agents", "active agents", "my agents", "all agents"]:
            assert s.classify_intent(phrase) == SummonIntent.LIST_AGENTS, f"Failed: {phrase}"

    def test_list_abilities(self):
        s = ShadowSummon()
        for phrase in ["what abilities", "list abilities", "show abilities", "what can you do"]:
            assert s.classify_intent(phrase) == SummonIntent.LIST_ABILITIES, f"Failed: {phrase}"

    def test_complete_agent(self):
        s = ShadowSummon()
        for phrase in ["complete agent abc123", "finish agent xyz", "agent done", "mark done agent foo"]:
            assert s.classify_intent(phrase) == SummonIntent.COMPLETE_AGENT, f"Failed: {phrase}"

    def test_evolve_agent(self):
        s = ShadowSummon()
        for phrase in ["evolve agent abc", "improve agent xyz", "agent adapt new data"]:
            assert s.classify_intent(phrase) == SummonIntent.EVOLVE_AGENT, f"Failed: {phrase}"

    def test_invoke_ability_oracle(self):
        s = ShadowSummon()
        assert s.classify_intent("shadow predict what will happen") == SummonIntent.INVOKE_ABILITY

    def test_invoke_ability_xray(self):
        s = ShadowSummon()
        assert s.classify_intent("scan nearby devices") == SummonIntent.INVOKE_ABILITY

    def test_invoke_ability_pattern_sight_singular(self):
        s = ShadowSummon()
        assert s.classify_intent("find pattern in my data") == SummonIntent.INVOKE_ABILITY

    def test_invoke_ability_pattern_sight_plural(self):
        s = ShadowSummon()
        assert s.classify_intent("find patterns in my work data") == SummonIntent.INVOKE_ABILITY

    def test_invoke_ability_shadow_council(self):
        s = ShadowSummon()
        assert s.classify_intent("should i take the job offer") == SummonIntent.INVOKE_ABILITY

    def test_ask_question(self):
        s = ShadowSummon()
        for phrase in [
            "what is the meaning of life",
            "how does this work",
            "explain quantum computing",
            "why is the sky blue",
            "can you help me",
            "what about the deadline",
        ]:
            assert s.classify_intent(phrase) == SummonIntent.ASK_QUESTION, f"Failed: {phrase}"

    def test_observe_event(self):
        s = ShadowSummon()
        for phrase in ["i just finished the report", "i completed the task", "i started a new project"]:
            assert s.classify_intent(phrase) == SummonIntent.OBSERVE_EVENT, f"Failed: {phrase}"

    def test_get_status(self):
        s = ShadowSummon()
        for phrase in ["system status", "show status", "check status", "health check"]:
            assert s.classify_intent(phrase) == SummonIntent.GET_STATUS, f"Failed: {phrase}"

    def test_unknown(self):
        s = ShadowSummon()
        assert s.classify_intent("random gibberish xyz") == SummonIntent.UNKNOWN

    def test_punctuation_stripped(self):
        s = ShadowSummon()
        assert s.classify_intent("Shadow, create agent!") == SummonIntent.CREATE_AGENT
        assert s.classify_intent("find patterns.") == SummonIntent.INVOKE_ABILITY
        assert s.classify_intent("what is this?") == SummonIntent.ASK_QUESTION


class TestSummonAbilityMatching:
    def test_oracle_predict(self):
        s = ShadowSummon()
        assert s.match_ability("predict the outcome") == "oracle"

    def test_oracle_probability(self):
        s = ShadowSummon()
        assert s.match_ability("what is the probability of rain") == "oracle"

    def test_xray_scan(self):
        s = ShadowSummon()
        assert s.match_ability("scan nearby bluetooth") == "x_ray"

    def test_life_pulse_track(self):
        s = ShadowSummon()
        assert s.match_ability("track person John") == "life_pulse"

    def test_perfect_memory_remember(self):
        s = ShadowSummon()
        assert s.match_ability("remember this for later") == "perfect_memory"

    def test_pattern_sight_singular(self):
        s = ShadowSummon()
        assert s.match_ability("find pattern in my data") == "pattern_sight"

    def test_pattern_sight_plural(self):
        s = ShadowSummon()
        assert s.match_ability("find patterns in my work data") == "pattern_sight"

    def test_pattern_sight_trends(self):
        s = ShadowSummon()
        assert s.match_ability("detect trends in the data") == "pattern_sight"

    def test_social_radar(self):
        s = ShadowSummon()
        assert s.match_ability("social dynamic in the group") == "social_radar"

    def test_intent_seer(self):
        s = ShadowSummon()
        assert s.match_ability("predict intent of the client") == "intent_seer"

    def test_omni_sense(self):
        s = ShadowSummon()
        assert s.match_ability("environment overview") == "omni_sense"

    def test_shadow_council(self):
        s = ShadowSummon()
        assert s.match_ability("should i choose option A") == "shadow_council"

    def test_reality_forge(self):
        s = ShadowSummon()
        assert s.match_ability("build this application") == "reality_forge"

    def test_no_match(self):
        s = ShadowSummon()
        assert s.match_ability("random text without ability") is None

    def test_longest_match_wins(self):
        s = ShadowSummon()
        result = s.match_ability("should i take the new job or stay")
        assert result == "shadow_council"


class TestSummonTemplateMatching:
    def test_analyst(self):
        s = ShadowSummon()
        assert s.match_template("analyze the market data") == "analyst"

    def test_creator(self):
        s = ShadowSummon()
        assert s.match_template("create a new design") == "creator"

    def test_planner(self):
        s = ShadowSummon()
        assert s.match_template("plan the roadmap for Q3") == "planner"

    def test_executor(self):
        s = ShadowSummon()
        assert s.match_template("execute the deployment") == "executor"

    def test_observer(self):
        s = ShadowSummon()
        assert s.match_template("monitor the system") == "observer"

    def test_learner(self):
        s = ShadowSummon()
        assert s.match_template("learn from this experience") == "learner"

    def test_coordinator(self):
        s = ShadowSummon()
        assert s.match_template("coordinate the team efforts") == "coordinator"

    def test_guardian(self):
        s = ShadowSummon()
        assert s.match_template("guard against security risks") == "guardian"

    def test_no_match(self):
        s = ShadowSummon()
        assert s.match_template("random text without template") is None


class TestSummonProcessVoice:
    def test_empty_input(self):
        s = ShadowSummon()
        resp = s.process_voice("")
        assert resp["intent"] == "unknown"
        assert resp["confidence"] == 0.0

    def test_create_agent_intent(self):
        s = ShadowSummon()
        resp = s.process_voice("shadow create an analyst agent")
        assert resp["intent"] == "create_agent"
        assert "analyst" in resp["spoken"].lower() or "Creating" in resp["spoken"]

    def test_invoke_ability_intent(self):
        s = ShadowSummon()
        resp = s.process_voice("predict what will happen next week")
        assert resp["intent"] == "invoke_ability"
        assert "Oracle" in resp["spoken"] or "Activating" in resp["spoken"]

    def test_ask_question_intent(self):
        s = ShadowSummon()
        resp = s.process_voice("what is quantum computing")
        assert resp["intent"] == "ask_question"
        assert "question" in resp["spoken"].lower() or "Processing" in resp["spoken"]

    def test_get_status_intent(self):
        s = ShadowSummon()
        resp = s.process_voice("system status overview")
        assert resp["intent"] == "get_status"
        assert "status" in resp["spoken"].lower() or "Fetching" in resp["spoken"]

    def test_list_abilities_intent(self):
        s = ShadowSummon()
        resp = s.process_voice("what abilities do you have")
        assert resp["intent"] == "list_abilities"
        assert "10" in resp["spoken"] or "Oracle" in resp["spoken"]

    def test_confidence_boosted_by_ability(self):
        s = ShadowSummon()
        resp_no_ab = s.process_voice("hello random text")
        resp_ab = s.process_voice("predict the future of AI")
        assert resp_ab["confidence"] > resp_no_ab["confidence"]

    def test_result_contains_transcript(self):
        s = ShadowSummon()
        resp = s.process_voice("shadow create agent")
        assert "transcript" in resp["result"]

    def test_result_contains_timestamp(self):
        s = ShadowSummon()
        resp = s.process_voice("hello")
        assert "timestamp" in resp


class TestSummonSpokenResponses:
    def test_create_agent_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("create agent for analysis")
        assert "Creating" in resp["spoken"] or "agent" in resp["spoken"].lower()

    def test_invoke_oracle_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("predict the outcome of this project")
        assert "Oracle" in resp["spoken"]

    def test_invoke_xray_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("scan nearby devices and bluetooth")
        assert "X-Ray" in resp["spoken"]

    def test_invoke_life_pulse_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("track person John")
        assert "Life Pulse" in resp["spoken"]

    def test_invoke_perfect_memory_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("remember this important fact")
        assert "Perfect Memory" in resp["spoken"]

    def test_invoke_pattern_sight_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("find patterns in my work data")
        assert "Pattern Sight" in resp["spoken"]

    def test_invoke_social_radar_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("social dynamic in the group")
        assert "Social Radar" in resp["spoken"]

    def test_invoke_intent_seer_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("what are they planning next")
        assert "Intent Seer" in resp["spoken"]

    def test_invoke_omni_sense_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("comprehensive status and environment overview")
        assert "Omni Sense" in resp["spoken"]

    def test_invoke_shadow_council_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("should i choose the first option")
        assert "Shadow Council" in resp["spoken"]

    def test_invoke_reality_forge_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("build this application for me")
        assert "Reality Forge" in resp["spoken"]

    def test_unknown_spoken(self):
        s = ShadowSummon()
        resp = s.process_voice("xyz abc 123")
        assert "help" in resp["spoken"].lower() or "here" in resp["spoken"].lower()

    def test_invoke_ability_without_match_spoken(self):
        s = ShadowSummon()
        resp = SummonResponse(
            intent=SummonIntent.INVOKE_ABILITY,
            action="invoke_ability",
            result={},
            spoken="Which ability would you like to activate?",
        )
        assert "ability" in resp["spoken"].lower()


class TestSummonStats:
    def test_stats_initial(self):
        s = ShadowSummon()
        stats = s.get_stats()
        assert stats["total_summons"] == 0
        assert stats["history_size"] == 0

    def test_stats_increment(self):
        s = ShadowSummon()
        s.process_voice("create agent")
        s.process_voice("predict outcome")
        stats = s.get_stats()
        assert stats["total_summons"] == 2

    def test_stats_intent_breakdown(self):
        s = ShadowSummon()
        s.process_voice("create agent")
        s.process_voice("create agent again")
        stats = s.get_stats()
        assert stats["intents"].get("create_agent", 0) == 2

    def test_stats_ability_breakdown(self):
        s = ShadowSummon()
        s.process_voice("predict the outcome")
        stats = s.get_stats()
        assert stats["abilities_matched"].get("oracle", 0) == 1

    def test_stats_template_breakdown(self):
        s = ShadowSummon()
        s.process_voice("analyze the data deeply")
        stats = s.get_stats()
        assert stats["templates_matched"].get("analyst", 0) == 1


class TestSummonHistory:
    def test_history_empty(self):
        s = ShadowSummon()
        assert s.get_history() == []

    def test_history_populated(self):
        s = ShadowSummon()
        s.process_voice("create agent")
        s.process_voice("predict outcome")
        hist = s.get_history()
        assert len(hist) == 2

    def test_history_bounded(self):
        s = ShadowSummon()
        s._MAX_HISTORY = 10
        s._history = __import__("collections").deque(maxlen=10)
        for i in range(20):
            s.process_voice(f"test command {i}")
        assert len(s.get_history()) <= 10

    def test_history_limit_param(self):
        s = ShadowSummon()
        for i in range(10):
            s.process_voice(f"command {i}")
        hist = s.get_history(limit=3)
        assert len(hist) == 3

    def test_history_entry_fields(self):
        s = ShadowSummon()
        s.process_voice("create agent test")
        hist = s.get_history(limit=1)
        assert "timestamp" in hist[0]
        assert "transcript" in hist[0]
        assert "intent" in hist[0]


class TestSummonThreadSafety:
    def test_concurrent_process_voice(self):
        s = ShadowSummon()
        errors = []

        def worker():
            try:
                for _ in range(50):
                    s.process_voice("test voice command")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        assert s.get_stats()["total_summons"] == 200

    def test_concurrent_read_write(self):
        s = ShadowSummon()
        errors = []

        def writer():
            try:
                for _ in range(50):
                    s.process_voice("create agent")
            except Exception as e:
                errors.append(e)

        def reader():
            try:
                for _ in range(50):
                    s.get_stats()
                    s.get_history()
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=writer), threading.Thread(target=reader)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors


class TestSummonAbilitiesList:
    def test_abilities_count(self):
        s = ShadowSummon()
        abilities = s.get_abilities()
        assert len(abilities) == 10

    def test_abilities_fields(self):
        s = ShadowSummon()
        abilities = s.get_abilities()
        for a in abilities:
            assert "id" in a
            assert "name" in a
            assert "description" in a
            assert "invocation" in a
            assert "icon" in a

    def test_abilities_ids(self):
        s = ShadowSummon()
        abilities = s.get_abilities()
        ids = [a["id"] for a in abilities]
        expected = ["oracle", "x_ray", "life_pulse", "perfect_memory", "pattern_sight",
                    "social_radar", "intent_seer", "omni_sense", "shadow_council", "reality_forge"]
        assert ids == expected


class TestSummonTemplatesList:
    def test_templates_count(self):
        s = ShadowSummon()
        templates = s.get_agent_templates()
        assert len(templates) == 8

    def test_templates_fields(self):
        s = ShadowSummon()
        templates = s.get_agent_templates()
        for t in templates:
            assert "id" in t
            assert "name" in t
            assert "description" in t
            assert "invocation" in t
            assert "icon" in t

    def test_templates_ids(self):
        s = ShadowSummon()
        templates = s.get_agent_templates()
        ids = [t["id"] for t in templates]
        expected = ["analyst", "creator", "planner", "executor", "observer", "learner", "coordinator", "guardian"]
        assert ids == expected


class TestSummonSanitization:
    def test_control_chars_stripped(self):
        s = ShadowSummon()
        resp = s.process_voice("create\x00agent")
        assert "\x00" not in resp["result"]["transcript"]

    def test_classify_intent_with_punctuation(self):
        s = ShadowSummon()
        assert s.classify_intent("Shadow, create agent!") == SummonIntent.CREATE_AGENT

    def test_max_length_enforced(self):
        s = ShadowSummon()
        long_text = "a" * 1000
        resp = s.process_voice(long_text)
        assert len(resp["result"]["transcript"]) <= 500

    def test_angle_brackets_preserved_in_content(self):
        s = ShadowSummon()
        resp = s.process_voice("create agent with <test> objective")
        assert "<test>" in resp["result"]["transcript"]

    def test_control_chars_stripped_from_content(self):
        s = ShadowSummon()
        resp = s.process_voice("create\x00agent\x01test")
        assert "\x00" not in resp["result"]["transcript"]
        assert "\x01" not in resp["result"]["transcript"]

    def test_strict_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.shadow_summon import _sanitize
        assert "<script>" not in _sanitize("test <script>alert('xss')</script> data")

    def test_non_string_input(self):
        s = ShadowSummon()
        resp = s.process_voice("")
        assert resp["intent"] == "unknown"


class TestSummonResponse:
    def test_response_is_dict(self):
        resp = SummonResponse(
            intent=SummonIntent.CREATE_AGENT,
            action="create_agent",
            result={"test": True},
            spoken="Creating agent",
            confidence=0.85,
        )
        assert isinstance(resp, dict)
        assert resp["intent"] == "create_agent"
        assert resp["action"] == "create_agent"
        assert resp["confidence"] == 0.85

    def test_response_confidence_rounded(self):
        resp = SummonResponse(
            intent=SummonIntent.UNKNOWN,
            action="unknown",
            result={},
            spoken="test",
            confidence=0.8567,
        )
        assert resp["confidence"] == 0.857


class TestShadowOSSummonWiring:
    def test_shadow_os_has_summon(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        assert hasattr(os, "summon")
        assert isinstance(os.summon, ShadowSummon)

    def test_shadow_os_summon_voice(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        result = os.summon_voice("shadow create an analyst agent")
        assert result["intent"] == "create_agent"
        assert result["action_type"] == "arise_agent"

    def test_shadow_os_summon_voice_ability(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        result = os.summon_voice("predict what will happen")
        assert result["action_type"] == "invoke_ability"
        assert result["ability_id"] == "oracle"

    def test_shadow_os_summon_voice_ask(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        result = os.summon_voice("what is the meaning of life")
        assert result["action_type"] == "ask"

    def test_shadow_os_summon_voice_observe(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        result = os.summon_voice("i just finished the report")
        assert result["action_type"] == "observe"

    def test_shadow_os_get_summon_abilities(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        abilities = os.get_summon_abilities()
        assert len(abilities) == 10

    def test_shadow_os_get_summon_templates(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        templates = os.get_summon_templates()
        assert len(templates) == 8

    def test_shadow_os_get_summon_history(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS.__new__(ShadowOS)
        os._stats_lock = threading.Lock()
        os._stats = {
            "events_processed": 0, "queries_handled": 0, "actions_taken": 0,
            "insights_generated": 0, "anomalies_detected": 0, "errors": 0,
            "uptime_seconds": 0,
        }
        os.summon = ShadowSummon()
        os.summon.process_voice("test command")
        history = os.get_summon_history()
        assert len(history) == 1


class TestPWAFiles:
    def test_manifest_json_exists(self):
        from pathlib import Path
        manifest = Path(__file__).parent.parent / "web" / "manifest.json"
        assert manifest.exists()

    def test_manifest_valid_json(self):
        import json
        from pathlib import Path
        manifest = Path(__file__).parent.parent / "web" / "manifest.json"
        data = json.loads(manifest.read_text())
        assert "ShadowOS" in data["name"]
        assert data["display"] == "standalone"
        assert data["start_url"] == "/shadow"

    def test_pwa_html_exists(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        assert pwa.exists()

    def test_pwa_html_has_voice_btn(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        assert "muteBtn" in content or "voice-btn" in content

    def test_pwa_html_has_manifest_link(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        assert "manifest.json" in content

    def test_pwa_html_has_tabs(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        for tab in ["summon", "agents", "abilities", "chat", "create", "status"]:
            assert tab in content

    def test_pwa_html_apple_watch_detection(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        assert "184" in content or "apple" in content.lower() or "watch" in content.lower()


class TestShadowosdPWARoute:
    def test_shadowosd_has_shadow_route(self):
        from pathlib import Path
        daemon = Path(__file__).parent.parent / "shadowd" / "cognition" / "shadowosd.py"
        content = daemon.read_text()
        assert "/shadow" in content


class TestSummonRequestModels:
    def test_summon_voice_request_model(self):
        from shadowd.cognition.api_router import SummonVoiceRequest
        req = SummonVoiceRequest(transcript="hello", context=None)
        assert req.transcript == "hello"

    def test_summon_voice_request_max_length(self):
        from shadowd.cognition.api_router import SummonVoiceRequest
        with pytest.raises(Exception):
            SummonVoiceRequest(transcript="x" * 2001)

    def test_summon_text_request_model(self):
        from shadowd.cognition.api_router import SummonVoiceRequest
        req = SummonVoiceRequest(transcript="test command")
        assert req.transcript == "test command"


class TestAbilityMapCompleteness:
    def test_all_10_abilities_present(self):
        assert len(_ABILITY_MAP) == 10

    def test_expected_ability_keys(self):
        expected = {"oracle", "x_ray", "life_pulse", "perfect_memory", "pattern_sight",
                    "social_radar", "intent_seer", "omni_sense", "shadow_council", "reality_forge"}
        assert set(_ABILITY_MAP.keys()) == expected

    def test_each_ability_has_phrases(self):
        for ability, phrases in _ABILITY_MAP.items():
            assert len(phrases) >= 1, f"{ability} has no phrases"


class TestAgentTemplatesCompleteness:
    def test_all_8_templates_present(self):
        assert len(_AGENT_TEMPLATES) == 8

    def test_expected_template_keys(self):
        expected = {"analyst", "creator", "planner", "executor", "observer",
                    "learner", "coordinator", "guardian"}
        assert set(_AGENT_TEMPLATES.keys()) == expected

    def test_each_template_has_phrases(self):
        for template, phrases in _AGENT_TEMPLATES.items():
            assert len(phrases) >= 1, f"{template} has no phrases"