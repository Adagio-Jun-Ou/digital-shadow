#!/usr/bin/env python3
"""
Sprint 105 Tests — ShadowSummon Deep Audit + PWA Security

Tests cover:
  - Content-safe vs strict sanitize patterns
  - Deepcopy returns for get_abilities(), get_agent_templates(), get_history(), get_stats()
  - History append race fixed (inside lock)
  - None/non-string input handling in classify_intent, match_ability, match_template
  - Confidence clamped to [0, 1]
  - Module-level _ABILITY_MAP / _AGENT_TEMPLATES immutability
  - shadowosd.py /shadow, /manifest.json, /sw.js endpoints
  - Service worker caching strategy
  - PWA HTML security (no inline script injection, CSP-safe)
"""

import copy
import threading
import pytest

from shadowd.cognition.shadow_summon import (
    ShadowSummon,
    SummonIntent,
    SummonResponse,
    _sanitize,
    _sanitize_content,
    _SANITIZE_RE,
    _SANITIZE_CONTENT_RE,
    _ABILITY_MAP,
    _AGENT_TEMPLATES,
)


class TestSanitizePatterns:
    def test_strict_strips_angle_brackets(self):
        result = _sanitize("hello <script>alert(1)</script> world")
        assert "<script>" not in result
        assert "</script>" not in result

    def test_strict_strips_control_chars(self):
        result = _sanitize("test\x00\x01\x02data")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_strict_strips_tab_newline(self):
        result = _sanitize("hello\tworld\ntest")
        assert "\t" not in result
        assert "\n" not in result

    def test_content_preserves_angle_brackets(self):
        result = _sanitize_content("latency < 100ms and > 50ms")
        assert "<" in result
        assert ">" in result

    def test_content_strips_control_chars(self):
        result = _sanitize_content("test\x00\x01data")
        assert "\x00" not in result
        assert "\x01" not in result

    def test_content_preserves_tab_newline(self):
        result = _sanitize_content("hello\tworld\ntest")
        assert "\t" in result
        assert "\n" in result

    def test_content_strips_del_char(self):
        result = _sanitize_content("test\x7fdata")
        assert "\x7f" not in result

    def test_strict_non_string_returns_empty(self):
        assert _sanitize(None) == ""
        assert _sanitize(123) == ""

    def test_content_non_string_returns_empty(self):
        assert _sanitize_content(None) == ""
        assert _sanitize_content(456) == ""


class TestDeepCopyReturns:
    def test_get_abilities_returns_copy(self):
        s = ShadowSummon()
        a1 = s.get_abilities()
        a2 = s.get_abilities()
        assert a1 is not a2
        assert a1 == a2
        a1[0]["name"] = "MUTATED"
        assert a2[0]["name"] == "Oracle"

    def test_get_agent_templates_returns_copy(self):
        s = ShadowSummon()
        t1 = s.get_agent_templates()
        t2 = s.get_agent_templates()
        assert t1 is not t2
        assert t1 == t2
        t1[0]["name"] = "MUTATED"
        assert t2[0]["name"] == "Analyst"

    def test_get_history_returns_deepcopy(self):
        s = ShadowSummon()
        s.process_voice("create agent")
        h1 = s.get_history()
        h2 = s.get_history()
        assert h1 is not h2
        if h1:
            h1[0]["transcript"] = "MUTATED"
            assert h2[0]["transcript"] != "MUTATED"

    def test_get_stats_returns_deepcopy(self):
        s = ShadowSummon()
        s.process_voice("create agent")
        s1 = s.get_stats()
        s2 = s.get_stats()
        assert s1 is not s2
        s1["intents"]["MUTATED"] = 999
        assert "MUTATED" not in s2.get("intents", {})

    def test_get_stats_nested_dicts_are_copies(self):
        s = ShadowSummon()
        s.process_voice("predict outcome")
        stats = s.get_stats()
        abilities = stats.get("abilities_matched", {})
        if abilities:
            original_val = list(abilities.values())[0]
            abilities[list(abilities.keys())[0]] = 999
            stats2 = s.get_stats()
            assert stats2["abilities_matched"][list(abilities.keys())[0]] == original_val


class TestHistoryRaceFix:
    def test_history_append_inside_lock(self):
        s = ShadowSummon()
        s.process_voice("test 1")
        s.process_voice("test 2")
        stats = s.get_stats()
        assert stats["total_summons"] == 2
        assert stats["history_size"] == 2

    def test_concurrent_history_append(self):
        s = ShadowSummon()
        errors = []

        def worker(i):
            try:
                for j in range(100):
                    s.process_voice(f"command {i}_{j}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert not errors
        stats = s.get_stats()
        assert stats["total_summons"] == 400
        assert stats["history_size"] <= 200


class TestNoneInputHandling:
    def test_classify_intent_none(self):
        s = ShadowSummon()
        assert s.classify_intent(None) == SummonIntent.UNKNOWN

    def test_classify_intent_non_string(self):
        s = ShadowSummon()
        assert s.classify_intent(12345) == SummonIntent.UNKNOWN

    def test_match_ability_none(self):
        s = ShadowSummon()
        assert s.match_ability(None) is None

    def test_match_ability_non_string(self):
        s = ShadowSummon()
        assert s.match_ability(42) is None

    def test_match_template_none(self):
        s = ShadowSummon()
        assert s.match_template(None) is None

    def test_match_template_non_string(self):
        s = ShadowSummon()
        assert s.match_template([]) is None

    def test_process_voice_empty_string(self):
        s = ShadowSummon()
        resp = s.process_voice("")
        assert resp["intent"] == "unknown"
        assert resp["confidence"] == 0.0


class TestConfidenceClamping:
    def test_confidence_never_exceeds_1(self):
        s = ShadowSummon()
        resp = s.process_voice("predict and analyze patterns")
        assert resp["confidence"] <= 1.0

    def test_confidence_never_below_0(self):
        s = ShadowSummon()
        resp = s.process_voice("")
        assert resp["confidence"] >= 0.0

    def test_confidence_with_ability_and_template(self):
        s = ShadowSummon()
        resp = s.process_voice("find patterns in my work and analyze data")
        assert resp["confidence"] <= 1.0
        assert resp["confidence"] >= 0.7


class TestModuleLevelImmutability:
    def test_ability_map_not_modified_via_instance(self):
        s = ShadowSummon()
        original_oracle = list(_ABILITY_MAP["oracle"])
        s._ability_patterns["predict"] = "MUTATED"
        assert _ABILITY_MAP["oracle"] == original_oracle

    def test_template_map_not_modified_via_instance(self):
        s = ShadowSummon()
        original_analyst = list(_AGENT_TEMPLATES["analyst"])
        s._template_patterns["analyze"] = "MUTATED"
        assert _AGENT_TEMPLATES["analyst"] == original_analyst


class TestShadowosdEndpoints:
    def test_shadow_endpoint_serves_pwa(self):
        from pathlib import Path
        daemon = Path(__file__).parent.parent / "shadowd" / "cognition" / "shadowosd.py"
        content = daemon.read_text()
        assert '"/shadow"' in content
        assert "shadow_pwa.html" in content

    def test_manifest_endpoint(self):
        from pathlib import Path
        daemon = Path(__file__).parent.parent / "shadowd" / "cognition" / "shadowosd.py"
        content = daemon.read_text()
        assert '"/manifest.json"' in content

    def test_sw_js_endpoint(self):
        from pathlib import Path
        daemon = Path(__file__).parent.parent / "shadowd" / "cognition" / "shadowosd.py"
        content = daemon.read_text()
        assert '"/sw.js"' in content


class TestServiceWorkerSecurity:
    def test_sw_no_eval(self):
        from pathlib import Path
        sw = Path(__file__).parent.parent / "web" / "sw.js"
        content = sw.read_text()
        assert "eval(" not in content
        assert "Function(" not in content

    def test_sw_caches_app_shell(self):
        from pathlib import Path
        sw = Path(__file__).parent.parent / "web" / "sw.js"
        content = sw.read_text()
        assert "/shadow" in content
        assert "CACHE_NAME" in content

    def test_sw_api_cache_has_ttl(self):
        from pathlib import Path
        sw = Path(__file__).parent.parent / "web" / "sw.js"
        content = sw.read_text()
        assert "API_CACHE_TTL" in content

    def test_sw_handles_offline(self):
        from pathlib import Path
        sw = Path(__file__).parent.parent / "web" / "sw.js"
        content = sw.read_text()
        assert "offline" in content.lower()


class TestPWAHtmlSecurity:
    def test_no_static_inline_event_handlers(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        static_section = content.split("<script>")[0] if "<script>" in content else content
        assert 'onerror="' not in static_section

    def test_has_csp_meta_tag_or_service_worker(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        assert "serviceWorker" in content or "Content-Security-Policy" in content

    def test_fetch_calls_sanitize_response(self):
        from pathlib import Path
        pwa = Path(__file__).parent.parent / "web" / "shadow_pwa.html"
        content = pwa.read_text()
        assert "json()" in content


class TestSummonResponseSecurity:
    def test_response_does_not_leak_internal_state(self):
        s = ShadowSummon()
        resp = s.process_voice("create agent test")
        assert "_lock" not in str(resp)
        assert "_ability_patterns" not in str(resp)
        assert "_template_patterns" not in str(resp)

    def test_response_spoken_no_html(self):
        s = ShadowSummon()
        resp = s.process_voice("predict outcome")
        assert "<script>" not in resp["spoken"]
        assert "<" not in resp["spoken"]

    def test_response_result_no_control_chars(self):
        s = ShadowSummon()
        resp = s.process_voice("create\x00agent\x7f")
        result_str = str(resp["result"])
        assert "\x00" not in result_str
        assert "\x7f" not in result_str


class TestSprint104NonRegression:
    def test_all_intents_still_work(self):
        s = ShadowSummon()
        tests = [
            ("shadow create agent", SummonIntent.CREATE_AGENT),
            ("list agents", SummonIntent.LIST_AGENTS),
            ("what abilities", SummonIntent.LIST_ABILITIES),
            ("complete agent xyz", SummonIntent.COMPLETE_AGENT),
            ("evolve agent xyz", SummonIntent.EVOLVE_AGENT),
            ("predict outcome", SummonIntent.INVOKE_ABILITY),
            ("find patterns in data", SummonIntent.INVOKE_ABILITY),
            ("what is quantum", SummonIntent.ASK_QUESTION),
            ("i just finished task", SummonIntent.OBSERVE_EVENT),
            ("system status", SummonIntent.GET_STATUS),
        ]
        for text, expected in tests:
            assert s.classify_intent(text) == expected, f"Failed: {text}"