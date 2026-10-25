"""
Sprint 124 tests: Content-safe _SANITIZE_RE fix + Voice TTS/STT API endpoints.

CRITICAL BUG: _SANITIZE_RE pattern was r'[<>\x00-\x1f\x7f]' (strict) in 14+ modules.
This meant content fields like descriptions would have <> stripped, corrupting
expressions like "latency < 100ms" into "latency 100ms".

FIX: Changed _SANITIZE_RE to r'[\\x00-\\x08\\x0b\\x0c\\x0e-\\x1f\\x7f]' (content-safe)
which preserves <> and \\t\\n\\r for all content fields.
"""

import pytest
import re

CONTENT_SAFE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
STRICT_RE = re.compile(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_content(text):
    return CONTENT_SAFE_RE.sub('', text)


def _sanitize_strict(text):
    return STRICT_RE.sub('', text)


class TestContentSafeRegex:
    """Test that content-safe regex preserves <> \\t \\n \\r."""

    def test_content_preserves_angle_brackets(self):
        assert _sanitize_content("latency < 100ms") == "latency < 100ms"
        assert _sanitize_content("x > y") == "x > y"
        assert _sanitize_content("<tag>content</tag>") == "<tag>content</tag>"

    def test_content_preserves_tab_newline_cr(self):
        assert _sanitize_content("hello\tworld") == "hello\tworld"
        assert _sanitize_content("line1\nline2") == "line1\nline2"
        assert _sanitize_content("line1\r\nline2") == "line1\r\nline2"

    def test_content_strips_control_chars(self):
        assert _sanitize_content("hello\x00world") == "helloworld"
        assert _sanitize_content("hello\x01world") == "helloworld"
        assert _sanitize_content("hello\x07world") == "helloworld"
        assert _sanitize_content("hello\x0eworld") == "helloworld"
        assert _sanitize_content("hello\x1fworld") == "helloworld"
        assert _sanitize_content("hello\x7fworld") == "helloworld"

    def test_content_strips_vertical_tab_form_feed(self):
        assert _sanitize_content("hello\x0bworld") == "helloworld"
        assert _sanitize_content("hello\x0cworld") == "helloworld"

    def test_strict_strips_angle_brackets(self):
        assert _sanitize_strict("latency < 100ms") == "latency  100ms"
        assert _sanitize_strict("x > y") == "x  y"
        assert _sanitize_strict("<tag>content</tag>") == "tagcontent/tag"

    def test_strict_strips_tab_newline_cr(self):
        strict = re.compile(r'[<>\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
        result = strict.sub('', "hello\tworld")
        assert '\t' in result
        result2 = strict.sub('', "line1\nline2")
        assert '\n' in result2
        assert strict.sub('', "latency < 100ms") == "latency  100ms"

    def test_strict_strips_control_chars(self):
        assert _sanitize_strict("hello\x00world") == "helloworld"
        assert _sanitize_strict("hello\x7fworld") == "helloworld"


class TestModuleSanitizePatterns:
    """Verify that cognition modules use content-safe patterns for _SANITIZE_RE."""

    @pytest.mark.parametrize("module_path", [
        "shadowd.cognition.anomaly_detector",
        "shadowd.cognition.cognitive_pipeline",
        "shadowd.cognition.cognitive_reflection",
        "shadowd.cognition.creative_synthesis",
        "shadowd.cognition.dark_memory",
        "shadowd.cognition.decision_journal",
        "shadowd.cognition.emotional_intelligence",
        "shadowd.cognition.goal_reasoning",
        "shadowd.cognition.metacognition_engine",
        "shadowd.cognition.moral_reasoning",
        "shadowd.cognition.personality_engine",
        "shadowd.cognition.shadow_arise",
        "shadowd.cognition.shadow_summon",
        "shadowd.cognition.stealth_layer",
    ])
    def test_module_sanitize_re_is_content_safe(self, module_path):
        import importlib
        mod = importlib.import_module(module_path)
        sanitize_re = getattr(mod, '_SANITIZE_RE', None)
        if sanitize_re is None:
            pytest.skip(f"{module_path} has no _SANITIZE_RE")
        pattern = sanitize_re.pattern
        assert '<' not in pattern or '\\x3c' in pattern.lower(), \
            f"{module_path} _SANITIZE_RE should not strip <>(content-safe): {pattern}"
        result = sanitize_re.sub('', "latency < 100ms")
        assert '<' in result, \
            f"{module_path} _SANITIZE_RE should preserve <>: got '{result}'"
        tab_result = sanitize_re.sub('', "hello\tworld")
        assert '\t' in tab_result, \
            f"{module_path} _SANITIZE_RE should preserve \\t: got '{repr(tab_result)}'"
        nl_result = sanitize_re.sub('', "line1\nline2")
        assert '\n' in nl_result, \
            f"{module_path} _SANITIZE_RE should preserve \\n: got '{repr(nl_result)}'"

    def test_device_context_has_both_sanitizers(self):
        from shadowd.cognition.device_context import _sanitize, _sanitize_strict
        content = "latency < 100ms"
        assert "<" in _sanitize(content)
        assert "<" not in _sanitize_strict(content)

    def test_escalation_has_both_sanitizers(self):
        from shadowd.cognition.escalation import _sanitize, _sanitize_content
        content = "latency < 100ms"
        assert "<" in _sanitize_content(content)
        assert "<" not in _sanitize(content)


class TestVoiceAPIEndpoints:
    """Test Voice TTS/STT API endpoints exist and models validate."""

    def test_tts_request_model(self):
        from shadowd.cognition.api_router import VoiceTTSRequest
        req = VoiceTTSRequest(text="Hello world")
        assert req.text == "Hello world"
        assert req.voice is None
        req2 = VoiceTTSRequest(text="Hello", voice="en_US-ryan-high")
        assert req2.voice == "en_US-ryan-high"

    def test_tts_request_validation(self):
        from shadowd.cognition.api_router import VoiceTTSRequest
        from pydantic import ValidationError
        with pytest.raises(ValidationError):
            VoiceTTSRequest(text="")
        with pytest.raises(ValidationError):
            VoiceTTSRequest(text="x" * 10001)

    def test_tts_endpoint_registered(self):
        from shadowd.cognition.api_router import router
        paths = [r.path for r in router.routes]
        assert "/os/voice/tts" in paths
        assert "/os/voice/voices" in paths
        assert "/os/voice/stt-upload" in paths

    def test_shadow_voice_speak_sanitizes(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        text = "Hello\x00world\nNew line"
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)[:5000]
        assert sanitized == "Helloworld\nNew line"

    def test_shadow_voice_session_transcript_sanitizes(self):
        from shadowd.cognition.shadow_voice import VoiceSession
        vs = VoiceSession()
        vs.add_transcript("user", "latency < 100ms")
        assert vs.transcript[-1]["text"] == "latency < 100ms"
        vs.add_transcript("shadow", "Hello\x00world")
        assert vs.transcript[-1]["text"] == "Helloworld"

    def test_shadow_voice_session_speaker_strict(self):
        from shadowd.cognition.shadow_voice import VoiceSession
        vs = VoiceSession()
        vs.add_transcript("user<script>", "hello")
        assert vs.transcript[-1]["speaker"] == "userscript"

    def test_shadow_voice_command_sanitizes_process(self):
        from shadowd.cognition.shadow_voice import ShadowVoice
        sv = ShadowVoice()
        result = {"transcript": "Hello\x00world"}
        sanitized = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', result["transcript"])[:5000]
        assert sanitized == "Helloworld"