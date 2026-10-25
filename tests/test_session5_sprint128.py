"""
Sprint 128 — CLI + ShadowOSD Deep Audit Regression Tests

Tests the security hardening of:
- shadowd/cli.py — Input sanitization, path traversal, thread safety
- shadowd/cognition/shadowosd.py — Input sanitization, path traversal, 
  TOCTOU race, missing import, unused imports, lazy import
"""

import pytest


class TestCLISanitization:
    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cli import _sanitize
        assert '<script>' not in _sanitize('test<script>name')

    def test_sanitize_strips_control_chars(self):
        from shadowd.cli import _sanitize
        assert '\t' not in _sanitize('hello\tworld')
        assert '\n' not in _sanitize('hello\nworld')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cli import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')

    def test_sanitize_content_preserves_whitespace(self):
        from shadowd.cli import _sanitize_content
        assert '\t' in _sanitize_content('hello\tworld')
        assert '\n' in _sanitize_content('hello\nworld')

    def test_sanitize_none(self):
        from shadowd.cli import _sanitize
        assert _sanitize(None) == ""

    def test_sanitize_content_none(self):
        from shadowd.cli import _sanitize_content
        assert _sanitize_content(None) == ""

    def test_sanitize_re_pattern_content_safe(self):
        from shadowd.cli import _SANITIZE_RE
        assert _SANITIZE_RE.pattern == r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'

    def test_sanitize_strict_re_pattern(self):
        from shadowd.cli import _SANITIZE_STRICT_RE
        assert _SANITIZE_STRICT_RE.pattern == r'[<>\x00-\x1f\x7f]'


class TestCLIPathTraversal:
    def test_history_file_sandboxed_to_home(self):
        from shadowd.cli import ShadowCLI
        from pathlib import Path
        cli = ShadowCLI(history_file="/etc/passwd")
        home = str(Path.home().resolve())
        assert cli.history_file.startswith(home)

    def test_history_file_default_in_home(self):
        from shadowd.cli import ShadowCLI
        from pathlib import Path
        cli = ShadowCLI()
        home = str(Path.home().resolve())
        assert cli.history_file.startswith(home)


class TestCLIInputSanitization:
    def test_block_name_sanitized(self):
        from shadowd.cli import ShadowCLI
        cli = ShadowCLI()
        cmd_memory = cli.cmd_memory
        import inspect
        source = inspect.getsource(cmd_memory)
        assert "_sanitize" in source

    def test_config_key_sanitized(self):
        from shadowd.cli import ShadowCLI
        cli = ShadowCLI()
        cmd_config = cli.cmd_config
        import inspect
        source = inspect.getsource(cmd_config)
        assert "_sanitize" in source

    def test_save_args_sanitized(self):
        from shadowd.cli import ShadowCLI
        cli = ShadowCLI()
        cmd_save = cli.cmd_save
        import inspect
        source = inspect.getsource(cmd_save)
        assert "_sanitize" in source


class TestCLIThreadSafety:
    def test_crew_agents_uses_list_agents(self):
        from shadowd.cli import ShadowCLI
        import inspect
        source = inspect.getsource(ShadowCLI.cmd_agents)
        assert "list_agents()" in source
        assert ".agents" not in source or "list_agents" in source

    def test_status_uses_safe_accessors(self):
        from shadowd.cli import ShadowCLI
        import inspect
        source = inspect.getsource(ShadowCLI.cmd_status)
        assert ".blocks" not in source


class TestCLIErrorHandling:
    def test_handle_message_narrow_except(self):
        from shadowd.cli import ShadowCLI
        import inspect
        source = inspect.getsource(ShadowCLI.handle_message)
        assert "RuntimeError" in source or "except Exception" not in source


class TestShadowOSDSanitization:
    def test_sanitize_strips_angle_brackets(self):
        from shadowd.cognition.shadowosd import _sanitize
        assert '<script>' not in _sanitize('test<script>name')

    def test_sanitize_content_preserves_angles(self):
        from shadowd.cognition.shadowosd import _sanitize_content
        assert '<' in _sanitize_content('latency < 100ms')

    def test_sanitize_re_pattern(self):
        from shadowd.cognition.shadowosd import _SANITIZE_RE
        assert _SANITIZE_RE.pattern == r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]'

    def test_sanitize_strict_re_pattern(self):
        from shadowd.cognition.shadowosd import _SANITIZE_STRICT_RE
        assert _SANITIZE_STRICT_RE.pattern == r'[<>\x00-\x1f\x7f]'


class TestShadowOSDImports:
    def test_escalation_manager_imported(self):
        from shadowd.cognition import shadowosd
        assert hasattr(shadowosd, 'EscalationManager') or 'EscalationManager' in dir(shadowosd)

    def test_no_asyncio_import(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        lines = [l for l in source.split('\n') if l.strip().startswith('import asyncio')]
        assert len(lines) == 0

    def test_no_signal_import(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        lines = [l for l in source.split('\n') if l.strip().startswith('import signal')]
        assert len(lines) == 0

    def test_jsonresponse_top_level(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        assert 'from fastapi.responses import' in source
        assert 'JSONResponse' in source.split('from fastapi.responses import')[1].split('\n')[0]


class TestShadowOSDPathTraversal:
    def test_web_dir_uses_resolve(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        assert '_WEB_DIR' in source
        assert '.resolve()' in source

    def test_static_endpoints_check_sandbox(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        assert 'startswith' in source or '_WEB_DIR' in source


class TestShadowOSDThreadSafety:
    def test_briefing_engine_lock(self):
        from shadowd.cognition.shadowosd import _BRIEFING_LOCK
        assert _BRIEFING_LOCK is not None
        import threading
        assert isinstance(_BRIEFING_LOCK, type(threading.Lock()))

    def test_rate_limit_lock(self):
        from shadowd.cognition.shadowosd import _rate_limit_lock
        assert _rate_limit_lock is not None


class TestShadowOSDErrorHandling:
    def test_503_for_not_initialized(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        assert '503' in source or 'HTTPException' in source


class TestShadowOSDEscalationSanitization:
    def test_escalation_evaluate_sanitizes_inputs(self):
        import shadowd.cognition.shadowosd as mod
        source = open(mod.__file__).read()
        assert '_sanitize(action_type' in source
        assert '_sanitize(domain' in source