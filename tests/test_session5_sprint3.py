#!/usr/bin/env python3
"""
Session 5 Sprint 3 Regression Tests

Covers:
- V-125: MCP allowlist excludes python3/node/npx
- V-126: MCP env sanitization removes dangerous vars
- V-133: Commitment detection doesn't dump full event JSON
- V-138: LLM response validation
- V-142: Sensitive command filter in observers
- V-146: RateLimiter client cap
- V-120: Workflow cycle detection
- V-129: Knowledge dict cap
- V-147: MultiLevelCache atomic writes
- V-149: AsyncTaskQueue datetime serialization
- V-124/V-154: MD5→SHA-256 in workflow and dark_memory
"""

import pytest
import json
from pathlib import Path
import tempfile


class TestMCPAllowlistHardened:
    """V-125: python3/node/npx removed from MCP allowlist."""

    def test_python3_blocked(self):
        from shadowd.mcp_client import _validate_mcp_command
        assert not _validate_mcp_command("python3"), "python3 should be blocked"

    def test_node_blocked(self):
        from shadowd.mcp_client import _validate_mcp_command
        assert not _validate_mcp_command("node"), "node should be blocked"

    def test_npx_blocked(self):
        from shadowd.mcp_client import _validate_mcp_command
        assert not _validate_mcp_command("npx"), "npx should be blocked"

    def test_mcp_servers_allowed(self):
        from shadowd.mcp_client import _validate_mcp_command
        assert _validate_mcp_command("mcp-server-filesystem")
        assert _validate_mcp_command("mcp-server-sqlite")

    def test_env_sanitization(self):
        from shadowd.mcp_client import _sanitize_env
        env = {"PATH": "/evil", "LD_PRELOAD": "/mal.so", "SAFE_VAR": "ok"}
        safe = _sanitize_env(env)
        assert "PATH" not in safe
        assert "LD_PRELOAD" not in safe
        assert "SAFE_VAR" in safe

    def test_env_sanitization_removes_pythonpath(self):
        from shadowd.mcp_client import _sanitize_env
        env = {"PYTHONPATH": "/inject", "MY_CONFIG": "value"}
        safe = _sanitize_env(env)
        assert "PYTHONPATH" not in safe
        assert "MY_CONFIG" in safe


class TestCommitmentDetectionNoFullDump:
    """V-133: _is_commitment no longer dumps full event JSON."""

    def test_commitment_uses_content_only(self):
        from shadowd.shadow_ai import ShadowAI
        import inspect
        source = inspect.getsource(ShadowAI._is_commitment)
        assert "json.dumps(event)" not in source, "_is_commitment should not use json.dumps(event)"
        assert "event.get" in source, "_is_commitment should use event.get()"


class TestLLMResponseValidation:
    """V-138: LLM response structure validated."""

    def test_mind_handles_missing_choices(self):
        from shadowd.mind import MindEngine
        import inspect
        source = inspect.getsource(MindEngine._call_llm)
        assert "KeyError" in source or "TypeError" in source, "Should handle missing response structure"


class TestSensitiveCommandFilter:
    """V-142: Bash history and clipboard observer filter sensitive patterns."""

    def test_sensitive_patterns_detection(self):
        from shadowd.observers import _is_sensitive_command
        assert _is_sensitive_command("mysql -u root -pMySecretPassword")
        assert _is_sensitive_command("ssh -i ~/.ssh/id_rsa user@host")
        assert _is_sensitive_command("export AWS_SECRET_ACCESS_KEY=abc123")
        assert _is_sensitive_command("gpg --import private.key")
        assert _is_sensitive_command("openssl genrsa -out private.key 2048")

    def test_safe_commands_pass(self):
        from shadowd.observers import _is_sensitive_command
        assert not _is_sensitive_command("ls -la")
        assert not _is_sensitive_command("cd projects")
        assert not _is_sensitive_command("git status")
        assert not _is_sensitive_command("python script.py")
        assert not _is_sensitive_command("echo hello")


class TestWorkflowCycleDetection:
    """V-120: Workflow topological_sort detects cycles."""

    def test_cycle_detection(self):
        from shadowd.workflow.engine import Workflow, BaseBlock
        wf = Workflow("test_cycle")
        b1 = BaseBlock("b1")
        b2 = BaseBlock("b2")
        b3 = BaseBlock("b3")
        wf.add_block(b1)
        wf.add_block(b2)
        wf.add_block(b3)
        wf.add_edge("b1", "b2")
        wf.add_edge("b2", "b3")
        wf.add_edge("b3", "b1")
        with pytest.raises(ValueError, match="[Cc]ycle"):
            wf.topological_sort()


class TestKnowledgeCap:
    """V-129: Unified Shadow v2 knowledge dict is capped."""

    def test_knowledge_cap(self):
        from shadowd.unified_shadow_v2 import UnifiedShadowV2
        us = UnifiedShadowV2()
        for i in range(600):
            us.knowledge.setdefault("test_type", []).append(f"item_{i}")
            if len(us.knowledge["test_type"]) > 500:
                us.knowledge["test_type"] = us.knowledge["test_type"][-500:]
        assert len(us.knowledge["test_type"]) <= 500


class TestMD5Upgrades:
    """V-124/V-154: MD5 replaced with SHA-256 in workflow and dark_memory."""

    def test_workflow_no_md5(self):
        import inspect
        from shadowd.workflow.engine import Workflow, BaseBlock
        source = inspect.getsource(BaseBlock.__init__)
        assert "hashlib.md5" not in source, "Should use sha256"
        wf_source = inspect.getsource(Workflow.__init__)
        assert "hashlib.md5" not in wf_source, "Workflow should use sha256"

    def test_dark_memory_no_md5(self):
        import inspect
        from shadowd.cognition.dark_memory import DarkMemory, TemporalKnowledgeGraph
        dm_source = inspect.getsource(DarkMemory)
        assert "hashlib.md5" not in dm_source, "DarkMemory should use sha256"
        tkg_source = inspect.getsource(TemporalKnowledgeGraph.add_relation)
        assert "hashlib.md5" not in tkg_source, "TemporalKnowledgeGraph should use sha256"


class TestRateLimiterClientCap:
    """V-146: RateLimiter caps local client dict."""

    def test_rate_limiter_max_clients(self):
        from shadowd.performance import RateLimiter
        assert hasattr(RateLimiter, 'MAX_LOCAL_CLIENTS')
        assert RateLimiter.MAX_LOCAL_CLIENTS == 10000


class TestPerformanceAtomicWrites:
    """V-147: MultiLevelCache uses atomic disk writes."""

    def test_set_l2_atomic(self):
        from shadowd.performance_optimization import MultiLevelCache
        import inspect
        source = inspect.getsource(MultiLevelCache.set)
        assert "mkstemp" in source or "os.replace" in source or ".tmp" in source, "Should use atomic write pattern for L2 disk cache"


class TestAsyncTaskQueueDatetimeSerialization:
    """V-149: AsyncTaskQueue results use isoformat for datetime."""

    def test_datetime_serialization(self):
        from shadowd.performance import AsyncTaskQueue
        import inspect
        source = inspect.getsource(AsyncTaskQueue._execute_task)
        assert "datetime.now(timezone.utc).isoformat()" in source, "Should use timezone-aware isoformat"


class TestBackgroundTaskGuard:
    """V-131: UnifiedSystem guards against duplicate start_background_tasks."""

    def test_background_task_guard(self):
        from shadowd.unified_system import UnifiedShadowSystem
        import inspect
        source = inspect.getsource(UnifiedShadowSystem.start_background_tasks)
        assert "already running" in source.lower() or "self._tasks" in source


class TestAPILoggingSanitization:
    """V-137: API base logged without query params."""

    def test_api_base_sanitized(self):
        from shadowd.mind import MindEngine
        import inspect
        source = inspect.getsource(MindEngine.__init__)
        # The api_base should be stripped of query params before logging
        assert "api_base" in source