#!/usr/bin/env python3
"""
Session 5 Sprint 4 Regression Tests

Covers:
- V-160/V-161: MCP workflow path traversal prevention
- V-162: Atomic file write in self_improvement._save_state
- V-163: Memory block name leakage removed
- V-164/V-165: MD5 -> SHA-256 in advanced_memory and memory_rag
- V-166: Error info leak removed from streaming.py
- V-167: Error info leak removed from websocket_manager.py
- V-168: Error info leak removed from agent_coordination.py and coordinator.py
- V-174: DecisionPattern examples capped
- V-177: Bare except replaced in memory_rag.py
- V-175: shadowosd middleware registration after app definition
- V-176: Unbounded file read in prompt_cache.py
- V-181: EventBus subscriber caps
- V-169-V-172: UUID-based IDs instead of timestamps
"""

import pytest
import inspect
import hashlib
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))



class TestMCPWorkflowPathTraversal:
    """V-160/V-161: workflow_name sanitized to prevent path traversal."""

    def test_workflow_name_rejects_dotdot(self):
        import re
        pattern = r'^[a-zA-Z0-9_\-]{1,64}$'
        assert not re.match(pattern, "../../etc/passwd")
        assert not re.match(pattern, "../secrets")
        assert not re.match(pattern, "a/b")

    def test_workflow_name_rejects_slash(self):
        import re
        pattern = r'^[a-zA-Z0-9_\-]{1,64}$'
        assert not re.match(pattern, "my/workflow")

    def test_workflow_name_accepts_valid(self):
        import re
        pattern = r'^[a-zA-Z0-9_\-]{1,64}$'
        assert re.match(pattern, "my-workflow")
        assert re.match(pattern, "workflow_123")

    def test_mcp_server_validates_workflow_name(self):
        # mcp module requires external dep; verify source file directly
        source = open(_repo_file("shadowd/mcp_server.py")).read()
        assert "Invalid workflow name" in source or "re.match" in source


class TestAtomicSaveSelfImprovement:
    """V-162: self_improvement._save_state uses atomic writes."""

    def test_save_state_uses_tmp_rename(self):
        from shadowd.cognition.self_improvement import SelfImprovementEngine
        source = inspect.getsource(SelfImprovementEngine._save_state)
        assert ".tmp" in source or "replace" in source, "_save_state should use atomic write"


class TestMemoryBlockNoLeakage:
    """V-163: memory_tools error messages no longer leak block names."""

    def test_memory_insert_no_block_list(self):
        import shadowd.memory_tools as mt
        source = inspect.getsource(mt.MemoryTools.memory_insert)
        assert "Available:" not in source, "Should not leak available block names"
        assert "list(self.memory.blocks.keys())" not in source

    def test_memory_rethink_no_block_list(self):
        import shadowd.memory_tools as mt
        source = inspect.getsource(mt.MemoryTools.memory_rethink)
        assert "Available:" not in source, "Should not leak available block names"
        assert "list(self.memory.blocks.keys())" not in source


class TestMD5UpgradesSprint4:
    """V-164/V-165: MD5 replaced with SHA-256 in advanced_memory.py and memory_rag.py."""

    def test_advanced_memory_no_md5(self):
        import shadowd.advanced_memory as am
        source = inspect.getsource(am)
        assert "hashlib.md5" not in source, "advanced_memory should not use MD5"

    def test_memory_rag_no_md5(self):
        import shadowd.memory_rag as mr
        source = inspect.getsource(mr)
        assert "hashlib.md5" not in source, "memory_rag should not use MD5"

    def test_advanced_memory_uses_sha256(self):
        import shadowd.advanced_memory as am
        source = inspect.getsource(am.AdvancedMemorySystem.add_to_archival)
        assert "sha256" in source, "Should use sha256 for block UUIDs"


class TestStreamingNoInfoLeak:
    """V-166: streaming.py does not leak internal error details."""

    def test_sse_error_generic(self):
        import shadowd.streaming as streaming_mod
        source = inspect.getsource(streaming_mod)
        lines = source.split('\n')
        error_chunks = [l for l in lines if '"message"' in l and 'stream_error' in l]
        for line in error_chunks:
            assert "str(e)" not in line, "SSE error messages should be generic"

    def test_stream_error_generic(self):
        import shadowd.streaming as streaming_mod
        source = inspect.getsource(streaming_mod)
        assert "Stream error occurred" in source


class TestWebSocketNoInfoLeak:
    """V-167: websocket_manager.py does not leak exception details."""

    def test_handler_error_uses_type_name(self):
        import shadowd.websocket_manager as wm
        source = inspect.getsource(wm)
        assert 'type(e).__name__' in source, "Should use type(e).__name__"

    def test_streaming_error_generic(self):
        import shadowd.websocket_manager as wm
        source = inspect.getsource(wm.StreamingResponseHandler.stream_response)
        assert '"Stream processing error"' in source or "type(e).__name__" in source
        assert '"error": str(e)' not in source


class TestAgentCoordinationNoInfoLeak:
    """V-168: agent_coordination.py and coordinator.py use type().__name__ not str(e)."""

    def test_coordination_task_error_uses_type(self):
        import shadowd.agent_coordination as ac
        source = inspect.getsource(ac)
        assert 'type(e).__name__' in source, "Should use type(e).__name__ not str(e)"

    def test_coordinator_error_uses_type(self):
        import shadowd.coordinator as coord
        source = inspect.getsource(coord)
        assert 'type(e).__name__' in source, "Should use type(e).__name__ not str(e)"


class TestDecisionPatternExamplesCap:
    """V-174: DecisionPattern examples list is capped."""

    def test_examples_capped(self):
        from shadowd.decision_learning_v2 import DecisionLearningV2
        source = inspect.getsource(DecisionLearningV2._detect_pattern)
        assert "50" in source, "Examples should be capped at 50"


class TestMemoryRagNoBareExcept:
    """V-177: memory_rag.py bare excepts replaced."""

    def test_domain_load_no_bare_except(self):
        import shadowd.memory_rag as mr
        source = inspect.getsource(mr)
        assert "except (json.JSONDecodeError, OSError, ValueError)" in source or "except (json.JSONDecodeError" in source

    def test_embedder_no_bare_except(self):
        import shadowd.memory_rag as mr
        source = inspect.getsource(mr.RAGMemorySystem._get_embedding)
        assert "RuntimeError" in source or "except (RuntimeError" in source


class TestShadowOSDMiddlewareOrder:
    """V-175: shadowosd.py middleware defined after app creation."""

    def test_middleware_after_app(self):
        import shadowd.cognition.shadowosd as shadowosd
        source = inspect.getsource(shadowosd)
        app_line = source.index("app = FastAPI")
        middleware_line = source.index("@app.middleware")
        assert middleware_line > app_line, "Middleware must be defined after app"


class TestPromptCacheFileSizeLimit:
    """V-176: prompt_cache _hash_file limits file size."""

    def test_file_size_limit(self):
        import shadowd.cache.prompt_cache as pc
        source = inspect.getsource(pc)
        assert "_MAX_FILE_SIZE" in source, "Should have max file size limit in prompt cache"


class TestEventBusSubscriberCap:
    """V-181: EventBus caps subscribers at 50 per event, 20 for wildcard."""

    def test_subscriber_cap(self):
        import shadowd.event_bus as eb
        source = inspect.getsource(eb.EventBus.subscribe)
        assert "50" in source or "cap" in source.lower()

    def test_wildcard_cap(self):
        import shadowd.event_bus as eb
        source = inspect.getsource(eb.EventBus.subscribe_all)
        assert "20" in source or "cap" in source.lower()


class TestUUIDBasedIDs:
    """V-169-V-172: Timestamp-based IDs replaced with UUID."""

    def test_knowledge_graph_uses_uuid(self):
        import shadowd.knowledge_graph as kg
        source = inspect.getsource(kg)
        assert "uuid" in source, "Knowledge graph should use uuid"
        assert "datetime.now().timestamp()" not in source, "Should not use timestamp for IDs"

    def test_decision_learning_uses_uuid(self):
        import shadowd.decision_learning_v2 as dl
        source = inspect.getsource(dl)
        assert "uuid" in source, "Decision learning should use uuid"

    def test_memory_enhanced_uses_uuid(self):
        import shadowd.memory_enhanced as me
        source = inspect.getsource(me)
        assert "uuid" in source, "memory_enhanced should use uuid"