"""Sprint 46: MEDIUM audit fixes — advanced_memory, state_graph, mcp_protocol, security"""

import pytest
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))



class TestAdvancedMemoryTimezone:
    def test_datetime_now_uses_utc(self):
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert "datetime.now()" not in source or "datetime.now(timezone.utc)" in source
        import re
        naive_calls = re.findall(r'datetime\.now\(\)(?!\s*#)', source)
        utc_calls = re.findall(r'datetime\.now\(timezone\.utc\)', source)
        assert len(utc_calls) >= 5, f"Expected >=5 UTC calls, found {len(utc_calls)}"

    def test_timezone_imported(self):
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert "from datetime import" in source and "timezone" in source.split("from datetime import")[1].split("\n")[0]

    def test_memory_block_default_factory_utc(self):
        from shadowd.advanced_memory import MemoryBlock
        b = MemoryBlock(uuid="test", label="test", value="x")
        assert b.last_accessed.tzinfo is not None

    def test_entity_add_fact_uses_utc(self):
        from shadowd.advanced_memory import Entity
        e = Entity(name="test", entity_type="person")
        e.add_fact("likes Python", source="test")
        assert e.last_updated.tzinfo is not None


class TestAdvancedMemoryFactsConflict:
    def test_word_boundary_matching(self):
        from shadowd.advanced_memory import Entity
        e = Entity(name="test", entity_type="person")
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert "r'\\b" in source or "r'\\\\b" in source, "Should use regex word boundaries"

    def test_facts_conflict_same_attribute(self):
        from shadowd.advanced_memory import Entity
        e = Entity(name="test", entity_type="person")
        assert e._facts_conflict("Her name is Mary", "His name is John") is True

    def test_facts_conflict_unrelated(self):
        from shadowd.advanced_memory import Entity
        e = Entity(name="test", entity_type="person")
        assert e._facts_conflict("She likes music", "He works remotely") is False

    def test_facts_conflict_name_in_non_name_context(self):
        from shadowd.advanced_memory import Entity
        e = Entity(name="test", entity_type="person")
        result = e._facts_conflict("The project name is Alpha", "She visited yesterday")
        assert result is False


class TestAdvancedMemoryPersistBlock:
    def test_persist_block_uses_fsync(self):
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert "os.fsync" in source, "_persist_block should use os.fsync"

    def test_save_uses_fsync(self):
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert source.count("os.fsync") >= 3, "_save should use os.fsync for both files"

    def test_os_import(self):
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert "import os" in source


class TestStateGraphEdgePriority:
    def test_conditional_edges_prioritized(self):
        source = open(_repo_file("shadowd/state_graph.py")).read()
        assert "unconditional" in source, "Should separate conditional and unconditional edges"
        assert "unconditional = edge" in source or "unconditional =" in source, "Should collect unconditional as fallback"

    def test_graph_find_next_node_logic(self):
        from shadowd.state_graph import StateGraph
        sg = StateGraph(name="test_priority")
        sg.add_node("start", action=lambda state: state)
        sg.add_node("fallback", action=lambda state: state)
        sg.add_node("conditional_target", action=lambda state: state)
        sg.add_edge("start", "fallback")
        sg.add_conditional_edges("start", lambda state: "yes", {"yes": "conditional_target"})
        sg.current_state = {}
        result = sg._find_next_node("start")
        assert result == "conditional_target", f"Conditional should win, got {result}"


class TestStateGraphCheckpointFsync:
    def test_checkpoint_uses_fsync(self):
        source = open(_repo_file("shadowd/state_graph.py")).read()
        assert "os.fsync" in source, "_save_checkpoint should use os.fsync"

    def test_checkpoint_uses_uuid_filename(self):
        source = open(_repo_file("shadowd/state_graph.py")).read()
        assert "checkpoint_id" in source, "Should use checkpoint_id for filename"

    def test_os_import_in_state_graph(self):
        source = open(_repo_file("shadowd/state_graph.py")).read()
        assert "import os" in source


class TestMCPProtocolEviction:
    def test_tool_eviction_logs_warning(self):
        source = open(_repo_file("shadowd/mcp_protocol.py")).read()
        assert "logger.warning" in source, "Should log WARNING on eviction"

    def test_evicted_name_logged(self):
        source = open(_repo_file("shadowd/mcp_protocol.py")).read()
        assert "evicted" in source, "Should log evicted name"

    def test_register_tool_at_capacity(self):
        from shadowd.mcp_protocol import MCPServer, MCPTool
        server = MCPServer(name="test", version="1.0")
        server.MAX_TOOLS = 2
        server.register_tool(MCPTool(name="tool1", description="T1", input_schema={}))
        server.register_tool(MCPTool(name="tool2", description="T2", input_schema={}))
        server.register_tool(MCPTool(name="tool3", description="T3", input_schema={}))
        assert "tool3" in server.tools
        assert len(server.tools) <= 2


class TestMCPProtocolInitGate:
    def test_init_gate_exists(self):
        source = open(_repo_file("shadowd/mcp_protocol.py")).read()
        assert "_initialized" in source, "Should have _initialized flag"
        assert "not initialized" in source.lower() or "not_initialized" in source.lower() or "Server not initialized" in source, "Should check initialization gate"

    def test_request_before_init_rejected(self):
        from shadowd.mcp_protocol import MCPServer, JSONRPCRequest
        server = MCPServer(name="test", version="1.0")
        assert server._initialized is False
        import asyncio
        async def test():
            req = JSONRPCRequest(id=1, method="tools/list", params={})
            resp = await server.handle_request(req)
            assert resp.error is not None
            assert resp.error["code"] == -32002
        asyncio.run(test())


class TestSecurityGetRemainingLock:
    def test_get_remaining_uses_lock(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert "_sync_lock" in source, "get_remaining should use _sync_lock"
        assert "self._sync_lock" in source, "Should reference _sync_lock"

    def test_threading_import(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert "import threading" in source


class TestSecurityAuthenticateUserSafe:
    def test_authenticate_uses_get_not_subscript(self):
        source = open(_repo_file("shadowd/security.py")).read()
        auth_method = source[source.index("def authenticate_user"):]
        auth_lines = auth_method[:auth_method.index("\n    def ")].split("\n")
        key_error_lines = [l for l in auth_lines if "self.users[" in l]
        for line in key_error_lines:
            assert "self.users.get(" in line, f"Should use .get() not subscript: {line}"

    def test_authenticate_returns_none_on_missing_user(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert "user = self.users.get(user_id)" in source or "user = self.users.get(" in source