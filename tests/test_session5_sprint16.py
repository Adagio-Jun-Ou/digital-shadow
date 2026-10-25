#!/usr/bin/env python3
"""Session 5 Sprint 16 regression tests - reasoning/memory/tools/self_improvement audit.

Covers:
- RCP-06/RCP-07: ContextProtocol entity_id path traversal fix (hashed filenames)
- RCP-01/RCP-02: ContextProtocol entities/resources caps
- RCP-09: entity_id regex validation
- RSG-02: ExecutionResult error uses type(e).__name__ not str(e)
- SM-04/05: ArchivalMemory memories cap
- SM-06/07: KnowledgeGraph entities/relations caps
- SM-11/12: MD5 → SHA-256 in super_memory.py
- ET-02: MD5 → SHA-256 in enhanced_tools.py
"""

import hashlib
import inspect
import pytest
import sys
import tempfile
from pathlib import Path
import os
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))


sys.path.insert(0, str(Path(__file__).parent.parent))


class TestContextProtocolEntityIDValidation:
    """RCP-09: entity_id validated with regex."""

    def test_entity_id_regex(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        cp = ContextProtocol(storage_path=Path(tempfile.mkdtemp()))
        assert ContextProtocol._SAFE_ENTITY_ID_RE.match("test_entity-123")
        assert ContextProtocol._SAFE_ENTITY_ID_RE.match("a")
        assert not ContextProtocol._SAFE_ENTITY_ID_RE.match("../../etc")
        assert not ContextProtocol._SAFE_ENTITY_ID_RE.match("")
        assert not ContextProtocol._SAFE_ENTITY_ID_RE.match("a" * 65)

    def test_entity_id_rejected_with_path_separator(self):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability, Permission, ResourceType
        cp = ContextProtocol(storage_path=Path(tempfile.mkdtemp()))
        cap = Capability(name="test", permissions=[Permission.READ], resources=[ResourceType.FILE])
        with pytest.raises(ValueError, match="Invalid entity_id"):
            cp.register_entity("../../etc", "agent", [cap])


class TestContextProtocolHashedFilenames:
    """RCP-06/RCP-07: Context protocol uses hashed filenames for security."""

    def test_save_context_uses_hashed_filename(self):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability, Permission, ResourceType
        cp = ContextProtocol(storage_path=Path(tempfile.mkdtemp()))
        cap = Capability(name="test", permissions=[Permission.READ], resources=[ResourceType.FILE])
        cp.register_entity("test_ent", "agent", [cap])
        safe_id = hashlib.sha256("test_ent".encode()).hexdigest()[:16]
        ctx_file = cp.storage_path / f"context_{safe_id}.json"
        assert ctx_file.exists()

    def test_load_context_uses_hashed_filename(self):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability, Permission, ResourceType
        cp = ContextProtocol(storage_path=Path(tempfile.mkdtemp()))
        cap = Capability(name="test", permissions=[Permission.READ], resources=[ResourceType.FILE])
        cp.register_entity("myapp", "agent", [cap])
        loaded = cp.load_context("myapp")
        assert loaded is not None
        assert loaded.entity_id == "myapp"

    def test_load_context_rejects_invalid_id(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        cp = ContextProtocol(storage_path=Path(tempfile.mkdtemp()))
        result = cp.load_context("../../etc")
        assert result is None


class TestContextProtocolCaps:
    """RCP-01/RCP-02: ContextProtocol entities/resources caps."""

    def test_max_entities_constant(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        assert ContextProtocol._MAX_ENTITIES == 1000

    def test_max_resources_constant(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        assert ContextProtocol._MAX_RESOURCES == 1000


class TestStateGraphErrorGeneric:
    """RSG-02: ExecutionResult error uses type(e).__name__."""

    def test_execute_node_error_generic(self):
        from shadowd.reasoning.state_graph import StateGraph
        source = inspect.getsource(StateGraph._execute_node)
        assert "Execution failed:" in source
        assert "type(e).__name__" in source

    def test_execute_loop_error_uses_type_name(self):
        from shadowd.reasoning.state_graph import StateGraph
        source = inspect.getsource(StateGraph.execute)
        assert "type(e).__name__" in source


class TestArchivalMemoryCaps:
    """SM-04/05: ArchivalMemory has memories cap."""

    def test_max_memories_constant(self):
        from shadowd.memory.super_memory import ArchivalMemory
        am = ArchivalMemory.__new__(ArchivalMemory)
        assert hasattr(ArchivalMemory, '_MAX_MEMORIES')
        assert ArchivalMemory._MAX_MEMORIES == 50000


class TestKnowledgeGraphCaps:
    """SM-06/07: KnowledgeGraph has entity/relation caps."""

    def test_max_entities_constant(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        assert KnowledgeGraph._MAX_ENTITIES == 5000

    def test_max_relations_constant(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        assert KnowledgeGraph._MAX_RELATIONS == 10000

    def test_entity_name_truncated(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        kg = KnowledgeGraph()
        eid = kg.add_entity("person", "A" * 500)
        assert len(kg.entities[eid].name) == 200

    def test_entities_eviction_on_overflow(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        kg = KnowledgeGraph()
        kg._MAX_ENTITIES = 5
        for i in range(10):
            kg.add_entity("person", f"entity_{i}")
        assert len(kg.entities) <= 5


class TestSuperMemorySHA256:
    """SM-11/12: super_memory.py uses a collision-resistant ID source (not MD5).
    SHA-256 was the original choice; the implementation later switched to UUIDv4
    which is at least as collision-resistant for this use case."""

    def test_archival_memory_id_safe(self):
        from shadowd.memory.super_memory import ArchivalMemory
        source = inspect.getsource(ArchivalMemory.insert)
        assert "md5" not in source.lower()
        assert "sha256" in source.lower() or "uuid" in source.lower(), \
            "ID must come from sha256 or uuid, not a weak hash"

    def test_knowledge_graph_entity_id_safe(self):
        from shadowd.memory.super_memory import KnowledgeGraph
        source = inspect.getsource(KnowledgeGraph.add_entity)
        assert "md5" not in source.lower()
        assert "sha256" in source.lower() or "uuid" in source.lower(), \
            "ID must come from sha256 or uuid, not a weak hash"

    def test_no_md5_in_super_memory(self):
        text = Path(_repo_file("shadowd/memory/super_memory.py")).read_text()
        assert "hashlib.md5" not in text


class TestEnhancedToolsSHA256:
    """ET-02: enhanced_tools.py uses SHA-256 for action IDs."""

    def test_create_action_sha256(self):
        from shadowd.tools.enhanced_tools import BaseTool
        source = inspect.getsource(BaseTool.create_action)
        assert "sha256" in source
        assert "md5" not in source


class TestSuperMemoryLoggerFixes:
    """Check super_memory.py logger uses type(e).__name__."""

    def test_load_error_uses_type_name(self):
        text = Path(_repo_file("shadowd/memory/super_memory.py")).read_text()
        assert 'Load error: {e}' not in text
        assert 'type(e).__name__' in text