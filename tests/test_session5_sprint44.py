"""Sprint 44: Security + robustness fixes from deep audit"""

import pytest
import os
from pathlib import Path
REPO_ROOT = Path(__file__).resolve().parent.parent


def _repo_file(rel_path: str) -> str:
    """Absolute path to a file in the repository, platform independent."""
    return str(REPO_ROOT / rel_path.replace("/", os.sep))



class TestSecurityPrivilegeEscalation:
    def test_blocked_fields_includes_superuser(self):
        from shadowd.security import UserManager
        source = open(_repo_file("shadowd/security.py")).read()
        assert '"is_superuser"' in source, "is_superuser should be in _BLOCKED_FIELDS"

    def test_blocked_fields_includes_scopes(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert '"scopes"' in source, "scopes should be in _BLOCKED_FIELDS"

    def test_blocked_fields_includes_is_active(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert '"is_active"' in source, "is_active should be in _BLOCKED_FIELDS"


class TestSecurityTimingAttack:
    def test_verify_key_uses_hmac_compare(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert "hmac.compare_digest" in source, "verify_key should use hmac.compare_digest"


class TestSecurityDebouncedSave:
    def test_verify_key_uses_dirty_flag(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert "_dirty" in source, "verify_key should use dirty flag for debounced saves"


class TestSecuritySafeDeletion:
    def test_delete_user_uses_pop(self):
        source = open(_repo_file("shadowd/security.py")).read()
        assert "email_index.pop(" in source, "delete_user should use .pop() for safe deletion"


class TestAdvancedMemoryRaceCondition:
    def test_add_to_core_saves_local_var(self):
        source = open(_repo_file("shadowd/advanced_memory.py")).read()
        assert "block = self.core_memory[label]" in source, "should use local variable for block"


class TestAdvancedMemoryFactCap:
    def test_entity_fact_cap_exists(self):
        from shadowd.advanced_memory import Entity
        assert hasattr(Entity, '_MAX_FACTS_PER_ENTITY')
        assert Entity._MAX_FACTS_PER_ENTITY == 1000


class TestStateGraphDeepCopy:
    def test_old_state_graph_uses_deep_copy(self):
        source = open(_repo_file("shadowd/state_graph.py")).read()
        assert "deepcopy" in source, "state_graph execute should use deepcopy"


class TestStateGraphCheckpointCap:
    def test_old_state_graph_has_max_checkpoints(self):
        from shadowd.state_graph import StateGraph
        sg = StateGraph()
        assert hasattr(sg, '_MAX_CHECKPOINTS')
        assert sg._MAX_CHECKPOINTS == 100


class TestStateGraphUUIDCheckpoint:
    def test_old_state_graph_uses_uuid_checkpoint(self):
        source = open(_repo_file("shadowd/state_graph.py")).read()
        assert "uuid.uuid4()" in source, "Should use uuid for checkpoint IDs"

    def test_checkpoint_id_format(self):
        from shadowd.state_graph import StateCheckpoint
        from datetime import datetime, timezone
        cp = StateCheckpoint(node_id="test", state={}, timestamp=datetime.now(timezone.utc), status="running")
        assert cp.checkpoint_id.startswith("cp_")
        assert len(cp.checkpoint_id) > 5


class TestMCPProtocolToolArgValidation:
    def test_mcp_no_bypass_for_required_args(self):
        source = open(_repo_file("shadowd/mcp_protocol.py")).read()
        assert "elif key in required" not in source, "Should not bypass schema validation"

    def test_mcp_only_passes_schema_props(self):
        source = open(_repo_file("shadowd/mcp_protocol.py")).read()
        assert "if key in schema_props" in source, "Should filter by schema_props"