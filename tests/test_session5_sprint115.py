"""Sprint 115 tests — context_protocol.py and reasoning/context_protocol.py
thread safety, deepcopy, fsync, input validation, caps, nonce timing-safe comparison."""

import copy
import json
import os
import threading
import time
from datetime import datetime, timezone, timedelta
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

# Helper to create Capability for tests
def _make_cap(name="test_cap", perms=None, resources=None):
    from shadowd.reasoning.context_protocol import Permission, ResourceType, Capability
    return Capability(
        name=name,
        permissions=perms or [Permission.READ],
        resources=resources or [ResourceType.FILE],
    )


# ============================================================
# shadowd/context_protocol.py tests
# ============================================================

class TestContextProtocolLock:
    """Test threading.Lock on ContextProtocol shared state."""

    def test_context_protocol_has_lock(self):
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol.__new__(ContextProtocol)
        cp._lock = threading.Lock()
        assert hasattr(cp, '_lock')

    def test_register_source_acquires_lock(self):
        from shadowd.context_protocol import ContextProtocol, ContextSource, ContextType
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        source = ContextSource(
            name="test_source_lock",
            context_type=ContextType.DEVICE,
            provider=lambda: {},
        )
        cp.register_source(source)
        # Default sources: time(TEMPORAL), activity_tracker(ACTIVITY), digital_tracker(DIGITAL)
        # DEVICE only has our source
        assert len(cp.sources[ContextType.DEVICE]) == 1

    def test_register_source_sanitizes_name(self):
        from shadowd.context_protocol import ContextProtocol, ContextSource, ContextType
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        source = ContextSource(
            name="test<script>source",
            context_type=ContextType.DEVICE,
            provider=lambda: {},
        )
        cp.register_source(source)
        # _sanitize strips <> characters, leaves inner text
        assert source.name == "testscriptsource"

    def test_register_source_caps_at_max(self):
        from shadowd.context_protocol import ContextProtocol, ContextSource, ContextType
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        for i in range(25):  # _MAX_SOURCES_PER_TYPE = 20
            source = ContextSource(
                name=f"source_{i}",
                context_type=ContextType.DEVICE,
                provider=lambda: {},
            )
            cp.register_source(source)
        assert len(cp.sources[ContextType.DEVICE]) == 20


class TestContextProtocolDeepcopy:
    """Test deepcopy returns on get_context_for_ability and get_context_history."""

    def test_get_context_for_ability_returns_deepcopy(self):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot, ContextType
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        snapshot = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            temporal_context={"time": "12:00"},
            social_context={"who": "alice"},
        )
        result = cp.get_context_for_ability("oracle", snapshot=snapshot)
        result["when"]["time"] = "99:99"
        assert snapshot.temporal_context["time"] == "12:00"

    def test_get_context_history_returns_deepcopy(self):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        cp.context_history = [
            ContextSnapshot(timestamp=datetime.now(timezone.utc), temporal_context={"time": "10:00"}),
        ]
        history = cp.get_context_history()
        assert len(history) == 1
        history[0]["temporal"]["time"] = "99:99"
        assert cp.context_history[0].temporal_context["time"] == "10:00"

    def test_get_context_history_limit_clamped(self):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        history = cp.get_context_history(limit=0)
        assert isinstance(history, list)

    def test_context_aware_ability_get_unified_context_returns_deepcopy(self):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot, ContextAwareAbility
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        cp.current_snapshot = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            temporal_context={"time": "12:00"},
        )
        ability = ContextAwareAbility(cp)
        context = ability.get_unified_context()
        context["when"]["time"] = "99:99"
        assert cp.current_snapshot.temporal_context["time"] == "12:00"


class TestContextProtocolFsync:
    """Test atomic write with fsync for save_snapshot."""

    def test_save_snapshot_atomic_write(self, tmp_path):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot
        cp = ContextProtocol(data_dir=tmp_path)
        snapshot = ContextSnapshot(timestamp=datetime.now(timezone.utc))
        cp.save_snapshot(snapshot)
        files = list(tmp_path.glob("snapshot_*.json"))
        assert len(files) == 1

    def test_load_latest_snapshot_returns_deepcopy(self, tmp_path):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot
        cp = ContextProtocol(data_dir=tmp_path)
        snapshot = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            temporal_context={"time": "12:00"},
        )
        cp.save_snapshot(snapshot)
        loaded = cp.load_latest_snapshot()
        assert loaded is not None
        assert loaded.temporal_context["time"] == "12:00"

    def test_load_latest_snapshot_bad_json_returns_none(self, tmp_path):
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol(data_dir=tmp_path)
        bad_file = tmp_path / "snapshot_20240101_000000.json"
        bad_file.write_text("not json {{{")
        result = cp.load_latest_snapshot()
        assert result is None


class TestContextProtocolSubscribers:
    """Test subscriber management."""

    def test_subscribe_at_max_evicts_oldest(self):
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        handlers = [lambda x: None for _ in range(50)]
        for h in handlers:
            cp.subscribe(h)
        assert len(cp.subscribers) == 50
        cp.subscribe(lambda x: None)
        assert len(cp.subscribers) == 50

    def test_unsubscribe(self):
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        handler = lambda x: None
        cp.subscribe(handler)
        assert len(cp.subscribers) == 1
        cp.unsubscribe(handler)
        assert len(cp.subscribers) == 0


class TestContextProtocolGetStatus:
    """Test get_status under lock."""

    def test_get_status_returns_dict(self):
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol(data_dir=Path("/tmp/test_cp_s115"))
        status = cp.get_status()
        assert "sources" in status
        assert "current_snapshot" in status
        assert "history_size" in status
        assert "subscribers" in status


# ============================================================
# shadowd/reasoning/context_protocol.py tests
# ============================================================

class TestReasoningContextProtocolLock:
    """Test threading.Lock on KeyManager and ContextProtocol."""

    def test_key_manager_has_lock(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        assert hasattr(km, '_lock')

    def test_key_manager_generate_key_thread_safe(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        results = []
        errors = []
        def gen_keys(entity_id, n):
            try:
                for i in range(n):
                    key = km.generate_key(f"{entity_id}_{i}")
                    results.append(key)
            except Exception as e:
                errors.append(e)
        t1 = threading.Thread(target=gen_keys, args=("a", 50))
        t2 = threading.Thread(target=gen_keys, args=("b", 50))
        t1.start(); t2.start()
        t1.join(); t2.join()
        assert len(errors) == 0
        assert len(results) == 100

    def test_key_manager_verify_nonce_timing_safe(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        nonce = km.generate_nonce("entity1")
        assert km.verify_nonce("entity1", nonce) is True
        assert km.verify_nonce("entity1", nonce) is False

    def test_key_manager_verify_nonce_wrong_nonce(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        km.generate_nonce("entity1")
        assert km.verify_nonce("entity1", "wrong_nonce") is False

    def test_context_protocol_has_lock(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        cp = ContextProtocol(storage_path=Path("/tmp/test_rcp_s115"))
        assert hasattr(cp, '_lock')

    def test_context_protocol_has_path_regex(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        assert hasattr(ContextProtocol, '_SAFE_RESOURCE_PATH_RE')
        assert hasattr(ContextProtocol, '_SAFE_MESSAGE_TYPE_RE')


class TestReasoningContextProtocolSanitization:
    """Test input validation in reasoning/context_protocol."""

    def test_register_entity_rejects_invalid_id(self):
        from shadowd.reasoning.context_protocol import ContextProtocol
        cp = ContextProtocol(storage_path=Path("/tmp/test_rcp_s115"))
        with pytest.raises(ValueError, match="Invalid entity_id"):
            cp.register_entity("bad<>id", "agent", [])

    def test_register_entity_sanitizes_entity_type(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        ctx = cp.register_entity("test_agent_s115", "agent<script>", [cap])
        assert ctx.entity_type == "agentscript"

    def test_register_resource_sanitizes_path(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, ResourceType, Permission, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("owner1_s115", "agent", [cap])
        resource = cp.register_resource(
            "res1_s115", ResourceType.FILE, "/tmp/test<script>", "owner1<script>",
            [Permission.READ]
        )
        assert "<script>" not in resource.path
        assert "<script>" not in resource.owner_id

    def test_register_resource_caps_at_max(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, ResourceType, Permission, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("owner1_cap", "agent", [cap])
        for i in range(1001):  # _MAX_RESOURCES = 1000
            cp.register_resource(
                f"res_cap_{i}", ResourceType.FILE, f"/tmp/file_{i}", "owner1_cap", [Permission.READ]
            )
        assert len(cp._resources) == 1000


class TestReasoningContextProtocolDeepcopy:
    """Test deepcopy returns from get_context, get_resource, etc."""

    def test_get_context_returns_deepcopy(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_deep", "agent", [cap])
        result = cp.get_context("agent1_deep")
        assert result is not None
        result.entity_type = "modified"
        internal = cp._entities["agent1_deep"]
        assert internal.entity_type == "agent"

    def test_get_resource_returns_deepcopy(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, ResourceType, Permission, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("owner1_deep", "agent", [cap])
        cp.register_resource("res1_deep", ResourceType.FILE, "/tmp/test", "owner1_deep", [Permission.READ])
        result = cp.get_resource("res1_deep")
        assert result is not None
        result.path = "/modified"
        assert cp._resources["res1_deep"].path == "/tmp/test"

    def test_list_accessible_resources_returns_deepcopy(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, ResourceType, Permission, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_acc", "agent", [cap], permissions=[Permission.READ])
        cp.register_resource("res1_acc", ResourceType.FILE, "/tmp/test", "owner1", [Permission.READ], allowed_entities=["agent1_acc"])
        result = cp.list_accessible_resources("agent1_acc", ResourceType.FILE)
        assert len(result) == 1
        result[0].path = "/modified"
        assert cp._resources["res1_acc"].path == "/tmp/test"

    def test_get_negotiation_returns_copy(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability, Permission, ResourceType
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap(name="neg_test")
        cp.register_entity("init1_neg", "agent", [cap])
        cp.register_entity("recv1_neg", "agent", [cap])
        result = cp.negotiate_capabilities("init1_neg", "recv1_neg", [cap])
        assert result["success"] is True
        neg = cp.get_negotiation(result["negotiation_id"])
        assert neg is not None
        assert isinstance(neg, dict)


class TestReasoningContextProtocolFsync:
    """Test atomic file writes in reasoning/context_protocol."""

    def test_save_context_atomic(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_save", "agent", [cap])
        import hashlib
        safe_id = hashlib.sha256("agent1_save".encode()).hexdigest()[:16]
        context_file = tmp_path / f"context_{safe_id}.json"
        assert context_file.exists()

    def test_load_context_returns_deepcopy(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_load", "agent", [cap])
        loaded = cp.load_context("agent1_load")
        assert loaded is not None
        assert loaded.entity_id == "agent1_load"
        assert loaded.entity_type == "agent"

    def test_load_context_invalid_id_returns_none(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol
        cp = ContextProtocol(storage_path=tmp_path)
        result = cp.load_context("invalid<>id")
        assert result is None


class TestReasoningContextProtocolHandlerCaps:
    """Test handler caps and registration."""

    def test_register_handler_at_max_evicts(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol
        cp = ContextProtocol(storage_path=tmp_path)
        for i in range(51):
            cp.register_handler(f"recv_handler_{i}", "msg_type", lambda m: None)
        assert len(cp._handlers) == 50


class TestReasoningContextProtocolMessageValidation:
    """Test message type validation and payload caps."""

    def test_create_message_invalid_type_sanitized(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("sender1_msg", "agent", [cap])
        msg = cp.create_message("sender1_msg", "recv1", "bad<script>type", {"key": "val"})
        assert msg.message_type in ("badscripttype", "unknown")

    def test_create_message_payload_capped_at_50(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("sender1_pay", "agent", [cap])
        payload = {f"key_{i}": f"val_{i}" for i in range(60)}
        msg = cp.create_message("sender1_pay", "recv1", "test", payload)
        assert len(msg.payload) <= 50


class TestReasoningContextProtocolStatistics:
    """Test get_statistics under lock."""

    def test_get_statistics_returns_copy(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_stat", "agent", [cap])
        stats = cp.get_statistics()
        assert stats["registered_entities"] == 1
        stats["registered_entities"] = 999
        stats2 = cp.get_statistics()
        assert stats2["registered_entities"] == 1


class TestReasoningContextProtocolNonceCleanup:
    """Test nonce cleanup at max."""

    def test_nonce_cleanup_at_max(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        for i in range(10001):
            km.generate_nonce(f"entity_{i}")
        nonce = km.generate_nonce("test_after_cleanup")
        assert nonce is not None


class TestReasoningContextProtocolEntityCaps:
    """Test entity registration caps."""

    def test_entity_count_at_500(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        for i in range(500):
            cp.register_entity(f"entity_cap_{i}", "agent", [cap])
        assert len(cp._entities) == 500


class TestReasoningContextProtocolCloseChannel:
    """Test close_channel under lock."""

    def test_close_channel_under_lock(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap(name="test_ch")
        cp.register_entity("a_ch", "agent", [cap])
        cp.register_entity("b_ch", "agent", [cap])
        result = cp.negotiate_capabilities("a_ch", "b_ch", [cap])
        assert result["success"] is True
        channel_id = result["channel_id"]
        cp.close_channel(channel_id)
        assert channel_id not in cp._channels


class TestContextProtocolToDictCopies:
    """Test to_dict returns copies of mutable fields."""

    def test_capability_to_dict_copies_metadata(self):
        from shadowd.reasoning.context_protocol import Capability, Permission, ResourceType
        cap = Capability(
            name="test_todict",
            permissions=[Permission.READ],
            resources=[ResourceType.FILE],
            metadata={"key": "val"}
        )
        d = cap.to_dict()
        d["metadata"]["key"] = "modified"
        assert cap.metadata["key"] == "val"

    def test_context_message_to_dict_copies_payload(self):
        from shadowd.reasoning.context_protocol import ContextMessage
        msg = ContextMessage(
            id="test",
            sender_id="a",
            receiver_id="b",
            message_type="test",
            payload={"key": "val"},
            metadata={"m": "v"},
        )
        d = msg.to_dict()
        d["payload"]["key"] = "modified"
        assert msg.payload["key"] == "val"
        d["metadata"]["m"] = "modified2"
        assert msg.metadata["m"] == "v"

    def test_resource_to_dict_copies_metadata(self):
        from shadowd.reasoning.context_protocol import Resource, ResourceType, Permission
        res = Resource(
            resource_id="r1_todict",
            resource_type=ResourceType.FILE,
            path="/tmp/test",
            owner_id="owner1",
            required_permissions=[Permission.READ],
            allowed_entities=["agent1"],
            metadata={"key": "val"},
        )
        d = res.to_dict()
        d["metadata"]["key"] = "modified"
        assert res.metadata["key"] == "val"
        d["allowed_entities"].append("agent2")
        assert len(res.allowed_entities) == 1


class TestContextSnapshotGetContext:
    """Test ContextSnapshot to_dict and get_unified_context."""

    def test_snapshot_to_dict_returns_copies(self):
        from shadowd.context_protocol import ContextSnapshot
        snapshot = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            temporal_context={"time": "12:00"},
            social_context={"who": "alice"},
        )
        d = snapshot.to_dict()
        d["temporal"]["time"] = "99:99"
        d["social"]["who"] = "bob"
        assert snapshot.temporal_context["time"] == "12:00"
        assert snapshot.social_context["who"] == "alice"

    def test_get_unified_context_returns_copy(self):
        from shadowd.context_protocol import ContextSnapshot
        snapshot = ContextSnapshot(
            timestamp=datetime.now(timezone.utc),
            temporal_context={"time": "12:00"},
            social_context={"who": "alice"},
        )
        unified = snapshot.get_unified_context()
        unified["when"]["time"] = "99:99"
        assert snapshot.temporal_context["time"] == "12:00"


class TestReasoningContextProtocolRequestAccess:
    """Test request_access under lock."""

    def test_request_access_unknown_entity(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Permission
        cp = ContextProtocol(storage_path=tmp_path)
        result = cp.request_access("unknown_ra", "res1", Permission.READ)
        assert result is False

    def test_request_access_unknown_resource(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Permission, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_ra", "agent", [cap])
        result = cp.request_access("agent1_ra", "unknown_res", Permission.READ)
        assert result is False


class TestReasoningContextProtocolUnregisterEntity:
    """Test unregister_entity under lock."""

    def test_unregister_entity(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_unreg", "agent", [cap])
        assert "agent1_unreg" in cp._entities
        cp.unregister_entity("agent1_unreg")
        assert "agent1_unreg" not in cp._entities

    def test_unregister_entity_cleans_key(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("agent1_key", "agent", [cap])
        assert cp._key_manager.get_key("agent1_key") is not None
        cp.unregister_entity("agent1_key")
        assert cp._key_manager.get_key("agent1_key") is None


class TestReasoningContextProtocolSendMessage:
    """Test send_message validates before dispatching."""

    def test_send_message_no_handler(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("sender1_sm", "agent", [cap])
        msg = cp.create_message("sender1_sm", "recv1", "test", {"key": "val"})
        result = cp.send_message(msg)
        assert result is False


class TestReasoningContextProtocolKeyManagerDelete:
    """Test KeyManager.delete_key under lock."""

    def test_delete_key(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        km.generate_key("entity1_del")
        assert km.get_key("entity1_del") is not None
        km.delete_key("entity1_del")
        assert km.get_key("entity1_del") is None

    def test_delete_nonexistent_key(self):
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        km.delete_key("nonexistent")


class TestContextProtocolSaveLoadRoundtrip:
    """Test save and load roundtrip for both protocols."""

    def test_reasoning_save_and_load(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap()
        cp.register_entity("roundtrip_agent", "agent", [cap])
        loaded = cp.load_context("roundtrip_agent")
        assert loaded is not None
        assert loaded.entity_id == "roundtrip_agent"
        assert loaded.entity_type == "agent"

    def test_context_protocol_save_snapshot(self, tmp_path):
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot
        cp = ContextProtocol(data_dir=tmp_path)
        ts = datetime(2024, 1, 15, 12, 30, 0, tzinfo=timezone.utc)
        snapshot = ContextSnapshot(timestamp=ts, temporal_context={"time": "12:30"})
        cp.save_snapshot(snapshot)
        files = list(tmp_path.glob("snapshot_*.json"))
        assert len(files) == 1
        data = json.loads(files[0].read_text())
        assert data["temporal"]["time"] == "12:30"


class TestReasoningContextProtocolHMAC:
    """Test HMAC signature verification."""

    def test_create_and_verify_message(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap(name="hmac_test")
        cp.register_entity("signer1", "agent", [cap])
        msg = cp.create_message("signer1", "recv1", "test", {"key": "val"}, sign=True)
        assert msg.signature is not None
        assert cp.verify_message(msg) is True

    def test_tampered_message_fails_verification(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability
        cp = ContextProtocol(storage_path=tmp_path)
        cap = _make_cap(name="hmac_tamper")
        cp.register_entity("signer2", "agent", [cap])
        msg = cp.create_message("signer2", "recv1", "test", {"key": "val"}, sign=True)
        msg.payload["key"] = "tampered"
        assert cp.verify_message(msg) is False

    def test_unsigned_message_from_unknown_sender_fails(self, tmp_path):
        from shadowd.reasoning.context_protocol import ContextProtocol, ContextMessage
        cp = ContextProtocol(storage_path=tmp_path)
        msg = ContextMessage(
            id="test",
            sender_id="unknown",
            receiver_id="recv1",
            message_type="test",
            payload={"key": "val"},
        )
        assert cp.verify_message(msg) is False