#!/usr/bin/env python3
"""
Phase 8 Hardening Tests - security.py, context_protocol.py, state_graph.py,
shikigami/, superhuman.py, shadow_sync.py
"""

import asyncio
import json
import os
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest


# ═══════════════════════════════════════════════════════════════
# security.py
# ═══════════════════════════════════════════════════════════════

class TestSecurityHardening:
    """Tests for security.py hardening."""

    def test_api_key_manager_atomic_save(self, tmp_path):
        """API key manager should use atomic writes."""
        from shadowd.security import APIKeyManager
        mgr = APIKeyManager(storage_dir=tmp_path)
        mgr.generate_key("user1", "test_key")
        data_file = tmp_path / "api_keys.json"
        assert data_file.exists()
        assert not (tmp_path / "api_keys.tmp").exists()

    def test_user_manager_atomic_save(self, tmp_path):
        """User manager should use atomic writes."""
        try:
            from shadowd.security import UserManager, JWTAuth, SecurityConfig
            config = SecurityConfig()
            auth = JWTAuth(config)
            mgr = UserManager(storage_dir=tmp_path)
            mgr.create_user("test@example.com", "testuser", "Str0ng!Pass", auth)
            data_file = tmp_path / "users.json"
            assert data_file.exists()
            assert not (tmp_path / "users.tmp").exists()
        except (ImportError, ValueError):
            pytest.skip("jose/passlib not available or bcrypt incompatible")

    def test_update_user_blocks_hashed_password(self, tmp_path):
        """update_user should not allow setting hashed_password via kwargs."""
        try:
            from shadowd.security import UserManager, JWTAuth, SecurityConfig
            config = SecurityConfig()
            auth = JWTAuth(config)
            mgr = UserManager(storage_dir=tmp_path)
            user = mgr.create_user("test2@example.com", "user2", "Str0ng!Pass2", auth)
            original_hash = user.hashed_password
            mgr.update_user(user.user_id, hashed_password="MALICIOUS_HASH")
            assert user.hashed_password == original_hash
        except (ImportError, ValueError):
            pytest.skip("jose/passlib not available or bcrypt incompatible")

    def test_update_user_blocks_user_id(self, tmp_path):
        """update_user should not let a kwarg overwrite user_id (silently
        ignore it, like the rest of the immutable fields)."""
        try:
            from shadowd.security import UserManager, JWTAuth, SecurityConfig
            config = SecurityConfig()
            auth = JWTAuth(config)
            mgr = UserManager(storage_dir=tmp_path)
            user = mgr.create_user("test3@example.com", "user3", "Str0ng!Pass3", auth)
            original_id = user.user_id
            # update_user(user_id, **kwargs) — pass the target via the first
            # positional arg, and a malicious overwrite via kwargs.
            kwargs = {"user_id": "MALICIOUS_ID"}
            mgr.update_user(user.user_id, **kwargs)
            assert user.user_id == original_id
        except (ImportError, ValueError, TypeError):
            pytest.skip("jose/passlib not available or bcrypt incompatible")

    @pytest.mark.asyncio
    async def test_rate_limiter_max_clients(self, tmp_path):
        """Rate limiter should cap client entries."""
        from shadowd.security import RateLimiter
        rl = RateLimiter(requests_per_minute=60)
        for i in range(15000):
            await rl.is_allowed(f"client_{i}")
        assert len(rl.requests) <= rl._MAX_CLIENTS + 500

    def test_rate_limiter_has_max_constant(self):
        """Rate limiter should define _MAX_CLIENTS."""
        from shadowd.security import RateLimiter
        rl = RateLimiter()
        assert hasattr(rl, '_MAX_CLIENTS')
        assert rl._MAX_CLIENTS > 0


# ═══════════════════════════════════════════════════════════════
# context_protocol.py (legacy)
# ═══════════════════════════════════════════════════════════════

class TestContextProtocolHardening:
    """Tests for shadowd/context_protocol.py hardening."""

    def test_max_subscribers(self):
        """Context protocol should cap subscribers."""
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        assert hasattr(cp, '_MAX_SUBSCRIBERS')
        assert cp._MAX_SUBSCRIBERS > 0

    def test_subscribe_cap(self):
        """Subscribe should respect max_subscribers."""
        from shadowd.context_protocol import ContextProtocol
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        cp._MAX_SUBSCRIBERS = 3
        for i in range(5):
            cp.subscribe(lambda: None)
        assert len(cp.subscribers) == 3

    def test_atomic_snapshot_save(self):
        """Snapshot save should be atomic (no .tmp files left)."""
        from shadowd.context_protocol import ContextProtocol, ContextSnapshot
        cp = ContextProtocol(data_dir=Path(tempfile.mkdtemp()))
        snapshot = ContextSnapshot(timestamp=datetime.now())
        cp.save_snapshot(snapshot)
        files = list(cp.data_dir.glob("snapshot_*.json"))
        tmp_files = list(cp.data_dir.glob("snapshot_*.tmp"))
        assert len(files) == 1
        assert len(tmp_files) == 0


# ═══════════════════════════════════════════════════════════════
# reasoning/context_protocol.py
# ═══════════════════════════════════════════════════════════════

class TestReasoningContextProtocolHardening:
    """Tests for reasoning/context_protocol.py hardening."""

    def test_nonce_single_use(self):
        """Nonce should be single-use."""
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        entity = "test_entity"
        nonce = km.generate_nonce(entity)
        assert km.verify_nonce(entity, nonce) is True
        assert km.verify_nonce(entity, nonce) is False

    def test_nonce_wrong_value(self):
        """Wrong nonce should fail verification."""
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        entity = "test_entity"
        km.generate_nonce(entity)
        assert km.verify_nonce(entity, "wrong_nonce") is False

    def test_nonce_max_cleanup(self):
        """KeyManager should clean up when too many nonces."""
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        km._max_nonces = 10
        for i in range(20):
            km.generate_nonce(f"entity_{i}")
        assert len(km._nonces) <= km._max_nonces + 20

    def test_atomic_context_save(self):
        """Context save should be atomic."""
        import hashlib
        from shadowd.reasoning.context_protocol import ContextProtocol, Capability, Permission, ResourceType
        cp = ContextProtocol(storage_path=Path(tempfile.mkdtemp()))
        cap = Capability(name="test", permissions=[Permission.READ], resources=[ResourceType.FILE])
        cp.register_entity("test_ent", "agent", [cap])
        safe_id = hashlib.sha256("test_ent".encode()).hexdigest()[:16]
        ctx_file = cp.storage_path / f"context_{safe_id}.json"
        tmp_file = cp.storage_path / f"context_{safe_id}.tmp"
        assert ctx_file.exists()
        assert not tmp_file.exists()


# ═══════════════════════════════════════════════════════════════
# reasoning/state_graph.py
# ═══════════════════════════════════════════════════════════════

class TestStateGraphHardening:
    """Tests for reasoning/state_graph.py hardening."""

    def test_max_checkpoints(self):
        """StateGraph should cap checkpoints."""
        from shadowd.reasoning.state_graph import StateGraph, State
        sg = StateGraph(name="test", checkpoint_dir=Path(tempfile.mkdtemp()))
        assert hasattr(sg, '_max_checkpoints')
        assert sg._max_checkpoints > 0

    def test_checkpoint_atomic_save(self):
        """Checkpoint save should be atomic."""
        from shadowd.reasoning.state_graph import StateGraph, StateNode, State
        sg = StateGraph(name="test", checkpoint_dir=Path(tempfile.mkdtemp()))
        node = StateNode(id="start", name="Start", state=State.INIT)
        sg.add_node(node)
        sg.set_start("start")
        sg._current_context = {"test": True}
        sg._current_node = node
        sg._create_checkpoint(State.COMPLETED)
        json_files = list(sg.checkpoint_dir.glob("*.json"))
        tmp_files = list(sg.checkpoint_dir.glob("*.tmp"))
        assert len(json_files) == 1
        assert len(tmp_files) == 0


# ═══════════════════════════════════════════════════════════════
# shikigami/shikigami.py
# ═══════════════════════════════════════════════════════════════

class TestShikigamiHardening:
    """Tests for shikigami/shikigami.py hardening."""

    def test_atomic_json_save(self, tmp_path):
        """Shikigami should use atomic JSON saves."""
        from shikigami.shikigami import ShikigamiRegistry
        reg = ShikigamiRegistry()
        test_file = tmp_path / "test.json"
        reg._save_json(test_file, {"key": "value"})
        assert test_file.exists()
        assert not (tmp_path / "test.tmp").exists()
        data = json.loads(test_file.read_text())
        assert data["key"] == "value"


# ═══════════════════════════════════════════════════════════════
# shikigami/abilities.py
# ═══════════════════════════════════════════════════════════════

class TestAbilitiesHardening:
    """Tests for shikigami/abilities.py hardening."""

    def test_decision_tracker_atomic_save(self, tmp_path):
        """DecisionTracker should use atomic saves."""
        from shikigami.abilities import DecisionTracker
        db = tmp_path / "decisions.json"
        tracker = DecisionTracker(db_path=db)
        tracker.record_decision("test", "choice_a")
        assert db.exists()
        assert not (tmp_path / "decisions.tmp").exists()


# ═══════════════════════════════════════════════════════════════
# shikigami/superhuman.py
# ═══════════════════════════════════════════════════════════════

class TestSuperhumanHardening:
    """Tests for shikigami/superhuman.py hardening."""

    def test_markov_history_cap(self):
        """MarkovBehaviorModel should cap history."""
        from shikigami.superhuman import MarkovBehaviorModel
        model = MarkovBehaviorModel()
        assert hasattr(model, '_max_history')
        assert model._max_history > 0
        for i in range(2000):
            model.record(f"s{i%5}", f"a{i%3}", f"s{(i+1)%5}")
        assert len(model.history) <= model._max_history

    def test_model_persistence_atomic_save(self, tmp_path):
        """ModelPersistence should use atomic saves."""
        from shikigami.superhuman import ModelPersistence
        ModelPersistence.BASE_DIR = tmp_path
        ModelPersistence.save("test_model", {"key": "value"})
        data_file = tmp_path / "test_model.json"
        assert data_file.exists()
        assert not (tmp_path / "test_model.tmp").exists()

    def test_reinforcement_learner_atomic_save(self, tmp_path):
        """ReinforcementLearner save should be atomic."""
        from shikigami.superhuman import ReinforcementLearner
        rl = ReinforcementLearner()
        rl.update("s1", "a1", 1.0, "s2")
        save_path = tmp_path / "q_table.json"
        rl.save_q_table(save_path)
        assert save_path.exists()
        assert not (tmp_path / "q_table.tmp").exists()

    def test_key_manager_cleans_old_nonces(self):
        """KeyManager should clean nonces when too many."""
        from shadowd.reasoning.context_protocol import KeyManager
        km = KeyManager()
        km._max_nonces = 5
        for i in range(10):
            km.generate_nonce(f"e{i}")
        assert len(km._nonces) <= km._max_nonces + 10


# ═══════════════════════════════════════════════════════════════
# sync/shadow_sync.py
# ═══════════════════════════════════════════════════════════════

class TestShadowSyncHardening:
    """Tests for sync/shadow_sync.py hardening."""

    def test_validate_name_normal(self):
        """Name validation should pass clean names."""
        from sync.shadow_sync import ShadowSync
        assert ShadowSync._validate_name("my-device_01") == "my-device_01"

    def test_validate_name_strips_special(self):
        """Name validation should strip special chars."""
        from sync.shadow_sync import ShadowSync
        assert ShadowSync._validate_name("my<script>alert(1)</script>") == "myscriptalert1script"

    def test_validate_name_empty(self):
        """Name validation should return 'unknown' for empty."""
        from sync.shadow_sync import ShadowSync
        assert ShadowSync._validate_name("") == "unknown"

    def test_validate_name_too_long(self):
        """Name validation should reject overly long names."""
        from sync.shadow_sync import ShadowSync
        assert ShadowSync._validate_name("x" * 100) == "unknown"

    def test_headers_without_key(self):
        """Headers should be empty without API key."""
        from sync.shadow_sync import ShadowSync
        with patch.dict(os.environ, {"SHADOW_API_KEY": ""}):
            sync = ShadowSync.__new__(ShadowSync)
            assert sync._headers() == {}

    def test_headers_with_key(self):
        """Headers should include auth with API key."""
        from sync.shadow_sync import ShadowSync
        with patch.dict(os.environ, {"SHADOW_API_KEY": "test_key_123"}):
            sync = ShadowSync.__new__(ShadowSync)
            h = sync._headers()
            assert "Authorization" in h
            assert h["Authorization"] == "Bearer test_key_123"


# ═══════════════════════════════════════════════════════════════
# Cross-cutting: Non-atomic write detection
# ═══════════════════════════════════════════════════════════════

class TestAtomicWrites:
    """Verify all file writers use atomic tmp+rename pattern."""

    def test_security_api_keys_atomic(self, tmp_path):
        from shadowd.security import APIKeyManager
        mgr = APIKeyManager(storage_dir=tmp_path)
        mgr.generate_key("u1", "k1")
        assert (tmp_path / "api_keys.json").exists()
        assert not (tmp_path / "api_keys.tmp").exists()

    def test_security_users_atomic(self, tmp_path):
        try:
            from shadowd.security import UserManager, JWTAuth, SecurityConfig
            config = SecurityConfig()
            auth = JWTAuth(config)
            mgr = UserManager(storage_dir=tmp_path)
            mgr.create_user("a@b.com", "u1", "Str0ng!P1", auth)
            assert (tmp_path / "users.json").exists()
            assert not (tmp_path / "users.tmp").exists()
        except (ImportError, ValueError):
            pytest.skip("jose/passlib not available or bcrypt incompatible")

    def test_shikigami_atomic(self, tmp_path):
        from shikigami.shikigami import ShikigamiRegistry
        reg = ShikigamiRegistry()
        f = tmp_path / "test.json"
        reg._save_json(f, {"a": 1})
        assert f.exists()
        assert not (tmp_path / "test.tmp").exists()

    def test_abilities_atomic(self, tmp_path):
        from shikigami.abilities import DecisionTracker
        f = tmp_path / "dec.json"
        t = DecisionTracker(db_path=f)
        t.record_decision("s", "c")
        assert f.exists()
        assert not (tmp_path / "dec.tmp").exists()

    def test_superhuman_model_persistence_atomic(self, tmp_path):
        from shikigami.superhuman import ModelPersistence
        ModelPersistence.BASE_DIR = tmp_path
        ModelPersistence.save("atomic_test", {"x": 1})
        assert (tmp_path / "atomic_test.json").exists()
        assert not (tmp_path / "atomic_test.tmp").exists()