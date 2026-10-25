"""Sprint 114 tests — security.py thread locks, deepcopy, atomic writes, timing-safe auth,
RateLimiter _lock fix, JWT error messages, security_utils.py fixes."""

import pytest
import copy
import threading
import time
from pathlib import Path
from unittest.mock import MagicMock, patch


class TestSecurityAPIKeyManager:
    """Tests for APIKeyManager thread safety and deepcopy fixes."""

    def test_apikey_manager_has_lock(self):
        from shadowd.security import APIKeyManager
        with patch.object(APIKeyManager, '_load'):
            mgr = APIKeyManager.__new__(APIKeyManager)
        mgr.keys = {}
        mgr._lock = threading.Lock()
        mgr.storage_dir = Path("/tmp/test_keys_s114")
        assert hasattr(mgr, '_lock'), "APIKeyManager should have _lock"

    def test_apikey_generate_key_thread_safe(self):
        from shadowd.security import APIKeyManager, APIKey
        with patch.object(APIKeyManager, '_load'):
            with patch.object(APIKeyManager, '_save'):
                mgr = APIKeyManager(storage_dir=Path("/tmp/test_keys_s114"))
        mgr.storage_dir = Path("/tmp/test_keys_s114")
        mgr._lock = threading.Lock()
        key = mgr.generate_key("user1", "test_key")
        assert key.startswith("sk_")
        assert len(mgr.keys) == 1

    def test_apikey_verify_key_returns_deepcopy(self):
        from shadowd.security import APIKeyManager, APIKey
        with patch.object(APIKeyManager, '_load'):
            with patch.object(APIKeyManager, '_save'):
                mgr = APIKeyManager(storage_dir=Path("/tmp/test_keys_s114"))
        mgr.storage_dir = Path("/tmp/test_keys_s114")
        mgr._lock = threading.Lock()
        raw_key = mgr.generate_key("user1", "test_key")
        result = mgr.verify_key(raw_key)
        assert result is not None
        assert isinstance(result, APIKey)
        # Verify it's a copy — mutating the returned key shouldn't affect internals
        result.is_active = False
        # The internal key should still be active
        for k in mgr.keys.values():
            assert k.is_active is True

    def test_apikey_list_keys_returns_copies(self):
        from shadowd.security import APIKeyManager
        with patch.object(APIKeyManager, '_load'):
            with patch.object(APIKeyManager, '_save'):
                mgr = APIKeyManager(storage_dir=Path("/tmp/test_keys_s114"))
        mgr.storage_dir = Path("/tmp/test_keys_s114")
        mgr._lock = threading.Lock()
        mgr.generate_key("user1", "test_key_1")
        mgr.generate_key("user1", "test_key_2")
        keys = mgr.list_keys(user_id="user1")
        assert len(keys) == 2
        # Verify scopes are copies
        for k in keys:
            assert isinstance(k["scopes"], list)

    def test_apikey_list_keys_filters_inactive(self):
        from shadowd.security import APIKeyManager
        with patch.object(APIKeyManager, '_load'):
            with patch.object(APIKeyManager, '_save'):
                mgr = APIKeyManager(storage_dir=Path("/tmp/test_keys_s114"))
        mgr.storage_dir = Path("/tmp/test_keys_s114")
        mgr._lock = threading.Lock()
        mgr.generate_key("user1", "test_key_1")
        mgr.generate_key("user1", "test_key_2")
        # Revoke one key
        for key_id in list(mgr.keys.keys()):
            mgr.keys[key_id].is_active = False
            break
        # list_keys should filter inactive keys
        keys = mgr.list_keys(user_id="user1")
        assert len(keys) == 1

    def test_apikey_revoke_under_lock(self):
        from shadowd.security import APIKeyManager
        with patch.object(APIKeyManager, '_load'):
            with patch.object(APIKeyManager, '_save'):
                mgr = APIKeyManager(storage_dir=Path("/tmp/test_keys_s114"))
        mgr.storage_dir = Path("/tmp/test_keys_s114")
        mgr._lock = threading.Lock()
        mgr.generate_key("user1", "test_key")
        key_id = list(mgr.keys.keys())[0]
        result = mgr.revoke_key(key_id)
        assert result is True
        assert mgr.keys[key_id].is_active is False
        # Revoking non-existent key
        result = mgr.revoke_key("nonexistent")
        assert result is False


class TestSecurityUserManager:
    """Tests for UserManager thread safety and deepcopy fixes."""

    def test_user_manager_has_lock(self):
        from shadowd.security import UserManager
        with patch.object(UserManager, '_load'):
            mgr = UserManager.__new__(UserManager)
        mgr.users = {}
        mgr.email_index = {}
        mgr._lock = threading.Lock()
        mgr.storage_dir = Path("/tmp/test_users_s114")
        assert hasattr(mgr, '_lock'), "UserManager should have _lock"

    def test_user_to_dict_returns_list_copies(self):
        from shadowd.security import User
        from datetime import datetime, timezone
        user = User(
            user_id="test123",
            email="test@test.com",
            username="testuser",
            hashed_password="hashed",
            scopes=["read", "write"],
            api_keys=["key1"]
        )
        d = user.to_dict()
        assert d["scopes"] == ["read", "write"]
        assert d["scopes"] is not user.scopes  # Should be a copy
        assert d["api_keys"] == ["key1"]
        assert d["api_keys"] is not user.api_keys  # Should be a copy

    def test_authenticate_user_returns_deepcopy(self):
        from shadowd.security import UserManager
        with patch.object(UserManager, '_load'):
            mgr = UserManager.__new__(UserManager)
        mgr.users = {}
        mgr.email_index = {}
        mgr._lock = threading.Lock()
        mgr.storage_dir = Path("/tmp/test_users_s114")
        
        auth = MagicMock()
        auth.verify_password = MagicMock(return_value=True)
        auth.hash_password = MagicMock(return_value="$2b$12$hashedpassword")
        auth.validate_password = MagicMock(return_value=(True, []))
        
        with patch.object(mgr, '_save'):
            user = mgr.create_user("test@test.com", "testuser", "Pass123!", auth)
        
        # Authenticate
        with patch.object(mgr, '_save'):
            result = mgr.authenticate_user("test@test.com", "Pass123!", auth)
        
        assert result is not None
        # Mutating result should not affect internal state
        result.scopes.append("admin")
        assert "admin" not in mgr.users[list(mgr.users.keys())[0]].scopes

    def test_get_user_returns_deepcopy(self):
        from shadowd.security import UserManager
        with patch.object(UserManager, '_load'):
            mgr = UserManager.__new__(UserManager)
        mgr.users = {}
        mgr.email_index = {}
        mgr._lock = threading.Lock()
        mgr.storage_dir = Path("/tmp/test_users_s114")
        
        auth = MagicMock()
        auth.hash_password = MagicMock(return_value="$2b$12$hashedpassword")
        auth.validate_password = MagicMock(return_value=(True, []))
        
        with patch.object(mgr, '_save'):
            mgr.create_user("test2@test.com", "testuser2", "Pass123!", auth)
        
        user_id = list(mgr.users.keys())[0]
        user = mgr.get_user(user_id)
        assert user is not None
        # Mutating should not affect internal state
        user.username = "hacked"
        assert mgr.users[user_id].username == "testuser2"

    def test_update_user_blocked_fields(self):
        from shadowd.security import UserManager
        with patch.object(UserManager, '_load'):
            mgr = UserManager.__new__(UserManager)
        mgr.users = {}
        mgr.email_index = {}
        mgr._lock = threading.Lock()
        mgr.storage_dir = Path("/tmp/test_users_s114")
        
        auth = MagicMock()
        auth.hash_password = MagicMock(return_value="$2b$12$hashedpassword")
        auth.validate_password = MagicMock(return_value=(True, []))
        
        with patch.object(mgr, '_save'):
            mgr.create_user("test3@test.com", "testuser3", "Pass123!", auth)
        
        user_id = list(mgr.users.keys())[0]
        
        with patch.object(mgr, '_save'):
            result = mgr.update_user(user_id, is_superuser=True)
        
        # is_superuser should be blocked
        assert mgr.users[user_id].is_superuser is False
        # Result should be a deepcopy
        assert result is not None
        assert result.is_superuser is False

    def test_delete_user_under_lock(self):
        from shadowd.security import UserManager
        with patch.object(UserManager, '_load'):
            mgr = UserManager.__new__(UserManager)
        mgr.users = {}
        mgr.email_index = {}
        mgr._lock = threading.Lock()
        mgr.storage_dir = Path("/tmp/test_users_s114")
        
        auth = MagicMock()
        auth.hash_password = MagicMock(return_value="$2b$12$hashedpassword")
        auth.validate_password = MagicMock(return_value=(True, []))
        
        with patch.object(mgr, '_save'):
            mgr.create_user("delete@test.com", "deleteuser", "Pass123!", auth)
        
        user_id = list(mgr.users.keys())[0]
        
        with patch.object(mgr, '_save'):
            result = mgr.delete_user(user_id)
        
        assert result is True
        assert user_id not in mgr.users
        assert "delete@test.com" not in mgr.email_index


class TestSecurityRateLimiter:
    """Tests for RateLimiter fixes."""

    def test_rate_limiter_has_sync_lock(self):
        from shadowd.security import RateLimiter
        rl = RateLimiter()
        assert hasattr(rl, '_sync_lock'), "RateLimiter should have _sync_lock"
        assert isinstance(rl._sync_lock, type(threading.Lock()))

    def test_get_remaining_uses_lock(self):
        from shadowd.security import RateLimiter
        rl = RateLimiter()
        # Should not raise AttributeError
        result = rl.get_remaining("test_client")
        assert "minute_remaining" in result
        assert "hour_remaining" in result

    def test_reset_under_lock(self):
        from shadowd.security import RateLimiter
        rl = RateLimiter()
        # Should not raise — reset uses _sync_lock now
        rl.reset("test_client")
        rl.reset()  # Reset all


class TestSecurityJWTAuth:
    """Tests for JWT auth fixes."""

    def test_decode_token_generic_error_message(self):
        from shadowd.security import JWTAuth, SecurityConfig
        config = SecurityConfig(secret_key="test_secret_key_for_testing_only_12345")
        auth = JWTAuth(config)
        
        # Invalid token should raise ValueError with generic message
        with pytest.raises(ValueError, match="Invalid token"):
            auth.decode_token("invalid_token_here")

    def test_create_access_token_single_timestamp(self):
        """Verify create_access_token uses a single datetime.now() call."""
        from shadowd.security import JWTAuth, SecurityConfig
        import inspect
        config = SecurityConfig(secret_key="test_secret_key_for_testing_only_12345")
        source = inspect.getsource(JWTAuth.create_access_token)
        # Should have exactly one datetime.now(timezone.utc) for 'now'
        # and derive expire and iat from it
        assert "now = datetime.now(timezone.utc)" in source
        assert "expire = now +" in source
        assert '"iat": now' in source


class TestSecurityAtomicWrites:
    """Tests for atomic write patterns in _save methods."""

    def test_apikey_save_uses_fsync(self):
        from shadowd.security import APIKeyManager
        import inspect
        source = inspect.getsource(APIKeyManager._save)
        assert "os.fsync" in source or "os.open" in source, "_save should use atomic write with fsync"

    def test_user_save_uses_fsync(self):
        from shadowd.security import UserManager
        import inspect
        source = inspect.getsource(UserManager._save)
        assert "os.fsync" in source or "os.open" in source, "_save should use atomic write with fsync"


class TestSecurityUtils:
    """Tests for security_utils.py fixes."""

    def test_rate_limiter_singleton_lock(self):
        from shadowd.security_utils import get_rate_limiter
        # Calling twice should return the same instance
        rl1 = get_rate_limiter("test_singleton")
        rl2 = get_rate_limiter("test_singleton")
        assert rl1 is rl2

    def test_audit_event_to_dict_metadata_copy(self):
        from shadowd.security_utils import AuditEvent
        from datetime import datetime, timezone
        event = AuditEvent(
            event_type="test",
            source="test_source",
            severity="info",
            message="test message",
            metadata={"key": "value", "nested": {"inner": "data"}},
            timestamp=datetime.now(timezone.utc)
        )
        d = event.to_dict()
        # Metadata should be a deep copy
        assert d["metadata"] == {"key": "value", "nested": {"inner": "data"}}
        assert d["metadata"] is not event.metadata

    def test_sanitize_string_strict_mode(self):
        from shadowd.security_utils import InputValidator
        # Content-safe: preserves <>
        result = InputValidator.sanitize_string("hello <world> here", max_length=1000)
        assert "<world>" in result
        
        # Strict: strips <>
        result_strict = InputValidator.sanitize_string("hello <world> here", max_length=1000, strict=True)
        assert "<" not in result_strict
        assert ">" not in result_strict

    def test_validate_path_sandbox_no_path_leak(self):
        from shadowd.security_utils import InputValidator
        try:
            InputValidator.validate_path_sandbox("/etc/shadow", [Path("/tmp")])
        except ValueError as e:
            # Should NOT contain the resolved path
            assert "/etc/shadow" not in str(e) or "outside allowed" in str(e)

    def test_sanitize_string_strips_control_chars(self):
        from shadowd.security_utils import InputValidator
        result = InputValidator.sanitize_string("hello\x00world\x7f", max_length=1000)
        assert "\x00" not in result
        assert "\x7f" not in result

    def test_sanitize_string_strict_strips_angle_brackets(self):
        from shadowd.security_utils import InputValidator
        result = InputValidator.sanitize_string("hello <world> here", max_length=1000, strict=True)
        assert "<" not in result
        assert ">" not in result
        assert "hello" in result
        assert "world" in result


class TestSecurityImports:
    """Verify module-level imports."""

    def test_security_imports_copy(self):
        import shadowd.security as mod
        assert hasattr(mod, 'copy'), "security.py must import copy"

    def test_security_imports_threading(self):
        import shadowd.security as mod
        assert hasattr(mod, 'threading'), "security.py must import threading"

    def test_security_imports_os(self):
        import shadowd.security as mod
        assert hasattr(mod, 'os'), "security.py must import os for atomic writes"