#!/usr/bin/env python3
"""
Security and Authentication Module for Shadow AI

Implements:
- JWT token authentication
- API key management
- Password hashing
- Rate limiting
- Input validation
- Security middleware
"""

from datetime import datetime, timedelta, timezone

def _safe_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from collections import defaultdict
import copy
import hashlib
import hmac
import os
import threading
import secrets
import asyncio
import re
import logging
from pathlib import Path
import json

try:
    from jose import jwt, JWTError
    from passlib.context import CryptContext
except ImportError:
    jwt = None
    JWTError = Exception
    CryptContext = None

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class SecurityConfig(BaseModel):
    """Configuration for security settings."""
    secret_key: str = Field(default_factory=lambda: os.environ.get("SHADOW_SECRET_KEY", secrets.token_urlsafe(32)))
    algorithm: str = "HS256"
    access_token_expire_minutes: int = 30
    refresh_token_expire_days: int = 7
    api_key_prefix: str = "sk"
    rate_limit_per_minute: int = 60
    rate_limit_per_hour: int = 1000
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_digit: bool = True
    password_require_special: bool = True
    
    @classmethod
    def load(cls, path: Path) -> 'SecurityConfig':
        if path.exists():
            data = json.loads(path.read_text())
            return cls(**data)
        return cls()


@dataclass
class User:
    """User model."""
    user_id: str
    email: str
    username: str
    hashed_password: str
    scopes: List[str] = field(default_factory=list)
    is_active: bool = True
    is_superuser: bool = False
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_login: Optional[datetime] = None
    api_keys: List[str] = field(default_factory=list)
    
    def to_dict(self) -> Dict:
        return {
            "user_id": self.user_id,
            "email": self.email,
            "username": self.username,
            "hashed_password": self.hashed_password,
            "scopes": list(self.scopes),
            "is_active": self.is_active,
            "is_superuser": self.is_superuser,
            "created_at": self.created_at.isoformat(),
            "last_login": self.last_login.isoformat() if self.last_login else None,
            "api_keys": list(self.api_keys)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'User':
        return cls(
            user_id=data["user_id"],
            email=data["email"],
            username=data["username"],
            hashed_password=data["hashed_password"],
            scopes=data.get("scopes", []),
            is_active=data.get("is_active", True),
            is_superuser=data.get("is_superuser", False),
            created_at=_safe_ts(data, "created_at", datetime.now(timezone.utc)),
            last_login=_safe_ts(data, "last_login"),
            api_keys=data.get("api_keys", [])
        )


@dataclass
class APIKey:
    """API key model."""
    key_id: str
    key_hash: str
    user_id: str
    name: str
    scopes: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: Optional[datetime] = None
    last_used: Optional[datetime] = None
    is_active: bool = True
    
    def to_dict(self) -> Dict:
        return {
            "key_id": self.key_id,
            "key_hash": self.key_hash,
            "user_id": self.user_id,
            "name": self.name,
            "scopes": list(self.scopes),
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "last_used": self.last_used.isoformat() if self.last_used else None,
            "is_active": self.is_active
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'APIKey':
        return cls(
            key_id=data["key_id"],
            key_hash=data["key_hash"],
            user_id=data["user_id"],
            name=data["name"],
            scopes=data.get("scopes", []),
            created_at=_safe_ts(data, "created_at", datetime.now(timezone.utc)),
            expires_at=_safe_ts(data, "expires_at"),
            last_used=_safe_ts(data, "last_used"),
            is_active=data.get("is_active", True)
        )


class JWTAuth:
    """JWT authentication manager."""
    
    def __init__(self, config: SecurityConfig):
        if jwt is None:
            raise ImportError("jose package required: pip install python-jose[cryptography]")
        if CryptContext is None:
            raise ImportError("passlib package required: pip install passlib[bcrypt]")
        
        self.config = config
        self.secret_key = config.secret_key
        self.algorithm = config.algorithm
        self.pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
    
    def create_access_token(self, user_id: str, scopes: List[str] = None,
                           expires_delta: timedelta = None) -> str:
        """Create JWT access token."""
        now = datetime.now(timezone.utc)
        if expires_delta:
            expire = now + expires_delta
        else:
            expire = now + timedelta(minutes=self.config.access_token_expire_minutes)
        
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "access",
            "scopes": scopes or [],
            "iat": now
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def create_refresh_token(self, user_id: str) -> str:
        """Create JWT refresh token."""
        expire = datetime.now(timezone.utc) + timedelta(days=self.config.refresh_token_expire_days)
        payload = {
            "sub": user_id,
            "exp": expire,
            "type": "refresh",
            "iat": datetime.now(timezone.utc)
        }
        return jwt.encode(payload, self.secret_key, algorithm=self.algorithm)
    
    def decode_token(self, token: str) -> Dict[str, Any]:
        """Decode and validate JWT token."""
        try:
            payload = jwt.decode(token, self.secret_key, algorithms=[self.algorithm])
            return payload
        except JWTError:
            logger.error("JWT token validation failed")
            raise ValueError("Invalid token")
    
    def verify_password(self, plain_password: str, hashed_password: str) -> bool:
        """Verify password against hash."""
        return self.pwd_context.verify(plain_password, hashed_password)
    
    def hash_password(self, password: str) -> str:
        """Hash password using bcrypt."""
        return self.pwd_context.hash(password)
    
    def validate_password(self, password: str) -> tuple:
        """Validate password meets requirements."""
        errors = []
        
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters")
        
        if self.config.password_require_uppercase and not re.search(r"[A-Z]", password):
            errors.append("Password must contain an uppercase letter")
        
        if self.config.password_require_lowercase and not re.search(r"[a-z]", password):
            errors.append("Password must contain a lowercase letter")
        
        if self.config.password_require_digit and not re.search(r"\d", password):
            errors.append("Password must contain a digit")
        
        if self.config.password_require_special and not re.search(r"[!@#$%^&*(),.?\":{}|<>]", password):
            errors.append("Password must contain a special character")
        
        return len(errors) == 0, errors


class APIKeyManager:
    """API key management."""
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or (Path.home() / "digital-shadow" / "data" / "api_keys")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.keys: Dict[str, APIKey] = {}
        self._lock = threading.Lock()
        self._load()
    
    def generate_key(self, user_id: str, name: str, scopes: List[str] = None,
                    expires_days: int = 365) -> str:
        """Generate new API key."""
        prefix = "sk"
        raw_key = f"{prefix}_{secrets.token_urlsafe(32)}"
        key_id = hashlib.sha256(raw_key.encode()).hexdigest()[:16]
        key_hash = hashlib.sha512(raw_key.encode()).hexdigest()
        
        api_key = APIKey(
            key_id=key_id,
            key_hash=key_hash,
            user_id=user_id,
            name=name,
            scopes=scopes or ["read", "write"],
            expires_at=datetime.now(timezone.utc) + timedelta(days=expires_days)
        )
        
        with self._lock:
            self.keys[key_id] = api_key
            self._save()
        
        return raw_key
    
    def verify_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify API key."""
        key_hash = hashlib.sha512(raw_key.encode()).hexdigest()
        
        with self._lock:
            for key in self.keys.values():
                if hmac.compare_digest(key.key_hash, key_hash) and key.is_active:
                    if key.expires_at and datetime.now(timezone.utc) > key.expires_at:
                        return None
                    
                    key.last_used = datetime.now(timezone.utc)
                    self._dirty = True
                    if not hasattr(self, '_last_save_time') or (datetime.now(timezone.utc) - self._last_save_time).total_seconds() > 60:
                        self._save()
                        self._dirty = False
                        self._last_save_time = datetime.now(timezone.utc)
                    return copy.deepcopy(key)
        
        return None
    
    def revoke_key(self, key_id: str) -> bool:
        """Revoke API key."""
        with self._lock:
            if key_id in self.keys:
                self.keys[key_id].is_active = False
                self._save()
                return True
            return False
    
    def list_keys(self, user_id: str = None) -> List[Dict]:
        """List API keys."""
        with self._lock:
            keys = []
            for key in self.keys.values():
                if (user_id is None or key.user_id == user_id) and key.is_active:
                    keys.append({
                        "key_id": key.key_id,
                        "name": key.name,
                        "scopes": list(key.scopes),
                        "created_at": key.created_at.isoformat(),
                        "expires_at": key.expires_at.isoformat() if key.expires_at else None,
                        "last_used": key.last_used.isoformat() if key.last_used else None,
                        "is_active": key.is_active
                    })
            return keys
    
    def _save(self):
        """Save to disk (atomic with fsync)."""
        data_file = self.storage_dir / "api_keys.json"
        data = {kid: key.to_dict() for kid, key in self.keys.items()}
        tmp = data_file.with_suffix(f".tmp.{secrets.token_hex(4)}")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps(data, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(data_file))
        except (OSError, ValueError, TypeError):
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
            raise
    
    def _load(self):
        """Load from disk."""
        data_file = self.storage_dir / "api_keys.json"
        if data_file.exists():
            try:
                data = json.loads(data_file.read_text())
                self.keys = {kid: APIKey.from_dict(kdata) for kid, kdata in data.items()}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.error("Failed to load API keys: %s", type(e).__name__)
                self.keys = {}


class RateLimiter:
    """Rate limiting with sliding window."""
    _MAX_CLIENTS = 10000

    def __init__(self, requests_per_minute: int = 60, requests_per_hour: int = 1000):
        self.rate_per_minute = requests_per_minute
        self.rate_per_hour = requests_per_hour
        self.requests: Dict[str, List[datetime]] = defaultdict(list)
        self.lock = asyncio.Lock()
        self._sync_lock = threading.Lock()
    
    async def is_allowed(self, client_id: str) -> bool:
        """Check if client is within rate limits."""
        async with self.lock:
            now = datetime.now(timezone.utc)
            
            if len(self.requests) > self._MAX_CLIENTS:
                oldest_clients = sorted(
                    self.requests.keys(),
                    key=lambda c: self.requests[c][0] if self.requests[c] else now
                )
                for c in oldest_clients[:len(self.requests) - self._MAX_CLIENTS // 2]:
                    del self.requests[c]
            
            # Clean old requests
            self.requests[client_id] = [
                t for t in self.requests[client_id]
                if now - t < timedelta(hours=1)
            ]
            
            minute_requests = len([
                t for t in self.requests[client_id]
                if now - t < timedelta(minutes=1)
            ])
            
            hour_requests = len(self.requests[client_id])
            
            if minute_requests >= self.rate_per_minute:
                return False
            
            if hour_requests >= self.rate_per_hour:
                return False
            
            self.requests[client_id].append(now)
            return True
    
    def get_remaining(self, client_id: str) -> dict:
        """Get remaining requests."""
        now = datetime.now(timezone.utc)
        with self._sync_lock:
            client_requests = list(self.requests.get(client_id, []))
            client_requests = self.requests.get(client_id, [])
            minute_requests = len([
                t for t in client_requests
                if now - t < timedelta(minutes=1)
            ])
            hour_requests = len([
                t for t in client_requests
                if now - t < timedelta(hours=1)
            ])
            
            return {
                "minute_remaining": max(0, self.rate_per_minute - minute_requests),
                "hour_remaining": max(0, self.rate_per_hour - hour_requests),
                "minute_limit": self.rate_per_minute,
                "hour_limit": self.rate_per_hour
            }
    
    def reset(self, client_id: str = None):
        """Reset rate limits."""
        with self._sync_lock:
            if client_id:
                self.requests[client_id] = []
            else:
                self.requests.clear()


class UserManager:
    """User management with storage."""
    
    _BLOCKED_FIELDS = frozenset({"hashed_password", "user_id", "is_superuser", "is_active", "scopes"})
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or (Path.home() / "digital-shadow" / "data" / "users")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.users: Dict[str, User] = {}
        self.email_index: Dict[str, str] = {}
        self._lock = threading.Lock()
        self._load()
    
    def create_user(self, email: str, username: str, password: str,
                   auth: JWTAuth, scopes: List[str] = None,
                   is_superuser: bool = False) -> User:
        """Create new user."""
        with self._lock:
            if email in self.email_index:
                raise ValueError("Email already registered")
            
            for user in self.users.values():
                if user.username == username:
                    raise ValueError("Username already taken")
            
            valid, errors = auth.validate_password(password)
            if not valid:
                raise ValueError(f"Password validation failed: {', '.join(errors)}")
            
            user_id = secrets.token_urlsafe(16)
            hashed_password = auth.hash_password(password)
            
            user = User(
                user_id=user_id,
                email=email,
                username=username,
                hashed_password=hashed_password,
                scopes=scopes or ["read", "write"],
                is_superuser=is_superuser
            )
            
            self.users[user_id] = user
            self.email_index[email] = user_id
            self._save()
            
            return copy.deepcopy(user)
    
    def authenticate_user(self, email: str, password: str, auth: JWTAuth) -> Optional[User]:
        """Authenticate user (timing-safe)."""
        with self._lock:
            user_id = self.email_index.get(email)
            if user_id is None:
                auth.verify_password(password, "$2b$12$dummyhashtopreventtimingattackxxxxxxxxxxxxxxxxxxxxxxxxx")
                return None
            
            user = self.users.get(user_id)
            if user is None:
                auth.verify_password(password, "$2b$12$dummyhashtopreventtimingattackxxxxxxxxxxxxxxxxxxxxxxxxx")
                return None
            
            if not user.is_active:
                return None
            
            if not auth.verify_password(password, user.hashed_password):
                return None
            
            user.last_login = datetime.now(timezone.utc)
            self._save()
            
            return copy.deepcopy(user)
    
    def get_user(self, user_id: str) -> Optional[User]:
        """Get user by ID (returns copy)."""
        with self._lock:
            user = self.users.get(user_id)
            return copy.deepcopy(user) if user else None
    
    def get_user_by_email(self, email: str) -> Optional[User]:
        """Get user by email (returns copy)."""
        with self._lock:
            if email in self.email_index:
                user = self.users.get(self.email_index[email])
                return copy.deepcopy(user) if user else None
            return None
    
    def update_user(self, user_id: str, **kwargs) -> Optional[User]:
        """Update user."""
        with self._lock:
            if user_id not in self.users:
                return None
            
            user = self.users[user_id]
            
            for key, value in kwargs.items():
                if key in self._BLOCKED_FIELDS:
                    continue
                if hasattr(user, key):
                    setattr(user, key, value)
            
            self._save()
            return copy.deepcopy(user)
    
    def delete_user(self, user_id: str) -> bool:
        """Delete user."""
        with self._lock:
            if user_id not in self.users:
                return False
            
            user = self.users[user_id]
            self.email_index.pop(user.email, None)
            del self.users[user_id]
            self._save()
            
            return True
    
    def _save(self):
        """Save to disk (atomic with fsync)."""
        data_file = self.storage_dir / "users.json"
        data = {uid: user.to_dict() for uid, user in self.users.items()}
        tmp = data_file.with_suffix(f".tmp.{secrets.token_hex(4)}")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps(data, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(data_file))
        except (OSError, ValueError, TypeError):
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
            raise
    
    def _load(self):
        """Load from disk."""
        data_file = self.storage_dir / "users.json"
        if data_file.exists():
            try:
                data = json.loads(data_file.read_text())
                self.users = {uid: User.from_dict(udata) for uid, udata in data.items()}
                self.email_index = {user.email: uid for uid, user in self.users.items()}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.error("Failed to load users: %s", type(e).__name__)
                self.users = {}
                self.email_index = {}


class SecurityManager:
    """Main security manager combining all components."""
    
    def __init__(self, config: SecurityConfig = None, storage_dir: Path = None):
        self.config = config or SecurityConfig()
        self.storage_dir = storage_dir or (Path.home() / "digital-shadow" / "data" / "security")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.auth = JWTAuth(self.config)
        self.api_keys = APIKeyManager(self.storage_dir / "api_keys")
        self.rate_limiter = RateLimiter(
            requests_per_minute=self.config.rate_limit_per_minute,
            requests_per_hour=self.config.rate_limit_per_hour
        )
        self.users = UserManager(self.storage_dir / "users")
    
    def create_user(self, email: str, username: str, password: str,
                   scopes: List[str] = None, is_superuser: bool = False) -> User:
        """Create new user."""
        return self.users.create_user(email, username, password, self.auth, scopes, is_superuser)
    
    def login(self, email: str, password: str) -> Optional[Dict[str, str]]:
        """Login user and return tokens."""
        user = self.users.authenticate_user(email, password, self.auth)
        if not user:
            return None
        
        access_token = self.auth.create_access_token(user.user_id, user.scopes)
        refresh_token = self.auth.create_refresh_token(user.user_id)
        
        return {
            "access_token": access_token,
            "refresh_token": refresh_token,
            "token_type": "bearer",
            "user_id": user.user_id
        }
    
    def refresh_access_token(self, refresh_token: str) -> Optional[str]:
        """Refresh access token."""
        try:
            payload = self.auth.decode_token(refresh_token)
            if payload.get("type") != "refresh":
                return None
            
            user_id = payload.get("sub")
            user = self.users.get_user(user_id)
            if not user or not user.is_active:
                return None
            
            return self.auth.create_access_token(user_id, user.scopes)
        except ValueError:
            return None
    
    def verify_token(self, token: str) -> Optional[Dict[str, Any]]:
        """Verify access token."""
        try:
            return self.auth.decode_token(token)
        except ValueError:
            return None
    
    def create_api_key(self, user_id: str, name: str, scopes: List[str] = None) -> str:
        """Create API key for user."""
        return self.api_keys.generate_key(user_id, name, scopes)
    
    def verify_api_key(self, raw_key: str) -> Optional[APIKey]:
        """Verify API key."""
        return self.api_keys.verify_key(raw_key)
    
    async def check_rate_limit(self, client_id: str) -> bool:
        """Check if request is within rate limits."""
        return await self.rate_limiter.is_allowed(client_id)
    
    def get_rate_limit_status(self, client_id: str) -> dict:
        """Get rate limit status."""
        return self.rate_limiter.get_remaining(client_id)


if __name__ == "__main__":
    print("Security Module")
    print("  - JWTAuth: JWT token creation and verification")
    print("  - APIKeyManager: API key generation and management")
    print("  - RateLimiter: Request rate limiting")
    print("  - UserManager: User account management")
    print("  - SecurityManager: Combined security management")