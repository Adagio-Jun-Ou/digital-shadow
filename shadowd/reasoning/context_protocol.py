"""
Context Protocol - Secure communication between agents and tools

Implements:
- Capability negotiation
- Secure channel establishment
- Context sharing with permissions
- Resource access control
"""

from __future__ import annotations

import copy
import hashlib
import hmac
import json
import logging
import os
import re
import secrets
import threading
import time
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone

def _rcp_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import base64

logger = logging.getLogger(__name__)


class Permission(str, Enum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"
    ADMIN = "admin"


class ResourceType(str, Enum):
    FILE = "file"
    MEMORY = "memory"
    TOOL = "tool"
    NETWORK = "network"
    AGENT = "agent"


@dataclass
class Capability:
    """Agent or tool capability."""
    name: str
    version: str = "1.0"
    permissions: list[Permission] = field(default_factory=list)
    resources: list[ResourceType] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "name": self.name,
            "version": self.version,
            "permissions": list(self.permissions),
            "resources": list(self.resources),
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "Capability":
        return cls(
            name=_SANITIZE_STRICT_RE.sub('', str(data.get("name", ""))),
            version=str(data.get("version", "1.0"))[:20],
            permissions=[Permission(p) for p in data.get("permissions", [])],
            resources=[ResourceType(r) for r in data.get("resources", [])],
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class SecurityContext:
    """Security context for context sharing."""
    entity_id: str
    entity_type: str  # "agent", "tool", "user"
    capabilities: list[Capability] = field(default_factory=list)
    permissions_granted: list[Permission] = field(default_factory=list)
    resources_allowed: list[ResourceType] = field(default_factory=list)
    session_id: str = field(default_factory=lambda: secrets.token_hex(16))
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    parent_context_id: str | None = None
    
    def can_access(self, resource: ResourceType, permission: Permission) -> bool:
        """Check if context has permission for resource."""
        if self.expires_at and datetime.now(timezone.utc) > self.expires_at:
            return False
        
        if permission in self.permissions_granted:
            return True
        
        for cap in self.capabilities:
            if permission in cap.permissions and resource in cap.resources:
                return True
        
        return False
    
    def to_dict(self) -> dict:
        return {
            "entity_id": self.entity_id,
            "entity_type": self.entity_type,
            "capabilities": [c.to_dict() if hasattr(c, 'to_dict') else str(c) for c in self.capabilities],
            "permissions_granted": [p.value if hasattr(p, 'value') else str(p) for p in self.permissions_granted],
            "resources_allowed": [r.value if hasattr(r, 'value') else str(r) for r in self.resources_allowed],
            "session_id": self.session_id,
            "created_at": self.created_at.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "parent_context_id": self.parent_context_id,
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "SecurityContext":
        return cls(
            entity_id=_SANITIZE_STRICT_RE.sub('', str(data.get("entity_id", ""))),
            entity_type=_SANITIZE_STRICT_RE.sub('', str(data.get("entity_type", ""))),
            capabilities=[Capability.from_dict(c) for c in data.get("capabilities", [])],
            permissions_granted=[Permission(p) for p in data.get("permissions_granted", [])],
            resources_allowed=[ResourceType(r) for r in data.get("resources_allowed", [])],
            session_id=data.get("session_id", secrets.token_hex(16)),
            created_at=_rcp_ts(data, "created_at", datetime.now(timezone.utc)),
            expires_at=_rcp_ts(data, "expires_at"),
            parent_context_id=data.get("parent_context_id"),
        )


@dataclass
class ContextMessage:
    """Message exchanged between entities."""
    id: str
    sender_id: str
    receiver_id: str
    message_type: str
    payload: dict
    security_context_id: str | None = None
    signature: str | None = None
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    expires_at: datetime | None = None
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "sender_id": self.sender_id,
            "receiver_id": self.receiver_id,
            "message_type": self.message_type,
            "payload": copy.deepcopy(self.payload) if self.payload else {},
            "security_context_id": self.security_context_id,
            "signature": self.signature,
            "timestamp": self.timestamp.isoformat(),
            "expires_at": self.expires_at.isoformat() if self.expires_at else None,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "ContextMessage":
        return cls(
            id=data.get("id", secrets.token_hex(16)),
            sender_id=_SANITIZE_STRICT_RE.sub('', str(data.get("sender_id", ""))),
            receiver_id=_SANITIZE_STRICT_RE.sub('', str(data.get("receiver_id", ""))),
            message_type=_SANITIZE_STRICT_RE.sub('', str(data.get("message_type", ""))),
            payload=dict(data.get("payload", {})),
            security_context_id=data.get("security_context_id"),
            signature=data.get("signature"),
            timestamp=_rcp_ts(data, "timestamp", datetime.now(timezone.utc)),
            expires_at=_rcp_ts(data, "expires_at"),
            metadata=dict(data.get("metadata", {})),
        )


@dataclass
class Resource:
    """Protected resource."""
    resource_id: str
    resource_type: ResourceType
    path: str
    owner_id: str
    required_permissions: list[Permission] = field(default_factory=list)
    allowed_entities: list[str] | None = None  # None = all authenticated
    metadata: dict = field(default_factory=dict)
    
    def to_dict(self) -> dict:
        return {
            "resource_id": self.resource_id,
            "resource_type": self.resource_type.value,
            "path": self.path,
            "owner_id": self.owner_id,
            "required_permissions": [p.value for p in self.required_permissions],
            "allowed_entities": list(self.allowed_entities) if self.allowed_entities else None,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


class KeyManager:
    """Manages encryption keys for secure channels."""
    
    def __init__(self):
        self._keys: dict[str, bytes] = {}
        self._nonces: dict[str, str] = {}
        self._max_nonces = 10000
        self._lock = threading.Lock()
    
    def generate_key(self, entity_id: str) -> str:
        """Generate new key for entity."""
        with self._lock:
            key = secrets.token_bytes(32)
            self._keys[entity_id] = key
            return base64.b64encode(key).decode()
    
    def get_key(self, entity_id: str) -> bytes | None:
        """Get key for entity."""
        with self._lock:
            return self._keys.get(entity_id)
    
    def generate_nonce(self, entity_id: str) -> str:
        """Generate nonce for replay protection."""
        with self._lock:
            if len(self._nonces) > self._max_nonces:
                self._nonces.clear()
            nonce = secrets.token_hex(16)
            self._nonces[entity_id] = nonce
            return nonce
    
    def verify_nonce(self, entity_id: str, nonce: str) -> bool:
        """Verify nonce for replay protection (single-use)."""
        with self._lock:
            expected = self._nonces.get(entity_id)
            if expected is not None and hmac.compare_digest(expected, nonce):
                del self._nonces[entity_id]
                return True
            return False
    
    def delete_key(self, entity_id: str):
        """Delete key for entity."""
        with self._lock:
            self._keys.pop(entity_id, None)
            self._nonces.pop(entity_id, None)


class ContextProtocol:
    """
    Secure context sharing protocol.
    
    Features:
    - Capability negotiation
    - Secure channel establishment
    - Resource access control
    - Message signing
    - Session management
    """
    
    _MAX_ENTITIES = 1000
    _MAX_RESOURCES = 1000
    _MAX_CHANNELS = 500
    _MAX_HANDLERS = 50
    _MAX_PENDING_NEGOTIATIONS = 100
    _SAFE_ENTITY_ID_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')
    _SAFE_RESOURCE_PATH_RE = re.compile(r'^[a-zA-Z0-9_\-./]{1,256}$')
    _SAFE_MESSAGE_TYPE_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')

    def __init__(self, storage_path: Path | None = None, max_age_seconds: int = 300):
        self.storage_path = storage_path or Path.home() / "digital-shadow" / "data" / "context_protocol"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self._max_age_seconds = max_age_seconds
        
        self._lock = threading.Lock()
        self._entities: dict[str, SecurityContext] = {}
        self._resources: dict[str, Resource] = {}
        self._channels: dict[str, tuple[str, str]] = {}  # channel_id -> (entity1, entity2)
        self._key_manager = KeyManager()
        self._handlers: dict[str, Callable] = {}
        self._pending_negotiations: dict[str, dict] = {}
    
    def register_entity(
        self,
        entity_id: str,
        entity_type: str,
        capabilities: list[Capability],
        permissions: list[Permission] | None = None,
        expires_in_seconds: int | None = None,
    ) -> SecurityContext:
        """Register an entity (agent, tool, user)."""
        entity_type = _SANITIZE_STRICT_RE.sub('', entity_type)
        if not self._SAFE_ENTITY_ID_RE.match(entity_id):
            raise ValueError("Invalid entity_id: must match ^[a-zA-Z0-9_\\-]{1,64}$")
        
        context = SecurityContext(
            entity_id=entity_id,
            entity_type=entity_type,
            capabilities=capabilities,
            permissions_granted=permissions or [],
        )
        
        if expires_in_seconds:
            context.expires_at = datetime.fromtimestamp(
                time.time() + expires_in_seconds,
                tz=timezone.utc
            )
        
        with self._lock:
            if entity_id in self._entities:
                raise ValueError(f"Entity {entity_id} already registered")
            if len(self._entities) >= self._MAX_ENTITIES:
                oldest = next(iter(self._entities))
                logger.warning("Evicting entity %s to make room", oldest)
                self._key_manager.delete_key(oldest)
                del self._entities[oldest]
            self._entities[entity_id] = context
        
        self._key_manager.generate_key(entity_id)
        
        # Persist
        self._save_context(context)
        
        return context
    
    def unregister_entity(self, entity_id: str):
        """Unregister entity."""
        with self._lock:
            self._entities.pop(entity_id, None)
            self._key_manager.delete_key(entity_id)
    
    def register_resource(
        self,
        resource_id: str,
        resource_type: ResourceType,
        path: str,
        owner_id: str,
        required_permissions: list[Permission],
        allowed_entities: list[str] | None = None,
    ) -> Resource:
        """Register a protected resource."""
        resource_id = _SANITIZE_STRICT_RE.sub('', resource_id)
        path = _SANITIZE_STRICT_RE.sub('', path)
        if not self._SAFE_RESOURCE_PATH_RE.match(path):
            path = "/"
        owner_id = _SANITIZE_STRICT_RE.sub('', owner_id)
        resource = Resource(
            resource_id=resource_id,
            resource_type=resource_type,
            path=path,
            owner_id=owner_id,
            required_permissions=required_permissions,
            allowed_entities=allowed_entities,
        )
        
        with self._lock:
            if len(self._resources) >= self._MAX_RESOURCES:
                oldest_rid = next(iter(self._resources))
                logger.warning("Evicting resource %s", oldest_rid)
                del self._resources[oldest_rid]
            self._resources[resource_id] = resource
        
        return resource
    
    def request_access(
        self,
        entity_id: str,
        resource_id: str,
        requested_permission: Permission,
    ) -> bool:
        """Request access to a resource."""
        with self._lock:
            if entity_id not in self._entities:
                return False
            
            if resource_id not in self._resources:
                return False
            
            context = self._entities[entity_id]
            resource = self._resources[resource_id]
        
        # Check if entity is allowed
        if resource.allowed_entities is not None:
            if entity_id not in resource.allowed_entities:
                return False
        
        # Check if entity has required permission
        if not context.can_access(resource.resource_type, requested_permission):
            return False
        
        return True
    
    def negotiate_capabilities(
        self,
        initiator_id: str,
        recipient_id: str,
        requested_capabilities: list[Capability],
    ) -> dict:
        """Negotiate capabilities between entities."""
        with self._lock:
            if initiator_id not in self._entities or recipient_id not in self._entities:
                return {"success": False, "error": "Unknown entity"}
            
            # Clean expired negotiations
            now = datetime.now(timezone.utc)
            expired = []
            for nid, neg in list(self._pending_negotiations.items()):
                try:
                    created = datetime.fromisoformat(neg["created_at"])
                    if (now - created).total_seconds() > self._max_age_seconds:
                        expired.append(nid)
                except (ValueError, TypeError, KeyError):
                    expired.append(nid)
            for nid in expired:
                self._pending_negotiations.pop(nid, None)
            
            if len(self._pending_negotiations) >= self._MAX_PENDING_NEGOTIATIONS:
                oldest_nid = next(iter(self._pending_negotiations))
                self._pending_negotiations.pop(oldest_nid)
                logger.warning("Evicting negotiation %s", oldest_nid)
            
            recipient_context = self._entities[recipient_id]
            
            # Find overlapping capabilities
            negotiated = []
            for cap in requested_capabilities:
                # Check if recipient has matching or higher capability
                for recipient_cap in recipient_context.capabilities:
                    if recipient_cap.name == cap.name:
                        # Merge permissions
                        merged_permissions = list(
                            set(cap.permissions) & set(recipient_cap.permissions)
                        )
                        merged_resources = list(
                            set(cap.resources) & set(recipient_cap.resources)
                        )
                        
                        if merged_permissions:
                            negotiated.append(Capability(
                                name=cap.name,
                                version=cap.version,
                                permissions=merged_permissions,
                                resources=merged_resources,
                            ))
                        break
            
            # Generate channel for communication
            channel_id = secrets.token_hex(16)
            if len(self._channels) >= self._MAX_CHANNELS:
                oldest_ch = next(iter(self._channels))
                logger.warning("Evicting channel %s", oldest_ch)
                self._channels.pop(oldest_ch, None)
            self._channels[channel_id] = (initiator_id, recipient_id)
            
            negotiation_id = secrets.token_hex(8)
            self._pending_negotiations[negotiation_id] = {
                "channel_id": channel_id,
                "initiator": initiator_id,
                "recipient": recipient_id,
                "capabilities": [c.to_dict() for c in negotiated],
                "status": "established",
                "created_at": datetime.now(timezone.utc).isoformat(),
            }
            
            return {
                "success": True,
                "negotiation_id": negotiation_id,
                "channel_id": channel_id,
                "capabilities": [c.to_dict() for c in negotiated],
            }
    
    def create_message(
        self,
        sender_id: str,
        receiver_id: str,
        message_type: str,
        payload: dict,
        sign: bool = True,
    ) -> ContextMessage:
        """Create and optionally sign a message."""
        with self._lock:
            if sender_id not in self._entities:
                raise ValueError("Unknown sender")
        
        if not self._SAFE_MESSAGE_TYPE_RE.match(message_type):
            message_type = "unknown"
        payload = dict(payload) if payload else {}
        if len(payload) > 50:
            payload = dict(list(payload.items())[:50])
        
        with self._lock:
            session_id = self._entities[sender_id].session_id if sender_id in self._entities else None
        
        message = ContextMessage(
            id=secrets.token_hex(16),
            sender_id=sender_id,
            receiver_id=receiver_id,
            message_type=message_type,
            payload=payload,
            security_context_id=session_id,
        )
        
        if sign:
            key = self._key_manager.get_key(sender_id)
            if key:
                message_str = json.dumps(message.to_dict(), sort_keys=True)
                signature = hmac.new(
                    key,
                    message_str.encode(),
                    hashlib.sha256
                ).hexdigest()
                message.signature = signature
        
        return message
    
    def verify_message(self, message: ContextMessage) -> bool:
        """Verify message authenticity."""
        with self._lock:
            if message.sender_id not in self._entities:
                return False
        
        # Check expiration
        if message.expires_at and datetime.now(timezone.utc) > message.expires_at:
            return False
        
        # Verify signature
        if message.signature:
            key = self._key_manager.get_key(message.sender_id)
            if not key:
                return False
            
            message_copy = ContextMessage(
                id=message.id,
                sender_id=message.sender_id,
                receiver_id=message.receiver_id,
                message_type=message.message_type,
                payload=message.payload,
                security_context_id=message.security_context_id,
                signature=None,
                timestamp=message.timestamp,
                metadata=message.metadata,
            )
            
            message_str = json.dumps(message_copy.to_dict(), sort_keys=True)
            expected_sig = hmac.new(
                key,
                message_str.encode(),
                hashlib.sha256
            ).hexdigest()
            
            if not hmac.compare_digest(message.signature, expected_sig):
                return False
        
        return True
    
    def send_message(self, message: ContextMessage) -> bool:
        """Send message through protocol."""
        if not self.verify_message(message):
            return False
        
        with self._lock:
            handler_key = f"{message.receiver_id}:{message.message_type}"
            handler = self._handlers.get(handler_key)
        
        if handler:
            handler(message)
            return True
        
        logger.warning("No handler for %s, message %s discarded", handler_key, message.id)
        return False
    
    def register_handler(
        self,
        receiver_id: str,
        message_type: str,
        handler: Callable[[ContextMessage], None],
    ):
        """Register message handler."""
        with self._lock:
            if len(self._handlers) >= self._MAX_HANDLERS:
                oldest_key = next(iter(self._handlers))
                logger.warning("Evicting handler %s", oldest_key)
                del self._handlers[oldest_key]
            handler_key = f"{_SANITIZE_STRICT_RE.sub('', receiver_id)}:{_SANITIZE_STRICT_RE.sub('', message_type)}"
            self._handlers[handler_key] = handler
    
    def get_context(self, entity_id: str) -> SecurityContext | None:
        """Get security context for entity."""
        with self._lock:
            ctx = self._entities.get(entity_id)
            return copy.deepcopy(ctx) if ctx else None
    
    def get_resource(self, resource_id: str) -> Resource | None:
        """Get resource by ID."""
        with self._lock:
            res = self._resources.get(resource_id)
            return copy.deepcopy(res) if res else None
    
    def list_accessible_resources(
        self,
        entity_id: str,
        resource_type: ResourceType | None = None,
    ) -> list[Resource]:
        """List resources accessible to entity."""
        with self._lock:
            if entity_id not in self._entities:
                return []
            
            context = self._entities[entity_id]
            accessible = []
            
            for resource in self._resources.values():
                if resource_type and resource.resource_type != resource_type:
                    continue
                
                if resource.allowed_entities is not None:
                    if entity_id not in resource.allowed_entities:
                        continue
                
                has_perms = all(
                    context.can_access(resource.resource_type, perm)
                    for perm in resource.required_permissions
                )
                
                if has_perms:
                    accessible.append(copy.deepcopy(resource))
            
            return accessible
    
    def _save_context(self, context: SecurityContext):
        """Persist context to disk (atomic with fsync)."""
        safe_id = hashlib.sha256(context.entity_id.encode()).hexdigest()[:16]
        context_file = self.storage_path / f"context_{safe_id}.json"
        try:
            tmp = context_file.with_suffix(f".tmp.{os.getpid()}")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                json.dump(context.to_dict(), f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(context_file))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save context: %s", type(e).__name__)
    
    def load_context(self, entity_id: str) -> SecurityContext | None:
        """Load context from disk."""
        if not self._SAFE_ENTITY_ID_RE.match(entity_id):
            return None
        safe_id = hashlib.sha256(entity_id.encode()).hexdigest()[:16]
        context_file = self.storage_path / f"context_{safe_id}.json"
        if not context_file.exists():
            return None
        
        try:
            data = json.loads(context_file.read_text())
            context = SecurityContext.from_dict(data)
            with self._lock:
                self._entities[entity_id] = context
            return copy.deepcopy(context)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
            logger.error("Failed to load context: %s", type(e).__name__)
            return None
    
    def get_negotiation(self, negotiation_id: str) -> dict | None:
        """Get negotiation details."""
        with self._lock:
            neg = self._pending_negotiations.get(negotiation_id)
            return dict(neg) if neg else None
    
    def close_channel(self, channel_id: str):
        """Close secure channel."""
        with self._lock:
            self._channels.pop(channel_id, None)
    
    def get_statistics(self) -> dict:
        """Get protocol statistics."""
        with self._lock:
            return {
                "registered_entities": len(self._entities),
                "protected_resources": len(self._resources),
                "active_channels": len(self._channels),
                "pending_negotiations": len(self._pending_negotiations),
            }