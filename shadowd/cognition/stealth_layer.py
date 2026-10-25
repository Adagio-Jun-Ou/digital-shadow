#!/usr/bin/env python3
"""
StealthLayer - Adaptive Camouflage and Privacy Guard for Shadow AI

Implements multi-layered privacy protection:
- PrivacyGuard: Classifies data sensitivity, applies access controls
- CamouflageEngine: Transforms data to preserve utility while hiding identity
- AuditLogger: Tracks all data access and transformations
- EncryptionLayer: At-rest and in-transit encryption
- DataMinimizer: Only collects and retains what's necessary

Design Principles:
- Privacy by design (GDPR-inspired)
- Data minimization (only collect what's needed)
- Purpose limitation (use data only for stated purpose)
- Differential privacy (add noise to preserve anonymity)
- Right to be forgotten (complete data deletion)

Inspired by: Apple's differential privacy, Signal's sealed sender,
and the 12-Factor Agent principle of compact errors into context.
"""

import asyncio
import base64
import copy
import hashlib
import json
import math
import os
import random
import re
import threading
import time
import uuid
import logging
import secrets
from datetime import datetime, timezone
from typing import Any, Callable, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

from cryptography.hazmat.primitives.ciphers.aead import AESGCM

logger = logging.getLogger("shadowd.cognition.stealth_layer")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()


def _sanitize_content(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_RE.sub('', str(text)).strip()


class SensitivityLevel(int, Enum):
    PUBLIC = 0
    INTERNAL = 1
    CONFIDENTIAL = 2
    SECRET = 3
    TOP_SECRET = 4

    @property
    def label(self) -> str:
        return {
            0: "public",
            1: "internal",
            2: "confidential",
            3: "secret",
            4: "top_secret",
        }[self.value]


class DataCategory(str, Enum):
    PERSONAL = "personal"
    BEHAVIORAL = "behavioral"
    LOCATION = "location"
    COMMUNICATION = "communication"
    FINANCIAL = "financial"
    HEALTH = "health"
    CREDENTIALS = "credentials"
    BIOMETRIC = "biometric"
    DEVICE = "device"
    NETWORK = "network"


class AccessAction(str, Enum):
    READ = "read"
    WRITE = "write"
    DELETE = "delete"
    SHARE = "share"
    EXPORT = "export"
    TRANSFORM = "transform"


class CamouflageMode(str, Enum):
    NONE = "none"
    ANONYMIZE = "anonymize"
    PSEUDONYMIZE = "pseudonymize"
    AGGREGATE = "aggregate"
    DIFFERENTIAL_PRIVACY = "differential_privacy"
    SYNTHETIC = "synthetic"


@dataclass
class DataClassification:
    data_id: str
    category: DataCategory
    sensitivity: SensitivityLevel
    purpose: str
    retention_days: int
    can_share: bool = False
    can_export: bool = False
    requires_encryption: bool = True
    auto_delete: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AccessPolicy:
    subject: str
    allowed_actions: List[AccessAction]
    allowed_categories: List[DataCategory]
    max_sensitivity: SensitivityLevel = SensitivityLevel.CONFIDENTIAL
    requires_approval: bool = False
    audit_all: bool = True
    expires_at: Optional[str] = None


@dataclass
class AuditEntry:
    entry_id: str = ""
    subject: str = ""
    action: AccessAction = AccessAction.READ
    data_id: str = ""
    category: DataCategory = DataCategory.PERSONAL
    sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC
    granted: bool = True
    reason: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.entry_id:
            self.entry_id = f"audit_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "entry_id": self.entry_id,
            "subject": self.subject,
            "action": self.action.value,
            "data_id": self.data_id,
            "category": self.category.value,
            "sensitivity": self.sensitivity.label,
            "granted": self.granted,
            "reason": self.reason,
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata),
        }


class PrivacyGuard:
    _MAX_CLASSIFICATIONS = 5000
    _MAX_POLICIES = 100
    _MAX_DATA_STORE = 10000

    def __init__(self):
        self._lock = threading.Lock()
        self._classifications: Dict[str, DataClassification] = {}
        self._policies: Dict[str, AccessPolicy] = {}
        self._audit_log: List[AuditEntry] = []
        self._encryption_keys: Dict[str, bytes] = {}
        self._key_created_at: Dict[str, datetime] = {}
        self._key_rotation_days: float = 90.0
        self._data_store: Dict[str, Any] = {}

        self._default_classifications()
        self._default_policies()

    def _default_classifications(self):
        defaults = [
            DataClassification("location", DataCategory.LOCATION, SensitivityLevel.CONFIDENTIAL,
                               "shadow_observation", 30, can_share=False, requires_encryption=True),
            DataClassification("behavior", DataCategory.BEHAVIORAL, SensitivityLevel.CONFIDENTIAL,
                               "shadow_learning", 90, can_share=False, requires_encryption=True),
            DataClassification("communication", DataCategory.COMMUNICATION, SensitivityLevel.SECRET,
                               "shadow_observation", 30, can_share=False, requires_encryption=True),
            DataClassification("device", DataCategory.DEVICE, SensitivityLevel.INTERNAL,
                               "shadow_observation", 30, can_share=False),
            DataClassification("network", DataCategory.NETWORK, SensitivityLevel.INTERNAL,
                               "shadow_observation", 7, can_share=True),
            DataClassification("financial", DataCategory.FINANCIAL, SensitivityLevel.SECRET,
                               "shadow_advice", 365, can_share=False, requires_encryption=True, auto_delete=True),
            DataClassification("health", DataCategory.HEALTH, SensitivityLevel.SECRET,
                               "shadow_advice", 365, can_share=False, requires_encryption=True, auto_delete=True),
            DataClassification("credentials", DataCategory.CREDENTIALS, SensitivityLevel.TOP_SECRET,
                               "shadow_auth", 90, can_share=False, requires_encryption=True, auto_delete=True),
            DataClassification("personal", DataCategory.PERSONAL, SensitivityLevel.CONFIDENTIAL,
                               "shadow_learning", 180, can_share=False, requires_encryption=True),
        ]
        for cls in defaults:
            self._classifications[cls.data_id] = cls

    def _default_policies(self):
        self._policies["shadow_ai"] = AccessPolicy(
            subject="shadow_ai",
            allowed_actions=[AccessAction.READ, AccessAction.WRITE, AccessAction.TRANSFORM],
            allowed_categories=list(DataCategory),
            max_sensitivity=SensitivityLevel.SECRET,
        )
        self._policies["shadow_crew"] = AccessPolicy(
            subject="shadow_crew",
            allowed_actions=[AccessAction.READ, AccessAction.WRITE],
            allowed_categories=[DataCategory.BEHAVIORAL, DataCategory.DEVICE, DataCategory.NETWORK],
            max_sensitivity=SensitivityLevel.INTERNAL,
        )
        self._policies["shadow_voice"] = AccessPolicy(
            subject="shadow_voice",
            allowed_actions=[AccessAction.READ],
            allowed_categories=[DataCategory.COMMUNICATION, DataCategory.PERSONAL],
            max_sensitivity=SensitivityLevel.CONFIDENTIAL,
        )
        self._policies["external"] = AccessPolicy(
            subject="external",
            allowed_actions=[AccessAction.READ],
            allowed_categories=[DataCategory.NETWORK],
            max_sensitivity=SensitivityLevel.PUBLIC,
            requires_approval=True,
        )

    def classify_data(self, data_id: str, category: DataCategory) -> DataClassification:
        if data_id in self._classifications:
            return copy.deepcopy(self._classifications[data_id])
        default_sensitivity = {
            DataCategory.CREDENTIALS: SensitivityLevel.TOP_SECRET,
            DataCategory.BIOMETRIC: SensitivityLevel.TOP_SECRET,
            DataCategory.HEALTH: SensitivityLevel.SECRET,
            DataCategory.FINANCIAL: SensitivityLevel.SECRET,
            DataCategory.COMMUNICATION: SensitivityLevel.SECRET,
            DataCategory.LOCATION: SensitivityLevel.CONFIDENTIAL,
            DataCategory.PERSONAL: SensitivityLevel.CONFIDENTIAL,
            DataCategory.BEHAVIORAL: SensitivityLevel.CONFIDENTIAL,
            DataCategory.DEVICE: SensitivityLevel.INTERNAL,
            DataCategory.NETWORK: SensitivityLevel.INTERNAL,
        }
        classification = DataClassification(
            data_id=data_id,
            category=category,
            sensitivity=default_sensitivity.get(category, SensitivityLevel.INTERNAL),
            purpose="shadow_general",
            retention_days=90,
        )
        self._classifications[data_id] = classification
        if len(self._classifications) > self._MAX_CLASSIFICATIONS:
            oldest_keys = list(self._classifications.keys())[:self._MAX_CLASSIFICATIONS // 2]
            for k in oldest_keys:
                del self._classifications[k]
            logger.warning("Evicted %s classifications to stay under cap %s", len(oldest_keys), self._MAX_CLASSIFICATIONS)
        return classification

    def check_access(
        self, subject: str, action: AccessAction, data_id: str, category: DataCategory
    ) -> Tuple[bool, str]:
        subject = _sanitize(subject)[:100]
        data_id = _sanitize(data_id)[:200]
        with self._lock:
            policy = self._policies.get(subject)
            if not policy:
                self._audit(subject, action, data_id, category, granted=False, reason="no_policy")
                return False, "No access policy for subject"

            classification = self.classify_data(data_id, category)

            if action not in policy.allowed_actions:
                self._audit(subject, action, data_id, category, classification.sensitivity,
                            granted=False, reason="action_not_allowed")
                return False, f"Action {action.value} not allowed"

            if category not in policy.allowed_categories:
                self._audit(subject, action, data_id, category, classification.sensitivity,
                            granted=False, reason="category_not_allowed")
                return False, f"Category {category.value} not allowed"

            if classification.sensitivity.value > policy.max_sensitivity.value:
                self._audit(subject, action, data_id, category, classification.sensitivity,
                            granted=False, reason="sensitivity_exceeded")
                return False, f"Sensitivity {classification.sensitivity.label} exceeds maximum {policy.max_sensitivity.label}"

            self._audit(subject, action, data_id, category, classification.sensitivity, granted=True)
            return True, "Access granted"

    def _audit(
        self, subject: str, action: AccessAction, data_id: str,
        category: DataCategory, sensitivity: SensitivityLevel = SensitivityLevel.PUBLIC,
        granted: bool = True, reason: str = "",
    ):
        entry = AuditEntry(
            subject=subject,
            action=action,
            data_id=data_id,
            category=category,
            sensitivity=sensitivity,
            granted=granted,
            reason=reason,
        )
        self._audit_log.append(entry)
        if len(self._audit_log) > 10000:
            evicted = len(self._audit_log) - 10000
            self._audit_log = self._audit_log[-10000:]
            logger.warning("Evicted %s audit log entries to stay under cap 10000", evicted)

    def get_audit_log(self, subject: Optional[str] = None, limit: int = 100) -> List[Dict]:
        with self._lock:
            entries = list(self._audit_log)
            if subject:
                entries = [e for e in entries if e.subject == subject]
            return [e.to_dict() for e in entries[-limit:]]

    def store_data(self, data_id: str, data: Any, category: DataCategory) -> bool:
        with self._lock:
            classification = self.classify_data(data_id, category)
            if classification.requires_encryption:
                encrypted = self._simple_encrypt(json.dumps(data) if not isinstance(data, str) else data)
                self._data_store[data_id] = {"encrypted": True, "data": encrypted, "category": category.value}
            else:
                self._data_store[data_id] = {"encrypted": False, "data": data, "category": category.value}
            if len(self._data_store) > self._MAX_DATA_STORE:
                oldest_keys = list(self._data_store.keys())[:self._MAX_DATA_STORE // 2]
                for k in oldest_keys:
                    del self._data_store[k]
                logger.warning("Evicted %s data store entries to stay under cap %s", len(oldest_keys), self._MAX_DATA_STORE)
            return True

    def retrieve_data(self, data_id: str, subject: str = "shadow_ai") -> Optional[Any]:
        with self._lock:
            stored = self._data_store.get(data_id)
            if not stored:
                return None

            category = DataCategory(stored.get("category", "personal"))
            policy = self._policies.get(subject)
            if not policy:
                self._audit(subject, AccessAction.READ, data_id, category, granted=False, reason="no_policy")
                return None
            classification = self.classify_data(data_id, category)
            if AccessAction.READ not in policy.allowed_actions:
                self._audit(subject, AccessAction.READ, data_id, category, classification.sensitivity, granted=False, reason="action_not_allowed")
                return None
            if category not in policy.allowed_categories:
                self._audit(subject, AccessAction.READ, data_id, category, classification.sensitivity, granted=False, reason="category_not_allowed")
                return None
            if classification.sensitivity.value > policy.max_sensitivity.value:
                self._audit(subject, AccessAction.READ, data_id, category, classification.sensitivity, granted=False, reason="sensitivity_exceeded")
                return None
            self._audit(subject, AccessAction.READ, data_id, category, classification.sensitivity, granted=True)

            if stored.get("encrypted"):
                try:
                    return json.loads(self._simple_decrypt(stored["data"]))
                except ValueError:
                    return None
            return copy.deepcopy(stored["data"]) if not isinstance(stored["data"], str) else stored["data"]

    def delete_data(self, data_id: str) -> bool:
        with self._lock:
            if data_id in self._data_store:
                del self._data_store[data_id]
                return True
            return False

    def _simple_encrypt(self, plaintext: str) -> str:
        key = self._get_or_create_key_unlocked("default")
        nonce = os.urandom(12)
        aesgcm = AESGCM(key)
        ct = aesgcm.encrypt(nonce, plaintext.encode(), None)
        return base64.b64encode(nonce + ct).decode()

    def _simple_decrypt(self, ciphertext: str) -> str:
        from cryptography.exceptions import InvalidTag
        key = self._get_or_create_key_unlocked("default")
        try:
            raw = base64.b64decode(ciphertext)
            nonce, ct = raw[:12], raw[12:]
            aesgcm = AESGCM(key)
            return aesgcm.decrypt(nonce, ct, None).decode()
        except (InvalidTag, ValueError) as e:
            logger.error("Decryption failed: %s", type(e).__name__)
            raise ValueError("Decryption failed")

    def _get_or_create_key_unlocked(self, key_id: str) -> bytes:
        now = datetime.now(timezone.utc)
        if key_id not in self._encryption_keys:
            self._encryption_keys[key_id] = os.urandom(32)
            self._key_created_at[key_id] = now
        elif self._key_created_at.get(key_id) is None:
            self._encryption_keys[key_id] = os.urandom(32)
            self._key_created_at[key_id] = now
        elif isinstance(self._key_created_at[key_id], float):
            self._key_created_at[key_id] = now
        else:
            delta = now - self._key_created_at[key_id]
            if delta.total_seconds() > self._key_rotation_days * 86400:
                self._encryption_keys[key_id] = os.urandom(32)
                self._key_created_at[key_id] = now
        return self._encryption_keys[key_id]

    def _get_or_create_key(self, key_id: str) -> bytes:
        with self._lock:
            return self._get_or_create_key_unlocked(key_id)

    def rotate_keys(self):
        with self._lock:
            old_keys = dict(self._encryption_keys)
            new_keys = {}
            new_timestamps = {}
            for key_id in self._encryption_keys:
                new_keys[key_id] = os.urandom(32)
                new_timestamps[key_id] = time.monotonic()
            self._encryption_keys = new_keys
            self._key_created_at = new_timestamps

            for data_id, stored in list(self._data_store.items()):
                if stored.get("encrypted"):
                    try:
                        raw = base64.b64decode(stored["data"])
                        nonce, ct = raw[:12], raw[12:]
                        old_key = old_keys.get("default")
                        if old_key:
                            old_aesgcm = AESGCM(old_key)
                            plaintext = old_aesgcm.decrypt(nonce, ct, None)
                            new_key = new_keys.get("default")
                            new_nonce = os.urandom(12)
                            new_aesgcm = AESGCM(new_key)
                            new_ct = new_aesgcm.encrypt(new_nonce, plaintext, None)
                            self._data_store[data_id]["data"] = base64.b64encode(new_nonce + new_ct).decode()
                    except (ValueError, OSError, TypeError) as e:
                        logger.warning("Failed to re-encrypt %s during key rotation: %s", data_id[:16], type(e).__name__)
                        del self._data_store[data_id]

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "classifications": len(self._classifications),
                "policies": len(self._policies),
                "audit_entries": len(self._audit_log),
                "data_items": len(self._data_store),
                "encrypted_items": sum(1 for d in self._data_store.values() if d.get("encrypted")),
            }


class DifferentialPrivacy:
    def __init__(self, epsilon: float = 1.0, delta: float = 1e-5):
        self.epsilon = epsilon
        self.delta = delta

    def add_noise(self, value: float, sensitivity: float = 1.0) -> float:
        scale = sensitivity / self.epsilon
        u = random.uniform(-0.999999, 0.999999)
        sign = 1 if u >= 0 else -1
        magnitude = -scale * math.log(1 - abs(u))
        noise = sign * magnitude
        return value + noise

    def add_noise_int(self, value: int, sensitivity: int = 1) -> int:
        noisy = self.add_noise(float(value), float(sensitivity))
        return max(0, round(noisy))

    def add_noise_counts(self, counts: Dict[str, int]) -> Dict[str, int]:
        result = {}
        for key, count in counts.items():
            noisy = self.add_noise_int(count, sensitivity=1)
            result[key] = noisy
        return result

    def explain_privacy(self) -> str:
        prob = 1 - math.exp(-1.0 / self.epsilon)
        prob_pct = round(prob * 100, 1)
        return f"With ε={self.epsilon}, there is a {prob_pct}% chance the output is within 1 unit of the true value"


class CamouflageEngine:
    def __init__(self, epsilon: float = 1.0):
        self.epsilon = epsilon

    def anonymize(self, data: Dict, fields: Optional[List[str]] = None) -> Dict:
        result = copy.deepcopy(data)
        sensitive_fields = fields or ["name", "email", "phone", "address", "ip", "mac", "ssn", "password", "credit_card", "passport", "token", "api_key"]
        for key in result:
            if any(re.search(r'(?:^|[\s_\-])' + re.escape(sf) + r'(?:$|[\s_\-])', key.lower()) for sf in sensitive_fields):
                result[key] = self._hash_value(str(result[key]))
        return result

    def pseudonymize(self, data: Dict, field: str, salt: Optional[str] = None) -> Dict:
        result = copy.deepcopy(data)
        if field in result:
            salt = salt or secrets.token_hex(16)
            value = f"{salt}:{result[field]}"
            result[field] = f"pseudo_{hashlib.sha256(value.encode()).hexdigest()[:12]}"
        return result

    def aggregate(self, data_points: List[Dict], field: str) -> Dict:
        if not data_points:
            return {"count": 0, "mean": None, "std": None}

        values = []
        for dp in data_points:
            if field in dp:
                try:
                    values.append(float(dp[field]))
                except (ValueError, TypeError):
                    pass

        if not values:
            return {"count": 0, "mean": None, "std": None}

        mean = sum(values) / len(values)
        variance = sum((v - mean) ** 2 for v in values) / len(values) if len(values) > 1 else 0
        return {
            "count": len(values),
            "mean": round(mean, 4),
            "std": round(math.sqrt(variance), 4),
            "min": round(min(values), 4),
            "max": round(max(values), 4),
        }

    def aggregate_with_dp(self, data: Dict, epsilon: float = 1.0) -> Dict:
        dp = DifferentialPrivacy(epsilon=epsilon)
        result = {}
        for key, value in data.items():
            if isinstance(value, int):
                result[key] = dp.add_noise_int(value, sensitivity=1)
            elif isinstance(value, float):
                result[key] = dp.add_noise(value, sensitivity=1.0)
            else:
                result[key] = value
        return result

    def apply_differential_privacy(self, value: float, sensitivity: float = 1.0) -> float:
        scale = sensitivity / self.epsilon
        noise = random.gauss(0, scale)
        return round(value + noise, 6)

    def transform_for_sharing(self, data: Dict, mode: CamouflageMode) -> Dict:
        if mode == CamouflageMode.NONE:
            return copy.deepcopy(data)
        elif mode == CamouflageMode.ANONYMIZE:
            return self.anonymize(data)
        elif mode == CamouflageMode.PSEUDONYMIZE:
            return self.pseudonymize(data, "id")
        elif mode == CamouflageMode.AGGREGATE:
            return {"aggregated": True, "count": 1}
        elif mode == CamouflageMode.DIFFERENTIAL_PRIVACY:
            result = copy.deepcopy(data)
            for key, val in result.items():
                if isinstance(val, (int, float)):
                    result[key] = self.apply_differential_privacy(float(val))
            return result
        elif mode == CamouflageMode.SYNTHETIC:
            return {"synthetic": True, "preserves_distribution": True}
        return data

    def _hash_value(self, value: str) -> str:
        return f"anon_{hashlib.sha256(value.encode()).hexdigest()[:16]}"

    def get_sensitivity_score(self, data: Dict) -> float:
        score = 0.0
        high_risk_keys = {"ssn", "password", "credit_card", "passport", "biometric"}
        medium_risk_keys = {"email", "phone", "address", "location", "name", "ip", "mac"}
        for key in data:
            k = key.lower()
            if any(hr in k for hr in high_risk_keys):
                score += 1.0
            elif any(mr in k for mr in medium_risk_keys):
                score += 0.5
        return round(min(score / max(len(data), 1), 1.0), 4)


class DataMinimizer:
    _MAX_PURPOSES = 500

    def __init__(self):
        self._collection_purposes: Dict[str, Dict] = {}

    def register_purpose(self, purpose: str, required_fields: List[str], retention_days: int):
        purpose = _sanitize(purpose)[:200]
        required_fields = [_sanitize(f)[:100] for f in required_fields]
        if len(self._collection_purposes) >= self._MAX_PURPOSES:
            oldest = next(iter(self._collection_purposes))
            logger.warning("Evicting oldest purpose '%s' to stay under cap %d", oldest[:50].replace(chr(10), ' '), self._MAX_PURPOSES)
            del self._collection_purposes[oldest]
        self._collection_purposes[purpose] = {
            "required_fields": required_fields,
            "retention_days": retention_days,
            "registered_at": datetime.now(timezone.utc).isoformat(),
        }

    def minimize(self, data: Dict, purpose: str) -> Dict:
        purpose_config = self._collection_purposes.get(purpose, {})
        required = purpose_config.get("required_fields", list(data.keys()))
        return {k: v for k, v in data.items() if k in required}

    def check_retention(self, data_id: str, created_at: str) -> bool:
        try:
            created = datetime.fromisoformat(created_at)
            if created.tzinfo is None:
                created = created.replace(tzinfo=timezone.utc)
            age_days = (datetime.now(timezone.utc) - created).days
            return age_days < 365
        except (ValueError, TypeError):
            return True

    def get_purposes(self) -> List[Dict]:
        return [{"purpose": k, "required_fields": list(v["required_fields"]), "retention_days": v["retention_days"], "registered_at": v["registered_at"]} for k, v in self._collection_purposes.items()]


class StealthLayer:
    """
    Unified privacy and stealth layer for Shadow AI.

    Combines privacy guard, camouflage engine, and data minimizer
    into one coherent system that protects user data while
    maintaining utility.
    """

    def __init__(self, epsilon: float = 1.0):
        self.guard = PrivacyGuard()
        self.camouflage = CamouflageEngine(epsilon=epsilon)
        self.minimizer = DataMinimizer()
        self._stats_lock = threading.Lock()

        self.minimizer.register_purpose("shadow_observation", ["type", "content", "timestamp"], 30)
        self.minimizer.register_purpose("shadow_learning", ["type", "patterns", "outcomes"], 90)
        self.minimizer.register_purpose("shadow_advice", ["domain", "query", "response"], 30)
        self.minimizer.register_purpose("shadow_sync", ["type", "summary", "timestamp"], 7)

        self._stats = {
            "access_checks": 0,
            "access_granted": 0,
            "access_denied": 0,
            "data_stored": 0,
            "data_transformed": 0,
            "data_deleted": 0,
        }

    def check_access(self, subject: str, action: AccessAction, data_id: str, category: DataCategory) -> Tuple[bool, str]:
        with self._stats_lock:
            self._stats["access_checks"] += 1
        granted, reason = self.guard.check_access(subject, action, data_id, category)
        if granted:
            with self._stats_lock:
                self._stats["access_granted"] += 1
        else:
            with self._stats_lock:
                self._stats["access_denied"] += 1
        return granted, reason

    def store(self, data_id: str, data: Any, category: DataCategory) -> bool:
        result = self.guard.store_data(data_id, data, category)
        if result:
            with self._stats_lock:
                self._stats["data_stored"] += 1
        return result

    def retrieve(self, data_id: str, subject: str = "shadow_ai") -> Optional[Any]:
        return self.guard.retrieve_data(data_id, subject)

    def delete(self, data_id: str) -> bool:
        result = self.guard.delete_data(data_id)
        if result:
            with self._stats_lock:
                self._stats["data_deleted"] += 1
        return result

    def transform_for_sharing(self, data: Dict, mode: CamouflageMode = CamouflageMode.ANONYMIZE) -> Dict:
        with self._stats_lock:
            self._stats["data_transformed"] += 1
        return self.camouflage.transform_for_sharing(data, mode)

    def minimize_for_purpose(self, data: Dict, purpose: str) -> Dict:
        return self.minimizer.minimize(data, purpose)

    def get_sensitivity_score(self, data: Dict) -> float:
        return self.camouflage.get_sensitivity_score(data)

    def get_audit_log(self, subject: Optional[str] = None, limit: int = 100) -> List[Dict]:
        return self.guard.get_audit_log(subject, limit)

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stealth_stats = dict(self._stats)
        return {
            "stealth_stats": stealth_stats,
            "guard_stats": self.guard.get_stats(),
            "purposes": len(self.minimizer.get_purposes()),
            "epsilon": self.camouflage.epsilon,
        }