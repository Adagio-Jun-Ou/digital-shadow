#!/usr/bin/env python3
"""
Escalation Spectrum - Graduated Autonomy with Guardrails

Implements a spectrum of autonomy levels for Shadow AI actions,
from fully autonomous (Level 0) to human-required (Level 3).

Inspired by:
- OpenAI Agents SDK guardrails (input/output validation)
- Agno governance model (approval workflows, audit logs)
- 12-Factor Agents principle: "Contact human as a tool call"
- NASA's automation levels for spacecraft

Escalation Levels:
  Level 0: Auto - Shadow acts autonomously (routine, low-risk)
  Level 1: Notify - Shadow acts, then notifies you (moderate risk)
  Level 2: Ask - Shadow asks permission first (significant decisions)
  Level 3: Forbid - Shadow cannot act (requires direct human action)

Guardrails:
  - InputGuardrail: Validates/transforms input before processing
  - OutputGuardrail: Validates/transforms output before returning
  - SafetyGuardrail: Prevents harmful/offensive/inappropriate content
  - BudgetGuardrail: Enforces rate limits, token limits, cost limits
"""

import copy
import json
import os
import re
import threading
import time
import uuid
import logging
from datetime import datetime, timezone, timedelta
from typing import Any, Callable, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 2000) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_RE.sub('', text).strip()[:max_length]

logger = logging.getLogger("shadowd.cognition.escalation")


class AutonomyLevel(int, Enum):
    AUTO = 0
    NOTIFY = 1
    ASK = 2
    FORBID = 3


class GuardrailResult(str, Enum):
    PASS = "pass"
    TRANSFORM = "transform"
    REJECT = "reject"
    ESCALATE = "escalate"


class RiskCategory(str, Enum):
    DATA_LOSS = "data_loss"
    PRIVACY = "privacy"
    SECURITY = "security"
    FINANCIAL = "financial"
    REPUTATION = "reputation"
    OPERATIONAL = "operational"
    SOCIAL = "social"
    LEGAL = "legal"


@dataclass
class RiskAssessment:
    category: RiskCategory
    probability: float
    impact: float
    description: str
    mitigation: str = ""

    @property
    def risk_score(self) -> float:
        return self.probability * self.impact

    def to_dict(self) -> Dict:
        return {
            "category": self.category.value,
            "probability": round(self.probability, 3),
            "impact": round(self.impact, 3),
            "risk_score": round(self.risk_score, 3),
            "description": self.description,
            "mitigation": self.mitigation,
        }


@dataclass
class ActionRequest:
    action_id: str = ""
    action_type: str = ""
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)
    domain: Optional[str] = None
    risk_assessments: List[RiskAssessment] = field(default_factory=list)
    suggested_level: AutonomyLevel = AutonomyLevel.AUTO
    actual_level: AutonomyLevel = AutonomyLevel.AUTO
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.action_id:
            self.action_id = f"act_{uuid.uuid4().hex[:12]}"

    @property
    def max_risk(self) -> float:
        if not self.risk_assessments:
            return 0.0
        return max(r.risk_score for r in self.risk_assessments)

    def to_dict(self) -> Dict:
        return {
            "action_id": self.action_id,
            "action_type": self.action_type,
            "description": self.description,
            "parameters": copy.deepcopy(self.parameters),
            "domain": self.domain,
            "risk_assessments": [r.to_dict() for r in self.risk_assessments],
            "max_risk": round(self.max_risk, 3),
            "suggested_level": self.suggested_level.name,
            "actual_level": self.actual_level.name,
            "reason": self.reason,
        }


@dataclass
class EscalationRecord:
    record_id: str = ""
    action_request: Optional[ActionRequest] = None
    level: AutonomyLevel = AutonomyLevel.AUTO
    decision: str = "pending"
    decided_by: str = "shadow"
    reason: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.record_id:
            self.record_id = f"esc_{uuid.uuid4().hex[:12]}"

    def to_dict(self) -> Dict:
        return {
            "record_id": self.record_id,
            "level": self.level.name,
            "decision": self.decision,
            "decided_by": self.decided_by,
            "reason": self.reason,
            "action": self.action_request.to_dict() if self.action_request else None,
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


class Guardrail:
    def __init__(self, name: str, description: str = ""):
        self.name = name
        self.description = description

    async def check(self, input_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        return GuardrailResult.PASS, input_data, ""

    def to_dict(self) -> Dict:
        return {"name": self.name, "description": self.description}


class SafetyGuardrail(Guardrail):
    FORBIDDEN_PATTERNS = [
        r"(?i)(?:delete|remove|drop)\s+(?:all\s+)?(?:data|files|database|records)",
        r"(?i)(?:format|wipe|erase)\s+(?:disk|drive|system)",
        r"(?i)(?:send|transmit|upload)\s+(?:password|credential|secret|token|key).*?(?:to|over|via)",
        r"(?i)(?:sudo|root|admin)\s+(?:rm|delete|drop|format)",
    ]

    def __init__(self):
        super().__init__("safety", "Prevents dangerous or destructive actions")

    async def check(self, input_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        text = str(input_data) if input_data else ""
        for pattern in self.FORBIDDEN_PATTERNS:
            if re.search(pattern, text):
                return (
                    GuardrailResult.ESCALATE,
                    input_data,
                    "Potentially dangerous action detected",
                )
        return GuardrailResult.PASS, input_data, ""


class PrivacyGuardrail(Guardrail):
    SENSITIVE_PATTERNS = [
        r"\b\d{3}-\d{2}-\d{4}\b",
        r"\b\d{4}[-\s]?\d{4}[-\s]?\d{4}[-\s]?\d{4}\b",
        r"\b[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}\b",
        r"\b\d{3}[.\-]\d{3}[.\-]\d{4}\b",
    ]

    def __init__(self):
        super().__init__("privacy", "Detects and masks sensitive data in outputs")

    async def check(self, input_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        text = str(input_data) if input_data else ""
        found = False
        for pattern in self.SENSITIVE_PATTERNS:
            if re.search(pattern, text):
                found = True
                text = re.sub(pattern, "[REDACTED]", text)

        if found:
            return GuardrailResult.TRANSFORM, text, "Sensitive data detected and redacted"
        return GuardrailResult.PASS, input_data, ""


class BudgetGuardrail(Guardrail):
    def __init__(self, max_tokens_per_hour: int = 100000, max_requests_per_minute: int = 60):
        super().__init__("budget", "Enforces rate and cost limits")
        self.max_tokens_per_hour = max_tokens_per_hour
        self.max_requests_per_minute = max_requests_per_minute
        self._token_usage: List[Tuple[float, int]] = []
        self._request_times: List[float] = []
        self._lock = threading.Lock()

    async def check(self, input_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        with self._lock:
            now = time.monotonic()

            self._request_times = [t for t in self._request_times if now - t < 60]
            self._request_times.append(now)

            if len(self._request_times) > self.max_requests_per_minute:
                return GuardrailResult.REJECT, input_data, f"Rate limit: {self.max_requests_per_minute}/min exceeded"

            self._token_usage = [(t, u) for t, u in self._token_usage if now - t < 3600]
            hour_total = sum(u for _, u in self._token_usage)

            if context and "estimated_tokens" in context:
                est = context["estimated_tokens"]
                if hour_total + est > self.max_tokens_per_hour:
                    return GuardrailResult.ESCALATE, input_data, f"Token budget: {hour_total}/{self.max_tokens_per_hour}"

        return GuardrailResult.PASS, input_data, ""


class OutputQualityGuardrail(Guardrail):
    def __init__(self, max_length: int = 10000, min_relevance: float = 0.3):
        super().__init__("output_quality", "Validates output quality")
        self.max_length = max_length
        self.min_relevance = min_relevance

    async def check(self, input_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        if input_data is None:
            return GuardrailResult.REJECT, input_data, "Empty output rejected"

        text = str(input_data)

        if len(text) > self.max_length:
            return GuardrailResult.TRANSFORM, text[:self.max_length], f"Output truncated to {self.max_length} chars"

        if isinstance(input_data, (dict, list)) and len(input_data) == 0:
            return GuardrailResult.PASS, input_data, ""

        if not text.strip():
            return GuardrailResult.REJECT, input_data, "Empty output rejected"

        return GuardrailResult.PASS, input_data, ""


class EscalationManager:
    """
    Manages the escalation spectrum and guardrails for Shadow AI.

    Every action the shadow takes goes through:
    1. Risk assessment (what could go wrong?)
    2. Autonomy level determination (auto/notify/ask/forbid)
    3. Input guardrails (is this request safe?)
    4. Action execution (at the determined level)
    5. Output guardrails (is this response safe?)
    6. Audit logging (record everything)
    """

    _DESTRUCTIVE_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b', re.IGNORECASE) for kw in ["delete", "remove", "drop", "format", "erase", "wipe"]]

    def __init__(self, data_dir: Optional[str] = None, escalation_timeout_hours: float = 24):
        self._action_rules: Dict[str, AutonomyLevel] = {}
        self._domain_rules: Dict[str, AutonomyLevel] = {}
        self._guardrails: List[Guardrail] = []
        self._escalation_queue: List[EscalationRecord] = []
        self._history: List[EscalationRecord] = []
        self._callbacks: Dict[str, List[Callable]] = defaultdict(list)
        self._max_history = 1000
        self._max_queue = 100
        self._max_guardrails = 20
        self._queue_lock = threading.Lock()
        self._history_lock = threading.Lock()
        self._rate_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._rate_requests: List[float] = []
        self._rate_limit_per_minute: int = 120
        self._escalation_timeout_hours: float = escalation_timeout_hours

        self._init_default_rules()
        self._init_guardrails()

        self._stats = {
            "actions_evaluated": 0,
            "auto_approved": 0,
            "notified": 0,
            "asked_permission": 0,
            "forbidden": 0,
            "guardrail_passes": 0,
            "guardrail_blocks": 0,
        }

        self._data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data" / "escalation"
        if data_dir:
            self._data_dir.mkdir(parents=True, exist_ok=True)

    def _init_default_rules(self):
        self._action_rules = {
            "observe": AutonomyLevel.AUTO,
            "search": AutonomyLevel.AUTO,
            "analyze": AutonomyLevel.AUTO,
            "predict": AutonomyLevel.AUTO,
            "summarize": AutonomyLevel.AUTO,
            "respond": AutonomyLevel.AUTO,
            "notify": AutonomyLevel.NOTIFY,
            "schedule": AutonomyLevel.NOTIFY,
            "delegate": AutonomyLevel.NOTIFY,
            "write_file": AutonomyLevel.ASK,
            "send_message": AutonomyLevel.ASK,
            "execute_command": AutonomyLevel.ASK,
            "delete": AutonomyLevel.FORBID,
            "modify_credentials": AutonomyLevel.FORBID,
            "financial_transaction": AutonomyLevel.FORBID,
        }

        self._domain_rules = {
            "career": AutonomyLevel.NOTIFY,
            "health": AutonomyLevel.ASK,
            "finance": AutonomyLevel.FORBID,
            "social": AutonomyLevel.ASK,
            "learning": AutonomyLevel.AUTO,
            "creative": AutonomyLevel.NOTIFY,
            "home": AutonomyLevel.NOTIFY,
        }

    def _init_guardrails(self):
        self._guardrails = [
            SafetyGuardrail(),
            PrivacyGuardrail(),
            BudgetGuardrail(),
            OutputQualityGuardrail(),
        ]

    def add_guardrail(self, guardrail: Guardrail):
        with self._queue_lock:
            if len(self._guardrails) >= self._max_guardrails:
                self._guardrails.pop(0)
            self._guardrails.append(guardrail)

    def set_action_level(self, action: str, level: AutonomyLevel):
        action = _SANITIZE_STRICT_RE.sub('', action).strip()[:100]
        self._action_rules[action] = level

    def set_domain_level(self, domain: str, level: AutonomyLevel):
        domain = _SANITIZE_STRICT_RE.sub('', domain).strip()[:100]
        self._domain_rules[domain] = level

    async def assess_risk(self, action_type: str, parameters: Dict, domain: Optional[str] = None) -> List[RiskAssessment]:
        risks = []

        if any(p.search(action_type) for p in self._DESTRUCTIVE_PATTERNS):
            risks.append(RiskAssessment(
                category=RiskCategory.DATA_LOSS,
                probability=0.8, impact=0.9,
                description="Potentially destructive action",
                mitigation="Require explicit confirmation before execution",
            ))

        sensitive_keys = {"password", "token", "key", "secret", "credential", "ssn"}
        sensitive_data = any(
            k.lower() in sensitive_keys for k in parameters.keys()
        )
        if sensitive_data:
            risks.append(RiskAssessment(
                category=RiskCategory.PRIVACY,
                probability=0.7, impact=0.8,
                description="Action involves sensitive data",
                mitigation="Ensure encryption and access controls",
            ))

        if domain == "finance":
            risks.append(RiskAssessment(
                category=RiskCategory.FINANCIAL,
                probability=0.3, impact=0.9,
                description="Financial domain action",
                mitigation="Require explicit approval",
            ))

        if domain == "health":
            risks.append(RiskAssessment(
                category=RiskCategory.LEGAL,
                probability=0.2, impact=0.7,
                description="Health data handling",
                mitigation="Anonymize data, follow HIPAA guidelines",
            ))

        if not risks:
            risks.append(RiskAssessment(
                category=RiskCategory.OPERATIONAL,
                probability=0.1, impact=0.1,
                description="Low-risk operational action",
            ))

        return risks

    def determine_level(self, request: ActionRequest) -> AutonomyLevel:
        action_level = self._action_rules.get(request.action_type, AutonomyLevel.ASK)
        domain_level = self._domain_rules.get(request.domain, AutonomyLevel.ASK) if request.domain else AutonomyLevel.AUTO
        risk_level = AutonomyLevel.AUTO

        if request.max_risk > 0.7:
            risk_level = AutonomyLevel.FORBID
        elif request.max_risk > 0.5:
            risk_level = AutonomyLevel.ASK
        elif request.max_risk > 0.3:
            risk_level = AutonomyLevel.NOTIFY

        return max(action_level, domain_level, risk_level, key=lambda x: x.value)

    async def evaluate(self, action_type: str, parameters: Dict, domain: Optional[str] = None) -> ActionRequest:
        action_type = _sanitize(action_type, 100)
        if domain:
            domain = _sanitize(domain, 100)
        if parameters and isinstance(parameters, dict):
            parameters = {_sanitize(str(k), 50): _sanitize_content(str(v), 500) for k, v in list(parameters.items())[:50]}
        now = time.monotonic()
        with self._rate_lock:
            self._rate_requests = [t for t in self._rate_requests if now - t < 60]
            if len(self._rate_requests) >= self._rate_limit_per_minute:
                raise ValueError(f"Rate limit exceeded: {self._rate_limit_per_minute}/min")
            self._rate_requests.append(now)

        with self._stats_lock:
            self._stats["actions_evaluated"] += 1

        risks = await self.assess_risk(action_type, parameters, domain)

        request = ActionRequest(
            action_type=action_type,
            parameters=parameters,
            domain=domain,
            risk_assessments=risks,
        )

        level = self.determine_level(request)
        request.suggested_level = level
        request.actual_level = level

        reasons = []
        if level == self._action_rules.get(action_type, AutonomyLevel.ASK):
            reasons.append(f"Action '{action_type}' requires {level.name}")
        if domain and level == self._domain_rules.get(domain, AutonomyLevel.ASK):
            reasons.append(f"Domain '{domain}' requires {level.name}")
        if request.max_risk > 0.3:
            reasons.append(f"Risk score {request.max_risk:.2f} requires {level.name}")
        request.reason = "; ".join(reasons) if reasons else "Default rules applied"

        level_names = {0: "auto_approved", 1: "notified", 2: "asked_permission", 3: "forbidden"}
        with self._stats_lock:
            self._stats[level_names[level.value]] += 1

        return request

    async def check_input_guardrails(self, input_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        current_data = input_data
        for guardrail in list(self._guardrails):
            result, transformed, reason = await guardrail.check(current_data, context)
            if result == GuardrailResult.REJECT:
                with self._stats_lock:
                    self._stats["guardrail_blocks"] += 1
                return result, transformed, f"[{guardrail.name}] {reason}"
            elif result == GuardrailResult.TRANSFORM:
                current_data = transformed
            elif result == GuardrailResult.ESCALATE:
                with self._stats_lock:
                    self._stats["guardrail_blocks"] += 1
                return result, transformed, f"[{guardrail.name}] {reason}"

        with self._stats_lock:
            self._stats["guardrail_passes"] += 1
        return GuardrailResult.PASS, current_data, ""

    async def check_output_guardrails(self, output_data: Any, context: Dict = None) -> Tuple[GuardrailResult, Any, str]:
        result, transformed, reason = await self.check_input_guardrails(output_data, context)
        return result, transformed, reason

    async def can_execute(self, action_type: str, parameters: Dict, domain: Optional[str] = None) -> Tuple[bool, ActionRequest]:
        self.expire_stale_escalations()
        request = await self.evaluate(action_type, parameters, domain)

        input_result, _, input_reason = await self.check_input_guardrails(parameters)
        if input_result in (GuardrailResult.REJECT, GuardrailResult.ESCALATE):
            request.actual_level = AutonomyLevel.FORBID
            request.reason = f"Guardrail blocked: {input_reason}"

        record = EscalationRecord(action_request=request, level=request.actual_level)

        if request.actual_level in (AutonomyLevel.AUTO, AutonomyLevel.NOTIFY):
            record.decision = "approved"
            record.decided_by = "shadow"
            record.resolved_at = datetime.now(timezone.utc).isoformat()
        elif request.actual_level == AutonomyLevel.ASK:
            record.decision = "pending_human"
            with self._queue_lock:
                self._escalation_queue.append(record)
                if len(self._escalation_queue) > self._max_queue:
                    self._escalation_queue = self._escalation_queue[-self._max_queue:]
        else:
            record.decision = "denied"
            record.decided_by = "system"
            record.reason = "Action forbidden by policy"
            record.resolved_at = datetime.now(timezone.utc).isoformat()

        with self._history_lock:
            self._history.append(record)
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        self._persist_history()

        can_execute = request.actual_level in (AutonomyLevel.AUTO, AutonomyLevel.NOTIFY)
        return can_execute, request

    def expire_stale_escalations(self) -> int:
        expired_count = 0
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._escalation_timeout_hours)
        with self._queue_lock:
            for record in self._escalation_queue[:]:
                if record.decision == "pending_human":
                    try:
                        created = datetime.fromisoformat(record.created_at)
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        # <= so a zero-hour timeout expires immediately.
                        if created <= cutoff:
                            record.decision = "denied"
                            record.decided_by = "system_timeout"
                            record.reason = f"Auto-denied: pending escalation exceeded {self._escalation_timeout_hours}h timeout"
                            record.resolved_at = datetime.now(timezone.utc).isoformat()
                            self._escalation_queue.remove(record)
                            with self._history_lock:
                                self._history.append(record)
                            expired_count += 1
                    except (ValueError, TypeError, AttributeError):
                        pass
        with self._history_lock:
            if len(self._history) > self._max_history:
                self._history = self._history[-self._max_history:]
        return expired_count

    @property
    def stale_escalation_count(self) -> int:
        cutoff = datetime.now(timezone.utc) - timedelta(hours=self._escalation_timeout_hours)
        count = 0
        with self._queue_lock:
            for record in self._escalation_queue:
                if record.decision == "pending_human":
                    try:
                        created = datetime.fromisoformat(record.created_at)
                        if created.tzinfo is None:
                            created = created.replace(tzinfo=timezone.utc)
                        if created <= cutoff:
                            count += 1
                    except (ValueError, TypeError):
                        pass
        return count

    def get_pending_escalations(self) -> List[Dict]:
        self.expire_stale_escalations()
        with self._queue_lock:
            return [r.to_dict() for r in self._escalation_queue[:] if r.decision == "pending_human"]

    def resolve_escalation(self, record_id: str, approved: bool, decided_by: str = "user") -> bool:
        with self._queue_lock:
            for record in self._escalation_queue[:]:
                if record.record_id == record_id:
                    record.decision = "approved" if approved else "denied"
                    record.decided_by = decided_by
                    record.resolved_at = datetime.now(timezone.utc).isoformat()
                    self._escalation_queue.remove(record)
                    if approved:
                        with self._stats_lock:
                            self._stats["asked_permission"] = max(0, self._stats["asked_permission"] - 1)
                            self._stats["auto_approved"] += 1
                    return True
        return False

    def get_history(self, limit: int = 50) -> List[Dict]:
        with self._history_lock:
            return [r.to_dict() for r in self._history[-limit:]]

    def get_guardrails(self) -> List[Dict]:
        with self._queue_lock:
            return [g.to_dict() for g in list(self._guardrails)]

    def get_action_rules(self) -> Dict[str, str]:
        with self._queue_lock:
            return {k: v.name for k, v in dict(self._action_rules).items()}

    def get_domain_rules(self) -> Dict[str, str]:
        with self._queue_lock:
            return {k: v.name for k, v in dict(self._domain_rules).items()}

    def get_stats(self) -> Dict:
        with self._queue_lock:
            pending_count = len(self._escalation_queue)
            guardrails_count = len(self._guardrails)
            action_rules_count = len(self._action_rules)
            domain_rules_count = len(self._domain_rules)
        with self._history_lock:
            history_size = len(self._history)
        with self._stats_lock:
            return {
                "stats": dict(self._stats),
                "pending_escalations": pending_count,
                "history_size": history_size,
                "guardrails_count": guardrails_count,
                "action_rules_count": action_rules_count,
                "domain_rules_count": domain_rules_count,
            }

    def _persist_history(self):
        try:
            if self._data_dir:
                with self._history_lock:
                    data = [r.to_dict() for r in self._history[-100:]]
                self._data_dir.mkdir(parents=True, exist_ok=True)
                hist_file = self._data_dir / "escalation_history.json"
                tmp_file = self._data_dir / "escalation_history.json.tmp"
                tmp_file.write_text(json.dumps(data, indent=2, default=str))
                fd = os.open(str(tmp_file), os.O_WRONLY)
                os.fsync(fd)
                os.close(fd)
                tmp_file.replace(hist_file)
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to persist escalation history: %s", type(e).__name__)