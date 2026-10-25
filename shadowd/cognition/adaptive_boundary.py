#!/usr/bin/env python3
"""
AdaptiveBoundary -- Cognitive Boundary Management and Energy Conservation.

Manages cognitive boundaries and energy budgets. Tracks cognitive energy
expenditure, enforces boundaries between domains (work/family/health/creative),
prevents cross-domain contamination, and enforces rest/recovery cycles.

Inspired by ego boundaries in psychology and energy conservation in physics.

Key concepts:
  - BoundaryRule: regulates information flow between cognitive domains
  - EnergyBudget: tracks energy allocation and expenditure per domain
  - BoundaryViolation: records boundary crossing events
  - CognitiveEnergyProfile: aggregate energy state across all domains

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import logging
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.adaptive_boundary")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_RULES = 500
_MAX_VIOLATIONS = 1000
_MAX_BUDGETS = 8
_MAX_HISTORY = 500

_BOUNDARY_PERMEABILITY_MAP = {
    "rigid": 0.0,
    "flexible": 0.3,
    "permeable": 0.7,
    "dissolved": 1.0,
}

_ENERGY_STATE_THRESHOLDS = {
    "depleted": (0.0, 0.2),
    "low": (0.2, 0.4),
    "moderate": (0.4, 0.6),
    "high": (0.6, 0.8),
    "full": (0.8, 1.0),
}

_RECOVERY_SUGGESTIONS = {
    "depleted": [
        "Stop all non-essential cognitive activity immediately",
        "Initiate deep rest cycle -- minimum 30 minutes",
        "Transfer active tasks to lower-energy domains",
        "Activate conservation mode across all domains",
    ],
    "low": [
        "Reduce cross-domain task switching",
        "Prioritize single-domain focus for 20 minutes",
        "Defer low-priority tasks to next recovery cycle",
        "Limit social and creative domain expenditure",
    ],
    "moderate": [
        "Schedule brief rest in 15 minutes",
        "Batch similar-domain tasks together",
        "Avoid high-energy cross-domain transitions",
        "Monitor spending rate for acceleration",
    ],
    "high": [
        "Maintain current allocation strategy",
        "Consider investing energy in growth domains",
        "Boundary permeability can be temporarily increased",
    ],
    "full": [
        "All domains fully resourced",
        "Ideal time for creative synthesis and cross-pollination",
        "Consider increasing boundary permeability for innovation",
    ],
}


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _classify_energy(energy: float) -> str:
    if energy < 0.2:
        return "depleted"
    elif energy < 0.4:
        return "low"
    elif energy < 0.6:
        return "moderate"
    elif energy < 0.8:
        return "high"
    else:
        return "full"


class BoundaryType(str, Enum):
    RIGID = "rigid"
    FLEXIBLE = "flexible"
    PERMEABLE = "permeable"
    DISSOLVED = "dissolved"


class DomainType(str, Enum):
    WORK = "work"
    FAMILY = "family"
    HEALTH = "health"
    CREATIVE = "creative"
    SOCIAL = "social"
    FINANCIAL = "financial"
    PERSONAL_GROWTH = "personal_growth"
    SPIRITUAL = "spiritual"


class EnergyState(str, Enum):
    DEPLETED = "depleted"
    LOW = "low"
    MODERATE = "moderate"
    HIGH = "high"
    FULL = "full"


@dataclass
class BoundaryRule:
    rule_id: str = ""
    source_domain: str = ""
    target_domain: str = ""
    boundary_type: str = "flexible"
    permeability: float = 0.3
    energy_cost: float = 0.1

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = "br_%s" % uuid.uuid4().hex[:12]
        self.rule_id = _sanitize(self.rule_id, 64)
        self.source_domain = _sanitize(self.source_domain, 100)
        self.target_domain = _sanitize(self.target_domain, 100)
        if isinstance(self.boundary_type, BoundaryType):
            self.boundary_type = self.boundary_type.value
        self.boundary_type = _sanitize(self.boundary_type, 50)
        self.permeability = _clamp(self.permeability)
        self.energy_cost = _clamp(self.energy_cost)

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "boundary_type": self.boundary_type,
            "permeability": round(self.permeability, 4),
            "energy_cost": round(self.energy_cost, 4),
        }


@dataclass
class EnergyBudget:
    budget_id: str = ""
    domain: str = ""
    available_energy: float = 1.0
    allocated_energy: float = 0.0
    spent_energy: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.budget_id:
            self.budget_id = "eb_%s" % uuid.uuid4().hex[:12]
        self.budget_id = _sanitize(self.budget_id, 64)
        self.domain = _sanitize(self.domain, 100)
        self.available_energy = _clamp(self.available_energy)
        self.allocated_energy = _clamp(self.allocated_energy)
        self.spent_energy = _clamp(self.spent_energy)

    def to_dict(self) -> Dict:
        return {
            "budget_id": self.budget_id,
            "domain": self.domain,
            "available_energy": round(self.available_energy, 4),
            "allocated_energy": round(self.allocated_energy, 4),
            "spent_energy": round(self.spent_energy, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class BoundaryViolation:
    violation_id: str = ""
    source_domain: str = ""
    target_domain: str = ""
    content: str = ""
    severity: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.violation_id:
            self.violation_id = "bv_%s" % uuid.uuid4().hex[:12]
        self.violation_id = _sanitize(self.violation_id, 64)
        self.source_domain = _sanitize(self.source_domain, 100)
        self.target_domain = _sanitize(self.target_domain, 100)
        self.content = _sanitize_content(self.content, 10000)
        self.severity = _clamp(self.severity)

    def to_dict(self) -> Dict:
        return {
            "violation_id": self.violation_id,
            "source_domain": self.source_domain,
            "target_domain": self.target_domain,
            "content": self.content,
            "severity": round(self.severity, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class CognitiveEnergyProfile:
    profile_id: str = ""
    total_energy: float = 1.0
    domain_allocations: Dict[str, float] = field(default_factory=dict)
    recovery_rate: float = 0.1
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.profile_id:
            self.profile_id = "cep_%s" % uuid.uuid4().hex[:12]
        self.profile_id = _sanitize(self.profile_id, 64)
        self.total_energy = _clamp(self.total_energy)
        self.recovery_rate = _clamp(self.recovery_rate)
        sanitized = {}
        for k, v in list(self.domain_allocations.items())[:50]:
            sanitized[_sanitize(str(k), 100)] = _clamp(float(v)) if isinstance(v, (int, float)) else 0.0
        self.domain_allocations = sanitized

    def to_dict(self) -> Dict:
        return {
            "profile_id": self.profile_id,
            "total_energy": round(self.total_energy, 4),
            "domain_allocations": copy.deepcopy(self.domain_allocations),
            "recovery_rate": round(self.recovery_rate, 4),
            "timestamp": self.timestamp,
        }


class AdaptiveBoundary:
    """Cognitive Boundary Management and Energy Conservation.

    Manages cognitive boundaries and energy budgets across life domains.
    Tracks cognitive energy expenditure, enforces boundaries between domains,
    prevents cross-domain contamination, and enforces rest/recovery cycles.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._rules: Dict[str, BoundaryRule] = {}
        self._rule_order: deque = deque(maxlen=_MAX_RULES)
        self._budgets: Dict[str, EnergyBudget] = {}
        self._violations: Dict[str, BoundaryViolation] = {}
        self._violation_order: deque = deque(maxlen=_MAX_VIOLATIONS)
        self._profiles: deque = deque(maxlen=_MAX_HISTORY)
        self._stats = {
            "boundaries_set": 0,
            "boundaries_removed": 0,
            "boundary_checks": 0,
            "violations_reported": 0,
            "energy_allocations": 0,
            "energy_spends": 0,
            "energy_recoveries": 0,
            "profiles_generated": 0,
        }

    def _get_domain_key(self, domain_a: str, domain_b: str) -> str:
        a = _sanitize(domain_a, 100).lower()
        b = _sanitize(domain_b, 100).lower()
        if a < b:
            return "%s|%s" % (a, b)
        return "%s|%s" % (b, a)

    def _find_rule_for_domains(self, domain_a: str, domain_b: str) -> Optional[BoundaryRule]:
        pair_key = self._get_domain_key(domain_a, domain_b)
        for rule in self._rules.values():
            rk = self._get_domain_key(rule.source_domain, rule.target_domain)
            if rk == pair_key:
                return rule
        return None

    def set_boundary(self, domain_a: str, domain_b: str,
                     boundary_type: str = "flexible",
                     permeability = None,
                     energy_cost: float = 0.1) -> BoundaryRule:
        domain_a = _sanitize(domain_a, 100)
        domain_b = _sanitize(domain_b, 100)
        if isinstance(boundary_type, BoundaryType):
            boundary_type = boundary_type.value
        boundary_type = _sanitize(boundary_type, 50)

        if permeability is None or isinstance(permeability, str):
            perm_type = permeability if isinstance(permeability, str) else boundary_type
            if perm_type in _BOUNDARY_PERMEABILITY_MAP:
                permeability = _BOUNDARY_PERMEABILITY_MAP[perm_type]
            else:
                permeability = 0.3
        else:
            permeability = _clamp(float(permeability))

        energy_cost = _clamp(energy_cost)

        with self._lock:
            existing = self._find_rule_for_domains(domain_a, domain_b)
            if existing:
                existing.source_domain = domain_a
                existing.target_domain = domain_b
                existing.boundary_type = boundary_type
                existing.permeability = permeability
                existing.energy_cost = energy_cost
                with self._stats_lock:
                    self._stats["boundaries_set"] += 1
                return copy.deepcopy(existing)

            rule = BoundaryRule(
                rule_id="br_%s" % uuid.uuid4().hex[:12],
                source_domain=domain_a,
                target_domain=domain_b,
                boundary_type=boundary_type,
                permeability=permeability,
                energy_cost=energy_cost,
            )

            if len(self._rules) >= _MAX_RULES:
                evict_id = self._rule_order.popleft()
                self._rules.pop(evict_id, None)
                logger.warning("Evicting rule %s (cap=%d)", evict_id, _MAX_RULES)
            self._rules[rule.rule_id] = rule
            self._rule_order.append(rule.rule_id)
            with self._stats_lock:
                self._stats["boundaries_set"] += 1

            return copy.deepcopy(rule)

    def remove_boundary(self, rule_id: str) -> bool:
        rid = _sanitize(rule_id, 64)

        with self._lock:
            if rid in self._rules:
                del self._rules[rid]
                try:
                    self._rule_order.remove(rid)
                except ValueError:
                    pass
                with self._stats_lock:
                    self._stats["boundaries_removed"] += 1
                return True
        return False

    def check_boundary(self, source_domain: str, target_domain: str,
                       content: str = "") -> Dict[str, Any]:
        source_domain = _sanitize(source_domain, 100)
        target_domain = _sanitize(target_domain, 100)
        content_preview = _sanitize_content(content, 500)[:200]

        with self._lock:
            with self._stats_lock:
                self._stats["boundary_checks"] += 1

            rule = self._find_rule_for_domains(source_domain, target_domain)
            if not rule:
                return {
                    "allowed": True,
                    "reason": "no_rule",
                    "source_domain": source_domain,
                    "target_domain": target_domain,
                    "boundary_type": "none",
                    "permeability": 1.0,
                    "energy_cost": 0.0,
                }

            allowed = rule.permeability >= 0.5 or rule.boundary_type in ("permeable", "dissolved")

            if rule.boundary_type == "rigid":
                allowed = False
            elif rule.boundary_type == "flexible":
                content_lower = content_preview.lower()
                sensitive_markers = ["confidential", "private", "secret", "personal", "sensitive"]
                for marker in sensitive_markers:
                    if re.search(r'\b' + re.escape(marker) + r'\b', content_lower):
                        allowed = False
                        break

            return {
                "allowed": allowed,
                "reason": rule.boundary_type if not allowed else "permitted",
                "source_domain": source_domain,
                "target_domain": target_domain,
                "boundary_type": rule.boundary_type,
                "permeability": round(rule.permeability, 4),
                "energy_cost": round(rule.energy_cost, 4),
            }

    def allocate_energy(self, domain: str, amount: float) -> Optional[EnergyBudget]:
        domain = _sanitize(domain, 100)
        amount = _clamp(amount)

        with self._lock:
            if domain not in self._budgets:
                if len(self._budgets) >= _MAX_BUDGETS:
                    evict_domain = next(iter(self._budgets))
                    del self._budgets[evict_domain]
                    logger.warning("Evicting budget for domain %s (cap=%d)", evict_domain, _MAX_BUDGETS)
                self._budgets[domain] = EnergyBudget(
                    budget_id="eb_%s" % uuid.uuid4().hex[:12],
                    domain=domain,
                    available_energy=1.0,
                    allocated_energy=0.0,
                    spent_energy=0.0,
                )

            budget = self._budgets[domain]
            available = budget.available_energy - budget.allocated_energy - budget.spent_energy
            actual = min(amount, max(0.0, available))
            budget.allocated_energy = _clamp(budget.allocated_energy + actual)
            budget.timestamp = datetime.now(timezone.utc).isoformat()

            with self._stats_lock:
                self._stats["energy_allocations"] += 1

            return copy.deepcopy(budget)

    def spend_energy(self, domain: str, amount: float) -> Optional[EnergyBudget]:
        domain = _sanitize(domain, 100)
        amount = _clamp(amount)

        with self._lock:
            budget = self._budgets.get(domain)
            if not budget:
                return None

            available = budget.allocated_energy - budget.spent_energy
            actual = min(amount, max(0.0, available))
            budget.spent_energy = _clamp(budget.spent_energy + actual)
            budget.allocated_energy = _clamp(budget.allocated_energy)
            budget.timestamp = datetime.now(timezone.utc).isoformat()

            with self._stats_lock:
                self._stats["energy_spends"] += 1

            return copy.deepcopy(budget)

    def recover_energy(self, domain: str = "", amount: float = 0.1) -> CognitiveEnergyProfile:
        amount = _clamp(amount)

        with self._lock:
            domain_allocations = {}

            if domain:
                d = _sanitize(domain, 100)
                budget = self._budgets.get(d)
                if budget:
                    recovered = min(amount, budget.spent_energy)
                    budget.spent_energy = _clamp(budget.spent_energy - recovered)
                    net = budget.available_energy - budget.spent_energy
                    domain_allocations[d] = _clamp(net)
                    budget.timestamp = datetime.now(timezone.utc).isoformat()
            else:
                for d, budget in self._budgets.items():
                    spent = budget.spent_energy
                    recovered = spent * amount
                    budget.spent_energy = _clamp(budget.spent_energy - recovered)

                    over_allocated = budget.allocated_energy - budget.spent_energy
                    if over_allocated > 0:
                        release = over_allocated * amount * 0.5
                        budget.allocated_energy = _clamp(budget.allocated_energy - release)

                    spent_remaining = budget.spent_energy
                    allocated_remaining = budget.allocated_energy
                    domain_allocations[d] = _clamp(1.0 - spent_remaining - allocated_remaining)

                    budget.timestamp = datetime.now(timezone.utc).isoformat()

            total = 0.0
            num_domains = max(1, len(self._budgets))
            for alloc in domain_allocations.values():
                total += alloc
            total_energy = _clamp(total / num_domains) if domain_allocations else 1.0

            profile = CognitiveEnergyProfile(
                profile_id="cep_%s" % uuid.uuid4().hex[:12],
                total_energy=total_energy,
                domain_allocations=copy.deepcopy(domain_allocations),
                recovery_rate=amount,
            )

            if len(self._profiles) >= _MAX_HISTORY:
                self._profiles.popleft()
            self._profiles.append(profile)

            with self._stats_lock:
                self._stats["energy_recoveries"] += 1
                self._stats["profiles_generated"] += 1

        return copy.deepcopy(profile)

    def get_energy_profile(self) -> Dict[str, Any]:
        with self._lock:
            domain_allocations = {}
            for domain, budget in self._budgets.items():
                net = budget.available_energy - budget.spent_energy
                domain_allocations[domain] = {
                    "total_energy": round(_clamp(net), 4),
                    "allocated": round(budget.allocated_energy, 4),
                    "spent": round(budget.spent_energy, 4),
                    "available": round(budget.available_energy, 4),
                }

            total = 0.0
            num_domains = max(1, len(domain_allocations))
            if domain_allocations:
                for v in domain_allocations.values():
                    total += v["total_energy"]
                total_energy = _clamp(total / num_domains)
            else:
                total_energy = 1.0

        return {
            "total_energy": round(total_energy, 4),
            "domain_allocations": copy.deepcopy(domain_allocations),
        }

    def report_violation(self, source_domain: str, target_domain: str,
                         description: str = "", severity = 0.5) -> BoundaryViolation:
        source_domain = _sanitize(source_domain, 100)
        target_domain = _sanitize(target_domain, 100)
        description = _sanitize_content(description, 10000)
        if isinstance(severity, str):
            severity_map = {"low": 0.2, "medium": 0.5, "high": 0.8, "critical": 1.0}
            severity = severity_map.get(severity.lower(), 0.5)
        severity = _clamp(float(severity))

        violation = BoundaryViolation(
            violation_id="bv_%s" % uuid.uuid4().hex[:12],
            source_domain=source_domain,
            target_domain=target_domain,
            content=description,
            severity=severity,
        )

        with self._lock:
            if len(self._violations) >= _MAX_VIOLATIONS:
                evict_id = self._violation_order.popleft()
                self._violations.pop(evict_id, None)
                logger.warning("Evicting violation %s (cap=%d)", evict_id, _MAX_VIOLATIONS)
            self._violations[violation.violation_id] = violation
            self._violation_order.append(violation.violation_id)
            with self._stats_lock:
                self._stats["violations_reported"] += 1

        return copy.deepcopy(violation)

    def get_violations(self, domain: str = "", limit: int = 50) -> List[Dict]:
        domain = _sanitize(domain, 100)
        limit = max(1, min(500, limit))

        with self._lock:
            violations = []
            for vid in reversed(list(self._violation_order)):
                v = self._violations.get(vid)
                if not v:
                    continue
                if domain and v.source_domain != domain and v.target_domain != domain:
                    continue
                violations.append(copy.deepcopy(v).to_dict())
                if len(violations) >= limit:
                    break

        return violations

    def suggest_recovery(self, state: str = "") -> Dict[str, Any]:
        profile = self.get_energy_profile()
        total_energy = profile.get("total_energy", 1.0)
        domain_allocs = profile.get("domain_allocations", {})
        if state:
            energy_state = state if state in ("depleted", "low", "moderate", "high", "full") else _classify_energy(_clamp(float(state)) if state.replace(".", "").isdigit() else 0.0)
        else:
            energy_state = _classify_energy(total_energy)
        suggestions = list(_RECOVERY_SUGGESTIONS.get(energy_state, []))

        depleted_domains = []
        for domain, alloc_data in domain_allocs.items():
            if isinstance(alloc_data, dict):
                domain_energy = alloc_data.get("total_energy", 1.0)
            else:
                domain_energy = _clamp(float(alloc_data))
            dom_state = _classify_energy(domain_energy)
            if dom_state in ("depleted", "low"):
                depleted_domains.append({
                    "domain": domain,
                    "energy_level": round(domain_energy, 4),
                    "state": dom_state,
                })

        domain_suggestions = []
        for dd in depleted_domains:
            if dd["state"] == "depleted":
                domain_suggestions.append(
                    "Domain '%s' is depleted (%.1f%%) -- immediate rest required" % (
                        dd["domain"], dd["energy_level"] * 100
                    )
                )
            else:
                domain_suggestions.append(
                    "Domain '%s' is low (%.1f%%) -- reduce activity" % (
                        dd["domain"], dd["energy_level"] * 100
                    )
                )

        return {
            "energy_state": energy_state,
            "total_energy": round(total_energy, 4),
            "general_suggestions": suggestions,
            "depleted_domains": depleted_domains,
            "domain_specific_suggestions": domain_suggestions,
        }

    def get_boundary_rules(self, domain: str = "") -> List[Dict]:
        domain = _sanitize(domain, 100)

        with self._lock:
            rules = []
            for rule in self._rules.values():
                if domain and rule.source_domain != domain and rule.target_domain != domain:
                    continue
                rules.append(copy.deepcopy(rule).to_dict())

        return rules

    def get_stats(self) -> Dict:
        with self._lock:
            total_rules = len(self._rules)
            total_violations = len(self._violations)
            total_budgets = len(self._budgets)
            total_profiles = len(self._profiles)

            violations_by_severity = {"low": 0, "medium": 0, "high": 0}
            for v in self._violations.values():
                if v.severity < 0.3:
                    violations_by_severity["low"] += 1
                elif v.severity < 0.7:
                    violations_by_severity["medium"] += 1
                else:
                    violations_by_severity["high"] += 1

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_rules"] = total_rules
        stats["total_violations"] = total_violations
        stats["total_budgets"] = total_budgets
        stats["total_profiles"] = total_profiles
        stats["violations_by_severity"] = violations_by_severity
        return stats