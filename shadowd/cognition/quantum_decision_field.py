"""
QuantumDecisionField — Superposition Decision Exploration.

Decisions exist in a superposition of possible outcomes until observation
collapses them into a single chosen path. Entangled decisions influence each
other's probability distributions through correlation links, and constraints
adjust probabilities before collapse.

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import logging
import re
import threading
import time
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_DECISIONS = 1000
_MAX_ENTANGLEMENTS = 500
_MAX_OPTIONS = 20
_MAX_CONSTRAINTS = 10


class QuantumFieldType(str, Enum):
    SUPERPOSITION = "superposition"
    COLLAPSING = "collapsing"
    COLLAPSED = "collapsed"
    ENTANGLED = "entangled"


@dataclass
class DecisionOption:
    option_id: str = ""
    label: str = ""
    description: str = ""
    base_probability: float = 0.0
    adjusted_probability: float = 0.0
    constraints: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.option_id:
            self.option_id = uuid.uuid4().hex[:12]
        self.label = _sanitize(str(self.label), 100)
        self.description = _sanitize_content(str(self.description), 500)
        self.base_probability = max(0.0, min(1.0, self.base_probability))
        self.adjusted_probability = max(0.0, min(1.0, self.adjusted_probability))
        self.constraints = [_sanitize(str(c), 200) for c in self.constraints[:_MAX_CONSTRAINTS]]

    def to_dict(self) -> Dict:
        return {
            "option_id": self.option_id,
            "label": self.label,
            "description": self.description,
            "base_probability": round(self.base_probability, 4),
            "adjusted_probability": round(self.adjusted_probability, 4),
            "constraints": list(self.constraints),
        }


@dataclass
class QuantumDecision:
    decision_id: str = ""
    options: List[DecisionOption] = field(default_factory=list)
    probabilities: Dict[str, float] = field(default_factory=dict)
    superposition_state: str = ""
    collapse_trigger: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = ""

    def __post_init__(self):
        if not self.decision_id:
            self.decision_id = "qd_%s" % uuid.uuid4().hex[:12]
        self.options = self.options[:_MAX_OPTIONS]
        self.probabilities = {_sanitize(str(k), 50): max(0.0, min(1.0, float(v))) for k, v in list(self.probabilities.items())[:_MAX_OPTIONS]}
        self.superposition_state = _sanitize(str(self.superposition_state), 50)
        self.collapse_trigger = _sanitize(str(self.collapse_trigger), 100)
        self.metadata = {_sanitize(str(k), 50): _sanitize(str(v), 500) for k, v in list(self.metadata.items())[:20]}
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "options": [o.to_dict() for o in self.options],
            "probabilities": dict(self.probabilities),
            "superposition_state": self.superposition_state,
            "collapse_trigger": self.collapse_trigger,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class EntanglementLink:
    link_id: str = ""
    decision_id_1: str = ""
    decision_id_2: str = ""
    correlation: float = 0.0
    created_at: str = ""

    def __post_init__(self):
        if not self.link_id:
            self.link_id = "ent_%s" % uuid.uuid4().hex[:12]
        self.decision_id_1 = _sanitize(str(self.decision_id_1), 50)
        self.decision_id_2 = _sanitize(str(self.decision_id_2), 50)
        self.correlation = max(-1.0, min(1.0, self.correlation))
        self.created_at = self.created_at or datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "link_id": self.link_id,
            "decision_id_1": self.decision_id_1,
            "decision_id_2": self.decision_id_2,
            "correlation": round(self.correlation, 4),
            "created_at": self.created_at,
        }


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_len]


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


class QuantumDecisionField:
    """Superposition Decision Exploration Engine.

    Decisions exist in superposition of possible outcomes until observation
    collapses them. Entangled decisions influence each other's probability
    distributions through correlation links.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._decisions: Dict[str, QuantumDecision] = {}
        self._decision_order: deque = deque(maxlen=_MAX_DECISIONS)
        self._entanglements: Dict[str, EntanglementLink] = {}
        self._entanglement_order: deque = deque(maxlen=_MAX_ENTANGLEMENTS)
        self._stats = {
            "decisions_created": 0,
            "observations_made": 0,
            "collapses_forced": 0,
            "entanglements_created": 0,
            "probability_adjustments": 0,
            "constraints_added": 0,
        }

    def create_decision(self, options: Optional[List[Dict]] = None,
                        context: Optional[str] = None,
                        metadata: Optional[Dict] = None) -> QuantumDecision:
        if options is None:
            options = []
        if metadata is None:
            metadata = {}

        decision_options = []
        equal_prob = 1.0 / max(len(options), 1) if options else 0.5
        for i, opt in enumerate(options[:_MAX_OPTIONS]):
            if isinstance(opt, dict):
                base_prob = opt.get("probability", equal_prob)
                do = DecisionOption(
                    label=_sanitize(opt.get("label", f"Option {i + 1}"), 100),
                    description=_sanitize_content(opt.get("description", ""), 500),
                    base_probability=_clamp(float(base_prob), 0.0, 1.0),
                    adjusted_probability=_clamp(float(base_prob), 0.0, 1.0),
                    constraints=[_sanitize(str(c), 200) for c in opt.get("constraints", [])[:_MAX_CONSTRAINTS]],
                )
            else:
                base_prob = getattr(opt, "base_probability", equal_prob)
                orig_id = getattr(opt, "option_id", "") or ""
                do = DecisionOption(
                    option_id=orig_id,
                    label=_sanitize(getattr(opt, "label", f"Option {i + 1}") or f"Option {i + 1}", 100),
                    description=_sanitize_content(getattr(opt, "description", "") or "", 500),
                    base_probability=_clamp(float(base_prob), 0.0, 1.0),
                    adjusted_probability=_clamp(float(base_prob), 0.0, 1.0),
                    constraints=[_sanitize(str(c), 200) for c in (getattr(opt, "constraints", None) or [])[:_MAX_CONSTRAINTS]],
                )
            decision_options.append(do)

        if not decision_options:
            decision_options.append(DecisionOption(label="Default", base_probability=1.0, adjusted_probability=1.0))

        probabilities = {o.option_id: o.adjusted_probability for o in decision_options}
        self._normalize_probabilities(probabilities)

        for o in decision_options:
            o.adjusted_probability = probabilities.get(o.option_id, 0.0)

        decision = QuantumDecision(
            options=decision_options,
            probabilities=probabilities,
            superposition_state=QuantumFieldType.SUPERPOSITION.value,
            collapse_trigger=_sanitize(context or "manual", 100),
            metadata=metadata,
        )

        with self._lock:
            if len(self._decisions) >= _MAX_DECISIONS:
                evict_id = self._decision_order.popleft()
                evicted = self._decisions.pop(evict_id, None)
                if evicted:
                    logger.warning("Evicting decision %s (cap=%d)", evict_id, _MAX_DECISIONS)
                self._entanglements = {k: v for k, v in self._entanglements.items()
                                       if v.decision_id_1 != evict_id and v.decision_id_2 != evict_id}
            self._decisions[decision.decision_id] = decision
            self._decision_order.append(decision.decision_id)
            with self._stats_lock:
                self._stats["decisions_created"] += 1

        return copy.deepcopy(decision)

    def add_constraint(self, decision_id: str, option_id: str, constraint: str) -> bool:
        decision_id = _sanitize(decision_id, 64)
        option_id = _sanitize(option_id, 64)
        constraint = _sanitize(constraint, 200)

        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                return False
            if decision.superposition_state == QuantumFieldType.COLLAPSED.value:
                return False

            for opt in decision.options:
                if opt.option_id == option_id:
                    if len(opt.constraints) >= _MAX_CONSTRAINTS:
                        opt.constraints.pop(0)
                    opt.constraints.append(constraint)
                    opt.adjusted_probability = max(0.01, opt.adjusted_probability * 0.9)
                    break
            else:
                return False

            self._normalize_probabilities(decision.probabilities)
            for opt in decision.options:
                if opt.option_id in decision.probabilities:
                    opt.adjusted_probability = decision.probabilities[opt.option_id]
            decision.probabilities = {o.option_id: o.adjusted_probability for o in decision.options}

            with self._stats_lock:
                self._stats["constraints_added"] += 1
        return True

    def entangle(self, decision_id_1: str, decision_id_2: str, correlation: float) -> EntanglementLink:
        decision_id_1 = _sanitize(decision_id_1, 64)
        decision_id_2 = _sanitize(decision_id_2, 64)
        correlation = _clamp(correlation, -1.0, 1.0)

        link = EntanglementLink(
            decision_id_1=decision_id_1,
            decision_id_2=decision_id_2,
            correlation=correlation,
        )

        with self._lock:
            if len(self._entanglements) >= _MAX_ENTANGLEMENTS:
                evict_id = self._entanglement_order.popleft()
                self._entanglements.pop(evict_id, None)
                logger.warning("Evicting entanglement %s (cap=%d)", evict_id, _MAX_ENTANGLEMENTS)
            self._entanglements[link.link_id] = link
            self._entanglement_order.append(link.link_id)

            d1 = self._decisions.get(decision_id_1)
            d2 = self._decisions.get(decision_id_2)
            if d1:
                d1.superposition_state = QuantumFieldType.ENTANGLED.value
            if d2:
                d2.superposition_state = QuantumFieldType.ENTANGLED.value
            with self._stats_lock:
                self._stats["entanglements_created"] += 1

        return copy.deepcopy(link)

    def observe(self, decision_id: str, option_id: str) -> Optional[DecisionOption]:
        decision_id = _sanitize(decision_id, 64)
        option_id = _sanitize(option_id, 64)

        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                return None
            if decision.superposition_state == QuantumFieldType.COLLAPSED.value:
                return None

            chosen = None
            for opt in decision.options:
                if opt.option_id == option_id:
                    chosen = copy.deepcopy(opt)
                    break
            if chosen is None:
                return None

            for opt in decision.options:
                if opt.option_id == option_id:
                    opt.adjusted_probability = 1.0
                else:
                    opt.adjusted_probability = 0.0

            decision.probabilities = {o.option_id: o.adjusted_probability for o in decision.options}
            decision.superposition_state = QuantumFieldType.COLLAPSED.value
            decision.collapse_trigger = "observation:%s" % option_id

            for link in self._entanglements.values():
                other_id = None
                if link.decision_id_1 == decision_id:
                    other_id = link.decision_id_2
                elif link.decision_id_2 == decision_id:
                    other_id = link.decision_id_1
                if other_id and other_id in self._decisions:
                    other = self._decisions[other_id]
                    if other.superposition_state != QuantumFieldType.COLLAPSED.value:
                        self._apply_entanglement_influence(other, link.correlation)

            with self._stats_lock:
                self._stats["observations_made"] += 1

        return chosen

    def _apply_entanglement_influence(self, decision: QuantumDecision, correlation: float) -> None:
        adjustment = correlation * 0.1
        for opt in decision.options:
            opt.adjusted_probability = _clamp(opt.adjusted_probability + adjustment, 0.01, 0.99)
        self._normalize_probabilities(decision.probabilities)
        for opt in decision.options:
            if opt.option_id in decision.probabilities:
                opt.adjusted_probability = decision.probabilities[opt.option_id]
        decision.probabilities = {o.option_id: o.adjusted_probability for o in decision.options}

    def collapse(self, decision_id: str) -> Optional[QuantumDecision]:
        decision_id = _sanitize(decision_id, 64)

        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                return None
            if decision.superposition_state == QuantumFieldType.COLLAPSED.value:
                return copy.deepcopy(decision)

            max_prob = 0.0
            chosen_option = None
            for opt in decision.options:
                if opt.adjusted_probability > max_prob:
                    max_prob = opt.adjusted_probability
                    chosen_option = opt

            if chosen_option is None and decision.options:
                chosen_option = decision.options[0]

            for opt in decision.options:
                if opt is chosen_option:
                    opt.adjusted_probability = 1.0
                else:
                    opt.adjusted_probability = 0.0

            decision.probabilities = {o.option_id: o.adjusted_probability for o in decision.options}
            decision.superposition_state = QuantumFieldType.COLLAPSED.value
            decision.collapse_trigger = "forced:collapse"

            for link in self._entanglements.values():
                other_id = None
                if link.decision_id_1 == decision_id:
                    other_id = link.decision_id_2
                elif link.decision_id_2 == decision_id:
                    other_id = link.decision_id_1
                if other_id and other_id in self._decisions:
                    other = self._decisions[other_id]
                    if other.superposition_state != QuantumFieldType.COLLAPSED.value:
                        self._apply_entanglement_influence(other, link.correlation)

            with self._stats_lock:
                self._stats["collapses_forced"] += 1

        return copy.deepcopy(decision)

    def _normalize_probabilities(self, probabilities: Dict[str, float]) -> None:
        total = sum(probabilities.values())
        if total > 0:
            for key in probabilities:
                probabilities[key] = probabilities[key] / total
        else:
            count = len(probabilities)
            for key in probabilities:
                probabilities[key] = 1.0 / max(count, 1)

    def get_decision(self, decision_id: str) -> Optional[QuantumDecision]:
        decision_id = _sanitize(decision_id, 64)
        with self._lock:
            decision = self._decisions.get(decision_id)
        return copy.deepcopy(decision) if decision else None

    def list_decisions(self, limit: int = 50, offset: int = 0) -> List[QuantumDecision]:
        limit = max(1, min(limit, _MAX_DECISIONS))
        with self._lock:
            decisions = list(self._decisions.values())
        return [copy.deepcopy(d) for d in decisions[offset:offset + limit]]

    def get_entanglements(self, decision_id: str) -> List[EntanglementLink]:
        decision_id = _sanitize(decision_id, 64)
        with self._lock:
            links = [link for link in self._entanglements.values()
                     if link.decision_id_1 == decision_id or link.decision_id_2 == decision_id]
        return [copy.deepcopy(l) for l in links]

    def adjust_probability(self, decision_id: str, option_id: str, delta: float) -> bool:
        decision_id = _sanitize(decision_id, 64)
        option_id = _sanitize(option_id, 64)

        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                return False
            if decision.superposition_state == QuantumFieldType.COLLAPSED.value:
                return False

            for opt in decision.options:
                if opt.option_id == option_id:
                    opt.adjusted_probability = _clamp(opt.adjusted_probability + float(delta), 0.01, 0.99)
                    break
            else:
                return False

            decision.probabilities = {o.option_id: o.adjusted_probability for o in decision.options}
            self._normalize_probabilities(decision.probabilities)
            for opt in decision.options:
                if opt.option_id in decision.probabilities:
                    opt.adjusted_probability = decision.probabilities[opt.option_id]
            decision.probabilities = {o.option_id: o.adjusted_probability for o in decision.options}

            with self._stats_lock:
                self._stats["probability_adjustments"] += 1
        return True

    def get_probability_field(self, decision_id: str) -> Dict[str, float]:
        decision_id = _sanitize(decision_id, 64)
        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                return {}
            return {k: round(v, 4) for k, v in decision.probabilities.items()}

    def compute_interference(self, decision_id: str) -> Dict[str, float]:
        decision_id = _sanitize(decision_id, 64)

        with self._lock:
            decision = self._decisions.get(decision_id)
            if decision is None:
                return {}

            interference = {o.option_id: 0.0 for o in decision.options}

            for link in self._entanglements.values():
                other_id = None
                if link.decision_id_1 == decision_id:
                    other_id = link.decision_id_2
                elif link.decision_id_2 == decision_id:
                    other_id = link.decision_id_1
                if other_id is None:
                    continue

                other = self._decisions.get(other_id)
                if other is None:
                    continue

                for opt in decision.options:
                    for other_opt in other.options:
                        correlation_effect = link.correlation * other_opt.adjusted_probability * 0.1
                        interference[opt.option_id] = interference.get(opt.option_id, 0.0) + correlation_effect

            result = {}
            for opt_id, val in interference.items():
                result[opt_id] = round(val, 4)

            return result

    def get_stats(self) -> Dict:
        with self._stats_lock:
            return dict(self._stats)