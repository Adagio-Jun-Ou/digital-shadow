#!/usr/bin/env python3
"""
NeuralSymbolicBridge — Neural Pattern ↔ Symbolic Reasoning Bridge.

Translates between distributed neural activation patterns (sub-symbolic,
analog, graded) and discrete symbolic representations (concepts, rules,
propositions). Enables the system to reason symbolically about patterns
it detects neurally, and to inject symbolic knowledge back as pattern
priors.

Key concepts:
  - NeuralPattern: a distributed activation vector over semantic dimensions
  - SymbolicRule: a discrete proposition with premises → conclusion
  - BridgeMapping: learned correspondences between pattern clusters and symbols
  - InferenceMode: NEURO_TO_SYMBOLIC, SYMBOLIC_TO_NEURO, BIDIRECTIONAL

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
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("shadowd.cognition.neural_symbolic_bridge")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_PATTERNS = 5000
_MAX_RULES = 1000
_MAX_MAPPINGS = 2000
_MAX_DIMENSIONS = 200
_MAX_HISTORY = 2000
_MAX_METADATA_KEYS = 20
_MAX_REASON_LENGTH = 500
_MAX_SYMBOL_INDEX = 5000
_MAX_PATTERNS_PER_SYMBOL = 100


class InferenceMode(str, Enum):
    NEURO_TO_SYMBOLIC = "neuro_to_symbolic"
    SYMBOLIC_TO_NEURO = "symbolic_to_neuro"
    BIDIRECTIONAL = "bidirectional"


class PatternType(str, Enum):
    ACTIVATION = "activation"
    INHIBITION = "inhibition"
    RESONANCE = "resonance"
    DECAY = "decay"
    BURST = "burst"


class RuleStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    AXIOMATIC = "axiomatic"


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0, decimals: int = 4) -> float:
    return round(max(lo, min(hi, value)), decimals)


@dataclass
class NeuralPattern:
    pattern_id: str
    pattern_type: PatternType
    dimensions: Dict[str, float]
    symbol: Optional[str]
    confidence: float
    context: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pattern_id:
            self.pattern_id = uuid.uuid4().hex[:12]
        self.pattern_id = _sanitize(self.pattern_id, 50)
        self.symbol = _sanitize(self.symbol, 100) if self.symbol else None
        self.context = _sanitize_content(self.context, _MAX_REASON_LENGTH)
        self.confidence = _clamp(self.confidence)
        if len(self.dimensions) > _MAX_DIMENSIONS:
            items = list(self.dimensions.items())[:_MAX_DIMENSIONS]
            self.dimensions = {k: _clamp(v) for k, v in items}
        else:
            self.dimensions = {str(k)[:50]: _clamp(v) for k, v in self.dimensions.items()}
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type.value,
            "dimensions": dict(self.dimensions),
            "symbol": self.symbol,
            "confidence": self.confidence,
            "context": self.context,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class SymbolicRule:
    rule_id: str
    premises: List[str]
    conclusion: str
    strength: RuleStrength
    confidence: float
    domain: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = uuid.uuid4().hex[:12]
        self.rule_id = _sanitize(self.rule_id, 50)
        self.premises = [_sanitize_content(p, 300) for p in (self.premises or [])[:20]]
        self.conclusion = _sanitize_content(self.conclusion, 500)
        self.domain = _sanitize(self.domain, 100)
        self.confidence = _clamp(self.confidence)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "premises": list(self.premises),
            "conclusion": self.conclusion,
            "strength": self.strength.value,
            "confidence": self.confidence,
            "domain": self.domain,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class BridgeMapping:
    mapping_id: str
    neural_pattern_id: str
    symbolic_rule_id: str
    mode: InferenceMode
    bidirectional_strength: float
    neuro_to_symbolic_accuracy: float
    symbolic_to_neuro_accuracy: float
    evidence_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.mapping_id:
            self.mapping_id = uuid.uuid4().hex[:12]
        self.mapping_id = _sanitize(self.mapping_id, 50)
        self.neural_pattern_id = _sanitize(self.neural_pattern_id, 50)
        self.symbolic_rule_id = _sanitize(self.symbolic_rule_id, 50)
        self.bidirectional_strength = _clamp(self.bidirectional_strength)
        self.neuro_to_symbolic_accuracy = _clamp(self.neuro_to_symbolic_accuracy)
        self.symbolic_to_neuro_accuracy = _clamp(self.symbolic_to_neuro_accuracy)
        self.evidence_count = max(0, self.evidence_count)

    def to_dict(self) -> Dict:
        return {
            "mapping_id": self.mapping_id,
            "neural_pattern_id": self.neural_pattern_id,
            "symbolic_rule_id": self.symbolic_rule_id,
            "mode": self.mode.value,
            "bidirectional_strength": self.bidirectional_strength,
            "neuro_to_symbolic_accuracy": self.neuro_to_symbolic_accuracy,
            "symbolic_to_neuro_accuracy": self.symbolic_to_neuro_accuracy,
            "evidence_count": self.evidence_count,
            "created_at": self.created_at,
        }


class NeuralSymbolicBridge:
    """Bridge between neural pattern activation and symbolic reasoning.

    Translates distributed neural patterns into discrete symbols and
    vice versa, enabling hybrid neuro-symbolic cognition.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._patterns: Dict[str, NeuralPattern] = {}
        self._rules: Dict[str, SymbolicRule] = {}
        self._mappings: Dict[str, BridgeMapping] = {}
        self._symbol_index: Dict[str, List[str]] = {}
        self._symbol_index_order: deque = deque(maxlen=_MAX_SYMBOL_INDEX)
        self._pattern_order: deque = deque(maxlen=_MAX_PATTERNS)
        self._rule_order: deque = deque(maxlen=_MAX_RULES)
        self._mapping_order: deque = deque(maxlen=_MAX_MAPPINGS)
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._stats = {
            "patterns_created": 0,
            "rules_created": 0,
            "mappings_created": 0,
            "neuro_to_symbolic_translations": 0,
            "symbolic_to_neuro_translations": 0,
            "bidirectional_translations": 0,
            "avg_mapping_strength": 0.0,
            "total_evidence": 0,
        }

    def add_pattern(self, pattern_type: PatternType, dimensions: Dict[str, float],
                    symbol: Optional[str] = None, confidence: float = 0.5,
                    context: str = "", metadata: Optional[Dict] = None) -> NeuralPattern:
        pattern = NeuralPattern(
            pattern_id=uuid.uuid4().hex[:12],
            pattern_type=pattern_type,
            dimensions=dimensions,
            symbol=symbol,
            confidence=confidence,
            context=context,
            metadata=metadata or {},
        )
        with self._lock:
            if pattern.pattern_id in self._patterns:
                return self._patterns[pattern.pattern_id]
            if len(self._patterns) >= _MAX_PATTERNS:
                evicted_id = self._pattern_order.popleft()
                self._patterns.pop(evicted_id, None)
                if evicted_id in self._symbol_index:
                    for sym_list in self._symbol_index.values():
                        try:
                            sym_list.remove(evicted_id)
                        except ValueError:
                            pass
                logger.warning("Evicted neural pattern %s", evicted_id)
            self._patterns[pattern.pattern_id] = pattern
            self._pattern_order.append(pattern.pattern_id)
            if pattern.symbol:
                if pattern.symbol not in self._symbol_index and len(self._symbol_index) >= _MAX_SYMBOL_INDEX:
                    evicted_sym = self._symbol_index_order.popleft()
                    self._symbol_index.pop(evicted_sym, None)
                is_new = pattern.symbol not in self._symbol_index
                sym_list = self._symbol_index.setdefault(pattern.symbol, [])
                if len(sym_list) >= _MAX_PATTERNS_PER_SYMBOL:
                    sym_list.pop(0)
                sym_list.append(pattern.pattern_id)
                if is_new:
                    self._symbol_index_order.append(pattern.symbol)
            with self._stats_lock:
                self._stats["patterns_created"] += 1
        return copy.deepcopy(pattern)

    def get_pattern(self, pattern_id: str) -> Optional[NeuralPattern]:
        with self._lock:
            p = self._patterns.get(_sanitize(pattern_id, 50))
            return copy.deepcopy(p) if p else None

    def add_rule(self, premises: List[str], conclusion: str,
                strength: RuleStrength = RuleStrength.MODERATE,
                confidence: float = 0.5, domain: str = "",
                metadata: Optional[Dict] = None) -> SymbolicRule:
        rule = SymbolicRule(
            rule_id=uuid.uuid4().hex[:12],
            premises=premises,
            conclusion=conclusion,
            strength=strength,
            confidence=confidence,
            domain=domain,
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._rules) >= _MAX_RULES:
                evicted_id = self._rule_order.popleft()
                self._rules.pop(evicted_id, None)
                logger.warning("Evicted symbolic rule %s", evicted_id)
            self._rules[rule.rule_id] = rule
            self._rule_order.append(rule.rule_id)
            with self._stats_lock:
                self._stats["rules_created"] += 1
        return copy.deepcopy(rule)

    def get_rule(self, rule_id: str) -> Optional[SymbolicRule]:
        with self._lock:
            r = self._rules.get(_sanitize(rule_id, 50))
            return copy.deepcopy(r) if r else None

    def create_mapping(self, neural_pattern_id: str, symbolic_rule_id: str,
                       mode: InferenceMode = InferenceMode.BIDIRECTIONAL,
                       bidirectional_strength: float = 0.5,
                       neuro_to_symbolic_accuracy: float = 0.5,
                       symbolic_to_neuro_accuracy: float = 0.5) -> Optional[BridgeMapping]:
        nid = _sanitize(neural_pattern_id, 50)
        sid = _sanitize(symbolic_rule_id, 50)
        with self._lock:
            if nid not in self._patterns or sid not in self._rules:
                return None
            mapping = BridgeMapping(
                mapping_id=uuid.uuid4().hex[:12],
                neural_pattern_id=nid,
                symbolic_rule_id=sid,
                mode=mode,
                bidirectional_strength=bidirectional_strength,
                neuro_to_symbolic_accuracy=neuro_to_symbolic_accuracy,
                symbolic_to_neuro_accuracy=symbolic_to_neuro_accuracy,
            )
            if len(self._mappings) >= _MAX_MAPPINGS:
                evicted_id = self._mapping_order.popleft()
                self._mappings.pop(evicted_id, None)
                logger.warning("Evicted bridge mapping %s", evicted_id)
            self._mappings[mapping.mapping_id] = mapping
            self._mapping_order.append(mapping.mapping_id)
            with self._stats_lock:
                self._stats["mappings_created"] += 1
        return copy.deepcopy(mapping)

    def _translate_neuro_to_symbolic_unlocked(self, pid: str) -> List[SymbolicRule]:
        results = []
        for m in self._mappings.values():
            if m.neural_pattern_id == pid and m.mode in (InferenceMode.NEURO_TO_SYMBOLIC, InferenceMode.BIDIRECTIONAL):
                rule = self._rules.get(m.symbolic_rule_id)
                if rule and m.bidirectional_strength >= 0.3:
                    results.append((m.neuro_to_symbolic_accuracy, rule))
        results.sort(key=lambda x: x[0], reverse=True)
        with self._stats_lock:
            self._stats["neuro_to_symbolic_translations"] += 1
        return [copy.deepcopy(r) for _, r in results[:10]]

    def _translate_symbolic_to_neuro_unlocked(self, rid: str) -> List[NeuralPattern]:
        results = []
        for m in self._mappings.values():
            if m.symbolic_rule_id == rid and m.mode in (InferenceMode.SYMBOLIC_TO_NEURO, InferenceMode.BIDIRECTIONAL):
                pattern = self._patterns.get(m.neural_pattern_id)
                if pattern and m.bidirectional_strength >= 0.3:
                    results.append((m.symbolic_to_neuro_accuracy, pattern))
        results.sort(key=lambda x: x[0], reverse=True)
        with self._stats_lock:
            self._stats["symbolic_to_neuro_translations"] += 1
        return [copy.deepcopy(p) for _, p in results[:10]]

    def translate_neuro_to_symbolic(self, pattern_id: str) -> List[SymbolicRule]:
        pid = _sanitize(pattern_id, 50)
        with self._lock:
            return self._translate_neuro_to_symbolic_unlocked(pid)

    def translate_symbolic_to_neuro(self, rule_id: str) -> List[NeuralPattern]:
        rid = _sanitize(rule_id, 50)
        with self._lock:
            return self._translate_symbolic_to_neuro_unlocked(rid)

    def translate_bidirectional(self, source_id: str, source_type: str = "pattern") -> Dict:
        sid = _sanitize(source_id, 50)
        with self._lock:
            if source_type == "pattern":
                neuro_results = self._translate_neuro_to_symbolic_unlocked(sid)
                sym_results = []
                for rule in neuro_results[:5]:
                    further = self._translate_symbolic_to_neuro_unlocked(rule.rule_id)
                    sym_results.extend(further[:3])
                with self._stats_lock:
                    self._stats["bidirectional_translations"] += 1
                return {"source_type": "pattern", "symbolic_rules": [r.to_dict() for r in neuro_results], "chain_patterns": [p.to_dict() for p in sym_results[:5]]}
            else:
                sym_results = self._translate_symbolic_to_neuro_unlocked(sid)
                neuro_results = []
                for pattern in sym_results[:5]:
                    further = self._translate_neuro_to_symbolic_unlocked(pattern.pattern_id)
                    neuro_results.extend(further[:3])
                with self._stats_lock:
                    self._stats["bidirectional_translations"] += 1
                return {"source_type": "rule", "neural_patterns": [p.to_dict() for p in sym_results], "chain_rules": [r.to_dict() for r in neuro_results[:5]]}

    def reinforce_mapping(self, mapping_id: str, accuracy_delta: float = 0.01) -> bool:
        mid = _sanitize(mapping_id, 50)
        with self._lock:
            mapping = self._mappings.get(mid)
            if not mapping:
                return False
            mapping.neuro_to_symbolic_accuracy = _clamp(mapping.neuro_to_symbolic_accuracy + accuracy_delta)
            mapping.symbolic_to_neuro_accuracy = _clamp(mapping.symbolic_to_neuro_accuracy + accuracy_delta)
            mapping.bidirectional_strength = _clamp(
                (mapping.neuro_to_symbolic_accuracy + mapping.symbolic_to_neuro_accuracy) / 2
            )
            mapping.evidence_count += 1
            with self._stats_lock:
                self._stats["total_evidence"] += 1
            return True

    def find_patterns_by_symbol(self, symbol: str) -> List[NeuralPattern]:
        sym = _sanitize(symbol, 100)
        with self._lock:
            pids = self._symbol_index.get(sym, [])
            return [copy.deepcopy(self._patterns[pid]) for pid in pids if pid in self._patterns]

    def find_rules_by_domain(self, domain: str) -> List[SymbolicRule]:
        dom = _sanitize(domain, 100)
        with self._lock:
            return [copy.deepcopy(r) for r in self._rules.values() if r.domain == dom]

    def find_rules_by_premise(self, premise: str) -> List[SymbolicRule]:
        pre = _sanitize_content(premise, 300)
        with self._lock:
            results = []
            for r in self._rules.values():
                for p in r.premises:
                    if re.search(r'\b' + re.escape(pre) + r'\b', p, re.IGNORECASE):
                        results.append(r)
                        break
            return [copy.deepcopy(r) for r in results[:20]]

    def compute_pattern_similarity(self, pattern_id_1: str, pattern_id_2: str) -> float:
        pid1 = _sanitize(pattern_id_1, 50)
        pid2 = _sanitize(pattern_id_2, 50)
        with self._lock:
            p1 = self._patterns.get(pid1)
            p2 = self._patterns.get(pid2)
            if not p1 or not p2:
                return 0.0
            all_keys = set(p1.dimensions.keys()) | set(p2.dimensions.keys())
            if not all_keys:
                return 0.0
            dot = sum(p1.dimensions.get(k, 0.0) * p2.dimensions.get(k, 0.0) for k in all_keys)
            mag1 = sum(v ** 2 for v in p1.dimensions.values()) ** 0.5
            mag2 = sum(v ** 2 for v in p2.dimensions.values()) ** 0.5
            if mag1 == 0 or mag2 == 0:
                return 0.0
            return round(dot / (mag1 * mag2), 4)

    def deduct(self, facts: List[str], domain: str = "") -> List[SymbolicRule]:
        clean_facts = [_sanitize_content(f, 300).lower() for f in (facts or [])[:20]]
        dom = _sanitize(domain, 100) if domain else ""
        with self._lock:
            results = []
            for rule in self._rules.values():
                if dom and rule.domain != dom:
                    continue
                matched = sum(1 for p in rule.premises
                              if re.search(r'\b' + re.escape(p.lower()) + r'\b',
                                           " ".join(clean_facts)))
                coverage = matched / len(rule.premises) if rule.premises else 0
                if coverage >= 0.5:
                    results.append((coverage * rule.confidence, rule))
            results.sort(key=lambda x: x[0], reverse=True)
            return [copy.deepcopy(r) for _, r in results[:10]]

    def map_neural_cluster_to_symbol(self, dimensions: Dict[str, float],
                                      pattern_type: PatternType = PatternType.ACTIVATION,
                                      confidence: float = 0.5) -> Tuple[NeuralPattern, Optional[SymbolicRule]]:
        pattern = self.add_pattern(pattern_type, dimensions, confidence=confidence)
        best_rule = None
        best_score = 0.0
        with self._lock:
            for m in self._mappings.values():
                if m.neural_pattern_id == pattern.pattern_id:
                    rule = self._rules.get(m.symbolic_rule_id)
                    if rule and m.bidirectional_strength > best_score:
                        best_score = m.bidirectional_strength
                        best_rule = rule
        return copy.deepcopy(pattern), copy.deepcopy(best_rule) if best_rule else None

    def list_patterns(self, limit: int = 100, offset: int = 0) -> List[NeuralPattern]:
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        with self._lock:
            pids = list(self._pattern_order)[offset:offset + limit]
            return [copy.deepcopy(self._patterns[pid]) for pid in pids if pid in self._patterns]

    def list_rules(self, limit: int = 100, offset: int = 0) -> List[SymbolicRule]:
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        with self._lock:
            rids = list(self._rule_order)[offset:offset + limit]
            return [copy.deepcopy(self._rules[rid]) for rid in rids if rid in self._rules]

    def get_mappings(self, limit: int = 100) -> List[BridgeMapping]:
        limit = max(1, min(500, limit))
        with self._lock:
            mids = list(self._mapping_order)[:limit]
            return [copy.deepcopy(self._mappings[mid]) for mid in mids if mid in self._mappings]

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            total_strength = sum(m.bidirectional_strength for m in self._mappings.values())
            count = len(self._mappings)
            stats["avg_mapping_strength"] = round(total_strength / count, 4) if count else 0.0
            stats["pattern_count"] = len(self._patterns)
            stats["rule_count"] = len(self._rules)
            stats["mapping_count"] = len(self._mappings)
            return stats