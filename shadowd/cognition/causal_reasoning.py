#!/usr/bin/env python3
"""
CausalReasoning -- Causal Inference and Counterfactual Engine.

Discovers and reasons about cause-and-effect relationships. Supports
intervention calculus (do-calculus), counterfactual reasoning, and
causal graph traversal. Three levels of causation: association,
intervention, and counterfactual (Pearl's Ladder).

Key concepts:
  - CausalVariable: a variable in the causal graph
  - CausalEdge: a cause-effect link with strength and evidence
  - InterventionResult: outcome of a do(X=x) intervention
  - CounterfactualResult: "what if" alternative reality estimation
  - CausalPath: a chain of cause-effect relationships

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
from typing import Any, Dict, List, Optional, Set

logger = logging.getLogger("shadowd.cognition.causal_reasoning")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_VARIABLES = 500
_MAX_EDGES = 2000
_MAX_INTERVENTIONS = 1000
_MAX_COUNTERFACTUALS = 1000
_MAX_CONFOUNDERS = 50
_MAX_METADATA_KEYS = 20
_MAX_METADATA_VALUE = 500
_MAX_NAME_LENGTH = 200
_MAX_TYPE_LENGTH = 100


class CausalLevel(str, Enum):
    ASSOCIATION = "association"
    INTERVENTION = "intervention"
    COUNTERFACTUAL = "counterfactual"


class RelationType(str, Enum):
    CAUSES = "causes"
    PREVENTS = "prevents"
    ENABLES = "enables"
    INHIBITS = "inhibits"
    MEDIATES = "mediates"
    CONFOUNDS = "confounds"


class EvidenceStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CONCLUSIVE = "conclusive"


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


@dataclass
class CausalVariable:
    variable_id: str = ""
    name: str = ""
    var_type: str = ""
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.variable_id:
            self.variable_id = "cv_%s" % uuid.uuid4().hex[:12]
        self.variable_id = _sanitize(self.variable_id, 64)
        self.name = _sanitize(self.name, _MAX_NAME_LENGTH)
        self.var_type = _sanitize(self.var_type, _MAX_TYPE_LENGTH)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), _MAX_METADATA_VALUE)
            for k, v in list(self.metadata.items())[:_MAX_METADATA_KEYS]
        }

    def to_dict(self) -> Dict:
        return {
            "variable_id": self.variable_id,
            "name": self.name,
            "var_type": self.var_type,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class CausalEdge:
    edge_id: str = ""
    source_id: str = ""
    target_id: str = ""
    relation_type: RelationType = RelationType.CAUSES
    strength: float = 0.5
    evidence_strength: EvidenceStrength = EvidenceStrength.MODERATE
    confounders: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.edge_id:
            self.edge_id = "ce_%s" % uuid.uuid4().hex[:12]
        self.edge_id = _sanitize(self.edge_id, 64)
        self.source_id = _sanitize(self.source_id, 64)
        self.target_id = _sanitize(self.target_id, 64)
        if isinstance(self.relation_type, str):
            try:
                self.relation_type = RelationType(self.relation_type)
            except ValueError:
                self.relation_type = RelationType.CAUSES
        if isinstance(self.evidence_strength, str):
            try:
                self.evidence_strength = EvidenceStrength(self.evidence_strength)
            except ValueError:
                self.evidence_strength = EvidenceStrength.MODERATE
        self.strength = _clamp(self.strength)
        self.confounders = [_sanitize(c, 64) for c in (self.confounders or [])[:_MAX_CONFOUNDERS]]
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), _MAX_METADATA_VALUE)
            for k, v in list(self.metadata.items())[:_MAX_METADATA_KEYS]
        }

    def to_dict(self) -> Dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "relation_type": self.relation_type.value,
            "strength": round(self.strength, 4),
            "evidence_strength": self.evidence_strength.value,
            "confounders": list(self.confounders),
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class InterventionResult:
    intervention_id: str = ""
    variable: str = ""
    do_value: str = ""
    expected_effects: Dict[str, float] = field(default_factory=dict)
    confidence: float = 0.5
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.intervention_id:
            self.intervention_id = "ir_%s" % uuid.uuid4().hex[:12]
        self.intervention_id = _sanitize(self.intervention_id, 64)
        self.variable = _sanitize(self.variable, _MAX_NAME_LENGTH)
        self.do_value = _sanitize_content(self.do_value, 1000)
        self.expected_effects = {
            _sanitize(str(k), 64): _clamp(float(v))
            for k, v in list(self.expected_effects.items())[:_MAX_METADATA_KEYS]
        }
        self.confidence = _clamp(self.confidence)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), _MAX_METADATA_VALUE)
            for k, v in list(self.metadata.items())[:_MAX_METADATA_KEYS]
        }

    def to_dict(self) -> Dict:
        return {
            "intervention_id": self.intervention_id,
            "variable": self.variable,
            "do_value": self.do_value,
            "expected_effects": {k: round(v, 4) for k, v in self.expected_effects.items()},
            "confidence": round(self.confidence, 4),
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class CounterfactualResult:
    counterfactual_id: str = ""
    premise: str = ""
    observed_outcome: str = ""
    alternative_outcome: str = ""
    probability: float = 0.5
    confidence: float = 0.5

    def __post_init__(self):
        if not self.counterfactual_id:
            self.counterfactual_id = "cf_%s" % uuid.uuid4().hex[:12]
        self.counterfactual_id = _sanitize(self.counterfactual_id, 64)
        self.premise = _sanitize_content(self.premise, 1000)
        self.observed_outcome = _sanitize_content(self.observed_outcome, 500)
        self.alternative_outcome = _sanitize_content(self.alternative_outcome, 500)
        self.probability = _clamp(self.probability)
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "counterfactual_id": self.counterfactual_id,
            "premise": self.premise,
            "observed_outcome": self.observed_outcome,
            "alternative_outcome": self.alternative_outcome,
            "probability": round(self.probability, 4),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class CausalPath:
    path_id: str = ""
    source_id: str = ""
    target_id: str = ""
    nodes: List[str] = field(default_factory=list)
    edges: List[str] = field(default_factory=list)
    total_strength: float = 0.0

    def __post_init__(self):
        if not self.path_id:
            self.path_id = "cp_%s" % uuid.uuid4().hex[:12]
        self.path_id = _sanitize(self.path_id, 64)
        self.source_id = _sanitize(self.source_id, 64)
        self.target_id = _sanitize(self.target_id, 64)
        self.nodes = [_sanitize(n, 64) for n in (self.nodes or [])[:_MAX_EDGES]]
        self.edges = [_sanitize(e, 64) for e in (self.edges or [])[:_MAX_EDGES]]
        self.total_strength = _clamp(self.total_strength)

    def to_dict(self) -> Dict:
        return {
            "path_id": self.path_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "nodes": list(self.nodes),
            "edges": list(self.edges),
            "total_strength": round(self.total_strength, 4),
        }


class CausalReasoning:
    """Causal Inference and Counterfactual Engine.

    Discovers and reasons about cause-and-effect relationships using
    a directed acyclic graph (DAG). Supports intervention calculus,
    counterfactual reasoning, and causal path traversal.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._variables: Dict[str, CausalVariable] = {}
        self._variable_order: deque = deque(maxlen=_MAX_VARIABLES)
        self._edges: Dict[str, CausalEdge] = {}
        self._edge_order: deque = deque(maxlen=_MAX_EDGES)
        self._interventions: Dict[str, InterventionResult] = {}
        self._intervention_order: deque = deque(maxlen=_MAX_INTERVENTIONS)
        self._counterfactuals: Dict[str, CounterfactualResult] = {}
        self._counterfactual_order: deque = deque(maxlen=_MAX_COUNTERFACTUALS)
        self._paths: Dict[str, CausalPath] = {}
        self._stats = {
            "variables_added": 0,
            "causes_added": 0,
            "causes_removed": 0,
            "interventions_computed": 0,
            "counterfactuals_computed": 0,
            "paths_found": 0,
            "confounders_found": 0,
            "mediators_found": 0,
            "validations_performed": 0,
        }

    def add_variable(self, name: str, var_type: str = "",
                     metadata: Optional[Dict] = None) -> CausalVariable:
        name = _sanitize(name, _MAX_NAME_LENGTH)
        var_type = _sanitize(var_type, _MAX_TYPE_LENGTH)
        if metadata is None:
            metadata = {}
        metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), _MAX_METADATA_VALUE)
            for k, v in list(metadata.items())[:_MAX_METADATA_KEYS]
        }
        variable = CausalVariable(
            variable_id="cv_%s" % uuid.uuid4().hex[:12],
            name=name,
            var_type=var_type,
            metadata=metadata,
        )
        with self._lock:
            if len(self._variables) >= _MAX_VARIABLES:
                evict_id = self._variable_order.popleft()
                self._variables.pop(evict_id, None)
                self._evict_edges_for(evict_id)
                logger.warning("Evicting variable %s (cap=%d)", evict_id, _MAX_VARIABLES)
            self._variables[variable.variable_id] = variable
            self._variable_order.append(variable.variable_id)
            with self._stats_lock:
                self._stats["variables_added"] += 1
        return copy.deepcopy(variable)

    def add_cause(self, source_id: str, target_id: str,
                  relation_type: RelationType = RelationType.CAUSES,
                  strength: float = 0.5,
                  evidence: EvidenceStrength = EvidenceStrength.MODERATE,
                  confounders: Optional[List[str]] = None) -> Optional[CausalEdge]:
        source_id = _sanitize(source_id, 64)
        target_id = _sanitize(target_id, 64)
        strength = _clamp(strength)
        if isinstance(relation_type, str):
            try:
                relation_type = RelationType(relation_type)
            except ValueError:
                relation_type = RelationType.CAUSES
        if isinstance(evidence, str):
            try:
                evidence = EvidenceStrength(evidence)
            except ValueError:
                evidence = EvidenceStrength.MODERATE
        if confounders is None:
            confounders = []
        confounders = [_sanitize(c, 64) for c in confounders[:_MAX_CONFOUNDERS]]

        with self._lock:
            if source_id not in self._variables or target_id not in self._variables:
                logger.warning("Cannot add cause: source %s or target %s not found",
                               source_id, target_id)
                return None
            edge = CausalEdge(
                edge_id="ce_%s" % uuid.uuid4().hex[:12],
                source_id=source_id,
                target_id=target_id,
                relation_type=relation_type,
                strength=strength,
                evidence_strength=evidence,
                confounders=confounders,
            )
            if len(self._edges) >= _MAX_EDGES:
                evict_id = self._edge_order.popleft()
                self._edges.pop(evict_id, None)
                logger.warning("Evicting edge %s (cap=%d)", evict_id, _MAX_EDGES)
            self._edges[edge.edge_id] = edge
            self._edge_order.append(edge.edge_id)
            with self._stats_lock:
                self._stats["causes_added"] += 1
            return copy.deepcopy(edge)

    def remove_cause(self, edge_id: str) -> bool:
        eid = _sanitize(edge_id, 64)
        with self._lock:
            if eid not in self._edges:
                return False
            self._edges.pop(eid, None)
            self._edge_order = deque(
                [x for x in self._edge_order if x != eid],
                maxlen=_MAX_EDGES,
            )
            with self._stats_lock:
                self._stats["causes_removed"] += 1
        return True

    def discover_causes(self, effect_id: str,
                        min_strength: float = 0.0) -> List[CausalEdge]:
        effect_id = _sanitize(effect_id, 64)
        min_strength = _clamp(min_strength)
        with self._lock:
            results = []
            for edge in self._edges.values():
                if edge.target_id == effect_id and edge.strength >= min_strength:
                    results.append(copy.deepcopy(edge))
        return results

    def discover_effects(self, cause_id: str,
                        min_strength: float = 0.0) -> List[CausalEdge]:
        cause_id = _sanitize(cause_id, 64)
        min_strength = _clamp(min_strength)
        with self._lock:
            results = []
            for edge in self._edges.values():
                if edge.source_id == cause_id and edge.strength >= min_strength:
                    results.append(copy.deepcopy(edge))
        return results

    def intervene(self, variable_id: str, do_value: str,
                  context: Optional[Dict] = None) -> InterventionResult:
        variable_id = _sanitize(variable_id, 64)
        do_value = _sanitize_content(do_value, 1000)
        if context is None:
            context = {}
        context = {
            _sanitize(str(k), 50): _sanitize_content(str(v), _MAX_METADATA_VALUE)
            for k, v in list(context.items())[:_MAX_METADATA_KEYS]
        }
        with self._lock:
            expected_effects: Dict[str, float] = {}
            confidence = 0.5
            var_edges = [
                e for e in self._edges.values()
                if e.source_id == variable_id
            ]
            for edge in var_edges:
                effect_strength = edge.strength
                if edge.relation_type == RelationType.PREVENTS:
                    effect_strength = -effect_strength
                elif edge.relation_type == RelationType.INHIBITS:
                    effect_strength = -effect_strength * 0.7
                elif edge.relation_type == RelationType.ENABLES:
                    effect_strength = effect_strength * 0.8
                elif edge.relation_type == RelationType.MEDIATES:
                    effect_strength = effect_strength * 0.6
                elif edge.relation_type == RelationType.CONFOUNDS:
                    effect_strength = effect_strength * 0.3
                expected_effects[edge.target_id] = round(abs(effect_strength), 4)
                second_order = [
                    e2 for e2 in self._edges.values()
                    if e2.source_id == edge.target_id
                ]
                for e2 in second_order:
                    if e2.target_id not in expected_effects:
                        indirect = abs(effect_strength) * e2.strength * 0.3
                        expected_effects[e2.target_id] = round(indirect, 4)
            if expected_effects:
                values = list(expected_effects.values())
                confidence = round(sum(min(v, 1.0) for v in values) / len(values), 4)
            result = InterventionResult(
                intervention_id="ir_%s" % uuid.uuid4().hex[:12],
                variable=variable_id,
                do_value=do_value,
                expected_effects=expected_effects,
                confidence=_clamp(confidence),
                metadata=context,
            )
            if len(self._interventions) >= _MAX_INTERVENTIONS:
                evict_id = self._intervention_order.popleft()
                self._interventions.pop(evict_id, None)
                logger.warning("Evicting intervention %s (cap=%d)",
                               evict_id, _MAX_INTERVENTIONS)
            self._interventions[result.intervention_id] = result
            self._intervention_order.append(result.intervention_id)
            with self._stats_lock:
                self._stats["interventions_computed"] += 1
        return copy.deepcopy(result)

    def counterfactual(self, premise: str, observed: str,
                       alternative: str) -> CounterfactualResult:
        premise = _sanitize_content(premise, 1000)
        observed = _sanitize_content(observed, 500)
        alternative = _sanitize_content(alternative, 500)
        with self._lock:
            probability = 0.5
            confidence = 0.3
            premise_words = set(re.findall(r'[a-zA-Z0-9]+', premise.lower()))
            observed_words = set(re.findall(r'[a-zA-Z0-9]+', observed.lower()))
            alternative_words = set(re.findall(r'[a-zA-Z0-9]+', alternative.lower()))
            obs_alt_overlap = len(observed_words & alternative_words) / max(
                len(observed_words | alternative_words), 1
            )
            premise_alt_overlap = len(premise_words & alternative_words) / max(
                len(premise_words | alternative_words), 1
            )
            causally_relevant = []
            for word in alternative_words:
                for edge in self._edges.values():
                    src_var = self._variables.get(edge.source_id)
                    tgt_var = self._variables.get(edge.target_id)
                    if src_var and re.search(r'\b%s\b' % re.escape(word),
                                            src_var.name.lower()):
                        causally_relevant.append(edge.strength)
                    elif tgt_var and re.search(r'\b%s\b' % re.escape(word),
                                               tgt_var.name.lower()):
                        causally_relevant.append(edge.strength * 0.5)
            edge_count = len(self._edges)
            graph_density = min(edge_count / max(len(self._variables), 1), 1.0)
            if causally_relevant:
                avg_causal = sum(causally_relevant) / len(causally_relevant)
                probability = _clamp(0.3 + avg_causal * 0.4 + premise_alt_overlap * 0.2)
                confidence = _clamp(0.4 + len(causally_relevant) * 0.05)
            else:
                probability = _clamp(0.5 + premise_alt_overlap * 0.3 - obs_alt_overlap * 0.2)
                confidence = _clamp(0.2 + premise_alt_overlap * 0.3)
            probability = _clamp(probability)
            confidence = _clamp(confidence * (1.0 - graph_density * 0.2))
            result = CounterfactualResult(
                counterfactual_id="cf_%s" % uuid.uuid4().hex[:12],
                premise=premise,
                observed_outcome=observed,
                alternative_outcome=alternative,
                probability=_clamp(probability),
                confidence=_clamp(confidence),
            )
            if len(self._counterfactuals) >= _MAX_COUNTERFACTUALS:
                evict_id = self._counterfactual_order.popleft()
                self._counterfactuals.pop(evict_id, None)
                logger.warning("Evicting counterfactual %s (cap=%d)",
                               evict_id, _MAX_COUNTERFACTUALS)
            self._counterfactuals[result.counterfactual_id] = result
            self._counterfactual_order.append(result.counterfactual_id)
            with self._stats_lock:
                self._stats["counterfactuals_computed"] += 1
        return copy.deepcopy(result)

    def find_confounders(self, cause_id: str, effect_id: str) -> List[str]:
        cause_id = _sanitize(cause_id, 64)
        effect_id = _sanitize(effect_id, 64)
        with self._lock:
            confounders = set()
            candidates = set()
            for edge in self._edges.values():
                if edge.target_id == cause_id:
                    candidates.add(edge.source_id)
                if edge.target_id == effect_id:
                    candidates.add(edge.source_id)
            for candidate in candidates:
                if candidate != cause_id and candidate != effect_id:
                    confounders.add(candidate)
            for edge in self._edges.values():
                if (edge.source_id == cause_id and edge.target_id == effect_id
                        and edge.relation_type == RelationType.CONFOUNDS):
                    for c in edge.confounders:
                        confounders.add(c)
            with self._stats_lock:
                self._stats["confounders_found"] += len(confounders)
        return sorted(confounders)

    def find_mediators(self, cause_id: str, effect_id: str) -> List[str]:
        cause_id = _sanitize(cause_id, 64)
        effect_id = _sanitize(effect_id, 64)
        with self._lock:
            mediators = set()
            for edge in self._edges.values():
                if (edge.source_id == cause_id and
                        edge.relation_type == RelationType.MEDIATES):
                    mediating = edge.target_id
                    downstream = [
                        e for e in self._edges.values()
                        if e.source_id == mediating
                    ]
                    for ds in downstream:
                        if ds.target_id == effect_id:
                            mediators.add(mediating)
                            break
            for edge in self._edges.values():
                if (edge.source_id == cause_id and edge.target_id != effect_id):
                    bridging = [
                        e for e in self._edges.values()
                        if e.source_id == edge.target_id and e.target_id == effect_id
                    ]
                    if bridging:
                        mediators.add(edge.target_id)
            with self._stats_lock:
                self._stats["mediators_found"] += len(mediators)
        return sorted(mediators)

    def find_causal_paths(self, source_id: str, target_id: str,
                          max_depth: int = 10) -> List[CausalPath]:
        source_id = _sanitize(source_id, 64)
        target_id = _sanitize(target_id, 64)
        max_depth = max(1, min(50, max_depth))
        with self._lock:
            paths: List[CausalPath] = []
            adjacency: Dict[str, List[CausalEdge]] = {}
            for edge in self._edges.values():
                if edge.source_id not in adjacency:
                    adjacency[edge.source_id] = []
                adjacency[edge.source_id].append(edge)
            stack: List[tuple] = [(source_id, [source_id], [], 1.0)]
            visited_paths: Set[str] = set()
            while stack:
                current, node_path, edge_path, strength = stack.pop(0)
                if current == target_id and len(node_path) > 1:
                    path_key = "->".join(node_path)
                    if path_key not in visited_paths:
                        visited_paths.add(path_key)
                        path = CausalPath(
                            path_id="cp_%s" % uuid.uuid4().hex[:12],
                            source_id=source_id,
                            target_id=target_id,
                            nodes=list(node_path),
                            edges=list(edge_path),
                            total_strength=_clamp(strength),
                        )
                        paths.append(path)
                    continue
                if len(node_path) > max_depth:
                    continue
                if current in adjacency:
                    for edge in adjacency[current]:
                        if edge.target_id not in node_path:
                            new_strength = strength * edge.strength
                            if edge.relation_type in (RelationType.PREVENTS,
                                                       RelationType.INHIBITS):
                                new_strength *= -1
                            stack.append((
                                edge.target_id,
                                node_path + [edge.target_id],
                                edge_path + [edge.edge_id],
                                abs(new_strength),
                            ))
            for path in paths:
                self._paths[path.path_id] = path
            with self._stats_lock:
                self._stats["paths_found"] += len(paths)
        return [copy.deepcopy(p) for p in paths]

    def validate_cause(self, source_id: str, target_id: str) -> bool:
        source_id = _sanitize(source_id, 64)
        target_id = _sanitize(target_id, 64)
        with self._lock:
            with self._stats_lock:
                self._stats["validations_performed"] += 1
            if source_id == target_id:
                return False
            visited: Set[str] = set()
            stack = [target_id]
            while stack:
                current = stack.pop()
                if current == source_id:
                    return False
                if current in visited:
                    continue
                visited.add(current)
                for edge in self._edges.values():
                    if edge.source_id == current:
                        stack.append(edge.target_id)
            return True

    def get_stats(self) -> Dict:
        with self._lock:
            var_count = len(self._variables)
            edge_count = len(self._edges)
            intervention_count = len(self._interventions)
            counterfactual_count = len(self._counterfactuals)
            avg_strength = 0.0
            if edge_count > 0:
                avg_strength = round(
                    sum(e.strength for e in self._edges.values()) / edge_count, 4
                )
            with self._stats_lock:
                stats = dict(self._stats)
        stats["variables"] = var_count
        stats["edges"] = edge_count
        stats["interventions"] = intervention_count
        stats["counterfactuals"] = counterfactual_count
        stats["avg_edge_strength"] = avg_strength
        return stats

    def _evict_edges_for(self, variable_id: str) -> None:
        to_remove = [
            eid for eid, edge in self._edges.items()
            if edge.source_id == variable_id or edge.target_id == variable_id
        ]
        for eid in to_remove:
            self._edges.pop(eid, None)
        if to_remove:
            self._edge_order = deque(
                [x for x in self._edge_order if x not in set(to_remove)],
                maxlen=_MAX_EDGES,
            )
            logger.warning("Evicting %d edges for variable %s",
                           len(to_remove), variable_id)