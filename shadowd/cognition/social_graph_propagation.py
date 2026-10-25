#!/usr/bin/env python3
"""
SocialGraphPropagation — Influence Ripple Across Social Modules.

When a social event occurs (new interaction, conflict escalation, group change),
SocialGraphPropagation routes influence ripples across three social-cognitive modules:

  SocialIntelligence  →  EmpathyMapping  →  ConflictResolution
       ↑                      |                    |
       └──────────────────────┘────────────────────┘

Key concepts:
  - InfluenceRipple: a weighted signal that propagates from one module to others
  - PropagationRule: defines which events trigger ripples and their target modules
  - RippleEffect: the measured outcome of a propagated ripple on a target module
  - SocialGraph: unified view merging group membership, empathy scores, and conflicts

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

logger = logging.getLogger("shadowd.cognition.social_graph_propagation")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_RIPPLES = 2000
_MAX_EFFECTS = 3000
_MAX_RULES = 100
_MAX_GRAPH_NODES = 500
_MAX_GRAPH_EDGES = 2000
_MAX_METADATA_KEYS = 20
_MAX_REASON_LENGTH = 500


class PropagationSource(str, Enum):
    SOCIAL_INTELLIGENCE = "social_intelligence"
    EMPATHY_MAPPING = "empathy_mapping"
    CONFLICT_RESOLUTION = "conflict_resolution"


class RippleType(str, Enum):
    INTERACTION_RECORDED = "interaction_recorded"
    GROUP_MEMBERSHIP_CHANGED = "group_membership_changed"
    INFLUENCE_ADDED = "influence_added"
    ROLE_CHANGED = "role_changed"
    COHESION_SHIFT = "cohesion_shift"
    TRIGGER_DETECTED = "trigger_detected"
    SENTIMENT_SHIFT = "sentiment_shift"
    CONFLICT_CREATED = "conflict_created"
    CONFLICT_ESCALATED = "conflict_escalated"
    CONFLICT_RESOLVED = "conflict_resolved"
    PERSPECTIVE_ADDED = "perspective_added"
    STRATEGY_SUGGESTED = "strategy_suggested"
    EMPATHY_SCORE_CHANGED = "empathy_score_changed"
    RELATIONSHIP_HEALTH_SHIFT = "relationship_health_shift"


class RippleStrength(str, Enum):
    WEAK = "weak"
    MODERATE = "moderate"
    STRONG = "strong"
    CRITICAL = "critical"


class GraphNodeType(str, Enum):
    PERSON = "person"
    GROUP = "group"
    CONFLICT = "conflict"


@dataclass
class PropagationRule:
    rule_id: str
    source: PropagationSource
    ripple_type: RippleType
    target: PropagationSource
    weight: float
    strength_threshold: float
    enabled: bool = True

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "source": self.source.value,
            "ripple_type": self.ripple_type.value,
            "target": self.target.value,
            "weight": round(max(0.0, min(2.0, self.weight)), 3),
            "strength_threshold": round(max(0.0, min(1.0, self.strength_threshold)), 3),
            "enabled": self.enabled,
        }


@dataclass
class InfluenceRipple:
    ripple_id: str
    source: PropagationSource
    ripple_type: RippleType
    strength: float
    person_id: Optional[str]
    group_id: Optional[str]
    conflict_id: Optional[str]
    reason: str
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    propagated: bool = False

    def to_dict(self) -> Dict:
        return {
            "ripple_id": self.ripple_id,
            "source": self.source.value,
            "ripple_type": self.ripple_type.value,
            "strength": round(max(0.0, min(1.0, self.strength)), 3),
            "person_id": self.person_id,
            "group_id": self.group_id,
            "conflict_id": self.conflict_id,
            "reason": self.reason[:_MAX_REASON_LENGTH],
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])),
            "created_at": self.created_at,
            "propagated": self.propagated,
        }


@dataclass
class RippleEffect:
    effect_id: str
    ripple_id: str
    target_module: PropagationSource
    action_taken: str
    impact_score: float
    notes: str
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "effect_id": self.effect_id,
            "ripple_id": self.ripple_id,
            "target_module": self.target_module.value,
            "action_taken": self.action_taken[:200],
            "impact_score": round(max(0.0, min(1.0, self.impact_score)), 3),
            "notes": self.notes[:300],
            "created_at": self.created_at,
        }


@dataclass
class SocialGraphNode:
    node_id: str
    node_type: GraphNodeType
    label: str
    empathy_score: float = 0.0
    influence_score: float = 0.0
    conflict_count: int = 0
    group_count: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "node_id": self.node_id,
            "node_type": self.node_type.value,
            "label": self.label[:100],
            "empathy_score": round(max(0.0, min(100.0, self.empathy_score)), 2),
            "influence_score": round(max(0.0, min(1.0, self.influence_score)), 3),
            "conflict_count": self.conflict_count,
            "group_count": self.group_count,
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])),
        }


@dataclass
class SocialGraphEdge:
    edge_id: str
    source_id: str
    target_id: str
    edge_type: str
    weight: float
    bidirectional: bool = False

    def to_dict(self) -> Dict:
        return {
            "edge_id": self.edge_id,
            "source_id": self.source_id,
            "target_id": self.target_id,
            "edge_type": self.edge_type[:50],
            "weight": round(max(0.0, min(1.0, self.weight)), 3),
            "bidirectional": self.bidirectional,
        }


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_len]


def _sanitize_content(text: str, max_len: int = _MAX_REASON_LENGTH) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_len]


class SocialGraphPropagation:
    """
    Influence ripple engine across SocialIntelligence, EmpathyMapping, ConflictResolution.

    When a social event occurs in one module, this engine:
    1. Creates an InfluenceRipple describing the event
    2. Finds matching PropagationRules
    3. Propagates the ripple to target modules
    4. Records the RippleEffect (what happened in the target)
    5. Maintains a unified SocialGraph merging data from all three modules
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._ripples: deque = deque(maxlen=_MAX_RIPPLES)
        self._effects: deque = deque(maxlen=_MAX_EFFECTS)
        self._rules: Dict[str, PropagationRule] = {}
        self._rule_order: List[str] = []
        self._graph_nodes: Dict[str, SocialGraphNode] = {}
        self._graph_edges: Dict[str, SocialGraphEdge] = {}
        self._stats = {
            "ripples_created": 0,
            "ripples_propagated": 0,
            "effects_recorded": 0,
            "rules_active": 0,
            "graph_nodes": 0,
            "graph_edges": 0,
        }
        self._init_default_rules()

    def _init_default_rules(self):
        defaults = [
            ("r01", PropagationSource.SOCIAL_INTELLIGENCE, RippleType.INTERACTION_RECORDED, PropagationSource.EMPATHY_MAPPING, 0.8, 0.2),
            ("r02", PropagationSource.SOCIAL_INTELLIGENCE, RippleType.GROUP_MEMBERSHIP_CHANGED, PropagationSource.EMPATHY_MAPPING, 0.6, 0.3),
            ("r03", PropagationSource.SOCIAL_INTELLIGENCE, RippleType.INFLUENCE_ADDED, PropagationSource.EMPATHY_MAPPING, 0.7, 0.2),
            ("r04", PropagationSource.SOCIAL_INTELLIGENCE, RippleType.COHESION_SHIFT, PropagationSource.CONFLICT_RESOLUTION, 0.5, 0.4),
            ("r05", PropagationSource.SOCIAL_INTELLIGENCE, RippleType.ROLE_CHANGED, PropagationSource.EMPATHY_MAPPING, 0.5, 0.3),
            ("r06", PropagationSource.EMPATHY_MAPPING, RippleType.TRIGGER_DETECTED, PropagationSource.CONFLICT_RESOLUTION, 0.7, 0.3),
            ("r07", PropagationSource.EMPATHY_MAPPING, RippleType.SENTIMENT_SHIFT, PropagationSource.SOCIAL_INTELLIGENCE, 0.6, 0.3),
            ("r08", PropagationSource.EMPATHY_MAPPING, RippleType.RELATIONSHIP_HEALTH_SHIFT, PropagationSource.CONFLICT_RESOLUTION, 0.6, 0.4),
            ("r09", PropagationSource.EMPATHY_MAPPING, RippleType.EMPATHY_SCORE_CHANGED, PropagationSource.SOCIAL_INTELLIGENCE, 0.4, 0.5),
            ("r10", PropagationSource.CONFLICT_RESOLUTION, RippleType.CONFLICT_CREATED, PropagationSource.EMPATHY_MAPPING, 0.8, 0.2),
            ("r11", PropagationSource.CONFLICT_RESOLUTION, RippleType.CONFLICT_ESCALATED, PropagationSource.EMPATHY_MAPPING, 0.9, 0.2),
            ("r12", PropagationSource.CONFLICT_RESOLUTION, RippleType.CONFLICT_ESCALATED, PropagationSource.SOCIAL_INTELLIGENCE, 0.7, 0.3),
            ("r13", PropagationSource.CONFLICT_RESOLUTION, RippleType.CONFLICT_RESOLVED, PropagationSource.EMPATHY_MAPPING, 0.5, 0.2),
            ("r14", PropagationSource.CONFLICT_RESOLUTION, RippleType.PERSPECTIVE_ADDED, PropagationSource.EMPATHY_MAPPING, 0.6, 0.3),
            ("r15", PropagationSource.CONFLICT_RESOLUTION, RippleType.STRATEGY_SUGGESTED, PropagationSource.SOCIAL_INTELLIGENCE, 0.4, 0.4),
        ]
        for rid, src, rtype, tgt, weight, threshold in defaults:
            rule = PropagationRule(
                rule_id=rid,
                source=src,
                ripple_type=rtype,
                target=tgt,
                weight=weight,
                strength_threshold=threshold,
            )
            self._rules[rid] = rule
            self._rule_order.append(rid)
        with self._stats_lock:
            self._stats["rules_active"] = len(defaults)

    def create_ripple(
        self,
        source: str,
        ripple_type: str,
        strength: float,
        reason: str,
        person_id: Optional[str] = None,
        group_id: Optional[str] = None,
        conflict_id: Optional[str] = None,
        metadata: Optional[Dict] = None,
    ) -> InfluenceRipple:
        try:
            src = PropagationSource(source)
        except ValueError:
            raise ValueError("invalid source: %s" % source[:50])
        try:
            rt = RippleType(ripple_type)
        except ValueError:
            raise ValueError("invalid ripple_type: %s" % ripple_type[:50])

        strength = max(0.0, min(1.0, strength))
        reason = _sanitize_content(reason)
        if person_id:
            person_id = _sanitize(person_id, 64)
        if group_id:
            group_id = _sanitize(group_id, 64)
        if conflict_id:
            conflict_id = _sanitize(conflict_id, 64)
        if metadata and len(metadata) > _MAX_METADATA_KEYS:
            metadata = dict(list(metadata.items())[:_MAX_METADATA_KEYS])
        if metadata:
            metadata = {str(k)[:50]: _sanitize_content(str(v), 200) for k, v in metadata.items()}

        ripple = InfluenceRipple(
            ripple_id=self._next_id("rip_"),
            source=src,
            ripple_type=rt,
            strength=strength,
            person_id=person_id,
            group_id=group_id,
            conflict_id=conflict_id,
            reason=reason,
            metadata=metadata or {},
        )

        with self._lock:
            self._ripples.append(ripple)
            with self._stats_lock:
                self._stats["ripples_created"] += 1

        return copy.deepcopy(ripple)

    def propagate(
        self,
        ripple: InfluenceRipple,
        social_intel=None,
        empathy=None,
        conflict=None,
    ) -> List[RippleEffect]:
        matching_rules = []
        with self._lock:
            for rule in self._rules.values():
                if not rule.enabled:
                    continue
                if rule.source == ripple.source and rule.ripple_type == ripple.ripple_type:
                    if ripple.strength >= rule.strength_threshold:
                        matching_rules.append(rule)

        effects = []
        for rule in matching_rules:
            effect = self._apply_rule(rule, ripple, social_intel, empathy, conflict)
            if effect is not None:
                effects.append(effect)

        with self._lock:
            ripple.propagated = True
            for stored in self._ripples:
                if stored.ripple_id == ripple.ripple_id:
                    stored.propagated = True
                    break
            with self._stats_lock:
                self._stats["ripples_propagated"] += 1

        return [copy.deepcopy(e) for e in effects]

    def _apply_rule(
        self,
        rule: PropagationRule,
        ripple: InfluenceRipple,
        social_intel=None,
        empathy=None,
        conflict=None,
    ) -> Optional[RippleEffect]:
        weighted_strength = ripple.strength * rule.weight
        action_taken = ""
        notes = ""
        target = rule.target

        try:
            if target == PropagationSource.EMPATHY_MAPPING and empathy is not None:
                action_taken, notes = self._propagate_to_empathy(ripple, weighted_strength, empathy)
            elif target == PropagationSource.CONFLICT_RESOLUTION and conflict is not None:
                action_taken, notes = self._propagate_to_conflict(ripple, weighted_strength, conflict)
            elif target == PropagationSource.SOCIAL_INTELLIGENCE and social_intel is not None:
                action_taken, notes = self._propagate_to_social_intel(ripple, weighted_strength, social_intel)
            else:
                return None
        except (ValueError, TypeError, AttributeError, KeyError) as e:
            logger.debug("Propagation error: %s → %s: %s", rule.source.value, target.value, type(e).__name__)
            return None

        if not action_taken:
            return None

        effect = RippleEffect(
            effect_id=self._next_id("eff_"),
            ripple_id=ripple.ripple_id,
            target_module=target,
            action_taken=action_taken,
            impact_score=weighted_strength,
            notes=notes[:300],
        )

        with self._lock:
            self._effects.append(effect)
            with self._stats_lock:
                self._stats["effects_recorded"] += 1

        return effect

    def _propagate_to_empathy(self, ripple: InfluenceRipple, strength: float, empathy) -> Tuple[str, str]:
        action_parts = []
        if ripple.person_id:
            try:
                profile = empathy.get_person_profile(ripple.person_id)
                if profile is not None:
                    action_parts.append("noted_person")
                    if ripple.ripple_type == RippleType.CONFLICT_ESCALATED:
                        action_parts.append("conflict_stress_flag")
                    elif ripple.ripple_type == RippleType.CONFLICT_CREATED:
                        action_parts.append("new_conflict_awareness")
            except KeyError:
                action_parts.append("person_not_found")

        if ripple.ripple_type in (RippleType.GROUP_MEMBERSHIP_CHANGED, RippleType.ROLE_CHANGED) and ripple.person_id:
            action_parts.append("social_context_update")

        if ripple.ripple_type == RippleType.CONFLICT_RESOLVED and ripple.person_id:
            action_parts.append("tension_released")

        action = "+".join(action_parts) if action_parts else "observed"
        notes = "strength=%.2f person=%s" % (strength, ripple.person_id or "none")
        return action, notes

    def _propagate_to_conflict(self, ripple: InfluenceRipple, strength: float, conflict) -> Tuple[str, str]:
        action_parts = []
        if ripple.ripple_type == RippleType.COHESION_SHIFT and ripple.group_id:
            action_parts.append("group_instability_check")
            if strength > 0.6:
                action_parts.append("high_risk_alert")

        if ripple.ripple_type == RippleType.TRIGGER_DETECTED and ripple.person_id:
            action_parts.append("sensitivity_noted")
            if strength > 0.5:
                action_parts.append("escalation_risk")

        if ripple.ripple_type == RippleType.RELATIONSHIP_HEALTH_SHIFT:
            action_parts.append("relationship_stress")
            if strength > 0.6:
                action_parts.append("intervention_candidate")

        action = "+".join(action_parts) if action_parts else "observed"
        notes = "strength=%.2f type=%s" % (strength, ripple.ripple_type.value)
        return action, notes

    def _propagate_to_social_intel(self, ripple: InfluenceRipple, strength: float, social_intel) -> Tuple[str, str]:
        action_parts = []
        if ripple.ripple_type == RippleType.SENTIMENT_SHIFT and ripple.person_id:
            action_parts.append("sentiment_propagation")
            if strength > 0.5:
                action_parts.append("group_mood_impact")

        if ripple.ripple_type == RippleType.EMPATHY_SCORE_CHANGED and ripple.person_id:
            action_parts.append("influence_recalibration")

        if ripple.ripple_type == RippleType.CONFLICT_ESCALATED:
            action_parts.append("social_tension_alert")

        if ripple.ripple_type == RippleType.STRATEGY_SUGGESTED:
            action_parts.append("mediation_pattern_learned")

        action = "+".join(action_parts) if action_parts else "observed"
        notes = "strength=%.2f source=%s" % (strength, ripple.source.value)
        return action, notes

    def build_graph(
        self,
        social_intel=None,
        empathy=None,
        conflict=None,
    ) -> Dict:
        nodes: Dict[str, SocialGraphNode] = {}
        edges: Dict[str, SocialGraphEdge] = {}

        if empathy is not None:
            try:
                empathy_graph = empathy.get_social_graph()
                for n in empathy_graph.get("nodes", []):
                    nid = _sanitize(n.get("id", ""), 64)
                    if not nid:
                        continue
                    nodes[nid] = SocialGraphNode(
                        node_id=nid,
                        node_type=GraphNodeType.PERSON,
                        label=n.get("name", "unknown")[:100],
                        empathy_score=n.get("empathy_score", 0.0),
                        metadata={"source": "empathy"},
                    )
                for e in empathy_graph.get("edges", []):
                    src = _sanitize(e.get("source", ""), 64)
                    tgt = _sanitize(e.get("target", ""), 64)
                    if src and tgt:
                        eid = "emp_%s_%s" % (src[:12], tgt[:12])
                        edges[eid] = SocialGraphEdge(
                            edge_id=eid,
                            source_id=src,
                            target_id=tgt,
                            edge_type="empathy_affinity",
                            weight=e.get("affinity", 0.3),
                            bidirectional=True,
                        )
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug("Empathy graph build error: %s", type(e).__name__)

        if social_intel is not None:
            try:
                groups = social_intel.list_groups()
                for g in groups[:_MAX_GRAPH_NODES]:
                    gid = _sanitize(g.group_id, 64)
                    if not gid:
                        continue
                    nodes[gid] = SocialGraphNode(
                        node_id=gid,
                        node_type=GraphNodeType.GROUP,
                        label=g.name[:100],
                        influence_score=g.cohesion_score,
                        group_count=1,
                        metadata={"source": "social_intel", "group_type": g.group_type.value},
                    )
                    for member in g.members[:50]:
                        mid = _sanitize(member, 64)
                        if mid in nodes:
                            nodes[mid].group_count += 1
                        eid = "grp_%s_%s" % (mid[:12], gid[:12])
                        edges[eid] = SocialGraphEdge(
                            edge_id=eid,
                            source_id=mid,
                            target_id=gid,
                            edge_type="group_member",
                            weight=0.5,
                        )
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug("Social intel graph build error: %s", type(e).__name__)

        if conflict is not None:
            try:
                conflicts = conflict.list_conflicts()
                for c in conflicts[:100]:
                    cid = _sanitize(c.conflict_id, 64)
                    if not cid:
                        continue
                    nodes[cid] = SocialGraphNode(
                        node_id=cid,
                        node_type=GraphNodeType.CONFLICT,
                        label=c.title[:100],
                        conflict_count=1,
                        metadata={
                            "source": "conflict_resolution",
                            "conflict_type": c.conflict_type.value,
                            "intensity": c.intensity.value,
                            "status": c.status.value,
                        },
                    )
                    for p in c.perspectives[:10]:
                        pid = _sanitize(p.party, 64)
                        if pid in nodes:
                            nodes[pid].conflict_count += 1
                            eid = "conf_%s_%s" % (pid[:12], cid[:12])
                            edges[eid] = SocialGraphEdge(
                                edge_id=eid,
                                source_id=pid,
                                target_id=cid,
                                edge_type="conflict_party",
                                weight=0.7,
                            )
            except (ValueError, TypeError, AttributeError) as e:
                logger.debug("Conflict graph build error: %s", type(e).__name__)

        with self._lock:
            self._graph_nodes = {k: copy.deepcopy(v) for k, v in list(nodes.items())[:_MAX_GRAPH_NODES]}
            self._graph_edges = {k: copy.deepcopy(v) for k, v in list(edges.items())[:_MAX_GRAPH_EDGES]}
            with self._stats_lock:
                self._stats["graph_nodes"] = len(self._graph_nodes)
                self._stats["graph_edges"] = len(self._graph_edges)

        return {
            "nodes": [n.to_dict() for n in self._graph_nodes.values()],
            "edges": [e.to_dict() for e in self._graph_edges.values()],
            "total_nodes": len(self._graph_nodes),
            "total_edges": len(self._graph_edges),
        }

    def _next_id(self, prefix: str) -> str:
        return "%s%s" % (prefix, uuid.uuid4().hex[:12])

    def add_rule(
        self,
        source: str,
        ripple_type: str,
        target: str,
        weight: float = 0.5,
        strength_threshold: float = 0.3,
    ) -> PropagationRule:
        try:
            src = PropagationSource(source)
        except ValueError:
            raise ValueError("invalid source: %s" % source[:50])
        try:
            rt = RippleType(ripple_type)
        except ValueError:
            raise ValueError("invalid ripple_type: %s" % ripple_type[:50])
        try:
            tgt = PropagationSource(target)
        except ValueError:
            raise ValueError("invalid target: %s" % target[:50])

        weight = max(0.0, min(2.0, weight))
        strength_threshold = max(0.0, min(1.0, strength_threshold))

        with self._lock:
            if len(self._rules) >= _MAX_RULES:
                evict_id = self._rule_order.pop(0)
                evicted = self._rules.pop(evict_id, None)
                if evicted:
                    logger.warning("Evicting rule %s (cap=%d)", evict_id, _MAX_RULES)
            rule_id = self._next_id("r_")
            rule = PropagationRule(
                rule_id=rule_id,
                source=src,
                ripple_type=rt,
                target=tgt,
                weight=weight,
                strength_threshold=strength_threshold,
            )
            self._rules[rule_id] = rule
            self._rule_order.append(rule_id)
            with self._stats_lock:
                self._stats["rules_active"] = sum(1 for r in self._rules.values() if r.enabled)

        return copy.deepcopy(rule)

    def toggle_rule(self, rule_id: str, enabled: bool) -> bool:
        rule_id = _sanitize(rule_id, 32)
        with self._lock:
            rule = self._rules.get(rule_id)
            if rule is None:
                return False
            rule.enabled = enabled
            with self._stats_lock:
                self._stats["rules_active"] = sum(1 for r in self._rules.values() if r.enabled)
            return True

    def get_ripples(self, limit: int = 50, source: Optional[str] = None) -> List[InfluenceRipple]:
        limit = max(1, min(500, limit))
        with self._lock:
            result = list(self._ripples)
        if source:
            try:
                src = PropagationSource(source)
                result = [r for r in result if r.source == src]
            except ValueError:
                pass
        return [copy.deepcopy(r) for r in result[-limit:]]

    def get_effects(self, limit: int = 50, target: Optional[str] = None) -> List[RippleEffect]:
        limit = max(1, min(500, limit))
        with self._lock:
            result = list(self._effects)
        if target:
            try:
                tgt = PropagationSource(target)
                result = [e for e in result if e.target_module == tgt]
            except ValueError:
                pass
        return [copy.deepcopy(e) for e in result[-limit:]]

    def get_rules(self) -> List[PropagationRule]:
        with self._lock:
            return [copy.deepcopy(r) for r in self._rules.values()]

    def get_graph(self) -> Dict:
        with self._lock:
            return {
                "nodes": [n.to_dict() for n in self._graph_nodes.values()],
                "edges": [e.to_dict() for e in self._graph_edges.values()],
                "total_nodes": len(self._graph_nodes),
                "total_edges": len(self._graph_edges),
            }

    def get_propagation_summary(self) -> Dict:
        with self._lock:
            ripples = list(self._ripples)
            effects = list(self._effects)
        total = len(ripples)
        propagated = sum(1 for r in ripples if r.propagated)
        by_source = {}
        for r in ripples:
            key = r.source.value
            by_source[key] = by_source.get(key, 0) + 1
        by_type = {}
        for e in effects:
            key = e.target_module.value
            by_type[key] = by_type.get(key, 0) + 1
        return {
            "total_ripples": total,
            "propagated_ripples": propagated,
            "propagation_rate": round(propagated / max(total, 1), 3),
            "ripples_by_source": by_source,
            "effects_by_target": by_type,
            "total_effects": len(effects),
        }

    def get_stats(self) -> Dict:
        with self._stats_lock:
            return dict(self._stats)