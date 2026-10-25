#!/usr/bin/env python3
"""
SocialIntelligence - Social Dynamics and Influence Mapping Engine

Analyzes group dynamics, social influence patterns, communication networks,
and interpersonal power structures.

Architecture:
  - SocialGroup: defines a social group with members and cohesion
  - InfluenceRelation: directed influence between two people
  - SocialRole: a person's role within a group
  - CommunicationPattern: detected communication patterns in groups
  - SocialIntelligence: main engine managing groups, influence, roles, patterns
  - Thread-safe, bounded history, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.social_intelligence")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


class GroupType(str, Enum):
    FORMAL = "formal"
    INFORMAL = "informal"
    PROJECT = "project"
    SOCIAL = "social"
    FAMILY = "family"
    PROFESSIONAL = "professional"
    COMMUNITY = "community"


class InfluenceType(str, Enum):
    MENTOR = "mentor"
    PEER = "peer"
    AUTHORITY = "authority"
    EXPERT = "expert"
    SOCIAL = "social"
    INFORMATIONAL = "informational"


class RoleType(str, Enum):
    LEADER = "leader"
    FACILITATOR = "facilitator"
    CONTRIBUTOR = "contributor"
    OBSERVER = "observer"
    BRIDGE = "bridge"
    GATEKEEPER = "gatekeeper"


@dataclass
class SocialGroup:
    group_id: str
    name: str
    description: str
    members: List[str] = field(default_factory=list)
    group_type: GroupType = GroupType.INFORMAL
    cohesion_score: float = 0.5
    formed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict = field(default_factory=dict)

    def to_dict(self) -> Dict:
        return {
            "group_id": self.group_id,
            "name": self.name[:200],
            "description": self.description[:1000],
            "members": list(self.members[:50]),
            "group_type": self.group_type.value,
            "cohesion_score": round(max(0.0, min(1.0, self.cohesion_score)), 3),
            "formed_at": self.formed_at,
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:20])),
        }


@dataclass
class InfluenceRelation:
    relation_id: str
    source: str
    target: str
    influence_type: InfluenceType = InfluenceType.PEER
    strength: float = 0.5
    context: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "relation_id": self.relation_id,
            "source": self.source,
            "target": self.target,
            "influence_type": self.influence_type.value,
            "strength": round(max(0.0, min(1.0, self.strength)), 3),
            "context": self.context[:500],
            "created_at": self.created_at,
        }


@dataclass
class SocialRole:
    role_id: str
    person_id: str
    group_id: str
    role_type: RoleType = RoleType.CONTRIBUTOR
    influence_level: float = 0.5
    assumed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "role_id": self.role_id,
            "person_id": self.person_id,
            "group_id": self.group_id,
            "role_type": self.role_type.value,
            "influence_level": round(max(0.0, min(1.0, self.influence_level)), 3),
            "assumed_at": self.assumed_at,
        }


@dataclass
class CommunicationPattern:
    pattern_id: str
    participants: List[str] = field(default_factory=list)
    frequency: int = 1
    pattern_type: str = "information_flow"
    avg_sentiment: float = 0.5
    detected_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "participants": list(self.participants[:20]),
            "frequency": self.frequency,
            "pattern_type": self.pattern_type[:50],
            "avg_sentiment": round(max(-1.0, min(1.0, self.avg_sentiment)), 3),
            "detected_at": self.detected_at,
        }


class SocialIntelligence:
    _MAX_GROUPS = 500
    _MAX_INFLUENCE = 2000
    _MAX_ROLES = 1000
    _MAX_PATTERNS = 500
    _MAX_METADATA_KEYS = 20
    _MAX_MEMBERS = 50

    def __init__(self):
        self._groups: Dict[str, SocialGroup] = {}
        self._influences: Dict[str, InfluenceRelation] = {}
        self._roles: Dict[str, SocialRole] = {}
        self._patterns: Dict[str, CommunicationPattern] = {}
        self._group_order: List[str] = []
        self._influence_order: List[str] = []
        self._role_order: List[str] = []
        self._pattern_order: List[str] = []
        self._lock = threading.Lock()
        self._stats = {
            "groups_tracked": 0,
            "influence_relations": 0,
            "roles_assigned": 0,
            "patterns_detected": 0,
            "total_members": 0,
        }
        self._stats_lock = threading.Lock()

    def create_group(self, name: str, group_type: GroupType, description: str = "", members: Optional[List[str]] = None) -> SocialGroup:
        name = _sanitize(name, 200)
        description = _sanitize_content(description, 1000)
        members = (members or [])[:self._MAX_MEMBERS]
        members = [_sanitize(m, 64) for m in members]
        group_id = f"grp_{uuid.uuid4().hex[:12]}"
        with self._lock:
            if len(self._groups) >= self._MAX_GROUPS:
                evict_id = self._group_order.pop(0)
                evicted = self._groups.pop(evict_id, None)
                if evicted:
                    logger.warning("Evicting group %s (cap=%d)", evict_id, self._MAX_GROUPS)
                    with self._stats_lock:
                        self._stats["total_members"] -= len(evicted.members)
                    roles_to_remove = [rid for rid, r in self._roles.items() if r.group_id == evict_id]
                    for rid in roles_to_remove:
                        self._roles.pop(rid, None)
                        self._role_order.remove(rid) if rid in self._role_order else None
                        with self._stats_lock:
                            self._stats["roles_assigned"] = max(0, self._stats["roles_assigned"] - 1)
            group = SocialGroup(
                group_id=group_id,
                name=name,
                description=description,
                members=members,
                group_type=group_type,
            )
            self._groups[group_id] = group
            self._group_order.append(group_id)
            with self._stats_lock:
                self._stats["groups_tracked"] = len(self._groups)
                self._stats["total_members"] += len(members)
            return copy.deepcopy(group)

    def add_member(self, group_id: str, person_id: str) -> bool:
        group_id = _sanitize(group_id, 64)
        person_id = _sanitize(person_id, 64)
        with self._lock:
            group = self._groups.get(group_id)
            if not group:
                return False
            if person_id in group.members:
                return True
            if len(group.members) >= self._MAX_MEMBERS:
                return False
            group.members.append(person_id)
            with self._stats_lock:
                self._stats["total_members"] += 1
            return True

    def remove_member(self, group_id: str, person_id: str) -> bool:
        group_id = _sanitize(group_id, 64)
        person_id = _sanitize(person_id, 64)
        with self._lock:
            group = self._groups.get(group_id)
            if not group:
                return False
            if person_id not in group.members:
                return False
            group.members.remove(person_id)
            with self._stats_lock:
                self._stats["total_members"] = max(0, self._stats["total_members"] - 1)
            return True

    def add_influence(self, source: str, target: str, influence_type: InfluenceType, strength: float = 0.5, context: str = "") -> InfluenceRelation:
        source = _sanitize(source, 64)
        target = _sanitize(target, 64)
        context = _sanitize_content(context, 500)
        strength = max(0.0, min(1.0, strength))
        relation_id = f"inf_{uuid.uuid4().hex[:12]}"
        with self._lock:
            if len(self._influences) >= self._MAX_INFLUENCE:
                evict_id = self._influence_order.pop(0)
                self._influences.pop(evict_id, None)
                logger.warning("Evicting influence relation %s (cap=%d)", evict_id, self._MAX_INFLUENCE)
            rel = InfluenceRelation(
                relation_id=relation_id,
                source=source,
                target=target,
                influence_type=influence_type,
                strength=strength,
                context=context,
            )
            self._influences[relation_id] = rel
            self._influence_order.append(relation_id)
            with self._stats_lock:
                self._stats["influence_relations"] = len(self._influences)
            return copy.deepcopy(rel)

    def assign_role(self, person_id: str, group_id: str, role_type: RoleType, influence_level: float = 0.5) -> SocialRole:
        person_id = _sanitize(person_id, 64)
        group_id = _sanitize(group_id, 64)
        influence_level = max(0.0, min(1.0, influence_level))
        role_id = f"role_{uuid.uuid4().hex[:12]}"
        with self._lock:
            if len(self._roles) >= self._MAX_ROLES:
                evict_id = self._role_order.pop(0)
                self._roles.pop(evict_id, None)
                logger.warning("Evicting role %s (cap=%d)", evict_id, self._MAX_ROLES)
            role = SocialRole(
                role_id=role_id,
                person_id=person_id,
                group_id=group_id,
                role_type=role_type,
                influence_level=influence_level,
            )
            self._roles[role_id] = role
            self._role_order.append(role_id)
            with self._stats_lock:
                self._stats["roles_assigned"] = len(self._roles)
            return copy.deepcopy(role)

    def detect_communication_patterns(self, group_id: str) -> List[CommunicationPattern]:
        group_id = _sanitize(group_id, 64)
        with self._lock:
            group = self._groups.get(group_id)
            if not group:
                return []
            members_set = set(group.members)
            group_influences = [
                rel for rel in self._influences.values()
                if rel.source in members_set or rel.target in members_set
            ]
            if not group_influences:
                return []
            type_groups: Dict[str, List[InfluenceRelation]] = {}
            for rel in group_influences:
                key = f"{rel.influence_type.value}"
                type_groups.setdefault(key, []).append(rel)
            detected = []
            for itype, rels in type_groups.items():
                participants = list(set(r.source for r in rels) | set(r.target for r in rels))[:20]
                strengths = [r.strength for r in rels]
                avg_strength = sum(strengths) / len(strengths) if strengths else 0.5
                avg_sentiment = (avg_strength - 0.5) * 2.0
                avg_sentiment = max(-1.0, min(1.0, avg_sentiment))
                pattern_type_map = {
                    "mentor": "information_flow",
                    "peer": "support",
                    "authority": "decision-making",
                    "expert": "information_flow",
                    "social": "support",
                    "informational": "information_flow",
                }
                pat_type = pattern_type_map.get(itype, "information_flow")
                pattern_id = f"cpat_{uuid.uuid4().hex[:12]}"
                pattern = CommunicationPattern(
                    pattern_id=pattern_id,
                    participants=participants,
                    frequency=len(rels),
                    pattern_type=pat_type,
                    avg_sentiment=avg_sentiment,
                )
                detected.append(pattern)
                if len(self._patterns) >= self._MAX_PATTERNS:
                    evict_id = self._pattern_order.pop(0)
                    self._patterns.pop(evict_id, None)
                    logger.warning("Evicting communication pattern %s (cap=%d)", evict_id, self._MAX_PATTERNS)
                self._patterns[pattern_id] = pattern
                self._pattern_order.append(pattern_id)
                with self._stats_lock:
                    self._stats["patterns_detected"] = len(self._patterns)
            return [copy.deepcopy(p) for p in detected]

    def get_group_dynamics(self, group_id: str) -> Dict:
        group_id = _sanitize(group_id, 64)
        with self._lock:
            group = self._groups.get(group_id)
            if not group:
                return {}
            members_set = set(group.members)
            group_roles = [r for r in self._roles.values() if r.group_id == group_id]
            role_distribution: Dict[str, int] = {}
            for r in group_roles:
                rt = r.role_type.value
                role_distribution[rt] = role_distribution.get(rt, 0) + 1
            group_influences = [
                rel for rel in self._influences.values()
                if rel.source in members_set or rel.target in members_set
            ]
            influence_network: Dict[str, List[Dict]] = {}
            for rel in group_influences:
                if rel.source not in influence_network:
                    influence_network[rel.source] = []
                influence_network[rel.source].append({
                    "target": rel.target,
                    "type": rel.influence_type.value,
                    "strength": round(rel.strength, 3),
                })
            cohesion = group.cohesion_score
            if group_influences:
                avg_strength = sum(r.strength for r in group_influences) / len(group_influences)
                cohesion = max(0.0, min(1.0, avg_strength))
                group.cohesion_score = cohesion
            return {
                "group_id": group_id,
                "cohesion_score": round(cohesion, 3),
                "member_count": len(group.members),
                "role_distribution": role_distribution,
                "influence_network": influence_network,
                "influence_count": len(group_influences),
            }

    def find_influencers(self, group_id: Optional[str] = None, min_influence: float = 0.7) -> List[Dict]:
        min_influence = max(0.0, min(1.0, min_influence))
        if group_id:
            group_id = _sanitize(group_id, 64)
        with self._lock:
            if group_id:
                group = self._groups.get(group_id)
                if not group:
                    return []
                members_set = set(group.members)
                relevant = [
                    rel for rel in self._influences.values()
                    if rel.source in members_set and rel.strength >= min_influence
                ]
            else:
                relevant = [
                    rel for rel in self._influences.values()
                    if rel.strength >= min_influence
                ]
            influencer_scores: Dict[str, Dict] = {}
            for rel in relevant:
                if rel.source not in influencer_scores:
                    influencer_scores[rel.source] = {
                        "person_id": rel.source,
                        "total_influence": 0.0,
                        "relation_count": 0,
                        "influence_types": set(),
                    }
                influencer_scores[rel.source]["total_influence"] += rel.strength
                influencer_scores[rel.source]["relation_count"] += 1
                influencer_scores[rel.source]["influence_types"].add(rel.influence_type.value)
            result = []
            for pid, data in influencer_scores.items():
                result.append({
                    "person_id": pid,
                    "avg_influence": round(data["total_influence"] / data["relation_count"], 3) if data["relation_count"] > 0 else 0.0,
                    "relation_count": data["relation_count"],
                    "influence_types": sorted(data["influence_types"]),
                })
            result.sort(key=lambda x: x["avg_influence"], reverse=True)
            return result

    def get_influence_network(self, person_id: str) -> Dict:
        person_id = _sanitize(person_id, 64)
        with self._lock:
            outgoing = [rel for rel in self._influences.values() if rel.source == person_id]
            incoming = [rel for rel in self._influences.values() if rel.target == person_id]
            return {
                "person_id": person_id,
                "outgoing": [r.to_dict() for r in outgoing],
                "incoming": [r.to_dict() for r in incoming],
                "outgoing_count": len(outgoing),
                "incoming_count": len(incoming),
            }

    def get_group(self, group_id: str) -> Optional[SocialGroup]:
        with self._lock:
            group = self._groups.get(group_id)
            return copy.deepcopy(group) if group else None

    def list_groups(self, group_type: Optional[GroupType] = None) -> List[SocialGroup]:
        with self._lock:
            if group_type:
                groups = [g for g in self._groups.values() if g.group_type == group_type]
            else:
                groups = list(self._groups.values())
            return [copy.deepcopy(g) for g in groups]

    def get_stats(self) -> Dict:
        with self._stats_lock:
            total_members = self._stats["total_members"]
        with self._lock:
            groups_count = len(self._groups)
            influences_count = len(self._influences)
            roles_count = len(self._roles)
            patterns_count = len(self._patterns)
        return {
            "groups_tracked": groups_count,
            "influence_relations": influences_count,
            "roles_assigned": roles_count,
            "patterns_detected": patterns_count,
            "total_members": total_members,
        }