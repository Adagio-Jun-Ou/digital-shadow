#!/usr/bin/env python3
"""
AdaptiveSkillTree - Dynamic Ability Leveling Based on Usage Patterns

Tracks skills/abilities across domains, automatically levels them based on
usage frequency and success rate, unlocks skill dependencies, and provides
recommendations for skill development paths.

Architecture:
  - SkillNode: individual skill with level, XP, dependencies
  - SkillTree: tree of skills with leveling, unlocking, recommendations
  - Usage tracking with decay for recency bias
  - Thread-safe with bounded history
"""

import re
import copy
import logging
import threading
import time
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.adaptive_skill_tree")


class SkillDomain(str, Enum):
    COMMUNICATION = "communication"
    REASONING = "reasoning"
    CREATIVITY = "creativity"
    ANALYSIS = "analysis"
    LEADERSHIP = "leadership"
    TECHNICAL = "technical"
    SOCIAL = "social"
    WELLNESS = "wellness"
    PRODUCTIVITY = "productivity"
    ADAPTABILITY = "adaptability"


class SkillLevel(int, Enum):
    NOVICE = 0
    BEGINNER = 1
    INTERMEDIATE = 2
    ADVANCED = 3
    EXPERT = 4
    MASTER = 5


_XP_PER_LEVEL = [0, 100, 300, 700, 1500, 3000]
_LEVEL_NAMES = ["novice", "beginner", "intermediate", "advanced", "expert", "master"]


@dataclass
class SkillNode:
    skill_id: str
    name: str
    domain: SkillDomain
    level: int = 0
    xp: float = 0.0
    dependencies: List[str] = field(default_factory=list)
    unlocked: bool = True
    usage_count: int = 0
    success_count: int = 0
    last_used: Optional[str] = None
    metadata: Dict = field(default_factory=dict)

    @property
    def level_name(self) -> str:
        if 0 <= self.level < len(_LEVEL_NAMES):
            return _LEVEL_NAMES[self.level]
        return "unknown"

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return min(1.0, self.success_count / max(self.usage_count, 1))

    @property
    def xp_to_next(self) -> float:
        if self.level >= len(_XP_PER_LEVEL) - 1:
            return 0.0
        return max(0.0, _XP_PER_LEVEL[self.level + 1] - self.xp)

    @property
    def progress(self) -> float:
        if self.level >= len(_XP_PER_LEVEL) - 1:
            return 1.0
        current_threshold = _XP_PER_LEVEL[self.level]
        next_threshold = _XP_PER_LEVEL[self.level + 1]
        if next_threshold == current_threshold:
            return 1.0
        return min(1.0, (self.xp - current_threshold) / (next_threshold - current_threshold))

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "name": self.name,
            "domain": self.domain.value,
            "level": self.level,
            "level_name": self.level_name,
            "xp": round(self.xp, 1),
            "xp_to_next": round(self.xp_to_next, 1),
            "progress": round(self.progress, 3),
            "dependencies": list(self.dependencies),
            "unlocked": self.unlocked,
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "success_rate": round(self.success_rate, 3),
            "last_used": self.last_used,
        }


@dataclass
class SkillRecommendation:
    skill_id: str
    reason: str
    priority: float
    current_level: int
    target_level: int

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "reason": self.reason,
            "priority": round(self.priority, 3),
            "current_level": self.current_level,
            "target_level": self.target_level,
        }


@dataclass
class LevelUpEvent:
    skill_id: str
    skill_name: str
    old_level: int
    new_level: int
    timestamp: str

    def to_dict(self) -> Dict:
        return {
            "skill_id": self.skill_id,
            "skill_name": self.skill_name,
            "old_level": self.old_level,
            "new_level": self.new_level,
            "timestamp": self.timestamp,
        }


_XP_BASE = 10.0
_XP_SUCCESS_BONUS = 5.0
_MAX_SKILLS = 500
_MAX_HISTORY = 1000
_MAX_NAME_LENGTH = 100
_MAX_METADATA_LENGTH = 1000

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


class AdaptiveSkillTree:
    _MAX_SKILLS = _MAX_SKILLS

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._skills: Dict[str, SkillNode] = {}
        self._level_up_history: deque = deque(maxlen=_MAX_HISTORY)
        self._recent_usage: deque = deque(maxlen=1000)
        self._stats = {
            "skills_added": 0,
            "skills_used": 0,
            "level_ups": 0,
        }

    def _sanitize(self, text: str) -> str:
        if not isinstance(text, str):
            text = str(text)
        return _SANITIZE_STRICT_RE.sub('', text).strip()[:_MAX_NAME_LENGTH]

    def _sanitize_content(self, text: str, max_len: int = 5000) -> str:
        if not isinstance(text, str):
            text = str(text)
        return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]

    def _validate_id(self, skill_id: str) -> str:
        skill_id = skill_id.strip()
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', skill_id):
            raise ValueError(f"Invalid skill_id: {skill_id!r}")
        return skill_id

    def add_skill(
        self,
        skill_id: str,
        name: str,
        domain: str,
        dependencies: Optional[List[str]] = None,
        initial_xp: float = 0.0,
        metadata: Optional[Dict] = None,
    ) -> SkillNode:
        skill_id = self._validate_id(skill_id)
        name = self._sanitize(name)

        try:
            domain_enum = SkillDomain(domain)
        except ValueError:
            raise ValueError(f"Invalid domain: {domain!r}")

        deps = [self._validate_id(d) for d in (dependencies or [])]
        meta = {}
        if metadata:
            for k, v in list(metadata.items())[:20]:
                if isinstance(k, str) and len(k) < 100:
                    if isinstance(v, (str, int, float, bool)):
                        meta[self._sanitize(str(k))] = self._sanitize_content(str(v), 200) if isinstance(v, str) else v
                    elif isinstance(v, list):
                        meta[self._sanitize(str(k))] = [self._sanitize_content(str(item), 200) for item in v[:50]]
                    elif isinstance(v, dict):
                        meta[self._sanitize(str(k))] = {sk: str(sv)[:200] for sk, sv in list(v.items())[:20]}

        unlocked = True  
        initial_xp = max(0.0, initial_xp)

        node = SkillNode(
            skill_id=skill_id,
            name=name,
            domain=domain_enum,
            xp=initial_xp,
            dependencies=deps,
            unlocked=unlocked,
            metadata=meta,
        )

        new_level = self._calculate_level(node.xp)
        node.level = new_level

        with self._lock:
            if skill_id in self._skills:
                raise ValueError(f"Skill already exists: {skill_id}")
            if len(self._skills) >= self._MAX_SKILLS:
                oldest = next(iter(self._skills))
                logger.warning("Evicting oldest skill %s to make room", oldest)
                del self._skills[oldest]
                for sk in self._skills.values():
                    if oldest in sk.dependencies:
                        sk.dependencies = [d for d in sk.dependencies if d != oldest]
                        if not sk.dependencies:
                            sk.unlocked = True
                        else:
                            sk.unlocked = all(d in self._skills and self._skills[d].unlocked for d in sk.dependencies)
            node.unlocked = all(d in self._skills for d in deps) if deps else True
            self._skills[skill_id] = node

            with self._stats_lock:
                self._stats["skills_added"] += 1

        return node

    def use_skill(self, skill_id: str, success: bool = True, context: Optional[Dict] = None) -> SkillNode:
        skill_id = self._validate_id(skill_id)
        with self._lock:
            skill = self._skills.get(skill_id)
            if skill is None:
                raise ValueError(f"Skill not found: {skill_id}")
            if not skill.unlocked:
                raise ValueError(f"Skill locked: {skill_id}")

            old_level = skill.level
            xp_gain = _XP_BASE
            if success:
                xp_gain += _XP_SUCCESS_BONUS
                skill.success_count += 1

            skill.usage_count += 1
            skill.last_used = datetime.now(timezone.utc).isoformat()
            skill.xp += xp_gain
            skill.level = self._calculate_level(skill.xp)

            if skill.level > old_level:
                event = LevelUpEvent(
                    skill_id=skill_id,
                    skill_name=skill.name,
                    old_level=old_level,
                    new_level=skill.level,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                )
                self._level_up_history.append(event)
                self._check_unlocks()
                with self._stats_lock:
                    self._stats["level_ups"] += 1

            self._recent_usage.append(time.monotonic())
            with self._stats_lock:
                self._stats["skills_used"] += 1
            return copy.deepcopy(skill)

    def _calculate_level(self, xp: float) -> int:
        for i in range(len(_XP_PER_LEVEL) - 1, -1, -1):
            if xp >= _XP_PER_LEVEL[i]:
                return i
        return 0

    def _check_unlocks(self):
        for skill in self._skills.values():
            if not skill.unlocked and skill.dependencies:
                if all(d in self._skills and self._skills[d].level >= 1 for d in skill.dependencies):
                    skill.unlocked = True

    def get_skill(self, skill_id: str) -> Optional[SkillNode]:
        with self._lock:
            skill = self._skills.get(skill_id)
            return copy.deepcopy(skill) if skill else None

    def list_skills(self, domain: Optional[str] = None) -> List[SkillNode]:
        with self._lock:
            skills = [copy.deepcopy(s) for s in self._skills.values()]
        if domain:
            try:
                domain_enum = SkillDomain(domain)
            except ValueError:
                return []
            skills = [s for s in skills if s.domain == domain_enum]
        return sorted(skills, key=lambda s: s.xp, reverse=True)

    def get_recommendations(self, limit: int = 10) -> List[SkillRecommendation]:
        with self._lock:
            skill_snapshots = []
            for skill in self._skills.values():
                skill_snapshots.append((
                    skill.skill_id, skill.name, skill.unlocked,
                    skill.dependencies, skill.usage_count, skill.success_rate,
                    skill.last_used, skill.progress, skill.level,
                    [d for d in skill.dependencies if d not in self._skills or not self._skills[d].unlocked],
                ))
            recent_usage_len = len(self._recent_usage)

        recs = []
        for (skill_id, name, unlocked, deps, usage_count, success_rate,
             last_used, progress, level, locked_deps) in skill_snapshots:
            if not unlocked:
                if locked_deps:
                    for dep_id in locked_deps[:1]:
                        dep_snap = [s for s in skill_snapshots if s[0] == dep_id]
                        if dep_snap:
                            dep = dep_snap[0]
                            priority = 0.9 * (1.0 - dep[7])
                            recs.append(SkillRecommendation(
                                skill_id=dep_id,
                                reason=f"Required to unlock '{name}'",
                                priority=priority,
                                current_level=dep[8],
                                target_level=dep[8] + 1,
                            ))
                continue

            usage_freq = usage_count / max(recent_usage_len, 1)
            success_bonus = success_rate * 0.3
            stagnation = 0.0
            if last_used:
                try:
                    last = datetime.fromisoformat(last_used)
                    days_since = (datetime.now(timezone.utc) - last).days
                    if days_since > 30:
                        stagnation = min(0.5, days_since / 180)
                except (ValueError, TypeError):
                    stagnation = 0.1

            priority = (0.3 + usage_freq + success_bonus + stagnation)
            if progress >= 0.8:
                priority += 0.2

            recs.append(SkillRecommendation(
                skill_id=skill_id,
                reason="Continue developing" if progress >= 0.5 else "Build foundational skills",
                priority=min(1.0, priority),
                current_level=level,
                target_level=min(level + 1, 5),
            ))

        recs.sort(key=lambda r: r.priority, reverse=True)
        return recs[:limit]

    def get_domain_summary(self) -> Dict[str, Dict]:
        with self._lock:
            domains = {}
            for domain in SkillDomain:
                domain_skills = [s for s in self._skills.values() if s.domain == domain]
                if not domain_skills:
                    domains[domain.value] = {
                        "level": 0,
                        "xp": 0.0,
                        "skills": 0,
                        "unlocked": 0,
                        "avg_success_rate": 0.0,
                    }
                    continue
                domains[domain.value] = {
                    "level": round(sum(s.level for s in domain_skills) / len(domain_skills), 2),
                    "xp": round(sum(s.xp for s in domain_skills), 1),
                    "skills": len(domain_skills),
                    "unlocked": sum(1 for s in domain_skills if s.unlocked),
                    "avg_success_rate": round(
                        sum(s.success_rate for s in domain_skills) / len(domain_skills), 3
                    ),
                }
            return domains

    def get_level_ups(self, skill_id: Optional[str] = None, limit: int = 50) -> List[LevelUpEvent]:
        with self._lock:
            events = [copy.deepcopy(e) for e in self._level_up_history]
        if skill_id:
            events = [e for e in events if e.skill_id == skill_id]
        return events[:limit]

    def remove_skill(self, skill_id: str) -> bool:
        with self._lock:
            if skill_id not in self._skills:
                return False
            del self._skills[skill_id]
            for skill in self._skills.values():
                if not skill.unlocked and skill.dependencies:
                    if all(d in self._skills and self._skills[d].level >= 1 for d in skill.dependencies):
                        skill.unlocked = True
            return True

    def get_stats(self) -> Dict:
        with self._lock:
            total_skills = len(self._skills)
            unlocked = sum(1 for s in self._skills.values() if s.unlocked)
            levels = [s.level for s in self._skills.values()]
            totals = {
                "total_skills": total_skills,
                "unlocked_skills": unlocked,
                "locked_skills": total_skills - unlocked,
                "average_level": round(sum(levels) / max(len(levels), 1), 2),
                "max_level": max(levels) if levels else 0,
                "total_usage": sum(s.usage_count for s in self._skills.values()),
                "recent_usage_count": len(self._recent_usage),
                "level_ups": len(self._level_up_history),
            }
        with self._stats_lock:
            stats = dict(self._stats)
        stats.update(totals)
        return stats

    def get_skill_tree(self) -> Dict:
        with self._lock:
            nodes = {}
            for skill in self._skills.values():
                nodes[skill.skill_id] = {
                    "name": skill.name,
                    "domain": skill.domain.value,
                    "level": skill.level,
                    "level_name": skill.level_name,
                    "unlocked": skill.unlocked,
                    "dependencies": list(skill.dependencies),
                    "progress": round(skill.progress, 3),
                }
            return copy.deepcopy(nodes)