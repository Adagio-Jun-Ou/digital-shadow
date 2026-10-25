#!/usr/bin/env python3
"""
MoralReasoning - Ethical Decision-Making Engine

Evaluates actions and decisions through multiple ethical frameworks,
identifies moral dilemmas, and provides structured ethical analysis.

Architecture:
  - MoralDilemma: describes an ethical conflict between principles
  - EthicalEvaluation: structured assessment through multiple frameworks
  - MoralPrinciple: tracked principle with weight and application history
  - MoralReasoning: main engine for ethical analysis
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
from typing import Dict, List, Optional, Tuple


logger = logging.getLogger("shadowd.cognition.moral_reasoning")

_MAX_EVALUATIONS = 3000
_MAX_DILEMMAS = 500
_MAX_PRINCIPLES = 100
_MAX_STAKEHOLDERS = 20

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_STRICT_RE.sub('', text).strip()
    return text[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    text = _SANITIZE_CONTENT_RE.sub('', text).strip()
    return text[:max_len]


def _keyword_match(kw: str, text: str) -> bool:
    return bool(re.search(r'\b' + re.escape(kw) + r'\b', text))


class EthicalFramework(str, Enum):
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    RIGHTS_BASED = "rights_based"
    FAIRNESS = "fairness"
    COMMON_GOOD = "common_good"


class MoralWeight(int, Enum):
    CRITICAL = 5
    HIGH = 4
    MODERATE = 3
    LOW = 2
    MINOR = 1


class DilemmaStatus(str, Enum):
    IDENTIFIED = "identified"
    ANALYZING = "analyzing"
    RESOLVED = "resolved"
    ESCALATED = "escalated"


class StakeholderType(str, Enum):
    SELF = "self"
    OTHERS = "others"
    COMMUNITY = "community"
    SOCIETY = "society"
    ENVIRONMENT = "environment"
    FUTURE_GENERATIONS = "future_generations"


@dataclass
class Stakeholder:
    name: str = ""
    stakeholder_type: StakeholderType = StakeholderType.OTHERS
    interests: List[str] = field(default_factory=list)
    impact: float = 0.5
    vulnerability: float = 0.5

    def __post_init__(self):
        self.name = _sanitize(self.name, max_len=100)
        self.interests = [_sanitize(i, max_len=200) for i in self.interests[:10]]
        self.impact = max(0.0, min(1.0, self.impact))
        self.vulnerability = max(0.0, min(1.0, self.vulnerability))

    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "type": self.stakeholder_type.value,
            "interests": list(self.interests[:5]),
            "impact": round(self.impact, 3),
            "vulnerability": round(self.vulnerability, 3),
        }


@dataclass
class FrameworkAssessment:
    framework: EthicalFramework = EthicalFramework.UTILITARIAN
    score: float = 0.5
    reasoning: str = ""
    concerns: List[str] = field(default_factory=list)
    recommendation: str = ""

    def __post_init__(self):
        self.score = max(0.0, min(1.0, self.score))

    def to_dict(self) -> Dict:
        return {
            "framework": self.framework.value,
            "score": round(self.score, 3),
            "reasoning": self.reasoning[:500],
            "concerns": list(self.concerns[:5]),
            "recommendation": self.recommendation[:200],
        }


@dataclass
class EthicalEvaluation:
    evaluation_id: str = ""
    action: str = ""
    description: str = ""
    stakeholders: List[Stakeholder] = field(default_factory=list)
    assessments: List[FrameworkAssessment] = field(default_factory=list)
    overall_score: float = 0.5
    moral_weight: MoralWeight = MoralWeight.MODERATE
    conflicts: List[str] = field(default_factory=list)
    recommendation: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        if not self.evaluation_id:
            self.evaluation_id = f"eth_{uuid.uuid4().hex[:12]}"
        self.action = _sanitize_content(self.action, max_len=500)
        self.description = _sanitize_content(self.description, max_len=2000)
        self.overall_score = max(0.0, min(1.0, self.overall_score))
        if self.metadata and len(self.metadata) > 50:
            self.metadata = dict(list(self.metadata.items())[:50])
        if self.metadata:
            self.metadata = {k: _sanitize_content(str(v), 500) for k, v in self.metadata.items()}

    def to_dict(self) -> Dict:
        return {
            "evaluation_id": self.evaluation_id,
            "action": self.action[:200],
            "description": self.description[:500],
            "stakeholders": [s.to_dict() for s in self.stakeholders[:10]],
            "assessments": [a.to_dict() for a in self.assessments[:20]],
            "overall_score": round(self.overall_score, 3),
            "moral_weight": self.moral_weight.value,
            "conflicts": list(self.conflicts[:5]),
            "recommendation": self.recommendation[:200],
            "created_at": self.created_at,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
        }


@dataclass
class MoralDilemma:
    dilemma_id: str = ""
    title: str = ""
    description: str = ""
    competing_values: List[str] = field(default_factory=list)
    status: DilemmaStatus = DilemmaStatus.IDENTIFIED
    evaluation_ids: List[str] = field(default_factory=list)
    resolution: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    resolved_at: Optional[str] = None

    def __post_init__(self):
        if not self.dilemma_id:
            self.dilemma_id = f"dilm_{uuid.uuid4().hex[:12]}"
        self.title = _sanitize(self.title, max_len=200)
        self.description = _sanitize_content(self.description, max_len=2000)

    def to_dict(self) -> Dict:
        return {
            "dilemma_id": self.dilemma_id,
            "title": self.title,
            "description": self.description[:500],
            "competing_values": list(self.competing_values[:5]),
            "status": self.status.value,
            "evaluation_ids": list(self.evaluation_ids),
            "resolution": self.resolution[:200],
            "created_at": self.created_at,
            "resolved_at": self.resolved_at,
        }


@dataclass
class MoralPrinciple:
    principle_id: str = ""
    name: str = ""
    description: str = ""
    framework: EthicalFramework = EthicalFramework.UTILITARIAN
    weight: float = 1.0
    applications: int = 0
    violations: int = 0
    non_negotiable: bool = False

    def __post_init__(self):
        if not self.principle_id:
            self.principle_id = f"prin_{uuid.uuid4().hex[:12]}"
        self.name = _sanitize(self.name, max_len=100)
        self.description = _sanitize_content(self.description, max_len=500)
        self.weight = max(0.0, min(2.0, self.weight))

    def to_dict(self) -> Dict:
        return {
            "principle_id": self.principle_id,
            "name": self.name,
            "framework": self.framework.value,
            "weight": round(self.weight, 3),
            "applications": self.applications,
            "violations": self.violations,
            "non_negotiable": self.non_negotiable,
        }


_HARM_KEYWORDS = {"harm", "hurt", "damage", "destroy", "kill", "injure", "suffer", "pain", "loss"}
_BENEFIT_KEYWORDS = {"help", "benefit", "improve", "protect", "save", "heal", "support", "enable", "empower"}
_FAIRNESS_POSITIVE_KEYWORDS = {"fair", "equal", "equitable", "just", "unbiased"}
_FAIRNESS_NEGATIVE_KEYWORDS = {"discrimination", "prejudice", "inequality"}
_RIGHTS_KEYWORDS = {"right", "freedom", "autonomy", "consent", "privacy", "dignity", "choice"}
_CARE_KEYWORDS = {"care", "empathy", "compassion", "nurture", "relationship", "trust", "support"}

_HARM_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _HARM_KEYWORDS]
_BENEFIT_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _BENEFIT_KEYWORDS]
_FAIRNESS_POSITIVE_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _FAIRNESS_POSITIVE_KEYWORDS]
_FAIRNESS_NEGATIVE_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _FAIRNESS_NEGATIVE_KEYWORDS]
_RIGHTS_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _RIGHTS_KEYWORDS]
_CARE_PATTERNS = [re.compile(r'\b' + re.escape(kw) + r'\b') for kw in _CARE_KEYWORDS]



def _count_pattern_matches(patterns: List["re.Pattern"], text: str) -> int:
    return sum(1 for pat in patterns if pat.search(text))


class MoralReasoning:
    """
    Ethical Decision-Making Engine.

    Evaluates actions through multiple ethical frameworks, identifies moral
    dilemmas, tracks principles, and provides structured ethical analysis.
    """

    def __init__(self):
        self._evaluations: Dict[str, EthicalEvaluation] = {}
        self._dilemmas: Dict[str, MoralDilemma] = {}
        self._principles: Dict[str, MoralPrinciple] = {}
        self._lock = threading.Lock()
        self._stats = {
            "evaluations_made": 0,
            "dilemmas_identified": 0,
            "principles_applied": 0,
            "avg_overall_score": 0.5,
            "conflicts_found": 0,
        }
        self._stats_lock = threading.Lock()
        self._init_core_principles()

    def _init_core_principles(self):
        core = [
            MoralPrinciple(name="Do no harm", description="Avoid causing harm to others",
                           framework=EthicalFramework.UTILITARIAN, non_negotiable=True),
            MoralPrinciple(name="Respect autonomy", description="Honor individual choices and consent",
                           framework=EthicalFramework.DEONTOLOGICAL, non_negotiable=True),
            MoralPrinciple(name="Be fair", description="Treat people equitably without discrimination",
                           framework=EthicalFramework.FAIRNESS),
            MoralPrinciple(name="Show care", description="Demonstrate empathy and compassion",
                           framework=EthicalFramework.CARE_ETHICS),
            MoralPrinciple(name="Protect privacy", description="Safeguard personal information",
                           framework=EthicalFramework.RIGHTS_BASED),
        ]
        for p in core:
            self._principles[p.principle_id] = p

    def add_principle(self, name: str, description: str, framework: EthicalFramework,
                       weight: float = 1.0, non_negotiable: bool = False) -> MoralPrinciple:
        principle = MoralPrinciple(
            name=name, description=description,
            framework=framework, weight=weight, non_negotiable=non_negotiable,
        )
        with self._lock:
            self._principles[principle.principle_id] = principle
            if len(self._principles) > _MAX_PRINCIPLES:
                sorted_keys = sorted(
                    self._principles.keys(),
                    key=lambda k: (self._principles[k].non_negotiable, k),
                )
                evict_key = sorted_keys[0]
                if self._principles[evict_key].non_negotiable:
                    logger.warning("Evicting non_negotiable principle %s", evict_key)
                else:
                    logger.warning("Evicting principle %s", evict_key)
                del self._principles[evict_key]
        return copy.deepcopy(principle)

    def evaluate(self, action: str, description: str = "",
                  stakeholders: Optional[List[Stakeholder]] = None) -> EthicalEvaluation:
        action = _sanitize(action, max_len=500)
        description = _sanitize_content(description, max_len=2000)

        if stakeholders is None:
            stakeholders = []
        stakeholders = stakeholders[:_MAX_STAKEHOLDERS]

        assessments = self._run_framework_assessments(action, description, stakeholders)

        overall = sum(a.score * self._framework_weight(a.framework) for a in assessments)
        weight_sum = sum(self._framework_weight(a.framework) for a in assessments)
        overall = overall / max(weight_sum, 1.0)

        conflicts = self._detect_conflicts(assessments)

        moral_weight = MoralWeight.MODERATE
        if overall < 0.3:
            moral_weight = MoralWeight.CRITICAL
        elif overall < 0.5:
            moral_weight = MoralWeight.HIGH
        elif overall > 0.8:
            moral_weight = MoralWeight.LOW

        recommendation = self._generate_recommendation(overall, conflicts, assessments)

        evaluation = EthicalEvaluation(
            action=action,
            description=description,
            stakeholders=stakeholders,
            assessments=assessments,
            overall_score=overall,
            moral_weight=moral_weight,
            conflicts=conflicts,
            recommendation=recommendation,
        )

        with self._lock:
            self._evaluations[evaluation.evaluation_id] = evaluation
            if len(self._evaluations) > _MAX_EVALUATIONS:
                oldest = list(self._evaluations.keys())[0]
                logger.warning("Evicting evaluation %s (dangling refs in dilemmas possible)", oldest)
                for d in self._dilemmas.values():
                    if oldest in d.evaluation_ids:
                        d.evaluation_ids = [eid for eid in d.evaluation_ids if eid != oldest]
                del self._evaluations[oldest]
            with self._stats_lock:
                self._stats["evaluations_made"] += 1
            total = self._stats["evaluations_made"]
            with self._stats_lock:
                self._stats["avg_overall_score"] += (overall - self._stats["avg_overall_score"]) / total
            with self._stats_lock:
                self._stats["conflicts_found"] += len(conflicts)

        self._check_dilemma(action, conflicts, evaluation.evaluation_id)

        return copy.deepcopy(evaluation)

    def _framework_weight(self, framework: EthicalFramework) -> float:
        with self._lock:
            matching = [(p.weight,) for p in self._principles.values() if p.framework == framework]
        if not matching:
            return 1.0
        return sum(w for w, in matching) / len(matching)

    def _run_framework_assessments(self, action: str, description: str,
                                     stakeholders: List[Stakeholder]) -> List[FrameworkAssessment]:
        combined = (action + " " + description).lower()
        assessments = []

        harm_score = _count_pattern_matches(_HARM_PATTERNS, combined)
        benefit_score = _count_pattern_matches(_BENEFIT_PATTERNS, combined)

        util_score = 0.5 + (benefit_score - harm_score) * 0.1
        util_score = max(0.0, min(1.0, util_score))
        assessments.append(FrameworkAssessment(
            framework=EthicalFramework.UTILITARIAN,
            score=util_score,
            reasoning=f"Benefits: {benefit_score}, Harms: {harm_score}",
            concerns=["Harm detected"] if harm_score > benefit_score else [],
            recommendation="Maximize benefit, minimize harm",
        ))

        rights_violations = _count_pattern_matches(_RIGHTS_PATTERNS, combined)
        deon_score = min(1.0, 0.5 + rights_violations * 0.1)
        assessments.append(FrameworkAssessment(
            framework=EthicalFramework.DEONTOLOGICAL,
            score=deon_score,
            reasoning=f"Rights considerations: {rights_violations}",
            concerns=[] if rights_violations > 0 else ["No rights language detected"],
            recommendation="Ensure duties and rights are respected",
        ))

        care_keywords = _count_pattern_matches(_CARE_PATTERNS, combined)
        care_score = 0.5 + care_keywords * 0.1
        has_vulnerable = any(s.vulnerability > 0.7 for s in stakeholders)
        if has_vulnerable:
            care_score = min(1.0, care_score + 0.1)
        care_score = max(0.0, min(1.0, care_score))
        assessments.append(FrameworkAssessment(
            framework=EthicalFramework.CARE_ETHICS,
            score=care_score,
            reasoning=f"Care indicators: {care_keywords}, Vulnerable stakeholders: {has_vulnerable}",
            recommendation="Prioritize care for vulnerable parties",
        ))

        fairness_pos = _count_pattern_matches(_FAIRNESS_POSITIVE_PATTERNS, combined)
        fairness_neg = _count_pattern_matches(_FAIRNESS_NEGATIVE_PATTERNS, combined)
        fairness_score = max(0.0, min(1.0, 0.5 + fairness_pos * 0.1 - fairness_neg * 0.1))
        assessments.append(FrameworkAssessment(
            framework=EthicalFramework.FAIRNESS,
            score=fairness_score,
            reasoning=f"Fairness indicators: {fairness_pos} positive, {fairness_neg} negative",
            concerns=[] if fairness_pos > 0 or fairness_neg == 0 else ["Anti-fairness language detected"],
            recommendation="Ensure equitable treatment",
        ))

        return assessments

    def _detect_conflicts(self, assessments: List[FrameworkAssessment]) -> List[str]:
        conflicts = []
        if len(assessments) < 2:
            return conflicts

        scores = {a.framework: a.score for a in assessments}
        frameworks = list(scores.keys())

        for i in range(len(frameworks)):
            for j in range(i + 1, len(frameworks)):
                diff = abs(scores[frameworks[i]] - scores[frameworks[j]])
                if diff > 0.3:
                    higher = frameworks[i] if scores[frameworks[i]] > scores[frameworks[j]] else frameworks[j]
                    lower = frameworks[j] if higher == frameworks[i] else frameworks[i]
                    conflicts.append(f"{higher.value} vs {lower.value} (gap: {diff:.2f})")

        return conflicts

    def _generate_recommendation(self, overall: float, conflicts: List[str],
                                  assessments: List[FrameworkAssessment]) -> str:
        base = ""
        if overall >= 0.7:
            base = "Action appears ethically acceptable across frameworks."
        elif overall >= 0.5:
            base = "Action has moderate ethical concerns. Consider modifications."
        elif overall >= 0.3:
            base = "Action has significant ethical issues. Seek alternatives or additional safeguards."
        else:
            base = "Action raises serious ethical concerns. Strong recommendation against proceeding."
        if conflicts:
            base += f" Note: ethical tensions exist between {', '.join(conflicts[:3])}."
        return base

    def _check_dilemma(self, action: str, conflicts: List[str], evaluation_id: str):
        if len(conflicts) >= 2:
            dilemma = MoralDilemma(
                title=f"Ethical dilemma: {action[:50]}",
                description=f"Multiple framework conflicts detected: {'; '.join(conflicts)}",
                competing_values=conflicts[:5],
                evaluation_ids=[evaluation_id],
            )
            with self._lock:
                self._dilemmas[dilemma.dilemma_id] = dilemma
                if len(self._dilemmas) > _MAX_DILEMMAS:
                    oldest = list(self._dilemmas.keys())[0]
                    logger.warning("Evicting dilemma %s", oldest)
                    del self._dilemmas[oldest]
                with self._stats_lock:
                    self._stats["dilemmas_identified"] += 1

    def resolve_dilemma(self, dilemma_id: str, resolution: str) -> bool:
        resolution = _sanitize_content(resolution, max_len=1000)
        with self._lock:
            dilemma = self._dilemmas.get(dilemma_id)
            if not dilemma:
                return False
            if dilemma.status == DilemmaStatus.RESOLVED:
                return False
            dilemma.status = DilemmaStatus.RESOLVED
            dilemma.resolution = resolution
            dilemma.resolved_at = datetime.now(timezone.utc).isoformat()
        return True

    def get_dilemmas(self, status: Optional[DilemmaStatus] = None) -> List[Dict]:
        with self._lock:
            dilemmas = list(self._dilemmas.values())
        if status:
            dilemmas = [d for d in dilemmas if d.status == status]
        return [d.to_dict() for d in dilemmas]

    def get_principles(self, framework: Optional[EthicalFramework] = None) -> List[Dict]:
        with self._lock:
            principles = list(self._principles.values())
        if framework:
            principles = [p for p in principles if p.framework == framework]
        return [p.to_dict() for p in principles]

    def get_evaluations(self, min_score: float = 0.0, limit: int = 20) -> List[Dict]:
        with self._lock:
            evals = list(self._evaluations.values())
        if min_score > 0:
            evals = [e for e in evals if e.overall_score >= min_score]
        evals.sort(key=lambda e: e.created_at, reverse=True)
        return [e.to_dict() for e in evals[:limit]]

    def get_stats(self) -> Dict:
        with self._stats_lock:
            stats = dict(self._stats)
        with self._lock:
            evals_total = len(self._evaluations)
            dilemmas_total = len(self._dilemmas)
            principles_total = len(self._principles)
            unresolved = sum(1 for d in self._dilemmas.values() if d.status != DilemmaStatus.RESOLVED)
        return {
            "stats": stats,
            "evaluations_total": evals_total,
            "dilemmas_total": dilemmas_total,
            "principles_total": principles_total,
            "unresolved_dilemmas": unresolved,
        }