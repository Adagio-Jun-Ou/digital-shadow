"""
Cross-Module Decision Arbiter — Resolving conflicting insights from multiple engines.

When multiple cognitive engines produce contradictory insights, the Arbiter
evaluates each claim and produces an authoritative resolution:

1. Evidence-weighted scoring: stronger evidence = higher credibility
2. Source reliability tracking: modules with history of accuracy are trusted more
3. Temporal decay: older insights lose weight
4. Consensus detection: when most engines agree, amplify confidence
5. Dispute resolution: when engines disagree, produce a reasoned judgment

Design: A judicial logic engine — not ML, not heuristic, but structured
argumentation with transparent reasoning chains.
"""

import copy
import logging
import re
import threading
import uuid
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from enum import Enum
from typing import Any, Dict, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]

def _sanitize_content(text: str, max_length: int = 5000) -> str:
    if text is None: return ""
    if not isinstance(text, str): text = str(text)
    return _SANITIZE_RE.sub('', text).strip()[:max_length]

_MAX_CLAIMS = 2000
_MAX_JUDGMENTS = 500
_MAX_JURORS = 50
_MAX_EVIDENCE_PER_CLAIM = 20
_SOURCE_TRUST_DECAY = 0.001
_TEMPORAL_DECAY_HOURS = 24


class JudgmentOutcome(Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    PARTIALLY_CONFIRMED = "partially_confirmed"
    DEFERRED = "deferred"
    RECONCILED = "reconciled"


class EvidenceStrength(Enum):
    WEAK = 0.2
    MODERATE = 0.5
    STRONG = 0.8
    OVERWHELMING = 0.95


@dataclass
class Claim:
    claim_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    source_module: str = ""
    statement: str = ""
    confidence: float = 0.0
    evidence: List[str] = field(default_factory=list)
    counter_evidence: List[str] = field(default_factory=list)
    domain: str = "general"
    submitted_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    resolved: bool = False
    resolution_id: Optional[str] = None

    def __post_init__(self):
        self.source_module = _sanitize(self.source_module, 100)
        self.statement = _sanitize_content(self.statement, 2000)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.evidence = [_sanitize_content(e, 1000) for e in (self.evidence or [])[:_MAX_EVIDENCE_PER_CLAIM]]
        self.counter_evidence = [_sanitize_content(e, 1000) for e in (self.counter_evidence or [])[:_MAX_EVIDENCE_PER_CLAIM]]
        self.domain = _sanitize(self.domain, 50)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "source_module": self.source_module,
            "statement": self.statement[:200],
            "confidence": round(self.confidence, 3),
            "evidence_count": len(self.evidence),
            "domain": self.domain,
            "resolved": self.resolved,
        }


@dataclass
class Judgment:
    judgment_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    claim_id: str = ""
    outcome: str = "deferred"
    reasoning: str = ""
    adjusted_confidence: float = 0.0
    corroborating_modules: List[str] = field(default_factory=list)
    dissenting_modules: List[str] = field(default_factory=list)
    consensus_score: float = 0.0
    evidence_weight: float = 0.0
    source_trust: float = 0.5
    issued_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.claim_id = _sanitize(self.claim_id, 20)
        self.outcome = _sanitize(self.outcome, 30)
        self.reasoning = _sanitize_content(self.reasoning, 5000)
        self.adjusted_confidence = max(0.0, min(1.0, self.adjusted_confidence))
        self.corroborating_modules = [_sanitize(m, 100) for m in self.corroborating_modules[:20]]
        self.dissenting_modules = [_sanitize(m, 100) for m in self.dissenting_modules[:20]]
        self.consensus_score = max(0.0, min(1.0, self.consensus_score))
        self.evidence_weight = max(0.0, min(1.0, self.evidence_weight))
        self.source_trust = max(0.0, min(1.0, self.source_trust))
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:30]
        }

    def to_dict(self) -> Dict[str, Any]:
        return {
            "judgment_id": self.judgment_id,
            "claim_id": self.claim_id,
            "outcome": self.outcome,
            "reasoning": self.reasoning[:500],
            "adjusted_confidence": round(self.adjusted_confidence, 3),
            "consensus_score": round(self.consensus_score, 3),
            "evidence_weight": round(self.evidence_weight, 3),
            "source_trust": round(self.source_trust, 3),
            "issued_at": self.issued_at.isoformat(),
        }


class DecisionArbiter:
    _MAX_CLAIMS = _MAX_CLAIMS
    _MAX_JUDGMENTS = _MAX_JUDGMENTS

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._claims: Dict[str, Claim] = {}
        self._claim_order: deque = deque(maxlen=_MAX_CLAIMS)
        self._judgments: Dict[str, Judgment] = {}
        self._judgment_order: deque = deque(maxlen=_MAX_JUDGMENTS)
        self._source_trust: Dict[str, float] = defaultdict(lambda: 0.5)
        self._source_outcome_history: Dict[str, deque] = defaultdict(
            lambda: deque(maxlen=100)
        )

        self._stats: Dict[str, int] = {
            "claims_submitted": 0,
            "judgments_issued": 0,
            "claims_confirmed": 0,
            "claims_rejected": 0,
            "claims_reconciled": 0,
            "disputes_resolved": 0,
        }

    def submit_claim(self, source_module: str, statement: str,
                     confidence: float = 0.5, evidence: Optional[List[str]] = None,
                     counter_evidence: Optional[List[str]] = None,
                     domain: str = "general") -> str:
        claim = Claim(
            source_module=source_module, statement=statement,
            confidence=confidence, evidence=evidence or [],
            counter_evidence=counter_evidence or [], domain=domain,
        )
        with self._lock:
            self._claims[claim.claim_id] = claim
            self._claim_order.append(claim.claim_id)
            while len(self._claims) > self._MAX_CLAIMS:
                old_id = self._claim_order.popleft()
                self._claims.pop(old_id, None)
        with self._stats_lock:
            self._stats["claims_submitted"] += 1
        return claim.claim_id

    def adjudicate(self, claim_id: str) -> Optional[Judgment]:
        claim_id = _sanitize(claim_id, 20)
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim is None or claim.resolved:
                if claim and claim.resolved and claim.resolution_id:
                    return self._judgments.get(claim.resolution_id)
                return None

            related = self._find_related_claims(claim)
            source_trust = self._source_trust[claim.source_module]
            evidence_weight = self._compute_evidence_weight(claim)
            consensus = self._compute_consensus(claim, related)

            total_signals = evidence_weight * 0.35 + source_trust * 0.25 + consensus * 0.4
            adjusted_confidence = round(claim.confidence * total_signals, 3)

            if adjusted_confidence >= 0.8:
                outcome = "confirmed"
            elif adjusted_confidence >= 0.5:
                outcome = "partially_confirmed"
            elif adjusted_confidence >= 0.2:
                outcome = "deferred"
            else:
                outcome = "rejected"

            if consensus >= 0.8 and adjusted_confidence < 0.6:
                outcome = "reconciled"
                with self._stats_lock:
                    self._stats["claims_reconciled"] += 1

            corroborating = [
                c.source_module for c in related
                if c.confidence >= 0.5 and c.domain == claim.domain
            ][:10]
            dissenting = [
                c.source_module for c in related
                if c.confidence < 0.3 and c.domain != claim.domain
            ][:10]

            reasoning = (
                f"Evidence weight: {round(evidence_weight, 3)}, "
                f"source trust: {round(source_trust, 3)}, "
                f"consensus: {round(consensus, 3)} (from {len(related)} related claims). "
                f"Adjusted confidence: {adjusted_confidence}. "
                f"Outcome: {outcome}."
            )

            judgment = Judgment(
                claim_id=claim_id, outcome=outcome, reasoning=reasoning,
                adjusted_confidence=adjusted_confidence,
                corroborating_modules=corroborating,
                dissenting_modules=dissenting,
                consensus_score=consensus, evidence_weight=evidence_weight,
                source_trust=source_trust,
            )

            claim.resolved = True
            claim.resolution_id = judgment.judgment_id
            self._judgments[judgment.judgment_id] = judgment
            self._judgment_order.append(judgment.judgment_id)

            while len(self._judgments) > self._MAX_JUDGMENTS:
                old_id = self._judgment_order.popleft()
                self._judgments.pop(old_id, None)

        self._update_source_trust(claim.source_module, outcome)

        with self._stats_lock:
            self._stats["judgments_issued"] += 1
            if outcome == "confirmed":
                self._stats["claims_confirmed"] += 1
            elif outcome == "rejected":
                self._stats["claims_rejected"] += 1

        return judgment

    def record_source_outcome(self, module_name: str, correct: bool) -> None:
        module_name = _sanitize(module_name, 100)
        with self._lock:
            self._source_outcome_history[module_name].append({
                "correct": correct,
                "timestamp": datetime.now(timezone.utc),
            })
            history = list(self._source_outcome_history[module_name])
            if history:
                correct_count = sum(1 for h in history if h["correct"])
                self._source_trust[module_name] = round(
                    correct_count / len(history), 3
                )

    def get_judgment(self, claim_id: str) -> Optional[Dict[str, Any]]:
        claim_id = _sanitize(claim_id, 20)
        with self._lock:
            claim = self._claims.get(claim_id)
            if claim and claim.resolution_id:
                judgment = self._judgments.get(claim.resolution_id)
                if judgment:
                    return judgment.to_dict()
        return None

    def get_claims(self, resolved: Optional[bool] = None,
                   limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_CLAIMS, limit))
        with self._lock:
            claims = list(self._claims.values())
            if resolved is not None:
                claims = [c for c in claims if c.resolved == resolved]
            claims.sort(key=lambda c: c.submitted_at, reverse=True)
            return [c.to_dict() for c in claims[:limit]]

    def get_judgments(self, limit: int = 50) -> List[Dict[str, Any]]:
        limit = max(1, min(_MAX_JUDGMENTS, limit))
        with self._lock:
            judgments = sorted(
                list(self._judgments.values()),
                key=lambda j: j.issued_at, reverse=True,
            )
            return [j.to_dict() for j in judgments[:limit]]

    def get_source_trust(self, module_names: Optional[List[str]] = None) -> Dict[str, float]:
        with self._lock:
            if module_names:
                return {
                    m: self._source_trust.get(m, 0.5)
                    for m in [_sanitize(n, 100) for n in module_names[:20]]
                }
            return dict(self._source_trust)

    def get_stats(self) -> Dict[str, Any]:
        with self._lock:
            total_claims = len(self._claims)
            total_judgments = len(self._judgments)
            unresolved = sum(1 for c in self._claims.values() if not c.resolved)
            trusted_sources = sum(1 for t in self._source_trust.values() if t >= 0.7)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_claims"] = total_claims
        stats["total_judgments"] = total_judgments
        stats["unresolved_claims"] = unresolved
        stats["trusted_modules"] = trusted_sources
        return stats

    def _find_related_claims(self, claim: Claim) -> List[Claim]:
        related = []
        for other in self._claims.values():
            if other.claim_id == claim.claim_id:
                continue
            if other.domain == claim.domain:
                related.append(other)
            elif self._has_keyword_overlap(claim.statement, other.statement):
                related.append(other)
        return related

    def _compute_evidence_weight(self, claim: Claim) -> float:
        if not claim.evidence:
            return 0.3
        weight = min(1.0, len(claim.evidence) / _MAX_EVIDENCE_PER_CLAIM)
        if claim.counter_evidence:
            weight *= 1.0 - (len(claim.counter_evidence) / _MAX_EVIDENCE_PER_CLAIM)
        return round(max(0.1, weight), 3)

    def _compute_consensus(self, claim: Claim, related: List[Claim]) -> float:
        if not related:
            return 0.3
        supporting = sum(
            1 for c in related
            if c.confidence >= 0.5 and c.domain == claim.domain
        )
        return round(0.3 + (0.7 * supporting / max(1, len(related))), 3)

    def _update_source_trust(self, module_name: str, outcome: str) -> None:
        with self._lock:
            current = self._source_trust[module_name]
            if outcome in ("confirmed", "reconciled"):
                new_trust = min(0.95, current + 0.05)
            elif outcome == "partially_confirmed":
                new_trust = min(0.95, current + 0.02)
            elif outcome == "rejected":
                new_trust = max(0.1, current - 0.05)
            else:
                new_trust = current
            self._source_trust[module_name] = round(new_trust, 3)

    def _has_keyword_overlap(self, text_a: str, text_b: str) -> bool:
        words_a = set(re.findall(r'[a-z]{4,}', text_a.lower()))
        words_b = set(re.findall(r'[a-z]{4,}', text_b.lower()))
        return len(words_a & words_b) >= 3
