#!/usr/bin/env python3
"""
DecisionJournal — Decision Logging, Outcome Tracking, and Bias Retrospectives

Records decisions made by the cognitive system, tracks their outcomes over time,
and performs retrospective bias analysis to identify systematic decision-making flaws.

Architecture:
  - DecisionRecord: a single decision with alternatives, reasoning, confidence
  - BiasRetrospective: analysis of decisions over a time period for bias patterns
  - DecisionPattern: recurring decision-making patterns detected
  - DecisionJournal: main engine managing decisions, outcomes, and analysis
  - Thread-safe, bounded history, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.decision_journal")


class DecisionType(str, Enum):
    STRATEGIC = "strategic"
    TACTICAL = "tactical"
    OPERATIONAL = "operational"
    ETHICAL = "ethical"
    SOCIAL = "social"
    CREATIVE = "creative"
    ROUTINE = "routine"


class DecisionStatus(str, Enum):
    PENDING = "pending"
    EXECUTING = "executing"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


@dataclass
class DecisionRecord:
    decision_id: str
    title: str
    description: str
    alternatives: List[str]
    chosen: str
    reasoning: str
    confidence: float
    biases_flagged: List[str]
    domain: str
    decision_type: DecisionType
    status: DecisionStatus
    outcome: Optional[str]
    outcome_accuracy: Optional[float]
    outcome_recorded_at: Optional[str]
    created_at: str
    metadata: Dict

    def to_dict(self) -> Dict:
        return {
            "decision_id": self.decision_id,
            "title": self.title,
            "description": self.description,
            "alternatives": list(self.alternatives),
            "chosen": self.chosen,
            "reasoning": self.reasoning,
            "confidence": round(self.confidence, 3),
            "biases_flagged": list(self.biases_flagged),
            "domain": self.domain,
            "decision_type": self.decision_type.value,
            "status": self.status.value,
            "outcome": self.outcome,
            "outcome_accuracy": round(self.outcome_accuracy, 4) if self.outcome_accuracy is not None else None,
            "outcome_recorded_at": self.outcome_recorded_at,
            "created_at": self.created_at,
            "metadata": copy.deepcopy(dict(list(self.metadata.items())[:20])),
        }


@dataclass
class BiasRetrospective:
    retro_id: str
    period_start: str
    period_end: str
    decisions_reviewed: int
    biases_found: Dict[str, int]
    accuracy_avg: float
    overconfidence_rate: float
    findings: List[str]
    recommendations: List[str]
    created_at: str

    def to_dict(self) -> Dict:
        return {
            "retro_id": self.retro_id,
            "period_start": self.period_start,
            "period_end": self.period_end,
            "decisions_reviewed": self.decisions_reviewed,
            "biases_found": dict(self.biases_found),
            "accuracy_avg": round(self.accuracy_avg, 4),
            "overconfidence_rate": round(self.overconfidence_rate, 4),
            "findings": list(self.findings),
            "recommendations": list(self.recommendations),
            "created_at": self.created_at,
        }


@dataclass
class DecisionPattern:
    pattern_id: str
    pattern_type: str
    description: str
    frequency: int
    confidence: float
    domains: List[str]
    created_at: str

    def to_dict(self) -> Dict:
        return {
            "pattern_id": self.pattern_id,
            "pattern_type": self.pattern_type,
            "description": self.description,
            "frequency": self.frequency,
            "confidence": round(self.confidence, 3),
            "domains": list(self.domains),
            "created_at": self.created_at,
        }


class DecisionJournal:
    _MAX_DECISIONS = 5000
    _MAX_RETROSPECTIVES = 200
    _MAX_PATTERNS = 500
    _MAX_METADATA_KEYS = 20
    _MAX_ALTERNATIVES = 10
    _MAX_BIASES = 10
    _MAX_DOMAINS_PER_PATTERN = 5

    def __init__(self):
        self._lock = threading.Lock()
        self._decisions: Dict[str, DecisionRecord] = {}
        self._decision_order: List[str] = []
        self._retrospectives: Dict[str, BiasRetrospective] = {}
        self._retro_order: List[str] = []
        self._patterns: Dict[str, DecisionPattern] = {}
        self._pattern_order: List[str] = []
        self._stats = {
            "decisions_recorded": 0,
            "outcomes_recorded": 0,
            "retrospectives_run": 0,
            "patterns_detected": 0,
        }
        self._stats_lock = threading.Lock()

    def record_decision(
        self,
        title: str,
        chosen: str,
        alternatives: Optional[List[str]] = None,
        reasoning: str = "",
        confidence: float = 0.5,
        biases_flagged: Optional[List[str]] = None,
        domain: str = "",
        decision_type: str = "routine",
        description: str = "",
        metadata: Optional[Dict] = None,
    ) -> DecisionRecord:
        title = _sanitize(title, 200)
        chosen = _sanitize_content(chosen, 200)
        description = _sanitize_content(description, 2000)
        reasoning = _sanitize_content(reasoning, 2000)
        domain = _sanitize(domain, 100)

        if alternatives is None:
            alternatives = []
        alternatives = [_sanitize_content(a, 200) for a in alternatives[:self._MAX_ALTERNATIVES]]

        confidence = max(0.0, min(1.0, float(confidence)))

        if biases_flagged is None:
            biases_flagged = []
        biases_flagged = [_sanitize(b, 50) for b in biases_flagged[:self._MAX_BIASES]]

        try:
            dt_enum = DecisionType(decision_type)
        except ValueError:
            dt_enum = DecisionType.ROUTINE

        if metadata is None:
            metadata = {}
        metadata = {k: v for k, v in list(metadata.items())[:self._MAX_METADATA_KEYS]}
        clean_meta = {}
        for k, v in metadata.items():
            if isinstance(v, str):
                clean_meta[k] = _sanitize_content(v, 500)
            elif isinstance(v, (int, float, bool)):
                clean_meta[k] = v
            else:
                clean_meta[k] = _sanitize_content(str(v), 500)
        metadata = clean_meta

        decision_id = f"dec_{uuid.uuid4().hex[:12]}"
        now = datetime.now(timezone.utc).isoformat()

        record = DecisionRecord(
            decision_id=decision_id,
            title=title,
            description=description,
            alternatives=alternatives,
            chosen=chosen,
            reasoning=reasoning,
            confidence=confidence,
            biases_flagged=biases_flagged,
            domain=domain,
            decision_type=dt_enum,
            status=DecisionStatus.PENDING,
            outcome=None,
            outcome_accuracy=None,
            outcome_recorded_at=None,
            created_at=now,
            metadata=metadata,
        )

        with self._lock:
            if len(self._decisions) >= self._MAX_DECISIONS:
                evict_id = self._decision_order.pop(0)
                self._decisions.pop(evict_id, None)
                logger.warning("Evicting decision %s (cap=%d)", evict_id, self._MAX_DECISIONS)
            self._decisions[decision_id] = record
            self._decision_order.append(decision_id)
            with self._stats_lock:
                self._stats["decisions_recorded"] += 1

        return copy.deepcopy(record)

    def update_status(self, decision_id: str, status: str) -> bool:
        try:
            status_enum = DecisionStatus(status)
        except ValueError:
            return False

        with self._lock:
            rec = self._decisions.get(decision_id)
            if rec is None:
                return False
            rec.status = status_enum
            return True

    def record_outcome(self, decision_id: str, outcome: str, accuracy: float) -> bool:
        outcome = _sanitize_content(outcome, 2000)
        accuracy = max(0.0, min(1.0, float(accuracy)))

        with self._lock:
            rec = self._decisions.get(decision_id)
            if rec is None:
                return False
            rec.outcome = outcome
            rec.outcome_accuracy = accuracy
            rec.outcome_recorded_at = datetime.now(timezone.utc).isoformat()
            if accuracy >= 0.5:
                rec.status = DecisionStatus.COMPLETED
            else:
                rec.status = DecisionStatus.FAILED
            with self._stats_lock:
                self._stats["outcomes_recorded"] += 1
            return True

    def run_retrospective(self, period_days: int = 30) -> BiasRetrospective:
        period_days = max(1, min(3650, period_days))
        now_dt = datetime.now(timezone.utc)
        period_start_dt = now_dt - timedelta(days=period_days)
        period_start = period_start_dt.isoformat()
        period_end = now_dt.isoformat()

        with self._lock:
            recent = [
                copy.deepcopy(rec) for rec in self._decisions.values()
                if rec.outcome_accuracy is not None and rec.outcome_recorded_at is not None
                and rec.status in (DecisionStatus.COMPLETED, DecisionStatus.FAILED)
            ]

        bias_counts: Dict[str, int] = {}
        accuracies = []
        overconfident = 0
        total_with_outcome = 0

        for rec in recent:
            total_with_outcome += 1
            accuracies.append(rec.outcome_accuracy)
            if rec.confidence > rec.outcome_accuracy:
                overconfident += 1
            for b in rec.biases_flagged:
                bias_counts[b] = bias_counts.get(b, 0) + 1

        avg_accuracy = sum(accuracies) / len(accuracies) if accuracies else 0.5
        overconfidence_rate = overconfident / total_with_outcome if total_with_outcome else 0.0

        findings = []
        recommendations = []

        if overconfidence_rate > 0.5:
            findings.append(f"Overconfidence detected: {overconfidence_rate:.0%} of decisions had higher confidence than accuracy")
            recommendations.append("Consider dampening confidence estimates by 10-20%")

        sorted_biases = sorted(bias_counts.items(), key=lambda x: x[1], reverse=True)
        for bias_name, count in sorted_biases[:5]:
            findings.append(f"Bias '{bias_name}' appeared in {count} decisions")
            recommendations.append(f"Add counter-measures for {bias_name} bias")

        if avg_accuracy < 0.5:
            findings.append(f"Below-average accuracy: {avg_accuracy:.2f}")
            recommendations.append("Review decision-making criteria and evidence thresholds")

        domain_acc: Dict[str, List[float]] = {}
        for rec in recent:
            if rec.domain:
                domain_acc.setdefault(rec.domain, []).append(rec.outcome_accuracy)
        for dom, accs in domain_acc.items():
            dom_avg = sum(accs) / len(accs)
            if dom_avg < 0.4:
                findings.append(f"Low accuracy in domain '{dom}': {dom_avg:.2f}")
                recommendations.append(f"Seek more evidence or second opinions in {dom} decisions")

        findings = findings[:10]
        recommendations = recommendations[:10]

        retro_id = f"retro_{uuid.uuid4().hex[:12]}"
        retro = BiasRetrospective(
            retro_id=retro_id,
            period_start=period_start,
            period_end=period_end,
            decisions_reviewed=total_with_outcome,
            biases_found=bias_counts,
            accuracy_avg=round(avg_accuracy, 4),
            overconfidence_rate=round(overconfidence_rate, 4),
            findings=findings,
            recommendations=recommendations,
            created_at=now_dt.isoformat(),
        )

        with self._lock:
            if len(self._retrospectives) >= self._MAX_RETROSPECTIVES:
                evict_id = self._retro_order.pop(0)
                self._retrospectives.pop(evict_id, None)
                logger.warning("Evicting retrospective %s (cap=%d)", evict_id, self._MAX_RETROSPECTIVES)
            self._retrospectives[retro_id] = retro
            self._retro_order.append(retro_id)
            with self._stats_lock:
                self._stats["retrospectives_run"] += 1

        return copy.deepcopy(retro)

    def detect_patterns(self, min_frequency: int = 3) -> List[DecisionPattern]:
        min_frequency = max(1, min_frequency)
        patterns = []

        with self._lock:
            decisions_snapshot = [copy.deepcopy(rec) for rec in self._decisions.values()]

        domain_confidence: Dict[str, List[float]] = {}
        domain_accuracy: Dict[str, List[float]] = {}
        chosen_count: Dict[str, int] = {}
        type_confidence: Dict[str, List[float]] = {}
        type_accuracy: Dict[str, List[float]] = {}

        for rec in decisions_snapshot:
            domain_key = rec.domain or "unknown"
            domain_confidence.setdefault(domain_key, []).append(rec.confidence)
            if rec.outcome_accuracy is not None:
                domain_accuracy.setdefault(domain_key, []).append(rec.outcome_accuracy)
            chosen_key = f"chosen:{rec.chosen[:50]}"
            chosen_count[chosen_key] = chosen_count.get(chosen_key, 0) + 1
            type_key = rec.decision_type.value
            type_confidence.setdefault(type_key, []).append(rec.confidence)
            if rec.outcome_accuracy is not None:
                type_accuracy.setdefault(type_key, []).append(rec.outcome_accuracy)

        for domain, confs in domain_confidence.items():
            accs = domain_accuracy.get(domain, [])
            if len(confs) >= min_frequency and accs:
                avg_conf = sum(confs) / len(confs)
                avg_acc = sum(accs) / len(accs)
                if avg_conf - avg_acc > 0.2:
                    pid = f"dpat_{uuid.uuid4().hex[:12]}"
                    patterns.append(DecisionPattern(
                        pattern_id=pid,
                        pattern_type="consistently_overconfident",
                        description=f"Overconfident in domain '{domain}': avg confidence {avg_conf:.2f} vs accuracy {avg_acc:.2f}",
                        frequency=len(confs),
                        confidence=min(1.0, (avg_conf - avg_acc) * 2.5),
                        domains=[domain][:self._MAX_DOMAINS_PER_PATTERN],
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))

        for dt, confs in type_confidence.items():
            accs = type_accuracy.get(dt, [])
            if len(confs) >= min_frequency and accs:
                avg_conf = sum(confs) / len(confs)
                avg_acc = sum(accs) / len(accs)
                if avg_conf - avg_acc > 0.2:
                    pid = f"dpat_{uuid.uuid4().hex[:12]}"
                    patterns.append(DecisionPattern(
                        pattern_id=pid,
                        pattern_type="consistently_overconfident",
                        description=f"Overconfident in type '{dt}': avg confidence {avg_conf:.2f} vs accuracy {avg_acc:.2f}",
                        frequency=len(confs),
                        confidence=min(1.0, (avg_conf - avg_acc) * 2.5),
                        domains=[],
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))

        for chosen_key, count in chosen_count.items():
            if count >= min_frequency:
                pid = f"dpat_{uuid.uuid4().hex[:12]}"
                patterns.append(DecisionPattern(
                    pattern_id=pid,
                    pattern_type="repeated_choice",
                    description=f"Repeatedly chooses '{chosen_key}' ({count} times)",
                    frequency=count,
                    confidence=min(1.0, count / len(decisions_snapshot)) if decisions_snapshot else 0.0,
                    domains=[],
                    created_at=datetime.now(timezone.utc).isoformat(),
                ))

        for domain, confs in domain_confidence.items():
            if len(confs) >= min_frequency:
                avg_conf = sum(confs) / len(confs)
                if avg_conf < 0.3:
                    pid = f"dpat_{uuid.uuid4().hex[:12]}"
                    patterns.append(DecisionPattern(
                        pattern_id=pid,
                        pattern_type="domain_bias",
                        description=f"Consistently low confidence in domain '{domain}': {avg_conf:.2f}",
                        frequency=len(confs),
                        confidence=1.0 - avg_conf,
                        domains=[domain][:self._MAX_DOMAINS_PER_PATTERN],
                        created_at=datetime.now(timezone.utc).isoformat(),
                    ))

        with self._lock:
            for p in patterns:
                if len(self._patterns) >= self._MAX_PATTERNS:
                    evict_id = self._pattern_order.pop(0)
                    self._patterns.pop(evict_id, None)
                    logger.warning("Evicting pattern %s (cap=%d)", evict_id, self._MAX_PATTERNS)
                self._patterns[p.pattern_id] = p
                self._pattern_order.append(p.pattern_id)
            with self._stats_lock:
                self._stats["patterns_detected"] = len(self._patterns)

        return [copy.deepcopy(p) for p in patterns]

    def get_decision(self, decision_id: str) -> Optional[DecisionRecord]:
        with self._lock:
            rec = self._decisions.get(decision_id)
            if rec is None:
                return None
            return copy.deepcopy(rec)

    def list_decisions(
        self,
        domain: Optional[str] = None,
        status: Optional[str] = None,
        limit: int = 50,
    ) -> List[DecisionRecord]:
        limit = max(1, min(200, limit))
        try:
            status_enum = DecisionStatus(status) if status else None
        except ValueError:
            status_enum = None

        with self._lock:
            results = []
            for did in reversed(self._decision_order):
                rec = self._decisions.get(did)
                if rec is None:
                    continue
                if domain and rec.domain != domain:
                    continue
                if status_enum and rec.status != status_enum:
                    continue
                results.append(copy.deepcopy(rec))
                if len(results) >= limit:
                    break
            return results

    def get_decision_accuracy(self, domain: Optional[str] = None) -> Dict:
        with self._lock:
            resolved = [
                copy.deepcopy(rec) for rec in self._decisions.values()
                if rec.outcome_accuracy is not None
            ]

        if domain:
            resolved = [r for r in resolved if r.domain == domain]

        if not resolved:
            return {
                "total_resolved": 0,
                "avg_accuracy": 0.0,
                "avg_confidence": 0.0,
                "overconfidence_count": 0,
                "domain": domain,
            }

        accuracies = [r.outcome_accuracy for r in resolved]
        confidences = [r.confidence for r in resolved]
        overconfident = sum(1 for r in resolved if r.confidence > r.outcome_accuracy)

        return {
            "total_resolved": len(resolved),
            "avg_accuracy": round(sum(accuracies) / len(accuracies), 4),
            "avg_confidence": round(sum(confidences) / len(confidences), 4),
            "overconfidence_count": overconfident,
            "domain": domain,
        }

    def get_overconfidence_report(self) -> Dict:
        with self._lock:
            resolved = [
                copy.deepcopy(rec) for rec in self._decisions.values()
                if rec.outcome_accuracy is not None
            ]

        if not resolved:
            return {
                "total": 0,
                "overconfident_count": 0,
                "overconfidence_rate": 0.0,
                "avg_overconfidence_gap": 0.0,
                "by_domain": {},
                "by_type": {},
            }

        overconfident = [r for r in resolved if r.confidence > r.outcome_accuracy]
        gaps = [r.confidence - r.outcome_accuracy for r in overconfident]

        by_domain: Dict[str, Dict] = {}
        for r in resolved:
            d = r.domain or "unknown"
            if d not in by_domain:
                by_domain[d] = {"total": 0, "overconfident": 0}
            by_domain[d]["total"] += 1
            if r.confidence > r.outcome_accuracy:
                by_domain[d]["overconfident"] += 1

        by_type: Dict[str, Dict] = {}
        for r in resolved:
            t = r.decision_type.value
            if t not in by_type:
                by_type[t] = {"total": 0, "overconfident": 0}
            by_type[t]["total"] += 1
            if r.confidence > r.outcome_accuracy:
                by_type[t]["overconfident"] += 1

        return {
            "total": len(resolved),
            "overconfident_count": len(overconfident),
            "overconfidence_rate": round(len(overconfident) / len(resolved), 4),
            "avg_overconfidence_gap": round(sum(gaps) / len(gaps), 4) if gaps else 0.0,
            "by_domain": by_domain,
            "by_type": by_type,
        }

    def get_stats(self) -> Dict:
        with self._stats_lock:
            decisions_recorded = self._stats["decisions_recorded"]
            outcomes_recorded = self._stats["outcomes_recorded"]
            retrospectives_run = self._stats["retrospectives_run"]
            patterns_detected = self._stats["patterns_detected"]
        with self._lock:
            current_decisions = len(self._decisions)
            current_retrospectives = len(self._retrospectives)
            current_patterns = len(self._patterns)
        return {
            "decisions_recorded": decisions_recorded,
            "outcomes_recorded": outcomes_recorded,
            "retrospectives_run": retrospectives_run,
            "patterns_detected": patterns_detected,
            "current_decisions": current_decisions,
            "current_retrospectives": current_retrospectives,
            "current_patterns": current_patterns,
        }