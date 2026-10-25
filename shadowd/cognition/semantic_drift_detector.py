#!/usr/bin/env python3
"""
SemanticDriftDetector -- Concept Evolution and Meaning Shift Tracker.

Tracks how the meaning of concepts evolves over time. Detects semantic
drift -- when a concept's meaning shifts, narrows, broadens, or diverges.
Computes drift magnitude from definition word overlap and association
changes, producing a full lineage of concept evolution.

Key concepts:
  - ConceptSnapshot: point-in-time record of a concept's definition and context
  - DriftEvent: detected shift between two snapshots
  - ConceptLineage: full evolution history of a single concept
  - DriftReport: aggregate drift status across all tracked concepts

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

logger = logging.getLogger("shadowd.cognition.semantic_drift_detector")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_CONCEPTS = 2000
_MAX_SNAPSHOTS = 50
_MAX_DRIFT_EVENTS = 5000
_MAX_ASSOCIATIONS = 100


class DriftType(str, Enum):
    NARROWING = "narrowing"
    BROADENING = "broadening"
    SHIFTING = "shifting"
    DIVERGING = "diverging"
    MERGING = "merging"
    SPLITTING = "splitting"
    AMELIORATION = "amelioration"
    PEJORATION = "pejoration"


class ConceptStatus(str, Enum):
    ACTIVE = "active"
    ARCHIVED = "archived"
    MERGED = "merged"
    SPLIT = "split"
    DEPRECATED = "deprecated"


class EvidenceType(str, Enum):
    USAGE = "usage"
    CONTEXT = "context"
    ASSOCIATION = "association"
    CONTRADICTION = "contradiction"
    CORROBORATION = "corroboration"


def _sanitize(text: str, max_len: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


def _sanitize_content(text: str, max_len: int = 5000) -> str:
    if not isinstance(text, str):
        text = str(text)
    return _SANITIZE_CONTENT_RE.sub('', text.strip())[:max_len]


def _clamp(value: float, lo: float = 0.0, hi: float = 1.0) -> float:
    return max(lo, min(hi, value))


def _word_set(text: str) -> set:
    return set(re.findall(r'[a-zA-Z0-9]+', text.lower()))


def _jaccard(a: set, b: set) -> float:
    if not a and not b:
        return 1.0
    if not a or not b:
        return 0.0
    return len(a & b) / len(a | b)


@dataclass
class ConceptSnapshot:
    snapshot_id: str = ""
    concept: str = ""
    definition: str = ""
    context: str = ""
    associations: List[str] = field(default_factory=list)
    confidence: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.snapshot_id:
            self.snapshot_id = "snap_%s" % uuid.uuid4().hex[:12]
        self.snapshot_id = _sanitize(self.snapshot_id, 64)
        self.concept = _sanitize(self.concept, 200)
        self.definition = _sanitize_content(self.definition, 10000)
        self.context = _sanitize_content(self.context, 5000)
        self.associations = [_sanitize(a, 200) for a in (self.associations or [])[:_MAX_ASSOCIATIONS]]
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "snapshot_id": self.snapshot_id,
            "concept": self.concept,
            "definition": self.definition,
            "context": self.context,
            "associations": list(self.associations),
            "confidence": round(self.confidence, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class DriftEvent:
    drift_id: str = ""
    concept: str = ""
    drift_type: DriftType = DriftType.SHIFTING
    from_definition: str = ""
    to_definition: str = ""
    drift_magnitude: float = 0.0
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.5

    def __post_init__(self):
        if not self.drift_id:
            self.drift_id = "drift_%s" % uuid.uuid4().hex[:12]
        self.drift_id = _sanitize(self.drift_id, 64)
        self.concept = _sanitize(self.concept, 200)
        if isinstance(self.drift_type, str):
            try:
                self.drift_type = DriftType(self.drift_type)
            except ValueError:
                self.drift_type = DriftType.SHIFTING
        self.from_definition = _sanitize_content(self.from_definition, 10000)
        self.to_definition = _sanitize_content(self.to_definition, 10000)
        self.drift_magnitude = _clamp(self.drift_magnitude)
        self.evidence = [_sanitize_content(e, 500) for e in (self.evidence or [])[:20]]
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "drift_id": self.drift_id,
            "concept": self.concept,
            "drift_type": self.drift_type.value,
            "from_definition": self.from_definition,
            "to_definition": self.to_definition,
            "drift_magnitude": round(self.drift_magnitude, 4),
            "evidence": list(self.evidence),
            "confidence": round(self.confidence, 4),
        }


@dataclass
class ConceptLineage:
    lineage_id: str = ""
    concept: str = ""
    snapshots: List[str] = field(default_factory=list)
    drift_events: List[str] = field(default_factory=list)
    current_status: ConceptStatus = ConceptStatus.ACTIVE

    def __post_init__(self):
        if not self.lineage_id:
            self.lineage_id = "lin_%s" % uuid.uuid4().hex[:12]
        self.lineage_id = _sanitize(self.lineage_id, 64)
        self.concept = _sanitize(self.concept, 200)
        self.snapshots = [_sanitize(s, 64) for s in (self.snapshots or [])[:_MAX_SNAPSHOTS]]
        self.drift_events = [_sanitize(d, 64) for d in (self.drift_events or [])[:200]]
        if isinstance(self.current_status, str):
            try:
                self.current_status = ConceptStatus(self.current_status)
            except ValueError:
                self.current_status = ConceptStatus.ACTIVE

    def to_dict(self) -> Dict:
        return {
            "lineage_id": self.lineage_id,
            "concept": self.concept,
            "snapshots": list(self.snapshots),
            "drift_events": list(self.drift_events),
            "current_status": self.current_status.value,
        }


@dataclass
class DriftReport:
    report_id: str = ""
    concepts_count: int = 0
    drift_events_count: int = 0
    avg_drift_rate: float = 0.0
    most_drifted: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = "rpt_%s" % uuid.uuid4().hex[:12]
        self.report_id = _sanitize(self.report_id, 64)
        self.avg_drift_rate = _clamp(self.avg_drift_rate)
        self.most_drifted = [_sanitize(c, 200) for c in (self.most_drifted or [])[:50]]

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "concepts_count": self.concepts_count,
            "drift_events_count": self.drift_events_count,
            "avg_drift_rate": round(self.avg_drift_rate, 4),
            "most_drifted": list(self.most_drifted),
        }


class SemanticDriftDetector:
    """Concept Evolution and Meaning Shift Tracker.

    Registers concepts, records usage sightings, and detects semantic
    drift when definitions or associations shift beyond thresholds.
    Maintains full lineage history per concept.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._concepts: Dict[str, ConceptSnapshot] = {}
        self._concept_order: deque = deque(maxlen=_MAX_CONCEPTS)
        self._lineages: Dict[str, ConceptLineage] = {}
        self._snapshots: Dict[str, ConceptSnapshot] = {}
        self._snapshot_order: deque = deque(maxlen=_MAX_SNAPSHOTS * _MAX_CONCEPTS)
        self._drift_events: Dict[str, DriftEvent] = {}
        self._drift_order: deque = deque(maxlen=_MAX_DRIFT_EVENTS)
        self._stats = {
            "concepts_registered": 0,
            "usages_recorded": 0,
            "drifts_detected": 0,
            "merges_performed": 0,
            "splits_performed": 0,
            "snapshots_created": 0,
        }

    def register_concept(self, concept: str, definition: str,
                         context: str = "", associations: Optional[List[str]] = None) -> ConceptSnapshot:
        concept = _sanitize(concept, 200)
        definition = _sanitize_content(definition, 10000)
        context = _sanitize_content(context, 5000)
        if associations is None:
            associations = []
        associations = [_sanitize(a, 200) for a in associations[:_MAX_ASSOCIATIONS]]

        snapshot = ConceptSnapshot(
            snapshot_id="snap_%s" % uuid.uuid4().hex[:12],
            concept=concept,
            definition=definition,
            context=context,
            associations=associations,
            confidence=0.5,
        )

        with self._lock:
            if concept in self._concepts:
                existing = self._concepts[concept]
                new_snapshot = ConceptSnapshot(
                    snapshot_id="snap_%s" % uuid.uuid4().hex[:12],
                    concept=concept,
                    definition=definition,
                    context=context,
                    associations=associations,
                    confidence=0.5,
                )
                self._store_snapshot_unlocked(new_snapshot)
                lineage = self._lineages.get(concept)
                if lineage:
                    lineage.snapshots.append(new_snapshot.snapshot_id)
                    if len(lineage.snapshots) > _MAX_SNAPSHOTS:
                        evicted = lineage.snapshots.pop(0)
                        self._snapshots.pop(evicted, None)
                        logger.warning("Evicting snapshot %s for concept %s (cap=%d)", evicted, concept, _MAX_SNAPSHOTS)
                self._concepts[concept] = new_snapshot
                with self._stats_lock:
                    self._stats["snapshots_created"] += 1
                return copy.deepcopy(new_snapshot)

            if len(self._concepts) >= _MAX_CONCEPTS:
                evict_concept = self._concept_order.popleft()
                self._concepts.pop(evict_concept, None)
                self._lineages.pop(evict_concept, None)
                logger.warning("Evicting concept %s (cap=%d)", evict_concept, _MAX_CONCEPTS)

            self._concepts[concept] = snapshot
            self._concept_order.append(concept)
            lineage = ConceptLineage(
                lineage_id="lin_%s" % uuid.uuid4().hex[:12],
                concept=concept,
                snapshots=[snapshot.snapshot_id],
                drift_events=[],
                current_status=ConceptStatus.ACTIVE,
            )
            self._lineages[concept] = lineage
            self._store_snapshot_unlocked(snapshot)
            with self._stats_lock:
                self._stats["concepts_registered"] += 1
                self._stats["snapshots_created"] += 1

        return copy.deepcopy(snapshot)

    def record_usage(self, concept: str, context: str = "",
                     definition_hint: str = "") -> Optional[ConceptSnapshot]:
        concept = _sanitize(concept, 200)
        context = _sanitize_content(context, 5000)
        definition_hint = _sanitize_content(definition_hint, 10000)

        with self._lock:
            if concept not in self._concepts:
                return None

            current = self._concepts[concept]

            if definition_hint:
                new_snapshot = ConceptSnapshot(
                    snapshot_id="snap_%s" % uuid.uuid4().hex[:12],
                    concept=concept,
                    definition=definition_hint,
                    context=context or current.context,
                    associations=list(current.associations),
                    confidence=_clamp(current.confidence + 0.05),
                )
                self._store_snapshot_unlocked(new_snapshot)
                self._concepts[concept] = new_snapshot

                lineage = self._lineages.get(concept)
                if lineage:
                    lineage.snapshots.append(new_snapshot.snapshot_id)
                    if len(lineage.snapshots) > _MAX_SNAPSHOTS:
                        evicted = lineage.snapshots.pop(0)
                        self._snapshots.pop(evicted, None)
                        logger.warning("Evicting snapshot %s for concept %s (cap=%d)", evicted, concept, _MAX_SNAPSHOTS)

                with self._stats_lock:
                    self._stats["snapshots_created"] += 1

            with self._stats_lock:
                self._stats["usages_recorded"] += 1

            return copy.deepcopy(self._concepts[concept])

    def detect_drift(self, concept: str, min_magnitude: float = 0.1) -> Optional[DriftEvent]:
        concept = _sanitize(concept, 200)
        min_magnitude = _clamp(min_magnitude)

        with self._lock:
            if concept not in self._concepts:
                return None

            if concept not in self._lineages:
                return None

            lineage = self._lineages[concept]
            if len(lineage.snapshots) < 2:
                return None

            prev_id = lineage.snapshots[-2]
            curr_id = lineage.snapshots[-1]

            prev_snap = self._snapshots.get(prev_id)
            curr_snap = self._snapshots.get(curr_id)
            if not prev_snap or not curr_snap:
                return None

            prev_words = _word_set(prev_snap.definition)
            curr_words = _word_set(curr_snap.definition)
            overlap = _jaccard(prev_words, curr_words)
            drift_magnitude = 1.0 - overlap

            prev_assoc = set(prev_snap.associations)
            curr_assoc = set(curr_snap.associations)
            assoc_overlap = _jaccard(prev_assoc, curr_assoc) if (prev_assoc or curr_assoc) else 1.0
            assoc_drift = 1.0 - assoc_overlap

            drift_magnitude = _clamp((drift_magnitude + assoc_drift) / 2.0)

            if drift_magnitude < min_magnitude:
                return None

            drift_type = self._classify_drift(prev_words, curr_words, prev_assoc, curr_assoc, overlap)

            evidence = []
            if drift_magnitude > 0.3:
                evidence.append("high definition divergence")
            if assoc_drift > 0.3:
                evidence.append("significant association change")
            lost_words = prev_words - curr_words
            if lost_words:
                evidence.append("lost terms: %s" % ", ".join(list(lost_words)[:5]))
            gained_words = curr_words - prev_words
            if gained_words:
                evidence.append("gained terms: %s" % ", ".join(list(gained_words)[:5]))

            drift = DriftEvent(
                drift_id="drift_%s" % uuid.uuid4().hex[:12],
                concept=concept,
                drift_type=drift_type,
                from_definition=prev_snap.definition,
                to_definition=curr_snap.definition,
                drift_magnitude=drift_magnitude,
                evidence=evidence,
                confidence=_clamp(min(prev_snap.confidence, curr_snap.confidence)),
            )

            if len(self._drift_events) >= _MAX_DRIFT_EVENTS:
                evict_id = self._drift_order.popleft()
                self._drift_events.pop(evict_id, None)
                logger.warning("Evicting drift event %s (cap=%d)", evict_id, _MAX_DRIFT_EVENTS)

            self._drift_events[drift.drift_id] = drift
            self._drift_order.append(drift.drift_id)
            lineage.drift_events.append(drift.drift_id)
            if len(lineage.drift_events) > 200:
                evicted_de = lineage.drift_events.pop(0)
                logger.warning("Evicting drift event ref %s from lineage %s", evicted_de, lineage.lineage_id)

            with self._stats_lock:
                self._stats["drifts_detected"] += 1

            return copy.deepcopy(drift)

    def get_lineage(self, concept: str) -> Optional[ConceptLineage]:
        concept = _sanitize(concept, 200)
        with self._lock:
            lineage = self._lineages.get(concept)
            if not lineage:
                return None
            return copy.deepcopy(lineage)

    def compare_concepts(self, concept_a: str, concept_b: str) -> Dict:
        a = _sanitize(concept_a, 200)
        b = _sanitize(concept_b, 200)
        with self._lock:
            snap_a = self._concepts.get(a)
            snap_b = self._concepts.get(b)
            if not snap_a or not snap_b:
                return {"distance": 1.0, "similarity": 0.0, "concept_a": a, "concept_b": b}

            words_a = _word_set(snap_a.definition)
            words_b = _word_set(snap_b.definition)
            definition_similarity = _jaccard(words_a, words_b)

            assoc_a = set(snap_a.associations)
            assoc_b = set(snap_b.associations)
            assoc_similarity = _jaccard(assoc_a, assoc_b) if (assoc_a or assoc_b) else 1.0

            context_a = _word_set(snap_a.context)
            context_b = _word_set(snap_b.context)
            context_similarity = _jaccard(context_a, context_b) if (context_a or context_b) else 0.5

            similarity = _clamp((definition_similarity * 0.5 + assoc_similarity * 0.3 + context_similarity * 0.2))
            distance = round(1.0 - similarity, 4)

            return {
                "distance": distance,
                "similarity": round(similarity, 4),
                "definition_similarity": round(definition_similarity, 4),
                "association_similarity": round(assoc_similarity, 4),
                "context_similarity": round(context_similarity, 4),
                "concept_a": a,
                "concept_b": b,
            }

    def get_drift_report(self) -> DriftReport:
        with self._lock:
            concepts_count = len(self._concepts)
            drift_events_count = len(self._drift_events)

            concept_drift = {}
            for drift in self._drift_events.values():
                c = drift.concept
                concept_drift[c] = concept_drift.get(c, 0.0) + drift.drift_magnitude

            if concept_drift:
                total_drift = sum(concept_drift.values())
                avg_drift_rate = _clamp(total_drift / max(concepts_count, 1))
            else:
                avg_drift_rate = 0.0

            sorted_concepts = sorted(concept_drift.items(), key=lambda x: x[1], reverse=True)
            most_drifted = [c for c, _ in sorted_concepts[:50]]

            report = DriftReport(
                report_id="rpt_%s" % uuid.uuid4().hex[:12],
                concepts_count=concepts_count,
                drift_events_count=drift_events_count,
                avg_drift_rate=avg_drift_rate,
                most_drifted=most_drifted,
            )

        return report

    def merge_concepts(self, concept_a: str, concept_b: str,
                       new_definition: str) -> Optional[ConceptSnapshot]:
        a = _sanitize(concept_a, 200)
        b = _sanitize(concept_b, 200)
        new_definition = _sanitize_content(new_definition, 10000)

        with self._lock:
            snap_a = self._concepts.get(a)
            snap_b = self._concepts.get(b)
            if not snap_a or not snap_b:
                return None

            merged_associations = list(set(snap_a.associations + snap_b.associations))[:_MAX_ASSOCIATIONS]
            merged_context = (snap_a.context + " " + snap_b.context).strip()[:5000]
            merged_confidence = _clamp((snap_a.confidence + snap_b.confidence) / 2.0)

            snapshot = ConceptSnapshot(
                snapshot_id="snap_%s" % uuid.uuid4().hex[:12],
                concept=a,
                definition=new_definition,
                context=merged_context,
                associations=merged_associations,
                confidence=merged_confidence,
            )

            self._store_snapshot_unlocked(snapshot)
            self._concepts[a] = snapshot

            lineage_a = self._lineages.get(a)
            if lineage_a:
                lineage_a.snapshots.append(snapshot.snapshot_id)
                if len(lineage_a.snapshots) > _MAX_SNAPSHOTS:
                    evicted = lineage_a.snapshots.pop(0)
                    self._snapshots.pop(evicted, None)
                    logger.warning("Evicting snapshot %s for concept %s (cap=%d)", evicted, a, _MAX_SNAPSHOTS)

            lineage_b = self._lineages.get(b)
            if lineage_b:
                lineage_b.current_status = ConceptStatus.MERGED

            self._concepts.pop(b, None)
            try:
                self._concept_order.remove(b)
            except ValueError:
                pass

            drift = DriftEvent(
                drift_id="drift_%s" % uuid.uuid4().hex[:12],
                concept=a,
                drift_type=DriftType.MERGING,
                from_definition=snap_a.definition,
                to_definition=new_definition,
                drift_magnitude=_clamp(1.0 - _jaccard(_word_set(snap_a.definition), _word_set(new_definition))),
                evidence=["merged with %s" % b],
                confidence=merged_confidence,
            )
            if len(self._drift_events) >= _MAX_DRIFT_EVENTS:
                evict_id = self._drift_order.popleft()
                self._drift_events.pop(evict_id, None)
                logger.warning("Evicting drift event %s (cap=%d)", evict_id, _MAX_DRIFT_EVENTS)
            self._drift_events[drift.drift_id] = drift
            self._drift_order.append(drift.drift_id)
            if lineage_a:
                lineage_a.drift_events.append(drift.drift_id)

            with self._stats_lock:
                self._stats["merges_performed"] += 1
                self._stats["drifts_detected"] += 1
                self._stats["snapshots_created"] += 1

            return copy.deepcopy(snapshot)

    def split_concept(self, concept: str, definitions: List[str]) -> List[ConceptSnapshot]:
        concept = _sanitize(concept, 200)
        definitions = [_sanitize_content(d, 10000) for d in (definitions or [])[:20]]

        results = []
        with self._lock:
            original = self._concepts.get(concept)
            if not original:
                return []

            lineage = self._lineages.get(concept)
            if lineage:
                lineage.current_status = ConceptStatus.SPLIT

            self._concepts.pop(concept, None)
            try:
                self._concept_order.remove(concept)
            except ValueError:
                pass

            for i, defn in enumerate(definitions):
                new_concept = "%s_%d" % (concept, i + 1)
                snapshot = ConceptSnapshot(
                    snapshot_id="snap_%s" % uuid.uuid4().hex[:12],
                    concept=new_concept,
                    definition=defn,
                    context=original.context if original else "",
                    associations=list(original.associations)[:_MAX_ASSOCIATIONS] if original else [],
                    confidence=_clamp(original.confidence * 0.8) if original else 0.3,
                )

                if len(self._concepts) >= _MAX_CONCEPTS:
                    evict_concept = self._concept_order.popleft()
                    self._concepts.pop(evict_concept, None)
                    self._lineages.pop(evict_concept, None)
                    logger.warning("Evicting concept %s (cap=%d)", evict_concept, _MAX_CONCEPTS)

                self._concepts[new_concept] = snapshot
                self._concept_order.append(new_concept)
                self._store_snapshot_unlocked(snapshot)

                new_lineage = ConceptLineage(
                    lineage_id="lin_%s" % uuid.uuid4().hex[:12],
                    concept=new_concept,
                    snapshots=[snapshot.snapshot_id],
                    drift_events=[],
                    current_status=ConceptStatus.ACTIVE,
                )
                self._lineages[new_concept] = new_lineage

                drift = DriftEvent(
                    drift_id="drift_%s" % uuid.uuid4().hex[:12],
                    concept=new_concept,
                    drift_type=DriftType.SPLITTING,
                    from_definition=original.definition if original else "",
                    to_definition=defn,
                    drift_magnitude=_clamp(1.0 - _jaccard(_word_set(original.definition if original else ""), _word_set(defn))),
                    evidence=["split from %s (sense %d)" % (concept, i + 1)],
                    confidence=snapshot.confidence,
                )
                if len(self._drift_events) >= _MAX_DRIFT_EVENTS:
                    evict_id = self._drift_order.popleft()
                    self._drift_events.pop(evict_id, None)
                    logger.warning("Evicting drift event %s (cap=%d)", evict_id, _MAX_DRIFT_EVENTS)
                self._drift_events[drift.drift_id] = drift
                self._drift_order.append(drift.drift_id)
                new_lineage.drift_events.append(drift.drift_id)
                results.append(copy.deepcopy(snapshot))

            with self._stats_lock:
                self._stats["splits_performed"] += 1
                self._stats["drifts_detected"] += len(definitions)
                self._stats["concepts_registered"] += len(definitions)
                self._stats["snapshots_created"] += len(definitions)

        return results

    def get_concept(self, concept_name: str) -> Optional[ConceptSnapshot]:
        concept_name = _sanitize(concept_name, 200)
        with self._lock:
            snapshot = self._concepts.get(concept_name)
            if not snapshot:
                return None
            return copy.deepcopy(snapshot)

    def list_concepts(self, status: Optional[ConceptStatus] = None) -> List[Dict]:
        if isinstance(status, str):
            try:
                status = ConceptStatus(status)
            except ValueError:
                status = None

        with self._lock:
            results = []
            for concept, snap in self._concepts.items():
                lineage = self._lineages.get(concept)
                curr_status = lineage.current_status if lineage else ConceptStatus.ACTIVE
                if status and curr_status != status:
                    continue
                results.append({
                    "concept": snap.concept,
                    "definition": snap.definition[:200],
                    "confidence": round(snap.confidence, 4),
                    "status": curr_status.value,
                    "snapshot_count": len(lineage.snapshots) if lineage else 0,
                    "drift_count": len(lineage.drift_events) if lineage else 0,
                })
        return results

    def get_stats(self) -> Dict:
        with self._lock:
            concepts_count = len(self._concepts)
            snapshots_count = len(self._snapshots)
            drift_count = len(self._drift_events)
            active = sum(1 for lin in self._lineages.values() if lin.current_status == ConceptStatus.ACTIVE)
            merged = sum(1 for lin in self._lineages.values() if lin.current_status == ConceptStatus.MERGED)
            split = sum(1 for lin in self._lineages.values() if lin.current_status == ConceptStatus.SPLIT)
            avg_conf = 0.0
            if self._concepts:
                avg_conf = sum(s.confidence for s in self._concepts.values()) / concepts_count
        with self._stats_lock:
            stats = dict(self._stats)
        stats["concepts_count"] = concepts_count
        stats["snapshots_count"] = snapshots_count
        stats["drift_events_count"] = drift_count
        stats["active_concepts"] = active
        stats["merged_concepts"] = merged
        stats["split_concepts"] = split
        stats["avg_confidence"] = round(avg_conf, 4)
        return stats

    def _store_snapshot_unlocked(self, snapshot: ConceptSnapshot) -> None:
        self._snapshots[snapshot.snapshot_id] = snapshot
        self._snapshot_order.append(snapshot.snapshot_id)

    def _classify_drift(self, prev_words: set, curr_words: set,
                        prev_assoc: set, curr_assoc: set,
                        overlap: float) -> DriftType:
        if not prev_words or not curr_words:
            return DriftType.SHIFTING

        ratio = len(curr_words) / len(prev_words) if prev_words else 1.0
        lost = prev_words - curr_words
        gained = curr_words - prev_words

        if ratio < 0.7 and len(lost) > len(gained):
            return DriftType.NARROWING
        elif ratio > 1.4 and len(gained) > len(lost):
            return DriftType.BROADENING
        elif len(lost) > 0 and len(gained) > 0 and overlap < 0.3:
            return DriftType.DIVERGING
        elif prev_assoc != curr_assoc and overlap < 0.5:
            return DriftType.SHIFTING
        elif overlap >= 0.5:
            return DriftType.AMELIORATION if len(gained) >= len(lost) else DriftType.PEJORATION
        else:
            return DriftType.SHIFTING