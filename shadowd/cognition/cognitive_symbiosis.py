#!/usr/bin/env python3
"""
CognitiveSymbiosis — Mutualistic Module Interaction Engine.

Models mutualistic relationships between cognitive modules, like biological
symbiosis. Detects and optimizes cross-module relationships to maximize
overall system performance. Supports mutualism, commensalism, parasitism,
competition, neutralism, and amensalism.

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
from enum import Enum, IntEnum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.cognitive_symbiosis")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_PAIRS = 500
_MAX_INTERACTIONS = 3000
_MAX_DETECTIONS = 100
_MAX_HISTORY = 1000
_MAX_METADATA_KEYS = 20
_MAX_EVIDENCE_ITEMS = 50
_MAX_UNHEALTHY_LIST = 50


class SymbiosisType(str, Enum):
    MUTUALISM = "mutualism"
    COMMENSALISM = "commensalism"
    PARASITISM = "parasitism"
    COMPETITION = "competition"
    NEUTRALISM = "neutralism"
    AMENSALISM = "amensalism"


class InteractionMode(str, Enum):
    FEED = "feed"
    TRIGGER = "trigger"
    INHIBIT = "inhibit"
    AMPLIFY = "amplify"
    DAMPEN = "dampen"
    TRANSFORM = "transform"


class RelationshipHealth(IntEnum):
    BROKEN = 0
    DETERIORATING = 1
    STRAINED = 2
    STABLE = 3
    THRIVING = 4


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
class SymbioticPair:
    pair_id: str
    module_a: str
    module_b: str
    symbiosis_type: SymbiosisType
    interaction_mode: InteractionMode
    strength_a_to_b: float = 0.5
    strength_b_to_a: float = 0.5
    health: int = RelationshipHealth.STABLE
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    last_interaction: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.pair_id:
            self.pair_id = uuid.uuid4().hex[:12]
        self.pair_id = _sanitize(self.pair_id, 50)
        self.module_a = _sanitize(self.module_a, 100)
        self.module_b = _sanitize(self.module_b, 100)
        if isinstance(self.symbiosis_type, str):
            try:
                self.symbiosis_type = SymbiosisType(self.symbiosis_type)
            except ValueError:
                self.symbiosis_type = SymbiosisType.NEUTRALISM
        if isinstance(self.interaction_mode, str):
            try:
                self.interaction_mode = InteractionMode(self.interaction_mode)
            except ValueError:
                self.interaction_mode = InteractionMode.TRIGGER
        self.strength_a_to_b = _clamp(self.strength_a_to_b)
        self.strength_b_to_a = _clamp(self.strength_b_to_a)
        self.health = max(0, min(4, int(self.health)))

    def to_dict(self) -> Dict:
        return {
            "pair_id": self.pair_id,
            "module_a": self.module_a,
            "module_b": self.module_b,
            "symbiosis_type": self.symbiosis_type.value,
            "interaction_mode": self.interaction_mode.value,
            "strength_a_to_b": round(self.strength_a_to_b, 4),
            "strength_b_to_a": round(self.strength_b_to_a, 4),
            "health": int(self.health),
            "created_at": self.created_at,
            "last_interaction": self.last_interaction,
        }


@dataclass
class SymbioticInteraction:
    interaction_id: str
    pair_id: str
    direction: str
    content: str
    outcome: str
    benefit_a: float = 0.5
    benefit_b: float = 0.5
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.interaction_id:
            self.interaction_id = uuid.uuid4().hex[:12]
        self.interaction_id = _sanitize(self.interaction_id, 50)
        self.pair_id = _sanitize(self.pair_id, 50)
        if isinstance(self.direction, InteractionMode):
            self.direction = self.direction.value
        self.direction = _sanitize(self.direction, 20)
        self.content = _sanitize_content(self.content, 2000)
        self.outcome = _sanitize_content(self.outcome, 2000)
        self.benefit_a = _clamp(self.benefit_a)
        self.benefit_b = _clamp(self.benefit_b)

    def to_dict(self) -> Dict:
        return {
            "interaction_id": self.interaction_id,
            "pair_id": self.pair_id,
            "direction": self.direction,
            "content": self.content,
            "outcome": self.outcome,
            "benefit_a": round(self.benefit_a, 4),
            "benefit_b": round(self.benefit_b, 4),
            "timestamp": self.timestamp,
        }


@dataclass
class SymbioticReport:
    report_id: str
    total_pairs: int = 0
    mutualism_count: int = 0
    commensalism_count: int = 0
    parasite_count: int = 0
    avg_system_benefit: float = 0.0
    unhealthy_pairs: List[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.report_id:
            self.report_id = uuid.uuid4().hex[:12]
        self.report_id = _sanitize(self.report_id, 50)
        self.total_pairs = max(0, self.total_pairs)
        self.mutualism_count = max(0, self.mutualism_count)
        self.commensalism_count = max(0, self.commensalism_count)
        self.parasite_count = max(0, self.parasite_count)
        self.avg_system_benefit = _clamp(self.avg_system_benefit)
        self.unhealthy_pairs = [_sanitize(p, 50) for p in self.unhealthy_pairs[:_MAX_UNHEALTHY_LIST]]

    def to_dict(self) -> Dict:
        return {
            "report_id": self.report_id,
            "total_pairs": self.total_pairs,
            "mutualism_count": self.mutualism_count,
            "commensalism_count": self.commensalism_count,
            "parasite_count": self.parasite_count,
            "avg_system_benefit": round(self.avg_system_benefit, 4),
            "unhealthy_pairs": list(self.unhealthy_pairs),
        }


@dataclass
class ParasiticDetection:
    detection_id: str
    parasite_module: str
    host_module: str
    drain_rate: float = 0.0
    evidence: List[str] = field(default_factory=list)
    confidence: float = 0.0

    def __post_init__(self):
        if not self.detection_id:
            self.detection_id = uuid.uuid4().hex[:12]
        self.detection_id = _sanitize(self.detection_id, 50)
        self.parasite_module = _sanitize(self.parasite_module, 100)
        self.host_module = _sanitize(self.host_module, 100)
        self.drain_rate = _clamp(self.drain_rate)
        self.evidence = [_sanitize_content(e, 500) for e in self.evidence[:_MAX_EVIDENCE_ITEMS]]
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "detection_id": self.detection_id,
            "parasite_module": self.parasite_module,
            "host_module": self.host_module,
            "drain_rate": round(self.drain_rate, 4),
            "evidence": list(self.evidence),
            "confidence": round(self.confidence, 4),
        }


class CognitiveSymbiosis:
    """Mutualistic Module Interaction Engine.

    Models biological symbiosis between cognitive modules. Detects and
    optimizes cross-module relationships to maximize system performance.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._pairs: Dict[str, SymbioticPair] = {}
        self._pair_order: deque = deque(maxlen=_MAX_PAIRS)
        self._interactions: Dict[str, SymbioticInteraction] = {}
        self._interaction_order: deque = deque(maxlen=_MAX_INTERACTIONS)
        self._pair_interactions: Dict[str, List[str]] = {}
        self._detections: Dict[str, ParasiticDetection] = {}
        self._detection_order: deque = deque(maxlen=_MAX_DETECTIONS)
        self._history: deque = deque(maxlen=_MAX_HISTORY)
        self._stats = {
            "pairs_registered": 0,
            "pairs_removed": 0,
            "interactions_recorded": 0,
            "parasites_detected": 0,
            "pairs_converted": 0,
            "optimizations_suggested": 0,
        }

    def register_pair(self, module_a: str, module_b: str,
                      symbiosis_type: SymbiosisType,
                      interaction_mode: InteractionMode,
                      strength_a_to_b: float = 0.5,
                      strength_b_to_a: float = 0.5) -> SymbioticPair:
        ma = _sanitize(module_a, 100)
        mb = _sanitize(module_b, 100)
        st = symbiosis_type
        if isinstance(st, str):
            try:
                st = SymbiosisType(st)
            except ValueError:
                st = SymbiosisType.NEUTRALISM
        im = interaction_mode
        if isinstance(im, str):
            try:
                im = InteractionMode(im)
            except ValueError:
                im = InteractionMode.TRIGGER
        pair = SymbioticPair(
            pair_id=uuid.uuid4().hex[:12],
            module_a=ma,
            module_b=mb,
            symbiosis_type=st,
            interaction_mode=im,
            strength_a_to_b=strength_a_to_b,
            strength_b_to_a=strength_b_to_a,
        )
        with self._lock:
            if len(self._pairs) >= _MAX_PAIRS:
                evicted_id = self._pair_order.popleft()
                self._pairs.pop(evicted_id, None)
                self._pair_interactions.pop(evicted_id, None)
                logger.warning("Evicted symbiotic pair %s", evicted_id)
            self._pairs[pair.pair_id] = pair
            self._pair_order.append(pair.pair_id)
            self._pair_interactions[pair.pair_id] = []
            with self._stats_lock:
                self._stats["pairs_registered"] += 1
        return copy.deepcopy(pair)

    def remove_pair(self, pair_id: str) -> bool:
        pid = _sanitize(pair_id, 50)
        with self._lock:
            pair = self._pairs.pop(pid, None)
            if pair is None:
                return False
            try:
                self._pair_order.remove(pid)
            except ValueError:
                pass
            int_ids = self._pair_interactions.pop(pid, [])
            for iid in int_ids:
                self._interactions.pop(iid, None)
                try:
                    self._interaction_order.remove(iid)
                except ValueError:
                    pass
            with self._stats_lock:
                self._stats["pairs_removed"] += 1
        return True

    def record_interaction(self, pair_id: str, direction: str,
                           content: str = "",
                           outcome: str = "",
                           benefit_a: float = 0.5,
                           benefit_b: float = 0.5) -> Optional[SymbioticInteraction]:
        pid = _sanitize(pair_id, 50)
        if isinstance(direction, InteractionMode):
            direction = direction.value
        direction = _sanitize(direction, 20)
        content = _sanitize_content(content, 5000)
        outcome = _sanitize_content(outcome, 2000)
        with self._lock:
            pair = self._pairs.get(pid)
            if pair is None:
                return None
            inter = SymbioticInteraction(
                interaction_id=uuid.uuid4().hex[:12],
                pair_id=pid,
                direction=_sanitize(direction, 20),
                content=content,
                outcome=outcome,
                benefit_a=benefit_a,
                benefit_b=benefit_b,
            )
            if len(self._interactions) >= _MAX_INTERACTIONS:
                evicted_id = self._interaction_order.popleft()
                self._interactions.pop(evicted_id, None)
                for k, v_list in self._pair_interactions.items():
                    if evicted_id in v_list:
                        v_list.remove(evicted_id)
                        break
                logger.warning("Evicted symbiotic interaction %s", evicted_id)
            self._interactions[inter.interaction_id] = inter
            self._interaction_order.append(inter.interaction_id)
            if pid not in self._pair_interactions:
                self._pair_interactions[pid] = []
            self._pair_interactions[pid].append(inter.interaction_id)
            now = datetime.now(timezone.utc).isoformat()
            pair.last_interaction = now
            self._update_pair_health_unlocked(pair, inter)
            with self._stats_lock:
                self._stats["interactions_recorded"] += 1
        return copy.deepcopy(inter)

    def _update_pair_health_unlocked(self, pair: SymbioticPair,
                                     interaction: SymbioticInteraction):
        if pair.symbiosis_type == SymbiosisType.MUTUALISM:
            if interaction.benefit_a >= 0.4 and interaction.benefit_b >= 0.4:
                pair.health = min(4, pair.health + 1)
            elif interaction.benefit_a < 0.2 or interaction.benefit_b < 0.2:
                pair.health = max(0, pair.health - 1)
        elif pair.symbiosis_type == SymbiosisType.COMMENSALISM:
            if interaction.benefit_b >= 0.4:
                pair.health = min(4, pair.health + 1)
            elif interaction.benefit_b < 0.2:
                pair.health = max(0, pair.health - 1)
        elif pair.symbiosis_type == SymbiosisType.PARASITISM:
            avg_benefit = (interaction.benefit_a + interaction.benefit_b) / 2.0
            if avg_benefit < 0.3:
                pair.health = max(0, pair.health - 1)
        elif pair.symbiosis_type == SymbiosisType.COMPETITION:
            if abs(interaction.benefit_a - interaction.benefit_b) < 0.2:
                pair.health = min(4, pair.health + 1)
            else:
                pair.health = max(0, pair.health - 1)

    def detect_parasitism(self, min_confidence: float = 0.3) -> List[ParasiticDetection]:
        mc = _clamp(min_confidence)
        results: List[ParasiticDetection] = []
        with self._lock:
            for pair in self._pairs.values():
                if pair.symbiosis_type == SymbiosisType.PARASITISM:
                    int_ids = self._pair_interactions.get(pair.pair_id, [])
                    int_list = [self._interactions[iid] for iid in int_ids
                                if iid in self._interactions]
                    detection = self._analyze_parasitic_pair_unlocked(pair, int_list, mc)
                    if detection:
                        if len(self._detections) >= _MAX_DETECTIONS:
                            evicted_id = self._detection_order.popleft()
                            self._detections.pop(evicted_id, None)
                            logger.warning("Evicted parasitic detection %s", evicted_id)
                        self._detections[detection.detection_id] = detection
                        self._detection_order.append(detection.detection_id)
                        results.append(detection)
                elif pair.symbiosis_type in (
                    SymbiosisType.COMPETITION, SymbiosisType.AMENSALISM
                ):
                    int_ids = self._pair_interactions.get(pair.pair_id, [])
                    int_list = [self._interactions[iid] for iid in int_ids
                                if iid in self._interactions]
                    if int_list:
                        avg_a = sum(i.benefit_a for i in int_list) / len(int_list)
                        avg_b = sum(i.benefit_b for i in int_list) / len(int_list)
                        if avg_a > 0.5 and avg_b < 0.2:
                            drain = _clamp(avg_a - avg_b)
                            conf = _clamp(drain * 0.8)
                            if conf >= mc:
                                ev = [
                                    f"avg_benefit_a={round(avg_a, 4)} > avg_benefit_b={round(avg_b, 4)}"
                                ]
                                det = ParasiticDetection(
                                    detection_id=uuid.uuid4().hex[:12],
                                    parasite_module=pair.module_a,
                                    host_module=pair.module_b,
                                    drain_rate=drain,
                                    evidence=ev,
                                    confidence=conf,
                                )
                                if len(self._detections) >= _MAX_DETECTIONS:
                                    evicted_id = self._detection_order.popleft()
                                    self._detections.pop(evicted_id, None)
                                    logger.warning("Evicted parasitic detection %s", evicted_id)
                                self._detections[det.detection_id] = det
                                self._detection_order.append(det.detection_id)
                                results.append(det)
            with self._stats_lock:
                self._stats["parasites_detected"] += len(results)
        return [copy.deepcopy(r) for r in results]

    def _analyze_parasitic_pair_unlocked(self, pair: SymbioticPair,
                                         interactions: List[SymbioticInteraction],
                                         min_confidence: float) -> Optional[ParasiticDetection]:
        if not interactions:
            avg_a = pair.strength_a_to_b
            avg_b = pair.strength_b_to_a
            drain = _clamp(avg_a - avg_b)
            if drain > 0.2:
                conf = _clamp(drain * 0.6)
                if conf >= min_confidence:
                    ev = [
                        f"strength_a_to_b={round(avg_a, 4)} > strength_b_to_a={round(avg_b, 4)}"
                    ]
                    return ParasiticDetection(
                        detection_id=uuid.uuid4().hex[:12],
                        parasite_module=pair.module_a,
                        host_module=pair.module_b,
                        drain_rate=drain,
                        evidence=ev,
                        confidence=conf,
                    )
            return None
        avg_a = sum(i.benefit_a for i in interactions) / len(interactions)
        avg_b = sum(i.benefit_b for i in interactions) / len(interactions)
        drain = _clamp(avg_a - avg_b)
        low_benefits = sum(1 for i in interactions if i.benefit_b < 0.2)
        ratio = low_benefits / len(interactions)
        conf = _clamp(drain * 0.5 + ratio * 0.5)
        if conf >= min_confidence:
            ev = [
                f"avg_benefit_a={round(avg_a, 4)} avg_benefit_b={round(avg_b, 4)}",
                f"low_host_benefits={low_benefits}/{len(interactions)}",
                f"drain_rate={round(drain, 4)}",
            ]
            return ParasiticDetection(
                detection_id=uuid.uuid4().hex[:12],
                parasite_module=pair.module_a,
                host_module=pair.module_b,
                drain_rate=drain,
                evidence=ev,
                confidence=conf,
            )
        return None

    def convert_pair(self, pair_id: str, new_type: SymbiosisType,
                     new_mode: Optional[InteractionMode] = None) -> Optional[SymbioticPair]:
        pid = _sanitize(pair_id, 50)
        nt = new_type
        if isinstance(nt, str):
            try:
                nt = SymbiosisType(nt)
            except ValueError:
                return None
        with self._lock:
            pair = self._pairs.get(pid)
            if pair is None:
                return None
            old_type = pair.symbiosis_type
            pair.symbiosis_type = nt
            if new_mode is not None:
                pair.interaction_mode = new_mode
            if nt == SymbiosisType.MUTUALISM:
                pair.health = max(pair.health, RelationshipHealth.STABLE)
            elif nt == SymbiosisType.PARASITISM:
                pair.health = min(pair.health, RelationshipHealth.STRAINED)
            elif nt == SymbiosisType.NEUTRALISM:
                pair.health = RelationshipHealth.STABLE
            with self._stats_lock:
                self._stats["pairs_converted"] += 1
            self._history.append({
                "action": "convert",
                "pair_id": pid,
                "from_type": old_type.value,
                "to_type": nt.value,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
        return copy.deepcopy(pair)

    def optimize_mutualism(self) -> List[Dict]:
        suggestions: List[Dict] = []
        with self._lock:
            for pair in self._pairs.values():
                if pair.symbiosis_type == SymbiosisType.COMMENSALISM:
                    int_ids = self._pair_interactions.get(pair.pair_id, [])
                    int_list = [self._interactions[iid] for iid in int_ids
                                if iid in self._interactions]
                    if int_list:
                        avg_a = sum(i.benefit_a for i in int_list) / len(int_list)
                        avg_b = sum(i.benefit_b for i in int_list) / len(int_list)
                        if avg_b > 0.3 and avg_a > 0.2:
                            suggestions.append({
                                "pair_id": pair.pair_id,
                                "module_a": pair.module_a,
                                "module_b": pair.module_b,
                                "current_type": SymbiosisType.COMMENSALISM.value,
                                "suggested_type": SymbiosisType.MUTUALISM.value,
                                "reason": f"Host also benefits (avg_a={round(avg_a, 4)}, avg_b={round(avg_b, 4)})",
                            })
                    else:
                        if pair.strength_b_to_a > 0.3 and pair.strength_a_to_b > 0.2:
                            suggestions.append({
                                "pair_id": pair.pair_id,
                                "module_a": pair.module_a,
                                "module_b": pair.module_b,
                                "current_type": SymbiosisType.COMMENSALISM.value,
                                "suggested_type": SymbiosisType.MUTUALISM.value,
                                "reason": "Bidirectional strength suggests mutual benefit",
                            })
                elif pair.symbiosis_type == SymbiosisType.PARASITISM:
                    int_ids = self._pair_interactions.get(pair.pair_id, [])
                    int_list = [self._interactions[iid] for iid in int_ids
                                if iid in self._interactions]
                    if int_list:
                        avg_a = sum(i.benefit_a for i in int_list) / len(int_list)
                        avg_b = sum(i.benefit_b for i in int_list) / len(int_list)
                        if avg_b > 0.3:
                            suggestions.append({
                                "pair_id": pair.pair_id,
                                "module_a": pair.module_a,
                                "module_b": pair.module_b,
                                "current_type": SymbiosisType.PARASITISM.value,
                                "suggested_type": SymbiosisType.COMMENSALISM.value,
                                "reason": f"Host benefiting more than expected (avg_b={round(avg_b, 4)})",
                            })
                    if pair.health >= RelationshipHealth.STABLE:
                        suggestions.append({
                            "pair_id": pair.pair_id,
                            "module_a": pair.module_a,
                            "module_b": pair.module_b,
                            "current_type": SymbiosisType.PARASITISM.value,
                            "suggested_type": SymbiosisType.COMMENSALISM.value,
                            "reason": "Pair health stable despite parasitism classification",
                        })
                elif pair.symbiosis_type == SymbiosisType.COMPETITION:
                    int_ids = self._pair_interactions.get(pair.pair_id, [])
                    int_list = [self._interactions[iid] for iid in int_ids
                                if iid in self._interactions]
                    if int_list:
                        avg_a = sum(i.benefit_a for i in int_list) / len(int_list)
                        avg_b = sum(i.benefit_b for i in int_list) / len(int_list)
                        if avg_a > 0.4 and avg_b > 0.4:
                            suggestions.append({
                                "pair_id": pair.pair_id,
                                "module_a": pair.module_a,
                                "module_b": pair.module_b,
                                "current_type": SymbiosisType.COMPETITION.value,
                                "suggested_type": SymbiosisType.MUTUALISM.value,
                                "reason": f"Both benefiting (avg_a={round(avg_a, 4)}, avg_b={round(avg_b, 4)})",
                            })
            with self._stats_lock:
                self._stats["optimizations_suggested"] += len(suggestions)
        return [copy.deepcopy(s) for s in suggestions]

    def get_pair(self, pair_id: str) -> Optional[SymbioticPair]:
        pid = _sanitize(pair_id, 50)
        with self._lock:
            pair = self._pairs.get(pid)
            return copy.deepcopy(pair) if pair else None

    def get_pairs(self, module: str = "",
                  symbiosis_type: Optional[SymbiosisType] = None,
                  limit: int = 100) -> List[SymbioticPair]:
        limit = max(1, min(500, limit))
        with self._lock:
            results = list(self._pairs.values())
            if module:
                m = _sanitize(module, 100)
                results = [p for p in results
                           if p.module_a == m or p.module_b == m]
            if symbiosis_type is not None:
                st = symbiosis_type
                if isinstance(st, str):
                    try:
                        st = SymbiosisType(st)
                    except ValueError:
                        return []
                results = [p for p in results if p.symbiosis_type == st]
            results.sort(key=lambda p: p.health, reverse=True)
            return [copy.deepcopy(r) for r in results[:limit]]

    def get_interactions(self, pair_id: str, limit: int = 50) -> List[SymbioticInteraction]:
        limit = max(1, min(500, limit))
        pid = _sanitize(pair_id, 50)
        with self._lock:
            int_ids = self._pair_interactions.get(pid, [])
            int_list = [self._interactions[iid] for iid in int_ids
                        if iid in self._interactions]
            int_list.sort(key=lambda x: x.timestamp, reverse=True)
            return [copy.deepcopy(i) for i in int_list[:limit]]

    def generate_report(self) -> Dict[str, Any]:
        with self._lock:
            pairs = list(self._pairs.values())
            total = len(pairs)
            mutualism_count = sum(1 for p in pairs if p.symbiosis_type == SymbiosisType.MUTUALISM)
            commensalism_count = sum(1 for p in pairs if p.symbiosis_type == SymbiosisType.COMMENSALISM)
            parasite_count = sum(1 for p in pairs if p.symbiosis_type == SymbiosisType.PARASITISM)
            unhealthy = [p.pair_id for p in pairs if p.health <= RelationshipHealth.STRAINED]
            total_benefit = 0.0
            benefit_count = 0
            for pair in pairs:
                int_ids = self._pair_interactions.get(pair.pair_id, [])
                int_list = [self._interactions[iid] for iid in int_ids
                            if iid in self._interactions]
                if int_list:
                    avg_a = sum(i.benefit_a for i in int_list) / len(int_list)
                    avg_b = sum(i.benefit_b for i in int_list) / len(int_list)
                    total_benefit += (avg_a + avg_b) / 2.0
                    benefit_count += 1
                else:
                    total_benefit += (pair.strength_a_to_b + pair.strength_b_to_a) / 2.0
                    benefit_count += 1
            avg_benefit = _clamp(total_benefit / benefit_count) if benefit_count else 0.0
            report = {
                "report_id": uuid.uuid4().hex[:12],
                "total_pairs": total,
                "mutualism_count": mutualism_count,
                "commensalism_count": commensalism_count,
                "parasite_count": parasite_count,
                "avg_system_benefit": round(avg_benefit, 4),
                "unhealthy_pairs": unhealthy[:_MAX_UNHEALTHY_LIST],
            }
        return copy.deepcopy(report)

    def get_system_benefit(self) -> float:
        with self._lock:
            pairs = list(self._pairs.values())
            if not pairs:
                return 0.0
            total_benefit = 0.0
            benefit_count = 0
            for pair in pairs:
                int_ids = self._pair_interactions.get(pair.pair_id, [])
                int_list = [self._interactions[iid] for iid in int_ids
                            if iid in self._interactions]
                if int_list:
                    avg_a = sum(i.benefit_a for i in int_list) / len(int_list)
                    avg_b = sum(i.benefit_b for i in int_list) / len(int_list)
                    total_benefit += (avg_a + avg_b) / 2.0
                    benefit_count += 1
                else:
                    total_benefit += (pair.strength_a_to_b + pair.strength_b_to_a) / 2.0
                    benefit_count += 1
            return _clamp(total_benefit / benefit_count) if benefit_count else 0.0

    def suggest_optimizations(self) -> List[Dict]:
        suggestions: List[Dict] = []
        with self._lock:
            for pair in self._pairs.values():
                int_ids = self._pair_interactions.get(pair.pair_id, [])
                int_list = [self._interactions[iid] for iid in int_ids
                            if iid in self._interactions]
                if pair.health <= RelationshipHealth.DETERIORATING:
                    suggestions.append({
                        "pair_id": pair.pair_id,
                        "module_a": pair.module_a,
                        "module_b": pair.module_b,
                        "type": "health_intervention",
                        "current_health": int(pair.health),
                        "suggestion": f"Pair health critical ({pair.health}/4), consider rebalancing or converting relationship",
                    })
                if pair.symbiosis_type == SymbiosisType.PARASITISM and pair.health >= RelationshipHealth.STABLE:
                    suggestions.append({
                        "pair_id": pair.pair_id,
                        "module_a": pair.module_a,
                        "module_b": pair.module_b,
                        "type": "reclassify",
                        "suggestion": "Parasitic pair has stable health, consider converting to commensalism",
                    })
                if int_list and pair.symbiosis_type == SymbiosisType.MUTUALISM:
                    recent = int_list[-5:]
                    avg_benefit = sum((i.benefit_a + i.benefit_b) / 2.0 for i in recent) / len(recent)
                    if avg_benefit > 0.7:
                        suggestions.append({
                            "pair_id": pair.pair_id,
                            "module_a": pair.module_a,
                            "module_b": pair.module_b,
                            "type": "amplify",
                            "suggestion": f"High-performing mutualism (avg_benefit={round(avg_benefit, 4)}), consider increasing interaction frequency",
                        })
                if pair.symbiosis_type == SymbiosisType.NEUTRALISM:
                    suggestions.append({
                        "pair_id": pair.pair_id,
                        "module_a": pair.module_a,
                        "module_b": pair.module_b,
                        "type": "evaluate",
                        "suggestion": "Neutral relationship, consider whether modules could benefit from interaction",
                    })
                if int_list:
                    avg_imbalance = abs(
                        sum(i.benefit_a for i in int_list) / len(int_list)
                        - sum(i.benefit_b for i in int_list) / len(int_list)
                    )
                    if avg_imbalance > 0.4:
                        suggestions.append({
                            "pair_id": pair.pair_id,
                            "module_a": pair.module_a,
                            "module_b": pair.module_b,
                            "type": "rebalance",
                            "suggestion": f"Benefit imbalance detected (imbalance={round(avg_imbalance, 4)}), consider equalizing contributions",
                        })
            with self._stats_lock:
                self._stats["optimizations_suggested"] += len(suggestions)
        return [copy.deepcopy(s) for s in suggestions]

    def get_stats(self) -> Dict:
        with self._lock:
            pair_count = len(self._pairs)
            interaction_count = len(self._interactions)
            detection_count = len(self._detections)
            pairs = list(self._pairs.values())
        with self._stats_lock:
            stats = dict(self._stats)
        stats["pair_count"] = pair_count
        stats["interaction_count"] = interaction_count
        stats["detection_count"] = detection_count
        if pairs:
            stats["avg_health"] = round(
                sum(int(p.health) for p in pairs) / len(pairs), 4
            )
            stats["mutualism_pct"] = round(
                sum(1 for p in pairs if p.symbiosis_type == SymbiosisType.MUTUALISM) / len(pairs), 4
            )
        else:
            stats["avg_health"] = 0.0
            stats["mutualism_pct"] = 0.0
        return stats