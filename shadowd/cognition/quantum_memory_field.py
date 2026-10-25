#!/usr/bin/env python3
"""
QuantumMemoryField -- Superposition Memory Access Engine.

Memory access where multiple memories coexist in superposition until a
"query" collapses the field. Inspired by quantum mechanics -- memories
are probabilistic, entangled, and context-dependent. Observation
collapses the field to the most relevant memory.

Key concepts:
  - QuantumMemory: a memory in superposition (amplitude encodes relevance)
  - MemoryEntanglement: correlation link between two memories
  - CollapsedResult: outcome of a field observation
  - FieldState: aggregate statistics of the quantum field

Thread-safe, bounded, sanitized, deepcopy returns.
"""

import copy
import hashlib
import logging
import math
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.quantum_memory_field")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_MEMORIES = 5000
_MAX_ENTANGLEMENTS = 3000
_MAX_COLLAPSED = 1000
_MAX_TAGS = 20


class MemoryState(str, Enum):
    SUPERPOSITION = "superposition"
    COLLAPSED = "collapsed"
    DECOHERENT = "decoherent"


class EntanglementType(str, Enum):
    TEMPORAL = "temporal"
    SEMANTIC = "semantic"
    EMOTIONAL = "emotional"
    CAUSAL = "causal"
    CONTEXTUAL = "contextual"


class AccessType(str, Enum):
    OBSERVE = "observe"
    RECALL = "recall"
    ASSOCIATE = "associate"
    PREDICT = "predict"


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


def _word_overlap(a: str, b: str) -> float:
    words_a = set(re.findall(r'[a-zA-Z0-9]+', a.lower()))
    words_b = set(re.findall(r'[a-zA-Z0-9]+', b.lower()))
    if not words_a or not words_b:
        return 0.0
    intersection = words_a & words_b
    union = words_a | words_b
    return len(intersection) / len(union)


@dataclass
class QuantumMemory:
    memory_id: str = ""
    content: str = ""
    state: MemoryState = MemoryState.SUPERPOSITION
    probability_amplitude: float = 0.5
    tags: List[str] = field(default_factory=list)
    domain: str = ""
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.memory_id:
            self.memory_id = "qm_%s" % uuid.uuid4().hex[:12]
        self.memory_id = _sanitize(self.memory_id, 64)
        self.content = _sanitize_content(self.content, 50000)
        if isinstance(self.state, str):
            try:
                self.state = MemoryState(self.state)
            except ValueError:
                self.state = MemoryState.SUPERPOSITION
        self.probability_amplitude = _clamp(self.probability_amplitude)
        self.tags = [_sanitize(t, 100) for t in (self.tags or [])[:_MAX_TAGS]]
        self.domain = _sanitize(self.domain, 100)
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list(self.metadata.items())[:20]
        }

    def to_dict(self) -> Dict:
        return {
            "memory_id": self.memory_id,
            "content": self.content,
            "state": self.state.value,
            "probability_amplitude": round(self.probability_amplitude, 4),
            "tags": list(self.tags),
            "domain": self.domain,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class MemoryEntanglement:
    entanglement_id: str = ""
    memory_id_1: str = ""
    memory_id_2: str = ""
    entanglement_type: EntanglementType = EntanglementType.SEMANTIC
    correlation: float = 0.5
    strength: float = 0.5

    def __post_init__(self):
        if not self.entanglement_id:
            self.entanglement_id = "ent_%s" % uuid.uuid4().hex[:12]
        self.entanglement_id = _sanitize(self.entanglement_id, 64)
        self.memory_id_1 = _sanitize(self.memory_id_1, 64)
        self.memory_id_2 = _sanitize(self.memory_id_2, 64)
        if isinstance(self.entanglement_type, str):
            try:
                self.entanglement_type = EntanglementType(self.entanglement_type)
            except ValueError:
                self.entanglement_type = EntanglementType.SEMANTIC
        self.correlation = _clamp(self.correlation)
        self.strength = _clamp(self.strength)

    def to_dict(self) -> Dict:
        return {
            "entanglement_id": self.entanglement_id,
            "memory_id_1": self.memory_id_1,
            "memory_id_2": self.memory_id_2,
            "entanglement_type": self.entanglement_type.value,
            "correlation": round(self.correlation, 4),
            "strength": round(self.strength, 4),
        }


@dataclass
class CollapsedResult:
    result_id: str = ""
    query: str = ""
    collapsed_memories: List[str] = field(default_factory=list)
    confidence: float = 0.0
    decay_rate: float = 0.01
    context: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        if not self.result_id:
            self.result_id = "cr_%s" % uuid.uuid4().hex[:12]
        self.result_id = _sanitize(self.result_id, 64)
        self.query = _sanitize_content(self.query, 1000)
        self.collapsed_memories = [_sanitize(m, 64) for m in (self.collapsed_memories or [])[:100]]
        self.confidence = _clamp(self.confidence)
        self.decay_rate = _clamp(self.decay_rate, 0.0, 1.0)
        self.context = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list(self.context.items())[:20]
        }

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "query": self.query,
            "collapsed_memories": list(self.collapsed_memories),
            "confidence": round(self.confidence, 4),
            "decay_rate": round(self.decay_rate, 4),
            "context": copy.deepcopy(self.context),
        }


@dataclass
class FieldState:
    total_memories: int = 0
    superposition_count: int = 0
    collapsed_count: int = 0
    decoherent_count: int = 0
    avg_amplitude: float = 0.0

    def to_dict(self) -> Dict:
        return {
            "total_memories": self.total_memories,
            "superposition_count": self.superposition_count,
            "collapsed_count": self.collapsed_count,
            "decoherent_count": self.decoherent_count,
            "avg_amplitude": round(self.avg_amplitude, 4),
        }


class QuantumMemoryField:
    """Superposition Memory Access Engine.

    Memories exist in a quantum-like superposition until a query
    collapses the field, selecting the most relevant memories based
    on probability amplitudes and entanglement correlations.
    """

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._memories: Dict[str, QuantumMemory] = {}
        self._memory_order: deque = deque(maxlen=_MAX_MEMORIES)
        self._entanglements: Dict[str, MemoryEntanglement] = {}
        self._entanglement_order: deque = deque(maxlen=_MAX_ENTANGLEMENTS)
        self._collapsed_results: Dict[str, CollapsedResult] = {}
        self._collapsed_order: deque = deque(maxlen=_MAX_COLLAPSED)
        self._stats = {
            "memories_stored": 0,
            "observations_made": 0,
            "entanglements_created": 0,
            "entanglements_removed": 0,
            "recalls_performed": 0,
            "associations_made": 0,
            "predictions_made": 0,
            "decoherence_applications": 0,
            "field_collapses": 0,
        }

    def store(self, content: str, domain: str = "", tags: Optional[List[str]] = None,
              probability: float = 0.5, metadata: Optional[Dict] = None) -> QuantumMemory:
        content = _sanitize_content(content, 50000)
        domain = _sanitize(domain, 100)
        probability = _clamp(probability)
        if tags is None:
            tags = []
        tags = [_sanitize(t, 100) for t in tags[:_MAX_TAGS]]
        if metadata is None:
            metadata = {}
        metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list(metadata.items())[:20]
        }
        memory = QuantumMemory(
            memory_id="qm_%s" % uuid.uuid4().hex[:12],
            content=content,
            state=MemoryState.SUPERPOSITION,
            probability_amplitude=probability,
            tags=tags,
            domain=domain,
            metadata=metadata,
        )
        with self._lock:
            if len(self._memories) >= _MAX_MEMORIES:
                evict_id = self._memory_order.popleft()
                self._memories.pop(evict_id, None)
                self._evict_entanglements_for(evict_id)
                logger.warning("Evicting memory %s (cap=%d)", evict_id, _MAX_MEMORIES)
            self._memories[memory.memory_id] = memory
            self._memory_order.append(memory.memory_id)
            with self._stats_lock:
                self._stats["memories_stored"] += 1
        return copy.deepcopy(memory)

    def observe(self, query: str, context: Optional[Dict] = None,
                access_type: AccessType = AccessType.OBSERVE) -> CollapsedResult:
        query = _sanitize_content(query, 1000)
        if context is None:
            context = {}
        if isinstance(access_type, str):
            try:
                access_type = AccessType(access_type)
            except ValueError:
                access_type = AccessType.OBSERVE

        with self._lock:
            candidates = self._compute_relevance(query, context)
            threshold = 0.1 if access_type == AccessType.ASSOCIATE else 0.2
            collapsed_ids = []
            total_amp = 0.0
            for mid, score in candidates:
                if score < threshold:
                    continue
                memory = self._memories.get(mid)
                if not memory:
                    continue
                if memory.state != MemoryState.SUPERPOSITION:
                    continue
                memory.state = MemoryState.COLLAPSED
                memory.probability_amplitude = _clamp(score)
                memory.accessed_at = datetime.now(timezone.utc).isoformat()
                collapsed_ids.append(mid)
                total_amp += score
                if len(collapsed_ids) >= 20:
                    break
            for mid, _ in candidates:
                memory = self._memories.get(mid)
                if memory and memory.state == MemoryState.SUPERPOSITION:
                    self._propagate_entanglement_collapse(mid, set(collapsed_ids))

            confidence = _clamp(total_amp / max(len(collapsed_ids), 1)) if collapsed_ids else 0.0
            result = CollapsedResult(
                result_id="cr_%s" % uuid.uuid4().hex[:12],
                query=query,
                collapsed_memories=collapsed_ids,
                confidence=confidence,
                decay_rate=0.01,
                context={
                    "access_type": access_type.value,
                    "candidate_count": len(candidates),
                    **{
                        _sanitize(str(k), 50): _sanitize_content(str(v), 500)
                        for k, v in list(context.items())[:10]
                    },
                },
            )
            if len(self._collapsed_results) >= _MAX_COLLAPSED:
                evict_id = self._collapsed_order.popleft()
                self._collapsed_results.pop(evict_id, None)
                logger.warning("Evicting collapsed result %s (cap=%d)", evict_id, _MAX_COLLAPSED)
            self._collapsed_results[result.result_id] = result
            self._collapsed_order.append(result.result_id)
            with self._stats_lock:
                self._stats["observations_made"] += 1
                self._stats["field_collapses"] += 1
                if access_type == AccessType.ASSOCIATE:
                    self._stats["associations_made"] += 1
                elif access_type == AccessType.PREDICT:
                    self._stats["predictions_made"] += 1

        return copy.deepcopy(result)

    def entangle(self, memory_id_1: str, memory_id_2: str,
                 entanglement_type: EntanglementType = EntanglementType.SEMANTIC,
                 correlation: float = 0.5) -> Optional[MemoryEntanglement]:
        mid1 = _sanitize(memory_id_1, 64)
        mid2 = _sanitize(memory_id_2, 64)
        correlation = _clamp(correlation)
        if isinstance(entanglement_type, str):
            try:
                entanglement_type = EntanglementType(entanglement_type)
            except ValueError:
                entanglement_type = EntanglementType.SEMANTIC

        with self._lock:
            if mid1 not in self._memories or mid2 not in self._memories:
                return None
            strength = self._compute_entanglement_strength(mid1, mid2)
            entanglement = MemoryEntanglement(
                entanglement_id="ent_%s" % uuid.uuid4().hex[:12],
                memory_id_1=mid1,
                memory_id_2=mid2,
                entanglement_type=entanglement_type,
                correlation=correlation,
                strength=strength,
            )
            if len(self._entanglements) >= _MAX_ENTANGLEMENTS:
                evict_id = self._entanglement_order.popleft()
                self._entanglements.pop(evict_id, None)
                logger.warning("Evicting entanglement %s (cap=%d)", evict_id, _MAX_ENTANGLEMENTS)
            self._entanglements[entanglement.entanglement_id] = entanglement
            self._entanglement_order.append(entanglement.entanglement_id)
            with self._stats_lock:
                self._stats["entanglements_created"] += 1
            return copy.deepcopy(entanglement)

    def disentangle(self, entanglement_id: str) -> bool:
        eid = _sanitize(entanglement_id, 64)
        with self._lock:
            if eid not in self._entanglements:
                return False
            self._entanglements.pop(eid, None)
            self._entanglement_order = deque(
                [x for x in self._entanglement_order if x != eid],
                maxlen=_MAX_ENTANGLEMENTS,
            )
            with self._stats_lock:
                self._stats["entanglements_removed"] += 1
        return True

    def recall(self, memory_id: str) -> Optional[QuantumMemory]:
        mid = _sanitize(memory_id, 64)
        with self._lock:
            memory = self._memories.get(mid)
            if not memory:
                return None
            if memory.state == MemoryState.SUPERPOSITION:
                memory.state = MemoryState.COLLAPSED
                memory.accessed_at = datetime.now(timezone.utc).isoformat()
            elif memory.state == MemoryState.DECOHERENT:
                memory.probability_amplitude = _clamp(memory.probability_amplitude * 0.8)
                memory.accessed_at = datetime.now(timezone.utc).isoformat()
            with self._stats_lock:
                self._stats["recalls_performed"] += 1
            return copy.deepcopy(memory)

    def associate(self, query: str, max_results: int = 10) -> List[Dict]:
        query = _sanitize_content(query, 1000)
        max_results = max(1, min(100, max_results))
        with self._lock:
            candidates = self._compute_relevance(query, {"_associate": True})
            results = []
            seen = set()
            for mid, score in candidates:
                if mid in seen:
                    continue
                seen.add(mid)
                memory = self._memories.get(mid)
                if not memory:
                    continue
                entangled = self._get_entangled_memories(mid)
                results.append({
                    "memory": copy.deepcopy(memory).to_dict(),
                    "relevance_score": round(score, 4),
                    "entangled_memories": [copy.deepcopy(self._memories[eid]).to_dict()
                                           for eid in entangled
                                           if eid in self._memories],
                })
                if len(results) >= max_results:
                    break
            with self._stats_lock:
                self._stats["associations_made"] += 1
                self._stats["observations_made"] += 1
        return results

    def predict_next(self, context: str, domain: str = "") -> Optional[Dict]:
        context = _sanitize_content(context, 1000)
        domain = _sanitize(domain, 100)
        with self._lock:
            domain_memories = [
                m for m in self._memories.values()
                if m.state == MemoryState.SUPERPOSITION
                and (not domain or m.domain == domain)
            ]
            if not domain_memories:
                domain_memories = [
                    m for m in self._memories.values()
                    if m.state == MemoryState.SUPERPOSITION
                ]
            if not domain_memories:
                with self._stats_lock:
                    self._stats["predictions_made"] += 1
                return None

            scored = []
            for memory in domain_memories:
                score = memory.probability_amplitude
                overlap = _word_overlap(context, memory.content)
                score += overlap * 0.4
                entangled = self._get_entangled_memories(memory.memory_id)
                for eid in entangled:
                    ent = self._memories.get(eid)
                    if ent and ent.state == MemoryState.COLLAPSED:
                        score += 0.1
                scored.append((memory, _clamp(score)))
            scored.sort(key=lambda x: x[1], reverse=True)
            best_memory, best_score = scored[0]
            with self._stats_lock:
                self._stats["predictions_made"] += 1
            return {
                "predicted_memory": copy.deepcopy(best_memory).to_dict(),
                "prediction_confidence": round(best_score, 4),
                "domain": domain,
            }

    def decohere(self, decay_rate: float = 0.01) -> int:
        decay_rate = _clamp(decay_rate, 0.0, 1.0)
        decohered_count = 0
        with self._lock:
            for memory in self._memories.values():
                if memory.state == MemoryState.SUPERPOSITION:
                    memory.probability_amplitude = _clamp(
                        memory.probability_amplitude * (1.0 - decay_rate)
                    )
                    if memory.probability_amplitude < 0.05:
                        memory.state = MemoryState.DECOHERENT
                        decohered_count += 1
            for memory in self._memories.values():
                if memory.state == MemoryState.DECOHERENT:
                    memory.probability_amplitude = _clamp(
                        memory.probability_amplitude * (1.0 - decay_rate * 2.0)
                    )
            with self._stats_lock:
                self._stats["decoherence_applications"] += 1
        return decohered_count

    def get_field_state(self) -> FieldState:
        with self._lock:
            total = len(self._memories)
            superposition = 0
            collapsed = 0
            decoherent = 0
            amp_sum = 0.0
            for memory in self._memories.values():
                if memory.state == MemoryState.SUPERPOSITION:
                    superposition += 1
                elif memory.state == MemoryState.COLLAPSED:
                    collapsed += 1
                elif memory.state == MemoryState.DECOHERENT:
                    decoherent += 1
                amp_sum += memory.probability_amplitude
            avg_amp = round(amp_sum / max(total, 1), 4) if total > 0 else 0.0
        return FieldState(
            total_memories=total,
            superposition_count=superposition,
            collapsed_count=collapsed,
            decoherent_count=decoherent,
            avg_amplitude=avg_amp,
        )

    def get_entanglements(self, memory_id: str) -> List[MemoryEntanglement]:
        mid = _sanitize(memory_id, 64)
        with self._lock:
            results = [
                ent for ent in self._entanglements.values()
                if ent.memory_id_1 == mid or ent.memory_id_2 == mid
            ]
        return [copy.deepcopy(e) for e in results]

    def get_stats(self) -> Dict:
        with self._lock:
            total = len(self._memories)
            superposition = 0
            collapsed = 0
            decoherent = 0
            amp_sum = 0.0
            for memory in self._memories.values():
                if memory.state == MemoryState.SUPERPOSITION:
                    superposition += 1
                elif memory.state == MemoryState.COLLAPSED:
                    collapsed += 1
                elif memory.state == MemoryState.DECOHERENT:
                    decoherent += 1
                amp_sum += memory.probability_amplitude
            avg_amp = round(amp_sum / max(total, 1), 4) if total > 0 else 0.0
            ent_count = len(self._entanglements)
        with self._stats_lock:
            stats = dict(self._stats)
        stats["field_state"] = {
            "total_memories": total,
            "superposition_count": superposition,
            "collapsed_count": collapsed,
            "decoherent_count": decoherent,
            "avg_amplitude": avg_amp,
        }
        stats["total_memories"] = total
        stats["total_entanglements"] = ent_count
        return stats

    def _compute_relevance(self, query: str, context: Optional[Dict] = None) -> List[tuple]:
        query_lower = query.lower()
        query_words = set(re.findall(r'[a-zA-Z0-9]+', query_lower))
        scored = []
        for mid, memory in self._memories.items():
            if memory.state == MemoryState.DECOHERENT:
                continue
            score = memory.probability_amplitude
            content_lower = memory.content.lower()
            content_words = set(re.findall(r'[a-zA-Z0-9]+', content_lower))
            word_overlap = len(query_words & content_words) / max(len(query_words | content_words), 1)
            score += word_overlap * 0.5
            for tag in memory.tags:
                if re.search(r'\b' + re.escape(tag.lower()) + r'\b', query_lower):
                    score += 0.15
                elif _word_overlap(tag, query) > 0.5:
                    score += 0.1
            if memory.domain and context and isinstance(context, dict):
                domain_hint = context.get("domain", "")
                if memory.domain == domain_hint:
                    score += 0.2
            if memory.state == MemoryState.COLLAPSED:
                score *= 0.3
            entangled = self._get_entangled_memories(mid)
            if entangled:
                ent_boost = 0.0
                for eid in entangled:
                    ent_mem = self._memories.get(eid)
                    if ent_mem and ent_mem.state == MemoryState.COLLAPSED:
                        ent_overlap = _word_overlap(query, ent_mem.content)
                        ent_boost += ent_overlap * 0.15
                score += ent_boost
            prob_squared = memory.probability_amplitude ** 2
            construct_interference = prob_squared * word_overlap
            score += construct_interference * 0.2
            scored.append((mid, _clamp(score)))
        scored.sort(key=lambda x: x[1], reverse=True)
        return scored

    def _compute_entanglement_strength(self, mid1: str, mid2: str) -> float:
        mem1 = self._memories.get(mid1)
        mem2 = self._memories.get(mid2)
        if not mem1 or not mem2:
            return 0.0
        overlap = _word_overlap(mem1.content, mem2.content)
        tag_overlap = 0.0
        if mem1.tags and mem2.tags:
            tags1 = set(t.lower() for t in mem1.tags)
            tags2 = set(t.lower() for t in mem2.tags)
            if tags1 | tags2:
                tag_overlap = len(tags1 & tags2) / len(tags1 | tags2)
        domain_match = 0.0
        if mem1.domain and mem2.domain and mem1.domain == mem2.domain:
            domain_match = 0.3
        strength = overlap * 0.5 + tag_overlap * 0.2 + domain_match
        return _clamp(strength)

    def _get_entangled_memories(self, memory_id: str) -> List[str]:
        entangled = []
        for ent in self._entanglements.values():
            if ent.memory_id_1 == memory_id:
                entangled.append(ent.memory_id_2)
            elif ent.memory_id_2 == memory_id:
                entangled.append(ent.memory_id_1)
        return entangled[:50]

    def _propagate_entanglement_collapse(self, source_id: str, collapsed_ids: set) -> None:
        for ent in self._entanglements.values():
            other_id = None
            if ent.memory_id_1 == source_id:
                other_id = ent.memory_id_2
            elif ent.memory_id_2 == source_id:
                other_id = ent.memory_id_1
            if other_id and other_id in self._memories and other_id not in collapsed_ids:
                other = self._memories[other_id]
                if other.state == MemoryState.SUPERPOSITION:
                    other.probability_amplitude = _clamp(
                        other.probability_amplitude + ent.correlation * 0.1
                    )

    def _evict_entanglements_for(self, memory_id: str) -> None:
        to_remove = [
            eid for eid, ent in self._entanglements.items()
            if ent.memory_id_1 == memory_id or ent.memory_id_2 == memory_id
        ]
        for eid in to_remove:
            self._entanglements.pop(eid, None)
        if to_remove:
            self._entanglement_order = deque(
                [x for x in self._entanglement_order if x not in set(to_remove)],
                maxlen=_MAX_ENTANGLEMENTS,
            )
            logger.warning("Evicting %d entanglements for memory %s", len(to_remove), memory_id)