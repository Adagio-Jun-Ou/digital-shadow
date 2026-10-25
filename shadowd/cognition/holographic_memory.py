#!/usr/bin/env python3
"""
HolographicMemory — Every Fragment Contains the Whole.

Distributed holographic encoding where any fragment can reconstruct the
whole (with some loss). Content is encoded into multiple redundant
fragments using simple hash-based vector distribution. More fragments
yield higher reconstruction quality.

Key concepts:
  - HologramFragment: a partial encoding of original content
  - ReconstructionResult: content rebuilt from available fragments
  - HologramType: FULL, PARTIAL, REDUNDANT, CORRUPTED

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

logger = logging.getLogger("shadowd.cognition.holographic_memory")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_FRAGMENTS = 5000
_MAX_RECONSTRUCTIONS = 500
_MAX_ENCODING_DIMENSIONS = 100


class HologramType(str, Enum):
    FULL = "full"
    PARTIAL = "partial"
    REDUNDANT = "redundant"
    CORRUPTED = "corrupted"


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


def _hash_to_dimension(content: str, dimension: int, num_dimensions: int) -> float:
    h = hashlib.sha256(f"{content}|dim_{dimension}".encode()).hexdigest()
    value = int(h[:8], 16) / 0xFFFFFFFF
    return round(value, 4)


def _encode_content(content: str, num_dimensions: int = 50) -> Dict[str, float]:
    dimensions = min(num_dimensions, _MAX_ENCODING_DIMENSIONS)
    encoding = {}
    for i in range(dimensions):
        key = f"d_{i}"
        value = _hash_to_dimension(content, i, dimensions)
        encoding[key] = value
    return encoding


def _content_hash(content: str) -> str:
    return hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class HologramFragment:
    fragment_id: str
    content: str
    encoding: Dict[str, float] = field(default_factory=dict)
    reconstruction_quality: float = 0.5
    source_hash: str = ""
    fragment_type: HologramType = HologramType.PARTIAL
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.fragment_id:
            self.fragment_id = uuid.uuid4().hex[:12]
        self.fragment_id = _sanitize(self.fragment_id, 50)
        self.content = _sanitize_content(self.content, 50000)
        self.reconstruction_quality = _clamp(self.reconstruction_quality)
        self.source_hash = _sanitize(self.source_hash, 50)
        if len(self.encoding) > _MAX_ENCODING_DIMENSIONS:
            items = list(self.encoding.items())[:_MAX_ENCODING_DIMENSIONS]
            self.encoding = {k: _clamp(v) for k, v in items}
        else:
            self.encoding = {str(k)[:50]: _clamp(v) for k, v in self.encoding.items()}
        if len(self.metadata) > 20:
            self.metadata = dict(list(self.metadata.items())[:20])

    def to_dict(self) -> Dict:
        return {
            "fragment_id": self.fragment_id,
            "content": self.content,
            "encoding": dict(self.encoding),
            "reconstruction_quality": round(self.reconstruction_quality, 4),
            "source_hash": self.source_hash,
            "fragment_type": self.fragment_type.value,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class ReconstructionResult:
    result_id: str
    source_fragments: List[str]
    reconstructed_content: str
    quality: float = 0.0
    method: str = "holographic"
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.result_id:
            self.result_id = uuid.uuid4().hex[:12]
        self.result_id = _sanitize(self.result_id, 50)
        self.source_fragments = [_sanitize(f, 50) for f in (self.source_fragments or [])[:50]]
        self.reconstructed_content = _sanitize_content(self.reconstructed_content, 50000)
        self.quality = _clamp(self.quality)
        self.method = _sanitize(self.method, 100)
        if len(self.metadata) > 20:
            self.metadata = dict(list(self.metadata.items())[:20])

    def to_dict(self) -> Dict:
        return {
            "result_id": self.result_id,
            "source_fragments": list(self.source_fragments),
            "reconstructed_content": self.reconstructed_content,
            "quality": round(self.quality, 4),
            "method": self.method,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


class HolographicMemory:
    """Distributed holographic memory where every fragment contains
    the whole (with some loss). Content encoded into redundant
    fragments; any subset can partially reconstruct the original."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._fragments: Dict[str, HologramFragment] = {}
        self._reconstructions: Dict[str, ReconstructionResult] = {}
        self._fragment_order: deque = deque(maxlen=_MAX_FRAGMENTS)
        self._reconstruction_order: deque = deque(maxlen=_MAX_RECONSTRUCTIONS)
        self._stats = {
            "fragments_created": 0,
            "full_encodings_created": 0,
            "reconstructions_attempted": 0,
            "reconstructions_succeeded": 0,
            "deletions": 0,
            "integrity_checks_passed": 0,
            "integrity_checks_failed": 0,
        }

    def encode(self, content: str, metadata: Optional[Dict] = None) -> HologramFragment:
        content_clean = _sanitize_content(content, 50000)
        src_hash = _content_hash(content_clean)
        encoding = _encode_content(content_clean)
        fragment = HologramFragment(
            fragment_id=uuid.uuid4().hex[:12],
            content=content_clean,
            encoding=encoding,
            reconstruction_quality=1.0,
            source_hash=src_hash,
            fragment_type=HologramType.FULL,
            metadata=metadata or {},
        )
        with self._lock:
            if len(self._fragments) >= _MAX_FRAGMENTS:
                evicted_id = self._fragment_order.popleft()
                self._fragments.pop(evicted_id, None)
                logger.warning("Evicted hologram fragment %s", evicted_id)
            self._fragments[fragment.fragment_id] = fragment
            self._fragment_order.append(fragment.fragment_id)
            with self._stats_lock:
                self._stats["fragments_created"] += 1
        return copy.deepcopy(fragment)

    def encode_full(self, content: str, redundancy: int = 3,
                    metadata: Optional[Dict] = None) -> List[HologramFragment]:
        content_clean = _sanitize_content(content, 50000)
        redundancy = max(1, min(10, int(redundancy)))
        src_hash = _content_hash(content_clean)
        fragments = []
        for i in range(redundancy):
            offset_seed = f"{content_clean}|fragment_{i}"
            dim_count = max(10, _MAX_ENCODING_DIMENSIONS // redundancy)
            partial_encoding = {}
            for d in range(dim_count):
                key = f"d_{d}"
                val = _hash_to_dimension(offset_seed, d, dim_count)
                partial_encoding[key] = val
            quality = _clamp(1.0 / redundancy)
            frag_type = HologramType.REDUNDANT if i > 0 else HologramType.PARTIAL
            fragment = HologramFragment(
                fragment_id=uuid.uuid4().hex[:12],
                content=content_clean[:max(50, len(content_clean) // redundancy)] if i > 0 else content_clean,
                encoding=partial_encoding,
                reconstruction_quality=quality,
                source_hash=src_hash,
                fragment_type=frag_type,
                metadata=metadata or {},
            )
            fragments.append(fragment)
        with self._lock:
            for frag in fragments:
                if len(self._fragments) >= _MAX_FRAGMENTS:
                    evicted_id = self._fragment_order.popleft()
                    self._fragments.pop(evicted_id, None)
                    logger.warning("Evicted hologram fragment %s", evicted_id)
                self._fragments[frag.fragment_id] = frag
                self._fragment_order.append(frag.fragment_id)
            with self._stats_lock:
                self._stats["fragments_created"] += len(fragments)
                self._stats["full_encodings_created"] += 1
        return [copy.deepcopy(f) for f in fragments]

    def retrieve(self, fragment_id: str) -> Optional[HologramFragment]:
        fid = _sanitize(fragment_id, 50)
        with self._lock:
            frag = self._fragments.get(fid)
            return copy.deepcopy(frag) if frag else None

    def reconstruct(self, fragment_ids: List[str]) -> ReconstructionResult:
        clean_ids = [_sanitize(fid, 50) for fid in (fragment_ids or [])[:50]]
        with self._lock:
            with self._stats_lock:
                self._stats["reconstructions_attempted"] += 1
            fragments = []
            for fid in clean_ids:
                frag = self._fragments.get(fid)
                if frag:
                    fragments.append(frag)
            if not fragments:
                result = ReconstructionResult(
                    result_id=uuid.uuid4().hex[:12],
                    source_fragments=clean_ids,
                    reconstructed_content="",
                    quality=0.0,
                    method="holographic",
                    metadata={"error": "no_valid_fragments"},
                )
                self._store_reconstruction(result)
                return copy.deepcopy(result)
            best_fragment = max(fragments, key=lambda f: f.reconstruction_quality)
            full_fragments = [f for f in fragments if f.fragment_type == HologramType.FULL]
            if full_fragments:
                best = full_fragments[0]
                quality = 1.0
                reconstructed = best.content
            else:
                total_quality = sum(f.reconstruction_quality for f in fragments)
                quality = _clamp(total_quality)
                all_content = " ".join(
                    f.content for f in fragments if f.content
                )
                content_lens = [len(f.content) for f in fragments if f.content]
                target_len = max(content_lens) if content_lens else (len(best_fragment.content) or 5000)
                reconstructed = all_content[:target_len]
            # `result` must be built regardless of which branch above ran; the
            # original code nested this inside the `else` and then referenced
            # `result` outside it, hitting UnboundLocalError on FULL fragments.
            result = ReconstructionResult(
                result_id=uuid.uuid4().hex[:12],
                source_fragments=[f.fragment_id for f in fragments],
                reconstructed_content=reconstructed,
                quality=quality,
                method="holographic",
                metadata={
                    "fragment_count": len(fragments),
                    "fragment_types": [f.fragment_type.value for f in fragments],
                },
            )
            with self._stats_lock:
                self._stats["reconstructions_succeeded"] += 1
            self._store_reconstruction(result)
        return copy.deepcopy(result)

    def _store_reconstruction(self, result: ReconstructionResult) -> None:
        if len(self._reconstructions) >= _MAX_RECONSTRUCTIONS:
            evicted_id = self._reconstruction_order.popleft()
            self._reconstructions.pop(evicted_id, None)
            logger.warning("Evicted reconstruction result %s", evicted_id)
        self._reconstructions[result.result_id] = result
        self._reconstruction_order.append(result.result_id)

    def partial_reconstruct(self, fragment_ids: List[str],
                            min_quality: float = 0.5) -> Optional[ReconstructionResult]:
        min_q = _clamp(min_quality)
        result = self.reconstruct(fragment_ids)
        if result.quality >= min_q:
            return result
        return None

    def delete(self, fragment_id: str) -> bool:
        fid = _sanitize(fragment_id, 50)
        with self._lock:
            if fid not in self._fragments:
                return False
            self._fragments.pop(fid, None)
            self._fragment_order = deque(
                [x for x in self._fragment_order if x != fid],
                maxlen=_MAX_FRAGMENTS,
            )
            with self._stats_lock:
                self._stats["deletions"] += 1
        return True

    def find_fragments(self, query: str = "", limit: int = 100) -> List[HologramFragment]:
        query_clean = _sanitize_content(query, 1000).lower()
        limit = max(1, min(500, limit))
        with self._lock:
            fragments = list(self._fragments.values())
            if query_clean:
                fragments = [
                    f for f in fragments
                    if query_clean in f.content.lower()
                    or query_clean in f.source_hash.lower()
                ]
            fragments.sort(key=lambda f: f.reconstruction_quality, reverse=True)
            return [copy.deepcopy(f) for f in fragments[:limit]]

    def get_redundancy_level(self, fragment_id: str) -> float:
        fid = _sanitize(fragment_id, 50)
        with self._lock:
            frag = self._fragments.get(fid)
            if not frag:
                return 0.0
            same_source = [
                f for f in self._fragments.values()
                if f.source_hash == frag.source_hash and f.fragment_id != fid
            ]
            if not same_source:
                return 0.0
            total_peers = len(same_source) + 1
            redundancy = _clamp(min(total_peers / 3.0, 1.0))
            return redundancy

    def verify_integrity(self, fragment_id: str) -> bool:
        fid = _sanitize(fragment_id, 50)
        with self._lock:
            frag = self._fragments.get(fid)
            if not frag:
                return False
            expected_hash = _content_hash(_sanitize_content(frag.content, 50000))
            if frag.fragment_type == HologramType.FULL:
                matches = frag.source_hash == expected_hash
            else:
                matches = len(frag.encoding) > 0
            with self._stats_lock:
                if matches:
                    self._stats["integrity_checks_passed"] += 1
                else:
                    self._stats["integrity_checks_failed"] += 1
            return matches

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            stats["fragment_count"] = len(self._fragments)
            stats["reconstruction_count"] = len(self._reconstructions)
            full = sum(1 for f in self._fragments.values() if f.fragment_type == HologramType.FULL)
            partial = sum(1 for f in self._fragments.values() if f.fragment_type == HologramType.PARTIAL)
            redundant = sum(1 for f in self._fragments.values() if f.fragment_type == HologramType.REDUNDANT)
            stats["full_fragments"] = full
            stats["partial_fragments"] = partial
            stats["redundant_fragments"] = redundant
            stats["avg_quality"] = round(
                sum(f.reconstruction_quality for f in self._fragments.values()) / max(1, len(self._fragments)), 4
            )
            return stats