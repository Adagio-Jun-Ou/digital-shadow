"""
Embedding-backed semantic index for Perfect Memory.

Talks to an Ollama-compatible embeddings endpoint (local `ollama serve` or
remote Ollama Cloud) and caches vectors on disk by content hash so repeated
queries don't re-encode the same observations.

When no embeddings backend is configured / reachable, callers should fall
back to TF-IDF (`shikigami.semantic_memory`). This module's `embed_texts`
returns None to signal that — never raises.

Configuration:
- OLLAMA_EMBED_BASE   = "http://127.0.0.1:11434/api/embed"   (default; local)
                       or "https://ollama.com/api/embed"
- OLLAMA_EMBED_MODEL  = "nomic-embed-text"                   (default)
- OLLAMA_EMBED_API_KEY = optional bearer token
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

logger = logging.getLogger("shadowd.embeddings")

_EMBED_BASE = os.environ.get("OLLAMA_EMBED_BASE", "http://127.0.0.1:11434/api/embed")
_EMBED_MODEL = os.environ.get("OLLAMA_EMBED_MODEL", "nomic-embed-text")
_EMBED_API_KEY = (
    os.environ.get("OLLAMA_EMBED_API_KEY")
    or os.environ.get("LLM_API_KEY")
    or os.environ.get("OLLAMA_API_KEY", "")
)
_EMBED_TIMEOUT = float(os.environ.get("OLLAMA_EMBED_TIMEOUT", "20"))

_CACHE_PATH = Path(os.environ.get(
    "OLLAMA_EMBED_CACHE",
    str(Path.home() / "digital-shadow" / "data" / "memory" / "embeddings.json"),
))

_BACKEND_PROBED = False
_BACKEND_OK: Optional[bool] = None
_BACKEND_LOCK = threading.Lock()


def _content_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()[:32]


class EmbeddingCache:
    """Persistent JSON cache keyed by content hash."""

    def __init__(self, path: Path = _CACHE_PATH, max_entries: int = 5000):
        self.path = path
        self.max_entries = max_entries
        self._lock = threading.Lock()
        self._cache: Dict[str, List[float]] = {}
        self._load()

    def _load(self) -> None:
        if not self.path.exists():
            return
        try:
            data = json.loads(self.path.read_text())
            if isinstance(data, dict):
                self._cache = {
                    str(k): list(v) for k, v in data.items()
                    if isinstance(v, list) and v and all(isinstance(x, (int, float)) for x in v)
                }
        except (OSError, json.JSONDecodeError, TypeError) as e:
            logger.warning("embedding cache load failed: %s", type(e).__name__)
            self._cache = {}

    def get(self, text: str) -> Optional[List[float]]:
        with self._lock:
            return self._cache.get(_content_hash(text))

    def set(self, text: str, vec: List[float]) -> None:
        with self._lock:
            self._cache[_content_hash(text)] = vec
            # Evict oldest entries when oversized (FIFO is good enough — the
            # hashes are content-stable so re-encode cost is minimal).
            if len(self._cache) > self.max_entries:
                excess = len(self._cache) - self.max_entries
                for k in list(self._cache.keys())[:excess]:
                    del self._cache[k]

    def save(self) -> None:
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            tmp = self.path.with_suffix(".tmp")
            with self._lock:
                tmp.write_text(json.dumps(self._cache))
            tmp.replace(self.path)
        except OSError as e:
            logger.warning("embedding cache save failed: %s", type(e).__name__)


_CACHE_SINGLETON: Optional[EmbeddingCache] = None


def _cache() -> EmbeddingCache:
    global _CACHE_SINGLETON
    if _CACHE_SINGLETON is None:
        _CACHE_SINGLETON = EmbeddingCache()
    return _CACHE_SINGLETON


def _request_embed(text: str) -> Optional[List[float]]:
    """Single-text encode against the configured backend. Returns None on
    failure. Handles two endpoint shapes — Ollama-native and OpenAI-compatible."""
    try:
        import httpx
    except ImportError:
        return None

    headers = {"Content-Type": "application/json"}
    if _EMBED_API_KEY:
        headers["Authorization"] = f"Bearer {_EMBED_API_KEY}"

    # OpenAI shape: { "model": ..., "input": "text" } → { "data": [{"embedding": [...]}] }
    openai_style = _EMBED_BASE.rstrip("/").endswith(("/v1/embeddings", "/embeddings"))

    payload = (
        {"model": _EMBED_MODEL, "input": text}
        if openai_style else
        {"model": _EMBED_MODEL, "input": text}  # Ollama native also accepts "input"
    )

    try:
        with httpx.Client(timeout=_EMBED_TIMEOUT) as client:
            r = client.post(_EMBED_BASE, headers=headers, json=payload)
            if r.status_code != 200:
                logger.debug("embed HTTP %d: %s", r.status_code, r.text[:200])
                return None
            data = r.json()
            # OpenAI shape
            if isinstance(data, dict) and "data" in data and data["data"]:
                emb = data["data"][0].get("embedding")
                if isinstance(emb, list):
                    return emb
            # Ollama native shape: { "embeddings": [[...]] } or { "embedding": [...] }
            if isinstance(data, dict):
                if "embeddings" in data and isinstance(data["embeddings"], list) and data["embeddings"]:
                    first = data["embeddings"][0]
                    if isinstance(first, list):
                        return first
                if "embedding" in data and isinstance(data["embedding"], list):
                    return data["embedding"]
            return None
    except Exception as e:
        logger.debug("embed request error: %s", type(e).__name__)
        return None


def backend_available(force_recheck: bool = False) -> bool:
    """Probe the configured embeddings endpoint once and cache the result."""
    global _BACKEND_PROBED, _BACKEND_OK
    with _BACKEND_LOCK:
        if not force_recheck and _BACKEND_PROBED:
            return bool(_BACKEND_OK)
        _BACKEND_PROBED = True
        vec = _request_embed("ping")
        _BACKEND_OK = vec is not None and len(vec) > 0
        if _BACKEND_OK:
            logger.info("embeddings backend reachable: %s (%s, %d dims)",
                        _EMBED_BASE, _EMBED_MODEL, len(vec))
        else:
            logger.info("embeddings backend unavailable; Perfect Memory will use TF-IDF only")
        return _BACKEND_OK


def embed_texts(texts: Sequence[str]) -> Optional[List[List[float]]]:
    """Embed many texts, using cache where possible. Returns None if the
    backend is unreachable (caller should fall back to TF-IDF)."""
    if not texts:
        return []
    if not backend_available():
        return None

    cache = _cache()
    out: List[Optional[List[float]]] = [None] * len(texts)
    misses: List[Tuple[int, str]] = []

    for i, t in enumerate(texts):
        hit = cache.get(t) if t else None
        if hit is not None:
            out[i] = hit
        elif t:
            misses.append((i, t))

    if misses:
        # Encode misses one at a time; the Ollama native endpoint takes a single
        # input per call. For Cloud-scale workloads this is the bottleneck —
        # the cache handles repeated lookups.
        for idx, text in misses:
            vec = _request_embed(text)
            if vec is None:
                # Mid-batch failure — give up on this batch, return None so the
                # caller falls back to TF-IDF rather than partial results.
                return None
            out[idx] = vec
            cache.set(text, vec)
        cache.save()

    # Replace any remaining Nones (empty texts) with zero-vectors so the index
    # has a consistent shape.
    dim = next((len(v) for v in out if v), 0)
    if dim == 0:
        return None
    return [v if v is not None else [0.0] * dim for v in out]


def cosine_search(
    docs: Sequence[str],
    query: str,
    top_k: int = 5,
    min_score: float = 0.5,
) -> List[Tuple[int, float]]:
    """Embedding-based similarity search. Returns [(doc_index, score), ...]
    sorted descending. Empty list if backend unavailable — caller falls back."""
    if not docs or not query:
        return []
    doc_vecs = embed_texts(list(docs))
    if doc_vecs is None:
        return []
    q_vecs = embed_texts([query])
    if not q_vecs:
        return []

    doc_arr = np.asarray(doc_vecs, dtype=np.float64)
    q = np.asarray(q_vecs[0], dtype=np.float64)
    doc_norms = np.linalg.norm(doc_arr, axis=1)
    q_norm = np.linalg.norm(q)
    if q_norm == 0:
        return []
    doc_norms[doc_norms == 0] = 1.0  # avoid divide-by-zero
    scores = (doc_arr @ q) / (doc_norms * q_norm)
    order = np.argsort(-scores)
    return [
        (int(i), float(scores[i]))
        for i in order[:top_k]
        if scores[i] >= min_score
    ]
