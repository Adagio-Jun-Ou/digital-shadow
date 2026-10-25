"""
TF-IDF semantic search for Perfect Memory.

Augments the existing keyword / temporal / entity searches in _perfect_memory
with a vector-space relevance score so queries match observations that share
*meaning* (token overlap weighted by inverse document frequency), not just
exact substrings.

Pure-numpy implementation — no sklearn / transformer dependencies. If the
caller has Ollama embeddings configured, swap the `_vectorize` strategy by
subclassing.
"""

from __future__ import annotations

import json
import math
import re
from collections import Counter
from typing import Dict, List, Sequence, Tuple

import numpy as np

# Strip punctuation, lowercase, keep alphanumerics + dashes. Short tokens and
# common English stopwords add noise to a small index, so drop them.
_TOKEN_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9\-]*")

_STOPWORDS = frozenset({
    "a", "an", "the", "and", "or", "but", "if", "while", "with", "without",
    "of", "for", "to", "from", "in", "on", "at", "by", "is", "are", "was",
    "were", "be", "been", "being", "have", "has", "had", "do", "does", "did",
    "this", "that", "these", "those", "it", "its", "as", "than", "then",
    "what", "when", "where", "which", "who", "whom", "why", "how",
    "i", "me", "my", "you", "your", "we", "us", "our", "they", "them",
    "their", "he", "she", "his", "her",
    "over", "under", "into", "out", "up", "down", "off", "about", "between",
})


def _tokenize(text: str) -> List[str]:
    return [
        t.lower() for t in _TOKEN_RE.findall(text)
        if len(t) >= 3 and t.lower() not in _STOPWORDS
    ]


def _observation_to_text(obs: Dict) -> str:
    """Flatten an observation dict into the text we'll index."""
    parts = [str(obs.get("type", "")), str(obs.get("source", ""))]
    content = obs.get("content")
    if isinstance(content, dict):
        parts.extend(str(v) for v in content.values() if v)
    elif isinstance(content, str):
        parts.append(content)
    elif content is not None:
        parts.append(json.dumps(content))
    return " ".join(parts)


class SemanticIndex:
    """In-memory TF-IDF index. Rebuilds cheaply; intended to be constructed
    per Perfect Memory query, not cached across calls (observation lists are
    small enough that the cost is dominated by tokenization, not arithmetic)."""

    def __init__(self, observations: Sequence[Dict]):
        self.observations = list(observations)
        self._tokens: List[List[str]] = [_tokenize(_observation_to_text(o)) for o in self.observations]
        self.vocab: Dict[str, int] = {}
        self.idf: np.ndarray = np.zeros(0)
        self._tfidf: np.ndarray = np.zeros((0, 0))
        self._norms: np.ndarray = np.zeros(0)
        self._build()

    def _build(self) -> None:
        # Vocab: every token that appears in at least one doc.
        vocab_set: Dict[str, int] = {}
        for toks in self._tokens:
            for t in toks:
                vocab_set.setdefault(t, len(vocab_set))
        self.vocab = vocab_set
        n_docs = len(self._tokens)
        n_terms = len(vocab_set)
        if n_docs == 0 or n_terms == 0:
            return

        # Document frequency per term.
        df = np.zeros(n_terms, dtype=np.float64)
        tf = np.zeros((n_docs, n_terms), dtype=np.float64)
        for i, toks in enumerate(self._tokens):
            counts = Counter(toks)
            for term, c in counts.items():
                j = vocab_set[term]
                tf[i, j] = c
            for term in counts:
                df[vocab_set[term]] += 1

        # Smoothed IDF (matches sklearn's default formula).
        self.idf = np.log((1.0 + n_docs) / (1.0 + df)) + 1.0
        self._tfidf = tf * self.idf
        self._norms = np.linalg.norm(self._tfidf, axis=1)
        # Avoid division-by-zero for empty docs.
        self._norms[self._norms == 0] = 1.0

    def query(self, text: str, top_k: int = 10, min_score: float = 0.05) -> List[Tuple[int, float]]:
        """Return [(observation_index, score), ...] sorted by descending score.
        Scores are cosine similarity in [0, 1]."""
        if self._tfidf.size == 0:
            return []
        q_tokens = _tokenize(text)
        if not q_tokens:
            return []
        q_counts = Counter(q_tokens)
        q_vec = np.zeros(len(self.vocab), dtype=np.float64)
        for term, c in q_counts.items():
            j = self.vocab.get(term)
            if j is not None:
                q_vec[j] = c
        q_vec = q_vec * self.idf
        q_norm = np.linalg.norm(q_vec)
        if q_norm == 0:
            return []

        scores = (self._tfidf @ q_vec) / (self._norms * q_norm)
        ranked = np.argsort(-scores)
        return [
            (int(i), float(scores[i]))
            for i in ranked[:top_k]
            if scores[i] >= min_score
        ]

    def explain(self, text: str, idx: int, top_terms: int = 5) -> List[str]:
        """Top contributing terms shared between query and observation idx."""
        if idx < 0 or idx >= len(self._tokens):
            return []
        q_terms = set(_tokenize(text))
        shared = [t for t in self._tokens[idx] if t in q_terms]
        if not shared:
            return []
        # Rank shared terms by IDF (rarer = more informative).
        scored = sorted(
            ((t, self.idf[self.vocab[t]]) for t in set(shared)),
            key=lambda x: -x[1],
        )
        return [t for t, _ in scored[:top_terms]]


def semantic_search(
    observations: Sequence[Dict],
    query: str,
    top_k: int = 10,
    min_score: float = 0.05,
) -> List[Dict]:
    """Convenience wrapper used by Perfect Memory. Returns observations with
    `_score` and `_matched_terms` annotations, ranked by relevance."""
    idx = SemanticIndex(observations)
    out: List[Dict] = []
    for i, score in idx.query(query, top_k=top_k, min_score=min_score):
        o = dict(observations[i])
        o["_score"] = round(score, 3)
        o["_matched_terms"] = idx.explain(query, i)
        out.append(o)
    return out
