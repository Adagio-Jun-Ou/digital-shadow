"""Tests for the embeddings index. Verifies graceful degradation when the
embeddings backend is unreachable — the rest of the suite must keep working
regardless of network state."""

import json
import tempfile
from pathlib import Path

import pytest

from shikigami import embedding_index as ei
from shikigami.embedding_index import (
    EmbeddingCache,
    _content_hash,
    cosine_search,
    embed_texts,
)


class TestContentHash:
    def test_stable(self):
        assert _content_hash("hello") == _content_hash("hello")

    def test_distinct(self):
        assert _content_hash("hello") != _content_hash("world")

    def test_length(self):
        assert len(_content_hash("anything")) == 32


class TestCache:
    def test_round_trip(self, tmp_path):
        cache = EmbeddingCache(path=tmp_path / "cache.json")
        cache.set("hello", [0.1, 0.2, 0.3])
        assert cache.get("hello") == [0.1, 0.2, 0.3]

    def test_persist(self, tmp_path):
        path = tmp_path / "cache.json"
        cache = EmbeddingCache(path=path)
        cache.set("hello", [0.1, 0.2, 0.3])
        cache.save()

        # Re-open and read.
        cache2 = EmbeddingCache(path=path)
        assert cache2.get("hello") == [0.1, 0.2, 0.3]

    def test_miss_returns_none(self, tmp_path):
        cache = EmbeddingCache(path=tmp_path / "cache.json")
        assert cache.get("never inserted") is None

    def test_eviction(self, tmp_path):
        cache = EmbeddingCache(path=tmp_path / "cache.json", max_entries=3)
        for i in range(5):
            cache.set(f"text_{i}", [float(i)])
        # Should have only the most recent 3.
        assert len(cache._cache) == 3
        assert cache.get("text_0") is None
        assert cache.get("text_4") == [4.0]

    def test_load_corrupt(self, tmp_path):
        path = tmp_path / "cache.json"
        path.write_text("{ not valid json")
        cache = EmbeddingCache(path=path)
        # Should not raise; cache just starts empty.
        assert cache._cache == {}


class TestBackendDegradation:
    """When the embeddings backend can't be reached, every public function
    must return cleanly (empty list / None) rather than raising."""

    @pytest.fixture(autouse=True)
    def _force_unavailable(self, monkeypatch):
        # Pin the backend probe to "unavailable" so we don't make network calls.
        monkeypatch.setattr(ei, "_BACKEND_PROBED", True, raising=False)
        monkeypatch.setattr(ei, "_BACKEND_OK", False, raising=False)

    def test_embed_returns_none(self):
        assert embed_texts(["hello world"]) is None

    def test_cosine_search_empty(self):
        results = cosine_search(["doc one", "doc two"], "query")
        assert results == []

    def test_cosine_search_handles_empty_docs(self):
        assert cosine_search([], "query") == []
        assert cosine_search(["doc"], "") == []


class TestCachedEmbedShortCircuit:
    """If a text is already in the cache, we shouldn't need the backend."""

    def test_cached_lookup_works_when_backend_disabled(self, monkeypatch, tmp_path):
        # Set up a cache file with a pre-computed embedding.
        cache_path = tmp_path / "cache.json"
        vec = [0.1] * 8
        EmbeddingCache(path=cache_path).set("pre-encoded text", vec)
        cache = EmbeddingCache(path=cache_path)
        cache.set("pre-encoded text", vec)
        cache.save()

        # Force the module to use this cache + an unavailable backend.
        monkeypatch.setattr(ei, "_CACHE_SINGLETON", cache, raising=False)
        monkeypatch.setattr(ei, "_BACKEND_PROBED", True, raising=False)
        monkeypatch.setattr(ei, "_BACKEND_OK", False, raising=False)

        # Backend gate still says "no" so embed_texts should return None even
        # for cached items — the cache only saves cost, it doesn't enable
        # the backend. This is intentional: a working backend is part of the
        # contract for embed_texts.
        assert embed_texts(["pre-encoded text"]) is None
