"""Tests for the TF-IDF semantic index used by Perfect Memory."""

import pytest

from shikigami.semantic_memory import (
    SemanticIndex,
    semantic_search,
    _observation_to_text,
    _tokenize,
)


def _obs(otype, **content):
    return {"type": otype, "content": content, "timestamp": "2026-05-01T10:00:00+00:00"}


class TestTokenizer:
    def test_drops_stopwords(self):
        toks = _tokenize("the quick brown fox jumps over a lazy dog")
        assert "the" not in toks
        assert "over" not in toks
        assert "quick" in toks
        assert "brown" in toks

    def test_drops_short_tokens(self):
        toks = _tokenize("a be cat dog elephant")
        assert "a" not in toks
        assert "be" not in toks
        assert "cat" in toks
        assert "elephant" in toks

    def test_lowercases(self):
        toks = _tokenize("Database Schema MIGRATION")
        assert toks == ["database", "schema", "migration"]


class TestObservationFlattening:
    def test_dict_content(self):
        text = _observation_to_text(_obs("note", text="hello world"))
        assert "note" in text
        assert "hello world" in text

    def test_string_content(self):
        obs = {"type": "clipboard", "content": "raw string content"}
        text = _observation_to_text(obs)
        assert "raw string content" in text
        assert "clipboard" in text

    def test_handles_missing_content(self):
        text = _observation_to_text({"type": "ping"})
        assert "ping" in text


class TestSemanticIndex:
    def test_empty_observations(self):
        idx = SemanticIndex([])
        assert idx.query("anything") == []

    def test_finds_obvious_match(self):
        obs = [
            _obs("note", text="reviewed the database migration plan"),
            _obs("note", text="had coffee with Sarah"),
            _obs("note", text="bought groceries"),
        ]
        results = semantic_search(obs, "database migration", min_score=0.05)
        assert len(results) >= 1
        assert "database" in results[0]["_matched_terms"] or "migration" in results[0]["_matched_terms"]

    def test_ranks_more_specific_higher(self):
        obs = [
            _obs("note", text="something unrelated"),
            _obs("note", text="database migration schema update"),
            _obs("note", text="quick mention of database"),
        ]
        results = semantic_search(obs, "database migration schema", min_score=0.01)
        assert len(results) >= 2
        # The "database migration schema update" entry should score higher
        # than the single-keyword "quick mention of database".
        assert results[0]["_score"] > results[-1]["_score"]

    def test_no_match_below_threshold(self):
        obs = [_obs("note", text="quantum physics paper")]
        results = semantic_search(obs, "lunch with friends", min_score=0.1)
        assert results == []

    def test_top_k_limits_results(self):
        obs = [_obs("note", text=f"database entry number {i}") for i in range(20)]
        results = semantic_search(obs, "database entry", top_k=5)
        assert len(results) <= 5

    def test_score_in_unit_range(self):
        obs = [_obs("note", text="database schema review")]
        results = semantic_search(obs, "database schema")
        assert all(0.0 <= r["_score"] <= 1.0 for r in results)

    def test_explain_returns_shared_terms(self):
        obs = [_obs("note", text="negotiated the contract terms successfully")]
        idx = SemanticIndex(obs)
        hits = idx.query("contract negotiation terms")
        assert hits
        explained = idx.explain("contract negotiation terms", hits[0][0])
        # `negotiated` won't match because we don't stem; `contract` and `terms`
        # are exact word matches.
        assert any(t in {"contract", "terms"} for t in explained)
