#!/usr/bin/env python3
"""
EmergentGrammar — Self-Evolving Cognition Language.

Discovers and evolves its own internal language and grammar rules from
observed patterns. Symbols emerge from repeated observations, compose
into structured sentences, and undergo abstraction, merge, and split
operations as the language evolves.

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

logger = logging.getLogger("shadowd.cognition.emergent_grammar")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_RULES = 2000
_MAX_SYMBOLS = 5000
_MAX_SENTENCES = 1000
_MAX_CONTEXTS = 100
_MAX_METADATA_KEYS = 20


class EmergenceType(str, Enum):
    PATTERN_DISCOVERY = "pattern_discovery"
    COMPOSITION = "composition"
    ABSTRACTION = "abstraction"
    REFINEMENT = "refinement"
    MERGE = "merge"
    SPLIT = "split"


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
class GrammarRule:
    rule_id: str
    pattern: str
    replacement: str
    context: str
    frequency: int = 1
    confidence: float = 0.5
    domain: str = ""
    emergence_type: EmergenceType = EmergenceType.PATTERN_DISCOVERY
    metadata: Dict[str, Any] = field(default_factory=dict)
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.rule_id:
            self.rule_id = uuid.uuid4().hex[:12]
        self.rule_id = _sanitize(self.rule_id, 50)
        self.pattern = _sanitize_content(self.pattern, 2000)
        self.replacement = _sanitize_content(self.replacement, 2000)
        self.context = _sanitize_content(self.context, 1000)
        self.frequency = max(1, int(self.frequency))
        self.confidence = _clamp(self.confidence)
        self.domain = _sanitize(self.domain, 100)
        if len(self.metadata) > _MAX_METADATA_KEYS:
            self.metadata = dict(list(self.metadata.items())[:_MAX_METADATA_KEYS])

    def to_dict(self) -> Dict:
        return {
            "rule_id": self.rule_id,
            "pattern": self.pattern,
            "replacement": self.replacement,
            "context": self.context,
            "frequency": self.frequency,
            "confidence": round(self.confidence, 4),
            "domain": self.domain,
            "emergence_type": self.emergence_type.value,
            "metadata": copy.deepcopy(self.metadata),
            "created_at": self.created_at,
        }


@dataclass
class GrammarSymbol:
    symbol_id: str
    symbol: str
    meaning: str
    usage_count: int = 0
    contexts: List[str] = field(default_factory=list)
    entropy: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.symbol_id:
            self.symbol_id = uuid.uuid4().hex[:12]
        self.symbol_id = _sanitize(self.symbol_id, 50)
        self.symbol = _sanitize(self.symbol, 200)
        self.meaning = _sanitize_content(self.meaning, 2000)
        self.usage_count = max(0, int(self.usage_count))
        self.entropy = _clamp(self.entropy)
        self.contexts = [_sanitize(c, 200) for c in (self.contexts or [])[:_MAX_CONTEXTS]]

    def to_dict(self) -> Dict:
        return {
            "symbol_id": self.symbol_id,
            "symbol": self.symbol,
            "meaning": self.meaning,
            "usage_count": self.usage_count,
            "contexts": list(self.contexts),
            "entropy": round(self.entropy, 4),
            "created_at": self.created_at,
        }


@dataclass
class GrammarSentence:
    sentence_id: str
    symbols: List[str]
    structure: str
    meaning: str
    confidence: float = 0.5
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.sentence_id:
            self.sentence_id = uuid.uuid4().hex[:12]
        self.sentence_id = _sanitize(self.sentence_id, 50)
        self.symbols = [_sanitize(s, 100) for s in (self.symbols or [])[:50]]
        self.structure = _sanitize_content(self.structure, 1000)
        self.meaning = _sanitize_content(self.meaning, 2000)
        self.confidence = _clamp(self.confidence)

    def to_dict(self) -> Dict:
        return {
            "sentence_id": self.sentence_id,
            "symbols": list(self.symbols),
            "structure": self.structure,
            "meaning": self.meaning,
            "confidence": round(self.confidence, 4),
            "created_at": self.created_at,
        }


class EmergentGrammar:
    """Self-evolving cognition language that discovers and refines grammar
    rules and symbols from observed patterns."""

    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._rules: Dict[str, GrammarRule] = {}
        self._symbols: Dict[str, GrammarSymbol] = {}
        self._sentences: Dict[str, GrammarSentence] = {}
        self._rule_order: deque = deque(maxlen=_MAX_RULES)
        self._symbol_order: deque = deque(maxlen=_MAX_SYMBOLS)
        self._sentence_order: deque = deque(maxlen=_MAX_SENTENCES)
        self._symbol_by_name: Dict[str, str] = {}
        self._stats = {
            "rules_created": 0,
            "symbols_created": 0,
            "sentences_created": 0,
            "compositions": 0,
            "abstractions": 0,
            "merges": 0,
            "splits": 0,
            "refinements": 0,
        }

    def observe(self, pattern: str, context: str = "",
                confidence: float = 0.5, domain: str = "") -> GrammarRule:
        pattern_clean = _sanitize_content(pattern, 2000)
        context_clean = _sanitize_content(context, 1000)
        rule = GrammarRule(
            rule_id=uuid.uuid4().hex[:12],
            pattern=pattern_clean,
            replacement=pattern_clean,
            context=context_clean,
            frequency=1,
            confidence=_clamp(confidence),
            domain=_sanitize(domain, 100) if domain else "",
            emergence_type=EmergenceType.PATTERN_DISCOVERY,
        )
        with self._lock:
            if len(self._rules) >= _MAX_RULES:
                evicted_id = self._rule_order.popleft()
                self._rules.pop(evicted_id, None)
                logger.warning("Evicted grammar rule %s", evicted_id)
            self._rules[rule.rule_id] = rule
            self._rule_order.append(rule.rule_id)
            with self._stats_lock:
                self._stats["rules_created"] += 1
        return copy.deepcopy(rule)

    def compose(self, symbols: List[str], context: str = "") -> GrammarSentence:
        clean_symbols = [_sanitize(s, 100) for s in (symbols or [])[:50]]
        structure = " ".join(clean_symbols)
        meaning = _sanitize_content(context, 2000) if context else structure
        sentence = GrammarSentence(
            sentence_id=uuid.uuid4().hex[:12],
            symbols=clean_symbols,
            structure=structure,
            meaning=meaning,
            confidence=0.5,
        )
        with self._lock:
            if len(self._sentences) >= _MAX_SENTENCES:
                evicted_id = self._sentence_order.popleft()
                self._sentences.pop(evicted_id, None)
                logger.warning("Evicted grammar sentence %s", evicted_id)
            self._sentences[sentence.sentence_id] = sentence
            self._sentence_order.append(sentence.sentence_id)
            for sym_str in clean_symbols:
                sid = self._symbol_by_name.get(sym_str)
                if sid:
                    sym = self._symbols.get(sid)
                    if sym:
                        sym.usage_count += 1
            with self._stats_lock:
                self._stats["sentences_created"] += 1
                self._stats["compositions"] += 1
        return copy.deepcopy(sentence)

    def abstract(self, pattern: str, level: int = 1) -> GrammarSymbol:
        pattern_clean = _sanitize_content(pattern, 2000)
        level = max(1, min(10, int(level)))
        abbr = pattern_clean[:max(3, 8 - level)]
        symbol = GrammarSymbol(
            symbol_id=uuid.uuid4().hex[:12],
            symbol=abbr,
            meaning=pattern_clean,
            usage_count=0,
            contexts=[],
            entropy=_clamp(0.3 + 0.07 * level),
        )
        with self._lock:
            if len(self._symbols) >= _MAX_SYMBOLS:
                evicted_id = self._symbol_order.popleft()
                evicted_sym = self._symbols.pop(evicted_id, None)
                if evicted_sym:
                    self._symbol_by_name.pop(evicted_sym.symbol, None)
                logger.warning("Evicted grammar symbol %s", evicted_id)
            self._symbols[symbol.symbol_id] = symbol
            self._symbol_order.append(symbol.symbol_id)
            self._symbol_by_name[symbol.symbol] = symbol.symbol_id
            with self._stats_lock:
                self._stats["symbols_created"] += 1
                self._stats["abstractions"] += 1
        return copy.deepcopy(symbol)

    def refine(self, rule_id: str, new_confidence: float) -> bool:
        rid = _sanitize(rule_id, 50)
        with self._lock:
            rule = self._rules.get(rid)
            if not rule:
                return False
            rule.confidence = _clamp(new_confidence)
            rule.frequency += 1
            rule.emergence_type = EmergenceType.REFINEMENT
            with self._stats_lock:
                self._stats["refinements"] += 1
        return True

    def merge_symbols(self, symbol_id_1: str, symbol_id_2: str,
                      new_symbol: str) -> Optional[GrammarSymbol]:
        sid1 = _sanitize(symbol_id_1, 50)
        sid2 = _sanitize(symbol_id_2, 50)
        new_sym_clean = _sanitize(new_symbol, 200)
        with self._lock:
            s1 = self._symbols.get(sid1)
            s2 = self._symbols.get(sid2)
            if not s1 or not s2:
                return None
            merged_contexts = list(set(s1.contexts + s2.contexts))[:_MAX_CONTEXTS]
            merged = GrammarSymbol(
                symbol_id=uuid.uuid4().hex[:12],
                symbol=new_sym_clean,
                meaning=f"Merge of {s1.symbol} and {s2.symbol}",
                usage_count=s1.usage_count + s2.usage_count,
                contexts=merged_contexts,
                entropy=_clamp((s1.entropy + s2.entropy) / 2),
            )
            if len(self._symbols) >= _MAX_SYMBOLS:
                evicted_id = self._symbol_order.popleft()
                evicted_sym = self._symbols.pop(evicted_id, None)
                if evicted_sym:
                    self._symbol_by_name.pop(evicted_sym.symbol, None)
                logger.warning("Evicted grammar symbol %s on merge", evicted_id)
            self._symbols.pop(sid1, None)
            self._symbols.pop(sid2, None)
            self._symbol_by_name.pop(s1.symbol, None)
            self._symbol_by_name.pop(s2.symbol, None)
            self._symbol_order = deque(
                [x for x in self._symbol_order if x not in (sid1, sid2)],
                maxlen=_MAX_SYMBOLS,
            )
            self._symbols[merged.symbol_id] = merged
            self._symbol_order.append(merged.symbol_id)
            self._symbol_by_name[merged.symbol] = merged.symbol_id
            with self._stats_lock:
                self._stats["merges"] += 1
        return copy.deepcopy(merged)

    def split_symbol(self, symbol_id: str,
                     contexts: Optional[List[str]] = None) -> List[GrammarSymbol]:
        sid = _sanitize(symbol_id, 50)
        clean_contexts = [_sanitize(c, 200) for c in (contexts or [])[:_MAX_CONTEXTS]]
        results = []
        with self._lock:
            sym = self._symbols.get(sid)
            if not sym:
                return []
            if not clean_contexts:
                clean_contexts = sym.contexts[:2] if len(sym.contexts) >= 2 else ["ctx_a", "ctx_b"]
            for i, ctx in enumerate(clean_contexts[:_MAX_CONTEXTS]):
                new_sym = GrammarSymbol(
                    symbol_id=uuid.uuid4().hex[:12],
                    symbol=f"{sym.symbol}_{i}",
                    meaning=f"Split of {sym.symbol} for {ctx}",
                    usage_count=max(1, sym.usage_count // len(clean_contexts)),
                    contexts=[ctx],
                    entropy=_clamp(sym.entropy + 0.1),
                )
                if len(self._symbols) >= _MAX_SYMBOLS:
                    evicted_id = self._symbol_order.popleft()
                    evicted_sym = self._symbols.pop(evicted_id, None)
                    if evicted_sym:
                        self._symbol_by_name.pop(evicted_sym.symbol, None)
                    logger.warning("Evicted grammar symbol %s on split", evicted_id)
                self._symbols[new_sym.symbol_id] = new_sym
                self._symbol_order.append(new_sym.symbol_id)
                self._symbol_by_name[new_sym.symbol] = new_sym.symbol_id
                results.append(new_sym)
            self._symbols.pop(sid, None)
            self._symbol_by_name.pop(sym.symbol, None)
            self._symbol_order = deque(
                [x for x in self._symbol_order if x != sid],
                maxlen=_MAX_SYMBOLS,
            )
            with self._stats_lock:
                self._stats["splits"] += 1
        return [copy.deepcopy(r) for r in results]

    def find_rules(self, domain: str = "", limit: int = 100) -> List[GrammarRule]:
        dom = _sanitize(domain, 100)
        limit = max(1, min(500, limit))
        with self._lock:
            rules = list(self._rules.values())
            if dom:
                rules = [r for r in rules if r.domain == dom]
            rules.sort(key=lambda r: r.confidence, reverse=True)
            return [copy.deepcopy(r) for r in rules[:limit]]

    def find_symbols(self, min_usage: int = 0, limit: int = 100) -> List[GrammarSymbol]:
        min_usage = max(0, int(min_usage))
        limit = max(1, min(500, limit))
        with self._lock:
            symbols = list(self._symbols.values())
            symbols = [s for s in symbols if s.usage_count >= min_usage]
            symbols.sort(key=lambda s: s.usage_count, reverse=True)
            return [copy.deepcopy(s) for s in symbols[:limit]]

    def generate_sentence(self, context: str = "",
                          min_confidence: float = 0.3) -> Optional[GrammarSentence]:
        ctx_clean = _sanitize_content(context, 1000)
        min_conf = _clamp(min_confidence)
        with self._lock:
            eligible = [
                (r, r.confidence) for r in self._rules.values()
                if r.confidence >= min_conf
            ]
            if not eligible:
                return None
            eligible.sort(key=lambda x: x[1], reverse=True)
            best_rules = eligible[:5]
            symbols = [r.pattern for r, _ in best_rules]
            combined_meaning = " ".join(r.context for r, _ in best_rules if r.context)
            if ctx_clean:
                combined_meaning = ctx_clean + " " + combined_meaning
            sentence = GrammarSentence(
                sentence_id=uuid.uuid4().hex[:12],
                symbols=symbols,
                structure=" ".join(symbols),
                meaning=_sanitize_content(combined_meaning, 2000),
                confidence=_clamp(sum(c for _, c in best_rules) / len(best_rules)),
            )
            if len(self._sentences) >= _MAX_SENTENCES:
                evicted_id = self._sentence_order.popleft()
                self._sentences.pop(evicted_id, None)
                logger.warning("Evicted grammar sentence %s", evicted_id)
            self._sentences[sentence.sentence_id] = sentence
            self._sentence_order.append(sentence.sentence_id)
            with self._stats_lock:
                self._stats["sentences_created"] += 1
        return copy.deepcopy(sentence)

    def list_sentences(self, limit: int = 100, offset: int = 0) -> List[GrammarSentence]:
        limit = max(1, min(500, limit))
        offset = max(0, offset)
        with self._lock:
            sids = list(self._sentence_order)[offset:offset + limit]
            return [copy.deepcopy(self._sentences[sid]) for sid in sids if sid in self._sentences]

    def get_stats(self) -> Dict:
        with self._lock:
            stats = dict(self._stats)
            stats["rule_count"] = len(self._rules)
            stats["symbol_count"] = len(self._symbols)
            stats["sentence_count"] = len(self._sentences)
            return stats