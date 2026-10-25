"""
Pre-parsed, indexed observation facade.

When the council fans out to N abilities, each ability iterates the full
observation list and re-parses every timestamp. For 100 obs × 21
abilities that's 2100 redundant parses; for 5000 obs × 21 that's 100k.

This module is opt-in. The Council can hand a `ParsedObservations`
instance via `context["observations"]` instead of raw dicts — abilities
that don't know about it still work (it iterates like a list).

Abilities that *do* know about it can call `.by_type(type)` and
`.timestamp_for(obs)` to skip repeated work.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime, timezone
from typing import Any, Dict, Iterable, List, Optional


def _parse_ts(ts: Any) -> Optional[datetime]:
    if not ts:
        return None
    if isinstance(ts, datetime):
        return ts if ts.tzinfo else ts.replace(tzinfo=timezone.utc)
    try:
        t = datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        return t if t.tzinfo else t.replace(tzinfo=timezone.utc)
    except (ValueError, TypeError):
        return None


class ParsedObservations(Sequence):
    """List-compatible facade with timestamp caching and type indexing.

    Behaves like the original list for any ability that just iterates or
    indexes into it. Abilities aware of the facade can use:

        obs.timestamp_for(o)   # cached parsed datetime or None
        obs.by_type("note")    # iterator over obs of one type (cached)
        obs.types              # set of types seen
    """

    def __init__(self, raw: Iterable[Dict]):
        self._raw: List[Dict] = [o for o in raw if isinstance(o, dict)]
        self._ts_cache: Dict[int, Optional[datetime]] = {}
        self._type_index: Optional[Dict[str, List[Dict]]] = None

    # Sequence protocol -----------------------------------------------------
    def __len__(self) -> int:
        return len(self._raw)

    def __getitem__(self, idx):
        return self._raw[idx]

    def __iter__(self):
        return iter(self._raw)

    def __contains__(self, item):
        return item in self._raw

    # Cache API -------------------------------------------------------------
    def timestamp_for(self, obs: Dict) -> Optional[datetime]:
        key = id(obs)
        if key in self._ts_cache:
            return self._ts_cache[key]
        ts = _parse_ts(obs.get("timestamp")) if isinstance(obs, dict) else None
        self._ts_cache[key] = ts
        return ts

    def precompute_timestamps(self) -> None:
        """Parse every timestamp up front. Safe to call multiple times."""
        for o in self._raw:
            self.timestamp_for(o)

    def _ensure_type_index(self) -> Dict[str, List[Dict]]:
        if self._type_index is None:
            idx: Dict[str, List[Dict]] = {}
            for o in self._raw:
                t = o.get("type", "")
                idx.setdefault(t, []).append(o)
            self._type_index = idx
        return self._type_index

    def by_type(self, type_name: str) -> List[Dict]:
        return self._ensure_type_index().get(type_name, [])

    @property
    def types(self) -> set:
        return set(self._ensure_type_index().keys())

    @property
    def raw(self) -> List[Dict]:
        return self._raw
