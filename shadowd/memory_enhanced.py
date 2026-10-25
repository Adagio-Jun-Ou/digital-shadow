#!/usr/bin/env python3
"""
Enhanced Memory System for Shadow AI
Episode-based memory with importance scoring
"""

import copy
import json
import re as _re
import threading
import uuid
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional
from pathlib import Path
from dataclasses import dataclass, field
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()

@dataclass
class Episode:
    id: str
    timestamp: datetime
    type: str
    content: str
    metadata: Dict = field(default_factory=dict)
    importance: float = 0.5
    access_count: int = 0
    
    def __post_init__(self):
        self.type = _sanitize(self.type)
        self.content = _sanitize_content(self.content)
        self.importance = max(0.0, min(1.0, self.importance))

    def to_dict(self):
        return {
            "id": self.id,
            "timestamp": self.timestamp.isoformat(),
            "type": self.type,
            "content": self.content,
            "importance": round(self.importance, 3),
            "metadata": copy.deepcopy(self.metadata)
        }

class EnhancedMemory:
    _MAX_EPISODES = 10000

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or Path.home() / "digital-shadow" / "data"
        self.episodes: List[Episode] = []
        self.core_memory: Dict[str, str] = {}
        self._lock = threading.Lock()

    def add_episode(self, type: str, content: str, importance: float = None) -> Episode:
        if importance is None:
            importance = 0.9 if type == "commitment" else 0.5
        importance = max(0.0, min(1.0, importance))
        
        episode = Episode(
            id=f"ep_{uuid.uuid4().hex[:12]}",
            timestamp=datetime.now(timezone.utc),
            type=type,
            content=_sanitize_content(content)[:100000],
            importance=importance
        )
        with self._lock:
            self.episodes.append(episode)
            if len(self.episodes) > self._MAX_EPISODES:
                self.episodes = self.episodes[-self._MAX_EPISODES:]
        return episode

    def add_to_core(self, key: str, value: str):
        """Add or update a core memory key-value pair."""
        key = _sanitize(key)
        value = _sanitize_content(str(value)).strip()[:10000]
        with self._lock:
            self.core_memory[key] = value
    
    def search(self, query: str, limit: int = 10) -> List[Episode]:
        query_lower = query.lower()
        with self._lock:
            results = [ep for ep in self.episodes if query_lower in ep.content.lower()]
            return copy.deepcopy(sorted(results, key=lambda e: e.importance, reverse=True)[:limit])
    
    def get_context(self, max_items: int = 10) -> str:
        with self._lock:
            parts = []
            if self.core_memory:
                parts.append("## Core Memory")
                for k, v in self.core_memory.items():
                    parts.append(f"{k}: {v}")
            
            important = sorted(self.episodes, key=lambda e: e.importance, reverse=True)[:5]
            if important:
                parts.append("\n## Important")
                for ep in important:
                    parts.append(f"- {ep.content.strip()[:50]}")
            
            return "\n".join(parts)
    
    def get_stats(self) -> Dict:
        with self._lock:
            return {
                "total_episodes": len(self.episodes),
                "by_type": {t: len([e for e in self.episodes if e.type == t]) 
                           for t in set(e.type for e in self.episodes)}
            }

if __name__ == "__main__":
    import tempfile
    m = EnhancedMemory(Path(tempfile.mkdtemp()))
    m.add_episode("commitment", "Call John")
    print(m.get_context())
    print(m.get_stats())
