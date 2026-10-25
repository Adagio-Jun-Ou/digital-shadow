#!/usr/bin/env python3
"""
DarkMemory - Unified Memory System for Shadow AI

Consolidates the 3+ existing memory systems into one coherent architecture:
- SQLite for structured queries (observations, decisions, people)
- In-memory vector index for semantic search
- Temporal Knowledge Graph for entity relationships over time
- Memory blocks (Letta-style) for always-in-context info

Architecture:
  DarkMemory
  ├── BlockStore (Core/Persona/Human/Domain blocks - Letta style)
  ├── ObservationStore (SQLite - fast structured queries)
  ├── VectorIndex (Semantic similarity search)
  ├── TemporalKG (Entity-relation graph with time)
  └── MemoryCompactor (Auto-summarization + archival)

Design Principles (12-Factor Agents):
- Memory is a first-class primitive
- Context engineering > prompt engineering
- All state loaded fresh from external store each invocation
- Compact errors into context, don't dump raw data
"""

import asyncio
import copy
import json
import hashlib
import math
import sqlite3
import threading
import time
import logging
import re
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from collections import Counter, defaultdict

logger = logging.getLogger("shadowd.cognition.dark_memory")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_ENTITIES = 5000
_MAX_RELATIONS = 10000
_MAX_ENTITY_EPISODES = 100


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


class MemoryTier(str, Enum):
    CORE = "core"
    WORKING = "working"
    ARCHIVAL = "archival"
    EPISODIC = "episodic"


class AtomType(str, Enum):
    OBSERVATION = "observation"
    DECISION = "decision"
    PATTERN = "pattern"
    PERSON = "person"
    INSIGHT = "insight"
    ENTITY = "entity"
    ACTION = "action"
    CONVERSATION = "conversation"


@dataclass
class MemoryAtom:
    atom_id: str = ""
    atom_type: AtomType = AtomType.OBSERVATION
    tier: MemoryTier = MemoryTier.WORKING
    content: str = ""
    embedding: List[float] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    tags: List[str] = field(default_factory=list)
    domain: Optional[str] = None
    importance: float = 0.5
    access_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    accessed_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    expires_at: Optional[str] = None
    source: str = "shadow"

    def __post_init__(self):
        if not self.atom_id:
            self.atom_id = f"atom_{hashlib.sha256(self.content.encode()).hexdigest()[:12]}_{int(time.time())}"

    def to_dict(self) -> Dict:
        return {
            "atom_id": self.atom_id,
            "atom_type": self.atom_type.value,
            "tier": self.tier.value,
            "content": self.content,
            "metadata": copy.deepcopy(self.metadata),
            "tags": list(self.tags),
            "domain": self.domain,
            "importance": round(self.importance, 3),
            "access_count": self.access_count,
            "created_at": self.created_at,
            "accessed_at": self.accessed_at,
        }

    def touch(self):
        self.access_count += 1
        self.accessed_at = datetime.now(timezone.utc).isoformat()


class InMemoryVectorIndex:
    """In-memory vector index using cosine similarity.

    THREAD SAFETY WARNING: ``_lock`` is an ``asyncio.Lock`` and is
    NOT thread-safe. If DarkMemory is accessed from multiple threads,
    callers must ensure that all async methods on this class are
    invoked from the same event-loop thread, or wrap calls with the
    thread-safe ``_db_thread_lock`` provided by the owning DarkMemory
    instance.

    Lock ordering: When DarkMemory needs both _db_thread_lock and
    vector index access, always acquire _db_thread_lock FIRST, then
    await the asyncio.Lock. Never reverse this order.
    """

    def __init__(self, dimension: int = 64):
        self.dimension = dimension
        self._vectors: Dict[str, List[float]] = {}
        self._id_to_atom: Dict[str, str] = {}
        self._lock = asyncio.Lock()

    async def add(self, atom_id: str, vector: List[float]):
        if len(vector) != self.dimension:
            vector = self._pad_or_truncate(vector, self.dimension)
        async with self._lock:
            self._vectors[atom_id] = vector

    async def remove(self, atom_id: str):
        async with self._lock:
            self._vectors.pop(atom_id, None)

    async def search(self, query_vector: List[float], top_k: int = 10) -> List[Tuple[str, float]]:
        async with self._lock:
            if len(query_vector) != self.dimension:
                query_vector = self._pad_or_truncate(query_vector, self.dimension)

            results = []
            for atom_id, vec in self._vectors.items():
                sim = self._cosine_similarity(query_vector, vec)
                results.append((atom_id, sim))

            results.sort(key=lambda x: x[1], reverse=True)
            return results[:top_k]

    def _cosine_similarity(self, a: List[float], b: List[float]) -> float:
        dot = sum(x * y for x, y in zip(a, b))
        norm_a = math.sqrt(sum(x * x for x in a))
        norm_b = math.sqrt(sum(x * x for x in b))
        if norm_a == 0 or norm_b == 0:
            return 0.0
        return dot / (norm_a * norm_b)

    def _pad_or_truncate(self, vec: List[float], target_dim: int) -> List[float]:
        if len(vec) > target_dim:
            return vec[:target_dim]
        return vec + [0.0] * (target_dim - len(vec))

    def size(self) -> int:
        return len(self._vectors)


class TemporalKnowledgeGraph:
    def __init__(self):
        self._entities: Dict[str, Dict] = {}
        self._relations: List[Dict] = []
        self._entity_embeddings: Dict[str, List[float]] = {}
        self._lock = threading.Lock()

    def add_entity(self, entity_id: str, entity_type: str, properties: Dict[str, Any]) -> str:
        entity_id = _sanitize(entity_id, 200)
        entity_type = _sanitize(entity_type, 100)
        with self._lock:
            if len(self._entities) >= _MAX_ENTITIES:
                evict_id = next(iter(self._entities))
                logger.warning("Evicting entity %s (cap=%d)", evict_id, _MAX_ENTITIES)
                del self._entities[evict_id]
            self._entities[entity_id] = {
                "id": entity_id,
                "type": entity_type,
                "properties": properties,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "updated_at": datetime.now(timezone.utc).isoformat(),
            }
        return entity_id

    def add_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> str:
        source_id = _sanitize(source_id, 200)
        target_id = _sanitize(target_id, 200)
        relation_type = _sanitize(relation_type, 100)
        relation_id = f"rel_{len(self._relations)}_{hashlib.sha256(f'{source_id}_{relation_type}_{target_id}'.encode()).hexdigest()[:12]}"
        relation = {
            "id": relation_id,
            "source": source_id,
            "target": target_id,
            "type": relation_type,
            "properties": properties or {},
            "valid_from": valid_from or datetime.now(timezone.utc).isoformat(),
            "valid_to": valid_to,
            "created_at": datetime.now(timezone.utc).isoformat(),
        }
        with self._lock:
            if len(self._relations) >= _MAX_RELATIONS:
                self._relations = self._relations[len(self._relations) - _MAX_RELATIONS:]
                logger.warning("Evicting old relations (cap=%d)", _MAX_RELATIONS)
            self._relations.append(relation)
        return relation_id

    def get_entity(self, entity_id: str) -> Optional[Dict]:
        with self._lock:
            entity = self._entities.get(entity_id)
            return copy.deepcopy(entity) if entity else None

    def get_relations(
        self,
        entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        valid_at: Optional[str] = None,
    ) -> List[Dict]:
        with self._lock:
            results = []
            for rel in self._relations:
                if entity_id and rel["source"] != entity_id and rel["target"] != entity_id:
                    continue
                if relation_type and rel["type"] != relation_type:
                    continue
                if valid_at:
                    if rel["valid_from"] > valid_at:
                        continue
                    if rel.get("valid_to") and rel["valid_to"] < valid_at:
                        continue
                results.append(copy.deepcopy(rel))
            return results

    def get_neighbors(self, entity_id: str, depth: int = 1, _visited: Optional[Set[str]] = None, _max_results: int = 200) -> Dict[str, List[Dict]]:
        if depth > 3:
            depth = 3
        result: Dict[str, List[Dict]] = {"outgoing": [], "incoming": []}
        visited = _visited or {entity_id}
        for rel in self._relations:
            if len(result["outgoing"]) + len(result["incoming"]) >= _max_results:
                break
            if rel["source"] == entity_id:
                result["outgoing"].append(copy.deepcopy(rel))
            if rel["target"] == entity_id:
                result["incoming"].append(copy.deepcopy(rel))
        if depth > 1:
            for rel in result["outgoing"] + result["incoming"]:
                neighbor_id = rel["target"] if rel["source"] == entity_id else rel["source"]
                if neighbor_id not in visited:
                    visited.add(neighbor_id)
                    sub = self.get_neighbors(neighbor_id, depth - 1, visited, _max_results)
                    result["outgoing"].extend(sub["outgoing"])
                    result["incoming"].extend(sub["incoming"])
        return result

    def search_entities(self, query: str, entity_type: Optional[str] = None) -> List[Dict]:
        with self._lock:
            results = []
            query_lower = query.lower()
            query_pattern = re.compile(r'\b' + re.escape(query_lower) + r'\b', re.IGNORECASE)
            for entity in self._entities.values():
                if entity_type and entity.get("type") != entity_type:
                    continue
                props_str = json.dumps(entity.get("properties", {})).lower()
                if query_pattern.search(entity.get("id", "").lower()) or query_pattern.search(props_str):
                    results.append(copy.deepcopy(entity))
            return results

    def get_timeline(self, entity_id: str) -> List[Dict]:
        with self._lock:
            timeline = []
            entity = self._entities.get(entity_id)
            if entity:
                timeline.append({"type": "created", "data": copy.deepcopy(entity), "timestamp": entity.get("created_at")})
            for rel in self._relations:
                if rel["source"] == entity_id or rel["target"] == entity_id:
                    timeline.append({"type": "relation", "data": copy.deepcopy(rel), "timestamp": rel.get("valid_from")})
            timeline.sort(key=lambda x: x.get("timestamp", ""))
            return timeline

    def get_stats(self) -> Dict:
        with self._lock:
            entities = dict(self._entities)
            relations = list(self._relations)
        return {
            "entities": len(entities),
            "relations": len(relations),
            "entity_types": dict(Counter(e["type"] for e in entities.values())),
            "relation_types": dict(Counter(r["type"] for r in relations)),
        }


class BlockStore:
    def __init__(self):
        self._blocks: Dict[str, Dict] = {}
        self._lock = threading.Lock()

    def search_blocks(self, query: str) -> List[Dict]:
        query_lower = query.lower().strip()
        if not query_lower:
            return []
        query_pattern = re.compile(r'\b' + re.escape(query_lower) + r'\b', re.IGNORECASE)
        with self._lock:
            results = []
            for label, block in self._blocks.items():
                value = block.get("value", "")
                if query_pattern.search(value.lower()):
                    results.append({
                        "label": label,
                        "value": value[:200],
                        "last_modified": block.get("last_modified"),
                    })
            return results

    def diff_blocks(self, label: str, old_value: str, new_value: str) -> Dict:
        old_lines = old_value.splitlines()
        new_lines = new_value.splitlines()
        added = []
        removed = []
        unchanged = []

        old_set = set(old_lines)
        new_set = set(new_lines)

        for line in old_lines:
            if line not in new_set:
                removed.append(line)
            else:
                unchanged.append(line)

        for line in new_lines:
            if line not in old_set:
                added.append(line)

        return {
            "label": label,
            "added": added,
            "removed": removed,
            "unchanged": unchanged,
            "summary": f"+{len(added)}/-{len(removed)}/~{len(unchanged)} lines",
        }

    def initialize_default_blocks(self):
        defaults = {
            "identity": {
                "label": "identity",
                "value": "I am Digital Shadow, a personal AI assistant.",
                "limit": 2000,
                "protected": True,
            },
            "persona": {
                "label": "persona",
                "value": "I observe, learn, and help proactively. I am calm and thoughtful.",
                "limit": 2000,
                "protected": False,
            },
            "human": {
                "label": "human",
                "value": "My user's preferences and habits will be learned over time.",
                "limit": 2000,
                "protected": True,
            },
            "relationships": {
                "label": "relationships",
                "value": "",
                "limit": 3000,
                "protected": False,
            },
            "knowledge": {
                "label": "knowledge",
                "value": "",
                "limit": 5000,
                "protected": False,
            },
            "context": {
                "label": "context",
                "value": "",
                "limit": 2000,
                "protected": False,
            },
            "career": {
                "label": "career",
                "value": "",
                "limit": 2000,
                "protected": False,
            },
            "health": {
                "label": "health",
                "value": "",
                "limit": 2000,
                "protected": False,
            },
            "finance": {
                "label": "finance",
                "value": "",
                "limit": 2000,
                "protected": False,
            },
        }
        for label, block_data in defaults.items():
            if label not in self._blocks:
                self._blocks[label] = {
                    **block_data,
                    "last_modified": datetime.now(timezone.utc).isoformat(),
                    "access_count": 0,
                }

    def get_block(self, label: str) -> Optional[Dict]:
        with self._lock:
            block = self._blocks.get(label)
            if block:
                block["access_count"] = block.get("access_count", 0) + 1
                return dict(block)
            return None

    def set_block(self, label: str, value: str) -> bool:
        with self._lock:
            block = self._blocks.get(label)
            if not block:
                return False
            if block.get("protected"):
                if value != block.get("value", ""):
                    logger.warning("Attempted to modify protected block: %s", label.replace(chr(10), ' ').replace(chr(13), ' '))
                    return False
            if len(value) > block.get("limit", 2000):
                value = value[:block["limit"]]
            block["value"] = value
            block["last_modified"] = datetime.now(timezone.utc).isoformat()
            return True

    def replace_in_block(self, label: str, old_text: str, new_text: str) -> bool:
        with self._lock:
            block = self._blocks.get(label)
            if not block:
                return False
            current = block.get("value", "")
            if old_text not in current:
                return False
            new_value = current.replace(old_text, new_text, 1)
            if block.get("protected"):
                if new_value != block.get("value", ""):
                    logger.warning("Attempted to modify protected block: %s", label.replace(chr(10), ' ').replace(chr(13), ' '))
                    return False
            if len(new_value) > block.get("limit", 2000):
                new_value = new_value[:block["limit"]]
            block["value"] = new_value
            block["last_modified"] = datetime.now(timezone.utc).isoformat()
            return True

    def insert_in_block(self, label: str, line_number: int, content: str) -> bool:
        with self._lock:
            block = self._blocks.get(label)
            if not block:
                return False
            lines = block.get("value", "").split("\n")
            if line_number < 0 or line_number > len(lines):
                return False
            lines.insert(line_number, content)
            new_value = "\n".join(lines)
            if block.get("protected"):
                logger.warning("Attempted to modify protected block: %s", label.replace(chr(10), ' ').replace(chr(13), ' '))
                return False
            if len(new_value) > block.get("limit", 2000):
                new_value = new_value[:block["limit"]]
            block["value"] = new_value
            block["last_modified"] = datetime.now(timezone.utc).isoformat()
            return True

    def list_blocks(self) -> List[Dict]:
        with self._lock:
            return [
                {"label": label, "limit": b.get("limit"), "protected": b.get("protected"),
                 "length": len(b.get("value", "")), "last_modified": b.get("last_modified"),
                 "access_count": b.get("access_count", 0)}
                for label, b in self._blocks.items()
            ]

    def get_core_context(self) -> str:
        with self._lock:
            core_blocks = ["identity", "persona", "human", "context"]
            parts = []
            for label in core_blocks:
                block = self._blocks.get(label)
                if block and block.get("value"):
                    parts.append(f"<{label}>\n{block['value']}\n</{label}>")
            return "\n".join(parts)


class MemoryCompactor:
    def __init__(self, max_working: int = 200, max_archival: int = 1000):
        self.max_working = max_working
        self.max_archival = max_archival
        self._compaction_count = 0
        self._summary_atoms: Dict[str, MemoryAtom] = {}

    def should_compact(self, working_count: int, archival_count: int) -> bool:
        return working_count > self.max_working or archival_count > self.max_archival

    def compact(self, atoms: List[MemoryAtom]) -> Tuple[List[MemoryAtom], List[MemoryAtom]]:
        self._compaction_count += 1
        working = [a for a in atoms if a.tier == MemoryTier.WORKING]
        archival = [a for a in atoms if a.tier == MemoryTier.ARCHIVAL]

        to_delete = []

        if len(working) > self.max_working:
            working.sort(key=lambda a: (a.importance, a.access_count), reverse=True)
            overflow = working[self.max_working:]
            for atom in overflow:
                if atom.importance < 0.3:
                    to_delete.append(atom)
                else:
                    atom.tier = MemoryTier.ARCHIVAL
                    archival.append(atom)
            working = working[:self.max_working]

        if len(archival) > self.max_archival:
            archival.sort(key=lambda a: (a.importance, a.access_count))
            evicted = archival[:len(archival) - self.max_archival]
            to_delete.extend(evicted)
            self._produce_summaries(evicted)
            archival = archival[len(archival) - self.max_archival:]

        kept = working + archival
        return kept, to_delete

    def _produce_summaries(self, evicted_atoms: List[MemoryAtom]):
        by_domain = {}
        for atom in evicted_atoms:
            domain = atom.domain or "general"
            by_domain.setdefault(domain, []).append(atom)
        for domain, atoms in by_domain.items():
            if len(atoms) >= 3:
                summary = self.summarize_atoms(atoms)
                summary_id = hashlib.sha256(f"summary:{domain}:{self._compaction_count}".encode()).hexdigest()[:12]
                summary_atom = MemoryAtom(
                    atom_id=summary_id,
                    atom_type=AtomType.OBSERVATION,
                    content=f"[COMPACTED SUMMARY - {domain}] {summary}",
                    domain=domain,
                    importance=0.25,
                    tier=MemoryTier.ARCHIVAL,
                )
                self._summary_atoms[summary_id] = summary_atom

    def summarize_atoms(self, atoms: List[MemoryAtom]) -> str:
        if not atoms:
            return ""
        summary_parts = []
        by_type = defaultdict(list)
        for atom in atoms:
            by_type[atom.atom_type.value].append(atom.content)
        for atom_type, contents in by_type.items():
            summary_parts.append(f"{atom_type}: {len(contents)} items")
            for content in contents[:5]:
                summary_parts.append(f"  - {content[:100]}")
        return "\n".join(summary_parts)


class DarkMemory:

    def __init__(self, data_dir: Optional[str] = None, db_path: Optional[str] = None,
                 max_atoms: int = 5000, compaction_interval: float = 300.0,
                 default_ttl_hours: float = 720.0):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data" / "dark_memory"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.db_path = db_path or str(self.data_dir / "shadow_memory.db")
        self._block_store = BlockStore()
        self._block_store.initialize_default_blocks()
        self._vector_index = InMemoryVectorIndex(dimension=64)
        self._temporal_kg = TemporalKnowledgeGraph()
        self._compactor = MemoryCompactor()
        self._max_atoms = max_atoms
        self._compaction_interval = compaction_interval
        self._compaction_task: Optional[asyncio.Task] = None
        self._default_ttl_hours = default_ttl_hours
        self._ttl_eviction_interval = 3600.0
        self._ttl_eviction_task: Optional[asyncio.Task] = None

        self._atoms: Dict[str, MemoryAtom] = {}
        self._atoms_lock = threading.Lock()
        self._db_conn: Optional[sqlite3.Connection] = None

        self._stats = {
            "atoms_stored": 0,
            "atoms_retrieved": 0,
            "searches": 0,
            "compactions": 0,
            "kg_queries": 0,
            "atoms_evicted": 0,
            "auto_compactions": 0,
        }
        self._stats_lock = threading.Lock()

        self._init_db()

    def _init_db(self):
        self._db_conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._db_conn.execute("PRAGMA journal_mode=WAL")
        self._db_thread_lock = threading.Lock()
        self._db_lock = asyncio.Lock()
        self._db_conn.execute("""
            CREATE TABLE IF NOT EXISTS observations (
                id TEXT PRIMARY KEY,
                atom_type TEXT NOT NULL,
                content TEXT NOT NULL,
                metadata TEXT DEFAULT '{}',
                tags TEXT DEFAULT '[]',
                domain TEXT,
                importance REAL DEFAULT 0.5,
                access_count INTEGER DEFAULT 0,
                created_at TEXT NOT NULL,
                accessed_at TEXT NOT NULL
            )
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_type ON observations(atom_type)
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_domain ON observations(domain)
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_importance ON observations(importance DESC)
        """)
        self._db_conn.execute("""
            CREATE INDEX IF NOT EXISTS idx_observations_created ON observations(created_at DESC)
        """)
        self._db_conn.commit()

    def _generate_embedding(self, text: str) -> List[float]:
        embedding = [0.0] * 64
        words = re.findall(r'\w+', text.lower())
        for i, word in enumerate(words):
            h = int(hashlib.sha256(word.encode()).hexdigest()[:8], 16)
            for j in range(0, min(8, 64), 8):
                idx = (h + j + i) % 64
                embedding[idx] += 1.0 / (1 + i)
        norm = math.sqrt(sum(x * x for x in embedding))
        if norm > 0:
            embedding = [x / norm for x in embedding]
        return embedding

    async def store(
        self,
        content: str,
        atom_type: AtomType = AtomType.OBSERVATION,
        tier: MemoryTier = MemoryTier.WORKING,
        tags: Optional[List[str]] = None,
        domain: Optional[str] = None,
        importance: float = 0.5,
        metadata: Optional[Dict] = None,
    ) -> MemoryAtom:
        importance = max(0.0, min(1.0, importance))
        content = _sanitize_content(content, 100000) if content else content
        if tags:
            tags = [_sanitize(t, 100) for t in tags[:20]]
        if domain:
            domain = _sanitize(domain, 100)
        if metadata and len(metadata) > 100:
            metadata = dict(list(metadata.items())[:100])
        atom = MemoryAtom(
            atom_type=atom_type,
            tier=tier,
            content=content,
            tags=tags or [],
            domain=domain,
            importance=importance,
            metadata=metadata or {},
        )

        atom.embedding = self._generate_embedding(content)

        if importance < 0.3:
            atom.expires_at = (datetime.now(timezone.utc) + timedelta(hours=self._default_ttl_hours)).isoformat()

        with self._atoms_lock:
            self._atoms[atom.atom_id] = atom
        try:
            await self._vector_index.add(atom.atom_id, atom.embedding)
        except (ValueError, IndexError, RuntimeError) as e:
            logger.warning("Vector index add failed for %s: %s", atom.atom_id[:16], type(e).__name__)

        try:
            with self._db_thread_lock:
                async with self._db_lock:
                    atom_type_str = atom.atom_type.value if hasattr(atom.atom_type, 'value') else str(atom.atom_type)
                    self._db_conn.execute(
                        "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (atom.atom_id, atom_type_str, atom.content,
                         json.dumps(atom.metadata), json.dumps(atom.tags),
                         atom.domain, atom.importance, atom.access_count,
                         atom.created_at, atom.accessed_at)
                    )
                    self._db_conn.commit()
        except (OSError, ValueError, TypeError, sqlite3.Error) as e:
            with self._atoms_lock:
                self._atoms.pop(atom.atom_id, None)
            try:
                await self._vector_index.remove(atom.atom_id)
            except (ValueError, KeyError, IndexError):
                pass
            logger.error("Failed to store atom in DB, rolled back: %s", type(e).__name__)
            raise

        with self._stats_lock:
            self._stats["atoms_stored"] += 1
        return atom

    async def retrieve(self, atom_id: str) -> Optional[MemoryAtom]:
        with self._atoms_lock:
            atom = self._atoms.get(atom_id)
            if atom:
                atom.touch()
        if atom:
            with self._stats_lock:
                self._stats["atoms_retrieved"] += 1
            try:
                with self._db_thread_lock:
                    self._db_conn.execute(
                        "UPDATE observations SET access_count = ?, accessed_at = ? WHERE id = ?",
                        (atom.access_count, atom.accessed_at, atom.atom_id),
                    )
                    self._db_conn.commit()
            except (OSError, ValueError, TypeError, sqlite3.Error) as e:
                logger.error("Failed to persist access_count for %s: %s", atom_id[:16], type(e).__name__)
            return copy.deepcopy(atom)

        try:
            async with self._db_lock:
                cursor = self._db_conn.execute(
                    "SELECT * FROM observations WHERE id = ?", (atom_id,)
                )
                row = cursor.fetchone()
            if row:
                atom = MemoryAtom(
                    atom_id=row[0],
                    atom_type=AtomType(row[1]),
                    content=row[2],
                    metadata=json.loads(row[3]) if row[3] else {},
                    tags=json.loads(row[4]) if row[4] else [],
                    domain=row[5],
                    importance=row[6],
                    access_count=row[7] + 1,
                    created_at=row[8],
                    accessed_at=datetime.now(timezone.utc).isoformat(),
                )
                with self._atoms_lock:
                    self._atoms[atom.atom_id] = atom
                with self._stats_lock:
                    self._stats["atoms_retrieved"] += 1
                try:
                    with self._db_thread_lock:
                        self._db_conn.execute(
                            "UPDATE observations SET access_count = ?, accessed_at = ? WHERE id = ?",
                            (atom.access_count, atom.accessed_at, atom.atom_id),
                        )
                        self._db_conn.commit()
                except (OSError, ValueError, TypeError, sqlite3.Error) as e:
                    logger.error("Failed to persist access_count for %s: %s", atom.atom_id[:16], type(e).__name__)
                return copy.deepcopy(atom)
        except (OSError, ValueError, TypeError, KeyError, sqlite3.Error) as e:
            logger.error("Failed to retrieve atom from DB: %s", type(e).__name__)

        return None

    async def search_with_boost(
        self,
        query: str,
        atom_type: Optional[AtomType] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
        domain_boost: Optional[str] = None,
    ) -> List[MemoryAtom]:
        with self._stats_lock:
            self._stats["searches"] += 1

        query_embedding = self._generate_embedding(query)
        vector_results = await self._vector_index.search(query_embedding, top_k=limit * 3)

        results = []
        seen = set()
        with self._atoms_lock:
            atoms_snapshot = dict(self._atoms)
        for atom_id, score in vector_results:
            atom = atoms_snapshot.get(atom_id)
            if not atom:
                continue
            if atom_id in seen:
                continue
            if atom_type and atom.atom_type != atom_type:
                continue
            if domain and atom.domain != domain:
                continue
            if tags and not all(t in atom.tags for t in tags):
                continue
            if atom.importance < min_importance:
                continue
            seen.add(atom_id)
            boosted_score = score
            if domain_boost and atom.domain == domain_boost:
                boosted_score += 0.2
            results.append((atom, boosted_score))

        if len(results) < limit:
            keyword_results = self._keyword_search(query, atom_type, domain, limit - len(results), atoms_snapshot)
            for atom in keyword_results:
                if atom.atom_id not in seen:
                    seen.add(atom.atom_id)
                    base_score = 0.0
                    if domain_boost and atom.domain == domain_boost:
                        base_score += 0.2
                    results.append((atom, base_score))

        results.sort(key=lambda x: x[1], reverse=True)
        return [copy.deepcopy(atom) for atom, _ in results[:limit]]

    def get_domain_summary(self) -> Dict[str, int]:
        with self._atoms_lock:
            atoms_items = list(self._atoms.values())
        summary: Dict[str, int] = {}
        for atom in atoms_items:
            domain = atom.domain or "unclassified"
            summary[domain] = summary.get(domain, 0) + 1
        return summary

    async def search(
        self,
        query: str,
        atom_type: Optional[AtomType] = None,
        domain: Optional[str] = None,
        tags: Optional[List[str]] = None,
        limit: int = 10,
        min_importance: float = 0.0,
    ) -> List[MemoryAtom]:
        with self._stats_lock:
            self._stats["searches"] += 1

        query_embedding = self._generate_embedding(query)
        vector_results = await self._vector_index.search(query_embedding, top_k=limit * 3)

        results = []
        seen = set()
        with self._atoms_lock:
            atoms_snapshot = dict(self._atoms)
        for atom_id, score in vector_results:
            atom = atoms_snapshot.get(atom_id)
            if not atom:
                continue
            if atom_id in seen:
                continue
            if atom_type and atom.atom_type != atom_type:
                continue
            if domain and atom.domain != domain:
                continue
            if tags and not all(t in atom.tags for t in tags):
                continue
            if atom.importance < min_importance:
                continue
            seen.add(atom_id)
            results.append((atom, score))

        if len(results) < limit:
            keyword_results = self._keyword_search(query, atom_type, domain, limit - len(results), atoms_snapshot)
            for atom in keyword_results:
                if atom.atom_id not in seen:
                    seen.add(atom.atom_id)
                    results.append((atom, 0.0))

        results.sort(key=lambda x: x[1], reverse=True)
        return [copy.deepcopy(atom) for atom, _ in results[:limit]]

    def _keyword_search(
        self,
        query: str,
        atom_type: Optional[AtomType] = None,
        domain: Optional[str] = None,
        limit: int = 10,
        atoms: Optional[Dict[str, MemoryAtom]] = None,
    ) -> List[MemoryAtom]:
        results = []
        query_lower = query.lower()
        query_pattern = re.compile(r'\b' + re.escape(query_lower) + r'\b', re.IGNORECASE)
        source = atoms if atoms is not None else self._atoms
        for atom in source.values():
            if query_pattern.search(atom.content.lower()):
                if atom_type and atom.atom_type != atom_type:
                    continue
                if domain and atom.domain != domain:
                    continue
                results.append(atom)
                if len(results) >= limit:
                    break
        return results

    async def search_db(
        self,
        query: str,
        atom_type: Optional[AtomType] = None,
        domain: Optional[str] = None,
        limit: int = 20,
    ) -> List[Dict]:
        try:
            safe_query = query.replace("%", "\\%").replace("_", "\\_")
            sql = "SELECT * FROM observations WHERE content LIKE ? ESCAPE '\\'"
            params = [f"%{safe_query}%"]
            if atom_type:
                sql += " AND atom_type = ?"
                params.append(atom_type.value)
            if domain:
                sql += " AND domain = ?"
                params.append(domain)
            sql += " ORDER BY importance DESC, created_at DESC LIMIT ?"
            params.append(limit)

            with self._db_thread_lock:
                async with self._db_lock:
                    cursor = self._db_conn.execute(sql, params)
                    rows = cursor.fetchall()
            return [
                {"id": r[0], "type": r[1], "content": r[2][:200],
                 "domain": r[5], "importance": r[6], "created_at": r[8]}
                for r in rows
            ]
        except (OSError, ValueError, sqlite3.Error) as e:
            logger.error("DB search error: %s", type(e).__name__)
            return []

    async def store_entity(self, entity_id: str, entity_type: str, properties: Dict) -> str:
        return self._temporal_kg.add_entity(entity_id, entity_type, properties)

    async def store_relation(
        self,
        source_id: str,
        target_id: str,
        relation_type: str,
        properties: Optional[Dict] = None,
        valid_from: Optional[str] = None,
        valid_to: Optional[str] = None,
    ) -> str:
        return self._temporal_kg.add_relation(
            source_id, target_id, relation_type,
            properties, valid_from, valid_to
        )

    async def query_kg(
        self,
        entity_id: Optional[str] = None,
        relation_type: Optional[str] = None,
        valid_at: Optional[str] = None,
        depth: int = 1,
    ) -> Dict:
        with self._stats_lock:
            self._stats["kg_queries"] += 1
        if entity_id:
            neighbors = self._temporal_kg.get_neighbors(entity_id, depth)
            entity = self._temporal_kg.get_entity(entity_id)
            return {"entity": entity, "neighbors": neighbors}
        relations = self._temporal_kg.get_relations(entity_id, relation_type, valid_at)
        return {"relations": relations}

    def get_block(self, label: str) -> Optional[Dict]:
        return self._block_store.get_block(label)

    def set_block(self, label: str, value: str) -> bool:
        return self._block_store.set_block(label, value)

    def list_blocks(self) -> List[Dict]:
        return self._block_store.list_blocks()

    def get_core_context(self) -> str:
        return self._block_store.get_core_context()

    async def compact(self) -> int:
        with self._atoms_lock:
            atoms = list(self._atoms.values())
            pre_ids = set(self._atoms.keys())
        kept, removed = self._compactor.compact(atoms)
        removed_ids = [a.atom_id for a in removed]
        summary_atoms = {sid: a for sid, a in self._compactor._summary_atoms.items() if sid not in self._atoms}
        with self._atoms_lock:
            for aid in removed_ids:
                self._atoms.pop(aid, None)
            for a in kept:
                self._atoms[a.atom_id] = a
            for sid, sa in summary_atoms.items():
                self._atoms[sid] = sa
        for atom_id in removed_ids:
            await self._vector_index.remove(atom_id)
        for sa in summary_atoms.values():
            await self._vector_index.add(sa.atom_id, self._generate_embedding(sa.content))
            try:
                with self._db_thread_lock:
                    created = sa.created_at if isinstance(sa.created_at, str) else sa.created_at.isoformat()
                    accessed = sa.accessed_at if isinstance(sa.accessed_at, str) else sa.accessed_at.isoformat()
                    self._db_conn.execute(
                        """INSERT OR REPLACE INTO observations
                           (id, atom_type, content, metadata, tags, domain, importance, access_count, created_at, accessed_at)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (sa.atom_id, sa.atom_type.value, sa.content, json.dumps(sa.metadata, default=str),
                         json.dumps(sa.tags, default=str), sa.domain or "general", sa.importance,
                         sa.access_count, created, accessed)
                    )
                    self._db_conn.commit()
            except (OSError, ValueError, sqlite3.Error) as e:
                logger.error("Failed to save compact summary to DB: %s", type(e).__name__)
        self._compactor._summary_atoms.clear()
        try:
            with self._db_thread_lock:
                if removed_ids:
                    placeholders = ",".join("?" * len(removed_ids))
                    self._db_conn.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", removed_ids)
                    self._db_conn.commit()
        except (OSError, ValueError, sqlite3.Error) as e:
            logger.error("Failed to delete compacted atoms from DB: %s", type(e).__name__)
        with self._stats_lock:
            self._stats["compactions"] += 1
            self._stats["atoms_evicted"] += len(removed)
            self._stats["summaries_generated"] = self._stats.get("summaries_generated", 0) + len(summary_atoms)
        return len(removed)

    async def get_recent(self, limit: int = 20, atom_type: Optional[AtomType] = None) -> List[MemoryAtom]:
        with self._atoms_lock:
            atoms = list(self._atoms.values())
        if atom_type:
            atoms = [a for a in atoms if a.atom_type == atom_type]
        atoms.sort(key=lambda a: a.created_at, reverse=True)
        return [copy.deepcopy(a) for a in atoms[:limit]]

    async def get_by_domain(self, domain: str, limit: int = 20) -> List[MemoryAtom]:
        with self._atoms_lock:
            atoms = [a for a in self._atoms.values() if a.domain == domain]
        atoms.sort(key=lambda a: a.importance, reverse=True)
        return [copy.deepcopy(a) for a in atoms[:limit]]

    def get_stats(self) -> Dict:
        with self._atoms_lock:
            atoms_items = list(self._atoms.values())
        by_type = defaultdict(int)
        by_tier = defaultdict(int)
        by_domain = defaultdict(int)
        for atom in atoms_items:
            by_type[atom.atom_type.value] += 1
            by_tier[atom.tier.value] += 1
            if atom.domain:
                by_domain[atom.domain] += 1

        with self._stats_lock:
            stats_copy = dict(self._stats)

        return {
            "total_atoms": len(atoms_items),
            "by_type": dict(by_type),
            "by_tier": dict(by_tier),
            "by_domain": dict(by_domain),
            "vector_index_size": self._vector_index.size(),
            "kg_stats": self._temporal_kg.get_stats(),
            "block_count": len(self._block_store.list_blocks()),
            "db_path": self.db_path,
            "stats": stats_copy,
        }

    def close(self):
        if self._compaction_task and not self._compaction_task.done():
            self._compaction_task.cancel()
        if self._ttl_eviction_task and not self._ttl_eviction_task.done():
            self._ttl_eviction_task.cancel()
        conn = getattr(self, "_db_conn", None)
        if conn is not None:
            try:
                conn.close()
            except (OSError, RuntimeError):
                pass
            finally:
                self._db_conn = None

    def __del__(self):
        try:
            self.close()
        except (OSError, RuntimeError, AttributeError):
            pass

    async def start_auto_compaction(self):
        if self._compaction_task and not self._compaction_task.done():
            return
        self._compaction_task = asyncio.create_task(self._auto_compaction_loop())

    async def _auto_compaction_loop(self):
        while True:
            try:
                await asyncio.sleep(self._compaction_interval)
                if len(self._atoms) > self._max_atoms:
                    evicted = await self.compact()
                    with self._stats_lock:
                        self._stats["auto_compactions"] += 1
                        self._stats["atoms_evicted"] += evicted
                    logger.info("Auto-compaction: evicted %d atoms, %d remaining", evicted, len(self._atoms))
                stale = await self._evict_stale(ttl_hours=168)
                if stale > 0:
                    logger.info("TTL eviction: removed %d stale atoms", stale)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Auto-compaction error: %s", type(e).__name__)

    async def start_ttl_eviction(self):
        if self._ttl_eviction_task and not self._ttl_eviction_task.done():
            return
        self._ttl_eviction_task = asyncio.create_task(self._ttl_eviction_loop())

    async def _ttl_eviction_loop(self):
        while True:
            try:
                await asyncio.sleep(self._ttl_eviction_interval)
                evicted = await self._evict_expired()
                if evicted > 0:
                    logger.info("TTL eviction loop: removed %d expired atoms", evicted)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("TTL eviction loop error: %s", type(e).__name__)

    async def _evict_expired(self) -> int:
        now = datetime.now(timezone.utc)
        to_remove = []
        with self._atoms_lock:
            atom_items = list(self._atoms.items())
        for atom_id, atom in atom_items:
            if atom.expires_at is not None:
                try:
                    expires = datetime.fromisoformat(atom.expires_at)
                    if expires.tzinfo is None:
                        expires = expires.replace(tzinfo=timezone.utc)
                    if now >= expires:
                        to_remove.append(atom_id)
                except (ValueError, TypeError):
                    pass
        for atom_id in to_remove:
            with self._atoms_lock:
                self._atoms.pop(atom_id, None)
            await self._vector_index.remove(atom_id)
        if to_remove:
            try:
                with self._db_thread_lock:
                    placeholders = ",".join("?" * len(to_remove))
                    self._db_conn.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", to_remove)
                    self._db_conn.commit()
            except (OSError, sqlite3.Error, ValueError) as e:
                logger.error("Failed to evict expired atoms from DB: %s", type(e).__name__)
        with self._stats_lock:
            self._stats["atoms_evicted"] += len(to_remove)
        return len(to_remove)

    async def _evict_stale(self, ttl_hours: int = 168) -> int:
        cutoff = (datetime.now(timezone.utc) - timedelta(hours=ttl_hours)).isoformat()
        to_remove = []
        with self._atoms_lock:
            for atom_id, atom in list(self._atoms.items()):
                if atom.tier == MemoryTier.ARCHIVAL and atom.accessed_at < cutoff and atom.importance < 0.3:
                    to_remove.append(atom_id)
        for atom_id in to_remove:
            with self._atoms_lock:
                self._atoms.pop(atom_id, None)
            await self._vector_index.remove(atom_id)
        if to_remove:
            try:
                with self._db_thread_lock:
                    placeholders = ",".join("?" * len(to_remove))
                    self._db_conn.execute(f"DELETE FROM observations WHERE id IN ({placeholders})", to_remove)
                    self._db_conn.commit()
            except (OSError, sqlite3.Error, ValueError) as e:
                logger.error("Failed to evict stale atoms from DB: %s", type(e).__name__)
        return len(to_remove)

    def get_health(self) -> Dict[str, Any]:
        """Compute a health score for the memory system (0-100)."""
        score = 100.0
        issues = []

        with self._atoms_lock:
            atom_count = len(self._atoms)
            if atom_count == 0:
                return {"score": 100.0, "status": "empty", "issues": []}

            low_importance = sum(1 for a in self._atoms.values() if a.importance < 0.2)
            unaccessed = sum(1 for a in self._atoms.values() if a.access_count == 0 and a.tier == MemoryTier.WORKING)

        if atom_count > self._max_atoms * 0.9:
            score -= 20
            issues.append(f"Near capacity: {atom_count}/{self._max_atoms}")

        low_ratio = low_importance / atom_count
        if low_ratio > 0.5:
            score -= 15
            issues.append(f"High low-importance ratio: {low_ratio:.0%}")

        if unaccessed > atom_count * 0.3:
            score -= 10
            issues.append(f"Many unaccessed working atoms: {unaccessed}")

        vec_size = self._vector_index.size()
        if vec_size != atom_count:
            score -= 10
            issues.append(f"Vector index mismatch: {vec_size} vectors vs {atom_count} atoms")

        db_ok = self._db_conn is not None
        if not db_ok:
            score -= 30
            issues.append("Database connection unavailable")

        score = max(0.0, min(100.0, score))
        status = "healthy" if score >= 80 else "degraded" if score >= 50 else "critical"

        return {"score": score, "status": status, "issues": issues}

    async def export_data(self) -> str:
        with self._atoms_lock:
            atoms_list = [a.to_dict() for a in self._atoms.values()]
        data = {
            "atoms": atoms_list,
            "blocks": self._block_store.list_blocks(),
            "kg_entities": len(self._temporal_kg._entities),
            "kg_relations": len(self._temporal_kg._relations),
            "stats": self._stats,
            "exported_at": datetime.now(timezone.utc).isoformat(),
        }
        return json.dumps(data, indent=2, default=str)

    async def import_data(self, data_str: str, merge: bool = True) -> Dict:
        if len(data_str) > 50 * 1024 * 1024:
            return {"imported": 0, "skipped": 0, "error": "Data too large (max 50MB)"}
        data = json.loads(data_str)
        imported = 0
        skipped = 0
        max_import = 5000
        max_content_length = 100 * 1024
        for atom_dict in data.get("atoms", [])[:max_import]:
            atom_id = atom_dict.get("atom_id")
            if not atom_id or len(atom_id) > 200:
                skipped += 1
                continue
            content = atom_dict.get("content", "")
            if len(content) > max_content_length:
                skipped += 1
                continue
            atom_type_str = atom_dict.get("atom_type", "observation")
            try:
                atom_type = AtomType(atom_type_str)
            except ValueError:
                atom_type = AtomType.OBSERVATION
            tier_str = atom_dict.get("tier", "working")
            try:
                tier = MemoryTier(tier_str)
            except ValueError:
                tier = MemoryTier.WORKING
            atom = MemoryAtom(
                atom_type=atom_type,
                tier=tier,
                content=atom_dict.get("content", ""),
                tags=atom_dict.get("tags", []),
                domain=atom_dict.get("domain"),
                importance=max(0.0, min(1.0, atom_dict.get("importance", 0.5))),
                metadata=atom_dict.get("metadata", {}),
            )
            atom.atom_id = atom_id
            atom.embedding = self._generate_embedding(atom.content)
            with self._atoms_lock:
                self._atoms[atom.atom_id] = atom
            try:
                await self._vector_index.add(atom.atom_id, atom.embedding)
            except (ValueError, KeyError):
                logger.debug("Vector index add failed for %s", atom.atom_id[:16])
            try:
                with self._db_thread_lock:
                    atom_type_str = atom.atom_type.value if hasattr(atom.atom_type, 'value') else str(atom.atom_type)
                    self._db_conn.execute(
                        "INSERT OR REPLACE INTO observations VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        (atom.atom_id, atom_type_str, atom.content,
                         json.dumps(atom.metadata), json.dumps(atom.tags),
                         atom.domain, atom.importance, atom.access_count,
                         atom.created_at, atom.accessed_at)
                    )
                    self._db_conn.commit()
            except (OSError, ValueError) as e:
                logger.debug("DB persist failed for %s: %s", atom.atom_id[:16], type(e).__name__)
            imported += 1
        return {"imported": imported, "skipped": skipped}

    async def find_duplicates(self, similarity_threshold: float = 0.95) -> List[List[str]]:
        """Find groups of near-duplicate atoms based on content similarity.

        Note: The embedding generation uses a deterministic SHA-256 hash
        over individual words. This means find_duplicates() effectively
        performs exact-content deduplication — semantically different
        texts with different words will NOT be matched as duplicates.
        No external embedding model is used.
        """
        with self._atoms_lock:
            atom_list = list(self._atoms.values())
        if len(atom_list) < 2:
            return []

        groups: List[List[str]] = []
        seen: Set[str] = set()

        for i, atom_a in enumerate(atom_list):
            if atom_a.atom_id in seen:
                continue
            group = [atom_a.atom_id]
            for j in range(i + 1, len(atom_list)):
                atom_b = atom_list[j]
                if atom_b.atom_id in seen:
                    continue
                if atom_a.atom_type != atom_b.atom_type:
                    continue
                if not atom_a.embedding or not atom_b.embedding:
                    continue
                sim = self._vector_index._cosine_similarity(atom_a.embedding, atom_b.embedding)
                if sim >= similarity_threshold:
                    group.append(atom_b.atom_id)
                    seen.add(atom_b.atom_id)
            if len(group) > 1:
                groups.append(group)
                seen.add(atom_a.atom_id)

        return groups

    async def deduplicate(self, similarity_threshold: float = 0.95, strategy: str = "keep_important") -> int:
        """Merge near-duplicate atoms. Returns number of atoms removed."""
        groups = await self.find_duplicates(similarity_threshold)
        removed = 0

        for group_ids in groups:
            with self._atoms_lock:
                atoms = [self._atoms[aid] for aid in group_ids if aid in self._atoms]
            if len(atoms) < 2:
                continue

            if strategy == "keep_important":
                atoms.sort(key=lambda a: (a.importance, a.access_count), reverse=True)
            elif strategy == "keep_recent":
                atoms.sort(key=lambda a: a.created_at, reverse=True)

            keeper = atoms[0]
            for duplicate in atoms[1:]:
                if duplicate.importance > keeper.importance:
                    keeper.importance = duplicate.importance
                if duplicate.access_count > keeper.access_count:
                    keeper.access_count = duplicate.access_count
                keeper.tags = list(set(keeper.tags + duplicate.tags))
                for k, v in duplicate.metadata.items():
                    if k not in keeper.metadata:
                        keeper.metadata[k] = v

                with self._atoms_lock:
                    self._atoms.pop(duplicate.atom_id, None)
                await self._vector_index.remove(duplicate.atom_id)
                try:
                    with self._db_thread_lock:
                        self._db_conn.execute("DELETE FROM observations WHERE id = ?", (duplicate.atom_id,))
                        self._db_conn.commit()
                except (OSError, ValueError, sqlite3.Error) as e:
                    logger.warning("Failed to delete duplicate %s from DB: %s", duplicate.atom_id[:16], type(e).__name__)
                removed += 1

        with self._stats_lock:
            self._stats["atoms_evicted"] += removed
        return removed