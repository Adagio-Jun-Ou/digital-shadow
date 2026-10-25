#!/usr/bin/env python3
"""
Advanced Memory Architecture - Based on Letta/MemGPT Research

Implements:
- Block-based memory organization (from Letta)
- Temporal knowledge graphs (from Graphiti/Zep)
- Core/Archival/Working memory hierarchy
- Episode-based provenance
"""

from datetime import datetime, timedelta, timezone

def _am_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default if default is not None else datetime.now(timezone.utc)
from typing import Dict, List, Optional, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
import json
import os
import copy
import hashlib
import re
import uuid
import threading
from collections import defaultdict

logger = __import__('logging').getLogger(__name__)

_SAFE_LABEL_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_label(label: str) -> str:
    if not _SAFE_LABEL_RE.match(label):
        return re.sub(r'[^a-zA-Z0-9_\-]', '_', label)[:64]
    return label


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()


def _sanitize_strict(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()


@dataclass
class MemoryBlock:
    """
    Labeled memory segment (from Letta pattern).
    
    Memory blocks are typed, labeled memory segments that can be
    retrieved and updated independently.
    """
    uuid: str
    label: str  # "user", "shadow_context", "learned_patterns"
    value: str
    embedding: Optional[List[float]] = None
    last_accessed: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def touch(self):
        """Mark block as accessed."""
        self.last_accessed = datetime.now(timezone.utc)
        self.access_count += 1
    
    def to_dict(self) -> Dict:
        return {
            "uuid": self.uuid,
            "label": self.label,
            "value": self.value,
            "embedding": list(self.embedding) if self.embedding else None,
            "last_accessed": self.last_accessed.isoformat(),
            "access_count": self.access_count,
            "created_at": self.created_at.isoformat(),
            "metadata": copy.deepcopy(self.metadata)
        }


@dataclass
class Episode:
    """
    Episode-based provenance (from Graphiti pattern).
    
    Every fact stored has provenance - it came from somewhere
    (a conversation, an observation, a decision).
    """
    uuid: str
    content: str
    source_type: str  # "conversation", "observation", "decision"
    source_id: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    entities: List[str] = field(default_factory=list)
    facts: List[str] = field(default_factory=list)
    valid_at: Optional[datetime] = None
    invalid_at: Optional[datetime] = None
    
    def __post_init__(self):
        self.source_type = _sanitize_strict(self.source_type)
        self.content = _sanitize_content(self.content)[:100000]
        self.source_id = _sanitize_strict(self.source_id)

    def to_dict(self) -> Dict:
        return {
            "uuid": self.uuid,
            "content": self.content,
            "source_type": self.source_type,
            "source_id": self.source_id,
            "timestamp": self.timestamp.isoformat(),
            "entities": list(self.entities),
            "facts": list(self.facts),
            "valid_at": self.valid_at.isoformat() if self.valid_at else None,
            "invalid_at": self.invalid_at.isoformat() if self.invalid_at else None
        }


@dataclass
class Entity:
    """
    Entity with temporal facts (from Graphiti pattern).
    
    Entities have facts that are valid for specific time ranges.
    Old facts are invalidated when new facts emerge.
    """
    _MAX_FACTS_PER_ENTITY = 1000
    name: str
    entity_type: str  # "person", "location", "concept", "event"
    facts: List[Dict] = field(default_factory=list)  # {"fact": str, "valid_at": datetime, "invalid_at": datetime, "source": str}
    summary: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def __post_init__(self):
        self.name = _sanitize_content(self.name)[:200]
        self.entity_type = _sanitize_strict(self.entity_type)
    
    def add_fact(self, fact: str, source: str, valid_at: datetime = None):
        """Add a new fact, invalidating old conflicting facts."""
        now = datetime.now(timezone.utc)
        for existing_fact in self.facts:
            if self._facts_conflict(existing_fact["fact"], fact):
                existing_fact["invalid_at"] = now
        
        # Add new fact
        self.facts.append({
            "fact": fact,
            "valid_at": valid_at or now,
            "invalid_at": None,
            "source": source
        })
        
        # Cap facts at max
        if len(self.facts) > self._MAX_FACTS_PER_ENTITY:
            self.facts = [f for f in self.facts if f.get("invalid_at") is None] + \
                         [f for f in self.facts if f.get("invalid_at") is not None][-100:]
        
        self._update_summary()
    
    def _facts_conflict(self, fact1: str, fact2: str) -> bool:
        """Check if two facts conflict."""
        conflict_attrs = [
            (r'\bname\b', 'name'), (r'\bage\b', 'age'),
            (r'\blocation\b', 'location'), (r'\bstatus\b', 'status'),
            (r'\bjob\b', 'job'), (r'\btitle\b', 'title'),
            (r'\baddress\b', 'address'), (r'\bphone\b', 'phone'),
        ]
        f1_lower = fact1.lower()
        f2_lower = fact2.lower()
        for pattern, attr in conflict_attrs:
            if re.search(pattern, f1_lower) and re.search(pattern, f2_lower):
                return True
        return False
    
    def _update_summary(self):
        """Update entity summary from current valid facts."""
        valid_facts = [f["fact"] for f in self.facts if f["invalid_at"] is None]
        self.summary = "; ".join(valid_facts[:5])  # Top 5 facts
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "facts": copy.deepcopy(self.facts),
            "summary": self.summary,
            "last_updated": self.last_updated.isoformat()
        }


class TemporalKnowledgeGraph:
    _MAX_EPISODES = 10000
    _MAX_ENTITIES = 5000
    _MAX_ENTITY_EPISODES = 100

    """
    Knowledge graph with temporal fact validity.
    
    From Graphiti/Zep research:
    - Facts have valid_at/invalid_at timestamps
    - Facts are sourced from episodes
    - Entities have summaries that evolve
    """
    
    def __init__(self):
        self.entities: Dict[str, Entity] = {}
        self.episodes: List[Episode] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)  # entity_name -> set of episode_uuids
        self._lock = threading.Lock()
    
    def add_episode(
        self,
        content: str,
        source_type: str,
        source_id: str,
        entities: List[str] = None,
        facts: List[str] = None
    ) -> Episode:
        """Add an episode and extract facts."""
        episode_uuid = uuid.uuid4().hex[:16]
        
        episode = Episode(
            uuid=episode_uuid,
            content=content,
            source_type=source_type,
            source_id=source_id,
            entities=entities or [],
            facts=facts or []
        )
        
        with self._lock:
            self.episodes.append(episode)
            if len(self.episodes) > self._MAX_EPISODES:
                self.episodes = self.episodes[-self._MAX_EPISODES:]
            
            for entity_name in (entities or []):
                self.entity_index[entity_name].add(episode_uuid)
                if len(self.entity_index[entity_name]) > self._MAX_ENTITY_EPISODES:
                    evict = list(self.entity_index[entity_name])[:len(self.entity_index[entity_name]) - self._MAX_ENTITY_EPISODES]
                    for e in evict:
                        self.entity_index[entity_name].discard(e)
        
        return episode
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity with current valid facts."""
        with self._lock:
            entity = self.entities.get(name)
            return copy.deepcopy(entity) if entity else None
    
    def get_entity_episodes(self, name: str, limit: int = 10) -> List[Episode]:
        """Get episodes mentioning this entity."""
        with self._lock:
            episode_uuids = self.entity_index.get(name, set())
            episodes = [e for e in self.episodes if e.uuid in episode_uuids]
            return copy.deepcopy(episodes[-limit:])
    
    def get_valid_facts(self, entity_name: str, as_of: datetime = None) -> List[str]:
        """Get facts valid as of a specific time."""
        with self._lock:
            entity = self.entities.get(entity_name)
            if not entity:
                return []
            
            now = as_of or datetime.now(timezone.utc)
            
            valid_facts = []
            for fact in entity.facts:
                if fact["valid_at"] and fact["valid_at"] > now:
                    continue
                if fact["invalid_at"] and fact["invalid_at"] <= now:
                    continue
                valid_facts.append(fact["fact"])
            
            return list(valid_facts)
    
    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "entities": {name: entity.to_dict() for name, entity in self.entities.items()},
                "episodes": [e.to_dict() for e in self.episodes],
                "entity_index": {k: list(v) for k, v in self.entity_index.items()}
            }


class AdvancedMemorySystem:
    """
    Three-tier memory architecture (from Letta pattern).
    
    Memory Hierarchy:
    1. Core Memory - Always available, limited size
    2. Archival Memory - Semantic search, unlimited
    3. Working Memory - Ephemeral, system-managed
    
    Plus:
    - Temporal knowledge graph (from Graphiti)
    - Episode provenance (from Zep)
    """
    
    _MAX_ARCHIVAL = 50000
    _MAX_CORE_BLOCK_CHARS = 100000
    _MAX_APPEND_BUFFER = 1000
    _MAX_WORKING_MEMORY = 100

    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "advanced_memory")
        self.data_dir.mkdir(parents=True, exist_ok=True)

        # Three memory tiers (Letta pattern)
        self.core_memory: Dict[str, MemoryBlock] = {}
        self.archival_memory: List[MemoryBlock] = []
        self.working_memory: List[Dict] = []

        # Temporal knowledge graph (Graphiti pattern)
        self.knowledge_graph = TemporalKnowledgeGraph()

        # Thread safety
        self._lock = threading.Lock()

        # Append buffer for efficient appends (V-289: avoids O(n^2) concat)
        self._append_buffer: Dict[str, List[str]] = defaultdict(list)
        
        # Core memory blocks (always available)
        self._init_core_blocks()
        
        # Load persisted memory
        self._load()
    
    def _init_core_blocks(self):
        """Initialize core memory blocks."""
        default_blocks = [
            ("user", "User information and preferences"),
            ("shadow_context", "Current Shadow AI context and state"),
            ("learned_patterns", "Patterns learned from observations"),
            ("active_commitments", "Current commitments and deadlines"),
            ("key_entities", "Important people, places, concepts")
        ]
        
        for label, description in default_blocks:
            if label not in self.core_memory:
                self.core_memory[label] = MemoryBlock(
                    uuid=hashlib.sha256(label.encode()).hexdigest()[:16],
                    label=label,
                    value="",
                    metadata={"description": description}
                )
    
    def _flush_append_buffer(self, label: str):
        """Flush append buffer for a label into the core memory block value."""
        if label in self._append_buffer and self._append_buffer[label]:
            parts = self._append_buffer[label]
            self._append_buffer[label] = []
            current = self.core_memory[label].value
            new_value = current + "\n" + "\n".join(parts)
            if len(new_value) > self._MAX_CORE_BLOCK_CHARS:
                new_value = new_value[-self._MAX_CORE_BLOCK_CHARS:]
            self.core_memory[label].value = new_value

    def _flush_all_buffers(self):
        """Flush all append buffers into core memory blocks."""
        for label in list(self._append_buffer.keys()):
            if self._append_buffer[label]:
                self._flush_append_buffer(label)

    def _get_core_value(self, label: str) -> str:
        """Get current core block value including buffered appends."""
        self._flush_append_buffer(label)
        return self.core_memory[label].value if label in self.core_memory else ""

    def add_to_core(self, label: str, content: str, append: bool = True):
        """Add content to core memory block."""
        label = _sanitize_label(label)
        content = _sanitize_content(content)[:self._MAX_CORE_BLOCK_CHARS]
        with self._lock:
            if label not in self.core_memory:
                self.core_memory[label] = MemoryBlock(
                    uuid=uuid.uuid4().hex[:16],
                    label=label,
                    value=content
                )
            else:
                if append:
                    self._append_buffer[label].append(content)
                    if len(self._append_buffer[label]) >= self._MAX_APPEND_BUFFER:
                        self._flush_append_buffer(label)
                else:
                    self._flush_append_buffer(label)
                    self.core_memory[label].value = content

            self.core_memory[label].touch()
            block = self.core_memory[label]
        self._persist_block(block)
    
    def add_to_archival(
        self,
        content: str,
        label: str = None,
        entities: List[str] = None,
        source_type: str = "observation",
        source_id: str = None
    ) -> MemoryBlock:
        """Add content to archival memory with episode."""
        with self._lock:
            block_uuid = hashlib.sha256(f"{content}:{datetime.now(timezone.utc)}".encode()).hexdigest()[:16]

            block = MemoryBlock(
                uuid=block_uuid,
                label=label or "archival",
                value=content
            )

            self.archival_memory.append(block)
            if len(self.archival_memory) > self._MAX_ARCHIVAL:
                self.archival_memory = self.archival_memory[-self._MAX_ARCHIVAL:]

            self.knowledge_graph.add_episode(
                content=content,
                source_type=source_type,
                source_id=source_id or block_uuid,
                entities=entities or []
            )

        self._persist_block(block)
        return block
    
    def search_archival(self, query: str, limit: int = 10) -> List[MemoryBlock]:
        """Search archival memory (simple keyword for now, would use embeddings)."""
        with self._lock:
            self._flush_all_buffers()
            results = []
            query_lower = query.lower()

            for block in reversed(self.archival_memory):
                if query_lower in block.value.lower():
                    block.touch()
                    results.append(block)
                    if len(results) >= limit:
                        break

            return [copy.deepcopy(b) for b in results]

    def get_working_context(self) -> str:
        """Get working memory context (system-managed window)."""
        with self._lock:
            self._flush_all_buffers()
            context_parts = []

            for label, block in self.core_memory.items():
                if block.value:
                    context_parts.append(f"## {label.upper()}\n{block.value}")

            if self.archival_memory:
                recent = self.archival_memory[-3:]
                context_parts.append("## RECENT MEMORIES\n" +
                                    "\n".join(f"- {m.value[:100]}" for m in recent))

            return "\n\n".join(context_parts)
    
    def add_entity_fact(
        self,
        entity_name: str,
        entity_type: str,
        fact: str,
        source: str
    ):
        """Add fact to entity in knowledge graph."""
        with self._lock:
            if entity_name not in self.knowledge_graph.entities:
                if len(self.knowledge_graph.entities) >= self.knowledge_graph._MAX_ENTITIES:
                    oldest = min(self.knowledge_graph.entities, key=lambda k: self.knowledge_graph.entities[k].last_updated)
                    del self.knowledge_graph.entities[oldest]
                self.knowledge_graph.entities[entity_name] = Entity(
                    name=entity_name,
                    entity_type=entity_type
                )

            self.knowledge_graph.entities[entity_name].add_fact(fact, source)
        self._save()
    
    def get_entity_summary(self, entity_name: str) -> str:
        """Get current summary of entity."""
        with self._lock:
            entity = self.knowledge_graph.entities.get(entity_name)
            return entity.summary if entity else ""
    
    def _persist_block(self, block: MemoryBlock):
        """Persist single block to disk atomically."""
        safe_label = _sanitize_label(block.label)
        block_file = self.data_dir / f"block_{safe_label}_{block.uuid[:8]}.json"
        tmp = block_file.with_suffix('.json.tmp')
        try:
            data = json.dumps(block.to_dict(), indent=2)
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                os.write(fd, data.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp), str(block_file))
        except (OSError, TypeError, ValueError) as e:
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            logger.error(f"Failed to persist block: {type(e).__name__}")
    
    def _save(self):
        """Save entire memory state atomically."""
        with self._lock:
            self._flush_all_buffers()
            try:
                kg_file = self.data_dir / "knowledge_graph.json"
                kg_tmp = kg_file.with_suffix('.json.tmp')
                data = json.dumps(self.knowledge_graph.to_dict(), indent=2)
                fd = os.open(str(kg_tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                try:
                    os.write(fd, data.encode())
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(str(kg_tmp), str(kg_file))

                core_file = self.data_dir / "core_memory.json"
                core_tmp = core_file.with_suffix('.json.tmp')
                data = json.dumps(
                    {label: block.to_dict() for label, block in self.core_memory.items()},
                    indent=2
                )
                fd = os.open(str(core_tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                try:
                    os.write(fd, data.encode())
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(str(core_tmp), str(core_file))
            except (OSError, TypeError, ValueError) as e:
                logger.error(f"Failed to save memory: {type(e).__name__}")
    
    def _load(self):
        """Load persisted memory."""
        try:
            # Load core memory
            core_file = self.data_dir / "core_memory.json"
            if core_file.exists():
                data = json.loads(core_file.read_text())
                for label, block_data in data.items():
                    self.core_memory[label] = MemoryBlock(
                        uuid=block_data["uuid"],
                        label=label,
                        value=block_data["value"],
                        embedding=block_data.get("embedding"),
                        last_accessed=_am_ts(block_data, "last_accessed"),
                        access_count=block_data["access_count"],
                        created_at=_am_ts(block_data, "created_at"),
                        metadata=block_data.get("metadata", {})
                    )
            
            # Load knowledge graph
            kg_file = self.data_dir / "knowledge_graph.json"
            if kg_file.exists():
                data = json.loads(kg_file.read_text())
                for name, entity_data in data.get("entities", {}).items():
                    entity = Entity(
                        name=name,
                        entity_type=entity_data["entity_type"],
                        facts=entity_data["facts"],
                        summary=entity_data["summary"],
                        last_updated=_am_ts(entity_data, "last_updated")
                    )
                    self.knowledge_graph.entities[name] = entity
                
                for episode_data in data.get("episodes", []):
                    episode = Episode(
                        uuid=episode_data["uuid"],
                        content=episode_data["content"],
                        source_type=episode_data["source_type"],
                        source_id=episode_data["source_id"],
                        timestamp=_am_ts(episode_data, "timestamp"),
                        entities=episode_data["entities"],
                        facts=episode_data["facts"]
                    )
                    self.knowledge_graph.episodes.append(episode)
        
        except (json.JSONDecodeError, OSError, KeyError, TypeError, ValueError) as e:
            logger.error(f"Failed to load memory: {type(e).__name__}")
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            return {
                "core_blocks": len(self.core_memory),
                "archival_count": len(self.archival_memory),
                "working_count": len(self.working_memory),
                "entities": len(self.knowledge_graph.entities),
                "episodes": len(self.knowledge_graph.episodes),
                "total_facts": sum(len(e.facts) for e in self.knowledge_graph.entities.values())
            }