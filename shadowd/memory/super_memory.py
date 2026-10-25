#!/usr/bin/env python3
"""
Shadow AI Super Memory - Letta-style with Knowledge Graph

Combines:
- Letta memory blocks (Core/Working/Archival)
- Knowledge graph for entity-relationships
- Bayesian priors for predictions
- Markov transitions for intent prediction
- Q-learning for decision learning
"""

import json
import os
import hashlib
import logging
import re
import tempfile
import copy
import uuid
from pathlib import Path
from datetime import datetime, timedelta, timezone

def _parse_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default if default is not None else datetime.now(timezone.utc)
from typing import Dict, List, Optional, Any, Set, Tuple
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import threading
import time

from ..atomic_io import fsync_dir

logger = logging.getLogger("shadowd.super_memory")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_RE.sub('', str(text)).strip()


def _sanitize_strict(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_STRICT_RE.sub('', str(text)).strip()


def _sanitize_content(text: str) -> str:
    if text is None:
        return ""
    return _SANITIZE_CONTENT_RE.sub('', str(text)).strip()


# ═════════════════════════════════════════════════════════════════
# MEMORY BLOCKS (Letta-style)
# ═════════════════════════════════════════════════════════════════

@dataclass
class MemoryBlock:
    """
    A block of memory (Letta pattern).
    
    Memory blocks are labeled segments that can be:
    - Core: Always in context (permanent beliefs)
    - Archival: Retrieved via search (long-term)
    - Working: Temporary context (recent)
    """
    label: str
    value: str
    limit: int = 2000
    protected: bool = False
    metadata: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    importance: float = 0.5  # 0-1 importance score
    
    def append(self, content: str) -> bool:
        """Append content to block if under limit."""
        if len(self.value) + len(content) + 1 > self.limit:
            return False
        self.value = self.value + "\n" + content if self.value else content
        self.last_modified = datetime.now(timezone.utc)
        return True
    
    def replace(self, new_value: str) -> bool:
        """Replace entire block content."""
        if len(new_value) > self.limit:
            return False
        self.value = new_value
        self.last_modified = datetime.now(timezone.utc)
        return True
    
    def trim(self) -> bool:
        """Trim content to fit limit, keeping most recent."""
        if len(self.value) <= self.limit:
            return False
        # Keep last portion
        self.value = "..." + self.value[-(self.limit-3):]
        self.last_modified = datetime.now(timezone.utc)
        return True
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "value": self.value,
            "limit": self.limit,
            "protected": self.protected,
            "metadata": copy.deepcopy(self.metadata) if self.metadata else {},
            "created": self.created.isoformat() if self.created else "",
            "last_modified": self.last_modified.isoformat() if self.last_modified else "",
            "access_count": self.access_count,
            "importance": round(self.importance, 3)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryBlock':
        return cls(
            label=data["label"],
            value=data["value"],
            limit=data.get("limit", 2000),
            protected=data.get("protected", False),
            metadata=data.get("metadata", {}),
            created=_parse_ts(data, "created", datetime.now(timezone.utc)),
            last_modified=_parse_ts(data, "last_modified", datetime.now(timezone.utc)),
            access_count=data.get("access_count", 0),
            importance=data.get("importance", 0.5)
        )


class CoreMemory:
    """
    Core memory always in context (Letta pattern).
    
    Contains permanent facts about:
    - Persona: Agent's identity and capabilities
    - Human: User preferences and facts
    - Recent: Recent important events
    - Goals: Current objectives
    """
    
    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path
        self.blocks: Dict[str, MemoryBlock] = {}
        self._lock = threading.Lock()
        self._initialize_default_blocks()
    
    def _initialize_default_blocks(self):
        """Initialize default memory blocks."""
        defaults = {
            "persona": MemoryBlock(
                label="persona",
                value="I am Shadow AI, a personal assistant with superhuman abilities.\nI can see through walls (X-Ray), predict events (Oracle), remember everything (Perfect Memory), and learn from past decisions (Shadow Council).",
                limit=2000,
                protected=True,
                importance=1.0
            ),
            "human": MemoryBlock(
                label="human",
                value="",
                limit=2000,
                protected=False,
                importance=1.0
            ),
            "recent": MemoryBlock(
                label="recent",
                value="",
                limit=1000,
                protected=False,
                importance=0.8
            ),
            "goals": MemoryBlock(
                label="goals",
                value="",
                limit=1000,
                protected=False,
                importance=0.9
            ),
        }
        
        for label, block in defaults.items():
            if label not in self.blocks:
                self.blocks[label] = block
    
    def get_block(self, label: str) -> Optional[MemoryBlock]:
        """Get a memory block (returns a deep copy)."""
        with self._lock:
            block = self.blocks.get(label)
            if block:
                block.access_count += 1
                return copy.deepcopy(block)
        return None
    
    def set_block(self, label: str, value: str) -> bool:
        """Set block content."""
        label = _sanitize_strict(label)
        with self._lock:
            if label not in self.blocks:
                self.blocks[label] = MemoryBlock(label=label, value=value)
                return True
            else:
                if self.blocks[label].protected:
                    return False
                self.blocks[label].replace(value)
                return True
    
    def append_block(self, label: str, content: str) -> bool:
        """Append to block."""
        label = _sanitize_strict(label)
        with self._lock:
            if label not in self.blocks:
                self.blocks[label] = MemoryBlock(label=label, value=content)
                return True
            
            if not self.blocks[label].append(content):
                # Trim and try again
                self.blocks[label].trim()
                return self.blocks[label].append(content)
            return True
    
    def update_block_importance(self, label: str, delta: float = 0.1):
        """Update importance score based on usage."""
        if label in self.blocks:
            self.blocks[label].importance = min(1.0, max(0.0, self.blocks[label].importance + delta))
    
    def get_context(self, include_labels: List[str] = None) -> str:
        """Get core memory context for LLM."""
        with self._lock:
            if include_labels:
                blocks = [self.blocks[l] for l in include_labels if l in self.blocks]
            else:
                # Sort by importance
                blocks = sorted(self.blocks.values(), key=lambda b: -b.importance)
        
            lines = ["# Core Memory\n"]
            for block in blocks:
                if block.value:
                    lines.append(f"\n## {block.label.upper()}")
                    lines.append(block.value)
            
            return "\n".join(lines)
    
    def to_dict(self) -> Dict:
        with self._lock:
            return {label: block.to_dict() for label, block in self.blocks.items()}
    
    @classmethod
    def from_dict(cls, data: Dict, storage_path: Path = None) -> 'CoreMemory':
        memory = cls(storage_path)
        memory.blocks = {label: MemoryBlock.from_dict(block_data) for label, block_data in data.items()}
        return memory
    
    def save(self):
        """Save to disk atomically."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.storage_path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(self.to_dict(), f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                fsync_dir(self.storage_path.parent)
                os.replace(tmp_path, str(self.storage_path))
            except (json.JSONDecodeError, OSError, ValueError):
                logger.error("Core memory save error")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    
    def load(self):
        """Load from disk."""
        if self.storage_path and self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                with self._lock:
                    self.blocks = {label: MemoryBlock.from_dict(block_data) for label, block_data in data.items()}
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as e:
                logger.error(f"Load error: {type(e).__name__}")


class WorkingMemory:
    """
    Working memory for temporary context.
    
    Manages:
    - Conversation history
    - Recent observations
    - Current task context
    - Attention window
    """
    _MAX_ATTENTION_FOCUS = 10
    _MAX_TASK_KEYS = 50

    def __init__(self, max_messages: int = 20, max_tokens: int = 8000):
        self.max_messages = max_messages
        self.max_tokens = max_tokens
        self.messages: List[Dict[str, Any]] = []
        self.current_task: Optional[Dict] = None
        self.attention_focus: List[str] = []
        self._lock = threading.Lock()
    
    def add_message(self, role: str, content: str, metadata: Dict = None):
        """Add a message to working memory."""
        message = {
            "role": _sanitize(role),
            "content": _sanitize_content(content),
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metadata": copy.deepcopy(metadata) if metadata else {},
            "tokens": self._estimate_tokens(content)
        }
        with self._lock:
            self.messages.append(message)
            self._trim_if_needed()
    
    def _estimate_tokens(self, text: str) -> int:
        """Estimate token count."""
        return len(text) // 4
    
    def _trim_if_needed(self):
        """Trim messages to fit limits. Must be called under self._lock."""
        while len(self.messages) > self.max_messages:
            self.messages.pop(0)
        total_tokens = sum(m["tokens"] for m in self.messages)
        while total_tokens > self.max_tokens and len(self.messages) > 1:
            removed = self.messages.pop(0)
            total_tokens -= removed["tokens"]
    
    def set_task(self, task: Dict):
        """Set current task."""
        if task and isinstance(task, dict):
            sanitized = {}
            for k, v in list(task.items())[:self._MAX_TASK_KEYS]:
                sanitized[_sanitize_strict(str(k))] = _sanitize_content(str(v)) if isinstance(v, str) else v
            task = sanitized
        with self._lock:
            self.current_task = task
    
    def add_attention(self, focus: str):
        """Add something to attention focus."""
        focus = _sanitize_content(focus)[:500]
        with self._lock:
            self.attention_focus.append(focus)
            if len(self.attention_focus) > self._MAX_ATTENTION_FOCUS:
                self.attention_focus.pop(0)
    
    def get_context(self) -> str:
        """Get working memory context."""
        with self._lock:
            msgs = self.messages[-10:]
            task = copy.deepcopy(self.current_task) if self.current_task else None
            focus = list(self.attention_focus)
        
        lines = ["# Working Memory\n"]
        if task:
            lines.append("\n## Current Task")
            lines.append(json.dumps(task, indent=2))
        if focus:
            lines.append("\n## Attention Focus")
            for f in focus:
                lines.append(f"- {f}")
        if msgs:
            lines.append(f"\n## Recent Messages ({len(self.messages)})")
            for msg in msgs:
                lines.append(f"[{msg['role']}] {msg['content'][:200]}")
        return "\n".join(lines)
    
    def clear(self):
        """Clear working memory."""
        with self._lock:
            self.messages = []
            self.current_task = None
            self.attention_focus = []
    
    def to_dict(self) -> Dict:
        with self._lock:
            return {
                "messages": copy.deepcopy(self.messages),
                "current_task": copy.deepcopy(self.current_task),
                "attention_focus": list(self.attention_focus),
                "max_messages": self.max_messages,
                "max_tokens": self.max_tokens
            }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkingMemory':
        memory = cls(
            max_messages=data.get("max_messages", 20),
            max_tokens=data.get("max_tokens", 8000)
        )
        memory.messages = data.get("messages", [])
        memory.current_task = data.get("current_task")
        memory.attention_focus = data.get("attention_focus", [])
        return memory


class ArchivalMemory:
    """
    Archival memory for long-term storage.
    
    Features:
    - Unlimited capacity
    - Semantic search (keyword-based, upgradeable)
    - Time-based retrieval
    - Importance ranking
    """
    
    _MAX_MEMORIES = 50000

    def __init__(self, storage_path: Path = None, storage_dir: Path = None):
        if storage_path:
            self.storage_path = storage_path
        elif storage_dir:
            self.storage_path = storage_dir / "archival"
        else:
            self.storage_path = Path.home() / "digital-shadow" / "data" / "memory" / "archival"
        self.storage_path.mkdir(parents=True, exist_ok=True)
        self.memories: List[Dict[str, Any]] = []
        self.index: Dict[str, Set[int]] = defaultdict(set)
        self._lock = threading.Lock()
        self._load()
    
    def insert(self, content: str, metadata: Dict = None, importance: float = 0.5) -> str:
        """Insert into archival memory."""
        content = _sanitize_content(content)[:100000]
        if metadata:
            metadata = {str(k): str(v)[:500] for k, v in list(metadata.items())[:50]}
        importance = max(0.0, min(1.0, importance))
        memory_id = f"arch_{uuid.uuid4().hex[:12]}"
        
        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "importance": importance,
            "access_count": 0
        }
        
        with self._lock:
            self.memories.append(memory)
            if len(self.memories) > self._MAX_MEMORIES:
                self.memories = self.memories[-self._MAX_MEMORIES:]
                self._rebuild_index()
            memory_index = len(self.memories) - 1
            
            words = self._tokenize(content)
            for word in words:
                self.index[word].add(memory_index)
            
            if metadata:
                for key, value in metadata.items():
                    self.index[f"{key}:{str(value).lower()}"].add(memory_index)
            
            self._save()
        return memory_id
    
    def _tokenize(self, text: str) -> Set[str]:
        """Tokenize text for indexing."""
        words = set()
        for word in re.findall(r'\w+', text.lower()):
            if len(word) > 2:
                words.add(word)
        return words
    
    def search(self, query: str, limit: int = 10, time_range: Tuple[datetime, datetime] = None) -> List[Dict]:
        """Search archival memory."""
        with self._lock:
            query_words = self._tokenize(query)
            
            matching_indices: Counter = Counter()
            for word in query_words:
                for idx in self.index.get(word, set()):
                    matching_indices[idx] += 1
            
            sorted_indices = sorted(
                matching_indices.keys(),
                key=lambda i: (matching_indices[i], self.memories[i].get("importance", 0.5)),
                reverse=True
            )
            
            results = []
            for idx in sorted_indices[:limit * 2]:
                memory = self.memories[idx]
                if time_range:
                    try:
                        ts = datetime.fromisoformat(memory["timestamp"])
                        if not (time_range[0] <= ts <= time_range[1]):
                            continue
                    except (ValueError, TypeError):
                        pass
                results.append(copy.deepcopy(memory))
                if len(results) >= limit:
                    break
            
            return results
    
    def get_recent(self, hours: int = 24, limit: int = 20) -> List[Dict]:
        """Get recent memories."""
        cutoff = datetime.now(timezone.utc) - timedelta(hours=hours)
        with self._lock:
            recent = []
            for memory in reversed(self.memories):
                try:
                    ts = datetime.fromisoformat(memory["timestamp"])
                    if ts >= cutoff:
                        recent.append(copy.deepcopy(memory))
                        if len(recent) >= limit:
                            break
                except (ValueError, TypeError):
                    pass
            return recent
    
    def get_important(self, threshold: float = 0.7, limit: int = 10) -> List[Dict]:
        """Get important memories."""
        with self._lock:
            important = [copy.deepcopy(m) for m in self.memories if m.get("importance", 0.5) >= threshold]
            important.sort(key=lambda m: (-m.get("importance", 0), -m.get("access_count", 0)))
            return important[:limit]
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        with self._lock:
            for i, memory in enumerate(self.memories):
                if memory["id"] == memory_id:
                    self.memories.pop(i)
                    self._rebuild_index()
                    self._save()
                    return True
            return False
    
    def _rebuild_index(self):
        """Rebuild the search index. Must be called under self._lock."""
        self.index = defaultdict(set)
        for i, memory in enumerate(self.memories):
            words = self._tokenize(memory["content"])
            for word in words:
                self.index[word].add(i)
            for key, value in memory.get("metadata", {}).items():
                self.index[f"{key}:{str(value).lower()}"].add(i)
    
    def _save(self):
        """Save to disk atomically. Must be called under self._lock."""
        try:
            data_file = self.storage_path / "archival_memories.json"
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.storage_path), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(self.memories, f, indent=2, default=str)
                    f.flush()
                    os.fsync(f.fileno())
                fsync_dir(self.storage_path)
                os.replace(tmp_path, str(data_file))
            except (json.JSONDecodeError, OSError, ValueError):
                logger.error("Archival save error")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
        except (OSError, ValueError):
            logger.error("Archival save error")
    
    def _load(self):
        """Load from disk."""
        try:
            data_file = self.storage_path / "archival_memories.json"
            if data_file.exists():
                data = json.loads(data_file.read_text())
                with self._lock:
                    self.memories = data
                    self._rebuild_index()
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
            logger.error("Archival load error")
            self.memories = []
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            return {
                "total_memories": len(self.memories),
                "index_size": len(self.index),
                "oldest": self.memories[0]["timestamp"] if self.memories else None,
                "newest": self.memories[-1]["timestamp"] if self.memories else None,
                "avg_importance": sum(m.get("importance", 0.5) for m in self.memories) / max(1, len(self.memories))
            }


# ═════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH
# ═════════════════════════════════════════════════════════════════

@dataclass
class Entity:
    """Entity in knowledge graph."""
    id: str
    entity_type: str
    name: str
    attributes: Dict[str, Any] = field(default_factory=dict)
    created: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    importance: float = 0.5

    def __post_init__(self):
        self.id = _sanitize_strict(self.id)
        self.entity_type = _sanitize_strict(self.entity_type)
        self.name = _sanitize_content(self.name).strip()[:200]
        self.importance = max(0.0, min(1.0, self.importance))
        if self.attributes:
            self.attributes = {_sanitize_strict(str(k)).strip()[:50]: _sanitize_content(str(v)).strip()[:500] for k, v in list(self.attributes.items())[:50]}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "type": self.entity_type,
            "name": self.name,
            "attributes": copy.deepcopy(self.attributes) if self.attributes else {},
            "created": self.created.isoformat(),
            "importance": round(self.importance, 3)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Entity':
        return cls(
            id=data["id"],
            entity_type=data["type"],
            name=data["name"],
            attributes=data.get("attributes", {}),
            created=_parse_ts(data, "created", datetime.now(timezone.utc)),
            importance=data.get("importance", 0.5)
        )


@dataclass
class Relation:
    """Relation between entities."""
    id: str
    subject_id: str
    predicate: str
    object_id: str
    context: Dict[str, Any] = field(default_factory=dict)
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    confidence: float = 1.0

    def __post_init__(self):
        self.subject_id = _sanitize_strict(self.subject_id)
        self.predicate = _sanitize_strict(self.predicate)
        self.object_id = _sanitize_strict(self.object_id)
        self.confidence = max(0.0, min(1.0, self.confidence))
        if self.context:
            self.context = {_sanitize_strict(str(k)).strip()[:50]: _sanitize_content(str(v)).strip()[:500] for k, v in list(self.context.items())[:50]}

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "subject": self.subject_id,
            "predicate": self.predicate,
            "object": self.object_id,
            "context": copy.deepcopy(self.context) if self.context else {},
            "timestamp": self.timestamp.isoformat(),
            "confidence": round(self.confidence, 3)
        }

    @classmethod
    def from_dict(cls, data: Dict) -> 'Relation':
        return cls(
            id=data["id"],
            subject_id=data["subject"],
            predicate=data["predicate"],
            object_id=data["object"],
            context=data.get("context", {}),
            timestamp=_parse_ts(data, "timestamp", datetime.now(timezone.utc)),
            confidence=data.get("confidence", 1.0)
        )


class KnowledgeGraph:
    """
    Knowledge graph for entity-relationship storage.
    
    Features:
    - Entity management
    - Relation tracking
    - Semantic queries
    - Temporal queries
    - Confidence scoring
    """
    
    _MAX_ENTITIES = 5000
    _MAX_RELATIONS = 10000

    def __init__(self, storage_path: Path = None):
        self.storage_path = storage_path
        self.entities: Dict[str, Entity] = {}
        self.relations: Dict[str, Relation] = {}
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[str, Set[str]] = defaultdict(set)
        self._lock = threading.Lock()
    
    def add_entity(self, entity_type: str, name: str, attributes: Dict = None, importance: float = 0.5) -> str:
        """Add entity and return ID."""
        entity_type = _sanitize_strict(entity_type)
        name = _sanitize_content(name)[:200]
        entity_id = uuid.uuid4().hex[:12]
        
        with self._lock:
            if len(self.entities) >= self._MAX_ENTITIES:
                oldest_id = next(iter(self.entities))
                old_type = self.entities[oldest_id].entity_type
                self.entity_index[old_type].discard(oldest_id)
                if not self.entity_index[old_type]:
                    del self.entity_index[old_type]
                logger.warning(f"KG entity eviction: removed {oldest_id}")
                del self.entities[oldest_id]
            
            entity = Entity(
                id=entity_id,
                entity_type=entity_type,
                name=name,
                attributes=attributes or {},
                importance=importance
            )
            
            self.entities[entity_id] = entity
            self.entity_index[entity_type].add(entity_id)
        
        return entity_id
    
    def add_relation(self, subject_id: str, predicate: str, object_id: str,
                     context: Dict = None, confidence: float = 1.0) -> str:
        """Add relation: subject --[predicate]--> object."""
        subject_id = _sanitize_strict(subject_id)
        predicate = _sanitize_strict(predicate)
        object_id = _sanitize_strict(object_id)
        confidence = max(0.0, min(1.0, confidence))
        
        with self._lock:
            if len(self.relations) >= self._MAX_RELATIONS:
                oldest_rid = next(iter(self.relations))
                removed = self.relations.pop(oldest_rid)
                old_pred = removed.predicate
                self.relation_index[old_pred].discard(removed.id)
                if not self.relation_index[old_pred]:
                    del self.relation_index[old_pred]
                logger.warning(f"KG relation eviction: removed {removed.id}")
            
            relation_id = uuid.uuid4().hex[:12]
            
            relation = Relation(
                id=relation_id,
                subject_id=subject_id,
                predicate=predicate,
                object_id=object_id,
                context=context or {},
                confidence=confidence
            )
            
            self.relations[relation_id] = relation
            self.relation_index[predicate].add(relation_id)
        
        return relation_id
    
    def get_entity(self, entity_id: str) -> Optional[Entity]:
        """Get entity by ID. Returns a deep copy."""
        with self._lock:
            entity = self.entities.get(entity_id)
            return copy.deepcopy(entity) if entity else None
    
    def find_entities(self, entity_type: str = None, name_contains: str = None) -> List[Entity]:
        """Find entities matching criteria."""
        with self._lock:
            candidates = set(self.entities.keys())
            if entity_type:
                candidates &= self.entity_index.get(entity_type, set())
            
            results = []
            for eid in candidates:
                entity = self.entities[eid]
                if name_contains:
                    if name_contains.lower() not in entity.name.lower():
                        continue
                results.append(copy.deepcopy(entity))
            
            results.sort(key=lambda e: -e.importance)
            return results
    
    def get_relations(self, subject_id: str = None, predicate: str = None,
                      object_id: str = None) -> List[Relation]:
        """Get relations matching criteria."""
        with self._lock:
            results = []
            candidates = self.relation_index.get(predicate, set(self.relations.keys())) if predicate else set(self.relations.keys())
            
            for rid in candidates:
                relation = self.relations.get(rid)
                if relation is None:
                    continue
                if subject_id and relation.subject_id != subject_id:
                    continue
                if object_id and relation.object_id != object_id:
                    continue
                results.append(copy.deepcopy(relation))
            
            return results
    
    def query(self, query_spec: Dict) -> List[Dict]:
        """
        Complex query on knowledge graph.
        
        query_spec: {
            "entity_type": "person",
            "name_contains": "Sarah",
            "relations": [{"predicate": "met"}],
            "time_range": {"start": "...", "end": "..."}
        }
        """
        entities = self.find_entities(
            entity_type=query_spec.get("entity_type"),
            name_contains=query_spec.get("name_contains")
        )
        
        results = []
        
        for entity in entities:
            result = {"entity": entity, "relations": {}}
            
            for rel_spec in query_spec.get("relations", []):
                predicate = rel_spec.get("predicate")
                relations = self.get_relations(subject_id=entity.id, predicate=predicate)
                
                for rel in relations:
                    obj = self.get_entity(rel.object_id)
                    if obj:
                        if predicate not in result["relations"]:
                            result["relations"][predicate] = []
                        result["relations"][predicate].append({
                            "entity": obj,
                            "context": rel.context,
                            "timestamp": rel.timestamp.isoformat()
                        })
            
            results.append(result)
        
        return results
    
    def get_timeline(self, entity_id: str = None, start: datetime = None,
                     end: datetime = None) -> List[Dict]:
        """Get timeline of events."""
        with self._lock:
            events = []
            for relation in self.relations.values():
                if entity_id and relation.subject_id != entity_id and relation.object_id != entity_id:
                    continue
                
                try:
                    ts = datetime.fromisoformat(relation.timestamp.isoformat()) if isinstance(relation.timestamp, datetime) else datetime.fromisoformat(str(relation.timestamp))
                    if start and ts < start:
                        continue
                    if end and ts > end:
                        continue
                except (ValueError, TypeError):
                    pass
                
                events.append({
                    "subject": copy.deepcopy(self.entities.get(relation.subject_id, {})),
                    "predicate": relation.predicate,
                    "object": copy.deepcopy(self.entities.get(relation.object_id, {})),
                    "timestamp": relation.timestamp.isoformat(),
                    "context": copy.deepcopy(relation.context) if relation.context else {}
                })
            
            events.sort(key=lambda e: e["timestamp"])
            return events
    
    def to_dict(self) -> Dict:
        with self._lock:
            return self._to_dict_unlocked()

    def _to_dict_unlocked(self) -> Dict:
        return {
            "entities": {eid: e.to_dict() for eid, e in self.entities.items()},
            "relations": {rid: r.to_dict() for rid, r in self.relations.items()}
        }
    
    @classmethod
    def from_dict(cls, data: Dict, storage_path: Path = None) -> 'KnowledgeGraph':
        kg = cls(storage_path)
        kg.entities = {eid: Entity.from_dict(e) for eid, e in data.get("entities", {}).items()}
        kg.relations = {r["id"]: Relation.from_dict(r) for r in data.get("relations", [])}
        
        for eid, e in kg.entities.items():
            kg.entity_index[e.entity_type].add(eid)
        for rid, r in kg.relations.items():
            kg.relation_index[r.predicate].add(r.id)
        
        return kg
    
    def save(self):
        """Save to disk atomically."""
        if self.storage_path:
            self.storage_path.parent.mkdir(parents=True, exist_ok=True)
            with self._lock:
                data = self._to_dict_unlocked()
            tmp_fd, tmp_path = tempfile.mkstemp(
                dir=str(self.storage_path.parent), suffix=".tmp"
            )
            try:
                with os.fdopen(tmp_fd, "w") as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
                fsync_dir(self.storage_path.parent)
                os.replace(tmp_path, str(self.storage_path))
            except (json.JSONDecodeError, OSError, ValueError):
                logger.error("KG save error")
                try:
                    os.unlink(tmp_path)
                except OSError:
                    pass
    
    def load(self):
        """Load from disk."""
        if self.storage_path and self.storage_path.exists():
            try:
                data = json.loads(self.storage_path.read_text())
                kg = self.from_dict(data, self.storage_path)
                with self._lock:
                    self.entities = kg.entities
                    self.relations = kg.relations
                    self.entity_index = kg.entity_index
                    self.relation_index = kg.relation_index
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
                logger.error(f"KG load error: {type(e).__name__}")


# ═════════════════════════════════════════════════════════════════
# SUPER MEMORY (Full Integration)
# ═════════════════════════════════════════════════════════════════

class SuperMemory:
    """
    Full memory integration.
    
    Combines:
    - Core memory (permanent facts)
    - Working memory (current context)
    - Archival memory (long-term storage)
    - Knowledge graph (entity-relationships)
    """
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or (Path.home() / "digital-shadow" / "data" / "memory")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        self.core_memory = CoreMemory(self.storage_dir / "core_memory.json")
        self.working_memory = WorkingMemory()
        self.archival_memory = ArchivalMemory(self.storage_dir / "archival")
        self.knowledge_graph = KnowledgeGraph(self.storage_dir / "knowledge_graph.json")
        
        self._lock = threading.Lock()
        
        self.core_memory.load()
        self.knowledge_graph.load()
        
        self._save_thread = None
        self._save_interval = 60
        self._running = True
    
    def start_background_save(self):
        """Start background save thread."""
        def _save_loop():
            while self._running:
                time.sleep(self._save_interval)
                self.save()
        
        self._save_thread = threading.Thread(target=_save_loop, daemon=True)
        self._save_thread.start()
    
    def stop_background_save(self):
        """Stop background save thread."""
        self._running = False
        self.save()
    
    def save(self):
        """Save all memory."""
        self.core_memory.save()
        self.archival_memory._save()
        self.knowledge_graph.save()
    
    def remember(self, content: str, memory_type: str = "archival",
                 metadata: Dict = None, importance: float = None):
        """
        Remember something.
        
        memory_type: "core", "working", "archival"
        """
        content = _sanitize_content(content)[:100000]
        if memory_type == "core":
            self.core_memory.append_block("recent", content)
        elif memory_type == "working":
            sanitized_metadata = None
            if metadata:
                sanitized_metadata = {str(k)[:50]: str(v)[:500] for k, v in list(metadata.items())[:50]}
            self.working_memory.add_message("observation", content, sanitized_metadata)
        else:
            sanitized_metadata = None
            if metadata:
                sanitized_metadata = {str(k)[:50]: str(v)[:500] for k, v in list(metadata.items())[:50]}
            self.archival_memory.insert(content, sanitized_metadata, importance or 0.5)
        
        self._extract_entities(content, metadata)
    
    def _extract_entities(self, content: str, metadata: Dict = None):
        """Extract entities for knowledge graph (simple extraction)."""
        person_matches = re.findall(r'\b([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)\b', content)
        for name in person_matches[:3]:
            if name not in ["I", "The", "This", "That"]:
                existing = self.knowledge_graph.find_entities(entity_type="person", name_contains=name)
                if not existing:
                    self.knowledge_graph.add_entity("person", name, importance=0.7)
    
    def recall(self, query: str, limit: int = 10) -> Dict:
        """
        Recall from all memory layers.
        """
        results = {
            "core": [],
            "working": [],
            "archival": [],
            "knowledge": []
        }
        
        with self._lock:
            for label, block in self.core_memory.blocks.items():
                if query.lower() in block.value.lower():
                    results["core"].append({
                        "label": label,
                        "content": block.value,
                        "importance": block.importance
                    })
        
        with self.working_memory._lock:
            for msg in self.working_memory.messages:
                if query.lower() in msg["content"].lower():
                    results["working"].append(copy.deepcopy(msg))
        
        results["archival"] = self.archival_memory.search(query, limit=limit)
        results["knowledge"] = self.knowledge_graph.query({"name_contains": query})
        
        return results
    
    def get_full_context(self) -> str:
        """Get full context for LLM."""
        parts = []
        
        parts.append(self.core_memory.get_context())
        parts.append("\n" + self.working_memory.get_context())
        
        recent = self.archival_memory.get_recent(hours=24, limit=5)
        if recent:
            parts.append("\n## Recent Memories")
            for m in recent:
                parts.append(f"- {m['content'][:100]}")
        
        return "\n".join(parts)
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            recent_msgs = self.working_memory.messages
        with self.archival_memory._lock:
            archival_count = len(self.archival_memory.memories)
        return {
            "core_blocks": len(self.core_memory.blocks),
            "working_messages": len(recent_msgs),
            "archival_memories": archival_count,
            "knowledge_entities": len(self.knowledge_graph.entities),
            "knowledge_relations": len(self.knowledge_graph.relations)
        }


__all__ = [
    'MemoryBlock',
    'CoreMemory',
    'WorkingMemory',
    'ArchivalMemory',
    'Entity',
    'Relation',
    'KnowledgeGraph',
    'SuperMemory',
]