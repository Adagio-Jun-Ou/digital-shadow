#!/usr/bin/env python3
"""
Block-Based Memory System - Based on Letta/MemGPT Research

Implements:
- Memory blocks with labels (core, persona, human, etc.)
- Three-tier memory: Core, Archival, Working
- Self-editing context
- Recall/retrieve operations
"""

from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

def _bm_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default if default is not None else datetime.now(timezone.utc)
from pathlib import Path
import copy
import json
import os
import re as _re
import threading
import uuid
import logging

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text)


@dataclass
class MemoryBlock:
    """
    A block of memory (from Letta pattern).
    
    Memory blocks are labeled segments that can be:
    - Core: Always in context
    - Archival: Retrieved via search
    - Working: Temporary context
    """
    label: str
    value: str
    limit: int = 2000  # Character limit
    protected: bool = False  # Can't be modified by AI
    metadata: Dict[str, Any] = field(default_factory=dict)
    last_modified: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    access_count: int = 0
    
    def __post_init__(self):
        """Truncate value if over limit."""
        self.label = _SANITIZE_STRICT_RE.sub('', self.label) if self.label else self.label
        if len(self.value) > self.limit and self.limit > 0:
            self.value = self.value[:self.limit]
    
    def append(self, content: str) -> bool:
        """Append content to block."""
        if len(self.value) + len(content) > self.limit:
            return False
        
        self.value += "\n" + content
        self.last_modified = datetime.now(timezone.utc)
        return True
    
    def replace(self, new_value: str) -> bool:
        """Replace entire block content."""
        if len(new_value) > self.limit:
            return False
        
        self.value = new_value
        self.last_modified = datetime.now(timezone.utc)
        return True
    
    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "value": self.value,
            "limit": self.limit,
            "protected": self.protected,
            "metadata": copy.deepcopy(self.metadata),
            "last_modified": self.last_modified.isoformat(),
            "access_count": self.access_count
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'MemoryBlock':
        return cls(
            label=data["label"],
            value=data["value"],
            limit=data.get("limit", 2000),
            protected=data.get("protected", False),
            metadata=data.get("metadata", {}),
            last_modified=_bm_ts(data, "last_modified", datetime.now(timezone.utc)),
            access_count=data.get("access_count", 0)
        )


class CoreMemory:
    """
    Core memory always in context (Letta pattern).
    
    Limited capacity, directly accessible.
    """
    
    def __init__(self):
        self.blocks: Dict[str, MemoryBlock] = {}
        self._initialize_default_blocks()
    
    def _initialize_default_blocks(self):
        """Initialize default memory blocks."""
        # Human context (about the user)
        self.blocks["human"] = MemoryBlock(
            label="human",
            value="",
            limit=2000,
            protected=False
        )
        
        # Persona (agent behavior)
        self.blocks["persona"] = MemoryBlock(
            label="persona",
            value="I am a helpful AI assistant...",
            limit=2000,
            protected=True
        )
        
        # Recent events
        self.blocks["recent"] = MemoryBlock(
            label="recent",
            value="",
            limit=1000,
            protected=False
        )
    
    def get_block(self, label: str) -> Optional[MemoryBlock]:
        """Get a memory block (returns a copy)."""
        block = self.blocks.get(label)
        if block:
            block.access_count += 1
            return copy.deepcopy(block)
        return None
    
    def get(self, label: str) -> Optional[str]:
        """Get block content (alias for get_block returning value)."""
        block = self.get_block(label)
        return block.value if block else None
    
    def set_block(self, label: str, value: str) -> bool:
        """Set block content."""
        label = _SANITIZE_STRICT_RE.sub('', label) if label else label
        if label not in self.blocks:
            self.blocks[label] = MemoryBlock(label=label, value=value, protected=True)
        else:
            if self.blocks[label].protected:
                return False
            self.blocks[label].replace(value)
        return True
    
    def append_block(self, label: str, content: str) -> bool:
        """Append to block."""
        if label not in self.blocks:
            return self.set_block(label, content)
        return self.blocks[label].append(content)
    
    def get_context(self, include_labels: List[str] = None) -> str:
        """
        Get core memory context for LLM.
        
        Args:
            include_labels: Specific blocks to include (None = all)
        
        Returns:
            Formatted context string
        """
        if include_labels:
            blocks = [self.blocks[l] for l in include_labels if l in self.blocks]
        else:
            blocks = list(self.blocks.values())
        
        lines = []
        for block in blocks:
            if block.value:
                lines.append(f"### {block.label.upper()}\n{block.value}\n")
        
        return "\n".join(lines)
    
    def save(self):
        """No-op for compatibility. Use BlockBasedMemory.save() for persistence."""
        pass
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'CoreMemory':
        memory = cls()
        memory.blocks = {label: MemoryBlock.from_dict(block_data) 
                        for label, block_data in data.items()}
        return memory
    
    def get_all(self) -> List[str]:
        """Get all block labels."""
        return list(self.blocks.keys())
    
    def search(self, query: str) -> List[Dict]:
        """Search blocks for query."""
        results = []
        query_lower = query.lower()
        for label, block in self.blocks.items():
            if query_lower in block.value.lower():
                results.append({
                    "label": label,
                    "value": block.value,
                    "match": True
                })
        return results

    def clear(self):
        """Clear all non-protected blocks."""
        for label, block in self.blocks.items():
            if not block.protected:
                block.value = ""
                block.last_modified = datetime.now(timezone.utc)

    def clear_block(self, label: str):
        """Clear a specific block if not protected."""
        if label in self.blocks and not self.blocks[label].protected:
            self.blocks[label].value = ""
            self.blocks[label].last_modified = datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        return {label: block.to_dict() for label, block in self.blocks.items()}


class ArchivalMemory:
    """
    Archival memory for long-term storage (Letta pattern).
    
    Unlimited capacity, retrieved via search.
    """
    
    _MAX_LONG_TERM = 50000
    
    def __init__(self, storage_dir: Path = None):
        self.storage_dir = storage_dir or (Path.home() / "digital-shadow" / "data" / "archival")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        
        # In-memory index
        self.memories: List[Dict[str, Any]] = []
        self._dirty = False
        self._last_save_time = 0.0
        self._save_interval = 5.0  # debounce: save at most every 5 seconds
        self._save_lock = threading.Lock()
        self._load()
    
    def insert(self, content: str, metadata: Dict = None) -> str:
        """
        Insert into archival memory.
        
        Returns:
            Memory ID
        """
        memory_id = f"arch_{uuid.uuid4().hex[:12]}"
        
        memory = {
            "id": memory_id,
            "content": content,
            "metadata": metadata or {},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "access_count": 0
        }
        
        self.memories.append(memory)
        if len(self.memories) > self._MAX_LONG_TERM:
            self.memories = self.memories[-self._MAX_LONG_TERM:]
        self._mark_dirty()
        
        return memory_id
    
    def search(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search archival memory.
        
        Simple keyword search (would use embeddings in production).
        
        Note: This method mutates access_count on matching memories as a
        side effect, incrementing it by 1 for each search hit. This affects
        result ordering (results are sorted by access_count descending) and
        persists through _mark_dirty() / _save() calls.
        """
        results = []
        query_lower = query.lower()
        
        for memory in self.memories:
            if query_lower in memory["content"].lower():
                memory["access_count"] += 1
                results.append(memory)
        
        # Sort by access count (most accessed first)
        results.sort(key=lambda m: m["access_count"], reverse=True)
        
        return copy.deepcopy(results[:limit])
    
    def get_all(self, limit: int = 100) -> List[Dict]:
        """Get all archival memories."""
        return copy.deepcopy(self.memories[:limit])
    
    def delete(self, memory_id: str) -> bool:
        """Delete a memory."""
        for i, memory in enumerate(self.memories):
            if memory["id"] == memory_id:
                self.memories.pop(i)
                self._save()
                return True
        return False
    
    def _mark_dirty(self):
        """Mark data as dirty and debounce saves."""
        import time
        with self._save_lock:
            self._dirty = True
            now = time.time()
            if now - self._last_save_time >= self._save_interval:
                self._save()
                self._dirty = False
                self._last_save_time = now
    
    def flush(self):
        """Force save any dirty data."""
        with self._save_lock:
            if self._dirty:
                self._save()
                self._dirty = False
                import time
                self._last_save_time = time.time()
    
    def _save(self):
        """Save to disk atomically."""
        import tempfile
        data_file = self.storage_dir / "archival.json"
        tmp = data_file.with_suffix('.json.tmp')
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json.dumps(self.memories, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp), str(data_file))
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(str(tmp))
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError):
            raise
    
    def _load(self):
        """Load from disk."""
        data_file = self.storage_dir / "archival.json"
        if data_file.exists():
            try:
                self.memories = json.loads(data_file.read_text())
            except (json.JSONDecodeError, OSError, ValueError):
                self.memories = []
    
    def size(self) -> int:
        """Return number of memories."""
        return len(self.memories)
    
    @property
    def entries(self) -> List[Dict]:
        """Alias for memories (for test compatibility)."""
        return self.memories
    
    @property
    def blocks(self) -> List[Dict]:
        """Another alias for memories."""
        return self.memories


class WorkingMemory:
    """
    Working memory for temporary context (Letta pattern).
    
    System-managed, automatically sized.
    """
    
    def __init__(self, max_messages: int = 10):
        self.max_messages = max_messages
        self.messages: List[Dict[str, str]] = []
    
    def add_message(self, role: str, content: str):
        """Add a message to working memory."""
        # Trim first if at capacity, then append — avoids temporarily
        # exceeding max_messages by 1.
        if len(self.messages) >= self.max_messages:
            self.messages = self.messages[-(self.max_messages - 1):]
        self.messages.append({
            "role": role,
            "content": content,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
    
    def get_messages(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return list(self.messages)
    
    def clear(self):
        """Clear working memory."""
        self.messages = []
    
    def get_context(self) -> str:
        """Get formatted context string."""
        lines = []
        for msg in self.messages:
            role = msg.get("role", "unknown")
            content = msg.get("content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def get_recent(self, n: int = 10) -> List[Dict[str, str]]:
        """Get recent messages."""
        return list(self.messages[-n:])
    
    def get_all(self) -> List[Dict[str, str]]:
        """Get all messages."""
        return list(self.messages)
    
    def size(self) -> int:
        """Get number of messages."""
        return len(self.messages)
    
    def empty(self) -> bool:
        """Check if memory is empty."""
        return len(self.messages) == 0
    
    def to_dict(self) -> Dict:
        return {
            "messages": copy.deepcopy(self.messages),
            "max_messages": self.max_messages
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'WorkingMemory':
        memory = cls(max_messages=data.get("max_messages", 10))
        memory.messages = data.get("messages", [])
        return memory


class BlockBasedMemory:
    """
    Complete memory system (Letta-inspired).
    
    Combines:
    - Core memory (always in context)
    - Archival memory (retrieved on demand)
    - Working memory (temporary)
    """
    
    def __init__(self, storage_dir: Path = None, working_limit: int = None, 
                 data_dir: Path = None, autosave: bool = False,
                 core_limit: int = None, archival_limit: int = None):
        # Support both storage_dir and data_dir parameters
        self.storage_dir = storage_dir or data_dir or (Path.home() / "digital-shadow" / "data" / "memory_blocks")
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.storage_dir  # Alias
        self.autosave = autosave
        self.core_limit = core_limit
        self.archival_limit = archival_limit
        
        self.core_memory = CoreMemory()
        self.archival_memory = ArchivalMemory(self.storage_dir / "archival")
        self.working_memory = WorkingMemory(max_messages=working_limit or 10)
        
        self._load()
    
    @property
    def core(self) -> 'CoreMemory':
        """Alias for core_memory."""
        return self.core_memory
    
    @property
    def working(self) -> 'WorkingMemory':
        """Alias for working_memory."""
        return self.working_memory
    
    @property
    def archival(self) -> 'ArchivalMemory':
        """Alias for archival_memory."""
        return self.archival_memory
    
    def core_append(self, label: str, content: str) -> bool:
        """Append to core memory block."""
        return self.core_memory.append_block(label, content)
    
    def core_set(self, label: str, value: str) -> bool:
        """Set core memory block."""
        return self.core_memory.set_block(label, value)
    
    def core_get(self, label: str) -> Optional[str]:
        """Get core memory block content."""
        block = self.core_memory.get_block(label)
        return block.value if block else None
    
    def archival_insert(self, content: str, metadata: Dict = None) -> str:
        """Insert into archival memory."""
        return self.archival_memory.insert(content, metadata)
    
    def archival_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search archival memory."""
        return self.archival_memory.search(query, limit)
    
    def working_add(self, role: str, content: str):
        """Add to working memory."""
        self.working_memory.add_message(role, content)
    
    def working_get(self) -> List[Dict[str, str]]:
        """Get working memory."""
        return self.working_memory.get_messages()
    
    def get_full_context(self) -> str:
        """
        Get full context for LLM.
        
        Includes:
        - Core memory
        - Working memory
        """
        lines = ["# Memory Context\n"]
        
        # Core memory
        lines.append(self.core_memory.get_context())
        lines.append("")
        
        # Working memory
        if self.working_memory.messages:
            lines.append("### Recent Conversation")
            for msg in self.working_memory.messages[-5:]:  # Last 5 messages
                role = msg["role"].upper()
                content = msg["content"][:200]
                lines.append(f"{role}: {content}\n")
        
        return "\n".join(lines)
    
    def recall(self, query: str, limit: int = None) -> Union[str, List[Dict]]:
        """
        Recall from memory (Letta pattern).
        
        Returns:
            str: Formatted context string when limit is None (default).
            List[Dict]: Search results when limit is provided.
        """
        # If limit provided, return list
        if limit is not None:
            return self.recall_search(query, limit=limit)
        
        # Otherwise return string context
        archival_results = self.archival_search(query, limit=3)
        
        lines = [f"# Recall: {query}\n"]
        
        if archival_results:
            lines.append("### Relevant Memories")
            for memory in archival_results:
                lines.append(f"- {memory['content'][:200]}\n")
        
        # Add core memory
        lines.append("\n### Core Memory")
        lines.append(self.core_memory.get_context())
        
        return "\n".join(lines)
    
    def recall_search(self, query: str, limit: int = 10) -> List[Dict]:
        """Search across all memory tiers, returning list."""
        results = []
        
        # Search archival
        for mem in self.archival_search(query, limit=limit):
            results.append({
                "tier": "archival",
                "content": mem["content"],
                "metadata": mem.get("metadata", {})
            })
        
        # Search core
        for label, block in self.core_memory.blocks.items():
            if query.lower() in block.value.lower():
                results.append({
                    "tier": "core",
                    "label": label,
                    "content": block.value
                })
        
        return results[:limit]
    
    def _save(self):
        """Save all memory atomically."""
        data = {
            "core_memory": self.core_memory.to_dict(),
            "working_memory": self.working_memory.to_dict()
        }
        data_file = self.storage_dir / "memory_blocks.json"
        tmp = data_file.with_suffix('.json.tmp')
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json.dumps(data, indent=2, default=str))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp), str(data_file))
            except (OSError, TypeError, ValueError):
                try:
                    os.unlink(str(tmp))
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError):
            raise
    
    def _load(self):
        """Load all memory."""
        data_file = self.storage_dir / "memory_blocks.json"
        if data_file.exists():
            try:
                data = json.loads(data_file.read_text())
                if "core_memory" in data:
                    self.core_memory = CoreMemory.from_dict(data["core_memory"])
                if "working_memory" in data:
                    self.working_memory = WorkingMemory.from_dict(data["working_memory"])
            except (json.JSONDecodeError, OSError, ValueError):
                pass
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        return {
            "core_blocks": len(self.core_memory.blocks),
            "core_total_chars": sum(len(b.value) for b in self.core_memory.blocks.values()),
            "archival_count": len(self.archival_memory.memories),
            "working_messages": len(self.working_memory.messages)
        }
    
    # Alias methods for backward compatibility with tests
    def add_to_core(self, label: str, content: str) -> bool:
        """Alias for core_append."""
        return self.core_append(label, content)
    
    def add_to_archival(self, content: str, metadata: Dict = None) -> str:
        """Alias for archival_insert."""
        return self.archival_insert(content, metadata)
    
    def add_to_working(self, content: str, role: str = "user") -> None:
        """Add content to working memory with role."""
        return self.working_add(role, content)
    
    def save(self) -> None:
        """Alias for _save (public method)."""
        self._save()
    
    def recall_list(self, query: str, limit: int = 10) -> List[Dict]:
        """Recall that returns a list for compatibility."""
        results = []
        
        # Search archival
        archival_results = self.archival_search(query, limit=limit)
        for mem in archival_results:
            results.append({
                "tier": "archival",
                "content": mem["content"],
                "metadata": mem.get("metadata", {})
            })
        
        # Search core
        for label, block in self.core_memory.blocks.items():
            if query.lower() in block.value.lower():
                results.append({
                    "tier": "core",
                    "label": label,
                    "content": block.value
                })
        
        return results[:limit]


if __name__ == "__main__":
    # Example usage
    from pathlib import Path
    import tempfile
    
    memory = BlockBasedMemory(Path(tempfile.mkdtemp()))
    
    # Core memory
    memory.core_set("human", "User prefers Python over JavaScript")
    memory.core_append("recent", "Asked about memory systems")
    
    # Archival memory
    memory.archival_insert("Learned about LangGraph state management", {"source": "research"})
    memory.archival_insert("Implemented Letta-style blocks", {"source": "coding"})
    
    # Working memory
    memory.working_add("user", "What did I ask about earlier?")
    memory.working_add("assistant", "You asked about memory systems.")
    
    # Recall
    print(memory.recall("memory"))
    
    # Stats
    print(f"\nStats: {memory.get_stats()}")