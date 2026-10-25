#!/usr/bin/env python3
"""
RAG Memory System for Shadow AI
Based on LaminaAI architecture: Vector embeddings + Summarization + Semantic recall

Implements:
- Long-term memory via ChromaDB
- Message summarization
- Short-term memory queue
- Semantic search retrieval
"""

import json
import asyncio
import copy
import os
import re as _re
import threading
import uuid
from datetime import datetime, timedelta, timezone

def _mr_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default if default is not None else datetime.now(timezone.utc)
from typing import Dict, List, Optional, Any
from pathlib import Path
from dataclasses import dataclass, field
import hashlib
import logging

logger = logging.getLogger("shadowd.memory_rag")

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()


def _sanitize_strict(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()


try:
    import chromadb
    from chromadb.config import Settings
    CHROMADB_AVAILABLE = True
except ImportError:
    CHROMADB_AVAILABLE = False
    logger.warning("ChromaDB not installed, using JSON-based memory")

try:
    from sentence_transformers import SentenceTransformer
    EMBEDDINGS_AVAILABLE = True
except ImportError:
    EMBEDDINGS_AVAILABLE = False
    logger.warning("sentence-transformers not installed, using simple embeddings")


@dataclass
class Memory:
    """A single memory entry."""
    content: str
    timestamp: datetime
    memory_type: str  # 'observation', 'insight', 'commitment', 'learning'
    metadata: Dict = field(default_factory=dict)
    summary: Optional[str] = None
    embedding_id: Optional[str] = None

    def __post_init__(self):
        self.content = _sanitize_content(self.content)
        if self.summary:
            self.summary = _sanitize_content(self.summary)

    def to_dict(self) -> Dict:
        return {
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "memory_type": self.memory_type,
            "metadata": copy.deepcopy(self.metadata),
            "summary": self.summary,
            "embedding_id": self.embedding_id
        }


@dataclass  
class ContextWindow:
    """Short-term context queue."""
    messages: List[Dict] = field(default_factory=list)
    max_size: int = 10
    _lock: threading.Lock = field(default_factory=threading.Lock, init=False, repr=False)
    
    def add(self, message: Dict):
        with self._lock:
            self.messages.append(message)
            if len(self.messages) > self.max_size:
                return self.messages.pop(0)
            return None
    
    def get_context(self) -> List[Dict]:
        with self._lock:
            return list(self.messages[-self.max_size:])


class RAGMemorySystem:
    """
    Retrieval-Augmented Generation Memory System.
    
    Features:
    - Short-term: Recent messages (context queue)
    - Long-term: Summarized + embedded in vector DB
    - Semantic search for context recall
    - Automatic summarization
    """
    
    _DOMAIN_RE = _re.compile(r'^[a-zA-Z0-9_\-]{1,50}$')
    _MAX_LONG_TERM = 10000
    _MAX_PER_DOMAIN = 1000
    
    @classmethod
    def _validate_domain(cls, domain: str) -> str:
        """Validate domain name to prevent path traversal. Only allow alphanumeric, underscore, hyphen; max 50 chars."""
        if not domain or len(domain) > 50:
            raise ValueError(f"Domain name must be 1-50 characters, got {len(domain)}")
        if not cls._DOMAIN_RE.match(domain):
            raise ValueError(f"Domain name contains invalid characters: {domain!r}")
        return domain
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "memory")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Short-term memory (recent)
        self.context_window = ContextWindow(max_size=10)
        
        # Long-term memory (summarized)
        self.long_term_memory: List[Memory] = []
        
        # Vector database
        self.collection = None
        self.embedder = None
        self._init_vector_db()
        
        # Life domains (from Digital Twin Template)
        self.domains = {
            "health": [],
            "finance": [],
            "legal": [],
            "growth": [],
            "career": [],
            "home": [],
            "admin": []
        }
        
        self._lock = threading.Lock()
        
        self._load_memory()
    
    def _init_vector_db(self):
        """Initialize ChromaDB for vector storage."""
        if not CHROMADB_AVAILABLE:
            logger.info("Using JSON-based memory storage")
            return
        
        try:
            client = chromadb.PersistentClient(path=str(self.data_dir / "chromadb"))
            self.collection = client.get_or_create_collection(
                name="shadow_memory",
                metadata={"hnsw:space": "cosine"}
            )
            logger.info("ChromaDB initialized successfully")
        except (ValueError, OSError, RuntimeError):
            logger.error("Failed to initialize ChromaDB")
            self.collection = None
    
    def _init_embedder(self):
        """Initialize embedding model."""
        if not EMBEDDINGS_AVAILABLE:
            return
        
        try:
            self.embedder = SentenceTransformer('all-MiniLM-L6-v2')
            logger.info("Sentence transformer initialized")
        except (ValueError, OSError, RuntimeError):
            logger.error("Failed to initialize sentence transformer")
            self.embedder = None
    
    def _load_memory(self):
        """Load persisted memory."""
        memory_file = self.data_dir / "long_term_memory.json"
        if memory_file.exists():
            try:
                data = json.loads(memory_file.read_text())
                self.long_term_memory = [
                    Memory(
                        content=m["content"],
                        timestamp=_mr_ts(m, "timestamp"),
                        memory_type=m["memory_type"],
                        metadata=m.get("metadata", {}),
                        summary=m.get("summary")
                    )
                    for m in data.get("memories", [])
                ]
                logger.info(f"Loaded {len(self.long_term_memory)} memories")
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                logger.error("Failed to load long-term memory")
        
        # Load domain-specific memories
        for domain in self.domains:
            domain_file = self.data_dir / f"domain_{domain}.json"
            if domain_file.exists():
                try:
                    self.domains[domain] = json.loads(domain_file.read_text())
                except (json.JSONDecodeError, OSError, ValueError):
                    pass
    def _save_memory(self):
        """Persist memory to disk atomically."""
        import tempfile
        memory_file = self.data_dir / "long_term_memory.json"
        try:
            data = {
                "memories": [
                    {
                        "content": m.content,
                        "timestamp": m.timestamp.isoformat(),
                        "memory_type": m.memory_type,
                        "metadata": m.metadata,
                        "summary": m.summary
                    }
                    for m in self.long_term_memory
                ]
            }
            tmp = memory_file.with_suffix('.json.tmp')
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                with os.fdopen(fd, 'w') as f:
                    f.write(json.dumps(data, indent=2))
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp), str(memory_file))
            except (json.JSONDecodeError, OSError, ValueError):
                try:
                    os.unlink(str(tmp))
                except OSError:
                    pass
                raise
        except (OSError, TypeError, ValueError):
            logger.error("Failed to save memory")
    
    def _get_embedding(self, text: str) -> List[float]:
        """Get embedding for text."""
        if self.embedder:
            try:
                return self.embedder.encode(text).tolist()
            except (RuntimeError, ValueError, OSError):
                pass
        
        # Fallback: simple hash-based embedding
        hash_obj = hashlib.sha256(text.encode())
        hash_bytes = hash_obj.digest()
        return [float(b) / 255.0 for b in hash_bytes[:32]]
    
    async def add_observation(self, content: str, metadata: Dict = None, domain: str = None):
        """
        Add an observation to memory.
        
        Args:
            content: The observation content
            metadata: Additional metadata
            domain: Life domain (health, finance, etc.)
        """
        content = _sanitize_content(content)[:100000]
        if metadata:
            metadata = {str(k)[:50]: str(v)[:500] for k, v in list(metadata.items())[:50]}
        if domain:
            try:
                domain = self._validate_domain(domain)
            except ValueError:
                domain = None
        now = datetime.now(timezone.utc)

        # Determine memory type before lock
        memory_type = self._classify_memory_type(content)
        
        with self._lock:
            # Add to short-term
            self.context_window.add({
                "content": content,
                "timestamp": now.isoformat(),
                "metadata": metadata or {}
            })
            
            # Create memory
            memory = Memory(
                content=content,
                timestamp=now,
                memory_type=memory_type,
                metadata=metadata or {}
            )
            
            # Add to domain if specified
            if domain and domain in self.domains:
                self.domains[domain].append({
                    "content": content,
                    "timestamp": now.isoformat(),
                    "metadata": metadata
                })
                if len(self.domains[domain]) > self._MAX_PER_DOMAIN:
                    self.domains[domain] = self.domains[domain][-self._MAX_PER_DOMAIN:]
                self._save_domain(domain)
        
        # Add to vector DB (outside lock — can be slow)
        embedding_id = None
        if self.collection:
            try:
                embedding = self._get_embedding(content)
                memory_id = f"mem_{uuid.uuid4().hex[:12]}"
                self.collection.add(
                    embeddings=[embedding],
                    documents=[content],
                    metadatas=[{"type": memory_type, "timestamp": now.isoformat()}],
                    ids=[memory_id]
                )
                embedding_id = memory_id
            except (ValueError, RuntimeError, OSError):
                logger.error("Failed to add to vector DB")
        
        with self._lock:
            memory.embedding_id = embedding_id
            self.long_term_memory.append(memory)
            
            if len(self.long_term_memory) > self._MAX_LONG_TERM:
                self.long_term_memory = self.long_term_memory[-self._MAX_LONG_TERM:]
            
            self._save_memory()
        
        # Check if summarization needed
        if len(self.long_term_memory) % 50 == 0:
            await self._summarize_old_memories()
    
    def _classify_memory_type(self, content: str) -> str:
        """Classify memory type based on content."""
        content_lower = content.lower()
        
        if _re.search(r'\b(call|promise|tomorrow|later|will|meeting)\b', content_lower):
            return "commitment"
        elif _re.search(r'\b(learned|discovered|pattern|found)\b', content_lower):
            return "learning"
        elif _re.search(r'\b(should|could|insight|realized)\b', content_lower):
            return "insight"
        else:
            return "observation"
    
    async def _summarize_old_memories(self):
        """Summarize old memories to save space."""
        if len(self.long_term_memory) < 100:
            return
        
        # Group old memories by day
        old_memories = self.long_term_memory[:-50]  # Keep last 50 fresh
        
        # Group by date
        by_date = {}
        for mem in old_memories:
            date_str = mem.timestamp.date().isoformat()
            by_date.setdefault(date_str, []).append(mem)
        
        # Create summaries
        for date_str, memories in by_date.items():
            summary = f"{len(memories)} events on {date_str}: " + \
                     "; ".join([m.content[:50] for m in memories[:3]])
            
            # Replace with summary
            summary_mem = Memory(
                content=summary,
                timestamp=memories[0].timestamp,
                memory_type="summary",
                metadata={"count": len(memories), "date": date_str},
                summary=True
            )
            
            # Remove old, add summary
            for mem in memories:
                if mem in self.long_term_memory:
                    self.long_term_memory.remove(mem)
            self.long_term_memory.append(summary_mem)
        
        logger.info(f"Summarized {len(old_memories)} memories")
    
    def _save_domain(self, domain: str):
        """Save domain-specific data atomically. Domain is validated before calling."""
        validated = self._validate_domain(domain)
        domain_file = self.data_dir / f"domain_{validated}.json"
        try:
            data = json.dumps(self.domains[domain], indent=2, default=str)
            tmp = domain_file.with_suffix('.json.tmp')
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            try:
                os.write(fd, data.encode())
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(str(tmp), str(domain_file))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save domain %s: %s", validated, type(e).__name__)
    
    async def search(self, query: str, n_results: int = 5, domain: str = None) -> List[Dict]:
        """
        Semantic search for relevant memories.
        
        Args:
            query: Search query
            n_results: Number of results
            domain: Filter by domain (optional)
        
        Returns:
            List of matching memories
        """
        results = []
        
        # Vector search (outside lock — can be slow)
        if self.collection:
            try:
                query_embedding = self._get_embedding(query)
                vector_results = self.collection.query(
                    query_embeddings=[query_embedding],
                    n_results=n_results
                )
                
                if vector_results.get("documents"):
                    for i, doc in enumerate(vector_results["documents"][0]):
                        results.append({
                            "content": doc,
                            "metadata": vector_results["metadatas"][0][i] if vector_results.get("metadatas") else {},
                            "distance": vector_results["distances"][0][i] if vector_results.get("distances") else 0
                        })
            except (ValueError, RuntimeError, OSError):
                logger.error("Vector search failed")
        
        with self._lock:
            # Domain-specific search
            if domain and domain in self.domains:
                for item in self.domains[domain]:
                    if query.lower() in item.get("content", "").lower():
                        results.append({
                            "content": item["content"],
                            "metadata": copy.deepcopy(item.get("metadata", {})),
                            "domain": domain
                        })
            
            # Keyword search in short-term
            for msg in self.context_window.get_context():
                if query.lower() in msg.get("content", "").lower():
                    results.append({
                        "content": msg["content"],
                        "metadata": copy.deepcopy(msg.get("metadata", {})),
                        "type": "recent"
                    })
        
        return results[:n_results]
    
    def get_context_for_query(self, query: str) -> str:
        """
        Get rich context for a query.
        
        Combines:
        - Recent context (short-term)
        - Relevant memories (vector search)
        - Domain knowledge
        """
        context_parts = []
        
        # Recent context
        recent = self.context_window.get_context()
        if recent:
            context_parts.append("## Recent Activity:\n" + 
                                "\n".join([f"- {m['content']}" for m in recent[-5:]]))
        
        # Relevant long-term memories
        # Note: Cannot use async in sync context
        # TODO: Make this method async or use cached search
        
        # Check for domain keywords
        for domain in ["health", "finance", "legal", "growth", "career", "home"]:
            if _re.search(r'\b' + _re.escape(domain) + r'\b', query.lower()) and self.domains.get(domain):
                context_parts.append(f"\n## {domain.title()} Context:\n" +
                                   "\n".join([f"- {item['content']}" 
                                            for item in self.domains[domain][-3:]]))
        
        return "\n".join(context_parts) if context_parts else ""
    
    async def get_commitments(self) -> List[Memory]:
        """Get all commitments (promises, deadlines)."""
        with self._lock:
            return copy.deepcopy([m for m in self.long_term_memory if m.memory_type == "commitment"])
    
    async def get_learnings(self) -> List[Memory]:
        """Get all learned patterns."""
        with self._lock:
            return copy.deepcopy([m for m in self.long_term_memory if m.memory_type == "learning"])
    
    async def get_insights(self) -> List[Memory]:
        """Get all insights."""
        with self._lock:
            return copy.deepcopy([m for m in self.long_term_memory if m.memory_type == "insight"])
    
    def get_stats(self) -> Dict:
        """Get memory statistics."""
        with self._lock:
            return {
                "total_memories": len(self.long_term_memory),
                "short_term": len(self.context_window.messages),
                "by_type": {
                    "observation": len([m for m in self.long_term_memory if m.memory_type == "observation"]),
                    "commitment": len([m for m in self.long_term_memory if m.memory_type == "commitment"]),
                    "learning": len([m for m in self.long_term_memory if m.memory_type == "learning"]),
                    "insight": len([m for m in self.long_term_memory if m.memory_type == "insight"])
                },
                "domains": {k: len(v) for k, v in self.domains.items()},
                "vector_db": "enabled" if self.collection else "disabled"
            }
    
    def clear_old_memories(self, days: int = 30):
        """Clear memories older than specified days."""
        cutoff = datetime.now(timezone.utc) - timedelta(days=days)
        with self._lock:
            before_count = len(self.long_term_memory)
            self.long_term_memory = [
                m for m in self.long_term_memory 
                if m.timestamp > cutoff or m.memory_type == "commitment"
            ]
            self._save_memory()
        return before_count - len(self.long_term_memory)


class RAGContext:
    """
    Context manager for RAG-enhanced queries.
    Integrates with Shadow AI.
    """
    
    def __init__(self, memory_system: RAGMemorySystem):
        self.memory = memory_system
    
    async def prepare_context(self, query: str, domain: str = None) -> Dict:
        """
        Prepare rich context for a query.
        
        Returns:
            Dict with recent context, relevant memories, domain data
        """
        context = {
            "query": query,
            "recent": [],
            "relevant": [],
            "domain": None,
            "stats": self.memory.get_stats()
        }
        
        # Get recent context
        context["recent"] = self.memory.context_window.get_context()
        
        # Get relevant memories via semantic search
        context["relevant"] = await self.memory.search(query, n_results=5, domain=domain)
        
        # Get domain context
        if domain and domain in self.memory.domains:
            context["domain"] = self.memory.domains[domain][-10:]
        
        return context
    
    def context_to_prompt(self, context: Dict) -> str:
        """Convert context to LLM prompt."""
        prompt_parts = []
        
        if context.get("recent"):
            prompt_parts.append("Recent activity:\n" + 
                              "\n".join([f"- {m['content']}" for m in context["recent"][-3:]]))
        
        if context.get("relevant"):
            prompt_parts.append("Relevant context:\n" +
                              "\n".join([f"- {m['content']}" for m in context["relevant"][:3]]))
        
        if context.get("domain"):
            prompt_parts.append("Domain knowledge:\n" +
                              "\n".join([f"- {item['content']}" for item in context["domain"][-3:]]))
        
        return "\n\n".join(prompt_parts) if prompt_parts else "No relevant context available."