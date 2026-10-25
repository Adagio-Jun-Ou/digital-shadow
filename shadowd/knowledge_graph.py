#!/usr/bin/env python3
"""
Knowledge Graph with Temporal Awareness - Based on Graphiti/Zep Research

Implements:
- Temporal facts (valid_at/invalid_at)
- Episode provenance (every fact from somewhere)
- Entity summaries that evolve
- Knowledge lifecycle
"""

import copy
import hashlib
import json
import os
import re as _re
import threading
import uuid
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone

def _kg_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple
import logging

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text)


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text)


@dataclass
class TemporalFact:
    """A fact that is valid for a time range."""
    fact_id: str
    content: str
    entities: List[str]  # Entities mentioned
    valid_from: datetime
    valid_until: Optional[datetime] = None  # None = still valid
    confidence: float = 1.0
    source_episode: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.content = _sanitize_content(self.content)
        self.source_episode = _sanitize(self.source_episode) if self.source_episode else self.source_episode
        if self.entities and isinstance(self.entities, list):
            self.entities = [_sanitize(e) for e in self.entities]
        self.confidence = max(0.0, min(1.0, self.confidence))
    
    def is_valid_at(self, when: datetime = None) -> bool:
        """Check if fact is valid at a given time.

        Validity is a half-open interval ``[valid_from, valid_until)``: a fact
        invalidated at time T is no longer valid at exactly T. This matters
        when invalidation and the validity check happen in quick succession.
        """
        when = when or datetime.now(timezone.utc)
        
        if when < self.valid_from:
            return False
        
        if self.valid_until and when >= self.valid_until:
            return False
        
        return True
    
    def invalidate(self, when: datetime = None):
        """Invalidate this fact."""
        self.valid_until = when or datetime.now(timezone.utc)
    
    def to_dict(self) -> Dict:
        return {
            "fact_id": self.fact_id,
            "content": self.content,
            "entities": list(self.entities),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "source_episode": self.source_episode,
            "created_at": self.created_at.isoformat()
        }


class RelationType:
    KNOWS = "knows"
    WORKS_FOR = "works_for"
    LIVES_IN = "lives_in"
    REPORTS_TO = "reports_to"
    MEMBER_OF = "member_of"
    LOCATED_IN = "located_in"
    RELATED_TO = "related_to"
    MENTORED_BY = "mentored_by"


@dataclass
class TypedRelation:
    """A typed, directed relation between two entities."""
    relation_id: str
    from_entity: str
    to_entity: str
    relation_type: str
    properties: Dict = field(default_factory=dict)
    confidence: float = 1.0
    valid_from: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    valid_until: Optional[datetime] = None
    source_episode: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    VALID_TYPES = {
        RelationType.KNOWS, RelationType.WORKS_FOR, RelationType.LIVES_IN,
        RelationType.REPORTS_TO, RelationType.MEMBER_OF, RelationType.LOCATED_IN,
        RelationType.RELATED_TO, RelationType.MENTORED_BY,
    }

    def __post_init__(self):
        if isinstance(self.from_entity, str):
            self.from_entity = _sanitize(self.from_entity)
        if isinstance(self.to_entity, str):
            self.to_entity = _sanitize(self.to_entity)
        if isinstance(self.relation_type, str):
            self.relation_type = _sanitize(self.relation_type)
        if self.source_episode:
            self.source_episode = _sanitize(self.source_episode)
        self.confidence = max(0.0, min(1.0, self.confidence))

    def is_valid_at(self, when: datetime = None) -> bool:
        when = when or datetime.now(timezone.utc)
        if when < self.valid_from:
            return False
        if self.valid_until and when >= self.valid_until:
            return False
        return True

    def invalidate(self, when: datetime = None):
        self.valid_until = when or datetime.now(timezone.utc)

    def to_dict(self) -> Dict:
        return {
            "relation_id": self.relation_id,
            "from_entity": self.from_entity,
            "to_entity": self.to_entity,
            "relation_type": self.relation_type,
            "properties": copy.deepcopy(self.properties) if self.properties else {},
            "confidence": round(max(0.0, min(1.0, self.confidence)), 3),
            "valid_from": self.valid_from.isoformat(),
            "valid_until": self.valid_until.isoformat() if self.valid_until else None,
            "source_episode": self.source_episode,
            "created_at": self.created_at.isoformat(),
        }


@dataclass
class Episode:
    """An episode (observation, event, conversation)."""
    episode_id: str
    timestamp: datetime
    content: str
    source: str  # Where did this come from?
    facts_extracted: List[TemporalFact] = field(default_factory=list)
    entities_seen: Set[str] = field(default_factory=set)

    def __post_init__(self):
        self.content = _sanitize_content(self.content)
        self.source = _sanitize(self.source)
    
    def to_dict(self) -> Dict:
        return {
            "episode_id": self.episode_id,
            "timestamp": self.timestamp.isoformat(),
            "content": self.content,
            "source": self.source,
            "facts_extracted": [f.to_dict() for f in self.facts_extracted],
            "entities_seen": list(self.entities_seen)
        }

@dataclass
class Entity:
    """An entity (person, place, thing) with evolving knowledge."""
    name: str
    entity_type: str  # person, place, thing, concept
    facts: List[TemporalFact] = field(default_factory=list)
    summary: str = ""
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    def __post_init__(self):
        self.name = _sanitize(self.name)
        self.entity_type = _sanitize(self.entity_type)
        self.summary = _sanitize_content(self.summary)
    def get_valid_facts(self, when: datetime = None) -> List[TemporalFact]:
        """Get facts valid at a given time."""
        return [f for f in self.facts if f.is_valid_at(when)]
    
    def get_summary(self, when: datetime = None) -> str:
        """Get entity summary at a given time."""
        valid_facts = self.get_valid_facts(when)
        
        if not valid_facts:
            return self.summary
        
        return "; ".join(f.content for f in valid_facts[:5])
    
    def add_fact(self, fact: TemporalFact):
        """Add a fact, potentially invalidating old conflicting facts."""
        # Invalidate conflicting facts
        for existing in self.facts:
            if existing.is_valid_at() and self._facts_conflict(existing, fact):
                existing.invalidate()
        
        self.facts.append(fact)
        if len(self.facts) > 1000:
            self.facts = self.facts[-1000:]
        self.summary = self.get_summary()
        self.last_updated = datetime.now(timezone.utc)
    
    def _facts_conflict(self, fact1: TemporalFact, fact2: TemporalFact) -> bool:
        """Check if two facts conflict. Requires shared entities + high word overlap."""
        shared_entities = set(fact1.entities) & set(fact2.entities)
        if not shared_entities:
            return False
        words1 = set(_re.findall(r'[a-z][a-z\'-]*', fact1.content.lower()))
        words2 = set(_re.findall(r'[a-z][a-z\'-]*', fact2.content.lower()))
        overlap = words1 & words2
        return len(overlap) > max(len(words1), len(words2)) * 0.5
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "entity_type": self.entity_type,
            "facts": [f.to_dict() for f in self.facts],
            "summary": self.summary,
            "last_updated": self.last_updated.isoformat()
        }

class TemporalKnowledgeGraph:
    """
    Knowledge graph with temporal awareness.
    
    From Graphiti/Zep research:
    - Facts have temporal validity
    - Every fact has provenance
    - Entities have evoling summaries
    """
    
    _SAVE_DIRTY = True
    _SAVE_CLEAN = False
    _MAX_ENTITIES = 5000
    _MAX_EPISODES = 10000
    _MAX_ENTITY_EPISODES = 100
    
    def __init__(self, data_dir: Path = None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data" / "knowledge_graph")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.entities: Dict[str, Entity] = {}
        self.episodes: List[Episode] = []
        self.entity_episodes: Dict[str, List[str]] = defaultdict(list)
        self.relations: List[TypedRelation] = []
        self._relation_index: Dict[str, List[int]] = defaultdict(list)
        self._MAX_RELATIONS = 10000
        self._dirty: bool = False
        self._auto_save: bool = True
        self._lock = threading.Lock()
        
        self._load()
    
    def add_episode(
        self,
        content: str,
        source: str,
        facts: List[Dict] = None,
        entities: List[str] = None
    ) -> Episode:
        """
        Add an episode and extract facts.
        
        Args:
            content: Episode content
            source: Where this came from
            facts: Extracted facts (list of {content, entities})
            entities: Entities mentioned
        
        Returns:
            Created episode
        """
        content = _sanitize_content(content)
        source = _sanitize(source)
        entities = [_sanitize(e) for e in (entities or [])]
        
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        
        episode = Episode(
            episode_id=episode_id,
            timestamp=datetime.now(timezone.utc),
            content=content,
            source=source,
            entities_seen=set(entities)
        )
        
        # Extract facts
        for fact_data in (facts or []):
            fact_content = _sanitize_content(fact_data.get("content", ""))
            fact_entities = [_sanitize(e) for e in fact_data.get("entities", [])]
            fact = TemporalFact(
                fact_id=f"f_{uuid.uuid4().hex[:12]}_{hashlib.sha256(fact_content.encode()).hexdigest()[:8]}",
                content=fact_content,
                entities=fact_entities,
                valid_from=datetime.now(timezone.utc),
                source_episode=episode_id
            )
            
            episode.facts_extracted.append(fact)
            
            # Add to entities
            for entity_name in fact.entities:
                with self._lock:
                    if entity_name not in self.entities:
                        if len(self.entities) >= self._MAX_ENTITIES:
                            logger.warning("Entity limit reached (%d), evicting oldest", self._MAX_ENTITIES)
                            oldest_key = next(iter(self.entities))
                            del self.entities[oldest_key]
                        self.entities[entity_name] = Entity(
                            name=entity_name,
                            entity_type="unknown"
                        )
                    
                    self.entities[entity_name].add_fact(fact)
        
        with self._lock:
            self.episodes.append(episode)
            if len(self.episodes) > self._MAX_EPISODES:
                self.episodes = self.episodes[-self._MAX_EPISODES:]
            
            # Track episode->entity mapping
            for entity in entities:
                self.entity_episodes[entity].append(episode_id)
                if len(self.entity_episodes[entity]) > self._MAX_ENTITY_EPISODES:
                    self.entity_episodes[entity] = self.entity_episodes[entity][-self._MAX_ENTITY_EPISODES:]
        
        self._save()
        return episode
    
    def add_relation(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str,
        properties: Dict = None,
        confidence: float = 1.0,
        source_episode: str = None,
    ) -> TypedRelation:
        """Add a typed relation between two entities."""
        from_entity = _sanitize(from_entity)
        to_entity = _sanitize(to_entity)
        if relation_type not in TypedRelation.VALID_TYPES:
            relation_type = RelationType.RELATED_TO
        confidence = max(0.0, min(1.0, confidence))
        if properties:
            properties = {str(k)[:50]: str(v)[:500] for k, v in list(properties.items())[:50]}
        if source_episode:
            source_episode = _sanitize(source_episode)
        relation = TypedRelation(
            relation_id=f"rel_{uuid.uuid4().hex[:12]}",
            from_entity=from_entity,
            to_entity=to_entity,
            relation_type=relation_type,
            properties=properties or {},
            confidence=confidence,
            source_episode=source_episode,
        )
        with self._lock:
            self.relations.append(relation)
            idx = len(self.relations) - 1
            self._relation_index[from_entity].append(idx)
            self._relation_index[to_entity].append(idx)
            if len(self.relations) > self._MAX_RELATIONS:
                self.relations = self.relations[-self._MAX_RELATIONS:]
                self._rebuild_relation_index()
        self._save()
        return relation
    
    def get_relations(
        self,
        entity: str,
        relation_type: str = None,
        direction: str = "both",
        when: datetime = None,
    ) -> List[TypedRelation]:
        """Get relations involving an entity."""
        with self._lock:
            indices = self._relation_index.get(entity, [])
            results = []
            for idx in indices:
                if idx >= len(self.relations):
                    continue
                rel = self.relations[idx]
                if direction == "outgoing" and rel.from_entity != entity:
                    continue
                if direction == "incoming" and rel.to_entity != entity:
                    continue
                if relation_type and rel.relation_type != relation_type:
                    continue
                if not rel.is_valid_at(when):
                    continue
                results.append(rel)
            return copy.deepcopy(results)
    
    def get_relation_between(
        self,
        from_entity: str,
        to_entity: str,
        relation_type: str = None,
    ) -> List[TypedRelation]:
        """Get relations from one entity to another."""
        with self._lock:
            results = []
            for rel in self.relations:
                if rel.from_entity == from_entity and rel.to_entity == to_entity:
                    if relation_type and rel.relation_type != relation_type:
                        continue
                    if rel.is_valid_at():
                        results.append(rel)
            return copy.deepcopy(results)
    
    def _rebuild_relation_index(self):
        self._relation_index = defaultdict(list)
        for idx, rel in enumerate(self.relations):
            self._relation_index[rel.from_entity].append(idx)
            self._relation_index[rel.to_entity].append(idx)
    
    def get_entity(self, name: str) -> Optional[Entity]:
        """Get entity by name."""
        with self._lock:
            entity = self.entities.get(name)
            return copy.deepcopy(entity) if entity else None
    
    def get_entity_facts(
        self, 
        name: str, 
        when: datetime = None
    ) -> List[TemporalFact]:
        """Get facts about entity at a given time."""
        with self._lock:
            entity = self.entities.get(name)
            if not entity:
                return []
            return copy.deepcopy(entity.get_valid_facts(when))
    
    def get_entity_summary(self, name: str, when: datetime = None) -> str:
        """Get entity summary at a given time."""
        with self._lock:
            entity = self.entities.get(name)
            if not entity:
                return ""
            return entity.get_summary(when)
    
    def search(
        self, 
        query: str, 
        when: datetime = None,
        entities: List[str] = None
    ) -> List[TemporalFact]:
        """
        Search for facts.
        
        Args:
            query: Search query
            when: Time point (default = now UTC)
            entities: Filter to specific entities
        
        Returns:
            Matching facts
        """
        when = when or datetime.now(timezone.utc)
        query_lower = query.lower()
        
        with self._lock:
            results = []
            
            for entity_name, entity in self.entities.items():
                if entities and entity_name not in entities:
                    continue
                
                for fact in entity.facts:
                    if not fact.is_valid_at(when):
                        continue
                    
                    if _re.search(r'\b' + _re.escape(query_lower) + r'\b', fact.content.lower()):
                        results.append(fact)
            
            return copy.deepcopy(results)
    
    def get_episodes_for_entity(self, entity: str, limit: int = 10) -> List[Episode]:
        """Get episodes mentioning an entity."""
        with self._lock:
            episode_ids = self.entity_episodes.get(entity, [])
            
            episodes = [
                ep for ep in self.episodes 
                if ep.episode_id in episode_ids
            ]
            
            return copy.deepcopy(episodes[:limit])
    
    def invalidate_fact(self, fact_id: str, when: datetime = None):
        """Invalidate a fact."""
        when = when or datetime.now(timezone.utc)
        with self._lock:
            for entity in self.entities.values():
                for fact in entity.facts:
                    if fact.fact_id == fact_id:
                        fact.invalidate(when)
                        entity.last_updated = datetime.now(timezone.utc)
                        self._save()
                        return True
        
        return False
    
    def get_all_entities(self) -> List[str]:
        """Get all entity names."""
        with self._lock:
            return list(self.entities.keys())
    
    def get_all_facts(self, when: datetime = None) -> List[TemporalFact]:
        """Get all valid facts at a time."""
        with self._lock:
            all_facts = []
            
            for entity in self.entities.values():
                all_facts.extend(entity.get_valid_facts(when))
            
            return copy.deepcopy(all_facts)
    
    def _save(self):
        """Mark dirty and schedule save (debounced). Call flush() to force immediate save."""
        self._dirty = True
        if self._auto_save:
            self.flush()
    
    def flush(self):
        """Force immediate save to disk. Thread-safe via _lock."""
        with self._lock:
            self._dirty = False
            data = {
                "entities": {name: entity.to_dict() for name, entity in self.entities.items()},
                "episodes": [ep.to_dict() for ep in self.episodes],
                "entity_episodes": dict(self.entity_episodes),
                "relations": [rel.to_dict() for rel in self.relations],
            }
            save_file = self.data_dir / "knowledge_graph.json"
            tmp = save_file.with_suffix(f'.json.tmp.{uuid.uuid4().hex[:8]}')
            try:
                data_bytes = json.dumps(data, indent=2, default=str).encode()
                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
                try:
                    os.write(fd, data_bytes)
                    os.fsync(fd)
                finally:
                    os.close(fd)
                os.replace(str(tmp), str(save_file))
            except (OSError, TypeError, ValueError) as e:
                logger.error("Failed to save knowledge graph: %s", type(e).__name__)
                if tmp.exists():
                    try:
                        tmp.unlink()
                    except OSError:
                        pass
    
    def _load(self):
        """Load from disk."""
        save_file = self.data_dir / "knowledge_graph.json"
        
        if not save_file.exists():
            return
        
        try:
            data = json.loads(save_file.read_text())
            
            # Load entities
            for name, entity_data in data.get("entities", {}).items():
                entity = Entity(
                    name=name,
                    entity_type=entity_data.get("entity_type", "unknown"),
                    summary=entity_data.get("summary", "")
                )
                
                for fact_data in entity_data.get("facts", []):
                    fact = TemporalFact(
                        fact_id=fact_data["fact_id"],
                        content=fact_data["content"],
                        entities=fact_data["entities"],
                        valid_from=_kg_ts(fact_data, "valid_from"),
                        valid_until=_kg_ts(fact_data, "valid_until"),
                        confidence=fact_data.get("confidence", 1.0),
                        source_episode=fact_data.get("source_episode"),
                        created_at=_kg_ts(fact_data, "created_at")
                    )
                    entity.facts.append(fact)
                
                self.entities[name] = entity
            
            # Load episodes
            for ep_data in data.get("episodes", []):
                episode = Episode(
                    episode_id=ep_data["episode_id"],
                    timestamp=_kg_ts(ep_data, "timestamp"),
                    content=ep_data["content"],
                    source=ep_data["source"],
                    entities_seen=set(ep_data.get("entities_seen", []))
                )
                self.episodes.append(episode)
            
            # Load entity-episode mapping
            self.entity_episodes = defaultdict(list, data.get("entity_episodes", {}))
            
            # Load relations
            for rel_data in data.get("relations", []):
                relation = TypedRelation(
                    relation_id=rel_data["relation_id"],
                    from_entity=rel_data["from_entity"],
                    to_entity=rel_data["to_entity"],
                    relation_type=rel_data["relation_type"],
                    properties=rel_data.get("properties", {}),
                    confidence=rel_data.get("confidence", 1.0),
                    valid_from=_kg_ts(rel_data, "valid_from") or datetime.now(timezone.utc),
                    valid_until=_kg_ts(rel_data, "valid_until"),
                    source_episode=rel_data.get("source_episode"),
                    created_at=_kg_ts(rel_data, "created_at") or datetime.now(timezone.utc),
                )
                self.relations.append(relation)
            self._rebuild_relation_index()
            
            logger.info(f"Loaded {len(self.entities)} entities, {len(self.episodes)} episodes, {len(self.relations)} relations")
        
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error(f"Failed to load knowledge graph: {type(e).__name__}")
    
    def get_stats(self) -> Dict:
        """Get statistics."""
        with self._lock:
            return {
                "entities": len(self.entities),
                "episodes": len(self.episodes),
                "relations": len(self.relations),
                "total_facts": sum(len(e.facts) for e in self.entities.values()),
                "valid_facts": sum(len(e.get_valid_facts()) for e in self.entities.values()),
                "entity_types": dict(Counter(e.entity_type for e in self.entities.values())),
            }


if __name__ == "__main__":
    from pathlib import Path
    import tempfile
    
    kg = TemporalKnowledgeGraph(Path(tempfile.mkdtemp()))
    
    # Add episode with facts
    episode = kg.add_episode(
        content="John called Sarah about the meeting",
        source="conversation",
        facts=[
            {"content": "John called Sarah", "entities": ["John", "Sarah"]},
            {"content": "Meeting mentioned", "entities": ["Meeting"]}
        ],
        entities=["John", "Sarah", "Meeting"]
    )
    
    # Get entity
    john = kg.get_entity("John")
    print(f"John: {john}")
    
    # Get facts
    facts = kg.get_entity_facts("John")
    print(f"Facts about John: {[f.content for f in facts]}")
    
    # Get summary
    summary = kg.get_entity_summary("John")
    print(f"Summary: {summary}")
    
    # Search
    results = kg.search("Sarah")
    print(f"Search 'Sarah': {[f.content for f in results]}")
    
    # Stats
    stats = kg.get_stats()
    print(f"Stats: {stats}")