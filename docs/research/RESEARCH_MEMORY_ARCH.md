# Memory Architecture Research Notes

## Executive Summary

This research analyzes four major memory architecture patterns for AI agents: Letta (formerly MemGPT), Zep, Graphiti, and reviews community knowledge of LaminaAI and MemoryMesh. The insights are directly applicable to the Shadow AI project's memory system design.

---

## 1. Letta (formerly MemGPT)

**Repository**: https://github.com/letta-ai/letta  
**Stars**: 21.9k  
**License**: Apache-2.0

### Overview

Letta is a platform for building stateful AI agents with advanced memory that can learn and self-improve over time. It provides both a CLI tool (Letta Code) and an API for integrating agents into applications.

### Key Architectural Patterns

#### 1. Memory Block Architecture

Letta organizes memory into labeled blocks:

```python
# Memory blocks as typed, labeled memory segments
agent_state = client.agents.create(
    model="openai/gpt-5.2",
    memory_blocks=[
        {
            "label": "human",      # User context
            "value": "Name: Timber. Status: dog. Occupation: building Letta..."
        },
        {
            "label": "persona",   # Agent behavior context
            "value": "I am a self-improving superintelligence..."
        }
    ],
    tools=["web_search", "fetch_webpage"]
)
```

**Key Insight**: Memory is segmented by purpose (human context, persona, etc.), enabling granular retrieval and updates.

#### 2. Memory Categories

Letta uses three primary memory types:

1. **Core Memory** - Always available in-context
   - Human information (user facts)
   - Persona (agent behavior patterns)
   
2. **Archival Memory** - Long-term storage retrieved via search
   - Unlimited capacity
   - Requires explicit retrieval

3. **Working Memory** - Temporary context management
   - System-managed context window

### Memory Organization Strategy

| Memory Type | Persistence | Retrieval Method | Capacity |
|------------|--------------|------------------|----------|
| Core Memory | Persistent | Direct access | Limited (~context window) |
| Archival Memory | Persistent | Semantic search | Unlimited |
| Working Memory | Ephemeral | System-managed | Context window |

### Retrieval Techniques

```python
# Example query pattern for archival memory
response = await agent.archival_memory.search(
    query="previous conversations about preferences",
    limit=10
)
```

Letta uses:
- **Semantic embeddings** for archival memory search
- **Direct access** for core memory blocks
- **Function calls** for memory operations (core_memory_append, archival_memory_insert)

### Code Patterns for Shadow AI

```python
# Adapt block-based memory for Shadow AI
class MemoryBlock(BaseModel):
    uuid: str
    label: str  # "user", "shadow_context", "learned_patterns"
    value: str
    embedding: list[float] | None = None
    last_accessed: datetime
    access_count: int = 0

# Self-improvement memory tracking
class LearnedPattern(MemoryBlock):
    label: str = "learned_pattern"
    confidence: float = 0.0
    pattern_type: str  # "behavior", "preference", "fact"
```

---

## 2. Zep

**Repository**: https://github.com/getzep/zep  
**Stars**: 4.4k  
**License**: Apache-2.0

### Overview

Zep is an end-to-end context engineering platform delivering context assembly with sub-200ms latency. It manages chat history, business data, documents, and app events into relationship-aware context.

### Key Architectural Patterns

#### 1. Graph RAG Architecture

Zep uses Graph RAG (Retrieval-Augmented Generation) to:
- Automatically extract relationships from data
- Maintain temporal knowledge graphs
- Understand how context evolves over time

#### 2. Multi-Source Context Assembly

```python
from zep_cloud import Zep

client = Zep(api_key=os.getenv("ZEP_API_KEY"))

# Add context from multiple sources
await client.memory.add(
    session_id="user_123",
    messages=[...],        # Chat history
    metadata={...},        # Business data
    # Documents and events auto-ingested
)

# Retrieve assembled context
context = await client.memory.search(
    session_id="user_123",
    query="user preferences about X",
    limit=10
)
```

### Memory Organization Strategy

Zep organizes data into:
1. **Episodes** - Raw data (messages, documents, events)
2. **Facts** - Extracted relationships with temporal validity
3. **Entities** - People, concepts, objects
4. **Communities** - Clustered entity groups

### Temporal Knowledge Model

```python
# Zep tracks temporal validity of facts
Fact(
    content="User prefers dark mode",
    valid_at=datetime(2024, 1, 1),
    invalid_at=datetime(2024, 6, 1),  # When fact became stale
    source_episode_id="..."
)
```

### Code Patterns for Shadow AI

```python
# Temporal fact tracking
class TemporalFact(BaseModel):
    uuid: str
    content: str
    valid_at: datetime
    invalid_at: datetime | None = None  # None = still valid
    episodes: list[str] = []  # Source references
    embedding: list[float] | None = None
    confidence: float
    
    def is_valid_at(self, timestamp: datetime) -> bool:
        if self.invalid_at is None:
            return timestamp >= self.valid_at
        return self.valid_at <= timestamp < self.invalid_at
```

---

## 3. Graphiti (Zep's Core Engine)

**Repository**: https://github.com/getzep/graphiti  
**Stars**: 24.5k  
**License**: Apache-2.0

### Overview

Graphiti is the open-source temporal context graph engine powering Zep. It builds knowledge graphs that track how facts change over time, maintaining provenance to source data.

### Key Architectural Patterns

#### 1. Context Graph Components

```python
from graphiti_core import Graphiti
from graphiti_core.nodes import EntityNode, EpisodicNode
from graphiti_core.edges import EntityEdge

# Graph structure
class ContextGraph:
    """
    Context graph contains:
    - Entities (nodes): People, products, policies with evolving summaries
    - Facts/Relationships (edges): Triplets with temporal validity windows
    - Episodes (provenance): Raw data ground truth
    """
    entities: list[EntityNode]
    edges: list[EntityEdge]
    episodes: list[EpisodicNode]
```

#### 2. Temporal Fact Management

```python
# Key insight: Facts have validity windows
EntityEdge(
    source_node_uuid="entity_user",
    target_node_uuid="entity_preference",
    name="prefers",
    fact="User prefers dark mode",
    valid_at=datetime(2024, 1, 1),      # When it became true
    invalid_at=datetime(2024, 6, 1),    # When superseded (None if current)
    reference_time=datetime.now(),      # Episode timestamp
    episodes=["ep_123"],                # Provenance
)
```

#### 3. Episode-Based Provenance

```python
# Episodes = raw data that produced facts
EpisodicNode(
    content="User mentioned they prefer dark mode for coding",
    source=EpisodeType.message,
    valid_at=datetime(2024, 1, 1),
    entity_edges=["edge_1", "edge_2"],  # Derived facts
)
```

#### 4. Hybrid Retrieval

```python
# Multi-modal retrieval combining:
# 1. Semantic embeddings
# 2. Keyword search (BM25)
# 3. Graph traversal

results = await graphiti.search(
    query="user preferences",
    config=SearchConfig(
        limit=10,
        search_type="hybrid",
        filters=SearchFilters(entity_types=["Preference"])
    )
)
```

### Core Data Models (Adapted from nodes.py and edges.py)

```python
class Node(BaseModel, ABC):
    """Abstract base for all graph nodes"""
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    group_id: str  # Partition key
    labels: list[str] = Field(default_factory=list)
    created_at: datetime = Field(default_factory=utc_now)

class EpisodicNode(Node):
    """Raw episode data (source of truth)"""
    source: EpisodeType  # message, json, text
    source_description: str
    content: str
    valid_at: datetime
    entity_edges: list[str] = []  # Derived edges

class EntityNode(Node):
    """Entity with attributes and summary"""
    name_embedding: list[float] | None
    summary: str  # Regional summary of surrounding edges
    attributes: dict[str, Any] = {}  # Dynamic attributes

class EntityEdge(Edge):
    """Temporal relationship between entities"""
    name: str  # Relationship type
    fact: str
    fact_embedding: list[float] | None
    episodes: list[str] = []  # Source episodes
    expired_at: datetime | None  # When invalidated
    valid_at: datetime | None  # When became true
    invalid_at: datetime | None  # When stopped being true
    reference_time: datetime | None
    attributes: dict[str, Any] = {}
```

### Incremental Graph Construction

```python
async def add_episode(
    self,
    name: str,
    episode_body: str,
    source_description: str,
    reference_time: datetime,
    source: EpisodeType = EpisodeType.message,
    entity_types: dict[str, type[BaseModel]] | None = None,
    excluded_entity_types: list[str] | None = None,
    edge_types: dict[str, type[BaseModel]] | None = None,
) -> AddEpisodeResults:
    """
    1. Extract nodes & edges from episode
    2. Resolve against existing graph
    3. Detect contradictions & invalidate old facts
    4. Update entity summaries
    5. Build episodic edges
    """
```

### Code Patterns for Shadow AI

```python
# Shadow AI can use Graphiti's pattern for temporal facts
class ShadowMemoryGraph:
    """
    Adapt Graphiti's context graph for Shadow AI:
    - Episodes = Shadow observations/interactions
    - Entities = Users, patterns, behaviors
    - Edges = Learned relationships with temporal validity
    """
    
    async def add_observation(
        self,
        observation: str,
        source: str,
        timestamp: datetime,
    ):
        # Extract entities and relationships
        # Resolve against existing graph
        # Invalidate stale facts
        # Update summaries
        pass
    
    async def query_context(
        self,
        query: str,
        temporal_filter: datetime | None = None,
    ) -> list[TemporalFact]:
        # Hybrid retrieval
        # Filter by temporal validity
        # Return with provenance
        pass
```

---

## 4. LaminaAI

**Repository**: https://github.com/LaminaAI  
**Status**: Limited public repository (hypersonic project only)

### Findings

Based on the GitHub organization:
- LaminaAI has minimal public presence with only 1 repository (hypersonic)
- Focus appears to be on memory-first AI architecture
- Limited code available for analysis

### Architectural Concepts (Inferred from Name)

"Lamina" suggests a layered architecture:
- Multi-layer memory systems
- Progressive depth retrieval
- Stratified context handling

---

## 5. MemoryMesh

**Repository**: https://github.com/MemoryMesh  
**Status**: Organization exists but no public repositories

### Findings

- MemoryMesh organization exists on GitHub
- No public repositories available for analysis
- Concept from name suggests dynamic mesh-based memory organization

### Conceptual Pattern (Inferred)

Dynamic memory mesh suggests:
- Interconnected memory clusters
- Dynamic weighting of connections
- Mesh topology for retrieval

---

## Synthesis: Patterns for Shadow AI

### Recommended Architecture

Based on this research, Shadow AI should implement:

```python
# Core memory architecture for Shadow AI

from datetime import datetime
from typing import Any
from uuid import uuid4
from pydantic import BaseModel, Field

# === 1. Block-Based Memory (from Letta) ===

class MemoryBlock(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    label: str  # "user_context", "shadow_patterns", "temporal_facts"
    content: str
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_accessed: datetime = Field(default_factory=datetime.utcnow)
    access_count: int = 0
    
# === 2. Temporal Knowledge Graph (from Graphiti) ===

class TemporalFact(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    content: str  # e.g., "User prefers dark mode"
    source_episode_uuid: str
    valid_at: datetime  # When fact became true
    invalid_at: datetime | None = None  # When superseded (None = current)
    confidence: float = 1.0
    embedding: list[float] | None = None
    
    def is_current(self) -> bool:
        now = datetime.utcnow()
        if self.invalid_at:
            return self.valid_at <= now < self.invalid_at
        return now >= self.valid_at

class ShadowEntity(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    name: str
    type: str  # "user", "pattern", "behavior", "preference"
    summary: str
    attributes: dict[str, Any] = {}
    embedding: list[float] | None = None

class ShadowEdge(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    source_uuid: str
    target_uuid: str
    relationship: str
    fact: TemporalFact
    weight: float = 1.0

# === 3. Episode-Based Provenance (from Zep/Graphiti) ===

class Episode(BaseModel):
    uuid: str = Field(default_factory=lambda: str(uuid4()))
    content: str
    source_type: str  # "observation", "interaction", "external_data"
    source_description: str
    timestamp: datetime
    extracted_entities: list[str] = []
    extracted_edges: list[str] = []

# === 4. Hybrid Retrieval System ===

class MemoryRetriever:
    """
    Combines retrieval strategies from Graphiti:
    - Semantic search (embeddings)
    - Keyword search (BM25/facets)
    - Graph traversal (relationship chains)
    """
    
    async def retrieve(
        self,
        query: str,
        k: int = 10,
        filters: dict | None = None,
        temporal_filter: datetime | None = None,
    ) -> list[TemporalFact]:
        # Step 1: Semantic search on fact embeddings
        # Step 2: Keyword/facet filtering
        # Step 3: Graph traversal for relationship expansion
        # Step 4: Temporal filtering (current facts only)
        pass

# === 5. Memory Manager ===

class ShadowMemory:
    """
    Central memory management combining best patterns:
    - Block-based organization (Letta)
    - Temporal knowledge graph (Graphiti)
    - Episode provenance (Zep)
    """
    
    def __init__(self):
        self.core_blocks: dict[str, MemoryBlock] = {}
        self.archival_graph: ShadowKnowledgeGraph = ShadowKnowledgeGraph()
        self.episodes: list[Episode] = []
        
    async def observe(
        self,
        content: str,
        source: str,
        timestamp: datetime | None = None,
    ):
        """Add new observation with automatic extraction"""
        # Create episode
        episode = Episode(
            content=content,
            source_type="observation",
            source_description=source,
            timestamp=timestamp or datetime.utcnow(),
        )
        
        # Extract entities and relationships (using LLM)
        entities, edges = await self._extract(content)
        
        # Resolve against existing graph
        # Detect contradictions and invalidate old facts
        # Update summaries
        # Store episode with provenance
        
        self.episodes.append(episode)
        await self.archival_graph.integrate(entities, edges, episode)
        
    async def query(
        self,
        query: str,
        include_historical: bool = False,
    ) -> list[TemporalFact]:
        """Query memory with temporal awareness"""
        retriever = MemoryRetriever()
        results = await retriever.retrieve(
            query=query,
            temporal_filter=None if include_historical else datetime.utcnow()
        )
        return results
```

### Specific Improvements for Shadow AI

Based on this research, implement these enhancements:

1. **Temporal Validity Tracking**
   - Track when facts became true and when they expired
   - Auto-invalidate contradictory beliefs
   - Support historical queries

2. **Episode-Based Provenance**
   - Every derived fact traces back to source
   - Enables trust scoring and verification
   - Supports fact correction/deletion cascades

3. **Hybrid Retrieval**
   - Combine semantic, keyword, and graph traversal
   - Support temporal filtering
   - Rank by confidence and recency

4. **Entity Summaries**
   - Maintain evolving summaries of entities
   - Regional summaries for context neighborhoods
   - Update summaries as new facts arrive

5. **Community Detection**
   - Cluster related entities
   - Enable efficient neighborhood retrieval
   - Support hierarchical summarization

---

## Implementation Priority

| Feature | Priority | Source | Complexity |
|---------|----------|--------|------------|
| Memory blocks | HIGH | Letta | Low |
| Temporal facts | HIGH | Graphiti | Medium |
| Episode provenance | MEDIUM | Zep | Medium |
| Entity summaries | MEDIUM | Graphiti | Medium |
| Hybrid retrieval | LOW | Graphiti | High |
| Community detection | LOW | Graphiti | High |

---

## Key Files to Study

### From Letta
- Memory block organization patterns
- Core vs archival memory distinction
- Self-improvement memory management

### From Graphiti (Primary Source)
- `graphiti_core/nodes.py` - Entity and Episode models
- `graphiti_core/edges.py` - Temporal edge models
- `graphiti_core/graphiti.py` - Episode ingestion and retrieval
- `graphiti_core/search/` - Hybrid retrieval implementation

### From Zep
- Context assembly patterns
- Multi-source integration
- Temporal knowledge management

---

## Conclusion

The most relevant patterns for Shadow AI come from Graphiti/Zep's temporal knowledge graph approach. The key innovations are:

1. **Episodes as ground truth** - All facts derive from verifiable sources
2. **Temporal validity windows** - Facts know when they're true
3. **Automatic contradiction handling** - New facts invalidate old ones
4. **Hybrid retrieval** - Multiple retrieval strategies combined
5. **Entity summaries** - Evolving understanding of entities

Shadow AI should implement these patterns to create a memory system that learns, adapts, and maintains accurate context over time.