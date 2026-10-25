# GitHub Research Hour 2: Memory & RAG Systems

**Date**: April 4, 2026  
**Duration**: 60 minutes  
**Focus**: Systematic research of TOP Memory & RAG repositories on GitHub

## Executive Summary

This report documents comprehensive research on the leading Memory and Retrieval-Augmented Generation (RAG) systems. These systems form the backbone of modern AI applications that require persistent memory, semantic search, and context management.

---

## 1. ChromaDB (27.1k stars)

**Repository**: https://github.com/chroma-core/chroma  
**License**: Apache 2.0  
**Language**: Rust (66.5%) + Python (17.1%) + TypeScript (7.6%)

### Architecture Overview

ChromaDB is an open-source embedding database designed for simplicity and developer experience. It provides both embedded and server modes with a minimal API surface.

```python
import chromadb

# In-memory mode (prototyping)
client = chromadb.Client()

# Persistent mode (production)
client = chromadb.PersistentClient(path="/path/to/chroma_db")

# Create collection with automatic embedding
collection = client.create_collection(
    name="documents",
    metadata={"description": "My document collection"}
)

# Add documents with automatic embedding
collection.add(
    documents=["Document 1 text", "Document 2 text"],
    metadatas=[{"source": "web"}, {"source": "pdf"}],
    ids=["doc1", "doc2"]
)

# Query with filtering
results = collection.query(
    query_texts=["search query"],
    n_results=5,
    where={"source": {"$eq": "web"}},  # Metadata filtering
    where_document={"$contains": "keyword"}  # Content filtering
)
```

### Key Features

1. **Multi-Modal Embeddings**: Supports text, images, and audio
2. **Hybrid Search**: Dense vectors + sparse vectors (BM25-like)
3. **Metadata Filtering**: Rich filtering capabilities at query time
4. **Embedding Functions**: Built-in support for OpenAI, Cohere, HuggingFace, etc.
5. **Serverless Option**: Chroma Cloud for managed hosting

### Vector Storage Architecture

- **HNSW Index**: Hierarchical Navigable Small World graphs for ANN search
- **SQLite Backend**: Default persistent storage
- **Segmented Storage**: Separates vectors, metadata, and full-text content
- **Embedding Cache**: Caches embeddings to avoid recomputation

### Document Chunking Patterns

```python
from chromadb.utils import embedding_functions

# Built-in chunking strategies
from chromadb.utils.chunking import (
    RecursiveCharacterTextSplitter,
    MarkdownTextSplitter,
    PythonCodeTextSplitter
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
    length_function=len,
    separators=["\n\n", "\n", " ", ""]
)

chunks = splitter.split_text(document_text)
```

### Scaling Approach

- **Multi-tenant**: Per-tenant collections
- **Horizontal Scaling**: Chroma Cloud handles distribution
- **Vector Quantization**: Reduces memory footprint
- **Lazy Loading**: Streams large collections

---

## 2. Weaviate (15.9k stars)

**Repository**: https://github.com/weaviate/weaviate  
**License**: BSD-3-Clause  
**Language**: Go (96.9%)

### Architecture Overview

Weaviate is a cloud-native vector database that stores both objects and vectors, enabling semantic search with structured filtering.

```go
// Core concepts: Collections, Objects, Vectors
// Weaviate uses GraphQL for queries

// Near text search example
{
  Get {
    Article(nearText: {
      concepts: ["AI safety"],
      limit: 10
    }) {
      title
      content
      _additional {
        certainty
        distance
      }
    }
  }
}
```

### Key Features

1. **Hybrid Search**: Combines BM25 keyword search with vector similarity
2. **Modules System**: Pluggable vectorizers, rerankers, generative modules
3. **Multi-Tenancy**: Native multi-tenant isolation
4. **Replication & Sharding**: Built-in horizontal scaling
5. **gRPC API**: High-performance batch operations
6. **RAG Built-in**: Generative search modules for RAG workflows

### Hybrid Search Implementation

```python
from weaviate.classes.config import Configure
from weaviate.classes.query import Hybrid

# Hybrid search combining vector + keyword
results = collection.query.hybrid(
    query="machine learning algorithms",
    alpha=0.5,  # 0.5 = balanced vector/keyword
    limit=10,
    return_metadata=["score", "explainScore"]
)
```

### Vector Indexing

- **HNSW**: Default ANN algorithm
- **Flat Index**: Exact k-NN for small datasets
- **Dynamic Indexing**: Add vectors without rebuilding
- **Vector Compression**: Product quantization

### Persistence Mechanisms

1. **Write-Ahead Log**: Durability guarantees
2. **Snapshotting**: Point-in-time backups
3. **Tiered Storage**: Hot/warm/cold data separation
4. **Async I/O**: Uses `io_uring` for Linux optimization

---

## 3. Pinecone (various SDKs)

**Repository**: https://github.com/pinecone-io  
**License**: Apache 2.0 (Client SDKs)  
**Type**: Cloud-managed vector database

### Architecture Overview

Pinecone is a fully managed vector database service with serverless and pod-based deployment options.

```python
from pinecone import Pinecone, ServerlessSpec

pc = Pinecone(api_key="YOUR_API_KEY")

# Serverless index (auto-scaling)
pc.create_index(
    name="my-index",
    dimension=1536,
    metric="cosine",
    spec=ServerlessSpec(
        cloud="aws",
        region="us-east-1"
    )
)

# Pod-based index (dedicated resources)
from pinecone import PodSpec
pc.create_index(
    name="my-pod-index",
    dimension=1536,
    metric="dotproduct",
    spec=PodSpec(
        environment="production",
        pod_type="p1.x1",
        pods=1
    )
)
```

### Key Features

1. **Serverless**: Zero infrastructure, pay-per-query
2. **Integrated Inference**: Built-in embedding and reranking
3. **Metadata Filtering**: Query-time filtering without performance penalty
4. **Namespace Isolation**: Logical separation within indexes
5. **Streaming Ingestion**: Real-time vector updates

### Query Patterns

```python
index = pc.Index("my-index")

# Upsert with metadata
index.upsert([
    ("id1", [0.1, 0.2, ...], {"category": "A", "price": 100}),
    ("id2", [0.2, 0.3, ...], {"category": "B", "price": 200})
])

# Query with filtering
results = index.query(
    vector=query_vector,
    top_k=10,
    filter={
        "category": {"$eq": "A"},
        "price": {"$lte": 150}
    },
    include_metadata=True
)
```

### Scaling Approach

- **Pod Autoscaling**: Automatic pod count adjustments
- **Collection Limits**: 1M+ vectors per pod
- **Replicas**: Multiple copies for fault tolerance
- **Integrated Reranking**: Secondary re-scoring of results

---

## 4. Qdrant (30k stars)

**Repository**: https://github.com/qdrant/qdrant  
**License**: Apache 2.0  
**Language**: Rust (87.4%)

### Architecture Overview

Qdrant is a high-performance vector database written in Rust, optimized for filtering and large-scale deployments.

```python
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct

client = QdrantClient(":memory:")  # In-memory for testing
# Or: QdrantClient(path="path/to/db")  # Persistent

# Create collection
client.create_collection(
    collection_name="documents",
    vectors_config=VectorParams(
        size=1536,
        distance=Distance.COSINE
    )
)

# Insert points with payload
client.upsert(
    collection_name="documents",
    points=[
        PointStruct(
            id=1,
            vector=[0.1, 0.2, ...],
            payload={"title": "Document 1", "tags": ["ai", "ml"]}
        )
    ]
)
```

### Key Features

1. **Payload Filtering**: Rich filtering with `must`, `should`, `must_not`
2. **Sparse Vectors**: Native BM25/sparse embedding support
3. **Quantization**: Multiple compression strategies
4. **gRPC Support**: Faster than REST for bulk operations
5. **On-Disk Storage**: Memory-mapped vectors
6. **Distributed Mode**: Sharding and replication

### Filtering Architecture

```python
from qdrant_client.models import Filter, FieldCondition, MatchValue

# Complex filtering
results = client.query(
    collection_name="documents",
    query_vector=[...],
    filter=Filter(
        must=[
            FieldCondition(
                key="category",
                match=MatchValue(value="AI")
            )
        ],
        should=[
            FieldCondition(key="tags", match=MatchValue(value="tutorial"))
        ],
        must_not=[
            FieldCondition(key="archived", match=MatchValue(value=True))
        ]
    ),
    limit=10
)
```

### Performance Optimizations

- **SIMD Acceleration**: Hardware-accelerated distance computation
- **HNSW Optimization**: Custom HNSW implementation
- **Async I/O**: Non-blocking disk operations
- **Query Planning**: Optimizes filter execution order
- **Quantization**: Product, binary, scalar quantization

---

## 5. Milvus (43.6k stars)

**Repository**: https://github.com/milvus-io/milvus  
**License**: Apache 2.0  
**Language**: Go + C++

### Architecture Overview

Milvus is a cloud-native, distributed vector database designed for massive scale deployments (billions of vectors).

```python
from pymilvus import MilvusClient

# Connect to Milvus
client = MilvusClient("milvus_demo.db")

# Create collection
client.create_collection(
    collection_name="demo",
    dimension=768  # Vector dimension
)

# Insert data
data = [
    {"id": 1, "vector": [0.1, 0.2, ...], "text": "Document 1"},
    {"id": 2, "vector": [0.3, 0.4, ...], "text": "Document 2"}
]
client.insert("demo", data)

# Search
results = client.search(
    collection_name="demo",
    data=[query_vector],
    limit=3,
    output_fields=["text", "id"]
)
```

### Key Features

1. **Distributed Architecture**: Separates compute and storage
2. **Multi-Vector Search**: Dense, sparse, and hybrid
3. **GPU Indexing**: NVIDIA CAGRA support
4. **Scalar Indexing**: Metadata filtering with indexes
5. **Hot/Cold Storage**: Tiered storage management
6. **Multi-Tenancy**: Database/collection/partition isolation

### Distributed Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    Milvus Cluster                        │
├─────────────────────────────────────────────────────────┤
│  ┌─────────┐  ┌─────────┐  ┌─────────────────────────┐  │
│  │  Proxy  │  │  API    │  │    Query Coordinator   │  │
│  └─────────┘  └─────────┘  └─────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Query Nodes (Stateless)               │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │              Data Nodes (Stateful)                │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │      Storage (MinIO/S3 + etcd + Pulsar/Kafka)    │  │
│  └──────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

### Index Types

1. **HNSW**: High recall, fast search
2. **IVF_FLAT**: Balanced speed/recall
3. **IVF_PQ**: Memory-efficient
4. **DiskANN**: On-disk storage
5. **GPU Indexes**: CAGRA for GPU acceleration
6. **Sparse Indexes**: For BM25/sparse embeddings

---

## 6. Letta (MemGPT) (21.9k stars)

**Repository**: https://github.com/letta-ai/letta  
**License**: Apache 2.0  
**Language**: Python (99.5%)

### Architecture Overview

Letta (formerly MemGPT) is a platform for building stateful AI agents with advanced memory management.

```python
from letta_client import Letta

client = Letta(api_key=os.getenv("LETTA_API_KEY"))

# Create agent with memory blocks
agent = client.agents.create(
    model="openai/gpt-5.2",
    memory_blocks=[
        {
            "label": "human",
            "value": "User preferences and facts"
        },
        {
            "label": "persona",
            "value": "Agent personality and behavior"
        }
    ],
    tools=["web_search", "fetch_webpage"]
)

# Send message (agent manages memory automatically)
response = client.agents.messages.create(
    agent_id=agent.id,
    input="What do you know about me?"
)
```

### Key Features

1. **Memory Blocks**: Structured, labeled memory segments
2. **Self-Improving Memory**: Agents update their own context
3. **Multi-Block Memory**: Separate user/persona/system blocks
4. **Tool Integration**: Built-in and custom tool support
5. **CLI Agent**: Letta Code for terminal-based agents
6. **Model Agnostic**: Works with any LLM API

### Memory Architecture

```
┌─────────────────────────────────────────────────────────┐
│                   Agent Memory System                    │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐ │
│  │ Human Block  │  │Persona Block │  │ System Block │ │
│  │              │  │              │  │              │ │
│  │ User facts   │  │ Agent        │  │ Operating    │ │
│  │ Preferences  │  │ personality  │  │ instructions │ │
│  │ History      │  │ Behavior     │  │              │ │
│  └──────────────┘  └──────────────┘  └──────────────┘ │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │            Archival Memory (Long-term)            │  │
│  │   - Persistent knowledge store                     │  │
│  │   - RAG retrieval for context                      │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
│  ┌──────────────────────────────────────────────────┐  │
│  │           Recall Memory (Recent context)          │  │
│  │   - Recent messages                               │  │
│  │   - Working context window                        │  │
│  └──────────────────────────────────────────────────┘  │
│                                                          │
└─────────────────────────────────────────────────────────┘
```

### Memory Management Strategies

1. **Core Memory Blocks**: Always in context
2. **Archival Memory**: Retrieved via RAG when needed
3. **Recall Memory**: Recent conversation history
4. **Self-Editing**: Agents call functions to update memory
5. **Context Management**: Automatic summarization and compression

---

## 7. Zep (4.4k stars)

**Repository**: https://github.com/getzep/zep  
**License**: Apache 2.0  
**Language**: Python + Go

### Architecture Overview

Zep is an end-to-end context engineering platform that delivers relationship-aware context from multiple data sources.

```python
from zep_cloud import ZepClient

client = ZepClient(api_key="YOUR_API_KEY")

# Add messages to a session
client.memory.add_session_memory(
    session_id="user-123",
    messages=[
        {"role": "user", "content": "I love Italian food"},
        {"role": "assistant", "content": "I'll remember that!"}
    ]
)

# Get assembled context for LLM
context = client.memory.get_session_memory(
    session_id="user-123",
    lastn=10  # Last N messages
)
```

### Key Features

1. **Temporal Knowledge Graphs**: Facts with validity windows
2. **Graph RAG**: Relationship-aware retrieval
3. **Context Assembly**: Pre-formatted context blocks
4. **Sub-200ms Latency**: Production-ready performance
5. **Multi-Source**: Chat, business data, documents, events

### Temporal Knowledge Graph

```python
# Zep automatically extracts and manages temporal facts

# Example knowledge graph evolution:
# Episode 1: "User likes Italian food"
#   → Fact: User.likes(Italian food, valid_from: 2026-01-01)
#
# Episode 2: "User mentions they're now vegan"
#   → Fact invalidated: User.likes(Italian food, invalid_at: 2026-03-15)
#   → New Fact: User.dietary_preference(vegan, valid_from: 2026-03-15)
```

### Context Assembly

```python
# Zep assembles context as relationship-aware blocks

context_blocks = client.context.assemble(
    session_id="user-123",
    strategies=["chat_history", "knowledge_graph", "user_facts"]
)

# Returns formatted blocks:
# ┌─────────────────────────────────────────────────────────┐
# │ USER FACTS:                                              │
# │ - User prefers Italian cuisine (as of Jan 2026)         │
# │ - User is vegan (as of March 2026)                      │
# └─────────────────────────────────────────────────────────┘
# ┌─────────────────────────────────────────────────────────┐
# │ RECENT CONVERSATION:                                     │
# │ [Last 10 messages formatted for LLM]                     │
# └─────────────────────────────────────────────────────────┘
```

---

## 8. RAGAS (13.2k stars)

**Repository**: https://github.com/explodinggradients/ragas  
**License**: Apache 2.0  
**Language**: Python (83%)

### Architecture Overview

RAGAS is a framework for evaluating RAG systems with objective metrics and test generation.

```python
import asyncio
from ragas.metrics import DiscreteMetric
from ragas.llms import llm_factory

# Create custom evaluation metric
metric = DiscreteMetric(
    name="summary_accuracy",
    allowed_values=["accurate", "inaccurate"],
    prompt="""Evaluate if the summary is accurate.
    Response: {response}
    Answer with only 'accurate' or 'inaccurate'."""
)

# Score application output
async def evaluate():
    score = await metric.ascore(
        llm=llm,
        response="The summary of the text is..."
    )
    print(f"Score: {score.value}")
    print(f"Reason: {score.reason}")
```

### Key Features

1. **Objective Metrics**: LLM-based and traditional evaluation
2. **Test Data Generation**: Automatic test dataset creation
3. **Framework Integrations**: LangChain, LlamaIndex support
4. **Production Alignment**: Tests based on real user queries
5. **Feedback Loops**: Use production data to improve

### Core Metrics

```python
from ragas.metrics import (
    Faithfulness,              # Is answer supported by context?
    Answer Relevancy,          # Is answer relevant to question?
    Context Precision,         # Are retrieved docs relevant?
    Context Recall,            # Did we retrieve all needed docs?
    Context Entity Recall,     # Entity-level recall
    Answer Similarity,         # Semantic similarity to ground truth
    Answer Correctness         # Correctness scoring
)

# Evaluate RAG pipeline
from ragas import evaluate
from datasets import Dataset

results = evaluate(
    dataset=Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truths": ground_truths
    }),
    metrics=[
        Faithfulness(),
        AnswerRelevancy(),
        ContextPrecision()
    ]
)
```

### Evaluation Patterns

```
┌─────────────────────────────────────────────────────────┐
│                RAGAS Evaluation Pipeline                 │
├─────────────────────────────────────────────────────────┤
│                                                          │
│  ┌──────────────┐   ┌──────────────┐   ┌──────────────┐│
│  │ Test         │   │ Run         │   │ Evaluate     │ │
│  │ Generation   │──▶│ RAG Pipeline│──▶│ Metrics      │ │
│  │              │   │              │   │              │ │
│  └──────────────┘   └──────────────┘   └──────────────┘│
│         │                                      │        │
│         │                                      │        │
│         ▼                                      ▼        │
│  ┌──────────────┐                       ┌──────────────┐│
│  │ Synthetic    │                       │ Feedback     ││
│  │ Questions    │                       │ Loop         ││
│  └──────────────┘                       └──────────────┘│
│                                                          │
└─────────────────────────────────────────────────────────┘
```

---

## 9. Graphiti (24.5k stars)

**Repository**: https://github.com/getzep/graphiti  
**License**: Apache 2.0  
**Language**: Python (99.3%)

### Architecture Overview

Graphiti is a framework for building temporal context graphs - knowledge graphs that track state changes over time.

```python
from graphiti_core import Graphiti
from graphiti_core.llm_client.openai_client import OpenAIClient

# Initialize Graphiti
graphiti = Graphiti(
    uri="bolt://localhost:7687",
    user="neo4j",
    password="password",
    llm_client=OpenAIClient()
)

# Add episode (raw data)
await graphiti.add_episode(
    name="conversation_1",
    source_description="User said they prefer Italian food",
    source_content="Full conversation text..."
)

# Search with hybrid retrieval
results = await graphiti.search(
    query="What food does the user like?",
    num_results=10
)
```

### Key Features

1. **Temporal Facts**: Facts with `valid_at` and `invalid_at`
2. **Episode Provenance**: Full lineage to source data
3. **Ontology Support**: Prescribed and learned entities
4. **Incremental Updates**: No batch recomputation
5. **Hybrid Retrieval**: Semantic + keyword + graph traversal

### Context Graph Structure

```python
# Graphiti structures data as:

# Entities (Nodes)
# - Person, Product, Concept, etc.
# - Each has evolving summary
# - Type can be prescribed or learned

# Relationships (Edges)
# - Entity -> Relation -> Entity
# - Temporal validity window
# - Provenance to episode

# Episodes
# - Raw data as ingested
# - Ground truth stream
# - Source of all graph updates

# Example evolution:
# Episode 1: "Alice works at Google"
#   → Entity: Alice (type: Person)
#   → Fact: Alice.works_at(Google, valid_from: t1)
#
# Episode 2: "Alice left Google for Microsoft"
#   → Fact invalidated: Alice.works_at(Google, invalid_at: t2)
#   → New Fact: Alice.works_at(Microsoft, valid_from: t2)
```

### Hybrid Search

```python
# Graphiti combines three retrieval methods:

results = await graphiti.search(
    query="user preferences",
    search_methods=[
        "semantic",      # Vector similarity
        "keyword",       # BM25 full-text
        "graph"          # Graph traversal
    ],
    rerank=True,       # Optional cross-encoder reranking
    limit=20
)
```

---

## Comparative Analysis

### Vector Database Comparison

| Feature | ChromaDB | Weaviate | Pinecone | Qdrant | Milvus |
|---------|----------|----------|----------|--------|--------|
| **Self-Hosted** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Cloud Managed** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Language** | Rust/Python | Go | N/A | Rust | Go/C++ |
| **Open Source** | ✅ | ✅ | ❌ | ✅ | ✅ |
| **Hybrid Search** | ✅ | ✅ | ✅ | ✅ | ✅ |
| **Distributed** | Cloud only | ✅ | ✅ | ✅ | ✅ |
| **GPU Index** | ❌ | ❌ | ❌ | ❌ | ✅ |
| **Max Scale** | Medium | Large | Very Large | Large | Very Large |

### Memory System Comparison

| Feature | Letta | Zep | Graphiti |
|---------|-------|-----|-----------|
| **Temporal Awareness** | ✅ | ✅ | ✅ |
| **Knowledge Graph** | ❌ | ✅ | ✅ |
| **Self-Editing Memory** | ✅ | ❌ | ❌ |
| **Context Assembly** | Basic | Advanced | Advanced |
| **Open Source** | ✅ | Cloud/Core | ✅ |
| **Latency Target** | 500ms | <200ms | <500ms |

---

## Code Patterns for Shadow AI Memory System

### 1. Document Chunking

```python
# Recommended: Sentence-based with overlap for semantic coherence
from langchain.text_splitter import RecursiveCharacterTextSplitter

def chunk_documents(documents: list[str], chunk_size: int = 1000, overlap: int = 200):
    """Chunk documents for vector storage with semantic coherence."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=overlap,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    
    chunks = []
    for doc in documents:
        doc_chunks = splitter.split_text(doc)
        for i, chunk in enumerate(doc_chunks):
            chunks.append({
                "content": chunk,
                "metadata": {
                    "chunk_index": i,
                    "chunk_hash": hash(chunk),
                    "source_length": len(doc)
                }
            })
    return chunks
```

### 2. Embedding Generation with Caching

```python
from typing import Optional
import hashlib
from functools import lru_cache

class EmbeddingCache:
    """Cache embeddings to avoid recomputation."""
    
    def __init__(self, vector_db, embedding_model):
        self.db = vector_db
        self.model = embedding_model
        self._cache = {}
    
    def get_or_compute(self, text: str) -> list[float]:
        """Return cached embedding or compute and cache."""
        text_hash = hashlib.md5(text.encode()).hexdigest()
        
        if text_hash in self._cache:
            return self._cache[text_hash]
        
        # Check persistence
        cached = self.db.get_embedding(text_hash)
        if cached:
            self._cache[text_hash] = cached
            return cached
        
        # Compute and cache
        embedding = self.model.embed(text)
        self._cache[text_hash] = embedding
        self.db.store_embedding(text_hash, embedding)
        return embedding
```

### 3. Hybrid Search Implementation

```python
from typing import TypedDict
import numpy as np

class SearchResult(TypedDict):
    id: str
    score: float
    content: str
    metadata: dict

def hybrid_search(
    query: str,
    vector_store,
    alpha: float = 0.5,
    limit: int = 10
) -> list[SearchResult]:
    """
    Combine vector and keyword search.
    
    Args:
        query: Search query
        vector_store: Vector database instance
        alpha: Weight for vector vs keyword (0.5 = balanced)
        limit: Number of results
    
    Returns:
        Combined and ranked results
    """
    # Get vector results
    query_embedding = get_embedding(query)
    vector_results = vector_store.query(
        vector=query_embedding,
        limit=limit * 2  # Fetch more for re-ranking
    )
    
    # Get keyword results
    keyword_results = vector_store.query(
        text=query,
        limit=limit * 2
    )
    
    # Combine scores
    combined = {}
    for r in vector_results:
        combined[r['id']] = {'score': alpha * r['score']}
    for r in keyword_results:
        if r['id'] in combined:
            combined[r['id']]['score'] += (1 - alpha) * r['score']
        else:
            combined[r['id']] = {'score': (1 - alpha) * r['score']}
    
    # Sort and return
    results = [
        {'id': k, 'score': v['score']}
        for k, v in combined.items()
    ]
    results.sort(key=lambda x: x['score'], reverse=True)
    
    return results[:limit]
```

### 4. Memory Summarization

```python
from datetime import datetime
from typing import Protocol

class MemoryStore(Protocol):
    """Abstract memory store interface."""
    
    def add_memory(self, key: str, value: str, metadata: dict) -> None: ...
    def get_memory(self, key: str) -> Optional[str]: ...
    def search(self, query: str, limit: int) -> list: ...

class TemporalMemoryManager:
    """Manage memories with temporal awareness."""
    
    def __init__(self, store: MemoryStore, llm_client):
        self.store = store
        self.llm = llm_client
    
    async def add_memory(
        self,
        content: str,
        user_id: str,
        valid_from: datetime = None
    ):
        """Add memory with temporal validity."""
        # Extract entities and relationships
        entities = await self._extract_entities(content)
        
        # Store with metadata
        for entity in entities:
            self.store.add_memory(
                key=f"{user_id}:{entity['key']}",
                value=entity['value'],
                metadata={
                    'valid_from': valid_from or datetime.utcnow(),
                    'valid_until': None,
                    'source': content,
                    'user_id': user_id
                }
            )
    
    async def get_context(self, query: str, user_id: str) -> str:
        """Retrieve relevant context for query."""
        # Search memories
        results = self.store.search(
            query=f"{user_id}:{query}",
            limit=10
        )
        
        # Filter by temporal validity
        valid_results = [
            r for r in results
            if self._is_valid(r)
        ]
        
        # Format for LLM
        context = self._format_context(valid_results)
        return context
    
    def _is_valid(self, memory: dict) -> bool:
        """Check if memory is temporally valid."""
        now = datetime.utcnow()
        valid_from = memory['metadata'].get('valid_from')
        valid_until = memory['metadata'].get('valid_until')
        
        if valid_from and valid_from > now:
            return False
        if valid_until and valid_until <= now:
            return False
        return True
```

### 5. RAG Pipeline for Shadow AI

```python
from dataclasses import dataclass
from typing import Optional

@dataclass
class RAGContext:
    query: str
    documents: list[str]
    context: str
    sources: list[str]

class ShadowAIRAG:
    """RAG pipeline for Shadow AI memory retrieval."""
    
    def __init__(
        self,
        vector_store,
        llm_client,
        embedder,
        reranker=None
    ):
        self.vector_store = vector_store
        self.llm = llm_client
        self.embedder = embedder
        self.reranker = reranker
    
    async def retrieve(
        self,
        query: str,
        user_id: str,
        filters: Optional[dict] = None,
        top_k: int = 10
    ) -> list[str]:
        """Retrieve relevant documents from memory."""
        
        # Generate query embedding
        query_embedding = await self.embedder.embed(query)
        
        # Hybrid search
        results = await self.vector_store.hybrid_search(
            vector=query_embedding,
            text=query,
            alpha=0.6,  # Favor semantic
            filter={
                'user_id': user_id,
                **(filters or {})
            },
            limit=top_k * 2  # For reranking
        )
        
        # Optional reranking
        if self.reranker:
            results = await self.reranker.rerank(
                query=query,
                documents=results,
                top_k=top_k
            )
        
        return results[:top_k]
    
    async def generate_context(
        self,
        query: str,
        user_id: str,
        max_tokens: int = 4000
    ) -> RAGContext:
        """Generate complete context for LLM."""
        
        # Retrieve documents
        docs = await self.retrieve(query, user_id)
        
        # Extract context
        context_parts = []
        total_tokens = 0
        sources = []
        
        for doc in docs:
            tokens = self._count_tokens(doc['content'])
            if total_tokens + tokens > max_tokens:
                break
            
            context_parts.append(doc['content'])
            sources.append(doc.get('source', 'unknown'))
            total_tokens += tokens
        
        # Compile context
        context = "\n\n---\n\n".join(context_parts)
        
        return RAGContext(
            query=query,
            documents=[d['content'] for d in docs[:len(context_parts)]],
            context=context,
            sources=sources
        )
    
    def _count_tokens(self, text: str) -> int:
        """Estimate token count."""
        # Simple estimation: ~4 characters per token
        return len(text) // 4
```

---

## Key Learnings for Shadow AI

### 1. **Vector Database Selection Criteria**

For Shadow AI's memory system:

| Requirement | Recommended | Reason |
|-------------|------------|--------|
| **Self-hosted** | Qdrant or ChromaDB | Full control, no external API |
| **Large scale** | Milvus | Proven at billion+ vectors |
| **Developer experience** | ChromaDB | Simplest API |
| **Performance critical** | Qdrant | Rust-based, optimized filters |
| **Hybrid search** | Weaviate or Qdrant | Native BM25 + vector |

### 2. **Memory Architecture Patterns**

```
┌──────────────────────────────────────────────────────────┐
│              Shadow AI Memory Architecture               │
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │              Working Memory (Hot)                   │  │
│  │  - Current conversation context                     │  │
│  │  - Active user preferences                          │  │
│  │  - Task-relevant information                        │  │
│  │  Storage: In-memory with TTL                       │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                │
│                         ▼                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │           Episodic Memory (Warm)                   │  │
│  │  - Recent conversations                             │  │
│  │  - Session-specific facts                          │  │
│  │  - Temporal knowledge graph                        │  │
│  │  Storage: Vector DB with timestamps                │  │
│  └────────────────────────────────────────────────────┘  │
│                         │                                │
│                         ▼                                │
│  ┌────────────────────────────────────────────────────┐  │
│  │           Long-term Memory (Cold)                   │  │
│  │  - Persistent user facts                            │  │
│  │  - Learned patterns                                │  │
│  │  - Historical context                              │  │
│  │  Storage: Knowledge graph + vector DB              │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
└──────────────────────────────────────────────────────────┘
```

### 3. **Critical Implementation Patterns**

#### Temporal Fact Management
```python
# Track fact validity over time
fact = {
    "subject": "user",
    "predicate": "prefers",
    "object": "Italian cuisine",
    "valid_from": "2026-01-15T10:00:00Z",
    "valid_until": None,  # Still valid
    "confidence": 0.95,
    "source": "conversation:abc123"
}

# When user changes preference
new_fact = {
    "subject": "user",
    "predicate": "prefers",
    "object": "Japanese cuisine",
    "valid_from": "2026-03-20T14:30:00Z",
    "valid_until": None
}

# Invalidate old fact
old_fact["valid_until"] = "2026-03-20T14:30:00Z"
```

#### Context Window Management
```python
# Prioritize context for limited window
def prioritize_context(
    memories: list,
    max_tokens: int,
    query: str
) -> list:
    """Score and rank memories for context window."""
    
    scored = []
    for mem in memories:
        # Recency score
        recency = 1 / (1 + hours_ago(mem['timestamp']))
        
        # Relevance score
        relevance = semantic_similarity(query, mem['content'])
        
        # Importance score (user-defined or learned)
        importance = mem.get('importance', 0.5)
        
        # Combined score
        score = 0.3 * recency + 0.5 * relevance + 0.2 * importance
        scored.append((mem, score))
    
    # Sort by score, fill to max_tokens
    scored.sort(key=lambda x: x[1], reverse=True)
    selected = []
    total = 0
    
    for mem, score in scored:
        tokens = count_tokens(mem['content'])
        if total + tokens <= max_tokens:
            selected.append(mem)
            total += tokens
    
    return selected
```

### 4. **Evaluation Metrics**

```python
# Key metrics for Shadow AI memory system

# Retrieval Quality
retrieval_precision = "relevant_docs_retrieved / total_docs_retrieved"
retrieval_recall = "relevant_docs_retrieved / total_relevant_docs"
ndcg = "ranking_quality_measure"

# Memory Quality
fact_accuracy = "correct_facts_stored / total_facts_stored"
temporal_consistency = "facts_correctly_invalidated / facts_to_invalidate"
entity_resolution = "correctly_linked_entities / total_entities"

# Context Quality
context_relevance = "context_useful_for_query / total_context"
context_completeness = "required_info_in_context / total_required_info"
context_conciseness = "necessary_info / total_context_size"

# Performance
latency_p99 = "99th_percentile_response_time"
throughput = "queries_per_second"
memory_growth = "storage_increase_per_conversation"
```

---

## Recommendations for Shadow AI

### 1. **Memory System Stack**

```
Primary Vector Store: Qdrant (self-hosted, high performance, Rust-based)
├── Hybrid search for semantic + keyword
├── Rich filtering for temporal queries
├── On-disk storage for large memories
└── GPU indexing (optional) for speed

Knowledge Graph: Graphiti (temporal graph for relationships)
├── Track entity relationships over time
├── Handle contradictions gracefully
├── Support both prescribed and learned ontology
└── Hybrid retrieval (semantic + graph + keyword)

Memory Management: Custom (inspired by Letta/Zep)
├── Working memory: In-memory cache with TTL
├── Episodic memory: Qdrant with timestamp filtering
├── Long-term memory: Graphiti + Qdrant
└── Temporal awareness throughout
```

### 2. **Implementation Priority**

1. **Phase 1**: Basic vector search with Qdrant
   - Document chunking with semantic boundaries
   - Hybrid search (alpha: 0.6 vector, 0.4 keyword)
   - Metadata filtering for user isolation

2. **Phase 2**: Temporal memory layer
   - Add timestamp and validity tracking
   - Implement fact invalidation workflow
   - Track memory provenance

3. **Phase 3**: Knowledge graph integration
   - Extract entities and relationships with Graphiti
   - Build temporal knowledge graph
   - Hybrid retrieval across graph + vectors

4. **Phase 4**: Advanced features
   - Memory summarization pipeline
   - Self-editing memory capabilities
   - Context window optimization

### 3. **Key Design Decisions**

| Decision | Recommendation | Rationale |
|----------|---------------|-----------|
| **Vector dimensions** | 1536 (OpenAI) or 768 (Open source) | Balance quality vs storage |
| **Chunk size** | 512-1024 tokens with 20% overlap | Semantic coherence |
| **Embedding model** | text-embedding-3-small or all-MiniLM | Cost vs quality tradeoff |
| **Index type** | HNSW | Best recall-speed balance |
| **Persistence** | Qdrant with on-disk storage | Memory efficiency |
| **Filtering** | Query-time with indexed metadata | Performance critical |
| **Temporal queries** | Graphiti + timestamp in metadata | Best of both worlds |

---

## Conclusion

This research provides a comprehensive overview of the leading Memory and RAG systems available in 2026. Key takeaways:

1. **Vector Databases**: Qdrant and Milvus lead for self-hosted solutions; Pinecone for managed cloud
2. **Memory Systems**: Letta excels at self-editing memory; Zep/Graphiti for temporal knowledge graphs
3. **Architecture Patterns**: Hybrid search + temporal awareness are essential for production systems
4. **Evaluation**: RAGAS provides objective metrics for continuous improvement

For Shadow AI's memory system, we recommend a hybrid approach combining Qdrant for vector retrieval, Graphiti for temporal knowledge graphs, and custom memory management inspired by Letta's block-based architecture.

---

**Research Duration**: 60 minutes  
**Repositories Studied**: 9  
**Total Stars Analyzed**: 180,000+  
**Last Updated**: April 4, 2026