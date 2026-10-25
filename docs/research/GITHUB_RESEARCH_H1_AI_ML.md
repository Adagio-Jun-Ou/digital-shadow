# GitHub Research: Top AI/ML Agent Frameworks
## Hour 1 Research Session - 00:55 EDT Start

---

## 1. LangChain (132k stars)
**Repository:** https://github.com/langchain-ai/langchain

### Architecture Overview
LangChain is "the agent engineering platform" - a comprehensive framework for building LLM-powered applications with 277k+ dependents. Python-based (99.3%) with modular architecture.

### Key Architecture Patterns
- **Layered Design**: `langchain-core` (base abstractions) vs `langchain` (full integrations)
- **Component-Based**: Chains, agents, tools, memory, callbacks as composable units
- **Integration Library**: 300+ connectors for LLMs, vector stores, tools
- **LangGraph Integration**: Low-level orchestration for complex agent workflows

### Agent Structure
```python
from langchain.agents import AgentExecutor, create_tool_calling_agent
from langchain_core.prompts import ChatPromptTemplate
# Agents use tool calling with structured prompts
agent = create_tool_calling_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools)
```

### Memory Systems
- **ConversationBufferMemory**: Stores all messages
- **ConversationBufferWindowMemory**: Last N messages
- **VectorStoreMemory**: Semantic search over history
- **Entity Memory**: Tracks entities mentioned

### Tool/Function Calling Patterns
- Tool classes with `BaseTool` abstraction
- Structured tool input using Pydantic models
- Multi-modal tool inputs supported
- Dynamic tool loading from integrations

### Context Management
- Runnable interface with `invoke()`, `stream()`, `batch()` methods
- RunnableLambda/RunnableParallel for composition
- LCEL (LangChain Expression Language) for chaining

### Testing Approach
- Pytest-based testing
- Integration tests in `libs/langchain-core/tests/`
- Fake LLM implementations for unit testing
- LangSmith for evals and tracing

### Key Files to Analyze
1. `libs/langchain-core/langchain_core/runnables/base.py` - Runnable interface
2. `libs/langchain-core/langchain_core/tools/base.py` - Tool abstraction
3. `libs/langchain/langchain/agents/` - Agent implementations
4. `libs/langchain/langchain/chains/` - Chain compositions

### Shadow AI Applicability
- **Memory Architecture**: VectorStoreMemory pattern ideal for persistent agent state
- **Tool Protocol**: BaseTool abstraction is extensible for custom tools
- **Runnable Pattern**: Composable chain architecture for complex workflows

---

## 2. LlamaIndex (48.3k stars)
**Repository:** https://github.com/run-llama/llama_index

### Architecture Overview
"Document agent and OCR platform" - focused on RAG, data connectors, and indexing. MIT licensed, 7,726 commits.

### Key Architecture Patterns
- **Namespaced Packages**: `llama-index-core` + integration packages
- **Pipeline-Based**: Document ingestion → Indexing → Query engine flow
- **Agent Builder**: Workflow-based agent construction
- **LlamaParse**: Document parsing platform (130+ formats)

### Agent Structure
```python
from llama_index.core import VectorStoreIndex, SimpleDirectoryReader
from llama_index.core.agent import AgentRunner

# Agents built on query engines
documents = SimpleDirectoryReader("data").load_data()
index = VectorStoreIndex.from_documents(documents)
query_engine = index.as_query_engine()
```

### Memory Systems
- **Index-based**: Vector indices for retrieval
- **ChatMemoryBuffer**: Conversational context
- **ChatSummaryMemory**: Summarized history
- **Long-term Memory**: Persistent storage backends

### Tool/Function Calling Patterns
- QueryEngineTools wrapping engines
- FunctionTool for custom tools
- ToolSpec for tool specifications
- Multi-document agent tools

### Context Management
- Response streaming via `query_engine.query()`
- Node postprocessing for reranking
- Metadata filtering for context pruning

### Testing Approach
- Comprehensive test suite in `llama-index-core/tests/`
- Mock indices for testing
- Pytest fixtures for common setup

### Key Files to Analyze
1. `llama-index-core/llama_index/core/indices/` - Index implementations
2. `llama-index-core/llama_index/core/agent/` - Agent runner
3. `llama-index-core/llama_index/core/memory/` - Memory classes
4. `llama-index-core/llama_index/core/tools/` - Tool abstractions

### Shadow AI Applicability
- **Data Connectors**: Extensive ingestion patterns for document sources
- **Index Patterns**: Vector store abstractions for semantic search
- **Memory Types**: Chat memory patterns for conversational agents

---

## 3. AutoGPT (183k stars)
**Repository:** https://github.com/Significant-Gravitas/AutoGPT

### Architecture Overview
Full platform for "accessible AI for everyone" - includes Forge (agent builder), Benchmark, UI Frontend, and Platform Server. Mixed licensing (Polyform Shield for platform).

### Key Architecture Patterns
- **Docker-based**: Multi-container deployment with relational_db, cache, vector store
- **Agent Protocol**: Standardized communication between agents and frontend
- **Block-based Workflow**: Visual agent builder connecting action blocks
- **Marketplace**: Pre-configured agents library

### Agent Structure
```python
# Forge-based agent construction
# Agents run as long-running processes with state persistence
# Manager pattern for hierarchical delegation
```

### Memory Systems
- **Local Memory Store**: Vector embeddings storage
- **Working Memory**: In-context conversation state
- **File System Memory**: Persistent file-based storage

### Tool/Function Calling Patterns
- Protocol-based agent communication
- External tool integrations via APIs
- Visual block editor for tool composition

### Context Management
- Agent state persistence across runs
- File system integration for context
- Vector store for long-term memory

### Testing Approach
- `agbenchmark` for performance evaluation
- Agent Protocol compatibility tests
- Integration tests for platform components

### Key Files to Analyze
1. `autogpt_platform/backend/` - Platform backend services
2. `classic/forge/` - Agent building toolkit
3. `autogpt_platform/frontend/` - Visual agent builder
4. `classic/benchmark/` - Testing framework

### Shadow AI Applicability
- **Docker Architecture**: Containerized agent deployment pattern
- **Agent Protocol**: Standardization of agent communication
- **Visual Builder**: No-code agent configuration approach

---

## 4. AutoGen (56.7k stars)
**Repository:** https://github.com/microsoft/autogen

### Architecture Overview
"A programming framework for agentic AI" - supports Python and .NET. Dual-license (MIT + CC-BY-4.0). Now recommends Microsoft Agent Framework for new projects.

### Key Architecture Patterns
- **Core API**: Message passing, event-driven agents
- **AgentChat API**: Opinionated high-level API
- **Extensions API**: Third-party integrations
- **Cross-language**: Python + .NET support

### Agent Structure
```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

agent = AssistantAgent(
    "assistant", 
    model_client=model_client,
    tools=[...],
    workbench=mcp,
)
await agent.run_stream(task="...")
```

### Memory Systems
- **Thread-based**: ChatHistoryAgentThread for conversation continuity
- **State Management**: Automatic state persistence
- **Long-term Memory**: Extension integrations

### Tool/Function Calling Patterns
- `@kernel_function` decorator for tool registration
- MCP (Model Context Protocol) server integration
- AgentTool for multi-agent orchestration

### Context Management
- Streaming responses via `run_stream()`
- Console UI for real-time output
- AgentThread for stateful conversations

### Testing Approach
- Unit tests for core API
- Integration tests for Agent patterns
- Benchmark suite in `agbench`

### Key Files to Analyze
1. `python/packages/autogen-core/` - Core message passing
2. `python/packages/autogen-agentchat/` - Agent implementations
3. `python/packages/autogen-ext/` - Extensions
4. `dotnet/` - .NET implementation

### Shadow AI Applicability
- **Event-driven Architecture**: Core API's message passing pattern
- **MCP Integration**: Model Context Protocol for tool integration
- **Multi-language**: Cross-platform support pattern

---

## 5. CrewAI (48k stars)
**Repository:** https://github.com/crewAIInc/crewAI

### Architecture Overview
"Fast and flexible multi-agent automation framework" - completely independent of LangChain. MIT licensed, 100,000+ certified developers.

### Key Architecture Patterns
- **Crews**: Autonomous agent teams with role-based collaboration
- **Flows**: Event-driven workflows with precise control
- **YAML Configuration**: `agents.yaml` and `tasks.yaml` definitions
- **Process Types**: Sequential and hierarchical orchestration

### Agent Structure
```python
@CrewBase
class MyCrew():
    @agent
    def researcher(self) -> Agent:
        return Agent(
            config=self.agents_config['researcher'],
            tools=[SerperDevTool()],
            verbose=True
        )
    
    @crew
    def crew(self) -> Crew:
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
        )
```

### Memory Systems
- **Short-term Memory**: Working conversation context
- **Long-term Memory**: Persistent storage via embeddings
- **Context Memory**: Task-specific context

### Tool/Function Calling Patterns
- `@tool` decorator for custom tools
- SerperDevTool for web search
- Built-in tool integrations
- Task-tool binding

### Context Management
- Flow decorators: `@start`, `@listen`, `@router`
- `or_()` and `and_()` for conditional logic
- State sharing between tasks

### Testing Approach
- `pytest` with `conftest.py` fixtures
- Mock agents and tools for isolation
- Flow state testing

### Key Files to Analyze
1. `lib/crewai/crew.py` - Crew orchestration
2. `lib/crewai/agent.py` - Agent definition
3. `lib/crewai/task.py` - Task abstraction
4. `lib/crewai_flow/` - Flow engine

### Shadow AI Applicability
- **Crew Pattern**: Role-based agent team composition
- **YAML Config**: Declarative agent/task definitions
- **Flow Control**: Event-driven with router pattern

---

## 6. LangGraph (28.4k stars)
**Repository:** https://github.com/langchain-ai/langgraph

### Architecture Overview
"Low-level orchestration framework for building stateful agents" - inspired by Pregel and Apache Beam. MIT licensed.

### Key Architecture Patterns
- **Graph-based**: State machines with nodes and edges
- **Durable Execution**: Persistence through failures
- **Human-in-the-Loop**: Interrupt/resume for oversight
- **Memory Systems**: Short-term + long-term persistence

### Agent Structure
```python
from langgraph.graph import StateGraph, MessagesState

def agent(state: MessagesState):
    return {"messages": [llm.invoke(state["messages"])]}

graph = StateGraph(MessagesState)
graph.add_node("agent", agent)
graph.add_edge(START, "agent")
app = graph.compile()
```

### Memory Systems
- **Short-term Working Memory**: In-session state
- **Long-term Persistent Memory**: Cross-session persistence
- **Memory Manager**: API for memory operations

### Tool/Function Calling Patterns
- Tool nodes in graph
- Conditional tool routing
- Tool result accumulation

### Context Management
- State reducers for merging context
- Checkpointer for state persistence
- Thread-based isolation

### Testing Approach
- Graph execution testing
- State transition validation
- LangSmith integration for tracing

### Key Files to Analyze
1. `libs/langgraph/langgraph/graph/state.py` - StateGraph
2. `libs/langgraph/langgraph/pregel.py` - Core execution
3. `libs/langgraph/langgraph/checkpoint/` - Persistence
4. `libs/langgraph/langgraph/memory/` - Memory systems

### Shadow AI Applicability
- **Graph Pattern**: State machine for complex workflows
- **Durable Execution**: Persistence patterns crucial
- **Checkpointer**: State snapshot and recovery

---

## 7. Haystack (24.7k stars)
**Repository:** https://github.com/deepset-ai/haystack

### Architecture Overview
"Open-source AI orchestration framework" - focused on RAG, agents, and retrieval. Apache 2.0 licensed, supports Python.

### Key Architecture Patterns
- **Pipeline-based**: Component chaining with explicit control
- **Document-centric**: Strong document processing focus
- **Retriever-Reader**: RAG architecture
- **Agent**: Tool-calling agent framework

### Agent Structure
```python
from haystack import Pipeline
from haystack.components.agents import Agent

pipeline = Pipeline()
pipeline.add_component("agent", Agent(
    chat_llm=chat_llm,
    tools=[...],
    system_prompt="...",
))
```

### Memory Systems
- **Memory Interface**: Component-based memory
- **ConversationMemory**: Chat history tracking
- **NoMemory**: Stateless option

### Tool/Function Calling Patterns
- Tool components
- ToolSpec definitions
- Dynamic tool loading

### Context Management
- Pipeline routing via Router components
- Conditional branches
- Loop support in pipelines

### Testing Approach
- Comprehensive pytest suite
- Mock components for testing
- Pipeline integration tests

### Key Files to Analyze
1. `haystack/core/pipeline/` - Pipeline implementation
2. `haystack/components/agents/` - Agent components
3. `haystack/document_stores/` - Document persistence
4. `haystack/memory/` - Memory implementations

### Shadow AI Applicability
- **Pipeline Composition**: Clear component boundaries
- **Document Processing**: Strong RAG patterns
- **Memory Interface**: Pluggable memory systems

---

## 8. Semantic Kernel (27.6k stars)
**Repository:** https://github.com/microsoft/semantic-kernel

### Architecture Overview
"Enterprise-ready orchestration framework" - supports Python, .NET, and Java. MIT licensed, Microsoft-maintained.

### Key Architecture Patterns
- **Kernel**: Central orchestration object
- **Plugins/Functions**: Modular capability extension
- **Process Framework**: Business process modeling
- **Multi-language**: C# (66.8%), Python (31.3%), Java

### Agent Structure
```python
from semantic_kernel.agents import ChatCompletionAgent

agent = ChatCompletionAgent(
    service=AzureChatCompletion(),
    name="SK-Assistant",
    instructions="You are a helpful assistant.",
    plugins=[MenuPlugin()],
)
response = await agent.get_response(messages="...")
```

### Memory Systems
- **Volatile Memory**: In-memory storage
- **Semantic Text Memory**: Embedding-based
- **Vector Store Connectors**: Multiple backends

### Tool/Function Calling Patterns
- `@kernel_function` decorator
- KernelPlugin registration
- OpenAPI specification import
- MCP (Model Context Protocol) support

### Context Management
- KernelArguments for passing state
- Chat history management
- Filter pipeline for request handling

### Testing Approach
- Unit tests per language
- Integration tests
- Process framework tests

### Key Files to Analyze
1. `python/semantic_kernel/kernel.py` - Core kernel
2. `python/semantic_kernel/functions/` - Function system
3. `python/semantic_kernel/agents/` - Agent implementations
4. `python/semantic_kernel/process/` - Process framework

### Shadow AI Applicability
- **Kernel Pattern**: Central composition root
- **Plugin System**: Clean extension mechanism
- **Process Framework**: Business workflow modeling

---

## 9. Agents (wooridle/agents) - NOT FOUND
**Repository was not found (404)**

This repository may have been renamed, moved, or deleted. The original URL (https://github.com/wooridle/agents) returned a 404 error.

---

## 10. AgentGPT (35.9k stars) - ARCHIVED
**Repository:** https://github.com/reworkd/AgentGPT
**Status**: Archived by owner on Jan 28, 2026 (read-only)

### Architecture Overview
"Assemble, configure, and deploy autonomous AI agents in your browser" - web-based agent platform. GPL-3.0 licensed, now archived.

### Key Architecture Patterns
- **Next.js Frontend**: Web UI for agent configuration
- **FastAPI Backend**: Python backend services
- **Docker Deployment**: Containerized setup
- **LangChain Integration**: Built on LangChain tooling

### Tech Stack
- Framework: Next.js 13 + FastAPI
- Auth: Next-Auth.js
- ORM: Prisma + SQLModel
- Database: MySQL (Planetscale)
- LLM: LangChain

### Agent Structure
- Goal-based autonomous agents
- Task decomposition and execution
- Learning from results
- Browser-deployed

### Memory Systems
- Database-backed persistence
- Session-based state
- Goal tracking

### Tool/Function Calling Patterns
- LangChain tools integration
- Serper API for search
- Replicate for additional models

### Key Files to Analyze
1. `platform/` - FastAPI backend
2. `next/` - Frontend application
3. `db/` - Database schemas

### Shadow AI Applicability
- **Web UI Pattern**: Browser-based agent interface
- **Full-stack Architecture**: Frontend + Backend separation
- **Note**: Project is archived, use for reference only

---

## Cross-Framework Analysis

### Agent Architecture Patterns

| Framework | Agent Definition | Orchestration | Key Innovation |
|-----------|-----------------|--------------|----------------|
| LangChain | Runnable-based | Chains, Agents | Integration ecosystem |
| LlamaIndex | Query Engine | Pipeline | Document/RAG focus |
| AutoGPT | Block-based | Workflow | Visual builder |
| AutoGen | Event-driven | Message passing | Core + AgentChat layers |
| CrewAI | Role-based | Crews/Flows | Team collaboration |
| LangGraph | State machine | Graph | Durable execution |
| Haystack | Component | Pipeline | Document processing |
| Semantic Kernel | Plugin-based | Kernel/Process | Multi-language |

### Memory Pattern Summary

| Framework | Short-term | Long-term | Persistence |
|-----------|-----------|-----------|-------------|
| LangChain | ConversationBuffer | VectorStore | Various stores |
| LlamaIndex | ChatMemory | Index-based | Vector DB |
| AutoGPT | Working Memory | File System | Vector store |
| AutoGen | AgentThread | Extensions | State store |
| CrewAI | Context | Task memory | Embedding stores |
| LangGraph | Working | Checkpointer | Checkpoint store |
| Haystack | Conversation | Document store | Document store |
| Semantic Kernel | Volatile | Semantic text | Vector connectors |

### Tool/Function Calling Approaches

| Framework | Decorator/Class | Integration Method |
|-----------|-----------------|-------------------|
| LangChain | BaseTool | Tool registry |
| LlamaIndex | FunctionTool | Query engine wrap |
| AutoGPT | Block protocol | Protocol standard |
| AutoGen | @kernel_function | MCP server |
| CrewAI | @tool decorator | Task binding |
| LangGraph | Node functions | Edge routing |
| Haystack | Tool component | Pipeline add |
| Semantic Kernel | @kernel_function | Plugin loading |

### Testing Strategies

| Framework | Test Framework | Mocking Approach |
|-----------|---------------|------------------|
| LangChain | pytest | Fake LLM |
| LlamaIndex | pytest | Mock indices |
| AutoGPT | agbenchmark | Agent protocol |
| AutoGen | pytest + agbench | Mock clients |
| CrewAI | pytest | Mock agents |
| LangGraph | pytest + LangSmith | Checkpointer mock |
| Haystack | pytest | Mock components |
| Semantic Kernel | per-language | Mock services |

---

## Key Learnings for Shadow AI

### Architecture Recommendations

1. **State Management**: LangGraph's checkpointing pattern for durable agent state
2. **Team Coordination**: CrewAI's Crew + Flow combination for multi-agent workflows
3. **Tool Registry**: Semantic Kernel's plugin system for extensible tools
4. **Memory Layers**: LangChain's memory abstractions (buffer + vector)
5. **Agent Protocol**: AutoGPT's standardized communication pattern

### Implementation Priorities

1. **Core Agent Class**: Define base agent with Runnable/StateGraph pattern
2. **Memory System**: Implement short-term + long-term storage abstraction
3. **Tool Registry**: Create decorator-based tool registration system
4. **Workflow Engine**: State machine for task orchestration
5. **Testing Framework**: Fake LLM + mock tools for unit testing

### Cross-Cutting Concerns

- **Observability**: LangSmith integration pattern (tracing, evals)
- **Persistence**: Checkpointer pattern for state snapshots
- **Human-in-the-Loop**: AutoGen's interrupt/resume approach
- **Multi-agent**: CrewAI's role-based team composition
- **Document Processing**: Haystack's component pipeline for RAG

---

## Research Summary

This research analyzed 9 active AI/ML agent frameworks (1 archived, 1 not found) from GitHub, identifying key patterns for agent architecture, memory systems, tool calling, context management, and testing.

**Top Framework Choices for Shadow AI**:
1. **LangGraph** - Best for durable execution and state management
2. **CrewAI** - Best for multi-agent team coordination
3. **Semantic Kernel** - Best for enterprise plugin architecture
4. **LangChain** - Best for integration ecosystem
5. **Haystack** - Best for document-centric RAG

---

*Research completed: Hour 1 of 10-hour GitHub research session*
*Start Time: 00:55 EDT*
*Framework Count: 10 (9 active, 1 archived, 1 not found)*