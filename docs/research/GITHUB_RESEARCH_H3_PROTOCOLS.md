# GitHub Research Hour 3: Context & Protocol Systems

## MCP (Model Context Protocol)

**GitHub**: https://github.com/modelcontextprotocol/servers (82.9k stars)
**SDK**: https://github.com/modelcontextprotocol/python-sdk (22.5k stars)

### Overview
MCP is an open protocol for connecting AI assistants to systems like:
- Content repositories (file systems, databases)
- Business tools (Slack, GitHub, Jira)
- Development environments (IDEs, version control)

### Core Concepts

#### 1. Resources (Read-only data)
```python
@mcp.resource("file://documents/{name}")
def read_document(name: str) -> str:
    return f"Content of {name}"
```

#### 2. Tools (Execute actions)
```python
@mcp.tool()
def sum(a: int, b: int) -> int:
    """Add two numbers together."""
    return a + b
```

#### 3. Prompts (Reusable templates)
```python
@mcp.prompt(title="Code Review")
def review_code(code: str) -> str:
    return f"Please review this code:\n\n{code}"
```

#### 4. Context (Access to session)
```python
@mcp.tool()
async def my_tool(x: int, ctx: Context) -> str:
    await ctx.info("Starting...")
    await ctx.report_progress(progress=0.5)
    return "result"
```

### Key Features
- **Structured Output**: Supports Pydantic models, TypedDicts, dataclasses
- **Transport**: stdio, SSE, Streamable HTTP
- **Authentication**: OAuth 2.1 Resource Server support
- **Sampling**: LLM generation via server
- **Elicitation**: Request user input with schema validation
- **Completions**: Argument auto-completion for prompts/resources
- **Progress Reporting**: Real-time progress updates
- **Logging**: Debug, info, warning, error levels

### Server Implementation Pattern
```python
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("My App")

@mcp.tool()
def my_tool() -> str:
    return "Hello"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### Streamable HTTP Transport
- Default for production servers
- CORS configuration for browser clients
- Mountable to existing ASGI servers (FastAPI, etc.)

### Reference Servers (Official)
- **Memory**: Knowledge graph-based persistent memory
- **Filesystem**: Secure file operations
- **Git**: Repository operations
- **Fetch**: Web content fetching
- **Sequential Thinking**: Problem-solving steps
- **Everything**: Test server with all features

---

## A2A (Agent2Agent Protocol)

**GitHub**: https://github.com/a2aproject/A2A (23k stars)

### Overview
An open protocol enabling communication and interoperability between opaque agentic applications.

### Why A2A?
- **Break Down Silos**: Connect agents across different ecosystems
- **Complex Collaboration**: Allow specialized agents to work together
- **Open Standards**: Community-driven approach
- **Preserve Opacity**: No need to share internal memory/logic

### Key Features
1. **Standardized Communication**: JSON-RPC 2.0 over HTTP(S)
2. **Agent Discovery**: Via "Agent Cards" (capabilities + connection info)
3. **Flexible Interaction**:
   - Synchronous request/response
   - Streaming (SSE)
   - Asynchronous push notifications
4. **Rich Data Exchange**: Text, files, structured JSON
5. **Enterprise-Ready**: Security, authentication, observability

### Agent Card
Describes agent capabilities:
```json
{
  "name": "Weather Agent",
  "capabilities": ["get_weather", "forecast"],
  "skills": [...],
  "endpoints": {...}
}
```

### SDKs Available
- Python: `pip install a2a-sdk`
- Go: `go get github.com/a2aproject/a2a-go`
- JavaScript: `npm install @a2a-js/sdk`
- Java (Maven)
- .NET (NuGet): `dotnet add package A2A`

---

## LangGraph Updates

**GitHub**: https://github.com/langchain-ai/langgraph (28.4k stars)

### What is LangGraph?
Low-level orchestration framework for building **stateful agents** as graphs.

### Key Concepts

#### 1. Durable Execution
- Persist through failures
- Resume from exact state
- Long-running workflows

#### 2. Human-in-the-Loop
- Inspect agent state at any point
- Modify state during execution
- Approval workflows

#### 3. Memory
- **Short-term**: Working memory for reasoning
- **Long-term**: Persistent memory across sessions

#### 4. Graph-based Architecture
- Nodes: Functions/agents
- Edges: Conditional transitions
- State: Typed dictionaries

### Core Pattern
```python
from langgraph.graph import StateGraph, END

workflow = StateGraph(State)

# Add nodes
workflow.add_node("thinking", think_node)
workflow.add_node("action", action_node)

# Add edges
workflow.add_edge("thinking", "action")
workflow.add_conditional_edges(
    "action",
    should_continue,
    {True: "thinking", False: END}
)

# Compile
app = workflow.compile()
```

### Memory Checkpointing
```python
from langgraph.checkpoint.memory import MemorySaver
from langgraph.checkpoint.postgres import PostgresSaver

# In-memory
checkpointer = MemorySaver()

# PostgreSQL persistent
checkpointer = PostgresSaver(conn_string)
```

---

## Implementation Patterns for Shadow AI

### MCP Server for Shadow AI

```python
# shadowd/mcp_server.py
from mcp.server.fastmcp import FastMCP, Context
from pathlib import Path
import json

mcp = FastMCP(
    "Shadow AI",
    version="1.0.0"
)

# Resources: User context
@mcp.resource("shadow://context/{user_id}")
def get_user_context(user_id: str) -> str:
    """Load user context/memory."""
    path = Path(f"data/users/{user_id}/context.json")
    if path.exists():
        return path.read_text()
    return "{}"

# Tools: Core actions
@mcp.tool()
async def search_memory(query: str, ctx: Context) -> list[str]:
    """Search user memory for relevant information."""
    await ctx.info(f"Searching for: {query}")
    # Use Shadow's search
    return ["result1", "result2"]

@mcp.tool()
async def execute_workflow(workflow_name: str, params: dict) -> dict:
    """Execute a predefined workflow."""
    # Use Shadow's state graph
    return {"status": "completed"}

# Prompts
@mcp.prompt(title="Shadow Analysis")
def analyze_interaction(context: str) -> str:
    return f"Analyze this user interaction:\n{context}"

if __name__ == "__main__":
    mcp.run(transport="streamable-http")
```

### A2A Agent Protocol

```python
# shadowd/a2a_server.py
from a2a_sdk import A2AServer, AgentCard, Skill

# Define agent card
card = AgentCard(
    name="Shadow AI",
    description="Personal AI assistant with context & memory",
    skills=[
        Skill(
            id="remember",
            name="Remember Information",
            description="Store information in knowledge graph"
        ),
        Skill(
            id="recall",
            name="Recall Information",
            description="Search and retrieve from memory"
        )
    ]
)

server = A2AServer(card)

@server.skill("remember")
async def remember(content: str, metadata: dict) -> dict:
    # Use Shadow's knowledge graph
    return {"status": "stored"}

@server.skill("recall")
async def recall(query: str) -> dict:
    # Use Shadow's search
    return {"results": [...]}
```

### Integration Points

1. **MCP Resources** → Shadow's context/memory
2. **MCP Tools** → Shadow's workflows/actions
3. **A2A Skills** → Shadow's capabilities
4. **LangGraph State** → Shadow's state management

---

## Summary: Protocol Comparison

| Feature | MCP | A2A | LangGraph |
|---------|-----|-----|-----------|
| **Purpose** | Context for LLMs | Agent-to-Agent | Workflow graphs |
| **Transport** | HTTP/SSE/stdio | HTTP/SSE | Internal |
| **Discovery** | Resources | Agent Cards | N/A |
| **Actions** | Tools | Skills | Nodes |
| **State** | Per-request | Per-task | Persistent graphs |
| **Memory** | External | External | Built-in |

### Best for Shadow AI

1. **MCP**: Expose Shadow as a context provider for other LLM apps
2. **A2A**: Allow other agents to use Shadow's capabilities
3. **LangGraph**: Already integrated in Shadow's internal workflows

### Implementation Priority

1. **Phase 1**: MCP Server (expose memory/context/tools)
2. **Phase 2**: MCP Client (connect to other MCP servers)
3. **Phase 3**: A2A Integration (inter-agent communication)
4. **Phase 4**: Combined protocol gateway

---

## Key Takeaways

1. **MCP is the standard protocol** for LLM context exchange
2. **A2A enables agent collaboration** without exposing internals
3. **LangGraph provides state management** for complex workflows
4. **Shadow AI can be both MCP server AND client**
5. **Protocols complement each other** - not competing standards