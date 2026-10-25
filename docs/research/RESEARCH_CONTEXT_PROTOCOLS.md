# Model Context Protocol (MCP) & Agent Communication Research

**Date:** April 3, 2026
**Purpose:** Research context protocols for integration with digital-shadow project

---

## Executive Summary

This research covers three major protocol systems for AI context and agent communication:

1. **Model Context Protocol (MCP)** - Anthropic's standard for connecting AI to data sources and tools
2. **Context7** - Up-to-date documentation retrieval system for LLMs
3. **Agent2Agent (A2A)** - Google's protocol for inter-agent communication

**Key Finding:** Our existing `context_protocol.py` implements MCP-inspired patterns but lacks the standardized primitives and client-server architecture needed for interoperability.

---

## 1. Model Context Protocol (MCP)

### Overview

MCP is an open standard created by Anthropic for connecting AI applications to external systems. It uses JSON-RPC 2.0 over STDIO or HTTP transports.

**Official Documentation:** https://modelcontextprotocol.io
**GitHub:** https://github.com/modelcontextprotocol
**License:** MIT License (Linux Foundation)

### Core Architecture

```
┌─────────────────────────────────────────┐
│          MCP Host (AI Application)      │
│                                         │
│  ┌──────────┐  ┌──────────┐  ┌────────┐│
│  │Client 1  │  │Client 2  │  │Client 3││
│  └────┬─────┘  └────┬─────┘  └───┬────┘│
└───────┼─────────────┼────────────┼──────┘
        │             │            │
    ┌───▼────┐    ┌───▼────┐   ┌───▼────┐
    │Server A│    │Server B│   │Server C│
    │(Files) │    │(DB)    │   │(Remote)│
    └────────┘    └────────┘   └────────┘
```

### Protocol Layers

#### 1. Transport Layer
- **STDIO Transport**: Local process communication (filesystem, local tools)
- **HTTP Transport**: Remote server communication with OAuth/bearer tokens

#### 2. Data Layer (JSON-RPC 2.0)

**Core Primitives:**

| Primitive | Purpose | Methods |
|-----------|---------|---------|
| **Tools** | Executable functions for AI actions | `tools/list`, `tools/call` |
| **Resources** | Data sources for context | `resources/list`, `resources/read` |
| **Prompts** | Reusable interaction templates | `prompts/list`, `prompts/get` |

**Client Primitives:**
- **Sampling**: Request LLM completions (`sampling/complete`)
- **Elicitation**: Request user input (`elicitation/request`)
- **Logging**: Debug messages to client

### Lifecycle Management

```json
// 1. Initialize handshake
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "initialize",
  "params": {
    "protocolVersion": "2025-06-18",
    "capabilities": { "tools": { "listChanged": true } },
    "clientInfo": { "name": "shadow-ai", "version": "1.0.0" }
  }
}

// 2. Server response
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "protocolVersion": "2025-06-18",
    "capabilities": { 
      "tools": { "listChanged": true },
      "resources": {}
    },
    "serverInfo": { "name": "shadow-server", "version": "1.0.0" }
  }
}

// 3. Client confirms ready
{ "jsonrpc": "2.0", "method": "notifications/initialized" }
```

### Key Features

1. **Capability Negotiation** - Client and server agree on supported features
2. **Notifications** - Real-time updates when server capabilities change
3. **Tasks (Experimental)** - Long-running operations with status tracking
4. **Multi-transport** - Works locally (STDIO) and remotely (HTTP)

### Official SDKs

| Language | Package | Status |
|----------|---------|--------|
| TypeScript | `@modelcontextprotocol/sdk` | Production |
| Python | `mcp` | Production |
| Java | `io.modelcontextprotocol:mcp` | Production |
| Kotlin | `io.modelcontextprotocol:mcp` | Production |
| C# | `ModelContextProtocol` | Production |
| Go | `go.mcp` | Production |
| Ruby | `mcp` | Production |
| Rust | `mcp-rust-sdk` | Production |
| Swift | `MCP` | Production |
| PHP | `mcp` | Production |

---

## 2. Context7 Protocol

### Overview

Context7 is a documentation retrieval system that provides up-to-date, version-specific library documentation to LLMs via MCP.

**Website:** https://context7.com
**GitHub:** https://github.com/upstash/context7
**Stars:** 51.6k+ (very popular)

### Purpose

Solves the problem of LLMs having outdated training data. Context7 fetches current documentation and injects it into prompts.

### Architecture

```
User Prompt ("use context7")
         │
         ▼
┌──────────────────┐
│  AI Code Editor   │
│  (Cursor/Claude)  │
└────────┬─────────┘
         │
         ▼
┌──────────────────┐      ┌─────────────────┐
│  Context7 MCP     │─────▶│ Context7 API    │
│  Server           │      │ (up-to-date docs)│
└──────────────────┘      └─────────────────┘
```

### MCP Tools

1. **`resolve-library-id`**
   - Input: library name
   - Output: Context7-compatible library ID (e.g., `/vercel/next.js`)

2. **`query-docs`**
   - Input: library ID + query
   - Output: relevant documentation snippets

### Usage Examples

```bash
# CLI usage
ctx7 library <name> <query>       # Search for library
ctx7 docs <libraryId> <query>     # Get documentation

# In prompts (with "use context7")
"Create Next.js 14 middleware for auth. use context7"
"How to use Supabase auth API. use context7"
```

### Integration Pattern

```typescript
// MCP Server provides these tools:
{
  "name": "resolve-library-id",
  "inputSchema": {
    "type": "object",
    "properties": {
      "libraryName": { "type": "string" },
      "query": { "type": "string" }
    }
  }
}

{
  "name": "query-docs",
  "inputSchema": {
    "type": "object",
    "properties": {
      "libraryId": { "type": "string" },
      "query": { "type": "string" }
    }
  }
}
```

### Key Insights for Integration

- Context7 is **MCP-compliant** - uses standard MCP tools
- Real-time documentation fetching prevents hallucinated APIs
- Version-specific (can request docs for specific library versions)
- Works as MCP server with standard transport

---

## 3. Agent2Agent (A2A) Protocol

### Overview

A2A is Google's open protocol for enabling communication between AI agents across different frameworks and servers.

**Website:** https://a2a-protocol.org
**GitHub:** https://github.com/a2aproject/A2A
**License:** Apache 2.0
**Stars:** 23k+

### Purpose

Enable **opaque agents** to collaborate without exposing:
- Internal state
- Memory
- Tool implementations

### Core Concepts

1. **Agent Cards** - Discovery metadata describing capabilities
2. **Tasks** - Long-running collaborative operations
3. **Messages** - Communication between agents
4. **Artifacts** - Files, media, structured data

### Key Features

| Feature | Description |
|---------|-------------|
| **Discovery** | Agent Cards with capabilities and connection info |
| **Modalities** | Text, forms, media support |
| **Transport** | JSON-RPC 2.0 over HTTP(S) |
| **Streaming** | Server-Sent Events (SSE) |
| **Async** | Push notifications for long tasks |
| **Security** | Auth schemes in Agent Cards |

### Agent Card Structure

```json
{
  "id": "agent-001",
  "name": "Shadow Intelligence Agent",
  "capabilities": {
    "skills": ["context_analysis", "pattern_recognition"],
    "modalities": ["text", "structured_data"],
    "authentication": ["bearer_token"]
  },
  "endpoints": {
    "base_url": "https://shadow.example.com/a2a",
    "tasks": "/tasks",
    "messages": "/messages"
  }
}
```

### Communication Pattern

```
Agent A                    Agent B
   │                          │
   │─── Agent Card ─────────▶│  (discovery)
   │                          │
   │─── Create Task ─────────▶│
   │                          │
   │◀── Task ID ─────────────│
   │                          │
   │─── Message ─────────────▶│
   │                          │
   │◀── Message/Artifact ─────│
   │                          │
   │◀── Task Complete ───────│
```

### A2A vs MCP

| Aspect | MCP | A2A |
|--------|-----|-----|
| **Purpose** | AI ↔ Tools/Data | Agent ↔ Agent |
| **Protocol** | JSON-RPC 2.0 | JSON-RPC 2.0 |
| **Transport** | STDIO, HTTP | HTTP(S), SSE |
| **Discovery** | Capability negotiation | Agent Cards |
| **State** | Per-connection | Task-based |
| **Best for** | Single agent tools | Multi-agent collaboration |

**Complementary:** A2A can use MCP for tool access within agents.

### Official SDKs

| Language | Package |
|----------|---------|
| Python | `pip install a2a-sdk` |
| Go | `go get github.com/a2aproject/a2a-go` |
| JavaScript | `npm install @a2a-js/sdk` |
| Java | Maven dependency |
| .NET | `dotnet add package A2A` |

---

## 4. Integration with Existing Code

### Current Implementation Analysis

Our `context_protocol.py` (lines 1-421) implements:

```python
# Current structure (simplified)
class ContextType(Enum):
    DEVICE = "device"
    LOCATION = "location"
    ACTIVITY = "activity"
    TEMPORAL = "temporal"
    SOCIAL = "social"
    MEMORY = "memory"
    ENVIRONMENT = "environment"
    DIGITAL = "digital"

class ContextSource:
    name: str
    context_type: ContextType
    provider: Callable
    update_interval: int

class ContextSnapshot:
    timestamp: datetime
    # ... context fields ...

class ContextProtocol:
    sources: Dict[ContextType, List[ContextSource]]
    current_snapshot: ContextSnapshot
    
    def get_context_for_ability(self, ability_name: str) -> Dict
```

### Gap Analysis

| Feature | MCP/A2A Standard | Our Implementation | Gap |
|---------|------------------|-------------------|-----|
| **Transport** | STDIO/HTTP (standardized) | Direct Python calls | ❌ No standard transport |
| **Discovery** | Capability negotiation/Agent Cards | Hard-coded abilities | ❌ No dynamic discovery |
| **Primitives** | Tools, Resources, Prompts | Context types only | ⚠️ Tools missing |
| **Protocol** | JSON-RPC 2.0 | None | ❌ No wire protocol |
| **Notifications** | Real-time updates | Subscriber pattern | ✅ Partially implemented |
| **Client/Server** | Standard architecture | Monolithic | ❌ No separation |

### Recommended Improvements

#### Phase 1: MCP Compliance (Priority: High)

```python
# 1. Add MCP transport layer
from mcp import Client, Server, StdioTransport, HttpTransport

class ShadowMCPServer:
    """MCP-compliant server for digital-shadow."""
    
    def __init__(self):
        self.server = Server(
            name="digital-shadow",
            version="1.0.0"
        )
        self._register_tools()
        self._register_resources()
    
    def _register_tools(self):
        # Context snapshot tool
        @self.server.tool("get_context")
        async def get_context(ability: str = None) -> dict:
            """Get current context snapshot, optionally filtered by ability."""
            return await self.context_protocol.get_context_for_ability(ability)
        
        # Context query tool
        @self.server.tool("query_context")
        async def query_context(
            context_types: List[str],
            time_range: Optional[dict] = None
        ) -> dict:
            """Query historical context data."""
            return await self.context_protocol.query_history(
                context_types, time_range
            )
    
    def _register_resources(self):
        # Context as resource
        @self.server.resource("context://current")
        async def get_current_context() -> str:
            """Current context snapshot."""
            return json.dumps(self.context_protocol.current_snapshot.to_dict())
        
        # Context history resource
        @self.server.resource("context://history/{limit}")
        async def get_context_history(limit: int = 100) -> str:
            """Context history."""
            snapshots = self.context_protocol.context_history[-limit:]
            return json.dumps([s.to_dict() for s in snapshots])
```

#### Phase 2: Standard Primitives

```python
# Define MCP-compliant tool schemas
TOOLS = [
    {
        "name": "get_context",
        "title": "Get Context",
        "description": "Get current context snapshot for an ability",
        "inputSchema": {
            "type": "object",
            "properties": {
                "ability": {
                    "type": "string",
                    "enum": [
                        "oracle", "x_ray", "life_pulse", 
                        "perfect_memory", "pattern_sight",
                        "social_radar", "intent_seer",
                        "omni_sense", "shadow_council", "reality_forge"
                    ],
                    "description": "Ability to get context for"
                }
            }
        }
    },
    {
        "name": "query_temporal",
        "title": "Query Temporal Context",
        "description": "Query temporal context (time/date/season)",
        "inputSchema": {
            "type": "object",
            "properties": {
                "time_range": {
                    "type": "object",
                    "properties": {
                        "start": {"type": "string", "format": "date-time"},
                        "end": {"type": "string", "format": "date-time"}
                    }
                }
            }
        }
    },
    {
        "name": "query_activity",
        "title": "Query Activity Context",
        "description": "Query user activity context",
        "inputSchema": {
            "type": "object",
            "properties": {
                "include_idle": {"type": "boolean", "default": False}
            }
        }
    }
]
```

#### Phase 3: A2A Agent Card

```python
# A2A Agent Card for multi-agent collaboration
def get_agent_card() -> dict:
    return {
        "id": "digital-shadow-main",
        "name": "Digital Shadow Intelligence",
        "description": "Ambient intelligence with context awareness",
        "capabilities": {
            "skills": [
                "context_analysis",
                "pattern_recognition",
                "temporal_analysis",
                "activity_tracking",
                "digital_context"
            ],
            "modalities": ["text", "structured_data"],
            "streaming": True
        },
        "endpoints": {
            "base_url": "http://localhost:8080",
            "mcp": "/mcp",
            "a2a": "/a2a"
        },
        "authentication": {
            "schemes": ["bearer_token", "api_key"]
        }
    }
```

#### Phase 4: Unified Architecture

```python
# Hybrid MCP+A2A architecture
class DigitalShadowProtocol:
    """
    Unified protocol supporting:
    - MCP for tool/resource access
    - A2A for agent collaboration
    - Internal context management
    """
    
    def __init__(self):
        # MCP Server for tool access
        self.mcp_server = ShadowMCPServer()
        
        # A2A Client for agent collaboration
        self.a2a_client = A2AClient()
        
        # Context Protocol (existing)
        self.context_protocol = ContextProtocol()
        
        # Bridge: Context changes → MCP notifications
        self.context_protocol.subscribe(self._on_context_update)
    
    async def _on_context_update(self, snapshot: ContextSnapshot):
        """Bridge context updates to MCP notifications."""
        await self.mcp_server.notify_resource_updated("context://current")
        
        # If collaborating with other agents, notify them
        if self.a2a_client.has_active_tasks():
            await self.a2a_client.broadcast_event({
                "type": "context_update",
                "snapshot": snapshot.to_dict()
            })
```

---

## 5. Implementation Roadmap

### Immediate Actions (Week 1-2)

1. **Install MCP Python SDK**
   ```bash
   pip install mcp
   ```

2. **Create MCP Server Wrapper**
   - Wrap existing `ContextProtocol` as MCP tools
   - Implement capability negotiation
   - Add STDIO transport

3. **Define Tool Schemas**
   - Convert context getters to MCP tools
   - Add input schemas for validation
   - Document each tool

### Short-term (Week 3-4)

4. **Add Resource Primitives**
   - Context snapshot as resource
   - Historical context as resource
   - Subscription for updates

5. **Implement Notifications**
   - `resources/list_changed` when context updates
   - Real-time context sync

6. **Testing with MCP Clients**
   - Test with Claude Desktop
   - Test with Claude Code
   - Validate JSON-RPC protocol

### Medium-term (Week 5-8)

7. **A2A Integration**
   - Create Agent Card
   - Implement A2A endpoints
   - Test agent collaboration

8. **HTTP Transport**
   - Add HTTP transport for remote access
   - Implement authentication (OAuth/bearer tokens)
   - Add CORS for web clients

9. **Context7 Integration**
   - Use Context7 MCP server for up-to-date docs
   - Integrate library documentation retrieval

### Long-term (Week 9+)

10. **Multi-Agent Architecture**
    - Enable A2A collaboration between abilities
    - Task delegation between agents
    - Result aggregation

11. **Enterprise Features**
    - Authentication/authorization
    - Rate limiting
    - Observability (logging, metrics)

---

## 6. Code Samples

### MCP Server Example (Python)

```python
from mcp import Server, StdioTransport
from mcp.types import Tool, TextContent
import asyncio

class ShadowMCPServer:
    def __init__(self, context_protocol):
        self.context = context_protocol
        self.server = Server(
            name="digital-shadow",
            version="1.0.0"
        )
        self._setup_handlers()
    
    def _setup_handlers(self):
        @self.server.list_tools()
        async def list_tools():
            return [
                Tool(
                    name="get_context",
                    description="Get current context snapshot",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "ability": {
                                "type": "string",
                                "enum": ["oracle", "x_ray", "life_pulse", "..."]
                            }
                        }
                    }
                ),
                Tool(
                    name="query_temporal",
                    description="Query temporal context",
                    inputSchema={
                        "type": "object",
                        "properties": {
                            "time_range": {"type": "object"}
                        }
                    }
                )
            ]
        
        @self.server.call_tool()
        async def call_tool(name: str, arguments: dict):
            if name == "get_context":
                ability = arguments.get("ability")
                context = await self.context.get_context_for_ability(ability)
                return [TextContent(
                    type="text",
                    text=json.dumps(context, indent=2)
                )]
            # ... other tools
    
    async def run(self):
        transport = StdioTransport()
        await self.server.connect(transport)
        await self.server.run()

# Usage
context = ContextProtocol()
server = ShadowMCPServer(context)
asyncio.run(server.run())
```

### A2A Task Example

```python
from a2a import AgentClient, Task, Message

async def collaborate_with_agent():
    # Create A2A client
    client = AgentClient(base_url="https://other-agent.example.com")
    
    # Get agent card for discovery
    card = await client.get_agent_card()
    print(f"Connected to: {card.name}")
    print(f"Capabilities: {card.capabilities.skills}")
    
    # Create a task
    task = await client.create_task({
        "type": "analysis",
        "description": "Analyze temporal patterns"
    })
    
    # Send message
    await client.send_message(task.id, Message(
        content={
            "type": "text",
            "text": "Please analyze activity patterns"
        }
    ))
    
    # Get results (with streaming)
    async for artifact in client.stream_artifacts(task.id):
        print(f"Received: {artifact}")
```

---

## 7. Benefits of Integration

### MCP Benefits

1. **Interoperability**
   - Work with any MCP-compliant client (Claude, Cursor, VS Code)
   - Standard protocol for tool access

2. **Discoverability**
   - Dynamic capability negotiation
   - Clients auto-discover available tools

3. **Ecosystem**
   - Join growing MCP ecosystem (7.7k+ stars)
   - Access to pre-built MCP servers

4. **Real-time**
   - Notifications for context updates
   - Push-based rather than polling

### A2A Benefits

1. **Agent Collaboration**
   - Delegate tasks to specialized agents
   - Multi-agent workflows

2. **Privacy**
   - Opaque agents (don't expose internals)
   - Share only necessary information

3. **Flexibility**
   - Agents from different frameworks can collaborate
   - No vendor lock-in

### Context7 Benefits

1. **Up-to-date Knowledge**
   - Current library versions
   - No hallucinated APIs

2. **Reduced Errors**
   - Accurate code generation
   - Version-specific examples

---

## 8. Recommended Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                    Digital Shadow System                     │
│                                                              │
│  ┌──────────────┐      ┌──────────────┐      ┌───────────┐│
│  │ MCP Server   │◄────►│ Context      │◄────►│ A2A       ││
│  │ (Tools/Res)  │      │ Protocol     │      │ Client    ││
│  └──────┬───────┘      └──────┬───────┘      └─────┬─────┘│
│         │                     │                     │      │
└─────────┼─────────────────────┼─────────────────────┼──────┘
          │                     │                     │
          │ STDIO/HTTP           │ Memory/DB           │ HTTP
          ▼                     ▼                     ▼
┌──────────────────┐  ┌─────────────────┐  ┌─────────────────┐
│ Claude Desktop   │  │ Shadow Storage  │  │ Other Agents    │
│ Cursor           │  │ (Files/DBs)     │  │ (via A2A)       │
│ VS Code          │  └─────────────────┘  └─────────────────┘
│ Any MCP Client   │
└──────────────────┘

Data Flow:
1. Sensors → Context Protocol → Shadow Abilities
2. Context Protocol → MCP Server → External AI clients
3. Context Protocol → A2A Client → Other agents for collaboration
```

---

## 9. Key Takeaways

1. **MCP is the standard for AI-context communication**
   - Use MCP to expose digital-shadow capabilities to external AI clients
   - Implement as STDIO server for local use, HTTP for remote

2. **Context7 is MCP-compliant documentation retrieval**
   - Use it to keep AI knowledge current
   - Integrate as MCP server in client applications

3. **A2A enables multi-agent collaboration**
   - Use for complex tasks requiring multiple specialized agents
   - Complements MCP (agents can use MCP internally)

4. **Our context_protocol.py is protocol-agnostic**
   - Needs MCP wrapper to be interoperable
   - Can add A2A client for agent collaboration

5. **Priority: MCP integration first**
   - MCP is more mature and widely adopted
   - A2A can be added later for multi-agent scenarios

---

## 10. Resources

### Official Documentation

- **MCP:** https://modelcontextprotocol.io
- **MCP Spec:** https://modelcontextprotocol.io/specification/latest
- **A2A:** https://a2a-protocol.org
- **Context7:** https://context7.com/docs

### GitHub Repositories

- **MCP:** https://github.com/modelcontextprotocol/modelcontextprotocol
- **MCP Servers:** https://github.com/modelcontextprotocol/servers
- **MCP Python SDK:** https://github.com/modelcontextprotocol/python-sdk
- **A2A Protocol:** https://github.com/a2aproject/A2A
- **A2A Python SDK:** https://github.com/a2aproject/a2a-python
- **Context7:** https://github.com/upstash/context7

### DeepLearning.AI Course

- **A2A Protocol Course:** https://goo.gle/dlai-a2a

---

## Conclusion

The digital-shadow project should:

1. **Adopt MCP protocol** to expose capabilities as standard tools/resources
2. **Keep existing ContextProtocol** as internal context manager
3. **Add MCP Server wrapper** for external client access
4. **Consider A2A integration** for future multi-agent scenarios
5. **Use Context7 MCP** in client applications for up-to-date docs

This positions digital-shadow as a first-class citizen in the AI tooling ecosystem while maintaining its unique context-aware capabilities.