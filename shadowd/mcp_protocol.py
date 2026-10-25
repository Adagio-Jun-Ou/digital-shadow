#!/usr/bin/env python3
"""
MCP-Compliant Context Protocol - Based on Anthropic's Model Context Protocol

Implements:
- JSON-RPC 2.0 protocol
- Tools, Resources, Prompts primitives
- STDIO and HTTP transports
- Client-server architecture
"""

import asyncio
import copy
import json
import re as _re
import threading
from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
from datetime import datetime, timezone
import logging

logger = logging.getLogger("shadowd.mcp_protocol")


# JSON-RPC 2.0 Types

@dataclass
class JSONRPCRequest:
    """JSON-RPC 2.0 Request."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            "jsonrpc": self.jsonrpc,
            "id": self.id,
            "method": self.method,
            "params": self.params
        })
    
    @staticmethod
    def from_json(data: str) -> 'JSONRPCRequest':
        d = json.loads(data)
        return JSONRPCRequest(
            jsonrpc=d.get("jsonrpc", "2.0"),
            id=d.get("id"),
            method=d.get("method", ""),
            params=d.get("params", {})
        )


@dataclass
class JSONRPCResponse:
    """JSON-RPC 2.0 Response."""
    jsonrpc: str = "2.0"
    id: Optional[int] = None
    result: Optional[Any] = None
    error: Optional[Dict] = None
    
    def to_json(self) -> str:
        data = {"jsonrpc": self.jsonrpc, "id": self.id}
        if self.result is not None:
            data["result"] = self.result
        if self.error:
            data["error"] = self.error
        return json.dumps(data)


@dataclass
class JSONRPCNotification:
    """JSON-RPC 2.0 Notification (no id)."""
    jsonrpc: str = "2.0"
    method: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    
    def to_json(self) -> str:
        return json.dumps({
            "jsonrpc": self.jsonrpc,
            "method": self.method,
            "params": self.params
        })


# MCP Primitives

@dataclass
class MCPTool:
    """MCP Tool definition."""
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "inputSchema": copy.deepcopy(self.input_schema)
        }


@dataclass
class MCPResource:
    """MCP Resource definition."""
    uri: str
    name: str
    description: str
    mime_type: str = "text/plain"
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            "uri": self.uri,
            "name": self.name,
            "description": self.description,
            "mimeType": self.mime_type
        }


_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    return _SANITIZE_STRICT_RE.sub('', text) if text else text


def _sanitize_content(text: str) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text) if text else text


@dataclass
class MCPPrompt:
    """MCP Prompt template."""
    name: str
    description: str
    arguments: List[Dict[str, Any]] = field(default_factory=list)
    template: str = ""
    handler: Optional[Callable] = None
    
    def to_dict(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "arguments": copy.deepcopy(self.arguments)
        }


class MCPServer:
    """
    MCP Server implementation (STDIO transport).

    Provides Shadow AI capabilities as MCP-compliant tools/resources.
    """

    MAX_TOOLS = 100
    MAX_RESOURCES = 100
    MAX_PROMPTS = 50

    def __init__(self, name: str = "shadow-ai", version: str = "1.0.0"):
        self.name = _sanitize(name)
        self.version = _sanitize(version)
        self.protocol_version = "2025-06-18"

        self.tools: Dict[str, MCPTool] = {}
        self.resources: Dict[str, MCPResource] = {}
        self.prompts: Dict[str, MCPPrompt] = {}

        self.client_capabilities: Dict = {}
        self.client_info: Dict = {}

        self._initialized = False
        self._lock = threading.Lock()

    def register_tool(self, tool: MCPTool):
        """Register an MCP tool."""
        with self._lock:
            if len(self.tools) >= self.MAX_TOOLS:
                oldest_name = next(iter(self.tools))
                del self.tools[oldest_name]
                logger.warning(f"MCP tool evicted to make room: {oldest_name}")
            self.tools[_sanitize(tool.name)] = tool
            logger.info(f"Registered MCP tool: {tool.name}")

    def register_resource(self, resource: MCPResource):
        """Register an MCP resource."""
        with self._lock:
            if len(self.resources) >= self.MAX_RESOURCES:
                oldest_uri = next(iter(self.resources))
                del self.resources[oldest_uri]
                logger.warning(f"MCP resource evicted to make room: {oldest_uri}")
            self.resources[_sanitize(resource.uri)] = resource
            logger.info(f"Registered MCP resource: {resource.uri}")

    def register_prompt(self, prompt: MCPPrompt):
        """Register an MCP prompt."""
        with self._lock:
            if len(self.prompts) >= self.MAX_PROMPTS:
                oldest_name = next(iter(self.prompts))
                del self.prompts[oldest_name]
                logger.warning(f"MCP prompt evicted to make room: {oldest_name}")
            self.prompts[_sanitize(prompt.name)] = prompt
            logger.info(f"Registered MCP prompt: {prompt.name}")
    
    async def handle_request(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle incoming JSON-RPC request."""
        try:
            method = _sanitize(request.method)
            
            # Initialization gate: only 'initialize' allowed before init
            if method != "initialize" and not self._initialized:
                return JSONRPCResponse(
                    id=request.id,
                    error={
                        "code": -32002,
                        "message": "Server not initialized. Send 'initialize' first."
                    }
                )
            
            # Handle standard MCP methods
            if method == "initialize":
                return await self._handle_initialize(request)
            elif method == "tools/list":
                return await self._handle_tools_list(request)
            elif method == "tools/call":
                return await self._handle_tools_call(request)
            elif method == "resources/list":
                return await self._handle_resources_list(request)
            elif method == "resources/read":
                return await self._handle_resources_read(request)
            elif method == "prompts/list":
                return await self._handle_prompts_list(request)
            elif method == "prompts/get":
                return await self._handle_prompts_get(request)
            else:
                return JSONRPCResponse(
                    id=request.id,
                    error={
                        "code": -32601,
                        "message": f"Method not found: {method}"
                    }
                )
        
        except Exception as e:
            logger.error(f"Error handling request: {type(e).__name__}")
            return JSONRPCResponse(
                id=request.id,
                error={
                    "code": -32603,
                    "message": "Internal server error"
                }
            )
    
    async def _handle_initialize(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Handle initialization handshake."""
        self.client_capabilities = request.params.get("capabilities", {})
        self.client_info = request.params.get("clientInfo", {})
        self._initialized = True
        
        logger.info(f"MCP client initialized: {self.client_info}")
        
        return JSONRPCResponse(
            id=request.id,
            result={
                "protocolVersion": self.protocol_version,
                "capabilities": {
                    "tools": {"listChanged": True},
                    "resources": {"listChanged": True},
                    "prompts": {}
                },
                "serverInfo": {
                    "name": self.name,
                    "version": self.version
                }
            }
        )
    
    async def _handle_tools_list(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """List available tools."""
        with self._lock:
            tools = [tool.to_dict() for tool in self.tools.values()]
        return JSONRPCResponse(
            id=request.id,
            result={
                "tools": tools
            }
        )
    
    async def _handle_tools_call(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Execute a tool."""
        tool_name = _sanitize(str(request.params.get("name", "")))
        arguments = request.params.get("arguments", {})
        
        if tool_name not in self.tools:
            return JSONRPCResponse(
                id=request.id,
                error={
                    "code": -32602,
                    "message": f"Tool not found: {tool_name}"
                }
            )
        
        tool = self.tools[tool_name]
        
        if tool.handler:
            try:
                valid_args = {}
                if isinstance(arguments, dict):
                    schema_props = tool.input_schema.get("properties", {})
                    required = tool.input_schema.get("required", [])
                    for key in required:
                        if key not in arguments:
                            return JSONRPCResponse(
                                id=request.id,
                                error={"code": -32602, "message": f"Missing required argument: {key}"}
                            )
                    for key, value in arguments.items():
                        if key in schema_props:
                            valid_args[key] = value
                result = await tool.handler(**valid_args) if asyncio.iscoroutinefunction(tool.handler) else tool.handler(**valid_args)
                return JSONRPCResponse(id=request.id, result={"content": [{"type": "text", "text": str(result)}]})
            except (ValueError, KeyError, TypeError, OSError):
                return JSONRPCResponse(
                    id=request.id,
                    result={"content": [{"type": "text", "text": "Error: Tool execution failed"}], "isError": True}
                )
        else:
            return JSONRPCResponse(
                id=request.id,
                error={"code": -32603, "message": "Tool has no handler"}
            )
    
    async def _handle_resources_list(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """List available resources."""
        with self._lock:
            resources = [res.to_dict() for res in self.resources.values()]
        return JSONRPCResponse(
            id=request.id,
            result={
                "resources": resources
            }
        )
    
    async def _handle_resources_read(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Read a resource."""
        uri = _sanitize(str(request.params.get("uri", "")))
        
        if uri not in self.resources:
            return JSONRPCResponse(
                id=request.id,
                error={
                    "code": -32602,
                    "message": f"Resource not found: {uri}"
                }
            )
        
        resource = self.resources[uri]
        
        if resource.handler:
            content = await resource.handler() if asyncio.iscoroutinefunction(resource.handler) else resource.handler()
            return JSONRPCResponse(
                id=request.id,
                result={
                    "contents": [{
                        "uri": uri,
                        "mimeType": resource.mime_type,
                        "text": str(content)
                    }]
                }
            )
        else:
            return JSONRPCResponse(
                id=request.id,
                error={"code": -32603, "message": "Resource has no handler"}
            )
    
    async def _handle_prompts_list(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """List available prompts."""
        with self._lock:
            prompts = [prompt.to_dict() for prompt in self.prompts.values()]
        return JSONRPCResponse(
            id=request.id,
            result={
                "prompts": prompts
            }
        )
    
    async def _handle_prompts_get(self, request: JSONRPCRequest) -> JSONRPCResponse:
        """Get a prompt template."""
        prompt_name = _sanitize(str(request.params.get("name", "")))
        arguments = request.params.get("arguments", {})
        
        if prompt_name not in self.prompts:
            return JSONRPCResponse(
                id=request.id,
                error={
                    "code": -32602,
                    "message": f"Prompt not found: {prompt_name}"
                }
            )
        
        prompt = self.prompts[prompt_name]
        valid_args = {}
        if isinstance(arguments, dict):
            for arg_def in prompt.arguments:
                arg_name = arg_def.get("name", "")
                if arg_name in arguments:
                    valid_args[arg_name] = arguments[arg_name]
        
        if prompt.handler:
            try:
                rendered = await prompt.handler(**valid_args) if asyncio.iscoroutinefunction(prompt.handler) else prompt.handler(**valid_args)
            except (ValueError, KeyError, TypeError, OSError):
                rendered = prompt.template
        else:
            rendered = prompt.template
            for key, value in valid_args.items():
                rendered = rendered.replace(f"{{{key}}}", str(value))
        
        return JSONRPCResponse(
            id=request.id,
            result={
                "messages": [{
                    "role": "user",
                    "content": {"type": "text", "text": rendered}
                }]
            }
        )
    
    async def run_stdio(self):
        """Run MCP server over STDIO (for AI clients like Claude)."""
        logger.info(f"Starting MCP server '{self.name}' on STDIO")
        
        while True:
            try:
                # Read from stdin
                line = await asyncio.get_event_loop().run_in_executor(None, input)
                
                if not line:
                    continue
                
                # Parse request
                request = JSONRPCRequest.from_json(line)
                
                # Handle request
                response = await self.handle_request(request)
                
                # Write response to stdout
                print(response.to_json(), flush=True)
            
            except EOFError:
                logger.info("MCP client disconnected")
                break
            except Exception as e:
                logger.error(f"Error in STDIO loop: {type(e).__name__}")
    
    def send_notification(self, method: str, params: Dict = None):
        """Send a notification to client."""
        notification = JSONRPCNotification(method=method, params=params or {})
        print(notification.to_json(), flush=True)


# Shadow AI MCP Integration

def create_shadow_mcp_server(shadow_system) -> MCPServer:
    """Create MCP server exposing Shadow AI capabilities."""
    
    server = MCPServer(name="shadow-ai", version="1.0.0")
    
    # Tool: Ask Shadow
    async def handle_ask(query: str, domain: str = None) -> str:
        result = await shadow_system.ask(_sanitize_content(query), _sanitize(domain) if domain else None)
        return json.dumps(result, indent=2)
    
    server.register_tool(MCPTool(
        name="shadow_ask",
        description="Ask Shadow AI a question",
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Question to ask"},
                "domain": {"type": "string", "description": "Optional domain (health, finance, career, etc.)"}
            },
            "required": ["query"]
        },
        handler=handle_ask
    ))
    
    # Tool: Track Decision
    def handle_track_decision(context: str, chosen: str, reasoning: str) -> str:
        decision_id = shadow_system.track_decision(
            context=_sanitize_content(context),
            options=[],
            chosen=_sanitize_content(chosen),
            reasoning=_sanitize_content(reasoning)
        )
        return decision_id
    
    server.register_tool(MCPTool(
        name="shadow_track_decision",
        description="Track a decision for learning",
        input_schema={
            "type": "object",
            "properties": {
                "context": {"type": "string", "description": "Decision context"},
                "chosen": {"type": "string", "description": "Choice made"},
                "reasoning": {"type": "string", "description": "Reasoning for choice"}
            },
            "required": ["context", "chosen"]
        },
        handler=handle_track_decision
    ))
    
    # Resource: Current Context
    async def get_current_context() -> str:
        return json.dumps(shadow_system.get_unified_status(), indent=2)
    
    server.register_resource(MCPResource(
        uri="shadow://context/current",
        name="Current Context",
        description="Current Shadow AI context and status",
        handler=get_current_context
    ))
    
    # Resource: Proactive Insights
    async def get_insights() -> str:
        insights = shadow_system.get_proactive_insights()
        return json.dumps(insights, indent=2)
    
    server.register_resource(MCPResource(
        uri="shadow://insights/proactive",
        name="Proactive Insights",
        description="Current proactive insights",
        handler=get_insights
    ))
    
    # Prompt: Decision Help
    def render_decision_prompt(situation: str) -> str:
        return f"""I'm facing a decision: {situation}

Please help me by:
1. Identifying my options
2. Analyzing potential outcomes
3. Recalling similar past decisions
4. Suggesting the best approach

Context:
- Current time: {datetime.now(timezone.utc).strftime('%A at %H:%M')}
- Location: {shadow_system.domains.get_status().get('most_active', 'unknown')} domain focus

Use Shadow Council ability to check for similar past decisions."""
    
    server.register_prompt(MCPPrompt(
        name="decision_help",
        description="Get help making a decision",
        arguments=[
            {"name": "situation", "description": "The decision situation", "required": True}
        ],
        template=render_decision_prompt,
        handler=render_decision_prompt
    ))
    
    # Prompt: Pattern Analysis
    def render_pattern_prompt(query: str) -> str:
        return f"""Analyze patterns for: {query}

Use Pattern Sight ability to identify:
1. Recurring patterns in my data
2. Causal relationships (X causes Y, not just correlation)
3. Temporal patterns (time-based)
4. Cross-domain patterns

Provide specific, actionable insights."""
    
    server.register_prompt(MCPPrompt(
        name="pattern_analysis",
        description="Analyze patterns in your data",
        arguments=[
            {"name": "query", "description": "Pattern to analyze", "required": True}
        ],
        template=render_pattern_prompt,
        handler=render_pattern_prompt
    ))
    
    return server


# Example usage
if __name__ == "__main__":
    # This would be used with actual Shadow AI system
    async def main():
        server = MCPServer(name="shadow-ai-standalone", version="1.0.0")
        
        await server.run_stdio()
    
    asyncio.run(main())