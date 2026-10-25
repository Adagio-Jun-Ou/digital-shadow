#!/usr/bin/env python3
"""
Shadow AI MCP Client - Connect to External MCP Servers

Based on research from:
- modelcontextprotocol/python-sdk (22.5k stars)

This allows Shadow AI to connect to and use external MCP servers
for additional capabilities (file systems, databases, APIs, etc.)
"""

from typing import Dict, List, Optional, Any, Callable
from dataclasses import dataclass, field
from pathlib import Path
import asyncio
import copy
import json
import logging
import re
import shlex
import threading

logger = logging.getLogger(__name__)

_ALLOWED_MCP_COMMANDS = {
    "mcp-server-filesystem", "mcp-server-github", "mcp-server-brave-search",
    "mcp-server-sqlite", "mcp-server-fetch", "mcp-server-git",
    "mcp-server-puppeteer", "mcp-server-sequential-thinking",
}

_DANGEROUS_ENV_VARS = {
    "PATH", "LD_PRELOAD", "LD_LIBRARY_PATH", "PYTHONPATH", "HOME",
    "PYTHONSTARTUP", "PYTHONHOME", "HOSTNAME", "SHELL", "EDITOR",
}

_SAFE_SERVER_NAME_RE = re.compile(r'^[a-zA-Z0-9_\-]{1,64}$')

_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    return _SANITIZE_STRICT_RE.sub('', text) if text else text


def _sanitize_content(text: str) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text) if text else text


def _validate_server_name(name: str) -> str:
    """Validate and sanitize MCP server name."""
    if not name or not _SAFE_SERVER_NAME_RE.match(name):
        raise ValueError("Invalid server name: must be 1-64 alphanumeric/hyphen/underscore chars")
    return name


def _validate_mcp_command(command: str) -> bool:
    """Validate MCP server command is from a known-safe set."""
    base = Path(shlex.split(command)[0]).name if command else ""
    return base in _ALLOWED_MCP_COMMANDS


def _sanitize_env(env: Dict[str, str]) -> Dict[str, str]:
    """Remove dangerous environment variables from MCP server env."""
    safe = {}
    for key, value in env.items():
        if key.upper() not in _DANGEROUS_ENV_VARS and not key.startswith("_"):
            safe[key] = value
    return safe


@dataclass
class MCPTool:
    name: str
    description: str
    input_schema: Dict[str, Any]
    handler: Optional[Callable] = None

    def __post_init__(self):
        self.name = _sanitize(self.name)
        self.description = _sanitize_content(self.description)


@dataclass
class MCPResource:
    uri: str
    name: str
    description: str
    mime_type: Optional[str] = None

    def __post_init__(self):
        self.uri = _sanitize_content(self.uri)
        self.name = _sanitize(self.name)
        self.description = _sanitize_content(self.description)


@dataclass
class MCPPrompt:
    name: str
    description: str
    arguments: List[Dict[str, Any]]

    def __post_init__(self):
        self.name = _sanitize(self.name)
        self.description = _sanitize_content(self.description)


@dataclass
class MCPServerInfo:
    name: str
    version: str
    instructions: str
    tools: List[MCPTool] = field(default_factory=list)
    resources: List[MCPResource] = field(default_factory=list)
    prompts: List[MCPPrompt] = field(default_factory=list)

    def __post_init__(self):
        self.name = _sanitize(self.name)
        self.version = _sanitize(self.version)
        self.instructions = _sanitize_content(self.instructions)


class MCPClient:
    """
    Client for connecting to MCP servers.
    """

    MAX_SERVERS = 50

    def __init__(self):
        self._servers: Dict[str, Any] = {}
        self._sessions: Dict[str, Any] = {}
        self._lock = threading.Lock()

    async def connect_stdio(
        self,
        server_name: str,
        command: str,
        args: List[str] = None,
        env: Dict[str, str] = None
    ) -> str:
        """
        Connect to an MCP server over stdio.

        Args:
            server_name: Friendly name for this server
            command: Command to run (e.g., "mcp-server-filesystem")
            args: Arguments for the command
            env: Environment variables

        Returns:
            Server ID (same as server_name)
        """
        _validate_server_name(server_name)

        with self._lock:
            if len(self._sessions) >= self.MAX_SERVERS:
                raise RuntimeError(f"Maximum connected servers ({self.MAX_SERVERS}) reached")
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client
        
        args = args or []
        env = _sanitize_env(env or {})
        
        if not _validate_mcp_command(command):
            raise ValueError(f"MCP command not in allowed list: {command}")
        
        server_params = StdioServerParameters(
            command=command,
            args=args,
            env=env
        )
        
        read, write = await stdio_client(server_params)
        session = ClientSession(read, write)
        await session.initialize()
        
        with self._lock:
            self._servers[server_name] = {
                "transport": "stdio",
                "command": command,
                "args": args
            }
            self._sessions[server_name] = session
        
        logger.info("Connected to MCP server: %s", server_name)
        return server_name
    
    async def connect_http(
        self,
        server_name: str,
        url: str,
        headers: Dict[str, str] = None
    ) -> str:
        """
        Connect to an MCP server over HTTP.

        Args:
            server_name: Friendly name for this server
            url: Server URL
            headers: Optional headers

        Returns:
            Server ID (same as server_name)
        """
        _validate_server_name(server_name)

        with self._lock:
            if len(self._sessions) >= self.MAX_SERVERS:
                raise RuntimeError(f"Maximum connected servers ({self.MAX_SERVERS}) reached")
        from mcp import ClientSession
        from mcp.client.streamable_http import streamablehttp_client
        
        headers = headers or {}
        
        client = streamablehttp_client(url, headers)
        read, write = await client.__aenter__()
        
        session = ClientSession(read, write)
        await session.initialize()
        
        with self._lock:
            self._servers[server_name] = {
                "transport": "http",
                "url": url,
                "headers": headers
            }
            self._sessions[server_name] = session
        
        logger.info("Connected to MCP server via HTTP: %s", server_name)
        return server_name
    
    async def disconnect(self, server_name: str):
        """Disconnect from a server."""
        with self._lock:
            if server_name in self._sessions:
                session = self._sessions.pop(server_name)
                self._servers.pop(server_name, None)
            else:
                session = None
        if session:
            await session.close()
            logger.info("Disconnected from: %s", server_name)
    
    async def list_tools(self, server_name: str) -> List[MCPTool]:
        """List available tools from a server."""
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"Not connected to server: {server_name}")
        
        result = await session.list_tools()
        
        tools = []
        for tool in result.tools:
            tools.append(MCPTool(
                name=tool.name,
                description=tool.description or "",
                input_schema=tool.inputSchema or {}
            ))
        
        return tools
    
    async def list_resources(self, server_name: str) -> List[MCPResource]:
        """List available resources from a server."""
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"Not connected to server: {server_name}")
        
        result = await session.list_resources()
        
        resources = []
        for resource in result.resources:
            resources.append(MCPResource(
                uri=resource.uri,
                name=resource.name or resource.uri,
                description=resource.description or "",
                mime_type=resource.mimeType
            ))
        
        return resources
    
    async def list_prompts(self, server_name: str) -> List[MCPPrompt]:
        """List available prompts from a server."""
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"Not connected to server: {server_name}")
        
        result = await session.list_prompts()
        
        prompts = []
        for prompt in result.prompts:
            prompts.append(MCPPrompt(
                name=prompt.name,
                description=prompt.description or "",
                arguments=[
                    {"name": arg.name, "required": arg.required}
                    for arg in (prompt.arguments or [])
                ]
            ))
        
        return prompts
    
    async def call_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Call a tool on a server.
        
        Args:
            server_name: Server to call
            tool_name: Tool to invoke
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"Not connected to server: {server_name}")
        
        result = await session.call_tool(tool_name, arguments)
        
        if result.isError:
            raise RuntimeError(f"Tool error: {result.content}")
        
        if hasattr(result, 'structuredContent'):
            return result.structuredContent
        
        content = result.content
        if isinstance(content, list) and len(content) == 1:
            item = content[0]
            if hasattr(item, 'text'):
                return item.text
        
        return content
    
    async def read_resource(
        self,
        server_name: str,
        uri: str
    ) -> str:
        """
        Read a resource from a server.
        
        Args:
            server_name: Server to read from
            uri: Resource URI
        
        Returns:
            Resource content as string
        """
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"Not connected to server: {server_name}")
        
        result = await session.read_resource(uri)
        
        contents = result.contents
        if isinstance(contents, list) and len(contents) > 0:
            if hasattr(contents[0], 'text'):
                return contents[0].text
        
        return str(contents)
    
    async def get_prompt(
        self,
        server_name: str,
        prompt_name: str,
        arguments: Dict[str, str] = None
    ) -> str:
        """
        Get a prompt from a server.
        
        Args:
            server_name: Server to query
            prompt_name: Prompt to get
            arguments: Prompt arguments
        
        Returns:
            Prompt text
        """
        session = self._sessions.get(server_name)
        if not session:
            raise ValueError(f"Not connected to server: {server_name}")
        
        arguments = arguments or {}
        result = await session.get_prompt(prompt_name, arguments)
        
        messages = result.messages
        if messages:
            text_parts = []
            for msg in messages:
                if hasattr(msg, 'content') and hasattr(msg.content, 'text'):
                    text_parts.append(msg.content.text)
            return "\n".join(text_parts)
        
        return str(result)
    
    def get_server_info(self, server_name: str) -> Optional[MCPServerInfo]:
        """Get information about a connected server."""
        with self._lock:
            session = self._sessions.get(server_name)
            if not session:
                return None
            server_data = self._servers.get(server_name, {})
        
        return MCPServerInfo(
            name=server_name,
            version=server_data.get("version", "unknown"),
            instructions=server_data.get("instructions", ""),
            tools=[],
            resources=[],
            prompts=[]
        )
    
    def list_connected_servers(self) -> List[str]:
        """List all connected servers."""
        with self._lock:
            return list(self._sessions.keys())


class MCPRegistry:
    """
    Registry for discovering and connecting to MCP servers.
    
    Maintains a list of known MCP servers and provides
    easy connection methods.
    """
    
    def __init__(self):
        self._known_servers: Dict[str, Dict] = {}
        self._load_builtins()
    
    def _load_builtins(self):
        """Load known MCP servers."""
        self._known_servers = {
            "filesystem": {
                "command": "mcp-server-filesystem",
                "description": "Filesystem operations",
                "category": "core"
            },
            "memory": {
                "command": "mcp-server-memory",
                "description": "Knowledge graph memory",
                "category": "core"
            },
            "git": {
                "command": "mcp-server-git",
                "description": "Git operations",
                "category": "development"
            },
            "github": {
                "command": "mcp-server-github",
                "description": "GitHub API",
                "category": "development"
            },
            "fetch": {
                "command": "mcp-server-fetch",
                "description": "Web content fetching",
                "category": "web"
            },
            "postgres": {
                "command": "mcp-server-postgres",
                "description": "PostgreSQL database",
                "category": "database"
            },
            "sqlite": {
                "command": "mcp-server-sqlite",
                "description": "SQLite database",
                "category": "database"
            },
            "slack": {
                "command": "mcp-server-slack",
                "description": "Slack integration",
                "category": "communication"
            },
            "puppeteer": {
                "command": "mcp-server-puppeteer",
                "description": "Browser automation",
                "category": "automation"
            },
            "sequential-thinking": {
                "command": "mcp-server-sequential-thinking",
                "description": "Step-by-step reasoning",
                "category": "ai"
            }
        }
    
    def list_servers(self, category: str = None) -> List[Dict]:
        """List available MCP servers."""
        servers = []
        for name, info in self._known_servers.items():
            if category and info.get("category") != category:
                continue
            servers.append({
                "name": name,
                **info
            })
        return servers
    
    def get_server(self, name: str) -> Optional[Dict]:
        """Get information about a specific server."""
        return self._known_servers.get(name)
    
    def register_server(
        self,
        name: str,
        command: str,
        description: str,
        category: str = "custom"
    ):
        """Register a custom MCP server."""
        if not _validate_mcp_command(command):
            raise ValueError(f"MCP command not in allowed list: {command}")
        name = _sanitize(name)
        description = _sanitize_content(description)
        category = _sanitize(category)
        self._known_servers[name] = {
            "command": command,
            "description": description,
            "category": category
        }


class ShadowMCPIntegration:
    """
    Integration layer between Shadow AI and MCP servers.
    
    This allows Shadow AI to use external MCP servers as
    additional capabilities/tools.
    """
    
    def __init__(self):
        self.client = MCPClient()
        self.registry = MCPRegistry()
        self._initialized_servers: Dict[str, Any] = {}
    
    async def connect_server(
        self,
        server_name: str,
        args: List[str] = None,
        env: Dict[str, str] = None
    ) -> bool:
        """
        Connect to an MCP server by name.
        
        Args:
            server_name: Name from registry
            args: Additional arguments
            env: Environment variables
        
        Returns:
            True if connected successfully
        """
        server_info = self.registry.get_server(server_name)
        if not server_info:
            logger.error(f"Unknown server: {server_name}")
            return False
        
        try:
            await self.client.connect_stdio(
                server_name=server_name,
                command=server_info["command"],
                args=args or [],
                env=env or {}
            )
            
            tools = await self.client.list_tools(server_name)
            
            self._initialized_servers[server_name] = {
                "info": server_info,
                "tools": tools
            }
            
            logger.info("Connected to %s with %d tools", server_name, len(tools))
            return True
            
        except (ConnectionError, OSError, ValueError, RuntimeError) as e:
            logger.error("Failed to connect to %s: %s", server_name, type(e).__name__)
            return False
    
    async def execute_tool(
        self,
        server_name: str,
        tool_name: str,
        arguments: Dict[str, Any]
    ) -> Any:
        """
        Execute a tool on an MCP server.
        
        Args:
            server_name: MCP server name
            tool_name: Tool to execute
            arguments: Tool arguments
        
        Returns:
            Tool result
        """
        if server_name not in self._initialized_servers:
            raise ValueError(f"Server not connected: {server_name}")
        
        return await self.client.call_tool(server_name, tool_name, arguments)
    
    async def get_all_tools(self) -> Dict[str, List[MCPTool]]:
        """Get all tools from all connected servers."""
        all_tools = {}
        
        for server_name in self._initialized_servers:
            tools = await self.client.list_tools(server_name)
            all_tools[server_name] = tools
        
        return all_tools
    
    def get_tool_schema(self) -> Dict[str, Any]:
        """
        Get a schema of all available MCP tools.
        
        This can be used to expose MCP tools to Shadow AI's
        workflow system.
        """
        schema = {
            "servers": {},
            "tools_by_category": {}
        }
        
        for server_name, data in self._initialized_servers.items():
            server_info = data["info"]
            tools = data["tools"]
            
            schema["servers"][server_name] = {
                "description": server_info.get("description", ""),
                "category": server_info.get("category", ""),
                "tool_count": len(tools)
            }
            
            for tool in tools:
                category = server_info.get("category", "general")
                if category not in schema["tools_by_category"]:
                    schema["tools_by_category"][category] = []
                
                schema["tools_by_category"][category].append({
                    "server": server_name,
                    "name": tool.name,
                    "description": tool.description,
                    "input_schema": copy.deepcopy(tool.input_schema) if isinstance(tool.input_schema, dict) else tool.input_schema
                })
        
        return schema


async def demo():
    """Demo MCP client usage."""
    
    integration = ShadowMCPIntegration()
    
    print("Available MCP servers:")
    for server in integration.registry.list_servers():
        print(f"  - {server['name']}: {server['description']}")
    
    print("\nTo connect to a server:")
    print("  await integration.connect_server('filesystem', args=['/home/user'])")
    print("  tools = await integration.get_all_tools()")
    print("  result = await integration.execute_tool('filesystem', 'read_file', {'path': '/home/user/file.txt'})")


if __name__ == "__main__":
    asyncio.run(demo())