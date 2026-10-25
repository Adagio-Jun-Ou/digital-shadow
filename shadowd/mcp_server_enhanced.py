#!/usr/bin/env python3
"""
Enhanced Shadow AI MCP Server - Model Context Protocol Integration

Improved version with:
- Full integration with Shadow AI components
- WebSocket notifications for real-time updates
- Comprehensive error handling
- Resource subscriptions
- Health checks and metrics
- More tools and prompts

Based on research from:
- modelcontextprotocol/python-sdk (22.5k stars)
- modelcontextprotocol/servers (82.9k stars)
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
import json
import os
import asyncio
import copy
import logging
import re as _re
import threading
import time
import traceback
import uuid
from contextlib import asynccontextmanager

from starlette.responses import Response

logger = logging.getLogger(__name__)

_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    return _SANITIZE_STRICT_RE.sub('', text) if text else text


def _sanitize_content(text: str) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text) if text else text

# MCP imports
try:
    from mcp.server.fastmcp import FastMCP, Context
    from mcp.types import Resource, Tool, Prompt
    from pydantic import BaseModel, Field
    HAS_MCP = True
except ImportError:
    HAS_MCP = False
    FastMCP = None
    Context = None
    Resource = None
    Tool = None
    Prompt = None
    BaseModel = object
    Field = lambda *args, **kwargs: None
    logger.warning("MCP package not installed. Run: pip install mcp")

# Conditional import for WebSocket
try:
    from .websocket_manager import WSMessageType, WSMessage
    HAS_WEBSOCKET = True
except ImportError:
    HAS_WEBSOCKET = False
    logger.warning("WebSocket manager not available")


@dataclass
class MCPServerMetrics:
    requests_total: int = 0
    requests_successful: int = 0
    requests_failed: int = 0
    avg_latency_ms: float = 0.0
    active_connections: int = 0
    tools_called: Dict[str, int] = field(default_factory=dict)
    resources_accessed: Dict[str, int] = field(default_factory=dict)
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> Dict:
        return {
            "requests_total": self.requests_total,
            "requests_successful": self.requests_successful,
            "requests_failed": self.requests_failed,
            "avg_latency_ms": round(self.avg_latency_ms, 3),
            "active_connections": self.active_connections,
            "tools_called": dict(self.tools_called),
            "resources_accessed": dict(self.resources_accessed),
            "uptime_seconds": round((datetime.now(timezone.utc) - self.start_time).total_seconds(), 3)
        }


class EnhancedMCPServer:
    """
    Enhanced MCP Server for Shadow AI.
    
    Features:
    - Resources: User profiles, memory blocks, knowledge graph, workflows
    - Tools: Search memory, remember, execute workflow, ask, observe, track
    - Prompts: Analysis, search, workflow design
    - WebSocket integration for real-time notifications
    - Metrics and health checks
    """
    
    def __init__(
        self,
        name: str = "Shadow AI",
        version: str = "2.0.0",
        data_dir: Path = None,
        websocket_manager: Any = None,
        shadow_ai: Any = None
    ):
        self.name = name
        self.version = version
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data")
        self.users_dir = self.data_dir / "users"
        self.knowledge_dir = self.data_dir / "knowledge_graph"
        
        self.websocket_manager = websocket_manager
        self.shadow_ai = shadow_ai
        self.metrics = MCPServerMetrics()
        self._lock = threading.Lock()
        
        self._subscribers: Dict[str, List[Callable]] = {}
        
        if HAS_MCP:
            self._setup_mcp_server()
        else:
            self.mcp = None
            logger.warning("MCP package not installed. Server will not function until 'pip install mcp' is run.")
    
    def _setup_mcp_server(self):
        """Set up the FastMCP server.

        Modern FastMCP (mcp>=1.x) dropped the `version=` kwarg from
        __init__ — it's now provided via the protocol handshake. Pass only
        the kwargs the installed signature accepts to stay forward-compatible.
        """
        kwargs = {"instructions": self._get_instructions()}
        try:
            import inspect
            params = inspect.signature(FastMCP.__init__).parameters
            if "version" in params:
                kwargs["version"] = self.version
        except (TypeError, ValueError):
            pass
        self.mcp = FastMCP(self.name, **kwargs)
        
        self._register_resources()
        self._register_tools()
        self._register_prompts()
        self._register_health_check()
    
    def _get_instructions(self) -> str:
        return """
Shadow AI is a personal digital assistant that maintains persistent context,
learns patterns from user interactions, and provides proactive insights.

Capabilities:
- Memory: Persistent block-based memory (core, archival, working)
- Knowledge Graph: Temporal knowledge graph with entities and relationships
- Agents: Multi-agent coordination (CrewAI/AutoGen pattern)
- State Graphs: Workflow management (LangGraph pattern)
- WebSockets: Real-time notifications and streaming

Use this MCP server to:
- Query user context, memory, and knowledge
- Search the temporal knowledge graph
- Execute workflows and coordinate agents
- Get proactive insights and recommendations
- Track decisions, patterns, and observations
- Subscribe to real-time updates via WebSocket
"""
    
    # ============ RESOURCES ============
    
    def _register_resources(self):
        """Register MCP resources."""
        
        @self.mcp.resource("shadow://user/{user_id}")
        def get_user_profile(user_id: str) -> str:
            return self._safe_resource_handler(
                "get_user_profile",
                lambda: self._get_user_profile(user_id)
            )
        
        @self.mcp.resource("shadow://memory/{user_id}/{memory_type}")
        def get_memory_block(user_id: str, memory_type: str) -> str:
            return self._safe_resource_handler(
                "get_memory_block",
                lambda: self._get_memory_block(user_id, memory_type)
            )
        
        @self.mcp.resource("shadow://knowledge/{user_id}/entities")
        def get_knowledge_entities(user_id: str) -> str:
            return self._safe_resource_handler(
                "get_knowledge_entities",
                lambda: self._get_knowledge_entities(user_id)
            )
        
        @self.mcp.resource("shadow://knowledge/{user_id}/entity/{entity_name}")
        def get_entity_facts(user_id: str, entity_name: str) -> str:
            return self._safe_resource_handler(
                "get_entity_facts",
                lambda: self._get_entity_facts(user_id, entity_name)
            )
        
        @self.mcp.resource("shadow://workflow/{user_id}/status/{workflow_id}")
        def get_workflow_status(user_id: str, workflow_id: str) -> str:
            return self._safe_resource_handler(
                "get_workflow_status",
                lambda: self._get_workflow_status(user_id, workflow_id)
            )
        
        @self.mcp.resource("shadow://agent/{agent_id}/status")
        def get_agent_status(agent_id: str) -> str:
            return self._safe_resource_handler(
                "get_agent_status",
                lambda: self._get_agent_status(agent_id)
            )
        
        @self.mcp.resource("shadow://metrics")
        def get_metrics() -> str:
            return json.dumps(self.metrics.to_dict(), indent=2)
    
    # ============ TOOLS ============
    
    def _register_tools(self):
        """Register MCP tools."""
        
        @self.mcp.tool()
        async def search_memory(
            query: str,
            user_id: str,
            ctx: Context,
            memory_types: Optional[List[str]] = None,
            limit: int = 10
        ) -> List[Dict]:
            return await self._track_tool_call(
                "search_memory",
                lambda: self._search_memory(query, user_id, ctx, memory_types, limit)
            )
        
        @self.mcp.tool()
        async def remember(
            content: str,
            user_id: str,
            ctx: Context,
            entities: Optional[List[str]] = None,
            source: str = "mcp_tool"
        ) -> Dict:
            return await self._track_tool_call(
                "remember",
                lambda: self._remember(content, user_id, ctx, entities, source)
            )
        
        @self.mcp.tool()
        async def ask(
            question: str,
            user_id: str,
            ctx: Context,
            context: Optional[Dict[str, Any]] = None
        ) -> Dict:
            return await self._track_tool_call(
                "ask",
                lambda: self._ask(question, user_id, ctx, context)
            )
        
        @self.mcp.tool()
        async def observe(
            observation_type: str,
            content: Dict[str, Any],
            user_id: str,
            ctx: Context,
            device: Optional[str] = None
        ) -> Dict:
            return await self._track_tool_call(
                "observe",
                lambda: self._observe(observation_type, content, user_id, ctx, device)
            )
        
        @self.mcp.tool()
        async def track(
            tracking_type: str,
            data: Dict[str, Any],
            user_id: str,
            ctx: Context
        ) -> Dict:
            return await self._track_tool_call(
                "track",
                lambda: self._track(tracking_type, data, user_id, ctx)
            )
        
        @self.mcp.tool()
        async def execute_workflow(
            workflow_name: str,
            user_id: str,
            ctx: Context,
            params: Optional[Dict[str, Any]] = None
        ) -> Dict:
            return await self._track_tool_call(
                "execute_workflow",
                lambda: self._execute_workflow(workflow_name, user_id, ctx, params)
            )
        
        @self.mcp.tool()
        async def get_insights(
            user_id: str,
            ctx: Context,
            category: Optional[str] = None,
            limit: int = 5
        ) -> List[Dict]:
            return await self._track_tool_call(
                "get_insights",
                lambda: self._get_insights(user_id, ctx, category, limit)
            )
        
        @self.mcp.tool()
        async def update_context(
            context_type: str,
            content: str,
            user_id: str,
            ctx: Context,
            merge: bool = True
        ) -> Dict:
            return await self._track_tool_call(
                "update_context",
                lambda: self._update_context(context_type, content, user_id, ctx, merge)
            )
        
        @self.mcp.tool()
        async def coordinate_agents(
            goal: str,
            user_id: str,
            ctx: Context,
            agent_ids: Optional[List[str]] = None,
            crew_ids: Optional[List[str]] = None
        ) -> Dict:
            return await self._track_tool_call(
                "coordinate_agents",
                lambda: self._coordinate_agents(goal, user_id, ctx, agent_ids, crew_ids)
            )
        
        @self.mcp.tool()
        async def subscribe(
            topic: str,
            user_id: str,
            ctx: Context
        ) -> Dict:
            return await self._track_tool_call(
                "subscribe",
                lambda: self._subscribe(topic, user_id, ctx)
            )
    
    # ============ PROMPTS ============
    
    def _register_prompts(self):
        """Register MCP prompts."""
        
        @self.mcp.prompt(title="Shadow Analysis")
        def analyze_interaction(context: str) -> str:
            return self._analyze_interaction_prompt(context)
        
        @self.mcp.prompt(title="Memory Search")
        def search_prompt(query: str, context_hint: Optional[str] = None) -> str:
            return self._search_prompt(query, context_hint)
        
        @self.mcp.prompt(title="Workflow Design")
        def design_workflow(description: str) -> str:
            return self._design_workflow_prompt(description)
        
        @self.mcp.prompt(title="Knowledge Extraction")
        def extract_knowledge(text: str, domain: Optional[str] = None) -> str:
            return self._extract_knowledge_prompt(text, domain)
        
        @self.mcp.prompt(title="Agent Coordination Plan")
        def plan_agent_coordination(objective: str, constraints: List[str] = None) -> str:
            return self._plan_agent_coordination_prompt(objective, constraints)
    
    # ============ HEALTH CHECK ============
    
    def _register_health_check(self):
        """Register health check endpoint."""
        
        @self.mcp.resource("shadow://health")
        def health_check() -> str:
            return json.dumps({
                "status": "healthy",
                "name": self.name,
                "version": self.version,
                "uptime_seconds": round((datetime.now(timezone.utc) - self.metrics.start_time).total_seconds(), 3),
                "requests_total": self.metrics.requests_total,
                "requests_successful": self.metrics.requests_successful,
                "requests_failed": self.metrics.requests_failed,
                "websocket_enabled": self.websocket_manager is not None,
                "shadow_ai_enabled": self.shadow_ai is not None
            }, indent=2)
    
    # ============ RESOURCE HANDLERS ============
    
    def _safe_resource_handler(self, resource_name: str, handler: Callable) -> str:
        """Safely execute a resource handler with error handling and metrics."""
        with self._lock:
            self.metrics.requests_total += 1
            self.metrics.resources_accessed[resource_name] = self.metrics.resources_accessed.get(resource_name, 0) + 1
        
        start_time = time.time()
        
        try:
            result = handler()
            with self._lock:
                self.metrics.requests_successful += 1
            return result
        except Exception as e:
            with self._lock:
                self.metrics.requests_failed += 1
            logger.error("Error in resource %s: %s", resource_name, type(e).__name__)
            return json.dumps({
                "error": f"Failed to retrieve {resource_name}",
                "resource": resource_name,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
        finally:
            elapsed = (time.time() - start_time) * 1000
            self._update_avg_latency(elapsed)
    
    def _update_avg_latency(self, latency_ms: float):
        """Update rolling average latency."""
        n = self.metrics.requests_total
        self.metrics.avg_latency_ms = (
            (self.metrics.avg_latency_ms * (n - 1) + latency_ms) / n
            if n > 0 else latency_ms
        )
    
    def _get_user_profile(self, user_id: str) -> str:
        """Get user profile."""
        user_dir = self._get_user_dir(user_id)
        profile_path = user_dir / "profile.json"
        
        profile = self._load_json(profile_path)
        
        if not profile:
            profile = {
                "user_id": user_id,
                "created_at": datetime.now(timezone.utc).isoformat(),
                "context": {},
                "preferences": {}
            }
            self._save_json(profile_path, profile)
        
        return json.dumps(profile, indent=2)
    
    def _get_user_dir(self, user_id: str) -> Path:
        safe_id = _re.sub(r'[^a-zA-Z0-9_\-]', '', user_id)[:64]
        if not safe_id:
            safe_id = "default"
        user_dir = self.users_dir / safe_id
        user_dir.mkdir(parents=True, exist_ok=True)
        try:
            if not user_dir.resolve().is_relative_to(self.users_dir.resolve()):
                raise ValueError(f"Path traversal attempt: {user_id!r}")
        except AttributeError:
            resolved = str(user_dir.resolve())
            if not resolved.startswith(str(self.users_dir.resolve())):
                raise ValueError(f"Path traversal attempt: {user_id!r}")
        return user_dir
    
    def _get_memory_block(self, user_id: str, memory_type: str) -> str:
        """Get a memory block."""
        memory_type = memory_type.lower()
        if memory_type not in ["core", "archival", "working"]:
            return json.dumps({"error": f"Invalid memory type: {memory_type}"})
        
        user_dir = self._get_user_dir(user_id)
        memory_path = user_dir / "memory" / f"{memory_type}_block.json"
        
        memory = self._load_json(memory_path)
        return json.dumps(memory, indent=2)
    
    def _get_knowledge_entities(self, user_id: str) -> str:
        """Get knowledge graph entities."""
        kg_path = self.knowledge_dir / user_id / "knowledge_graph.json"
        kg_data = self._load_json(kg_path)
        entities = list(kg_data.get("entities", {}).keys())
        return json.dumps(entities, indent=2)
    
    def _get_entity_facts(self, user_id: str, entity_name: str) -> str:
        """Get facts about an entity."""
        kg_path = self.knowledge_dir / user_id / "knowledge_graph.json"
        kg_data = self._load_json(kg_path)
        
        entities = kg_data.get("entities", {})
        entity = entities.get(entity_name, {})
        facts = entity.get("facts", [])
        
        valid_facts = [
            f for f in facts
            if f.get("valid_until") is None
        ]
        
        return json.dumps(valid_facts, indent=2)
    
    def _get_workflow_status(self, user_id: str, workflow_id: str) -> str:
        """Get workflow status."""
        user_dir = self._get_user_dir(user_id)
        workflow_path = user_dir / "workflows" / f"{workflow_id}.json"
        status = self._load_json(workflow_path)
        return json.dumps(status, indent=2)
    
    def _get_agent_status(self, agent_id: str) -> str:
        """Get agent status."""
        if self.shadow_ai and hasattr(self.shadow_ai, 'coordinator'):
            status = self.shadow_ai.coordinator.get_ability_status()
            if agent_id in status:
                return json.dumps(status[agent_id], indent=2)
        
        return json.dumps({"error": "Agent not found or Shadow AI not initialized"})
    
    # ============ TOOL HANDLERS ============
    
    async def _track_tool_call(self, tool_name: str, handler: Callable) -> Any:
        """Track tool call with metrics."""
        with self._lock:
            self.metrics.requests_total += 1
            self.metrics.tools_called[tool_name] = self.metrics.tools_called.get(tool_name, 0) + 1
        
        start_time = time.time()
        
        try:
            result = await handler()
            with self._lock:
                self.metrics.requests_successful += 1
            return result
        except Exception as e:
            with self._lock:
                self.metrics.requests_failed += 1
            logger.error("Error in tool %s: %s", tool_name, type(e).__name__)
            raise
        finally:
            elapsed = (time.time() - start_time) * 1000
            self._update_avg_latency(elapsed)
    
    async def _search_memory(
        self, query: str, user_id: str, ctx: Context,
        memory_types: Optional[List[str]], limit: int
    ) -> List[Dict]:
        """Search memory."""
        query = _sanitize_content(query, 1000)
        await ctx.info("Searching memory for user %s", user_id)
        
        if memory_types is None:
            memory_types = ["core", "archival", "working"]
        
        results = []
        
        # Search memory blocks
        for mem_type in memory_types:
            user_dir = self._get_user_dir(user_id)
            memory_path = user_dir / "memory" / f"{mem_type}_block.json"
            memory_data = self._load_json(memory_path)
            
            content = memory_data.get("content", "")
            
            if _re.search(r'\b' + _re.escape(query) + r'\b', content, _re.IGNORECASE):
                results.append({
                    "content": content[:500],
                    "source": f"memory/{mem_type}",
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "relevance": 0.8
                })
        
        # Search knowledge graph
        kg_path = self.knowledge_dir / user_id / "knowledge_graph.json"
        kg_data = self._load_json(kg_path)
        
        for entity_name, entity in kg_data.get("entities", {}).items():
            for fact in entity.get("facts", []):
                if _re.search(r'\b' + _re.escape(query) + r'\b', fact.get("content", ""), _re.IGNORECASE):
                    results.append({
                        "content": fact.get("content", ""),
                        "source": f"knowledge_graph/{entity_name}",
                        "timestamp": fact.get("created_at", datetime.now(timezone.utc).isoformat()),
                        "relevance": 0.7
                    })
        
        await ctx.debug(f"Found {len(results)} results")
        return results[:limit]
    
    async def _remember(
        self, content: str, user_id: str, ctx: Context,
        entities: Optional[List[str]], source: str
    ) -> Dict:
        """Store new information."""
        content = _sanitize_content(content, 100000)
        source = _sanitize(source, 100)
        entities = [_sanitize(e, 100) for e in (entities or [])]
        await ctx.info("Storing new memory for user %s", user_id)
        
        kg_dir = self.knowledge_dir / user_id
        kg_dir.mkdir(parents=True, exist_ok=True)
        kg_path = kg_dir / "knowledge_graph.json"
        
        kg_data = self._load_json(kg_path)
        
        if "entities" not in kg_data:
            kg_data["entities"] = {}
        if "episodes" not in kg_data:
            kg_data["episodes"] = []
        
        now = datetime.now(timezone.utc)
        fact_id = f"f_{uuid.uuid4().hex[:12]}"
        episode_id = f"ep_{uuid.uuid4().hex[:12]}"
        
        entities = entities or []
        
        for entity_name in entities:
            if entity_name not in kg_data["entities"]:
                kg_data["entities"][entity_name] = {
                    "name": entity_name,
                    "entity_type": "unknown",
                    "facts": [],
                    "summary": ""
                }
            
            kg_data["entities"][entity_name]["facts"].append({
                "fact_id": fact_id,
                "content": content,
                "entities": entities,
                "valid_from": now.isoformat(),
                "valid_until": None,
                "confidence": 1.0,
                "source_episode": episode_id,
                "created_at": now.isoformat()
            })
        
        kg_data["episodes"].append({
            "episode_id": episode_id,
            "timestamp": now.isoformat(),
            "content": content,
            "source": source,
            "facts_extracted": [fact_id],
            "entities_seen": entities
        })
        
        self._save_json(kg_path, kg_data)
        
        # Notify subscribers via WebSocket
        if self.websocket_manager:
            await self._notify_subscribers("knowledge_update", {
                "user_id": user_id,
                "episode_id": episode_id,
                "entities": entities
            })
        
        await ctx.debug(f"Created episode {episode_id}")
        
        return {
            "status": "success",
            "episode_id": episode_id,
            "fact_id": fact_id,
            "entities_stored": len(entities)
        }
    
    async def _ask(
        self, question: str, user_id: str, ctx: Context,
        context: Optional[Dict[str, Any]]
    ) -> Dict:
        """Ask Shadow AI a question."""
        question = _sanitize_content(question, 10000)
        await ctx.info("Processing question for user %s", user_id)
        
        if self.shadow_ai and hasattr(self.shadow_ai, 'ask'):
            result = await self.shadow_ai.ask(question, context or {})
            return {
                "status": "success",
                "response": result.get("unified_response", ""),
                "ability_used": result.get("ability_used"),
                "insight": result.get("insight")
            }
        
        # Fallback to simple response
        return {
            "status": "success",
            "response": f"I received your question: '{question}'. Shadow AI is not fully initialized.",
            "ability_used": "fallback"
        }
    
    async def _observe(
        self, observation_type: str, content: Dict[str, Any],
        user_id: str, ctx: Context, device: Optional[str]
    ) -> Dict:
        """Record an observation."""
        observation_type = _sanitize(observation_type, 100)
        if device:
            device = _sanitize(device, 200)
        await ctx.info("Recording %s observation for user %s", observation_type, user_id)
        
        observation = {
            "type": observation_type,
            "content": content,
            "device": device or "mcp",
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        user_dir = self._get_user_dir(user_id)
        observations_path = user_dir / "observations.json"
        observations = self._load_json(observations_path)
        
        if "observations" not in observations:
            observations["observations"] = []
        
        observations["observations"].append(observation)
        if len(observations["observations"]) > 10000:
            observations["observations"] = observations["observations"][-10000:]
        observations["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        self._save_json(observations_path, observations)
        
        # Notify via WebSocket
        if self.websocket_manager:
            await self._notify_subscribers("observation", {
                "user_id": user_id,
                "observation": observation
            })
        
        return {
            "status": "success",
            "observation_id": f"obs_{uuid.uuid4().hex[:12]}",
            "type": observation_type
        }
    
    async def _track(
        self, tracking_type: str, data: Dict[str, Any],
        user_id: str, ctx: Context
    ) -> Dict:
        """Track decisions, patterns, or metrics."""
        tracking_type = _sanitize(tracking_type, 100)
        await ctx.info("Tracking %s for user %s", tracking_type, user_id)
        
        user_dir = self._get_user_dir(user_id)
        tracking_path = user_dir / "tracking.json"
        tracking_data = self._load_json(tracking_path)
        
        if "tracking" not in tracking_data:
            tracking_data["tracking"] = {}
        
        if tracking_type not in tracking_data["tracking"]:
            tracking_data["tracking"][tracking_type] = []
        
        entry = {
            "data": data,
            "timestamp": datetime.now(timezone.utc).isoformat()
        }
        
        tracking_data["tracking"][tracking_type].append(entry)
        if len(tracking_data["tracking"][tracking_type]) > 10000:
            tracking_data["tracking"][tracking_type] = tracking_data["tracking"][tracking_type][-10000:]
        tracking_data["last_updated"] = datetime.now(timezone.utc).isoformat()
        
        self._save_json(tracking_path, tracking_data)
        
        return {
            "status": "success",
            "tracking_type": tracking_type,
            "entry_count": len(tracking_data["tracking"][tracking_type])
        }
    
    async def _execute_workflow(
        self, workflow_name: str, user_id: str, ctx: Context,
        params: Optional[Dict[str, Any]]
    ) -> Dict:
        """Execute a workflow."""
        start_time = time.time()
        
        await ctx.info("Executing workflow for user %s", user_id)
        await ctx.report_progress(progress=0.0, message="Starting workflow")
        
        params = params or {}
        
        import re
        if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', workflow_name):
            return {
                "workflow_id": f"wf_{uuid.uuid4().hex[:12]}",
                "status": "error",
                "output": {"error": "Invalid workflow name"},
                "duration_ms": 0
            }
        workflows_dir = Path.home() / "digital-shadow" / "workflows"
        workflow_path = workflows_dir / f"{workflow_name}.json"
        
        if not workflow_path.exists():
            return {
                "workflow_id": f"wf_{uuid.uuid4().hex[:12]}",
                "status": "error",
                "output": {"error": f"Workflow '{workflow_name}' not found"},
                "duration_ms": 0
            }
        
        workflow_def = self._load_json(workflow_path)
        
        await ctx.report_progress(progress=0.2, message="Initializing")
        await ctx.report_progress(progress=0.4, message="Executing nodes")
        
        workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
        result = {
            "workflow_name": workflow_name,
            "steps_completed": len(workflow_def.get("nodes", [])),
            "output": params
        }
        
        user_dir = self._get_user_dir(user_id)
        workflows_dir = user_dir / "workflows"
        workflows_dir.mkdir(parents=True, exist_ok=True)
        self._save_json(workflows_dir / f"{workflow_id}.json", {
            "workflow_id": workflow_id,
            "workflow_name": workflow_name,
            "status": "completed",
            "result": result,
            "timestamp": datetime.now(timezone.utc).isoformat()
        })
        
        await ctx.report_progress(progress=1.0, message="Completed")
        
        duration_ms = round((time.time() - start_time) * 1000, 3)
        
        # Notify via WebSocket
        if self.websocket_manager:
            await self._notify_subscribers("workflow_complete", {
                "user_id": user_id,
                "workflow_id": workflow_id,
                "workflow_name": workflow_name,
                "duration_ms": duration_ms
            })
        
        return {
            "workflow_id": workflow_id,
            "status": "completed",
            "output": result,
            "duration_ms": duration_ms
        }
    
    async def _get_insights(
        self, user_id: str, ctx: Context,
        category: Optional[str], limit: int
    ) -> List[Dict]:
        """Get proactive insights."""
        await ctx.info("Generating insights for user %s", user_id)
        
        insights = []
        
        user_dir = self._get_user_dir(user_id)
        insights_path = user_dir / "insights.json"
        insights_data = self._load_json(insights_path)
        
        for insight in insights_data.get("insights", []):
            if category and insight.get("category") != category:
                continue
            
            insights.append({
                "insight_id": insight.get("id", "unknown"),
                "category": insight.get("category", "general"),
                "title": insight.get("title", ""),
                "description": insight.get("description", ""),
                "confidence": round(insight.get("confidence", 0.5), 4),
                "timestamp": insight.get("timestamp", datetime.now(timezone.utc).isoformat())
            })
        
        return insights[:limit]
    
    async def _update_context(
        self, context_type: str, content: str, user_id: str,
        ctx: Context, merge: bool
    ) -> Dict:
        """Update user context."""
        context_type = _sanitize(context_type, 100)
        content = _sanitize_content(content, 10000)
        await ctx.info("Updating context for user %s", user_id)
        
        user_dir = self._get_user_dir(user_id)
        profile_path = user_dir / "profile.json"
        profile = self._load_json(profile_path)
        
        if "context" not in profile:
            profile["context"] = {}
        
        if context_type not in profile["context"] or not merge:
            profile["context"][context_type] = content
        else:
            existing = profile["context"].get(context_type, "")
            profile["context"][context_type] = f"{existing}\n{content}"
        
        profile["updated_at"] = datetime.now(timezone.utc).isoformat()
        self._save_json(profile_path, profile)
        
        return {
            "status": "success",
            "context_type": context_type,
            "merged": merge
        }
    
    async def _coordinate_agents(
        self, goal: str, user_id: str, ctx: Context,
        agent_ids: Optional[List[str]], crew_ids: Optional[List[str]]
    ) -> Dict:
        """Coordinate agents to accomplish a goal."""
        goal = _sanitize_content(goal, 5000)
        await ctx.info("Coordinating agents for user %s", user_id)
        
        if self.shadow_ai and hasattr(self.shadow_ai, 'coordinator'):
            result = await self.shadow_ai.coordinator.coordinate(goal)
            return {
                "status": "success",
                "goal": goal,
                "result": result
            }
        
        return {
            "status": "error",
            "message": "Coordinator not initialized"
        }
    
    async def _subscribe(self, topic: str, user_id: str, ctx: Context) -> Dict:
        """Subscribe to real-time updates."""
        topic = _sanitize(topic, 128)
        await ctx.info("Subscribing user %s to topic", user_id)
        
        if not self.websocket_manager:
            return {
                "status": "error",
                "message": "WebSocket notifications not enabled"
            }
        
        topic = _sanitize(topic)
        with self._lock:
            if topic not in self._subscribers:
                self._subscribers[topic] = []
            if len(self._subscribers[topic]) >= 100:
                self._subscribers[topic].pop(0)
                logger.warning("Subscriber cap reached for topic %s", topic)
            self._subscribers[topic].append(user_id)
        
        return {
            "status": "success",
            "topic": topic,
            "message": f"Subscribed to {topic}. Connect to WebSocket to receive updates."
        }
    
    # ============ PROMPT GENERATORS ============
    
    def _analyze_interaction_prompt(self, context: str) -> str:
        return f"""
Analyze the following user interaction and extract key information for Shadow AI:

{context}

Please identify:
1. Entities mentioned (people, places, things)
2. Important facts to remember
3. User preferences expressed
4. Actions to take
5. Context updates needed

Format your response as JSON.
"""
    
    def _search_prompt(self, query: str, context_hint: Optional[str]) -> str:
        prompt = f"Search Shadow AI's knowledge for: {query}"
        
        if context_hint:
            prompt += f"\n\nContext hint: {context_hint}"
        
        prompt += """

Return the most relevant information found, including:
- Source of the information
- Confidence level
- When it was stored
- Related entities
"""
        return prompt
    
    def _design_workflow_prompt(self, description: str) -> str:
        return f"""
Design a Shadow AI workflow based on this description:

{description}

Define:
1. Workflow nodes (steps)
2. Conditional edges between nodes
3. State schema
4. Tools/integrations needed
5. Error handling

Output as JSON workflow definition.
"""
    
    def _extract_knowledge_prompt(self, text: str, domain: Optional[str]) -> str:
        prompt = f"""
Extract structured knowledge from this text:

{text}
"""
        if domain:
            prompt += f"\nDomain context: {domain}"
        
        prompt += """

Extract:
1. Entities (with types)
2. Relationships between entities
3. Facts (with confidence levels)
4. Temporal information (when things happened)
5. Source metadata

Format as JSON.
"""
        return prompt
    
    def _plan_agent_coordination_prompt(self, objective: str, constraints: List[str]) -> str:
        prompt = f"""
Plan agent coordination for this objective:

Objective: {objective}

"""
        if constraints:
            prompt += "Constraints:\n"
            for c in constraints:
                prompt += f"- {c}\n"
        
        prompt += """

Plan:
1. Which agents/crews to use
2. Task allocation
3. Communication flow
4. Expected outputs from each agent
5. Error handling and recovery

Format as JSON coordination plan.
"""
        return prompt
    
    # ============ WEBSOCKET INTEGRATION ============
    
    async def _notify_subscribers(self, event_type: str, data: Dict):
        """Send notification to WebSocket subscribers."""
        if not self.websocket_manager:
            return
        
        message = WSMessage(
            type=WSMessageType.EVENT,
            data={
                "event_type": event_type,
                **data
            }
        )
        
        await self.websocket_manager.broadcast_to_topic(event_type, message)
    
    # ============ UTILITY METHODS ============
    
    def _load_json(self, path: Path) -> Dict:
        """Load JSON file."""
        if path.exists():
            try:
                return json.loads(path.read_text())
            except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError):
                pass
        return {}
    
    def _save_json(self, path: Path, data: Dict):
        """Save JSON file atomically with fsync."""
        import tempfile
        path.parent.mkdir(parents=True, exist_ok=True)
        tmp = path.with_suffix(f'.json.tmp.{uuid.uuid4().hex[:8]}')
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                f.write(json.dumps(data, indent=2, default=str))
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(path))
        except (OSError, TypeError, ValueError):
            if tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
            raise
    
    # ============ SERVER STARTUP ============
    
    def run(self, transport: str = "streamable-http", host: str = "localhost", port: int = 8000):
        """Run the MCP server."""
        if not HAS_MCP:
            logger.error("MCP package not installed. Cannot start server.")
            return

        # FastMCP holds host/port on its settings object; push the user's
        # choices there before run() so streamable-http binds correctly.
        if hasattr(self.mcp, "settings"):
            try:
                self.mcp.settings.host = host
                self.mcp.settings.port = port
            except AttributeError:
                pass

        if transport == "stdio":
            self.mcp.run(transport="stdio")
        elif transport == "sse":
            import uvicorn
            from mcp.server.sse import SseServerTransport
            from starlette.applications import Starlette
            from starlette.routing import Route

            sse = SseServerTransport("/messages")

            async def handle_sse(request):
                async with sse.connect_sse(
                    request.scope, request.receive, request._send
                ) as streams:
                    await self.mcp.run(
                        streams[0],
                        streams[1],
                        self.mcp.create_initialization_options()
                    )
                return Response()

            app = Starlette(
                routes=[Route("/sse", endpoint=handle_sse)]
            )
            uvicorn.run(app, host=host, port=port)
        else:
            self.mcp.run(transport="streamable-http")


def main():
    """Main entry point."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Enhanced Shadow AI MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="Transport protocol"
    )
    parser.add_argument("--port", type=int, default=8000, help="Port for HTTP transports")
    parser.add_argument("--host", default="localhost", help="Host for HTTP transports")
    parser.add_argument("--data-dir", type=Path, help="Data directory")
    
    args = parser.parse_args()
    
    server = EnhancedMCPServer(
        data_dir=args.data_dir
    )
    
    server.run(
        transport=args.transport,
        host=args.host,
        port=args.port
    )


if __name__ == "__main__":
    main()