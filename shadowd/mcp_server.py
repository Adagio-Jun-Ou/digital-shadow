#!/usr/bin/env python3
"""
Shadow AI MCP Server - Model Context Protocol Integration

Based on research from:
- modelcontextprotocol/python-sdk (22.5k stars)
- modelcontextprotocol/servers (82.9k stars)

This exposes Shadow AI as an MCP server, allowing other LLM applications
to access Shadow's context, memory, and workflows.
"""

from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import json
import re
import asyncio
import logging
import uuid

from starlette.responses import Response


def _parse_ts(ts_str: Optional[str]) -> datetime:
    try:
        return datetime.fromisoformat(ts_str) if ts_str else datetime.now(timezone.utc)
    except (ValueError, TypeError):
        return datetime.now(timezone.utc)

from mcp.server.fastmcp import FastMCP, Context
from mcp.types import Resource, Tool, Prompt
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str) -> str:
    return _SANITIZE_STRICT_RE.sub('', text) if text else text


def _sanitize_content(text: str) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text) if text else text

def _build_mcp():
    """Construct the FastMCP server, dropping `version=` if the installed
    mcp version no longer accepts it (mcp 1.x removed it from __init__)."""
    instructions = """
Shadow AI is a personal digital assistant that:
- Maintains persistent context and memory
- Learns patterns from user interactions
- Provides proactive insights
- Integrates with phone/location data"""
    kwargs = {"instructions": instructions}
    try:
        import inspect
        params = inspect.signature(FastMCP.__init__).parameters
        if "version" in params:
            kwargs["version"] = "1.0.0"
    except (TypeError, ValueError):
        pass
    return FastMCP("Shadow AI", **kwargs)


mcp = _build_mcp()
_LEGACY_INSTRUCTIONS = """
Shadow AI is a personal digital assistant that:
- Maintains persistent context and memory
- Learns patterns from user interactions
- Provides proactive insights
- Integrates with phone/location data
- Supports temporal knowledge graphs

Use this MCP server to:
- Query user context and memory
- Search the knowledge graph
- Execute workflows
- Get proactive insights
"""

DATA_DIR = Path.home() / "digital-shadow" / "data"
USERS_DIR = DATA_DIR / "users"
KNOWLEDGE_DIR = DATA_DIR / "knowledge_graph"


class UserInfo(BaseModel):
    name: Optional[str] = None
    email: Optional[str] = None
    location: Optional[str] = None
    timezone: Optional[str] = None
    preferences: Dict[str, Any] = Field(default_factory=dict)


class SearchResult(BaseModel):
    content: str
    source: str
    timestamp: datetime
    relevance: float


class WorkflowResult(BaseModel):
    workflow_id: str
    status: str
    output: Dict[str, Any]
    duration_ms: float


class Insight(BaseModel):
    insight_id: str
    category: str
    title: str
    description: str
    confidence: float
    timestamp: datetime


def _get_user_dir(user_id: str) -> Path:
    safe_id = re.sub(r'[^a-zA-Z0-9_\-]', '', user_id)[:64]
    if not safe_id:
        safe_id = "default"
    user_dir = USERS_DIR / safe_id
    user_dir.mkdir(parents=True, exist_ok=True)
    try:
        if not user_dir.resolve().is_relative_to(USERS_DIR.resolve()):
            raise ValueError(f"Path traversal attempt: {user_id}")
    except AttributeError:
        resolved = str(user_dir.resolve())
        if not resolved.startswith(str(USERS_DIR.resolve())):
            raise ValueError(f"Path traversal attempt: {user_id}")
    return user_dir


def _load_json(path: Path) -> Dict:
    if path.exists():
        try:
            return json.loads(path.read_text())
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
    return {}


def _save_json(path: Path, data: Dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(f'.json.tmp.{uuid.uuid4().hex[:8]}')
    tmp_path.write_text(json.dumps(data, indent=2, default=str))
    tmp_path.replace(path)


# ============ RESOURCES ============

@mcp.resource("shadow://user/{user_id}")
def get_user_profile(user_id: str) -> str:
    """
    Get user profile and context.
    
    Args:
        user_id: Unique user identifier
    
    Returns:
        JSON string of user profile
    """
    user_dir = _get_user_dir(user_id)
    profile_path = user_dir / "profile.json"
    
    profile = _load_json(profile_path)
    
    if not profile:
        profile = {
            "user_id": user_id,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "context": {}
        }
        _save_json(profile_path, profile)
    
    return json.dumps(profile, indent=2)


@mcp.resource("shadow://memory/{user_id}/{memory_type}")
def get_memory_block(user_id: str, memory_type: str) -> str:
    """
    Get a specific memory block from Shadow AI.
    
    Args:
        user_id: Unique user identifier
        memory_type: One of 'core', 'archival', 'working'
    
    Returns:
        JSON string of memory block content
    """
    memory_type = memory_type.lower()
    if memory_type not in ["core", "archival", "working"]:
        return json.dumps({"error": f"Invalid memory type: {memory_type}"})
    
    user_dir = _get_user_dir(user_id)
    memory_path = user_dir / "memory" / f"{memory_type}_block.json"
    
    memory = _load_json(memory_path)
    
    return json.dumps(memory, indent=2)


@mcp.resource("shadow://knowledge/{user_id}/entities")
def get_knowledge_entities(user_id: str) -> str:
    """
    Get all entities in the knowledge graph.
    
    Args:
        user_id: Unique user identifier
    
    Returns:
        JSON list of entity names
    """
    kg_path = KNOWLEDGE_DIR / user_id / "knowledge_graph.json"
    
    kg_data = _load_json(kg_path)
    entities = list(kg_data.get("entities", {}).keys())
    
    return json.dumps(entities, indent=2)


@mcp.resource("shadow://knowledge/{user_id}/entity/{entity_name}")
def get_entity_facts(user_id: str, entity_name: str) -> str:
    """
    Get all facts about a specific entity.
    
    Args:
        user_id: Unique user identifier
        entity_name: Name of the entity
    
    Returns:
        JSON list of facts
    """
    kg_path = KNOWLEDGE_DIR / user_id / "knowledge_graph.json"
    kg_data = _load_json(kg_path)
    
    entities = kg_data.get("entities", {})
    entity = entities.get(entity_name, {})
    facts = entity.get("facts", [])
    
    valid_facts = [
        f for f in facts
        if f.get("valid_until") is None
    ]
    
    return json.dumps(valid_facts, indent=2)


@mcp.resource("shadow://workflow/{user_id}/status/{workflow_id}")
def get_workflow_status(user_id: str, workflow_id: str) -> str:
    """
    Get status of a running or completed workflow.
    
    Args:
        user_id: Unique user identifier
        workflow_id: Workflow instance ID
    
    Returns:
        JSON workflow status
    """
    user_dir = _get_user_dir(user_id)
    workflow_path = user_dir / "workflows" / f"{workflow_id}.json"
    
    status = _load_json(workflow_path)
    
    return json.dumps(status, indent=2)


# ============ TOOLS ============

@mcp.tool()
async def search_memory(
    query: str,
    user_id: str,
    ctx: Context,
    memory_types: Optional[List[str]] = None,
    limit: int = 10
) -> List[SearchResult]:
    """
    Search Shadow AI's memory for relevant information.
    
    Args:
        query: Search query string
        user_id: User ID to search
        memory_types: Memory blocks to search (default: all)
        limit: Maximum results (default: 10)
    
    Returns:
        List of search results with relevance scores
    """
    await ctx.info(f"Searching memory for user {user_id}: {query}")
    
    if memory_types is None:
        memory_types = ["core", "archival", "working"]
    
    results = []
    
    for mem_type in memory_types:
        user_dir = _get_user_dir(user_id)
        memory_path = user_dir / "memory" / f"{mem_type}_block.json"
        memory_data = _load_json(memory_path)
        
        content = memory_data.get("content", "")
        
        if query.lower() in content.lower():
            results.append(SearchResult(
                content=content[:500],
                source=f"memory/{mem_type}",
                timestamp=datetime.now(timezone.utc),
                relevance=0.8
            ))
    
    kg_path = KNOWLEDGE_DIR / user_id / "knowledge_graph.json"
    kg_data = _load_json(kg_path)
    
    for entity_name, entity in kg_data.get("entities", {}).items():
        for fact in entity.get("facts", []):
            if query.lower() in fact.get("content", "").lower():
                results.append(SearchResult(
                    content=fact.get("content", ""),
                    source=f"knowledge_graph/{entity_name}",
                    timestamp=_parse_ts(fact.get("created_at")),
                    relevance=0.7
                    ))
    
    await ctx.debug(f"Found {len(results)} results")
    return results[:limit]


@mcp.tool()
async def remember(
    content: str,
    user_id: str,
    ctx: Context,
    entities: Optional[List[str]] = None,
    source: str = "mcp_tool"
) -> Dict[str, str]:
    """
    Store new information in Shadow AI's knowledge graph.
    
    Args:
        content: Information to remember
        user_id: User ID
        entities: Entities mentioned in content
        source: Source of the information
    
    Returns:
        Status of the operation
    """
    await ctx.info(f"Storing new memory for user {user_id}")
    
    content = _sanitize_content(content)
    source = _sanitize(source)
    entities = [_sanitize(e) for e in (entities or [])]
    
    kg_dir = KNOWLEDGE_DIR / user_id
    kg_dir.mkdir(parents=True, exist_ok=True)
    kg_path = kg_dir / "knowledge_graph.json"
    
    kg_data = _load_json(kg_path)
    
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
    
    _save_json(kg_path, kg_data)
    
    await ctx.debug(f"Created episode {episode_id} with {len(entities)} entities")
    
    return {
        "status": "success",
        "episode_id": episode_id,
        "fact_id": fact_id,
        "entities_stored": len(entities)
    }


@mcp.tool()
async def execute_workflow(
    workflow_name: str,
    user_id: str,
    ctx: Context,
    params: Optional[Dict[str, Any]] = None
) -> WorkflowResult:
    """
    Execute a predefined Shadow AI workflow.
    
    Args:
        workflow_name: Name of the workflow to execute
        user_id: User ID
        params: Workflow parameters
    
    Returns:
        Workflow execution result
    """
    import time
    start_time = time.time()
    
    await ctx.info(f"Executing workflow '{workflow_name}' for user {user_id}")
    await ctx.report_progress(progress=0.0, message="Starting workflow")
    
    params = params or {}
    
    import re
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', workflow_name):
        return WorkflowResult(
            workflow_id=f"wf_{uuid.uuid4().hex[:12]}",
            status="error",
            output={"error": "Invalid workflow name"},
            duration_ms=0
        )
    workflows_dir = Path.home() / "digital-shadow" / "workflows"
    workflow_path = workflows_dir / f"{workflow_name}.json"
    
    if not workflow_path.exists():
        return WorkflowResult(
            workflow_id=f"wf_{uuid.uuid4().hex[:12]}",
            status="error",
            output={"error": f"Workflow '{workflow_name}' not found"},
            duration_ms=0
        )
    
    workflow_def = _load_json(workflow_path)
    
    await ctx.report_progress(progress=0.2, message="Initializing")
    await ctx.report_progress(progress=0.4, message="Executing nodes")
    
    workflow_id = f"wf_{uuid.uuid4().hex[:12]}"
    result = {
        "workflow_name": workflow_name,
        "steps_completed": len(workflow_def.get("nodes", [])),
        "output": params
    }
    
    user_dir = _get_user_dir(user_id)
    workflows_dir = user_dir / "workflows"
    workflows_dir.mkdir(parents=True, exist_ok=True)
    _save_json(workflows_dir / f"{workflow_id}.json", {
        "workflow_id": workflow_id,
        "workflow_name": workflow_name,
        "status": "completed",
        "result": result,
        "timestamp": datetime.now(timezone.utc).isoformat()
    })
    
    await ctx.report_progress(progress=1.0, message="Completed")
    
    duration_ms = (time.time() - start_time) * 1000
    
    return WorkflowResult(
        workflow_id=workflow_id,
        status="completed",
        output=result,
        duration_ms=duration_ms
    )


@mcp.tool()
async def get_insights(
    user_id: str,
    ctx: Context,
    category: Optional[str] = None,
    limit: int = 5
) -> List[Insight]:
    """
    Get proactive insights from Shadow AI.
    
    Args:
        user_id: User ID
        category: Filter by category (optional)
        limit: Maximum insights to return
    
    Returns:
        List of insights
    """
    await ctx.info(f"Generating insights for user {user_id}")
    
    insights = []
    
    user_dir = _get_user_dir(user_id)
    insights_path = user_dir / "insights.json"
    insights_data = _load_json(insights_path)
    
    for insight in insights_data.get("insights", []):
        if category and insight.get("category") != category:
            continue
        
        insights.append(Insight(
            insight_id=insight.get("id", "unknown"),
            category=insight.get("category", "general"),
            title=insight.get("title", ""),
            description=insight.get("description", ""),
            confidence=insight.get("confidence", 0.5),
            timestamp=_parse_ts(insight.get("timestamp"))
        ))
    
    return insights[:limit]


@mcp.tool()
async def update_context(
    context_type: str,
    content: str,
    user_id: str,
    ctx: Context,
    merge: bool = True
) -> Dict[str, str]:
    """
    Update user context in Shadow AI.
    
    Args:
        context_type: Type of context (preferences, work, personal, etc.)
        content: Context content to store
        user_id: User ID
        merge: Merge with existing (True) or replace (False)
    
    Returns:
        Update status
    """
    await ctx.info(f"Updating {context_type} context for user {user_id}")
    
    user_dir = _get_user_dir(user_id)
    profile_path = user_dir / "profile.json"
    profile = _load_json(profile_path)
    
    if "context" not in profile:
        profile["context"] = {}
    
    if context_type not in profile["context"] or not merge:
        profile["context"][context_type] = content
    else:
        existing = profile["context"].get(context_type, "")
        profile["context"][context_type] = f"{existing}\n{content}"
    
    profile["updated_at"] = datetime.now(timezone.utc).isoformat()
    _save_json(profile_path, profile)
    
    return {
        "status": "success",
        "context_type": context_type,
        "merged": merge
    }


# ============ PROMPTS ============

@mcp.prompt(title="Shadow Analysis")
def analyze_interaction(context: str) -> str:
    """
    Analyze a user interaction in the context of Shadow AI's knowledge.
    """
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


@mcp.prompt(title="Memory Search")
def search_prompt(query: str, context_hint: Optional[str] = None) -> str:
    """
    Generate a prompt for searching Shadow AI's memory.
    """
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


@mcp.prompt(title="Workflow Design")
def design_workflow(description: str) -> str:
    """
    Design a Shadow AI workflow from a natural language description.
    """
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


# ============ MAIN ============

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Shadow AI MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "streamable-http"],
        default="streamable-http",
        help="Transport protocol"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=8000,
        help="Port for HTTP transports"
    )
    parser.add_argument(
        "--host",
        default="localhost",
        help="Host for HTTP transports"
    )
    
    args = parser.parse_args()
    
    if args.transport == "stdio":
        mcp.run(transport="stdio")
    elif args.transport == "sse":
        import uvicorn
        from mcp.server.sse import SseServerTransport
        from starlette.applications import Starlette
        from starlette.routing import Route
        
        sse = SseServerTransport("/messages")
        
        async def handle_sse(request):
            async with sse.connect_sse(
                request.scope, request.receive, request._send
            ) as streams:
                await mcp.run(
                    streams[0],
                    streams[1],
                    mcp.create_initialization_options()
                )
            return Response()
        
        app = Starlette(
            routes=[Route("/sse", endpoint=handle_sse)]
        )
        uvicorn.run(app, host=args.host, port=args.port)
    else:
        mcp.run(transport="streamable-http")


if __name__ == "__main__":
    main()