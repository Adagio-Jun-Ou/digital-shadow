#!/usr/bin/env python3
"""
Shadow AI MCP Client Examples

Examples of using the MCP client to connect to Shadow AI MCP server.

Usage:
    python examples/mcp_client_example.py
"""

import asyncio
import sys
from pathlib import Path

# Add project to path
sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from shadowd.mcp_client import MCPClient
    HAS_MCP_CLIENT = True
except ImportError:
    HAS_MCP_CLIENT = False
    print("MCP package not installed. Run: pip install mcp")


async def example_connect_stdio():
    """Example: Connect to Shadow AI via stdio."""
    print("\n=== Example: Connect via stdio ===\n")
    
    client = MCPClient()
    
    try:
        # Connect to local MCP server
        server_id = await client.connect_stdio(
            server_name="shadow_local",
            command="python",
            args=["-m", "shadowd.run_mcp_server", "--transport", "stdio"]
        )
        
        print(f"Connected to server: {server_id}")
        
        # List available tools
        tools = await client.list_tools(server_id)
        print(f"\nAvailable tools ({len(tools)}):")
        for tool in tools:
            print(f"  - {tool.name}: {tool.description[:50]}...")
        
        # Call a tool
        print("\n--- Calling search_memory ---")
        result = await client.call_tool(
            server_id,
            "search_memory",
            arguments={
                "query": "Python programming",
                "user_id": "example_user",
                "limit": 5
            }
        )
        
        print(f"Result: {result}")
        
        # List resources
        print("\n--- Listing resources ---")
        resources = await client.list_resources(server_id)
        for resource in resources:
            print(f"  - {resource.uri}")
        
        # Read a resource
        print("\n--- Reading resource ---")
        profile = await client.read_resource(server_id, "shadow://user/example_user")
        print(f"Profile: {profile.content[:200]}...")
        
    finally:
        await client.disconnect(server_id)
        print("\nDisconnected.")


async def example_connect_http():
    """Example: Connect to Shadow AI via HTTP."""
    print("\n=== Example: Connect via HTTP ===\n")
    
    client = MCPClient()
    
    try:
        # Connect to HTTP server
        server_id = await client.connect_http(
            server_name="shadow_http",
            url="http://localhost:8000/mcp",
            headers={"Authorization": "Bearer your-token"}
        )
        
        print(f"Connected to HTTP server: {server_id}")
        
        # Use tools
        print("\n--- Using tools ---")
        
        # Remember something
        result = await client.call_tool(
            server_id,
            "remember",
            arguments={
                "content": "User Alice prefers dark mode and Python",
                "user_id": "example_user",
                "entities": ["Alice", "dark mode", "Python"]
            }
        )
        print(f"Remember result: {result}")
        
        # Ask a question
        result = await client.call_tool(
            server_id,
            "ask",
            arguments={
                "question": "What do you know about my preferences?",
                "user_id": "example_user"
            }
        )
        print(f"Ask result: {result}")
        
        # Get insights
        result = await client.call_tool(
            server_id,
            "get_insights",
            arguments={
                "user_id": "example_user",
                "limit": 5
            }
        )
        print(f"Insights: {result}")
        
    finally:
        await client.disconnect(server_id)
        print("Disconnected.")


async def example_workflow():
    """Example: Execute a workflow."""
    print("\n=== Example: Execute Workflow ===\n")
    
    client = MCPClient()
    
    try:
        server_id = await client.connect_http(
            server_name="shadow_workflow",
            url="http://localhost:8000/mcp"
        )
        
        # Execute workflow
        result = await client.call_tool(
            server_id,
            "execute_workflow",
            arguments={
                "workflow_name": "daily_summary",
                "user_id": "example_user",
                "params": {
                    "date": "2024-01-15",
                    "include_insights": True
                }
            }
        )
        
        print(f"Workflow result: {result}")
        
    finally:
        await client.disconnect(server_id)


async def example_observe_and_track():
    """Example: Observe and track data."""
    print("\n=== Example: Observe and Track ===\n")
    
    client = MCPClient()
    
    try:
        server_id = await client.connect_http(
            server_name="shadow_observe",
            url="http://localhost:8000/mcp"
        )
        
        # Record an observation
        result = await client.call_tool(
            server_id,
            "observe",
            arguments={
                "observation_type": "app_usage",
                "content": {
                    "app": "VSCode",
                    "duration_minutes": 45,
                    "files_edited": 3
                },
                "user_id": "example_user",
                "device": "laptop"
            }
        )
        print(f"Observation recorded: {result}")
        
        # Track a decision
        result = await client.call_tool(
            server_id,
            "track",
            arguments={
                "tracking_type": "decision",
                "data": {
                    "decision": "Use FastAPI instead of Flask",
                    "reason": "Better async support",
                    "confidence": 8
                },
                "user_id": "example_user"
            }
        )
        print(f"Decision tracked: {result}")
        
    finally:
        await client.disconnect(server_id)


async def example_agent_coordination():
    """Example: Coordinate multiple agents."""
    print("\n=== Example: Agent Coordination ===\n")
    
    client = MCPClient()
    
    try:
        server_id = await client.connect_http(
            server_name="shadow_agents",
            url="http://localhost:8000/mcp"
        )
        
        # Coordinate agents to accomplish a goal
        result = await client.call_tool(
            server_id,
            "coordinate_agents",
            arguments={
                "goal": "Analyze user productivity patterns and suggest improvements",
                "user_id": "example_user",
                "agent_ids": ["researcher", "analyzer", "advisor"]
            }
        )
        
        print(f"Coordination result: {result}")
        
    finally:
        await client.disconnect(server_id)


async def example_prompts():
    """Example: Use MCP prompts."""
    print("\n=== Example: Use Prompts ===\n")
    
    client = MCPClient()
    
    try:
        server_id = await client.connect_http(
            server_name="shadow_prompts",
            url="http://localhost:8000/mcp"
        )
        
        # Get prompt templates
        prompts = await client.list_prompts(server_id)
        print(f"Available prompts: {[p.name for p in prompts]}")
        
        # Use a prompt
        prompt_result = await client.get_prompt(
            server_id,
            name="analyze_interaction",
            arguments={"context": "User spent 3 hours on VSCode today"}
        )
        
        print(f"\nPrompt template:\n{prompt_result}")
        
    finally:
        await client.disconnect(server_id)


async def example_resources():
    """Example: Access MCP resources."""
    print("\n=== Example: Access Resources ===\n")
    
    client = MCPClient()
    
    try:
        server_id = await client.connect_http(
            server_name="shadow_resources",
            url="http://localhost:8000/mcp"
        )
        
        # List all resources
        resources = await client.list_resources(server_id)
        print(f"Available resources:")
        for r in resources:
            print(f"  - {r.uri}: {r.name}")
        
        # Read user profile
        print("\n--- User Profile ---")
        profile = await client.read_resource(
            server_id,
            "shadow://user/example_user"
        )
        print(profile.content[:300])
        
        # Read memory blocks
        print("\n--- Core Memory ---")
        memory = await client.read_resource(
            server_id,
            "shadow://memory/example_user/core"
        )
        print(memory.content[:300])
        
        # Read knowledge graph
        print("\n--- Knowledge Entities ---")
        entities = await client.read_resource(
            server_id,
            "shadow://knowledge/example_user/entities"
        )
        print(entities.content)
        
        # Read specific entity
        print("\n--- Entity Facts ---")
        facts = await client.read_resource(
            server_id,
            "shadow://knowledge/example_user/entity/Alice"
        )
        print(facts.content[:300])
        
    finally:
        await client.disconnect(server_id)


async def main():
    """Run all examples."""
    if not HAS_MCP_CLIENT:
        print("MCP client not available. Install with: pip install mcp")
        return
    
    print("=" * 60)
    print("Shadow AI MCP Client Examples")
    print("=" * 60)
    
    # Examples to run
    examples = [
        ("Workflow Execution", example_workflow),
        ("Observe and Track", example_observe_and_track),
        ("Agent Coordination", example_agent_coordination),
        ("Prompts", example_prompts),
        ("Resources", example_resources),
    ]
    
    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")
    
    print("\nNote: These examples require a running MCP server.")
    print("Start with: python -m shadowd.run_mcp_server")
    
    # Run examples
    for name, example in examples[:2]:  # Run first 2 by default
        try:
            await example()
        except Exception as e:
            print(f"Error in {name}: {e}")


if __name__ == "__main__":
    asyncio.run(main())