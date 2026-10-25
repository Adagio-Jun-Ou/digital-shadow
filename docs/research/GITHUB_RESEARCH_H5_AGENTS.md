# Hour 5 Research: Agent Frameworks

## Overview
Research on multi-agent coordination frameworks for Shadow AI, based on GitHub projects.

---

## CrewAI (48k stars)
**GitHub**: https://github.com/crewAIInc/crewAI

### Key Concepts
- **Crews**: Teams of AI agents with role-based collaboration
- **Flows**: Event-driven workflows with precise control
- **Agents**: Role-playing autonomous agents
- **Tasks**: Unit of work with goals

### Pattern
```python
from crewai import Agent, Crew, Task, Process

researcher = Agent(
    role="Researcher",
    goal="Find information",
    backstory="Expert researcher",
    tools=[search_tool]
)

writer = Agent(
    role="Writer", 
    goal="Write content",
    backstory="Skilled writer"
)

crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential
)

result = crew.kickoff()
```

---

## AutoGen (56.7k stars)
**GitHub**: https://github.com/microsoft/autogen

### Key Concepts (v0.7+)
- **Core API**: Message passing, event-driven agents, distributed runtime
- **AgentChat API**: Multi-agent patterns (two-agent chat, group chat)
- **Extensions**: LLM clients, code execution, tools

### Pattern
```python
from autogen_agentchat.agents import AssistantAgent
from autogen_ext.models.openai import OpenAIChatCompletionClient

model_client = OpenAIChatCompletionClient(model="gpt-4")

agent = AssistantAgent(
    "assistant",
    model_client=model_client,
    system_message="You are a helpful assistant"
)

result = await agent.run(task="Hello!")
```

### Multi-Agent Orchestration
```python
from autogen_agentchat.tools import AgentTool

math_agent = AssistantAgent("math", model_client=model_client)
code_agent = AssistantAgent("code", model_client=model_client)

orchestrator = AssistantAgent(
    "orchestrator",
    model_client=model_client,
    tools=[AgentTool(math_agent), AgentTool(code_agent)]
)
```

---

## AgentOps (5.4k stars)
**GitHub**: https://github.com/AgentOps-AI/agentops

### Key Features
- Session replay and debugging
- LLM cost tracking
- Multi-agent framework integration
- Self-hosted option

### Integration
```python
import agentops

agentops.init(api_key="...")

@agentops.session
def my_workflow():
    # Agent operations tracked automatically
    pass

agentops.end_session("Success")
```

---

## Key Patterns for Shadow AI

### 1. Agent Architecture
```python
class ShadowAgent:
    def __init__(self, role, goal, backstory):
        self.role = role
        self.goal = goal
        self.backstory = backstory
        self.memory = BlockBasedMemory()
        self.tools = []
    
    async def execute(self, task):
        # Use memory + knowledge graph
        context = self.memory.recall(task)
        # Execute with context
        result = await self.process(task, context)
        # Store new knowledge
        self.memory.add(result)
        return result
```

### 2. Crew Coordination
```python
class ShadowCrew:
    def __init__(self):
        self.agents = []
        self.coordinator = Coordinator()
        self.knowledge_graph = TemporalKnowledgeGraph()
    
    async def kickoff(self, goal):
        # Decompose goal
        tasks = self.coordinator.plan(goal)
        # Assign to agents
        results = []
        for task in tasks:
            agent = self.select_agent(task)
            result = await agent.execute(task)
            self.knowledge_graph.add_episode(result)
            results.append(result)
        return results
```

### 3. Tool Integration
```python
class ToolRegistry:
    def __init__(self):
        self.tools = {}
        self.register_mcp_tools()  # From MCP server
    
    def register(self, name, handler):
        self.tools[name] = handler
    
    async def execute(self, name, *args):
        return await self.tools[name](*args)
```

## Summary
- **CrewAI**: Role-based collaboration, sequential/hierarchical processes
- **AutoGen**: Conversation patterns, event-driven, Microsoft backing  
- **AgentOps**: Observability, session replay, cost tracking
- **Shadow AI**: Combine memory + knowledge graph + agent coordination