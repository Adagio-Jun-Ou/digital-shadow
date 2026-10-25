#!/usr/bin/env python3
"""
Agent Coordination System - Based on CrewAI / AutoGen Research

Implements:
- ShadowAgent: Role-playing autonomous agents with memory
- ShadowCrew: Team coordination with task assignment
- Task planning and execution
- Message passing between agents
- Tool registry integration
"""

import copy
import re as _re
import threading
import uuid
from typing import Dict, List, Optional, Any, Callable, Union
from dataclasses import dataclass, field
from datetime import datetime, timezone

def _ac_ts(data, key, default=None):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from enum import Enum
import asyncio
import json
import logging
from pathlib import Path

from .block_memory import BlockBasedMemory
from .state_graph import StateGraph

logger = logging.getLogger(__name__)

_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text)


def _sanitize_content(text: str) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text)


class AgentRole(Enum):
    COORDINATOR = "coordinator"
    RESEARCHER = "researcher"
    EXECUTOR = "executor"
    ANALYZER = "analyzer"
    CREATIVE = "creative"
    REVIEWER = "reviewer"


class TaskStatus(Enum):
    PENDING = "pending"
    ASSIGNED = "assigned"
    IN_PROGRESS = "in_progress"
    COMPLETED = "completed"
    FAILED = "failed"
    BLOCKED = "blocked"


@dataclass
class AgentTool:
    """Tool available to an agent."""
    name: str
    handler: Callable
    description: str = ""
    parameters: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.name = _sanitize(self.name)
        self.description = _sanitize_content(self.description)
    
    async def execute(self, **kwargs) -> Any:
        """Execute the tool."""
        if asyncio.iscoroutinefunction(self.handler):
            return await self.handler(**kwargs)
        return self.handler(**kwargs)


@dataclass
class AgentMessage:
    """Message from one agent to another (AutoGen pattern)."""
    sender: str
    receiver: str
    content: str
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.sender = _sanitize(self.sender)
        self.receiver = _sanitize(self.receiver)
        self.content = _sanitize_content(self.content)

    def to_dict(self) -> Dict:
        return {
            "sender": self.sender,
            "receiver": self.receiver,
            "content": self.content,
            "timestamp": self.timestamp.isoformat(),
            "metadata": copy.deepcopy(self.metadata)
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'AgentMessage':
        return cls(
            sender=data["sender"],
            receiver=data["receiver"],
            content=data["content"],
            timestamp=_ac_ts(data, "timestamp", datetime.now(timezone.utc)),
            metadata=data.get("metadata", {})
        )


@dataclass
class ShadowTask:
    """Task definition (CrewAI pattern)."""
    id: str
    description: str
    goal: str
    assigned_agent: Optional[str] = None
    status: TaskStatus = TaskStatus.PENDING
    dependencies: List[str] = field(default_factory=list)
    context: Dict[str, Any] = field(default_factory=dict)
    result: Optional[Any] = None
    error: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    completed_at: Optional[datetime] = None

    def __post_init__(self):
        self.id = _sanitize(self.id)
        self.description = _sanitize_content(self.description)
        self.goal = _sanitize_content(self.goal)
        if self.error:
            self.error = _sanitize_content(self.error)

    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "description": self.description,
            "goal": self.goal,
            "assigned_agent": self.assigned_agent,
            "status": self.status.value,
            "dependencies": list(self.dependencies),
            "context": copy.deepcopy(self.context),
            "result": copy.deepcopy(self.result) if self.result is not None else None,
            "error": self.error,
            "created_at": self.created_at.isoformat(),
            "completed_at": self.completed_at.isoformat() if self.completed_at else None
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'ShadowTask':
        return cls(
            id=data["id"],
            description=data["description"],
            goal=data["goal"],
            assigned_agent=data.get("assigned_agent"),
            status=TaskStatus(data.get("status", "pending")),
            dependencies=data.get("dependencies", []),
            context=data.get("context", {}),
            result=data.get("result"),
            error=data.get("error"),
            created_at=_ac_ts(data, "created_at", datetime.now(timezone.utc)),
            completed_at=_ac_ts(data, "completed_at")
        )


class ShadowAgent:
    """
    Role-playing autonomous agent (CrewAI / AutoGen pattern).
    
    Features:
    - Role-based behavior
    - Goal-driven execution
    - Memory integration (block-based)
    - Tool registry
    - Message inbox
    """
    
    _MAX_INBOX = 500
    _MAX_OUTBOX = 500
    _MAX_TASK_HISTORY = 200
    
    def __init__(
        self,
        agent_id: str,
        role: AgentRole,
        goal: str,
        backstory: str = "",
        tools: List[AgentTool] = None,
        memory: BlockBasedMemory = None,
        model_client: Any = None
    ):
        self.agent_id = _sanitize(agent_id)
        self.role = role
        self.goal = _sanitize_content(goal)
        self.backstory = _sanitize_content(backstory)
        self.tools = {t.name: t for t in (tools or [])}
        self.memory = memory
        self.model_client = model_client
        
        self.inbox: List[AgentMessage] = []
        self.outbox: List[AgentMessage] = []
        self.current_task: Optional[ShadowTask] = None
        self.task_history: List[Dict[str, Any]] = []
        self._lock = threading.Lock()
        
        self._initialize_memory()
    
    def _initialize_memory(self):
        """Initialize memory blocks for this agent."""
        if not self.memory:
            self.memory = BlockBasedMemory()
        
        if "role" not in self.memory.core_memory.blocks:
            self.memory.core_set("role", f"Role: {self.role.value}\nGoal: {self.goal}")
        if "context" not in self.memory.core_memory.blocks:
            self.memory.core_set("context", "")
    
    def system_prompt(self) -> str:
        """Generate system prompt for LLM."""
        return f"""You are {self.agent_id}, {self.role.value}.

Your role: {self.role.value}
Your goal: {self.goal}
Your backstory: {self.backstory}

You have access to these tools: {list(self.tools.keys())}

Always maintain context about your tasks and collaborate with other agents when needed.
"""
    
    async def receive_message(self, message: AgentMessage):
        """Receive a message (AutoGen pattern)."""
        with self._lock:
            self.inbox.append(message)
            if len(self.inbox) > self._MAX_INBOX:
                self.inbox = self.inbox[-self._MAX_INBOX:]
        self.memory.core_append("context", f"From {message.sender}: {message.content[:200]}")
    
    async def send_message(self, receiver_id: str, content: str, metadata: Dict = None):
        """Send a message to another agent (AutoGen pattern)."""
        message = AgentMessage(
            sender=self.agent_id,
            receiver=_sanitize(receiver_id),
            content=content,
            metadata=metadata or {}
        )
        with self._lock:
            self.outbox.append(message)
            if len(self.outbox) > self._MAX_OUTBOX:
                self.outbox = self.outbox[-self._MAX_OUTBOX:]
        return message
    
    async def use_tool(self, tool_name: str, **kwargs) -> Any:
        """Execute a tool."""
        if tool_name not in self.tools:
            raise ValueError(f"Tool '{tool_name}' not available to agent {self.agent_id}")
        
        tool = self.tools[tool_name]
        result = await tool.execute(**kwargs)
        
        self.memory.core_append("recent", f"Used tool {tool_name}: {str(result)[:200]}")
        
        return result
    
    async def execute_task(self, task: ShadowTask) -> Dict[str, Any]:
        """Execute a task."""
        self.current_task = task
        task.status = TaskStatus.IN_PROGRESS
        
        try:
            self.memory.core_append("recent", f"Starting task: {task.description[:100]}")
            
            context = self._prepare_context(task)
            
            if self.tools:
                tool_results = await self._execute_tools_for_task(task)
                context.update(tool_results)
            
            result = {
                "task_id": task.id,
                "agent": self.agent_id,
                "context": context,
                "output": f"Task '{task.description}' completed successfully",
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.memory.archival_insert(
                f"Task {task.id}: {task.description[:100]}",
                {"source": "task", "task_type": task.goal}
            )
            task.result = result
            task.status = TaskStatus.COMPLETED
            task.completed_at = datetime.now(timezone.utc)
            with self._lock:
                self.task_history.append(task.to_dict())
                if len(self.task_history) > self._MAX_TASK_HISTORY:
                    self.task_history = self.task_history[-self._MAX_TASK_HISTORY:]
            
            return result
            
        except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
            logger.error("Agent %s failed task %s: %s", self.agent_id, task.id, type(e).__name__)
            task.status = TaskStatus.FAILED
            task.error = f"Task failed: {type(e).__name__}"
            raise
        
        finally:
            self.current_task = None
    
    def _prepare_context(self, task: ShadowTask) -> Dict[str, Any]:
        """Prepare execution context."""
        context = {
            "task": task.context,
            "memory": {
                "core": self.memory.core_memory.get_context(),
                "recent": self.memory.working_memory.get_messages()[-3:]
            }
        }
        
        archival_context = self.memory.archival_search(task.description, limit=3)
        if archival_context:
            context["relevant_memories"] = [m["content"][:200] for m in archival_context]
        
        return context
    
    async def _execute_tools_for_task(self, task: ShadowTask) -> Dict[str, Any]:
        """Execute relevant tools for a task (if needed)."""
        results = {}
        
        for tool_name, tool in self.tools.items():
            if self._should_use_tool(tool_name, task):
                try:
                    result = await self.use_tool(tool_name, query=task.description)
                    results[tool_name] = result
                except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
                    results[tool_name] = {"error": type(e).__name__}
        
        return results
    
    def _should_use_tool(self, tool_name: str, task: ShadowTask) -> bool:
        """Determine if a tool should be used for a task."""
        if tool_name in ["search_memory", "execute_workflow", "get_insights"]:
            return True
        return False
    
    def get_status(self) -> Dict[str, Any]:
        """Get agent status."""
        with self._lock:
            return {
                "agent_id": self.agent_id,
                "role": self.role.value,
                "current_task": self.current_task.id if self.current_task else None,
                "inbox_count": len(self.inbox),
                "outbox_count": len(self.outbox),
                "task_history_count": len(self.task_history),
                "memory_stats": self.memory.get_stats() if self.memory else None
            }


class ShadowCrew:
    """
    Team of agents with coordination (CrewAI pattern).
    
    Features:
    - Multiple agents with roles
    - Task assignment
    - Process types (sequential, hierarchical)
    - Knowledge sharing
    """
    
    _MAX_AGENTS = 50
    _MAX_TASKS = 500
    _MAX_MESSAGE_BUS = 2000
    
    def __init__(
        self,
        crew_id: str,
        name: str = "",
        process: str = "sequential",
        knowledge_graph: Any = None
    ):
        self.crew_id = _sanitize(crew_id)
        self.name = _sanitize(name) or self.crew_id
        self.process = _sanitize(process)
        self.knowledge_graph = knowledge_graph
        self.agents: Dict[str, ShadowAgent] = {}
        self.tasks: Dict[str, ShadowTask] = {}
        self.task_queue: List[str] = []
        self.message_bus: List[AgentMessage] = []
        self._lock = threading.Lock()
        
        self._initialize_default_agents()
    
    def _initialize_default_agents(self):
        """Initialize default agent roles."""
        self.add_agent(ShadowAgent(
            agent_id="coordinator",
            role=AgentRole.COORDINATOR,
            goal="Coordinate tasks between agents",
            backstory="Expert at task delegation and coordination"
        ))
        
        self.add_agent(ShadowAgent(
            agent_id="researcher",
            role=AgentRole.RESEARCHER,
            goal="Search and retrieve information",
            backstory="Expert at finding relevant information"
        ))
        
        self.add_agent(ShadowAgent(
            agent_id="executor",
            role=AgentRole.EXECUTOR,
            goal="Execute actions and workflows",
            backstory="Expert at executing tasks efficiently"
        ))
    
    def add_agent(self, agent: ShadowAgent):
        """Add an agent to the crew."""
        with self._lock:
            if len(self.agents) >= self._MAX_AGENTS:
                logger.warning("Agent limit reached (%d), cannot add %s", self._MAX_AGENTS, agent.agent_id)
                return
            self.agents[agent.agent_id] = agent
    
    def add_task(self, task: ShadowTask):
        """Add a task to the queue."""
        with self._lock:
            if len(self.tasks) >= self._MAX_TASKS:
                logger.warning("Task limit reached (%d), cannot add %s", self._MAX_TASKS, task.id)
                return
            self.tasks[task.id] = task
            self.task_queue.append(task.id)
    
    def create_task(
        self,
        description: str,
        goal: str,
        dependencies: List[str] = None,
        context: Dict[str, Any] = None
    ) -> ShadowTask:
        """Create a new task."""
        self._task_counter = getattr(self, '_task_counter', 0) + 1
        task_id = f"task_{uuid.uuid4().hex[:12]}"
        
        task = ShadowTask(
            id=task_id,
            description=description,
            goal=goal,
            dependencies=dependencies or [],
            context=context or {}
        )
        
        self.add_task(task)
        return task
    
    def assign_task(self, task_id: str, agent_id: str) -> bool:
        """Assign a task to an agent."""
        with self._lock:
            if task_id not in self.tasks or agent_id not in self.agents:
                return False
            
            task = self.tasks[task_id]
            
            if self._dependencies_met(task):
                task.assigned_agent = agent_id
                task.status = TaskStatus.ASSIGNED
                return True
            
            return False
    
    def _dependencies_met(self, task: ShadowTask) -> bool:
        """Check if task dependencies are met."""
        for dep_id in task.dependencies:
            if dep_id not in self.tasks:
                return False
            if self.tasks[dep_id].status != TaskStatus.COMPLETED:
                return False
        return True
    
    async def kickoff(self, goal: str) -> Dict[str, Any]:
        """
        Execute the crew's tasks (CrewAI kickoff pattern).
        
        Args:
            goal: High-level goal for the crew
        
        Returns:
            Execution results
        """
        results = {
            "goal": goal,
            "tasks_completed": [],
            "tasks_failed": [],
            "knowledge_generated": [],
            "messages": []
        }
        
        plan = await self._plan_tasks(goal)
        
        for task_id in plan["task_order"]:
            task = self.tasks.get(task_id)
            if not task:
                continue
            
            if task.assigned_agent is None:
                agent_id = self._select_agent_for_task(task)
                if not agent_id:
                    results["tasks_failed"].append({
                        "task_id": task_id,
                        "error": "No suitable agent available"
                    })
                    continue
                self.assign_task(task_id, agent_id)
            
            agent = self.agents[task.assigned_agent]
            
            try:
                task_result = await agent.execute_task(task)
                
                results["tasks_completed"].append({
                    "task_id": task_id,
                    "agent": agent.agent_id,
                    "result": task_result
                })
                
                if self.knowledge_graph:
                    await self._add_to_knowledge_graph(task, task_result)
                    results["knowledge_generated"].append(task_id)
                
                messages = agent.outbox.copy()
                agent.outbox.clear()
                results["messages"].extend([m.to_dict() for m in messages])
                
            except (RuntimeError, ValueError, TypeError, OSError, KeyError, AttributeError) as e:
                logger.error("Task %s failed: %s", task_id, type(e).__name__)
                results["tasks_failed"].append({
                    "task_id": task_id,
                    "error": type(e).__name__
                })
        
        return results
    
    async def _plan_tasks(self, goal: str) -> Dict[str, Any]:
        """Plan task execution order."""
        task_order = []
        visiting = set()
        
        def add_task_and_deps(task_id: str):
            if task_id in task_order:
                return
            if task_id in visiting:
                raise ValueError(f"Circular dependency detected involving task '{task_id}'")
            visiting.add(task_id)
            task = self.tasks.get(task_id)
            if task:
                for dep in task.dependencies:
                    add_task_and_deps(dep)
            visiting.discard(task_id)
            if task_id not in task_order:
                task_order.append(task_id)
        
        for task_id in self.task_queue:
            add_task_and_deps(task_id)
        
        return {
            "goal": goal,
            "task_order": task_order,
            "total_tasks": len(task_order)
        }
    
    def _select_agent_for_task(self, task: ShadowTask) -> Optional[str]:
        """Select the best agent for a task based on role."""
        role_mapping = {
            "research": AgentRole.RESEARCHER,
            "analysis": AgentRole.ANALYZER,
            "execution": AgentRole.EXECUTOR,
            "creative": AgentRole.CREATIVE,
            "review": AgentRole.REVIEWER,
            "coordination": AgentRole.COORDINATOR
        }
        
        goal_lower = task.goal.lower()
        for keyword, role in role_mapping.items():
            if _re.search(r'\b' + _re.escape(keyword) + r'\b', goal_lower):
                for agent_id, agent in self.agents.items():
                    if agent.role == role and agent.current_task is None:
                        return agent_id
        
        for agent_id, agent in self.agents.items():
            if agent.current_task is None:
                return agent_id
        
        return None
    
    async def _add_to_knowledge_graph(self, task: ShadowTask, result: Dict[str, Any]):
        """Add task result to knowledge graph."""
        if not self.knowledge_graph:
            return
        
        try:
            if hasattr(self.knowledge_graph, 'add_fact'):
                await self.knowledge_graph.add_fact(
                    f"Task {task.id}: {task.description[:100]}",
                    source="crew_execution",
                    metadata={"agent": task.assigned_agent, "result": result}
                )
        except (RuntimeError, ValueError, TypeError, AttributeError, OSError) as e:
            logger.error("Failed to add to knowledge graph: %s", type(e).__name__)
    
    def send_message(self, from_agent: str, to_agent: str, content: str):
        """Send message between agents."""
        with self._lock:
            if from_agent not in self.agents or to_agent not in self.agents:
                return False
        
            message = AgentMessage(
                sender=from_agent,
                receiver=to_agent,
                content=content
            )
        
            self.message_bus.append(message)
            if len(self.message_bus) > self._MAX_MESSAGE_BUS:
                self.message_bus = self.message_bus[-self._MAX_MESSAGE_BUS:]
            return message
    
    def get_agent_status(self) -> Dict[str, Any]:
        """Get status of all agents."""
        with self._lock:
            return {
                agent_id: agent.get_status()
                for agent_id, agent in self.agents.items()
            }
    
    def get_crew_status(self) -> Dict[str, Any]:
        """Get overall crew status."""
        with self._lock:
            completed = sum(1 for t in self.tasks.values() if t.status == TaskStatus.COMPLETED)
            pending = sum(1 for t in self.tasks.values() if t.status == TaskStatus.PENDING)
            in_progress = sum(1 for t in self.tasks.values() if t.status == TaskStatus.IN_PROGRESS)
            
            return {
                "crew_id": self.crew_id,
                "name": self.name,
                "process": self.process,
                "agents_count": len(self.agents),
                "tasks_total": len(self.tasks),
                "tasks_completed": completed,
                "tasks_pending": pending,
                "tasks_in_progress": in_progress,
                "message_bus_size": len(self.message_bus)
            }


class AgentSwarm:
    """
    Orchestrator for multiple crews (AutoGen AgentTool pattern).
    
    Allows crews to be used as tools by other agents.
    """
    
    _MAX_CREWS = 20
    
    def __init__(self):
        self.crews: Dict[str, ShadowCrew] = {}
        self.crew_registry: Dict[str, AgentTool] = {}
        self._lock = threading.Lock()
    
    def add_crew(self, crew: ShadowCrew):
        """Add a crew to the swarm."""
        with self._lock:
            if len(self.crews) >= self._MAX_CREWS:
                logger.warning("Crew limit reached (%d)", self._MAX_CREWS)
                return
            self.crews[crew.crew_id] = crew
        
            async def crew_tool_handler(query: str, **kwargs):
                return await crew.kickoff(query)
        
            self.crew_registry[crew.crew_id] = AgentTool(
                name=f"crew_{crew.crew_id}",
                handler=crew_tool_handler,
                description=f"Execute {crew.name} crew tasks"
            )
    
    async def coordinate(self, goal: str, crew_ids: List[str] = None) -> Dict[str, Any]:
        """Coordinate multiple crews."""
        with self._lock:
            crew_ids_to_use = crew_ids or list(self.crews.keys())
            crews_to_use = [
                self.crews[cid] for cid in crew_ids_to_use
                if cid in self.crews
            ]
        
        results = {}
        
        for crew in crews_to_use:
            crew_result = await crew.kickoff(goal)
            results[crew.crew_id] = crew_result
        
        return {
            "goal": goal,
            "coordinated_crews": [c.crew_id for c in crews_to_use],
            "results": results
        }
    
    def get_status(self) -> Dict[str, Any]:
        """Get swarm status."""
        with self._lock:
            return {
                "crews_count": len(self.crews),
                "crews": {cid: crew.get_crew_status() for cid, crew in self.crews.items()},
                "tools": list(self.crew_registry.keys())
            }


if __name__ == "__main__":
    print("Agent Coordination Module")
    print("Based on CrewAI and AutoGen patterns")
    print("  - ShadowAgent: Role-playing agents with memory")
    print("  - ShadowCrew: Team coordination")
    print("  - AgentSwarm: Multi-crew orchestration")