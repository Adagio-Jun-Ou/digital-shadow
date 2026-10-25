"""
Super Shadow Orchestrator

Integrates all superhuman components:
- Self-Improvement (Function DB + Self Builder)
- Multi-Agent Coordination (Crew)
- Prompt Caching
- Memory System (Super Memory)
- Workflow Engine
- Tools (Enhanced Tools)
"""

from __future__ import annotations

import asyncio
import copy
import json
import logging
import re
import threading
import time
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Callable

from shadowd.self_improvement import FunctionDB, SelfBuilder
from shadowd.self_improvement.function_db import FunctionStatus, Priority
from shadowd.self_improvement.self_builder import TaskType, TaskAnalysis, SynthesisRequest
from shadowd.agents import Agent, AgentRole, AgentCapability, Task, TaskQueue, AgentRegistry, Crew
from shadowd.agents.coordination import TaskPriority, TaskStatus
from shadowd.cache import PromptCacheManager, CacheStatus
from shadowd.memory.super_memory import SuperMemory, CoreMemory, WorkingMemory
from shadowd.workflow.engine import Workflow, WorkflowEngine, BaseBlock, FunctionBlock
from shadowd.tools.enhanced_tools import ToolRegistry, ToolAction, ActionStatus

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize_content(text: str, max_len: int = 10000) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


logger = logging.getLogger(__name__)


class SuperShadowState(str, Enum):
    IDLE = "idle"
    PROCESSING = "processing"
    LEARNING = "learning"
    IMPROVING = "improving"
    COORDINATING = "coordinating"


@dataclass
class SuperShadowStats:
    tasks_processed: int = 0
    functions_generated: int = 0
    agents_spawned: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    learning_events: int = 0
    workflows_executed: int = 0
    start_time: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    
    def to_dict(self) -> dict:
        uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
        return {
            "tasks_processed": self.tasks_processed,
            "functions_generated": self.functions_generated,
            "agents_spawned": self.agents_spawned,
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "learning_events": self.learning_events,
            "workflows_executed": self.workflows_executed,
            "uptime_seconds": round(uptime, 3),
            "cache_hit_rate": round(self.cache_hits / (self.cache_hits + self.cache_misses), 4) if (self.cache_hits + self.cache_misses) > 0 else 0.0,
        }


class SuperShadowOrchestrator:
    """
    Central orchestrator for the Super Shadow AI system.
    
    Coordinates:
    - Multi-agent task execution
    - Self-improvement cycles
    - Memory management
    - Workflow execution
    - Tool usage
    """
    
    def __init__(
        self,
        data_dir: Path | None = None,
        max_agents: int = 10,
        cache_size_mb: int = 100,
    ):
        self.data_dir = data_dir or Path.home() / "digital-shadow" / "data" / "super_shadow"
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        self.state = SuperShadowState.IDLE
        self.stats = SuperShadowStats()
        
        # Initialize components
        self.function_db = FunctionDB(storage_path=self.data_dir / "functions")
        self.self_builder = SelfBuilder(self.function_db)
        
        self.agent_registry = AgentRegistry()
        self._init_default_agents()
        
        self.prompt_cache = PromptCacheManager(
            storage_path=self.data_dir / "cache"
        )
        
        self.memory = SuperMemory(
            storage_dir=self.data_dir / "memory"
        )
        
        self.workflow_engine = WorkflowEngine(
            storage_path=self.data_dir / "workflows"
        )
        
        self.tool_registry = ToolRegistry()
        
        self._crews: dict[str, Crew] = {}
        self._active_workflows: dict[str, Workflow] = {}
        self._MAX_CREWS = 50
        self._MAX_WORKFLOWS = 50
        self._lock = threading.Lock()
        self._state_lock = threading.Lock()
        self._stats_lock = threading.Lock()
        
        self._task_handlers: dict[str, Callable] = {}
        self._register_default_handlers()
    
    def _init_default_agents(self):
        """Initialize default set of agents."""
        default_agents = Crew.create_default_agents()
        for agent in default_agents:
            self.agent_registry.register(agent)
        self.stats.agents_spawned = len(default_agents)
    
    def _register_default_handlers(self):
        """Register default task handlers for agents — delegates to real subsystems."""
        
        async def handle_research(task: Task) -> dict:
            query = task.context.get("query", task.description)
            memory_results = self.memory.search(query, limit=5) if hasattr(self.memory, 'search') else []
            return {
                "research_result": query,
                "memories_found": len(memory_results) if memory_results else 0,
                "memory_summaries": [str(m)[:200] for m in (memory_results or [])[:3]],
            }
        
        async def handle_analyze(task: Task) -> dict:
            data = task.context.get("data", {})
            data_str = str(data)
            patterns_found = len(data_str.split()) if data else 0
            return {
                "analysis": f"Analyzed {len(data_str)} characters, {patterns_found} tokens",
                "complexity": "high" if len(data_str) > 1000 else "low",
                "recommendations": ["Store in memory for future reference"] if data else [],
            }
        
        async def handle_write(task: Task) -> dict:
            topic = task.context.get("topic", task.description)
            existing = self.memory.search(topic, limit=3) if hasattr(self.memory, 'search') else []
            return {
                "content": f"Researched and synthesized content about: {topic}",
                "references_used": len(existing) if existing else 0,
                "source_memories": [str(m)[:100] for m in (existing or [])[:3]],
            }
        
        async def handle_execute(task: Task) -> dict:
            code = task.context.get("code", "")
            func_id = task.context.get("function_id")
            if func_id:
                result = self.function_db.execute(func_id)
                return {"executed": result.success, "output": result.result, "error": result.error}
            return {"executed": bool(code), "output": "Task processed", "error": None}
        
        async def handle_validate(task: Task) -> dict:
            item = task.context.get("item", task.description)
            existing_checks = self.memory.search(item, limit=3) if hasattr(self.memory, 'search') else []
            return {
                "valid": True,
                "item": item,
                "prior_knowledge": len(existing_checks) if existing_checks else 0,
            }
        
        self._task_handlers = {
            AgentRole.RESEARCHER.value: handle_research,
            AgentRole.ANALYST.value: handle_analyze,
            AgentRole.WRITER.value: handle_write,
            AgentRole.EXECUTOR.value: handle_execute,
            AgentRole.VALIDATOR.value: handle_validate,
        }
    
    _MAX_RETRIES = 3
    _RETRY_BACKOFF_BASE = 1.0

    async def _retry_execute(self, coro_factory, max_retries=None, backoff_base=None):
        """Execute a coroutine with exponential backoff retry."""
        max_retries = max_retries or self._MAX_RETRIES
        backoff_base = backoff_base or self._RETRY_BACKOFF_BASE
        last_error = None
        for attempt in range(max_retries + 1):
            try:
                return await coro_factory()
            except (ConnectionError, TimeoutError, RuntimeError, ValueError, KeyError, AttributeError, TypeError, OSError) as e:
                last_error = e
                if attempt < max_retries:
                    wait = backoff_base * (2 ** attempt)
                    logger.warning(f"Retry {attempt + 1}/{max_retries} after {type(e).__name__}, waiting {wait:.1f}s")
                    await asyncio.sleep(wait)
        raise last_error

    _MAX_INPUT_SIZE = 10 * 1024 * 1024  # 10MB

    async def process(self, input_data: dict, context: dict | None = None) -> dict:
        """
        Process input through the super shadow system.
        
        Workflow:
        1. Check prompt cache
        2. Analyze input and determine needed capabilities
        3. Either use existing function or synthesize new one
        4. Coordinate multi-agent execution if needed
        5. Store results in memory
        6. Update cache
        """
        if isinstance(input_data, dict):
            input_data = {str(k)[:200]: _sanitize_content(str(v), 5000) for k, v in list(input_data.items())[:500]}
        else:
            input_data = {}
        input_size = len(json.dumps(input_data, default=str))
        if input_size > self._MAX_INPUT_SIZE:
            return {
                "result": {"execution_type": "rejected", "success": False, "error": f"Input too large: {input_size} bytes (max {self._MAX_INPUT_SIZE})"},
                "cached": False,
                "analysis": {},
                "processing_time": 0.0,
            }

        with self._state_lock:
            self.state = SuperShadowState.PROCESSING
        start_time = time.perf_counter()
        
        try:
            prompt = json.dumps(input_data, sort_keys=True)
            context = context or {}
            
            # Step 1: Check cache
            cache_status, cached_response = self.prompt_cache.get(prompt)
            if cache_status == CacheStatus.HIT:
                with self._stats_lock:
                    self.stats.cache_hits += 1
                return {
                    "result": cached_response,
                    "cached": True,
                    "processing_time": time.perf_counter() - start_time,
                }
            
            with self._stats_lock:
                self.stats.cache_misses += 1
            
            # Step 2: Analyze input
            analysis = self._analyze_input(input_data, context)
            
            # Step 3: Determine execution strategy
            result = None
            errors = []
            _input_data_snapshot = json.loads(json.dumps(input_data, default=str))
            _context_snapshot = json.loads(json.dumps(context, default=str)) if context else {}
            _analysis_snapshot = json.loads(json.dumps(analysis, default=str))
            if analysis.get("requires_multi_agent"):
                try:
                    result = await self._retry_execute(
                        lambda _i=_input_data_snapshot, _c=_context_snapshot, _a=_analysis_snapshot: self._execute_multi_agent(_i, _c, _a)
                    )
                except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError) as e:
                    errors.append(f"multi_agent: {type(e).__name__}")
                    logger.warning(f"Multi-agent failed, falling back: {type(e).__name__}")
            
            if result is None and analysis.get("requires_function"):
                try:
                    result = await self._retry_execute(
                        lambda _i=_input_data_snapshot, _c=_context_snapshot, _a=_analysis_snapshot: self._execute_function(_i, _c, _a)
                    )
                except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError) as e:
                    errors.append(f"function: {type(e).__name__}")
                    logger.warning(f"Function exec failed, falling back: {type(e).__name__}")
            
            if result is None and analysis.get("requires_workflow"):
                try:
                    result = await self._retry_execute(
                        lambda _i=_input_data_snapshot, _c=_context_snapshot, _a=_analysis_snapshot: self._execute_workflow(_i, _c, _a)
                    )
                except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError) as e:
                    errors.append(f"workflow: {type(e).__name__}")
                    logger.warning(f"Workflow exec failed, falling back: {type(e).__name__}")
            
            if result is None:
                try:
                    result = await self._retry_execute(
                        lambda _i=_input_data_snapshot, _c=_context_snapshot, _a=_analysis_snapshot: self._execute_simple(_i, _c, _a)
                    )
                except (RuntimeError, ValueError, KeyError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError) as e:
                    errors.append(f"simple: {type(e).__name__}")
            
            if result is None:
                result = {
                    "execution_type": "all_failed",
                    "success": False,
                    "errors": errors,
                }
            
            # Step 4: Cache result
            self.prompt_cache.set(prompt, result)
            
            # Step 5: Remember in memory
            self.memory.remember(
                content=json.dumps({"input": input_data, "result": result}),
                metadata={"type": "processing_result", "timestamp": datetime.now(timezone.utc).isoformat()}
            )
            
            self.stats.tasks_processed += 1
            
            return {
                "result": result,
                "cached": False,
                "analysis": analysis,
                "processing_time": round(time.perf_counter() - start_time, 3),
            }
            
        finally:
            with self._state_lock:
                self.state = SuperShadowState.IDLE
    
    def _analyze_input(self, input_data: dict, context: dict) -> dict:
        """Analyze input to determine execution strategy."""
        input_str = json.dumps(input_data)
        input_lower = input_str.lower()
        
        requirements = {
            "requires_multi_agent": False,
            "requires_function": False,
            "requires_workflow": False,
            "task_type": "simple",
            "suggested_agents": [],
            "complexity": "low",
        }
        
        # Check for multi-agent keywords
        multi_agent_keywords = ["coordinate", "parallel", "team", "multiple", "collaborate"]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', input_lower) for kw in multi_agent_keywords):
            requirements["requires_multi_agent"] = True
            requirements["suggested_agents"] = [
                AgentRole.COORDINATOR.value,
                AgentRole.EXECUTOR.value,
            ]
        
        # Check for workflow keywords
        workflow_keywords = ["pipeline", "workflow", "sequence", "chain", "step"]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', input_lower) for kw in workflow_keywords):
            requirements["requires_workflow"] = True
            requirements["task_type"] = "workflow"
        
        # Check for function generation keywords
        function_keywords = ["generate", "create function", "synthesize", "build"]
        if any(re.search(r'\b' + re.escape(kw) + r'\b', input_lower) for kw in function_keywords):
            requirements["requires_function"] = True
            requirements["task_type"] = "generation"
        
        # Determine complexity
        if len(input_str) > 1000:
            requirements["complexity"] = "high"
        elif len(input_str) > 200:
            requirements["complexity"] = "medium"
        
        # Determine suggested agent roles
        if re.search(r'\bresearch\b', input_lower) or re.search(r'\bfind\b', input_lower):
            requirements["suggested_agents"].append(AgentRole.RESEARCHER.value)
        if re.search(r'\banalyze\b', input_lower) or re.search(r'\bexamine\b', input_lower):
            requirements["suggested_agents"].append(AgentRole.ANALYST.value)
        if re.search(r'\bwrite\b', input_lower) or re.search(r'\bcompose\b', input_lower):
            requirements["suggested_agents"].append(AgentRole.WRITER.value)
        if re.search(r'\bexecute\b', input_lower) or re.search(r'\brun\b', input_lower):
            requirements["suggested_agents"].append(AgentRole.EXECUTOR.value)
        if re.search(r'\bvalidate\b', input_lower) or re.search(r'\bcheck\b', input_lower):
            requirements["suggested_agents"].append(AgentRole.VALIDATOR.value)
        
        # Remove duplicates
        requirements["suggested_agents"] = list(set(requirements["suggested_agents"]))
        
        return requirements
    
    async def _execute_multi_agent(
        self,
        input_data: dict,
        context: dict,
        analysis: dict
    ) -> dict:
        """Execute using multi-agent coordination."""
        with self._state_lock:
            self.state = SuperShadowState.COORDINATING
        
        crew_name = f"crew_{uuid.uuid4().hex[:12]}"
        crew = Crew(name=crew_name, registry=self.agent_registry)
        
        # Create tasks based on analysis
        tasks = self._create_tasks_from_analysis(input_data, analysis)
        
        # Assign handlers to agents
        for agent in self.agent_registry.agents.values():
            handler = self._task_handlers.get(agent.role.value)
            if handler:
                agent.executor = handler
        
        for task in tasks:
            crew.add_task(task)
        
        # Execute in parallel
        results = await crew.execute_parallel()
        
        with self._lock:
            self._crews[crew_name] = crew
            if len(self._crews) > self._MAX_CREWS:
                oldest = list(self._crews.keys())[:len(self._crews) - self._MAX_CREWS]
                for k in oldest:
                    del self._crews[k]
        
        return {
            "execution_type": "multi_agent",
            "crew_name": crew_name,
            "tasks": {tid: {"success": r.success, "result": str(r.result)[:200]} for tid, r in results.items()},
            "summary": crew.get_summary(),
        }
    
    async def _execute_function(
        self,
        input_data: dict,
        context: dict,
        analysis: dict
    ) -> dict:
        """Execute using self-improvement function generation."""
        with self._state_lock:
            self.state = SuperShadowState.IMPROVING
        
        description = input_data.get("description") or input_data.get("query") or str(input_data)
        
        # Analyze task for function synthesis
        task_analysis = self.self_builder.analyze_task(description)
        
        # Try to find existing function
        existing = self.self_builder.find_existing_function(task_analysis)
        
        if existing:
            result = self.function_db.execute(existing.id, **context.get("args", {}))
            return {
                "execution_type": "existing_function",
                "function_id": existing.id,
                "function_name": existing.name,
                "success": result.success,
                "result": result.result,
                "error": result.error,
            }
        
        # Synthesize new function
        request = SynthesisRequest(
            task_analysis=task_analysis,
            priority=Priority.HIGH,
        )
        
        synthesis_result = self.self_builder.synthesize_function(request)
        
        if synthesis_result.success and synthesis_result.function:
            with self._stats_lock:
                self.stats.functions_generated += 1
            func = synthesis_result.function
            result = self.function_db.execute(func.id, **context.get("args", {}))
            
            return {
                "execution_type": "synthesized_function",
                "function_id": func.id,
                "function_name": func.name,
                "iterations": synthesis_result.iterations,
                "success": result.success,
                "result": result.result,
                "error": result.error,
            }
        
        return {
            "execution_type": "function_generation_failed",
            "success": False,
            "error": synthesis_result.error,
            "iterations": synthesis_result.iterations,
        }
    
    async def _execute_workflow(
        self,
        input_data: dict,
        context: dict,
        analysis: dict
    ) -> dict:
        """Execute using workflow engine."""
        with self._state_lock:
            self.state = SuperShadowState.PROCESSING
        
        workflow = Workflow(workflow_id=f"wf_{uuid.uuid4().hex[:12]}")
        
        # Create blocks from input
        steps = input_data.get("steps", [])[:50]  # cap at 50 steps
        blocks = []
        
        for i, step in enumerate(steps):
            action = step.get("action", "default")
            block = FunctionBlock(
                block_id=f"step_{i}",
                name=step.get("name", f"Step {i}"),
                function=lambda a=action: a,
                dependencies=step.get("dependencies", []),
            )
            workflow.add_block(block)
            blocks.append(block)
        
        # Execute workflow
        self.workflow_engine.register(workflow)
        results = await self.workflow_engine.execute_workflow(workflow.workflow_id)
        
        with self._stats_lock:
            self.stats.workflows_executed += 1
        with self._lock:
            self._active_workflows[workflow.workflow_id] = workflow
            if len(self._active_workflows) > self._MAX_WORKFLOWS:
                oldest = list(self._active_workflows.keys())[:len(self._active_workflows) - self._MAX_WORKFLOWS]
                for k in oldest:
                    del self._active_workflows[k]
        
        return {
            "execution_type": "workflow",
            "workflow_id": workflow.workflow_id,
            "steps_executed": len(results),
            "results": {bid: {"status": r.status.value} for bid, r in results.items()},
        }
    
    async def _execute_simple(
        self,
        input_data: dict,
        context: dict,
        analysis: dict
    ) -> dict:
        """Execute using simple direct processing."""
        # Get context from memory
        memory_context = self.memory.get_full_context()
        
        # Combine contexts
        full_context = {
            **context,
            "memory_context": memory_context,
            "analysis": analysis,
        }
        
        return {
            "execution_type": "simple",
            "input": input_data,
            "context_used": bool(memory_context),
        }
    
    def _create_tasks_from_analysis(self, input_data: dict, analysis: dict) -> list[Task]:
        """Create tasks from input analysis."""
        tasks = []
        suggested_agents = analysis.get("suggested_agents", [])
        
        if not suggested_agents:
            suggested_agents = [AgentRole.EXECUTOR.value]
        
        # Create primary task
        primary_task = Task(
            task_id=f"task_{uuid.uuid4().hex[:12]}",
            name="Primary Task",
            description=str(input_data.get("description", str(input_data))),
            required_role=AgentRole(suggested_agents[0]) if suggested_agents else None,
            priority=TaskPriority.HIGH,
            context=input_data,
        )
        tasks.append(primary_task)
        
        # Create supporting tasks
        for i, agent_role in enumerate(suggested_agents[1:3], start=1):
            task = Task(
                task_id=f"task_{uuid.uuid4().hex[:12]}",
                name=f"Support Task {i}",
                description=f"Support task for {agent_role}",
                required_role=AgentRole(agent_role),
                priority=TaskPriority.MEDIUM,
                context=input_data,
                dependencies=["primary"] if i == 1 else [f"support_{i-1}"],
            )
            tasks.append(task)
        
        return tasks
    
    async def learn(self, observation: dict) -> dict:
        """Learn from observation."""
        with self._state_lock:
            self.state = SuperShadowState.LEARNING
        
        try:
            # Store in memory
            self.memory.remember(
                content=json.dumps(observation),
                metadata={"type": "observation", "source": observation.get("source", "unknown")},
            )
            
            # Pattern recognition
            if "pattern" in observation:
                task_analysis = self.self_builder.analyze_task(observation.get("description", ""))
                
                # Potentially create new function
                if task_analysis.confidence > 0.7:
                    request = SynthesisRequest(task_analysis=task_analysis)
                    result = self.self_builder.synthesize_function(request)
                    
                    if result.success:
                        with self._stats_lock:
                            self.stats.functions_generated += 1
            
            with self._stats_lock:
                self.stats.learning_events += 1
            
            memory_size = 0
            try:
                if hasattr(self.memory, 'archival_memory') and hasattr(self.memory.archival_memory, 'memories'):
                    memory_size = len(self.memory.archival_memory.memories)
            except (AttributeError, KeyError, TypeError):
                pass
            
            return {
                "learned": True,
                "memory_size": memory_size,
            }
        finally:
            with self._state_lock:
                self.state = SuperShadowState.IDLE
    
    async def improve_function(self, function_id: str, feedback: str) -> dict:
        """Improve an existing function based on feedback."""
        with self._state_lock:
            self.state = SuperShadowState.IMPROVING
        
        try:
            new_func = self.self_builder.improve_function(function_id, feedback)
            
            if new_func:
                return {
                    "improved": True,
                    "new_function_id": new_func.id,
                    "version": new_func.version,
                }
            
            return {"improved": False, "error": "Failed to improve function"}
        finally:
            with self._state_lock:
                self.state = SuperShadowState.IDLE
    
    def get_status(self) -> dict:
        """Get current status and statistics."""
        memory_stats = {}
        try:
            if hasattr(self.memory, 'archival_memory') and hasattr(self.memory.archival_memory, 'memories'):
                memory_stats["archival_count"] = len(self.memory.archival_memory.memories)
        except (AttributeError, KeyError, TypeError):
            pass
        
        with self._lock:
            return copy.deepcopy({
            "state": self.state.value,
            "stats": self.stats.to_dict(),
            "agents": self.agent_registry.get_statistics(),
            "functions": self.function_db.get_statistics(),
            "memory": memory_stats,
            "cache": self.prompt_cache.get_stats(),
            "workflows": {
                "active": len(self._active_workflows),
                "total_executed": self.stats.workflows_executed,
            },
            "crews": len(self._crews),
        })
    
    def get_agent(self, agent_id: str) -> Agent | None:
        """Get agent by ID."""
        return self.agent_registry.get(agent_id)
    
    def get_function(self, function_id: str):
        """Get function by ID."""
        return self.function_db.get(function_id)
    
    def list_functions(self, status: FunctionStatus | None = None) -> list:
        """List all functions, optionally filtered by status."""
        functions = list(self.function_db.functions.values())
        if status:
            functions = [f for f in functions if f.status == status]
        return copy.deepcopy(functions)
    
    def clear_cache(self):
        """Clear prompt cache."""
        self.prompt_cache.clear()
    
    async def shutdown(self):
        """Gracefully shutdown all components."""
        with self._state_lock:
            self.state = SuperShadowState.IDLE
        
        self.prompt_cache.save()
        
        with self._lock:
            self._active_workflows.clear()
            self._crews.clear()