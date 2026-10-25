#!/usr/bin/env python3
"""
Superhuman Integration Tests for Shadow AI

Tests the full integration of all new components:
- Self-Improvement Module
- Multi-Agent Coordination
- Prompt Caching
- Super Memory
- Workflow Engine
- Tools
- Super Orchestrator
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime

# All component imports
from shadowd.self_improvement import FunctionDB, SelfBuilder, GeneratedFunction
from shadowd.self_improvement.function_db import FunctionStatus
from shadowd.self_improvement.self_builder import TaskType, SynthesisRequest
from shadowd.agents import Agent, AgentRole, AgentCapability, Task, TaskQueue, AgentRegistry, Crew
from shadowd.agents.coordination import TaskPriority
from shadowd.cache import PromptCacheManager, CacheStatus
from shadowd.memory.super_memory import SuperMemory
from shadowd.workflow.engine import Workflow, WorkflowEngine, FunctionBlock, StuckDetector
from shadowd.tools.enhanced_tools import ToolRegistry, FileReadTool, FileWriteTool
from shadowd.super_orchestrator import SuperShadowOrchestrator, SuperShadowState


class TestEndToEnd:
    """Test complete end-to-end integration."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        return SuperShadowOrchestrator(data_dir=temp_dir)
    
    @pytest.mark.asyncio
    async def test_full_processing_pipeline(self, orchestrator):
        """Test complete processing pipeline from input to output."""
        # First request
        result1 = await orchestrator.process({
            "query": "Analyze the data",
            "metadata": {"type": "analysis"}
        })
        
        assert "result" in result1
        assert result1["cached"] is False
        assert "processing_time" in result1
        
        # Cached request
        result2 = await orchestrator.process({
            "query": "Analyze the data",
            "metadata": {"type": "analysis"}
        })
        
        assert result2["cached"] is True
        assert orchestrator.stats.cache_hits >= 1
    
    @pytest.mark.asyncio
    async def test_learning_pipeline(self, orchestrator):
        """Test learning and self-improvement pipeline."""
        # Initial status
        initial_functions = len(orchestrator.list_functions())
        
        # Learn from observation
        result = await orchestrator.learn({
            "pattern": "test_pattern",
            "description": "Test pattern for analysis",
            "data": {"key": "value"}
        })
        
        assert result["learned"] is True
        assert orchestrator.stats.learning_events >= 1
        
        # Memory should be updated
        status = orchestrator.get_status()
        assert "memory" in status
    
    @pytest.mark.asyncio
    async def test_multi_agent_workflow(self, temp_dir):
        """Test multi-agent coordination through workflow."""
        orchestrator = SuperShadowOrchestrator(data_dir=temp_dir)
        
        result = await orchestrator.process({
            "query": "Research the topic, coordinate analysis, and produce summary",
            "requires_coordination": True
        })
        
        assert "result" in result
        assert orchestrator.stats.tasks_processed >= 1


class TestSelfImprovementWorkflow:
    """Test self-improvement integration with other components."""
    
    @pytest.fixture
    def function_db(self, temp_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield FunctionDB(storage_path=Path(tmpdir) / "functions")
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_function_lifecycle(self, function_db):
        """Test complete function lifecycle."""
        # Create function
        func = GeneratedFunction(
            name="test_multiply",
            description="Multiply two numbers",
            code="def test_multiply(a, b): return a * b",
            status=FunctionStatus.DRAFT,
            tags=["calculation", "math"],
        )
        
        # Register
        func_id = function_db.register(func)
        assert func_id is not None
        
        # Get and execute
        result = function_db.execute(func_id, 3, 4)
        
        assert result.success
        assert result.result == 12
        
        # Check statistics
        stats = function_db.get_statistics()
        assert stats["total_functions"] >= 1
        assert stats["total_executions"] >= 1
    
    def test_function_improvement_cycle(self, function_db):
        """Test function improvement cycle."""
        # Create original function
        original = GeneratedFunction(
            name="improve_test",
            description="Test function",
            code="def improve_test(x): return x",
            status=FunctionStatus.APPROVED,
        )
        
        func_id = function_db.register(original)
        
        # Create improved version
        improved_id = function_db.create_improved_version(
            func_id,
            "def improve_test(x): return x * 2  # Improved version",
            reason="Added multiplication"
        )
        
        # Verify version increased
        improved_func = function_db.get(improved_id)
        assert improved_func.version == 2
        assert improved_func.parent_id == func_id


class TestAgentMemoryIntegration:
    """Test agents working with memory."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def memory(self, temp_dir):
        return SuperMemory(storage_dir=temp_dir / "memory")
    
    @pytest.fixture
    def registry(self):
        reg = AgentRegistry()
        for agent in Crew.create_default_agents():
            reg.register(agent)
        return reg
    
    @pytest.mark.asyncio
    async def test_agent_with_memory_context(self, memory, registry):
        """Test agent execution with memory context."""
        # Store something in memory
        memory.remember(
            content=json.dumps({"fact": "The sky is blue", "importance": 0.8}),
            metadata={"type": "knowledge"},
            importance=0.8
        )
        
        # Get memory context
        context = memory.get_full_context()
        
        assert len(context) > 0
        
        # Create task with memory context
        task = Task(
            task_id="memory_task",
            name="Use Memory",
            description="Use stored memory",
            context={"memory_context": context},
        )
        
        # Create crew
        crew = Crew(name="memory_crew", registry=registry)
        crew.add_task(task)
        
        # Execute
        results = await crew.execute_sequential()
        
        assert len(results) > 0


class TestWorkflowCacheIntegration:
    """Test workflow with caching."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def workflow_engine(self, temp_dir):
        return WorkflowEngine(storage_path=temp_dir / "workflows")
    
    @pytest.fixture
    def cache(self, temp_dir):
        return PromptCacheManager(storage_path=temp_dir / "cache")
    
    @pytest.mark.asyncio
    async def test_workflow_with_caching(self, workflow_engine, cache):
        """Test workflow execution with caching."""
        # Create workflow
        workflow = Workflow(name="cached_workflow")
        
        results = []
        
        def track_step(context):
            results.append("step1")
            return {"tracked": "step1"}
        
        def track_step2(context):
            results.append("step2")
            return {"tracked": "step2"}
        
        block1 = FunctionBlock(
            func=track_step,
            block_id="step1",
            name="First Step"
        )
        
        block2 = FunctionBlock(
            func=track_step2,
            block_id="step2",
            name="Second Step"
        )
        block2.dependencies = ["step1"]
        
        workflow.add_block(block1)
        workflow.add_block(block2)
        
        # Store workflow in engine
        workflow_engine.workflows[workflow.workflow_id] = workflow
        
        # Execute
        exec_results = await workflow_engine.execute_workflow(workflow.workflow_id)
        
        # Cache workflow result
        workflow_key = f"workflow_{workflow.workflow_id}"
        cache.set(workflow_key, {"results": {k: v.status.value for k, v in exec_results.items()}})
        
        # Retrieve from cache
        status, cached = cache.get(workflow_key)
        
        assert status == CacheStatus.HIT
        assert "results" in cached
    
    def test_cache_invalidation(self, cache, temp_dir):
        """Test cache invalidation with file changes."""
        # Create test file
        test_file = temp_dir / "test_data.txt"
        test_file.write_text("initial data")
        
        # Cache with file dependency
        cache.set("test_query", {"answer": "test"}, file_paths=[str(test_file)])
        
        # Verify cached
        status, _ = cache.get("test_query", file_paths=[str(test_file)])
        assert status == CacheStatus.HIT
        
        # Modify file
        test_file.write_text("modified data")
        
        # Should be stale
        status, _ = cache.get("test_query", file_paths=[str(test_file)])
        assert status == CacheStatus.STALE


class TestStuckDetector:
    """Test stuck detection in complex workflows."""
    
    @pytest.fixture
    def detector(self):
        return StuckDetector()
    
    def test_detects_repeated_actions(self, detector):
        """Test detection of repeated actions."""
        # Record same action multiple times
        for _ in range(5):
            detector.record({"tool": "read", "path": "/same/file"})
        
        is_stuck, reason = detector.is_stuck()
        
        assert is_stuck
        assert "repeated" in reason.lower()
    
    def test_detects_error_loops(self, detector):
        """Test detection of error loops."""
        # Record actions with errors
        for i in range(4):
            detector.record({
                "tool": "execute",
                "error": "Permission denied"
            })
        
        is_stuck, reason = detector.is_stuck()
        
        assert is_stuck
    
    def test_provides_recovery_options(self, detector):
        """Test recovery options are provided."""
        # Get stuck
        for _ in range(5):
            detector.record({"tool": "same"})
        
        is_stuck, _ = detector.is_stuck()
        assert is_stuck
        
        # Get recovery options
        options = detector.get_stuck_recovery_options()
        
        assert len(options) > 0
        assert any("restart" in opt.lower() or "skip" in opt.lower() for opt in options)
    
    def test_allows_progress(self, detector):
        """Test that progress is not detected as stuck."""
        # Record different actions (making progress)
        for i in range(5):
            detector.record({"tool": f"action_{i}", "result": f"success_{i}"})
        
        is_stuck, _ = detector.is_stuck()
        
        assert not is_stuck


class TestToolRegistryIntegration:
    """Test tools integration with other components."""
    
    @pytest.fixture
    def registry(self):
        reg = ToolRegistry()
        reg.register(FileReadTool())
        reg.register(FileWriteTool())
        return reg
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.mark.asyncio
    async def test_tool_workflow(self, registry, temp_dir):
        """Test complete tool workflow."""
        test_file = temp_dir / "tool_test.txt"
        content = "Test content for tools"
        
        # Write using async execute
        write_tool = registry.get("file_write")
        write_result = await write_tool.execute(
            params={"path": str(test_file), "content": content}
        )
        
        assert write_result.success is True
        
        # Read using async execute
        read_tool = registry.get("file_read")
        read_result = await read_tool.execute(
            params={"path": str(test_file)}
        )
        
        assert read_result.success is True
        assert content in read_result.output
    
    def test_tool_schemas(self, registry):
        """Test tool schema generation."""
        schemas = registry.get_schemas()
        
        # Schemas is a list of dicts
        schema_names = [s["name"] for s in schemas]
        
        assert "file_read" in schema_names
        assert "file_write" in schema_names
        
        # Verify schema structure
        for schema in schemas:
            assert "name" in schema
            assert "parameters" in schema


class TestOrchestratorIntegration:
    """Test orchestrator with all components."""
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        return SuperShadowOrchestrator(data_dir=temp_dir, max_agents=5)
    
    @pytest.mark.asyncio
    async def test_orchestrator_status(self, orchestrator):
        """Test orchestrator status reporting."""
        status = orchestrator.get_status()
        
        assert status["state"] == "idle"
        assert "stats" in status
        assert "agents" in status
        assert "functions" in status
        assert "cache" in status
    
    @pytest.mark.asyncio
    async def test_orchestrator_shutdown(self, orchestrator, temp_dir):
        """Test graceful shutdown."""
        # Process something
        await orchestrator.process({"query": "test"})
        
        # Shutdown
        await orchestrator.shutdown()
        
        assert orchestrator.state == "idle"
    
    @pytest.mark.asyncio
    async def test_agent_management(self, orchestrator):
        """Test agent management through orchestrator."""
        # Get existing agent
        agent = orchestrator.get_agent("coordinator_1")
        
        assert agent is not None
        assert agent.role == AgentRole.COORDINATOR
    
    @pytest.mark.asyncio
    async def test_function_management(self, orchestrator):
        """Test function management through orchestrator."""
        # List functions
        functions = orchestrator.list_functions()
        
        assert isinstance(functions, list)
    
    @pytest.mark.asyncio
    async def test_cache_management(self, orchestrator):
        """Test cache management."""
        # Set cache
        orchestrator.prompt_cache.set("test", {"data": "test"})
        
        # Get cache stats
        stats = orchestrator.prompt_cache.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
        
        # Clear cache
        orchestrator.clear_cache()
        
        # Verify cleared
        status, _ = orchestrator.prompt_cache.get("test")
        assert status in [CacheStatus.MISS, CacheStatus.STALE]


class TestAnalysisFunctionality:
    """Test input analysis and routing."""
    
    @pytest.fixture
    def orchestrator(self, temp_dir):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield SuperShadowOrchestrator(data_dir=Path(tmpdir))
    
    @pytest.fixture
    def temp_dir(self):
        with tempfile.TemporaryDirectory() as tmpdir:
            yield Path(tmpdir)
    
    def test_identifies_multi_agent_needs(self, orchestrator):
        """Test identification of multi-agent needs."""
        analysis = orchestrator._analyze_input({
            "query": "Research and coordinate multiple tasks"
        }, {})
        
        assert analysis["requires_multi_agent"] is True
        assert len(analysis["suggested_agents"]) > 0
    
    def test_identifies_workflow_needs(self, orchestrator):
        """Test identification of workflow needs."""
        analysis = orchestrator._analyze_input({
            "query": "Create a pipeline for data processing"
        }, {})
        
        assert analysis["requires_workflow"] is True
    
    def test_identifies_function_needs(self, orchestrator):
        """Test identification of function generation needs."""
        analysis = orchestrator._analyze_input({
            "query": "Generate a new function for calculation"
        }, {})
        
        assert analysis["requires_function"] is True
    
    def test_determines_complexity(self, orchestrator):
        """Test complexity determination."""
        # Simple
        simple = orchestrator._analyze_input({"query": "test"}, {})
        
        # Complex
        complex_input = {
            "query": "Analyze this complex multi-step process: " + "x" * 500,
            "data": list(range(100))
        }
        complex_analysis = orchestrator._analyze_input(complex_input, {})
        
        # Both should have complexity assessment
        assert simple["complexity"] in ["low", "medium", "high"]
        assert complex_analysis["complexity"] in ["low", "medium", "high"]


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])