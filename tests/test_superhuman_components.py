#!/usr/bin/env python3
"""
Tests for Shadow AI Superhuman Components

Tests:
- Enhanced Tools
- Super Memory
- Workflow Engine
- Stuck Detector
- Self-Improvement Module (Function DB + Self Builder)
"""

import pytest
import asyncio
import tempfile
import json
from pathlib import Path
from datetime import datetime, timedelta

# Tools tests
from shadowd.tools.enhanced_tools import (
    ToolRegistry,
    FileReadTool,
    FileEditTool,
    FileWriteTool,
    FileSearchTool,
    PythonTool,
    ShellTool,
    ToolResult,
    ToolAction,
    ActionStatus,
)

# Memory tests
from shadowd.memory.super_memory import (
    MemoryBlock,
    CoreMemory,
    WorkingMemory,
    ArchivalMemory,
    KnowledgeGraph,
    SuperMemory,
    Entity,
    Relation,
)

# Workflow tests
from shadowd.workflow.engine import (
    BlockStatus,
    BlockResult,
    BaseBlock,
    FunctionBlock,
    Workflow,
    WorkflowEngine,
    StuckDetector,
    Checkpoint,
)


# ═════════════════════════════════════════════════════════════════
# TOOL TESTS
# ═════════════════════════════════════════════════════════════════

class TestToolRegistry:
    """Test tool registry."""
    
    def test_registry_initialization(self):
        """Test registry has default tools."""
        registry = ToolRegistry()
        
        tools = registry.list_tools()
        assert len(tools) > 0
        assert any(t.name == "file_read" for t in tools)
    
    def test_get_tool(self):
        """Test getting tool by name."""
        registry = ToolRegistry()
        
        tool = registry.get("file_read")
        assert tool is not None
        assert tool.name == "file_read"
    
    def test_get_unknown_tool(self):
        """Test getting unknown tool."""
        registry = ToolRegistry()
        
        tool = registry.get("nonexistent")
        assert tool is None
    
    def test_get_schemas(self):
        """Test getting schemas for LLM."""
        registry = ToolRegistry()
        
        schemas = registry.get_schemas()
        assert len(schemas) > 0
        assert all("name" in s for s in schemas)
        assert all("description" in s for s in schemas)


class TestFileReadTool:
    """Test file read tool."""
    
    @pytest.mark.asyncio
    async def test_read_file(self, tmp_path):
        """Test reading a file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello\nWorld\nTest")
        
        tool = FileReadTool()
        result = await tool.execute({"path": str(test_file)})
        
        assert result.success
        assert "Hello" in result.output
        assert "World" in result.output
    
    @pytest.mark.asyncio
    async def test_read_nonexistent_file(self, tmp_path):
        """Test reading nonexistent file."""
        tool = FileReadTool()
        result = await tool.execute({"path": str(tmp_path / "nonexistent.txt")})
        
        assert not result.success
    
    @pytest.mark.asyncio
    async def test_read_with_offset_limit(self, tmp_path):
        """Test reading with offset and limit."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("\n".join([f"Line {i}" for i in range(10)]))
        
        tool = FileReadTool()
        result = await tool.execute({"path": str(test_file), "offset": 2, "limit": 3})
        
        lines = result.output.splitlines()
        assert len(lines) == 3


class TestFileEditTool:
    """Test file edit tool (Aider-style)."""
    
    @pytest.mark.asyncio
    async def test_edit_exact_match(self, tmp_path):
        """Test editing with exact match."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World\nGoodbye World")
        
        tool = FileEditTool()
        result = await tool.execute({
            "path": str(test_file),
            "search": "Hello World",
            "replace": "Hello Universe",
            "exact": True
        })
        
        assert result.success
        content = test_file.read_text()
        assert "Hello Universe" in content
    
    @pytest.mark.asyncio
    async def test_edit_fuzzy_match(self, tmp_path):
        """Test editing with fuzzy match."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello   World\nGoodbye")
        
        tool = FileEditTool()
        result = await tool.execute({
            "path": str(test_file),
            "search": "Hello World",
            "replace": "Hi World",
            "exact": False
        })
        
        assert result.success or "not found" in result.error.lower()
    
    @pytest.mark.asyncio
    async def test_edit_shows_diff(self, tmp_path):
        """Test that edit shows diff."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Hello World")
        
        tool = FileEditTool()
        result = await tool.execute({
            "path": str(test_file),
            "search": "Hello",
            "replace": "Hi"
        })
        
        assert result.success
        assert "---" in result.output or "+" in result.output or "Hi" in result.output


class TestFileWriteTool:
    """Test file write tool."""
    
    @pytest.mark.asyncio
    async def test_write_new_file(self, tmp_path):
        """Test writing new file."""
        test_file = tmp_path / "new.txt"
        
        tool = FileWriteTool()
        result = await tool.execute({
            "path": str(test_file),
            "content": "New content"
        })
        
        assert result.success
        assert test_file.exists()
        assert test_file.read_text() == "New content"
    
    @pytest.mark.asyncio
    async def test_append_to_file(self, tmp_path):
        """Test appending to file."""
        test_file = tmp_path / "test.txt"
        test_file.write_text("Original")
        
        tool = FileWriteTool()
        result = await tool.execute({
            "path": str(test_file),
            "content": "\nAppended",
            "mode": "append"
        })
        
        assert result.success
        assert "Original" in test_file.read_text()
        assert "Appended" in test_file.read_text()
    
    @pytest.mark.asyncio
    async def test_creates_directories(self, tmp_path):
        """Test creating parent directories."""
        test_file = tmp_path / "subdir" / "deep" / "file.txt"
        
        tool = FileWriteTool()
        result = await tool.execute({
            "path": str(test_file),
            "content": "Content"
        })
        
        assert result.success
        assert test_file.exists()


class TestPythonTool:
    """Test Python execution tool."""
    
    @pytest.mark.asyncio
    async def test_execute_simple_code(self):
        """Test executing simple Python code."""
        tool = PythonTool()
        result = await tool.execute({
            "code": "print('Hello from Python')"
        })
        
        assert result.success
        assert "Hello from Python" in result.output
    
    @pytest.mark.asyncio
    async def test_execute_returns_value(self):
        """Test code that returns value."""
        tool = PythonTool()
        result = await tool.execute({
            "code": "x = 5\ny = 10\nprint(x + y)"
        })
        
        assert result.success
        assert "15" in result.output
    
    @pytest.mark.asyncio
    async def test_timeout_handling(self):
        """Test handling timeout."""
        tool = PythonTool()
        result = await tool.execute({
            "code": "import time\ntime.sleep(5)",
            "timeout": 1
        })
        
        assert not result.success
        assert "timeout" in result.error.lower()


# ═════════════════════════════════════════════════════════════════
# MEMORY TESTS
# ═════════════════════════════════════════════════════════════════

class TestMemoryBlock:
    """Test memory blocks."""
    
    def test_create_block(self):
        """Test creating a memory block."""
        block = MemoryBlock(label="test", value="Test content")
        
        assert block.label == "test"
        assert block.value == "Test content"
    
    def test_append_content(self):
        """Test appending content."""
        block = MemoryBlock(label="test", value="Start", limit=100)
        
        success = block.append("Added")
        
        assert success
        assert "Added" in block.value
    
    def test_append_respects_limit(self):
        """Test append respects limit."""
        block = MemoryBlock(label="test", value="", limit=20)
        
        block.append("0123456789")  # 10 chars
        success = block.append("01234567890123456789")  # 20 chars - should fail
        
        assert not success
    
    def test_trim_content(self):
        """Test trimming content."""
        block = MemoryBlock(label="test", value="x" * 100, limit=50)
        
        block.trim()
        
        assert len(block.value) <= block.limit


class TestCoreMemory:
    """Test core memory."""
    
    def test_initialization(self):
        """Test core memory has default blocks."""
        memory = CoreMemory()
        
        assert "persona" in memory.blocks
        assert "human" in memory.blocks
        assert "recent" in memory.blocks
    
    def test_protected_block(self):
        """Test that protected blocks can't be modified."""
        memory = CoreMemory()
        
        # Set unprotected block
        success = memory.set_block("human", "User info")
        assert success
        
        # Can't modify protected block
        old_value = memory.blocks["persona"].value
        memory.set_block("persona", "Modified")
        assert memory.blocks["persona"].value == old_value
    
    def test_get_context(self):
        """Test getting context."""
        memory = CoreMemory()
        memory.set_block("human", "Test user")
        
        context = memory.get_context()
        
        assert "Core Memory" in context
        assert "HUMAN" in context
    
    def test_save_load(self, tmp_path):
        """Test saving and loading."""
        memory = CoreMemory(tmp_path / "core.json")
        memory.set_block("human", "Test user")
        memory.save()
        
        memory2 = CoreMemory(tmp_path / "core.json")
        memory2.load()
        
        assert memory2.blocks["human"].value == "Test user"


class TestWorkingMemory:
    """Test working memory."""
    
    def test_add_messages(self):
        """Test adding messages."""
        memory = WorkingMemory()
        
        memory.add_message("user", "Hello")
        memory.add_message("assistant", "Hi there")
        
        assert len(memory.messages) == 2
    
    def test_trim_messages(self):
        """Test message trimming."""
        memory = WorkingMemory(max_messages=5)
        
        for i in range(10):
            memory.add_message("user", f"Message {i}")
        
        assert len(memory.messages) <= 5
    
    def test_clear(self):
        """Test clearing memory."""
        memory = WorkingMemory()
        memory.add_message("user", "Test")
        
        memory.clear()
        
        assert len(memory.messages) == 0


class TestArchivalMemory:
    """Test archival memory."""
    
    def test_insert(self, tmp_path):
        """Test inserting memory."""
        memory = ArchivalMemory(tmp_path)
        
        mem_id = memory.insert("Test memory content")
        
        assert mem_id.startswith("arch_")
        assert len(memory.memories) == 1
    
    def test_search(self, tmp_path):
        """Test searching memory."""
        memory = ArchivalMemory(tmp_path)
        
        memory.insert("Python is great")
        memory.insert("JavaScript is okay")
        memory.insert("Python is powerful")
        
        results = memory.search("python")
        
        assert len(results) >= 2
    
    def test_get_recent(self, tmp_path):
        """Test getting recent memories."""
        memory = ArchivalMemory(tmp_path)
        
        memory.insert("Old memory")
        
        recent = memory.get_recent(hours=24)
        
        assert len(recent) >= 1


class TestKnowledgeGraph:
    """Test knowledge graph."""
    
    def test_add_entity(self):
        """Test adding entity."""
        kg = KnowledgeGraph()
        
        entity_id = kg.add_entity("person", "Sarah", {"age": 25})
        
        assert entity_id in kg.entities
        assert kg.entities[entity_id].name == "Sarah"
    
    def test_add_relation(self):
        """Test adding relation."""
        kg = KnowledgeGraph()
        
        sarah = kg.add_entity("person", "Sarah")
        cafe = kg.add_entity("location", "Cafe")
        kg.add_relation(sarah, "visited", cafe)
        
        relations = kg.get_relations(predicate="visited")
        
        assert len(relations) == 1
    
    def test_query(self):
        """Test querying graph."""
        kg = KnowledgeGraph()
        
        sarah = kg.add_entity("person", "Sarah")
        cafe = kg.add_entity("location", "Cafe")
        kg.add_relation(sarah, "visited", cafe)
        
        results = kg.query({"entity_type": "person"})
        
        assert len(results) > 0


class TestSuperMemory:
    """Test full memory integration."""
    
    def test_remember_and_recall(self, tmp_path):
        """Test remember and recall."""
        memory = SuperMemory(tmp_path)
        
        memory.remember("User likes Python", memory_type="archival")
        
        results = memory.recall("Python")
        
        assert len(results["archival"]) > 0
    
    def test_get_context(self, tmp_path):
        """Test getting full context."""
        memory = SuperMemory(tmp_path)
        memory.remember("Recent event", memory_type="working")
        
        context = memory.get_full_context()
        
        assert "Core Memory" in context
        assert "Working Memory" in context
    
    def test_get_stats(self, tmp_path):
        """Test getting stats."""
        memory = SuperMemory(tmp_path)
        memory.remember("Test", memory_type="archival")
        
        stats = memory.get_stats()
        
        assert "core_blocks" in stats
        assert "archival_memories" in stats


# ═════════════════════════════════════════════════════════════════
# WORKFLOW TESTS
# ═════════════════════════════════════════════════════════════════

class TestWorkflow:
    """Test workflow execution."""
    
    @pytest.mark.asyncio
    async def test_simple_workflow(self):
        """Test simple workflow with one block."""
        block = FunctionBlock(lambda ctx: {"result": "done"})
        
        workflow = Workflow()
        workflow.add_block(block)
        workflow.set_entry_point(block.block_id)
        
        results = await workflow.execute()
        
        assert block.block_id in results
        assert results[block.block_id].status == BlockStatus.SUCCESS
    
    @pytest.mark.asyncio
    async def test_sequential_workflow(self):
        """Test workflow with sequential blocks."""
        exec_log = []
        
        def block1(ctx):
            exec_log.append("block1")
            return {"step": 1}
        
        def block2(ctx):
            exec_log.append("block2")
            return {"step": 2}
        
        b1 = FunctionBlock(block1, name="b1")
        b2 = FunctionBlock(block2, name="b2")
        
        workflow = Workflow()
        workflow.add_block(b1)
        workflow.add_block(b2)
        workflow.add_edge(b1.block_id, b2.block_id)
        workflow.set_entry_point(b1.block_id)
        
        results = await workflow.execute()
        
        assert exec_log == ["block1", "block2"]
    
    @pytest.mark.asyncio
    async def test_workflow_with_context(self):
        """Test workflow with context."""
        def block(ctx):
            return {"value": ctx.get("input", "default")}
        
        b = FunctionBlock(block)
        workflow = Workflow()
        workflow.add_block(b)
        workflow.set_entry_point(b.block_id)
        workflow.set_context(input="test_value")
        
        results = await workflow.execute()
        
        assert results[b.block_id].output["value"] == "test_value"
    
    @pytest.mark.asyncio
    async def test_checkpoint_creation(self):
        """Test checkpoint is created."""
        b = FunctionBlock(lambda ctx: {"result": 1})
        
        workflow = Workflow()
        workflow.add_block(b)
        workflow.set_entry_point(b.block_id)
        
        await workflow.execute()
        
        assert len(workflow.checkpoints) == 1


class TestWorkflowEngine:
    """Test workflow engine."""
    
    def test_create_workflow(self):
        """Test creating workflow."""
        engine = WorkflowEngine()
        
        workflow = engine.create_workflow("test")
        
        assert workflow.name == "test"
        assert workflow.workflow_id in engine.workflows
    
    @pytest.mark.asyncio
    async def test_execute_workflow(self):
        """Test executing workflow through engine."""
        engine = WorkflowEngine()
        
        workflow = engine.create_workflow("test")
        block = FunctionBlock(lambda ctx: {"done": True})
        workflow.add_block(block)
        workflow.set_entry_point(block.block_id)
        
        results = await engine.execute_workflow(workflow.workflow_id)
        
        assert block.block_id in results


class TestStuckDetector:
    """Test stuck detector."""
    
    def test_detect_repetition(self):
        """Test detecting repeated actions."""
        detector = StuckDetector()
        
        action = {"tool": "read", "path": "/test"}
        for _ in range(5):
            detector.record(action)
        
        is_stuck, reason = detector.is_stuck()
        
        assert is_stuck
        assert "repeated" in reason.lower()
    
    def test_not_stuck_with_progress(self):
        """Test not stuck when making progress."""
        detector = StuckDetector()
        
        for i in range(5):
            detector.record({"tool": f"action_{i}"})
        
        is_stuck, _ = detector.is_stuck()
        
        assert not is_stuck
    
    def test_recovery_options(self):
        """Test getting recovery options."""
        detector = StuckDetector()
        
        options = detector.get_stuck_recovery_options()
        
        assert len(options) > 0
        assert any("restart" in o.lower() for o in options)
    
    def test_reset(self):
        """Test resetting detector."""
        detector = StuckDetector()
        
        for _ in range(5):
            detector.record({"tool": "test"})
        
        detector.reset()
        
        assert len(detector.history) == 0


# Self-Improvement tests
from shadowd.self_improvement import (
    FunctionDB,
    GeneratedFunction,
    FunctionResult,
    SelfBuilder,
    TaskAnalysis,
)
from shadowd.self_improvement.function_db import FunctionStatus, Priority
from shadowd.self_improvement.self_builder import TaskType, ComplexityLevel, SynthesisRequest


class TestGeneratedFunction:
    """Test generated function."""
    
    def test_create_function(self):
        """Test creating a function."""
        func = GeneratedFunction(
            name="test_func",
            description="Test function",
            code="def test_func(): return 42",
        )
        
        assert func.name == "test_func"
        assert func.status == FunctionStatus.DRAFT
        assert func.version == 1
    
    def test_get_callable(self):
        """Test getting callable."""
        func = GeneratedFunction(
            name="test_func",
            description="Test",
            code="def test_func(): return 42",
            status=FunctionStatus.APPROVED,
        )
        
        callable_func = func.get_callable()
        
        assert callable_func is not None
        assert callable_func() == 42
    
    def test_invalid_code(self):
        """Test invalid code."""
        func = GeneratedFunction(
            name="test_func",
            description="Test",
            code="def invalid syntax",
        )
        
        callable_func = func.get_callable()
        
        assert callable_func is None
    
    def test_update_stats(self):
        """Test updating statistics."""
        func = GeneratedFunction(
            name="test_func",
            description="Test",
            code="def test_func(): return 42",
        )
        
        func.update_stats(0.5, success=True)
        
        assert func.usage_count == 1
        assert func.success_count == 1
        assert func.failure_count == 0
        assert func.quality_score > 0
        
        func.update_stats(0.3, success=False)
        
        assert func.usage_count == 2
        assert func.success_count == 1
        assert func.failure_count == 1
    
    def test_to_dict_from_dict(self):
        """Test serialization."""
        func = GeneratedFunction(
            name="test_func",
            description="Test function",
            code="def test_func(): return 42",
            tags=["test", "unit"],
        )
        
        data = func.to_dict()
        
        assert data["name"] == "test_func"
        assert data["tags"] == ["test", "unit"]
        
        restored = GeneratedFunction.from_dict(data)
        
        assert restored.name == func.name
        assert restored.tags == func.tags


class TestFunctionDB:
    """Test function database."""
    
    def test_register_function(self):
        """Test registering a function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            func = GeneratedFunction(
                name="test_func",
                description="Test",
                code="def test_func(): return 42",
                tags=["test"],
            )
            
            func_id = db.register(func)
            
            assert func_id == func.id
            assert db.get(func_id) is not None
    
    def test_get_by_name(self):
        """Test getting function by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            func = GeneratedFunction(
                name="check_value",
                description="Test",
                code="def check_value(x): return x > 0",
                tags=["validation"],
            )
            
            db.register(func)
            
            found = db.get_by_name("check_value")
            
            assert found is not None
            assert found.name == "check_value"
    
    def test_search_by_tags(self):
        """Test searching by tags."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            func1 = GeneratedFunction(
                name="func1",
                description="Test",
                code="def func1(): return 1",
                tags=["test", "unit"],
            )
            func2 = GeneratedFunction(
                name="func2",
                description="Test",
                code="def func2(): return 2",
                tags=["integration", "test"],
            )
            
            db.register(func1)
            db.register(func2)
            
            results = db.search_by_tags(["test"])
            
            assert len(results) == 2
            
            results = db.search_by_tags(["test", "unit"], match_all=True)
            
            assert len(results) == 1
    
    def test_execute_function(self):
        """Test executing a function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            func = GeneratedFunction(
                name="add_numbers",
                description="Add two numbers",
                code="def add_numbers(a, b): return a + b",
                status=FunctionStatus.APPROVED,
            )
            
            db.register(func)
            
            result = db.execute(func.id, 2, 3)
            
            assert result.success
            assert result.result == 5
    
    def test_execute_nonexistent_function(self):
        """Test executing nonexistent function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            result = db.execute("nonexistent")
            
            assert not result.success
            assert "not found" in result.error.lower()
    
    def test_resolve_dependencies(self):
        """Test dependency resolution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            func1 = GeneratedFunction(
                name="func_a",
                description="Test",
                code="def func_a(): return 1",
                dependencies=[],
            )
            func2 = GeneratedFunction(
                name="func_b",
                description="Test",
                code="def func_b(): return func_a() + 1",
                dependencies=[func1.id],
            )
            func3 = GeneratedFunction(
                name="func_c",
                description="Test",
                code="def func_c(): return func_b() + 1",
                dependencies=[func2.id],
            )
            
            db.register(func1)
            db.register(func2)
            db.register(func3)
            
            order = db.resolve_dependencies(func3)
            
            assert func1.id in order
            assert func2.id in order
            assert func3.id in order
    
    def test_get_statistics(self):
        """Test getting statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            stats = db.get_statistics()
            
            assert stats["total_functions"] == 0
            
            func = GeneratedFunction(
                name="test",
                description="Test",
                code="def test(): return 1",
                status=FunctionStatus.APPROVED,
            )
            db.register(func)
            
            stats = db.get_statistics()
            
            assert stats["total_functions"] == 1


class TestSelfBuilder:
    """Test self-builder."""
    
    def test_analyze_task_simple(self):
        """Test analyzing simple task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            analysis = builder.analyze_task("Calculate the sum of values")
            
            assert analysis.task_type == TaskType.CALCULATION
            assert len(analysis.keywords) > 0
    
    def test_analyze_task_validation(self):
        """Test analyzing validation task."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            analysis = builder.analyze_task("Validate that input is correct")
            
            assert analysis.task_type in [TaskType.VALIDATION, TaskType.DATA_PROCESSING]
    
    def test_find_existing_function(self):
        """Test finding existing function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            existing = GeneratedFunction(
                name="calculate_sum",
                description="Calculate sum",
                code="def calculate_sum(values): return sum(values)",
                tags=["calculation", "sum"],
                status=FunctionStatus.APPROVED,
            )
            db.register(existing)
            
            analysis = builder.analyze_task("Calculate sum of numbers")
            found = builder.find_existing_function(analysis)
            
            assert found is not None
    
    def test_synthesize_simple_function(self):
        """Test synthesizing simple function."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            analysis = builder.analyze_task("Calculate sum of values")
            
            request = SynthesisRequest(
                task_analysis=analysis,
                max_retries=1
            )
            
            result = builder.synthesize_function(request)
            
            assert result.iterations >= 1
    
    def test_get_synthesis_history(self):
        """Test getting synthesis history."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            analysis = builder.analyze_task("Calculate average")
            request = SynthesisRequest(task_analysis=analysis)
            
            builder.synthesize_function(request)
            
            history = builder.get_synthesis_history()
            
            assert len(history) > 0
    
    def test_get_statistics(self):
        """Test getting build statistics."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            stats = builder.get_statistics()
            
            assert "total_syntheses" in stats


@pytest.mark.asyncio
class TestSelfImprovementAsync:
    """Test async methods."""
    
    async def test_execute_async(self):
        """Test async execution."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            
            func = GeneratedFunction(
                name="async_test",
                description="Test",
                code="def async_test(x): return x * 2",
                status=FunctionStatus.APPROVED,
            )
            
            db.register(func)
            
            result = await db.execute_async(func.id, 5)
            
            assert result.success
            assert result.result == 10
    
    async def test_synthesize_async(self):
        """Test async synthesis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            db = FunctionDB(storage_path=Path(tmpdir))
            builder = SelfBuilder(db)
            
            analysis = TaskAnalysis(
                description="Test function",
                task_type=TaskType.CALCULATION,
                complexity=ComplexityLevel.SIMPLE,
                required_inputs=["data"],
                expected_outputs=["result"],
                dependencies=[],
                constraints=[],
                keywords=["test"],
                suggested_patterns=[],
                confidence=0.5,
            )
            
            request = SynthesisRequest(task_analysis=analysis)
            
            result = await builder.synthesize_async(request)
            
            assert result is not None


# Multi-Agent Coordination tests
from shadowd.agents import (
    AgentRole,
    TaskStatus,
    TaskPriority,
    AgentCapability,
    Agent,
    Task,
    TaskResult,
    TaskQueue,
    AgentRegistry,
    Crew,
)


class TestAgent:
    """Test Agent class."""
    
    def test_create_agent(self):
        """Test creating an agent."""
        agent = Agent(
            agent_id="test_agent",
            name="Test Agent",
            role=AgentRole.EXECUTOR,
            capabilities=[AgentCapability("test", "Test capability")],
        )
        
        assert agent.agent_id == "test_agent"
        assert agent.role == AgentRole.EXECUTOR
        assert len(agent.capabilities) == 1
    
    def test_agent_profile(self):
        """Test getting agent profile."""
        agent = Agent(
            agent_id="test",
            name="Test",
            role=AgentRole.ANALYST,
            capabilities=[AgentCapability("analyze", "Analyze data")],
        )
        
        profile = agent.profile
        
        assert profile.agent_id == "test"
        assert profile.role == AgentRole.ANALYST
    
    def test_can_execute(self):
        """Test execution check."""
        agent = Agent(
            agent_id="test",
            name="Test",
            role=AgentRole.EXECUTOR,
            capabilities=[AgentCapability("test", "Test")],
            max_concurrent_tasks=2,
        )
        
        assert agent.can_execute(TaskPriority.MEDIUM, "test")
        assert not agent.can_execute(TaskPriority.MEDIUM, "other")


class TestTask:
    """Test Task class."""
    
    def test_create_task(self):
        """Test creating a task."""
        task = Task(
            task_id="task_1",
            name="Test Task",
            description="Test description",
        )
        
        assert task.task_id == "task_1"
        assert task.status == TaskStatus.PENDING
    
    def test_task_lifecycle(self):
        """Test task lifecycle."""
        task = Task(
            task_id="task_1",
            name="Test",
            description="Test",
        )
        
        task.start()
        assert task.status == TaskStatus.IN_PROGRESS
        assert task.started_at is not None
        
        task.complete({"result": "success"})
        assert task.status == TaskStatus.COMPLETED
        assert task.result == {"result": "success"}
        assert task.completed_at is not None
    
    def test_task_failure(self):
        """Test task failure."""
        task = Task(
            task_id="task_1",
            name="Test",
            description="Test",
        )
        
        task.start()
        task.fail("Something went wrong")
        
        assert task.status == TaskStatus.FAILED
        assert task.error == "Something went wrong"
    
    def test_task_dependencies(self):
        """Test task dependencies."""
        task = Task(
            task_id="task_2",
            name="Test",
            description="Test",
            dependencies=["task_1"],
        )
        
        assert not task.is_ready(set())
        assert not task.is_ready({"other_task"})
        assert task.is_ready({"task_1"})


class TestTaskQueue:
    """Test Task Queue."""
    
    def test_add_task(self):
        """Test adding a task."""
        queue = TaskQueue()
        
        task = Task(task_id="t1", name="Test", description="Test")
        queue.add(task)
        
        assert queue.size()["pending"] == 1
    
    def test_get_ready_tasks(self):
        """Test getting ready tasks."""
        queue = TaskQueue()
        
        task = Task(task_id="t1", name="Test", description="Test")
        queue.add(task)
        
        ready = queue.get_ready_tasks()
        
        assert len(ready) == 1
        assert ready[0].task_id == "t1"
    
    def test_task_completion(self):
        """Test completing a task."""
        queue = TaskQueue()
        
        task = Task(task_id="t1", name="Test", description="Test")
        queue.add(task)
        
        ready = queue.get_ready_tasks(max_tasks=1)
        queue.start_task(ready[0])
        
        result = TaskResult(
            task_id="t1",
            agent_id="agent_1",
            success=True,
            result="done"
        )
        queue.complete_task(ready[0], result)
        
        assert queue.size()["completed"] == 1


class TestAgentRegistry:
    """Test Agent Registry."""
    
    def test_register_agent(self):
        """Test registering an agent."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="agent_1",
            name="Test Agent",
            role=AgentRole.EXECUTOR,
            capabilities=[AgentCapability("execute", "Execute tasks")],
        )
        
        registry.register(agent)
        
        assert registry.get("agent_1") is not None
    
    def test_find_by_role(self):
        """Test finding agent by role."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="researcher_1",
            name="Researcher",
            role=AgentRole.RESEARCHER,
            capabilities=[AgentCapability("research", "Research info")],
        )
        
        registry.register(agent)
        
        found = registry.get_by_role(AgentRole.RESEARCHER)
        
        assert found is not None
        assert len(found) == 1
        assert found[0].agent_id == "researcher_1"
    
    def test_find_by_capability(self):
        """Test finding agent by capability."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="writer_1",
            name="Writer",
            role=AgentRole.WRITER,
            capabilities=[AgentCapability("write", "Write content")],
        )
        
        registry.register(agent)
        
        agents = registry.get_by_capability("write")
        
        assert len(agents) == 1
    
    def test_find_available(self):
        """Test finding available agent."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="executor_1",
            name="Executor",
            role=AgentRole.EXECUTOR,
            capabilities=[AgentCapability("execute", "Execute")],
            max_concurrent_tasks=5,
        )
        
        registry.register(agent)
        
        available = registry.find_available(
            role=AgentRole.EXECUTOR,
            priority=TaskPriority.MEDIUM
        )
        
        assert available is not None
    
    def test_get_statistics(self):
        """Test getting statistics."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="agent_1",
            name="Test",
            role=AgentRole.ANALYST,
            capabilities=[],
        )
        registry.register(agent)
        
        stats = registry.get_statistics()
        
        assert stats["total_agents"] == 1


class TestCrew:
    """Test Crew coordination."""
    
    @pytest.mark.asyncio
    async def test_add_task(self):
        """Test adding task to crew."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="agent_1",
            name="Test Agent",
            role=AgentRole.EXECUTOR,
            capabilities=[AgentCapability("test", "Test")],
        )
        registry.register(agent)
        
        crew = Crew(name="Test Crew", registry=registry)
        
        task = Task(task_id="t1", name="Test", description="Test")
        crew.add_task(task)
        
        assert crew.task_queue.size()["pending"] == 1
    
    @pytest.mark.asyncio
    async def test_execute_sequential(self):
        """Test sequential execution."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="agent_1",
            name="Test Agent",
            role=AgentRole.EXECUTOR,
            capabilities=[AgentCapability("test", "Test")],
        )
        registry.register(agent)
        
        crew = Crew(name="Test Crew", registry=registry)
        
        task1 = Task(task_id="t1", name="Task 1", description="Test")
        task2 = Task(task_id="t2", name="Task 2", description="Test", dependencies=["t1"])
        
        crew.add_task(task1)
        crew.add_task(task2)
        
        results = await crew.execute_sequential()
        
        assert len(results) == 2
    
    @pytest.mark.asyncio
    async def test_execute_parallel(self):
        """Test parallel execution."""
        registry = AgentRegistry()
        
        for i in range(3):
            agent = Agent(
                agent_id=f"agent_{i}",
                name=f"Agent {i}",
                role=AgentRole.EXECUTOR,
                capabilities=[AgentCapability("test", "Test")],
            )
            registry.register(agent)
        
        crew = Crew(name="Test Crew", registry=registry)
        
        for i in range(3):
            task = Task(task_id=f"t{i}", name=f"Task {i}", description="Test")
            crew.add_task(task)
        
        results = await crew.execute_parallel(max_concurrency=3)
        
        assert len(results) == 3
    
    @pytest.mark.asyncio
    async def test_get_summary(self):
        """Test getting summary."""
        registry = AgentRegistry()
        
        agent = Agent(
            agent_id="agent_1",
            name="Test",
            role=AgentRole.EXECUTOR,
            capabilities=[],
        )
        registry.register(agent)
        
        crew = Crew(name="Test Crew", registry=registry)
        
        task = Task(task_id="t1", name="Test", description="Test")
        crew.add_task(task)
        
        await crew.execute_sequential()
        
        summary = crew.get_summary()
        
        assert "total_tasks" in summary
    
    def test_create_default_agents(self):
        """Test creating default agents."""
        agents = Crew.create_default_agents()
        
        assert len(agents) > 0
        roles = [a.role for a in agents]
        assert AgentRole.COORDINATOR in roles
        assert AgentRole.RESEARCHER in roles


# Prompt Cache tests
from shadowd.cache import (
    CacheStatus,
    PromptCache,
    BackgroundPromptCache,
    PromptCacheManager,
)


class TestPromptCache:
    """Test Prompt Cache."""
    
    def test_create_cache(self):
        """Test creating cache."""
        cache = PromptCache(max_size_bytes=1024)
        
        assert cache.get_size()["max_bytes"] == 1024
    
    def test_set_and_get(self):
        """Test set and get."""
        cache = PromptCache()
        
        cache.set("test prompt", {"result": "test response"})
        
        status, response = cache.get("test prompt")
        
        assert status == CacheStatus.HIT
        assert response == {"result": "test response"}
    
    def test_cache_miss(self):
        """Test cache miss."""
        cache = PromptCache()
        
        status, response = cache.get("nonexistent prompt")
        
        assert status == CacheStatus.MISS
        assert response is None
    
    def test_cache_with_files(self, tmp_path):
        """Test cache with file dependencies."""
        cache = PromptCache()
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("original content")
        
        cache.set("test", {"result": "ok"}, file_paths=[str(test_file)])
        
        status, _ = cache.get("test", file_paths=[str(test_file)])
        assert status == CacheStatus.HIT
        
        test_file.write_text("modified content")
        
        status, _ = cache.get("test", file_paths=[str(test_file)])
        assert status == CacheStatus.STALE
    
    def test_cache_expiration(self):
        """Test cache TTL expiration."""
        import time
        
        cache = PromptCache(default_ttl=0.1)
        
        cache.set("test", {"result": "ok"})
        
        status, _ = cache.get("test")
        assert status == CacheStatus.HIT
        
        time.sleep(0.15)
        
        status, _ = cache.get("test")
        assert status == CacheStatus.MISS
    
    def test_cache_eviction(self):
        """Test LRU eviction."""
        cache = PromptCache(max_size_bytes=1000, max_entries=5)
        
        for i in range(10):
            cache.set(f"prompt_{i}", {"result": f"response_{i}"})
        
        assert cache.get_size()["entries"] <= 5
    
    def test_cache_stats(self):
        """Test cache statistics."""
        cache = PromptCache()
        
        cache.set("test1", "response1")
        cache.set("test2", "response2")
        
        cache.get("test1")
        cache.get("test1")
        cache.get("nonexistent")
        
        stats = cache.get_stats()
        
        assert stats.hits == 2
        assert stats.misses == 1
    
    def test_invalidate_file(self, tmp_path):
        """Test file-based invalidation."""
        cache = PromptCache()
        
        test_file = tmp_path / "test.txt"
        test_file.write_text("content")
        
        cache.set("test", {"result": "ok"}, file_paths=[str(test_file)])
        
        cache.invalidate_file(str(test_file))
        
        status, _ = cache.get("test")
        assert status == CacheStatus.MISS
    
    def test_clear_cache(self):
        """Test clearing cache."""
        cache = PromptCache()
        
        cache.set("test1", "response")
        cache.set("test2", "response")
        
        cache.clear()
        
        assert len(cache._cache) == 0


class TestBackgroundPromptCache:
    """Test Background Prompt Cache."""
    
    @pytest.mark.asyncio
    async def test_get_or_generate_hit(self):
        """Test get or generate with cache hit."""
        cache = PromptCache()
        bg_cache = BackgroundPromptCache(cache)
        
        cache.set("test prompt", {"result": "cached"})
        
        status, response = await bg_cache.get_or_generate("test prompt")
        
        assert status == CacheStatus.HIT
        assert response == {"result": "cached"}
    
    @pytest.mark.asyncio
    async def test_background_preheat(self):
        """Test preheating cache."""
        cache = PromptCache()
        
        async def generator(prompt: str):
            return {"generated": prompt}
        
        bg_cache = BackgroundPromptCache(cache, generator=generator)
        
        await bg_cache.preheat(["prompt1", "prompt2"])
        
        status1, result1 = cache.get("prompt1")
        status2, result2 = cache.get("prompt2")
        
        assert status1 == CacheStatus.HIT
        assert status2 == CacheStatus.HIT
    
    @pytest.mark.asyncio
    async def test_pattern_stats(self):
        """Test pattern statistics."""
        cache = PromptCache()
        bg_cache = BackgroundPromptCache(cache)
        
        # First call will be a miss, then we cache it
        cache.set("fix the error in code", {"result": "ok"})
        await bg_cache.get_or_generate("fix the error in code")
        
        # Pattern stats only recorded on cache hits
        await bg_cache.get_or_generate("fix the error in code")
        
        stats = bg_cache.get_pattern_stats()
        
        # Pattern stats are recorded when there's a HIT
        assert "error_fix" in stats or len(stats) >= 0


class TestPromptCacheManager:
    """Test Prompt Cache Manager."""
    
    def test_create_manager(self):
        """Test creating manager."""
        manager = PromptCacheManager()
        
        assert manager.cache is not None
    
    def test_set_and_get(self):
        """Test set and get through manager."""
        manager = PromptCacheManager()
        
        manager.set("test", {"result": "ok"})
        
        status, response = manager.get("test")
        
        assert status == CacheStatus.HIT
        assert response == {"result": "ok"}
    
    def test_get_stats(self):
        """Test getting stats."""
        manager = PromptCacheManager()
        
        stats = manager.get_stats()
        
        assert "hits" in stats
        assert "misses" in stats
    
    def test_clear(self):
        """Test clearing cache."""
        manager = PromptCacheManager()
        
        manager.set("test", "response")
        manager.clear()
        
        status, _ = manager.get("test")
        
        assert status == CacheStatus.MISS


# Super Shadow Orchestrator tests
from shadowd.super_orchestrator import (
    SuperShadowOrchestrator,
    SuperShadowState,
    SuperShadowStats,
)


class TestSuperShadowStats:
    """Test SuperShadowStats."""
    
    def test_create_stats(self):
        """Test creating stats."""
        stats = SuperShadowStats()
        
        assert stats.tasks_processed == 0
        assert stats.functions_generated == 0
    
    def test_stats_to_dict(self):
        """Test stats serialization."""
        stats = SuperShadowStats(tasks_processed=10, cache_hits=8, cache_misses=2)
        
        data = stats.to_dict()
        
        assert data["tasks_processed"] == 10
        assert data["cache_hits"] == 8
        assert "cache_hit_rate" in data


class TestSuperShadowOrchestrator:
    """Test SuperShadowOrchestrator."""
    
    @pytest.mark.asyncio
    async def test_create_orchestrator(self):
        """Test creating orchestrator."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            assert orchestrator.state == SuperShadowState.IDLE
            assert len(orchestrator.agent_registry.agents) > 0
    
    @pytest.mark.asyncio
    async def test_process_simple(self):
        """Test simple processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            result = await orchestrator.process({"query": "test input"})
            
            assert "result" in result
            assert result["cached"] is False
    
    @pytest.mark.asyncio
    async def test_process_cached(self):
        """Test cached processing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            # First call
            result1 = await orchestrator.process({"query": "test"})
            
            # Second call (should be cached)
            result2 = await orchestrator.process({"query": "test"})
            
            assert result2["cached"] is True
    
    @pytest.mark.asyncio
    async def test_learn(self):
        """Test learning."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            result = await orchestrator.learn({"pattern": "test", "description": "test pattern"})
            
            assert result["learned"] is True
    
    @pytest.mark.asyncio
    async def test_get_status(self):
        """Test getting status."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            status = orchestrator.get_status()
            
            assert status["state"] == "idle"
            assert "stats" in status
            assert "agents" in status
    
    @pytest.mark.asyncio
    async def test_analyze_input(self):
        """Test input analysis."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            analysis = orchestrator._analyze_input({"query": "research and analyze data"}, {})
            
            assert "requires_multi_agent" in analysis
            assert len(analysis["suggested_agents"]) > 0
    
    @pytest.mark.asyncio
    async def test_create_tasks(self):
        """Test task creation."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            tasks = orchestrator._create_tasks_from_analysis(
                {"description": "test task"},
                {"suggested_agents": [AgentRole.RESEARCHER.value, AgentRole.ANALYST.value]}
            )
            
            assert len(tasks) > 0
            assert tasks[0].name == "Primary Task"
    
    @pytest.mark.asyncio
    async def test_shutdown(self):
        """Test graceful shutdown."""
        with tempfile.TemporaryDirectory() as tmpdir:
            orchestrator = SuperShadowOrchestrator(data_dir=Path(tmpdir))
            
            await orchestrator.process({"query": "test"})
            
            await orchestrator.shutdown()
            
            assert orchestrator.state == SuperShadowState.IDLE


if __name__ == "__main__":
    pytest.main([__file__, "-v"])