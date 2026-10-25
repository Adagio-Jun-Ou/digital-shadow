#!/usr/bin/env python3
"""
Comprehensive Test Suite for Shadow AI

Tests all components:
1. Multi-Agent Coordinator
2. Advanced Memory (Letta pattern)
3. Context Protocol (MCP)
4. Ambient Intelligence (Calm Tech)
5. Decision Learning
6. Performance Optimization
7. Integration
"""

import asyncio
import sys
from pathlib import Path
from datetime import datetime
import tempfile
import json

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

print("=" * 70)
print("SHADOW AI - COMPREHENSIVE TEST SUITE")
print("=" * 70)


def section_test(name: str):
    """Decorator for test sections."""
    def decorator(func):
        def wrapper():
            print(f"\n{'=' * 70}")
            print(f"Testing: {name}")
            print("=" * 70)
            try:
                result = func()
                print(f"✓ {name} PASSED")
                return True, result
            except Exception as e:
                print(f"✗ {name} FAILED: {e}")
                import traceback
                traceback.print_exc()
                return False, str(e)
        return wrapper
    return decorator


@section_test("Advanced Memory (Letta Pattern)")
def test_advanced_memory():
    from shadowd.advanced_memory import AdvancedMemorySystem, MemoryBlock, Entity
    
    # Create temp directory
    temp_dir = Path(tempfile.mkdtemp())
    memory = AdvancedMemorySystem(temp_dir)
    
    # Test core memory blocks
    memory.add_to_core("user", "Name: Test User. Location: Test City.")
    
    block = memory.core_memory.get("user")
    assert block is not None, "Core block should exist"
    assert "Test User" in block.value, "Core block should contain user info"
    print("✓ Core memory blocks work")
    
    # Test archival memory
    block = memory.add_to_archival(
        "This is a test observation",
        label="test",
        entities=["Test User"],
        source_type="observation"
    )
    assert block.uuid is not None, "Archival block should have UUID"
    print("✓ Archival memory works")
    
    # Test knowledge graph
    memory.add_entity_fact("Test User", "person", "lives in Test City", "test")
    
    entity = memory.knowledge_graph.get_entity("Test User")
    assert entity is not None, "Entity should exist"
    assert len(entity.facts) == 1, "Entity should have one fact"
    print("✓ Knowledge graph works")
    
    # Test entity episodes
    episodes = memory.knowledge_graph.get_entity_episodes("Test User")
    assert len(episodes) > 0, "Entity should have episodes"
    print("✓ Episode provenance works")
    
    # Test stats
    stats = memory.get_stats()
    assert stats["core_blocks"] == 5, "Should have 5 core blocks"
    assert stats["episodes"] > 0, "Should have episodes"
    print(f"✓ Memory stats: {stats}")
    
    return True


@section_test("MCP Protocol")
def test_mcp_protocol():
    from shadowd.mcp_protocol import (
        MCPServer, JSONRPCRequest, JSONRPCResponse,
        MCPTool, MCPResource, MCPPrompt
    )
    
    # Create server
    server = MCPServer(name="test-shadow", version="1.0.0")
    
    # Test tool registration
    async def dummy_tool(query: str):
        return f"Result for: {query}"
    
    tool = MCPTool(
        name="test_tool",
        description="A test tool",
        input_schema={"type": "object", "properties": {"query": {"type": "string"}}}
    )
    server.register_tool(tool)
    
    assert "test_tool" in server.tools, "Tool should be registered"
    print("✓ Tool registration works")
    
    # Test resource registration
    resource = MCPResource(
        uri="test://resource/1",
        name="Test Resource",
        description="A test resource"
    )
    server.register_resource(resource)
    
    assert "test://resource/1" in server.resources, "Resource should be registered"
    print("✓ Resource registration works")
    
    # Test prompt registration
    prompt = MCPPrompt(
        name="test_prompt",
        description="A test prompt",
        arguments=[{"name": "topic", "required": True}]
    )
    server.register_prompt(prompt)
    
    assert "test_prompt" in server.prompts, "Prompt should be registered"
    print("✓ Prompt registration works")
    
    # Test initialize handshake
    async def test_initialize():
        request = JSONRPCRequest(
            id=1,
            method="initialize",
            params={
                "protocolVersion": "2025-06-18",
                "capabilities": {},
                "clientInfo": {"name": "test-client", "version": "1.0"}
            }
        )
        
        response = await server.handle_request(request)
        
        assert response.result is not None, "Should have result"
        assert "serverInfo" in response.result, "Should have serverInfo"
        print("✓ Initialize handshake works")
        
        return response
    
    # Test tools/list
    async def test_tools_list():
        request = JSONRPCRequest(id=2, method="tools/list", params={})
        response = await server.handle_request(request)
        
        assert response.result is not None, "Should have result"
        assert "tools" in response.result, "Should have tools"
        print("✓ Tools list works")
        
        return response
    
    # Run async tests
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    
    try:
        loop.run_until_complete(test_initialize())
        loop.run_until_complete(test_tools_list())
    finally:
        loop.close()
    
    return True


@section_test("Calm Ambient Intelligence")
def test_calm_ambient():
    from shadowd.calm_ambient import CalmAmbient, AttentionManager, AttentionEvent
    
    # Test attention manager
    attention = AttentionManager(max_focus_items=3)
    
    # Test DND mode
    attention.set_dnd(hours=1)
    assert attention.dnd_mode, "DND should be set"
    assert not attention.can_interrupt("focus"), "Should not interrupt during DND"
    print("✓ DND mode works")
    
    # Test attention budget
    attention.dnd_mode = False
    attention.sacred_times = {}  # Disable sacred times for testing
    assert attention.can_interrupt("focus"), "Should be able to interrupt"
    
    # Add events
    event1 = AttentionEvent(
        event_type="test",
        message="Test periphery",
        priority="periphery"
    )
    assert attention.add_event(event1), "Should add periphery event"
    
    event2 = AttentionEvent(
        event_type="test",
        message="Test focus",
        priority="focus"
    )
    assert attention.add_event(event2), "Should add focus event"
    print("✓ Attention events work")
    
    # Test glanceable summary
    summary = attention.get_glanceable_summary()
    assert "periphery" in summary, "Should have periphery count"
    assert "focus" in summary, "Should have focus count"
    print(f"✓ Glanceable summary: {summary}")
    
    # Test calm ambient
    ambient = CalmAmbient()
    
    # Test passive observation
    observation = {"type": "test", "content": "Test observation"}
    assert ambient.observe(observation), "Should observe passively"
    print("✓ Passive observation works")
    
    # Test pattern detection
    for i in range(5):
        ambient.observe({"type": "recurring", "count": i})
    
    patterns = ambient.detected_patterns
    assert "type:recurring" in patterns, "Should detect recurring pattern"
    print("✓ Pattern detection works")
    
    # Test glanceable context
    context = ambient.get_glanceable_context()
    assert "attention" in context, "Should have attention"
    assert "context" in context, "Should have context"
    print(f"✓ Glanceable context: {context}")
    
    return True


@section_test("Decision Learning V2")
def test_decision_learning_v2():
    from shadowd.decision_learning_v2 import (
        DecisionLearningV2, DecisionStatus, MCDAMethods
    )
    
    temp_dir = Path(tempfile.mkdtemp())
    learning = DecisionLearningV2(temp_dir)
    
    # Test decision creation
    decision, recommendation = learning.create_decision(
        situation="Should I take the new job?",
        options=["Yes, accept", "No, decline", "Negotiate"],
        context="New position at different company",
        tags=["career", "job"]
    )
    
    assert decision.id is not None, "Decision should have ID"
    assert decision.status == DecisionStatus.PENDING, "Should be pending"
    assert decision.category == "career", "Should detect career category"
    print("✓ Decision creation works")
    
    # Test decision making
    decided = learning.decide(
        decision.id,
        chosen="Negotiate",
        reasoning="Better salary would improve finances",
        confidence=0.8
    )
    
    assert decided.status == DecisionStatus.IN_PROGRESS, "Should be in progress"
    assert decided.decided_at is not None, "Should have decision time"
    print("✓ Decision making works")
    
    # Test outcome recording
    resolved = learning.record_outcome(
        decision.id,
        outcome="Got 20% raise, accepted offer",
        satisfaction=9
    )
    
    assert resolved.status == DecisionStatus.RESOLVED, "Should be resolved"
    assert len(resolved.lessons_learned) > 0, "Should have lessons"
    print("✓ Outcome recording works")
    
    # Test pattern learning
    # Create more decisions to learn patterns (need 10 for confidence >= 0.5)
    for i in range(10):
        d, _ = learning.create_decision(
            situation=f"Should I buy house {i}?",
            options=["Buy", "Don't buy", "Wait"],
            tags=["finance", "house"]
        )
        learning.decide(d.id, "Buy", "Good investment", 0.7)
        learning.record_outcome(d.id, "Purchased", 7)
    
    patterns = learning.get_patterns(min_confidence=0.5)
    assert len(patterns) > 0, "Should have learned patterns"
    print(f"✓ Pattern learning works: {len(patterns)} patterns")
    
    # Test MCDA
    mcda = MCDAMethods()
    
    # Test AHP
    ranked = mcda.ahp(
        criteria=["salary", "location", "team"],
        alternatives=["Job A", "Job B"],
        criteria_weights={"salary": 0.5, "location": 0.3, "team": 0.2},
        alternative_scores={
            "Job A": {"salary": 8, "location": 6, "team": 9},
            "Job B": {"salary": 7, "location": 9, "team": 7}
        }
    )
    assert len(ranked) == 2, "Should rank alternatives"
    print(f"✓ AHP ranking: {ranked}")
    
    # Test journal generation
    entry = learning.generate_journal_entry(decision.id)
    assert "Should I take the new job" in entry, "Should include situation"
    assert "Negotiate" in entry, "Should include choice"
    print("✓ Journal generation works")
    
    # Test stats
    stats = learning.get_stats()
    assert stats["total_decisions"] >= 4, "Should have decisions"
    print(f"✓ Decision stats: {stats}")
    
    return True


@section_test("Performance Optimization")
def test_performance_optimization():
    from shadowd.performance_optimization import (
        MultiLevelCache, AsyncTaskPool, ObjectPool,
        PerformanceProfiler
    )
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Test multi-level cache
    cache = MultiLevelCache(l1_size=10, l2_dir=temp_dir / "cache", ttl=60)
    
    # Test cache miss
    result = cache.get("test_key")
    assert result is None, "Should miss on first get"
    assert cache.hits["miss"] == 1, "Should count miss"
    print("✓ Cache miss works")
    
    # Test cache set and get
    cache.set("test_key", {"data": "test_value"})
    
    # Should hit L1
    result = cache.get("test_key")
    assert result == {"data": "test_value"}, "Should get cached value"
    assert cache.hits["l1"] == 1, "Should count L1 hit"
    print("✓ L1 cache works")
    
    # Test L2 cache (clear L1 first)
    cache.l1_cache.clear()
    cache.l1_metadata.clear()
    
    # Should hit L2
    result = cache.get("test_key")
    assert result == {"data": "test_value"}, "Should get from L2"
    assert cache.hits["l2"] == 1, "Should count L2 hit"
    print("✓ L2 cache works")
    
    # Test stats
    stats = cache.get_stats()
    assert "hit_rate" in stats, "Should have hit rate"
    print(f"✓ Cache stats: {stats}")
    
    # Test async task pool
    async def test_task_pool():
        pool = AsyncTaskPool(max_concurrent=3, rate_limit=10)
        
        async def dummy_task(x):
            await asyncio.sleep(0.01)
            return x * 2
        
        # Submit batch
        results = await asyncio.gather(*[
            pool.submit(dummy_task, i) for i in range(5)
        ])
        
        assert len(results) == 5, "Should have 5 results"
        assert all(r == i * 2 for i, r in enumerate(results)), "Results should be doubled"
        print("✓ Async task pool works")
        
        return results
    
    loop = asyncio.new_event_loop()
    loop.run_until_complete(test_task_pool())
    loop.close()
    
    # Test object pool
    class TestObject:
        def __init__(self):
            self.value = 0
    
    pool = ObjectPool(factory=TestObject, max_size=5)
    
    obj1 = pool.acquire()
    assert obj1 is not None, "Should acquire object"
    obj1.value = 42
    print("✓ Object pool acquire works")
    
    pool.release(obj1)
    obj2 = pool.acquire()
    assert obj2.value == 42, "Should reuse object"
    print("✓ Object pool reuse works")
    
    # Test performance profiler
    profiler = PerformanceProfiler()
    
    @profiler.time_it("test_operation")
    def test_op():
        time.sleep(0.01)
        return "result"
    
    # Run a few times
    for _ in range(3):
        test_op()
    
    stats = profiler.get_stats()
    assert "test_operation" in stats["timings"], "Should have timing"
    assert stats["timings"]["test_operation"]["calls"] == 3, "Should have 3 calls"
    print(f"✓ Performance profiling: {stats['timings']}")
    
    return True


@section_test("Full Integration")
def test_full_integration():
    """Test all components working together."""
    from shadowd.unified_system import create_unified_system
    from shadowd.performance_optimization import PerformanceOptimizedShadowAI
    
    temp_dir = Path(tempfile.mkdtemp())
    
    # Create unified system
    system = create_unified_system(data_dir=temp_dir)
    
    # Test initialization
    status = system.get_unified_status()
    assert "components" in status, "Should have components"
    print(f"✓ Unified system initialized")
    print(f"  Components: {list(status['components'].keys())}")
    
    # Test observation
    import asyncio
    loop = asyncio.new_event_loop()
    
    async def test_observe():
        await system.observe({
            "type": "test",
            "content": "Test observation",
            "domain": "health"
        })
        print("✓ System observation works")
    
    loop.run_until_complete(test_observe())
    
    # Test integration test
    results = system.test_integration()
    print(f"\nIntegration Tests:")
    print(f"  Passed: {results['passed']}")
    print(f"  Failed: {results['failed']}")
    print(f"  Success Rate: {results['success_rate']}")
    
    for test in results['tests']:
        status = "✓" if test['passed'] else "✗"
        print(f"  {status} {test['name']}: {test.get('message', '')}")
    
    loop.close()
    
    return results['failed'] == 0


def main():
    """Run all tests."""
    print("\nRunning comprehensive test suite...\n")
    
    results = {}
    
    # Run all tests
    test_functions = [
        ("Advanced Memory", test_advanced_memory),
        ("MCP Protocol", test_mcp_protocol),
        ("Calm Ambient", test_calm_ambient),
        ("Decision Learning V2", test_decision_learning_v2),
        ("Performance Optimization", test_performance_optimization),
        ("Full Integration", test_full_integration)
    ]
    
    for name, test_func in test_functions:
        passed, result = test_func()
        results[name] = {
            "passed": passed,
            "result": result
        }
    
    # Summary
    print("\n" + "=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)
    
    passed_count = sum(1 for r in results.values() if r["passed"])
    total_count = len(results)
    
    for name, result in results.items():
        status = "✓ PASS" if result["passed"] else "✗ FAIL"
        print(f"{status}: {name}")
    
    print(f"\nTotal: {passed_count}/{total_count} tests passed")
    print(f"Success Rate: {passed_count/total_count:.1%}")
    
    if passed_count == total_count:
        print("\n✓ ALL TESTS PASSED!")
        return 0
    else:
        print(f"\n✗ {total_count - passed_count} tests failed")
        return 1


if __name__ == "__main__":
    import time
    sys.exit(main())