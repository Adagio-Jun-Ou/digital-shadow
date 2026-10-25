#!/usr/bin/env python3
"""
Unified Shadow System - Integration of all components

Brings together:
- Multi-Agent Coordinator
- RAG Memory
- Context Protocol
- Ambient Monitor
- Learning System
- Domain Expertise
- Memory Optimization
- Proactive Insights
"""

import asyncio
import copy
import re
import threading
from datetime import datetime, timezone
from typing import Dict, List, Optional, Any
from pathlib import Path
import logging

from .coordinator import AbilityCoordinator, AbilityType
from .memory_rag import RAGMemorySystem, RAGContext
from .context_protocol import ContextProtocol, ContextSnapshot
from .ambient_monitor import AmbientMonitor
from .learning_system import LearningSystem
from .domain_expertise import DomainExpertiseManager
from .memory_optimization import MemoryOptimizer
from .proactive_insights import ProactiveInsightEngine

logger = logging.getLogger("shadowd.unified_system")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_OBSERVATION_SIZE = 65536
_MAX_QUESTION_SIZE = 10000


class UnifiedShadowSystem:
    """
    Unified integration of all Shadow AI components.
    """

    _MAX_TASKS = 10

    def __init__(self, data_dir: Path = None, shikigami_registry=None, mind_engine=None):
        self.data_dir = data_dir or (Path.home() / "digital-shadow" / "data")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        
        # Core components
        self.shikigami = shikigami_registry
        self.mind = mind_engine
        
        # Initialize all systems
        logger.info("Initializing Unified Shadow System...")
        
        # 1. RAG Memory (Cycle 2)
        self.memory = RAGMemorySystem(self.data_dir / "memory")
        self.rag_context = RAGContext(self.memory)
        logger.info("✓ RAG Memory System initialized")
        
        # 2. Context Protocol (Cycle 3)
        self.context = ContextProtocol(self.data_dir / "context")
        logger.info("✓ Context Protocol initialized")
        
        # 3. Ambient Monitor (Cycle 4)
        self.ambient = AmbientMonitor(self.context, self.data_dir / "ambient")
        logger.info("✓ Ambient Monitor initialized")
        
        # 4. Learning System (Cycle 5)
        self.learning = LearningSystem(self.data_dir / "learning")
        logger.info("✓ Learning System initialized")
        
        # 5. Domain Expertise (Cycle 6)
        self.domains = DomainExpertiseManager()
        logger.info("✓ Domain Expertise initialized")
        
        # 6. Memory Optimization (Cycle 7)
        self.optimizer = MemoryOptimizer(self.data_dir / "optimized")
        logger.info("✓ Memory Optimization initialized")
        
        # 7. Proactive Insights (Cycle 8)
        self.proactive = ProactiveInsightEngine()
        logger.info("✓ Proactive Insights initialized")
        
        # 8. Coordinator (Cycle 1) - created last, depends on others
        if shikigami_registry:
            # Temporarily create a shadow_ai wrapper for coordinator
            self.shadow_ai = type('ShadowAI', (), {
                'registry': shikigami_registry
            })()
            self.coordinator = AbilityCoordinator(self.shadow_ai)
            logger.info("✓ Multi-Agent Coordinator initialized")
        else:
            self.coordinator = None
            logger.warning("⚠ Shikigami registry not provided, coordinator not initialized")
        
        # Background tasks
        self._tasks: List[asyncio.Task] = []
        self._lock = threading.Lock()
        
        logger.info("═══════════════════════════════════════════")
        logger.info("  Unified Shadow System Ready")
        logger.info("  10 Abilities • 7 Domains • AI Powered")
        logger.info("═══════════════════════════════════════════")
    
    async def start_background_tasks(self):
        """Start all background monitoring tasks."""
        new_tasks = []
        with self._lock:
            if self._tasks:
                logger.warning("Background tasks already running, skipping start")
                return
            if len(self._tasks) >= self._MAX_TASKS:
                logger.warning("Maximum background tasks reached, skipping start")
                return
            
            new_tasks.append(
                asyncio.create_task(self.context.start_monitoring(interval=30))
            )
            new_tasks.append(
                asyncio.create_task(self.ambient.run_continuous_monitoring(interval=60))
            )
            new_tasks.append(
                asyncio.create_task(self.proactive.run_proactive_monitor(interval=300))
            )
            self._tasks.extend(new_tasks)
        
        logger.info("Started %d background tasks", len(new_tasks))
    
    async def stop_background_tasks(self):
        """Stop all background tasks."""
        logger.info("Stopping background tasks...")
        with self._lock:
            tasks_to_cancel = list(self._tasks)
            self._tasks = []

        for task in tasks_to_cancel:
            task.cancel()

        if tasks_to_cancel:
            await asyncio.gather(*tasks_to_cancel, return_exceptions=True)
    
    async def observe(self, event: Dict):
        """
        Observe an event (main entry point for data).

        Routes to all relevant subsystems:
        - Memory (RAG)
        - Ambient Monitor
        - Domain Expertise
        - Learning System
        """
        event_content = event.get("content", "")
        if not isinstance(event_content, str):
            event_content = str(event_content)
        event_content = _SANITIZE_CONTENT_RE.sub('', event_content).strip()[:_MAX_OBSERVATION_SIZE]

        domain = self.domains.classify_domain(event_content)

        try:
            optimization = self.optimizer.optimize_retrieval(
                event_content,
                context=domain
            )

            await self.memory.add_observation(
                content=event_content,
                metadata=optimization,
                domain=domain
            )
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"Memory observation failed: {type(e).__name__}")

        try:
            await self.ambient.observe(event)
        except (ValueError, TypeError, RuntimeError, OSError) as e:
            logger.error(f"Ambient observe failed: {type(e).__name__}")

        try:
            self.domains.add_observation(event, domain)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Domain observation failed: {type(e).__name__}")
        
        # Generate insights
        now = datetime.now(timezone.utc)
        try:
            ambient_commitments_count = len(self.ambient.get_pending_commitments(hours=24)) if hasattr(self.ambient, 'get_pending_commitments') else 0
            pending_count = len(self.ambient.commitments) if hasattr(self.ambient, 'commitments') else 0
        except (ValueError, TypeError, AttributeError):
            ambient_commitments_count = 0
            pending_count = 0
        
        context_data = {
            "hour": now.hour,
            "day_of_week": now.strftime("%A"),
            "pending_commitments": pending_count,
            "urgent_deadlines": ambient_commitments_count,
        }
        
        try:
            self.proactive.evaluate_rules(context_data)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Proactive rules failed: {type(e).__name__}")
    
    async def ask(self, question: str, domain: str = None) -> Dict:
        """
        Main query interface.
        
        Uses:
        1. Coordinator to select abilities
        2. RAG for context
        3. Domain expertise for insights
        4. Memory optimization for performance
        """
        question = _SANITIZE_CONTENT_RE.sub('', str(question)).strip()[:_MAX_QUESTION_SIZE]
        # Optimize query
        optimization = self.optimizer.optimize_retrieval(question, domain)
        
        # Check cache
        if optimization.get("cached_result"):
            return copy.deepcopy(optimization["cached_result"])
        
        # Prepare RAG context
        rag_context = await self.rag_context.prepare_context(question, domain)
        
        # Get domain insights
        domain_analysis = self.domains.analyze(question, domain)
        
        # Get similar past decisions (for Shadow Council)
        similar_decisions = self.learning.query_similar_decisions(question)
        
        # Use coordinator if available
        if self.coordinator:
            result = await self.coordinator.coordinate(question, domain)
            
            # Enhance with domain insights
            result["domain_analysis"] = domain_analysis
            result["rag_context"] = {
                "recent_count": len(rag_context.get("recent", [])),
                "memories_found": len(rag_context.get("relevant", []))
            }
            
            # Cache result
            self.optimizer.store_result(question, result, domain)
            
            return result
        
        # Fallback: direct ability execution
        return copy.deepcopy({
            "question": question,
            "domain": domain,
            "domain_analysis": domain_analysis,
            "similar_decisions": len(similar_decisions),
            "context": rag_context,
            "optimization": optimization.get("cache_stats")
        })
    
    def track_decision(
        self,
        context: str,
        options: List[str],
        chosen: str,
        reasoning: str
    ) -> str:
        """Track a decision for learning."""
        context = _SANITIZE_CONTENT_RE.sub('', str(context)).strip()[:5000]
        reasoning = _SANITIZE_CONTENT_RE.sub('', str(reasoning)).strip()[:5000]
        options = [_SANITIZE_CONTENT_RE.sub('', str(o)).strip()[:500] for o in (options or [])[:20]]
        chosen = _SANITIZE_CONTENT_RE.sub('', str(chosen)).strip()[:500]
        decision = self.learning.track_decision(
            context=context,
            options=options,
            chosen=chosen,
            reasoning=reasoning
        )
        return decision.id
    
    def record_outcome(self, decision_id: str, outcome: str, satisfaction: int):
        """Record outcome of a decision."""
        decision_id = _SANITIZE_STRICT_RE.sub('', str(decision_id)).strip()[:200]
        outcome = _SANITIZE_CONTENT_RE.sub('', str(outcome)).strip()[:5000]
        satisfaction = max(1, min(10, int(satisfaction))) if satisfaction else 5
        self.learning.record_outcome(decision_id, outcome, satisfaction)
    
    def get_proactive_insights(self) -> List[Dict]:
        """Get current proactive insights."""
        insights = self.proactive.get_active_insights()
        
        try:
            commitments = self.ambient.get_pending_commitments(hours=24)
            commitment_insights = self.proactive.check_commitments(commitments)
            insights.extend(commitment_insights)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Commitment insights failed: {type(e).__name__}")
        
        try:
            patterns = list(self.ambient.patterns.values()) if hasattr(self.ambient, 'patterns') else []
            pattern_insights = self.proactive.check_patterns(patterns)
            insights.extend(pattern_insights)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.error(f"Pattern insights failed: {type(e).__name__}")
        
        # Sort by priority
        priority_order = {"critical": 0, "high": 1, "medium": 2, "low": 3}
        insights.sort(key=lambda i: priority_order.get(i.priority, 3))
        
        return [
            {
                "type": i.insight_type,
                "priority": i.priority,
                "title": i.title,
                "message": i.message,
                "action": i.action,
                "confidence": round(i.confidence, 3)
            }
            for i in insights[:10]  # Top 10
        ]
    
    def get_unified_status(self) -> Dict:
        """Get comprehensive system status."""
        with self._lock:
            task_count = len(self._tasks)
            try:
                memory_stats = copy.deepcopy(self.memory.get_stats())
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                memory_stats = {}
            try:
                ambient_stats = copy.deepcopy(self.ambient.get_status())
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                ambient_stats = {}
            try:
                learning_stats = copy.deepcopy(self.learning.get_status())
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                learning_stats = {}
            try:
                domain_stats = copy.deepcopy(self.domains.get_status())
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                domain_stats = {}
            try:
                optimization_stats = copy.deepcopy(self.optimizer.get_optimization_stats())
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                optimization_stats = {}
            try:
                proactive_stats = copy.deepcopy(self.proactive.get_summary())
            except (ValueError, TypeError, AttributeError, RuntimeError, OSError):
                proactive_stats = {}
            try:
                profiles_count = len(self.coordinator.context.profiles) if self.coordinator else 0
            except (ValueError, TypeError, AttributeError):
                profiles_count = 0
            try:
                context_snapshot = self.context.current_snapshot is not None
                context_history_size = len(self.context.context_history)
            except (ValueError, TypeError, AttributeError):
                context_snapshot = False
                context_history_size = 0
        
        return {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "components": {
                "memory": memory_stats,
                "context": {
                    "current_snapshot": context_snapshot,
                    "history_size": context_history_size,
                },
                "ambient": ambient_stats,
                "learning": learning_stats,
                "domains": domain_stats,
                "optimization": optimization_stats,
                "proactive": proactive_stats,
            },
            "coordinator": {
                "initialized": self.coordinator is not None,
                "ability_profiles": profiles_count,
            },
            "background_tasks": task_count,
        }
    
    def get_domain_summary(self, domain: str = None) -> Dict:
        """Get summary of domain expertise."""
        if domain:
            return copy.deepcopy(self.domains.get_domain_summary(domain))
        return copy.deepcopy(self.domains.get_cross_domain_insights())
    
    async def search_memory(self, query: str, n_results: int = 5) -> List[Dict]:
        """Search memory with optimization."""
        n_results = max(1, min(100, n_results))
        return copy.deepcopy(await self.memory.search(query, n_results))
    
    def test_integration(self) -> Dict:
        """
        Test all integrated components.
        
        Returns test results for each component.
        """
        results = {
            "tests": [],
            "passed": 0,
            "failed": 0
        }
        
        def test(name: str, condition: bool, message: str = ""):
            results["tests"].append({
                "name": name,
                "passed": condition,
                "message": message
            })
            if condition:
                results["passed"] += 1
            else:
                results["failed"] += 1
        
        # Test 1: Memory System
        test("Memory System", self.memory is not None)
        test("Memory Stats", "total_memories" in self.memory.get_stats())
        
        # Test 2: Context Protocol
        test("Context Protocol", self.context is not None)
        test("Context Types", len(self.context.sources) > 0)
        
        # Test 3: Ambient Monitor
        test("Ambient Monitor", self.ambient is not None)
        test("Ambient Commitments", hasattr(self.ambient, 'commitments'))
        
        # Test 4: Learning System
        test("Learning System", self.learning is not None)
        test("Learning Patterns", len(self.learning.patterns) >= 0)
        
        # Test 5: Domain Expertise
        test("Domain Expertise", self.domains is not None)
        test("Domain Count", len(self.domains.experts) == 7)
        
        # Test 6: Memory Optimization
        test("Memory Optimization", self.optimizer is not None)
        test("Cache Stats", "hit_rate" in self.optimizer.cache.get_stats())
        
        # Test 7: Proactive Insights
        test("Proactive Insights", self.proactive is not None)
        test("Proactive Rules", len(self.proactive.rules) > 0)
        
        # Test 8: Coordinator (if available)
        if self.coordinator:
            test("Coordinator", True)
            test("Ability Profiles", len(self.coordinator.context.profiles) == 10)
        else:
            test("Coordinator", False, "Not initialized (no shikigami)")
        
        # Overall
        results["success_rate"] = f"{results['passed']}/{results['passed'] + results['failed']}"
        
        return results


# Integration function
def create_unified_system(
    data_dir: Path = None,
    shikigami_registry = None,
    mind_engine = None
) -> UnifiedShadowSystem:
    """
    Factory function to create unified system.
    
    Args:
        data_dir: Data directory (default: ~/digital-shadow/data)
        shikigami_registry: Shikigami registry (required for coordinator)
        mind_engine: Mind engine for AI responses
    
    Returns:
        UnifiedShadowSystem instance
    """
    return UnifiedShadowSystem(
        data_dir=data_dir,
        shikigami_registry=shikigami_registry,
        mind_engine=mind_engine
    )