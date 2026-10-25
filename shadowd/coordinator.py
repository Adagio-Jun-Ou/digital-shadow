#!/usr/bin/env python3
"""
Multi-Agent Coordinator - Coordinates all 10 Shikigami abilities

Based on research from:
- Laddr: Coordinator-specialist pattern, Redis Streams, observability
- ContextAgent: Context-central design, template+state management
- Legion: Decorator API, graph workflows, chains and teams

Implements:
- Unified context across all abilities
- Coordinator pattern for task routing
- Ability delegation with dependencies
- Result synthesis
"""

import asyncio
import copy
import re
import threading
from datetime import datetime
from typing import Dict, List, Optional, Any, Callable
from pathlib import Path
from dataclasses import dataclass, field
from enum import Enum
import logging

from shikigami import ShikigamiType

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_QUERY_SIZE = 10000


def _sanitize(text: str, max_len: int = 500) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_len]


def _sanitize_content(text: str, max_len: int = 10000) -> str:
    if not isinstance(text, str):
        return str(text) if text is not None else ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_len]


logger = logging.getLogger("shadowd.coordinator")

_MAX_STATE_KEYS = 200
_MAX_COMPLETED = 1000
_MAX_PENDING = 50
_MAX_MEMORIES = 5000


# Use ShikigamiType from shikigami as the canonical enum
AbilityType = ShikigamiType

@dataclass
class AbilityProfile:
    """Profile for each ability - defines capabilities and dependencies"""
    name: str
    ability_type: AbilityType
    description: str
    tools: List[str] = field(default_factory=list)
    dependencies: List[AbilityType] = field(default_factory=list)
    priority: int = 5  # 1-10, higher = more important
    tags: List[str] = field(default_factory=list)
    
    # Context requirements
    requires_context: List[str] = field(default_factory=list)
    provides_context: List[str] = field(default_factory=list)
    
    # Execution config
    max_iterations: int = 15
    timeout: int = 300

@dataclass  
class UnifiedContext:
    """Shared context across all abilities (from ContextAgent pattern)"""
    # State (dynamic, changes during execution)
    state: Dict[str, Any] = field(default_factory=dict)
    
    # Profiles (static, loaded at init)
    profiles: Dict[str, AbilityProfile] = field(default_factory=dict)
    
    # Memory (from RAG)
    recent_memories: List[Dict] = field(default_factory=list)
    relevant_memories: List[Dict] = field(default_factory=list)
    
    # Domain knowledge (from 7 life domains)
    domain_context: Dict[str, Any] = field(default_factory=dict)
    
    # Current query
    query: str = ""
    domain: Optional[str] = None
    
    # Execution tracking
    completed_abilities: List[AbilityType] = field(default_factory=list)
    pending_results: Dict[AbilityType, Any] = field(default_factory=dict)
    
    def get_context_for_ability(self, ability_type: AbilityType) -> Dict:
        """Get relevant context for a specific ability"""
        profile = self.profiles.get(ability_type.value)
        if not profile:
            return {}
        
        context = {
            "query": self.query,
            "domain": self.domain,
            "recent_memories": list(self.recent_memories[-5:]),
            "state": {k: v for k, v in list(self.state.items())[:_MAX_STATE_KEYS]
                      if k in profile.requires_context}
        }
        
        if self.domain:
            context["domain_data"] = copy.deepcopy(self.domain_context.get(self.domain, {}))
        
        for dep in profile.dependencies:
            if dep in self.pending_results:
                context[f"{dep.value}_result"] = copy.deepcopy(self.pending_results[dep])
        
        return context
    
    def update_from_result(self, ability_type: AbilityType, result: Dict):
        """Update context after ability execution"""
        profile = self.profiles.get(ability_type.value)
        if not profile:
            return
        
        for key in profile.provides_context:
            if key in result:
                self.state[key] = result[key]
        
        if len(self.state) > _MAX_STATE_KEYS:
            oldest = list(self.state.keys())[:len(self.state) - _MAX_STATE_KEYS]
            for k in oldest:
                del self.state[k]
        
        self.completed_abilities.append(ability_type)
        if len(self.completed_abilities) > _MAX_COMPLETED:
            self.completed_abilities = self.completed_abilities[-_MAX_COMPLETED:]
        self.pending_results[ability_type] = result
        if len(self.pending_results) > _MAX_PENDING:
            oldest_key = next(iter(self.pending_results))
            del self.pending_results[oldest_key]

class AbilityCoordinator:
    """
    Coordinates all 10 abilities using patterns from:
    - Laddr: Coordinator-specialist delegation
    - ContextAgent: Unified context management
    - Legion: Graph-based execution
    """
    
    _MAX_COMPLETED = 1000
    _MAX_PENDING = 100
    _MAX_MEMORIES = 5000
    _MAX_STATE_KEYS = 200
    
    def __init__(self, shadow_ai):
        self.shadow_ai = shadow_ai
        self._lock = threading.Lock()
        self.context = UnifiedContext()
        self._init_ability_profiles()
        
    def _init_ability_profiles(self):
        """Initialize profiles for all 10 abilities"""
        profiles = {
            # Oracle - Precognition
            AbilityType.ORACLE: AbilityProfile(
                name="Oracle",
                ability_type=AbilityType.ORACLE,
                description="Predicts specific events with time and confidence",
                tools=["_oracle"],
                priority=7,
                tags=["prediction", "forecast", "probability"],
                requires_context=["observations", "patterns"],
                provides_context=["prediction", "confidence"]
            ),
            
            # X-Ray - See through walls
            AbilityType.X_RAY: AbilityProfile(
                name="X-Ray",
                ability_type=AbilityType.X_RAY,
                description="Identifies who is present using device detection",
                tools=["_scan_bt", "_scan_wifi"],
                priority=8,
                tags=["presence", "detection", "devices"],
                requires_context=[],
                provides_context=["present_people", "location", "devices"]
            ),
            
            # Life Pulse - Person identification
            AbilityType.LIFE_PULSE: AbilityProfile(
                name="Life Pulse",
                ability_type=AbilityType.LIFE_PULSE,
                description="Identifies specific people by device constellation",
                tools=["_scan_bt"],
                dependencies=[AbilityType.X_RAY],  # Needs device scan
                priority=8,
                tags=["identification", "tracking", "people"],
                requires_context=["devices"],
                provides_context=["identified_people"]
            ),
            
            # Perfect Memory - Total recall
            AbilityType.PERFECT_MEMORY: AbilityProfile(
                name="Perfect Memory",
                ability_type=AbilityType.PERFECT_MEMORY,
                description="Finds EXACT past events with timestamps",
                tools=["_search_memory"],
                priority=9,
                tags=["memory", "recall", "history"],
                requires_context=["memory_db"],
                provides_context=["memories_found", "commitments"]
            ),
            
            # Pattern Sight - Causation detection
            AbilityType.PATTERN_SIGHT: AbilityProfile(
                name="Pattern Sight",
                ability_type=AbilityType.PATTERN_SIGHT,
                description="Finds what causes patterns, not just correlation",
                tools=["_analyze_patterns"],
                priority=6,
                tags=["patterns", "causation", "analysis"],
                requires_context=["observations"],
                provides_context=["patterns", "causations"]
            ),
            
            # Social Radar - People prediction
            AbilityType.SOCIAL_RADAR: AbilityProfile(
                name="Social Radar",
                ability_type=AbilityType.SOCIAL_RADAR,
                description="Predicts specific human behaviors",
                tools=["_predict_behavior"],
                dependencies=[AbilityType.LIFE_PULSE],  # Needs person data
                priority=6,
                tags=["social", "prediction", "behavior"],
                requires_context=["tracked_people"],
                provides_context=["predictions"]
            ),
            
            # Intent Seer - Action prediction  
            AbilityType.INTENT_SEER: AbilityProfile(
                name="Intent Seer",
                ability_type=AbilityType.INTENT_SEER,
                description="Predicts next actions from sequences",
                tools=["_predict_sequence"],
                priority=7,
                tags=["intent", "prediction", "actions"],
                requires_context=["recent_actions"],
                provides_context=["next_action", "confidence"]
            ),
            
            # Omni Sense - Complete awareness
            AbilityType.OMNI_SENSE: AbilityProfile(
                name="Omni Sense",
                ability_type=AbilityType.OMNI_SENSE,
                description="Synthesizes all data into unified picture",
                dependencies=[
                    AbilityType.X_RAY,
                    AbilityType.LIFE_PULSE,
                    AbilityType.PATTERN_SIGHT
                ],  # Needs environmental data
                priority=10,
                tags=["environment", "synthesis", "awareness"],
                requires_context=["devices", "people", "location"],
                provides_context=["unified_snapshot"]
            ),
            
            # Shadow Council - Past wisdom
            AbilityType.SHADOW_COUNCIL: AbilityProfile(
                name="Shadow Council",
                ability_type=AbilityType.SHADOW_COUNCIL,
                description="Learns from past decisions",
                tools=["_query_decisions"],
                priority=7,
                tags=["decisions", "wisdom", "learning"],
                requires_context=["decision_db"],
                provides_context=["recommendation"]
            ),
            
            # Reality Forge - Manifestation
            AbilityType.REALITY_FORGE: AbilityProfile(
                name="Reality Forge",
                ability_type=AbilityType.REALITY_FORGE,
                description="Creates and executes code/commands",
                tools=["_generate_files", "_execute_commands"],
                priority=5,
                tags=["creation", "execution", "forge"],
                requires_context=[],
                provides_context=["files_created", "commands_run"]
            )
        }
        
        for ability_type, profile in profiles.items():
            self.context.profiles[ability_type.value] = profile
    
    async def analyze_query(self, query: str) -> List[AbilityType]:
        """
        Analyze query to determine which abilities are needed.
        Uses pattern matching and context requirements.
        """
        query_lower = _sanitize_content(query, _MAX_QUERY_SIZE).lower()
        needed_abilities = []
        
        # Pattern-based ability selection (word-boundary matching)
        patterns = {
            AbilityType.ORACLE: ["will", "predict", "happen", "when", "probability"],
            AbilityType.X_RAY: ["who", "is anyone", "present", "nearby", "where"],
            AbilityType.LIFE_PULSE: ["is sarah", "is john", "where is", "track"],
            AbilityType.PERFECT_MEMORY: ["remember", "when did", "what did", "recall"],
            AbilityType.PATTERN_SIGHT: ["why", "pattern", "cause", "trend"],
            AbilityType.SOCIAL_RADAR: ["will they", "behavior", "what will person"],
            AbilityType.INTENT_SEER: ["next", "going to", "about to", "planning"],
            AbilityType.OMNI_SENSE: ["where am i", "what's around", "environment", "status"],
            AbilityType.SHADOW_COUNCIL: ["should i", "decide", "choice", "past"],
            AbilityType.REALITY_FORGE: ["create", "build", "make", "set up"]
        }
        
        for ability, keywords in patterns.items():
            if any(re.search(r'\b' + re.escape(kw) + r'\b', query_lower) for kw in keywords):
                # Check if dependencies can be satisfied
                profile = self.context.profiles[ability.value]
                deps_satisfied = all(
                    dep in needed_abilities or dep in self.context.completed_abilities
                    for dep in profile.dependencies
                )
                if deps_satisfied:
                    needed_abilities.append(ability)
        
        # Always include Omni Sense if multiple abilities needed
        if len(needed_abilities) > 2 and AbilityType.OMNI_SENSE not in needed_abilities:
            needed_abilities.append(AbilityType.OMNI_SENSE)
        
        # Default to Omni Sense if nothing matched
        if not needed_abilities:
            needed_abilities.append(AbilityType.OMNI_SENSE)
        
        return needed_abilities
    
    async def execute_ability(
        self, 
        ability_type: AbilityType, 
        query: str,
        extra_context: Dict = None
    ) -> Dict:
        """Execute a single ability with context management"""
        
        ability_context = self.context.get_context_for_ability(ability_type)
        
        if extra_context:
            ability_context.update(extra_context)
        
        result = await self.shadow_ai.registry.execute(
            ability_type,
            query,
            ability_context
        )
        
        result_data = {
            "success": result.success,
            "output": result.output,
            "data": result.data
        }
        
        self.context.update_from_result(ability_type, result_data)
        
        return result_data
    
    async def coordinate(self, query: str, domain: str = None) -> Dict:
        """
        Main coordination method (from Laddr pattern).
        
        Steps:
        1. Analyze query to determine needed abilities
        2. Build dependency graph
        3. Execute in topological order
        4. Synthesize results
        """
        
        query = _sanitize_content(query, _MAX_QUERY_SIZE)
        if domain:
            domain = _sanitize(domain, 100)
        
        with self._lock:
            self.context.query = query
            self.context.domain = domain
            self.context.completed_abilities = []
            self.context.pending_results = {}
        
        needed_abilities = await self.analyze_query(query)
        
        try:
            execution_order = self._build_execution_order(needed_abilities)
        except ValueError:
            logger.warning("Circular dependency detected in abilities, falling back to flat execution order")
            execution_order = list(needed_abilities)
        
        results = {}
        for ability_type in execution_order:
            try:
                result = await self.execute_ability(ability_type, query)
                results[ability_type.value] = result
                logger.info("Executed %s: success=%s", ability_type.value, result['success'])
            except (ValueError, TypeError, RuntimeError, AttributeError, KeyError, OSError, ConnectionError, TimeoutError) as e:
                logger.error("Failed %s: %s", ability_type.value, type(e).__name__)
                results[ability_type.value] = {"success": False, "error": type(e).__name__}
        
        synthesis = await self._synthesize_results(query, results)
        
        return {
            "query": query,
            "domain": domain,
            "abilities_used": [a.value for a in execution_order],
            "results": copy.deepcopy(results),
            "synthesis": synthesis,
            "context_state": copy.deepcopy(dict(self.context.state))
        }
    
    def _build_execution_order(self, abilities: List[AbilityType]) -> List[AbilityType]:
        """Build topological order respecting dependencies (from Legion pattern)"""
        
        # Track visited and in-progress
        visited = set()
        in_progress = set()
        order = []
        
        def visit(ability: AbilityType):
            if ability in visited:
                return
            if ability in in_progress:
                raise ValueError(f"Circular dependency detected at {ability}")
            
            in_progress.add(ability)
            
            # Visit dependencies first
            profile = self.context.profiles.get(ability.value)
            if profile:
                for dep in profile.dependencies:
                    if dep in abilities:
                        visit(dep)
            
            in_progress.remove(ability)
            visited.add(ability)
            order.append(ability)
        
        # Visit all abilities
        for ability in abilities:
            if ability not in visited:
                visit(ability)
        
        return order
    
    async def _synthesize_results(self, query: str, results: Dict) -> str:
        """Synthesize results from multiple abilities using LLM when available."""
        synthesis_parts = []

        for ability_name, result in results.items():
            if result.get("success"):
                output = result.get("output", "")
                synthesis_parts.append(f"**{ability_name.title()}**:\n{output[:500]}\n")

        if not synthesis_parts:
            return "No abilities could answer your query."

        if len(synthesis_parts) == 1:
            return synthesis_parts[0]

        combined = "\n".join(synthesis_parts)
        mind = getattr(self.shadow_ai, 'mind', None)
        if mind and hasattr(mind, 'ask'):
            try:
                prompt = (
                    f"Synthesize the following ability results into a unified answer for: \"{query}\"\n\n"
                    f"{combined}\n\n"
                    f"Provide a clear, concise synthesis that combines all insights."
                )
                response = await mind.ask(prompt)
                if response:
                    return response
            except (ValueError, RuntimeError, TypeError, AttributeError, ConnectionError, TimeoutError, OSError) as e:
                logger.warning("LLM synthesis failed, falling back to concatenation: %s", type(e).__name__)

        return "\n".join(synthesis_parts)
    
    def get_ability_status(self) -> Dict:
        """Get status of all abilities"""
        with self._lock:
            return {
                ability_name: {
                    "profile": {
                        "name": profile.name,
                        "priority": profile.priority,
                        "dependencies": [d.value for d in profile.dependencies],
                        "tags": list(profile.tags),
                    }
                }
                for ability_name, profile in self.context.profiles.items()
            }

# Decorator for defining abilities (from Legion pattern)
def ability(
    name: str,
    ability_type: AbilityType,
    dependencies: List[AbilityType] = None,
    tools: List[str] = None,
    priority: int = 5
):
    """Decorator to define an ability profile"""
    def decorator(cls):
        profile = AbilityProfile(
            name=name,
            ability_type=ability_type,
            dependencies=dependencies or [],
            tools=tools or [],
            priority=priority
        )
        cls._profile = profile
        return cls
    return decorator