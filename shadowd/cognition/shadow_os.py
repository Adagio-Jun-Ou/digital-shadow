#!/usr/bin/env python3
"""
ShadowOS - Unified Operating System for Shadow AI

The integration layer that wires all cognitive modules together and exposes
them through a unified API. This is the "Control Plane" inspired by Agno.

Architecture:
  ShadowOS (Control Plane)
  ├── ShadowGraph (cognitive state machine)
  ├── DarkMemory (unified memory)
  ├── ShadowCrew (multi-agent team)
  ├── NeuralMesh (distributed communication)
  ├── StealthLayer (privacy guard)
  ├── PersonalityEngine (user personality)
  ├── ShadowVoice (voice interface)
  └── Legacy Bridges (shadow_ai, mind, shikigami, event_bus)

Lifecycle:
  1. Initialize all modules
  2. Wire module inter-dependencies
  3. Start background tasks (heartbeat, compaction, distillation)
  4. Expose unified API
  5. Handle graceful shutdown
"""

import asyncio
import copy
import json
import os
import re
import time
import logging
import threading
from collections import deque
from datetime import datetime, timedelta, timezone
from typing import Any, Callable, Dict, List, Optional
from dataclasses import dataclass, field
from pathlib import Path

from .shadow_graph import ShadowGraph, CognitiveMode, CognitiveState, TransitionTrigger
from .dark_memory import DarkMemory, MemoryAtom, AtomType, MemoryTier
from .shadow_crew import ShadowCrew, ShadowAgent, CrewMission, DelegationMode
from .neural_mesh import NeuralMesh, MeshNode, MeshMessage, MeshMessageType
from .stealth_layer import StealthLayer, DataCategory, AccessAction, CamouflageMode
from .personality_engine import PersonalityEngine, PersonalityProfile, BehaviorPattern
from .shadow_voice import ShadowVoice, VoiceSession, VoiceCommand, VoiceIntent
from .anomaly_detector import AnomalyDetectionEngine, Anomaly, AnomalyType, AnomalySeverity
from .goal_planner import CognitiveGoalPlanner, Goal, GoalStatus, GoalPriority
from .thought_sense import ThoughtSense
from .recall_synthesis import RecallSynthesis
from .temporal_reasoning import TemporalReasoning
from .empathy_mapping import EmpathyMapping
from .presence_detection import PresenceDetection
from .cognitive_trinity import CognitiveTrinity
from .dream_state import DreamState
from .adaptive_skill_tree import AdaptiveSkillTree
from .narrative_memory import NarrativeMemory
from .procedural_memory import ProceduralMemory
from .conflict_resolution import ConflictResolution
from .emotional_intelligence import EmotionalIntelligence, PatternType, EmotionCategory
from .creative_synthesis import CreativeSynthesis, SynthesisMethod
from .metacognition_engine import MetacognitionEngine, ReasoningType
from .moral_reasoning import MoralReasoning
from .attention_engine import AttentionEngine
from .cognitive_pipeline import CognitivePipeline
from .social_intelligence import SocialIntelligence
from .decision_journal import DecisionJournal
from .cognitive_reflection import CognitiveReflection
from .goal_reasoning import GoalReasoning
from .device_context import DeviceContextEngine, DeviceType, LocationType, ActivityType, StressLevel, FocusLevel
from .shadow_arise import ShadowArise, AgentLifecycle, CapabilityType, MissionPriority, AgentStatus
from .shadow_summon import ShadowSummon
from .evolution_guide import EvolutionGuide
from .evolutionary_engine import EvolutionaryEngine
from .oracle_bridge import OracleBridge
from .vision_perception import VisionPerceptionEngine
from .notification_bus import NotificationBus
from .visual_observation_bridge import VisualObservationBridge
from .shadow_deploy import ShadowDeployEngine
from .shadow_journal import ShadowJournal
from .dashboard_aggregator import DashboardAggregator
from .predictive_alerts import PredictiveAlertEngine
from .shadow_pool import ShadowPool
from .autonomous_daemon import AutonomousDaemon, UserActivityState, UserRhythmModel, RhythmSample, ObservationType, SecurityEventType, ThreatLevel
from .feed_loop import FeedLoop
from .anticipatory_engine import AnticipatoryEngine, AnticipationContext
from .cognitive_resonance import CognitiveResonanceDetector, ResonanceLink
from .temporal_self import TemporalSelf
from .health_dashboard import CognitiveHealthDashboard
from .creative_dream import CreativeDreamIntegration
from .social_graph_propagation import SocialGraphPropagation, PropagationSource, RippleType, RippleStrength
from .neural_symbolic_bridge import NeuralSymbolicBridge
from .quantum_decision_field import QuantumDecisionField
from .emergent_grammar import EmergentGrammar
from .holographic_memory import HolographicMemory
from .syncognitive_feedback import SyncognitiveFeedback
from .morphic_resonance_engine import MorphicResonanceEngine
from .existential_calculus import ExistentialCalculus
from .chrono_synthesis import ChronoSynthesis
from .phoenix_regenerator import PhoenixRegenerator
from .causal_reasoning import CausalReasoning
from .quantum_memory_field import QuantumMemoryField
from .semantic_drift_detector import SemanticDriftDetector
from .cognitive_immune_system import CognitiveImmuneSystem
from .temporal_parallax import TemporalParallax
from .adaptive_boundary import AdaptiveBoundary
from .cognitive_resonance_engine import CognitiveResonanceEngine
from .cognitive_economy import CognitiveEconomy
from .cross_module_synthesis import CrossModuleSynthesisEngine, ModuleSignal
from .self_modification import SelfModificationEngine
from .cognitive_visualizer import CognitiveVisualizer
from .performance_profiler import PerformanceProfiler
from .intelligence_report import IntelligenceReportEngine
from .anomaly_correlation import AnomalyCorrelationEngine
from .cross_module_bus import CrossModuleEventBus, ModuleCapability, TypedEvent
from .optimization_daemon import CognitiveOptimizationDaemon
from .module_registry import ModuleCapabilityRegistry
from .decision_arbiter import DecisionArbiter
from .shadow_powers import ShadowPowers
from .cognitive_topology import CognitiveTopology
from .cognitive_symbiosis import CognitiveSymbiosis

from .daily_briefing import DailyBriefingEngine
from .escalation import EscalationManager

from ..mind import MindEngine
from ..shadow_ai import ShadowAI
from ..event_bus import get_event_bus
from ..atomic_io import fsync_dir

logger = logging.getLogger("shadowd.cognition.shadow_os")

MAX_QUERY_LENGTH = 2000
MAX_OBSERVATION_KEYS = 50
MAX_OBSERVATION_VALUE_LENGTH = 10000

# FeedLoop dispatcher: max signal IDs remembered for exactly-once dispatch.
_FEED_DEDUP_MAX = 5000
# Minimum effective intensity (signal.intensity) before a target module is invoked.
_FEED_DISPATCH_MIN_INTENSITY = 0.1
# Higher bar for expensive "analysis" calls (trinity, morality, creative synthesis).
_FEED_TRINITY_MIN_INTENSITY = 0.7


def sanitize_input(text: str, max_length: int = MAX_QUERY_LENGTH) -> str:
    text = re.sub(r'<script[^>]*>.*?</script>', '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'<[^>]+>', '', text)
    text = re.sub(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]', '', text)
    text = re.sub(r'javascript:', '', text, flags=re.IGNORECASE)
    text = re.sub(r'on\w+\s*=', '', text, flags=re.IGNORECASE)
    text = text[:max_length]
    return text.strip()


def validate_observation(observation: Dict) -> Dict:
    cleaned = {}
    for k, v in list(observation.items())[:MAX_OBSERVATION_KEYS]:
        if isinstance(k, str) and len(k) < 100:
            if isinstance(v, str):
                cleaned[k] = v[:MAX_OBSERVATION_VALUE_LENGTH]
            elif isinstance(v, (int, float, bool)):
                cleaned[k] = v
            elif isinstance(v, dict):
                cleaned[k] = validate_observation(v)
            elif isinstance(v, list):
                cleaned[k] = v[:100]
    return cleaned


class ShadowOS:
    """
    Shadow Operating System - Unified Control Plane.

    Integrates all 7 cognitive modules and provides:
    - Single point of initialization
    - Cross-module wiring
    - Unified API surface
    - Background task management
    - Health checking
    - Graceful shutdown
    """

    def __init__(self, data_dir: Optional[str] = None, shikigami_registry=None):
        self.data_dir = Path(data_dir) if data_dir else Path.home() / "digital-shadow" / "data" / "shadow_os"
        self.data_dir.mkdir(parents=True, exist_ok=True)

        self.graph = ShadowGraph(graph_id="shadow_os", data_dir=str(self.data_dir / "graphs"))
        self.memory = DarkMemory(data_dir=str(self.data_dir / "memory"))
        self.crew = ShadowCrew(crew_id="shadow_os")
        self.mesh = NeuralMesh(
            node_name="shadow-os-local",
            device_type="server",
            port=8765,
        )
        self.stealth = StealthLayer(epsilon=1.0)
        self.personality = PersonalityEngine(data_dir=str(self.data_dir / "personality"))
        self.voice = ShadowVoice()
        self.anomaly_detector = AnomalyDetectionEngine()
        self.goal_planner = CognitiveGoalPlanner(data_dir=str(self.data_dir / "goals"))
        self.thought_sense = ThoughtSense()
        self.recall_synthesis = RecallSynthesis()
        self.temporal_reasoning = TemporalReasoning()
        self.empathy = EmpathyMapping()
        self.presence = PresenceDetection()
        self.trinity = CognitiveTrinity()
        self.dream = DreamState()
        self.skills = AdaptiveSkillTree()
        self.narrative = NarrativeMemory()
        self.procedural = ProceduralMemory()
        self.conflict = ConflictResolution()
        self.emotional_intel = EmotionalIntelligence()
        self.creative = CreativeSynthesis()
        self.metacognition = MetacognitionEngine()
        self.moral = MoralReasoning()
        self.attention = AttentionEngine()
        self.pipeline = CognitivePipeline(
            thought_sense=self.thought_sense,
            emotional_intelligence=self.emotional_intel,
            metacognition=self.metacognition,
            moral_reasoning=self.moral,
        )
        self.social_intel = SocialIntelligence()
        self.decision_journal = DecisionJournal()
        self.reflection = CognitiveReflection()
        self.goal_reasoning = GoalReasoning()
        self.device_context = DeviceContextEngine()
        self.arise = ShadowArise()
        self.summon = ShadowSummon()
        self.evolution = EvolutionGuide()
        self.evolutionary = EvolutionaryEngine(n_islands=3, island_size=20)
        self.oracle = OracleBridge()
        self.daemon = AutonomousDaemon(base_tick_seconds=15.0)
        self.daemon.set_shadow_os(self)
        self.daily = DailyBriefingEngine(
            memory=self.memory, personality=self.personality, mesh=self.mesh,
            evolution_guide=self.evolution,
        )
        self._escalation_manager = EscalationManager()
        self.feed_loop = FeedLoop()
        self.anticipatory = AnticipatoryEngine()
        self.resonance = CognitiveResonanceDetector()
        self.temporal_self = TemporalSelf()
        self.health_dashboard = CognitiveHealthDashboard()
        self.creative_dream = CreativeDreamIntegration(self.dream, self.creative)
        self.social_propagation = SocialGraphPropagation()
        self.neural_bridge = NeuralSymbolicBridge()
        self.quantum_field = QuantumDecisionField()
        self.grammar = EmergentGrammar()
        self.holographic = HolographicMemory()
        self.syncognitive = SyncognitiveFeedback()
        self.morphic = MorphicResonanceEngine()
        self.existential = ExistentialCalculus()
        self.chrono = ChronoSynthesis()
        self.phoenix = PhoenixRegenerator()
        self.causal = CausalReasoning()
        self.quantum_memory = QuantumMemoryField()
        self.drift_detector = SemanticDriftDetector()
        self.synthesis_engine = CrossModuleSynthesisEngine()
        self.profiler = PerformanceProfiler()
        self.intel_report = IntelligenceReportEngine()
        self.anomaly_corr = AnomalyCorrelationEngine()
        self.module_bus = CrossModuleEventBus()
        self.opt_daemon = CognitiveOptimizationDaemon(shadow_os=self)
        self.module_registry = ModuleCapabilityRegistry()
        self.arbiter = DecisionArbiter()
        self.visualizer = CognitiveVisualizer()
        self.self_mod = SelfModificationEngine()
        self.immune = CognitiveImmuneSystem()
        self.parallax = TemporalParallax()
        self.boundary = AdaptiveBoundary()
        self.resonance_engine = CognitiveResonanceEngine()
        self.economy = CognitiveEconomy()
        self.topology = CognitiveTopology()
        self.symbiosis = CognitiveSymbiosis()
        self.shadow_powers = ShadowPowers()
        self.vision = VisionPerceptionEngine()
        self.notifications = NotificationBus(shadow_os=self)
        self.visual_bridge = VisualObservationBridge(shadow_os=self)
        self.shadow_deploy = ShadowDeployEngine(shadow_os=self)
        self.journal = ShadowJournal(shadow_os=self)
        self.dashboard_agg = DashboardAggregator(shadow_os=self)
        self.predictive = PredictiveAlertEngine(shadow_os=self)
        self.shadow_pool = ShadowPool(shadow_os=self, max_workers=20)

        self.mind = MindEngine()
        self.event_bus = get_event_bus()

        self._shikigami_registry = shikigami_registry
        self.shadow_ai = None
        if shikigami_registry is not None:
            self.shadow_ai = ShadowAI(shikigami_registry, self.mind, data_dir=self.data_dir)

        self._running = False
        self._start_time: Optional[float] = None
        self._background_tasks: List[asyncio.Task] = []
        self._MAX_BACKGROUND_TASKS = 50

        self._stats_lock = threading.Lock()
        self._stats = {
            "events_processed": 0,
            "queries_handled": 0,
            "actions_taken": 0,
            "insights_generated": 0,
            "anomalies_detected": 0,
            "errors": 0,
            "uptime_seconds": 0,
            "feed_dispatch_cycles": 0,
            "feed_signals_dispatched": 0,
            "feed_signals_dropped": 0,
        }

        self._feed_dispatch_lock = threading.Lock()
        self._processed_signal_ids: set = set()
        self._processed_signal_order: deque = deque(maxlen=_FEED_DEDUP_MAX)
        self._MAX_FEED_SIGNALS_PER_CYCLE = 200

        self._wire_modules()

    def _inc_stat(self, key: str, delta: int = 1):
        with self._stats_lock:
            self._stats[key] += delta

    # ------------------------------------------------------------------
    # FeedLoop dispatcher: turn queued cross-module signals into real
    # module calls exactly once. Producers (process_observation, graph
    # handlers) call feed_loop.emit(...); the dispatcher is the consumer.
    # ------------------------------------------------------------------

    def _feed_mark_processed(self, signal_id: str) -> bool:
        """Atomically mark a signal as dispatched.

        Returns True if this call was the first to claim the ID, False if the
        signal was already dispatched. The ID set is bounded FIFO so a long
        running daemon cannot leak memory.
        """
        if not signal_id:
            return False
        with self._feed_dispatch_lock:
            if signal_id in self._processed_signal_ids:
                return False
            if len(self._processed_signal_order) >= _FEED_DEDUP_MAX:
                oldest = self._processed_signal_order.popleft()
                self._processed_signal_ids.discard(oldest)
            self._processed_signal_ids.add(signal_id)
            self._processed_signal_order.append(signal_id)
            return True

    def _dispatch_feed_signals(self) -> Dict:
        """Consume pending FeedLoop signals and route them to target modules.

        Idempotent: each signal ID is dispatched at most once. Low-intensity
        signals are dropped. Returns per-cycle counters for observability.
        """
        dispatched = 0
        dropped = 0
        try:
            signals = self.feed_loop.snapshot_signals(
                limit=self._MAX_FEED_SIGNALS_PER_CYCLE
            )
        except (AttributeError, RuntimeError, ValueError, TypeError) as e:
            logger.warning("Feed dispatch snapshot failed: %s", type(e).__name__)
            signals = []

        handlers = {
            "attention_engine": self._dispatch_to_attention,
            "personality": self._dispatch_to_personality,
            "evolution_guide": self._dispatch_to_evolution,
            "dream_state": self._dispatch_to_dream,
            "narrative_memory": self._dispatch_to_narrative,
            "cognitive_trinity": self._dispatch_to_trinity,
            "goal_reasoning": self._dispatch_to_goal_reasoning,
            "decision_journal": self._dispatch_to_decision_journal,
            "moral_reasoning": self._dispatch_to_moral_reasoning,
            "metacognition_engine": self._dispatch_to_metacognition,
            "temporal_reasoning": self._dispatch_to_temporal_reasoning,
            "creative_synthesis": self._dispatch_to_creative_synthesis,
            "oracle_bridge": self._dispatch_to_oracle,
            "empathy_mapping": self._dispatch_to_empathy,
            "conflict_resolution": self._dispatch_to_conflict,
            "social_intelligence": self._dispatch_to_social_intelligence,
            "procedural_memory": self._dispatch_to_procedural_memory,
            "daily_briefing": self._dispatch_to_daily_briefing,
            "autonomous_daemon": self._dispatch_to_daemon,
        }

        for signal in signals:
            signal_id = getattr(signal, "signal_id", "")
            if not self._feed_mark_processed(signal_id):
                continue
            targets = getattr(signal, "target_modules", []) or []
            if not targets:
                dropped += 1
                continue
            effective_intensity = float(getattr(signal, "intensity", 0.0) or 0.0)
            if effective_intensity < _FEED_DISPATCH_MIN_INTENSITY:
                dropped += 1
                continue
            handled = False
            for target in targets:
                handler = handlers.get(target)
                if handler is None:
                    continue
                try:
                    handler(signal, effective_intensity)
                    handled = True
                except (ValueError, TypeError, KeyError, AttributeError,
                        RuntimeError, OSError) as e:
                    logger.debug(
                        "Feed dispatch to %s failed: %s (%s)",
                        target, type(e).__name__, str(e)[:120],
                    )
            if handled:
                dispatched += 1
            else:
                dropped += 1

        with self._stats_lock:
            self._stats["feed_dispatch_cycles"] += 1
            self._stats["feed_signals_dispatched"] += dispatched
            self._stats["feed_signals_dropped"] += dropped
        return {"dispatched": dispatched, "dropped": dropped,
                "scanned": len(signals)}

    @staticmethod
    def _signal_content(signal) -> Dict:
        content = getattr(signal, "content", {})
        return content if isinstance(content, dict) else {}

    @staticmethod
    def _signal_topic(signal, fallback: str = "signal") -> str:
        content = ShadowOS._signal_content(signal)
        for key in ("topic", "description", "title", "intent", "primary",
                    "emotion", "content_snippet"):
            value = content.get(key)
            if value:
                return str(value)[:200]
        return f"{signal.source_module} {signal.signal_type}"[:200]

    @staticmethod
    def _signal_domain(signal, fallback: str = "general") -> str:
        content = ShadowOS._signal_content(signal)
        domain = content.get("domain")
        return str(domain)[:100] if domain else fallback

    def _dispatch_to_attention(self, signal, intensity: float) -> None:
        level = ("critical" if intensity >= 0.9 else
                 "high" if intensity >= 0.7 else
                 "medium" if intensity >= 0.4 else "low")
        self.attention.add_attention_item(
            topic=self._signal_topic(signal),
            level=level,
            domain=self._signal_domain(signal),
            urgency=round(intensity, 4),
            importance=round(intensity, 4),
        )

    def _dispatch_to_personality(self, signal, intensity: float) -> None:
        content = self._signal_content(signal)
        details = {"signal_type": signal.signal_type,
                   "source": signal.source_module,
                   "intensity": round(intensity, 4)}
        for key in ("description", "topic", "primary", "intent"):
            if content.get(key):
                details[key] = str(content[key])[:200]
        self.personality.record_behavior(
            f"feed_{signal.signal_type}", details
        )

    def _dispatch_to_evolution(self, signal, intensity: float) -> None:
        observation_type = {
            "anomaly": "anomaly",
            "pattern": "pattern",
            "goal": "behavior",
            "health": "metric",
            "cognitive": "behavior",
            "behavior": "behavior",
        }.get(signal.signal_type, "event")
        self.evolution.add_observation(
            description=self._signal_topic(signal),
            domain=self._signal_domain(signal, "productivity"),
            observation_type=observation_type,
            evidence=f"from {signal.source_module}",
            confidence=round(intensity, 4),
        )

    def _dispatch_to_dream(self, signal, intensity: float) -> None:
        self.dream.add_memory(
            content=self._signal_topic(signal),
            domain=self._signal_domain(signal),
            importance=round(intensity, 4),
            tags=[signal.signal_type],
        )

    def _dispatch_to_narrative(self, signal, intensity: float) -> None:
        self.narrative.add_event(
            content=self._signal_topic(signal),
            importance=round(intensity, 4),
            metadata={"source": signal.source_module,
                      "signal_type": signal.signal_type},
        )

    def _dispatch_to_trinity(self, signal, intensity: float) -> None:
        if intensity < _FEED_TRINITY_MIN_INTENSITY:
            return
        self.trinity.analyze(
            situation=self._signal_topic(signal),
            context={"source": signal.source_module,
                     "signal_type": signal.signal_type,
                     "intensity": round(intensity, 4)},
        )

    def _dispatch_to_goal_reasoning(self, signal, intensity: float) -> None:
        if signal.signal_type == "anomaly":
            self.goal_reasoning.detect_obstacle(
                goal_id=f"signal_{signal.source_module}"[:64],
                description=self._signal_topic(
                    signal, "anomaly affecting active goals"),
                severity=("high" if intensity >= 0.7 else
                          "medium" if intensity >= 0.4 else "low"),
            )
        else:
            self.goal_reasoning.assess_goal(
                goal_id=f"signal_{signal.source_module}"[:64],
                progress=max(0.0, min(1.0, 1.0 - intensity)),
                blockers=[self._signal_topic(signal)],
                context={"source": signal.source_module},
            )

    def _dispatch_to_decision_journal(self, signal, intensity: float) -> None:
        self.decision_journal.record_decision(
            title=self._signal_topic(signal, "signal-driven decision"),
            chosen=f"acknowledge {signal.signal_type}",
            reasoning=f"Routed from {signal.source_module}",
            confidence=round(intensity, 4),
            domain=self._signal_domain(signal),
            decision_type="signal_response",
        )

    def _dispatch_to_moral_reasoning(self, signal, intensity: float) -> None:
        if intensity < _FEED_TRINITY_MIN_INTENSITY:
            return
        self.moral.evaluate(
            action=self._signal_topic(signal, "signal response"),
            description=f"Signal from {signal.source_module}",
        )

    def _dispatch_to_metacognition(self, signal, intensity: float) -> None:
        episode = self.metacognition.start_thinking(
            reasoning_type=ReasoningType.SYSTEMATIC,
            topic=self._signal_topic(signal),
            confidence=round(intensity, 4),
            cognitive_load=round(min(1.0, intensity * 0.5), 4),
        )
        try:
            self.metacognition.end_thinking(
                episode_id=episode.episode_id,
                outcome=f"Dispatched {signal.signal_type} signal",
                confidence_after=round(intensity, 4),
            )
        except (ValueError, TypeError, KeyError, AttributeError):
            pass

    def _dispatch_to_temporal_reasoning(self, signal, intensity: float) -> None:
        self.temporal_reasoning.record_event(
            event_type=f"{signal.source_module}_{signal.signal_type}"[:200],
            metadata={"topic": self._signal_topic(signal),
                      "intensity": round(intensity, 4)},
        )

    def _dispatch_to_creative_synthesis(self, signal, intensity: float) -> None:
        if intensity < _FEED_TRINITY_MIN_INTENSITY:
            return
        content = self._signal_content(signal)
        concepts = [self._signal_topic(signal)]
        if content.get("domain"):
            concepts.append(str(content["domain"])[:100])
        self.creative.synthesize(concepts=concepts)

    def _dispatch_to_oracle(self, signal, intensity: float) -> None:
        if signal.signal_type != "prediction":
            return
        self.oracle.set_domain_state(
            domain="productivity",
            state_values={"signal_intensity": round(intensity, 4),
                          "source": signal.source_module},
            source="feed_loop",
        )

    def _dispatch_to_empathy(self, signal, intensity: float) -> None:
        content = self._signal_content(signal)
        person_id = content.get("person_id") or content.get("person")
        if not person_id:
            return
        self.empathy.record_interaction(
            person_id=str(person_id)[:64],
            interaction_type="text",
            sentiment=round((intensity * 2.0) - 1.0, 4),
            notes=self._signal_topic(signal),
        )

    def _dispatch_to_conflict(self, signal, intensity: float) -> None:
        if intensity < _FEED_TRINITY_MIN_INTENSITY:
            return
        self.conflict.create_conflict(
            title=self._signal_topic(signal, "signal conflict"),
            conflict_type="process",
            intensity=max(1, min(4, int(round(intensity * 4)))),
            description=f"Detected from {signal.source_module}",
        )

    def _dispatch_to_social_intelligence(self, signal, intensity: float) -> None:
        content = self._signal_content(signal)
        person_id = content.get("person_id") or content.get("person")
        group_id = content.get("group_id") or content.get("group")
        if not person_id or not group_id:
            return
        self.social_intel.assign_role(
            person_id=str(person_id)[:64],
            group_id=str(group_id)[:64],
            role_type="contributor",
            influence_level=round(intensity, 4),
        )

    def _dispatch_to_procedural_memory(self, signal, intensity: float) -> None:
        content = self._signal_content(signal)
        procedure_id = content.get("procedure_id")
        if not procedure_id:
            return
        self.procedural.practice_procedure(
            procedure_id=str(procedure_id)[:64],
            success=intensity >= 0.5,
        )

    def _dispatch_to_daily_briefing(self, signal, intensity: float) -> None:
        self.attention.add_attention_item(
            topic=self._signal_topic(signal),
            level="high" if intensity >= 0.7 else "medium",
            domain=self._signal_domain(signal, "health"),
            urgency=round(intensity, 4),
            importance=round(intensity, 4),
        )

    def _dispatch_to_daemon(self, signal, intensity: float) -> None:
        record = getattr(self.daemon, "record_observation_signal", None)
        if callable(record):
            record(signal.source_module, signal.signal_type,
                   self._signal_topic(signal), intensity)

    def get_feed_dispatch_stats(self) -> Dict:
        with self._stats_lock:
            return {
                "signals_dispatched": self._stats.get("feed_signals_dispatched", 0),
                "signals_dropped": self._stats.get("feed_signals_dropped", 0),
                "cycles": self._stats.get("feed_dispatch_cycles", 0),
            }

    def _wire_modules(self):
        self.graph.on_transition(self._on_cognitive_transition)
        self.graph.on_error(self._on_cognitive_error)
        self.graph.register_handler("analyze", self._handle_analyze)
        self.graph.register_handler("reason", self._handle_reason)
        self.graph.register_handler("respond", self._handle_respond)
        self.graph.register_handler("learn", self._handle_learn)
        self.graph.register_handler("reflect", self._handle_reflect)

        self.voice.register_command_handler(VoiceIntent.QUERY, self._handle_voice_query)
        self.voice.register_command_handler(VoiceIntent.COMMAND, self._handle_voice_command)

        self.personality.on_profile_update(self._on_personality_update)

        self.mesh.on_message(MeshMessageType.INSIGHT_SHARE, self._handle_mesh_insight)
        self.mesh.on_message(MeshMessageType.HEARTBEAT, self._handle_mesh_heartbeat)

        logger.info("All modules wired together")

    async def _handle_analyze(self, state: CognitiveState) -> CognitiveState:
        """Deep analysis phase: enrich context with ALL cognitive modules."""
        if len(state.observations) > MAX_OBSERVATION_KEYS:
            state.observations = state.observations[-MAX_OBSERVATION_KEYS:]

        # --- Memory enrichment ---
        search_results = []
        try:
            recent = await self.memory.get_recent(limit=20)
            for atom in recent:
                if len(state.observations) < MAX_OBSERVATION_KEYS:
                    state.observations.append(atom.to_dict())
        except (AttributeError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Memory get_recent failed: %s", type(e).__name__)

        try:
            if state.current_query:
                search_results = await self.memory.search(state.current_query, limit=5)
                for atom in search_results:
                    state.context[f"search_{atom.atom_id}"] = atom.content[:200]
        except (AttributeError, RuntimeError, ValueError, OSError) as e:
            logger.warning("Memory search failed: %s", type(e).__name__)

        # --- Anomaly detection ---
        try:
            anomalies = self.anomaly_detector.get_anomalies(limit=5, unacknowledged_only=True)
            if anomalies:
                state.context["active_anomalies"] = anomalies
                for a in anomalies:
                    try:
                        self.anomaly_detector.acknowledge(a.get("anomaly_id", ""))
                    except (ValueError, TypeError, AttributeError, KeyError):
                        pass
                if anomalies:
                    self.journal.record(
                        "alert", "Anomalies detected: %d" % len(anomalies),
                        category="security", importance=0.8,
                        tags=["anomaly","automated"]
                    )
        except (AttributeError, RuntimeError, ValueError, KeyError, TypeError) as e:
            logger.warning("Anomaly detection failed: %s", type(e).__name__)


        # --- Security auto-alert ---
        try:
            sec_events = self.daemon.guardian.get_recent_events(limit=50)
            recent_cutoff = datetime.now(timezone.utc) - timedelta(hours=1)
            recent_events = [e for e in sec_events if e.get("timestamp", "") >= recent_cutoff.isoformat()]
            if recent_events:
                self.journal_auto_security(
                    "Security events in last hour: %d" % len(recent_events),
                    max(e.get("threat_level", 1) for e in recent_events) if recent_events else 1
                )
        except (AttributeError, RuntimeError, ValueError, TypeError):
            pass

        # --- Device context (stress, focus, activity) ---
        device_ctx = self.device_context.get_context()
        base_confidence = min(1.0, len(search_results) * 0.1 + 0.3) if state.current_query or search_results else 0.3
        if device_ctx and device_ctx.context_confidence > 0.1:
            stress_penalty = {"low": 0, "moderate": 0.05, "high": 0.15, "critical": 0.3}.get(device_ctx.health.stress_level.value, 0)
            focus_bonus = {"deep_work": 0.1, "active": 0.05, "scanning": 0, "distracted": -0.1, "fatigued": -0.15, "resting": 0}.get(device_ctx.focus.value, 0)
            state.confidence = max(0.0, min(1.0, base_confidence + focus_bonus - stress_penalty))
            state.context["device_context"] = {
                "activity": device_ctx.activity.value,
                "focus": device_ctx.focus.value,
                "stress_level": device_ctx.health.stress_level.value,
                "hr": device_ctx.health.heart_rate,
            }
        else:
            state.confidence = base_confidence

        # --- ThoughtSense: detect intent, emotion, urgency from observations ---
        obs_text = state.current_query or ""
        if obs_text:
            try:
                thought = self.thought_sense.analyze(obs_text)
                thought_dict = thought.to_dict() if hasattr(thought, "to_dict") else (thought if isinstance(thought, dict) else {"urgency": "low"})
                intent_list = thought_dict.get("intents", [])
                emotion_list = thought_dict.get("emotions", [])
                state.context["intent"] = intent_list[0].get("type", "unknown") if isinstance(intent_list, list) and intent_list else "unknown"
                state.context["emotion"] = emotion_list[0].get("primary", "neutral") if isinstance(emotion_list, list) and emotion_list else "neutral"
                state.context["urgency"] = thought_dict.get("urgency", "low")
                state.context["cognitive_load"] = thought_dict.get("cognitive_load", 0.5)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("ThoughtSense analyze failed: %s", type(e).__name__)

        # --- RecallSynthesis: cross-domain pattern retrieval ---
        if search_results:
            try:
                recall_result = self.recall_synthesis.synthesize(obs_text)
                if recall_result and hasattr(recall_result, "insights"):
                    state.context["recall_insights"] = recall_result.insights[:3] if isinstance(recall_result.insights, list) else []
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("RecallSynthesis failed: %s", type(e).__name__)

        # --- Temporal patterns ---
        try:
            patterns = self.temporal_reasoning.detect_patterns("observation")
            if patterns:
                state.context["temporal_patterns"] = [p.to_dict() if hasattr(p, "to_dict") else p for p in patterns[:3]]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("TemporalReasoning failed: %s", type(e).__name__)

        # --- Attention engine: prioritize what matters ---
        try:
            priorities = self.attention.get_priorities()
            if priorities:
                state.context["attention_priorities"] = priorities[:3]
            # Detect if we should enter deep focus
            focus_state = self.attention.detect_focus_state()
            if focus_state:
                state.context["focus_state"] = focus_state
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("AttentionEngine failed: %s", type(e).__name__)

        # --- Empathy: understand social context ---
        try:
            empathy_stats = self.empathy.get_stats()
            if empathy_stats and empathy_stats.get("people_tracked", 0) > 0:
                state.context["social_empathy"] = empathy_stats.get("avg_empathy", 0.5)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("EmpathyMapping failed: %s", type(e).__name__)

        # --- Skill context: what abilities are available ---
        try:
            skill_summary = self.skills.get_domain_summary()
            if skill_summary:
                state.context["skill_domains"] = list(skill_summary.keys())[:5]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("SkillTree failed: %s", type(e).__name__)

        # --- Goal context: active goals ---
        active_goals = []
        try:
            goals = self.goal_planner.list_goals(limit=3)
            active_goals = [g for g in goals if g.get("status") in ("in_progress", "pending")]
            if active_goals:
                state.context["active_goals"] = active_goals[:3]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("GoalPlanner failed: %s", type(e).__name__)

        # --- Goal reasoning: assess obstacles ---
        if obs_text and active_goals:
            try:
                assessment = self.goal_reasoning.assess_goal(
                    active_goals[0].get("goal_id", ""),
                    current_progress=0.5,
                    context_description=obs_text[:500],
                )
                if assessment:
                    state.context["goal_assessment"] = {
                        "urgency": assessment.urgency_score,
                        "confidence": assessment.confidence,
                        "obstacles": len(assessment.obstacles),
                    }
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                logger.warning("GoalReasoning failed: %s", type(e).__name__)

        # --- Procedural memory: relevant habits/routines ---
        try:
            procedures = self.procedural.list_procedures(category="general")
            if procedures:
                state.context["procedural_habits"] = len(procedures)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("ProceduralMemory failed: %s", type(e).__name__)

        # --- Narrative context: recent life themes ---
        try:
            themes = self.narrative.get_dominant_themes(limit=5)
            if themes:
                state.context["life_themes"] = [t[0] for t in themes[:3]]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("NarrativeMemory failed: %s", type(e).__name__)

        # --- Evolution: detect friction ---
        try:
            friction = self.evolution.get_friction_points()
            if friction:
                state.context["friction_points"] = len(friction)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("EvolutionGuide failed: %s", type(e).__name__)

        # --- Oracle predictions ---
        try:
            oracle_domains = self.oracle.get_domain_states()
            if oracle_domains:
                state.context["oracle_domains"] = list(oracle_domains)[:5]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("OracleBridge failed: %s", type(e).__name__)

        # --- FeedLoop: emit signals from analysis to downstream modules ---
        try:
            if state.context.get("intent") or state.context.get("emotion"):
                self.feed_loop.emit(
                    "thought_sense", "cognitive",
                    min(1.0, state.context.get("cognitive_load", 0.5)),
                    {"intent": state.context.get("intent", "unknown"),
                     "emotion": state.context.get("emotion", "neutral"),
                     "urgency": state.context.get("urgency", "low")},
                )
            if state.context.get("active_anomalies"):
                self.feed_loop.emit(
                    "anomaly_detector", "anomaly",
                    0.8,
                    {"anomaly_count": len(state.context.get("active_anomalies", []))},
                )
            if state.context.get("device_context"):
                dc = state.context["device_context"]
                self.feed_loop.emit(
                    "device_context", "health",
                    max(0.3, {"low": 0.3, "moderate": 0.6, "high": 0.85, "critical": 0.95}.get(dc.get("stress_level", ""), 0.3)),
                    dc,
                )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("FeedLoop emit failed: %s", type(e).__name__)

        return state

    async def _handle_reason(self, state: CognitiveState) -> CognitiveState:
        """Deep reasoning phase: personality + LLM + cognitive pipeline + trinity + moral + metacognition."""
        if state.current_query:
            # --- Personality prediction ---
            try:
                prediction = self.personality.predict_decision(
                    [{"name": "option_a", "reward": 0.7, "risk_level": 0.3, "novelty": 0.5}],
                    context=state.context,
                    device_context=self.device_context,
                )
                state.context["personality_prediction"] = prediction
            except (AttributeError, RuntimeError, ValueError, TypeError, KeyError) as e:
                logger.warning("Personality prediction failed: %s", type(e).__name__)

            # --- LLM reasoning ---
            try:
                recent_events = self.event_bus.get_history(limit=10)
                observations = [{"type": e.event_type.value, "content": e.data, "timestamp": e.timestamp} for e in recent_events]
                llm_response = await self.mind.think(
                    state.current_query,
                    context=state.context,
                    observations=observations,
                )
                state.context["llm_response"] = llm_response
            except (RuntimeError, ValueError, ConnectionError, OSError, TimeoutError) as e:
                logger.warning("MindEngine think failed: %s", type(e).__name__)

            # --- Cognitive Pipeline: ThoughtSense → Emotion → Metacognition → Moral ---
            try:
                pipeline_result = self.pipeline.process(
                    text=state.current_query,
                    context=state.context,
                )
                if pipeline_result:
                    state.context["cognitive_pipeline"] = {
                        "overall_confidence": round(pipeline_result.get("overall_confidence", 0.5), 3),
                        "modules_used": pipeline_result.get("modules_used", []),
                    }
                    intent_data = pipeline_result.get("thought", {})
                    if intent_data:
                        state.context["pipeline_intent"] = intent_data.get("intent", "unknown")
                        state.context["pipeline_emotion"] = intent_data.get("emotion", "neutral")
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                logger.warning("CognitivePipeline failed: %s", type(e).__name__)

            # --- Trinity: 3-mind decision synthesis ---
            try:
                trinity_result = self.trinity.analyze(
                    situation=state.current_query,
                    context=state.context,
                )
                if trinity_result and hasattr(trinity_result, "synthesis"):
                    state.context["trinity_consensus"] = trinity_result.synthesis.get("consensus", "split")
                    state.context["trinity_confidence"] = round(trinity_result.synthesis.get("confidence", 0.5), 3)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Trinity analysis failed: %s", type(e).__name__)

            # --- Moral reasoning: ethical guardrails ---
            concern_text = state.context.get("emotion", "") + " " + state.current_query
            if concern_text:
                try:
                    moral_eval = self.moral.evaluate(
                        action=state.current_query[:500],
                    )
                    if moral_eval:
                        moral_dict = moral_eval.to_dict() if hasattr(moral_eval, "to_dict") else moral_eval
                        state.context["moral_assessment"] = {
                            "dominant_framework": moral_dict.get("dominant_framework", "unknown"),
                            "ethical_score": round(moral_dict.get("ethical_score", 0.5), 3),
                            "has_conflicts": len(moral_dict.get("conflicts", [])) > 0,
                        }
                except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                    logger.warning("MoralReasoning failed: %s", type(e).__name__)

            # --- Metacognition: bias detection and cognitive load ---
            try:
                meta_result = self.metacognition.start_thinking(
                    reasoning_type=ReasoningType.SYSTEMATIC,
                    topic=state.current_query[:500],
                    confidence=state.confidence,
                )
                if meta_result:
                    meta_dict = meta_result.to_dict() if hasattr(meta_result, "to_dict") else meta_result
                    state.context["metacognition"] = {
                        "biases_detected": len(meta_dict.get("biases", [])),
                        "cognitive_load": round(meta_dict.get("cognitive_load", 0.5), 3),
                        "episode_id": getattr(meta_result, "episode_id", ""),
                    }
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("MetacognitionEngine failed: %s", type(e).__name__)

            # --- Social intelligence: group dynamics ---
            try:
                groups = self.social_intel.list_groups()
                if groups:
                    state.context["social_groups"] = len(groups)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("SocialIntelligence failed: %s", type(e).__name__)

            # --- Conflict detection ---
            try:
                conflicts = self.conflict.list_conflicts()
                active_conflicts = [c for c in conflicts if c.get("status") == "mediating"]
                if active_conflicts:
                    state.context["active_conflicts"] = len(active_conflicts)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("ConflictResolution failed: %s", type(e).__name__)

            # --- Decision journal: record reasoning context ---
            try:
                decision_id = self.decision_journal.record_decision(
                    title=state.current_query[:200],
                    description=state.current_query[:500],
                    decision_type="query",
                    chosen="analyze",
                    alternatives=["respond", "delegate", "escalate"],
                    reasoning="Automatic cognitive analysis",
                    confidence=state.confidence,
                    metadata={"mode": state.mode.value, "domain": state.domain},
                )
                state.context["decision_id"] = decision_id
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("DecisionJournal failed: %s", type(e).__name__)

        if state.urgency > 0.7:
            state.active_abilities = ["oracle", "intent_seer"]

        return state

    async def _handle_respond(self, state: CognitiveState) -> CognitiveState:
        """Deep response phase: generate action, update narrative/procedural/goals, bridge to shadow_ai."""
        self._inc_stat("queries_handled")

        llm_response = state.context.get("llm_response", "")
        response_content = f"Responded to: {state.current_query or 'observation'}"
        if llm_response:
            response_content = llm_response

        # --- Store response in memory ---
        await self.memory.store(
            content=response_content,
            atom_type=AtomType.ACTION,
            metadata={
                "cognitive_mode": state.mode.value,
                "confidence": state.confidence,
                "trinity_consensus": state.context.get("trinity_consensus", ""),
                "moral_score": state.context.get("moral_assessment", {}).get("ethical_score", 0.5),
            },
        )

        # --- Creative synthesis: generate ideas from context ---
        if state.current_query:
            try:
                ideas = self.creative.synthesize(
                    concepts=[state.current_query[:200]],
                    method=SynthesisMethod.COMBINATION,
                )
                if ideas and hasattr(ideas, 'ideas') and ideas.ideas:
                    top_idea = ideas.ideas[0]
                    state.context["creative_suggestion"] = getattr(top_idea, 'description', str(top_idea))[:200]
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("CreativeSynthesis failed: %s", type(e).__name__)

        # --- Narrative memory: record event ---
        try:
            self.narrative.add_event(
                content=response_content[:500],
                importance=round(state.confidence, 2),
                metadata={"event_type": "action", "domain": state.domain or "general"},
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("NarrativeMemory add_event failed: %s", type(e).__name__)

        # --- Procedural memory: mark skill usage ---
        try:
            proc = self.procedural.get_procedure("cognitive_analysis")
            if proc:
                self.procedural.practice_procedure(
                    procedure_id="cognitive_analysis",
                    success=state.confidence > 0.5,
                )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("ProceduralMemory practice failed: %s", type(e).__name__)

        # --- Attention: mark priority resolved ---
        try:
            if state.current_query:
                self.attention.add_attention_item(
                    topic=state.current_query[:100],
                    level="medium",
                    domain=state.domain or "general",
                    urgency=state.urgency,
                    importance=state.confidence,
                )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("AttentionEngine add failed: %s", type(e).__name__)

        # --- Evolution: record observation for friction detection ---
        try:
            self.evolution.observe(
                observation_type="action",
                content=response_content[:300],
                domain=state.domain or "general",
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("EvolutionGuide observe failed: %s", type(e).__name__)

        # --- Oracle: record state for prediction ---
        try:
            if state.domain:
                self.oracle.set_domain_state(
                    domain=state.domain,
                    value=round(state.confidence, 2),
                )
        except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
            logger.warning("OracleBridge set_domain_state failed: %s", type(e).__name__)

        # --- Bridge to ShadowAI ---
        if self.shadow_ai and state.current_query:
            try:
                await self.shadow_ai.observe({
                    "type": "response",
                    "content": state.current_query,
                    "response": llm_response[:500] if llm_response else "",
                })
            except (AttributeError, RuntimeError, ValueError, ConnectionError) as e:
                logger.warning("ShadowAI bridge in _handle_respond failed: %s", type(e).__name__)

        # --- FeedLoop: emit response signals ---
        try:
            if state.context.get("creative_suggestion"):
                self.feed_loop.emit("creative_synthesis", "pattern", 0.6,
                                   {"suggestion": state.context.get("creative_suggestion", "")[:200]})
            if state.context.get("moral_assessment"):
                self.feed_loop.emit("moral_reasoning", "cognitive",
                                    state.context.get("moral_assessment", {}).get("ethical_score", 0.5),
                                    state.context.get("moral_assessment", {}))
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("FeedLoop emit in respond failed: %s", type(e).__name__)

        return state

    async def _handle_learn(self, state: CognitiveState) -> CognitiveState:
        """Deep learning phase: update personality, skills, decisions, empathy, social, and goal progress."""
        # --- Personality recording ---
        if state.current_query:
            self.personality.record_behavior("question_asked", {
                "query": state.current_query[:200],
                "domain": state.domain or "general",
                "complexity": "moderate",
            })

        if state.decisions:
            for decision in state.decisions:
                self.personality.record_behavior("decision_made", decision)

        # --- Skill usage tracking ---
        if state.active_abilities:
            for ability in state.active_abilities:
                try:
                    self.skills.use_skill(
                        skill_id=ability,
                        success=state.confidence > 0.5,
                    )
                except (ValueError, TypeError, AttributeError, RuntimeError):
                    pass

        # --- Empathy: learn from interaction context ---
        if state.context.get("emotion"):
            try:
                self.emotional_intel.record_pattern(
                    pattern_type=PatternType.CYCLE,
                    name="interaction_emotion",
                    emotions=[EmotionCategory.NEUTRAL],
                    metadata={"intensity": 0.5, "domain": state.domain or "general"},
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("EmotionalIntelligence record_pattern failed: %s", type(e).__name__)

        # --- Goal progress update ---
        if state.context.get("active_goals"):
            try:
                for goal_data in state.context["active_goals"][:3]:
                    goal_id = goal_data.get("goal_id", "")
                    if goal_id:
                        goal = self.goal_planner.get_goal(goal_id)
                        if goal and goal.get("steps"):
                            for step_data in goal["steps"][:3]:
                                if step_data.get("status") == "pending":
                                    self.goal_planner.complete_step(goal_id, step_data.get("step_id", ""),
                                                                    {"confidence": state.confidence})
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("GoalPlanner track failed: %s", type(e).__name__)

        # --- Recall: observe for cross-domain patterns ---
        if state.current_query and state.domain:
            try:
                self.recall_synthesis.add_observation(
                    content=state.current_query[:300],
                    domain=state.domain,
                    importance=round(state.confidence, 2),
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("RecallSynthesis observe failed: %s", type(e).__name__)

        # --- Attention: learn from focus patterns ---
        try:
            self.attention.add_attention_item(
                topic=state.current_query[:100] if state.current_query else "observation",
                level="medium" if state.confidence > 0.5 else "low",
                domain=state.domain or "general",
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("AttentionEngine learn failed: %s", type(e).__name__)

        # --- Decision journal: update decision outcome ---
        decision_id = state.context.get("decision_id")
        if decision_id:
            try:
                self.decision_journal.update_status(
                    decision_id=decision_id,
                    status="completed",
                )
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError):
                pass

        return state

    async def _handle_reflect(self, state: CognitiveState) -> CognitiveState:
        """Deep reflection phase: distill personality, compact memory, reflect on decisions, dream consolidation."""
        # --- Personality distillation ---
        try:
            distill_result = await self.personality.distill()
            state.metadata["distill_result"] = distill_result
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.warning("Personality distill failed: %s", type(e).__name__)

        # --- Memory compaction ---
        try:
            compacted = await self.memory.compact()
            state.metadata["memory_compacted"] = compacted
        except (OSError, ValueError, RuntimeError, AttributeError) as e:
            logger.warning("Memory compact failed: %s", type(e).__name__)

        # --- Cognitive reflection: self-assess reasoning ---
        if state.current_query:
            try:
                reflection = self.reflection.add_reflection(
                    topic=state.current_query[:200],
                    description="Processed query with confidence %.2f" % state.confidence,
                    confidence_before=state.confidence,
                )
                if reflection:
                    state.metadata["reflection_id"] = getattr(reflection, "entry_id", "")
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("CognitiveReflection failed: %s", type(e).__name__)

        # --- Dream state: offline memory consolidation ---
        try:
            result = self.dream.consolidate()
            if result:
                state.metadata["dream_insights"] = getattr(result, "insights_generated", 0)
                state.metadata["dream_patterns"] = getattr(result, "patterns_strengthened", 0) or getattr(result, "patterns_strengthened_count", 0)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("DreamState consolidation failed: %s", type(e).__name__)

        # --- Metacognition: end thinking session ---
        if state.context.get("metacognition"):
            try:
                self.metacognition.end_thinking(
                    outcome="completed",
                    confidence_after=state.confidence,
                )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Metacognition end_thinking failed: %s", type(e).__name__)

        # --- Sync autonomous daemon vitals ---
        try:
            self.sync_evolution_from_device()
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("EvolutionGuide sync failed: %s", type(e).__name__)

        return state

    def _on_cognitive_transition(self, prev: CognitiveState, current: CognitiveState, node: str):
        logger.info("Cognitive transition: %s -> %s (via %s)", prev.mode.value, current.mode.value, node)

    def _on_cognitive_error(self, state: CognitiveState, error: Exception):
        logger.error("Cognitive error in %s: %s", state.mode.value, type(error).__name__)
        self._inc_stat("errors")

    async def _handle_voice_query(self, command: VoiceCommand) -> Dict:
        result = await self.process_query(command.transcript)
        return {"command": command.to_dict(), "result": result}

    async def _handle_voice_command(self, command: VoiceCommand) -> Dict:
        mission = await self.crew.create_mission(
            objective=command.transcript,
            mode=DelegationMode.HIERARCHICAL,
        )
        result = await self.crew.execute_mission(mission.mission_id)
        return {"command": command.to_dict(), "mission_result": result}

    def _on_personality_update(self, profile: PersonalityProfile):
        persona_block = profile.to_system_prompt()
        self.memory.set_block("persona", persona_block)
        human_block = self.memory.get_block("human")
        if not human_block or not human_block.get("protected", False):
            self.memory.set_block("human", profile.big_five.to_prompt_fragment())

    async def _handle_mesh_insight(self, message: MeshMessage):
        insight = message.payload.get("insight", {})
        await self.memory.store(
            content=json.dumps(insight)[:500],
            atom_type=AtomType.INSIGHT,
            metadata={"source": "mesh", "sender": message.sender_id},
        )
        self._inc_stat("insights_generated")

    async def _handle_mesh_heartbeat(self, message: MeshMessage):
        status = message.payload.get("status", "online")
        if status not in ("online", "offline", "busy", "dormant", "syncing"):
            logger.warning("Invalid peer status from %s: %s", message.sender_id[:16].replace(chr(10), ' '), str(status).replace(chr(10), ' '))
            return
        capabilities = message.payload.get("capabilities", [])
        if not isinstance(capabilities, list) or len(capabilities) > 50:
            capabilities = []
        validated_capabilities = [
            c for c in capabilities if isinstance(c, str) and len(c) < 100
        ]
        try:
            updated = self.mesh.update_peer_status(
                node_id=message.sender_id,
                status=status,
                capabilities=validated_capabilities,
            )
            if not updated:
                logger.debug("Heartbeat from unknown peer %s", message.sender_id[:16])
        except (ValueError, TypeError, AttributeError, KeyError, RuntimeError):
            logger.warning("Failed to update peer %s from heartbeat", message.sender_id)

    async def start(self):
        if self._running:
            return

        self._running = True
        self._start_time = time.monotonic()

        await self.graph.resume_from_checkpoint()
        await self.memory.start_auto_compaction()
        await self.mesh.start()
        await self.daemon.start()

        self._health_task = asyncio.create_task(self._health_monitor_loop())
        logger.info("ShadowOS started - all modules initialized + autonomous daemon running")

    async def stop(self):
        if not self._running:
            return

        self._running = False

        await self.daemon.stop()

        if hasattr(self, '_health_task') and self._health_task:
            self._health_task.cancel()
            try:
                await self._health_task
            except asyncio.CancelledError:
                pass

        await self._save_state()

        for task in self._background_tasks:
            task.cancel()
        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        await self.mesh.stop()
        await self.voice.close()
        self.memory.close()
        logger.info("ShadowOS stopped gracefully")

    def close(self):
        """Release resources without a full async stop().

        Safe to call repeatedly. Closes the DarkMemory SQLite connection so
        Windows test temp-dir cleanup and process shutdown don't hit
        sharing violations on shadow_memory.db.
        """
        try:
            self.memory.close()
        except (OSError, RuntimeError, AttributeError) as e:
            logger.debug("ShadowOS close error: %s", type(e).__name__)

    async def _save_state(self):
        state_file = self.data_dir / "shadow_os_state.json"
        try:
            state = {
                "stats": dict(self._stats),
                "personality": self.personality.get_profile(),
                "memory_blocks": self.memory.list_blocks(),
                "graph_mode": self.graph.state.mode.value,
                "peers": {pid: p.to_dict() for pid, p in list(self.mesh._peers.items())[:50]},
                "saved_at": datetime.now(timezone.utc).isoformat(),
            }
            tmp_file = state_file.with_suffix(".tmp")
            tmp_file.write_text(json.dumps(state, indent=2, default=str), encoding="utf-8")
            fd = os.open(str(tmp_file), os.O_WRONLY)
            os.fsync(fd)
            os.close(fd)
            fsync_dir(self.data_dir)
            tmp_file.replace(state_file)
        except (OSError, TypeError, ValueError) as e:
            logger.error("Failed to save state: %s", type(e).__name__)

    async def _load_state(self):
        state_file = self.data_dir / "shadow_os_state.json"
        if not state_file.exists():
            return
        try:
            state = json.loads(state_file.read_text())
            with self._stats_lock:
                self._stats.update(state.get("stats", {}))
            for block_info in state.get("memory_blocks", []):
                label = block_info.get("label")
                if label and not block_info.get("protected"):
                    self.memory.set_block(label, "")
            logger.info("Restored state from %s", state.get('saved_at', 'unknown'))
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, AttributeError, OSError) as e:
            logger.error("Failed to load state: %s", type(e).__name__)

    async def _health_monitor_loop(self):
        while self._running:
            try:
                await asyncio.sleep(60)
                health = self.get_health()
                if health.get("status") != "healthy":
                    logger.warning("ShadowOS health check: %s", health.get('status'))
                if self.memory.get_stats().get("total_atoms", 0) > self.memory._max_atoms * 0.9:
                    evicted = await self.memory.compact()
                    logger.info("Health-triggered compaction: %s atoms evicted", evicted)
                self._auto_update_health_dashboard()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error("Health monitor error: %s", type(e).__name__)

    def _auto_update_health_dashboard(self):
        modules_map = {
            "shadow_graph": self.graph,
            "dark_memory": self.memory,
            "shadow_crew": self.crew,
            "neural_mesh": self.mesh,
            "stealth_layer": self.stealth,
            "personality": self.personality,
            "shadow_voice": self.voice,
            "anomaly_detector": self.anomaly_detector,
            "goal_planner": self.goal_planner,
            "thought_sense": self.thought_sense,
            "recall_synthesis": self.recall_synthesis,
            "temporal_reasoning": self.temporal_reasoning,
            "empathy": self.empathy,
            "presence": self.presence,
            "cognitive_trinity": self.trinity,
            "dream_state": self.dream,
            "skills": self.skills,
            "narrative": self.narrative,
            "procedural": self.procedural,
            "conflict_resolution": self.conflict,
            "emotional_intelligence": self.emotional_intel,
            "creative_synthesis": self.creative,
            "metacognition": self.metacognition,
            "moral_reasoning": self.moral,
            "attention": self.attention,
            "cognitive_pipeline": self.pipeline,
            "social_intelligence": self.social_intel,
            "decision_journal": self.decision_journal,
            "cognitive_reflection": self.reflection,
            "goal_reasoning": self.goal_reasoning,
            "device_context": self.device_context,
            "shadow_arise": self.arise,
            "shadow_summon": self.summon,
            "evolution_guide": self.evolution,
            "evolutionary_engine": self.evolutionary,
            "oracle_bridge": self.oracle,
            "feed_loop": self.feed_loop,
            "anticipatory": self.anticipatory,
            "resonance": self.resonance,
            "temporal_self": self.temporal_self,
            "creative_dream": self.creative_dream,
            "social_propagation": self.social_propagation,
            "causal": self.causal,
            "quantum_memory": self.quantum_memory,
            "drift_detector": self.drift_detector,
            "immune": self.immune,
            "parallax": self.parallax,
            "boundary": self.boundary,
            "resonance_engine": self.resonance_engine,
            "economy": self.economy,
            "topology": self.topology,
            "symbiosis": self.symbiosis,
            "synthesis_engine": self.synthesis_engine,
            "visualizer": self.visualizer,
            "self_mod": self.self_mod,
            "intel_report": self.intel_report,
            "anomaly_corr": self.anomaly_corr,
            "module_bus": self.module_bus,
        }
        self.health_dashboard.auto_update(modules_map)

    async def process_query(self, query: str, domain: Optional[str] = None) -> Dict:
        query = sanitize_input(query)
        if domain:
            domain = sanitize_input(domain, max_length=50)
        self._inc_stat("queries_handled")

        await self.memory.store(
            content=query,
            atom_type=AtomType.OBSERVATION,
            domain=domain,
            metadata={"source": "user_query"},
        )

        result = await self.graph.process_event(
            event={"query": query, "domain": domain},
            trigger=TransitionTrigger.HUMAN_REQUEST,
        )

        self.personality.record_behavior("question_asked", {
            "query": query,
            "domain": domain,
            "complexity": "moderate" if len(query) > 20 else "simple",
        })

        return {
            "query": query,
            "cognitive_mode": result.mode.value,
            "confidence": result.confidence,
            "domain": result.domain,
            "context_size": len(result.context),
        }

    async def ask(self, question: str, context: Optional[Dict] = None) -> Dict:
        """
        Unified ask — combines v2 cognitive processing with v1 LLM reasoning.
        
        Routes through ShadowGraph (analyze→reason→respond) and feeds
        the cognitive context into MindEngine for LLM-powered responses.
        """
        question = sanitize_input(question)
        if context is None:
            context = {}

        await self.memory.store(
            content=question,
            atom_type=AtomType.OBSERVATION,
            metadata={"source": "unified_ask"},
        )

        graph_result = await self.graph.process_event(
            event={"query": question, "context": context},
            trigger=TransitionTrigger.HUMAN_REQUEST,
        )

        cognitive_context = dict(graph_result.context)
        llm_response = cognitive_context.get("llm_response", "")

        if not llm_response:
            try:
                recent_obs = []
                try:
                    recent = await self.memory.get_recent(limit=10)
                    recent_obs = [a.to_dict() for a in recent]
                except (OSError, ValueError, RuntimeError, AttributeError):
                    pass
                llm_response = await self.mind.think(
                    question,
                    context=cognitive_context,
                    observations=recent_obs,
                )
            except (RuntimeError, ValueError, ConnectionError, OSError, TimeoutError) as e:
                logger.warning("MindEngine fallback think failed: %s", type(e).__name__)
                llm_response = ""

        personality_style = self.personality.get_profile().get("communication_style", "casual")

        self.personality.record_behavior("question_asked", {
            "query": question,
            "complexity": "moderate" if len(question) > 20 else "simple",
        })

        response = {
            "response": llm_response,
            "query": question,
            "cognitive_mode": graph_result.mode.value,
            "confidence": graph_result.confidence,
            "personality_style": personality_style,
        }

        if self.shadow_ai:
            try:
                await self.shadow_ai.observe({
                    "type": "ask",
                    "content": question[:500],
                    "response": llm_response[:500] if llm_response else "",
                })
            except (AttributeError, RuntimeError, ValueError, ConnectionError) as e:
                logger.warning("ShadowAI bridge failed: %s", type(e).__name__)

        # --- Feed evolutionary engine with outcomes ---
        try:
            self.evolutionary.record_outcome(
                question[:200],
                {
                    "accuracy": round(graph_result.confidence, 3),
                    "calibration": round(graph_result.confidence, 3),
                    "quality": round(len(llm_response) / max(1, len(question)), 3) if llm_response else 0.5,
                    "diversity": 0.7,
                    "efficiency": 0.8,
                    "robustness": 0.7,
                },
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Evolutionary engine record_outcome failed: %s", type(e).__name__)

        return response

    async def stream_ask(self, question: str, context: Optional[Dict] = None):
        """
        Streaming ask — yields SSE chunks as the LLM generates the response.
        Delegates to MindEngine's streaming if available, otherwise falls back.
        """
        question = sanitize_input(question)
        graph_result = await self.graph.process_event(
            event={"query": question, "context": context or {}},
            trigger=TransitionTrigger.HUMAN_REQUEST,
        )

        yield {"type": "cognitive_state", "mode": graph_result.mode.value, "confidence": graph_result.confidence}

        try:
            yield {"type": "response", "content": "Streaming not yet implemented for Ollama Cloud API. Use /os/ask for full responses."}
        except (RuntimeError, ValueError, ConnectionError, OSError) as e:
            yield {"type": "error", "content": type(e).__name__}

    async def observe_event(self, event: Dict) -> Dict:
        """
        Unified observe — bridges v1 and v2 observation pipelines.
        Routes through StealthLayer, DarkMemory, ShadowGraph, and v1 ShadowAI.
        """
        result = await self.process_observation(event)

        if self.shadow_ai:
            try:
                await self.shadow_ai.observe(event)
            except (AttributeError, RuntimeError, ValueError, ConnectionError) as e:
                logger.warning("ShadowAI observe bridge failed: %s", type(e).__name__)

        try:
            from ..event_bus import ShadowEvent, EventType
            await self.event_bus.publish(ShadowEvent(
                event_type=EventType.OBSERVATION,
                source="shadow_os",
                data=event,
                priority=3,
            ))
        except (AttributeError, RuntimeError, ValueError) as e:
            logger.warning("EventBus publish failed: %s", type(e).__name__)

        return result

    async def process_observation(self, observation: Dict) -> Dict:
        observation = validate_observation(observation)
        self._inc_stat("events_processed")

        # Enrich observation with device context
        device_ctx = self.device_context.get_context()
        if device_ctx and device_ctx.context_confidence > 0.1:
            if device_ctx.activity.value != "unknown" and "activity" not in observation:
                observation["activity"] = device_ctx.activity.value
            if device_ctx.health.stress_level.value != "low" and "stress" not in observation:
                observation["stress_level"] = device_ctx.health.stress_level.value
            if device_ctx.focus.value not in ("scanning", "unknown") and "focus_level" not in observation:
                observation["focus_level"] = device_ctx.focus.value
            if device_ctx.environment.battery_level is not None and "battery" not in observation:
                observation["battery"] = device_ctx.environment.battery_level
            if device_ctx.digital.screen_on and "screen_active" not in observation:
                observation["screen_active"] = True
            if device_ctx.digital.active_app and "active_app" not in observation:
                observation["active_app"] = device_ctx.digital.active_app
            if device_ctx.people_nearby and "people_nearby" not in observation:
                observation["people_nearby"] = len(device_ctx.people_nearby)

        obs_type = observation.get("type", "observation")
        content = observation.get("content", str(observation))
        domain = observation.get("domain")
        sensitivity = self.stealth.get_sensitivity_score(observation)

        category = self._infer_category(obs_type, sensitivity)
        granted, reason = self.stealth.check_access(
            "shadow_ai", AccessAction.WRITE, obs_type, category
        )

        if not granted:
            return {"status": "denied", "reason": reason}

        importance = max(0.1, 1.0 - sensitivity * 0.5)
        atom = await self.memory.store(
            content=str(content),
            atom_type=self._map_atom_type(obs_type),
            domain=domain,
            importance=importance,
            metadata={"source": "observation", "sensitivity": sensitivity},
        )

        result = await self.graph.process_event(
            event={"observations": [observation]},
            trigger=TransitionTrigger.EVENT_RECEIVED,
        )

        self.personality.record_behavior(obs_type, observation)

        # --- Cognitive Memory Bridge: auto-trigger modules based on observation ---
        await self._enrich_with_cognitive_modules(observation, obs_type, domain or "general", importance)

        anomaly = self.anomaly_detector.observe_temporal(obs_type)
        if anomaly:
            self._inc_stat("anomalies_detected")

        # Consume any signals emitted above so routed modules actually run.
        try:
            self._dispatch_feed_signals()
        except (ValueError, TypeError, KeyError, AttributeError, RuntimeError, OSError) as e:
            logger.warning("Feed dispatch failed: %s", type(e).__name__)

        if sensitivity < 0.5:
            try:
                peers = self.mesh.get_peers()
                if peers:
                    safe_data = self.stealth.transform_for_sharing(observation, CamouflageMode.ANONYMIZE)
                    await self.mesh.share_insight({"type": obs_type, "data": safe_data})
            except (ConnectionError, OSError, RuntimeError, ValueError):
                pass

        return {
            "status": "observed",
            "atom_id": atom.atom_id,
            "cognitive_mode": result.mode.value,
            "sensitivity": round(sensitivity, 3),
        }

    def get_enriched_context(self) -> Dict:
        """Get current context enriched with device data (iPhone/Watch/Laptop)."""
        ctx = self.device_context.get_context()
        if ctx is None:
            return {"available": False, "reason": "no_device_data"}
        result = ctx.to_dict()
        result["available"] = True

        enrichment = {}
        if ctx.activity != ActivityType.UNKNOWN:
            enrichment["activity"] = ctx.activity.value
        if ctx.focus != FocusLevel.SCANNING:
            enrichment["focus"] = ctx.focus.value
        if ctx.health.stress_level != StressLevel.LOW:
            enrichment["stress"] = ctx.health.stress_level.value
        if ctx.health.heart_rate:
            enrichment["heart_rate"] = ctx.health.heart_rate
        if ctx.health.sleep_phase != "unknown":
            enrichment["sleep_phase"] = ctx.health.sleep_phase.value if hasattr(ctx.health.sleep_phase, 'value') else ctx.health.sleep_phase
        if ctx.environment.battery_level is not None and ctx.environment.battery_level < 0.2:
            enrichment["low_battery"] = True

        result["enrichment"] = enrichment
        result["confidence"] = ctx.context_confidence
        return result

    async def execute_action(self, action: str, parameters: Dict) -> Dict:
        action = sanitize_input(action, max_length=500)
        parameters = validate_observation(parameters)
        self._inc_stat("actions_taken")

        mission = await self.crew.create_mission(
            objective=action,
            mode=DelegationMode.HIERARCHICAL,
        )

        result = await self.crew.execute_mission(mission.mission_id)

        await self.memory.store(
            content=f"Action: {action} with params: {json.dumps(parameters)[:200]}",
            atom_type=AtomType.ACTION,
            importance=0.7,
            metadata={"action": action, "mission_id": mission.mission_id, "result_status": result.get("status")},
        )

        self.personality.record_behavior("task_completed", {
            "type": action,
            "speed": "moderate",
            "quality": "medium",
            "on_time": True,
        })

        return result

    async def create_mission(self, objective: str, mode: DelegationMode = DelegationMode.HIERARCHICAL) -> Dict:
        mission = await self.crew.create_mission(objective, mode)
        return mission.to_dict()

    async def execute_mission(self, mission_id: str) -> Dict:
        return await self.crew.execute_mission(mission_id)

    async def start_voice_session(self, voice: Optional[str] = None) -> Dict:
        session = await self.voice.start_session(voice)
        return session.to_dict()

    async def process_voice(self, transcript: str) -> Dict:
        return await self.voice.process_speech(transcript)

    async def end_voice_session(self) -> Optional[Dict]:
        return await self.voice.end_session()

    # --- Cross-Module Synthesis convenience methods ---

    def synthesis_ingest(self, module: str, category: str, signal_type: str,
                         content: str, confidence: float, domains=None, entities=None) -> str:
        """Convenience: ingest a module signal into the synthesis engine."""
        from .cross_module_synthesis import ModuleSignal
        signal = ModuleSignal(
            source_module=module, category=category, signal_type=signal_type,
            content=content, confidence=confidence,
            domains=domains or [], entities=entities or [],
        )
        return self.synthesis_engine.ingest_signal(signal)

    def synthesis_run(self, max_syntheses: int = 5) -> List[Dict]:
        """Convenience: run cross-module synthesis and return results."""
        return [s.to_dict() for s in self.synthesis_engine.synthesize(max_syntheses=max_syntheses)]

    def synthesis_get_latest(self, limit: int = 10) -> List[Dict]:
        """Convenience: get latest syntheses."""
        return self.synthesis_engine.get_synthesis_history(limit=limit)

    def synthesis_get_stats(self) -> Dict:
        """Convenience: get synthesis engine stats."""
        return self.synthesis_engine.get_stats()

    def _infer_category(self, obs_type: str, sensitivity: float) -> DataCategory:
        category_map = {
            "location": DataCategory.LOCATION,
            "bluetooth": DataCategory.DEVICE,
            "wifi": DataCategory.NETWORK,
            "communication": DataCategory.COMMUNICATION,
            "financial": DataCategory.FINANCIAL,
            "health": DataCategory.HEALTH,
            "credential": DataCategory.CREDENTIALS,
        }
        return category_map.get(obs_type, DataCategory.PERSONAL)

    def _map_atom_type(self, obs_type: str) -> AtomType:
        type_map = {
            "decision": AtomType.DECISION,
            "pattern": AtomType.PATTERN,
            "person": AtomType.PERSON,
            "insight": AtomType.INSIGHT,
            "action": AtomType.ACTION,
        }
        return type_map.get(obs_type, AtomType.OBSERVATION)

    async def _enrich_with_cognitive_modules(self, observation: Dict, obs_type: str,
                                              domain: str, importance: float) -> None:
        content = observation.get("content", str(observation))[:500]
        if not content:
            return

        try:
            result = self.emotional_intel.detect_emotion(content)
            if result:
                edict = result.to_dict() if hasattr(result, "to_dict") else {}
                primary = edict.get("primary_emotion", "neutral")
                intensity = edict.get("intensity", 0.3)
                self.feed_loop.emit("emotional_intelligence", "emotion",
                                   intensity, {"primary": primary, "content_snippet": content[:100]})
                self.feed_loop.emit("emotional_intelligence", "social",
                                    intensity * 0.7, {"emotion": primary})
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Emotion enrichment failed: %s", type(e).__name__)

        try:
            thought = self.thought_sense.analyze(content)
            if thought:
                td = thought.to_dict() if hasattr(thought, "to_dict") else {}
                urgency = td.get("urgency", "low")
                if urgency in ("high", "critical"):
                    self.feed_loop.emit("thought_sense", "cognitive", 0.8,
                                       {"intent": td.get("intent", "unknown"), "urgency": urgency})
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Thought enrichment failed: %s", type(e).__name__)

        try:
            self.recall_synthesis.add_observation(text=content, domain=domain, importance=importance)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Recall enrichment failed: %s", type(e).__name__)

        try:
            self.temporal_reasoning.record_event(
                event_type=obs_type,
                timestamp=datetime.now(timezone.utc).isoformat(),
                metadata={"content": content[:500], "domain": domain},
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Temporal enrichment failed: %s", type(e).__name__)

        try:
            level = "high" if importance > 0.7 else "medium" if importance > 0.4 else "low"
            self.attention.add_attention_item(
                topic=content[:100],
                level=level,
                domain=domain,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Attention enrichment failed: %s", type(e).__name__)

        try:
            self.narrative.add_event(
                content=content,
                importance=importance,
                metadata={"event_type": obs_type, "domain": domain},
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Narrative enrichment failed: %s", type(e).__name__)

        try:
            if domain in ("career", "work", "productivity", "technical", "learning"):
                self.skills.use_skill(skill_id=f"observe_{domain}", success=importance > 0.5)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Skill enrichment failed: %s", type(e).__name__)

        try:
            goals = self.goal_planner.list_goals(limit=3)
            active = [g for g in goals if g.get("status") in ("in_progress", "pending")]
            if active and importance > 0.5:
                self.feed_loop.emit("goal_reasoning", "goal", importance,
                                   {"observation": content[:200], "active_goals": len(active)})
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Goal enrichment failed: %s", type(e).__name__)

        try:
            self.evolution.observe(
                observation_type=obs_type,
                content=content,
                domain=domain,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Evolution enrichment failed: %s", type(e).__name__)

        try:
            self.procedural.practice_procedure(
                procedure_id=f"observe_{obs_type}",
                success=importance > 0.5,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Procedural enrichment failed: %s", type(e).__name__)

        obs_people = observation.get("people_nearby", 0)
        if isinstance(obs_people, int) and obs_people > 0:
            try:
                self.empathy.record_interaction(
                    person_id="group",
                    interaction_type="proximity",
                    sentiment=0.5,
                    notes=f"Near {obs_people} people during {obs_type}",
                )
                self.feed_loop.emit("presence_detection", "social", 0.6,
                                    {"people_nearby": obs_people})
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Empathy enrichment failed: %s", type(e).__name__)

        # --- Decision journal: auto-record as implicit decision context ---
        if obs_type in ("decision", "action"):
            try:
                self.decision_journal.record_decision(
                    title=content[:200],
                    description=content[:500],
                    decision_type=obs_type,
                    chosen="observed",
                    alternatives=["observed"],
                    reasoning="Automatic observation tracking",
                    confidence=importance,
                )
                self.feed_loop.emit("decision_journal", "cognitive", importance,
                                   {"decision_type": obs_type})
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                logger.warning("Journal enrichment failed: %s", type(e).__name__)

        # --- Conflict detection from observations ---
        if any(re.search(r'\b' + re.escape(kw) + r'\b', content.lower()) for kw in ("conflict", "disagree", "argue", "oppose", "object")):
            try:
                self.feed_loop.emit("conflict_resolution", "social", 0.7,
                                   {"content": content[:200], "type": "observed_conflict"})
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("Conflict detection failed: %s", type(e).__name__)

    async def analyze_communication(self, text: str, domain: str = None) -> Dict:
        text = sanitize_input(text, max_length=10000)
        if domain:
            domain = sanitize_input(domain, max_length=50)
        loop = asyncio.get_running_loop()
        thought_result = await loop.run_in_executor(None, self.thought_sense.analyze, text)
        urgency_conf = getattr(thought_result, 'urgency_confidence', None) or 0
        self.recall_synthesis.add_observation(
            text=text, domain=domain, importance=min(1.0, urgency_conf + 0.3)
        )
        recall_result = await loop.run_in_executor(
            None, lambda: self.recall_synthesis.synthesize(query=text, domain=domain)
        )
        thought_dict = thought_result.to_dict() if hasattr(thought_result, 'to_dict') else thought_result
        recall_dict = recall_result.to_dict() if hasattr(recall_result, 'to_dict') else recall_result
        return {
            "thought_analysis": thought_dict,
            "recall_synthesis": recall_dict,
        }

    def get_temporal_patterns(self, event_type: str = None):
        return self.temporal_reasoning.detect_patterns(event_type=event_type)

    def predict_next_event(self, event_type: str):
        return self.temporal_reasoning.predict_next(event_type)

    def get_health(self) -> Dict:
        uptime = time.monotonic() - self._start_time if self._start_time else 0

        if not self._running:
            with self._stats_lock:
                stats = dict(self._stats)
            return {
                "status": "stopped",
                "uptime_seconds": 0,
                "modules": {},
                "stats": stats,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }

        module_health = {}
        for name, check_fn in [
            ("shadow_graph", lambda: self.graph.state.mode.value),
            ("dark_memory", lambda: self.memory.get_stats().get("total_atoms", 0)),
            ("shadow_crew", lambda: len(self.crew.list_agents())),
            ("stealth_layer", lambda: len(self.stealth.get_audit_log())),
            ("personality_engine", lambda: self.personality.get_profile() is not None),
            ("shadow_voice", lambda: self.voice.get_state() is not None),
            ("thought_sense", lambda: self.thought_sense is not None),
            ("recall_synthesis", lambda: self.recall_synthesis is not None),
            ("temporal_reasoning", lambda: self.temporal_reasoning is not None),
            ("empathy_mapping", lambda: self.empathy.get_stats().get("total_persons", 0)),
            ("presence_detection", lambda: (lambda s: s.get("total_devices", 0) if isinstance(s, dict) else 0)(self.presence.get_stats())),
            ("cognitive_trinity", lambda: self.trinity.get_stats()["total_analyses"]),
            ("dream_state", lambda: self.dream.get_stats()["total_memories"]),
            ("skill_tree", lambda: self.skills.get_stats()["total_skills"]),
            ("narrative_memory", lambda: self.narrative.get_stats()["total_events"]),
            ("procedural_memory", lambda: self.procedural.get_stats()["total_procedures"]),
            ("conflict_resolution", lambda: self.conflict.get_stats()["total_conflicts"]),
            ("emotional_intelligence", lambda: self.emotional_intel.get_stats()["stats"]["states_recorded"]),
            ("creative_synthesis", lambda: self.creative.get_stats()["stats"]["ideas_generated"]),
            ("metacognition", lambda: self.metacognition.get_stats()["stats"]["episodes_tracked"]),
            ("moral_reasoning", lambda: self.moral.get_stats()["stats"]["evaluations_made"]),
            ("attention_engine", lambda: self.attention.get_stats()["items_tracked"]),
            ("cognitive_pipeline", lambda: self.pipeline.get_pipeline_stats()["stats"]["pipelines_run"]),
            ("social_intelligence", lambda: self.social_intel.get_stats()["groups_tracked"]),
            ("decision_journal", lambda: self.decision_journal.get_stats()["decisions_recorded"]),
            ("cognitive_reflection", lambda: self.reflection.get_stats()["reflections_added"]),
            ("goal_reasoning", lambda: self.goal_reasoning.get_stats()["assessments_made"]),
            ("device_context", lambda: self.device_context.get_stats()["stats"].get("context_refreshes", 0)),
            ("shadow_arise", lambda: self.arise.get_stats()["active_agents"]),
            ("autonomous_daemon", lambda: self.daemon.get_health()),
            ("daily_briefing", lambda: self.daily is not None),
            ("escalation", lambda: len(self._escalation_manager.get_action_rules()) if self._escalation_manager else 0),
            ("evolution_guide", lambda: self.evolution.get_stats()["observations_added"]),
            ("evolutionary_engine", lambda: self.evolutionary.get_stats()["total_cycles"]),
            ("oracle_bridge", lambda: self.oracle.get_stats()["predictions_generated"]),
            ("feed_loop", lambda: self.feed_loop.get_stats()["signals_emitted"]),
            ("anticipatory", lambda: self.anticipatory.get_stats()["signals_generated"]),
            ("resonance", lambda: self.resonance.get_stats()["activities_logged"]),
            ("temporal_self", lambda: self.temporal_self.get_stats()["milestones_recorded"]),
            ("health_dashboard", lambda: self.health_dashboard.get_stats()["total_checks"]),
            ("creative_dream", lambda: self.creative_dream.get_stats()["cycles_run"]),
            ("social_propagation", lambda: self.social_propagation.get_stats()["ripples_created"]),
            ("neural_bridge", lambda: self.neural_bridge.get_stats()["mappings_created"]),
            ("quantum_field", lambda: self.quantum_field.get_stats()["decisions_created"]),
            ("grammar", lambda: self.grammar.get_stats()["rules_created"]),
            ("holographic", lambda: self.holographic.get_stats()["fragments_created"]),
            ("syncognitive", lambda: self.syncognitive.get_stats()["observations_created"]),
            ("morphic", lambda: self.morphic.get_stats()["patterns_registered"]),
            ("existential", lambda: self.existential.get_stats()["frames_created"]),
            ("chrono", lambda: self.chrono.get_stats()["past_pattern_count"]),
            ("phoenix", lambda: self.phoenix.get_stats()["component_count"]),
            ("causal", lambda: self.causal.get_stats()["edges"]),
            ("quantum_memory", lambda: self.quantum_memory.get_stats()["memories_stored"]),
            ("drift_detector", lambda: self.drift_detector.get_stats()["concepts_count"]),
            ("immune", lambda: self.immune.get_stats()["pathogens_detected"]),
            ("parallax", lambda: self.parallax.get_stats()["events_registered"]),
            ("boundary", lambda: self.boundary.get_stats()["total_rules"] if isinstance(self.boundary.get_stats(), dict) else 0),
            ("resonance_engine", lambda: self.resonance_engine.get_stats().get("patterns_emitted", 0) if isinstance(self.resonance_engine.get_stats(), dict) else 0),
            ("economy", lambda: self.economy.get_stats().get("allocations_requested", 0) if isinstance(self.economy.get_stats(), dict) else 0),
            ("topology", lambda: self.topology.get_stats().get("nodes_added", 0) if isinstance(self.topology.get_stats(), dict) else 0),
            ("symbiosis", lambda: self.symbiosis.get_stats().get("pairs_registered", 0) if isinstance(self.symbiosis.get_stats(), dict) else 0),
            ("shadow_powers", lambda: self.shadow_powers.get_stats()["total_applications"] if isinstance(self.shadow_powers.get_stats(), dict) else 0),
            ("synthesis_engine", lambda: self.synthesis_engine.get_stats()["signals_ingested"]),
            ("self_mod", lambda: self.self_mod.get_stats()["files_snapshotted"]),
            ("profiler", lambda: self.profiler.get_stats()["samples_recorded"]),
            ("intel_report", lambda: self.intel_report.get_stats()["reports_generated"]),
            ("anomaly_corr", lambda: self.anomaly_corr.get_stats()["anomalies_recorded"]),
            ("module_bus", lambda: self.module_bus.get_stats()["events_published"]),
            ("opt_daemon", lambda: self.opt_daemon.get_stats()["tunings_attempted"]),
            ("module_registry", lambda: self.module_registry.get_stats()["modules_registered"]),
            ("vision", lambda: self.vision.get_stats()["stats"]["captures"]),
            ("notifications", lambda: self.notifications.get_stats()["notifications_created"]),
            ("visual_bridge", lambda: self.visual_bridge.get_stats()["events_processed"]),
            ("shadow_deploy", lambda: self.shadow_deploy.get_stats()["shadows_deployed"]),
            ("journal", lambda: self.journal.get_stats()["entries_recorded"]),
        ]:
            try:
                result = check_fn()
                module_health[name] = {"status": "healthy", "detail": str(result)[:100]}
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                module_health[name] = {"status": "unhealthy", "error": type(e).__name__}

        mesh_stats = self.mesh.get_stats() if self._running else {}
        mesh_status = "healthy"
        module_health["neural_mesh"] = {
            "status": mesh_status,
            "peers": mesh_stats.get("peers_count", 0) if isinstance(mesh_stats, dict) else 0,
            "circuit_breakers": mesh_stats.get("circuit_breakers", {}) if isinstance(mesh_stats, dict) else {},
        }

        all_healthy = all(v.get("status") == "healthy" for v in module_health.values())

        with self._stats_lock:
            stats_copy = dict(self._stats)

        return {
            "status": "healthy" if all_healthy else "degraded",
            "uptime_seconds": round(uptime, 1),
            "modules": module_health,
            "stats": stats_copy,
            "timestamp": datetime.now(timezone.utc).isoformat(),
        }

    def get_status(self) -> Dict:
        with self._stats_lock:
            stats_copy = dict(self._stats)
        return {
            "running": self._running,
            "uptime_seconds": round(time.monotonic() - self._start_time, 1) if self._start_time else 0,
            "graph_status": self.graph.get_status(),
            "memory_stats": self.memory.get_stats(),
            "crew_stats": self.crew.get_stats(),
            "mesh_stats": self.mesh.get_stats() if self._running else {},
            "personality_stats": self.personality.get_stats(),
            "thought_sense_stats": self.thought_sense.get_stats(),
            "recall_synthesis_stats": self.recall_synthesis.get_stats(),
            "temporal_reasoning_stats": self.temporal_reasoning.get_stats(),
            "empathy_stats": self.empathy.get_stats(),
            "presence_stats": self.presence.get_stats(),
            "trinity_stats": self.trinity.get_stats(),
            "dream_stats": self.dream.get_stats(),
            "skill_stats": self.skills.get_stats(),
            "narrative_stats": self.narrative.get_stats(),
            "procedural_stats": self.procedural.get_stats(),
            "conflict_stats": self.conflict.get_stats(),
            "emotional_intelligence_stats": self.emotional_intel.get_stats(),
            "creative_synthesis_stats": self.creative.get_stats(),
            "metacognition_stats": self.metacognition.get_stats(),
            "moral_reasoning_stats": self.moral.get_stats(),
            "attention_stats": self.attention.get_stats(),
            "pipeline_stats": self.pipeline.get_pipeline_stats(),
            "social_intelligence_stats": self.social_intel.get_stats(),
            "decision_journal_stats": self.decision_journal.get_stats(),
            "reflection_stats": self.reflection.get_stats(),
            "goal_reasoning_stats": self.goal_reasoning.get_stats(),
            "device_context_stats": self.device_context.get_stats(),
            "arise_stats": self.arise.get_stats(),
            "daemon_status": self.daemon.get_status(),
            "feed_loop_stats": self.feed_loop.get_stats(),
            "feed_dispatch": self.get_feed_dispatch_stats(),
            "anticipatory_stats": self.anticipatory.get_stats(),
            "resonance_stats": self.resonance.get_stats(),
            "temporal_self_stats": self.temporal_self.get_stats(),
            "health_dashboard_stats": self.health_dashboard.get_stats(),
            "creative_dream_stats": self.creative_dream.get_stats(),
            "social_propagation_stats": self.social_propagation.get_stats(),
            "neural_bridge_stats": self.neural_bridge.get_stats(),
            "quantum_field_stats": self.quantum_field.get_stats(),
            "grammar_stats": self.grammar.get_stats(),
            "holographic_stats": self.holographic.get_stats(),
            "syncognitive_stats": self.syncognitive.get_stats(),
            "morphic_stats": self.morphic.get_stats(),
            "existential_stats": self.existential.get_stats(),
            "chrono_stats": self.chrono.get_stats(),
            "phoenix_stats": self.phoenix.get_stats(),
            "causal_stats": self.causal.get_stats(),
            "quantum_memory_stats": self.quantum_memory.get_stats(),
            "drift_detector_stats": self.drift_detector.get_stats(),
            "immune_stats": self.immune.get_stats(),
            "parallax_stats": self.parallax.get_stats(),
            "boundary_stats": self.boundary.get_stats(),
            "resonance_engine_stats": self.resonance_engine.get_stats(),
            "economy_stats": self.economy.get_stats(),
            "topology_stats": self.topology.get_stats(),
            "symbiosis_stats": self.symbiosis.get_stats(),
            "shadow_powers_stats": self.shadow_powers.get_stats(),
            "synthesis_engine_stats": self.synthesis_engine.get_stats(),
            "self_mod_stats": self.self_mod.get_stats(),
            "profiler_stats": self.profiler.get_stats(),
            "intel_report_stats": self.intel_report.get_stats(),
            "anomaly_corr_stats": self.anomaly_corr.get_stats(),
            "module_bus_stats": self.module_bus.get_stats(),
            "opt_daemon_stats": self.opt_daemon.get_stats(),
            "module_registry_stats": self.module_registry.get_stats(),
            "vision_stats": self.vision.get_stats(),
            "notification_stats": self.notifications.get_stats(),
            "visual_bridge_stats": self.visual_bridge.get_stats(),
            "shadow_deploy_stats": self.shadow_deploy.get_stats(),
            "journal_stats": self.journal.get_stats(),
            "stats": stats_copy,
        }

    def get_personality(self) -> Dict:
        return copy.deepcopy(self.personality.get_profile())

    def get_memory_blocks(self) -> List[Dict]:
        return self.memory.list_blocks()

    def get_memory_block(self, label: str) -> Optional[Dict]:
        block = self.memory.get_block(label)
        return block

    def set_memory_block(self, label: str, value: str) -> bool:
        return self.memory.set_block(label, value)

    async def search_memory(self, query: str, limit: int = 10, domain: Optional[str] = None) -> List[Dict]:
        results = await self.memory.search(query, domain=domain, limit=limit)
        return [r.to_dict() for r in results]

    async def get_insights(self, limit: int = 10) -> List[Dict]:
        results = await self.memory.search("", atom_type=AtomType.INSIGHT, limit=limit)
        return [r.to_dict() for r in results]

    def get_audit_log(self, subject: Optional[str] = None, limit: int = 100) -> List[Dict]:
        return self.stealth.get_audit_log(subject, limit)

    def get_graph_visualization(self) -> str:
        return self.graph.get_graph_visualization()

    def get_crew_agents(self) -> List[Dict]:
        return self.crew.list_agents()

    def get_mesh_peers(self) -> List[Dict]:
        return self.mesh.get_peers()

    def add_person(self, name: str, relationship: str = "acquaintance", metadata: Dict = None) -> str:
        return self.empathy.add_person(name, relationship, metadata)

    def get_relationship_health(self, person_id: str) -> Dict:
        health = self.empathy.get_relationship_health(person_id)
        if health is None:
            return {}
        return health.to_dict()

    def analyze_situation(self, situation: str, context: Dict = None) -> Dict:
        situation = sanitize_input(situation, max_length=5000)
        result = self.trinity.analyze(situation, context)
        return result.to_dict()

    def detect_environment(self) -> Dict:
        result = self.presence.detect_environment()
        return result.to_dict()

    def consolidate_memories(self) -> Dict:
        result = self.dream.consolidate()
        return result.to_dict()

    def arise_agent(self, name: str, objective: str, template_id: Optional[str] = None,
                    capabilities: Optional[List] = None, priority: Optional[str] = None,
                    constraints: Optional[List[str]] = None,
                    success_criteria: Optional[List[str]] = None,
                    context_data: Optional[Dict] = None) -> Dict:
        """Create a universal agent for a task via ShadowArise."""
        name = sanitize_input(name, max_length=100)
        objective = sanitize_input(objective, max_length=2000)
        if priority is not None:
            priority = MissionPriority(priority)
        else:
            priority = MissionPriority.MEDIUM
        caps = [CapabilityType(c) for c in (capabilities or [])]
        agent = self.arise.arise(
            name=name, objective=objective, template_id=template_id,
            capabilities=caps or None, priority=priority,
            constraints=constraints, success_criteria=success_criteria,
            context_data=context_data,
        )
        return agent.to_dict()

    def execute_agent_step(self, agent_id: str, description: str,
                           capability: str, input_data: Optional[Dict] = None,
                           confidence: float = 0.5, duration_ms: int = 0) -> Dict:
        """Record an execution step for an active agent."""
        agent_id = sanitize_input(agent_id, max_length=12)
        description = sanitize_input(description, max_length=1000)
        step = self.arise.execute_step(
            agent_id=agent_id, description=description,
            capability_used=CapabilityType(capability),
            input_data=input_data, confidence=confidence,
            duration_ms=duration_ms,
        )
        return step.to_dict() if step else {"status": "error", "reason": "agent not found or not active"}

    def evolve_agent(self, agent_id: str, feedback: str,
                     adjustments: Optional[Dict[str, float]] = None,
                     new_capabilities: Optional[List[str]] = None) -> Dict:
        """Evolve an agent based on feedback."""
        agent_id = sanitize_input(agent_id, max_length=12)
        feedback = sanitize_input(feedback, max_length=2000)
        new_caps = [CapabilityType(c) for c in (new_capabilities or [])]
        agent = self.arise.evolve(agent_id, feedback=feedback,
                                  capability_adjustments=adjustments,
                                  new_capabilities=new_caps or None)
        return agent.to_dict() if agent else {"status": "error", "reason": "agent not found"}

    def get_arise_stats(self) -> Dict:
        """Get ShadowArise engine statistics."""
        return self.arise.get_stats()

    def shadow_powers_on_agent_created(self, agent_name: str, template: str) -> Optional[Dict]:
        """Delegate to ShadowPowers.on_agent_created when an agent arises."""
        agent_name = sanitize_input(agent_name, max_length=100)
        template = sanitize_input(template, max_length=100)
        result = self.shadow_powers.on_agent_created(agent_name, template)
        return result.to_dict() if result else None

    def shadow_powers_on_skill_leveled(self, skill_id: str, domain: str, new_level: str) -> Optional[Dict]:
        """Delegate to ShadowPowers.on_skill_leveled when a skill levels up."""
        skill_id = sanitize_input(skill_id, max_length=64)
        domain = sanitize_input(domain, max_length=50)
        new_level = sanitize_input(new_level, max_length=50)
        result = self.shadow_powers.on_skill_leveled(skill_id, domain, new_level)
        return result.to_dict() if result else None

    def shadow_powers_on_resonance(self, signal_type: str, source: str, strength: float) -> Optional[Dict]:
        """Delegate to ShadowPowers.on_resonance_event when a resonance fires."""
        signal_type = sanitize_input(signal_type, max_length=30)
        source = sanitize_input(source, max_length=50)
        strength = max(0.0, min(1.0, float(strength)))
        result = self.shadow_powers.on_resonance_event(signal_type, source, strength)
        return result.to_dict() if result else None

    def summon_voice(self, transcript: str, context: Optional[Dict] = None) -> Dict:
        """Process a voice transcript through ShadowSummon."""
        transcript = sanitize_input(transcript, max_length=2000)
        response = self.summon.process_voice(transcript, context)
        intent = response["intent"]

        if intent == "create_agent":
            template = response["result"].get("template")
            return {
                **response,
                "action_type": "arise_agent",
                "template_id": template,
                "speak": response["spoken"],
            }
        elif intent == "invoke_ability":
            ability = response["result"].get("ability")
            return {
                **response,
                "action_type": "invoke_ability",
                "ability_id": ability,
                "speak": response["spoken"],
            }
        elif intent == "ask_question":
            return {
                **response,
                "action_type": "ask",
                "speak": response["spoken"],
            }
        elif intent == "observe_event":
            return {
                **response,
                "action_type": "observe",
                "speak": response["spoken"],
            }
        else:
            return {
                **response,
                "action_type": intent,
                "speak": response["spoken"],
            }

    def get_summon_abilities(self) -> List[Dict]:
        """Get list of Shikigami abilities with voice invocation phrases."""
        return self.summon.get_abilities()

    def get_summon_templates(self) -> List[Dict]:
        """Get list of agent templates with voice invocation phrases."""
        return self.summon.get_agent_templates()

    def get_summon_history(self, limit: int = 50) -> List[Dict]:
        """Get recent summon history."""
        return self.summon.get_history(limit)

    def get_daemon_status(self) -> Dict:
        return self.daemon.get_status()

    def get_daemon_observations(self, limit: int = 50) -> List[Dict]:
        return self.daemon.get_observations(limit)

    def get_daemon_decisions(self, limit: int = 50) -> List[Dict]:
        return self.daemon.get_decisions(limit)

    def get_security_events(self, limit: int = 50) -> List[Dict]:
        return self.daemon.get_security_events(limit)

    def daemon_add_watch(self, path: str) -> bool:
        return self.daemon.add_watch_path(path)

    def daemon_remove_watch(self, path: str) -> bool:
        return self.daemon.remove_watch_path(path)

    def daemon_force_observe(self, obs_type: str = "system_metrics") -> Dict:
        try:
            ot = ObservationType(obs_type)
        except ValueError:
            ot = ObservationType.SYSTEM_METRICS
        return self.daemon.force_observe(ot)

    def daemon_kill_switch(self) -> Dict:
        return self.daemon.engage_kill_switch()

    def daemon_release_kill_switch(self) -> Dict:
        return self.daemon.disengage_kill_switch()

    def ingest_user_vitals(self, sample: RhythmSample) -> None:
        self.daemon.ingest_user_vitals(sample)

    def get_user_rhythm_baselines(self) -> Dict:
        return self.daemon.user_rhythm.get_baselines()

    def get_user_activity_state(self) -> UserActivityState:
        return self.daemon.get_user_state()

    def sync_evolution_from_device(self) -> List[str]:
        """Feed DeviceContext data into EvolutionGuide for domain observations."""
        synced = []
        try:
            ctx = self.device_context
            if ctx and hasattr(ctx, 'get_context'):
                self.evolution.ingest_device_context(ctx.get_context())
                synced.append("device_context")
        except (AttributeError, RuntimeError, ValueError):
            pass
        return synced

    def sync_oracle_from_cognition(self) -> List[str]:
        """Feed TemporalReasoning and AnomalyDetector data into OracleBridge."""
        synced = []
        try:
            tr = self.temporal_reasoning
            if tr and hasattr(tr, 'detect_patterns'):
                self.oracle.ingest_temporal_data(tr)
                synced.append("temporal_reasoning")
        except (AttributeError, RuntimeError, ValueError):
            pass
        try:
            ad = self.anomaly_detector
            if ad and hasattr(ad, 'get_recent_anomalies'):
                self.oracle.ingest_anomalies(ad)
                synced.append("anomaly_detector")
        except (AttributeError, RuntimeError, ValueError):
            pass
        return synced

    def get_evolution_briefing_items(self, limit: int = 5) -> List[Dict]:
        """Get top evolution insights for DailyBriefing integration."""
        return self.evolution.get_briefing_evolution_items(limit=limit)

    def run_creative_dream_cycle(self) -> Dict:
        return self.creative_dream.run_cycle().to_dict()

    def get_dream_inspirations(self, limit: int = 20) -> List[Dict]:
        return self.creative_dream.get_inspirations(limit)

    def get_creativity_incubations(self, limit: int = 20) -> List[Dict]:
        return self.creative_dream.get_incubations(limit)

    def generate_daily_briefing(self) -> Dict:
        """Generate a daily intelligence briefing.
        Uses real data from all cognitive modules.
        """
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                future = asyncio.run_coroutine_threadsafe(
                    self.daily.generate_briefing(
                        device_context=self.device_context,
                        evolution_guide=self.evolution,
                    ), loop
                )
                briefing = future.result(timeout=15)
            else:
                briefing = asyncio.run(self.daily.generate_briefing(
                    device_context=self.device_context,
                    evolution_guide=self.evolution,
                ))
            return briefing.to_dict()
        except (RuntimeError, ValueError, TypeError, AttributeError,
                OSError, concurrent.futures.TimeoutError) as e:
            logger.warning("Briefing generation failed: %s", type(e).__name__)
            return {"error": "Briefing unavailable"}

    def push_notification(self, title: str, body: str, priority: str = "normal",
                          source: str = "shadow_os", category: str = "general") -> str:
        """Create and push a notification through all enabled channels."""
        return self.notifications.notify(title=title, body=body, priority=priority,
                                         source_module=source, category=category)

    def run_predictive_analysis(self) -> List[Dict]:
        """Run predictive analysis across all modules, returning preemptive alerts."""
        alerts = self.predictive.analyze_and_predict()
        return [a.to_dict() for a in alerts]

    def get_full_dashboard(self, force: bool = True) -> Dict:
        """Get a complete dashboard snapshot with live data from all 70+ modules.
        Optimized for PWA polling - uses caching and parallel data collection.
        """
        return self.dashboard_agg.get_snapshot(force=force).to_dict()


    def journal_auto_anomaly(self, description: str, count: int = 1) -> str:
        """Auto-record anomaly in journal (called by daemon handlers)."""
        return self.journal.record(
            "alert", description, category="security", importance=0.8,
            tags=["anomaly","automated","daemon"]
        )

    def journal_auto_security(self, description: str, threat_level: int) -> str:
        """Auto-record security events in journal."""
        importance = 0.9 if threat_level >= 3 else 0.7 if threat_level >= 2 else 0.5
        return self.journal.record(
            "alert", description, category="security", importance=importance,
            tags=["security","automated","daemon"]
        )

    def journal_auto_decision(self, title: str, chosen: str, importance: float = 0.7) -> str:
        """Auto-record decision outcomes."""
        return self.journal.record(
            "decision", f"Decision: {title} -> {chosen}", category="work",
            importance=importance, tags=["decision","automated"]
        )

    def journal_auto_observation(self, content: str, category: str = "system",
                                  importance: float = 0.5) -> str:
        """Auto-record system observations."""
        return self.journal.record(
            "observation", content, category=category, importance=importance,
            tags=["automated","system"]
        )

    def journal_record(self, entry_type: str, content: str, category: str = "general",
                     importance: float = 0.5, tags: Optional[List[str]] = None) -> str:
        """Record an entry in the daily activity journal."""
        return self.journal.record(entry_type=entry_type, content=content,
                                   category=category, importance=importance, tags=tags)

    def journal_daily_summary(self, date: Optional[str] = None) -> Dict:
        """Generate a summary of today's journal entries."""
        return self.journal.generate_daily_summary(date=date).to_dict()

    def journal_search(self, query: str, limit: int = 20) -> List[Dict]:
        """Search the journal for entries matching a query."""
        return self.journal.search(query=query, limit=limit)

    def provision_shadow(self, name: str, device_id: str = "", mode: str = "sentry",
                       location: str = "") -> Dict:
        """Deploy a shadow to a remote device (e.g. Android phone)."""
        shadow = self.shadow_deploy.provision_shadow(
            name=name, device_id=device_id, mode=mode, location=location
        )
        return shadow.to_dict()

    def get_deployed_shadows(self) -> List[Dict]:
        """List all deployed shadow instances."""
        return self.shadow_deploy.list_shadows()

    def process_visual_for_cognition(self, description: str, scene_category: str = "",
                                  objects: Optional[List[str]] = None, faces: int = 0,
                                  has_text: bool = False, importance: float = 0.5) -> Dict:
        """Route a visual observation through all cognitive modules."""
        return self.visual_bridge.process_visual_observation(
            description=description, scene_category=scene_category,
            objects=objects, faces=faces, has_text=has_text, importance=importance,
        ).to_dict()


    def push_critical_alert(self, title: str, body: str, source: str = "shadow_os",
                            category: str = "security") -> str:
        """Push a CRITICAL-priority alert (shown immediately in PWA)."""
        return self.notifications.notify_critical(title=title, body=body,
                                                  source_module=source, category=category)

    def get_unread_notifications(self, limit: int = 20) -> List[Dict]:
        """Get latest unread notifications."""
        return self.notifications.get_notifications(unread_only=True, limit=limit)

    def get_notification_digest(self) -> Dict:
        """Generate and return a notification digest summary."""
        return self.notifications.generate_digest().to_dict()

    def get_notification_count(self) -> int:
        """Get total unread notification count."""
        return self.notifications.get_unread_count()


    def analyze_visual(self, image_source=None, description: str = "", source: str = "api") -> Dict:
        """Analyze an image and store in visual memory."""
        result = self.vision.capture_and_analyze(image_source, description=description, source=source)
        return result.to_dict()

    def search_visual_memory(self, query: str, limit: int = 10) -> List[Dict]:
        """Search visual memory by description or tags."""
        return self.vision.search_memory(query=query, limit=limit)

    def get_recent_visuals(self, limit: int = 10) -> List[Dict]:
        """Get recent visual memories."""
        return self.vision.get_recent(limit=limit)

    def propagate_social_ripple(self, source: str, ripple_type: str, strength: float,
                                reason: str, person_id: str = None, group_id: str = None,
                                conflict_id: str = None, metadata: Dict = None) -> Dict:
        ripple = self.social_propagation.create_ripple(
            source=source, ripple_type=ripple_type, strength=strength, reason=reason,
            person_id=person_id, group_id=group_id, conflict_id=conflict_id, metadata=metadata,
        )
        effects = self.social_propagation.propagate(
            ripple,
            social_intel=self.social_intel,
            empathy=self.empathy,
            conflict=self.conflict,
        )
        return {
            "ripple": ripple.to_dict(),
            "effects": [e.to_dict() for e in effects],
            "effect_count": len(effects),
        }

    def build_social_graph(self) -> Dict:
        return self.social_propagation.build_graph(
            social_intel=self.social_intel,
            empathy=self.empathy,
            conflict=self.conflict,
        )

    def get_social_propagation_summary(self) -> Dict:
        return self.social_propagation.get_propagation_summary()

    def refresh_health_dashboard(self) -> Dict:
        modules_map = {
            "shadow_graph": self.graph,
            "dark_memory": self.memory,
            "shadow_crew": self.crew,
            "neural_mesh": self.mesh,
            "stealth_layer": self.stealth,
            "personality": self.personality,
            "shadow_voice": self.voice,
            "anomaly_detector": self.anomaly_detector,
            "goal_planner": self.goal_planner,
            "thought_sense": self.thought_sense,
            "recall_synthesis": self.recall_synthesis,
            "temporal_reasoning": self.temporal_reasoning,
            "empathy": self.empathy,
            "presence": self.presence,
            "cognitive_trinity": self.trinity,
            "dream_state": self.dream,
            "skills": self.skills,
            "narrative": self.narrative,
            "procedural": self.procedural,
            "conflict_resolution": self.conflict,
            "emotional_intelligence": self.emotional_intel,
            "creative_synthesis": self.creative,
            "metacognition": self.metacognition,
            "moral_reasoning": self.moral,
            "attention": self.attention,
            "cognitive_pipeline": self.pipeline,
            "social_intelligence": self.social_intel,
            "decision_journal": self.decision_journal,
            "cognitive_reflection": self.reflection,
            "goal_reasoning": self.goal_reasoning,
            "device_context": self.device_context,
            "shadow_arise": self.arise,
            "shadow_summon": self.summon,
            "evolution_guide": self.evolution,
            "evolutionary_engine": self.evolutionary,
            "oracle_bridge": self.oracle,
            "feed_loop": self.feed_loop,
            "anticipatory": self.anticipatory,
            "resonance": self.resonance,
            "temporal_self": self.temporal_self,
            "creative_dream": self.creative_dream,
            "social_propagation": self.social_propagation,
            "neural_bridge": self.neural_bridge,
            "quantum_field": self.quantum_field,
            "grammar": self.grammar,
            "holographic": self.holographic,
            "syncognitive": self.syncognitive,
            "morphic": self.morphic,
            "existential": self.existential,
            "chrono": self.chrono,
            "phoenix": self.phoenix,
            "causal": self.causal,
            "quantum_memory": self.quantum_memory,
            "drift_detector": self.drift_detector,
            "immune": self.immune,
            "parallax": self.parallax,
            "boundary": self.boundary,
            "resonance_engine": self.resonance_engine,
            "economy": self.economy,
            "topology": self.topology,
            "symbiosis": self.symbiosis,
            "synthesis_engine": self.synthesis_engine,
            "self_mod": self.self_mod,
            "visualizer": self.visualizer,
            "profiler": self.profiler,
            "intel_report": self.intel_report,
            "anomaly_corr": self.anomaly_corr,
            "module_bus": self.module_bus,
        }
        return self.health_dashboard.auto_update(modules_map)

    def bridge_neuro_to_symbolic(self, pattern_id: str) -> Dict:
        """Translate neural patterns to symbolic rules via NeuralSymbolicBridge,
        then feed results into existing ReasoningModule/DecisionJournal."""
        pattern_id = sanitize_input(pattern_id, max_length=200)
        try:
            rules = self.neural_bridge.translate_neuro_to_symbolic(pattern_id)
            rule_dicts = [r.to_dict() if hasattr(r, 'to_dict') else r for r in rules]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("NeuralBridge translate failed: %s", type(e).__name__)
            rules = []
            rule_dicts = []

        for rule_dict in rule_dicts[:5]:
            try:
                conclusion = rule_dict.get("conclusion", "")[:200] if isinstance(rule_dict, dict) else str(rule_dict)[:200]
                self.decision_journal.record_decision(
                    title=f"Neuro-symbolic rule: {conclusion}",
                    description=f"Translated from neural pattern {pattern_id}",
                    decision_type="neuro_symbolic",
                    chosen="apply",
                    alternatives=["defer", "reject"],
                    reasoning="Auto-generated symbolic rule from neural pattern translation",
                    confidence=min(1.0, rule_dict.get("confidence", 0.5)) if isinstance(rule_dict, dict) else 0.5,
                )
            except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
                logger.warning("Journal recording for neuro-symbolic rule failed: %s", type(e).__name__)

        self._inc_stat("insights_generated")
        return {
            "pattern_id": pattern_id,
            "rules_generated": len(rule_dicts),
            "rules": rule_dicts,
        }

    def explore_decision_space(self, context: str, options: Optional[List[Dict]] = None) -> Dict:
        """Use QuantumDecisionField to explore superposition of decision options,
        integrating with PersonalityEngine and CognitiveTrinity."""
        context = sanitize_input(context, max_length=2000)
        if options is None:
            options = [{"name": "option_a", "reward": 0.5}, {"name": "option_b", "reward": 0.5}]

        try:
            decision = self.quantum_field.create_decision(options=options, context=context)
            decision_id = decision.decision_id if hasattr(decision, 'decision_id') else ""
            prob_field = self.quantum_field.get_probability_field(decision_id) if decision_id else {}
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("QuantumDecisionField create failed: %s", type(e).__name__)
            return {"status": "error", "error": type(e).__name__}

        try:
            personality_pred = self.personality.predict_decision(
                options,
                context={"quantum_context": context[:200]},
            )
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Personality prediction in quantum failed: %s", type(e).__name__)
            personality_pred = None

        try:
            trinity_result = self.trinity.analyze(situation=context, context={"quantum": True})
            trinity_consensus = trinity_result.synthesis if hasattr(trinity_result, 'synthesis') else {}
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Trinity analysis in quantum failed: %s", type(e).__name__)
            trinity_consensus = {}

        try:
            collapsed = self.quantum_field.collapse(decision_id) if decision_id else None
            chosen = collapsed.chosen_option if hasattr(collapsed, 'chosen_option') else None
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("Quantum collapse failed: %s", type(e).__name__)
            chosen = None

        return {
            "decision_id": decision_id,
            "probability_field": prob_field if isinstance(prob_field, dict) else {},
            "personality_prediction": personality_pred,
            "trinity_consensus": trinity_consensus if isinstance(trinity_consensus, dict) else {},
            "chosen": chosen.to_dict() if hasattr(chosen, 'to_dict') else (str(chosen) if chosen else None),
        }

    async def evolve_grammar(self, text: str) -> Dict:
        """Feed text through EmergentGrammar to evolve cognitive language,
        store discovered rules in DarkMemory."""
        text = sanitize_input(text, max_length=5000)
        if not text:
            return {"rules_created": 0, "rules": []}

        try:
            rule = self.grammar.observe(pattern=text, context="cognitive_evolution")
            rule_dict = rule.to_dict() if hasattr(rule, 'to_dict') else str(rule)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("EmergentGrammar observe failed: %s", type(e).__name__)
            return {"rules_created": 0, "rules": []}

        atom_id = ""
        try:
            stored = await self.memory.store(
                content=f"Grammar rule: {str(rule_dict)[:500]}",
                atom_type=AtomType.PATTERN,
                domain="language",
                importance=0.6,
                metadata={"source": "emergent_grammar", "rule_id": getattr(rule, 'rule_id', '')},
            )
            atom_id = stored.atom_id if hasattr(stored, 'atom_id') else ""
        except (RuntimeError, ValueError, AttributeError, OSError):
            pass

        try:
            all_rules = self.grammar.list_sentences()
            rule_count = len(all_rules) if all_rules else 0
        except (ValueError, TypeError, AttributeError, RuntimeError):
            rule_count = 0

        return {
            "rules_created": 1,
            "total_rules": rule_count,
            "new_rule": rule_dict,
            "memory_atom_id": atom_id,
        }

    def holographic_recall(self, query: str, min_quality: float = 0.5) -> Dict:
        """Encode query in HolographicMemory, reconstruct from fragments,
        return with quality assessment."""
        query = sanitize_input(query, max_length=2000)
        min_quality = max(0.0, min(1.0, min_quality))

        try:
            fragment = self.holographic.encode(content=query, metadata={"recall_query": True})
            fragment_id = fragment.fragment_id if hasattr(fragment, 'fragment_id') else ""
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("HolographicMemory encode failed: %s", type(e).__name__)
            return {"status": "error", "error": type(e).__name__}

        try:
            existing = self.holographic.find_fragments(query)
            fragment_ids = [f.fragment_id for f in existing if hasattr(f, 'fragment_id')]
            if fragment_id:
                fragment_ids.insert(0, fragment_id)
            fragment_ids = fragment_ids[:20]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("HolographicMemory find_fragments failed: %s", type(e).__name__)
            fragment_ids = [fragment_id] if fragment_id else []

        if len(fragment_ids) >= 2:
            try:
                result = self.holographic.partial_reconstruct(fragment_ids, min_quality=min_quality)
                if result:
                    reconstruction = result.to_dict() if hasattr(result, 'to_dict') else str(result)
                    quality = getattr(result, 'quality_score', 0.0)
                else:
                    reconstruction = None
                    quality = 0.0
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("HolographicMemory reconstruct failed: %s", type(e).__name__)
                reconstruction = None
                quality = 0.0
        else:
            reconstruction = None
            quality = 0.0

        try:
            integrity = self.holographic.verify_integrity(fragment_ids) if fragment_ids else None
        except (ValueError, TypeError, AttributeError, RuntimeError):
            integrity = None

        return {
            "query": query,
            "fragments_found": len(fragment_ids),
            "reconstruction": reconstruction,
            "quality_score": round(quality, 3),
            "integrity": integrity,
            "min_quality": min_quality,
        }

    def self_correct(self, process_name: str, details: str = "") -> Dict:
        """Use SyncognitiveFeedback to detect bias and self-correct,
        integrating with MetacognitionEngine."""
        process_name = sanitize_input(process_name, max_length=200)
        details = sanitize_input(details, max_length=2000)

        try:
            biases = self.syncognitive.detect_bias(process_name)
            bias_dicts = [b.to_dict() if hasattr(b, 'to_dict') else b for b in biases]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("SyncognitiveFeedback detect_bias failed: %s", type(e).__name__)
            biases = []
            bias_dicts = []

        corrections_applied = []
        for bias in biases[:5]:
            bias_name = getattr(bias, 'observation_type', 'unknown') if hasattr(bias, 'observation_type') else 'unknown'
            correction_text = f"Detected {bias_name} in {process_name}: adjusting confidence downward"
            confidence_delta = -0.1
            try:
                obs = self.syncognitive.correct_self(
                    process_name=process_name,
                    correction=correction_text,
                    confidence_delta=confidence_delta,
                )
                corrections_applied.append(obs.to_dict() if hasattr(obs, 'to_dict') else str(obs))
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("SyncognitiveFeedback correct_self failed: %s", type(e).__name__)

        try:
            bias_names = [getattr(b, 'observation_type', 'unknown') for b in biases]
            meta_result = self.metacognition.start_thinking(
                reasoning_type=ReasoningType.SYSTEMATIC,
                topic=f"Self-correction for {process_name}",
                cognitive_load=0.3 + len(biases) * 0.1,
            )
            meta_info = {
                "biases_found": bias_names,
                "cognitive_load": round(getattr(meta_result, 'cognitive_load', 0.5), 3),
            }
        except (ValueError, TypeError, AttributeError, RuntimeError, ImportError) as e:
            logger.warning("MetacognitionEngine integration failed: %s", type(e).__name__)
            meta_info = {}

        return {
            "process_name": process_name,
            "biases_detected": len(bias_dicts),
            "biases": bias_dicts,
            "corrections_applied": len(corrections_applied),
            "correction_details": corrections_applied,
            "metacognition": meta_info,
        }

    def propagate_resonance(self, domain: str, threshold: float = 0.3) -> Dict:
        """Propagate patterns via MorphicResonanceEngine across domains,
        feed insights to EvolutionGuide."""
        domain = sanitize_input(domain, max_length=100)
        threshold = max(0.0, min(1.0, threshold))

        try:
            patterns = self.morphic.detect_resonance(domain=domain, pattern="")
            pattern_dicts = [p.to_dict() if hasattr(p, 'to_dict') else p for p in patterns]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("MorphicResonanceEngine detect failed: %s", type(e).__name__)
            patterns = []
            pattern_dicts = []

        try:
            all_domains = ["work", "health", "creativity", "social", "learning", "financial", "emotional", "productivity"]
            target_domains = [d for d in all_domains if d != domain]
            all_influences = []
            for target in target_domains:
                influences = self.morphic.propagate(
                    source_domain=domain, target_domain=target, min_confidence=threshold,
                )
                all_influences.extend(influences)
            influence_dicts = [i.to_dict() if hasattr(i, 'to_dict') else i for i in all_influences[:20]]
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("MorphicResonanceEngine propagate failed: %s", type(e).__name__)
            influence_dicts = []

        for influence in influence_dicts[:5]:
            try:
                target = influence.get("target_domain", "unknown") if isinstance(influence, dict) else "unknown"
                strength = influence.get("strength", 0.0) if isinstance(influence, dict) else 0.0
                if strength > threshold:
                    self.evolution.observe(
                        observation_type="pattern",
                        content=f"Morphic resonance: {domain} -> {target} (strength={round(strength, 3)})",
                        domain=target,
                    )
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("EvolutionGuide morphic integration failed: %s", type(e).__name__)

        return {
            "source_domain": domain,
            "resonance_patterns": len(pattern_dicts),
            "patterns": pattern_dicts[:10],
            "influences_propagated": len(influence_dicts),
            "influences": influence_dicts[:10],
            "threshold": threshold,
        }

    def evaluate_purpose(self, event_id: str, description: str) -> Dict:
        """Evaluate significance via ExistentialCalculus,
        update GoalReasoning with purpose alignment."""
        event_id = sanitize_input(event_id, max_length=200)
        description = sanitize_input(description, max_length=2000)

        try:
            result = self.existential.evaluate_significance(event_id=event_id, description=description)
            result_dict = result.to_dict() if hasattr(result, 'to_dict') else str(result)
            significance = getattr(result, 'significance_score', 0.5)
            purpose_alignment = getattr(result, 'purpose_alignment', 0.0)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("ExistentialCalculus evaluate failed: %s", type(e).__name__)
            result_dict = {}
            significance = 0.0
            purpose_alignment = 0.0

        if significance > 0.5:
            try:
                goals = self.goal_planner.list_goals(limit=3)
                active = [g for g in goals if g.get("status") in ("in_progress", "pending")]
                if active:
                    gid = active[0].get("goal_id", "")
                    if gid:
                        assessment = self.goal_reasoning.assess_goal(
                            goal_id=gid,
                            progress=purpose_alignment,
                            context_description=description[:500],
                        )
                        assessment_info = {
                            "urgency": getattr(assessment, 'urgency_score', 0.0),
                            "confidence": getattr(assessment, 'confidence', 0.0),
                        } if assessment else {}
                    else:
                        assessment_info = {}
                else:
                    assessment_info = {}
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("GoalReasoning purpose integration failed: %s", type(e).__name__)
                assessment_info = {}
        else:
            assessment_info = {}

        try:
            themes = self.existential.get_themes()
            theme_list = list(themes.keys())[:5] if isinstance(themes, dict) else []
        except (ValueError, TypeError, AttributeError, RuntimeError):
            theme_list = []

        return {
            "event_id": event_id,
            "significance_score": round(significance, 3),
            "purpose_alignment": round(purpose_alignment, 3),
            "result": result_dict,
            "life_themes": theme_list,
            "goal_assessment": assessment_info,
        }

    def synthesize_temporal(self, context_id: str) -> Dict:
        """Synthesize past/future via ChronoSynthesis,
        feed into DecisionJournal."""
        context_id = sanitize_input(context_id, max_length=200)

        try:
            decision = self.chrono.synthesize(context_id=context_id)
            decision_dict = decision.to_dict() if hasattr(decision, 'to_dict') else str(decision)
            chosen = getattr(decision, 'chosen_option', None)
            confidence = getattr(decision, 'confidence', 0.5)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("ChronoSynthesis synthesize failed: %s", type(e).__name__)
            decision_dict = {}
            chosen = None
            confidence = 0.5

        try:
            journal_id = self.decision_journal.record_decision(
                title=f"Temporal synthesis: {context_id[:100]}",
                description=f"Past-future synthesis for context {context_id}",
                decision_type="temporal_synthesis",
                chosen=chosen if isinstance(chosen, str) else "balanced",
                alternatives=["past_weighted", "future_weighted"],
                reasoning="ChronoSynthesis past-future integration",
                confidence=confidence,
            )
        except (ValueError, TypeError, AttributeError, RuntimeError, KeyError) as e:
            logger.warning("DecisionJournal temporal integration failed: %s", type(e).__name__)
            journal_id = None

        try:
            past_patterns = self.chrono.list_past_patterns()
            future_projections = self.chrono.list_future_projections()
            past_count = len(past_patterns) if past_patterns else 0
            future_count = len(future_projections) if future_projections else 0
        except (ValueError, TypeError, AttributeError, RuntimeError):
            past_count = 0
            future_count = 0

        return {
            "context_id": context_id,
            "synthesis": decision_dict,
            "past_patterns_available": past_count,
            "future_projections_available": future_count,
            "journal_decision_id": journal_id,
        }

    def regenerate_component(self, component_id: str) -> Dict:
        """Full regeneration cycle via PhoenixRegenerator:
        diagnose -> plan -> execute -> preserve learning."""
        component_id = sanitize_input(component_id, max_length=200)

        try:
            health = self.phoenix.check_health(component_id)
            if health and hasattr(health, 'health_score') and health.health_score >= 0.8:
                return {
                    "component_id": component_id,
                    "status": "healthy",
                    "health_score": round(health.health_score, 3),
                    "action": "none_needed",
                }
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("PhoenixRegenerator check_health failed: %s", type(e).__name__)

        diagnosis = None
        try:
            diagnosis = self.phoenix.diagnose(component_id)
            diag_dict = diagnosis.to_dict() if hasattr(diagnosis, 'to_dict') else str(diagnosis)
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("PhoenixRegenerator diagnose failed: %s", type(e).__name__)
            diag_dict = None

        plan = None
        plan_dict = None
        plan_id = None
        if diagnosis:
            try:
                plan = self.phoenix.create_regeneration_plan(component_id)
                if plan:
                    plan_dict = plan.to_dict() if hasattr(plan, 'to_dict') else str(plan)
                    plan_id = getattr(plan, 'plan_id', None)
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("PhoenixRegenerator create_plan failed: %s", type(e).__name__)

        result = None
        result_dict = None
        if plan and plan_id:
            try:
                regen_result = self.phoenix.execute_regeneration(plan_id)
                if regen_result:
                    result_dict = regen_result.to_dict() if hasattr(regen_result, 'to_dict') else str(regen_result)
                    result = regen_result
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.warning("PhoenixRegenerator execute failed: %s", type(e).__name__)

        learning = {}
        try:
            preserved = self.phoenix.preserve_learning(component_id)
            if preserved:
                learning = preserved if isinstance(preserved, dict) else {"preserved": True}
        except (ValueError, TypeError, AttributeError, RuntimeError) as e:
            logger.warning("PhoenixRegenerator preserve_learning failed: %s", type(e).__name__)

        return {
            "component_id": component_id,
            "diagnosis": diag_dict,
            "plan": plan_dict,
            "execution_result": result_dict,
            "preserved_learning": learning,
        }