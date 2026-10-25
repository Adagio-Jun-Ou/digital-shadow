#!/usr/bin/env python3
"""
ShadowSummon — Voice Summon Router

Bridges voice input (STT) to cognitive actions:
  - Natural language intent parsing → ShadowArise agent creation
  - Shikigami ability invocation via voice
  - ShadowOS cognitive commands via voice
  - Real-time streaming responses (TTS-ready)

Architecture:
  ShadowSummon
  ├── IntentRouter (voice text → action mapping)
  ├── SummonPhrases (wake words and command patterns)
  ├── ShikigamiBridge (maps voice to 10 Shikigami abilities)
  └── AgentSummoner (maps voice to ShadowArise agents)

Usage:
  POST /os/summon/voice   — Process a voice transcript
  POST /os/summon/text    — Process a text command
  GET  /os/summon/agents  — List active agents
  GET  /os/summon/abilities — List available Shikigami abilities
  POST /os/summon/ability  — Invoke a Shikigami ability by name
"""

import copy
import re
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass
from enum import Enum
from typing import Dict, List, Optional
from collections import deque

logger = logging.getLogger("shadowd.cognition.shadow_summon")


_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 500) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 500) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


class SummonIntent(str, Enum):
    CREATE_AGENT = "create_agent"
    INVOKE_ABILITY = "invoke_ability"
    ASK_QUESTION = "ask_question"
    OBSERVE_EVENT = "observe_event"
    GET_STATUS = "get_status"
    LIST_AGENTS = "list_agents"
    LIST_ABILITIES = "list_abilities"
    COMPLETE_AGENT = "complete_agent"
    EVOLVE_AGENT = "evolve_agent"
    UNKNOWN = "unknown"


class SummonResponse(dict):
    """Structured response from a summon command."""

    def __init__(self, intent: SummonIntent, action: str, result: Dict,
                 spoken: str, confidence: float = 1.0):
        super().__init__({
            "intent": intent.value,
            "action": action,
            "result": result,
            "spoken": spoken,
            "confidence": round(confidence, 3),
            "timestamp": datetime.now(timezone.utc).isoformat(),
        })


_ABILITY_MAP = {
    "oracle": ["predict", "prediction", "forecast", "what will happen", "what will be",
                "see the future", "likely outcome", "probability of"],
    "x_ray": ["scan nearby", "detect devices", "bluetooth scan", "wifi scan",
              "who is around", "whos nearby", "nearby devices"],
    "life_pulse": ["track person", "where is someone", "relationship status",
                   "track someone", "person tracker"],
    "perfect_memory": ["remember this", "recall what", "what did i", "when did i",
                       "forget that", "memory recall", "remind me"],
    "pattern_sight": ["find pattern", "find patterns", "see pattern", "see patterns", "detect trend", "detect trends", "correlation",
                      "recurring pattern", "cyclical", "pattern in my data",
                      "patterns in", "trends in", "pattern in", "habit pattern"],
    "social_radar": ["social dynamic", "group behavior", "interaction pattern",
                      "social radar", "people dynamics"],
    "intent_seer": ["predict intent", "what will they do", "next action",
                     "what are they planning", "intent seer", "going to do next"],
    "omni_sense": ["environment overview", "full context", "complete summary",
                   "comprehensive status", "omni sense", "everything happening"],
    "shadow_council": ["should i choose", "advise me", "counsel on",
                       "recommend between", "help me decide", "which option",
                       "council advice", "should i take"],
    "reality_forge": ["build this", "set up a", "implement this", "deploy a",
                       "create and run", "forge a", "make it happen"],
}

_AGENT_TEMPLATES = {
    "analyst": ["analyze", "research", "investigate", "study", "examine", "evaluate", "deep dive"],
    "creator": ["create", "design", "write", "generate", "ideate", "compose", "brainstorm"],
    "planner": ["plan", "organize", "schedule", "coordinate", "strategy", "roadmap", "timeline"],
    "executor": ["execute", "implement", "run", "do", "complete", "automate", "deploy"],
    "observer": ["monitor", "watch", "observe", "track", "report", "alert", "surveillance"],
    "learner": ["learn", "improve", "adapt", "train", "practice", "skill", "acquire"],
    "coordinator": ["coordinate", "manage", "delegate", "orchestrate", "team", "assign", "organize"],
    "guardian": ["safe", "guard", "protect", "risk", "compliance", "review", "audit", "security"],
}

_SUMMON_PHRASES = [
    "shadow arise", "shadow summon", "arise", "summon", "hey shadow",
    "shadow create", "shadow make", "shadow activate", "shadow start",
    "shadow analyze", "shadow observe", "shadow monitor", "shadow plan",
    "shadow research", "shadow investigate", "shadow protect", "shadow guard",
]


class ShadowSummon:
    """
    Voice Summon Router — bridges voice/text to ShadowArise + Shikigami + ShadowOS.

    Parses natural language intent, routes to the appropriate subsystem,
    and returns a structured response with a spoken (TTS-ready) summary.
    """

    _MAX_HISTORY = 200
    _MAX_PHRASES = 50

    def __init__(self):
        self._lock = threading.Lock()
        self._history: deque = deque(maxlen=self._MAX_HISTORY)
        self._ability_patterns = {}
        for ability, phrases in _ABILITY_MAP.items():
            for phrase in phrases:
                self._ability_patterns[phrase] = ability
        self._template_patterns = {}
        for template, phrases in _AGENT_TEMPLATES.items():
            for phrase in phrases:
                self._template_patterns[phrase] = template
        self._stats = {
            "total_summons": 0,
            "intents": {},
            "templates_matched": {},
            "abilities_matched": {},
        }
        self._stats_lock = threading.Lock()

    def classify_intent(self, text: str) -> SummonIntent:
        if not isinstance(text, str):
            return SummonIntent.UNKNOWN
        text_lower = re.sub(r'[,;.!?]', '', text.lower().strip())

        if any(p in text_lower for p in ["create agent", "arise agent", "spawn agent",
                                          "new agent", "make agent", "build agent",
                                          "shadow arise", "shadow create", "shadow summon",
                                          "summon agent", "launch agent"]):
            return SummonIntent.CREATE_AGENT

        if any(p in text_lower for p in ["list agents", "show agents", "active agents",
                                          "running agents", "what agents", "agent list",
                                          "show my agents", "my agents", "all agents"]):
            return SummonIntent.LIST_AGENTS

        if any(p in text_lower for p in ["what abilities", "list abilities", "show abilities",
                                          "available abilities", "what abilities", "what can you",
                                          "what shikigami"]):
            return SummonIntent.LIST_ABILITIES

        if any(p in text_lower for p in ["complete agent", "finish agent", "done with agent",
                                          "agent done", "agent complete", "mark done", "mark complete",
                                          "end agent", "close agent", "complete the",
                                          "finish the"]):
            return SummonIntent.COMPLETE_AGENT

        if any(p in text_lower for p in ["evolve agent", "improve agent", "agent learn",
                                          "agent adapt", "agent evolve",
                                          "evolve the", "improve the"]):
            return SummonIntent.EVOLVE_AGENT

        for phrase, ability in self._ability_patterns.items():
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                return SummonIntent.INVOKE_ABILITY

        if any(p in text_lower for p in ["what is", "how does", "tell me about",
                                           "should i", "what if", "would it", "could we",
                                           "how to", "how do", "when will",
                                           "can you", "what about",
                                           "what patterns", "what do you"]):
            return SummonIntent.ASK_QUESTION

        if re.search(r'\b(explain|why|describe)\b', text_lower):
            return SummonIntent.ASK_QUESTION

        if re.search(r'\b(notice|happened)\b', text_lower) or any(
            p in text_lower for p in ["i just", "i finished", "i completed", "i started"]
        ):
            return SummonIntent.OBSERVE_EVENT

        if re.search(r'\b(observe|saw|event|update)\b', text_lower):
            return SummonIntent.OBSERVE_EVENT

        if any(p in text_lower for p in ["system status", "status overview", "status summary",
                                          "show status", "check status", "get status",
                                          "give me status", "health check"]):
            return SummonIntent.GET_STATUS

        return SummonIntent.UNKNOWN

    def match_ability(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        text_lower = re.sub(r'[,;.!?]', '', text.lower().strip())
        best_ability = None
        best_len = 0
        for phrase, ability in self._ability_patterns.items():
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                if len(phrase) > best_len:
                    best_ability = ability
                    best_len = len(phrase)
        return best_ability

    def match_template(self, text: str) -> Optional[str]:
        if not isinstance(text, str):
            return None
        text_lower = re.sub(r'[,;.!?]', '', text.lower().strip())
        best_template = None
        best_len = 0
        for phrase, template in self._template_patterns.items():
            if re.search(r'\b' + re.escape(phrase) + r'\b', text_lower):
                if len(phrase) > best_len:
                    best_template = template
                    best_len = len(phrase)
        return best_template

    def process_voice(self, transcript: str, context: Optional[Dict] = None) -> SummonResponse:
        transcript = _sanitize_content(transcript)
        if not transcript:
            return SummonResponse(
                intent=SummonIntent.UNKNOWN,
                action="empty_input",
                result={},
                spoken="I didn't catch that. Could you repeat?",
                confidence=0.0,
            )

        intent = self.classify_intent(transcript)
        ability = self.match_ability(transcript)
        template = self.match_template(transcript)

        confidence = min(1.0, max(0.0, 0.7 + (0.15 if ability else 0) + (0.1 if template else 0)))

        response_data = {
            "transcript": transcript[:500],
            "intent": intent.value,
            "ability": ability,
            "template": template,
        }

        with self._lock:
            with self._stats_lock:
                self._stats["total_summons"] += 1
            self._stats["intents"][intent.value] = self._stats["intents"].get(intent.value, 0) + 1
            if ability:
                self._stats["abilities_matched"][ability] = self._stats["abilities_matched"].get(ability, 0) + 1
            if template:
                self._stats["templates_matched"][template] = self._stats["templates_matched"].get(template, 0) + 1
            self._history.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "transcript": transcript[:500],
                "intent": intent.value,
                "ability": ability,
                "template": template,
            })

        spoken = self._generate_spoken_response(intent, ability, template, transcript)
        response_data["confidence"] = round(confidence, 3)

        return SummonResponse(
            intent=intent,
            action=intent.value,
            result=response_data,
            spoken=spoken,
            confidence=confidence,
        )

    def _generate_spoken_response(self, intent: SummonIntent, ability: Optional[str],
                                   template: Optional[str], transcript: str) -> str:
        """Generate a natural language response suitable for TTS."""
        if intent == SummonIntent.CREATE_AGENT:
            tpl = template or "general-purpose"
            return f"Creating {tpl} agent. What's the mission objective?"
        elif intent == SummonIntent.INVOKE_ABILITY:
            if ability:
                names = {
                    "oracle": "Oracle", "x_ray": "X-Ray", "life_pulse": "Life Pulse",
                    "perfect_memory": "Perfect Memory", "pattern_sight": "Pattern Sight",
                    "social_radar": "Social Radar", "intent_seer": "Intent Seer",
                    "omni_sense": "Omni Sense", "shadow_council": "Shadow Council",
                    "reality_forge": "Reality Forge",
                }
                return f"Activating {names.get(ability, ability)} ability."
            return "Which ability would you like to activate?"
        elif intent == SummonIntent.ASK_QUESTION:
            return "Processing your question through the cognitive pipeline."
        elif intent == SummonIntent.OBSERVE_EVENT:
            return "Observation recorded. I'll process that."
        elif intent == SummonIntent.GET_STATUS:
            return "Fetching system status."
        elif intent == SummonIntent.LIST_ABILITIES:
            return "You have 10 Shikigami abilities. Oracle, X-Ray, Life Pulse, Perfect Memory, Pattern Sight, Social Radar, Intent Seer, Omni Sense, Shadow Council, and Reality Forge."
        elif intent == SummonIntent.LIST_AGENTS:
            return "Checking active agents."
        elif intent == SummonIntent.COMPLETE_AGENT:
            return "Marking agent as complete."
        elif intent == SummonIntent.EVOLVE_AGENT:
            return "Evolving agent based on feedback."
        else:
            context_clues = []
            if ability:
                context_clues.append(f"ability: {ability}")
            if template:
                context_clues.append(f"template: {template}")
            if context_clues:
                return f"I detected {' and '.join(context_clues)}. How would you like to proceed?"
            return "I'm here. How can I help?"

    def get_abilities(self) -> List[Dict]:
        return copy.deepcopy([
            {"id": "oracle", "name": "Oracle", "description": "Predict future outcomes using Monte Carlo simulation",
             "invocation": "Shadow, predict... / Shadow, what will happen if...",
             "icon": "🔮"},
            {"id": "x_ray", "name": "X-Ray", "description": "Scan and detect nearby devices, people, and networks",
             "invocation": "Shadow, scan... / Shadow, who's nearby...",
             "icon": "👁️"},
            {"id": "life_pulse", "name": "Life Pulse", "description": "Track people and relationships with confidence scoring",
             "invocation": "Shadow, track... / Shadow, where is...",
             "icon": "💓"},
            {"id": "perfect_memory", "name": "Perfect Memory", "description": "Recall anything from your history with perfect clarity",
             "invocation": "Shadow, remember... / Shadow, what did I...",
             "icon": "🧠"},
            {"id": "pattern_sight", "name": "Pattern Sight", "description": "Detect patterns, trends, and correlations in your data",
             "invocation": "Shadow, find patterns in... / Shadow, what's the trend...",
             "icon": "🔗"},
            {"id": "social_radar", "name": "Social Radar", "description": "Analyze social dynamics and predict group behavior",
             "invocation": "Shadow, analyze social... / Shadow, group dynamics...",
             "icon": "📡"},
            {"id": "intent_seer", "name": "Intent Seer", "description": "Predict what people will do next using Markov chains",
             "invocation": "Shadow, what will they... / Shadow, what's the intent...",
             "icon": "⏭️"},
            {"id": "omni_sense", "name": "Omni Sense", "description": "Get a complete environmental overview and summary",
             "invocation": "Shadow, overview / Shadow, status / Shadow, summarize...",
             "icon": "🌟"},
            {"id": "shadow_council", "name": "Shadow Council", "description": "Get consensus advice from multiple reasoning perspectives",
             "invocation": "Shadow, should I... / Shadow, advise on...",
             "icon": "⚖️"},
            {"id": "reality_forge", "name": "Reality Forge", "description": "Take action and make things happen in the real world",
             "invocation": "Shadow, create... / Shadow, build... / Shadow, set up...",
             "icon": "🔨"},
        ])

    def get_agent_templates(self) -> List[Dict]:
        return copy.deepcopy([
            {"id": "analyst", "name": "Analyst", "description": "Deep analysis and research agent",
             "invocation": "Shadow, analyze... / Shadow, research...",
             "icon": "🔬"},
            {"id": "creator", "name": "Creator", "description": "Creative generation and ideation agent",
             "invocation": "Shadow, create... / Shadow, design... / Shadow, write...",
             "icon": "🎨"},
            {"id": "planner", "name": "Planner", "description": "Strategic planning and coordination agent",
             "invocation": "Shadow, plan... / Shadow, organize... / Shadow, schedule...",
             "icon": "📋"},
            {"id": "executor", "name": "Executor", "description": "Action-oriented task execution agent",
             "invocation": "Shadow, execute... / Shadow, implement... / Shadow, run...",
             "icon": "⚡"},
            {"id": "observer", "name": "Observer", "description": "Passive monitoring and insight agent",
             "invocation": "Shadow, monitor... / Shadow, watch... / Shadow, observe...",
             "icon": "👁️"},
            {"id": "learner", "name": "Learner", "description": "Adaptive learning agent that evolves from feedback",
             "invocation": "Shadow, learn... / Shadow, improve... / Shadow, adapt...",
             "icon": "📚"},
            {"id": "coordinator", "name": "Coordinator", "description": "Multi-agent coordination and delegation agent",
             "invocation": "Shadow, coordinate... / Shadow, manage team... / Shadow, delegate...",
             "icon": "🎭"},
            {"id": "guardian", "name": "Guardian", "description": "Safety, compliance, and risk assessment agent",
             "invocation": "Shadow, guard... / Shadow, protect... / Shadow, audit...",
             "icon": "🛡️"},
        ])

    def get_history(self, limit: int = 50) -> List[Dict]:
        limit = min(max(1, limit), self._MAX_PHRASES)
        with self._lock:
            return copy.deepcopy(list(self._history)[-limit:])

    def get_stats(self) -> Dict:
        with self._lock:
            return {
                **copy.deepcopy(dict(self._stats)),
                "history_size": len(self._history),
            }