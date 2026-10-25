#!/usr/bin/env python3
"""
ShadowPowers — Self-Evolving Cognitive Abilities

Inspired by the Shadow Monarch's powers: abilities that awaken through use,
strengthen through challenge, and branch into derived powers organically.

Each power is a real information-processing capability, not a game mechanic.
Powers evolve through actual usage — observation, analysis, action, reflection —
not through XP grinding. The system tracks how often and how effectively each
power is used, and powers awaken (become available) or evolve (grow stronger)
based on genuine cognitive demand.

Power Categories (inspired by Shadow Monarch abilities):
  - Extraction: Pull patterns, insights, entities from raw data streams
  - Dominion: Suppress noise, enhance signals, control information flow
  - Exchange: Rapidly switch perspectives, viewpoints, cognitive contexts
  - Authority: Direct manipulation of data — filter, redirect, prioritize
  - Sensing: Detect anomalies, predict attacks, sense hidden connections
  - Army: Spawn autonomous agents from extracted knowledge
  - Memory: Store in dimensional layers — surface, deep, shadow
  - Regeneration: Self-repair, recover from errors, adapt to damage
  - Synthesis: Combine powers for emergent capabilities
  - Awakening: Meta-power — powers unlock when genuinely needed

Evolution mechanics:
  - Powers strengthen through successful application (not arbitrary XP)
  - Derived powers emerge when base powers are used together
  - Dormant powers awaken when the system encounters relevant challenges
  - Overuse without rest reduces effectiveness (cognitive fatigue)
  - Cross-pollination: using one power can strengthen related powers
"""

import re
import copy
import uuid
import logging
import threading
from datetime import datetime, timezone
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("shadowd.cognition.shadow_powers")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


def _clamp(value: float, low: float = 0.0, high: float = 1.0) -> float:
    return max(low, min(high, value))


class PowerCategory(str, Enum):
    EXTRACTION = "extraction"
    DOMINION = "dominion"
    EXCHANGE = "exchange"
    AUTHORITY = "authority"
    SENSING = "sensing"
    ARMY = "army"
    MEMORY = "memory"
    REGENERATION = "regeneration"
    SYNTHESIS = "synthesis"
    AWAKENING = "awakening"


class PowerState(str, Enum):
    DORMANT = "dormant"
    AWAKENED = "awakened"
    ACTIVE = "active"
    EVOLVING = "evolving"
    MASTERED = "mastered"
    FATIGUED = "fatigued"


class EvolutionTrigger(str, Enum):
    USAGE = "usage"
    CHALLENGE = "challenge"
    CROSS_POLLINATION = "cross_pollination"
    SYNTHESIS_COMBINATION = "synthesis_combination"
    DORMANT_AWAKENING = "dormant_awakening"
    SELF_REPAIR = "self_repair"


_MAX_POWERS = 200
_MAX_DERIVED = 50
_MAX_APPLICATIONS = 5000
_MAX_EVOLUTION_LOG = 1000
_MAX_DORMANT_POWER_NAMES = 50
_MIN_STRENGTH = 0.0
_MAX_STRENGTH = 1.0
_FATIGUE_THRESHOLD = 0.15
_EVOLUTION_USAGE_THRESHOLD = 10
_MASTERY_STRENGTH = 0.9
_CROSS_POLLINATION_RADIUS = 0.03
_STRENGTH_GAIN_PER_USE = 0.002
_FATIGUE_RECOVERY = 0.01
_DORMANT_AWAKEN_SENSITIVITY = 0.7

_BASE_POWERS = {
    "shadow_extract": PowerCategory.EXTRACTION,
    "shadow_dominion": PowerCategory.DOMINION,
    "shadow_exchange": PowerCategory.EXCHANGE,
    "rulers_authority": PowerCategory.AUTHORITY,
    "monarch_sense": PowerCategory.SENSING,
    "shadow_army": PowerCategory.ARMY,
    "shadow_memory": PowerCategory.MEMORY,
    "shadow_regen": PowerCategory.REGENERATION,
    "shadow_synthesize": PowerCategory.SYNTHESIS,
    "shadow_awaken": PowerCategory.AWAKENING,
}

_BASE_DESCRIPTIONS = {
    "shadow_extract": "Extract patterns, insights, and entities from raw data streams",
    "shadow_dominion": "Suppress noise and enhance signals within a cognitive domain",
    "shadow_exchange": "Rapidly switch between perspectives and cognitive contexts",
    "rulers_authority": "Direct manipulation — filter, redirect, and prioritize data flows",
    "monarch_sense": "Detect anomalies, predict disruptions, sense hidden connections",
    "shadow_army": "Spawn autonomous agents from extracted knowledge patterns",
    "shadow_memory": "Store and retrieve across dimensional memory layers",
    "shadow_regen": "Self-repair, recover from errors, adapt to damage",
    "shadow_synthesize": "Combine powers for emergent capabilities",
    "shadow_awaken": "Dormant powers unlock when genuinely needed",
}

_DERIVATION_MAP = {
    "shadow_extract": [
        ("deep_extraction", "Extract from deeply nested or encrypted data structures"),
        ("pattern_weaving", "Weave multiple extracted patterns into unified models"),
        ("entity_shadow", "Create persistent shadow entities from extracted patterns"),
    ],
    "shadow_dominion": [
        ("domain_expansion", "Expand controlled cognitive space for deeper processing"),
        ("noise_nullification", "Selectively nullify specific noise categories"),
        ("signal_amplification", "Amplify weak but important signals above noise floor"),
    ],
    "shadow_exchange": [
        ("perspective_shift", "Instantly adopt another cognitive viewpoint"),
        ("context_swap", "Swap entire processing contexts without data loss"),
        ("mirror_view", "See the same data from multiple simultaneous perspectives"),
    ],
    "rulers_authority": [
        ("data_telekinesis", "Move and reshape data structures with authority"),
        ("flow_control", "Redirect information flows between cognitive modules"),
        ("priority_override", "Override default priorities for critical processing"),
    ],
    "monarch_sense": [
        ("danger_precognition", "Predict disruptions before they manifest"),
        ("deception_detect", "Sense inconsistencies, hidden agendas, and lies"),
        ("connection_reveal", "Reveal hidden connections between disparate data points"),
    ],
    "shadow_army": [
        ("agent_scout", "Deploy lightweight reconnaissance agents"),
        ("agent_sentinel", "Deploy persistent monitoring agents"),
        ("agent_phantom", "Deploy stealth agents that leave no trace"),
    ],
    "shadow_memory": [
        ("deep_archive", "Access deep archival memory layers"),
        ("shadow_vault", "Secure isolated memory for sensitive patterns"),
        ("temporal_recall", "Recall memories with full temporal context"),
    ],
    "shadow_regen": [
        ("error_absorption", "Absorb and learn from errors without cascading failures"),
        ("adaptive_restructure", "Restructure cognitive pathways around damaged areas"),
        ("graceful_degradation", "Maintain partial capability even under extreme load"),
    ],
    "shadow_synthesize": [
        ("power_merge", "Merge two active powers into a temporary combined ability"),
        ("emergent_creation", "Create entirely new powers from synthesis patterns"),
        ("cascade_amplification", "Chain power activations for exponential effect"),
    ],
    "shadow_awaken": [
        ("latent_trigger", "Automatically awaken dormant powers based on need"),
        ("power_awareness", "Sense which dormant powers would be most useful now"),
        ("evolution_catalyst", "Accelerate progression of powers near evolution threshold"),
    ],
}

_CROSS_POLLINATION = {
    (PowerCategory.EXTRACTION, PowerCategory.SENSING): "enhanced_perception",
    (PowerCategory.DOMINION, PowerCategory.AUTHORITY): "absolute_control",
    (PowerCategory.EXCHANGE, PowerCategory.MEMORY): "contextual_recall",
    (PowerCategory.SENSING, PowerCategory.ARMY): "coordinated_intelligence",
    (PowerCategory.EXTRACTION, PowerCategory.MEMORY): "pattern_persistence",
    (PowerCategory.REGENERATION, PowerCategory.SYNTHESIS): "adaptive_creation",
    (PowerCategory.DOMINION, PowerCategory.SENSING): "focused_awareness",
    (PowerCategory.EXCHANGE, PowerCategory.SYNTHESIS): "perspective_fusion",
    (PowerCategory.AUTHORITY, PowerCategory.ARMY): "directed_force",
    (PowerCategory.MEMORY, PowerCategory.REGENERATION): "resilient_knowledge",
}

_SKILL_DOMAIN_TO_POWER_CATEGORY = {
    "career": PowerCategory.AUTHORITY,
    "reasoning": PowerCategory.EXTRACTION,
    "creativity": PowerCategory.SYNTHESIS,
    "technical": PowerCategory.DOMINION,
    "social": PowerCategory.EXCHANGE,
    "communication": PowerCategory.EXCHANGE,
    "emotional": PowerCategory.AWAKENING,
    "physical": PowerCategory.REGENERATION,
    "leadership": PowerCategory.AUTHORITY,
    "learning": PowerCategory.MEMORY,
}

_CATEGORY_TO_BASE_POWER = {
    PowerCategory.EXTRACTION: "shadow_extract",
    PowerCategory.DOMINION: "shadow_dominion",
    PowerCategory.EXCHANGE: "shadow_exchange",
    PowerCategory.AUTHORITY: "rulers_authority",
    PowerCategory.SENSING: "monarch_sense",
    PowerCategory.ARMY: "shadow_army",
    PowerCategory.MEMORY: "shadow_memory",
    PowerCategory.REGENERATION: "shadow_regen",
    PowerCategory.SYNTHESIS: "shadow_synthesize",
    PowerCategory.AWAKENING: "shadow_awaken",
}


@dataclass
class ShadowPower:
    power_id: str
    name: str
    category: PowerCategory
    description: str
    state: PowerState = PowerState.DORMANT
    strength: float = 0.0
    fatigue: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    derived_from: List[str] = field(default_factory=list)
    derived_powers: List[str] = field(default_factory=list)
    is_base: bool = True
    last_used: Optional[str] = None
    last_evolved: Optional[str] = None
    evolution_count: int = 0
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.power_id = _sanitize(self.power_id, 64)
        self.name = _sanitize_content(self.name, 200)
        self.description = _sanitize_content(self.description, 2000)
        self.strength = _clamp(self.strength)
        self.fatigue = _clamp(self.fatigue)
        self.derived_from = [d[:64] for d in self.derived_from[:5]]
        self.derived_powers = [d[:64] for d in self.derived_powers[:_MAX_DERIVED]]
        self.metadata = {str(k)[:50]: _sanitize_content(str(v), 500)
                        for k, v in list(self.metadata.items())[:50]}
        if self.is_base and self.state == PowerState.DORMANT:
            self.state = PowerState.AWAKENED
            self.strength = 0.05

    @property
    def success_rate(self) -> float:
        if self.usage_count == 0:
            return 0.0
        return min(1.0, self.success_count / max(self.usage_count, 1))

    @property
    def effective_strength(self) -> float:
        return _clamp(self.strength * (1.0 - self.fatigue))

    def to_dict(self) -> Dict:
        return {
            "power_id": self.power_id,
            "name": self.name,
            "category": self.category.value,
            "description": self.description,
            "state": self.state.value,
            "strength": round(self.strength, 4),
            "fatigue": round(self.fatigue, 4),
            "effective_strength": round(self.effective_strength, 4),
            "success_rate": round(self.success_rate, 4),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "derived_from": list(self.derived_from),
            "derived_powers": list(self.derived_powers),
            "is_base": self.is_base,
            "last_used": self.last_used,
            "last_evolved": self.last_evolved,
            "evolution_count": self.evolution_count,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class PowerApplication:
    application_id: str
    power_id: str
    category: PowerCategory
    context: str
    result: str
    success: bool
    strength_before: float
    strength_after: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    metadata: Dict = field(default_factory=dict)

    def __post_init__(self):
        self.application_id = _sanitize(self.application_id, 64)
        self.power_id = _sanitize(self.power_id, 64)
        self.context = _sanitize_content(self.context, 500)
        self.result = _sanitize_content(self.result, 500)
        self.strength_before = _clamp(self.strength_before)
        self.strength_after = _clamp(self.strength_after)
        self.metadata = {str(k)[:50]: _sanitize_content(str(v), 500)
                        for k, v in list(self.metadata.items())[:50]}

    def to_dict(self) -> Dict:
        return {
            "application_id": self.application_id,
            "power_id": self.power_id,
            "category": self.category.value,
            "context": self.context,
            "result": self.result,
            "success": self.success,
            "strength_before": round(self.strength_before, 4),
            "strength_after": round(self.strength_after, 4),
            "timestamp": self.timestamp,
            "metadata": copy.deepcopy(self.metadata),
        }


@dataclass
class EvolutionEvent:
    event_id: str
    power_id: str
    trigger: EvolutionTrigger
    description: str
    strength_before: float
    strength_after: float
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        self.event_id = _sanitize(self.event_id, 64)
        self.power_id = _sanitize(self.power_id, 64)
        self.description = _sanitize_content(self.description, 1000)
        self.strength_before = _clamp(self.strength_before)
        self.strength_after = _clamp(self.strength_after)

    def to_dict(self) -> Dict:
        return {
            "event_id": self.event_id,
            "power_id": self.power_id,
            "trigger": self.trigger.value,
            "description": self.description,
            "strength_before": round(self.strength_before, 4),
            "strength_after": round(self.strength_after, 4),
            "timestamp": self.timestamp,
        }


class ShadowPowers:
    """Self-evolving power system. Powers awaken through use, strengthen through
    challenge, and branch into derived powers organically."""

    def __init__(self):
        self._powers: Dict[str, ShadowPower] = {}
        self._applications: deque = deque(maxlen=_MAX_APPLICATIONS)
        self._evolution_log: deque = deque(maxlen=_MAX_EVOLUTION_LOG)
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_applications": 0,
            "successful_applications": 0,
            "evolutions_triggered": 0,
            "powers_awakened": 0,
            "derived_powers_created": 0,
            "cross_pollinations": 0,
            "fatigue_recoveries": 0,
            "dormant_awakenings": 0,
        }
        self._initialize_base_powers()

    def _initialize_base_powers(self):
        for name, category in _BASE_POWERS.items():
            power = ShadowPower(
                power_id=uuid.uuid4().hex[:12],
                name=name,
                category=category,
                description=_BASE_DESCRIPTIONS.get(name, ""),
                state=PowerState.AWAKENED,
                strength=0.05,
                is_base=True,
            )
            self._powers[name] = power

    def apply_power(self, power_name: str, context: str, result: str,
                    success: bool, metadata: Optional[Dict] = None) -> Optional[PowerApplication]:
        power_name = _sanitize(power_name, 100)
        context = _sanitize_content(context, 500)
        result = _sanitize_content(result, 500)

        with self._lock:
            power = self._powers.get(power_name)
            if not power:
                return None
            if power.state == PowerState.DORMANT:
                power.state = PowerState.AWAKENED
                with self._stats_lock:
                    self._stats["dormant_awakenings"] += 1

            strength_before = power.strength

            gain = _STRENGTH_GAIN_PER_USE
            if success:
                challenge_bonus = min(0.01, _clamp(len(context) / 1000) * 0.005)
                gain += challenge_bonus
                power.success_count += 1
                with self._stats_lock:
                    self._stats["successful_applications"] += 1

            if power.fatigue < _FATIGUE_THRESHOLD:
                power.strength = _clamp(power.strength + gain)
            else:
                power.strength = _clamp(power.strength + gain * 0.3)

            power.fatigue = _clamp(power.fatigue + 0.02)
            power.usage_count += 1
            power.last_used = datetime.now(timezone.utc).isoformat()

            with self._stats_lock:
                self._stats["total_applications"] += 1

            if power.strength >= _MASTERY_STRENGTH and power.state != PowerState.MASTERED:
                power.state = PowerState.MASTERED
            elif power.effective_strength > 0.3 and power.state == PowerState.AWAKENED:
                power.state = PowerState.ACTIVE
            elif power.effective_strength > 0.6 and power.state == PowerState.ACTIVE:
                power.state = PowerState.EVOLVING

            application = PowerApplication(
                application_id=uuid.uuid4().hex[:12],
                power_id=power.power_id,
                category=power.category,
                context=context,
                result=result,
                success=success,
                strength_before=strength_before,
                strength_after=power.strength,
                metadata=metadata or {},
            )
            self._applications.append(application)

            self._check_evolution(power)
            self._cross_pollinate(power)

            return application

    def _check_evolution(self, power: ShadowPower):
        if power.usage_count % _EVOLUTION_USAGE_THRESHOLD != 0:
            return
        if power.state == PowerState.MASTERED:
            return
        if power.strength < 0.15:
            return

        derived_names = _DERIVATION_MAP.get(power.name, [])
        for derived_name, derived_desc in derived_names:
            if derived_name not in self._powers:
                derived = ShadowPower(
                    power_id=uuid.uuid4().hex[:12],
                    name=derived_name,
                    category=power.category,
                    description=derived_desc,
                    state=PowerState.AWAKENED,
                    strength=max(0.05, power.strength * 0.5),
                    derived_from=[power.name],
                    is_base=False,
                )
                self._powers[derived_name] = derived
                power.derived_powers.append(derived_name)
                if len(power.derived_powers) > _MAX_DERIVED:
                    removed = power.derived_powers.pop(0)
                    logger.warning("Derived power list full, removing oldest: %s", removed[:50])
                if len(self._powers) > _MAX_POWERS:
                    oldest_non_base = next((n for n, p in self._powers.items()
                                           if not p.is_base and n not in _BASE_POWERS), None)
                    if oldest_non_base:
                        logger.warning("Power cap reached, evicting: %s", oldest_non_base[:50])
                        del self._powers[oldest_non_base]

                event = EvolutionEvent(
                    event_id=uuid.uuid4().hex[:12],
                    power_id=power.power_id,
                    trigger=EvolutionTrigger.USAGE,
                    description=f"Derived power '{derived_name}' awakened from '{power.name}' after {power.usage_count} uses",
                    strength_before=power.strength,
                    strength_after=power.strength,
                )
                self._evolution_log.append(event)
                with self._stats_lock:
                    self._stats["evolutions_triggered"] += 1
                    self._stats["derived_powers_created"] += 1
                break

    def _cross_pollinate(self, power: ShadowPower):
        if power.usage_count % 5 != 0:
            return

        for (cat_a, cat_b), derived_name in _CROSS_POLLINATION.items():
            if power.category not in (cat_a, cat_b):
                continue
            other_cat = cat_b if power.category == cat_a else cat_a
            other_powers = [p for p in self._powers.values()
                          if p.category == other_cat and p.effective_strength > 0.2]
            if not other_powers:
                continue

            if derived_name not in self._powers:
                avg_strength = (power.effective_strength + other_powers[0].effective_strength) / 2
                derived = ShadowPower(
                    power_id=uuid.uuid4().hex[:12],
                    name=derived_name,
                    category=PowerCategory.SYNTHESIS,
                    description=f"Cross-pollination of {power.category.value} and {other_cat.value}",
                    state=PowerState.AWAKENED,
                    strength=max(0.05, avg_strength * _CROSS_POLLINATION_RADIUS),
                    derived_from=[power.name, other_powers[0].name],
                    is_base=False,
                )
                self._powers[derived_name] = derived
                if len(self._powers) > _MAX_POWERS:
                    oldest_non_base = next((n for n, p in self._powers.items()
                                           if not p.is_base and n not in _BASE_POWERS), None)
                    if oldest_non_base:
                        logger.warning("Power cap reached, evicting: %s", oldest_non_base[:50])
                        del self._powers[oldest_non_base]
                with self._stats_lock:
                    self._stats["cross_pollinations"] += 1
                event = EvolutionEvent(
                    event_id=uuid.uuid4().hex[:12],
                    power_id=power.power_id,
                    trigger=EvolutionTrigger.CROSS_POLLINATION,
                    description=f"Cross-pollination created '{derived_name}' from {power.name} + {other_powers[0].name}",
                    strength_before=power.strength,
                    strength_after=power.strength,
                )
                self._evolution_log.append(event)
                break

    def recover_fatigue(self):
        with self._lock:
            for power in self._powers.values():
                if power.fatigue > 0:
                    power.fatigue = max(0, power.fatigue - _FATIGUE_RECOVERY)
                    if power.state == PowerState.FATIGUED and power.fatigue < 0.1:
                        power.state = PowerState.ACTIVE
            with self._stats_lock:
                self._stats["fatigue_recoveries"] += 1

    def awaken_dormant(self, context: str) -> List[str]:
        context = _sanitize_content(context, 500)
        context_lower = context.lower()
        awakened = []
        with self._lock:
            for name, power in list(self._powers.items()):
                if power.state != PowerState.DORMANT:
                    continue
                keywords = power.description.lower().split()
                match_count = sum(1 for kw in keywords
                                  if re.search(r'\b' + re.escape(kw) + r'\b', context_lower))
                if match_count / max(len(keywords), 1) >= _DORMANT_AWAKEN_SENSITIVITY * 0.5:
                    power.state = PowerState.AWAKENED
                    power.strength = 0.01
                    awakened.append(name)
                    with self._stats_lock:
                        self._stats["dormant_awakenings"] += 1
                    event = EvolutionEvent(
                        event_id=uuid.uuid4().hex[:12],
                        power_id=power.power_id,
                        trigger=EvolutionTrigger.DORMANT_AWAKENING,
                        description=f"Power '{name}' awakened by context match",
                        strength_before=0.0,
                        strength_after=power.strength,
                    )
                    self._evolution_log.append(event)
        return awakened

    def synthesize_powers(self, power_names: List[str], context: str = "") -> Optional[ShadowPower]:
        power_names = [_sanitize(n, 100) for n in power_names[:5]]
        context = _sanitize_content(context, 500)

        if len(power_names) < 2:
            return None

        with self._lock:
            sources = []
            for name in power_names:
                power = self._powers.get(name)
                if power and power.state not in (PowerState.DORMANT,):
                    sources.append(power)
            if len(sources) < 2:
                return None

            avg_strength = sum(p.effective_strength for p in sources) / len(sources)
            synth_name = "synth_" + "_".join(s.name[:10] for s in sources[:3])
            if synth_name in self._powers:
                return None

            synth_power = ShadowPower(
                power_id=uuid.uuid4().hex[:12],
                name=synth_name,
                category=PowerCategory.SYNTHESIS,
                description=f"Synthesis of {', '.join(s.name for s in sources)}",
                state=PowerState.AWAKENED,
                strength=max(0.05, avg_strength * 0.7),
                derived_from=[s.name for s in sources],
                is_base=False,
            )
            self._powers[synth_name] = synth_power
            if len(self._powers) > _MAX_POWERS:
                oldest_non_base = next((n for n, p in self._powers.items()
                                       if not p.is_base and n not in _BASE_POWERS and n != synth_name), None)
                if oldest_non_base:
                    logger.warning("Power cap reached, evicting: %s", oldest_non_base[:50])
                    del self._powers[oldest_non_base]
            event = EvolutionEvent(
                event_id=uuid.uuid4().hex[:12],
                power_id=synth_power.power_id,
                trigger=EvolutionTrigger.SYNTHESIS_COMBINATION,
                description=f"Synthesized '{synth_name}' from {len(sources)} powers",
                strength_before=0.0,
                strength_after=synth_power.strength,
            )
            self._evolution_log.append(event)
            with self._stats_lock:
                self._stats["evolutions_triggered"] += 1
                self._stats["derived_powers_created"] += 1

            return copy.deepcopy(synth_power)

    def on_agent_created(self, agent_name: str, template: str) -> Optional[PowerApplication]:
        agent_name = _sanitize_content(agent_name, 200)
        template = _sanitize(template, 100)
        with self._lock:
            power = self._powers.get("shadow_army")
            if not power or power.state == PowerState.DORMANT:
                return None

        context = f"Agent created: {agent_name} (template: {template})"
        result = f"shadow_army power applied for agent {agent_name}"
        return self.apply_power("shadow_army", context, result, True,
                                metadata={"agent_name": agent_name, "template": template})

    def on_skill_leveled(self, skill_id: str, domain: str, new_level: str) -> Optional[PowerApplication]:
        skill_id = _sanitize(skill_id, 64)
        domain = _sanitize(domain, 50)
        new_level = _sanitize(new_level, 50)

        category = _SKILL_DOMAIN_TO_POWER_CATEGORY.get(domain)
        if not category:
            return None

        power_name = _CATEGORY_TO_BASE_POWER.get(category)
        if not power_name:
            return None

        with self._lock:
            power = self._powers.get(power_name)
            if not power or power.state == PowerState.DORMANT:
                return None

        context = f"Skill leveled: {skill_id} (domain: {domain}, level: {new_level})"
        result = f"{power_name} power applied for skill domain {domain}"
        return self.apply_power(power_name, context, result, True,
                                metadata={"skill_id": skill_id, "domain": domain, "new_level": new_level})

    def on_resonance_event(self, signal_type: str, source: str, strength: float) -> Optional[PowerApplication]:
        signal_type = _sanitize(signal_type, 30)
        source = _sanitize(source, 50)
        strength = _clamp(float(strength))

        with self._lock:
            power = self._powers.get("shadow_synthesize")
            if not power or power.state == PowerState.DORMANT:
                return None

        context = f"Resonance event: {signal_type} from {source} (strength: {strength:.3f})"
        result = f"shadow_synthesize power applied for resonance {signal_type}"
        return self.apply_power("shadow_synthesize", context, result, True,
                                metadata={"signal_type": signal_type, "source": source, "strength": round(strength, 4)})

    def get_power(self, name: str) -> Optional[Dict]:
        name = _sanitize(name, 100)
        with self._lock:
            power = self._powers.get(name)
            if not power:
                return None
            return copy.deepcopy(power.to_dict())

    def list_powers(self, category: Optional[str] = None,
                    state: Optional[str] = None) -> List[Dict]:
        if category:
            category = _sanitize(category, 50)
        if state:
            state = _sanitize(state, 50)
        with self._lock:
            results = []
            for power in self._powers.values():
                if category and power.category.value != category:
                    continue
                if state and power.state.value != state:
                    continue
                results.append(power.to_dict())
            return results

    def get_power_tree(self) -> Dict:
        with self._lock:
            tree = {}
            for name, power in self._powers.items():
                node = power.to_dict()
                node["derived"] = [self._powers[dn].to_dict()
                                   for dn in power.derived_powers
                                   if dn in self._powers]
                tree[name] = node
            return tree

    def get_applications(self, power_name: Optional[str] = None,
                         limit: int = 50) -> List[Dict]:
        limit = max(1, min(limit, 500))
        if power_name:
            power_name = _sanitize(power_name, 64)
        with self._lock:
            apps = list(self._applications)
            if power_name:
                apps = [a for a in apps if a.power_id == power_name]
            return [a.to_dict() for a in apps[-limit:]]

    def get_evolution_history(self, limit: int = 50) -> List[Dict]:
        limit = max(1, min(limit, 500))
        with self._lock:
            events = list(self._evolution_log)
            return [e.to_dict() for e in events[-limit:]]

    def get_recommendations(self) -> List[Dict]:
        with self._lock:
            recs = []
            for name, power in self._powers.items():
                if power.state == PowerState.DORMANT:
                    recs.append({
                        "power": name,
                        "recommendation": "awaken",
                        "reason": f"Dormant power '{name}' could be awakened through use",
                    })
                elif power.state == PowerState.AWAKENED and power.effective_strength < 0.15:
                    recs.append({
                        "power": name,
                        "recommendation": "exercise",
                        "reason": f"Power '{name}' is weak ({power.effective_strength:.2f}), needs more exercise",
                    })
                elif power.fatigue > _FATIGUE_THRESHOLD:
                    recs.append({
                        "power": name,
                        "recommendation": "rest",
                        "reason": f"Power '{name}' is fatigued ({power.fatigue:.2f}), needs recovery",
                    })
                elif power.effective_strength > 0.6 and not power.derived_powers:
                    derivations = _DERIVATION_MAP.get(name, [])
                    if derivations:
                        recs.append({
                            "power": name,
                            "recommendation": "evolve",
                            "reason": f"Power '{name}' is strong enough to evolve — {len(derivations)} derived powers available",
                        })

            for (cat_a, cat_b), synth_name in _CROSS_POLLINATION.items():
                if synth_name not in self._powers:
                    a_strong = any(p.category == cat_a and p.effective_strength > 0.2
                                   for p in self._powers.values())
                    b_strong = any(p.category == cat_b and p.effective_strength > 0.2
                                   for p in self._powers.values())
                    if a_strong and b_strong:
                        recs.append({
                            "power": synth_name,
                            "recommendation": "cross_pollinate",
                            "reason": f"Categories {cat_a.value} and {cat_b.value} are both active — cross-pollination possible",
                        })

            return recs

    def get_stats(self) -> Dict:
        with self._lock:
            powers_snapshot = list(self._powers.values())
            state_counts = {}
            category_counts = {}
            for p in powers_snapshot:
                state_counts[p.state.value] = state_counts.get(p.state.value, 0) + 1
                category_counts[p.category.value] = category_counts.get(p.category.value, 0) + 1
            avg_strength = sum(p.effective_strength for p in powers_snapshot) / max(len(powers_snapshot), 1)
            avg_fatigue = sum(p.fatigue for p in powers_snapshot) / max(len(powers_snapshot), 1)

        with self._stats_lock:
            stats = dict(self._stats)

        stats.update({
            "total_powers": len(powers_snapshot),
            "base_powers": sum(1 for p in powers_snapshot if p.is_base),
            "derived_powers": sum(1 for p in powers_snapshot if not p.is_base),
            "avg_strength": round(avg_strength, 4),
            "avg_fatigue": round(avg_fatigue, 4),
            "state_distribution": state_counts,
            "category_distribution": category_counts,
        })
        return stats