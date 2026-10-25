#!/usr/bin/env python3
"""
Enhanced Shikigami Abilities - TRUE SUPERHUMAN POWERS

Each ability is:
- Grounded in REAL mathematical/ML techniques
- TESTED with proper test cases
- SPECIFIC in outputs (names, times, probabilities)
- ACTIONABLE with concrete recommendations
"""

import asyncio
import copy
import json
import os
import re
import subprocess
import random
import math
import statistics
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
import logging

logger = logging.getLogger("shadowd.abilities")

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 10000) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 100000) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


# Try to import numpy, fall back gracefully
try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

# Try to import scipy for FFT
try:
    from scipy import fft as scipy_fft
    from scipy import signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False

DATA_DIR = Path.home() / "digital-shadow" / "data"
PEOPLE_DB = DATA_DIR / "people.json"
DECISIONS_DB = DATA_DIR / "decisions.json"
PATTERNS_DB = DATA_DIR / "patterns.json"
OBSERVATIONS_DB = DATA_DIR / "observations.json"
SEQUENCES_DB = DATA_DIR / "sequences.json"

for p in [PEOPLE_DB, DECISIONS_DB, PATTERNS_DB, OBSERVATIONS_DB, SEQUENCES_DB]:
    p.parent.mkdir(parents=True, exist_ok=True)


@dataclass
class AbilityResult:
    success: bool
    output: str
    data: Dict[str, Any] = field(default_factory=dict)
    confidence: float = 0.0
    commands_run: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)


class MonteCarloSimulator:
    """Monte Carlo simulation for Oracle predictions."""

    def __init__(self, seed: int = None):
        self.seed = seed
        if seed is not None:
            random.seed(seed)

    def simulate_outcome(self, base_probability: float, num_simulations: int = 10000) -> Dict:
        """Run Monte Carlo simulations to get confidence intervals."""
        if HAS_NUMPY:
            outcomes = np.random.binomial(1, base_probability, num_simulations)
            mean = float(np.mean(outcomes))
            std = float(np.std(outcomes))
            ci_95 = (round(mean - 1.96 * std, 6), round(mean + 1.96 * std, 6))
        else:
            outcomes = [1 if random.random() < base_probability else 0 for _ in range(num_simulations)]
            mean = sum(outcomes) / num_simulations
            variance = sum((x - mean) ** 2 for x in outcomes) / num_simulations
            std = math.sqrt(variance)
            ci_95 = (round(mean - 1.96 * std, 6), round(mean + 1.96 * std, 6))

        return {
            "probability": round(mean, 6),
            "std_dev": round(std, 6),
            "confidence_interval_95": ci_95,
            "simulations": num_simulations
        }

    def predict_time(self, base_hours: float, variance: float, num_simulations: int = 10000) -> Dict:
        """Predict completion time with uncertainty."""
        if HAS_NUMPY:
            times = np.random.normal(base_hours, variance, num_simulations)
            mean = float(np.mean(times))
            std = float(np.std(times))
            p95 = float(np.percentile(times, 95))
        else:
            times = [random.gauss(base_hours, variance) for _ in range(num_simulations)]
            mean = sum(times) / num_simulations
            variance_sum = sum((t - mean) ** 2 for t in times)
            std = math.sqrt(variance_sum / num_simulations)
            times_sorted = sorted(times)
            p95 = times_sorted[int(0.95 * num_simulations)]

        return {
            "expected_hours": round(mean, 3),
            "std_dev_hours": round(std, 3),
            "p95_hours": round(p95, 3),
            "simulations": num_simulations
        }


class DeviceFingerprinter:
    """Device fingerprinting for X-Ray and Life Pulse."""

    _MAX_KNOWN_DEVICES = 5000
    _MAX_DEVICE_SIGNATURES = 5000

    # Word-boundary device name patterns
    _DEVICE_PATTERNS = [
        (re.compile(r'\biphone\b'), "apple_mobile"),
        (re.compile(r'\bipad\b'), "apple_mobile"),
        (re.compile(r'\bwatch\b'), "smartwatch"),
        (re.compile(r'\bairpod\w*\b'), "audio"),
        (re.compile(r'\bheadphone\w*\b'), "audio"),
        (re.compile(r'\bmac\b'), "apple_computer"),
        (re.compile(r'\bmacbook\b'), "apple_computer"),
        (re.compile(r'\bgalaxy\b'), "android_mobile"),
        (re.compile(r'\bandroid\b'), "android_mobile"),
    ]

    def __init__(self):
        self._known_devices: Dict[str, Dict] = {}
        self._known_devices_order: List[str] = []
        self._device_signatures: Dict[str, List[str]] = {}
        self._lock = threading.Lock()

    def load_people(self, people_db: Path) -> Dict:
        """Load known people and their devices."""
        try:
            if people_db.exists():
                data = json.loads(people_db.read_text())
                with self._lock:
                    for name, info in data.items():
                        sanitized_name = _sanitize(str(name), 200)
                        for mac in info.get("devices", []):
                            mac_upper = _sanitize(str(mac), 50).upper()
                            self._known_devices[mac_upper] = {
                                "name": sanitized_name,
                                "device_type": _sanitize(str(info.get("device_types", {}).get(mac, "unknown")), 100),
                                "patterns": copy.deepcopy(info.get("patterns", {}))
                            }
                            self._known_devices_order.append(mac_upper)
                        # Evict oldest devices if over cap
                        while len(self._known_devices_order) > self._MAX_KNOWN_DEVICES:
                            old_mac = self._known_devices_order.pop(0)
                            self._known_devices.pop(old_mac, None)
                return data
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Load error: %s", type(e).__name__)
        return {}

    def identify_device(self, mac: str, name: str = "") -> Dict:
        """Identify a device from scan data."""
        mac = _sanitize(str(mac), 50).upper()
        name = _sanitize_content(str(name), 500)

        with self._lock:
            if mac in self._known_devices:
                person = self._known_devices[mac]
                return {
                    "identified": True,
                    "person": person["name"],
                    "device_type": person["device_type"],
                    "confidence": 0.95
                }

        # Device type inference from name using word boundaries
        device_type = "unknown"
        name_lower = name.lower()
        for pattern, dtype in self._DEVICE_PATTERNS:
            if pattern.search(name_lower):
                device_type = dtype
                break

        return {
            "identified": False,
            "device_type": device_type,
            "suggested_name": name,
            "confidence": 0.3
        }

    def get_person_constellation(self, person_name: str) -> List[str]:
        """Get all devices associated with a person."""
        person_name = _sanitize(str(person_name), 200)
        with self._lock:
            return [mac for mac, info in self._known_devices.items()
                    if info["name"] == person_name]

    def calculate_presence_confidence(self, detected_macs: List[str], person_name: str) -> float:
        """Calculate confidence that a person is present based on device constellation."""
        constellation = self.get_person_constellation(person_name)
        if not constellation:
            return 0.0

        detected_set = {m.upper() for m in detected_macs}
        constellation_set = {m.upper() for m in constellation}

        detected_from_person = detected_set & constellation_set
        total_devices = len(constellation_set)

        if not detected_from_person:
            return 0.0

        # More devices = higher confidence
        ratio = len(detected_from_person) / total_devices

        # Weight by number of devices (constellation size matters)
        # 1 device: ~50% confidence, 2 devices: ~75%, 3+ devices: ~90%+
        confidence = min(0.95, 0.4 * ratio + 0.15 * len(detected_from_person))

        return round(confidence, 4)


class PatternAnalyzer:
    """Pattern detection with FFT and statistical analysis."""

    _MAX_PATTERNS = 1000
    _MAX_OBSERVATIONS_PER_PATTERN = 500

    def __init__(self):
        self.patterns: Dict[str, List[Tuple[datetime, Any]]] = defaultdict(list)
        self._pattern_order: List[str] = []
        self._lock = threading.Lock()

    def add_observation(self, pattern_name: str, value: Any, timestamp: datetime = None):
        """Add observation to pattern tracking."""
        pattern_name = _sanitize(str(pattern_name), 200)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        with self._lock:
            self.patterns[pattern_name].append((timestamp, value))
            if pattern_name not in self._pattern_order:
                self._pattern_order.append(pattern_name)
            # Cap observations per pattern (FIFO eviction)
            if len(self.patterns[pattern_name]) > self._MAX_OBSERVATIONS_PER_PATTERN:
                self.patterns[pattern_name] = self.patterns[pattern_name][-self._MAX_OBSERVATIONS_PER_PATTERN // 2:]
            # Cap total patterns
            while len(self._pattern_order) > self._MAX_PATTERNS:
                old_pattern = self._pattern_order.pop(0)
                self.patterns.pop(old_pattern, None)

    def detect_frequency(self, values: List[float]) -> Dict:
        """Detect frequency patterns using FFT."""
        if len(values) < 4:
            return {"dominant_frequency": None, "period": None}

        if HAS_SCIPY:
            # Remove DC component
            values_array = np.array(values)
            values_centered = values_array - np.mean(values_array)

            # FFT
            fft_values = scipy_fft.fft(values_centered)
            freqs = scipy_fft.fftfreq(len(values))

            # Find dominant frequency (positive frequencies only)
            positive_freq_idx = np.argmax(np.abs(fft_values[:len(fft_values)//2]))
            dominant_freq = abs(freqs[positive_freq_idx])

            period = 1.0 / dominant_freq if dominant_freq > 0 else None

            return {
                "dominant_frequency": round(float(dominant_freq), 6),
                "period": round(float(period), 6) if period is not None else None,
                "magnitude": round(float(np.abs(fft_values[positive_freq_idx])), 6)
            }
        else:
            # Fallback: simple autocorrelation
            n = len(values)
            mean = sum(values) / n
            variance = sum((v - mean) ** 2 for v in values) / n

            if variance == 0:
                return {"dominant_frequency": None, "period": None}

            # Find period by checking for repeating patterns
            for lag in range(1, min(n // 2, 50)):
                autocorr = sum((values[i] - mean) * (values[i + lag] - mean) for i in range(n - lag)) / (n * variance)
                if autocorr > 0.7:
                    return {
                        "dominant_frequency": round(1.0 / lag, 6),
                        "period": lag,
                        "autocorrelation": round(autocorr, 6)
                    }

            return {"dominant_frequency": None, "period": None}

    def find_correlations(self, observations: List[Dict]) -> List[Dict]:
        """Find correlations between observation types."""
        # Group by type
        by_type: Dict[str, List[float]] = defaultdict(list)

        for obs in observations:
            obs_type = obs.get("type", "unknown")
            # Convert to numeric if possible
            value = obs.get("value", 1)
            if isinstance(value, (int, float)):
                by_type[obs_type].append(value)
            else:
                by_type[obs_type].append(1.0)  # Binary presence

        correlations = []
        types = list(by_type.keys())

        for i, t1 in enumerate(types):
            for t2 in types[i+1:]:
                v1 = by_type[t1]
                v2 = by_type[t2]

                # Make same length
                min_len = min(len(v1), len(v2))
                if min_len < 3:
                    continue

                v1_slice = v1[:min_len]
                v2_slice = v2[:min_len]

                # Pearson correlation
                mean1 = sum(v1_slice) / min_len
                mean2 = sum(v2_slice) / min_len

                numerator = sum((v1_slice[i] - mean1) * (v2_slice[i] - mean2) for i in range(min_len))
                denom1 = math.sqrt(sum((v - mean1) ** 2 for v in v1_slice))
                denom2 = math.sqrt(sum((v - mean2) ** 2 for v in v2_slice))

                if denom1 > 0 and denom2 > 0:
                    corr = numerator / (denom1 * denom2)
                    if abs(corr) > 0.3:  # Only report meaningful correlations
                        correlations.append({
                            "type1": t1,
                            "type2": t2,
                            "correlation": round(corr, 3),
                            "strength": "strong" if abs(corr) > 0.7 else "moderate" if abs(corr) > 0.5 else "weak"
                        })

        return sorted(correlations, key=lambda x: abs(x["correlation"]), reverse=True)

    def detect_causation(self, observations: List[Dict], cause_type: str, effect_type: str,
                         lag_hours: int = 24) -> Dict:
        """Detect if cause_type occurrences lead to effect_type within lag_hours."""
        cause_type = _sanitize(str(cause_type), 200)
        effect_type = _sanitize(str(effect_type), 200)
        cause_times = []
        effect_times = []

        for obs in observations:
            timestamp = obs.get("timestamp")
            if not timestamp:
                continue

            try:
                dt = datetime.fromisoformat(timestamp) if isinstance(timestamp, str) else timestamp
            except (ValueError, TypeError):
                continue

            if obs.get("type") == cause_type:
                cause_times.append(dt)
            elif obs.get("type") == effect_type:
                effect_times.append(dt)

        if not cause_times or not effect_times:
            return {"causation_detected": False, "confidence": 0, "explanation": "Insufficient data"}

        # Count effects that follow causes
        effects_following_cause = 0
        total_causes = len(cause_times)

        for cause_time in cause_times:
            # Look for effect within lag window
            lag_delta = timedelta(hours=lag_hours)
            for effect_time in effect_times:
                if 0 <= (effect_time - cause_time).total_seconds() <= lag_delta.total_seconds():
                    effects_following_cause += 1
                    break  # Only count once per cause

        causation_rate = effects_following_cause / total_causes if total_causes > 0 else 0

        # Compare to baseline (random effect occurrence)
        effect_rate = len(effect_times) / max(1, len(observations))

        # Causation confidence: how much higher is our rate than baseline?
        if effect_rate > 0:
            lift = causation_rate / effect_rate
        else:
            lift = 0

        safe_cause = _sanitize_content(str(cause_type))
        safe_effect = _sanitize_content(str(effect_type))
        return {
            "causation_detected": causation_rate > 0.5 and lift > 1.5,
            "cause_type": cause_type,
            "effect_type": effect_type,
            "causation_rate": round(causation_rate, 3),
            "baseline_rate": round(effect_rate, 3),
            "lift": round(lift, 2),
            "confidence": round(min(0.95, causation_rate * 0.8), 4),
            "explanation": f"{safe_cause} -> {safe_effect} occurs {causation_rate*100:.1f}% of the time within {lag_hours}h (baseline: {effect_rate*100:.1f}%)"
        }


# Word-boundary for tag extraction keywords
_TAG_KEYWORDS = {
    "work": [re.compile(r'\b' + re.escape(w) + r'\b') for w in ["work", "project", "meeting", "deadline", "boss", "job", "career"]],
    "relationship": [re.compile(r'\b' + re.escape(w) + r'\b') for w in ["friend", "partner", "family", "relationship", "love", "date"]],
    "health": [re.compile(r'\b' + re.escape(w) + r'\b') for w in ["health", "exercise", "diet", "doctor", "sleep", "feel"]],
    "finance": [re.compile(r'\b' + re.escape(w) + r'\b') for w in ["money", "buy", "invest", "save", "spend", "price", "cost"]],
    "learning": [re.compile(r'\b' + re.escape(w) + r'\b') for w in ["learn", "study", "course", "book", "skill", "practice"]],
}


class BehaviorPredictor:
    """Predict human behavior based on patterns."""

    _MAX_PEOPLE = 500
    _MAX_CONTEXTS = 50

    def __init__(self):
        self.behavior_models: Dict[str, Dict] = {}
        self._people_order: List[str] = []
        self._lock = threading.Lock()

    def track_behavior(self, person: str, behavior_type: str, occurred: bool,
                       context: Dict = None):
        """Track behavior for a person."""
        person = _sanitize(str(person), 200)
        behavior_type = _sanitize(str(behavior_type), 200)
        with self._lock:
            if person not in self.behavior_models:
                self.behavior_models[person] = {}
                self._people_order.append(person)
                # Evict oldest person if over cap
                while len(self._people_order) > self._MAX_PEOPLE:
                    old_person = self._people_order.pop(0)
                    self.behavior_models.pop(old_person, None)

            if behavior_type not in self.behavior_models[person]:
                self.behavior_models[person][behavior_type] = {
                    "total": 0,
                    "occurred": 0,
                    "contexts": []
                }

            self.behavior_models[person][behavior_type]["total"] += 1
            if occurred:
                self.behavior_models[person][behavior_type]["occurred"] += 1
            if context:
                sanitized_context = {str(k)[:100]: str(v)[:500] for k, v in (context.items() if isinstance(context, dict) else [])}
                self.behavior_models[person][behavior_type]["contexts"].append(sanitized_context)
                if len(self.behavior_models[person][behavior_type]["contexts"]) > self._MAX_CONTEXTS:
                    self.behavior_models[person][behavior_type]["contexts"] = \
                        self.behavior_models[person][behavior_type]["contexts"][-self._MAX_CONTEXTS // 2:]

    def predict_behavior(self, person: str, behavior_type: str = None) -> Dict:
        """Predict behavior probability for a person."""
        person = _sanitize(str(person), 200)
        if behavior_type:
            behavior_type = _sanitize(str(behavior_type), 200)
        with self._lock:
            if person not in self.behavior_models:
                return {"confidence": 0, "behaviors": {}}

            predictions = {}

            behaviors_to_check = [behavior_type] if behavior_type else list(self.behavior_models[person].keys())

            for btype in behaviors_to_check:
                if btype not in self.behavior_models[person]:
                    continue

                data = self.behavior_models[person][btype]
                total = data["total"]
                occurred = data["occurred"]

                if total > 0:
                    probability = occurred / total
                    confidence = min(0.95, total / 10) * probability  # Higher confidence with more data

                    predictions[btype] = {
                        "probability": round(probability, 3),
                        "confidence": round(confidence, 3),
                        "sample_size": total
                    }

            return {
                "person": person,
                "behaviors": predictions,
                "confidence": round(max((b.get("confidence", 0) for b in predictions.values()), default=0), 4)
            }

    def get_warning_signs(self, person: str) -> List[Dict]:
        """Get signs that indicate imminent behavior."""
        person = _sanitize(str(person), 200)
        with self._lock:
            if person not in self.behavior_models:
                return []

            signs = []
            for btype, data in self.behavior_models[person].items():
                if data["occurred"] / max(1, data["total"]) > 0.5:
                    # Look for common contexts
                    context_counts = Counter(str(c) for c in data["contexts"])
                    for ctx, count in context_counts.most_common(3):
                        if count / max(1, data["occurred"]) > 0.3:  # Present in >30% of occurrences
                            signs.append({
                                "behavior": btype,
                                "warning_sign": _sanitize_content(str(ctx)),
                                "probability": round(data["occurred"] / data["total"], 3)
                            })

            return signs


class ActionSequencePredictor:
    """Predict next actions from sequence."""

    _MAX_SEQUENCES = 5000
    _MAX_TRANSITIONS = 5000
    _MAX_ACTIONS_PER_SEQUENCE = 500

    def __init__(self):
        self.sequences: Dict[str, List[str]] = defaultdict(list)
        self._sequence_order: List[str] = []
        self.transitions: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._transition_order: List[str] = []
        self._lock = threading.Lock()

    def record_sequence(self, sequence_id: str, actions: List[str]):
        """Record an action sequence."""
        sequence_id = _sanitize(str(sequence_id), 200)
        actions = [_sanitize(str(a), 200) for a in actions[:self._MAX_ACTIONS_PER_SEQUENCE]]
        with self._lock:
            self.sequences[sequence_id] = actions
            if sequence_id not in self._sequence_order:
                self._sequence_order.append(sequence_id)
            # Evict oldest sequences if over cap
            while len(self._sequence_order) > self._MAX_SEQUENCES:
                old_id = self._sequence_order.pop(0)
                self.sequences.pop(old_id, None)

            # Build transition matrix
            for i in range(len(actions) - 1):
                current = actions[i]
                next_action = actions[i + 1]
                self.transitions[current][next_action] += 1
                if current not in self._transition_order:
                    self._transition_order.append(current)
            # Evict oldest transitions if over cap
            while len(self._transition_order) > self._MAX_TRANSITIONS:
                old_key = self._transition_order.pop(0)
                self.transitions.pop(old_key, None)

    def predict_next(self, current_actions: List[str], top_k: int = 3) -> List[Dict]:
        """Predict next actions."""
        if not current_actions:
            return []

        last_action = _sanitize(str(current_actions[-1]), 200)

        with self._lock:
            if last_action not in self.transitions:
                return []

            # Get possible next actions with counts
            next_counts = dict(self.transitions[last_action])
            total = sum(next_counts.values())

        predictions = []
        for action, count in sorted(next_counts.items(), key=lambda x: -x[1])[:top_k]:
            predictions.append({
                "action": action,
                "probability": round(count / total, 3),
                "count": count
            })

        return predictions

    def predict_sequence(self, current_actions: List[str], length: int = 5) -> List[str]:
        """Predict a sequence of future actions."""
        predicted = []
        actions = [_sanitize(str(a), 200) for a in current_actions]

        for _ in range(length):
            next_preds = self.predict_next(actions, top_k=1)
            if not next_preds:
                break
            predicted.append(next_preds[0]["action"])
            actions.append(next_preds[0]["action"])

        return predicted


class DecisionTracker:
    """Track past decisions and outcomes."""

    _MAX_DECISIONS = 10000

    def __init__(self, db_path: Path = DECISIONS_DB):
        self.db_path = db_path
        self.decisions: List[Dict] = []
        self._lock = threading.Lock()
        self._load()

    def _load(self):
        try:
            if self.db_path.exists():
                data = json.loads(self.db_path.read_text())
                if isinstance(data, list):
                    self.decisions = data
                elif isinstance(data, dict):
                    self.decisions = list(data.values())
                # Cap at max decisions
                if len(self.decisions) > self._MAX_DECISIONS:
                    self.decisions = self.decisions[-self._MAX_DECISIONS:]
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Load decisions error: %s", type(e).__name__)
            self.decisions = []

    def _save(self):
        try:
            tmp = self.db_path.with_suffix(f".tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                json.dump(self.decisions, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(self.db_path))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Save decisions error: %s", type(e).__name__)

    def record_decision(self, situation: str, choice: str, outcome: str = None,
                        satisfaction: int = None) -> Dict:
        """Record a decision for future reference."""
        situation = _sanitize_content(str(situation), 10000)
        choice = _sanitize_content(str(choice), 5000)
        if outcome is not None:
            outcome = _sanitize_content(str(outcome), 10000)
        if satisfaction is not None:
            satisfaction = max(1, min(10, int(satisfaction)))
        entry = {
            "id": f"dec_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}_{random.randint(1000,9999)}",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "situation": situation,
            "choice": choice,
            "outcome": outcome,
            "satisfaction": satisfaction,
            "tags": self._extract_tags(situation)
        }
        with self._lock:
            self.decisions.append(entry)
            # Evict oldest if over cap
            if len(self.decisions) > self._MAX_DECISIONS:
                self.decisions = self.decisions[-self._MAX_DECISIONS:]
            self._save()
        return entry

    def _extract_tags(self, text: str) -> List[str]:
        """Extract meaningful tags from text using word boundary matching."""
        text_lower = text.lower()
        return [tag for tag, patterns in _TAG_KEYWORDS.items()
                if any(p.search(text_lower) for p in patterns)]

    def find_similar(self, situation: str, limit: int = 5) -> List[Dict]:
        """Find similar past decisions."""
        situation_words = set(situation.lower().split())
        with self._lock:
            scored = []

            for dec in self.decisions:
                dec_words = set(dec.get("situation", "").lower().split())
                # Word overlap
                overlap = len(situation_words & dec_words)
                # Tag overlap
                tag_overlap = len(set(dec.get("tags", [])) & set(self._extract_tags(situation)))

                score = overlap + tag_overlap * 2  # Weight tags more
                if score > 0:
                    scored.append((score, dec))

            scored.sort(key=lambda x: -x[0])
            return [copy.deepcopy(d) for _, d in scored[:limit]]

    def get_recommendation(self, situation: str) -> Dict:
        """Get recommendation based on past decisions."""
        similar = self.find_similar(situation, limit=10)

        if not similar:
            return {"recommendation": None, "confidence": 0, "reason": "No similar past decisions"}

        # Group by choice
        choice_outcomes: Dict[str, List[Dict]] = defaultdict(list)
        for dec in similar:
            choice_outcomes[dec.get("choice", "unknown")].append(dec)

        # Find best choice by average satisfaction
        best_choice = None
        best_satisfaction = 0
        best_count = 0

        for choice, decs in choice_outcomes.items():
            satisfactions = [d.get("satisfaction", 5) for d in decs if d.get("satisfaction") is not None]
            if satisfactions:
                avg_sat = sum(satisfactions) / len(satisfactions)
                if avg_sat > best_satisfaction or (avg_sat == best_satisfaction and len(decs) > best_count):
                    best_satisfaction = avg_sat
                    best_choice = choice
                    best_count = len(decs)

        return {
            "recommendation": best_choice,
            "confidence": round(min(0.9, best_count / 5), 4),
            "avg_satisfaction": round(best_satisfaction, 1),
            "sample_size": best_count,
            "similar_situations": len(similar)
        }


# Export classes
__all__ = [
    'AbilityResult',
    'MonteCarloSimulator',
    'DeviceFingerprinter',
    'PatternAnalyzer',
    'BehaviorPredictor',
    'ActionSequencePredictor',
    'DecisionTracker',
]