#!/usr/bin/env python3
"""
SUPERHUMAN ABILITIES - Next Level Powers

Each ability uses REAL computational techniques:
- Bayesian inference for predictions
- Signal triangulation for location
- Markov decision processes for behavior
- Knowledge graphs for memory
- Reinforcement learning patterns for decisions

No hallucinations. All grounded in math and data.
"""

import asyncio
import json
import subprocess
import re
import math
import random
import hashlib
import threading
import copy
import os
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
from collections import defaultdict, Counter
from statistics import mean, stdev
import logging

logger = logging.getLogger("shadowd.superhuman")

try:
    import numpy as np
    HAS_NUMPY = True
except ImportError:
    HAS_NUMPY = False
    np = None

DATA_DIR = Path.home() / "digital-shadow" / "data"

_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: Any, max_length: int = 200) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()[:max_length]
    return _SANITIZE_STRICT_RE.sub('', text)


def _sanitize_content(text: Any, max_length: int = 10000) -> str:
    if not isinstance(text, str):
        text = str(text) if text is not None else ""
    text = text.strip()[:max_length]
    return _SANITIZE_CONTENT_RE.sub('', text)


# ═════════════════════════════════════════════════════════════════
# BAYESIAN PREDICTION ENGINE
# ═════════════════════════════════════════════════════════════════

class BayesianPredictor:
    """
    Bayesian inference for predictions.
    
    P(H|E) = P(E|H) * P(H) / P(E)
    
    Real Bayesian updating, not fake percentages.
    """
    
    _MAX_HYPOTHESES = 200

    def __init__(self):
        self._lock = threading.Lock()
        self.hypotheses: Dict[str, Dict] = {}
        self.evidence_counts: Dict[str, int] = defaultdict(int)
        self.total_observations = 0
    
    def add_hypothesis(self, name: str, prior: float, evidence_types: List[str]):
        name = _sanitize(name)
        with self._lock:
            if len(self.hypotheses) >= self._MAX_HYPOTHESES:
                oldest = next(iter(self.hypotheses))
                del self.hypotheses[oldest]
            self.hypotheses[name] = {
                "prior": prior,
                "posterior": prior,
                "evidence_likelihoods": {_sanitize(e): 0.5 for e in evidence_types},
                "count": 0
            }
    
    def observe(self, evidence: str, hypotheses_match: Dict[str, bool]):
        evidence = _sanitize(evidence)
        with self._lock:
            self.evidence_counts[evidence] += 1
            self.total_observations += 1
            
            for h_name, h_data in self.hypotheses.items():
                if evidence in h_data["evidence_likelihoods"]:
                    matches = hypotheses_match.get(h_name, False)
                    if matches:
                        h_data["count"] += 1
                    
                    alpha = h_data["count"] + 1
                    beta = self.evidence_counts[evidence] - h_data["count"] + 1
                    h_data["evidence_likelihoods"][evidence] = alpha / (alpha + beta)
            
            self._update_posteriors_unlocked()
    
    def _update_posteriors_unlocked(self):
        unnormalized = {}
        for h_name, h_data in self.hypotheses.items():
            posterior = h_data["prior"]
            for evidence_name, likelihood in h_data["evidence_likelihoods"].items():
                posterior *= likelihood
            unnormalized[h_name] = posterior
        
        total = sum(unnormalized.values())
        if total > 0:
            for h_name in unnormalized:
                self.hypotheses[h_name]["posterior"] = unnormalized[h_name] / total
        else:
            for h_name in self.hypotheses:
                self.hypotheses[h_name]["posterior"] = 1.0 / len(self.hypotheses)
    
    def predict(self, evidence: Dict[str, bool]) -> List[Dict]:
        with self._lock:
            predictions = []
            for h_name, h_data in self.hypotheses.items():
                posterior = h_data["prior"]
                for e_type, matches in evidence.items():
                    if e_type in h_data["evidence_likelihoods"]:
                        likelihood = h_data["evidence_likelihoods"][e_type]
                        if matches:
                            posterior *= (likelihood + 0.01) / 0.5
                        else:
                            posterior *= (1 - likelihood + 0.01) / 0.5
                predictions.append({
                    "hypothesis": h_name,
                    "probability": posterior,
                    "confidence": round(min(0.95, posterior * (h_data["count"] + 1) / (self.total_observations + 1)), 4)
                })
            total = sum(p["probability"] for p in predictions)
            for p in predictions:
                if total > 0:
                    p["probability"] = round(p["probability"] / total, 4)
            return sorted(predictions, key=lambda x: -x["probability"])


# ═════════════════════════════════════════════════════════════════
# SIGNAL TRIANGULATION (X-RAY)
# ═════════════════════════════════════════════════════════════════

class SignalTriangulator:
    """
    Triangulate device locations using RSSI signal strength.
    
    Uses trilateration algorithm:
    - Record signal strength from multiple angles
    - Estimate distance using path loss model
    - Triangulate position
    """
    
    _MAX_MACS = 1000
    _MAX_PER_MAC = 100
    _MAX_CALIBRATIONS = 1000

    def __init__(self):
        self._lock = threading.Lock()
        self.signal_history: Dict[str, List[Dict]] = defaultdict(list)
        self._mac_order: List[str] = []
        self.calibration: Dict[str, Dict] = {}
        self._cal_order: List[str] = []
        self.path_loss_exponent = 2.5
    
    def record_signal(self, mac: str, rssi: float, timestamp: datetime = None):
        mac = _sanitize(mac, 64)
        if timestamp is None:
            timestamp = datetime.now(timezone.utc)
        with self._lock:
            if mac not in self.signal_history or len(self.signal_history[mac]) == 0:
                if mac not in self._mac_order:
                    self._mac_order.append(mac)
            
            self.signal_history[mac].append({
                "rssi": rssi,
                "timestamp": timestamp.timestamp(),
                "distance": self._estimate_distance(rssi)
            })
            
            if len(self.signal_history[mac]) > self._MAX_PER_MAC:
                self.signal_history[mac] = self.signal_history[mac][-self._MAX_PER_MAC // 2:]
            
            while len(self._mac_order) > self._MAX_MACS:
                old_mac = self._mac_order.pop(0)
                self.signal_history.pop(old_mac, None)
                self.calibration.pop(old_mac, None)
                if old_mac in self._cal_order:
                    self._cal_order.remove(old_mac)
    
    def _estimate_distance(self, rssi: float, tx_power: float = -59) -> float:
        """
        Estimate distance from RSSI using log-distance path loss model.
        
        d = 10 ^ ((TxPower - RSSI) / (10 * n))
        """
        if rssi >= tx_power:
            return 0.0
        
        ratio = (tx_power - rssi) / (10 * self.path_loss_exponent)
        distance = pow(10, ratio)
        
        return min(distance, 50.0)  # Cap at 50 meters
    
    def get_distance(self, mac: str) -> Optional[float]:
        with self._lock:
            if mac not in self.signal_history or not self.signal_history[mac]:
                return None
            recent = self.signal_history[mac][-10:]
            distances = [m["distance"] for m in recent]
        return round(median(distances), 3)
    
    def get_movement_vector(self, mac: str, window_seconds: int = 30) -> Optional[Dict]:
        with self._lock:
            if mac not in self.signal_history:
                return None
            history = list(self.signal_history[mac])
        now = datetime.now(timezone.utc).timestamp()
        recent = [h for h in history if now - h["timestamp"] <= window_seconds]
        
        recent = [h for h in history if now - h["timestamp"] <= window_seconds]
        
        if len(recent) < 3:
            return None
        
        distances = [h["distance"] for h in recent]
        rssis = [h["rssi"] for h in recent]
        
        # Check for movement
        if HAS_NUMPY:
            dist_arr = np.array(distances)
            rssi_arr = np.array(rssis)
            
            dist_trend = np.polyfit(range(len(dist_arr)), dist_arr, 1)[0]
            rssi_var = np.var(rssi_arr)
        else:
            # Simple fallback
            dist_trend = (distances[-1] - distances[0]) / len(distances)
            rssi_var = sum((r - mean(rssis)) ** 2 for r in rssis) / len(rssis)
        
        moving = abs(dist_trend) > 0.1 or rssi_var > 10
        
        return {
            "moving": moving,
            "distance_trend": round(dist_trend, 4),
            "rssi_variance": round(rssi_var, 4),
            "current_distance": round(distances[-1], 3),
            "approaching": dist_trend < -0.1 if moving else False,
            "departing": dist_trend > 0.1 if moving else False
        }
    
    def triangulate_position(self, readings: List[Dict]) -> Optional[Dict]:
        """
        Triangulate position using multiple signal readings.
        
        readings: [{"mac": str, "rssi": float, "x": float, "y": float}, ...]
        
        Uses weighted centroid for robust estimation.
        """
        if len(readings) < 3:
            # Not enough for triangulation, return centroid
            if not readings:
                return None
            avg_x = mean(r.get("x", 0) for r in readings)
            avg_y = mean(r.get("y", 0) for r in readings)
            return {"x": avg_x, "y": avg_y, "confidence": 0.3}
        
        # Weight by distance (closer = more weight)
        weights = []
        positions = []
        
        for reading in readings:
            distance = self._estimate_distance(reading["rssi"])
            weight = 1.0 / (distance + 0.1)  # Avoid div by zero
            weights.append(weight)
            positions.append((reading.get("x", 0), reading.get("y", 0)))
        
        # Weighted centroid
        total_weight = sum(weights)
        x = sum(w * p[0] for w, p in zip(weights, positions)) / total_weight
        y = sum(w * p[1] for w, p in zip(weights, positions)) / total_weight
        
        # Confidence based on variance of distances
        distances = [self._estimate_distance(r["rssi"]) for r in readings]
        distance_var = sum((d - mean(distances)) ** 2 for d in distances) / len(distances)
        confidence = max(0.3, min(0.95, 1.0 - distance_var / 100))
        
        return {
            "x": round(x, 4),
            "y": round(y, 4),
            "confidence": round(confidence, 4),
            "method": "weighted_centroid",
            "readings_used": len(readings)
        }


# ═════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH (PERFECT MEMORY)
# ═════════════════════════════════════════════════════════════════

class KnowledgeGraph:
    """
    Knowledge graph for Perfect Memory.
    
    Entities connected by relationships:
    - (Person) --[met]--> (Location)
    - (Person) --[said]--> (Statement)
    - (Activity) --[at]--> (Time)
    
    Enables semantic queries like:
    - "Who did I see at the coffee shop?"
    - "What did Sarah say about work?"
    """
    
    _MAX_ENTITIES = 5000
    _MAX_RELATIONS = 10000
    _MAX_MENTIONS = 100

    def __init__(self):
        self._lock = threading.Lock()
        self.entities: Dict[str, Dict] = {}
        self.relations: List[Dict] = []
        self.entity_index: Dict[str, Set[str]] = defaultdict(set)
        self.relation_index: Dict[str, Set[int]] = defaultdict(set)
    
    def add_entity(self, entity_type: str, name: str, attributes: Dict = None) -> str:
        entity_type = _sanitize(entity_type, 100)
        name = _sanitize_content(name, 200)
        entity_id = hashlib.sha256(f"{entity_type}:{name}".encode()).hexdigest()[:12]
        if attributes is None:
            attributes = {}
        attributes = {str(k): str(v)[:500] for k, v in list(attributes.items())[:50]}
        with self._lock:
            if entity_id not in self.entities:
                if len(self.entities) >= self._MAX_ENTITIES:
                    oldest_id = next(iter(self.entities))
                    old_type = self.entities[oldest_id].get("type", "")
                    self.entity_index[old_type].discard(oldest_id)
                    del self.entities[oldest_id]
                self.entities[entity_id] = {
                    "id": entity_id,
                    "type": entity_type,
                    "name": name[:200],
                    "attributes": dict(attributes),
                    "created": datetime.now(timezone.utc).isoformat(),
                    "mentioned": []
                }
                self.entity_index[entity_type].add(entity_id)
            else:
                self.entities[entity_id]["attributes"].update(attributes)
                self.entities[entity_id]["mentioned"].append(datetime.now(timezone.utc).isoformat())
                if len(self.entities[entity_id]["mentioned"]) > self._MAX_MENTIONS:
                    self.entities[entity_id]["mentioned"] = self.entities[entity_id]["mentioned"][-self._MAX_MENTIONS:]
        
        return entity_id
    
    def add_relation(self, subject_id: str, predicate: str, obj_id: str, 
                     context: Dict = None) -> int:
        subject_id = _sanitize(subject_id, 64)
        predicate = _sanitize(predicate, 100)
        obj_id = _sanitize(obj_id, 64)
        if context is None:
            context = {}
        safe_context = {str(k): str(v)[:500] for k, v in list(context.items())[:50]}
        with self._lock:
            if len(self.relations) >= self._MAX_RELATIONS:
                removed = self.relations.pop(0)
                old_pred = removed.get("predicate", "")
                self.relation_index[old_pred].discard(removed.get("id", 0))
            relation_id = len(self.relations)
            
            relation = {
                "id": relation_id,
                "subject": subject_id,
                "predicate": predicate,
                "object": obj_id,
                "context": safe_context,
                "timestamp": datetime.now(timezone.utc).isoformat()
            }
            
            self.relations.append(relation)
            self.relation_index[predicate].add(relation_id)
            
            if subject_id in self.entities:
                self.entities[subject_id]["mentioned"].append(relation_id)
            if obj_id in self.entities:
                self.entities[obj_id]["mentioned"].append(relation_id)
            
            return relation_id
    
    def query(self, query: Dict) -> List[Dict]:
        with self._lock:
            results = []
            entity_type = query.get("entity_type")
            name_contains = query.get("name_contains", "").lower()
            name_pattern = re.compile(r'\b' + re.escape(name_contains) + r'\b', re.IGNORECASE) if name_contains else None
            
            candidate_ids = set()
            if entity_type and entity_type in self.entity_index:
                candidate_ids = set(self.entity_index[entity_type])
            else:
                candidate_ids = set(self.entities.keys())
            
            if name_pattern:
                candidate_ids = {
                    eid for eid in candidate_ids 
                    if name_pattern.search(self.entities[eid]["name"])
                }
            
            for rel_query in query.get("relations", []):
                predicate = rel_query.get("predicate")
                obj_type = rel_query.get("object_type")
                obj_name = rel_query.get("object_name", "").lower()
                obj_pattern = re.compile(r'\b' + re.escape(obj_name) + r'\b', re.IGNORECASE) if obj_name else None
                
                matching = set()
                for rid in self.relation_index.get(predicate, []):
                    if rid >= len(self.relations):
                        continue
                    relation = self.relations[rid]
                    if relation["subject"] not in candidate_ids:
                        continue
                    if obj_type:
                        obj = self.entities.get(relation["object"], {})
                        if obj.get("type") != obj_type:
                            continue
                    if obj_pattern:
                        obj = self.entities.get(relation["object"], {})
                        if not obj_pattern.search(obj.get("name", "")):
                            continue
                    matching.add(relation["subject"])
                candidate_ids &= matching
            
            for eid in candidate_ids:
                entity = self.entities[eid]
                related = self._get_related_unlocked(eid)
                results.append({
                    "entity": copy.deepcopy(entity),
                    "relations": copy.deepcopy(related),
                    "confidence": round(min(0.95, 0.5 + len(entity["mentioned"]) * 0.05), 4)
                })
            results.sort(key=lambda x: -x["confidence"])
            return results
    
    def _get_related(self, entity_id: str) -> Dict[str, List[Dict]]:
        with self._lock:
            return self._get_related_unlocked(entity_id)
    
    def _get_related_unlocked(self, entity_id: str) -> Dict[str, List[Dict]]:
        related = defaultdict(list)
        for relation in self.relations:
            if relation["subject"] == entity_id:
                obj = self.entities.get(relation["object"], {})
                related[relation["predicate"]].append({
                    "entity": obj,
                    "context": relation["context"],
                    "timestamp": relation["timestamp"]
                })
            elif relation["object"] == entity_id:
                subj = self.entities.get(relation["subject"], {})
                related[f"inverse_{relation['predicate']}"].append({
                    "entity": subj,
                    "context": relation["context"],
                    "timestamp": relation["timestamp"]
                })
        return dict(related)
    
    def get_timeline(self, entity_id: str = None, entity_type: str = None,
                     start: datetime = None, end: datetime = None) -> List[Dict]:
        with self._lock:
            events = []
            for relation in self.relations:
                if entity_id and relation["subject"] != entity_id and relation["object"] != entity_id:
                    continue
                try:
                    ts = datetime.fromisoformat(relation["timestamp"])
                    if start and ts < start:
                        continue
                    if end and ts > end:
                        continue
                except (ValueError, TypeError):
                    pass
                events.append({
                    "subject": copy.deepcopy(self.entities.get(relation["subject"], {})),
                    "predicate": relation["predicate"],
                    "object": copy.deepcopy(self.entities.get(relation["object"], {})),
                    "timestamp": relation["timestamp"],
                    "context": copy.deepcopy(relation.get("context", {}))
                })
            events.sort(key=lambda x: x["timestamp"])
            return events


# ═════════════════════════════════════════════════════════════════
# BEHAVIOR MARKOV MODEL (INTENT SEER)
# ═════════════════════════════════════════════════════════════════

class MarkovBehaviorModel:
    """
    Markov Decision Process for behavior prediction.
    
    States: observed situations/states
    Actions: possible transitions
    Rewards: learned from outcomes
    
    Predicts: next action given current state
    """
    
    _MAX_STATES = 500
    _MAX_HISTORY = 1000

    def __init__(self):
        self._lock = threading.Lock()
        self.transitions: Dict[str, Dict[str, Dict[str, int]]] = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))
        self.state_counts: Dict[str, int] = defaultdict(int)
        self._state_order: List[str] = []
        # Action counts per state
        self.action_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        # Rewards: (state, action) -> average_reward
        self.rewards: Dict[Tuple[str, str], float] = {}
        # History for backtracking
        self.history: List[Dict] = []
        self._max_history = self._MAX_HISTORY
    
    def record(self, state: str, action: str, next_state: str, reward: float = 0.0):
        state = _sanitize(state, 200)
        action = _sanitize(action, 200)
        next_state = _sanitize(next_state, 200)
        with self._lock:
            self.transitions[state][action][next_state] += 1
            self.state_counts[state] += 1
            if state not in self._state_order:
                self._state_order.append(state)
            self.action_counts[state][action] += 1
            
            while len(self._state_order) > self._MAX_STATES:
                old_state = self._state_order.pop(0)
                self.transitions.pop(old_state, None)
                self.state_counts.pop(old_state, None)
                self.action_counts.pop(old_state, None)
                self.rewards = {k: v for k, v in self.rewards.items() if k[0] != old_state}
            
            key = (state, action)
            if key in self.rewards:
                old = self.rewards[key]
                count = sum(self.transitions[state][action].values())
                self.rewards[key] = old + (reward - old) / count
            else:
                self.rewards[key] = reward
            
            self.history.append({
                "state": state,
                "action": action,
                "next_state": next_state,
                "reward": reward,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if len(self.history) > self._max_history:
                self.history = self.history[-self._max_history // 2:]
    
    def predict_next_state(self, state: str, top_k: int = 3) -> List[Dict]:
        with self._lock:
            if state not in self.transitions:
                return []
            predictions = []
            total_transitions = sum(
                sum(self.transitions[state][action].values())
                for action in self.transitions[state]
            )
            for action, next_states in self.transitions[state].items():
                for next_state, count in next_states.items():
                    probability = count / max(1, total_transitions)
                    reward = self.rewards.get((state, action), 0.0)
                    predictions.append({
                        "action": action,
                        "next_state": next_state,
                        "probability": round(probability, 4),
                        "count": count,
                        "expected_reward": round(reward, 4),
                        "value": round(probability * (1 + reward), 4)
                    })
            predictions.sort(key=lambda x: -x["value"])
            return predictions[:top_k]
    
    def get_policy(self, state: str) -> Optional[str]:
        predictions = self.predict_next_state(state, top_k=1)
        if predictions:
            return predictions[0]["action"]
        return None
    
    def simulate(self, start_state: str, steps: int = 10) -> List[str]:
        states = [start_state]
        current = start_state
        for _ in range(steps):
            predictions = self.predict_next_state(current, top_k=1)
            if not predictions:
                break
            r = random.random()
            cumulative = 0
            all_preds = self.predict_next_state(current, top_k=5)
            for pred in all_preds:
                cumulative += pred["probability"]
                if r < cumulative:
                    states.append(pred["next_state"])
                    current = pred["next_state"]
                    break
            else:
                if predictions:
                    states.append(predictions[0]["next_state"])
                    current = predictions[0]["next_state"]
        return states
    
    def get_value_estimate(self, state: str, gamma: float = 0.9, _depth: int = 0) -> float:
        if _depth > 10 or state not in self.action_counts:
            return 0.0
        # Snapshot the data we need under the lock, then run the recursion
        # outside it. The previous code held a non-reentrant lock while
        # recursively calling self.get_value_estimate — instant deadlock.
        with self._lock:
            actions = list(self.action_counts[state])
            rewards_snap = {a: self.rewards.get((state, a), 0.0) for a in actions}
            transitions_snap = {
                a: dict(self.transitions[state].get(a, {})) for a in actions
            }

        best_value = 0.0
        for action in actions:
            immediate_reward = rewards_snap[action]
            future_value = 0.0
            outcomes = transitions_snap[action]
            total_from_action = sum(outcomes.values())
            if total_from_action > 0:
                for next_state, count in outcomes.items():
                    p = count / total_from_action
                    future_value += p * self.get_value_estimate(next_state, gamma * 0.9, _depth + 1)
            value = immediate_reward + gamma * future_value
            best_value = max(best_value, value)
        return round(best_value, 4)


# ═════════════════════════════════════════════════════════════════
# REINFORCEMENT LEARNING DECIDER (SHADOW COUNCIL)
# ═════════════════════════════════════════════════════════════════

class ReinforcementLearner:
    """
    Q-Learning for decision making.
    
    Learns optimal decisions from experience.
    
    Q(s,a) = Q(s,a) + alpha * (R + gamma * max(Q(s',a')) - Q(s,a))
    """
    
    _MAX_Q_TABLE = 50000
    _MAX_EXPERIENCES = 1000

    def __init__(self, learning_rate: float = 0.1, discount_factor: float = 0.95):
        self._lock = threading.Lock()
        self.q_table: Dict[Tuple[str, str], float] = defaultdict(float)
        self._q_order: List[Tuple[str, str]] = []
        self.alpha = learning_rate
        self.gamma = discount_factor
        self.experiences: List[Dict] = []
    
    def encode_state(self, features: Dict) -> str:
        """Convert feature dict to state string."""
        # Discretize continuous features
        parts = []
        for k, v in sorted(features.items()):
            if isinstance(v, float):
                parts.append(f"{k}:{v:.2f}")
            elif isinstance(v, bool):
                parts.append(f"{k}:{int(v)}")
            elif v is not None:
                parts.append(f"{k}:{v}")
        return "|".join(parts)
    
    def get_q(self, state: str, action: str) -> float:
        """Get Q-value for state-action pair."""
        return self.q_table.get((state, action), 0.0)
    
    def best_action(self, state: str, possible_actions: List[str] = None,
                    epsilon: float = 0.1) -> Tuple[str, float]:
        with self._lock:
            if possible_actions is None:
                actions = set()
                for (s, a), q in self.q_table.items():
                    if s == state:
                        actions.add(a)
                possible_actions = list(actions)
        
        if not possible_actions:
            return None, 0.0
        
        if random.random() < epsilon:
            action = random.choice(possible_actions)
            return action, self.get_q(state, action)
        
        best_action = max(possible_actions, key=lambda a: self.get_q(state, a))
        best_q = self.get_q(state, best_action)
        
        return best_action, best_q
    
    def update(self, state: str, action: str, reward: float, next_state: str,
               possible_next_actions: List[str] = None):
        state = _sanitize(state, 200)
        action = _sanitize(action, 200)
        next_state = _sanitize(next_state, 200)
        with self._lock:
            current_q = self.q_table.get((state, action), 0.0)
            if possible_next_actions:
                max_next_q = max(self.q_table.get((next_state, a), 0.0) for a in possible_next_actions)
            else:
                max_next_q = 0.0
            new_q = current_q + self.alpha * (reward + self.gamma * max_next_q - current_q)
            self.q_table[(state, action)] = round(new_q, 6)
            if (state, action) not in self._q_order:
                self._q_order.append((state, action))
            while len(self._q_order) > self._MAX_Q_TABLE:
                old_key = self._q_order.pop(0)
                self.q_table.pop(old_key, None)
            self.experiences.append({
                "state": state,
                "action": action,
                "reward": round(reward, 6),
                "next_state": next_state,
                "timestamp": datetime.now(timezone.utc).isoformat()
            })
            if len(self.experiences) > self._MAX_EXPERIENCES:
                self.experiences = self.experiences[-self._MAX_EXPERIENCES // 2:]
    
    def get_recommendation(self, state: str, possible_actions: List[str],
                           exploration_rate: float = 0.0) -> Dict:
        with self._lock:
            action_values = []
            for action in possible_actions:
                q_value = self.q_table.get((state, action), 0.0)
                count = sum(1 for e in self.experiences 
                           if e["state"] == state and e["action"] == action)
                action_values.append({
                    "action": action,
                    "q_value": round(q_value, 4),
                    "confidence": round(min(0.95, count / 10), 4),
                    "experiences": count
                })
            action_values.sort(key=lambda x: -x["q_value"])
            best = action_values[0] if action_values else {"action": None, "q_value": 0, "confidence": 0}
        best_action, best_q = self.best_action(state, possible_actions, exploration_rate)
        return {
            "recommended_action": best_action,
            "q_value": round(best_q, 4),
            "alternatives": copy.deepcopy(action_values[:5]),
            "confidence": best["confidence"],
            "total_experiences": len(self.experiences)
        }
    
    def learn_from_outcome(self, state: str, action: str, outcome: str,
                          satisfaction: int, possible_actions: List[str]):
        satisfaction = max(1, min(10, satisfaction)) if satisfaction else 5
        reward = (satisfaction - 1) / 9.0 if satisfaction else 0.5
        next_state = f"outcome:{_sanitize(outcome, 200)}"
        self.update(state, action, round(reward, 4), next_state, possible_actions)
    
    def save_q_table(self, path: Path):
        with self._lock:
            data = {
                "q_table": {f"{s}|{a}": round(q, 6) for (s, a), q in self.q_table.items()},
                "experiences": list(self.experiences[-1000:]),
                "alpha": self.alpha,
                "gamma": self.gamma
            }
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(path))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Save Q-table %s error: %s", path, type(e).__name__)
            try:
                os.unlink(str(tmp))
            except OSError:
                pass
    
    def load_q_table(self, path: Path):
        """Load Q-table from file."""
        if not path.exists():
            return
        
        try:
            data = json.loads(path.read_text())
            for key, q in data.get("q_table", {}).items():
                parts = key.split("|", 1)
                if len(parts) == 2:
                    self.q_table[(parts[0], parts[1])] = q
            self.experiences = data.get("experiences", [])
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Load Q-table error: %s", type(e).__name__)


# ═════════════════════════════════════════════════════════════════
# ENVIRONMENT MODEL (OMNI SENSE)
# ═════════════════════════════════════════════════════════════════

class EnvironmentModel:
    """
    Comprehensive environment model for Omni Sense.
    
    Fuses all available data:
    - Device presence
    - Network information
    - Activity patterns
    - Time context
    """
    
    _MAX_OBSERVATIONS = 1000
    _MAX_PATTERNS = 200

    def __init__(self):
        self._lock = threading.Lock()
        self.observations: List[Dict] = []
        self.patterns: Dict[str, Dict] = {}
        self.predictions: Dict[str, float] = {}
    
    def observe(self, observation: Dict):
        safe_obs = {str(k)[:50]: str(v)[:1000] for k, v in list(observation.items())[:50]}
        safe_obs["timestamp"] = observation.get("timestamp", datetime.now(timezone.utc).isoformat())
        if "type" in observation:
            safe_obs["type"] = _sanitize(str(observation["type"]), 100)
        with self._lock:
            self.observations.append(safe_obs)
            if len(self.observations) > self._MAX_OBSERVATIONS:
                self.observations = self.observations[-self._MAX_OBSERVATIONS // 2:]
            self._update_patterns(safe_obs)
    
    def _update_patterns(self, obs: Dict):
        obs_type = obs.get("type", "unknown")
        if obs_type not in self.patterns:
            if len(self.patterns) >= self._MAX_PATTERNS:
                oldest = next(iter(self.patterns))
                del self.patterns[oldest]
            self.patterns[obs_type] = {
                "count": 0,
                "by_hour": defaultdict(int),
                "by_day": defaultdict(int),
                "by_weekday": defaultdict(int),
                "contexts": []
            }
        pattern = self.patterns[obs_type]
        # Patterns deserialized from JSON lose the defaultdict factory; promote
        # them back so the in-place increments below don't KeyError.
        for sub in ("by_hour", "by_day", "by_weekday"):
            if not isinstance(pattern.get(sub), defaultdict):
                pattern[sub] = defaultdict(int, pattern.get(sub) or {})
        pattern["count"] += 1
        try:
            ts = datetime.fromisoformat(str(obs["timestamp"]))
            pattern["by_hour"][ts.hour] += 1
            pattern["by_day"][ts.date().isoformat()] += 1
            pattern["by_weekday"][ts.weekday()] += 1
        except (ValueError, TypeError):
            pass
        context = {
            "location": obs.get("location"),
            "people": list(obs.get("people", []))[:20],
            "activity": obs.get("activity")
        }
        pattern["contexts"].append(context)
        if len(pattern["contexts"]) > 100:
            pattern["contexts"] = pattern["contexts"][-100:]
    
    def get_environment_snapshot(self, devices: List[Dict], networks: List[Dict],
                                   activities: List[str] = None) -> Dict:
        now = datetime.now(timezone.utc)
        hour = now.hour
        if 6 <= hour < 12:
            time_context = "morning"
        elif 12 <= hour < 18:
            time_context = "afternoon"
        elif 18 <= hour < 22:
            time_context = "evening"
        else:
            time_context = "night"
        location, location_confidence = self._infer_location(networks)
        device_count = len(devices)
        if device_count == 0:
            activity_level = "empty"
        elif device_count < 3:
            activity_level = "quiet"
        elif device_count < 10:
            activity_level = "moderate"
        else:
            activity_level = "busy"
        people_present = self._identify_people(devices)
        with self._lock:
            predictions = self._predict_activities_unlocked(hour, activity_level, location)
            observations_snapshot = list(self.observations[-100:])
        anomalies = self._detect_anomalies(now, location, activity_level, people_present)
        return {
            "timestamp": now.isoformat(),
            "time": {
                "hour": hour,
                "context": time_context,
                "weekday": now.weekday(),
                "date": now.date().isoformat()
            },
            "location": {
                "name": location,
                "confidence": round(location_confidence, 4),
                "network": networks[0].get("ssid") if networks else None
            },
            "devices": {
                "count": device_count,
                "activity_level": activity_level,
                "known": sum(1 for d in devices if d.get("known")),
                "unknown": sum(1 for d in devices if not d.get("known"))
            },
            "people": {
                "present": people_present,
                "count": len(people_present)
            },
            "predictions": predictions,
            "anomalies": anomalies,
            "activities": list(activities or [])
        }
    
    def _infer_location(self, networks: List[Dict]) -> Tuple[str, float]:
        if not networks:
            return "unknown", 0.1
        ssid = networks[0].get("ssid", "").lower()
        location_patterns = {
            "home": ["home", "my", "fam", "residence"],
            "office": ["office", "work", "corp", "business", "company"],
            "cafe": ["cafe", "coffee", "starbucks", "espresso", "dunkin"],
            "public": ["public", "free", "guest", "library", "mall"],
        }
        # Pass 1: strict boundary match — high confidence.
        for location, patterns in location_patterns.items():
            for pattern in patterns:
                if re.search(r'(?:^|[\s_\-])' + re.escape(pattern) + r'(?:$|[\s_\-])', ssid):
                    return location, 0.85
                if re.search(r'\b' + re.escape(pattern) + r'\b', ssid):
                    return location, 0.85
        # Pass 2: substring match (camelCase / concatenated SSIDs like
        # "HomeWiFi" or "OfficeCorp"). Lower confidence since this is fuzzier.
        for location, patterns in location_patterns.items():
            for pattern in patterns:
                if pattern in ssid:
                    return location, 0.65
        return "unknown", 0.3
    
    def _identify_people(self, devices: List[Dict]) -> List[str]:
        """Identify people from devices."""
        people = set()
        
        for device in devices:
            person = device.get("person")
            if person:
                people.add(person)
        
        return list(people)
    
    def _predict_activities_unlocked(self, hour: int, activity_level: str, location: str) -> List[Dict]:
        predictions = []
        for obs_type, pattern in self.patterns.items():
            hour_count = pattern["by_hour"].get(hour, 0)
            total = pattern["count"]
            if total > 0 and hour_count > 0:
                probability = hour_count / total
                if activity_level == "quiet" and hour_count < pattern["by_hour"].get(12, 0):
                    probability *= 1.2
                elif activity_level == "busy" and hour_count > pattern["by_hour"].get(3, 0):
                    probability *= 1.2
                predictions.append({
                    "activity": obs_type,
                    "probability": round(min(1.0, probability), 4),
                    "count": hour_count
                })
        predictions.sort(key=lambda x: -x["probability"])
        return predictions[:5]
    
    def _detect_anomalies(self, now: datetime, location: str, 
                          activity_level: str, people: List[str]) -> List[Dict]:
        """Detect environmental anomalies."""
        anomalies = []
        
        # Check hour anomaly
        hour = now.hour
        for obs_type, pattern in self.patterns.items():
            if pattern["count"] < 10:
                continue
            
            if pattern["by_hour"].get(hour, 0) == 0 and pattern["count"] > 50:
                # Never observed this hour before with significant observations
                anomalies.append({
                    "type": "unusual_hour",
                    "activity": obs_type,
                    "description": f"{obs_type} not usually observed at {hour}:00",
                    "severity": "info"
                })
        
        # Check location anomaly
        historic_locations = set()
        for obs in self.observations[-100:]:
            loc = obs.get("location")
            if loc:
                historic_locations.add(loc)
        
        if location not in historic_locations and len(historic_locations) > 3:
            anomalies.append({
                "type": "new_location",
                "description": f"Rare location: {location}",
                "severity": "info"
            })
        
        return anomalies


# Helper function
def median(values: List[float]) -> float:
    """Calculate median without numpy."""
    if not values:
        return 0.0
    sorted_values = sorted(values)
    n = len(sorted_values)
    if n % 2 == 0:
        return (sorted_values[n//2 - 1] + sorted_values[n//2]) / 2
    else:
        return sorted_values[n//2]


class TemporalPatternEngine:
    """
    Temporal pattern mining for Oracle predictions.
    
    Extracts:
    - Cyclical patterns (daily, weekly, monthly)
    - Trend detection (increasing/decreasing activity)
    - Periodicity detection (autocorrelation-based)
    - Time-of-day probability distributions
    """

    _MAX_DAY_COUNTS = 365
    _MAX_TRANSITIONS = 5000
    _MAX_TYPE_KEYS = 100

    def __init__(self):
        self._lock = threading.Lock()
        self.hour_counts: Dict[int, int] = defaultdict(int)
        self.day_counts: Dict[int, int] = defaultdict(int)
        self._day_order: List[int] = []
        self.weekday_counts: Dict[int, int] = defaultdict(int)
        self.type_hour_counts: Dict[str, Dict[int, int]] = defaultdict(lambda: defaultdict(int))
        self.type_day_counts: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self.transitions: Dict[Tuple[int, int], Dict[str, int]] = defaultdict(lambda: defaultdict(int))
        self._transition_order: List[Tuple[int, int]] = []
        self.total_observations = 0
        self._trend_window: List[Tuple[str, int]] = []

    def observe(self, observation: Dict):
        ts = observation.get("timestamp")
        if not ts:
            return
        try:
            dt = datetime.fromisoformat(str(ts))
        except (ValueError, TypeError):
            return
        obs_type = _sanitize(str(observation.get("type", "unknown")), 100)
        with self._lock:
            self.hour_counts[dt.hour] += 1
            day_ord = dt.toordinal()
            self.day_counts[day_ord] += 1
            if day_ord not in self.day_counts or self.day_counts[day_ord] == 1:
                if day_ord not in self._day_order:
                    self._day_order.append(day_ord)
            self.weekday_counts[dt.weekday()] += 1
            self.type_hour_counts[obs_type][dt.hour] += 1
            self.type_day_counts[obs_type][dt.date().isoformat()] += 1
            self.total_observations += 1
            while len(self._day_order) > self._MAX_DAY_COUNTS:
                old_day = self._day_order.pop(0)
                self.day_counts.pop(old_day, None)
            while len(self._transition_order) > self._MAX_TRANSITIONS:
                old_key = self._transition_order.pop(0)
                self.transitions.pop(old_key, None)
            if len(self.type_day_counts) > self._MAX_TYPE_KEYS:
                oldest_type = next(iter(self.type_day_counts))
                del self.type_day_counts[oldest_type]
            if len(self.type_hour_counts) > self._MAX_TYPE_KEYS:
                oldest_type = next(iter(self.type_hour_counts))
                del self.type_hour_counts[oldest_type]
            hour_slot = dt.hour // 4
            self._trend_window.append((dt.isoformat(), hour_slot))
            if len(self._trend_window) > 2000:
                self._trend_window = self._trend_window[-1000:]

    def forecast_hour(self, target_hour: int, obs_type: str = None) -> Dict:
        with self._lock:
            total = sum(self.hour_counts.values()) or 1
            hour_prob = self.hour_counts.get(target_hour, 0) / total
            if obs_type and obs_type in self.type_hour_counts:
                type_total = sum(self.type_hour_counts[obs_type].values()) or 1
                type_prob = self.type_hour_counts[obs_type].get(target_hour, 0) / type_total
                prob = 0.5 * hour_prob + 0.5 * type_prob
            else:
                prob = hour_prob
        return {
            "hour": target_hour,
            "probability": round(min(1.0, prob), 4),
            "observations_at_hour": self.hour_counts.get(target_hour, 0),
            "total_observations": self.total_observations,
        }

    def forecast_weekday(self, target_weekday: int) -> Dict:
        with self._lock:
            total = sum(self.weekday_counts.values()) or 1
            prob = self.weekday_counts.get(target_weekday, 0) / total
        return {
            "weekday": target_weekday,
            "probability": round(prob, 4),
            "observations": self.weekday_counts.get(target_weekday, 0),
        }

    def detect_trend(self, window: int = 14) -> Dict:
        with self._lock:
            if len(self.day_counts) < 3:
                return {"trend": "unknown", "slope": 0.0, "confidence": 0.0}
            days = sorted(self.day_counts.keys())[-window:]
            counts = [self.day_counts[d] for d in days]
        if len(counts) < 3:
            return {"trend": "unknown", "slope": 0.0, "confidence": 0.0}
        n = len(counts)
        x_mean = (n - 1) / 2
        y_mean = sum(counts) / n
        slope_num = sum((i - x_mean) * (counts[i] - y_mean) for i in range(n))
        slope_den = sum((i - x_mean) ** 2 for i in range(n))
        slope = slope_num / slope_den if slope_den else 0.0
        if slope > 0.5:
            trend = "increasing"
        elif slope < -0.5:
            trend = "decreasing"
        else:
            trend = "stable"
        return {
            "trend": trend,
            "slope": round(slope, 4),
            "confidence": round(min(0.95, max(0.1, abs(slope) * 10 / max(1, y_mean))), 4),
            "window_days": n,
        }

    def detect_periodicity(self) -> List[Dict]:
        with self._lock:
            if self.total_observations < 20:
                return []
            hourly = [self.hour_counts.get(h, 0) for h in range(24)]
        total = sum(hourly) or 1
        n = len(hourly)
        mean_val = sum(hourly) / n
        peaks = []
        for lag in range(1, 13):
            num = sum((hourly[i] - mean_val) * (hourly[(i + lag) % n] - mean_val) for i in range(n))
            den = sum((hourly[i] - mean_val) ** 2 for i in range(n))
            autocorr = num / den if den > 0 else 0
            if autocorr > 0.3:
                peaks.append({"lag_hours": lag, "autocorrelation": round(autocorr, 4)})
        peaks.sort(key=lambda x: -x["autocorrelation"])
        return peaks[:3]

    def predict_next_activity(self, current_hour: int = None) -> Dict:
        if current_hour is None:
            current_hour = datetime.now(timezone.utc).hour
        next_hours = [(current_hour + i) % 24 for i in range(1, 7)]
        predictions = []
        for h in next_hours:
            fc = self.forecast_hour(h)
            predictions.append(fc)
        best = max(predictions, key=lambda x: x["probability"])
        with self._lock:
            peak_hour = max(self.hour_counts, key=self.hour_counts.get) if self.hour_counts else 12
        return {
            "predicted_hour": best["hour"],
            "predicted_probability": round(best["probability"], 4),
            "next_6_hours": predictions,
            "peak_hour": peak_hour,
        }


class KalmanFilter1D:
    """
    1D Kalman Filter for X-Ray signal smoothing.
    
    Tracks RSSI signals over time, producing smooth distance estimates
    with uncertainty bounds. Uses standard Kalman update equations:
    
    Predict: x_pred = A*x + B*u, P_pred = A*P*A^T + Q
    Update:  K = P_pred * H^T * (H*P_pred*H^T + R)^-1
             x = x_pred + K*(z - H*x_pred)
             P = (I - K*H) * P_pred
    """

    def __init__(self, process_noise: float = 0.01, measurement_noise: float = 1.0):
        self.state = 0.0
        self.covariance = 1.0
        self.process_noise = process_noise
        self.measurement_noise = measurement_noise

    def predict(self, control: float = 0.0):
        self.state += control
        self.covariance += self.process_noise
        return self.state, self.covariance

    def update(self, measurement: float) -> Tuple[float, float]:
        kalman_gain = self.covariance / (self.covariance + self.measurement_noise)
        self.state = self.state + kalman_gain * (measurement - self.state)
        self.covariance = (1 - kalman_gain) * self.covariance
        return self.state, self.covariance

    def filter_sequence(self, measurements: List[float]) -> List[Tuple[float, float]]:
        results = []
        for m in measurements:
            self.predict()
            state, cov = self.update(m)
            results.append((state, cov))
        return results

    @property
    def uncertainty(self) -> float:
        return math.sqrt(max(0, self.covariance))

    @property
    def estimated_value(self) -> float:
        return self.state


class MultiArmedBandit:
    """
    Multi-Armed Bandit (UCB1 + Thompson Sampling) for Social Radar.
    
    Determines optimal engagement strategy with people and situations.
    Tracks arm rewards and selects using Upper Confidence Bound or
    Thompson Sampling for exploration-exploitation trade-off.
    """

    _MAX_ARMS = 500

    def __init__(self, algorithm: str = "ucb1"):
        self._lock = threading.Lock()
        self.algorithm = algorithm
        self.arms: Dict[str, Dict] = {}
        self.total_pulls = 0

    def add_arm(self, arm_id: str, prior_mean: float = 0.5, description: str = ""):
        arm_id = _sanitize(arm_id, 100)
        description = _sanitize_content(description, 500)
        with self._lock:
            if len(self.arms) >= self._MAX_ARMS:
                oldest = next(iter(self.arms))
                del self.arms[oldest]
            self.arms[arm_id] = {
                "pulls": 0,
                "sum_rewards": 0.0,
                "sum_sq_rewards": 0.0,
                "alpha": 1.0 + max(0.001, prior_mean),
                "beta": 1.0 + max(0.001, 1.0 - prior_mean),
                "description": description,
            }

    def select(self, available_arms: List[str] = None) -> Tuple[str, float]:
        with self._lock:
            # Lazily register any candidates not yet in the bandit so callers
            # can pass action names that haven't been seen before (e.g. a
            # newly-tracked person).
            if available_arms:
                for arm_id in available_arms:
                    if arm_id not in self.arms:
                        if len(self.arms) >= self._MAX_ARMS:
                            oldest = next(iter(self.arms))
                            del self.arms[oldest]
                        self.arms[arm_id] = {
                            "pulls": 0,
                            "sum_rewards": 0.0,
                            "sum_sq_rewards": 0.0,
                            "alpha": 1.0,
                            "beta": 1.0,
                            "description": "",
                        }
            candidates = available_arms or list(self.arms.keys())
        if not candidates:
            return None, 0.0
        self.total_pulls += 1
        if self.algorithm == "ucb1":
            return self._select_ucb1(candidates)
        else:
            return self._select_thompson(candidates)

    def _select_ucb1(self, candidates: List[str]) -> Tuple[str, float]:
        best_arm = None
        best_score = float("-inf")

        for arm_id in candidates:
            arm = self.arms.get(arm_id)
            if arm is None:
                # Defensive: select() above registers unknown arms, so this
                # shouldn't fire — but if a caller skipped select() and reached
                # here via another path, treat unknown arms as unexplored.
                return arm_id, float("inf")
            if arm["pulls"] == 0:
                return arm_id, float("inf")

            mean = arm["sum_rewards"] / arm["pulls"]
            ucb = mean + math.sqrt(2 * math.log(self.total_pulls) / arm["pulls"])
            if ucb > best_score:
                best_score = ucb
                best_arm = arm_id

        return best_arm, best_score

    def _select_thompson(self, candidates: List[str]) -> Tuple[str, float]:
        best_arm = None
        best_sample = float("-inf")

        for arm_id in candidates:
            arm = self.arms.get(arm_id)
            if arm is None:
                return arm_id, float("inf")
            sample = random.betavariate(arm["alpha"], arm["beta"])
            if sample > best_sample:
                best_sample = sample
                best_arm = arm_id

        return best_arm, best_sample

    def update(self, arm_id: str, reward: float):
        arm_id = _sanitize(arm_id, 100)
        with self._lock:
            if arm_id not in self.arms:
                self.add_arm(arm_id)
            arm = self.arms[arm_id]
            arm["pulls"] += 1
            arm["sum_rewards"] += reward
            arm["sum_sq_rewards"] += reward ** 2
            arm["alpha"] += max(0.0, reward)
            arm["beta"] += max(0.0, 1.0 - reward)
            self.total_pulls += 1

    def get_stats(self) -> Dict:
        with self._lock:
            arms_copy = dict(self.arms)
        return {
            "total_pulls": self.total_pulls,
            "arms": {
                arm_id: {
                    "pulls": arm["pulls"],
                    "mean_reward": round(arm["sum_rewards"] / max(1, arm["pulls"]), 4),
                    "confidence": round(min(0.95, arm["pulls"] / 10), 4),
                    "description": arm["description"],
                }
                for arm_id, arm in arms_copy.items()
            },
        }


class AnomalyDetector:
    """
    Statistical anomaly detection for Omni Sense.
    
    Uses:
    - Z-score for point anomalies
    - Moving average deviation for contextual anomalies
    - Grubbs' test for outliers
    - Entropy-based novelty detection
    """

    _MAX_BASELINES = 1000
    _MAX_ANOMALY_LOG = 500

    def __init__(self, window_size: int = 50, z_threshold: float = 2.5):
        self._lock = threading.Lock()
        self.window_size = window_size
        self.z_threshold = z_threshold
        self.baseline: Dict[str, List[float]] = defaultdict(list)
        self.anomaly_log: List[Dict] = []
    
    def update_baseline(self, metric: str, value: float):
        metric = _sanitize(metric, 100)
        with self._lock:
            self.baseline[metric].append(value)
            if len(self.baseline[metric]) > self.window_size:
                self.baseline[metric] = self.baseline[metric][-self.window_size:]
            if len(self.baseline) > self._MAX_BASELINES:
                oldest = next(iter(self.baseline))
                del self.baseline[oldest]

    def check_zscore(self, metric: str, value: float) -> Optional[Dict]:
        metric = _sanitize(metric, 100)
        with self._lock:
            history = list(self.baseline.get(metric, []))
        if len(history) < 5:
            return None
        mean_val = sum(history) / len(history)
        variance = sum((x - mean_val) ** 2 for x in history) / len(history)
        std_val = math.sqrt(variance) if variance > 0 else 0.001
        z = (value - mean_val) / std_val
        if abs(z) > self.z_threshold:
            anomaly = {
                "metric": metric,
                "value": round(value, 4),
                "z_score": round(z, 4),
                "expected_range": (round(mean_val - self.z_threshold * std_val, 4), round(mean_val + self.z_threshold * std_val, 4)),
                "severity": "critical" if abs(z) > 3.0 else "warning",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with self._lock:
                self.anomaly_log.append(anomaly)
                if len(self.anomaly_log) > self._MAX_ANOMALY_LOG:
                    self.anomaly_log = self.anomaly_log[-self._MAX_ANOMALY_LOG // 2:]
            return anomaly
        return None

    def check_entropy_novelty(self, metric: str, value: float, bins: int = 10) -> Optional[Dict]:
        metric = _sanitize(metric, 100)
        with self._lock:
            history = list(self.baseline.get(metric, []))
        if len(history) < 10:
            return None
        min_val = min(history)
        max_val = max(history)
        if max_val == min_val:
            return None
        extend_ratio = 0.1 * (max_val - min_val)
        if value > max_val + extend_ratio or value < min_val - extend_ratio:
            bin_width = (max_val - min_val) / bins
            counts = defaultdict(int)
            for v in history:
                bin_idx = min(int((v - min_val) / bin_width), bins - 1)
                counts[bin_idx] += 1
            total = len(history)
            baseline_entropy = -sum(
                (c / total) * math.log2(c / total) for c in counts.values() if c > 0
            )
            anomaly = {
                "metric": metric,
                "value": round(value, 4),
                "type": "novelty",
                "baseline_entropy": round(baseline_entropy, 4),
                "severity": "critical",
                "timestamp": datetime.now(timezone.utc).isoformat(),
            }
            with self._lock:
                self.anomaly_log.append(anomaly)
                if len(self.anomaly_log) > self._MAX_ANOMALY_LOG:
                    self.anomaly_log = self.anomaly_log[-self._MAX_ANOMALY_LOG // 2:]
            return anomaly
        return None

    def detect_contextual(self, metric: str, value: float, context: Dict) -> Optional[Dict]:
        z_anomaly = self.check_zscore(metric, value)
        if z_anomaly:
            z_anomaly["context"] = copy.deepcopy(context) if context else {}
            return z_anomaly
        return None
    
    def get_anomalies(self, since: datetime = None, severity: str = None) -> List[Dict]:
        with self._lock:
            results = list(self.anomaly_log)
        if since:
            results = [a for a in results if a.get("timestamp", "") >= since.isoformat()]
        if severity:
            results = [a for a in results if a.get("severity") == severity]
        return copy.deepcopy(results)


class DecisionTree:
    """
    Simple Decision Tree classifier for Pattern Sight causation detection.
    
    Uses information gain (ID3 algorithm) to split on features.
    Limited depth to prevent overfitting.
    """

    def __init__(self, max_depth: int = 5, min_samples: int = 5):
        self.max_depth = max_depth
        self.min_samples = min_samples
        self.tree = None
        self.classes = []
        self.feature_names = []

    class _Node:
        def __init__(self, feature=None, threshold=None, left=None, right=None, value=None):
            self.feature = feature
            self.threshold = threshold
            self.left = left
            self.right = right
            self.value = value

    def _entropy(self, labels: List) -> float:
        if not labels:
            return 0.0
        counts = Counter(labels)
        total = len(labels)
        return -sum((c / total) * math.log2(c / total) for c in counts.values() if c > 0)

    def _information_gain(self, labels: List, left_labels: List, right_labels: List) -> float:
        parent_entropy = self._entropy(labels)
        n = len(labels)
        n_left = len(left_labels)
        n_right = len(right_labels)
        if n_left == 0 or n_right == 0:
            return 0.0
        return parent_entropy - (n_left / n) * self._entropy(left_labels) - (n_right / n) * self._entropy(right_labels)

    def _best_split(self, features: List[List], labels: List):
        best_gain = -1.0
        best_feature = 0
        best_threshold = 0.0

        for feature_idx in range(len(features[0])):
            values = sorted(set(f[feature_idx] for f in features))
            for i in range(len(values) - 1):
                threshold = (values[i] + values[i + 1]) / 2
                left_labels = [labels[j] for j in range(len(labels)) if features[j][feature_idx] <= threshold]
                right_labels = [labels[j] for j in range(len(labels)) if features[j][feature_idx] > threshold]
                gain = self._information_gain(labels, left_labels, right_labels)
                if gain > best_gain:
                    best_gain = gain
                    best_feature = feature_idx
                    best_threshold = threshold

        return best_feature, best_threshold, best_gain

    def _build_tree(self, features: List[List], labels: List, depth: int):
        if len(set(labels)) == 1:
            return self._Node(value=labels[0])
        if depth >= self.max_depth or len(labels) < self.min_samples:
            counts = Counter(labels)
            return self._Node(value=counts.most_common(1)[0][0])

        feature, threshold, gain = self._best_split(features, labels)
        if gain <= 0:
            counts = Counter(labels)
            return self._Node(value=counts.most_common(1)[0][0])

        left_idx = [i for i in range(len(features)) if features[i][feature] <= threshold]
        right_idx = [i for i in range(len(features)) if features[i][feature] > threshold]

        if not left_idx or not right_idx:
            counts = Counter(labels)
            return self._Node(value=counts.most_common(1)[0][0])

        left_features = [features[i] for i in left_idx]
        left_labels = [labels[i] for i in left_idx]
        right_features = [features[i] for i in right_idx]
        right_labels = [labels[i] for i in right_idx]

        return self._Node(
            feature=feature,
            threshold=threshold,
            left=self._build_tree(left_features, left_labels, depth + 1),
            right=self._build_tree(right_features, right_labels, depth + 1),
        )

    def fit(self, features: List[List], labels: List, feature_names: List[str] = None):
        self.feature_names = feature_names or [f"f{i}" for i in range(len(features[0]))]
        self.classes = list(set(labels))
        self.tree = self._build_tree(features, labels, 0)

    def predict(self, feature_vector: List) -> Any:
        node = self.tree
        while node.value is None:
            if feature_vector[node.feature] <= node.threshold:
                node = node.left
            else:
                node = node.right
        return node.value

    def predict_proba(self, feature_vector: List) -> Dict:
        prediction = self.predict(feature_vector)
        return {cls: (1.0 if cls == prediction else 0.0) for cls in self.classes}

    def feature_importance(self) -> Dict[str, float]:
        if not self.tree:
            return {}
        counts = defaultdict(int)
        self._count_splits(self.tree, counts)
        total = sum(counts.values()) or 1
        return {self.feature_names[f]: c / total for f, c in counts.items()}

    def _count_splits(self, node: '_Node', counts: Dict):
        if node.feature is not None:
            counts[node.feature] += 1
            self._count_splits(node.left, counts)
            self._count_splits(node.right, counts)


class ModelPersistence:
    """
    Save/load for all superhuman models to disk.
    Enables continuity across restarts.
    """

    BASE_DIR = DATA_DIR / "models"

    @classmethod
    def save(cls, name: str, data: Dict):
        name = _sanitize(name, 100)
        cls.BASE_DIR.mkdir(parents=True, exist_ok=True)
        path = cls.BASE_DIR / f"{name}.json"
        tmp = path.with_suffix(f".tmp.{os.getpid()}")
        try:
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2, default=str)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(path))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Save model %s error: %s", name, type(e).__name__)
            try:
                os.unlink(str(tmp))
            except OSError:
                pass

    @classmethod
    def load(cls, name: str) -> Optional[Dict]:
        name = _sanitize(name, 100)
        path = cls.BASE_DIR / f"{name}.json"
        if not path.exists():
            return None
        try:
            return json.loads(path.read_text())
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Load model %s error: %s", name, type(e).__name__)
            return None

    @classmethod
    def save_bayesian(cls, predictor: 'BayesianPredictor'):
        data = {
            "hypotheses": {
                name: {
                    "prior": h["prior"],
                    "posterior": h["posterior"],
                    "count": h["count"],
                    "evidence_likelihoods": h["evidence_likelihoods"],
                }
                for name, h in predictor.hypotheses.items()
            },
            "evidence_counts": dict(predictor.evidence_counts),
            "total_observations": predictor.total_observations,
        }
        cls.save("bayesian", data)

    @classmethod
    def load_bayesian(cls, predictor: 'BayesianPredictor'):
        data = cls.load("bayesian")
        if not data:
            return
        predictor.total_observations = data.get("total_observations", 0)
        predictor.evidence_counts = defaultdict(int, data.get("evidence_counts", {}))
        for name, h_data in data.get("hypotheses", {}).items():
            predictor.hypotheses[name] = {
                "prior": h_data["prior"],
                "posterior": h_data["posterior"],
                "count": h_data["count"],
                "evidence_likelihoods": h_data.get("evidence_likelihoods", {}),
            }

    @classmethod
    def save_markov(cls, model: 'MarkovBehaviorModel'):
        data = {
            "transitions": {
                s: {a: {ns: c for ns, c in targets.items()}
                    for a, targets in actions.items()}
                for s, actions in model.transitions.items()
            },
            "state_counts": dict(model.state_counts),
            "rewards": {f"{s}|{a}": r for (s, a), r in model.rewards.items()},
        }
        cls.save("markov", data)

    @classmethod
    def load_markov(cls, model: 'MarkovBehaviorModel'):
        data = cls.load("markov")
        if not data:
            return
        for s, actions in data.get("transitions", {}).items():
            for a, targets in actions.items():
                for ns, c in targets.items():
                    model.transitions[s][a][ns] = c
        model.state_counts = defaultdict(int, data.get("state_counts", {}))
        for key, r in data.get("rewards", {}).items():
            parts = key.split("|", 1)
            if len(parts) == 2:
                model.rewards[(parts[0], parts[1])] = r

    @classmethod
    def save_knowledge_graph(cls, kg: 'KnowledgeGraph'):
        data = {
            "entities": {eid: {k: v for k, v in e.items() if k != "mentioned"}
                         for eid, e in kg.entities.items()},
            "relations": kg.relations,
        }
        cls.save("knowledge_graph", data)

    @classmethod
    def load_knowledge_graph(cls, kg: 'KnowledgeGraph'):
        data = cls.load("knowledge_graph")
        if not data:
            return
        kg.entities = {}
        for eid, e_data in data.get("entities", {}).items():
            e_data["mentioned"] = []
            kg.entities[eid] = e_data
            kg.entity_index[e_data.get("type", "unknown")].add(eid)
        kg.relations = data.get("relations", [])
        for r in kg.relations:
            kg.relation_index[r.get("predicate", "")].add(r.get("id", 0))

    @classmethod
    def save_bandit(cls, bandit: 'MultiArmedBandit'):
        cls.save("bandit", bandit.get_stats())

    @classmethod
    def load_bandit(cls, bandit: 'MultiArmedBandit'):
        data = cls.load("bandit")
        if not data:
            return
        for arm_id, stats in data.get("arms", {}).items():
            bandit.add_arm(arm_id, stats.get("mean_reward", 0.5), stats.get("description", ""))
            for _ in range(stats.get("pulls", 0)):
                bandit.arms[arm_id]["pulls"] = stats["pulls"]


class FeedbackLoop:
    _MAX_HISTORY = 5000

    def __init__(self, max_history: int = 5000):
        self._lock = threading.Lock()
        self.predictions: List[Dict] = []
        self.outcomes: List[Dict] = []
        self._max_history = min(max_history, self._MAX_HISTORY) if max_history > 0 else self._MAX_HISTORY

    def record_prediction(self, ability: str, prediction: str, confidence: float,
                          context: Dict = None):
        ability = _sanitize(ability, 100)
        prediction = _sanitize_content(prediction, 500)
        confidence = max(0.0, min(1.0, confidence))
        entry = {
            "id": hashlib.sha256(f"{ability}:{prediction}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:12],
            "ability": ability,
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "context": {str(k)[:50]: str(v)[:500] for k, v in list((context or {}).items())[:50]},
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "resolved": False,
            "actual": None,
            "accuracy": None,
        }
        with self._lock:
            self.predictions.append(entry)
            if len(self.predictions) > self._max_history:
                self.predictions = self.predictions[-self._max_history // 2:]

    def record_outcome(self, ability: str, prediction_id: str, actual: str, correct: bool):
        ability = _sanitize(ability, 100)
        actual = _sanitize_content(actual, 500)
        with self._lock:
            for pred in reversed(self.predictions):
                if pred["id"] == prediction_id and pred["ability"] == ability:
                    pred["resolved"] = True
                    pred["actual"] = actual
                    pred["accuracy"] = correct
                    break
            self.outcomes.append({
                "ability": ability,
                "prediction_id": prediction_id,
                "actual": actual,
                "correct": correct,
                "timestamp": datetime.now(timezone.utc).isoformat(),
            })
            if len(self.outcomes) > self._max_history:
                self.outcomes = self.outcomes[-self._max_history // 2:]

    def get_accuracy(self, ability: str = None, last_n: int = 100) -> Dict:
        with self._lock:
            recent_outcomes = list(self.outcomes[-last_n:])
            recent_preds = list(self.predictions[-last_n:])
        if ability:
            recent_outcomes = [o for o in recent_outcomes if o["ability"] == ability]
        if not recent_outcomes:
            return {"accuracy": 0.0, "count": 0, "confidence_calibration": 0.0}
        correct = sum(1 for o in recent_outcomes if o.get("correct"))
        total = len(recent_outcomes)
        if ability:
            preds_with_conf = [p for p in recent_preds if p["resolved"] and p["ability"] == ability]
        else:
            preds_with_conf = [p for p in recent_preds if p["resolved"]]
        avg_confidence = sum(p["confidence"] for p in preds_with_conf) / len(preds_with_conf) if preds_with_conf else 0.0
        return {
            "accuracy": round(correct / total if total > 0 else 0.0, 4),
            "count": total,
            "confidence_calibration": round(avg_confidence, 4),
            "calibration_error": round(abs(avg_confidence - (correct / total if total > 0 else 0.0)), 4),
        }

    def calibrate_confidence(self, ability: str, raw_confidence: float) -> float:
        accuracy_data = self.get_accuracy(ability)
        if accuracy_data["count"] < 5:
            return round(raw_confidence, 4)
        observed_accuracy = accuracy_data["accuracy"]
        calibration_error = accuracy_data["calibration_error"]
        if calibration_error < 0.05:
            return round(raw_confidence, 4)
        if observed_accuracy < raw_confidence:
            return round(max(0.1, raw_confidence - calibration_error * 0.5), 4)
        return round(min(0.99, raw_confidence + calibration_error * 0.3), 4)


__all__ = [
    'BayesianPredictor',
    'SignalTriangulator',
    'KnowledgeGraph',
    'MarkovBehaviorModel',
    'ReinforcementLearner',
    'EnvironmentModel',
    'TemporalPatternEngine',
    'KalmanFilter1D',
    'MultiArmedBandit',
    'AnomalyDetector',
    'DecisionTree',
    'ModelPersistence',
    'FeedbackLoop',
]