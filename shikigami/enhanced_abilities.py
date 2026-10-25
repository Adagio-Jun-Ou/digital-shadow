#!/usr/bin/env python3
"""
Enhanced Shikigami - TRUE SUPERHUMAN ABILITIES

Each ability uses real mathematical/statistical techniques:
- Oracle: Monte Carlo simulation
- X-Ray: Device fingerprinting with signal analysis
- Life Pulse: Constellation matching with confidence scoring
- Pattern Sight: FFT, correlation, causation detection
- Intent Seer: Markov chain prediction
- etc.

All abilities are TESTED and grounded in data.
"""

import asyncio
import json
import os
import subprocess
import re
import threading
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from pathlib import Path
from enum import Enum
import logging

import httpx

from .abilities import (
    AbilityResult,
    MonteCarloSimulator,
    DeviceFingerprinter,
    PatternAnalyzer,
    BehaviorPredictor,
    ActionSequencePredictor,
    DecisionTracker,
    DATA_DIR,
    PEOPLE_DB,
    DECISIONS_DB,
    PATTERNS_DB,
    OBSERVATIONS_DB,
)

from .superhuman import (
    BayesianPredictor,
    SignalTriangulator,
    KnowledgeGraph,
    MarkovBehaviorModel,
    ReinforcementLearner,
    EnvironmentModel,
    TemporalPatternEngine,
    KalmanFilter1D,
    MultiArmedBandit,
    AnomalyDetector,
    DecisionTree,
    ModelPersistence,
    FeedbackLoop,
)

logger = logging.getLogger("shadowd.shikigami_enhanced")

_SAFE_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_\-\.]{1,128}\.(py|txt|md|json|yaml|yml|sh|toml|cfg|ini|conf)$')
_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_OBSERVATIONS = 5000
_MAX_PATTERNS = 500

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "https://ollama.com/v1")


class ShikigamiType(Enum):
    ORACLE = "oracle"
    X_RAY = "x_ray"
    LIFE_PULSE = "life_pulse"
    PERFECT_MEMORY = "perfect_memory"
    PATTERN_SIGHT = "pattern_sight"
    SOCIAL_RADAR = "social_radar"
    INTENT_SEER = "intent_seer"
    OMNI_SENSE = "omni_sense"
    SHADOW_COUNCIL = "shadow_council"
    REALITY_FORGE = "reality_forge"
    HABIT_TRACKER = "habit_tracker"
    ENERGY_FLOW = "energy_flow"
    FOCUS_SIGHT = "focus_sight"
    CONNECTION = "connection"
    COMPASS = "compass"
    HEARTH = "hearth"
    TIME_WITNESS = "time_witness"
    MIRROR = "mirror"
    LANTERN = "lantern"
    TIDE = "tide"
    ECHO = "echo"


class EnhancedShikigamiRegistry:
    """
    10 Superhuman abilities with real statistical/mathematical grounding.
    
    Each ability:
    1. Uses actual algorithms (Monte Carlo, FFT, Markov chains, etc.)
    2. Returns specific, actionable predictions
    3. Provides confidence scores based on data
    4. Is independently testable
    """
    
    def __init__(self):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "oracle_calls": 0, "xray_calls": 0, "life_pulse_calls": 0,
            "perfect_memory_calls": 0, "pattern_sight_calls": 0,
            "social_radar_calls": 0, "intent_seer_calls": 0,
            "omni_sense_calls": 0, "shadow_council_calls": 0,
            "reality_forge_calls": 0, "people_tracked": 0,
            "decisions_tracked": 0, "patterns_tracked": 0,
        }
        # Initialize core analyzers
        self.monte_carlo = MonteCarloSimulator()
        self.device_fingerprinter = DeviceFingerprinter()
        self.pattern_analyzer = PatternAnalyzer()
        self.behavior_predictor = BehaviorPredictor()
        self.action_predictor = ActionSequencePredictor()
        self.decision_tracker = DecisionTracker()
        
        # Initialize superhuman algorithms
        self.bayesian = BayesianPredictor()
        self.signal_triangulator = SignalTriangulator()
        self.knowledge_graph = KnowledgeGraph()
        self.markov = MarkovBehaviorModel()
        self.rl_learner = ReinforcementLearner()
        self.env_model = EnvironmentModel()
        self.temporal_engine = TemporalPatternEngine()
        self.kalman_filters: Dict[str, KalmanFilter1D] = {}
        self._MAX_KALMAN_FILTERS = 500
        self.social_bandit = MultiArmedBandit(algorithm="ucb1")
        self.anomaly_detector = AnomalyDetector()
        self.decision_tree = DecisionTree(max_depth=5)
        self.feedback_loop = FeedbackLoop()
        
        # Initialize Bayesian hypotheses for common predictions
        self._init_bayesian_hypotheses()
        
        # Load persisted models
        self._load_models()
        
        # Load data
        self.people = self.device_fingerprinter.load_people(PEOPLE_DB)
        self.observations: List[Dict] = self._load_observations()[-_MAX_OBSERVATIONS:]
        self.patterns: Dict = self._load_patterns()
        if len(self.patterns) > _MAX_PATTERNS:
            keys = list(self.patterns.keys())[:_MAX_PATTERNS]
            self.patterns = {k: self.patterns[k] for k in keys}
        
        # Train Markov model from observations
        self._train_markov_from_observations()
        
        # LLM client
        self._http_client = httpx.AsyncClient(timeout=60.0)
    
    async def close(self):
        """Clean up resources and save models."""
        self._save_models()
        if self._http_client:
            await self._http_client.aclose()
    
    def _load_models(self):
        """Load persisted models from disk."""
        ModelPersistence.load_bayesian(self.bayesian)
        ModelPersistence.load_markov(self.markov)
        ModelPersistence.load_knowledge_graph(self.knowledge_graph)
        ModelPersistence.load_bandit(self.social_bandit)
        rl_data = ModelPersistence.load("rl_learner")
        if rl_data:
            self.rl_learner.q_table = defaultdict(float, {
                tuple(k.split("|")): v for k, v in rl_data.get("q_table", {}).items()
            })
            self.rl_learner.experiences = rl_data.get("experiences", [])[-1000:]
        env_data = ModelPersistence.load("env_model")
        if env_data and "patterns" in env_data:
            self.env_model.patterns = env_data["patterns"]
    
    def _save_models(self):
        """Persist all models to disk."""
        ModelPersistence.save_bayesian(self.bayesian)
        ModelPersistence.save_markov(self.markov)
        ModelPersistence.save_knowledge_graph(self.knowledge_graph)
        ModelPersistence.save_bandit(self.social_bandit)
        ModelPersistence.save("rl_learner", {
            "q_table": {f"{s}|{a}": q for (s, a), q in self.rl_learner.q_table.items()},
            "experiences": self.rl_learner.experiences[-1000:],
            "alpha": self.rl_learner.alpha,
            "gamma": self.rl_learner.gamma,
        })
        ModelPersistence.save("env_model", {
            "patterns": {k: {kk: dict(vv) if isinstance(vv, defaultdict) else vv
                            for kk, vv in v.items()}
                        for k, v in self.env_model.patterns.items()},
        })
        ModelPersistence.save("feedback", {
            "predictions": self.feedback_loop.predictions[-500:],
            "outcomes": self.feedback_loop.outcomes[-500:],
        })
    
    def _load_json(self, path: Path) -> Dict:
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except (json.JSONDecodeError, OSError, ValueError):
            return {}
    
    def _load_observations(self) -> List[Dict]:
        """Load observation history."""
        data = self._load_json(OBSERVATIONS_DB)
        if isinstance(data, list):
            return data
        return []
    
    def _load_patterns(self) -> Dict:
        """Load learned patterns."""
        return self._load_json(PATTERNS_DB)
    
    def _init_bayesian_hypotheses(self):
        """Initialize Bayesian hypotheses for common prediction scenarios."""
        self.bayesian.add_hypothesis("user_active", 0.5, ["terminal_command", "file_change", "git_commit"])
        self.bayesian.add_hypothesis("user_idle", 0.5, ["no_activity", "long_gap"])
        self.bayesian.add_hypothesis("user_working", 0.4, ["terminal_command", "file_change", "git_commit", "window_focus"])
        self.bayesian.add_hypothesis("user_social", 0.3, ["device_nearby", "communication", "calendar_event"])
        self.bayesian.add_hypothesis("user_resting", 0.3, ["no_activity", "late_hour", "weekend"])
    
    def _train_markov_from_observations(self):
        """Train Markov model from existing observation sequences."""
        if len(self.observations) < 2:
            return
        for i in range(len(self.observations) - 1):
            state = self.observations[i].get("type", "unknown")
            next_state = self.observations[i + 1].get("type", "unknown")
            action = f"{state}_to_{next_state}"
            self.markov.record(state, action, next_state)
    
    def _save_observations(self):
        """Save observation history (atomic with fsync)."""
        with self._lock:
            truncated = self.observations[-_MAX_OBSERVATIONS:]
        try:
            tmp = OBSERVATIONS_DB.with_suffix(".tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(truncated, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                os.close(fd)
                raise
            os.replace(str(tmp), str(OBSERVATIONS_DB))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Save observations error: %s", type(e).__name__)
    
    async def _llm(self, prompt: str, max_tokens: int = 2000) -> str:
        """Call LLM for synthesis. Falls back to algorithm-only output when unavailable."""
        if not OLLAMA_API_KEY or not OLLAMA_API_KEY.strip():
            return self._algorithmic_synthesis(prompt)
        try:
            r = await self._http_client.post(
                f"{OLLAMA_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
                json={"model": "glm-5.1", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens},
                timeout=15,
            )
            data = r.json()
            return data["choices"][0]["message"]["content"]
        except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError,
                OSError, RuntimeError, httpx.HTTPError) as e:
            logger.debug("LLM unavailable, using algorithmic fallback: %s", type(e).__name__)
            return self._algorithmic_synthesis(prompt)

    def _algorithmic_synthesis(self, prompt: str) -> str:
        """Generate output from algorithmic/statistical data when LLM is unavailable.
        Extracts key statistics from the prompt and formats them as usable insights."""
        import re
        lines = prompt.split('\n')
        stats = {}
        for line in lines:
            m = re.search(r'(.+?):\s*(.+)', line)
            if m: stats[m.group(1).strip().lower()] = m.group(2).strip()

        parts = []
        request_line = [l for l in lines if l.startswith('Request:')]
        request = request_line[0].replace('Request:', '').strip() if request_line else ''

        if 'monte carlo mean' in stats:
            mc = stats.get('monte carlo mean', '')
            parts.append("Based on %s observations across %s days, the Monte Carlo simulation gives %s probability." % (
                stats.get('total observations', '0'), stats.get('days tracked', '1'), mc))
        if 'trend' in stats:
            parts.append("The trend is %s." % stats.get('trend', 'stable').split('(')[0].strip())
        if 'markov next activity' in stats:
            parts.append("Markov model predicts next activity: %s." % stats.get('markov next activity', 'unknown'))
        if 'bayesian top hypothesis' in stats:
            parts.append("Bayesian inference suggests: %s." % stats.get('bayesian top hypothesis', 'N/A'))
        if any(k in stats for k in ['average activities/day', 'peak hour']):
            parts.append("Activity peaks at %s with avg %s/day." % (
                stats.get('peak hour', '12:00'), stats.get('average activities/day', '0')))

        # Specific, actionable hints based on what's actually missing in the data.
        try:
            n_obs = int(stats.get('total observations', '0').split()[0])
        except (ValueError, IndexError):
            n_obs = 0
        try:
            n_people = int(stats.get('tracked people', '0').split()[0])
        except (ValueError, IndexError):
            n_people = 0
        try:
            n_decisions = int(stats.get('tracked decisions', '0').split()[0])
        except (ValueError, IndexError):
            n_decisions = 0

        next_steps: List[str] = []
        if n_obs < 50:
            next_steps.append(
                f"Bootstrap: only {n_obs} observation(s) tracked. Send a few via "
                f"`POST /observe` (or use the observers that should be already running: "
                f"clipboard, terminal, filesystem, window, git)."
            )
        if n_people == 0 and re.search(r"\b(who|person|people|sarah|john|mike|nearby|target)\b", request, re.I):
            next_steps.append(
                "No people tracked yet. Register one with "
                "`POST /track/person {name:'Sarah', devices:'AA:BB:CC:...', relation:'friend'}`."
            )
        if n_decisions < 3 and re.search(r"\b(should|decide|choose|recommend|advise|which|better)\b", request, re.I):
            next_steps.append(
                "Council needs more tracked decisions. Add with "
                "`POST /track/decision {situation, choice, outcome, satisfaction(1-10)}`."
            )

        if not parts and not next_steps:
            parts.append("Analysis based on available statistical data patterns.")
        if not parts and next_steps:
            parts.append("Insufficient data for a confident statistical inference yet.")

        # The original "continue monitoring" filler was deleted because it
        # never told the user what to *do*. The next_steps list above replaces it.
        if next_steps:
            parts.append("\n\n**Next steps**:\n  - " + "\n  - ".join(next_steps))

        return ' '.join(parts)
    
    ALLOWED_CMDS = {"bluetoothctl", "nmcli", "uptime", "echo"}

    async def _async_cmd(self, cmd: str) -> Tuple[int, str, str]:
        """Async-safe command execution using subprocess."""
        try:
            parts = cmd.split(None, 1)
            if not parts or parts[0] not in self.ALLOWED_CMDS:
                logger.warning("Blocked command: %s", cmd[:80].replace(chr(10), ' ').replace(chr(13), ' '))
                return -1, "", "command not allowed"
            proc = await asyncio.create_subprocess_exec(
                *parts, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return -1, "", "timeout"
        except (OSError, subprocess.SubprocessError, asyncio.TimeoutError, ValueError) as e:
            logger.error("Command error: %s", type(e).__name__)
            return -1, "", "execution failed"

    def _cmd(self, cmd: str) -> Tuple[int, str, str]:
        """Synchronous command execution (for backwards compat)."""
        try:
            parts = cmd.split(None, 1)
            if not parts or parts[0] not in self.ALLOWED_CMDS:
                logger.warning("Blocked command: %s", cmd[:80].replace(chr(10), ' ').replace(chr(13), ' '))
                return -1, "", "command not allowed"
            r = subprocess.run(cmd.split(), capture_output=True, text=True, timeout=10)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.error("Command error: %s", type(e).__name__)
            return -1, "", "execution failed"
    async def _scan_bt(self, duration: int = 6) -> List[Dict]:
        """Active BLE discovery. Delegates to phone.BluetoothScanner so X-Ray
        and Life Pulse see currently-present devices (not just paired ones)."""
        try:
            from phone.bluetooth_scanner import BluetoothScanner
        except ImportError:
            return []
        scanner = BluetoothScanner()
        devices = await scanner.scan(duration=duration)
        return [
            {"mac": d.mac, "name": d.name, "rssi": d.rssi, "type": d.device_type}
            for d in devices
        ]

    async def _scan_wifi(self) -> List[Dict]:
        """WiFi scan via phone.WiFiScanner (nmcli, then iwlist)."""
        try:
            from phone.wifi_scanner import WiFiScanner
        except ImportError:
            return []
        scanner = WiFiScanner()
        networks = await scanner.scan()
        return [
            {"ssid": n.ssid, "bssid": n.bssid, "signal": n.rssi,
             "channel": n.channel, "security": n.security}
            for n in networks
        ]
    
    # ═════════════════════════════════════════════════════════════════
    # 1. ORACLE - MONTE CARLO PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    async def _oracle(self, request: str, context: Dict) -> AbilityResult:
        """
        SPECIFIC predictions using Monte Carlo simulation + Bayesian inference
        + Temporal Pattern Engine + Feedback Loop calibration.
        
        Uses actual probability distributions, not vague guesses.
        Returns confidence intervals and sample sizes.
        """
        with self._lock:
            obs = list(context.get("observations", self.observations))
        now = datetime.now(timezone.utc)
        
        # Feed all observations to temporal engine
        for o in obs:
            self.temporal_engine.observe(o)
        
        # Analyze historical patterns
        activity_by_hour = {}
        for o in obs:
            if "timestamp" in o:
                try:
                    t = datetime.fromisoformat(o["timestamp"])
                    hour = t.hour
                    activity_by_hour[hour] = activity_by_hour.get(hour, 0) + 1
                except (ValueError, TypeError):
                    pass
        
        # Find peak hours
        peak_hour = max(activity_by_hour.items(), key=lambda x: x[1]) if activity_by_hour else (12, 0)
        
        # Calculate activity probability
        total_activities = len(obs)
        days_span = 1
        if obs and obs[0].get("timestamp"):
            try:
                days_span = max(1, (datetime.now(timezone.utc) - datetime.fromisoformat(obs[0]["timestamp"])).days)
            except (ValueError, TypeError):
                pass
        avg_daily = total_activities / days_span
        
        # Run Monte Carlo for prediction confidence
        activity_prob = self.monte_carlo.simulate_outcome(min(0.9, avg_daily / 20), num_simulations=10000)
        
        # Bayesian prediction from observation types
        recent_types = [o.get("type", "unknown") for o in obs[-20:]]
        evidence = {
            "terminal_command": "terminal_command" in recent_types,
            "file_change": "file_change" in recent_types,
            "git_commit": "git_commit" in recent_types,
            "window_focus": "window_focus" in recent_types,
            "no_activity": len(recent_types) < 3,
        }
        bayesian_predictions = self.bayesian.predict(evidence)
        
        # Markov prediction for next activity
        last_type = obs[-1].get("type", "unknown") if obs else "unknown"
        markov_preds = self.markov.predict_next_state(last_type, top_k=3)
        
        # Temporal forecast
        temporal_forecast = self.temporal_engine.predict_next_activity(now.hour)
        trend = self.temporal_engine.detect_trend()
        periodicity = self.temporal_engine.detect_periodicity()
        
        # Get device data
        devices = await self._scan_bt()
        known_count = sum(1 for d in devices if d["mac"].upper() in self.device_fingerprinter._known_devices)
        
        # Calibrate confidence with feedback loop
        raw_confidence = activity_prob['probability']
        calibrated_confidence = self.feedback_loop.calibrate_confidence("oracle", raw_confidence)
        
        # Record prediction
        self.feedback_loop.record_prediction("oracle", f"activity_at_{now.hour+1}", raw_confidence)

        with self._stats_lock:
            self._stats["oracle_calls"] += 1

        # Build LLM context with REAL statistics
        llm_context = f"""You are Oracle. Make ONE specific prediction based on ACTUAL statistical data.

CURRENT TIME: {now.strftime("%A %I:%M %p")}
DATA STATISTICS:
- Total observations: {total_activities}
- Days tracked: {days_span}
- Average activities/day: {avg_daily:.1f}
- Peak hour: {peak_hour[0]:02d}:00 ({peak_hour[1]} events)
- Activity probability (Monte Carlo 95% CI): {round(activity_prob['confidence_interval_95'][0], 4)}, {round(activity_prob['confidence_interval_95'][1], 4)}
- Monte Carlo mean: {round(activity_prob['probability'], 4)}
- Devices nearby: {len(devices)} ({known_count} known)
- Trend: {trend['trend']} (slope={round(trend['slope'], 4)}, confidence={round(trend['confidence'], 4)})
- Temporal forecast: activity at hour {temporal_forecast['predicted_hour']} ({round(temporal_forecast['predicted_probability'], 4)})
- Bayesian top hypothesis: {bayesian_predictions[0]['hypothesis'] if bayesian_predictions else 'N/A'} ({round(bayesian_predictions[0]['probability'] * 100, 1)}% probability)
- Markov next activity: {markov_preds[0]['next_state'] if markov_preds else 'unknown'} ({round(markov_preds[0]['probability'] * 100, 1) if markov_preds else 'N/A'}% probability)
- Feedback calibration: raw={round(raw_confidence, 4)}, calibrated={round(calibrated_confidence, 4)}
- Periodicity: {periodicity[:2] if periodicity else 'none detected'}

Request: {request}

Answer format:
1. ONE specific prediction (exact, not vague)
2. TIME estimate with range
3. CONFIDENCE % (calibrated: {min(95, total_activities/10)}% max)
4. DATA POINTS used
5. ONE ACTION to take

Max 100 words. Be precise. Use the statistics above."""

        prediction = await self._llm(llm_context)
        
        output = f"**Oracle - PRECOGNITION**\n\n"
        output += f"**Time**: {now.strftime('%A %I:%M %p')}\n"
        output += f"**Data Points**: {total_activities}\n"
        output += f"**Monte Carlo**: {activity_prob['simulations']} simulations\n"
        output += f"**95% CI**: [{round(activity_prob['confidence_interval_95'][0], 4)}, {round(activity_prob['confidence_interval_95'][1], 4)}]\n"
        output += f"**Trend**: {trend['trend']} (slope={round(trend['slope'], 4)})\n"
        if bayesian_predictions:
            output += f"**Bayesian**: {bayesian_predictions[0]['hypothesis']} ({round(bayesian_predictions[0]['probability'], 4)})\n"
        if markov_preds:
            output += f"**Markov Next**: {markov_preds[0]['next_state']} ({round(markov_preds[0]['probability'], 4)})\n"
        output += f"**Temporal**: Hour {temporal_forecast['predicted_hour']} ({round(temporal_forecast['predicted_probability'], 4)})\n"
        if periodicity:
            periodicity_str = ', '.join(f"{p['lag_hours']}h (r={round(p['autocorrelation'], 4)})" for p in periodicity[:2])
            output += f"**Periodicity**: {periodicity_str}\n"
        output += f"**Calibrated Confidence**: {round(calibrated_confidence, 4)}\n"
        output += f"\n{prediction}"
        
        data = {
            "devices": len(devices),
            "monte_carlo": {k: round(v, 4) if isinstance(v, float) else v for k, v in activity_prob.items()},
            "bayesian": [{"hypothesis": p["hypothesis"], "probability": round(p["probability"], 4), "confidence": round(p["confidence"], 4)} for p in bayesian_predictions][:5] if bayesian_predictions else [],
            "markov": [{"action": p["action"], "next_state": p["next_state"], "probability": round(p["probability"], 4)} for p in markov_preds][:5] if markov_preds else [],
            "temporal_forecast": {k: round(v, 4) if isinstance(v, float) else v for k, v in temporal_forecast.items()},
            "trend": {k: round(v, 4) if isinstance(v, float) else v for k, v in trend.items()},
            "calibrated_confidence": round(calibrated_confidence, 4),
        }
        
        return AbilityResult(
            True, output,
            data,
            confidence=calibrated_confidence
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 2. X-RAY - DEVICE FINGERPRINTING
    # ═════════════════════════════════════════════════════════════════
    
    async def _xray(self, request: str, context: Dict) -> AbilityResult:
        """
        SEE THROUGH WALLS using device fingerprinting + signal triangulation.
        
        Identifies WHO is present using:
        - Bluetooth device scanning with RSSI
        - MAC address database of known people
        - Device constellation matching
        - Signal triangulation for distance estimation
        - Confidence scoring
        """
        devices = await self._scan_bt()
        networks = await self._scan_wifi()
        
        # Feed Bluetooth data to signal triangulator with Kalman filtering
        for d in devices:
            rssi = d.get("rssi", -70)
            mac_upper = d["mac"].upper()
            # Apply Kalman filter for smooth distance estimation
            if mac_upper not in self.kalman_filters:
                if len(self.kalman_filters) >= self._MAX_KALMAN_FILTERS:
                    oldest_key = next(iter(self.kalman_filters))
                    del self.kalman_filters[oldest_key]
                self.kalman_filters[mac_upper] = KalmanFilter1D(process_noise=0.01, measurement_noise=1.0)
                self.kalman_filters[mac_upper].update(rssi)
            else:
                self.kalman_filters[mac_upper].update(rssi)
            # Use Kalman-smoothed RSSI for triangulation
            smoothed_rssi = self.kalman_filters[mac_upper].estimated_value
            self.signal_triangulator.record_signal(mac_upper, smoothed_rssi)
        
        # Identify all detected devices
        identified = []
        unknown = []
        
        for d in devices:
            identification = self.device_fingerprinter.identify_device(d["mac"], d["name"])
            if identification["identified"]:
                distance = self.signal_triangulator.get_distance(d["mac"].upper())
                movement = self.signal_triangulator.get_movement_vector(d["mac"].upper())
                identified.append({
                    "person": identification["person"],
                    "device": d["name"],
                    "mac": d["mac"],
                    "confidence": round(identification["confidence"], 4),
                    "distance": round(distance, 2) if distance is not None else None,
                    "movement": movement,
                })
            else:
                unknown.append({
                    "device": d["name"],
                    "mac": d["mac"],
                    "type": identification["device_type"]
                })
        
        # Group identified by person with confidence
        people_present = {}
        for p in identified:
            name = p["person"]
            if name not in people_present:
                people_present[name] = {
                    "devices": [],
                    "confidence": p["confidence"],
                    "distances": [],
                    "movement": None,
                }
            people_present[name]["devices"].append(p["device"])
            people_present[name]["confidence"] = round(min(0.95, people_present[name]["confidence"] + 0.1), 4)
            if p["distance"] is not None:
                people_present[name]["distances"].append(p["distance"])
            if p["movement"]:
                people_present[name]["movement"] = p["movement"]
        
        # Calculate presence confidence for each person
        for name in people_present:
            detected_macs = [d["mac"].upper() for d in devices]
            people_present[name]["confidence"] = self.device_fingerprinter.calculate_presence_confidence(
                detected_macs, name
            )
        
        # Location inference from WiFi
        location = "unknown"
        location_confidence = 0.3
        for n in networks[:3]:
            ssid = n.get("ssid", "").lower()
            if any(w in ssid for w in ["home", "my"]):
                location = "home"
                location_confidence = 0.9
                break
            elif any(w in ssid for w in ["office", "corp", "work"]):
                location = "office"
                location_confidence = 0.85
                break
            elif any(w in ssid for w in ["starbucks", "cafe", "coffee"]):
                location = "café"
                location_confidence = 0.8
                break
        
        # Estimate people count (1 person approx 2-3 devices)
        estimated_people = max(1, len(devices) // 3)
        
        output = f"**X-Ray Vision - SEE THROUGH WALLS**\n\n"
        output += f"**Location**: {location.title()} ({round(location_confidence, 4)} confidence)\n"
        output += f"**Primary Network**: {networks[0]['ssid'] if networks else 'None'}\n"
        output += f"**Devices Detected**: {len(devices)}\n"
        output += f"**People Estimate**: {estimated_people}\n\n"
        
        if people_present:
            output += f"**IDENTIFIED** ({len(people_present)}):\n"
            for name, data in sorted(people_present.items(), key=lambda x: -x[1]["confidence"]):
                output += f"  **{name}** ({round(data['confidence'], 4)} confidence)\n"
                output += f"    Devices: {', '.join(data['devices'])}\n"
                if data["distances"]:
                    avg_dist = sum(data["distances"]) / len(data["distances"])
                    output += f"    Distance: {avg_dist:.1f}m\n"
                if data["movement"]:
                    mv = data["movement"]
                    if mv.get("approaching"):
                        output += f"    Movement: Approaching\n"
                    elif mv.get("departing"):
                        output += f"    Movement: Departing\n"
        
        if unknown:
            output += f"\n**UNKNOWN DEVICES** ({len(unknown)}):\n"
            for d in unknown[:5]:
                output += f"  - {d['device']} ({d['type']})\n"
        
        output += "\n**Track someone**: 'Track [name] has device [MAC]'"
        
        return AbilityResult(
            True, output,
            {
                "identified": len(identified),
                "unknown": len(unknown),
                "location": location,
                "people_estimate": estimated_people
            },
            confidence=round(max((p["confidence"] for p in people_present.values()), default=0), 4)
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 3. LIFE PULSE - CONSTELLATION MATCHING
    # ═════════════════════════════════════════════════════════════════
    
    async def _life_pulse(self, request: str, context: Dict) -> AbilityResult:
        """
        Identify specific people via device constellation matching.
        
        Each person has a unique constellation of devices (phone, watch, earbuds).
        When we detect multiple devices from someone's constellation,
        we can identify them with high confidence.
        """
        devices = await self._scan_bt()
        request_lower = request.lower()
        
        # Find target person if mentioned (word boundary)
        target = None
        for name in self.people.keys():
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', request_lower):
                target = name
                break
        
        # Identify all devices
        detected_macs = [d["mac"].upper() for d in devices]
        
        # Check each known person's constellation
        present_people = {}
        for name in self.people.keys():
            confidence = self.device_fingerprinter.calculate_presence_confidence(detected_macs, name)
            if confidence > 0.3:  # At least one device detected
                present_people[name] = {
                    "confidence": confidence,
                    "devices_found": [],
                    "total_devices": len(self.device_fingerprinter.get_person_constellation(name))
                }
                for d in devices:
                    if d["mac"].upper() in self.device_fingerprinter.get_person_constellation(name):
                        present_people[name]["devices_found"].append(d["name"])
        
        # Format output
        output = f"💓 **Life Pulse - PRESENCE DETECTION**\n\n"
        output += f"**Scanned**: {len(devices)} devices\n"
        output += f"**Known People**: {len(self.people)}\n\n"
        
        if target:
            if target in present_people:
                data = present_people[target]
                output += f"**{target} IS PRESENT** ✓\n\n"
                output += f"**Confidence**: {round(data['confidence'], 4)}\n"
                output += f"**Devices Detected**: {data['devices_found']}\n"
                output += f"**Constellation Size**: {data['total_devices']} devices\n\n"
                
                # Add last seen pattern if available
                if target in self.people:
                    patterns = self.people[target].get("patterns", {})
                    if patterns:
                        output += f"**Patterns**: {json.dumps(patterns)}\n"
            else:
                output += f"**{target} NOT PRESENT** ✗\n\n"
                output += f"**Last Seen**: Not tracked\n"
                output += f"**Recommendation**: Track this person's devices\n"
        else:
            if present_people:
                output += f"**PRESENT** ({len(present_people)}):\n"
                for name, data in sorted(present_people.items(), key=lambda x: -x[1]["confidence"]):
                    output += f"✓ **{name}** ({round(data['confidence'], 4)})\n"
                    output += f"  Devices: {', '.join(data['devices_found'])}\n\n"
            else:
                output += "**No tracked people present**\n\n"

            # Surface unknown devices when there are any — useful even when no
            # one we know is around. RSSI-sorted, closest first.
            known_macs = set()
            for name in self.people.keys():
                known_macs.update(
                    m.upper() for m in self.device_fingerprinter.get_person_constellation(name)
                )
            unknowns = [d for d in devices if d["mac"].upper() not in known_macs]
            unknowns.sort(key=lambda d: -d.get("rssi", -100))
            if unknowns:
                output += f"**UNKNOWN NEARBY** ({len(unknowns)}):\n"
                for d in unknowns[:6]:
                    rssi = d.get("rssi", 0)
                    name_str = d.get("name") or "<unnamed>"
                    output += f"  • {d['mac']}  {name_str:<24s}  RSSI={rssi}dBm\n"
                output += (
                    "\n_Track a device-owner: "
                    "`POST /track/person {name, devices, relation}`_\n"
                )
            elif not present_people:
                output += "Track someone: `POST /track/person {name, devices, relation}`\n"

        return AbilityResult(
            True, output,
            {
                "present": list(present_people.keys()),
                "target": target,
                "confidences": {k: round(v["confidence"], 4) for k, v in present_people.items()},
                "unknown_count": sum(
                    1 for d in devices
                    if d["mac"].upper() not in {
                        m.upper() for n in self.people.keys()
                        for m in self.device_fingerprinter.get_person_constellation(n)
                    }
                ),
            },
            confidence=round(max((v["confidence"] for v in present_people.values()), default=0), 4),
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 4. PERFECT MEMORY - SEMANTIC SEARCH
    # ═════════════════════════════════════════════════════════════════
    
    async def _perfect_memory(self, request: str, context: Dict) -> AbilityResult:
        """
        Total recall with temporal and semantic search + Knowledge Graph.
        
        - Temporal: "What did I do Tuesday?"
        - Keyword: "coding"
        - Entity: "Sarah"
        - Knowledge Graph: semantic relationship queries
        """
        with self._lock:
            obs = list(context.get("observations", self.observations))
        request_lower = request.lower()
        
        # Auto-populate knowledge graph from observations
        for o in obs[-50:]:
            o_type = o.get("type", "unknown")
            content = o.get("content", {})
            if isinstance(content, str):
                try:
                    content = json.loads(content)
                except (json.JSONDecodeError, TypeError):
                    content = {"text": content}
            
            entity_name = None
            if o_type == "git_commit":
                entity_name = content.get("message", content.get("commit", ""))[:50]
                self.knowledge_graph.add_entity("commit", entity_name, content)
                self.knowledge_graph.add_relation(
                    self.knowledge_graph.add_entity("activity", f"git_{entity_name[:20]}", {"type": o_type}),
                    "produced",
                    self.knowledge_graph.add_entity("commit", entity_name, content),
                    content
                )
            elif o_type == "terminal_command":
                cmd = content.get("command", content.get("text", ""))
                self.knowledge_graph.add_entity("command", cmd[:50], content)
            elif "person" in request_lower and o_type in ("device_nearby", "clipboard"):
                pass  # People entities are tracked separately
        
        # Knowledge Graph query if entity mentioned (word boundary)
        kg_results = []
        for name in self.people.keys():
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', request_lower):
                kg_results = self.knowledge_graph.query({
                    "entity_type": "person",
                    "name_contains": name,
                    "relations": [{"predicate": "met"}, {"predicate": "said"}],
                })
                break
        
        matches = []
        match_reasons = []
        
        # Temporal search — first try the natural-language time window parser
        # (handles "yesterday", "this week", "last 3 days", "past hour", etc.).
        try:
            from .time_parser import parse_time_window, in_window
            window = parse_time_window(request_lower)
            if window:
                matches = [o for o in obs if in_window(o.get("timestamp", ""), window)]
                match_reasons.append(
                    f"window:{window[0].isoformat()[:16]}→{window[1].isoformat()[:16]}"
                )
        except Exception as e:
            logger.debug("time window parser failed: %s", type(e).__name__)

        # Fall back to weekday-name matching ("Tuesday", "Friday", …).
        day_names = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
                    "friday": 4, "saturday": 5, "sunday": 6}

        if not matches:
            for day_name, day_num in day_names.items():
                if day_name in request_lower:
                    days_ago = (datetime.now(timezone.utc).weekday() - day_num) % 7
                    target_date = (datetime.now(timezone.utc) - timedelta(days=days_ago)).date()
                    matches = [o for o in obs if str(target_date) in o.get("timestamp", "")]
                    match_reasons.append(f"date:{target_date}")
                    break
        
        # Keyword search (word boundary)
        if not matches:
            keywords = [w for w in request_lower.split() if len(w) > 3 and w not in ["what", "when", "where", "which", "did"]]
            kw_pattern = r'\b(' + '|'.join(re.escape(k) for k in keywords) + r')\b'
            for o in obs:
                o_json = json.dumps(o).lower()
                if re.search(kw_pattern, o_json):
                    matches.append(o)
            if keywords:
                match_reasons.append(f"keywords:{keywords}")
        
        # Entity search (from people DB, word boundary)
        entity_matches = []
        for name in self.people.keys():
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', request_lower):
                entity_matches = [o for o in obs if re.search(r'\b' + re.escape(name.lower()) + r'\b', json.dumps(o).lower())]
                if entity_matches:
                    matches = entity_matches
                    match_reasons.append(f"entity:{name}")
                    break
        
        # Commitment detection
        commitment_keywords = [r'\bcall\b', r'\bpromise\b', r'\btomorrow\b', r'\blater\b', r'\bwill\b', r'\bshould\b', r'\bneed\s+to\b', r'\bhave\s+to\b']
        commitments = []
        for o in obs:
            o_json = json.dumps(o).lower()
            if any(re.search(k, o_json) for k in commitment_keywords):
                commitments.append(o)
        
        # Semantic ranking: try real embeddings first, fall back to TF-IDF.
        # Embeddings catch cross-vocabulary matches (e.g. "lunch with Bob"
        # finds "had a meal with Bob"); TF-IDF only catches token overlap.
        semantic_results: List[Dict] = []
        semantic_backend = "none"
        try:
            from .semantic_memory import semantic_search, _observation_to_text
            from . import embedding_index
            search_pool = matches if len(matches) > 20 else obs

            # Try embeddings path. Stays cheap because doc vectors are cached
            # on disk by content hash.
            doc_texts = [_observation_to_text(o) for o in search_pool]
            emb_hits = embedding_index.cosine_search(
                doc_texts, request, top_k=5, min_score=0.55,
            )
            if emb_hits:
                semantic_backend = "embeddings"
                for i, score in emb_hits:
                    o = dict(search_pool[i])
                    o["_score"] = round(score, 3)
                    o["_matched_terms"] = ["(embedding)"]
                    semantic_results.append(o)
            else:
                semantic_backend = "tfidf"
                semantic_results = semantic_search(
                    search_pool, request, top_k=5, min_score=0.08,
                )
        except Exception as e:
            logger.debug("Semantic search skipped: %s", type(e).__name__)

        output = f"🧠 **Perfect Memory - TOTAL RECALL**\n\n"
        output += f"**Searched**: {len(obs)} observations\n"
        output += f"**Found**: {len(matches)} matches\n"
        if match_reasons:
            output += f"**Matched by**: {', '.join(match_reasons)}\n\n"

        if matches:
            output += "**RESULTS**:\n"
            for m in matches[-10:]:  # Last 10 matches
                ts = m.get("timestamp", "?")[:19]
                typ = m.get("type", "?")
                content = json.dumps(m.get("content", m), indent=None)[:150]
                output += f"[{ts}] {typ}\n  {content}\n\n"
        else:
            output += "No exact matches found.\n\n"
            if not semantic_results:
                output += "**Try**: 'What did I do Tuesday?' or 'coding' or 'Sarah'\n"

        if semantic_results:
            output += f"\n**SEMANTIC MATCHES** ({len(semantic_results)}, via {semantic_backend}):\n"
            for r in semantic_results:
                ts = r.get("timestamp", "?")[:19]
                typ = r.get("type", "?")
                score = r.get("_score", 0)
                terms = ", ".join(r.get("_matched_terms", [])) or "—"
                content = json.dumps(r.get("content", r), indent=None)[:120]
                output += f"[{ts}] {typ}  (sim={score}, via: {terms})\n  {content}\n"
        
        if commitments:
            output += f"\n**⚠️ {len(commitments)} POTENTIAL COMMITMENTS**:\n"
            for c in commitments[-3:]:
                ts = c.get("timestamp", "?")[:10]
                output += f"[{ts}] {json.dumps(c.get('content', c), indent=None)[:80]}\n"
        
        if kg_results:
            output += f"\n**KNOWLEDGE GRAPH** ({len(kg_results)} entities):\n"
            for r in kg_results[:5]:
                entity = r.get("entity", {})
                output += f"  {entity.get('type', '?')}: {entity.get('name', '?')} (conf: {round(r.get('confidence', 0), 4)})\n"
                for predicate, rels in r.get("relations", {}).items():
                    for rel in rels[:2]:
                        rel_entity = rel.get("entity", {})
                        output += f"    {predicate} → {rel_entity.get('name', '?')}\n"

        # Unified context snapshot — compact summary of everything known right now
        total_observations = len(obs)
        domains_seen = {}
        for o in obs[-100:]:
            t = o.get("type", "unknown")
            domains_seen[t] = domains_seen.get(t, 0) + 1
        context_snapshot = f"\n\n**CONTEXT SNAPSHOT**\n"
        context_snapshot += f"Total observations tracked: {total_observations}\n"
        context_snapshot += f"Recent activity breakdown: {', '.join(f'{t}({c})' for t, c in list(domains_seen.items())[:8])}\n"
        context_snapshot += f"People tracked: {len(self.people)} ({', '.join(list(self.people.keys())[:5])})\n" if self.people else ""
        context_snapshot += f"Decisions recorded: {len(self.decisions)}\n" if hasattr(self, 'decisions') and self.decisions else ""
        context_snapshot += f"Compacted summaries: {sum(1 for o in obs if isinstance(o, dict) and o.get('type') == 'compacted_summary')}\n"
        
        return AbilityResult(
            True, output + context_snapshot,
            {"matches": len(matches), "commitments": len(commitments), "kg_entities": len(kg_results),
             "total_observations": total_observations, "semantic_backend": semantic_backend},
            confidence=round(min(1.0, len(matches) / 100), 4)
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 5. PATTERN SIGHT - FFT & CAUSATION
    # ═════════════════════════════════════════════════════════════════
    
    async def _pattern_sight(self, request: str, context: Dict) -> AbilityResult:
        """
        Detect patterns using FFT, correlation, and causation analysis.
        
        - Frequency patterns (daily/weekly cycles)
        - Correlations (what correlates with what)
        - Correlation detection
        - Causation (what causes what)
        """
        with self._lock:
            obs = list(context.get("observations", self.observations))
        
        # Extract time series
        by_hour = {}
        by_type = defaultdict(list)
        
        for o in obs:
            if "timestamp" in o:
                try:
                    t = datetime.fromisoformat(o["timestamp"])
                    by_hour[t.hour] = by_hour.get(t.hour, 0) + 1
                    by_type[o.get("type", "unknown")].append(t)
                except (ValueError, TypeError):
                    pass
        
        # Frequency analysis
        if len(by_hour) >= 4:
            hour_values = [by_hour.get(h, 0) for h in range(24)]
            freq_analysis = self.pattern_analyzer.detect_frequency(hour_values)
        else:
            freq_analysis = {"dominant_frequency": None, "period": None}
        
        # Correlation detection
        correlations = self.pattern_analyzer.find_correlations(obs)
        
        # Causation detection (look for stored patterns)
        causations = []
        for pattern_name, pattern_data in self.patterns.items():
            if isinstance(pattern_data, dict):
                for cause_type, effect_data in pattern_data.items():
                    if isinstance(effect_data, dict) and effect_data.get("causation_rate"):
                        causations.append({
                            "cause": cause_type,
                            "effect": pattern_name,
                            "rate": effect_data["causation_rate"],
                            "lift": effect_data.get("lift", 1.0)
                        })
        
        # Decision Tree classification (if enough data)
        dt_predictions = {}
        dt_importance = {}
        if len(obs) >= 10 and len(by_type) >= 2:
            try:
                feature_names = ["hour", "weekday", "prev_type_encoded"]
                features = []
                labels = []
                type_to_idx = {t: i for i, t in enumerate(sorted(by_type.keys()))}
                for i in range(1, len(obs)):
                    o = obs[i]
                    prev_o = obs[i - 1]
                    if "timestamp" in o:
                        try:
                            t = datetime.fromisoformat(o["timestamp"])
                            features.append([
                                t.hour,
                                t.weekday(),
                                type_to_idx.get(prev_o.get("type", "unknown"), 0),
                            ])
                            labels.append(o.get("type", "unknown"))
                        except (ValueError, TypeError):
                            pass
                if len(features) >= 5 and len(set(labels)) >= 2:
                    self.decision_tree.fit(features, labels, feature_names)
                    current_hour = datetime.now(timezone.utc).hour
                    current_weekday = datetime.now(timezone.utc).weekday()
                    prev_type = obs[-1].get("type", "unknown") if obs else "unknown"
                    prediction = self.decision_tree.predict([
                        current_hour, current_weekday, type_to_idx.get(prev_type, 0)
                    ])
                    dt_predictions = {
                        "predicted_next": prediction,
                        "feature_importance": self.decision_tree.feature_importance(),
                    }
                    dt_importance = self.decision_tree.feature_importance()
            except (ValueError, TypeError, AttributeError, RuntimeError) as e:
                logger.debug("Decision tree error: %s", type(e).__name__)
        
        # Build insight
        peak_hour = max(by_hour.items(), key=lambda x: x[1]) if by_hour else (0, 0)
        
        output = f"👁️‍🗨️ **Pattern Sight - CAUSATION DETECTION**\n\n"
        output += f"**Data Points**: {len(obs)}\n"
        output += f"**Peak Activity**: {peak_hour[0]:02d}:00 ({peak_hour[1]} events)\n"
        
        if freq_analysis.get("period"):
            output += f"**Cycle Period**: Every {round(freq_analysis['period'], 4)} hours\n"
        
        output += f"**Correlations**: {len(correlations)}\n"
        output += f"**Causations**: {len(causations)}\n\n"
        
        if correlations:
            output += "**TOP CORRELATIONS**:\n"
            for c in correlations[:3]:
                output += f"  {c['type1']} ↔ {c['type2']} ({round(c['correlation'], 4)}) {c['strength']}\n"
        
        if causations:
            output += "\n**CAUSATIONS DETECTED**:\n"
            for c in causations[:3]:
                output += f"  {c['cause']} → {c['effect']} ({round(c['rate'], 4)} rate, {round(c['lift'], 4)}x lift)\n"
        
        if dt_predictions:
            output += f"\n**DECISION TREE**:\n"
            output += f"  Predicted next activity: {dt_predictions.get('predicted_next', 'unknown')}\n"
            if dt_importance:
                sorted_imp = sorted(dt_importance.items(), key=lambda x: -x[1])
                for name, imp in sorted_imp:
                    output += f"  Feature importance: {name} = {round(imp, 4)}\n"

        # Cognitive bias detection — surfaced inline so the user sees signals
        # alongside the standard pattern report.
        bias_signals = []
        try:
            from .bias_detector import detect_biases
            bias_signals = detect_biases(
                obs,
                decisions=list(self.decision_tracker.decisions),
                tracked_people=self.people,
            )
        except Exception as e:
            logger.debug("bias detection skipped: %s", type(e).__name__)
        if bias_signals:
            output += "\n**⚠️ COGNITIVE BIAS SIGNALS**:\n"
            for s in bias_signals[:5]:
                output += f"  • {s.name:<14s} strength={s.strength:.2f}\n"
                for ev in s.evidence[:2]:
                    output += f"      → {ev}\n"
                output += f"      💡 {s.suggestion}\n"
        
        # LLM synthesis
        llm_prompt = f"""Analyze patterns for causation insights.

ACTIVITY BY HOUR: {dict(by_hour)}
PEAK HOUR: {peak_hour[0]:02d}:00
CORRELATIONS: {correlations[:5]}
CAUSATIONS: {causations[:5]}
STORED PATTERNS: {len(self.patterns)}
DECISION TREE PREDICTION: {dt_predictions}

Answer:
1. CAUSATION: What causes what (X → Y)
2. BEST TIME: Optimal hour for activity
3. WORST TIME: Hour to avoid
4. INSIGHT: Key pattern discovered
5. ACTION: One recommendation

Max 100 words. Be specific."""

        synthesis = await self._llm(llm_prompt)
        output += "\n" + synthesis
        
        return AbilityResult(
            True, output,
            {
                "peak_hour": peak_hour[0],
                "correlations": len(correlations),
                "causations": len(causations),
                "frequency": freq_analysis,
                "decision_tree": dt_predictions,
                "bias_signals": [
                    {"name": s.name, "strength": s.strength,
                     "evidence": s.evidence, "suggestion": s.suggestion}
                    for s in bias_signals
                ],
            },
            confidence=round(min(0.95, len(correlations) / 10), 4),
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 6. SOCIAL RADAR - BEHAVIOR PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    async def _social_radar(self, request: str, context: Dict) -> AbilityResult:
        """
        Predict behavior of specific people using tracked patterns + RL
        recommendations + sentiment / manipulation analysis of recent messages.
        """
        request_lower = request.lower()
        with self._lock:
            obs = list(context.get("observations", self.observations))

        # Find target person (word boundary)
        target = None
        for name in self.people.keys():
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', request_lower):
                target = name
                break

        # Get behavior prediction
        if target:
            prediction = self.behavior_predictor.predict_behavior(target)
            warning_signs = self.behavior_predictor.get_warning_signs(target)
            person_data = self.people.get(target, {})
        else:
            prediction = {"behaviors": {}, "confidence": 0}
            warning_signs = []
            person_data = {}

        # Sentiment / manipulation analysis on recent communications. We pull
        # text from clipboard / message / email observations — broadly anything
        # that looks like content authored by or about the target.
        comms_text: List[str] = []
        target_l = target.lower() if target else None
        for o in obs[-200:]:
            otype = o.get("type", "")
            if otype not in {"clipboard", "message", "email", "communication", "chat", "note"}:
                continue
            content = o.get("content", {})
            text = ""
            if isinstance(content, dict):
                for k in ("text", "body", "message", "preview", "content"):
                    if isinstance(content.get(k), str):
                        text = content[k]
                        break
            elif isinstance(content, str):
                text = content
            if not text:
                continue
            if target_l is None or target_l in text.lower():
                comms_text.append(text)

        social_signals = {}
        if comms_text:
            try:
                from .sentiment import analyze_messages
                social_signals = analyze_messages(comms_text)
            except Exception as e:
                logger.debug("sentiment analysis skipped: %s", type(e).__name__)
                social_signals = {}
        
        # RL-based decision recommendation if target found
        rl_recommendation = None
        if target and self.decision_tracker.decisions:
            state = self.rl_learner.encode_state({"person": target, "context": request_lower[:50]})
            possible_actions = ["engage", "observe", "wait", "communicate", "avoid"]
            rl_recommendation = self.rl_learner.get_recommendation(state, possible_actions)
        
        # Multi-Armed Bandit engagement strategy
        bandit_recommendation = None
        if target:
            engagement_actions = [
                f"engage_{target}", f"observe_{target}", f"communicate_{target}",
                "engage_general", "observe_general", "communicate_general"
            ]
            best_arm, bandit_score = self.social_bandit.select(engagement_actions)
            # Unexplored arms come back as +inf; clamp to a JSON-safe sentinel.
            if bandit_score == float("inf"):
                bandit_score = 999.0
                bandit_unexplored = True
            else:
                bandit_unexplored = False
            bandit_recommendation = {
                "action": best_arm,
                "ucb_score": bandit_score,
                "unexplored": bandit_unexplored,
                "arm_stats": self.social_bandit.get_stats(),
            }
        
        # Get past interactions
        past_interactions = []
        decisions_list = self.decision_tracker.decisions
        for d in decisions_list:
            if target and target.lower() in json.dumps(d).lower():
                past_interactions.append(d)
        
        # LLM synthesis
        llm_prompt = f"""Predict behavior for a person.

TARGET: {target or "general"}
TRACKED DATA: {json.dumps(person_data, indent=2)}
BEHAVIOR MODEL: {json.dumps(prediction, indent=2)}
WARNING SIGNS: {warning_signs[:3]}
PAST INTERACTIONS: {len(past_interactions)}
RL RECOMMENDATION: {rl_recommendation}
RECENT TONE: {social_signals.get('sentiment_label', 'unknown')} (trend: {social_signals.get('sentiment_trend', 'n/a')})
MANIPULATION SCORE: {social_signals.get('manipulation_score', 0)}

Request: {request}

Answer:
1. BASELINE: What's normal for this person
2. PREDICTION: ONE specific behavior (name it)
3. CONFIDENCE: [X]% based on data
4. SIGNALS: 2-3 signs to watch for
5. RESPONSE: What to do

Max 150 words. Be specific."""

        synthesis = await self._llm(llm_prompt)

        output = f"**Social Radar - PEOPLE PREDICTION**\n\n"
        if target:
            output += f"**Target**: {target}\n"
            output += f"**Tracked Behaviors**: {len(prediction.get('behaviors', {}))}\n"
            output += f"**Prediction Confidence**: {round(prediction.get('confidence', 0), 4)}\n"
            if rl_recommendation:
                output += f"**RL Recommended Action**: {rl_recommendation.get('recommended_action', 'N/A')} (Q={round(rl_recommendation.get('q_value', 0), 4)})\n"
            if bandit_recommendation:
                output += f"**Bandit Strategy**: {bandit_recommendation['action'].replace('_', ' ')} (UCB={round(bandit_recommendation['ucb_score'], 4)})\n"
        else:
            output += f"**Tracked People**: {len(self.people)}\n\n"

        if social_signals.get("count", 0) > 0:
            output += (
                f"**Recent Messages**: {social_signals['count']} analyzed | "
                f"tone={social_signals['sentiment_label']} ({social_signals['avg_sentiment']:+.2f}) | "
                f"trend={social_signals['sentiment_trend']}\n"
            )
            manip = social_signals.get("manipulation_score", 0)
            if manip > 0.3:
                tactics = ", ".join(social_signals.get("manipulation_tactics", {}).keys()) or "—"
                output += f"**⚠️ Manipulation Risk**: {manip:.2f} ({tactics})\n"

        output += "\n" + synthesis

        if warning_signs:
            output += f"\n\n**WARNING SIGNS**:\n"
            for s in warning_signs[:2]:
                output += f"  - {s['behavior']}: {s['warning_sign']} ({round(s['probability'], 4)})\n"

        if social_signals.get("examples"):
            ex = social_signals["examples"][0]
            output += f"\n**MANIPULATION SAMPLE** (score={ex['score']}):\n  {ex['message'][:150]}\n  tactics: {', '.join(ex['tactics'])}\n"

        return AbilityResult(
            True, output,
            {
                "target": target,
                "prediction": prediction,
                "rl_recommendation": rl_recommendation,
                "bandit_strategy": bandit_recommendation,
                "social_signals": social_signals,
            },
            confidence=round(prediction.get("confidence", 0), 4),
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 7. INTENT SEER - MARKOV CHAIN PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    async def _intent_seer(self, request: str, context: Dict) -> AbilityResult:
        """
        Predict next actions using Markov chain modeling + action sequence prediction.
        """
        with self._lock:
            obs = list(context.get("observations", self.observations))
        
        # Train Markov model from recent observations
        for i in range(min(len(obs) - 1, 100)):
            if obs[i].get("type") and obs[i+1].get("type"):
                state = obs[i]["type"]
                next_state = obs[i+1]["type"]
                action = f"{state}_to_{next_state}"
                self.markov.record(state, action, next_state)
        
        # Get recent action types
        # Coerce to str — obs may have `type: None` explicitly, which would
        # otherwise blow up the `' → '.join()` calls below.
        last_types = [str(o.get("type") or "?") for o in obs[-5:]]
        
        # Markov predictions (primary)
        last_state = last_types[-1] if last_types else "unknown"
        markov_preds = self.markov.predict_next_state(last_state, top_k=3)
        
        # Fallback: action sequence predictor
        seq_preds = self.action_predictor.predict_next(last_types, top_k=3)
        
        # Combine predictions
        combined = []
        seen = set()
        for p in markov_preds:
            key = p["next_state"]
            if key not in seen:
                combined.append({
                    "action": p["action"],
                    "next_state": key,
                    "probability": p["probability"],
                    "source": "markov",
                    "value": p.get("value", 0),
                })
                seen.add(key)
        for p in seq_preds:
            key = p.get("action", "?")
            if key not in seen:
                combined.append({
                    "action": key,
                    "next_state": key,
                    "probability": p.get("probability", 0),
                    "source": "sequence",
                    "value": 0,
                })
                seen.add(key)
        
        # Predicted sequence via Markov simulation. Coerce to strings so None
        # states from a sparse model don't blow up the join below.
        predicted_sequence = [str(s) if s is not None else "?"
                              for s in self.markov.simulate(last_state, steps=5)]
        
        # LLM synthesis
        llm_prompt = f"""Predict the NEXT action from sequence.

RECENT ACTIONS: {' → '.join(last_types)}
MARKOV PREDICTIONS: {markov_preds[:3]}
COMBINED PREDICTIONS: {combined[:5]}
PREDICTED SEQUENCE: {' → '.join(predicted_sequence)}

Request: {request}

Predict:
1. NEXT: One specific action
2. WHEN: Timeframe
3. CONFIDENCE: [X]% based on transition probability
4. SEQUENCE: Likely next 3 actions
5. PREPARE: What to do now

Max 100 words. Be specific."""

        synthesis = await self._llm(llm_prompt)
        
        output = f"**Intent Seer - ACTION PREDICTION**\n\n"
        output += f"**Recent**: {' → '.join(last_types[-5:])}\n"
        output += f"**Markov States**: {self.markov.state_counts.get(last_state, 0)} observations of '{last_state}'\n\n"
        
        if combined:
            output += "**PREDICTIONS**:\n"
            for p in combined[:5]:
                src = p.get("source", "unknown")
                output += f"  {p['action']}: {round(p['probability'], 4)} ({src})\n"
        
        output += "\n" + synthesis
        
        return AbilityResult(
            True, output,
            {"markov_states": len(self.markov.state_counts), "predictions": combined},
            round(max((p["probability"] for p in combined), default=0), 4)
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 8. OMNI SENSE - ENVIRONMENTAL FUSION
    # ═════════════════════════════════════════════════════════════════
    
    async def _omni_sense(self, request: str, context: Dict) -> AbilityResult:
        """
        Complete environmental awareness by fusing all sensors + EnvironmentModel.
        Pulls BT, WiFi, audio activity, battery, system load, and uptime in parallel.
        """
        # Native sensor probes are independent of BT/WiFi — fire them concurrently.
        try:
            from sensors.native_sensors import native_snapshot
            native_task = asyncio.create_task(native_snapshot())
        except ImportError:
            native_task = None

        devices = await self._scan_bt()
        networks = await self._scan_wifi()
        rcode, uptime, _ = await self._async_cmd("uptime")
        native = await native_task if native_task is not None else {}
        with self._lock:
            obs = list(context.get("observations", self.observations))
        
        # Feed observations to environment model
        for o in obs[-20:]:
            self.env_model.observe(o)
            # Feed numeric metrics to anomaly detector
            if "type" in o:
                hour = datetime.now(timezone.utc).hour
                self.anomaly_detector.update_baseline(f"activity_{o['type']}_h{hour}", 1.0)
        
        # Check for anomalies in device count and activity
        anomalies_from_detector = []
        if len(devices) > 0:
            anomaly = self.anomaly_detector.check_zscore("device_count", float(len(devices)))
            if anomaly:
                anomalies_from_detector.append(anomaly)
        if obs:
            activity_rate = len(obs[-10:]) / max(1, 10)
            anomaly = self.anomaly_detector.check_zscore("activity_rate", activity_rate)
            if anomaly:
                anomalies_from_detector.append(anomaly)
        
        # Identify people
        detected_macs = [d["mac"].upper() for d in devices]
        present = [name for name in self.people.keys()
                   if self.device_fingerprinter.calculate_presence_confidence(detected_macs, name) > 0.3]
        
        # Build device list for environment model
        device_data = [{"mac": d["mac"], "name": d["name"], "known": d["mac"].upper() in self.device_fingerprinter._known_devices, "person": self.device_fingerprinter._known_devices.get(d["mac"].upper(), {}).get("name")} for d in devices]
        
        # Get environment snapshot from model
        snapshot = self.env_model.get_environment_snapshot(device_data, networks)
        
        output = f"**Omni Sense - ENVIRONMENTAL AWARENESS**\n\n"
        output += f"**Location**: {snapshot['location']['name'].title()}\n"
        output += f"**Location Confidence**: {round(snapshot['location']['confidence'], 4)}\n"
        output += f"**Primary Network**: {snapshot['location']['network'] or 'None'}\n"
        output += f"**Devices**: {snapshot['devices']['count']} nearby ({snapshot['devices']['known']} known, {snapshot['devices']['unknown']} unknown)\n"
        output += f"**Tracked People**: {len(present)} ({', '.join(present[:3]) if present else 'none'})\n"
        output += f"**Activity Level**: {snapshot['devices']['activity_level']}\n"
        output += f"**Time**: {snapshot['time']['date']} {snapshot['time']['context']}\n"
        output += f"**System Uptime**: {uptime.strip()[:50] if rcode == 0 else 'unknown'}\n"

        # Native sensor fusion — audio / battery / load.
        if native:
            audio = native.get("audio", {})
            battery = native.get("battery", {})
            load = native.get("load", {})
            output += "\n**NATIVE SENSORS**:\n"
            if audio.get("backend") == "pactl":
                state = audio.get("state", "?")
                output += (f"  🔊 Audio: {state}  "
                           f"(play={audio.get('playing_streams', 0)}, "
                           f"rec={audio.get('recording_streams', 0)}, "
                           f"vol={audio.get('default_sink_volume_pct', '?')}%)\n")
            if battery:
                cap = battery.get("capacity_pct", "?")
                status = battery.get("status", "?")
                output += f"  🔋 Battery: {cap}% {status}\n"
            if load:
                load_1m = load.get("load_1m", 0)
                ncpu = load.get("ncpu", 1)
                output += f"  ⚙️  Load: {load_1m:.2f} / {ncpu} CPUs ({100*load_1m/ncpu:.0f}% utilization)\n"
            inferred = native.get("inferred_activity")
            if inferred:
                output += f"  🎯 Inferred activity: {inferred}\n"
        
        if snapshot.get("predictions"):
            output += f"\n**PREDICTED ACTIVITIES**:\n"
            for p in snapshot["predictions"][:3]:
                output += f"  - {p['activity']}: {round(p['probability'], 4)} probability\n"
        
        if snapshot.get("anomalies"):
            output += f"\n**ANOMALIES**:\n"
            for a in snapshot["anomalies"][:3]:
                output += f"  - {a['type']}: {a['description']}\n"
        
        if anomalies_from_detector:
            output += f"\n**STATISTICAL ANOMALIES**:\n"
            for a in anomalies_from_detector[:3]:
                output += f"  - {a['metric']}: value={a.get('value', '?')}, z={a.get('z_score', 0):.1f} ({a.get('severity', 'unknown')})\n"
        
        output += f"\n**ENVIRONMENTAL ASSESSMENT**:\n"
        output += f"You are at {snapshot['location']['name']}. "
        output += f"{len(present)} tracked person(s) nearby. "
        output += f"Activity level is {snapshot['devices']['activity_level']}."
        
        return AbilityResult(
            True, output,
            {
                "location": snapshot["location"]["name"],
                "people": len(present),
                "devices": len(devices),
                "activity": snapshot["devices"]["activity_level"],
                "predictions": snapshot.get("predictions", []),
                "anomalies": snapshot.get("anomalies", []),
                "native": native,
            },
            confidence=round(snapshot["location"]["confidence"], 4),
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 9. SHADOW COUNCIL - DECISION LEARNING
    # ═════════════════════════════════════════════════════════════════
    
    async def _shadow_council(self, request: str, context: Dict) -> AbilityResult:
        """
        Cross-decision synthesis: action-category clustering with recency
        weighting, trend detection, and counterfactual surfacing.
        """
        similar = self.decision_tracker.find_similar(request, limit=10)
        legacy_recommendation = self.decision_tracker.get_recommendation(request)

        try:
            from .council_synthesis import synthesize
            verdict = synthesize(self.decision_tracker.decisions, request, similar=similar)
        except Exception as e:
            logger.warning("Council synthesis failed: %s", type(e).__name__)
            verdict = {"action_stats": {}, "recommendation":
                       {"action": None, "confidence": 0.0, "lift": 0.0, "reasoning": str(e)},
                       "trend": {}, "counterfactual": None}

        output = "👥 **Shadow Council - PAST WISDOM**\n\n"
        output += f"**Tracked Decisions**: {len(self.decision_tracker.decisions)}\n"
        output += f"**Similar Situations**: {len(similar)}\n\n"

        rec = verdict["recommendation"]
        if rec.get("action"):
            output += f"**COUNCIL VERDICT**: {rec['action']}\n"
            output += f"  → Confidence: {rec['confidence']}\n"
            output += f"  → Lift over baseline: +{rec['lift']}\n"
            output += f"  → {rec['reasoning']}\n\n"

        if verdict["action_stats"]:
            output += "**ACTION CATEGORY BREAKDOWN** (recency-weighted):\n"
            # Sort by avg satisfaction descending.
            ranked = sorted(verdict["action_stats"].items(),
                            key=lambda kv: -kv[1]["avg_satisfaction"])
            for category, stats in ranked[:6]:
                output += (f"  {category:<14}  n={stats['n']:<3} "
                           f"avg={stats['avg_satisfaction']:>4.1f}/10  "
                           f"success={stats['success_rate']*100:>3.0f}%\n")
            output += "\n"

        if verdict["trend"]:
            improving = [(c, t) for c, t in verdict["trend"].items() if t["direction"] == "improving"]
            declining = [(c, t) for c, t in verdict["trend"].items() if t["direction"] == "declining"]
            if improving or declining:
                output += "**TREND**:\n"
                for c, t in improving:
                    output += f"  ↑ {c} improving ({t['old_avg']} → {t['recent_avg']})\n"
                for c, t in declining:
                    output += f"  ↓ {c} declining ({t['old_avg']} → {t['recent_avg']})\n"
                output += "\n"

        if verdict["counterfactual"]:
            cf = verdict["counterfactual"]
            marker = {"affirming": "✓", "warning": "⚠", "neutral": "•"}.get(cf["polarity"], "•")
            output += (f"**MOST INFORMATIVE PAST** {marker} ({cf['polarity']}, "
                       f"sat={cf['satisfaction']}/10)\n")
            output += f"  [{cf['timestamp']}] {cf['situation']}\n"
            output += f"  → chose: {cf['choice']}\n"
            if cf["outcome"]:
                output += f"  → outcome: {cf['outcome']}\n"
            output += "\n"

        if similar:
            output += "**SIMILAR DECISIONS** (last 5):\n"
            for s in similar[-5:]:
                ts = s.get("timestamp", "?")[:10]
                choice = s.get("choice", "?")
                outcome = s.get("outcome", "?")[:60]
                sat = s.get("satisfaction", "?")
                output += f"  [{ts}] {choice} → {outcome} (sat={sat}/10)\n"
        elif not verdict["action_stats"]:
            output += "No tracked decisions yet. "
            advice_prompt = f"""Give brief advice on: {request}

Format:
1. CONSIDER: Main factors
2. AVOID: Common mistake
3. RECOMMEND: One action

Max 80 words."""
            advice = await self._llm(advice_prompt)
            output += f"**General Advice**:\n{advice}\n"

        output += "\n**Track decisions**: POST /track/decision with situation/choice/outcome/satisfaction\n"

        return AbilityResult(
            True, output,
            {
                "similar": len(similar),
                "verdict": verdict,
                "legacy_recommendation": legacy_recommendation,
            },
            confidence=rec.get("confidence", 0),
        )
    
    # ═════════════════════════════════════════════════════════════════
    # 10. REALITY FORGE - MANIFESTATION
    # ═════════════════════════════════════════════════════════════════
    
    async def _reality_forge(self, request: str, context: Dict) -> AbilityResult:
        """
        Create files and execute commands to manifest intentions.
        """
        workspace = Path.home() / "digital-shadow" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        prompt = f"""Create: {request}

Output format:
---FILE: filename.ext---
<content here>
---END---

---CMD: command here---
---END---

Create WORKING code only. Be practical."""

        generated = await self._llm(prompt, max_tokens=4000)
        
        files = re.findall(r'---FILE: (.+?)---\n(.*?)\n---END---', generated, re.DOTALL)
        created = []
        for fp, content in files:
            safe_name = fp.strip().lstrip("/.")
            if "/" in safe_name or "\\" in safe_name:
                logger.warning("Blocked path traversal: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            if not _SAFE_FILENAME_RE.match(safe_name):
                logger.warning("Blocked unsafe filename: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            if len(content) > 65536:
                logger.warning("Blocked oversized file: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            path = workspace / safe_name
            try:
                path.resolve().relative_to(workspace.resolve())
            except ValueError:
                logger.warning("Blocked path traversal: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp = path.with_suffix(".tmp")
                tmp.write_text(content.strip())
                tmp.replace(path)
            except OSError:
                logger.warning("Failed to write file: %s", safe_name)
                continue
            created.append(str(path))
        
        # Parse commands. Run them through the safe runner: allowlist-checked,
        # workspace-cwd'd, with output capped. Reports policy decisions so the
        # user sees why a command did or didn't run.
        commands = re.findall(r'---CMD: (.+?)---', generated)
        executed: List[Dict] = []
        if commands:
            try:
                from .safe_runner import run_commands, RunnerPolicy
                policy = RunnerPolicy(workspace=workspace)
                results = await run_commands(commands[:5], policy=policy)
                executed = [
                    {
                        "cmd": r.cmd,
                        "success": r.success,
                        "policy_decision": r.policy_decision,
                        "rc": r.returncode,
                        "stdout": r.stdout[:400] if r.stdout else "",
                        "stderr": r.stderr[:400] if r.stderr else "",
                    }
                    for r in results
                ]
            except Exception as e:
                logger.warning("safe_runner failed: %s", type(e).__name__)
                executed = [{"cmd": c, "success": False, "policy_decision": f"runner_error:{type(e).__name__}"}
                            for c in commands[:5]]

        output = f"🔨 **Reality Forge - MANIFESTATION**\n\n"

        if created:
            output += f"**Created**:\n"
            for f in created:
                output += f"  📄 {f}\n"

        if executed:
            output += f"\n**Commands**:\n"
            for e in executed:
                decision = e.get("policy_decision", "?")
                if decision == "executed":
                    mark = "✓" if e["success"] else "✗"
                    output += f"  {mark} `{e['cmd']}`  rc={e.get('rc')}\n"
                    if e.get("stdout"):
                        output += f"      → {e['stdout'][:150]}\n"
                elif decision == "would_run_dry":
                    output += f"  ▷ (dry-run) `{e['cmd']}`\n"
                else:
                    output += f"  ⛔ `{e['cmd']}`  ({decision})\n"

        if not created and not executed:
            output += "**LLM Plan** (no files/commands extracted yet):\n"
            output += generated[:500] + "\n\n"
            output += (
                "_To get files/commands, ask for a concrete artifact, e.g._\n"
                "  `summon --shikigami reality_forge \"create a python script that prints hello\"`\n"
                "_The LLM must produce blocks like `---FILE: name.ext---\\n…\\n---END---` "
                "and `---CMD: command---`._\n"
            )

        if "localhost" in output or "127.0.0.1" in generated:
            output += "\n**✓ Service may be running. Check URLs above.**"

        return AbilityResult(
            True, output,
            {"files": len(created), "cmds": len(executed),
             "executed_count": sum(1 for e in executed if e.get("policy_decision") == "executed")},
            commands_run=[e["cmd"] for e in executed if e.get("policy_decision") == "executed"],
            files_created=created,
        )

    # ═════════════════════════════════════════════════════════════════
    # 11. HABIT TRACKER - RECURRING BEHAVIOR DETECTION
    # ═════════════════════════════════════════════════════════════════

    async def _habit_tracker(self, request: str, context: Dict) -> AbilityResult:
        """Detect recurring behaviors in observations and report streaks,
        consistency, and broken-habit alerts."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .habit_tracker import detect_habits, find_broken_streaks, format_habit
        except ImportError as e:
            return AbilityResult(False, f"Habit tracker unavailable: {e}")

        habits = detect_habits(obs, min_frequency=3, min_consistency=0.35)
        broken = find_broken_streaks(habits)

        output = "🌀 **Habit Tracker - RECURRING BEHAVIOR DETECTION**\n\n"
        output += f"**Observations analyzed**: {len(obs)}\n"
        output += f"**Habits detected**: {len(habits)}\n\n"

        if habits:
            output += "**TOP HABITS** (by frequency):\n"
            for h in habits[:10]:
                output += f"  • {format_habit(h)}\n"
        else:
            output += "No habits detected yet. Need more observations across multiple weeks.\n"

        if broken:
            output += f"\n**⚠️ BROKEN STREAKS** ({len(broken)}):\n"
            for h in broken:
                output += (f"  • {h.activity_type} @ {h.bucket}: "
                           f"last seen {h.days_since}d ago "
                           f"(was a {h.longest_streak}-week streak)\n")

        # Sample alignment with the request — if it names a specific habit,
        # surface that one prominently.
        request_lower = request.lower()
        focus: Optional["Habit"] = None  # type: ignore[name-defined]
        for h in habits:
            if h.activity_type.lower() in request_lower:
                focus = h
                break
        if focus:
            output += f"\n**FOCUS**: '{focus.activity_type}' bucket {focus.bucket} — "
            output += f"{focus.frequency} firings, "
            output += f"avg {focus.avg_per_week}/wk, "
            output += f"streak {focus.streak}w (longest {focus.longest_streak}w).\n"

        return AbilityResult(
            True, output,
            {
                "habits_detected": len(habits),
                "broken_streaks": len(broken),
                "habits": [
                    {
                        "type": h.activity_type, "bucket": h.bucket,
                        "frequency": h.frequency, "consistency": h.consistency,
                        "streak": h.streak, "longest_streak": h.longest_streak,
                        "days_since": h.days_since,
                    } for h in habits[:20]
                ],
            },
            confidence=round(min(1.0, len(habits) / 5.0), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # 12. ENERGY FLOW - CIRCADIAN ENERGY CURVE
    # ═════════════════════════════════════════════════════════════════

    async def _energy_flow(self, request: str, context: Dict) -> AbilityResult:
        """Compute the user's hourly energy curve from observation density and
        forecast the next 6 hours of energy."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .energy_curve import compute_energy_curve, render_curve_ascii
        except ImportError as e:
            return AbilityResult(False, f"Energy curve unavailable: {e}")

        report = compute_energy_curve(obs)
        ascii_chart = render_curve_ascii(report.hourly_curve, report.current_hour)

        output = "⚡ **Energy Flow - CIRCADIAN CURVE**\n\n"
        output += f"**Now**: hour {report.current_hour:02d}:00 — "
        output += f"**{report.current_level.upper()}** energy ({report.current_score:.0%})\n"
        output += f"**Peak hour**: {report.peak_hour:02d}:00\n"
        output += f"**Trough hour**: {report.trough_hour:02d}:00\n"
        output += f"**Sample**: {report.sample_size} observations over {report.days_observed} day(s)\n\n"
        output += "**24h curve** (* = now):\n```\n" + ascii_chart + "\n```\n\n"
        if report.forecast_6h:
            output += "**Next 6 hours**:\n"
            for f in report.forecast_6h:
                output += f"  {f['hour']:02d}:00 — {f['level']:<6s} ({f['score']:.0%})\n"

        if report.sample_size < 24:
            output += ("\n_Curve will sharpen with more observations spread "
                       "across the day._\n")

        return AbilityResult(
            True, output,
            {
                "current_level": report.current_level,
                "current_score": report.current_score,
                "peak_hour": report.peak_hour,
                "trough_hour": report.trough_hour,
                "forecast_6h": report.forecast_6h,
                "sample_size": report.sample_size,
            },
            confidence=round(min(1.0, report.sample_size / 50.0), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # 13. FOCUS SIGHT - DEEP-WORK BLOCK DETECTION
    # ═════════════════════════════════════════════════════════════════

    async def _focus_sight(self, request: str, context: Dict) -> AbilityResult:
        """Detect deep-work blocks in observation history and classify
        current focus state (in_flow / scattered / idle)."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .focus_blocks import find_focus_blocks, current_focus_state, summarize_focus
        except ImportError as e:
            return AbilityResult(False, f"Focus blocks unavailable: {e}")

        blocks = find_focus_blocks(obs)
        summary = summarize_focus(blocks)
        state = current_focus_state(obs)

        state_marker = {
            "in_flow": "🌊", "scattered": "🪟", "idle": "😴",
        }.get(state["state"], "•")

        output = "🎯 **Focus Sight - DEEP-WORK DETECTION**\n\n"
        output += f"**Current state**: {state_marker} **{state['state'].upper()}**\n"
        if state["dominant"]:
            output += f"  → focused on: {state['dominant']} (purity {state.get('purity', 0):.0%})\n"
        if state.get("since_last_event_min") is not None:
            output += f"  → last event: {state['since_last_event_min']:.1f} min ago\n"
        output += "\n"
        output += f"**Total focus blocks**: {summary['total_blocks']}\n"
        if summary["total_blocks"]:
            output += f"**Total focused time**: {summary['total_focused_minutes']:.1f} min "
            output += f"({summary['total_focused_minutes']/60:.1f} hours)\n"
            output += f"**Longest block**: {summary['longest_block_minutes']:.1f} min\n"
            output += f"**Favorite focus type**: {summary['favorite_focus_type']}\n\n"
            output += "**RECENT BLOCKS** (last 5):\n"
            for b in blocks[-5:]:
                output += (f"  [{b.start[:16]}] {b.duration_min:.0f}min  "
                           f"{b.dominant_type} (purity {b.purity:.0%})\n")
        else:
            output += "\nNo focus blocks detected yet — need contiguous activity in observation history.\n"

        return AbilityResult(
            True, output,
            {
                "current_state": state["state"],
                "dominant_type": state.get("dominant"),
                "blocks_total": summary["total_blocks"],
                "focused_minutes": summary["total_focused_minutes"],
                "longest_minutes": summary["longest_block_minutes"],
                "recent_blocks": [
                    {"start": b.start, "end": b.end, "min": b.duration_min,
                     "type": b.dominant_type, "purity": b.purity}
                    for b in blocks[-10:]
                ],
            },
            confidence=round(min(1.0, summary["total_blocks"] / 5.0), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # 14. CONNECTION - RELATIONSHIP STRENGTH SCORING
    # ═════════════════════════════════════════════════════════════════

    async def _connection(self, request: str, context: Dict) -> AbilityResult:
        """Score every tracked person on relationship strength, surfacing
        rising stars and at-risk relationships."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .connection_strength import score_all_connections, trend_marker
        except ImportError as e:
            return AbilityResult(False, f"Connection scoring unavailable: {e}")

        connections = score_all_connections(self.people, obs)

        request_lower = request.lower()
        target: Optional[str] = None
        for name in self.people.keys():
            if re.search(r"\b" + re.escape(name.lower()) + r"\b", request_lower):
                target = name
                break

        output = "🔗 **Connection - RELATIONSHIP STRENGTH**\n\n"
        if not connections:
            output += "No tracked people. Add some via `POST /track/person`.\n"
            return AbilityResult(
                True, output, {"connections": []}, confidence=0.0,
            )

        if target:
            # Focused view for a specific person.
            focused = next((c for c in connections if c.name.lower() == target.lower()), None)
            if focused:
                output += f"**Focus: {focused.name}** ({focused.relation or 'unknown'})\n"
                output += f"  Score: **{focused.score:.0%}** {trend_marker(focused.trend)} ({focused.trend})\n"
                output += f"  Last contact: {focused.days_since_contact}d ago\n" \
                          if focused.days_since_contact is not None else \
                          "  Last contact: no recorded interaction\n"
                output += f"  30d / 90d interactions: {focused.interactions_30d} / {focused.interactions_90d}\n"
                output += "  Components:\n"
                for k, v in focused.components.items():
                    output += f"    - {k}: {v:.2f}\n"
                output += "\n"

        output += "**ALL CONNECTIONS** (strongest → weakest):\n"
        for c in connections[:15]:
            ds = f"{c.days_since_contact}d" if c.days_since_contact is not None else "—"
            output += (f"  {trend_marker(c.trend)} {c.name:<20s} {c.score:.0%}  "
                       f"({c.trend}, last={ds}, 30d={c.interactions_30d})\n")

        rising = [c for c in connections if c.trend == "rising"]
        dormant = [c for c in connections if c.trend == "dormant"]
        if rising:
            output += f"\n**RISING ({len(rising)})**: " + ", ".join(c.name for c in rising[:5]) + "\n"
        if dormant:
            output += f"**DORMANT ({len(dormant)}, ≥60d quiet)**: " + ", ".join(c.name for c in dormant[:5]) + "\n"

        return AbilityResult(
            True, output,
            {
                "connections": [
                    {
                        "name": c.name, "relation": c.relation,
                        "score": c.score, "trend": c.trend,
                        "components": c.components,
                        "interactions_30d": c.interactions_30d,
                        "interactions_90d": c.interactions_90d,
                        "days_since_contact": c.days_since_contact,
                    } for c in connections
                ],
                "rising_count": len(rising),
                "dormant_count": len(dormant),
            },
            confidence=round(min(1.0, len(connections) / 5.0), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # 15. COMPASS - GOAL ALIGNMENT
    # ═════════════════════════════════════════════════════════════════

    async def _compass(self, request: str, context: Dict) -> AbilityResult:
        """Score each tracked goal on alignment with recent observations."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .compass import CompassStore, score_goal, trend_marker
        except ImportError as e:
            return AbilityResult(False, f"Compass unavailable: {e}")

        # Persistent store of goals, separate from decisions.
        store = getattr(self, "_compass_store", None)
        if store is None:
            store = CompassStore()
            self._compass_store = store
        goals = store.list()

        if not goals:
            return AbilityResult(
                True,
                "🧭 **Compass - GOAL ALIGNMENT**\n\n"
                "No goals tracked yet. Add one with:\n"
                "  `POST /track/goal {title, keywords:[…], target_date?}`\n",
                {"goals_total": 0, "progress": []},
                confidence=0.0,
            )

        progresses = [score_goal(g, obs) for g in goals]
        progresses.sort(key=lambda p: -p.alignment)

        active = [p for p in progresses if p.goal.status == "active"]
        rising = [p for p in active if p.trend == "rising"]
        dormant = [p for p in active if p.trend == "dormant"]

        output = "🧭 **Compass - GOAL ALIGNMENT**\n\n"
        output += f"**Goals tracked**: {len(goals)}  (active: {len(active)})\n\n"
        output += "**GOALS** (best alignment first):\n"
        for p in progresses[:15]:
            tgt = f"in {p.days_to_target}d" if p.days_to_target and p.days_to_target > 0 else \
                  (f"{abs(p.days_to_target)}d overdue" if p.days_to_target is not None and p.days_to_target < 0
                   else "no deadline")
            status_mark = {"active": "●", "paused": "‖", "done": "✓", "abandoned": "✗"}.get(p.goal.status, "•")
            output += (
                f"  {status_mark} {trend_marker(p.trend)} {p.goal.title[:50]:<50s}  "
                f"align={p.alignment:.0%}  ({p.mentions_last_30d}/30d, {p.mentions_last_7d}/7d)  {tgt}\n"
            )

        if rising:
            output += f"\n**RISING**: " + ", ".join(p.goal.title[:30] for p in rising[:5]) + "\n"
        if dormant:
            output += f"**DORMANT** (no recent progress): " + \
                      ", ".join(p.goal.title[:30] for p in dormant[:5]) + "\n"

        return AbilityResult(
            True, output,
            {
                "goals_total": len(goals),
                "active": len(active),
                "rising_count": len(rising),
                "dormant_count": len(dormant),
                "progress": [
                    {
                        "goal_id": p.goal.goal_id, "title": p.goal.title,
                        "status": p.goal.status,
                        "alignment": p.alignment, "trend": p.trend,
                        "mentions_30d": p.mentions_last_30d,
                        "mentions_7d": p.mentions_last_7d,
                        "days_to_target": p.days_to_target,
                    } for p in progresses
                ],
            },
            confidence=round(min(1.0, len(active) / 3.0), 4) if active else 0.0,
        )

    # ═════════════════════════════════════════════════════════════════
    # 16. HEARTH - LOCATION PATTERNS
    # ═════════════════════════════════════════════════════════════════

    async def _hearth(self, request: str, context: Dict) -> AbilityResult:
        """Surface location patterns: time per place, hourly mode,
        transitions, anomalies."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .hearth import compute_hearth
        except ImportError as e:
            return AbilityResult(False, f"Hearth unavailable: {e}")

        report = compute_hearth(obs)

        output = "🏠 **Hearth - LOCATION PATTERNS**\n\n"
        output += f"**Observations**: {report.total_observations} "
        output += f"({report.labeled_observations} had location data)\n\n"

        if not report.time_per_label_min:
            output += (
                "No location data yet. Add WiFi/GPS observations via the phone "
                "API or `POST /observe` with `type=wifi` and an SSID/location "
                "in `content`.\n"
            )
        else:
            total = sum(report.time_per_label_min.values())
            output += "**TIME PER PLACE**:\n"
            for label, mins in sorted(report.time_per_label_min.items(),
                                       key=lambda kv: -kv[1]):
                hours = mins / 60.0
                pct = (mins / total * 100) if total else 0
                output += f"  {label:<10s}  {hours:>5.1f}h  ({pct:.0f}%)\n"

            if report.top_transitions:
                output += "\n**FREQUENT TRANSITIONS** (where you go next):\n"
                for a, b, n in report.top_transitions:
                    output += f"  {a} → {b}   ({n}x)\n"

            if report.hour_label_mode:
                output += "\n**YOUR USUAL HOUR-OF-DAY LOCATION**:\n"
                # Compact: show only hours with a mode (skip gaps).
                for h in sorted(report.hour_label_mode.keys()):
                    output += f"  {h:02d}:00 → {report.hour_label_mode[h]}\n"

            if report.anomalies:
                output += "\n**⚠️ TODAY'S ANOMALIES**:\n"
                for a in report.anomalies:
                    output += f"  • {a}\n"

        return AbilityResult(
            True, output,
            {
                "labels": list(report.time_per_label_min.keys()),
                "time_per_label_min": report.time_per_label_min,
                "transitions": report.top_transitions,
                "anomalies": report.anomalies,
                "labeled_observations": report.labeled_observations,
            },
            confidence=round(min(1.0, report.labeled_observations / 30.0), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # 17. TIME WITNESS - WHERE THE HOURS WENT
    # ═════════════════════════════════════════════════════════════════

    async def _time_witness(self, request: str, context: Dict) -> AbilityResult:
        """Show how the last 7 days of attention got spent, and how it
        compares to the prior week."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .time_witness import compute_time_report, format_bar
        except ImportError as e:
            return AbilityResult(False, f"Time Witness unavailable: {e}")

        # Tunable window via request: "this week" / "last 30 days" / default 7.
        window_days = 7
        m = re.search(r"\b(?:last|past)\s+(\d{1,3})\s+days?\b", request, re.I)
        if m:
            window_days = max(1, min(90, int(m.group(1))))

        report = compute_time_report(obs, window_days=window_days)

        output = f"⏳ **Time Witness - LAST {report.window_days} DAYS**\n\n"
        output += f"**Tracked**: {report.total_minutes:.0f} min of attention "
        output += f"({report.total_minutes/60:.1f} h, {report.total_events} events)\n"
        output += f"**Coverage**: {report.coverage:.0%} of the window\n"
        output += f"**Idle/untracked**: {report.idle_minutes/60:.1f} h\n\n"

        if not report.buckets:
            output += "No activity in the window. Send observations via `POST /observe`.\n"
            return AbilityResult(
                True, output,
                {"total_minutes": 0.0, "buckets": [], "window_days": window_days},
                confidence=0.0,
            )

        output += "**WHERE TIME WENT**:\n"
        for b in report.buckets:
            bar = format_bar(b.fraction, width=20)
            delta = f"  Δ{b.delta_vs_prev:+.0f}m vs prev" if abs(b.delta_vs_prev) >= 5 else ""
            output += (
                f"  {b.label:<14s} {bar}  {b.fraction:>4.0%}  "
                f"{b.minutes/60:>5.1f}h  ({b.events} events){delta}\n"
            )
        if report.top_label:
            output += f"\n**Top category**: {report.top_label}\n"

        return AbilityResult(
            True, output,
            {
                "window_days": window_days,
                "total_minutes": report.total_minutes,
                "idle_minutes": report.idle_minutes,
                "coverage": report.coverage,
                "buckets": [
                    {"label": b.label, "minutes": b.minutes,
                     "fraction": b.fraction, "events": b.events,
                     "delta_vs_prev": b.delta_vs_prev, "rank": b.rank}
                    for b in report.buckets
                ],
            },
            confidence=round(min(1.0, report.coverage * 2), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # 18. MIRROR - SELF-REFLECTION FROM YOUR OWN TEXT
    # ═════════════════════════════════════════════════════════════════

    async def _mirror(self, request: str, context: Dict) -> AbilityResult:
        """Surface recurring inner-state themes from the user's own first-person
        text in observations."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .mirror import reflect
        except ImportError as e:
            return AbilityResult(False, f"Mirror unavailable: {e}")

        report = reflect(obs)

        output = "🪞 **Mirror - SELF-REFLECTION**\n\n"
        output += f"**Sentences analyzed**: {report.total_sentences}\n"
        output += f"**Self-talk** (I/me/my): {report.self_talk_sentences}  "
        output += f"({report.self_talk_ratio:.0%})\n"
        if report.self_talk_sentences:
            tone = "constructive" if report.avg_self_sentiment > 0.15 else \
                   "harsh" if report.avg_self_sentiment < -0.15 else "neutral"
            output += f"**Tone of self-talk**: {tone} ({report.avg_self_sentiment:+.2f})\n\n"

        if not report.theme_counts:
            output += (
                "\nNo clear inner-state themes detected yet — try writing more "
                "first-person notes (clipboard, journal observations).\n"
            )
        else:
            output += "**RECURRING THEMES**:\n"
            for theme, n in sorted(report.theme_counts.items(), key=lambda kv: -kv[1])[:8]:
                output += f"  • {theme:<14s} {n}x\n"

        if report.rising_themes:
            output += "\n**RISING this week**: " + ", ".join(report.rising_themes) + "\n"
        if report.declining_themes:
            output += "**DECLINING this week**: " + ", ".join(report.declining_themes) + "\n"

        if report.sample_hits:
            output += "\n**SAMPLES**:\n"
            for hit in report.sample_hits[:6]:
                output += f"  [{hit.theme}] {hit.timestamp[:10]} ({hit.sentiment:+.2f})\n"
                output += f"    \"{hit.excerpt[:140]}\"\n"

        return AbilityResult(
            True, output,
            {
                "self_talk_ratio": report.self_talk_ratio,
                "avg_self_sentiment": report.avg_self_sentiment,
                "themes": report.theme_counts,
                "rising": report.rising_themes,
                "declining": report.declining_themes,
                "sample_count": len(report.sample_hits),
            },
            confidence=round(min(1.0, report.self_talk_sentences / 20.0), 4),
        )

    async def _lantern(self, request: str, context: Dict) -> AbilityResult:
        """Surface project state from file_change observations."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .lantern import compute_lantern
        except ImportError as e:
            return AbilityResult(False, f"Lantern unavailable: {e}")

        report = compute_lantern(obs)

        output = "🏮 **Lantern - PROJECT STATE**\n\n"
        output += f"**Total file changes**: {report.total_changes}\n"
        output += f"**Projects detected**: {len(report.projects)}  "
        output += f"(active: {len(report.active)}, idle: {len(report.idle)}, dormant: {len(report.dormant)})\n\n"

        if report.active:
            output += "**ACTIVE** (touched in last 7 days):\n"
            for p in [pp for pp in report.projects if pp.status == "active"][:5]:
                output += (f"  • {p.name:<24s} {p.change_count:>4}× changes  "
                           f"{p.weekly_frequency:>5.1f}/wk  last={int(p.days_since)}d\n")
        if report.idle:
            output += "\n**IDLE** (7-30 days since touched):\n"
            for p in [pp for pp in report.projects if pp.status == "idle"][:4]:
                output += f"  • {p.name:<24s} {p.change_count:>4}× changes  last={int(p.days_since)}d\n"
        if report.dormant:
            output += "\n**DORMANT** (>30 days):\n"
            for p in [pp for pp in report.projects if pp.status == "dormant"][:4]:
                output += f"  • {p.name:<24s} {p.change_count:>4}× changes  last={int(p.days_since)}d\n"
        if report.revisit_suggestions:
            output += "\n**REVISIT CANDIDATES**:\n"
            for s in report.revisit_suggestions:
                output += f"  ↻ {s}\n"
        if not report.projects:
            output += "\nNo file_change observations yet — emit some via the file watcher.\n"

        return AbilityResult(
            True, output,
            {
                "total_changes": report.total_changes,
                "active": report.active,
                "idle": report.idle,
                "dormant": report.dormant,
                "projects": [
                    {
                        "name": p.name, "status": p.status,
                        "change_count": p.change_count,
                        "file_count": p.file_count,
                        "days_since": p.days_since,
                        "weekly_frequency": p.weekly_frequency,
                        "first_seen": p.first_seen,
                        "last_seen": p.last_seen,
                        "top_files": p.top_files,
                    }
                    for p in report.projects[:30]
                ],
                "revisit_suggestions": report.revisit_suggestions,
            },
            confidence=round(min(1.0, len(report.projects) / 5.0), 4),
        )

    async def _tide(self, request: str, context: Dict) -> AbilityResult:
        """Weekly sentiment trend across text observations."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .tide import compute_tide
        except ImportError as e:
            return AbilityResult(False, f"Tide unavailable: {e}")

        report = compute_tide(obs)

        arrow = "→" if report.slope == "stable" else ("↑" if report.slope == "rising" else "↓")
        output = "🌊 **Tide - EMOTIONAL TREND**\n\n"
        output += f"**Samples**: {report.total_samples}  weeks={len(report.weeks)}\n"
        output += f"**Overall**: {report.overall_label} ({report.overall_avg:+.2f}) {arrow} {report.slope}\n"
        if report.weeks:
            output += f"**Slope**: {report.slope_value:+.3f} per week\n"
        if report.best_week and report.best_week != report.worst_week:
            output += f"**Best week**: {report.best_week}\n"
            output += f"**Worst week**: {report.worst_week}\n"
        output += f"**Longest positive streak**: {report.longest_positive_streak}\n"
        output += f"**Longest negative streak**: {report.longest_negative_streak}\n"

        if report.weeks:
            output += "\n**WEEKLY**:\n"
            for w in report.weeks[-12:]:
                bar = "█" * int(round(abs(w.avg_sentiment) * 10))
                sign = "+" if w.avg_sentiment >= 0 else "-"
                output += f"  {w.iso_week}  n={w.sample_count:>3}  {w.avg_sentiment:+.2f}  {sign}{bar}\n"
        else:
            output += "\n" + report.summary + "\n"

        return AbilityResult(
            True, output,
            {
                "total_samples": report.total_samples,
                "overall_avg": report.overall_avg,
                "overall_label": report.overall_label,
                "slope": report.slope,
                "slope_value": report.slope_value,
                "best_week": report.best_week,
                "worst_week": report.worst_week,
                "longest_positive_streak": report.longest_positive_streak,
                "longest_negative_streak": report.longest_negative_streak,
                "weeks": [
                    {
                        "iso_week": w.iso_week, "start": w.start, "end": w.end,
                        "sample_count": w.sample_count,
                        "avg_sentiment": w.avg_sentiment, "label": w.label,
                        "top_positive": w.top_positive,
                        "top_negative": w.top_negative,
                    }
                    for w in report.weeks
                ],
                "summary": report.summary,
            },
            confidence=round(min(1.0, report.total_samples / 30.0), 4),
        )

    async def _echo(self, request: str, context: Dict) -> AbilityResult:
        """Counterfactual analysis over past decisions."""
        with self._lock:
            obs = list(context.get("observations", self.observations))

        try:
            from .echo import compute_echo
        except ImportError as e:
            return AbilityResult(False, f"Echo unavailable: {e}")

        report = compute_echo(obs, decision_tracker=self.decision_tracker)

        output = "🌀 **Echo - COUNTERFACTUALS**\n\n"
        output += f"**Decisions in history**: {report.decisions_seen}\n"
        output += f"**Counterfactuals generated**: {len(report.counterfactuals)}\n"
        output += f"\n{report.summary}\n"

        if report.top_regret:
            c = report.top_regret
            output += f"\n**TOP REGRET PRESSURE** (Δ{c.delta:+.1f}):\n"
            output += f"  Your choice : {c.decision.choice[:80]}\n"
            output += f"  Alternative : {c.alt_choice[:80]}\n"
            output += f"  History: chose-side n={c.chosen_sample_n} (avg {c.chosen_avg_sat:.1f}), "
            output += f"alt-side n={c.alt_sample_n} (avg {c.alt_avg_sat:.1f})\n"

        if report.top_relief:
            c = report.top_relief
            output += f"\n**TOP RELIEF** (Δ{c.delta:+.1f}):\n"
            output += f"  Your choice : {c.decision.choice[:80]}\n"
            output += f"  Avoided     : {c.alt_choice[:80]}\n"

        if report.counterfactuals:
            output += "\n**OTHER COUNTERFACTUALS**:\n"
            for c in report.counterfactuals[:5]:
                output += f"  • Δ{c.delta:+.1f}  {c.alt_choice[:60]}\n"

        if not report.counterfactuals:
            output += (
                "\nNo counterfactuals yet — Echo needs decisions in DecisionTracker "
                "with satisfaction scores. Use track_decision() to record outcomes.\n"
            )

        return AbilityResult(
            True, output,
            {
                "decisions_seen": report.decisions_seen,
                "counterfactual_count": len(report.counterfactuals),
                "top_regret": (
                    {
                        "alt_choice": report.top_regret.alt_choice,
                        "delta": report.top_regret.delta,
                        "regret_pressure": report.top_regret.regret_pressure,
                        "chosen_avg_sat": report.top_regret.chosen_avg_sat,
                        "alt_avg_sat": report.top_regret.alt_avg_sat,
                        "narrative": report.top_regret.narrative,
                    } if report.top_regret else None
                ),
                "top_relief": (
                    {
                        "alt_choice": report.top_relief.alt_choice,
                        "delta": report.top_relief.delta,
                        "narrative": report.top_relief.narrative,
                    } if report.top_relief else None
                ),
                "summary": report.summary,
            },
            confidence=round(min(1.0, len(report.counterfactuals) / 5.0), 4),
        )

    # ═════════════════════════════════════════════════════════════════
    # TRACKING API
    # ═════════════════════════════════════════════════════════════════
    
    _MAX_PEOPLE = 1000
    _MAX_DEVICES_PER_PERSON = 50

    def track_person(self, name: str, devices: str, relation: str = "") -> str:
        """Track a person by their devices."""
        name = _SANITIZE_STRICT_RE.sub('', name.strip())[:200]
        device_list = [d.strip() for d in devices.replace(",", " ").split() if ":" in d or len(d) > 8]
        
        with self._lock:
            if name not in self.people:
                self.people[name] = {"devices": [], "patterns": {}, "relation": relation}
            
            self.people[name]["devices"].extend(device_list[:self._MAX_DEVICES_PER_PERSON])
            self.people[name]["devices"] = list(set(self.people[name]["devices"]))[:self._MAX_DEVICES_PER_PERSON]
            
            if len(self.people) > self._MAX_PEOPLE:
                oldest = list(self.people.keys())[:self._MAX_PEOPLE // 2]
                for k in oldest:
                    del self.people[k]
            
            for device in device_list:
                self.device_fingerprinter._known_devices[device.upper()] = {
                    "name": name,
                    "device_type": "unknown"
                }
        
        self._save_json(PEOPLE_DB, self.people)
        return f"Tracking {name} with {len(device_list)} devices"
    
    def track_decision(self, situation: str, choice: str, outcome: str = None, 
                       satisfaction: int = None) -> str:
        """Record a decision for future reference."""
        with self._stats_lock:
            self._stats["decisions_tracked"] += 1
        entry = self.decision_tracker.record_decision(situation, choice, outcome, satisfaction)
        return f"Decision tracked: {_SANITIZE_STRICT_RE.sub('', choice[:80])} (ID: {entry['id']})"
    
    def track_pattern(self, cause: str, effect: str, observed: bool = True) -> str:
        """Track a causal pattern."""
        cause = _SANITIZE_RE.sub('', cause.strip())[:200]
        effect = _SANITIZE_RE.sub('', effect.strip())[:200]
        with self._lock:
            if effect not in self.patterns:
                self.patterns[effect] = {}
            if cause not in self.patterns[effect]:
                self.patterns[effect][cause] = {"occurred": 0, "total": 0}
            
            self.patterns[effect][cause]["total"] += 1
            if observed:
                self.patterns[effect][cause]["occurred"] += 1
            
            if len(self.patterns) > _MAX_PATTERNS:
                oldest = list(self.patterns.keys())[:_MAX_PATTERNS // 2]
                for k in oldest:
                    del self.patterns[k]
        
        self._save_json(PATTERNS_DB, self.patterns)
        return f"Pattern tracked: {cause} → {effect}"
    
    def _save_json(self, path: Path, data):
        """Save JSON data (atomic with fsync)."""
        try:
            tmp = path.with_suffix(".tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(fd, 'w') as f:
                    json.dump(data, f, indent=2)
                    f.flush()
                    os.fsync(f.fileno())
            except OSError:
                os.close(fd)
                raise
            os.replace(str(tmp), str(path))
        except (OSError, TypeError, ValueError) as e:
            logger.error("Save error: %s", type(e).__name__)
    
    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════
    
    async def execute(self, shikigami_type: ShikigamiType, request: str, 
                      context: Dict = None) -> AbilityResult:
        """Execute a Shikigami ability."""
        if context is None:
            context = {"observations": self.observations}
        
        methods = {
            ShikigamiType.ORACLE: self._oracle,
            ShikigamiType.X_RAY: self._xray,
            ShikigamiType.LIFE_PULSE: self._life_pulse,
            ShikigamiType.PERFECT_MEMORY: self._perfect_memory,
            ShikigamiType.PATTERN_SIGHT: self._pattern_sight,
            ShikigamiType.SOCIAL_RADAR: self._social_radar,
            ShikigamiType.INTENT_SEER: self._intent_seer,
            ShikigamiType.OMNI_SENSE: self._omni_sense,
            ShikigamiType.SHADOW_COUNCIL: self._shadow_council,
            ShikigamiType.REALITY_FORGE: self._reality_forge,
            ShikigamiType.HABIT_TRACKER: self._habit_tracker,
            ShikigamiType.ENERGY_FLOW: self._energy_flow,
            ShikigamiType.FOCUS_SIGHT: self._focus_sight,
            ShikigamiType.CONNECTION: self._connection,
            ShikigamiType.COMPASS: self._compass,
            ShikigamiType.HEARTH: self._hearth,
            ShikigamiType.TIME_WITNESS: self._time_witness,
            ShikigamiType.MIRROR: self._mirror,
            ShikigamiType.LANTERN: self._lantern,
            ShikigamiType.TIDE: self._tide,
            ShikigamiType.ECHO: self._echo,
        }
        
        method = methods.get(shikigami_type)
        if method:
            return await method(request, context)
        
        return AbilityResult(False, f"Unknown ability: {shikigami_type}")
    
    def list_shikigami(self) -> List[Dict]:
        """List all abilities."""
        return [
            {"name": "Oracle", "type": "oracle", "emoji": "🐍", "power": "Monte Carlo prediction"},
            {"name": "X-Ray", "type": "x_ray", "emoji": "👁️", "power": "Device fingerprinting"},
            {"name": "Life Pulse", "type": "life_pulse", "emoji": "💓", "power": "Constellation matching"},
            {"name": "Perfect Memory", "type": "perfect_memory", "emoji": "🧠", "power": "Semantic search"},
            {"name": "Pattern Sight", "type": "pattern_sight", "emoji": "👁️‍🗨️", "power": "FFT & causation"},
            {"name": "Social Radar", "type": "social_radar", "emoji": "📡", "power": "Behavior prediction"},
            {"name": "Intent Seer", "type": "intent_seer", "emoji": "🎯", "power": "Markov chains"},
            {"name": "Omni Sense", "type": "omni_sense", "emoji": "🔮", "power": "Sensor fusion"},
            {"name": "Shadow Council", "type": "shadow_council", "emoji": "👥", "power": "Decision learning"},
            {"name": "Reality Forge", "type": "reality_forge", "emoji": "🔨", "power": "File/command execution"},
            {"name": "Habit Tracker", "type": "habit_tracker", "emoji": "🌀", "power": "Recurring behavior detection"},
            {"name": "Energy Flow", "type": "energy_flow", "emoji": "⚡", "power": "Circadian energy curve & forecast"},
            {"name": "Focus Sight", "type": "focus_sight", "emoji": "🎯", "power": "Deep-work block detection"},
            {"name": "Connection", "type": "connection", "emoji": "🔗", "power": "Relationship strength scoring"},
            {"name": "Compass", "type": "compass", "emoji": "🧭", "power": "Goal alignment & drift detection"},
            {"name": "Hearth", "type": "hearth", "emoji": "🏠", "power": "Location patterns & transitions"},
            {"name": "Time Witness", "type": "time_witness", "emoji": "⏳", "power": "Where the hours actually went"},
            {"name": "Mirror", "type": "mirror", "emoji": "🪞", "power": "Self-reflection from your own text"},
            {"name": "Lantern", "type": "lantern", "emoji": "🏮", "power": "Project state — active, idle, dormant"},
            {"name": "Tide", "type": "tide", "emoji": "🌊", "power": "Emotional sentiment trend over time"},
            {"name": "Echo", "type": "echo", "emoji": "🌀", "power": "Counterfactual analysis of past decisions"},
        ]
    
    def get_status(self) -> Dict:
        """Get registry status with thread-safe copies."""
        with self._lock:
            return {
                "people_tracked": len(self.people),
                "observations": len(self.observations),
                "patterns_tracked": len(self.patterns),
                "kalman_filters": len(self.kalman_filters),
                "stats": dict(self._stats),
            }
    
    def auto_select(self, request: str) -> ShikigamiType:
        """Auto-select ability based on request."""
        r = request.lower()

        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["will", "predict", "happen", "probability", "chance"]):
            return ShikigamiType.ORACLE
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["is anyone", "is someone", "who's", "nearby", "present", "here"]):
            return ShikigamiType.X_RAY
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["is sarah", "is john", "where is", "track"]):
            return ShikigamiType.LIFE_PULSE
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["remember", "when did", "what did", "recall", "history"]):
            return ShikigamiType.PERFECT_MEMORY
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["pattern", "why", "trend", "correlation", "cause", "cycle"]):
            return ShikigamiType.PATTERN_SIGHT
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["will they", "what will", "behavior", "lying", "manipulation"]):
            return ShikigamiType.SOCIAL_RADAR
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["going to", "about to", "planning", "intent", "next action"]):
            return ShikigamiType.INTENT_SEER
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["around me", "environment", "where am i", "what's happening", "snapshot"]):
            return ShikigamiType.OMNI_SENSE
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["should i", "past", "before", "chose", "recommend"]):
            return ShikigamiType.SHADOW_COUNCIL
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["create", "build", "make", "set up", "generate", "write"]):
            return ShikigamiType.REALITY_FORGE
        
        return ShikigamiType.OMNI_SENSE


# Singleton
_registry = None

def get_registry() -> EnhancedShikigamiRegistry:
    global _registry
    if _registry is None:
        _registry = EnhancedShikigamiRegistry()
    return _registry