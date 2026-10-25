#!/usr/bin/env python3
"""
OracleBridge - Native Prediction Integration for ShadowOS

Bridges the Oracle AI prediction engine into ShadowOS cognition,
providing foresight capabilities without leaving the cognitive platform.

Architecture:
  - PredictionDomain: 37 Oracle domains mapped to ShadowOS cognition
  - DomainStateSnapshot: Current state of a prediction domain
  - PredictionResult: A prediction with confidence intervals
  - ForecastSummary: Aggregate forecast across domains
  - OracleBridge: Main bridge managing predictions, diagnoses, and state
    - HTTP fallback for remote Oracle service (port 8005)
    - Daily oracle synthesis, compound hazard assessment, regime shift detection, prophecy generation

  The bridge does NOT replicate Oracle's domain models — it delegates to
  OracleEngine for actual computation and provides:
  - ShadowOS-compatible API (thread-safe, bounded, sanitized, deepcopy)
  - Integration with ShadowOS modules (TemporalReasoning, AnomalyDetector, etc.)
  - Lightweight state caching so repeated predictions are fast
  - Domain state translation between ShadowOS and Oracle formats

  Thread-safe, bounded collections, input sanitization
"""

import re
import copy
import uuid
import logging
import threading
from datetime import date, datetime, timezone, timedelta
from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Dict, List, Optional, Tuple
from collections import deque

logger = logging.getLogger("shadowd.cognition.oracle_bridge")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')

_MAX_STATE_HISTORY = 500
_MAX_PREDICTIONS = 200
_MAX_DIAGNOSES = 100
_MAX_FORECASTS = 50
_MAX_STRICT = 200
_MAX_TEXT = 10000


def _sanitize(text: str, max_length: int = _MAX_STRICT) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = _MAX_TEXT) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


class PredictionDomain(str, Enum):
    ECONOMICS = "economics"
    FINANCE = "finance"
    GEOPOLITICS = "geopolitics"
    TECHNOLOGY = "technology"
    CLIMATE = "climate"
    DEMOGRAPHICS = "demographics"
    HEALTH = "health"
    EPIDEMIOLOGY = "epidemiology"
    ENERGY = "energy"
    COMMODITIES = "commodities"
    SOCIAL = "social"
    INFRASTRUCTURE = "infrastructure"
    SUPPLY_CHAIN = "supply_chain"
    CYBER = "cyber"
    WATER = "water"
    FOOD = "food"
    POLITICAL = "political"
    BIODIVERSITY = "biodiversity"
    PANDEMIC = "pandemic"
    SPACE = "space"
    MENTAL_HEALTH = "mental_health"
    HOUSING = "housing"
    SEISMIC = "seismic"
    SOLAR = "solar"
    OCEANIC = "oceanic"
    NEO = "near_earth_object"
    ATMOSPHERIC = "atmospheric"
    ECOLOGICAL = "ecological"
    PALEOCLIMATE = "paleoclimate"
    COSMIC_RADIATION = "cosmic_radiation"
    GRAVITATIONAL = "gravitational"
    COMPOUND_HAZARD = "compound_hazard"
    WORLD_STATE = "world_state"
    FATE = "fate"
    PERSONAL_LIFE = "personal_life"


class ConfidenceLevel(str, Enum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    VERY_HIGH = "very_high"


CONFIDENCE_MAP = {
    ConfidenceLevel.LOW: 0.68,
    ConfidenceLevel.MEDIUM: 0.90,
    ConfidenceLevel.HIGH: 0.95,
    ConfidenceLevel.VERY_HIGH: 0.99,
}

_DOMAIN_WEIGHT_MAP = {
    PredictionDomain.ECONOMICS: 0.9,
    PredictionDomain.GEOPOLITICS: 0.85,
    PredictionDomain.CLIMATE: 0.8,
    PredictionDomain.HEALTH: 0.75,
    PredictionDomain.TECHNOLOGY: 0.8,
    PredictionDomain.CYBER: 0.7,
    PredictionDomain.ENERGY: 0.75,
    PredictionDomain.SUPPLY_CHAIN: 0.7,
}


@dataclass
class DomainStateSnapshot:
    domain: PredictionDomain = PredictionDomain.ECONOMICS
    state_values: Dict = field(default_factory=dict)
    timestamp: str = ""
    source: str = "manual"

    def __post_init__(self):
        if isinstance(self.domain, str):
            try:
                self.domain = PredictionDomain(self.domain)
            except ValueError:
                self.domain = PredictionDomain.ECONOMICS
        self.state_values = {str(k): float(v) if isinstance(v, (int, float)) else str(v)
                            for k, v in list(self.state_values.items())[:50]}
        self.source = _sanitize(self.source, 100)
        if not self.timestamp:
            self.timestamp = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "domain": self.domain.value,
            "state_values": {k: round(v, 4) if isinstance(v, float) else v
                            for k, v in self.state_values.items()},
            "timestamp": self.timestamp,
            "source": self.source,
        }


@dataclass
class PredictionResult:
    prediction_id: str = ""
    domain: PredictionDomain = PredictionDomain.ECONOMICS
    variable: str = ""
    predicted_value: float = 0.0
    confidence_interval: Tuple[float, float] = (0.0, 0.0)
    confidence_level: ConfidenceLevel = ConfidenceLevel.MEDIUM
    target_date: str = ""
    model_name: str = ""
    signals_used: List[str] = field(default_factory=list)
    created_at: str = ""

    def __post_init__(self):
        if not self.prediction_id:
            self.prediction_id = uuid.uuid4().hex[:12]
        self.prediction_id = _sanitize(self.prediction_id)
        if isinstance(self.domain, str):
            try:
                self.domain = PredictionDomain(self.domain)
            except ValueError:
                self.domain = PredictionDomain.ECONOMICS
        if isinstance(self.confidence_level, str):
            try:
                self.confidence_level = ConfidenceLevel(self.confidence_level)
            except ValueError:
                self.confidence_level = ConfidenceLevel.MEDIUM
        self.variable = _sanitize(self.variable, 500)
        self.model_name = _sanitize(self.model_name, 200)
        self.signals_used = [_sanitize(s, 200) for s in (self.signals_used or [])[:20]]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        ci = self.confidence_interval
        return {
            "prediction_id": self.prediction_id,
            "domain": self.domain.value,
            "variable": self.variable,
            "predicted_value": round(self.predicted_value, 4),
            "confidence_interval": [round(ci[0], 4), round(ci[1], 4)],
            "confidence_level": self.confidence_level.value,
            "target_date": self.target_date,
            "model_name": self.model_name,
            "signals_used": list(self.signals_used),
            "created_at": self.created_at,
        }


@dataclass
class ForecastSummary:
    forecast_id: str = ""
    target_date: str = ""
    domains_predicted: List[str] = field(default_factory=list)
    predictions: List[PredictionResult] = field(default_factory=list)
    diagnosis_summary: Dict = field(default_factory=dict)
    confidence_level: ConfidenceLevel = ConfidenceLevel.HIGH
    created_at: str = ""

    def __post_init__(self):
        if not self.forecast_id:
            self.forecast_id = uuid.uuid4().hex[:12]
        self.forecast_id = _sanitize(self.forecast_id)
        if isinstance(self.confidence_level, str):
            try:
                self.confidence_level = ConfidenceLevel(self.confidence_level)
            except ValueError:
                self.confidence_level = ConfidenceLevel.HIGH
        self.domains_predicted = [_sanitize(d, 50) for d in (self.domains_predicted or [])[:30]]
        if not self.created_at:
            self.created_at = datetime.now(timezone.utc).isoformat()

    def to_dict(self) -> Dict:
        return {
            "forecast_id": self.forecast_id,
            "target_date": self.target_date,
            "domains_predicted": list(self.domains_predicted),
            "prediction_count": len(self.predictions),
            "predictions": [p.to_dict() for p in self.predictions[:20]],
            "diagnosis_summary": copy.deepcopy(self.diagnosis_summary) if self.diagnosis_summary else {},
            "confidence_level": self.confidence_level.value,
            "created_at": self.created_at,
        }


_SETTER_MAP = {
    PredictionDomain.ECONOMICS: "set_economic_state",
    PredictionDomain.FINANCE: "set_economic_state",
    PredictionDomain.GEOPOLITICS: "set_geopolitical_state",
    PredictionDomain.CLIMATE: "set_climate_state",
    PredictionDomain.DEMOGRAPHICS: "set_demographics_state",
    PredictionDomain.HEALTH: "set_epidemiology_state",
    PredictionDomain.EPIDEMIOLOGY: "set_epidemiology_state",
    PredictionDomain.TECHNOLOGY: "set_technology_state",
    PredictionDomain.SOCIAL: "set_social_state",
    PredictionDomain.ENERGY: "set_energy_state",
    PredictionDomain.COMMODITIES: "set_commodities_state",
    PredictionDomain.INFRASTRUCTURE: "set_infrastructure_state",
    PredictionDomain.SUPPLY_CHAIN: "set_supply_chain_state",
    PredictionDomain.CYBER: "set_cyber_state",
    PredictionDomain.WATER: "set_water_state",
    PredictionDomain.FOOD: "set_food_state",
    PredictionDomain.POLITICAL: "set_political_state",
    PredictionDomain.BIODIVERSITY: "set_biodiversity_state",
    PredictionDomain.PANDEMIC: "set_pandemic_state",
    PredictionDomain.SPACE: "set_space_state",
    PredictionDomain.MENTAL_HEALTH: "set_mental_health_state",
    PredictionDomain.HOUSING: "set_housing_state",
    PredictionDomain.SEISMIC: "set_seismic_state",
    PredictionDomain.SOLAR: "set_solar_state",
    PredictionDomain.OCEANIC: "set_oceanic_state",
    PredictionDomain.NEO: "set_neo_state",
    PredictionDomain.ATMOSPHERIC: "set_atmospheric_state",
    PredictionDomain.ECOLOGICAL: "set_ecological_state",
    PredictionDomain.PALEOCLIMATE: "set_paleoclimate_state",
    PredictionDomain.COSMIC_RADIATION: "set_cosmic_radiation_state",
    PredictionDomain.GRAVITATIONAL: "set_gravitational_state",
    PredictionDomain.COMPOUND_HAZARD: "set_compound_hazard_state",
    PredictionDomain.WORLD_STATE: "set_compound_hazard_state",
    PredictionDomain.FATE: "set_fate_state",
    PredictionDomain.PERSONAL_LIFE: "set_personal_fate_state",
}


class OracleBridge:
    """
    Bridge between ShadowOS cognition and Oracle prediction engine.
    Provides foresight capabilities natively within the cognitive platform.
    """

    def __init__(self, oracle_engine=None, oracle_url: Optional[str] = None):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._engine = oracle_engine
        self._oracle_url = oracle_url or "http://localhost:8005"
        self._state_snapshots: Dict[str, DomainStateSnapshot] = {}
        self._state_history: deque = deque(maxlen=_MAX_STATE_HISTORY)
        self._predictions: Dict[str, PredictionResult] = {}
        self._prediction_order: deque = deque(maxlen=_MAX_PREDICTIONS)
        self._diagnoses: deque = deque(maxlen=_MAX_DIAGNOSES)
        self._forecasts: Dict[str, ForecastSummary] = {}
        self._forecast_order: deque = deque(maxlen=_MAX_FORECASTS)

        self._stats = {
            "states_set": 0,
            "predictions_generated": 0,
            "forecasts_generated": 0,
            "diagnoses_run": 0,
            "domains_active": 0,
            "temporal_ingestions": 0,
            "anomaly_ingestions": 0,
            "daily_oracle_calls": 0,
            "compound_hazard_assessments": 0,
            "regime_shift_assessments": 0,
            "prophecies_generated": 0,
        }

    def _get_engine(self):
        if self._engine is not None:
            return self._engine
        with self._lock:
            if self._engine is not None:
                return self._engine
            try:
                from oracle.prediction.engine import OracleEngine
                self._engine = OracleEngine()
                return self._engine
            except ImportError:
                logger.warning("Oracle engine not available - install oracle package")
                return None

    def _http_post(self, path: str, payload: Dict) -> Optional[Dict]:
        try:
            import httpx
        except ImportError:
            return None
        try:
            url = f"{self._oracle_url}/api{path}"
            resp = httpx.post(url, json=payload, timeout=2.0)
            if resp.status_code == 200:
                return resp.json()
        except httpx.HTTPError as e:
            logger.debug("Oracle HTTP call failed: %s", type(e).__name__)
        except (OSError, ValueError, TypeError) as e:
            logger.debug("Oracle HTTP call failed: %s", type(e).__name__)
        return None

    def _http_get(self, path: str) -> Optional[Dict]:
        try:
            import httpx
        except ImportError:
            return None
        try:
            url = f"{self._oracle_url}/api{path}"
            resp = httpx.get(url, timeout=2.0)
            if resp.status_code == 200:
                return resp.json()
        except httpx.HTTPError as e:
            logger.debug("Oracle HTTP call failed: %s", type(e).__name__)
        except (OSError, ValueError, TypeError) as e:
            logger.debug("Oracle HTTP call failed: %s", type(e).__name__)
        return None

    def set_domain_state(
        self,
        domain: str,
        state_values: Dict,
        source: str = "manual",
    ) -> DomainStateSnapshot:
        domain_enum = PredictionDomain(domain) if domain in [e.value for e in PredictionDomain] else PredictionDomain.ECONOMICS
        snapshot = DomainStateSnapshot(
            domain=domain_enum,
            state_values=state_values,
            source=source,
        )
        with self._lock:
            self._state_snapshots[domain_enum.value] = snapshot
            self._state_history.append(snapshot.to_dict())

        engine = self._get_engine()
        if engine:
            setter_name = _SETTER_MAP.get(domain_enum)
            if setter_name and hasattr(engine, setter_name):
                setter = getattr(engine, setter_name)
                try:
                    numeric_values = {k: v for k, v in state_values.items() if isinstance(v, (int, float)) and isinstance(k, str)}
                    setter(**numeric_values)
                except (TypeError, ValueError) as e:
                    logger.warning("Oracle setter %s failed: %s", setter_name, type(e).__name__)

        with self._lock:
            snapshot_count = len(self._state_snapshots)
        with self._stats_lock:
            self._stats["states_set"] += 1
            self._stats["domains_active"] = snapshot_count

        return copy.deepcopy(snapshot)

    def predict(
        self,
        target_date: Optional[str] = None,
        confidence: str = "high",
        domains: Optional[List[str]] = None,
    ) -> ForecastSummary:
        if not target_date:
            target_date = (date.today() + timedelta(days=365)).isoformat()
        try:
            target = date.fromisoformat(target_date)
        except ValueError:
            target = date.today() + timedelta(days=365)

        try:
            conf_enum = ConfidenceLevel(confidence)
        except ValueError:
            conf_enum = ConfidenceLevel.HIGH

        engine = self._get_engine()
        predictions = []
        diagnosis_summary = {}

        if engine:
            try:
                from oracle.core.types import Domain, ConfidenceLevel as OracleConf
                conf_map = {
                    ConfidenceLevel.LOW: OracleConf.LOW,
                    ConfidenceLevel.MEDIUM: OracleConf.MEDIUM,
                    ConfidenceLevel.HIGH: OracleConf.HIGH,
                    ConfidenceLevel.VERY_HIGH: OracleConf.VERY_HIGH,
                }
                oracle_conf = conf_map.get(conf_enum, OracleConf.HIGH)
                forecast = engine.predict(target, oracle_conf)
                domain_filter = set(domains) if domains else None

                for pred in forecast.predictions:
                    domain_val = pred.domain.value if hasattr(pred.domain, 'value') else str(pred.domain)
                    if domain_filter and domain_val not in domain_filter:
                        continue
                    try:
                        pd_enum = PredictionDomain(domain_val)
                    except ValueError:
                        pd_enum = PredictionDomain.ECONOMICS
                    pr = PredictionResult(
                        domain=pd_enum,
                        variable=_sanitize_content(pred.variable, 500) if hasattr(pred, 'variable') else "unknown",
                        predicted_value=float(pred.value) if hasattr(pred, 'value') else 0.0,
                        confidence_interval=tuple(pred.confidence_interval) if hasattr(pred, 'confidence_interval') else (0.0, 0.0),
                        confidence_level=conf_enum,
                        target_date=target_date,
                        model_name=_sanitize(pred.model_name, 200) if hasattr(pred, 'model_name') else "oracle",
                        signals_used=list(pred.signals_used)[:20] if hasattr(pred, 'signals_used') else [],
                    )
                    predictions.append(pr)
            except (ImportError, TypeError, ValueError, AttributeError, KeyError, RuntimeError, OSError) as e:
                logger.warning("Oracle predict failed: %s", type(e).__name__)
                diagnosis_summary["error"] = type(e).__name__

        if not predictions:
            http_result = self._http_post("/predict", {
                "target_date": target_date,
                "confidence": confidence,
                "domains": domains or [],
            })
            domain_filter_set = set(domains) if domains else None
            if http_result and isinstance(http_result, dict):
                for pred_data in http_result.get("predictions", [])[:_MAX_PREDICTIONS]:
                    try:
                        dom_str = pred_data.get("domain", "economics")
                        try:
                            pd_enum = PredictionDomain(dom_str)
                        except ValueError:
                            pd_enum = PredictionDomain.ECONOMICS
                        if domain_filter_set and pd_enum.value not in domain_filter_set:
                            continue
                        try:
                            ci = tuple(pred_data.get("confidence_interval", [0.0, 0.0]))
                        except (TypeError, ValueError):
                            ci = (0.0, 0.0)
                        pr = PredictionResult(
                            domain=pd_enum,
                            variable=_sanitize(pred_data.get("variable", ""), 500),
                            predicted_value=float(pred_data.get("predicted_value", 0.0)),
                            confidence_interval=ci,
                            confidence_level=conf_enum,
                            target_date=target_date,
                            model_name=_sanitize(pred_data.get("model_name", ""), 200),
                        )
                        predictions.append(pr)
                    except (ValueError, TypeError, KeyError):
                        continue

        if not predictions:
            snapshot = self._generate_heuristic_prediction(target_date, domains)
            predictions = snapshot

        fs = ForecastSummary(
            target_date=target_date,
            domains_predicted=list(set(p.domain.value for p in predictions)),
            predictions=predictions,
            diagnosis_summary=diagnosis_summary,
            confidence_level=conf_enum,
        )
        with self._lock:
            self._forecasts[fs.forecast_id] = fs
            self._forecast_order.append(fs.forecast_id)
            for p in predictions:
                self._predictions[p.prediction_id] = p
                self._prediction_order.append(p.prediction_id)
                if len(self._predictions) > _MAX_PREDICTIONS:
                    evict_id = self._prediction_order.popleft()
                    self._predictions.pop(evict_id, None)
            if len(self._forecasts) > _MAX_FORECASTS:
                evict_id = self._forecast_order.popleft()
                self._forecasts.pop(evict_id, None)
        with self._stats_lock:
            self._stats["predictions_generated"] += len(predictions)
            self._stats["forecasts_generated"] += 1

        return copy.deepcopy(fs)

    def _generate_heuristic_prediction(
        self,
        target_date: str,
        domains: Optional[List[str]] = None,
    ) -> List[PredictionResult]:
        target_domains = [PredictionDomain(d) for d in (domains or []) if d in [e.value for e in PredictionDomain]]
        if not target_domains:
            with self._lock:
                target_domains = [s.domain for s in self._state_snapshots.values()] or [PredictionDomain.ECONOMICS]

        results = []
        for dom in target_domains[:5]:
            weight = _DOMAIN_WEIGHT_MAP.get(dom, 0.5)
            ci_width = 0.1 * (1.0 - weight)
            results.append(PredictionResult(
                domain=dom,
                variable="composite_index",
                predicted_value=round(weight * 100, 2),
                confidence_interval=(round((weight - ci_width) * 100, 2), round((weight + ci_width) * 100, 2)),
                confidence_level=ConfidenceLevel.MEDIUM,
                target_date=target_date,
                model_name="heuristic",
                signals_used=["domain_weight", "state_history"],
            ))
        return results

    def diagnose(self) -> Dict:
        engine = self._get_engine()
        if engine:
            try:
                diag = engine.diagnose()
                summary = {}
                for key in ["summary", "cycle_diagnosis", "regime_analysis", "anomalies"]:
                    if key in diag:
                        val = diag[key]
                        if isinstance(val, dict):
                            summary[key] = {str(k): str(v)[:200] for k, v in list(val.items())[:10]}
                        elif isinstance(val, list):
                            summary[key] = f"{len(val)} items"
                        else:
                            summary[key] = str(val)[:200]
                with self._lock:
                    self._diagnoses.append({
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "summary": summary,
                    })
                with self._stats_lock:
                    self._stats["diagnoses_run"] += 1
                return summary
            except (ImportError, TypeError, ValueError, AttributeError, KeyError, RuntimeError, OSError) as e:
                logger.warning("Oracle diagnose failed: %s", type(e).__name__)
                return {"error": type(e).__name__, "engine": "oracle_failed"}
        return {"status": "no_engine", "engine": "not_initialized"}

    def get_predictions(
        self,
        domain: Optional[str] = None,
        limit: int = 20,
    ) -> List[PredictionResult]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            results = []
            for p in self._predictions.values():
                if domain and p.domain.value != domain:
                    continue
                results.append(copy.deepcopy(p))
                if len(results) >= limit:
                    break
        return results

    def get_forecasts(self, limit: int = 10) -> List[ForecastSummary]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            results = []
            for fid in reversed(list(self._forecast_order)):
                fs = self._forecasts.get(fid)
                if fs:
                    results.append(copy.deepcopy(fs))
                    if len(results) >= limit:
                        break
        return results

    def get_domain_states(self) -> Dict:
        with self._lock:
            return {k: v.to_dict() for k, v in self._state_snapshots.items()}

    def get_diagnoses(self, limit: int = 10) -> List[Dict]:
        limit = max(1, min(limit, 1000))
        with self._lock:
            return copy.deepcopy(list(self._diagnoses)[-limit:])

    def get_stats(self) -> Dict:
        with self._stats_lock:
            return dict(self._stats)

    def get_available_domains(self) -> List[Dict]:
        return [{"domain": d.value, "setter": _SETTER_MAP.get(d, "unknown")} for d in PredictionDomain]

    def ingest_temporal_data(self, temporal_reasoning) -> None:
        """
        Ingest temporal patterns from TemporalReasoning into Oracle domain states.
        Translates detected patterns into Oracle-compatible state snapshots.
        """
        try:
            patterns = None
            if hasattr(temporal_reasoning, 'detect_patterns'):
                patterns = temporal_reasoning.detect_patterns()
            if not patterns:
                return
            for pattern in patterns:
                try:
                    ptype = pattern.pattern_type if hasattr(pattern, 'pattern_type') else str(pattern)
                    freq = pattern.frequency if hasattr(pattern, 'frequency') else 0.5
                    conf = pattern.confidence if hasattr(pattern, 'confidence') else 0.5
                    state_values = {
                        "pattern_type": str(ptype),
                        "frequency": round(freq, 4),
                        "confidence": round(conf, 4),
                        "trend": "stable",
                    }
                    if hasattr(pattern, 'trend'):
                        state_values["trend"] = str(pattern.trend) if pattern.trend else "stable"
                    domain = PredictionDomain.ECONOMICS
                    if ptype in ("daily", "weekly", "monthly", "seasonal"):
                        domain = PredictionDomain.ECONOMICS
                    self.set_domain_state(domain.value, state_values)
                except (ValueError, TypeError, AttributeError, KeyError):
                    continue
            with self._stats_lock:
                self._stats["temporal_ingestions"] += len(patterns)
        except (ValueError, TypeError, AttributeError, KeyError, OSError):
            pass

    def ingest_anomalies(self, anomaly_detector) -> None:
        """
        Ingest anomalies from AnomalyDetector into Oracle domain states.
        Translates detected anomalies into prediction-relevant state snapshots.
        """
        try:
            anomalies = None
            if hasattr(anomaly_detector, 'get_recent_anomalies'):
                anomalies = anomaly_detector.get_recent_anomalies(limit=10)
            if not anomalies:
                return
            for anomaly in anomalies:
                try:
                    a_dict = anomaly.to_dict() if hasattr(anomaly, 'to_dict') else anomaly
                    severity = a_dict.get("severity", "medium")
                    metric = a_dict.get("metric", "unknown")
                    z_score = a_dict.get("z_score", 0.0)
                    state_values = {
                        "anomaly_detected": True,
                        "severity": str(severity),
                        "z_score": round(float(z_score), 4),
                        "metric": str(metric),
                    }
                    domain = PredictionDomain.ECONOMICS
                    metric_str = str(metric).lower()
                    if re.search(r'\b(?:cpu|memory|disk|load)\b', metric_str):
                        domain = PredictionDomain.ECONOMICS
                    self.set_domain_state(domain.value, state_values)
                except (ValueError, TypeError, AttributeError, KeyError):
                    continue
            with self._stats_lock:
                self._stats["anomaly_ingestions"] += len(anomalies)
        except (ValueError, TypeError, AttributeError, KeyError, OSError):
            pass

    def get_daily_oracle(self) -> Dict:
        """
        Retrieve Oracle's daily prediction synthesis.
        Tries direct engine access first, then HTTP fallback, then heuristic.
        """
        engine = self._get_engine()
        if engine:
            try:
                if hasattr(engine, 'generate_daily_oracle'):
                    result = engine.generate_daily_oracle()
                    if isinstance(result, dict):
                        with self._stats_lock:
                            self._stats["daily_oracle_calls"] = self._stats.get("daily_oracle_calls", 0) + 1
                        return {k: _sanitize_content(str(v), 500) if isinstance(v, str) else v
                                for k, v in list(result.items())[:50]}
            except (TypeError, ValueError, AttributeError, KeyError, RuntimeError, OSError) as e:
                logger.warning("Oracle daily oracle failed: %s", type(e).__name__)

        http_result = self._http_get("/daily-oracle")
        if http_result and isinstance(http_result, dict):
            with self._stats_lock:
                self._stats["daily_oracle_calls"] = self._stats.get("daily_oracle_calls", 0) + 1
            return http_result

        result = self._heuristic_daily_oracle()
        with self._stats_lock:
            self._stats["daily_oracle_calls"] += 1
        return result

    def _heuristic_daily_oracle(self) -> Dict:
        """Generate a heuristic daily oracle from cached state snapshots."""
        now = datetime.now(timezone.utc).isoformat()
        with self._lock:
            snapshots = dict(self._state_snapshots)
        domains = list(snapshots.keys())[:10]
        overall_trend = "neutral"
        if domains:
            avg_values = []
            for d in domains:
                snap = snapshots[d]
                vals = [v for v in snap.state_values.values() if isinstance(v, (int, float))]
                if vals:
                    avg_values.append(sum(vals) / len(vals))
            if avg_values:
                avg = sum(avg_values) / len(avg_values)
                if avg > 60:
                    overall_trend = "positive"
                elif avg < 40:
                    overall_trend = "negative"
        return {
            "timestamp": now,
            "overall_trend": overall_trend,
            "domains_active": len(domains),
            "confidence": "medium",
            "source": "heuristic",
        }

    def assess_compound_hazard(self, domains: Optional[List[str]] = None) -> Dict:
        """
        Assess compound hazard risk across domains.
        Uses cross-domain interaction weights to estimate cascading risk.
        """
        target_domains = set()
        if domains:
            for d in domains:
                try:
                    target_domains.add(PredictionDomain(d))
                except ValueError:
                    pass
        if not target_domains:
            with self._lock:
                target_domains = {s.domain for s in self._state_snapshots.values()}
        if not target_domains:
            target_domains = {PredictionDomain.ECONOMICS}

        domain_risks = {}
        with self._lock:
            for dom in target_domains:
                snap = self._state_snapshots.get(dom.value)
                if snap:
                    vals = [v for v in snap.state_values.values() if isinstance(v, (int, float))]
                    if vals:
                        avg = sum(vals) / len(vals)
                        domain_risks[dom.value] = round(avg / 100.0, 4) if avg > 1 else round(avg, 4)
                    else:
                        domain_risks[dom.value] = 0.5
                else:
                    domain_risks[dom.value] = 0.3

        _CROSS_HAZARD_WEIGHTS = {
            ("economics", "geopolitics"): 0.7,
            ("economics", "climate"): 0.4,
            ("climate", "food"): 0.8,
            ("climate", "water"): 0.8,
            ("health", "pandemic"): 0.9,
            ("geopolitics", "infrastructure"): 0.5,
            ("cyber", "infrastructure"): 0.6,
            ("seismic", "infrastructure"): 0.5,
            ("supply_chain", "economics"): 0.7,
            ("energy", "economics"): 0.6,
            ("energy", "climate"): 0.5,
            ("mental_health", "health"): 0.7,
            ("housing", "economics"): 0.5,
            ("pandemic", "health"): 0.9,
        }

        cascade_risks = []
        for (d1, d2), weight in _CROSS_HAZARD_WEIGHTS.items():
            r1 = domain_risks.get(d1, 0.3)
            r2 = domain_risks.get(d2, 0.3)
            if r1 > 0.4 or r2 > 0.4:
                cascade_risks.append({
                    "domains": [d1, d2],
                    "weight": weight,
                    "combined_risk": round(r1 * r2 * weight, 4),
                })

        overall_risk = 0.3
        if domain_risks:
            risk_values = list(domain_risks.values())
            overall_risk = round(sum(risk_values) / len(risk_values) + sum(r["combined_risk"] for r in cascade_risks) * 0.1, 4)
            overall_risk = min(overall_risk, 1.0)

        with self._stats_lock:
            self._stats["compound_hazard_assessments"] = self._stats.get("compound_hazard_assessments", 0) + 1

        return {
            "overall_risk": overall_risk,
            "domain_risks": domain_risks,
            "cascade_risks": cascade_risks[:20],
            "domains_assessed": len(domain_risks),
            "source": "bridge",
        }

    def get_regime_shift_probability(self, domain: str = "economics") -> Dict:
        """
        Estimate regime shift probability for a given domain.
        Uses z-score anomaly detection on cached state values.
        """
        try:
            dom_enum = PredictionDomain(domain)
        except ValueError:
            dom_enum = PredictionDomain.ECONOMICS

        with self._stats_lock:
            self._stats["regime_shift_assessments"] += 1

        with self._lock:
            snap = self._state_snapshots.get(dom_enum.value)

        if not snap:
            return {"domain": dom_enum.value, "probability": 0.0, "source": "no_data"}

        vals = [v for v in snap.state_values.values() if isinstance(v, (int, float))]
        if len(vals) < 2:
            return {"domain": dom_enum.value, "probability": 0.0, "source": "insufficient_data"}

        mean_val = sum(vals) / len(vals)
        variance = sum((v - mean_val) ** 2 for v in vals) / len(vals)
        std_dev = variance ** 0.5 if variance > 0 else 0.001
        z_scores = [abs(v - mean_val) / std_dev for v in vals]
        extreme_count = sum(1 for z in z_scores if z > 2.0)

        probability = round(min(extreme_count / max(len(vals), 1), 1.0), 4)

        return {
            "domain": dom_enum.value,
            "probability": probability,
            "mean": round(mean_val, 4),
            "std_dev": round(std_dev, 4),
            "extreme_values": extreme_count,
            "total_values": len(vals),
            "source": "bridge",
        }

    def get_prophecy(self, horizon_days: int = 90) -> Dict:
        """
        Generate a multi-horizon prophecy synthesis.
        Combines predictions across time horizons into a narrative.
        """
        valid_horizons = {1, 7, 30, 90, 365}
        if horizon_days not in valid_horizons:
            horizon_days = min(valid_horizons, key=lambda h: abs(h - horizon_days))

        target_date = (date.today() + timedelta(days=horizon_days)).isoformat()
        forecast = self.predict(target_date=target_date, confidence="high")

        domains_predicted = [p.domain.value for p in forecast.predictions]
        avg_confidence = 0.5
        if forecast.predictions:
            widths = [p.confidence_interval[1] - p.confidence_interval[0] for p in forecast.predictions if p.confidence_interval]
            if widths:
                avg_width = sum(widths) / len(widths)
                avg_confidence = round(max(0.0, 1.0 - avg_width / 100.0), 4)

        key_signals = []
        for p in forecast.predictions[:5]:
            key_signals.extend(p.signals_used[:3])
        key_signals = list(dict.fromkeys(key_signals))[:10]

        narrative_tone = "neutral"
        if avg_confidence > 0.7:
            narrative_tone = "optimistic"
        elif avg_confidence < 0.3:
            narrative_tone = "cautious"

        with self._stats_lock:
            self._stats["prophecies_generated"] = self._stats.get("prophecies_generated", 0) + 1

        return {
            "horizon_days": horizon_days,
            "target_date": target_date,
            "domains_predicted": domains_predicted,
            "confidence": avg_confidence,
            "narrative_tone": narrative_tone,
            "key_signals": key_signals,
            "prediction_count": len(forecast.predictions),
            "source": "bridge",
        }