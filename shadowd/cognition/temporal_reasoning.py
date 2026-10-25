#!/usr/bin/env python3
"""
TemporalReasoning - Time-Aware Inference and Temporal Pattern Detection

Analyzes temporal patterns in observations:
- Detects recurring time patterns (daily/weekly/monthly cycles)
- Predicts future events based on past patterns
- Identifies temporal anomalies (events happening at unusual times)
- Tracks time-based trends and seasonality
- Estimates time-to-completion for ongoing activities
"""

import logging
import math
import re
import copy
import threading
import uuid
from collections import Counter, defaultdict, deque
from dataclasses import dataclass, field, asdict
from datetime import datetime, timezone, timedelta
from enum import Enum
from statistics import mean, median, stdev
from typing import Dict, List, Optional

logger = logging.getLogger("shadowd.cognition.temporal_reasoning")

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _SANITIZE_RE
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_STRICT_RE.sub('', text).strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 2000) -> str:
    if not isinstance(text, str):
        return ""
    return _SANITIZE_CONTENT_RE.sub('', text).strip()[:max_length]


class PatternType(str, Enum):
    DAILY = "daily"
    WEEKLY = "weekly"
    MONTHLY = "monthly"
    SEASONAL = "seasonal"
    ONCE = "once"


class TrendDirection(str, Enum):
    INCREASING = "increasing"
    DECREASING = "decreasing"
    STABLE = "stable"
    UNKNOWN = "unknown"


@dataclass
class TemporalPattern:
    pattern_id: str
    pattern_type: PatternType
    event_type: str
    frequency: float
    typical_time: Optional[str]
    typical_day: Optional[str]
    confidence: float
    next_predicted: Optional[str]
    sample_size: int

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pattern_type"] = self.pattern_type.value
        d["frequency"] = round(self.frequency, 3)
        d["confidence"] = round(self.confidence, 3)
        return d


@dataclass
class Prediction:
    event_type: str
    predicted_time: str
    confidence: float
    reasoning: str
    pattern_type: PatternType

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["pattern_type"] = self.pattern_type.value
        d["confidence"] = round(self.confidence, 3)
        return d


@dataclass
class TemporalAnomaly:
    event_type: str
    expected_time: Optional[str]
    actual_time: str
    deviation_score: float
    description: str

    def to_dict(self) -> Dict:
        d = copy.deepcopy(asdict(self))
        d["deviation_score"] = round(self.deviation_score, 3)
        return d


@dataclass
class TrendAnalysis:
    event_type: str
    direction: TrendDirection
    rate_of_change: float
    confidence: float
    window_days: int
    frequency_per_day: float

    def to_dict(self) -> Dict:
        d = asdict(self)
        d["direction"] = self.direction.value
        d["rate_of_change"] = round(self.rate_of_change, 3)
        d["confidence"] = round(self.confidence, 3)
        d["frequency_per_day"] = round(self.frequency_per_day, 3)
        return d


@dataclass
class DurationEstimate:
    event_type: str
    mean_minutes: float
    median_minutes: float
    min_minutes: float
    max_minutes: float
    confidence: float
    sample_size: int

    def to_dict(self) -> Dict:
        d = copy.deepcopy(asdict(self))
        d["mean_minutes"] = round(self.mean_minutes, 3)
        d["median_minutes"] = round(self.median_minutes, 3)
        d["min_minutes"] = round(self.min_minutes, 3)
        d["max_minutes"] = round(self.max_minutes, 3)
        d["confidence"] = round(self.confidence, 3)
        return d


class TemporalReasoning:
    """
    Time-Aware Inference and Temporal Pattern Detection

    Analyzes temporal patterns in observations:
    - Detects recurring time patterns (daily/weekly/monthly cycles)
    - Predicts future events based on past patterns
    - Identifies temporal anomalies (events happening at unusual times)
    - Tracks time-based trends and seasonality
    - Estimates time-to-completion for ongoing activities
    """

    _MAX_EVENTS = 10000
    _MAX_PATTERNS = 500
    _MAX_PREDICTIONS = 200

    _DAY_NAMES = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
    _SEASON_MONTHS = {
        "winter": [12, 1, 2],
        "spring": [3, 4, 5],
        "summer": [6, 7, 8],
        "fall": [9, 10, 11],
    }

    def __init__(self):
        self._events: deque = deque(maxlen=self._MAX_EVENTS)
        self._patterns: List[TemporalPattern] = []
        self._predictions: List[Prediction] = []
        self._lock = threading.Lock()
        self._stats = {
            "events_recorded": 0,
            "patterns_detected": 0,
            "predictions_made": 0,
            "anomalies_found": 0,
            "prediction_hits": 0,
            "prediction_total": 0,
        }
        self._stats_lock = threading.Lock()

    def record_event(self, event_type: str, timestamp: str = None, metadata: Dict = None) -> str:
        event_type = _sanitize(event_type, 200)
        if not event_type:
            raise ValueError("event_type must not be empty")
        if metadata:
            metadata = {_sanitize(str(k), 100): _sanitize_content(str(v), 500) for k, v in list(metadata.items())[:100]}
        event_id = f"tevt_{uuid.uuid4().hex[:12]}"
        if timestamp is None:
            ts = datetime.now(timezone.utc)
        else:
            try:
                ts = datetime.fromisoformat(timestamp)
            except (ValueError, TypeError):
                ts = datetime.now(timezone.utc)
            if ts.tzinfo is None:
                ts = ts.replace(tzinfo=timezone.utc)

        event = {
            "event_id": event_id,
            "event_type": event_type,
            "timestamp": ts,
            "iso_timestamp": ts.isoformat(),
            "hour": ts.hour,
            "day_of_week": ts.weekday(),
            "day_name": self._DAY_NAMES[ts.weekday()],
            "month": ts.month,
            "year": ts.year,
            "metadata": metadata or {},
        }

        with self._lock:
            self._events.append(event)
            with self._stats_lock:
                self._stats["events_recorded"] += 1

        self._update_prediction_accuracy(event)
        return event_id

    def detect_patterns(self, event_type: str = None) -> List[TemporalPattern]:
        if event_type:
            event_type = _sanitize(event_type, 200)
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]

        if len(events) < 3:
            return []

        type_groups = defaultdict(list)
        for e in events:
            type_groups[e["event_type"]].append(e)

        patterns = []
        for etype, etype_events in type_groups.items():
            if len(etype_events) < 3:
                continue

            patterns.extend(self._detect_daily_pattern(etype, etype_events))
            patterns.extend(self._detect_weekly_pattern(etype, etype_events))
            patterns.extend(self._detect_monthly_pattern(etype, etype_events))
            patterns.extend(self._detect_seasonal_pattern(etype, etype_events))

        patterns = [p for p in patterns if p.confidence >= 0.5]

        with self._lock:
            self._patterns = patterns[:self._MAX_PATTERNS]
            self._stats["patterns_detected"] = len(self._patterns)

        return patterns[:self._MAX_PATTERNS]

    def predict_next(self, event_type: str, within_hours: float = 48) -> List[Prediction]:
        event_type = _sanitize(event_type, 200)
        within_hours = max(0.1, min(8760, within_hours))
        patterns = self.detect_patterns(event_type)
        with self._lock:
            events = list(self._events)

        etype_events = [e for e in events if e["event_type"] == event_type]
        if not etype_events and not patterns:
            return []

        predictions = []
        now = datetime.now(timezone.utc)
        cutoff = now + timedelta(hours=within_hours)

        relevant = [p for p in patterns if p.event_type == event_type]
        for pattern in relevant:
            if pattern.next_predicted:
                try:
                    pred_time = datetime.fromisoformat(pattern.next_predicted)
                    if pred_time.tzinfo is None:
                        pred_time = pred_time.replace(tzinfo=timezone.utc)
                except (ValueError, TypeError):
                    continue
                if pred_time <= cutoff:
                    reasoning = (
                        f"Based on {pattern.sample_size} observations, "
                        f"this event follows a {pattern.pattern_type.value} pattern "
                        f"{'at ' + pattern.typical_time if pattern.typical_time else ''} "
                        f"{'on ' + pattern.typical_day if pattern.typical_day else ''}".strip()
                    )
                    pred = Prediction(
                        event_type=event_type,
                        predicted_time=pattern.next_predicted,
                        confidence=pattern.confidence * 0.9,
                        reasoning=reasoning,
                        pattern_type=pattern.pattern_type,
                    )
                    predictions.append(pred)

        if not predictions and etype_events:
            last_event = max(etype_events, key=lambda e: e["timestamp"])
            intervals = []
            sorted_events = sorted(etype_events, key=lambda e: e["timestamp"])
            for i in range(1, len(sorted_events)):
                delta = (sorted_events[i]["timestamp"] - sorted_events[i - 1]["timestamp"]).total_seconds() / 3600
                intervals.append(delta)

            if intervals:
                avg_interval = mean(intervals)
                next_time = last_event["timestamp"] + timedelta(hours=avg_interval)
                if next_time <= cutoff:
                    pred = Prediction(
                        event_type=event_type,
                        predicted_time=next_time.isoformat(),
                        confidence=min(0.5, len(intervals) / 20),
                        reasoning=f"Based on average interval of {avg_interval:.1f}h over {len(intervals)} gaps",
                        pattern_type=PatternType.ONCE,
                    )
                    predictions.append(pred)

        with self._lock:
            self._predictions = predictions[:self._MAX_PREDICTIONS]
            self._stats["predictions_made"] = len(self._predictions)

        return predictions

    def detect_anomalies(self, event_type: str = None) -> List[TemporalAnomaly]:
        patterns = self.detect_patterns(event_type)
        with self._lock:
            events = list(self._events)

        if event_type:
            events = [e for e in events if e["event_type"] == event_type]

        type_groups = defaultdict(list)
        for e in events:
            type_groups[e["event_type"]].append(e)

        anomalies = []
        for etype, etype_events in type_groups.items():
            if len(etype_events) < 5:
                continue

            etype_patterns = [p for p in patterns if p.event_type == etype]

            daily_patterns = [p for p in etype_patterns if p.pattern_type == PatternType.DAILY]
            if daily_patterns:
                hours = [e["hour"] for e in etype_events]
                if len(hours) >= 2:
                    mean_h = mean(hours)
                    std_h = stdev(hours) if len(hours) > 1 else 0
                    if std_h > 0:
                        for e in etype_events:
                            dev = min(abs(e["hour"] - mean_h), 24 - abs(e["hour"] - mean_h))
                            z = dev / std_h
                            if z > 2.0:
                                expected = f"{int(mean_h):02d}:00"
                                actual = f"{e['hour']:02d}:00"
                                anomalies.append(TemporalAnomaly(
                                    event_type=etype,
                                    expected_time=expected,
                                    actual_time=actual,
                                    deviation_score=min(round(z, 2), 10.0),
                                    description=(
                                        f"{etype} occurred at {actual} "
                                        f"(expected ~{expected}, z={z:.2f})"
                                    ),
                                ))

            weekly_patterns = [p for p in etype_patterns if p.pattern_type == PatternType.WEEKLY]
            if weekly_patterns:
                day_counts = Counter(e["day_of_week"] for e in etype_events)
                most_common_day = day_counts.most_common(1)[0][0]
                unusual_events = [e for e in etype_events if e["day_of_week"] != most_common_day]
                for e in unusual_events:
                    anomalies.append(TemporalAnomaly(
                        event_type=etype,
                        expected_time=f"{self._DAY_NAMES[most_common_day]} usual",
                        actual_time=f"{e['day_name']} {e['hour']:02d}:00",
                        deviation_score=0.7,
                        description=(
                            f"{etype} occurred on {e['day_name']} "
                            f"instead of usual {self._DAY_NAMES[most_common_day]}"
                        ),
                    ))

        anomalies.sort(key=lambda a: a.deviation_score, reverse=True)
        anomalies = anomalies[:200]

        with self._lock:
            with self._stats_lock:
                self._stats["anomalies_found"] += len(anomalies)

        return anomalies

    def analyze_trend(self, event_type: str, window_days: int = 30) -> TrendAnalysis:
        event_type = _sanitize(event_type, 200)
        window_days = max(1, min(3650, window_days))
        with self._lock:
            events = list(self._events)

        events = [e for e in events if e["event_type"] == event_type]
        if not events:
            return TrendAnalysis(
                event_type=event_type,
                direction=TrendDirection.UNKNOWN,
                rate_of_change=0.0,
                confidence=0.0,
                window_days=window_days,
                frequency_per_day=0.0,
            )

        now = datetime.now(timezone.utc)
        window_start = now - timedelta(days=window_days)
        windowed = [e for e in events if e["timestamp"] >= window_start]

        if not windowed:
            return TrendAnalysis(
                event_type=event_type,
                direction=TrendDirection.UNKNOWN,
                rate_of_change=0.0,
                confidence=0.0,
                window_days=window_days,
                frequency_per_day=0.0,
            )

        freq_per_day = len(windowed) / window_days

        half = window_days / 2
        mid_point = now - timedelta(days=half)
        first_half = [e for e in windowed if e["timestamp"] < mid_point]
        second_half = [e for e in windowed if e["timestamp"] >= mid_point]

        first_freq = len(first_half) / max(half, 1)
        second_freq = len(second_half) / max(half, 1)

        rate = second_freq - first_freq

        if abs(rate) < 0.05:
            direction = TrendDirection.STABLE
        elif rate > 0:
            direction = TrendDirection.INCREASING
        else:
            direction = TrendDirection.DECREASING

        confidence = min(1.0, len(windowed) / 20)

        return TrendAnalysis(
            event_type=event_type,
            direction=direction,
            rate_of_change=round(rate, 4),
            confidence=round(confidence, 4),
            window_days=window_days,
            frequency_per_day=round(freq_per_day, 4),
        )

    def estimate_duration(self, event_type: str) -> DurationEstimate:
        event_type = _sanitize(event_type, 200)
        with self._lock:
            events = list(self._events)

        events = [e for e in events if e["event_type"] == event_type]

        durations = []
        sorted_events = sorted(events, key=lambda e: e["timestamp"])
        for i in range(len(sorted_events) - 1):
            delta = (sorted_events[i + 1]["timestamp"] - sorted_events[i]["timestamp"]).total_seconds() / 60
            if 1 <= delta <= 1440:
                durations.append(delta)

        if not durations:
            return DurationEstimate(
                event_type=event_type,
                mean_minutes=0.0,
                median_minutes=0.0,
                min_minutes=0.0,
                max_minutes=0.0,
                confidence=0.0,
                sample_size=0,
            )

        confidence = min(1.0, len(durations) / 10)

        return DurationEstimate(
            event_type=event_type,
            mean_minutes=round(mean(durations), 2),
            median_minutes=round(median(durations), 2),
            min_minutes=round(min(durations), 2),
            max_minutes=round(max(durations), 2),
            confidence=round(confidence, 4),
            sample_size=len(durations),
        )

    def get_stats(self) -> Dict:
        with self._lock:
            return dict(self._stats)

    def _detect_daily_pattern(self, event_type: str, events: List[Dict]) -> List[TemporalPattern]:
        if len(events) < 5:
            return []

        hours = [e["hour"] for e in events]
        hour_counts = Counter(hours)
        most_common_hour, count = hour_counts.most_common(1)[0]
        frequency = len(events) / max(1, self._day_span(events))

        consistency = count / len(hours) if hours else 0
        confidence = min(1.0, len(events) / 10) * consistency

        if confidence < 0.5:
            return []

        typical_time = f"{most_common_hour:02d}:00"
        next_pred = self._predict_next_occurrence(events, PatternType.DAILY)

        return [TemporalPattern(
            pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.DAILY,
            event_type=event_type,
            frequency=round(frequency, 4),
            typical_time=typical_time,
            typical_day=None,
            confidence=round(confidence, 4),
            next_predicted=next_pred,
            sample_size=len(events),
        )]

    def _detect_weekly_pattern(self, event_type: str, events: List[Dict]) -> List[TemporalPattern]:
        if len(events) < 5:
            return []

        days = [e["day_of_week"] for e in events]
        day_counts = Counter(days)
        most_common_day, count = day_counts.most_common(1)[0]

        day_span = max(1, self._day_span(events))
        weeks = day_span / 7
        frequency = len(events) / max(1, weeks)

        consistency = count / len(days) if days else 0
        confidence = min(1.0, len(events) / 10) * consistency

        if confidence < 0.5:
            return []

        typical_day = self._DAY_NAMES[most_common_day]
        hour_for_day = [e["hour"] for e in events if e["day_of_week"] == most_common_day]
        typical_time = f"{Counter(hour_for_day).most_common(1)[0][0]:02d}:00" if hour_for_day else None

        next_pred = self._predict_next_occurrence(events, PatternType.WEEKLY)

        return [TemporalPattern(
            pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.WEEKLY,
            event_type=event_type,
            frequency=round(frequency, 4),
            typical_time=typical_time,
            typical_day=typical_day,
            confidence=round(confidence, 4),
            next_predicted=next_pred,
            sample_size=len(events),
        )]

    def _detect_monthly_pattern(self, event_type: str, events: List[Dict]) -> List[TemporalPattern]:
        if len(events) < 6:
            return []

        day_of_months = []
        for e in events:
            dom = e["timestamp"].day
            day_of_months.append(dom)

        dom_counts = Counter(day_of_months)
        most_common_dom, count = dom_counts.most_common(1)[0]

        consistency = count / len(day_of_months) if day_of_months else 0
        confidence = min(1.0, len(events) / 10) * consistency

        if confidence < 0.5:
            return []

        month_span = max(1, self._month_span(events))
        frequency = len(events) / max(1, month_span)

        next_pred = self._predict_next_occurrence(events, PatternType.MONTHLY)

        return [TemporalPattern(
            pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.MONTHLY,
            event_type=event_type,
            frequency=round(frequency, 4),
            typical_time=None,
            typical_day=f"day_{most_common_dom}",
            confidence=round(confidence, 4),
            next_predicted=next_pred,
            sample_size=len(events),
        )]

    def _detect_seasonal_pattern(self, event_type: str, events: List[Dict]) -> List[TemporalPattern]:
        if len(events) < 6:
            return []

        season_counts = Counter()
        for e in events:
            month = e["month"]
            for season, months in self._SEASON_MONTHS.items():
                if month in months:
                    season_counts[season] += 1
                    break

        if not season_counts:
            return []

        top_season, count = season_counts.most_common(1)[0]
        consistency = count / len(events)
        confidence = min(1.0, len(events) / 10) * consistency

        if confidence < 0.5:
            return []

        season_months = self._SEASON_MONTHS[top_season]
        hour_for_season = [e["hour"] for e in events if e["month"] in season_months]
        typical_time = f"{Counter(hour_for_season).most_common(1)[0][0]:02d}:00" if hour_for_season else None

        next_pred = self._predict_next_occurrence(events, PatternType.SEASONAL)

        return [TemporalPattern(
            pattern_id=f"pat_{uuid.uuid4().hex[:12]}",
            pattern_type=PatternType.SEASONAL,
            event_type=event_type,
            frequency=round(count / max(1, self._year_span(events)), 4),
            typical_time=typical_time,
            typical_day=top_season,
            confidence=round(confidence, 4),
            next_predicted=next_pred,
            sample_size=len(events),
        )]

    def _predict_next_occurrence(self, events: List[Dict], pattern_type: PatternType) -> Optional[str]:
        if not events:
            return None

        now = datetime.now(timezone.utc)
        last_event = max(events, key=lambda e: e["timestamp"])
        last_ts = last_event["timestamp"]

        if pattern_type == PatternType.DAILY:
            intervals = []
            sorted_events = sorted(events, key=lambda e: e["timestamp"])
            for i in range(1, len(sorted_events)):
                delta = (sorted_events[i]["timestamp"] - sorted_events[i - 1]["timestamp"]).total_seconds() / 3600
                if 1 <= delta <= 48:
                    intervals.append(delta)
            if intervals:
                next_time = last_ts + timedelta(hours=mean(intervals))
            else:
                next_time = last_ts + timedelta(hours=24)
        elif pattern_type == PatternType.WEEKLY:
            next_time = last_ts + timedelta(weeks=1)
        elif pattern_type == PatternType.MONTHLY:
            next_time = last_ts + timedelta(days=30)
        elif pattern_type == PatternType.SEASONAL:
            next_time = last_ts + timedelta(days=90)
        else:
            return None

        if next_time < now:
            iters = 0
            max_iters = 10000
            if pattern_type == PatternType.DAILY:
                max_iters = max(100, int(((now - last_ts).total_seconds() / 3600) / (mean(intervals) if intervals else 24)) + 100)
                while next_time < now:
                    next_time += timedelta(hours=mean(intervals) if intervals else 24)
                    iters += 1
                    if iters > max_iters:
                        break
            elif pattern_type == PatternType.WEEKLY:
                max_iters = max(100, int(((now - last_ts).total_seconds() / 3600) / 168) + 100)
                while next_time < now:
                    next_time += timedelta(weeks=1)
                    iters += 1
                    if iters > max_iters:
                        break
            elif pattern_type == PatternType.MONTHLY:
                max_iters = max(100, int(((now - last_ts).total_seconds() / 3600) / 720) + 100)
                while next_time < now:
                    next_time += timedelta(days=30)
                    iters += 1
                    if iters > max_iters:
                        break
            elif pattern_type == PatternType.SEASONAL:
                max_iters = max(100, int(((now - last_ts).total_seconds() / 3600) / 2160) + 100)
                while next_time < now:
                    next_time += timedelta(days=90)
                    iters += 1
                    if iters > max_iters:
                        break

        return next_time.isoformat()

    def _day_span(self, events: List[Dict]) -> float:
        if len(events) < 2:
            return 1
        timestamps = [e["timestamp"] for e in events]
        delta = (max(timestamps) - min(timestamps)).total_seconds() / 86400
        return max(1, delta)

    def _month_span(self, events: List[Dict]) -> float:
        if len(events) < 2:
            return 1
        timestamps = [e["timestamp"] for e in events]
        delta = (max(timestamps) - min(timestamps)).total_seconds() / (86400 * 30)
        return max(1, delta)

    def _year_span(self, events: List[Dict]) -> float:
        if len(events) < 2:
            return 1
        timestamps = [e["timestamp"] for e in events]
        delta = (max(timestamps) - min(timestamps)).total_seconds() / (86400 * 365)
        return max(1, delta)

    def _update_prediction_accuracy(self, event: Dict) -> None:
        with self._lock:
            predictions = list(self._predictions)

        for pred in predictions:
            if pred.event_type != event["event_type"]:
                continue
            try:
                pred_time = datetime.fromisoformat(pred.predicted_time)
                if pred_time.tzinfo is None:
                    pred_time = pred_time.replace(tzinfo=timezone.utc)
            except (ValueError, TypeError):
                continue

            actual_time = event["timestamp"]
            diff_hours = abs((actual_time - pred_time).total_seconds()) / 3600

            with self._lock:
                with self._stats_lock:
                    self._stats["prediction_total"] += 1
                tolerance = 2.0 if pred.pattern_type == PatternType.DAILY else 12.0
                if diff_hours <= tolerance:
                    with self._stats_lock:
                        self._stats["prediction_hits"] += 1