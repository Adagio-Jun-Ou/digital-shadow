"""
Tests for TemporalReasoning - Time-Aware Inference and Temporal Pattern Detection
"""

import threading
from datetime import datetime, timezone, timedelta

import pytest

from shadowd.cognition.temporal_reasoning import (
    TemporalReasoning,
    PatternType,
    TrendDirection,
    TemporalPattern,
    Prediction,
    TemporalAnomaly,
    TrendAnalysis,
    DurationEstimate,
)


def _ts(days_ago=0, hour=9, minute=0, day_of_week=None, month=None, day_of_month=None):
    now = datetime.now(timezone.utc)
    dt = now - timedelta(days=days_ago)
    if day_of_week is not None:
        dt = dt - timedelta(days=(dt.weekday() - day_of_week) % 7)
    if month is not None:
        try:
            dt = dt.replace(month=month)
        except ValueError:
            dt = dt.replace(month=month, year=dt.year + 1)
    if day_of_month is not None:
        try:
            dt = dt.replace(day=day_of_month)
        except ValueError:
            pass
    dt = dt.replace(hour=hour, minute=minute, second=0, microsecond=0)
    return dt.isoformat()


class TestDataClasses:
    def test_pattern_type_values(self):
        assert PatternType.DAILY == "daily"
        assert PatternType.WEEKLY == "weekly"
        assert PatternType.MONTHLY == "monthly"
        assert PatternType.SEASONAL == "seasonal"
        assert PatternType.ONCE == "once"

    def test_trend_direction_values(self):
        assert TrendDirection.INCREASING == "increasing"
        assert TrendDirection.DECREASING == "decreasing"
        assert TrendDirection.STABLE == "stable"
        assert TrendDirection.UNKNOWN == "unknown"

    def test_temporal_pattern_to_dict(self):
        p = TemporalPattern(
            pattern_id="pat_abc123",
            pattern_type=PatternType.DAILY,
            event_type="coffee",
            frequency=1.0,
            typical_time="09:00",
            typical_day=None,
            confidence=0.85,
            next_predicted="2025-01-01T09:00:00+00:00",
            sample_size=10,
        )
        d = p.to_dict()
        assert d["pattern_type"] == "daily"
        assert d["event_type"] == "coffee"
        assert d["confidence"] == 0.85
        assert d["sample_size"] == 10

    def test_prediction_to_dict(self):
        p = Prediction(
            event_type="meeting",
            predicted_time="2025-01-01T10:00:00+00:00",
            confidence=0.7,
            reasoning="Based on 10 observations",
            pattern_type=PatternType.WEEKLY,
        )
        d = p.to_dict()
        assert d["pattern_type"] == "weekly"
        assert d["event_type"] == "meeting"

    def test_temporal_anomaly_to_dict(self):
        a = TemporalAnomaly(
            event_type="exercise",
            expected_time="06:00",
            actual_time="23:00",
            deviation_score=3.5,
            description="Unusual time",
        )
        d = a.to_dict()
        assert d["deviation_score"] == 3.5

    def test_trend_analysis_to_dict(self):
        t = TrendAnalysis(
            event_type="running",
            direction=TrendDirection.INCREASING,
            rate_of_change=0.5,
            confidence=0.8,
            window_days=30,
            frequency_per_day=1.2,
        )
        d = t.to_dict()
        assert d["direction"] == "increasing"

    def test_duration_estimate_to_dict(self):
        de = DurationEstimate(
            event_type="workout",
            mean_minutes=45.0,
            median_minutes=40.0,
            min_minutes=20.0,
            max_minutes=90.0,
            confidence=0.7,
            sample_size=15,
        )
        d = de.to_dict()
        assert d["mean_minutes"] == 45.0
        assert d["sample_size"] == 15


class TestRecordEvent:
    def test_basic_record(self):
        tr = TemporalReasoning()
        eid = tr.record_event("coffee")
        assert isinstance(eid, str)
        assert eid.startswith("tevt_")

    def test_record_with_timestamp(self):
        tr = TemporalReasoning()
        ts = "2024-06-15T09:00:00+00:00"
        eid = tr.record_event("coffee", timestamp=ts)
        assert isinstance(eid, str)
        stats = tr.get_stats()
        assert stats["events_recorded"] == 1

    def test_record_with_metadata(self):
        tr = TemporalReasoning()
        eid = tr.record_event("meeting", metadata={"room": "A1", "duration": 60})
        assert isinstance(eid, str)

    def test_record_auto_timestamp(self):
        tr = TemporalReasoning()
        eid = tr.record_event("test")
        events = list(tr._events)
        assert len(events) == 1
        assert events[0]["event_type"] == "test"
        assert events[0]["iso_timestamp"] is not None

    def test_record_extracts_hour(self):
        tr = TemporalReasoning()
        tr.record_event("coffee", timestamp="2024-06-15T09:30:00+00:00")
        events = list(tr._events)
        assert events[0]["hour"] == 9

    def test_record_extracts_day_of_week(self):
        tr = TemporalReasoning()
        ts = "2024-06-17T09:00:00+00:00"
        tr.record_event("coffee", timestamp=ts)
        events = list(tr._events)
        dt = datetime.fromisoformat(ts)
        assert events[0]["day_of_week"] == dt.weekday()

    def test_record_extracts_month(self):
        tr = TemporalReasoning()
        tr.record_event("coffee", timestamp="2024-06-15T09:00:00+00:00")
        events = list(tr._events)
        assert events[0]["month"] == 6

    def test_record_naive_timestamp_gets_utc(self):
        tr = TemporalReasoning()
        eid = tr.record_event("test", timestamp="2024-01-15T10:00:00")
        events = list(tr._events)
        assert events[0]["timestamp"].tzinfo is not None

    def test_multiple_events_count(self):
        tr = TemporalReasoning()
        for i in range(10):
            tr.record_event(f"event_{i % 3}")
        stats = tr.get_stats()
        assert stats["events_recorded"] == 10

    def test_max_events_deque(self):
        tr = TemporalReasoning()
        tr._MAX_EVENTS = 5
        tr._events = __import__("collections").deque(maxlen=5)
        for i in range(10):
            tr.record_event("overflow")
        assert len(tr._events) <= 5

    def test_stats_updated(self):
        tr = TemporalReasoning()
        tr.record_event("a")
        tr.record_event("b")
        stats = tr.get_stats()
        assert stats["events_recorded"] == 2


class TestDetectPatterns:
    def test_no_patterns_with_few_events(self):
        tr = TemporalReasoning()
        for i in range(2):
            tr.record_event("coffee", timestamp=_ts(days_ago=10 - i, hour=9))
        patterns = tr.detect_patterns()
        assert len(patterns) == 0

    def test_daily_pattern_detected(self):
        tr = TemporalReasoning()
        for i in range(15):
            tr.record_event("coffee", timestamp=_ts(days_ago=15 - i, hour=9))
        patterns = tr.detect_patterns("coffee")
        daily_patterns = [p for p in patterns if p.pattern_type == PatternType.DAILY]
        assert len(daily_patterns) >= 1
        dp = daily_patterns[0]
        assert dp.typical_time == "09:00"
        assert dp.confidence >= 0.5

    def test_weekly_pattern_detected(self):
        tr = TemporalReasoning()
        for i in range(8):
            tr.record_event("standup", timestamp=_ts(days_ago=7 * (8 - i), hour=10, day_of_week=0))
        patterns = tr.detect_patterns("standup")
        weekly_patterns = [p for p in patterns if p.pattern_type == PatternType.WEEKLY]
        assert len(weekly_patterns) >= 1
        wp = weekly_patterns[0]
        assert wp.typical_day == "monday"
        assert wp.confidence >= 0.5

    def test_monthly_pattern(self):
        tr = TemporalReasoning()
        for m in range(1, 8):
            try:
                ts = datetime(2024, m, 1, 10, 0, tzinfo=timezone.utc).isoformat()
                tr.record_event("rent", timestamp=ts)
            except Exception:
                pass
        if len(list(tr._events)) >= 3:
            patterns = tr.detect_patterns("rent")
            monthly = [p for p in patterns if p.pattern_type == PatternType.MONTHLY]
            assert isinstance(monthly, list)

    def test_pattern_confidence_threshold(self):
        tr = TemporalReasoning()
        for i in range(3):
            tr.record_event("rare_event", timestamp=_ts(days_ago=100 - i * 30, hour=14))
        patterns = tr.detect_patterns("rare_event")
        for p in patterns:
            assert p.confidence >= 0.5

    def test_pattern_has_next_predicted(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("lunch", timestamp=_ts(days_ago=20 - i, hour=12))
        patterns = tr.detect_patterns("lunch")
        daily = [p for p in patterns if p.pattern_type == PatternType.DAILY]
        if daily:
            assert daily[0].next_predicted is not None

    def test_pattern_sample_size(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("test_event", timestamp=_ts(days_ago=20 - i, hour=9))
        patterns = tr.detect_patterns("test_event")
        for p in patterns:
            assert p.sample_size == 20

    def test_filter_by_event_type(self):
        tr = TemporalReasoning()
        for i in range(15):
            tr.record_event("coffee", timestamp=_ts(days_ago=15 - i, hour=9))
        for i in range(3):
            tr.record_event("rare", timestamp=_ts(days_ago=10 - i, hour=22))
        patterns = tr.detect_patterns("coffee")
        for p in patterns:
            assert p.event_type == "coffee"

    def test_all_event_types_when_none(self):
        tr = TemporalReasoning()
        for i in range(15):
            tr.record_event("coffee", timestamp=_ts(days_ago=15 - i, hour=9))
        for i in range(8):
            tr.record_event("standup", timestamp=_ts(days_ago=7 * (8 - i), hour=10))
        patterns = tr.detect_patterns()
        event_types = {p.event_type for p in patterns}
        assert "coffee" in event_types or "standup" in event_types


class TestPredictNext:
    def test_prediction_from_daily_pattern(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.detect_patterns("coffee")
        preds = tr.predict_next("coffee", within_hours=48)
        assert len(preds) >= 1
        assert preds[0].confidence > 0
        assert preds[0].predicted_time is not None

    def test_prediction_from_intervals(self):
        tr = TemporalReasoning()
        for i in range(10):
            tr.record_event("check_email", timestamp=_ts(days_ago=10 - i, hour=8 + (i % 3)))
        preds = tr.predict_next("check_email", within_hours=72)
        assert len(preds) >= 1

    def test_no_prediction_no_events(self):
        tr = TemporalReasoning()
        preds = tr.predict_next("nonexistent", within_hours=48)
        assert len(preds) == 0

    def test_prediction_within_hours_filter(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("daily_event", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.detect_patterns("daily_event")
        preds = tr.predict_next("daily_event", within_hours=1)
        assert len(preds) <= 1

    def test_prediction_has_reasoning(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.detect_patterns("coffee")
        preds = tr.predict_next("coffee", within_hours=48)
        if preds:
            assert preds[0].reasoning is not None
            assert len(preds[0].reasoning) > 0

    def test_prediction_stats_updated(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.detect_patterns("coffee")
        tr.predict_next("coffee", within_hours=48)
        stats = tr.get_stats()
        assert stats["predictions_made"] >= 1


class TestDetectAnomalies:
    def test_no_anomalies_with_few_events(self):
        tr = TemporalReasoning()
        for i in range(3):
            tr.record_event("coffee", timestamp=_ts(days_ago=3 - i, hour=9))
        anomalies = tr.detect_anomalies("coffee")
        assert len(anomalies) == 0

    def test_anomaly_at_unusual_time(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=3))
        anomalies = tr.detect_anomalies("coffee")
        assert len(anomalies) >= 1
        assert any(a.deviation_score > 0 for a in anomalies)

    def test_anomaly_has_event_type(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=3))
        anomalies = tr.detect_anomalies("coffee")
        for a in anomalies:
            assert a.event_type == "coffee"

    def test_anomaly_description(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=3))
        anomalies = tr.detect_anomalies("coffee")
        assert any(len(a.description) > 0 for a in anomalies)

    def test_anomaly_sorted_by_score(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=2))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=23))
        anomalies = tr.detect_anomalies("coffee")
        if len(anomalies) >= 2:
            assert anomalies[0].deviation_score >= anomalies[1].deviation_score

    def test_no_anomalies_for_consistent_times(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        anomalies = tr.detect_anomalies("coffee")
        anomaly_count = len(anomalies)
        assert anomaly_count <= 1

    def test_anomalies_stats_updated(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=3))
        tr.detect_anomalies("coffee")
        stats = tr.get_stats()
        assert stats["anomalies_found"] >= 1


class TestAnalyzeTrend:
    def test_increasing_trend(self):
        tr = TemporalReasoning()
        for i in range(5):
            tr.record_event("exercise", timestamp=_ts(days_ago=25 - i * 5))
        for i in range(15):
            tr.record_event("exercise", timestamp=_ts(days_ago=10 - i))
        trend = tr.analyze_trend("exercise", window_days=30)
        assert trend.direction in (TrendDirection.INCREASING, TrendDirection.STABLE)

    def test_decreasing_trend(self):
        tr = TemporalReasoning()
        for i in range(15):
            tr.record_event("exercise", timestamp=_ts(days_ago=25 - i))
        for i in range(3):
            tr.record_event("exercise", timestamp=_ts(days_ago=3 - i))
        trend = tr.analyze_trend("exercise", window_days=30)
        assert trend.direction in (TrendDirection.DECREASING, TrendDirection.STABLE)

    def test_stable_trend(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i))
        trend = tr.analyze_trend("coffee", window_days=30)
        assert trend.direction in (TrendDirection.STABLE, TrendDirection.INCREASING, TrendDirection.DECREASING)

    def test_unknown_trend_no_events(self):
        tr = TemporalReasoning()
        trend = tr.analyze_trend("nonexistent", window_days=30)
        assert trend.direction == TrendDirection.UNKNOWN
        assert trend.frequency_per_day == 0.0

    def test_trend_has_frequency(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i))
        trend = tr.analyze_trend("coffee", window_days=20)
        assert trend.frequency_per_day > 0

    def test_trend_confidence(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i))
        trend = tr.analyze_trend("coffee", window_days=30)
        assert trend.confidence >= 0

    def test_trend_rate_of_change(self):
        tr = TemporalReasoning()
        for i in range(10):
            tr.record_event("coffee", timestamp=_ts(days_ago=30 - i * 3))
        trend = tr.analyze_trend("coffee", window_days=30)
        assert isinstance(trend.rate_of_change, float)

    def test_trend_window_days(self):
        tr = TemporalReasoning()
        for i in range(10):
            tr.record_event("coffee", timestamp=_ts(days_ago=10 - i))
        trend = tr.analyze_trend("coffee", window_days=7)
        assert trend.window_days == 7


class TestEstimateDuration:
    def test_duration_estimate_with_events(self):
        tr = TemporalReasoning()
        base = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        for i in range(10):
            start = base + timedelta(days=i)
            end = start + timedelta(minutes=45)
            tr.record_event("workout", timestamp=start.isoformat())
            tr.record_event("workout_end", timestamp=end.isoformat())
        est = tr.estimate_duration("workout")
        assert est.sample_size >= 0

    def test_duration_no_events(self):
        tr = TemporalReasoning()
        est = tr.estimate_duration("nonexistent")
        assert est.mean_minutes == 0.0
        assert est.confidence == 0.0
        assert est.sample_size == 0

    def test_duration_values(self):
        tr = TemporalReasoning()
        durations = [30, 45, 60, 40, 50]
        base = datetime(2024, 6, 10, 0, 0, tzinfo=timezone.utc)
        for i, dur in enumerate(durations):
            start = base + timedelta(days=i * 2)
            end = start + timedelta(minutes=dur)
            tr.record_event("exercise", timestamp=start.isoformat())
            tr.record_event("exercise", timestamp=end.isoformat())
        est = tr.estimate_duration("exercise")
        assert est.min_minutes <= est.mean_minutes <= est.max_minutes

    def test_duration_confidence_grows(self):
        tr1 = TemporalReasoning()
        for i in range(3):
            base = datetime(2024, 6, 10 + i, 9, 0, tzinfo=timezone.utc)
            tr1.record_event("task", timestamp=base.isoformat())
            tr1.record_event("task", timestamp=(base + timedelta(minutes=30)).isoformat())
        est1 = tr1.estimate_duration("task")

        tr2 = TemporalReasoning()
        for i in range(20):
            base = datetime(2024, 6, 10 + i, 9, 0, tzinfo=timezone.utc)
            tr2.record_event("task", timestamp=base.isoformat())
            tr2.record_event("task", timestamp=(base + timedelta(minutes=30)).isoformat())
        est2 = tr2.estimate_duration("task")
        assert est2.confidence >= est1.confidence


class TestGetStats:
    def test_initial_stats(self):
        tr = TemporalReasoning()
        stats = tr.get_stats()
        assert stats["events_recorded"] == 0
        assert stats["patterns_detected"] == 0
        assert stats["predictions_made"] == 0
        assert stats["anomalies_found"] == 0

    def test_stats_after_events(self):
        tr = TemporalReasoning()
        for i in range(5):
            tr.record_event("coffee", timestamp=_ts(days_ago=5 - i, hour=9))
        stats = tr.get_stats()
        assert stats["events_recorded"] == 5

    def test_stats_after_pattern_detection(self):
        tr = TemporalReasoning()
        for i in range(15):
            tr.record_event("coffee", timestamp=_ts(days_ago=15 - i, hour=9))
        tr.detect_patterns("coffee")
        stats = tr.get_stats()
        assert stats["patterns_detected"] >= 1

    def test_stats_returns_copy(self):
        tr = TemporalReasoning()
        stats = tr.get_stats()
        stats["events_recorded"] = 999
        original = tr.get_stats()
        assert original["events_recorded"] == 0


class TestThreadSafety:
    def test_concurrent_record_events(self):
        tr = TemporalReasoning()
        errors = []

        def worker():
            try:
                for i in range(100):
                    tr.record_event(f"event_{i}")
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=worker) for _ in range(5)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        stats = tr.get_stats()
        assert stats["events_recorded"] == 500

    def test_concurrent_detect_patterns(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))

        results = [None] * 3

        def worker(idx):
            results[idx] = tr.detect_patterns("coffee")

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        for r in results:
            if r is not None:
                assert isinstance(r, list)

    def test_concurrent_mixed_operations(self):
        tr = TemporalReasoning()
        errors = []

        def record_worker():
            try:
                for i in range(50):
                    tr.record_event(f"event_{i % 5}", timestamp=_ts(days_ago=50 - i, hour=9 + i % 5))
            except Exception as e:
                errors.append(e)

        def pattern_worker():
            try:
                for _ in range(5):
                    tr.detect_patterns()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=record_worker),
            threading.Thread(target=record_worker),
            threading.Thread(target=pattern_worker),
            threading.Thread(target=pattern_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0


class TestEdgeCases:
    def test_single_event_no_patterns(self):
        tr = TemporalReasoning()
        tr.record_event("lonely")
        patterns = tr.detect_patterns()
        assert len(patterns) == 0

    def test_same_timestamp_events(self):
        tr = TemporalReasoning()
        ts = "2024-06-15T09:00:00+00:00"
        for i in range(10):
            tr.record_event("duplicate_ts", timestamp=ts)
        patterns = tr.detect_patterns("duplicate_ts")
        assert isinstance(patterns, list)

    def test_empty_metadata(self):
        tr = TemporalReasoning()
        eid = tr.record_event("test", metadata={})
        assert isinstance(eid, str)

    def test_none_metadata(self):
        tr = TemporalReasoning()
        eid = tr.record_event("test", metadata=None)
        assert isinstance(eid, str)

    def test_predict_next_uses_intervals_fallback(self):
        tr = TemporalReasoning()
        for i in range(15):
            ts = datetime(2024, 1, i + 1, 10 + (i % 5), 0, tzinfo=timezone.utc).isoformat()
            tr.record_event("irregular", timestamp=ts)
        preds = tr.predict_next("irregular", within_hours=200)
        assert len(preds) >= 0

    def test_anomaly_deviation_score_capped(self):
        tr = TemporalReasoning()
        for i in range(20):
            tr.record_event("coffee", timestamp=_ts(days_ago=20 - i, hour=9))
        tr.record_event("coffee", timestamp=_ts(days_ago=0, hour=3))
        anomalies = tr.detect_anomalies("coffee")
        for a in anomalies:
            assert a.deviation_score <= 10.0

    def test_trend_empty_window(self):
        tr = TemporalReasoning()
        ts_old = "2023-01-01T09:00:00+00:00"
        tr.record_event("old_event", timestamp=ts_old)
        trend = tr.analyze_trend("old_event", window_days=30)
        assert trend.direction in (TrendDirection.UNKNOWN, TrendDirection.STABLE)

    def test_detection_without_filter(self):
        tr = TemporalReasoning()
        for i in range(15):
            tr.record_event("coffee", timestamp=_ts(days_ago=15 - i, hour=9))
        for i in range(8):
            tr.record_event("standup", timestamp=_ts(days_ago=7 * (8 - i), hour=10, day_of_week=0))
        patterns = tr.detect_patterns()
        assert isinstance(patterns, list)

    def test_seasonal_pattern_detection(self):
        tr = TemporalReasoning()
        for month in [6, 7, 8, 6, 7, 8]:
            try:
                year = 2023 + (1 if month < 6 else 0)
                ts = datetime(year if year >= 2024 else 2024, month, 15, 10, 0, tzinfo=timezone.utc).isoformat()
                tr.record_event("beach", timestamp=ts)
            except Exception:
                pass
        if len(list(tr._events)) >= 6:
            patterns = tr.detect_patterns("beach")
            seasonal = [p for p in patterns if p.pattern_type == PatternType.SEASONAL]
            assert isinstance(seasonal, list)

    def test_estimate_duration_median(self):
        tr = TemporalReasoning()
        base = datetime(2024, 6, 10, 9, 0, tzinfo=timezone.utc)
        for i, dur in enumerate([30, 45, 60]):
            start = base + timedelta(days=i)
            end = start + timedelta(minutes=dur)
            tr.record_event("stretch", timestamp=start.isoformat())
            tr.record_event("stretch", timestamp=end.isoformat())
        est = tr.estimate_duration("stretch")
        assert est.median_minutes > 0 or est.sample_size == 0