import copy
import re
import threading
from unittest.mock import MagicMock, patch
from datetime import datetime, timezone, timedelta

import pytest

from shadowd.cognition.decision_journal import (
    DecisionJournal,
    DecisionType,
    DecisionStatus,
    DecisionRecord,
    BiasRetrospective,
    DecisionPattern,
    _sanitize,
    _SANITIZE_RE,
)


class TestDecisionJournalBasics:
    def test_init(self):
        j = DecisionJournal()
        stats = j.get_stats()
        assert stats["decisions_recorded"] == 0
        assert stats["current_decisions"] == 0

    def test_record_decision_minimal(self):
        j = DecisionJournal()
        rec = j.record_decision(title="Test", chosen="A")
        assert rec.decision_id.startswith("dec_")
        assert len(rec.decision_id) == 16
        assert rec.title == "Test"
        assert rec.chosen == "A"
        assert rec.status == DecisionStatus.PENDING
        assert rec.decision_type == DecisionType.ROUTINE
        assert rec.confidence == 0.5

    def test_record_decision_full(self):
        j = DecisionJournal()
        rec = j.record_decision(
            title="Strategic pivot",
            chosen="Option B",
            alternatives=["Option A", "Option C"],
            reasoning="B has best risk/reward",
            confidence=0.8,
            biases_flagged=["confirmation", "anchoring"],
            domain="business",
            decision_type="strategic",
            description="Major strategic decision",
            metadata={"priority": "high"},
        )
        assert rec.title == "Strategic pivot"
        assert rec.chosen == "Option B"
        assert len(rec.alternatives) == 2
        assert rec.reasoning == "B has best risk/reward"
        assert rec.confidence == 0.8
        assert rec.biases_flagged == ["confirmation", "anchoring"]
        assert rec.domain == "business"
        assert rec.decision_type == DecisionType.STRATEGIC
        assert rec.description == "Major strategic decision"
        assert rec.metadata == {"priority": "high"}

    def test_record_decision_invalid_type_defaults_routine(self):
        j = DecisionJournal()
        rec = j.record_decision(title="X", chosen="Y", decision_type="nonexistent")
        assert rec.decision_type == DecisionType.ROUTINE

    def test_record_decision_confidence_clamped(self):
        j = DecisionJournal()
        rec = j.record_decision(title="X", chosen="Y", confidence=1.5)
        assert rec.confidence == 1.0
        rec2 = j.record_decision(title="X2", chosen="Y2", confidence=-0.5)
        assert rec2.confidence == 0.0

    def test_get_decision(self):
        j = DecisionJournal()
        rec = j.record_decision(title="Find me", chosen="A")
        fetched = j.get_decision(rec.decision_id)
        assert fetched is not None
        assert fetched.title == "Find me"
        assert fetched is not rec

    def test_get_decision_not_found(self):
        j = DecisionJournal()
        assert j.get_decision("dec_nonexistent") is None

    def test_get_decision_deepcopy(self):
        j = DecisionJournal()
        rec = j.record_decision(title="Original", chosen="A")
        fetched = j.get_decision(rec.decision_id)
        fetched.title = "Modified"
        original = j.get_decision(rec.decision_id)
        assert original.title == "Original"

    def test_list_decisions_empty(self):
        j = DecisionJournal()
        result = j.list_decisions()
        assert result == []

    def test_list_decisions_returns_all(self):
        j = DecisionJournal()
        j.record_decision(title="A", chosen="X")
        j.record_decision(title="B", chosen="Y")
        result = j.list_decisions()
        assert len(result) == 2

    def test_list_decisions_limit(self):
        j = DecisionJournal()
        for i in range(10):
            j.record_decision(title=f"D{i}", chosen="A")
        result = j.list_decisions(limit=3)
        assert len(result) == 3

    def test_list_decisions_filter_domain(self):
        j = DecisionJournal()
        j.record_decision(title="A", chosen="X", domain="tech")
        j.record_decision(title="B", chosen="Y", domain="health")
        result = j.list_decisions(domain="tech")
        assert len(result) == 1
        assert result[0].domain == "tech"

    def test_list_decisions_filter_status(self):
        j = DecisionJournal()
        r = j.record_decision(title="A", chosen="X")
        j.update_status(r.decision_id, "executing")
        result = j.list_decisions(status="executing")
        assert len(result) == 1
        assert result[0].status == DecisionStatus.EXECUTING

    def test_list_decisions_invalid_status_returns_all(self):
        j = DecisionJournal()
        j.record_decision(title="A", chosen="X")
        result = j.list_decisions(status="invalid_status")
        assert len(result) == 1

    def test_decision_id_format(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="C")
        assert rec.decision_id.startswith("dec_")
        assert len(rec.decision_id.split("_")[1]) == 12

    def test_created_at_isoformat(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="C")
        datetime.fromisoformat(rec.created_at)

    def test_to_dict(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="C", decision_type="strategic")
        d = rec.to_dict()
        assert d["decision_type"] == "strategic"
        assert d["status"] == "pending"
        assert "decision_id" in d


class TestDecisionOutcomes:
    def test_record_outcome_completed(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", confidence=0.6)
        success = j.record_outcome(rec.decision_id, "Good result", 0.7)
        assert success
        updated = j.get_decision(rec.decision_id)
        assert updated.status == DecisionStatus.COMPLETED
        assert updated.outcome == "Good result"
        assert updated.outcome_accuracy == 0.7
        assert updated.outcome_recorded_at is not None

    def test_record_outcome_failed(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", confidence=0.6)
        j.record_outcome(rec.decision_id, "Bad result", 0.3)
        updated = j.get_decision(rec.decision_id)
        assert updated.status == DecisionStatus.FAILED

    def test_record_outcome_boundary_completed(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        j.record_outcome(rec.decision_id, "OK", 0.5)
        updated = j.get_decision(rec.decision_id)
        assert updated.status == DecisionStatus.COMPLETED

    def test_record_outcome_not_found(self):
        j = DecisionJournal()
        assert j.record_outcome("dec_nonexistent", "outcome", 0.5) is False

    def test_accuracy_clamped(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        j.record_outcome(rec.decision_id, "test", 1.5)
        updated = j.get_decision(rec.decision_id)
        assert updated.outcome_accuracy == 1.0

    def test_accuracy_clamped_negative(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        j.record_outcome(rec.decision_id, "test", -0.1)
        updated = j.get_decision(rec.decision_id)
        assert updated.outcome_accuracy == 0.0

    def test_outcome_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        j.record_outcome(rec.decision_id, "test\x00bad\x01data", 0.5)
        updated = j.get_decision(rec.decision_id)
        assert "\x00" not in updated.outcome
        assert "\x01" not in updated.outcome

    def test_update_status(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        assert j.update_status(rec.decision_id, "executing")
        updated = j.get_decision(rec.decision_id)
        assert updated.status == DecisionStatus.EXECUTING

    def test_update_status_all_values(self):
        j = DecisionJournal()
        for status_val in ["pending", "executing", "completed", "failed", "cancelled"]:
            rec = j.record_decision(title="T", chosen="A")
            assert j.update_status(rec.decision_id, status_val)
            updated = j.get_decision(rec.decision_id)
            assert updated.status == DecisionStatus(status_val)

    def test_update_status_invalid(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        assert j.update_status(rec.decision_id, "invalid") is False

    def test_update_status_not_found(self):
        j = DecisionJournal()
        assert j.update_status("dec_nonexistent", "completed") is False

    def test_outcomes_recorded_stat(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        j.record_outcome(rec.decision_id, "ok", 0.6)
        assert j.get_stats()["outcomes_recorded"] == 1


class TestRetrospectives:
    def test_run_retrospective_no_data(self):
        j = DecisionJournal()
        retro = j.run_retrospective()
        assert retro.retro_id.startswith("retro_")
        assert retro.decisions_reviewed == 0
        assert retro.accuracy_avg == 0.5
        assert retro.overconfidence_rate == 0.0
        assert retro.biases_found == {}

    def test_run_retrospective_with_data(self):
        j = DecisionJournal()
        rec = j.record_decision(
            title="T", chosen="A", confidence=0.9,
            biases_flagged=["confirmation"], domain="tech"
        )
        j.record_outcome(rec.decision_id, "OK", 0.4)
        retro = j.run_retrospective()
        assert retro.decisions_reviewed == 1
        assert "confirmation" in retro.biases_found
        assert retro.overconfidence_rate == 1.0
        assert len(retro.findings) > 0
        assert len(retro.recommendations) > 0

    def test_retro_period(self):
        j = DecisionJournal()
        retro = j.run_retrospective(period_days=60)
        period_start = datetime.fromisoformat(retro.period_start)
        period_end = datetime.fromisoformat(retro.period_end)
        delta = period_end - period_start
        assert 59 <= delta.days <= 61

    def test_retro_period_clamped(self):
        j = DecisionJournal()
        retro = j.run_retrospective(period_days=9999)
        period_start = datetime.fromisoformat(retro.period_start)
        period_end = datetime.fromisoformat(retro.period_end)
        delta = period_end - period_start
        assert delta.days <= 3650

    def test_retro_id_format(self):
        j = DecisionJournal()
        retro = j.run_retrospective()
        assert retro.retro_id.startswith("retro_")
        assert len(retro.retro_id.split("_")[1]) == 12

    def test_retro_findings_overconfidence(self):
        j = DecisionJournal()
        for i in range(5):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.9)
            j.record_outcome(rec.decision_id, "low", 0.2)
        retro = j.run_retrospective()
        assert any("Overconfidence" in f for f in retro.findings)

    def test_retro_recommendations_for_bias(self):
        j = DecisionJournal()
        rec = j.record_decision(
            title="T", chosen="A", confidence=0.9,
            biases_flagged=["anchoring", "anchoring", "anchoring"]
        )
        j.record_outcome(rec.decision_id, "ok", 0.4)
        retro = j.run_retrospective()
        assert any("anchoring" in r for r in retro.recommendations)

    def test_retro_low_accuracy_finding(self):
        j = DecisionJournal()
        for i in range(3):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.5)
            j.record_outcome(rec.decision_id, "bad", 0.2)
        retro = j.run_retrospective()
        assert any("Below-average accuracy" in f for f in retro.findings)

    def test_retro_low_domain_accuracy_finding(self):
        j = DecisionJournal()
        for i in range(3):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.5, domain="finance")
            j.record_outcome(rec.decision_id, "bad", 0.3)
        retro = j.run_retrospective()
        assert any("finance" in f for f in retro.findings)

    def test_retro_findings_capped_at_10(self):
        j = DecisionJournal()
        for i in range(15):
            rec = j.record_decision(
                title=f"T{i}", chosen="A", confidence=0.9,
                biases_flagged=[f"bias_{i}"]
            )
            j.record_outcome(rec.decision_id, "low", 0.2)
        retro = j.run_retrospective()
        assert len(retro.findings) <= 10

    def test_retro_recommendations_capped_at_10(self):
        j = DecisionJournal()
        for i in range(15):
            rec = j.record_decision(
                title=f"T{i}", chosen="A", confidence=0.9,
                biases_flagged=[f"bias_{i}"]
            )
            j.record_outcome(rec.decision_id, "low", 0.2)
        retro = j.run_retrospective()
        assert len(retro.recommendations) <= 10

    def test_retro_to_dict(self):
        j = DecisionJournal()
        retro = j.run_retrospective()
        d = retro.to_dict()
        assert "retro_id" in d
        assert "biases_found" in d

    def test_stats_retrospectives_run(self):
        j = DecisionJournal()
        j.run_retrospective()
        j.run_retrospective()
        assert j.get_stats()["retrospectives_run"] == 2

    def test_retro_ignores_pending_decisions(self):
        j = DecisionJournal()
        j.record_decision(title="T", chosen="A", confidence=0.9, biases_flagged=["bias1"])
        retro = j.run_retrospective()
        assert retro.decisions_reviewed == 0


class TestPatternDetection:
    def test_detect_patterns_empty(self):
        j = DecisionJournal()
        patterns = j.detect_patterns()
        assert patterns == []

    def test_detect_overconfidence_pattern_domain(self):
        j = DecisionJournal()
        for i in range(5):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.9, domain="tech")
            j.record_outcome(rec.decision_id, "low", 0.3)
        patterns = j.detect_patterns(min_frequency=3)
        overconfident = [p for p in patterns if p.pattern_type == "consistently_overconfident"]
        assert len(overconfident) >= 1
        assert any("tech" in p.description for p in overconfident)

    def test_detect_overconfidence_pattern_type(self):
        j = DecisionJournal()
        for i in range(5):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.9, decision_type="strategic")
            j.record_outcome(rec.decision_id, "low", 0.3)
        patterns = j.detect_patterns(min_frequency=3)
        overconfident = [p for p in patterns if p.pattern_type == "consistently_overconfident"]
        assert len(overconfident) >= 1
        assert any("strategic" in p.description for p in overconfident)

    def test_detect_always_avoids_pattern(self):
        j = DecisionJournal()
        for i in range(5):
            j.record_decision(title=f"T{i}", chosen="safe_option")
        patterns = j.detect_patterns(min_frequency=3)
        avoid_patterns = [p for p in patterns if p.pattern_type == "repeated_choice"]
        assert len(avoid_patterns) >= 1
        assert any("safe_option" in p.description for p in avoid_patterns)

    def test_detect_low_confidence_domain_pattern(self):
        j = DecisionJournal()
        for i in range(5):
            j.record_decision(title=f"T{i}", chosen="A", confidence=0.1, domain="nuclear")
        patterns = j.detect_patterns(min_frequency=3)
        domain_biases = [p for p in patterns if p.pattern_type == "domain_bias"]
        assert len(domain_biases) >= 1
        assert any("nuclear" in p.description for p in domain_biases)

    def test_pattern_id_format(self):
        j = DecisionJournal()
        for i in range(5):
            j.record_decision(title=f"T{i}", chosen="same")
        patterns = j.detect_patterns(min_frequency=3)
        for p in patterns:
            assert p.pattern_id.startswith("dpat_")
            assert len(p.pattern_id.split("_")[1]) == 12

    def test_pattern_to_dict(self):
        j = DecisionJournal()
        for i in range(5):
            j.record_decision(title=f"T{i}", chosen="same")
        patterns = j.detect_patterns(min_frequency=3)
        for p in patterns:
            d = p.to_dict()
            assert "pattern_id" in d
            assert "pattern_type" in d
            assert "frequency" in d

    def test_min_frequency_filter(self):
        j = DecisionJournal()
        for i in range(2):
            j.record_decision(title=f"T{i}", chosen="same")
        patterns = j.detect_patterns(min_frequency=3)
        assert len(patterns) == 0

    def test_patterns_stored_in_journal(self):
        j = DecisionJournal()
        for i in range(5):
            j.record_decision(title=f"T{i}", chosen="same")
        j.detect_patterns(min_frequency=3)
        assert j.get_stats()["patterns_detected"] > 0

    def test_patterns_capped_domains(self):
        j = DecisionJournal()
        for i in range(5):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.9, domain="tech")
            j.record_outcome(rec.decision_id, "low", 0.3)
        patterns = j.detect_patterns(min_frequency=3)
        for p in patterns:
            assert len(p.domains) <= 5


class TestDecisionAccuracy:
    def test_accuracy_no_data(self):
        j = DecisionJournal()
        result = j.get_decision_accuracy()
        assert result["total_resolved"] == 0
        assert result["avg_accuracy"] == 0.0

    def test_accuracy_with_data(self):
        j = DecisionJournal()
        rec1 = j.record_decision(title="T1", chosen="A", confidence=0.7)
        j.record_outcome(rec1.decision_id, "good", 0.8)
        rec2 = j.record_decision(title="T2", chosen="B", confidence=0.6)
        j.record_outcome(rec2.decision_id, "ok", 0.6)
        result = j.get_decision_accuracy()
        assert result["total_resolved"] == 2
        assert abs(result["avg_accuracy"] - 0.7) < 0.01
        assert result["overconfidence_count"] == 0

    def test_accuracy_filter_domain(self):
        j = DecisionJournal()
        rec1 = j.record_decision(title="T1", chosen="A", domain="tech")
        j.record_outcome(rec1.decision_id, "ok", 0.6)
        rec2 = j.record_decision(title="T2", chosen="B", domain="health")
        j.record_outcome(rec2.decision_id, "ok", 0.7)
        result = j.get_decision_accuracy(domain="tech")
        assert result["total_resolved"] == 1
        assert result["domain"] == "tech"

    def test_accuracy_empty_domain(self):
        j = DecisionJournal()
        result = j.get_decision_accuracy(domain="nonexistent")
        assert result["total_resolved"] == 0

    def test_overconfidence_report_no_data(self):
        j = DecisionJournal()
        report = j.get_overconfidence_report()
        assert report["total"] == 0
        assert report["overconfidence_rate"] == 0.0

    def test_overconfidence_report_with_data(self):
        j = DecisionJournal()
        rec1 = j.record_decision(title="T1", chosen="A", confidence=0.9)
        j.record_outcome(rec1.decision_id, "bad", 0.3)
        rec2 = j.record_decision(title="T2", chosen="B", confidence=0.5)
        j.record_outcome(rec2.decision_id, "ok", 0.6)
        report = j.get_overconfidence_report()
        assert report["total"] == 2
        assert report["overconfident_count"] == 1
        assert report["overconfidence_rate"] == 0.5
        assert report["avg_overconfidence_gap"] > 0

    def test_overconfidence_by_domain(self):
        j = DecisionJournal()
        rec1 = j.record_decision(title="T1", chosen="A", confidence=0.9, domain="tech")
        j.record_outcome(rec1.decision_id, "bad", 0.3)
        report = j.get_overconfidence_report()
        assert "tech" in report["by_domain"]
        assert report["by_domain"]["tech"]["overconfident"] == 1

    def test_overconfidence_by_type(self):
        j = DecisionJournal()
        rec1 = j.record_decision(title="T1", chosen="A", confidence=0.9, decision_type="strategic")
        j.record_outcome(rec1.decision_id, "bad", 0.3)
        report = j.get_overconfidence_report()
        assert "strategic" in report["by_type"]


class TestBoundedEviction:
    def test_decision_cap(self):
        j = DecisionJournal()
        j._MAX_DECISIONS = 10
        ids = []
        for i in range(12):
            rec = j.record_decision(title=f"T{i}", chosen="A")
            ids.append(rec.decision_id)
        assert len(j._decisions) <= 10
        assert ids[0] not in j._decisions

    def test_retro_cap(self):
        j = DecisionJournal()
        j._MAX_RETROSPECTIVES = 5
        for i in range(7):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.9)
            j.record_outcome(rec.decision_id, "low", 0.3)
            j.run_retrospective()
        assert len(j._retrospectives) <= 5

    def test_pattern_cap(self):
        j = DecisionJournal()
        j._MAX_PATTERNS = 5
        for i in range(20):
            j.record_decision(title=f"T{i}", chosen="same_choice")
        for _ in range(5):
            j.detect_patterns(min_frequency=1)
        assert len(j._patterns) <= 5

    def test_decision_fifo_order(self):
        j = DecisionJournal()
        j._MAX_DECISIONS = 3
        rec0 = j.record_decision(title="T0", chosen="A")
        rec1 = j.record_decision(title="T1", chosen="B")
        rec2 = j.record_decision(title="T2", chosen="C")
        rec3 = j.record_decision(title="T3", chosen="D")
        assert rec0.decision_id not in j._decisions
        assert rec1.decision_id in j._decisions

    def test_eviction_logs_warning(self):
        j = DecisionJournal()
        j._MAX_DECISIONS = 2
        with patch("shadowd.cognition.decision_journal.logger") as mock_logger:
            j.record_decision(title="T0", chosen="A")
            j.record_decision(title="T1", chosen="B")
            j.record_decision(title="T2", chosen="C")
            mock_logger.warning.assert_called()


class TestInputSanitization:
    def test_sanitize_strips_control_chars(self):
        result = _sanitize("hello\x00world\x01test")
        assert result == "helloworldtest"

    def test_sanitize_truncates(self):
        long_text = "a" * 500
        result = _sanitize(long_text, 200)
        assert len(result) == 200

    def test_sanitize_strips(self):
        result = _sanitize("  hello  ")
        assert result == "hello"

    def test_sanitize_non_string(self):
        result = _sanitize(123)
        assert result == "123"

    def test_title_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="test\x00title", chosen="A")
        assert "\x00" not in rec.title

    def test_description_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", description="desc\x01ription")
        assert "\x01" not in rec.description

    def test_reasoning_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", reasoning="reason\x02ing")
        assert "\x02" not in rec.reasoning

    def test_alternatives_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", alternatives=["alt\x00one"])
        assert "\x00" not in rec.alternatives[0]

    def test_alternatives_capped_at_10(self):
        j = DecisionJournal()
        alts = [f"alt_{i}" for i in range(15)]
        rec = j.record_decision(title="T", chosen="A", alternatives=alts)
        assert len(rec.alternatives) == 10

    def test_biases_capped_at_10(self):
        j = DecisionJournal()
        biases = [f"bias_{i}" for i in range(15)]
        rec = j.record_decision(title="T", chosen="A", biases_flagged=biases)
        assert len(rec.biases_flagged) == 10

    def test_metadata_capped_at_20_keys(self):
        j = DecisionJournal()
        meta = {f"k{i}": i for i in range(30)}
        rec = j.record_decision(title="T", chosen="A", metadata=meta)
        assert len(rec.metadata) <= 20

    def test_chosen_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="choice\x03here")
        assert "\x03" not in rec.chosen

    def test_domain_sanitized(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", domain="tech\x04field")
        assert "\x04" not in rec.domain

    def test_title_truncated_200(self):
        j = DecisionJournal()
        rec = j.record_decision(title="A" * 500, chosen="A")
        assert len(rec.title) <= 200

    def test_description_truncated_2000(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A", description="D" * 5000)
        assert len(rec.description) <= 2000


class TestThreadSafety:
    def test_concurrent_decisions(self):
        j = DecisionJournal()
        errors = []
        results = []

        def worker(idx):
            try:
                rec = j.record_decision(title=f"T{idx}", chosen=f"C{idx}")
                results.append(rec.decision_id)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=worker, args=(i,)) for i in range(50)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 50
        assert len(set(results)) == 50

    def test_concurrent_outcomes(self):
        j = DecisionJournal()
        recs = []
        for i in range(20):
            recs.append(j.record_decision(title=f"T{i}", chosen="A"))

        errors = []

        def record_outcome(idx):
            try:
                j.record_outcome(recs[idx].decision_id, "ok", 0.6)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=record_outcome, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert j.get_stats()["outcomes_recorded"] == 20

    def test_concurrent_status_updates(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")

        errors = []

        def update_status():
            try:
                j.update_status(rec.decision_id, "executing")
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=update_status) for _ in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        updated = j.get_decision(rec.decision_id)
        assert updated.status == DecisionStatus.EXECUTING

    def test_concurrent_retrospective(self):
        j = DecisionJournal()
        for i in range(5):
            rec = j.record_decision(title=f"T{i}", chosen="A", confidence=0.7)
            j.record_outcome(rec.decision_id, "ok", 0.6)

        errors = []
        results = []

        def run_retro():
            try:
                retro = j.run_retrospective()
                results.append(retro.retro_id)
            except Exception as e:
                errors.append(type(e).__name__)

        threads = [threading.Thread(target=run_retro) for _ in range(10)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0
        assert len(results) == 10


class TestDecisionEnums:
    def test_decision_type_values(self):
        expected = ["strategic", "tactical", "operational", "ethical", "social", "creative", "routine"]
        actual = [e.value for e in DecisionType]
        assert actual == expected

    def test_decision_status_values(self):
        expected = ["pending", "executing", "completed", "failed", "cancelled"]
        actual = [e.value for e in DecisionStatus]
        assert actual == expected

    def test_decision_type_is_str_enum(self):
        assert isinstance(DecisionType.STRATEGIC, str)

    def test_decision_status_is_str_enum(self):
        assert isinstance(DecisionStatus.PENDING, str)


class TestDataclasses:
    def test_decision_record_to_dict_keys(self):
        j = DecisionJournal()
        rec = j.record_decision(title="T", chosen="A")
        d = rec.to_dict()
        expected_keys = {
            "decision_id", "title", "description", "alternatives",
            "chosen", "reasoning", "confidence", "biases_flagged",
            "domain", "decision_type", "status", "outcome",
            "outcome_accuracy", "outcome_recorded_at", "created_at",
            "metadata",
        }
        assert set(d.keys()) == expected_keys

    def test_bias_retrospective_to_dict_keys(self):
        j = DecisionJournal()
        retro = j.run_retrospective()
        d = retro.to_dict()
        expected_keys = {
            "retro_id", "period_start", "period_end", "decisions_reviewed",
            "biases_found", "accuracy_avg", "overconfidence_rate",
            "findings", "recommendations", "created_at",
        }
        assert set(d.keys()) == expected_keys

    def test_decision_pattern_to_dict_keys(self):
        j = DecisionJournal()
        for i in range(5):
            j.record_decision(title=f"T{i}", chosen="same")
        patterns = j.detect_patterns(min_frequency=3)
        if patterns:
            d = patterns[0].to_dict()
            expected_keys = {
                "pattern_id", "pattern_type", "description",
                "frequency", "confidence", "domains", "created_at",
            }
            assert set(d.keys()) == expected_keys

    def test_list_decisions_returns_deepcopy(self):
        j = DecisionJournal()
        j.record_decision(title="Original", chosen="A")
        result = j.list_decisions()
        result[0].title = "Modified"
        result2 = j.list_decisions()
        assert result2[0].title == "Original"


class TestShadowOSIntegration:
    def test_journal_in_shadow_os(self):
        # ShadowOS exposes both:
        #   self.journal           — ShadowJournal (free-form event log)
        #   self.decision_journal  — DecisionJournal (structured choices)
        # The DecisionJournal is the one this sprint introduced.
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        assert hasattr(os, 'decision_journal')
        assert isinstance(os.decision_journal, DecisionJournal)
        # The legacy `journal` slot remains present for back-compat.
        assert hasattr(os, 'journal')

    def test_journal_in_status(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        status = os.get_status()
        assert "decision_journal_stats" in status

    def test_journal_in_health(self):
        from shadowd.cognition.shadow_os import ShadowOS
        os = ShadowOS()
        os._running = True
        os._start_time = 0
        health = os.get_health()
        assert "decision_journal" in health["modules"]