"""Tests for the ability invocation audit log."""

import json
from pathlib import Path

import pytest

from shadowd.ability_audit import AuditLog, _hash_query, reset_for_tests, get_audit


@pytest.fixture
def audit_dir(tmp_path):
    return tmp_path / "audit"


@pytest.fixture
def audit(audit_dir):
    return AuditLog(audit_dir=audit_dir, enabled=True)


class TestHashQuery:
    def test_stable(self):
        assert _hash_query("hello") == _hash_query("hello")

    def test_different_for_different_inputs(self):
        assert _hash_query("hello") != _hash_query("world")

    def test_returns_short_hex(self):
        h = _hash_query("anything")
        assert len(h) == 16
        assert all(c in "0123456789abcdef" for c in h)

    def test_handles_non_string(self):
        # Doesn't raise.
        h = _hash_query(None)
        assert len(h) == 16


class TestAuditLog:
    def test_record_creates_directory(self, audit_dir):
        a = AuditLog(audit_dir=audit_dir, enabled=True)
        a.record("oracle", "test query", True, 0.5, 0.8, observation_count=10)
        assert audit_dir.exists()
        files = list(audit_dir.glob("abilities-*.jsonl"))
        assert len(files) == 1

    def test_record_writes_jsonl(self, audit):
        audit.record("oracle", "test", True, 0.5, 0.8, observation_count=10)
        files = audit.list_log_files()
        assert files
        text = Path(files[0]).read_text()
        line = text.strip().splitlines()[0]
        parsed = json.loads(line)
        assert parsed["ability"] == "oracle"
        assert parsed["success"] is True
        assert parsed["latency_s"] == 0.5
        assert parsed["confidence"] == 0.8
        assert parsed["obs_count"] == 10
        assert "query_hash" in parsed
        assert "ts" in parsed

    def test_records_failure_with_error_type(self, audit):
        audit.record("oracle", "q", False, 0.1, 0.0, error_type="ValueError")
        recent = audit.read_recent()
        assert recent[-1]["success"] is False
        assert recent[-1]["error_type"] == "ValueError"

    def test_redacts_query_content(self, audit):
        sensitive = "MY_SECRET_PASSWORD=hunter2"
        audit.record("oracle", sensitive, True, 0.1)
        recent = audit.read_recent()
        # Persisted line should not contain the secret.
        files = audit.list_log_files()
        text = Path(files[0]).read_text()
        assert "MY_SECRET_PASSWORD" not in text
        assert "hunter2" not in text
        # But the hash should be present.
        assert recent[-1]["query_hash"]

    def test_read_recent_orders_by_time(self, audit):
        for i in range(5):
            audit.record(f"ab_{i}", f"query_{i}", True, 0.1 * i)
        recent = audit.read_recent(lines=10)
        assert len(recent) == 5
        assert [r["ability"] for r in recent] == [f"ab_{i}" for i in range(5)]

    def test_read_recent_limit(self, audit):
        for i in range(20):
            audit.record(f"ab_{i}", "q", True, 0.0)
        recent = audit.read_recent(lines=5)
        assert len(recent) == 5

    def test_disabled_is_noop(self, audit_dir):
        a = AuditLog(audit_dir=audit_dir, enabled=False)
        a.record("oracle", "q", True, 0.1)
        # No directory should be created when disabled.
        # (audit_dir might not exist at all.)
        assert not audit_dir.exists() or not any(audit_dir.iterdir())
        assert a.read_recent() == []

    def test_concurrent_writes(self, audit):
        import threading
        def worker(i):
            for j in range(50):
                audit.record(f"ab", f"q{j}", True, 0.01)
        threads = [threading.Thread(target=worker, args=(i,)) for i in range(4)]
        for t in threads: t.start()
        for t in threads: t.join()
        recent = audit.read_recent(lines=500)
        # 200 entries expected, all parseable.
        assert len(recent) == 200


class TestRetention:
    def test_cleanup_removes_old_files(self, audit_dir):
        from datetime import datetime, timezone, timedelta
        a = AuditLog(audit_dir=audit_dir, enabled=True, retention_days=7)
        # Manually create some old + new dummy files.
        a.audit_dir.mkdir(parents=True, exist_ok=True)
        old = audit_dir / "abilities-2025-01-01.jsonl"
        recent = audit_dir / "abilities-2026-05-15.jsonl"
        old.write_text("{}\n")
        recent.write_text("{}\n")
        now = datetime(2026, 5, 17, tzinfo=timezone.utc)
        removed = a.cleanup_old(now=now)
        assert removed == 1
        assert not old.exists()
        assert recent.exists()

    def test_cleanup_keeps_within_retention(self, audit_dir):
        from datetime import datetime, timezone
        a = AuditLog(audit_dir=audit_dir, enabled=True, retention_days=30)
        a.audit_dir.mkdir(parents=True, exist_ok=True)
        # 25 days old — within retention.
        recent = audit_dir / "abilities-2026-04-22.jsonl"
        recent.write_text("{}\n")
        now = datetime(2026, 5, 17, tzinfo=timezone.utc)
        removed = a.cleanup_old(now=now)
        assert removed == 0
        assert recent.exists()

    def test_cleanup_ignores_malformed_filenames(self, audit_dir):
        from datetime import datetime, timezone
        a = AuditLog(audit_dir=audit_dir, enabled=True, retention_days=7)
        a.audit_dir.mkdir(parents=True, exist_ok=True)
        # Wrong format — should be skipped.
        (audit_dir / "abilities-not-a-date.jsonl").write_text("{}\n")
        (audit_dir / "abilities-2026-99-99.jsonl").write_text("{}\n")
        now = datetime(2026, 5, 17, tzinfo=timezone.utc)
        # Doesn't raise.
        removed = a.cleanup_old(now=now)
        assert removed == 0
        # Both files still present.
        assert (audit_dir / "abilities-not-a-date.jsonl").exists()

    def test_cleanup_throttled_to_once_per_day(self, audit_dir):
        """Cleanup runs on first record of the day, then skips."""
        from datetime import datetime, timezone
        a = AuditLog(audit_dir=audit_dir, enabled=True, retention_days=7)
        # First record should trigger cleanup.
        a.record("oracle", "q", True, 0.1)
        first_state = a._last_cleanup_date
        assert first_state is not None
        # Stash the date, then prove _maybe_cleanup is a noop next call.
        prior_listing = a.list_log_files()
        a.record("oracle", "q", True, 0.1)
        assert a._last_cleanup_date == first_state
        # Files still exist.
        assert a.list_log_files() == prior_listing

    def test_disabled_cleanup_is_noop(self, audit_dir):
        a = AuditLog(audit_dir=audit_dir, enabled=False, retention_days=7)
        assert a.cleanup_old() == 0


class TestSingleton:
    def test_reset_for_tests(self, tmp_path):
        a = reset_for_tests(audit_dir=tmp_path / "x", enabled=True)
        assert a.enabled
        # The module-level singleton is now the one returned.
        assert get_audit() is a

    def test_singleton_returned(self, tmp_path):
        reset_for_tests(audit_dir=tmp_path / "y", enabled=True)
        a1 = get_audit()
        a2 = get_audit()
        assert a1 is a2
