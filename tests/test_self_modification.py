"""Tests for SelfModificationEngine — Code Introspection, Modification, and Validation."""

import os
import copy
import threading
import tempfile
import time

import pytest

from shadowd.cognition.self_modification import (
    CodeSnapshot,
    CodeModification,
    ValidationResultType,
    SelfModificationEngine,
    _sanitize,
    _sanitize_content,
    _DANGEROUS_PATTERNS,
)


class TestCodeSnapshot:
    def test_snapshot_creation(self):
        snap = CodeSnapshot(
            filepath="test.py",
            content="print('hello')",
            hash="",
        )
        assert snap.filepath == "test.py"
        assert snap.content == "print('hello')"
        assert len(snap.snapshot_id) == 12
        assert len(snap.hash) == 64
        assert snap.captured_at is not None

    def test_hash_computation(self):
        snap = CodeSnapshot(
            filepath="test.py",
            content="x = 1",
            hash="",
        )
        expected = __import__("hashlib").sha256(b"x = 1").hexdigest()
        assert snap.hash == expected

    def test_sanitize_filepath(self):
        snap = CodeSnapshot(
            filepath="test\x00\x01.py",
            content="x = 1",
            hash="",
        )
        assert ".py" in snap.filepath
        assert "\x00" not in snap.filepath
        assert "\x01" not in snap.filepath

    def test_to_dict(self):
        snap = CodeSnapshot(
            filepath="test.py",
            content="x = 1",
            hash="",
        )
        d = snap.to_dict()
        assert d["snapshot_id"] == snap.snapshot_id
        assert d["filepath"] == "test.py"
        assert d["hash"] == snap.hash
        assert d["content_length"] == len(snap.content)
        assert "captured_at" in d
        assert "content" not in d


class TestCodeModification:
    def test_create(self):
        mod = CodeModification(
            filepath="test.py",
            original_code="x = 1",
            modified_code="x = 2",
            modification_type="fix_bug",
            description="Fix x",
        )
        assert len(mod.modification_id) == 12
        assert mod.filepath == "test.py"
        assert mod.original_code == "x = 1"
        assert mod.modified_code == "x = 2"
        assert mod.status == "proposed"
        assert mod.confidence == 0.0
        assert mod.safety_approved is False
        assert mod.author == "shadow_ai"

    def test_compute_diff(self):
        mod = CodeModification(
            filepath="test.py",
            original_code="x = 1\ny = 2\n",
            modified_code="x = 1\ny = 3\n",
        )
        diff = mod.compute_diff()
        assert "---" in diff
        assert "+++" in diff
        assert len(diff) > 0
        assert mod.diff == diff

    def test_to_dict_deepcopy(self):
        mod = CodeModification(
            filepath="test.py",
            original_code="x = 1",
            modified_code="x = 2",
            validation_results={"syntax": "passed"},
            metadata={"info": "original"},
        )
        d = mod.to_dict()
        d["validation_results"]["syntax"] = "modified"
        d["metadata"]["info"] = "mutated"
        assert mod.validation_results["syntax"] == "passed"
        assert mod.metadata["info"] == "original"

    def test_status_clamped(self):
        mod = CodeModification(
            filepath="test.py",
            modification_type="fix_bug",
            confidence=1.5,
        )
        assert mod.confidence == 1.0
        mod2 = CodeModification(
            filepath="test.py",
            modification_type="fix_bug",
            confidence=-0.5,
        )
        assert mod2.confidence == 0.0

    def test_author_sanitized(self):
        mod = CodeModification(author="evil\x01user")
        assert mod.author == "eviluser"


class TestSelfModificationEngine:

    def _engine(self, root):
        return SelfModificationEngine(workspace_root=root)

    def _write(self, root, path, content):
        fp = os.path.join(root, path)
        os.makedirs(os.path.dirname(fp), exist_ok=True)
        with open(fp, "w") as f:
            f.write(content)
        return fp

    # --- analyze_codebase ---

    def test_analyze_codebase_counts(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "a.py", "def f(): pass\nclass A: pass\n")
            self._write(d, "b.py", "x = 1\n")
            self._write(d, "tests/test_a.py", "def test(): pass\ndef test2(): pass\n")
            analysis = engine.analyze_codebase(d)
            assert analysis["python_files"] == 3
            assert analysis["test_files"] == 1
            assert analysis["total_lines"] >= 5
            assert analysis["total_functions"] >= 3
            assert analysis["total_classes"] >= 1

    # --- snapshot_file ---

    def test_snapshot_file(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "mod.py", "print(1)")
            snap_id = engine.snapshot_file("mod.py")
            assert snap_id is not None
            assert len(snap_id) == 12
            assert engine._snapshots[snap_id].content == "print(1)"

    def test_snapshot_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            snap_id = engine.snapshot_file("nonexistent.py")
            assert snap_id is None

    def test_snapshot_path_traversal_dots(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            snap_id = engine.snapshot_file("../etc/passwd")
            assert snap_id is None

    def test_snapshot_path_traversal_absolute(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            snap_id = engine.snapshot_file("/etc/passwd")
            assert snap_id is None

    def test_snapshot_file_too_large(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            big_content = "x" * 300000
            self._write(d, "big.py", big_content)
            snap_id = engine.snapshot_file("big.py")
            assert snap_id is None

    # --- read_module ---

    def test_read_module(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "lib/utils.py", "HELLO = 1")
            content = engine.read_module("lib/utils.py")
            assert content == "HELLO = 1"

    def test_read_module_nonexistent(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            content = engine.read_module("ghost.py")
            assert content is None

    def test_read_module_cached(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "cached.py", "V = 1")
            c1 = engine.read_module("cached.py")
            c2 = engine.read_module("cached.py")
            assert c1 == c2
            assert "cached.py" in engine._module_cache

    # --- propose_modification ---

    def test_propose_modification(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "target.py", "OLD = 1")
            mod = engine.propose_modification(
                "target.py",
                "NEW = 2",
                modification_type="refactor",
                description="Rename constant",
            )
            assert mod is not None
            assert mod.filepath == "target.py"
            assert mod.original_code == "OLD = 1"
            assert mod.modified_code == "NEW = 2"
            assert mod.modification_type == "refactor"
            assert mod.snapshot_id != ""
            assert len(mod.diff) > 0

    def test_propose_no_change(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "same.py", "X = 1")
            mod = engine.propose_modification("same.py", "X = 1")
            assert mod is None

    def test_propose_nonexistent_file(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            mod = engine.propose_modification("no.py", "x = 1")
            assert mod is None

    # --- validate_modification ---

    def test_validate_modification_syntax_passes(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "valid.py", "x = 1")
            mod = engine.propose_modification("valid.py", "y = 2\nz = 3\n")
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert results["syntax_check"] == ValidationResultType.PASSED.value
            assert results["import_check"] == ValidationResultType.PASSED.value
            assert results["safety_check"] == ValidationResultType.PASSED.value
            assert results["style_check"] == ValidationResultType.PASSED.value

    def test_validate_syntax_failure(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "broken.py", "x = 1")
            mod = engine.propose_modification("broken.py", "def !!!invalid syntax")
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert "failed_syntax" in results["syntax_check"]

    def test_validate_safety_fails_os_system(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "safe.py", "x = 1")
            mod = engine.propose_modification(
                "safe.py",
                "import os\nos.system('rm -rf /')\n",
            )
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert "failed_safety" in results["safety_check"]

    def test_validate_safety_fails_subprocess(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "safe.py", "x = 1")
            mod = engine.propose_modification(
                "safe.py",
                "import subprocess\nsubprocess.run('ls')\n",
            )
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert "failed_safety" in results["safety_check"]

    def test_validate_safety_fails_eval(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "safe.py", "x = 1")
            mod = engine.propose_modification(
                "safe.py",
                "eval('1 + 1')\n",
            )
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert "failed_safety" in results["safety_check"]

    def test_validate_safety_fails_exec(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "s3.py", "x=1")
            mod = engine.propose_modification("s3.py", "exec('x=2')\n")
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert "failed_safety" in results["safety_check"]

    def test_validate_safety_fails_import_danger(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "s4.py", "x=1")
            mod = engine.propose_modification("s4.py", "__import__('os').system('ls')\n")
            assert mod is not None
            results = engine.validate_modification(mod.modification_id)
            assert "failed_safety" in results["safety_check"]

    def test_validate_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            results = engine.validate_modification("nonexistent_id")
            assert results == {"error": "Modification not found"}

    # --- apply_modification ---

    def test_apply_modification_not_validated(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "apply.py", "OLD = 1")
            mod = engine.propose_modification("apply.py", "NEW = 2")
            assert mod is not None
            result = engine.apply_modification(mod.modification_id)
            assert result is False

    def test_apply_modification_after_validate(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "apply2.py", "OLD = 1")
            mod = engine.propose_modification("apply2.py", "NEW = 2")
            assert mod is not None
            engine.validate_modification(mod.modification_id)
            result = engine.apply_modification(mod.modification_id)
            assert result is True
            with open(os.path.join(d, "apply2.py")) as f:
                assert f.read() == "NEW = 2"

    def test_apply_unsafe_modification(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "unsafe.py", "x=1")
            mod = engine.propose_modification(
                "unsafe.py",
                "import os\nos.system('ls')\n",
            )
            engine.validate_modification(mod.modification_id)
            result = engine.apply_modification(mod.modification_id)
            assert result is False

    def test_apply_nonexistent(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            result = engine.apply_modification("no_such_id")
            assert result is False

    # --- reject_modification ---

    def test_reject_modification(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "rej.py", "OLD = 1")
            mod = engine.propose_modification("rej.py", "NEW = 2")
            assert mod is not None
            result = engine.reject_modification(mod.modification_id)
            assert result is True

    def test_reject_already_applied(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "rej2.py", "OLD = 1")
            mod = engine.propose_modification("rej2.py", "NEW = 2")
            engine.validate_modification(mod.modification_id)
            engine.apply_modification(mod.modification_id)
            result = engine.reject_modification(mod.modification_id)
            assert result is False

    def test_reject_nonexistent(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            result = engine.reject_modification("nonexistent")
            assert result is False

    # --- get_modification_history ---

    def test_get_modification_history(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "hist1.py", "a = 1")
            self._write(d, "hist2.py", "b = 2")
            engine.propose_modification("hist1.py", "a = 10")
            engine.propose_modification("hist2.py", "b = 20")
            history = engine.get_modification_history(limit=10)
            assert len(history) == 2

    def test_get_modification_history_status_filter(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "filt.py", "z = 1")
            m1 = engine.propose_modification("filt.py", "z = 2")
            engine.validate_modification(m1.modification_id)
            engine.apply_modification(m1.modification_id)

            m2 = engine.propose_modification("filt.py", "z = 3")
            engine.reject_modification(m2.modification_id)

            applied = engine.get_modification_history(status="applied")
            assert len(applied) >= 1
            assert all(m["status"] == "applied" for m in applied)

            rejected = engine.get_modification_history(status="rejected")
            assert len(rejected) >= 1
            assert all(m["status"] == "rejected" for m in rejected)

    def test_get_modification_history_respects_limit(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "lim.py", "L = 0")
            for i in range(10):
                engine.propose_modification("lim.py", f"L = {i}")
            history = engine.get_modification_history(limit=3)
            assert len(history) == 3

    # --- get_modification_detail ---

    def test_get_modification_detail(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "detail.py", "D = 1")
            mod = engine.propose_modification("detail.py", "D = 99")
            detail = engine.get_modification_detail(mod.modification_id)
            assert detail is not None
            assert detail["modification_id"] == mod.modification_id
            assert detail["filepath"] == "detail.py"
            assert "diff" in detail
            assert "original_code" in detail
            assert "modified_code" in detail

    def test_get_modification_detail_not_found(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            detail = engine.get_modification_detail("no_such_id")
            assert detail is None

    # --- get_stats ---

    def test_get_stats(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "stats1.py", "s1 = 1")
            m = engine.propose_modification("stats1.py", "s1 = 2")
            engine.validate_modification(m.modification_id)
            engine.reject_modification(m.modification_id)

            stats = engine.get_stats()
            assert "files_snapshotted" in stats
            assert "modifications_proposed" in stats
            assert "modifications_validated" in stats
            assert "modifications_rejected" in stats
            assert "total_modifications" in stats
            assert "total_snapshots" in stats
            assert "active_modifications" in stats
            assert stats["modifications_proposed"] >= 1
            assert stats["modifications_rejected"] >= 1

    def test_stats_copy_safety(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "stcopy.py", "x=1")
            m = engine.propose_modification("stcopy.py", "x=2")
            engine.validate_modification(m.modification_id)

            stats1 = engine.get_stats()
            stats1["modifications_proposed"] = 9999

            stats2 = engine.get_stats()
            assert stats2["modifications_proposed"] != 9999

    # --- thread safety ---

    def test_thread_safety_concurrent_snapshots(self):
        num_files = 20
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            for i in range(num_files):
                self._write(d, f"t{i}.py", f"V{i} = {i}")

            errors = []

            def snapshotter(i):
                try:
                    engine.snapshot_file(f"t{i}.py")
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(num_files):
                t = threading.Thread(target=snapshotter, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0
            stats = engine.get_stats()
            assert stats["files_snapshotted"] >= num_files

    def test_thread_safety_concurrent_propose(self):
        num_ops = 10
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "tp.py", "X = 0")

            errors = []

            def proposer(i):
                try:
                    engine.propose_modification("tp.py", f"X = {i}")
                except Exception as e:
                    errors.append(e)

            threads = []
            for i in range(num_ops):
                t = threading.Thread(target=proposer, args=(i,))
                threads.append(t)
                t.start()

            for t in threads:
                t.join()

            assert len(errors) == 0
            assert len(engine._modifications) <= num_ops

    # --- caps ---

    def test_modifications_capped(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "cap.py", "x=1")

            for i in range(engine._MAX_MODIFICATIONS + 5):
                engine.propose_modification("cap.py", f"x={i}")

            assert len(engine._modifications) <= engine._MAX_MODIFICATIONS

    # --- rollback edge cases ---

    def test_rollback_not_applied(self):
        with tempfile.TemporaryDirectory() as d:
            engine = self._engine(d)
            self._write(d, "rb2.py", "X = 1")
            mod = engine.propose_modification("rb2.py", "X = 2")
            result = engine.rollback_modification(mod.modification_id)
            assert result is False


class TestSelfModificationIntegration:
    def test_full_lifecycle(self):
        with tempfile.TemporaryDirectory() as d:
            engine = SelfModificationEngine(workspace_root=d)

            fp = os.path.join(d, "lifecycle.py")
            with open(fp, "w") as f:
                f.write("STATUS = 'old'\n")

            mod = engine.propose_modification(
                "lifecycle.py",
                "STATUS = 'new'\n",
                modification_type="refactor",
                description="Update status",
            )
            assert mod is not None
            assert mod.status == "proposed"

            results = engine.validate_modification(mod.modification_id)
            assert results["syntax_check"] == ValidationResultType.PASSED.value
            assert results["safety_check"] == ValidationResultType.PASSED.value

            applied = engine.apply_modification(mod.modification_id)
            assert applied is True

            with open(fp) as f:
                assert f.read() == "STATUS = 'new'\n"

            detail = engine.get_modification_detail(mod.modification_id)
            assert detail["status"] == "applied"
            assert detail["applied_at"] is not None

    def test_rollback_flow(self):
        with tempfile.TemporaryDirectory() as d:
            engine = SelfModificationEngine(workspace_root=d)

            fp = os.path.join(d, "rollback.py")
            with open(fp, "w") as f:
                f.write("VERSION = 1\n")

            mod = engine.propose_modification(
                "rollback.py",
                "VERSION = 2\n",
                modification_type="fix_bug",
            )
            assert mod is not None
            assert mod.snapshot_id != ""

            engine.validate_modification(mod.modification_id)
            engine.apply_modification(mod.modification_id)

            with open(fp) as f:
                assert f.read() == "VERSION = 2\n"

            rolled = engine.rollback_modification(mod.modification_id)
            assert rolled is True

            with open(fp) as f:
                assert f.read() == "VERSION = 1\n"

            detail = engine.get_modification_detail(mod.modification_id)
            assert detail["status"] == "rolled_back"
            assert detail["rolled_back_at"] is not None
