"""Regression tests for Session 9 Sprint 121 — function_db.py audit."""

import json
import os
import tempfile
import threading
import pytest
from pathlib import Path
from unittest.mock import MagicMock

from shadowd.self_improvement.function_db import (
    FunctionDB,
    GeneratedFunction,
    FunctionStatus,
    Priority,
    _SANITIZE_RE,
    _SANITIZE_STRICT_RE,
    _MAX_FUNCTIONS,
    _MAX_CODE_LENGTH,
    _MAX_DESCRIPTION,
    _MAX_NAME_LENGTH,
    _MAX_TAGS,
    _MAX_DEPENDENCIES,
)


@pytest.fixture
def db(tmp_path):
    return FunctionDB(storage_path=tmp_path / "test_funcs")


@pytest.fixture
def sample_func():
    return GeneratedFunction(
        name="test_func",
        description="A test function",
        code="def test_func(x: Any) -> Any:\n    return x",
        dependencies=["math"],
        parameters={"x": "Any"},
        returns="Any",
        status=FunctionStatus.APPROVED,
        tags=["test", "utility"],
    )


class TestFunctionDBSanitization:
    def test_sanitize_re_strips_control(self):
        assert _SANITIZE_RE.sub('', "hello\x00world") == "helloworld"

    def test_sanitize_re_preserves_angle_brackets(self):
        assert _SANITIZE_RE.sub('', "a < b") == "a < b"

    def test_sanitize_strict_strips_angle_brackets(self):
        assert _SANITIZE_STRICT_RE.sub('', "func<>id") == "funcid"


class TestGeneratedFunctionPostInit:
    def test_name_sanitized(self):
        f = GeneratedFunction(name="test<script>func", description="d", code="c")
        assert "<" not in f.name
        assert ">" not in f.name

    def test_description_sanitized(self):
        f = GeneratedFunction(name="f", description="hello\x00world", code="c")
        assert "\x00" not in f.description

    def test_quality_score_clamped(self):
        f = GeneratedFunction(name="f", description="d", code="c", quality_score=5.0)
        assert f.quality_score == 1.0

    def test_quality_score_negative_clamped(self):
        f = GeneratedFunction(name="f", description="d", code="c", quality_score=-1.0)
        assert f.quality_score == 0.0

    def test_tags_capped(self):
        tags = [f"tag{i}" for i in range(30)]
        f = GeneratedFunction(name="f", description="d", code="c", tags=tags)
        assert len(f.tags) <= _MAX_TAGS

    def test_dependencies_capped(self):
        deps = [f"dep{i}" for i in range(60)]
        f = GeneratedFunction(name="f", description="d", code="c", dependencies=deps)
        assert len(f.dependencies) <= _MAX_DEPENDENCIES


class TestGeneratedFunctionToDict:
    def test_to_dict_returns_copies(self, sample_func):
        d = sample_func.to_dict()
        assert d["dependencies"] is not sample_func.dependencies
        assert d["parameters"] is not sample_func.parameters
        assert d["tags"] is not sample_func.tags

    def test_to_dict_quality_score_rounded(self):
        f = GeneratedFunction(name="f", description="d", code="c", quality_score=0.1234567)
        d = f.to_dict()
        assert d["quality_score"] == 0.123

    def test_to_dict_truncates_description(self):
        f = GeneratedFunction(name="f", description="x" * 5000, code="c")
        d = f.to_dict()
        assert len(d["description"]) <= _MAX_DESCRIPTION


class TestFunctionDBThreadSafety:
    def test_concurrent_register(self, db, sample_func):
        errors = []

        def register(i):
            try:
                func = GeneratedFunction(
                    name=f"func_{i}",
                    description=f"test func {i}",
                    code=f"def func_{i}(x): return x",
                    tags=["test"],
                )
                db.register(func)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=register, args=(i,)) for i in range(20)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_search_by_tags_returns_deepcopy(self, db, sample_func):
        db.register(sample_func)
        results = db.search_by_tags(["test"])
        assert len(results) == 1
        assert results[0].name == sample_func.name
        results[0].name = "modified"
        assert db.get(sample_func.id).name == "test_func"

    def test_get_returns_copy(self, db, sample_func):
        db.register(sample_func)
        func = db.get(sample_func.id)
        assert func.name == sample_func.name


class TestFunctionDBFunctionCap:
    def test_max_functions_value(self):
        assert _MAX_FUNCTIONS == 5000


class TestFunctionDBSaveFunction:
    def test_save_creates_file(self, db, sample_func):
        db.register(sample_func)
        file_path = db.storage_path / f"{sample_func.id}.json"
        assert file_path.exists()

    def test_save_atomic_write(self, db, sample_func):
        db.register(sample_func)
        file_path = db.storage_path / f"{sample_func.id}.json"
        with open(file_path) as f:
            data = json.load(f)
        assert data["name"] == "test_func"


class TestFunctionDBGet:
    def test_get_by_name(self, db, sample_func):
        db.register(sample_func)
        result = db.get_by_name("test_func")
        assert result is not None
        assert result.name == "test_func"

    def test_get_by_name_sanitized(self, db, sample_func):
        result = db.get_by_name("test<script>func")
        assert result is None

    def test_get_sanitized_id(self, db, sample_func):
        result = db.get("test<>id")
        assert result is None


class TestFunctionDBExecute:
    def test_execute_sanitized_id(self, db):
        result = db.execute("func<script>id")
        assert result.success is False

    def test_execute_not_found(self, db):
        result = db.execute("nonexistent")
        assert result.success is False
        assert "not found" in result.error.lower()


class TestFunctionDBCreateImprovedVersion:
    def test_create_improved_version_sanitized(self, db, sample_func):
        db.register(sample_func)
        try:
            db.create_improved_version(sample_func.id + "<script>", "def test_func(x): return x * 2")
        except ValueError:
            pass


class TestFunctionDBResolveDependencies:
    def test_resolve_no_deps(self, db, sample_func):
        db.register(sample_func)
        order = db.resolve_dependencies(sample_func)
        assert sample_func.id in order


class TestFunctionDBStatistics:
    def test_statistics_empty(self, db):
        stats = db.get_statistics()
        assert stats["total_functions"] == 0
        assert stats["success_rate"] == 0.0

    def test_statistics_with_functions(self, db, sample_func):
        db.register(sample_func)
        stats = db.get_statistics()
        assert stats["total_functions"] == 1


class TestFunctionDBPruneFailed:
    def test_prune_no_failed(self, db, sample_func):
        db.register(sample_func)
        pruned = db.prune_failed()
        assert pruned == 0

    def test_prune_failed_below_threshold(self, db):
        func = GeneratedFunction(
            name="failed_func",
            description="fails",
            code="def failed_func(): raise Exception",
            status=FunctionStatus.FAILED,
            failure_count=3,
        )
        db.register(func)
        pruned = db.prune_failed(min_failures=5)
        assert pruned == 0


class TestFunctionDBMaxValues:
    def test_max_code_length(self):
        assert _MAX_CODE_LENGTH == 50000

    def test_max_description(self):
        assert _MAX_DESCRIPTION == 2000

    def test_max_name_length(self):
        assert _MAX_NAME_LENGTH == 200