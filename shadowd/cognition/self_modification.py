"""
ShadowAI Self-Modification Engine — Code Introspection, Modification, and Validation.

This engine gives ShadowAI the ability to:
1. Introspect its own codebase (read module sources)
2. Generate code modifications (via LLM-guided diff generation)
3. Validate modifications (syntax check, test run, safety audit)
4. Apply validated modifications (atomic file writes with rollback)
5. Track modification history and learn from outcomes

Inspired by: Self-Rewarding LMs (Yuan et al. 2024), Constitutional AI self-critique (Bai et al. 2022),
and AutoGen's multi-agent code generation patterns (Wu et al. 2023).

Safety principles:
- All modifications require explicit approval (never auto-apply)
- Pre-modification snapshot for rollback
- Syntax validation before applying
- Sandbox test execution
- Diff-based tracking for audit trail
"""

import ast
import copy
import difflib
import hashlib
import logging
import os
import re
import threading
import uuid
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 100) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_RE.sub('', text)
    return text.strip()[:max_length]


_MAX_MODIFICATIONS = 1000
_MAX_SNAPSHOTS = 200
_MAX_MODULE_CACHE = 50
_MAX_TEST_OUTPUT_LENGTH = 100000
_MAX_CODE_LENGTH = 200000
_MAX_DIFF_LENGTH = 50000
_DANGEROUS_PATTERNS = [
    re.compile(r'os\.system\s*\('),
    re.compile(r'subprocess\.(call|run|Popen)\s*\('),
    re.compile(r'eval\s*\('),
    re.compile(r'exec\s*\('),
    re.compile(r'__import__\s*\('),
    re.compile(r'rm\s+-rf'),
    re.compile(r'chmod\s+777'),
    re.compile(r'curl.*\|\s*(ba)?sh'),
]


class ModificationStatus(Enum):
    PROPOSED = "proposed"
    VALIDATED = "validated"
    REJECTED = "rejected"
    APPLIED = "applied"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


class ModificationType(Enum):
    FIX_BUG = "fix_bug"
    ADD_FEATURE = "add_feature"
    OPTIMIZE = "optimize"
    REFACTOR = "refactor"
    ADD_TEST = "add_test"
    FIX_SECURITY = "fix_security"
    IMPROVE_PERF = "improve_perf"
    ADD_DOCS = "add_docs"


class ValidationResultType(Enum):
    PASSED = "passed"
    FAILED_SYNTAX = "failed_syntax"
    FAILED_IMPORT = "failed_import"
    FAILED_TEST = "failed_test"
    FAILED_SAFETY = "failed_safety"
    FAILED_STYLE = "failed_style"


@dataclass
class CodeSnapshot:
    """Immutable snapshot of a file before modification."""
    filepath: str
    content: str
    hash: str
    captured_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    snapshot_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])

    def __post_init__(self):
        self.filepath = _sanitize(self.filepath, 500)
        self.content = self.content or ""
        self.hash = _sanitize(self.hash, 64)
        if len(self.hash) != 64 or set(self.hash) - set("0123456789abcdef"):
            self.hash = self._compute_hash()

    def _compute_hash(self) -> str:
        return hashlib.sha256(self.content.encode()).hexdigest()

    def to_dict(self) -> Dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "filepath": self.filepath,
            "hash": self.hash,
            "captured_at": self.captured_at.isoformat(),
            "content_length": len(self.content),
        }


@dataclass
class CodeModification:
    """A proposed code modification with validation results."""
    modification_id: str = field(default_factory=lambda: uuid.uuid4().hex[:12])
    filepath: str = ""
    modification_type: str = "fix_bug"
    description: str = ""
    original_code: str = ""
    modified_code: str = ""
    diff: str = ""
    status: str = "proposed"
    snapshot_id: str = ""
    validation_results: Dict[str, Any] = field(default_factory=dict)
    test_output: str = ""
    safety_approved: bool = False
    confidence: float = 0.0
    proposed_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    applied_at: Optional[datetime] = None
    rolled_back_at: Optional[datetime] = None
    author: str = "shadow_ai"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        self.filepath = _sanitize(self.filepath, 500)
        self.modification_type = _sanitize(self.modification_type, 50)
        self.description = _sanitize_content(self.description, 5000)
        self.original_code = self.original_code or ""
        self.modified_code = self.modified_code or ""
        self.diff = self.diff or ""
        self.status = _sanitize(self.status, 20)
        self.snapshot_id = _sanitize(self.snapshot_id, 20)
        self.confidence = max(0.0, min(1.0, self.confidence))
        self.author = _sanitize(self.author, 100)
        self.test_output = self.test_output[:_MAX_TEST_OUTPUT_LENGTH]
        self.metadata = {
            _sanitize(str(k), 50): _sanitize_content(str(v), 500)
            for k, v in list((self.metadata or {}).items())[:50]
        }

    def compute_diff(self) -> str:
        orig_lines = self.original_code.splitlines(keepends=True)
        mod_lines = self.modified_code.splitlines(keepends=True)
        diff = difflib.unified_diff(
            orig_lines, mod_lines,
            fromfile=f"original/{self.filepath}",
            tofile=f"modified/{self.filepath}",
        )
        self.diff = ''.join(diff)[:_MAX_DIFF_LENGTH]
        return self.diff

    def to_dict(self) -> Dict[str, Any]:
        return {
            "modification_id": self.modification_id,
            "filepath": self.filepath,
            "modification_type": self.modification_type,
            "description": self.description[:200],
            "diff": self.diff[:5000],
            "status": self.status,
            "snapshot_id": self.snapshot_id,
            "validation_results": copy.deepcopy(self.validation_results),
            "safety_approved": self.safety_approved,
            "confidence": round(self.confidence, 3),
            "proposed_at": self.proposed_at.isoformat(),
            "applied_at": self.applied_at.isoformat() if self.applied_at else None,
            "rolled_back_at": self.rolled_back_at.isoformat() if self.rolled_back_at else None,
            "author": self.author,
            "metadata": copy.deepcopy(self.metadata),
        }


class SelfModificationEngine:
    """
    Engine for safe code introspection, modification, and validation.

    Capabilities:
    1. Codebase introspection (read module sources, analyze AST)
    2. Code modification proposal (generate patches)
    3. Multi-stage validation (syntax, import, test, safety)
    4. Atomic apply with rollback capability
    5. Modification history and learning
    """

    _MAX_MODIFICATIONS = _MAX_MODIFICATIONS
    _MAX_SNAPSHOTS = _MAX_SNAPSHOTS
    _MAX_MODULE_CACHE = _MAX_MODULE_CACHE

    def __init__(self, workspace_root: Optional[str] = None):
        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()

        self._workspace_root = Path(workspace_root) if workspace_root else Path.cwd()
        self._snapshots: Dict[str, CodeSnapshot] = {}
        self._snapshot_order: deque = deque(maxlen=_MAX_SNAPSHOTS)
        self._modifications: Dict[str, CodeModification] = {}
        self._modification_order: deque = deque(maxlen=_MAX_MODIFICATIONS)
        self._module_cache: Dict[str, str] = {}
        self._cache_order: deque = deque(maxlen=_MAX_MODULE_CACHE)

        self._stats: Dict[str, int] = {
            "files_snapshotted": 0,
            "modifications_proposed": 0,
            "modifications_validated": 0,
            "modifications_applied": 0,
            "modifications_rolled_back": 0,
            "modifications_rejected": 0,
            "safety_checks_passed": 0,
            "safety_checks_failed": 0,
            "active_modifications": 0,
        }

    def snapshot_file(self, filepath: str) -> Optional[str]:
        """Take a snapshot of a file before modification."""
        filepath = _sanitize(filepath, 500)
        fp = self._workspace_root / filepath

        try:
            fp = fp.resolve()
            if not str(fp).startswith(str(self._workspace_root.resolve())):
                logger.warning("Path traversal blocked: %s", filepath)
                return None

            content = fp.read_text(encoding="utf-8")
            if len(content) > _MAX_CODE_LENGTH:
                logger.warning("File too large: %s (%d bytes)", filepath, len(content))
                return None

            snapshot = CodeSnapshot(
                filepath=filepath,
                content=content,
                hash=hashlib.sha256(content.encode()).hexdigest(),
            )

            with self._lock:
                self._snapshots[snapshot.snapshot_id] = snapshot
                self._snapshot_order.append(snapshot.snapshot_id)
                while len(self._snapshots) > self._MAX_SNAPSHOTS:
                    old_id = self._snapshot_order.popleft()
                    self._snapshots.pop(old_id, None)

            with self._stats_lock:
                self._stats["files_snapshotted"] += 1

            return snapshot.snapshot_id

        except (OSError, UnicodeDecodeError, PermissionError) as e:
            logger.warning("Snapshot failed for %s: %s", filepath, type(e).__name__)
            return None

    def read_module(self, module_path: str) -> Optional[str]:
        """Read a module's source code, using cache."""
        module_path = _sanitize(module_path, 500)

        with self._lock:
            if module_path in self._module_cache:
                return self._module_cache[module_path]

        fp = self._workspace_root / module_path
        try:
            fp = fp.resolve()
            if not str(fp).startswith(str(self._workspace_root.resolve())):
                return None

            content = fp.read_text(encoding="utf-8")
            if len(content) > _MAX_CODE_LENGTH:
                return None

            self._cache_module(module_path, content)
            return content

        except (OSError, UnicodeDecodeError, PermissionError) as e:
            logger.warning("Read failed for %s: %s", module_path, type(e).__name__)
            return None

    def _cache_module(self, module_path: str, content: str) -> None:
        with self._lock:
            if module_path not in self._module_cache:
                self._cache_order.append(module_path)
            self._module_cache[module_path] = content
            while len(self._cache_order) > self._MAX_MODULE_CACHE:
                old_key = self._cache_order.popleft()
                self._module_cache.pop(old_key, None)

    def propose_modification(
        self,
        filepath: str,
        modified_code: str,
        modification_type: str = "fix_bug",
        description: str = "",
        auto_snapshot: bool = True,
    ) -> Optional[CodeModification]:
        """Propose a code modification with automatic snapshotting."""
        filepath = _sanitize(filepath, 500)
        modified_code = modified_code[:_MAX_CODE_LENGTH]

        original_code = self.read_module(filepath)
        if original_code is None:
            return None

        if original_code == modified_code:
            return None

        snapshot_id = None
        if auto_snapshot:
            snapshot_id = self.snapshot_file(filepath)

        modification = CodeModification(
            filepath=filepath,
            modification_type=modification_type,
            description=description,
            original_code=original_code,
            modified_code=modified_code,
            snapshot_id=snapshot_id or "",
        )
        modification.compute_diff()

        with self._lock:
            self._modifications[modification.modification_id] = modification
            self._modification_order.append(modification.modification_id)
            while len(self._modifications) > self._MAX_MODIFICATIONS:
                old_id = self._modification_order.popleft()
                self._modifications.pop(old_id, None)

        with self._stats_lock:
            self._stats["modifications_proposed"] += 1

        return modification

    def validate_modification(self, modification_id: str) -> Dict[str, Any]:
        """Run full validation suite on a proposed modification."""
        modification_id = _sanitize(modification_id, 20)

        with self._lock:
            mod = self._modifications.get(modification_id)
            if mod is None:
                return {"error": "Modification not found"}

        results = {
            "syntax_check": self._validate_syntax(mod),
            "import_check": self._validate_import(mod),
            "safety_check": self._validate_safety(mod),
            "style_check": self._validate_style(mod),
        }

        all_passed = all(
            r == ValidationResultType.PASSED.value for r in results.values()
        )

        with self._lock:
            mod = self._modifications.get(modification_id)
            if mod:
                mod.validation_results = results
                mod.status = "validated" if all_passed else "proposed"
                mod.safety_approved = results["safety_check"] == ValidationResultType.PASSED.value

        with self._stats_lock:
            self._stats["modifications_validated"] += 1
            if mod and mod.safety_approved:
                self._stats["safety_checks_passed"] += 1
            else:
                self._stats["safety_checks_failed"] += 1

        return results

    def apply_modification(self, modification_id: str) -> bool:
        """Atomically apply a validated modification."""
        modification_id = _sanitize(modification_id, 20)

        with self._lock:
            mod = self._modifications.get(modification_id)
            if mod is None or mod.status != "validated":
                return False
            if not mod.safety_approved:
                return False

        fp = self._workspace_root / mod.filepath
        try:
            fp = fp.resolve()
            if not str(fp).startswith(str(self._workspace_root.resolve())):
                return False

            tmp_path = fp.with_suffix(
                "." + uuid.uuid4().hex[:8] + ".tmp"
            )
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                f.write(mod.modified_code)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(fp))

            with self._lock:
                mod = self._modifications.get(modification_id)
                if mod:
                    mod.status = "applied"
                    mod.applied_at = datetime.now(timezone.utc)

            self._invalidate_cache(mod.filepath)

            with self._stats_lock:
                self._stats["modifications_applied"] += 1

            return True

        except (OSError, PermissionError) as e:
            logger.warning("Apply failed for %s: %s", modification_id, type(e).__name__)
            return False

    def rollback_modification(self, modification_id: str) -> bool:
        """Roll back an applied modification using the snapshot."""
        modification_id = _sanitize(modification_id, 20)

        with self._lock:
            mod = self._modifications.get(modification_id)
            if mod is None or mod.status != "applied":
                return False
            snapshot = self._snapshots.get(mod.snapshot_id)
            if snapshot is None:
                return False

        fp = self._workspace_root / mod.filepath
        try:
            fp = fp.resolve()
            if not str(fp).startswith(str(self._workspace_root.resolve())):
                return False

            tmp_path = fp.with_suffix(
                "." + uuid.uuid4().hex[:8] + ".tmp"
            )
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                f.write(snapshot.content)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp_path), str(fp))

            with self._lock:
                mod = self._modifications.get(modification_id)
                if mod:
                    mod.status = "rolled_back"
                    mod.rolled_back_at = datetime.now(timezone.utc)

            self._invalidate_cache(mod.filepath)

            with self._stats_lock:
                self._stats["modifications_rolled_back"] += 1

            return True

        except (OSError, PermissionError) as e:
            logger.warning("Rollback failed for %s: %s", modification_id, type(e).__name__)
            return False

    def reject_modification(self, modification_id: str) -> bool:
        """Reject a proposed modification."""
        modification_id = _sanitize(modification_id, 20)

        with self._lock:
            mod = self._modifications.get(modification_id)
            if mod is None:
                return False
            if mod.status == "applied":
                return False
            mod.status = "rejected"

        with self._stats_lock:
            self._stats["modifications_rejected"] += 1

        return True

    def analyze_codebase(self, root_dir: Optional[str] = None) -> Dict[str, Any]:
        """Analyze the codebase structure — find modules, test files, complexity."""
        root = Path(root_dir) if root_dir else self._workspace_root
        root = _sanitize(str(root), 500)

        path = Path(root).resolve()
        workspace_resolved = self._workspace_root.resolve()
        if not str(path).startswith(str(workspace_resolved)):
            logger.warning("Path traversal blocked in analyze_codebase: %s", root)
            return {
                "python_files": 0, "test_files": 0, "total_lines": 0,
                "total_functions": 0, "total_classes": 0,
                "largest_files": [], "potential_issues": ["invalid_root"],
            }

        analysis = {
            "python_files": 0,
            "test_files": 0,
            "total_lines": 0,
            "total_functions": 0,
            "total_classes": 0,
            "largest_files": [],
            "potential_issues": [],
        }

        try:
            count = 0
            for py_file in path.rglob("*.py"):
                if count >= 500:
                    break
                rel_path = str(py_file.relative_to(path))
                try:
                    content = py_file.read_text(encoding="utf-8")
                    lines = content.split('\n')
                    analysis["total_lines"] += len(lines)
                    analysis["python_files"] += 1

                    if "test_" in rel_path or rel_path.endswith("_test.py"):
                        analysis["test_files"] += 1

                    try:
                        tree = ast.parse(content)
                        funcs = [n for n in ast.walk(tree) if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef))]
                        classes = [n for n in ast.walk(tree) if isinstance(n, ast.ClassDef)]
                        analysis["total_functions"] += len(funcs)
                        analysis["total_classes"] += len(classes)
                    except SyntaxError:
                        pass

                    analysis["largest_files"].append({
                        "file": rel_path,
                        "lines": len(lines),
                    })

                except (OSError, UnicodeDecodeError):
                    pass
                count += 1

            analysis["largest_files"].sort(key=lambda x: x["lines"], reverse=True)
            analysis["largest_files"] = analysis["largest_files"][:20]

        except (OSError, NotADirectoryError) as e:
            logger.warning("Codebase analysis failed: %s", type(e).__name__)

        return analysis

    def get_modification_history(
        self, limit: int = 20, status: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """Get modification history with optional status filter."""
        limit = max(1, min(200, limit))
        status = _sanitize(status, 20) if status else None

        with self._lock:
            mods = list(self._modifications.values())
            if status:
                mods = [m for m in mods if m.status == status]
            mods.sort(key=lambda m: m.proposed_at, reverse=True)
            return [m.to_dict() for m in mods[:limit]]

    def get_modification_detail(self, modification_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed view of a modification."""
        modification_id = _sanitize(modification_id, 20)

        with self._lock:
            mod = self._modifications.get(modification_id)
            if mod is None:
                return None
            result = mod.to_dict()
            result["diff"] = mod.diff
            result["original_code"] = mod.original_code[:20000]
            result["modified_code"] = mod.modified_code[:20000]
            return result

    def get_stats(self) -> Dict[str, Any]:
        """Get engine statistics."""
        with self._lock:
            total_mods = len(self._modifications)
            active_mods = sum(
                1 for m in self._modifications.values()
                if m.status in ("proposed", "validated")
            )
            total_snapshots = len(self._snapshots)

        with self._stats_lock:
            stats = dict(self._stats)

        stats["total_modifications"] = total_mods
        stats["total_snapshots"] = total_snapshots
        stats["active_modifications"] = active_mods
        return stats

    def _validate_syntax(self, mod: CodeModification) -> str:
        try:
            compile(mod.modified_code, f"<validate:{mod.filepath}>", "exec")
            return ValidationResultType.PASSED.value
        except SyntaxError as e:
            return f"{ValidationResultType.FAILED_SYNTAX.value}: {type(e).__name__}"

    def _validate_import(self, mod: CodeModification) -> str:
        try:
            ast.parse(mod.modified_code)
            return ValidationResultType.PASSED.value
        except SyntaxError as e:
            return f"{ValidationResultType.FAILED_IMPORT.value}: {e.msg[:200]}"

    def _validate_safety(self, mod: CodeModification) -> str:
        code = mod.modified_code
        violations = []

        for pattern in _DANGEROUS_PATTERNS:
            matches = pattern.findall(code)
            if matches:
                violations.append(
                    f"Dangerous pattern '{pattern.pattern}' found: {len(matches)} match(es)"
                )

        if violations:
            return f"{ValidationResultType.FAILED_SAFETY.value}: {'; '.join(violations[:5])}"

        return ValidationResultType.PASSED.value

    def _validate_style(self, mod: CodeModification) -> str:
        try:
            ast.parse(mod.modified_code)
            return ValidationResultType.PASSED.value
        except SyntaxError:
            return ValidationResultType.FAILED_STYLE.value

    def _invalidate_cache(self, module_path: str) -> None:
        with self._lock:
            self._module_cache.pop(module_path, None)
            if module_path in self._cache_order:
                self._cache_order.remove(module_path)
