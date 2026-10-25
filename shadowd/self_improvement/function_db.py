"""
Self-Improvement Module: BabyAGI-style Function Database

Implements:
- Function generation based on task analysis
- Dependency resolution between functions
- Quality scoring and function selection
- Automatic function synthesis and improvement
"""

from __future__ import annotations

import copy
import hashlib
import inspect
import json
import logging
import os
import re
import sys
import threading
import traceback
import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone

def _fdb_ts(data, key, default):
    try:
        return datetime.fromisoformat(data[key])
    except (ValueError, TypeError, KeyError):
        return default
from enum import Enum
from pathlib import Path
from typing import Any, Callable
import asyncio
import textwrap

logger = logging.getLogger(__name__)

_SANITIZE_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_FUNCTIONS = 5000
_MAX_CODE_LENGTH = 50000
_MAX_DESCRIPTION = 2000
_MAX_NAME_LENGTH = 200
_MAX_TAGS = 20
_MAX_DEPENDENCIES = 50


class FunctionStatus(str, Enum):
    DRAFT = "draft"
    TESTING = "testing"
    APPROVED = "approved"
    DEPRECATED = "deprecated"
    FAILED = "failed"


class Priority(int, Enum):
    CRITICAL = 1
    HIGH = 2
    MEDIUM = 3
    LOW = 4


@dataclass
class GeneratedFunction:
    name: str
    description: str
    code: str
    dependencies: list[str] = field(default_factory=list)
    parameters: dict[str, Any] = field(default_factory=dict)
    returns: str = "Any"
    status: FunctionStatus = FunctionStatus.DRAFT
    quality_score: float = 0.0
    usage_count: int = 0
    success_count: int = 0
    failure_count: int = 0
    avg_execution_time: float = 0.0
    created_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    updated_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    version: int = 1
    parent_id: str | None = None
    tags: list[str] = field(default_factory=list)
    
    _callable: Callable | None = field(default=None, repr=False)
    _id: str = field(default="")
    
    def __post_init__(self):
        if not self._id:
            self._id = self._generate_id()
        self.name = _SANITIZE_STRICT_RE.sub('', str(self.name))[:_MAX_NAME_LENGTH]
        self.description = _SANITIZE_RE.sub('', str(self.description))[:_MAX_DESCRIPTION]
        self.quality_score = max(0.0, min(1.0, self.quality_score))
        self.dependencies = self.dependencies[:_MAX_DEPENDENCIES]
        self.tags = self.tags[:_MAX_TAGS]
        if len(self.code) > _MAX_CODE_LENGTH:
            self.code = self.code[:_MAX_CODE_LENGTH]
    
    def _generate_id(self) -> str:
        content = f"{self.name}_{self.code}"
        return hashlib.sha256(content.encode()).hexdigest()[:16]
    
    @property
    def id(self) -> str:
        return self._id
    
    DANGEROUS_PATTERNS = [
        re.compile(r'__\s*(class|sub|super|bases|mro|dict|globals|locals|code|frame|name)\s*__', re.IGNORECASE),
        re.compile(r'\b(import|from)\s+', re.IGNORECASE),
        re.compile(r'\bexec\s*\(', re.IGNORECASE),
        re.compile(r'\beval\s*\(', re.IGNORECASE),
        re.compile(r'\bcompile\s*\(', re.IGNORECASE),
        re.compile(r'\bopen\s*\(', re.IGNORECASE),
        re.compile(r'\b__import__\s*\(', re.IGNORECASE),
        re.compile(r'\bglobals\s*\(\s*\)', re.IGNORECASE),
        re.compile(r'\blocals\s*\(\s*\)', re.IGNORECASE),
        re.compile(r'\bvars\s*\(\s*\)', re.IGNORECASE),
        re.compile(r'\bdir\s*\(\s*\)', re.IGNORECASE),
        re.compile(r'\bgetattr\s*\(', re.IGNORECASE),
        re.compile(r'\bsetattr\s*\(', re.IGNORECASE),
        re.compile(r'\bdelattr\s*\(', re.IGNORECASE),
        re.compile(r'\btype\s*\(', re.IGNORECASE),
        re.compile(r'\bos\s*\.', re.IGNORECASE),
        re.compile(r'\bsys\s*\.', re.IGNORECASE),
        re.compile(r'\bsubprocess\b', re.IGNORECASE),
        re.compile(r'\bsocket\b', re.IGNORECASE),
        re.compile(r'\bpty\b', re.IGNORECASE),
    ]

    def _validate_code_safety(self, code: str) -> bool:
        for pattern in self.DANGEROUS_PATTERNS:
            if pattern.search(code):
                return False
        return True

    def get_callable(self) -> Callable | None:
        if self._callable is not None:
            return self._callable
        try:
            if not self._validate_code_safety(self.code):
                logger.error("Function %s contains dangerous code patterns", self.name)
                return None
            SAFE_BUILTINS = {
                'len': len, 'range': range, 'int': int, 'float': float, 'str': str,
                'list': list, 'dict': dict, 'set': set, 'tuple': tuple, 'bool': bool,
                'sorted': sorted, 'enumerate': enumerate, 'zip': zip, 'map': map,
                'filter': filter, 'sum': sum, 'min': min, 'max': max, 'abs': abs,
                'round': round, 'isinstance': isinstance, 'hasattr': hasattr,
                'True': True, 'False': False, 'None': None,
                'print': print, 'repr': repr, 'len': len,
                'abs': abs, 'all': all, 'any': any, 'chr': chr, 'ord': ord,
                'hex': hex, 'oct': oct, 'bin': bin, 'pow': pow, 'divmod': divmod,
                'reversed': reversed, 'slice': slice, 'frozenset': frozenset,
                'bytes': bytes, 'bytearray': bytearray, 'memoryview': memoryview,
                'complex': complex, 'hash': hash, 'id': id, 'callable': callable,
            }
            namespace = {"__builtins__": SAFE_BUILTINS}
            exec(self.code, namespace)
            self._callable = namespace.get(self.name)
            return self._callable
        except Exception as e:
            logger.error("Failed to compile function %s: %s", self.name, type(e).__name__)
            return None
    
    def update_stats(self, execution_time: float, success: bool):
        self.usage_count += 1
        if success:
            self.success_count += 1
        else:
            self.failure_count += 1
        alpha = 0.2
        self.avg_execution_time = alpha * execution_time + (1 - alpha) * self.avg_execution_time
        self.quality_score = (self.success_count / self.usage_count) * (
            1.0 / (1.0 + self.avg_execution_time)
        )
        self.updated_at = datetime.now(timezone.utc)
    
    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "name": self.name[:_MAX_NAME_LENGTH],
            "description": self.description[:_MAX_DESCRIPTION],
            "code": self.code[:_MAX_CODE_LENGTH],
            "dependencies": list(self.dependencies),
            "parameters": copy.deepcopy(self.parameters),
            "returns": self.returns,
            "status": self.status.value,
            "quality_score": round(max(0.0, min(1.0, self.quality_score)), 3),
            "usage_count": self.usage_count,
            "success_count": self.success_count,
            "failure_count": self.failure_count,
            "avg_execution_time": round(self.avg_execution_time, 6),
            "created_at": self.created_at.isoformat(),
            "updated_at": self.updated_at.isoformat(),
            "version": self.version,
            "parent_id": self.parent_id,
            "tags": list(self.tags),
        }
    
    @classmethod
    def from_dict(cls, data: dict) -> "GeneratedFunction":
        return cls(
            name=data["name"],
            description=data["description"],
            code=data["code"],
            dependencies=data.get("dependencies", []),
            parameters=data.get("parameters", {}),
            returns=data.get("returns", "Any"),
            status=FunctionStatus(data.get("status", "draft")),
            quality_score=data.get("quality_score", 0.0),
            usage_count=data.get("usage_count", 0),
            success_count=data.get("success_count", 0),
            failure_count=data.get("failure_count", 0),
            avg_execution_time=data.get("avg_execution_time", 0.0),
            created_at=_fdb_ts(data, "created_at", datetime.now(timezone.utc)),
            updated_at=_fdb_ts(data, "updated_at", datetime.now(timezone.utc)),
            version=data.get("version", 1),
            parent_id=data.get("parent_id"),
            tags=data.get("tags", []),
        )


@dataclass
class FunctionResult:
    function_id: str
    success: bool
    result: Any
    error: str | None = None
    execution_time: float = 0.0
    timestamp: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class FunctionDB:
    """
    Self-building function database with dependency resolution.
    
    BabyAGI-style approach:
    1. Analyze task requirements
    2. Check if existing function matches
    3. If not, generate new function
    4. Test and validate
    5. Store for reuse
    """
    
    def __init__(self, storage_path: Path | None = None):
        self.functions: dict[str, GeneratedFunction] = {}
        self.name_index: dict[str, str] = {}
        self.tag_index: dict[str, set[str]] = {}
        self.dependency_graph: dict[str, set[str]] = {}
        self.storage_path = storage_path or Path.home() / "digital-shadow" / "data" / "functions"
        self._execution_order_cache: dict[str, list[str]] = {}
        self._lock = threading.Lock()
        
        self._load_from_storage()
    
    def _load_from_storage(self):
        if not self.storage_path.exists():
            return
        try:
            for file_path in self.storage_path.glob("*.json"):
                with open(file_path) as f:
                    data = json.load(f)
                    func = GeneratedFunction.from_dict(data)
                    self.functions[func.id] = func
                    self.name_index[func.name] = func.id
                    for tag in func.tags:
                        if tag not in self.tag_index:
                            self.tag_index[tag] = set()
                        self.tag_index[tag].add(func.id)
                    self.dependency_graph[func.id] = set(func.dependencies)
        except (json.JSONDecodeError, KeyError, TypeError, ValueError, OSError) as e:
            logger.error("Failed to load functions from storage: %s", type(e).__name__)
    
    def _save_function(self, func: GeneratedFunction):
        self.storage_path.mkdir(parents=True, exist_ok=True)
        file_path = self.storage_path / f"{func.id}.json"
        tmp_path = file_path.with_suffix(f'.{uuid.uuid4().hex[:8]}.tmp')
        try:
            fd = os.open(str(tmp_path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o644)
            with os.fdopen(fd, 'w') as f:
                json.dump(func.to_dict(), f, indent=2)
                f.flush()
                os.fsync(fd)
            os.replace(str(tmp_path), str(file_path))
        except (OSError, TypeError, ValueError) as e:
            logger.warning("Failed to save function %s: %s", func.id, type(e).__name__)
            if tmp_path.exists():
                try:
                    tmp_path.unlink()
                except OSError:
                    pass
    
    def register(self, func: GeneratedFunction) -> str:
        with self._lock:
            if len(self.functions) >= _MAX_FUNCTIONS and func.id not in self.functions:
                evict_id = next(iter(self.functions))
                evicted = self.functions.pop(evict_id, None)
                if evicted:
                    self.name_index.pop(evicted.name, None)
                    self.name_index.pop(f"{evicted.name}_v{evicted.version}", None)
                    for tag in evicted.tags:
                        self.tag_index.get(tag, set()).discard(evict_id)
                    self.dependency_graph.pop(evict_id, None)
                    logger.warning("Evicting function %s (cap=%d)", evict_id, _MAX_FUNCTIONS)
            
            if func.id in self.functions:
                existing = self.functions[func.id]
                if func.version > existing.version:
                    existing.status = FunctionStatus.DEPRECATED
                else:
                    return func.id
            
            self.functions[func.id] = func
            self.name_index[func.name] = func.id
            self.name_index[f"{func.name}_v{func.version}"] = func.id
            
            for tag in func.tags:
                if tag not in self.tag_index:
                    self.tag_index[tag] = set()
                self.tag_index[tag].add(func.id)
            
            self.dependency_graph[func.id] = set(func.dependencies)
            self._execution_order_cache.clear()
        
        self._save_function(func)
        return func.id
    
    def get(self, function_id: str) -> GeneratedFunction | None:
        function_id = _SANITIZE_STRICT_RE.sub('', str(function_id))[:100]
        with self._lock:
            if function_id in self.name_index:
                function_id = self.name_index[function_id]
            return self.functions.get(function_id)
    
    def get_by_name(self, name: str) -> GeneratedFunction | None:
        name = _SANITIZE_STRICT_RE.sub('', str(name))[:_MAX_NAME_LENGTH]
        with self._lock:
            function_id = self.name_index.get(name)
            if function_id:
                return self.functions.get(function_id)
            return None
    
    def search_by_tags(self, tags: list[str], match_all: bool = False) -> list[GeneratedFunction]:
        if not tags:
            return []
        tags = [_SANITIZE_STRICT_RE.sub('', str(t))[:50] for t in tags[:_MAX_TAGS]]
        
        with self._lock:
            if match_all:
                matching_ids = None
                for tag in tags:
                    tag_funcs = self.tag_index.get(tag, set())
                    if matching_ids is None:
                        matching_ids = tag_funcs.copy()
                    else:
                        matching_ids &= tag_funcs
                if matching_ids is None:
                    return []
            else:
                matching_ids = set()
                for tag in tags:
                    matching_ids |= self.tag_index.get(tag, set())
            
            return [copy.deepcopy(self.functions[fid]) for fid in matching_ids if fid in self.functions]
    
    def get_best_function(self, task_tags: list[str]) -> GeneratedFunction | None:
        candidates = self.search_by_tags(task_tags, match_all=False)
        
        approved = [f for f in candidates if f.status == FunctionStatus.APPROVED]
        if not approved:
            approved = [f for f in candidates if f.status == FunctionStatus.TESTING]
        if not approved:
            return None
        
        return max(approved, key=lambda f: (f.quality_score, -f.avg_execution_time))
    
    def resolve_dependencies(self, func: GeneratedFunction) -> list[str]:
        with self._lock:
            if func.id in self._execution_order_cache:
                return list(self._execution_order_cache[func.id])
        
        visited = set()
        stack = []
        order = []
        
        def visit(node_id: str, depth: int = 0):
            if node_id in visited:
                return
            if node_id in stack:
                raise ValueError("Circular dependency detected")
            if depth >= 10:
                logger.warning("Dependency depth limit reached for %s", node_id)
                return
            
            stack.append(node_id)
            
            with self._lock:
                deps = self.dependency_graph.get(node_id, set())
            for dep_id in deps:
                visit(dep_id, depth + 1)
            
            stack.remove(node_id)
            visited.add(node_id)
            order.append(node_id)
        
        visit(func.id)
        with self._lock:
            self._execution_order_cache[func.id] = order
        return list(order)
    
    def execute(
        self,
        function_id: str,
        *args,
        **kwargs
    ) -> FunctionResult:
        import time
        
        function_id = _SANITIZE_STRICT_RE.sub('', str(function_id))[:100]
        func = self.get(function_id)
        if not func:
            return FunctionResult(
                function_id=function_id,
                success=False,
                result=None,
                error="Function not found"
            )
        
        if func.status == FunctionStatus.DEPRECATED:
            logger.warning(f"Using deprecated function: {func.name}")
        
        callable_func = func.get_callable()
        if not callable_func:
            return FunctionResult(
                function_id=function_id,
                success=False,
                result=None,
                error="Failed to compile function"
            )
        
        start_time = time.perf_counter()
        try:
            result = callable_func(*args, **kwargs)
            execution_time = time.perf_counter() - start_time
            
            func.update_stats(execution_time, success=True)
            self._save_function(func)
            
            return FunctionResult(
                function_id=function_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
        except Exception as e:
            execution_time = time.perf_counter() - start_time
            func.update_stats(execution_time, success=False)
            self._save_function(func)
            
            return FunctionResult(
                function_id=function_id,
                success=False,
                result=None,
                error=f"Execution failed: {type(e).__name__}",
                execution_time=execution_time
            )
    
    async def execute_async(
        self,
        function_id: str,
        *args,
        **kwargs
    ) -> FunctionResult:
        import time
        import asyncio
        
        function_id = _SANITIZE_STRICT_RE.sub('', str(function_id))[:100]
        func = self.get(function_id)
        if not func:
            return FunctionResult(
                function_id=function_id,
                success=False,
                result=None,
                error="Function not found"
            )
        
        callable_func = func.get_callable()
        if not callable_func:
            return FunctionResult(
                function_id=function_id,
                success=False,
                result=None,
                error="Failed to compile function"
            )
        
        start_time = time.perf_counter()
        try:
            if asyncio.iscoroutinefunction(callable_func):
                result = await callable_func(*args, **kwargs)
            else:
                result = await asyncio.get_running_loop().run_in_executor(
                    None, lambda: callable_func(*args, **kwargs)
                )
            execution_time = time.perf_counter() - start_time
            
            func.update_stats(execution_time, success=True)
            self._save_function(func)
            
            return FunctionResult(
                function_id=function_id,
                success=True,
                result=result,
                execution_time=execution_time
            )
        except (OSError, TypeError, ValueError):
            execution_time = time.perf_counter() - start_time
            func.update_stats(execution_time, success=False)
            self._save_function(func)
            
            return FunctionResult(
                function_id=function_id,
                success=False,
                result=None,
                error="Async execution failed",
                execution_time=execution_time
            )
    
    def create_improved_version(
        self,
        original_id: str,
        improved_code: str,
        reason: str = ""
    ) -> str:
        original_id = _SANITIZE_STRICT_RE.sub('', str(original_id))[:100]
        improved_code = _SANITIZE_RE.sub('', str(improved_code))[:_MAX_CODE_LENGTH]
        reason = _SANITIZE_RE.sub('', str(reason))[:500]
        original = self.get(original_id)
        if not original:
            raise ValueError(f"Original function not found: {original_id}")
        
        new_func = GeneratedFunction(
            name=original.name,
            description=f"{original.description}\n\nImprovement: {reason}",
            code=improved_code,
            dependencies=original.dependencies.copy(),
            parameters=original.parameters.copy(),
            returns=original.returns,
            status=FunctionStatus.TESTING,
            version=original.version + 1,
            parent_id=original_id,
            tags=original.tags.copy(),
        )
        
        return self.register(new_func)
    
    def get_statistics(self) -> dict:
        with self._lock:
            functions_snapshot = list(self.functions.values())
        
        if not functions_snapshot:
            return {
                "total_functions": 0,
                "by_status": {},
                "avg_quality_score": 0.0,
                "total_executions": 0,
                "success_rate": 0.0,
            }
        
        by_status = {}
        for status in FunctionStatus:
            by_status[status.value] = sum(
                1 for f in functions_snapshot if f.status == status
            )
        
        total_executions = sum(f.usage_count for f in functions_snapshot)
        total_success = sum(f.success_count for f in functions_snapshot)
        success_rate = total_success / total_executions if total_executions > 0 else 0.0
        avg_quality = sum(f.quality_score for f in functions_snapshot) / len(functions_snapshot)
        
        return {
            "total_functions": len(functions_snapshot),
            "by_status": by_status,
            "avg_quality_score": round(avg_quality, 3),
            "total_executions": total_executions,
            "success_rate": round(success_rate, 3),
        }
    
    def clear_cache(self):
        with self._lock:
            self._execution_order_cache.clear()
    
    def prune_failed(self, min_failures: int = 5) -> int:
        with self._lock:
            to_remove = []
            for func in self.functions.values():
                if (
                    func.status == FunctionStatus.FAILED
                    and func.failure_count >= min_failures
                ):
                    to_remove.append(func.id)
            
            for func_id in to_remove:
                func = self.functions[func_id]
                self.name_index.pop(func.name, None)
                self.name_index.pop(f"{func.name}_v{func.version}", None)
                for tag in func.tags:
                    self.tag_index.get(tag, set()).discard(func_id)
                self.dependency_graph.pop(func_id, None)
                self.functions.pop(func_id, None)
        
        for func_id in to_remove:
            file_path = self.storage_path / f"{func_id}.json"
            try:
                if file_path.exists():
                    file_path.unlink()
            except OSError:
                pass
        
        return len(to_remove)