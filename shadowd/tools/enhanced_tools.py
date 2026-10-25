#!/usr/bin/env python3
"""
Shadow AI Tools - Enhanced tool system with robust operations

Based on patterns from:
- Aider: SEARCH/REPLACE with multi-level matching
- Open Interpreter: Direct execution
- OpenHands: Action confirmation states
- Claw-code: Tool registry pattern
"""

import os
import re
import json
import subprocess
import shutil
import hashlib
import difflib
import logging
import asyncio
from pathlib import Path
from typing import Dict, List, Optional, Any, Tuple, Callable
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timezone
from abc import ABC, abstractmethod
import tempfile
try:
    import chardet
    HAS_CHARDET = True
except ImportError:
    HAS_CHARDET = False

logger = logging.getLogger("shadowd.tools")


def _parse_sandbox_bases(raw: Optional[str]) -> List[Path]:
    """Parse SHADOW_FILE_SANDBOX into resolved base paths.

    Uses os.pathsep (``;`` on Windows, ``:`` on POSIX) so Windows drive
    letters (``C:\\Users\\...``) are not split into bogus entries. A bare
    ``:`` (or ``;``) separated value is also accepted on either platform.
    """
    if not raw:
        return [Path.home().resolve()]
    bases: List[Path] = []
    raw = raw.strip()
    # On Windows os.pathsep is ';' and ':' appears inside drive letters
    # (C:\...), so only ';' and newlines may act as separators there.
    separators = [os.pathsep, "\n"]
    if os.name != "nt":
        separators.append(":")
    parts = [raw]
    for sep in separators:
        expanded: List[str] = []
        for part in parts:
            expanded.extend(part.split(sep))
        parts = expanded
    for part in parts:
        part = part.strip()
        if not part:
            continue
        try:
            bases.append(Path(part).resolve())
        except (OSError, ValueError):
            continue
    return bases or [Path.home().resolve()]


_SANDBOX_BASES = _parse_sandbox_bases(os.environ.get("SHADOW_FILE_SANDBOX"))


def _validate_sandbox_path(path: Path) -> Tuple[bool, str]:
    """Ensure path is within at least one sandbox directory."""
    if not _SANDBOX_BASES or _SANDBOX_BASES == [Path("/").resolve()]:
        return True, ""
    try:
        resolved = path.resolve()
        for base in _SANDBOX_BASES:
            try:
                resolved.relative_to(base)
                return True, ""
            except ValueError:
                continue
        return False, f"Path outside sandbox: {resolved} (sandbox: {_SANDBOX_BASES})"
    except (OSError, ValueError):
        return False, "Invalid path"


class ToolCategory(Enum):
    FILE = "file"
    EXECUTION = "execution"
    WEB = "web"
    KNOWLEDGE = "knowledge"
    ABILITIES = "abilities"
    SYSTEM = "system"


class ActionStatus(Enum):
    PENDING = "pending"
    AWAITING_CONFIRMATION = "awaiting_confirmation"
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    RUNNING = "running"
    SUCCESS = "success"
    FAILED = "failed"
    TIMEOUT = "timeout"


@dataclass
class ToolResult:
    success: bool
    output: str
    data: Dict[str, Any] = field(default_factory=dict)
    error: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    elapsed_ms: float = 0.0
    
    def to_dict(self) -> Dict:
        return {
            "success": self.success,
            "output": self.output,
            "data": self.data,
            "error": self.error,
            "artifacts": self.artifacts,
            "elapsed_ms": round(self.elapsed_ms, 3)
        }


@dataclass
class ToolAction:
    id: str
    tool_name: str
    parameters: Dict[str, Any]
    status: ActionStatus = ActionStatus.PENDING
    thought: str = ""
    result: Optional[ToolResult] = None
    timestamp: datetime = field(default_factory=datetime.now)
    requires_confirmation: bool = False
    security_risk: str = "low"  # low, medium, high, critical
    
    def to_dict(self) -> Dict:
        return {
            "id": self.id,
            "tool_name": self.tool_name,
            "parameters": self.parameters,
            "status": self.status.value,
            "thought": self.thought,
            "result": self.result.to_dict() if self.result else None,
            "timestamp": self.timestamp.isoformat(),
            "requires_confirmation": self.requires_confirmation,
            "security_risk": self.security_risk
        }


class BaseTool(ABC):
    """Base class for all tools."""
    
    name: str = "base_tool"
    description: str = "Base tool"
    category: ToolCategory = ToolCategory.SYSTEM
    parameters_schema: Dict = {}
    requires_confirmation: bool = False
    timeout_seconds: int = 60
    
    def __init__(self):
        self._validate_parameters_schema()
    
    def _validate_parameters_schema(self):
        if self.parameters_schema:
            required = self.parameters_schema.get("required", [])
            properties = self.parameters_schema.get("properties", {})
            for prop in required:
                if prop not in properties:
                    logger.warning(f"Schema issue: {prop} in required but not in properties")
    
    def validate_parameters(self, params: Dict) -> Tuple[bool, str]:
        """Validate parameters against schema."""
        if not self.parameters_schema:
            return True, ""
        
        required = self.parameters_schema.get("required", [])
        for req in required:
            if req not in params:
                return False, f"Missing required parameter: {req}"
        
        properties = self.parameters_schema.get("properties", {})
        for key, value in params.items():
            if key not in properties:
                continue
            
            expected_type = properties[key].get("type")
            if expected_type:
                type_map = {
                    "string": str,
                    "integer": int,
                    "number": (int, float),
                    "boolean": bool,
                    "array": list,
                    "object": dict
                }
                expected_python = type_map.get(expected_type)
                if expected_python and not isinstance(value, expected_python):
                    return False, f"Parameter {key} should be {expected_type}, got {type(value).__name__}"
        
        return True, ""
    
    @abstractmethod
    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        """Execute the tool."""
        pass
    
    def create_action(self, params: Dict, thought: str = "") -> ToolAction:
        """Create an action for this tool."""
        action_id = hashlib.sha256(f"{self.name}:{json.dumps(params)}:{datetime.now(timezone.utc).isoformat()}".encode()).hexdigest()[:12]
        return ToolAction(
            id=action_id,
            tool_name=self.name,
            parameters=params,
            thought=thought,
            requires_confirmation=self.requires_confirmation
        )


# ═════════════════════════════════════════════════════════════════
# FILE TOOLS (Aider-style robust operations)
# ═════════════════════════════════════════════════════════════════

class FileReadTool(BaseTool):
    """Read file with encoding detection and error recovery."""
    
    name = "file_read"
    description = "Read contents of a file with automatic encoding detection"
    category = ToolCategory.FILE
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "offset": {"type": "integer", "description": "Start line (0-indexed)"},
            "limit": {"type": "integer", "description": "Number of lines to read"},
        },
        "required": ["path"]
    }
    
    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        
        valid, error = self.validate_parameters(params)
        if not valid:
            return ToolResult(False, "", error=error)
        
        path = Path(params["path"]).expanduser().resolve()
        
        sandbox_ok, sandbox_err = _validate_sandbox_path(path)
        if not sandbox_ok:
            return ToolResult(False, "", error=sandbox_err)
        
        if not path.exists():
            return ToolResult(False, "", error=f"File not found: {path}")
        
        if not path.is_file():
            return ToolResult(False, "", error=f"Not a file: {path}")
        
        try:
            # Detect encoding
            raw_bytes = path.read_bytes()
            if HAS_CHARDET:
                detected = chardet.detect(raw_bytes)
                encoding = detected.get("encoding", "utf-8") or "utf-8"
            else:
                encoding = "utf-8"
            
            # Read with detected encoding
            content = raw_bytes.decode(encoding, errors="replace")
            lines = content.splitlines()
            
            # Apply offset and limit
            offset = params.get("offset", 0)
            limit = params.get("limit", len(lines))
            
            selected_lines = lines[offset:offset + limit]
            
            # Format output with line numbers
            output_lines = []
            for i, line in enumerate(selected_lines):
                line_num = offset + i + 1
                output_lines.append(f"{line_num:6d}: {line}")
            
            output = "\n".join(output_lines)
            
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return ToolResult(
                True,
                output,
                data={
                    "path": str(path),
                    "total_lines": len(lines),
                    "encoding": encoding,
                    "bytes": len(raw_bytes)
                },
                elapsed_ms=elapsed
            )
        
        except (OSError, ValueError, TypeError, RuntimeError):
            return ToolResult(False, "", error="Failed to read file")

class FileEditTool(BaseTool):
    name = "file_edit"
    description = "Edit file using SEARCH/REPLACE blocks with multi-level matching"
    category = ToolCategory.FILE
    requires_confirmation = True
    security_risk = "medium"
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "search": {"type": "string", "description": "Text to search for"},
            "replace": {"type": "string", "description": "Replacement text"},
            "exact": {"type": "boolean", "description": "Require exact match"},
        },
        "required": ["path", "search", "replace"]
    }

    SEARCH_REPLACE_RE = re.compile(
        r'file_path:\s*(.+?)\n'
        r'(?:search:\s*(.*?)\n)?'
        r'(?:<<<<<<<\s*SEARCH\s*\n(.*?)=======\s*\n)'
        r'(.*?)>>>>>>> REPLACE',
        re.DOTALL,
    )
    BLOCK_RE = re.compile(
        r'<<<<<<<\s*SEARCH\s*\n(.*?)=======\s*\n(.*?)>>>>>>> REPLACE',
        re.DOTALL,
    )
    INLINE_RE = re.compile(
        r'^file_path:\s*(.+?)\s*\n'
        r'search:\s*(.*?)\n'
        r'replace:\s*(.*?)\s*$',
        re.DOTALL,
    )

    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        start_time = datetime.now(timezone.utc)

        valid, error = self.validate_parameters(params)
        if not valid:
            return ToolResult(False, "", error=error)

        raw_path = params.get("path", "")
        raw_search = params.get("search", "")
        raw_replace = params.get("replace", "")
        exact = params.get("exact", False)

        if not raw_path and not raw_search:
            return ToolResult(False, "", error="Missing path and search text")

        path = Path(raw_path).expanduser().resolve() if raw_path else None

        if path:
            sandbox_ok, sandbox_err = _validate_sandbox_path(path)
            if not sandbox_ok:
                return ToolResult(False, "", error=sandbox_err)

        if path and not path.exists():
            return ToolResult(False, "", error=f"File not found: {path}")

        try:
            if path:
                content = path.read_text()
            else:
                content = ""

            original_content = content

            blocks = self._parse_blocks(raw_search, raw_replace)

            if blocks:
                for search_text, replace_text in blocks:
                    result = self._apply_edit(content, search_text, replace_text, exact)
                    if result is None:
                        similar = self._find_similar(content, search_text)
                        return ToolResult(
                            False,
                            "",
                            error=f"Search text not found. Similar text:\n{similar}",
                        )
                    content = result
                new_content = content
            else:
                result = self._apply_edit(content, raw_search, raw_replace, exact)
                if result is None:
                    similar = self._find_similar(content, raw_search)
                    return ToolResult(
                        False,
                        "",
                        error=f"Search text not found. Similar text:\n{similar}",
                    )
                new_content = result

            if path:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=path.suffix + ".tmp"
                )
                try:
                    with os.fdopen(tmp_fd, "w") as f:
                        f.write(new_content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, str(path))
                except OSError:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise

            diff = self._create_diff(original_content, new_content, str(path) if path else "buffer")

            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000

            return ToolResult(
                True,
                f"Successfully edited {path}\n\n{diff}",
                data={
                    "path": str(path) if path else None,
                    "bytes_added": len(new_content) - len(original_content),
                    "lines_changed": len([l for l in diff.split("\n") if l.startswith("+") or l.startswith("-")])
                },
                artifacts=[str(path)] if path else [],
                elapsed_ms=elapsed,
            )

        except (OSError, ValueError, TypeError, RuntimeError):
            return ToolResult(False, "", error="Failed to edit file")

    def _parse_blocks(self, search_text: str, replace_text: str) -> List[Tuple[str, str]]:
        blocks = []

        for m in self.BLOCK_RE.finditer(search_text + "\n" + replace_text):
            blocks.append((m.group(1).rstrip("\n"), m.group(2).rstrip("\n")))

        if not blocks:
            inline = self.INLINE_RE.match(search_text)
            if inline:
                blocks.append((inline.group(2), replace_text))

        if not blocks and "<<<<<<< SEARCH" in search_text:
            parts = search_text.split(">>>>>>> REPLACE")
            for part in parts:
                m = re.search(r'<<<<<<<\s*SEARCH\s*\n(.*?)=======\s*\n(.*)', part, re.DOTALL)
                if m:
                    blocks.append((m.group(1).rstrip("\n"), m.group(2).rstrip("\n")))

        return blocks

    def _apply_edit(self, content: str, search: str, replace: str, exact: bool) -> Optional[str]:
        if not content and not search:
            return replace

        if search in content:
            return content.replace(search, replace, 1)

        if exact:
            return None

        result = self._try_whitespace_flexible(content, search, replace)
        if result is not None:
            return result

        result = self._try_fuzzy_match(content, search, replace, threshold=0.6)
        if result is not None:
            return result

        return None

    def _try_whitespace_flexible(self, content: str, search: str, replace: str) -> Optional[str]:
        search_stripped = search.strip()
        search_lines = search_stripped.splitlines()
        content_lines = content.splitlines()

        if len(search_lines) <= 1:
            search_norm = re.sub(r'\s+', ' ', search_stripped).strip()
            content_norm_lines = [re.sub(r'\s+', ' ', line).strip() for line in content_lines]
            for i, norm_line in enumerate(content_norm_lines):
                if norm_line == search_norm:
                    return "\n".join(content_lines[:i] + replace.splitlines() + content_lines[i + 1 :])
            pattern = re.escape(search_norm).replace(r'\ ', r'\s+')
            m = re.search(pattern, content)
            if m:
                return content[: m.start()] + replace + content[m.end() :]
            return None

        search_norm_lines = [re.sub(r'\s+', ' ', sl).strip() for sl in search_lines]
        content_norm_lines = [re.sub(r'\s+', ' ', cl).strip() for cl in content_lines]

        best_start = -1
        best_score = 0
        search_len = len(search_norm_lines)

        for i in range(len(content_norm_lines) - search_len + 1):
            matches = sum(
                1 for j in range(search_len)
                if content_norm_lines[i + j] == search_norm_lines[j]
            )
            score = matches / search_len
            if score > best_score:
                best_score = score
                best_start = i

        if best_start >= 0 and best_score >= 0.6:
            lines_before = content_lines[:best_start]
            lines_after = content_lines[best_start + search_len :]
            return "\n".join(lines_before + replace.splitlines() + lines_after)

        return None

    def _try_fuzzy_match(self, content: str, search: str, replace: str, threshold: float = 0.6) -> Optional[str]:
        lines = content.splitlines()
        search_lines = search.splitlines()

        if not search_lines:
            return None

        best_ratio = 0.0
        best_start = -1
        best_len = len(search_lines)

        max_match_len = min(len(search_lines) * 2, len(lines)) if len(search_lines) > 1 else len(lines)

        for window_len in range(max(1, len(search_lines) - 1), min(len(search_lines) + 2, len(lines) + 1)):
            for i in range(len(lines) - window_len + 1):
                candidate = "\n".join(lines[i : i + window_len])
                ratio = difflib.SequenceMatcher(None, candidate, search).ratio()
                if ratio > best_ratio:
                    best_ratio = ratio
                    best_start = i
                    best_len = window_len

        if best_start >= 0 and best_ratio >= threshold:
            lines_before = lines[:best_start]
            lines_after = lines[best_start + best_len :]
            return "\n".join(lines_before + replace.splitlines() + lines_after)

        return None

    def _find_similar(self, content: str, search: str) -> str:
        lines = content.splitlines()
        search_words = set(search.lower().split())

        similar_lines = []
        for i, line in enumerate(lines):
            line_words = set(line.lower().split())
            overlap = len(search_words & line_words)
            if overlap > len(search_words) * 0.3:
                similar_lines.append(f"Line {i+1}: {line[:80]}")

        return "\n".join(similar_lines[:5]) if similar_lines else "No similar lines found"

    def _create_diff(self, original: str, new: str, filename: str) -> str:
        diff_lines = list(difflib.unified_diff(
            original.splitlines(keepends=True),
            new.splitlines(keepends=True),
            fromfile=f"{filename} (original)",
            tofile=f"{filename} (modified)",
            lineterm=""
        ))
        return "".join(diff_lines)


class FileWriteTool(BaseTool):
    """Write content to file, creating directories if needed."""
    
    name = "file_write"
    description = "Write content to a file"
    category = ToolCategory.FILE
    requires_confirmation = True
    security_risk = "medium"
    parameters_schema = {
        "type": "object",
        "properties": {
            "path": {"type": "string", "description": "Path to file"},
            "content": {"type": "string", "description": "Content to write"},
            "mode": {"type": "string", "enum": ["write", "append"], "description": "Write mode"},
        },
        "required": ["path", "content"]
    }
    
    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        
        valid, error = self.validate_parameters(params)
        if not valid:
            return ToolResult(False, "", error=error)
        
        path = Path(params["path"]).expanduser().resolve()
        sandbox_ok, sandbox_err = _validate_sandbox_path(path)
        if not sandbox_ok:
            return ToolResult(False, "", error=sandbox_err)
        content = params["content"]
        mode = params.get("mode", "write")
        
        try:
            # Create parent directories
            path.parent.mkdir(parents=True, exist_ok=True)
            
            # Check if file exists
            existed = path.exists()
            original_size = path.stat().st_size if existed else 0
            
            # Write content
            if mode == "append" and existed:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=path.suffix + ".tmp"
                )
                try:
                    existing = path.read_text()
                    with os.fdopen(tmp_fd, "w") as f:
                        f.write(existing + content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, str(path))
                except OSError:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            else:
                tmp_fd, tmp_path = tempfile.mkstemp(
                    dir=str(path.parent), suffix=path.suffix + ".tmp"
                )
                try:
                    with os.fdopen(tmp_fd, "w") as f:
                        f.write(content)
                        f.flush()
                        os.fsync(f.fileno())
                    os.replace(tmp_path, str(path))
                except OSError:
                    try:
                        os.unlink(tmp_path)
                    except OSError:
                        pass
                    raise
            
            new_size = path.stat().st_size
            
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return ToolResult(
                True,
                f"{'Appended to' if mode == 'append' else 'Wrote'} {path}\n{new_size} bytes",
                data={
                    "path": str(path),
                    "existed": existed,
                    "original_size": original_size,
                    "new_size": new_size,
                    "mode": mode
                },
                artifacts=[str(path)],
                elapsed_ms=elapsed
            )
        
        except (OSError, ValueError, TypeError, RuntimeError):
            return ToolResult(False, "", error="Failed to write file")


class FileSearchTool(BaseTool):
    """Search files using glob patterns and content search."""
    
    name = "file_search"
    description = "Search for files matching pattern"
    category = ToolCategory.FILE
    parameters_schema = {
        "type": "object",
        "properties": {
            "pattern": {"type": "string", "description": "Glob pattern"},
            "directory": {"type": "string", "description": "Directory to search"},
            "content": {"type": "string", "description": "Content to search in files"},
        },
        "required": ["pattern"]
    }
    
    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        
        valid, error = self.validate_parameters(params)
        if not valid:
            return ToolResult(False, "", error=error)
        
        pattern = params["pattern"]
        directory = Path(params.get("directory", ".")).expanduser().resolve()
        sandbox_ok, sandbox_err = _validate_sandbox_path(directory)
        if not sandbox_ok:
            return ToolResult(False, "", error=sandbox_err)
        content_search = params.get("content")
        
        try:
            # Glob search
            matches = list(directory.glob(pattern))
            
            output_lines = [f"Found {len(matches)} files matching '{pattern}' in {directory}:\n"]
            
            for match in sorted(matches)[:50]:
                rel_path = match.relative_to(directory) if match.is_relative_to(directory) else match
                output_lines.append(f"  {rel_path}")
            
            # Content search if specified
            if content_search:
                output_lines.append(f"\nSearching for '{content_search}' in files:\n")
                for match in matches[:20]:
                    if match.is_file():
                        try:
                            file_content = match.read_text()
                            if content_search.lower() in file_content.lower():
                                match_rel_path = match.relative_to(directory) if match.is_relative_to(directory) else match
                                output_lines.append(f"  ✓ {match_rel_path}")
                        except (OSError, UnicodeDecodeError):
                            pass
            
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            return ToolResult(
                True,
                "\n".join(output_lines),
                data={
                    "pattern": pattern,
                    "directory": str(directory),
                    "match_count": len(matches)
                },
                elapsed_ms=elapsed
            )
        
        except (OSError, ValueError, TypeError, RuntimeError):
            return ToolResult(False, "", error="File search failed")


# ═════════════════════════════════════════════════════════════════
# EXECUTION TOOLS (Open Interpreter-style)
# ═════════════════════════════════════════════════════════════════

class PythonTool(BaseTool):
    """Execute Python code in sandbox."""
    
    name = "python_exec"
    description = "Execute Python code"
    category = ToolCategory.EXECUTION
    requires_confirmation = True
    security_risk = "high"
    timeout_seconds = 30
    parameters_schema = {
        "type": "object",
        "properties": {
            "code": {"type": "string", "description": "Python code to execute"},
            "timeout": {"type": "integer", "description": "Timeout in seconds"},
        },
        "required": ["code"]
    }
    
    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        
        valid, error = self.validate_parameters(params)
        if not valid:
            return ToolResult(False, "", error=error)
        
        code = params["code"]
        timeout = params.get("timeout", self.timeout_seconds)
        
        # Create temp file for execution
        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_path = f.name
        
        try:
            # Execute with timeout
            proc = await asyncio.create_subprocess_exec(
                "python3", temp_path,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(False, "", error=f"Timeout after {timeout}s")
            
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                output += f"\n--- stderr ---\n{stderr.decode('utf-8', errors='replace')}"
            
            return ToolResult(
                proc.returncode == 0,
                output or "No output",
                data={"return_code": proc.returncode, "timeout": timeout},
                elapsed_ms=elapsed
            )
        
        except (OSError, ValueError, TypeError, RuntimeError):
            return ToolResult(False, "", error="Python execution failed")
        
        finally:
            os.unlink(temp_path)


class ShellTool(BaseTool):
    """Execute shell commands with safety checks."""
    
    name = "shell_exec"
    description = "Execute shell command"
    category = ToolCategory.EXECUTION
    requires_confirmation = True
    security_risk = "critical"
    timeout_seconds = 60
    BLOCKED_PATTERNS = [
        re.compile(r'\brm\s', re.IGNORECASE),
        re.compile(r'\bmkfs\b'),
        re.compile(r'\bdd\s+if='),
        re.compile(r'>\s*/dev/'),
        re.compile(r'\bchmod\s', re.IGNORECASE),
        re.compile(r'\bcurl\b.*\|\s*\b.*(ba)?sh\b'),
        re.compile(r'\bwget\b.*\|\s*\b.*(ba)?sh\b'),
        re.compile(r'\bfind\b.*-delete\b'),
        re.compile(r'\bsudo\b'),
        re.compile(r'\bformat\b'),
        re.compile(r'\bshred\b'),
        re.compile(r'\b:\s*\(\)\s*\{'),
        re.compile(r'\bsubprocess\b'),
        re.compile(r'\bos\s*\.\s*system\b'),
        re.compile(r'\b__import__\s*\('),
    ]
    
    parameters_schema = {
        "type": "object",
        "properties": {
            "command": {"type": "string", "description": "Shell command"},
            "timeout": {"type": "integer", "description": "Timeout in seconds"},
        },
        "required": ["command"]
    }
    
    async def execute(self, params: Dict, context: Dict = None) -> ToolResult:
        start_time = datetime.now(timezone.utc)
        
        valid, error = self.validate_parameters(params)
        if not valid:
            return ToolResult(False, "", error=error)
        
        command = params["command"]
        timeout = params.get("timeout", self.timeout_seconds)
        
        # Safety check
        for pattern in self.BLOCKED_PATTERNS:
            if pattern.search(command):
                return ToolResult(False, "", error="Blocked command pattern detected")
        
        try:
            proc = await asyncio.create_subprocess_shell(
                command,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            except asyncio.TimeoutError:
                proc.kill()
                return ToolResult(False, "", error=f"Timeout after {timeout}s")
            
            elapsed = (datetime.now(timezone.utc) - start_time).total_seconds() * 1000
            
            output = ""
            if stdout:
                output += stdout.decode("utf-8", errors="replace")
            if stderr:
                output += f"\n--- stderr ---\n{stderr.decode('utf-8', errors='replace')}"
            
            return ToolResult(
                proc.returncode == 0,
                output or "No output",
                data={
                    "return_code": proc.returncode,
                    "command": command[:200]  # Truncate for logging
                },
                elapsed_ms=elapsed
            )
        
        except (OSError, ValueError, TypeError, RuntimeError):
            return ToolResult(False, "", error="Shell execution failed")


# ═════════════════════════════════════════════════════════════════
# TOOL REGISTRY
# ═════════════════════════════════════════════════════════════════

class ToolRegistry:
    """Registry for all tools."""
    
    def __init__(self):
        self._tools: Dict[str, BaseTool] = {}
        self._register_default_tools()
    
    def _register_default_tools(self):
        """Register all default tools."""
        default_tools = [
            FileReadTool(),
            FileEditTool(),
            FileWriteTool(),
            FileSearchTool(),
            PythonTool(),
            ShellTool(),
        ]
        
        for tool in default_tools:
            self.register(tool)
    
    def register(self, tool: BaseTool):
        """Register a tool."""
        self._tools[tool.name] = tool
    
    def get(self, name: str) -> Optional[BaseTool]:
        """Get a tool by name."""
        return self._tools.get(name)
    
    def list_tools(self, category: ToolCategory = None) -> List[BaseTool]:
        """List all tools, optionally filtered by category."""
        tools = list(self._tools.values())
        if category:
            tools = [t for t in tools if t.category == category]
        return tools
    
    def get_schemas(self) -> List[Dict]:
        """Get all tool schemas for LLM function calling."""
        schemas = []
        for tool in self._tools.values():
            schemas.append({
                "name": tool.name,
                "description": tool.description,
                "parameters": tool.parameters_schema,
                "requires_confirmation": tool.requires_confirmation
            })
        return schemas
    
    async def execute_tool(self, name: str, params: Dict, context: Dict = None) -> ToolResult:
        """Execute a tool by name."""
        tool = self.get(name)
        if not tool:
            return ToolResult(False, "", error=f"Unknown tool: {name}")
        
        return await tool.execute(params, context)


# Global registry
_registry: Optional[ToolRegistry] = None

def get_registry() -> ToolRegistry:
    """Get global tool registry."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry


__all__ = [
    'ToolCategory',
    'ActionStatus',
    'ToolResult',
    'ToolAction',
    'BaseTool',
    'FileReadTool',
    'FileEditTool',
    'FileWriteTool',
    'FileSearchTool',
    'PythonTool',
    'ShellTool',
    'ToolRegistry',
    'get_registry',
]