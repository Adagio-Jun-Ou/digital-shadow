"""
Cross-platform atomic file I/O helpers.

The codebase needs durable, crash-safe writes: write to a temp file, fsync
it, optionally fsync the parent directory (POSIX), then atomically rename.

Directory fsync is a POSIX-only guarantee. On Windows, ``os.open(dir,
os.O_RDONLY)`` fails with PermissionError and there is no portable directory
fsync, so it must be skipped. These helpers centralise that platform check so
callers don't repeat fragile ``try/except`` scaffolding.
"""

from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any, Optional, Union

__all__ = ["fsync_dir", "atomic_write_bytes", "atomic_write_text", "atomic_write_json"]

PathLike = Union[str, Path]


def fsync_dir(path: PathLike) -> bool:
    """Best-effort fsync of a directory entry.

    Returns True when the directory was synced, False when the platform does
    not support it (e.g. Windows) or the call failed. Never raises — durability
    of the directory entry is desirable but must not break a successful write.
    """
    if os.name == "nt":
        return False
    try:
        dir_fd = os.open(str(path), os.O_RDONLY)
    except OSError:
        return False
    try:
        os.fsync(dir_fd)
        return True
    except OSError:
        return False
    finally:
        try:
            os.close(dir_fd)
        except OSError:
            pass


def atomic_write_bytes(path: PathLike, data: bytes) -> None:
    """Atomically replace *path* with *data* (tmp file + fsync + rename)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp_name = tempfile.mkstemp(
        prefix=path.name + ".", suffix=".tmp", dir=str(path.parent)
    )
    try:
        try:
            f = os.fdopen(fd, "wb")
        except (OSError, ValueError):
            # The raw descriptor is still open; close it so the temp file can
            # be removed (critical on Windows).
            try:
                os.close(fd)
            except OSError:
                pass
            raise
        with f:
            f.write(data)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_name, str(path))
        fsync_dir(path.parent)
    except BaseException:
        try:
            os.unlink(tmp_name)
        except OSError:
            pass
        raise


def atomic_write_text(path: PathLike, text: str,
                      encoding: str = "utf-8") -> None:
    """Atomically replace *path* with *text*."""
    atomic_write_bytes(path, text.encode(encoding))


def atomic_write_json(path: PathLike, data: Any, indent: Optional[int] = 2,
                      default: Optional[Any] = str) -> None:
    """Atomically replace *path* with the JSON serialization of *data*."""
    atomic_write_text(path, json.dumps(data, indent=indent, default=default))
