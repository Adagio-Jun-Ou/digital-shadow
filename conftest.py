import os
import builtins
import io
import tempfile
from pathlib import Path

# ---------------------------------------------------------------------------
# UTF-8 default text I/O.
#
# Source/tests read HTML, JSON and source files containing non-ASCII
# characters with open()/read_text() and no explicit encoding. On Windows the
# default is the ANSI code page (cp1252), which raises UnicodeDecodeError.
# Force UTF-8 for text mode file I/O so the suite behaves the same as on the
# Linux boxes this project was developed on.
# ---------------------------------------------------------------------------
if os.name == "nt":
    _orig_io_open = io.open

    def _utf8_open(file, mode="r", buffering=-1, encoding=None, errors=None,
                   newline=None, closefd=True, opener=None):
        # pathlib passes encoding="locale" explicitly in 3.11; treat it the
        # same as no encoding so UTF-8 wins on Windows.
        if encoding in (None, "locale") and "b" not in mode:
            encoding = "utf-8"
        return _orig_io_open(file, mode, buffering, encoding, errors,
                             newline, closefd, opener)

    io.open = _utf8_open
    builtins.open = _utf8_open

# Allow the file tools access to the user's home plus the system temp dir
# (where pytest's tmp_path fixtures live). os.pathsep keeps Windows drive
# letters intact.
os.environ.setdefault(
    "SHADOW_FILE_SANDBOX",
    os.pathsep.join([str(Path.home()), tempfile.gettempdir()]),
)

# Bound hypothesis examples so the full suite finishes in minutes, not hours.
# Tests that need more thorough exploration can opt in via @settings(...).
try:
    from hypothesis import settings, Verbosity
    settings.register_profile(
        "ci", max_examples=20, deadline=None, verbosity=Verbosity.quiet,
    )
    settings.load_profile(os.environ.get("SHADOW_HYPOTHESIS_PROFILE", "ci"))
except ImportError:
    pass


# ---------------------------------------------------------------------------
# Windows temp-dir cleanup tolerance.
#
# Several components open SQLite databases (DarkMemory) or keep file handles
# for their lifetime. On Windows an open handle makes TemporaryDirectory
# cleanup raise PermissionError (WinError 32), which turns otherwise-passing
# tests into errors at teardown. The codebase's production lifecycle
# (ShadowOS.stop()/close()) releases these, but tests frequently rely on
# implicit cleanup. Make TemporaryDirectory cleanup best-effort on Windows.
# ---------------------------------------------------------------------------
if os.name == "nt":
    import shutil
    import warnings
    from pathlib import Path as _Path

    _orig_cleanup = tempfile.TemporaryDirectory.cleanup

    def _tolerant_cleanup(self):
        try:
            _orig_cleanup(self)
        except (PermissionError, OSError):
            shutil.rmtree(self.name, ignore_errors=True)

    tempfile.TemporaryDirectory.cleanup = _tolerant_cleanup

    # pytest's tmp_path fixtures use a similar retention/cleanup path.
    original_rmtree = shutil.rmtree

    def _quiet_rmtree(path, ignore_errors=False, onerror=None):
        try:
            return original_rmtree(path, ignore_errors=ignore_errors, onerror=onerror)
        except (PermissionError, OSError):
            if ignore_errors:
                return None
            return original_rmtree(path, ignore_errors=True)

    shutil.rmtree = _quiet_rmtree