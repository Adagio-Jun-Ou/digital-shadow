"""
Native Linux sensors for Omni Sense fusion.

Adds audio activity, battery state, CPU load, and system uptime/idle to the
sensor pipeline so Omni Sense isn't limited to BT/WiFi presence. Each probe
is best-effort: missing hardware or backends just leave that field absent.
"""

from __future__ import annotations

import asyncio
import os
import re
import shutil
import subprocess
import time
from pathlib import Path
from typing import Dict, Optional


# ---------- Audio --------------------------------------------------------------

_PACTL_AVAILABLE: Optional[bool] = None


async def _check_pactl() -> bool:
    global _PACTL_AVAILABLE
    if _PACTL_AVAILABLE is not None:
        return _PACTL_AVAILABLE
    _PACTL_AVAILABLE = shutil.which("pactl") is not None
    return _PACTL_AVAILABLE


async def _run(cmd: list[str], timeout: float = 3.0) -> tuple[int, str]:
    try:
        proc = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        try:
            stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
            return proc.returncode or 0, stdout.decode(errors="replace")
        except asyncio.TimeoutError:
            proc.kill()
            await proc.communicate()
            return -1, ""
    except (OSError, FileNotFoundError):
        return -1, ""


async def audio_state() -> Dict:
    """Report whether the system is actively playing/recording audio.

    On Linux/PulseAudio/PipeWire: enumerate sink-inputs (something is playing)
    and source-outputs (something is listening). Counts > 0 mean active.
    """
    if not await _check_pactl():
        return {"backend": "unavailable"}
    rc, out = await _run(["pactl", "list", "short", "sink-inputs"])
    sink_inputs = [line for line in out.splitlines() if line.strip()] if rc == 0 else []
    rc, out = await _run(["pactl", "list", "short", "source-outputs"])
    source_outputs = [line for line in out.splitlines() if line.strip()] if rc == 0 else []

    # Default sink mute / volume snapshot for "is audio plausibly audible"
    muted = False
    volume_pct = None
    rc, out = await _run(["pactl", "get-sink-mute", "@DEFAULT_SINK@"])
    if rc == 0 and "yes" in out.lower():
        muted = True
    rc, out = await _run(["pactl", "get-sink-volume", "@DEFAULT_SINK@"])
    if rc == 0:
        m = re.search(r"(\d+)%", out)
        if m:
            volume_pct = int(m.group(1))

    state = "playing" if sink_inputs else "recording" if source_outputs else "quiet"
    return {
        "backend": "pactl",
        "state": state,
        "playing_streams": len(sink_inputs),
        "recording_streams": len(source_outputs),
        "default_sink_muted": muted,
        "default_sink_volume_pct": volume_pct,
    }


# ---------- Battery ------------------------------------------------------------

_BAT_BASE = Path("/sys/class/power_supply")


def battery_state() -> Dict:
    """Read battery state from /sys. Returns empty dict if no battery."""
    if not _BAT_BASE.exists():
        return {}
    for entry in _BAT_BASE.iterdir():
        name = entry.name
        # Take the first thing that looks like a battery.
        if not (name.startswith("BAT") or name.startswith("battery")):
            continue
        try:
            data: Dict[str, object] = {"name": name}
            for key in ("capacity", "status", "charge_now", "charge_full", "energy_now", "energy_full"):
                path = entry / key
                if path.exists():
                    value = path.read_text().strip()
                    if key == "capacity":
                        try:
                            data["capacity_pct"] = int(value)
                        except ValueError:
                            pass
                    elif key == "status":
                        data["status"] = value
                    else:
                        try:
                            data[key] = int(value)
                        except ValueError:
                            pass
            return data
        except OSError:
            continue
    return {}


# ---------- Load / CPU / Idle --------------------------------------------------

def load_state() -> Dict:
    """1/5/15-min load averages and CPU count."""
    try:
        with open("/proc/loadavg", "r") as f:
            parts = f.read().strip().split()
        return {
            "load_1m": float(parts[0]),
            "load_5m": float(parts[1]),
            "load_15m": float(parts[2]),
            "ncpu": os.cpu_count() or 1,
        }
    except (OSError, ValueError, IndexError):
        return {}


def uptime_state() -> Dict:
    """Uptime in seconds + system idle seconds (sum across CPUs)."""
    try:
        with open("/proc/uptime", "r") as f:
            up_s, idle_s = (float(x) for x in f.read().split()[:2])
        return {
            "uptime_s": round(up_s, 1),
            "idle_s_total": round(idle_s, 1),
            "uptime_h": round(up_s / 3600.0, 2),
        }
    except (OSError, ValueError):
        return {}


# ---------- Activity inference -------------------------------------------------

def infer_activity_level(load: Dict, audio: Dict, battery: Dict) -> str:
    """One-word qualitative summary suitable for Omni Sense's 'activity_level'."""
    load_1m = load.get("load_1m", 0.0)
    ncpu = load.get("ncpu", 1) or 1
    norm_load = load_1m / ncpu

    if audio.get("state") == "playing" and norm_load > 0.3:
        return "active_media"
    if norm_load > 0.7:
        return "heavy"
    if norm_load > 0.3:
        return "moderate"
    if audio.get("state") == "recording":
        return "listening"
    if battery.get("status") == "Charging":
        return "idle_charging"
    return "idle"


# ---------- Aggregate snapshot -------------------------------------------------

async def native_snapshot() -> Dict:
    """Single async call gathering all best-effort native sensor data."""
    audio_t = audio_state()  # async
    # The synchronous ones are cheap (/proc reads); call them inline.
    load = load_state()
    battery = battery_state()
    uptime = uptime_state()
    audio = await audio_t
    activity = infer_activity_level(load, audio, battery)
    return {
        "audio": audio,
        "battery": battery,
        "load": load,
        "uptime": uptime,
        "inferred_activity": activity,
        "captured_at": int(time.time()),
    }


if __name__ == "__main__":
    import json
    async def _demo():
        snap = await native_snapshot()
        print(json.dumps(snap, indent=2))
    asyncio.run(_demo())
