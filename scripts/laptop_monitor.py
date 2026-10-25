#!/usr/bin/env python3
"""
Shadow AI - Linux Laptop System Monitor

Collects system metrics from the Linux laptop and pushes them
to the ShadowOS DeviceContext engine every 30 seconds.

Metrics collected:
- Active window (app name, category)
- Screen lock state (on/off)
- CPU/memory/disk usage
- Battery level and charging state
- WiFi SSID
- Bluetooth devices nearby
- Browser tab count (via hyprctl or xdotool)
- Terminal sessions (via ps)
- Media playing (via playerctl or mpris)
- Active calendar events (via calcurse or Thunderbird)

Usage:
  python laptop_monitor.py [--url URL] [--interval SECONDS] [--api-key KEY]

Environment variables:
  SHADOW_OS_URL      - Base URL (default: http://localhost:8766)
  SHADOW_OS_API_KEY  - API key for authentication
  SHADOW_DEVICE_ID   - Device identifier (default: laptop_demo)
  SHADOW_INTERVAL    - Update interval in seconds (default: 30)
"""

import argparse
import json
import os
import subprocess
import sys
import time
import logging
import signal
from datetime import datetime, timezone
from urllib.request import Request, urlopen
from urllib.error import URLError

logger = logging.getLogger("shadow_ai.laptop_monitor")

_APP_CATEGORIES = {
    "code": "development", "vscode": "development", "vim": "development",
    "nvim": "development", "emacs": "development", "kate": "development",
    "jetbrains": "development", "intellij": "development", "pycharm": "development",
    "terminal": "development", "alacritty": "development", "kitty": "development",
    "gnome-terminal": "development", "konsole": "development",
    "chrome": "browsing", "firefox": "browsing", "safari": "browsing",
    "brave": "browsing", "chromium": "browsing",
    "slack": "communication", "discord": "communication", "telegram": "communication",
    "messages": "communication", "mail": "communication", "thunderbird": "communication",
    "spotify": "entertainment", "vlc": "entertainment", "mpv": "entertainment",
    "youtube": "entertainment", "netflix": "entertainment", "plex": "entertainment",
    "notion": "productivity", "obsidian": "productivity", "todoist": "productivity",
    "evernote": "productivity", "zotero": "productivity",
    "zoom": "communication", "teams": "communication", "meet": "communication",
    "figma": "productivity", "gimp": "productivity", "inkscape": "productivity",
}


def _run_cmd(cmd, timeout=5):
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout)
        return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError, OSError):
        return ""


def get_active_window():
    """Get the currently active window title and app."""
    # Try hyprctl first (Hyprland)
    result = _run_cmd(["hyprctl", "activewindow", "-j"])
    if result:
        try:
            data = json.loads(result)
            title = data.get("title", "")
            app_class = data.get("class", "")
            return title, app_class.lower()
        except (json.JSONDecodeError, KeyError):
            pass

    # Try xdotool (X11)
    result = _run_cmd(["xdotool", "getactivewindow", "getwindowname"])
    if result:
        app_name = result.split(" - ")[-1].strip().lower() if " - " in result else result.strip().lower()
        return result.strip(), app_name

    # Try swaymsg (Sway/i3)
    result = _run_cmd(["swaymsg", "-r", "-t", "get_tree"])
    if result:
        try:
            tree = json.loads(result)
            # Walk the tree to find focused window
            nodes = tree.get("nodes", [])
            for node in nodes:
                if node.get("focused"):
                    return node.get("name", ""), node.get("app_id", "").lower() or node.get("window_properties", {}).get("class", "").lower()
        except (json.JSONDecodeError, KeyError):
            pass

    return "", ""


def get_screen_locked():
    """Check if the screen is locked."""
    # Try loginctl
    result = _run_cmd(["loginctl", "show-session", "$(loginctl | grep $(whoami) | awk '{print $1}')", "-p", "IdleHint"])
    if "yes" in result.lower():
        return True

    # Try xssstate (X11)
    result = _run_cmd(["xssstate", "-s"])
    if "DISABLED" in result:
        return False
    if "ENABLED" in result:
        return True

    # Try hyprctl
    result = _run_cmd(["hyprctl", "activewindow"])
    if not result or "error" in result.lower():
        return True  # No active window = likely locked

    return False


def get_cpu_usage():
    """Get CPU usage percentage."""
    try:
        with open("/proc/stat", "r") as f:
            line = f.readline()
        fields = [int(x) for x in line.split()[1:]]
        idle = fields[3]
        total = sum(fields)
        return round((1 - idle / total) * 100, 1) if total > 0 else 0.0
    except (FileNotFoundError, ZeroDivisionError, ValueError):
        return None


def get_memory_usage():
    """Get memory usage percentage."""
    try:
        with open("/proc/meminfo", "r") as f:
            lines = f.readlines()
        info = {}
        for line in lines:
            parts = line.split()
            info[parts[0].rstrip(":")] = int(parts[1])
        total = info.get("MemTotal", 0)
        available = info.get("MemAvailable", 0)
        if total > 0:
            return round((1 - available / total) * 100, 1)
        return None
    except (FileNotFoundError, ZeroDivisionError, ValueError):
        return None


def get_disk_usage():
    """Get disk usage percentage for root filesystem."""
    result = _run_cmd(["df", "-h", "/"])
    if result:
        lines = result.split("\n")
        if len(lines) >= 2:
            parts = lines[1].split()
            if len(parts) >= 5:
                usage_str = parts[4].replace("%", "")
                try:
                    return float(usage_str)
                except ValueError:
                    pass
    return None


def get_battery():
    """Get battery level and charging state."""
    try:
        battery_dir = "/sys/class/power_supply/BAT0"
        if not os.path.exists(battery_dir):
            battery_dir = "/sys/class/power_supply/BAT1"
        if not os.path.exists(battery_dir):
            return None, False

        with open(os.path.join(battery_dir, "capacity"), "r") as f:
            level = int(f.read().strip())

        with open(os.path.join(battery_dir, "status"), "r") as f:
            status = f.read().strip()

        charging = status in ("Charging", "Full")
        return float(level), charging
    except (FileNotFoundError, ValueError):
        return None, False


def get_wifi_ssid():
    """Get the current WiFi SSID."""
    # Try nmcli
    result = _run_cmd(["nmcli", "-t", "-f", "active,ssid", "dev", "wifi"])
    if result:
        for line in result.split("\n"):
            if line.startswith("yes:"):
                return line.split(":", 1)[1]
    # Try iwgetid
    result = _run_cmd(["iwgetid", "-r"])
    if result:
        return result
    return ""


def get_bluetooth_devices():
    """Get list of connected Bluetooth devices."""
    result = _run_cmd(["bluetoothctl", "devices", "Connected"])
    if result:
        devices = []
        for line in result.split("\n"):
            if line.strip():
                parts = line.strip().split(" ", 2)
                if len(parts) >= 3:
                    devices.append(parts[2])
        return devices
    return []


def get_media_playing():
    """Check if media is playing and get title."""
    result = _run_cmd(["playerctl", "status"])
    is_playing = result.lower() == "playing" if result else False
    title = ""
    media_type = ""
    if is_playing:
        title = _run_cmd(["playerctl", "title"]) or ""
        artist = _run_cmd(["playerctl", "artist"]) or ""
        if artist:
            title = f"{title} - {artist}"
        media_type = _run_cmd(["playerctl", "-l"]) or "music"
    return is_playing, title, media_type


def get_terminal_sessions():
    """Count active terminal sessions."""
    result = _run_cmd(["ps", "aux"])
    if result:
        count = 0
        for line in result.split("\n"):
            if any(term in line.lower() for term in ["terminal", "bash", "zsh", "fish", "alacritty", "kitty"]):
                count += 1
        return max(0, count - 1)  # Subtract 1 for the ps command itself
    return 0


def collect_data():
    """Collect all laptop metrics."""
    title, app_class = get_active_window()
    app_category = _APP_CATEGORIES.get(app_class, "")
    if not app_category and app_class:
        for key, cat in _APP_CATEGORIES.items():
            if key in app_class:
                app_category = cat
                break
    cpu = get_cpu_usage()
    memory = get_memory_usage()
    disk = get_disk_usage()
    battery, charging = get_battery()
    ssid = get_wifi_ssid()
    bt_devices = get_bluetooth_devices()
    is_playing, media_title, media_type = get_media_playing()
    term_sessions = get_terminal_sessions()
    screen_locked = get_screen_locked()

    return {
        "active_window": title,
        "active_app": app_class,
        "active_app_category": app_category,
        "screen_on": not screen_locked,
        "cpu_percent": cpu,
        "memory_percent": memory,
        "disk_percent": disk,
        "battery_level": battery,
        "battery_charging": charging,
        "wifi_ssid": ssid,
        "connected_devices": len(bt_devices) + (1 if battery is not None else 0),
        "bluetooth_devices": bt_devices,
        "media_playing": is_playing,
        "media_title": media_title,
        "media_type": media_type,
        "terminal_sessions": term_sessions,
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


def push_data(base_url, api_key, device_id, data):
    """Push collected data to ShadowOS."""
    url = f"{base_url.rstrip('/')}/os/device/laptop"
    if not base_url.startswith(("http://", "https://")):
        logger.error("Invalid base URL scheme")
        return False
    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key
    payload = json.dumps({"device_id": device_id, "data": data}).encode("utf-8")
    req = Request(url, data=payload, headers=headers, method="POST")
    try:
        with urlopen(req, timeout=10) as resp:
            return resp.status == 200
    except URLError:
        logger.error("Failed to push data: URLError")
        return False


def main():
    parser = argparse.ArgumentParser(description="Shadow AI Laptop Monitor")
    parser.add_argument("--url", default=os.environ.get("SHADOW_OS_URL", "http://localhost:8766"))
    parser.add_argument("--api-key", default=os.environ.get("SHADOW_OS_API_KEY", ""))
    parser.add_argument("--device-id", default=os.environ.get("SHADOW_DEVICE_ID", "laptop_demo"))
    parser.add_argument("--interval", type=int, default=int(os.environ.get("SHADOW_INTERVAL", "30")))
    parser.add_argument("--verbose", "-v", action="store_true")
    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

    logger.info("Starting Shadow AI Laptop Monitor (device_id=%s, interval=%ds)", args.device_id, args.interval)
    logger.info("Pushing to %s/os/device/laptop", args.url)

    running = True

    def signal_handler(sig, frame):
        nonlocal running
        logger.info("Shutting down...")
        running = False

    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)

    while running:
        try:
            data = collect_data()
            success = push_data(args.url, args.api_key, args.device_id, data)
            if success:
                logger.debug("Data pushed successfully: app=%s cpu=%s%% mem=%s%%",
                             data.get("active_window", "?"),
                             data.get("cpu_percent", "?"),
                             data.get("memory_percent", "?"))
            else:
                logger.warning("Failed to push data")
        except (RuntimeError, ValueError, OSError) as e:
            logger.error("Error collecting data: %s", type(e).__name__)

        for _ in range(args.interval):
            if not running:
                break
            time.sleep(1)

    logger.info("Laptop monitor stopped")


if __name__ == "__main__":
    main()