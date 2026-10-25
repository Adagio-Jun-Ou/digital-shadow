#!/usr/bin/env python3
"""
Bluetooth Scanner for Shadow AI
Real BLE discovery via bluetoothctl, with graceful fallback when unavailable.

Used by:
- Life Pulse ability (person identification via device constellations)
- X-Ray ability (presence/proximity detection)
"""

import asyncio
import logging
import os
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("shadowd.bluetooth")

_DEVICE_LINE_RE = re.compile(r"^Device\s+([0-9A-Fa-f:]{17})\s+(.*)$")
_RSSI_RE = re.compile(r"RSSI:\s*(?:0x[0-9A-Fa-f]+\s*\()?(-?\d+)")


@dataclass
class BluetoothDevice:
    mac: str
    name: str
    rssi: int  # Signal strength in dBm; 0 means unknown
    device_type: str
    last_seen: datetime


class BluetoothScanner:
    """BLE scanner. Uses `bluetoothctl` when present; safe no-op otherwise."""

    def __init__(self):
        self.known_devices: Dict[str, Dict] = {}
        self.scan_results: List[Dict] = []
        self._backend_checked = False
        self._backend_available = False

    async def _check_backend(self) -> bool:
        if self._backend_checked:
            return self._backend_available
        self._backend_checked = True
        try:
            proc = await asyncio.create_subprocess_exec(
                "bluetoothctl", "--version",
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE,
            )
            try:
                await asyncio.wait_for(proc.communicate(), timeout=2)
                self._backend_available = proc.returncode == 0
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
        except (OSError, FileNotFoundError):
            self._backend_available = False
        if not self._backend_available:
            logger.info("bluetoothctl unavailable; scanner will return empty results")
        return self._backend_available

    async def _run(self, *args: str, timeout: float = 5.0) -> str:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            try:
                stdout, _ = await asyncio.wait_for(proc.communicate(), timeout=timeout)
                return stdout.decode(errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return ""
        except (OSError, FileNotFoundError):
            return ""

    async def scan(self, duration: int = 10) -> List[BluetoothDevice]:
        """
        Scan for nearby Bluetooth devices.

        Triggers active discovery via `bluetoothctl --timeout N scan on`, then
        enumerates discovered devices and queries each for RSSI. Falls back to
        already-cached devices when discovery is unavailable.
        """
        if not await self._check_backend():
            self.scan_results.append({
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "devices": [],
                "duration": duration,
                "backend": "unavailable",
            })
            return []

        # Active discovery for `duration` seconds. The --timeout flag returns
        # control once the scan window ends.
        logger.info("BLE discovery: %ds", duration)
        await self._run("bluetoothctl", "--timeout", str(int(duration)), "scan", "on",
                        timeout=duration + 5)

        # Enumerate everything bluez now knows about.
        listing = await self._run("bluetoothctl", "devices", timeout=5)
        devices: List[BluetoothDevice] = []
        now = datetime.now(timezone.utc)

        for line in listing.strip().splitlines():
            m = _DEVICE_LINE_RE.match(line.strip())
            if not m:
                continue
            mac, name = m.group(1).upper(), m.group(2).strip()

            # Pull RSSI per device. `bluetoothctl info` returns RSSI only for
            # devices currently advertising, which is exactly what we want.
            info = await self._run("bluetoothctl", "info", mac, timeout=3)
            rssi = 0
            device_type = "unknown"
            for info_line in info.splitlines():
                rm = _RSSI_RE.search(info_line)
                if rm:
                    rssi = int(rm.group(1))
                if "Icon:" in info_line:
                    device_type = info_line.split("Icon:", 1)[1].strip() or "unknown"

            # Skip stale entries with no signal — they're paired but not present.
            if rssi == 0 and "Connected: yes" not in info:
                continue

            devices.append(BluetoothDevice(
                mac=mac, name=name, rssi=rssi,
                device_type=device_type, last_seen=now,
            ))

        self.scan_results.append({
            "timestamp": now.isoformat(),
            "devices": [{"mac": d.mac, "name": d.name, "rssi": d.rssi} for d in devices],
            "duration": duration,
            "backend": "bluetoothctl",
        })
        logger.info("BLE discovery found %d device(s)", len(devices))
        return devices

    def register_known_device(self, mac: str, name: str, owner: Optional[str] = None):
        """Register a device for identification by Life Pulse / X-Ray."""
        self.known_devices[mac.upper()] = {
            "mac": mac.upper(),
            "name": name,
            "owner": owner,
            "registered": datetime.now(timezone.utc).isoformat(),
        }
        logger.info("Registered known device: %s (%s)", name, mac)

    def identify_present_devices(self, scan_results: List[BluetoothDevice]) -> List[Dict]:
        """Filter a scan to only registered (known) devices."""
        present = []
        now = datetime.now(timezone.utc).isoformat()
        for device in scan_results:
            mac = device.mac.upper()
            if mac in self.known_devices:
                known = self.known_devices[mac]
                present.append({
                    "mac": mac,
                    "name": known["name"],
                    "owner": known.get("owner"),
                    "rssi": device.rssi,
                    "distance_m": round(self.estimate_distance(device.rssi), 2),
                    "detected": now,
                })
        return present

    def estimate_distance(self, rssi: int) -> float:
        """Path-loss estimate in meters from RSSI (dBm). Returns -1 if unknown."""
        tx_power = -59  # calibrated reference power at 1m
        if rssi == 0:
            return -1.0
        ratio = rssi / tx_power
        if ratio < 1.0:
            return ratio ** 10
        return 0.89976 * (rssi / tx_power) ** 7.7095 + 0.111

    def get_scan_history(self, limit: int = 10) -> List[Dict]:
        return self.scan_results[-limit:]


if __name__ == "__main__":
    async def _demo():
        scanner = BluetoothScanner()
        scanner.register_known_device("AA:BB:CC:DD:EE:FF", "John's iPhone", "John")
        devices = await scanner.scan(duration=int(os.environ.get("SCAN_SECS", "8")))
        print(f"Found {len(devices)} device(s)")
        for d in devices:
            print(f"  {d.mac}  {d.name!r:30s}  RSSI={d.rssi}dBm  ~{scanner.estimate_distance(d.rssi):.1f}m")
        present = scanner.identify_present_devices(devices)
        print(f"Known present: {present}")

    asyncio.run(_demo())
