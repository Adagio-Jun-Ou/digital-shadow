#!/usr/bin/env python3
"""
WiFi Scanner for Shadow AI
Real network scanning via nmcli (preferred) or iwlist (fallback), with graceful
no-op when neither is available.

Used by:
- X-Ray ability (location identification via SSID fingerprints)
- Omni Sense ability (environmental awareness)
"""

import asyncio
import logging
import re
from datetime import datetime, timezone
from typing import List, Dict, Optional
from dataclasses import dataclass

logger = logging.getLogger("shadowd.wifi")


@dataclass
class WiFiNetwork:
    ssid: str
    bssid: str  # MAC address
    rssi: int  # Signal strength in dBm (negative); 0 = unknown
    channel: int
    security: str  # WPA2, WPA3, Open, etc.


# nmcli outputs SIGNAL as a 0-100 percentage. Approximate dBm = signal/2 - 100,
# matching the linear mapping NetworkManager uses internally.
def _signal_pct_to_dbm(pct: int) -> int:
    if pct <= 0:
        return -100
    if pct >= 100:
        return -50
    return pct // 2 - 100


_NMCLI_FIELDS = "SSID,BSSID,SIGNAL,CHAN,SECURITY"
_IWLIST_CELL_RE = re.compile(
    r"Cell\s+\d+.*?Address:\s*([0-9A-Fa-f:]{17}).*?Channel:(\d+).*?"
    r"Signal level=(-?\d+).*?Encryption key:(on|off).*?ESSID:\"([^\"]*)\"",
    re.DOTALL,
)


class WiFiScanner:
    """WiFi scanner. Uses nmcli, falls back to iwlist, then to a safe no-op."""

    def __init__(self):
        self.known_networks: Dict[str, Dict] = {}
        self.scan_history: List[Dict] = []
        self.locations: Dict[str, List[str]] = {
            "home": ["MyHomeWiFi", "Home_5G"],
            "work": ["Office_2.4G", "CorpWiFi"],
            "coffee_shop": ["Starbucks_WiFi", "CoffeeNet"],
        }
        self._backend: Optional[str] = None  # cached after first scan

    async def _run(self, *args: str, timeout: float = 8.0) -> tuple[int, str]:
        try:
            proc = await asyncio.create_subprocess_exec(
                *args,
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

    async def _scan_nmcli(self) -> List[WiFiNetwork]:
        # Force a rescan so we get fresh APs rather than NM's cached list.
        await self._run("nmcli", "dev", "wifi", "rescan", timeout=5)
        rc, out = await self._run(
            "nmcli", "-t", "-f", _NMCLI_FIELDS, "dev", "wifi", "list", timeout=8,
        )
        if rc != 0 or not out.strip():
            return []
        networks: List[WiFiNetwork] = []
        for line in out.strip().splitlines():
            # nmcli's terse output escapes colons inside fields with `\:`. Split
            # carefully so BSSID's colons aren't mistaken for field separators.
            parts = re.split(r"(?<!\\):", line)
            parts = [p.replace(r"\:", ":") for p in parts]
            if len(parts) < 5:
                continue
            ssid, bssid, signal, chan, security = parts[:5]
            try:
                pct = int(signal)
                channel = int(chan) if chan else 0
            except ValueError:
                continue
            networks.append(WiFiNetwork(
                ssid=ssid,
                bssid=bssid.upper(),
                rssi=_signal_pct_to_dbm(pct),
                channel=channel,
                security=security or "Open",
            ))
        return networks

    async def _scan_iwlist(self) -> List[WiFiNetwork]:
        # Discover the wireless interface name.
        rc, ifaces = await self._run("iw", "dev", timeout=3)
        if rc != 0:
            return []
        m = re.search(r"Interface\s+(\S+)", ifaces)
        if not m:
            return []
        iface = m.group(1)
        rc, out = await self._run("sudo", "-n", "iwlist", iface, "scan", timeout=10)
        if rc != 0 or not out.strip():
            return []
        networks: List[WiFiNetwork] = []
        for cm in _IWLIST_CELL_RE.finditer(out):
            bssid, chan, rssi, enc, ssid = cm.groups()
            try:
                networks.append(WiFiNetwork(
                    ssid=ssid,
                    bssid=bssid.upper(),
                    rssi=int(rssi),
                    channel=int(chan),
                    security="WPA2" if enc == "on" else "Open",
                ))
            except ValueError:
                continue
        return networks

    async def scan(self) -> List[WiFiNetwork]:
        """Scan for nearby WiFi networks via the first available backend."""
        logger.info("WiFi scan")
        networks = await self._scan_nmcli()
        backend = "nmcli" if networks else None
        if not networks:
            networks = await self._scan_iwlist()
            backend = "iwlist" if networks else "unavailable"

        self._backend = backend
        self.scan_history.append({
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "networks": [{"ssid": n.ssid, "bssid": n.bssid, "rssi": n.rssi} for n in networks],
            "backend": backend,
        })
        logger.info("WiFi scan: %d network(s) via %s", len(networks), backend)
        return networks

    def detect_location(self, networks: List[WiFiNetwork]) -> str:
        """Detect location from WiFi signatures (majority match wins)."""
        detected_ssids = {n.ssid for n in networks}
        best, best_matches = "unknown", 0
        for location, signatures in self.locations.items():
            matches = sum(1 for sig in signatures if sig in detected_ssids)
            if signatures and matches >= len(signatures) / 2 and matches > best_matches:
                best, best_matches = location, matches
        if best != "unknown":
            logger.info("Detected location: %s (%d matches)", best, best_matches)
        return best

    def register_location(self, name: str, network_ssids: List[str]):
        self.locations[name] = network_ssids
        logger.info("Registered location: %s with %d networks", name, len(network_ssids))

    def register_known_network(self, bssid: str, ssid: str, location: str):
        self.known_networks[bssid.upper()] = {
            "bssid": bssid.upper(),
            "ssid": ssid,
            "location": location,
            "registered": datetime.now(timezone.utc).isoformat(),
        }

    def get_scan_history(self, limit: int = 10) -> List[Dict]:
        return self.scan_history[-limit:]

    def is_home(self, networks: List[WiFiNetwork]) -> bool:
        return self.detect_location(networks) == "home"

    def is_work(self, networks: List[WiFiNetwork]) -> bool:
        return self.detect_location(networks) == "work"


if __name__ == "__main__":
    async def _demo():
        scanner = WiFiScanner()
        nets = await scanner.scan()
        print(f"Found {len(nets)} network(s) via {scanner._backend}")
        for n in nets[:10]:
            print(f"  {n.bssid}  ch{n.channel:>3} {n.rssi:>4}dBm  {n.security:<10}  {n.ssid!r}")
        print(f"Location: {scanner.detect_location(nets)}")

    asyncio.run(_demo())
