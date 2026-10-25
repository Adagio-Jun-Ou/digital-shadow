#!/usr/bin/env python3
"""
LEGACY / v1 shikigami registry — kept for backwards compatibility with older
tests (test_simulation.py, test_session5_sprint15.py, test_phase8_hardening.py,
test_session5_sprint192_197.py). Production code paths import from
`shikigami.enhanced_abilities` via `shikigami.__init__`, which exposes the
canonical EnhancedShikigamiRegistry. Do not add new functionality here.
"""

import asyncio
import copy
import json
import os
import subprocess
import re
import hashlib
import threading
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from pathlib import Path
from enum import Enum
import logging

import httpx

logger = logging.getLogger("shadowd.shikigami")

_SAFE_FILENAME_RE = re.compile(r'^[a-zA-Z0-9_\-\.]{1,128}\.(py|txt|md|json|yaml|yml|sh|toml|cfg|ini|conf)$')
_SANITIZE_CONTENT_RE = re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = re.compile(r'[<>\x00-\x1f\x7f]')
_MAX_SCAN_RESULTS = 100
_MAX_REQUEST_LENGTH = 10000
_MAX_LLM_PROMPT_LENGTH = 50000

OLLAMA_API_KEY = os.environ.get("OLLAMA_API_KEY", "")
OLLAMA_API_BASE = os.environ.get("OLLAMA_API_BASE", "https://ollama.com/v1")

DATA_DIR = Path.home() / "digital-shadow" / "data"
PEOPLE_DB = DATA_DIR / "people.json"
DECISIONS_DB = DATA_DIR / "decisions.json"
PATTERNS_DB = DATA_DIR / "patterns.json"


def _sanitize_content(text: str, max_len: int = 10000) -> str:
    return _SANITIZE_CONTENT_RE.sub('', text.strip())[:max_len]


def _sanitize_strict(text: str, max_len: int = 200) -> str:
    return _SANITIZE_STRICT_RE.sub('', text.strip())[:max_len]


for p in [PEOPLE_DB, DECISIONS_DB, PATTERNS_DB]:
    p.parent.mkdir(parents=True, exist_ok=True, mode=0o700)


class ShikigamiType(Enum):
    ORACLE = "oracle"
    X_RAY = "x_ray"
    LIFE_PULSE = "life_pulse"
    PERFECT_MEMORY = "perfect_memory"
    PATTERN_SIGHT = "pattern_sight"
    SOCIAL_RADAR = "social_radar"
    INTENT_SEER = "intent_seer"
    OMNI_SENSE = "omni_sense"
    SHADOW_COUNCIL = "shadow_council"
    REALITY_FORGE = "reality_forge"


@dataclass
class Result:
    success: bool
    output: str
    data: Dict = field(default_factory=dict)
    commands_run: List[str] = field(default_factory=list)
    files_created: List[str] = field(default_factory=list)


class ShikigamiRegistry:
    """10 Superhuman abilities grounded in actual data."""
    
    def __init__(self):
        self.people = self._load_json(PEOPLE_DB)
        self.decisions = self._load_json(DECISIONS_DB)
        self.patterns = self._load_json(PATTERNS_DB)
        self._http_client: Optional[httpx.AsyncClient] = None
        self._lock = threading.Lock()
        self._client_lock = asyncio.Lock()
    
    async def _get_client(self) -> httpx.AsyncClient:
        async with self._client_lock:
            if self._http_client is None or self._http_client.is_closed:
                self._http_client = httpx.AsyncClient(timeout=60.0)
            return self._http_client
    
    async def close(self):
        if self._http_client and not self._http_client.is_closed:
            await self._http_client.aclose()
            self._http_client = None
    
    def _load_json(self, path: Path):
        try:
            return json.loads(path.read_text()) if path.exists() else {}
        except (json.JSONDecodeError, OSError, ValueError, TypeError) as e:
            logger.warning("Failed to load %s: %s", path, type(e).__name__)
            return {}
    
    def _save_json(self, path: Path, data):
        try:
            tmp = path.with_suffix("." + hashlib.md5(str(path).encode()).hexdigest()[:8] + ".tmp")
            fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
            with os.fdopen(fd, 'w') as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(str(tmp), str(path))
        except (OSError, TypeError, ValueError, json.JSONDecodeError) as e:
            logger.error("Save failed: %s", type(e).__name__)
    
    async def _llm(self, prompt: str, max_tokens: int = 2000) -> str:
        prompt = _sanitize_content(prompt, _MAX_LLM_PROMPT_LENGTH)
        try:
            client = await self._get_client()
            r = await client.post(
                f"{OLLAMA_API_BASE}/chat/completions",
                headers={"Authorization": f"Bearer {OLLAMA_API_KEY}"},
                json={"model": "glm-5", "messages": [{"role": "user", "content": prompt}], "max_tokens": max_tokens}
            )
            return r.json()["choices"][0]["message"]["content"]
        except (ConnectionError, TimeoutError, ValueError, KeyError, AttributeError, OSError, RuntimeError) as e:
            return f"Error: {type(e).__name__}"
    
    ALLOWED_CMDS = {"bluetoothctl", "nmcli", "uptime", "echo"}

    async def _async_cmd(self, cmd: str) -> Tuple[int, str, str]:
        """Async-safe command execution."""
        try:
            parts = cmd.split(None, 1)
            if not parts or parts[0] not in self.ALLOWED_CMDS:
                logger.warning("Blocked command: %s", cmd[:80].replace(chr(10), ' ').replace(chr(13), ' '))
                return -1, "", "command not allowed"
            proc = await asyncio.create_subprocess_exec(
                *parts, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
            )
            try:
                stdout, stderr = await asyncio.wait_for(proc.communicate(), timeout=10)
                return proc.returncode, stdout.decode(errors="replace"), stderr.decode(errors="replace")
            except asyncio.TimeoutError:
                proc.kill()
                await proc.communicate()
                return -1, "", "timeout"
        except (OSError, subprocess.SubprocessError, asyncio.TimeoutError, ValueError) as e:
            logger.error("Command error: %s", type(e).__name__)
            return -1, "", "execution failed"

    def _cmd(self, cmd: str) -> Tuple[int, str, str]:
        """Synchronous command execution (for backwards compat)."""
        try:
            parts = cmd.split(None, 1)
            if not parts or parts[0] not in self.ALLOWED_CMDS:
                logger.warning("Blocked command: %s", cmd[:80].replace(chr(10), ' ').replace(chr(13), ' '))
                return -1, "", "command not allowed"
            r = subprocess.run(parts, capture_output=True, text=True, timeout=10)
            return r.returncode, r.stdout, r.stderr
        except subprocess.TimeoutExpired:
            return -1, "", "timeout"
        except (OSError, subprocess.SubprocessError, ValueError) as e:
            logger.error("Command error: %s", type(e).__name__)
            return -1, "", "execution failed"
    
    async def _scan_bt(self, duration: int = 6) -> List[Dict]:
        """Active BLE discovery via phone.BluetoothScanner."""
        try:
            from phone.bluetooth_scanner import BluetoothScanner
        except ImportError:
            return []
        scanner = BluetoothScanner()
        devices = await scanner.scan(duration=duration)
        results = [
            {"mac": d.mac, "name": _sanitize_strict(d.name, 100), "rssi": d.rssi}
            for d in devices
        ]
        return results[:_MAX_SCAN_RESULTS]

    async def _scan_wifi(self) -> List[Dict]:
        """WiFi scan via phone.WiFiScanner."""
        try:
            from phone.wifi_scanner import WiFiScanner
        except ImportError:
            return []
        scanner = WiFiScanner()
        networks = await scanner.scan()
        results = [
            {"ssid": _sanitize_strict(n.ssid, 100), "signal": str(n.rssi)[:10]}
            for n in networks
        ]
        return results[:_MAX_SCAN_RESULTS]

    # ═════════════════════════════════════════════════════════════════
    # 1. ORACLE - SPECIFIC PRECOGNITION
    # ═════════════════════════════════════════════════════════════════
    
    async def _oracle(self, request: str, context: Dict) -> Result:
        obs = context.get("observations", [])
        now = datetime.now(timezone.utc)
        
        device_count = len(await self._scan_bt())
        recent = obs[-10:] if obs else []
        activity_types = [o.get("type", "unknown") for o in recent]
        
        prediction = await self._llm(f"""You are Oracle. Make ONE specific prediction based on ACTUAL data.

CURRENT TIME: {now.strftime("%A %I:%M %p")}
DEVICES NEARBY: {device_count}
ACTIVITY: {activity_types[-5:] if activity_types else "None"}
RECENT EVENTS: {json.dumps(recent[-3:], indent=2) if recent else "None"}

Question: {request}

Answer with:
1. ONE specific event (exact, not vague)
2. TIME estimate
3. CONFIDENCE percentage (based on data certainty)
4. ONE action to take

Be concise (max 100 words). If unsure, admit uncertainty. Use real data only.""")

        output = f"🐍 **Oracle - PRECOGNITION**\n\n"
        output += f"**Time**: {now.strftime('%A %I:%M %p')}\n"
        output += f"**Devices**: {device_count}\n\n"
        output += prediction
        
        return Result(True, output, {"devices": device_count, "predictions": 1})

    # ═════════════════════════════════════════════════════════════════
    # 2. X-RAY - SEE THROUGH WALLS
    # ═════════════════════════════════════════════════════════════════
    
    async def _xray(self, request: str, context: Dict) -> Result:
        devices = await self._scan_bt()
        networks = await self._scan_wifi()
        
        with self._lock:
            people_snapshot = copy.deepcopy(self.people)
        
        present = []
        unknown = []
        for d in devices:
            found = False
            for name, data in people_snapshot.items():
                if d["mac"] in data.get("devices", []):
                    present.append({"name": name, "device": d["name"]})
                    found = True
            if not found:
                unknown.append(d)
        
        location = "unknown"
        for n in networks[:3]:
            ssid = n.get("ssid", "").lower()
            if re.search(r'(?:^|[\s_\-])(home|my)(?:$|[\s_\-])', ssid):
                location = "home"
            elif re.search(r'(?:^|[\s_\-])(office|corp)(?:$|[\s_\-])', ssid):
                location = "office"
            elif re.search(r'(?:^|[\s_\-])(starbucks|cafe)(?:$|[\s_\-])', ssid):
                location = "café"
        
        est_people = max(1, len(devices) // 3)
        
        output = f"👁️ **X-Ray Vision - SEE THROUGH WALLS**\n\n"
        output += f"**Location**: {location.title()}\n"
        output += f"**Network**: {networks[0]['ssid'] if networks else 'none'}\n"
        output += f"**Devices**: {len(devices)}\n"
        output += f"**People Estimate**: {est_people}\n\n"
        
        if present:
            output += f"**IDENTIFIED** ({len(present)}):\n"
            for p in present:
                output += f"  ✓ **{p['name']}** ({p['device']})\n"
        
        if unknown:
            output += f"\n**UNKNOWN** ({len(unknown)}):\n"
            for d in unknown[:5]:
                output += f"  • {d['name']} ({d['mac'][:8]}...)\n"
        
        output += "\n**To track**: 'Track [name] has device [mac]'\n"
        
        return Result(True, output, {"present": len(present), "unknown": len(unknown)})

    # ═════════════════════════════════════════════════════════════════
    # 3. LIFE PULSE - PERSON IDENTIFICATION
    # ═════════════════════════════════════════════════════════════════
    
    async def _life_pulse(self, request: str, context: Dict) -> Result:
        devices = await self._scan_bt()
        request_lower = request.lower()
        
        target = None
        for name in self.people.keys():
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', request_lower):
                target = name
                break
        
        present = []
        with self._lock:
            people_snapshot = copy.deepcopy(self.people)
        
        for name, data in people_snapshot.items():
            for d in devices:
                if d["mac"] in data.get("devices", []):
                    present.append({"name": name, "device": d})
        
        output = f"💓 **Life Pulse - PRESENCE DETECTION**\n\n"
        output += f"**Scanned**: {len(devices)} devices\n\n"
        
        if target:
            found = any(p["name"] == target for p in present)
            if found:
                output += f"**{target} IS HERE** ✓\n\n"
                for p in present:
                    if p["name"] == target:
                        output += f"Device: {p['device']['name']}\n"
                        if target in people_snapshot:
                            patterns = people_snapshot[target].get("patterns", {})
                            if patterns:
                                output += f"Patterns: {json.dumps(patterns)}\n"
            else:
                output += f"**{target} NOT PRESENT** ✗\n\n"
                output += "**Last seen**: Unknown (no tracked visits)\n"
        else:
            if present:
                output += f"**PRESENT** ({len(present)}):\n"
                for p in present:
                    output += f"✓ **{p['name']}**\n  Device: {p['device']['name']}\n\n"
            else:
                output += "**No tracked people present**\n\n"
                output += "Track someone: 'Track [name] has device [MAC address]'\n"
        
        return Result(True, output, {"present": [p["name"] for p in present], "target": target})

    # ═════════════════════════════════════════════════════════════════
    # 4. PERFECT MEMORY - TOTAL RECALL
    # ═════════════════════════════════════════════════════════════════
    
    async def _perfect_memory(self, request: str, context: Dict) -> Result:
        obs = context.get("observations", [])
        request = _sanitize_content(request, _MAX_REQUEST_LENGTH)
        request_lower = request.lower()
        
        matches = []
        
        if any(re.search(r'\b' + re.escape(d) + r'\b', request_lower) for d in ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]):
            days = {"monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3, 
                   "friday": 4, "saturday": 5, "sunday": 6}
            for day, num in days.items():
                if re.search(r'\b' + re.escape(day) + r'\b', request_lower):
                    target = (datetime.now(timezone.utc) - timedelta(days=(datetime.now(timezone.utc).weekday() - num) % 7)).date()
                    matches = [o for o in obs if str(target) in o.get("timestamp", "")]
                    break
        
        if not matches:
            keywords = [w for w in request_lower.split() if len(w) > 3 and w not in ["what", "when", "where", "which"]]
            matches = [o for o in obs if any(k in json.dumps(o).lower() for k in keywords)]
        
        output = f"🧠 **Perfect Memory - TOTAL RECALL**\n\n"
        output += f"**Searched**: {len(obs)} observations\n"
        output += f"**Found**: {len(matches)} matches\n\n"
        
        if matches:
            output += "**RESULTS**:\n"
            for m in matches[-10:]:
                ts = m.get("timestamp", "?")[:19]
                typ = m.get("type", "?")
                content = json.dumps(m.get("content", m), indent=None)[:120]
                output += f"[{ts}] {typ}\n  {content}\n\n"
        else:
            output += "No matches found.\n\n"
            output += "**Try**: 'What did I do Tuesday?' or 'coding'\n"
        
        commitments = [o for o in obs if any(re.search(r'\b' + re.escape(w) + r'\b', json.dumps(o).lower()) for w in ["call", "promise", "tomorrow", "later"])]
        if commitments:
            output += f"\n**⚠️ {len(commitments)} potential commitments**\n"
        
        return Result(True, output, {"matches": len(matches)})

    # ═════════════════════════════════════════════════════════════════
    # 5. PATTERN SIGHT - CAUSATION DETECTION
    # ═════════════════════════════════════════════════════════════════
    
    async def _pattern_sight(self, request: str, context: Dict) -> Result:
        obs = context.get("observations", [])
        request = _sanitize_content(request, _MAX_REQUEST_LENGTH)
        with self._lock:
            patterns_snapshot = copy.deepcopy(self.patterns)
        
        by_hour = {}
        for o in obs:
            if "timestamp" in o:
                try:
                    t = datetime.fromisoformat(o["timestamp"])
                    by_hour.setdefault(t.hour, []).append(o["type"])
                except (ValueError, TypeError):
                    pass
        
        hour_counts = {h: len(v) for h, v in by_hour.items()}
        peak_hour = max(hour_counts.items(), key=lambda x: x[1]) if hour_counts else (0, 0)
        
        insights = await self._llm(f"""Analyze activity patterns for CAUSATION.

ACTIVITY BY HOUR: {json.dumps(hour_counts)}
PEAK HOUR: {peak_hour[0]:02d}:00 ({peak_hour[1]} events)
STORED PATTERNS: {json.dumps(patterns_snapshot)}

Question: {request}

Find:
1. CAUSATION: What causes what? (X → Y)
2. OPTIMAL TIME: Best hour to do {request.split()[0] if request else "activity"}
3. AVOID: Worst hour
4. PATTERN: Any stored pattern that matches

Be specific. Use data. Format:
- CAUSATION: [X causes Y]
- BEST TIME: [hour]
- WORST TIME: [hour]
- ADVICE: [one action]""")

        output = f"👁️‍🗨️ **Pattern Sight - CAUSATION DETECTION**\n\n"
        output += f"**Data**: {len(obs)} observations\n"
        output += f"**Peak**: {peak_hour[0]:02d}:00 ({peak_hour[1]} events)\n"
        output += f"**Stored Patterns**: {len(patterns_snapshot)}\n\n"
        output += insights
        
        return Result(True, output, {"peak_hour": peak_hour[0], "patterns": len(patterns_snapshot)})

    # ═════════════════════════════════════════════════════════════════
    # 6. SOCIAL RADAR - PEOPLE PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    async def _social_radar(self, request: str, context: Dict) -> Result:
        request = _sanitize_content(request, _MAX_REQUEST_LENGTH)
        request_lower = request.lower()
        target = None
        with self._lock:
            people_snapshot = copy.deepcopy(self.people)
            decisions_snapshot = copy.deepcopy(self.decisions) if isinstance(self.decisions, list) else []
        
        for name in people_snapshot.keys():
            if re.search(r'\b' + re.escape(name.lower()) + r'\b', request_lower):
                target = name
                break
        
        person_data = people_snapshot.get(target, {}) if target else {}
        past_choices = []
        for d in decisions_snapshot:
            if target and target.lower() in json.dumps(d).lower():
                past_choices.append(d)
        
        prediction = await self._llm(f"""Predict behavior for a person.

TARGET: {target or "general"}
TRACKED DATA: {json.dumps(person_data, indent=2)}
PAST INTERACTIONS: {len(past_choices)}

Question: {request}

Answer:
1. BASELINE: What's normal for this person type?
2. PREDICTION: ONE specific behavior
3. CONFIDENCE: [X]%
4. SIGNALS: 2-3 signs to watch
5. RESPONSE: What to do

Max 150 words. Be specific.""")

        output = f"📡 **Social Radar - PEOPLE PREDICTION**\n\n"
        if target:
            output += f"**Target**: {target}\n"
            output += f"**Data**: {json.dumps(person_data.get('patterns', {}))}\n\n"
        else:
            output += f"**Tracked People**: {len(people_snapshot)}\n\n"
        output += prediction
        
        return Result(True, output, {"target": target})

    # ═════════════════════════════════════════════════════════════════
    # 7. INTENT SEER - ACTION PREDICTION
    # ═════════════════════════════════════════════════════════════════
    
    async def _intent_seer(self, request: str, context: Dict) -> Result:
        obs = context.get("observations", [])
        
        sequences = []
        for i in range(min(len(obs) - 1, 10)):
            if obs[i].get("type") and obs[i+1].get("type"):
                sequences.append(f"{obs[i]['type']} → {obs[i+1]['type']}")
        
        last_types = [o.get("type", "?") for o in obs[-5:]]
        
        prediction = await self._llm(f"""Predict the NEXT action from sequence.

RECENT ACTIONS: {last_types}
SEQUENCES: {sequences[-10:] if sequences else "None"}

Question: {request}

Predict:
1. NEXT: One specific action (not vague)
2. WHEN: Timeframe
3. CONFIDENCE: [X]%
4. PREPARE: What to do now

Max 100 words. Be specific.""")

        output = f"🎯 **Intent Seer - ACTION PREDICTION**\n\n"
        output += f"**Recent**: {' → '.join(last_types)}\n"
        output += f"**Sequences**: {len(sequences)}\n\n"
        output += prediction
        
        return Result(True, output, {"sequences": len(sequences)})

    # ═════════════════════════════════════════════════════════════════
    # 8. OMNI SENSE - ENVIRONMENTAL AWARENESS
    # ═════════════════════════════════════════════════════════════════
    
    async def _omni_sense(self, request: str, context: Dict) -> Result:
        devices = await self._scan_bt()
        networks = await self._scan_wifi()
        try:
            r = subprocess.run(["uptime"], capture_output=True, text=True, timeout=5)
            uptime = r.stdout.strip()[:50]
        except (OSError, subprocess.SubprocessError, subprocess.TimeoutExpired):
            uptime = "unknown"
        obs = context.get("observations", [])
        
        with self._lock:
            people_snapshot = copy.deepcopy(self.people)
        
        present = [name for name, data in people_snapshot.items() 
                   for d in devices if d["mac"] in data.get("devices", [])]
        
        location = "unknown"
        for n in networks[:5]:
            ssid = n.get("ssid", "").lower()
            if re.search(r'(?:^|[\s_\-])(home|my)(?:$|[\s_\-])', ssid):
                location = "home"
            elif re.search(r'(?:^|[\s_\-])(office|corp)(?:$|[\s_\-])', ssid):
                location = "office"
            elif re.search(r'(?:^|[\s_\-])(cafe|starbucks)(?:$|[\s_\-])', ssid):
                location = "café"
        
        activity = "quiet" if len(devices) < 4 else "moderate" if len(devices) < 10 else "busy"
        recent = obs[-5:] if obs else []
        recent_text = ", ".join([o.get("type", "?") for o in recent]) if recent else "None"
        
        output = f"🔮 **Omni Sense - ENVIRONMENTAL AWARENESS**\n\n"
        output += f"**Location**: {location.title()}\n"
        output += f"**Network**: {networks[0]['ssid'] if networks else 'None'}\n"
        output += f"**Devices**: {len(devices)} nearby\n"
        output += f"**People**: {len(present)} tracked ({', '.join(present) if present else 'none'})\n"
        output += f"**Activity**: {activity}\n"
        output += f"**Time**: {datetime.now(timezone.utc).strftime('%A %I:%M %p')}\n"
        output += f"**Recent**: {recent_text}\n"
        output += f"**Uptime**: {uptime.strip()[:50]}\n\n"
        
        output += f"**Snapshot**: You are at {location}. "
        output += f"{len(present)} tracked person(s) nearby. "
        output += f"Last activity: {recent_text.split(',')[-1].strip() if recent_text != 'None' else 'unknown'}. "
        output += f"Environment is {activity}."
        
        return Result(True, output, {"location": location, "people": len(present), "devices": len(devices)})

    # ═════════════════════════════════════════════════════════════════
    # 9. SHADOW COUNCIL - PAST WISDOM
    # ═════════════════════════════════════════════════════════════════
    
    async def _shadow_council(self, request: str, context: Dict) -> Result:
        request = _sanitize_content(request, _MAX_REQUEST_LENGTH)
        request_words = set(request.lower().split())
        
        similar = []
        with self._lock:
            decisions_snapshot = list(self.decisions) if isinstance(self.decisions, list) else []
        
        for d in decisions_snapshot:
            d_words = set(d.get("situation", "").lower().split())
            if request_words & d_words:
                similar.append(d)
        
        output = f"👥 **Shadow Council - PAST WISDOM**\n\n"
        output += f"**Tracked Decisions**: {len(decisions_snapshot)}\n"
        output += f"**Similar Situations**: {len(similar)}\n\n"
        
        if similar:
            output += "**PAST CHOICES**:\n"
            for s in similar[-5:]:
                ts = s.get("timestamp", "?")[:10]
                choice = s.get("choice", "?")
                outcome = s.get("outcome", "?")[:50]
                sat = s.get("satisfaction", "?")
                output += f"[{ts}] {choice}\n"
                output += f"  Result: {outcome}\n"
                output += f"  Satisfaction: {sat}/10\n\n"
            
            choices = {}
            for s in similar:
                c = s.get("choice", "?")
                choices.setdefault(c, []).append(s.get("satisfaction", 5))
            
            best = max(choices.items(), key=lambda x: sum(x[1])/len(x[1]))
            avg = sum(best[1])/len(best[1])
            output += f"**BEST CHOICE**: '{best[0]}' = {avg:.1f}/10 avg\n"
        else:
            output += "No similar past decisions tracked.\n\n"
            output += "**General Advice**: "
            advice = await self._llm(f"""Give brief advice on: {request}

Format:
1. CONSIDER: Main factors
2. AVOID: Common mistake
3. RECOMMEND: One action

Max 80 words.""")
            output += advice
        
        output += "\n\n**Track decisions**: 'I chose X, outcome Y, satisfaction Z/10'\n"
        
        return Result(True, output, {"similar": len(similar)})

    # ═════════════════════════════════════════════════════════════════
    # 10. REALITY FORGE - MANIFESTATION
    # ═════════════════════════════════════════════════════════════════
    
    async def _reality_forge(self, request: str, context: Dict) -> Result:
        request = _sanitize_content(request, _MAX_REQUEST_LENGTH)
        workspace = Path.home() / "digital-shadow" / "workspace"
        workspace.mkdir(parents=True, exist_ok=True)
        
        prompt = f"""Create: {request}

Output format:
---FILE: filename.ext---
<content here>
---END---

---CMD: command here---
---END---

Create WORKING code only."""
        
        generated = await self._llm(prompt, max_tokens=4000)
        
        files = re.findall(r'---FILE: (.+?)---\n(.*?)\n---END---', generated, re.DOTALL)
        created = []
        for fp, content in files:
            safe_name = fp.strip().lstrip("/.")
            if "/" in safe_name or "\\" in safe_name:
                logger.warning("Blocked path traversal: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            if not _SAFE_FILENAME_RE.match(safe_name):
                logger.warning("Blocked unsafe filename: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            if len(content) > 65536:
                logger.warning("Blocked oversized file: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            path = workspace / safe_name
            try:
                path.resolve().relative_to(workspace.resolve())
            except ValueError:
                logger.warning("Blocked path traversal: %s", fp.replace(chr(10), ' ').replace(chr(13), ''))
                continue
            path.parent.mkdir(parents=True, exist_ok=True)
            try:
                tmp = path.with_suffix("." + hashlib.md5(str(path).encode()).hexdigest()[:8] + ".tmp")
                fd = os.open(str(tmp), os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
                with os.fdopen(fd, 'w') as f:
                    f.write(content.strip())
                    f.flush()
                    os.fsync(f.fileno())
                os.replace(str(tmp), str(path))
            except OSError:
                logger.warning("Failed to write file: %s", safe_name)
                continue
            created.append(str(path))
        
        commands = re.findall(r'---CMD: (.+?)---', generated)
        executed = []
        for cmd in commands[:3]:
            logger.info("Reality Forge command (not auto-executed): %s", cmd.replace(chr(10), ' ').replace(chr(13), ''))
            executed.append({"cmd": cmd, "success": False, "output": "Command not auto-executed for safety. Run manually if trusted."})
        
        output = f"🔨 **Reality Forge - MANIFESTATION**\n\n"
        
        if created:
            output += f"**Created**:\n"
            for f in created:
                output += f"  📄 {f}\n"
        
        if executed:
            output += f"\n**Executed**:\n"
            for e in executed:
                mark = "✓" if e["success"] else "✗"
                output += f"  {mark} {e['cmd']}\n"
        
        if not created and not executed:
            output += generated[:500]
        
        if "localhost" in output or "127.0.0.1" in generated:
            output += "\n\n**✓ Service may be running. Check URLs above.**"
        
        return Result(True, output, {"files": len(created), "cmds": len(executed)}, 
                     commands_run=[e["cmd"] for e in executed], 
                     files_created=created)

    # ═════════════════════════════════════════════════════════════════
    # TRACKING
    # ═════════════════════════════════════════════════════════════════
    
    _MAX_PEOPLE = 1000
    _MAX_DEVICES_PER_PERSON = 50
    _MAX_DECISIONS = 10000

    def track_person(self, name: str, devices: str, relation: str = ""):
        name = _sanitize_strict(name, 200)
        relation = _sanitize_content(relation, 200)
        device_list = [_sanitize_strict(d.strip(), 50) for d in devices.replace(",", " ").split() if ":" in d or len(d) > 8]
        with self._lock:
            if name not in self.people:
                self.people[name] = {"devices": [], "patterns": {}, "relation": relation}
            self.people[name]["devices"].extend(device_list[:self._MAX_DEVICES_PER_PERSON])
            self.people[name]["devices"] = list(set(self.people[name]["devices"]))[:self._MAX_DEVICES_PER_PERSON]
            if len(self.people) > self._MAX_PEOPLE:
                oldest = list(self.people.keys())[:self._MAX_PEOPLE // 2]
                for k in oldest:
                    del self.people[k]
            self._save_json(PEOPLE_DB, self.people)
        return f"Tracking {name} with {len(device_list)} devices"
    
    def track_decision(self, situation: str, choice: str, outcome: str, satisfaction: int = 5):
        satisfaction = max(1, min(10, satisfaction))
        entry = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "situation": _sanitize_content(situation, 500),
            "choice": _sanitize_content(choice, 500),
            "outcome": _sanitize_content(outcome, 500),
            "satisfaction": satisfaction
        }
        with self._lock:
            if isinstance(self.decisions, dict):
                self.decisions = []
            self.decisions.append(entry)
            if len(self.decisions) > self._MAX_DECISIONS:
                self.decisions = self.decisions[-self._MAX_DECISIONS:]
            self._save_json(DECISIONS_DB, self.decisions)
        return f"Decision tracked: {choice}"
    
    # ═════════════════════════════════════════════════════════════════
    # PUBLIC API
    # ═════════════════════════════════════════════════════════════════
    
    async def execute(self, shikigami_type: ShikigamiType, request: str, context: Dict) -> Result:
        methods = {
            ShikigamiType.ORACLE: self._oracle,
            ShikigamiType.X_RAY: self._xray,
            ShikigamiType.LIFE_PULSE: self._life_pulse,
            ShikigamiType.PERFECT_MEMORY: self._perfect_memory,
            ShikigamiType.PATTERN_SIGHT: self._pattern_sight,
            ShikigamiType.SOCIAL_RADAR: self._social_radar,
            ShikigamiType.INTENT_SEER: self._intent_seer,
            ShikigamiType.OMNI_SENSE: self._omni_sense,
            ShikigamiType.SHADOW_COUNCIL: self._shadow_council,
            ShikigamiType.REALITY_FORGE: self._reality_forge,
        }
        
        method = methods.get(shikigami_type)
        if method:
            return await method(request, context)
        return Result(False, f"Unknown: {shikigami_type}")
    
    def list_shikigami(self) -> List[Dict]:
        return copy.deepcopy([
            {"name": "Oracle", "type": "oracle", "emoji": "🐍", "power": "Predicts events"},
            {"name": "X-Ray", "type": "x_ray", "emoji": "👁️", "power": "See through walls"},
            {"name": "Life Pulse", "type": "life_pulse", "emoji": "💓", "power": "Identify people"},
            {"name": "Perfect Memory", "type": "perfect_memory", "emoji": "🧠", "power": "Total recall"},
            {"name": "Pattern Sight", "type": "pattern_sight", "emoji": "👁️‍🗨️", "power": "Find causation"},
            {"name": "Social Radar", "type": "social_radar", "emoji": "📡", "power": "Predict behavior"},
            {"name": "Intent Seer", "type": "intent_seer", "emoji": "🎯", "power": "Predict actions"},
            {"name": "Omni Sense", "type": "omni_sense", "emoji": "🔮", "power": "Full awareness"},
            {"name": "Shadow Council", "type": "shadow_council", "emoji": "👥", "power": "Past wisdom"},
            {"name": "Reality Forge", "type": "reality_forge", "emoji": "🔨", "power": "Create & execute"},
        ])
    
    def auto_select(self, request: str) -> ShikigamiType:
        r = request.lower()
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["will", "predict", "happen", "probability"]): return ShikigamiType.ORACLE
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["is anyone", "is someone", "who's", "nearby", "present"]): return ShikigamiType.X_RAY
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["where is", "track"]): return ShikigamiType.LIFE_PULSE
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["remember", "when did", "what did", "recall"]): return ShikigamiType.PERFECT_MEMORY
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["pattern", "why", "trend", "correlation", "cause"]): return ShikigamiType.PATTERN_SIGHT
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["will they", "what will", "behavior", "lying"]): return ShikigamiType.SOCIAL_RADAR
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["going to", "about to", "planning", "intent"]): return ShikigamiType.INTENT_SEER
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["around me", "environment", "where am i", "what's happening"]): return ShikigamiType.OMNI_SENSE
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["should i", "past", "before", "chose"]): return ShikigamiType.SHADOW_COUNCIL
        if any(re.search(r'\b' + re.escape(x) + r'\b', r) for x in ["create", "build", "make", "set up"]): return ShikigamiType.REALITY_FORGE
        return ShikigamiType.OMNI_SENSE


_registry = None
_registry_lock = threading.Lock()

def get_registry() -> ShikigamiRegistry:
    global _registry
    if _registry is None:
        with _registry_lock:
            if _registry is None:
                _registry = ShikigamiRegistry()
    return _registry