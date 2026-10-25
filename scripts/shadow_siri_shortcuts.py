#!/usr/bin/env python3
"""
Shadow Siri Shortcuts Integration v2

Generates Apple Shortcuts-compatible actions for ShadowOS voice commands
and device data push from iPhone/Apple Watch.

Shortcuts connect to ShadowOS via the Tailscale network URL.
Authentication is via X-API-Key header when SHADOW_OS_API_KEY is set.

Usage:
  python scripts/shadow_siri_shortcuts.py --generate
  python scripts/shadow_siri_shortcuts.py --generate --base-url https://your-url:8766
  python scripts/shadow_siri_shortcuts.py --generate --api-key your-api-key

Device Data Push Shortcuts:
  - iPhone HealthKit data → /os/device/iphone
  - Apple Watch health data → /os/device/watch
  - Location updates → /os/device/iphone (with location dict)
  - Calendar events → /os/device/iphone (with calendar dict)
"""

import argparse
import json
import sys
from pathlib import Path

SHORTCUTS_DIR = Path(__file__).parent.parent / "shortcuts"

SHADOWOS_BASE = "https://your-host.your-tailnet.ts.net:8766"

SHORTCUT_TEMPLATES = [
    {
        "name": "Shadow Summon",
        "phrase": "Shadow Summon",
        "description": "Voice-command ShadowOS to create agents, invoke abilities, or ask questions",
        "endpoint": "/os/summon/text",
        "method": "POST",
        "body_key": "transcript",
        "response_key": "spoken",
        "icon": "🧙",
    },
    {
        "name": "Shadow Create Agent",
        "phrase": "Shadow Create Agent",
        "description": "Create a new agent with a specific role",
        "endpoint": "/os/arise",
        "method": "POST",
        "body_key": "objective",
        "response_key": "name",
        "icon": "🤖",
    },
    {
        "name": "Shadow Status",
        "phrase": "Shadow Status",
        "description": "Get current ShadowOS system status",
        "endpoint": "/os/status",
        "method": "GET",
        "response_key": None,
        "icon": "📊",
    },
    {
        "name": "Shadow Predict",
        "phrase": "Shadow Predict",
        "description": "Use Oracle ability to predict outcomes",
        "endpoint": "/os/summon/text",
        "method": "POST",
        "body_key": "transcript",
        "response_key": "spoken",
        "icon": "🔮",
        "prefix": "Shadow, predict ",
    },
    {
        "name": "Shadow Council",
        "phrase": "Shadow Council",
        "description": "Get advice from the Shadow Council",
        "endpoint": "/os/summon/text",
        "method": "POST",
        "body_key": "transcript",
        "response_key": "spoken",
        "icon": "⚖️",
        "prefix": "Shadow, should I ",
    },
    {
        "name": "Shadow Agents",
        "phrase": "Shadow Agents",
        "description": "List all active agents",
        "endpoint": "/os/arise",
        "method": "GET",
        "response_key": None,
        "icon": "👥",
    },
]

HEALTHKIT_SHORTCUTS = [
    {
        "name": "Push iPhone Health",
        "phrase": "Push Health to Shadow",
        "description": "Send HealthKit data (heart rate, steps, sleep) to ShadowOS",
        "endpoint": "/os/device/iphone",
        "device_id": "iphone_siri",
        "device_type": "iphone",
        "data_template": {
            "health": {
                "heart_rate": 72,
                "hrv": 45,
                "steps": 8500,
                "sleep_hours": 7.5,
                "spo2": 98,
                "stress_level": "low",
                "active_energy": 350,
            },
            "battery": {
                "level": 85,
                "charging": False,
            },
            "notifications": [],
        },
        "icon": "❤️",
    },
    {
        "name": "Push Watch Health",
        "phrase": "Push Watch to Shadow",
        "description": "Send Apple Watch health data (HR, HRV, SpO2) to ShadowOS",
        "endpoint": "/os/device/watch",
        "device_id": "apple_watch_siri",
        "device_type": "watch",
        "data_template": {
            "heart_rate": 68,
            "hrv": 50,
            "steps": 3200,
            "sleep_hours": 7.0,
            "spo2": 99,
            "stress_level": "low",
            "active_energy": 180,
            "standing_hours": 6,
        },
        "icon": "⌚",
    },
    {
        "name": "Push Location",
        "phrase": "Push Location to Shadow",
        "description": "Send current location to ShadowOS for context awareness",
        "endpoint": "/os/device/iphone",
        "device_id": "iphone_siri",
        "device_type": "iphone",
        "data_template": {
            "location": {
                "latitude": 37.7749,
                "longitude": -122.4194,
                "accuracy": 10,
                "name": "Current Location",
            },
        },
        "icon": "📍",
    },
    {
        "name": "Push Battery",
        "phrase": "Push Battery to Shadow",
        "description": "Send iPhone battery level to ShadowOS",
        "endpoint": "/os/device/iphone",
        "device_id": "iphone_siri",
        "device_type": "iphone",
        "data_template": {
            "battery": {
                "level": 85,
                "charging": False,
            },
        },
        "icon": "🔋",
    },
]


def generate_shortcut_plist(shortcut, api_key=""):
    name = shortcut["name"]
    endpoint = shortcut["endpoint"]
    method = shortcut["method"]
    url = f"{SHADOWOS_BASE}{endpoint}"
    phrase = shortcut.get("prefix", "")

    actions = []

    if method == "POST" and shortcut.get("body_key"):
        actions.append({
            "WFWorkflowActionIdentifier": "is.workflow.actions.askforinput",
            "WFWorkflowActionParameters": {
                "WFAskForInputType": "Text",
                "WFQuestion": f"What should I {shortcut['body_key']}?",
            }
        })
        if phrase:
            actions.append({
                "WFWorkflowActionIdentifier": "is.workflow.actions.text.concatenate",
                "WFWorkflowActionParameters": {}
            })

        body = {shortcut["body_key"]: "ShortcutInput"}
        headers = {"Content-Type": "application/json"}
        if api_key:
            headers["X-API-Key"] = api_key
        actions.append({
            "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
            "WFWorkflowActionParameters": {
                "WFURL": url,
                "WFHTTPMethod": "POST",
                "WFHTTPBodyType": "JSON",
                "WFHTTPBody": json.dumps(body),
                "WFHTTPHeaders": headers,
            }
        })
    else:
        headers = {}
        if api_key:
            headers["X-API-Key"] = api_key
        actions.append({
            "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
            "WFWorkflowActionParameters": {
                "WFURL": url,
                "WFHTTPMethod": "GET",
                "WFHTTPHeaders": headers,
            }
        })

    if shortcut.get("response_key"):
        actions.append({
            "WFWorkflowActionIdentifier": "is.workflow.actions.getdictionaryvalue",
            "WFWorkflowActionParameters": {
                "WFDictionaryKey": shortcut["response_key"],
            }
        })

    actions.append({
        "WFWorkflowActionIdentifier": "is.workflow.actions.showresult",
        "WFWorkflowActionParameters": {}
    })

    return {
        "WFWorkflowName": name,
        "WFWorkflowIcon": {
            "WFWorkflowIconStartColor": 4294967295,
            "WFWorkflowIconGlyphNumber": 0,
        },
        "WFWorkflowActions": actions,
        "WFWorkflowImportQuestions": [],
        "WFWorkflowMinimumClientVersion": 411,
    }


def generate_health_shortcut(shortcut, api_key=""):
    name = shortcut["name"]
    url = f"{SHADOWOS_BASE}{shortcut['endpoint']}"
    data = shortcut["data_template"]
    device_id = shortcut["device_id"]

    headers = {"Content-Type": "application/json"}
    if api_key:
        headers["X-API-Key"] = api_key

    body = {"device_id": device_id, "data": data}

    actions = [
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.downloadurl",
            "WFWorkflowActionParameters": {
                "WFURL": url,
                "WFHTTPMethod": "POST",
                "WFHTTPBodyType": "JSON",
                "WFHTTPBody": json.dumps(body),
                "WFHTTPHeaders": headers,
            }
        },
        {
            "WFWorkflowActionIdentifier": "is.workflow.actions.showresult",
            "WFWorkflowActionParameters": {}
        }
    ]

    return {
        "WFWorkflowName": name,
        "WFWorkflowIcon": {
            "WFWorkflowIconStartColor": 65280,
            "WFWorkflowIconGlyphNumber": 0,
        },
        "WFWorkflowActions": actions,
        "WFWorkflowImportQuestions": [],
        "WFWorkflowMinimumClientVersion": 411,
    }


def generate_curl_script(shortcut, api_key=""):
    name = shortcut["name"]
    url = f"{SHADOWOS_BASE}{shortcut['endpoint']}"
    method = shortcut["method"]

    auth_header = f'  -H "X-API-Key: {api_key}" \\' if api_key else ""

    if method == "POST" and shortcut.get("body_key"):
        body_key = shortcut["body_key"]
        prefix = shortcut.get("prefix", "")
        data_payload = f'{{"{body_key}": "{prefix}$INPUT"}}'
        return f"""#!/bin/bash
# {name} — Send data to ShadowOS via curl
# Usage: {name.lower().replace(' ', '-')}.sh "your input"

echo "🎤 {name}"
echo "Enter your {body_key}:"
read -r INPUT

curl -s -X POST "{url}" \\
  -H "Content-Type: application/json" \\
{auth_header}  -d '{data_payload}' | python3 -m json.tool
"""
    else:
        auth_line = f'\n  -H "X-API-Key: {api_key}"' if api_key else ""
        return f"""#!/bin/bash
# {name} — Siri Shortcut via curl
curl -s "{url}"{auth_line} | python3 -m json.tool
"""


def generate_health_curl_script(shortcut, api_key=""):
    name = shortcut["name"]
    url = f"{SHADOWOS_BASE}{shortcut['endpoint']}"
    device_id = shortcut["device_id"]
    data = shortcut["data_template"]

    auth_header = f'  -H "X-API-Key: {api_key}" \\' if api_key else ""

    body = json.dumps({"device_id": device_id, "data": data}, indent=2)

    return f"""#!/bin/bash
# {name} — Push device data to ShadowOS
# This is a template. For real HealthKit data, use the Siri Shortcut
# on your iPhone which auto-collects health metrics.

curl -s -X POST "{url}" \\
  -H "Content-Type: application/json" \\
{auth_header}  -d '{body}' | python3 -m json.tool
"""


def generate_all(api_key=""):
    SHORTCUTS_DIR.mkdir(parents=True, exist_ok=True)

    for template in SHORTCUT_TEMPLATES:
        name = template["name"].replace(" ", "_").lower()
        shortcut_data = generate_shortcut_plist(template, api_key)
        shortcut_file = SHORTCUTS_DIR / f"{name}.shortcut.json"
        shortcut_file.write_text(json.dumps(shortcut_data, indent=2))
        print(f"  Generated: {shortcut_file}")

        curl_script = generate_curl_script(template, api_key)
        curl_file = SHORTCUTS_DIR / f"{name}.sh"
        curl_file.write_text(curl_script)
        curl_file.chmod(0o755)
        print(f"  Generated: {curl_file}")

    for template in HEALTHKIT_SHORTCUTS:
        name = template["name"].replace(" ", "_").lower()
        shortcut_data = generate_health_shortcut(template, api_key)
        shortcut_file = SHORTCUTS_DIR / f"{name}.shortcut.json"
        shortcut_file.write_text(json.dumps(shortcut_data, indent=2))
        print(f"  Generated: {shortcut_file}")

        curl_script = generate_health_curl_script(template, api_key)
        curl_file = SHORTCUTS_DIR / f"{name}.sh"
        curl_file.write_text(curl_script)
        curl_file.chmod(0o755)
        print(f"  Generated: {curl_file}")

    index = {
        "shortcuts": [
            {
                "name": s["name"],
                "phrase": s.get("phrase", ""),
                "description": s["description"],
                "icon": s.get("icon", "🔮"),
                "endpoint": s["endpoint"],
                "files": [
                    f"{s['name'].replace(' ', '_').lower()}.shortcut.json",
                    f"{s['name'].replace(' ', '_').lower()}.sh",
                ],
            }
            for s in SHORTCUT_TEMPLATES
        ],
        "health_shortcuts": [
            {
                "name": s["name"],
                "phrase": s.get("phrase", ""),
                "description": s["description"],
                "icon": s.get("icon", "📡"),
                "endpoint": s["endpoint"],
                "device_id": s["device_id"],
                "device_type": s["device_type"],
                "files": [
                    f"{s['name'].replace(' ', '_').lower()}.shortcut.json",
                    f"{s['name'].replace(' ', '_').lower()}.sh",
                ],
            }
            for s in HEALTHKIT_SHORTCUTS
        ],
        "base_url": SHADOWOS_BASE,
        "authentication": {
            "type": "X-API-Key",
            "header": "X-API-Key",
            "required": bool(api_key),
            "env_var": "SHADOW_OS_API_KEY",
        },
        "api_endpoints": {
            "device_register": {"method": "POST", "path": "/os/device/register", "description": "Register a new device"},
            "iphone_ingest": {"method": "POST", "path": "/os/device/iphone", "description": "Push iPhone data (health, location, battery, notifications)"},
            "watch_ingest": {"method": "POST", "path": "/os/device/watch", "description": "Push Apple Watch health data (HR, HRV, SpO2, steps)"},
            "laptop_ingest": {"method": "POST", "path": "/os/device/laptop", "description": "Push laptop data (active window, CPU, battery, WiFi)"},
            "device_context": {"method": "GET", "path": "/os/device/context", "description": "Get current unified device context"},
            "health_trends": {"method": "GET", "path": "/os/device/health-trends", "description": "Get health data trends"},
            "digital_wellbeing": {"method": "GET", "path": "/os/device/digital-wellbeing", "description": "Get digital wellbeing metrics"},
        },
        "data_schemas": {
            "iphone": {
                "health": {
                    "heart_rate": "number (bpm)",
                    "hrv": "number (ms, heart rate variability)",
                    "steps": "number (daily step count)",
                    "sleep_hours": "number (hours slept last night)",
                    "spo2": "number (blood oxygen %)",
                    "stress_level": "string (low/medium/high)",
                    "active_energy": "number (kcal burned)",
                },
                "location": {
                    "latitude": "number",
                    "longitude": "number",
                    "accuracy": "number (meters)",
                    "name": "string (place name)",
                },
                "battery": {
                    "level": "number (0-100)",
                    "charging": "boolean",
                },
                "notifications": [
                    {"app": "string", "category": "string"}
                ],
                "calendar": [
                    {"title": "string", "start": "ISO datetime", "end": "ISO datetime"}
                ],
            },
            "watch": {
                "heart_rate": "number (bpm)",
                "hrv": "number (ms)",
                "steps": "number",
                "sleep_hours": "number",
                "spo2": "number (blood oxygen %)",
                "stress_level": "string (low/medium/high)",
                "active_energy": "number (kcal)",
                "standing_hours": "number",
            },
            "laptop": {
                "active_window": "string (app name)",
                "screen_on": "boolean",
                "cpu_usage": "number (0-100)",
                "memory_usage": "number (0-100)",
                "battery_level": "number (0-100)",
                "battery_charging": "boolean",
                "wifi_ssid": "string",
                "media_playing": "string or null",
            },
        },
        "setup_instructions": [
            "1. Open Shortcuts app on iPhone",
            "2. Tap '+' to create a new shortcut",
            "3. Add 'Get Contents of URL' action",
            f"4. Set URL to: {SHADOWOS_BASE}/os/summon/text",
            "5. Set method to POST",
            "6. Add JSON body with 'transcript' key",
            "7. Add 'Show Result' action",
            "8. Name the shortcut 'Shadow Summon'",
            "9. Enable 'Show in Share Sheet' and 'Siri Shortcuts'",
            "10. Say 'Hey Siri, Shadow Summon' to invoke",
            "",
            "For HealthKit data push:",
            "11. Create an Automation in Shortcuts",
            "12. Set trigger: 'Health Sample' (heart rate, steps, etc.)",
            "13. Add 'Get Contents of URL' action",
            f"14. Set URL to: {SHADOWOS_BASE}/os/device/iphone",
            "15. Set method to POST",
            "16. Set body to JSON with device_id and health data",
            "17. Add X-API-Key header if authentication is enabled",
            "",
            "For Apple Watch data:",
            "18. Create a shortcut that reads Watch health metrics",
            f"19. Push to: {SHADOWOS_BASE}/os/device/watch",
            "20. Set device_id to 'apple_watch_siri'",
        ],
    }
    index_file = SHORTCUTS_DIR / "index.json"
    index_file.write_text(json.dumps(index, indent=2))
    print(f"\n  Generated: {index_file}")

    register_curl = f"""#!/bin/bash
# Register iPhone and Apple Watch devices with ShadowOS
# Run this once after setting up ShadowOS

echo "📱 Registering iPhone..."
curl -s -X POST "{SHADOWOS_BASE}/os/device/register" \\
  -H "Content-Type: application/json" \\
{'  -H "X-API-Key: ' + api_key + '" \\' + chr(10) + '  ' if api_key else '  '}-d '{{"device_id": "iphone_siri", "device_type": "iphone", "name": "Demo iPhone"}}'

echo ""
echo "⌚ Registering Apple Watch..."
curl -s -X POST "{SHADOWOS_BASE}/os/device/register" \\
  -H "Content-Type: application/json" \\
{'  -H "X-API-Key: ' + api_key + '" \\' + chr(10) + '  ' if api_key else '  '}-d '{{"device_id": "apple_watch_siri", "device_type": "watch", "name": "Demo Apple Watch"}}'

echo ""
echo "💻 Registering Laptop..."
curl -s -X POST "{SHADOWOS_BASE}/os/device/register" \\
  -H "Content-Type: application/json" \\
{'  -H "X-API-Key: ' + api_key + '" \\' + chr(10) + '  ' if api_key else '  '}-d '{{"device_id": "laptop_demo", "device_type": "laptop", "name": "Demo Laptop"}}'

echo ""
echo "✅ All devices registered!"
"""
    register_file = SHORTCUTS_DIR / "register_devices.sh"
    register_file.write_text(register_curl)
    register_file.chmod(0o755)
    print(f"  Generated: {register_file}")


def main():
    global SHADOWOS_BASE
    parser = argparse.ArgumentParser(description="ShadowOS Siri Shortcuts Generator v2")
    parser.add_argument("--generate", action="store_true", help="Generate shortcut files")
    parser.add_argument("--base-url", default=SHADOWOS_BASE, help="ShadowOS base URL")
    parser.add_argument("--api-key", default=os.environ.get("SHADOW_OS_API_KEY", ""), help="API key for authentication")
    args = parser.parse_args()

    if args.base_url:
        SHADOWOS_BASE = args.base_url

    if args.generate:
        print("Generating ShadowOS Siri Shortcuts v2...")
        print("  Including: HealthKit device data push shortcuts")
        if args.api_key:
            print("  Authentication: X-API-Key header enabled")
        generate_all(api_key=args.api_key)
        print("\nDone! Import the shortcuts into your iPhone Shortcuts app.")
        print("Run scripts/shortcuts/register_devices.sh to register devices first.")
    else:
        parser.print_help()


if __name__ == "__main__":
    import os
    main()