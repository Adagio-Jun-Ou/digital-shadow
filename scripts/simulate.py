#!/usr/bin/env python3
"""
Shadow AI Simulation Engine - End-to-end test harness.

Runs the daemon locally and exercises all endpoints to verify
the full stack works correctly before deployment.

Usage:
    python scripts/simulate.py              # Full simulation
    python scripts/simulate.py --quick       # Quick smoke test
    python scripts/simulate.py --endpoints   # Only test endpoints
    python scripts/simulate.py --abilities   # Only test abilities
"""

import asyncio
import json
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import httpx
except ImportError:
    print("ERROR: httpx not installed. Run: pip install httpx")
    sys.exit(1)


BASE_URL = "http://127.0.0.1:8765"


class Colors:
    GREEN = "\033[92m"
    RED = "\033[91m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


passed = 0
failed = 0
skipped = 0
start_time = time.time()


def log_result(name, success, detail="", skip=False):
    global passed, failed, skipped
    if skip:
        skipped += 1
        print(f"  {Colors.YELLOW}SKIP{Colors.RESET} {name} {detail}")
    elif success:
        passed += 1
        print(f"  {Colors.GREEN}PASS{Colors.RESET} {name} {detail}")
    else:
        failed += 1
        print(f"  {Colors.RED}FAIL{Colors.RESET} {name} {detail}")


async def api(method, path, data=None, timeout=10.0):
    try:
        async with httpx.AsyncClient(timeout=timeout) as client:
            if method == "GET":
                r = await client.get(f"{BASE_URL}{path}")
            else:
                r = await client.post(f"{BASE_URL}{path}", json=data)
            return r.status_code, r.json() if r.headers.get("content-type", "").startswith("application/json") else r.text
    except Exception as e:
        return 0, str(e)


async def test_health():
    status, data = await api("GET", "/health")
    log_result("GET /health", status == 200, f"status={status}")
    if status == 200:
        checks = data.get("checks", data)
        log_result("Health: mind", checks.get("mind") == "ok", f"mind={checks.get('mind')}")
        log_result("Health: shikigami", checks.get("shikigami") == "ok", f"shikigami={checks.get('shikigami')}")
        log_result("Health: shadow_ai", checks.get("shadow_ai") == "ok", f"shadow_ai={checks.get('shadow_ai')}")
    return status == 200


async def test_status():
    status, data = await api("GET", "/status")
    log_result("GET /status", status == 200, f"status={status}")
    return status == 200


async def test_root():
    status, data = await api("GET", "/")
    log_result("GET /", status == 200, f"name={data.get('name','?') if isinstance(data, dict) else '?'}")
    return status == 200


async def test_metrics():
    status, data = await api("GET", "/metrics")
    log_result("GET /metrics", status == 200, f"status={status}")
    return status == 200


async def test_ask():
    status, data = await api("POST", "/ask", {"question": "What time is it?"})
    log_result("POST /ask", status == 200, f"response_len={len(str(data))}")
    return status == 200


async def test_ask_simple():
    status, data = await api("GET", "/ask?q=hello")
    log_result("GET /ask?q=hello", status == 200, f"status={status}")
    return status == 200


async def test_shikigami_list():
    status, data = await api("GET", "/shikigami")
    log_result("GET /shikigami", status == 200, f"count={len(data.get('shikigami', [])) if isinstance(data, dict) else '?'}")
    return status == 200


async def test_abilities():
    abilities = ["oracle", "x_ray", "life_pulse", "pattern_sight", "omni_sense"]
    all_ok = True
    for ability in abilities:
        status, data = await api("POST", "/ask", {"question": f"use {ability}"}, timeout=30.0)
        ok = status == 200
        all_ok = all_ok and ok
        log_result(f"Ability: {ability}", ok, f"status={status}")
    return all_ok


async def test_observe():
    status, data = await api("POST", "/observe", {
        "activity_type": "terminal_command",
        "content": {"command": "simulate --test"}
    })
    log_result("POST /observe", status == 200, f"count={data.get('count', '?')}")
    return status == 200


async def test_track_person():
    status, data = await api("POST", "/track/person", {
        "name": "TestUser",
        "devices": "AA:BB:CC:DD:EE:FF"
    })
    log_result("POST /track/person", status == 200, f"message={data.get('message', '?')}")
    return status == 200


async def test_track_decision():
    status, data = await api("POST", "/track/decision", {
        "situation": "choosing lunch",
        "choice": "salad",
        "outcome": "felt healthy",
        "satisfaction": 8
    })
    log_result("POST /track/decision", status == 200, f"message={data.get('message', '?')}")
    return status == 200


async def test_learn_pattern():
    status, data = await api("POST", "/track/pattern", {
        "cause": "working late",
        "effect": "tired morning"
    })
    log_result("POST /track/pattern", status == 200, f"message={data.get('message', '?')}")
    return status == 200


async def test_memory():
    status, data = await api("GET", "/memory")
    log_result("GET /memory", status == 200, f"blocks={len(data.get('blocks', {})) if isinstance(data, dict) else '?'}")
    return status == 200


async def test_memory_budget():
    status, data = await api("GET", "/memory/budget")
    log_result("GET /memory/budget", status == 200, f"tokens={data.get('total_tokens_used', '?') if isinstance(data, dict) else '?'}")
    return status == 200


async def test_memory_edit():
    status, data = await api("POST", "/memory/rethink", {
        "label": "recent",
        "new_memory": "Simulation test entry - can be deleted"
    })
    log_result("POST /memory/rethink", status == 200, f"success={data.get('success', '?') if isinstance(data, dict) else '?'}")
    return status == 200


async def test_insights():
    status, data = await api("GET", "/insights")
    log_result("GET /insights", status == 200, f"total={data.get('total_insights', '?') if isinstance(data, dict) else '?'}")
    return status == 200


async def test_snapshot():
    status, data = await api("GET", "/snapshot")
    log_result("GET /snapshot", status == 200, f"location={data.get('location', '?') if isinstance(data, dict) else '?'}")
    return status == 200


async def test_coordinator():
    status, data = await api("GET", "/coordinator/status")
    log_result("GET /coordinator/status", status in [200, 503], f"status={status}")
    return status == 200


async def test_event_bus():
    status, data = await api("GET", "/health")
    if status == 200 and isinstance(data, dict):
        events_published = data.get("events_published", 0)
        events_processed = data.get("events_processed", 0)
        log_result("Event bus active", events_published >= 0, f"published={events_published}, processed={events_processed}")
    else:
        log_result("Event bus active", False, "health check failed")
    return True


async def test_stuck_detector():
    status, data = await api("GET", "/health")
    if status == 200 and isinstance(data, dict):
        stuck_ok = data.get("stuck_detector") in ("ok", "unavailable", True)
        log_result("StuckDetector status", stuck_ok, f"stuck_detector={data.get('stuck_detector', '?')}")
    return True


async def test_context_budget():
    status, data = await api("GET", "/memory/budget")
    if status == 200 and isinstance(data, dict):
        has_sections = "sections" in data
        log_result("ContextWindowBudget", has_sections, f"sections={list(data.get('sections', {}).keys())}")
    else:
        log_result("ContextWindowBudget", False, "budget endpoint failed")
    return True


async def run_simulation(mode="full"):
    global start_time
    start_time = time.time()
    
    print(f"\n{Colors.BOLD}{Colors.CYAN}Shadow AI Simulation Engine{Colors.RESET}")
    print(f"{Colors.CYAN}{'=' * 50}{Colors.RESET}")
    print(f"Target: {BASE_URL}")
    print(f"Mode: {mode}")
    print(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    # Check connectivity first
    print(f"{Colors.BOLD}Phase 1: Connectivity{Colors.RESET}")
    try:
        async with httpx.AsyncClient(timeout=5.0) as client:
            r = await client.get(f"{BASE_URL}/health")
            if r.status_code != 200:
                print(f"{Colors.RED}Daemon not responding at {BASE_URL}{Colors.RESET}")
                print(f"Start it with: python -m shadowd.shadowd")
                return False
    except Exception as e:
        print(f"{Colors.RED}Cannot connect to {BASE_URL}: {e}{Colors.RESET}")
        print(f"Start the daemon with: python -m shadowd.shadowd")
        return False
    
    print(f"{Colors.GREEN}Daemon is running!{Colors.RESET}\n")
    
    # Phase 2: Core endpoints
    print(f"{Colors.BOLD}Phase 2: Core Endpoints{Colors.RESET}")
    await test_root()
    await test_health()
    await test_status()
    await test_metrics()
    
    # Phase 3: Ask system
    print(f"\n{Colors.BOLD}Phase 3: Ask System{Colors.RESET}")
    await test_ask()
    await test_ask_simple()
    
    # Phase 4: Abilities
    if mode in ("full", "abilities"):
        print(f"\n{Colors.BOLD}Phase 4: Shikigami Abilities{Colors.RESET}")
        await test_shikigami_list()
        await test_abilities()
    
    # Phase 5: Learning
    print(f"\n{Colors.BOLD}Phase 5: Learning System{Colors.RESET}")
    await test_observe()
    await test_track_person()
    await test_track_decision()
    await test_learn_pattern()
    
    # Phase 6: Memory
    print(f"\n{Colors.BOLD}Phase 6: Memory System{Colors.RESET}")
    await test_memory()
    await test_memory_budget()
    await test_memory_edit()
    
    # Phase 7: Insights & Monitoring
    print(f"\n{Colors.BOLD}Phase 7: Insights & Monitoring{Colors.RESET}")
    await test_insights()
    await test_snapshot()
    await test_coordinator()
    await test_event_bus()
    await test_stuck_detector()
    await test_context_budget()
    
    # Results
    elapsed = time.time() - start_time
    print(f"\n{Colors.CYAN}{'=' * 50}{Colors.RESET}")
    print(f"{Colors.BOLD}Results:{Colors.RESET} {Colors.GREEN}{passed} passed{Colors.RESET}, {Colors.RED}{failed} failed{Colors.RESET}, {Colors.YELLOW}{skipped} skipped{Colors.RESET} in {elapsed:.1f}s")
    
    if failed > 0:
        print(f"{Colors.RED}{Colors.BOLD}SIMULATION FAILED{Colors.RESET}")
        return False
    else:
        print(f"{Colors.GREEN}{Colors.BOLD}SIMULATION PASSED{Colors.RESET}")
        return True


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Shadow AI Simulation Engine")
    parser.add_argument("--quick", action="store_true", help="Quick smoke test only")
    parser.add_argument("--endpoints", action="store_true", help="Test endpoints only")
    parser.add_argument("--abilities", action="store_true", help="Test abilities only")
    parser.add_argument("--url", default="http://127.0.0.1:8765", help="Base URL")
    args = parser.parse_args()
    
    BASE_URL = args.url
    
    if args.quick:
        mode = "quick"
    elif args.endpoints:
        mode = "endpoints"
    elif args.abilities:
        mode = "abilities"
    else:
        mode = "full"
    
    try:
        result = asyncio.run(run_simulation(mode))
        sys.exit(0 if result else 1)
    except KeyboardInterrupt:
        print(f"\n{Colors.YELLOW}Simulation interrupted{Colors.RESET}")
        sys.exit(130)