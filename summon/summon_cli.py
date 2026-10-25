#!/usr/bin/env python3
"""
Summon CLI — talk to your Digital Shadow daemon.

Examples:
    summon "What was I working on yesterday?"        # auto-selects ability
    summon --shikigami oracle "Will tomorrow be productive?"
    summon --shikigami x_ray "Who's nearby?"
    summon --list                                    # show all 10 abilities
    summon --status
    summon --memory
    summon --insights                                # recent proactive insights
    summon --observers                               # observer task health
    summon --login                                   # interactive login (if auth enabled)
    summon --logout                                  # forget saved token

Auth (only when SHADOW_AUTH_ENABLED=true on the daemon):
    Token is loaded from $SHADOW_TOKEN if set, otherwise from
    ~/.config/shadow/token (written by --login).
"""

import argparse
import getpass
import json
import os
import sys
from pathlib import Path

import httpx

SHADOW_API = os.environ.get("SHADOW_API", "http://127.0.0.1:8765")
TOKEN_FILE = Path(os.environ.get("SHADOW_TOKEN_FILE",
                                  str(Path.home() / ".config" / "shadow" / "token")))


def _load_token() -> str:
    env = os.environ.get("SHADOW_TOKEN")
    if env:
        return env.strip()
    if TOKEN_FILE.exists():
        try:
            data = json.loads(TOKEN_FILE.read_text())
            return data.get("access_token", "") if isinstance(data, dict) else str(data)
        except Exception:
            return ""
    return ""


def _save_token(payload: dict) -> None:
    TOKEN_FILE.parent.mkdir(parents=True, exist_ok=True)
    TOKEN_FILE.write_text(json.dumps(payload, indent=2))
    try:
        TOKEN_FILE.chmod(0o600)
    except OSError:
        pass


def _auth_headers() -> dict:
    tok = _load_token()
    return {"Authorization": f"Bearer {tok}"} if tok else {}


def _post(path: str, payload: dict, timeout: float = 120.0) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.post(f"{SHADOW_API}{path}", json=payload, headers=_auth_headers())
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "Shadow daemon not running. Start it with ./start-shadowd.sh"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def _get(path: str, params=None, timeout: float = 10.0) -> dict:
    try:
        with httpx.Client(timeout=timeout) as client:
            r = client.get(f"{SHADOW_API}{path}", params=params or {}, headers=_auth_headers())
            r.raise_for_status()
            return r.json()
    except httpx.ConnectError:
        return {"error": "Shadow daemon not running"}
    except httpx.HTTPStatusError as e:
        return {"error": f"HTTP {e.response.status_code}: {e.response.text[:200]}"}
    except Exception as e:
        return {"error": f"{type(e).__name__}: {e}"}


def cmd_summon(query, shikigami):
    payload = {"question": query}
    if shikigami:
        payload["shikigami"] = shikigami
    result = _post("/shikigami/summon", payload)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    ability = result.get("ability", "?")
    auto = " (auto-selected)" if result.get("auto_selected") else ""
    print(f"━━━ {ability}{auto} ━━━")
    print(result.get("output", "<no output>"))
    return 0 if result.get("success") else 2


def cmd_health():
    """Print per-ability invocation stats from /abilities/health."""
    result = _get("/abilities/health")
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    abilities = result.get("abilities", [])
    if not abilities:
        print("No abilities registered.")
        return 0
    # Header.
    print(f"{'Ability':<16s} {'n':>5s} {'OK':>5s} {'rate':>6s} {'p50':>7s} {'p95':>7s}")
    print("-" * 50)
    for a in abilities:
        rate = f"{a['success_rate']*100:.0f}%" if a.get("success_rate") is not None else "  -"
        p50 = f"{a['p50_latency_s']:.3f}" if a.get("p50_latency_s") is not None else "    -"
        p95 = f"{a['p95_latency_s']:.3f}" if a.get("p95_latency_s") is not None else "    -"
        print(f"{a['type']:<16s} {a['count']:>5d} {a['successes']:>5d} {rate:>6s} {p50:>7s} {p95:>7s}")
    return 0


def cmd_council(query, abilities):
    """Fan a question out to multiple abilities via /shikigami/council."""
    payload = {"question": query}
    if abilities:
        payload["abilities"] = abilities
    result = _post("/shikigami/council", payload, timeout=180.0)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    print(result.get("verdict", "<no verdict>"))
    sel = result.get("selection")
    if sel == "relevance":
        print("  (abilities auto-selected for relevance to your question)")
    elif sel == "default":
        print("  (no clear signal — consulted the default council blend)")
    # Surface the woven narrative cleanly if present.
    narr = result.get("narrative") or {}
    if narr.get("contradictions"):
        print()
        print("━━━ CONTRADICTIONS ━━━")
        for c in narr["contradictions"]:
            abs_str = ", ".join(c.get("abilities", []))
            print(f"  ⚠ {c.get('description', '')}  ({abs_str})")
    if narr.get("recommendations"):
        print()
        print("━━━ RECOMMENDATIONS ━━━")
        for rec in narr["recommendations"]:
            print(f"  [{rec.get('priority', '?')}] {rec.get('title', '')}")
            rationale = rec.get("rationale", "")
            if rationale:
                print(f"      {rationale}")
    return 0


def cmd_ask(query):
    result = _post("/ask", {"question": query})
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    print(result.get("answer") or result.get("response") or json.dumps(result, indent=2))
    return 0


def cmd_list():
    result = _get("/shikigami")
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    abilities = result.get("shikigami", [])
    if not abilities:
        print("No abilities listed.")
        return 1
    print(f"{'NAME':<20} {'TYPE':<18} {'POWER'}")
    print("─" * 70)
    for a in abilities:
        emoji = a.get("emoji", "  ")
        print(f"{emoji} {a.get('name',''):<17} {a.get('type',''):<18} {a.get('power','')}")
    return 0


def cmd_status():
    result = _get("/status")
    print(json.dumps(result, indent=2))
    return 1 if "error" in result else 0


def cmd_memory(limit):
    result = _get("/memory", params={"limit": limit})
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    obs = result.get("observations") or result.get("memory") or []
    print(json.dumps(obs, indent=2))
    return 0


def cmd_insights():
    result = _get("/insights/recent", params={"limit": 20})
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    insights = result.get("insights", [])
    if not insights:
        print("No insights yet — the proactive scanner needs data + time.")
        return 0
    print(f"━━━ {len(insights)} recent insights ━━━")
    for ins in insights:
        severity = ins.get("severity", "?")
        kind = ins.get("kind", "?")
        marker = {"high": "⚠️ ", "medium": "❗ ", "low": "• "}.get(severity, "  ")
        print(f"{marker}[{kind}] {ins.get('explanation') or ins.get('reasoning') or ''}")
        for k, v in ins.items():
            if k in ("kind", "severity", "explanation", "reasoning", "emitted_at"):
                continue
            if isinstance(v, (str, int, float, bool)) or v is None:
                print(f"    {k}: {v}")
    return 0


def cmd_observers():
    result = _get("/observers/status")
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    obs = result.get("observers", {})
    print(f"{'OBSERVER':<14} {'STATUS':<10} {'NOTE'}")
    print("─" * 60)
    for name, info in obs.items():
        alive = info.get("alive", False)
        status = "alive" if alive else ("cancelled" if info.get("cancelled") else "dead")
        note = info.get("error") or ""
        print(f"{name:<14} {status:<10} {note}")
    return 0


def cmd_observe(activity_type, content, device):
    try:
        content_dict = json.loads(content) if content.startswith("{") else {"text": content}
    except json.JSONDecodeError:
        content_dict = {"text": content}
    payload = {"activity_type": activity_type, "content": content_dict}
    if device:
        payload["device"] = device
    result = _post("/observe", payload)
    if "error" in result:
        print(result["error"], file=sys.stderr)
        return 1
    print(json.dumps(result, indent=2))
    return 0


def cmd_login():
    email = input("Email (default admin@shadow.local): ").strip() or "admin@shadow.local"
    password = getpass.getpass("Password: ")
    if not password:
        print("password required", file=sys.stderr)
        return 1
    # Note: /auth/login doesn't require auth itself.
    try:
        with httpx.Client(timeout=10.0) as client:
            r = client.post(f"{SHADOW_API}/auth/login", json={"email": email, "password": password})
            if r.status_code == 401:
                print("invalid credentials", file=sys.stderr)
                return 1
            if r.status_code == 501:
                print("auth not enabled on the daemon (set SHADOW_AUTH_ENABLED=true)", file=sys.stderr)
                return 1
            r.raise_for_status()
            data = r.json()
    except httpx.ConnectError:
        print("Shadow daemon not running", file=sys.stderr)
        return 1
    except Exception as e:
        print(f"login error: {type(e).__name__}: {e}", file=sys.stderr)
        return 1
    _save_token(data)
    print(f"✓ Token saved to {TOKEN_FILE}")
    return 0


def cmd_logout():
    if TOKEN_FILE.exists():
        TOKEN_FILE.unlink()
        print(f"✓ Removed {TOKEN_FILE}")
    else:
        print("no saved token")
    return 0


def main():
    parser = argparse.ArgumentParser(
        description="Summon your Digital Shadow",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("query", nargs="?", help="Question/request for your shadow")
    parser.add_argument("--shikigami", "-k", help="Force a specific ability (oracle, x_ray, life_pulse, ...)")
    parser.add_argument("--ask", action="store_true", help="Use LLM-backed /ask instead of an ability")
    parser.add_argument("--council", action="store_true",
                        help="Fan the query out to multiple abilities via /shikigami/council")
    parser.add_argument("--ability", action="append", default=[],
                        help="Limit council to a specific ability (repeat to add more)")
    parser.add_argument("--list", "-l", action="store_true", help="List all abilities")
    parser.add_argument("--status", "-s", action="store_true", help="Show daemon status")
    parser.add_argument("--memory", "-m", type=int, nargs="?", const=10,
                        help="Show recent N observations (default 10)")
    parser.add_argument("--insights", action="store_true", help="Show recent proactive insights")
    parser.add_argument("--observers", action="store_true", help="Show per-observer task health")
    parser.add_argument("--observe", nargs=2, metavar=("TYPE", "CONTENT"),
                        help="Submit a new observation")
    parser.add_argument("--device", help="Device label for --observe")
    parser.add_argument("--login", action="store_true", help="Interactive login (saves JWT to token file)")
    parser.add_argument("--logout", action="store_true", help="Forget saved token")
    parser.add_argument("--health", action="store_true",
                        help="Show per-ability invocation stats (count, success rate, latency)")
    args = parser.parse_args()

    if args.login:
        return cmd_login()
    if args.logout:
        return cmd_logout()
    if args.health:
        return cmd_health()
    if args.list:
        return cmd_list()
    if args.status:
        return cmd_status()
    if args.insights:
        return cmd_insights()
    if args.observers:
        return cmd_observers()
    if args.memory is not None:
        return cmd_memory(args.memory)
    if args.observe:
        return cmd_observe(args.observe[0], args.observe[1], args.device)
    if not args.query:
        parser.print_help()
        return 1
    if args.ask:
        return cmd_ask(args.query)
    if args.council:
        return cmd_council(args.query, args.ability or None)
    return cmd_summon(args.query, args.shikigami)


if __name__ == "__main__":
    sys.exit(main())
