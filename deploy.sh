#!/usr/bin/env bash
# ShadowAI Deployment Manager — start, stop, status, verify
set -euo pipefail

SHADOW_DIR="${SHADOW_DIR:-/home/user/digital-shadow}"
VENV_PYTHON="$SHADOW_DIR/venv/bin/python"

red() { echo -e "\033[31m$1\033[0m"; }
green() { echo -e "\033[32m$1\033[0m"; }
yellow() { echo -e "\033[33m$1\033[0m"; }

cmd="${1:-help}"
shift 2>/dev/null || true

case "$cmd" in
start)
    echo "=== Starting ShadowAI Services ==="
    if systemctl is-active --quiet shadowos 2>/dev/null; then
        yellow "shadowos already running on :8766"
    else
        sudo systemctl start shadowos && green "shadowos started on :8766" || red "shadowos failed to start"
    fi
    if systemctl is-active --quiet digital-shadow 2>/dev/null; then
        yellow "digital-shadow already running on :8765"
    else
        sudo systemctl start digital-shadow && green "digital-shadow started on :8765" || red "digital-shadow failed to start"
    fi
    ;;

stop)
    echo "=== Stopping ShadowAI Services ==="
    sudo systemctl stop shadowos 2>/dev/null && green "shadowos stopped" || yellow "shadowos was not running"
    sudo systemctl stop digital-shadow 2>/dev/null && green "digital-shadow stopped" || yellow "digital-shadow was not running"
    ;;

restart)
    "$0" stop && sleep 2 && "$0" start
    ;;

status)
    echo "=== ShadowAI Status ==="
    for svc in shadowos digital-shadow; do
        if systemctl is-active --quiet "$svc" 2>/dev/null; then
            green "  $svc: RUNNING"
        else
            red "  $svc: STOPPED"
        fi
    done

    # Quick health check
    for port in 8765 8766; do
        if curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null | grep -q "200"; then
            green "  port $port: HEALTHY (HTTP 200)"
        else
            yellow "  port $port: UNREACHABLE"
        fi
    done
    ;;

verify)
    echo "=== ShadowAI Deployment Verification ==="
    if [ ! -d "$SHADOW_DIR" ]; then red "ShadowAI directory not found: $SHADOW_DIR"; exit 1; fi
    cd "$SHADOW_DIR"

    # Check venv
    if [ -f "$VENV_PYTHON" ]; then green "venv: OK"; else red "venv: missing"; fi

    # Check imports
    if "$VENV_PYTHON" -c "from shadowd.cognition.shadow_os import ShadowOS" 2>/dev/null; then
        green "ShadowOS import: OK"
    else
        red "ShadowOS import: FAILED"
    fi

    # Check all engines
    "$VENV_PYTHON" -c "
from shadowd.cognition.shadow_os import ShadowOS
os = ShadowOS()
engines = ['synthesis_engine','self_mod','visualizer','profiler','intel_report',
           'anomaly_corr','module_bus','opt_daemon','module_registry','arbiter','evolutionary',
           'creative_dream','social_propagation','health_dashboard','anticipatory','resonance',
           'temporal_self','device_context','evolution','oracle','feed_loop']
fail = 0
for e in engines:
    ok = hasattr(os, e) and getattr(os, e) is not None
    if not ok:
        print(f'  {e}: FAIL')
        fail += 1
if fail == 0:
    print(f'  All {len(engines)} engines: OK')
else:
    print(f'  {fail}/{len(engines)} engines FAILED')
" 2>/dev/null

    # Check API liveness
    for port in 8765 8766; do
        resp=$(curl -s -o /dev/null -w "%{http_code}" "http://127.0.0.1:$port/health" 2>/dev/null || echo "000")
        if [ "$resp" = "200" ]; then
            green "port $port: HEALTHY (HTTP $resp)"
        elif [ "$resp" = "000" ]; then
            yellow "port $port: NOT LISTENING"
        else
            yellow "port $port: HTTP $resp"
        fi
    done

    # Run core tests
    echo ""
    echo "=== Running Core Tests ==="
    "$VENV_PYTHON" -m pytest \
        "$SHADOW_DIR/tests/test_8_engine_integration.py" \
        "$SHADOW_DIR/tests/test_cross_module_synthesis.py" \
        "$SHADOW_DIR/tests/test_self_modification.py" \
        "$SHADOW_DIR/tests/test_evolutionary_engine.py" \
        -q --tb=short 2>/dev/null && green "Core tests: PASSED" || red "Core tests: FAILED"
    ;;

enable)
    echo "=== Enabling ShadowAI at Boot ==="
    sudo systemctl enable shadowos && green "shadowos enabled at boot" || red "shadowos enable failed"
    sudo systemctl enable digital-shadow && green "digital-shadow enabled at boot" || red "digital-shadow enable failed"
    ;;

logs)
    journalctl -u shadowos -u digital-shadow -f --since "5 min ago" "$@"
    ;;

*)
    echo "ShadowAI Deployment Manager"
    echo ""
    echo "Usage: $0 {start|stop|restart|status|verify|enable|logs}"
    echo ""
    echo "  start    Start all ShadowAI services"
    echo "  stop     Stop all ShadowAI services"
    echo "  restart  Restart all services"
    echo "  status   Show service and health status"
    echo "  verify   Run full deployment verification"
    echo "  enable   Enable services at boot"
    echo "  logs     Tail service logs"
    ;;
esac
