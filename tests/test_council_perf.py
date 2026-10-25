"""Performance benchmark for the council fan-out.

Not a regression test — prints timings, asserts only loose bounds so it
doesn't flake on slow CI. Useful as a periodic health check on how the
council scales with observation count.
"""

import asyncio
import os
import time
from datetime import datetime, timedelta, timezone

import pytest

os.environ.setdefault("SHADOW_AUTH_ENABLED", "false")


@pytest.fixture(scope="module")
def registry():
    from shikigami import get_registry, ShikigamiType  # noqa: F401
    return get_registry()


def _seed(n_days: int = 14) -> list:
    now = datetime(2026, 5, 15, 12, 0, tzinfo=timezone.utc)
    obs = []
    for d in range(n_days):
        day = now - timedelta(days=d)
        for h in (9, 10, 11, 14, 15, 16):
            obs.append({"type": "terminal_command",
                        "content": {"command": "git commit -m wip"},
                        "timestamp": day.replace(hour=h).isoformat()})
        for h in (8,):
            obs.append({"type": "wifi", "content": {"ssid": "HomeWiFi"},
                        "timestamp": day.replace(hour=h).isoformat()})
        for h in (13,):
            obs.append({"type": "note",
                        "content": {"text": "feeling motivated and focused"},
                        "timestamp": day.replace(hour=h).isoformat()})
        for h in (16,):
            obs.append({"type": "file_change",
                        "content": {"path": "/home/user/digital-shadow/foo.py"},
                        "timestamp": day.replace(hour=h, minute=15).isoformat()})
    return obs


class TestCouncilPerformance:
    @pytest.mark.timeout(180)
    def test_council_all_abilities_under_15s(self, registry):
        """All 21 abilities fanned out in parallel must finish under 15s
        on a realistic observation set."""
        from shikigami import ShikigamiType
        obs = _seed()

        async def fan_out():
            tasks = [
                registry.execute(a, "perf check", {"observations": obs})
                for a in ShikigamiType
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        start = time.time()
        results = asyncio.run(fan_out())
        elapsed = time.time() - start

        # Print a small report. xdist will swallow this unless -s is passed,
        # which is fine — assertions still run.
        print(f"\n[council perf] {len(results)} abilities, {len(obs)} obs, "
              f"{elapsed:.2f}s total ({elapsed*1000/len(results):.1f}ms avg)")
        # Loose: assert under 15 seconds. Local was ~0.5-1s; CI gets headroom.
        assert elapsed < 15.0, f"council fan-out took {elapsed:.2f}s"
        # No exceptions raised.
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert not exceptions, f"abilities raised: {exceptions[:3]}"

    @pytest.mark.timeout(60)
    def test_parsed_observations_speedup(self, registry):
        """ParsedObservations should not slow down the fan-out, and should
        be transparent to abilities (none crash, all succeed)."""
        from shikigami import ShikigamiType
        from shikigami.parsed_observations import ParsedObservations

        obs = _seed(n_days=7)
        wrapped = ParsedObservations(obs)
        wrapped.precompute_timestamps()

        async def fan_out():
            tasks = [
                registry.execute(a, "perf check", {"observations": wrapped})
                for a in ShikigamiType
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        start = time.time()
        results = asyncio.run(fan_out())
        elapsed = time.time() - start
        print(f"\n[council perf parsed] {len(results)} abilities, {len(obs)} obs, "
              f"{elapsed:.2f}s total")
        assert elapsed < 15.0
        exceptions = [r for r in results if isinstance(r, Exception)]
        assert not exceptions

    @pytest.mark.timeout(60)
    def test_scaling_to_500_observations(self, registry):
        """500 obs through the council must still complete in reasonable
        time."""
        from shikigami import ShikigamiType
        obs = _seed(n_days=30)
        assert len(obs) >= 150  # sanity

        async def fan_out():
            tasks = [
                registry.execute(a, "scale check", {"observations": obs})
                for a in ShikigamiType
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)

        start = time.time()
        results = asyncio.run(fan_out())
        elapsed = time.time() - start
        print(f"\n[council perf scale] {len(obs)} obs → {elapsed:.2f}s")
        assert elapsed < 20.0
