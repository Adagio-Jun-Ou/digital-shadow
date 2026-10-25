"""
End-to-end integration test: seed a realistic 14-day observation history,
hit every ability, assert each returns non-trivial useful output.

This is the canary that catches:
- abilities silently returning empty output
- abilities crashing on the standard observation mix
- regressions from changes to shared utilities (e.g. timestamp parsing,
  text extraction)
"""

import asyncio
from datetime import datetime, timezone, timedelta
from typing import List, Dict

import pytest

from shikigami import get_registry, ShikigamiType
from shikigami.abilities import AbilityResult


def _seed_observations(now: datetime, days: int = 14) -> List[Dict]:
    """Build a realistic mixed-activity observation history."""
    obs: List[Dict] = []
    for d in range(days):
        day = now - timedelta(days=d)
        weekday = day.weekday()
        # Morning home → coding routine (every weekday)
        if weekday < 5:
            obs.append({
                "type": "wifi", "content": {"ssid": "HomeWiFi"},
                "timestamp": day.replace(hour=7, minute=30, second=0, microsecond=0).isoformat(),
            })
            for hour, minute in [(9, 0), (9, 30), (10, 0), (10, 45), (11, 0)]:
                obs.append({
                    "type": "terminal_command",
                    "content": {"command": "git commit -m 'refactor database layer'"},
                    "timestamp": day.replace(hour=hour, minute=minute, second=0, microsecond=0).isoformat(),
                })
            obs.append({
                "type": "wifi", "content": {"ssid": "CorpWiFi"},
                "timestamp": day.replace(hour=12, minute=0, second=0, microsecond=0).isoformat(),
            })
            obs.append({
                "type": "calendar_event",
                "content": {"title": "Team standup", "location": "Office"},
                "timestamp": day.replace(hour=10, minute=0, second=0, microsecond=0).isoformat(),
            })
        # Afternoon notes (self-reflection)
        obs.append({
            "type": "note",
            "content": {"text": "I'm feeling motivated about the new feature. I'm learning a lot today."},
            "timestamp": day.replace(hour=14, minute=0, second=0, microsecond=0).isoformat(),
        })
        # Messages
        obs.append({
            "type": "message",
            "content": {"body": "Hey Sarah, great work on the launch! Let's catch up."},
            "timestamp": day.replace(hour=15, minute=0, second=0, microsecond=0).isoformat(),
        })
        # File changes
        for i in range(3):
            obs.append({
                "type": "file_change",
                "content": {"path": "/home/user/project/src/main.py", "change_type": "modified"},
                "timestamp": day.replace(hour=16, minute=i*10, second=0, microsecond=0).isoformat(),
            })
        # Web visits
        obs.append({
            "type": "web_visit",
            "content": {"url": "https://docs.python.org", "title": "Python docs"},
            "timestamp": day.replace(hour=11, minute=15, second=0, microsecond=0).isoformat(),
        })
    return obs


@pytest.fixture(scope="module")
def now():
    return datetime(2026, 5, 13, 12, 0, tzinfo=timezone.utc)


@pytest.fixture(scope="module")
def observations(now):
    return _seed_observations(now)


@pytest.fixture(scope="module")
def registry():
    r = get_registry()
    # Track Sarah so Social Radar/Connection have someone to score.
    r.track_person("Sarah", "AA:BB:CC:DD:EE:01", "friend")
    return r


# Every ability should return success=True and at least 80 chars of output
# when fed a realistic history.
@pytest.mark.parametrize("ability", list(ShikigamiType))
def test_ability_returns_useful_output(registry, observations, ability):
    result = asyncio.run(
        registry.execute(ability, "give me the overview", {"observations": observations})
    )
    assert isinstance(result, AbilityResult)
    assert result.success, f"{ability.value} reported success=False: {result.output[:200]}"
    assert len(result.output) >= 80, \
        f"{ability.value} output too short: {result.output!r}"
    # Output should mention the ability or have substantive content.
    assert "**" in result.output or "🌀" in result.output or len(result.output) > 200


def test_all_abilities_run_in_parallel(registry, observations):
    """All abilities firing concurrently must complete without exceptions."""
    async def go():
        tasks = [
            registry.execute(ab, "concurrent test", {"observations": observations})
            for ab in ShikigamiType
        ]
        return await asyncio.gather(*tasks, return_exceptions=True)
    results = asyncio.run(go())
    failed = [r for r in results if not isinstance(r, AbilityResult)]
    assert not failed, f"abilities raised exceptions: {failed[:3]}"
    assert all(r.success for r in results), \
        "some abilities reported success=False under concurrent load"


def test_abilities_count_matches_registry(registry):
    """Sanity: enum size matches the registered ability list."""
    listed = registry.list_shikigami()
    enum_count = len(list(ShikigamiType))
    assert len(listed) == enum_count, \
        f"registry has {len(listed)} but enum has {enum_count}"


@pytest.mark.timeout(60)
def test_data_field_is_serializable(registry, observations):
    """Every ability's `data` field must be JSON-serializable so it survives
    the daemon's response path. Runs all abilities concurrently so this stays
    fast even with BT/WiFi-touching abilities."""
    import json

    async def go():
        tasks = [
            registry.execute(ab, "serializability check",
                             {"observations": observations})
            for ab in ShikigamiType
        ]
        return await asyncio.gather(*tasks)

    results = asyncio.run(go())
    for ability, result in zip(ShikigamiType, results):
        try:
            json.dumps(result.data, default=str)
        except (TypeError, ValueError) as e:
            pytest.fail(f"{ability.value}.data is not JSON-serializable: {e}")
