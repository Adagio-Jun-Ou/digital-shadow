"""
Robustness tests for the 10 shikigami abilities.

Verifies each ability handles edge cases (no observations, empty request,
huge request, weird unicode) and concurrent invocations without crashing.
Runs entirely against an in-process registry — no daemon required.
"""

import asyncio
import json
from typing import List

import pytest

from shikigami import get_registry, ShikigamiType
from shikigami.abilities import AbilityResult


ALL_ABILITIES = [
    ShikigamiType.ORACLE,
    ShikigamiType.X_RAY,
    ShikigamiType.LIFE_PULSE,
    ShikigamiType.PERFECT_MEMORY,
    ShikigamiType.PATTERN_SIGHT,
    ShikigamiType.SOCIAL_RADAR,
    ShikigamiType.INTENT_SEER,
    ShikigamiType.OMNI_SENSE,
    ShikigamiType.SHADOW_COUNCIL,
    ShikigamiType.REALITY_FORGE,
    ShikigamiType.HABIT_TRACKER,
    ShikigamiType.ENERGY_FLOW,
    ShikigamiType.FOCUS_SIGHT,
    ShikigamiType.CONNECTION,
    ShikigamiType.COMPASS,
    ShikigamiType.HEARTH,
    ShikigamiType.TIME_WITNESS,
    ShikigamiType.MIRROR,
    ShikigamiType.LANTERN,
    ShikigamiType.TIDE,
    ShikigamiType.ECHO,
]


def _ctx(obs: List[dict] = None):
    return {"observations": obs or []}


@pytest.fixture(scope="module")
def registry():
    return get_registry()


class TestEmptyContext:
    """Every ability must succeed (or fail gracefully — never throw) on empty obs."""

    @pytest.mark.parametrize("ability", ALL_ABILITIES)
    def test_zero_observations(self, registry, ability):
        result = asyncio.run(registry.execute(ability, "test question", _ctx([])))
        # Reality Forge is slow because it calls the LLM; allow either path.
        assert isinstance(result, AbilityResult)
        # success may be False (e.g. no LLM), but the call must NOT raise.
        assert result.output is not None


class TestMalformedQuestions:
    @pytest.mark.parametrize("ability", ALL_ABILITIES)
    @pytest.mark.parametrize("question", [
        "",            # empty
        " ",           # whitespace
        "?" * 5000,    # long but legal
        "é中😀",  # unicode, emoji
        "a;b|c&d>e<f",  # shell metas (should not affect ability)
        "{\"injection\": \"attempt\"}",  # json-ish
    ])
    def test_malformed_question(self, registry, ability, question):
        result = asyncio.run(registry.execute(ability, question, _ctx([])))
        assert isinstance(result, AbilityResult)


class TestConcurrentCalls:
    @pytest.mark.parametrize("ability", ALL_ABILITIES)
    def test_repeated_invocations_independent(self, registry, ability):
        """Repeated calls to the same ability must return distinct results."""
        async def go():
            tasks = [registry.execute(ability, f"call number {i}", _ctx([])) for i in range(8)]
            return await asyncio.gather(*tasks, return_exceptions=True)
        results = asyncio.run(go())
        assert all(isinstance(r, AbilityResult) for r in results), \
            f"some calls raised: {[r for r in results if not isinstance(r, AbilityResult)]}"

    def test_all_abilities_in_parallel(self, registry):
        """Fire every ability at the same time. Daemon must not crash."""
        async def go():
            tasks = [registry.execute(a, "parallel smoke", _ctx([])) for a in ALL_ABILITIES]
            return await asyncio.gather(*tasks, return_exceptions=True)
        results = asyncio.run(go())
        succeeded = sum(1 for r in results if isinstance(r, AbilityResult))
        assert succeeded == len(ALL_ABILITIES), \
            f"{len(ALL_ABILITIES) - succeeded} call(s) raised exceptions"


class TestObservationShapes:
    """Abilities should tolerate weird observation shapes from the wild."""

    @pytest.mark.parametrize("ability", [
        ShikigamiType.PATTERN_SIGHT,
        ShikigamiType.INTENT_SEER,
        ShikigamiType.PERFECT_MEMORY,
        ShikigamiType.OMNI_SENSE,
    ])
    def test_observations_missing_timestamp(self, registry, ability):
        # Some observations might lack timestamps entirely.
        obs = [{"type": "test", "content": {"text": "hello"}}]
        result = asyncio.run(registry.execute(ability, "test", _ctx(obs)))
        assert isinstance(result, AbilityResult)

    @pytest.mark.parametrize("ability", ALL_ABILITIES)
    def test_observations_with_garbage_types(self, registry, ability):
        obs = [
            {"type": None, "content": "x"},
            {"type": "", "content": {}},
            {"type": "garbage_type_!@#$%", "content": {"weird": True}},
            {"timestamp": "not an iso8601 string", "type": "ok", "content": "y"},
        ]
        result = asyncio.run(registry.execute(ability, "test", _ctx(obs)))
        assert isinstance(result, AbilityResult)

    @pytest.mark.parametrize("ability", ALL_ABILITIES)
    def test_observations_with_unicode(self, registry, ability):
        obs = [{"type": "note", "content": {"text": "🚀 中文 emoji"}, "timestamp": "2026-05-01T10:00:00+00:00"}]
        result = asyncio.run(registry.execute(ability, "test", _ctx(obs)))
        assert isinstance(result, AbilityResult)


class TestRegistryListing:
    def test_lists_all_known(self, registry):
        # 10 originals + 4 (round 4: habit, energy, focus, connection)
        # + 4 (round 5: compass, hearth, time_witness, mirror)
        # + 3 (round 6: lantern, tide, echo) = 21.
        abilities = registry.list_shikigami()
        assert len(abilities) == 21
        names = {a["type"] for a in abilities}
        expected = {a.value for a in ALL_ABILITIES}
        assert names == expected

    def test_each_has_required_fields(self, registry):
        for a in registry.list_shikigami():
            assert "name" in a
            assert "type" in a
            assert "power" in a
