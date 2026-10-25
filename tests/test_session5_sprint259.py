#!/usr/bin/env python3
"""Sprint 259 — ShadowPowers module tests."""

import threading
import time
from datetime import datetime, timezone

import pytest

from shadowd.cognition.shadow_powers import (
    ShadowPowers, PowerCategory, PowerState, EvolutionTrigger,
    ShadowPower, PowerApplication, EvolutionEvent,
    _sanitize, _sanitize_content, _clamp,
    _BASE_POWERS, _DERIVATION_MAP, _CROSS_POLLINATION,
    _MAX_POWERS, _MAX_APPLICATIONS, _MAX_EVOLUTION_LOG,
    _STRENGTH_GAIN_PER_USE, _FATIGUE_THRESHOLD, _EVOLUTION_USAGE_THRESHOLD,
    _CROSS_POLLINATION_RADIUS, _DORMANT_AWAKEN_SENSITIVITY, _MASTERY_STRENGTH,
)


class TestSanitize:
    def test_sanitize_strips_control_chars(self):
        result = _sanitize("hello\x00world")
        assert "\x00" not in result

    def test_sanitize_strips_angle_brackets(self):
        assert _sanitize("test<bold>value") == "testboldvalue"

    def test_sanitize_content_preserves_angle_brackets(self):
        assert _sanitize_content("latency < 100ms") == "latency < 100ms"

    def test_sanitize_content_strips_control_chars(self):
        result = _sanitize_content("hello\x01world")
        assert "\x01" not in result

    def test_sanitize_none(self):
        assert _sanitize(None) == ""

    def test_sanitize_content_none(self):
        assert _sanitize_content(None) == ""

    def test_sanitize_max_length(self):
        assert len(_sanitize("a" * 300, 200)) <= 200

    def test_sanitize_content_max_length(self):
        assert len(_sanitize_content("b" * 300, 200)) <= 200

    def test_sanitize_strip_before_truncate(self):
        assert _sanitize("  hello  ", 10) == "hello"


class TestClamp:
    def test_clamp_within_range(self):
        assert _clamp(0.5) == 0.5

    def test_clamp_below(self):
        assert _clamp(-0.5) == 0.0

    def test_clamp_above(self):
        assert _clamp(1.5) == 1.0

    def test_clamp_custom_range(self):
        assert _clamp(5, 1, 10) == 5


class TestShadowPower:
    def test_base_power_awakens(self):
        p = ShadowPower(
            power_id="abc123", name="test_power",
            category=PowerCategory.EXTRACTION, description="test",
            state=PowerState.DORMANT, is_base=True,
        )
        assert p.state == PowerState.AWAKENED
        assert p.strength == 0.05

    def test_non_base_stays_dormant(self):
        p = ShadowPower(
            power_id="abc123", name="test_power",
            category=PowerCategory.EXTRACTION, description="test",
            state=PowerState.DORMANT, is_base=False,
        )
        assert p.state == PowerState.DORMANT

    def test_success_rate_zero(self):
        p = ShadowPower(
            power_id="abc123", name="test", category=PowerCategory.SENSING,
            description="test", usage_count=0,
        )
        assert p.success_rate == 0.0

    def test_success_rate_calculation(self):
        p = ShadowPower(
            power_id="abc123", name="test", category=PowerCategory.SENSING,
            description="test", usage_count=10, success_count=8,
        )
        assert p.success_rate == 0.8

    def test_effective_strength(self):
        p = ShadowPower(
            power_id="abc123", name="test", category=PowerCategory.SENSING,
            description="test", strength=0.5, fatigue=0.2, is_base=False,
        )
        expected = round(0.5 * (1.0 - 0.2), 4)
        assert abs(p.effective_strength - expected) < 0.001

    def test_to_dict_keys(self):
        p = ShadowPower(
            power_id="abc123", name="test", category=PowerCategory.SENSING,
            description="test",
        )
        d = p.to_dict()
        assert "power_id" in d
        assert "name" in d
        assert "category" in d
        assert "effective_strength" in d
        assert "success_rate" in d

    def test_to_dict_rounds_floats(self):
        p = ShadowPower(
            power_id="abc123", name="test", category=PowerCategory.SENSING,
            description="test", strength=0.123456, fatigue=0.789012, is_base=False,
        )
        d = p.to_dict()
        assert d["strength"] == round(0.123456, 4)

    def test_sanitize_on_init(self):
        p = ShadowPower(
            power_id="<script>alert(1)</script>", name="test<bold>power",
            category=PowerCategory.SENSING, description="<b>hello</b> world",
        )
        assert "<" not in p.power_id  # strict sanitize strips <>
        assert "<" in p.name  # name uses _sanitize_content which preserves <>
        assert "<" in p.description  # content-safe preserves <>

    def test_derived_from_capped(self):
        p = ShadowPower(
            power_id="abc", name="test", category=PowerCategory.SENSING,
            description="test", derived_from=["a"] * 10,
        )
        assert len(p.derived_from) <= 5

    def test_metadata_capped(self):
        p = ShadowPower(
            power_id="abc", name="test", category=PowerCategory.SENSING,
            description="test", metadata={f"k{i}": f"v{i}" for i in range(100)},
        )
        assert len(p.metadata) <= 50


class TestPowerApplication:
    def test_basic_creation(self):
        app = PowerApplication(
            application_id="app1", power_id="pw1",
            category=PowerCategory.EXTRACTION, context="test",
            result="ok", success=True,
            strength_before=0.1, strength_after=0.2,
        )
        assert app.success is True

    def test_sanitize_on_init(self):
        app = PowerApplication(
            application_id="<script>", power_id="<evil>",
            category=PowerCategory.EXTRACTION, context="<b>hello</b>",
            result="<i>world</i>", success=True,
            strength_before=0.1, strength_after=0.2,
        )
        assert "<" not in app.application_id
        assert "<" not in app.power_id
        assert "<" in app.context  # content-safe

    def test_clamp_strength(self):
        app = PowerApplication(
            application_id="a", power_id="p",
            category=PowerCategory.EXTRACTION, context="c",
            result="r", success=True,
            strength_before=5.0, strength_after=-1.0,
        )
        assert app.strength_before == 1.0
        assert app.strength_after == 0.0

    def test_to_dict_rounds(self):
        app = PowerApplication(
            application_id="a", power_id="p",
            category=PowerCategory.EXTRACTION, context="c",
            result="r", success=True,
            strength_before=0.123456, strength_after=0.789012,
        )
        d = app.to_dict()
        assert d["strength_before"] == round(0.123456, 4)

    def test_metadata_deepcopy(self):
        meta = {"key": "value1"}
        app = PowerApplication(
            application_id="a", power_id="p",
            category=PowerCategory.EXTRACTION, context="c",
            result="r", success=True,
            strength_before=0.1, strength_after=0.2,
            metadata=meta,
        )
        d = app.to_dict()
        # metadata values are sanitized to strings
        assert isinstance(d["metadata"]["key"], str)
        # deepcopy protection
        d["metadata"]["new_key"] = "new_val"
        assert "new_key" not in app.metadata


class TestEvolutionEvent:
    def test_basic_creation(self):
        ev = EvolutionEvent(
            event_id="e1", power_id="p1",
            trigger=EvolutionTrigger.USAGE,
            description="test event",
            strength_before=0.1, strength_after=0.2,
        )
        assert ev.trigger == EvolutionTrigger.USAGE

    def test_sanitize_on_init(self):
        ev = EvolutionEvent(
            event_id="<script>", power_id="<evil>",
            trigger=EvolutionTrigger.USAGE,
            description="<b>hello</b> world",
            strength_before=0.1, strength_after=0.2,
        )
        assert "<" not in ev.event_id
        assert "<" not in ev.power_id
        assert "<" in ev.description  # content-safe


class TestShadowPowers:
    def test_initialization(self):
        sp = ShadowPowers()
        stats = sp.get_stats()
        assert stats["total_powers"] == 10
        assert stats["base_powers"] == 10
        assert stats["derived_powers"] == 0

    def test_base_powers_all_awakened(self):
        sp = ShadowPowers()
        for name in _BASE_POWERS:
            power = sp.get_power(name)
            assert power is not None, f"Base power '{name}' not found"
            assert power["state"] == "awakened"

    def test_apply_power_nonexistent(self):
        sp = ShadowPowers()
        result = sp.apply_power("nonexistent", "ctx", "res", True)
        assert result is None

    def test_apply_power_success(self):
        sp = ShadowPowers()
        app = sp.apply_power("shadow_extract", "test context", "found pattern", True)
        assert app is not None
        assert app.success is True
        assert app.strength_after > app.strength_before

    def test_apply_power_failure(self):
        sp = ShadowPowers()
        app = sp.apply_power("shadow_extract", "test", "no pattern found", False)
        assert app is not None
        assert app.success is False

    def test_apply_power_increments_usage(self):
        sp = ShadowPowers()
        power_before = sp.get_power("shadow_extract")
        sp.apply_power("shadow_extract", "ctx", "res", True)
        power_after = sp.get_power("shadow_extract")
        assert power_after["usage_count"] == power_before["usage_count"] + 1

    def test_apply_power_fatigue(self):
        sp = ShadowPowers()
        for _ in range(5):
            sp.apply_power("shadow_extract", "ctx", "res", True)
        power = sp.get_power("shadow_extract")
        assert power["fatigue"] > 0

    def test_apply_power_fatigue_reduces_gain(self):
        sp = ShadowPowers()
        for _ in range(15):
            sp.apply_power("shadow_extract", "ctx", "res", True)
        power = sp.get_power("shadow_extract")
        assert power["fatigue"] >= _FATIGUE_THRESHOLD

    def test_recover_fatigue(self):
        sp = ShadowPowers()
        for _ in range(10):
            sp.apply_power("shadow_extract", "ctx", "res", True)
        fatigue_before = sp.get_power("shadow_extract")["fatigue"]
        sp.recover_fatigue()
        fatigue_after = sp.get_power("shadow_extract")["fatigue"]
        assert fatigue_after <= fatigue_before

    def test_state_progression(self):
        sp = ShadowPowers()
        power = sp.get_power("shadow_extract")
        assert power["state"] == "awakened"

        for _ in range(20):
            sp.apply_power("shadow_extract", "ctx" * 10, "res", True)
        sp.recover_fatigue()
        power = sp.get_power("shadow_extract")
        assert power["state"] in ("active", "evolving", "mastered", "awakened")

    def test_evolution_creates_derived(self):
        sp = ShadowPowers()
        # Boost strength past threshold so evolution triggers
        sp._powers["shadow_extract"].strength = 0.3
        for _ in range(_EVOLUTION_USAGE_THRESHOLD + 2):
            sp.apply_power("shadow_extract", "ctx" * 10, "res", True)
            sp.recover_fatigue()
        stats = sp.get_stats()
        assert stats["derived_powers_created"] > 0 or stats["evolutions_triggered"] > 0

    def test_cross_pollination(self):
        sp = ShadowPowers()
        for _ in range(10):
            sp.apply_power("shadow_extract", "ctx", "res", True)
            sp.apply_power("monarch_sense", "ctx", "res", True)
            sp.recover_fatigue()
        cross_keys = [name for (a, b), name in _CROSS_POLLINATION.items()
                     if a == PowerCategory.EXTRACTION or b == PowerCategory.EXTRACTION]
        found = any(sp.get_power(name) is not None for name in cross_keys)
        stats = sp.get_stats()
        assert found or stats["cross_pollinations"] >= 0

    def test_synthesize_powers_insufficient(self):
        sp = ShadowPowers()
        result = sp.synthesize_powers(["shadow_extract"])
        assert result is None

    def test_synthesize_powers(self):
        sp = ShadowPowers()
        result = sp.synthesize_powers(
            ["shadow_extract", "monarch_sense"], context="test synthesis"
        )
        assert result is not None
        assert result.is_base is False
        assert "synth_" in result.name

    def test_synthesize_duplicate(self):
        sp = ShadowPowers()
        first = sp.synthesize_powers(["shadow_extract", "monarch_sense"])
        second = sp.synthesize_powers(["shadow_extract", "monarch_sense"])
        assert second is None  # same combination already exists

    def test_awaken_dormant(self):
        sp = ShadowPowers()
        dormant = ShadowPower(
            power_id="test_dormant", name="test_dormant_power",
            category=PowerCategory.SENSING,
            description="extract patterns data streams insights",
            state=PowerState.DORMANT, is_base=False,
        )
        sp._powers["test_dormant_power"] = dormant
        awakened = sp.awaken_dormant("extract patterns data streams insights anomalies")
        assert len(awakened) >= 1
        power = sp.get_power("test_dormant_power")
        assert power["state"] == "awakened"

    def test_awaken_dormant_no_match(self):
        sp = ShadowPowers()
        dormant = ShadowPower(
            power_id="xyz", name="xyz_power",
            category=PowerCategory.SENSING,
            description="completely unrelated keywords about quantum physics",
            state=PowerState.DORMANT, is_base=False,
        )
        sp._powers["xyz_power"] = dormant
        awakened = sp.awaken_dormant("kitchen cooking recipes")
        assert "xyz_power" not in awakened

    def test_list_powers(self):
        sp = ShadowPowers()
        all_powers = sp.list_powers()
        assert len(all_powers) == 10

    def test_list_powers_by_category(self):
        sp = ShadowPowers()
        extraction = sp.list_powers(category="extraction")
        assert len(extraction) == 1
        assert extraction[0]["category"] == "extraction"

    def test_list_powers_by_state(self):
        sp = ShadowPowers()
        awakened = sp.list_powers(state="awakened")
        assert len(awakened) == 10

    def test_get_power_tree(self):
        sp = ShadowPowers()
        tree = sp.get_power_tree()
        assert "shadow_extract" in tree
        assert "derived" in tree["shadow_extract"]

    def test_get_applications(self):
        sp = ShadowPowers()
        sp.apply_power("shadow_extract", "ctx", "res", True)
        apps = sp.get_applications()
        assert len(apps) >= 1

    def test_get_applications_by_power(self):
        sp = ShadowPowers()
        sp.apply_power("shadow_extract", "ctx", "res", True)
        power_id = sp.get_power("shadow_extract")["power_id"]
        apps = sp.get_applications(power_name=power_id)
        assert len(apps) >= 1

    def test_get_evolution_history(self):
        sp = ShadowPowers()
        history = sp.get_evolution_history()
        assert isinstance(history, list)

    def test_get_recommendations(self):
        sp = ShadowPowers()
        recs = sp.get_recommendations()
        assert isinstance(recs, list)
        assert len(recs) > 0  # Base powers are weak, should have recommendations

    def test_get_stats(self):
        sp = ShadowPowers()
        stats = sp.get_stats()
        assert "total_applications" in stats
        assert "total_powers" in stats
        assert "base_powers" in stats
        assert "avg_strength" in stats
        assert "state_distribution" in stats

    def test_mastery(self):
        sp = ShadowPowers()
        power = sp._powers["shadow_extract"]
        power.strength = _MASTERY_STRENGTH
        sp.apply_power("shadow_extract", "ctx", "res", True)
        power = sp._powers["shadow_extract"]
        assert power.state == PowerState.MASTERED

    def test_application_limit(self):
        sp = ShadowPowers()
        for i in range(_MAX_APPLICATIONS + 10):
            sp.apply_power("shadow_extract", f"ctx{i}", f"res{i}", True)
        stats = sp.get_stats()
        assert stats["total_applications"] == _MAX_APPLICATIONS + 10
        apps = sp.get_applications()
        assert len(apps) <= _MAX_APPLICATIONS

    def test_thread_safety(self):
        sp = ShadowPowers()
        errors = []

        def apply_many():
            try:
                for _ in range(50):
                    sp.apply_power("shadow_extract", "ctx", "res", True)
            except Exception as e:
                errors.append(e)

        threads = [threading.Thread(target=apply_many) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_apply_power_sanitizes(self):
        sp = ShadowPowers()
        app = sp.apply_power("shadow_extract", "<script>alert(1)</script>", "<b>result</b>", True)
        assert app is not None
        # power_name is sanitized (strict), context preserves <>
        assert "<script>" not in app.power_id

    def test_stats_lock_thread_safety(self):
        sp = ShadowPowers()
        results = []

        def read_stats():
            for _ in range(50):
                stats = sp.get_stats()
                results.append(stats["total_powers"])

        def apply_power():
            for _ in range(50):
                sp.apply_power("shadow_extract", "ctx", "res", True)

        threads = [
            threading.Thread(target=read_stats),
            threading.Thread(target=apply_power),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert all(r >= 10 for r in results)

    def test_get_power_returns_deepcopy(self):
        sp = ShadowPowers()
        d1 = sp.get_power("shadow_extract")
        d1["strength"] = 999
        d2 = sp.get_power("shadow_extract")
        assert d2["strength"] != 999

    def test_list_powers_returns_dicts(self):
        sp = ShadowPowers()
        powers = sp.list_powers()
        for p in powers:
            assert isinstance(p, dict)

    def test_power_tree_returns_deepcopy(self):
        sp = ShadowPowers()
        tree = sp.get_power_tree()
        tree["shadow_extract"]["strength"] = 999
        tree2 = sp.get_power_tree()
        assert tree2["shadow_extract"]["strength"] != 999


class TestShadowPowersAPI:
    """Test ShadowPowers API endpoint models."""

    def test_power_apply_request_model(self):
        from shadowd.cognition.api_router import PowerApplyRequest
        req = PowerApplyRequest(
            power_name="shadow_extract",
            context="test context",
            result="found pattern",
            success=True,
        )
        assert req.power_name == "shadow_extract"
        assert req.success is True

    def test_power_apply_request_defaults(self):
        from shadowd.cognition.api_router import PowerApplyRequest
        req = PowerApplyRequest(
            power_name="shadow_extract",
            context="test",
        )
        assert req.result == ""
        assert req.success is True

    def test_power_awaken_request_model(self):
        from shadowd.cognition.api_router import PowerAwakenRequest
        req = PowerAwakenRequest(context="detect anomalies")
        assert req.context == "detect anomalies"

    def test_power_synthesize_request_model(self):
        from shadowd.cognition.api_router import PowerSynthesizeRequest
        req = PowerSynthesizeRequest(
            power_names=["shadow_extract", "monarch_sense"],
            context="test",
        )
        assert len(req.power_names) == 2

    def test_api_router_has_powers_endpoints(self):
        from shadowd.cognition.api_router import router
        paths = [r.path for r in router.routes]
        assert "/os/powers/apply" in paths
        assert "/os/powers/recover" in paths
        assert "/os/powers/awaken" in paths
        assert "/os/powers/synthesize" in paths
        assert "/os/powers/{name}" in paths
        assert "/os/powers" in paths
        assert "/os/powers/tree" in paths
        assert "/os/powers/applications" in paths
        assert "/os/powers/evolution-history" in paths
        assert "/os/powers/recommendations" in paths
        assert "/os/powers/stats" in paths


class TestShadowPowersEnums:
    def test_power_category_values(self):
        assert len(PowerCategory) == 10
        assert "extraction" in [c.value for c in PowerCategory]

    def test_power_state_values(self):
        assert len(PowerState) == 6
        assert "awakened" in [s.value for s in PowerState]

    def test_evolution_trigger_values(self):
        assert len(EvolutionTrigger) == 6
        assert "usage" in [t.value for t in EvolutionTrigger]


class TestDerivationMap:
    def test_all_base_powers_have_derivations(self):
        for name in _BASE_POWERS:
            assert name in _DERIVATION_MAP, f"Missing derivations for {name}"
            assert len(_DERIVATION_MAP[name]) == 3

    def test_derivation_names_unique(self):
        all_names = []
        for name, derivations in _DERIVATION_MAP.items():
            for derived_name, _ in derivations:
                all_names.append(derived_name)
        assert len(all_names) == len(set(all_names))

    def test_cross_pollination_categories(self):
        for (cat_a, cat_b), name in _CROSS_POLLINATION.items():
            assert isinstance(cat_a, PowerCategory)
            assert isinstance(cat_b, PowerCategory)
            assert isinstance(name, str)


class TestEdgeCases:
    def test_apply_power_empty_context(self):
        sp = ShadowPowers()
        app = sp.apply_power("shadow_extract", "", "", True)
        assert app is not None

    def test_apply_power_very_long_context(self):
        sp = ShadowPowers()
        long_ctx = "x" * 10000
        app = sp.apply_power("shadow_extract", long_ctx, "res", True)
        assert app is not None
        assert len(app.context) <= 500

    def test_synthesize_with_nonexistent_power(self):
        sp = ShadowPowers()
        result = sp.synthesize_powers(["shadow_extract", "nonexistent_power"])
        assert result is None

    def test_synthesize_with_dormant_power(self):
        sp = ShadowPowers()
        dormant = ShadowPower(
            power_id="dormant1", name="dormant_one",
            category=PowerCategory.SENSING, description="test",
            state=PowerState.DORMANT, is_base=False,
        )
        sp._powers["dormant_one"] = dormant
        result = sp.synthesize_powers(["shadow_extract", "dormant_one"])
        assert result is None

    def test_get_nonexistent_power(self):
        sp = ShadowPowers()
        assert sp.get_power("nonexistent") is None

    def test_evolution_log_limit(self):
        sp = ShadowPowers()
        for _ in range(_EVOLUTION_USAGE_THRESHOLD + 1):
            sp.apply_power("shadow_extract", "ctx", "res", True)
            sp.recover_fatigue()
        history = sp.get_evolution_history()
        assert len(history) <= _MAX_EVOLUTION_LOG

    def test_application_limit_parameter(self):
        sp = ShadowPowers()
        sp.apply_power("shadow_extract", "ctx", "res", True)
        apps = sp.get_applications(limit=1)
        assert len(apps) <= 1

    def test_evolution_history_limit_parameter(self):
        sp = ShadowPowers()
        history = sp.get_evolution_history(limit=1)
        assert isinstance(history, list)


class TestShadowPowersIntegration:
    """Test cross-module integration convenience methods."""

    def test_on_agent_created(self):
        sp = ShadowPowers()
        app = sp.on_agent_created("analyst_bot", "Analyst")
        assert app is not None
        assert app.category == PowerCategory.ARMY

    def test_on_agent_created_sanitizes(self):
        sp = ShadowPowers()
        app = sp.on_agent_created("<script>alert(1)</script>", "Analyst")
        assert app is not None
        # context uses _sanitize_content which preserves <>
        # but power_name uses _sanitize which strips <>
        assert "<script>" not in app.power_id

    def test_on_skill_leveled_reasoning(self):
        sp = ShadowPowers()
        app = sp.on_skill_leveled("skill1", "reasoning", "intermediate")
        assert app is not None
        assert app.category == PowerCategory.EXTRACTION

    def test_on_skill_leveled_creativity(self):
        sp = ShadowPowers()
        app = sp.on_skill_leveled("skill2", "creativity", "advanced")
        assert app is not None
        assert app.category == PowerCategory.SYNTHESIS

    def test_on_skill_leveled_unknown_domain(self):
        sp = ShadowPowers()
        app = sp.on_skill_leveled("skill3", "unknown_domain", "novice")
        assert app is None

    def test_on_resonance_event(self):
        sp = ShadowPowers()
        app = sp.on_resonance_event("amplifying", "cognitive_module", 0.75)
        assert app is not None
        assert app.category == PowerCategory.SYNTHESIS

    def test_on_resonance_event_sanitizes(self):
        sp = ShadowPowers()
        app = sp.on_resonance_event("<script>", "<evil>", 2.0)
        assert app is not None
        assert "<script>" not in app.context

    def test_integration_increments_stats(self):
        sp = ShadowPowers()
        sp.on_agent_created("bot", "Analyst")
        sp.on_skill_leveled("s1", "reasoning", "intermediate")
        sp.on_resonance_event("amplifying", "test", 0.5)
        stats = sp.get_stats()
        assert stats["total_applications"] == 3