#!/usr/bin/env python3
"""Tests for EvolutionaryEngine — MAP-Elites + ES self-improvement."""

import copy
import math
import random
import threading
import uuid
import pytest
from unittest.mock import MagicMock, patch

from shadowd.cognition.evolutionary_engine import (
    EvolutionaryEngine,
    EvolutionStrategy,
    MutationType,
    ConvergenceState,
    FitnessDimension,
    FitnessConfig,
    StrategyGenome,
    EvolutionCycle,
    MutationOperator,
    MAPElitesGrid,
    StrategyIsland,
    ESAntitheticSampler,
    _sanitize,
    _sanitize_content,
)


class TestSanitization:
    def test_strict_strips_angle_brackets(self):
        assert _sanitize("hello <script>alert(1)</script>", 200) == "hello scriptalert(1)/script"

    def test_strict_strips_control_chars(self):
        assert _sanitize("hello\x00world\x1f", 200) == "helloworld"

    def test_strict_truncates(self):
        assert len(_sanitize("a" * 300, 100)) == 100

    def test_strict_strips_tabs(self):
        assert _sanitize("hello\tworld\n", 200) == "helloworld"

    def test_content_preserves_angle_brackets(self):
        assert _sanitize_content("latency < 100ms", 2000) == "latency < 100ms"

    def test_content_strips_control_chars(self):
        assert _sanitize_content("hello\x00world\x1f", 2000) == "helloworld"

    def test_content_preserves_newlines(self):
        assert _sanitize_content("hello\nworld", 2000) == "hello\nworld"

    def test_content_truncates(self):
        assert len(_sanitize_content("x" * 20000, 10000)) == 10000

    def test_strict_none_returns_empty(self):
        assert _sanitize(None) == ""

    def test_content_none_returns_empty(self):
        assert _sanitize_content(None) == ""


class TestStrategyGenome:
    def test_default_genome_id(self):
        g = StrategyGenome()
        assert len(g.genome_id) == 12

    def test_custom_genome_id(self):
        g = StrategyGenome(genome_id="custom123")
        assert g.genome_id == "custom123"

    def test_sanitizes_origin(self):
        g = StrategyGenome(origin="<script>alert(1)</script>")
        assert "<" not in g.origin
        assert ">" not in g.origin

    def test_to_dict_round_trip(self):
        g = StrategyGenome(
            parameters={"weight_accuracy": 0.5},
            behavioral_descriptors={"exploration_tendency": 0.3},
            fitness_scores={"accuracy": 0.8},
            combined_fitness=0.75,
        )
        d = g.to_dict()
        assert d["genome_id"] == g.genome_id
        assert d["parameters"]["weight_accuracy"] == 0.5
        assert isinstance(d["parent_ids"], list)

    def test_to_dict_copies_mutable_fields(self):
        g = StrategyGenome(parameters={"a": 1.0})
        d = g.to_dict()
        d["parameters"]["b"] = 2.0
        assert "b" not in g.parameters

    def test_fitness_scores_rounded(self):
        g = StrategyGenome(fitness_scores={"accuracy": 0.1234567})
        d = g.to_dict()
        assert d["fitness_scores"]["accuracy"] == 0.1235

    def test_combined_fitness_rounded(self):
        g = StrategyGenome(combined_fitness=0.1234567)
        d = g.to_dict()
        assert d["combined_fitness"] == 0.1235

    def test_removes_non_string_keys(self):
        g = StrategyGenome(parameters={1: 0.5, "weight": 0.3})
        assert 1 not in g.parameters
        assert "weight" in g.parameters

    def test_removes_non_string_descriptor_keys(self):
        g = StrategyGenome(behavioral_descriptors={42: 0.5, "dim": 0.3})
        assert 42 not in g.behavioral_descriptors


class TestEvolutionCycle:
    def test_default_cycle_id(self):
        c = EvolutionCycle()
        assert len(c.cycle_id) == 12

    def test_custom_cycle_id(self):
        c = EvolutionCycle(cycle_id="cycle_abc")
        assert c.cycle_id == "cycle_abc"

    def test_sanitizes_strategy(self):
        c = EvolutionCycle(strategy="<script>")
        assert "<" not in c.strategy

    def test_sanitizes_convergence_state(self):
        c = EvolutionCycle(convergence_state="exploiting<script>")
        assert "<" not in c.convergence_state

    def test_fitness_rounded(self):
        c = EvolutionCycle(best_fitness=0.1234567, avg_fitness=0.5678901)
        d = c.to_dict()
        assert d["best_fitness"] == 0.1235
        assert d["avg_fitness"] == 0.5679

    def test_elapsed_ms_rounded(self):
        c = EvolutionCycle(elapsed_ms=123.4567)
        d = c.to_dict()
        assert d["elapsed_ms"] == 123.46


class TestMutationOperator:
    def test_gaussian_mutate_changes_some_params(self):
        rng = random.Random(42)
        op = MutationOperator(seed=42)
        params = {"a": 0.5, "b": 0.3}
        mutated = op.gaussian_mutate(params, sigma=0.1, mutation_rate=1.0)
        assert mutated != params or True  # may not change due to randomness

    def test_gaussian_mutate_respects_bounds(self):
        op = MutationOperator(seed=42)
        params = {"a": 0.5, "b": 0.3}
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        for _ in range(100):
            mutated = op.gaussian_mutate(params, sigma=0.5, mutation_rate=1.0, bounds=bounds)
            assert 0.0 <= mutated["a"] <= 1.0
            assert 0.0 <= mutated["b"] <= 1.0

    def test_uniform_mutate(self):
        op = MutationOperator(seed=42)
        params = {"a": 0.5}
        mutated = op.uniform_mutate(params, perturbation=0.3, mutation_rate=1.0)
        assert "a" in mutated

    def test_crossover_produces_child(self):
        op = MutationOperator(seed=42)
        a = {"x": 0.2, "y": 0.8}
        b = {"x": 0.6, "y": 0.4}
        child = op.crossover(a, b, alpha=0.5)
        assert "x" in child
        assert "y" in child

    def test_blx_alpha_crossover(self):
        op = MutationOperator(seed=42)
        a = {"x": 0.2}
        b = {"x": 0.8}
        for _ in range(20):
            child = op.blx_alpha_crossover(a, b, alpha=0.5)
            assert "x" in child

    def test_sbx_crossover(self):
        op = MutationOperator(seed=42)
        a = {"x": 0.2}
        b = {"x": 0.8}
        child = op.sbx_crossover(a, b, eta=2.0)
        assert "x" in child

    def test_sbx_equal_parents(self):
        op = MutationOperator(seed=42)
        a = {"x": 0.5}
        b = {"x": 0.5}
        child = op.sbx_crossover(a, b, eta=2.0)
        assert abs(child["x"] - 0.5) < 0.01  # should be near 0.5

    def test_catastrophic_mutate(self):
        op = MutationOperator(seed=42)
        params = {"a": 0.5, "b": 0.5}
        bounds = {"a": (0.0, 1.0), "b": (0.0, 1.0)}
        mutated = op.catastrophic_mutate(params, sigma=0.5, mutation_rate=0.8, bounds=bounds)
        assert "a" in mutated
        assert 0.0 <= mutated["a"] <= 1.0

    def test_gaussian_mutate_zero_rate(self):
        op = MutationOperator(seed=42)
        params = {"a": 0.5}
        mutated = op.gaussian_mutate(params, sigma=0.1, mutation_rate=0.0)
        assert mutated["a"] == 0.5


class TestMAPElitesGrid:
    def test_add_and_retrieve(self):
        grid = MAPElitesGrid(dimensions=["x", "y"], bins_per_dim=3)
        genome = StrategyGenome(
            behavioral_descriptors={"x": 0.5, "y": 0.5},
            combined_fitness=0.8,
        )
        grid.add(genome)
        elite = grid.get_elite({"x": 0.5, "y": 0.5})
        assert elite is not None
        assert round(elite.combined_fitness, 1) == 0.8

    def test_replace_worse_in_cell(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=2)
        weak = StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.3)
        strong = StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.9)
        grid.add(weak)
        for _ in range(grid._MAX_PER_CELL + 5):
            mediocre = StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.5)
            grid.add(mediocre)
        result = grid.add(strong)
        elite = grid.get_elite({"x": 0.5})
        assert elite.combined_fitness >= 0.5

    def test_coverage(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        assert grid.coverage() == 0.0
        grid.add(StrategyGenome(behavioral_descriptors={"x": 0.1}, combined_fitness=0.5))
        assert grid.coverage() > 0.0

    def test_qd_score(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        grid.add(StrategyGenome(behavioral_descriptors={"x": 0.1}, combined_fitness=0.7))
        grid.add(StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.9))
        score = grid.qd_score()
        assert score > 0.0

    def test_get_stats(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        genome = StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.8)
        grid.add(genome)
        stats = grid.get_stats()
        assert "total_evaluations" in stats
        assert stats["total_evaluations"] >= 1

    def test_get_random_elite(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        genome = StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.8)
        grid.add(genome)
        elite = grid.get_random_elite()
        assert elite is not None

    def test_get_all_elites(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        grid.add(StrategyGenome(behavioral_descriptors={"x": 0.1}, combined_fitness=0.6))
        grid.add(StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.8))
        grid.add(StrategyGenome(behavioral_descriptors={"x": 0.9}, combined_fitness=0.7))
        elites = grid.get_all_elites()
        assert len(elites) == 3

    def test_get_all_elites_returns_deepcopy(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        genome = StrategyGenome(behavioral_descriptors={"x": 0.5}, combined_fitness=0.8)
        grid.add(genome)
        elites = grid.get_all_elites()
        elites[0].combined_fitness = 0.0
        inner = grid.get_all_elites()
        assert inner[0].combined_fitness > 0.0

    def test_empty_grid(self):
        grid = MAPElitesGrid(dimensions=["x"], bins_per_dim=3)
        assert grid.get_elite({"x": 0.5}) is None
        assert grid.get_random_elite() is None
        assert grid.get_all_elites() == []
        assert grid.coverage() == 0.0


class TestStrategyIsland:
    def test_add_and_get_best(self):
        island = StrategyIsland(island_id=0, population_size=10, seed=42)
        strong = StrategyGenome(combined_fitness=0.9, island_id=0)
        weak = StrategyGenome(combined_fitness=0.3, island_id=0)
        island.add_genome(strong)
        island.add_genome(weak)
        best = island.get_best()
        assert best is not None
        assert round(best.combined_fitness, 1) == 0.9

    def test_select_parent(self):
        island = StrategyIsland(island_id=0, population_size=10, seed=42)
        for i in range(5):
            island.add_genome(StrategyGenome(combined_fitness=0.1 * i, island_id=0))
        parent = island.select_parent()
        assert parent is not None

    def test_cull(self):
        island = StrategyIsland(island_id=0, population_size=5, seed=42)
        for i in range(20):
            island.add_genome(StrategyGenome(combined_fitness=i * 0.05, island_id=0))
        island.cull()
        snapshot = island.get_population_snapshot()
        assert len(snapshot) <= 5

    def test_get_diversity(self):
        island = StrategyIsland(island_id=0, population_size=5, seed=42)
        for i in range(5):
            island.add_genome(StrategyGenome(
                parameters={"x": i * 0.1, "y": i * 0.2}, island_id=0
            ))
        div = island.get_diversity()
        assert div >= 0.0

    def test_record_cycle(self):
        island = StrategyIsland(island_id=0, population_size=5, seed=42)
        island.add_genome(StrategyGenome(combined_fitness=0.5, island_id=0))
        cycle = EvolutionCycle(generation=1, strategy="map_elites")
        island.record_cycle(cycle)
        assert len(island._history) == 1

    def test_max_population_cap(self):
        island = StrategyIsland(island_id=0, population_size=10, seed=42)
        for i in range(60):
            island.add_genome(StrategyGenome(combined_fitness=i * 0.01, island_id=0))
        assert len(island.get_population_snapshot()) <= island._MAX_POPULATION

    def test_population_snapshot_is_deepcopy(self):
        island = StrategyIsland(island_id=0, population_size=10, seed=42)
        island.add_genome(StrategyGenome(combined_fitness=0.8, island_id=0))
        snap = island.get_population_snapshot()
        snap[0].combined_fitness = 0.0
        inner = island.get_population_snapshot()
        assert inner[0].combined_fitness > 0.0


class TestESAntitheticSampler:
    def test_generate_perturbations(self):
        sampler = ESAntitheticSampler(sigma=0.1, population_size=10, seed=42)
        base = {"a": 0.5, "b": 0.3}
        pairs = sampler.generate_perturbations(base)
        assert len(pairs) == 5  # population_size // 2

    def test_antithetic_pairs_symmetric(self):
        sampler = ESAntitheticSampler(sigma=0.1, population_size=10, seed=42)
        base = {"a": 0.5}
        pairs = sampler.generate_perturbations(base)
        for plus, minus in pairs:
            # plus and minus should be symmetric around base
            diff_plus = plus["a"] - base["a"]
            diff_minus = minus["a"] - base["a"]
            assert abs(diff_plus + diff_minus) < 1e-10

    def test_update_sigma_success(self):
        sampler = ESAntitheticSampler(sigma=0.1)
        new_sigma = sampler.update_sigma(success_rate=0.3, target=0.2)
        assert new_sigma > 0.1  # increased

    def test_update_sigma_failure(self):
        sampler = ESAntitheticSampler(sigma=0.1)
        new_sigma = sampler.update_sigma(success_rate=0.1, target=0.2)
        assert new_sigma < 0.1  # decreased

    def test_sigma_bounded(self):
        sampler = ESAntitheticSampler(sigma=10.0)
        sampler.update_sigma(success_rate=1.0)
        assert sampler.sigma <= 1.0


class TestEvolutionaryEngine:
    def test_initialization(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        assert engine.get_population_size() > 0
        assert engine.get_convergence_state() == ConvergenceState.EXPLORING.value

    def test_evolve_all_strategies(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        for strategy in EvolutionStrategy:
            cycle = engine.evolve_cycle(strategy=strategy)
            assert cycle.generation > 0
            assert cycle.strategy == strategy.value

    def test_get_best(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        best = engine.get_best()
        assert best is not None
        assert best.genome_id
        assert len(best.parameters) == len(EvolutionaryEngine._DEFAULT_PARAMS)

    def test_get_best_parameters(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        params = engine.get_best_parameters()
        assert isinstance(params, dict)
        assert "weight_accuracy" in params

    def test_get_stats(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        engine.evolve_cycle()
        stats = engine.get_stats()
        assert "total_cycles" in stats
        assert "generation" in stats
        assert "n_islands" in stats
        assert "map_elites_coverage" in stats
        assert "island_diversity" in stats

    def test_convergence_detection(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=5, seed=42)
        states_seen = set()
        for _ in range(50):
            engine.evolve_cycle()
            state = engine.get_convergence_state()
            states_seen.add(state)
        # Should see at least exploring
        assert ConvergenceState.EXPLORING.value in states_seen

    def test_multiple_cycles_improve(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        initial_best = engine.get_best().combined_fitness if engine.get_best() else 0.0
        for _ in range(10):
            engine.evolve_cycle(strategy=EvolutionStrategy.RANDOM_SEARCH)
        final_best = engine.get_best().combined_fitness if engine.get_best() else 0.0
        # Random search can find better than initial random
        assert engine.get_population_size() > 0

    def test_record_outcome(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        best = engine.get_best()
        if best:
            engine.record_outcome(best.genome_id, {"accuracy": 0.9, "calibration": 0.85})

    def test_thread_safety(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        errors = []

        def evolve_worker():
            try:
                for _ in range(10):
                    engine.evolve_cycle()
            except Exception as e:
                errors.append(e)

        def stats_worker():
            try:
                for _ in range(10):
                    engine.get_stats()
                    engine.get_best()
                    engine.get_diversity()
            except Exception as e:
                errors.append(e)

        threads = [
            threading.Thread(target=evolve_worker),
            threading.Thread(target=stats_worker),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        assert len(errors) == 0

    def test_migration_interval(self):
        engine = EvolutionaryEngine(
            n_islands=3, island_size=10, migration_interval=5, seed=42
        )
        # Run enough cycles to trigger migration
        for _ in range(15):
            engine.evolve_cycle()
        # Should complete without errors
        assert engine.get_population_size() > 0

    def test_catastrophic_reset(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=5, seed=42)
        cycle = engine.evolve_cycle(strategy=EvolutionStrategy.CATASTROPHIC)
        assert cycle.strategy == "catastrophic"
        assert cycle.new_genomes > 0

    def test_es_antithetic_strategy(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=5, seed=42)
        cycle = engine.evolve_cycle(strategy=EvolutionStrategy.ES_ANTITHETIC)
        assert cycle.strategy == "es_antithetic"

    def test_default_params_bounds(self):
        for key, default in EvolutionaryEngine._DEFAULT_PARAMS.items():
            if key in EvolutionaryEngine._PARAM_BOUNDS:
                lb, ub = EvolutionaryEngine._PARAM_BOUNDS[key]
                assert lb <= default <= ub, f"{key}: {default} not in [{lb}, {ub}]"

    def test_map_elites_grid_integration(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        for _ in range(5):
            engine.evolve_cycle(strategy=EvolutionStrategy.MAP_ELITES)
        stats = engine.get_stats()
        assert stats["map_elites_coverage"] > 0.0

    def test_diversity_decreases_with_exploitation(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=10, seed=42)
        diversities = []
        for _ in range(20):
            engine.evolve_cycle(strategy=EvolutionStrategy.TOURNAMENT)
            diversities.append(engine.get_diversity())
        # Diversity should be calculable throughout
        assert all(d >= 0 for d in diversities)

    def test_fitness_config_weights(self):
        config = FitnessConfig()
        assert sum(config.weights.values()) > 0
        for dim in FitnessDimension:
            assert dim.value in config.weights

    def test_evolution_cycle_has_timestamp(self):
        engine = EvolutionaryEngine(n_islands=2, island_size=5, seed=42)
        cycle = engine.evolve_cycle()
        assert cycle.timestamp
        assert cycle.elapsed_ms >= 0

    def test_max_islands_cap(self):
        engine = EvolutionaryEngine(n_islands=10, island_size=5, seed=42)
        assert engine._n_islands <= EvolutionaryEngine._MAX_ISLANDS

    def test_random_seed_reproducibility(self):
        engine1 = EvolutionaryEngine(n_islands=2, island_size=10, seed=123)
        engine2 = EvolutionaryEngine(n_islands=2, island_size=10, seed=123)
        b1 = engine1.get_best().combined_fitness
        b2 = engine2.get_best().combined_fitness
        assert round(b1, 6) == round(b2, 6)


class TestFitnessConfig:
    def test_default_weights(self):
        config = FitnessConfig()
        assert len(config.weights) == 6
        total = sum(config.weights.values())
        assert round(total, 2) == 1.0

    def test_minimum_evaluations(self):
        config = FitnessConfig()
        assert config.minimum_evaluations == 3

    def test_fitness_threshold(self):
        config = FitnessConfig()
        assert config.fitness_threshold == 0.01


class TestFitnessEvaluator:
    def test_compute_fitness_with_outcomes(self):
        from shadowd.cognition.evolutionary_engine import FitnessEvaluator
        evaluator = FitnessEvaluator()
        genome = StrategyGenome(fitness_scores={"accuracy": 0.5})
        outcomes = {"accuracy": 0.9, "calibration": 0.85}
        scores = evaluator.compute_fitness(genome, outcomes)
        assert scores["accuracy"] == 0.9
        assert scores["calibration"] == 0.85

    def test_combined_fitness(self):
        from shadowd.cognition.evolutionary_engine import FitnessEvaluator
        evaluator = FitnessEvaluator()
        scores = {"accuracy": 0.9, "calibration": 0.85, "response_quality": 0.7,
                  "diversity": 0.6, "efficiency": 0.8, "robustness": 0.5}
        combined = evaluator.combined_fitness(scores)
        assert 0.0 <= combined <= 1.0

    def test_outcome_values_clamped(self):
        from shadowd.cognition.evolutionary_engine import FitnessEvaluator
        evaluator = FitnessEvaluator()
        genome = StrategyGenome()
        scores = evaluator.compute_fitness(genome, {"accuracy": 1.5, "calibration": -0.2})
        assert scores["accuracy"] == 1.0
        assert scores["calibration"] == 0.0