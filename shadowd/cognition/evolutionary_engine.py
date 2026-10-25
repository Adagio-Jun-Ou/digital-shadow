#!/usr/bin/env python3
"""
Evolutionary Self-Improvement Engine - MAP-Elites + ES for ShadowAI

Implements quality-diversity optimization (MAP-Elites) combined with
Evolution Strategies (ES) for continuous self-improvement of ShadowAI's
cognitive strategies, decision policies, and behavioral parameters.

Architecture:
  EvolutionaryEngine
  ├── MAPElitesGrid — Quality-diversity archive for strategy variants
  ├── StrategyPopulation — Island-based population with migration
  ├── FitnessEvaluator — Multi-objective fitness scoring
  ├── MutationOperators — Parameter perturbation and crossover
  ├── SelectionPressure — Tournament + diversity-based selection
  └── EvolutionCoordinator — Orchestrates cycles, manages convergence

Inspired by:
  - OpenAI ES (Salimans et al., 2017) — scalar communication, scalable ES
  - EGGROLL (Sarkar et al., 2025) — low-rank perturbation for efficient ES
  - ES for LLM Fine-Tuning (Qiu et al., 2025) — robust, reward-hack resistant
  - OpenEvolve (Sharma, 2025) — MAP-Elites + LLM code evolution with islands

Key Design Decisions:
  1. MAP-Elites grid stores strategy variants across behavioral descriptors
  2. Island model prevents premature convergence via migration
  3. ES-inspired gradient estimation via antithetic sampling
  4. Fitness is multi-objective: accuracy, calibration, response_quality, diversity
  5. Evolution operates on SHADOWAI STRATEGY PARAMETERS, not code mutations
  6. Convergence detection prevents wasting cycles on plateaued strategies
"""

import copy
import math
import os
import random
import threading
import time
import uuid
import logging
import json
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
from collections import defaultdict
from pathlib import Path

logger = logging.getLogger("shadowd.cognition.evolutionary_engine")

import re as _re

_SANITIZE_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_CONTENT_RE = _re.compile(r'[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]')
_SANITIZE_STRICT_RE = _re.compile(r'[<>\x00-\x1f\x7f]')


def _sanitize(text: str, max_length: int = 200) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_STRICT_RE.sub('', text)
    return text.strip()[:max_length]


def _sanitize_content(text: str, max_length: int = 10000) -> str:
    if text is None:
        return ""
    if not isinstance(text, str):
        text = str(text)
    text = _SANITIZE_CONTENT_RE.sub('', text)
    return text.strip()[:max_length]


# ============ ENUMS ============

class EvolutionStrategy(str, Enum):
    MAP_ELITES = "map_elites"
    TOURNAMENT = "tournament"
    ES_ANTITHETIC = "es_antithetic"
    RANDOM_SEARCH = "random_search"
    CATASTROPHIC = "catastrophic"


class MutationType(str, Enum):
    GAUSSIAN = "gaussian"
    UNIFORM = "uniform"
    ADAPTIVE = "adaptive"
    CROSSOVER = "crossover"
    CATASTROPHIC = "catastrophic"


class ConvergenceState(str, Enum):
    EXPLORING = "exploring"
    EXPLOITING = "exploiting"
    CONVERGED = "converged"
    STAGNANT = "stagnant"


class FitnessDimension(str, Enum):
    ACCURACY = "accuracy"
    CALIBRATION = "calibration"
    RESPONSE_QUALITY = "response_quality"
    DIVERSITY = "diversity"
    EFFICIENCY = "efficiency"
    ROBUSTNESS = "robustness"


# ============ DATA CLASSES ============

@dataclass
class StrategyGenome:
    genome_id: str = ""
    parent_ids: List[str] = field(default_factory=list)
    generation: int = 0
    island_id: int = 0
    parameters: Dict[str, float] = field(default_factory=dict)
    behavioral_descriptors: Dict[str, float] = field(default_factory=dict)
    fitness_scores: Dict[str, float] = field(default_factory=dict)
    combined_fitness: float = 0.0
    evaluation_count: int = 0
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    origin: str = "initial"

    def __post_init__(self):
        if not self.genome_id:
            self.genome_id = uuid.uuid4().hex[:12]
        self.origin = _sanitize(self.origin, 50)
        for k in list(self.parameters.keys()):
            if not isinstance(k, str):
                del self.parameters[k]
        for k in list(self.behavioral_descriptors.keys()):
            if not isinstance(k, str):
                del self.behavioral_descriptors[k]
        for k in list(self.fitness_scores.keys()):
            if not isinstance(k, str):
                del self.fitness_scores[k]

    def to_dict(self) -> Dict[str, Any]:
        return {
            "genome_id": self.genome_id,
            "parent_ids": list(self.parent_ids),
            "generation": self.generation,
            "island_id": self.island_id,
            "parameters": dict(self.parameters),
            "behavioral_descriptors": dict(self.behavioral_descriptors),
            "fitness_scores": {k: round(v, 4) for k, v in self.fitness_scores.items()},
            "combined_fitness": round(self.combined_fitness, 4),
            "evaluation_count": self.evaluation_count,
            "created_at": self.created_at,
            "origin": self.origin,
        }


@dataclass
class EvolutionCycle:
    cycle_id: str = ""
    generation: int = 0
    strategy: str = ""
    population_size: int = 0
    best_fitness: float = 0.0
    avg_fitness: float = 0.0
    diversity_score: float = 0.0
    convergence_state: str = ConvergenceState.EXPLORING.value
    new_genomes: int = 0
    elapsed_ms: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def __post_init__(self):
        if not self.cycle_id:
            self.cycle_id = uuid.uuid4().hex[:12]
        self.convergence_state = _sanitize(self.convergence_state, 30)
        self.strategy = _sanitize(self.strategy, 50)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "cycle_id": self.cycle_id,
            "generation": self.generation,
            "strategy": self.strategy,
            "population_size": self.population_size,
            "best_fitness": round(self.best_fitness, 4),
            "avg_fitness": round(self.avg_fitness, 4),
            "diversity_score": round(self.diversity_score, 4),
            "convergence_state": self.convergence_state,
            "new_genomes": self.new_genomes,
            "elapsed_ms": round(self.elapsed_ms, 2),
            "timestamp": self.timestamp,
        }


@dataclass
class FitnessConfig:
    weights: Dict[str, float] = field(default_factory=lambda: {
        FitnessDimension.ACCURACY.value: 0.30,
        FitnessDimension.CALIBRATION.value: 0.20,
        FitnessDimension.RESPONSE_QUALITY.value: 0.20,
        FitnessDimension.DIVERSITY.value: 0.15,
        FitnessDimension.EFFICIENCY.value: 0.10,
        FitnessDimension.ROBUSTNESS.value: 0.05,
    })
    minimum_evaluations: int = 3
    fitness_threshold: float = 0.01


# ============ MUTATION OPERATORS ============

class MutationOperator:
    def __init__(self, seed: Optional[int] = None):
        self._rng = random.Random(seed or 42)

    def gaussian_mutate(
        self,
        params: Dict[str, float],
        sigma: float = 0.1,
        mutation_rate: float = 0.3,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        result = dict(params)
        for key in result:
            if self._rng.random() < mutation_rate:
                result[key] += self._rng.gauss(0, sigma)
                if bounds and key in bounds:
                    result[key] = max(bounds[key][0], min(bounds[key][1], result[key]))
        return result

    def uniform_mutate(
        self,
        params: Dict[str, float],
        perturbation: float = 0.2,
        mutation_rate: float = 0.3,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        result = dict(params)
        for key in result:
            if self._rng.random() < mutation_rate:
                result[key] += self._rng.uniform(-perturbation, perturbation)
                if bounds and key in bounds:
                    result[key] = max(bounds[key][0], min(bounds[key][1], result[key]))
        return result

    def crossover(
        self,
        parent_a: Dict[str, float],
        parent_b: Dict[str, float],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        child = {}
        for key in parent_a:
            if key in parent_b:
                if self._rng.random() < alpha:
                    child[key] = parent_a[key]
                else:
                    child[key] = parent_b[key]
            else:
                child[key] = parent_a[key]
        for key in parent_b:
            if key not in child:
                child[key] = parent_b[key]
        return child

    def blx_alpha_crossover(
        self,
        parent_a: Dict[str, float],
        parent_b: Dict[str, float],
        alpha: float = 0.5,
    ) -> Dict[str, float]:
        child = {}
        for key in parent_a:
            if key in parent_b:
                a_val = parent_a[key]
                b_val = parent_b[key]
                lower = min(a_val, b_val)
                upper = max(a_val, b_val)
                expansion = alpha * (upper - lower)
                child[key] = self._rng.uniform(lower - expansion, upper + expansion)
            else:
                child[key] = parent_a[key]
        return child

    def sbx_crossover(
        self,
        parent_a: Dict[str, float],
        parent_b: Dict[str, float],
        eta: float = 2.0,
    ) -> Dict[str, float]:
        child = {}
        for key in parent_a:
            if key in parent_b:
                a_val = parent_a[key]
                b_val = parent_b[key]
                if abs(a_val - b_val) < 1e-10:
                    child[key] = (a_val + b_val) / 2.0
                else:
                    u = self._rng.random()
                    if u <= 0.5:
                        beta = (2.0 * u) ** (1.0 / (eta + 1.0))
                    else:
                        beta = (1.0 / (2.0 * (1.0 - u))) ** (1.0 / (eta + 1.0))
                    child[key] = 0.5 * ((1 + beta) * a_val + (1 - beta) * b_val)
            else:
                child[key] = parent_a[key]
        return child

    def catastrophic_mutate(
        self,
        params: Dict[str, float],
        sigma: float = 0.5,
        mutation_rate: float = 0.8,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ) -> Dict[str, float]:
        result = dict(params)
        for key in result:
            if self._rng.random() < mutation_rate:
                result[key] += self._rng.gauss(0, sigma)
                if bounds and key in bounds:
                    result[key] = max(bounds[key][0], min(bounds[key][1], result[key]))
        return result


# ============ MAP-ELITES GRID ============

class MAPElitesGrid:
    _MAX_PER_CELL = 10

    def __init__(
        self,
        dimensions: List[str],
        bins_per_dim: int = 6,
        bounds: Optional[Dict[str, Tuple[float, float]]] = None,
    ):
        self.dimensions = dimensions
        self.bins_per_dim = bins_per_dim
        self.bounds = bounds or {}
        self._grid: Dict[Tuple[int, ...], List[StrategyGenome]] = {}
        self._lock = threading.Lock()
        self._stats = {
            "total_evaluations": 0,
            "cells_occupied": 0,
            "total_improvements": 0,
        }
        self._stats_lock = threading.Lock()

    def _discretize(self, descriptors: Dict[str, float]) -> Tuple[int, ...]:
        indices = []
        for dim in self.dimensions:
            val = descriptors.get(dim, 0.5)
            if dim in self.bounds:
                lo, hi = self.bounds[dim]
                val = max(lo, min(hi, val))
                normalized = (val - lo) / (hi - lo) if hi > lo else 0.5
            else:
                normalized = max(0.0, min(1.0, val))
            bin_idx = min(int(normalized * self.bins_per_dim), self.bins_per_dim - 1)
            indices.append(bin_idx)
        return tuple(indices)

    def add(self, genome: StrategyGenome) -> bool:
        cell_key = self._discretize(genome.behavioral_descriptors)
        improved = False
        with self._lock:
            if cell_key not in self._grid:
                self._grid[cell_key] = []
            cell = self._grid[cell_key]
            if len(cell) < self._MAX_PER_CELL:
                cell.append(copy.deepcopy(genome))
                improved = True
            else:
                worst_idx = min(range(len(cell)), key=lambda i: cell[i].combined_fitness)
                if genome.combined_fitness > cell[worst_idx].combined_fitness:
                    cell[worst_idx] = copy.deepcopy(genome)
                    improved = True
            if improved:
                with self._stats_lock:
                    self._stats["total_evaluations"] += 1
                    self._stats["cells_occupied"] = len(self._grid)
                    self._stats["total_improvements"] += 1
        return improved

    def get_elite(self, descriptors: Dict[str, float]) -> Optional[StrategyGenome]:
        cell_key = self._discretize(descriptors)
        with self._lock:
            cell = self._grid.get(cell_key, [])
            if not cell:
                return None
            best = max(cell, key=lambda g: g.combined_fitness)
            return copy.deepcopy(best)

    def get_all_elites(self) -> List[StrategyGenome]:
        with self._lock:
            result = []
            for cell in self._grid.values():
                if cell:
                    best = max(cell, key=lambda g: g.combined_fitness)
                    result.append(copy.deepcopy(best))
            return result

    def get_random_elite(self, rng: Optional[random.Random] = None) -> Optional[StrategyGenome]:
        rng = rng or random
        with self._lock:
            occupied_keys = [k for k, v in self._grid.items() if v]
            if not occupied_keys:
                return None
            cell_key = rng.choice(occupied_keys)
            cell = self._grid[cell_key]
            best = max(cell, key=lambda g: g.combined_fitness)
            return copy.deepcopy(best)

    def coverage(self) -> float:
        with self._lock:
            total_cells = self.bins_per_dim ** len(self.dimensions)
            if total_cells == 0:
                return 0.0
            return round(len(self._grid) / total_cells, 4)

    def qd_score(self) -> float:
        with self._lock:
            scores = []
            for cell in self._grid.values():
                if cell:
                    scores.append(max(g.combined_fitness for g in cell))
            return round(sum(scores) / max(len(scores), 1), 4) if scores else 0.0

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            return dict(self._stats)


# ============ ISLAND POPULATION ============

class StrategyIsland:
    _MAX_POPULATION = 50
    _MAX_HISTORY = 200

    def __init__(self, island_id: int, population_size: int = 20, seed: Optional[int] = None):
        self.island_id = island_id
        self.population_size = min(population_size, self._MAX_POPULATION)
        self._population: List[StrategyGenome] = []
        self._history: List[EvolutionCycle] = []
        self._lock = threading.Lock()
        self._rng = random.Random(seed or (42 + island_id))

    def add_genome(self, genome: StrategyGenome) -> bool:
        with self._lock:
            if len(self._population) >= self._MAX_POPULATION:
                worst_idx = min(range(len(self._population)),
                                key=lambda i: self._population[i].combined_fitness)
                if genome.combined_fitness > self._population[worst_idx].combined_fitness:
                    self._population[worst_idx] = copy.deepcopy(genome)
                    return True
                return False
            self._population.append(copy.deepcopy(genome))
            return True

    def select_parent(self, tournament_size: int = 3) -> Optional[StrategyGenome]:
        with self._lock:
            if not self._population:
                return None
            candidates = self._rng.sample(
                self._population, min(tournament_size, len(self._population))
            )
            best = max(candidates, key=lambda g: g.combined_fitness)
            return copy.deepcopy(best)

    def get_best(self) -> Optional[StrategyGenome]:
        with self._lock:
            if not self._population:
                return None
            return copy.deepcopy(max(self._population, key=lambda g: g.combined_fitness))

    def get_diversity(self) -> float:
        with self._lock:
            if len(self._population) < 2:
                return 0.0
            total_dist = 0.0
            count = 0
            params_sets = [g.parameters for g in self._population]
            keys = set()
            for ps in params_sets:
                keys.update(ps.keys())
            keys = list(keys)
            for i in range(len(params_sets)):
                for j in range(i + 1, len(params_sets)):
                    dist = 0.0
                    for k in keys:
                        diff = params_sets[i].get(k, 0.0) - params_sets[j].get(k, 0.0)
                        dist += diff * diff
                    total_dist += math.sqrt(dist)
                    count += 1
            return round(total_dist / max(count, 1), 4) if count > 0 else 0.0

    def cull(self) -> None:
        with self._lock:
            if len(self._population) > self.population_size:
                self._population.sort(key=lambda g: g.combined_fitness, reverse=True)
                self._population = self._population[:self.population_size]

    def record_cycle(self, cycle: EvolutionCycle) -> None:
        with self._lock:
            self._history.append(cycle)
            if len(self._history) > self._MAX_HISTORY:
                self._history = self._history[-self._MAX_HISTORY:]

    def get_population_snapshot(self) -> List[StrategyGenome]:
        with self._lock:
            return [copy.deepcopy(g) for g in self._population]


# ============ FITNESS EVALUATOR ============

class FitnessEvaluator:
    def __init__(self, config: Optional[FitnessConfig] = None):
        self.config = config or FitnessConfig()

    def compute_fitness(
        self,
        genome: StrategyGenome,
        outcomes: Optional[Dict[str, float]] = None,
    ) -> Dict[str, float]:
        scores = {}
        if outcomes:
            for dim in FitnessDimension:
                dim_key = dim.value
                if dim_key in outcomes:
                    scores[dim_key] = max(0.0, min(1.0, outcomes[dim_key]))
                elif dim_key in genome.fitness_scores:
                    scores[dim_key] = genome.fitness_scores[dim_key]
        else:
            scores = dict(genome.fitness_scores)
        scores[self._diversity_from_params(genome.parameters)] = scores.get(
            self._diversity_from_params(genome.parameters), 0.5
        )
        return scores

    def combined_fitness(self, scores: Dict[str, float]) -> float:
        total = 0.0
        weight_sum = 0.0
        for dim_key, weight in self.config.weights.items():
            if dim_key in scores:
                total += weight * scores[dim_key]
                weight_sum += weight
        return round(total / max(weight_sum, 1e-10), 4)

    @staticmethod
    def _diversity_from_params(params: Dict[str, float]) -> str:
        return FitnessDimension.DIVERSITY.value


# ============ ES ANTITHETIC SAMPLER ============

class ESAntitheticSampler:
    def __init__(self, sigma: float = 0.1, population_size: int = 10, seed: Optional[int] = None):
        self.sigma = sigma
        self.population_size = population_size
        self._rng = random.Random(seed or 42)

    def generate_perturbations(self, base_params: Dict[str, float]) -> List[Tuple[Dict[str, float], Dict[str, float]]]:
        pairs = []
        for _ in range(self.population_size // 2):
            noise = {}
            for key in base_params:
                n = self._rng.gauss(0, self.sigma)
                noise[key] = n
            plus_params = {k: v + noise.get(k, 0) for k, v in base_params.items()}
            minus_params = {k: v - noise.get(k, 0) for k, v in base_params.items()}
            pairs.append((plus_params, minus_params))
        return pairs

    def estimate_gradient(
        self,
        base_fitness: float,
        fitness_plus: List[float],
        fitness_minus: List[float],
    ) -> Dict[str, float]:
        gradient = {}
        n = len(fitness_plus)
        if n == 0:
            return gradient
        avg_diff = sum(fp - fm for fp, fm in zip(fitness_plus, fitness_minus)) / n
        for key in self._last_keys if hasattr(self, '_last_keys') else []:
            gradient[key] = avg_diff / (2 * self.sigma * n)
        return gradient

    def update_sigma(self, success_rate: float, target: float = 0.2) -> float:
        if success_rate > target:
            self.sigma *= 1.1
        else:
            self.sigma *= 0.9
        self.sigma = max(0.001, min(1.0, self.sigma))
        return self.sigma


# ============ EVOLUTIONARY ENGINE ============

class EvolutionaryEngine:
    _DEFAULT_PARAMS = {
        "weight_accuracy": 0.30,
        "weight_calibration": 0.20,
        "weight_quality": 0.20,
        "weight_diversity": 0.15,
        "weight_efficiency": 0.10,
        "weight_robustness": 0.05,
        "exploration_rate": 0.3,
        "mutation_sigma": 0.1,
        "crossover_rate": 0.5,
        "selection_pressure": 2.0,
        "learning_rate": 0.05,
        "decay_rate": 0.01,
        "temperature": 1.0,
    }

    _PARAM_BOUNDS = {
        "weight_accuracy": (0.0, 1.0),
        "weight_calibration": (0.0, 1.0),
        "weight_quality": (0.0, 1.0),
        "weight_diversity": (0.0, 1.0),
        "weight_efficiency": (0.0, 1.0),
        "weight_robustness": (0.0, 1.0),
        "exploration_rate": (0.01, 1.0),
        "mutation_sigma": (0.001, 1.0),
        "crossover_rate": (0.0, 1.0),
        "selection_pressure": (0.5, 5.0),
        "learning_rate": (0.001, 0.5),
        "decay_rate": (0.0, 0.1),
        "temperature": (0.1, 10.0),
    }

    _BEHAVIORAL_DESCRIPTOR_DIMS = [
        "exploration_tendency",
        "response_specificity",
        "confidence_level",
    ]

    _MAX_ISLANDS = 5
    _MAX_CYCLES = 10000
    _CONVERGENCE_WINDOW = 20
    _STAGNATION_THRESHOLD = 0.001
    _ES_POPULATION_SIZE = 10

    def __init__(
        self,
        n_islands: int = 3,
        island_size: int = 20,
        migration_interval: int = 10,
        mutation_rate: float = 0.3,
        crossover_rate: float = 0.5,
        seed: Optional[int] = None,
        data_dir: Optional[str] = None,
    ):
        self._n_islands = min(n_islands, self._MAX_ISLANDS)
        self._island_size = island_size
        self._migration_interval = migration_interval
        self._mutation_rate = mutation_rate
        self._crossover_rate = crossover_rate
        self._seed = seed or 42
        self._data_dir = data_dir
        self._rng = random.Random(self._seed)

        self._islands: List[StrategyIsland] = [
            StrategyIsland(island_id=i, population_size=island_size, seed=self._seed + i)
            for i in range(self._n_islands)
        ]

        self._map_elites = MAPElitesGrid(
            dimensions=self._BEHAVIORAL_DESCRIPTOR_DIMS,
            bins_per_dim=6,
            bounds={
                "exploration_tendency": (0.0, 1.0),
                "response_specificity": (0.0, 1.0),
                "confidence_level": (0.0, 1.0),
            },
        )

        self._fitness_eval = FitnessEvaluator()
        self._mutation_op = MutationOperator(seed=self._seed)
        self._es_sampler = ESAntitheticSampler(seed=self._seed)

        self._current_best: Optional[StrategyGenome] = None
        self._generation = 0
        self._total_evaluations = 0
        self._convergence_state = ConvergenceState.EXPLORING
        self._best_fitness_history: List[float] = []

        self._lock = threading.Lock()
        self._stats_lock = threading.Lock()
        self._stats = {
            "total_cycles": 0,
            "total_improvements": 0,
            "total_evaluations": 0,
            "best_fitness_ever": 0.0,
            "avg_fitness_last_10": 0.0,
            "convergence_state": ConvergenceState.EXPLORING.value,
            "map_elites_coverage": 0.0,
            "map_elites_qd_score": 0.0,
        }

        self._initialize_population()

    def _initialize_population(self) -> None:
        for island in self._islands:
            for _ in range(self._island_size):
                params = {}
                for key, default in self._DEFAULT_PARAMS.items():
                    lb, ub = self._PARAM_BOUNDS.get(key, (0.0, 1.0))
                    params[key] = lb + self._rng.random() * (ub - lb)
                descriptors = {
                    dim: self._rng.random() for dim in self._BEHAVIORAL_DESCRIPTOR_DIMS
                }
                genome = StrategyGenome(
                    generation=0,
                    island_id=island.island_id,
                    parameters=params,
                    behavioral_descriptors=descriptors,
                    fitness_scores={dim.value: 0.5 for dim in FitnessDimension},
                    combined_fitness=0.5,
                    origin="random_init",
                )
                island.add_genome(genome)
                self._map_elites.add(genome)

    def evolve_cycle(
        self,
        outcomes: Optional[Dict[str, Dict[str, float]]] = None,
        strategy: Optional[EvolutionStrategy] = None,
    ) -> EvolutionCycle:
        start_time = time.monotonic()
        self._generation += 1

        if strategy is None:
            if self._convergence_state == ConvergenceState.STAGNANT:
                strategy = EvolutionStrategy.ES_ANTITHETIC
            elif self._convergence_state == ConvergenceState.CONVERGED:
                strategy = EvolutionStrategy.CATASTROPHIC
            elif self._rng.random() < self._mutation_rate:
                strategy = EvolutionStrategy.MAP_ELITES
            else:
                strategy = EvolutionStrategy.TOURNAMENT

        cycle = EvolutionCycle(
            generation=self._generation,
            strategy=strategy.value,
        )

        total_fitness = 0.0
        new_genomes = 0
        total_pop = 0

        for island in self._islands:
            if strategy == EvolutionStrategy.MAP_ELITES:
                new_genome = self._evolve_map_elites(island, outcomes)
            elif strategy == EvolutionStrategy.ES_ANTITHETIC:
                new_genome = self._evolve_es_antithetic(island, outcomes)
            elif strategy == EvolutionStrategy.CATASTROPHIC:
                new_genome = self._evolve_catastrophic(island)
            elif strategy == EvolutionStrategy.RANDOM_SEARCH:
                new_genome = self._evolve_random(island)
            else:
                new_genome = self._evolve_tournament(island, outcomes)

            if new_genome:
                island.add_genome(new_genome)
                self._map_elites.add(new_genome)
                new_genomes += 1
                total_fitness += new_genome.combined_fitness

            best = island.get_best()
            if best:
                total_fitness += best.combined_fitness
            total_pop += len(island.get_population_snapshot())

        if self._generation % self._migration_interval == 0:
            self._migrate()

        for island in self._islands:
            island.cull()

        best_overall = self.get_best()
        if best_overall:
            self._current_best = best_overall
            self._best_fitness_history.append(best_overall.combined_fitness)
            if len(self._best_fitness_history) > 1000:
                self._best_fitness_history = self._best_fitness_history[-1000:]

        self._update_convergence()

        cycle.population_size = total_pop
        cycle.best_fitness = best_overall.combined_fitness if best_overall else 0.0
        cycle.avg_fitness = round(total_fitness / max(total_pop, 1), 4)
        cycle.diversity_score = self.get_diversity()
        cycle.new_genomes = new_genomes
        cycle.convergence_state = self._convergence_state.value
        cycle.elapsed_ms = round((time.monotonic() - start_time) * 1000, 2)

        for island in self._islands:
            island.record_cycle(cycle)

        with self._stats_lock:
            self._stats["total_cycles"] += 1
            self._stats["total_improvements"] += new_genomes
            self._stats["best_fitness_ever"] = max(
                self._stats["best_fitness_ever"],
                best_overall.combined_fitness if best_overall else 0.0,
            )
            recent = self._best_fitness_history[-10:]
            self._stats["avg_fitness_last_10"] = round(
                sum(recent) / max(len(recent), 1), 4
            )
            self._stats["convergence_state"] = self._convergence_state.value
            self._stats["map_elites_coverage"] = self._map_elites.coverage()
            self._stats["map_elites_qd_score"] = self._map_elites.qd_score()

        return cycle

    def _evolve_tournament(
        self,
        island: StrategyIsland,
        outcomes: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[StrategyGenome]:
        parent_a = island.select_parent(tournament_size=max(3, island.population_size // 4))
        parent_b = island.select_parent(tournament_size=max(3, island.population_size // 4))
        if not parent_a or not parent_b:
            return None

        child_params = self._mutation_op.crossover(
            parent_a.parameters, parent_b.parameters, alpha=self._crossover_rate
        )
        child_params = self._mutation_op.gaussian_mutate(
            child_params, sigma=self._get_adaptive_sigma(), mutation_rate=self._mutation_rate,
            bounds=self._PARAM_BOUNDS,
        )

        descriptors = {
            dim: (parent_a.behavioral_descriptors.get(dim, 0.5)
                  + parent_b.behavioral_descriptors.get(dim, 0.5)) / 2.0
                  + self._rng.gauss(0, 0.05)
            for dim in self._BEHAVIORAL_DESCRIPTOR_DIMS
        }
        for dim in descriptors:
            descriptors[dim] = max(0.0, min(1.0, descriptors[dim]))

        fitness_scores = {}
        if outcomes:
            iso_outcomes = outcomes.get(str(island.island_id), outcomes.get("default", {}))
            fitness_scores = self._fitness_eval.compute_fitness(parent_a, iso_outcomes)
        else:
            fitness_scores = dict(parent_a.fitness_scores)

        combined = self._fitness_eval.combined_fitness(fitness_scores)

        return StrategyGenome(
            parent_ids=[parent_a.genome_id, parent_b.genome_id],
            generation=self._generation,
            island_id=island.island_id,
            parameters=child_params,
            behavioral_descriptors=descriptors,
            fitness_scores=fitness_scores,
            combined_fitness=combined,
            evaluation_count=1,
            origin="tournament_crossover",
        )

    def _evolve_map_elites(
        self,
        island: StrategyIsland,
        outcomes: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[StrategyGenome]:
        elite = self._map_elites.get_random_elite(self._rng)
        if not elite:
            elite = island.select_parent()
        if not elite:
            return None

        child_params = self._mutation_op.gaussian_mutate(elite.parameters, self._get_adaptive_sigma(),
                                                          self._mutation_rate, self._PARAM_BOUNDS)

        descriptors = {
            dim: elite.behavioral_descriptors.get(dim, 0.5) + self._rng.gauss(0, 0.05)
            for dim in self._BEHAVIORAL_DESCRIPTOR_DIMS
        }
        for dim in descriptors:
            descriptors[dim] = max(0.0, min(1.0, descriptors[dim]))

        fitness_scores = {}
        if outcomes:
            iso_outcomes = outcomes.get(str(island.island_id), outcomes.get("default", {}))
            fitness_scores = self._fitness_eval.compute_fitness(elite, iso_outcomes)
        else:
            fitness_scores = (
                dict(elite.fitness_scores) if elite.fitness_scores else
                {dim.value: 0.5 for dim in FitnessDimension}
            )

        combined = self._fitness_eval.combined_fitness(fitness_scores)

        return StrategyGenome(
            parent_ids=[elite.genome_id],
            generation=self._generation,
            island_id=island.island_id,
            parameters=child_params,
            behavioral_descriptors=descriptors,
            fitness_scores=fitness_scores,
            combined_fitness=combined,
            evaluation_count=1,
            origin="map_elites",
        )

    def _evolve_es_antithetic(
        self,
        island: StrategyIsland,
        outcomes: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> Optional[StrategyGenome]:
        base = island.get_best()
        if not base:
            base = island.select_parent()
        if not base:
            return None

        perturbation_pairs = self._es_sampler.generate_perturbations(base.parameters)
        if not perturbation_pairs:
            return None

        plus_params, minus_params = perturbation_pairs[0]

        child_plus = StrategyGenome(parameters=plus_params, parent_ids=[base.genome_id])
        child_minus = StrategyGenome(parameters=minus_params, parent_ids=[base.genome_id])
        # `outcomes` is optional — callers may omit it. Coerce to empty dict so
        # the .get chain works either way.
        outcomes = outcomes or {}
        outcome_for_island = outcomes.get(str(island.island_id)) or outcomes.get("default") or {}
        plus_scores = self._fitness_eval.compute_fitness(child_plus, outcome_for_island)
        minus_scores = self._fitness_eval.compute_fitness(child_minus, outcome_for_island)
        plus_fitness = self._fitness_eval.combined_fitness(plus_scores)
        minus_fitness = self._fitness_eval.combined_fitness(minus_scores)

        better_params = plus_params if plus_fitness >= minus_fitness else minus_params
        for key in better_params:
            if key in self._PARAM_BOUNDS:
                lb, ub = self._PARAM_BOUNDS[key]
                better_params[key] = max(lb, min(ub, better_params[key]))

        descriptors = {
            dim: base.behavioral_descriptors.get(dim, 0.5) + self._rng.gauss(0, 0.03)
            for dim in self._BEHAVIORAL_DESCRIPTOR_DIMS
        }
        for dim in descriptors:
            descriptors[dim] = max(0.0, min(1.0, descriptors[dim]))

        fitness_scores = dict(base.fitness_scores) if base.fitness_scores else {}
        if outcomes:
            iso_outcomes = outcomes.get(str(island.island_id), outcomes.get("default", {}))
            computed = self._fitness_eval.compute_fitness(base, iso_outcomes)
            fitness_scores.update(computed)

        combined = self._fitness_eval.combined_fitness(fitness_scores)

        success = combined > base.combined_fitness
        self._es_sampler.update_sigma(1.0 if success else 0.0)

        return StrategyGenome(
            parent_ids=[base.genome_id],
            generation=self._generation,
            island_id=island.island_id,
            parameters=better_params,
            behavioral_descriptors=descriptors,
            fitness_scores=fitness_scores,
            combined_fitness=combined,
            evaluation_count=1,
            origin="es_antithetic",
        )

    def _evolve_catastrophic(self, island: StrategyIsland) -> Optional[StrategyGenome]:
        parent = island.select_parent()
        if not parent:
            return None

        child_params = self._mutation_op.catastrophic_mutate(
            parent.parameters, sigma=0.5, mutation_rate=0.8, bounds=self._PARAM_BOUNDS,
        )

        descriptors = {
            dim: self._rng.random() for dim in self._BEHAVIORAL_DESCRIPTOR_DIMS
        }

        fitness_scores = {dim.value: self._rng.random() * 0.3 + 0.2 for dim in FitnessDimension}
        combined = self._fitness_eval.combined_fitness(fitness_scores)

        return StrategyGenome(
            parent_ids=[parent.genome_id],
            generation=self._generation,
            island_id=island.island_id,
            parameters=child_params,
            behavioral_descriptors=descriptors,
            fitness_scores=fitness_scores,
            combined_fitness=combined,
            evaluation_count=1,
            origin="catastrophic_reset",
        )

    def _evolve_random(self, island: StrategyIsland) -> Optional[StrategyGenome]:
        params = {}
        for key, default in self._DEFAULT_PARAMS.items():
            lb, ub = self._PARAM_BOUNDS.get(key, (0.0, 1.0))
            params[key] = lb + self._rng.random() * (ub - lb)

        descriptors = {dim: self._rng.random() for dim in self._BEHAVIORAL_DESCRIPTOR_DIMS}
        fitness_scores = {dim.value: self._rng.random() for dim in FitnessDimension}
        combined = self._fitness_eval.combined_fitness(fitness_scores)

        return StrategyGenome(
            generation=self._generation,
            island_id=island.island_id,
            parameters=params,
            behavioral_descriptors=descriptors,
            fitness_scores=fitness_scores,
            combined_fitness=combined,
            evaluation_count=1,
            origin="random_search",
        )

    def _migrate(self) -> None:
        migrants_per_island = max(1, self._island_size // 5)
        for i, island in enumerate(self._islands):
            source = island
            target = self._islands[(i + 1) % self._n_islands]
            for _ in range(migrants_per_island):
                migrant = source.select_parent()
                if migrant:
                    migrant_copy = copy.deepcopy(migrant)
                    migrant_copy.island_id = target.island_id
                    migrant_copy.origin = "migration"
                    target.add_genome(migrant_copy)

    def _get_adaptive_sigma(self) -> float:
        if self._convergence_state == ConvergenceState.EXPLORING:
            return 0.15
        elif self._convergence_state == ConvergenceState.EXPLOITING:
            return 0.05
        elif self._convergence_state == ConvergenceState.STAGNANT:
            return 0.3
        elif self._convergence_state == ConvergenceState.CONVERGED:
            return 0.02
        return 0.1

    def _update_convergence(self) -> None:
        if len(self._best_fitness_history) < 5:
            self._convergence_state = ConvergenceState.EXPLORING
            return

        recent = self._best_fitness_history[-5:]
        improvement = max(recent) - min(recent)

        if improvement < self._STAGNATION_THRESHOLD * 0.1:
            if len(self._best_fitness_history) > self._CONVERGENCE_WINDOW:
                longer = self._best_fitness_history[-self._CONVERGENCE_WINDOW:]
                longer_improvement = max(longer) - min(longer)
                if longer_improvement < self._STAGNATION_THRESHOLD:
                    self._convergence_state = ConvergenceState.CONVERGED
                    return
            self._convergence_state = ConvergenceState.EXPLOITING
        elif improvement < self._STAGNATION_THRESHOLD:
            self._convergence_state = ConvergenceState.STAGNANT
        else:
            self._convergence_state = ConvergenceState.EXPLORING

    def get_best(self) -> Optional[StrategyGenome]:
        best = None
        for island in self._islands:
            island_best = island.get_best()
            if island_best and (best is None or island_best.combined_fitness > best.combined_fitness):
                best = island_best
        if best:
            return copy.deepcopy(best)
        return None

    def get_best_parameters(self) -> Dict[str, float]:
        best = self.get_best()
        if best:
            return dict(best.parameters)
        return dict(self._DEFAULT_PARAMS)

    def get_diversity(self) -> float:
        total = 0.0
        count = 0
        for island in self._islands:
            total += island.get_diversity()
            count += 1
        return round(total / max(count, 1), 4)

    def record_outcome(
        self,
        genome_id: str,
        outcomes: Dict[str, float],
    ) -> None:
        self._total_evaluations += 1
        for island in self._islands:
            pop = island.get_population_snapshot()
            for genome in pop:
                if genome.genome_id == genome_id:
                    updated_scores = self._fitness_eval.compute_fitness(genome, outcomes)
                    genome.fitness_scores = updated_scores
                    genome.combined_fitness = self._fitness_eval.combined_fitness(updated_scores)
                    genome.evaluation_count += 1
                    island.add_genome(genome)
                    self._map_elites.add(genome)
                    break

    def get_convergence_state(self) -> str:
        return self._convergence_state.value

    def get_stats(self) -> Dict[str, Any]:
        with self._stats_lock:
            stats = dict(self._stats)
        stats["generation"] = self._generation
        stats["n_islands"] = self._n_islands
        stats["convergence_state"] = self._convergence_state.value
        stats["total_evaluations"] = self._total_evaluations
        stats["map_elites_coverage"] = self._map_elites.coverage()
        stats["map_elites_qd_score"] = self._map_elites.qd_score()
        stats["es_sigma"] = self._es_sampler.sigma
        stats["island_diversity"] = self.get_diversity()
        best = self.get_best()
        if best:
            stats["best_fitness"] = round(best.combined_fitness, 4)
            stats["best_genome_id"] = best.genome_id
            stats["best_origin"] = best.origin
        return stats

    def get_population_size(self) -> int:
        total = 0
        for island in self._islands:
            total += len(island.get_population_snapshot())
        return total