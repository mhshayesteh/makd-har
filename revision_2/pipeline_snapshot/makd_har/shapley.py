from __future__ import annotations

import itertools
import math
import time
from dataclasses import dataclass
from typing import Callable, Iterable

import numpy as np

Coalition = tuple[int, ...]
ValueFunction = Callable[[Coalition], float]


@dataclass(frozen=True)
class ShapleyResult:
    values: np.ndarray
    coalition_values: dict[Coalition, float]
    evaluations: int
    elapsed_seconds: float
    mode: str
    permutations: int | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "values": self.values.tolist(),
            "coalition_values": {
                ",".join(map(str, key)): value
                for key, value in sorted(self.coalition_values.items())
            },
            "evaluations": self.evaluations,
            "elapsed_seconds": self.elapsed_seconds,
            "mode": self.mode,
            "permutations": self.permutations,
        }


def _canonical(coalition: Iterable[int]) -> Coalition:
    return tuple(sorted(set(int(member) for member in coalition)))


def _cached_value(
    coalition: Iterable[int], value_fn: ValueFunction, cache: dict[Coalition, float]
) -> float:
    key = _canonical(coalition)
    if key not in cache:
        value = float(value_fn(key))
        if not np.isfinite(value):
            raise ValueError(f"Non-finite coalition value for {key}: {value}")
        cache[key] = value
    return cache[key]


def exact_shapley(num_players: int, value_fn: ValueFunction) -> ShapleyResult:
    """Compute exact Shapley values using all 2**K coalitions."""

    if num_players < 1:
        raise ValueError("num_players must be positive")
    started = time.perf_counter()
    cache: dict[Coalition, float] = {}
    values = np.zeros(num_players, dtype=np.float64)
    players = tuple(range(num_players))
    denominator = math.factorial(num_players)

    for player in players:
        others = tuple(p for p in players if p != player)
        for size in range(len(others) + 1):
            coefficient = (
                math.factorial(size)
                * math.factorial(num_players - size - 1)
                / denominator
            )
            for subset in itertools.combinations(others, size):
                without = _cached_value(subset, value_fn, cache)
                with_player = _cached_value((*subset, player), value_fn, cache)
                values[player] += coefficient * (with_player - without)

    return ShapleyResult(
        values=values,
        coalition_values=cache,
        evaluations=len(cache),
        elapsed_seconds=time.perf_counter() - started,
        mode="exact",
    )


def permutation_shapley(
    num_players: int,
    value_fn: ValueFunction,
    permutations: int,
    seed: int,
) -> ShapleyResult:
    """Monte Carlo Shapley estimate from random player permutations.

    Coalition scores are cached. ``evaluations`` therefore reports the actual
    number of student retrainings, not merely the number of marginal terms.
    """

    if num_players < 1 or permutations < 1:
        raise ValueError("num_players and permutations must be positive")
    started = time.perf_counter()
    rng = np.random.default_rng(seed)
    cache: dict[Coalition, float] = {}
    values = np.zeros(num_players, dtype=np.float64)

    for _ in range(permutations):
        order = rng.permutation(num_players).tolist()
        coalition: list[int] = []
        previous = _cached_value(coalition, value_fn, cache)
        for player in order:
            coalition.append(int(player))
            current = _cached_value(coalition, value_fn, cache)
            values[player] += current - previous
            previous = current

    values /= permutations
    return ShapleyResult(
        values=values,
        coalition_values=cache,
        evaluations=len(cache),
        elapsed_seconds=time.perf_counter() - started,
        mode="permutation",
        permutations=permutations,
    )


def standardize_shapley(values: np.ndarray) -> np.ndarray:
    """Standardize contributions so weighting is independent of metric units.

    Coalition utilities such as macro-F1 normally occupy [0, 1], making raw
    Shapley differences numerically small. Standardizing prevents a softmax
    temperature of one from collapsing meaningful relative contributions into
    an almost-uniform distribution. Equal contributions remain uniform.
    """

    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or len(values) == 0 or not np.all(np.isfinite(values)):
        raise ValueError("values must be a finite non-empty vector")
    standard_deviation = float(values.std())
    if standard_deviation <= 1e-12:
        return np.zeros_like(values)
    return (values - values.mean()) / standard_deviation


def shapley_to_weights(values: np.ndarray, temperature: float = 1.0) -> np.ndarray:
    """Map standardized Shapley values to stable non-negative weights."""

    if temperature <= 0:
        raise ValueError("temperature must be positive")
    scaled = standardize_shapley(values) / temperature
    scaled -= np.max(scaled)
    weights = np.exp(scaled)
    return weights / weights.sum()


def weight_diagnostics(weights: np.ndarray) -> dict[str, object]:
    """Quantify how far contribution weights depart from uniform averaging."""

    weights = np.asarray(weights, dtype=np.float64)
    if weights.ndim != 1 or len(weights) == 0 or np.any(weights < 0):
        raise ValueError("weights must be a non-negative vector")
    weights = weights / weights.sum()
    uniform = np.full(len(weights), 1.0 / len(weights), dtype=np.float64)
    l1_distance = float(np.abs(weights - uniform).sum())
    max_difference = float(np.abs(weights - uniform).max())
    entropy = float(-(weights * np.log(np.maximum(weights, 1e-15))).sum())
    normalized_entropy = entropy / float(np.log(len(weights))) if len(weights) > 1 else 1.0
    near_uniform = l1_distance < 0.10 or max_difference < 0.02
    return {
        "uniform_weights": uniform.tolist(),
        "l1_distance_from_uniform": l1_distance,
        "max_absolute_difference_from_uniform": max_difference,
        "effective_teacher_count": float(1.0 / np.square(weights).sum()),
        "normalized_entropy": normalized_entropy,
        "near_uniform": near_uniform,
        "warning": (
            "Shapley weights are nearly uniform; do not claim an adaptive-weighting benefit "
            "unless repeated experiments distinguish it from uniform aggregation."
            if near_uniform
            else None
        ),
    }
