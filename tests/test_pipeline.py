from __future__ import annotations

import numpy as np

from makd_har.data import _group_holdout, extract_wisdm_features
from makd_har.shapley import (
    exact_shapley,
    permutation_shapley,
    shapley_to_weights,
    standardize_shapley,
    weight_diagnostics,
)


def test_exact_shapley_recovers_additive_values() -> None:
    contributions = np.asarray([0.1, 0.2, -0.05, 0.4])

    def value(coalition: tuple[int, ...]) -> float:
        return float(contributions[list(coalition)].sum()) if coalition else 0.0

    result = exact_shapley(len(contributions), value)
    np.testing.assert_allclose(result.values, contributions, atol=1e-12)
    assert result.evaluations == 2 ** len(contributions)


def test_permutation_shapley_is_exact_for_additive_game() -> None:
    contributions = np.asarray([0.3, -0.1, 0.2])

    def value(coalition: tuple[int, ...]) -> float:
        return float(contributions[list(coalition)].sum()) if coalition else 0.0

    result = permutation_shapley(3, value, permutations=20, seed=7)
    np.testing.assert_allclose(result.values, contributions, atol=1e-12)


def test_softmax_weights_are_valid_with_negative_shapley_values() -> None:
    weights = shapley_to_weights(np.asarray([-2.0, 0.0, 1.0]), temperature=0.7)
    assert np.all(weights > 0)
    np.testing.assert_allclose(weights.sum(), 1.0)
    assert weights.argmax() == 2


def test_shapley_weights_are_invariant_to_positive_affine_scaling() -> None:
    values = np.asarray([-0.03, 0.005, 0.008, -0.004, 0.006])
    original = shapley_to_weights(values)
    rescaled = shapley_to_weights(100.0 * values + 17.0)
    np.testing.assert_allclose(original, rescaled, atol=1e-12)
    assert weight_diagnostics(original)["near_uniform"] is False


def test_equal_shapley_values_produce_uniform_weights() -> None:
    values = np.asarray([0.2, 0.2, 0.2])
    np.testing.assert_allclose(standardize_shapley(values), np.zeros(3))
    weights = shapley_to_weights(values)
    np.testing.assert_allclose(weights, np.full(3, 1 / 3))
    assert weight_diagnostics(weights)["near_uniform"] is True


def test_group_holdout_is_subject_disjoint_and_class_complete() -> None:
    subjects = np.repeat(np.arange(1, 11), 6)
    labels = np.tile(np.arange(6), 10)
    train, holdout = _group_holdout(labels, subjects, fraction=0.2, seed=2026)
    assert not set(subjects[train]) & set(subjects[holdout])
    assert set(labels[train]) == set(range(6))
    assert set(labels[holdout]) == set(range(6))


def test_wisdm_features_are_real_and_finite() -> None:
    t = np.arange(128) / 20.0
    window = np.stack(
        [np.sin(2 * np.pi * t), np.cos(2 * np.pi * 2 * t), 0.1 * t], axis=1
    ).astype(np.float32)
    values, names = extract_wisdm_features(window, sample_rate_hz=20.0)
    assert len(values) == len(names) == 139
    assert np.all(np.isfinite(values))
    assert "x_dominant_frequency_hz" in names
    assert "corr_x_y" in names
