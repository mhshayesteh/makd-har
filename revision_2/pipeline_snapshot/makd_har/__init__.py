"""Reproducible multi-agent knowledge-distillation experiments for HAR."""

from .config import ExperimentConfig
from .data import DatasetBundle, load_uci_har, load_wisdm_v1
from .shapley import (
    ShapleyResult,
    exact_shapley,
    permutation_shapley,
    shapley_to_weights,
    standardize_shapley,
    weight_diagnostics,
)

__all__ = [
    "DatasetBundle",
    "ExperimentConfig",
    "ShapleyResult",
    "exact_shapley",
    "load_uci_har",
    "load_wisdm_v1",
    "permutation_shapley",
    "shapley_to_weights",
    "standardize_shapley",
    "weight_diagnostics",
]
