from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any


DEFAULT_SEEDS = (11, 23, 37, 53, 71, 89, 107, 131, 167, 197)


@dataclass(frozen=True)
class ExperimentConfig:
    """All choices that can affect an experiment result.

    The dataset split seed is deliberately separate from model seeds. The test
    partition stays fixed while initialization and minibatch order vary.
    """

    split_seed: int = 2026
    model_seeds: tuple[int, ...] = DEFAULT_SEEDS
    validation_fraction: float = 0.20
    wisdm_test_fraction: float = 0.20
    batch_size: int = 128
    teacher_epochs: int = 60
    student_epochs: int = 60
    coalition_epochs: int = 20
    patience: int = 10
    learning_rate: float = 1e-3
    temperature: float = 4.0
    hard_label_weight: float = 0.5
    shapley_softmax_temperature: float = 1.0
    shapley_mode: str = "exact"
    shapley_permutations: int = 64
    latency_warmups: int = 30
    latency_repetitions: int = 300
    teacher_names: tuple[str, ...] = field(
        default_factory=lambda: ("hcf_mlp", "raw_cnn", "lstm", "tcn", "transformer")
    )

    def validate(self) -> None:
        if not 0 < self.validation_fraction < 0.5:
            raise ValueError("validation_fraction must be between 0 and 0.5")
        if not 0 < self.wisdm_test_fraction < 0.5:
            raise ValueError("wisdm_test_fraction must be between 0 and 0.5")
        if not 0 <= self.hard_label_weight <= 1:
            raise ValueError("hard_label_weight must be in [0, 1]")
        if self.temperature <= 0 or self.shapley_softmax_temperature <= 0:
            raise ValueError("temperatures must be positive")
        if self.shapley_mode not in {"exact", "permutation"}:
            raise ValueError("shapley_mode must be 'exact' or 'permutation'")
        if len(self.teacher_names) < 3:
            raise ValueError("At least three teachers are required for the revised study")

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
