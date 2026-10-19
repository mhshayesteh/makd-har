from __future__ import annotations

from dataclasses import dataclass, field
from collections import Counter
from pathlib import Path
from typing import Iterable

import numpy as np
from .reproducibility import file_sha256


UCI_SIGNAL_NAMES = (
    "body_acc_x",
    "body_acc_y",
    "body_acc_z",
    "body_gyro_x",
    "body_gyro_y",
    "body_gyro_z",
    "total_acc_x",
    "total_acc_y",
    "total_acc_z",
)

WISDM_LABELS = {
    "Walking": 0,
    "Jogging": 1,
    "Sitting": 2,
    "Standing": 3,
    "Upstairs": 4,
    "Downstairs": 5,
}


@dataclass(frozen=True)
class DatasetBundle:
    name: str
    raw_train: np.ndarray
    raw_val: np.ndarray
    raw_test: np.ndarray
    features_train: np.ndarray
    features_val: np.ndarray
    features_test: np.ndarray
    y_train: np.ndarray
    y_val: np.ndarray
    y_test: np.ndarray
    subjects_train: np.ndarray
    subjects_val: np.ndarray
    subjects_test: np.ndarray
    class_names: tuple[str, ...]
    feature_names: tuple[str, ...]
    split_description: str
    preprocessing: dict[str, object] = field(default_factory=dict)

    def validate(self) -> None:
        partitions = (
            ("train", self.raw_train, self.features_train, self.y_train, self.subjects_train),
            ("val", self.raw_val, self.features_val, self.y_val, self.subjects_val),
            ("test", self.raw_test, self.features_test, self.y_test, self.subjects_test),
        )
        expected_classes = set(range(len(self.class_names)))
        subject_sets: dict[str, set[int]] = {}
        for name, raw, features, labels, subjects in partitions:
            if not (len(raw) == len(features) == len(labels) == len(subjects)):
                raise ValueError(f"{name} arrays have inconsistent lengths")
            if set(np.unique(labels).astype(int)) != expected_classes:
                raise ValueError(f"{name} does not contain every class")
            if not np.all(np.isfinite(raw)) or not np.all(np.isfinite(features)):
                raise ValueError(f"{name} contains NaN or infinity")
            subject_sets[name] = set(np.unique(subjects).astype(int))
        if subject_sets["train"] & subject_sets["val"]:
            raise ValueError("Subject leakage between train and validation")
        if subject_sets["train"] & subject_sets["test"]:
            raise ValueError("Subject leakage between train and test")
        if subject_sets["val"] & subject_sets["test"]:
            raise ValueError("Subject leakage between validation and test")

    def manifest(self) -> dict[str, object]:
        def part(labels: np.ndarray, subjects: np.ndarray) -> dict[str, object]:
            values, counts = np.unique(labels, return_counts=True)
            return {
                "samples": int(len(labels)),
                "subjects": np.unique(subjects).astype(int).tolist(),
                "class_counts": {str(int(v)): int(c) for v, c in zip(values, counts)},
            }

        result = {
            "dataset": self.name,
            "raw_shape": list(self.raw_train.shape[1:]),
            "feature_count": int(self.features_train.shape[1]),
            "feature_names": list(self.feature_names),
            "split_description": self.split_description,
            "train": part(self.y_train, self.subjects_train),
            "validation": part(self.y_val, self.subjects_val),
            "test": part(self.y_test, self.subjects_test),
        }
        if self.preprocessing:
            result["preprocessing"] = self.preprocessing
        return result


def _resolve_uci_root(root: str | Path) -> Path:
    root = Path(root)
    candidates = (root, root / "UCI HAR Dataset", root / "Dataset" / "UCI HAR Dataset")
    for candidate in candidates:
        if (candidate / "train" / "X_train.txt").exists():
            return candidate
    raise FileNotFoundError(
        f"Could not find UCI HAR Dataset under {root}. Expected train/X_train.txt."
    )


def _load_uci_partition(root: Path, partition: str) -> tuple[np.ndarray, ...]:
    base = root / partition
    inertial = base / "Inertial Signals"
    suffix = "train" if partition == "train" else "test"
    raw = np.stack(
        [np.loadtxt(inertial / f"{name}_{suffix}.txt") for name in UCI_SIGNAL_NAMES],
        axis=-1,
    ).astype(np.float32)
    features = np.loadtxt(base / f"X_{suffix}.txt").astype(np.float32)
    labels = np.loadtxt(base / f"y_{suffix}.txt").astype(np.int64) - 1
    subjects = np.loadtxt(base / f"subject_{suffix}.txt").astype(np.int64)
    return raw, features, labels, subjects


def _class_complete(labels: np.ndarray, num_classes: int) -> bool:
    return set(np.unique(labels).astype(int)) == set(range(num_classes))


def _group_holdout(
    labels: np.ndarray,
    subjects: np.ndarray,
    fraction: float,
    seed: int,
    attempts: int = 1000,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a deterministic subject-disjoint split with all classes on both sides."""

    unique_subjects = np.unique(subjects)
    holdout_count = max(1, int(round(len(unique_subjects) * fraction)))
    num_classes = len(np.unique(labels))
    for offset in range(attempts):
        rng = np.random.default_rng(seed + offset)
        shuffled = rng.permutation(unique_subjects)
        held_out = set(shuffled[:holdout_count].tolist())
        holdout_mask = np.fromiter((s in held_out for s in subjects), dtype=bool)
        train_mask = ~holdout_mask
        if _class_complete(labels[train_mask], num_classes) and _class_complete(
            labels[holdout_mask], num_classes
        ):
            return np.flatnonzero(train_mask), np.flatnonzero(holdout_mask)
    raise RuntimeError("Could not find a class-complete subject-disjoint split")


def _fit_raw_standardizer(raw: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    axes = tuple(range(raw.ndim - 1))
    mean = raw.mean(axis=axes, keepdims=True)
    std = raw.std(axis=axes, keepdims=True)
    return mean, np.maximum(std, 1e-7)


def _fit_feature_standardizer(features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    mean = features.mean(axis=0, keepdims=True)
    std = features.std(axis=0, keepdims=True)
    return mean, np.maximum(std, 1e-7)


def load_uci_har(
    root: str | Path,
    validation_fraction: float = 0.15,
    split_seed: int = 2026,
) -> DatasetBundle:
    """Load UCI-HAR without mixing its official test subjects into training."""

    root = _resolve_uci_root(root)
    raw_official_train, feat_official_train, y_official_train, s_official_train = (
        _load_uci_partition(root, "train")
    )
    raw_test, feat_test, y_test, s_test = _load_uci_partition(root, "test")
    train_idx, val_idx = _group_holdout(
        y_official_train, s_official_train, validation_fraction, split_seed
    )

    raw_mean, raw_std = _fit_raw_standardizer(raw_official_train[train_idx])
    feat_mean, feat_std = _fit_feature_standardizer(feat_official_train[train_idx])

    bundle = DatasetBundle(
        name="UCI-HAR",
        raw_train=(raw_official_train[train_idx] - raw_mean) / raw_std,
        raw_val=(raw_official_train[val_idx] - raw_mean) / raw_std,
        raw_test=(raw_test - raw_mean) / raw_std,
        features_train=(feat_official_train[train_idx] - feat_mean) / feat_std,
        features_val=(feat_official_train[val_idx] - feat_mean) / feat_std,
        features_test=(feat_test - feat_mean) / feat_std,
        y_train=y_official_train[train_idx],
        y_val=y_official_train[val_idx],
        y_test=y_test,
        subjects_train=s_official_train[train_idx],
        subjects_val=s_official_train[val_idx],
        subjects_test=s_test,
        class_names=(
            "walking",
            "walking_upstairs",
            "walking_downstairs",
            "sitting",
            "standing",
            "laying",
        ),
        feature_names=tuple(f"uci_feature_{i + 1}" for i in range(feat_official_train.shape[1])),
        split_description=(
            "Official UCI subject-disjoint test partition retained; validation subjects "
            "held out only from the official training partition; scalers fitted on train only."
        ),
    )
    bundle.validate()
    return bundle


def _spectral_entropy(power: np.ndarray) -> float:
    total = float(power.sum())
    if total <= 0:
        return 0.0
    probability = power / total
    probability = probability[probability > 0]
    return float(-(probability * np.log2(probability)).sum())


def _signal_features(signal: np.ndarray, sample_rate_hz: float) -> list[float]:
    signal = np.asarray(signal, dtype=np.float64)
    centered = signal - signal.mean()
    std = float(signal.std())
    normalized = centered / max(std, 1e-12)
    spectrum = np.fft.rfft(centered)
    power = np.abs(spectrum) ** 2
    frequencies = np.fft.rfftfreq(len(signal), d=1.0 / sample_rate_hz)
    power_sum = float(power.sum())
    centroid = float((frequencies * power).sum() / max(power_sum, 1e-12))
    dominant = float(frequencies[int(np.argmax(power))]) if len(power) else 0.0
    return [
        float(signal.mean()),
        std,
        float(signal.min()),
        float(signal.max()),
        float(np.median(signal)),
        float(np.mean(np.abs(centered))),
        float(np.sqrt(np.mean(signal**2))),
        float(np.mean(signal**2)),
        float(np.percentile(signal, 75) - np.percentile(signal, 25)),
        float(np.mean(normalized**3)),
        float(np.mean(normalized**4) - 3.0),
        centroid,
        dominant,
        _spectral_entropy(power),
        float(power[frequencies < 0.5].sum()),
        float(power[(frequencies >= 0.5) & (frequencies < 3.0)].sum()),
        float(power[frequencies >= 3.0].sum()),
    ]


def extract_wisdm_features(
    window: np.ndarray, sample_rate_hz: float = 20.0
) -> tuple[np.ndarray, tuple[str, ...]]:
    """Extract genuine WISDM features; no artificial 561-dimensional padding."""

    axes = [window[:, index] for index in range(3)]
    magnitude = np.linalg.norm(window, axis=1)
    jerks = [np.diff(axis, prepend=axis[0]) * sample_rate_hz for axis in axes]
    jerk_magnitude = np.linalg.norm(np.stack(jerks, axis=1), axis=1)
    signals = (*axes, magnitude, *jerks, jerk_magnitude)
    signal_names = ("x", "y", "z", "magnitude", "jerk_x", "jerk_y", "jerk_z", "jerk_magnitude")
    statistic_names = (
        "mean",
        "std",
        "min",
        "max",
        "median",
        "mad",
        "rms",
        "energy",
        "iqr",
        "skew",
        "excess_kurtosis",
        "spectral_centroid_hz",
        "dominant_frequency_hz",
        "spectral_entropy",
        "power_0_0.5_hz",
        "power_0.5_3_hz",
        "power_3_nyquist_hz",
    )
    values: list[float] = []
    names: list[str] = []
    for signal_name, signal in zip(signal_names, signals):
        values.extend(_signal_features(signal, sample_rate_hz))
        names.extend(f"{signal_name}_{stat}" for stat in statistic_names)
    for left, right in ((0, 1), (0, 2), (1, 2)):
        correlation = np.corrcoef(window[:, left], window[:, right])[0, 1]
        values.append(float(np.nan_to_num(correlation)))
        names.append(f"corr_{'xyz'[left]}_{'xyz'[right]}")
    return np.asarray(values, dtype=np.float32), tuple(names)


def _iter_wisdm_segments(path: Path, audit: dict | None = None) -> Iterable[tuple[int, int, np.ndarray]]:
    """Repair record syntax and split reliable clock gaps over one second.

    Duplicate/invalid clocks retain samples in file order at nominal 20 Hz.
    Positive clocks moving backward also end a segment. Invalid sensor records
    break continuity rather than silently joining samples across missing data.
    """
    current_key: tuple[int, int] | None = None
    current_values: list[tuple[float, float, float]] = []
    previous_stamp = None
    counts = Counter()
    with path.open("r", encoding="utf-8", errors="ignore") as stream:
        for line in stream:
            for record in line.split(';'):
                record = record.strip()
                if not record:
                    continue
                counts['records_seen'] += 1
                parts = record.split(',')
                if len(parts) == 7 and not parts[-1].strip():
                    parts.pop()
                    counts['trailing_comma_repaired'] += 1
                try:
                    if len(parts) != 6 or parts[1] not in WISDM_LABELS:
                        raise ValueError('Malformed record')
                    key = (int(parts[0]), WISDM_LABELS[parts[1]])
                    value = tuple(float(x) for x in parts[3:6])
                    if not np.all(np.isfinite(value)):
                        raise ValueError('Nonfinite sensor')
                except ValueError:
                    counts['invalid_records'] += 1
                    if current_values:
                        counts['segments'] += 1
                        yield current_key[0], current_key[1], np.asarray(current_values, dtype=np.float32)
                    current_key, current_values, previous_stamp = None, [], None
                    continue
                try:
                    stamp = int(parts[2])
                    if stamp <= 0:
                        stamp = None
                except ValueError:
                    stamp = None
                if stamp is None:
                    counts['invalid_clock_records_retained'] += 1
                boundary = current_key is not None and key != current_key
                if key == current_key and stamp is not None and previous_stamp is not None:
                    delta = stamp - previous_stamp
                    if delta == 0:
                        counts['duplicate_clock_pairs_retained'] += 1
                    if delta < 0 or delta > 1_000_000_000:
                        counts['backward_clock_boundaries' if delta < 0 else 'long_gap_boundaries'] += 1
                        boundary = True
                if boundary and current_values:
                    counts['segments'] += 1
                    yield current_key[0], current_key[1], np.asarray(current_values, dtype=np.float32)
                    current_values = []
                current_key = key
                current_values.append(value)
                previous_stamp = stamp
                counts['accepted_records'] += 1
    if current_key is not None and current_values:
        counts['segments'] += 1
        yield current_key[0], current_key[1], np.asarray(current_values, dtype=np.float32)
    if audit is not None:
        audit.update(dict(counts))


def _three_way_subject_split(
    labels: np.ndarray,
    subjects: np.ndarray,
    validation_fraction: float,
    test_fraction: float,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    development_idx, test_idx = _group_holdout(labels, subjects, test_fraction, seed)
    relative_val_fraction = validation_fraction / (1.0 - test_fraction)
    train_rel, val_rel = _group_holdout(
        labels[development_idx], subjects[development_idx], relative_val_fraction, seed + 10000
    )
    return development_idx[train_rel], development_idx[val_rel], test_idx


def load_wisdm_v1(
    raw_file: str | Path,
    validation_fraction: float = 0.15,
    test_fraction: float = 0.20,
    split_seed: int = 2026,
    window_size: int = 128,
    overlap: int = 64,
    sample_rate_hz: float = 20.0,
) -> DatasetBundle:
    """Load classic WISDM v1.1 with subject-disjoint partitions."""

    raw_file = Path(raw_file)
    if not raw_file.exists():
        raise FileNotFoundError(raw_file)
    windows: list[np.ndarray] = []
    features: list[np.ndarray] = []
    labels: list[int] = []
    subjects: list[int] = []
    feature_names: tuple[str, ...] | None = None
    stride = window_size - overlap
    if stride < 1:
        raise ValueError("overlap must be smaller than window_size")

    parse_audit: dict[str, object] = {}
    for subject, label, segment in _iter_wisdm_segments(raw_file, parse_audit):
        for start in range(0, len(segment) - window_size + 1, stride):
            window = segment[start : start + window_size]
            feature_vector, names = extract_wisdm_features(window, sample_rate_hz)
            feature_names = names
            windows.append(window)
            features.append(feature_vector)
            labels.append(label)
            subjects.append(subject)

    raw = np.asarray(windows, dtype=np.float32)
    feature_matrix = np.asarray(features, dtype=np.float32)
    y = np.asarray(labels, dtype=np.int64)
    subject_array = np.asarray(subjects, dtype=np.int64)
    train_idx, val_idx, test_idx = _three_way_subject_split(
        y, subject_array, validation_fraction, test_fraction, split_seed
    )
    raw_mean, raw_std = _fit_raw_standardizer(raw[train_idx])
    feat_mean, feat_std = _fit_feature_standardizer(feature_matrix[train_idx])

    bundle = DatasetBundle(
        name="WISDM-v1.1",
        raw_train=(raw[train_idx] - raw_mean) / raw_std,
        raw_val=(raw[val_idx] - raw_mean) / raw_std,
        raw_test=(raw[test_idx] - raw_mean) / raw_std,
        features_train=(feature_matrix[train_idx] - feat_mean) / feat_std,
        features_val=(feature_matrix[val_idx] - feat_mean) / feat_std,
        features_test=(feature_matrix[test_idx] - feat_mean) / feat_std,
        y_train=y[train_idx],
        y_val=y[val_idx],
        y_test=y[test_idx],
        subjects_train=subject_array[train_idx],
        subjects_val=subject_array[val_idx],
        subjects_test=subject_array[test_idx],
        class_names=tuple(WISDM_LABELS.keys()),
        feature_names=feature_names or (),
        split_description=(
            "Subject-disjoint train/validation/test partitions; 128-sample windows with "
            f"{overlap}-sample overlap; {sample_rate_hz:g} Hz spectral features; scalers fitted "
            "on train only; no zero-channel or zero-feature padding. "
            "Semicolon records repaired; split at invalid sensors, user/activity changes, "
            "backward positive clocks and gaps over 1 second; duplicate/invalid clocks "
            "retain samples in file order at nominal 20 Hz."
        ),
        preprocessing={"protocol": "wisdm-records-v2", "raw_file_sha256": file_sha256(raw_file),
                       "max_gap_ns": 1_000_000_000, "clock_units": "nanoseconds",
                       "parse_audit": parse_audit},
    )
    bundle.validate()
    return bundle
