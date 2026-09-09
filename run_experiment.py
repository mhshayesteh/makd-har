from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from makd_har.config import DEFAULT_SEEDS, ExperimentConfig
from makd_har.data import DatasetBundle, load_uci_har, load_wisdm_v1
from makd_har.models import baseline_registry, build_student, teacher_registry
from makd_har.reproducibility import set_global_seed, write_json_atomic
from makd_har.shapley import (
    exact_shapley,
    permutation_shapley,
    shapley_to_weights,
    standardize_shapley,
    weight_diagnostics,
)
from makd_har.training import (
    aggregate_teacher_probabilities,
    benchmark_latency,
    classification_metrics,
    logits_to_probabilities,
    model_footprint,
    predict_logits,
    serialized_model_size_mb,
    train_distilled_student,
    train_supervised,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run reproducible multi-teacher Shapley distillation experiments."
    )
    parser.add_argument("--dataset", choices=("uci", "wisdm"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seeds", default=",".join(map(str, DEFAULT_SEEDS)))
    parser.add_argument("--shapley-mode", choices=("exact", "permutation"), default="exact")
    parser.add_argument("--shapley-permutations", type=int, default=64)
    parser.add_argument("--teacher-epochs", type=int, default=60)
    parser.add_argument("--student-epochs", type=int, default=60)
    parser.add_argument("--coalition-epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--quick", action="store_true", help="One short smoke-test run")
    parser.add_argument("--verbose", type=int, default=0, choices=(0, 1, 2))
    return parser.parse_args()


def load_dataset(args: argparse.Namespace, config: ExperimentConfig) -> DatasetBundle:
    if args.dataset == "uci":
        return load_uci_har(
            args.data_root,
            validation_fraction=config.validation_fraction,
            split_seed=config.split_seed,
        )
    raw_file = args.data_root
    if raw_file.is_dir():
        matches = list(raw_file.rglob("WISDM_ar_v1.1_raw.txt"))
        if not matches:
            raise FileNotFoundError("WISDM_ar_v1.1_raw.txt was not found under data-root")
        raw_file = matches[0]
    return load_wisdm_v1(
        raw_file,
        validation_fraction=config.validation_fraction,
        test_fraction=config.wisdm_test_fraction,
        split_seed=config.split_seed,
    )


def ensure_same_or_write(path: Path, payload: dict[str, object]) -> None:
    """Prevent an output directory from silently mixing experiment protocols."""

    normalized = json.loads(json.dumps(payload))
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != normalized:
            raise RuntimeError(
                f"Existing {path.name} differs from this run. Use a new output directory."
            )
        return
    write_json_atomic(path, normalized)


def input_for(bundle: DatasetBundle, kind: str, partition: str) -> np.ndarray:
    return getattr(bundle, f"{kind}_{partition}")


def train_teacher_bank(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    verbose: int,
) -> tuple[list[str], dict[str, dict[str, np.ndarray]], dict[str, object]]:
    import tensorflow as tf

    registry = teacher_registry()
    probability_bank: dict[str, dict[str, np.ndarray]] = {}
    reports: dict[str, object] = {}
    for index, teacher_name in enumerate(config.teacher_names):
        tf.keras.backend.clear_session()
        spec = registry[teacher_name]
        teacher_seed = seed * 100 + index
        set_global_seed(teacher_seed)
        train_x = input_for(bundle, spec.input_kind, "train")
        val_x = input_for(bundle, spec.input_kind, "val")
        test_x = input_for(bundle, spec.input_kind, "test")
        model = spec.builder(tuple(train_x.shape[1:]), len(bundle.class_names))
        model, history = train_supervised(
            model,
            train_x,
            bundle.y_train,
            val_x,
            bundle.y_val,
            seed=teacher_seed,
            epochs=config.teacher_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=config.patience,
            verbose=verbose,
        )
        logits = {
            "train": predict_logits(model, train_x, config.batch_size),
            "val": predict_logits(model, val_x, config.batch_size),
            "test": predict_logits(model, test_x, config.batch_size),
        }
        probability_bank[teacher_name] = {
            part: logits_to_probabilities(values, config.temperature)
            for part, values in logits.items()
        }
        model_path = output_dir / "models" / f"teacher-{teacher_name}.keras"
        reports[teacher_name] = {
            "input_kind": spec.input_kind,
            "seed": teacher_seed,
            "history": history,
            "validation": classification_metrics(bundle.y_val, logits["val"]),
            "test": classification_metrics(bundle.y_test, logits["test"]),
            "footprint": model_footprint(model),
            "serialized_model_mb": serialized_model_size_mb(model, model_path),
            "latency": benchmark_latency(
                model,
                test_x,
                config.latency_warmups,
                config.latency_repetitions,
            ),
        }
    return list(config.teacher_names), probability_bank, reports


def train_student_variant(
    name: str,
    bundle: DatasetBundle,
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    train_probabilities: np.ndarray | None,
    val_probabilities: np.ndarray | None,
    test_probabilities: np.ndarray | None,
    epochs: int | None = None,
) -> tuple[float, dict[str, object]]:
    import tensorflow as tf

    tf.keras.backend.clear_session()
    set_global_seed(seed)
    model = build_student(tuple(bundle.raw_train.shape[1:]), len(bundle.class_names))
    model, history = train_distilled_student(
        model,
        bundle.raw_train,
        bundle.y_train,
        bundle.raw_val,
        bundle.y_val,
        train_probabilities,
        val_probabilities,
        seed=seed,
        epochs=epochs or config.student_epochs,
        batch_size=config.batch_size,
        learning_rate=config.learning_rate,
        patience=config.patience,
        temperature=config.temperature,
        hard_label_weight=config.hard_label_weight,
    )
    val_logits = predict_logits(model, bundle.raw_val, config.batch_size)
    val_metrics = classification_metrics(bundle.y_val, val_logits)
    report: dict[str, object] = {
        "seed": seed,
        "history": history,
        "validation": val_metrics,
    }
    if test_probabilities is not None or name == "supervised_student":
        test_logits = predict_logits(model, bundle.raw_test, config.batch_size)
        model_path = output_dir / "models" / f"{name}.keras"
        report.update(
            {
                "test": classification_metrics(bundle.y_test, test_logits),
                "footprint": model_footprint(model),
                "serialized_model_mb": serialized_model_size_mb(model, model_path),
                "latency": benchmark_latency(
                    model,
                    bundle.raw_test,
                    config.latency_warmups,
                    config.latency_repetitions,
                ),
            }
        )
    return float(val_metrics["macro_f1"]), report


def tune_and_evaluate_svm(bundle: DatasetBundle) -> dict[str, object]:
    from sklearn.metrics import f1_score
    from sklearn.svm import SVC

    candidates = [
        {"kernel": "linear", "C": c} for c in (0.01, 0.1, 1.0, 10.0)
    ] + [
        {"kernel": "rbf", "C": c, "gamma": gamma}
        for c in (1.0, 10.0, 100.0)
        for gamma in ("scale", 0.01, 0.001)
    ]
    best_model = None
    best_params = None
    best_f1 = -np.inf
    tuning_results = []
    for params in candidates:
        model = SVC(**params, cache_size=4096)
        model.fit(bundle.features_train, bundle.y_train)
        predictions = model.predict(bundle.features_val)
        score = float(f1_score(bundle.y_val, predictions, average="macro", zero_division=0))
        tuning_results.append({"params": params, "validation_macro_f1": score})
        if score > best_f1:
            best_f1, best_model, best_params = score, model, params
    started = time.perf_counter()
    predictions = best_model.predict(bundle.features_test)
    elapsed = time.perf_counter() - started
    decision = best_model.decision_function(bundle.features_test)
    return {
        "selected_params": best_params,
        "tuning_results": tuning_results,
        "validation_macro_f1": best_f1,
        "test": classification_metrics(bundle.y_test, decision),
        "inference": {
            "batch_total_seconds": elapsed,
            "samples": int(len(bundle.y_test)),
            "mean_ms_per_sample": elapsed * 1000.0 / len(bundle.y_test),
            "note": "SVM batch timing; neural latency uses synchronous batch-size-one timing.",
        },
    }


def train_neural_baselines(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    verbose: int,
) -> dict[str, object]:
    import tensorflow as tf

    reports: dict[str, object] = {}
    for name, builder in baseline_registry().items():
        tf.keras.backend.clear_session()
        baseline_seed = seed * 1000 + 31
        set_global_seed(baseline_seed)
        model = builder(tuple(bundle.raw_train.shape[1:]), len(bundle.class_names))
        model, history = train_supervised(
            model,
            bundle.raw_train,
            bundle.y_train,
            bundle.raw_val,
            bundle.y_val,
            seed=baseline_seed,
            epochs=config.student_epochs,
            batch_size=config.batch_size,
            learning_rate=config.learning_rate,
            patience=config.patience,
            verbose=verbose,
        )
        val_logits = predict_logits(model, bundle.raw_val, config.batch_size)
        test_logits = predict_logits(model, bundle.raw_test, config.batch_size)
        model_path = output_dir / "models" / f"baseline-{name}.keras"
        reports[name] = {
            "seed": baseline_seed,
            "history": history,
            "validation": classification_metrics(bundle.y_val, val_logits),
            "test": classification_metrics(bundle.y_test, test_logits),
            "footprint": model_footprint(model),
            "serialized_model_mb": serialized_model_size_mb(model, model_path),
            "latency": benchmark_latency(
                model,
                bundle.raw_test,
                config.latency_warmups,
                config.latency_repetitions,
            ),
        }
    reports["svm"] = tune_and_evaluate_svm(bundle)
    return reports


def run_seed(
    bundle: DatasetBundle,
    config: ExperimentConfig,
    seed: int,
    output_dir: Path,
    verbose: int,
) -> dict[str, object]:
    seed_dir = output_dir / f"seed-{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    teacher_names, bank, teacher_reports = train_teacher_bank(
        bundle, config, seed, seed_dir, verbose
    )

    coalition_reports: dict[str, object] = {}

    def coalition_value(coalition: tuple[int, ...]) -> float:
        key = "empty" if not coalition else "-".join(map(str, coalition))
        cache_path = seed_dir / "coalitions" / f"{key}.json"
        if cache_path.exists():
            return float(json.loads(cache_path.read_text(encoding="utf-8"))["validation_macro_f1"])
        if coalition:
            selected = [teacher_names[index] for index in coalition]
            train_probs = aggregate_teacher_probabilities([bank[name]["train"] for name in selected])
            val_probs = aggregate_teacher_probabilities([bank[name]["val"] for name in selected])
        else:
            selected, train_probs, val_probs = [], None, None
        # Use the same initialization/minibatch seed for every coalition in an
        # outer run. Otherwise random initialization is confounded with the
        # marginal contribution attributed to a teacher.
        coalition_seed = seed * 100000 + 17
        score, report = train_student_variant(
            f"coalition-{key}",
            bundle,
            config,
            coalition_seed,
            seed_dir,
            train_probs,
            val_probs,
            None,
            epochs=config.coalition_epochs,
        )
        compact = {
            "coalition_indices": list(coalition),
            "teacher_names": selected,
            "seed": coalition_seed,
            "validation_macro_f1": score,
            "history": report["history"],
        }
        write_json_atomic(cache_path, compact)
        coalition_reports[key] = compact
        return score

    if config.shapley_mode == "exact":
        shapley_result = exact_shapley(len(teacher_names), coalition_value)
    else:
        shapley_result = permutation_shapley(
            len(teacher_names),
            coalition_value,
            config.shapley_permutations,
            seed,
        )
    weights = shapley_to_weights(
        shapley_result.values, config.shapley_softmax_temperature
    )
    weighting_diagnostics = weight_diagnostics(weights)
    if weighting_diagnostics["warning"]:
        print(f"WARNING: {weighting_diagnostics['warning']}")

    weighted = {
        part: aggregate_teacher_probabilities(
            [bank[name][part] for name in teacher_names], weights
        )
        for part in ("train", "val", "test")
    }
    comparison_seed = seed * 1000000 + 7
    _, proposed_report = train_student_variant(
        "proposed_shapley_student",
        bundle,
        config,
        comparison_seed,
        seed_dir,
        weighted["train"],
        weighted["val"],
        weighted["test"],
    )

    uniform = {
        part: aggregate_teacher_probabilities([bank[name][part] for name in teacher_names])
        for part in ("train", "val", "test")
    }
    _, uniform_report = train_student_variant(
        "uniform_teacher_student",
        bundle,
        config,
        comparison_seed,
        seed_dir,
        uniform["train"],
        uniform["val"],
        uniform["test"],
    )
    _, supervised_report = train_student_variant(
        "supervised_student",
        bundle,
        config,
        comparison_seed,
        seed_dir,
        None,
        None,
        None,
    )

    return {
        "dataset_manifest": bundle.manifest(),
        "config": config.to_dict(),
        "model_seed": seed,
        "teachers": teacher_reports,
        "teacher_order": teacher_names,
        "shapley": shapley_result.to_dict(),
        "standardized_shapley_values": standardize_shapley(shapley_result.values).tolist(),
        "shapley_weights": weights.tolist(),
        "shapley_weight_diagnostics": weighting_diagnostics,
        "coalition_reports_created_this_run": coalition_reports,
        "methods": {
            "proposed": proposed_report,
            "uniform_teachers": uniform_report,
            "supervised_student": supervised_report,
            **train_neural_baselines(bundle, config, seed, seed_dir, verbose),
        },
        "measurement_notes": {
            "loss_column": "Not exported for cross-method comparison.",
            "energy": "Not estimated from assumed power. Measure on target hardware separately.",
            "macs": "Not exported until a recurrent/attention-aware profiler is validated.",
            "feature_distillation": (
                "This repaired milestone validates logit-level Shapley distillation. "
                "Feature-alignment ablations must be added before manuscript resubmission."
            ),
        },
    }


def main() -> None:
    args = parse_args()
    seeds = tuple(int(value) for value in args.seeds.split(",") if value.strip())
    if args.quick:
        seeds = seeds[:1]
    config = ExperimentConfig(
        model_seeds=seeds,
        teacher_epochs=2 if args.quick else args.teacher_epochs,
        student_epochs=2 if args.quick else args.student_epochs,
        coalition_epochs=1 if args.quick else args.coalition_epochs,
        patience=1 if args.quick else args.patience,
        batch_size=args.batch_size,
        shapley_mode="permutation" if args.quick else args.shapley_mode,
        shapley_permutations=4 if args.quick else args.shapley_permutations,
        latency_warmups=2 if args.quick else 30,
        latency_repetitions=5 if args.quick else 300,
    )
    config.validate()
    bundle = load_dataset(args, config)
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(args.output / "dataset-manifest.json", bundle.manifest())
    ensure_same_or_write(args.output / "experiment-config.json", config.to_dict())
    for seed in config.model_seeds:
        result_path = args.output / f"seed-{seed}.json"
        if result_path.exists():
            print(f"Skipping completed seed {seed}: {result_path}")
            continue
        print(f"Running {bundle.name}, seed {seed}")
        result = run_seed(bundle, config, seed, args.output, args.verbose)
        write_json_atomic(result_path, result)
        print(f"Completed seed {seed}: {result_path}")


if __name__ == "__main__":
    main()
