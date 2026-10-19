from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np

from makd_har.config import ExperimentConfig
from makd_har.models import teacher_registry
from makd_har.reproducibility import set_global_seed, write_json_atomic
from makd_har.shapley import exact_shapley, shapley_to_weights, weight_diagnostics
from makd_har.training import (
    aggregate_teacher_probabilities,
    classification_metrics,
    logits_to_probabilities,
    predict_logits,
    train_supervised,
)
from run_experiment import load_dataset, train_student_variant


PILOT_SEEDS = (11, 23, 37)
COALITION_BUDGETS = (20, 60)
SHRINKAGE_LEVELS = (0.0, 0.5)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validation-only Shapley weighting pilot")
    parser.add_argument("--dataset", choices=("uci", "wisdm"), required=True)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--verbose", type=int, default=1, choices=(0, 1, 2))
    return parser.parse_args()


def shrink_weights(weights: np.ndarray, shrinkage: float) -> np.ndarray:
    """Shrink toward uniform; 0 keeps Shapley and 1 becomes uniform."""
    weights = np.asarray(weights, dtype=np.float64)
    if not 0 <= shrinkage <= 1 or weights.ndim != 1 or len(weights) == 0:
        raise ValueError("invalid weights or shrinkage")
    weights = weights / weights.sum()
    uniform = np.full(len(weights), 1.0 / len(weights))
    return (1.0 - shrinkage) * weights + shrinkage * uniform


def candidate_name(budget: int, shrinkage: float) -> str:
    suffix = "raw" if shrinkage == 0 else f"shrink-{int(100 * shrinkage)}"
    return f"shapley-{budget}-{suffix}"


def build_config() -> ExperimentConfig:
    return ExperimentConfig(
        model_seeds=PILOT_SEEDS,
        teacher_epochs=60,
        student_epochs=60,
        coalition_epochs=60,
        patience=10,
        batch_size=128,
        shapley_mode="exact",
        protocol_version="records-v2-predict-v2-weight-pilot-v1",
    )


def ensure_same_or_write(path: Path, payload: dict[str, object]) -> None:
    normalized = json.loads(json.dumps(payload))
    if path.exists():
        if json.loads(path.read_text(encoding="utf-8")) != normalized:
            raise RuntimeError(f"Existing {path.name} differs; use a new output directory")
    else:
        write_json_atomic(path, normalized)


def teacher_bank(bundle, config, seed: int, seed_dir: Path, verbose: int):
    import tensorflow as tf

    cache = seed_dir / "teacher-probabilities.npz"
    report_path = seed_dir / "teacher-report.json"
    if cache.exists() and report_path.exists():
        arrays = np.load(cache)
        bank = {
            name: {part: arrays[f"{name}_{part}"] for part in ("train", "val")}
            for name in config.teacher_names
        }
        return bank, json.loads(report_path.read_text(encoding="utf-8"))

    registry = teacher_registry()
    bank, reports, arrays = {}, {}, {}
    for index, name in enumerate(config.teacher_names):
        tf.keras.backend.clear_session()
        spec = registry[name]
        teacher_seed = seed * 100 + index
        set_global_seed(teacher_seed)
        train_x = getattr(bundle, f"{spec.input_kind}_train")
        val_x = getattr(bundle, f"{spec.input_kind}_val")
        model = spec.builder(tuple(train_x.shape[1:]), len(bundle.class_names))
        model, history = train_supervised(
            model, train_x, bundle.y_train, val_x, bundle.y_val,
            seed=teacher_seed, epochs=config.teacher_epochs,
            batch_size=config.batch_size, learning_rate=config.learning_rate,
            patience=config.patience, verbose=verbose,
        )
        bank[name] = {}
        for part, inputs in (("train", train_x), ("val", val_x)):
            logits = predict_logits(model, inputs, config.batch_size)
            bank[name][part] = logits_to_probabilities(logits, config.temperature)
            arrays[f"{name}_{part}"] = bank[name][part]
            if part == "val":
                validation = classification_metrics(bundle.y_val, logits)
        reports[name] = {"seed": teacher_seed, "input_kind": spec.input_kind,
                         "history": history, "validation": validation}
    temporary = cache.with_suffix(".npz.tmp")
    with temporary.open("wb") as stream:
        np.savez_compressed(stream, **arrays)
    temporary.replace(cache)
    write_json_atomic(report_path, reports)
    return bank, reports


def coalition_game(bundle, config, seed, seed_dir, bank, budget):
    teacher_names = list(config.teacher_names)
    coalition_dir = seed_dir / f"coalitions-{budget}"
    coalition_dir.mkdir(parents=True, exist_ok=True)

    def value(coalition: tuple[int, ...]) -> float:
        key = "empty" if not coalition else "-".join(map(str, coalition))
        path = coalition_dir / f"{key}.json"
        if path.exists():
            return float(json.loads(path.read_text(encoding="utf-8"))["validation_macro_f1"])
        selected = [teacher_names[i] for i in coalition]
        if selected:
            train_probs = aggregate_teacher_probabilities([bank[n]["train"] for n in selected])
            val_probs = aggregate_teacher_probabilities([bank[n]["val"] for n in selected])
        else:
            train_probs = val_probs = None
        coalition_seed = seed * 100000 + 17
        score, report = train_student_variant(
            f"pilot-coalition-{budget}-{key}", bundle, config, coalition_seed,
            seed_dir, train_probs, val_probs, None, epochs=budget,
        )
        payload = {"coalition_indices": list(coalition), "teacher_names": selected,
                   "seed": coalition_seed, "budget": budget,
                   "selected_epochs": len(report["history"]["val_macro_f1"]),
                   "validation_macro_f1": score, "history": report["history"]}
        write_json_atomic(path, payload)
        return score

    result = exact_shapley(len(teacher_names), value)
    return result, shapley_to_weights(result.values, config.shapley_softmax_temperature)


def train_candidate(bundle, config, seed, seed_dir, bank, name, weights):
    path = seed_dir / "candidates" / f"{name}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))
    train_probs = aggregate_teacher_probabilities(
        [bank[n]["train"] for n in config.teacher_names], weights)
    val_probs = aggregate_teacher_probabilities(
        [bank[n]["val"] for n in config.teacher_names], weights)
    comparison_seed = seed * 1000000 + 7
    score, report = train_student_variant(
        f"pilot-{name}", bundle, config, comparison_seed, seed_dir,
        train_probs, val_probs, None, epochs=config.student_epochs,
    )
    payload = {"name": name, "seed": comparison_seed, "weights": weights.tolist(),
               "validation_macro_f1": score,
               "selected_epochs": len(report["history"]["val_macro_f1"]),
               "history": report["history"]}
    write_json_atomic(path, payload)
    return payload


def summarize(output: Path) -> dict[str, object]:
    runs = [json.loads((output / f"seed-{s}.json").read_text()) for s in PILOT_SEEDS]
    names = list(runs[0]["candidates"])
    rows = []
    for name in names:
        values = np.asarray([r["candidates"][name]["validation_macro_f1"] for r in runs])
        rows.append({"candidate": name, "runs": len(values), "validation_macro_f1_mean": float(values.mean()),
                     "validation_macro_f1_sd": float(values.std(ddof=1)),
                     "wins_vs_uniform": int(np.sum(values > np.asarray([
                         r["candidates"]["uniform"]["validation_macro_f1"] for r in runs])))})
    order = {"uniform": 0, "shapley-20-shrink-50": 1, "shapley-20-raw": 2,
             "shapley-60-shrink-50": 3, "shapley-60-raw": 4}
    selected = max(rows, key=lambda row: (round(row["validation_macro_f1_mean"], 12), -order[row["candidate"]]))
    result = {"selection_measure": "mean validation macro-F1 across fixed pilot seeds",
              "exploratory_only": True, "test_metrics_computed": False,
              "selected_candidate": selected["candidate"], "rows": rows}
    write_json_atomic(output / "pilot-summary.json", result)
    with (output / "pilot-summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)
    return result


def main() -> None:
    args = parse_args()
    config = build_config(); config.validate()
    bundle = load_dataset(args, config)
    args.output.mkdir(parents=True, exist_ok=True)
    ensure_same_or_write(args.output / "experiment-config.json", config.to_dict())
    ensure_same_or_write(args.output / "dataset-manifest.json", bundle.manifest())
    for seed in PILOT_SEEDS:
        final = args.output / f"seed-{seed}.json"
        if final.exists():
            print(f"Skipping completed pilot seed {seed}"); continue
        seed_dir = args.output / f"seed-{seed}"; seed_dir.mkdir(parents=True, exist_ok=True)
        bank, teachers = teacher_bank(bundle, config, seed, seed_dir, args.verbose)
        games = {}
        for budget in COALITION_BUDGETS:
            result, raw_weights = coalition_game(bundle, config, seed, seed_dir, bank, budget)
            games[str(budget)] = {"shapley": result.to_dict(), "raw_weights": raw_weights.tolist(),
                                  "raw_weight_diagnostics": weight_diagnostics(raw_weights)}
        candidates = {}
        uniform = np.full(len(config.teacher_names), 1 / len(config.teacher_names))
        candidates["uniform"] = train_candidate(bundle, config, seed, seed_dir, bank, "uniform", uniform)
        for budget in COALITION_BUDGETS:
            raw = np.asarray(games[str(budget)]["raw_weights"])
            for shrinkage in SHRINKAGE_LEVELS:
                name = candidate_name(budget, shrinkage)
                candidates[name] = train_candidate(bundle, config, seed, seed_dir, bank, name,
                                                     shrink_weights(raw, shrinkage))
        write_json_atomic(final, {"model_seed": seed, "teacher_order": list(config.teacher_names),
                                  "teachers": teachers, "games": games, "candidates": candidates,
                                  "test_metrics_computed": False})
        print(f"Completed pilot seed {seed}")
    if all((args.output / f"seed-{s}.json").exists() for s in PILOT_SEEDS):
        result = summarize(args.output)
        print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
