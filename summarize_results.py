from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize completed seeded experiments")
    parser.add_argument("results", type=Path)
    parser.add_argument("--output", type=Path, default=None)
    return parser.parse_args()


def holm_adjust(p_values: list[float]) -> list[float]:
    order = np.argsort(p_values)
    adjusted = np.empty(len(p_values), dtype=float)
    running = 0.0
    for rank, index in enumerate(order):
        value = min(1.0, (len(p_values) - rank) * p_values[index])
        running = max(running, value)
        adjusted[index] = running
    return adjusted.tolist()


def main() -> None:
    args = parse_args()
    files = sorted(args.results.glob("seed-*.json"))
    if not files:
        raise SystemExit(f"No seed-*.json files found in {args.results}")
    runs = [json.loads(path.read_text(encoding="utf-8")) for path in files]
    method_names = sorted(set.intersection(*(set(run["methods"]) for run in runs)))
    rows: list[dict[str, object]] = []
    per_method: dict[str, np.ndarray] = {}
    for method in method_names:
        values = np.asarray(
            [run["methods"][method]["test"]["macro_f1"] for run in runs], dtype=float
        )
        per_method[method] = values
        mean = float(values.mean())
        std = float(values.std(ddof=1)) if len(values) > 1 else 0.0
        ci95 = 1.96 * std / math.sqrt(len(values)) if len(values) > 1 else 0.0
        accuracies = np.asarray(
            [run["methods"][method]["test"]["accuracy"] for run in runs], dtype=float
        )
        rows.append(
            {
                "method": method,
                "runs": len(values),
                "accuracy_mean": float(accuracies.mean()),
                "accuracy_sd": float(accuracies.std(ddof=1)) if len(values) > 1 else 0.0,
                "macro_f1_mean": mean,
                "macro_f1_sd": std,
                "macro_f1_ci95_half_width": ci95,
            }
        )

    tests: list[dict[str, object]] = []
    raw_p_values: list[float] = []
    if len(runs) > 1 and "proposed" in per_method:
        from scipy.stats import wilcoxon

        proposed = per_method["proposed"]
        for method in method_names:
            if method == "proposed":
                continue
            differences = proposed - per_method[method]
            try:
                statistic, p_value = wilcoxon(differences, alternative="two-sided")
            except ValueError:
                statistic, p_value = 0.0, 1.0
            raw_p_values.append(float(p_value))
            tests.append(
                {
                    "comparison": f"proposed vs {method}",
                    "paired_runs": len(runs),
                    "mean_f1_difference": float(differences.mean()),
                    "wilcoxon_statistic": float(statistic),
                    "p_raw": float(p_value),
                    "effect_size_dz": float(
                        differences.mean() / differences.std(ddof=1)
                    )
                    if differences.std(ddof=1) > 0
                    else None,
                }
            )
        for row, adjusted in zip(tests, holm_adjust(raw_p_values)):
            row["p_holm"] = adjusted

    output_dir = args.output or args.results
    output_dir.mkdir(parents=True, exist_ok=True)
    with (output_dir / "summary.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    (output_dir / "significance-tests.json").write_text(
        json.dumps(tests, indent=2), encoding="utf-8"
    )
    print(f"Wrote {output_dir / 'summary.csv'}")
    print(f"Wrote {output_dir / 'significance-tests.json'}")


if __name__ == "__main__":
    main()
