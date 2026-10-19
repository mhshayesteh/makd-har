from __future__ import annotations

import argparse
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent


def markdown(source: str) -> dict[str, object]:
    return {"cell_type": "markdown", "metadata": {}, "source": source.splitlines(keepends=True)}


def code(source: str) -> dict[str, object]:
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": source.splitlines(keepends=True),
    }


def notebook(dataset: str) -> dict[str, object]:
    display = "UCI-HAR" if dataset == "uci" else "WISDM v1.1"
    data_hint = (
        "Path('/content/UCI HAR Dataset')"
        if dataset == "uci"
        else "Path('/content/WISDM_ar_v1.1_raw.txt')"
    )
    return {
        "nbformat": 4,
        "nbformat_minor": 5,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3"},
            "colab": {"name": f"MAKD-HAR-{display}-revised.ipynb", "provenance": []},
        },
        "cells": [
            markdown(
                f"# Reproducible MAKD-HAR experiment: {display}\n\n"
                "This notebook replaces the earlier sequential split and fixed 50/50 teacher average. "
                "It uses subject-disjoint partitions, train-only scaling, five distinct teachers, real "
                "coalition-based Shapley values, scale-independent contribution weighting, saved per-seed "
                "results, and automatic statistical summaries.\n\n"
                "Run the quick check first. Do not copy quick-check values into the manuscript."
            ),
            markdown(
                "## 1. Make the repaired pipeline available\n\n"
                "Upload or synchronize the entire `revised_pipeline` directory to Colab. Set `PROJECT_ROOT` "
                "to that directory. The original notebooks are intentionally not imported."
            ),
            code(
                "from pathlib import Path\n"
                "PROJECT_ROOT = Path('/content/revised_pipeline')  # change if stored in Google Drive\n"
                "assert (PROJECT_ROOT / 'run_experiment.py').exists(), PROJECT_ROOT\n"
                "%cd {PROJECT_ROOT}\n"
            ),
            markdown("## 2. Install the declared environment"),
            code("%pip install -q -r requirements-colab.txt\n"),
            markdown(
                "## 3. Select the dataset location\n\n"
                "Use the official UCI-HAR archive for UCI-HAR. For WISDM, point to the original "
                "`WISDM_ar_v1.1_raw.txt`. Dataset acquisition is kept separate so the input can be checksummed "
                "and archived with the experiment."
            ),
            code(
                f"DATA_ROOT = {data_hint}\n"
                f"RESULTS_ROOT = PROJECT_ROOT / 'results' / '{dataset}'\n"
                "assert DATA_ROOT.exists(), DATA_ROOT\n"
                "RESULTS_ROOT.mkdir(parents=True, exist_ok=True)\n"
            ),
            markdown("## 4. Quick structural check (one seed, short training)"),
            code(
                "import subprocess, sys\n"
                "quick_command = [\n"
                "    sys.executable, 'run_experiment.py',\n"
                f"    '--dataset', '{dataset}',\n"
                "    '--data-root', str(DATA_ROOT),\n"
                "    '--output', str(RESULTS_ROOT / 'quick-check'),\n"
                "    '--quick', '--verbose', '1',\n"
                "]\n"
                "subprocess.run(quick_command, check=True)\n"
            ),
            markdown(
                "## 5. Production experiment\n\n"
                "This is the manuscript-producing run: ten seeds, all 32 coalitions for five teachers, early "
                "stopping, and resumable outputs. Coalition training is expensive. Completed seed files are "
                "skipped automatically if the command is restarted."
            ),
            code(
                "production_command = [\n"
                "    sys.executable, 'run_experiment.py',\n"
                f"    '--dataset', '{dataset}',\n"
                "    '--data-root', str(DATA_ROOT),\n"
                "    '--output', str(RESULTS_ROOT / 'production'),\n"
                "    '--shapley-mode', 'exact',\n"
                "    '--teacher-epochs', '60',\n"
                "    '--student-epochs', '60',\n"
                "    '--coalition-epochs', '20',\n"
                "    '--batch-size', '128',\n"
                "    '--patience', '10',\n"
                "    '--verbose', '1',\n"
                "]\n"
                "subprocess.run(production_command, check=True)\n"
            ),
            markdown("## 6. Generate mean ± SD and corrected significance tests"),
            code(
                "subprocess.run([\n"
                "    sys.executable, 'summarize_results.py',\n"
                "    str(RESULTS_ROOT / 'production'),\n"
                "], check=True)\n"
            ),
            markdown(
                "## Required audit before manuscript use\n\n"
                "Confirm ten complete seed files; no NaN values; disjoint subject lists in `dataset-manifest.json`; "
                "all 32 coalition evaluations per exact-Shapley run; non-uniform weight diagnostics when claiming "
                "an adaptive-weighting benefit; and agreement between `summary.csv` and all "
                "manuscript tables. Energy and MAC values are intentionally absent until they are measured with "
                "validated target-hardware procedures."
            ),
        ],
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate clean notebook entry points without overwriting user edits by default."
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Replace existing notebooks. Omit this flag to preserve user-edited notebooks.",
    )
    args = parser.parse_args()
    for dataset, filename in (
        ("uci", "KD-HAR-With-Game-UCI-HAR-revised.ipynb"),
        ("wisdm", "KD-HAR-With-Game-WISDM-revised.ipynb"),
    ):
        path = ROOT / filename
        if path.exists() and not args.force:
            print(f"Preserved existing notebook: {path}")
            continue
        path.write_text(json.dumps(notebook(dataset), indent=1), encoding="utf-8")
        print(path)


if __name__ == "__main__":
    main()
