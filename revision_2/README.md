# Revision-2 experiment implementations

These implementations cover the completed second-review studies. Code, input-hash manifests, subject-role specifications and tests are included. Datasets, saved models and outputs remain external.

| Experiment | Notebook | Runner | Research scope |
|---|---|---|---|
| Coalition noise | `Revision-2-Coalition-Noise-fixed.ipynb` | `run_coalition_noise.py`, `run_wisdm_noise_deterministic.py` | Repeated empty/full coalitions; 60 students |
| UCI budget | `Revision-2-UCI-Budget-Sensitivity.ipynb` | `uci_budget/run_uci_budget.py` | 96 coalitions and six matched final students |
| Teacher removal | `Revision-2-Teacher-Removal.ipynb` | `teacher_pruning/run_teacher_pruning.py` | All single removals and full controls; 120 students |
| Training-only weighting | `Revision-2-Training-Only-Weighting-Pilot.ipynb` | `weighting_pilot/run_pilot.py` | Four training-subject roles; 30 teachers, 222 students |
| Ensemble inference | `Revision-2-Teacher-Ensemble-Evaluation.ipynb` | `teacher_ensemble/run_ensemble.py` | Twenty banks, 100 teachers and fixed ensembles; no fitting |
| CPU runtime | `Revision-2-Deployment-CPU-Benchmark.ipynb` | `runtime_benchmark/run_runtime.py` | Twenty students, 65 conditions, three timing rounds; one auxiliary SVC fit |

The training studies total 504 students and 30 teachers, excluding separate smoke runs. Completed fits need not be repeated. The WISDM gap audit in `../analysis/analyze_wisdm_gap.py` adds no training.

## Execution

Keep this directory unchanged at `PROJECT/revision_2/`, datasets at `PROJECT/resources/`, and original results at `PROJECT/results/`. Colab notebooks default to `/content/drive/MyDrive/makd-har`; edit `PROJECT` if necessary. Each verifies inputs, separates smoke from research and resumes valid completed results. The superseded coalition diagnostic notebook is omitted.

Verify without fitting, for example:

```bash
python revision_2/run_coalition_noise.py --drive-root "/path/to/PROJECT" --dataset all --verify-files-only
```

For WISDM noise, use the deterministic launcher. It enables deterministic operations and disables TensorFloat-32 before frozen-teacher predictions. Do not relax hash or parity checks to bypass an incompatible runtime.

CPU timing requires a CPU runtime and fresh session below `results/revision-2/runtime-benchmark/`. Preparation, smoke and research remain separate. Do not combine partial sessions across processes. Classifier-only timing excludes preprocessing and loading and is not a physical-device or energy measurement.

## Frozen specifications and interpretation

`pipeline_snapshot/` and the runner/test files retain the original bytes checked by the experiment manifests. `frozen-inputs.json` files specify code and external input hashes. `weighting_pilot/uci-split.json` and `wisdm-split.json` specify subject roles. These are configuration files required by the guards.

Original replay requires the original external artifacts. Freshly trained checkpoints have new identities; do not edit expected hashes to present a new run as the original study. Keep new research outputs separate.

Retain negative outcomes. The training-only pilot does not replace original test results. Teacher rankings and utilities vary. Nonsignificant comparisons do not establish equivalence, predictive superiority or a pruning advantage. Original WISDM validation reuse remains a limitation.
