# MAKD-HAR

Implementation accompanying **Shapley Based Multi Teacher Knowledge Distillation for Human Activity Recognition**, revised for IEEE Access. Five frozen teachers supply probability targets to a compact student. Separately trained coalition students define a validation macro-F1 game; enumerating its 32 coalitions gives exact Shapley values for the recorded game. Those values are standardized and converted to fixed softmax distillation weights.

Exact enumeration does not remove training noise. The experiments do not establish predictive superiority over SVMs, a significant advantage over uniform distillation, or physical-device energy savings. See the paper for measured outcomes and limitations, including original validation reuse.

## Contents

| Location | Implementation |
|---|---|
| `makd_har/`, `run_experiment.py` | Corrected parsing, subject splits, teachers, coalition training, attribution, students and baselines |
| `summarize_results.py` | Seed summaries, Student-t intervals and paired comparisons |
| `run_weighting_pilot.py` | Historical validation-only budget/shrinkage pilot |
| Two `KD-HAR-With-Game-*-revised.ipynb` notebooks | Benchmark workflows with corrected WISDM parsing and SVC prediction scoring |
| `SVM-vs-Student-Colab-Benchmark.ipynb` | Historical saved-student/SVM benchmark used by the later audit |
| `revision_2/` | Noise, UCI budget, teacher removal, training-only weighting, ensemble and CPU runtime experiments |
| `analysis/` | Independent verification, statistics, tables, figures and WISDM gap diagnostics |
| `tests/` and experiment guard tests | Splits, parsing, attribution, provenance, resume and analysis checks |

This is a **code-only distribution**. Manuscripts, reviewer correspondence, datasets, models, prediction arrays, result summaries and generated figures are excluded. The small JSON files under `revision_2/` specify required input hashes and subject roles; they are configuration, not result exports. Notebook execution outputs are cleared.

## Installation and checks

Use Python 3.10 or newer in a separate environment. TensorFlow training is intended for configured Colab or local compute.

```bash
python -m pip install -r requirements-colab.txt
python -m pip install -r requirements-analysis.txt
python -m pytest -q
```

Dependency ranges support installation, not bitwise reproduction across all versions. The controlled CPU study used TensorFlow 2.21.0, Keras 3.13.2 and scikit-learn 1.6.1 with one numerical/runtime thread. Preserve frozen code and manifests for original-evidence replay.

## External data and working layout

Obtain datasets separately from [UCI HAR](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones) and [WISDM v1.1](https://www.cis.fordham.edu/wisdm/dataset.php), following provider citation and redistribution requirements.

Place the checkout at the working project root, typically `MyDrive/makd-har/` in Colab:

```text
makd-har/
  run_experiment.py
  makd_har/
  revision_2/
  analysis/
  resources/                 # external, ignored
    UCI HAR Dataset.zip
    WISDM_ar_v1.1_raw.txt
  results/                   # external/generated, ignored
```

Other locations work with `--drive-root` or `--drive`. Frozen revision-2 audits require the unchanged code tree plus original seed records, saved checkpoints, selection records and historical SVM benchmark under the paths recorded by each `frozen-inputs.json`. They deliberately reject missing or changed inputs. A new training run produces new artifacts and cannot silently replace original evidence. This code-only checkout therefore requires external artifacts for checkpoint-dependent replay.

## Benchmark examples

Extract UCI data separately. Quick check:

```bash
python run_experiment.py --dataset uci --data-root "/path/to/UCI HAR Dataset" --output results/uci/quick-check --seeds 11 --quick
```

New ten-seed UCI run:

```bash
python run_experiment.py --dataset uci --data-root "/path/to/UCI HAR Dataset" --output results/uci/new-production --seeds 11,23,37,53,71,89,107,131,167,197 --teacher-epochs 60 --student-epochs 60 --coalition-epochs 20 --batch-size 128 --patience 10 --shapley-mode exact
python summarize_results.py results/uci/new-production
```

WISDM uses its raw text file, corrected parser and 60-epoch coalition cap. The historical `production-v2` notebook also requires its external validation-selection record. Use fresh output folders for changed protocols; quick checks are excluded from research evidence.

See [revision_2/README.md](revision_2/README.md) for all six later experiments and [analysis/README.md](analysis/README.md) for reconstruction. Completed fits need not be rerun to update this repository. `revision_2/pipeline_snapshot/` preserves the exact implementation checked by the manifests. `revised_pipeline/run_experiment.py` preserves one historical path checked by the UCI budget manifest; the root runner is the ordinary entry point.

## Citation and license

Please cite M. H. Shayesteh, B. Shahrokhzadeh, M. Gholipour, B. Masoumi and A. E. Kelishomi, **Shapley Based Multi Teacher Knowledge Distillation for Human Activity Recognition**. Publication details and DOI will be added after publication.

No open-source license has been selected. Copyright remains with the authors; additional reuse and redistribution rights are not granted here.
