# MAKD-HAR

MAKD-HAR is a reproducible five-teacher knowledge-distillation pipeline for human activity recognition. The teachers are an HCF-MLP, raw-signal CNN, bidirectional LSTM, temporal convolutional network, and Transformer. Their validation utilities are evaluated across teacher coalitions, converted to exact Shapley values, standardized, and transformed into nonnegative softmax weights for a compact student model.

The project contains the implementation used for the revised UCI-HAR and WISDM experiments. Teacher training, coalition evaluation, Shapley attribution, final student training, uniform-weight and supervised ablations, baselines, statistical summaries, and resumable per-seed outputs are included.

## Repository contents

- `KD-HAR-With-Game-UCI-HAR-revised.ipynb`: Google Colab workflow for UCI-HAR.
- `KD-HAR-With-Game-WISDM-revised.ipynb`: Google Colab workflow for WISDM v1.1.
- `run_experiment.py`: resumable experiment runner.
- `summarize_results.py`: statistical summaries and paired comparisons.
- `makd_har/`: data preparation, models, training, Shapley attribution, configuration, and reproducibility utilities.
- `tests/`: fast tests for splitting, feature extraction, and Shapley calculations.
- `results/verified/`: compact result summaries used to check the reported aggregate values; datasets and checkpoints are excluded.

## Installation

Python 3.10 or newer is recommended. TensorFlow training is intended for Google Colab or another configured TensorFlow environment.

```bash
python -m venv .venv
python -m pip install --upgrade pip
python -m pip install -r requirements-colab.txt
```

Run the lightweight checks before an experiment:

```bash
python -m pytest -q
python -m compileall makd_har run_experiment.py summarize_results.py
```

## Datasets

The datasets are not redistributed in this repository.

### UCI Human Activity Recognition Using Smartphones

- Official dataset page: <https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones>
- Expected Colab archive: `UCI HAR Dataset.zip`

### WISDM Activity Recognition v1.1

- Official dataset page: <https://www.cis.fordham.edu/wisdm/dataset.php>
- Expected Colab file: `WISDM_ar_v1.1_raw.txt`

Retain the documentation supplied by the dataset providers and follow their citation and redistribution instructions.

## Google Colab and Google Drive

Place the repository in either supported Drive layout:

```text
MyDrive/mk-har/
├── run_experiment.py
├── makd_har/
├── resources/
│   ├── UCI HAR Dataset.zip
│   └── WISDM_ar_v1.1_raw.txt
└── results/
```

or place the code under `MyDrive/mk-har/revised_pipeline/`, while retaining `resources/` and `results/` under `MyDrive/mk-har/`.

Open the notebook for the required dataset in Colab and run its cells in order. Each notebook mounts Drive, copies code and data into the Colab runtime, performs a one-seed quick check, and provides a production cell. Quick-check values are smoke-test outputs and must not be used as publication results.

## Command-line examples

Quick check:

```bash
python run_experiment.py \
  --dataset uci \
  --data-root "/path/to/UCI HAR Dataset" \
  --output results/uci/quick-check \
  --seeds 11 \
  --quick
```

Ten-seed UCI-HAR production run:

```bash
python run_experiment.py \
  --dataset uci \
  --data-root "/path/to/UCI HAR Dataset" \
  --output results/uci/production \
  --seeds 11,23,37,53,71,89,107,131,167,197 \
  --shapley-mode exact
```

For WISDM, use `--dataset wisdm`, point `--data-root` to `WISDM_ar_v1.1_raw.txt`, and use a separate output directory. The revised WISDM protocol uses the 60-epoch coalition budget selected through validation-only sensitivity analysis; the notebook contains the corresponding production command.

Completed seed JSON files are detected and skipped when the same command is restarted. A changed protocol must use a different output directory so incompatible results cannot be mixed.

## Experimental safeguards

- UCI-HAR test subjects never enter training or validation.
- WISDM partitions are subject-disjoint.
- Scalers are fitted on training data only.
- Every production seed evaluates all 32 coalitions for five teachers.
- Coalition utility uses validation macro-F1; held-out test data is not used to choose teacher weights.
- Shapley, uniform-weight, and supervised students share the same architecture within a dataset.
- Cross-method training loss and unmeasured energy estimates are intentionally not reported.
- Results are serialized per seed with the manifest, configuration, teacher metrics, coalition utilities, weights, and final metrics.

## Verified summaries

`results/verified/` contains aggregate UCI-HAR and corrected WISDM summaries, 95% confidence intervals, the validation-only WISDM weighting audit, and the configuration and split manifests needed to interpret the reported values. Reproduction should use the source datasets and code rather than treating these summaries as input data.

## Citation

If this implementation supports your work, please cite the associated paper:

> M. H. Shayesteh, B. Shahrokhzadeh, M. Gholipour, B. Masoumi, and A. E. Kelishomi, “Cooperative Multi-Agent Knowledge Distillation for Real-Time Human Activity Recognition on Edge Devices.”

The final journal, year, volume, pages, and DOI should be added after publication.

## License

No open-source license has been selected yet. Until the authors add a license, the code remains copyright protected and may be viewed for research reproducibility, but reuse and redistribution are not automatically granted.
