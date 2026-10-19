# Independent analysis implementations

These scripts verify completed outputs and regenerate statistics, tables, reports and figures. Data, checkpoint artifacts, prediction arrays and generated outputs are external and ignored by Git. Manuscript-editing scripts are excluded.

Install `../requirements-analysis.txt`. Most analyses import the unchanged runners at `PROJECT/revision_2/`; restore that code tree alongside external results. Choose a fresh writable analysis directory, preferably outside the checkout.

Run the original benchmark analysis first:

```bash
python analysis/analyze_existing_results.py --drive "/path/to/PROJECT" --paper "/path/to/ANALYSIS_ROOT"
```

It writes `ANALYSIS_ROOT/results_analysis/revision-2/`. Use that directory as `OUTPUT` for later analyses:

```bash
python analysis/analyze_coalition_noise.py --drive "/path/to/PROJECT" --output "/path/to/OUTPUT"
python analysis/analyze_uci_budget.py --drive "/path/to/PROJECT" --output "/path/to/OUTPUT"
python analysis/analyze_teacher_pruning.py --drive "/path/to/PROJECT" --output "/path/to/OUTPUT"
python analysis/analyze_weighting_pilot.py --drive "/path/to/PROJECT" --output "/path/to/OUTPUT"
python analysis/analyze_teacher_ensemble.py --drive "/path/to/PROJECT" --output "/path/to/OUTPUT"
python analysis/analyze_runtime_benchmark.py --drive "/path/to/PROJECT" --session "/path/to/PROJECT/results/revision-2/runtime-benchmark/SESSION" --output "/path/to/OUTPUT"
python analysis/analyze_wisdm_gap.py --drive "/path/to/PROJECT" --output "/path/to/OUTPUT"
```

The gap script verifies the original runtime session `session-20261019-013326-b29922da`, ensemble outputs, data splits and original selection evidence. It intentionally audits that recorded evidence rather than arbitrary replacement sessions. The noise script accepts `--wisdm-results` for an explicitly renamed local Drive folder, while still checking recorded hashes.

`verify_data_manifests.py --help` describes independent data-manifest reconstruction. `test_wisdm_gap_guards.py` checks interpretation-sensitive arithmetic without fitting or private datasets.

Intervals and tests describe model-seed variability on fixed subject splits. The original six-comparison and additional three-student Holm families remain distinct. Seeds are not independent participant cohorts. Negative outcomes, validation reuse and timing limits are retained.
