# `evaluateOnData` — Segmentation evaluation runner

`src/segmentation/evaluate/evaluateOnData.py` is the canonical entry
point for re-evaluating the deployed segmenter every time the dataset
changes. It runs the active accelerometer template-match detector
across the experiments you select, reproduces every figure that appears
in the *Segmentation & Detection / Evaluation* subsection of
`docs/latex/main.tex`, and writes everything (plus a record of how the
run was configured) into a timestamped folder.

## TL;DR

```bash
# from repo root, with venv activated
venv/bin/python -m src.segmentation.evaluate.evaluateOnData
```

Output lands under
`elevator_reports/seg_eval/run_<YYYYMMDD-HHMMSS>/`.

## CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--algorithm`        | `acc_template_match`  | Detector to run. Set to `pressure_filter` to evaluate the barometer-fallback. |
| `--kind`             | `all`                 | `all` / `train` / `test`. Mirrors `classify_experiment_type` (Beit Yitzchaki = test, everything else = train). |
| `--source`           | _any_                 | Filter by `metadata.source` (`experiment` / `ido` / `realWorld`). Repeatable to allow multiple. |
| `--include`          | _all_                 | Whitelist of experiment folder names. Other filters still apply. |
| `--exclude`          | _none_                | Skip these experiments entirely. |
| `--cleaned-exclude`  | three Pixel/A23 outliers | Excluded *only* from the cleaned-pooled aggregates (the “after removing three outliers” parallel set). Pass an empty argument list to skip the cleaned pass. |
| `--out-root`         | `elevator_reports/seg_eval` | Base folder for the timestamped run directory. |
| `--run-name`         | `run_<timestamp>`     | Override the timestamp folder. Useful for stable diff-viewing. |

## What gets produced

Every run writes the following inside its run folder:

```
run_settings.json                       # full CLI flags + resolved config + experiment list
metrics.json                            # train / test / pooled / cleaned-pooled metrics
per_experiment.csv                      # IntervalPredictionMetrics per experiment

train/                                  # live evaluator's bundle (only if train segments survive filters)
  cdf_iou.png
  cdf_start_residual.png
  cdf_end_residual.png
  cdf_duration_error.png
  failure_modes.png

test/                                   # same on the held-out test split
  ...

# Combined figures (consumed by main.tex):
failure_modes_train_vs_test.png         # § Failure Mode Distribution
per_experiment_failure_bar.png          # § Failure Mode Distribution
cdf_pdf_iou.png                         # § Edge Quality
iou_vs_duration.png                     # § Edge Quality
pred_vs_gt_duration.png                 # diagnostic scatter
phone_breakdown.png                     # § Phone-Model Breakdown
timeline_best.png                       # § Per-Experiment Detection Timelines
timeline_typical.png                    # ↑
timeline_worst.png                      # ↑

# Cleaned-pooled (only when --cleaned-exclude is non-empty):
failure_modes_train_vs_test_cleaned.png # § After Removing Three Outlier Experiments
per_experiment_failure_bar_cleaned.png  # ↑
cdf_pdf_iou_cleaned.png                 # ↑
iou_vs_duration_cleaned.png             # ↑

# Constraint-justification (rendered when src/segmentation/algorithms/improvement_iterations/
# iter_16_lower_peak_a/per_gt.csv is present):
constraint_pair_joint_r2.png            # § Why These Constraints Hold
constraint_pair_A.png
constraint_heatmap_energy.png
constraint_jointR2_vs_pairA.png
constraint_peak_R2.png
constraint_peak_A.png
constraint_reject_reasons.png
```

## Mapping: figures ↔ `main.tex`

| Figure                              | `main.tex` reference |
|---|---|
| `failure_modes_train_vs_test.png`   | `\label{fig:seg-eval-failure-split}` |
| `per_experiment_failure_bar.png`    | `\label{fig:seg-eval-failure-perexp}` |
| `cdf_pdf_iou.png`                   | `\label{fig:seg-eval-iou-dist}` |
| `iou_vs_duration.png`               | `\label{fig:seg-eval-duration-scatter}` |
| `phone_breakdown.png`               | `\label{fig:seg-eval-phones}` |
| `timeline_best/typical/worst.png`   | `\label{fig:seg-eval-timeline-*}` |
| `failure_modes_train_vs_test_cleaned.png` / `cdf_pdf_iou_cleaned.png` | `\label{fig:seg-eval-cleaned-diagnostics}` |
| `per_experiment_failure_bar_cleaned.png` | `\label{fig:seg-eval-cleaned-perexp}` |
| `iou_vs_duration_cleaned.png`       | `\label{fig:seg-eval-cleaned-duration}` |
| `constraint_*.png`                  | `\label{fig:seg-eval-constraint-*}` |

## Things this script does *not* do

* **Hyperparameter sweeps.** The `seg_sweep/sweep_*.png` figures in
  `main.tex` come from a separate driver. Use the existing sweep entry
  point (`scripts/sweep_acc_segmentation.py`) or
  `python -m src.segmentation.evaluate --sweep grid.json` for those.
* **LaTeX macro / table emission.** That stays in
  `scripts/segmentation_evaluation_report.py` (which writes into
  `docs/latex/figures/seg_eval/`). `evaluateOnData.py` is the
  data-evaluation companion: it produces the same figures so you can
  inspect them without recompiling LaTeX.

## Common recipes

```bash
# Train-only run (skip the held-out Beit Yitzchaki test set):
venv/bin/python -m src.segmentation.evaluate.evaluateOnData --kind train

# Evaluate on every controlled-experiment recording, ignoring the three
# damped-phone outliers:
venv/bin/python -m src.segmentation.evaluate.evaluateOnData \
    --source experiment \
    --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 \
              UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 \
              eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1

# Run on a single experiment for debugging:
venv/bin/python -m src.segmentation.evaluate.evaluateOnData \
    --include eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 \
    --cleaned-exclude
```

When you add new recordings to `rawData/`, just rerun the default
command — the script will pick them up automatically through
`list_experiments()` and produce a fresh report folder.
