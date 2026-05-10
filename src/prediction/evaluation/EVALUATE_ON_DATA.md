# `evaluateOnData` — Prediction evaluation runner

`src/prediction/evaluation/evaluateOnData.py` is the canonical entry
point for re-evaluating the two accelerometer-only Δh predictors (ZUPT
+ trapezoid pulse-pair) every time the dataset changes. It loads every
elevator segment under your filters, runs both predictors, refits the
conformal calibrator on the train half, applies it to the test half,
and writes the per-algorithm figure bundle that the *Prediction
Results* subsection of `docs/latex/main.tex` (`prediction_sections.tex`)
consumes — into a timestamped folder.

## TL;DR

```bash
# from repo root, with venv activated
venv/bin/python -m src.prediction.evaluation.evaluateOnData
```

Output lands under
`elevator_reports/pred_eval/run_<YYYYMMDD-HHMMSS>/`.

## CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--kind`              | `all`           | `all` / `train` / `test`. With `all` (default), the script refits conformal on the train half and applies it to the test half. |
| `--source`            | _any_           | Filter by `metadata.source` (`experiment` / `ido` / `realWorld`). Repeatable. |
| `--include`           | _all_           | Whitelist of experiment folder names. |
| `--exclude`           | _none_          | Drop these experiment names from the run. |
| `--mode`              | `auto`          | `auto` (refit on train + apply to test), `train` (refit only), or `test` (apply existing calibration only — requires `--calibration-dir`). |
| `--calibration-dir`   | _none_          | Directory holding `calibration_<algo>.json` checkpoints. Used in `--mode test`. |
| `--out-root`          | `elevator_reports/pred_eval` | Base folder for the timestamped run directory. |
| `--run-name`          | `run_<timestamp>` | Override the timestamp folder. |
| `--verbose`           | _off_           | Print per-experiment segment counts during loading. |

## What gets produced

```
run_settings.json                       # CLI flags + active configs + resolved exp list
metrics.json                            # MetricsBundle per (split, algorithm)

train/
  predictions_zupt.csv
  predictions_trapezoid.csv
  per_experiment_zupt.csv
  per_experiment_trapezoid.csv
  calibration_zupt.json                 # refit conformal checkpoint
  calibration_trapezoid.json
  fig_compare_algorithms_train.png      # cross-algorithm CDF
  figures_zupt/
    fig_scatter.png                     # § Figures — predicted vs true
    fig_cdf.png                         # § Figures — error CDF
    fig_hist.png
    fig_per_ride.png
    fig_ci.png                          # § Figures — per-segment CI bars
    fig_quality.png
    fig_per_exp.png
    fig_reject.png
    fig_ci_vs_dh.png
    fig_cov_bins.png                    # § Figures — coverage by distance bin
    fig_reliability.png                 # § Figures — reliability diagram
    fig_signed_err.png                  # § Figures — signed error
  figures_trapezoid/
    ...                                 # same set under the trapezoid algorithm

test/                                   # same layout, but predictions are made
  ...                                   # with the train-side calibration applied
  fig_compare_algorithms_test.png
```

## Mapping: figures ↔ `prediction_sections.tex`

| Figure (per-algo)                  | `prediction_sections.tex` reference |
|---|---|
| `figures_<algo>/fig_scatter.png`        | `\includegraphics{prediction/<split>_<algo>_scatter.png}` |
| `figures_<algo>/fig_cdf.png`            | `prediction/<split>_<algo>_cdf.png` |
| `figures_<algo>/fig_ci.png`             | `prediction/<split>_<algo>_ci.png` |
| `figures_<algo>/fig_cov_bins.png`       | `prediction/<split>_<algo>_cov_bins.png` |
| `figures_<algo>/fig_reliability.png`    | `prediction/<split>_<algo>_reliability.png` |
| `figures_<algo>/fig_signed_err.png`     | `prediction/<split>_<algo>_signed_err.png` |
| `fig_compare_algorithms_<split>.png`    | `prediction/<split>_fig_compare_algorithms_<split>.png` |

The `scripts/build_prediction_report_assets.py` driver still owns the
LaTeX-side staging (copying figures into `docs/latex/figures/prediction/`
under the `<split>_<algo>_<key>.png` naming convention and emitting
`results_macros.tex`). `evaluateOnData.py` produces the same figures so
you can inspect them directly without recompiling LaTeX.

## Things this script does *not* do

* **Comparison with the barometer baseline** — the prediction LaTeX
  section uses ZUPT + trapezoid only. Add `BarometerHeightDiffConfig`
  manually if you need a sanity baseline.
* **Per-segment failure analysis** — the *Worst Predictions* section in
  the report comes from `scripts/analyze_short_ride_failures.py`. This
  driver only emits the aggregate figures.
* **Manual GT-correction visualisation** — that's `scripts/audit_gt_spikes.py`.

## Common recipes

```bash
# Default: refit calibration on train, apply to test, render both halves.
venv/bin/python -m src.prediction.evaluation.evaluateOnData

# Train-only sanity pass (skip the held-out test set):
venv/bin/python -m src.prediction.evaluation.evaluateOnData --kind train

# Reuse an existing calibration on the test half:
venv/bin/python -m src.prediction.evaluation.evaluateOnData \
    --kind test --mode test \
    --calibration-dir elevator_reports/pred_eval/run_20260507-101500/train

# Evaluate on controlled experiments only, dropping a known-bad one:
venv/bin/python -m src.prediction.evaluation.evaluateOnData \
    --source experiment \
    --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026
```

When you add new recordings, rerunning the default command picks them
up automatically; the calibration is refit fresh from the full surviving
train pool every time.
