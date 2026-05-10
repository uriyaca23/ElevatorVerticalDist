# `evaluateOnData` — End-to-end pipeline evaluation

`src/pipelines/evaluate/evaluateOnData.py` is the canonical entry point
for re-evaluating the deployed pipeline (segmentation → prediction → Δh)
every time the dataset changes. It runs the segmenter, then the
predictor, then the barometer truth, on every experiment surviving your
filters; builds the three error views (GT-segments / matched-segmenter /
all-vs-barometer); and renders the figures consumed by the *Pipeline*
section of `docs/latex/main.tex` for both the full pass *and* the
accepted-only (post-quality-filter) pass — into a timestamped folder.

## TL;DR

```bash
# from repo root, with venv activated
venv/bin/python -m src.pipelines.evaluate.evaluateOnData
```

Output lands under
`elevator_reports/pipeline_eval/run_<YYYYMMDD-HHMMSS>/`.

## CLI flags

| Flag | Default | What it does |
|---|---|---|
| `--seg-algorithm`     | `acc_template_match`    | Segmentation algorithm. |
| `--pred-algorithm`    | `trapezoid_accel`       | Prediction algorithm. |
| `--kind`              | `all`                   | `all` / `train` / `test`. Mirrors `classify_experiment_type`. |
| `--source`            | _any_                   | Filter by `metadata.source` (`experiment` / `ido` / `realWorld`). Repeatable. |
| `--include`           | _all_                   | Whitelist of experiment folder names. |
| `--exclude`           | _none_                  | Drop these experiment names from the run. |
| `--calibration-path`  | `src/data/structuredData/test_results/prediction/train/calibration_trapezoid.json` | Conformal-calibration JSON to load onto the predictor before predicting. |
| `--out-root`          | `elevator_reports/pipeline_eval` | Base folder for the timestamped run directory. |
| `--run-name`          | `run_<timestamp>`       | Override the timestamp folder. |

## What gets produced

```
run_settings.json                       # CLI flags + seg & pred configs + experiment list
metrics.json                            # three-views summaries (full + accepted) + FP/accept stats
gt_records.csv                          # one row per GT ride
seg_records.csv                         # one row per predicted segment

# Full-pass figures (every prediction the predictor returned):
cdf_pooled.png                          # § Error Distributions
cdf_pooled_zoom.png
cdf_train.png
cdf_test.png
bar_mae_overall.png                     # § MAE at a Glance
per_exp_mae.png                         # § Per-Experiment View
scatter_three.png                       # § Error Distributions
signed_error_pdf.png                    # ↑
fp_predicted_dh.png                     # § Phantom Altitude
fp_predicted_altitude.png               # ↑
fp_vs_clean_dh.png                      # ↑
clean_predicted_altitude.png            # ↑
baro_vs_gt_sanity.png                   # § Sanity Check

# Accepted-only pass (same set, restricted to predictions accepted by the quality filter):
cdf_pooled_acc.png
cdf_pooled_zoom_acc.png
cdf_train_acc.png
cdf_test_acc.png
bar_mae_overall_acc.png
per_exp_mae_acc.png
scatter_three_acc.png
signed_error_pdf_acc.png
fp_predicted_dh_acc.png
fp_predicted_altitude_acc.png
fp_vs_clean_dh_acc.png
clean_predicted_altitude_acc.png
```

## Mapping: figures ↔ `main.tex`

| Figure                          | `main.tex` reference |
|---|---|
| `cdf_pooled.png`                | `\label{fig:pipe-cdf-pooled}` |
| `cdf_pooled_zoom.png`           | `\label{fig:pipe-cdf-pooled-zoom}` |
| `cdf_train.png` / `cdf_test.png` | `\label{fig:pipe-cdf-split}` |
| `bar_mae_overall.png`           | `\label{fig:pipe-bar-mae}` |
| `per_exp_mae.png`               | `\label{fig:pipe-per-exp-mae}` |
| `scatter_three.png`             | `\label{fig:pipe-scatter-three}` |
| `signed_error_pdf.png`          | `\label{fig:pipe-signed-pdf}` |
| `fp_predicted_dh.png`           | `\label{fig:pipe-fp-dh}` |
| `fp_predicted_altitude.png`     | `\label{fig:pipe-fp-altitude}` |
| `fp_vs_clean_dh.png`            | `\label{fig:pipe-fp-vs-clean}` |
| `clean_predicted_altitude.png`  | `\label{fig:pipe-clean-altitude}` |
| `baro_vs_gt_sanity.png`         | `\label{fig:pipe-baro-vs-gt}` |
| `*_acc.png`                     | accepted-only mirrors in § Adding the Quality Filter |

## Things this script does *not* do

* **Refit the predictor's conformal calibrator.** It expects an
  existing `calibration_*.json` and bails to the algorithm's default
  multiplier when the file is missing. To refit, run
  `python -m src.prediction.evaluation.evaluateOnData` first and point
  `--calibration-path` at the resulting `train/calibration_trapezoid.json`.
* **LaTeX macro / table emission.** That stays in
  `scripts/pipeline_evaluation_report.py` (which writes into
  `docs/latex/figures/pipeline/`). `evaluateOnData.py` is the
  data-evaluation companion: same figures, no LaTeX side effects.

## Common recipes

```bash
# Default: deployed configs, default calibration, all experiments.
venv/bin/python -m src.pipelines.evaluate.evaluateOnData

# Pipeline on the train split only:
venv/bin/python -m src.pipelines.evaluate.evaluateOnData --kind train

# Use a fresh calibration just produced by the prediction evaluator:
venv/bin/python -m src.pipelines.evaluate.evaluateOnData \
    --calibration-path \
      elevator_reports/pred_eval/run_20260507-101500/train/calibration_trapezoid.json

# Skip the three damped-phone outliers from the segmentation report:
venv/bin/python -m src.pipelines.evaluate.evaluateOnData \
    --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026 \
              UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2 \
              eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1
```

## Programmatic API

If you want to compose the runner without the CLI, the primitives live
in `src.pipelines.evaluate.runner`:

```python
from pathlib import Path
import pandas as pd

from src.pipelines.evaluate import (
    PipelineConfig, build_views, render_view_figures,
)
from src.pipelines.evaluate.runner import run_all_experiments

cfg = PipelineConfig(
    calibration_path=Path("…/calibration_trapezoid.json"),
)
gt_df, seg_df = run_all_experiments(["exp_a", "exp_b"], cfg, verbose=True)

# Three views (mae / median / rmse / coverage):
views = build_views(gt_df, seg_df, accepted_only=False)
print({k: v["summary"] for k, v in views.items()})

# Render every figure into ./out/:
render_view_figures(gt_df, seg_df, Path("out"), suffix="")
```

When you add new recordings, rerunning the default CLI command picks
them up automatically.
