"""Parallel hyperparameter sweep for the accelerometer template-match segmenter.

Runs independent 1-D sweeps on every tunable of :class:`TemplateMatchConfig`
and (optionally) a final combined-best evaluation. Results are written as
per-parameter CSVs + PNG figures, plus a top-level summary JSON under
``elevator_reports/seg_acc_sweep/``.
"""

from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import list_experiments
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.evaluate.evaluator import (
    _pool_intervals,
    _run_on_experiments,
)
from src.segmentation.algorithms.metrics import IntervalPredictionMetrics


OUT_DIR = ROOT / "elevator_reports" / "seg_acc_sweep"
OUT_DIR.mkdir(parents=True, exist_ok=True)


# --------------------------------------------------------------------------
# Baseline + sweep grid
# --------------------------------------------------------------------------
BASELINE: dict = {
    "r2_peak_thresh":        0.80,
    "min_peak_abs_a":        0.5,
    "nms_radius_s":          0.5,
    "same_sign_min_gap_s":   3.0,
    "min_ride_s":            0.0,
    "max_ride_s":            120.0,
    "joint_r2_thresh":       0.75,
    "min_pair_abs_a":        0.5,
    "heatmap_energy_thresh": 0.30,
    "w_min_s":               0.4,
    "w_max_s":               3.0,
    "n_w":                   30,
    "f_min":                 0.05,
    "f_max":                 0.80,
    "n_f":                   15,
    "noise_sigma_multiplier": 6.0,
}

SWEEPS: dict[str, list] = {
    "r2_peak_thresh":        [0.55, 0.65, 0.70, 0.75, 0.80, 0.85, 0.90],
    "min_peak_abs_a":        [0.10, 0.25, 0.40, 0.50, 0.70, 1.00, 1.50],
    "nms_radius_s":          [0.10, 0.25, 0.50, 1.00, 1.50, 2.00],
    "same_sign_min_gap_s":   [1.0, 2.0, 3.0, 4.0, 5.0, 7.0, 10.0],
    "joint_r2_thresh":       [0.50, 0.60, 0.70, 0.75, 0.80, 0.85, 0.90],
    "min_pair_abs_a":        [0.10, 0.25, 0.40, 0.50, 0.70, 1.00, 1.50],
    "heatmap_energy_thresh": [0.05, 0.15, 0.25, 0.30, 0.40, 0.50, 0.60],
    "max_ride_s":            [30.0, 45.0, 60.0, 90.0, 120.0, 180.0],
}


# --------------------------------------------------------------------------
# Worker
# --------------------------------------------------------------------------
def _evaluate_config(overrides: dict, experiments: list[str]) -> dict:
    """Run segmenter with ``overrides`` across ``experiments`` and return
    the aggregated metrics as a flat dict."""
    cfg = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        overrides=overrides,
    )
    exps = _run_on_experiments(cfg, experiments, verbose=False, phone_model=None)
    total = IntervalPredictionMetrics.sum(
        IntervalPredictionMetrics.from_intervals(e.gt_rides, e.preds)
        for e in exps
    )
    pooled_gt, pooled_pred = _pool_intervals(exps)
    iou_metrics = IntervalPredictionMetrics.iou_f1(
        pooled_gt, pooled_pred, iou_threshold=0.5,
    )
    return {**overrides, **total.as_dict(), **iou_metrics}


# --------------------------------------------------------------------------
# Sweep orchestration
# --------------------------------------------------------------------------
def _parallel_evaluate(
    configs: list[dict], experiments: list[str], max_workers: int,
) -> list[dict]:
    results: list[dict] = [None] * len(configs)  # type: ignore[list-item]
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=max_workers) as pool:
        futures = {
            pool.submit(_evaluate_config, ov, experiments): i
            for i, ov in enumerate(configs)
        }
        done = 0
        for fut in as_completed(futures):
            i = futures[fut]
            results[i] = fut.result()
            done += 1
            print(
                f"  [{done:3d}/{len(configs)}] "
                f"f1={results[i]['f1_like']:.3f} "
                f"iou_f1={results[i]['iou_f1@0.5']:.3f} "
                f"({time.time() - t0:.1f}s elapsed)",
                flush=True,
            )
    return results


def run_all_sweeps(
    experiments: list[str], max_workers: int = 8,
) -> tuple[dict[str, pd.DataFrame], dict, dict]:
    # Baseline
    print(f"[baseline] {len(experiments)} experiments")
    baseline = _evaluate_config(BASELINE, experiments)
    print(
        f"  baseline f1_like={baseline['f1_like']:.3f} "
        f"iou_f1@0.5={baseline['iou_f1@0.5']:.3f}"
    )

    sweep_tables: dict[str, pd.DataFrame] = {}
    per_param_best: dict[str, dict] = {}
    for param, values in SWEEPS.items():
        print(f"\n[sweep] {param} over {values}")
        configs = [{**BASELINE, param: v} for v in values]
        results = _parallel_evaluate(configs, experiments, max_workers)
        df = pd.DataFrame(results).sort_values(param).reset_index(drop=True)
        sweep_tables[param] = df
        best_row = df.iloc[df["f1_like"].idxmax()].to_dict()
        per_param_best[param] = {
            "value": best_row[param],
            "f1_like": best_row["f1_like"],
            "iou_f1@0.5": best_row["iou_f1@0.5"],
            "recall": best_row["recall"],
            "precision": best_row["precision"],
        }
        print(
            f"  best {param}={best_row[param]}  "
            f"f1={best_row['f1_like']:.3f}  iou_f1={best_row['iou_f1@0.5']:.3f}"
        )
    return sweep_tables, baseline, per_param_best


def combined_best(per_param_best: dict[str, dict]) -> dict:
    combined = dict(BASELINE)
    for k, v in per_param_best.items():
        combined[k] = v["value"]
    return combined


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------
def plot_param_sweep(
    df: pd.DataFrame, param: str, baseline_value, out_path: Path,
) -> None:
    fig, ax1 = plt.subplots(figsize=(6.5, 4.0))
    x = df[param].to_numpy()
    ax1.plot(x, df["f1_like"], marker="o", color="tab:blue", label="f1_like")
    ax1.plot(x, df["iou_f1@0.5"], marker="s", color="tab:orange",
             label="iou_f1@0.5")
    ax1.plot(x, df["recall"], marker="^", color="tab:green",
             linestyle="--", alpha=0.6, label="recall")
    ax1.plot(x, df["precision"], marker="v", color="tab:red",
             linestyle="--", alpha=0.6, label="precision")
    ax1.axvline(baseline_value, color="grey", linestyle=":", linewidth=1.0,
                label=f"baseline ({baseline_value})")
    ax1.set_xlabel(param)
    ax1.set_ylabel("score")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)
    ax1.legend(fontsize=8, loc="best")
    ax1.set_title(f"1-D sweep: {param}")
    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# --------------------------------------------------------------------------
# Entry point
# --------------------------------------------------------------------------
def main() -> int:
    experiments = list_experiments(kind="train")
    if not experiments:
        print("no training experiments found", file=sys.stderr)
        return 1

    sweep_tables, baseline, per_param_best = run_all_sweeps(
        experiments, max_workers=8,
    )

    # Write per-param CSVs + plots
    for param, df in sweep_tables.items():
        csv_path = OUT_DIR / f"sweep_{param}.csv"
        df.to_csv(csv_path, index=False)
        plot_param_sweep(df, param, BASELINE[param],
                         OUT_DIR / f"sweep_{param}.png")

    # Evaluate the combined-best config (every sweep's winning value applied
    # together on top of the baseline)
    combined = combined_best(per_param_best)
    print("\n[combined-best] evaluating")
    combined_result = _evaluate_config(combined, experiments)
    print(
        f"  combined f1_like={combined_result['f1_like']:.3f}  "
        f"iou_f1={combined_result['iou_f1@0.5']:.3f}"
    )

    summary = {
        "baseline_config":  BASELINE,
        "baseline_metrics": baseline,
        "per_param_best":   per_param_best,
        "combined_best_config":  combined,
        "combined_best_metrics": combined_result,
        "experiments_used":      experiments,
    }
    (OUT_DIR / "summary.json").write_text(json.dumps(summary, indent=2,
                                                     default=float))
    print(f"\nwrote summary + per-param artifacts → {OUT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
