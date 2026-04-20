"""End-to-end evaluation driver for the two accelerometer-only
prediction algorithms (ZUPT and trapezoid pulse-pair).

Train mode (default) computes train-set metrics and figures and
refits the conformal calibration. Test mode runs the final blind-test
pass — I only run it once, right at the end, so intermediate
iterations cannot peek at the test set.

Usage::

    # train-set analysis + conformal fit (default)
    python scripts/run_prediction_evaluation.py

    # blind test (run once, at the end)
    python scripts/run_prediction_evaluation.py --mode test

Outputs go under ``src/data/structuredData/test_results/prediction/``
  - ``train/`` or ``test/`` sub-folders
  - ``figures_<algo>/`` — per-algorithm figure bundle
  - ``calibration_<algo>.json`` — conformal checkpoint
  - ``metrics_<mode>.json`` — aggregate metrics
  - ``predictions_<algo>_<mode>.csv`` — per-segment predictions + labels
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import pandas as pd

# Allow running as a plain script from the repo root.
_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from src.prediction.evaluation.dataset import load_all_segments
from src.prediction.evaluation.figures import save_all_figures, fig_compare_algorithms
from src.prediction.evaluation.metrics import compute_metrics, per_experiment_metrics
from src.prediction.evaluation.runner import (
    RecordPrediction, run_predictions, collect_predictions,
    to_calibration_samples,
)


ALGORITHMS: dict[str, PredictAlgorithm] = {
    "zupt":      PredictAlgorithm.ZUPT_ACCEL,
    "trapezoid": PredictAlgorithm.TRAPEZOID_ACCEL,
}

DEFAULT_OUTPUT = _REPO_ROOT / "src" / "data" / "structuredData" / "test_results" / "prediction"


def _split_records(records):
    train = [r for r in records if r.experiment_type == "train"]
    test = [r for r in records if r.experiment_type == "test"]
    return train, test


def _save_json(obj, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)


def _metrics_dict(df: pd.DataFrame) -> dict:
    m = compute_metrics(df)
    return asdict(m)


def run_train_mode(records, output_root: Path) -> dict:
    train, test = _split_records(records)
    print(f"[train-mode] train segments: {len(train)} | test withheld: {len(test)}")

    train_out = output_root / "train"
    (train_out).mkdir(parents=True, exist_ok=True)

    train_dfs: dict[str, pd.DataFrame] = {}
    results_summary: dict[str, dict] = {}

    for algo_name, algo_enum in ALGORITHMS.items():
        algo_out = train_out / f"figures_{algo_name}"
        algo_out.mkdir(exist_ok=True)
        print(f"\n[train-mode][{algo_name}] predicting on {len(train)} train segments ...")
        t0 = time.time()
        p = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo_enum))
        preds = run_predictions(p, train)
        dt = time.time() - t0
        print(f"  ... done in {dt:.1f}s")

        # 1) Fit conformal on current (uncalibrated) predictions — default
        #    multiplier is 1.645 so ci_half_width will still be meaningful.
        samples = to_calibration_samples(preds)
        calib = p.calibrate(samples)
        print(f"  [{algo_name}] conformal fit: {calib}")

        # 2) Persist the checkpoint next to the figures.
        calib_path = train_out / f"calibration_{algo_name}.json"
        p.save_calibration(calib_path)

        # 3) Re-run inference with calibrated conformal so the CI columns
        #    reflect the final scaling. (Predictions themselves are
        #    invariant to conformal — the multiplier only scales the CI.)
        preds_cal = run_predictions(p, train)
        df = collect_predictions(preds_cal)
        train_dfs[algo_name] = df
        df.to_csv(train_out / f"predictions_{algo_name}_train.csv", index=False)

        # 4) Metrics + per-experiment + figures.
        metrics = _metrics_dict(df)
        results_summary[algo_name] = {
            "calibration": calib,
            "metrics": metrics,
        }
        print(f"  [{algo_name}] clean coverage={metrics['clean_coverage_90']:.1%} "
              f"accepted-clean coverage={metrics['accepted_clean_coverage_90']:.1%} "
              f"median |err|={metrics['clean_median_abs_err']:.2f}m "
              f"median CI=±{metrics['clean_median_ci']:.2f}m")

        per_exp = per_experiment_metrics(df)
        per_exp.to_csv(train_out / f"per_experiment_{algo_name}_train.csv", index=False)
        save_all_figures(df, algo_out, label=f"{algo_name.upper()} / TRAIN")

    # Cross-algo comparison
    fig_compare_algorithms(
        {k.upper(): v for k, v in train_dfs.items()},
        train_out / "fig_compare_algorithms_train.png",
        title="Train — algorithm comparison (clean only)",
    )

    _save_json(results_summary, train_out / "metrics_train.json")
    print(f"\n[train-mode] results written under {train_out}")
    return results_summary


def run_test_mode(records, output_root: Path) -> dict:
    train, test = _split_records(records)
    print(f"[test-mode] train seen: {len(train)} | test rides: {len(test)}")
    if not test:
        raise SystemExit("No test segments found — cannot run test mode.")

    train_out = output_root / "train"
    test_out = output_root / "test"
    test_out.mkdir(parents=True, exist_ok=True)

    test_dfs: dict[str, pd.DataFrame] = {}
    results_summary: dict[str, dict] = {}

    for algo_name, algo_enum in ALGORITHMS.items():
        algo_out = test_out / f"figures_{algo_name}"
        algo_out.mkdir(exist_ok=True)
        calib_path = train_out / f"calibration_{algo_name}.json"
        if not calib_path.exists():
            raise SystemExit(
                f"No calibration found at {calib_path}; run train mode first."
            )

        print(f"\n[test-mode][{algo_name}] loading calibration from {calib_path}")
        p = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo_enum))
        p.load_calibration(calib_path)

        print(f"  predicting on {len(test)} test segments ...")
        t0 = time.time()
        preds = run_predictions(p, test)
        dt = time.time() - t0
        print(f"  ... done in {dt:.1f}s")

        df = collect_predictions(preds)
        df.to_csv(test_out / f"predictions_{algo_name}_test.csv", index=False)
        test_dfs[algo_name] = df

        metrics = _metrics_dict(df)
        results_summary[algo_name] = metrics
        print(f"  [{algo_name}] TEST clean coverage={metrics['clean_coverage_90']:.1%} "
              f"accepted-clean coverage={metrics['accepted_clean_coverage_90']:.1%} "
              f"median |err|={metrics['clean_median_abs_err']:.2f}m "
              f"median CI=±{metrics['clean_median_ci']:.2f}m")

        per_exp = per_experiment_metrics(df)
        per_exp.to_csv(test_out / f"per_experiment_{algo_name}_test.csv", index=False)
        save_all_figures(df, algo_out, label=f"{algo_name.upper()} / TEST")

    fig_compare_algorithms(
        {k.upper(): v for k, v in test_dfs.items()},
        test_out / "fig_compare_algorithms_test.png",
        title="Test — algorithm comparison (clean only)",
    )

    _save_json(results_summary, test_out / "metrics_test.json")
    print(f"\n[test-mode] results written under {test_out}")
    return results_summary


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--mode", choices=("train", "test"), default="train")
    ap.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT)
    ap.add_argument("--experiments", nargs="*", default=None,
                    help="Subset of experiment folder names to evaluate; "
                         "default = all.")
    ap.add_argument("--verbose", action="store_true")
    args = ap.parse_args()

    args.output_root.mkdir(parents=True, exist_ok=True)

    print(f"[load] scanning experiments ...")
    records = load_all_segments(
        experiments=args.experiments, verbose=args.verbose,
    )
    print(f"[load] {len(records)} elevator segments loaded "
          f"({sum(r.signal_clear for r in records)} clean).")

    if args.mode == "train":
        run_train_mode(records, args.output_root)
    else:
        run_test_mode(records, args.output_root)


if __name__ == "__main__":
    main()
