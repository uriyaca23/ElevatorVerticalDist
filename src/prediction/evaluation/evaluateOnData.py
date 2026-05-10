"""Reproducible Δh-prediction evaluation across an experiment subset.

Loads every elevator segment surviving the requested filters, runs the
two accelerometer-only predictors (ZUPT + trapezoid) on them, refits the
conformal calibrator on the train half (when train segments are
present), then renders the per-algorithm figure bundle that
``docs/latex/prediction_sections.tex`` consumes (scatter, error CDF,
per-segment CI bars, coverage-by-distance bins, reliability diagram,
signed-error scatter, …) plus the cross-algorithm comparison CDF on
both train and test halves.

Typical usage::

    venv/bin/python -m src.prediction.evaluation.evaluateOnData
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --kind train --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

A timestamped ``run_YYYYMMDD-HHMMSS/`` directory is created under
``--out-root`` (default ``elevator_reports/pred_eval``). It contains:

* ``run_settings.json``           — every flag, resolved experiments,
                                    active config dump.
* ``train/figures_<algo>/``       — per-algo figure bundle.
* ``test/figures_<algo>/``        — same on the held-out test half.
* ``train/predictions_<algo>.csv``, ``test/predictions_<algo>.csv``.
* ``train/per_experiment_<algo>.csv`` (and the test mirror).
* ``train/calibration_<algo>.json`` — refit conformal checkpoint.
* ``train/fig_compare_algorithms.png`` (and test mirror).
* ``metrics.json``                — aggregate :class:`MetricsBundle`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.loader import (
    EXPERIMENT_TYPES,
    VALID_SOURCES,
    classify_experiment_type,
    getExperimentData,
    list_experiments,
)
from src.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
    Predictor,
)

from .dataset import build_segment_records
from .figures import fig_compare_algorithms, save_all_figures
from .metrics import compute_metrics, per_experiment_metrics
from .runner import (
    collect_predictions,
    run_predictions,
    to_calibration_samples,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "elevator_reports" / "pred_eval"

ALGORITHMS: dict[str, PredictAlgorithm] = {
    "zupt":      PredictAlgorithm.ZUPT_ACCEL,
    "trapezoid": PredictAlgorithm.TRAPEZOID_ACCEL,
}


# --------------------------------------------------------------------------
# Filter helpers
# --------------------------------------------------------------------------
def _experiment_metadata(name: str) -> dict | None:
    try:
        _, _, meta = getExperimentData(name)
    except Exception:
        return None
    return meta


def _resolve_experiments(
    kind: str,
    sources: list[str] | None,
    include: list[str] | None,
    exclude: list[str] | None,
) -> list[str]:
    """Return experiment names surviving every filter (and existing on disk)."""
    candidates = list(include) if include else list_experiments(kind="all")
    excluded = set(exclude or [])
    out: list[str] = []
    for name in candidates:
        if name in excluded:
            continue
        if kind != "all" and classify_experiment_type(name) != kind:
            continue
        if sources:
            meta = _experiment_metadata(name)
            src = (meta or {}).get("source", "")
            if src not in sources:
                continue
        out.append(name)
    return out


def _load_records(experiments: list[str], verbose: bool):
    records = []
    for name in experiments:
        recs = build_segment_records(name, verbose=verbose)
        if verbose:
            print(f"  {name}: +{len(recs)} segments")
        records.extend(recs)
    return records


# --------------------------------------------------------------------------
# Per-half runner
# --------------------------------------------------------------------------
def _run_split(
    split_name: str,
    records: list,
    split_dir: Path,
    *,
    refit_calibration: bool,
    calibration_dir: Path | None = None,
) -> tuple[dict[str, pd.DataFrame], dict[str, dict]]:
    """Run every algorithm on ``records`` and dump figures + CSVs.

    When ``refit_calibration`` is True, the conformal multiplier is
    refit on the supplied records and persisted alongside the figures.
    Otherwise we expect a calibration JSON to already exist under
    ``calibration_dir``; passing ``None`` falls back to the algorithm's
    default multiplier (1.645).
    """
    split_dir.mkdir(parents=True, exist_ok=True)
    dfs: dict[str, pd.DataFrame] = {}
    summary: dict[str, dict] = {}

    if not records:
        return dfs, summary

    for algo_name, algo_enum in ALGORITHMS.items():
        fig_dir = split_dir / f"figures_{algo_name}"
        fig_dir.mkdir(exist_ok=True)
        print(f"\n[{split_name}][{algo_name}] predicting on "
              f"{len(records)} segments ...")
        t0 = time.time()
        p = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo_enum))

        if refit_calibration:
            preds = run_predictions(p, records)
            calib = p.calibrate(to_calibration_samples(preds))
            print(f"  conformal fit: {calib}")
            calib_path = split_dir / f"calibration_{algo_name}.json"
            p.save_calibration(calib_path)
        else:
            calib = None
            if calibration_dir is not None:
                calib_path = calibration_dir / f"calibration_{algo_name}.json"
                if calib_path.exists():
                    p.load_calibration(calib_path)
                    print(f"  loaded calibration from {calib_path.name}")
                else:
                    print(f"  [warn] no calibration at {calib_path}; "
                          "using algorithm default multiplier")

        # Re-run inference so CIs reflect the active multiplier.
        preds_cal = run_predictions(p, records)
        df = collect_predictions(preds_cal)
        dfs[algo_name] = df
        df.to_csv(split_dir / f"predictions_{algo_name}.csv", index=False)

        metrics = asdict(compute_metrics(df))
        summary[algo_name] = {"calibration": calib, "metrics": metrics}
        print(f"  clean coverage={metrics['clean_coverage_90']:.1%} "
              f"accepted-clean coverage={metrics['accepted_clean_coverage_90']:.1%} "
              f"median |err|={metrics['clean_median_abs_err']:.2f}m "
              f"median CI=±{metrics['clean_median_ci']:.2f}m "
              f"({time.time() - t0:.1f}s)")

        per_exp = per_experiment_metrics(df)
        per_exp.to_csv(split_dir / f"per_experiment_{algo_name}.csv",
                       index=False)
        save_all_figures(df, fig_dir,
                         label=f"{algo_name.upper()} / {split_name.upper()}")

    fig_compare_algorithms(
        {k.upper(): v for k, v in dfs.items()},
        split_dir / f"fig_compare_algorithms_{split_name}.png",
        title=f"{split_name.title()} — algorithm comparison (clean only)",
    )
    return dfs, summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluateOnData",
        description="Reproducible prediction evaluation: filter "
                    "experiments, run ZUPT + trapezoid predictors, refit "
                    "conformal, and render every figure consumed by the "
                    "prediction subsection of docs/latex/main.tex.",
    )
    p.add_argument(
        "--kind", default="all",
        choices=("all", *EXPERIMENT_TYPES),
        help="Restrict to train, test, or all experiments. With 'all' "
             "(default) the script refits calibration on the train half "
             "and applies it to the test half.",
    )
    p.add_argument(
        "--source", action="append", default=None,
        choices=list(VALID_SOURCES),
        help="Filter by metadata.source — repeatable.",
    )
    p.add_argument(
        "--include", nargs="*", default=None,
        help="Whitelist of experiment names (still subject to other filters).",
    )
    p.add_argument(
        "--exclude", nargs="*", default=None,
        help="Drop these experiment names from the run.",
    )
    p.add_argument(
        "--mode", default="auto",
        choices=("auto", "train", "test"),
        help="auto = refit on train + apply to test (default). train = "
             "only refit/score on the train half. test = only run on the "
             "test half, expecting --calibration-dir to point at an "
             "existing calibration_*.json bundle.",
    )
    p.add_argument(
        "--calibration-dir", type=Path, default=None,
        help="Directory holding calibration_<algo>.json checkpoints "
             "(used when --mode test).",
    )
    p.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Base directory; output is written to <out-root>/run_<ts>/.",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Override the timestamp folder name.",
    )
    p.add_argument("--verbose", action="store_true")
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    # --- resolve experiments per split ---
    train_exps = (_resolve_experiments(
        kind="train", sources=args.source,
        include=args.include, exclude=args.exclude,
    ) if args.kind in ("all", "train") else [])
    test_exps = (_resolve_experiments(
        kind="test", sources=args.source,
        include=args.include, exclude=args.exclude,
    ) if args.kind in ("all", "test") else [])
    if not train_exps and not test_exps:
        print("no experiments survived filtering; nothing to do",
              file=sys.stderr)
        return 1

    # --- record run settings up-front ---
    cfg_dump = {}
    for n, e in ALGORITHMS.items():
        c = PREDICT_ALGORITHM_CONFIG(algorithm=e)
        cfg_dump[n] = {
            "algorithm": c.algorithm.value,
            "config_path": str(c.config_path),
            "overrides": c.overrides,
            "active_params": c.load_params(),
        }
    settings = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "configs": cfg_dump,
        "experiments": {
            "train": train_exps, "test": test_exps,
            "n_train": len(train_exps), "n_test": len(test_exps),
        },
    }
    (run_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )

    # --- load segments per split ---
    print(f"\nloading segments  (train={len(train_exps)} exps, "
          f"test={len(test_exps)} exps)")
    t0 = time.time()
    train_records = _load_records(train_exps, args.verbose) if train_exps else []
    test_records  = _load_records(test_exps,  args.verbose) if test_exps  else []
    print(f"  → {len(train_records)} train segments, "
          f"{len(test_records)} test segments "
          f"({time.time() - t0:.1f}s)")

    # --- run train and/or test ---
    summary: dict = {}
    if args.mode in ("auto", "train") and train_records:
        train_dir = run_dir / "train"
        _, train_metrics = _run_split(
            "train", train_records, train_dir,
            refit_calibration=True,
        )
        summary["train"] = train_metrics

    if args.mode in ("auto", "test") and test_records:
        test_dir = run_dir / "test"
        # In auto mode the calibration we just refit lives under run_dir/train.
        # In test-only mode the user must point us at an existing calibration.
        if args.mode == "auto":
            calib_dir = run_dir / "train"
        else:
            if args.calibration_dir is None:
                print("--mode test requires --calibration-dir; using "
                      "default multiplier", file=sys.stderr)
                calib_dir = None
            else:
                calib_dir = args.calibration_dir
        _, test_metrics = _run_split(
            "test", test_records, test_dir,
            refit_calibration=False,
            calibration_dir=calib_dir,
        )
        summary["test"] = test_metrics

    (run_dir / "metrics.json").write_text(
        json.dumps(summary, indent=2, default=str)
    )
    print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
