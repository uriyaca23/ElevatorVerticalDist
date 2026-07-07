"""Reproducible Δh-prediction evaluation across an experiment subset.

Loads every elevator segment surviving the requested filters, runs the
two accelerometer-only predictors (ZUPT + trapezoid) on them, and
renders a per-algorithm figure bundle (scatter, error CDF, per-segment
CI bars, coverage-by-distance bins, reliability diagram, signed-error
scatter, …) plus the cross-algorithm comparison CDF.

Conformal calibration lives with each algorithm as a committed
``calibration.json`` next to its code
(``src/prediction/algorithms/accelerometer_only/<algo>/calibration.json``)
and is auto-loaded whenever a ``Predictor`` is built — so every run
here, and every other consumer (the pipeline evaluator, the analysis
scripts), is calibrated by default with no wiring. Pass ``--calibrate``
to refit the conformal multiplier on the resolved segments and
**overwrite** that committed file (typically ``--kind train
--calibrate``). Without the flag the committed calibration is used
as-is and never touched, so a ``--kind test`` run cannot leak into the
calibration.

``--kind`` picks which experiments feed the run — ``train``, ``test`` or
``all`` — and the run works on exactly that data, mirroring the
segmentation and pipeline evaluators. ``signal_clear`` is sourced from
``signalClearRecording`` in gt.csv.

Each invocation writes a self-describing directory
``{timestamp}_pred_{kind}/`` under ``--out-root`` (default
``elevator_reports``). Figures are split three ways — by noise class,
then algorithm, then the quality filter's accept decision::

    20260701-124443_pred_test/
      run_settings.json
      metrics.json
      all/  clean/  noisy/                 # noise class (all = every segment)
        fig_compare_algorithms.png         # cross-algorithm, per noise class
        zupt/  trapezoid/                  # algorithm
          predictions.csv
          per_experiment.csv
          accepted/  fig_*.png             # quality-filter acceptance split
          rejected/  fig_*.png
          all/       fig_*.png

Typical usage::

    # 1. Refit calibration on the train half and overwrite the committed
    #    calibration.json for both algorithms.
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --kind train --calibrate

    # 2. Score the held-out test half with the committed calibration
    #    (no refit — the test set never touches calibration).
    venv/bin/python -m src.prediction.evaluation.evaluateOnData --kind test

    # 3. One source only
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source experiment

    # 4. Two sources (Ido + real-world)
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source ido --source real_world

    # 5. Drop a known-bad experiment
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

    # 6. Whitelist a couple of experiments
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --include eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 \\
                  UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1

    # 7. Custom output root + stable run name
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source experiment \\
        --out-root /tmp/pred_eval --run-name source_experiment_only
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pandas as pd

from src.data.loader import (
    add_selection_args,
    resolve_experiments,
)
from src.physics.reconstruct_az import RECONSTRUCT_CHOICES
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
DEFAULT_OUT_ROOT = REPO_ROOT / "elevator_reports"

ALGORITHMS: dict[str, PredictAlgorithm] = {
    "zupt":      PredictAlgorithm.ZUPT_ACCEL,
    "trapezoid": PredictAlgorithm.TRAPEZOID_ACCEL,
}

# Noise subsets rendered every run. Each selector slices a prediction
# DataFrame by its ``signal_clear`` column (True == clean recording,
# sourced from gt.csv:signalClearRecording).
NOISE_SUBSETS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "all":   lambda df: df,
    "clean": lambda df: df[df["signal_clear"] == True],   # noqa: E712
    "noisy": lambda df: df[df["signal_clear"] == False],  # noqa: E712
}

# Acceptance subsets, sliced by the quality filter's accept/reject
# verdict (the ``accepted`` column). ``all`` keeps both.
ACCEPT_SUBSETS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "accepted": lambda df: df[df["accepted"] == True],    # noqa: E712
    "rejected": lambda df: df[df["accepted"] == False],   # noqa: E712
    "all":      lambda df: df,
}


def _load_records(experiments: list[str], verbose: bool):
    records = []
    for name in experiments:
        recs = build_segment_records(name, verbose=verbose)
        if verbose:
            print(f"  {name}: +{len(recs)} segments")
        records.extend(recs)
    return records


# --------------------------------------------------------------------------
# Evaluation runner
# --------------------------------------------------------------------------
def _run_evaluation(
    records: list,
    run_dir: Path,
    *,
    calibrate: bool,
    reconstruct: str = "none",
) -> dict[str, dict]:
    """Predict with each algorithm and render a figure bundle + metrics
    into a ``noise -> algorithm -> acceptance`` tree under ``run_dir``.

    Each algorithm's ``Predictor`` auto-loads its committed
    ``calibration.json`` at construction. When ``calibrate`` is set the
    conformal multiplier is refit on ``records`` and that committed file
    is overwritten before inference is re-run so the CIs reflect it.
    """
    # 1. predict once per algorithm on the full resolved record set
    full_dfs: dict[str, pd.DataFrame] = {}
    calibs: dict[str, dict | None] = {}
    for algo_name, algo_enum in ALGORITHMS.items():
        print(f"\n[{algo_name}] predicting on {len(records)} segments ...")
        t0 = time.time()
        p = Predictor(PREDICT_ALGORITHM_CONFIG(
            algorithm=algo_enum, reconstruct=reconstruct))

        calib: dict | None = None
        if calibrate:
            preds = run_predictions(p, records)
            calib = p.calibrate(to_calibration_samples(preds))
            print(f"  conformal fit: {calib}")
            # Overwrite the algorithm's own committed calibration.json.
            p.save_calibration()
            print(f"  saved calibration → {p._algo_impl.CALIBRATION_PATH}")

        # Re-run inference so CIs reflect the active multiplier.
        preds_cal = run_predictions(p, records)
        full_dfs[algo_name] = collect_predictions(preds_cal)
        calibs[algo_name] = calib
        print(f"  done ({time.time() - t0:.1f}s)")

    # 2. render into noise -> algorithm -> {accepted, rejected, all}
    summary: dict[str, dict] = {}
    for subset, select in NOISE_SUBSETS.items():
        sub_dir = run_dir / subset
        sub_dir.mkdir(parents=True, exist_ok=True)
        compare_dfs: dict[str, pd.DataFrame] = {}
        algo_summary: dict[str, dict] = {}

        for algo_name in ALGORITHMS:
            ndf = select(full_dfs[algo_name]).copy()
            algo_dir = sub_dir / algo_name
            algo_dir.mkdir(parents=True, exist_ok=True)
            ndf.to_csv(algo_dir / "predictions.csv", index=False)
            if ndf.empty:
                algo_summary[algo_name] = {
                    "calibration": calibs[algo_name],
                    "n_segments": 0, "note": "empty subset",
                }
                continue
            per_experiment_metrics(ndf).to_csv(
                algo_dir / "per_experiment.csv", index=False)
            compare_dfs[algo_name] = ndf
            algo_summary[algo_name] = {
                "calibration": calibs[algo_name],
                "n_segments": int(len(ndf)),
                "metrics": asdict(compute_metrics(ndf)),
            }

            for acc_name, acc_select in ACCEPT_SUBSETS.items():
                adf = acc_select(ndf).copy()
                acc_dir = algo_dir / acc_name
                acc_dir.mkdir(parents=True, exist_ok=True)
                if adf.empty:
                    print(f"  [skip] {subset}/{algo_name}/{acc_name}: "
                          "empty subset")
                    continue
                save_all_figures(
                    adf, acc_dir,
                    label=f"{algo_name.upper()} / {subset.upper()} / "
                          f"{acc_name}",
                )

        if compare_dfs:
            fig_compare_algorithms(
                {k.upper(): v for k, v in compare_dfs.items()},
                sub_dir / "fig_compare_algorithms.png",
                title=f"{subset.title()} — algorithm comparison",
            )
        n_seg = max((len(d) for d in compare_dfs.values()), default=0)
        print(f"  [{subset}] up to {n_seg} segments → {sub_dir}")
        summary[subset] = algo_summary

    return summary


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluateOnData",
        description="Reproducible prediction evaluation: filter "
                    "experiments, run ZUPT + trapezoid predictors, fit or "
                    "load conformal calibration, and render every "
                    "evaluation figure into the run directory.",
    )
    add_selection_args(p)
    p.add_argument(
        "--calibrate", action="store_true",
        help="Refit the conformal multiplier on the resolved segments and "
             "overwrite each algorithm's committed calibration.json "
             "(typically with --kind train). Without it, the committed "
             "calibration is used as-is and never touched.",
    )
    p.add_argument(
        "--reconstruct", default="none", choices=RECONSTRUCT_CHOICES,
        help="Accel+gyro orientation reconstruction applied as the first "
             "step of prediction (default: none = raw accelerometer). Other "
             "values track device orientation and feed a virtually-flat-phone "
             "accel to both predictors.",
    )
    p.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Base directory; output is written to "
             "<out-root>/<timestamp>_pred_<kind>/.",
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
    run_dir = args.out_root / (args.run_name or f"{timestamp}_pred_{args.kind}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    experiments = resolve_experiments(
        kind=args.kind, sources=args.source,
        include=args.include, exclude=args.exclude,
    )
    if not experiments:
        print("no experiments survived filtering; nothing to do",
              file=sys.stderr)
        return 1

    # --- record run settings up-front ---
    cfg_dump = {}
    for n, e in ALGORITHMS.items():
        c = PREDICT_ALGORITHM_CONFIG(algorithm=e, reconstruct=args.reconstruct)
        cfg_dump[n] = {
            "algorithm": c.algorithm.value,
            "config_path": str(c.config_path),
            "overrides": c.overrides,
            "reconstruct": c.reconstruct,
            "active_params": c.load_params(),
        }
    settings = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "configs": cfg_dump,
        "experiments": {
            "names": experiments,
            "n": len(experiments),
        },
        "kind": args.kind,
    }
    (run_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )

    # --- load segments for the resolved experiments ---
    print(f"\nloading segments  ({len(experiments)} experiments)")
    t0 = time.time()
    records = _load_records(experiments, args.verbose)
    print(f"  → {len(records)} segments ({time.time() - t0:.1f}s)")
    if not records:
        print("no segments after filtering; nothing to do", file=sys.stderr)
        return 2

    # --- calibration: refit + overwrite each algorithm's committed
    # calibration.json when --calibrate is set, else use it as-is ---
    if args.calibrate:
        print("refitting conformal calibration on the resolved segments "
              "(overwrites each algorithm's committed calibration.json)")
    else:
        print("using each algorithm's committed calibration.json "
              "(pass --calibrate to refit)")
    if args.reconstruct != "none":
        print(f"orientation reconstruction: {args.reconstruct}")
    summary = _run_evaluation(
        records, run_dir, calibrate=args.calibrate,
        reconstruct=args.reconstruct,
    )

    metrics = {
        "kind": args.kind,
        "calibrated": args.calibrate,
        "reconstruct": args.reconstruct,
        "n_segments": len(records),
        "by_noise": summary,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
