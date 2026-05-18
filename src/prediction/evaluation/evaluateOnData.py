"""Reproducible Δh-prediction evaluation across an experiment subset.

Loads every elevator segment surviving the requested filters, runs the
two accelerometer-only predictors (ZUPT + trapezoid) on them, fits (or
loads) the conformal calibrator, then renders the per-algorithm figure
bundle (scatter, error CDF, per-segment CI bars, coverage-by-distance
bins, reliability diagram, signed-error scatter, …) plus the
cross-algorithm comparison CDF.

``--kind`` picks which experiments feed the run — ``train``, ``test`` or
``all`` — and the run works on exactly that data, mirroring the
segmentation and pipeline evaluators. Every run renders three figure
bundles — for the full set and the clean / noisy subsets — into ``all/``,
``clean/`` and ``noisy/`` sub-directories, so one run shows prediction
accuracy on each noise class. ``signal_clear`` is sourced from
``signalClearRecording`` in gt.csv.

Conformal calibration is fit once on the resolved segments and applied
to all three subsets. Pass ``--calibration-dir`` to load an existing
``calibration_<algo>.json`` bundle instead (the held-out workflow: one
``--kind train`` run produces the calibration, a second
``--kind test --calibration-dir <that run>`` scores the test half).

Typical usage::

    # 1. Defaults: every source, train+test, refit calibration
    venv/bin/python -m src.prediction.evaluation.evaluateOnData

    # 2. One source only
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source experiment

    # 3. Two sources (Ido + real-world)
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source ido --source real_world

    # 4. Train half only (refit + score on the train experiments)
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --kind train --source experiment

    # 5. Test half, reusing a calibration produced by an earlier run
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --kind test \\
        --calibration-dir elevator_reports/pred_eval/run_<earlier-train>

    # 6. Drop a known-bad experiment
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

    # 7. Whitelist a couple of experiments
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --include eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 \\
                  UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1

    # 8. Custom output root + stable run name
    venv/bin/python -m src.prediction.evaluation.evaluateOnData \\
        --source experiment \\
        --out-root /tmp/pred_eval --run-name source_experiment_only

Each invocation writes a timestamped directory ``run_YYYYMMDD-HHMMSS/``
under ``--out-root`` (default ``elevator_reports/pred_eval``):

* ``run_settings.json``           — every flag, resolved experiments,
                                    active config dump.
* ``calibration_<algo>.json``     — conformal checkpoint (refit runs).
* ``all/`` ``clean/`` ``noisy/``  — one sub-directory per noise subset,
  each holding ``predictions_<algo>.csv``, ``per_experiment_<algo>.csv``,
  ``figures_<algo>/`` (per-algorithm figure bundle) and
  ``fig_compare_algorithms.png``.
* ``metrics.json``                — aggregate :class:`MetricsBundle`
                                    per subset per algorithm.
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

# Noise subsets rendered every run. Each selector slices a prediction
# DataFrame by its ``signal_clear`` column (True == clean recording,
# sourced from gt.csv:signalClearRecording).
NOISE_SUBSETS: dict[str, Callable[[pd.DataFrame], pd.DataFrame]] = {
    "all":   lambda df: df,
    "clean": lambda df: df[df["signal_clear"] == True],   # noqa: E712
    "noisy": lambda df: df[df["signal_clear"] == False],  # noqa: E712
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
    return sorted(out)


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
    refit_calibration: bool,
    calibration_dir: Path | None = None,
) -> dict[str, dict]:
    """Predict with each algorithm, then write a figure bundle + metrics
    for the full set and the clean / noisy subsets, each into its own
    ``all/`` / ``clean/`` / ``noisy/`` sub-directory of ``run_dir``.

    Conformal calibration is fit once on ``records`` (or loaded from
    ``calibration_dir``) and applied to all three subsets.
    """
    # 1. predict once per algorithm on the full resolved record set
    full_dfs: dict[str, pd.DataFrame] = {}
    calibs: dict[str, dict | None] = {}
    for algo_name, algo_enum in ALGORITHMS.items():
        print(f"\n[{algo_name}] predicting on {len(records)} segments ...")
        t0 = time.time()
        p = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo_enum))

        if refit_calibration:
            preds = run_predictions(p, records)
            calib = p.calibrate(to_calibration_samples(preds))
            print(f"  conformal fit: {calib}")
            p.save_calibration(run_dir / f"calibration_{algo_name}.json")
        else:
            calib = None
            if calibration_dir is not None:
                calib_path = calibration_dir / f"calibration_{algo_name}.json"
                if calib_path.exists():
                    p.load_calibration(calib_path)
                    print(f"  loaded calibration from {calib_path}")
                else:
                    print(f"  [warn] no calibration at {calib_path}; "
                          "using algorithm default multiplier")

        # Re-run inference so CIs reflect the active multiplier.
        preds_cal = run_predictions(p, records)
        full_dfs[algo_name] = collect_predictions(preds_cal)
        calibs[algo_name] = calib
        print(f"  done ({time.time() - t0:.1f}s)")

    # 2. one figure bundle + metrics per noise subset, in its own subdir
    summary: dict[str, dict] = {}
    for subset, select in NOISE_SUBSETS.items():
        sub_dir = run_dir / subset
        sub_dir.mkdir(parents=True, exist_ok=True)
        sub_dfs: dict[str, pd.DataFrame] = {}
        algo_summary: dict[str, dict] = {}

        for algo_name in ALGORITHMS:
            df = select(full_dfs[algo_name])
            df.to_csv(sub_dir / f"predictions_{algo_name}.csv", index=False)
            if df.empty:
                algo_summary[algo_name] = {
                    "calibration": calibs[algo_name],
                    "n_segments": 0, "note": "empty subset",
                }
                continue
            sub_dfs[algo_name] = df
            algo_summary[algo_name] = {
                "calibration": calibs[algo_name],
                "n_segments": int(len(df)),
                "metrics": asdict(compute_metrics(df)),
            }
            per_experiment_metrics(df).to_csv(
                sub_dir / f"per_experiment_{algo_name}.csv", index=False)
            fig_dir = sub_dir / f"figures_{algo_name}"
            fig_dir.mkdir(exist_ok=True)
            save_all_figures(df, fig_dir,
                             label=f"{algo_name.upper()} / {subset.upper()}")

        if sub_dfs:
            fig_compare_algorithms(
                {k.upper(): v for k, v in sub_dfs.items()},
                sub_dir / "fig_compare_algorithms.png",
                title=f"{subset.title()} — algorithm comparison (clean only)",
            )
        n_seg = max((len(d) for d in sub_dfs.values()), default=0)
        print(f"  [{subset}] {n_seg} segments → {sub_dir}")
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
    p.add_argument(
        "--kind", default="all",
        choices=("all", *EXPERIMENT_TYPES),
        help="Which experiments to run on — train, test, or all "
             "(default). The run works on exactly that data.",
    )
    p.add_argument(
        "--source", action="append", default=None,
        choices=[*VALID_SOURCES, "all"],
        help="Filter by metadata.source — repeatable. Pass 'all' (or "
             "omit the flag) to keep every source.",
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
        "--calibration-dir", type=Path, default=None,
        help="Directory holding calibration_<algo>.json checkpoints. When "
             "given, calibration is loaded from it instead of refit on "
             "the resolved segments (the held-out test workflow).",
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
    args = p.parse_args(argv)
    # 'all' is a convenience alias for "no source filter".
    if args.source and "all" in args.source:
        args.source = None
    return args


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    experiments = _resolve_experiments(
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

    # --- refit calibration on the resolved segments, or load an
    # existing bundle when --calibration-dir is given ---
    refit = args.calibration_dir is None
    if refit:
        print("refitting conformal calibration on the resolved segments")
    else:
        print(f"loading conformal calibration from {args.calibration_dir}")
    summary = _run_evaluation(
        records, run_dir,
        refit_calibration=refit, calibration_dir=args.calibration_dir,
    )

    metrics = {
        "kind": args.kind,
        "calibration_refit": refit,
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
