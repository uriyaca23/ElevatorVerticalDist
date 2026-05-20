"""Reproducible end-to-end pipeline evaluation.

Runs every experiment surviving the requested filters through the
deployed segmentation + prediction stack, compares against barometer
truth, and renders the pipeline figures for both passes — full and
accepted-only (post-quality-filter).

The pipeline always runs on the full resolved experiment set — there is
no noise filter. ``metrics.json`` reports the run's ``overall`` metrics
and, under ``by_noise``, the prediction accuracy on clean vs noisy GT
rides — so one run shows performance on each noise class. False
positives stay in ``overall`` only: an FP prediction matches no GT ride
and so cannot be attributed to a noise class. ``--kind`` / ``--source``
/ ``--include`` / ``--exclude`` pick *which experiments* feed the run.

Typical usage::

    # 1. Defaults: every source, train+test
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData

    # 2. One source only
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --source experiment

    # 3. Two sources (Ido + real-world, skip lab experiments)
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --source ido --source real_world

    # 4. Train only / test only
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData --kind train
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData --kind test

    # 5. Drop a known-bad experiment
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --source experiment \\
        --exclude UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026

    # 6. Whitelist a couple of experiments
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --include eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1 \\
                  UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1

    # 7. Custom output root + stable run name (no timestamp)
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --source experiment \\
        --out-root /tmp/pipe_eval --run-name source_experiment_only

    # 8. Override which segmentation / prediction algorithm runs
    venv/bin/python -m src.pipelines.evaluate.evaluateOnData \\
        --seg-algorithm acc_template_match \\
        --pred-algorithm trapezoid_accel

A timestamped ``run_YYYYMMDD-HHMMSS/`` directory is created under
``--out-root`` (default ``elevator_reports/pipeline_eval``):

* ``run_settings.json``      — every flag, resolved experiments,
                               active config dump.
* ``all/`` ``clean/`` ``noisy/`` — one sub-directory per noise subset,
  each holding ``gt_records.csv``, ``seg_records.csv`` and the full
  figure bundle (``cdf_pooled.png``, ``bar_mae_overall.png``,
  ``scatter_three.png``, ``signed_error_pdf.png``, ``fp_*.png``,
  ``clean_predicted_altitude.png``, ``per_exp_mae.png``,
  ``baro_vs_gt_sanity.png``) — each figure also with an ``_acc``
  accepted-only variant.
* ``metrics.json`` — the run's ``overall`` summaries (full + accepted,
  FP / accept-rate counts) and a ``by_noise`` block with prediction
  accuracy on clean vs noisy GT rides.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    add_selection_args,
    load_experiment_index,
    resolve_experiments,
)
from src.prediction.algorithms.configTypes import (
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
)
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)

from .runner import (
    PipelineConfig,
    build_views,
    render_view_figures,
    run_all_experiments,
)


REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT_ROOT = REPO_ROOT / "elevator_reports" / "pipeline_eval"
DEFAULT_CALIBRATION = (
    REPO_ROOT / "src" / "data" / "structuredData" / "test_results"
    / "prediction" / "train" / "calibration_trapezoid.json"
)


def _summary_block(gt_df: pd.DataFrame, seg_df: pd.DataFrame,
                   accepted_only: bool) -> dict:
    pooled = build_views(gt_df, seg_df, accepted_only=accepted_only)
    train  = build_views(gt_df[gt_df["kind"] == "train"],
                         seg_df[seg_df["kind"] == "train"],
                         accepted_only=accepted_only)
    test   = build_views(gt_df[gt_df["kind"] == "test"],
                         seg_df[seg_df["kind"] == "test"],
                         accepted_only=accepted_only)
    return {
        "pooled": {k: v["summary"] for k, v in pooled.items()},
        "train":  {k: v["summary"] for k, v in train.items()},
        "test":   {k: v["summary"] for k, v in test.items()},
    }


def _accept_stats(seg_df: pd.DataFrame, gt_df: pd.DataFrame) -> dict:
    n_seg_total   = int(len(seg_df))
    n_seg_acc     = int((seg_df["pred_accepted"] == True).sum())  # noqa: E712
    n_clean_total = int((seg_df["status"] == "clean").sum())
    n_clean_acc   = int(((seg_df["status"] == "clean")
                         & (seg_df["pred_accepted"] == True)).sum())  # noqa: E712
    n_fp_total    = int((seg_df["status"] == "fp").sum())
    n_fp_acc      = int(((seg_df["status"] == "fp")
                         & (seg_df["pred_accepted"] == True)).sum())  # noqa: E712
    n_gt_total    = int(len(gt_df))
    n_gt_acc      = int((gt_df["oracle_accepted"] == True).sum())  # noqa: E712
    pct = lambda a, b: float(100.0 * a / b) if b else 0.0
    return {
        "accept_rate_all":    pct(n_seg_acc, n_seg_total),
        "accept_rate_clean":  pct(n_clean_acc, n_clean_total),
        "accept_rate_fp":     pct(n_fp_acc, n_fp_total),
        "accept_rate_gt":     pct(n_gt_acc, n_gt_total),
        "n_seg_total": n_seg_total, "n_seg_acc": n_seg_acc,
        "n_clean_total": n_clean_total, "n_clean_acc": n_clean_acc,
        "n_fp_total": n_fp_total, "n_fp_acc": n_fp_acc,
        "n_gt_total": n_gt_total, "n_gt_acc": n_gt_acc,
    }


def _fp_stats(seg_df: pd.DataFrame) -> dict:
    fp = seg_df[(seg_df["status"] == "fp") & seg_df["pred_dh"].notna()]
    if fp.empty:
        return {"n": 0, "median_signed": 0.0, "median_abs": 0.0,
                "mean_abs": 0.0}
    return {
        "n": int(len(fp)),
        "median_signed": float(np.median(fp["pred_dh"])),
        "median_abs":    float(np.median(fp["pred_dh"].abs())),
        "mean_abs":      float(np.mean(fp["pred_dh"].abs())),
    }


def _print_pooled(summary_block: dict, indent: str = "  ") -> None:
    """Console dump of the pooled / full-pass view summaries."""
    for view, s in summary_block["pooled"].items():
        print(f"{indent}{view:8s}: n={s['n']:4d}  MAE={s['mae']:.3f} m  "
              f"median={s['median']:.3f} m  rmse={s['rmse']:.3f} m  "
              f"<=1.5m={100*s['p_within_1_5m']:.1f}%")


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="evaluateOnData",
        description="Reproducible end-to-end pipeline evaluation: filter "
                    "experiments, run segmentation + prediction + barometer, "
                    "and render every pipeline figure into the run "
                    "directory.",
    )
    p.add_argument(
        "--seg-algorithm", default=SegmentAlgorithm.ACC_TEMPLATE_MATCH.value,
        choices=[a.value for a in SegmentAlgorithm],
        help="Segmentation algorithm (default acc_template_match).",
    )
    p.add_argument(
        "--pred-algorithm", default=PredictAlgorithm.TRAPEZOID_ACCEL.value,
        choices=[a.value for a in PredictAlgorithm],
        help="Prediction algorithm (default trapezoid_accel).",
    )
    add_selection_args(p)
    p.add_argument(
        "--calibration-path", type=Path, default=DEFAULT_CALIBRATION,
        help="Conformal-calibration JSON to load onto the predictor "
             "(default: the trapezoid checkpoint produced by "
             "src.prediction.evaluation.evaluateOnData on the train half).",
    )
    p.add_argument(
        "--out-root", type=Path, default=DEFAULT_OUT_ROOT,
        help="Base directory; output → <out-root>/run_<timestamp>/.",
    )
    p.add_argument(
        "--run-name", default=None,
        help="Override the timestamp folder name.",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)

    timestamp = _dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    run_dir = args.out_root / (args.run_name or f"run_{timestamp}")
    run_dir.mkdir(parents=True, exist_ok=True)
    print(f"writing run artefacts under {run_dir}")

    seg_cfg  = SEGMENT_ALGORITHM_CONFIG(
        algorithm=SegmentAlgorithm(args.seg_algorithm))
    pred_cfg = PREDICT_ALGORITHM_CONFIG(
        algorithm=PredictAlgorithm(args.pred_algorithm))
    calibration_path = (
        args.calibration_path
        if args.calibration_path and Path(args.calibration_path).exists()
        else None
    )
    config = PipelineConfig(
        seg_cfg=seg_cfg, pred_cfg=pred_cfg,
        calibration_path=calibration_path,
    )

    experiments = resolve_experiments(
        kind=args.kind, sources=args.source,
        include=args.include, exclude=args.exclude,
    )
    if not experiments:
        print("no experiments survived filtering; nothing to do",
              file=sys.stderr)
        return 1

    index = load_experiment_index()
    settings = {
        "timestamp": timestamp,
        "argv": sys.argv,
        "args": {k: (str(v) if isinstance(v, Path) else v)
                 for k, v in vars(args).items()},
        "configs": {
            "segmentation": {
                "algorithm": seg_cfg.algorithm.value,
                "config_path": str(seg_cfg.config_path),
                "overrides": seg_cfg.overrides,
                "active_params": seg_cfg.load_params(),
            },
            "prediction": {
                "algorithm": pred_cfg.algorithm.value,
                "config_path": str(pred_cfg.config_path),
                "overrides": pred_cfg.overrides,
                "active_params": pred_cfg.load_params(),
                "calibration_path":
                    str(calibration_path) if calibration_path else None,
            },
        },
        "experiments": {
            "n": len(experiments),
            "names": experiments,
            "n_train": sum(1 for e in experiments
                            if index.get(e, {}).get("experiment_type") == "train"),
            "n_test":  sum(1 for e in experiments
                            if index.get(e, {}).get("experiment_type") == "test"),
        },
    }
    (run_dir / "run_settings.json").write_text(
        json.dumps(settings, indent=2, default=str)
    )

    # --- run pipeline on the full resolved experiment set ---
    print(f"\nrunning pipeline on {len(experiments)} experiments")
    t0 = time.time()
    gt_df, seg_df = run_all_experiments(experiments, config, verbose=True)
    print(f"\npipeline finished in {time.time() - t0:.1f}s "
          f"({len(gt_df)} GT rows / {len(seg_df)} pred rows)")

    if gt_df.empty and seg_df.empty:
        print("no data after pipeline; nothing to render", file=sys.stderr)
        return 2

    # --- render figure sets + data into all/ clean/ noisy/ subdirs;
    # within each, the accepted-only pass carries the ``_acc`` suffix ---
    gt_acc  = gt_df[gt_df["oracle_accepted"] == True]    # noqa: E712
    seg_acc = seg_df[seg_df["pred_accepted"] == True]    # noqa: E712
    print("\nrendering pipeline figure sets — all / clean / noisy:")
    for nlabel, gt_n, seg_n in (
        ("all",   gt_df, seg_df),
        ("clean", gt_df[gt_df["signal_clear"] == True],     # noqa: E712
                  seg_df[seg_df["signal_clear"] == True]),  # noqa: E712
        ("noisy", gt_df[gt_df["signal_clear"] == False],    # noqa: E712
                  seg_df[seg_df["signal_clear"] == False]),  # noqa: E712
    ):
        sub_dir = run_dir / nlabel
        sub_dir.mkdir(parents=True, exist_ok=True)
        gt_n.to_csv(sub_dir / "gt_records.csv", index=False)
        seg_n.to_csv(sub_dir / "seg_records.csv", index=False)
        if gt_n.empty and seg_n.empty:
            print(f"  [skip] {nlabel}: empty subset")
            continue
        print(f"  figures [{nlabel}]: gt={len(gt_n)} pred={len(seg_n)}")
        render_view_figures(gt_n, seg_n, sub_dir, suffix="")
        render_view_figures(
            gt_n[gt_n["oracle_accepted"] == True],   # noqa: E712
            seg_n[seg_n["pred_accepted"] == True],   # noqa: E712
            sub_dir, suffix="_acc",
        )

    # --- metrics: overall + a clean/noisy breakdown by gt-side
    # signal_clear. Only GT-matched predictions carry a noise class;
    # FP predictions (signal_clear is None) stay in ``overall`` only. ---
    overall = {
        "full":          _summary_block(gt_df, seg_df, accepted_only=False),
        "accepted_only": _summary_block(gt_df, seg_df, accepted_only=True),
        "fp_stats":      _fp_stats(seg_df),
        "fp_stats_acc":  _fp_stats(seg_acc),
        "accept_stats":  _accept_stats(seg_df, gt_df),
    }

    by_noise: dict[str, dict] = {}
    for lbl, sc in (("clean", True), ("noisy", False)):
        gt_n  = gt_df[gt_df["signal_clear"] == sc]
        seg_n = seg_df[seg_df["signal_clear"] == sc]
        if gt_n.empty and seg_n.empty:
            by_noise[lbl] = {"n_gt": 0, "n_pred": 0, "note": "empty subset"}
            continue
        by_noise[lbl] = {
            "n_gt": int(len(gt_n)),
            "n_pred": int(len(seg_n)),
            "full":          _summary_block(gt_n, seg_n, accepted_only=False),
            "accepted_only": _summary_block(gt_n, seg_n, accepted_only=True),
        }

    metrics = {
        "kind": args.kind,
        "overall": overall,
        "by_noise": by_noise,
    }
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, default=str)
    )

    # --- console summary ---
    print("\noverall summary (pooled, full pass):")
    _print_pooled(overall["full"])
    fp = overall["fp_stats"]
    print(f"  fps: n={fp['n']}  |dh| median={fp['median_abs']:.2f} m "
          f"mean={fp['mean_abs']:.2f} m")

    print("\nby noise class (pooled, full pass):")
    for lbl in ("clean", "noisy"):
        bn = by_noise[lbl]
        if "note" in bn:
            print(f"  {lbl}: (no GT rides / predictions in this run)")
            continue
        print(f"  {lbl} (n_gt={bn['n_gt']}, n_pred={bn['n_pred']}):")
        _print_pooled(bn["full"], indent="    ")

    print(f"\nartefacts: {run_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
