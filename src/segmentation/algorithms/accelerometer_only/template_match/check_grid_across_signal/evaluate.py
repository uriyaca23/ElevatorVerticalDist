"""Evaluate interval predictions against GT rides + sweep tunables.

Thin glue: loads data via :func:`src.data.loader.getExperimentData`,
runs the detector through :func:`detect.predict_intervals` with each
:class:`~detect.DetectConfig`, and scores via
:class:`~src.segmentation.algorithms.metrics.IntervalPredictionMetrics`.

Three CLI modes:

    # single run with default config across all train exps
    venv/bin/python -m src....check_grid_across_signal.evaluate

    # sweep — MIN/MAX ride length are pinned per user spec
    venv/bin/python -m src....check_grid_across_signal.evaluate --sweep \\
        [--min-ride-s 10] [--max-ride-s 120] [--top 20] \\
        [--out results.csv]

    # same, plus persist the winning config to JSON
    venv/bin/python -m src....check_grid_across_signal.evaluate --sweep \\
        --save-best elevator_reports/best_detect_config.json

To keep the sweep tractable, we cache the expensive ``(W, f)`` grid
sweep per experiment and re-run only the fast peak-pick + pair-filter
stages (which are the parts ``DetectConfig`` actually tunes) for each
hyperparameter combination.
"""
from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import list_experiments
from src.data.loader import getExperimentData
from src.segmentation.algorithms.metrics import IntervalPredictionMetrics

from . import detect, pair_filter
from .detect import DEFAULT_CONFIG, DetectConfig


# --------------------------------------------------------------------------
# Data plumbing
# --------------------------------------------------------------------------
def _extract_gt_rides(gt: pd.DataFrame, t0_ms: float) -> list[dict]:
    """GT rides with timestamps relative to the ACC session start."""
    rides: list[dict] = []
    if gt is None or gt.empty:
        return rides
    for _, row in gt.iterrows():
        if row.get("type") not in ("up", "down"):
            continue
        rides.append({
            "type": row["type"],
            "t_start_s": (float(row["start_ms"]) - t0_ms) / 1000.0,
            "t_end_s":   (float(row["end_ms"])   - t0_ms) / 1000.0,
        })
    return rides


def _load_experiments(names: list[str]) -> list[dict]:
    """Load each exp and cache the expensive (W, f) grid sweep once.

    Returns list of ``{name, state, gt_rides}``. ``state`` is the
    detection state dict from :func:`detect.detect` under
    :data:`DEFAULT_CONFIG` — we only keep the sweep arrays; the
    per-config peak-pick + pair-filter are redone per sweep step.
    """
    exps: list[dict] = []
    t0 = time.time()
    for n in names:
        try:
            sensors, gt, _ = getExperimentData(n)
        except Exception as exc:
            print(f"  [error] {n}: {type(exc).__name__}: {exc}")
            continue
        state = detect.detect(sensors.get("ACC"), DEFAULT_CONFIG)
        if state is None:
            print(f"  [skip]  {n}: empty ACC")
            continue
        gt_rides = _extract_gt_rides(gt, state["t0_ms"])
        exps.append({"name": n, "state": state, "gt_rides": gt_rides})
        print(f"  [ok]    {n}: gt_rides={len(gt_rides)} "
              f"({time.time() - t0:.1f}s)")
    return exps


# --------------------------------------------------------------------------
# Per-config evaluation
# --------------------------------------------------------------------------
def _rerun_peaks(state: dict, cfg: DetectConfig) -> dict:
    """Redo peak-pick + same-sign NMS on the cached sweep arrays under a
    new config. Returns a fresh state dict (same shape as
    :func:`detect.detect`'s output) that the pair filter accepts."""
    nms_samples = max(1, int(round(cfg.nms_radius_s * state["fs"])))
    amp_gate = np.abs(state["best_A"]) >= cfg.min_peak_abs_a
    best_r2_gated = np.where(amp_gate, state["best_r2"], -np.inf)
    initial_peaks = detect._peak_pick(
        best_r2_gated, cfg.r2_peak_thresh, nms_samples,
    )
    final_peaks = detect._same_sign_nms(
        initial_peaks, best_r2_gated, state["signs"],
        state["t"], cfg.same_sign_min_gap_s,
    )
    return {
        **state,
        "best_r2_gated": best_r2_gated,
        "initial_peaks": initial_peaks,
        "final_peaks": final_peaks,
        "config": cfg,
    }


def evaluate_config(
    exps: list[dict], cfg: DetectConfig,
) -> tuple[IntervalPredictionMetrics, list[tuple[str, IntervalPredictionMetrics]], dict[str, float]]:
    """Score ``cfg`` over all cached experiments.

    Returns ``(total, per_exp, iou_metrics)``. ``iou_metrics`` holds the
    classical F1 @ IoU ≥ 0.5 computed over *all* GT/pred pairs pooled
    across experiments — useful to compare the detector against external
    temporal-detection baselines.
    """
    per_exp: list[tuple[str, IntervalPredictionMetrics]] = []
    pooled_gt: list[dict] = []
    pooled_pred: list[dict] = []
    for e in exps:
        state = _rerun_peaks(e["state"], cfg)
        preds = pair_filter.predict_pairs(state, cfg)
        per_exp.append((
            e["name"],
            IntervalPredictionMetrics.from_intervals(e["gt_rides"], preds),
        ))
        # Offset each exp's time axis so pairs from different exps can
        # never accidentally match in the pooled IoU calculation.
        offset = (len(pooled_gt) + len(pooled_pred)) * 1e6 + 1e9
        pooled_gt.extend({**g, "t_start_s": g["t_start_s"] + offset,
                          "t_end_s":   g["t_end_s"]   + offset} for g in e["gt_rides"])
        pooled_pred.extend({"t_start_s": p["t_start_s"] + offset,
                            "t_end_s":   p["t_end_s"]   + offset} for p in preds)
    total = IntervalPredictionMetrics.sum(m for _, m in per_exp)
    iou_metrics = IntervalPredictionMetrics.iou_f1(pooled_gt, pooled_pred, iou_threshold=0.5)
    return total, per_exp, iou_metrics


# --------------------------------------------------------------------------
# Sweep grid — MIN/MAX ride length are pinned; the rest vary.
# --------------------------------------------------------------------------
# Focused grid — 2 × 4 × 1 × 3 × 2 × 3 = 144 combos. We saw a wall-
# clock cost of ~30 s per combo on 22 cached experiments (pair filter
# over the full (W, f) template grid dominates), so the sweep finishes
# in ~75 minutes. Values chosen to bracket the known-important knobs:
#  - `min_peak_abs_a`     — principal FP lever for pedestrian motion.
#  - `joint_r2_thresh`    — shape-agreement gate on pair side.
#  - `same_sign_min_gap_s`— too short and the filter keeps doublet lobes,
#                           too long and it kills back-to-back rides.
# Coarser on the other three; NMS radius is fixed.
DEFAULT_GRID: dict[str, list[float]] = {
    "r2_peak_thresh":      [0.75, 0.85],
    "min_peak_abs_a":      [0.3, 0.5, 0.7, 0.9],
    "nms_radius_s":        [0.5],
    "same_sign_min_gap_s": [5.0, 15.0, 25.0],
    "joint_r2_thresh":     [0.75, 0.85],
    "min_pair_abs_a":      [0.3, 0.5, 0.7],
}


def _iter_configs(
    grid: dict[str, list[float]],
    min_ride_s: float, max_ride_s: float,
) -> list[DetectConfig]:
    keys = list(grid.keys())
    out: list[DetectConfig] = []
    for combo in itertools.product(*(grid[k] for k in keys)):
        out.append(replace(
            DEFAULT_CONFIG,
            **dict(zip(keys, combo)),
            min_ride_s=min_ride_s, max_ride_s=max_ride_s,
        ))
    return out


def run_sweep(
    exps: list[dict], grid: dict[str, list[float]],
    min_ride_s: float, max_ride_s: float,
) -> pd.DataFrame:
    configs = _iter_configs(grid, min_ride_s, max_ride_s)
    rows: list[dict] = []
    t0 = time.time()
    for i, cfg in enumerate(configs):
        total, _, iou = evaluate_config(exps, cfg)
        rows.append({**cfg.__dict__, **total.as_dict(), **iou})
        if (i + 1) % 10 == 0 or i == 0 or i == len(configs) - 1:
            r = total.rates()
            print(
                f"  [{i + 1:5d}/{len(configs)}] "
                f"f1={r['f1_like']:.3f}  iou_f1={iou['iou_f1@0.5']:.3f}  "
                f"clean={total.clean}  miss={total.missed}  "
                f"merge={total.pred_merged}  fp={total.fp}  "
                f"({time.time() - t0:.1f}s)", flush=True,
            )
    return (
        pd.DataFrame(rows)
        .sort_values(["f1_like", "iou_f1@0.5"], ascending=[False, False])
        .reset_index(drop=True)
    )


# --------------------------------------------------------------------------
# Pretty-printing
# --------------------------------------------------------------------------
def _print_per_exp(per_exp: list[tuple[str, IntervalPredictionMetrics]]) -> None:
    header = (
        f"{'exp':70s} {'gt':>3s} {'pred':>4s} {'clean':>5s} "
        f"{'miss':>4s} {'merge':>5s} {'split':>5s} {'fp':>3s}"
    )
    print(header)
    print("-" * len(header))
    for name, m in per_exp:
        short = name if len(name) <= 70 else name[:67] + "..."
        print(
            f"{short:70s} {m.n_gt:3d} {m.n_pred:4d} "
            f"{m.clean:5d} {m.missed:4d} "
            f"{m.pred_merged:5d} {m.gt_split:5d} {m.fp:3d}"
        )


def _print_totals(m: IntervalPredictionMetrics) -> None:
    r = m.rates()
    print(
        f"\nTOTALS: gt={m.n_gt}  pred={m.n_pred}  clean={m.clean}  "
        f"miss={m.missed}  merged_pred={m.pred_merged}  "
        f"split_gt={m.gt_split}  fp={m.fp}"
    )
    print(
        f"RATES : f1_like={r['f1_like']:.3f}  recall={r['recall']:.3f}  "
        f"precision={r['precision']:.3f}  miss={r['miss_rate']:.3f}  "
        f"merge={r['merge_rate']:.3f}  fp={r['fp_rate']:.3f}"
    )


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", help="evaluate a single experiment by name")
    parser.add_argument("--kind", default="train",
                        choices=("train", "test", "all"),
                        help="experiment split (default: train)")
    parser.add_argument("--sweep", action="store_true",
                        help="run the tunable sweep")
    parser.add_argument("--min-ride-s", type=float, default=10.0,
                        help="pinned min_ride_s (default 10)")
    parser.add_argument("--max-ride-s", type=float, default=120.0,
                        help="pinned max_ride_s (default 120)")
    parser.add_argument("--top", type=int, default=20,
                        help="print this many top configs at end of sweep")
    parser.add_argument("--out", type=Path,
                        help="optional CSV path for full sweep results")
    parser.add_argument("--save-best", type=Path,
                        help="write the winning DetectConfig + metrics as "
                             "JSON so downstream code can load it")
    args = parser.parse_args()

    names = [args.only] if args.only else list_experiments(kind=args.kind)
    if not names:
        print("no experiments found")
        return 1

    print(f"preparing {len(names)} experiment(s) (running full (W,f) sweep)…")
    exps = _load_experiments(names)
    if not exps:
        print("no usable experiments")
        return 1

    if not args.sweep:
        cfg = replace(
            DEFAULT_CONFIG,
            min_ride_s=args.min_ride_s, max_ride_s=args.max_ride_s,
        )
        print(f"\nrunning once with config = {cfg}")
        total, per_exp, iou = evaluate_config(exps, cfg)
        _print_per_exp(per_exp)
        _print_totals(total)
        print(f"IOU   : f1@0.5={iou['iou_f1@0.5']:.3f}  "
              f"p@0.5={iou['iou_precision@0.5']:.3f}  "
              f"r@0.5={iou['iou_recall@0.5']:.3f}  "
              f"mean_iou={iou['iou_mean@0.5']:.3f}")
        return 0

    print(f"\nsweeping (min_ride_s={args.min_ride_s}, "
          f"max_ride_s={args.max_ride_s})…")
    df = run_sweep(exps, DEFAULT_GRID, args.min_ride_s, args.max_ride_s)

    print(f"\nTop {args.top} configs (by f1_like):")
    cols = (
        list(DEFAULT_GRID.keys())
        + ["min_ride_s", "max_ride_s", "clean", "missed",
           "pred_merged", "gt_split", "fp", "f1_like", "recall", "precision"]
    )
    with pd.option_context(
        "display.max_columns", None, "display.width", 200,
        "display.float_format", "{:.3f}".format,
    ):
        print(df[cols].head(args.top).to_string(index=False))

    # Per-exp breakdown for the best config.
    best = df.iloc[0]
    best_cfg = DetectConfig(**{
        k: float(best[k]) for k in DetectConfig.__dataclass_fields__
    })
    print(f"\nBest config: {best_cfg}")
    print("Per-exp breakdown:")
    total, per_exp, iou = evaluate_config(exps, best_cfg)
    _print_per_exp(per_exp)
    _print_totals(total)
    print(f"IOU   : f1@0.5={iou['iou_f1@0.5']:.3f}  "
          f"p@0.5={iou['iou_precision@0.5']:.3f}  "
          f"r@0.5={iou['iou_recall@0.5']:.3f}  "
          f"mean_iou={iou['iou_mean@0.5']:.3f}")

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(args.out, index=False)
        print(f"\nwrote full sweep results → {args.out}")

    if args.save_best:
        args.save_best.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "config": asdict(best_cfg),
            "metrics": {**total.as_dict(), **iou},
            "grid": DEFAULT_GRID,
            "n_experiments": len(exps),
            "kind": args.kind,
        }
        args.save_best.write_text(json.dumps(payload, indent=2))
        print(f"wrote best config → {args.save_best}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
