"""Cross-reconstruction comparison report (segmentation + prediction).

Runs both stages across ``{none} + the five reconstruct_az filters`` and
emits the LaTeX tables and figures consumed by the
"Vertical-acceleration reconstruction" section of
``docs/latex/algorithm_report.tex``. It answers, on the real elevator data:
*how much does reconstructing a_z improve segmentation and Δh prediction, and
which filter is best?*

For each variant it sets ``config.reconstruct=<variant>`` on the two
dispatchers (so the reconstruction runs as the first step of ``Predictor.predict``
/ ``Segmenter.detect``) and re-scores. Prediction records are built once
(they carry the gyroscope); only inference is repeated per variant.

Outputs (into ``--out``, default ``docs/latex/figures/reconstruct_az/``):

* ``pred_comparison.tex`` / ``seg_comparison.tex`` — booktabs tables.
* ``pred_cdf_<algo>.png`` — |Δh error| CDF overlay across variants.
* ``pred_mae_bar.png`` / ``seg_f1_bar.png`` — headline bars.
* ``before_after_avert.png`` — raw vs reconstructed vertical accel on the
  most-rotating ride.
* ``diagnostics_rotation.png`` — per-ride gyro energy vs MAE improvement.
* ``comparison_metrics.json`` — every number, machine-readable.

Usage::

    venv/bin/python scripts/reconstruction_comparison_report.py --kind all
    venv/bin/python scripts/reconstruction_comparison_report.py \\
        --include RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.data.loader import add_selection_args, resolve_experiments
from src.physics.reconstruct_az import RECONSTRUCTORS, SensorChannel, build
from src.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
    Predictor,
)
from src.prediction.evaluation.dataset import build_segment_records
from src.prediction.evaluation.figures import fig_compare_algorithms
from src.prediction.evaluation.metrics import compute_metrics
from src.prediction.evaluation.runner import collect_predictions, run_predictions
from src.segmentation.algorithms.configTypes import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
)
from src.segmentation.evaluate import evaluator
from src.segmentation.evaluate.evaluateOnData import _aggregate_filtered
from src.utils.accelerometer_utils import compute_a_vert

REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUT = REPO_ROOT / "docs" / "latex" / "figures" / "reconstruct_az"

VARIANTS: list[str] = ["none", *RECONSTRUCTORS]
PRED_ALGOS: dict[str, PredictAlgorithm] = {
    "zupt": PredictAlgorithm.ZUPT_ACCEL,
    "trapezoid": PredictAlgorithm.TRAPEZOID_ACCEL,
}


# --------------------------------------------------------------------------
# Runs
# --------------------------------------------------------------------------
def _load_records(experiments: list[str]) -> list:
    records = []
    for name in experiments:
        records.extend(build_segment_records(name))
    return records


def run_prediction(records: list, variants: list[str]):
    """Return ``{(variant, algo): predictions_df}`` and ``{(variant, algo): MetricsBundle}``."""
    dfs: dict[tuple[str, str], pd.DataFrame] = {}
    metrics: dict[tuple[str, str], object] = {}
    for v in variants:
        for algo_name, algo_enum in PRED_ALGOS.items():
            p = Predictor(PREDICT_ALGORITHM_CONFIG(
                algorithm=algo_enum, reconstruct=v))
            df = collect_predictions(run_predictions(p, records))
            dfs[(v, algo_name)] = df
            metrics[(v, algo_name)] = compute_metrics(df)
            print(f"  pred  {v:14s} {algo_name:10s} "
                  f"MAE={metrics[(v, algo_name)].clean_mae:.3f}")
    return dfs, metrics


def run_segmentation(experiments: list[str], variants: list[str]):
    """Return ``{variant: (IntervalPredictionMetrics total, iou_dict)}``."""
    out: dict[str, tuple] = {}
    for v in variants:
        cfg = SEGMENT_ALGORITHM_CONFIG(
            algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH, reconstruct=v)
        raw = evaluator._run_on_experiments(cfg, experiments)
        _per_exp, total, iou, _pairs = _aggregate_filtered(raw, set())
        out[v] = (total, iou)
        print(f"  seg   {v:14s} F1*={total.score():.3f} "
              f"IoU-F1={iou.get('iou_f1@0.5', 0):.3f} fp={total.fp}")
    return out


# --------------------------------------------------------------------------
# LaTeX tables
# --------------------------------------------------------------------------
def _f(x: float, nd: int = 3) -> str:
    return f"{x:.{nd}f}" if x is not None and np.isfinite(x) else "--"


def _row(cells: list[str], bold: bool = False) -> str:
    if bold:
        cells = [f"\\textbf{{{c}}}" for c in cells]
    return " & ".join(cells) + r" \\"


def write_pred_table(metrics: dict, variants: list[str], out_path: Path) -> None:
    """One booktabs block per prediction algorithm; best MAE row bolded."""
    parts: list[str] = []
    for algo_name in PRED_ALGOS:
        rows = [(v, metrics[(v, algo_name)]) for v in variants]
        best_v = min(rows, key=lambda r: r[1].clean_mae
                     if np.isfinite(r[1].clean_mae) else 1e9)[0]
        parts.append(r"\begin{tabular}{lrrrrr}")
        parts.append(r"\toprule")
        parts.append(r"\multicolumn{6}{l}{\textbf{%s}} \\" % algo_name.upper())
        parts.append(r"\midrule")
        parts.append(_row(["Reconstruction", "MAE (m)", "RMSE (m)",
                           "Cov@90", "$\\le$1.5\\,m", "med CI (m)"]))
        parts.append(r"\midrule")
        for v, m in rows:
            parts.append(_row([
                v, _f(m.clean_mae), _f(m.clean_rmse),
                _f(m.clean_coverage_90), _f(m.clean_frac_within_1_5m),
                _f(m.clean_median_ci),
            ], bold=(v == best_v)))
        parts.append(r"\bottomrule")
        parts.append(r"\end{tabular}")
        parts.append(r"\vspace{1em}")
        parts.append("")
    out_path.write_text("\n".join(parts))


def write_seg_table(seg: dict, variants: list[str], out_path: Path) -> None:
    """Booktabs segmentation table; best IoU-F1 row bolded."""
    best_v = max(variants, key=lambda v: seg[v][1].get("iou_f1@0.5", 0.0))
    parts = [
        r"\begin{tabular}{lrrrrrrrr}",
        r"\toprule",
        _row(["Reconstruction", "$n_{gt}$", "detect", "miss", "FP",
              "F1*", "IoU-F1", "recall", "prec"]),
        r"\midrule",
    ]
    for v in variants:
        total, iou = seg[v]
        r = total.rates()
        parts.append(_row([
            v, str(total.n_gt), str(total.clean), str(total.missed),
            str(total.fp), _f(total.score()), _f(iou.get("iou_f1@0.5", 0.0)),
            _f(r["recall"]), _f(r["precision"]),
        ], bold=(v == best_v)))
    parts += [r"\bottomrule", r"\end{tabular}", ""]
    out_path.write_text("\n".join(parts))


# --------------------------------------------------------------------------
# Figures
# --------------------------------------------------------------------------
def _ride_gyro_energy(rec) -> float:
    """Mean |ω| (rad/s) over the ride window, or nan when no gyro."""
    g = rec.gyr
    if g is None or g.empty:
        return float("nan")
    w = g[["x", "y", "z"]].to_numpy(dtype=float)
    return float(np.mean(np.linalg.norm(w, axis=1)))


def figure_pred_cdfs(dfs: dict, variants: list[str], out_dir: Path) -> None:
    for algo_name in PRED_ALGOS:
        fig_compare_algorithms(
            {v: dfs[(v, algo_name)] for v in variants},
            out_dir / f"pred_cdf_{algo_name}.png",
            title=f"|Δh error| CDF — {algo_name} (reconstruction variants)",
        )


def figure_mae_bar(metrics: dict, variants: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(variants))
    w = 0.38
    for i, algo_name in enumerate(PRED_ALGOS):
        vals = [metrics[(v, algo_name)].clean_mae for v in variants]
        bars = ax.bar(x + (i - 0.5) * w, vals, w, label=algo_name)
        ax.bar_label(bars, fmt="%.2f", fontsize=7, padding=2)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("MAE (m)"); ax.set_title("Prediction MAE by reconstruction")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def figure_f1_bar(seg: dict, variants: list[str], out_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4.2))
    x = np.arange(len(variants))
    f1 = [seg[v][0].score() for v in variants]
    iouf1 = [seg[v][1].get("iou_f1@0.5", 0.0) for v in variants]
    b1 = ax.bar(x - 0.2, f1, 0.4, label="F1*")
    b2 = ax.bar(x + 0.2, iouf1, 0.4, label="IoU-F1@0.5")
    ax.bar_label(b1, fmt="%.2f", fontsize=7); ax.bar_label(b2, fmt="%.2f", fontsize=7)
    ax.set_xticks(x); ax.set_xticklabels(variants, rotation=20, ha="right")
    ax.set_ylabel("score"); ax.set_ylim(0, 1)
    ax.set_title("Segmentation F1 by reconstruction")
    ax.legend(); ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def figure_before_after(records: list, best: str, out_path: Path) -> None:
    """Raw fixed-gravity a_vert vs reconstructed a_z on the most-rotating ride."""
    with_gyro = [(r, _ride_gyro_energy(r)) for r in records]
    with_gyro = [(r, e) for r, e in with_gyro if np.isfinite(e) and len(r.acc) > 10]
    if not with_gyro:
        return
    rec, energy = max(with_gyro, key=lambda t: t[1])
    acc = rec.acc
    ax_, ay_, az_ = (acc["x"].to_numpy(float), acc["y"].to_numpy(float),
                     acc["z"].to_numpy(float))
    ts = acc["timestamp_ms"].to_numpy(float)
    fs = 1000.0 / float(np.median(np.diff(ts))) if len(ts) > 1 else 50.0
    t = (ts - ts[0]) / 1000.0

    raw_avert = compute_a_vert(ax_, ay_, az_, fs)
    r = build(best).reconstruct(
        {SensorChannel.ACC: acc, SensorChannel.GYR: rec.gyr})

    fig, ax = plt.subplots(figsize=(9, 4.2))
    ax.plot(t, raw_avert, color="tab:gray", lw=1.2,
            label="raw (fixed gravity)")
    ax.plot(t, r.az, color="tab:purple", lw=1.7,
            label=f"reconstructed ({best})")
    ax.axhline(0, color="k", lw=0.6, alpha=0.5)
    ax.set_xlabel("time (s)"); ax.set_ylabel(r"$a_\mathrm{vert}$ (m/s$^2$)")
    ax.set_title(f"{rec.exp_name} seg {rec.seg_idx} · mean|ω|={energy:.2f} rad/s")
    ax.legend(); ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


def figure_diagnostics(records, dfs, best: str, out_path: Path) -> None:
    """Per-ride rotation energy vs trapezoid |error| improvement (none→best)."""
    energy = {(r.exp_name, r.seg_idx): _ride_gyro_energy(r) for r in records}
    d_none = dfs[("none", "trapezoid")].set_index(["exp_name", "seg_idx"])
    d_best = dfs[(best, "trapezoid")].set_index(["exp_name", "seg_idx"])
    xs, ys = [], []
    for key in d_none.index.intersection(d_best.index):
        e = energy.get(tuple(key))
        if e is None or not np.isfinite(e):
            continue
        xs.append(e)
        ys.append(float(d_none.loc[key, "abs_error"]) -
                  float(d_best.loc[key, "abs_error"]))
    if not xs:
        return
    fig, ax = plt.subplots(figsize=(7, 4.4))
    ax.scatter(xs, ys, s=22, alpha=0.7, color="tab:purple")
    ax.axhline(0, color="k", lw=0.7)
    ax.set_xlabel("ride rotation energy — mean |ω| (rad/s)")
    ax.set_ylabel(f"|error| reduction, none→{best} (m)")
    ax.set_title("Reconstruction helps most on rotating rides (trapezoid)")
    ax.grid(True, alpha=0.3)
    fig.tight_layout(); fig.savefig(out_path, dpi=110); plt.close(fig)


# --------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------
def _parse_args(argv=None) -> argparse.Namespace:
    p = argparse.ArgumentParser(prog="reconstruction_comparison_report")
    add_selection_args(p)
    p.add_argument("--variants", nargs="*", default=VARIANTS,
                   help="Subset of variants (default: none + all filters).")
    p.add_argument("--out", type=Path, default=DEFAULT_OUT,
                   help="Output directory for tables/figures.")
    return p.parse_args(argv)


def main(argv=None) -> int:
    args = _parse_args(argv)
    variants = [v for v in args.variants if v == "none" or v in RECONSTRUCTORS]
    args.out.mkdir(parents=True, exist_ok=True)

    experiments = resolve_experiments(
        kind=args.kind, sources=args.source,
        include=args.include, exclude=args.exclude)
    if not experiments:
        print("no experiments after filtering", file=sys.stderr)
        return 1
    print(f"{len(experiments)} experiments · variants={variants}")

    print("\nloading prediction records ...")
    records = _load_records(experiments)
    print(f"  {len(records)} segments")

    print("\nprediction:")
    dfs, pred_metrics = run_prediction(records, variants)
    print("\nsegmentation:")
    seg = run_segmentation(experiments, variants)

    # Best non-none variant by trapezoid MAE (drives the diagnostic figures).
    non_none = [v for v in variants if v != "none"] or variants
    best = min(non_none, key=lambda v: pred_metrics[(v, "trapezoid")].clean_mae)
    print(f"\nbest (trapezoid MAE): {best}")

    # Tables
    write_pred_table(pred_metrics, variants, args.out / "pred_comparison.tex")
    write_seg_table(seg, variants, args.out / "seg_comparison.tex")

    # Figures
    figure_pred_cdfs(dfs, variants, args.out)
    figure_mae_bar(pred_metrics, variants, args.out / "pred_mae_bar.png")
    figure_f1_bar(seg, variants, args.out / "seg_f1_bar.png")
    figure_before_after(records, best, args.out / "before_after_avert.png")
    figure_diagnostics(records, dfs, best, args.out / "diagnostics_rotation.png")

    # Machine-readable dump
    payload = {
        "experiments": experiments,
        "variants": variants,
        "best_variant": best,
        "prediction": {f"{v}/{a}": asdict(pred_metrics[(v, a)])
                       for v in variants for a in PRED_ALGOS},
        "segmentation": {v: {"total": seg[v][0].as_dict(), "iou": seg[v][1]}
                         for v in variants},
    }
    (args.out / "comparison_metrics.json").write_text(
        json.dumps(payload, indent=2, default=str))
    print(f"\nartefacts → {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
