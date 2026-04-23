"""Diagnostic: why does the trapezoid predictor struggle on 0-3 m rides?

Re-runs the current Predictor on every segment with ``|true_dh| <= 3`` and
writes one diagnostic PNG per ride plus a summary figure + CSV. The
per-ride plot overlays the fitted template on the smoothed vertical
accel and annotates the lobe geometry (W, Δt_c, the Δt_c/(2W) overlap
ratio, mode pair/joined, R²).

Note: the blind test split (BeitYitzchaki) has NO rides with |Δh| <= 3 m
(min |Δh| = 3.74 m), so this analysis runs against the *training* short
rides — the only population where the 1-floor behaviour is visible.

Outputs go under ``src/data/structuredData/test_results/prediction/
short_ride_analysis/``.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction.algorithms import (
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from src.prediction.evaluation.dataset import build_segment_records

OUT = _REPO_ROOT / "src/data/structuredData/test_results/prediction/short_ride_analysis"
CALIB = _REPO_ROOT / "src/data/structuredData/test_results/prediction/train/calibration_trapezoid.json"

# Experiments that contribute 0-3 m rides (from predictions_trapezoid_train.csv).
EXPS = [
    "RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026",
    "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026",
]


def _plot_one(rec, out, path: Path) -> None:
    meta = out.meta
    t = np.asarray(meta.get("t_sec", []))
    a_smooth = np.asarray(meta.get("a_smooth", []))
    a_template = np.asarray(meta.get("a_template", []))
    mode = meta.get("mode", "?")
    params = meta.get("params", {})
    W = float(params.get("W", np.nan))
    f = float(params.get("f", np.nan))
    tc1 = float(params.get("t_c1", np.nan))
    tc2 = float(params.get("t_c2", np.nan))
    A_fit = float(params.get("A_fit", np.nan))
    A_used = float(params.get("A_used", np.nan))
    r2 = float(params.get("joint_r2", np.nan))
    dtc = float(meta.get("delta_tc_sec", np.nan))
    overlap = dtc / (2.0 * W) if W > 0 else float("nan")

    fig, ax = plt.subplots(figsize=(9.0, 4.2))
    if t.size:
        ax.plot(t, a_smooth, color="#1f77b4", lw=1.1, label="smoothed vertical accel")
        if a_template.size == t.size:
            ax.plot(t, a_template, color="#d62728", lw=1.6,
                    label=f"trapezoid fit ({mode})")
    # Lobe centres and widths
    if np.isfinite(tc1) and np.isfinite(W):
        ax.axvspan(tc1 - W, tc1 + W, color="orange", alpha=0.12, label="lobe 1 (±W)")
        ax.axvline(tc1, color="orange", ls="--", lw=0.8)
    if np.isfinite(tc2) and np.isfinite(W) and tc2 != tc1:
        ax.axvspan(tc2 - W, tc2 + W, color="green", alpha=0.12, label="lobe 2 (±W)")
        ax.axvline(tc2, color="green", ls="--", lw=0.8)
    ax.axhline(0.0, color="k", lw=0.5, alpha=0.4)

    ride_tag = f"seg {rec.seg_idx}  ({rec.ride_type})"
    err = out.height_diff - rec.true_dh
    title = (
        f"{rec.exp_name}  —  {ride_tag}\n"
        f"true Δh = {rec.true_dh:+.2f} m   pred = {out.height_diff:+.2f} m   "
        f"err = {err:+.2f} m   dur = {rec.duration_sec:.2f} s\n"
        f"mode = {mode}   Δt_c = {dtc:.2f} s   W = {W:.2f} s   "
        f"Δt_c/(2W) = {overlap:.2f}   f = {f:.2f}   R² = {r2:.2f}   "
        f"A_fit = {A_fit:.2f}   A_used = {A_used:.2f}\n"
        f"accepted = {out.accepted}   reject = {out.reject_reason or '—'}"
    )
    ax.set_title(title, fontsize=8.5, loc="left")
    ax.set_xlabel("time (s, ride-local)")
    ax.set_ylabel("vertical accel (m/s²)")
    ax.legend(loc="best", fontsize=8, framealpha=0.85)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    per_exp_dir = OUT / "per_ride"
    per_exp_dir.mkdir(exist_ok=True)

    p = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=PredictAlgorithm.TRAPEZOID_ACCEL))
    if CALIB.exists():
        p.load_calibration(CALIB)
        print(f"[calib] loaded {CALIB}")
    else:
        print("[calib] WARNING: no calibration found, using default multiplier")

    rows = []
    for exp in EXPS:
        print(f"[load] {exp}")
        recs = build_segment_records(exp)
        short = [r for r in recs if abs(r.true_dh) <= 3.0]
        print(f"       {len(short)} short rides (|Δh| <= 3 m)")
        for rec in short:
            out = p.predict(rec.acc, phone_model=rec.phone,
                             pre=rec.pre_acc, post=rec.post_acc)
            meta = out.meta or {}
            params = meta.get("params", {}) if isinstance(meta, dict) else {}
            W = float(params.get("W", np.nan)) if params else np.nan
            dtc = float(meta.get("delta_tc_sec", np.nan)) if isinstance(meta, dict) else np.nan
            overlap = dtc / (2.0 * W) if W and W > 0 else np.nan
            rows.append({
                "exp_name": rec.exp_name,
                "seg_idx": rec.seg_idx,
                "ride_type": rec.ride_type,
                "duration_sec": rec.duration_sec,
                "true_dh": rec.true_dh,
                "pred_dh": out.height_diff,
                "abs_error": abs(out.height_diff - rec.true_dh),
                "ci_half_width": out.ci_half_width,
                "accepted": out.accepted,
                "reject_reason": out.reject_reason,
                "mode": meta.get("mode", ""),
                "W": W,
                "f": float(params.get("f", np.nan)) if params else np.nan,
                "delta_tc_sec": dtc,
                "overlap_ratio": overlap,
                "joint_r2": float(params.get("joint_r2", np.nan)) if params else np.nan,
                "A_fit": float(params.get("A_fit", np.nan)) if params else np.nan,
                "A_used": float(params.get("A_used", np.nan)) if params else np.nan,
                "pair_r2": float(meta.get("pair_r2", np.nan)) if isinstance(meta, dict) else np.nan,
                "joined_r2": float(meta.get("joined_r2", np.nan)) if isinstance(meta, dict) else np.nan,
            })
            fname = f"{rec.exp_name.split('_')[1]}_seg{rec.seg_idx:03d}_{rec.ride_type}.png"
            _plot_one(rec, out, per_exp_dir / fname)

    summary = pd.DataFrame(rows)
    summary_path = OUT / "summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"[write] summary → {summary_path}")

    # --- Aggregate figure 1: abs_error vs overlap ratio, colored by mode ---
    fig, ax = plt.subplots(figsize=(8.0, 5.0))
    for mode, grp in summary.groupby("mode"):
        ax.scatter(grp["overlap_ratio"], grp["abs_error"], s=40, alpha=0.8,
                   label=f"{mode} (n={len(grp)})")
    ax.axvline(1.0, color="k", ls="--", lw=0.8, alpha=0.5,
               label="Δt_c = 2W (pulses touch)")
    ax.set_xlabel("Δt_c / (2W)  —  < 1 ⇒ lobes overlap (joint pulse)")
    ax.set_ylabel("absolute error |pred − true| (m)")
    ax.set_title("Short rides (|Δh| ≤ 3 m): error vs lobe overlap")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "overlap_vs_error.png", dpi=140)
    plt.close(fig)
    print(f"[write] scatter → {OUT / 'overlap_vs_error.png'}")

    # --- Aggregate figure 2: predicted vs true, mode-colored ---
    fig, ax = plt.subplots(figsize=(6.0, 6.0))
    ax.plot([-4, 4], [-4, 4], "k--", lw=0.7, alpha=0.5, label="y = x")
    for mode, grp in summary.groupby("mode"):
        ax.scatter(grp["true_dh"], grp["pred_dh"], s=40, alpha=0.8,
                   label=f"{mode} (n={len(grp)})")
    ax.set_xlabel("true Δh (m)")
    ax.set_ylabel("predicted Δh (m)")
    ax.set_title("Short rides: predicted vs true Δh")
    ax.legend(loc="best")
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal")
    fig.tight_layout()
    fig.savefig(OUT / "pred_vs_true.png", dpi=140)
    plt.close(fig)
    print(f"[write] pred-vs-true → {OUT / 'pred_vs_true.png'}")

    # --- Print summary stats ---
    n = len(summary)
    acc = summary["accepted"].sum()
    joined = (summary["mode"] == "joined").sum()
    pair = (summary["mode"] == "pair").sum()
    overlap_sub = summary["overlap_ratio"] < 1.0
    print(f"\n=== short-ride summary (|Δh| ≤ 3 m) ===")
    print(f"  total rides               : {n}")
    print(f"  accepted by quality filter: {acc}  ({acc/n:.0%})")
    print(f"  mode: joined={joined}  pair={pair}")
    print(f"  rides with Δt_c<2W (pulses overlap/joint): "
          f"{int(overlap_sub.sum())}  ({overlap_sub.mean():.0%})")
    print(f"  median |err|              : {summary['abs_error'].median():.2f} m")
    print(f"  mean   |err|              : {summary['abs_error'].mean():.2f} m")
    print(f"  reject_reason breakdown   :")
    print(summary["reject_reason"].fillna("ACCEPTED").value_counts().to_string())


if __name__ == "__main__":
    main()
