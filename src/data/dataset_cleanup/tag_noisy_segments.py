"""Flip ``signalClearRecording=False`` in gt.csv for the two experiments
Uriya flagged as having a "dirty second half" — Roy Turgeman's Haari
session and Uriya's BarIlan2 session. Per Uriya's guidance (2026-04-19):
use the experiment's time midpoint as the default split line, then
refine with an SNR check per elevator segment so we don't falsely mark
clean rides that happen to sit after the midpoint.

SNR metric per up/down segment
-------------------------------
On the accelerometer magnitude inside the segment:

* ``signal`` = RMS of a low-pass-filtered ``|a|`` in the elevator band
  (≤ 0.5 Hz, capturing the characteristic trapezoidal takeoff +
  landing lobes).
* ``noise``  = RMS of the residual (``|a|`` minus its low-pass version) —
  the high-frequency content that dominates pocket/walking recordings.
* ``snr_db`` = ``20·log10(signal / noise)``.

A clean elevator ride has SNR in the +10 dB range (big trapezoid lobes
dominate the noise); a segment recorded with the phone bouncing in a
pocket sits around 0 dB or lower.

Rule
----
A segment is marked ``signalClearRecording=False`` when **both** are true:

* its ``start_ms`` lies past the midpoint of the experiment's gt timeline;
* its SNR falls below ``SNR_THRESHOLD_DB`` (default: 3 dB — conservative
  so we only flip clearly-noisy rides, leaving marginal ones flagged in
  the plot for manual review).

Outputs per experiment (under ``structuredData/test_results/noisy_segments/``):

* ``<exp>__noisy_review.png`` — altitude + |a| + SNR bar per segment + a
  dashed vertical line at the midpoint. Every segment that gets flipped
  to noisy is shaded red.
* ``<exp>__noisy_decisions.csv`` — start_ms / end_ms / type / snr_db /
  post_midpoint / flipped, so you can see exactly what changed.

And the gt.csv itself gets its ``signalClearRecording`` column updated
and written back in place.

Run with::

    python -m src.data.dataset_cleanup.tag_noisy_segments
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader.constants import (
    GT_COLUMNS,
    GT_CSV,
    STRUCTURED_DATA_DIR,
    STRUCTURED_ROOT,
)

TARGETS: list[str] = [
    "RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026",
    "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026",
]

SNR_THRESHOLD_DB = 3.0
ELEVATOR_BAND_HZ = 0.5  # anything slower than this is "signal"; faster is "noise"

REVIEW_DIR = STRUCTURED_ROOT / "test_results" / "noisy_segments"


def _butter_lowpass(x: np.ndarray, fs: float, cutoff_hz: float) -> np.ndarray:
    from scipy.signal import butter, sosfiltfilt
    if x.size < 8 or fs <= 2 * cutoff_hz:
        return np.zeros_like(x)
    nyq = 0.5 * fs
    sos = butter(2, cutoff_hz / nyq, btype="low", output="sos")
    return sosfiltfilt(sos, x)


def _segment_snr_db(acc: pd.DataFrame, s_ms: int, e_ms: int) -> float:
    ts = acc["timestamp_ms"].to_numpy(dtype=np.int64)
    mask = (ts >= s_ms) & (ts < e_ms)
    if mask.sum() < 32:
        return float("nan")
    t = (ts[mask] - ts[mask][0]) / 1000.0
    fs = 1.0 / float(np.median(np.diff(t)))
    mag = np.sqrt(
        acc.loc[mask, "x"].to_numpy(dtype=float) ** 2
        + acc.loc[mask, "y"].to_numpy(dtype=float) ** 2
        + acc.loc[mask, "z"].to_numpy(dtype=float) ** 2
    )
    mag = mag - float(np.mean(mag))
    lp = _butter_lowpass(mag, fs, ELEVATOR_BAND_HZ)
    residual = mag - lp
    sig = float(np.sqrt(np.mean(lp ** 2)))
    nse = float(np.sqrt(np.mean(residual ** 2)))
    if nse < 1e-6:
        return float("inf")
    return 20.0 * float(np.log10(sig / nse))


def _process_one(exp_name: str) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    gt_path = exp_dir / GT_CSV
    if not gt_path.exists():
        return {"exp_name": exp_name, "status": "no gt.csv"}
    gt = pd.read_csv(gt_path)
    acc = pd.read_csv(exp_dir / "ACC.csv")

    if not len(gt):
        return {"exp_name": exp_name, "status": "empty gt"}
    t0 = int(gt["start_ms"].min())
    t1 = int(gt["end_ms"].max())
    midpoint = (t0 + t1) // 2

    decisions: list[dict] = []
    for i, row in gt.iterrows():
        if str(row["type"]).lower() not in ("up", "down"):
            decisions.append({
                "segment_idx":    int(i),
                "type":           row["type"],
                "start_ms":       int(row["start_ms"]),
                "end_ms":         int(row["end_ms"]),
                "snr_db":         float("nan"),
                "post_midpoint":  bool(int(row["start_ms"]) > midpoint),
                "flipped":        False,
            })
            continue
        snr = _segment_snr_db(acc, int(row["start_ms"]), int(row["end_ms"]))
        post = int(row["start_ms"]) > midpoint
        flip = post and (not np.isnan(snr)) and snr < SNR_THRESHOLD_DB
        decisions.append({
            "segment_idx":   int(i),
            "type":          row["type"],
            "start_ms":      int(row["start_ms"]),
            "end_ms":        int(row["end_ms"]),
            "snr_db":        round(snr, 2) if not np.isnan(snr) else float("nan"),
            "post_midpoint": post,
            "flipped":       bool(flip),
        })

    dec_df = pd.DataFrame(decisions)
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    dec_df.to_csv(REVIEW_DIR / f"{exp_name}__noisy_decisions.csv", index=False)

    # Apply flips into gt.csv.
    for d in decisions:
        if d["flipped"]:
            gt.at[d["segment_idx"], "signalClearRecording"] = False
    out_cols = list(GT_COLUMNS)
    if "exp_name" in gt.columns:
        out_cols.append("exp_name")
    gt[out_cols].to_csv(gt_path, index=False)

    _plot_noisy_review(exp_name, exp_dir, gt, acc, decisions, midpoint, t0, t1)

    flipped = sum(1 for d in decisions if d["flipped"])
    flagged_for_review = sum(
        1 for d in decisions
        if d["post_midpoint"] and d["type"] in ("up", "down") and not d["flipped"]
    )
    return {
        "exp_name":               exp_name,
        "n_segments":             len(gt),
        "n_up_down":              sum(1 for d in decisions if d["type"] in ("up", "down")),
        "n_flipped":              flipped,
        "n_post_mid_but_clean":   flagged_for_review,
        "midpoint_ms":            midpoint,
    }


def _plot_noisy_review(
    exp_name: str, exp_dir: Path, gt: pd.DataFrame, acc: pd.DataFrame,
    decisions: list[dict], midpoint: int, t0: int, t1: int,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    prs_path = exp_dir / "PRS.csv"
    if prs_path.exists():
        prs = pd.read_csv(prs_path)
        p_t = prs["timestamp_ms"].to_numpy(dtype=np.int64)
        p_h = prs["GT_height_m"].to_numpy(dtype=float) if "GT_height_m" in prs.columns else np.zeros_like(p_t)
    else:
        p_t = np.array([t0, t1]); p_h = np.zeros(2)

    a_t = acc["timestamp_ms"].to_numpy(dtype=np.int64)
    a_mag = np.sqrt(
        acc["x"].to_numpy(dtype=float) ** 2
        + acc["y"].to_numpy(dtype=float) ** 2
        + acc["z"].to_numpy(dtype=float) ** 2
    )

    fig, axes = plt.subplots(3, 1, figsize=(14, 8), sharex=True)
    fig.suptitle(f"Noisy-segment review — {exp_name}", fontsize=11)

    # Panel 1: altitude with segment shading
    axes[0].plot((p_t - t0) / 1000.0, p_h, color="tab:blue", lw=0.9)
    axes[0].set_ylabel("altitude (m)")
    axes[0].grid(True, alpha=0.3)
    axes[0].axvline((midpoint - t0) / 1000.0, color="black", ls="--", lw=1,
                    label="midpoint")

    # Panel 2: |a| with segment shading
    axes[1].plot((a_t - t0) / 1000.0, a_mag, color="tab:gray", lw=0.5, alpha=0.7)
    axes[1].set_ylabel("|a| (m/s²)")
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline((midpoint - t0) / 1000.0, color="black", ls="--", lw=1)

    # Shade flipped segments
    for d in decisions:
        if d["type"] not in ("up", "down"):
            continue
        s_rel = (d["start_ms"] - t0) / 1000.0
        e_rel = (d["end_ms"] - t0) / 1000.0
        color = "tab:red" if d["flipped"] else ("tab:orange" if d["post_midpoint"] else "tab:green")
        alpha = 0.22 if d["flipped"] else 0.10
        axes[0].axvspan(s_rel, e_rel, color=color, alpha=alpha)
        axes[1].axvspan(s_rel, e_rel, color=color, alpha=alpha)

    # Panel 3: per-segment SNR as bars at segment mid-times
    snr_t = [(d["start_ms"] + d["end_ms"]) / 2 / 1000 - t0 / 1000
             for d in decisions if d["type"] in ("up", "down")]
    snr_v = [d["snr_db"] for d in decisions if d["type"] in ("up", "down")]
    snr_f = [d["flipped"] for d in decisions if d["type"] in ("up", "down")]
    colors = ["tab:red" if f else "tab:green" for f in snr_f]
    axes[2].bar(snr_t, snr_v, width=15.0, color=colors, alpha=0.8)
    axes[2].axhline(SNR_THRESHOLD_DB, color="black", ls="--", lw=1,
                    label=f"SNR threshold = {SNR_THRESHOLD_DB} dB")
    axes[2].axvline((midpoint - t0) / 1000.0, color="black", ls="--", lw=1)
    axes[2].set_ylabel("SNR (dB)")
    axes[2].set_xlabel(f"time (s, from experiment start  t0={t0})")
    axes[2].grid(True, alpha=0.3)
    axes[2].legend(loc="upper right", fontsize=8)

    axes[0].legend(loc="upper right", fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(REVIEW_DIR / f"{exp_name}__noisy_review.png", dpi=120)
    plt.close(fig)


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    results: list[dict] = []
    for name in TARGETS:
        results.append(_process_one(name))

    summary_path = REVIEW_DIR / "summary.csv"
    REVIEW_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(results).to_csv(summary_path, index=False)
    print(f"[noisy-tag] SNR threshold = {SNR_THRESHOLD_DB} dB")
    print(f"[noisy-tag] Plots + per-exp decisions: {REVIEW_DIR}")
    print()
    print(pd.DataFrame(results).to_string(index=False))


if __name__ == "__main__":
    main()
