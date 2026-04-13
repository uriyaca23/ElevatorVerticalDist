"""Compare pressure-GT segmentation with the ACC-only segmenter end-to-end.

Run with:
    python -m src.tests.segmentations.main_acc [experimenter]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
)
from src.plotting import plot_acc_with_gt


def build_acc_frame(acc_raw: pd.DataFrame, t0_ms: float) -> pd.DataFrame:
    t = (acc_raw["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    mag = np.sqrt(acc_raw["x"] ** 2 + acc_raw["y"] ** 2 + acc_raw["z"] ** 2).to_numpy()
    return pd.DataFrame({
        "time": t,
        "x": acc_raw["x"].to_numpy(),
        "y": acc_raw["y"].to_numpy(),
        "z": acc_raw["z"].to_numpy(),
        "mag": mag,
    })


def build_height_frame(prs: pd.DataFrame, t0_ms: float) -> pd.DataFrame:
    t = (prs["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    h = prs["GT_height_m"].to_numpy()
    h_smooth = pd.Series(h).rolling(window=51, center=True, min_periods=1).median().to_numpy()
    return pd.DataFrame({"time": t, "height": h_smooth})


def main(name: str = "oria") -> None:
    data = load_experimenter(name)
    acc_raw = data["ACC"]
    prs = data.get("PRS")

    t0_ms = float(acc_raw["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(acc_raw, t0_ms)
    height_frame = build_height_frame(prs, t0_ms) if prs is not None else None

    # --- pressure (GT) ---
    pressure_segmenter = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER))
    gt_segments = pressure_segmenter.detect(height_frame) if height_frame is not None else pd.DataFrame()
    print("Pressure GT segments:")
    print(gt_segments)

    # --- ACC-only ---
    acc_segmenter = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.ACC_ONLY))
    acc_frame_for_detect = acc_frame[["time", "x", "y", "z"]]
    acc_segments = acc_segmenter.detect(acc_frame_for_detect)
    print("ACC-only segments:")
    print(acc_segments)

    # --- plot ---
    fig, (ax_gt, ax_pred) = plt.subplots(2, 1, figsize=(14, 9), sharex=True)

    _plot_on(ax_gt, acc_frame, height_frame, gt_segments, f"Pressure GT — {name}", color="tab:green")
    _plot_on(ax_pred, acc_frame, height_frame, acc_segments, f"ACC-only — {name}", color="tab:red",
             show_probs=True)

    fig.tight_layout()
    plt.show()


def _plot_on(ax, acc_frame, height_frame, segments, title, color, show_probs=False):
    t = acc_frame["time"].to_numpy()
    ax.plot(t, acc_frame["mag"], color="black", linewidth=0.6, alpha=0.8, label="|acc|")
    ax.set_ylabel("|acc| (m/s²)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if height_frame is not None:
        ax_h = ax.twinx()
        ax_h.plot(height_frame["time"], height_frame["height"],
                  color="tab:blue", linewidth=1.5, label="height (m)")
        ax_h.set_ylabel("height (m)", color="tab:blue")

    if len(segments):
        for _, row in segments.iterrows():
            s_lo, s_hi = row["start_ci"]
            e_lo, e_hi = row["end_ci"]
            s_mid = 0.5 * (s_lo + s_hi)
            e_mid = 0.5 * (e_lo + e_hi)
            ax.axvspan(s_mid, e_mid, color=color, alpha=0.18)
            if s_hi > s_lo:
                ax.axvspan(s_lo, s_hi, color=color, alpha=0.08)
            if e_hi > e_lo:
                ax.axvspan(e_lo, e_hi, color=color, alpha=0.08)
            if show_probs:
                p_lo, p_hi = row["probability_ci"]
                p_mid = 0.5 * (p_lo + p_hi)
                txt = f"p={p_mid:.2f}\nCI=[{p_lo:.2f},{p_hi:.2f}]"
                half_s = 0.5 * (s_hi - s_lo)
                if half_s > 0:
                    txt += f"\n±{half_s:.1f}s"
                ax.text(s_mid, ax.get_ylim()[1] * 0.9, txt, fontsize=8, va="top")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oria")
