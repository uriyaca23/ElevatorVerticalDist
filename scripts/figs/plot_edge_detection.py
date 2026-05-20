"""Generate ``paper_phd/figures/edge_detection.png``.

Three stacked panels show how the barometer interval detector — see
``src/segmentation/algorithms/barometer_only/height_segmentation.py`` —
turns an altitude trace into discrete ``up``/``down``/``outside``
intervals. The figure is referenced from Appendix~B (Dataset
construction) as ``fig:app-edge-detection``.

Panels (top to bottom):
    1. Low-pass altitude ``z_lp`` returned by
       :meth:`HeightSegmenter.filter_height`.
    2. Smoothed vertical velocity ``vz_smooth`` (the actual decision
       signal) with the ±0.15 m/s threshold and shaded "moving" regions.
    3. The surviving ``up`` (green) / ``down`` (red) / ``outside``
       (grey) intervals after the merge + duration + Δh filters.

Defaults zoom into a Millennium-Hotel session window that contains at
least two clear rides plus a quiet inter-ride gap so the threshold and
merge behaviour are both visible.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physics.barometric import P0_HPA, pressure_to_altitude
from src.segmentation.algorithms.barometer_only.height_segmentation import (
    HeightSegmenter,
)
from src.segmentation.algorithms.configTypes import PressureFilterConfig

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper_phd/figures"
DEFAULT_EXP = "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2"
# Zoom window (seconds since recording start) — picked so the panel
# contains two consecutive rides separated by a quiet pause.
DEFAULT_WIN = (500.0, 670.0)


def _altitude_frame(exp: str) -> pd.DataFrame:
    csv = REPO / "src/data/structuredData/data" / exp / "PRS.csv"
    df = pd.read_csv(csv)
    meta = REPO / "src/data/structuredData/data" / exp / "metadata.csv"
    t_c = float(pd.read_csv(meta).iloc[0]["temperature_c"]) if meta.exists() else None
    t = (df["timestamp_ms"].to_numpy() - df["timestamp_ms"].iloc[0]) / 1000.0
    h = pressure_to_altitude(
        df["pressure"].to_numpy(), p0_hpa=P0_HPA, temperature_c=t_c
    )
    return pd.DataFrame({"time": t, "height": h})


def main(exp: str = DEFAULT_EXP, window: tuple[float, float] = DEFAULT_WIN) -> None:
    cfg = PressureFilterConfig()  # defaults match the live config.json
    seg = HeightSegmenter(cfg)
    frame = _altitude_frame(exp)
    z_lp = seg.filter_height(frame)
    # Reproduce the velocity exactly as ``HeightSegmenter.segment`` does.
    t = frame["time"].to_numpy()
    dt = np.diff(t)
    fs = 1.0 / np.median(dt[dt > 0])
    vz = np.zeros_like(t)
    vz[1:] = np.diff(z_lp) / np.where(dt > 0, dt, np.nan)
    vz = np.nan_to_num(vz, nan=0.0)
    win = max(1, int(round(cfg.smooth_window_sec * fs)))
    vz_smooth = np.convolve(vz, np.ones(win) / win, mode="same")

    rides = seg.segment(frame)

    t0, t1 = window
    mask = (t >= t0) & (t <= t1)

    fig, axes = plt.subplots(3, 1, figsize=(9, 6.0), sharex=True)

    ax = axes[0]
    ax.plot(t[mask], z_lp[mask], color="#2ca02c", linewidth=1.0)
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        "Edge-point detection on barometric altitude "
        f"(low-pass {cfg.height_lowpass_sec:.0f} s, threshold "
        f"{cfg.velocity_threshold:.2f} m/s)",
        fontsize=10,
    )
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    ax = axes[1]
    ax.plot(t[mask], vz_smooth[mask], color="#1f77b4", linewidth=1.0)
    ax.axhline(cfg.velocity_threshold, color="#7f7f7f", linestyle="--", linewidth=0.8)
    ax.axhline(-cfg.velocity_threshold, color="#7f7f7f", linestyle="--", linewidth=0.8)
    is_moving = np.abs(vz_smooth) > cfg.velocity_threshold
    ax.fill_between(
        t,
        vz_smooth,
        0,
        where=is_moving,
        color="#1f77b4",
        alpha=0.18,
        linewidth=0,
        interpolate=True,
    )
    ax.set_ylabel("Vertical velocity (m/s)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_xlim(t0, t1)

    ax = axes[2]
    ax.axhline(0, color="black", linewidth=0.6)
    colours = {"up": "#2ca02c", "down": "#d62728"}
    drew_labels: set[str] = set()
    # Background: outside (grey) covers the whole window first.
    ax.axhspan(-1, 1, color="#b8b8b8", alpha=0.18, label="outside")
    for _, row in rides.iterrows():
        s, _ = row["start_ci"]
        _, e = row["end_ci"]
        if e < t0 or s > t1:
            continue
        s_c = max(s, t0)
        e_c = min(e, t1)
        label = row["type"] if row["type"] not in drew_labels else None
        drew_labels.add(row["type"])
        ax.axvspan(s_c, e_c, color=colours[row["type"]], alpha=0.55, label=label)
    ax.set_yticks([])
    ax.set_ylim(-1, 1)
    ax.set_xlabel("Time since recording start (s)")
    ax.set_xlim(t0, t1)
    ax.legend(loc="upper right", fontsize=8, frameon=True, framealpha=0.9)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "edge_detection.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    n_rides_in_win = sum(
        1
        for _, r in rides.iterrows()
        if not (r["end_ci"][1] < t0 or r["start_ci"][0] > t1)
    )
    print(
        f"wrote {out_path.relative_to(REPO)} "
        f"(exp={exp}, window={window}, rides_in_window={n_rides_in_win}/{len(rides)})"
    )


if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        main(sys.argv[1])
    else:
        main()
