"""End-to-end pipeline test for the segmentation stack:

    load → segment → plot.

Run with:
    python -m src.tests.segmentations.main [experimenter_name]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

# allow running as a script
if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    Segmenter,
)
from src.plotting import plot_acc_with_gt, plot_acc_integrals_with_gt


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


def build_integrals_frame(acc_frame: pd.DataFrame) -> pd.DataFrame:
    t = acc_frame["time"].to_numpy()
    a_lin = acc_frame["mag"].to_numpy() - acc_frame["mag"].mean()
    dt = np.diff(t, prepend=t[0])
    vel = np.cumsum(a_lin * dt)
    pos = np.cumsum(vel * dt)
    return pd.DataFrame({"time": t, "a_lin": a_lin, "vel": vel, "pos": pos})


def main(name: str = "oria") -> None:
    data = load_experimenter(name)
    acc_raw = data["ACC"]
    prs = data.get("PRS")

    t0_ms = float(acc_raw["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(acc_raw, t0_ms)

    height_frame = build_height_frame(prs, t0_ms) if prs is not None else None

    segments: pd.DataFrame | None = None
    if height_frame is not None:
        cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
        segments = Segmenter(cfg).detect(height_frame)
        print(f"Detected {len(segments)} segment(s):")
        print(segments)

    plot_acc_with_gt(
        acc_frame, height=height_frame, segments=segments,
        title=f"Accelerometer + GT height — {name}",
    )

    integrals = build_integrals_frame(acc_frame)
    plot_acc_integrals_with_gt(
        integrals, height=height_frame, segments=segments,
        title=f"|acc| and integrals vs GT — {name}",
    )


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "oria")
