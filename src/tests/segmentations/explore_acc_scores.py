"""Study what signals best distinguish elevator vs walking on both experimenters.

Signals evaluated per sample:
  (a) stillness = rolling std of (|a| - g) over 2s   (LOW inside elevator)
  (b) walk_band = rolling std of bandpassed 1.2-2.8 Hz component of |a|
  (c) vert_v   = LPF of integrated a_vert
  (d) vert_speed = |d/dt vert_v|  smoothed
We print median/percentiles in-GT vs out-GT for each."""
from __future__ import annotations

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter, ci_center,
)
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert,
)
from src.tests.segmentations.main_acc import build_acc_frame, build_height_frame


def butter_band(x, fs, lo, hi, order=4):
    sos = butter(order, [lo/(0.5*fs), hi/(0.5*fs)], btype="band", output="sos")
    return sosfiltfilt(sos, x)


def butter_low(x, fs, hi, order=4):
    sos = butter(order, hi/(0.5*fs), btype="low", output="sos")
    return sosfiltfilt(sos, x)


def dump(name):
    data = load_experimenter(name)
    acc_raw, prs = data["ACC"], data["PRS"]
    t0_ms = float(acc_raw["timestamp_ms"].iloc[0])
    acc = build_acc_frame(acc_raw, t0_ms)
    h = build_height_frame(prs, t0_ms)
    gt = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)).detect(h)

    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_mag = np.sqrt(ax*ax + ay*ay + az*az)
    a_vert = _compute_a_vert(ax, ay, az, fs)

    # stillness = std of |a| residual over 2s
    w2 = int(2.0 * fs)
    still = pd.Series(a_mag - pd.Series(a_mag).rolling(w2, center=True, min_periods=1).mean()).rolling(w2, center=True, min_periods=1).std().to_numpy()
    # walk-band energy
    ab = butter_band(a_mag, fs, 1.2, 2.8)
    walk = pd.Series(ab*ab).rolling(w2, center=True, min_periods=1).mean().to_numpy()
    walk = np.sqrt(walk)
    # vertical velocity
    v = np.cumsum(a_vert - a_vert.mean()) / fs
    v_lpf = butter_low(v, fs, 0.3)
    v_speed = pd.Series(np.abs(np.gradient(v_lpf, 1.0/fs))).rolling(int(1.0*fs), center=True, min_periods=1).mean().to_numpy()

    in_gt = np.zeros_like(t, dtype=bool)
    for _, row in gt.iterrows():
        lo = ci_center(row["start_ci"]); hi = ci_center(row["end_ci"])
        in_gt |= (t >= lo) & (t <= hi)

    print(f"\n### {name}  GT frac = {in_gt.mean():.3f}  (|GT|={len(gt)})")
    for label, sig in [("stillness σ(|a|)", still),
                       ("walkband RMS   ", walk),
                       ("|v_lpf|        ", np.abs(v_lpf)),
                       ("vert_speed     ", v_speed)]:
        sin, sout = sig[in_gt], sig[~in_gt]
        print(f"{label}  inGT p25/50/75/95 = "
              f"{np.percentile(sin,25):.3f} / {np.percentile(sin,50):.3f} / "
              f"{np.percentile(sin,75):.3f} / {np.percentile(sin,95):.3f}   "
              f"outGT p50/75/90/95 = "
              f"{np.percentile(sout,50):.3f} / {np.percentile(sout,75):.3f} / "
              f"{np.percentile(sout,90):.3f} / {np.percentile(sout,95):.3f}")


if __name__ == "__main__":
    for n in ("oria", "roy_turgman"):
        dump(n)
