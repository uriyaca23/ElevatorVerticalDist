"""Inspect peak-to-peak displacement signal per GT ride."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter, ci_center,
)
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, lowpass, walkband_rms,
)
from src.tests.segmentations.main_acc import build_acc_frame, build_height_frame


for name in ("uriya", "roy_turgeman"):
    d = load_experimenter(name)
    t0 = float(d["ACC"]["timestamp_ms"].iloc[0])
    acc = build_acc_frame(d["ACC"], t0)
    h = build_height_frame(d["PRS"], t0)
    gt = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)).detect(h)

    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)
    from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import _bandpass
    a_band = _bandpass(a_vert, fs, 0.03, 0.3)
    v_band = np.cumsum(a_band)/fs
    W = int(8*fs)
    pp = pd.Series(np.abs(v_band)).rolling(W, center=True, min_periods=1).max().to_numpy()

    walk = walkband_rms(ax, ay, az, fs, 1.2, 2.8, 2.0)
    wn = walk / (np.percentile(walk, 75) + 1e-9)

    in_gt = np.zeros_like(t, dtype=bool)
    for _, r in gt.iterrows():
        lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
        in_gt |= (t >= lo) & (t <= hi)
    print(f"\n### {name}")
    print(f"  pp in-GT p25/50/75 = {np.percentile(pp[in_gt],25):.3f} / {np.percentile(pp[in_gt],50):.3f} / {np.percentile(pp[in_gt],75):.3f}")
    print(f"  pp out-GT p50/75/90/95 = {np.percentile(pp[~in_gt],50):.3f} / {np.percentile(pp[~in_gt],75):.3f} / {np.percentile(pp[~in_gt],90):.3f} / {np.percentile(pp[~in_gt],95):.3f}")
    # per-ride peak pp
    print("  per-ride max-pp and mean walk_norm:")
    for i, (_, r) in enumerate(gt.iterrows()):
        lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
        m = (t >= lo) & (t <= hi)
        print(f"    ride{i:02d} pp_max={pp[m].max():.3f} wn_mean={wn[m].mean():.3f}")
