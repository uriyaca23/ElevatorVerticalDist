"""Sliding-window ZUPT displacement: per-ride peak vs session background."""
from __future__ import annotations
import sys
from pathlib import Path
import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter, ci_center,
)
from src.algorithms.segmentation_algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, sliding_zupt_disp,
)
from src.tests.segmentations.main_acc import build_acc_frame, build_height_frame


for name in ("oria", "roy_turgman"):
    d = load_experimenter(name)
    t0 = float(d["ACC"]["timestamp_ms"].iloc[0])
    acc = build_acc_frame(d["ACC"], t0)
    h = build_height_frame(d["PRS"], t0)
    gt = Segmenter(SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)).detect(h)
    fs = 100.0
    t = acc["time"].to_numpy()
    ax, ay, az = [acc[c].to_numpy() for c in "xyz"]
    a_vert = _compute_a_vert(ax, ay, az, fs)

    import pandas as pd
    v = np.cumsum(a_vert - a_vert.mean()) / fs
    for W_long in (60.0,):
        v_hp = v - pd.Series(v).rolling(int(W_long*fs), center=True, min_periods=1).mean().to_numpy()
        for W in (10.0, 12.0, 8.0):
            Wn = int(W*fs)
            score = pd.Series(v_hp).rolling(Wn, center=True, min_periods=1).max().to_numpy() - pd.Series(v_hp).rolling(Wn, center=True, min_periods=1).min().to_numpy()
            print(f"\n*** W_long={W_long}s W_pp={W}s ***")
        in_gt = np.zeros_like(t, dtype=bool)
        for _, r in gt.iterrows():
            lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
            in_gt |= (t >= lo) & (t <= hi)
        sin, sout = score[in_gt], score[~in_gt]
        print(f"\n### {name}  W={W}s")
        print(f"  in-GT p25/50/75/95 = {np.percentile(sin,25):.2f} / {np.percentile(sin,50):.2f} / {np.percentile(sin,75):.2f} / {np.percentile(sin,95):.2f}")
        print(f"  out-GT p50/75/90/95 = {np.percentile(sout,50):.2f} / {np.percentile(sout,75):.2f} / {np.percentile(sout,90):.2f} / {np.percentile(sout,95):.2f}")
        # per-ride peak
        print("  per-ride peak score:")
        peaks_in = []
        for i, (_, r) in enumerate(gt.iterrows()):
            lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
            m = (t >= lo) & (t <= hi)
            peaks_in.append(float(score[m].max()))
        peaks_in = np.array(peaks_in)
        print(f"    per-ride peak: min={peaks_in.min():.2f}  p25={np.percentile(peaks_in,25):.2f}  p50={np.percentile(peaks_in,50):.2f}  p75={np.percentile(peaks_in,75):.2f}  max={peaks_in.max():.2f}")
