"""combined log_disp - 0.7*log_var score distribution."""
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
    _compute_a_vert, sliding_zupt_disp,
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

    disp = sliding_zupt_disp(a_vert, fs, 10.0)
    W = int(4*fs)
    var_av = pd.Series(a_vert*a_vert).rolling(W, center=True, min_periods=1).mean().to_numpy() - (
        pd.Series(a_vert).rolling(W, center=True, min_periods=1).mean().to_numpy()**2
    )
    sc = np.log10(np.maximum(disp,1e-4)) - 0.7*np.log10(np.maximum(var_av,1e-6))
    scn = sc - np.median(sc)

    in_gt = np.zeros_like(t, dtype=bool)
    for _, r in gt.iterrows():
        lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
        in_gt |= (t >= lo) & (t <= hi)
    sin, sout = scn[in_gt], scn[~in_gt]
    print(f"\n### {name}")
    print(f"  score_norm inGT p25/50/75/95 = {np.percentile(sin,25):.3f}/{np.percentile(sin,50):.3f}/{np.percentile(sin,75):.3f}/{np.percentile(sin,95):.3f}")
    print(f"  outGT p25/50/75/90/95 = {np.percentile(sout,25):.3f}/{np.percentile(sout,50):.3f}/{np.percentile(sout,75):.3f}/{np.percentile(sout,90):.3f}/{np.percentile(sout,95):.3f}")
    # per-ride peak
    peaks = []
    for _, r in gt.iterrows():
        lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
        m = (t>=lo)&(t<=hi); peaks.append(float(scn[m].max()))
    peaks = np.array(peaks)
    print(f"  per-ride peak: min={peaks.min():.3f} p25={np.percentile(peaks,25):.3f} p50={np.percentile(peaks,50):.3f}")
