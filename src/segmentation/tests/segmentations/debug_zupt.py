"""Per-GT ride: check if step-rate stillness covers it & ZUPT displacement."""
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
    _compute_a_vert, step_rate, zupt_integrate,
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
    sr = step_rate(ax, ay, az, fs, 4.0)

    print(f"\n### {name}  n_GT={len(gt)}")
    print("  per-ride step_rate stats & ZUPT displacement (assuming GT window is 'still')")
    for i, (_, r) in enumerate(gt.iterrows()):
        lo, hi = ci_center(r["start_ci"]), ci_center(r["end_ci"])
        i0 = int(np.searchsorted(t, lo)); i1 = int(np.searchsorted(t, hi))
        if i1 <= i0:
            continue
        sr_seg = sr[i0:i1]
        vc, dd = zupt_integrate(a_vert[i0:i1], fs)
        d_pp = float(dd.max() - dd.min())
        print(f"    ride{i:02d} [{lo:.0f},{hi:.0f}] dur={hi-lo:.0f}s  sr p50={np.median(sr_seg):.2f} p90={np.percentile(sr_seg,90):.2f}  Δd={d_pp:.2f}m  max|v|={float(np.abs(vc).max()):.2f}m/s")
