"""Check: does a simple stillness threshold catch every GT ride?"""
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
    walkband_rms, _compute_a_vert, compute_velocity, lowpass,
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
    walk = walkband_rms(ax, ay, az, fs, 1.2, 2.8, 2.0)
    walk_norm = walk / (np.percentile(walk, 75) + 1e-9)

    a_vert = _compute_a_vert(ax, ay, az, fs)
    v = lowpass(compute_velocity(a_vert, fs), fs)

    print(f"\n{name}  n_GT={len(gt)}  walk_median={np.median(walk):.3f}  p25={np.percentile(walk,25):.3f}")
    print(f"  per-ride walkband stats & velocity excursion:")
    for i, (_, r) in enumerate(gt.iterrows()):
        lo = ci_center(r["start_ci"]); hi = ci_center(r["end_ci"])
        m = (t >= lo) & (t <= hi)
        wseg = walk[m]; wnseg = walk_norm[m]; vseg = v[m]
        dv = float(np.max(vseg) - np.min(vseg)) if len(vseg) else 0.0
        print(f"    ride{i:02d} [{lo:.0f},{hi:.0f}]  walk p50={np.median(wseg):.3f} p75={np.percentile(wseg,75):.3f}  "
              f"walk_norm p50={np.median(wnseg):.3f}  Δv={dv:.3f}")
    # also: how many contiguous stillness blocks exist at enter=0.6?
    enter = 0.6
    mask = walk_norm <= enter
    blocks = []
    in_b = False
    for i, m in enumerate(mask):
        if m and not in_b:
            in_b = True; s = i
        elif not m and in_b:
            in_b = False; blocks.append((s, i))
    if in_b: blocks.append((s, len(mask)))
    durs = [(t[e-1]-t[s]) for s, e in blocks]
    print(f"  stillness<={enter}: {len(blocks)} blocks, durations median={np.median(durs):.1f}s max={max(durs) if durs else 0:.1f}s, total {sum(durs):.0f}s")
