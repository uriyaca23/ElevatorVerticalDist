"""Replicate the file-upload code path's data preparation and run the
detector on the result, to figure out why the UI sees only 1 segment.
"""
from __future__ import annotations
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import pandas as pd

from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import detect as _detect

GRAVITY = 9.80665

CSV = REPO / "tmp_boutique_capture" / "csvs" / "clean_S23_milleniumOutside.csv"
df = pd.read_csv(CSV)
print("upload csv columns:", df.columns.tolist())
print("rows:", len(df), "duration_s:", df["time_s"].iloc[-1])

t_raw = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
a = pd.to_numeric(df["a_vert_ms2"], errors="coerce").to_numpy(dtype=float)
good = np.isfinite(t_raw) & np.isfinite(a)
t_raw = t_raw[good]
a = a[good]
span = float(t_raw[-1] - t_raw[0])
if span > 0 and span < 1e4:
    ts_ms = (t_raw * 1000.0).astype("int64")
else:
    ts_ms = t_raw.astype("int64")
n = t_raw.size
acc_upload = pd.DataFrame({
    "timestamp_ms": ts_ms,
    "x": np.zeros(n),
    "y": np.zeros(n),
    "z": GRAVITY + a,
})
print("\nupload-mode ACC head:")
print(acc_upload.head(3))
print("upload-mode ACC z stats: mean=%.3f std=%.3f range=[%.3f,%.3f]" %
      (acc_upload['z'].mean(), acc_upload['z'].std(),
       acc_upload['z'].min(), acc_upload['z'].max()))

preds_upload, state_upload = _detect.predict_intervals(acc_upload)
print(f"\nupload-mode predictions: {len(preds_upload)}")
for p in preds_upload[:5]:
    print(f"  {p['ride_type']:5s} t=[{p['t_start_s']:.1f}, {p['t_end_s']:.1f}]s "
          f"r2={p.get('joint_r2_mean',0):.3f}")

# Compare to real 3-axis
exp = "eyalyakir_milleniumOutside_SamsungSM-S911B_15-04-2026_exp3"
real_acc = pd.read_csv(REPO / "src" / "data" / "structuredData" / "data" / exp / "ACC.csv")
real_acc = real_acc[["timestamp_ms", "x", "y", "z"]].copy()
print("\nreal ACC z stats: mean=%.3f std=%.3f" %
      (real_acc['z'].mean(), real_acc['z'].std()))
preds_real, state_real = _detect.predict_intervals(real_acc)
print(f"real-mode predictions: {len(preds_real)}")
for p in preds_real[:5]:
    print(f"  {p['ride_type']:5s} t=[{p['t_start_s']:.1f}, {p['t_end_s']:.1f}]s "
          f"r2={p.get('joint_r2_mean',0):.3f}")
