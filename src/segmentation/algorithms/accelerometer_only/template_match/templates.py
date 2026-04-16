"""Per-experimenter pulse templates fitted from GT-labeled rides.

A ride's LPF velocity has two stereotyped sub-bumps (Peters, Ideal Lift
Kinematics, 1995): one during the entry acceleration phase ("pulse-up"),
one during the exit deceleration phase ("pulse-down"). The same is true
of the gravity-projected vertical acceleration, with sharper edges. We
extract both signals over each GT segment, split into entry / exit halves,
amplitude-normalize, time-warp resample to a fixed length, and average
across rides for that experimenter.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from src.segmentation.algorithms.accelerometer_only.acc_segmentation import (
    _compute_a_vert, compute_velocity, lowpass,
)
from src.segmentation.algorithms.metrics import ci_center


@dataclass
class Templates:
    pulse_up_v: np.ndarray
    pulse_down_v: np.ndarray
    pulse_up_a: np.ndarray
    pulse_down_a: np.ndarray
    meta: dict[str, Any] = field(default_factory=dict)


def _resample(x: np.ndarray, n_out: int) -> np.ndarray:
    if len(x) == 0:
        return np.zeros(n_out)
    src = np.linspace(0.0, 1.0, len(x))
    dst = np.linspace(0.0, 1.0, n_out)
    return np.interp(dst, src, x)


def _build_signals(
    acc_frame: pd.DataFrame, fs: float, lpf_hz: float,
    time_col: str, x_col: str, y_col: str, z_col: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    t = acc_frame[time_col].to_numpy(dtype=float)
    ax = acc_frame[x_col].to_numpy(dtype=float)
    ay = acc_frame[y_col].to_numpy(dtype=float)
    az = acc_frame[z_col].to_numpy(dtype=float)
    a_vert = _compute_a_vert(ax, ay, az, fs)
    v = compute_velocity(a_vert, fs)
    v_lpf = lowpass(v, fs, cutoff_hz=lpf_hz)
    return t, v_lpf, a_vert


def fit_templates(
    acc_frame: pd.DataFrame,
    gt_segments: pd.DataFrame,
    config,
    name: str = "",
) -> Templates:
    """Build per-experimenter pulse templates from GT-labeled segments.

    Each GT segment defines a window of LPF velocity and a_vert; the entry
    and exit halves are resampled and averaged across rides.
    """
    fs = float(config.fs_hz)
    n = int(config.template_len)
    t, v_lpf, a_vert = _build_signals(
        acc_frame, fs, float(config.lpf_hz),
        config.time_col, config.x_col, config.y_col, config.z_col,
    )
    if len(gt_segments) == 0:
        zeros = np.zeros(n)
        return Templates(zeros, zeros, zeros, zeros,
                         meta={"name": name, "n_rides": 0})

    ups_v, downs_v, ups_a, downs_a = [], [], [], []
    durations = []
    for _, row in gt_segments.iterrows():
        s = ci_center(row["start_ci"])
        e = ci_center(row["end_ci"])
        if e <= s:
            continue
        i0 = int(np.searchsorted(t, s))
        i1 = int(np.searchsorted(t, e))
        if i1 - i0 < 8:
            continue
        seg_v = v_lpf[i0:i1]
        seg_a = a_vert[i0:i1]
        L = len(seg_v)
        n_entry = max(2, int(round(L * float(config.entry_frac))))
        n_exit = max(2, int(round(L * float(config.exit_frac))))
        # raw halves (no amplitude normalization) — we want the template to
        # carry the per-elevator velocity amplitude.
        ups_v.append(_resample(seg_v[:n_entry], n))
        downs_v.append(_resample(seg_v[-n_exit:], n))
        ups_a.append(_resample(seg_a[:n_entry], n))
        downs_a.append(_resample(seg_a[-n_exit:], n))
        durations.append(e - s)

    if not ups_v:
        zeros = np.zeros(n)
        return Templates(zeros, zeros, zeros, zeros,
                         meta={"name": name, "n_rides": 0})

    return Templates(
        pulse_up_v=np.mean(ups_v, axis=0),
        pulse_down_v=np.mean(downs_v, axis=0),
        pulse_up_a=np.mean(ups_a, axis=0),
        pulse_down_a=np.mean(downs_a, axis=0),
        meta={
            "name": name,
            "n_rides": len(ups_v),
            "mean_duration_sec": float(np.mean(durations)),
            "median_duration_sec": float(np.median(durations)),
            "template_len": n,
            "fs_hz": fs,
            "entry_frac": float(config.entry_frac),
            "exit_frac": float(config.exit_frac),
        },
    )


def save_templates(templates: Templates, path: Path | str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "pulse_up_v": templates.pulse_up_v.tolist(),
        "pulse_down_v": templates.pulse_down_v.tolist(),
        "pulse_up_a": templates.pulse_up_a.tolist(),
        "pulse_down_a": templates.pulse_down_a.tolist(),
        "meta": templates.meta,
    }
    with open(path, "w") as f:
        json.dump(payload, f, indent=2)


def load_templates(path: Path | str) -> Templates:
    with open(path, "r") as f:
        d = json.load(f)
    return Templates(
        pulse_up_v=np.asarray(d["pulse_up_v"], dtype=float),
        pulse_down_v=np.asarray(d["pulse_down_v"], dtype=float),
        pulse_up_a=np.asarray(d["pulse_up_a"], dtype=float),
        pulse_down_a=np.asarray(d["pulse_down_a"], dtype=float),
        meta=d.get("meta", {}),
    )
