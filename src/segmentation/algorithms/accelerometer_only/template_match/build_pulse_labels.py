"""Build per-ride pulse-shape labels from TRAIN-set GT.

For every ``up`` / ``down`` GT ride in every training experiment, compute
the smoothed vertical velocity using the project's standard chain

    gravity project → integrate → 0.3 Hz low-pass

and save three artifact sets under ``template_match/labels/pulseShapes/``:

    pulseShapes/up/<exp>__<idx>.png            individual ride plot
    pulseShapes/down/<exp>__<idx>.png          individual ride plot
    pulseShapes/by_experiment/<exp>.png        grid of all rides for that exp

Run:
    venv/bin/python -m src.segmentation.algorithms.accelerometer_only.template_match.build_pulse_labels
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import (  # noqa: E402
    ExperimentPipeline,
    getExperimentData,
    list_experiments,
)
from src.physics import calculate_velocity_from_accelerometer  # noqa: E402

LABELS_ROOT = Path(__file__).with_name("labels") / "pulseShapes"
UP_DIR = LABELS_ROOT / "up"
DOWN_DIR = LABELS_ROOT / "down"
BY_EXP_DIR = LABELS_ROOT / "by_experiment"

TYPE_COLOR = {"up": "#27ae60", "down": "#e74c3c"}


def estimate_fs_from_ms(acc: pd.DataFrame, default: float = 100.0) -> float:
    ts = acc["timestamp_ms"].to_numpy(dtype=float)
    if len(ts) < 2:
        return default
    dt_ms = float(np.median(np.diff(ts)))
    if dt_ms <= 0:
        return default
    return 1000.0 / dt_ms


def compute_session_velocity(acc: pd.DataFrame, fs: float) -> tuple[np.ndarray, np.ndarray, int]:
    """Return (t_sec, v, t0_ms) for the full ACC frame."""
    t0_ms = int(acc["timestamp_ms"].iloc[0])
    t_sec = (acc["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
    v = calculate_velocity_from_accelerometer(
        acc["x"].to_numpy(dtype=float),
        acc["y"].to_numpy(dtype=float),
        acc["z"].to_numpy(dtype=float),
        fs,
    )
    return t_sec, v, t0_ms


def save_ride_plot(path: Path, t: np.ndarray, v: np.ndarray, title: str, color: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 3))
    ax.plot(t, v, color=color, linewidth=1.3)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("vz (m/s)")
    ax.set_title(title, fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def save_experiment_grid(path: Path, exp_name: str, rides: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    n = len(rides)
    ncols = min(4, n)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(4.0 * ncols, 2.6 * nrows),
        squeeze=False,
    )
    for i, ride in enumerate(rides):
        ax = axes[i // ncols][i % ncols]
        ax.plot(ride["t"], ride["v"], color=TYPE_COLOR[ride["type"]], linewidth=1.1)
        ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
        ax.set_title(
            f"R{ride['idx']} — {ride['type']} ({ride['duration']:.1f}s)",
            fontsize=9,
        )
        ax.set_xlabel("t (s)", fontsize=8)
        ax.set_ylabel("v (m/s)", fontsize=8)
        ax.tick_params(labelsize=7)
    for i in range(n, nrows * ncols):
        axes[i // ncols][i % ncols].axis("off")
    fig.suptitle(f"{exp_name} — {n} GT rides", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)


def process_experiment(name: str) -> int:
    """Save labels for one experiment. Returns number of rides written."""
    sensors, gt, meta = getExperimentData(name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        return 0

    acc = sensors["ACC"]
    fs = estimate_fs_from_ms(acc)
    t_sec, v_lpf, t0_ms = compute_session_velocity(acc, fs)

    rides: list[dict] = []
    ride_idx = 0
    for _slice, row, _meta in ExperimentPipeline(sensors, gt, meta):
        typ = str(row["type"])
        if typ not in ("up", "down"):
            continue
        ride_idx += 1
        s_sec = (int(row["start_ms"]) - t0_ms) / 1000.0
        e_sec = (int(row["end_ms"]) - t0_ms) / 1000.0
        i0 = int(np.searchsorted(t_sec, s_sec))
        i1 = int(np.searchsorted(t_sec, e_sec))
        if i1 <= i0:
            continue
        t_ride = t_sec[i0:i1] - s_sec
        v_ride = v_lpf[i0:i1]
        duration = float(t_ride[-1])

        target_dir = UP_DIR if typ == "up" else DOWN_DIR
        fname = f"{name}__{ride_idx:02d}.png"
        save_ride_plot(
            target_dir / fname,
            t_ride, v_ride,
            title=f"{name}\nR{ride_idx} {typ} — {duration:.1f}s",
            color=TYPE_COLOR[typ],
        )
        rides.append({
            "idx": ride_idx,
            "type": typ,
            "t": t_ride,
            "v": v_ride,
            "duration": duration,
        })

    if rides:
        save_experiment_grid(BY_EXP_DIR / f"{name}.png", name, rides)
    return len(rides)


def main() -> int:
    UP_DIR.mkdir(parents=True, exist_ok=True)
    DOWN_DIR.mkdir(parents=True, exist_ok=True)
    BY_EXP_DIR.mkdir(parents=True, exist_ok=True)

    total = 0
    names = list_experiments(kind="train")
    print(f"processing {len(names)} TRAIN experiments → {LABELS_ROOT}")
    for name in names:
        try:
            n = process_experiment(name)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            continue
        print(f"[ok]    {name}: {n} rides")
        total += n
    print(f"\nwrote {total} rides across {len(names)} experiments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
