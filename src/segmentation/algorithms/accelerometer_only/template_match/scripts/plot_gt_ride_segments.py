"""Per-GT-segment PNGs for the TRAIN set.

For each TRAIN experiment, slice every ground-truth ride (``up`` / ``down``
intervals from ``gt.csv``) out of the ACC stream and render the two signals
that label_review cares about — accelerometer magnitude ``|a|`` and the
smoothed vertical velocity ``vz`` — for that window only. Two artefacts are
written per experiment:

1. ``ride_<NN>_<type>.png`` — one two-panel PNG per ride.
2. ``_all_rides.png`` — a grid of small multiples covering every ride in
   the experiment, so all GT can be eyeballed in a single figure.

Outputs land under
``template_match/labels/ride_segments/<exp>/``.

Run:
    venv/bin/python -m src.segmentation.algorithms.accelerometer_only.\
template_match.scripts.plot_gt_ride_segments
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.physics import calculate_velocity_from_accelerometer  # noqa: E402

OUT_ROOT = Path(__file__).resolve().parents[1] / "labels" / "ride_segments"

# Fraction of the ride duration to pad on each side when slicing context.
CONTEXT_PAD_FRAC = 0.15
# Minimum padding (seconds) so very short rides still get visible context.
CONTEXT_PAD_MIN_S = 0.5

TYPE_COLORS: dict[str, str] = {
    "up": "#27ae60",
    "down": "#e74c3c",
}


@dataclass
class RideSlice:
    index: int
    ride_type: str
    start_s: float          # GT start, seconds since experiment t0
    end_s: float            # GT end, seconds since experiment t0
    t_rel: np.ndarray       # time axis, seconds, zeroed at slice start
    mag: np.ndarray         # |a| over the slice (with context padding)
    vz: np.ndarray          # smoothed vertical velocity over the slice
    gt_t0: float            # GT start time, in t_rel coordinates
    gt_t1: float            # GT end time, in t_rel coordinates


def _estimate_fs_hz(ts_ms: np.ndarray, default: float = 100.0) -> float:
    if ts_ms.size < 2:
        return default
    dt_ms = float(np.median(np.diff(ts_ms)))
    if dt_ms <= 0:
        return default
    return 1000.0 / dt_ms


def _slice_rides(acc: pd.DataFrame, gt: pd.DataFrame) -> list[RideSlice]:
    ts_ms = acc["timestamp_ms"].to_numpy(dtype=float)
    if ts_ms.size == 0:
        return []
    t0_ms = float(ts_ms[0])
    fs_hz = _estimate_fs_hz(ts_ms)
    t_sec = (ts_ms - t0_ms) / 1000.0

    ax = acc["x"].to_numpy(dtype=float)
    ay = acc["y"].to_numpy(dtype=float)
    az = acc["z"].to_numpy(dtype=float)
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    vz = calculate_velocity_from_accelerometer(ax, ay, az, fs_hz)

    rides = gt[gt["type"].isin(("up", "down"))].reset_index(drop=True)
    out: list[RideSlice] = []
    for i, row in rides.iterrows():
        s = (float(row["start_ms"]) - t0_ms) / 1000.0
        e = (float(row["end_ms"]) - t0_ms) / 1000.0
        if e <= s:
            continue
        pad = max(CONTEXT_PAD_MIN_S, CONTEXT_PAD_FRAC * (e - s))
        ws, we = s - pad, e + pad
        mask = (t_sec >= ws) & (t_sec <= we)
        if not np.any(mask):
            continue
        t_local = t_sec[mask] - (t_sec[mask][0])
        gt_t0 = s - t_sec[mask][0]
        gt_t1 = e - t_sec[mask][0]
        out.append(
            RideSlice(
                index=int(i),
                ride_type=str(row["type"]),
                start_s=s,
                end_s=e,
                t_rel=t_local,
                mag=mag[mask],
                vz=vz[mask],
                gt_t0=gt_t0,
                gt_t1=gt_t1,
            )
        )
    return out


def _draw_ride(
    ax_top: plt.Axes, ax_bot: plt.Axes, ride: RideSlice, *, title: str | None = None
) -> None:
    color = TYPE_COLORS.get(ride.ride_type, "#7f8c8d")
    ax_top.axvspan(ride.gt_t0, ride.gt_t1, color=color, alpha=0.18, zorder=0)
    ax_top.plot(ride.t_rel, ride.mag, color="#2c3e50", lw=0.7)
    ax_top.axhline(9.81, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax_top.set_ylabel("|a| (m/s²)")
    ax_top.grid(True, alpha=0.25)

    ax_bot.axvspan(ride.gt_t0, ride.gt_t1, color=color, alpha=0.18, zorder=0)
    ax_bot.plot(ride.t_rel, ride.vz, color="#2980b9", lw=1.0)
    ax_bot.axhline(0.0, color="gray", lw=0.4, alpha=0.5)
    ax_bot.set_ylabel("vz (m/s)")
    ax_bot.set_xlabel("t (s, ride-local)")
    ax_bot.grid(True, alpha=0.25)

    if title:
        ax_top.set_title(title, fontsize=10)


def _save_per_ride(ride: RideSlice, out_dir: Path, exp_name: str) -> Path:
    fig, axes = plt.subplots(2, 1, figsize=(7, 4.2), sharex=True)
    duration = ride.end_s - ride.start_s
    title = (
        f"{exp_name}\n"
        f"ride {ride.index:02d} ({ride.ride_type}) — "
        f"{duration:.1f}s @ t={ride.start_s:.1f}s"
    )
    _draw_ride(axes[0], axes[1], ride, title=title)
    fig.tight_layout()
    out_path = out_dir / f"ride_{ride.index:02d}_{ride.ride_type}.png"
    fig.savefig(out_path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_combined(rides: list[RideSlice], out_dir: Path, exp_name: str) -> Path:
    n = len(rides)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    # One outer cell per ride; each cell holds a 2-row subgridspec (|a| on top,
    # vz on bottom). Outer hspace gives breathing room for titles.
    fig = plt.figure(figsize=(4.6 * cols, 3.6 * rows))
    outer = fig.add_gridspec(rows, cols, hspace=0.55, wspace=0.35)

    for i, ride in enumerate(rides):
        r, c = divmod(i, cols)
        inner = outer[r, c].subgridspec(2, 1, hspace=0.08)
        ax_top = fig.add_subplot(inner[0])
        ax_bot = fig.add_subplot(inner[1], sharex=ax_top)
        duration = ride.end_s - ride.start_s
        title = (
            f"#{ride.index:02d} {ride.ride_type} — "
            f"{duration:.1f}s @ t={ride.start_s:.0f}s"
        )
        _draw_ride(ax_top, ax_bot, ride, title=title)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        ax_top.set_xlabel("")

    fig.suptitle(f"{exp_name} — all GT rides", fontsize=13)
    out_path = out_dir / "_all_rides.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def process(name: str) -> tuple[int, Path | None]:
    sensors, gt, _meta = getExperimentData(name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        return 0, None
    acc = sensors["ACC"]
    rides = _slice_rides(acc, gt)
    if not rides:
        return 0, None

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    for ride in rides:
        _save_per_ride(ride, out_dir, name)
    combined = _save_combined(rides, out_dir, name)
    return len(rides), combined


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    names = list_experiments(kind="train")
    print(f"processing {len(names)} TRAIN experiments → {OUT_ROOT}")

    total_rides = 0
    written_exps = 0
    for name in names:
        try:
            n_rides, combined = process(name)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            continue
        if n_rides == 0:
            print(f"[skip]  {name}: no usable GT rides")
            continue
        total_rides += n_rides
        written_exps += 1
        print(f"[ok]    {name}: {n_rides} rides → {combined.parent}")

    print(f"\nwrote {total_rides} ride PNGs across {written_exps} experiments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
