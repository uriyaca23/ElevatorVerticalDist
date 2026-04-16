"""Per-experiment overview plot: accelerometer + smoothed velocity + GT shading.

Shows the full session timeline so you can visually sanity-check GT against
the raw and derived signals.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physics import calculate_velocity_from_accelerometer

TYPE_COLORS: dict[str, str] = {
    "up": "#27ae60",
    "down": "#e74c3c",
    "outside": "#bdc3c7",
}


def _estimate_fs_hz(acc: pd.DataFrame, default: float = 100.0) -> float:
    ts = acc["timestamp_ms"].to_numpy(dtype=float)
    if len(ts) < 2:
        return default
    dt_ms = float(np.median(np.diff(ts)))
    if dt_ms <= 0:
        return default
    return 1000.0 / dt_ms


def _shade_gt(ax: Any, gt: pd.DataFrame, t0_ms: int, draw_outside: bool) -> None:
    for _, row in gt.iterrows():
        typ = str(row["type"])
        color = TYPE_COLORS.get(typ)
        if color is None:
            continue
        if typ == "outside" and not draw_outside:
            continue
        s = (int(row["start_ms"]) - t0_ms) / 1000.0
        e = (int(row["end_ms"]) - t0_ms) / 1000.0
        alpha = 0.08 if typ == "outside" else 0.22
        ax.axvspan(s, e, color=color, alpha=alpha, zorder=0)


def plot_experiment_overview(
    acc: pd.DataFrame,
    gt: pd.DataFrame,
    *,
    name: str = "",
    fs: float | None = None,
    draw_outside: bool = False,
    save_path: Path | str | None = None,
    show: bool = False,
) -> plt.Figure:
    """Two-panel overview of a single experiment.

    Panel 1: accelerometer magnitude |a| with GT intervals shaded behind.
    Panel 2: smoothed vertical velocity (``gravity project → integrate →
    0.3 Hz low-pass``) with the same shading.

    Args:
        acc: loader-schema ACC frame with ``timestamp_ms, x, y, z``.
        gt: loader-schema GT frame with ``start_ms, end_ms, type``.
        name: optional experiment name used in the figure title.
        fs: sample rate in Hz. Estimated from ``timestamp_ms`` when None.
        draw_outside: when True, ``outside`` GT rows are faintly shaded
            (default False — usually these cover most of the timeline).
        save_path: writes a PNG if given.
        show: calls ``plt.show()`` if True.

    Returns the Figure.
    """
    if acc.empty:
        raise ValueError("acc frame is empty")

    fs_hz = float(fs) if fs is not None else _estimate_fs_hz(acc)
    t0_ms = int(acc["timestamp_ms"].iloc[0])
    t_sec = (acc["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
    ax_arr = acc["x"].to_numpy(dtype=float)
    ay_arr = acc["y"].to_numpy(dtype=float)
    az_arr = acc["z"].to_numpy(dtype=float)
    mag = np.sqrt(ax_arr * ax_arr + ay_arr * ay_arr + az_arr * az_arr)

    v = calculate_velocity_from_accelerometer(ax_arr, ay_arr, az_arr, fs_hz)

    fig, axes = plt.subplots(
        2, 1, figsize=(16, 6), sharex=True,
        gridspec_kw={"height_ratios": [1.0, 1.0]},
    )

    _shade_gt(axes[0], gt, t0_ms, draw_outside=draw_outside)
    axes[0].plot(t_sec, mag, linewidth=0.4, color="#2c3e50", alpha=0.9)
    axes[0].axhline(9.81, color="gray", linewidth=0.4, linestyle="--", alpha=0.5)
    axes[0].set_ylabel("|a| (m/s²)")
    axes[0].set_ylim(
        max(0.0, float(np.percentile(mag, 0.5)) - 1.0),
        float(np.percentile(mag, 99.5)) + 1.0,
    )
    axes[0].grid(True, alpha=0.25)

    _shade_gt(axes[1], gt, t0_ms, draw_outside=draw_outside)
    axes[1].plot(t_sec, v, linewidth=0.8, color="#2980b9", alpha=0.95)
    axes[1].axhline(0.0, color="gray", linewidth=0.4, alpha=0.5)
    axes[1].set_ylabel("vz (m/s)")
    axes[1].set_xlabel("time (s)")
    axes[1].grid(True, alpha=0.25)

    handles = [
        plt.Line2D([0], [0], color=TYPE_COLORS["up"], lw=8, alpha=0.3, label="up"),
        plt.Line2D([0], [0], color=TYPE_COLORS["down"], lw=8, alpha=0.3, label="down"),
    ]
    if draw_outside:
        handles.append(
            plt.Line2D([0], [0], color=TYPE_COLORS["outside"], lw=8, alpha=0.3, label="outside")
        )
    axes[0].legend(handles=handles, loc="upper right", fontsize=9)

    title = f"{name} — |a| + cumsum velocity (fs≈{fs_hz:.1f} Hz)" if name \
        else f"|a| + cumsum velocity (fs≈{fs_hz:.1f} Hz)"
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=(0, 0, 1, 0.97))

    if save_path is not None:
        path = Path(save_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, dpi=130, bbox_inches="tight")
    if show:
        plt.show()

    return fig
