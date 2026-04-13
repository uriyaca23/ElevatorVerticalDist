"""Accelerometer plots with GT height overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_acc_with_gt(
    acc: pd.DataFrame,
    height: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
    title: str = "Accelerometer with GT height",
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot ACC x/y/z and |acc| over time, with GT height in the background.

    Parameters
    ----------
    acc : DataFrame with columns ``['time', 'x', 'y', 'z', 'mag']``
        All values are already in their display units (seconds, m/s²).
    height : optional DataFrame with ``['time', 'height']``.
    segments : optional DataFrame with ``['start_ci', 'end_ci', ...]`` where
        each CI is a ``(low, high)`` tuple. The shaded span uses the CI
        centers; CI bounds are drawn as faint edge lines.
    """
    t = acc["time"].to_numpy()

    fig, ax = plt.subplots(figsize=(14, 6))

    lines = []
    lines += ax.plot(t, acc["x"], label="acc_x", linewidth=0.8, alpha=0.8)
    lines += ax.plot(t, acc["y"], label="acc_y", linewidth=0.8, alpha=0.8)
    lines += ax.plot(t, acc["z"], label="acc_z", linewidth=0.8, alpha=0.8)
    lines += ax.plot(t, acc["mag"], label="|acc|", linewidth=1.0, color="black", alpha=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (m/s²)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    seg_patch = None
    if segments is not None and len(segments) > 0:
        for i, row in enumerate(segments.itertuples(index=False)):
            s_lo, s_hi = row.start_ci
            e_lo, e_hi = row.end_ci
            s_mid = 0.5 * (s_lo + s_hi)
            e_mid = 0.5 * (e_lo + e_hi)
            patch = ax.axvspan(s_mid, e_mid, color="tab:red", alpha=0.18)
            if s_hi > s_lo:
                ax.axvspan(s_lo, s_hi, color="tab:red", alpha=0.08)
            if e_hi > e_lo:
                ax.axvspan(e_lo, e_hi, color="tab:red", alpha=0.08)
            if i == 0:
                seg_patch = patch
                seg_patch.set_label("Elevator Active")

    if height is not None and len(height) > 0:
        ax_h = ax.twinx()
        baro_line, = ax_h.plot(
            height["time"].to_numpy(), height["height"].to_numpy(),
            color="tab:blue", linewidth=2.0, label="Barometer height (m)",
        )
        ax_h.set_ylabel("height (m)", color="tab:blue")
        ax_h.tick_params(axis="y", labelcolor="tab:blue")

        handles = lines + [baro_line]
        if seg_patch is not None:
            handles.append(seg_patch)
        ax.legend(handles=handles, loc="upper left", ncol=3, fontsize=9)
    else:
        ax.legend(loc="upper left", ncol=4, fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig
