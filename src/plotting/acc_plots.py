"""Accelerometer plots with GT height overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.algorithms.height_segmentation import detect_elevator_segments_from_height


def plot_acc_with_gt(
    acc: pd.DataFrame,
    prs: pd.DataFrame | None = None,
    title: str = "Accelerometer with GT height",
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot ACC x/y/z and |acc| over time, with GT height in the background.

    Parameters
    ----------
    acc : DataFrame with columns ['timestamp_ms', 'x', 'y', 'z']
    prs : optional DataFrame with ['timestamp_ms', 'GT_height_m'] (background)
    """
    t0_ms = acc["timestamp_ms"].iloc[0]
    t = (acc["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
    mag = np.sqrt(acc["x"] ** 2 + acc["y"] ** 2 + acc["z"] ** 2)

    has_gt = prs is not None and "GT_height_m" in prs.columns and len(prs) > 0

    fig, ax = plt.subplots(figsize=(14, 6))

    lines = []
    lines += ax.plot(t, acc["x"], label="acc_x", linewidth=0.8, alpha=0.8)
    lines += ax.plot(t, acc["y"], label="acc_y", linewidth=0.8, alpha=0.8)
    lines += ax.plot(t, acc["z"], label="acc_z", linewidth=0.8, alpha=0.8)
    lines += ax.plot(t, mag, label="|acc|", linewidth=1.0, color="black", alpha=0.9)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("acceleration (m/s²)")
    ax.set_title(title)
    ax.grid(True, alpha=0.3)

    if has_gt:
        t_prs = (prs["timestamp_ms"].to_numpy() - t0_ms) / 1000.0
        height = prs["GT_height_m"].to_numpy() - prs["GT_height_m"].iloc[0]
        height_smooth = pd.Series(height).rolling(window=51, center=True, min_periods=1).median().to_numpy()

        segments = detect_elevator_segments_from_height(t_prs, height_smooth)
        seg_patch = None
        for i, (s, e) in enumerate(segments):
            patch = ax.axvspan(s, e, color="tab:red", alpha=0.18)
            if i == 0:
                seg_patch = patch
                seg_patch.set_label("Elevator Active")

        ax_h = ax.twinx()
        baro_line, = ax_h.plot(
            t_prs, height_smooth, color="tab:blue", linewidth=2.0,
            label="Barometer Δheight (m)",
        )
        ax_h.set_ylabel("Δheight (m)", color="tab:blue")
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


if __name__ == "__main__":
    import sys

    if __package__ in (None, ""):
        sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

    from src.data.loader import load_experimenter

    name = sys.argv[1] if len(sys.argv) > 1 else "oria"
    data = load_experimenter(name)
    plot_acc_with_gt(
        data["ACC"],
        data.get("PRS"),
        title=f"Accelerometer + GT height — {name}",
    )
