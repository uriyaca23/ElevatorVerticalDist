"""Plot |acc| and its first / second integrals with GT height overlay."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def plot_acc_integrals_with_gt(
    integrals: pd.DataFrame,
    height: pd.DataFrame | None = None,
    segments: pd.DataFrame | None = None,
    title: str = "|acc| and its integrals vs GT height",
    save_path: Path | str | None = None,
    show: bool = True,
) -> plt.Figure:
    """Plot linear acceleration, velocity and displacement vs time.

    Parameters
    ----------
    integrals : DataFrame with columns ``['time', 'a_lin', 'vel', 'pos']``.
    height : optional DataFrame with ``['time', 'height']`` — overlaid on a
        twin axis in every panel.
    segments : optional DataFrame with ``['start', 'end', ...]`` — shaded.
    """
    t = integrals["time"].to_numpy()

    fig, axes = plt.subplots(
        3, 1, figsize=(14, 9), sharex=True,
        gridspec_kw={"hspace": 0.08},
    )
    ax_acc, ax_vel, ax_pos = axes

    ax_acc.plot(t, integrals["a_lin"], color="black", linewidth=0.7, label="|acc| − ḡ")
    ax_acc.set_ylabel("acc (m/s²)")
    ax_acc.set_title(title)
    ax_acc.grid(True, alpha=0.3)

    ax_vel.plot(t, integrals["vel"], color="tab:orange", linewidth=0.9, label="∫(|acc|−ḡ) dt")
    ax_vel.set_ylabel("velocity (m/s)")
    ax_vel.grid(True, alpha=0.3)

    ax_pos.plot(t, integrals["pos"], color="tab:green", linewidth=1.2, label="∫∫ (estimated Δheight)")
    ax_pos.set_ylabel("displacement (m)")
    ax_pos.set_xlabel("time (s)")
    ax_pos.grid(True, alpha=0.3)

    if segments is not None and len(segments) > 0:
        seg_label_used = False
        for row in segments.itertuples(index=False):
            s_lo, s_hi = row.start_ci
            e_lo, e_hi = row.end_ci
            s_mid = 0.5 * (s_lo + s_hi)
            e_mid = 0.5 * (e_lo + e_hi)
            for ax in axes:
                ax.axvspan(
                    s_mid, e_mid, color="tab:red", alpha=0.15,
                    label=None if seg_label_used else "Elevator Active",
                )
                if s_hi > s_lo:
                    ax.axvspan(s_lo, s_hi, color="tab:red", alpha=0.07)
                if e_hi > e_lo:
                    ax.axvspan(e_lo, e_hi, color="tab:red", alpha=0.07)
                seg_label_used = True

    if height is not None and len(height) > 0:
        th = height["time"].to_numpy()
        hv = height["height"].to_numpy()
        for ax in axes:
            ax_h = ax.twinx()
            ax_h.plot(th, hv, color="tab:blue", linewidth=2.0, label="Barometer height (m)")
            ax_h.set_ylabel("baro height (m)", color="tab:blue")
            ax_h.tick_params(axis="y", labelcolor="tab:blue")
            ax_h.legend(loc="upper right", fontsize=9)

    ax_acc.legend(loc="upper left", fontsize=9)
    ax_vel.legend(loc="upper left", fontsize=9)
    ax_pos.legend(loc="upper left", fontsize=9)

    fig.tight_layout()

    if save_path is not None:
        fig.savefig(save_path, dpi=150)
    if show:
        plt.show()

    return fig
