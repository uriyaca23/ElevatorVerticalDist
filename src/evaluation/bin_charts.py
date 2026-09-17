"""The inclusion-by-bin chart, shared by every evaluator that draws one.

Three evaluators used to carry their own near-copy of this figure — the
prediction evaluator binned by |Δh| and by duration, the pipeline
evaluator binned by duration — and they drifted apart in wording, bin
edges and styling. They all call :func:`bin_rate_chart` now, so a change
here lands in every run.

The chart answers one question: *does the calibrated CI hold up across
the range of rides, or only on average?* Bars are the inclusion rate
(the fraction of segments whose truth fell inside the predicted CI) per
bin, the red line is MAE on the right-hand axis, and the target is drawn
as a **band** rather than a line — see
:func:`pyramidElevatorDist.utils.binomial_ci.target_band`. Each bin
holds a finite number of segments, so its rate carries sampling error;
the band is the range of rates consistent with a true 90 % inclusion
probability at that bin's own ``n``. A bar inside the band is on target,
whatever the eye says about the gap; a bar clearing it is genuinely off.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from pyramidElevatorDist.utils.binomial_ci import target_band


# Confidence level of the band drawn around the inclusion target.
BAND_CONFIDENCE = 0.95

# Ride-magnitude bins, in metres. Starts at 3 m: a sub-3 m ride is one
# floor, where the interval is wide relative to the distance and the
# rate says little, so those segments are left out of the figure.
DISTANCE_BIN_EDGES = np.array([3.0, 6.0, 12.0, 24.0, 60.0])

# Ride-duration bins, in seconds.
DURATION_BIN_EDGES = np.array([0.0, 5.0, 10.0, 20.0, 40.0, 90.0])


def _empty(out_path: Path, title: str, dpi: int | None) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.set_title(title + " (empty)")
    fig.savefig(out_path, **({"dpi": dpi} if dpi else {}))
    plt.close(fig)


def bin_rate_chart(
    values,
    covered,
    abs_error,
    out_path: Path,
    *,
    title: str,
    bin_edges: np.ndarray,
    unit: str,
    xlabel: str,
    bar_color: str = "tab:green",
    target: float = 0.9,
    band_confidence: float = BAND_CONFIDENCE,
    dpi: int | None = None,
) -> None:
    """Inclusion rate + MAE per bin of ``values``, with a target band.

    ``values`` is the quantity being binned (|Δh| in metres, duration in
    seconds, …), ``covered`` the per-segment boolean "truth fell inside
    the CI", ``abs_error`` the per-segment |error| in metres. The three
    arrays are parallel. ``unit`` labels the bin ticks, ``xlabel`` the
    axis.
    """
    values = np.asarray(values, dtype=float)
    covered = np.asarray(covered, dtype=bool)
    abs_error = np.asarray(abs_error, dtype=float)

    if values.size == 0:
        _empty(out_path, title, dpi)
        return

    bins = np.digitize(values, bin_edges) - 1
    rows = []
    for b in range(len(bin_edges) - 1):
        mask = bins == b
        if mask.sum() == 0:
            continue
        rows.append({
            "label": f"{bin_edges[b]:.0f}–{bin_edges[b + 1]:.0f} {unit}",
            "n": int(mask.sum()),
            "inclusion": float(np.mean(covered[mask])),
            "mae": float(np.mean(abs_error[mask])),
        })
    if not rows:
        _empty(out_path, title, dpi)
        return
    df_b = pd.DataFrame(rows)

    fig, ax1 = plt.subplots(figsize=(8, 5))
    x = np.arange(len(df_b))
    ax1.bar(x, df_b["inclusion"], color=bar_color, alpha=0.6, label="inclusion")

    # The target is a probability read off a finite number of trials, so
    # it is drawn as a band: the score-test acceptance region at each
    # bin's own n. A bar clearing it is off-target by more than sampling
    # noise explains.
    band_lo, band_hi = target_band(target, df_b["n"].to_numpy(), band_confidence)
    xb = np.concatenate(([x[0] - 0.5], x, [x[-1] + 0.5]))   # span outer bars
    ax1.fill_between(xb, np.r_[band_lo[0], band_lo, band_lo[-1]],
                     np.r_[band_hi[0], band_hi, band_hi[-1]],
                     step="mid", color="b", alpha=0.15,
                     label=f"{target:.0%} target, {band_confidence:.0%} CI on n")
    ax1.axhline(target, color="b", ls=":", alpha=0.8, lw=1.2)

    ax1.set_xticks(x); ax1.set_xticklabels(df_b["label"])
    ax1.set_xlabel(xlabel)
    ax1.set_ylabel("inclusion"); ax1.set_ylim(0, 1.05)
    ax1.grid(True, axis="y", alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(x, df_b["mae"], "r-o", lw=1.5, label="MAE")
    ax2.set_ylabel("MAE (m)")

    for xi, n in zip(x, df_b["n"]):
        ax1.text(xi, 0.03, f"n={n}", ha="center", color="black", fontsize=8)
    ax1.set_title(title)

    # bars fill the axes, so the merged legend goes underneath them
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax1.legend(h1 + h2, l1 + l2, loc="upper center",
               bbox_to_anchor=(0.5, -0.14), ncol=3, frameon=False)

    fig.savefig(out_path, bbox_inches="tight", **({"dpi": dpi} if dpi else {}))
    plt.close(fig)
