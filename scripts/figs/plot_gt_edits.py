"""Visualise the worst GT corrections from ``audit_gt_spikes``.

For each edit in ``src/data/structuredData/gt_edits.csv`` we plot the
segment's barometer trace (the sensor the GT is derived from), the
stored-old GT height-diff, and the cross-phone consensus height-diff
the editor chose as replacement. Side-by-side this makes the "stupid
spike in the sensor used to derive the GT" argument visible.

We plot the top-N edits by magnitude of the old→new delta so the
paper figure shows the most egregious cases first.

Output:
* ``docs/latex/figures/gt_edits.png`` — 4-panel collage.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

EDITS = REPO / "src/data/structuredData/gt_edits.csv"
DATA_ROOT = REPO / "src/data/structuredData/data"
PAPER_OUT = REPO / "docs/latex/figures"


def _pressure_to_altitude(p_hpa):
    # Barometric formula — good to ~0.1 m over small windows.
    p0 = 1013.25
    return 44330.0 * (1.0 - (p_hpa / p0) ** 0.1903)


def _plot_one(ax, edit_row, rank: int):
    exp_name = edit_row["exp_name"]
    prs_path = DATA_ROOT / exp_name / "PRS.csv"
    if not prs_path.exists():
        ax.set_axis_off()
        ax.set_title(f"#{rank}: no PRS data")
        return
    prs = pd.read_csv(prs_path)
    # Filter to a window around the segment ± 5s
    t_lo = int(edit_row["start_ms"]) - 5000
    t_hi = int(edit_row["end_ms"]) + 5000
    sub = prs[(prs["timestamp_ms"] >= t_lo) & (prs["timestamp_ms"] <= t_hi)].copy()
    if sub.empty:
        ax.set_axis_off()
        ax.set_title(f"#{rank}: no PRS samples in window")
        return
    sub["height_m"] = _pressure_to_altitude(sub["pressure"].to_numpy())
    t_rel = (sub["timestamp_ms"].to_numpy() - int(edit_row["start_ms"])) / 1000.0
    h = sub["height_m"].to_numpy() - sub["height_m"].iloc[0]

    ax.plot(t_rel, h, color="black", linewidth=1.0, label="barometer $h(t)$")
    seg_dur = (int(edit_row["end_ms"]) - int(edit_row["start_ms"])) / 1000.0
    ax.axvspan(0, seg_dur, color="#ffe0b3", alpha=0.4, label="GT interval")

    old = float(edit_row["old"])
    new = float(edit_row["new"])
    ax.axhline(old, color="#d62728", linestyle="--", linewidth=1.3,
               label=f"stored GT Δh = {old:+.2f}m (spike)")
    ax.axhline(new, color="#2ca02c", linestyle="--", linewidth=1.3,
               label=f"4-phone consensus Δh = {new:+.2f}m")

    ax.set_xlabel("t (s, segment-local)", fontsize=8)
    ax.set_ylabel("Δh (m)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="best")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)
    phone = exp_name.split("_")[2]
    title = (
        f"#{rank}: {exp_name.split('_')[1]} / {phone} / "
        f"seg start={int(edit_row['start_ms']) % 1000000:06d}ms\n"
        f"|old - new| = {abs(old-new):.2f}m  |  "
        f"cross-phone std = {edit_row['cross_phone_std']:.2f}m  |  "
        f"n_agree = {int(edit_row['n_phones_accepted'])}/4"
    )
    ax.set_title(title, fontsize=8)


def main():
    if not EDITS.exists():
        print(f"ERROR: {EDITS} not found. Run scripts/audit_gt_spikes.py --apply first.")
        return
    df = pd.read_csv(EDITS)
    # Keep only real edits, not book-keeping rows
    df = df[df["status"] == "edited"].copy()
    df["delta"] = (df["old"] - df["new"]).abs()
    # Dedupe: 4 phones × segment — keep one row per segment (largest delta)
    df["key"] = df["start_ms"].astype(str) + "_" + df["end_ms"].astype(str)
    df = df.sort_values("delta", ascending=False).drop_duplicates("key")
    top = df.head(4)

    PAPER_OUT.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(2, 2, figsize=(15, 8))
    axes = axes.ravel()
    for i, (_, row) in enumerate(top.iterrows()):
        _plot_one(axes[i], row, rank=i + 1)
    for j in range(len(top), 4):
        axes[j].set_axis_off()
    fig.suptitle(
        "GT corrections driven by cross-phone accelerometer consensus",
        fontsize=11, y=1.00,
    )
    fig.tight_layout()
    out_path = PAPER_OUT / "gt_edits.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}")

    # Summary table: all unique edits with old/new
    top_all = df.copy().sort_values("delta", ascending=False).reset_index(drop=True)
    top_all[["exp_name", "start_ms", "end_ms", "type", "old", "new",
             "cross_phone_std", "n_phones_accepted"]]\
        .to_csv(PAPER_OUT / "gt_edits_summary.csv", index=False)
    print(f"wrote summary to {(PAPER_OUT / 'gt_edits_summary.csv').relative_to(REPO)}")


if __name__ == "__main__":
    main()
