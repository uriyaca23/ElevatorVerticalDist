"""Generate ``paper_phd/figures/gramushka_snap.png``.

A worked-data illustration of the gramushka snap step described in
Appendix~B.3 (Buildings and phones). Two stacked panels on one
Bar-Ilan~2 recording window:

    Top  — barometer-derived altitude with the detected ride
           intervals shaded (the output of \\texttt{HeightSegmenter},
           same trace family as Figure~\\ref{fig:app-edge-detection}).

    Bottom — a zoom on one ride's endpoint, with the building's
             gramushka floor lines overlaid and an annotated red
             arrow from the raw barometer altitude to the snapped
             floor elevation.

The figure is cited from the gramushka paragraph alongside
Figure~\\ref{fig:app-gramushka-pair}.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physics.barometric import P0_HPA, pressure_to_altitude
from src.segmentation.algorithms.barometer_only.height_segmentation import (
    HeightSegmenter,
)
from src.segmentation.algorithms.configTypes import PressureFilterConfig

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper_phd/figures"
EXP = "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026"
BUILDING_FOLDER = "בר אילן 2"
# Zoom window picked to show 3–4 consecutive rides on multiple floors.
WIN = (60.0, 280.0)


def _load_altitude(exp: str) -> tuple[np.ndarray, np.ndarray, float | None]:
    prs = pd.read_csv(REPO / "src/data/structuredData/data" / exp / "PRS.csv")
    meta = pd.read_csv(REPO / "src/data/structuredData/data" / exp / "metadata.csv")
    t_c = float(meta.iloc[0]["temperature_c"])
    t = (prs["timestamp_ms"].to_numpy() - prs["timestamp_ms"].iloc[0]) / 1000.0
    h = pressure_to_altitude(prs["pressure"].to_numpy(), p0_hpa=P0_HPA, temperature_c=t_c)
    return t, h, t_c


def _load_gramushka(folder: str) -> pd.DataFrame:
    df = pd.read_csv(REPO / "src/data/gramushka" / folder / "gramushka.csv")
    df["Elevation (m)"] = (
        df["Elevation (m)"]
        .astype(str)
        .str.replace("±", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["Elevation (m)"] = pd.to_numeric(df["Elevation (m)"], errors="coerce")
    return df.dropna(subset=["Elevation (m)"]).reset_index(drop=True)


def main() -> None:
    t, h, _ = _load_altitude(EXP)
    frame = pd.DataFrame({"time": t, "height": h})

    cfg = PressureFilterConfig()
    seg = HeightSegmenter(cfg)
    z_lp = seg.filter_height(frame)
    rides = seg.segment(frame)

    # Anchor altitude on the median of the first 10 s so the trace shows
    # building-absolute height (the same anchor the snap step uses
    # after reading the session's start floor from metadata).
    anchor = float(np.median(z_lp[t < 10.0]))
    z_lp_shifted = z_lp - anchor

    gram = _load_gramushka(BUILDING_FOLDER)
    floors = gram["Elevation (m)"].to_numpy()
    floor_names = gram["Floor Name"].tolist()

    t0, t1 = WIN
    mask = (t >= t0) & (t <= t1)

    fig, axes = plt.subplots(2, 1, figsize=(8, 5.6), sharex=False,
                             gridspec_kw=dict(height_ratios=[1.0, 1.4]))

    # ---- TOP: detected rides on the barometer altitude.
    ax = axes[0]
    ax.plot(t[mask], z_lp_shifted[mask], color="#2ca02c", linewidth=1.1)
    drew: set[str] = set()
    colours = {"up": "#2ca02c", "down": "#d62728"}
    for _, r in rides.iterrows():
        s = r["start_ci"][0]
        e = r["end_ci"][1]
        if e < t0 or s > t1:
            continue
        label = r["type"] if r["type"] not in drew else None
        drew.add(r["type"])
        ax.axvspan(max(s, t0), min(e, t1), color=colours[r["type"]], alpha=0.20, label=label)
    ax.set_ylabel("Altitude (m)")
    ax.set_title(
        "Snapping detected rides to the gramushka floor table (Bar-Ilan~2)",
        fontsize=10,
    )
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.legend(loc="upper left", fontsize=8, frameon=True, framealpha=0.9)
    ax.set_xlim(t0, t1)

    # Mark which ride endpoint we zoom into in the bottom panel.
    rides_in_window = [
        r for _, r in rides.iterrows()
        if t0 <= r["end_ci"][1] <= t1
    ]
    # Pick a ride whose endpoint sits a clear distance from the nearest
    # floor, so the snap arrow is unambiguous.
    pick = None
    best_gap = -1.0
    for r in rides_in_window:
        idx = int(np.searchsorted(t, r["end_ci"][1], side="right") - 1)
        raw_z = float(z_lp_shifted[idx])
        gap = float(np.min(np.abs(floors - raw_z)))
        if gap > best_gap:
            best_gap = gap
            pick = (r, idx, raw_z)
    if pick is None:
        raise RuntimeError("no rides in window to draw snap arrow on")
    r_pick, idx_pick, raw_z = pick
    e_t = float(r_pick["end_ci"][1])
    nearest_idx = int(np.argmin(np.abs(floors - raw_z)))
    snap_z = float(floors[nearest_idx])
    snap_name = floor_names[nearest_idx]
    snap_delta = raw_z - snap_z  # positive: raw above floor

    # Highlight the picked endpoint on the top panel.
    axes[0].plot([e_t], [raw_z], marker="o", color="#b22222",
                 markersize=6, zorder=10)
    axes[0].annotate("zoom (see below)", xy=(e_t, raw_z),
                     xytext=(e_t + 4, raw_z - 7),
                     fontsize=8, color="#b22222",
                     arrowprops=dict(arrowstyle="-", color="#b22222",
                                     lw=0.6, alpha=0.7))

    # ---- BOTTOM: zoomed snap.
    ax = axes[1]
    # Zoom window around the picked endpoint.
    z_t0 = e_t - 6.0
    z_t1 = e_t + 6.0
    zoom_mask = (t >= z_t0) & (t <= z_t1)
    ax.plot(t[zoom_mask], z_lp_shifted[zoom_mask],
            color="#2ca02c", linewidth=1.4, zorder=3)

    # Floor lines visible near the endpoint (a few above and below).
    yc = raw_z
    visible_lo = yc - 5.5
    visible_hi = yc + 5.5
    visible_floors = [
        (name, elev)
        for name, elev in zip(floor_names, floors)
        if visible_lo <= elev <= visible_hi
    ]
    for name, elev in visible_floors:
        col = "#1f77b4" if elev == snap_z else "#888"
        lw = 1.1 if elev == snap_z else 0.6
        alpha = 0.95 if elev == snap_z else 0.7
        ax.axhline(elev, color=col, linestyle="--", linewidth=lw,
                   alpha=alpha, zorder=1)
        ax.text(z_t1 + 0.3, elev,
                f"{name}  ({elev:+.2f} m)" if elev == snap_z else name,
                va="center", ha="left", fontsize=8,
                color=col, fontweight="bold" if elev == snap_z else "normal")

    # Raw endpoint marker + snap arrow.
    ax.plot([e_t], [raw_z], marker="o", color="#b22222",
            markersize=7, zorder=5)
    ax.annotate(
        "",
        xy=(e_t, snap_z),
        xytext=(e_t, raw_z),
        arrowprops=dict(arrowstyle="-|>", color="#b22222",
                        lw=1.8, shrinkA=0, shrinkB=0,
                        mutation_scale=14),
        zorder=6,
    )
    # Label the snap distance.
    mid_z = 0.5 * (raw_z + snap_z)
    ax.text(e_t + 0.4, mid_z,
            f"snap {snap_delta:+.2f} m",
            ha="left", va="center", fontsize=8, color="#b22222",
            bbox=dict(boxstyle="round,pad=0.25", fc="#fff5f5",
                      ec="#b22222", lw=0.6))
    # Label the raw endpoint altitude.
    ax.text(e_t - 0.4, raw_z + 0.45,
            f"raw {raw_z:+.2f} m",
            ha="right", va="bottom", fontsize=8, color="#444")

    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Time since recording start (s)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.4)
    ax.set_xlim(z_t0, z_t1)
    ax.set_ylim(visible_lo, visible_hi)
    ax.set_title(
        f"Snap of one ride endpoint to {snap_name}",
        fontsize=9.5, pad=4,
    )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gramushka_snap.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path.relative_to(REPO)} (exp={EXP}, window={WIN}, "
        f"zoomed_endpoint=t{e_t:.1f}s raw={raw_z:+.2f}m snapped_to={snap_name} "
        f"at {snap_z:+.2f}m, snap={snap_delta:+.2f}m)"
    )


if __name__ == "__main__":
    main()
