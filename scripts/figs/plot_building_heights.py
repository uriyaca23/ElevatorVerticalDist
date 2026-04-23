"""Regenerate ``docs/latex/figures/building_heights.png`` (overview) and
``docs/latex/figures/building_heights_per_building.png`` (per-building
detail) from the central gramushka tables.

One bar per elevator in the overview — Millenium Hotel and
Millenium Outside are listed separately even though they share the
same central gramushka folder, because their runs hit different
floor ranges. The per-building figure has one panel per elevator
showing the elevation vs floor-index profile.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
GRAMUSHKA_ROOT = REPO / "src/data/gramushka"
OUT_DIR = REPO / "docs/latex/figures"

# Display name → (gramushka folder, color, optional "only keep floors at-or-below this" cap)
BUILDINGS: dict[str, tuple[str, str, tuple[float, float] | None]] = {
    "Millenium Hotel\n(internal, exp1+exp2)": ("פרימה מילניום", "#1f77b4", (-1.0, 45.0)),
    "Millenium Outside\n(external, exp3)":    ("פרימה מילניום", "#ff7f0e", (-1.0, 20.0)),
    "Acro (office)":                          ("אקרו נדלן",      "#2ca02c", None),
    "Beit Mansour 1":                         ("בית_מנצור_1",    "#d62728", None),
    "Beit Yitzchaki Raanana\n(test)":         ("בית יצחקי ב",    "#9467bd", None),
    "Bar-Ilan 2 Herzeliya":                   ("בר אילן 2",      "#8c564b", None),
    "Haari 3 (Ramat Gan)":                    ("haari",           "#e377c2", None),
}


def _load_gramushka(folder: str) -> pd.DataFrame:
    path = GRAMUSHKA_ROOT / folder / "gramushka.csv"
    df = pd.read_csv(path)
    df["Elevation (m)"] = (
        df["Elevation (m)"].astype(str)
        .str.replace("±", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["Elevation (m)"] = pd.to_numeric(df["Elevation (m)"], errors="coerce")
    df = df.dropna(subset=["Elevation (m)"]).reset_index(drop=True)
    return df


def _per_building_frame(folder: str, clip: tuple[float, float] | None) -> pd.DataFrame:
    df = _load_gramushka(folder).copy()
    if clip is not None:
        lo, hi = clip
        df = df[(df["Elevation (m)"] >= lo) & (df["Elevation (m)"] <= hi)].reset_index(drop=True)
    return df


def main() -> None:
    # ----- Per-building panel grid -----
    n = len(BUILDINGS)
    cols = 4
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 3 * rows), sharey=False)
    axes = np.atleast_2d(axes).ravel()

    for ax, (name, (folder, colour, clip)) in zip(axes, BUILDINGS.items()):
        df = _per_building_frame(folder, clip)
        if df.empty:
            ax.set_visible(False)
            continue
        y = df["Elevation (m)"].to_numpy()
        x = np.arange(len(y))
        ax.bar(x, y, color=colour, edgecolor="black", linewidth=0.5)
        ax.set_xticks(x)
        ax.set_xticklabels(df["Floor Name"].tolist(), rotation=55, ha="right", fontsize=8)
        ax.set_ylabel("Elevation (m)")
        ax.set_title(name, fontsize=10)
        ax.grid(axis="y", linestyle=":", linewidth=0.5, alpha=0.6)
        # Horizontal line at the max floor reached
        ax.axhline(y.max(), color=colour, linestyle="--", linewidth=0.8, alpha=0.6)
    for ax in axes[n:]:
        ax.set_visible(False)
    fig.suptitle("Per-building floor elevations (from central gramushka tables)", fontsize=11, y=1.01)
    fig.tight_layout()
    per_path = OUT_DIR / "building_heights_per_building.png"
    fig.savefig(per_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {per_path.relative_to(REPO)}")

    # ----- Overview bar chart: total height traversable per elevator -----
    rows_overview = []
    for name, (folder, _colour, clip) in BUILDINGS.items():
        df = _per_building_frame(folder, clip)
        if df.empty:
            continue
        rows_overview.append({
            "name": name.replace("\n", " "),
            "min_m": float(df["Elevation (m)"].min()),
            "max_m": float(df["Elevation (m)"].max()),
            "range_m": float(df["Elevation (m)"].max() - df["Elevation (m)"].min()),
            "n_floors": int(len(df)),
        })
    ov = pd.DataFrame(rows_overview).sort_values("range_m")
    fig, ax = plt.subplots(figsize=(9, 4))
    colours = [BUILDINGS[name.replace("  (", "\n(").replace(") ", ")\n")][1]
               if name in BUILDINGS else "#4c72b0" for name in ov["name"]]
    # Simpler: look up by partial match
    colours = []
    for short in ov["name"]:
        match = None
        for full, (_folder, c, _clip) in BUILDINGS.items():
            if full.replace("\n", " ") == short:
                match = c; break
        colours.append(match or "#4c72b0")
    ax.barh(ov["name"], ov["range_m"], color=colours, edgecolor="black", linewidth=0.5)
    for i, (nf, rg) in enumerate(zip(ov["n_floors"], ov["range_m"])):
        ax.text(rg + 0.5, i, f"{nf} floors, {rg:.1f} m", va="center", fontsize=9)
    ax.set_xlabel("Vertical range of floor-elevation table (m)")
    ax.set_title("Total height range per elevator in the field campaign")
    ax.grid(axis="x", linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_xlim(0, ov["range_m"].max() * 1.25)
    fig.tight_layout()
    ov_path = OUT_DIR / "building_heights.png"
    fig.savefig(ov_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {ov_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
