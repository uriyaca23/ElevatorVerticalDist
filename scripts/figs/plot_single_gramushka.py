"""Generate ``paper_phd/figures/gramushka_example.png``.

One building's gramushka floor table rendered as a vertical stack of
labelled floors, with an arrow annotating a single floor's surveyed
elevation. The figure is referenced from Appendix~B (Dataset
construction) as ``fig:app-gramushka-example`` and serves as the worked
example of how a gramushka becomes a floor-to-height lookup.

Defaults pick Bar-Ilan~2 (Herzeliya) — the building already named on
the appendix's gramushka drawing — and annotates Floor~10 at +42.80 m.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper_phd/figures"
GRAMUSHKA_ROOT = REPO / "src/data/gramushka"

DEFAULT_BUILDING = "בר אילן 2"
DEFAULT_DISPLAY_NAME = "Bar-Ilan 2 (Herzeliya)"
DEFAULT_ANNOTATED_FLOOR = "Floor 10"


def _load(folder: str) -> pd.DataFrame:
    df = pd.read_csv(GRAMUSHKA_ROOT / folder / "gramushka.csv")
    df["Elevation (m)"] = (
        df["Elevation (m)"]
        .astype(str)
        .str.replace("±", "", regex=False)
        .str.replace("+", "", regex=False)
        .str.strip()
    )
    df["Elevation (m)"] = pd.to_numeric(df["Elevation (m)"], errors="coerce")
    return df.dropna(subset=["Elevation (m)"]).reset_index(drop=True)


def main(
    folder: str = DEFAULT_BUILDING,
    display_name: str = DEFAULT_DISPLAY_NAME,
    annotate: str = DEFAULT_ANNOTATED_FLOOR,
) -> None:
    df = _load(folder)
    y = df["Elevation (m)"].to_numpy()

    fig, ax = plt.subplots(figsize=(4.8, 7.2))
    # One horizontal line per surveyed floor.
    ax.hlines(y, xmin=0.02, xmax=0.55, color="#4c72b0", linewidth=1.4)
    # Right-side floor labels with minimum-separation y-pushing.
    # When two floors sit closer than ``min_sep`` m apart in display
    # space, push later labels down and draw a short leader line from
    # the floor's true elevation to the label.
    y_range = float(y.max() - y.min())
    min_sep = max(1.6, 0.025 * y_range)  # tuned: ~1.6 m at minimum
    names = df["Floor Name"].tolist()
    order = sorted(range(len(y)), key=lambda i: -y[i])  # top to bottom
    label_y: dict[int, float] = {}
    last = None
    for i in order:
        target = float(y[i])
        if last is None or last - target >= min_sep:
            label_y[i] = target
        else:
            label_y[i] = last - min_sep
        last = label_y[i]
    for i in range(len(y)):
        ly = label_y[i]
        true_y = float(y[i])
        ax.text(0.62, ly, names[i], va="center", ha="left",
                fontsize=8, color="#333")
        if abs(ly - true_y) > 1e-3:
            ax.plot([0.55, 0.61], [true_y, ly],
                    color="#999", linewidth=0.5, solid_capstyle="round")
    # Left axis tick marks at every elevation.
    ax.set_yticks(y)
    ax.set_yticklabels([f"{v:+.2f}" for v in y], fontsize=8)
    ax.set_ylabel("Surveyed elevation (m)")
    ax.set_xticks([])
    ax.set_xlim(0, 1.8)
    ax.set_ylim(y.min() - 3, y.max() + 3)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["bottom"].set_visible(False)
    ax.set_title(
        f"Gramushka floor table — {display_name}", fontsize=10, pad=8
    )

    # Annotate one floor with an arrow.
    if annotate in df["Floor Name"].values:
        elev = float(df.loc[df["Floor Name"] == annotate, "Elevation (m)"].iloc[0])
        ax.annotate(
            f"{annotate}\n→  +{elev:.2f} m",
            xy=(0.30, elev),
            xytext=(1.35, elev + 1.5),
            fontsize=10,
            color="#b22222",
            ha="left",
            arrowprops=dict(
                arrowstyle="->",
                color="#b22222",
                linewidth=1.4,
                shrinkA=0,
                shrinkB=2,
            ),
            bbox=dict(boxstyle="round,pad=0.3", fc="#fff5f5", ec="#b22222", lw=0.8),
        )

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "gramushka_example.png"
    fig.savefig(out_path, dpi=170, bbox_inches="tight")
    plt.close(fig)
    print(
        f"wrote {out_path.relative_to(REPO)} "
        f"(building={display_name}, annotated={annotate})"
    )


if __name__ == "__main__":
    main()
