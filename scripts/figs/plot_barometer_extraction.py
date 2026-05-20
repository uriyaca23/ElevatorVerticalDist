"""Generate ``paper_phd/figures/barometer_extraction.png``.

Two stacked panels on one full recording show how raw barometric pressure
(hPa) becomes a metric altitude trace (m) via the ISA tropospheric
inversion implemented in :mod:`src.physics.barometric`. The figure is
referenced from Appendix~B (Dataset construction) as
``fig:app-barometer-extraction``.

Defaults pick a Millennium-Hotel session: the recording contains several
elevator rides interleaved with walking and stair use, so the converted
altitude trace shows both the floor-to-floor step pattern the detector
relies on and the noise/flat regions between rides.
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.physics.barometric import P0_HPA, pressure_to_altitude

REPO = Path(__file__).resolve().parents[2]
OUT_DIR = REPO / "paper_phd/figures"
DEFAULT_EXP = "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp1"


def _load_prs(exp: str) -> pd.DataFrame:
    csv = REPO / "src/data/structuredData/data" / exp / "PRS.csv"
    df = pd.read_csv(csv)
    return df


def _load_temperature_c(exp: str) -> float | None:
    meta = REPO / "src/data/structuredData/data" / exp / "metadata.csv"
    if not meta.exists():
        return None
    row = pd.read_csv(meta).iloc[0]
    return float(row["temperature_c"]) if "temperature_c" in row else None


def main(exp: str = DEFAULT_EXP) -> None:
    prs = _load_prs(exp)
    # PRS schema: timestamp_ms, pressure (hPa) — see src/data/loader/constants.py
    t = (prs["timestamp_ms"].to_numpy() - prs["timestamp_ms"].iloc[0]) / 1000.0
    p = prs["pressure"].to_numpy()
    t_c = _load_temperature_c(exp)
    h = pressure_to_altitude(p, p0_hpa=P0_HPA, temperature_c=t_c)
    h = h - np.median(h[: max(50, len(h) // 50)])  # anchor near recording start

    fig, axes = plt.subplots(2, 1, figsize=(9, 4.6), sharex=True)

    ax = axes[0]
    ax.plot(t, p, color="#1f77b4", linewidth=0.9)
    ax.set_ylabel("Pressure (hPa)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_title(
        "Raw barometer pressure (top) and ISA-inverted altitude (bottom)",
        fontsize=10,
    )
    ax.text(
        0.985,
        0.92,
        r"$h \,=\, (T_0/L)\,[\,1 - (P/P_0)^{1/5.255}\,]$",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="0.6", lw=0.6),
    )

    ax = axes[1]
    ax.plot(t, h, color="#2ca02c", linewidth=0.9)
    ax.set_ylabel("Altitude (m)")
    ax.set_xlabel("Time since recording start (s)")
    ax.grid(True, linestyle=":", linewidth=0.5, alpha=0.6)

    fig.tight_layout()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUT_DIR / "barometer_extraction.png"
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)} (exp={exp}, T0={t_c} °C)")


if __name__ == "__main__":
    import sys

    main(sys.argv[1] if len(sys.argv) > 1 else DEFAULT_EXP)
