"""Per-shape parameter cluster plots (separate trapezoid and parabola figures).

Outputs:
    run_results/param_clusters_trapezoid.png
    run_results/param_clusters_parabola.png

Each figure has a top description of the parameters, a 3D scatter over the
three independent shape parameters, and three 2D projections.

Run:
    python3 -m src.tests.segmentations.param_clusters
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (registers 3d projection)

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.algorithms.segmentation_algorithms.template_match.scripts.strategy_search import load_rides


OUT_DIR = Path(__file__).resolve().parents[1] / "results"

TRAP_DESC = (
    "Trapezoid model:  v(t) = clip( a_max · min(t−t_start, t_end−t), 0, v_max )\n"
    "Independent params:\n"
    "   • a_max  — max acceleration (slope of ramps, m/s²)\n"
    "   • v_max  — plateau / cap velocity (m/s)\n"
    "   • W       — full ride duration t_end − t_start (s)\n"
    "Derived:    plateau = W − 2·v_max/a_max  (0 ⇒ triangle, >0 ⇒ true trapezoid)"
)

PAR_DESC = (
    "Generalized parabola model:  v(t) = v_peak · max(0, 1 − ((t−t_c)/W)²)^p\n"
    "Independent params:\n"
    "   • v_peak — peak velocity (m/s)\n"
    "   • W       — full width from root to root (s)\n"
    "   • p       — shape exponent  (p=1 ⇒ pure parabola; p<1 ⇒ flatter top; p>1 ⇒ sharper peak)"
)


def collect(rides, labels):
    trap = {"a_max": [], "v_max": [], "W": [], "plateau": [], "lbl": []}
    par = {"v_peak": [], "W": [], "p": [], "a_max": [], "lbl": []}
    for r in rides:
        lbl = labels[r["key"]]
        if r["trap"].get("ok") and lbl == "trapezoid":
            t = r["trap"]; W = t["t_end"] - t["t_start"]
            trap["a_max"].append(t["a_max"]); trap["v_max"].append(abs(t["v_max"]))
            trap["W"].append(W); trap["plateau"].append(max(W - 2 * t["v_max"] / t["a_max"], 0.0))
            trap["lbl"].append(lbl)
        if r["par"].get("ok") and lbl == "parabola":
            p = r["par"]
            par["v_peak"].append(abs(p["v_peak"])); par["W"].append(p["W"])
            par["p"].append(p["p"]); par["a_max"].append(p["a_max"])
            par["lbl"].append(lbl)
    for d in (trap, par):
        for k, v in list(d.items()):
            if k != "lbl":
                d[k] = np.array(v)
    return trap, par


def render_trapezoid(trap: dict) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.text(0.02, 0.93, TRAP_DESC, fontsize=10, family="monospace",
             va="top", ha="left",
             bbox=dict(facecolor="#fff7e6", edgecolor="#888", boxstyle="round,pad=0.6"))

    # 3D scatter of (a_max, v_max, W)
    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(trap["a_max"], trap["v_max"], trap["W"],
                 c=trap["plateau"], cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
    ax3d.set_xlabel("a_max (m/s²)"); ax3d.set_ylabel("v_max (m/s)"); ax3d.set_zlabel("W (s)")
    ax3d.set_title("3D: a_max × v_max × W  (color=plateau duration)")

    # 2D projections
    ax = fig.add_subplot(2, 2, 2)
    sc = ax.scatter(trap["a_max"], trap["v_max"], c=trap["plateau"],
                    cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("a_max (m/s²)"); ax.set_ylabel("v_max (m/s)")
    ax.set_title("a_max vs v_max (color=plateau)")
    ax.grid(True, alpha=0.3); plt.colorbar(sc, ax=ax, label="plateau (s)")

    ax = fig.add_subplot(2, 2, 3)
    sc = ax.scatter(trap["W"], trap["v_max"], c=trap["plateau"],
                    cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("W (s)"); ax.set_ylabel("v_max (m/s)")
    ax.set_title("v_max vs W (color=plateau)")
    ax.grid(True, alpha=0.3); plt.colorbar(sc, ax=ax, label="plateau (s)")

    ax = fig.add_subplot(2, 2, 4)
    ax.scatter(trap["W"], trap["plateau"], c=trap["v_max"],
               cmap="plasma", s=45, edgecolor="k", linewidth=0.4)
    ax.plot([0, max(trap["W"])], [0, max(trap["W"])], "k:", lw=0.5, label="plateau=W")
    ax.set_xlabel("W (s)"); ax.set_ylabel("plateau (s)")
    ax.set_title("plateau vs W (color=v_max)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)

    fig.suptitle(f"Trapezoid parameter clusters — {len(trap['a_max'])} rides labeled trapezoid",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = OUT_DIR / "param_clusters_trapezoid.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def render_parabola(par: dict) -> None:
    fig = plt.figure(figsize=(15, 10))
    fig.text(0.02, 0.93, PAR_DESC, fontsize=10, family="monospace",
             va="top", ha="left",
             bbox=dict(facecolor="#e6f0ff", edgecolor="#888", boxstyle="round,pad=0.6"))

    ax3d = fig.add_subplot(2, 2, 1, projection="3d")
    ax3d.scatter(par["v_peak"], par["W"], par["p"],
                 c=par["a_max"], cmap="plasma", s=45, edgecolor="k", linewidth=0.4)
    ax3d.set_xlabel("v_peak (m/s)"); ax3d.set_ylabel("W (s)"); ax3d.set_zlabel("p (shape)")
    ax3d.set_title("3D: v_peak × W × p  (color=a_max)")

    ax = fig.add_subplot(2, 2, 2)
    sc = ax.scatter(par["v_peak"], par["W"], c=par["p"],
                    cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
    ax.set_xlabel("v_peak (m/s)"); ax.set_ylabel("W (s)")
    ax.set_title("W vs v_peak (color=p)")
    ax.grid(True, alpha=0.3); plt.colorbar(sc, ax=ax, label="p")

    ax = fig.add_subplot(2, 2, 3)
    sc = ax.scatter(par["W"], par["p"], c=par["v_peak"],
                    cmap="plasma", s=45, edgecolor="k", linewidth=0.4)
    ax.axhline(1.0, color="k", ls=":", lw=0.6, label="p=1 (pure parabola)")
    ax.set_xlabel("W (s)"); ax.set_ylabel("p (shape exponent)")
    ax.set_title("p vs W (color=v_peak)")
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8); plt.colorbar(sc, ax=ax, label="v_peak (m/s)")

    ax = fig.add_subplot(2, 2, 4)
    sc = ax.scatter(par["v_peak"], par["p"], c=par["W"],
                    cmap="viridis", s=45, edgecolor="k", linewidth=0.4)
    ax.axhline(1.0, color="k", ls=":", lw=0.6)
    ax.set_xlabel("v_peak (m/s)"); ax.set_ylabel("p (shape exponent)")
    ax.set_title("p vs v_peak (color=W)")
    ax.grid(True, alpha=0.3); plt.colorbar(sc, ax=ax, label="W (s)")

    fig.suptitle(f"Parabola parameter clusters — {len(par['v_peak'])} rides labeled parabola",
                 fontsize=13, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.88])
    out = OUT_DIR / "param_clusters_parabola.png"
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"Saved {out}")


def main() -> None:
    rides, labels = load_rides()
    trap, par = collect(rides, labels)
    render_trapezoid(trap)
    render_parabola(par)

    def stats(name, d, keys):
        print(f"\n{name} ({len(d[keys[0]])} rides):")
        for k in keys:
            a = d[k]
            print(f"  {k:10s}  mean={a.mean():7.3f}  std={a.std():7.3f}  "
                  f"min={a.min():7.3f}  max={a.max():7.3f}")

    stats("Trapezoid", trap, ["a_max", "v_max", "W", "plateau"])
    stats("Parabola", par, ["v_peak", "W", "p", "a_max"])


if __name__ == "__main__":
    main()
