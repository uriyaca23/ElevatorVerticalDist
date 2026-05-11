"""Generate the comparison figures consumed by the
"Gravity Calculation Change" section of ``docs/latex/main.tex``.

Numbers are hard-coded from the comparison runs
(``scripts/compare_input_signal.py`` and
``scripts/compare_prediction_input_signal.py``). The pipeline figure
(``gravity_pipeline_compare.png``) is overwritten by the end-to-end
runner ``scripts/compare_pipeline_input_signal.py`` when it produces
its CSV output, but this script also draws an initial placeholder so
the LaTeX builds before that finishes.

Usage:
    venv/bin/python -m scripts.build_gravity_change_figures
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

import matplotlib  # noqa: E402
matplotlib.use("Agg")  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT_DIR = _REPO_ROOT / "docs" / "latex" / "figures" / "gravity_change"
OUT_DIR.mkdir(parents=True, exist_ok=True)

_BLUE = "#3498db"   # a_vert (baseline)
_GREEN = "#27ae60"  # |a|-g (treatment)
_BAR_KW = dict(width=0.38, edgecolor="white", linewidth=1.0)


def _grouped_bar(ax, labels, baseline, treatment, ylabel, title,
                 ymax=None, fmt="{:.2f}", baseline_label="$a_{\\rm vert}$",
                 treatment_label="$|a|-g$"):
    x = np.arange(len(labels))
    ax.bar(x - 0.2, baseline,  color=_BLUE,  label=baseline_label,  **_BAR_KW)
    ax.bar(x + 0.2, treatment, color=_GREEN, label=treatment_label, **_BAR_KW)
    for xi, b, t in zip(x, baseline, treatment):
        ax.text(xi - 0.2, b, fmt.format(b), ha="center", va="bottom",
                fontsize=8, color="#2c3e50")
        ax.text(xi + 0.2, t, fmt.format(t), ha="center", va="bottom",
                fontsize=8, color="#1e6b3a")
    ax.set_xticks(x); ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=11)
    if ymax is not None:
        ax.set_ylim(0, ymax)
    ax.grid(axis="y", alpha=0.3, linestyle="--", linewidth=0.6)
    ax.set_axisbelow(True)
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)


def _segmentation_figure() -> Path:
    """Bar chart: a_vert vs |a|-g detection metrics on 22 TRAIN exps."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    _grouped_bar(
        ax1,
        ["F1", "Precision", "Recall"],
        [0.813, 0.896, 0.745],
        [0.926, 0.981, 0.877],
        ylabel="score",
        title="Segmentation detection metrics (22 TRAIN exps, 415 GT rides)",
        ymax=1.05, fmt="{:.3f}",
    )
    _grouped_bar(
        ax2,
        ["clean matches", "missed", "false positives"],
        [309, 95, 29],
        [364, 42, 0],
        ylabel="count",
        title="Failure-mode counts (lower is better for miss / FP)",
        ymax=max(309, 364) * 1.15, fmt="{:.0f}",
    )
    fig.suptitle("Segmentation matched-filter — gravity-projected vs magnitude residual",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "gravity_segmentation_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _prediction_figure() -> Path:
    """Bar chart: a_vert vs |a|-g prediction errors (TRAIN+TEST)."""
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    # ZUPT — top row
    _grouped_bar(
        axes[0, 0],
        ["MAE", "medAE", "RMSE"],
        [1.18, 0.34, 3.00],
        [2.12, 0.37, 4.43],
        ylabel="metres",
        title="ZUPT — TRAIN (n=382 clean)",
        ymax=5.0,
    )
    _grouped_bar(
        axes[0, 1],
        ["MAE", "medAE", "RMSE"],
        [6.45, 0.19, 43.98],
        [1.73, 0.21, 5.82],
        ylabel="metres",
        title="ZUPT — TEST (n=108)",
        ymax=50.0,
    )
    # Trapezoid — bottom row
    _grouped_bar(
        axes[1, 0],
        ["MAE", "medAE", "RMSE"],
        [0.93, 0.26, 2.59],
        [1.18, 0.26, 2.61],
        ylabel="metres",
        title="Trapezoid pulse-pair — TRAIN (n=382)",
        ymax=3.5,
    )
    _grouped_bar(
        axes[1, 1],
        ["MAE", "medAE", "RMSE"],
        [3.60, 0.29, 25.82],
        [1.25, 0.29, 4.25],
        ylabel="metres",
        title="Trapezoid pulse-pair — TEST (n=108)",
        ymax=30.0,
    )
    fig.suptitle("Prediction error: $a_{\\rm vert}$ vs $|a|-g$ (medAE is identical; "
                 "differences live entirely in the outlier tail)",
                 fontsize=12, y=1.00)
    fig.tight_layout()
    out = OUT_DIR / "gravity_prediction_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _theory_figure() -> Path:
    """Conceptual: a_vert vs |a|-g under a tilt sweep, no user motion."""
    theta = np.linspace(0, np.pi / 2, 200)
    g = 9.81
    a_ride = 1.0  # m/s^2 lobe amplitude (typical elevator take-off)

    # Phone tilted by angle θ from vertical AFTER the calibration window
    # captured gvec when θ = 0.
    # a_vert measured = (a_total · g_hat_old) - g_old_mag
    #                 = (g + a_ride) * cos(θ) - g
    a_vert_recovered = (g + a_ride) * np.cos(theta) - g

    # |a| − g_old:  magnitude is invariant, so always = (g+a_ride) − g = a_ride
    a_mag_recovered = np.full_like(theta, a_ride)

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    deg = np.degrees(theta)
    ax.plot(deg, a_vert_recovered, color=_BLUE, lw=2.0,
            label="$a_{\\rm vert}$ (frozen $\\hat g$)")
    ax.plot(deg, a_mag_recovered, color=_GREEN, lw=2.0,
            label="$|a|-g$ (magnitude residual)")
    ax.axhline(a_ride, color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.text(2, a_ride + 0.05, f"true $a_{{\\rm ride}} = {a_ride:.1f}\\,\\mathrm{{m/s^2}}$",
            fontsize=9, color="#555")
    ax.set_xlabel("Phone rotation $\\theta$ since calibration (degrees)")
    ax.set_ylabel("Recovered lobe amplitude (m/s$^2$)")
    ax.set_title("Sensitivity to in-ride phone rotation\n"
                 "(no user translation, $a_{\\rm ride}=1.0$ m/s$^2$)",
                 fontsize=11)
    ax.set_xlim(0, 90); ax.set_ylim(-10.5, 1.4)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="lower left", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "gravity_theory_tilt.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _theory_horizontal_motion_figure() -> Path:
    """Horizontal user-motion sensitivity (the cost side of the swap)."""
    a_h = np.linspace(0, 3.0, 200)  # horizontal acceleration of phone (m/s^2)
    g = 9.81
    a_ride = 1.0

    # a_vert: under correct projection, a_h is orthogonal → no leak.
    a_vert_recovered = np.full_like(a_h, a_ride)

    # |a|-g: |a| = sqrt((g+a_ride)^2 + a_h^2) → leaks a_h^2 / (2g) at first order.
    a_mag_recovered = np.sqrt((g + a_ride) ** 2 + a_h ** 2) - g

    fig, ax = plt.subplots(figsize=(7.5, 4.0))
    ax.plot(a_h, a_vert_recovered, color=_BLUE, lw=2.0,
            label="$a_{\\rm vert}$ (orthogonal $a_h$ filtered)")
    ax.plot(a_h, a_mag_recovered, color=_GREEN, lw=2.0,
            label="$|a|-g$  (leaks $a_h^2/2g$)")
    ax.axhline(a_ride, color="#555", lw=0.8, ls="--", alpha=0.6)
    ax.text(0.05, a_ride + 0.02, f"true $a_{{\\rm ride}}={a_ride:.1f}$ m/s$^2$",
            fontsize=9, color="#555")
    ax.set_xlabel("Horizontal user-motion amplitude $a_h$ (m/s$^2$)")
    ax.set_ylabel("Recovered lobe amplitude (m/s$^2$)")
    ax.set_title("Sensitivity to horizontal user motion "
                 "(perfect orientation)",
                 fontsize=11)
    ax.set_xlim(0, 3.0); ax.set_ylim(0.95, 1.55)
    ax.grid(alpha=0.3, linestyle="--", linewidth=0.6)
    ax.legend(loc="upper left", fontsize=10)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    out = OUT_DIR / "gravity_theory_horizontal.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _pipeline_figure() -> Path:
    """End-to-end pipeline (segmentation → trapezoid prediction)
    matched-view MAE / RMSE per split, both signal modes.

    Numbers from ``scripts/compare_pipeline_input_signal.py`` run on
    22 train + 4 test experiments, ``trapezoid_accel`` predictor.
    """
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    _grouped_bar(
        axes[0],
        ["TRAIN", "TEST", "TRAIN+TEST"],
        [1.46, 1.49, 1.46],
        [1.70, 0.77, 1.51],
        ylabel="MAE (m)",
        title="End-to-end pipeline — matched-view MAE",
        ymax=2.2,
    )
    _grouped_bar(
        axes[1],
        ["TRAIN", "TEST", "TRAIN+TEST"],
        [6.32, 7.66, 6.67],
        [5.99, 2.31, 5.42],
        ylabel="RMSE (m)",
        title="End-to-end pipeline — matched-view RMSE",
        ymax=9.0,
    )
    fig.suptitle("End-to-end pipeline (segmentation $\\rightarrow$ trapezoid prediction): "
                 "matched-view Δh errors against barometer truth",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "gravity_pipeline_compare.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def _pipeline_all_views_figure() -> Path:
    """Companion figure showing all three views (gt / matched / all)
    in the TEST split where the gap is largest."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
    views = ["GT (oracle)", "matched (clean)", "all (vs baro)"]
    _grouped_bar(
        ax1, views,
        [3.60, 1.49, 1.37],
        [1.25, 0.77, 1.13],
        ylabel="MAE (m)",
        title="TEST set MAE per view",
        ymax=4.2,
    )
    _grouped_bar(
        ax2, views,
        [25.82, 7.66, 3.59],
        [4.25, 2.31, 3.78],
        ylabel="RMSE (m)",
        title="TEST set RMSE per view",
        ymax=30,
    )
    fig.suptitle("All three pipeline views on TEST — the magnitude "
                 "residual collapses the $a_{\\rm vert}$ outlier tail",
                 fontsize=12, y=1.02)
    fig.tight_layout()
    out = OUT_DIR / "gravity_pipeline_test_views.png"
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    return out


def main() -> int:
    paths = [
        _theory_figure(),
        _theory_horizontal_motion_figure(),
        _segmentation_figure(),
        _prediction_figure(),
        _pipeline_figure(),
        _pipeline_all_views_figure(),
    ]
    for p in paths:
        print(f"wrote {p.relative_to(_REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
