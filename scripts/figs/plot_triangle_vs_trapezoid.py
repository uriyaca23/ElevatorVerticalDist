"""Visualise every triangular-velocity ride (joined-pulse mode) the
trapezoid estimator fitted, plus a handful of matched trapezoid-mode
rides for contrast. Dumps one PNG per segment to
``docs/examples/triangles/`` and, separately, two curated showcase
figures to ``docs/latex/figures/`` that are embedded in the paper.

For every segment we re-run ``predict_segment`` to recover the
fitted template and residuals (these aren't persisted to the CSV),
then draw:

* vertical acceleration trace (smoothed)
* fitted template overlaid in colour (joined or pair)
* shaded lobe supports so the touching / separated geometry is
  visible at a glance
* header annotations: mode, R², W, Delta_tc, predicted vs true Δh,
  accepted/rejected flag

The "joined" panels are what you sanity-check against to decide
whether the extension from \S\ref{sec:algo-joined} fires on the
right segments.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
TRIANGLES_OUT = REPO / "docs" / "examples" / "triangles"
TRAPEZOIDS_OUT = REPO / "docs" / "examples" / "trapezoids"
PAPER_OUT = REPO / "docs" / "latex" / "figures"


def _fmt_dh(x):
    try:
        return f"{x:+.2f} m"
    except Exception:
        return "?"


def _plot_segment(ax, t, a_smooth, a_template, fit_meta, record, out):
    mode = fit_meta.get("mode", "")
    params = fit_meta.get("params", {})
    W = params.get("W", float("nan"))
    f = params.get("f", float("nan"))
    A = params.get("A_used", params.get("A_fit", float("nan")))
    t_c1 = params.get("t_c1", 0.0)
    t_c2 = params.get("t_c2", 0.0)
    joint_r2 = params.get("joint_r2", float("nan"))

    ax.plot(t, a_smooth, color="black", linewidth=1.2, label="smoothed $a_\\mathrm{vert}$")
    ax.plot(t, a_template, color="#d62728" if mode == "joined" else "#1f77b4",
            linewidth=2.0, alpha=0.8,
            label=f"{mode} fit  ($R^2={joint_r2:.2f}$)")
    # Shade lobe support(s)
    for t_c, col in [(t_c1, "#ff7f0e"), (t_c2, "#2ca02c")]:
        ax.axvspan(t_c - W, t_c + W, color=col, alpha=0.10)
        ax.axvline(t_c, color=col, linestyle=":", linewidth=0.8, alpha=0.6)

    # Touching marker: if |Δt_c - 2W| is small, mark it
    dtc = abs(t_c2 - t_c1)
    touching = dtc <= 2.1 * W if W > 0 else False
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.6)

    title = (
        f"[{mode}] {record['exp_name']}  seg_idx={record['seg_idx']}  "
        f"true={_fmt_dh(record['true_dh'])}  pred={_fmt_dh(out.height_diff)}  "
        f"accepted={out.accepted}\n"
        f"W={W:.2f}s  f={f:.2f}  A={A:.2f}  "
        f"$\\Delta t_c$={dtc:.2f}s  2W={2*W:.2f}s  "
        f"{'(touching)' if touching else '(separated)'}  |  CI=±{out.ci_half_width:.2f}m"
    )
    ax.set_title(title, fontsize=9)
    ax.set_xlabel("t (s, ride-local)")
    ax.set_ylabel("$a_\\mathrm{vert}$ (m/s$^2$)")
    ax.legend(fontsize=8, loc="upper right")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--max", type=int, default=None, help="cap how many segments to plot per mode")
    ap.add_argument("--paper-only", action="store_true",
                    help="only produce the two curated paper figures, skip per-segment dump")
    args = ap.parse_args()

    TRIANGLES_OUT.mkdir(parents=True, exist_ok=True)
    TRAPEZOIDS_OUT.mkdir(parents=True, exist_ok=True)
    PAPER_OUT.mkdir(parents=True, exist_ok=True)

    from src.prediction.evaluation.dataset import load_all_segments
    from src.prediction.algorithms.configTypes import TrapezoidAccelConfig
    from src.prediction.algorithms.accelerometer_only.trapezoid_accel.estimator import TrapezoidAccelEstimator

    print("loading segments ...")
    segs = load_all_segments()
    est = TrapezoidAccelEstimator(TrapezoidAccelConfig())

    n_tri = 0; n_trap = 0
    best_triangle = None; best_trapezoid = None   # best R² in each mode, for the paper
    for rec in segs:
        out = est.predict_segment(rec.acc, phone_model=rec.phone,
                                   pre=rec.pre_acc, post=rec.post_acc)
        meta = out.meta or {}
        mode = meta.get("mode", "")
        if mode not in ("pair", "joined"):
            continue
        t = meta.get("t_sec")
        a_smooth = meta.get("a_smooth")
        a_template = meta.get("a_template")
        if t is None or a_smooth is None or a_template is None:
            continue
        record = {
            "exp_name": rec.exp_name,
            "seg_idx": rec.seg_idx,
            "true_dh": rec.true_dh,
        }
        # Pick best-R² examples for the paper
        r2 = meta.get("params", {}).get("joint_r2", 0.0)
        if mode == "joined" and (best_triangle is None or r2 > best_triangle[0]):
            best_triangle = (r2, t.copy(), a_smooth.copy(), a_template.copy(), dict(meta), dict(record), out)
        if mode == "pair" and (best_trapezoid is None or r2 > best_trapezoid[0]):
            # prefer one with a visible cruise (Δt_c clearly > 2W)
            W = meta.get("params", {}).get("W", 1.0)
            t_c1 = meta.get("params", {}).get("t_c1", 0.0)
            t_c2 = meta.get("params", {}).get("t_c2", 0.0)
            if (t_c2 - t_c1) > 2.5 * W:
                best_trapezoid = (r2, t.copy(), a_smooth.copy(), a_template.copy(), dict(meta), dict(record), out)

        if args.paper_only:
            continue

        # Dump per-segment plot
        out_dir = TRIANGLES_OUT if mode == "joined" else TRAPEZOIDS_OUT
        fn = f"{rec.exp_name}__seg{rec.seg_idx:03d}.png"
        fn = fn.replace("/", "_")
        fig, ax = plt.subplots(figsize=(10, 4))
        _plot_segment(ax, t, a_smooth, a_template, meta, record, out)
        fig.tight_layout()
        fig.savefig(out_dir / fn, dpi=110, bbox_inches="tight")
        plt.close(fig)
        if mode == "joined":
            n_tri += 1
            if args.max and n_tri >= args.max:
                pass  # keep counting to find best example
        else:
            n_trap += 1

    if not args.paper_only:
        print(f"wrote {n_tri} triangle PNGs to {TRIANGLES_OUT.relative_to(REPO)}")
        print(f"wrote {n_trap} trapezoid PNGs to {TRAPEZOIDS_OUT.relative_to(REPO)}")

    # ---- Curated showcase figure for the paper ----
    if best_triangle is None or best_trapezoid is None:
        print("WARNING: could not find both a triangle and a trapezoid exemplar; skipping paper figure.")
        return

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.5))
    _plot_segment(
        axes[0],
        best_triangle[1], best_triangle[2], best_triangle[3],
        best_triangle[4], best_triangle[5], best_triangle[6],
    )
    axes[0].set_title("Triangle-velocity ride  —  joined-pulse fit", fontsize=11)
    _plot_segment(
        axes[1],
        best_trapezoid[1], best_trapezoid[2], best_trapezoid[3],
        best_trapezoid[4], best_trapezoid[5], best_trapezoid[6],
    )
    axes[1].set_title("Trapezoid-velocity ride  —  pair fit", fontsize=11)
    fig.suptitle(
        "Two regimes of the matched-filter fit. Left: short ride, "
        "lobes touch ($\\Delta t_c \\approx 2W$), velocity profile "
        "is triangular. Right: long ride, lobes separated by a cruise "
        "window, velocity profile is trapezoidal.",
        fontsize=10, y=1.03,
    )
    fig.tight_layout()
    out_path = PAPER_OUT / "triangle_vs_trapezoid.png"
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote paper figure to {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
