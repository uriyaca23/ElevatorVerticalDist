"""Visualise the top-N worst accepted predictions on train and test.

For each worst segment we re-run ``predict_segment`` so we can overlay
the fitted template, and we annotate root-cause features that a human
reader can use to diagnose the failure (mode, R², W, Δt_c, A-anchor
ratio, gravity drift, out-of-lobe residual density ratio). The goal
is not to improve the algorithm further but to be honest in the paper
about why these particular rides are still hard.

Outputs:

* ``docs/latex/figures/worst_train.png`` — 4-panel collage of the
  four worst train predictions.
* ``docs/latex/figures/worst_test.png`` — same, for the test split.

These are embedded in the paper's failure-mode analysis subsection.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

TEST_CSV = REPO / "src/data/structuredData/test_results/prediction/test/predictions_trapezoid_test.csv"
TRAIN_CSV = REPO / "src/data/structuredData/test_results/prediction/train/predictions_trapezoid_train.csv"
PAPER_OUT = REPO / "docs/latex/figures"


def _find_record(records, exp_name: str, seg_idx: int):
    for r in records:
        if r.exp_name == exp_name and int(r.seg_idx) == int(seg_idx):
            return r
    return None


def _plot_one(ax, rec, out, row, rank: int):
    meta = out.meta or {}
    t = meta.get("t_sec"); a_smooth = meta.get("a_smooth"); a_template = meta.get("a_template")
    if t is None or a_smooth is None or a_template is None:
        ax.set_axis_off()
        ax.set_title(f"#{rank}: no template available")
        return
    params = meta.get("params", {})
    mode = meta.get("mode", "")
    W = params.get("W", 0.0); f = params.get("f", 0.0)
    t_c1 = params.get("t_c1", 0.0); t_c2 = params.get("t_c2", 0.0)
    joint_r2 = params.get("joint_r2", float("nan"))
    A_anchor = meta.get("A_anchor_ratio", float("nan"))
    out_of_lobe = meta.get("out_of_lobe_residual_frac", float("nan"))

    ax.plot(t, a_smooth, color="black", linewidth=1.1, label="smoothed $a_\\mathrm{vert}$")
    ax.plot(t, a_template,
            color="#d62728" if mode == "joined" else "#1f77b4",
            linewidth=1.8, alpha=0.85, label=f"{mode} fit  $R^2={joint_r2:.2f}$")
    for t_c, col in [(t_c1, "#ff7f0e"), (t_c2, "#2ca02c")]:
        ax.axvspan(t_c - W, t_c + W, color=col, alpha=0.10)
    ax.axhline(0, color="gray", linewidth=0.4, alpha=0.5)
    ax.set_xlabel("t (s)", fontsize=8)
    ax.set_ylabel("$a_\\mathrm{vert}$ (m/s$^2$)", fontsize=8)
    ax.tick_params(labelsize=8)
    ax.legend(fontsize=7, loc="upper right")
    ax.grid(linestyle=":", linewidth=0.4, alpha=0.5)

    title = (
        f"#{rank}: {rec.exp_name.split('_')[1]} / seg{int(rec.seg_idx):03d}  "
        f"true={row['true_dh']:+.2f}m  pred={row['pred_dh']:+.2f}m  "
        f"err={row['abs_error']:.2f}m\n"
        f"mode={mode}  $R^2$={joint_r2:.2f}  W={W:.2f}s  "
        f"$\\Delta t_c$={abs(t_c2-t_c1):.2f}s  "
        f"A_anchor={A_anchor:.2f}  out/in={out_of_lobe:.2f}"
    )
    ax.set_title(title, fontsize=8)


def make_worst_figure(df: pd.DataFrame, split: str, records, est,
                      out_path: Path, top_n: int = 4):
    # Only consider accepted clean-ground-truth segments — the worst
    # predictions the algorithm actually SHIPPED.
    cand = df.copy()
    cand = cand[cand["accepted"].astype(bool)]
    cand = cand[cand["signal_clear"].astype(bool)]
    cand["abs_error"] = (cand["pred_dh"] - cand["true_dh"]).abs()
    # Dedupe: the same physical ride is recorded by up to 4 phones at
    # once, so sorting by abs_error can produce near-duplicate panels
    # from different phones of the same ride. Key on
    # (start_ms, end_ms) and keep the worst per physical segment.
    cand = (
        cand.sort_values("abs_error", ascending=False)
            .drop_duplicates(subset=["start_ms", "end_ms"], keep="first")
    )
    worst = cand.head(top_n)

    fig, axes = plt.subplots(2, 2, figsize=(15, 9))
    axes = axes.ravel()
    for i, (_, row) in enumerate(worst.iterrows()):
        ax = axes[i]
        rec = _find_record(records, row["exp_name"], row["seg_idx"])
        if rec is None:
            ax.set_axis_off()
            continue
        out = est.predict_segment(
            rec.acc, phone_model=rec.phone, pre=rec.pre_acc, post=rec.post_acc,
        )
        _plot_one(ax, rec, out, row, rank=i + 1)
    # If there are fewer than top_n, hide unused panels
    for j in range(len(worst), 4):
        axes[j].set_axis_off()
    fig.suptitle(
        f"Worst {top_n} accepted predictions on the {split} split",
        fontsize=12, y=1.00,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote {out_path.relative_to(REPO)}  ({len(worst)} panels)")
    return worst


def main():
    from src.prediction.evaluation.dataset import load_all_segments
    from src.prediction.algorithms.configTypes import TrapezoidAccelConfig
    from src.prediction.algorithms.accelerometer_only.trapezoid_accel.estimator import TrapezoidAccelEstimator

    print("loading all segments ...")
    segs = load_all_segments()
    est = TrapezoidAccelEstimator(TrapezoidAccelConfig())

    PAPER_OUT.mkdir(parents=True, exist_ok=True)

    if TRAIN_CSV.exists():
        df_train = pd.read_csv(TRAIN_CSV)
        worst_train = make_worst_figure(
            df_train, "train", segs, est, PAPER_OUT / "worst_train.png", top_n=4,
        )
        worst_train.to_csv(PAPER_OUT / "worst_train.csv", index=False)
    if TEST_CSV.exists():
        df_test = pd.read_csv(TEST_CSV)
        worst_test = make_worst_figure(
            df_test, "test", segs, est, PAPER_OUT / "worst_test.png", top_n=4,
        )
        worst_test.to_csv(PAPER_OUT / "worst_test.csv", index=False)


if __name__ == "__main__":
    main()
