"""Render LaTeX-ready diagnostic figures for the three damped-phone
outlier experiments discussed in §3.3.10 of docs/latex/main.tex.

For each outlier:
  * loads sensors+gt via getExperimentData,
  * runs the deployed detector (``predict_intervals``) on the ACC stream,
  * writes a session-overview PNG (altitude / a_vert / signed-R²) with GT
    bands and detector predictions overlaid,
  * picks the worst-scored missed GT ride, calls ``diagnose_window`` on
    it, and writes a detail PNG showing the per-window heatmaps + zoomed
    signal + signed-R² panel — the same content the editor's right pane
    shows when a GT row is clicked,
  * walks every GT ride in the session, calls ``diagnose_window`` on
    each, and writes a per-experiment scoreboard PNG (peak-|A|, peak-R²,
    pair-|A|, pair-R²) plus a CSV row with summary statistics.

Outputs land in ``docs/latex/figures/seg_eval/outliers/``.
"""
from __future__ import annotations

from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.gt_editor import TYPE_COLORS  # noqa: E402
from src.data.loader import RAW_DATA_ROOT, getExperimentData  # noqa: E402
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E402
    trapezoid_kernel,
)


OUT_DIR = _REPO_ROOT / "docs" / "latex" / "figures" / "seg_eval" / "outliers"
OUT_DIR.mkdir(parents=True, exist_ok=True)

PRED_COLORS = {"up": "#1f3a5f", "down": "#7d3c98"}

OUTLIERS = [
    ("BarIlan2_Pix10",
     "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026"),
    ("milleniumHotel_Pix10",
     "UriyaCohenEliya_milleniumHotel_GooglePixel10_15-04-2026_exp2"),
    ("milleniumHotel_A23",
     "eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1"),
]


def _gt_rows(gt: pd.DataFrame, t0_ms: float) -> list[tuple[int, float, float, str]]:
    rows: list[tuple[int, float, float, str]] = []
    if gt is None or gt.empty:
        return rows
    for i, row in gt.iterrows():
        rt = str(row.get("type", ""))
        if rt not in ("up", "down"):
            continue
        s = (float(row["start_ms"]) - t0_ms) / 1000.0
        e = (float(row["end_ms"]) - t0_ms) / 1000.0
        if e > s:
            rows.append((int(i), s, e, rt))
    return rows


def _matched(t_lo: float, t_hi: float, predictions: list[dict]) -> bool:
    for p in predictions:
        if p["t_start_s"] <= t_hi and p["t_end_s"] >= t_lo:
            return True
    return False


def _summarise_reject_flags(diag: dict) -> list[str]:
    """Return a flat list of failure tags collected from ``diag``."""
    tags: list[str] = []
    pos = diag.get("pos_peak")
    neg = diag.get("neg_peak")
    cfg = diag.get("config")  # not present; we read from state separately
    return tags  # placeholder — we reuse text from verdict_lines instead


def _classify_gt(diag: dict, cfg) -> dict:
    """Boil ``diagnose_window`` output down to numbers we can aggregate."""
    pos = diag.get("pos_peak")
    neg = diag.get("neg_peak")
    pair = diag.get("pair")
    out: dict = {
        "pos_A": float(pos[1]) if pos else np.nan,
        "pos_R2": float(pos[2]) if pos else np.nan,
        "neg_A": float(neg[1]) if neg else np.nan,
        "neg_R2": float(neg[2]) if neg else np.nan,
        "pair_A": pair["A_abs"] if pair else np.nan,
        "pair_R2": pair["joint_r2_mean"] if pair else np.nan,
        "pair_heatmap": pair["heatmap_energy"] if pair else np.nan,
        "reject_flags": (pair["reject_flags"] if pair else []),
        "fail_low_peak_A": False,
        "fail_low_peak_R2": False,
        "fail_pair_R2": False,
        "fail_pair_A": False,
        "fail_heatmap": False,
    }
    # Per-peak gates (pos and neg both need to clear to even attempt pair).
    for peak_A, peak_R2 in ((out["pos_A"], out["pos_R2"]),
                            (out["neg_A"], out["neg_R2"])):
        if not np.isfinite(peak_A) or not np.isfinite(peak_R2):
            continue
        if abs(peak_A) < cfg.min_peak_abs_a:
            out["fail_low_peak_A"] = True
        if peak_R2 < cfg.r2_peak_thresh:
            out["fail_low_peak_R2"] = True
    if pair is not None:
        for f in pair["reject_flags"]:
            if f.startswith("joint R²"):
                out["fail_pair_R2"] = True
            elif f.startswith("pair |A|"):
                out["fail_pair_A"] = True
            elif f.startswith("heatmap energy"):
                out["fail_heatmap"] = True
    else:
        # If pair was never attempted because one side had no sample, count
        # it as a peak-stage failure.
        if pos is None or neg is None:
            out["fail_low_peak_A"] = True
    return out


def _draw_session_overview(
    out_path: Path, exp_name: str, sensors: dict, gt: pd.DataFrame,
    state: dict, predictions: list[dict],
) -> None:
    prs = sensors.get("PRS")
    acc = sensors.get("ACC")
    t0_ms = float(prs["timestamp_ms"].iloc[0]) if (prs is not None and not prs.empty) \
        else float(acc["timestamp_ms"].iloc[0])
    acc_t0_ms = float(acc["timestamp_ms"].iloc[0])
    offset = (acc_t0_ms - t0_ms) / 1000.0  # seconds to add to ACC-local times

    fig, axes = plt.subplots(3, 1, figsize=(11.5, 7.0), sharex=True)
    ax_alt, ax_vert, ax_r2 = axes

    # Altitude (PRS).
    if prs is not None and not prs.empty and "GT_height_m" in prs.columns:
        t_prs = (prs["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
        ax_alt.plot(t_prs, prs["GT_height_m"].to_numpy(dtype=float),
                    color="tab:green", lw=0.9)
    ax_alt.set_ylabel("altitude (m)")
    ax_alt.grid(True, alpha=0.3)

    # a_vert + a_smooth (ACC, projected onto gravity).
    t_acc = state["t"] + offset
    ax_vert.plot(t_acc, state["a_vert"], color="#2c3e50", lw=0.4,
                 label=r"$a_\mathrm{vert}$")
    ax_vert.plot(t_acc, state["a_smooth"], color="#e67e22", lw=0.7,
                 label="smoothed", alpha=0.9)
    ax_vert.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    cfg = state["config"]
    ax_vert.axhline(+cfg.min_peak_abs_a, color="#888", lw=0.5, ls=":",
                    alpha=0.7, label=fr"$\pm$min_peak_abs_a={cfg.min_peak_abs_a:.2f}")
    ax_vert.axhline(-cfg.min_peak_abs_a, color="#888", lw=0.5, ls=":", alpha=0.7)
    ax_vert.set_ylabel("a (m/s²)")
    ax_vert.grid(True, alpha=0.3)
    ax_vert.legend(loc="upper right", fontsize=7, ncol=2, frameon=False)

    # Signed-R² traces with classified peaks.
    pos_r2 = state["best_pos_r2"]
    neg_r2 = state["best_neg_r2"]
    ax_r2.plot(t_acc, np.where(np.isfinite(pos_r2), pos_r2, np.nan),
               color="#2980b9", lw=0.6, label="max R² (+)")
    ax_r2.plot(t_acc, np.where(np.isfinite(neg_r2), neg_r2, np.nan),
               color="#c0392b", lw=0.6, label="max R² (−)")
    ax_r2.axhline(cfg.r2_peak_thresh, color="gray", lw=0.5, ls="--", alpha=0.6,
                  label=fr"r2_peak_thresh={cfg.r2_peak_thresh:.2f}")
    ax_r2.set_ylim(0, 1.05)
    ax_r2.set_ylabel("R² (per sign)")
    ax_r2.set_xlabel("time (s)")
    ax_r2.grid(True, alpha=0.3)
    ax_r2.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)

    # GT shading on every axis.
    if gt is not None and not gt.empty:
        for _, row in gt.iterrows():
            rt = str(row.get("type", ""))
            if rt not in ("up", "down"):
                continue
            s = (float(row["start_ms"]) - t0_ms) / 1000.0
            e = (float(row["end_ms"]) - t0_ms) / 1000.0
            color = TYPE_COLORS.get(rt, "#cccccc")
            for ax in axes:
                ax.axvspan(s, e, color=color, alpha=0.15, zorder=0)

    # Detector predictions on every axis.
    for p in predictions:
        s = p["t_start_s"] + offset
        e = p["t_end_s"] + offset
        col = PRED_COLORS[p["ride_type"]]
        for ax in axes:
            ax.axvspan(s, e, color=col, alpha=0.20, hatch="//", zorder=1)
            ax.axvline(s, color=col, lw=0.7, ls="--", alpha=0.7)
            ax.axvline(e, color=col, lw=0.7, ls="--", alpha=0.7)

    fig.suptitle(
        f"{exp_name}\n"
        f"GT up/down rides shaded; detector predictions hatched. "
        f"matched={sum(_matched(s, e, predictions) for _, s, e, _ in _gt_rows(gt, acc_t0_ms))}"
        f" / {len(_gt_rows(gt, acc_t0_ms))} GT.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_gt_detail(
    out_path: Path, label: str,
    state: dict, t_lo: float, t_hi: float, ride_type: str,
    diag: dict,
) -> None:
    """Per-GT detail figure — same content the editor's right pane shows
    when a GT row is clicked: heatmaps at the best ± samples, zoomed
    a_vert + best-fit trapezoids, signed-R² panel."""
    t = state["t"]
    a_smooth = state["a_smooth"]
    a_vert = state["a_vert"]
    cfg = state["config"]

    fig = plt.figure(figsize=(11.5, 8.5))
    gs = fig.add_gridspec(3, 2, height_ratios=[1.0, 1.0, 0.7],
                          hspace=0.55, wspace=0.28)
    ax_h_pos = fig.add_subplot(gs[0, 0])
    ax_h_neg = fig.add_subplot(gs[0, 1])
    ax_sig = fig.add_subplot(gs[1, :])
    ax_r2 = fig.add_subplot(gs[2, :])

    grid_w_s = state["grid_w_s"]
    grid_f = state["grid_f"]
    extent = (grid_f[0], grid_f[-1], grid_w_s[0], grid_w_s[-1])

    def _plot_heat(ax, peak, label):
        if peak is None:
            ax.text(0.5, 0.5, f"no {label} sample in window",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")
            ax.set_xticks([]); ax.set_yticks([])
            ax.set_title(f"{label} lobe heatmap — n/a", fontsize=9)
            return
        i, A, _r2 = peak
        heat = _detect.heatmap_at(a_smooth, t, i, grid_w_s, grid_f)
        im = ax.imshow(heat, origin="lower", aspect="auto", extent=extent,
                       cmap="viridis", vmin=0.0, vmax=1.0)
        ax.set_xlabel("plateau f"); ax.set_ylabel("half-width W (s)")
        ax.set_title(f"{label} lobe @ t={t[i]:.1f}s  A={A:+.2f}", fontsize=9)
        fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    _plot_heat(ax_h_pos, diag.get("pos_peak"), "+")
    _plot_heat(ax_h_neg, diag.get("neg_peak"), "−")

    pad = max(5.0, (t_hi - t_lo) * 0.6)
    mask = (t >= t_lo - pad) & (t <= t_hi + pad)
    ax_sig.plot(t[mask], a_vert[mask], color="#2c3e50", lw=0.5, label=r"$a_\mathrm{vert}$")
    ax_sig.plot(t[mask], a_smooth[mask], color="#e67e22", lw=1.0, label="smoothed")
    ax_sig.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax_sig.axhline(+cfg.min_peak_abs_a, color="#888", lw=0.5, ls=":", alpha=0.7)
    ax_sig.axhline(-cfg.min_peak_abs_a, color="#888", lw=0.5, ls=":", alpha=0.7)
    ax_sig.axvspan(t_lo, t_hi, color=TYPE_COLORS[ride_type], alpha=0.22, zorder=0)
    for peak, color, tag in ((diag.get("pos_peak"), "#2980b9", "+"),
                             (diag.get("neg_peak"), "#c0392b", "−")):
        if peak is None:
            continue
        i, A, _ = peak
        ax_sig.axvline(t[i], color=color, lw=0.9, ls=":", alpha=0.85)
        ax_sig.scatter([t[i]], [A], color=color, s=24, zorder=5)
    pair = diag.get("pair")
    if pair is not None:
        for sign, key in ((+1, "i1"), (-1, "i2")):
            i = pair[key]
            tt = np.linspace(t[i] - pair["W"], t[i] + pair["W"], 200)
            yy = (sign * pair["A_abs"]) * trapezoid_kernel(
                tt, t[i], pair["W"], pair["frac_flat"]
            )
            ax_sig.plot(tt, yy, color="#c0392b", lw=1.2, alpha=0.9)
    ax_sig.set_xlabel("t (s, ACC-local)")
    ax_sig.set_ylabel("a (m/s²)")
    ax_sig.grid(True, alpha=0.25)
    ax_sig.legend(fontsize=8, loc="upper right")
    ax_sig.set_title(label, fontsize=10)

    pos_r2 = state["best_pos_r2"]
    neg_r2 = state["best_neg_r2"]
    t_lo_w = float(t_lo - pad); t_hi_w = float(t_hi + pad)
    rmask = (t >= t_lo_w) & (t <= t_hi_w)
    ax_r2.plot(t[rmask], np.where(np.isfinite(pos_r2[rmask]), pos_r2[rmask], np.nan),
               color="#2980b9", lw=0.9, label="max R² (+)")
    ax_r2.plot(t[rmask], np.where(np.isfinite(neg_r2[rmask]), neg_r2[rmask], np.nan),
               color="#c0392b", lw=0.9, label="max R² (−)")
    ax_r2.axhline(cfg.r2_peak_thresh, color="gray", lw=0.5, ls="--", alpha=0.6,
                  label=fr"r2_peak_thresh={cfg.r2_peak_thresh:.2f}")
    ax_r2.axvspan(t_lo, t_hi, color=TYPE_COLORS[ride_type], alpha=0.22, zorder=0)
    ax_r2.set_xlim(t_lo_w, t_hi_w)
    ax_r2.set_ylim(0, 1.05)
    ax_r2.set_ylabel("R² (per sign)")
    ax_r2.set_xlabel("t (s, ACC-local)")
    ax_r2.grid(True, alpha=0.25)
    ax_r2.legend(loc="upper right", fontsize=7, ncol=3, frameon=False)

    # Annotate verdict in the bottom-left corner of the signal panel.
    verdict = "\n".join(diag.get("verdict_lines", []))
    if verdict:
        ax_sig.text(
            0.005, 0.02, verdict, transform=ax_sig.transAxes,
            ha="left", va="bottom", fontsize=7, family="monospace",
            bbox=dict(facecolor="#ffffff", alpha=0.85,
                      edgecolor="#888", boxstyle="round,pad=0.3"),
            zorder=20,
        )

    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _draw_scoreboard(
    out_path: Path, exp_label: str, gt_diags: list[dict], cfg,
) -> None:
    """Per-experiment scoreboard: distribution of peak-|A|, peak-R², pair-|A|,
    pair-R² across every GT ride. Threshold lines drawn for context."""
    if not gt_diags:
        return
    df = pd.DataFrame(gt_diags)
    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.5))
    ax1, ax2, ax3, ax4 = axes.ravel()

    # Stack pos+neg peak distributions so each GT contributes two values.
    peak_A = np.concatenate([
        np.abs(df["pos_A"].dropna().to_numpy(dtype=float)),
        np.abs(df["neg_A"].dropna().to_numpy(dtype=float)),
    ])
    peak_R2 = np.concatenate([
        df["pos_R2"].dropna().to_numpy(dtype=float),
        df["neg_R2"].dropna().to_numpy(dtype=float),
    ])
    pair_A = df["pair_A"].dropna().to_numpy(dtype=float)
    pair_R2 = df["pair_R2"].dropna().to_numpy(dtype=float)

    def _hist(ax, vals, thr, label):
        if vals.size == 0:
            ax.text(0.5, 0.5, f"no {label} samples", transform=ax.transAxes,
                    ha="center", va="center", color="#888")
            ax.set_axis_off()
            return
        bins = np.linspace(0, max(np.nanmax(vals), thr * 1.5), 30)
        ax.hist(vals, bins=bins, color="#3498db", edgecolor="#222", lw=0.4)
        ax.axvline(thr, color="#c0392b", lw=1.2, ls="--",
                   label=f"deployed floor = {thr:.2f}")
        ax.set_xlabel(label)
        ax.set_ylabel("count (GT rides)")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    _hist(ax1, peak_A, cfg.min_peak_abs_a, "peak |A| (m/s²)")
    _hist(ax2, peak_R2, cfg.r2_peak_thresh, "peak R²")
    _hist(ax3, pair_A, cfg.min_pair_abs_a, "pair |A| (m/s²)")
    _hist(ax4, pair_R2, cfg.joint_r2_thresh, "pair joint R²")
    fig.suptitle(
        f"{exp_label} — per-GT score distributions over "
        f"{len(df)} GT rides. Threshold lines mark the deployed gates.",
        fontsize=10,
    )
    fig.tight_layout()
    fig.savefig(out_path, dpi=140, bbox_inches="tight")
    plt.close(fig)


def _process(label: str, exp_name: str) -> dict:
    print(f"\n=== {label} :: {exp_name} ===")
    sensors, gt, _meta = getExperimentData(RAW_DATA_ROOT / exp_name, use_cache=True)
    acc = sensors.get("ACC")
    if acc is None or acc.empty:
        raise RuntimeError(f"{exp_name}: no ACC data")
    predictions, state = _detect.predict_intervals(acc)
    if not state:
        raise RuntimeError(f"{exp_name}: detector returned no state")
    cfg = state["config"]

    # Session overview.
    overview_path = OUT_DIR / f"{label}_overview.png"
    _draw_session_overview(overview_path, exp_name, sensors, gt, state, predictions)
    print(f"  wrote {overview_path.name}")

    # Per-GT diagnoses.
    acc_t0_ms = float(acc["timestamp_ms"].iloc[0])
    rows = _gt_rows(gt, acc_t0_ms)
    gt_diags: list[dict] = []
    for gi, t_s, t_e, rt in rows:
        diag = _detect.diagnose_window(state, t_s, t_e, ride_type=rt)
        rec = _classify_gt(diag, cfg)
        rec.update({
            "gt_index": gi, "t_start": t_s, "t_end": t_e, "ride_type": rt,
            "matched": _matched(t_s, t_e, predictions),
            "verdict": " ".join(diag.get("verdict_lines") or []),
            "diag": diag,
        })
        gt_diags.append(rec)

    # Pick the worst (most-missed, lowest pair score) representative GT.
    missed = [g for g in gt_diags if not g["matched"]]
    if missed:
        # Prefer one where a pair was attempted, sort by pair R² then peak |A|.
        with_pair = [g for g in missed if np.isfinite(g["pair_R2"])]
        if with_pair:
            sample = min(with_pair, key=lambda g: (g["pair_R2"], -g["pair_A"]))
        else:
            sample = min(missed, key=lambda g: max(abs(g["pos_A"] or 0), abs(g["neg_A"] or 0)) or 0)
    else:
        sample = gt_diags[0]
    detail_path = OUT_DIR / f"{label}_gt{sample['gt_index']:02d}_detail.png"
    _draw_gt_detail(
        detail_path,
        f"GT #{sample['gt_index']:02d} {sample['ride_type']}  "
        f"window=[{sample['t_start']:.1f},{sample['t_end']:.1f}]s — "
        f"{'matched' if sample['matched'] else 'NOT matched'}",
        state, sample["t_start"], sample["t_end"], sample["ride_type"], sample["diag"],
    )
    print(f"  wrote {detail_path.name}")

    # Score-distribution scoreboard.
    sb_path = OUT_DIR / f"{label}_scoreboard.png"
    _draw_scoreboard(sb_path, exp_name, gt_diags, cfg)
    print(f"  wrote {sb_path.name}")

    # Numerical summary.
    n = len(gt_diags)
    n_matched = sum(g["matched"] for g in gt_diags)
    pos_A = np.abs([g["pos_A"] for g in gt_diags if np.isfinite(g["pos_A"])])
    neg_A = np.abs([g["neg_A"] for g in gt_diags if np.isfinite(g["neg_A"])])
    peak_A_all = np.concatenate([pos_A, neg_A]) if pos_A.size or neg_A.size else np.array([])
    pair_A_all = np.array([g["pair_A"] for g in gt_diags if np.isfinite(g["pair_A"])])
    pair_R2_all = np.array([g["pair_R2"] for g in gt_diags if np.isfinite(g["pair_R2"])])

    n_low_peak_A = sum(g["fail_low_peak_A"] for g in gt_diags)
    n_low_peak_R2 = sum(g["fail_low_peak_R2"] for g in gt_diags)
    n_pair_R2 = sum(g["fail_pair_R2"] for g in gt_diags)
    n_pair_A = sum(g["fail_pair_A"] for g in gt_diags)
    n_heatmap = sum(g["fail_heatmap"] for g in gt_diags)

    summary = {
        "label": label,
        "exp": exp_name,
        "n_gt": n,
        "n_matched": n_matched,
        "median_peak_A": float(np.median(peak_A_all)) if peak_A_all.size else float("nan"),
        "median_pair_A": float(np.median(pair_A_all)) if pair_A_all.size else float("nan"),
        "median_pair_R2": float(np.median(pair_R2_all)) if pair_R2_all.size else float("nan"),
        "n_fail_low_peak_A": n_low_peak_A,
        "n_fail_low_peak_R2": n_low_peak_R2,
        "n_fail_pair_R2": n_pair_R2,
        "n_fail_pair_A": n_pair_A,
        "n_fail_heatmap": n_heatmap,
        "min_peak_abs_a": cfg.min_peak_abs_a,
        "min_pair_abs_a": cfg.min_pair_abs_a,
        "joint_r2_thresh": cfg.joint_r2_thresh,
    }
    print(
        f"  {n} GT rides, matched={n_matched}.  "
        f"median peak |A|={summary['median_peak_A']:.3f} "
        f"(floor={cfg.min_peak_abs_a:.2f}).  "
        f"median pair |A|={summary['median_pair_A']:.3f} "
        f"(floor={cfg.min_pair_abs_a:.2f}).  "
        f"median pair R²={summary['median_pair_R2']:.3f} "
        f"(floor={cfg.joint_r2_thresh:.2f})."
    )
    print(
        f"  failure counts: peak |A|<floor: {n_low_peak_A}/{n}  "
        f"peak R²<floor: {n_low_peak_R2}/{n}  "
        f"pair R²<floor: {n_pair_R2}/{n}  "
        f"pair |A|<floor: {n_pair_A}/{n}  "
        f"heatmap energy<floor: {n_heatmap}/{n}"
    )
    return summary


def main() -> int:
    print(f"writing figures into {OUT_DIR}")
    summaries = [_process(label, exp) for label, exp in OUTLIERS]
    df = pd.DataFrame(summaries)
    csv_path = OUT_DIR / "outliers_summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nwrote {csv_path}")
    print(df.to_string(index=False))
    return 0


if __name__ == "__main__":
    sys.exit(main())
