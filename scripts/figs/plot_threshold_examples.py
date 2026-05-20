"""Generate the Appendix-D per-threshold example figures for the IMWUT paper.

Three PNGs land in ``paper_phd/figures/``:

* ``threshold_grid_energy.png``    — a rich vs a low-energy joint-R² heatmap,
  illustrating the grid-energy gate ``e_min`` (Gate 4).
* ``threshold_quiet_middle.png``   — a flat inter-lobe cruise (accepted) vs a
  non-flat cruise (rejected), illustrating the quiet-middle gate ``rho`` (Gate 5).
* ``threshold_duration_penalty.png`` — a multi-stop trip whose four lobes form
  two short rides plus one long "super pair", illustrating the duration
  penalty ``lambda`` and greedy resolution (Gate 6).

The script runs the deployed detector over the dataset, ranks candidate
recordings for each figure, auto-picks the strongest, and renders. Pass
``--scan-only`` to print the shortlists without rendering; ``--limit N`` caps
how many experiments are scanned. ``OVERRIDES`` pins specific picks once chosen.

Run:  venv/bin/python -m scripts.figs.plot_threshold_examples
"""
from __future__ import annotations

import argparse
import sys
import traceback
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.loader import (  # noqa: E402
    RAW_DATA_ROOT, getExperimentData, resolve_experiments,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    pair_filter as _pair,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E402
    trapezoid_kernel,
)

PAPER_FIG = REPO / "paper_phd" / "figures"
LAMBDA = 0.01  # _DURATION_PENALTY_LAMBDA in pair_filter.predict_pairs

# Pin chosen recordings here once --scan-only has been inspected. Leave a key
# unset to let the auto-picker choose. Each value: (exp_name, *indices).
OVERRIDES: dict[str, tuple] = {
    "energy":   ("eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5",),
    "cruise":   ("eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2",),
    "tilted":   ("eyalyakir_beitYitzchakiRaanana_Xiaomi22101320I_15-04-2026_exp6",),
    "duration": ("eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2",),
}


# ---------------------------------------------------------------------------
# Numeric helpers — mirror the gate computations in pair_filter.py
# ---------------------------------------------------------------------------

def joint_heatmap(
    a: np.ndarray, t: np.ndarray, i1: int, i2: int, s1: float, s2: float,
    grid_w_s: np.ndarray, grid_f: np.ndarray,
) -> np.ndarray:
    """``(nW, nF)`` grid of the shared-shape joint R² for one pair — the
    quantity ``joint_pair_score`` averages into ``heatmap_energy``. Sign-failed
    cells are 0 (they contribute 0 energy); off-signal cells are NaN."""
    n = a.size
    dt = float(np.median(np.diff(t))) if t.size > 1 else 0.01
    nW, nF = len(grid_w_s), len(grid_f)
    grid = np.full((nW, nF), np.nan)
    for wi, W in enumerate(grid_w_s):
        K = max(3, int(round(2 * W / dt)))
        if K % 2 == 0:
            K += 1
        half = K // 2
        if i1 - half < 0 or i1 + half >= n or i2 - half < 0 or i2 + half >= n:
            continue
        win1 = a[i1 - half: i1 + half + 1]
        win2 = a[i2 - half: i2 + half + 1]
        p1 = float(np.sum(win1 * win1))
        p2 = float(np.sum(win2 * win2))
        if p1 < 1e-9 or p2 < 1e-9:
            continue
        t_k = (np.arange(K) - half) * dt
        for fi, f in enumerate(grid_f):
            tpl = trapezoid_kernel(t_k, 0.0, float(W), float(f))
            norm_t = float(np.sum(tpl * tpl))
            if norm_t < 1e-9:
                continue
            u1 = s1 * float(np.dot(win1, tpl))
            u2 = s2 * float(np.dot(win2, tpl))
            if u1 <= 0 or u2 <= 0:
                grid[wi, fi] = 0.0
                continue
            A = (u1 + u2) / (2.0 * norm_t)
            if A <= 0:
                grid[wi, fi] = 0.0
                continue
            ss1 = p1 - 2.0 * A * u1 + A * A * norm_t
            ss2 = p2 - 2.0 * A * u2 + A * A * norm_t
            grid[wi, fi] = 0.5 * ((1.0 - ss1 / p1) + (1.0 - ss2 / p2))
    return grid


def heatmap_energy_of(grid: np.ndarray) -> float:
    """Mean of ``max(0, joint R²)`` over valid cells — matches the gate."""
    valid = grid[np.isfinite(grid)]
    if valid.size == 0:
        return 0.0
    return float(np.mean(np.maximum(valid, 0.0)))


def rms_mid(
    a_smooth: np.ndarray, t: np.ndarray, t_c1: float, t_c2: float, W: float,
) -> float | None:
    """RMS of the smoothed acceleration on the inter-lobe cruise
    ``(t_c1 + W, t_c2 - W)`` — the quiet-middle quantity."""
    lo, hi = t_c1 + W, t_c2 - W
    if hi <= lo:
        return None
    m = (t >= lo) & (t <= hi)
    if not m.any():
        return None
    seg = a_smooth[m]
    return float(np.sqrt(np.mean(seg * seg)))


def candidate_pairs(state: dict) -> list[tuple[int, int, float, float]]:
    """Opposite-sign detected-peak pairs inside the gap window — the pairs the
    pair filter scores before the gates run."""
    t = state["t"]
    signs = state["signs"]
    cfg = state["config"]
    peaks = state["final_peaks"]
    pos = [i for i in peaks if signs[i] > 0]
    neg = [i for i in peaks if signs[i] < 0]
    out: list[tuple[int, int, float, float]] = []
    for i1 in pos:
        for i2 in neg:
            if i2 > i1 and cfg.min_ride_s <= t[i2] - t[i1] <= cfg.max_ride_s:
                out.append((i1, i2, +1.0, -1.0))
    for i1 in neg:
        for i2 in pos:
            if i2 > i1 and cfg.min_ride_s <= t[i2] - t[i1] <= cfg.max_ride_s:
                out.append((i1, i2, -1.0, +1.0))
    return out


# ---------------------------------------------------------------------------
# Scan — collect figure candidates from one experiment
# ---------------------------------------------------------------------------

def scan_experiment(name: str) -> dict | None:
    """Run the detector on one experiment and collect figure candidates."""
    try:
        sensors, _gt, _meta = getExperimentData(RAW_DATA_ROOT / name, use_cache=True)
    except Exception:
        return None
    acc = sensors.get("ACC")
    if acc is None or acc.empty:
        return None
    predictions, state = _detect.predict_intervals(acc)
    if not state:
        return None
    cfg = state["config"]
    t = state["t"]
    a_smooth = state["a_smooth"]
    grid_w_s = state["grid_w_s"]
    grid_f = state["grid_f"]

    found: dict = {"name": name, "rich": [], "low_energy": [], "flat": [],
                   "tilted": [], "multistop": []}

    # Accepted predictions feed the "rich heatmap" (F1) and "flat cruise" (F2).
    for p in predictions:
        l1, l2 = p["lobe1"], p["lobe2"]
        W = float(l1["half_width_s"])
        gap = float(l2["t_c"]) - float(l1["t_c"])
        rms = rms_mid(a_smooth, t, float(l1["t_c"]), float(l2["t_c"]), W)
        a_abs = abs(float(l1["a_peak"]))
        found["rich"].append({
            "energy": float(p["heatmap_energy"]), "pred": p, "name": name,
        })
        if rms is not None and gap > 3.0 * W and a_abs > 1e-6:
            found["flat"].append({
                "ratio": rms / a_abs, "rms": rms, "pred": p, "name": name,
            })

    # Rejected candidate pairs feed the "low-energy" (F1) and "tilted" (F3)
    # examples — the pair filter drops them before they reach `predictions`.
    for (i1, i2, s1, s2) in candidate_pairs(state):
        res = _pair.joint_pair_score(a_smooth, t, i1, i2, s1, s2, grid_w_s, grid_f)
        if res is None:
            continue
        score, W, f, a_abs, r2_1, r2_2, energy = res
        if score < cfg.joint_r2_thresh or a_abs < cfg.min_pair_abs_a:
            continue
        rec = {"name": name, "i1": int(i1), "i2": int(i2),
               "s1": s1, "s2": s2, "score": score, "W": W, "f": f,
               "A": a_abs, "energy": energy,
               "t_c1": float(t[i1]), "t_c2": float(t[i2])}
        if energy < cfg.heatmap_energy_thresh:
            found["low_energy"].append(rec)
            continue
        rms = rms_mid(a_smooth, t, float(t[i1]), float(t[i2]), W)
        if rms is not None and rms > cfg.quiet_middle_ratio * a_abs:
            rec["rms"] = rms
            rec["ratio"] = rms / a_abs
            rec["gap"] = float(t[i2]) - float(t[i1])
            found["tilted"].append(rec)

    # Two consecutive same-direction rides whose combined span still fits the
    # [0, 30] s window feed the multi-stop figure (F4).
    for k in range(len(predictions) - 1):
        a_pred, b_pred = predictions[k], predictions[k + 1]
        if a_pred["ride_type"] != b_pred["ride_type"]:
            continue
        span = float(b_pred["lobe2"]["t_c"]) - float(a_pred["lobe1"]["t_c"])
        inter = float(b_pred["t_start_s"]) - float(a_pred["t_end_s"])
        if span <= 30.0 and inter < 15.0:
            found["multistop"].append({
                "name": name, "inter": inter, "span": span,
                "ride_type": a_pred["ride_type"],
                "pa": a_pred, "pb": b_pred,
            })
    return found


# ---------------------------------------------------------------------------
# Figure renderers
# ---------------------------------------------------------------------------

def _heatmap_panel(ax, grid, grid_w_s, grid_f, title):
    extent = (grid_f[0], grid_f[-1], grid_w_s[0], grid_w_s[-1])
    im = ax.imshow(grid, origin="lower", aspect="auto", extent=extent,
                   cmap="viridis", vmin=0.0, vmax=1.0)
    ax.set_xlabel("plateau fraction $f$")
    ax.set_ylabel("half-width $W$ (s)")
    ax.set_title(title, fontsize=9)
    return im


def render_grid_energy(rich: dict, low: dict, out: Path) -> None:
    """F1 — rich vs low-energy joint-R² heatmaps."""
    sr, gw, gf = rich["state"], rich["state"]["grid_w_s"], rich["state"]["grid_f"]
    p = rich["pred"]
    i1 = int(np.argmin(np.abs(sr["t"] - float(p["lobe1"]["t_c"]))))
    i2 = int(np.argmin(np.abs(sr["t"] - float(p["lobe2"]["t_c"]))))
    s1 = 1.0 if p["ride_type"] == "up" else -1.0
    grid_rich = joint_heatmap(sr["a_smooth"], sr["t"], i1, i2, s1, -s1, gw, gf)

    sl = low["state"]
    grid_low = joint_heatmap(sl["a_smooth"], sl["t"], low["i1"], low["i2"],
                             low["s1"], low["s2"],
                             sl["grid_w_s"], sl["grid_f"])

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.3))
    im = _heatmap_panel(
        axes[0], grid_rich, gw, gf,
        f"genuine ride: broad bright band, $E={heatmap_energy_of(grid_rich):.2f}$",
    )
    _heatmap_panel(
        axes[1], grid_low, sl["grid_w_s"], sl["grid_f"],
        f"false pair: one lucky cell, $E={heatmap_energy_of(grid_low):.2f}$",
    )
    cbar = fig.colorbar(im, ax=axes, fraction=0.045, pad=0.04)
    cbar.set_label("joint $R^2$ of the $(W,f)$ template")
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def _cruise_panel(ax, state, t_c1, t_c2, W, f, A, s1, rms, ratio, accepted):
    t = state["t"]
    a_vert = state["a_vert"]
    a_smooth = state["a_smooth"]
    pad = max(2.0, W)
    lo, hi = t_c1 - W - pad, t_c2 + W + pad
    m = (t >= lo) & (t <= hi)
    t0 = t_c1 - W
    ax.plot(t[m] - t0, a_vert[m], color="#b9c2cc", lw=0.4, alpha=0.7,
            label=r"$a_\mathrm{vert}$")
    ax.plot(t[m] - t0, a_smooth[m], color="#2c3e50", lw=1.5, label="smoothed")
    for t_c, s, col in ((t_c1, s1, "#ff7f0e"), (t_c2, -s1, "#2ca02c")):
        tt = np.linspace(t_c - W, t_c + W, 240)
        ax.plot(tt - t0, s * A * trapezoid_kernel(tt, t_c, W, f),
                color="#d62728", lw=1.8, alpha=0.85)
        ax.axvspan(t_c - W - t0, t_c + W - t0, color=col, alpha=0.10)
    ax.axvspan(t_c1 + W - t0, t_c2 - W - t0, color="#1f77b4", alpha=0.12)
    ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.6)
    # Bound the y-axis to the smoothed signal — the raw a_vert carries
    # tap/drop spikes that would otherwise flatten the ride out of view.
    win_s = a_smooth[m]
    ymax = max(float(np.max(np.abs(win_s))) if win_s.size else A, A, 0.1) * 1.45
    ax.set_ylim(-ymax, ymax)
    verdict = "ACCEPTED" if accepted else "REJECTED"
    vcol = "#1a7a3a" if accepted else "#b3261e"
    rel = "\\leq" if accepted else ">"
    ax.text(0.02, 0.04,
            f"cruise $\\mathrm{{RMS}}_\\mathrm{{mid}}/A^\\star={ratio:.2f}$ "
            f"${rel}\\ \\rho=0.5$\n{verdict}",
            transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
            bbox=dict(facecolor="white", alpha=0.88, edgecolor=vcol,
                      boxstyle="round,pad=0.3"))
    ax.set_xlabel("t (s, ride-local)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def render_quiet_middle(flat: dict, tilted: dict, out: Path) -> None:
    """F2+F3 — accepted flat cruise vs rejected non-flat cruise."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.3))

    sf = flat["state"]
    p = flat["pred"]
    l1, l2 = p["lobe1"], p["lobe2"]
    s1 = 1.0 if p["ride_type"] == "up" else -1.0
    _cruise_panel(axes[0], sf, float(l1["t_c"]), float(l2["t_c"]),
                  float(l1["half_width_s"]), float(l1["frac_flat"]),
                  abs(float(l1["a_peak"])), s1, flat["rms"], flat["ratio"],
                  accepted=True)
    axes[0].set_title("flat cruise — constant velocity between the lobes",
                      fontsize=9)

    st = tilted["state"]
    _cruise_panel(axes[1], st, tilted["t_c1"], tilted["t_c2"], tilted["W"],
                  tilted["f"], tilted["A"], tilted["s1"], tilted["rms"],
                  tilted["ratio"], accepted=False)
    axes[1].set_title("non-flat cruise — the middle is not quiet", fontsize=9)

    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def render_duration_penalty(ms: dict, out: Path) -> None:
    """F4 — a multi-stop trip in two stacked panels. Panel 1: the
    acceleration trace with its four lobes, the two committed short rides
    shaded, and the super-pair span bracketed. Panel 2: the matched-filter
    signed-R² score trace, with each lobe's peak score marked. An
    annotation box compares the joint score ``S`` and the rank
    ``S - lambda*dt`` for the three candidate pairs — the penalty, not
    ``S``, is what splits the trip into two rides."""
    state = ms["state"]
    pa, pb = ms["pa"], ms["pb"]
    t = state["t"]
    a_smooth = state["a_smooth"]
    a_vert = state["a_vert"]
    gw, gf = state["grid_w_s"], state["grid_f"]

    def _finite(x: np.ndarray) -> np.ndarray:
        return np.where(np.isfinite(x), x, 0.0)

    pos_r2 = _finite(state["best_pos_r2"])
    neg_r2 = _finite(state["best_neg_r2"])

    # Four lobes in time order; each lobe's matched-filter sign is the sign
    # of its peak acceleration (take-off and landing have opposite signs).
    lobes = [pa["lobe1"], pa["lobe2"], pb["lobe1"], pb["lobe2"]]
    tcs = [float(lb["t_c"]) for lb in lobes]
    idxs = [int(np.argmin(np.abs(t - tc))) for tc in tcs]
    signs = [float(np.sign(lb["a_peak"])) for lb in lobes]

    # Three candidate pairings: ride 1 (lobes 1-2), ride 2 (lobes 3-4), and
    # the super pair (lobes 1-4). Score each and apply the duration penalty.
    def _cand(ai: int, bi: int) -> dict:
        res = _pair.joint_pair_score(a_smooth, t, idxs[ai], idxs[bi],
                                     signs[ai], signs[bi], gw, gf)
        score = float(res[0]) if res is not None else 0.0
        dt = tcs[bi] - tcs[ai]
        return {"S": score, "dt": dt, "rank": score - LAMBDA * dt}

    ride1, ride2, superp = _cand(0, 1), _cand(2, 3), _cand(0, 3)

    t_first, t_last = tcs[0], tcs[3]
    W0 = float(pa["lobe1"]["half_width_s"])
    pad = max(3.0, 1.5 * W0)
    lo, hi = t_first - W0 - pad, t_last + W0 + pad
    m = (t >= lo) & (t <= hi)
    t0 = lo

    fig, (ax1, ax2) = plt.subplots(
        2, 1, figsize=(11.5, 6.0), sharex=True,
        gridspec_kw={"height_ratios": [3, 2], "hspace": 0.13},
    )

    # --- Panel 1 — acceleration trace --------------------------------------
    ax1.plot(t[m] - t0, a_vert[m], color="#9aa6b2", lw=0.5,
             label=r"$a_\mathrm{vert}$")
    ax1.plot(t[m] - t0, a_smooth[m], color="#2c3e50", lw=1.3, label="smoothed")
    ax1.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.6)
    win_s = a_smooth[m]
    ymax = max(float(np.max(np.abs(win_s))) if win_s.size else 1.0, 0.3) * 1.5
    ax1.set_ylim(-ymax, ymax)
    for p, col in ((pa, "#2ca02c"), (pb, "#1f9e6f")):
        ax1.axvspan(float(p["lobe1"]["t_c"]) - t0,
                    float(p["lobe2"]["t_c"]) - t0, color=col, alpha=0.16)
    for n, tc in enumerate(tcs, start=1):
        ax1.axvline(tc - t0, color="#d62728", lw=0.9, ls=":", alpha=0.8)
        ax1.annotate(f"lobe {n}", (tc - t0, ymax), fontsize=7,
                     ha="center", va="top", color="#d62728")
    ax1.annotate("", xy=(t_last - t0, -ymax * 0.82),
                 xytext=(t_first - t0, -ymax * 0.82),
                 arrowprops=dict(arrowstyle="<->", color="#888", ls="dashed"))
    ax1.text((t_first + t_last) / 2 - t0, -ymax * 0.9,
             "super pair (lobes 1-4) — defeated", fontsize=7, ha="center",
             color="#888")
    box = (
        f"rank $= S - \\lambda\\,\\Delta t$,  "
        f"$\\lambda = 0.01\\ \\mathrm{{s}}^{{-1}}$\n"
        f"ride 1 (lobes 1-2):  $S = {ride1['S']:.2f}$,  "
        f"rank $= {ride1['rank']:.2f}$\n"
        f"ride 2 (lobes 3-4):  $S = {ride2['S']:.2f}$,  "
        f"rank $= {ride2['rank']:.2f}$\n"
        f"super (lobes 1-4):  $S = {superp['S']:.2f}$,  "
        f"rank $= {superp['rank']:.2f}$"
    )
    ax1.text(0.015, 0.96, box, transform=ax1.transAxes, ha="left", va="top",
             fontsize=7.5, bbox=dict(facecolor="white", alpha=0.92,
                                     edgecolor="#888",
                                     boxstyle="round,pad=0.35"))
    ax1.set_ylabel(r"$a$ (m/s$^2$)")
    ax1.set_title("a multi-stop trip — the duration penalty commits the two "
                  "short rides, not the super pair", fontsize=9)
    ax1.grid(True, alpha=0.25)
    ax1.legend(fontsize=7, loc="upper right")

    # --- Panel 2 — matched-filter signed-R² score --------------------------
    ax2.plot(t[m] - t0, pos_r2[m], color="#1f77b4", lw=1.1,
             label=r"$R^2_+$ (take-off)")
    ax2.plot(t[m] - t0, neg_r2[m], color="#d62728", lw=1.1,
             label=r"$R^2_-$ (landing)")
    ax2.set_ylim(0.0, 1.20)
    for n, (tc, idx, sg) in enumerate(zip(tcs, idxs, signs), start=1):
        trace = pos_r2 if sg > 0 else neg_r2
        col = "#1f77b4" if sg > 0 else "#d62728"
        r2_pk = float(trace[idx])
        ax2.axvline(tc - t0, color="#d62728", lw=0.7, ls=":", alpha=0.5)
        ax2.plot(tc - t0, r2_pk, "o", color=col, ms=5)
        ax2.annotate(f"lobe {n}: {r2_pk:.2f}", (tc - t0, r2_pk), fontsize=7,
                     ha="center", va="bottom", xytext=(0, 5),
                     textcoords="offset points", color=col)
    ax2.set_xlabel("t (s, trip-local)")
    ax2.set_ylabel(r"matched-filter $R^2$")
    ax2.grid(True, alpha=0.25)
    ax2.legend(fontsize=7, loc="upper right")

    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------

def _load_state(name: str) -> dict | None:
    try:
        sensors, _gt, _meta = getExperimentData(RAW_DATA_ROOT / name, use_cache=True)
        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            return None
        _preds, state = _detect.predict_intervals(acc)
        return state or None
    except Exception:
        traceback.print_exc()
        return None


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--limit", type=int, default=None,
                    help="cap how many experiments are scanned")
    ap.add_argument("--scan-only", action="store_true",
                    help="print candidate shortlists, render nothing")
    args = ap.parse_args()

    PAPER_FIG.mkdir(parents=True, exist_ok=True)
    names = [n for n in resolve_experiments(kind="all") if "__corrupted" not in n]
    if args.limit:
        names = names[:args.limit]
    print(f"scanning {len(names)} experiments ...")

    rich, low, flat, tilted, multistop = [], [], [], [], []
    for k, name in enumerate(names):
        res = scan_experiment(name)
        if res is None:
            continue
        rich += res["rich"]
        low += res["low_energy"]
        flat += res["flat"]
        tilted += res["tilted"]
        multistop += res["multistop"]
        if (k + 1) % 20 == 0:
            print(f"  ... {k + 1}/{len(names)}")

    rich.sort(key=lambda r: -r["energy"])
    low.sort(key=lambda r: r["energy"])
    flat.sort(key=lambda r: r["ratio"])
    # F3 wants a small-angle tilt — a ride-like pair whose middle is only
    # just above rho. Keep a visible, moderate cruise; smallest ratio first.
    tilted_ok = [r for r in tilted
                 if r["gap"] > 3.0 * r["W"] and 4.0 <= r["gap"] <= 9.0]
    tilted = tilted_ok or tilted
    tilted.sort(key=lambda r: r["ratio"])
    # The Gate-6 figure illustrates a passenger riding up in stages, so
    # keep only up-going multi-stop trips; rank by the inter-ride gap.
    multistop = [r for r in multistop if r["ride_type"] == "up"]
    multistop.sort(key=lambda r: r["inter"])

    def _show(tag, rows, fmt):
        print(f"\n[{tag}] {len(rows)} candidates")
        for r in rows[:8]:
            print("   " + fmt(r))

    _show("F1 rich heatmap", rich,
          lambda r: f"E={r['energy']:.2f}  {r['name']}")
    _show("F1 low-energy false pair", low,
          lambda r: f"E={r['energy']:.2f} score={r['score']:.2f} "
                    f"A={r['A']:.2f}  {r['name']}")
    _show("F2 flat cruise", flat,
          lambda r: f"RMSmid/A={r['ratio']:.2f}  {r['name']}")
    _show("F3 tilted cruise", tilted,
          lambda r: f"RMSmid/A={r['ratio']:.2f} score={r['score']:.2f}  "
                    f"{r['name']}")
    _show("F4 multi-stop trip", multistop,
          lambda r: f"inter={r['inter']:.1f}s span={r['span']:.1f}s  "
                    f"{r['name']}")

    if args.scan_only:
        return 0

    def _pick(tag, rows):
        if tag in OVERRIDES:
            want = OVERRIDES[tag][0]
            for r in rows:
                if r["name"] == want:
                    return r
        return rows[0] if rows else None

    # F1 + F2 + F3 + F4 need the full detector state of the chosen recording.
    state_cache: dict[str, dict] = {}

    def _state(name):
        if name not in state_cache:
            state_cache[name] = _load_state(name)
        return state_cache[name]

    pick_rich = _pick("energy", rich)
    pick_low = low[0] if low else None
    pick_flat = _pick("cruise", flat)
    pick_tilted = _pick("tilted", tilted)
    pick_ms = _pick("duration", multistop)

    if pick_rich and pick_low:
        pick_rich["state"] = _state(pick_rich["name"])
        pick_low["state"] = _state(pick_low["name"])
        if pick_rich["state"] and pick_low["state"]:
            render_grid_energy(pick_rich, pick_low,
                               PAPER_FIG / "threshold_grid_energy.png")
    else:
        print("  SKIP threshold_grid_energy.png — missing a candidate")

    if pick_flat and pick_tilted:
        pick_flat["state"] = _state(pick_flat["name"])
        pick_tilted["state"] = _state(pick_tilted["name"])
        if pick_flat["state"] and pick_tilted["state"]:
            render_quiet_middle(pick_flat, pick_tilted,
                                PAPER_FIG / "threshold_quiet_middle.png")
    else:
        print("  SKIP threshold_quiet_middle.png — missing a candidate")

    if pick_ms:
        pick_ms["state"] = _state(pick_ms["name"])
        if pick_ms["state"]:
            render_duration_penalty(pick_ms,
                                    PAPER_FIG / "threshold_duration_penalty.png")
    else:
        print("  SKIP threshold_duration_penalty.png — missing a candidate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
