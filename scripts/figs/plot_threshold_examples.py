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
                   "tilted": [], "multistop": [],
                   "low_score": [], "asym_amp": []}

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
        # Per-lobe amplitude proxies — the smoothed acceleration at each peak.
        A1 = float(abs(a_smooth[i1]))
        A2 = float(abs(a_smooth[i2]))
        rec = {"name": name, "i1": int(i1), "i2": int(i2),
               "s1": s1, "s2": s2, "score": score, "W": W, "f": f,
               "A": a_abs, "A1": A1, "A2": A2,
               "r2_1": r2_1, "r2_2": r2_2, "energy": energy,
               "t_c1": float(t[i1]), "t_c2": float(t[i2])}

        # Gate 2: a pair whose joint score sits well below r_pair = 0.90 —
        # different (W,f) cells fit each lobe, no shared trapezoid.
        if 0.40 <= score < 0.75 and a_abs >= 0.10:
            found["low_score"].append(dict(rec))
        # Gate 3: a pair whose shape is reasonable but whose per-lobe
        # amplitudes are very asymmetric, so the shared mean drops below
        # a_pair = 0.30.
        small = min(A1, A2)
        big = max(A1, A2)
        if score >= 0.70 and big > 1e-3 and small / big <= 0.45 and a_abs < 0.35:
            asym = dict(rec)
            asym["asym"] = big / max(small, 1e-3)
            found["asym_amp"].append(asym)

        if score < cfg.joint_r2_thresh or a_abs < cfg.min_pair_abs_a:
            continue
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
    """F4 — a multi-stop trip in THREE stacked panels.

    Panel 1: the raw acceleration trace, no overlays — four visible lobes.
    Panel 2: the same trace with ONE wide horizontal arrow spanning lobe 1
             to lobe 4 (the "super pair"), labelled with its rank.
    Panel 3: the same trace with TWO short arrows, each spanning one stop
             ride, both labelled with their rank.

    Reader sees the reorganisation directly: with no penalty the super
    pair scores as well as either short ride, but ``rank = S - λΔt``
    docks long spans and the two short rides commit first.
    """
    state = ms["state"]
    pa, pb = ms["pa"], ms["pb"]
    t = state["t"]
    a_smooth = state["a_smooth"]
    a_vert = state["a_vert"]
    gw, gf = state["grid_w_s"], state["grid_f"]

    lobes = [pa["lobe1"], pa["lobe2"], pb["lobe1"], pb["lobe2"]]
    tcs = [float(lb["t_c"]) for lb in lobes]
    idxs = [int(np.argmin(np.abs(t - tc))) for tc in tcs]
    signs = [float(np.sign(lb["a_peak"])) for lb in lobes]

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
    win_s = a_smooth[m]
    ymax = max(float(np.max(np.abs(win_s))) if win_s.size else 1.0, 0.3) * 1.5

    fig, axes = plt.subplots(3, 1, figsize=(11.0, 7.6), sharex=True,
                             gridspec_kw={"hspace": 0.18})

    def _trace(ax):
        ax.plot(t[m] - t0, a_vert[m], color="#9aa6b2", lw=0.5,
                label=r"$a_\mathrm{vert}$")
        ax.plot(t[m] - t0, a_smooth[m], color="#2c3e50", lw=1.3,
                label="smoothed")
        ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.6)
        ax.set_ylim(-ymax, ymax)
        ax.set_ylabel(r"$a$ (m/s$^2$)")
        ax.grid(True, alpha=0.25)

    # Panel 1: bare trace, four lobes named at the top.
    _trace(axes[0])
    for n, tc in enumerate(tcs, start=1):
        axes[0].axvline(tc - t0, color="#d62728", lw=0.7, ls=":", alpha=0.55)
        axes[0].annotate(f"lobe {n}", (tc - t0, ymax * 0.94), fontsize=7,
                         ha="center", va="top", color="#d62728")
    axes[0].set_title("(a) raw trace — four lobes, no pairings drawn",
                      fontsize=9)
    axes[0].legend(fontsize=7, loc="upper right")

    # Panel 2: one big arrow spanning lobe 1 -> lobe 4 (the super pair).
    _trace(axes[1])
    y_arr = -ymax * 0.74
    axes[1].annotate(
        "", xy=(t_last - t0, y_arr), xytext=(t_first - t0, y_arr),
        arrowprops=dict(arrowstyle="<->", color="#b3261e", lw=2.2),
    )
    axes[1].text((t_first + t_last) / 2 - t0, y_arr - ymax * 0.10,
                 f"super pair (lobes 1$\\to$4):  "
                 f"$S = {superp['S']:.2f}$,  "
                 f"rank $= S - \\lambda\\,\\Delta t = {superp['rank']:.2f}$",
                 ha="center", va="top", fontsize=8, color="#b3261e")
    axes[1].set_title("(b) super-pair candidate — high $S$, but the duration "
                      "penalty docks $\\lambda\\,\\Delta t$", fontsize=9)

    # Panel 3: two short arrows, one per stop-ride.
    _trace(axes[2])
    for ri, (ai, bi, info, col) in enumerate((
        (0, 1, ride1, "#1a7a3a"),
        (2, 3, ride2, "#1a7a3a"),
    )):
        ta, tb = tcs[ai], tcs[bi]
        axes[2].annotate(
            "", xy=(tb - t0, y_arr), xytext=(ta - t0, y_arr),
            arrowprops=dict(arrowstyle="<->", color=col, lw=2.2),
        )
        axes[2].text((ta + tb) / 2 - t0, y_arr - ymax * 0.10,
                     f"ride {ri + 1}:  $S = {info['S']:.2f}$,  "
                     f"rank $= {info['rank']:.2f}$",
                     ha="center", va="top", fontsize=8, color=col)
    axes[2].set_title("(c) two short stop-rides — each rank beats the "
                      "super pair, so the greedy resolver commits these "
                      "first", fontsize=9)
    axes[2].set_xlabel("t (s, trip-local)")

    fig.suptitle(
        r"the duration penalty $\lambda$ splits one super pair into "
        "two short stop-rides",
        fontsize=10, y=0.995,
    )
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


# ---------------------------------------------------------------------------
# New renderers: Gate 2 (joint score), Gate 3 (shared amplitude),
# Gate 5 (cruise angle), trapezoid-fit failure
# ---------------------------------------------------------------------------

def _pair_panel(ax, state, t_c1, t_c2, W, f, A, s1, title, badge, badge_ok):
    """Trapezoid pulse-pair overlay used by Gate 2 and Gate 3 panels."""
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
    ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.6)
    win_s = a_smooth[m]
    ymax = max(float(np.max(np.abs(win_s))) if win_s.size else A, A, 0.1) * 1.45
    ax.set_ylim(-ymax, ymax)
    vcol = "#1a7a3a" if badge_ok else "#b3261e"
    ax.text(0.02, 0.04, badge, transform=ax.transAxes, ha="left", va="bottom",
            fontsize=8, bbox=dict(facecolor="white", alpha=0.9, edgecolor=vcol,
                                  boxstyle="round,pad=0.3"))
    ax.set_xlabel("t (s, pair-local)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.set_title(title, fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")


def render_gate2_examples(high: dict, low: dict, out: Path) -> None:
    """Gate 2 — high vs low shared-shape joint score.

    Left panel: a real pair with score well above the r_pair = 0.90 floor;
    one (W,f) trapezoid fits both lobes.
    Right panel: a real pair with score in [0.40, 0.75]; no single (W,f)
    fits both lobes.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # Left: a high-score accepted ride from `rich`.
    sh = high["state"]
    p = high["pred"]
    l1, l2 = p["lobe1"], p["lobe2"]
    s1 = 1.0 if p["ride_type"] == "up" else -1.0
    W, f = float(l1["half_width_s"]), float(l1["frac_flat"])
    A = abs(float(l1["a_peak"]))
    score = float(p.get("joint_r2_mean", p.get("joint_r2", 0.0)))
    _pair_panel(
        axes[0], sh, float(l1["t_c"]), float(l2["t_c"]), W, f, A, s1,
        title="genuine ride — one $(W,f)$ template fits both lobes",
        badge=f"$S = {score:.2f}\\ \\geq\\ r_\\mathrm{{pair}} = 0.90$\nACCEPTED",
        badge_ok=True,
    )

    # Right: a low-score candidate.
    sl = low["state"]
    _pair_panel(
        axes[1], sl, low["t_c1"], low["t_c2"], low["W"], low["f"], low["A"],
        low["s1"],
        title="false pair — each lobe peaks at a different $(W,f)$",
        badge=f"$S = {low['score']:.2f}\\ <\\ r_\\mathrm{{pair}} = 0.90$\n"
              f"per-lobe $R^2 = {low['r2_1']:.2f},\\ {low['r2_2']:.2f}$\nREJECTED",
        badge_ok=False,
    )
    fig.suptitle("Gate 2 — the shared-shape joint score discriminates true "
                 "rides from accidental pairings", fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def render_gate3_examples(symmetric: dict, asymmetric: dict, out: Path) -> None:
    """Gate 3 — symmetric vs asymmetric per-lobe amplitudes.

    Left panel: both lobe amplitudes clear the per-lobe floor and their
    mean A* ≥ 0.30 → ACCEPTED.
    Right panel: one lobe is firm, the other is weak; A* falls below
    a_pair = 0.30 → REJECTED, even though the joint shape is plausible.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 4.4))

    # Left: symmetric accepted ride.
    ss = symmetric["state"]
    p = symmetric["pred"]
    l1, l2 = p["lobe1"], p["lobe2"]
    s1 = 1.0 if p["ride_type"] == "up" else -1.0
    W, f = float(l1["half_width_s"]), float(l1["frac_flat"])
    A1 = abs(float(l1["a_peak"]))
    A2 = abs(float(l2["a_peak"]))
    Astar = 0.5 * (A1 + A2)
    _pair_panel(
        axes[0], ss, float(l1["t_c"]), float(l2["t_c"]), W, f, Astar, s1,
        title="both lobes firm — shared amplitude clears the floor",
        badge=f"$A_1 = {A1:.2f}$,  $A_2 = {A2:.2f}$  "
              f"m/s$^2$\n$A^\\star = {Astar:.2f}\\ \\geq\\ "
              f"a_\\mathrm{{pair}} = 0.30$\nACCEPTED",
        badge_ok=True,
    )

    # Right: asymmetric rejected candidate.
    sa = asymmetric["state"]
    A1a, A2a = asymmetric["A1"], asymmetric["A2"]
    Astar_a = 0.5 * (A1a + A2a)
    _pair_panel(
        axes[1], sa, asymmetric["t_c1"], asymmetric["t_c2"], asymmetric["W"],
        asymmetric["f"], Astar_a, asymmetric["s1"],
        title="one strong, one weak — mean amplitude falls under the floor",
        badge=f"$A_1 = {A1a:.2f}$,  $A_2 = {A2a:.2f}$  "
              f"m/s$^2$\n$A^\\star = {Astar_a:.2f}\\ <\\ "
              f"a_\\mathrm{{pair}} = 0.30$\nREJECTED",
        badge_ok=False,
    )
    fig.suptitle("Gate 3 — the shared amplitude $A^\\star$ rejects "
                 "asymmetric pairings the per-lobe floor lets through",
                 fontsize=10, y=1.02)
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def render_gate5_angle(ms: dict, out: Path) -> None:
    """Gate 5 — visualise the quiet-middle check as a line tilt.

    Take the first short ride of the multi-stop trip used by Gate 6, slice
    the inter-lobe cruise samples, fit a regression line, and annotate the
    line's angle. The deployed gate uses RMS_mid/A* ≤ 0.5, but visually
    the equivalent statement is "the cruise tilts less than ≈15°".
    """
    state = ms["state"]
    pa = ms["pa"]
    t = state["t"]
    a_vert = state["a_vert"]
    a_smooth = state["a_smooth"]

    l1, l2 = pa["lobe1"], pa["lobe2"]
    W = float(l1["half_width_s"])
    t_c1, t_c2 = float(l1["t_c"]), float(l2["t_c"])
    A = abs(float(l1["a_peak"]))

    # Plot a window that frames both lobes and the cruise in between.
    pad = max(1.5, 0.6 * W)
    lo, hi = t_c1 - W - pad, t_c2 + W + pad
    m = (t >= lo) & (t <= hi)
    t0 = lo

    # Regression line through the cruise window (t_c1 + W, t_c2 - W).
    cruise_lo, cruise_hi = t_c1 + W, t_c2 - W
    mc = (t >= cruise_lo) & (t <= cruise_hi)
    if not mc.any():
        print("  SKIP gate5_cruise_angle.png — empty cruise window")
        return
    tc = t[mc]
    ac = a_smooth[mc]
    slope, intercept = np.polyfit(tc, ac, 1)
    angle_deg = float(np.degrees(np.arctan(slope)))
    rms_val = float(np.sqrt(np.mean(ac * ac)))
    ratio = rms_val / max(A, 1e-6)
    accepted = ratio <= 0.5

    fig, ax = plt.subplots(figsize=(9.2, 4.6))
    ax.plot(t[m] - t0, a_vert[m], color="#b9c2cc", lw=0.4, alpha=0.7,
            label=r"$a_\mathrm{vert}$")
    ax.plot(t[m] - t0, a_smooth[m], color="#2c3e50", lw=1.4, label="smoothed")
    ax.axvspan(cruise_lo - t0, cruise_hi - t0, color="#1f77b4", alpha=0.12,
               label="inter-lobe cruise")
    # Regression line.
    tt = np.linspace(cruise_lo, cruise_hi, 80)
    ax.plot(tt - t0, slope * tt + intercept, color="#d62728", lw=2.2,
            label=f"regression  $\\theta = {angle_deg:+.1f}^\\circ$")
    # Horizontal reference at the line's mean.
    mean_a = float(np.mean(ac))
    ax.plot([cruise_lo - t0, cruise_hi - t0], [mean_a, mean_a],
            color="#888", lw=1.0, ls="--", label=r"horizontal reference")
    ax.axhline(0, color="gray", lw=0.4, ls=":", alpha=0.6)
    ax.set_ylim(-1.45 * A, 1.45 * A)
    badge_col = "#1a7a3a" if accepted else "#b3261e"
    verdict = "ACCEPTED" if accepted else "REJECTED"
    ax.text(
        0.02, 0.04,
        f"$|\\theta| = {abs(angle_deg):.1f}^\\circ$  "
        f"($\\leq 15^\\circ$ intuition envelope)\n"
        f"$\\mathrm{{RMS}}_\\mathrm{{mid}}/A^\\star = {ratio:.2f}$  "
        f"($\\leq \\rho = 0.5$)\n{verdict}",
        transform=ax.transAxes, ha="left", va="bottom", fontsize=8,
        bbox=dict(facecolor="white", alpha=0.92, edgecolor=badge_col,
                  boxstyle="round,pad=0.35"),
    )
    ax.set_xlabel("t (s, ride-local)")
    ax.set_ylabel(r"$a$ (m/s$^2$)")
    ax.set_title("Gate 5 — fitting a line through the cruise: a flat ride "
                 "tilts less than $\\approx 15^\\circ$", fontsize=9)
    ax.grid(True, alpha=0.25)
    ax.legend(fontsize=7, loc="upper right")
    fig.tight_layout()
    fig.savefig(out, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out.name}")


def render_trapezoid_fit_failure(tilted: dict, out: Path) -> None:
    """§3.2.2 trapezoid pulse-pair fit — a bad-fit exemplar.

    The same `tilted` pick that used to power the right panel of
    threshold_quiet_middle.png. When the cruise wobbles, the closed-form
    integral of the trapezoid template no longer recovers ΔH reliably.
    """
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    st = tilted["state"]
    _cruise_panel(ax, st, tilted["t_c1"], tilted["t_c2"], tilted["W"],
                  tilted["f"], tilted["A"], tilted["s1"], tilted["rms"],
                  tilted["ratio"], accepted=False)
    ax.set_title("when the cruise is not flat, the closed-form trapezoid "
                 "integral becomes unreliable", fontsize=9)
    fig.tight_layout()
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
    low_score, asym_amp = [], []
    for k, name in enumerate(names):
        res = scan_experiment(name)
        if res is None:
            continue
        rich += res["rich"]
        low += res["low_energy"]
        flat += res["flat"]
        tilted += res["tilted"]
        multistop += res["multistop"]
        low_score += res["low_score"]
        asym_amp += res["asym_amp"]
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
    # Gate 2: prefer the lowest-score pair that still has visible structure.
    low_score.sort(key=lambda r: r["score"])
    # Gate 3: largest amplitude asymmetry first, A* below the per-pair floor.
    asym_amp.sort(key=lambda r: -r.get("asym", 0.0))

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
    _show("G2 low-score pair", low_score,
          lambda r: f"S={r['score']:.2f} A={r['A']:.2f} "
                    f"r2=({r['r2_1']:.2f},{r['r2_2']:.2f})  {r['name']}")
    _show("G3 asymmetric amplitude", asym_amp,
          lambda r: f"A1={r['A1']:.2f} A2={r['A2']:.2f} A*={r['A']:.2f} "
                    f"asym={r['asym']:.1f}x  {r['name']}")

    if args.scan_only:
        return 0

    def _pick(tag, rows):
        if tag in OVERRIDES:
            want = OVERRIDES[tag][0]
            for r in rows:
                if r["name"] == want:
                    return r
        return rows[0] if rows else None

    # All figures need the full detector state of the chosen recording.
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
    pick_low_score = _pick("low_score", low_score)
    pick_asym = _pick("asym_amp", asym_amp)

    if pick_rich and pick_low:
        pick_rich["state"] = _state(pick_rich["name"])
        pick_low["state"] = _state(pick_low["name"])
        if pick_rich["state"] and pick_low["state"]:
            render_grid_energy(pick_rich, pick_low,
                               PAPER_FIG / "threshold_grid_energy.png")
    else:
        print("  SKIP threshold_grid_energy.png — missing a candidate")

    # Gate 2: high-score (reuse a `rich` accepted ride) vs low-score pair.
    if pick_rich and pick_low_score:
        pick_rich["state"] = pick_rich.get("state") or _state(pick_rich["name"])
        pick_low_score["state"] = _state(pick_low_score["name"])
        if pick_rich["state"] and pick_low_score["state"]:
            render_gate2_examples(pick_rich, pick_low_score,
                                  PAPER_FIG / "gate2_joint_score.png")
    else:
        print("  SKIP gate2_joint_score.png — missing a candidate")

    # Gate 3: symmetric accepted ride vs asymmetric rejected pair.
    if pick_rich and pick_asym:
        pick_rich["state"] = pick_rich.get("state") or _state(pick_rich["name"])
        pick_asym["state"] = _state(pick_asym["name"])
        if pick_rich["state"] and pick_asym["state"]:
            render_gate3_examples(pick_rich, pick_asym,
                                  PAPER_FIG / "gate3_shared_amplitude.png")
    else:
        print("  SKIP gate3_shared_amplitude.png — missing a candidate")

    if pick_tilted:
        pick_tilted["state"] = _state(pick_tilted["name"])
        if pick_tilted["state"]:
            render_trapezoid_fit_failure(
                pick_tilted, PAPER_FIG / "trapezoid_fit_failure.png",
            )
    else:
        print("  SKIP trapezoid_fit_failure.png — missing a candidate")

    if pick_ms:
        pick_ms["state"] = _state(pick_ms["name"])
        if pick_ms["state"]:
            render_duration_penalty(pick_ms,
                                    PAPER_FIG / "threshold_duration_penalty.png")
            # Gate 5 reuses the multi-stop trip's first short ride.
            render_gate5_angle(pick_ms, PAPER_FIG / "gate5_cruise_angle.png")
    else:
        print("  SKIP threshold_duration_penalty.png — missing a candidate")
    return 0


if __name__ == "__main__":
    sys.exit(main())
