"""Compare scoring strategies for trapezoid-vs-parabola velocity fits.

For each ride, both fits are computed. Four strategies re-score them and
pick a "winner". Each strategy produces its own ride_context grid PNG into
`run_results/` at the repo root.

Run:
    python3 -m src.tests.segmentations.compare_fit_strategies [experimenter]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    Segmenter,
)
from src.tests.segmentations.main import build_acc_frame, build_height_frame
from src.algorithms.segmentation_algorithms.template_match.scripts.velocity_templates import (
    fit_trapezoid, fit_parabola, ride_velocity,
)


OUT_DIR = Path(__file__).resolve().parents[1] / "results"
CONTEXT_PAD_SEC = 4.0


# ---------- Scoring strategies ----------
def score_active_mask(fit: dict, v: np.ndarray) -> float:
    """MAE only where |v_measured| > 10% of peak (focus on the hump)."""
    peak = np.max(np.abs(v))
    if peak <= 0:
        return fit["mae"]
    mask = np.abs(v) > 0.1 * peak
    if not mask.any():
        return fit["mae"]
    return float(np.mean(np.abs(fit["fit"][mask] - v[mask])))


def score_aic(fit: dict, v: np.ndarray) -> float:
    """MAE + parsimony penalty k/N. trapezoid k=4, parabola k=3."""
    k = 4 if fit["shape"] in ("trapezoid", "triangle") else 3
    return fit["mae"] + k / len(v)


def score_triangle_vs_parabola(fit: dict, v: np.ndarray) -> float:
    """If trapezoid has no real plateau (<0.5s), it's a triangle — use MAE.
    If it has a real plateau, trapezoid wins with a big bonus (parabola can't
    represent a plateau)."""
    if fit["shape"] == "parabola":
        return fit["mae"]
    # trapezoid fit
    ramp = fit["v_max"] / fit["a_max"]
    total_w = fit["t_end"] - fit["t_start"]
    plateau = max(total_w - 2 * ramp, 0.0)
    if plateau < 0.5:
        return fit["mae"]          # behaves as triangle, fair comparison
    return fit["mae"] * 0.1         # real plateau: trapezoid strongly preferred


def score_normalized_shape(fit: dict, v: np.ndarray) -> float:
    """MAE after normalizing both curves by their peak (shape-only)."""
    peak_v = np.max(np.abs(v))
    peak_f = np.max(np.abs(fit["fit"]))
    if peak_v <= 0 or peak_f <= 0:
        return fit["mae"]
    return float(np.mean(np.abs(fit["fit"] / peak_f - v / peak_v)))


def _has_plateau(v: np.ndarray, ts: np.ndarray,
                 peak_tol: float = 0.05, min_sec: float = 1.5,
                 min_frac: float = 0.20) -> bool:
    peak = float(np.max(np.abs(v)))
    if peak <= 0:
        return False
    flat = np.abs(np.abs(v) - peak) < peak_tol * peak
    # longest run of True
    best = cur = 0
    for f in flat:
        cur = cur + 1 if f else 0
        best = max(best, cur)
    if best < 2:
        return False
    dt = float(np.median(np.diff(ts))) if len(ts) > 1 else 0.01
    dur = best * dt
    ride_dur = float(ts[-1] - ts[0]) if len(ts) > 1 else dur
    return dur >= min_sec and dur >= min_frac * ride_dur


def score_plateau_gate(fit: dict, v: np.ndarray, ts: np.ndarray | None = None) -> float:
    """If a real plateau is present, trapezoid wins outright; else MAE."""
    has_pl = _has_plateau(v, ts) if ts is not None else False
    if has_pl:
        return fit["mae"] if fit["shape"] in ("trapezoid", "triangle") else fit["mae"] + 1.0
    return fit["mae"]


def score_bounded_p(fit: dict, v: np.ndarray) -> float:
    """Penalize parabola if its shape exponent p drifted far from 1 (mimicking plateau)."""
    if fit["shape"] == "parabola":
        p = fit.get("p", 1.0)
        # penalty rises steeply when p<0.8 (plateau-mimic) or p>3 (spike)
        if p < 0.8:
            return fit["mae"] + 0.5 * (0.8 - p)
        if p > 3.0:
            return fit["mae"] + 0.1 * (p - 3.0)
    return fit["mae"]


STRATEGIES = {
    "A_active_mask":        ("MAE on active region (|v|>10% peak)",         score_active_mask),
    "B_aic":                ("MAE + parsimony penalty (k/N)",               score_aic),
    "C_triangle_vs_par":    ("Triangle-vs-parabola when no plateau",        score_triangle_vs_parabola),
    "D_normalized_shape":   ("MAE on shape-normalized curves",              score_normalized_shape),
    "E_plateau_gate":       ("Preproc: detect plateau -> force trapezoid",  score_plateau_gate),
    "F_bounded_p":          ("Penalize parabola p<0.8 (plateau-mimic)",     score_bounded_p),
}


def render_strategy(name_key: str, desc: str, scorer, rides, t_full, mag_full,
                    h_t, h_val, name: str) -> None:
    n = len(rides)
    ncols = 3
    nrows = int(np.ceil(n / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.8 * nrows), squeeze=False)
    trap_wins = par_wins = 0
    for k, r in enumerate(rides):
        ax = axes[k // ncols][k % ncols]
        t_lo = r["t_start"] - CONTEXT_PAD_SEC
        t_hi = r["t_end"] + CONTEXT_PAD_SEC
        mctx = (t_full >= t_lo) & (t_full <= t_hi)
        ax.plot(t_full[mctx], mag_full[mctx], color="0.55", lw=0.7, label="|acc|")
        ax.axvspan(r["t_start"], r["t_end"], color="yellow", alpha=0.18)

        ax2 = ax.twinx()
        ride_color = "tab:blue" if r["type"] == "up" else "tab:red"
        ax2.plot(r["ts"], r["v"], color=ride_color, lw=1.5, label="vel")
        ax2.axhline(0, color="k", lw=0.3)

        trap = r["trap"]; par = r["par"]
        import inspect
        needs_ts = "ts" in inspect.signature(scorer).parameters
        if needs_ts:
            trap_s = scorer(trap, r["v"], r["ts"]) if trap.get("ok") else np.inf
            par_s  = scorer(par,  r["v"], r["ts"]) if par.get("ok")  else np.inf
        else:
            trap_s = scorer(trap, r["v"]) if trap.get("ok") else np.inf
            par_s  = scorer(par,  r["v"]) if par.get("ok")  else np.inf

        if trap.get("ok"):
            ax2.plot(r["ts"], trap["fit"], color="black", lw=1.1,
                     linestyle="-", alpha=0.9)
        if par.get("ok"):
            ax2.plot(r["ts"], par["fit"], color="purple", lw=1.1,
                     linestyle="--", alpha=0.9)

        if trap_s <= par_s:
            winner = f"TRAP ({trap.get('shape','?')})"
            trap_wins += 1
        else:
            winner = "PARABOLA"
            par_wins += 1

        hctx = (h_t >= t_lo) & (h_t <= t_hi)
        if hctx.any():
            h_seg = h_val[hctx]
            rng = h_seg.max() - h_seg.min()
            if rng > 0:
                mag_lo, mag_hi = mag_full[mctx].min(), mag_full[mctx].max()
                h_scaled = (h_seg - h_seg.min()) / rng * (mag_hi - mag_lo) + mag_lo
                ax.plot(h_t[hctx], h_scaled, color="green", lw=0.7, alpha=0.6)

        info = (
            f"winner: {winner}\n"
            f"trap  score={trap_s:.4f}  ({trap.get('shape','?')})\n"
            f"par   score={par_s:.4f}"
        )
        ax.text(0.02, 0.97, info, transform=ax.transAxes, fontsize=6.5,
                va="top", ha="left",
                bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"))
        ax.set_title(f"Ride {r['idx']} — {r['type']}", fontsize=9)
        ax.set_xlabel("time (s)", fontsize=8)

    for j in range(n, nrows * ncols):
        axes[j // ncols][j % ncols].axis("off")
    fig.suptitle(f"Strategy {name_key}: {desc}   |   trap={trap_wins}  par={par_wins}   ({name})",
                 fontsize=11)
    fig.tight_layout()
    out = OUT_DIR / f"fit_strategy_{name_key}_{name}.png"
    fig.savefig(out, dpi=110)
    plt.close(fig)
    print(f"  {name_key}: trap_wins={trap_wins}  par_wins={par_wins}  -> {out.name}")


def main(name: str = "uriya") -> None:
    data = load_experimenter(name)
    t0_ms = float(data["ACC"]["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(data["ACC"], t0_ms)
    height_frame = build_height_frame(data["PRS"], t0_ms)

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    segments = Segmenter(cfg).detect(height_frame)
    print(f"{len(segments)} GT segments")

    rides = []
    for i, row in segments.iterrows():
        t_start = row["start_ci"][0]; t_end = row["end_ci"][1]
        ts, v, _ = ride_velocity(acc_frame, t_start, t_end)
        if v.size == 0:
            continue
        rides.append({
            "idx": int(i), "ts": ts, "v": v, "type": row["type"],
            "t_start": t_start, "t_end": t_end,
            "trap": fit_trapezoid(ts, v), "par": fit_parabola(ts, v),
        })

    OUT_DIR.mkdir(exist_ok=True)
    t_full = acc_frame["time"].to_numpy()
    mag_full = acc_frame["mag"].to_numpy()
    h_t = height_frame["time"].to_numpy()
    h_val = height_frame["height"].to_numpy()

    for key, (desc, scorer) in STRATEGIES.items():
        render_strategy(key, desc, scorer, rides, t_full, mag_full, h_t, h_val, name)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "uriya")
