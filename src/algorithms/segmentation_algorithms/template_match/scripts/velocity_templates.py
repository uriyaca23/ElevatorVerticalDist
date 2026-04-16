"""Extract GT per-ride velocity curves, low-pass filter, split at peak,
normalize, and plot up vs down spikes. Also plot each ride's context window
with the extracted spike overlaid.

Run:
    python -m src.tests.segmentations.velocity_templates [experimenter]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt
from scipy.optimize import minimize


def trapezoid_velocity(t: np.ndarray, t_start: float, t_end: float,
                       a_max: float, v_max: float) -> np.ndarray:
    """Symmetric trapezoidal velocity profile.

    v(t) = clip( a_max * min(t - t_start, t_end - t), 0, v_max )

    - If 2*v_max/a_max < (t_end - t_start): true trapezoid (reaches plateau).
    - Else: triangle / gaussian-like (peak = a_max*(t_end-t_start)/2 < v_max).
    """
    a_max = max(a_max, 1e-6)
    v_max = max(v_max, 1e-6)
    ramp = a_max * np.minimum(t - t_start, t_end - t)
    return np.clip(ramp, 0.0, v_max)


def fit_trapezoid(ts: np.ndarray, v: np.ndarray) -> dict:
    """Fit trapezoid to a (signed) velocity curve. Returns params and fitted curve."""
    sign = 1.0 if np.max(v) >= -np.min(v) else -1.0
    v_abs = v * sign

    peak = float(np.max(v_abs))
    if peak <= 0:
        return {"ok": False}
    # initial guesses
    above = v_abs > 0.1 * peak
    if not above.any():
        return {"ok": False}
    t_s0 = float(ts[above][0])
    t_e0 = float(ts[above][-1])
    dur0 = max(t_e0 - t_s0, 0.5)
    a0 = peak / (0.25 * dur0)
    v0 = peak

    lo = np.array([ts[0] - 2.0, ts[0] + 0.2, 0.05, 0.05])
    hi = np.array([ts[-1] - 0.2, ts[-1] + 2.0, 20.0, 10.0])

    def cost(p):
        t_s, t_e, a_m, v_m = p
        penalty = 0.0
        for val, l, h in zip(p, lo, hi):
            if val < l: penalty += 1e3 * (l - val)
            elif val > h: penalty += 1e3 * (val - h)
        if t_e <= t_s + 0.1:
            penalty += 1e3
        pred = trapezoid_velocity(ts, t_s, t_e, a_m, v_m)
        return np.mean(np.abs(pred - v_abs)) + penalty

    try:
        res = minimize(cost, x0=[t_s0, t_e0, a0, v0],
                       method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 4000})
    except Exception:
        return {"ok": False}

    t_s, t_e, a_m, v_m = res.x
    fit = sign * trapezoid_velocity(ts, t_s, t_e, a_m, v_m)
    peak_model = a_m * (t_e - t_s) / 2
    shape = "trapezoid" if peak_model > v_m else "triangle"
    return {
        "ok": True,
        "fit": fit,
        "t_start": float(t_s),
        "t_end": float(t_e),
        "a_max": float(a_m),
        "v_max": float(v_m),
        "sign": sign,
        "shape": shape,
        "mae": float(np.mean(np.abs(fit - v))),
    }


def parabola_velocity(t: np.ndarray, t_c: float, W: float, v_peak: float,
                      p: float = 1.0) -> np.ndarray:
    """Generalized parabola with shape exponent p:
        v(t) = v_peak * max(0, 1 - ((t - t_c)/W)^2)^p
    p=1 -> classic parabola, p<1 -> flatter top, p>1 -> narrower peak.
    """
    W = max(W, 1e-3)
    u = (t - t_c) / W
    core = 1.0 - u * u
    core = np.maximum(core, 0.0)
    return v_peak * np.power(core, max(p, 1e-3))


def fit_parabola(ts: np.ndarray, v: np.ndarray) -> dict:
    """Fit generalized parabola v = v_peak*(1 - ((t-t_c)/W)^2)^p to |v|."""
    sign = 1.0 if np.max(v) >= -np.min(v) else -1.0
    v_abs = v * sign
    peak = float(np.max(v_abs))
    if peak <= 0:
        return {"ok": False}
    above = v_abs > 0.1 * peak
    if not above.any():
        return {"ok": False}
    t_c0 = float(ts[int(np.argmax(v_abs))])
    W0 = max(0.5, float(ts[above][-1] - ts[above][0]) / 2.0)

    lo = np.array([ts[0] - 2.0, 0.3,  0.05, 0.2])
    hi = np.array([ts[-1] + 2.0, 30.0, 10.0, 5.0])

    def cost(p):
        t_c, W, vp, pw = p
        penalty = 0.0
        for val, l, h in zip(p, lo, hi):
            if val < l: penalty += 1e3 * (l - val)
            elif val > h: penalty += 1e3 * (val - h)
        pred = parabola_velocity(ts, t_c, W, vp, pw)
        return np.mean(np.abs(pred - v_abs)) + penalty

    try:
        res = minimize(cost, x0=[t_c0, W0, peak, 1.0],
                       method="Nelder-Mead",
                       options={"xatol": 1e-3, "fatol": 1e-5, "maxiter": 5000})
    except Exception:
        return {"ok": False}
    t_c, W, vp, pw = res.x
    fit = sign * parabola_velocity(ts, t_c, W, vp, pw)
    # equivalent a*t^2 + b*t coefficients (only exact when p=1, else reported as local quadratic at t_c)
    a_coef = -vp / (W * W) if pw == 1 else None
    return {
        "ok": True,
        "fit": fit,
        "t_c": float(t_c),
        "W": float(2 * W),       # full width (root-to-root)
        "v_peak": float(vp),
        "p": float(pw),
        "a": float(a_coef) if a_coef is not None else None,
        "b": None,
        "a_max": float(2 * vp / W),  # max slope at edge for p=1
        "sign": sign,
        "shape": "parabola",
        "mae": float(np.mean(np.abs(fit - v))),
    }


def fit_best(ts: np.ndarray, v: np.ndarray) -> dict:
    """Fit trapezoid/triangle and parabola; return the lower-RMSE one."""
    t = fit_trapezoid(ts, v)
    p = fit_parabola(ts, v)
    candidates = [c for c in (t, p) if c.get("ok")]
    if not candidates:
        return {"ok": False}
    return min(candidates, key=lambda c: c["mae"])

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    Segmenter,
)
from src.tests.segmentations.main import build_acc_frame, build_height_frame


# New home: src/algorithms/segmentation_algorithms/template_match/results/
OUT_DIR = Path(__file__).resolve().parents[1] / "results"
LPF_CUTOFF_HZ = 0.5   # elevator velocity profile is slow (seconds-scale)
LPF_ORDER = 3
CONTEXT_PAD_SEC = 4.0  # seconds before/after the GT window to show as context


def lowpass(x: np.ndarray, fs: float, cutoff: float = LPF_CUTOFF_HZ,
            order: int = LPF_ORDER) -> np.ndarray:
    if x.size < 9:
        return x
    nyq = 0.5 * fs
    wn = min(0.99, cutoff / nyq)
    b, a = butter(order, wn, btype="low")
    return filtfilt(b, a, x)


def ride_velocity(acc_frame, t_start: float, t_end: float) -> tuple[np.ndarray, np.ndarray, float]:
    t = acc_frame["time"].to_numpy()
    mask = (t >= t_start) & (t <= t_end)
    ts = t[mask]
    if ts.size < 3:
        return np.empty(0), np.empty(0), 0.0
    mag = acc_frame["mag"].to_numpy()[mask]
    a_lin = mag - mag.mean()
    dt = np.diff(ts, prepend=ts[0])
    vel = np.cumsum(a_lin * dt)
    n = len(vel)
    vel = vel - np.linspace(0.0, vel[-1], n)  # ZUPT linear detrend

    dt_pos = dt[dt > 0]
    fs = 1.0 / np.median(dt_pos) if dt_pos.size else 100.0
    vel_lpf = lowpass(vel, fs)
    return ts, vel_lpf, fs


def main(name: str = "roy_turgman") -> None:
    data = load_experimenter(name)
    t0_ms = float(data["ACC"]["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(data["ACC"], t0_ms)
    height_frame = build_height_frame(data["PRS"], t0_ms)

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    segments = Segmenter(cfg).detect(height_frame)
    print(f"{len(segments)} GT segments")

    t_full = acc_frame["time"].to_numpy()
    mag_full = acc_frame["mag"].to_numpy()
    h_t = height_frame["time"].to_numpy()
    h_val = height_frame["height"].to_numpy()

    rides = []
    for i, row in segments.iterrows():
        t_start = row["start_ci"][0]
        t_end = row["end_ci"][1]
        ts, v, fs = ride_velocity(acc_frame, t_start, t_end)
        if v.size == 0:
            continue
        peak_idx = int(np.argmax(np.abs(v)))
        rides.append({
            "idx": int(i),
            "ts": ts, "v": v, "fs": fs,
            "type": row["type"],
            "peak_idx": peak_idx,
            "t_start": t_start, "t_end": t_end,
        })

    OUT_DIR.mkdir(exist_ok=True)

    # ---- Plot 1: all GT (LPF) velocities ----
    fig, ax = plt.subplots(figsize=(10, 5))
    seen = set()
    for r in rides:
        color = "tab:blue" if r["type"] == "up" else "tab:red"
        lbl = r["type"] if r["type"] not in seen else None
        seen.add(r["type"])
        t_rel = r["ts"] - r["ts"][0]
        ax.plot(t_rel, r["v"], color=color, alpha=0.6, label=lbl)
        ax.axvline(t_rel[r["peak_idx"]], color=color, alpha=0.2, linestyle=":")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xlabel("time from ride start (s)")
    ax.set_ylabel("velocity (m/s)")
    ax.set_title(f"GT per-ride velocities (LPF {LPF_CUTOFF_HZ} Hz) — {name}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"gt_velocities_lpf_{name}.png", dpi=120)
    plt.close(fig)

    # ---- Plot 2: normalized up/down spikes aligned at peak ----
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)
    for r in rides:
        v = r["v"]
        vmax = np.max(np.abs(v))
        if vmax == 0:
            continue
        v_norm = v / vmax
        t_rel = r["ts"] - r["ts"][0]
        t_aligned = t_rel - t_rel[r["peak_idx"]]
        ax = axes[0] if r["type"] == "up" else axes[1]
        ax.plot(t_aligned, v_norm, alpha=0.7)
    axes[0].set_title("UP rides — normalized LPF velocity (peak at t=0)")
    axes[1].set_title("DOWN rides — normalized LPF velocity (peak at t=0)")
    for ax in axes:
        ax.axvline(0, color="k", lw=0.8, linestyle="--", label="split (peak)")
        ax.axhline(0, color="k", lw=0.5)
        ax.set_xlabel("time from peak (s)")
        ax.legend()
    axes[0].set_ylabel("velocity / max|v|")
    fig.suptitle(f"Up vs Down normalized velocity spikes — {name}")
    fig.tight_layout()
    fig.savefig(OUT_DIR / f"velocity_spikes_up_down_{name}.png", dpi=120)
    plt.close(fig)

    # ---- Plot 3: per-ride context (original signal + extracted spike) ----
    n = len(rides)
    if n > 0:
        ncols = 3
        nrows = int(np.ceil(n / ncols))
        fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 2.8 * nrows),
                                 squeeze=False)
        for k, r in enumerate(rides):
            ax = axes[k // ncols][k % ncols]
            t_lo = r["t_start"] - CONTEXT_PAD_SEC
            t_hi = r["t_end"] + CONTEXT_PAD_SEC

            # Original acc magnitude context (left y)
            mctx = (t_full >= t_lo) & (t_full <= t_hi)
            ax.plot(t_full[mctx], mag_full[mctx], color="0.55", lw=0.8,
                    label="|acc|")
            ax.axvspan(r["t_start"], r["t_end"], color="yellow", alpha=0.2,
                       label="GT window")

            # Ride velocity (LPF) on twin axis
            ax2 = ax.twinx()
            color = "tab:blue" if r["type"] == "up" else "tab:red"
            ax2.plot(r["ts"], r["v"], color=color, lw=1.6,
                     label=f"vel ({r['type']})")
            ax2.axvline(r["ts"][r["peak_idx"]], color=color, lw=0.8,
                        linestyle="--", alpha=0.7)
            ax2.axhline(0, color="k", lw=0.3)

            # Overlay BOTH fits: trapezoid/triangle and parabola (y=at²+bt)
            trap = fit_trapezoid(r["ts"], r["v"])
            par = fit_parabola(r["ts"], r["v"])
            lines = []
            if trap.get("ok"):
                ax2.plot(r["ts"], trap["fit"], color="black", lw=1.2,
                         linestyle="-", alpha=0.9,
                         label=f"trapezoid/triangle")
                lines.append(
                    f"{trap['shape']}  mae={trap['mae']:.3f}\n"
                    f"  a_max={trap['a_max']:.2f}  v_max={trap['v_max']:.2f}\n"
                    f"  W={trap['t_end']-trap['t_start']:.2f}s"
                )
            if par.get("ok"):
                ax2.plot(r["ts"], par["fit"], color="purple", lw=1.2,
                         linestyle="--", alpha=0.9, label="parabola")
                lines.append(
                    f"parabola  mae={par['mae']:.3f}\n"
                    f"  a={par['a']:.3f}  b={par['b']:.3f}\n"
                    f"  v_peak={par['v_peak']:.2f}  W={par['W']:.2f}s"
                )
            if lines:
                ax.text(
                    0.02, 0.97, "\n".join(lines),
                    transform=ax.transAxes, fontsize=6.5,
                    va="top", ha="left",
                    bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
                )

            # GT height faint on same window
            hctx = (h_t >= t_lo) & (h_t <= t_hi)
            if hctx.any():
                h_seg = h_val[hctx]
                # rescale height to acc axis for overlay readability
                h_range = h_seg.max() - h_seg.min()
                if h_range > 0:
                    mag_lo, mag_hi = mag_full[mctx].min(), mag_full[mctx].max()
                    h_scaled = (h_seg - h_seg.min()) / h_range * (mag_hi - mag_lo) + mag_lo
                    ax.plot(h_t[hctx], h_scaled, color="green", lw=0.8,
                            alpha=0.6, label="GT height (scaled)")

            ax.set_title(f"Ride {r['idx']} — {r['type']}")
            ax.set_xlabel("time (s)")
            ax.set_ylabel("|acc| (m/s²)")
            ax2.set_ylabel("vel (m/s)", color=color)
            ax.tick_params(axis="y")
            ax2.tick_params(axis="y", colors=color)
            if k == 0:
                lines1, lab1 = ax.get_legend_handles_labels()
                lines2, lab2 = ax2.get_legend_handles_labels()
                ax.legend(lines1 + lines2, lab1 + lab2, fontsize=7, loc="upper right")

        for j in range(n, nrows * ncols):
            axes[j // ncols][j % ncols].axis("off")

        fig.suptitle(f"Per-ride context: original data + extracted LPF velocity spike — {name}")
        fig.tight_layout()
        fig.savefig(OUT_DIR / f"ride_context_{name}.png", dpi=110)
        plt.close(fig)

    print(f"Saved plots to {OUT_DIR}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "roy_turgman")
