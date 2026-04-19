"""Independent per-lobe trapezoid fit via matched-filter grid search.

Motivation. The earlier joint-fit flavour of this module forced both lobes
of a ride to share ``(A, W, f)`` and used ``scipy.optimize.curve_fit`` to
find the best pair. In practice the two physical lobes often have
slightly different shapes (noise, orientation drift, door impulses), so
the shared-parameter optimum pushes one lobe to the edge of the ride
window and reports a high R² that doesn't visually match the signal.

This module instead fits each lobe **independently**:

  1. Build a 2-D grid of trapezoid shapes ``(W, f)``.
  2. For every ride, for each of the two lobes, cross-correlate every
     template against the gravity-projected vertical accelerometer
     ``a_vert`` within that lobe's half of the GT window.
  3. Pick the ``(t_c, A, W, f)`` that maximises the local coefficient of
     determination under the sign expected for that lobe:

        up ride:   lobe 1 = +A (take-off),   lobe 2 = -A (landing)
        down ride: lobe 1 = -A (take-off),   lobe 2 = +A (landing)

The fit for lobe 1 and lobe 2 are fully independent — different
amplitudes, different widths, different plateau fractions — so the
second lobe no longer has to live with the shape of the first.

Outputs (one per experiment) under
``template_match/labels/fit_elevator_paramater/<exp>/``:

  * ``_all_rides.png`` — small-multiples grid. Top: ``a_vert`` with both
    fitted trapezoids overlaid (red). Bottom: smoothed ``vz``.
  * ``parameters.json`` — per-ride, per-lobe ``(t_c, A, W, f, r2)``.

Run:
    venv/bin/python -m src.segmentation.algorithms.accelerometer_only.\
template_match.fit_trapezoid_pulses
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import dataclass, asdict, field
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.physics import calculate_velocity_from_accelerometer  # noqa: E402
from src.prediction.algorithms.quality_filter import estimate_gravity_vector  # noqa: E402

OUT_ROOT = Path(__file__).with_name("labels") / "fit_elevator_paramater" / "basicTreepzeGrid"
CONTEXT_PAD_FRAC = 0.15
CONTEXT_PAD_MIN_S = 0.5
SMOOTH_SEC = 0.4
TYPE_COLORS = {"up": "#27ae60", "down": "#e74c3c"}

# --------------------------------------------------------------------------
# Grid over trapezoid shape. ``W`` is the half-width (zero-crossing to
# center); ``f`` is the fraction of the half-width spent at the saturated
# peak. 30 × 15 = 450 templates is already well beyond the signal's
# resolving power at ~100 Hz and fits every ride in under a second.
# --------------------------------------------------------------------------
GRID_W_S = np.linspace(0.4, 3.0, 30)
GRID_F = np.linspace(0.0, 0.80, 15)

# Each lobe's search region expressed as fractions of the GT ride duration
# measured from `gt_t0`. Small overlap so lobes near the midpoint are
# reachable from either side.
LOBE1_REGION = (0.00, 0.60)
LOBE2_REGION = (0.40, 1.00)


@dataclass
class LobeFit:
    """Best-matching trapezoid for a single lobe."""

    t_c: float | None = None
    a_peak: float | None = None   # SIGNED amplitude
    half_width_s: float | None = None
    frac_flat: float | None = None
    r2_local: float | None = None  # 1 - SS_res / SS_tot over the ±W window


@dataclass
class RideFit:
    """Per-ride independent per-lobe fits."""

    index: int
    ride_type: str
    duration_s: float
    lobe1: LobeFit = field(default_factory=LobeFit)
    lobe2: LobeFit = field(default_factory=LobeFit)
    lobe_centroid_spacing_s: float | None = None


# --------------------------------------------------------------------------
# Signal preprocessing (unchanged from the joint-fit flavour)
# --------------------------------------------------------------------------

def _estimate_fs_hz(ts_ms: np.ndarray, default: float = 100.0) -> float:
    if ts_ms.size < 2:
        return default
    dt_ms = float(np.median(np.diff(ts_ms)))
    return default if dt_ms <= 0 else 1000.0 / dt_ms


def _vertical_accel(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    gvec, g_mag, _stab = estimate_gravity_vector(ax, ay, az, fs=fs, window_sec=0.5)
    g_hat = gvec / (np.linalg.norm(gvec) + 1e-12)
    return ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag


def trapezoid_kernel(t: np.ndarray, t_c: float, W: float, frac_flat: float) -> np.ndarray:
    """Unit-amplitude symmetric trapezoid kernel."""
    frac_flat = max(0.0, min(1.0, float(frac_flat)))
    W = max(1e-6, float(W))
    flat_half = frac_flat * W
    ramp_width = W - flat_half + 1e-9
    dt = np.abs(t - t_c)
    return np.where(
        dt <= flat_half, 1.0,
        np.where(dt < W, (W - dt) / ramp_width, 0.0),
    )


def _smooth(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    w = max(3, int(round(seconds * fs)))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()


# --------------------------------------------------------------------------
# Matched-filter grid search
# --------------------------------------------------------------------------

def _match_one_template(
    a: np.ndarray, t: np.ndarray, W: float, frac_flat: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Slide a unit trapezoid of shape ``(W, frac_flat)`` over signal ``a``.

    Returns
    -------
    A_hat : np.ndarray, shape (N,)
        Best-fit signed amplitude at each center index (under a
        least-squares local fit ``a ≈ A · template`` on the ±W window).
    r2_local : np.ndarray, shape (N,)
        Local coefficient of determination in the same ±W window. NaN
        where the window falls off the signal.
    """
    n = a.size
    if n == 0:
        return np.zeros(0), np.zeros(0)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 100.0
    K = max(3, int(round(2 * W / dt)))  # samples per ±W window (odd preferred)
    if K % 2 == 0:
        K += 1
    half = K // 2

    # Template on the window's local axis.
    t_kernel = (np.arange(K) - half) * dt
    tpl = trapezoid_kernel(t_kernel, 0.0, W, frac_flat)
    norm_t = float(np.sum(tpl * tpl))
    if norm_t < 1e-9:
        return np.full(n, np.nan), np.full(n, np.nan)

    # Cross-correlation at each center. mode='same' returns length-n;
    # positions within `half` of either edge are biased — we mask later.
    inner = np.convolve(a, tpl[::-1], mode="same")

    # Rolling window power of the signal.
    a2 = a * a
    csum = np.concatenate(([0.0], np.cumsum(a2)))
    local_power = np.full(n, np.nan)
    for i in range(half, n - half):
        local_power[i] = csum[i + half + 1] - csum[i - half]

    A_hat = np.full(n, np.nan)
    valid = np.isfinite(local_power)
    A_hat[valid] = inner[valid] / norm_t

    # SS_res = <a,a> - 2 A <a,tpl> + A² <tpl,tpl>
    #        = <a,a> - A·inner   when  A = inner / norm_t.
    r2_local = np.full(n, np.nan)
    denom = local_power[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = denom - A_hat[valid] * inner[valid]
        r2 = 1.0 - ss_res / np.where(denom > 1e-9, denom, np.nan)
    r2_local[valid] = r2
    return A_hat, r2_local


def _grid_search_lobe(
    a: np.ndarray, t: np.ndarray, center_lo: float, center_hi: float,
    target_sign: float, grid_W: np.ndarray, grid_F: np.ndarray,
) -> LobeFit:
    """Best-matching trapezoid across the ``(W, f)`` grid within the
    ``[center_lo, center_hi]`` center window, restricted to the sign of
    ``target_sign``."""
    n = a.size
    if n == 0:
        return LobeFit()

    # Boolean mask over sample indices whose *center* lies in the lobe region.
    in_region = (t >= center_lo) & (t <= center_hi)
    if not in_region.any():
        return LobeFit()

    best: LobeFit = LobeFit()
    best_score = -np.inf

    for W in grid_W:
        for f in grid_F:
            A_hat, r2 = _match_one_template(a, t, float(W), float(f))
            if not np.isfinite(r2).any():
                continue
            # Constrain sign of the recovered amplitude.
            mask = in_region & np.isfinite(r2) & (np.sign(A_hat) == target_sign)
            if not mask.any():
                continue
            # Pick the best r² in the allowed region.
            idx_candidates = np.where(mask)[0]
            # np.argmax can throw on nan — but we've already masked nan out.
            best_idx = idx_candidates[np.argmax(r2[idx_candidates])]
            score = float(r2[best_idx])
            if score > best_score:
                best_score = score
                best = LobeFit(
                    t_c=float(t[best_idx]),
                    a_peak=float(A_hat[best_idx]),
                    half_width_s=float(W),
                    frac_flat=float(f),
                    r2_local=float(score),
                )
    return best


def _fit_ride(
    t_ride: np.ndarray, a_vert_ride: np.ndarray,
    gt_t0: float, gt_t1: float,
    ride_idx: int, ride_type: str, fs: float,
) -> RideFit:
    """Independent per-lobe trapezoid fit over a single ride window."""
    duration = float(gt_t1 - gt_t0)
    fail = RideFit(index=ride_idx, ride_type=ride_type, duration_s=duration)
    if t_ride.size < 8 or duration <= 0:
        return fail

    a_smooth = _smooth(a_vert_ride, fs, SMOOTH_SEC)

    # Sign convention: up ride = (+, -); down ride = (-, +).
    ride_sign = 1.0 if ride_type == "up" else -1.0
    sign_lobe1 = +1.0 * ride_sign
    sign_lobe2 = -1.0 * ride_sign

    lo1 = gt_t0 + LOBE1_REGION[0] * duration
    hi1 = gt_t0 + LOBE1_REGION[1] * duration
    lo2 = gt_t0 + LOBE2_REGION[0] * duration
    hi2 = gt_t0 + LOBE2_REGION[1] * duration

    # Cap W by a fraction of the ride duration so the template can't wrap
    # around the whole ride and match a DC offset.
    W_cap = 0.5 * duration
    grid_W = GRID_W_S[GRID_W_S <= W_cap]
    if grid_W.size == 0:
        grid_W = GRID_W_S[:1]

    lobe1 = _grid_search_lobe(a_smooth, t_ride, lo1, hi1, sign_lobe1, grid_W, GRID_F)
    lobe2 = _grid_search_lobe(a_smooth, t_ride, lo2, hi2, sign_lobe2, grid_W, GRID_F)

    spacing: float | None = None
    if lobe1.t_c is not None and lobe2.t_c is not None:
        spacing = float(abs(lobe2.t_c - lobe1.t_c))

    return RideFit(
        index=ride_idx, ride_type=ride_type, duration_s=duration,
        lobe1=lobe1, lobe2=lobe2, lobe_centroid_spacing_s=spacing,
    )


def _slice_and_fit(
    acc: pd.DataFrame, gt: pd.DataFrame,
    prs: pd.DataFrame | None = None,
) -> tuple[list[dict], list[RideFit]]:
    ts_ms = acc["timestamp_ms"].to_numpy(dtype=float)
    if ts_ms.size == 0:
        return [], []
    t0_ms = float(ts_ms[0])
    fs = _estimate_fs_hz(ts_ms)
    t = (ts_ms - t0_ms) / 1000.0
    ax = acc["x"].to_numpy(dtype=float)
    ay = acc["y"].to_numpy(dtype=float)
    az = acc["z"].to_numpy(dtype=float)
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    vz = calculate_velocity_from_accelerometer(ax, ay, az, fs)
    a_vert = _vertical_accel(ax, ay, az, fs)

    # Barometer (optional — some phones don't have one). Placed on the same
    # ACC-relative time axis by subtracting ACC's t0_ms, so every ride slice
    # can be cross-referenced at a glance.
    prs_t: np.ndarray | None = None
    prs_h: np.ndarray | None = None
    if prs is not None and not prs.empty and "GT_height_m" in prs.columns:
        prs_ts_ms = prs["timestamp_ms"].to_numpy(dtype=float)
        prs_t = (prs_ts_ms - t0_ms) / 1000.0
        prs_h = prs["GT_height_m"].to_numpy(dtype=float)

    rides = gt[gt["type"].isin(("up", "down"))].reset_index(drop=True)
    slices: list[dict] = []
    fits: list[RideFit] = []
    for i, row in rides.iterrows():
        s_abs = (float(row["start_ms"]) - t0_ms) / 1000.0
        e_abs = (float(row["end_ms"]) - t0_ms) / 1000.0
        if e_abs <= s_abs:
            continue
        pad = max(CONTEXT_PAD_MIN_S, CONTEXT_PAD_FRAC * (e_abs - s_abs))
        ws, we = s_abs - pad, e_abs + pad
        mask = (t >= ws) & (t <= we)
        if not np.any(mask):
            continue
        t_ride = t[mask] - t[mask][0]
        gt_t0 = s_abs - t[mask][0]
        gt_t1 = e_abs - t[mask][0]
        fit = _fit_ride(
            t_ride, a_vert[mask], gt_t0, gt_t1,
            int(i), str(row["type"]), fs,
        )

        # Slice the barometer onto the ride's local axis, if we have it.
        prs_ride_t: np.ndarray | None = None
        prs_ride_h: np.ndarray | None = None
        if prs_t is not None and prs_h is not None:
            pm = (prs_t >= ws) & (prs_t <= we)
            if np.any(pm):
                prs_ride_t = prs_t[pm] - t[mask][0]
                prs_ride_h = prs_h[pm]

        slices.append({
            "index": int(i), "ride_type": str(row["type"]),
            "t": t_ride, "mag": mag[mask], "vz": vz[mask],
            "a_vert": a_vert[mask], "gt_t0": gt_t0, "gt_t1": gt_t1,
            "fit": fit,
            "prs_t": prs_ride_t, "prs_h": prs_ride_h,
        })
        fits.append(fit)
    return slices, fits


# --------------------------------------------------------------------------
# Plotting
# --------------------------------------------------------------------------

def _draw_lobe(
    ax: plt.Axes, lobe: LobeFit, color: str = "#c0392b",
    label: str | None = None,
) -> None:
    """Draw one fitted trapezoid over its ±W window."""
    if lobe.t_c is None or lobe.a_peak is None:
        return
    t_grid = np.linspace(lobe.t_c - lobe.half_width_s,
                         lobe.t_c + lobe.half_width_s, 400)
    y = lobe.a_peak * trapezoid_kernel(
        t_grid, lobe.t_c, lobe.half_width_s, lobe.frac_flat,
    )
    ax.plot(t_grid, y, color=color, lw=1.5, label=label)
    ax.scatter([lobe.t_c], [lobe.a_peak], color=color, s=22, zorder=4)


def _draw_ride_panel(
    ax_top: plt.Axes, ax_mid: plt.Axes, ax_bot: plt.Axes, ride: dict,
    *, title: str | None = None,
) -> None:
    color = TYPE_COLORS.get(ride["ride_type"], "#7f8c8d")
    fit: RideFit = ride["fit"]
    ax_top.axvspan(ride["gt_t0"], ride["gt_t1"], color=color, alpha=0.15, zorder=0)
    ax_top.plot(ride["t"], ride["a_vert"], color="#2c3e50", lw=0.7,
                label="$a_\\mathrm{vert}$")
    ax_top.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    _draw_lobe(ax_top, fit.lobe1, color="#c0392b")
    _draw_lobe(ax_top, fit.lobe2, color="#c0392b")
    ax_top.set_ylabel("$a_\\mathrm{vert}$ (m/s²)")
    ax_top.grid(True, alpha=0.25)

    # Diagnostics: per-lobe R², A, W, f stacked in one compact block.
    def _fmt(lobe: LobeFit, tag: str) -> str:
        if lobe.a_peak is None:
            return f"{tag}: fit failed"
        return (f"{tag}: R²={lobe.r2_local:.2f}  "
                f"A={lobe.a_peak:+.2f}  "
                f"W={lobe.half_width_s:.2f}s  "
                f"f={lobe.frac_flat:.2f}")

    ax_top.text(
        0.02, 0.97,
        "\n".join([_fmt(fit.lobe1, "L1"), _fmt(fit.lobe2, "L2")]),
        transform=ax_top.transAxes, ha="left", va="top",
        fontsize=8, color="#c0392b", family="monospace",
        bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
    )

    # Middle: vz (integrated from accelerometer).
    ax_mid.axvspan(ride["gt_t0"], ride["gt_t1"], color=color, alpha=0.15, zorder=0)
    ax_mid.plot(ride["t"], ride["vz"], color="#2980b9", lw=1.0)
    ax_mid.axhline(0.0, color="gray", lw=0.4, alpha=0.5)
    ax_mid.set_ylabel("vz (m/s)")
    ax_mid.grid(True, alpha=0.25)

    # Bottom: barometer-derived altitude (if available).
    ax_bot.axvspan(ride["gt_t0"], ride["gt_t1"], color=color, alpha=0.15, zorder=0)
    prs_t = ride.get("prs_t")
    prs_h = ride.get("prs_h")
    if prs_t is not None and prs_h is not None and prs_t.size > 0:
        ax_bot.plot(prs_t, prs_h, color="#8e44ad", lw=1.1, label="barometer height")
        ax_bot.set_ylabel("h (m)")
    else:
        ax_bot.text(
            0.5, 0.5, "no barometer",
            transform=ax_bot.transAxes, ha="center", va="center",
            fontsize=9, color="#7f8c8d", style="italic",
        )
        ax_bot.set_ylabel("h (m)")
    ax_bot.set_xlabel("t (s, ride-local)")
    ax_bot.grid(True, alpha=0.25)

    if title:
        ax_top.set_title(title, fontsize=10)


def _save_combined(slices: list[dict], out_dir: Path, exp_name: str) -> Path:
    n = len(slices)
    cols = min(4, n)
    rows = math.ceil(n / cols)
    fig = plt.figure(figsize=(4.6 * cols, 4.8 * rows))
    outer = fig.add_gridspec(rows, cols, hspace=0.75, wspace=0.35)
    for i, ride in enumerate(slices):
        r, c = divmod(i, cols)
        inner = outer[r, c].subgridspec(3, 1, hspace=0.10)
        ax_top = fig.add_subplot(inner[0])
        ax_mid = fig.add_subplot(inner[1], sharex=ax_top)
        ax_bot = fig.add_subplot(inner[2], sharex=ax_top)
        title = (
            f"#{ride['index']:02d} {ride['ride_type']} — "
            f"{ride['gt_t1'] - ride['gt_t0']:.1f}s"
        )
        _draw_ride_panel(ax_top, ax_mid, ax_bot, ride, title=title)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_mid.get_xticklabels(), visible=False)
        ax_top.set_xlabel("")
        ax_mid.set_xlabel("")
    fig.suptitle(
        f"{exp_name} — independent per-lobe trapezoid fit (matched-filter grid)",
        fontsize=13,
    )
    out_path = out_dir / "_all_rides.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def _save_parameters(fits: list[RideFit], out_dir: Path) -> Path:
    payload = [asdict(f) for f in fits]
    out_path = out_dir / "parameters.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def process(name: str) -> tuple[int, Path | None]:
    sensors, gt, _meta = getExperimentData(name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        return 0, None
    prs = sensors.get("PRS")
    slices, fits = _slice_and_fit(sensors["ACC"], gt, prs=prs)
    if not slices:
        return 0, None
    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = _save_combined(slices, out_dir, name)
    _save_parameters(fits, out_dir)
    return len(slices), combined


def main() -> int:
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    names = list_experiments(kind="train")
    print(f"processing {len(names)} TRAIN experiments → {OUT_ROOT}")
    total = 0
    ok = 0
    r2_all: list[float] = []
    for n in names:
        try:
            count, combined = process(n)
        except Exception as exc:
            print(f"[error] {n}: {type(exc).__name__}: {exc}")
            continue
        if count == 0:
            print(f"[skip]  {n}: no usable GT rides")
            continue
        total += count
        ok += 1
        pjson = combined.parent / "parameters.json"
        if pjson.exists():
            data = json.loads(pjson.read_text())
            for d in data:
                for k in ("lobe1", "lobe2"):
                    v = d.get(k) or {}
                    r2 = v.get("r2_local")
                    if r2 is not None:
                        r2_all.append(r2)
        print(f"[ok]    {n}: {count} rides → {combined.parent}")
    if r2_all:
        arr = np.array(r2_all)
        print(
            f"\nwrote {total} ride fits across {ok} experiments "
            f"(local R² over {arr.size} lobes — median={np.median(arr):.3f}, "
            f"mean={arr.mean():.3f}, p25={np.percentile(arr,25):.3f}, "
            f"p75={np.percentile(arr,75):.3f})"
        )
    else:
        print(f"\nwrote {total} ride fits across {ok} experiments")
    return 0


if __name__ == "__main__":
    sys.exit(main())
