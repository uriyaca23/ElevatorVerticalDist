"""Shared primitives for trapezoid-pulse fitting algorithms.

All fitters in this sub-package go through the same pipeline:

  1. Slice each GT ride out of the ACC stream with a small context pad.
  2. Project the accelerometer onto the gravity direction (``a_vert``) and
     smooth it.
  3. Search a 2-D grid of ``(W, f)`` trapezoid templates via the matched
     filter ``match_one_template``.
  4. Extract per-lobe ``(t_c, A, W, f, r²)`` into a :class:`RideFit` —
     the bit that differs per algorithm is the *selection rule* over
     the grid and sign constraints.
  5. Persist ``parameters.json`` + a small-multiples ``_all_rides.png``
     under ``labels/fit_elevator_paramater/<variant>/<exp>/``.

Each algorithm module (``basic_grid``, ``constrained_grid``) provides a
``fit_ride(...) -> RideFit`` callable and a variant directory name, and
plugs into :func:`run_fitter` which handles everything else.
"""

from __future__ import annotations

import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable, NamedTuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[6]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.physics import calculate_velocity_from_accelerometer  # noqa: E402
from src.prediction.algorithms.quality_filter import estimate_gravity_vector  # noqa: E402

LABELS_ROOT = (
    Path(__file__).resolve().parents[1]
    / "labels" / "fit_elevator_paramater"
)

# --------------------------------------------------------------------------
# Constants (shared across all fitters)
# --------------------------------------------------------------------------

CONTEXT_PAD_FRAC = 0.15
CONTEXT_PAD_MIN_S = 0.5
SMOOTH_SEC = 0.4
TYPE_COLORS = {"up": "#27ae60", "down": "#e74c3c"}

# (W, f) trapezoid-template grid — 30 × 15 = 450 templates, sampled
# densely vs. the signal's resolving power at 100 Hz. Both axes start at
# physically-motivated floors (not zero): W_MIN is just below the
# half-jerk-ramp duration T_j for commercial lifts, and F_MIN excludes
# the degenerate triangular-pulse case where the cabin never reaches
# a_max cruise.
W_MIN_S = 0
W_MAX_S = 3.0
F_MIN = 0.01
F_MAX = 0.
GRID_W_S = np.linspace(W_MIN_S, W_MAX_S, 30)
GRID_F = np.linspace(F_MIN, F_MAX, 15)

# Each lobe's search region expressed as fractions of the GT ride duration
# measured from ``gt_t0``. Small overlap so lobes near the midpoint are
# reachable from either side.
LOBE1_REGION = (0.00, 0.60)
LOBE2_REGION = (0.40, 1.00)


# --------------------------------------------------------------------------
# Dataclasses
# --------------------------------------------------------------------------

@dataclass
class LobeFit:
    """Best-matching trapezoid for a single lobe."""

    t_c: float | None = None
    a_peak: float | None = None        # SIGNED amplitude
    half_width_s: float | None = None
    frac_flat: float | None = None
    r2_local: float | None = None      # 1 - SS_res / SS_tot over the ±W window


@dataclass
class RideFit:
    """Per-ride fit result (two lobes)."""

    index: int
    ride_type: str
    duration_s: float
    lobe1: LobeFit = field(default_factory=LobeFit)
    lobe2: LobeFit = field(default_factory=LobeFit)
    lobe_centroid_spacing_s: float | None = None


# --------------------------------------------------------------------------
# Preprocessing
# --------------------------------------------------------------------------

def _estimate_fs_hz(ts_ms: np.ndarray, default: float = 100.0) -> float:
    if ts_ms.size < 2:
        return default
    dt_ms = float(np.median(np.diff(ts_ms)))
    return default if dt_ms <= 0 else 1000.0 / dt_ms


_DETREND_SEC = 8.0


def _vertical_accel(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    gvec, g_mag, _stab = estimate_gravity_vector(ax, ay, az, fs=fs, window_sec=0.5)
    g_hat = gvec / (np.linalg.norm(gvec) + 1e-12)
    a = ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag
    # DC-detrend: a single global ``g_hat`` can't cancel gravity when the
    # phone's orientation during rides differs from the calibration window.
    # Subtracting a slow rolling mean removes the residual bias without
    # eating sub-second ride lobes.
    w = max(3, int(round(_DETREND_SEC * fs)))
    dc = pd.Series(a).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return a - dc


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
# Matched-filter sweep (shared by every fitter)
# --------------------------------------------------------------------------

class TemplateScan(NamedTuple):
    """Result of sliding one ``(W, f)`` template across a ride window.

    ``A_hat[i]`` / ``r2_local[i]`` are the unconstrained least-squares
    amplitude and local R² if the template were centered at sample ``i``.
    ``inner[i] = A_hat[i] * norm_t`` is the raw ``<a, tpl>`` inner product,
    and ``local_power[i] = <a, a>`` on the same ±W window. Positions whose
    ±W window falls off the signal are NaN. These five quantities are
    enough to form any per-lobe or joint objective.
    """

    A_hat: np.ndarray
    r2_local: np.ndarray
    inner: np.ndarray
    local_power: np.ndarray
    norm_t: float


def match_one_template(a: np.ndarray, t: np.ndarray, W: float, frac_flat: float) -> TemplateScan:
    """Slide a unit trapezoid of shape ``(W, frac_flat)`` over signal ``a``."""
    n = a.size
    nan = np.full(n, np.nan)
    if n == 0:
        return TemplateScan(nan[:0], nan[:0], nan[:0], nan[:0], 0.0)

    dt = float(np.median(np.diff(t))) if t.size > 1 else 1.0 / 100.0
    K = max(3, int(round(2 * W / dt)))  # samples per ±W window
    if K % 2 == 0:
        K += 1
    half = K // 2

    t_kernel = (np.arange(K) - half) * dt
    tpl = trapezoid_kernel(t_kernel, 0.0, W, frac_flat)
    norm_t = float(np.sum(tpl * tpl))
    if norm_t < 1e-9:
        return TemplateScan(nan, nan, nan, nan, 0.0)

    # Cross-correlation at each center.
    inner = np.convolve(a, tpl[::-1], mode="same")

    # Rolling window power of the signal.
    a2 = a * a
    csum = np.concatenate(([0.0], np.cumsum(a2)))
    local_power = np.full(n, np.nan)
    if n - half > half:
        idx = np.arange(half, n - half)
        local_power[idx] = csum[idx + half + 1] - csum[idx - half]

    A_hat = np.full(n, np.nan)
    valid = np.isfinite(local_power)
    A_hat[valid] = inner[valid] / norm_t

    r2_local = np.full(n, np.nan)
    denom = local_power[valid]
    with np.errstate(divide="ignore", invalid="ignore"):
        ss_res = denom - A_hat[valid] * inner[valid]
        r2 = 1.0 - ss_res / np.where(denom > 1e-9, denom, np.nan)
    r2_local[valid] = r2

    # Zero out inner outside the valid (power-defined) window so downstream
    # sign masks don't trip on edge artefacts.
    inner_masked = np.where(np.isfinite(local_power), inner, np.nan)
    return TemplateScan(A_hat, r2_local, inner_masked, local_power, norm_t)


# --------------------------------------------------------------------------
# Ride slicing
# --------------------------------------------------------------------------

@dataclass
class RideSlice:
    """One GT ride, sliced out of the session with a small context pad.

    All arrays are on a ride-local time axis starting at 0.
    """

    index: int
    ride_type: str
    t: np.ndarray
    mag: np.ndarray
    vz: np.ndarray
    a_vert: np.ndarray       # raw (unsmoothed) vertical accel
    a_smooth: np.ndarray     # smoothed vertical accel (what fitters use)
    gt_t0: float
    gt_t1: float
    fs: float
    prs_t: np.ndarray | None
    prs_h: np.ndarray | None
    fit: RideFit | None = None


def build_ride_slices(
    acc: pd.DataFrame, gt: pd.DataFrame,
    prs: pd.DataFrame | None = None,
) -> list[RideSlice]:
    ts_ms = acc["timestamp_ms"].to_numpy(dtype=float)
    if ts_ms.size == 0:
        return []
    t0_ms = float(ts_ms[0])
    fs = _estimate_fs_hz(ts_ms)
    t = (ts_ms - t0_ms) / 1000.0
    ax = acc["x"].to_numpy(dtype=float)
    ay = acc["y"].to_numpy(dtype=float)
    az = acc["z"].to_numpy(dtype=float)
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    vz = calculate_velocity_from_accelerometer(ax, ay, az, fs)
    a_vert = _vertical_accel(ax, ay, az, fs)
    a_smooth_full = _smooth(a_vert, fs, SMOOTH_SEC)

    # Optional barometer (some phones don't have one). Mapped onto ACC's
    # time origin so every ride slice can be cross-referenced.
    prs_t: np.ndarray | None = None
    prs_h: np.ndarray | None = None
    if prs is not None and not prs.empty and "GT_height_m" in prs.columns:
        prs_ts_ms = prs["timestamp_ms"].to_numpy(dtype=float)
        prs_t = (prs_ts_ms - t0_ms) / 1000.0
        prs_h = prs["GT_height_m"].to_numpy(dtype=float)

    rides = gt[gt["type"].isin(("up", "down"))].reset_index(drop=True)
    slices: list[RideSlice] = []
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

        prs_ride_t: np.ndarray | None = None
        prs_ride_h: np.ndarray | None = None
        if prs_t is not None and prs_h is not None:
            pm = (prs_t >= ws) & (prs_t <= we)
            if np.any(pm):
                prs_ride_t = prs_t[pm] - t[mask][0]
                prs_ride_h = prs_h[pm]

        slices.append(RideSlice(
            index=int(i), ride_type=str(row["type"]),
            t=t_ride, mag=mag[mask], vz=vz[mask],
            a_vert=a_vert[mask], a_smooth=a_smooth_full[mask],
            gt_t0=gt_t0, gt_t1=gt_t1, fs=fs,
            prs_t=prs_ride_t, prs_h=prs_ride_h,
        ))
    return slices


# --------------------------------------------------------------------------
# Plotting / IO
# --------------------------------------------------------------------------

def _draw_lobe(ax: plt.Axes, lobe: LobeFit, color: str = "#c0392b") -> None:
    if lobe.t_c is None or lobe.a_peak is None:
        return
    t_grid = np.linspace(
        lobe.t_c - lobe.half_width_s, lobe.t_c + lobe.half_width_s, 400,
    )
    y = lobe.a_peak * trapezoid_kernel(
        t_grid, lobe.t_c, lobe.half_width_s, lobe.frac_flat,
    )
    ax.plot(t_grid, y, color=color, lw=1.5)
    ax.scatter([lobe.t_c], [lobe.a_peak], color=color, s=22, zorder=4)


def _draw_ride_panel(
    ax_top: plt.Axes, ax_mid: plt.Axes, ax_bot: plt.Axes,
    ride: RideSlice, *, title: str | None = None,
) -> None:
    color = TYPE_COLORS.get(ride.ride_type, "#7f8c8d")
    fit = ride.fit
    ax_top.axvspan(ride.gt_t0, ride.gt_t1, color=color, alpha=0.15, zorder=0)
    ax_top.plot(ride.t, ride.a_vert, color="#2c3e50", lw=0.7)
    ax_top.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    if fit is not None:
        _draw_lobe(ax_top, fit.lobe1, color="#c0392b")
        _draw_lobe(ax_top, fit.lobe2, color="#c0392b")
    ax_top.set_ylabel("$a_\\mathrm{vert}$ (m/s²)")
    ax_top.grid(True, alpha=0.25)

    def _fmt(lobe: LobeFit, tag: str) -> str:
        if lobe.a_peak is None:
            return f"{tag}: fit failed"
        return (f"{tag}: R²={lobe.r2_local:.2f}  "
                f"A={lobe.a_peak:+.2f}  "
                f"W={lobe.half_width_s:.2f}s  "
                f"f={lobe.frac_flat:.2f}")

    if fit is not None:
        ax_top.text(
            0.02, 0.97,
            "\n".join([_fmt(fit.lobe1, "L1"), _fmt(fit.lobe2, "L2")]),
            transform=ax_top.transAxes, ha="left", va="top",
            fontsize=8, color="#c0392b", family="monospace",
            bbox=dict(facecolor="white", alpha=0.75, edgecolor="none"),
        )

    ax_mid.axvspan(ride.gt_t0, ride.gt_t1, color=color, alpha=0.15, zorder=0)
    ax_mid.plot(ride.t, ride.vz, color="#2980b9", lw=1.0)
    ax_mid.axhline(0.0, color="gray", lw=0.4, alpha=0.5)
    ax_mid.set_ylabel("vz (m/s)")
    ax_mid.grid(True, alpha=0.25)

    ax_bot.axvspan(ride.gt_t0, ride.gt_t1, color=color, alpha=0.15, zorder=0)
    if ride.prs_t is not None and ride.prs_h is not None and ride.prs_t.size > 0:
        ax_bot.plot(ride.prs_t, ride.prs_h, color="#8e44ad", lw=1.1)
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


def save_combined(
    slices: list[RideSlice], out_dir: Path, exp_name: str, suptitle: str,
) -> Path:
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
        title = f"#{ride.index:02d} {ride.ride_type} — {ride.gt_t1 - ride.gt_t0:.1f}s"
        _draw_ride_panel(ax_top, ax_mid, ax_bot, ride, title=title)
        plt.setp(ax_top.get_xticklabels(), visible=False)
        plt.setp(ax_mid.get_xticklabels(), visible=False)
        ax_top.set_xlabel("")
        ax_mid.set_xlabel("")
    fig.suptitle(f"{exp_name} — {suptitle}", fontsize=13)
    out_path = out_dir / "_all_rides.png"
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return out_path


def save_parameters(fits: list[RideFit], out_dir: Path) -> Path:
    payload = [asdict(f) for f in fits]
    out_path = out_dir / "parameters.json"
    out_path.write_text(json.dumps(payload, indent=2))
    return out_path


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

FitRideFn = Callable[
    [np.ndarray, np.ndarray, float, float, int, str, float],
    RideFit,
]


def _process_experiment(
    name: str, out_root: Path, fit_ride: FitRideFn, suptitle: str,
) -> tuple[int, Path | None]:
    sensors, gt, _meta = getExperimentData(name)
    if "ACC" not in sensors or sensors["ACC"].empty:
        return 0, None
    slices = build_ride_slices(sensors["ACC"], gt, prs=sensors.get("PRS"))
    if not slices:
        return 0, None
    for s in slices:
        s.fit = fit_ride(
            s.t, s.a_smooth, s.gt_t0, s.gt_t1, s.index, s.ride_type, s.fs,
        )
    out_dir = out_root / name
    out_dir.mkdir(parents=True, exist_ok=True)
    combined = save_combined(slices, out_dir, name, suptitle=suptitle)
    save_parameters([s.fit for s in slices if s.fit is not None], out_dir)
    return len(slices), combined


def run_fitter(
    out_dir_name: str, fit_ride: FitRideFn, *, title_suffix: str,
) -> int:
    """Run a fitter across every TRAIN experiment and persist results."""
    out_root = LABELS_ROOT / out_dir_name
    out_root.mkdir(parents=True, exist_ok=True)
    names = list_experiments(kind="train")
    print(f"processing {len(names)} TRAIN experiments → {out_root}")
    total = 0
    ok = 0
    r2_all: list[float] = []
    for n in names:
        try:
            count, combined = _process_experiment(n, out_root, fit_ride, title_suffix)
        except Exception as exc:
            print(f"[error] {n}: {type(exc).__name__}: {exc}")
            continue
        if count == 0 or combined is None:
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
            f"mean={arr.mean():.3f}, p25={np.percentile(arr, 25):.3f}, "
            f"p75={np.percentile(arr, 75):.3f})"
        )
    else:
        print(f"\nwrote {total} ride fits across {ok} experiments")
    return 0
