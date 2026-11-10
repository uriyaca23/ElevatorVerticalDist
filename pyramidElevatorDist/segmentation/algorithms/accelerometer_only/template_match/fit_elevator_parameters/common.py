"""Shared primitives for trapezoid-pulse fitting.

The pure, numpy-only half of the fit pipeline: gravity-projected /
rotation-invariant vertical-acceleration preprocessing, smoothing, and the
per-lobe fit dataclasses. The matched-filter primitives are re-exported from
:mod:`pyramidElevatorDist.utils.trapezoid_template` so the detector's
``from ..fit_elevator_parameters.common import trapezoid_kernel,
match_one_template`` call sites work unchanged.

The offline grid fitters (grid constants, ride slicing, plotting, JSON
persistence) stay in the application repo — they need experiment data and
matplotlib, which this package deliberately does not depend on.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from pyramidElevatorDist.utils.accelerometer_utils import (
    estimate_gravity_stationary,
    vertical_accel_projected,
)
# Stage-agnostic matched-filter primitives — re-exported so the existing
# ``from ..fit_elevator_parameters.common import trapezoid_kernel,
# match_one_template, TemplateScan`` call sites keep working unchanged.
from pyramidElevatorDist.utils.trapezoid_template import (  # noqa: F401
    TemplateScan,
    match_one_template,
    trapezoid_kernel,
)

SMOOTH_SEC = 0.4


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
    gvec, _g_mag, _stab = estimate_gravity_stationary(ax, ay, az, fs=fs, window_sec=0.5)
    a = vertical_accel_projected(ax, ay, az, gvec)
    # DC-detrend: a single global ``g_hat`` can't cancel gravity when the
    # phone's orientation during rides differs from the calibration window.
    # Subtracting a slow rolling mean removes the residual bias without
    # eating sub-second ride lobes.
    w = max(3, int(round(_DETREND_SEC * fs)))
    dc = pd.Series(a).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return a - dc


def _a_mag_minus_g(ax: np.ndarray, ay: np.ndarray, az: np.ndarray, fs: float) -> np.ndarray:
    """Rotation-invariant ``|a(t)| − |ĝ|``.

    Why this exists: ``_vertical_accel`` projects onto a *frozen* gravity
    estimate, so an in-ride phone rotation invalidates the axis and the
    trapezoid signature collapses into the orthogonal channels. The
    magnitude residual is invariant under arbitrary rotation because
    ``|a_device| = |g · ĝ_dev + a_ride · ĝ_dev| = |g + a_ride|`` and
    ``ĝ_dev`` drops out. Sign is preserved: a take-off lobe lifts
    ``|a|`` above ``g``, a landing lobe pushes it below. Cost: any
    horizontal user motion ``a_h`` leaks in as ``a_h²/(2g)``.

    Postprocessing matches :func:`_vertical_accel` (same 8 s DC-detrend)
    so downstream thresholds tuned on ``a_vert`` carry over with
    comparable noise statistics.
    """
    mag = np.sqrt(ax * ax + ay * ay + az * az)
    _gvec, g_mag, _stab = estimate_gravity_stationary(
        ax, ay, az, fs=fs, window_sec=0.5,
    )
    a = mag - g_mag
    w = max(3, int(round(_DETREND_SEC * fs)))
    dc = pd.Series(a).rolling(w, center=True, min_periods=1).mean().to_numpy()
    return a - dc


def _smooth(x: np.ndarray, fs: float, seconds: float) -> np.ndarray:
    w = max(3, int(round(seconds * fs)))
    return pd.Series(x).rolling(w, center=True, min_periods=1).mean().to_numpy()
