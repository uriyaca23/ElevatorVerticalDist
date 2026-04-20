"""Trapezoid pulse-pair Δh estimator — public estimator class."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from ..common.accel_utils import (
    estimate_gravity_stationary,
    vertical_accel_magnitude,
    vertical_accel_projected,
    zupt_integrate,
)
from ..common.conformal import ConformalCalibrator
from ..common.noise_db import get_phone_accel_noise_sigma
from ..common.pulse_pair import (
    GRID_W_S,
    fit_shared_shape_pair,
    height_from_fit,
    smooth_rolling_mean,
    theoretical_sigma_height,
    trapezoid_kernel,
)
from ..common.types import CalibrationSample, PredictionOutput
from .config import TrapezoidAccelConfig
from .quality import assess as quality_assess


@dataclass
class _SegInp:
    ax: np.ndarray; ay: np.ndarray; az: np.ndarray
    t_sec: np.ndarray
    fs: float
    pre_ax: np.ndarray; pre_ay: np.ndarray; pre_az: np.ndarray
    post_ax: np.ndarray; post_ay: np.ndarray; post_az: np.ndarray
    phone: str


class TrapezoidAccelEstimator:
    """Shared-shape trapezoid-pulse-pair accelerometer Δh estimator.

    Pipeline per segment:
      1. Estimate gravity from pre/post stationary windows and project
         the 3-axis accelerometer onto the vertical.
      2. Detrend with a slow rolling mean (absorbs residual bias).
      3. Rolling-mean smooth at ``smooth_sec`` to kill hand-tremor leakage.
      4. Fit shared-shape trapezoid-pulse pair (W, f, |A|, t_c1, t_c2)
         by matched-filter grid search (see ``common.pulse_pair``).
      5. Analytic Δh = sign · |A| · W · (1+f) · (t_c2 − t_c1).
      6. Theoretical σ via delta method on (A, W, f, Δt_c), scaled by
         an empirical drift-factor + relative-|Δh| term that absorb
         systematic deviations from the ideal trapezoid shape.
      7. Quality filter produces accept/reject + quality score.
      8. Conformal multiplier (fit at calibration time) converts the
         theoretical σ into a 90% CI half-width.
    """

    def __init__(self, config: TrapezoidAccelConfig | None = None):
        self.config = config or TrapezoidAccelConfig()
        self.conformal = ConformalCalibrator(alpha=self.config.alpha)

    def save(self, path: Path | str) -> None:
        self.conformal.save(path)

    def load(self, path: Path | str) -> None:
        self.conformal = ConformalCalibrator.load(path)

    # ------------------------------------------------------------------
    # Segment plumbing (same shape as the ZUPT estimator so the two can
    # be called interchangeably by the evaluation runner)
    # ------------------------------------------------------------------
    def _extract(
        self, ride: pd.DataFrame, phone: str,
        pre: Optional[pd.DataFrame], post: Optional[pd.DataFrame],
    ) -> _SegInp:
        c = self.config
        t_ms = np.asarray(ride[c.time_col].to_numpy(), dtype=float)
        t_sec = (t_ms - t_ms[0]) / 1000.0 if t_ms.size else t_ms
        ax = np.asarray(ride[c.ax_col].to_numpy(), dtype=float)
        ay = np.asarray(ride[c.ay_col].to_numpy(), dtype=float)
        az = np.asarray(ride[c.az_col].to_numpy(), dtype=float)
        if t_sec.size > 1:
            dt_med = float(np.median(np.diff(t_sec)))
            fs = 1.0 / dt_med if dt_med > 0 else c.default_fs_hz
        else:
            fs = c.default_fs_hz

        def _axes(df):
            if df is None or df.empty:
                return np.array([]), np.array([]), np.array([])
            return (np.asarray(df[c.ax_col].to_numpy(), dtype=float),
                    np.asarray(df[c.ay_col].to_numpy(), dtype=float),
                    np.asarray(df[c.az_col].to_numpy(), dtype=float))

        pax, pay, paz = _axes(pre)
        qax, qay, qaz = _axes(post)
        return _SegInp(
            ax=ax, ay=ay, az=az, t_sec=t_sec, fs=fs,
            pre_ax=pax, pre_ay=pay, pre_az=paz,
            post_ax=qax, post_ay=qay, post_az=qaz,
            phone=phone or c.default_phone,
        )

    def _vertical(self, inp: _SegInp) -> tuple[np.ndarray, str]:
        """Project 3-axis accel onto vertical using the best-available
        gravity reference. Returns (a_vert_signed, method_used)."""
        c = self.config
        pre_g, pre_mag, pre_s = estimate_gravity_stationary(
            inp.pre_ax, inp.pre_ay, inp.pre_az, fs=inp.fs,
            window_sec=c.grav_window_sec,
        )
        post_g, post_mag, post_s = estimate_gravity_stationary(
            inp.post_ax, inp.post_ay, inp.post_az, fs=inp.fs,
            window_sec=c.grav_window_sec,
        )
        pre_ok = 8.0 < pre_mag < 12.0 and pre_s < c.grav_stability_max
        post_ok = 8.0 < post_mag < 12.0 and post_s < c.grav_stability_max
        if pre_ok and post_ok:
            w1 = 1.0 / max(pre_s, 1e-3); w2 = 1.0 / max(post_s, 1e-3)
            gvec = (pre_g * w1 + post_g * w2) / (w1 + w2)
            return vertical_accel_projected(inp.ax, inp.ay, inp.az, gvec), "projected_pre_post"
        if pre_ok:
            return vertical_accel_projected(inp.ax, inp.ay, inp.az, pre_g), "projected_pre"
        if post_ok:
            return vertical_accel_projected(inp.ax, inp.ay, inp.az, post_g), "projected_post"
        return vertical_accel_magnitude(inp.ax, inp.ay, inp.az), "magnitude"

    # ------------------------------------------------------------------
    # Per-segment prediction
    # ------------------------------------------------------------------
    def predict_segment(
        self, ride: pd.DataFrame, phone_model: str = "",
        pre: Optional[pd.DataFrame] = None, post: Optional[pd.DataFrame] = None,
    ) -> PredictionOutput:
        c = self.config
        inp = self._extract(ride, phone_model, pre, post)

        if inp.ax.size < c.min_segment_samples:
            return PredictionOutput(
                height_diff=0.0, ci_half_width=math.inf,
                theoretical_sigma=math.inf, accepted=False,
                quality_score=10.0, reject_reason="segment_too_short",
                meta={"n_samples": int(inp.ax.size)},
            )

        # ---- Vertical projection + (optional) detrend + smooth ----
        a_vert, vert_method = self._vertical(inp)
        if c.detrend_sec > 0:
            w_detrend = max(3, int(round(c.detrend_sec * inp.fs)))
            dc = pd.Series(a_vert).rolling(w_detrend, center=True, min_periods=1).mean().to_numpy()
            a_vert = a_vert - dc
        a_smooth = smooth_rolling_mean(a_vert, inp.fs, c.smooth_sec)

        # ---- Shared-shape pulse-pair fit ----
        # Ride-local time axis starts at 0. For prediction we let the
        # fitter pick direction; the GT label isn't part of the
        # inference contract. W floor adapts with ride duration to
        # counter the narrow-W bias the matched filter has on long
        # rides (see ``config.W_floor_alpha`` for the derivation).
        duration = float(inp.t_sec[-1])
        W_floor = max(c.W_floor_min_sec, c.W_floor_alpha * duration)
        grid_W = np.linspace(W_floor, GRID_W_S[-1], GRID_W_S.size)
        fit = fit_shared_shape_pair(
            a_smooth, inp.t_sec,
            gt_t0=0.0, gt_t1=duration,
            direction=None,
            grid_W=grid_W,
        )
        if fit is None:
            return PredictionOutput(
                height_diff=0.0, ci_half_width=math.inf,
                theoretical_sigma=math.inf, accepted=False,
                quality_score=8.0, reject_reason="no_pulse_pair_fit",
                meta={"vert_method": vert_method},
            )

        # ---- Velocity-anchored amplitude ----
        # The matched filter underestimates |A| on long rides because
        # it picks a narrow high-R² template that doesn't capture the
        # full cruise velocity. We rescale |A| so that
        # ``A · W · (1 + f)`` matches the measured cruise velocity
        # (ZUPT-integrated between the two lobes). For short rides
        # with no cruise window we skip the correction.
        A_fit = fit.A
        A_used = A_fit
        v_peak_meas = math.nan
        if c.velocity_anchor_A:
            _pos, vel = zupt_integrate(a_smooth, inp.t_sec)
            # Cruise region: between the lobes, excluding the lobe
            # support (±W around each centre).
            cruise_mask = (
                (inp.t_sec > fit.t_c1 + fit.W)
                & (inp.t_sec < fit.t_c2 - fit.W)
            )
            cruise_width = float(
                (inp.t_sec[cruise_mask][-1] - inp.t_sec[cruise_mask][0])
                if cruise_mask.sum() >= 2 else 0.0
            )
            if cruise_width >= c.velocity_anchor_min_cruise_sec:
                # Signed cruise velocity; compare to sign of fit
                v_peak_meas = float(np.mean(vel[cruise_mask]))
                # A_corrected gives v_peak_meas directly:
                #   v_peak = sign · A · W · (1+f)
                #   => A = |v_peak_meas| / (W · (1+f))
                denom = fit.W * (1.0 + fit.f)
                if denom > 1e-6:
                    A_used = abs(v_peak_meas) / denom

        # Re-derive Δh with the corrected amplitude
        if A_used != A_fit and A_used > 0:
            v_peak_corrected = A_used * fit.W * (1.0 + fit.f)
            height_diff = float(fit.sign * v_peak_corrected * fit.delta_t_c)
        else:
            height_diff = height_from_fit(fit)

        # ---- Theoretical σ ----
        # Use the *empirical* residual RMS as the effective σ_a instead
        # of the datasheet white-noise σ. The residuals after the
        # shared-shape template subtraction capture sensor noise plus
        # whatever coloured noise (hand tremor, gravity-projection
        # drift, sub-second rotational wobble) is actually present on
        # this ride — so σ_a_emp is a full error-energy accounting,
        # not just a datasheet promise. The datasheet σ is used only
        # as a minimum floor so segments with an implausibly perfect
        # fit don't end up with near-zero σ.
        sigma_a_datasheet = get_phone_accel_noise_sigma(inp.phone, inp.fs)
        residual_rms = float(np.sqrt(np.mean(np.asarray(fit.residuals) ** 2)))
        sigma_a_eff = max(residual_rms, sigma_a_datasheet)
        dt_sec = float(np.median(np.diff(inp.t_sec))) if inp.t_sec.size > 1 else 1.0 / inp.fs
        sig = theoretical_sigma_height(fit, sigma_a=sigma_a_eff, dt_sec=dt_sec)
        sigma_white = max(sig["sigma_dh"], c.min_theoretical_sigma_m)

        # Ride-drift angle from the quality features (recomputed here so
        # σ sees it whether the quality filter rejects or not).
        from .quality import _ride_gravity_drift
        max_drift, _ = _ride_gravity_drift(inp.ax, inp.ay, inp.az, inp.fs)
        # Scale: small drift is fine, large drift blows up σ linearly.
        drift_scale = 1.0 + c.drift_scale * max(0.0, max_drift)
        sigma_drift_scaled = sigma_white * drift_scale

        # Relative floor: gravity-projection errors scale with |Δh|.
        sigma_rel = c.relative_sigma_factor * abs(height_diff)

        # Combine independently-added terms in quadrature
        theo_sigma = math.sqrt(sigma_drift_scaled ** 2 + sigma_rel ** 2)
        theo_sigma = max(theo_sigma, c.min_theoretical_sigma_m)

        # ---- Conformal half-width ----
        ci = self.conformal.half_width(theo_sigma)
        ci = max(c.ci_absolute_floor_m, min(ci, c.ci_absolute_cap_m))

        # ---- Quality filter ----
        q = quality_assess(
            inp.ax, inp.ay, inp.az,
            inp.pre_ax, inp.pre_ay, inp.pre_az,
            inp.post_ax, inp.post_ay, inp.post_az,
            fs=inp.fs,
            fit_joint_r2=fit.joint_r2, fit_residuals=fit.residuals,
            predicted_abs_dh=abs(height_diff),
            delta_tc_sec=fit.delta_t_c,
            W_fit=fit.W, A_fit=A_used,
            duration_s=float(inp.t_sec[-1]),
            min_segment_samples=c.min_segment_samples,
            grav_window_sec=c.grav_window_sec,
            grav_stability_max=c.grav_stability_max,
            quality_score_reject=c.quality_score_reject,
            min_joint_r2=c.min_joint_r2,
            min_delta_tc_sec=c.min_delta_tc_sec,
            min_distance_m=c.min_distance_m,
            max_distance_m=c.max_distance_m,
            min_active_fraction=c.min_active_fraction,
        )

        meta: dict[str, Any] = {
            "vert_method": vert_method,
            "params": {
                "A_fit": fit.A, "A_used": A_used,
                "W": fit.W, "f": fit.f,
                "t_c1": fit.t_c1, "t_c2": fit.t_c2,
                "sign": fit.sign, "joint_r2": fit.joint_r2,
                "v_peak_measured": v_peak_meas,
            },
            "fit_r2_1": fit.r2_1, "fit_r2_2": fit.r2_2,
            "delta_tc_sec": fit.delta_t_c,
            "sigma_breakdown": sig,
            "sigma_white": sigma_white,
            "sigma_drift_scaled": sigma_drift_scaled,
            "sigma_rel": sigma_rel,
            "ride_drift_deg": float(max_drift),
            "quality_features": q.features,
            "a_smooth": a_smooth,
            "a_template": (
                fit.sign * A_used * trapezoid_kernel(inp.t_sec, fit.t_c1, fit.W, fit.f)
                - fit.sign * A_used * trapezoid_kernel(inp.t_sec, fit.t_c2, fit.W, fit.f)
            ),
            "t_sec": inp.t_sec,
        }
        return PredictionOutput(
            height_diff=float(height_diff),
            ci_half_width=float(ci),
            theoretical_sigma=float(theo_sigma),
            accepted=bool(q.accept),
            quality_score=float(q.quality_score),
            reject_reason=q.reject_reason,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, samples: list[CalibrationSample]) -> dict:
        usable = [s for s in samples if s.signal_clear and s.accepted]
        if not usable:
            return {"n_used": 0, "note": "no_usable_samples"}
        abs_errors = [s.abs_error for s in usable]
        sigmas = [s.theoretical_sigma for s in usable]
        self.conformal.fit(abs_errors, sigmas)
        return {
            "n_used": len(usable),
            "multiplier": self.conformal.multiplier,
            "p95_score": self.conformal.p95_score,
        }
