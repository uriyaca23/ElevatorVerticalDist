"""Trapezoid pulse-pair Δh estimator — public estimator class."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd

from src.utils.accelerometer_utils import (
    estimate_gravity_stationary,
    vertical_accel_magnitude,
    vertical_accel_projected,
    zupt_integrate,
)
from src.utils.conformal import ConformalCalibrator
from src.utils.sensor_noise import get_phone_accel_noise_sigma
from .pulse_pair import (
    GRID_W_S,
    fit_joined_pulse,
    fit_shared_shape_pair,
    height_from_fit,
    joined_kernel,
    smooth_rolling_mean,
    theoretical_sigma_height,
    theoretical_sigma_height_joined,
    trapezoid_kernel,
)
from ...common.types import CalibrationSample, PredictionOutput
from ...configTypes import TrapezoidAccelConfig
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
         by matched-filter grid search (see ``.pulse_pair``).
      5. Analytic Δh = sign · |A| · W · (1+f) · (t_c2 − t_c1).
      6. Theoretical σ by delta-method propagation of the matched-filter
         CRB through Δh = s·A·W(1+f)·Δt_c. Three physics-grounded
         enrichments: (i) σ_a is scaled by 1/max(R², r2_epsilon) to
         widen the CI on poor-fit segments; (ii) σ_Δt_c² is inflated
         by (2W/(Δt_c-2W))² in the lobe-overlap regime; (iii) when
         velocity-anchoring is active, σ_A² uses the cruise-velocity
         variance rather than the matched-filter CRB. No drift
         multiplier or relative-|Δh| term — those were ZUPT-style
         constructs unjustified for a matched-filter closed-form.
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
        # Two parallel fits are attempted: the unconstrained pair fit
        # (5 free parameters, Δt_c free) and the joined-pulse fit
        # (4 free parameters, Δt_c = 2W forced). The latter is the
        # natural model for short rides where the cabin never reaches
        # cruise velocity, in which case the accelerometer trace is a
        # single bipolar pulse. We compare joint R² and keep whichever
        # fit explains the signal better. When both fits fail (low
        # R²) we fall back to an inline ZUPT displacement as a
        # last-resort estimate so the segment is not silently dropped
        # by the matched-filter stage.
        duration = float(inp.t_sec[-1])
        W_floor = max(c.W_floor_min_sec, c.W_floor_alpha * duration)
        grid_W = np.linspace(W_floor, GRID_W_S[-1], GRID_W_S.size)

        pair_fit = fit_shared_shape_pair(
            a_smooth, inp.t_sec,
            gt_t0=0.0, gt_t1=duration,
            direction=None,
            grid_W=grid_W,
        )
        joined_fit = fit_joined_pulse(
            a_smooth, inp.t_sec,
            gt_t0=0.0, gt_t1=duration,
            direction=None,
            grid_W=grid_W,
        )

        # Pick the better fit by joint R². Applies a small advantage
        # threshold in favour of the joined model when it wins, since
        # the joined model has fewer free parameters (Occam's razor —
        # equivalent R² but with one less DOF should prefer joined).
        def _pair_r2(f): return f.joint_r2 if f is not None else -np.inf
        mode: str
        if pair_fit is None and joined_fit is None:
            # Both matched-filter fits failed. Fall through to ZUPT
            # fallback below.
            mode = "zupt_fallback"
            fit = None
        elif joined_fit is None:
            mode = "pair"; fit = pair_fit
        elif pair_fit is None:
            mode = "joined"; fit = joined_fit
        else:
            pair_r2 = _pair_r2(pair_fit)
            joined_r2 = _pair_r2(joined_fit)
            # Prefer joined when its R² is at least as good AND the pair fit
            # sits in the overlap regime (Δt_c close to 2W). Otherwise prefer
            # whichever fits better outright.
            pair_overlap_ratio = pair_fit.delta_t_c / (2.0 * max(pair_fit.W, 1e-6))
            if joined_r2 > pair_r2 + c.joined_r2_advantage:
                mode = "joined"; fit = joined_fit
            elif pair_overlap_ratio < c.pair_overlap_handoff and joined_r2 > pair_r2 - c.joined_r2_advantage:
                # Overlap regime — defer to joined if it's comparable.
                mode = "joined"; fit = joined_fit
            else:
                mode = "pair"; fit = pair_fit

        # ---- ZUPT fallback ----
        # Kicks in when both matched-filter fits failed, OR when the
        # kept fit has R² below ``zupt_fallback_r2`` — a sign that the
        # signal doesn't look like an elevator ride at all, and
        # integrating the vertical velocity is more honest than
        # forcing a pulse model.
        sigma_a_datasheet = get_phone_accel_noise_sigma(inp.phone, inp.fs)
        dt_sec = float(np.median(np.diff(inp.t_sec))) if inp.t_sec.size > 1 else 1.0 / inp.fs

        zupt_height = None
        if mode == "zupt_fallback" or (fit is not None and fit.joint_r2 < c.zupt_fallback_r2):
            pos, vel = zupt_integrate(a_smooth, inp.t_sec)
            if pos.size >= 2:
                zupt_height = float(pos[-1] - pos[0])
                # When both matched-filter fits failed, this is the
                # only estimate we have; σ uses a coarse ZUPT-style
                # bound ``σ_a · Δt² · √(N³/12)``.
                N = float(inp.ax.size)
                sigma_zupt = max(
                    sigma_a_datasheet * dt_sec ** 2 * math.sqrt(N ** 3 / 12.0),
                    c.min_theoretical_sigma_m,
                )
                if mode == "zupt_fallback":
                    theo_sigma = sigma_zupt
                    height_diff = zupt_height
                    ci = self.conformal.half_width(theo_sigma)
                    ci = max(c.ci_absolute_floor_m, min(ci, c.ci_absolute_cap_m))
                    return PredictionOutput(
                        height_diff=float(height_diff),
                        ci_half_width=float(ci),
                        theoretical_sigma=float(theo_sigma),
                        accepted=False,  # ZUPT fallback isn't trusted blindly
                        quality_score=5.0,
                        reject_reason="zupt_fallback_both_fits_failed",
                        meta={
                            "vert_method": vert_method,
                            "mode": mode,
                            "zupt_height": zupt_height,
                        },
                    )

        # ---- Velocity-anchored amplitude (pair-mode only) ----
        # The matched filter underestimates |A| on long rides because
        # it picks a narrow high-R² template that doesn't capture the
        # full cruise velocity. We rescale |A| so that
        # ``A · W · (1 + f)`` matches the measured cruise velocity
        # (ZUPT-integrated between the two lobes). The anchor is
        # disabled in joined mode because a joined ride has no cruise
        # window by construction.
        A_fit = fit.A
        A_used = A_fit
        v_peak_meas = math.nan
        cruise_width = 0.0
        cruise_v_std = 0.0
        anchored = False
        if mode == "pair" and c.velocity_anchor_A:
            _pos, vel = zupt_integrate(a_smooth, inp.t_sec)
            cruise_mask = (
                (inp.t_sec > fit.t_c1 + fit.W)
                & (inp.t_sec < fit.t_c2 - fit.W)
            )
            if cruise_mask.sum() >= 2:
                cruise_width = float(
                    inp.t_sec[cruise_mask][-1] - inp.t_sec[cruise_mask][0]
                )
                cruise_v_std = float(np.std(vel[cruise_mask]))
            if cruise_width >= c.velocity_anchor_min_cruise_sec:
                v_peak_meas = float(np.mean(vel[cruise_mask]))
                denom = fit.W * (1.0 + fit.f)
                if denom > 1e-6:
                    A_used = abs(v_peak_meas) / denom
                    anchored = True

        # ---- Δh computation ----
        if mode == "joined":
            # Joined pulse: Δh = 2·s·A·W²·(1+f)
            height_diff = float(
                fit.sign * 2.0 * fit.A * fit.W * fit.W * (1.0 + fit.f)
            )
        elif A_used != A_fit and A_used > 0:
            v_peak_corrected = A_used * fit.W * (1.0 + fit.f)
            height_diff = float(fit.sign * v_peak_corrected * fit.delta_t_c)
        else:
            height_diff = height_from_fit(fit)

        # ---- Theoretical σ ----
        # Pair mode: delta method on Δh = s·A·W(1+f)·Δt_c with R²-
        # scaled effective noise, overlap inflation on σ_Δt_c (only
        # when Δt_c < 2W), and optional velocity-anchored σ_A.
        # Joined mode: delta method on Δh = 2·s·A·W²·(1+f); σ_A uses
        # the joined-template CRB (norm 2·⟨τ,τ⟩) and the time-shift
        # parameter t_mid drops out because Δh is shift-invariant.
        residual_rms = float(np.sqrt(np.mean(np.asarray(fit.residuals) ** 2)))
        sigma_a_eff = max(residual_rms, sigma_a_datasheet)
        if mode == "pair":
            anchored_fit = fit
            if A_used != A_fit and A_used > 0:
                from dataclasses import replace as _replace
                anchored_fit = _replace(fit, A=A_used)
            sig = theoretical_sigma_height(
                anchored_fit, sigma_a=sigma_a_eff, dt_sec=dt_sec,
                joint_r2=fit.joint_r2, r2_epsilon=c.r2_epsilon,
                cruise_sec=cruise_width, anchored=anchored,
                overlap_delta=c.overlap_delta,
            )
        else:  # joined
            sig = theoretical_sigma_height_joined(
                fit, sigma_a=sigma_a_eff,
                joint_r2=fit.joint_r2, r2_epsilon=c.r2_epsilon,
            )
        theo_sigma = max(sig["sigma_dh"], c.min_theoretical_sigma_m)

        # ---- Conformal half-width ----
        ci = self.conformal.half_width(theo_sigma)
        ci = max(c.ci_absolute_floor_m, min(ci, c.ci_absolute_cap_m))

        # ---- Out-of-lobe residual concentration (quality feature) ----
        # For a well-fit ride the residuals should be roughly uniform in
        # time (just sensor noise). We flag rides where the residual
        # energy density OUTSIDE the lobes is disproportionately
        # larger than the density INSIDE the lobes — that pattern
        # signals a mid-ride disturbance (door bounce, walking) the
        # matched filter could not explain. In joined mode the two
        # lobes share a boundary so the "inside" mask covers both
        # halves of the bipolar pulse.
        res = np.asarray(fit.residuals, dtype=float)
        if res.size == inp.t_sec.size and res.size > 0:
            inside_lobes = (
                (np.abs(inp.t_sec - fit.t_c1) <= fit.W)
                | (np.abs(inp.t_sec - fit.t_c2) <= fit.W)
            )
            n_in = int(inside_lobes.sum())
            n_out = int((~inside_lobes).sum())
            e_in = float(np.sum(res[inside_lobes] ** 2))
            e_out = float(np.sum(res[~inside_lobes] ** 2))
            density_in = e_in / max(n_in, 1)
            density_out = e_out / max(n_out, 1)
            out_of_lobe_frac = density_out / max(density_in, 1e-9)
        else:
            out_of_lobe_frac = 1.0

        # ---- A anchor ratio (quality feature) ----
        A_anchor_ratio = float(A_used / A_fit) if A_fit > 1e-6 else 1.0

        # ---- Cruise velocity coefficient of variation ----
        if abs(v_peak_meas) > 1e-6:
            cruise_v_cv = cruise_v_std / abs(v_peak_meas)
        else:
            cruise_v_cv = 0.0

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
            # Outlier-catcher features (new)
            A_anchor_ratio=A_anchor_ratio,
            out_of_lobe_residual_frac=out_of_lobe_frac,
            cruise_v_cv=cruise_v_cv,
            min_segment_samples=c.min_segment_samples,
            grav_window_sec=c.grav_window_sec,
            grav_stability_max=c.grav_stability_max,
            quality_score_reject=c.quality_score_reject,
            min_r2_short=c.min_r2_short, min_r2_mid=c.min_r2_mid,
            min_r2_long=c.min_r2_long,
            overlap_delta=c.overlap_delta,
            min_delta_tc_sec=c.min_delta_tc_sec,
            min_distance_m=c.min_distance_m,
            max_distance_m=c.max_distance_m,
            min_active_fraction=c.min_active_fraction,
            max_A_anchor_ratio=c.max_A_anchor_ratio,
            min_A_anchor_ratio=c.min_A_anchor_ratio,
            max_out_of_lobe_frac=c.max_out_of_lobe_frac,
            max_cruise_v_cv=c.max_cruise_v_cv,
            max_pre_post_angle_deg=c.max_pre_post_angle_deg,
        )

        # Template reconstruction is mode-aware so downstream
        # visualisation shows what the estimator actually fitted.
        if mode == "joined":
            t_mid = 0.5 * (fit.t_c1 + fit.t_c2)
            a_template = fit.sign * A_used * joined_kernel(
                inp.t_sec, t_mid, fit.W, fit.f,
            )
        else:
            a_template = (
                fit.sign * A_used * trapezoid_kernel(inp.t_sec, fit.t_c1, fit.W, fit.f)
                - fit.sign * A_used * trapezoid_kernel(inp.t_sec, fit.t_c2, fit.W, fit.f)
            )

        meta: dict[str, Any] = {
            "vert_method": vert_method,
            "mode": mode,
            "params": {
                "A_fit": fit.A, "A_used": A_used,
                "W": fit.W, "f": fit.f,
                "t_c1": fit.t_c1, "t_c2": fit.t_c2,
                "sign": fit.sign, "joint_r2": fit.joint_r2,
                "v_peak_measured": v_peak_meas,
            },
            "pair_r2": float(_pair_r2(pair_fit)),
            "joined_r2": float(_pair_r2(joined_fit)),
            "fit_r2_1": fit.r2_1, "fit_r2_2": fit.r2_2,
            "delta_tc_sec": fit.delta_t_c,
            "sigma_breakdown": sig,
            "A_anchor_ratio": A_anchor_ratio,
            "out_of_lobe_residual_frac": out_of_lobe_frac,
            "cruise_v_cv": cruise_v_cv,
            "cruise_width_sec": cruise_width,
            "zupt_height": zupt_height,
            "quality_features": q.features,
            "a_smooth": a_smooth,
            "a_template": a_template,
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
