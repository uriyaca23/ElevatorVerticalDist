"""ZUPT (Zero-Velocity Update) accelerometer-only height-diff estimator.

The estimator is built as a class so that a calibrated conformal
multiplier can be carried alongside the hyperparameters after a
training pass. At inference time the caller only sees
:meth:`predict_segment`; calibration is a one-shot operation on
labelled training data via :meth:`calibrate`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from src.utils.accelerometer_utils import (
    estimate_gravity_stationary,
    vertical_accel_projected,
    zupt_integrate,
)
from src.utils.signal_processing import butter_lowpass
from src.utils.conformal import ConformalCalibrator
from src.utils.sensor_noise import get_phone_accel_noise_sigma
from ...common.types import CalibrationSample, PredictionOutput
from ...configTypes import ZuptAccelConfig
from .quality import assess as quality_assess
from .theoretical_ci import zupt_position_sigma


@dataclass
class _SegmentInputs:
    ax: np.ndarray
    ay: np.ndarray
    az: np.ndarray
    t_sec: np.ndarray
    fs: float
    pre_ax: np.ndarray
    pre_ay: np.ndarray
    pre_az: np.ndarray
    post_ax: np.ndarray
    post_ay: np.ndarray
    post_az: np.ndarray
    phone_model: str


def _find_motion_window(
    a_vert_magnitude: np.ndarray, fs: float,
    threshold: float, smooth: int, margin_sec: float,
) -> tuple[int, int]:
    n = len(a_vert_magnitude)
    if n == 0:
        return 0, 0
    kernel = np.ones(max(1, smooth)) / max(1, smooth)
    sm = np.convolve(np.abs(a_vert_magnitude), kernel, mode="same")
    active = np.where(sm > threshold)[0]
    if len(active) == 0:
        return 0, n - 1
    margin = int(fs * margin_sec)
    start = max(0, int(active[0]) - margin)
    end = min(n - 1, int(active[-1]) + margin)
    return start, end


def _zupt_integrate_windowed(
    a_vert: np.ndarray, t_sec: np.ndarray, start: int, end: int,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Zero-force a(t) outside [start, end] and double-integrate with
    linear drift correction applied across the window.

    Returns (position, velocity, n_active).
    """
    windowed = np.zeros_like(a_vert)
    windowed[start:end + 1] = a_vert[start:end + 1]
    pos, vel = zupt_integrate(windowed, t_sec)
    return pos, vel, float(end - start + 1)


class ZuptAccelEstimator:
    """ZUPT estimator with integrated quality filter + conformal CI.

    Usage pattern::

        est = ZuptAccelEstimator(ZuptAccelConfig())
        est.calibrate(calibration_samples)   # on train set only
        est.save(Path("checkpoint.json"))
        # ... later ...
        est.load(Path("checkpoint.json"))
        out = est.predict_segment(seg_df, phone_model="Pixel 10",
                                  pre=pre_df, post=post_df)
    """

    # ------------------------------------------------------------------
    # Construction / checkpoint
    # ------------------------------------------------------------------
    def __init__(self, config: ZuptAccelConfig | None = None):
        self.config = config or ZuptAccelConfig()
        self.conformal = ConformalCalibrator(alpha=self.config.alpha)

    def save(self, path: Path | str) -> None:
        self.conformal.save(path)

    def load(self, path: Path | str) -> None:
        self.conformal = ConformalCalibrator.load(path)

    # ------------------------------------------------------------------
    # Core estimate (no quality/CI)
    # ------------------------------------------------------------------
    def _extract_inputs(
        self,
        ride: pd.DataFrame,
        phone_model: str,
        pre: Optional[pd.DataFrame],
        post: Optional[pd.DataFrame],
    ) -> _SegmentInputs:
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

        def _axes(df: Optional[pd.DataFrame]) -> tuple[np.ndarray, ...]:
            if df is None or df.empty:
                return (np.array([]), np.array([]), np.array([]))
            return (
                np.asarray(df[c.ax_col].to_numpy(), dtype=float),
                np.asarray(df[c.ay_col].to_numpy(), dtype=float),
                np.asarray(df[c.az_col].to_numpy(), dtype=float),
            )

        pre_ax, pre_ay, pre_az = _axes(pre)
        post_ax, post_ay, post_az = _axes(post)

        return _SegmentInputs(
            ax=ax, ay=ay, az=az, t_sec=t_sec, fs=fs,
            pre_ax=pre_ax, pre_ay=pre_ay, pre_az=pre_az,
            post_ax=post_ax, post_ay=post_ay, post_az=post_az,
            phone_model=phone_model or c.default_phone,
        )

    def _pick_vertical_accel(self, inp: _SegmentInputs) -> tuple[np.ndarray, str, float]:
        """Return (a_vert_signed, method_used, g_reference)."""
        c = self.config

        pre_g, pre_mag, pre_stab = estimate_gravity_stationary(
            inp.pre_ax, inp.pre_ay, inp.pre_az,
            fs=inp.fs, window_sec=c.grav_window_sec,
        )
        post_g, post_mag, post_stab = estimate_gravity_stationary(
            inp.post_ax, inp.post_ay, inp.post_az,
            fs=inp.fs, window_sec=c.grav_window_sec,
        )
        pre_ok = 8.0 < pre_mag < 12.0 and pre_stab < c.grav_stability_max
        post_ok = 8.0 < post_mag < 12.0 and post_stab < c.grav_stability_max

        # Prefer average of pre + post gravity when both are stable —
        # this is the gold standard and cancels steady orientation drift.
        if pre_ok and post_ok:
            w_pre = 1.0 / max(pre_stab, 1e-3)
            w_post = 1.0 / max(post_stab, 1e-3)
            gvec = (pre_g * w_pre + post_g * w_post) / (w_pre + w_post)
            a_vert = vertical_accel_projected(inp.ax, inp.ay, inp.az, gvec)
            return a_vert, "projected_pre_post", float(np.linalg.norm(gvec))

        if pre_ok:
            a_vert = vertical_accel_projected(inp.ax, inp.ay, inp.az, pre_g)
            return a_vert, "projected_pre", float(pre_mag)

        if post_ok:
            a_vert = vertical_accel_projected(inp.ax, inp.ay, inp.az, post_g)
            return a_vert, "projected_post", float(post_mag)

        # Magnitude fallback — loses direction sign but robust
        mag = np.sqrt(inp.ax ** 2 + inp.ay ** 2 + inp.az ** 2)
        g_ref = float(np.median(mag))
        return mag - g_ref, "magnitude", g_ref

    # ------------------------------------------------------------------
    # Public inference
    # ------------------------------------------------------------------
    def predict_segment(
        self,
        ride: pd.DataFrame,
        phone_model: str = "",
        pre: Optional[pd.DataFrame] = None,
        post: Optional[pd.DataFrame] = None,
    ) -> PredictionOutput:
        c = self.config
        inp = self._extract_inputs(ride, phone_model, pre, post)

        if inp.ax.size < c.min_segment_samples:
            return PredictionOutput(
                height_diff=0.0, ci_half_width=math.inf,
                theoretical_sigma=math.inf, accepted=False,
                quality_score=10.0, reject_reason="segment_too_short",
                meta={"n_samples": int(inp.ax.size)},
            )

        # ---- Quality filter ----
        q = quality_assess(
            inp.ax, inp.ay, inp.az,
            inp.pre_ax, inp.pre_ay, inp.pre_az,
            inp.post_ax, inp.post_ay, inp.post_az,
            fs=inp.fs,
            min_segment_samples=c.min_segment_samples,
            grav_window_sec=c.grav_window_sec,
            grav_stability_max=c.grav_stability_max,
            grav_pre_post_angle_deg=c.grav_pre_post_angle_deg,
            max_ride_drift_deg=c.max_ride_drift_deg,
            max_peak_m_s2=c.max_peak_m_s2,
            min_active_fraction=c.min_active_fraction,
            quality_score_reject=c.quality_score_reject,
            active_threshold_m_s2=c.active_threshold_m_s2,
        )

        # ---- Vertical projection ----
        a_vert_raw, method, g_ref = self._pick_vertical_accel(inp)
        a_vert = butter_lowpass(a_vert_raw, inp.fs, c.lowpass_cutoff_hz)

        # ---- Active-motion window ----
        start, end = _find_motion_window(
            a_vert, inp.fs,
            threshold=c.active_threshold_m_s2,
            smooth=c.active_smooth_window,
            margin_sec=c.active_margin_sec,
        )
        if end <= start:
            return PredictionOutput(
                height_diff=0.0, ci_half_width=math.inf,
                theoretical_sigma=math.inf, accepted=False,
                quality_score=max(q.quality_score, 8.0),
                reject_reason="no_active_window",
                meta={"method": method, "features": q.features},
            )

        # ---- Double integration w/ ZUPT ----
        pos, vel, n_active = _zupt_integrate_windowed(a_vert, inp.t_sec, start, end)
        height_diff = float(pos[-1])

        # If we fell back to magnitude, sign from the first acceleration pulse
        if method == "magnitude":
            first_chunk = a_vert[start:start + max(3, (end - start) // 4 or 3)]
            pulse_sign = 1.0 if np.mean(first_chunk) > 0 else -1.0
            # ZUPT on |mag| always integrates to 0 by construction; sign the
            # magnitude once and re-integrate.
            a_signed = a_vert * pulse_sign
            pos_signed, vel, n_active = _zupt_integrate_windowed(
                a_signed, inp.t_sec, start, end,
            )
            height_diff = float(pos_signed[-1])
            pos = pos_signed

        # ---- Theoretical σ ----
        sigma_a = get_phone_accel_noise_sigma(inp.phone_model, inp.fs)
        dt_sec = (inp.t_sec[-1] - inp.t_sec[0]) / max(len(inp.t_sec) - 1, 1)
        pp_angle = float(q.features.get("pre_post_angle_deg", 0.0))
        theo_sigma = zupt_position_sigma(
            sigma_a=sigma_a, n_active=int(n_active), dt_sec=dt_sec,
            mechanical_jitter_m_s2=0.03,
            predicted_abs_dh_m=abs(height_diff),
            ride_drift_deg=float(q.features.get("max_gravity_drift_deg", 0.0)),
            pre_post_angle_deg=pp_angle if pp_angle >= 0 else 30.0,
            vert_method=method,
            relative_floor=0.05,
            min_sigma_m=c.min_theoretical_sigma_m,
        )

        # ---- Conformal CI ----
        ci = self.conformal.half_width(theo_sigma)
        # Floor / cap for sanity
        ci = max(c.ci_absolute_floor_m, min(ci, c.ci_absolute_cap_m))

        # ---- Post-fit rejection based on physical plausibility ----
        accepted = q.accept
        reject_reason = q.reject_reason
        if accepted and abs(height_diff) < c.min_displacement_m:
            accepted = False
            reject_reason = f"displacement_too_small_{abs(height_diff):.2f}m"
        if accepted and abs(height_diff) > c.max_distance_m:
            accepted = False
            reject_reason = f"displacement_too_large_{abs(height_diff):.1f}m"

        meta = {
            "method": method,
            "g_ref": float(g_ref),
            "n_active": int(n_active),
            "start_idx": int(start),
            "end_idx": int(end),
            "sigma_a_m_s2": float(sigma_a),
            "active_fraction": float(q.features.get("active_fraction", 0.0)),
            "features": q.features,
            "pos_curve": pos,
        }
        return PredictionOutput(
            height_diff=height_diff,
            ci_half_width=ci,
            theoretical_sigma=theo_sigma,
            accepted=accepted,
            quality_score=q.quality_score,
            reject_reason=reject_reason,
            meta=meta,
        )

    # ------------------------------------------------------------------
    # Calibration
    # ------------------------------------------------------------------
    def calibrate(self, samples: list[CalibrationSample]) -> dict:
        """Fit the conformal multiplier on the clean-only calibration pool.

        The project convention is that all reported metrics are
        computed on clean data regardless of the quality-filter
        verdict, so conformal is fit on the clean pool (not the
        accepted-clean subset). The filter still has a job:
        downstream inference code is expected to drop rejected
        predictions, but the CI returned alongside still has to be
        calibrated on the whole clean population so that a consumer
        who overrides the filter gets the promised $1-\\alpha$
        coverage.
        """
        usable = [s for s in samples if s.signal_clear]
        if not usable:
            return {"n_used": 0, "note": "no_usable_samples"}

        abs_errors = [s.abs_error for s in usable]
        sigmas = [s.theoretical_sigma for s in usable]
        self.conformal.fit(abs_errors, sigmas)

        n_accepted = sum(1 for s in usable if s.accepted)
        return {
            "n_used": len(usable),
            "n_accepted": n_accepted,
            "multiplier": self.conformal.multiplier,
            "p95_score": self.conformal.p95_score,
        }
