"""7-step S-curve fitter as a first-class :class:`PredictAlgorithm`.

Public entry points:

  * :class:`ScurveAccelEstimator` — predict / calibrate / save / load.
  * :meth:`ScurveAccelEstimator.predict_segment` — returns a
    :class:`PredictionOutput` with the fitted Δh, a 90% CI
    (theoretical σ ✕ conformal multiplier), and a quality verdict.

The velocity-domain fitter, grid-search initialisation, Bayesian
prior regularisation, and CRB covariance follow the main-branch
research report; all three are unchanged except for refactoring into
the :class:`ScurveAccelConfig` hyperparameter object.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from scipy.linalg import inv as scipy_inv
from scipy.optimize import least_squares

from ..common.accel_utils import (
    butter_lowpass,
    estimate_gravity_stationary,
    vertical_accel_magnitude,
    vertical_accel_projected,
    zupt_integrate,
)
from ..common.conformal import ConformalCalibrator
from ..common.types import CalibrationSample, PredictionOutput
from .config import ScurveAccelConfig
from .scurve_model import (
    FLOOR_HEIGHT_PRIOR,
    PRIOR_PARAMS,
    compute_phase_durations,
    compute_prior_log_probability,
    generate_profile_vectorized,
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _noise_std_mad(residuals: np.ndarray) -> float:
    if len(residuals) == 0:
        return 0.1
    mad = np.median(np.abs(residuals - np.median(residuals)))
    return max(1.4826 * mad, 0.001)


def _initial_guess(t, a, building_type):
    n = len(t)
    dt = np.diff(t, prepend=t[0]); dt[0] = dt[1] if len(dt) > 1 else 0.01
    abs_a = np.abs(a)
    sm = np.convolve(abs_a, np.ones(15) / 15, mode="same")
    above = np.where(sm > 0.08)[0]
    if len(above):
        t_offset = t[above[0]]
        motion_dur = t[above[-1]] - t[above[0]]
    else:
        t_offset = t[n // 4]; motion_dur = (t[-1] - t[0]) * 0.5

    win = max(5, int(0.05 * n))
    a_smooth = np.convolve(abs_a, np.ones(win) / win, mode="same")
    a_peak = float(np.max(a_smooth))
    a_max_g = float(np.clip(a_peak * 0.8, 0.3, 2.5))
    j_max_g = float(np.clip(a_max_g / 0.5, 0.5, 5.0))
    v_max_g = float(np.clip(a_max_g * motion_dur * 0.2, 0.2, 8.0))

    pos, _ = zupt_integrate(a, t)
    d_z = max(float(abs(pos[-1])), 0.5)

    fp = FLOOR_HEIGHT_PRIOR.get(building_type, FLOOR_HEIGHT_PRIOR["generic"])
    n_floors = max(1, round(d_z / fp["mean"]))
    d_guess = 0.7 * d_z + 0.3 * n_floors * fp["mean"]

    return {"j_max": j_max_g, "a_max": a_max_g, "v_max": v_max_g,
            "distance": d_guess, "t_offset": float(t_offset)}


def _grid_search_velocity(t, v_meas, direction, d_zupt, t_off_guess, building_type):
    prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS["generic"])
    # Distance candidates
    d_cand = set()
    for scale in np.arange(0.4, 2.21, 0.15):
        d_cand.add(round(d_zupt * scale, 1))
    for fh in (3.0, 3.3, 4.0, 4.5):
        for nf in range(1, max(2, int(d_zupt * 2.5 / fh) + 2)):
            d_cand.add(round(fh * nf, 1))
    d_cand = sorted([d for d in d_cand if 0.5 <= d <= 200.0])

    j_vals = [1.0, 1.5, 2.0, 3.0]
    a_vals = [0.6, 0.8, 1.0, 1.2, 1.5]
    v_vals = [0.5, 1.0, 1.5, 2.5]
    t_offs = [t_off_guess + d for d in (-0.5, -0.2, 0.0, 0.2, 0.5)]

    best_cost, best = np.inf, None

    # Phase 1: fix kinematics to prior mean, search (d, t_off)
    jm = prior["j_max"]["mean"]; am = prior["a_max"]["mean"]; vm = prior["v_max"]["mean"]
    for d in d_cand:
        for t_off in t_offs:
            _, v_t, _ = generate_profile_vectorized(t, jm, am, vm, d, direction, t_off)
            cost = float(np.sum((v_meas - v_t) ** 2))
            if cost < best_cost:
                best_cost, best = cost, [jm, am, vm, d, t_off]

    # Phase 2: around best d, sweep kinematics
    if best is not None:
        d_best, t_off_best = best[3], best[4]
        for j in j_vals:
            for a_val in a_vals:
                for v in v_vals:
                    for d in (d_best * 0.9, d_best, d_best * 1.1):
                        d = max(0.5, d)
                        _, v_t, _ = generate_profile_vectorized(t, j, a_val, v, d, direction, t_off_best)
                        cost = float(np.sum((v_meas - v_t) ** 2))
                        if cost < best_cost:
                            best_cost, best = cost, [j, a_val, v, d, t_off_best]
    return best, best_cost


def _velocity_residuals(x, t, v_meas, direction, sigma_v, building_type, prior_weight):
    j_max, a_max, v_max, distance, t_offset = x
    if j_max <= 0.01 or a_max <= 0.01 or v_max <= 0.01 or distance <= 0.1:
        return np.full(len(v_meas) + 5, 1e6)
    _, v_t, _ = generate_profile_vectorized(
        t, j_max, a_max, v_max, distance, direction, t_offset)
    data_res = (v_meas - v_t) / sigma_v

    prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS["generic"])
    prior_res = np.zeros(5)
    for idx, (val, key) in enumerate([(j_max, "j_max"), (a_max, "a_max"), (v_max, "v_max")]):
        q = prior[key]
        prior_res[idx] = prior_weight * (val - q["mean"]) / q["std"]

    fp = FLOOR_HEIGHT_PRIOR.get(building_type, FLOOR_HEIGHT_PRIOR["generic"])
    n_floors = max(1, round(distance / fp["mean"]))
    sd = np.sqrt((n_floors * fp["std"]) ** 2 + 0.5 ** 2)
    prior_res[3] = prior_weight * (distance - n_floors * fp["mean"]) / sd
    prior_res[4] = 0.0
    return np.concatenate([data_res, prior_res])


def _covariance_velocity(t, x, direction, sigma_v, building_type):
    n_p = 5
    eps = 1e-5
    _, v0, _ = generate_profile_vectorized(t, x[0], x[1], x[2], x[3], direction, x[4])
    J = np.zeros((len(t), n_p))
    for j in range(n_p):
        dx = max(eps * abs(x[j]), 1e-6)
        xp = x.copy(); xp[j] += dx
        _, v_p, _ = generate_profile_vectorized(
            t, xp[0], xp[1], xp[2], xp[3], direction, xp[4])
        J[:, j] = (v_p - v0) / dx
    try:
        FIM = J.T @ J / (sigma_v ** 2 + 1e-16)
        prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS["generic"])
        for idx, key in enumerate(["j_max", "a_max", "v_max"]):
            q = prior[key]
            FIM[idx, idx] += 1.0 / (q["std"] ** 2)
        fp = FLOOR_HEIGHT_PRIOR.get(building_type, FLOOR_HEIGHT_PRIOR["generic"])
        n_floors = max(1, round(x[3] / fp["mean"]))
        sd = np.sqrt((n_floors * fp["std"]) ** 2 + 0.5 ** 2)
        FIM[3, 3] += 1.0 / sd ** 2
        FIM += np.eye(n_p) * 1e-10
        return scipy_inv(FIM)
    except Exception:
        return None


def _fit_scurve(
    t: np.ndarray, a_meas: np.ndarray,
    direction: int, building_type: str, prior_weight: float,
) -> dict:
    """Full velocity-domain S-curve fit (returns fitted params, residual,
    distance CI from CRB, quality sub-components).
    """
    n = len(t)
    dt_arr = np.diff(t, prepend=t[0]); dt_arr[0] = dt_arr[1] if n > 1 else 0.01
    duration = t[-1] - t[0]
    fs = n / max(duration, 0.01)

    if direction == 0:
        pos, _ = zupt_integrate(a_meas, t)
        direction = 1 if pos[-1] > 0 else -1

    cutoff = min(3.0, fs / 2.5)
    a_f = butter_lowpass(a_meas, fs, cutoff=cutoff)

    v_raw = np.cumsum(a_f * dt_arr)
    v_meas = v_raw - np.linspace(0, v_raw[-1], n)
    pos_z = np.cumsum(v_meas * dt_arr)
    d_zupt = max(float(abs(pos_z[-1])), 0.5)

    v_smooth = butter_lowpass(v_meas, fs, cutoff=1.0)
    sigma_v = max(_noise_std_mad(v_meas - v_smooth), 0.001)

    init = _initial_guess(t, a_f, building_type)

    lo = [0.3, 0.2, 0.1, 0.5, t[0] - 2.0]
    hi = [8.0, 3.0, 10.0, 200.0, t[-1] + 1.0]
    x0 = [np.clip(init["j_max"], lo[0], hi[0]),
          np.clip(init["a_max"], lo[1], hi[1]),
          np.clip(init["v_max"], lo[2], hi[2]),
          np.clip(init["distance"], lo[3], hi[3]),
          np.clip(init["t_offset"], lo[4], hi[4])]

    grid, grid_cost = _grid_search_velocity(
        t, v_meas, direction, d_zupt, x0[4], building_type,
    )

    x0_list = []
    if grid is not None:
        x0_list.append(list(grid))
    x0_list.append(x0)

    prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS["generic"])
    for d_mult in (0.8, 1.0, 1.2):
        x0_list.append([
            prior["j_max"]["mean"], prior["a_max"]["mean"], prior["v_max"]["mean"],
            float(np.clip(d_zupt * d_mult, lo[3], hi[3])), x0[4],
        ])

    best_cost, best_x, best_res = np.inf, np.array(x0), None
    for xi in x0_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt = least_squares(
                    _velocity_residuals, xi,
                    args=(t, v_meas, direction, sigma_v, building_type, prior_weight),
                    bounds=(lo, hi), method="trf",
                    ftol=1e-7, xtol=1e-7, gtol=1e-7, max_nfev=150,
                )
            if opt.cost < best_cost:
                best_cost, best_x, best_res = opt.cost, opt.x.copy(), opt
        except Exception:
            pass

    converged = (best_res is not None) and (best_res.success or best_cost < n * 5)
    x = best_x
    residuals = best_res.fun if best_res is not None else np.zeros(n + 5)

    j, a_m, v_m, distance_fit, t_off = x
    a_tpl, v_tpl, s_tpl = generate_profile_vectorized(
        t, j, a_m, v_m, distance_fit, direction, t_off,
    )

    try:
        profile = compute_phase_durations(j, a_m, v_m, distance_fit)
    except Exception:
        profile = None

    # Residual statistics
    v_res = residuals[:n] * sigma_v
    a_res = a_f - a_tpl
    sigma_a = max(_noise_std_mad(a_res), 0.01)
    v_rss = float(np.sum(v_res ** 2))
    v_fit_ratio = v_rss / max(n * sigma_v ** 2, 1e-10)

    # CRB on distance
    cov = _covariance_velocity(t, x, direction, sigma_v, building_type)
    if cov is not None and cov.shape == (5, 5) and cov[3, 3] > 0:
        d_std = float(np.sqrt(cov[3, 3]))
    else:
        d_std = 0.3 * abs(distance_fit) + 1.0

    return {
        "j_max": float(j), "a_max": float(a_m), "v_max": float(v_m),
        "distance": float(abs(distance_fit)), "direction": int(direction),
        "height_diff": float(direction * abs(distance_fit)),
        "t_offset": float(t_off),
        "a_template": a_tpl, "v_template": v_tpl, "s_template": s_tpl,
        "v_measured": v_meas, "a_filtered": a_f,
        "residuals_a": a_res, "residuals_v": v_res,
        "sigma_a_residual": sigma_a, "sigma_v": sigma_v,
        "fit_ratio_v": float(v_fit_ratio),
        "converged": bool(converged),
        "profile": profile, "d_zupt": float(d_zupt),
        "distance_std_crb": float(d_std),
    }


def _quality_score(fit: dict, distance_ci_90: float, building_type: str) -> tuple[float, dict]:
    score = 0.0
    details = {}

    n = len(fit["residuals_a"])
    sigma = fit["sigma_a_residual"]
    if sigma > 0 and n >= 10:
        chi2 = float(np.sum(fit["residuals_a"] ** 2) / (n * sigma ** 2))
    else:
        chi2 = float("inf")
    details["chi_squared"] = chi2
    if chi2 > 5.0: score += 3.0
    elif chi2 > 3.0: score += min((chi2 - 1.0) * 0.8, 3.0)
    elif chi2 > 1.5: score += (chi2 - 1.0) * 0.3

    lp = compute_prior_log_probability(
        fit["j_max"], fit["a_max"], fit["v_max"], fit["distance"], building_type,
    )
    details["log_prior"] = lp
    if lp == -np.inf: score += 4.0
    elif lp < -10: score += 3.0
    elif lp < -5: score += 1.5
    elif lp < -2: score += 0.5

    ci_ratio = distance_ci_90 / max(fit["distance"], 1.0)
    details["ci_ratio"] = ci_ratio
    if ci_ratio > 1.0: score += 3.0
    elif ci_ratio > 0.6: score += 2.0
    elif ci_ratio > 0.4: score += 0.5

    if not fit["converged"]:
        score += 2.0

    if fit["profile"] is not None:
        details["profile_type"] = fit["profile"].profile_type
        details["total_time_s"] = fit["profile"].total_time
        if fit["profile"].total_time < 0.5 or fit["profile"].total_time > 120:
            score += 1.5
    else:
        score += 2.0

    # Residual auto-correlation — colourful residuals indicate un-modelled
    # systematic error (shaking, acceleration noise jumps).
    if n > 30:
        r = fit["residuals_a"] - np.mean(fit["residuals_a"])
        denom = float(np.sum(r ** 2))
        acf1 = float(np.sum(r[:-1] * r[1:]) / denom) if denom > 0 else 0.0
        details["residual_acf1"] = acf1
        if abs(acf1) > 0.6: score += 2.0
        elif abs(acf1) > 0.4: score += 1.0
        elif abs(acf1) > 0.25: score += 0.3

    details["total_score"] = float(score)
    return float(score), details


# ---------------------------------------------------------------------------
# Public estimator class
# ---------------------------------------------------------------------------

@dataclass
class _SegInp:
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
    phone: str


class ScurveAccelEstimator:
    def __init__(self, config: ScurveAccelConfig | None = None):
        self.config = config or ScurveAccelConfig()
        self.conformal = ConformalCalibrator(alpha=self.config.alpha)

    def save(self, path: Path | str) -> None:
        self.conformal.save(path)

    def load(self, path: Path | str) -> None:
        self.conformal = ConformalCalibrator.load(path)

    # ------------------------------------------------------------------
    # Input extraction + gravity fallback logic (shared shape with ZUPT)
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
        c = self.config
        pre_g, pre_mag, pre_s = estimate_gravity_stationary(
            inp.pre_ax, inp.pre_ay, inp.pre_az, fs=inp.fs, window_sec=c.grav_window_sec,
        )
        post_g, post_mag, post_s = estimate_gravity_stationary(
            inp.post_ax, inp.post_ay, inp.post_az, fs=inp.fs, window_sec=c.grav_window_sec,
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
    # Public inference
    # ------------------------------------------------------------------
    def predict_segment(
        self,
        ride: pd.DataFrame, phone_model: str = "",
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

        a_vert, vert_method = self._vertical(inp)

        # Direction from ZUPT drift + first-pulse consensus
        pos_zupt, _ = zupt_integrate(a_vert, inp.t_sec)
        z_dir = 1 if pos_zupt[-1] >= 0 else -1
        sm = np.convolve(np.abs(a_vert), np.ones(15) / 15, mode="same")
        above = np.where(sm > c.active_threshold_m_s2)[0]
        if len(above) == 0:
            return PredictionOutput(
                height_diff=0.0, ci_half_width=math.inf,
                theoretical_sigma=math.inf, accepted=False,
                quality_score=8.0, reject_reason="no_significant_motion",
                meta={"vert_method": vert_method},
            )
        first_chunk = above[: max(1, len(above) // 3)]
        p_dir = 1 if float(np.mean(a_vert[first_chunk])) > 0 else -1

        if z_dir == p_dir:
            direction = z_dir
            a_for_fit = a_vert * direction
            fit = _fit_scurve(inp.t_sec, a_for_fit, direction=1,
                              building_type=c.building_type,
                              prior_weight=c.prior_weight)
            fit["height_diff"] = direction * abs(fit["distance"])
            fit["direction"] = direction
        else:
            # Try both, keep best velocity-fit ratio
            results = []
            for d_try in (1, -1):
                r = _fit_scurve(inp.t_sec, a_vert * d_try, direction=1,
                                building_type=c.building_type,
                                prior_weight=c.prior_weight)
                r["height_diff"] = d_try * abs(r["distance"])
                r["direction"] = d_try
                results.append(r)
            fit = min(results, key=lambda r: r.get("fit_ratio_v", 1e9))

        # ---- Theoretical σ ----
        # Compose three per-segment components (variance-additive):
        #   (i)   CRB · safety factor               (Fisher information)
        #   (ii)  relative scale ∝ |predicted Δh|   (long rides drift
        #         proportionally more in practice than the CRB alone
        #         predicts, dominated by pathways-from-ideal-S-curve)
        #   (iii) ZUPT-vs-NLS disagreement         (cross-check)
        sigma_crb = c.ci_safety_factor * max(fit["distance_std_crb"], 0.0)
        sigma_rel = c.relative_sigma_factor * abs(fit["distance"])
        zupt_disagree = abs(fit["d_zupt"] - fit["distance"])
        sigma_disagree = 0.5 * zupt_disagree

        theo_sigma = math.sqrt(
            sigma_crb ** 2 + sigma_rel ** 2 + sigma_disagree ** 2
        )
        theo_sigma = max(theo_sigma, c.min_theoretical_sigma_m)

        # Enlarge σ when fit quality is poor (v_fit_ratio large) — more
        # honest per-segment CI that tracks ride-by-ride uncertainty.
        if fit["fit_ratio_v"] > 2.0:
            theo_sigma *= min(fit["fit_ratio_v"] / 2.0, 3.0)

        ci = self.conformal.half_width(theo_sigma)
        ci = max(c.ci_absolute_floor_m, min(ci, c.ci_absolute_cap_m))

        # ---- Quality + accept/reject ----
        quality, q_details = _quality_score(fit, ci, c.building_type)

        # ZUPT-vs-NLS disagreement: either the NLS found the wrong
        # direction or the ride is genuinely far from an S-curve. We
        # reject rides where the two estimates straddle zero *and* the
        # raw magnitudes disagree by > 40% — almost always a flipped
        # direction we can't recover from without orientation data.
        zupt_sign = 1 if fit["d_zupt"] >= 0 else -1
        nls_sign = fit["direction"] if fit["direction"] != 0 else 1
        zupt_disagree_frac = zupt_disagree / max(abs(fit["distance"]), 1e-6)
        direction_flip_suspected = (
            zupt_sign != nls_sign
            and zupt_disagree_frac > 0.4
            and abs(fit["d_zupt"]) > 1.0
        )

        accepted = (
            quality <= c.quality_score_reject
            and c.min_distance_m <= fit["distance"] <= c.max_distance_m
            and fit["converged"]
            and not direction_flip_suspected
        )
        reject_reason = ""
        if quality > c.quality_score_reject:
            reject_reason = f"quality_score_{quality:.1f}"
        elif fit["distance"] < c.min_distance_m:
            reject_reason = f"distance_too_small_{fit['distance']:.2f}m"
        elif fit["distance"] > c.max_distance_m:
            reject_reason = f"distance_too_large_{fit['distance']:.1f}m"
        elif not fit["converged"]:
            reject_reason = "nls_did_not_converge"
        elif direction_flip_suspected:
            reject_reason = "direction_flip_vs_zupt"

        meta: dict[str, Any] = {
            "vert_method": vert_method,
            "d_zupt": fit["d_zupt"],
            "d_nls": fit["distance"],
            "params": {
                "j_max": fit["j_max"], "a_max": fit["a_max"],
                "v_max": fit["v_max"], "t_offset": fit["t_offset"],
            },
            "fit_ratio_v": fit["fit_ratio_v"],
            "converged": fit["converged"],
            "distance_std_crb": fit["distance_std_crb"],
            "sigma_a_residual": fit["sigma_a_residual"],
            "profile_type": q_details.get("profile_type", "unknown"),
            "quality_details": q_details,
            "a_template": fit["a_template"],
            "v_template": fit["v_template"],
            "v_measured": fit["v_measured"],
            "s_template": fit["s_template"],
        }
        return PredictionOutput(
            height_diff=float(fit["height_diff"]),
            ci_half_width=float(ci),
            theoretical_sigma=float(theo_sigma),
            accepted=bool(accepted),
            quality_score=float(quality),
            reject_reason=reject_reason,
            meta=meta,
        )

    def calibrate(self, samples: list[CalibrationSample]) -> dict:
        """Fit the conformal multiplier on all clean calibration samples.

        See the ZUPT estimator's docstring for why the accepted flag
        is *not* used as a filter at calibration time: the reported
        CI has to be calibrated against the whole clean population,
        not just the subset the quality filter likes.
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
