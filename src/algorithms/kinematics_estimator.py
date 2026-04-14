"""
Kinematics-Based Optimal Elevator Height Estimator.

This module implements the core estimation algorithm that fits the
7-step S-curve elevator motion template to accelerometer measurements.

KEY INSIGHT (v2): Fitting raw acceleration has SNR ~1.5 in real data,
causing NLS to fit noise. Instead, we:
  1. Low-pass filter acceleration (3 Hz cutoff removes hand tremor)
  2. Integrate to velocity (natural low-pass, SNR ~10x better)
  3. Fit S-curve VELOCITY template (smooth bump, easy to match)
  4. Grid search over distance for robust global initialization

Two variants:
- Algorithm A: Accelerometer-only (magnitude-based, rotation-invariant)
- Algorithm B: Accelerometer + Orientation (3D vertical projection)
"""

import numpy as np
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from scipy.linalg import inv as scipy_inv
from typing import Dict, Optional, Tuple, Any
import warnings

from .scurve_model import (
    SCurveParams, SCurveProfile, compute_phase_durations,
    generate_profile, generate_profile_vectorized,
    compute_prior_log_probability, PRIOR_PARAMS, FLOOR_HEIGHT_PRIOR,
)


# ============================================================
# Helper functions
# ============================================================

def _estimate_noise_std(residuals: np.ndarray) -> float:
    """Robust noise standard deviation estimate using MAD."""
    if len(residuals) == 0:
        return 0.1
    mad = np.median(np.abs(residuals - np.median(residuals)))
    return max(1.4826 * mad, 0.001)


def _quaternion_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """
    Rotate vectors v by quaternions q (Hamilton convention: qw, qx, qy, qz).
    q: (N, 4), v: (N, 3) -> (N, 3)
    """
    qw, qx, qy, qz = q[:, 0], q[:, 1], q[:, 2], q[:, 3]
    vx, vy, vz = v[:, 0], v[:, 1], v[:, 2]

    # q * v * q^{-1} using expanded formula
    t0 = 2.0 * (qx * vx + qy * vy + qz * vz)
    t1 = qw * qw - (qx * qx + qy * qy + qz * qz)

    rx = t1 * vx + t0 * qx + 2.0 * qw * (qy * vz - qz * vy)
    ry = t1 * vy + t0 * qy + 2.0 * qw * (qz * vx - qx * vz)
    rz = t1 * vz + t0 * qz + 2.0 * qw * (qx * vy - qy * vx)

    return np.column_stack([rx, ry, rz])


def _lowpass_filter(data: np.ndarray, fs: float,
                    cutoff: float = 3.0, order: int = 2) -> np.ndarray:
    """Apply zero-phase Butterworth low-pass filter."""
    nyq = fs / 2.0
    if cutoff >= nyq or len(data) < 15:
        return data.copy()
    b, a = butter(order, cutoff / nyq, btype='low')
    try:
        return filtfilt(b, a, data)
    except Exception:
        return data.copy()


def _zupt_integrate(a_measured: np.ndarray, t: np.ndarray):
    """ZUPT (Zero-velocity Update) integration with drift correction."""
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.01

    vel = np.cumsum(a_measured * dt)
    # Linear drift correction (enforce v(end) = 0)
    n = len(vel)
    if n > 1:
        drift = vel[-1]
        vel -= np.linspace(0, drift, n)

    pos = np.cumsum(vel * dt)
    return pos, vel


def _initial_guess_from_data(t: np.ndarray, a_measured: np.ndarray,
                              building_type: str = 'generic') -> dict:
    """
    Compute initial parameter guess from acceleration data.
    Uses ZUPT integration for distance, signal analysis for kinematics.
    """
    n = len(t)
    dt = np.diff(t, prepend=t[0])
    dt[0] = dt[1] if len(dt) > 1 else 0.01

    # Find motion region
    threshold = 0.08
    abs_a = np.abs(a_measured)
    smoothed = np.convolve(abs_a, np.ones(15) / 15, mode='same')
    above = np.where(smoothed > threshold)[0]

    if len(above) > 0:
        motion_start = above[0]
        motion_end = above[-1]
        t_offset = t[motion_start]
        motion_duration = t[motion_end] - t[motion_start]
    else:
        t_offset = t[n // 4]
        motion_duration = (t[-1] - t[0]) * 0.5

    # Peak acceleration
    window = max(5, int(0.05 * n))
    a_smooth = np.convolve(abs_a, np.ones(window) / window, mode='same')
    a_peak = np.max(a_smooth)
    a_max_guess = np.clip(a_peak * 0.8, 0.3, 2.5)

    j_max_guess = np.clip(a_max_guess / 0.5, 0.5, 5.0)
    v_max_guess = np.clip(a_max_guess * motion_duration * 0.2, 0.2, 8.0)

    # ZUPT distance
    pos, _ = _zupt_integrate(a_measured, t)
    distance_guess = abs(pos[-1])
    distance_guess = max(distance_guess, 0.5)

    # Blend with prior
    prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS['generic'])
    floor_h = FLOOR_HEIGHT_PRIOR.get(building_type, FLOOR_HEIGHT_PRIOR['generic'])
    n_floors = max(1, round(distance_guess / floor_h['mean']))
    distance_prior = n_floors * floor_h['mean']
    distance_guess = 0.7 * distance_guess + 0.3 * distance_prior

    return {
        'j_max': j_max_guess,
        'a_max': a_max_guess,
        'v_max': v_max_guess,
        'distance': distance_guess,
        't_offset': t_offset,
    }


# ============================================================
# Velocity-Domain Grid Search
# ============================================================

def _grid_search_velocity(t, v_measured, direction, d_zupt,
                           t_offset_guess, building_type):
    """
    Brute-force grid search in velocity domain.
    
    For each candidate (j, a, v_max, distance, t_offset), generate the
    S-curve velocity template and compute the RSS against measured velocity.
    
    This is fast because each evaluation is O(n) and we use vectorized numpy.
    The velocity template is a smooth bump — much more distinctive than
    the acceleration template — so the search landscape has fewer local minima.
    """
    prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS['generic'])
    
    # Distance grid: centered on ZUPT, including floor-height multiples
    d_candidates = set()
    for scale in np.arange(0.4, 2.21, 0.15):
        d_candidates.add(round(d_zupt * scale, 1))
    for fh in [3.0, 3.3, 4.0, 4.5]:
        for nf in range(1, max(2, int(d_zupt * 2.5 / fh) + 2)):
            d_candidates.add(round(fh * nf, 1))
    d_candidates = sorted([d for d in d_candidates if 0.5 <= d <= 200.0])
    
    # Kinematic grid (coarse)
    j_vals = [1.0, 1.5, 2.0, 3.0]
    a_vals = [0.6, 0.8, 1.0, 1.2, 1.5]
    v_vals = [0.5, 1.0, 1.5, 2.5]
    t_offsets = [t_offset_guess + dt for dt in [-0.5, -0.2, 0.0, 0.2, 0.5]]
    
    best_cost = np.inf
    best_params = None
    
    # Phase 1: Fix kinematics to prior mean, search distance + t_offset
    j_def = prior['j_max']['mean']
    a_def = prior['a_max']['mean']
    v_def = prior['v_max']['mean']
    
    for d in d_candidates:
        for t_off in t_offsets:
            try:
                _, v_tmpl, _ = generate_profile_vectorized(
                    t, j_def, a_def, v_def, d, direction, t_off)
                cost = np.sum((v_measured - v_tmpl) ** 2)
                if cost < best_cost:
                    best_cost = cost
                    best_params = [j_def, a_def, v_def, d, t_off]
            except Exception:
                pass
    
    # Phase 2: Around best distance, search kinematic params
    if best_params is not None:
        d_best = best_params[3]
        t_off_best = best_params[4]
        
        for j in j_vals:
            for a_val in a_vals:
                for v in v_vals:
                    for d in [d_best * 0.9, d_best, d_best * 1.1]:
                        try:
                            _, v_tmpl, _ = generate_profile_vectorized(
                                t, j, a_val, v, max(0.5, d),
                                direction, t_off_best)
                            cost = np.sum((v_measured - v_tmpl) ** 2)
                            if cost < best_cost:
                                best_cost = cost
                                best_params = [j, a_val, v, max(0.5, d), t_off_best]
                        except Exception:
                            pass
    
    return best_params, best_cost


# ============================================================
# Core Velocity-Domain NLS Fitting
# ============================================================

def _velocity_residuals(params_vec, t, v_measured, direction, sigma_v,
                         building_type, prior_weight):
    """Compute weighted residuals in velocity domain."""
    j_max, a_max, v_max, distance, t_offset = params_vec

    if j_max <= 0.01 or a_max <= 0.01 or v_max <= 0.01 or distance <= 0.1:
        return np.full(len(v_measured) + 5, 1e6)

    try:
        _, v_template, _ = generate_profile_vectorized(
            t, j_max, a_max, v_max, distance, direction, t_offset
        )
        data_res = (v_measured - v_template) / sigma_v

        # Prior regularization (gentle - data should dominate)
        prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS['generic'])
        prior_res = np.zeros(5)
        for idx, (val, key) in enumerate([(j_max, 'j_max'), (a_max, 'a_max'),
                                           (v_max, 'v_max')]):
            p = prior[key]
            prior_res[idx] = prior_weight * (val - p['mean']) / p['std']

        floor_h = FLOOR_HEIGHT_PRIOR.get(building_type, FLOOR_HEIGHT_PRIOR['generic'])
        n_floors = max(1, round(distance / floor_h['mean']))
        expected_d = n_floors * floor_h['mean']
        sigma_d = np.sqrt((n_floors * floor_h['std']) ** 2 + 0.5 ** 2)
        prior_res[3] = prior_weight * (distance - expected_d) / sigma_d
        prior_res[4] = 0.0

        return np.concatenate([data_res, prior_res])
    except Exception:
        return np.full(len(v_measured) + 5, 1e6)


def fit_scurve_params(t: np.ndarray, a_measured: np.ndarray,
                       direction: int = 1,
                       building_type: str = 'generic',
                       initial_guess: Optional[dict] = None,
                       prior_weight: float = 0.5) -> Dict[str, Any]:
    """
    Fit 7-step S-curve parameters using velocity-domain approach.
    
    The key insight: real phone accelerometer data has SNR ~1.5, making
    direct acceleration-domain template matching unreliable. By integrating
    to velocity (a natural low-pass filter), SNR improves to ~10+, and
    the velocity S-curve shape (smooth bump) is much more distinctive.
    
    Pipeline:
      1. Low-pass filter acceleration at 3 Hz (removes hand tremor)
      2. ZUPT integration → velocity (smooth, high SNR)
      3. Grid search in velocity domain (robust global init)
      4. NLS refinement in velocity domain
      5. CI from Fisher Information on velocity Jacobian
    """
    n = len(t)
    duration = t[-1] - t[0]
    fs = n / max(duration, 0.01)

    # Auto-detect direction
    if direction == 0:
        pos, _ = _zupt_integrate(a_measured, t)
        direction = 1 if pos[-1] > 0 else -1

    # ---- Stage 1: Preprocessing ----
    cutoff_hz = min(3.0, fs / 2.5)
    a_filtered = _lowpass_filter(a_measured, fs, cutoff=cutoff_hz)

    # Integrate filtered acceleration → velocity with ZUPT
    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 0.01

    vel_raw = np.cumsum(a_filtered * dt_arr)
    vel_measured = vel_raw - np.linspace(0, vel_raw[-1], n)

    # ZUPT distance
    pos_zupt = np.cumsum(vel_measured * dt_arr)
    d_zupt = abs(pos_zupt[-1])
    d_zupt = max(d_zupt, 0.5)

    # Velocity noise estimate
    vel_smooth = _lowpass_filter(vel_measured, fs, cutoff=1.0)
    sigma_v = max(_estimate_noise_std(vel_measured - vel_smooth), 0.001)
    sigma_a = max(_estimate_noise_std(a_filtered), 0.01)

    # Initial guess from filtered data
    if initial_guess is None:
        initial_guess = _initial_guess_from_data(t, a_filtered, building_type)

    # Bounds
    bounds_lower = [0.3,  0.2,  0.1,   0.5,    t[0] - 2.0]
    bounds_upper = [8.0,  3.0,  10.0,  200.0,  t[-1] + 1.0]

    x0 = [
        np.clip(initial_guess['j_max'], bounds_lower[0], bounds_upper[0]),
        np.clip(initial_guess['a_max'], bounds_lower[1], bounds_upper[1]),
        np.clip(initial_guess['v_max'], bounds_lower[2], bounds_upper[2]),
        np.clip(initial_guess['distance'], bounds_lower[3], bounds_upper[3]),
        np.clip(initial_guess['t_offset'], bounds_lower[4], bounds_upper[4]),
    ]

    # ---- Stage 2: Grid search in velocity domain ----
    grid_params, grid_cost = _grid_search_velocity(
        t, vel_measured, direction, d_zupt,
        x0[4], building_type
    )

    # ---- Stage 3: NLS refinement in velocity domain ----
    x0_list = []
    if grid_params is not None:
        x0_list.append(list(grid_params))
    x0_list.append(x0)

    # ZUPT-anchored starts
    prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS['generic'])
    for d_mult in [0.8, 1.0, 1.2]:
        x0_list.append([
            prior['j_max']['mean'],
            prior['a_max']['mean'],
            prior['v_max']['mean'],
            np.clip(d_zupt * d_mult, bounds_lower[3], bounds_upper[3]),
            x0[4],
        ])

    best_cost = np.inf
    best_result = None
    best_x = np.array(x0)

    for x_init in x0_list:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                opt = least_squares(
                    _velocity_residuals, x_init,
                    args=(t, vel_measured, direction, sigma_v,
                          building_type, prior_weight),
                    bounds=(bounds_lower, bounds_upper),
                    method='trf', ftol=1e-7, xtol=1e-7, gtol=1e-7,
                    max_nfev=150, verbose=0,
                )
            if opt.cost < best_cost:
                best_cost = opt.cost
                best_result = opt
                best_x = opt.x.copy()
        except Exception:
            pass

    if best_result is not None:
        converged = best_result.success or best_cost < n * 5
        x_opt = best_x
        residuals_full = best_result.fun
    else:
        converged = False
        x_opt = np.array(x0)
        residuals_full = _velocity_residuals(
            x_opt, t, vel_measured, direction,
            sigma_v, building_type, prior_weight)

    j_max_fit, a_max_fit, v_max_fit, distance_fit, t_offset_fit = x_opt

    # Generate fitted templates
    a_template, v_template, s_template = generate_profile_vectorized(
        t, j_max_fit, a_max_fit, v_max_fit, distance_fit, direction, t_offset_fit
    )

    # Residuals
    data_residuals_v = residuals_full[:n] * sigma_v
    data_residuals_a = a_filtered - a_template
    sigma_noise = max(_estimate_noise_std(data_residuals_a), 0.01)

    # Profile
    try:
        profile = compute_phase_durations(j_max_fit, a_max_fit, v_max_fit,
                                           distance_fit)
    except Exception:
        profile = None

    # ---- Stage 4: Confidence Intervals ----
    cov_matrix = _compute_covariance_velocity(
        t, vel_measured, x_opt, direction, sigma_v, building_type
    )

    if cov_matrix is not None and cov_matrix.shape == (5, 5) and cov_matrix[3, 3] > 0:
        distance_std = np.sqrt(cov_matrix[3, 3])
        # The velocity-domain FIM gives tighter CRB than reality because:
        # (1) ZUPT drift correction introduces correlated errors
        # (2) Low-pass filtering smooths but doesn't remove all noise  
        # (3) Real elevators deviate from ideal S-curve
        # Empirical calibration requires ~7x safety factor
        CI_SAFETY_FACTOR = 7.0
        distance_ci_90 = 1.645 * distance_std * CI_SAFETY_FACTOR
    else:
        distance_ci_90 = 0.3 * abs(distance_fit) + 1.0

    # Velocity fit quality → widen CI if fit is poor
    v_rss = np.sum(data_residuals_v ** 2)
    v_expected_rss = n * sigma_v ** 2
    v_fit_ratio = v_rss / (v_expected_rss + 1e-10)
    if v_fit_ratio > 2.0:
        distance_ci_90 *= min(v_fit_ratio / 2.0, 3.0)

    # ZUPT cross-check: CI must cover disagreement
    d_nls = abs(distance_fit)
    zupt_disagreement = abs(d_zupt - d_nls)
    if zupt_disagreement > distance_ci_90 * 0.5:
        distance_ci_90 = max(distance_ci_90, zupt_disagreement * 1.2)

    # Distance-proportional CI floor: longer rides have more uncertainty
    # Minimum ~25% of distance or 1.5m, whichever is larger
    ci_floor = max(1.5, abs(distance_fit) * 0.25)
    distance_ci_90 = max(min(distance_ci_90, 50.0), ci_floor)

    # Quality Score
    quality_score, quality_details = _compute_quality_score(
        data_residuals_a, sigma_noise, x_opt, distance_ci_90,
        profile, building_type, converged
    )
    quality_details['d_zupt'] = float(d_zupt)
    quality_details['d_nls'] = float(d_nls)
    quality_details['v_fit_ratio'] = float(v_fit_ratio)

    return {
        'distance': float(abs(distance_fit)),
        'height': float(direction * abs(distance_fit)),
        'distance_ci_90': float(distance_ci_90),
        'direction': direction,
        'params': {
            'j_max': float(j_max_fit),
            'a_max': float(a_max_fit),
            'v_max': float(v_max_fit),
            'distance': float(distance_fit),
            't_offset': float(t_offset_fit),
        },
        'profile': profile,
        'residuals': data_residuals_a,
        'sigma_noise': float(sigma_noise),
        'fit_quality': float(v_fit_ratio),
        'quality_score': float(quality_score),
        'quality_details': quality_details,
        'a_template': a_template,
        'v_template': v_template,
        'v_measured': vel_measured,
        's_template': s_template,
        'converged': converged,
        'cov_matrix': cov_matrix,
    }


# ============================================================
# Covariance (Velocity Domain)
# ============================================================

def _compute_covariance_velocity(t, v_measured, x_opt, direction,
                                  sigma_v, building_type):
    """Compute parameter covariance via velocity-domain Jacobian."""
    n_params = 5
    n_data = len(t)
    eps_rel = 1e-5

    J = np.zeros((n_data, n_params))

    try:
        _, v_base, _ = generate_profile_vectorized(
            t, x_opt[0], x_opt[1], x_opt[2], x_opt[3], direction, x_opt[4]
        )
    except Exception:
        return None

    for j in range(n_params):
        x_pert = x_opt.copy()
        delta = max(eps_rel * abs(x_opt[j]), 1e-6)
        x_pert[j] += delta

        try:
            _, v_pert, _ = generate_profile_vectorized(
                t, x_pert[0], x_pert[1], x_pert[2], x_pert[3],
                direction, x_pert[4]
            )
            J[:, j] = (v_pert - v_base) / delta
        except Exception:
            J[:, j] = 0.0

    try:
        FIM = J.T @ J / (sigma_v ** 2 + 1e-16)

        # Prior regularization
        prior = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS['generic'])
        for idx, key in enumerate(['j_max', 'a_max', 'v_max']):
            p = prior[key]
            FIM[idx, idx] += 1.0 / (p['std'] ** 2)

        floor_h = FLOOR_HEIGHT_PRIOR.get(building_type,
                                          FLOOR_HEIGHT_PRIOR['generic'])
        n_floors = max(1, round(x_opt[3] / floor_h['mean']))
        sigma_d = np.sqrt((n_floors * floor_h['std']) ** 2 + 0.5 ** 2)
        FIM[3, 3] += 1.0 / (sigma_d ** 2)

        FIM += np.eye(n_params) * 1e-10

        cov = scipy_inv(FIM)
        if np.any(np.diag(cov) < 0):
            return None
        return cov
    except Exception:
        return None


# ============================================================
# Quality Score System
# ============================================================

def _compute_quality_score(data_residuals, sigma_noise, x_opt,
                            distance_ci_90, profile, building_type,
                            converged):
    """
    Compute quality score for estimation reliability.
    
    Score Interpretation:
        0.0 – 2.0: Excellent
        2.0 – 4.0: Good
        4.0 – 6.0: Marginal
        6.0+     : Poor (consider rejecting)
    """
    details = {}
    score = 0.0

    n = len(data_residuals)
    if n < 10:
        return 10.0, {'reason': 'too_few_samples'}

    # 1. Fit quality (reduced chi-squared)
    if sigma_noise > 0:
        chi2 = np.sum(data_residuals ** 2) / (n * sigma_noise ** 2)
    else:
        chi2 = float('inf')
    details['chi_squared'] = float(chi2)
    if chi2 > 5.0:
        score += 3.0
    elif chi2 > 3.0:
        score += min((chi2 - 1.0) * 0.8, 3.0)
    elif chi2 > 1.5:
        score += (chi2 - 1.0) * 0.3

    # 2. Parameter plausibility
    j_max, a_max, v_max, distance, t_offset = x_opt
    log_prior = compute_prior_log_probability(j_max, a_max, v_max, distance,
                                               building_type)
    details['log_prior'] = float(log_prior)
    if log_prior == -np.inf:
        score += 4.0
    elif log_prior < -10:
        score += 3.0
    elif log_prior < -5:
        score += 1.5
    elif log_prior < -2:
        score += 0.5

    # 3. Confidence interval width (relative to distance)
    details['ci_90_width'] = float(distance_ci_90)
    ci_ratio = distance_ci_90 / max(distance, 1.0)
    details['ci_ratio'] = float(ci_ratio)
    if ci_ratio > 1.0:
        score += 3.0
    elif ci_ratio > 0.6:
        score += 2.0
    elif ci_ratio > 0.4:
        score += 0.5

    # 4. Convergence
    details['converged'] = converged
    if not converged:
        score += 2.0

    # 5. Residual autocorrelation
    if n > 30:
        res_centered = data_residuals - np.mean(data_residuals)
        denom = np.sum(res_centered ** 2)
        if denom > 0:
            acf1 = np.sum(res_centered[:-1] * res_centered[1:]) / denom
        else:
            acf1 = 0.0
        details['residual_acf1'] = float(acf1)
        if abs(acf1) > 0.6:
            score += 2.0
        elif abs(acf1) > 0.4:
            score += 1.0
        elif abs(acf1) > 0.25:
            score += 0.3

    # 6. Profile consistency
    if profile is not None:
        details['profile_type'] = profile.profile_type
        details['total_time'] = float(profile.total_time)
        if profile.total_time < 0.5 or profile.total_time > 120:
            score += 1.5
    else:
        score += 2.0

    details['total_score'] = float(score)
    return float(score), details


# ============================================================
# Algorithm A: Accelerometer-Only (Magnitude-Based)
# ============================================================

def estimate_height_accel_only(t: np.ndarray, ax: np.ndarray,
                                ay: np.ndarray, az: np.ndarray,
                                pre_ax: Optional[np.ndarray] = None,
                                pre_ay: Optional[np.ndarray] = None,
                                pre_az: Optional[np.ndarray] = None,
                                building_type: str = 'generic',
                                prior_weight: float = 0.5) -> Dict[str, Any]:
    """
    Algorithm A: Accelerometer-only height estimation.
    
    Uses |a(t)| - g as vertical acceleration proxy (rotation-invariant).
    """
    mag = np.sqrt(ax ** 2 + ay ** 2 + az ** 2)

    if pre_ax is not None and len(pre_ax) > 10:
        pre_mag = np.sqrt(pre_ax ** 2 + pre_ay ** 2 + pre_az ** 2)
        g_est = np.median(pre_mag)
    else:
        g_est = np.median(mag)

    a_vert = mag - g_est

    # ---- Robust direction detection ----
    # Method 1: ZUPT integration direction (most reliable)
    pos_zupt, _ = _zupt_integrate(a_vert, t)
    d_zupt = pos_zupt[-1]
    zupt_direction = 1 if d_zupt > 0 else -1
    zupt_confidence = abs(d_zupt) / max(np.std(a_vert) * (t[-1] - t[0]) * 0.1, 0.1)
    
    # Method 2: First-pulse analysis  
    threshold = 0.1
    smoothed = np.convolve(np.abs(a_vert), np.ones(15) / 15, mode='same')
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return _make_reject_result(t, 'No significant motion detected',
                                    'accel_only')
    first_chunk = above[:max(1, len(above) // 3)]
    mean_first = np.mean(a_vert[first_chunk])
    pulse_direction = 1 if mean_first > 0 else -1
    
    # If methods agree, use their consensus. If they disagree, try both.
    if zupt_direction == pulse_direction or zupt_confidence > 3.0:
        direction = zupt_direction
        a_for_fitting = a_vert * direction
        result = fit_scurve_params(
            t, a_for_fitting,
            direction=1,
            building_type=building_type,
            prior_weight=prior_weight,
        )
    else:
        # Ambiguous direction — try both, keep better fit
        results = []
        for d_try in [1, -1]:
            a_try = a_vert * d_try
            r = fit_scurve_params(
                t, a_try,
                direction=1,
                building_type=building_type,
                prior_weight=prior_weight,
            )
            r['_try_direction'] = d_try
            results.append(r)
        
        # Pick the one with better velocity fit quality
        result = min(results, key=lambda r: r.get('fit_quality', 999))
        direction = result.pop('_try_direction')
        for r in results:
            r.pop('_try_direction', None)
        a_for_fitting = a_vert * direction

    result['height'] = float(direction * abs(result['distance']))
    result['direction'] = direction
    result['method'] = 'accel_only'
    result['a_measured_for_fit'] = a_for_fitting

    # Rejection logic
    result['rejected'] = False
    result['reject_reason'] = ''

    if result['quality_score'] > 7.0:
        result['rejected'] = True
        result['reject_reason'] = (
            f"Quality score too high: {result['quality_score']:.1f}")
    elif result['distance_ci_90'] > max(10.0, result['distance'] * 0.6):
        result['rejected'] = True
        result['reject_reason'] = (
            f"CI too wide: +/-{result['distance_ci_90']:.1f}m")
    elif not result['converged'] and result['quality_score'] > 6.0:
        result['rejected'] = True
        result['reject_reason'] = 'NLS did not converge + poor quality'
    elif result['distance'] < 0.5:
        result['rejected'] = True
        result['reject_reason'] = (
            f"Distance too small: {result['distance']:.2f}m")

    return result


# ============================================================
# Algorithm B: Accelerometer + Orientation
# ============================================================

def estimate_height_with_orientation(t: np.ndarray, ax: np.ndarray,
                                      ay: np.ndarray, az: np.ndarray,
                                      t_ori: np.ndarray,
                                      qw: np.ndarray, qx: np.ndarray,
                                      qy: np.ndarray, qz: np.ndarray,
                                      building_type: str = 'generic',
                                      prior_weight: float = 0.5) -> Dict[str, Any]:
    """
    Algorithm B: Accelerometer + Orientation height estimation.
    
    Uses quaternion orientation to project acceleration into world frame,
    canceling horizontal noise (walking, fidgeting, pocket mode).
    """
    n = len(t)

    # Interpolate orientation to accelerometer timestamps
    qw_i = np.interp(t, t_ori, qw)
    qx_i = np.interp(t, t_ori, qx)
    qy_i = np.interp(t, t_ori, qy)
    qz_i = np.interp(t, t_ori, qz)

    # Normalize
    q_norm = np.sqrt(qw_i ** 2 + qx_i ** 2 + qy_i ** 2 + qz_i ** 2)
    q_norm = np.maximum(q_norm, 1e-10)
    quats = np.column_stack([qw_i, qx_i, qy_i, qz_i]) / q_norm[:, None]

    # Rotate to world frame
    acc_body = np.column_stack([ax, ay, az])
    acc_world = _quaternion_rotate(quats, acc_body)

    # Vertical = Z minus gravity
    a_vertical = acc_world[:, 2] - 9.81

    # Direction detection
    threshold = 0.1
    smoothed = np.convolve(np.abs(a_vertical), np.ones(15) / 15, mode='same')
    above = np.where(smoothed > threshold)[0]
    if len(above) == 0:
        return _make_reject_result(t, 'No significant vertical motion',
                                    'accel_orientation')

    first_chunk = above[:max(1, len(above) // 3)]
    direction = 1 if np.mean(a_vertical[first_chunk]) > 0 else -1

    a_for_fitting = a_vertical * direction

    result = fit_scurve_params(
        t, a_for_fitting,
        direction=1,
        building_type=building_type,
        prior_weight=prior_weight,
    )

    result['height'] = float(direction * abs(result['distance']))
    result['direction'] = direction
    result['method'] = 'accel_orientation'
    result['a_vertical_world'] = a_vertical
    result['a_measured_for_fit'] = a_for_fitting

    # Rejection
    result['rejected'] = False
    result['reject_reason'] = ''

    if result['quality_score'] > 7.0:
        result['rejected'] = True
        result['reject_reason'] = (
            f"Quality score too high: {result['quality_score']:.1f}")
    elif result['distance_ci_90'] > max(10.0, result['distance'] * 0.6):
        result['rejected'] = True
        result['reject_reason'] = (
            f"CI too wide: +/-{result['distance_ci_90']:.1f}m")
    elif not result['converged'] and result['quality_score'] > 6.0:
        result['rejected'] = True
        result['reject_reason'] = 'NLS did not converge + poor quality'
    elif result['distance'] < 0.5:
        result['rejected'] = True
        result['reject_reason'] = (
            f"Distance too small: {result['distance']:.2f}m")

    return result


def _make_reject_result(t, reason, method):
    """Create a rejection result dict."""
    n = len(t)
    return {
        'distance': 0.0,
        'height': 0.0,
        'distance_ci_90': float('inf'),
        'direction': 0,
        'params': {},
        'profile': None,
        'residuals': np.zeros(n),
        'sigma_noise': float('inf'),
        'fit_quality': float('inf'),
        'quality_score': 10.0,
        'quality_details': {'reason': reason},
        'a_template': np.zeros(n),
        'v_template': np.zeros(n),
        'v_measured': np.zeros(n),
        's_template': np.zeros(n),
        'converged': False,
        'cov_matrix': None,
        'rejected': True,
        'reject_reason': reason,
        'method': method,
    }
