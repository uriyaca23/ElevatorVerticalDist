"""Quality filter for the ZUPT estimator.

Produces both a continuous score (higher = worse) and a binary
accept/reject flag with a free-form rejection reason. The score is
what we'll sort on in the analysis figures; the binary flag is what
the estimator returns to downstream inference code.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.accelerometer_utils import estimate_gravity_stationary


@dataclass
class ZuptQuality:
    accept: bool
    reject_reason: str
    quality_score: float
    features: dict


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("inf")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    cos = max(-1.0, min(1.0, cos))
    return float(np.degrees(np.arccos(cos)))


def _ride_gravity_drift(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    fs: float, chunk_sec: float = 1.0,
) -> tuple[float, float]:
    """Max angle (deg) the gravity vector sweeps across 1-s chunks of the
    ride, plus its std. A still phone held steady shows <5° drift;
    active handling shoots past 20°.
    """
    n = len(ax)
    chunk = max(10, int(fs * chunk_sec))
    n_chunks = max(1, n // chunk)
    if n_chunks < 2:
        return 0.0, 0.0

    gvecs = []
    for i in range(n_chunks):
        s = i * chunk
        e = min(s + chunk, n)
        gvecs.append(np.array([np.mean(ax[s:e]), np.mean(ay[s:e]), np.mean(az[s:e])]))

    g0 = gvecs[0]
    angles = [_angle_deg(g0, g) for g in gvecs[1:]]
    if not angles:
        return 0.0, 0.0
    return float(np.max(angles)), float(np.std(angles))


def assess(
    ride_ax: np.ndarray, ride_ay: np.ndarray, ride_az: np.ndarray,
    pre_ax: np.ndarray, pre_ay: np.ndarray, pre_az: np.ndarray,
    post_ax: np.ndarray, post_ay: np.ndarray, post_az: np.ndarray,
    fs: float,
    *,
    min_segment_samples: int,
    grav_window_sec: float,
    grav_stability_max: float,
    grav_pre_post_angle_deg: float,
    max_ride_drift_deg: float,
    max_peak_m_s2: float,
    min_active_fraction: float,
    quality_score_reject: float,
    active_threshold_m_s2: float,
) -> ZuptQuality:
    features: dict[str, float] = {}

    n = len(ride_ax)
    if n < min_segment_samples:
        return ZuptQuality(
            accept=False, reject_reason="segment_too_short",
            quality_score=10.0, features={"n_samples": float(n)},
        )

    # --- Pre/post gravity quality ---
    pre_g, pre_mag, pre_stab = estimate_gravity_stationary(
        pre_ax, pre_ay, pre_az, fs=fs, window_sec=grav_window_sec)
    post_g, post_mag, post_stab = estimate_gravity_stationary(
        post_ax, post_ay, post_az, fs=fs, window_sec=grav_window_sec)

    pre_ok = 8.0 < pre_mag < 12.0 and pre_stab < grav_stability_max
    post_ok = 8.0 < post_mag < 12.0 and post_stab < grav_stability_max
    features["pre_g_mag"] = pre_mag
    features["post_g_mag"] = post_mag
    features["pre_stability"] = pre_stab
    features["post_stability"] = post_stab

    if pre_ok and post_ok:
        pp_angle = _angle_deg(pre_g, post_g)
    else:
        pp_angle = float("inf")
    features["pre_post_angle_deg"] = pp_angle if np.isfinite(pp_angle) else -1.0

    # --- During-ride gravity drift ---
    max_drift, drift_std = _ride_gravity_drift(ride_ax, ride_ay, ride_az, fs)
    features["max_gravity_drift_deg"] = max_drift
    features["gravity_drift_std_deg"] = drift_std

    # --- Acceleration magnitude stats ---
    mag = np.sqrt(ride_ax ** 2 + ride_ay ** 2 + ride_az ** 2)
    mag_mean = float(np.mean(mag))
    mag_std = float(np.std(mag))
    max_peak = float(np.max(np.abs(mag - mag_mean)))
    features["mag_mean"] = mag_mean
    features["mag_std"] = mag_std
    features["max_peak_m_s2"] = max_peak

    # --- Active-motion fraction (magnitude-detrended) ---
    a_vert_quick = mag - mag_mean
    active_frac = float(np.mean(np.abs(a_vert_quick) > active_threshold_m_s2))
    features["active_fraction"] = active_frac

    # --- ZUPT-end-velocity ratio (quick, non-gravity-projected version) ---
    dt = 1.0 / max(fs, 1.0)
    vel_quick = np.cumsum(a_vert_quick) * dt
    end_ratio = float(abs(vel_quick[-1]) / (np.max(np.abs(vel_quick)) + 1e-6))
    features["end_vel_ratio"] = end_ratio

    # ============================================================
    # Score (lower = better)
    # ============================================================
    score = 0.0
    # Pre-ride gravity quality (the more trust we can put on projection,
    # the better; if no gravity calibration is available at all, penalise)
    if not pre_ok and not post_ok:
        score += 3.5
    elif not pre_ok:
        score += 1.2
    else:
        score += min(pre_stab * 2.0, 2.0)

    # Pre/post gravity vector angle — large means phone moved
    if np.isfinite(pp_angle):
        score += min(max(pp_angle - 10.0, 0.0) / 8.0, 3.0)

    # During-ride gravity drift
    score += min(max_drift / 15.0, 2.0)

    # Impact peaks
    if max_peak > 6.0:
        score += 2.0
    elif max_peak > 4.0:
        score += 1.0

    # ZUPT end-velocity ratio
    score += min(end_ratio * 2.0, 2.0)

    # ============================================================
    # Rejection rules
    # ============================================================
    if not pre_ok and not post_ok:
        return ZuptQuality(False, "no_gravity_calibration", score, features)

    if np.isfinite(pp_angle) and pp_angle > grav_pre_post_angle_deg:
        return ZuptQuality(
            False, f"orientation_changed_{pp_angle:.0f}deg", score, features,
        )

    if max_peak > max_peak_m_s2:
        return ZuptQuality(False, f"impact_{max_peak:.1f}m_s2", score, features)

    if max_drift > max_ride_drift_deg:
        return ZuptQuality(False, f"ride_drift_{max_drift:.0f}deg", score, features)

    if active_frac < min_active_fraction and n > 400:
        return ZuptQuality(False, "no_significant_motion", score, features)

    if score > quality_score_reject:
        return ZuptQuality(False, f"quality_score_{score:.1f}", score, features)

    return ZuptQuality(True, "", score, features)
