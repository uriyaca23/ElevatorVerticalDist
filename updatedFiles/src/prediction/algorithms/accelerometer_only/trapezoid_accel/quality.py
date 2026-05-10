"""Quality filter for the trapezoid-pulse-pair estimator.

Same shape as the ZUPT filter (scalar score + binary accept/reject),
but the features are adapted to the per-ride shared-shape fit:

* Gravity-drift features live on the pre/during/post windows (same as
  ZUPT — any fit is only as good as the vertical-axis estimate).
* Fit-quality features come from the matched-filter R² and the residual
  autocorrelation.
* Physical-plausibility features check that Δt_c, A, and W come out in
  realistic ranges for an elevator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.utils.accelerometer_utils import estimate_gravity_stationary


@dataclass
class TrapezoidQuality:
    accept: bool
    reject_reason: str
    quality_score: float
    features: dict


def _angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-9 or n2 < 1e-9:
        return float("inf")
    cos = float(np.dot(v1, v2) / (n1 * n2))
    return float(np.degrees(np.arccos(max(-1.0, min(1.0, cos)))))


def _ride_gravity_drift(
    ax: np.ndarray, ay: np.ndarray, az: np.ndarray,
    fs: float, chunk_sec: float = 1.0,
) -> tuple[float, float]:
    n = len(ax)
    chunk = max(10, int(fs * chunk_sec))
    n_chunks = max(1, n // chunk)
    if n_chunks < 2:
        return 0.0, 0.0
    gvecs = []
    for i in range(n_chunks):
        s = i * chunk; e = min(s + chunk, n)
        gvecs.append(np.array([np.mean(ax[s:e]), np.mean(ay[s:e]), np.mean(az[s:e])]))
    g0 = gvecs[0]
    angles = [_angle_deg(g0, g) for g in gvecs[1:]]
    return float(np.max(angles)), float(np.std(angles))


def _residual_acf1(residuals: np.ndarray) -> float:
    if residuals.size < 30:
        return 0.0
    r = residuals - np.mean(residuals)
    denom = float(np.sum(r * r))
    if denom < 1e-12:
        return 0.0
    return float(np.sum(r[:-1] * r[1:]) / denom)


def assess(
    ride_ax: np.ndarray, ride_ay: np.ndarray, ride_az: np.ndarray,
    pre_ax: np.ndarray, pre_ay: np.ndarray, pre_az: np.ndarray,
    post_ax: np.ndarray, post_ay: np.ndarray, post_az: np.ndarray,
    fs: float,
    *,
    fit_joint_r2: float,
    fit_residuals: np.ndarray,
    predicted_abs_dh: float,
    delta_tc_sec: float,
    W_fit: float,
    A_fit: float,
    duration_s: float,
    A_anchor_ratio: float,
    out_of_lobe_residual_frac: float,
    cruise_v_cv: float,
    vert_method: str,
    min_segment_samples: int,
    grav_window_sec: float,
    grav_stability_max: float,
    quality_score_reject: float,
    min_r2_short: float,
    min_r2_mid: float,
    min_r2_long: float,
    overlap_delta: float,
    min_delta_tc_sec: float,
    min_distance_m: float,
    max_distance_m: float,
    min_active_fraction: float,
    max_A_anchor_ratio: float,
    min_A_anchor_ratio: float,
    max_out_of_lobe_frac: float,
    max_cruise_v_cv: float,
    max_pre_post_angle_deg: float,
) -> TrapezoidQuality:
    features: dict[str, float] = {}

    n = len(ride_ax)
    if n < min_segment_samples:
        return TrapezoidQuality(
            accept=False, reject_reason="segment_too_short",
            quality_score=10.0, features={"n_samples": float(n)},
        )

    # --- Pre/post gravity quality ---
    pre_g, pre_mag, pre_stab = estimate_gravity_stationary(
        pre_ax, pre_ay, pre_az, fs=fs, window_sec=grav_window_sec,
    )
    post_g, post_mag, post_stab = estimate_gravity_stationary(
        post_ax, post_ay, post_az, fs=fs, window_sec=grav_window_sec,
    )
    pre_ok = 8.0 < pre_mag < 12.0 and pre_stab < grav_stability_max
    post_ok = 8.0 < post_mag < 12.0 and post_stab < grav_stability_max
    features["pre_g_mag"] = pre_mag
    features["post_g_mag"] = post_mag
    features["pre_stability"] = pre_stab
    features["post_stability"] = post_stab
    features["vert_method"] = vert_method  # kept on the row for diagnostics
    if pre_ok and post_ok:
        pp_angle = _angle_deg(pre_g, post_g)
    else:
        pp_angle = float("inf")
    features["pre_post_angle_deg"] = pp_angle if np.isfinite(pp_angle) else -1.0
    # The estimator may have fallen back to in-ride gravity when neither
    # pre nor post window is stationary. That fallback is good enough to
    # certify a Δh on a single-ride pulse (median-of-window-means is
    # robust on bipolar content), so the quality gate must trust the
    # method actually used rather than the raw pre/post flags.
    has_calibration = vert_method != "magnitude"

    # --- During-ride gravity drift ---
    max_drift, drift_std = _ride_gravity_drift(ride_ax, ride_ay, ride_az, fs)
    features["max_gravity_drift_deg"] = max_drift
    features["gravity_drift_std_deg"] = drift_std

    # --- Fit quality features ---
    features["joint_r2"] = float(fit_joint_r2)
    features["delta_tc_sec"] = float(delta_tc_sec)
    features["A_fit"] = float(A_fit)
    features["W_fit"] = float(W_fit)
    features["active_fraction"] = float((2.0 * W_fit + delta_tc_sec) / max(duration_s, 1e-6))
    acf1 = _residual_acf1(fit_residuals)
    features["residual_acf1"] = acf1

    # --- New outlier-catcher features ---
    features["A_anchor_ratio"] = float(A_anchor_ratio)
    features["out_of_lobe_residual_frac"] = float(out_of_lobe_residual_frac)
    features["cruise_v_cv"] = float(cruise_v_cv)

    # --- Graded R² minimum that depends on predicted |Δh| ---
    if predicted_abs_dh < 3.0:
        min_r2_effective = float(min_r2_short)
    elif predicted_abs_dh < 6.0:
        min_r2_effective = float(min_r2_mid)
    else:
        min_r2_effective = float(min_r2_long)
    features["min_r2_effective"] = min_r2_effective

    # --- W-relative overlap threshold ---
    overlap_threshold = 2.0 * W_fit * (1.0 + float(overlap_delta))
    features["overlap_threshold_sec"] = overlap_threshold

    # ============================================================
    # Score (lower = better)
    # ============================================================
    score = 0.0
    # Gravity-projection penalties. Tiered by what the estimator actually
    # used: a clean pre+post anchor is the gold standard, a one-sided
    # anchor is mid-tier, and an in-ride fallback gets a moderate penalty
    # (it is usable but noisier than a stationary anchor). Magnitude-only
    # is rejected outright below — the score branch never fires for it.
    if not pre_ok and not post_ok:
        # In-ride fallback (or magnitude). Apply a flat penalty smaller
        # than the old 3.5 because the projection is still signed and
        # axis-aware; the noise is in the gravity *direction* estimate.
        score += 2.0 if has_calibration else 3.5
    elif not pre_ok:
        score += 1.2
    else:
        score += min(pre_stab * 2.0, 2.0)
    if np.isfinite(pp_angle):
        score += min(max(pp_angle - 10.0, 0.0) / 8.0, 3.0)
    score += min(max_drift / 15.0, 2.0)

    # Fit quality
    if fit_joint_r2 < 0.6:
        score += 3.0 * (0.6 - fit_joint_r2) / 0.6
    elif fit_joint_r2 < 0.8:
        score += 1.0 * (0.8 - fit_joint_r2) / 0.2
    if abs(acf1) > 0.6: score += 2.0
    elif abs(acf1) > 0.4: score += 1.0
    elif abs(acf1) > 0.25: score += 0.3

    # Physical plausibility
    if features["active_fraction"] > 1.4:
        score += 2.0
    if A_fit < 0.1:
        score += 2.0

    # ============================================================
    # Rejection rules
    # ============================================================
    # Reject only when the estimator actually had no orientation
    # information at all (magnitude-only fallback). When pre/post fail
    # but the in-ride estimate succeeded ("projected_ride"), we trust
    # the result with the score-side penalty above.
    if not has_calibration:
        return TrapezoidQuality(False, "no_gravity_calibration", score, features)
    if (np.isfinite(pp_angle) and pp_angle > max_pre_post_angle_deg):
        return TrapezoidQuality(False, f"pre_post_angle_{pp_angle:.0f}deg", score, features)
    # Graded R² threshold (short rides are allowed lower R² because SNR
    # is intrinsically worse at short displacements).
    if fit_joint_r2 < min_r2_effective:
        return TrapezoidQuality(
            False, f"low_r2_{fit_joint_r2:.2f}_min_{min_r2_effective:.2f}",
            score, features,
        )
    # Absolute lobe-spacing floor only. The physically valid touching
    # regime (Δt_c = 2W) is handled by the joined-pulse branch in the
    # estimator, not by rejection here. We only reject when Δt_c
    # falls below ``min_delta_tc_sec`` — a tiny numerical floor
    # protecting against degenerate fits.
    if delta_tc_sec < min_delta_tc_sec:
        return TrapezoidQuality(
            False, f"delta_tc_below_floor_{delta_tc_sec:.2f}s",
            score, features,
        )
    if predicted_abs_dh < min_distance_m:
        return TrapezoidQuality(False, f"dh_too_small_{predicted_abs_dh:.2f}m", score, features)
    if predicted_abs_dh > max_distance_m:
        return TrapezoidQuality(False, f"dh_too_large_{predicted_abs_dh:.1f}m", score, features)
    if features["active_fraction"] < min_active_fraction and n > 400:
        return TrapezoidQuality(False, "no_significant_motion", score, features)
    # --- New outlier-catcher rules (Pixel-10 Yitzchaki style misfires) ---
    if A_anchor_ratio > max_A_anchor_ratio or A_anchor_ratio < min_A_anchor_ratio:
        return TrapezoidQuality(
            False, f"A_anchor_mismatch_{A_anchor_ratio:.2f}",
            score, features,
        )
    if out_of_lobe_residual_frac > max_out_of_lobe_frac:
        return TrapezoidQuality(
            False, f"out_of_lobe_frac_{out_of_lobe_residual_frac:.2f}",
            score, features,
        )
    if cruise_v_cv > max_cruise_v_cv:
        return TrapezoidQuality(
            False, f"cruise_v_cv_{cruise_v_cv:.2f}",
            score, features,
        )
    if score > quality_score_reject:
        return TrapezoidQuality(False, f"quality_score_{score:.1f}", score, features)

    return TrapezoidQuality(True, "", score, features)
