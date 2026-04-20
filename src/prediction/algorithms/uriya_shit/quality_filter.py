"""
Accelerometer-based quality filter for elevator ride segments.

Uses gravity vector stability (pitch/roll from accelerometer alone)
to decide whether a segment's height estimate will be reliable.
"""
import numpy as np


def estimate_gravity_vector(ax, ay, az, fs=100, window_sec=0.5):
    """
    Robust gravity vector estimation using median-of-windows.
    
    During stationary periods, [ax, ay, az] ≈ gravity vector.
    We split into short windows, compute mean in each, then take median.
    More robust to brief disturbances than a single mean.
    """
    n = len(ax)
    if n < 10:
        return np.array([0, 0, 9.81]), 9.81, float('inf')
    
    win = max(10, int(fs * window_sec))
    n_windows = max(1, n // win)

    gx_list, gy_list, gz_list, std_list = [], [], [], []
    for i in range(n_windows):
        s = i * win
        e = min(s + win, n)
        gx_list.append(np.mean(ax[s:e]))
        gy_list.append(np.mean(ay[s:e]))
        gz_list.append(np.mean(az[s:e]))
        std_list.append(np.std(ax[s:e]) + np.std(ay[s:e]) + np.std(az[s:e]))

    # Per-axis median over all windows breaks when orientation changes
    # mid-recording: medians drawn from different orientations compose
    # into a non-gravity vector with |g| far from 9.81. Restrict to the
    # most-stationary 20% of windows — they share the dominant resting
    # orientation, so per-axis median on the subset recovers |g|.
    std_arr = np.asarray(std_list)
    keep = std_arr <= np.quantile(std_arr, 0.2)
    gx = np.median(np.asarray(gx_list)[keep])
    gy = np.median(np.asarray(gy_list)[keep])
    gz = np.median(np.asarray(gz_list)[keep])
    gvec = np.array([gx, gy, gz])
    g_mag = np.linalg.norm(gvec)
    
    # Stability: std of window-level gravity estimates
    stability = np.sqrt(np.var(gx_list) + np.var(gy_list) + np.var(gz_list))
    
    return gvec, g_mag, stability


def angle_between_vectors(v1, v2):
    """Angle in degrees between two vectors."""
    cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12)
    return np.degrees(np.arccos(np.clip(cos_angle, -1, 1)))


def compute_ride_gravity_drift(ax, ay, az, fs=100, chunk_sec=1.0):
    """
    Compute how much the gravity vector drifts during the ride.
    
    Split ride into chunks, estimate gravity in each, measure drift.
    Returns: max angle from first chunk, std of angles.
    """
    n = len(ax)
    chunk = max(10, int(fs * chunk_sec))
    n_chunks = max(1, n // chunk)
    
    g_vectors = []
    for i in range(n_chunks):
        s = i * chunk
        e = min(s + chunk, n)
        gv = np.array([np.mean(ax[s:e]), np.mean(ay[s:e]), np.mean(az[s:e])])
        g_vectors.append(gv)
    
    if len(g_vectors) < 2:
        return 0.0, 0.0
    
    g0 = g_vectors[0]
    angles = [angle_between_vectors(g0, gv) for gv in g_vectors[1:]]
    
    return np.max(angles), np.std(angles)


def assess_segment_quality(ride_ax, ride_ay, ride_az,
                            pre_ax, pre_ay, pre_az,
                            post_ax, post_ay, post_az,
                            fs=100):
    """
    Assess whether an elevator segment is suitable for height estimation.
    
    Uses accelerometer-only features to detect orientation instability,
    impacts, and other conditions that degrade ZUPT accuracy.
    
    Returns:
        dict with:
            'accept': bool
            'reject_reason': str (empty if accepted)
            'quality_score': float (0=best, higher=worse)
            'features': dict of computed features
    """
    result = {
        'accept': True,
        'reject_reason': '',
        'quality_score': 0.0,
        'features': {}
    }
    
    n_ride = len(ride_ax)
    if n_ride < 20:
        result['accept'] = False
        result['reject_reason'] = 'Segment too short'
        result['quality_score'] = 10.0
        return result
    
    # ---- Feature 1: Pre-ride gravity vector quality ----
    pre_gvec, pre_g_mag, pre_stability = estimate_gravity_vector(pre_ax, pre_ay, pre_az, fs)
    result['features']['pre_g_mag'] = float(pre_g_mag)
    result['features']['pre_stability'] = float(pre_stability)
    
    pre_gravity_ok = (8.0 < pre_g_mag < 12.0) and (pre_stability < 1.0)
    
    # ---- Feature 2: Post-ride gravity vector quality ----
    post_gvec, post_g_mag, post_stability = estimate_gravity_vector(post_ax, post_ay, post_az, fs)
    result['features']['post_g_mag'] = float(post_g_mag)
    result['features']['post_stability'] = float(post_stability)
    
    post_gravity_ok = (8.0 < post_g_mag < 12.0) and (post_stability < 1.0)
    
    # If pre-ride is unstable but post-ride is stable, we can use post-ride
    # as calibration fallback (phone may have been placed in pocket just before ride)
    can_use_post = (8.0 < post_g_mag < 12.0) and (post_stability < 0.5)
    has_any_gravity_cal = pre_gravity_ok or can_use_post
    
    # ---- Feature 3: Pre/post gravity vector angle ----
    if pre_gravity_ok and post_gravity_ok:
        pre_post_angle = angle_between_vectors(pre_gvec, post_gvec)
        result['features']['pre_post_angle'] = float(pre_post_angle)
    else:
        pre_post_angle = float('inf')
        result['features']['pre_post_angle'] = -1.0  # unavailable
    
    # ---- Feature 4: During-ride gravity drift ----
    max_drift, drift_std = compute_ride_gravity_drift(ride_ax, ride_ay, ride_az, fs, chunk_sec=1.0)
    result['features']['max_gravity_drift'] = float(max_drift)
    result['features']['gravity_drift_std'] = float(drift_std)
    
    # ---- Feature 5: Acceleration magnitude consistency ----
    ride_mag = np.sqrt(ride_ax**2 + ride_ay**2 + ride_az**2)
    mag_mean = np.mean(ride_mag)
    mag_std = np.std(ride_mag)
    max_peak = np.max(np.abs(ride_mag - mag_mean))
    result['features']['mag_mean'] = float(mag_mean)
    result['features']['mag_std'] = float(mag_std)
    result['features']['max_peak'] = float(max_peak)
    
    # ---- Feature 6: End velocity check (proxy for ZUPT quality) ----
    # Quick integrate magnitude to check if velocity returns to zero
    dt = 1.0 / fs
    a_vert_quick = ride_mag - mag_mean
    vel_quick = np.cumsum(a_vert_quick) * dt
    end_vel_ratio = abs(vel_quick[-1]) / (np.max(np.abs(vel_quick)) + 1e-6)
    result['features']['end_vel_ratio'] = float(end_vel_ratio)
    
    # ---- Feature 7: Effective motion duration ----
    # How much of the ride has meaningful vertical acceleration
    threshold = 0.15  # m/s^2
    active_frac = np.mean(np.abs(a_vert_quick) > threshold)
    result['features']['active_fraction'] = float(active_frac)
    
    # ---- Compute quality score (weighted sum) ----
    score = 0.0
    
    # Pre-ride gravity quality
    if not pre_gravity_ok:
        score += 3.0
    else:
        score += pre_stability * 2.0
    
    # Pre/post angle
    if pre_post_angle != float('inf') and pre_post_angle > 15:
        score += min(pre_post_angle / 10, 3.0)
    
    # During-ride drift
    score += min(max_drift / 15, 2.0)
    
    # Impact peaks
    if max_peak > 5.0:
        score += 2.0
    elif max_peak > 3.0:
        score += 1.0
    
    # End velocity (should be near zero for good ZUPT)
    score += end_vel_ratio * 2.0
    
    result['quality_score'] = float(score)
    
    # ---- Apply rejection rules ----
    
    # Rule 1: No valid gravity calibration at all (neither pre nor post)
    if not has_any_gravity_cal and len(pre_ax) > 50:
        result['accept'] = False
        result['reject_reason'] = f'No stable gravity calibration available'
        return result
    
    # Rule 2: Large orientation change pre/post (> 25 degrees)
    if pre_post_angle != float('inf') and pre_post_angle > 25:
        result['accept'] = False
        result['reject_reason'] = f'Orientation changed {pre_post_angle:.1f} deg during ride'
        return result
    
    # Rule 3: Extreme impacts during ride
    if max_peak > 8.0:
        result['accept'] = False
        result['reject_reason'] = f'Impact detected ({max_peak:.1f} m/s2)'
        return result
    
    # Rule 4: High gravity drift during ride (phone being actively moved)
    # Rides with drift > 15deg tend to have much larger height estimation errors
    if max_drift > 15:
        result['accept'] = False
        result['reject_reason'] = f'High gravity drift during ride ({max_drift:.1f} deg)'
        return result
    
    # Rule 5: No meaningful motion detected
    if active_frac < 0.05 and n_ride > 500:
        result['accept'] = False
        result['reject_reason'] = f'No significant vertical motion detected'
        return result
    
    # Rule 6: High acceleration noise + very high peak (noisy pocket rides)
    if mag_std > 0.9 and max_peak > 10.0:
        result['accept'] = False
        result['reject_reason'] = f'High noise during ride (std={mag_std:.2f}, peak={max_peak:.1f})'
        return result
    
    # Rule 7: Pre-ride unstable AND no post-ride fallback AND moderate drift
    if not pre_gravity_ok and not can_use_post and max_drift > 10:
        result['accept'] = False
        result['reject_reason'] = f'No calibration + drift ({max_drift:.1f} deg)'
        return result
    
    return result
