"""
Robust Elevator Height Estimation Pipeline v4.

Three-stage pipeline with clear objectives:
  1. Detect & segment elevator rides (reliable, per-ride separation)
  2. Quality-based accept/reject (accelerometer-only orientation checks)
  3. Height estimation with conformal prediction (90% coverage, ≤1m intervals)

Uses 3-axis accelerometer data (no gyroscope required for core pipeline).
"""
import os
import sys
import json
import random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy.signal import butter, filtfilt

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from algorithms.quality_filter import (
    estimate_gravity_vector,
    angle_between_vectors,
    assess_segment_quality,
    compute_ride_gravity_drift,
)

FIGS = os.path.join("docs", "figures_v4")
os.makedirs(FIGS, exist_ok=True)


# =====================================================================
# STAGE 1: Elevator Detection & Segmentation
# =====================================================================

def detect_elevator_rides(t, ax, ay, az, fs=100,
                          var_window_sec=1.5,
                          var_thresh=1.5,
                          min_ride_sec=4.0,
                          min_displacement_m=1.0,
                          gap_sec=2.0):
    """
    Detect individual elevator rides from 3-axis accelerometer.
    
    Strategy:
    1. Compute acceleration magnitude and its rolling variance
    2. Find low-variance (standing still) contiguous blocks
    3. Within each block, integrate to find velocity/displacement
    4. Validate: must have clear accel+decel pattern, net displacement > min
    5. Split multi-ride blocks at velocity zero-crossings
    
    Returns list of dicts: {s_idx, e_idx, estimated_direction}
    """
    n = len(t)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)
    
    # Rolling variance of acceleration magnitude
    var_win = int(fs * var_window_sec)
    mag_series = pd.Series(acc_mag)
    rolling_var = mag_series.rolling(window=var_win, center=True, min_periods=1).var().values
    
    # Low-variance mask = person is standing still (not walking)
    is_still = rolling_var < var_thresh
    
    # Find contiguous still-blocks
    blocks = []
    in_block = False
    block_start = 0
    for i in range(n):
        if is_still[i] and not in_block:
            block_start = i
            in_block = True
        elif not is_still[i] and in_block:
            if (i - block_start) >= int(fs * min_ride_sec):
                blocks.append((block_start, i))
            in_block = False
    if in_block and (n - block_start) >= int(fs * min_ride_sec):
        blocks.append((block_start, n))
    
    # For each block, detect sub-rides
    all_rides = []
    dt = 1.0 / fs
    
    for bs, be in blocks:
        block_mag = acc_mag[bs:be]
        block_len = be - bs
        
        # Remove gravity (mean of block)
        g_mean = np.mean(block_mag)
        a_lin = block_mag - g_mean
        
        # Low-pass filter to remove noise
        nyq = fs / 2
        cutoff = min(2.0, nyq * 0.8)
        b_filt, a_filt_coeff = butter(2, cutoff / nyq, btype='low')
        if block_len > 3 * max(len(b_filt), len(a_filt_coeff)):
            a_smooth = filtfilt(b_filt, a_filt_coeff, a_lin)
        else:
            a_smooth = a_lin
        
        # Integrate to velocity
        vel = np.cumsum(a_smooth) * dt
        
        # Integrate to position (displacement)
        pos = np.cumsum(vel) * dt
        
        # Find sub-rides by looking for segments with significant displacement
        # A ride = accel phase + coast + decel phase
        # Between rides, velocity should cross zero
        
        # Find zero-crossings of velocity (potential ride boundaries)
        vel_sign = np.sign(vel)
        zero_crossings = np.where(np.diff(vel_sign) != 0)[0]
        
        # Add block boundaries
        boundaries = [0] + list(zero_crossings) + [block_len - 1]
        
        # Group consecutive same-direction sub-segments into rides
        # A ride should have: acceleration, then deceleration (opposite accel)
        # Look for segments where position changes monotonically
        
        # Simpler approach: split at velocity zero-crossings where position
        # has changed by at least min_displacement
        ride_candidates = []
        current_start = 0
        last_split_pos = pos[0]
        
        for zc in zero_crossings:
            segment_displacement = abs(pos[zc] - last_split_pos)
            # Check if we've accumulated enough displacement
            if segment_displacement >= min_displacement_m:
                # Check if there's a gap (near-zero velocity for a while)
                gap_samples = int(fs * gap_sec)
                # Look around the zero crossing for still period
                look_start = max(0, zc - gap_samples)
                look_end = min(block_len, zc + gap_samples)
                
                # Check velocity magnitude around zero-crossing
                vel_around = np.abs(vel[look_start:look_end])
                min_vel_around = np.min(vel_around)
                
                if min_vel_around < 0.1:  # velocity near zero = boundary
                    # Find the exact point of minimum |velocity|
                    split_point = look_start + np.argmin(vel_around)
                    
                    # Only split if current segment has meaningful acceleration pattern
                    seg_a = a_smooth[current_start:split_point]
                    if len(seg_a) > int(fs * 1.0):  # at least 1 second
                        max_pos_a = np.max(seg_a)
                        min_neg_a = np.min(seg_a)
                        if max_pos_a > 0.1 and min_neg_a < -0.1:
                            ride_candidates.append((current_start, split_point))
                            current_start = split_point
                            last_split_pos = pos[split_point]
        
        # Don't forget the last segment
        if current_start < block_len - int(fs * 1.0):
            seg_a = a_smooth[current_start:]
            max_pos_a = np.max(seg_a)
            min_neg_a = np.min(seg_a)
            segment_displacement = abs(pos[-1] - pos[current_start])
            if (max_pos_a > 0.1 and min_neg_a < -0.1 and 
                segment_displacement >= min_displacement_m):
                ride_candidates.append((current_start, block_len))
        
        # If no sub-rides found, treat whole block as one ride
        if not ride_candidates:
            max_pos_a = np.max(a_smooth)
            min_neg_a = np.min(a_smooth)
            net_disp = abs(pos[-1] - pos[0])
            if max_pos_a > 0.1 and min_neg_a < -0.1 and net_disp >= min_displacement_m:
                ride_candidates = [(0, block_len)]
        
        # Convert to global indices and validate
        for rs, re in ride_candidates:
            global_s = bs + rs
            global_e = bs + re
            
            # Trim leading/trailing near-zero acceleration
            seg_mag = acc_mag[global_s:global_e]
            seg_a_lin = seg_mag - np.mean(seg_mag)
            
            # Find first and last significant acceleration
            threshold = 0.08
            sig_mask = np.abs(seg_a_lin) > threshold
            if not np.any(sig_mask):
                continue
            first_sig = np.argmax(sig_mask)
            last_sig = len(sig_mask) - 1 - np.argmax(sig_mask[::-1])
            
            # Add margin
            margin = int(fs * 0.5)
            trim_s = max(0, first_sig - margin)
            trim_e = min(len(seg_a_lin), last_sig + margin)
            
            final_s = global_s + trim_s
            final_e = global_s + trim_e
            
            if final_e - final_s < int(fs * min_ride_sec):
                continue
            
            # Determine direction from position curve
            seg_pos = pos[rs + trim_s:rs + trim_e] if rs + trim_e <= len(pos) else pos[rs + trim_s:]
            if len(seg_pos) > 0:
                direction = 1 if seg_pos[-1] > seg_pos[0] else -1
            else:
                direction = 0
            
            all_rides.append({
                's_idx': final_s,
                'e_idx': final_e,
                'direction': direction,
            })
    
    # Merge overlapping rides
    all_rides.sort(key=lambda r: r['s_idx'])
    merged = []
    for ride in all_rides:
        if merged and ride['s_idx'] < merged[-1]['e_idx']:
            # Overlap - merge
            merged[-1]['e_idx'] = max(merged[-1]['e_idx'], ride['e_idx'])
        else:
            merged.append(ride)
    
    return merged


# =====================================================================
# STAGE 2: Height Estimation (Robust Gravity-Projected ZUPT)
# =====================================================================

def estimate_height_robust(t, ax, ay, az,
                            pre_ax, pre_ay, pre_az,
                            post_ax, post_ay, post_az,
                            fs=100):
    """
    Robust height estimator with vertical prior.
    
    Strategy:
    1. Estimate gravity direction from pre-ride stationary data  
    2. Project acceleration onto gravity direction to get vertical component
    3. Apply ZUPT integration with drift correction
    4. Use magnitude-based estimate as cross-check  
    5. Pick best estimate based on agreement and quality
    
    Returns dict with height, pos curve, method used, quality info.
    """
    n = len(t)
    dt_arr = np.diff(t, prepend=t[0])
    dt_arr[0] = 1.0 / fs  # fix first element
    dt = 1.0 / fs
    
    result = {
        'height': 0.0,
        'pos': np.zeros(n),
        'method': 'none',
        'quality': float('inf'),
        'all_estimates': {},
    }
    
    if n < 20:
        return result
    
    # ---- Method 1: Magnitude-based ZUPT (rotation-invariant) ----
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_ride = np.mean(mag)
    a_mag = mag - g_ride
    pos_mag, vel_mag = _zupt_integrate(a_mag, dt, fs)
    h_mag = pos_mag[-1]
    result['all_estimates']['magnitude'] = float(h_mag)
    
    # ---- Method 2: Gravity-projected ZUPT ----
    # Estimate gravity from pre-ride and post-ride
    pre_gvec, pre_g_mag, pre_stab = estimate_gravity_vector(pre_ax, pre_ay, pre_az, fs)
    post_gvec, post_g_mag, post_stab = estimate_gravity_vector(post_ax, post_ay, post_az, fs)
    
    h_gp = None
    pos_gp = None
    gp_quality = float('inf')
    
    # Determine best gravity calibration
    pre_ok = 8.0 < pre_g_mag < 12.0 and pre_stab < 1.0
    post_ok = 8.0 < post_g_mag < 12.0 and post_stab < 1.0
    
    # Primary: use pre-ride gravity
    # Fallback: if pre-ride is unstable, use post-ride gravity
    cal_gvec = None
    cal_g_mag = None
    cal_quality = float('inf')
    
    if pre_ok and post_ok:
        angle = angle_between_vectors(pre_gvec, post_gvec)
        if angle < 20:
            # Average pre and post gravity vectors (weighted by stability)
            w1 = 1.0 / (pre_stab + 0.001)
            w2 = 1.0 / (post_stab + 0.001)
            g_avg = (pre_gvec * w1 + post_gvec * w2) / (w1 + w2)
            cal_gvec = g_avg
            cal_g_mag = np.linalg.norm(g_avg)
            cal_quality = min(pre_stab, post_stab)
        else:
            # Orientation changed - use whichever is more stable
            if pre_stab < post_stab:
                cal_gvec = pre_gvec
                cal_g_mag = pre_g_mag
                cal_quality = pre_stab
            else:
                cal_gvec = post_gvec
                cal_g_mag = post_g_mag
                cal_quality = post_stab
    elif pre_ok:
        cal_gvec = pre_gvec
        cal_g_mag = pre_g_mag
        cal_quality = pre_stab
    elif post_ok:
        # Post-ride fallback
        cal_gvec = post_gvec
        cal_g_mag = post_g_mag
        cal_quality = post_stab
    elif 8.0 < pre_g_mag < 12.0 and pre_stab < 1.5:
        # Marginal pre-ride quality
        cal_gvec = pre_gvec
        cal_g_mag = pre_g_mag
        cal_quality = pre_stab
    elif 8.0 < post_g_mag < 12.0 and post_stab < 1.5:
        # Marginal post-ride quality
        cal_gvec = post_gvec
        cal_g_mag = post_g_mag
        cal_quality = post_stab
    
    if cal_gvec is not None:
        g_hat = cal_gvec / cal_g_mag
        gp_quality = cal_quality
        
        # Project acceleration onto gravity direction
        a_vert = ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - cal_g_mag
        
        pos_gp, vel_gp = _zupt_integrate(a_vert, dt, fs, accel_threshold=0.05)
        h_gp = pos_gp[-1]
        result['all_estimates']['gravity_proj'] = float(h_gp)

    
    # ---- Method 3: Sign-corrected magnitude ----
    # For pocket rides: magnitude gives absolute displacement, we need sign
    # Determine sign from initial acceleration pulse direction
    if h_gp is not None:
        h_signed_mag = np.sign(h_gp) * abs(h_mag) if abs(h_gp) > 0.5 else h_mag
    else:
        # Use the first significant acceleration pulse to determine sign
        # In an elevator: going UP -> initial positive acceleration (pushed into floor)
        # -> magnitude increases initially
        first_samples = min(n, int(fs * 3))
        early_a = a_mag[:first_samples]
        rising = np.sum(early_a[early_a > 0])
        falling = np.sum(early_a[early_a < 0])
        if abs(rising) > abs(falling) * 1.2:
            h_signed_mag = abs(h_mag)  # going up
        elif abs(falling) > abs(rising) * 1.2:
            h_signed_mag = -abs(h_mag)  # going down
        else:
            h_signed_mag = h_mag  # unclear, keep as-is
    result['all_estimates']['signed_magnitude'] = float(h_signed_mag)
    
    # ---- Select best estimate ----
    # Compute during-ride gravity drift to inform selection
    ride_drift, _ = compute_ride_gravity_drift(ax, ay, az, fs, chunk_sec=1.0)
    
    # When gravity drift is moderate and GP significantly exceeds magnitude,
    # the projection is picking up horizontal acceleration. Prefer magnitude.
    drift_override = False
    if h_gp is not None and ride_drift > 8 and abs(h_mag) > 1.0:
        gp_mag_ratio = abs(h_gp) / abs(h_mag)
        if gp_mag_ratio > 1.5:
            drift_override = True
            # Use signed magnitude instead
            result['height'] = float(h_signed_mag)
            result['pos'] = pos_mag
            result['method'] = 'drift_corrected_mag'
            result['quality'] = float(gp_quality)
            result['agreement'] = gp_mag_ratio
            return result
    
    if h_gp is not None and gp_quality < 0.5 and abs(h_gp) < 150:
        # High quality gravity projection available
        # Check agreement with magnitude
        if abs(h_mag) > 0.5:
            agree = 1.0 - abs(abs(h_gp) - abs(h_mag)) / max(abs(h_gp), abs(h_mag))
        else:
            agree = 0.5
        
        gp_too_large = abs(h_gp) > 4 * max(abs(h_mag), 1.0) and abs(h_gp) > 30
        
        if not gp_too_large:
            result['height'] = float(h_gp)
            result['pos'] = pos_gp
            result['method'] = 'gravity_proj'
            result['quality'] = float(gp_quality)
            result['agreement'] = float(agree)
            return result
    
    if h_gp is not None and abs(h_gp) < 150:
        # GP available but lower quality - use if reasonable
        if abs(h_mag) > 0.5:
            agree = 1.0 - abs(abs(h_gp) - abs(h_mag)) / max(abs(h_gp), abs(h_mag))
        else:
            agree = 0.5
            
        if agree > 0.3 and not (abs(h_gp) > 3 * max(abs(h_mag), 1.0)):
            result['height'] = float(h_gp)
            result['pos'] = pos_gp
            result['method'] = 'gravity_proj'
            result['quality'] = float(gp_quality)
            result['agreement'] = float(agree)
            return result
    
    # Fallback to signed magnitude
    result['height'] = float(h_signed_mag)
    result['pos'] = pos_mag
    result['method'] = 'signed_mag'
    result['quality'] = 0.5
    result['agreement'] = 0.0
    return result


def _zupt_integrate(a_vert, dt, fs, accel_threshold=0.08):
    """
    ZUPT integration with improved drift correction.
    
    1. Smooth to find active window
    2. Subtract mean bias from acceleration  
    3. Integrate with linear drift removal
    """
    n = len(a_vert)
    
    # Remove DC bias (crucial for preventing drift)
    a_vert = a_vert - np.mean(a_vert)
    
    # Find active motion window
    win = min(50, max(5, n // 3))
    kernel = np.ones(win) / win
    smooth_abs = np.convolve(np.abs(a_vert), kernel, mode='same')
    active = np.where(smooth_abs > accel_threshold)[0]
    
    if len(active) == 0:
        return np.zeros(n), np.zeros(n)
    
    margin = int(fs * 0.3)
    s = max(0, active[0] - margin)
    e = min(n - 1, active[-1] + margin)
    
    # Zero outside window
    a_windowed = np.zeros(n)
    a_windowed[s:e+1] = a_vert[s:e+1]
    
    # Integrate to velocity
    vel = np.cumsum(a_windowed) * dt
    vel[:s] = 0
    
    # Linear drift correction (velocity must be zero at end of ride)
    if e > s:
        drift = vel[e]
        correction = np.zeros(n)
        correction[s:e+1] = np.linspace(0, drift, e - s + 1)
        vel -= correction
    
    vel[e+1:] = 0  # Force zero after end
    
    # Integrate to position
    pos = np.cumsum(vel) * dt
    
    return pos, vel


# =====================================================================
# STAGE 3: Conformal Prediction
# =====================================================================

def compute_conformal_interval(train_errors, alpha=0.10):
    """
    Simple split-conformal prediction.
    
    Uses absolute errors from training set to compute the
    (1-alpha) quantile as the prediction interval half-width.
    
    This gives a constant interval width (no scaling by ride length),
    which is appropriate when errors are not systematically correlated
    with ride duration for the accepted (filtered) rides.
    """
    errors = np.abs(np.array(train_errors))
    n = len(errors)
    if n == 0:
        return 1.0  # fallback
    
    # Standard conformal: use ceil((n+1)*(1-alpha))/n quantile
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    
    interval = np.quantile(errors, q_level)
    return float(interval)


# =====================================================================
# MATCHING: GT rides <-> Detected rides
# =====================================================================

def match_gt_to_detected(gt_rides, det_rides, t, iou_threshold=0.3):
    """
    Match GT rides to detected rides based on temporal IoU.
    Returns list of (gt_idx, det_idx, iou) tuples.
    """
    n = len(t)
    matches = []
    
    for gi, gt in enumerate(gt_rides):
        gt_mask = np.zeros(n, dtype=bool)
        gt_mask[gt['s_idx']:gt['e_idx']] = True
        
        best_iou = 0
        best_di = -1
        
        for di, det in enumerate(det_rides):
            det_mask = np.zeros(n, dtype=bool)
            det_mask[det['s_idx']:det['e_idx']] = True
            
            inter = np.sum(gt_mask & det_mask)
            union = np.sum(gt_mask | det_mask)
            iou = inter / union if union > 0 else 0
            
            if iou > best_iou:
                best_iou = iou
                best_di = di
        
        if best_iou >= iou_threshold:
            matches.append((gi, best_di, best_iou))
        else:
            matches.append((gi, -1, best_iou))
    
    return matches


# =====================================================================
# MAIN PIPELINE
# =====================================================================

def main():
    print("=" * 70)
    print("ROBUST PIPELINE v4: Detection + Quality Filter + Conformal")
    print("=" * 70)
    
    # ---- Load data ----
    acc_f = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")
    gt_f = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
    
    df_acc = pd.read_csv(acc_f, names=["time_ms", "x", "y", "z"])
    df_gt = pd.read_csv(gt_f)
    
    t = df_acc["time_ms"].values / 1000.0
    raw_ax = df_acc["x"].values
    raw_ay = df_acc["y"].values
    raw_az = df_acc["z"].values
    acc_mag = np.sqrt(raw_ax**2 + raw_ay**2 + raw_az**2)
    fs = 1.0 / np.median(np.diff(t))
    
    gt_t = df_gt["time_sec"].values
    gt_h = df_gt["height_smooth"].values
    
    print(f"Dataset: {len(t)} samples, fs={fs:.0f}Hz, duration={t[-1]:.0f}s")
    
    # ---- Extract GT rides ----
    gt_ids = sorted([x for x in df_gt["elevator_segment_id"].unique() if x >= 0])
    gt_rides = []
    for sid in gt_ids:
        sub = df_gt[df_gt["elevator_segment_id"] == sid]
        t_start = sub["time_sec"].iloc[0]
        t_end = sub["time_sec"].iloc[-1]
        gt_rides.append({
            "id": int(sid),
            "t_start": t_start, "t_end": t_end,
            "h_start": sub["height_smooth"].iloc[0],
            "h_end": sub["height_smooth"].iloc[-1],
            "true_dh": sub["height_smooth"].iloc[-1] - sub["height_smooth"].iloc[0],
            "s_idx": int(np.argmin(np.abs(t - t_start))),
            "e_idx": int(np.argmin(np.abs(t - t_end))),
            "phone": sub["phone_position"].iloc[0],
        })
    print(f"GT rides: {len(gt_rides)}")
    
    # ==================================================================
    # OBJECTIVE 1: Elevator Detection & Segmentation
    # ==================================================================
    print(f"\n{'='*50}")
    print("OBJECTIVE 1: Elevator Detection & Segmentation")
    print(f"{'='*50}")
    
    det_rides = detect_elevator_rides(t, raw_ax, raw_ay, raw_az, fs=fs)
    print(f"Detected {len(det_rides)} segments")
    
    # Match GT to detected
    matches = match_gt_to_detected(gt_rides, det_rides, t)
    
    matched_gt = sum(1 for _, di, _ in matches if di >= 0)
    print(f"Matched: {matched_gt}/{len(gt_rides)} GT rides")
    
    for gi, di, iou in matches:
        gt = gt_rides[gi]
        status = f"MATCH(det={di}, IoU={iou:.2f})" if di >= 0 else "MISS"
        print(f"  GT {gt['id']:2d}: t=[{gt['t_start']:7.1f},{gt['t_end']:7.1f}] "
              f"dh={gt['true_dh']:+6.1f}m {gt['phone']:>6} -> {status}")
    
    # ==================================================================
    # OBJECTIVE 2 & 3: Quality Filter + Height Estimation
    # ==================================================================
    # We evaluate on GT rides directly (since we know the true boundaries)
    # This tests objectives 2 and 3 independently of detection quality
    
    print(f"\n{'='*50}")
    print("OBJECTIVES 2 & 3: Quality Filter + Height Estimation")
    print(f"{'='*50}")
    
    PRE_WINDOW = 5.0
    POST_WINDOW = 5.0
    
    results = []
    
    header = (f"{'Ride':>4} {'True':>7} {'Est':>7} {'Err':>5} "
              f"{'Method':>10} {'QScore':>6} {'Accept':>6} {'Phone':>6}")
    print(header)
    print("-" * len(header))
    
    for ri, gt in enumerate(gt_rides):
        si, ei = gt["s_idx"], gt["e_idx"]
        ride_ax = raw_ax[si:ei]
        ride_ay = raw_ay[si:ei]
        ride_az = raw_az[si:ei]
        
        # Pre-ride data
        pre_start = max(0, int(np.argmin(np.abs(t - max(0, gt["t_start"] - PRE_WINDOW)))))
        pre_ax = raw_ax[pre_start:si]
        pre_ay = raw_ay[pre_start:si]
        pre_az = raw_az[pre_start:si]
        
        # Post-ride data
        post_end = min(len(t), int(np.argmin(np.abs(t - min(t[-1], gt["t_end"] + POST_WINDOW)))))
        post_ax = raw_ax[ei:post_end]
        post_ay = raw_ay[ei:post_end]
        post_az = raw_az[ei:post_end]
        
        # Stage 2: Quality assessment
        qa = assess_segment_quality(
            ride_ax, ride_ay, ride_az,
            pre_ax, pre_ay, pre_az,
            post_ax, post_ay, post_az,
            fs=fs
        )
        
        # Stage 3: Height estimation
        est_result = estimate_height_robust(
            t[si:ei], ride_ax, ride_ay, ride_az,
            pre_ax, pre_ay, pre_az,
            post_ax, post_ay, post_az,
            fs=fs
        )
        
        est_dh = est_result['height']
        err = abs(est_dh - gt['true_dh'])
        
        # Post-estimation consistency checks
        all_ests = est_result.get('all_estimates', {})
        h_mag = all_ests.get('magnitude', None)
        h_gp = all_ests.get('gravity_proj', None)
        method = est_result['method']
        
        # Check 1: Implausible estimates
        if abs(est_dh) > 100:
            qa['accept'] = False
            qa['reject_reason'] = f'Estimate implausible: {est_dh:.1f}m'
        
        # Check 2: Gravity-proj vs magnitude disagreement
        # If gravity-projection gives >2.5x the magnitude estimate,
        # the projection is unreliable (picking up horizontal acceleration)
        # Use lower threshold (1.8x) when gravity drift is moderate (>10deg)
        if qa['accept'] and h_mag is not None and h_gp is not None:
            if abs(h_mag) > 1.0:  # only if magnitude estimate is meaningful
                ratio = abs(h_gp) / abs(h_mag)
                drift = qa.get('features', {}).get('max_gravity_drift', 0)
                # Tighter check for rides with moderate drift
                ratio_threshold = 1.8 if drift > 10 else 2.5
                if ratio > ratio_threshold:
                    qa['accept'] = False
                    qa['reject_reason'] = f'Projection/magnitude disagree: ratio={ratio:.1f} (drift={drift:.0f}deg)'
        
        # Check 3: Signed magnitude fallback for large estimates is unreliable
        # Sign determination from initial acceleration pulse breaks down for long rides
        if qa['accept'] and method == 'signed_mag' and abs(est_dh) > 15:
            qa['accept'] = False
            qa['reject_reason'] = f'Signed-mag unreliable for large estimate ({est_dh:.1f}m)'
        
        results.append({
            **gt,
            'est_dh': est_dh,
            'err': err,
            'pos_curve': est_result['pos'],
            'method': est_result['method'],
            'quality_score': qa['quality_score'],
            'accepted': qa['accept'],
            'reject_reason': qa['reject_reason'],
            'quality_features': qa['features'],
            'agreement': est_result.get('agreement', 0),
            'all_estimates': est_result.get('all_estimates', {}),
        })
        
        accept_str = "YES" if qa['accept'] else "REJ"
        print(f"{gt['id']:4d} {gt['true_dh']:+7.1f} {est_dh:+7.2f} {err:5.2f} "
              f"{est_result['method']:>10} {qa['quality_score']:6.2f} {accept_str:>6} {gt['phone']:>6}")
    
    # ---- Compute metrics ----
    accepted = [r for r in results if r['accepted']]
    rejected = [r for r in results if not r['accepted']]
    
    all_errors = [r['err'] for r in results]
    acc_errors = [r['err'] for r in accepted]
    rej_errors = [r['err'] for r in rejected]
    
    print(f"\n--- Summary ---")
    print(f"Total rides: {len(results)}")
    print(f"Accepted: {len(accepted)} ({len(accepted)/len(results)*100:.0f}%)")
    print(f"Rejected: {len(rejected)} ({len(rejected)/len(results)*100:.0f}%)")
    
    if acc_errors:
        print(f"\nAccepted rides:")
        print(f"  MAE: {np.mean(acc_errors):.3f}m")
        print(f"  Median: {np.median(acc_errors):.3f}m")
        print(f"  Max: {np.max(acc_errors):.3f}m")
        print(f"  <0.5m: {sum(1 for e in acc_errors if e < 0.5)}/{len(acc_errors)}")
        print(f"  <1.0m: {sum(1 for e in acc_errors if e < 1.0)}/{len(acc_errors)}")
        print(f"  <2.0m: {sum(1 for e in acc_errors if e < 2.0)}/{len(acc_errors)}")
        print(f"  <3.0m: {sum(1 for e in acc_errors if e < 3.0)}/{len(acc_errors)}")
    
    if rej_errors:
        print(f"\nRejected rides (errors that would have occurred):")
        for r in rejected:
            print(f"  Ride {r['id']}: err={r['err']:.2f}m, reason={r['reject_reason']}")
    
    # ---- Conformal prediction ----
    print(f"\n{'='*50}")
    print("CONFORMAL PREDICTION")
    print(f"{'='*50}")
    
    random.seed(42)
    acc_shuffled = list(accepted)
    random.shuffle(acc_shuffled)
    
    # 50/50 split
    split = len(acc_shuffled) // 2
    train_set = acc_shuffled[:split]
    test_set = acc_shuffled[split:]
    
    train_errors = [r['err'] for r in train_set]
    test_errors = [r['err'] for r in test_set]
    
    # Compute conformal interval
    interval = compute_conformal_interval(train_errors, alpha=0.10)
    
    # Evaluate on test set
    test_covered = sum(1 for e in test_errors if e <= interval)
    coverage = test_covered / len(test_set) * 100 if test_set else 0
    
    print(f"Train set: {len(train_set)} rides")
    print(f"Test set: {len(test_set)} rides")
    print(f"90% conformal interval: ±{interval:.3f}m")
    print(f"Test coverage: {coverage:.1f}% ({test_covered}/{len(test_set)})")
    
    # Also compute leave-one-out conformal for robustness check
    if len(accepted) >= 3:
        loo_intervals = []
        loo_covered = 0
        for i in range(len(accepted)):
            loo_train = [r['err'] for j, r in enumerate(accepted) if j != i]
            loo_interval = compute_conformal_interval(loo_train, alpha=0.10)
            loo_intervals.append(loo_interval)
            if accepted[i]['err'] <= loo_interval:
                loo_covered += 1
        loo_coverage = loo_covered / len(accepted) * 100
        loo_avg_interval = np.mean(loo_intervals)
        print(f"\nLeave-one-out validation:")
        print(f"  Average interval: ±{loo_avg_interval:.3f}m")
        print(f"  Coverage: {loo_coverage:.1f}% ({loo_covered}/{len(accepted)})")
    
    # ==================================================================
    # OBJECTIVE CHECKS
    # ==================================================================
    print(f"\n{'='*50}")
    print("OBJECTIVE VERIFICATION")
    print(f"{'='*50}")
    
    # Obj 1: Detection
    det_pass = matched_gt >= len(gt_rides) * 0.7
    print(f"Obj 1 - Detection: {matched_gt}/{len(gt_rides)} matched "
          f"({'PASS' if det_pass else 'NEEDS WORK'})")
    
    # Obj 2: Rejection quality
    if rejected:
        rej_would_be_bad = sum(1 for r in rejected if r['err'] > 1.0)
        rej_quality = rej_would_be_bad / len(rejected) * 100
        print(f"Obj 2 - Rejection: {rej_would_be_bad}/{len(rejected)} rejected rides "
              f"had error >1m ({rej_quality:.0f}%)")
    else:
        print(f"Obj 2 - Rejection: No rides rejected")
    
    acc_rate = len(accepted) / len(results) * 100
    print(f"  Acceptance rate: {acc_rate:.0f}%")
    
    # Obj 3: Conformal
    conf_pass = coverage >= 90 and interval <= 1.0
    print(f"Obj 3 - Conformal: interval={interval:.3f}m, coverage={coverage:.1f}% "
          f"({'PASS' if conf_pass else 'NEEDS WORK'})")
    
    # ==================================================================
    # GENERATE FIGURES
    # ==================================================================
    _generate_figures(t, gt_t, gt_h, gt_rides, det_rides, results,
                      accepted, rejected, matches,
                      train_set, test_set, interval, coverage,
                      acc_mag, fs)
    
    # Save results JSON
    summary = {
        'n_gt_rides': len(gt_rides),
        'n_detected': len(det_rides),
        'n_matched': matched_gt,
        'n_accepted': len(accepted),
        'n_rejected': len(rejected),
        'accepted_mae': float(np.mean(acc_errors)) if acc_errors else None,
        'accepted_median': float(np.median(acc_errors)) if acc_errors else None,
        'accepted_max_err': float(np.max(acc_errors)) if acc_errors else None,
        'conformal_interval': float(interval),
        'conformal_coverage': float(coverage),
        'per_ride': [{
            'id': r['id'], 'true_dh': r['true_dh'], 'est_dh': r['est_dh'],
            'err': r['err'], 'accepted': r['accepted'], 'method': r['method'],
            'quality_score': r['quality_score'], 'phone': r['phone'],
            'all_estimates': r.get('all_estimates', {}),
            'reject_reason': r.get('reject_reason', ''),
        } for r in results]
    }
    with open(os.path.join(FIGS, "v4_results.json"), "w") as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nFigures saved to {FIGS}/")
    print("Pipeline v4 complete.")
    
    return results, interval, coverage


def _generate_figures(t, gt_t, gt_h, gt_rides, det_rides, results,
                       accepted, rejected, matches,
                       train_set, test_set, interval, coverage,
                       acc_mag, fs):
    """Generate all visualization figures."""
    
    # FIG 1: Detection overview
    fig1, axes1 = plt.subplots(3, 1, figsize=(16, 10), sharex=True)
    
    # Panel 1: GT height profile with GT ride segments
    ax = axes1[0]
    ax.plot(gt_t, gt_h, 'k-', lw=1.5, label='GT Height')
    for r in gt_rides:
        ax.axvspan(r["t_start"], r["t_end"], color='green', alpha=0.15)
    ax.set_ylabel("Height (m)")
    ax.set_title("Ground Truth Height Profile (green = GT elevator rides)")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 2: Acceleration magnitude
    ax = axes1[1]
    ax.plot(t, acc_mag, 'b-', lw=0.3, alpha=0.6)
    ax.axhline(9.81, color='gray', ls='--', alpha=0.5, label='g=9.81')
    ax.set_ylabel("Accel Mag (m/s²)")
    ax.set_title("Accelerometer Magnitude")
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Panel 3: Detection result
    ax = axes1[2]
    # GT segments in green
    for r in gt_rides:
        ax.axvspan(r["t_start"], r["t_end"], color='green', alpha=0.3, label='_')
    # Detected segments in blue
    for d in det_rides:
        ax.axvspan(t[d["s_idx"]], t[d["e_idx"]], color='blue', alpha=0.3, label='_')
    ax.set_ylabel("Segments")
    ax.set_xlabel("Time (s)")
    ax.set_title(f"Detection: {len(det_rides)} detected (blue) vs {len(gt_rides)} GT (green)")
    custom_lines = [
        plt.Rectangle((0,0), 1, 1, fc='green', alpha=0.3),
        plt.Rectangle((0,0), 1, 1, fc='blue', alpha=0.3)
    ]
    ax.legend(custom_lines, ['GT Rides', 'Detected'], loc='upper right')
    ax.grid(True, alpha=0.3)
    
    fig1.tight_layout()
    fig1.savefig(os.path.join(FIGS, "fig1_detection.png"), dpi=150)
    plt.close(fig1)
    
    # FIG 2: True vs Estimated scatter
    fig2, ax = plt.subplots(figsize=(8, 8))
    trues_a = [r["true_dh"] for r in accepted]
    ests_a = [r["est_dh"] for r in accepted]
    ax.scatter(trues_a, ests_a, c='green', alpha=0.7, s=60, label=f'Accepted ({len(accepted)})')
    if rejected:
        trues_r = [r["true_dh"] for r in rejected]
        ests_r = [r["est_dh"] for r in rejected]
        ax.scatter(trues_r, ests_r, c='red', alpha=0.5, s=60, marker='x',
                   label=f'Rejected ({len(rejected)})')
    
    all_vals = [r["true_dh"] for r in results] + [r["est_dh"] for r in results]
    lim = max(abs(min(all_vals)), abs(max(all_vals))) + 5
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='Perfect')
    ax.set_xlabel("True Δh (m)")
    ax.set_ylabel("Estimated Δh (m)")
    acc_errors = [r['err'] for r in accepted]
    mae = np.mean(acc_errors) if acc_errors else 0
    ax.set_title(f"Height Estimation: Accepted MAE={mae:.2f}m")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    fig2.tight_layout()
    fig2.savefig(os.path.join(FIGS, "fig2_scatter.png"), dpi=150)
    plt.close(fig2)
    
    # FIG 3: Per-ride error bars
    fig3, ax = plt.subplots(figsize=(16, 5))
    x = range(len(results))
    colors = ['green' if r['accepted'] else 'red' for r in results]
    errors = [min(r['err'], 20) for r in results]
    ax.bar(x, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='blue', ls='--', alpha=0.7, lw=2, label='1m target')
    ax.axhline(3.0, color='orange', ls='--', alpha=0.5, label='1 floor (3m)')
    
    # Add ride IDs
    for i, r in enumerate(results):
        ax.text(i, min(r['err'], 20) + 0.3, f"{r['id']}", ha='center', va='bottom',
                fontsize=7, rotation=90)
    
    ax.set_xlabel("Ride Index")
    ax.set_ylabel("Absolute Error (m)")
    ax.set_title("Per-Ride Error (green=accepted, red=rejected)")
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, 22)
    fig3.tight_layout()
    fig3.savefig(os.path.join(FIGS, "fig3_per_ride_errors.png"), dpi=150)
    plt.close(fig3)
    
    # FIG 4: Conformal prediction coverage
    fig4, ax = plt.subplots(figsize=(12, 6))
    if test_set:
        x4 = range(len(test_set))
        test_ests = [r["est_dh"] for r in test_set]
        test_trues = [r["true_dh"] for r in test_set]
        
        # Error bars showing conformal interval
        ax.errorbar(x4, test_ests, yerr=interval, fmt='o', color='blue',
                     ecolor='lightblue', capsize=4, capthick=2,
                     label=f'Est ± {interval:.2f}m (90% CI)')
        ax.scatter(x4, test_trues, color='red', marker='x', s=80, zorder=5,
                   label='True Δh')
        
        # Color background for covered/not-covered
        for i, r in enumerate(test_set):
            covered = r['err'] <= interval
            ax.axvspan(i - 0.4, i + 0.4, 
                       color='lightgreen' if covered else 'lightsalmon',
                       alpha=0.2)
    
    ax.set_xlabel("Test Ride Index")
    ax.set_ylabel("Height Change (m)")
    ax.set_title(f"Conformal Prediction: {coverage:.0f}% coverage, "
                 f"interval=±{interval:.2f}m (target: ≥90%, ≤1m)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig4.tight_layout()
    fig4.savefig(os.path.join(FIGS, "fig4_conformal.png"), dpi=150)
    plt.close(fig4)
    
    # FIG 5: Quality score vs error
    fig5, ax = plt.subplots(figsize=(8, 6))
    q_acc = [r['quality_score'] for r in accepted]
    e_acc = [r['err'] for r in accepted]
    ax.scatter(q_acc, e_acc, c='green', alpha=0.7, s=60, label='Accepted')
    if rejected:
        q_rej = [r['quality_score'] for r in rejected]
        e_rej = [r['err'] for r in rejected]
        ax.scatter(q_rej, e_rej, c='red', alpha=0.5, s=60, marker='x', label='Rejected')
    ax.axhline(1.0, color='blue', ls='--', alpha=0.5, label='1m target')
    ax.set_xlabel("Quality Score (lower = better)")
    ax.set_ylabel("Absolute Error (m)")
    ax.set_title("Quality Score vs Height Error")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig5.tight_layout()
    fig5.savefig(os.path.join(FIGS, "fig5_quality_vs_error.png"), dpi=150)
    plt.close(fig5)
    
    # FIG 6: Overlay on GT height
    fig6, ax = plt.subplots(figsize=(16, 5))
    ax.plot(gt_t, gt_h, 'k--', lw=1, alpha=0.5, label='Ground Truth')
    for r in accepted:
        si, ei = r["s_idx"], r["e_idx"]
        ax.plot(t[si:ei], r["h_start"] + r["pos_curve"], 'g-', lw=1.5, alpha=0.8)
    for r in rejected:
        si, ei = r["s_idx"], r["e_idx"]
        ax.plot(t[si:ei], r["h_start"] + r["pos_curve"], 'r-', lw=0.8, alpha=0.3)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Height (m)")
    ax.set_title("Estimated Height Curves vs Ground Truth")
    custom = [
        Line2D([0], [0], color='k', ls='--'),
        Line2D([0], [0], color='g', lw=2),
        Line2D([0], [0], color='r', lw=1, alpha=0.3),
    ]
    ax.legend(custom, ['GT', 'Accepted', 'Rejected'])
    ax.grid(True, alpha=0.3)
    fig6.tight_layout()
    fig6.savefig(os.path.join(FIGS, "fig6_overlay.png"), dpi=150)
    plt.close(fig6)
    
    # FIG 7: Example rides (best and worst)
    acc_sorted = sorted(accepted, key=lambda r: r['err'])
    n_best = min(4, len(acc_sorted))
    n_worst = min(2, len(acc_sorted))
    examples = acc_sorted[:n_best] + acc_sorted[-n_worst:]
    
    if examples:
        fig7, axes7 = plt.subplots(len(examples), 1, figsize=(12, 3 * len(examples)))
        if len(examples) == 1:
            axes7 = [axes7]
        for idx, d in enumerate(examples):
            ax = axes7[idx]
            si, ei = d["s_idx"], d["e_idx"]
            ride_t = t[si:ei] - t[si]
            ax.plot(ride_t, d["pos_curve"], 'b-', lw=1.5, label='Estimated')
            gt_line = np.linspace(0, d["true_dh"], len(ride_t))
            ax.plot(ride_t, gt_line, 'r--', lw=1, label='GT (linear)')
            label = "BEST" if idx < n_best else "WORST"
            ax.set_ylabel("Δh (m)")
            ax.set_title(f"[{label}] Ride {d['id']}: True={d['true_dh']:+.1f}m, "
                         f"Est={d['est_dh']:+.2f}m, Err={d['err']:.2f}m [{d['phone']}]")
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        axes7[-1].set_xlabel("Time within ride (s)")
        fig7.tight_layout()
        fig7.savefig(os.path.join(FIGS, "fig7_examples.png"), dpi=150)
        plt.close(fig7)
    
    # FIG 8: Rejection analysis
    if rejected:
        fig8, axes8 = plt.subplots(1, 2, figsize=(14, 5))
        ax = axes8[0]
        acc_errors_local = [r['err'] for r in accepted]
        rej_errors_local = [r['err'] for r in rejected]
        all_err_sorted = sorted(acc_errors_local + rej_errors_local)
        acc_err_sorted = sorted(acc_errors_local)
        ax.plot(range(len(all_err_sorted)), all_err_sorted, 'r-o', ms=4,
                label=f'All ({len(all_err_sorted)})')
        ax.plot(range(len(acc_err_sorted)), acc_err_sorted, 'g-o', ms=4,
                label=f'Accepted ({len(acc_err_sorted)})')
        ax.axhline(1.0, color='blue', ls='--', alpha=0.5)
        ax.set_xlabel("Ride (sorted by error)")
        ax.set_ylabel("Absolute Error (m)")
        ax.set_title("Error Distribution: All vs Accepted")
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        ax = axes8[1]
        reasons = [r['reject_reason'] for r in rejected]
        reason_counts = {}
        for r in reasons:
            short = r[:30]
            reason_counts[short] = reason_counts.get(short, 0) + 1
        ax.barh(list(reason_counts.keys()), list(reason_counts.values()),
                color='red', alpha=0.7)
        ax.set_xlabel("Count")
        ax.set_title("Rejection Reasons")
        ax.grid(axis='x', alpha=0.3)
        
        fig8.tight_layout()
        fig8.savefig(os.path.join(FIGS, "fig8_rejection.png"), dpi=150)
        plt.close(fig8)
    
    print(f"8 figures generated in {FIGS}/")


if __name__ == "__main__":
    main()
