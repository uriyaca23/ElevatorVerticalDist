"""
ElevatorHeightPipeline — Production-ready inference pipeline.

Detects elevator rides from raw 3-axis accelerometer data, filters
unreliable segments, estimates vertical displacement, and provides
90% conformal prediction intervals.

Usage:
    from src.pipeline import ElevatorHeightPipeline
    
    pipeline = ElevatorHeightPipeline.load("model/")
    results = pipeline.process(acc_x, acc_y, acc_z, fs=100)
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt

# Ensure src is importable
_src_dir = os.path.dirname(os.path.abspath(__file__))
if _src_dir not in sys.path:
    sys.path.insert(0, _src_dir)

from algorithms.quality_filter import (
    estimate_gravity_vector,
    angle_between_vectors,
    assess_segment_quality,
    compute_ride_gravity_drift,
)


# ======================================================================
# Core Algorithms (extracted from run_robust_pipeline.py)
# ======================================================================

def _zupt_integrate(a_vert, dt, fs, accel_threshold=0.08):
    """
    ZUPT integration with linear drift correction.
    
    1. Remove DC bias
    2. Find active motion window
    3. Integrate to velocity with endpoint correction
    4. Integrate to position
    """
    n = len(a_vert)
    a_vert = a_vert - np.mean(a_vert)

    win = min(50, max(5, n // 3))
    kernel = np.ones(win) / win
    smooth_abs = np.convolve(np.abs(a_vert), kernel, mode='same')
    active = np.where(smooth_abs > accel_threshold)[0]

    if len(active) == 0:
        return np.zeros(n), np.zeros(n)

    margin = int(fs * 0.3)
    s = max(0, active[0] - margin)
    e = min(n - 1, active[-1] + margin)

    a_windowed = np.zeros(n)
    a_windowed[s:e+1] = a_vert[s:e+1]

    vel = np.cumsum(a_windowed) * dt
    vel[:s] = 0

    if e > s:
        drift = vel[e]
        correction = np.zeros(n)
        correction[s:e+1] = np.linspace(0, drift, e - s + 1)
        vel -= correction

    vel[e+1:] = 0
    pos = np.cumsum(vel) * dt
    return pos, vel


def detect_elevator_rides(t, ax, ay, az, fs=100,
                          var_window_sec=1.5, var_thresh=1.5,
                          min_ride_sec=4.0, min_displacement_m=1.0,
                          gap_sec=2.0):
    """
    Detect individual elevator rides from 3-axis accelerometer.
    
    Strategy:
    1. Compute acceleration magnitude and its rolling variance
    2. Find low-variance (standing still) contiguous blocks
    3. Within each block, integrate to find velocity/displacement
    4. Split multi-ride blocks at velocity zero-crossings
    
    Returns list of dicts: {s_idx, e_idx, direction}
    """
    n = len(t)
    acc_mag = np.sqrt(ax**2 + ay**2 + az**2)

    var_win = int(fs * var_window_sec)
    mag_series = pd.Series(acc_mag)
    rolling_var = mag_series.rolling(window=var_win, center=True, min_periods=1).var().values
    is_still = rolling_var < var_thresh

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

    all_rides = []
    dt = 1.0 / fs

    for bs, be in blocks:
        block_mag = acc_mag[bs:be]
        block_len = be - bs
        g_mean = np.mean(block_mag)
        a_lin = block_mag - g_mean

        nyq = fs / 2
        cutoff = min(2.0, nyq * 0.8)
        b_filt, a_filt_coeff = butter(2, cutoff / nyq, btype='low')
        if block_len > 3 * max(len(b_filt), len(a_filt_coeff)):
            a_smooth = filtfilt(b_filt, a_filt_coeff, a_lin)
        else:
            a_smooth = a_lin

        vel = np.cumsum(a_smooth) * dt
        pos = np.cumsum(vel) * dt

        vel_sign = np.sign(vel)
        zero_crossings = np.where(np.diff(vel_sign) != 0)[0]
        boundaries = [0] + list(zero_crossings) + [block_len - 1]

        ride_candidates = []
        current_start = 0
        last_split_pos = pos[0]

        for zc in zero_crossings:
            segment_displacement = abs(pos[zc] - last_split_pos)
            if segment_displacement >= min_displacement_m:
                gap_samples = int(fs * gap_sec)
                look_start = max(0, zc - gap_samples)
                look_end = min(block_len, zc + gap_samples)
                vel_around = np.abs(vel[look_start:look_end])
                min_vel_around = np.min(vel_around)

                if min_vel_around < 0.1:
                    split_point = look_start + np.argmin(vel_around)
                    seg_a = a_smooth[current_start:split_point]
                    if len(seg_a) > int(fs * 1.0):
                        max_pos_a = np.max(seg_a)
                        min_neg_a = np.min(seg_a)
                        if max_pos_a > 0.1 and min_neg_a < -0.1:
                            ride_candidates.append((current_start, split_point))
                            current_start = split_point
                            last_split_pos = pos[split_point]

        if current_start < block_len - int(fs * 1.0):
            seg_a = a_smooth[current_start:]
            max_pos_a = np.max(seg_a)
            min_neg_a = np.min(seg_a)
            segment_displacement = abs(pos[-1] - pos[current_start])
            if (max_pos_a > 0.1 and min_neg_a < -0.1 and
                segment_displacement >= min_displacement_m):
                ride_candidates.append((current_start, block_len))

        if not ride_candidates:
            max_pos_a = np.max(a_smooth)
            min_neg_a = np.min(a_smooth)
            net_disp = abs(pos[-1] - pos[0])
            if max_pos_a > 0.1 and min_neg_a < -0.1 and net_disp >= min_displacement_m:
                ride_candidates = [(0, block_len)]

        for rs, re in ride_candidates:
            global_s = bs + rs
            global_e = bs + re
            seg_mag = acc_mag[global_s:global_e]
            seg_a_lin = seg_mag - np.mean(seg_mag)

            threshold = 0.08
            sig_mask = np.abs(seg_a_lin) > threshold
            if not np.any(sig_mask):
                continue
            first_sig = np.argmax(sig_mask)
            last_sig = len(sig_mask) - 1 - np.argmax(sig_mask[::-1])

            margin_samp = int(fs * 0.5)
            trim_s = max(0, first_sig - margin_samp)
            trim_e = min(len(seg_a_lin), last_sig + margin_samp)
            final_s = global_s + trim_s
            final_e = global_s + trim_e

            if final_e - final_s < int(fs * min_ride_sec):
                continue

            seg_pos = pos[rs + trim_s:rs + trim_e] if rs + trim_e <= len(pos) else pos[rs + trim_s:]
            direction = 1 if len(seg_pos) > 0 and seg_pos[-1] > seg_pos[0] else -1

            all_rides.append({
                's_idx': final_s,
                'e_idx': final_e,
                'direction': direction,
            })

    all_rides.sort(key=lambda r: r['s_idx'])
    merged = []
    for ride in all_rides:
        if merged and ride['s_idx'] < merged[-1]['e_idx']:
            merged[-1]['e_idx'] = max(merged[-1]['e_idx'], ride['e_idx'])
        else:
            merged.append(ride)
    return merged


def estimate_height_robust(t, ax, ay, az,
                           pre_ax, pre_ay, pre_az,
                           post_ax, post_ay, post_az, fs=100):
    """
    Robust height estimator with gravity projection + drift-corrected
    magnitude fallback.
    
    Returns dict: height, pos, method, quality, all_estimates
    """
    n = len(t)
    dt = 1.0 / fs

    result = {
        'height': 0.0, 'pos': np.zeros(n),
        'method': 'none', 'quality': float('inf'),
        'all_estimates': {},
    }
    if n < 20:
        return result

    # Method 1: Magnitude-based ZUPT (rotation-invariant)
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_ride = np.mean(mag)
    a_mag = mag - g_ride
    pos_mag, vel_mag = _zupt_integrate(a_mag, dt, fs)
    h_mag = pos_mag[-1]
    result['all_estimates']['magnitude'] = float(h_mag)

    # Method 2: Gravity-projected ZUPT
    pre_gvec, pre_g_mag, pre_stab = estimate_gravity_vector(pre_ax, pre_ay, pre_az, fs)
    post_gvec, post_g_mag, post_stab = estimate_gravity_vector(post_ax, post_ay, post_az, fs)

    h_gp = None
    pos_gp = None
    gp_quality = float('inf')

    pre_ok = 8.0 < pre_g_mag < 12.0 and pre_stab < 1.0
    post_ok = 8.0 < post_g_mag < 12.0 and post_stab < 1.0

    cal_gvec, cal_g_mag, cal_quality = None, None, float('inf')

    if pre_ok and post_ok:
        angle = angle_between_vectors(pre_gvec, post_gvec)
        if angle < 20:
            w1 = 1.0 / (pre_stab + 0.001)
            w2 = 1.0 / (post_stab + 0.001)
            g_avg = (pre_gvec * w1 + post_gvec * w2) / (w1 + w2)
            cal_gvec, cal_g_mag = g_avg, np.linalg.norm(g_avg)
            cal_quality = min(pre_stab, post_stab)
        else:
            if pre_stab < post_stab:
                cal_gvec, cal_g_mag, cal_quality = pre_gvec, pre_g_mag, pre_stab
            else:
                cal_gvec, cal_g_mag, cal_quality = post_gvec, post_g_mag, post_stab
    elif pre_ok:
        cal_gvec, cal_g_mag, cal_quality = pre_gvec, pre_g_mag, pre_stab
    elif post_ok:
        cal_gvec, cal_g_mag, cal_quality = post_gvec, post_g_mag, post_stab
    elif 8.0 < pre_g_mag < 12.0 and pre_stab < 1.5:
        cal_gvec, cal_g_mag, cal_quality = pre_gvec, pre_g_mag, pre_stab
    elif 8.0 < post_g_mag < 12.0 and post_stab < 1.5:
        cal_gvec, cal_g_mag, cal_quality = post_gvec, post_g_mag, post_stab

    if cal_gvec is not None:
        g_hat = cal_gvec / cal_g_mag
        gp_quality = cal_quality
        a_vert = ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - cal_g_mag
        pos_gp, vel_gp = _zupt_integrate(a_vert, dt, fs, accel_threshold=0.05)
        h_gp = pos_gp[-1]
        result['all_estimates']['gravity_proj'] = float(h_gp)

    # Method 3: Sign-corrected magnitude
    if h_gp is not None:
        h_signed_mag = np.sign(h_gp) * abs(h_mag) if abs(h_gp) > 0.5 else h_mag
    else:
        first_samples = min(n, int(fs * 3))
        early_a = a_mag[:first_samples]
        rising = np.sum(early_a[early_a > 0])
        falling = np.sum(early_a[early_a < 0])
        if abs(rising) > abs(falling) * 1.2:
            h_signed_mag = abs(h_mag)
        elif abs(falling) > abs(rising) * 1.2:
            h_signed_mag = -abs(h_mag)
        else:
            h_signed_mag = h_mag
    result['all_estimates']['signed_magnitude'] = float(h_signed_mag)

    # Select best estimate
    ride_drift, _ = compute_ride_gravity_drift(ax, ay, az, fs, chunk_sec=1.0)

    # Drift-corrected magnitude fallback
    if h_gp is not None and ride_drift > 8 and abs(h_mag) > 1.0:
        if abs(h_gp) / abs(h_mag) > 1.5:
            result['height'] = float(h_signed_mag)
            result['pos'] = pos_mag
            result['method'] = 'drift_corrected_mag'
            result['quality'] = float(gp_quality)
            return result

    if h_gp is not None and gp_quality < 0.5 and abs(h_gp) < 150:
        if abs(h_mag) > 0.5:
            agree = 1.0 - abs(abs(h_gp) - abs(h_mag)) / max(abs(h_gp), abs(h_mag))
        else:
            agree = 0.5
        if not (abs(h_gp) > 4 * max(abs(h_mag), 1.0) and abs(h_gp) > 30):
            result['height'] = float(h_gp)
            result['pos'] = pos_gp
            result['method'] = 'gravity_proj'
            result['quality'] = float(gp_quality)
            return result

    if h_gp is not None and abs(h_gp) < 150:
        if abs(h_mag) > 0.5:
            agree = 1.0 - abs(abs(h_gp) - abs(h_mag)) / max(abs(h_gp), abs(h_mag))
        else:
            agree = 0.5
        if agree > 0.3 and not (abs(h_gp) > 3 * max(abs(h_mag), 1.0)):
            result['height'] = float(h_gp)
            result['pos'] = pos_gp
            result['method'] = 'gravity_proj'
            result['quality'] = float(gp_quality)
            return result

    result['height'] = float(h_signed_mag)
    result['pos'] = pos_mag
    result['method'] = 'signed_mag'
    result['quality'] = 0.5
    return result


def compute_conformal_interval(train_errors, alpha=0.10):
    """Split-conformal prediction interval from absolute errors."""
    errors = np.abs(np.array(train_errors))
    n = len(errors)
    if n == 0:
        return 1.0
    q_level = np.ceil((n + 1) * (1 - alpha)) / n
    q_level = min(q_level, 1.0)
    return float(np.quantile(errors, q_level))


# ======================================================================
# Main Pipeline Class
# ======================================================================

class ElevatorHeightPipeline:
    """
    Production-ready elevator height estimation pipeline.
    
    Three stages: Detection → Quality Filter → Estimation
    With conformal prediction for 90% coverage intervals.
    
    Usage:
        pipeline = ElevatorHeightPipeline.load("model/")
        results = pipeline.process(acc_x, acc_y, acc_z, fs=100)
    """

    # --- Default hyperparameters ---
    DEFAULT_PARAMS = {
        # Detection
        'var_window_sec': 1.5,
        'var_thresh': 1.5,
        'min_ride_sec': 4.0,
        'min_displacement_m': 1.0,
        'gap_sec': 2.0,
        # Quality filter context
        'pre_window_sec': 5.0,
        'post_window_sec': 5.0,
        # Post-estimation rejection
        'max_implausible_m': 100.0,
        'mag_cross_ratio': 1.8,
        'signed_mag_max_m': 15.0,
    }

    def __init__(self, fs=100, conformal_interval=None, conformal_residuals=None,
                 params=None):
        self.fs = fs
        self.conformal_interval = conformal_interval  # 90% interval half-width
        self.conformal_residuals = conformal_residuals or []
        self.params = {**self.DEFAULT_PARAMS, **(params or {})}

    @classmethod
    def load(cls, model_dir="model/"):
        """Load a pre-calibrated pipeline from saved parameters."""
        params_path = os.path.join(model_dir, "pipeline_params.json")
        if os.path.exists(params_path):
            with open(params_path, "r") as f:
                saved = json.load(f)
            return cls(
                fs=saved.get('fs', 100),
                conformal_interval=saved.get('conformal_interval'),
                conformal_residuals=saved.get('conformal_residuals', []),
                params=saved.get('params', {}),
            )
        return cls()

    def save(self, model_dir="model/"):
        """Save pipeline parameters and conformal calibration."""
        os.makedirs(model_dir, exist_ok=True)
        saved = {
            'fs': self.fs,
            'conformal_interval': self.conformal_interval,
            'conformal_residuals': self.conformal_residuals,
            'params': self.params,
        }
        path = os.path.join(model_dir, "pipeline_params.json")
        with open(path, "w") as f:
            json.dump(saved, f, indent=2)
        return path

    def calibrate(self, rides_with_gt):
        """
        Calibrate conformal prediction from labeled rides.
        
        Args:
            rides_with_gt: list of dicts with keys:
                - acc_x, acc_y, acc_z: 1D arrays of accelerometer data
                - true_height: ground truth height difference (meters)
                - fs: sample rate (optional, defaults to self.fs)
        
        Returns:
            dict with calibration statistics
        """
        residuals = []
        for ride in rides_with_gt:
            ax = ride['acc_x']
            ay = ride['acc_y']
            az = ride['acc_z']
            fs = ride.get('fs', self.fs)
            true_h = ride['true_height']
            
            t = np.arange(len(ax)) / fs
            # Use zeros for pre/post if not available
            pre_ax = ride.get('pre_acc_x', np.full(int(fs*2), np.mean(ax[:int(fs)])))
            pre_ay = ride.get('pre_acc_y', np.full(int(fs*2), np.mean(ay[:int(fs)])))
            pre_az = ride.get('pre_acc_z', np.full(int(fs*2), np.mean(az[:int(fs)])))
            post_ax = ride.get('post_acc_x', np.full(int(fs*2), np.mean(ax[-int(fs):])))
            post_ay = ride.get('post_acc_y', np.full(int(fs*2), np.mean(ay[-int(fs):])))
            post_az = ride.get('post_acc_z', np.full(int(fs*2), np.mean(az[-int(fs):])))
            
            est = estimate_height_robust(t, ax, ay, az,
                                         pre_ax, pre_ay, pre_az,
                                         post_ax, post_ay, post_az, fs=fs)
            residuals.append(abs(est['height'] - true_h))
        
        self.conformal_residuals = [float(r) for r in residuals]
        self.conformal_interval = compute_conformal_interval(residuals)
        return {
            'n_rides': len(residuals),
            'interval_90': self.conformal_interval,
            'mean_error': float(np.mean(residuals)),
            'median_error': float(np.median(residuals)),
        }

    def process(self, acc_x, acc_y, acc_z, fs=None):
        """
        Process raw 3-axis accelerometer recording.
        
        Args:
            acc_x, acc_y, acc_z: 1D numpy arrays of accelerometer data (m/s²)
            fs: Sampling frequency in Hz (default: self.fs)
        
        Returns:
            List of dicts, one per detected ride:
            {
                'start_time': float,    # start time in seconds
                'end_time': float,      # end time in seconds
                'height_estimate': float,  # estimated height difference (m)
                'confidence_interval_90': float or None,  # ±margin for 90% coverage
                'method': str,          # estimation method used
                'accepted': bool,       # whether quality filter accepted
                'reject_reason': str,   # reason for rejection (if any)
                'quality_features': dict,  # quality assessment features
            }
        """
        fs = fs or self.fs
        acc_x = np.asarray(acc_x, dtype=float)
        acc_y = np.asarray(acc_y, dtype=float)
        acc_z = np.asarray(acc_z, dtype=float)
        
        n = len(acc_x)
        t = np.arange(n) / fs

        # Stage 1: Detect rides
        det_rides = detect_elevator_rides(
            t, acc_x, acc_y, acc_z, fs=fs,
            **{k: self.params[k] for k in
               ['var_window_sec', 'var_thresh', 'min_ride_sec',
                'min_displacement_m', 'gap_sec']}
        )

        pre_win = int(fs * self.params['pre_window_sec'])
        post_win = int(fs * self.params['post_window_sec'])

        results = []
        for ride in det_rides:
            si, ei = ride['s_idx'], ride['e_idx']
            ride_ax = acc_x[si:ei]
            ride_ay = acc_y[si:ei]
            ride_az = acc_z[si:ei]
            ride_t = t[si:ei]

            # Context windows
            pre_s = max(0, si - pre_win)
            post_e = min(n, ei + post_win)
            pre_ax = acc_x[pre_s:si]
            pre_ay = acc_y[pre_s:si]
            pre_az = acc_z[pre_s:si]
            post_ax = acc_x[ei:post_e]
            post_ay = acc_y[ei:post_e]
            post_az = acc_z[ei:post_e]

            # Stage 2: Quality assessment
            qa = assess_segment_quality(
                ride_ax, ride_ay, ride_az,
                pre_ax, pre_ay, pre_az,
                post_ax, post_ay, post_az,
                fs=fs
            )

            # Stage 3: Height estimation
            est = estimate_height_robust(
                ride_t, ride_ax, ride_ay, ride_az,
                pre_ax, pre_ay, pre_az,
                post_ax, post_ay, post_az,
                fs=fs
            )

            # Post-estimation consistency checks
            h_mag = est['all_estimates'].get('magnitude')
            h_gp = est['all_estimates'].get('gravity_proj')
            method = est['method']

            if abs(est['height']) > self.params['max_implausible_m']:
                qa['accept'] = False
                qa['reject_reason'] = f'Estimate implausible: {est["height"]:.1f}m'

            if qa['accept'] and h_mag is not None and h_gp is not None:
                if abs(h_mag) > 1.0:
                    ratio = abs(h_gp) / abs(h_mag)
                    drift = qa.get('features', {}).get('max_gravity_drift', 0)
                    threshold = self.params['mag_cross_ratio']
                    if ratio > threshold:
                        qa['accept'] = False
                        qa['reject_reason'] = f'Projection/magnitude disagree: ratio={ratio:.1f}'

            if (qa['accept'] and method == 'signed_mag' and
                    abs(est['height']) > self.params['signed_mag_max_m']):
                qa['accept'] = False
                qa['reject_reason'] = f'Signed-mag unreliable for large estimate ({est["height"]:.1f}m)'

            results.append({
                'start_time': float(ride_t[0]),
                'end_time': float(ride_t[-1]),
                'height_estimate': float(est['height']),
                'confidence_interval_90': self.conformal_interval,
                'method': method,
                'accepted': qa['accept'],
                'reject_reason': qa.get('reject_reason', ''),
                'quality_features': qa.get('features', {}),
                'quality_score': qa.get('quality_score', 0),
            })

        return results

    def process_accepted(self, acc_x, acc_y, acc_z, fs=None):
        """
        Process and return only accepted (non-rejected) rides.
        Convenience wrapper around process().
        """
        all_results = self.process(acc_x, acc_y, acc_z, fs=fs)
        return [r for r in all_results if r['accepted']]

    def process_plot(self, acc_x, acc_y, acc_z, fs=None,
                     save_path=None, show=False, title=None):
        """
        Process accelerometer data and generate a visualization figure.

        Shows:
          - Top: Accelerometer magnitude with detected segments highlighted
          - Bottom: Bar chart of height estimates for each ride (green/red)

        Args:
            acc_x, acc_y, acc_z: 1D numpy arrays (m/s²)
            fs: Sampling frequency (Hz)
            save_path: Path to save figure (PNG). If None, returns the figure.
            show: If True, call plt.show()
            title: Optional figure title

        Returns:
            (results, fig): Tuple of (list of result dicts, matplotlib Figure)
        """
        import matplotlib
        if save_path and not show:
            matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D

        fs = fs or self.fs
        acc_x = np.asarray(acc_x, dtype=float)
        acc_y = np.asarray(acc_y, dtype=float)
        acc_z = np.asarray(acc_z, dtype=float)

        results = self.process(acc_x, acc_y, acc_z, fs=fs)
        n = len(acc_x)
        t = np.arange(n) / fs
        mag = np.sqrt(acc_x**2 + acc_y**2 + acc_z**2)

        fig, axes = plt.subplots(2, 1, figsize=(16, 8),
                                 gridspec_kw={'height_ratios': [2, 1.2]})

        # --- Top panel: Accelerometer magnitude + detected segments ---
        axes[0].plot(t, mag, linewidth=0.2, color='#7f8c8d', alpha=0.7)
        axes[0].axhline(9.81, color='red', linewidth=0.5, alpha=0.2)
        axes[0].set_ylabel('|a| (m/s²)')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylim(
            max(0, np.percentile(mag, 0.5) - 2),
            np.percentile(mag, 99.5) + 2
        )

        for i, r in enumerate(results):
            color = '#27ae60' if r['accepted'] else '#e74c3c'
            axes[0].axvspan(r['start_time'], r['end_time'],
                            alpha=0.2, color=color, zorder=2)
            # Label with ride number
            mid_t = (r['start_time'] + r['end_time']) / 2
            axes[0].text(mid_t, axes[0].get_ylim()[1] * 0.95,
                         str(i + 1), ha='center', fontsize=7,
                         fontweight='bold', color=color)

        axes[0].set_title(title or 'Detected Elevator Rides', fontsize=13)
        axes[0].legend(handles=[
            Line2D([0], [0], color='#27ae60', lw=8, alpha=0.3,
                   label='Accepted'),
            Line2D([0], [0], color='#e74c3c', lw=8, alpha=0.3,
                   label='Rejected'),
        ], fontsize=9, loc='upper right')

        # --- Bottom panel: Height estimates bar chart ---
        if results:
            ride_labels = []
            heights = []
            colors = []
            for i, r in enumerate(results):
                ride_labels.append(f"R{i+1}\n{r['start_time']:.0f}–{r['end_time']:.0f}s")
                heights.append(r['height_estimate'])
                colors.append('#27ae60' if r['accepted'] else '#e74c3c')

            x_pos = range(len(ride_labels))
            bars = axes[1].bar(x_pos, heights, color=colors,
                               edgecolor='white', width=0.7, alpha=0.85)

            # Add CI whiskers for accepted rides
            ci = self.conformal_interval
            if ci:
                for i, r in enumerate(results):
                    if r['accepted']:
                        axes[1].errorbar(i, r['height_estimate'], yerr=ci,
                                         color='#2c3e50', capsize=4,
                                         capthick=1.5, linewidth=1.5, zorder=5)

            # Value labels on bars
            for i, (h, bar) in enumerate(zip(heights, bars)):
                va = 'bottom' if h >= 0 else 'top'
                offset = 0.3 if h >= 0 else -0.3
                axes[1].text(i, h + offset, f'{h:+.1f}m',
                             ha='center', va=va, fontsize=8, fontweight='bold')

            axes[1].set_xticks(x_pos)
            axes[1].set_xticklabels(ride_labels, fontsize=7)
            axes[1].set_ylabel('Height Difference (m)')
            axes[1].axhline(0, color='gray', linewidth=0.5)

            ci_label = f'  (whiskers = ±{ci:.1f}m 90% CI)' if ci else ''
            axes[1].set_title(f'Height Estimates per Ride{ci_label}', fontsize=11)
        else:
            axes[1].text(0.5, 0.5, 'No rides detected',
                         transform=axes[1].transAxes, ha='center', fontsize=14)
            axes[1].set_title('No rides detected')

        fig.tight_layout()

        if save_path:
            fig.savefig(save_path, bbox_inches='tight', dpi=150)

        if show:
            plt.show()

        return results, fig
