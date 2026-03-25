"""
Robust Adaptive Gravity-Projected ZUPT Estimator v2.

Uses vectorized NumPy operations for speed.
Three strategies with automatic selection:
1. Static gravity projection (pre-ride + post-ride average)
2. Magnitude-based ZUPT (rotation-invariant fallback)
3. Predictive: weighted average of methods based on agreement

Quality checks determine which approach to use per ride.
"""
import numpy as np


def _estimate_stationary_gravity(ax, ay, az, fs=100):
    """Estimate gravity vector from a stationary window."""
    if len(ax) < 10:
        return np.array([0, 0, 9.81]), 9.81, float('inf')
    
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    win = min(int(fs), len(ax))
    if win < 10:
        win = len(ax)
    
    best_var = float('inf')
    best_s = 0
    for s in range(0, len(ax) - win + 1, max(1, win // 4)):
        v = np.var(mag[s:s+win])
        if v < best_var:
            best_var = v
            best_s = s
    
    sl = slice(best_s, best_s + win)
    gvec = np.array([np.mean(ax[sl]), np.mean(ay[sl]), np.mean(az[sl])])
    g_mag = np.linalg.norm(gvec)
    stat = np.std(mag[sl])
    return gvec, g_mag, stat


def _zupt_integrate_vec(a_vert, dt, fs, accel_threshold=0.05):
    """Vectorized ZUPT integration with drift correction."""
    n = len(a_vert)
    win = min(50, max(5, n // 3))
    kernel = np.ones(win) / win
    smooth = np.convolve(np.abs(a_vert), kernel, mode='same')
    active = np.where(smooth > accel_threshold)[0]
    
    if len(active) == 0:
        return np.zeros(n)
    
    margin = int(fs * 0.5)
    s = max(0, active[0] - margin)
    e = min(n - 1, active[-1] + margin)
    
    # Zero outside window, integrate inside
    a_windowed = np.zeros(n)
    a_windowed[s:e+1] = a_vert[s:e+1]
    
    vel = np.cumsum(a_windowed * dt)
    # Zero before start
    vel[:s] = 0
    
    # Linear drift correction within window
    if e > s:
        drift = vel[e]
        correction = np.zeros(n)
        correction[s:e+1] = np.linspace(0, drift, e - s + 1)
        vel -= correction
    
    # Zero after end
    vel[e+1:] = vel[e]
    
    pos = np.cumsum(vel * dt)
    return pos


def estimate_height_adaptive(t, ax, ay, az,
                              pre_ax, pre_ay, pre_az,
                              post_ax=None, post_ay=None, post_az=None,
                              fs=100):
    """
    Adaptive height estimator that picks the best approach per ride.
    
    Returns dict with:
        'height': final estimated height difference
        'pos': position time series  
        'method': which method was selected
        'quality': overall quality score
        'rejected': True if estimate is unreliable
        'reject_reason': explanation if rejected
        'all_estimates': dict of all method results for comparison
    """
    n = len(t)
    dt = np.diff(t, prepend=t[0])
    dt[0] = 0
    
    result = {
        'height': 0.0, 'pos': np.zeros(n), 'method': 'none',
        'quality': float('inf'), 'rejected': False, 'reject_reason': '',
        'all_estimates': {}
    }
    
    if n < 20:
        result['rejected'] = True
        result['reject_reason'] = 'Segment too short'
        return result
    
    # ======== METHOD 1: Magnitude ZUPT ========
    mag = np.sqrt(ax**2 + ay**2 + az**2)
    g_mag_ride = np.mean(mag)
    a_mag_vert = mag - g_mag_ride
    pos_mag = _zupt_integrate_vec(a_mag_vert, dt, fs, 0.1)
    h_mag = pos_mag[-1]
    result['all_estimates']['magnitude'] = float(h_mag)
    
    # ======== METHOD 2: Static gravity projection ========
    g_pre, g_mag_pre, stat_pre = _estimate_stationary_gravity(pre_ax, pre_ay, pre_az, fs)
    
    h_static = None
    pos_static = None
    stat_quality = stat_pre
    
    if 8.0 < g_mag_pre < 12.0 and stat_pre < 1.0:
        g_hat = g_pre / g_mag_pre
        
        # Post-ride gravity for averaging
        if post_ax is not None and len(post_ax) > 50:
            g_post, g_mag_post, stat_post = _estimate_stationary_gravity(post_ax, post_ay, post_az, fs)
            if 8.0 < g_mag_post < 12.0 and stat_post < 1.0:
                g_hat_post = g_post / g_mag_post
                cos_angle = np.clip(np.dot(g_hat, g_hat_post), -1, 1)
                if cos_angle > np.cos(np.radians(20)):
                    w1 = 1.0 / (stat_pre + 0.001)
                    w2 = 1.0 / (stat_post + 0.001)
                    g_avg = (g_pre * w1 + g_post * w2) / (w1 + w2)
                    g_mag_pre = np.linalg.norm(g_avg)
                    g_hat = g_avg / g_mag_pre
                    stat_quality = min(stat_pre, stat_post)
        
        # Project acceleration onto gravity direction
        a_vert = ax * g_hat[0] + ay * g_hat[1] + az * g_hat[2] - g_mag_pre
        pos_static = _zupt_integrate_vec(a_vert, dt, fs, 0.05)
        h_static = pos_static[-1]
        result['all_estimates']['static_gp'] = float(h_static)
    
    # ======== SELECT BEST METHOD ========
    # Agreement-based hybrid: when GP and magnitude agree, GP is more precise.
    # When they disagree, magnitude is safer (rotation-invariant).
    
    # Compute agreement: 1.0 = perfect match, 0.0 = no agreement, negative = opposite signs
    if max(abs(h_mag), abs(h_static) if h_static is not None else 0, 1.0) > 0:
        denominator = max(abs(h_mag), abs(h_static) if h_static is not None else 0, 1.0)
        agree = 1.0 - abs((h_static if h_static is not None else h_mag) - h_mag) / denominator
    else:
        agree = 1.0
    
    selected = None
    
    if h_static is not None and abs(h_static) < 150:
        # GP is available and plausible
        gp_too_large = abs(h_static) > 3 * max(abs(h_mag), 1.0) and abs(h_static) > 20
        
        if not gp_too_large:
            if agree > 0.5:
                # GP and magnitude agree — use GP (more precise with sign info)
                selected = ('static_gp', h_static, pos_static, stat_quality)
            elif stat_quality < 0.10:
                # Very high quality pre-ride — trust GP even without agreement
                selected = ('static_gp', h_static, pos_static, stat_quality)
            elif agree > 0.0 and stat_quality < 0.20:
                # Moderate agreement + good quality — cautiously use GP
                selected = ('static_gp', h_static, pos_static, stat_quality)
    
    # Fall back to magnitude
    if selected is None:
        if abs(h_mag) < 150:
            selected = ('magnitude', h_mag, pos_mag, 0.5)
        else:
            result['rejected'] = True
            result['reject_reason'] = f'All estimates implausible'
            return result
    
    result['height'] = float(selected[1])
    result['pos'] = selected[2]
    result['method'] = selected[0]
    result['quality'] = float(selected[3])
    result['agreement'] = float(agree)
    
    return result

