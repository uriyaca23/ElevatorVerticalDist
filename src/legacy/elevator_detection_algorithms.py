import numpy as np
from scipy.signal import butter, filtfilt
import fastdtw
from scipy.spatial.distance import euclidean

class ElevatorDetector:
    def __init__(self, fs=100.0):
        self.fs = fs

    def _get_linear_vertical_accel(self, acc_z):
        # We assume acc_z is roughly vertical, and we subtract gravity
        # In a generic dataset, gravity might be estimated via moving average
        g_estimate = pd.Series(acc_z).rolling(window=int(self.fs*10), center=True, min_periods=1).mean().values
        return acc_z - g_estimate

    def _get_moving_variance(self, signal, window_sec=1.5):
        w = int(self.fs * window_sec)
        import pandas as pd
        s = pd.Series(signal)
        return s.rolling(window=w, center=True, min_periods=1).var().values

    # ALGORITHM 1: STATE MACHINE VIA LOW-PASS FILTERING
    def detect_algorithm1_state_machine(self, time_sec, acc_z, var_thresh=0.8, acc_thresh=0.25):
        import pandas as pd
        a_lin = acc_z - np.mean(acc_z) # Simplified gravity removal assuming mean is g
        
        # Low pass filter
        b, a = butter(2, 0.5 / (self.fs / 2), btype='low')
        a_filt = filtfilt(b, a, a_lin)
        
        var_z = self._get_moving_variance(acc_z, 1.5)
        
        is_standing = var_z < var_thresh
        
        segments = []
        in_elevator = False
        start_idx = -1
        
        # State machine
        # We look for a continuous block where variance is low, and absolute acceleration exceeds threshold at least once
        current_stand_start = -1
        for i in range(1, len(is_standing)):
            if is_standing[i] and not is_standing[i-1]:
                current_stand_start = i
            elif not is_standing[i] and is_standing[i-1] and current_stand_start != -1:
                # End of a standing block
                block_a = a_filt[current_stand_start:i]
                if len(block_a) > self.fs * 5: # At least 5 seconds
                    if np.max(block_a) > acc_thresh and np.min(block_a) < -acc_thresh:
                        # Saw both positive and negative acceleration pulses without walking!
                        segments.append((current_stand_start, i))
                current_stand_start = -1
        
        return segments

    # ALGORITHM 2: SLIDING DTW PATTERN MATCHING
    def detect_algorithm2_dtw(self, time_sec, acc_z, var_thresh=0.8, dtw_thresh=500.0):
        # Create an ideal template: 1s +0.5 m/s2, 3s 0 m/s2, 1s -0.5 m/s2
        t_pos = np.ones(int(self.fs * 1.0)) * 0.5
        t_zero = np.zeros(int(self.fs * 3.0))
        t_neg = np.ones(int(self.fs * 1.0)) * -0.5
        template = np.concatenate([t_pos, t_zero, t_neg])
        
        a_lin = acc_z - np.mean(acc_z)
        var_z = self._get_moving_variance(acc_z, 1.5)
        is_standing = var_z < var_thresh
        
        segments = []
        current_stand_start = -1
        for i in range(1, len(is_standing)):
            if is_standing[i] and not is_standing[i-1]:
                current_stand_start = i
            elif not is_standing[i] and is_standing[i-1] and current_stand_start != -1:
                block_a = a_lin[current_stand_start:i]
                if len(block_a) > len(template):
                    # Downsample for speed
                    step = max(1, int(len(block_a) / 100))
                    b_down = block_a[::step]
                    t_down = template[::max(1, int(len(template)/20))]
                    
                    dist, path = fastdtw.fastdtw(b_down, t_down, dist=lambda x, y: abs(x - y))
                    # Normalize distance
                    dist = dist / len(b_down)
                    
                    if dist < dtw_thresh: # Need to tune this threshold!
                        segments.append((current_stand_start, i))
                current_stand_start = -1
                
        return segments

    # ALGORITHM 3: VELOCITY INTEGRAL BOUNDING (ZUPT-based)
    def detect_algorithm3_integral(self, time_sec, acc_z, var_thresh=0.8, min_v=0.8, min_h=2.0):
        a_lin = acc_z - np.mean(acc_z)
        var_z = self._get_moving_variance(acc_z, 1.5)
        is_standing = var_z < var_thresh
        
        segments = []
        dt = 1.0 / self.fs
        
        current_stand_start = -1
        for i in range(1, len(is_standing)):
            if is_standing[i] and not is_standing[i-1]:
                current_stand_start = i
            elif not is_standing[i] and is_standing[i-1] and current_stand_start != -1:
                block_a = a_lin[current_stand_start:i]
                if len(block_a) > self.fs * 5:
                    v = np.cumsum(block_a) * dt
                    h = np.cumsum(v) * dt
                    
                    max_v = np.max(np.abs(v))
                    net_h = np.abs(h[-1])
                    final_v = np.abs(v[-1])
                    
                    if max_v > min_v and net_h > min_h and final_v < 1.0:
                        segments.append((current_stand_start, i))
                        
                current_stand_start = -1
                
        return segments
