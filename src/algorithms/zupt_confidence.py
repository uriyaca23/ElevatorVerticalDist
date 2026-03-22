import numpy as np
import scipy.stats as stats

try:
    from algorithms.noise_db import SENSOR_CHIPS
except ImportError:
    SENSOR_CHIPS = None

class ZuptConfidenceAnalyzer:
    def __init__(self, dt=0.01, max_zupt_duration=120.0, max_stationary_var_multiplier=5.0, max_accel_peak=5.0):
        self.dt = dt
        self.max_zupt_duration = max_zupt_duration
        self.max_stationary_var_multiplier = max_stationary_var_multiplier
        self.max_accel_peak = max_accel_peak
        self.calibrated_margin = 0.0
        self.calibrated_multiplier = 1.645  # Default to 90% Normal Distribution (1.645 sigma)

    def get_noise_sigma(self, phone_model):
        if SENSOR_CHIPS and phone_model in SENSOR_CHIPS:
            params = SENSOR_CHIPS[phone_model].to_noise_params(sampling_rate_hz=1.0/self.dt)
            return params.accel_noise_sigma
        return 0.05  # Fallback

    def compute_theoretical_confidence(self, num_steps, phone_model):
        """
        Computes the theoretical standard deviation of position error
        due to integrated wideband acceleration noise over a ZUPT interval.
        """
        sigma_a = self.get_noise_sigma(phone_model)
        # Variance = sigma_a^2 * dt^4 * (N^3 / 12)
        # StdDev = sigma_a * dt^2 * sqrt(N^3 / 12)
        sigma_pos = sigma_a * (self.dt**2) * np.sqrt((num_steps**3) / 12.0)
        return sigma_pos

    def evaluate_rejection(self, az, start_idx, end_idx, phone_model):
        """
        Evaluates whether a sample should be rejected.
        Returns (rejected: bool, reason: str)
        
        Rejection criteria:
        1. Duration limit: ZUPT window > max_zupt_duration (very long rides)
        2. Impact detection: acceleration peak during motion exceeds max_accel_peak
        3. Stationary variance: variance of rest periods exceeds expected sensor noise
        4. Motion variance: excessive noise variance during active motion (shaking)
        """
        if start_idx >= end_idx:
            return True, "Invalid window (start >= end)"
            
        num_steps = end_idx - start_idx
        duration = num_steps * self.dt
        
        if duration > self.max_zupt_duration:
            return True, f"Duration too long ({duration:.1f}s > {self.max_zupt_duration}s)"
            
        # Estimate gravity from stationary period
        gravity = np.mean(az[:start_idx]) if start_idx > 0 else 9.81
        az_residual = az[start_idx:end_idx] - gravity
        
        # --- Impact detection ---
        # Check for isolated extreme peaks (spikes) in the motion window
        max_peak = np.max(np.abs(az_residual))
        if max_peak > self.max_accel_peak:
            return True, f"Impact detected (peak {max_peak:.1f} m/s² > {self.max_accel_peak} m/s²)"
            
        # --- Stationary variance check ---
        sigma_a = self.get_noise_sigma(phone_model)
        stat_points = np.concatenate([az[:max(1, start_idx)], az[min(len(az)-1, end_idx):]])
        if len(stat_points) > 10:
            stat_var = np.var(stat_points)
            if stat_var > (self.max_stationary_var_multiplier * sigma_a)**2:
                return True, f"Stationary noise too high (var={stat_var:.4f}, anomalous handling)"
                
        # --- Motion-window shaking detection ---
        # During normal elevator motion, the residual should be smooth (trapezoidal profile).
        # Shaking adds high-frequency noise. We check the variance of the high-pass
        # filtered residual: subtract a smoothed version and measure what remains.
        if num_steps > 100:
            # Smooth with a wide kernel to extract the elevator motion component
            kernel_size = min(101, num_steps // 2 * 2 + 1)  # must be odd
            smooth = np.convolve(az_residual, np.ones(kernel_size)/kernel_size, mode='same')
            high_freq = az_residual - smooth
            # The high-freq component should have variance ~ sensor noise
            hf_var = np.var(high_freq)
            # If it's > 10x the expected sensor noise variance, flag as shaking
            if hf_var > (self.max_stationary_var_multiplier * 2 * sigma_a)**2:
                return True, f"Shaking detected (HF variance {hf_var:.4f} >> sensor noise)"
                
        return False, "Accepted"

    def fit_conformal(self, errors, theoretical_sigmas, alpha=0.1):
        """
        Fits empirical confidence interval bounds using conformal prediction methods.
        alpha=0.1 means 90% confidence.
        We will find a scalar multiplier 'k' such that P(|error| <= k * sigma_theory) = 1 - alpha.
        """
        if len(errors) == 0:
            return
            
        errors = np.abs(np.array(errors))
        sigmas = np.array(theoretical_sigmas)
        
        # We calculate the non-conformity scores: score = |error| / theoretical_sigma
        # (Adding a small epsilon to avoid division by zero)
        scores = errors / (sigmas + 1e-9)
        
        n = len(scores)
        q_idx = int(np.ceil((n + 1) * (1 - alpha)))
        # Make sure q_idx is within bounds
        q_idx = min(q_idx, n - 1)
        
        sorted_scores = np.sort(scores)
        self.calibrated_multiplier = sorted_scores[q_idx]
        
        # We can also compute an additive margin just in case
        constant_margins = np.sort(errors - self.calibrated_multiplier * sigmas)
        self.calibrated_margin = max(0, constant_margins[q_idx])

    def get_confidence_interval(self, num_steps, phone_model):
        """
        Returns the final empirically calibrated 90% confidence margin.
        If fit_conformal() hasn't been called, it uses the 1.645 Z-score default.
        """
        sigma_pos = self.compute_theoretical_confidence(num_steps, phone_model)
        margin = (self.calibrated_multiplier * sigma_pos) + self.calibrated_margin
        return margin
