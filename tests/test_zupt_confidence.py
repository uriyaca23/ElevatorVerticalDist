import os
import sys
import numpy as np
import pytest
from pathlib import Path

# Add src to sys.path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from algorithms.zupt_confidence import ZuptConfidenceAnalyzer

def test_theoretical_confidence():
    analyzer = ZuptConfidenceAnalyzer(dt=0.01)
    
    # Very short interval (1s) vs 10s
    sigma_1s = analyzer.compute_theoretical_confidence(num_steps=100, phone_model="generic_premium")
    sigma_10s = analyzer.compute_theoretical_confidence(num_steps=1000, phone_model="generic_premium")
    
    assert sigma_10s > sigma_1s, "Longer integrals should accumulate more theoretical uncertainty"
    assert sigma_1s > 0, "Confidence should be positive"

def test_evaluate_rejection_duration():
    analyzer = ZuptConfidenceAnalyzer(dt=0.01, max_zupt_duration=10.0)
    # 1500 steps * 0.01 = 15.0s > 10.0s
    az_mock = np.ones(2000) * 9.81
    rejected, reason = analyzer.evaluate_rejection(az_mock, 100, 1600, phone_model="generic_premium")
    assert rejected is True
    assert "too long" in reason.lower() or "duration" in reason.lower()

def test_evaluate_rejection_impact():
    analyzer = ZuptConfidenceAnalyzer(dt=0.01, max_accel_peak=10.0)
    az_mock = np.ones(1000) * 9.81
    az_mock[500] = 9.81 + 15.0 # Spike
    rejected, reason = analyzer.evaluate_rejection(az_mock, 100, 900, phone_model="generic_premium")
    assert rejected is True
    assert "impact" in reason.lower() or "peak" in reason.lower()

def test_conformal_tuning():
    analyzer = ZuptConfidenceAnalyzer(dt=0.01)
    # Mock some data
    errors = np.array([0.1, 0.5, 0.2, 0.8, 1.2, 0.4, 0.3, 2.5, 1.1, 0.9])
    theoretical_sigmas = np.ones_like(errors) * 0.5
    
    analyzer.fit_conformal(errors, theoretical_sigmas, alpha=0.1)
    # n=10, 90% confidence index -> ceil(11 * 0.9) = 10 -> index 9
    # The sorted errors are [0.1, 0.2, 0.3, 0.4, 0.5, 0.8, 0.9, 1.1, 1.2, 2.5]
    # Corresponding scores (errors/0.5): max will be 2.5 / 0.5 = 5.0
    
    margin = analyzer.get_confidence_interval(num_steps=100, phone_model="generic_premium")
    assert margin > 0.0
    assert analyzer.calibrated_multiplier > 1.6 # Should be tuned upwards
