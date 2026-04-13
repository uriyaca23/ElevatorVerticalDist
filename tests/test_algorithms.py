import numpy as np
import sys


# Ensure modules can be imported
sys.path.append(r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist")

from src.algorithms.algo1_direct import estimate_height_direct
from src.algorithms.algo2_zupt import estimate_height_zupt
from src.algorithms.algo3_kalman import estimate_height_kalman

def generate_synthetic_profile():
    # 100 Hz simulation
    hz = 100
    # 0 to 1s: Rest
    t1 = np.linspace(0, 1, hz, endpoint=False)
    a1 = np.zeros_like(t1)
    
    # 1 to 2s: Accel up at 1 m/s^2
    t2 = np.linspace(1, 2, hz, endpoint=False)
    a2 = np.ones_like(t2)
    
    # 2 to 4s: Coasting 
    t3 = np.linspace(2, 4, 2*hz, endpoint=False)
    a3 = np.zeros_like(t3)
    
    # 4 to 5s: Decel down at 1 m/s^2
    t4 = np.linspace(4, 5, hz, endpoint=False)
    a4 = -np.ones_like(t4)
    
    # 5 to 6s: Rest
    t5 = np.linspace(5, 6, hz, endpoint=True)
    a5 = np.zeros_like(t5)
    
    t = np.concatenate([t1, t2, t3, t4, t5])
    az = np.concatenate([a1, a2, a3, a4, a5])
    
    # Expected height:
    # 0-1s: h=0
    # 1-2s: acc=1, t=1 -> h=0.5
    # 2-4s: v=1, t=2 -> h=2.5
    # 4-5s: dec=-1, v=1, h=0.5
    # total = 3.0m
    return t, az, 3.0

def test_direct_integration():
    t, az, expected_h = generate_synthetic_profile()
    h = estimate_height_direct(t, az)
    assert np.isclose(h[-1], expected_h, atol=0.1), f"Direct failed: {h[-1]} != {expected_h}"

def test_zupt_integration():
    t, az, expected_h = generate_synthetic_profile()
    h = estimate_height_zupt(t, az)
    assert np.isclose(h[-1], expected_h, atol=0.1), f"ZUPT failed: {h[-1]} != {expected_h}"

def test_kalman_filter():
    t, az, expected_h = generate_synthetic_profile()
    h = estimate_height_kalman(t, az)
    assert np.isclose(h[-1], expected_h, atol=0.1), f"Kalman failed: {h[-1]} != {expected_h}"

if __name__ == "__main__":
    test_direct_integration()
    test_zupt_integration()
    test_kalman_filter()
    print("All synthetic tests passed.")
