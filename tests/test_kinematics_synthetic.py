"""Test script for S-curve estimator on synthetic data."""
import sys
sys.path.insert(0, '.')
import numpy as np
from src.algorithms.scurve_model import SCurveParams, generate_profile
from src.algorithms.kinematics_estimator import estimate_height_accel_only

# Monte Carlo with detailed diagnostics
print("=== Monte Carlo CI coverage test (detailed) ===")
np.random.seed(0)

params = SCurveParams(j_max=1.8, a_max=1.0, v_max=1.2, distance=9.0, direction=1)
t_base = np.arange(0, 15.0, 0.01)
noise_std = 0.08
g = 9.81

errors = []
ci_widths = []
covered = 0
n_trials = 100

for trial in range(n_trials):
    np.random.seed(trial + 200)
    t = t_base + np.random.uniform(-0.001, 0.001, len(t_base))
    t = np.sort(t)

    a_true, _, _ = generate_profile(t, params, t_offset=2.0)
    ax = np.random.normal(0, noise_std, len(t))
    ay = np.random.normal(0, noise_std, len(t))
    az = g + a_true + np.random.normal(0, noise_std, len(t))
    pre_ax = np.random.normal(0, noise_std, 100)
    pre_ay = np.random.normal(0, noise_std, 100)
    pre_az = g + np.random.normal(0, noise_std, 100)

    r = estimate_height_accel_only(t, ax, ay, az,
                                     pre_ax=pre_ax, pre_ay=pre_ay, pre_az=pre_az,
                                     building_type='residential', prior_weight=0.3)
    err = abs(r['height'] - 9.0)
    errors.append(err)
    ci_widths.append(r['distance_ci_90'])
    if err <= r['distance_ci_90']:
        covered += 1

errors = np.array(errors)
ci_widths = np.array(ci_widths)

print(f"Coverage: {covered}/{n_trials} = {100*covered/n_trials:.1f}%")
print(f"Mean error: {np.mean(errors):.3f}m")
print(f"Median error: {np.median(errors):.3f}m")
print(f"Mean CI: +/-{np.mean(ci_widths):.3f}m")
print(f"Median CI: +/-{np.median(ci_widths):.3f}m")
print(f"90th percentile error: {np.percentile(errors, 90):.3f}m")
print(f"95th percentile error: {np.percentile(errors, 95):.3f}m")
print(f"Max error: {np.max(errors):.3f}m")
print(f"\nError distribution:")
for pct in [50, 75, 90, 95, 99]:
    print(f"  {pct}th percentile: {np.percentile(errors, pct):.4f}m")

# Need to scale CI so that 90% of errors are covered
target_ci = np.percentile(errors, 90)
print(f"\nRequired CI for 90% coverage: +/-{target_ci:.4f}m")
print(f"Current mean CI: +/-{np.mean(ci_widths):.4f}m")
print(f"Scale factor needed: {target_ci / np.mean(ci_widths):.2f}x")

# Test with different distances to check generalization
print("\n=== Multi-distance test ===")
for true_d in [3.0, 6.0, 9.0, 15.0, 30.0, 50.0]:
    np.random.seed(42)
    p = SCurveParams(j_max=1.8, a_max=1.0, v_max=1.2, distance=true_d, direction=1)
    dur = max(20.0, true_d / 0.8)
    t = np.arange(0, dur, 0.01)
    a_true, _, _ = generate_profile(t, p, t_offset=2.0)
    ax = np.random.normal(0, noise_std, len(t))
    ay = np.random.normal(0, noise_std, len(t))
    az = g + a_true + np.random.normal(0, noise_std, len(t))
    pre_ax = np.random.normal(0, noise_std, 100)
    pre_ay = np.random.normal(0, noise_std, 100)
    pre_az = g + np.random.normal(0, noise_std, 100)

    r = estimate_height_accel_only(t, ax, ay, az,
                                     pre_ax=pre_ax, pre_ay=pre_ay, pre_az=pre_az,
                                     building_type='residential', prior_weight=0.3)
    err = abs(r['height'] - true_d)
    print(f"  d={true_d:5.1f}m -> est={r['height']:+7.2f}m, err={err:.3f}m, "
          f"CI=+/-{r['distance_ci_90']:.3f}m, Q={r['quality_score']:.1f}, "
          f"rej={r['rejected']}")
