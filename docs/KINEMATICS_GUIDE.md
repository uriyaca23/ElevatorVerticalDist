# Kinematics-Based Elevator Height Estimation Guide

This guide describes the **Phase 3** kinematics-based estimator, which represents the current state-of-the-art in this repository for vertical distance estimation from smartphone accelerometer data.

## 1. Core Theory: Velocity-Domain Fitting

Unlike traditional methods (Direct Integration, ZUPT, Kalman Filtering) that treat the elevator signal as a generic acceleration pulse, our algorithm exploits the known **7-step S-curve** kinematic profile of modern elevators.

### Why Velocity Domain?
The fundamental breakthrough in this phase was moving the optimization from the acceleration domain to the **velocity domain**:
1. **SNR Boost**: Integration from acceleration to velocity acts as a first-order low-pass filter, suppressing hand tremor and high-frequency noise while preserving the elevator signal. SNR typically improves from ~2.0 to >10.0.
2. **Distinctive Shape**: The S-curve velocity profile is a smooth, bell-shaped bump, making it far more distinctive and robust to noise than the sharp-edged acceleration profile.
3. **Better Optimization**: The velocity-domain cost function is much smoother, significantly reducing the problem of the optimizer getting stuck in local minima (a primary failure mode of acceleration-domain fitting).

---

## 2. Usage Instructions

### Python API
The core of the algorithm is contained in `src/algorithms/kinematics_estimator.py`.

```python
from src.algorithms.kinematics_estimator import KinematicsEstimator

# Initialize estimator
estimator = KinematicsEstimator(fs=100)

# Process a segment (requires raw 3-axis accelerometer)
# acc_data: numpy array [N, 3] or dict with 'acc_x', 'acc_y', 'acc_z'
# quaternions: (Optional) numpy array [N, 4] for Algorithm B (projection-based)
result = estimator.estimate_height(acc_data, quaternions=None)

if not result['rejected']:
    print(f"Estimated Height: {result['estimated_height']:.2f}m")
    print(f"90% Confidence Interval: ±{result['distance_ci_90']:.2f}m")
else:
    print(f"Ride rejected: {result['reject_reason']}")
```

### Running Evaluation
To reproduce the research results on the ADVIO and Bar-Ilan datasets:

```bash
python scripts/run_kinematics_evaluation.py
```

This script will:
1. Run both **Algorithm A** (Accelerometer-only) and **Algorithm B** (Accel + Orientation).
2. Generate 10+ research-quality figures in `docs/figures_kinematics/`.
3. Save detailed JSON results to `evaluation_output/kinematics/results.json`.

---

## 3. Algorithm Parameters & Priors

The estimator uses **Bayesian Prior Regularization** to guide the fitting toward physically plausible elevator parameters (Max Jerk, Max Acceleration, Max Velocity). These priors are derived from industrial standards (ISO 18738-1) and building codes.

| Parameter | Residential (Typical) | High-Speed (Typical) |
|-----------|-----------------------|----------------------|
| $v_{max}$ | 0.6 - 1.0 m/s         | 1.6 - 6.0 m/s        |
| $a_{max}$ | 0.8 - 1.2 m/s²        | 1.0 - 1.6 m/s²       |
| $j_{max}$ | 1.0 - 2.0 m/s³        | 2.0 - 4.0 m/s³       |

### Quality Scoring
Estimates are assigned a multi-factor quality score (0 = Excellent, 10 = Failed). Factors include:
- **Fit Quality**: Chi-squared residual match in the velocity domain.
- **CI Density**: Width of the confidence interval relative to the travel distance.
- **Parameter Plausibility**: Log-likelihood of the fitted parameters under the prior distributions.
- **Convergence**: Whether the NLS optimizer (Trust Region Reflective) successfully converged.

---

## 4. Performance Summary (Latest)

Verified on the **Bar-Ilan Dataset** (33 rides, 3m-57.4m ranges):

| Metric | Algorithm A (Magnitude) | Algorithm B (3D Projected) |
|--------|--------------------------|----------------------------|
| **Acceptance Rate** | 70% (28/40*) | 62% (18/29) |
| **Median Error** | 0.49 m | **0.15 m** |
| **MAE** | 2.28 m | **1.01 m** |
| **CI Coverage** | 82.1% (target: 90%) | **88.9%** (target: 90%) |

*\*Includes ADVIO segments for combined statistics.*

For more details, see the full research report: `docs/Kinematics_Estimation_Report.docx`.
