"""7-step S-curve kinematic height-diff estimator (accelerometer-only).

Strategy:
  1. Low-pass filter vertical acceleration (~3 Hz) and integrate to
     velocity — the velocity-domain S-curve is a smooth bump with
     SNR ~10x the raw acceleration.
  2. Grid-search distance + t_offset with prior-mean kinematics.
  3. Trust-region NLS refinement in the velocity domain with a
     Bayesian prior regularization term (ISO 18738 / CIBSE D limits).
  4. Fisher-information CRB on the distance parameter → σ_d.
  5. Quality score from fit residual, parameter plausibility, CI
     density, convergence.
  6. Theoretical σ (bootstrapped from CRB + safety factor that the
     velocity-domain ZUPT cross-check gives us).
  7. Conformal calibrator on top (split conformal, α = 0.10).

This is the accelerometer-only variant (Algorithm A in the main-branch
reference). The orientation-aware variant (Algorithm B) is deferred to
a follow-up PR where we add quaternion integration to the Predictor
contract.
"""

from .config import ScurveAccelConfig
from .estimator import ScurveAccelEstimator
from . import scurve_model

__all__ = ["ScurveAccelConfig", "ScurveAccelEstimator", "scurve_model"]
