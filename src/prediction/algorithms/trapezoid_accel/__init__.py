"""Trapezoid pulse-pair height-diff estimator (accelerometer-only).

This replaces the 7-phase kinematic S-curve estimator. Instead of fitting
three kinematic bounds (``j_max``, ``a_max``, ``v_max``) in the velocity
domain with non-linear least squares, we fit **one** shared-shape
trapezoid-pulse pair in the acceleration domain using a matched-filter
grid search over (W, f). This exploits two project-relevant facts:

* The raw acceleration SNR is much higher in our recent dataset than in
  the earlier field data the original S-curve fitter was tuned on, so
  the velocity-domain noise-averaging trick is no longer needed.

* The up/down acceleration lobes of a real elevator must be symmetric
  in amplitude and shape (``∫a_up = −∫a_decel``), so enforcing a shared
  (W, f, |A|) across the pair removes 3 degrees of freedom that were
  otherwise just fitting noise.

See :mod:`src.prediction.algorithms.common.pulse_pair` for the
mathematical content and :class:`~.estimator.TrapezoidAccelEstimator`
for the public interface.
"""

from .config import TrapezoidAccelConfig
from .estimator import TrapezoidAccelEstimator

__all__ = ["TrapezoidAccelConfig", "TrapezoidAccelEstimator"]
