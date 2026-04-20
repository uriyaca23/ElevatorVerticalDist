"""ZUPT-based accelerometer-only height-difference estimator.

Strategy:
  1. Pick the best vertical-accel projection (pre/post gravity vector
     when both are stable; magnitude fallback otherwise).
  2. Low-pass filter, locate the active-motion window, zero-force
     velocity outside it, linearly detrend velocity inside it, double
     integrate to get displacement.
  3. Quality filter on gravity drift, impact peaks, noise, window
     length.
  4. Theoretical σ from integrated sensor noise: σ_pos = σ_a · dt² ·
     √(N³ / 12).
  5. Conformal multiplier on top of σ_pos.

See :mod:`.estimator` for the public :class:`ZuptAccelEstimator` class.
"""

from src.prediction.algorithms.configTypes import ZuptAccelConfig
from .estimator import ZuptAccelEstimator

__all__ = ["ZuptAccelConfig", "ZuptAccelEstimator"]
