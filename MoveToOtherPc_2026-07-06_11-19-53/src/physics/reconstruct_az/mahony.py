"""Mahony nonlinear complementary filter on SO(3).

A complementary filter formulated directly on the rotation group, with a
**PI** correction: the proportional term steers the estimate toward the
accelerometer's gravity direction, and the integral term estimates and
cancels the slowly-varying gyroscope bias. The bias estimate makes it a step
up from the plain complementary filter on real phone gyros (the dataset even
ships per-sample ``RAWGYR`` bias columns that corroborate this).

Reference: Mahony, Hamel, Pflimlin (2008), *Nonlinear Complementary Filters
on the Special Orthogonal Group*, IEEE TAC 53(5):1203–1218. See
``references/REFERENCES.md``.
"""

from __future__ import annotations

import numpy as np

from . import _quaternion as quat
from .base import AZReconstructor, SensorChannel


class MahonyReconstructor(AZReconstructor):
    """Mahony explicit complementary filter (IMU mode).

    Args:
        kp: proportional gain on the gravity-direction error.
        ki: integral gain driving the gyro-bias estimate.
    """

    def __init__(self, kp: float = 1.0, ki: float = 0.3, **kwargs) -> None:
        super().__init__(**kwargs)
        self.kp = float(kp)
        self.ki = float(ki)

    def dependencies(self) -> set[SensorChannel]:
        return {SensorChannel.ACC, SensorChannel.GYR}

    def _estimate_orientation(self, acc, gyr, dt, q0, g_mag):
        n = len(acc)
        out = np.empty((n, 4))
        acc_norm = np.linalg.norm(acc, axis=1)
        trust = self._accel_trust(acc_norm, g_mag)

        q = q0.copy()
        bias = np.zeros(3)
        for i in range(n):
            omega = gyr[i].copy()
            if acc_norm[i] > 1e-6:
                a_hat = acc[i] / acc_norm[i]
                # Estimated gravity direction in the body frame: Rᵀ·ẑ.
                v_hat = quat.rotate(quat.conjugate(q), quat.WORLD_UP)
                e = np.cross(a_hat, v_hat) * trust[i]      # measured × estimated
                bias = bias - self.ki * e * dt[i]          # ḃ = −kI·e
                omega = omega - bias + self.kp * e         # corrected rate
            q = quat.from_omega(q, omega, dt[i])
            out[i] = q
        return out
