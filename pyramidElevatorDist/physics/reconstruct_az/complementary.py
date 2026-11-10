"""Complementary filter — the simplest accel+gyro orientation fusion.

Integrate the gyroscope for the short-term orientation change, then nudge
the estimate so the measured gravity direction lines up with world-up, with
the nudge weighted by how much the accelerometer currently looks like pure
gravity. It is the direct upgrade to the project's current *fixed* gravity
projection: instead of one stationary gravity vector for the whole ride, the
gravity direction is tracked sample-by-sample.

Reference: Euston, Coote, Mahony, Kim, Hamel (2008), *A complementary filter
for attitude estimation of a fixed-wing UAV*, IROS; the complementary idea
goes back to Higgins (1975). See ``references/REFERENCES.md``.
"""

from __future__ import annotations

import numpy as np

from . import _quaternion as quat
from .base import AZReconstructor, SensorChannel


class ComplementaryReconstructor(AZReconstructor):
    """Gyro integration high-pass + accelerometer gravity low-pass.

    Args:
        gain: accelerometer correction strength per sample (fraction of the
            tilt error applied each step). Small values trust the gyro more;
            the effective gain is further scaled by the accel-trust gate so
            it backs off during the ride's acceleration pulses.
    """

    def __init__(self, gain: float = 0.02, **kwargs) -> None:
        super().__init__(**kwargs)
        self.gain = float(gain)

    def dependencies(self) -> set[SensorChannel]:
        return {SensorChannel.ACC, SensorChannel.GYR}

    def _estimate_orientation(self, acc, gyr, dt, q0, g_mag):
        n = len(acc)
        out = np.empty((n, 4))
        acc_norm = np.linalg.norm(acc, axis=1)
        trust = self._accel_trust(acc_norm, g_mag)

        q = q0.copy()
        for i in range(n):
            q = quat.from_omega(q, gyr[i], dt[i])          # gyro prediction
            if acc_norm[i] > 1e-6:
                a_hat = acc[i] / acc_norm[i]
                a_world = quat.rotate(q, a_hat)            # ≈ world-up if q is right
                dq = quat.from_two_vectors(a_world, quat.WORLD_UP)
                gain = self.gain * trust[i]
                q = quat.normalize(quat.mul(quat.slerp_identity(dq, gain), q))
            out[i] = q
        return out
