"""Valenti AQUA — Algebraic Quaternion Algorithm with adaptive gain.

Predicts orientation from the gyroscope, then applies an *algebraic* delta
quaternion that rotates the gyro-predicted gravity onto the measured gravity,
blended in by spherical interpolation (LERP near identity, SLERP otherwise).
Its hallmark is the **adaptive gain**: a two-threshold ramp that linearly
fades the accelerometer correction to zero once the external acceleration
(|‖a‖ − g|) gets large — precisely the elevator accel/decel pulses — so the
ride's own acceleration is never mistaken for a tilt.

Reference: Valenti, Dryanovski, Xiao (2015), *Keeping a Good Attitude: A
Quaternion-Based Orientation Filter for IMUs and MARGs*, Sensors
15(8):19302–19330. See ``references/valenti2015_aqua.pdf``.
"""

from __future__ import annotations

import numpy as np

from . import _quaternion as quat
from .base import AZReconstructor, SensorChannel


class ValentiReconstructor(AZReconstructor):
    """AQUA filter (IMU mode) with Valenti's adaptive-gain external-accel reject.

    Args:
        gain: base interpolation gain α for the accelerometer correction.
        t1: error (fraction of g) below which the accel is fully trusted.
        t2: error (fraction of g) above which the accel correction is dropped
            entirely; a linear ramp connects ``t1``→``t2``.
    """

    def __init__(
        self, gain: float = 0.02, t1: float = 0.1, t2: float = 0.2, **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.gain = float(gain)
        self.t1 = float(t1)
        self.t2 = float(t2)

    def dependencies(self) -> set[SensorChannel]:
        return {SensorChannel.ACC, SensorChannel.GYR}

    def _adaptive_factor(self, err: float) -> float:
        """Valenti's gain factor: 1 below ``t1``, 0 above ``t2``, linear between."""
        if err <= self.t1:
            return 1.0
        if err >= self.t2:
            return 0.0
        return (self.t2 - err) / (self.t2 - self.t1)

    def _estimate_orientation(self, acc, gyr, dt, q0, g_mag):
        n = len(acc)
        out = np.empty((n, 4))
        acc_norm = np.linalg.norm(acc, axis=1)

        q = q0.copy()
        for i in range(n):
            q = quat.from_omega(q, gyr[i], dt[i])          # gyro prediction
            an = acc_norm[i]
            if an > 1e-6:
                err = abs(an - g_mag) / max(g_mag, 1e-6)
                factor = self._adaptive_factor(err)
                if factor > 0.0:
                    a_world = quat.rotate(q, acc[i] / an)
                    dq = quat.from_two_vectors(a_world, quat.WORLD_UP)
                    q = quat.normalize(
                        quat.mul(quat.slerp_identity(dq, self.gain * factor), q)
                    )
            out[i] = q
        return out
