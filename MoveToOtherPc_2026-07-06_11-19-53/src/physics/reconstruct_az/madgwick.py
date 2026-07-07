"""Madgwick gradient-descent orientation filter.

Corrects the gyro-integrated orientation by one normalized gradient-descent
step toward the accelerometer's gravity direction each sample. Cheap, robust,
and the de-facto standard for low-cost IMUs. The single gain ``beta``
represents the gyroscope's error magnitude; here it is additionally scaled by
the accel-trust gate so the accelerometer correction backs off during the
ride's acceleration pulses (when the accelerometer is *not* measuring pure
gravity).

Reference: Madgwick (2010), *An efficient orientation filter for inertial and
inertial/magnetic sensor arrays*, x-io technical report (published 2011, IEEE
ICORR). Equations follow the IMU-mode objective f_g / Jacobian J_g. See
``references/madgwick2010_report.pdf``.
"""

from __future__ import annotations

import numpy as np

from . import _quaternion as quat
from .base import AZReconstructor, SensorChannel


class MadgwickReconstructor(AZReconstructor):
    """Madgwick filter (IMU mode: accelerometer + gyroscope).

    Args:
        beta: gradient-descent gain (gyro error magnitude). Larger → trusts
            the accelerometer more / converges faster but noisier.
    """

    def __init__(self, beta: float = 0.1, **kwargs) -> None:
        super().__init__(**kwargs)
        self.beta = float(beta)

    def dependencies(self) -> set[SensorChannel]:
        return {SensorChannel.ACC, SensorChannel.GYR}

    def _estimate_orientation(self, acc, gyr, dt, q0, g_mag):
        n = len(acc)
        out = np.empty((n, 4))
        acc_norm = np.linalg.norm(acc, axis=1)
        trust = self._accel_trust(acc_norm, g_mag)

        q = q0.copy()
        for i in range(n):
            wx, wy, wz = gyr[i]
            # Rate of change of orientation from the gyroscope: q̇ = ½ q ⊗ ω.
            q_dot = 0.5 * quat.mul(q, np.array([0.0, wx, wy, wz]))

            if acc_norm[i] > 1e-6:
                qw, qx, qy, qz = q
                ax, ay, az = acc[i] / acc_norm[i]
                # Objective f_g = (Rᵀẑ − â) and its Jacobian (IMU mode).
                f = np.array([
                    2.0 * (qx * qz - qw * qy) - ax,
                    2.0 * (qw * qx + qy * qz) - ay,
                    2.0 * (0.5 - qx * qx - qy * qy) - az,
                ])
                J = np.array([
                    [-2.0 * qy, 2.0 * qz, -2.0 * qw, 2.0 * qx],
                    [2.0 * qx, 2.0 * qw, 2.0 * qz, 2.0 * qy],
                    [0.0, -4.0 * qx, -4.0 * qy, 0.0],
                ])
                grad = J.T @ f
                gn = np.linalg.norm(grad)
                if gn > 1e-12:
                    beta = self.beta * trust[i]
                    q_dot = q_dot - beta * (grad / gn)

            q = quat.normalize(q + q_dot * dt[i])
            out[i] = q
        return out
