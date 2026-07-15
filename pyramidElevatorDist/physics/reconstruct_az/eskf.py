"""Error-State Kalman Filter (ESKF) for attitude + gyro bias.

The principled member of the family. The *nominal* state (orientation
quaternion + gyro bias) is propagated with the gyroscope; a small
**error state** — a 3-vector attitude error plus a 3-vector bias error — is
tracked with a Kalman filter and periodically injected back into the nominal
state and reset. The accelerometer supplies the measurement (its normalized
reading should equal the body-frame gravity direction Rᵀẑ), and its
measurement noise is inflated by the accel-trust gate so the ride's linear
acceleration is discounted. Estimating the gyro bias online is what lets it
hold attitude through long, low-excitation cruise phases.

Reference: Solà (2017), *Quaternion kinematics for the error-state Kalman
filter*, arXiv:1711.02508. See ``references/sola2017_eskf.pdf``.
"""

from __future__ import annotations

import numpy as np

from . import _quaternion as quat
from .base import AZReconstructor, SensorChannel


class ESKFReconstructor(AZReconstructor):
    """6-error-state (attitude + gyro bias) ESKF, IMU mode.

    Args:
        sigma_g: gyroscope white-noise std (rad/s) — attitude process noise.
        sigma_bg: gyro-bias random-walk std (rad/s/√s).
        sigma_acc: accelerometer direction-measurement std (unit vector);
            inflated by 1/trust when |‖a‖−g| is large.
        init_att_std: initial attitude-error std (rad).
        init_bias_std: initial gyro-bias std (rad/s).
    """

    def __init__(
        self,
        sigma_g: float = 0.015,
        sigma_bg: float = 1e-5,
        sigma_acc: float = 0.1,
        init_att_std: float = 0.1,
        init_bias_std: float = 0.01,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.sigma_g = float(sigma_g)
        self.sigma_bg = float(sigma_bg)
        self.sigma_acc = float(sigma_acc)
        self.init_att_std = float(init_att_std)
        self.init_bias_std = float(init_bias_std)

    def dependencies(self) -> set[SensorChannel]:
        return {SensorChannel.ACC, SensorChannel.GYR}

    def _estimate_orientation(self, acc, gyr, dt, q0, g_mag):
        n = len(acc)
        out = np.empty((n, 4))
        acc_norm = np.linalg.norm(acc, axis=1)
        trust = self._accel_trust(acc_norm, g_mag)

        I3 = np.eye(3)
        I6 = np.eye(6)
        q = q0.copy()
        bias = np.zeros(3)
        P = np.zeros((6, 6))
        P[:3, :3] = I3 * self.init_att_std ** 2
        P[3:, 3:] = I3 * self.init_bias_std ** 2

        for i in range(n):
            d = dt[i]
            omega = gyr[i] - bias

            # --- nominal propagation ---
            q = quat.from_omega(q, omega, d)

            # --- error-state propagation: δθ̇ = −[ω]ₓδθ − δb,  δḃ = w ---
            F = I6.copy()
            F[:3, :3] = I3 - quat.skew(omega) * d
            F[:3, 3:] = -I3 * d
            Q = np.zeros((6, 6))
            Q[:3, :3] = I3 * (self.sigma_g ** 2) * d * d
            Q[3:, 3:] = I3 * (self.sigma_bg ** 2) * d
            P = F @ P @ F.T + Q

            # --- accelerometer update (gravity direction) ---
            an = acc_norm[i]
            if an > 1e-6 and trust[i] > 1e-3:
                a_hat = acc[i] / an
                v = quat.rotate(quat.conjugate(q), quat.WORLD_UP)   # predicted Rᵀẑ
                H = np.zeros((3, 6))
                H[:, :3] = quat.skew(v)                              # ∂v/∂δθ
                R = I3 * (self.sigma_acc / trust[i]) ** 2
                S = H @ P @ H.T + R
                K = P @ H.T @ np.linalg.inv(S)
                dx = K @ (a_hat - v)

                # inject error into nominal state, then reset
                dtheta = dx[:3]
                q = quat.normalize(quat.mul(
                    q, quat.normalize(np.array([1.0, *(0.5 * dtheta)]))))
                bias = bias + dx[3:]
                P = (I6 - K @ H) @ P

            out[i] = q
        return out
