"""Minimal Hamilton-quaternion helpers for the orientation filters.

Private to :mod:`pyramidElevatorDist.physics.reconstruct_az`. Quaternions are stored as
``[w, x, y, z]`` numpy arrays and follow the **Hamilton** convention. A
quaternion ``q`` here is the *body-to-world* orientation: rotating a
body-frame vector by ``q`` expresses it in the world (Earth) frame,

    v_world = q ⊗ (0, v_body) ⊗ q*.

That single convention is shared by every filter in this package and by
:meth:`AZReconstructor.reconstruct`'s gravity-removal step, so signs stay
consistent end-to-end (the synthetic test in ``tests/`` pins it down).

Only the operations the filters actually need are implemented — no full
quaternion algebra library. For heavier use prefer
``scipy.spatial.transform.Rotation``; these stay dependency-free and
vectorised for the per-sample hot loop.
"""

from __future__ import annotations

import numpy as np

WORLD_UP = np.array([0.0, 0.0, 1.0])  # Earth +z; specific force at rest points here


def normalize(q: np.ndarray) -> np.ndarray:
    """Return ``q`` scaled to unit norm (identity if it is degenerate)."""
    n = float(np.linalg.norm(q))
    if n < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    return q / n


def mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a ⊗ b`` of two ``[w,x,y,z]`` quaternions."""
    aw, ax, ay, az = a
    bw, bx, by, bz = b
    return np.array([
        aw * bw - ax * bx - ay * by - az * bz,
        aw * bx + ax * bw + ay * bz - az * by,
        aw * by - ax * bz + ay * bw + az * bx,
        aw * bz + ax * by - ay * bx + az * bw,
    ])


def conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (inverse, for unit quaternions): negate the vector part."""
    return np.array([q[0], -q[1], -q[2], -q[3]])


def rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate a single 3-vector ``v`` by unit quaternion ``q`` (body→world)."""
    qv = q[1:4]
    t = 2.0 * np.cross(qv, v)
    return v + q[0] * t + np.cross(qv, t)


def rotate_batch(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Vectorised :func:`rotate`: ``q`` ``(N,4)`` and ``v`` ``(N,3)`` → ``(N,3)``.

    Uses the standard ``v' = v + 2 w (q_v × v) + 2 q_v × (q_v × v)`` form,
    valid for unit quaternions, so the final body→world projection of a
    whole ride is one numpy expression rather than a Python loop.
    """
    qw = q[:, 0:1]
    qv = q[:, 1:4]
    t = 2.0 * np.cross(qv, v)
    return v + qw * t + np.cross(qv, t)


def from_omega(q: np.ndarray, omega: np.ndarray, dt: float) -> np.ndarray:
    """First-order gyro integration: advance ``q`` by body rate ``omega`` (rad/s).

    ``q̇ = ½ q ⊗ (0, ω)``; returns ``normalize(q + q̇·dt)``. First order is
    the convention used by the complementary / Mahony / Madgwick filters;
    over a 50–100 Hz step the truncation error is negligible next to sensor
    noise.
    """
    omega_q = np.array([0.0, omega[0], omega[1], omega[2]])
    q_dot = 0.5 * mul(q, omega_q)
    return normalize(q + q_dot * dt)


def from_two_vectors(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Shortest-arc quaternion rotating unit-ish vector ``a`` onto ``b``.

    Used to seed orientation from the stationary gravity vector and as the
    accelerometer correction in the complementary / Valenti filters.
    Handles the parallel and anti-parallel degenerate cases.
    """
    a = a / (np.linalg.norm(a) + 1e-12)
    b = b / (np.linalg.norm(b) + 1e-12)
    d = float(np.dot(a, b))
    if d > 0.999999:                      # already aligned
        return np.array([1.0, 0.0, 0.0, 0.0])
    if d < -0.999999:                     # opposite: 180° about any ⟂ axis
        axis = np.cross(a, np.array([1.0, 0.0, 0.0]))
        if np.linalg.norm(axis) < 1e-6:
            axis = np.cross(a, np.array([0.0, 1.0, 0.0]))
        axis = axis / (np.linalg.norm(axis) + 1e-12)
        return np.array([0.0, axis[0], axis[1], axis[2]])
    axis = np.cross(a, b)
    return normalize(np.array([1.0 + d, axis[0], axis[1], axis[2]]))


def slerp_identity(dq: np.ndarray, gain: float) -> np.ndarray:
    """Interpolate from the identity quaternion toward ``dq`` by ``gain``.

    ``gain=0`` → identity (no correction), ``gain=1`` → full ``dq``. Uses
    LERP near identity (small angle, cheap and stable) and SLERP otherwise —
    exactly the adaptive-gain interpolation Valenti's AQUA filter applies to
    its accelerometer correction.
    """
    dq = normalize(dq)
    w = abs(float(dq[0]))
    if w > 0.9995:                        # tiny rotation → LERP is accurate
        out = (1.0 - gain) * np.array([1.0, 0.0, 0.0, 0.0]) + gain * dq
        return normalize(out)
    angle = np.arccos(np.clip(w, -1.0, 1.0))
    s = np.sin(angle)
    c0 = np.sin((1.0 - gain) * angle) / s
    c1 = np.sin(gain * angle) / s
    out = c0 * np.array([1.0, 0.0, 0.0, 0.0]) + c1 * dq
    return normalize(out)


def skew(v: np.ndarray) -> np.ndarray:
    """3×3 skew-symmetric matrix ``[v]ₓ`` such that ``[v]ₓ u = v × u``."""
    x, y, z = v
    return np.array([
        [0.0, -z, y],
        [z, 0.0, -x],
        [-y, x, 0.0],
    ])
