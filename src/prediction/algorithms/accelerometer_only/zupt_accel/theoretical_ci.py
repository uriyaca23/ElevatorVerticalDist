"""Per-segment theoretical σ of the ZUPT position error.

Three error sources compose additively (in variance) into the final
per-segment σ used to feed the conformal calibrator:

1. **White-noise integration error** — the classic ZUPT result,

       Var(x_N) = σ_a² · dt⁴ · N³ / 12
       σ_white  = σ_a · dt² · √(N³ / 12)

   where N is the number of samples in the active-motion window and
   σ_a is the accelerometer white-noise σ at the sampling rate
   (from the sensor datasheet, see :mod:`src.utils.sensor_noise`). The
   ``/12`` factor credits the linear drift correction for knocking
   out one degree of freedom.

2. **Mechanical / handling jitter** — the phone is not a rigid
   inertial platform: hand movement, pocket friction, and skin-
   coupled respiration add a coloured-noise band the datasheet
   doesn't cover. We model this as an additional acceleration σ that
   grows with the ride drift angle (the quality filter's
   ``max_gravity_drift_deg`` feature) and then integrate it the same
   way as the white-noise term. For a perfectly steady phone this
   term vanishes.

3. **Relative scale term** — empirically, the longer the ride, the
   larger the absolute error, roughly because (a) gravity projection
   errors are proportional to the path integral of acceleration, and
   (b) the dominant colored-noise contribution we *haven't* modelled
   is drift-linear.  A crude relative floor
   ``k_rel · |predicted Δh|`` captures this; ``k_rel = 0.05–0.1`` is
   a sensible prior before conformal empirical tightening.

The result is the per-segment σ the conformal layer multiplies by
its (1−α) non-conformity quantile.
"""

from __future__ import annotations

import math


def zupt_position_sigma(
    sigma_a: float,
    n_active: int,
    dt_sec: float,
    mechanical_jitter_m_s2: float = 0.03,
    predicted_abs_dh_m: float = 0.0,
    ride_drift_deg: float = 0.0,
    pre_post_angle_deg: float = 0.0,
    vert_method: str = "projected_pre_post",
    relative_floor: float = 0.05,
    min_sigma_m: float = 0.15,
) -> float:
    """Theoretical σ of the end-point ZUPT position estimate.

    Parameters
    ----------
    sigma_a : float
        Accelerometer white-noise σ (m/s²) at the sampling rate.
    n_active : int
        Samples in the active-motion window. Short windows (N<10)
        fall back to ``min_sigma_m``.
    dt_sec : float
        Sampling interval (s).
    mechanical_jitter_m_s2 : float
        Baseline coloured-noise σ (m/s²). Further scaled by the ride
        drift angle; a steady phone sees the baseline only.
    predicted_abs_dh_m : float
        Absolute predicted Δh (m). Drives the relative-scale term so
        long rides get a wider CI than short rides even when the
        white-noise σ is similar.
    ride_drift_deg : float
        Gravity-drift angle from the quality filter (``0`` = phone
        held perfectly still). Scales the mechanical jitter term
        up to 3×.
    relative_floor : float
        ``k_rel``: the per-meter floor on σ (dimensionless).
    min_sigma_m : float
        Absolute σ floor to stop the conformal multiplier blowing up.
    """
    if n_active < 10 or dt_sec <= 0:
        return float(min_sigma_m)

    # --- Term 1: datasheet white-noise integration ---
    sigma_white = sigma_a * (dt_sec ** 2) * math.sqrt(max(n_active, 1) ** 3 / 12.0)

    # --- Term 2: mechanical jitter integrated under ZUPT ---
    # Scale the baseline σ by (1 + drift_deg / 10) so a wobbly ride gets
    # a larger mechanical term than a steady one. Cap at 3× baseline
    # so a pathological ride doesn't dominate the budget.
    mech_scale = 1.0 + min(max(ride_drift_deg, 0.0) / 10.0, 2.0)
    sigma_mech_a = max(mechanical_jitter_m_s2, 0.0) * mech_scale
    sigma_mech = sigma_mech_a * (dt_sec ** 2) * math.sqrt(max(n_active, 1) ** 3 / 12.0)

    # --- Term 3: relative scale with |predicted Δh| ---
    sigma_rel = max(relative_floor, 0.0) * max(predicted_abs_dh_m, 0.0)

    # --- Term 4: projection-quality term ---
    # The magnitude fallback and one-sided projections lose 1–2 degrees
    # of freedom on the vertical-direction estimate. Empirically, rides
    # where the pre/post gravity vectors rotated a lot or where we had
    # no stable stationary window at all show 5–10× the error of the
    # nominal projected case. We encode that as a multiplicative factor
    # on the relative term and an additive term proportional to
    # |predicted Δh|. A small pre/post rotation (< 10°) contributes
    # negligibly; big rotations ramp quickly.
    proj_penalty = {
        "projected_pre_post": 1.0,
        "projected_pre": 1.2,
        "projected_post": 1.3,
        "magnitude": 3.0,
    }.get(vert_method, 1.5)
    # Pre/post rotation scales the penalty too (0 at 0°, 2× at 25°+).
    rot = max(pre_post_angle_deg, 0.0)
    proj_penalty *= 1.0 + min(rot / 12.5, 2.0)
    sigma_proj = (proj_penalty - 1.0) * max(predicted_abs_dh_m, 0.0) * 0.25

    sigma = math.sqrt(
        sigma_white ** 2 + sigma_mech ** 2 + sigma_rel ** 2 + sigma_proj ** 2
    )
    return float(max(sigma, min_sigma_m))
