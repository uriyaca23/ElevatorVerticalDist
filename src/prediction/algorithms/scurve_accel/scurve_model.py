"""7-step S-curve kinematic model.

Port of ``src/algorithms/scurve_model.py`` from the ``main`` branch.
Everything here is pure math on arrays — no I/O, no config — so it
can be shared with the main-branch reference tests without extra work.

The 7 phases (see §4.4 of the report):
  1: j = +j_max   (jerk-in)
  2: j = 0        (const +a_max)
  3: j = -j_max   (jerk-out)
  4: j = 0, a = 0 (cruise at v_max)
  5: j = -j_max   (jerk-in, decel)
  6: j = 0        (const -a_max)
  7: j = +j_max   (jerk-out)

For short rides, phases collapse: no cruise (t4=0), no const-acc
(t2=t6=0, triangular), or both.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import numpy as np


# ---------------------------------------------------------------------------
# Core dataclasses
# ---------------------------------------------------------------------------

@dataclass
class SCurveParams:
    j_max: float
    a_max: float
    v_max: float
    distance: float
    direction: int


@dataclass
class SCurveProfile:
    t1: float; t2: float; t3: float; t4: float
    t5: float; t6: float; t7: float
    a_peak: float; v_peak: float
    total_time: float
    total_distance: float
    profile_type: str

    @property
    def phase_boundaries(self) -> list[float]:
        T = [0.0]
        for t in [self.t1, self.t2, self.t3, self.t4, self.t5, self.t6, self.t7]:
            T.append(T[-1] + t)
        return T


# ---------------------------------------------------------------------------
# Phase durations
# ---------------------------------------------------------------------------

def compute_phase_durations(
    j_max: float, a_max: float, v_max: float, distance: float,
) -> SCurveProfile:
    """Solve for phase durations given kinematic bounds and a target
    distance. Handles all phase-collapse cases.
    """
    from scipy.optimize import brentq

    t_j = a_max / j_max                     # time to hit a_max under j_max
    v_jerk = a_max * t_j                    # velocity gained in phases 1+3 only

    # --- Case A: a_max is never reached (triangular / no-const-acc) ---
    if v_jerk >= v_max:
        t1_from_v = np.sqrt(v_max / j_max)
        t1_from_d = (distance / (2.0 * j_max)) ** (1.0 / 3.0)
        t1 = min(t1_from_v, t1_from_d)
        a_peak = j_max * t1
        v_peak = j_max * t1 ** 2

        d_total_no_cruise = 2.0 * j_max * t1 ** 3
        if d_total_no_cruise >= distance:
            t1 = (distance / (2.0 * j_max)) ** (1.0 / 3.0)
            a_peak = j_max * t1; v_peak = j_max * t1 ** 2
            return SCurveProfile(
                t1=t1, t2=0.0, t3=t1, t4=0.0, t5=t1, t6=0.0, t7=t1,
                a_peak=a_peak, v_peak=v_peak,
                total_time=4 * t1, total_distance=distance,
                profile_type="triangular",
            )
        d_cruise = distance - d_total_no_cruise
        t4 = d_cruise / max(v_peak, 1e-9)
        return SCurveProfile(
            t1=t1, t2=0.0, t3=t1, t4=t4, t5=t1, t6=0.0, t7=t1,
            a_peak=a_peak, v_peak=v_peak,
            total_time=4 * t1 + t4, total_distance=distance,
            profile_type="no_const_acc",
        )

    # --- Case B: a_max is reached; check v_max ---
    t2_for_vmax = (v_max - v_jerk) / a_max

    v1 = 0.5 * j_max * t_j ** 2
    d1 = (1.0 / 6.0) * j_max * t_j ** 3

    def _accel_distance(t2_val: float) -> float:
        v2 = v1 + a_max * t2_val
        d2 = v1 * t2_val + 0.5 * a_max * t2_val ** 2
        d3 = v2 * t_j + 0.5 * a_max * t_j ** 2 - (1.0 / 6.0) * j_max * t_j ** 3
        return d1 + d2 + d3

    d_accel_full = _accel_distance(t2_for_vmax)
    d_min_full = 2.0 * d_accel_full

    if d_min_full > distance:
        # v_max never reached; solve for t2
        if 2.0 * _accel_distance(0.0) >= distance:
            t1 = (distance / (2.0 * j_max)) ** (1.0 / 3.0)
            a_peak = j_max * t1; v_peak = j_max * t1 ** 2
            return SCurveProfile(
                t1=t1, t2=0.0, t3=t1, t4=0.0, t5=t1, t6=0.0, t7=t1,
                a_peak=a_peak, v_peak=v_peak,
                total_time=4 * t1, total_distance=distance,
                profile_type="triangular",
            )
        try:
            t2_sol = brentq(lambda x: 2.0 * _accel_distance(x) - distance,
                            0.0, t2_for_vmax)
        except Exception:
            t2_sol = 0.0
        v_achieved = v1 + a_max * t2_sol + v1
        return SCurveProfile(
            t1=t_j, t2=t2_sol, t3=t_j,
            t4=0.0, t5=t_j, t6=t2_sol, t7=t_j,
            a_peak=a_max, v_peak=v_achieved,
            total_time=4 * t_j + 2 * t2_sol, total_distance=distance,
            profile_type="no_cruise",
        )

    d_cruise = distance - d_min_full
    t4 = d_cruise / v_max
    return SCurveProfile(
        t1=t_j, t2=t2_for_vmax, t3=t_j,
        t4=t4, t5=t_j, t6=t2_for_vmax, t7=t_j,
        a_peak=a_max, v_peak=v_max,
        total_time=4 * t_j + 2 * t2_for_vmax + t4, total_distance=distance,
        profile_type="full",
    )


# ---------------------------------------------------------------------------
# Profile generation at arbitrary timestamps
# ---------------------------------------------------------------------------

def _phase_states(profile: SCurveProfile, j: float) -> list[tuple[float, float, float]]:
    """State (a, v, s) at the start of each of the 7 phases, assuming
    positive direction. Direction sign is applied by the caller.
    """
    jerk_seq = [j, 0.0, -j, 0.0, -j, 0.0, j]
    durations = [profile.t1, profile.t2, profile.t3, profile.t4,
                 profile.t5, profile.t6, profile.t7]
    states = [(0.0, 0.0, 0.0)]
    for phase_idx in range(7):
        a0, v0, s0 = states[-1]
        dt = durations[phase_idx]
        jk = jerk_seq[phase_idx]
        a1 = a0 + jk * dt
        v1 = v0 + a0 * dt + 0.5 * jk * dt ** 2
        s1 = s0 + v0 * dt + 0.5 * a0 * dt ** 2 + (1.0 / 6.0) * jk * dt ** 3
        states.append((a1, v1, s1))
    return states


def generate_profile(
    t: np.ndarray, params: SCurveParams, t_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Evaluate (a, v, s) at each timestamp in ``t``."""
    profile = compute_phase_durations(
        params.j_max, params.a_max, params.v_max, params.distance,
    )
    boundaries = profile.phase_boundaries
    sign = float(params.direction)
    jerk_seq = [params.j_max, 0.0, -params.j_max, 0.0,
                -params.j_max, 0.0, params.j_max]
    states = _phase_states(profile, params.j_max)

    n = len(t)
    a_out = np.zeros(n); v_out = np.zeros(n); s_out = np.zeros(n)

    for i in range(n):
        tau = t[i] - t_offset
        if tau < 0:
            continue
        if tau >= profile.total_time:
            s_out[i] = sign * profile.total_distance
            continue

        phase = 0
        for p in range(7):
            if tau < boundaries[p + 1]:
                phase = p
                break

        dt_in = tau - boundaries[phase]
        a0, v0, s0 = states[phase]
        jk = jerk_seq[phase]

        a_out[i] = sign * (a0 + jk * dt_in)
        v_out[i] = sign * (v0 + a0 * dt_in + 0.5 * jk * dt_in ** 2)
        s_out[i] = sign * (
            s0 + v0 * dt_in + 0.5 * a0 * dt_in ** 2 + (1.0 / 6.0) * jk * dt_in ** 3
        )

    return a_out, v_out, s_out


def generate_profile_vectorized(
    t: np.ndarray, j_max: float, a_max: float, v_max: float,
    distance: float, direction: int = 1, t_offset: float = 0.0,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Same as :func:`generate_profile` but with NaN/Inf guards — safe
    to call from inside an optimizer's inner loop.
    """
    if not all(np.isfinite([j_max, a_max, v_max, distance, t_offset])):
        n = len(t); return np.zeros(n), np.zeros(n), np.zeros(n)
    if j_max <= 0 or a_max <= 0 or v_max <= 0 or distance <= 0:
        n = len(t); return np.zeros(n), np.zeros(n), np.zeros(n)
    try:
        params = SCurveParams(j_max=j_max, a_max=a_max, v_max=v_max,
                              distance=distance, direction=direction)
        a_out, v_out, s_out = generate_profile(t, params, t_offset)
        if np.any(np.isnan(a_out)) or np.any(np.isnan(s_out)):
            n = len(t); return np.zeros(n), np.zeros(n), np.zeros(n)
        return a_out, v_out, s_out
    except Exception:
        n = len(t); return np.zeros(n), np.zeros(n), np.zeros(n)


# ---------------------------------------------------------------------------
# Bayesian priors — see §5 of the main-branch report
# ---------------------------------------------------------------------------

PRIOR_PARAMS = {
    "residential": {
        "j_max": {"mean": 1.5, "std": 0.5, "min": 0.5, "max": 4.0},
        "a_max": {"mean": 1.0, "std": 0.3, "min": 0.3, "max": 2.0},
        "v_max": {"mean": 1.0, "std": 0.4, "min": 0.15, "max": 2.5},
    },
    "commercial": {
        "j_max": {"mean": 2.0, "std": 0.6, "min": 0.8, "max": 5.0},
        "a_max": {"mean": 1.2, "std": 0.3, "min": 0.5, "max": 2.5},
        "v_max": {"mean": 2.5, "std": 1.5, "min": 0.5, "max": 8.0},
    },
    "generic": {
        "j_max": {"mean": 1.8, "std": 0.7, "min": 0.5, "max": 5.0},
        "a_max": {"mean": 1.1, "std": 0.3, "min": 0.3, "max": 2.5},
        "v_max": {"mean": 1.5, "std": 1.0, "min": 0.15, "max": 8.0},
    },
}

FLOOR_HEIGHT_PRIOR = {
    "residential": {"mean": 3.0, "std": 0.3, "min": 2.5, "max": 4.0},
    "commercial":  {"mean": 4.0, "std": 0.4, "min": 3.0, "max": 5.5},
    "ground_floor": {"mean": 5.0, "std": 0.8, "min": 3.5, "max": 8.0},
    "generic":     {"mean": 3.2, "std": 0.6, "min": 2.5, "max": 6.0},
}


def compute_prior_log_probability(
    j_max: float, a_max: float, v_max: float, distance: float,
    building_type: str = "generic",
) -> float:
    p = PRIOR_PARAMS.get(building_type, PRIOR_PARAMS["generic"])
    fp = FLOOR_HEIGHT_PRIOR.get(building_type, FLOOR_HEIGHT_PRIOR["generic"])

    log_p = 0.0
    for val, key in [(j_max, "j_max"), (a_max, "a_max"), (v_max, "v_max")]:
        q = p[key]
        if val < q["min"] or val > q["max"]:
            return -np.inf
        log_p -= 0.5 * ((val - q["mean"]) / q["std"]) ** 2

    # Distance prior — mixture of N(k·h, σ) over floor counts
    max_floors = max(1, int(distance / 2.0) + 3)
    probs = []
    for k in range(1, min(max_floors + 1, 50)):
        mu = k * fp["mean"]
        sd = np.sqrt((k * fp["std"]) ** 2 + 0.5 ** 2)
        probs.append(np.exp(-0.5 * ((distance - mu) / sd) ** 2) / sd)
    if sum(probs) > 0:
        log_p += np.log(sum(probs) + 1e-30)
    return log_p
