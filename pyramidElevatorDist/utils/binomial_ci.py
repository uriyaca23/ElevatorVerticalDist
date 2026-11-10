"""Binomial proportion confidence intervals for Bernoulli trial results.

Every rate this repo reports over a finite set of segments — the
inclusion rate of a CI, an accept rate, a detection rate — is a
Bernoulli proportion estimated from ``n`` trials, so it carries
sampling error of its own. A bin holding 91 segments and one holding
203 support very different claims even when both read 0.86.

Two constructions are provided:

* **Wald** — ``p̂ ± z · sqrt(p̂(1-p̂)/n)``. The textbook interval;
  symmetric, trivial to read, and unreliable for small ``n`` or for
  ``p̂`` near 0/1, where it runs past the [0, 1] boundary and its real
  coverage collapses well below nominal.
* **Wilson score** — the interval obtained by inverting the score
  test. It is pulled toward 1/2, never leaves [0, 1], and holds its
  nominal coverage far better at small ``n``. Prefer it for anything
  reported.

Inverting the score test is also what gives the *acceptance region*
for a hypothesised ``p₀``: the observed rates consistent with a true
probability ``p₀`` at ``n`` trials are ``p₀ ± z · sqrt(p₀(1-p₀)/n)``,
which uses the null SE rather than the estimated one. It is the mirror
image of the Wilson interval, and it is the one to reach for when the
uncertainty belongs on a *hypothesis* rather than on a measurement. The
inclusion-by-bin figures put it on the measurement: each bar carries its
own Wilson interval, and the 90 % target is a bare line.

The same formulas are mirrored in ``ts/binomialCI.ts`` for the
TypeScript side; keep the two in step.
"""

from __future__ import annotations

import numpy as np


# Two-sided normal quantiles z_{1-α/2} for the supported levels.
Z_SCORES: dict[float, float] = {
    0.90: 1.644853,
    0.95: 1.959964,
}


def z_score(confidence: float = 0.95) -> float:
    """Return the two-sided z quantile for ``confidence`` (0.90 or 0.95)."""
    try:
        return Z_SCORES[round(float(confidence), 2)]
    except KeyError:
        raise ValueError(
            f"unsupported confidence level {confidence!r}; "
            f"expected one of {sorted(Z_SCORES)}"
        ) from None


def standard_error(p_hat, n):
    """Bernoulli standard error ``sqrt(p̂(1-p̂)/n)``."""
    p = np.asarray(p_hat, dtype=float)
    n = np.asarray(n, dtype=float)
    return np.sqrt(p * (1.0 - p) / n)


def wald_interval(p_hat, n, confidence: float = 0.95):
    """Standard (Wald) interval ``p̂ ± z·SE``, clipped to [0, 1]."""
    p = np.asarray(p_hat, dtype=float)
    z = z_score(confidence)
    margin = z * standard_error(p, n)
    return np.clip(p - margin, 0.0, 1.0), np.clip(p + margin, 0.0, 1.0)


def wilson_interval(p_hat, n, confidence: float = 0.95):
    """Wilson score interval, clipped to [0, 1].

    ``center = (p̂ + z²/2n) / (1 + z²/n)`` and
    ``margin = z/(1 + z²/n) · sqrt(p̂(1-p̂)/n + z²/4n²)``.
    """
    p = np.asarray(p_hat, dtype=float)
    n_arr = np.asarray(n, dtype=float)
    z = z_score(confidence)
    z2 = z * z
    denom = 1.0 + z2 / n_arr
    center = (p + z2 / (2.0 * n_arr)) / denom
    margin = (z / denom) * np.sqrt(
        p * (1.0 - p) / n_arr + z2 / (4.0 * n_arr * n_arr)
    )
    return np.clip(center - margin, 0.0, 1.0), np.clip(center + margin, 0.0, 1.0)


def target_band(p0: float, n, confidence: float = 0.95):
    """Observed rates consistent with a true probability ``p0``.

    The score-test acceptance region ``p0 ± z·sqrt(p0(1-p0)/n)`` — the
    null SE, not the estimated one — so a bar falling outside the band
    differs from ``p0`` at that significance level.
    """
    n_arr = np.asarray(n, dtype=float)
    z = z_score(confidence)
    half = z * np.sqrt(p0 * (1.0 - p0) / n_arr)
    return np.clip(p0 - half, 0.0, 1.0), np.clip(p0 + half, 0.0, 1.0)
