"""Metric functions for the prediction evaluation.

The headline metrics (per project convention) are:

  * coverage — ``P(|err| ≤ CI)`` on clean segments,
  * median / mean absolute error on clean segments,
  * median CI half-width on clean segments.

All metrics are computed on **clean** data only (per
``signalClearRecording == True``) since that is what the analysis
cares about. The quality filter's accept/reject split is reported
separately so we can assess false-accept / false-reject rates.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Optional

import numpy as np
import pandas as pd


@dataclass
class MetricsBundle:
    n_total: int
    n_clean: int
    n_accepted_clean: int           # clean AND quality-filter accepted

    # --- metrics on clean (filter-ignored) ---
    clean_mae: float
    clean_median_abs_err: float
    clean_rmse: float
    clean_coverage_90: float
    clean_median_ci: float
    clean_mean_ci: float
    clean_p95_abs_err: float

    # --- metrics on clean AND accepted ---
    accepted_clean_mae: float
    accepted_clean_median_abs_err: float
    accepted_clean_rmse: float
    accepted_clean_coverage_90: float
    accepted_clean_median_ci: float

    # --- quality filter ---
    filter_accept_rate: float          # all segments
    filter_accept_rate_clean: float    # clean-only
    filter_reject_rate_unclean: float  # unclean-only — do we catch them?

    # --- within-1-floor stats (±1.5m) ---
    clean_frac_within_1_5m: float
    clean_frac_within_3m: float

    def to_dict(self) -> dict:
        return asdict(self)


def _safe(fn, arr, default=float("nan")):
    arr = np.asarray(arr, dtype=float)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return default
    return float(fn(arr))


def compute_metrics(df: pd.DataFrame) -> MetricsBundle:
    """Compute metrics on the flat predictions DataFrame from
    :func:`collect_predictions`.
    """
    total = len(df)
    clean = df[df["signal_clear"] == True]                 # noqa: E712
    unclean = df[df["signal_clear"] == False]              # noqa: E712
    accepted_clean = clean[clean["accepted"] == True]      # noqa: E712

    def _cov(sub: pd.DataFrame) -> float:
        if sub.empty: return float("nan")
        return float(np.mean(sub["covered"].to_numpy()))

    return MetricsBundle(
        n_total=int(total),
        n_clean=int(len(clean)),
        n_accepted_clean=int(len(accepted_clean)),

        clean_mae=_safe(np.mean, clean["abs_error"]),
        clean_median_abs_err=_safe(np.median, clean["abs_error"]),
        clean_rmse=_safe(lambda x: np.sqrt(np.mean(x ** 2)), clean["abs_error"]),
        clean_coverage_90=_cov(clean),
        clean_median_ci=_safe(np.median, clean["ci_half_width"]),
        clean_mean_ci=_safe(np.mean, clean["ci_half_width"]),
        clean_p95_abs_err=_safe(lambda x: np.quantile(x, 0.95), clean["abs_error"]),

        accepted_clean_mae=_safe(np.mean, accepted_clean["abs_error"]),
        accepted_clean_median_abs_err=_safe(np.median, accepted_clean["abs_error"]),
        accepted_clean_rmse=_safe(lambda x: np.sqrt(np.mean(x ** 2)), accepted_clean["abs_error"]),
        accepted_clean_coverage_90=_cov(accepted_clean),
        accepted_clean_median_ci=_safe(np.median, accepted_clean["ci_half_width"]),

        filter_accept_rate=_safe(np.mean, df["accepted"].astype(float)),
        filter_accept_rate_clean=_safe(np.mean, clean["accepted"].astype(float)),
        filter_reject_rate_unclean=(
            float("nan") if unclean.empty
            else float(1.0 - np.mean(unclean["accepted"].astype(float)))
        ),

        clean_frac_within_1_5m=_safe(lambda x: float(np.mean(x <= 1.5)), clean["abs_error"]),
        clean_frac_within_3m=_safe(lambda x: float(np.mean(x <= 3.0)), clean["abs_error"]),
    )


def per_experiment_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """Coverage / MAE per experiment on clean segments."""
    sub = df[df["signal_clear"] == True]                   # noqa: E712
    rows = []
    for exp, grp in sub.groupby("exp_name"):
        if grp.empty: continue
        rows.append({
            "exp_name": exp,
            "n_clean": int(len(grp)),
            "mae": float(np.mean(grp["abs_error"])),
            "median_abs_err": float(np.median(grp["abs_error"])),
            "coverage_90": float(np.mean(grp["covered"])),
            "median_ci": float(np.median(grp["ci_half_width"])),
            "experiment_type": grp["experiment_type"].iloc[0],
        })
    if not rows:
        return pd.DataFrame(columns=[
            "exp_name", "n_clean", "mae", "median_abs_err",
            "coverage_90", "median_ci", "experiment_type",
        ])
    return pd.DataFrame(rows).sort_values(["experiment_type", "exp_name"])
