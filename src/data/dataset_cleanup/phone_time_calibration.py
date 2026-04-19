"""Cross-phone time calibration anchored on the Pixel 10 recording.

Each physical experiment (``_expN`` at a given building) was recorded
simultaneously by several phones. Their recording clocks are nominally
wall-clock Unix ms (Eyal's contract), but hitting "start" a few seconds
apart on each phone leaves the sensor streams offset by up to ~30 s.

Task 2 (per Uriya, 2026-04-19): the Google Pixel 10 recording is the time
ground truth. For every other phone in the same experiment, estimate the
constant time offset via cross-correlation of the accelerometer magnitude
(which has strong, rotation-invariant signatures at the elevator takeoffs
and landings), then shift the non-Pixel phone's timestamps so its events
line up with the Pixel's.

Outputs:

* Shifts ``timestamp_ms`` in place in every per-sensor CSV under
  ``structuredData/data/<non_pixel_exp>/``.
* Shifts ``start_ms`` and ``end_ms`` in ``gt.csv`` by the same amount so
  the segment boundaries still correspond to the physical events.
* Writes a diagnostic plot ``phone_time_calibration.png`` next to the
  other sensor CSVs: Pixel's ACC magnitude vs. phone's ACC magnitude,
  before and after the shift.
* Writes a top-level summary ``structuredData/phone_time_calibration.csv``
  with (exp_name, pixel_ref, offset_ms, correlation).

Run with::

    python -m src.data.dataset_cleanup.phone_time_calibration              # all experiments
    python -m src.data.dataset_cleanup.phone_time_calibration <name>...    # specific experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader.constants import (
    GT_CSV,
    METADATA_CSV,
    STRUCTURED_DATA_DIR,
    STRUCTURED_ROOT,
)

# Grid resolution for resampling before xcorr — 10 Hz is plenty for elevator
# takeoff/landing lobes (rise-time ~1 s). Also keeps FFT cheap.
GRID_HZ = 10.0

# Maximum absolute lag (ms) we'll search. Observed peaks include some
# phones (notably Samsung SM-S911B) that land ~25-30 s off due to delayed
# recording start after the filename timestamp was stamped. 60 s gives us
# plenty of headroom without straying into the region where spurious
# correlations from repetitive walking patterns might dominate.
MAX_LAG_MS = 60_000

CAL_PLOT_FILENAME = "phone_time_calibration.png"
CAL_SUMMARY_CSV = "phone_time_calibration.csv"


# --------------------------------------------------------------------------
# Pixel reference resolution
# --------------------------------------------------------------------------

def _split_exp_name(exp_name: str) -> tuple[str | None, str | None]:
    """Return `(building_slug_lower, exp_suffix_lower)` for an exp name.

    Experiment folder names follow
    ``<experimenter>_<buildingSlug>_<phone>_<date>[_expN]``.
    Returns ``(None, None)`` if the exp suffix isn't of the `_expN` form.
    """
    parts = exp_name.split("_")
    # find `expN`
    exp_token = next((t for t in parts if t.lower().startswith("exp") and t[3:].isdigit()), None)
    if exp_token is None:
        return None, None
    # building slug is the second token for our layout
    if len(parts) < 2:
        return None, None
    return parts[1].lower(), exp_token.lower()


def find_pixel_reference_exp(exp_name: str) -> str | None:
    """Locate the Pixel 10 recording that corresponds to the same physical
    experiment as ``exp_name``. Returns ``None`` when no Pixel recording
    exists (archive experiments, BarIlan2 / Haari which are Pixel-only anyway,
    or exp-suffix-less names)."""
    building, exp_suffix = _split_exp_name(exp_name)
    if building is None:
        return None
    # Don't call a Pixel its own reference.
    if "pixel" in exp_name.lower():
        return None
    if not STRUCTURED_DATA_DIR.is_dir():
        return None
    for candidate in sorted(p.name for p in STRUCTURED_DATA_DIR.iterdir() if p.is_dir()):
        c_bld, c_exp = _split_exp_name(candidate)
        if c_bld != building or c_exp != exp_suffix:
            continue
        if "pixel" in candidate.lower():
            return candidate
    return None


# --------------------------------------------------------------------------
# Cross-correlation
# --------------------------------------------------------------------------

def _acc_magnitude(acc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    t = acc["timestamp_ms"].to_numpy(dtype=np.int64)
    mag = np.sqrt(
        acc["x"].to_numpy(dtype=float) ** 2
        + acc["y"].to_numpy(dtype=float) ** 2
        + acc["z"].to_numpy(dtype=float) ** 2
    )
    return t, mag


def _resample_to_grid(
    t_ms: np.ndarray, y: np.ndarray, grid_t_ms: np.ndarray,
) -> np.ndarray:
    """Linear-interpolate (t_ms, y) onto ``grid_t_ms``. Extrapolated edges use
    0 (no signal → no contribution to the xcorr)."""
    if t_ms.size < 2:
        return np.zeros_like(grid_t_ms, dtype=float)
    return np.interp(grid_t_ms.astype(float), t_ms.astype(float), y, left=0.0, right=0.0)


# Minimum xcorr score to trust the xcorr offset. Below this we fall back
# to min-altitude anchoring, and if that is also weak we skip the shift.
MIN_XCORR_SCORE = 0.6


def _min_altitude_offset_ms(
    pixel_prs: pd.DataFrame, phone_prs: pd.DataFrame,
) -> tuple[int, float]:
    """Fallback alignment: match the wall-clock times of the minimum-altitude
    sample in each recording.

    Returns ``(offset_ms, confidence)`` where confidence is a rough
    0-1 measure of how much the minimum stands out versus the session
    baseline (0 = totally flat, ≥ 1 = clean big excursion). Use when the
    xcorr score is too low to trust but there's a dominant altitude event.
    """
    def _min_info(prs: pd.DataFrame) -> tuple[int, float, float]:
        h = prs["GT_height_m"].to_numpy(dtype=float) if "GT_height_m" in prs.columns \
            else np.asarray(
                [pressure_to_altitude(v) for v in prs["pressure"].to_numpy(dtype=float)],
                dtype=float,
            )
        t = prs["timestamp_ms"].to_numpy(dtype=np.int64)
        h_s = pd.Series(h).rolling(window=101, center=True, min_periods=1).median().to_numpy()
        i = int(np.argmin(h_s))
        depth = float(np.median(h_s) - h_s[i])
        return int(t[i]), depth, float(np.std(h_s))

    from src.physics.barometric import pressure_to_altitude  # noqa: F401
    p_t_min, p_depth, p_std = _min_info(pixel_prs)
    q_t_min, q_depth, q_std = _min_info(phone_prs)
    offset_ms = int(p_t_min - q_t_min)
    # Confidence: smallest (depth / std) across both signals. If the min is a
    # big excursion relative to session variability, confidence is high.
    conf = float(min(p_depth / max(p_std, 1e-3), q_depth / max(q_std, 1e-3)))
    return offset_ms, conf


def _prs_altitude_series(prs: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamp_ms, altitude_m) for cross-phone xcorr.

    Uses the already-computed ``GT_height_m`` column when present; otherwise
    computes an ISA-standard conversion from ``pressure`` on the fly.
    """
    t = prs["timestamp_ms"].to_numpy(dtype=np.int64)
    if "GT_height_m" in prs.columns:
        h = prs["GT_height_m"].to_numpy(dtype=float)
    else:
        from src.physics.barometric import pressure_to_altitude
        h = np.asarray(
            pressure_to_altitude(prs["pressure"].to_numpy(dtype=float)),
            dtype=float,
        )
    return t, h


def _xcorr_segment_offset(
    pixel_mag: np.ndarray, phone_mag: np.ndarray, max_lag_samples: int,
) -> tuple[int, float]:
    """Xcorr two equal-sample-rate ACC-magnitude windows.

    `phone_mag` is assumed to be padded on each side by `max_lag_samples`
    so the valid-lag range is exactly ±max_lag_samples. Returns
    ``(best_lag_samples, score)`` where `best_lag_samples` is the number of
    samples to SHIFT the phone by (positive = phone's current timeline is
    behind Pixel → shift forward) and score is the peak normalised
    correlation (higher = more confident).
    """
    if pixel_mag.size < 4 or phone_mag.size < 4 or pixel_mag.size > phone_mag.size:
        return 0, 0.0

    def _zn(x: np.ndarray) -> np.ndarray:
        x = x - float(np.mean(x))
        s = float(np.std(x))
        return x / s if s > 1e-9 else x

    p = _zn(pixel_mag)
    q = _zn(phone_mag)

    from scipy.signal import correlate
    # Valid-mode slides the short signal (p, length N) over the long (q, length N + 2L).
    # Output length = 2L + 1, lags k = -L ... +L. At lag 0 the shorter signal sits
    # centered in the longer — which is where we expect zero offset.
    corr = correlate(q, p, mode="valid")
    lags = np.arange(-max_lag_samples, max_lag_samples + 1)
    if corr.size != lags.size:
        return 0, 0.0
    best = int(np.argmax(corr))
    best_lag = int(lags[best])
    score = float(corr[best] / p.size)
    return best_lag, score


def _acc_magnitude_in_window(
    acc: pd.DataFrame, t_lo_ms: int, t_hi_ms: int,
    grid_t_ms: np.ndarray, high_pass_hz: float | None = 0.15,
) -> np.ndarray:
    """Resample an ACC-magnitude slice onto `grid_t_ms` (in ms).

    When ``high_pass_hz`` is non-None (default 0.15 Hz), a zero-phase
    Butterworth high-pass is applied before resampling to remove gravity
    / slow orientation drift that otherwise dominates the xcorr
    normalisation (since ``|a|`` averages ~9.81 m/s² but elevator jerks
    are only ±0.5-1 m/s²).
    """
    t, mag = _acc_magnitude(acc)
    mask = (t >= t_lo_ms - 500) & (t <= t_hi_ms + 500)
    if mask.sum() < 2:
        return np.zeros_like(grid_t_ms, dtype=float)
    t_win = t[mask]
    m_win = mag[mask]

    if high_pass_hz is not None and m_win.size >= 32:
        from scipy.signal import butter, sosfiltfilt
        # Median sampling rate inside the window.
        dt = np.diff(t_win.astype(float)) / 1000.0
        dt_good = dt[dt > 0]
        if dt_good.size and np.median(dt_good) > 0:
            fs = 1.0 / float(np.median(dt_good))
            if fs > 2 * high_pass_hz:
                sos = butter(2, high_pass_hz / (0.5 * fs), btype="high", output="sos")
                m_win = sosfiltfilt(sos, m_win)

    return _resample_to_grid(t_win, m_win, grid_t_ms)


def _prs_altitude_in_window(
    prs: pd.DataFrame, t_lo_ms: int, t_hi_ms: int, grid_t_ms: np.ndarray,
) -> np.ndarray:
    """Resample an altitude slice onto `grid_t_ms` (in ms)."""
    t, h = _prs_altitude_series(prs)
    mask = (t >= t_lo_ms - 500) & (t <= t_hi_ms + 500)
    if mask.sum() < 2:
        return np.zeros_like(grid_t_ms, dtype=float)
    return _resample_to_grid(t[mask], h[mask], grid_t_ms)


def _per_segment_offsets_ms(
    pixel_sig: pd.DataFrame, phone_sig: pd.DataFrame,
    pixel_gt: pd.DataFrame,
    signal_kind: str,            # "acc" or "prs"
    grid_hz: float = 25.0, max_lag_ms: int = MAX_LAG_MS,
) -> tuple[list[int], list[float]]:
    """Per-elevator-segment time offsets. Returns ``(offsets_ms, scores)``.

    Sign convention: a NEGATIVE xcorr lag means the phone's matching event
    appears EARLIER than Pixel's tag in the padded phone window (phone clock
    is behind Pixel's for the same physical event). To align, we ADD ``-lag``
    to the phone's timestamps — hence the negation before returning.
    """
    step_ms = int(round(1000.0 / grid_hz))
    max_lag_samples = int(round(max_lag_ms / step_ms))
    rides = pixel_gt[pixel_gt["type"].astype(str).str.lower().isin(("up", "down"))]
    offsets: list[int] = []
    scores: list[float] = []

    window_fn = _acc_magnitude_in_window if signal_kind == "acc" else _prs_altitude_in_window
    for _, row in rides.iterrows():
        s, e = int(row["start_ms"]), int(row["end_ms"])
        if e - s < 1000:
            continue
        p_grid = np.arange(s, e + step_ms, step_ms, dtype=np.int64)
        q_grid = np.arange(
            s - max_lag_samples * step_ms,
            e + max_lag_samples * step_ms + step_ms,
            step_ms,
            dtype=np.int64,
        )
        p_s = window_fn(pixel_sig, s, e, p_grid)
        q_s = window_fn(phone_sig, s - max_lag_ms, e + max_lag_ms, q_grid)
        if np.std(p_s) < 0.05 or np.std(q_s) < 0.05:
            continue
        lag_samples, score = _xcorr_segment_offset(p_s, q_s, max_lag_samples)
        offsets.append(-lag_samples * step_ms)
        scores.append(score)

    return offsets, scores


def _combine_per_segment(
    offsets: list[int], scores: list[float],
) -> tuple[int, float, int, float]:
    """Collapse per-segment offsets into ``(median_offset, median_score,
    n_segments, consistency)``.

    Consistency rule:
      * ``MAD < 500 ms`` → 1.0 (per-segment answers agree within 0.5 s — the
        resolution floor at 25 Hz grid).
      * Else ``1 - MAD / (|median| + 500)``, clipped to [0, 1].
    """
    if not offsets:
        return 0, 0.0, 0, 0.0
    arr = np.asarray(offsets, dtype=float)
    med = float(np.median(arr))
    mad = float(np.median(np.abs(arr - med)))
    consistency = 1.0 if mad < 500.0 else max(0.0, 1.0 - mad / (abs(med) + 500.0))
    return int(round(med)), float(np.median(scores)), len(offsets), consistency


def _majority_vote_offset(
    offsets: list[int], scores: list[float],
    bin_ms: int = 1000,
) -> tuple[int, float, int, float]:
    """Uriya's majority-vote rule: all tagged segments in an experiment share
    one physical clock offset, so per-segment xcorr answers that agree
    across rides are the real signal — outlier rides (bad xcorr peaks from
    walking noise, dropped phones, etc.) should be ignored rather than
    pulled into the median.

    Groups per-segment offsets into ``bin_ms``-wide bins, finds the heaviest
    bin, and returns ``(median_of_that_bin, median_score, n_segments,
    support_fraction)`` where ``support_fraction`` is how many segments
    landed in the winning bin out of the total. The caller enforces a
    minimum ``support_fraction`` before trusting the answer.
    """
    if not offsets:
        return 0, 0.0, 0, 0.0

    arr = np.asarray(offsets, dtype=float)
    bins = np.round(arr / bin_ms).astype(int)
    # Mode: the bin with the most entries.
    uniq, counts = np.unique(bins, return_counts=True)
    winner_bin = int(uniq[int(np.argmax(counts))])
    mask = bins == winner_bin
    support = int(mask.sum())
    support_frac = float(support / len(offsets))

    in_bin_offsets = arr[mask]
    in_bin_scores = np.asarray(scores, dtype=float)[mask]
    winner_median = int(round(float(np.median(in_bin_offsets))))
    winner_score = float(np.median(in_bin_scores))
    return winner_median, winner_score, len(offsets), support_frac


def _xcorr_offset_ms(
    pixel_acc: pd.DataFrame, phone_acc: pd.DataFrame,
    pixel_prs: pd.DataFrame | None, phone_prs: pd.DataFrame | None,
    pixel_gt: pd.DataFrame,
    max_lag_ms: int = MAX_LAG_MS,
) -> tuple[int, float, int, float, str]:
    """Combine per-segment xcorr offsets. Prefers ACC; falls back to PRS when
    ACC's per-segment answers are too noisy to agree.

    Returns ``(offset_ms, median_score, n_segments, consistency, signal_used)``
    where ``signal_used`` is ``"acc"`` or ``"prs"`` or ``"acc(degraded)"`` when
    neither was clean.
    """
    def _trusted(result: tuple[int, float, int, float]) -> bool:
        # result = (offset_ms, median_score, n_segments, consistency)
        return result[2] >= 3 and result[3] >= 0.5 and result[1] >= 0.5

    acc_offsets, acc_scores = _per_segment_offsets_ms(
        pixel_acc, phone_acc, pixel_gt, "acc", max_lag_ms=max_lag_ms,
    )
    acc_result = _combine_per_segment(acc_offsets, acc_scores)
    if _trusted(acc_result):
        return (*acc_result, "acc")

    # Fallback 1: per-segment PRS altitude xcorr — rescues Xiaomi-style
    # phones whose ACC signatures are weaker during elevator rides.
    if pixel_prs is not None and phone_prs is not None:
        prs_offsets, prs_scores = _per_segment_offsets_ms(
            pixel_prs, phone_prs, pixel_gt, "prs", max_lag_ms=max_lag_ms,
        )
        prs_result = _combine_per_segment(prs_offsets, prs_scores)
        if _trusted(prs_result):
            return (*prs_result, "prs")

    # Fallback 2: majority vote over ACC per-segment offsets. Catches cases
    # where half the rides xcorr correctly and the other half hit spurious
    # peaks — the median gets pulled between the two clusters and looks
    # inconsistent, but the real cluster is still the heavier one.
    if len(acc_offsets) >= 3:
        mv = _majority_vote_offset(acc_offsets, acc_scores)
        # Trust when ≥30 % of rides fall in the winning bin AND the bin's
        # median xcorr score is itself >= 0.3 (not a pure-noise bin).
        if mv[3] >= 0.3 and mv[1] >= 0.3:
            return (*mv, "acc(majority)")

    return (*acc_result, "acc(degraded)")


# --------------------------------------------------------------------------
# Apply shifts
# --------------------------------------------------------------------------

def _shift_all_csvs(exp_dir: Path, offset_ms: int) -> list[str]:
    """Add ``offset_ms`` to every ``timestamp_ms`` / ``start_ms`` / ``end_ms``
    column in the CSVs under ``exp_dir``.

    Returns the list of modified filenames (for logging). Skips files that
    already lack a time column (baramoshka.csv, metadata.csv)."""
    modified: list[str] = []
    for path in sorted(exp_dir.glob("*.csv")):
        if path.name == METADATA_CSV or path.name == "baramoshka.csv":
            continue
        df = pd.read_csv(path)
        touched = False
        for col in ("timestamp_ms", "start_ms", "end_ms"):
            if col in df.columns:
                df[col] = df[col].astype("int64") + int(offset_ms)
                touched = True
        if touched:
            df.to_csv(path, index=False)
            modified.append(path.name)
    return modified


# --------------------------------------------------------------------------
# Diagnostic plot
# --------------------------------------------------------------------------

def _save_calibration_plot(
    exp_dir: Path,
    pixel_acc: pd.DataFrame,
    phone_acc: pd.DataFrame,
    pixel_gt: pd.DataFrame,
    offset_ms: int,
    score: float,
    consistency: float,
    will_apply: bool = True,
) -> None:
    """Save a per-elevator-segment ACC overlay showing Pixel vs phone pulses.

    Two display modes:

    * ``will_apply=True`` (default): orange=phone before the candidate shift,
      green=phone after — so the user can see whether the proposed shift
      actually lines green up with blue.
    * ``will_apply=False`` (xcorr returned a low-confidence/spurious peak
      that the caller decided to skip): only draw green=phone at its
      current timestamps, since drawing the rejected "after shift" trace
      would misleadingly suggest a problem that the calibration is not
      actually going to apply.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_t, p_mag = _acc_magnitude(pixel_acc)
    q_t, q_mag = _acc_magnitude(phone_acc)

    rides = pixel_gt[pixel_gt["type"].astype(str).str.lower().isin(("up", "down"))]
    rides = rides.reset_index(drop=True)
    n_plot = min(6, len(rides))
    if n_plot == 0:
        return

    if len(rides) >= n_plot:
        idxs = np.linspace(0, len(rides) - 1, n_plot, dtype=int)
    else:
        idxs = np.arange(len(rides))

    nrows = (n_plot + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3 * nrows), squeeze=False)
    if will_apply:
        title = (
            f"Per-segment ACC alignment — shift to add to phone = {offset_ms:+d} ms "
            f"  |  median xcorr score={score:.2f}  consistency={consistency:.2f}  "
            f"({len(rides)} tagged up/down segments)"
        )
    else:
        title = (
            f"Per-segment ACC alignment — current state (no shift applied)  "
            f"xcorr-proposed +{offset_ms} ms REJECTED (score={score:.2f}, "
            f"consistency={consistency:.2f} below trust thresholds)"
        )
    fig.suptitle(title, fontsize=11)

    for k, i in enumerate(idxs):
        ax = axes[k // 2][k % 2]
        row = rides.iloc[int(i)]
        s, e = int(row["start_ms"]), int(row["end_ms"])
        pad = 5_000
        p_mask = (p_t >= s - pad) & (p_t <= e + pad)

        ax.axvspan((s - s) / 1000.0, (e - s) / 1000.0, color="tab:red", alpha=0.08,
                   label="Pixel-tagged ride")
        ax.plot((p_t[p_mask] - s) / 1000.0, p_mag[p_mask],
                color="tab:blue", lw=0.9, alpha=0.9, label="Pixel |a|")

        if will_apply:
            q_mask_before = (q_t >= s - pad) & (q_t <= e + pad)
            q_mask_after = (q_t + offset_ms >= s - pad) & (q_t + offset_ms <= e + pad)
            ax.plot((q_t[q_mask_before] - s) / 1000.0, q_mag[q_mask_before],
                    color="tab:orange", lw=0.8, alpha=0.45,
                    label="phone |a| (before shift)")
            ax.plot(((q_t[q_mask_after] + offset_ms) - s) / 1000.0,
                    q_mag[q_mask_after],
                    color="tab:green", lw=0.8, alpha=0.85,
                    label="phone |a| (after shift)")
        else:
            q_mask = (q_t >= s - pad) & (q_t <= e + pad)
            ax.plot((q_t[q_mask] - s) / 1000.0, q_mag[q_mask],
                    color="tab:green", lw=0.9, alpha=0.85,
                    label="phone |a| (current state)")

        ax.set_title(f"Segment {int(i)} — {row['type']}  "
                     f"dur={(e - s) / 1000:.1f}s", fontsize=9)
        ax.set_xlabel("time (s, relative to segment start)")
        ax.set_ylabel("|a| (m/s²)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    total_axes = nrows * 2
    for k in range(n_plot, total_axes):
        axes[k // 2][k % 2].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(exp_dir / CAL_PLOT_FILENAME, dpi=120)
    plt.close(fig)


# --------------------------------------------------------------------------
# Driver
# --------------------------------------------------------------------------

def _calibrate_one(exp_name: str, pixel_name: str, apply: bool) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    pix_dir = STRUCTURED_DATA_DIR / pixel_name
    pix_acc_path = pix_dir / "ACC.csv"
    pix_gt_path = pix_dir / GT_CSV
    phone_acc_path = exp_dir / "ACC.csv"
    if not pix_acc_path.exists() or not phone_acc_path.exists() or not pix_gt_path.exists():
        return {
            "exp_name":      exp_name,
            "pixel_ref":     pixel_name,
            "offset_ms":     0,
            "method":        "missing_inputs",
            "median_score":  0.0,
            "n_segments":    0,
            "consistency":   0.0,
            "csvs_modified": "",
            "applied":       "False",
        }
    pix_acc = pd.read_csv(pix_acc_path)
    phone_acc = pd.read_csv(phone_acc_path)
    pix_gt = pd.read_csv(pix_gt_path)
    pix_prs_path = pix_dir / "PRS.csv"
    phone_prs_path = exp_dir / "PRS.csv"
    pix_prs = pd.read_csv(pix_prs_path) if pix_prs_path.exists() else None
    phone_prs = pd.read_csv(phone_prs_path) if phone_prs_path.exists() else None

    offset_ms, median_score, n_segs, consistency, signal_used = _xcorr_offset_ms(
        pix_acc, phone_acc, pix_prs, phone_prs, pix_gt,
    )

    # Trust rule: ACC, PRS, or ACC-majority-vote qualifies; "acc(degraded)" does not.
    # Also require ≥3 segments. Consistency/support thresholds already handled
    # inside `_xcorr_offset_ms` — we just accept whatever signal it returned.
    trustworthy = signal_used in ("acc", "prs", "acc(majority)") and n_segs >= 3
    if n_segs < 3:
        method = "skipped_too_few_segments"
    elif not trustworthy:
        method = f"skipped_low_confidence ({signal_used})"
    else:
        method = f"per_segment_{signal_used}_xcorr"

    modified: list[str] = []
    will_apply = apply and offset_ms != 0 and trustworthy
    if will_apply:
        modified = _shift_all_csvs(exp_dir, offset_ms)

    _save_calibration_plot(
        exp_dir, pix_acc, phone_acc, pix_gt,
        offset_ms, median_score, consistency,
        will_apply=trustworthy,
    )

    return {
        "exp_name":      exp_name,
        "pixel_ref":     pixel_name,
        "offset_ms":     offset_ms,
        "method":        method,
        "signal_used":   signal_used,
        "median_score":  round(median_score, 4),
        "n_segments":    n_segs,
        "consistency":   round(consistency, 3),
        "csvs_modified": ",".join(modified),
        "applied":       str(will_apply),
    }


def _skip_entry(exp_name: str, reason: str) -> dict:
    return {
        "exp_name":      exp_name,
        "pixel_ref":     "",
        "offset_ms":     0,
        "method":        reason,
        "median_score":  0.0,
        "n_segments":    0,
        "consistency":   0.0,
        "csvs_modified": "",
        "applied":       "False",
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    # Flag parsing: `--apply` makes it destructive (shifts CSVs in place).
    # Default is dry-run: compute offsets + save diagnostic plots only.
    raw_args = sys.argv[1:]
    apply = "--apply" in raw_args
    exp_args = [a for a in raw_args if not a.startswith("--")]

    if exp_args:
        exp_names = exp_args
    else:
        exp_names = sorted(
            p.name for p in STRUCTURED_DATA_DIR.iterdir()
            if p.is_dir() and (p / "ACC.csv").exists()
        )

    rows: list[dict] = []
    for name in exp_names:
        pix = find_pixel_reference_exp(name)
        if pix is None:
            rows.append(_skip_entry(name, "no pixel reference"))
            continue
        try:
            rows.append(_calibrate_one(name, pix, apply=apply))
        except Exception as e:
            rows.append({
                "exp_name":      name,
                "pixel_ref":     pix,
                "offset_ms":     0,
                "method":        f"failed: {type(e).__name__}: {e}",
                "median_score":  0.0,
                "n_segments":    0,
                "consistency":   0.0,
                "csvs_modified": "",
                "applied":       "False",
            })

    out = pd.DataFrame(rows)
    out_path = STRUCTURED_ROOT / "test_results" / CAL_SUMMARY_CSV
    out.to_csv(out_path, index=False)
    mode = "APPLY (CSVs shifted in place)" if apply else "DRY-RUN (plots only, no CSV writes)"
    print(f"[calibration] mode: {mode}")
    print(f"[calibration] Wrote summary: {out_path}")
    print()
    print(out.to_string(index=False))


if __name__ == "__main__":
    main()
