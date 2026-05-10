"""Generate corrupted copies of a structuredData experiment for testing.

Several corruption modes that exercise the gap-aware loader / UI:

* ``dropout``        — delete every sensor sample in a fixed time
  window. Forces the loader to split the timeline into 2 valid
  intervals; the UI must surface a blue "no data" band over the gap.
* ``thinning``       — keep only sparse samples in the same window so
  the spacing exceeds ``GAP_THRESHOLD_S`` (default 1 s). Mimics a phone
  throttling its IMU; loader should still produce 2 valid intervals.
* ``halving``        — drop every other sample across the whole frame.
  The signal stays uniform but at half the native rate. Loader should
  produce a single valid interval (no gap > 1 s) and resample cleanly.
* ``multi_dropout``  — five dropout windows of varying lengths
  (3 s, 7 s, 15 s, 22 s, 30 s) spread across the trace at fractional
  offsets 10, 25, 45, 65, 85 %. Useful for verifying the loader
  handles many gaps at once.
* ``between_rides``  — read ``gt.csv``, find every ``outside`` interval,
  and place a dropout inside each one (centred, with a configurable
  margin from the adjacent rides). The dropout length scales with the
  outside-interval length so longer "quiet" stretches get longer holes.
  Good periods stay at the native sampling rate — no thinning, no
  halving — so this mode tests "no data between rides" without
  changing the in-ride signal.
* ``long_dropout``   — one or two very long dropouts (default 90 s and
  150 s) placed inside the longest available ``outside`` intervals.
  Stress-tests the UI overlays on multi-minute holes.

Output goes to ``structuredData/data/<exp>__corrupted_<mode>/``. Sensor
CSVs are mutated in place; ``gt.csv``, ``metadata.csv``,
``baramoshka.csv`` and any diagnostic PNGs are copied byte-for-byte
(except ``metadata.csv.exp_name`` is rewritten to match the new folder).

Usage::

    python scripts/generate_corrupted_test_data.py <exp_name> \\
        --mode {dropout,thinning,halving,multi_dropout,
                between_rides,long_dropout}
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parent
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import (  # noqa: E402
    BAROMOSHKA_CSV,
    GAP_THRESHOLD_S,
    GT_CSV,
    METADATA_CSV,
    METADATA_COLUMNS,
    SENSOR_COLUMNS,
    STRUCTURED_DATA_DIR,
)


MODES = (
    "dropout", "thinning", "halving", "multi_dropout",
    "between_rides", "long_dropout", "halvingDiffrentParts",
)


def _apply_dropout(
    df: pd.DataFrame, gap_start_s: float, gap_len_s: float,
) -> pd.DataFrame:
    """Delete every row whose timestamp falls inside the gap window."""
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    t0 = int(ts[0])
    lo = t0 + int(round(gap_start_s * 1000.0))
    hi = lo + int(round(gap_len_s * 1000.0))
    keep = (ts < lo) | (ts >= hi)
    return df.loc[keep].reset_index(drop=True)


def _apply_multi_dropout(
    df: pd.DataFrame, windows: list[tuple[float, float]],
) -> pd.DataFrame:
    """Apply :func:`_apply_dropout` for each ``(start_s, len_s)`` window.

    Windows are processed in order; each is interpreted relative to the
    *original* first timestamp of ``df``, not the previous corruption,
    so they're easy to reason about.
    """
    if not windows:
        return df
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    t0 = int(ts[0])
    keep = np.ones(len(ts), dtype=bool)
    for start_s, len_s in windows:
        lo = t0 + int(round(start_s * 1000.0))
        hi = lo + int(round(len_s * 1000.0))
        keep &= (ts < lo) | (ts >= hi)
    return df.loc[keep].reset_index(drop=True)


def _read_outside_intervals_ms(
    src_dir: Path,
) -> list[tuple[int, int]]:
    """Return ``[(start_ms, end_ms), ...]`` for every ``outside`` row in
    the experiment's ``gt.csv``. Empty list when no GT exists.
    """
    gt_path = src_dir / GT_CSV
    if not gt_path.exists():
        return []
    gt = pd.read_csv(gt_path)
    if gt.empty or "type" not in gt.columns:
        return []
    out_rows = gt.loc[gt["type"].astype(str).str.lower() == "outside"]
    return [
        (int(r["start_ms"]), int(r["end_ms"]))
        for _, r in out_rows.iterrows()
    ]


def _windows_between_rides(
    src_dir: Path, t0_ms: int,
    margin_s: float = 2.0,
    occupancy: float = 0.55,
    min_outside_s: float = 6.0,
    min_window_s: float = 4.0,
    max_window_s: float = 90.0,
) -> list[tuple[float, float]]:
    """Pick a dropout window inside every long-enough ``outside``
    interval in ``gt.csv``.

    Each candidate window is centred in its outside interval, leaves
    ``margin_s`` of breathing room next to the adjacent rides, takes
    ``occupancy`` of the available middle, and is clipped to
    ``[min_window_s, max_window_s]``. Outside intervals shorter than
    ``min_outside_s`` are skipped entirely.

    Returned windows are in seconds since ``t0_ms`` so the rest of the
    script can feed them to :func:`_apply_multi_dropout` unchanged.
    """
    intervals = _read_outside_intervals_ms(src_dir)
    if not intervals:
        return []
    windows: list[tuple[float, float]] = []
    for s_ms, e_ms in intervals:
        outside_len_s = (e_ms - s_ms) / 1000.0
        if outside_len_s < min_outside_s:
            continue
        usable = outside_len_s - 2.0 * margin_s
        if usable < min_window_s:
            continue
        target = max(min_window_s, min(max_window_s, occupancy * usable))
        # Centre the dropout in the outside interval.
        outside_start_s = (s_ms - t0_ms) / 1000.0
        outside_end_s = (e_ms - t0_ms) / 1000.0
        centre = 0.5 * (outside_start_s + outside_end_s)
        win_start = centre - 0.5 * target
        win_start = max(win_start, outside_start_s + margin_s)
        win_end = win_start + target
        if win_end > outside_end_s - margin_s:
            win_end = outside_end_s - margin_s
            target = win_end - win_start
        if target < min_window_s:
            continue
        windows.append((win_start, target))
    return windows


def _windows_longest_outsides(
    src_dir: Path, t0_ms: int,
    durations_s: list[float],
    margin_s: float = 2.0,
) -> list[tuple[float, float]]:
    """Place each requested gap duration inside one of the longest
    available outside intervals (longest gap → longest interval).
    Falls back to fractional-position windows when ``gt.csv`` is absent
    or doesn't have enough room.
    """
    intervals = _read_outside_intervals_ms(src_dir)
    if not intervals:
        return []
    sorted_outs = sorted(
        intervals, key=lambda x: x[1] - x[0], reverse=True,
    )
    durations_sorted = sorted(durations_s, reverse=True)
    windows: list[tuple[float, float]] = []
    for dur in durations_sorted:
        # Find the first outside interval that still fits this duration
        # plus margins, and that hasn't already been used.
        chosen = None
        for idx, (s_ms, e_ms) in enumerate(sorted_outs):
            if (e_ms - s_ms) / 1000.0 < dur + 2.0 * margin_s:
                continue
            chosen = idx
            break
        if chosen is None:
            continue
        s_ms, e_ms = sorted_outs.pop(chosen)
        outside_start_s = (s_ms - t0_ms) / 1000.0
        outside_end_s = (e_ms - t0_ms) / 1000.0
        centre = 0.5 * (outside_start_s + outside_end_s)
        win_start = centre - 0.5 * dur
        win_start = max(win_start, outside_start_s + margin_s)
        win_end = win_start + dur
        if win_end > outside_end_s - margin_s:
            win_end = outside_end_s - margin_s
            dur = win_end - win_start
        windows.append((win_start, dur))
    # Sort by start time so the printed list is in chronological order.
    windows.sort(key=lambda w: w[0])
    return windows


def _default_multi_windows(
    duration_s: float,
) -> list[tuple[float, float]]:
    """Pick 5 dropout windows of varying lengths spread across the trace.

    Lengths grow from short (3 s) to long (30 s) so the operator can
    eyeball whether the loader / UI handle both regimes. Anchored at
    fractions of the trace so it doesn't matter how long the recording
    is — the script stays useful on a 30 s clip and a 1 h session
    alike.
    """
    fractions = [0.10, 0.25, 0.45, 0.65, 0.85]
    lengths_s = [3.0, 7.0, 15.0, 22.0, 30.0]
    out: list[tuple[float, float]] = []
    for frac, length in zip(fractions, lengths_s):
        start = max(0.0, frac * duration_s - length / 2.0)
        if start + length >= duration_s:
            continue
        out.append((start, length))
    return out


def _apply_thinning(
    df: pd.DataFrame, gap_start_s: float, gap_len_s: float,
    target_spacing_s: float,
) -> pd.DataFrame:
    """Keep only one sample every ``target_spacing_s`` inside the window.

    With ``target_spacing_s`` > ``GAP_THRESHOLD_S`` the resulting frame has
    consecutive-sample gaps wider than the loader's threshold, so the
    loader will split the timeline at every kept sample inside the window.
    """
    if target_spacing_s <= GAP_THRESHOLD_S:
        raise ValueError(
            f"target_spacing_s ({target_spacing_s}) must exceed GAP_THRESHOLD_S "
            f"({GAP_THRESHOLD_S}) for the thinning mode to actually create gaps."
        )
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    t0 = int(ts[0])
    lo = t0 + int(round(gap_start_s * 1000.0))
    hi = lo + int(round(gap_len_s * 1000.0))
    in_window = (ts >= lo) & (ts < hi)
    spacing_ms = int(round(target_spacing_s * 1000.0))
    # Keep samples outside the window untouched. Inside the window keep
    # the first sample, then any sample whose timestamp is at least
    # ``spacing_ms`` past the previously-kept in-window sample.
    keep = ~in_window
    last_kept_ms = -10**18
    for i, t in enumerate(ts):
        if not in_window[i]:
            continue
        if int(t) - last_kept_ms >= spacing_ms:
            keep[i] = True
            last_kept_ms = int(t)
    return df.loc[keep].reset_index(drop=True)


def _apply_halving(df: pd.DataFrame) -> pd.DataFrame:
    """Drop every other row across the entire frame."""
    return df.iloc[::2].reset_index(drop=True)


def _apply_halving_different_parts(
    df: pd.DataFrame,
    parts_config: list[dict],
    dead_regions_s: list[tuple[float, float]],
) -> pd.DataFrame:
    """Resample each part of the timeline onto a uniform grid at its
    assigned target rate, and emit no rows inside a dead region.

    ``parts_config`` is a list of dicts ``{"start_s", "end_s",
    "target_hz"}`` covering the timeline (relative to the frame's first
    timestamp). For every part:

    * Generate a uniform grid of timestamps at exactly ``target_hz``
      between ``start_s`` and ``end_s``.
    * Interpolate every numeric column linearly from the original
      ``timestamp_ms`` onto the new grid (``np.interp``).
    * Non-numeric columns (e.g. ``exp_name``) are propagated by
      nearest-neighbour lookup.

    Using interpolation rather than integer-``k`` decimation gives an
    exact target rate even when ``target_hz`` is close to (or above)
    the source's native rate — important for the test fixture, where
    each part should land at a *distinct* rate so the loader's
    per-interval 50 Hz upsampler is exercised on a different ratio for
    every part.

    ``dead_regions_s`` is a list of ``(start_s, end_s)`` windows that
    are skipped entirely; they become the "no data" gaps the loader
    splits the timeline at.
    """
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df
    ts = df["timestamp_ms"].astype("int64").to_numpy()
    if not len(ts):
        return df
    t0 = int(ts[0])

    value_cols = [
        c for c in df.columns
        if c != "timestamp_ms" and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [
        c for c in df.columns
        if c != "timestamp_ms" and c not in value_cols
    ]

    # Sort dead regions so we can subtract them from each part's span.
    dead_ms = [
        (t0 + int(round(ds * 1000.0)), t0 + int(round(de * 1000.0)))
        for ds, de in dead_regions_s
    ]
    dead_ms.sort()

    chunks: list[pd.DataFrame] = []
    for part in parts_config:
        s_ms = t0 + int(round(float(part["start_s"]) * 1000.0))
        e_ms = t0 + int(round(float(part["end_s"]) * 1000.0))
        target_hz = float(part["target_hz"])
        if target_hz <= 0 or e_ms <= s_ms:
            continue
        # Trim the part by any overlapping dead region — we want the
        # uniform grid to *not* include samples inside a dead band.
        sub_spans = [(s_ms, e_ms)]
        for d_lo, d_hi in dead_ms:
            new_spans: list[tuple[int, int]] = []
            for a, b in sub_spans:
                if d_hi <= a or d_lo >= b:
                    new_spans.append((a, b))
                    continue
                if d_lo > a:
                    new_spans.append((a, d_lo))
                if d_hi < b:
                    new_spans.append((d_hi, b))
            sub_spans = new_spans
            if not sub_spans:
                break

        for span_lo, span_hi in sub_spans:
            duration_s = (span_hi - span_lo) / 1000.0
            sub_mask = (ts >= span_lo) & (ts < span_hi)
            sub_ts = ts[sub_mask].astype(np.float64)
            if sub_ts.size < 2:
                continue
            native_hz = (sub_ts.size - 1) / max(duration_s, 1e-9)
            n_target = int(round(duration_s * target_hz))
            # Downsample-only: never fabricate more samples than the
            # source has. Sensors that natively run below the target rate
            # (e.g. GPS at ~0.5 Hz, PRS at ~25 Hz) just keep their
            # original samples for this span — upsampling them here
            # would invent data that didn't exist.
            if target_hz >= native_hz or n_target >= sub_ts.size:
                chunk_df = df.loc[sub_mask].reset_index(drop=True)
                if not chunk_df.empty:
                    chunks.append(chunk_df)
                continue
            if n_target < 2:
                continue
            new_ts = np.linspace(span_lo, span_hi, n_target).astype(np.int64)
            chunk = {"timestamp_ms": new_ts}
            for c in value_cols:
                col = df[c].to_numpy(dtype=float)[sub_mask]
                m = ~np.isnan(col)
                if not m.any():
                    chunk[c] = np.full(n_target, np.nan)
                    continue
                chunk[c] = np.interp(
                    new_ts.astype(np.float64), sub_ts[m], col[m],
                )
            for c in other_cols:
                col = df[c].to_numpy()[sub_mask]
                if col.size == 0:
                    chunk[c] = np.array([None] * n_target, dtype=object)
                    continue
                idx = np.searchsorted(sub_ts, new_ts.astype(np.float64))
                idx = np.clip(idx, 0, col.size - 1)
                chunk[c] = col[idx]
            chunks.append(pd.DataFrame(chunk))

    if not chunks:
        return df.iloc[0:0].copy()
    out = pd.concat(chunks, ignore_index=True)
    out = out.sort_values("timestamp_ms").reset_index(drop=True)
    return out


def _halving_different_parts_config(
    src_dir: Path, t0_ms: int, t_end_ms: int,
    rate_pool: list[int],
    dead_len_s: float = 8.0,
    margin_s: float = 2.0,
    seed: int = 0,
) -> tuple[list[dict], list[tuple[float, float]]]:
    """Build (parts_config, dead_regions) from the experiment's gt.csv.

    Splits the timeline into one part per (rate-pool entry) such that
    every part contains at least one ``up``/``down`` ride. Dead regions
    sit inside outside intervals between the rides at part boundaries
    so no GT ride is bisected.

    Returns ``([], [])`` when ``gt.csv`` doesn't have enough rides /
    outside intervals to host the requested split — caller should fall
    back to a uniform split or error out.
    """
    import random

    rng = random.Random(seed)

    gt_path = src_dir / GT_CSV
    if not gt_path.exists():
        return [], []
    gt = pd.read_csv(gt_path).sort_values("start_ms").reset_index(drop=True)
    rides = gt.loc[
        gt["type"].astype(str).str.lower().isin(("up", "down"))
    ].reset_index(drop=True)
    if rides.empty or len(rate_pool) < 2:
        return [], []

    n_parts = min(len(rate_pool), max(2, len(rides)))
    # How many rides go into each part. The last part absorbs any
    # remainder so every part holds ≥1 ride.
    base = max(1, len(rides) // n_parts)

    boundaries: list[tuple[int, int]] = []  # dead-region (start_ms, end_ms)
    for i in range(1, n_parts):
        last_ride_idx = min(i * base - 1, len(rides) - 2)
        if last_ride_idx < 0 or last_ride_idx >= len(rides) - 1:
            break
        last_end_ms = int(rides.iloc[last_ride_idx]["end_ms"])
        next_start_ms = int(rides.iloc[last_ride_idx + 1]["start_ms"])
        gap_ms = next_start_ms - last_end_ms
        # Need room for the dead region plus 2× margin so the dropout
        # doesn't bleed into the rides on either side.
        needed_ms = int((dead_len_s + 2.0 * margin_s) * 1000.0)
        if gap_ms < needed_ms:
            continue
        centre_ms = (last_end_ms + next_start_ms) // 2
        half = int(dead_len_s * 1000.0) // 2
        boundaries.append((centre_ms - half, centre_ms + half))

    if len(boundaries) < n_parts - 1:
        # Too few usable outside intervals; reduce the part count so the
        # config still spans the timeline correctly.
        n_parts = len(boundaries) + 1
        if n_parts < 2:
            return [], []

    parts_ms: list[tuple[int, int]] = []
    cursor = t0_ms
    for ds, de in boundaries[: n_parts - 1]:
        parts_ms.append((cursor, ds))
        cursor = de
    parts_ms.append((cursor, t_end_ms))

    pool = list(rate_pool)
    rng.shuffle(pool)
    rates = [pool[i % len(pool)] for i in range(len(parts_ms))]

    parts_config = [
        {
            "start_s": (s_ms - t0_ms) / 1000.0,
            "end_s": (e_ms - t0_ms) / 1000.0,
            "target_hz": int(rate),
        }
        for (s_ms, e_ms), rate in zip(parts_ms, rates)
    ]
    dead_regions = [
        ((ds - t0_ms) / 1000.0, (de - t0_ms) / 1000.0)
        for ds, de in boundaries[: n_parts - 1]
    ]
    return parts_config, dead_regions


def _corrupt_sensor(
    df: pd.DataFrame, mode: str, gap_start_s: float, gap_len_s: float,
    spacing_s: float, multi_windows: list[tuple[float, float]] | None = None,
    parts_config: list[dict] | None = None,
    dead_regions_s: list[tuple[float, float]] | None = None,
) -> pd.DataFrame:
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df
    if mode == "dropout":
        return _apply_dropout(df, gap_start_s, gap_len_s)
    if mode == "thinning":
        return _apply_thinning(df, gap_start_s, gap_len_s, spacing_s)
    if mode == "halving":
        return _apply_halving(df)
    if mode in ("multi_dropout", "between_rides", "long_dropout"):
        windows = multi_windows or []
        if not windows and mode == "multi_dropout":
            ts = df["timestamp_ms"].astype("int64").to_numpy()
            duration_s = float(ts[-1] - ts[0]) / 1000.0
            windows = _default_multi_windows(duration_s)
        return _apply_multi_dropout(df, windows)
    if mode == "halvingDiffrentParts":
        return _apply_halving_different_parts(
            df, parts_config or [], dead_regions_s or [],
        )
    raise ValueError(f"unknown mode: {mode!r} (expected one of {MODES})")


def _expected_intervals(
    mode: str, n_sensors_with_data: int, n_windows: int = 0,
) -> str:
    """Human description of what the loader should report after corruption."""
    if mode in ("dropout", "thinning"):
        return f"≥2 valid intervals per sensor ({n_sensors_with_data} sensors with data)"
    if mode == "halving":
        return f"1 valid interval per sensor ({n_sensors_with_data} sensors with data)"
    if mode == "multi_dropout":
        return f"≥6 valid intervals per sensor ({n_sensors_with_data} sensors with data)"
    if mode in ("between_rides", "long_dropout"):
        if n_windows:
            return (
                f"{n_windows + 1} valid intervals per sensor "
                f"({n_sensors_with_data} sensors with data)"
            )
        return f"≥2 valid intervals per sensor ({n_sensors_with_data} sensors with data)"
    if mode == "halvingDiffrentParts":
        if n_windows:
            return (
                f"{n_windows + 1} valid intervals per sensor "
                "(one per rate band; each upsampled to 50 Hz on load) "
                f"({n_sensors_with_data} sensors with data)"
            )
        return (
            f"≥2 valid intervals per sensor (per-band rate, upsampled to "
            f"50 Hz on load); {n_sensors_with_data} sensors with data"
        )
    return "?"


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("exp_name", help="Experiment folder under structuredData/data/")
    ap.add_argument(
        "--mode", choices=MODES, required=True,
        help="Which corruption to apply.",
    )
    ap.add_argument(
        "--gap-start-s", type=float, default=60.0,
        help="Where to place the corruption window, seconds from t0. "
             "(Used by dropout / thinning; ignored by halving.)",
    )
    ap.add_argument(
        "--gap-len-s", type=float, default=5.0,
        help="Length of the corruption window in seconds.",
    )
    ap.add_argument(
        "--thinning-spacing-s", type=float, default=1.5,
        help="In thinning mode, how far apart to space the surviving "
             "samples inside the window. Must exceed GAP_THRESHOLD_S "
             f"(={GAP_THRESHOLD_S}).",
    )
    ap.add_argument(
        "--multi-window", action="append", default=[], metavar="START_S,LEN_S",
        help="In multi_dropout / between_rides / long_dropout modes, add "
             "an explicit (start_s, len_s) window. May be passed multiple "
             "times. When omitted, the script picks reasonable defaults "
             "based on the mode (see help text for each mode).",
    )
    ap.add_argument(
        "--margin-s", type=float, default=2.0,
        help="In between_rides / long_dropout, leave this much breathing "
             "room between the dropout edges and the adjacent rides "
             "(seconds).",
    )
    ap.add_argument(
        "--occupancy", type=float, default=0.55,
        help="In between_rides, the fraction of each outside interval's "
             "usable middle to fill with a dropout (0–1).",
    )
    ap.add_argument(
        "--max-window-s", type=float, default=90.0,
        help="In between_rides, hard cap on a single dropout's length "
             "even when the surrounding outside interval is much longer.",
    )
    ap.add_argument(
        "--long-durations-s", type=str, default="90,150",
        help="In long_dropout mode, comma-separated list of dropout "
             "lengths (seconds) to place in the longest outside intervals. "
             "Default '90,150' produces a 90 s and a 150 s gap.",
    )
    ap.add_argument(
        "--rate-pool", type=str, default="10,20,30,40,70",
        help="In halvingDiffrentParts mode, comma-separated list of "
             "target Hz values to assign to each part (one part per "
             "entry, randomly shuffled). Rates ≥ the source's native "
             "rate are kept as-is (no decimation). Default "
             "'10,20,30,40,70' yields 5 parts.",
    )
    ap.add_argument(
        "--dead-len-s", type=float, default=8.0,
        help="In halvingDiffrentParts mode, length of each dead "
             "(no-data) region between parts in seconds. The dead "
             "region is centred in the outside interval between the "
             "rides at each part boundary so no GT ride is bisected.",
    )
    ap.add_argument(
        "--seed", type=int, default=0,
        help="Seed for the rate-pool shuffle in halvingDiffrentParts.",
    )
    ap.add_argument(
        "--out-suffix", default=None,
        help="Override the destination folder suffix. Default is "
             "'__corrupted_<mode>'.",
    )
    args = ap.parse_args()

    # Parse explicit multi-dropout windows from --multi-window flags.
    multi_windows: list[tuple[float, float]] = []
    for raw in args.multi_window:
        try:
            a, b = raw.split(",")
            multi_windows.append((float(a), float(b)))
        except ValueError:
            ap.error(
                f"--multi-window expects 'START_S,LEN_S' (got {raw!r})"
            )

    src_dir = STRUCTURED_DATA_DIR / args.exp_name
    if not src_dir.is_dir():
        ap.error(f"experiment not found: {src_dir}")

    # For modes that need to read gt.csv to pick gap windows, derive the
    # windows here once (the helpers below need t0_ms, which we read off
    # the first sensor CSV that exists).
    if args.mode in ("between_rides", "long_dropout") and not multi_windows:
        ref_sensor_path = next(
            (src_dir / f"{name}.csv" for name in ("ACC", "PRS", "GYR")
             if (src_dir / f"{name}.csv").exists()),
            None,
        )
        if ref_sensor_path is None:
            ap.error(
                f"could not find a reference sensor CSV in {src_dir} "
                "(need ACC.csv, PRS.csv, or GYR.csv to anchor t0_ms)"
            )
        ref_df = pd.read_csv(ref_sensor_path)
        if ref_df.empty or "timestamp_ms" not in ref_df.columns:
            ap.error(
                f"reference sensor CSV {ref_sensor_path} has no "
                "timestamp_ms column or is empty"
            )
        t0_ms_ref = int(ref_df["timestamp_ms"].iloc[0])
        if args.mode == "between_rides":
            multi_windows = _windows_between_rides(
                src_dir, t0_ms_ref,
                margin_s=args.margin_s,
                occupancy=args.occupancy,
                max_window_s=args.max_window_s,
            )
            if not multi_windows:
                ap.error(
                    f"between_rides: no usable outside intervals in "
                    f"{src_dir / GT_CSV} (need at least one ≥6 s outside "
                    "row). Use --multi-window to specify windows manually."
                )
        else:  # long_dropout
            try:
                long_durs = [float(x) for x in args.long_durations_s.split(",") if x.strip()]
            except ValueError:
                ap.error(
                    f"--long-durations-s expects comma-separated floats "
                    f"(got {args.long_durations_s!r})"
                )
            multi_windows = _windows_longest_outsides(
                src_dir, t0_ms_ref, long_durs, margin_s=args.margin_s,
            )
            if not multi_windows:
                ap.error(
                    "long_dropout: no outside intervals are long enough "
                    f"to host any of the requested durations "
                    f"{long_durs}. Try --long-durations-s with smaller "
                    "values."
                )
        print(f"Picked {len(multi_windows)} window(s) from gt.csv:")
        for s, ln in multi_windows:
            print(f"  start={s:>7.1f}s  length={ln:>6.1f}s")
        print()

    # halvingDiffrentParts needs a different config (parts-with-rates +
    # dead regions). Compute it here once, before iterating sensors.
    parts_config: list[dict] = []
    dead_regions_s: list[tuple[float, float]] = []
    if args.mode == "halvingDiffrentParts":
        ref_sensor_path = next(
            (src_dir / f"{name}.csv" for name in ("ACC", "PRS", "GYR")
             if (src_dir / f"{name}.csv").exists()),
            None,
        )
        if ref_sensor_path is None:
            ap.error(
                f"halvingDiffrentParts needs a reference sensor CSV in "
                f"{src_dir} (ACC.csv / PRS.csv / GYR.csv) to anchor the "
                "timeline."
            )
        ref_df = pd.read_csv(ref_sensor_path)
        if ref_df.empty or "timestamp_ms" not in ref_df.columns:
            ap.error(
                f"reference sensor CSV {ref_sensor_path} has no "
                "timestamp_ms column or is empty"
            )
        t0_ms_ref = int(ref_df["timestamp_ms"].iloc[0])
        t_end_ms_ref = int(ref_df["timestamp_ms"].iloc[-1])
        try:
            rate_pool = [
                int(x) for x in args.rate_pool.split(",") if x.strip()
            ]
        except ValueError:
            ap.error(
                f"--rate-pool expects comma-separated integers "
                f"(got {args.rate_pool!r})"
            )
        if len(rate_pool) < 2:
            ap.error(
                "--rate-pool needs at least two rates so the timeline "
                "can be split into multiple parts."
            )
        parts_config, dead_regions_s = _halving_different_parts_config(
            src_dir, t0_ms_ref, t_end_ms_ref,
            rate_pool=rate_pool,
            dead_len_s=args.dead_len_s,
            margin_s=args.margin_s,
            seed=args.seed,
        )
        if not parts_config:
            ap.error(
                "halvingDiffrentParts: gt.csv didn't have enough usable "
                "rides / outside intervals to build the requested split. "
                "Try a smaller --rate-pool or a shorter --dead-len-s."
            )
        print(f"Picked {len(parts_config)} parts (with dead regions between):")
        for i, part in enumerate(parts_config):
            ds = dead_regions_s[i - 1] if i > 0 else None
            if ds is not None:
                print(
                    f"   dead   {ds[0]:>7.1f}s → {ds[1]:>7.1f}s  "
                    f"({ds[1] - ds[0]:.1f}s)"
                )
            print(
                f"  part {i:>2d}  {part['start_s']:>7.1f}s → "
                f"{part['end_s']:>7.1f}s  @ {part['target_hz']:>3d} Hz"
            )
        print()

    suffix = args.out_suffix or f"__corrupted_{args.mode}"
    dst_name = f"{args.exp_name}{suffix}"
    dst_dir = STRUCTURED_DATA_DIR / dst_name
    if dst_dir.exists():
        ap.error(
            f"destination already exists: {dst_dir} "
            "(remove it first or pick another --out-suffix)"
        )
    dst_dir.mkdir(parents=True)

    # Copy the non-sensor artefacts byte-for-byte.
    n_sensors = 0
    for path in sorted(src_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix == ".csv" and path.stem in SENSOR_COLUMNS:
            df = pd.read_csv(path)
            n_before = len(df)
            df = _corrupt_sensor(
                df, args.mode, args.gap_start_s, args.gap_len_s,
                args.thinning_spacing_s,
                multi_windows=multi_windows or None,
                parts_config=parts_config or None,
                dead_regions_s=dead_regions_s or None,
            )
            df.to_csv(dst_dir / path.name, index=False)
            n_after = len(df)
            print(
                f"  {path.name:>12s}  {n_before:>7d} → {n_after:>7d} rows "
                f"({n_after / max(n_before, 1):.0%})"
            )
            n_sensors += 1
        elif path.name == METADATA_CSV:
            # Rewrite exp_name so the per-experiment metadata matches the
            # new folder; everything else carries over verbatim.
            mdf = pd.read_csv(path)
            if len(mdf):
                mdf.loc[mdf.index[0], "exp_name"] = dst_name
            cols = [c for c in METADATA_COLUMNS if c in mdf.columns]
            mdf.to_csv(dst_dir / path.name, index=False, columns=cols or None)
        else:
            shutil.copy2(path, dst_dir / path.name)

    n_windows_for_report = (
        len(dead_regions_s) if args.mode == "halvingDiffrentParts"
        else len(multi_windows)
    )
    print()
    print(f"Wrote corrupted experiment to: {dst_dir}")
    print(f"Mode: {args.mode}")
    print(
        "Expected loader output: "
        + _expected_intervals(args.mode, n_sensors, n_windows=n_windows_for_report)
    )
    print()
    print("Validate:")
    print(f"  python -c \"from src.data.loader import getExperimentData; \\")
    print(f"    _, gt, _ = getExperimentData('{dst_name}', use_cache=True); \\")
    print(f"    print(gt.attrs['valid_intervals_per_sensor'])\"")
    return 0


if __name__ == "__main__":
    sys.exit(main())
