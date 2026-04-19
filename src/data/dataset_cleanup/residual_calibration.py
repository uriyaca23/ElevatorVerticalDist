"""Residual calibration pass. Task 2 + Pixel-ref already landed; the plots
Uriya reviewed show several non-Pixel phones whose ACC pulses still sit
visibly to the right/left of the Pixel-tagged windows — meaning the
coarse xcorr chose a close-but-imperfect peak (or, in the Uriya-Xiaomi
exp1 Millenium case, the PRS-full fallback was off).

This pass:

1. Re-runs the per-elevator-segment ACC xcorr on each phone's *current*
   (already-shifted) CSVs with a 10 s search window and a 0.15 Hz ACC
   high-pass (kills gravity/orientation DC drift that was washing out
   the normalised xcorr).
2. Trims outlier per-segment offsets (anything > 3000 ms off the median).
3. Applies the trimmed median as an additional shift on top of whatever
   is already there — so running this twice is a no-op.

Only experiments whose trimmed-median-residual exceeds ``MIN_SHIFT_MS``
are shifted. Small (<150 ms) residuals are within the xcorr resolution
floor and left alone.

Usage::

    python -m src.data.dataset_cleanup.residual_calibration              # dry-run
    python -m src.data.dataset_cleanup.residual_calibration --apply      # shift CSVs in place
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader.constants import STRUCTURED_DATA_DIR, STRUCTURED_ROOT
from .phone_time_calibration import (
    _per_segment_offsets_ms,
    _shift_all_csvs,
    find_pixel_reference_exp,
)

MIN_SHIFT_MS = 150
OUTLIER_TRIM_MS = 3000
SEARCH_RANGE_MS = 10_000


def _trimmed_median(offsets: list[int]) -> tuple[int, int, int]:
    """Median after removing entries >``OUTLIER_TRIM_MS`` from the raw median.
    Returns ``(trimmed_median_ms, kept, total)``."""
    arr = np.asarray(offsets, dtype=float)
    if arr.size == 0:
        return 0, 0, 0
    raw_med = float(np.median(arr))
    kept_mask = np.abs(arr - raw_med) <= OUTLIER_TRIM_MS
    kept = arr[kept_mask]
    if kept.size == 0:
        return int(round(raw_med)), 0, int(arr.size)
    return int(round(float(np.median(kept)))), int(kept.size), int(arr.size)


def _residual_one(exp_name: str, pixel_name: str, apply: bool) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    pix_dir = STRUCTURED_DATA_DIR / pixel_name
    pix_acc = pd.read_csv(pix_dir / "ACC.csv")
    phone_acc = pd.read_csv(exp_dir / "ACC.csv")
    pix_gt = pd.read_csv(pix_dir / "gt.csv")
    offsets, scores = _per_segment_offsets_ms(
        pix_acc, phone_acc, pix_gt, "acc", max_lag_ms=SEARCH_RANGE_MS,
    )
    trimmed, kept, total = _trimmed_median(offsets)
    will_apply = apply and abs(trimmed) >= MIN_SHIFT_MS and kept >= 3

    if will_apply:
        _shift_all_csvs(exp_dir, trimmed)

    return {
        "exp_name":          exp_name,
        "pixel_ref":         pixel_name,
        "n_segments":        total,
        "n_after_trim":      kept,
        "trimmed_median_ms": trimmed,
        "median_score":      round(float(np.median(scores)) if scores else 0.0, 3),
        "shift_applied_ms":  trimmed if will_apply else 0,
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = sys.argv[1:]
    apply = "--apply" in args
    exp_args = [a for a in args if not a.startswith("--")]

    if exp_args:
        exp_names = exp_args
    else:
        exp_names = sorted(
            p.name for p in STRUCTURED_DATA_DIR.iterdir()
            if p.is_dir() and (p / "ACC.csv").exists()
            and "pixel" not in p.name.lower()
        )

    rows: list[dict] = []
    for name in exp_names:
        pix = find_pixel_reference_exp(name)
        if pix is None:
            continue
        rows.append(_residual_one(name, pix, apply))

    df = pd.DataFrame(rows)
    out_path = STRUCTURED_ROOT / "test_results" / "residual_calibration.csv"
    df.to_csv(out_path, index=False)
    mode = "APPLY" if apply else "DRY-RUN"
    print(f"[residual] mode: {mode} | MIN_SHIFT_MS={MIN_SHIFT_MS}  OUTLIER_TRIM_MS={OUTLIER_TRIM_MS}")
    print(f"[residual] summary: {out_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
