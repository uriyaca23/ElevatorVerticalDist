"""Brute-force MAE residual sweep — a more robust alternative to the
per-segment xcorr when individual segments hit spurious peaks.

For each non-Pixel experiment, sweeps candidate additional shifts in
``[-MAX_MS, +MAX_MS]`` at a fine step, and for each candidate measures
the *summed* mean-absolute-difference of high-passed ACC magnitudes
across every Pixel-tagged up/down segment. The minimum-MAE offset is
the global best alignment — harder to fool than per-segment xcorr
because a spurious local peak in one segment gets out-voted by the
rest.

Only shifts with both |offset| >= MIN_SHIFT_MS and total MAE
improvement >= MIN_IMPROVEMENT are applied. Safe-idempotent: re-running
should show zero shifts needed.

Run with::

    python -m src.data.dataset_cleanup.mae_residual_sweep              # dry-run
    python -m src.data.dataset_cleanup.mae_residual_sweep --apply      # apply shifts + re-run Pixel-ref
"""

from __future__ import annotations

import subprocess
import sys

import numpy as np
import pandas as pd

from ..loader.constants import STRUCTURED_DATA_DIR, STRUCTURED_ROOT
from .phone_time_calibration import (
    _acc_magnitude_in_window,
    _shift_all_csvs,
    find_pixel_reference_exp,
)

MAX_MS = 3000
STEP_MS = 50
MIN_SHIFT_MS = 150
MIN_IMPROVEMENT = 0.03
GRID_HZ = 50.0


def _experiment_mae_curve(
    pix_acc: pd.DataFrame, phone_acc: pd.DataFrame, rides: pd.DataFrame,
    offsets_ms: np.ndarray,
) -> np.ndarray:
    step_ms = int(round(1000.0 / GRID_HZ))
    sum_mae = np.zeros(len(offsets_ms), dtype=float)
    for o_i, o in enumerate(offsets_ms):
        total = 0.0
        n = 0
        for _, r in rides.iterrows():
            s, e = int(r["start_ms"]), int(r["end_ms"])
            grid = np.arange(s, e + step_ms, step_ms, dtype=np.int64)
            p = _acc_magnitude_in_window(pix_acc, s, e, grid)
            q_grid = grid - int(o)
            q = _acc_magnitude_in_window(phone_acc, s - int(o), e - int(o), q_grid)
            if p.size < 2 or q.size != p.size:
                continue
            p = p - float(np.mean(p))
            q = q - float(np.mean(q))
            total += float(np.mean(np.abs(p - q)))
            n += 1
        sum_mae[o_i] = total / max(n, 1)
    return sum_mae


def _process_one(exp_name: str, pixel_name: str, apply: bool) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    pix_dir = STRUCTURED_DATA_DIR / pixel_name
    pix_acc = pd.read_csv(pix_dir / "ACC.csv")
    phone_acc = pd.read_csv(exp_dir / "ACC.csv")
    pix_gt = pd.read_csv(pix_dir / "gt.csv")
    rides = pix_gt[pix_gt["type"].astype(str).str.lower().isin(("up", "down"))]
    if len(rides) < 3:
        return {"exp_name": exp_name, "status": "too_few_rides"}

    offsets = np.arange(-MAX_MS, MAX_MS + 1, STEP_MS, dtype=int)
    mae = _experiment_mae_curve(pix_acc, phone_acc, rides, offsets)

    best_i = int(np.argmin(mae))
    best_off = int(offsets[best_i])
    best_mae = float(mae[best_i])
    mae_at_zero = float(mae[np.argmin(np.abs(offsets))])
    improvement = mae_at_zero - best_mae

    will_apply = (
        apply
        and abs(best_off) >= MIN_SHIFT_MS
        and improvement >= MIN_IMPROVEMENT
    )
    if will_apply:
        _shift_all_csvs(exp_dir, best_off)

    return {
        "exp_name":         exp_name,
        "pixel_ref":        pixel_name,
        "best_offset_ms":   best_off,
        "mae_at_zero":      round(mae_at_zero, 4),
        "mae_at_best":      round(best_mae, 4),
        "improvement":      round(improvement, 4),
        "applied":          str(will_apply),
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
        rows.append(_process_one(name, pix, apply))

    df = pd.DataFrame(rows)
    out_path = STRUCTURED_ROOT / "test_results" / "mae_residual_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"[mae-sweep] summary: {out_path}")
    print()
    print(df.to_string(index=False))

    if apply and any(r.get("applied") == "True" for r in rows):
        print("\n[mae-sweep] running apply_pixel_reference + verify + save_test_results...")
        subprocess.run(
            [sys.executable, "-m", "src.data.dataset_cleanup.apply_pixel_reference", "--apply"],
            check=False, stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "-m", "src.data.dataset_cleanup.verify_calibration"],
            check=False, stdout=subprocess.DEVNULL,
        )
        subprocess.run(
            [sys.executable, "-m", "src.data.dataset_cleanup.save_test_results"],
            check=False, stdout=subprocess.DEVNULL,
        )
        print("[mae-sweep] done")


if __name__ == "__main__":
    main()
