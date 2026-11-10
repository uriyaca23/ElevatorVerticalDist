"""Visual + numeric sanity check of the Task 2 time calibration.

For each non-Pixel experiment that has a Pixel reference, overlays the
phone's ACC magnitude on Pixel's ACC magnitude inside each Pixel-tagged
``up``/``down`` segment. Two outputs per experiment:

* ``phone_time_verify.png`` — 3×2 grid of evenly-sampled tagged segments
  showing Pixel (blue) vs. phone (green) ACC magnitude with the
  Pixel-tagged window shaded red. After a good calibration the two traces
  track each other inside the red window and the phone's takeoff /
  landing pulses sit on top of Pixel's.
* ``phone_time_verify.csv`` — per-segment residual: mean absolute
  difference of ACC magnitudes inside the tagged window, peak lag of a
  *fine-grained* xcorr (±500 ms) as a "how much drift is left" number,
  and a simple pulse-alignment score.

Plus a top-level summary at
``structuredData/phone_time_verify_summary.csv`` with per-experiment
medians — easy to scan for the calibration jobs that still look wrong.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

from ..loader.constants import (
    GT_CSV,
    STRUCTURED_DATA_DIR,
    STRUCTURED_ROOT,
)
from .phone_time_calibration import (
    _acc_magnitude,
    _acc_magnitude_in_window,
    _xcorr_segment_offset,
    find_pixel_reference_exp,
)

VERIFY_PLOT = "phone_time_verify.png"
VERIFY_CSV = "phone_time_verify.csv"
SUMMARY_CSV = "phone_time_verify_summary.csv"


def _fine_residual_offset(
    pixel_acc: pd.DataFrame, phone_acc: pd.DataFrame,
    s_ms: int, e_ms: int, fine_max_lag_ms: int = 500, grid_hz: float = 50.0,
) -> tuple[int, float]:
    """How much residual drift is left inside a single tagged segment?

    Re-runs xcorr in a tight ±500 ms window. If calibration worked, the
    answer should be a few hundred ms at most.
    """
    step_ms = int(round(1000.0 / grid_hz))
    max_lag_samples = int(round(fine_max_lag_ms / step_ms))
    p_grid = np.arange(s_ms, e_ms + step_ms, step_ms, dtype=np.int64)
    q_grid = np.arange(
        s_ms - max_lag_samples * step_ms,
        e_ms + max_lag_samples * step_ms + step_ms,
        step_ms, dtype=np.int64,
    )
    p_mag = _acc_magnitude_in_window(pixel_acc, s_ms, e_ms, p_grid)
    q_mag = _acc_magnitude_in_window(
        phone_acc, s_ms - fine_max_lag_ms, e_ms + fine_max_lag_ms, q_grid,
    )
    if np.std(p_mag) < 0.05 or np.std(q_mag) < 0.05:
        return 0, 0.0
    lag, score = _xcorr_segment_offset(p_mag, q_mag, max_lag_samples)
    return -int(lag) * step_ms, float(score)


def _in_window_mae(
    pixel_acc: pd.DataFrame, phone_acc: pd.DataFrame, s_ms: int, e_ms: int,
    grid_hz: float = 50.0,
) -> float:
    """Mean-abs-diff of (temp-mean-removed) |a| inside the segment."""
    step_ms = int(round(1000.0 / grid_hz))
    grid = np.arange(s_ms, e_ms + step_ms, step_ms, dtype=np.int64)
    p = _acc_magnitude_in_window(pixel_acc, s_ms, e_ms, grid)
    q = _acc_magnitude_in_window(phone_acc, s_ms, e_ms, grid)
    if p.size < 2:
        return float("nan")
    p = p - float(np.mean(p))
    q = q - float(np.mean(q))
    return float(np.mean(np.abs(p - q)))


def _plot_overlays(
    exp_dir: Path,
    pixel_acc: pd.DataFrame, phone_acc: pd.DataFrame,
    pixel_gt: pd.DataFrame,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    p_t, p_mag = _acc_magnitude(pixel_acc)
    q_t, q_mag = _acc_magnitude(phone_acc)
    rides = pixel_gt[pixel_gt["type"].astype(str).str.lower().isin(("up", "down"))]
    rides = rides.reset_index(drop=True)
    n = min(6, len(rides))
    if n == 0:
        return
    idxs = np.linspace(0, len(rides) - 1, n, dtype=int)

    nrows = (n + 1) // 2
    fig, axes = plt.subplots(nrows, 2, figsize=(14, 3 * nrows), squeeze=False)
    fig.suptitle(
        f"Pixel vs phone ACC inside Pixel-tagged rides (after calibration) — {exp_dir.name}",
        fontsize=10,
    )
    for k, i in enumerate(idxs):
        ax = axes[k // 2][k % 2]
        row = rides.iloc[int(i)]
        s, e = int(row["start_ms"]), int(row["end_ms"])
        pad = 3_000
        ax.axvspan(0, (e - s) / 1000.0, color="tab:red", alpha=0.10,
                   label="Pixel-tagged ride")
        mask_p = (p_t >= s - pad) & (p_t <= e + pad)
        mask_q = (q_t >= s - pad) & (q_t <= e + pad)
        ax.plot((p_t[mask_p] - s) / 1000.0, p_mag[mask_p],
                color="tab:blue", lw=0.9, alpha=0.9, label="Pixel |a|")
        ax.plot((q_t[mask_q] - s) / 1000.0, q_mag[mask_q],
                color="tab:green", lw=0.9, alpha=0.8, label="phone |a|")
        ax.set_title(f"Segment {int(i)} — {row['type']}  dur={(e-s)/1000:.1f}s",
                     fontsize=9)
        ax.set_xlabel("time (s, rel to segment start)")
        ax.set_ylabel("|a| (m/s²)")
        ax.grid(True, alpha=0.3)
        if k == 0:
            ax.legend(loc="upper right", fontsize=8)

    total = nrows * 2
    for k in range(n, total):
        axes[k // 2][k % 2].axis("off")

    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(exp_dir / VERIFY_PLOT, dpi=120)
    plt.close(fig)


def _verify_one(exp_name: str, pixel_name: str) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    pix_dir = STRUCTURED_DATA_DIR / pixel_name
    pix_acc = pd.read_csv(pix_dir / "ACC.csv")
    phone_acc = pd.read_csv(exp_dir / "ACC.csv")
    pix_gt = pd.read_csv(pix_dir / GT_CSV)

    rides = pix_gt[pix_gt["type"].astype(str).str.lower().isin(("up", "down"))]
    rows: list[dict] = []
    for _, r in rides.iterrows():
        s, e = int(r["start_ms"]), int(r["end_ms"])
        residual_ms, score = _fine_residual_offset(pix_acc, phone_acc, s, e)
        mae = _in_window_mae(pix_acc, phone_acc, s, e)
        rows.append({
            "start_ms":          s,
            "end_ms":            e,
            "type":              r["type"],
            "duration_s":        round((e - s) / 1000, 2),
            "residual_ms":       residual_ms,
            "fine_xcorr_score":  round(score, 3),
            "in_window_mae":     round(mae, 3),
        })
    seg_df = pd.DataFrame(rows)
    seg_df.to_csv(exp_dir / VERIFY_CSV, index=False)
    _plot_overlays(exp_dir, pix_acc, phone_acc, pix_gt)

    if not rows:
        return {"exp_name": exp_name, "pixel_ref": pixel_name, "n": 0}

    res_arr = np.asarray([r["residual_ms"] for r in rows], dtype=float)
    return {
        "exp_name":            exp_name,
        "pixel_ref":           pixel_name,
        "n_segments":          len(rows),
        "median_residual_ms":  int(np.median(res_arr)),
        "max_residual_ms":     int(np.max(np.abs(res_arr))),
        "median_mae":          round(float(np.median([r["in_window_mae"] for r in rows])), 3),
        "median_score":        round(float(np.median([r["fine_xcorr_score"] for r in rows])), 3),
    }


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = sys.argv[1:]
    if args:
        exp_names = args
    else:
        exp_names = sorted(
            p.name for p in STRUCTURED_DATA_DIR.iterdir()
            if p.is_dir() and (p / "ACC.csv").exists()
        )

    rows: list[dict] = []
    for name in exp_names:
        pix = find_pixel_reference_exp(name)
        if pix is None:
            continue
        try:
            rows.append(_verify_one(name, pix))
        except Exception as e:
            rows.append({
                "exp_name": name, "pixel_ref": pix,
                "status": f"failed: {type(e).__name__}: {e}",
            })

    df = pd.DataFrame(rows)
    out_path = STRUCTURED_ROOT / "test_results" / SUMMARY_CSV
    df.to_csv(out_path, index=False)
    print(f"[verify] summary: {out_path}")
    print()
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
