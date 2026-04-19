"""Simple ZUPT-style height estimates per tagged elevator segment, run across
every phone in every experiment. Produces a summary that makes calibration
quality easy to scan:

* For each Pixel-tagged up/down segment, compute a naive vertical-height
  estimate on every phone (Pixel + other phones in the same experiment) by
  integrating ``|a| - mean(|a|)`` twice inside the segment window. Uriya's
  rule: on clean segments (most of what we have), a correctly-aligned phone
  should get an estimate close to Pixel's + close to the gramushka-snapped
  ground-truth ``height_diff_m``.

* If a phone's estimate disagrees strongly with the other phones that share
  the same tagged segment, either the calibration is wrong (the segment
  boundaries are cutting off the start or end pulse on that phone) or the
  ACC quality is bad on that phone for that ride.

Output:
  - ``structuredData/zupt_segments.csv``: one row per (experiment, segment_idx)
    with phone's naive |Δh| estimate, Pixel's estimate at the same tagged
    window, and the gramushka-snapped gt |Δh|.
  - ``structuredData/zupt_experiment_summary.csv``: per-experiment medians
    (absolute error phone-vs-Pixel, median error vs gt, max outlier).
"""

from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from ..loader.constants import (
    GT_CSV,
    STRUCTURED_DATA_DIR,
    STRUCTURED_ROOT,
)
from .phone_time_calibration import find_pixel_reference_exp


def _naive_zupt_abs_dh(acc: pd.DataFrame, s_ms: int, e_ms: int) -> float:
    """Rough height-magnitude estimate by integrating ``|a| - mean(|a|)``
    twice across the window. Returns ``|Δh|`` in meters.

    This is intentionally dumb — no gravity projection, no filtering, no
    ZUPT boundary corrections. We're just looking for rough agreement
    across phones carried simultaneously through the same physical ride.
    """
    ts = acc["timestamp_ms"].to_numpy(dtype=np.int64)
    mask = (ts >= s_ms) & (ts < e_ms)
    if mask.sum() < 5:
        return float("nan")
    t_sec = (ts[mask] - ts[mask][0]) / 1000.0
    mag = np.sqrt(
        acc.loc[mask, "x"].to_numpy(dtype=float) ** 2
        + acc.loc[mask, "y"].to_numpy(dtype=float) ** 2
        + acc.loc[mask, "z"].to_numpy(dtype=float) ** 2
    )
    a_lin = mag - float(np.mean(mag))
    v = np.concatenate([[0.0], np.cumsum(0.5 * (a_lin[:-1] + a_lin[1:]) * np.diff(t_sec))])
    h = np.concatenate([[0.0], np.cumsum(0.5 * (v[:-1] + v[1:]) * np.diff(t_sec))])
    return float(abs(h[-1]))


def _abs_dh_from_gt(gt_row: pd.Series) -> float:
    v = gt_row.get("height_diff_m")
    return float("nan") if pd.isna(v) else abs(float(v))


def _phones_in_same_experiment(pixel_name: str) -> list[str]:
    """Every non-Pixel experiment that has a Pixel reference of
    ``pixel_name``. Lets us line up all phones that were recorded
    simultaneously with the Pixel in one row per tagged segment.
    """
    out: list[str] = []
    if not STRUCTURED_DATA_DIR.is_dir():
        return out
    for p in sorted(STRUCTURED_DATA_DIR.iterdir()):
        if not p.is_dir() or "pixel" in p.name.lower():
            continue
        if find_pixel_reference_exp(p.name) == pixel_name:
            out.append(p.name)
    return out


def _pixel_exps() -> list[str]:
    return sorted(
        p.name for p in STRUCTURED_DATA_DIR.iterdir()
        if p.is_dir() and "pixel" in p.name.lower()
        and (p / "ACC.csv").exists() and (p / GT_CSV).exists()
    )


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    seg_rows: list[dict] = []
    exp_rows: list[dict] = []

    for pixel_name in _pixel_exps():
        pix_dir = STRUCTURED_DATA_DIR / pixel_name
        pix_acc = pd.read_csv(pix_dir / "ACC.csv")
        pix_gt = pd.read_csv(pix_dir / GT_CSV)
        phones = _phones_in_same_experiment(pixel_name)
        phone_acc = {ph: pd.read_csv(STRUCTURED_DATA_DIR / ph / "ACC.csv") for ph in phones}

        rides = pix_gt[pix_gt["type"].astype(str).str.lower().isin(("up", "down"))]
        per_exp_phone_err: dict[str, list[float]] = {ph: [] for ph in phones}

        for i, (_, row) in enumerate(rides.iterrows()):
            s, e = int(row["start_ms"]), int(row["end_ms"])
            gt_abs = _abs_dh_from_gt(row)
            pix_est = _naive_zupt_abs_dh(pix_acc, s, e)
            seg = {
                "pixel_name":   pixel_name,
                "segment_idx":  i,
                "type":         row["type"],
                "duration_s":   round((e - s) / 1000, 2),
                "gt_abs_dh_m":  round(gt_abs, 3),
                "pixel_abs_dh_m": round(pix_est, 3),
            }
            for ph in phones:
                est = _naive_zupt_abs_dh(phone_acc[ph], s, e)
                seg[f"{ph}__abs_dh"] = round(est, 3)
                if not np.isnan(est) and not np.isnan(pix_est):
                    per_exp_phone_err[ph].append(abs(est - pix_est))
            seg_rows.append(seg)

        # One summary row per phone (including the Pixel itself, error=gt-vs-pixel).
        pix_vs_gt_errs: list[float] = []
        for r in seg_rows:
            if r["pixel_name"] == pixel_name and not np.isnan(r["pixel_abs_dh_m"]) and not np.isnan(r["gt_abs_dh_m"]):
                pix_vs_gt_errs.append(abs(r["pixel_abs_dh_m"] - r["gt_abs_dh_m"]))
        exp_rows.append({
            "pixel_name":         pixel_name,
            "phone":              pixel_name + " (SELF)",
            "n_segments":         int(len(pix_vs_gt_errs)),
            "median_err_vs_pixel_m": 0.0,
            "median_err_vs_gt_m": round(float(np.median(pix_vs_gt_errs)) if pix_vs_gt_errs else float("nan"), 3),
            "max_err_vs_pixel_m": 0.0,
        })
        for ph, errs in per_exp_phone_err.items():
            if not errs:
                continue
            exp_rows.append({
                "pixel_name":            pixel_name,
                "phone":                 ph,
                "n_segments":            len(errs),
                "median_err_vs_pixel_m": round(float(np.median(errs)), 3),
                "median_err_vs_gt_m":    float("nan"),
                "max_err_vs_pixel_m":    round(float(np.max(errs)), 3),
            })

    seg_df = pd.DataFrame(seg_rows)
    exp_df = pd.DataFrame(exp_rows)
    seg_path = STRUCTURED_ROOT / "test_results" / "zupt_segments.csv"
    exp_path = STRUCTURED_ROOT / "test_results" / "zupt_experiment_summary.csv"
    seg_df.to_csv(seg_path, index=False)
    exp_df.to_csv(exp_path, index=False)
    print(f"[zupt] wrote {seg_path}")
    print(f"[zupt] wrote {exp_path}")
    print()
    print(exp_df.to_string(index=False))


if __name__ == "__main__":
    main()
