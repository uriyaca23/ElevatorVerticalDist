"""Dry-run comparison: old (pure-barometer) vs. new (gramushka-snapped) Δh.

Reads the current ``gt.csv`` for every experiment that has a populated
``baramoshka.csv`` + ``start_floor`` in ``metadata.csv``, runs the new
:func:`addGTtoSegment` in snap mode, and writes two artifacts:

* ``structuredData/gramushka_dry_run_summary.csv`` — one row per experiment
  (n segments, Δh RMSE old-vs-new, count of ambiguous snaps, snap mode).
* ``structuredData/gramushka_dry_run_segments.csv`` — one row per segment
  with old Δh, new Δh, snap distance, ambiguity flag.

This does NOT rewrite any ``gt.csv``. After review, run
``python -m src.data.dataset_cleanup.gramushka_apply`` (next script) to commit the updates.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ..loader.constants import (
    GT_CSV,
    METADATA_CSV,
    STRUCTURED_DATA_DIR,
    STRUCTURED_ROOT,
)
from ..loader.pipeline import (
    SNAP_AMBIGUITY_THRESHOLD_M,
    addGTtoSegment,
    load_baramoshka,
)


def _load_sensors_and_old_gt(
    exp_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Lightweight loader that avoids pulling in the segmentation stack."""
    data: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(exp_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem in ("gt", "metadata", "baramoshka"):
            continue
        data[stem] = pd.read_csv(csv_path)

    gt = pd.read_csv(exp_dir / GT_CSV)

    meta_df = pd.read_csv(exp_dir / METADATA_CSV)
    metadata = (
        {k: ("" if pd.isna(v) else str(v)) for k, v in meta_df.iloc[0].to_dict().items()}
        if len(meta_df) else {}
    )
    return data, gt, metadata


def _analyze_one(exp_name: str) -> tuple[dict, pd.DataFrame]:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    if not exp_dir.is_dir() or not (exp_dir / GT_CSV).exists():
        return {"exp_name": exp_name, "status": "missing"}, pd.DataFrame()

    data, old_gt, metadata = _load_sensors_and_old_gt(exp_dir)
    baramoshka = load_baramoshka(exp_name)

    new_gt = addGTtoSegment(data, old_gt, metadata=metadata, baramoshka=baramoshka)
    mode = new_gt.attrs.get("gramushka_mode", "unknown")
    flags = new_gt.attrs.get("gramushka_snap_flags", [])

    # Some older gt.csv files predate the `height_diff_m` column — treat the
    # old value as 0 in that case so the delta-from-old column still makes sense.
    if "height_diff_m" in old_gt.columns:
        old_dh = old_gt["height_diff_m"].astype(float).fillna(0.0).to_numpy()
    else:
        import numpy as np
        old_dh = np.zeros(len(old_gt), dtype=float)
    new_dh = new_gt["height_diff_m"].astype(float).fillna(0.0).to_numpy()
    diff = new_dh - old_dh
    import numpy as np
    rmse = float(np.sqrt(np.mean(diff ** 2))) if diff.size else 0.0
    max_abs = float(np.max(np.abs(diff))) if diff.size else 0.0

    summary = {
        "exp_name":            exp_name,
        "mode":                mode,
        "n_segments":          len(new_gt),
        "n_flags":             len(flags),
        "delta_rmse_m":        round(rmse, 3),
        "delta_max_abs_m":     round(max_abs, 3),
        "start_floor":         metadata.get("start_floor", ""),
        "temperature_c":       metadata.get("temperature_c", ""),
    }

    rows: list[dict] = []
    flag_set = {f["segment_idx"] for f in flags}
    for i, row in new_gt.iterrows():
        rows.append({
            "exp_name":          exp_name,
            "segment_idx":       i,
            "type":              row["type"],
            "start_ms":          int(row["start_ms"]),
            "end_ms":            int(row["end_ms"]),
            "duration_s":        round((int(row["end_ms"]) - int(row["start_ms"])) / 1000.0, 2),
            "old_dh_m":          round(float(old_dh[i]), 3),
            "new_dh_m":          round(float(new_dh[i]), 3),
            "delta_m":           round(float(new_dh[i] - old_dh[i]), 3),
            "ambiguous":         i in flag_set,
            "signalClearRecording": row.get("signalClearRecording", True),
        })
    flags_detail = {f["segment_idx"]: f for f in flags}
    for r in rows:
        f = flags_detail.get(r["segment_idx"])
        if f:
            r["snap_distance_m"] = round(f["snap_distance_m"], 3)
            r["estimated_end_alt_m"] = round(f["estimated_end_alt_m"], 3)
            r["snapped_floor"] = f["snapped_floor"]
        else:
            r["snap_distance_m"] = ""
            r["estimated_end_alt_m"] = ""
            r["snapped_floor"] = ""
    seg_df = pd.DataFrame(rows)
    return summary, seg_df


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
            if p.is_dir() and (p / GT_CSV).exists()
        )

    summaries: list[dict] = []
    all_segs: list[pd.DataFrame] = []
    for name in exp_names:
        summary, seg_df = _analyze_one(name)
        summaries.append(summary)
        if not seg_df.empty:
            all_segs.append(seg_df)

    summary_df = pd.DataFrame(summaries)
    summary_path = STRUCTURED_ROOT / "test_results" / "gramushka_dry_run_summary.csv"
    summary_df.to_csv(summary_path, index=False)

    seg_df_all = pd.concat(all_segs, ignore_index=True) if all_segs else pd.DataFrame()
    seg_path = STRUCTURED_ROOT / "test_results" / "gramushka_dry_run_segments.csv"
    seg_df_all.to_csv(seg_path, index=False)

    print(f"[dry-run] SNAP_AMBIGUITY_THRESHOLD_M = {SNAP_AMBIGUITY_THRESHOLD_M} m")
    print(f"[dry-run] Wrote summary:  {summary_path}")
    print(f"[dry-run] Wrote segments: {seg_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
