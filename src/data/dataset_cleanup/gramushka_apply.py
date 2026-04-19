"""Apply gramushka-snapped Δh to every experiment's ``gt.csv``.

This is the destructive companion to :mod:`src.data.dataset_cleanup.gramushka_dry_run`:

* For each experiment under ``structuredData/data/<exp>/``, loads sensors,
  metadata, baramoshka, and the current ``gt.csv``.
* Runs :func:`src.data.loader.pipeline.addGTtoSegment` with the new signature
  (``metadata=...``, ``baramoshka=...``). If the experiment has a populated
  ``baramoshka.csv`` and a resolvable ``start_floor`` in metadata, Δh is
  computed by integrating temperature-aware barometer altitudes from the
  start-floor height and snapping every segment endpoint to the nearest
  gramushka floor. Otherwise the temperature-aware raw-barometer Δh is used.
* Writes the updated ``gt.csv`` back in place.
* Writes a per-experiment ``gramushka_flags.csv`` listing segments whose
  snap distance exceeded :data:`SNAP_AMBIGUITY_THRESHOLD_M` — these are the
  cases Uriya asked to surface rather than silently snap.
* Writes a top-level ``structuredData/gramushka_flags_summary.csv`` aggregating
  the per-experiment flag counts and RMSE so it's easy to scan.

Run with::

    python -m src.data.dataset_cleanup.gramushka_apply              # all experiments
    python -m src.data.dataset_cleanup.gramushka_apply <name>...    # specific experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ..loader.constants import (
    GT_COLUMNS,
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

FLAGS_CSV = "gramushka_flags.csv"
FLAGS_SUMMARY_CSV = "gramushka_flags_summary.csv"

FLAG_COLUMNS = [
    "segment_idx", "type", "start_ms", "end_ms",
    "raw_dh_m", "corrected_dh_m",
    "estimated_end_alt_m", "snapped_floor", "snap_distance_m",
]


def _load_sensors_gt_metadata(
    exp_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Lightweight loader that avoids importing the segmentation stack."""
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


def _apply_one(exp_name: str) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    if not exp_dir.is_dir() or not (exp_dir / GT_CSV).exists():
        return {"exp_name": exp_name, "status": "missing"}

    data, gt, metadata = _load_sensors_gt_metadata(exp_dir)
    baramoshka = load_baramoshka(exp_name)

    new_gt = addGTtoSegment(data, gt, metadata=metadata, baramoshka=baramoshka)
    mode = new_gt.attrs.get("gramushka_mode", "unknown")
    flags = new_gt.attrs.get("gramushka_snap_flags", [])

    # Preserve the columns the schema requires plus exp_name (which tests and
    # downstream scripts rely on).
    expected_cols = list(GT_COLUMNS)
    if "exp_name" in new_gt.columns:
        expected_cols.append("exp_name")
    else:
        new_gt["exp_name"] = exp_name
        expected_cols.append("exp_name")
    new_gt[expected_cols].to_csv(exp_dir / GT_CSV, index=False)

    # Per-experiment flags file (overwritten every run).
    flags_path = exp_dir / FLAGS_CSV
    if flags:
        pd.DataFrame(flags)[FLAG_COLUMNS].to_csv(flags_path, index=False)
    elif flags_path.exists():
        flags_path.unlink()

    return {
        "exp_name":        exp_name,
        "mode":            mode,
        "n_segments":      len(new_gt),
        "n_flags":         len(flags),
        "start_floor":     metadata.get("start_floor", ""),
        "temperature_c":   metadata.get("temperature_c", ""),
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
            if p.is_dir() and (p / GT_CSV).exists()
        )

    summaries = [_apply_one(n) for n in exp_names]
    summary_df = pd.DataFrame(summaries)
    summary_path = STRUCTURED_ROOT / "test_results" / FLAGS_SUMMARY_CSV
    summary_df.to_csv(summary_path, index=False)

    print(f"[apply] SNAP_AMBIGUITY_THRESHOLD_M = {SNAP_AMBIGUITY_THRESHOLD_M} m")
    print(f"[apply] Wrote summary: {summary_path}")
    print()
    print(summary_df.to_string(index=False))


if __name__ == "__main__":
    main()
