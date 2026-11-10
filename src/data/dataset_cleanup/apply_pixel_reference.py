"""Post-Task-2 consolidation: make every non-Pixel experiment share its
Pixel 10 reference's ``gt.csv`` (segment boundaries) and PRS-derived Δh.

Why
---
After Task 2 aligned each phone's timestamps to its Pixel reference's
clock, the physically-correct segment bounds for that phone are exactly
the Pixel's gt segment bounds — the elevator ride started and ended at
the same wall-clock moment on every phone that was in the same
building. Per-phone barometers (Samsung, Xiaomi) have their own
bias/drift characteristics; using Pixel's PRS as the shared barometer
source eliminates phone-to-phone Δh scatter that's not a real physical
difference.

Result: every phone in a given experiment ends up with:

* identical ``start_ms``/``end_ms``/``type`` rows (copied from the Pixel),
* identical ``height_diff_m`` (computed from the Pixel's PRS, snapped to
  the phone's own baramoshka floor table),
* its own ``exp_name`` column entry.

Pixel recordings themselves are untouched. Experiments with no Pixel
reference (Haari, BarIlan2, archive) are skipped; they continue to use
their own gt.csv as before.

Safety
------
Saves the pre-existing gt.csv to ``gt_pre_pixel_ref_backup.csv`` before
overwriting, so this operation is recoverable.

Run with::

    python -m src.data.dataset_cleanup.apply_pixel_reference              # dry-run (no writes)
    python -m src.data.dataset_cleanup.apply_pixel_reference --apply      # rewrite gt.csv in place
"""

from __future__ import annotations

import shutil
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
from ..loader.pipeline import addGTtoSegment, load_baramoshka
from .phone_time_calibration import find_pixel_reference_exp


BACKUP_SUFFIX = "gt_pre_pixel_ref_backup.csv"


def _load_metadata(exp_dir: Path) -> dict[str, str]:
    p = exp_dir / METADATA_CSV
    if not p.exists():
        return {}
    df = pd.read_csv(p)
    if not len(df):
        return {}
    return {k: ("" if pd.isna(v) else str(v)) for k, v in df.iloc[0].to_dict().items()}


def _apply_one(exp_name: str, pixel_name: str, apply: bool) -> dict:
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    pix_dir = STRUCTURED_DATA_DIR / pixel_name
    pix_gt_path = pix_dir / GT_CSV
    pix_prs_path = pix_dir / "PRS.csv"
    if not pix_gt_path.exists() or not pix_prs_path.exists():
        return {"exp_name": exp_name, "status": "pixel ref missing gt/PRS"}

    pix_gt = pd.read_csv(pix_gt_path)
    pix_prs = pd.read_csv(pix_prs_path)
    metadata = _load_metadata(exp_dir)
    baramoshka = load_baramoshka(exp_name)

    # Take Pixel's gt as the shared timeline, stamp with this phone's
    # exp_name so downstream filtering by exp_name keeps working.
    shared_gt = pix_gt.copy()
    shared_gt["exp_name"] = exp_name

    # Use Pixel's PRS as the single-source barometer. gramushka snap runs
    # on Pixel's altitude — resulting Δh identical across phones in the
    # same experiment (which is what we want; physics says it must be).
    sensors_with_pixel_prs = {"PRS": pix_prs}
    new_gt = addGTtoSegment(
        sensors_with_pixel_prs, shared_gt,
        metadata=metadata, baramoshka=baramoshka,
    )

    out_cols = list(GT_COLUMNS) + ["exp_name"]

    status = "DRY-RUN (not written)"
    if apply:
        backup = exp_dir / BACKUP_SUFFIX
        current = exp_dir / GT_CSV
        if current.exists() and not backup.exists():
            shutil.copy2(current, backup)
        new_gt[out_cols].to_csv(current, index=False)
        status = f"written (backup: {BACKUP_SUFFIX})"

    n_flags = len(new_gt.attrs.get("gramushka_snap_flags", []))
    mode = new_gt.attrs.get("gramushka_mode", "unknown")
    return {
        "exp_name":           exp_name,
        "pixel_ref":          pixel_name,
        "n_segments":         len(new_gt),
        "mode":               mode,
        "n_gramushka_flags":  n_flags,
        "status":             status,
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
            if p.is_dir() and "pixel" not in p.name.lower()
            and (p / GT_CSV).exists()
        )

    rows: list[dict] = []
    for name in exp_names:
        pix = find_pixel_reference_exp(name)
        if pix is None:
            rows.append({
                "exp_name": name, "pixel_ref": "", "status": "no pixel reference",
                "n_segments": 0, "mode": "", "n_gramushka_flags": 0,
            })
            continue
        rows.append(_apply_one(name, pix, apply=apply))

    out_path = STRUCTURED_ROOT / "test_results" / "pixel_reference_summary.csv"
    pd.DataFrame(rows).to_csv(out_path, index=False)
    mode = "APPLY (gt.csv rewritten)" if apply else "DRY-RUN (no writes)"
    print(f"[pixel-ref] mode: {mode}")
    print(f"[pixel-ref] summary: {out_path}")
    print()
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
