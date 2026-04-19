"""Collect every per-phone calibration plot into one ``test_results/``
folder so Uriya can flip through them sequentially and manually confirm
that each phone's ACC pulses clearly overlap the Pixel's inside every
Pixel-tagged elevator segment.

For each non-Pixel experiment this copies, if present:

* ``phone_time_verify.png`` — Pixel vs phone ACC-magnitude overlay inside
  six evenly-sampled Pixel-tagged up/down segments.
* ``phone_time_calibration.png`` — the calibration dry-run plot (with
  "before shift" / "after shift" overlays of six segments).

The two files are renamed with the experiment name prefix so a single
``ls`` of the folder is enough to see what's in it. Also writes an
``INDEX.md`` that lists every file + its corresponding `offset_ms` +
method from ``phone_time_calibration.csv`` so you can tell at a glance
which experiments were shifted how.
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

import pandas as pd

from ..loader.constants import (
    STRUCTURED_DATA_DIR,
    STRUCTURED_ROOT,
)

TEST_RESULTS_DIR = STRUCTURED_ROOT / "test_results" / "phone_time_verify"
CAL_SUMMARY_CSV = STRUCTURED_ROOT / "test_results" / "phone_time_calibration.csv"
VERIFY_SUMMARY_CSV = STRUCTURED_ROOT / "test_results" / "phone_time_verify_summary.csv"


def main() -> None:
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    cal = pd.read_csv(CAL_SUMMARY_CSV) if CAL_SUMMARY_CSV.exists() else pd.DataFrame()
    verify = pd.read_csv(VERIFY_SUMMARY_CSV) if VERIFY_SUMMARY_CSV.exists() else pd.DataFrame()

    copied: list[dict] = []
    for exp_dir in sorted(STRUCTURED_DATA_DIR.iterdir()):
        if not exp_dir.is_dir():
            continue
        name = exp_dir.name
        src_verify = exp_dir / "phone_time_verify.png"
        src_cal = exp_dir / "phone_time_calibration.png"

        any_copied = False
        row = {"exp_name": name, "offset_ms": "", "method": "", "median_residual_ms": ""}

        if src_verify.exists():
            dst = TEST_RESULTS_DIR / f"{name}__verify.png"
            shutil.copy2(src_verify, dst)
            any_copied = True
        if src_cal.exists():
            dst = TEST_RESULTS_DIR / f"{name}__calibration.png"
            shutil.copy2(src_cal, dst)
            any_copied = True

        if not any_copied:
            continue

        if not cal.empty:
            match = cal[cal["exp_name"] == name]
            if len(match):
                row["offset_ms"] = str(match["offset_ms"].iloc[0])
                row["method"] = str(match.get("method", pd.Series([""])).iloc[0])
        if not verify.empty:
            match = verify[verify["exp_name"] == name]
            if len(match) and "median_residual_ms" in match.columns:
                row["median_residual_ms"] = str(match["median_residual_ms"].iloc[0])

        copied.append(row)

    # Write a plain-text index for easy browsing.
    idx_path = TEST_RESULTS_DIR / "INDEX.md"
    with idx_path.open("w", encoding="utf-8") as f:
        f.write("# Phone-time-calibration verification plots\n\n")
        f.write(
            "Per-experiment pulse-alignment plots for manual review. Each "
            "non-Pixel experiment has two images:\n\n"
            "* `<exp>__verify.png` — final ACC overlay (Pixel blue, phone green) "
            "inside six Pixel-tagged elevator segments. If green pulses sit "
            "inside the red window and on top of the blue curve, alignment is good.\n"
            "* `<exp>__calibration.png` — the calibration-step diagnostic "
            "(before-shift orange vs after-shift green).\n\n"
            "Scan in order; any plot where green pulses clearly sit outside the "
            "red window is a misaligned phone to flag.\n\n"
        )
        f.write("## Offsets applied\n\n")
        f.write("| Experiment | offset (ms) | method | median residual (ms) |\n")
        f.write("|---|---|---|---|\n")
        for r in copied:
            f.write(f"| `{r['exp_name']}` | {r['offset_ms']} | {r['method']} | "
                    f"{r['median_residual_ms']} |\n")

    print(f"[test_results] wrote {len(copied)} experiments' plots → {TEST_RESULTS_DIR}")
    print(f"[test_results] index: {idx_path}")


if __name__ == "__main__":
    main()
