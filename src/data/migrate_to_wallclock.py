"""One-shot migration: shift existing structuredData CSVs from boot-time
`timestamp_ms` to wall-clock Unix epoch ms.

For each `structuredData/data/<name>/`:
  1. Find the matching raw `sensorLog_YYYYMMDDTHHMMSS.txt`.
  2. Compute the offset = (filename Unix-epoch ms) - (smallest boot ms in raw log).
  3. Shift `timestamp_ms` in every sensor CSV and `start_ms`/`end_ms` in `gt.csv`
     by that offset, in place. Existing GT (which the user may have edited) is
     preserved — only its time axis is shifted.

Per-CSV idempotency: if a file's first numeric value already looks like Unix
epoch ms (i.e. > 10^12), it's left alone. So the script is safe to re-run.

Usage:
    venv/bin/python -m src.data.migrate_to_wallclock          # all experiments
    venv/bin/python -m src.data.migrate_to_wallclock <name>   # just one
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from .loader.constants import (
    BAROMOSHKA_CSV,
    GT_CSV,
    METADATA_CSV,
    RAW_DATA_ROOT,
    STRUCTURED_DATA_DIR,
)
from .loader.parsing import (
    _find_sensor_log,
    _first_boot_ms_in_log,
    _parse_iso_filename_to_ms,
)


# Anything ≥ this is treated as already-wall-clock (≈ year 2001-09 in epoch ms).
_WALLCLOCK_THRESHOLD_MS = 10**12


def _looks_like_wallclock(values: pd.Series) -> bool:
    if values.empty:
        return True  # nothing to shift
    first = values.iloc[0]
    try:
        return float(first) >= _WALLCLOCK_THRESHOLD_MS
    except (TypeError, ValueError):
        return True


def _shift_csv(csv_path: Path, offset_ms: int, columns: list[str]) -> str:
    """Add `offset_ms` to each named column. Returns a status string."""
    df = pd.read_csv(csv_path)
    present = [c for c in columns if c in df.columns]
    if not present:
        return f"skip (no target columns): {csv_path.name}"

    if all(_looks_like_wallclock(df[c]) for c in present):
        return f"skip (already wall-clock): {csv_path.name}"

    for c in present:
        df[c] = df[c].astype("int64") + int(offset_ms)
    df.to_csv(csv_path, index=False)
    return f"shifted {present} by {offset_ms:+d} ms: {csv_path.name}"


def migrate_one(name: str) -> bool:
    """Migrate a single experiment by name. Returns True if anything was changed."""
    out_dir = STRUCTURED_DATA_DIR / name
    if not out_dir.is_dir():
        print(f"[migrate] {name}: no structured dir; skipping")
        return False

    raw_dir = RAW_DATA_ROOT / name
    if not raw_dir.is_dir():
        print(f"[migrate] {name}: no raw dir at {raw_dir}; skipping")
        return False

    try:
        primary_log = _find_sensor_log(raw_dir)
    except FileNotFoundError as e:
        print(f"[migrate] {name}: {e}; skipping")
        return False

    try:
        wall_ms = _parse_iso_filename_to_ms(primary_log)
        first_boot = _first_boot_ms_in_log(primary_log)
    except (ValueError, OSError) as e:
        print(f"[migrate] {name}: {type(e).__name__}: {e}; skipping")
        return False

    offset = wall_ms - first_boot

    print(f"[migrate] {name}")
    print(f"    raw log:    {primary_log.name}")
    print(f"    first boot: {first_boot} ms")
    print(f"    wall start: {wall_ms} ms")
    print(f"    offset:     {offset:+d} ms")

    changed = False
    for csv_path in sorted(out_dir.glob("*.csv")):
        stem = csv_path.stem
        # metadata.csv has no timestamps; baramoshka.csv is floor/height.
        if csv_path.name in (METADATA_CSV, BAROMOSHKA_CSV):
            continue
        if csv_path.name == GT_CSV:
            cols = ["start_ms", "end_ms"]
        else:
            cols = ["timestamp_ms"]
        msg = _shift_csv(csv_path, offset, cols)
        print(f"    {msg}")
        if msg.startswith("shifted"):
            changed = True

    return changed


def migrate_all() -> None:
    if not STRUCTURED_DATA_DIR.is_dir():
        print(f"[migrate] no structured dir: {STRUCTURED_DATA_DIR}")
        return
    names = sorted(p.name for p in STRUCTURED_DATA_DIR.iterdir() if p.is_dir())
    if not names:
        print("[migrate] no experiments to migrate")
        return
    n_changed = 0
    for name in names:
        if migrate_one(name):
            n_changed += 1
    print(f"\n[migrate] done. {n_changed} of {len(names)} experiments updated.")


def main() -> None:
    if len(sys.argv) > 1:
        for name in sys.argv[1:]:
            migrate_one(name)
    else:
        migrate_all()


if __name__ == "__main__":
    main()
