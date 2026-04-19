"""Populate per-experiment baramoshka.csv + metadata fields from the central
gramushka reference tables.

For every experiment under ``structuredData/data/<exp>/``:

* Resolves the experiment's building via
  :func:`loader.gramushka.gramushka_building_for_exp`. If a building is
  resolved, copies the central ``gramushka/<bldg>/gramushka.csv`` into the
  per-experiment ``baramoshka.csv`` with columns renamed to ``floor, height``.
  Otherwise (archive experiment, exp3, Haari) leaves ``baramoshka.csv`` empty
  and the downstream corrector falls back to pure-barometer Δh.

* Backfills ``temperature_c`` in ``metadata.csv`` from the raw
  ``metadata.txt``'s ``Temperature:`` line.

* Backfills ``start_floor`` in ``metadata.csv`` using the project default
  (Ground Floor everywhere, Floor 12 only for ``milleniumHotel_..._exp1``).

Idempotent: re-running after edits reimports the central gramushka but does
not touch ``metadata.csv`` values that are already set (so your manual edits
to ``start_floor`` or ``temperature_c`` survive).

Run with::

    python -m src.data.dataset_cleanup.populate_baramoshka           # all experiments
    python -m src.data.dataset_cleanup.populate_baramoshka <name>... # specific experiments
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

from ..loader.constants import (
    BAROMOSHKA_COLUMNS,
    BAROMOSHKA_CSV,
    METADATA_COLUMNS,
    METADATA_CSV,
    METADATA_FILENAME,
    RAW_DATA_ROOT,
    STRUCTURED_DATA_DIR,
)
from ..loader.gramushka import (
    gramushka_building_for_exp,
    load_baramoshka_for_exp,
    resolve_start_floor_default,
)
from ..loader.parsing import _parse_metadata_file
from ..loader.pipeline import _parse_temperature_c, rebuild_metadata_index


def _load_existing_metadata_row(meta_path: Path) -> dict[str, str]:
    if not meta_path.exists():
        return {}
    df = pd.read_csv(meta_path)
    if df.empty:
        return {}
    return {
        k: ("" if pd.isna(v) else str(v))
        for k, v in df.iloc[0].to_dict().items()
    }


def _write_metadata_row(meta_path: Path, row: dict[str, str]) -> None:
    """Write a single-row metadata.csv using the full schema."""
    ordered = {c: row.get(c, "") for c in METADATA_COLUMNS}
    pd.DataFrame([ordered], columns=METADATA_COLUMNS).to_csv(meta_path, index=False)


def _update_metadata(exp_name: str, exp_dir: Path) -> tuple[bool, dict[str, str]]:
    """Ensure `metadata.csv` has `temperature_c` and `start_floor` populated.

    Existing non-empty values are preserved. Returns (changed, row).
    """
    meta_path = exp_dir / METADATA_CSV
    row = _load_existing_metadata_row(meta_path)

    # Canonical exp_name.
    if row.get("exp_name") != exp_name:
        row["exp_name"] = exp_name

    changed = False

    # temperature_c: parse from raw metadata.txt only if still blank.
    if not row.get("temperature_c"):
        raw_meta_path = RAW_DATA_ROOT / exp_name / METADATA_FILENAME
        if raw_meta_path.exists():
            raw = _parse_metadata_file(raw_meta_path)
            parsed = _parse_temperature_c(raw.get("Temperature", ""))
            if parsed:
                row["temperature_c"] = parsed
                changed = True

    # start_floor: apply project default only if still blank.
    if not row.get("start_floor"):
        row["start_floor"] = resolve_start_floor_default(exp_name)
        changed = True

    # Ensure all schema columns exist in the output (blanks OK).
    for col in METADATA_COLUMNS:
        row.setdefault(col, "")

    _write_metadata_row(meta_path, row)
    return changed, row


def _write_baramoshka(exp_dir: Path, frame: pd.DataFrame | None) -> bool:
    """Write baramoshka.csv for this experiment.

    * When `frame` is a populated DataFrame, write it.
    * When `frame` is `None`, leave whatever's already there intact (so manual
      edits stick); only touch the file if it is missing entirely, in which
      case we create it empty with the schema header.
    """
    path = exp_dir / BAROMOSHKA_CSV
    if frame is not None and not frame.empty:
        frame[BAROMOSHKA_COLUMNS].to_csv(path, index=False)
        return True
    if not path.exists():
        pd.DataFrame(columns=BAROMOSHKA_COLUMNS).to_csv(path, index=False)
        return True
    return False


def populate_one(exp_name: str) -> dict[str, str]:
    """Populate baramoshka + metadata for a single experiment."""
    exp_dir = STRUCTURED_DATA_DIR / exp_name
    if not exp_dir.is_dir():
        return {"exp": exp_name, "status": "missing-structured-dir"}

    building = gramushka_building_for_exp(exp_name)
    baramoshka_frame = load_baramoshka_for_exp(exp_name) if building else None

    bar_written = _write_baramoshka(exp_dir, baramoshka_frame)
    meta_changed, meta_row = _update_metadata(exp_name, exp_dir)

    return {
        "exp":            exp_name,
        "building":       building or "(none)",
        "bar_rows":       str(len(baramoshka_frame)) if baramoshka_frame is not None else "0",
        "bar_written":    str(bar_written),
        "start_floor":    meta_row.get("start_floor", ""),
        "temperature_c":  meta_row.get("temperature_c", ""),
        "meta_changed":   str(meta_changed),
    }


def populate_all(exp_names: list[str] | None = None) -> list[dict[str, str]]:
    if exp_names is None:
        exp_names = sorted(
            p.name for p in STRUCTURED_DATA_DIR.iterdir()
            if p.is_dir() and (p / METADATA_CSV).exists()
        )
    results = [populate_one(n) for n in exp_names]
    rebuild_metadata_index()
    return results


def main() -> None:
    # Windows console is cp1252 by default; reconfigure stdout to UTF-8 so
    # Hebrew building names print without UnicodeEncodeError.
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except AttributeError:
        pass

    args = sys.argv[1:]
    results = populate_all(args or None)
    # Simple tabular print.
    cols = ["exp", "building", "bar_rows", "start_floor", "temperature_c", "meta_changed"]
    widths = {c: max(len(c), *(len(r.get(c, "")) for r in results)) for c in cols}
    def fmt(row: dict[str, str]) -> str:
        return "  ".join(row.get(c, "").ljust(widths[c]) for c in cols)
    print(fmt({c: c for c in cols}))
    print(fmt({c: "-" * widths[c] for c in cols}))
    for r in results:
        print(fmt(r))


if __name__ == "__main__":
    main()
