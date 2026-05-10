"""Pipeline loader — main entry points downstream code uses.

The pipeline is CSV-based. For each experiment ``<name>``:

    rawData/<name>/                     <- inputs
        sensorLog_*.txt
        metadata.txt
        forBarometer/                   (optional secondary barometer source)

    structuredData/data/<name>/         <- processed artifacts
        <SENSOR>.csv                    (one per sensor present)
        gt.csv
        metadata.csv
        baramoshka.csv                  (floor → height, user-filled later)
        forBarometer_alignment.png      (diagnostic, if applicable)

    structuredData/metadata.csv         <- top-level index (auto-rebuilt)

On load, if all required CSVs exist under `structuredData/data/<name>/`, the
pipeline is hydrated from them. Otherwise the raw sensorLog is parsed, GT is
derived from the barometer, and the CSVs are written. Pickle is no longer used.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator

import numpy as np
import pandas as pd

from src.physics.barometric import pressure_to_altitude

from .alignment import _merge_secondary_prs
from .constants import (
    BAROMOSHKA_COLUMNS,
    BAROMOSHKA_CSV,
    EXPERIMENT_TYPE_TEST,
    EXPERIMENT_TYPE_TRAIN,
    EXPERIMENT_TYPES,
    FOR_BAROMETER_PLOT_FILENAME,
    FOR_BAROMETER_SUBDIR,
    GAP_THRESHOLD_S,
    GT_COLUMNS,
    GT_CSV,
    METADATA_COLUMNS,
    METADATA_CSV,
    METADATA_FILENAME,
    RAW_DATA_ROOT,
    SOURCE_EXPERIMENT,
    STRUCTURED_DATA_DIR,
    STRUCTURED_INDEX_CSV,
)
from .parsing import (
    _find_sensor_log,
    _first_boot_ms_in_log,
    _parse_iso_filename_to_ms,
    _parse_metadata_file,
    _parse_sensor_log,
)


# --------------------------------------------------------------------------
# Resolution helpers
# --------------------------------------------------------------------------

# Target rate that every public load goes through. 50 Hz is fast enough for
# the elevator dynamics we care about (≪ 1 Hz signal bandwidth) and keeps the
# accelerometer / barometer on the same grid for downstream code.
TARGET_SAMPLE_RATE_HZ = 50


def _resolve_raw_dir(exp: Path | str) -> Path:
    """Accept either a raw-folder path or a bare experiment name."""
    p = Path(exp)
    if p.is_absolute() or len(p.parts) > 1:
        return p
    return RAW_DATA_ROOT / p.name


def _structured_dir_for(name: str) -> Path:
    return STRUCTURED_DATA_DIR / name


def classify_experiment_type(exp_name: str) -> str:
    """Return `'test'` if the experiment was recorded at Beit Yitzchaki
    Raanana, else `'train'`.

    Project-level split (Uriya, 2026-04-19): the Beit Yitzchaki
    experiments are the held-out cross-building test set; everything
    else (Millenium Hotel, Millenium Outside, Acro, Beit Mansour,
    Bar-Ilan 2, Haari) is train.
    The rule is deterministic from folder names alone so
    `list_experiments` / metadata stays consistent across reruns.
    """
    return EXPERIMENT_TYPE_TEST if "beityitzchaki" in exp_name.lower() else EXPERIMENT_TYPE_TRAIN


def list_structured_experiments(
    structured_root: Path | str = STRUCTURED_DATA_DIR,
) -> list[str]:
    """Return names of experiments that already live under
    ``structuredData/data/<name>/``.

    Distinct from :func:`list_experiments`, which scans ``rawData/`` for
    folders that hold a raw sensorLog. An experiment that was materialised
    directly into ``structuredData/`` (e.g. via the GT-editor S3 ingest)
    won't appear in ``list_experiments`` because it has no raw log.
    """
    root = Path(structured_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        # Mirror the rawData filter: only count folders that look like a
        # real experiment. Stray dirs (e.g. `__pycache__`, accidental
        # `--help` from a CLI typo) lack any of the canonical files.
        if not ((p / "ACC.csv").exists()
                or (p / "metadata.csv").exists()
                or (p / "gt.csv").exists()):
            continue
        out.append(p.name)
    return out


def list_experiments(
    raw_root: Path | str = RAW_DATA_ROOT,
    kind: str = "all",
) -> list[str]:
    """Return names of experiments under `raw_root` that hold a raw sensor log
    directly (i.e., `sensorLog_*.txt` or the macOS-copy variant).

    Args:
        raw_root: folder to scan (defaults to ``rawData/``).
        kind: ``'all'`` (default), ``'train'``, or ``'test'``. When not
            ``'all'``, each candidate is passed through
            :func:`classify_experiment_type` and kept only if it matches.
    """
    if kind not in ("all", *EXPERIMENT_TYPES):
        raise ValueError(f"kind must be 'all', 'train', or 'test' (got {kind!r})")

    root = Path(raw_root)
    if not root.is_dir():
        return []
    out: list[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_dir():
            continue
        if not (list(p.glob("sensorLog_*.txt")) or list(p.glob("Copy of sensorLog_*.txt"))):
            continue
        if kind != "all" and classify_experiment_type(p.name) != kind:
            continue
        out.append(p.name)
    return out


# --------------------------------------------------------------------------
# Parsing + GT derivation
# --------------------------------------------------------------------------

def _wall_clock_offset_ms(primary_log: Path) -> int:
    """Offset to add to a boot-time `timestamp_ms` to put it on Unix-epoch ms.

    Reference point: the smallest valid sample timestamp in the raw
    sensorLog (boot ms) maps to the wall-clock start encoded in the
    sensorLog filename (Unix epoch ms, local-tz interpretation).
    """
    return _parse_iso_filename_to_ms(primary_log) - _first_boot_ms_in_log(primary_log)


def _shift_timestamps_inplace(
    frames: dict[str, pd.DataFrame], offset_ms: int,
) -> None:
    """Add `offset_ms` to every frame's `timestamp_ms` column (in place)."""
    if offset_ms == 0:
        return
    for df in frames.values():
        if df is None or df.empty or "timestamp_ms" not in df.columns:
            continue
        df["timestamp_ms"] = df["timestamp_ms"].astype("int64") + int(offset_ms)


def getExperimentRawParsed(
    exp: Path | str,
    plot_out_path: Path | str | None = None,
) -> dict[str, pd.DataFrame]:
    """Parse the experiment's sensorLog into per-sensor DataFrames.

    If `<raw_dir>/forBarometer/` exists and has a `sensorLog_*.txt`, the
    primary's PRS frame is swapped with the secondary's PRS (start-time
    aligned onto the primary's uptime timebase). When `plot_out_path` is
    provided, the diagnostic `forBarometer_alignment.png` is written there.

    The returned `timestamp_ms` is wall-clock Unix epoch ms (derived from the
    sensorLog filename's `YYYYMMDDTHHMMSS` token), not boot uptime.
    Secondary-barometer alignment runs in boot time first; the wall-clock
    shift is applied last so both devices end up on the same wall-clock.
    """
    raw_dir = _resolve_raw_dir(exp)
    primary_log = _find_sensor_log(raw_dir)
    frames = _parse_sensor_log(primary_log)

    fb_dir = raw_dir / FOR_BAROMETER_SUBDIR
    if fb_dir.is_dir():
        try:
            secondary_log = _find_sensor_log(fb_dir)
        except FileNotFoundError:
            print(f"[loader] forBarometer/ exists but has no sensorLog: {fb_dir}")
        else:
            plot_path = Path(plot_out_path) if plot_out_path is not None else None
            frames = _merge_secondary_prs(frames, primary_log, secondary_log, plot_path)

    try:
        offset_ms = _wall_clock_offset_ms(primary_log)
    except (ValueError, OSError) as e:
        print(f"[loader] could not derive wall-clock offset for {primary_log.name} "
              f"({type(e).__name__}: {e}); leaving timestamps in boot time")
    else:
        _shift_timestamps_inplace(frames, offset_ms)

    return frames


def _coerce_bool(v) -> bool:
    """Permissive bool coercion. NaN / unrecognised values default to True
    to match the column's default-on semantics."""
    if isinstance(v, bool):
        return v
    try:
        if pd.isna(v):
            return True
    except (TypeError, ValueError):
        pass
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("false", "0", "no", "n", "f"):
        return False
    if s in ("true", "1", "yes", "y", "t"):
        return True
    return True


def _segments_to_full_gt(
    segments: pd.DataFrame, t0_ms: int, t_end_ms: int,
) -> pd.DataFrame:
    """Convert segmenter output to alternating `[start_ms, end_ms, type]` rows
    covering `[t0_ms, t_end_ms]` with 'outside' filler between rides."""
    rides: list[dict] = []
    for _, row in segments.iterrows():
        s_lo, _ = row["start_ci"]
        _, e_hi = row["end_ci"]
        rides.append({
            "start_ms":    int(t0_ms + float(s_lo) * 1000),
            "end_ms":      int(t0_ms + float(e_hi) * 1000),
            "type":        str(row["type"]),
            "description": "",
            "signalClearRecording": True,
        })
    rides.sort(key=lambda r: r["start_ms"])

    for r in rides:
        r["start_ms"] = max(r["start_ms"], t0_ms)
        r["end_ms"] = min(r["end_ms"], t_end_ms)
    rides = [r for r in rides if r["end_ms"] > r["start_ms"]]

    out: list[dict] = []
    cursor = t0_ms
    for r in rides:
        if r["start_ms"] < cursor:
            r["start_ms"] = cursor
            if r["end_ms"] <= cursor:
                continue
        if r["start_ms"] > cursor:
            out.append({"start_ms": cursor, "end_ms": r["start_ms"],
                        "type": "outside", "description": "",
                        "signalClearRecording": True})
        out.append(r)
        cursor = r["end_ms"]

    if cursor < t_end_ms:
        out.append({"start_ms": cursor, "end_ms": t_end_ms,
                    "type": "outside", "description": "",
                    "signalClearRecording": True})
    if not out:
        out.append({"start_ms": t0_ms, "end_ms": t_end_ms,
                    "type": "outside", "description": "",
                    "signalClearRecording": True})

    return pd.DataFrame(out, columns=GT_COLUMNS)


def _derive_gt_from_prs(prs: pd.DataFrame) -> pd.DataFrame:
    # Lazy-import the segmenter so that merely importing `pipeline` doesn't pull
    # in the accelerometer/quality stack (useful for tools that only need I/O).
    from src.segmentation.algorithms import (
        SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
    )

    t0_ms = int(prs["timestamp_ms"].iloc[0])
    t_end_ms = int(prs["timestamp_ms"].iloc[-1])
    t_sec = (prs["timestamp_ms"].to_numpy(dtype=float) - t0_ms) / 1000.0
    h = prs["GT_height_m"].to_numpy(dtype=float)
    h_smooth = (pd.Series(h).rolling(window=51, center=True, min_periods=1)
                             .median().to_numpy())
    height_frame = pd.DataFrame({"time": t_sec, "height": h_smooth})
    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    segments = Segmenter(cfg).detect(height_frame)
    return _segments_to_full_gt(segments, t0_ms, t_end_ms)


# Threshold (meters) above which a gramushka snap is considered ambiguous.
# Uriya's rule (2026-04-19): ambiguous snaps mean something is wrong — flag
# rather than silently accept.
SNAP_AMBIGUITY_THRESHOLD_M = 1.5


def _coerce_temperature(raw: str | float | None) -> float | None:
    if raw is None or raw == "":
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _lookup_floor_height(baramoshka: pd.DataFrame, floor_name: str) -> float | None:
    """Find a floor's height from a baramoshka DataFrame.

    Matching order (each case-insensitive on trimmed names):
      1. exact match
      2. substring either way (handles ``"Ground Floor"`` ↔ ``"Street Level / Ground"``)
      3. only when `floor_name` contains the token ``"ground"``: the row
         whose numeric height is closest to 0.0 (covers the
         ``Bar Ilan 2`` entry ``"Street Level / Ground" = ±0.00``).

    Returns ``None`` when nothing matches (e.g. a building whose gramushka has
    no ground-level entry at all — user attention required).
    """
    if baramoshka is None or baramoshka.empty or not floor_name:
        return None
    target = str(floor_name).strip().lower()
    names = baramoshka["floor"].astype(str).str.strip().str.lower()

    # 1. exact
    exact = baramoshka.loc[names == target]
    if len(exact):
        return float(exact["height"].iloc[0])

    # 2. substring (either direction)
    sub = baramoshka.loc[names.str.contains(target, regex=False) | names.apply(lambda n: target in n or n in target)]
    if len(sub):
        return float(sub["height"].iloc[0])

    # 3. fallback for generic "ground" requests
    if "ground" in target:
        idx = int((baramoshka["height"].astype(float) - 0.0).abs().idxmin())
        if abs(float(baramoshka["height"].iloc[idx])) <= 1.0:
            return float(baramoshka["height"].iloc[idx])

    return None


def _snap_altitude_to_floor(
    altitude_m: float, baramoshka: pd.DataFrame,
) -> tuple[float, str, float]:
    """Snap `altitude_m` to the closest entry in `baramoshka`.

    Returns ``(snapped_height_m, floor_name, distance_m)``. `distance_m` is the
    absolute gap between the input altitude and the chosen floor — the caller
    flags the segment as inconsistent when this exceeds
    :data:`SNAP_AMBIGUITY_THRESHOLD_M`.
    """
    diffs = (baramoshka["height"] - altitude_m).abs()
    idx = int(diffs.idxmin())
    return (
        float(baramoshka["height"].iloc[idx]),
        str(baramoshka["floor"].iloc[idx]),
        float(diffs.iloc[idx]),
    )


def _compute_raw_dh_per_segment(
    prs: pd.DataFrame, gt: pd.DataFrame, temperature_c: float | None,
    edge_k: int = 1,
) -> list[float]:
    """Temperature-aware Δh per GT segment, edge-averaged against noise.

    Returns a list aligned with ``gt`` rows. Rows with <2 samples yield 0.0.
    """
    if prs is None or prs.empty or "pressure" not in prs.columns:
        return [0.0] * len(gt)

    ts = prs["timestamp_ms"].astype("int64").to_numpy()
    p_all = prs["pressure"].to_numpy(dtype=float)
    alt_all = pressure_to_altitude(p_all, temperature_c=temperature_c)

    import numpy as np
    alt_all = np.asarray(alt_all, dtype=float)
    diffs: list[float] = []
    for _, row in gt.iterrows():
        s, e = int(row["start_ms"]), int(row["end_ms"])
        mask = (ts >= s) & (ts < e)
        seg = alt_all[mask]
        if seg.size < 2:
            diffs.append(0.0)
            continue
        k = max(1, min(edge_k, seg.size // 2))
        diffs.append(float(np.mean(seg[-k:]) - np.mean(seg[:k])))
    return diffs


def addGTtoSegment(
    sensors: dict[str, pd.DataFrame],
    gt: pd.DataFrame,
    metadata: dict[str, str] | None = None,
    baramoshka: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Populate ``height_diff_m`` for each GT row.

    Modes:

    * **Snap mode** (``baramoshka`` is populated AND ``metadata["start_floor"]``
      resolves to a row in it): temperature-aware barometer altitudes are
      integrated forward from the known start-floor height and every segment
      endpoint is snapped to the nearest baramoshka floor. ``height_diff_m`` is
      the difference of successive snapped altitudes — robust to barometer
      drift because each segment re-anchors to a known floor.

      A per-segment snap-distance is tracked. Any segment whose estimated
      endpoint falls more than :data:`SNAP_AMBIGUITY_THRESHOLD_M` from any
      gramushka floor is flagged (see ``gt.attrs["gramushka_snap_flags"]``)
      — these are the cases Uriya asked to surface because an ambiguous snap
      indicates something is wrong (drift, segmentation-boundary noise, wrong
      start_floor, mis-labeled segment type, etc.).

    * **Pure-barometer mode** (no baramoshka or no resolvable start_floor):
      ``height_diff_m`` is the raw temperature-aware barometer Δh for each
      segment. Temperature comes from ``metadata["temperature_c"]`` when
      present; otherwise ISA-standard 15 °C is used.

    Signature change — existing callers that pass only ``(sensors, gt)`` keep
    the same behavior they had before, except Δh now uses a temperature-aware
    ISA formula (identical at 15 °C). New callers should pass both
    ``metadata`` and ``baramoshka`` to enable snap mode.
    """
    gt = gt.copy()
    prs = sensors.get("PRS")
    if prs is None or prs.empty or "pressure" not in prs.columns:
        gt["height_diff_m"] = float("nan")
        return gt

    temp_c: float | None = None
    start_floor_name = ""
    if metadata:
        temp_c = _coerce_temperature(metadata.get("temperature_c"))
        start_floor_name = str(metadata.get("start_floor", "") or "").strip()

    has_baramoshka = (
        baramoshka is not None
        and not baramoshka.empty
        and {"floor", "height"}.issubset(baramoshka.columns)
    )
    start_alt = (
        _lookup_floor_height(baramoshka, start_floor_name) if has_baramoshka else None
    )
    snap_mode = has_baramoshka and start_alt is not None

    raw_dhs = _compute_raw_dh_per_segment(prs, gt, temp_c)

    if not snap_mode:
        gt["height_diff_m"] = raw_dhs
        gt.attrs["gramushka_snap_flags"] = []
        gt.attrs["gramushka_mode"] = "pure_barometer"
        return gt

    snapped_dhs: list[float] = []
    flags: list[dict] = []
    running_alt = start_alt
    for i, (_, row) in enumerate(gt.iterrows()):
        raw = raw_dhs[i]
        estimated_end = running_alt + raw
        snapped_end, floor_name, snap_dist = _snap_altitude_to_floor(
            estimated_end, baramoshka,
        )
        corrected = snapped_end - running_alt
        if snap_dist > SNAP_AMBIGUITY_THRESHOLD_M:
            flags.append({
                "segment_idx":         i,
                "type":                str(row.get("type", "")),
                "start_ms":            int(row["start_ms"]),
                "end_ms":              int(row["end_ms"]),
                "raw_dh_m":            raw,
                "corrected_dh_m":      corrected,
                "estimated_end_alt_m": estimated_end,
                "snapped_floor":       floor_name,
                "snap_distance_m":     snap_dist,
            })
        snapped_dhs.append(corrected)
        running_alt = snapped_end

    gt["height_diff_m"] = snapped_dhs
    gt.attrs["gramushka_snap_flags"] = flags
    gt.attrs["gramushka_mode"] = "snap"
    return gt


def _iso_date_time_from_filename(raw_dir: Path) -> tuple[str, str]:
    """Return (date, time) parsed from `sensorLog_YYYYMMDDTHHMMSS.txt`,
    or ("", "") if unavailable.

    Date uses `D.M.YYYY` to match the convention in metadata.txt.
    Time uses `HH:MM`.
    """
    try:
        log_path = _find_sensor_log(raw_dir)
    except FileNotFoundError:
        return "", ""
    import re
    from datetime import datetime
    # Accept any `...YYYYMMDDTHHMMSS.txt` at the tail — some filenames have
    # extra tokens after `sensorLog_`.
    m = re.search(r"(\d{8}T\d{6})\.txt$", log_path.name)
    if not m:
        return "", ""
    try:
        dt = datetime.strptime(m.group(1), "%Y%m%dT%H%M%S")
    except ValueError:
        return "", ""
    return f"{dt.day}.{dt.month}.{dt.year}", f"{dt.hour:02d}:{dt.minute:02d}"


def _parse_temperature_c(raw_val: str) -> str:
    """Extract numeric Celsius value from a `Temperature:` metadata.txt value.

    Accepts `"14 degrees Celsius"`, `"14C"`, `"14 °C"`, `"14"` — anything whose
    leading token parses as a number. Returns the value as a bare string
    (e.g. `"14"` or `"14.5"`) so it round-trips cleanly through CSV I/O.
    Returns `""` when no number is found.
    """
    if not raw_val:
        return ""
    import re
    m = re.search(r"-?\d+(?:\.\d+)?", str(raw_val))
    return m.group(0) if m else ""


def _build_metadata_row(
    exp_name: str, raw_meta: dict[str, str], raw_dir: Path,
) -> dict[str, str]:
    """Map raw metadata.txt keys to the `METADATA_COLUMNS` schema.

    Falls back to the sensorLog filename's ISO timestamp for missing
    `Date` / `Time` fields. `temperature_c` is parsed from the raw
    `Temperature:` line; `start_floor` is left blank and filled in separately
    (it is not present in any raw metadata.txt).
    """
    iso_date, iso_time = _iso_date_time_from_filename(raw_dir)
    return {
        "exp_name":        exp_name,
        "experimenter":    raw_meta.get("Name", ""),
        "phone":           raw_meta.get("Phone", ""),
        "location":        raw_meta.get("Location", ""),
        "description":     raw_meta.get("Description", ""),
        "date":            raw_meta.get("Date", "") or iso_date,
        "time":            raw_meta.get("Time", "") or iso_time,
        "experiment_type": classify_experiment_type(exp_name),
        "temperature_c":   _parse_temperature_c(raw_meta.get("Temperature", "")),
        "start_floor":     "",
        "source":          SOURCE_EXPERIMENT,
    }


# --------------------------------------------------------------------------
# CSV I/O
# --------------------------------------------------------------------------

def _write_csvs(
    out_dir: Path,
    data: dict[str, pd.DataFrame],
    gt: pd.DataFrame,
    metadata_row: dict[str, str],
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in data.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    gt.to_csv(out_dir / GT_CSV, index=False)
    pd.DataFrame([metadata_row], columns=METADATA_COLUMNS).to_csv(
        out_dir / METADATA_CSV, index=False,
    )
    bar_path = out_dir / BAROMOSHKA_CSV
    if not bar_path.exists():
        pd.DataFrame(columns=BAROMOSHKA_COLUMNS).to_csv(bar_path, index=False)


def _required_sensor_csvs(raw_dir: Path) -> list[str]:
    """Infer which sensor CSVs should exist based on what's in the raw log."""
    frames = _parse_sensor_log(_find_sensor_log(raw_dir))
    return [f"{name}.csv" for name in frames]


def _has_complete_structured(out_dir: Path, required_sensors: list[str]) -> bool:
    if not out_dir.is_dir():
        return False
    if not (out_dir / GT_CSV).exists():
        return False
    if not (out_dir / METADATA_CSV).exists():
        return False
    for csv_name in required_sensors:
        if not (out_dir / csv_name).exists():
            return False
    return True


def load_baramoshka(exp: Path | str) -> pd.DataFrame | None:
    """Read ``structuredData/data/<name>/baramoshka.csv`` into a DataFrame.

    Returns ``None`` when the file is missing or empty (i.e. no gramushka is
    associated with this experiment — the corrector treats that as
    pure-barometer mode).
    """
    name = _resolve_raw_dir(exp).name
    path = _structured_dir_for(name) / BAROMOSHKA_CSV
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty or not {"floor", "height"}.issubset(df.columns):
        return None
    return df


def _load_structured_triplet(
    out_dir: Path,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Read `(sensors, gt, metadata)` from an already-populated
    `structuredData/data/<name>/` directory.

    Only CSVs whose stem matches a known sensor name in
    :data:`SENSOR_COLUMNS` are loaded into `data` — any other `*.csv`
    sitting in the experiment folder (auxiliary outputs like
    ``gramushka_flags.csv``, ``phone_time_verify.csv``, backups) is
    ignored so downstream code that iterates slices by ``timestamp_ms``
    can't accidentally pick them up as sensors.
    """
    from .constants import SENSOR_COLUMNS
    data: dict[str, pd.DataFrame] = {}
    for csv_path in sorted(out_dir.glob("*.csv")):
        stem = csv_path.stem
        if stem not in SENSOR_COLUMNS:
            continue
        data[stem] = pd.read_csv(csv_path)

    gt = pd.read_csv(out_dir / GT_CSV)
    # Backfill the description column if loading an older gt.csv that predates it.
    if "description" not in gt.columns:
        gt["description"] = ""
    gt["description"] = gt["description"].fillna("").astype(str)
    # Backfill signalClearRecording (default True) for older gt.csv files.
    if "signalClearRecording" not in gt.columns:
        gt["signalClearRecording"] = True
    gt["signalClearRecording"] = gt["signalClearRecording"].apply(_coerce_bool)
    gt = gt.reindex(columns=GT_COLUMNS)

    meta_df = pd.read_csv(out_dir / METADATA_CSV)
    metadata = (
        {k: ("" if pd.isna(v) else str(v)) for k, v in meta_df.iloc[0].to_dict().items()}
        if len(meta_df) else {}
    )
    return data, gt, metadata


def rebuild_metadata_index(structured_root: Path | str = STRUCTURED_DATA_DIR) -> pd.DataFrame:
    """Scan all per-experiment `metadata.csv` files and write the top-level
    index at `structuredData/metadata.csv`.

    Per-experiment metadata CSVs that are missing the `experiment_type`
    column (or have it blank) are backfilled in-place using
    :func:`classify_experiment_type`.
    """
    root = Path(structured_root)
    rows: list[dict[str, str]] = []
    if root.is_dir():
        for exp_dir in sorted(root.iterdir()):
            mpath = exp_dir / METADATA_CSV
            if not mpath.exists():
                continue
            try:
                df = pd.read_csv(mpath)
                if not len(df):
                    continue
                row = {c: ("" if pd.isna(v) else str(v))
                       for c, v in df.iloc[0].to_dict().items()}
                exp_name = row.get("exp_name") or exp_dir.name
                needs_rewrite = False
                if not row.get("experiment_type"):
                    row["experiment_type"] = classify_experiment_type(exp_name)
                    needs_rewrite = True
                # Default `source` to "experiment" for any pre-existing row
                # that predates the column. New rows written via
                # saveExperimentData / _build_metadata_row already set it.
                if not row.get("source"):
                    row["source"] = SOURCE_EXPERIMENT
                    needs_rewrite = True
                if needs_rewrite:
                    pd.DataFrame([row], columns=METADATA_COLUMNS).to_csv(mpath, index=False)
                rows.append(row)
            except Exception as e:
                print(f"[loader] skipping {mpath}: {type(e).__name__}: {e}")
    index_df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    STRUCTURED_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(STRUCTURED_INDEX_CSV, index=False)
    return index_df


# --------------------------------------------------------------------------
# Resampling
# --------------------------------------------------------------------------

def _is_uniformly_sampled(ts, rel_tol: float = 0.05) -> bool:
    """True when timestamp diffs have low coefficient of variation.

    `rel_tol` is the max allowed ``std(dt) / median(dt)`` ratio. ~5 %
    tolerates the typical phone-sensor jitter while flagging traces with
    dropped samples or burst-mode batching as non-uniform.
    """
    import numpy as np
    if len(ts) < 3:
        return True
    d = np.diff(np.asarray(ts, dtype=np.float64))
    md = float(np.median(d))
    if md <= 0 or len(d) == 0:
        return False
    return float(np.std(d)) / md <= rel_tol


def _detect_valid_intervals(
    ts_ms, gap_threshold_s: float = GAP_THRESHOLD_S,
) -> list[tuple[int, int]]:
    """Split a timestamp array into contiguous valid intervals.

    A "gap" is any spacing between consecutive samples greater than
    ``gap_threshold_s`` seconds — below the threshold frequency the data
    is too sparse to recover via interpolation and is treated as missing.

    Returns inclusive `[(start_ms, end_ms), ...]` covering the timestamp
    array minus the gaps. Empty input → ``[]``; a single sample → the
    span ``[(t, t)]`` (the resampler will short-circuit such cases).
    """
    import numpy as np
    ts = np.asarray(ts_ms, dtype=np.int64)
    if ts.size == 0:
        return []
    if ts.size == 1:
        return [(int(ts[0]), int(ts[0]))]
    diffs = np.diff(ts)
    gap_threshold_ms = int(round(gap_threshold_s * 1000.0))
    breaks = np.where(diffs > gap_threshold_ms)[0]
    if breaks.size == 0:
        return [(int(ts[0]), int(ts[-1]))]
    intervals: list[tuple[int, int]] = []
    start_idx = 0
    for b in breaks:
        intervals.append((int(ts[start_idx]), int(ts[b])))
        start_idx = b + 1
    intervals.append((int(ts[start_idx]), int(ts[-1])))
    return intervals


def _resample_sensor_to_hz(
    df: pd.DataFrame, target_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> pd.DataFrame:
    """Resample one sensor frame onto a uniform `target_hz` grid.

    Numeric columns: ``scipy.signal.resample_poly`` (with its built-in
    anti-alias filter) when input timestamps are uniformly sampled —
    otherwise ``np.interp``. Non-numeric columns (e.g. the ``exp_name``
    tag) are propagated by nearest-neighbor lookup. The output spans
    ``[t0, t1]`` of the input and stays on int64 epoch-ms timestamps so
    downstream timestamp slicing is unaffected.

    No-op for empty / single-row frames or frames missing
    ``timestamp_ms``.
    """
    import numpy as np
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df
    df = (df.sort_values("timestamp_ms")
            .drop_duplicates(subset="timestamp_ms")
            .reset_index(drop=True))
    if len(df) < 2:
        return df

    ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    t0, t1 = int(ts[0]), int(ts[-1])
    period_ms = 1000.0 / target_hz
    n = int(np.floor((t1 - t0) / period_ms)) + 1
    if n < 2:
        return df
    new_ts = np.round(t0 + np.arange(n) * period_ms).astype(np.int64)

    value_cols = [
        c for c in df.columns
        if c != "timestamp_ms" and pd.api.types.is_numeric_dtype(df[c])
    ]
    other_cols = [
        c for c in df.columns
        if c != "timestamp_ms" and c not in value_cols
    ]

    out = pd.DataFrame({"timestamp_ms": new_ts})

    if _is_uniformly_sampled(ts) and value_cols:
        from math import gcd
        from scipy.signal import resample_poly
        median_dt_ms = float(np.median(np.diff(ts.astype(np.float64))))
        src_hz = max(1, int(round(1000.0 / median_dt_ms)))
        g = gcd(target_hz, src_hz)
        up = target_hz // g
        down = src_hz // g
        for c in value_cols:
            x = df[c].to_numpy(dtype=float)
            if np.isnan(x).all():
                out[c] = np.nan
                continue
            if np.isnan(x).any():
                # resample_poly's filter would smear NaNs across the
                # signal; fill them with linear interp first.
                idx = np.arange(len(x))
                m = ~np.isnan(x)
                x = np.interp(idx, idx[m], x[m])
            y = resample_poly(x, up=up, down=down)
            if len(y) >= n:
                out[c] = y[:n]
            else:
                pad_val = float(y[-1]) if len(y) else float("nan")
                out[c] = np.concatenate([y, np.full(n - len(y), pad_val)])
    else:
        for c in value_cols:
            x = df[c].to_numpy(dtype=float)
            m = ~np.isnan(x)
            if not m.any():
                out[c] = np.nan
                continue
            out[c] = np.interp(
                new_ts.astype(np.float64),
                ts[m].astype(np.float64),
                x[m],
            )

    if other_cols:
        idx = np.searchsorted(ts, new_ts)
        idx = np.clip(idx, 0, len(ts) - 1)
        for c in other_cols:
            out[c] = df[c].to_numpy()[idx]

    return out


def _resample_sensors_to_hz(
    sensors: dict[str, pd.DataFrame], target_hz: int = TARGET_SAMPLE_RATE_HZ,
) -> dict[str, pd.DataFrame]:
    """Apply :func:`_resample_sensor_to_hz` to every sensor frame."""
    return {
        name: _resample_sensor_to_hz(df, target_hz=target_hz)
        for name, df in sensors.items()
    }


def _resample_sensor_with_gaps(
    df: pd.DataFrame,
    target_hz: int = TARGET_SAMPLE_RATE_HZ,
    gap_threshold_s: float = GAP_THRESHOLD_S,
) -> tuple[pd.DataFrame, list[tuple[int, int]]]:
    """Gap-aware variant of :func:`_resample_sensor_to_hz`.

    Splits ``df`` on every consecutive-sample gap larger than
    ``gap_threshold_s`` seconds, resamples each contiguous valid interval
    independently to ``target_hz``, and concatenates the results. The
    returned DataFrame has *no rows* in the gap regions — downstream
    consumers see a clean 50 Hz signal punctuated by holes.

    Returns ``(resampled_df, valid_intervals)`` where ``valid_intervals``
    is a list of ``[start_ms, end_ms]`` (inclusive, on the original raw
    timestamp scale).
    """
    import numpy as np
    if df is None or df.empty or "timestamp_ms" not in df.columns:
        return df, []
    df = (df.sort_values("timestamp_ms")
            .drop_duplicates(subset="timestamp_ms")
            .reset_index(drop=True))
    if len(df) < 2:
        ts0 = int(df["timestamp_ms"].iloc[0]) if len(df) else 0
        return df, [(ts0, ts0)] if len(df) else []

    ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
    intervals = _detect_valid_intervals(ts, gap_threshold_s=gap_threshold_s)
    if not intervals:
        return df.iloc[0:0].copy(), []

    resampled_chunks: list[pd.DataFrame] = []
    for s_ms, e_ms in intervals:
        mask = (ts >= s_ms) & (ts <= e_ms)
        chunk = df.loc[mask]
        if chunk.empty:
            continue
        chunk_resampled = _resample_sensor_to_hz(
            chunk.reset_index(drop=True), target_hz=target_hz,
        )
        if chunk_resampled is None or chunk_resampled.empty:
            continue
        resampled_chunks.append(chunk_resampled)

    if not resampled_chunks:
        return df.iloc[0:0].copy(), intervals
    out = pd.concat(resampled_chunks, ignore_index=True)
    return out, intervals


def _resample_sensors_with_gaps(
    sensors: dict[str, pd.DataFrame],
    target_hz: int = TARGET_SAMPLE_RATE_HZ,
    gap_threshold_s: float = GAP_THRESHOLD_S,
) -> tuple[dict[str, pd.DataFrame], dict[str, list[tuple[int, int]]]]:
    """Apply :func:`_resample_sensor_with_gaps` to every sensor frame.

    Each sensor has its own valid-interval list (its own clock). Algorithms
    that fuse multiple sensors must intersect their lists themselves.
    """
    out_sensors: dict[str, pd.DataFrame] = {}
    out_intervals: dict[str, list[tuple[int, int]]] = {}
    for name, df in sensors.items():
        resampled, intervals = _resample_sensor_with_gaps(
            df, target_hz=target_hz, gap_threshold_s=gap_threshold_s,
        )
        out_sensors[name] = resampled
        out_intervals[name] = intervals
    return out_sensors, out_intervals


def _detect_sensors_valid_intervals(
    sensors: dict[str, pd.DataFrame],
    gap_threshold_s: float = GAP_THRESHOLD_S,
) -> dict[str, list[tuple[int, int]]]:
    """Per-sensor gap detection without resampling.

    Mirrors the interval output of :func:`_resample_sensors_with_gaps`
    so downstream consumers can still rely on
    ``gt.attrs["valid_intervals_per_sensor"]`` while keeping the sensor
    frames at their native cadence. Used by :func:`getExperimentData`
    to preserve the original sample rate that the prediction stack
    (matched-filter shape grid, theoretical σ, conformal calibration)
    was tuned and validated against — public auto-resampling broke the
    conformal multipliers (~20× inflation) by changing the residual
    statistics underneath the calibrator.
    """
    out: dict[str, list[tuple[int, int]]] = {}
    for name, df in sensors.items():
        if df is None or df.empty or "timestamp_ms" not in df.columns:
            out[name] = []
            continue
        ts = df["timestamp_ms"].to_numpy(dtype=np.int64)
        out[name] = _detect_valid_intervals(ts, gap_threshold_s=gap_threshold_s)
    return out


# --------------------------------------------------------------------------
# Container
# --------------------------------------------------------------------------

@dataclass
class ExperimentPipeline:
    """Container for a fully preprocessed experiment.

    Attributes:
        data: per-sensor DataFrames covering the whole experiment.
        gt: alternating intervals with columns `start_ms`, `end_ms`, `type`
            ('up' | 'down' | 'outside'). Covers the full timeline with no gaps.
        metaData: parsed metadata (keys from `METADATA_COLUMNS`).
        valid_intervals: per-sensor `[(start_ms, end_ms), ...]` of regions
            where the raw signal was sampled densely enough (above
            ``THRESHOLD_FREQUENCY_HZ``) to be trusted. Anything outside is
            a gap and should not be fed to segmentation/prediction.
    """
    data: dict[str, pd.DataFrame]
    gt: pd.DataFrame
    metaData: dict[str, str]
    valid_intervals: dict[str, list[tuple[int, int]]] = field(default_factory=dict)

    def __iter__(self) -> Iterator[tuple[dict[str, pd.DataFrame], pd.Series, dict[str, str]]]:
        for _, row in self.gt.iterrows():
            s, e = int(row["start_ms"]), int(row["end_ms"])
            slice_dict = {
                name: df[(df["timestamp_ms"] >= s) & (df["timestamp_ms"] < e)]
                      .reset_index(drop=True)
                for name, df in self.data.items()
            }
            yield slice_dict, row, self.metaData

    def __len__(self) -> int:
        return len(self.gt)

    def gap_free_subslices(self, sensor: str = "ACC") -> list[tuple[int, int]]:
        """Return the named sensor's valid intervals.

        Convenience for plotting and per-interval iteration. Falls back to
        ``gt.attrs["valid_intervals_per_sensor"]`` when the field was not
        populated at construction time (e.g. legacy callers).
        """
        if self.valid_intervals and sensor in self.valid_intervals:
            return self.valid_intervals[sensor]
        attrs = getattr(self.gt, "attrs", {}) or {}
        return attrs.get("valid_intervals_per_sensor", {}).get(sensor, [])


# --------------------------------------------------------------------------
# Public entry points
# --------------------------------------------------------------------------

def getExperimentData(
    exp: Path | str, use_cache: bool = True,
) -> tuple[dict[str, pd.DataFrame], pd.DataFrame, dict[str, str]]:
    """Return `(sensors, gt, metadata)` for an experiment.

    * If `structuredData/data/<name>/` already contains all required CSVs,
      load and return them directly.
    * Otherwise parse the raw sensorLog under `rawData/<name>/`, derive GT
      from the barometer, materialise the CSVs, and return.

    `use_cache=False` forces a rebuild even when structured CSVs exist.

    Typical usage::

        sensors, gt, metadata = getExperimentData("eyalyakir_...")
        # Manipulate sensors/gt freely, then optionally wrap for iteration:
        pipeline = ExperimentPipeline(sensors, gt, metadata)
        for data_slice, gt_row, meta in pipeline:
            ...
    """
    raw_dir = _resolve_raw_dir(exp)
    name = raw_dir.name
    out_dir = _structured_dir_for(name)

    if use_cache and out_dir.is_dir() and (out_dir / GT_CSV).exists() \
            and (out_dir / METADATA_CSV).exists():
        # Sensor CSVs must match what the raw log would produce.
        try:
            required = _required_sensor_csvs(raw_dir)
        except FileNotFoundError:
            required = []  # raw log may be absent; accept whatever is in structured
        if _has_complete_structured(out_dir, required):
            try:
                data, gt, metadata_row = _load_structured_triplet(out_dir)
                _inject_exp_name(name, data, gt)
                if "height_diff_m" not in gt.columns or gt["height_diff_m"].isna().all():
                    bar = load_baramoshka(name)
                    gt = addGTtoSegment(data, gt, metadata=metadata_row, baramoshka=bar)
                    _inject_exp_name(name, data, gt)
                    gt.to_csv(out_dir / GT_CSV, index=False)
                # Public load preserves the native sensor cadence.
                # Earlier versions auto-resampled to TARGET_SAMPLE_RATE_HZ
                # here, but that silently changed the residual statistics
                # of every prediction algorithm — the conformal multipliers
                # baked into ``test_results/prediction/.../calibration_*.json``
                # (and the canonical numbers in
                # ``docs/latex/figures/prediction/results_macros.tex``) were
                # measured on raw-cadence data, so auto-resampling produced
                # ~20× inflated CIs and broke documented coverage. Callers
                # that genuinely want a uniform 50 Hz grid (e.g. the
                # boutique pipeline's CSV-upload path) call
                # :func:`resample_sensor_with_gaps` explicitly via
                # :func:`src.data.load_data.enrich_loaded`.
                #
                # Gap-aware: per-sensor valid intervals are still computed
                # so downstream UIs / segmenters can honour data dropouts.
                gt.attrs["valid_intervals_per_sensor"] = (
                    _detect_sensors_valid_intervals(data)
                )
                _inject_exp_name(name, data, gt)
                return data, gt, metadata_row
            except Exception as e:
                print(f"[loader] failed to load structured CSVs "
                      f"({type(e).__name__}: {e}); rebuilding")

    out_dir.mkdir(parents=True, exist_ok=True)
    plot_path = out_dir / FOR_BAROMETER_PLOT_FILENAME
    data = getExperimentRawParsed(raw_dir, plot_out_path=plot_path)

    if "PRS" in data and not data["PRS"].empty:
        gt = _derive_gt_from_prs(data["PRS"])
    else:
        if "ACC" not in data or data["ACC"].empty:
            raise ValueError(f"No PRS or ACC data in {raw_dir}; cannot build pipeline")
        acc = data["ACC"]
        t0_ms = int(acc["timestamp_ms"].iloc[0])
        t_end_ms = int(acc["timestamp_ms"].iloc[-1])
        gt = pd.DataFrame(
            [{"start_ms": t0_ms, "end_ms": t_end_ms,
              "type": "outside", "description": "",
              "signalClearRecording": True}],
            columns=GT_COLUMNS,
        )

    raw_meta = _parse_metadata_file(raw_dir / METADATA_FILENAME)
    metadata_row = _build_metadata_row(name, raw_meta, raw_dir)

    # First-time build: baramoshka hasn't been populated yet (populator runs
    # after structuredData exists). Falls through to pure-barometer Δh.
    bar = load_baramoshka(name)
    gt = addGTtoSegment(data, gt, metadata=metadata_row, baramoshka=bar)

    _inject_exp_name(name, data, gt)
    _write_csvs(out_dir, data, gt, metadata_row)
    rebuild_metadata_index()

    # CSVs are written at native sensor cadence (the cache-on-disk
    # contract) and the public load now preserves it — see the
    # cache-hit branch above for the conformal-stability rationale.
    # We still detect per-sensor gaps so downstream consumers can
    # honour them.
    gt.attrs["valid_intervals_per_sensor"] = (
        _detect_sensors_valid_intervals(data)
    )
    _inject_exp_name(name, data, gt)
    return data, gt, metadata_row





def _inject_exp_name(
    name: str, sensors: dict[str, pd.DataFrame], gt: pd.DataFrame,
) -> None:
    """Stamp an `exp_name` column on each sensor DataFrame and on gt (in-place).

    Keeps callers able to remember which experiment a row came from after
    `pd.concat`, `pd.merge`, etc.
    """
    for df in sensors.values():
        if df is None or df.empty:
            continue
        df["exp_name"] = name
    gt["exp_name"] = name


def saveExperimentData(
    name: str,
    sensors: dict[str, pd.DataFrame],
    gt: pd.DataFrame,
    metadata: dict[str, str],
) -> Path:
    """Persist the three components to `structuredData/data/<name>/`.

    Rewrites all sensor CSVs, `gt.csv`, and `metadata.csv`. Leaves
    `baramoshka.csv` untouched if it already exists. Rebuilds the top-level
    `structuredData/metadata.csv` index afterwards. Returns the output directory.
    """
    out_dir = _structured_dir_for(name)
    meta_row = {c: str(metadata.get(c, "")) for c in METADATA_COLUMNS}
    meta_row["exp_name"] = name  # exp_name is canonical
    if not meta_row.get("experiment_type"):
        meta_row["experiment_type"] = classify_experiment_type(name)
    _write_csvs(out_dir, sensors, gt, meta_row)
    rebuild_metadata_index()
    return out_dir
