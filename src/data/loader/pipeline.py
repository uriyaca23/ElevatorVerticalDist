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

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

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
    GT_COLUMNS,
    GT_CSV,
    METADATA_COLUMNS,
    METADATA_CSV,
    METADATA_FILENAME,
    RAW_DATA_ROOT,
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
                if not row.get("experiment_type"):
                    row["experiment_type"] = classify_experiment_type(exp_name)
                    pd.DataFrame([row], columns=METADATA_COLUMNS).to_csv(mpath, index=False)
                rows.append(row)
            except Exception as e:
                print(f"[loader] skipping {mpath}: {type(e).__name__}: {e}")
    index_df = pd.DataFrame(rows, columns=METADATA_COLUMNS)
    STRUCTURED_INDEX_CSV.parent.mkdir(parents=True, exist_ok=True)
    index_df.to_csv(STRUCTURED_INDEX_CSV, index=False)
    return index_df


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
    """
    data: dict[str, pd.DataFrame]
    gt: pd.DataFrame
    metaData: dict[str, str]

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
