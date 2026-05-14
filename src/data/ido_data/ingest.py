"""Bulk ingest of `source=ido` recordings into `structuredData/data/`.

Input is a single "manifest" CSV — one row per segment of one recording,
with a wrong-path pointer to the accelerometer CSV plus the real
sensor files living at ``<data_root>/<tool_id>/{acc,baro,gyro,magnet}_data_<n1>_<n2>.csv``.

The script groups rows by ``(tool_id, basename(acc_data))`` so each
group becomes one structured experiment. Only ``elevator`` segments
land in the saved GT; ``stairs`` and the inter-segment gaps are filled
in as ``outside``. Ride direction (``up`` / ``down``) is decided from
the barometer Δh across each elevator window — the GT editor can later
correct any mistakes by eye.

Usage:
    venv/bin/python -m src.data.ido_data.ingest \\
        --manifest /path/to/manifest.csv \\
        --data-root /path/to/rootDataFolder

If a target experiment already exists under ``structuredData/data/`` it
is skipped. One bad group never aborts the batch; the run ends with a
summary of created / skipped-existing / failed groups.
"""
from __future__ import annotations

import logging
import sys
import traceback
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from src.data.loader import (
    EXPERIMENT_TYPE_TEST,
    GT_COLUMNS,
    METADATA_COLUMNS,
    SOURCE_IDO,
    STRUCTURED_DATA_DIR,
    saveExperimentData,
)
from src.physics import pressure_to_altitude


logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------
# Config + stats
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class IngestConfig:
    """All hyperparameters for one batch ingest run.

    Pass `manifest_csv_path` and `data_root_folder` from the CLI; the
    rest have defaults matching the spec but can be overridden in code
    if a future manifest uses different column names or sensor prefixes.
    """

    manifest_csv_path: Path
    data_root_folder: Path

    # Manifest column names.
    col_tool_id:      str = "tool_id"
    col_location:     str = "location"
    col_lat:          str = "latitude"
    col_long:         str = "longitude"
    col_acc_path:     str = "acc_data"
    col_start_time:   str = "start_time"
    col_end_time:     str = "end_time"
    col_movement_env: str = "movment_env"
    col_session_id:   str = "session_id"

    # Sensor file prefix swap. The acc basename matches
    # `acc_<suffix>`; strip `acc_prefix`, swap in the other prefixes.
    acc_prefix:    str = "acc_"
    baro_prefix:   str = "baro_"
    gyro_prefix:   str = "gyro_"
    mag_prefix:    str = "magnet_"

    # Per-sensor source column names inside each sensor CSV.
    acc_xyz_cols:    tuple[str, str, str] = ("X", "Y", "Z")
    acc_time_col:    str = "REPORT_TIME"
    prs_pressure_col: str = "PRESSURE"
    prs_time_col:    str = "REPORT_TIME"
    gyro_xyz_cols:   tuple[str, str, str] = ("X", "Y", "Z")
    gyro_time_col:   str = "REPORT_TIME"
    mag_xyz_cols:    tuple[str, str, str] = ("X", "Y", "Z")
    mag_time_col:    str = "REPORT_TIME"

    # Metadata defaults.
    temperature_c:   str = "25"
    start_floor:     str = "0"
    experiment_type: str = EXPERIMENT_TYPE_TEST
    source:          str = SOURCE_IDO

    # Manifest enumeration values.
    elevator_env_value: str = "elevator"
    stairs_env_value:   str = "stairs"

    # Direction inference: median Δh across the first/last `direction_window_s`
    # of each elevator interval drives the up/down tag.
    direction_window_s: float = 1.0

    # Manifest datetime parsing order. Default is M/D/YYYY to match the
    # spec example ``5/9/2024 8:26:56 AM``; flip to True for D/M/YYYY
    # manifests. The sensor CSVs use ISO 8601 (``2024-05-05T08:22:00.002Z``)
    # which pandas parses identically regardless of this flag.
    time_day_first: bool = False


@dataclass
class IngestStats:
    """Tallies for one ingest run.

    `failed` entries hold ``(group_key, short_error, full_traceback)`` so
    the run log can show the full crash reason while stdout keeps to the
    short repr.
    """

    created:          list[str]                      = field(default_factory=list)
    skipped_existing: list[str]                      = field(default_factory=list)
    failed:           list[tuple[str, str, str]]     = field(default_factory=list)

    def summary_lines(self) -> list[str]:
        return [
            f"created:          {len(self.created)}",
            f"skipped_existing: {len(self.skipped_existing)}",
            f"failed:           {len(self.failed)}",
        ]


# --------------------------------------------------------------------------
# Datetime + canonical-sensor helpers (ported from gt_editor.py)
# --------------------------------------------------------------------------

def _parse_time_to_epoch_ms(
    series: pd.Series, day_first: bool,
) -> np.ndarray:
    """Convert a sensor-CSV time column to int64 epoch ms.

    Ported from `src/data/gt_editor.py:_parse_time_to_epoch_ms` so we
    don't need to import gt_editor (which pulls Tk transitively). Same
    rules: numeric columns use the magnitude heuristic, string columns
    go through `pd.to_datetime(dayfirst=...)`.
    """
    if series.empty:
        raise ValueError("Time column is empty.")

    nums = pd.to_numeric(series, errors="coerce")
    numeric_share = nums.notna().mean()
    if numeric_share > 0.95:
        ts = nums.to_numpy(dtype=float)
        n_bad = int(np.isnan(ts).sum())
        if n_bad:
            raise ValueError(
                f"{n_bad} numeric time values failed to parse — "
                f"drop empty rows first."
            )
        mx = float(np.nanmax(ts))
        span = mx - float(np.nanmin(ts))
        if mx > 1e12:
            return ts.astype("int64")
        if mx > 1e9:
            return (ts * 1000.0).astype("int64")
        if span < 1e4:
            return (ts * 1000.0).astype("int64")
        return ts.astype("int64")

    dts = pd.to_datetime(series, dayfirst=day_first, errors="coerce",
                         utc=False)
    n_bad = int(dts.isna().sum())
    if n_bad:
        raise ValueError(
            f"Could not parse {n_bad} time values as datetime "
            f"(try toggling 'day-first?')."
        )
    if getattr(dts.dt, "tz", None) is not None:
        dts = dts.dt.tz_convert("UTC").dt.tz_localize(None)
    return dts.astype("datetime64[ms]").astype("int64").to_numpy(dtype="int64")


def _canonicalize_sensor_df(
    raw: pd.DataFrame,
    time_col: str,
    col_map: dict[str, str],
    sensor_kind: str,
    day_first: bool,
) -> pd.DataFrame:
    """Apply the column mapping for one sensor file and return a
    canonical-schema DataFrame (`timestamp_ms` + the keys of
    ``col_map``). For PRS, also derives ``GT_height_m`` via
    :func:`pressure_to_altitude`.

    Pure equivalent of :meth:`src.data.gt_editor.GtEditor._build_canonical_sensor_df`
    (lines 1709-1765 of that module). Raises ``ValueError`` on missing
    columns, duplicate mappings, or unparseable time values.
    """
    used = [time_col, *col_map.values()]
    if len(set(used)) != len(used):
        raise ValueError(
            f"{sensor_kind}: each canonical column must map to a different "
            f"source column (got {used})."
        )
    missing = [c for c in used if c not in raw.columns]
    if missing:
        raise ValueError(
            f"{sensor_kind}: columns {missing!r} not found in source CSV "
            f"(have: {list(raw.columns)})."
        )

    sub = raw[used].dropna()
    if sub.empty:
        return pd.DataFrame(columns=["timestamp_ms", *col_map.keys()])

    ts_ms = _parse_time_to_epoch_ms(sub[time_col], day_first=day_first)

    df = pd.DataFrame({"timestamp_ms": ts_ms.astype("int64")})
    for canon, src in col_map.items():
        df[canon] = pd.to_numeric(sub[src], errors="coerce").to_numpy(dtype=float)
    df = (df.dropna()
            .sort_values("timestamp_ms")
            .drop_duplicates("timestamp_ms", keep="first")
            .reset_index(drop=True))

    if sensor_kind == "PRS" and "pressure" in df.columns:
        df["GT_height_m"] = pressure_to_altitude(df["pressure"].to_numpy())
    return df


# --------------------------------------------------------------------------
# Per-group helpers
# --------------------------------------------------------------------------

def _resolve_sensor_paths(
    data_root: Path, tool_id: str, acc_basename: str, cfg: IngestConfig,
) -> dict[str, Path]:
    """Map the manifest's (wrong) acc-path basename to the four real
    sensor files under ``<data_root>/<tool_id>/``.

    Implementation: strip `cfg.acc_prefix` from the basename to get the
    suffix `data_<n1>_<n2>.csv`, then swap in each sensor's prefix.
    Deliberately avoids `str.replace("acc_", …)` so a pathological name
    like `acc_data_acc_foo.csv` doesn't get double-substituted.
    """
    if not acc_basename.startswith(cfg.acc_prefix):
        raise ValueError(
            f"acc filename {acc_basename!r} does not start with "
            f"expected prefix {cfg.acc_prefix!r}."
        )
    suffix = acc_basename[len(cfg.acc_prefix):]  # e.g. "data_1_2.csv"
    folder = data_root / tool_id
    return {
        "ACC": folder / (cfg.acc_prefix  + suffix),
        "PRS": folder / (cfg.baro_prefix + suffix),
        "GYR": folder / (cfg.gyro_prefix + suffix),
        "MAG": folder / (cfg.mag_prefix  + suffix),
    }


def _parse_manifest_dt(s: str, day_first: bool) -> int:
    """Parse one manifest datetime cell (e.g. ``"5/9/2024 8:26:56 AM"``)
    to int64 epoch ms. Tz-aware values are coerced to UTC and the tz
    dropped before the ms cast (same as `_parse_time_to_epoch_ms`)."""
    ts = pd.to_datetime(s, dayfirst=day_first, errors="raise")
    if getattr(ts, "tzinfo", None) is not None:
        ts = ts.tz_convert("UTC").tz_localize(None)
    return int(ts.to_datetime64().astype("datetime64[ms]").astype("int64"))


def _load_group_sensors(
    paths: dict[str, Path], cfg: IngestConfig,
) -> dict[str, pd.DataFrame]:
    """Read each of ACC / PRS / GYR / MAG that exists on disk and
    canonicalize it. Missing sensor files are tolerated by skipping
    them; the caller decides whether their absence is fatal.
    """
    sensor_specs = {
        "ACC": (cfg.acc_time_col,
                {"x": cfg.acc_xyz_cols[0],
                 "y": cfg.acc_xyz_cols[1],
                 "z": cfg.acc_xyz_cols[2]}),
        "PRS": (cfg.prs_time_col,
                {"pressure": cfg.prs_pressure_col}),
        "GYR": (cfg.gyro_time_col,
                {"x": cfg.gyro_xyz_cols[0],
                 "y": cfg.gyro_xyz_cols[1],
                 "z": cfg.gyro_xyz_cols[2]}),
        "MAG": (cfg.mag_time_col,
                {"x": cfg.mag_xyz_cols[0],
                 "y": cfg.mag_xyz_cols[1],
                 "z": cfg.mag_xyz_cols[2]}),
    }
    out: dict[str, pd.DataFrame] = {}
    for sensor, (time_col, col_map) in sensor_specs.items():
        p = paths[sensor]
        if not p.exists():
            logger.warning("%s: file not found, skipping (%s)", sensor, p)
            continue
        try:
            raw = pd.read_csv(p)
        except Exception as e:
            raise ValueError(f"{sensor}: read_csv({p}) failed: {e}") from e
        df = _canonicalize_sensor_df(
            raw, time_col=time_col, col_map=col_map,
            sensor_kind=sensor, day_first=cfg.time_day_first,
        )
        if df.empty:
            logger.warning("%s: empty after canonicalization (%s)", sensor, p)
            continue
        out[sensor] = df
    return out


def _decide_direction(
    prs: pd.DataFrame, start_ms: int, end_ms: int, window_s: float,
) -> str | None:
    """Return ``"up"`` / ``"down"`` from the sign of Δh across the
    elevator window, using a median of the first/last ``window_s``
    seconds inside the window for robustness. Returns ``None`` if the
    PRS slice has too few samples to decide.
    """
    in_window = prs[(prs["timestamp_ms"] >= start_ms)
                    & (prs["timestamp_ms"] <= end_ms)]
    if len(in_window) < 4:
        return None
    h = in_window["GT_height_m"].to_numpy(dtype=float)
    ts = in_window["timestamp_ms"].to_numpy(dtype=float)
    edge_ms = window_s * 1000.0
    early = h[ts <= ts[0]  + edge_ms]
    late  = h[ts >= ts[-1] - edge_ms]
    if early.size == 0 or late.size == 0:
        early, late = h[:1], h[-1:]
    dh = float(np.median(late) - np.median(early))
    return "up" if dh > 0 else "down"


def _windows_to_full_gt(
    windows: list[dict], t0_ms: int, t_end_ms: int,
) -> pd.DataFrame:
    """Build a GT DataFrame from concrete ride windows + ``outside``
    filler covering ``[t0_ms, t_end_ms]``. Port of the shape of
    :func:`src.data.loader.pipeline._segments_to_full_gt` (lines 247-293
    in pipeline.py) but taking direct ``start_ms``/``end_ms`` instead
    of segmenter CI tuples.

    Each ``windows`` item must have keys: ``start_ms``, ``end_ms``,
    ``type`` (one of ``"up"`` / ``"down"``), ``description``.
    """
    rides: list[dict] = []
    for w in windows:
        rides.append({
            "start_ms":             int(w["start_ms"]),
            "end_ms":               int(w["end_ms"]),
            "type":                 str(w["type"]),
            "description":          str(w.get("description", "")),
            "signalClearRecording": True,
        })
    rides.sort(key=lambda r: r["start_ms"])

    for r in rides:
        r["start_ms"] = max(r["start_ms"], t0_ms)
        r["end_ms"]   = min(r["end_ms"],   t_end_ms)
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

    df = pd.DataFrame(out)
    df["height_diff_m"] = float("nan")
    return df[GT_COLUMNS]


def _build_metadata(
    tool_id: str, location: str, lat: float, long: float,
    first_ts_ms: int, session_ids: list[str], cfg: IngestConfig,
) -> dict[str, str]:
    """Assemble a metadata dict matching ``METADATA_COLUMNS``.

    ``location`` is ``{location}_{lat}_{long}`` per the ingest spec; date
    is ``DD-MM-YYYY`` to match the GT editor's add-experiment dialog
    convention.
    """
    first_dt = pd.Timestamp(first_ts_ms, unit="ms")
    session_str = ", ".join(sorted({str(s) for s in session_ids if s != ""}))
    return {
        "exp_name":        "",  # stamped by saveExperimentData from `name`
        "experimenter":    str(tool_id),
        "phone":           str(tool_id),
        "location":        _where_str(location, lat, long),
        "description":     f"ido import; sessions=[{session_str}]",
        "date":            first_dt.strftime("%d-%m-%Y"),
        "time":            first_dt.strftime("%H:%M"),
        "experiment_type": cfg.experiment_type,
        "temperature_c":   cfg.temperature_c,
        "start_floor":     cfg.start_floor,
        "source":          cfg.source,
    }


def _where_str(location: str, lat: float, long: float) -> str:
    """Build the ``where`` portion of an experiment name from the manifest.

    Format is ``{location}_{lat:.4f}_{long:.4f}`` per the user spec.
    Whitespace and the experiment-name separators ``/``, ``\\``, ``:`` are
    replaced with ``-`` so the result is safe to embed in a path-segment.
    """
    loc = str(location).strip()
    for ch in (" ", "/", "\\", ":", "\t"):
        loc = loc.replace(ch, "-")
    return f"{loc}_{float(lat):.4f}_{float(long):.4f}"


def _build_exp_name(
    tool_id: str, where: str, date_str: str, time_str: str,
) -> str:
    """Experiment name. ``time_str`` (``HH-MM-SS`` from the first elevator
    row's start time) disambiguates multiple recordings of the same
    tool_id+location on the same day."""
    return f"{tool_id}_{where}_{tool_id}_{date_str}_{time_str}"


def _stamp_exp_name(
    sensors: dict[str, pd.DataFrame], gt: pd.DataFrame, name: str,
) -> None:
    for df in sensors.values():
        df["exp_name"] = name
    gt["exp_name"] = name


# --------------------------------------------------------------------------
# Per-group orchestration
# --------------------------------------------------------------------------

def _process_group(
    group_key: tuple[str, str], group_df: pd.DataFrame,
    cfg: IngestConfig, stats: IngestStats,
) -> None:
    tool_id, acc_basename = group_key
    tool_id = str(tool_id)
    acc_basename = str(acc_basename)

    elevator_rows = group_df[group_df[cfg.col_movement_env] == cfg.elevator_env_value]
    if elevator_rows.empty:
        logger.info("%s / %s: no elevator rows in group, skipping",
                    tool_id, acc_basename)
        return

    # We need location/lat/long/date to compute the name. Use the first
    # elevator row's values; these should be consistent across the group.
    first_elev = elevator_rows.iloc[0]
    location = str(first_elev[cfg.col_location])
    lat = float(first_elev[cfg.col_lat])
    long_ = float(first_elev[cfg.col_long])
    first_dt = pd.to_datetime(
        first_elev[cfg.col_start_time], dayfirst=cfg.time_day_first,
    )
    date_str = first_dt.strftime("%d-%m-%Y")
    time_str = first_dt.strftime("%H-%M-%S")
    where = _where_str(location, lat, long_)
    name = _build_exp_name(tool_id, where, date_str, time_str)

    # Skip-if-exists — do this BEFORE any sensor I/O.
    if (STRUCTURED_DATA_DIR / name).exists():
        logger.info("[skip] %s already exists", name)
        stats.skipped_existing.append(name)
        return

    paths = _resolve_sensor_paths(
        cfg.data_root_folder, tool_id, acc_basename, cfg,
    )
    sensors = _load_group_sensors(paths, cfg)

    if "ACC" not in sensors or sensors["ACC"].empty:
        raise ValueError(
            f"ACC missing or empty for {tool_id}/{acc_basename} — refusing to save."
        )
    if "PRS" not in sensors or sensors["PRS"].empty:
        raise ValueError(
            f"PRS missing or empty for {tool_id}/{acc_basename} — needed "
            f"for up/down direction. Skipping group."
        )

    # Build ride windows with direction.
    windows: list[dict] = []
    for _, row in elevator_rows.iterrows():
        try:
            start_ms = _parse_manifest_dt(
                row[cfg.col_start_time], day_first=cfg.time_day_first,
            )
            end_ms = _parse_manifest_dt(
                row[cfg.col_end_time], day_first=cfg.time_day_first,
            )
        except Exception as e:
            logger.warning("  bad timestamp in row, dropping: %s", e)
            continue
        if end_ms <= start_ms:
            logger.warning("  end ≤ start, dropping window (%s → %s)",
                           start_ms, end_ms)
            continue
        direction = _decide_direction(
            sensors["PRS"], start_ms, end_ms, cfg.direction_window_s,
        )
        if direction is None:
            logger.warning("  too few PRS samples in [%d, %d], dropping window",
                           start_ms, end_ms)
            continue
        windows.append({
            "start_ms":    start_ms,
            "end_ms":      end_ms,
            "type":        direction,
            "description": f"session={row.get(cfg.col_session_id, '')}",
        })

    if not windows:
        raise ValueError(
            f"No usable elevator windows for {tool_id}/{acc_basename} "
            f"after PRS-direction filtering."
        )

    # Sensor union span for the GT bounds.
    t0_ms = min(int(df["timestamp_ms"].iloc[0])  for df in sensors.values())
    t_end_ms = max(int(df["timestamp_ms"].iloc[-1]) for df in sensors.values())
    gt = _windows_to_full_gt(windows, t0_ms, t_end_ms)

    metadata = _build_metadata(
        tool_id, location, lat, long_, t0_ms,
        session_ids=list(group_df[cfg.col_session_id].astype(str)),
        cfg=cfg,
    )

    _stamp_exp_name(sensors, gt, name)
    saved_dir = saveExperimentData(name, sensors, gt, metadata)
    logger.info("[ok] saved %s → %s (%d ride windows)",
                name, saved_dir, len(windows))
    stats.created.append(name)


# --------------------------------------------------------------------------
# Top-level entry points
# --------------------------------------------------------------------------

def ingest(cfg: IngestConfig) -> IngestStats:
    """Read the manifest, group by (tool_id, acc-basename), process each
    group, and return a summary. Per-group failures are caught and logged
    — they do not abort the run."""
    manifest = pd.read_csv(cfg.manifest_csv_path)
    required = [
        cfg.col_tool_id, cfg.col_location, cfg.col_lat, cfg.col_long,
        cfg.col_acc_path, cfg.col_start_time, cfg.col_end_time,
        cfg.col_movement_env, cfg.col_session_id,
    ]
    missing = [c for c in required if c not in manifest.columns]
    if missing:
        raise ValueError(
            f"Manifest is missing required columns {missing}. "
            f"Have: {list(manifest.columns)}."
        )

    # Group on (tool_id, basename(acc_data)). Each group → one experiment.
    manifest = manifest.copy()
    manifest["_acc_basename"] = manifest[cfg.col_acc_path].apply(
        lambda p: Path(str(p)).name,
    )
    grouped = manifest.groupby([cfg.col_tool_id, "_acc_basename"], sort=False)

    stats = IngestStats()
    for group_key, group_df in grouped:
        key_str = f"{group_key[0]}/{group_key[1]}"
        try:
            _process_group(group_key, group_df, cfg, stats)
        except Exception as e:
            tb = traceback.format_exc()
            logger.error("[fail] %s: %s", key_str, e)
            logger.debug("traceback:\n%s", tb)
            stats.failed.append((key_str, repr(e), tb))
    return stats


def _print_summary(stats: IngestStats) -> None:
    print("\n=== ingest summary ===")
    for line in stats.summary_lines():
        print(f"  {line}")
    if stats.created:
        print("\n  created:")
        for n in stats.created:
            print(f"    + {n}")
    if stats.skipped_existing:
        print("\n  skipped (already exist):")
        for n in stats.skipped_existing:
            print(f"    = {n}")
    if stats.failed:
        print("\n  failed:")
        for key, err, _tb in stats.failed:
            print(f"    ! {key}: {err}")


def _write_run_log(stats: IngestStats, log_path: Path, cfg: IngestConfig) -> None:
    """Persist a full record of the ingest run — successes, skips, and
    failures with their tracebacks — to `log_path`. Parent directories
    are created if missing."""
    log_path.parent.mkdir(parents=True, exist_ok=True)
    lines: list[str] = []
    lines.append(f"ido_data ingest run @ {datetime.now().isoformat(timespec='seconds')}")
    lines.append(f"manifest:  {cfg.manifest_csv_path}")
    lines.append(f"data_root: {cfg.data_root_folder}")
    lines.append("")
    lines.extend(stats.summary_lines())
    lines.append("")
    lines.append("=== created ===")
    if stats.created:
        for n in stats.created:
            lines.append(f"  + {n}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("=== skipped (already existed) ===")
    if stats.skipped_existing:
        for n in stats.skipped_existing:
            lines.append(f"  = {n}")
    else:
        lines.append("  (none)")
    lines.append("")
    lines.append("=== failed ===")
    if stats.failed:
        for key, err, tb in stats.failed:
            lines.append(f"  ! {key}")
            lines.append(f"    error: {err}")
            for tb_line in tb.rstrip().splitlines():
                lines.append(f"    | {tb_line}")
            lines.append("")
    else:
        lines.append("  (none)")
    log_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    # -------- USER-EDITED CONSTANTS (no CLI args) --------
    # Edit these three values, then run:
    #   venv/bin/python -m src.data.ido_data.ingest
    MANIFEST_CSV = Path("/path/to/manifest.csv")
    DATA_ROOT    = Path("/path/to/rootDataFolder")
    LOG_LEVEL    = "INFO"  # one of: "DEBUG", "INFO", "WARNING", "ERROR"
    # Run-log file (full per-experiment status + tracebacks for failures).
    # Default: `src/data/ido_data/logs/ingest_log_<UTC-timestamp>.txt`.
    LOG_DIR      = Path(__file__).resolve().parent / "logs"
    # -----------------------------------------------------

    logging.basicConfig(
        level=getattr(logging, LOG_LEVEL),
        format="%(levelname)s %(message)s",
    )

    cfg = IngestConfig(
        manifest_csv_path=MANIFEST_CSV,
        data_root_folder=DATA_ROOT,
    )
    stats = ingest(cfg)
    _print_summary(stats)

    log_path = LOG_DIR / f"ingest_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
    _write_run_log(stats, log_path, cfg)
    print(f"\nrun log written → {log_path}")

    return 0 if not stats.failed else 1


if __name__ == "__main__":
    sys.exit(main())
