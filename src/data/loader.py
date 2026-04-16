"""Sensor log loader.

Loads `sensorLog_*.txt` files for a given experimenter and experiment number
under `src/data/rawData/<name>/<expN>/` into pandas DataFrames.

File format (tab-separated, no header), one sample per line:
    <timestamp_ms>\t<SENSOR>\t<value1>\t<value2>...

Sensor schemas:
    ACC, GYR, MAG       -> x, y, z
    RAWGYR, RAWMAG      -> x, y, z, bias_x, bias_y, bias_z
    ORI                 -> w, x, y, z   (quaternion)
    PRS                 -> pressure
    GPS                 -> lat, lon, alt

Public API:
    loadBasicData(name, exp)   -> dict[str, DataFrame] of per-sensor data
    loadDataWithGT(name, exp)  -> same, but PRS has a `gt_label` column
                                  (idle/up/down). Cached to
                                  `<exp>/data_with_gt.xlsx`; a diagnostic
                                  plot `<exp>/gt_plot.png` is written too.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

if __package__ in (None, ""):
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.physics import pressure_to_altitude

DATA_ROOT = Path(__file__).resolve().parent / "rawData"

SENSOR_COLUMNS: dict[str, list[str]] = {
    "ACC":    ["x", "y", "z"],
    "GYR":    ["x", "y", "z"],
    "MAG":    ["x", "y", "z"],
    "RAWGYR": ["x", "y", "z", "bias_x", "bias_y", "bias_z"],
    "RAWMAG": ["x", "y", "z", "bias_x", "bias_y", "bias_z"],
    "ORI":    ["w", "x", "y", "z"],
    "PRS":    ["pressure"],
    "GPS":    ["lat", "lon", "alt"],
}

GT_FILENAME = "data_with_gt.xlsx"
GT_PLOT_FILENAME = "gt_plot.png"


def _normalize_exp(exp: int | str) -> str:
    if isinstance(exp, int):
        return f"exp{exp}"
    s = str(exp).strip()
    return s if s.startswith("exp") else f"exp{s}"


def _resolve_exp_dir(name: str, exp: int | str, data_root: Path | str = DATA_ROOT) -> Path:
    data_root = Path(data_root)
    exp_dir = data_root / name / _normalize_exp(exp)
    if not exp_dir.is_dir():
        name_dir = data_root / name
        available = (
            sorted(p.name for p in name_dir.iterdir() if p.is_dir())
            if name_dir.is_dir() else []
        )
        raise FileNotFoundError(
            f"Experiment directory not found: {exp_dir}. "
            f"Available exps under {name_dir}: {', '.join(available) or '(none)'}"
        )
    return exp_dir


def _find_sensor_log(exp_dir: Path) -> Path:
    matches = sorted(exp_dir.glob("sensorLog_*.txt"))
    if not matches:
        raise FileNotFoundError(f"No sensorLog_*.txt found in {exp_dir}")
    return matches[0]


def _parse_sensor_log(log_path: Path) -> dict[str, pd.DataFrame]:
    buckets: dict[str, list[list[str]]] = {s: [] for s in SENSOR_COLUMNS}
    with log_path.open() as f:
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) < 3:
                continue
            ts, sensor, *values = parts
            cols = SENSOR_COLUMNS.get(sensor)
            if cols is None or len(values) != len(cols):
                continue
            buckets[sensor].append([ts, *values])

    frames: dict[str, pd.DataFrame] = {}
    for sensor, rows in buckets.items():
        if not rows:
            continue
        df = pd.DataFrame(rows, columns=["timestamp_ms", *SENSOR_COLUMNS[sensor]])
        df["timestamp_ms"] = pd.to_numeric(df["timestamp_ms"], downcast="integer")
        for col in SENSOR_COLUMNS[sensor]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[sensor] = df.sort_values("timestamp_ms").reset_index(drop=True)

    if "PRS" in frames:
        frames["PRS"]["GT_height_m"] = pressure_to_altitude(frames["PRS"]["pressure"])

    return frames


def loadBasicData(
    name: str,
    exp: int | str,
    data_root: Path | str = DATA_ROOT,
) -> dict[str, pd.DataFrame]:
    """Load per-sensor DataFrames for a single experiment (no GT)."""
    exp_dir = _resolve_exp_dir(name, exp, data_root)
    return _parse_sensor_log(_find_sensor_log(exp_dir))


def _annotate_prs_with_gt(prs: pd.DataFrame) -> pd.DataFrame:
    """Add a `gt_label` column ('idle' / 'up' / 'down') to the PRS frame using
    the barometer-based segment detector."""
    import importlib
    config_mod = importlib.import_module("src.algorithms.segmentation_algorithms.class")
    seg_mod = importlib.import_module(
        "src.algorithms.segmentation_algorithms.barometer_only.height_segmentation"
    )

    cfg = config_mod.PressureFilterConfig(time_col="time", height_col="height")
    seg_input = pd.DataFrame({
        "time": prs["timestamp_ms"].to_numpy(dtype=float) / 1000.0,
        "height": prs["GT_height_m"].to_numpy(dtype=float),
    })
    segments = seg_mod.detect_elevator_segments_from_height(seg_input, cfg)

    labels = np.full(len(prs), "idle", dtype=object)
    t = seg_input["time"].to_numpy()
    for _, row in segments.iterrows():
        s, _ = row["start_ci"]
        e, _ = row["end_ci"]
        mask = (t >= s) & (t <= e)
        labels[mask] = row["type"]

    out = prs.copy()
    out["gt_label"] = labels
    return out


def _plot_gt(prs: pd.DataFrame, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t = prs["timestamp_ms"].to_numpy(dtype=float) / 1000.0
    t = t - t[0]
    height = prs["GT_height_m"].to_numpy(dtype=float)
    labels = prs["gt_label"].to_numpy()

    fig, (ax_h, ax_p) = plt.subplots(2, 1, figsize=(14, 6), sharex=True)
    ax_h.plot(t, height, color="black", lw=0.8, label="height (m)")
    ax_p.plot(t, prs["pressure"].to_numpy(dtype=float), color="gray", lw=0.8,
              label="pressure")

    color_map = {"up": "tab:green", "down": "tab:red"}
    in_seg = False
    seg_start = 0
    seg_label: str | None = None
    for i in range(len(t)):
        lab = labels[i]
        if lab in color_map and not in_seg:
            in_seg = True
            seg_start = i
            seg_label = lab
        elif (lab != seg_label) and in_seg:
            ax_h.axvspan(t[seg_start], t[i - 1], color=color_map[seg_label], alpha=0.25)
            ax_p.axvspan(t[seg_start], t[i - 1], color=color_map[seg_label], alpha=0.25)
            if lab in color_map:
                seg_start = i
                seg_label = lab
            else:
                in_seg = False
                seg_label = None
    if in_seg and seg_label is not None:
        ax_h.axvspan(t[seg_start], t[-1], color=color_map[seg_label], alpha=0.25)
        ax_p.axvspan(t[seg_start], t[-1], color=color_map[seg_label], alpha=0.25)

    ax_h.set_ylabel("Height (m)")
    ax_p.set_ylabel("Pressure")
    ax_p.set_xlabel("Time (s)")
    ax_h.set_title("Barometer with GT classifications (green=up, red=down)")
    for ax in (ax_h, ax_p):
        ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _read_gt_excel(xlsx_path: Path) -> dict[str, pd.DataFrame]:
    sheets = pd.read_excel(xlsx_path, sheet_name=None)
    return {name: df for name, df in sheets.items()}


def _write_gt_excel(frames: dict[str, pd.DataFrame], xlsx_path: Path) -> None:
    with pd.ExcelWriter(xlsx_path, engine="openpyxl") as writer:
        for sensor, df in frames.items():
            # Excel sheet name max length is 31 chars; sensor names are short.
            df.to_excel(writer, sheet_name=sensor, index=False)


def loadDataWithGT(
    name: str,
    exp: int | str,
    data_root: Path | str = DATA_ROOT,
    force: bool = False,
) -> dict[str, pd.DataFrame]:
    """Load per-sensor DataFrames with a barometer-derived GT label on PRS.

    Caches the result to `<exp>/data_with_gt.xlsx`. If the cache exists and
    ``force=False``, reads directly from the Excel file and skips parsing.
    Also writes `<exp>/gt_plot.png` showing height/pressure with the GT spans.
    """
    exp_dir = _resolve_exp_dir(name, exp, data_root)
    xlsx_path = exp_dir / GT_FILENAME

    if xlsx_path.exists() and not force:
        return _read_gt_excel(xlsx_path)

    frames = _parse_sensor_log(_find_sensor_log(exp_dir))
    if "PRS" not in frames:
        raise ValueError(f"No PRS (barometer) data in {exp_dir}; cannot build GT.")

    frames["PRS"] = _annotate_prs_with_gt(frames["PRS"])
    _write_gt_excel(frames, xlsx_path)
    _plot_gt(frames["PRS"], exp_dir / GT_PLOT_FILENAME)
    return frames


# ---- Backward compatibility ----------------------------------------------

def load_experimenter(
    experimenter: str,
    data_root: Path | str = DATA_ROOT,
) -> dict[str, pd.DataFrame]:
    """Legacy entrypoint. Loads the first available experiment.

    If `rawData/<name>/` contains `expN` subfolders, uses the lowest-numbered
    one. Otherwise falls back to reading `sensorLog_*.txt` directly from
    `rawData/<name>/`.
    """
    data_root = Path(data_root)
    name_dir = data_root / experimenter
    if not name_dir.is_dir():
        raise FileNotFoundError(f"Experimenter directory not found: {name_dir}")

    exps = sorted(p for p in name_dir.iterdir()
                  if p.is_dir() and p.name.startswith("exp"))
    if exps:
        return _parse_sensor_log(_find_sensor_log(exps[0]))
    return _parse_sensor_log(_find_sensor_log(name_dir))


if __name__ == "__main__":
    import sys
    name = sys.argv[1] if len(sys.argv) > 1 else "eyal"
    exp = sys.argv[2] if len(sys.argv) > 2 else 1
    data = loadDataWithGT(name, exp)
    for sensor, df in data.items():
        print(f"{sensor}: {len(df):>7} rows  cols={list(df.columns)}")
