"""Legacy `rawData/` loader with Excel-based GT caching.

Pre-pipeline code paths. Most callers use `load_experimenter` to grab the
first available experiment for a name in `rawData/`. `loadDataWithGT` adds a
barometer-derived `gt_label` column to the PRS frame and caches everything
to an Excel workbook alongside a diagnostic PNG.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .constants import DATA_ROOT, GT_FILENAME, GT_PLOT_FILENAME
from .parsing import (
    _find_sensor_log,
    _parse_sensor_log,
    _resolve_exp_dir,
)


# --------------------------------------------------------------------------
# Private helpers — only used by `loadDataWithGT`
# --------------------------------------------------------------------------

def _annotate_prs_with_gt(prs: pd.DataFrame) -> pd.DataFrame:
    """Add a `gt_label` column ('idle' / 'up' / 'down') to the PRS frame using
    the barometer-based segment detector."""
    from src.segmentation.algorithms.configTypes import PressureFilterConfig
    from src.segmentation.algorithms.barometer_only.height_segmentation import (
        HeightSegmenter,
    )

    cfg = PressureFilterConfig(time_col="time", height_col="height")
    seg_input = pd.DataFrame({
        "time": prs["timestamp_ms"].to_numpy(dtype=float) / 1000.0,
        "height": prs["GT_height_m"].to_numpy(dtype=float),
    })
    segments = HeightSegmenter(cfg).segment(seg_input)

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


# --------------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------------

def loadBasicData(
    name: str,
    exp: int | str,
    data_root: Path | str = DATA_ROOT,
) -> dict[str, pd.DataFrame]:
    """Load per-sensor DataFrames for a single experiment (no GT)."""
    exp_dir = _resolve_exp_dir(name, exp, data_root)
    return _parse_sensor_log(_find_sensor_log(exp_dir))


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
