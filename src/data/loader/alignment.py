"""forBarometer alignment: shift the secondary device's log onto the
primary's uptime timebase by matching recording start times, then swap the
primary's PRS with the aligned secondary PRS.

The 2-panel diagnostic (`forBarometer_alignment.png`) lets you eyeball
whether the start-time alignment actually puts the two motion signatures on
top of each other:

  * Top: primary vs secondary |a| overlaid + altitude on a twin y-axis.
  * Bottom: smoothed vertical velocity (same recipe as the pressure
    segmenter) with detected ride intervals shaded (green=up, red=down).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from .parsing import _parse_sensor_log


def _acc_magnitude(acc: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Return (timestamp_ms, |a|) arrays from an ACC DataFrame."""
    t = acc["timestamp_ms"].to_numpy(dtype=np.int64)
    mag = np.sqrt(
        acc["x"].to_numpy(dtype=float) ** 2
        + acc["y"].to_numpy(dtype=float) ** 2
        + acc["z"].to_numpy(dtype=float) ** 2
    )
    return t, mag


def _compute_start_offset(
    primary: dict[str, pd.DataFrame],
    secondary_frames: dict[str, pd.DataFrame],
) -> int:
    """Offset to add to secondary timestamps so its first sample lands at the
    primary's recording start.

    Assumes both devices were started at roughly the same wall-clock moment.
    """
    def _t0(frames: dict[str, pd.DataFrame]) -> int:
        for key in ("ACC", "PRS", "GYR"):
            if key in frames and len(frames[key]):
                return int(frames[key]["timestamp_ms"].iloc[0])
        raise ValueError("No sensors with data to compute t0")

    return _t0(primary) - _t0(secondary_frames)


def _smoothed_velocity(
    time_s: np.ndarray, height_m: np.ndarray,
    lowpass_sec: float = 8.0, smooth_sec: float = 3.0,
) -> np.ndarray:
    """Replicate the segmenter's vertical-velocity recipe.

    1. Rolling-mean low-pass on height (window = `lowpass_sec` * fs).
    2. Forward difference → instantaneous vz.
    3. Box-kernel smoothing (window = `smooth_sec` * fs).
    """
    if len(time_s) < 2:
        return np.zeros_like(time_s)

    dt = np.diff(time_s)
    fs = 1.0 / np.median(dt[dt > 0]) if np.any(dt > 0) else 1.0

    lp_win = max(1, int(round(lowpass_sec * fs)))
    z_lp = pd.Series(height_m).rolling(
        window=lp_win, center=True, min_periods=1,
    ).mean().to_numpy()

    vz = np.zeros_like(time_s)
    vz[1:] = np.diff(z_lp) / np.where(dt > 0, dt, np.nan)
    vz = np.nan_to_num(vz, nan=0.0)

    sm_win = max(1, int(round(smooth_sec * fs)))
    kernel = np.ones(sm_win) / sm_win
    return np.convolve(vz, kernel, mode="same")


def _plot_forBarometer_alignment(
    primary_acc: pd.DataFrame,
    secondary_acc_shifted: pd.DataFrame,
    secondary_prs_shifted: pd.DataFrame,
    offset_ms: int,
    out_path: Path,
) -> None:
    """Save the 2-panel start-time-alignment diagnostic."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    t0 = int(primary_acc["timestamp_ms"].iloc[0])

    prim_t, prim_mag = _acc_magnitude(primary_acc)
    sec_t, sec_mag = _acc_magnitude(secondary_acc_shifted)
    prim_sec = (prim_t - t0) / 1000.0
    sec_sec = (sec_t - t0) / 1000.0

    prs = secondary_prs_shifted.sort_values("timestamp_ms")
    prs_sec = (prs["timestamp_ms"].to_numpy(dtype=float) - t0) / 1000.0
    altitude = prs["GT_height_m"].to_numpy(dtype=float)

    # Match the pipeline's median-smoothing before segmentation.
    altitude_smooth = pd.Series(altitude).rolling(
        window=51, center=True, min_periods=1,
    ).median().to_numpy()

    # Run the segmenter on the aligned altitude to overlay GT on the velocity.
    segments = pd.DataFrame(columns=["start_ci", "end_ci", "type"])
    try:
        from src.segmentation.algorithms import (
            SEGMENT_ALGORITHM_CONFIG, SegmentAlgorithm, Segmenter,
        )
        height_frame = pd.DataFrame({"time": prs_sec, "height": altitude_smooth})
        cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
        segments = Segmenter(cfg).detect(height_frame)
    except Exception as e:
        print(f"[loader] could not run segmenter for diagnostic plot: "
              f"{type(e).__name__}: {e}")

    vz_smooth = _smoothed_velocity(prs_sec, altitude_smooth)

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 7), sharex=True)

    # ---- Panel 1: ACC overlaid + altitude ----
    ax1.plot(prim_sec, prim_mag, color="tab:blue", lw=0.7, alpha=0.8,
             label="primary |a|")
    ax1.plot(sec_sec, sec_mag, color="tab:orange", lw=0.7, alpha=0.55,
             label="secondary |a|")
    ax1.set_ylabel("|a| (m/s²)")
    ax1.set_title(f"ACC (both devices, start-time aligned) + altitude  "
                  f"(secondary offset = {offset_ms:+d} ms)")
    ax1.legend(loc="upper left")
    ax1.grid(True, alpha=0.3)

    ax1b = ax1.twinx()
    ax1b.plot(prs_sec, altitude, color="tab:green", lw=1.0,
              label="altitude (m)")
    ax1b.set_ylabel("altitude (m)", color="tab:green")
    ax1b.tick_params(axis="y", colors="tab:green")
    ax1b.legend(loc="upper right")

    # ---- Panel 2: ACC (both devices) + smoothed velocity + GT spans ----
    # Shade GT rides first so the signal lines draw on top.
    color_map = {"up": "tab:green", "down": "tab:red"}
    for _, row in segments.iterrows():
        s_lo, _ = row["start_ci"]
        _, e_hi = row["end_ci"]
        c = color_map.get(str(row["type"]), "tab:gray")
        ax2.axvspan(float(s_lo), float(e_hi), color=c, alpha=0.2)

    ax2.plot(prim_sec, prim_mag, color="tab:blue", lw=0.6, alpha=0.55,
             label="primary |a| (no baro)")
    ax2.plot(sec_sec, sec_mag, color="tab:orange", lw=0.6, alpha=0.45,
             label="secondary |a| (baro device)")
    ax2.set_ylabel("|a| (m/s²)")
    ax2.set_xlabel("time (s, primary base)")
    ax2.grid(True, alpha=0.3)

    ax2b = ax2.twinx()
    ax2b.plot(prs_sec, vz_smooth, color="black", lw=1.2,
              label="smoothed vz (m/s)")
    ax2b.axhline(0, color="gray", ls=":", lw=0.5)
    ax2b.set_ylabel("vz (m/s)")

    ax2.set_title(
        f"ACC (both) + smoothed vz from baro + GT "
        f"(green=up, red=down, {len(segments)} rides)"
    )
    # Combine legends from both y-axes
    h1, l1 = ax2.get_legend_handles_labels()
    h2, l2 = ax2b.get_legend_handles_labels()
    ax2.legend(h1 + h2, l1 + l2, loc="upper right")

    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _merge_secondary_prs(
    primary: dict[str, pd.DataFrame],
    primary_log: Path,
    secondary_log: Path,
    plot_out_path: Path | None,
) -> dict[str, pd.DataFrame]:
    """Parse secondary log, shift onto primary's timebase by start-time, swap PRS.

    Records a diagnostic PNG to `plot_out_path` when both logs have ACC.
    Gracefully falls back if the secondary lacks necessary sensors.
    """
    secondary_frames = _parse_sensor_log(secondary_log)
    if "PRS" not in secondary_frames or secondary_frames["PRS"].empty:
        print(f"[loader] forBarometer log lacks PRS rows: {secondary_log}; "
              f"keeping primary PRS")
        return primary

    offset = _compute_start_offset(primary, secondary_frames)

    prs = secondary_frames["PRS"].copy()
    prs["timestamp_ms"] = prs["timestamp_ms"] + offset
    prs = prs.sort_values("timestamp_ms").reset_index(drop=True)

    if plot_out_path is not None:
        has_prim_acc = "ACC" in primary and not primary["ACC"].empty
        has_sec_acc = "ACC" in secondary_frames and not secondary_frames["ACC"].empty
        if has_prim_acc and has_sec_acc:
            sec_acc = secondary_frames["ACC"].copy()
            sec_acc["timestamp_ms"] = sec_acc["timestamp_ms"] + offset
            try:
                _plot_forBarometer_alignment(
                    primary["ACC"], sec_acc, prs, offset, plot_out_path,
                )
            except Exception as e:
                print(f"[loader] forBarometer plot failed: "
                      f"{type(e).__name__}: {e}")
        else:
            print(f"[loader] skipping diagnostic plot: primary or secondary "
                  f"ACC missing (start-time offset = {offset:+d} ms)")

    primary["PRS"] = prs
    return primary
