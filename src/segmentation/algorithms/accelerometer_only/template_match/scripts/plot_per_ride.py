"""Generate one velocity-only PNG per GT ride for visual labeling.

Outputs:
    run_results/labels/{experimenter}/ride_{i:02d}_{type}.png

Run:
    python3 -m src.tests.segmentations.plot_per_ride
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[5]))

from src.data.loader import load_experimenter
from src.algorithms.segmentation_algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    Segmenter,
)
from src.tests.segmentations.main import build_acc_frame, build_height_frame
from src.algorithms.segmentation_algorithms.template_match.scripts.velocity_templates import ride_velocity


OUT_ROOT = Path(__file__).resolve().parents[1] / "labels"


def run(name: str) -> int:
    data = load_experimenter(name)
    t0_ms = float(data["ACC"]["timestamp_ms"].iloc[0])
    acc_frame = build_acc_frame(data["ACC"], t0_ms)
    height_frame = build_height_frame(data["PRS"], t0_ms)

    cfg = SEGMENT_ALGORITHM_CONFIG(algorithm=SegmentAlgorithm.PRESSURE_FILTER)
    segments = Segmenter(cfg).detect(height_frame)

    out_dir = OUT_ROOT / name
    out_dir.mkdir(parents=True, exist_ok=True)

    count = 0
    for i, row in segments.iterrows():
        ts, v, _ = ride_velocity(acc_frame, row["start_ci"][0], row["end_ci"][1])
        if v.size == 0:
            continue
        t_rel = ts - ts[0]
        fig, ax = plt.subplots(figsize=(6, 3.2))
        color = "tab:blue" if row["type"] == "up" else "tab:red"
        ax.plot(t_rel, v, color=color, lw=1.8)
        ax.axhline(0, color="k", lw=0.4)
        ax.set_xlabel("time from ride start (s)")
        ax.set_ylabel("velocity (m/s)")
        ax.set_title(f"{name} ride {i} ({row['type']})")
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(out_dir / f"ride_{int(i):02d}_{row['type']}.png", dpi=110)
        plt.close(fig)
        count += 1
    print(f"{name}: {count} PNGs -> {out_dir}")
    return count


def main() -> None:
    total = 0
    for name in ("uriya", "roy_turgeman"):
        total += run(name)
    print(f"Total: {total} ride PNGs written under {OUT_ROOT}")


if __name__ == "__main__":
    main()
