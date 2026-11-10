"""Per-experiment overview PNGs for the TRAIN set.

For each training experiment, save a two-panel figure (|acc| + smoothed
vertical velocity with GT shaded) under
``template_match/labels/experiment_overview/<exp>.png``.

Run:
    venv/bin/python -m src.segmentation.algorithms.accelerometer_only.template_match.plot_experiment_overviews
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loader import getExperimentData, list_experiments  # noqa: E402
from src.plotting import plot_experiment_overview  # noqa: E402

OVERVIEW_DIR = Path(__file__).with_name("labels") / "experiment_overview"


def main() -> int:
    OVERVIEW_DIR.mkdir(parents=True, exist_ok=True)
    names = list_experiments(kind="train")
    print(f"processing {len(names)} TRAIN experiments → {OVERVIEW_DIR}")

    written = 0
    for name in names:
        try:
            sensors, gt, _meta = getExperimentData(name)
        except Exception as exc:
            print(f"[error] {name}: {type(exc).__name__}: {exc}")
            continue
        if "ACC" not in sensors or sensors["ACC"].empty:
            print(f"[skip]  {name}: no ACC")
            continue

        out_path = OVERVIEW_DIR / f"{name}.png"
        fig = plot_experiment_overview(
            sensors["ACC"], gt,
            name=name,
            save_path=out_path,
            show=False,
        )
        plt.close(fig)
        written += 1
        print(f"[ok]    {name}")

    print(f"\nwrote {written} overviews")
    return 0


if __name__ == "__main__":
    sys.exit(main())
