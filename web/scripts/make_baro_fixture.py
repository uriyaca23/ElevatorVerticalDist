"""Generate parity fixtures for web/lib/barometer.ts from the Python segmenter.

Run from the repo root with the project venv:

    venv/bin/python web/scripts/make_baro_fixture.py

Writes web/tests/fixtures/baro_parity.json containing
  - "synthetic": a small deterministic pressure trace (committed, always tested)
    plus the rides the Python pipeline finds on it;
  - "experiments": expected rides for a few local structuredData experiments
    (the TS test reads their PRS.csv directly and skips when the corpus is absent).
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from pyramidElevatorDist.physics.barometric import pressure_to_altitude  # noqa: E402
from src.data.loader.pipeline import _derive_gt_from_prs  # noqa: E402

DATA_DIR = ROOT / "src/data/structuredData/data"
OUT = ROOT / "web/tests/fixtures/baro_parity.json"

EXPERIMENTS = [
    "eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2",
    "RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026",
    "UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5",
    "UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4__corrupted_dropout",
]


def rides(prs: pd.DataFrame) -> list[dict]:
    gt = _derive_gt_from_prs(prs)
    gt = gt[gt["type"] != "outside"]
    return [
        {"start_ms": int(r.start_ms), "end_ms": int(r.end_ms), "type": str(r.type)}
        for r in gt.itertuples()
    ]


def synthetic() -> dict:
    # ~10 Hz with jitter, three rides (up 9 m, down 15 m, short 1 m bump that
    # must be rejected), plus sensor noise.
    rng = np.random.default_rng(7)
    n = 3000
    dt_ms = rng.integers(95, 106, size=n)
    ts = 1_760_000_000_000 + np.concatenate([[0], np.cumsum(dt_ms[1:])])
    t = (ts - ts[0]) / 1000.0
    h = np.zeros(n)
    for t0, dur, dh in [(40, 12, 9.0), (120, 18, -15.0), (220, 4, 1.0)]:
        frac = np.clip((t - t0) / dur, 0, 1)
        h += dh * (frac - np.sin(2 * np.pi * frac) / (2 * np.pi))
    h += 30.0 + rng.normal(0, 0.08, size=n)
    # Round so the JSON round-trip feeds TS exactly the values Python used.
    p = np.round(1013.25 * (1 - h / 44330.77) ** 5.255, 6)
    prs = pd.DataFrame({"timestamp_ms": ts.astype(np.int64), "pressure": p})
    prs["GT_height_m"] = pressure_to_altitude(prs["pressure"])
    return {
        "timestamp_ms": prs["timestamp_ms"].tolist(),
        "pressure": prs["pressure"].tolist(),
        "rides": rides(prs),
    }


def main() -> None:
    experiments = {}
    for name in EXPERIMENTS:
        path = DATA_DIR / name / "PRS.csv"
        if not path.exists():
            print(f"skip {name}: no PRS.csv")
            continue
        prs = pd.read_csv(path)
        experiments[name] = rides(prs)
        print(f"{name}: {len(experiments[name])} rides")
    syn = synthetic()
    print(f"synthetic: {len(syn['rides'])} rides")
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"synthetic": syn, "experiments": experiments}))
    print(f"wrote {OUT.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
