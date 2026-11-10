"""Generate parity fixtures for web/lib/gpsStay.ts and web/lib/gnss.ts.

Run from the repo root with the project venv (gps_stay and nahori are sibling
repos, put on sys.path from --gps-stay / --nahori-src):

    venv/bin/python web/scripts/make_gps_stay_fixture.py

Writes web/tests/fixtures/gps_stay_parity.json containing
  - "synthetic": deterministic GPS tracks (inputs committed, always tested)
    and the segments gps_stay's ProbabilisticStayDetector finds on them;
  - "sessions": expected segments for the Shaft Lab sessions cached by
    scripts/house_entrance_figures.py (the TS test re-parses their loc.csv and
    skips when the cache is absent), plus a summary of nahori's per-second
    NMEA mean_snr for the TS parser to match.

The loc.csv preparation below mirrors parseLocCsv in web/lib/gnss.ts: the most
accurate fix of each second, ties to the first in file order.
"""

from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import math
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / "web/tests/fixtures/gps_stay_parity.json"
CACHE = ROOT / "elevator_reports/house_entrance/cache"


def segments(detector, t_ms, lat, lon) -> list[dict]:
    gps = pd.DataFrame({
        "timestamp": pd.to_datetime(np.asarray(t_ms, dtype="int64"), unit="ms", utc=True),
        "lat": np.asarray(lat, dtype=float),
        "lon": np.asarray(lon, dtype=float),
    })
    out = []
    for s in detector.detect(gps):
        out.append({
            "start_ms": int(pd.Timestamp(s.startTime, tz="UTC").value // 1_000_000),
            "end_ms": int(pd.Timestamp(s.endTime, tz="UTC").value // 1_000_000),
            "classification": str(getattr(s.classification, "value", s.classification)).lower(),
            "center_lat": s.centerLat,
            "center_lon": s.centerLon,
            "point_count": s.pointCount,
            "start_index": s.startIndex,
            "end_index": s.endIndex,
            "probability": s.probability,
        })
    return out


# --------------------------------------------------------------------------- synthetic

M_PER_DEG = 111_320.0
LAT0, LON0 = 32.0853, 34.7818


def _track(rng, legs, cadence_s: float, jitter_s: float = 0.0, throw_rate: float = 0.0):
    """Piecewise path: ("stay", seconds) holds position, ("move", seconds, vx, vy) moves (m/s),
    ("gap", seconds) emits nothing. Noise: Student-t-ish (Gaussian + rare throws)."""
    t, x, y = 0.0, 0.0, 0.0
    ts, xs, ys = [], [], []
    for leg in legs:
        kind, dur = leg[0], leg[1]
        vx, vy = (leg[2], leg[3]) if kind == "move" else (0.0, 0.0)
        end = t + dur
        while t < end:
            if kind != "gap":
                nx, ny = rng.normal(0, 6.0, 2)
                if rng.random() < throw_rate:
                    nx, ny = rng.normal(0, 150.0, 2)
                ts.append(t)
                xs.append(x + nx)
                ys.append(y + ny)
            step = cadence_s + (rng.uniform(-jitter_s, jitter_s) if jitter_s else 0.0)
            x += vx * step
            y += vy * step
            t += step
    t_ms = 1_792_000_000_000 + np.round(np.asarray(ts) * 1000).astype("int64")
    lat = LAT0 + np.asarray(ys) / M_PER_DEG
    lon = LON0 + np.asarray(xs) / (M_PER_DEG * math.cos(math.radians(LAT0)))
    return t_ms, np.round(lat, 7), np.round(lon, 7)


def synthetic() -> dict:
    rng = np.random.default_rng(11)
    cases = {
        # 1 Hz with jitter: stay, walk, stay, curving drive, stay; a few throws.
        "dense_1hz": _track(rng, [
            ("stay", 720), ("move", 360, 1.4, 0.3), ("stay", 600),
            ("move", 120, 12.0, 2.0), ("move", 120, 3.0, -11.0), ("stay", 420),
        ], cadence_s=1.0, jitter_s=0.2, throw_rate=0.02),
        # One fix per ~45 s, a 40-min silence inside a stay, then a slow walk.
        "sparse_45s": _track(rng, [
            ("stay", 2400), ("gap", 2400), ("stay", 1800), ("move", 900, 1.2, 1.0),
            ("stay", 1500), ("move", 600, 9.0, 0.0), ("stay", 1200),
        ], cadence_s=45.0, jitter_s=10.0),
        # 0.1 Hz for ~6 h — more than dp_window_min_points (1024) fixes, with a
        # 3 h stay, so the self-widening DP window is exercised.
        "long_10s": _track(rng, [
            ("stay", 10800), ("move", 1200, 8.0, 3.0), ("stay", 4800),
            ("move", 600, -1.3, 0.2), ("stay", 3600),
        ], cadence_s=10.0, jitter_s=1.0, throw_rate=0.01),
    }
    return {name: {"t_ms": t.tolist(), "lat": la.tolist(), "lon": lo.tolist()} for name, (t, la, lo) in cases.items()}


# --------------------------------------------------------------------------- cached sessions

def _num(s: str | None) -> float:
    if s is None or not s.strip():
        return math.nan
    try:
        return float(s)
    except ValueError:
        return math.nan


def parse_loc(path: Path):
    """Mirror of parseLocCsv (web/lib/gnss.ts)."""
    best: dict[int, tuple] = {}
    with path.open(newline="") as fh:
        reader = csv.reader(fh)
        header = [h.strip() for h in next(reader)]
        it, ila, ilo, iac = (header.index(c) for c in ("timestamp", "latitude", "longitude", "accuracy"))
        for row in reader:
            if not row:
                continue
            iso = re.sub(r"([+-]\d\d)(\d\d)$", r"\1:\2", row[it])
            ts = pd.to_datetime(iso, utc=True, format="ISO8601", errors="coerce")
            lat, lon = _num(row[ila]), _num(row[ilo])
            if pd.isna(ts) or not (math.isfinite(lat) and math.isfinite(lon)):
                continue
            if abs(lat) > 90 or abs(lon) > 180 or (lat == 0 and lon == 0):
                continue
            t = int(ts.value // 1_000_000)
            acc = _num(row[iac])
            rank = acc if math.isfinite(acc) else math.inf
            sec = t // 1000
            prev = best.get(sec)
            if prev is None or rank < prev[3]:
                best[sec] = (t, lat, lon, rank)
    rows = sorted(best.values())
    return [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]


def nmea_summary(parsed, path: Path) -> dict:
    """nahori's per-second mean_snr where it is defined (wall-clock-as-UTC ms)."""
    agg = parsed.parse_nmea_txt(path.read_bytes()).aggregate
    agg = agg[agg["mean_snr"].notna()]
    snr = agg["mean_snr"].astype(float).to_numpy()
    t = agg["t_ms"].astype("int64").to_numpy()
    pick = np.linspace(0, len(t) - 1, num=min(25, len(t)), dtype=int) if len(t) else []
    return {
        "count": int(len(t)),
        "sum": float(snr.sum()),
        "samples": [[int(t[i]), float(snr[i])] for i in pick],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--gps-stay", type=Path, default=ROOT.parent / "gps_stay")
    ap.add_argument("--nahori-src", type=Path, default=ROOT.parent / "nahori" / "src" / "algorithm" / "src")
    args = ap.parse_args()
    sys.path.insert(0, str(args.gps_stay))
    from gps_stay import ProbabilisticStayDetector

    parsed = None
    parsed_py = args.nahori_src / "data" / "parsed.py"
    if parsed_py.exists():
        # Load the one module directly: the `data` package imports cloud SDKs.
        spec = importlib.util.spec_from_file_location("nahori_parsed", parsed_py)
        parsed = importlib.util.module_from_spec(spec)
        sys.modules["nahori_parsed"] = parsed
        spec.loader.exec_module(parsed)

    det = ProbabilisticStayDetector()
    synth = synthetic()
    for case in synth.values():
        case["segments"] = segments(det, case["t_ms"], case["lat"], case["lon"])

    sessions = {}
    for d in sorted(CACHE.glob("*/"), key=lambda p: (len(p.name), p.name)) if CACHE.exists() else []:
        if not (d / "loc.csv").exists():
            continue
        t, la, lo = parse_loc(d / "loc.csv")
        entry = {"n_fixes": len(t), "segments": segments(det, t, la, lo) if t else []}
        if parsed is not None and (d / "nmea.txt").exists():
            entry["nmea"] = nmea_summary(parsed, d / "nmea.txt")
        sessions[d.name] = entry

    OUT.write_text(json.dumps({"synthetic": synth, "sessions": sessions}, separators=(",", ":")))
    for name, case in synth.items():
        print(f"synthetic {name}: {len(case['t_ms'])} fixes -> {[s['classification'] for s in case['segments']]}")
    for name, e in sessions.items():
        print(f"session {name}: {e['n_fixes']} fixes -> {len(e['segments'])} segments")
    print(f"wrote {OUT.relative_to(ROOT)} ({OUT.stat().st_size / 1024:.0f} KB)")


if __name__ == "__main__":
    main()
