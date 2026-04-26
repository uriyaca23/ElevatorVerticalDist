"""Survey real experiments to pick scenarios for the Boutique-Pipeline guide.

For each candidate experiment we:
  1. Load the structured ACC + GT data.
  2. Run the same detector the Streamlit app runs (predict_intervals).
  3. Compare detector output to gt.csv to count: clean matches, missed,
     FPs, merges, splits.
  4. Print a one-line summary so we can pick scenarios.
  5. Write a 2-column upload CSV (time_s, a_vert_ms2) per scenario, ready
     for the file-upload path of the Streamlit app.

Run:  python tmp_boutique_capture/survey.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import detect as _detect

OUT_DIR = REPO / "tmp_boutique_capture" / "csvs"
OUT_DIR.mkdir(parents=True, exist_ok=True)


CANDIDATES = [
    # (scenario_tag, exp_folder)
    ("clean_milleniumA23",  "UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2"),
    ("clean_beitMansourPix10", "UriyaCohenEliya_beitMansour1_GooglePixel10_15-04-2026_exp5"),
    ("clean_acroPix10",     "UriyaCohenEliya_acroBuilding_GooglePixel10_15-04-2026_exp4"),
    ("damped_BarIlanPix10", "UriyaCohenEliya_BarIlan2Herzelia_Pixel10_24-3-2026"),
    ("merged_split_Haari3", "RoyTurgeman_Haari3_SamsungGalaxyZFlip6_10-4-2026"),
    ("fp_milleniumA23exp1", "UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1"),
    ("noisy_xiaomi_beitMansour", "eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5"),
    ("clean_S23_milleniumOutside", "eyalyakir_milleniumOutside_SamsungSM-S911B_15-04-2026_exp3"),
]


def load_acc(exp: str) -> pd.DataFrame:
    p = REPO / "src" / "data" / "structuredData" / "data" / exp / "ACC.csv"
    if not p.exists():
        raise FileNotFoundError(p)
    df = pd.read_csv(p)
    return df[["timestamp_ms", "x", "y", "z"]].copy()


def load_gt(exp: str) -> pd.DataFrame:
    p = REPO / "src" / "data" / "structuredData" / "data" / exp / "gt.csv"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    return df


def project_vert(acc: pd.DataFrame) -> np.ndarray:
    """Approximate gravity-projected vertical acceleration.

    Strategy: average a-vector during the first 5 s of the trace (assumed
    stationary), use that as the gravity axis, project the rest onto it,
    and subtract the gravity magnitude. This matches what the Streamlit
    detector internally does on the upload path well enough for an
    upload-CSV.
    """
    t_ms = acc["timestamp_ms"].astype(float).to_numpy()
    t0 = t_ms[0]
    head = acc[t_ms - t0 < 5000.0]
    if len(head) < 50:
        head = acc.iloc[: max(50, len(acc) // 50)]
    g_vec = head[["x", "y", "z"]].mean().to_numpy()
    g_norm = np.linalg.norm(g_vec)
    if g_norm < 1e-3:
        g_unit = np.array([0.0, 0.0, 1.0])
    else:
        g_unit = g_vec / g_norm
    a = acc[["x", "y", "z"]].to_numpy()
    a_proj = a @ g_unit  # signed projection on gravity axis
    return a_proj - g_norm  # remove DC offset


def run_detector(acc: pd.DataFrame) -> tuple[list[dict], dict]:
    return _detect.predict_intervals(acc)


def overlap_classify(preds: list[dict], gt: pd.DataFrame, t0_ms: float) -> dict:
    """Cheap classifier: for each GT ride, count matches and unmatched.

    Normalises GT timestamps against the first ACC sample's timestamp so
    they align with detector seconds-from-start.
    """
    if gt.empty:
        return {"n_gt": 0, "n_pred": len(preds), "matched": 0,
                "miss": 0, "fp": len(preds), "matches": []}
    rides = gt[gt["type"].isin(["up", "down"])].copy()
    rides["start_s"] = (rides["start_ms"] - t0_ms) / 1000.0
    rides["end_s"]   = (rides["end_ms"]   - t0_ms) / 1000.0
    matched = 0
    pred_used = [False] * len(preds)
    matches: list[tuple[int, int, float]] = []
    for gi, g in rides.iterrows():
        gs, ge = float(g["start_s"]), float(g["end_s"])
        for pi, p in enumerate(preds):
            if pred_used[pi]:
                continue
            ps, pe = float(p["t_start_s"]), float(p["t_end_s"])
            ov = max(0.0, min(pe, ge) - max(ps, gs))
            iou_denom = max(pe, ge) - min(ps, gs)
            iou = ov / iou_denom if iou_denom > 0 else 0.0
            if iou > 0.3:
                pred_used[pi] = True
                matched += 1
                matches.append((gi, pi, iou))
                break
    fp = sum(1 for u in pred_used if not u)
    miss = len(rides) - matched
    return {
        "n_gt":     int(len(rides)),
        "n_pred":   int(len(preds)),
        "matched":  int(matched),
        "miss":     int(miss),
        "fp":       int(fp),
        "matches":  matches,
    }


def write_upload_csv(scenario: str, acc: pd.DataFrame, a_vert: np.ndarray) -> Path:
    """Write a 4-column CSV (time_s, ax, ay, az) so the patched upload
    handler can feed real 3-axis data into the detector. The single-axis
    projection upstream was breaking gravity estimation.
    """
    t_ms = acc["timestamp_ms"].astype(float).to_numpy()
    t_s = (t_ms - t_ms[0]) / 1000.0
    df = pd.DataFrame({
        "time_s": t_s,
        "ax_ms2": acc["x"].to_numpy(),
        "ay_ms2": acc["y"].to_numpy(),
        "az_ms2": acc["z"].to_numpy(),
    })
    out = OUT_DIR / f"{scenario}.csv"
    df.to_csv(out, index=False)
    return out


def main() -> None:
    summary = []
    for scenario, exp in CANDIDATES:
        try:
            acc = load_acc(exp)
            gt = load_gt(exp)
        except FileNotFoundError as e:
            print(f"[skip] {scenario}: {e}")
            continue

        a_vert = project_vert(acc)
        write_upload_csv(scenario, acc, a_vert)

        preds, state = run_detector(acc)
        t0_ms = float(acc["timestamp_ms"].iloc[0])
        info = overlap_classify(preds, gt, t0_ms)
        info["preds_summary"] = [
            {"i": i, "type": p["ride_type"],
             "t0": round(float(p["t_start_s"]), 1),
             "t1": round(float(p["t_end_s"]), 1),
             "joint_r2": round(float(p.get("joint_r2_mean", 0.0)), 3)}
            for i, p in enumerate(preds)
        ]
        info["scenario"] = scenario
        info["exp"] = exp
        info["duration_s"] = (
            (acc["timestamp_ms"].iloc[-1] - acc["timestamp_ms"].iloc[0]) / 1000.0
        )
        summary.append(info)

        print(
            f"{scenario:32s} | exp={exp[:60]:60s} "
            f"| dur={info['duration_s']:6.0f}s "
            f"| n_gt={info['n_gt']:3d} n_pred={info['n_pred']:3d} "
            f"| matched={info['matched']:3d} miss={info['miss']:3d} fp={info['fp']:3d}"
        )

    # Persist a JSON for the playwright capture step.
    import json
    out_json = OUT_DIR.parent / "scenarios.json"
    out_json.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nWrote {out_json}")


if __name__ == "__main__":
    main()
