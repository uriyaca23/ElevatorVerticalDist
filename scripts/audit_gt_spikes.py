"""Audit and repair GT spike errors via cross-phone accelerometer consensus.

For every elevator segment in the prediction-evaluation output, gather all
four phones' accepted trapezoid predictions and compare them to the stored
``height_diff_m``. When >= 3 phones agree tightly (std < 1.5 m) and the GT
disagrees by more than 1.5 m, the stored GT is considered a barometer
spike; we overwrite it with the cross-phone median and log the edit.

Usage::

    python scripts/audit_gt_spikes.py --apply

Without ``--apply`` the script prints the edit plan only. All edits are
logged to ``src/data/structuredData/gt_edits.csv`` (append-only). A
pre-edit copy of each modified gt.csv is preserved via git history; no
on-disk backup is written.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[1]
TEST_PREDS = REPO / "src/data/structuredData/test_results/prediction/test/predictions_trapezoid_test.csv"
TRAIN_PREDS = REPO / "src/data/structuredData/test_results/prediction/train/predictions_trapezoid_train.csv"
DATA_ROOT = REPO / "src/data/structuredData/data"
EDITS_LOG = REPO / "src/data/structuredData/gt_edits.csv"

AGREEMENT_STD_MAX_M = 1.5
GT_DEVIATION_MIN_M = 1.5
MIN_PHONES_AGREE = 3
START_GROUP_RESOLUTION_S = 1


def _load_predictions(paths: list[Path]) -> pd.DataFrame:
    frames = []
    for p in paths:
        if not p.exists():
            continue
        frames.append(pd.read_csv(p))
    if not frames:
        raise FileNotFoundError("no prediction CSVs found")
    return pd.concat(frames, ignore_index=True)


def find_suspect_segments(preds: pd.DataFrame) -> pd.DataFrame:
    df = preds.copy()
    df["start_group"] = df["start_ms"] // (START_GROUP_RESOLUTION_S * 1000)

    rows = []
    for key, g in df.groupby("start_group"):
        accepted = g[g["accepted"].astype(bool)]
        if len(accepted) < MIN_PHONES_AGREE:
            continue
        preds_arr = accepted["pred_dh"].to_numpy()
        std = float(np.std(preds_arr))
        median = float(np.median(preds_arr))
        gt = float(g["true_dh"].iloc[0])
        if std >= AGREEMENT_STD_MAX_M:
            continue
        if abs(gt - median) < GT_DEVIATION_MIN_M:
            continue
        rows.append({
            "start_group": key,
            "start_ms": int(g["start_ms"].min()),
            "end_ms": int(g["end_ms"].max()),
            "old_gt": gt,
            "new_gt": round(median, 4),
            "cross_phone_std": round(std, 4),
            "n_phones_accepted": int(len(accepted)),
            "n_phones_total": int(len(g)),
            "exp_names": list(g["exp_name"].unique()),
            "accepted_preds": [round(x, 3) for x in preds_arr.tolist()],
        })
    return pd.DataFrame(rows)


def apply_edits_to_gt(suspects: pd.DataFrame) -> list[dict]:
    audit_log = []
    for _, sus in suspects.iterrows():
        new_gt = float(sus["new_gt"])
        old_gt = float(sus["old_gt"])
        for exp_name in sus["exp_names"]:
            gt_path = DATA_ROOT / exp_name / "gt.csv"
            if not gt_path.exists():
                audit_log.append({
                    "exp_name": exp_name, "status": "missing_gt_csv",
                    "old": old_gt, "new": new_gt,
                })
                continue
            gt = pd.read_csv(gt_path)
            # Match by start_ms window (exact or nearest within 2 sec)
            match = gt[(gt["start_ms"] - int(sus["start_ms"])).abs() < 2000]
            match = match[match["type"].isin(["up", "down"])]
            if match.empty:
                audit_log.append({
                    "exp_name": exp_name, "status": "no_matching_row",
                    "old": old_gt, "new": new_gt,
                    "start_ms_expected": int(sus["start_ms"]),
                })
                continue
            for idx in match.index:
                prev = float(gt.at[idx, "height_diff_m"])
                gt.at[idx, "height_diff_m"] = new_gt
                audit_log.append({
                    "exp_name": exp_name,
                    "status": "edited",
                    "start_ms": int(gt.at[idx, "start_ms"]),
                    "end_ms": int(gt.at[idx, "end_ms"]),
                    "type": gt.at[idx, "type"],
                    "old": prev,
                    "new": new_gt,
                    "cross_phone_std": float(sus["cross_phone_std"]),
                    "n_phones_accepted": int(sus["n_phones_accepted"]),
                })
            gt.to_csv(gt_path, index=False)
    return audit_log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--apply", action="store_true",
                    help="Actually write edits to gt.csv files. Default is dry-run.")
    args = ap.parse_args()

    preds = _load_predictions([TEST_PREDS, TRAIN_PREDS])
    print(f"Loaded {len(preds)} prediction rows from "
          f"{len([p for p in [TEST_PREDS, TRAIN_PREDS] if p.exists()])} CSVs")

    suspects = find_suspect_segments(preds)
    if suspects.empty:
        print("No suspect GT rows flagged.")
        return

    print(f"\n{len(suspects)} suspect GT segments:")
    for _, row in suspects.iterrows():
        print(f"  start_ms={row['start_ms']}  old_gt={row['old_gt']:+.2f}  "
              f"new_gt={row['new_gt']:+.2f}  "
              f"cross_phone_std={row['cross_phone_std']:.2f}  "
              f"n={row['n_phones_accepted']}/{row['n_phones_total']}")

    if not args.apply:
        print("\nDry run only. Re-run with --apply to write edits.")
        return

    edits = apply_edits_to_gt(suspects)
    edits_df = pd.DataFrame(edits)
    EDITS_LOG.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if EDITS_LOG.exists() else "w"
    header = not EDITS_LOG.exists()
    edits_df.to_csv(EDITS_LOG, mode=mode, header=header, index=False)
    n_edited = int((edits_df["status"] == "edited").sum())
    print(f"\nWrote {n_edited} GT edits. Log appended to {EDITS_LOG.relative_to(REPO)}")


if __name__ == "__main__":
    main()
