#!/usr/bin/env python3
"""
Generate example dataset CSV from the Bar-Ilan dataset for testing
run_custom_evaluation.py.

This script reads the Bar-Ilan metadata and creates a properly formatted
CSV that can be used with run_custom_evaluation.py.
"""

import os
import sys
import numpy as np
import pandas as pd

def main():
    # Load Bar-Ilan metadata
    meta_path = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
    acc_path = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")

    if not os.path.exists(meta_path) or not os.path.exists(acc_path):
        print("Bar-Ilan dataset not found. Cannot generate example.")
        sys.exit(1)

    df_meta = pd.read_csv(meta_path)

    # Extract GT rides
    gt_ids = sorted([x for x in df_meta["elevator_segment_id"].unique() if x >= 0])

    rows = []
    for sid in gt_ids:
        sub = df_meta[df_meta["elevator_segment_id"] == sid]
        t_start = sub["time_sec"].iloc[0]
        t_end = sub["time_sec"].iloc[-1]
        true_dh = sub["height_smooth"].iloc[-1] - sub["height_smooth"].iloc[0]
        phone = sub["phone_position"].iloc[0]

        rows.append({
            "segment_id": int(sid),
            "acc_data_path": acc_path,
            "start_time": round(t_start, 2),
            "end_time": round(t_end, 2),
            "true_height": round(true_dh, 2),
            "phone_position": phone,
        })

    df_out = pd.DataFrame(rows)
    out_path = os.path.join("datasets", "bar_ilan_eval_example.csv")
    df_out.to_csv(out_path, index=False)
    print(f"Generated {out_path} with {len(rows)} segments")
    print(df_out.to_string())


if __name__ == "__main__":
    main()
