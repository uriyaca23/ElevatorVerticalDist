import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist")
from src.algorithms.algo1_direct import estimate_height_direct
from src.algorithms.algo2_zupt import estimate_height_zupt
from src.algorithms.algo3_kalman import estimate_height_kalman

base_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\ADVIO"
metadata_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\metadata\elevator_segments.json"

def get_barometer_height_diff(baro_path, start_t, end_t):
    df = pd.read_csv(baro_path, header=None)
    t = df[0].values
    ans = df[2].values if df.shape[1] > 2 else None
    if ans is None: return np.nan
    
    # closest matching index
    idx_start = np.argmin(np.abs(t - start_t))
    idx_end = np.argmin(np.abs(t - end_t))
    return abs(ans[idx_end] - ans[idx_start])

def evaluate_all():
    with open(metadata_path, 'r') as f:
        segments = json.load(f)
        
    results = []
    
    for ds, runs in segments.items():
        accel_path = os.path.join(base_path, ds, "iphone", "accelerometer.csv")
        baro_path = os.path.join(base_path, ds, "iphone", "barometer.csv")
        
        accel_df = pd.read_csv(accel_path, header=None)
        t_acc = accel_df[0].values
        
        # ADVIO accelerometer is already in m/s^2!
        ax = accel_df[1].values
        ay = accel_df[2].values
        az = accel_df[3].values
        a_mag = np.sqrt(ax**2 + ay**2 + az**2)
        
        for idx, run in enumerate(runs):
            s_t = run["start_time"]
            e_t = run["end_time"]
            
            mask = (t_acc >= (s_t - 2.0)) & (t_acc <= (e_t + 2.0))
            t_sub = t_acc[mask]
            a_sub = a_mag[mask]
            
            # Estimate Gravity dynamically from the first second of data
            mask_rest = (t_sub < s_t)
            if np.any(mask_rest):
                gravity_est = np.mean(a_sub[mask_rest])
            else:
                gravity_est = np.mean(a_sub[:10]) # fallback to first 10 samples
                
            a_clean = a_sub - gravity_est
            
            h_direct = estimate_height_direct(t_sub, a_clean)
            h_zupt = estimate_height_zupt(t_sub, a_clean, accel_threshold=0.2)
            h_kalman = estimate_height_kalman(t_sub, a_clean, accel_threshold=0.2)
            
            final_direct = abs(h_direct[-1])
            final_zupt = abs(h_zupt[-1])
            final_kalman = abs(h_kalman[-1])
            
            gt_h = run["height_diff"]
            baro_h = get_barometer_height_diff(baro_path, s_t, e_t)
            
            results.append({
                "dataset": ds,
                "segment": idx,
                "GT": gt_h,
                "Barometer": baro_h,
                "Algo1_Direct": final_direct,
                "Algo2_ZUPT": final_zupt,
                "Algo3_Kalman": final_kalman,
                "start": s_t,
                "end": e_t
            })
            
            print(f"{ds} Seg {idx}: GT={gt_h:.2f}m | Baro={baro_h:.2f}m | Dir={final_direct:.2f}m | ZUPT={final_zupt:.2f}m | Kal={final_kalman:.2f}m")

    # Save results as CSV for later report generation
    res_df = pd.DataFrame(results)
    out_dir = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\metadata"
    res_df.to_csv(os.path.join(out_dir, "evaluation_results.csv"), index=False)

if __name__ == "__main__":
    evaluate_all()
