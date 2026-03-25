import sys
import os
import pandas as pd

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), "..", "src"))
from elevator_detection_algorithms import ElevatorDetector

def main():
    print("Loading test data...")
    # We will test on Bar Ilan dataset from 1200s to 1400s
    acc_file = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")
    gt_file = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
    
    if not os.path.exists(acc_file) or not os.path.exists(gt_file):
        print("Dataset files missing!")
        return
        
    df_acc = pd.read_csv(acc_file, names=["time_ms", "x", "y", "z"])
    df_gt = pd.read_csv(gt_file)
    
    # 100Hz assumed for Android ACC (roughly)
    # Let's verify fs
    dt = df_acc["time_ms"].diff().median() / 1000.0
    fs = 1.0 / dt
    print(f"Detected ACC Sampling Frequency: {fs:.1f} Hz")
    
    detector = ElevatorDetector(fs=fs)
    
    time_sec = df_acc["time_ms"].values / 1000.0
    acc_z = df_acc["z"].values 
    # Use magnitude because phone could be in pocket (upside down), so Z isn't always gravity down!
    # Let's compute magnitude instead of Z to be totally rotation invariant!
    import numpy as np
    acc_mag = np.sqrt(df_acc["x"]**2 + df_acc["y"]**2 + df_acc["z"]**2).values
    
    # Test slice: 1200s to 1400s
    mask = (time_sec >= 1200.0) & (time_sec <= 1400.0)
    t_slice = time_sec[mask]
    a_slice = acc_mag[mask]
    
    print("Testing Algorithm 1 (State Machine):")
    seg1 = detector.detect_algorithm1_state_machine(t_slice, a_slice, var_thresh=0.8, acc_thresh=0.1)
    for s in seg1:
        print(f"  Det: {t_slice[s[0]]:.1f}s -> {t_slice[s[1]-1]:.1f}s")
        
    print("Testing Algorithm 2 (DTW):")
    # DTW is slow, so maybe a smaller slice 1200-1250?
    mask_short = (time_sec >= 1200.0) & (time_sec <= 1250.0)
    seg2 = detector.detect_algorithm2_dtw(time_sec[mask_short], acc_mag[mask_short], var_thresh=0.8, dtw_thresh=50.0)
    for s in seg2:
        print(f"  Det: {time_sec[mask_short][s[0]]:.1f}s -> {time_sec[mask_short][s[1]-1]:.1f}s")
        
    print("Testing Algorithm 3 (Integral):")
    seg3 = detector.detect_algorithm3_integral(t_slice, a_slice, var_thresh=0.8)
    for s in seg3:
        print(f"  Det: {t_slice[s[0]]:.1f}s -> {t_slice[s[1]-1]:.1f}s")
        
    print("\nGround Truth Elevators in 1200-1400s slice:")
    # GT masks
    gt_mask = (df_gt["time_sec"] >= 1200.0) & (df_gt["time_sec"] <= 1400.0) & df_gt["in_elevator"]
    gt_slice = df_gt[gt_mask]
    if len(gt_slice) > 0:
        for grp in gt_slice["elevator_segment_id"].unique():
            df_g = gt_slice[gt_slice["elevator_segment_id"] == grp]
            print(f"  GT: {df_g['time_sec'].min():.1f}s -> {df_g['time_sec'].max():.1f}s")

if __name__ == "__main__":
    main()
