import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from elevator_detection_algorithms import ElevatorDetector

def compute_iou(preds, gt_segments):
    # preds: list of (start_idx, end_idx)
    # gt_segments: list of (start_idx, end_idx)
    # Both sets are non-overlapping internally
    
    if len(preds) == 0 and len(gt_segments) == 0:
        return 1.0
    if len(preds) == 0 or len(gt_segments) == 0:
        return 0.0
        
    # We can discretize to a binary array
    max_idx = max([p[1] for p in preds] + [g[1] for g in gt_segments])
    
    pred_mask = np.zeros(max_idx, dtype=bool)
    gt_mask = np.zeros(max_idx, dtype=bool)
    
    for s, e in preds:
        pred_mask[s:e] = True
    for s, e in gt_segments:
        gt_mask[s:e] = True
        
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    
    return intersection / union if union > 0 else 0.0

def load_dataset(name="bar_ilan"):
    if name == "bar_ilan":
        acc_file = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")
        gt_file = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
        df_acc = pd.read_csv(acc_file, names=["time_ms", "x", "y", "z"])
        df_gt = pd.read_csv(gt_file)
        
        # We must align GT properly
        # Since gt is 10Hz and acc is 100Hz, let's map them to 100Hz timeline
        dt = df_acc["time_ms"].diff().median() / 1000.0
        fs = 1.0 / dt
        
        time_sec = df_acc["time_ms"].values / 1000.0
        acc_z = df_acc["z"].values 
        acc_mag = np.sqrt(df_acc["x"]**2 + df_acc["y"]**2 + df_acc["z"]**2).values
        
        # Build GT segments (in indices of acc_mag)
        gt_segments = []
        is_elev = np.zeros(len(acc_mag), dtype=bool)
        
        # interpolate gt onto acc
        gt_time = df_gt["time_sec"].values
        gt_val = df_gt["in_elevator"].values
        is_elev_acc = np.interp(time_sec, gt_time, gt_val) > 0.5
        
        # extract segments
        in_seg = False
        start = -1
        for i in range(len(is_elev_acc)):
            if is_elev_acc[i] and not in_seg:
                start = i
                in_seg = True
            elif not is_elev_acc[i] and in_seg:
                gt_segments.append((start, i))
                in_seg = False
        if in_seg:
            gt_segments.append((start, len(is_elev_acc)))
            
        return time_sec, acc_mag, gt_segments, fs

def main():
    print("Loading Bar Ilan dataset...")
    t, acc, gt_segs, fs = load_dataset("bar_ilan")
    
    detector = ElevatorDetector(fs=fs)
    
    print("\n--- Tuning Algorithm 1 (State Machine) ---")
    best_iou = 0
    best_params = {}
    
    for var_t in [0.5, 1.0, 1.5, 2.5, 3.5]:
        for acc_t in [0.1, 0.2, 0.3]:
            preds = detector.detect_algorithm1_state_machine(t, acc, var_thresh=var_t, acc_thresh=acc_t)
            iou = compute_iou(preds, gt_segs)
            if iou > best_iou:
                best_iou = iou
                best_params = {"var_thresh": var_t, "acc_thresh": acc_t}
    print(f"Alg 1 Best IoU: {best_iou*100:.1f}% | Params: {best_params}")

    print("\n--- Tuning Algorithm 3 (ZUPT Integral Bound) ---")
    best_iou3 = 0
    best_params3 = {}
    
    for var_t in [1.5, 2.5, 4.0, 6.0]:
        for min_v in [0.3, 0.5, 0.8]:
            preds = detector.detect_algorithm3_integral(t, acc, var_thresh=var_t, min_v=min_v, min_h=1.0)
            iou = compute_iou(preds, gt_segs)
            if iou > best_iou3:
                best_iou3 = iou
                best_params3 = {"var_thresh": var_t, "min_v": min_v}
    print(f"Alg 3 Best IoU: {best_iou3*100:.1f}% | Params: {best_params3}")

if __name__ == "__main__":
    main()
