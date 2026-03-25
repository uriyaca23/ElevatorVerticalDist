import os
import sys
import pandas as pd
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from elevator_detection_algorithms import ElevatorDetector

def compute_iou(preds, gt_segments, max_time_idx):
    if len(preds) == 0 and len(gt_segments) == 0:
        return 1.0 # Perfect rejection!
    if len(preds) == 0 or len(gt_segments) == 0:
        return 0.0
        
    pred_mask = np.zeros(max_time_idx, dtype=bool)
    gt_mask = np.zeros(max_time_idx, dtype=bool)
    
    for s, e in preds:
        if s < max_time_idx and e <= max_time_idx:
            pred_mask[s:e] = True
    for s, e in gt_segments:
        if s < max_time_idx and e <= max_time_idx:
            gt_mask[s:e] = True
            
    intersection = np.sum(pred_mask & gt_mask)
    union = np.sum(pred_mask | gt_mask)
    return intersection / union if union > 0 else 0.0

def _get_advio_gt(advio_path):
    pose_files = [
        os.path.join(advio_path, "ground-truth", "pose.csv"),
        os.path.join(advio_path, "ground-truth", "poses.csv"),
    ]
    gt_path = None
    for p in pose_files:
        if os.path.exists(p):
            gt_path = p
            break
    if gt_path is None:
        return []
        
    df_pose = pd.read_csv(gt_path, header=None)
    time_sec = df_pose[0].values
    
    # In ADVIO, poses are often (time, tx, ty, tz, qx, qy, qz, qw)
    # The height is either ty or tz depending on the camera frame (often ty is down/up)
    # Let's check variance to find the height axis
    v_vars = [df_pose[1].var(), df_pose[2].var(), df_pose[3].var()]
    h_idx = np.argmin(v_vars) + 1 # The vertical axis usually has less walking variance but huge elevator shifts
    # Wait, vertical axis has lower variance during walking?
    # Actually, people walk horizontally so X/Z change a lot. Y (height) changes only during stairs/elevators.
    h = df_pose[h_idx].values
    
    # Smooth h
    h_smooth = pd.Series(h).rolling(window=10, center=True, min_periods=1).mean().values
    
    dt = np.diff(time_sec)
    dt = np.append(dt, dt[-1]) # pad
    vel = np.zeros(len(h_smooth))
    mask_dt = dt > 0
    vel[mask_dt] = np.diff(h_smooth, append=h_smooth[-1])[mask_dt] / dt[mask_dt]
    
    is_elev = np.abs(pd.Series(vel).rolling(window=10, center=True).mean()) > 0.35
    is_elev = is_elev.rolling(window=50, center=True, min_periods=1).max() == 1.0 # dialated
    
    segments = []
    in_seg = False
    start = -1
    for i in range(len(is_elev)):
        if is_elev[i] and not in_seg:
            start = i
            in_seg = True
        elif not is_elev[i] and in_seg:
            segments.append({"start_t": time_sec[start], "end_t": time_sec[i]})
            in_seg = False
    if in_seg:
        segments.append({"start_t": time_sec[start], "end_t": time_sec[-1]})
        
    # Filter short segments
    return [s for s in segments if s["end_t"] - s["start_t"] > 5.0]

def load_all_datasets():
    datasets = []
    
    # 1. Bar Ilan
    acc_file = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")
    gt_file = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
    if os.path.exists(acc_file) and os.path.exists(gt_file):
        df_acc = pd.read_csv(acc_file, names=["time_ms", "x", "y", "z"])
        df_gt = pd.read_csv(gt_file)
        
        t_sec = df_acc["time_ms"].values / 1000.0
        acc_mag = np.sqrt(df_acc["x"]**2 + df_acc["y"]**2 + df_acc["z"]**2).values
        dt = np.median(np.diff(t_sec))
        fs = 1.0 / dt if dt > 0 else 100.0
        
        # GT
        gt_time = df_gt["time_sec"].values
        is_elev = df_gt["in_elevator"].values
        # interpol
        is_elev_acc = np.interp(t_sec, gt_time, is_elev) > 0.5
        
        segs = []
        in_s = False
        start = -1
        for i in range(len(is_elev_acc)):
            if is_elev_acc[i] and not in_s:
                start = i
                in_s = True
            elif not is_elev_acc[i] and in_s:
                segs.append((start, i))
                in_s = False
        if in_s:
            segs.append((start, len(is_elev_acc)))
            
        datasets.append({"name": "bar_ilan", "t": t_sec, "acc": acc_mag, "fs": fs, "gt": segs})

    # 2. ADVIO
    adv_base = os.path.join("datasets", "ADVIO")
    if os.path.exists(adv_base):
        for fold in os.listdir(adv_base):
            fold_p = os.path.join(adv_base, fold)
            if not os.path.isdir(fold_p):
                continue
                
            acc_p = os.path.join(fold_p, "pixel", "accelerometer.csv")
            if not os.path.exists(acc_p):
                acc_p = os.path.join(fold_p, "iphone", "accelerometer.csv")
                
            if os.path.exists(acc_p):
                df_acc = pd.read_csv(acc_p, header=None)
                if df_acc.shape[1] < 4:
                    continue
                t_sec = df_acc[0].values
                # acc shape x,y,z
                acc_mag = np.sqrt(df_acc[1]**2 + df_acc[2]**2 + df_acc[3]**2).values
                
                dt = np.median(np.diff(t_sec))
                fs = 1.0 / dt if dt>0 else 100.0
                
                gt_time_segs = _get_advio_gt(fold_p)
                
                gt_idx = []
                for gs in gt_time_segs:
                    # find idx
                    s_idx = np.argmin(np.abs(t_sec - gs["start_t"]))
                    e_idx = np.argmin(np.abs(t_sec - gs["end_t"]))
                    gt_idx.append((s_idx, e_idx))
                    
                datasets.append({"name": fold, "t": t_sec, "acc": acc_mag, "fs": fs, "gt": gt_idx})
                
    return datasets

def main():
    print("Loading all datasets...")
    datasets = load_all_datasets()
    print(f"Loaded {len(datasets)} datasets.")
    
    # We will gridsearch params for Alg 1 (State Machine) and Alg 3 (Integral) across ALL datasets
    # trying to maximize average IoU.
    
    print("\n--- Tuning Algorithm 1 (State Machine) ---")
    best_iou1 = 0
    best_p1 = {}
    
    for var_t in [0.8, 1.5, 2.0, 3.5, 5.0]:
        for acc_t in [0.1, 0.2, 0.35]:
            total_iou = 0
            for d in datasets:
                detector = ElevatorDetector(fs=d["fs"])
                preds = detector.detect_algorithm1_state_machine(d["t"], d["acc"], var_thresh=var_t, acc_thresh=acc_t)
                total_iou += compute_iou(preds, d["gt"], len(d["t"]))
            avg = total_iou / len(datasets)
            if avg > best_iou1:
                best_iou1 = avg
                best_p1 = {"var_thresh": var_t, "acc_thresh": acc_t}
    print(f"BEST ALG 1: Avg IoU={best_iou1*100:.1f}%, Params={best_p1}")

    print("\n--- Tuning Algorithm 3 (Integral ZUPT) ---")
    best_iou3 = 0
    best_p3 = {}
    
    for var_t in [2.0, 3.5, 5.0]:
        for min_v in [0.3, 0.8]:
            total_iou = 0
            for d in datasets:
                detector = ElevatorDetector(fs=d["fs"])
                preds = detector.detect_algorithm3_integral(d["t"], d["acc"], var_thresh=var_t, min_v=min_v, min_h=2.0)
                total_iou += compute_iou(preds, d["gt"], len(d["t"]))
            avg = total_iou / len(datasets)
            if avg > best_iou3:
                best_iou3 = avg
                best_p3 = {"var_thresh": var_t, "min_v": min_v}
    print(f"BEST ALG 3: Avg IoU={best_iou3*100:.1f}%, Params={best_p3}")

if __name__ == "__main__":
    main()
