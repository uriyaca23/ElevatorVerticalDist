import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from elevator_detection_algorithms import ElevatorDetector
from algorithms.algo1_direct import estimate_height_direct
from algorithms.algo2_zupt import estimate_height_zupt
from algorithms.algo3_kalman import estimate_height_kalman
from evaluate_all import compute_iou

def process_combinations():
    acc_file = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")
    gt_file = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
    
    df_acc = pd.read_csv(acc_file, names=["time_ms", "x", "y", "z"])
    df_gt = pd.read_csv(gt_file)
    
    t_sec = df_acc["time_ms"].values / 1000.0
    acc_mag = np.sqrt(df_acc["x"]**2 + df_acc["y"]**2 + df_acc["z"]**2).values
    gt_h = df_gt["height_smooth"].values
    gt_t = df_gt["time_sec"].values
    
    dt = np.median(np.diff(t_sec))
    fs = 1.0 / dt
    detector = ElevatorDetector(fs=fs)
    
    gt_is_elev = df_gt["in_elevator"].values
    is_elev_acc = np.interp(t_sec, gt_t, gt_is_elev) > 0.5
    gt_segs = []
    in_s = False; start = -1
    for i in range(len(is_elev_acc)):
        if is_elev_acc[i] and not in_s: start=i; in_s=True
        elif not is_elev_acc[i] and in_s: gt_segs.append((start, i)); in_s=False
    if in_s: gt_segs.append((start, len(is_elev_acc)))
    
    # 1. Best Segments (Alg 1)
    seg_best = detector.detect_algorithm1_state_machine(t_sec, acc_mag, var_thresh=3.5, acc_thresh=0.35)
    # 2. Worse Seg 1 (Alg 3 with standard params)
    seg_worse_1 = detector.detect_algorithm3_integral(t_sec, acc_mag, var_thresh=2.0, min_v=0.8)
    # 3. Worse Seg 2 (Alg 3 with poor params)
    seg_worse_2 = detector.detect_algorithm3_integral(t_sec, acc_mag, var_thresh=1.0, min_v=0.1)

    combs = [
        ("BEST (Alg1 + ZUPT)", seg_best, estimate_height_zupt),
        ("COMP A (Alg1 + Direct)", seg_best, estimate_height_direct),
        ("COMP B (Alg1 + Kalman)", seg_best, estimate_height_kalman),
        ("COMP C (Alg3_A + ZUPT)", seg_worse_1, estimate_height_zupt),
        ("COMP D (Alg3_B + ZUPT)", seg_worse_2, estimate_height_zupt)
    ]
    
    results = {}
    fig, axes = plt.subplots(5, 1, figsize=(12, 18), sharex=True)
    if not isinstance(axes, np.ndarray): axes = [axes]
        
    for idx, (name, segs, height_func) in enumerate(combs):
        iou = compute_iou(segs, gt_segs, len(t_sec))
        
        errors = []
        
        # Plot styling
        axes[idx].plot(gt_t, gt_h, 'k--', label='Ground Truth', alpha=0.5)
        
        for s_idx, e_idx in segs:
            s_idx, e_idx = int(s_idx), int(e_idx)
            t_seg = t_sec[s_idx:e_idx]
            a_seg = acc_mag[s_idx:e_idx]
            
            # To evaluate isolated segment error, find overlapping GT segment
            overlap_gt = None
            max_over = 0
            for gs, ge in gt_segs:
                over = max(0, min(e_idx, ge) - max(s_idx, gs))
                if over > max_over:
                    max_over = over
                    overlap_gt = (gs, ge)
                    
            if overlap_gt is not None:
                # True absolute height difference
                h_start = gt_h[np.argmin(np.abs(gt_t - t_sec[overlap_gt[0]]))]
                h_end = gt_h[np.argmin(np.abs(gt_t - t_sec[overlap_gt[1]]))]
                true_delta = h_end - h_start
                
                # Model estimate (isolated)
                try:
                    g_est = np.mean(a_seg)
                    if height_func == estimate_height_zupt:
                        pos = height_func(t_seg, a_seg, gravity=g_est, accel_threshold=0.1)
                    else:
                        pos = height_func(t_seg, a_seg)
                    est_delta = pos[-1]
                    
                    err = abs(est_delta - true_delta)
                    errors.append(err)
                    
                    # Plot just this segment starting from its true starting height
                    # This visualizes how well the estimated trajectory matches the ride
                    start_plot_h = gt_h[np.argmin(np.abs(gt_t - t_sec[s_idx]))]
                    axes[idx].plot(t_seg, start_plot_h + np.array(pos), 'b-', alpha=0.8)
                    axes[idx].axvspan(t_sec[s_idx], t_sec[e_idx-1], color='red', alpha=0.1)
                except Exception as e:
                    pass
            
        mean_err = np.mean(errors) if len(errors) > 0 else float('inf')
        results[name] = {"iou": iou, "mean_height_err": float(mean_err)}
        
        # Draw a custom legend
        from matplotlib.lines import Line2D
        custom_lines = [Line2D([0], [0], color='k', linestyle='--', lw=2),
                        Line2D([0], [0], color='b', lw=2),
                        Line2D([0], [0], color='red', alpha=0.3, lw=4)]
        axes[idx].legend(custom_lines, ['GT Full Trajectory', 'Predicted Isolated Segments', 'Segment Bound'], loc='upper right')
        
        axes[idx].set_title(f"{name} | IoU: {iou*100:.1f}% | Avg Segment Error: {mean_err:.2f}m")
        axes[idx].grid(True)
        
    plt.tight_layout()
    os.makedirs("docs", exist_ok=True)
    plt.savefig("docs/four_comparisons_isolated.png", dpi=150)
    
    with open("docs/four_comparisons_isolated_results.json", "w") as f:
        json.dump(results, f, indent=4)
        
if __name__ == "__main__":
    process_combinations()
