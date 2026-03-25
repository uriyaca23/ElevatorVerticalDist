import os
import sys
import numpy as np
import random
import json

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from evaluate_all import load_all_datasets
from algorithms.algo2_zupt import estimate_height_zupt
from algorithms.zupt_confidence import ZuptConfidenceAnalyzer

def train_test_pipeline():
    print("Loading all combined datasets for Train/Test split...")
    datasets = load_all_datasets()
    
    # We must extract all VALID elevator segments from the Ground Truth
    # so we can train ZUPT heights against them! 
    # But wait, ADVIO GT doesn't give precise true metric floor height, it gives path pose Z!
    # Bar Ilan definitely gives GT heights.
    # Let's extract the actual ground truth segments and calculate their GT Height.
    
    all_segments = []
    
    for d in datasets:
        # Evaluate Bar Ilan 
        if d["name"] == "bar_ilan":
            # For Bar Ilan, we have metadata_calibrated.csv or metadata.csv with exact heights!
            import pandas as pd
            gt_file = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
            df_gt = pd.read_csv(gt_file)
            gt_time = df_gt["time_sec"].values
            gt_heights = df_gt["height_smooth"].values
            
            for s_idx, e_idx in d["gt"]:
                s_t = d["t"][s_idx]
                e_t = d["t"][e_idx - 1]
                
                # find closest in GT
                s_h = gt_heights[np.argmin(np.abs(gt_time - s_t))]
                e_h = gt_heights[np.argmin(np.abs(gt_time - e_t))]
                true_height = e_h - s_h
                
                a_slice = d["acc"][s_idx:e_idx]
                t_slice = d["t"][s_idx:e_idx]
                all_segments.append({
                    "dataset": d["name"],
                    "acc": a_slice,
                    "t": t_slice,
                    "true_height": true_height,
                    "phone": "Pixel" # default for Bar Ilan
                })
        else:
            # ADVIO
            # the GT path pose gives us heights.
            # load it
            advio_path = os.path.join("datasets", "ADVIO", d["name"])
            pose_path = os.path.join(advio_path, "ground-truth", "pose.csv")
            if not os.path.exists(pose_path):
                 pose_path = os.path.join(advio_path, "ground-truth", "poses.csv")
            if os.path.exists(pose_path):
                df_pose = pd.read_csv(pose_path, header=None)
                v_vars = [df_pose[1].var(), df_pose[2].var(), df_pose[3].var()]
                h_idx = np.argmin(v_vars) + 1
                pos_t = df_pose[0].values
                pos_h = df_pose[h_idx].values
                
                for s_idx, e_idx in d["gt"]:
                    if e_idx <= s_idx: continue
                    s_t = d["t"][s_idx]
                    e_t = d["t"][e_idx - 1]
                    s_h = pos_h[np.argmin(np.abs(pos_t - s_t))]
                    e_h = pos_h[np.argmin(np.abs(pos_t - e_t))]
                    true_height = e_h - s_h
                    
                    a_slice = d["acc"][s_idx:e_idx]
                    t_slice = d["t"][s_idx:e_idx]
                    phone = "iPhone" if "iphone" in os.listdir(advio_path) else "Pixel"
                    all_segments.append({
                        "dataset": d["name"],
                        "acc": a_slice,
                        "t": t_slice,
                        "true_height": true_height,
                        "phone": phone
                    })
                    
    print(f"Extracted {len(all_segments)} raw elevator ride segments across all combined data.")
    
    # 50/50 Train / Test split
    random.seed(42)
    random.shuffle(all_segments)
    split_idx = int(len(all_segments) * 0.5)
    train_set = all_segments[:split_idx]
    test_set = all_segments[split_idx:]
    
    print(f"Train samples: {len(train_set)}, Test samples: {len(test_set)}")
    
    analyzer = ZuptConfidenceAnalyzer(dt=0.01)
    
    # Fit conformal on Train
    errors = []
    sigmas = []
    
    for seg in train_set:
        # Run ZUPT
        # Add slight margins
        try:
            g_est = np.mean(seg["acc"])
            pos = estimate_height_zupt(seg["t"], seg["acc"], gravity=g_est, accel_threshold=0.1)
            pred_h = pos[-1]
            
            err = np.abs(pred_h - seg["true_height"])
            n_steps = len(seg["acc"])
            sigma = analyzer.compute_theoretical_confidence(n_steps, seg["phone"])
            
            errors.append(err)
            sigmas.append(sigma)
        except Exception as e:
            pass
            
    print(f"Successfully processed {len(errors)} train segments. Fitting Conformal...")
    analyzer.fit_conformal(errors, sigmas, alpha=0.1)
    
    params = {
        "calibrated_multiplier": analyzer.calibrated_multiplier,
        "calibrated_margin": analyzer.calibrated_margin
    }
    with open("conformal_params.json", "w") as f:
        json.dump(params, f)
        
    print(f"Conformal prediction saved to conformal_params.json!")
    print(f"Multiplier: {analyzer.calibrated_multiplier:.3f}, Margin: {analyzer.calibrated_margin:.3f}")

if __name__ == "__main__":
    train_test_pipeline()
