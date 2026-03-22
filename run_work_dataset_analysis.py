import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))
from algorithms.algo2_zupt import estimate_height_zupt
from algorithms.zupt_confidence import ZuptConfidenceAnalyzer
from dataset.synthetic_work_dataset import create_dataset

def load_sample(sample_dir):
    df = pd.read_csv(os.path.join(sample_dir, "accel.csv"))
    with open(os.path.join(sample_dir, "metadata.json"), "r") as f:
        metadata = json.load(f)
    return df['time'].values, df['az'].values, metadata

def get_active_window(t, az, accel_threshold=0.05):
    dt = np.diff(t)
    dt = np.insert(dt, 0, 0)
    window_size = 50
    az_smooth = np.convolve(np.abs(az - 9.81), np.ones(window_size)/window_size, mode='same')
    active_indices = np.where(az_smooth > accel_threshold)[0]
    if len(active_indices) == 0:
        return 0, 0
    start_idx = active_indices[0]
    end_idx = active_indices[-1]
    margin = int(1.0 / np.mean(dt[1:])) if np.mean(dt[1:]) > 0 else 100
    return max(0, start_idx - margin), min(len(t) - 1, end_idx + margin)

def evaluate_dataset(dataset_dir, analyzer, is_train=False):
    samples = sorted(os.listdir(dataset_dir))
    results = []
    
    for sample in samples:
        sample_path = os.path.join(dataset_dir, sample)
        if not os.path.isdir(sample_path):
            continue
            
        t, az, meta = load_sample(sample_path)
        phone_model = meta['phone_model']
        
        # ZUPT algorithm natively estimates height.
        # We need to subtract gravity, which algo2_zupt.estimate_height_zupt can do if gravity=9.81
        # But wait, our az is ~9.81 + a.
        # Let's run ZUPT:
        pos = estimate_height_zupt(t, az, gravity=9.81, accel_threshold=0.05)
        height_est = pos[-1]
        
        # Rejection checking
        start_idx, end_idx = get_active_window(t, az, 0.05)
        num_steps = end_idx - start_idx
        
        should_reject, reason = analyzer.evaluate_rejection(az, start_idx, end_idx, phone_model)
        
        # Confidence interval
        margin = analyzer.get_confidence_interval(num_steps, phone_model)
        
        res = {
            'sample_id': sample,
            'height_est': height_est,
            'ci_90_margin': margin,
            'rejected': should_reject,
            'rejection_reason': reason,
            'phone_model': phone_model,
            'num_steps': num_steps
        }
        
        if is_train and 'gt_height_meters' in meta:
            res['gt_height'] = meta['gt_height_meters']
            res['error'] = height_est - meta['gt_height_meters']
            
        results.append(res)
        
    return results

def train(dataset_root, analyzer):
    train_dir = os.path.join(dataset_root, "train")
    if not os.path.exists(train_dir):
        print("Train directory not found. Please generate the dataset first.")
        return
        
    print(f"Loading and processing train data from {train_dir}...")
    train_results = evaluate_dataset(train_dir, analyzer, is_train=True)
    
    # Filter out rejected samples for conformal tuning
    valid_train = [r for r in train_results if not r['rejected']]
    rejected_train = [r for r in train_results if r['rejected']]
    
    errors = [np.abs(r['error']) for r in valid_train]
    theoretical_sigmas = [analyzer.compute_theoretical_confidence(r['num_steps'], r['phone_model']) for r in valid_train]
    
    print(f"Total train samples: {len(train_results)}")
    print(f"Rejected train samples: {len(rejected_train)}")
    print(f"Fitting conformal predictor on {len(valid_train)} valid samples...")
    
    analyzer.fit_conformal(errors, theoretical_sigmas, alpha=0.1)
    
    # Save the parameters
    params = {
        "calibrated_multiplier": analyzer.calibrated_multiplier,
        "calibrated_margin": analyzer.calibrated_margin
    }
    with open("conformal_params.json", "w") as f:
        json.dump(params, f)
        
    print(f"Conformal prediction calibrated:")
    print(f"  Sigma Multiplier: {analyzer.calibrated_multiplier:.3f}")
    print(f"  Constant Additive Margin: {analyzer.calibrated_margin:.3f} m")
    print(f"Parameters saved to conformal_params.json.")
    
    # Coverage check on train
    coverage = sum(1 for e, s in zip(errors, theoretical_sigmas) if e <= (analyzer.calibrated_multiplier * s + analyzer.calibrated_margin))
    print(f"Empricial coverage on valid train set: {coverage/len(errors)*100:.1f}%")

def predict(dataset_root, analyzer):
    test_dir = os.path.join(dataset_root, "test")
    if not os.path.exists(test_dir):
        print("Test directory not found. Please generate the dataset first.")
        return
        
    if os.path.exists("conformal_params.json"):
        with open("conformal_params.json", "r") as f:
            params = json.load(f)
            analyzer.calibrated_multiplier = params["calibrated_multiplier"]
            analyzer.calibrated_margin = params["calibrated_margin"]
        print("Loaded conformal parameters from conformal_params.json")
    else:
        print("WARNING: conformal_params.json not found. Using default theoretical bounds.")
        
    print(f"Loading and predicting on test data from {test_dir}...")
    test_results = evaluate_dataset(test_dir, analyzer, is_train=False)
    
    valid_test = [r for r in test_results if not r['rejected']]
    rejected_test = [r for r in test_results if r['rejected']]
    
    print(f"\n--- Prediction Report ---")
    print(f"Total Test Samples: {len(test_results)}")
    print(f"Rejected: {len(rejected_test)} (Check log for reasons)")
    print(f"Accepted: {len(valid_test)}")
    
    print("\nSample Output (Accepted):")
    for r in valid_test[:5]:
        print(f"  [{r['sample_id']}] Est: {r['height_est']:.2f}m | 90% CI: ±{r['ci_90_margin']:.2f}m | Phone: {r['phone_model']}")
        
    print("\nSample Output (Rejected):")
    for r in rejected_test[:5]:
        print(f"  [{r['sample_id']}] Est: {r['height_est']:.2f}m | REJECTED: {r['rejection_reason']}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Work Dataset ZUPT Analysis Tool")
    parser.add_argument("command", choices=["generate", "train", "predict"], help="Command to run")
    parser.add_argument("--dataset_dir", type=str, default="example/work_dataset", help="Path to the synthetic work dataset")
    args = parser.parse_args()
    
    root_dir = os.path.abspath(args.dataset_dir)
    analyzer = ZuptConfidenceAnalyzer(dt=0.01) # Assuming 100Hz
    
    if args.command == "generate":
        print(f"Generating synthetic dataset in {root_dir}...")
        create_dataset(root_dir, n_train=150, n_test=50)
    elif args.command == "train":
        train(root_dir, analyzer)
    elif args.command == "predict":
        predict(root_dir, analyzer)
