import os
import json
import random
import numpy as np
import pandas as pd
import shutil
import sys
from pathlib import Path

# Add the src folder to the python path to import noise_db
sys.path.append(str(Path(__file__).resolve().parent.parent))

try:
    from algorithms.noise_db import SENSOR_CHIPS
except ImportError:
    print("WARNING: Could not import noise_db. Using fallback generic noise.")
    SENSOR_CHIPS = None

def get_phone_noise_sigma(phone_model, fs=100.0):
    if SENSOR_CHIPS and phone_model in SENSOR_CHIPS:
        params = SENSOR_CHIPS[phone_model].to_noise_params(sampling_rate_hz=fs)
        return params.accel_noise_sigma
    return 0.05  # fallback 50 mg

def generate_elevator_acceleration(H, A, V_max, fs=100.0):
    """
    Generates an ideal acceleration profile for an elevator traveling H meters.
    """
    dt = 1.0 / fs
    if H < (V_max**2) / A:
        # Triangular profile
        V_max = np.sqrt(A * H)
        t_acc = V_max / A
        t_const = 0.0
    else:
        # Trapezoidal profile
        t_acc = V_max / A
        t_const = (H - V_max * t_acc) / V_max

    # Pad with stationary time before and after
    t_stat_start = random.uniform(2.0, 5.0)
    t_stat_end = random.uniform(2.0, 5.0)
    
    t_total = t_stat_start + t_acc + t_const + t_acc + t_stat_end
    t = np.arange(0, t_total, dt)
    a = np.zeros_like(t)

    # Acceleration phase
    idx_acc_start = int(t_stat_start / dt)
    idx_acc_end = idx_acc_start + int(t_acc / dt)
    a[idx_acc_start:idx_acc_end] = A

    # Deceleration phase
    idx_dec_start = idx_acc_end + int(t_const / dt)
    idx_dec_end = idx_dec_start + int(t_acc / dt)
    a[idx_dec_start:idx_dec_end] = -A

    return t, a

def generate_sample(output_dir, sample_name, include_gt=True, phone_models=None):
    if phone_models is None:
        phone_models = ["generic_premium", "generic_midrange", "generic_budget"]
    
    phone_model = random.choice(phone_models)
    base_sigma = get_phone_noise_sigma(phone_model)

    H = random.uniform(3.0, 100.0)
    A = random.uniform(0.5, 1.5)
    V_max = random.uniform(1.0, 5.0)
    direction = random.choice([1, -1])  # up or down
    
    t, a_ideal = generate_elevator_acceleration(H, A, V_max)
    a_ideal *= direction
    
    # Introduce anomalies
    anomaly_type = random.choices(
        ['clean', 'shaking', 'impact', 'long_stationary'],
        weights=[0.7, 0.1, 0.1, 0.1]
    )[0]
    
    n_samples = len(t)
    noise = np.random.normal(0, base_sigma, n_samples)
    
    if anomaly_type == 'shaking':
        # Increase noise by 5x in the middle of motion
        shake_idx = n_samples // 2
        shake_len = int(n_samples * 0.1)
        noise[shake_idx:shake_idx+shake_len] += np.random.normal(0, base_sigma * 5, shake_len)
    elif anomaly_type == 'impact':
        # One high spike (like bumping the phone)
        impact_idx = random.randint(int(n_samples*0.2), int(n_samples*0.8))
        a_ideal[impact_idx] += np.random.choice([-10.0, 10.0])
    elif anomaly_type == 'long_stationary':
        # Extend stationary to > 60 seconds to trigger theoretical rejection
        t_extra = np.arange(0, 70.0, 0.01)
        a_extra = np.random.normal(0, base_sigma, len(t_extra))
        t = np.concatenate([t, t[-1] + t_extra[1:]])
        a_ideal = np.concatenate([a_ideal, np.zeros(len(t_extra)-1)])
        noise = np.concatenate([noise, a_extra[1:]])
        n_samples = len(t)

    az = 9.81 + a_ideal + noise
    ax = np.random.normal(0, base_sigma, n_samples)
    ay = np.random.normal(0, base_sigma, n_samples)

    sample_dir = os.path.join(output_dir, sample_name)
    os.makedirs(sample_dir, exist_ok=True)

    df = pd.DataFrame({"time": t, "ax": ax, "ay": ay, "az": az})
    df.to_csv(os.path.join(sample_dir, "accel.csv"), index=False)

    metadata = {
        "phone_model": phone_model,
        "anomaly": anomaly_type
    }
    if include_gt:
        metadata["gt_height_meters"] = H * direction
    
    with open(os.path.join(sample_dir, "metadata.json"), "w") as f:
        json.dump(metadata, f, indent=4)
        
    return metadata

def create_dataset(base_dir, n_train=100, n_test=50):
    if os.path.exists(base_dir):
        shutil.rmtree(base_dir, ignore_errors=True)
        import time; time.sleep(0.5)
        if os.path.exists(base_dir):
            shutil.rmtree(base_dir, ignore_errors=True)
        
    train_dir = os.path.join(base_dir, "train")
    test_dir = os.path.join(base_dir, "test")
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    print(f"Generating {n_train} train samples...")
    phone_models = ["pixel_7", "iphone_14", "galaxy_s23", "generic_budget", "bmi160", "lsm6dso", "icm42688"]
    
    for i in range(n_train):
        generate_sample(train_dir, f"sample_{i:04d}", include_gt=True, phone_models=phone_models)
        
    print(f"Generating {n_test} test samples...")
    for i in range(n_test):
        generate_sample(test_dir, f"sample_{i:04d}", include_gt=False, phone_models=phone_models)
        
    print(f"Dataset successfully created at {base_dir}")

if __name__ == "__main__":
    project_root = Path(__file__).resolve().parent.parent.parent
    dataset_dir = os.path.join(project_root, "example", "work_dataset")
    create_dataset(dataset_dir)
