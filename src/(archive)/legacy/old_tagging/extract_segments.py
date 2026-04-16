import pandas as pd
import matplotlib.pyplot as plt
import os

base_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\ADVIO"
datasets = ["advio-07", "advio-14", "advio-18"]

for ds in datasets:
    pose_path = os.path.join(base_path, ds, "ground-truth", "pose.csv")
    accel_path = os.path.join(base_path, ds, "iphone", "accelerometer.csv")
    fix_path = os.path.join(base_path, ds, "ground-truth", "fixpoints.csv")
    
    # Read poses
    # format: timestamp, x, y, z, qx, qy, qz, qw (usually)
    try:
        pose_df = pd.read_csv(pose_path, header=None)
        timestamp_pose = pose_df[0]
        z_pose = pose_df[3]
    except Exception as e:
        print(f"Error reading {pose_path}: {e}")
        continue
        
    try:
        accel_df = pd.read_csv(accel_path, header=None)
        timestamp_accel = accel_df[0]
        z_accel = accel_df[3]
    except Exception as e:
        print(f"Error reading {accel_path}: {e}")
        continue
        
    plt.figure(figsize=(15, 8))
    
    plt.subplot(2, 1, 1)
    plt.plot(timestamp_pose, z_pose, label='Z Position (m)')
    plt.title(f"{ds} - Vertical Position")
    plt.ylabel("Z (m)")
    plt.grid(True)
    
    plt.subplot(2, 1, 2)
    plt.plot(timestamp_accel, z_accel, label='Z Acceleration (g)', alpha=0.7)
    plt.title(f"{ds} - Z Acceleration")
    plt.xlabel("Time (s)")
    plt.ylabel("Acceleration (g)")
    plt.grid(True)
    
    out_path = os.path.join(base_path, f"{ds}_vertical_analysis.png")
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()

print("Plots generated successfully in ADVIO base directory.")
