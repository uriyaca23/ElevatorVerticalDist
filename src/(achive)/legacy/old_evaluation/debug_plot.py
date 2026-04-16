import os
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append(r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist")
from src.algorithms.algo2_zupt import estimate_height_zupt

base_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\ADVIO"
accel_path = os.path.join(base_path, "advio-07", "iphone", "accelerometer.csv")
        
accel_df = pd.read_csv(accel_path, header=None)
t_acc = accel_df[0].values
g = 9.80665
ax = accel_df[1].values * g
ay = accel_df[2].values * g
az = accel_df[3].values * g
a_mag = np.sqrt(ax**2 + ay**2 + az**2)

# Seg 0
s_t = 17.0
e_t = 26.0
mask = (t_acc >= (s_t - 2.0)) & (t_acc <= (e_t + 2.0))
t_sub = t_acc[mask]
a_sub = a_mag[mask]
raw_z = az[mask]

# Gravity estimate
mask_rest = (t_sub < s_t)
gravity_est = np.mean(a_sub[mask_rest])
a_clean = a_sub - gravity_est

# Run zupt but return velocity as well to debug
a = a_clean
dt = np.diff(t_sub)
dt = np.insert(dt, 0, 0)
window_size = 50
az_smooth = np.convolve(np.abs(a), np.ones(window_size)/window_size, mode='same')
accel_threshold = 0.2
active_indices = np.where(az_smooth > accel_threshold)[0]
start_idx = active_indices[0] if len(active_indices) > 0 else 0
end_idx = active_indices[-1] if len(active_indices) > 0 else len(t_sub)-1

margin = int(1.0 / np.mean(dt[1:])) if np.mean(dt[1:]) > 0 else 100
start_idx = max(0, start_idx - margin)
end_idx = min(len(t_sub) - 1, end_idx + margin)

vel = np.zeros_like(t_sub)
pos = np.zeros_like(t_sub)

for i in range(start_idx + 1, end_idx + 1):
    vel[i] = vel[i-1] + a[i] * dt[i]

drift = vel[end_idx]
num_steps = end_idx - start_idx
if num_steps > 0:
    drift_rate = drift / num_steps
    for i in range(start_idx + 1, end_idx + 1):
        vel[i] = vel[i] - drift_rate * (i - start_idx)

for i in range(1, len(t_sub)):
    pos[i] = pos[i-1] + vel[i] * dt[i]

plt.figure(figsize=(12, 10))

plt.subplot(3, 1, 1)
plt.plot(t_sub, a, label="a_clean (Magnitude - g)")
plt.plot(t_sub, raw_z - np.mean(raw_z[mask_rest]), label="Raw Z - g", alpha=0.5)
plt.axvline(t_sub[start_idx], color='r')
plt.axvline(t_sub[end_idx], color='r')
plt.legend()
plt.title("Acceleration")

plt.subplot(3, 1, 2)
plt.plot(t_sub, vel, label="Corrected Velocity")
plt.legend()
plt.title("Velocity")

plt.subplot(3, 1, 3)
plt.plot(t_sub, pos, label="Position")
plt.legend()
plt.title(f"Position (Final: {pos[-1]:.2f}m)")

plt.tight_layout()
plt.savefig(os.path.join(base_path, "debug_zupt.png"))
print(f"Debug plot saved. Final pos: {pos[-1]:.2f}m. Window: {t_sub[start_idx]:.2f} to {t_sub[end_idx]:.2f}")
