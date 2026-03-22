import pandas as pd
import numpy as np
import json
import os
import matplotlib.pyplot as plt

base_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\ADVIO"
datasets = ["advio-07", "advio-14", "advio-18"]

# advio-14 has stairs and an elevator.
# According to the table, 14 has both. We should look at acceleration patterns. Elevators are smooth, stairs are bumpy.
# However, let's just extract segments where absolute vertical velocity > 0.2 m/s.
# We will save to a dictionary and plot to verify them.

metadata_dir = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\metadata"
os.makedirs(metadata_dir, exist_ok=True)
json_path = os.path.join(metadata_dir, "elevator_segments.json")

results = {}

for ds in datasets:
    pose_path = os.path.join(base_path, ds, "ground-truth", "pose.csv")
    
    try:
        pose_df = pd.read_csv(pose_path, header=None)
        # Format: timestamp, x, y, z, qx, qy, qz, qw
        t = pose_df[0].values
        z = pose_df[3].values
        
        # Calculate vertical velocity (dz/dt)
        dt = np.diff(t)
        dz = np.diff(z)
        vz = np.zeros_like(t)
        vz[1:] = dz / dt
        
        # Smooth the velocity (moving average over 1 second, assuming 100Hz -> 100 frames)
        window = 100
        vz_smooth = np.convolve(vz, np.ones(window)/window, mode='same')
        
        # Identify continuous segments where abs(vz_smooth) > threshold
        threshold = 0.2 # m/s (elevators usually go 1-2 m/s)
        is_moving = np.abs(vz_smooth) > threshold
        
        # Find rising and falling edges
        edges = np.diff(is_moving.astype(int))
        starts = np.where(edges == 1)[0]
        ends = np.where(edges == -1)[0]
        
        # Handle edge cases where it starts/ends moving at the boundaries
        if is_moving[0]:
            starts = np.insert(starts, 0, 0)
        if is_moving[-1]:
            ends = np.append(ends, len(t)-1)
            
        segments = []
        for s, e in zip(starts, ends):
            # Duration check (at least 3 seconds)
            duration = t[e] - t[s]
            if duration > 3.0:
                # Add 1 second padding to capture acceleration/deceleration fully
                start_t = max(0, t[s] - 1.0)
                end_t = min(t[-1], t[e] + 1.0)
                
                # Check height difference to ensure it's a real floor change (> 2 meters)
                height_diff = abs(z[e] - z[s])
                if height_diff > 2.0:
                    segments.append({
                        "start_time": float(start_t),
                        "end_time": float(end_t),
                        "duration": float(end_t - start_t),
                        "height_diff": float(height_diff),
                        "direction": "up" if z[e] > z[s] else "down"
                    })
                    
        results[ds] = segments
        
        # Plot to verify
        plt.figure(figsize=(10, 4))
        plt.plot(t, z, label="Z Position")
        for seg in segments:
            plt.axvspan(seg['start_time'], seg['end_time'], color='red', alpha=0.3, label="Detected Elevator")
        plt.title(f"{ds} - Detected Segments")
        # Removing duplicate labels
        handles, labels = plt.gca().get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        plt.legend(by_label.values(), by_label.keys())
        plt.savefig(os.path.join(base_path, f"{ds}_segments.png"))
        plt.close()
        
    except Exception as e:
        print(f"Error {ds}: {e}")

with open(json_path, 'w') as f:
    json.dump(results, f, indent=4)

print(f"Segments extracted and saved to {json_path}")
for ds, segs in results.items():
    print(f"{ds}: {len(segs)} segments")
    for idx, s in enumerate(segs):
        print(f"  [{idx}] {s['start_time']:.2f}s to {s['end_time']:.2f}s ({s['direction']}, {s['height_diff']:.2f}m)")
