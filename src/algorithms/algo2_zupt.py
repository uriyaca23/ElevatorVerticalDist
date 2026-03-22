import numpy as np

def estimate_height_zupt(t, az, gravity=None, accel_threshold=0.05):
    """
    Elevator ZUPT:
    Finds the active motion window (from first acceleration peak to last deceleration peak).
    Forces velocity to 0 outside this window.
    Applies linear drift correction to velocity inside the window so that it ends at 0.
    """
    if gravity is not None:
        az = az - gravity
        
    dt = np.diff(t)
    dt = np.insert(dt, 0, 0)
    
    # Smooth a copy to find active window robustly
    window_size = 50
    az_smooth = np.convolve(np.abs(az), np.ones(window_size)/window_size, mode='same')
    
    # 1. Identify active motion window
    active_indices = np.where(az_smooth > accel_threshold)[0]
    
    if len(active_indices) == 0:
        return np.zeros_like(t)
        
    start_idx = active_indices[0]
    end_idx = active_indices[-1]
    
    # Add a small margin to start and end
    margin = int(1.0 / np.mean(dt[1:])) if np.mean(dt[1:]) > 0 else 100
    start_idx = max(0, start_idx - margin)
    end_idx = min(len(t) - 1, end_idx + margin)
    
    vel = np.zeros_like(t)
    pos = np.zeros_like(t)
    
    # 2. Integrate velocity only within window
    for i in range(start_idx + 1, end_idx + 1):
        vel[i] = vel[i-1] + az[i] * dt[i]
        
    # 3. Calculate drift and apply linear correction
    drift = vel[end_idx]
    num_steps = end_idx - start_idx
    if num_steps > 0:
        drift_rate = drift / num_steps
        for i in range(start_idx + 1, end_idx + 1):
            vel[i] = vel[i] - drift_rate * (i - start_idx)
            
    # velocity outside [start_idx, end_idx] remains 0
    
    # 4. Integrate to position
    for i in range(1, len(t)):
        pos[i] = pos[i-1] + vel[i] * dt[i]
        
    return pos
