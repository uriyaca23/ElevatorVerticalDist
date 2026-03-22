import numpy as np

def estimate_height_kalman(t, az, gravity=None, accel_threshold=0.05):
    """
    1D Kalman Filter.
    Measures v=0 outside the active motion window.
    """
    if gravity is not None:
        az = az - gravity
        
    dt_arr = np.diff(t)
    dt_arr = np.insert(dt_arr, 0, 0)
    n = len(t)
    
    window_size = 50
    az_smooth = np.convolve(np.abs(az), np.ones(window_size)/window_size, mode='same')
    active_indices = np.where(az_smooth > accel_threshold)[0]
    
    is_stationary = np.ones(n, dtype=bool)
    if len(active_indices) > 0:
        margin = int(1.0 / np.mean(dt_arr[1:])) if np.mean(dt_arr[1:]) > 0 else 100
        start_idx = max(0, active_indices[0] - margin)
        end_idx = min(n - 1, active_indices[-1] + margin)
        is_stationary[start_idx:end_idx+1] = False
        
    X = np.zeros(3) # [p, v, b]
    P = np.eye(3) * 0.1
    pos_history = np.zeros(n)
    
    Q = np.diag([1e-6, 1e-4, 1e-6])
    R = 1e-6
    H = np.array([[0, 1, 0]])
    
    for i in range(1, n):
        dt = dt_arr[i]
        a = az[i]
        
        F = np.array([
            [1, dt, -0.5*dt**2],
            [0, 1,  -dt],
            [0, 0,   1]
        ])
        
        X_pred = F @ X + np.array([0.5*dt**2 * a, dt * a, 0])
        P_pred = F @ P @ F.T + Q
        
        if is_stationary[i]:
            z = 0
            y = z - (H @ X_pred)[0]
            S = H @ P_pred @ H.T + R
            K = P_pred @ H.T @ np.linalg.inv(S)
            X = X_pred + K.flatten() * y
            P = (np.eye(3) - K @ H) @ P_pred
        else:
            X = X_pred
            P = P_pred
            
        pos_history[i] = X[0]
        
    return pos_history
