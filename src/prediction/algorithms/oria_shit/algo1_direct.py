import numpy as np

def estimate_height_direct(t, az, gravity=None):
    """
    Direct numerical double integration of acceleration.
    Assumes az is in meters per second squared and gravity is already removed.
    If gravity is provided, we subtract it.
    If az is from ADVIO (g-units), we must convert it.
    """
    if gravity is not None:
        az = az - gravity
        
    # We assume 'az' input is strictly linear acceleration WITHOUT gravity in m/s^2 for the tests.
    
    dt = np.diff(t)
    # prepend 0 to match length
    dt = np.insert(dt, 0, 0)
    
    vel = np.cumsum(az * dt)
    pos = np.cumsum(vel * dt)
    
    return pos
