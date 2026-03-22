import pandas as pd
import numpy as np
import os

base_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\ADVIO"
datasets = ["advio-07", "advio-14", "advio-18"]

for ds in datasets:
    fix_path = os.path.join(base_path, ds, "ground-truth", "fixpoints.csv")
    accel_path = os.path.join(base_path, ds, "iphone", "accelerometer.csv")
    
    try:
        fix_df = pd.read_csv(fix_path, header=None)
        accel_df = pd.read_csv(accel_path, header=None)
        
        t_acc = accel_df[0].values
        z_acc = accel_df[3].values
        
        t_fix = fix_df[0].values
        # Z height is either column 3 or we use floor if available? Let's use column 3
        z_fix = fix_df[3].values
        
        print(f"\n--- {ds} ---")
        for i in range(len(t_fix) - 1):
            t_start = t_fix[i]
            t_end = t_fix[i+1]
            z_start = float(z_fix[i]) if str(z_fix[i]).lower() != 'nan' else np.nan
            z_end = float(z_fix[i+1]) if str(z_fix[i+1]).lower() != 'nan' else np.nan
            
            # If Z changes significantly
            if not np.isnan(z_start) and not np.isnan(z_end):
                diff = abs(z_end - z_start)
                if diff > 1.0: # More than 1 meter jump = floor change
                    # Calculate variance in accel
                    mask = (t_acc >= t_start) & (t_acc <= t_end)
                    if np.any(mask):
                        var = np.var(z_acc[mask])
                        print(f"Interval {t_start:.2f} to {t_end:.2f} (dt={t_end-t_start:.2f}s): Height diff {diff:.2f}m. Accel Var: {var:.4f}")
                    else:
                        print(f"Interval {t_start:.2f} to {t_end:.2f} (dt={t_end-t_start:.2f}s): Height diff {diff:.2f}m. No Accel data.")
                        
    except Exception as e:
        print(f"Error {ds}: {e}")
