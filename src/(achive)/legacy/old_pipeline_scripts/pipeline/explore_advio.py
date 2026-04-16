import os
import pandas as pd

def main():
    print("--- ADVIO Dataset Structure exploration ---")
    
    advio_path = os.path.join("datasets", "ADVIO", "advio-07")
    pixel_acc = os.path.join(advio_path, "pixel", "accelerometer.csv")
    gt_path = os.path.join(advio_path, "ground-truth", "groundtruth.csv")
    if not os.path.exists(gt_path):
        # Maybe it's txt?
        gt_path = os.path.join(advio_path, "ground-truth", "frames.csv")
        
    print(f"Acceleromater: {pixel_acc}")
    if os.path.exists(pixel_acc):
        df_acc = pd.read_csv(pixel_acc, nrows=5)
        print("Columns:", df_acc.columns.tolist())
        
    gt_dir = os.path.join(advio_path, "ground-truth")
    print("\nFiles in Ground-Truth:")
    for f in os.listdir(gt_dir):
        print("  -", f)
        if f.endswith(".csv"):
            df_gt = pd.read_csv(os.path.join(gt_dir, f), nrows=3)
            print("    Cols:", df_gt.columns.tolist())
            
if __name__ == "__main__":
    main()
