import json
import os
import pandas as pd

def test_elevator_segments_bounds():
    base_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\ADVIO"
    metadata_path = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist\metadata\elevator_segments.json"
    
    assert os.path.exists(metadata_path), "Metadata JSON does not exist"
    
    with open(metadata_path, 'r') as f:
        segments = json.load(f)
        
    for ds, runs in segments.items():
        accel_path = os.path.join(base_path, ds, "iphone", "accelerometer.csv")
        assert os.path.exists(accel_path), f"Accelerometer data missing for {ds}"
        
        df = pd.read_csv(accel_path, header=None)
        t_acc = df[0].values
        t_min, t_max = t_acc[0], t_acc[-1]
        
        for run in runs:
            assert run['start_time'] >= t_min, f"Start time {run['start_time']} before dataset start {t_min} in {ds}"
            assert run['end_time'] <= t_max, f"End time {run['end_time']} after dataset end {t_max} in {ds}"
            assert run['end_time'] > run['start_time'], f"End time before start time in {ds}"
            assert run['height_diff'] > 0, f"Height diff is 0 or negative in {ds}"
            assert run['direction'] in ["up", "down"], f"Invalid direction {run['direction']} in {ds}"

if __name__ == "__main__":
    test_elevator_segments_bounds()
    print("All tests passed!")
