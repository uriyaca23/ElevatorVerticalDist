import numpy as np
import os
import sys

sys.path.append(os.path.dirname(__file__))
from elevator_detection_algorithms import ElevatorDetector
from algorithms.algo2_zupt import estimate_height_zupt
from algorithms.zupt_confidence import ZuptConfidenceAnalyzer

import json

class FullElevatorPipeline:
    def __init__(self, conformal_params_path="conformal_params.json", fs=100.0):
        self.fs = fs
        self.detector = ElevatorDetector(fs=fs)
        self.analyzer = ZuptConfidenceAnalyzer(dt=1.0/fs)
        
        # Load conformal if exists
        self.has_conformal = False
        if os.path.exists(conformal_params_path):
            with open(conformal_params_path, "r") as f:
                params = json.load(f)
                self.analyzer.calibrated_multiplier = params["calibrated_multiplier"]
                self.analyzer.calibrated_margin = params["calibrated_margin"]
            self.has_conformal = True
            
    def process_accelerometer(self, time_sec, acc_z, phone_model="Unknown"):
        """
        Takes raw accelerometer data and outputs segments, heights, and conformal bounds
        """
        # Step 1: Detect Elevator Segments using best Algorithm (State Machine)
        # Optimized params from our testing: var_thresh=3.5, acc_thresh=0.35
        pred_indices = self.detector.detect_algorithm1_state_machine(
            time_sec, acc_z, var_thresh=3.5, acc_thresh=0.35
        )
        
        results = []
        for s_idx, e_idx in pred_indices:
            # We add a small margin to ZUPT
            margin = int(self.fs * 2.0)
            z_s = max(0, s_idx - margin)
            z_e = min(len(acc_z) - 1, e_idx + margin)
            
            t_seg = time_sec[z_s:z_e]
            a_seg = acc_z[z_s:z_e]
            
            # Step 2: Estimate Height
            try:
                # estimate_height_zupt expects t, az, gravity, accel_threshold
                # We can assume gravity is around mean or 9.81
                g_est = np.mean(a_seg)
                pos = estimate_height_zupt(t_seg, a_seg, gravity=g_est, accel_threshold=0.1)
                final_height = pos[-1]
                
                # Step 3: Conformal Prediction Bounds
                num_steps = len(a_seg)
                if self.has_conformal:
                    ci_90 = self.analyzer.get_confidence_interval(num_steps, phone_model)
                else:
                    ci_90 = self.analyzer.compute_theoretical_confidence(num_steps, phone_model) * 1.645 # 90% theoretical
                    
                results.append({
                    "start_time": t_seg[0],
                    "end_time": t_seg[-1],
                    "height": final_height,
                    "containment_90_margin": ci_90
                })
            except Exception as e:
                pass
                
        return results

def load_pipeline():
    """ 
    Return the loadable function for the user 
    """
    pipeline = FullElevatorPipeline()
    return pipeline.process_accelerometer
