#!/usr/bin/env python3
"""
Elevator Height Estimation — Inference Script

Usage:
    python run_inference.py --input <accelerometer_csv> [--output results.json] [--fs 100]

Input CSV format:
    Required columns: acc_x, acc_y, acc_z
    Optional column:  time (seconds). If absent, timestamps computed from --fs.

Output:
    JSON array of detected rides with height estimates and confidence intervals.
"""

import argparse
import json
import sys
import os
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import ElevatorHeightPipeline


def load_accelerometer_csv(path, fs=100):
    """Load accelerometer data from CSV file.
    
    Handles standard CSV with headers, headerless 4-column CSV (time_ms, x, y, z),
    and various column name conventions.
    """
    df = pd.read_csv(path)
    
    # Detect headerless CSV: if all column names look numeric, re-read with names
    all_numeric = all(_is_numeric(str(c)) for c in df.columns)
    if all_numeric and len(df.columns) == 4:
        df = pd.read_csv(path, names=["time_ms", "x", "y", "z"])
    
    # Flexible column name matching
    col_map = {}
    for col in df.columns:
        cl = col.lower().strip()
        if cl in ('acc_x', 'accelerometerx', 'ax', 'x'):
            col_map['acc_x'] = col
        elif cl in ('acc_y', 'accelerometery', 'ay', 'y'):
            col_map['acc_y'] = col
        elif cl in ('acc_z', 'accelerometerz', 'az', 'z'):
            col_map['acc_z'] = col
        elif cl in ('time', 'time_sec', 'timestamp', 't', 'time_ms'):
            col_map['time'] = col
            col_map['time_key'] = cl
    
    if 'acc_x' not in col_map or 'acc_y' not in col_map or 'acc_z' not in col_map:
        raise ValueError(
            f"CSV must have acc_x, acc_y, acc_z columns. Found: {list(df.columns)}"
        )
    
    acc_x = df[col_map['acc_x']].values.astype(float)
    acc_y = df[col_map['acc_y']].values.astype(float)
    acc_z = df[col_map['acc_z']].values.astype(float)
    
    if 'time' in col_map:
        t = df[col_map['time']].values.astype(float)
        # Convert ms to sec if needed
        if col_map.get('time_key') == 'time_ms' or (len(t) > 1 and np.median(np.diff(t)) > 1):
            t = t / 1000.0
        if len(t) > 1:
            fs = 1.0 / np.median(np.diff(t))
    
    return acc_x, acc_y, acc_z, fs


def _is_numeric(s):
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Elevator Height Estimation Pipeline"
    )
    parser.add_argument("--input", "-i", required=True,
                        help="Path to accelerometer CSV file")
    parser.add_argument("--output", "-o", default=None,
                        help="Output JSON file path (default: stdout)")
    parser.add_argument("--fs", type=float, default=100,
                        help="Sampling frequency in Hz (default: 100)")
    parser.add_argument("--model-dir", default="model/",
                        help="Directory containing pipeline parameters")
    parser.add_argument("--segments", "-s", default=None,
                        help="Path to JSON file with pre-computed segments "
                             "(list of {start_time, end_time}). "
                             "Skips detection, runs quality+estimation only.")
    parser.add_argument("--accepted-only", action="store_true",
                        help="Output only accepted (non-rejected) rides")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Print progress information")
    
    args = parser.parse_args()
    
    # Load pipeline
    if args.verbose:
        print(f"Loading pipeline from {args.model_dir}...")
    pipeline = ElevatorHeightPipeline.load(args.model_dir)
    
    # Load data
    if args.verbose:
        print(f"Loading accelerometer data from {args.input}...")
    acc_x, acc_y, acc_z, fs = load_accelerometer_csv(args.input, args.fs)
    
    if args.verbose:
        duration = len(acc_x) / fs
        print(f"  {len(acc_x)} samples, {fs:.0f} Hz, {duration:.1f}s duration")
    
    # Load user-provided segments if specified
    segments = None
    if args.segments:
        if args.verbose:
            print(f"Loading segments from {args.segments}...")
        with open(args.segments, "r") as f:
            segments = json.load(f)
        if args.verbose:
            print(f"  {len(segments)} segments provided (skipping detection)")
    
    # Process
    if args.verbose:
        print("Running pipeline...")
    
    if segments:
        results = pipeline.process_segments(acc_x, acc_y, acc_z, segments, fs=fs)
        if args.accepted_only:
            results = [r for r in results if r['accepted']]
    elif args.accepted_only:
        results = pipeline.process_accepted(acc_x, acc_y, acc_z, fs=fs)
    else:
        results = pipeline.process(acc_x, acc_y, acc_z, fs=fs)
    
    # Clean output (remove non-serializable items)
    output = []
    skip_keys = {'quality_features', 'pos_curve'}
    for r in results:
        out = {k: v for k, v in r.items() if k not in skip_keys}
        # Add simplified quality features
        if 'quality_features' in r:
            out['quality_features'] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in r['quality_features'].items()
            }
        output.append(out)
    
    # Print summary
    if args.verbose:
        n_total = len(output)
        n_accepted = sum(1 for r in output if r['accepted'])
        print(f"\nDetected {n_total} rides, {n_accepted} accepted:")
        for i, r in enumerate(output):
            status = "✓" if r['accepted'] else "✗"
            ci = f" ± {r['confidence_interval_90']:.2f}m" if r['confidence_interval_90'] else ""
            print(f"  {status} Ride {i+1}: "
                  f"t=[{r['start_time']:.1f}, {r['end_time']:.1f}]s  "
                  f"h={r['height_estimate']:+.2f}m{ci}  "
                  f"({r['method']})")
            if not r['accepted']:
                print(f"    Reason: {r['reject_reason']}")
    
    # Output
    json_str = json.dumps(output, indent=2)
    if args.output:
        with open(args.output, "w") as f:
            f.write(json_str)
        if args.verbose:
            print(f"\nResults saved to {args.output}")
    else:
        print(json_str)


if __name__ == "__main__":
    main()
