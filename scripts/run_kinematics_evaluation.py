"""
Kinematics-Based Elevator Height Estimation — Full Evaluation.

Runs both Algorithm A (accel-only) and Algorithm B (accel+orientation) on
the Bar-Ilan and ADVIO datasets. Generates all figures for the research report.

Usage:
    python scripts/run_kinematics_evaluation.py
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
import json
import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from pathlib import Path
from tqdm import tqdm

from src.algorithms.kinematics_estimator import (
    estimate_height_accel_only,
    estimate_height_with_orientation,
)
from src.algorithms.scurve_model import generate_profile_vectorized

# ============================================================
# Configuration
# ============================================================

OUTPUT_DIR = Path("docs/figures_kinematics")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

RESULTS_DIR = Path("evaluation_output/kinematics")
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

BAR_ILAN_ACC = "datasets/bar_ilan_dataset/sensors_synced/ACC.csv"
BAR_ILAN_ORI = "datasets/bar_ilan_dataset/sensors_synced/ORI.csv"
BAR_ILAN_META = "datasets/bar_ilan_dataset/metadata.csv"
ADVIO_SEGMENTS = "metadata/elevator_segments.json"

BUILDING_TYPE = 'residential'
PRIOR_WEIGHT = 0.3
PRE_WINDOW_SEC = 5.0
POST_WINDOW_SEC = 5.0


# ============================================================
# Data Loading
# ============================================================

def load_bar_ilan_data():
    """Load Bar-Ilan accelerometer, orientation, and metadata."""
    # Accelerometer (headerless: time_ms, x, y, z)
    acc = pd.read_csv(BAR_ILAN_ACC, header=None,
                       names=['time_ms', 'ax', 'ay', 'az'])
    acc['time_sec'] = (acc['time_ms'] - acc['time_ms'].iloc[0]) / 1000.0

    # Orientation (headerless: time_ms, qw, qx, qy, qz)
    ori = pd.read_csv(BAR_ILAN_ORI, header=None,
                       names=['time_ms', 'qw', 'qx', 'qy', 'qz'])
    ori['time_sec'] = (ori['time_ms'] - acc['time_ms'].iloc[0]) / 1000.0

    # Metadata
    meta = pd.read_csv(BAR_ILAN_META)

    return acc, ori, meta


def load_advio_data(seq_name):
    """Load ADVIO accelerometer data for a specific sequence."""
    seq_dir = f"datasets/ADVIO/{seq_name}"
    acc_file = os.path.join(seq_dir, "accelerometer.csv")
    if not os.path.exists(acc_file):
        # Try alternate format
        for f in os.listdir(seq_dir):
            if 'acc' in f.lower() and f.endswith('.csv'):
                acc_file = os.path.join(seq_dir, f)
                break

    if not os.path.exists(acc_file):
        return None

    acc = pd.read_csv(acc_file)
    return acc


def get_bar_ilan_segments(meta):
    """Extract ground-truth elevator segments from metadata."""
    segments = []
    for seg_id in sorted(meta[meta['in_elevator']]['elevator_segment_id'].unique()):
        if seg_id < 0:
            continue
        seg_rows = meta[meta['elevator_segment_id'] == seg_id]
        start_t = seg_rows['time_sec'].min()
        end_t = seg_rows['time_sec'].max()
        h_start = seg_rows['height_smooth'].iloc[0]
        h_end = seg_rows['height_smooth'].iloc[-1]
        true_height = h_end - h_start
        phone = seg_rows['phone_position'].iloc[0]
        segments.append({
            'seg_id': int(seg_id),
            'start_time': float(start_t),
            'end_time': float(end_t),
            'true_height': float(true_height),
            'phone_position': phone,
        })
    return segments


# ============================================================
# Main Evaluation
# ============================================================

def run_bar_ilan_evaluation():
    """Run evaluation on Bar-Ilan dataset."""
    print("=" * 70)
    print("BAR-ILAN DATASET EVALUATION")
    print("=" * 70)

    acc, ori, meta = load_bar_ilan_data()
    segments = get_bar_ilan_segments(meta)
    print(f"Loaded {len(segments)} elevator segments")

    acc_t = acc['time_sec'].values
    acc_x = acc['ax'].values
    acc_y = acc['ay'].values
    acc_z = acc['az'].values

    ori_t = ori['time_sec'].values
    ori_qw = ori['qw'].values
    ori_qx = ori['qx'].values
    ori_qy = ori['qy'].values
    ori_qz = ori['qz'].values

    # Estimate sampling rate
    fs_est = 1.0 / np.median(np.diff(acc_t))
    pre_win = int(PRE_WINDOW_SEC * fs_est)
    post_win = int(POST_WINDOW_SEC * fs_est)

    results_a = []  # Algorithm A results
    results_b = []  # Algorithm B results

    pbar = tqdm(segments, desc="Bar-Ilan segments",
                unit="seg", dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for seg in pbar:
        seg_id = seg['seg_id']
        pbar.set_postfix_str(
            f"Seg {seg_id} ({seg['phone_position']}) {seg['true_height']:+.0f}m")

        # Find data indices
        mask = (acc_t >= seg['start_time']) & (acc_t <= seg['end_time'])
        ride_idx = np.where(mask)[0]
        if len(ride_idx) < 20:
            continue

        si, ei = ride_idx[0], ride_idx[-1]
        t_ride = acc_t[si:ei + 1]
        ax_ride = acc_x[si:ei + 1]
        ay_ride = acc_y[si:ei + 1]
        az_ride = acc_z[si:ei + 1]

        # Pre/post context
        pre_s = max(0, si - pre_win)
        pre_ax = acc_x[pre_s:si]
        pre_ay = acc_y[pre_s:si]
        pre_az = acc_z[pre_s:si]

        # === Algorithm A: Accel-only ===
        try:
            result_a = estimate_height_accel_only(
                t_ride, ax_ride, ay_ride, az_ride,
                pre_ax=pre_ax, pre_ay=pre_ay, pre_az=pre_az,
                building_type=BUILDING_TYPE, prior_weight=PRIOR_WEIGHT,
            )
            result_a['seg_id'] = seg_id
            result_a['true_height'] = seg['true_height']
            result_a['phone_position'] = seg['phone_position']
            result_a['error'] = abs(result_a['height'] - seg['true_height'])
            result_a['t_ride'] = t_ride - t_ride[0]  # Store zero-based time
            # Compute GT velocity from barometer height
            seg_meta = meta[(meta['time_sec'] >= seg['start_time']) &
                           (meta['time_sec'] <= seg['end_time'])]
            if len(seg_meta) > 5:
                gt_t = seg_meta['time_sec'].values - seg['start_time']
                gt_h = seg_meta['height_smooth'].values
                gt_h = gt_h - gt_h[0]  # Zero-based height
                gt_vel = np.gradient(gt_h, gt_t)
                result_a['gt_time'] = gt_t
                result_a['gt_height'] = gt_h
                result_a['gt_velocity'] = gt_vel
            results_a.append(result_a)
        except Exception as e:
            tqdm.write(f"  Seg {seg_id} A: FAILED - {e}")

        # === Algorithm B: Accel+Orientation ===
        try:
            result_b = estimate_height_with_orientation(
                t_ride, ax_ride, ay_ride, az_ride,
                ori_t, ori_qw, ori_qx, ori_qy, ori_qz,
                building_type=BUILDING_TYPE, prior_weight=PRIOR_WEIGHT,
            )
            result_b['seg_id'] = seg_id
            result_b['true_height'] = seg['true_height']
            result_b['phone_position'] = seg['phone_position']
            result_b['error'] = abs(result_b['height'] - seg['true_height'])
            results_b.append(result_b)
        except Exception as e:
            tqdm.write(f"  Seg {seg_id} B: FAILED - {e}")

    pbar.close()

    # Print summary table
    print(f"\nBar-Ilan Results Summary:")
    print(f"{'Seg':>4} {'True':>6} {'AlgA':>6} {'Err_A':>6} {'AlgB':>6} {'Err_B':>6}")
    print("-" * 42)
    for seg in segments:
        sid = seg['seg_id']
        ra = next((r for r in results_a if r.get('seg_id') == sid), None)
        rb = next((r for r in results_b if r.get('seg_id') == sid), None)
        a_str = f"{ra['height']:+.1f}" if ra else "FAIL"
        ae_str = f"{ra['error']:.2f}" if ra else "N/A"
        b_str = f"{rb['height']:+.1f}" if rb else "FAIL"
        be_str = f"{rb['error']:.2f}" if rb else "N/A"
        print(f"{sid:>4} {seg['true_height']:>+6.1f} {a_str:>6} {ae_str:>6} {b_str:>6} {be_str:>6}")

    return results_a, results_b, segments


def run_advio_evaluation():
    """Run evaluation on ADVIO dataset."""
    print("\n" + "=" * 70)
    print("ADVIO DATASET EVALUATION")
    print("=" * 70)

    with open(ADVIO_SEGMENTS) as f:
        advio_segs = json.load(f)

    results_a = []

    # Count total segments for progress bar
    total_segs = sum(len(segs) for segs in advio_segs.values())
    pbar = tqdm(total=total_segs, desc="ADVIO segments",
                unit="seg", dynamic_ncols=True,
                bar_format='{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}]')

    for seq_name, segments in advio_segs.items():
        seq_dir = f"datasets/ADVIO/{seq_name}"

        # ADVIO structure: seq_dir/iphone/accelerometer.csv
        acc_file = os.path.join(seq_dir, "iphone", "accelerometer.csv")
        if not os.path.exists(acc_file):
            # Fallback: check directly in seq_dir
            for sub in ['iphone', 'pixel', 'tango', '.']:
                candidate = os.path.join(seq_dir, sub, "accelerometer.csv")
                if os.path.exists(candidate):
                    acc_file = candidate
                    break
            else:
                pbar.update(len(segments))
                continue

        try:
            # ADVIO accelerometer CSV is headerless:
            # columns: time_sec, ax, ay, az
            acc_data = pd.read_csv(acc_file, header=None,
                                    names=['time_sec', 'ax', 'ay', 'az'])
            acc_t = acc_data['time_sec'].values
            acc_x = acc_data['ax'].values
            acc_y = acc_data['ay'].values
            acc_z = acc_data['az'].values
        except Exception as e:
            pbar.update(len(segments))
            continue

        fs_est = 1.0 / max(np.median(np.diff(acc_t)), 0.001)
        pre_win = int(PRE_WINDOW_SEC * fs_est)

        for seg in segments:
            pbar.set_postfix_str(f"{seq_name}")
            start_time = seg['start_time']
            end_time = seg['end_time']
            true_height = seg['height_diff']
            direction_str = seg.get('direction', 'up')

            mask = (acc_t >= start_time) & (acc_t <= end_time)
            ride_idx = np.where(mask)[0]
            if len(ride_idx) < 20:
                pbar.update(1)
                continue

            si, ei = ride_idx[0], ride_idx[-1]
            t_ride = acc_t[si:ei + 1]
            ax_ride = acc_x[si:ei + 1]
            ay_ride = acc_y[si:ei + 1]
            az_ride = acc_z[si:ei + 1]

            # Pre-context
            pre_s = max(0, si - pre_win)
            pre_ax = acc_x[pre_s:si]
            pre_ay = acc_y[pre_s:si]
            pre_az = acc_z[pre_s:si]

            try:
                result = estimate_height_accel_only(
                    t_ride, ax_ride, ay_ride, az_ride,
                    pre_ax=pre_ax, pre_ay=pre_ay, pre_az=pre_az,
                    building_type='commercial', prior_weight=PRIOR_WEIGHT,
                )
                # ADVIO segments are all upward
                if direction_str == 'up':
                    result['true_height'] = abs(true_height)
                else:
                    result['true_height'] = -abs(true_height)
                result['error'] = abs(result['height'] - result['true_height'])
                result['seq_name'] = seq_name
                result['phone_position'] = 'hand'
                results_a.append(result)
            except Exception as e:
                tqdm.write(f"  {seq_name} [{start_time:.0f}-{end_time:.0f}s] FAILED: {e}")

            pbar.update(1)

    pbar.close()
    return results_a


# ============================================================
# Figure Generation
# ============================================================

def generate_figures(results_a_bar, results_b_bar, results_a_advio):
    """Generate all evaluation figures for the report."""
    # Merge Bar-Ilan results  
    all_a = results_a_bar + results_a_advio
    
    print("\n" + "=" * 70)
    print("GENERATING FIGURES")
    print("=" * 70)

    # --- Summary Statistics ---
    accepted_a = [r for r in all_a if not r.get('rejected', False)]
    rejected_a = [r for r in all_a if r.get('rejected', False)]
    accepted_b = [r for r in results_b_bar if not r.get('rejected', False)]
    rejected_b = [r for r in results_b_bar if r.get('rejected', False)]

    print(f"\nAlgorithm A (Accel-Only):")
    print(f"  Total: {len(all_a)}, Accepted: {len(accepted_a)}, Rejected: {len(rejected_a)}")
    if accepted_a:
        errors_a = [r['error'] for r in accepted_a]
        print(f"  MAE: {np.mean(errors_a):.3f}m, Median: {np.median(errors_a):.3f}m")
        ci_covered = sum(1 for r in accepted_a if r['error'] <= r['distance_ci_90'])
        print(f"  CI Coverage: {ci_covered}/{len(accepted_a)} = {100*ci_covered/len(accepted_a):.1f}%")

    print(f"\nAlgorithm B (Accel+Orientation, Bar-Ilan only):")
    print(f"  Total: {len(results_b_bar)}, Accepted: {len(accepted_b)}, Rejected: {len(rejected_b)}")
    if accepted_b:
        errors_b = [r['error'] for r in accepted_b]
        print(f"  MAE: {np.mean(errors_b):.3f}m, Median: {np.median(errors_b):.3f}m")
        ci_covered_b = sum(1 for r in accepted_b if r['error'] <= r['distance_ci_90'])
        print(f"  CI Coverage: {ci_covered_b}/{len(accepted_b)} = {100*ci_covered_b/len(accepted_b):.1f}%")

    # Fig 1: True vs Estimated scatter
    _plot_scatter(all_a, results_b_bar, "fig01_scatter.png")

    # Fig 2: Per-ride error bars
    _plot_per_ride_errors(results_a_bar, results_b_bar, "fig02_per_ride_errors.png")

    # Fig 3: Error histogram
    _plot_error_histogram(all_a, results_b_bar, "fig03_error_histogram.png")

    # Fig 4: Error CDF
    _plot_error_cdf(all_a, results_b_bar, "fig04_error_cdf.png")

    # Fig 5: CI coverage
    _plot_ci_coverage(all_a, results_b_bar, "fig05_ci_coverage.png")

    # Fig 6: S-curve overlay plots (key new figure!)
    _plot_scurve_overlays(results_a_bar, "fig06_scurve_overlays.png")

    # Fig 7: Quality score analysis
    _plot_quality_analysis(all_a, "fig07_quality_analysis.png")

    # Fig 8: Hand vs Pocket
    _plot_hand_vs_pocket(results_a_bar, results_b_bar, "fig08_hand_vs_pocket.png")

    # Fig 9: Algorithm comparison (A vs B)
    _plot_algo_comparison(results_a_bar, results_b_bar, "fig09_algo_comparison.png")

    # Fig 10: Summary dashboard
    _plot_summary_dashboard(all_a, results_b_bar, "fig10_summary_dashboard.png")

    # Save JSON results
    _save_results_json(results_a_bar, results_b_bar, results_a_advio)

    print(f"\nAll figures saved to {OUTPUT_DIR}")


def _plot_scatter(results_a, results_b, filename):
    """True vs Estimated height scatter plot."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, title in [
        (axes[0], results_a, "Algorithm A: Accel-Only"),
        (axes[1], results_b, "Algorithm B: Accel+Orientation"),
    ]:
        accepted = [r for r in results if not r.get('rejected')]
        rejected = [r for r in results if r.get('rejected')]

        if accepted:
            true_h = [r['true_height'] for r in accepted]
            est_h = [r['height'] for r in accepted]
            ax.scatter(true_h, est_h, c='#27ae60', s=60, alpha=0.8,
                      edgecolors='white', linewidth=0.5, zorder=5,
                      label='Accepted')
            # CI error bars
            for r in accepted:
                ax.plot([r['true_height'], r['true_height']],
                       [r['height'] - r['distance_ci_90'],
                        r['height'] + r['distance_ci_90']],
                       color='#27ae60', alpha=0.3, linewidth=1.5)

        if rejected:
            true_h = [r['true_height'] for r in rejected]
            est_h = [r['height'] for r in rejected]
            ax.scatter(true_h, est_h, c='#e74c3c', s=40, alpha=0.5,
                      marker='x', linewidth=1.5, label='Rejected')

        # Perfect line
        all_h = ([r['true_height'] for r in results] +
                 [r['height'] for r in results])
        if all_h:
            lim = max(abs(min(all_h)), abs(max(all_h))) * 1.1
            ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.3, label='Perfect')
            ax.set_xlim(-lim, lim)
            ax.set_ylim(-lim, lim)

        ax.set_xlabel('True Height (m)', fontsize=11)
        ax.set_ylabel('Estimated Height (m)', fontsize=11)
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_per_ride_errors(results_a, results_b, filename):
    """Per-ride error bar chart."""
    fig, axes = plt.subplots(2, 1, figsize=(16, 10))

    for ax, results, title in [
        (axes[0], results_a, "Algorithm A: Accel-Only (Bar-Ilan)"),
        (axes[1], results_b, "Algorithm B: Accel+Orientation (Bar-Ilan)"),
    ]:
        if not results:
            ax.text(0.5, 0.5, 'No results', ha='center', va='center')
            continue

        seg_ids = [r.get('seg_id', i) for i, r in enumerate(results)]
        errors = [r['error'] for r in results]
        colors = ['#27ae60' if not r.get('rejected') else '#e74c3c'
                  for r in results]

        bars = ax.bar(range(len(errors)), errors, color=colors,
                     edgecolor='white', alpha=0.85)

        # CI whiskers for accepted
        for i, r in enumerate(results):
            if not r.get('rejected'):
                ax.plot([i, i], [0, r['distance_ci_90']],
                       color='#2c3e50', linewidth=2, alpha=0.5)
                ax.plot([i - 0.2, i + 0.2],
                       [r['distance_ci_90'], r['distance_ci_90']],
                       color='#2c3e50', linewidth=2, alpha=0.5)

        ax.set_xlabel('Segment ID', fontsize=10)
        ax.set_ylabel('Absolute Error (m)', fontsize=10)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xticks(range(len(seg_ids)))
        ax.set_xticklabels([str(s) for s in seg_ids], fontsize=7)
        ax.axhline(1.0, color='gray', linestyle='--', alpha=0.5, label='1m target')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_error_histogram(results_a, results_b, filename):
    """Error distribution histogram."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for ax, results, title in [
        (axes[0], results_a, "Algorithm A"),
        (axes[1], results_b, "Algorithm B"),
    ]:
        accepted = [r for r in results if not r.get('rejected')]
        if not accepted:
            ax.text(0.5, 0.5, 'No accepted results', ha='center', va='center')
            continue
        errors = [r['error'] for r in accepted]
        ax.hist(errors, bins=15, color='#3498db', edgecolor='white', alpha=0.8)
        ax.axvline(np.mean(errors), color='red', linestyle='--',
                  label=f'MAE={np.mean(errors):.2f}m')
        ax.axvline(np.median(errors), color='orange', linestyle='--',
                  label=f'Median={np.median(errors):.2f}m')
        ax.set_xlabel('Absolute Error (m)', fontsize=11)
        ax.set_ylabel('Count', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_error_cdf(results_a, results_b, filename):
    """Cumulative error distribution."""
    fig, ax = plt.subplots(figsize=(10, 6))

    for results, label, color in [
        (results_a, "Algorithm A (Accel-Only)", '#3498db'),
        (results_b, "Algorithm B (Accel+Ori)", '#e67e22'),
    ]:
        accepted = [r for r in results if not r.get('rejected')]
        if not accepted:
            continue
        errors = sorted([r['error'] for r in accepted])
        cdf = np.arange(1, len(errors) + 1) / len(errors)
        ax.plot(errors, cdf, linewidth=2, color=color, label=label)

    ax.axhline(0.5, color='gray', linestyle=':', alpha=0.5)
    ax.axhline(0.9, color='gray', linestyle=':', alpha=0.5)
    ax.axvline(1.0, color='red', linestyle='--', alpha=0.3, label='1m')
    ax.axvline(1.5, color='orange', linestyle='--', alpha=0.3, label='1.5m')
    ax.set_xlabel('Absolute Error (m)', fontsize=12)
    ax.set_ylabel('CDF', fontsize=12)
    ax.set_title('Cumulative Error Distribution (Accepted Rides)', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, None)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_ci_coverage(results_a, results_b, filename):
    """Confidence interval coverage analysis."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for ax, results, title in [
        (axes[0], results_a, "Algorithm A"),
        (axes[1], results_b, "Algorithm B"),
    ]:
        accepted = [r for r in results if not r.get('rejected')]
        if not accepted:
            ax.text(0.5, 0.5, 'No accepted results', ha='center', va='center')
            continue

        errors = [r['error'] for r in accepted]
        cis = [r['distance_ci_90'] for r in accepted]
        covered = [r['error'] <= r['distance_ci_90'] for r in accepted]
        coverage_pct = 100 * sum(covered) / len(accepted)

        colors = ['#27ae60' if c else '#e74c3c' for c in covered]
        idx = range(len(accepted))

        ax.bar(idx, errors, color=colors, alpha=0.7, label='Error')
        ax.scatter(idx, cis, marker='_', color='#2c3e50', s=200,
                  linewidth=2, zorder=5, label='90% CI')
        ax.set_xlabel('Ride Index', fontsize=10)
        ax.set_ylabel('Distance (m)', fontsize=10)
        ax.set_title(f'{title} — Coverage: {coverage_pct:.0f}%',
                    fontsize=12, fontweight='bold')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_scurve_overlays(results, filename):
    """
    KEY FIGURE: Overlay fitted S-curve velocity on measured velocity.
    Shows velocity-domain fit (where fitting actually happens) with
    barometer-derived GT velocity for comparison.
    Each selected ride gets 2 subplots: velocity fit + displacement.
    """
    # Select up to 6 rides with varied characteristics
    accepted = [r for r in results if not r.get('rejected') and 'v_template' in r]
    if len(accepted) == 0:
        print(f"  Skipping {filename}: no accepted results with template data")
        return

    # Pick diverse examples
    by_error = sorted(accepted, key=lambda r: r['error'])
    by_dist = sorted(accepted, key=lambda r: abs(r['true_height']))

    selected = []
    seen_ids = set()
    for candidate_list in [
        [by_error[0]],  # Best
        [by_error[min(2, len(by_error)-1)]],  # Medium
        [by_error[-1]],  # Worst accepted
        [by_dist[0]],  # Shortest
        [by_dist[-1]],  # Longest
        [r for r in accepted if r.get('phone_position') == 'pocket'][:1],
    ]:
        for c in candidate_list:
            sid = c.get('seg_id', id(c))
            if sid not in seen_ids and len(selected) < 6:
                selected.append(c)
                seen_ids.add(sid)

    n_plots = len(selected)
    if n_plots == 0:
        return

    fig, axes = plt.subplots(n_plots, 2, figsize=(16, 3.5 * n_plots))
    if n_plots == 1:
        axes = axes.reshape(1, -1)

    for idx, r in enumerate(selected):
        # Get velocity data
        v_meas = r.get('v_measured', np.zeros(10))
        v_templ = r.get('v_template', np.zeros(len(v_meas)))
        s_templ = r.get('s_template', np.zeros(len(v_meas)))
        n = len(v_meas)

        # Reconstruct time axis from stored or inferred
        t_plot = r.get('t_ride', np.arange(n) * 0.01)
        if len(t_plot) != n:
            t_plot = np.linspace(0, len(t_plot) * 0.01, n)

        seg_id = r.get('seg_id', '?')
        true_h = r.get('true_height', 0)
        est_h = r.get('height', 0)
        err = r.get('error', 0)
        phone = r.get('phone_position', '?')
        direction = r.get('direction', 1)

        # ---- Left panel: Velocity fit ----
        ax_v = axes[idx, 0]
        ax_v.plot(t_plot, v_meas, color='#3498db', linewidth=1.0, alpha=0.7,
                 label='Measured velocity (ZUPT)')
        ax_v.plot(t_plot, v_templ, color='#e74c3c', linewidth=2.0,
                 label='Fitted S-curve velocity')

        # Overlay GT velocity from barometer if available
        gt_t = r.get('gt_time', None)
        gt_vel = r.get('gt_velocity', None)
        if gt_t is not None and gt_vel is not None:
            ax_v.plot(gt_t, gt_vel, color='#2ecc71', linewidth=1.5,
                     linestyle='--', alpha=0.8, label='GT velocity (barometer)')

        ax_v.set_title(
            f"Seg {seg_id} ({phone}) — True: {true_h:+.1f}m, "
            f"Est: {est_h:+.1f}m, Err: {err:.2f}m",
            fontsize=10, fontweight='bold'
        )
        ax_v.set_xlabel('Time (s)', fontsize=9)
        ax_v.set_ylabel('Velocity (m/s)', fontsize=9)
        ax_v.legend(fontsize=7, loc='upper right')
        ax_v.grid(True, alpha=0.2)

        # Add fitted parameters as text
        if 'params' in r and r['params']:
            p = r['params']
            param_txt = (f"j={p.get('j_max',0):.1f} "
                        f"a={p.get('a_max',0):.1f} "
                        f"v={p.get('v_max',0):.1f}")
            ax_v.text(0.02, 0.02, param_txt, transform=ax_v.transAxes,
                     fontsize=7, color='#2c3e50',
                     bbox=dict(facecolor='white', alpha=0.7))

        # ---- Right panel: Displacement ----
        ax_d = axes[idx, 1]

        # Compute measured displacement from velocity integration
        dt_arr = np.diff(t_plot, prepend=t_plot[0])
        dt_arr[0] = dt_arr[1] if len(dt_arr) > 1 else 0.01
        s_meas = np.cumsum(v_meas * dt_arr) * direction

        ax_d.plot(t_plot, s_meas, color='#3498db', linewidth=1.0, alpha=0.7,
                 label='Measured displacement')
        ax_d.plot(t_plot, s_templ * direction, color='#e74c3c', linewidth=2.0,
                 label='Fitted S-curve displacement')

        # Overlay GT height from barometer
        gt_h = r.get('gt_height', None)
        if gt_t is not None and gt_h is not None:
            ax_d.plot(gt_t, gt_h, color='#2ecc71', linewidth=1.5,
                     linestyle='--', alpha=0.8, label='GT height (barometer)')

        ax_d.axhline(y=true_h, color='#27ae60', linestyle=':', alpha=0.5,
                    label=f'True: {true_h:+.1f}m')
        ax_d.axhline(y=est_h, color='#c0392b', linestyle=':', alpha=0.5,
                    label=f'Est: {est_h:+.1f}m')
        ax_d.set_xlabel('Time (s)', fontsize=9)
        ax_d.set_ylabel('Displacement (m)', fontsize=9)
        ax_d.legend(fontsize=7, loc='best')
        ax_d.grid(True, alpha=0.2)

    fig.suptitle('Velocity-Domain S-Curve Fitting: Measured vs Fitted vs Ground Truth',
                fontsize=14, fontweight='bold', y=1.02)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_quality_analysis(results, filename):
    """Quality score vs error analysis."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    all_r = results
    accepted = [r for r in all_r if not r.get('rejected')]
    rejected = [r for r in all_r if r.get('rejected')]

    # Quality score vs error
    ax = axes[0]
    if accepted:
        ax.scatter([r['quality_score'] for r in accepted],
                  [r['error'] for r in accepted],
                  c='#27ae60', s=50, alpha=0.7, label='Accepted')
    if rejected:
        rej_errors = [r.get('error', 0) for r in rejected]
        ax.scatter([r['quality_score'] for r in rejected],
                  rej_errors,
                  c='#e74c3c', s=50, marker='x', alpha=0.7, label='Rejected')
    ax.set_xlabel('Quality Score', fontsize=11)
    ax.set_ylabel('Absolute Error (m)', fontsize=11)
    ax.set_title('Quality Score vs Estimation Error', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # CI width vs error
    ax = axes[1]
    if accepted:
        ci_widths = [r['distance_ci_90'] for r in accepted]
        errors = [r['error'] for r in accepted]
        covered = [r['error'] <= r['distance_ci_90'] for r in accepted]
        colors = ['#27ae60' if c else '#e74c3c' for c in covered]
        ax.scatter(ci_widths, errors, c=colors, s=50, alpha=0.7)
        max_v = max(max(ci_widths), max(errors)) * 1.1
        ax.plot([0, max_v], [0, max_v], 'k--', alpha=0.3, label='Error = CI')
    ax.set_xlabel('90% CI Width (m)', fontsize=11)
    ax.set_ylabel('Absolute Error (m)', fontsize=11)
    ax.set_title('CI Width vs Error (green = covered)', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # Quality score distribution
    ax = axes[2]
    if accepted:
        ax.hist([r['quality_score'] for r in accepted],
               bins=10, color='#27ae60', alpha=0.7, label='Accepted')
    if rejected:
        ax.hist([r['quality_score'] for r in rejected],
               bins=10, color='#e74c3c', alpha=0.7, label='Rejected')
    ax.set_xlabel('Quality Score', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.set_title('Quality Score Distribution', fontsize=12)
    ax.legend(fontsize=9)

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_hand_vs_pocket(results_a, results_b, filename):
    """Performance comparison: hand-held vs pocket."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, results, title in [
        (axes[0], results_a, "Algorithm A"),
        (axes[1], results_b, "Algorithm B"),
    ]:
        hand = [r for r in results if r.get('phone_position') == 'hand'
                and not r.get('rejected')]
        pocket = [r for r in results if r.get('phone_position') == 'pocket'
                  and not r.get('rejected')]

        data = []
        labels = []
        colors = []
        if hand:
            data.append([r['error'] for r in hand])
            labels.append(f'Hand (n={len(hand)})')
            colors.append('#3498db')
        if pocket:
            data.append([r['error'] for r in pocket])
            labels.append(f'Pocket (n={len(pocket)})')
            colors.append('#e67e22')

        if data:
            bp = ax.boxplot(data, labels=labels, patch_artist=True)
            for patch, color in zip(bp['boxes'], colors):
                patch.set_facecolor(color)
                patch.set_alpha(0.6)

        ax.set_ylabel('Absolute Error (m)', fontsize=11)
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_algo_comparison(results_a, results_b, filename):
    """Algorithm A vs B comparison."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Match segments between A and B
    a_by_seg = {r.get('seg_id'): r for r in results_a if not r.get('rejected')}
    b_by_seg = {r.get('seg_id'): r for r in results_b if not r.get('rejected')}
    common = set(a_by_seg.keys()) & set(b_by_seg.keys())

    # Paired error comparison
    ax = axes[0]
    if common:
        a_errors = [a_by_seg[s]['error'] for s in sorted(common)]
        b_errors = [b_by_seg[s]['error'] for s in sorted(common)]
        ax.scatter(a_errors, b_errors, c='#3498db', s=60, alpha=0.8)
        max_e = max(max(a_errors), max(b_errors)) * 1.1
        ax.plot([0, max_e], [0, max_e], 'k--', alpha=0.3)
        ax.set_xlabel('Algorithm A Error (m)', fontsize=11)
        ax.set_ylabel('Algorithm B Error (m)', fontsize=11)
        ax.set_title('Paired Error: A vs B', fontsize=12)
        ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Summary bar chart
    ax = axes[1]
    algos = ['A: Accel-Only', 'B: Accel+Ori']
    acc_a = [r for r in results_a if not r.get('rejected')]
    acc_b = [r for r in results_b if not r.get('rejected')]

    mae_vals = [
        np.mean([r['error'] for r in acc_a]) if acc_a else 0,
        np.mean([r['error'] for r in acc_b]) if acc_b else 0,
    ]
    med_vals = [
        np.median([r['error'] for r in acc_a]) if acc_a else 0,
        np.median([r['error'] for r in acc_b]) if acc_b else 0,
    ]

    x = np.arange(len(algos))
    w = 0.35
    ax.bar(x - w/2, mae_vals, w, color='#3498db', label='MAE', alpha=0.8)
    ax.bar(x + w/2, med_vals, w, color='#e67e22', label='Median', alpha=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(algos)
    ax.set_ylabel('Error (m)', fontsize=11)
    ax.set_title('Algorithm Comparison', fontsize=12)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.2, axis='y')

    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _plot_summary_dashboard(results_a, results_b, filename):
    """Summary performance dashboard."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    for row, (results, title) in enumerate([
        (results_a, "Algorithm A (Accel-Only)"),
        (results_b, "Algorithm B (Accel+Orientation)"),
    ]):
        accepted = [r for r in results if not r.get('rejected')]
        rejected = [r for r in results if r.get('rejected')]

        # Acceptance rate
        ax = axes[row, 0]
        if results:
            sizes = [len(accepted), len(rejected)]
            colors = ['#27ae60', '#e74c3c']
            ax.pie(sizes, labels=['Accepted', 'Rejected'], colors=colors,
                  autopct='%1.0f%%', startangle=90)
        ax.set_title(f'{title}\nAcceptance Rate', fontsize=11)

        # Error stats
        ax = axes[row, 1]
        if accepted:
            errors = [r['error'] for r in accepted]
            stats = {
                'MAE': np.mean(errors),
                'Median': np.median(errors),
                '90th%': np.percentile(errors, 90),
                'Max': np.max(errors),
            }
            ax.barh(list(stats.keys()), list(stats.values()),
                   color='#3498db', alpha=0.8)
            for i, (k, v) in enumerate(stats.items()):
                ax.text(v + 0.05, i, f'{v:.2f}m', va='center', fontsize=10)
        ax.set_xlabel('Error (m)', fontsize=10)
        ax.set_title('Error Statistics (Accepted)', fontsize=11)
        ax.grid(True, alpha=0.2, axis='x')

        # CI coverage
        ax = axes[row, 2]
        if accepted:
            ci_cov = sum(1 for r in accepted
                        if r['error'] <= r['distance_ci_90'])
            pct = 100 * ci_cov / len(accepted)
            ax.bar(['Covered', 'Not Covered'],
                  [ci_cov, len(accepted) - ci_cov],
                  color=['#27ae60', '#e74c3c'], alpha=0.8)
            ax.set_title(f'90% CI Coverage: {pct:.0f}%', fontsize=11)
            ax.set_ylabel('Count', fontsize=10)

    fig.suptitle('Performance Summary Dashboard',
                fontsize=15, fontweight='bold')
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / filename, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"  Saved {filename}")


def _save_results_json(results_a_bar, results_b_bar, results_a_advio):
    """Save detailed results to JSON."""
    def serialize(r):
        out = {}
        for k, v in r.items():
            if isinstance(v, np.ndarray):
                continue  # Skip arrays
            elif isinstance(v, (np.floating, np.integer)):
                out[k] = float(v)
            elif isinstance(v, (float, int, str, bool, type(None))):
                out[k] = v
            elif isinstance(v, dict):
                out[k] = {kk: float(vv) if isinstance(vv, (np.floating, np.integer))
                          else vv for kk, vv in v.items()
                          if not isinstance(vv, np.ndarray)}
        return out

    output = {
        'bar_ilan_algo_a': [serialize(r) for r in results_a_bar],
        'bar_ilan_algo_b': [serialize(r) for r in results_b_bar],
        'advio_algo_a': [serialize(r) for r in results_a_advio],
    }

    with open(RESULTS_DIR / 'results.json', 'w') as f:
        json.dump(output, f, indent=2, default=str)
    print(f"  Saved results.json")


# ============================================================
# Main
# ============================================================

if __name__ == '__main__':
    t_start = time.time()
    print("Kinematics-Based Elevator Height Estimation Evaluation")
    print("=" * 70)

    # Run evaluations
    print("\n[1/3] Bar-Ilan evaluation...")
    t1 = time.time()
    results_a_bar, results_b_bar, bar_segments = run_bar_ilan_evaluation()
    print(f"  Done in {time.time() - t1:.1f}s")

    print("\n[2/3] ADVIO evaluation...")
    t2 = time.time()
    results_a_advio = run_advio_evaluation()
    print(f"  Done in {time.time() - t2:.1f}s")

    # Generate figures
    print("\n[3/3] Generating figures...")
    t3 = time.time()
    generate_figures(results_a_bar, results_b_bar, results_a_advio)
    print(f"  Done in {time.time() - t3:.1f}s")

    total_time = time.time() - t_start
    print(f"\n{'=' * 70}")
    print(f"EVALUATION COMPLETE — Total time: {total_time:.1f}s ({total_time/60:.1f}min)")
    print(f"{'=' * 70}")
