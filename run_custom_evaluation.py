#!/usr/bin/env python3
"""
Custom Dataset Evaluation — Test-Set Evaluation Pipeline

Evaluates the elevator height estimation pipeline on a user-provided tagged
dataset. Uses the pre-calibrated model (no re-training/calibration on user data).

Two modes:
  - full:          Run detection + segmentation + quality + estimation,
                   then match detected segments against GT to evaluate the
                   entire pipeline end-to-end.
  - segments_only: Skip detection, use user-provided segment boundaries
                   directly, evaluate quality + height estimation only.

Usage:
    python run_custom_evaluation.py --dataset my_data.csv --mode segments_only
    python run_custom_evaluation.py --dataset my_data.csv --mode full

See README.md for full documentation on the expected CSV format.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from pipeline import ElevatorHeightPipeline, detect_elevator_rides, estimate_height_robust
from algorithms.quality_filter import assess_segment_quality


# ======================================================================
# Data Loading
# ======================================================================

def load_dataset(csv_path):
    """
    Load user-provided evaluation dataset CSV.

    Required columns:
        segment_id, acc_data_path, start_time, end_time, true_height
    Optional columns:
        phone_position, fs

    Returns:
        list of ride dicts, grouped by acc_data_path
    """
    df = pd.read_csv(csv_path)

    # Validate required columns
    required = ['segment_id', 'acc_data_path', 'start_time', 'end_time', 'true_height']
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Dataset CSV missing required columns: {missing}\n"
                         f"Found columns: {list(df.columns)}")

    rides = []
    for _, row in df.iterrows():
        ride = {
            'segment_id': row['segment_id'],
            'acc_data_path': row['acc_data_path'],
            'start_time': float(row['start_time']),
            'end_time': float(row['end_time']),
            'true_height': float(row['true_height']),
            'phone_position': str(row.get('phone_position', 'unknown'))
                              if pd.notna(row.get('phone_position')) else 'unknown',
            'fs': float(row['fs']) if 'fs' in row and pd.notna(row.get('fs')) else None,
        }
        rides.append(ride)

    return rides


def load_accelerometer_csv(path, default_fs=100):
    """Load accelerometer data from CSV, flexible column names.
    
    Handles:
    - Standard CSV with headers (acc_x, acc_y, acc_z, time)
    - Headerless CSV with 4 columns (time_ms, x, y, z) — e.g. Bar-Ilan format
    - Various column name conventions
    """
    df = pd.read_csv(path)

    # Detect headerless CSV: if all column names look numeric, re-read with names
    all_numeric_cols = all(_is_numeric_string(str(c)) for c in df.columns)
    if all_numeric_cols and len(df.columns) == 4:
        df = pd.read_csv(path, names=["time_ms", "x", "y", "z"])

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

    if not all(k in col_map for k in ['acc_x', 'acc_y', 'acc_z']):
        raise ValueError(f"CSV must have acc_x/y/z columns. Found: {list(df.columns)}")

    ax = df[col_map['acc_x']].values.astype(float)
    ay = df[col_map['acc_y']].values.astype(float)
    az = df[col_map['acc_z']].values.astype(float)

    fs = default_fs
    if 'time' in col_map:
        t_raw = df[col_map['time']].values.astype(float)
        # Convert ms to sec if needed
        if col_map.get('time_key') == 'time_ms' or (len(t_raw) > 1 and np.median(np.diff(t_raw)) > 1):
            t_raw = t_raw / 1000.0
        if len(t_raw) > 1:
            fs = 1.0 / np.median(np.diff(t_raw))
        t = t_raw - t_raw[0]  # zero-base
    else:
        t = np.arange(len(ax)) / fs

    return t, ax, ay, az, fs


def _is_numeric_string(s):
    """Check if a string looks like a number."""
    try:
        float(s)
        return True
    except (ValueError, TypeError):
        return False


# ======================================================================
# Matching (for full mode)
# ======================================================================

def match_gt_to_detected(gt_rides, det_rides, n_samples, t, iou_threshold=0.3):
    """Match GT rides to detected rides based on temporal IoU."""
    matches = []
    for gi, gt in enumerate(gt_rides):
        gt_si = int(np.argmin(np.abs(t - gt['start_time'])))
        gt_ei = int(np.argmin(np.abs(t - gt['end_time'])))
        gt_mask = np.zeros(n_samples, dtype=bool)
        gt_mask[gt_si:gt_ei] = True

        best_iou, best_di = 0, -1
        for di, det in enumerate(det_rides):
            det_mask = np.zeros(n_samples, dtype=bool)
            det_mask[det['s_idx']:det['e_idx']] = True
            inter = np.sum(gt_mask & det_mask)
            union = np.sum(gt_mask | det_mask)
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_di = di

        matches.append((gi, best_di if best_iou >= iou_threshold else -1, best_iou))
    return matches


# ======================================================================
# Evaluation Core
# ======================================================================

def evaluate_segments_only(pipeline, rides_by_file, verbose=False):
    """Evaluate using user-provided segment boundaries (skip detection)."""
    results = []
    fs_default = pipeline.fs

    for acc_path, rides in rides_by_file.items():
        if verbose:
            print(f"\nProcessing: {acc_path} ({len(rides)} segments)")

        fs = rides[0].get('fs') or fs_default
        t, ax, ay, az, fs = load_accelerometer_csv(acc_path, default_fs=fs)

        segments = [{'start_time': r['start_time'], 'end_time': r['end_time']}
                    for r in rides]

        seg_results = pipeline.process_segments(ax, ay, az, segments, fs=fs)

        for ride, seg_res in zip(rides, seg_results):
            err = abs(seg_res['height_estimate'] - ride['true_height'])
            results.append({
                'segment_id': ride['segment_id'],
                'acc_data_path': acc_path,
                'true_dh': ride['true_height'],
                'est_dh': seg_res['height_estimate'],
                'err': err,
                'method': seg_res['method'],
                'accepted': seg_res['accepted'],
                'reject_reason': seg_res.get('reject_reason', ''),
                'quality_score': seg_res.get('quality_score', 0),
                'quality_features': seg_res.get('quality_features', {}),
                'start_time': ride['start_time'],
                'end_time': ride['end_time'],
                'phone': ride.get('phone_position', 'unknown'),
                'confidence_interval_90': seg_res.get('confidence_interval_90'),
                'pos_curve': seg_res.get('pos_curve', np.array([])),
                'all_estimates': seg_res.get('all_estimates', {}),
            })

            if verbose:
                status = "✓" if seg_res['accepted'] else "✗"
                print(f"  {status} Seg {ride['segment_id']}: "
                      f"true={ride['true_height']:+.1f}m "
                      f"est={seg_res['height_estimate']:+.2f}m "
                      f"err={err:.2f}m [{seg_res['method']}]"
                      f"{'' if seg_res['accepted'] else ' — ' + seg_res.get('reject_reason', '')}")

    return results, None  # None = no detection info


def evaluate_full(pipeline, rides_by_file, verbose=False):
    """Evaluate full pipeline including detection + matching."""
    results = []
    all_detection_info = []
    fs_default = pipeline.fs

    for acc_path, rides in rides_by_file.items():
        if verbose:
            print(f"\nProcessing: {acc_path} ({len(rides)} GT segments)")

        fs = rides[0].get('fs') or fs_default
        t, ax, ay, az, fs = load_accelerometer_csv(acc_path, default_fs=fs)
        n = len(t)

        # Run full pipeline detection
        det_rides = detect_elevator_rides(t, ax, ay, az, fs=fs)
        if verbose:
            print(f"  Detected {len(det_rides)} segments")

        # Match GT to detected
        matches = match_gt_to_detected(rides, det_rides, n, t)
        matched_count = sum(1 for _, di, _ in matches if di >= 0)
        if verbose:
            print(f"  Matched: {matched_count}/{len(rides)}")

        # For each GT ride, evaluate height estimation
        pre_win = int(fs * pipeline.params['pre_window_sec'])
        post_win = int(fs * pipeline.params['post_window_sec'])

        for gi, (_, di, iou) in enumerate(matches):
            ride = rides[gi]

            # Use detected segment boundaries if matched, else GT
            if di >= 0:
                si = det_rides[di]['s_idx']
                ei = det_rides[di]['e_idx']
            else:
                si = int(np.argmin(np.abs(t - ride['start_time'])))
                ei = int(np.argmin(np.abs(t - ride['end_time'])))

            ride_ax, ride_ay, ride_az = ax[si:ei], ay[si:ei], az[si:ei]
            ride_t = t[si:ei]

            pre_s = max(0, si - pre_win)
            post_e = min(n, ei + post_win)
            pre_ax, pre_ay, pre_az = ax[pre_s:si], ay[pre_s:si], az[pre_s:si]
            post_ax, post_ay, post_az = ax[ei:post_e], ay[ei:post_e], az[ei:post_e]

            qa = assess_segment_quality(ride_ax, ride_ay, ride_az,
                                        pre_ax, pre_ay, pre_az,
                                        post_ax, post_ay, post_az, fs=fs)

            est = estimate_height_robust(ride_t, ride_ax, ride_ay, ride_az,
                                         pre_ax, pre_ay, pre_az,
                                         post_ax, post_ay, post_az, fs=fs)

            # Post-estimation checks (same as pipeline)
            h_mag = est['all_estimates'].get('magnitude')
            h_gp = est['all_estimates'].get('gravity_proj')
            if abs(est['height']) > pipeline.params['max_implausible_m']:
                qa['accept'] = False
                qa['reject_reason'] = f'Estimate implausible: {est["height"]:.1f}m'
            if qa['accept'] and h_mag is not None and h_gp is not None:
                if abs(h_mag) > 1.0:
                    ratio = abs(h_gp) / abs(h_mag)
                    if ratio > pipeline.params['mag_cross_ratio']:
                        qa['accept'] = False
                        qa['reject_reason'] = f'Projection/magnitude disagree: ratio={ratio:.1f}'
            if (qa['accept'] and est['method'] == 'signed_mag' and
                    abs(est['height']) > pipeline.params['signed_mag_max_m']):
                qa['accept'] = False
                qa['reject_reason'] = f'Signed-mag unreliable ({est["height"]:.1f}m)'

            err = abs(est['height'] - ride['true_height'])
            results.append({
                'segment_id': ride['segment_id'],
                'acc_data_path': acc_path,
                'true_dh': ride['true_height'],
                'est_dh': float(est['height']),
                'err': err,
                'method': est['method'],
                'accepted': qa['accept'],
                'reject_reason': qa.get('reject_reason', ''),
                'quality_score': qa.get('quality_score', 0),
                'quality_features': qa.get('features', {}),
                'start_time': ride['start_time'],
                'end_time': ride['end_time'],
                'phone': ride.get('phone_position', 'unknown'),
                'confidence_interval_90': pipeline.conformal_interval,
                'pos_curve': est.get('pos', np.array([])),
                'all_estimates': est.get('all_estimates', {}),
                'det_matched': di >= 0,
                'det_iou': iou,
            })

        detection_info = {
            'acc_data_path': acc_path,
            'n_gt': len(rides),
            'n_detected': len(det_rides),
            'n_matched': matched_count,
            'matches': matches,
            'det_rides': det_rides,
            'gt_rides': rides,
            't': t,
            'acc_mag': np.sqrt(ax**2 + ay**2 + az**2),
        }
        all_detection_info.append(detection_info)

    return results, all_detection_info


# ======================================================================
# Metrics Computation
# ======================================================================

def compute_metrics(results, conformal_interval):
    """Compute comprehensive evaluation metrics."""
    accepted = [r for r in results if r['accepted']]
    rejected = [r for r in results if not r['accepted']]

    all_errors = [r['err'] for r in results]
    acc_errors = [r['err'] for r in accepted]
    rej_errors = [r['err'] for r in rejected]

    metrics = {
        'n_total': len(results),
        'n_accepted': len(accepted),
        'n_rejected': len(rejected),
        'acceptance_rate': len(accepted) / len(results) * 100 if results else 0,
    }

    if all_errors:
        metrics['all_mae'] = float(np.mean(all_errors))
        metrics['all_median'] = float(np.median(all_errors))

    if acc_errors:
        metrics['accepted_mae'] = float(np.mean(acc_errors))
        metrics['accepted_median'] = float(np.median(acc_errors))
        metrics['accepted_max_err'] = float(np.max(acc_errors))
        metrics['accepted_within_05m'] = sum(1 for e in acc_errors if e < 0.5)
        metrics['accepted_within_1m'] = sum(1 for e in acc_errors if e < 1.0)
        metrics['accepted_within_2m'] = sum(1 for e in acc_errors if e < 2.0)
        metrics['accepted_within_3m'] = sum(1 for e in acc_errors if e < 3.0)

    # Conformal coverage check (using pre-calibrated interval)
    if conformal_interval and acc_errors:
        covered = sum(1 for e in acc_errors if e <= conformal_interval)
        metrics['conformal_interval'] = conformal_interval
        metrics['conformal_coverage'] = covered / len(acc_errors) * 100
        metrics['conformal_covered'] = covered
        metrics['conformal_total'] = len(acc_errors)

    # Rejection quality
    if rej_errors:
        metrics['rejection_would_be_bad'] = sum(1 for e in rej_errors if e > 1.0)
        metrics['rejection_quality'] = (
            metrics['rejection_would_be_bad'] / len(rej_errors) * 100
        )

    return metrics


# ======================================================================
# Figure Generation
# ======================================================================

def generate_figures(results, metrics, conformal_interval, output_dir, mode,
                     detection_info=None, fs=100):
    """Generate all evaluation figures."""
    accepted = [r for r in results if r['accepted']]
    rejected = [r for r in results if not r['accepted']]

    fig_num = 0

    # --- FIG: Detection timeline (full mode only) ---
    if mode == 'full' and detection_info:
        fig_num += 1
        for info in detection_info:
            fig, axes = plt.subplots(2, 1, figsize=(16, 6), sharex=True)

            # Accel magnitude
            axes[0].plot(info['t'], info['acc_mag'], 'b-', lw=0.3, alpha=0.6)
            axes[0].axhline(9.81, color='gray', ls='--', alpha=0.5)
            axes[0].set_ylabel('|a| (m/s²)')
            axes[0].set_title(f'Detection Timeline — {os.path.basename(info["acc_data_path"])}')
            axes[0].grid(True, alpha=0.3)

            # GT vs Detected
            for r in info['gt_rides']:
                axes[1].axvspan(r['start_time'], r['end_time'],
                                color='green', alpha=0.3)
            for d in info['det_rides']:
                axes[1].axvspan(info['t'][d['s_idx']], info['t'][d['e_idx']],
                                color='blue', alpha=0.3)
            axes[1].set_ylabel('Segments')
            axes[1].set_xlabel('Time (s)')
            n_det = info['n_detected']
            n_gt = info['n_gt']
            n_match = info['n_matched']
            axes[1].set_title(f'Detected: {n_det} (blue) vs GT: {n_gt} (green) — '
                              f'{n_match}/{n_gt} matched')
            custom_lines = [
                plt.Rectangle((0, 0), 1, 1, fc='green', alpha=0.3),
                plt.Rectangle((0, 0), 1, 1, fc='blue', alpha=0.3),
            ]
            axes[1].legend(custom_lines, ['GT', 'Detected'], loc='upper right')
            axes[1].grid(True, alpha=0.3)

            fig.tight_layout()
            fname = f'fig{fig_num:02d}_detection_{os.path.basename(info["acc_data_path"]).replace(".csv", "")}.png'
            fig.savefig(os.path.join(output_dir, fname), dpi=150)
            plt.close(fig)

    # --- FIG: True vs Estimated scatter ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(8, 8))
    if accepted:
        trues_a = [r['true_dh'] for r in accepted]
        ests_a = [r['est_dh'] for r in accepted]
        ax.scatter(trues_a, ests_a, c='green', alpha=0.7, s=60,
                   label=f'Accepted ({len(accepted)})')
    if rejected:
        trues_r = [r['true_dh'] for r in rejected]
        ests_r = [r['est_dh'] for r in rejected]
        ax.scatter(trues_r, ests_r, c='red', alpha=0.5, s=60, marker='x',
                   label=f'Rejected ({len(rejected)})')

    all_vals = [r['true_dh'] for r in results] + [r['est_dh'] for r in results]
    if all_vals:
        lim = max(abs(min(all_vals)), abs(max(all_vals))) + 5
        ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5, label='Perfect')
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)

    mae = metrics.get('accepted_mae', 0)
    ax.set_xlabel('True Δh (m)')
    ax.set_ylabel('Estimated Δh (m)')
    ax.set_title(f'Test Set — Height Estimation (Accepted MAE={mae:.2f}m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_scatter.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Per-ride error bars ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(max(10, len(results) * 0.5), 5))
    x = range(len(results))
    colors = ['green' if r['accepted'] else 'red' for r in results]
    errors = [min(r['err'], 20) for r in results]
    ax.bar(x, errors, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(1.0, color='blue', ls='--', alpha=0.7, lw=2, label='1m target')
    ax.axhline(3.0, color='orange', ls='--', alpha=0.5, label='1 floor (3m)')
    if conformal_interval:
        ax.axhline(conformal_interval, color='purple', ls=':', alpha=0.5,
                    label=f'Conformal ±{conformal_interval:.2f}m')

    for i, r in enumerate(results):
        ax.text(i, min(r['err'], 20) + 0.3, str(r['segment_id']),
                ha='center', va='bottom', fontsize=7, rotation=90)

    ax.set_xlabel('Ride Index')
    ax.set_ylabel('Absolute Error (m)')
    ax.set_title('Test Set — Per-Ride Error (green=accepted, red=rejected)')
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.set_ylim(0, min(22, max(errors) + 3) if errors else 5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_per_ride_errors.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Error histogram ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    acc_errors = [r['err'] for r in accepted]
    rej_errors = [r['err'] for r in rejected]
    if acc_errors:
        ax.hist(acc_errors, bins=15, color='green', alpha=0.6, label='Accepted', edgecolor='black')
    if rej_errors:
        ax.hist(rej_errors, bins=15, color='red', alpha=0.4, label='Rejected', edgecolor='black')
    ax.axvline(1.0, color='blue', ls='--', lw=2, label='1m target')
    ax.set_xlabel('Absolute Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Test Set — Error Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_error_histogram.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Error CDF ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(8, 5))
    if acc_errors:
        sorted_err = sorted(acc_errors)
        cdf = np.arange(1, len(sorted_err) + 1) / len(sorted_err)
        ax.plot(sorted_err, cdf * 100, 'g-o', ms=4, label=f'Accepted ({len(accepted)})')
    all_e = sorted([r['err'] for r in results])
    if all_e:
        cdf_all = np.arange(1, len(all_e) + 1) / len(all_e)
        ax.plot(all_e, cdf_all * 100, 'r--', ms=3, alpha=0.5, label=f'All ({len(results)})')
    ax.axvline(1.0, color='blue', ls='--', alpha=0.5, label='1m')
    ax.axvline(3.0, color='orange', ls='--', alpha=0.3, label='3m (1 floor)')
    ax.set_xlabel('Absolute Error (m)')
    ax.set_ylabel('Cumulative %')
    ax.set_title('Test Set — Error CDF')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim(0, max(all_e) * 1.1 if all_e else 5)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_error_cdf.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Conformal coverage ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(12, 6))
    if accepted and conformal_interval:
        x_conf = range(len(accepted))
        ests = [r['est_dh'] for r in accepted]
        trues = [r['true_dh'] for r in accepted]

        ax.errorbar(x_conf, ests, yerr=conformal_interval, fmt='o', color='blue',
                     ecolor='lightblue', capsize=4, capthick=2,
                     label=f'Est ± {conformal_interval:.2f}m (pre-calibrated 90% CI)')
        ax.scatter(x_conf, trues, color='red', marker='x', s=80, zorder=5,
                   label='True Δh')

        for i, r in enumerate(accepted):
            covered = r['err'] <= conformal_interval
            ax.axvspan(i - 0.4, i + 0.4,
                       color='lightgreen' if covered else 'lightsalmon', alpha=0.2)

        cov = metrics.get('conformal_coverage', 0)
        ax.set_title(f'Test Set — Pre-Calibrated Conformal Coverage: {cov:.0f}% '
                     f'(interval=±{conformal_interval:.2f}m, target: ≥90%)')
    else:
        ax.text(0.5, 0.5, 'No accepted rides or conformal interval',
                transform=ax.transAxes, ha='center')
        ax.set_title('Conformal Coverage — N/A')

    ax.set_xlabel('Accepted Ride Index')
    ax.set_ylabel('Height Change (m)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_conformal_coverage.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Quality score vs error ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    if accepted:
        ax.scatter([r['quality_score'] for r in accepted],
                   [r['err'] for r in accepted],
                   c='green', alpha=0.7, s=60, label='Accepted')
    if rejected:
        ax.scatter([r['quality_score'] for r in rejected],
                   [r['err'] for r in rejected],
                   c='red', alpha=0.5, s=60, marker='x', label='Rejected')
    ax.axhline(1.0, color='blue', ls='--', alpha=0.5, label='1m target')
    ax.set_xlabel('Quality Score (lower = better)')
    ax.set_ylabel('Absolute Error (m)')
    ax.set_title('Test Set — Quality Score vs Height Error')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_quality_vs_error.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Rejection reasons ---
    fig_num += 1
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    if rejected:
        # Left: error sorted comparison
        acc_e_sorted = sorted(acc_errors) if acc_errors else []
        rej_e_sorted = sorted(rej_errors)
        all_e_sorted = sorted(acc_errors + rej_errors)
        axes[0].plot(range(len(all_e_sorted)), all_e_sorted, 'r-o', ms=4,
                     label=f'All ({len(all_e_sorted)})')
        if acc_e_sorted:
            axes[0].plot(range(len(acc_e_sorted)), acc_e_sorted, 'g-o', ms=4,
                         label=f'Accepted ({len(acc_e_sorted)})')
        axes[0].axhline(1.0, color='blue', ls='--', alpha=0.5)
        axes[0].set_xlabel('Ride (sorted by error)')
        axes[0].set_ylabel('Absolute Error (m)')
        axes[0].set_title('Error Distribution: All vs Accepted')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # Right: rejection reasons
        reasons = [r['reject_reason'] for r in rejected]
        reason_counts = {}
        for r in reasons:
            short = r[:40]
            reason_counts[short] = reason_counts.get(short, 0) + 1
        axes[1].barh(list(reason_counts.keys()), list(reason_counts.values()),
                     color='red', alpha=0.7)
        axes[1].set_xlabel('Count')
        axes[1].set_title('Rejection Reasons')
        axes[1].grid(axis='x', alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, 'No rejected rides', transform=axes[0].transAxes, ha='center')
        axes[1].text(0.5, 0.5, 'No rejected rides', transform=axes[1].transAxes, ha='center')

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_rejection_analysis.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Error vs true height ---
    fig_num += 1
    fig, ax = plt.subplots(figsize=(8, 6))
    if accepted:
        ax.scatter([abs(r['true_dh']) for r in accepted],
                   [r['err'] for r in accepted],
                   c='green', alpha=0.7, s=60, label='Accepted')
    if rejected:
        ax.scatter([abs(r['true_dh']) for r in rejected],
                   [r['err'] for r in rejected],
                   c='red', alpha=0.5, s=60, marker='x', label='Rejected')
    ax.axhline(1.0, color='blue', ls='--', alpha=0.5)
    ax.set_xlabel('|True Height Difference| (m)')
    ax.set_ylabel('Absolute Error (m)')
    ax.set_title('Test Set — Error vs Ride Height')
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_error_vs_height.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Method breakdown ---
    fig_num += 1
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    methods = {}
    for r in results:
        m = r['method']
        methods.setdefault(m, {'accepted': [], 'rejected': []})
        if r['accepted']:
            methods[m]['accepted'].append(r['err'])
        else:
            methods[m]['rejected'].append(r['err'])

    method_names = list(methods.keys())
    acc_counts = [len(methods[m]['accepted']) for m in method_names]
    rej_counts = [len(methods[m]['rejected']) for m in method_names]
    x_m = range(len(method_names))
    axes[0].bar(x_m, acc_counts, color='green', alpha=0.7, label='Accepted')
    axes[0].bar(x_m, rej_counts, bottom=acc_counts, color='red', alpha=0.5,
                label='Rejected')
    axes[0].set_xticks(x_m)
    axes[0].set_xticklabels(method_names, rotation=45, ha='right')
    axes[0].set_ylabel('Count')
    axes[0].set_title('Method Usage')
    axes[0].legend()
    axes[0].grid(axis='y', alpha=0.3)

    # MAE per method (accepted only)
    method_maes = []
    method_labels = []
    for m in method_names:
        if methods[m]['accepted']:
            method_maes.append(np.mean(methods[m]['accepted']))
            method_labels.append(m)
    if method_maes:
        axes[1].bar(range(len(method_labels)), method_maes, color='steelblue', alpha=0.7)
        axes[1].set_xticks(range(len(method_labels)))
        axes[1].set_xticklabels(method_labels, rotation=45, ha='right')
        axes[1].set_ylabel('MAE (m)')
        axes[1].set_title('Accepted MAE by Method')
        axes[1].grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_method_breakdown.png'), dpi=150)
    plt.close(fig)

    # --- FIG: Individual ride analysis (best + worst) ---
    fig_num += 1
    acc_sorted = sorted(accepted, key=lambda r: r['err'])
    n_show = min(3, len(acc_sorted))
    examples = acc_sorted[:n_show]
    if len(acc_sorted) > n_show:
        examples += acc_sorted[-min(2, len(acc_sorted) - n_show):]

    if examples:
        fig, axes = plt.subplots(len(examples), 1, figsize=(12, 3 * len(examples)))
        if len(examples) == 1:
            axes = [axes]
        for idx, d in enumerate(examples):
            ax = axes[idx]
            pos = d.get('pos_curve', np.array([]))
            if len(pos) > 0:
                ride_t = np.arange(len(pos)) / fs
                ax.plot(ride_t, pos, 'b-', lw=1.5, label='Estimated')
                gt_line = np.linspace(0, d['true_dh'], len(ride_t))
                ax.plot(ride_t, gt_line, 'r--', lw=1, label='GT (linear)')
            label = "BEST" if idx < n_show else "WORST"
            ax.set_ylabel('Δh (m)')
            ax.set_title(f'[{label}] Seg {d["segment_id"]}: True={d["true_dh"]:+.1f}m, '
                         f'Est={d["est_dh"]:+.2f}m, Err={d["err"]:.2f}m [{d["phone"]}]')
            ax.legend(loc='upper right')
            ax.grid(True, alpha=0.3)
        axes[-1].set_xlabel('Time within ride (s)')
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_individual_rides.png'), dpi=150)
        plt.close(fig)

    # --- FIG: Phone position comparison (if applicable) ---
    phones = set(r['phone'] for r in results if r['phone'] != 'unknown')
    if len(phones) > 1:
        fig_num += 1
        fig, ax = plt.subplots(figsize=(8, 6))
        for phone in sorted(phones):
            phone_acc = [r['err'] for r in accepted if r['phone'] == phone]
            phone_rej = [r['err'] for r in rejected if r['phone'] == phone]
            if phone_acc:
                ax.scatter([phone] * len(phone_acc), phone_acc, c='green',
                           alpha=0.6, s=60)
            if phone_rej:
                ax.scatter([phone] * len(phone_rej), phone_rej, c='red',
                           alpha=0.4, s=60, marker='x')
        ax.axhline(1.0, color='blue', ls='--', alpha=0.5)
        ax.set_ylabel('Absolute Error (m)')
        ax.set_title('Test Set — Error by Phone Position')
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_phone_positions.png'), dpi=150)
        plt.close(fig)

    # --- FIG: Summary dashboard ---
    fig_num += 1
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('TEST SET — Evaluation Summary Dashboard', fontsize=16, fontweight='bold')

    # Top-left: key metrics table
    ax = axes[0, 0]
    ax.axis('off')
    table_data = [
        ['Total Rides', str(metrics['n_total'])],
        ['Accepted', f"{metrics['n_accepted']} ({metrics['acceptance_rate']:.0f}%)"],
        ['Rejected', str(metrics['n_rejected'])],
    ]
    if 'accepted_mae' in metrics:
        table_data.append(['Accepted MAE', f"{metrics['accepted_mae']:.3f}m"])
        table_data.append(['Accepted Median', f"{metrics['accepted_median']:.3f}m"])
        table_data.append(['Max Error', f"{metrics['accepted_max_err']:.3f}m"])
    if 'conformal_coverage' in metrics:
        table_data.append(['Conformal Interval', f"±{metrics['conformal_interval']:.2f}m"])
        cov = metrics['conformal_coverage']
        table_data.append(['Conformal Coverage',
                           f"{cov:.0f}% ({'✓ PASS' if cov >= 90 else '✗ BELOW TARGET'})"])
    table = ax.table(cellText=table_data, colLabels=['Metric', 'Value'],
                     loc='center', cellLoc='left')
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1, 1.5)
    ax.set_title('Key Metrics', fontweight='bold', pad=20)

    # Top-right: scatter
    ax = axes[0, 1]
    if accepted:
        ax.scatter([r['true_dh'] for r in accepted], [r['est_dh'] for r in accepted],
                   c='green', alpha=0.7, s=40)
    if rejected:
        ax.scatter([r['true_dh'] for r in rejected], [r['est_dh'] for r in rejected],
                   c='red', alpha=0.4, s=40, marker='x')
    if all_vals:
        lim2 = max(abs(min(all_vals)), abs(max(all_vals))) + 3
        ax.plot([-lim2, lim2], [-lim2, lim2], 'k--', alpha=0.4)
        ax.set_xlim(-lim2, lim2)
        ax.set_ylim(-lim2, lim2)
    ax.set_xlabel('True (m)')
    ax.set_ylabel('Est (m)')
    ax.set_title('True vs Estimated')
    ax.set_aspect('equal')
    ax.grid(True, alpha=0.3)

    # Bottom-left: error histogram
    ax = axes[1, 0]
    if acc_errors:
        ax.hist(acc_errors, bins=12, color='green', alpha=0.6, edgecolor='black')
    ax.axvline(1.0, color='blue', ls='--', lw=2)
    ax.set_xlabel('Error (m)')
    ax.set_ylabel('Count')
    ax.set_title('Accepted Error Distribution')
    ax.grid(True, alpha=0.3)

    # Bottom-right: per ride bar
    ax = axes[1, 1]
    if results:
        x_bar = range(len(results))
        bar_colors = ['green' if r['accepted'] else 'red' for r in results]
        ax.bar(x_bar, [r['err'] for r in results], color=bar_colors, alpha=0.7)
        ax.axhline(1.0, color='blue', ls='--')
        if conformal_interval:
            ax.axhline(conformal_interval, color='purple', ls=':',
                       label=f'CI={conformal_interval:.1f}m')
            ax.legend(fontsize=8)
    ax.set_xlabel('Ride')
    ax.set_ylabel('Error (m)')
    ax.set_title('Per-Ride Errors')
    ax.grid(axis='y', alpha=0.3)

    fig.tight_layout()
    fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_summary_dashboard.png'), dpi=150)
    plt.close(fig)

    # --- FIG: IoU distribution (full mode only) ---
    if mode == 'full' and detection_info:
        fig_num += 1
        all_ious = [r.get('det_iou', 0) for r in results]
        matched_ious = [iou for iou in all_ious if iou > 0]
        if matched_ious:
            fig, ax = plt.subplots(figsize=(8, 5))
            ax.hist(matched_ious, bins=15, color='steelblue', alpha=0.7,
                    edgecolor='black')
            ax.axvline(0.3, color='red', ls='--', label='IoU threshold (0.3)')
            ax.set_xlabel('IoU')
            ax.set_ylabel('Count')
            ax.set_title('Test Set — Detection IoU Distribution')
            ax.legend()
            ax.grid(True, alpha=0.3)
            fig.tight_layout()
            fig.savefig(os.path.join(output_dir, f'fig{fig_num:02d}_iou_distribution.png'),
                        dpi=150)
            plt.close(fig)

    print(f"  {fig_num} figures generated in {output_dir}/")
    return fig_num


# ======================================================================
# Main
# ======================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate elevator height estimation on a custom dataset (test set).",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Dataset CSV format:
  segment_id, acc_data_path, start_time, end_time, true_height [, phone_position, fs]

Example:
  python run_custom_evaluation.py --dataset my_data.csv --mode segments_only
  python run_custom_evaluation.py --dataset my_data.csv --mode full -v
        """
    )
    parser.add_argument("--dataset", "-d", required=True,
                        help="Path to dataset CSV index file")
    parser.add_argument("--mode", "-m", choices=['full', 'segments_only'],
                        default='segments_only',
                        help="Evaluation mode (default: segments_only)")
    parser.add_argument("--output-dir", "-o", default="evaluation_output",
                        help="Output directory for figures and results")
    parser.add_argument("--model-dir", default="model/",
                        help="Pipeline model directory")
    parser.add_argument("--verbose", "-v", action="store_true")

    args = parser.parse_args()

    print("=" * 70)
    print(f"CUSTOM DATASET EVALUATION — Mode: {args.mode.upper()}")
    print("=" * 70)

    # Load pipeline
    pipeline = ElevatorHeightPipeline.load(args.model_dir)
    conformal_interval = pipeline.conformal_interval
    print(f"Pipeline loaded from {args.model_dir}")
    print(f"Pre-calibrated conformal interval: ±{conformal_interval:.2f}m")

    # Load dataset
    rides = load_dataset(args.dataset)
    print(f"Dataset: {len(rides)} segments from {args.dataset}")

    # Group by accelerometer file
    rides_by_file = {}
    for r in rides:
        rides_by_file.setdefault(r['acc_data_path'], []).append(r)
    print(f"Recordings: {len(rides_by_file)} unique files")

    # Run evaluation
    print(f"\n{'='*50}")
    print(f"EVALUATION ({args.mode})")
    print(f"{'='*50}")

    if args.mode == 'segments_only':
        results, det_info = evaluate_segments_only(pipeline, rides_by_file,
                                                   verbose=args.verbose)
    else:
        results, det_info = evaluate_full(pipeline, rides_by_file,
                                          verbose=args.verbose)

    # Compute metrics
    metrics = compute_metrics(results, conformal_interval)

    # Print summary
    print(f"\n{'='*50}")
    print("RESULTS SUMMARY")
    print(f"{'='*50}")
    print(f"Total rides: {metrics['n_total']}")
    print(f"Accepted: {metrics['n_accepted']} ({metrics['acceptance_rate']:.0f}%)")
    print(f"Rejected: {metrics['n_rejected']}")

    if 'accepted_mae' in metrics:
        print(f"\nAccepted rides:")
        print(f"  MAE:    {metrics['accepted_mae']:.3f}m")
        print(f"  Median: {metrics['accepted_median']:.3f}m")
        print(f"  Max:    {metrics['accepted_max_err']:.3f}m")
        n_a = metrics['n_accepted']
        print(f"  <0.5m:  {metrics['accepted_within_05m']}/{n_a}")
        print(f"  <1.0m:  {metrics['accepted_within_1m']}/{n_a}")
        print(f"  <2.0m:  {metrics['accepted_within_2m']}/{n_a}")
        print(f"  <3.0m:  {metrics['accepted_within_3m']}/{n_a}")

    if 'conformal_coverage' in metrics:
        cov = metrics['conformal_coverage']
        print(f"\nPre-calibrated conformal check:")
        print(f"  Interval: ±{conformal_interval:.2f}m")
        print(f"  Coverage: {cov:.1f}% ({metrics['conformal_covered']}/"
              f"{metrics['conformal_total']})")
        print(f"  Status:   {'✓ PASS (≥90%)' if cov >= 90 else '✗ BELOW TARGET (<90%)'}")

    if 'rejection_quality' in metrics:
        print(f"\nRejection quality:")
        rejected = [r for r in results if not r['accepted']]
        for r in rejected:
            print(f"  Seg {r['segment_id']}: err={r['err']:.2f}m — {r['reject_reason']}")
        print(f"  {metrics['rejection_would_be_bad']}/{metrics['n_rejected']} "
              f"rejected had error >1m ({metrics['rejection_quality']:.0f}%)")

    # Detection summary (full mode)
    if args.mode == 'full' and det_info:
        print(f"\nDetection summary:")
        for info in det_info:
            print(f"  {os.path.basename(info['acc_data_path'])}: "
                  f"{info['n_matched']}/{info['n_gt']} matched "
                  f"({info['n_detected']} detected)")

    # Generate output
    os.makedirs(args.output_dir, exist_ok=True)

    # Save results JSON
    results_json = []
    for r in results:
        rj = {k: v for k, v in r.items() if k != 'pos_curve'}
        # Clean non-serializable
        if 'quality_features' in rj:
            rj['quality_features'] = {
                k: round(v, 4) if isinstance(v, float) else v
                for k, v in rj['quality_features'].items()
            }
        results_json.append(rj)

    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results_json, f, indent=2)

    with open(os.path.join(args.output_dir, "summary.json"), "w") as f:
        json.dump(metrics, f, indent=2)

    # Generate figures
    print(f"\nGenerating figures...")
    generate_figures(results, metrics, conformal_interval, args.output_dir,
                     args.mode, det_info, fs=pipeline.fs)

    print(f"\nResults saved to {args.output_dir}/")
    print(f"  results.json  — Per-ride detailed results")
    print(f"  summary.json  — Aggregate metrics")
    print("Evaluation complete.")

    return results, metrics


if __name__ == "__main__":
    main()
