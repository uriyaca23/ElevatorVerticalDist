"""
Generate new/fixed figures for the research report:
  - fig06 FIXED: Detection timeline with actual detected segments from pipeline
  - fig24 NEW:   Segmentation accuracy (GT vs detected boundaries)
  - fig25 NEW:   Per-ride analysis (Bar-Ilan) — acceleration + displacement curves  
  - fig26 NEW:   Per-ride analysis (ADVIO) — acceleration + displacement curves
"""
import os, sys, json, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
from scipy.signal import butter, filtfilt
warnings.filterwarnings('ignore')

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))
from pipeline import (
    detect_elevator_rides, estimate_height_robust, _zupt_integrate,
    ElevatorHeightPipeline,
)
from algorithms.quality_filter import (
    estimate_gravity_vector, angle_between_vectors,
    assess_segment_quality, compute_ride_gravity_drift,
)

BASE = os.path.join(os.path.dirname(__file__), "..")
OUTDIR = os.path.join(BASE, "docs", "report_figures")
os.makedirs(OUTDIR, exist_ok=True)

# ---- Load Bar-Ilan data ----
acc_df = pd.read_csv(os.path.join(BASE, "datasets", "bar_ilan_dataset",
                                   "sensors_synced", "ACC.csv"),
                      header=None, names=['time_ms','x','y','z'])
acc_df['time_sec'] = (acc_df['time_ms'] - acc_df['time_ms'].iloc[0]) / 1000.0
fs = 100
t_uniform = np.arange(0, acc_df['time_sec'].iloc[-1], 1.0/fs)
AX = np.interp(t_uniform, acc_df['time_sec'].values, acc_df['x'].values)
AY = np.interp(t_uniform, acc_df['time_sec'].values, acc_df['y'].values)
AZ = np.interp(t_uniform, acc_df['time_sec'].values, acc_df['z'].values)

meta = pd.read_csv(os.path.join(BASE, "datasets", "bar_ilan_dataset",
                                 "metadata_calibrated.csv"))
v4 = json.load(open(os.path.join(BASE, "docs", "figures_v4", "v4_results.json")))

plt.rcParams.update({
    'figure.dpi': 150, 'font.size': 10,
    'axes.titlesize': 12, 'axes.labelsize': 10,
})

def savefig(fig, name):
    path = os.path.join(OUTDIR, name)
    fig.savefig(path, bbox_inches='tight', dpi=150)
    plt.close(fig)
    print(f"  Saved: {name}")


# =====================================================================
# 1. Run detection to get actual detected segments
# =====================================================================
print("Running detection pipeline on Bar-Ilan data...")
det_rides = detect_elevator_rides(t_uniform, AX, AY, AZ, fs=fs)
print(f"  Detected {len(det_rides)} segments")

# Extract GT ride segments from metadata
t_meta = meta['time_sec'].values
h_meta = meta['height_smooth'].values
elev_flag = meta['in_elevator'].values.astype(str) == 'True'
seg_ids = meta['elevator_segment_id'].values

gt_rides = []
unique_segs = sorted(set(seg_ids[elev_flag & (seg_ids >= 0)]))
for sid in unique_segs:
    mask = (seg_ids == sid) & elev_flag
    if np.any(mask):
        idxs = np.where(mask)[0]
        t_start = t_meta[idxs[0]]
        t_end = t_meta[idxs[-1]]
        h_start = h_meta[idxs[0]]
        h_end = h_meta[idxs[-1]]
        gt_rides.append({
            'id': int(sid),
            't_start': t_start, 't_end': t_end,
            's_idx': int(t_start * fs), 'e_idx': int(t_end * fs),
            'height_diff': h_end - h_start,
        })
print(f"  Found {len(gt_rides)} GT rides")


# =====================================================================
# FIGURE 6 (FIXED): Detection Timeline with actual detected segments
# =====================================================================
def fig_detection_timeline_fixed():
    fig, axes = plt.subplots(3, 1, figsize=(18, 11), sharex=True)
    
    # Panel 1: Magnitude
    mag = np.sqrt(AX**2 + AY**2 + AZ**2)
    axes[0].plot(t_uniform, mag, linewidth=0.15, color='#7f8c8d', alpha=0.7)
    axes[0].set_ylabel('|a| (m/s²)')
    axes[0].set_title('Accelerometer Magnitude', fontsize=13)
    axes[0].set_ylim(5, 18)
    
    # Shade detected segments on magnitude too
    for ride in det_rides:
        ts = ride['s_idx'] / fs
        te = ride['e_idx'] / fs
        axes[0].axvspan(ts, te, alpha=0.15, color='#3498db')
    
    # Panel 2: Rolling variance
    var_win = int(fs * 1.5)
    rolling_var = pd.Series(mag).rolling(window=var_win, center=True, min_periods=1).var().values
    axes[1].plot(t_uniform, rolling_var, linewidth=0.3, color='#e67e22')
    axes[1].axhline(1.5, color='red', linestyle='--', alpha=0.5, label='Threshold=1.5')
    axes[1].set_ylabel('Variance')
    axes[1].set_title('Rolling Magnitude Variance', fontsize=13)
    axes[1].set_ylim(0, 10)
    axes[1].legend(fontsize=9, loc='upper right')
    
    # Panel 3: GT height with detection overlay
    axes[2].plot(t_meta, h_meta, linewidth=1.8, color='#2c3e50', zorder=3)
    
    # GT elevator shading (green)
    for gt in gt_rides:
        axes[2].axvspan(gt['t_start'], gt['t_end'], alpha=0.2, color='#27ae60', zorder=1)
    
    # Detected segment shading (blue) — THIS WAS MISSING BEFORE
    for ride in det_rides:
        ts = ride['s_idx'] / fs
        te = ride['e_idx'] / fs
        axes[2].axvspan(ts, te, alpha=0.15, color='#3498db', zorder=2)
    
    axes[2].set_ylabel('Height (m)')
    axes[2].set_xlabel('Time (s)')
    axes[2].set_title('Ground Truth Height with Detection Overlay', fontsize=13)
    axes[2].legend(handles=[
        Line2D([0],[0], color='#2c3e50', lw=2, label='GT Height'),
        Line2D([0],[0], color='#27ae60', lw=8, alpha=0.3, label='GT Elevator'),
        Line2D([0],[0], color='#3498db', lw=8, alpha=0.2, label='Detected'),
    ], fontsize=9, loc='upper left')
    
    fig.suptitle('Bar-Ilan Dataset: Detection & Segmentation Overview', fontsize=14, y=1.01)
    fig.tight_layout()
    savefig(fig, 'fig06_detection_timeline.png')


# =====================================================================
# FIGURE 24 (NEW): Segmentation Accuracy — GT vs Detected Boundaries
# =====================================================================
def fig_segmentation_accuracy():
    # Match GT to detected rides by IoU
    n = len(t_uniform)
    matches = []
    used_det = set()
    
    for gi, gt in enumerate(gt_rides):
        gt_mask = np.zeros(n, dtype=bool)
        gt_s = max(0, gt['s_idx'])
        gt_e = min(n, gt['e_idx'])
        gt_mask[gt_s:gt_e] = True
        
        best_iou = 0
        best_di = -1
        for di, det in enumerate(det_rides):
            if di in used_det:
                continue
            det_mask = np.zeros(n, dtype=bool)
            det_s = max(0, det['s_idx'])
            det_e = min(n, det['e_idx'])
            det_mask[det_s:det_e] = True
            
            inter = np.sum(gt_mask & det_mask)
            union = np.sum(gt_mask | det_mask)
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_di = di
        
        if best_iou > 0.1 and best_di >= 0:
            matches.append((gi, best_di, best_iou))
            used_det.add(best_di)
    
    print(f"  Matched {len(matches)}/{len(gt_rides)} GT rides")
    
    fig, axes = plt.subplots(2, 2, figsize=(16, 10))
    
    # Panel 1: Timeline with GT and detected boundaries marked
    ax = axes[0, 0]
    ax.set_xlim(0, t_uniform[-1])
    ax.set_ylim(-1, len(gt_rides) + 1)
    
    for i, gt in enumerate(gt_rides):
        # GT bar
        ax.barh(i, gt['t_end'] - gt['t_start'], left=gt['t_start'],
                height=0.4, color='#27ae60', alpha=0.7, edgecolor='white')
    
    for gi, di, iou in matches:
        gt = gt_rides[gi]
        det = det_rides[di]
        ts_det = det['s_idx'] / fs
        te_det = det['e_idx'] / fs
        ax.barh(gi + 0.4, te_det - ts_det, left=ts_det,
                height=0.4, color='#3498db', alpha=0.7, edgecolor='white')
    
    ax.set_ylabel('Ride Index')
    ax.set_xlabel('Time (s)')
    ax.set_title('GT (green) vs Detected (blue) Segment Boundaries')
    ax.legend(handles=[
        Line2D([0],[0], color='#27ae60', lw=6, alpha=0.7, label='GT'),
        Line2D([0],[0], color='#3498db', lw=6, alpha=0.7, label='Detected'),
    ], fontsize=9)
    
    # Panel 2: IoU distribution 
    ax2 = axes[0, 1]
    ious = [iou for _, _, iou in matches]
    ax2.hist(ious, bins=15, color='#3498db', edgecolor='white', alpha=0.8)
    ax2.axvline(0.3, color='red', linestyle='--', label=f'Threshold=0.3')
    ax2.axvline(np.mean(ious), color='orange', linestyle='--',
                label=f'Mean IoU={np.mean(ious):.2f}')
    ax2.set_xlabel('IoU')
    ax2.set_ylabel('Count')
    ax2.set_title(f'IoU Distribution ({len(matches)} matched rides)')
    ax2.legend(fontsize=9)
    
    # Panel 3: Start time error
    ax3 = axes[1, 0]
    start_errors = []
    end_errors = []
    for gi, di, iou in matches:
        gt = gt_rides[gi]
        det = det_rides[di]
        start_err = (det['s_idx'] / fs) - gt['t_start']
        end_err = (det['e_idx'] / fs) - gt['t_end']
        start_errors.append(start_err)
        end_errors.append(end_err)
    
    x = range(len(start_errors))
    ax3.bar(x, start_errors, width=0.4, color='#e74c3c', alpha=0.7, label='Start error')
    ax3.bar([xi + 0.4 for xi in x], end_errors, width=0.4, color='#3498db', alpha=0.7, label='End error')
    ax3.axhline(0, color='black', linewidth=0.5)
    ax3.set_xlabel('Matched Ride Index')
    ax3.set_ylabel('Boundary Error (s)')
    ax3.set_title('Segmentation Boundary Errors')
    ax3.legend(fontsize=9)
    
    # Panel 4: Duration comparison
    ax4 = axes[1, 1]
    gt_durs = []
    det_durs = []
    for gi, di, iou in matches:
        gt = gt_rides[gi]
        det = det_rides[di]
        gt_durs.append(gt['t_end'] - gt['t_start'])
        det_durs.append((det['e_idx'] - det['s_idx']) / fs)
    
    ax4.scatter(gt_durs, det_durs, c='#3498db', s=60, edgecolors='white', zorder=5)
    lims = [0, max(max(gt_durs), max(det_durs)) * 1.1]
    ax4.plot(lims, lims, 'k--', alpha=0.3, label='Perfect')
    ax4.set_xlabel('GT Duration (s)')
    ax4.set_ylabel('Detected Duration (s)')
    ax4.set_title('GT vs Detected Ride Duration')
    ax4.set_aspect('equal')
    ax4.legend(fontsize=9)
    
    fig.suptitle('Segmentation Accuracy Analysis', fontsize=14, y=1.02)
    fig.tight_layout()
    savefig(fig, 'fig24_segmentation_accuracy.png')
    
    return matches


# =====================================================================
# FIGURE 25 (NEW): Per-Ride Analysis — Bar-Ilan (6 examples)
# =====================================================================
def fig_bar_ilan_ride_analysis(matches):
    """
    Like the ADVIO graph: top panel = gravity-removed acceleration,
    bottom panel = displacement curves vs GT.
    """
    # Pick 6 interesting rides: 2 good hand, 2 good pocket, 1 drift-corrected, 1 rejected
    target_ids = []
    for r in v4['per_ride']:
        if r['accepted'] and r['err'] < 0.5 and r['phone'] == 'hand' and len(target_ids) < 1:
            target_ids.append(r['id'])
        elif r['accepted'] and r['err'] < 0.5 and r['phone'] == 'pocket' and len(target_ids) < 2:
            target_ids.append(r['id'])
        elif r['accepted'] and r['method'] == 'drift_corrected_mag' and len(target_ids) < 3:
            target_ids.append(r['id'])
    
    # Ensure 6 diverse rides
    for r in v4['per_ride']:
        if r['id'] not in target_ids and r['accepted'] and len(target_ids) < 4:
            target_ids.append(r['id'])
        elif r['id'] not in target_ids and not r['accepted'] and len(target_ids) < 5:
            target_ids.append(r['id'])
    for r in v4['per_ride']:
        if r['id'] not in target_ids and abs(r['true_dh']) > 30 and len(target_ids) < 6:
            target_ids.append(r['id'])
    # Fill up to 6
    for r in v4['per_ride']:
        if len(target_ids) >= 6:
            break
        if r['id'] not in target_ids:
            target_ids.append(r['id'])
    
    target_ids = target_ids[:6]
    print(f"  Selected Bar-Ilan rides for analysis: {target_ids}")
    
    fig, axes = plt.subplots(6, 2, figsize=(16, 30))
    
    pre_win = int(fs * 5)
    post_win = int(fs * 5)
    
    for row, rid in enumerate(target_ids):
        r_info = next(r for r in v4['per_ride'] if r['id'] == rid)
        gt = next((g for g in gt_rides if g['id'] == rid), None)
        if gt is None:
            continue
        
        # Find matching detected segment
        gt_s = gt['s_idx']
        gt_e = gt['e_idx']
        
        # Get the data for this ride (use detected or GT boundaries)
        si = max(0, gt_s - pre_win)
        ei = min(len(AX), gt_e + post_win)
        
        ride_ax = AX[gt_s:gt_e]
        ride_ay = AY[gt_s:gt_e]
        ride_az = AZ[gt_s:gt_e]
        ride_t = t_uniform[gt_s:gt_e]
        
        pre_ax = AX[si:gt_s]
        pre_ay = AY[si:gt_s]
        pre_az = AZ[si:gt_s]
        post_ax = AX[gt_e:ei]
        post_ay = AY[gt_e:ei]
        post_az = AZ[gt_e:ei]
        
        # Compute linear acceleration (gravity-removed magnitude)
        mag = np.sqrt(ride_ax**2 + ride_ay**2 + ride_az**2)
        a_lin = mag - np.mean(mag)
        
        # Run estimation
        est = estimate_height_robust(ride_t, ride_ax, ride_ay, ride_az,
                                     pre_ax, pre_ay, pre_az,
                                     post_ax, post_ay, post_az, fs=fs)
        
        # Also compute magnitude-only estimate
        pos_mag, _ = _zupt_integrate(a_lin, 1.0/fs, fs)
        
        true_dh = r_info['true_dh']
        status = '✓ Accepted' if r_info['accepted'] else '✗ Rejected'
        
        # Left panel: acceleration
        ax_accel = axes[row, 0]
        ax_accel.plot(ride_t, a_lin, linewidth=0.6, color='#e74c3c', alpha=0.8)
        ax_accel.axhline(0, color='gray', linewidth=0.5)
        ax_accel.set_ylabel('m/s²')
        ax_accel.set_title(f'Ride {rid} ({r_info["phone"]}) — Acceleration (gravity-removed)',
                          fontsize=10)
        
        # Right panel: displacement curves
        ax_disp = axes[row, 1]
        
        # Pipeline estimate position curve
        pos_est = est['pos']
        ax_disp.plot(ride_t, pos_est, linewidth=2, color='#3498db',
                     label=f'{est["method"]} ({est["height"]:+.2f}m)')
        
        # Magnitude position curve
        ax_disp.plot(ride_t, pos_mag, linewidth=1.5, color='#e67e22', linestyle='--',
                     label=f'Mag ZUPT ({pos_mag[-1]:+.2f}m)')
        
        # GT
        ax_disp.axhline(true_dh, color='#27ae60', linewidth=2.5,
                        label=f'GT ({true_dh:+.1f}m)')
        
        ax_disp.set_ylabel('Height (m)')
        ax_disp.set_title(f'Ride {rid} — {status} — Error={r_info["err"]:.2f}m',
                          fontsize=10)
        ax_disp.legend(fontsize=8, loc='best')
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('Bar-Ilan Dataset: Per-Ride Acceleration & Displacement Analysis',
                 fontsize=15, y=1.005)
    fig.tight_layout()
    savefig(fig, 'fig25_bar_ilan_ride_analysis.png')


# =====================================================================
# FIGURE 26 (NEW): Per-Ride Analysis — ADVIO (all 7 segments)
# =====================================================================
def fig_advio_ride_analysis():
    """
    For each ADVIO elevator segment, show acceleration + displacement curves
    with GT and all estimation methods.
    """
    advio_meta = pd.read_csv(os.path.join(BASE, "metadata", "evaluation_results.csv"))
    
    sequences = ['advio-07', 'advio-14', 'advio-18']
    segment_defs = {
        'advio-07': [(17.0, 26.0, 5.60), (36.5, 41.5, 4.53), (50.5, 57.5, 4.52),
                     (66.0, 73.5, 4.52), (81.5, 86.5, 4.46)],
        'advio-14': [(18.5, 35.5, 7.52)],
        'advio-18': [(73.5, 83.0, 7.81)],
    }
    
    total_segs = sum(len(v) for v in segment_defs.values())
    fig, axes = plt.subplots(total_segs, 2, figsize=(16, total_segs * 4.5))
    if total_segs == 1:
        axes = axes.reshape(1, 2)
    
    row = 0
    for seq in sequences:
        # Load ADVIO accelerometer
        acc_path = os.path.join(BASE, "datasets", "ADVIO", seq, "iphone", "accelerometer.csv")
        if not os.path.exists(acc_path):
            print(f"  WARNING: {acc_path} not found, skipping")
            continue
        
        adf = pd.read_csv(acc_path, header=None, names=['time','x','y','z'])
        # Resample to 100Hz
        t0 = adf['time'].iloc[0]
        adf['t'] = adf['time'] - t0
        t_advio = np.arange(0, adf['t'].iloc[-1], 1.0/fs)
        ax_advio = np.interp(t_advio, adf['t'].values, adf['x'].values)
        ay_advio = np.interp(t_advio, adf['t'].values, adf['y'].values)
        az_advio = np.interp(t_advio, adf['t'].values, adf['z'].values)
        
        for seg_i, (t_start, t_end, gt_h) in enumerate(segment_defs[seq]):
            si = max(0, int((t_start - 2) * fs))
            ei = min(len(ax_advio), int((t_end + 2) * fs))
            pre_s = max(0, int(t_start * fs) - int(5 * fs))
            pre_e = int(t_start * fs)
            post_s = int(t_end * fs)
            post_e = min(len(ax_advio), int(t_end * fs) + int(5 * fs))
            
            ride_si = int(t_start * fs)
            ride_ei = int(t_end * fs)
            
            r_ax = ax_advio[ride_si:ride_ei]
            r_ay = ay_advio[ride_si:ride_ei]
            r_az = az_advio[ride_si:ride_ei]
            r_t = t_advio[ride_si:ride_ei]
            
            pre_ax_ = ax_advio[pre_s:pre_e]
            pre_ay_ = ay_advio[pre_s:pre_e]
            pre_az_ = az_advio[pre_s:pre_e]
            post_ax_ = ax_advio[post_s:post_e]
            post_ay_ = ay_advio[post_s:post_e]
            post_az_ = az_advio[post_s:post_e]
            
            # Acceleration
            r_mag = np.sqrt(r_ax**2 + r_ay**2 + r_az**2)
            a_lin = r_mag - np.mean(r_mag)
            
            # Run pipeline estimation
            est = estimate_height_robust(r_t, r_ax, r_ay, r_az,
                                         pre_ax_, pre_ay_, pre_az_,
                                         post_ax_, post_ay_, post_az_, fs=fs)
            
            # ZUPT magnitude position
            pos_mag, _ = _zupt_integrate(a_lin, 1.0/fs, fs)
            
            # Also compute simple direct integration and Kalman for comparison
            # Direct integration
            dt = 1.0/fs
            pos_direct = np.cumsum(np.cumsum(a_lin) * dt) * dt
            
            # Left panel: acceleration
            ax_a = axes[row, 0]
            ax_a.plot(r_t, a_lin, linewidth=0.7, color='#e74c3c', alpha=0.8)
            ax_a.axhline(0, color='gray', linewidth=0.5)
            ax_a.set_ylabel('m/s²')
            ax_a.set_title(f'{seq} Segment {seg_i} — Acceleration (gravity-removed magnitude)',
                          fontsize=10)
            
            # Right panel: displacement
            ax_d = axes[row, 1]
            
            # GT horizontal line
            ax_d.axhline(gt_h, color='#27ae60', linewidth=2.5,
                        label=f'Ground Truth ({gt_h:.2f}m)')
            
            # Pipeline estimate
            ax_d.plot(r_t, est['pos'], linewidth=2, color='#3498db',
                     label=f'V4 Pipeline ({est["height"]:+.2f}m)')
            
            # Direct integration
            ax_d.plot(r_t, pos_direct, linewidth=1.2, color='black', linestyle=':',
                     label=f'Direct Int. ({pos_direct[-1]:+.2f}m)')
            
            # ZUPT magnitude
            ax_d.plot(r_t, pos_mag, linewidth=1.2, color='#e67e22', linestyle='--',
                     label=f'ZUPT Mag ({pos_mag[-1]:+.2f}m)')
            
            # Get barometer from ADVIO results if available
            baro_row = advio_meta[(advio_meta['dataset'] == seq) & (advio_meta['segment'] == seg_i)]
            if len(baro_row) > 0:
                baro_h = baro_row.iloc[0]['Barometer']
                ax_d.axhline(baro_h, color='#9b59b6', linewidth=1.5, linestyle='--',
                            label=f'Barometer ({baro_h:.2f}m)')
            
            ax_d.set_ylabel('Height (m)')
            ax_d.set_title(f'{seq} Segment {seg_i} — Vertical Displacement vs Ground Truth',
                          fontsize=10)
            ax_d.legend(fontsize=7, loc='lower right')
            
            row += 1
    
    axes[-1, 0].set_xlabel('Time (s)')
    axes[-1, 1].set_xlabel('Time (s)')
    
    fig.suptitle('ADVIO Dataset: Per-Segment Acceleration & Displacement Analysis',
                 fontsize=15, y=1.005)
    fig.tight_layout()
    savefig(fig, 'fig26_advio_ride_analysis.png')


# =====================================================================
# Run all
# =====================================================================
if __name__ == "__main__":
    print("Generating new/fixed report figures...")
    fig_detection_timeline_fixed()
    matches = fig_segmentation_accuracy()
    fig_bar_ilan_ride_analysis(matches)
    fig_advio_ride_analysis()
    print(f"\nDone! Check {OUTDIR}")
