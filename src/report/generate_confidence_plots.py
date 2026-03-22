"""
Comprehensive visualization generator for ZUPT Confidence Interval Analysis.
Produces all plots into metadata/ci_plots/ for use in the report.
"""
import os, sys, json, numpy as np, pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import matplotlib.patches as mpatches

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))
from src.algorithms.algo2_zupt import estimate_height_zupt
from src.algorithms.zupt_confidence import ZuptConfidenceAnalyzer
from src.dataset.synthetic_work_dataset import create_dataset

BASE = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DATASET_DIR = os.path.join(BASE, 'example', 'work_dataset')
PLOT_DIR = os.path.join(BASE, 'metadata', 'ci_plots')
os.makedirs(PLOT_DIR, exist_ok=True)

# ── Color palette ──
C_ZUPT = '#2980b9'
C_GT = '#27ae60'
C_CI = '#3498db'
C_ERR = '#e74c3c'
C_REJ = '#c0392b'
C_ACC = '#27ae60'

def load_sample(sample_dir):
    df = pd.read_csv(os.path.join(sample_dir, 'accel.csv'))
    with open(os.path.join(sample_dir, 'metadata.json')) as f:
        meta = json.load(f)
    return df['time'].values, df['az'].values, meta

def get_active_window(t, az, threshold=0.05):
    dt = np.diff(t); dt = np.insert(dt, 0, 0)
    ws = 50
    az_s = np.convolve(np.abs(az - 9.81), np.ones(ws)/ws, mode='same')
    idx = np.where(az_s > threshold)[0]
    if len(idx) == 0: return 0, 0
    margin = int(1.0 / np.mean(dt[1:])) if np.mean(dt[1:]) > 0 else 100
    return max(0, idx[0]-margin), min(len(t)-1, idx[-1]+margin)

def process_all_samples(dataset_dir, analyzer, is_train=True):
    results = []
    samples = sorted(os.listdir(dataset_dir))
    for s in samples:
        sp = os.path.join(dataset_dir, s)
        if not os.path.isdir(sp): continue
        t, az, meta = load_sample(sp)
        pos = estimate_height_zupt(t, az, gravity=9.81, accel_threshold=0.05)
        h_est = pos[-1]
        si, ei = get_active_window(t, az)
        ns = ei - si
        rej, reason = analyzer.evaluate_rejection(az, si, ei, meta['phone_model'])
        margin = analyzer.get_confidence_interval(ns, meta['phone_model'])
        theo_sigma = analyzer.compute_theoretical_confidence(ns, meta['phone_model'])
        r = {'sample': s, 'h_est': h_est, 'margin': margin, 'rejected': rej,
             'reason': reason, 'phone': meta['phone_model'], 'ns': ns,
             'theo_sigma': theo_sigma, 't': t, 'az': az, 'pos': pos,
             'si': si, 'ei': ei, 'anomaly': meta.get('anomaly','unknown')}
        if is_train and 'gt_height_meters' in meta:
            r['gt'] = meta['gt_height_meters']
            r['error'] = h_est - meta['gt_height_meters']
        results.append(r)
    return results

# ─────────────────────────── INDIVIDUAL SAMPLE PLOTS ───────────────────────────
def plot_sample_detail(r, idx, prefix, out_dir):
    """Plot a single sample: accel + height + CI band."""
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8), gridspec_kw={'height_ratios': [1, 1.3]})
    t, az, pos = r['t'], r['az'], r['pos']
    
    ax1.plot(t, az - 9.81, color='#c0392b', linewidth=0.6, alpha=0.7)
    ax1.axhspan(-0.05, 0.05, color='#27ae60', alpha=0.1, label='ZUPT threshold')
    if r['si'] < r['ei']:
        ax1.axvspan(t[r['si']], t[r['ei']], color='#f39c12', alpha=0.15, label='Active window')
    ax1.set_title(f'{prefix} Sample {idx} — Vertical Acceleration (gravity removed)', fontweight='bold', fontsize=11)
    ax1.set_ylabel('Acceleration (m/s²)')
    ax1.legend(fontsize=8, loc='upper right')
    ax1.grid(True, alpha=0.3)
    
    ax2.plot(t, pos, color=C_ZUPT, linewidth=2, label=f'ZUPT Est: {r["h_est"]:.2f} m')
    if 'gt' in r:
        ax2.axhline(r['gt'], color=C_GT, linewidth=2.5, linestyle='--', label=f'Ground Truth: {r["gt"]:.2f} m')
        ax2.fill_between(t, r['h_est'] - r['margin'], r['h_est'] + r['margin'],
                         alpha=0.15, color=C_CI, label=f'90% CI: ±{r["margin"]:.2f} m')
    else:
        ax2.fill_between([t[0], t[-1]], [r['h_est'] - r['margin']]*2, [r['h_est'] + r['margin']]*2,
                         alpha=0.15, color=C_CI, label=f'90% CI: ±{r["margin"]:.2f} m')
    
    status = 'REJECTED' if r['rejected'] else 'ACCEPTED'
    color = C_REJ if r['rejected'] else C_ACC
    ax2.text(0.02, 0.95, f'Status: {status}', transform=ax2.transAxes, fontsize=10,
             fontweight='bold', color=color, va='top',
             bbox=dict(boxstyle='round,pad=0.3', facecolor='white', edgecolor=color, alpha=0.8))
    ax2.text(0.02, 0.85, f'Phone: {r["phone"]} | Anomaly: {r["anomaly"]}',
             transform=ax2.transAxes, fontsize=8, va='top', color='#555')
    
    ax2.set_title('Height Estimation with 90% Confidence Interval', fontweight='bold', fontsize=11)
    ax2.set_xlabel('Time (s)')
    ax2.set_ylabel('Height (m)')
    ax2.legend(fontsize=8, loc='lower right')
    ax2.grid(True, alpha=0.3)
    
    fig.tight_layout()
    path = os.path.join(out_dir, f'{prefix}_sample_{idx}.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

# ─────────────────────────── AGGREGATE PLOTS ───────────────────────────
def plot_error_vs_ci(train_results, out_dir):
    """Scatter: |error| vs CI margin — shows coverage."""
    valid = [r for r in train_results if not r['rejected'] and 'gt' in r]
    errors = [abs(r['error']) for r in valid]
    margins = [r['margin'] for r in valid]
    covered = [abs(e) <= m for e, m in zip(errors, margins)]
    
    fig, ax = plt.subplots(figsize=(10, 7))
    for e, m, c, r in zip(errors, margins, covered, valid):
        color = C_ACC if c else C_ERR
        marker = 'o' if c else 'x'
        ax.scatter(m, e, c=color, marker=marker, s=40, alpha=0.7)
    
    mx = max(max(errors), max(margins)) * 1.1
    ax.plot([0, mx], [0, mx], 'k--', alpha=0.4, label='|Error| = CI Margin (boundary)')
    ax.fill_between([0, mx], [0, 0], [0, mx], alpha=0.05, color=C_ACC)
    
    cov = sum(covered) / len(covered) * 100
    ax.set_title(f'Absolute Error vs 90% Confidence Margin\n(Empirical Coverage: {cov:.1f}%)',
                 fontweight='bold', fontsize=13)
    ax.set_xlabel('90% CI Margin (m)', fontsize=11)
    ax.set_ylabel('|Estimation Error| (m)', fontsize=11)
    
    acc_patch = mpatches.Patch(color=C_ACC, label=f'Within CI ({sum(covered)})')
    rej_patch = mpatches.Patch(color=C_ERR, label=f'Outside CI ({len(covered)-sum(covered)})')
    ax.legend(handles=[acc_patch, rej_patch], fontsize=10, loc='upper left')
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(out_dir, 'error_vs_ci_scatter.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path, cov

def plot_error_histogram(train_results, out_dir):
    """Histogram of normalized errors (error / CI margin)."""
    valid = [r for r in train_results if not r['rejected'] and 'gt' in r]
    normalized = [abs(r['error']) / r['margin'] if r['margin'] > 0 else 0 for r in valid]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.hist(normalized, bins=30, color=C_ZUPT, alpha=0.7, edgecolor='white')
    ax.axvline(1.0, color=C_ERR, linewidth=2, linestyle='--', label='CI Boundary (ratio=1.0)')
    within = sum(1 for n in normalized if n <= 1.0)
    ax.set_title(f'Distribution of Normalized Errors (|Error|/CI Margin)\n'
                 f'{within}/{len(normalized)} samples within CI ({within/len(normalized)*100:.1f}%)',
                 fontweight='bold', fontsize=12)
    ax.set_xlabel('|Error| / CI Margin', fontsize=11)
    ax.set_ylabel('Count', fontsize=11)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    
    path = os.path.join(out_dir, 'normalized_error_hist.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_ci_by_phone(train_results, out_dir):
    """Box plot of errors grouped by phone model."""
    valid = [r for r in train_results if not r['rejected'] and 'gt' in r]
    phones = sorted(set(r['phone'] for r in valid))
    data = {p: [abs(r['error']) for r in valid if r['phone'] == p] for p in phones}
    
    fig, ax = plt.subplots(figsize=(12, 6))
    positions = range(len(phones))
    bp = ax.boxplot([data[p] for p in phones], positions=positions, patch_artist=True, widths=0.6)
    colors = plt.cm.Set2(np.linspace(0, 1, len(phones)))
    for patch, c in zip(bp['boxes'], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.7)
    ax.set_xticks(positions)
    ax.set_xticklabels(phones, rotation=30, ha='right', fontsize=9)
    ax.set_title('Absolute Error Distribution by Phone Model', fontweight='bold', fontsize=12)
    ax.set_ylabel('|Error| (m)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    path = os.path.join(out_dir, 'error_by_phone.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_rejection_breakdown(train_results, out_dir):
    """Clear horizontal bar chart of acceptance/rejection with reason counts."""
    rejected = [r for r in train_results if r['rejected']]
    accepted = [r for r in train_results if not r['rejected']]
    
    # Group rejections by simplified reason
    reason_map = {
        'Duration too long': 'Duration too long',
        'Impact detected': 'Impact detected',
        'Stationary noise too high': 'Stationary noise',
        'Shaking detected': 'Shaking detected',
        'Invalid window': 'Invalid window',
    }
    counts = {'Accepted': len(accepted)}
    for r in rejected:
        matched = False
        for key_prefix, label in reason_map.items():
            if r['reason'].startswith(key_prefix):
                counts[label] = counts.get(label, 0) + 1
                matched = True
                break
        if not matched:
            counts['Other rejection'] = counts.get('Other rejection', 0) + 1
    
    # Sort: Accepted first, then rejection reasons by count
    labels = ['Accepted'] + sorted([k for k in counts if k != 'Accepted'], key=lambda k: -counts[k])
    values = [counts[l] for l in labels]
    colors_list = [C_ACC] + [plt.cm.Reds(0.3 + 0.6 * i / max(1, len(labels)-2)) for i in range(len(labels)-1)]
    
    fig, ax = plt.subplots(figsize=(12, max(4, len(labels) * 0.8 + 2)))
    bars = ax.barh(range(len(labels)), values, color=colors_list, edgecolor='white', height=0.6)
    
    # Add count labels on bars
    for bar, val, total in zip(bars, values, values):
        pct = val / len(train_results) * 100
        ax.text(bar.get_width() + 0.5, bar.get_y() + bar.get_height()/2,
                f'{val}  ({pct:.1f}%)', va='center', fontsize=10, fontweight='bold')
    
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=11)
    ax.set_xlabel('Number of Samples', fontsize=11)
    ax.set_title(f'Sample Acceptance / Rejection Breakdown\n'
                 f'(Total: {len(train_results)} samples, {len(rejected)} rejected)',
                 fontweight='bold', fontsize=13)
    ax.invert_yaxis()
    ax.grid(True, alpha=0.3, axis='x')
    ax.set_xlim(0, max(values) * 1.25)
    
    fig.tight_layout()
    path = os.path.join(out_dir, 'rejection_breakdown.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_conformal_calibration(train_results, analyzer, out_dir):
    """Visualize conformal calibration: theoretical vs calibrated CI."""
    valid = [r for r in train_results if not r['rejected'] and 'gt' in r]
    errors = np.array([abs(r['error']) for r in valid])
    theo_sigmas = np.array([r['theo_sigma'] for r in valid])
    
    # Theoretical CI (1.645 sigma)
    theo_ci = 1.645 * theo_sigmas
    theo_cov = np.mean(errors <= theo_ci) * 100
    
    # Calibrated CI
    cal_ci = np.array([r['margin'] for r in valid])
    cal_cov = np.mean(errors <= cal_ci) * 100
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    
    # Left: Theoretical only
    order = np.argsort(theo_ci)
    ax1.scatter(range(len(errors)), errors[order], c=[C_ACC if e <= t else C_ERR for e, t in zip(errors[order], theo_ci[order])],
                s=15, alpha=0.7)
    ax1.plot(range(len(errors)), theo_ci[order], 'k--', linewidth=1.5, label='Theoretical 90% CI')
    ax1.set_title(f'Theoretical CI (1.645σ)\nCoverage: {theo_cov:.1f}%', fontweight='bold')
    ax1.set_xlabel('Sample (sorted by CI)')
    ax1.set_ylabel('|Error| (m)')
    ax1.legend(fontsize=9)
    ax1.grid(True, alpha=0.3)
    
    # Right: Calibrated
    order2 = np.argsort(cal_ci)
    ax2.scatter(range(len(errors)), errors[order2], c=[C_ACC if e <= c else C_ERR for e, c in zip(errors[order2], cal_ci[order2])],
                s=15, alpha=0.7)
    ax2.plot(range(len(errors)), cal_ci[order2], 'k--', linewidth=1.5, label='Calibrated 90% CI')
    ax2.set_title(f'Conformal Calibrated CI\nCoverage: {cal_cov:.1f}%', fontweight='bold')
    ax2.set_xlabel('Sample (sorted by CI)')
    ax2.set_ylabel('|Error| (m)')
    ax2.legend(fontsize=9)
    ax2.grid(True, alpha=0.3)
    
    fig.suptitle('Theoretical vs Conformal-Calibrated Confidence Intervals', fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'conformal_calibration.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path, theo_cov, cal_cov

def plot_anomaly_analysis(train_results, out_dir):
    """Error distribution by anomaly type with clear accepted/rejected counts."""
    valid = [r for r in train_results if 'gt' in r]
    anomalies = sorted(set(r['anomaly'] for r in valid))
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))
    
    # Left: Accepted count vs Rejected count per anomaly (stacked bar)
    acc_counts = [sum(1 for r in valid if r['anomaly'] == a and not r['rejected']) for a in anomalies]
    rej_counts = [sum(1 for r in valid if r['anomaly'] == a and r['rejected']) for a in anomalies]
    x = np.arange(len(anomalies))
    width = 0.5
    ax1.bar(x, acc_counts, width, label='Accepted', color=C_ACC, alpha=0.7)
    ax1.bar(x, rej_counts, width, bottom=acc_counts, label='Rejected', color=C_REJ, alpha=0.7)
    # Add count labels
    for i, (ac, rc) in enumerate(zip(acc_counts, rej_counts)):
        if ac > 0: ax1.text(i, ac/2, str(ac), ha='center', va='center', fontweight='bold', color='white', fontsize=11)
        if rc > 0: ax1.text(i, ac + rc/2, str(rc), ha='center', va='center', fontweight='bold', color='white', fontsize=11)
    ax1.set_xticks(x)
    ax1.set_xticklabels(anomalies, fontsize=10)
    ax1.set_ylabel('Count', fontsize=11)
    ax1.set_title('Accept/Reject Count by Anomaly Type', fontweight='bold', fontsize=12)
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3, axis='y')
    
    # Right: Error distribution for accepted samples
    data_acc = {a: [abs(r['error']) for r in valid if r['anomaly'] == a and not r['rejected']] for a in anomalies}
    bp_data = [data_acc[a] if data_acc[a] else [0] for a in anomalies]
    bp = ax2.boxplot(bp_data, positions=x, patch_artist=True, widths=0.5)
    anom_colors = {'clean': '#27ae60', 'shaking': '#e67e22', 'impact': '#e74c3c', 'long_stationary': '#8e44ad'}
    for patch, a in zip(bp['boxes'], anomalies):
        patch.set_facecolor(anom_colors.get(a, '#3498db'))
        patch.set_alpha(0.6)
    ax2.set_xticks(x)
    ax2.set_xticklabels(anomalies, fontsize=10)
    ax2.set_ylabel('|Error| (m)', fontsize=11)
    ax2.set_title('Error Distribution of Accepted Samples by Anomaly', fontweight='bold', fontsize=12)
    ax2.grid(True, alpha=0.3, axis='y')
    
    fig.suptitle('Anomaly Type Analysis: Rejection Effectiveness and Error Impact',
                 fontweight='bold', fontsize=14, y=1.02)
    fig.tight_layout()
    path = os.path.join(out_dir, 'anomaly_analysis.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

def plot_height_parity(train_results, out_dir):
    """Parity plot: estimated height vs GT height."""
    valid = [r for r in train_results if not r['rejected'] and 'gt' in r]
    gt = [r['gt'] for r in valid]
    est = [r['h_est'] for r in valid]
    margins = [r['margin'] for r in valid]
    
    fig, ax = plt.subplots(figsize=(9, 9))
    ax.errorbar(gt, est, yerr=margins, fmt='o', color=C_ZUPT, ecolor=C_CI, elinewidth=1,
                capsize=3, markersize=4, alpha=0.7, label='ZUPT ± 90% CI')
    mn, mx = min(min(gt), min(est)) - 5, max(max(gt), max(est)) + 5
    ax.plot([mn, mx], [mn, mx], 'k--', alpha=0.4, label='Perfect prediction')
    ax.set_xlim(mn, mx); ax.set_ylim(mn, mx)
    ax.set_xlabel('Ground Truth Height (m)', fontsize=11)
    ax.set_ylabel('Estimated Height (m)', fontsize=11)
    ax.set_title('Parity Plot: Estimated vs Ground Truth Height\nwith 90% Confidence Intervals',
                 fontweight='bold', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_aspect('equal')
    
    path = os.path.join(out_dir, 'height_parity.png')
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    return path

# ─────────────────────────── MAIN ───────────────────────────
def main():
    np.random.seed(42)
    
    # 1. Generate dataset
    print("Step 1: Generating synthetic dataset...")
    create_dataset(DATASET_DIR, n_train=150, n_test=50)
    
    # 2. Process train data
    print("Step 2: Processing train data...")
    analyzer = ZuptConfidenceAnalyzer(dt=0.01)
    train_results = process_all_samples(os.path.join(DATASET_DIR, 'train'), analyzer, is_train=True)
    
    # 3. Fit conformal predictor
    print("Step 3: Fitting conformal predictor...")
    valid_train = [r for r in train_results if not r['rejected'] and 'gt' in r]
    errors = [abs(r['error']) for r in valid_train]
    theo_sigmas = [r['theo_sigma'] for r in valid_train]
    analyzer.fit_conformal(errors, theo_sigmas, alpha=0.1)
    print(f"  Calibrated multiplier: {analyzer.calibrated_multiplier:.3f}")
    
    # Re-process with calibrated analyzer
    train_results = process_all_samples(os.path.join(DATASET_DIR, 'train'), analyzer, is_train=True)
    
    # 4. Generate per-category individual sample plots
    print("Step 4: Generating per-category sample visualizations...")
    categories = ['clean', 'shaking', 'impact', 'long_stationary']
    
    # For train: pick 1 accepted + 1 rejected per category (if available)
    train_plot_paths = []
    plot_idx = 0
    for cat in categories:
        cat_samples = [r for r in train_results if r['anomaly'] == cat]
        accepted = [r for r in cat_samples if not r['rejected']]
        rejected = [r for r in cat_samples if r['rejected']]
        if accepted:
            p = plot_sample_detail(accepted[0], plot_idx, f'train_{cat}_accepted', PLOT_DIR)
            train_plot_paths.append(p)
            print(f"  Train {cat} (accepted): {accepted[0]['sample']}")
            plot_idx += 1
        if rejected:
            p = plot_sample_detail(rejected[0], plot_idx, f'train_{cat}_rejected', PLOT_DIR)
            train_plot_paths.append(p)
            print(f"  Train {cat} (REJECTED): {rejected[0]['sample']}")
            plot_idx += 1
    
    # For test: pick 1 accepted + 1 rejected per category (if available)
    test_results = process_all_samples(os.path.join(DATASET_DIR, 'test'), analyzer, is_train=False)
    test_plot_paths = []
    plot_idx = 0
    for cat in categories:
        cat_samples = [r for r in test_results if r['anomaly'] == cat]
        accepted = [r for r in cat_samples if not r['rejected']]
        rejected = [r for r in cat_samples if r['rejected']]
        if accepted:
            p = plot_sample_detail(accepted[0], plot_idx, f'test_{cat}_accepted', PLOT_DIR)
            test_plot_paths.append(p)
            print(f"  Test {cat} (accepted): {accepted[0]['sample']}")
            plot_idx += 1
        if rejected:
            p = plot_sample_detail(rejected[0], plot_idx, f'test_{cat}_rejected', PLOT_DIR)
            test_plot_paths.append(p)
            print(f"  Test {cat} (REJECTED): {rejected[0]['sample']}")
            plot_idx += 1
    
    # 5. Aggregate plots
    print("Step 5: Generating aggregate visualizations...")
    scatter_path, cov = plot_error_vs_ci(train_results, PLOT_DIR)
    print(f"  Coverage scatter: {cov:.1f}%")
    
    hist_path = plot_error_histogram(train_results, PLOT_DIR)
    phone_path = plot_ci_by_phone(train_results, PLOT_DIR)
    rej_path = plot_rejection_breakdown(train_results, PLOT_DIR)
    conf_path, theo_cov, cal_cov = plot_conformal_calibration(train_results, analyzer, PLOT_DIR)
    print(f"  Theoretical coverage: {theo_cov:.1f}%, Calibrated: {cal_cov:.1f}%")
    
    anom_path = plot_anomaly_analysis(train_results, PLOT_DIR)
    parity_path = plot_height_parity(train_results, PLOT_DIR)
    
    # 6. Save metadata for report generator
    meta = {
        'n_train': len(train_results),
        'n_test': len(test_results),
        'n_rejected_train': sum(1 for r in train_results if r['rejected']),
        'n_rejected_test': sum(1 for r in test_results if r['rejected']),
        'calibrated_multiplier': analyzer.calibrated_multiplier,
        'calibrated_margin': analyzer.calibrated_margin,
        'theoretical_coverage': theo_cov,
        'calibrated_coverage': cal_cov,
        'train_plot_paths': train_plot_paths,
        'test_plot_paths': test_plot_paths,
        'aggregate_plots': {
            'scatter': scatter_path, 'histogram': hist_path, 'phone': phone_path,
            'rejection': rej_path, 'conformal': conf_path,
            'anomaly': anom_path, 'parity': parity_path
        }
    }
    meta_path = os.path.join(PLOT_DIR, 'plot_metadata.json')
    with open(meta_path, 'w') as f:
        json.dump(meta, f, indent=2)
    
    print(f"\nAll plots saved to {PLOT_DIR}")
    print(f"Metadata saved to {meta_path}")
    return meta

if __name__ == '__main__':
    main()
