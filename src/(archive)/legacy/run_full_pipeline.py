"""
Master Pipeline v3: Gravity-Projected ZUPT with Quality Scoring.

Evaluates height estimation per individual GT ride using:
- Original ZUPT (magnitude-based)
- New Gravity-Projected ZUPT (3-axis, with rejection)
- Compares accepted-only vs all-rides metrics
"""
import os, sys, json, random
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))
from elevator_detection_algorithms import ElevatorDetector
from algorithms.algo2_zupt import estimate_height_zupt
from algorithms.algo4_gravity_zupt import estimate_height_adaptive
from algorithms.zupt_confidence import ZuptConfidenceAnalyzer

FIGS = os.path.join("docs", "figures")
os.makedirs(FIGS, exist_ok=True)


def iou_binary(p, g):
    inter = np.sum(p & g)
    union = np.sum(p | g)
    return inter / union if union > 0 else (1.0 if not np.any(g) else 0.0)


def segs_to_mask(segs, n):
    m = np.zeros(n, dtype=bool)
    for s, e in segs:
        m[int(s):int(e)] = True
    return m


def main():
    print("=" * 70)
    print("PIPELINE v3: Gravity-Projected ZUPT with Quality Scoring")
    print("=" * 70)

    # ---- Load Bar-Ilan (3-axis) ----
    acc_f = os.path.join("datasets", "bar_ilan_dataset", "sensors_synced", "ACC.csv")
    gt_f = os.path.join("datasets", "bar_ilan_dataset", "metadata.csv")
    df_acc = pd.read_csv(acc_f, names=["time_ms", "x", "y", "z"])
    df_gt = pd.read_csv(gt_f)

    t = df_acc["time_ms"].values / 1000.0
    raw_ax = df_acc["x"].values
    raw_ay = df_acc["y"].values
    raw_az = df_acc["z"].values
    acc_mag = np.sqrt(raw_ax**2 + raw_ay**2 + raw_az**2)
    fs = 1.0 / np.median(np.diff(t))
    gt_t = df_gt["time_sec"].values
    gt_h = df_gt["height_smooth"].values
    print(f"Dataset: {len(t)} samples, fs={fs:.0f}Hz")

    # ---- Extract GT ride segments ----
    gt_ids = sorted([x for x in df_gt["elevator_segment_id"].unique() if x >= 0])
    gt_rides = []
    for sid in gt_ids:
        sub = df_gt[df_gt["elevator_segment_id"] == sid]
        t_start = sub["time_sec"].iloc[0]
        t_end = sub["time_sec"].iloc[-1]
        gt_rides.append({
            "id": int(sid),
            "t_start": t_start, "t_end": t_end,
            "h_start": sub["height_smooth"].iloc[0],
            "h_end": sub["height_smooth"].iloc[-1],
            "true_dh": sub["height_smooth"].iloc[-1] - sub["height_smooth"].iloc[0],
            "s_idx": int(np.argmin(np.abs(t - t_start))),
            "e_idx": int(np.argmin(np.abs(t - t_end))),
            "phone": sub["phone_position"].iloc[0]
        })
    print(f"GT rides: {len(gt_rides)} individual segments")

    # ---- Segment detection ----
    detector = ElevatorDetector(fs=fs)
    det_segs = detector.detect_algorithm1_state_machine(t, acc_mag, var_thresh=3.5, acc_thresh=0.35)
    gt_mask = np.zeros(len(t), dtype=bool)
    for r in gt_rides:
        gt_mask[r["s_idx"]:r["e_idx"]] = True
    det_mask = segs_to_mask(det_segs, len(t))
    iou = iou_binary(det_mask, gt_mask)
    print(f"Alg1 detection: {len(det_segs)} segments, IoU={iou*100:.1f}%")

    # ---- Height estimation: compare methods ----
    PRE_WINDOW = 5.0  # seconds before ride
    POST_WINDOW = 5.0  # seconds after ride

    results_mag = []
    results_gp = []

    print(f"\n{'Ride':>4} {'True':>7} {'Mag':>7} {'ErrM':>6} {'GP':>7} {'ErrGP':>6} {'Q':>5} {'Method':>6} {'Rej':>3} {'Phone':>6}", flush=True)

    for ri, r in enumerate(gt_rides):
        print(f"  Processing ride {ri+1}/{len(gt_rides)}...", end="\r", flush=True)
        si, ei = r["s_idx"], r["e_idx"]
        ride_t = t[si:ei]
        ride_ax = raw_ax[si:ei]
        ride_ay = raw_ay[si:ei]
        ride_az = raw_az[si:ei]
        ride_mag = acc_mag[si:ei]

        # Pre-ride stationary data
        pre_start = int(np.argmin(np.abs(t - max(0, r["t_start"] - PRE_WINDOW))))
        pre_ax = raw_ax[pre_start:si]
        pre_ay = raw_ay[pre_start:si]
        pre_az = raw_az[pre_start:si]

        # Post-ride stationary data
        post_end = int(np.argmin(np.abs(t - min(t[-1], r["t_end"] + POST_WINDOW))))
        post_ax = raw_ax[ei:post_end]
        post_ay = raw_ay[ei:post_end]
        post_az = raw_az[ei:post_end]

        # Method 1: Original magnitude ZUPT
        g_est = np.mean(ride_mag)
        pos_mag = estimate_height_zupt(ride_t, ride_mag, gravity=g_est, accel_threshold=0.1)
        est_mag = pos_mag[-1]
        err_mag = abs(est_mag - r["true_dh"])

        results_mag.append({**r, "est_dh": est_mag, "err": err_mag, "pos_curve": pos_mag})

        # Method 2: Adaptive Gravity ZUPT (picks best of 3 methods)
        gp_result = estimate_height_adaptive(
            ride_t, ride_ax, ride_ay, ride_az,
            pre_ax, pre_ay, pre_az,
            post_ax, post_ay, post_az,
            fs=fs
        )
        est_gp = gp_result['height']
        err_gp = abs(est_gp - r["true_dh"])
        quality = gp_result['quality']
        rejected = gp_result['rejected']

        # Post-hoc rejection: estimates > 80m are implausible for one ride
        if abs(est_gp) > 80:
            rejected = True
            gp_result['reject_reason'] = f'Estimate implausible: {est_gp:.1f}m'

        results_gp.append({
            **r, "est_dh": est_gp, "err": err_gp,
            "pos_curve": gp_result['pos'],
            "quality": quality, "rejected": rejected,
            "reject_reason": gp_result.get('reject_reason', ''),
            "method": gp_result.get('method', 'unknown'),
            "agreement": gp_result.get('agreement', 0),
            "all_estimates": gp_result.get('all_estimates', {})
        })

        rej_str = "REJ" if rejected else ""
        method_str = gp_result.get('method', '?')[:6]
        agree_str = f"{gp_result.get('agreement', 0):.2f}"
        print(f"{r['id']:4d} {r['true_dh']:+7.1f} {est_mag:+7.2f} {err_mag:6.2f} {est_gp:+7.2f} {err_gp:6.2f} {quality:5.2f} {agree_str:>5} {method_str:>6} {rej_str:>3} {r['phone']:>6}")

    # ---- Compute metrics ----
    mag_errors = [r["err"] for r in results_mag]
    gp_errors_all = [r["err"] for r in results_gp]
    gp_accepted = [r for r in results_gp if not r["rejected"]]
    gp_rejected = [r for r in results_gp if r["rejected"]]
    gp_errors_accepted = [r["err"] for r in gp_accepted]

    print(f"\n{'='*50}")
    print(f"Magnitude ZUPT:")
    print(f"  All rides MAE: {np.mean(mag_errors):.2f}m, Median: {np.median(mag_errors):.2f}m")
    print(f"  <1m: {sum(1 for e in mag_errors if e<1)}/{len(mag_errors)}")
    print(f"  <2m: {sum(1 for e in mag_errors if e<2)}/{len(mag_errors)}")
    print(f"  <3m: {sum(1 for e in mag_errors if e<3)}/{len(mag_errors)}")

    print(f"\nGravity-Projected ZUPT:")
    print(f"  All rides MAE: {np.mean(gp_errors_all):.2f}m, Median: {np.median(gp_errors_all):.2f}m")
    print(f"  Accepted: {len(gp_accepted)}/{len(results_gp)}, Rejected: {len(gp_rejected)}")
    if gp_errors_accepted:
        print(f"  Accepted MAE: {np.mean(gp_errors_accepted):.2f}m, Median: {np.median(gp_errors_accepted):.2f}m")
        print(f"  <1m: {sum(1 for e in gp_errors_accepted if e<1)}/{len(gp_errors_accepted)}")
        print(f"  <2m: {sum(1 for e in gp_errors_accepted if e<2)}/{len(gp_errors_accepted)}")
        print(f"  <3m: {sum(1 for e in gp_errors_accepted if e<3)}/{len(gp_errors_accepted)}")

    for r in gp_rejected:
        print(f"  Rejected ride {r['id']}: {r['reject_reason']}")

    # Save results
    summary = {
        "mag_mae": float(np.mean(mag_errors)),
        "mag_median": float(np.median(mag_errors)),
        "gp_all_mae": float(np.mean(gp_errors_all)),
        "gp_accepted_mae": float(np.mean(gp_errors_accepted)) if gp_errors_accepted else None,
        "gp_accepted_median": float(np.median(gp_errors_accepted)) if gp_errors_accepted else None,
        "gp_accepted_count": len(gp_accepted),
        "gp_rejected_count": len(gp_rejected),
        "gp_under_1m": sum(1 for e in gp_errors_accepted if e < 1),
        "gp_under_2m": sum(1 for e in gp_errors_accepted if e < 2),
    }
    with open(os.path.join(FIGS, "combo_results.json"), "w") as f:
        json.dump(summary, f, indent=2)

    # ---- Conformal prediction on GP accepted ----
    random.seed(42)
    random.shuffle(gp_accepted)
    split = len(gp_accepted) // 2
    train_gp = gp_accepted[:split]
    test_gp = gp_accepted[split:]

    analyzer = ZuptConfidenceAnalyzer(dt=1.0/fs)
    train_errors = [r["err"] for r in train_gp]
    train_sigmas = [analyzer.compute_theoretical_confidence(r["e_idx"]-r["s_idx"], "Pixel") for r in train_gp]
    if train_errors:
        analyzer.fit_conformal(train_errors, train_sigmas, alpha=0.1)
    params = {"calibrated_multiplier": float(analyzer.calibrated_multiplier),
              "calibrated_margin": float(analyzer.calibrated_margin)}
    with open("conformal_params.json", "w") as f:
        json.dump(params, f, indent=2)

    test_covered = sum(1 for r in test_gp if r["err"] <= analyzer.get_confidence_interval(r["e_idx"]-r["s_idx"], "Pixel"))
    cov = test_covered / len(test_gp) * 100 if test_gp else 0
    print(f"\nConformal: mult={params['calibrated_multiplier']:.3f}, test coverage={cov:.0f}%")

    # ========== FIGURES ==========

    # FIG 1: GT Height Profile
    fig1, ax1 = plt.subplots(figsize=(14, 4))
    ax1.plot(gt_t, gt_h, 'k-', lw=1.5, label='GT Height')
    for r in gt_rides:
        ax1.axvspan(r["t_start"], r["t_end"], color='salmon', alpha=0.25)
    ax1.set_xlabel("Time (s)"); ax1.set_ylabel("Height (m)")
    ax1.set_title("Bar-Ilan: Ground Truth Height Profile (red = individual rides)")
    ax1.legend(); ax1.grid(True, alpha=0.3)
    fig1.tight_layout(); fig1.savefig(os.path.join(FIGS, "fig1_gt_height.png"), dpi=150); plt.close(fig1)

    # FIG 2: True vs Estimated scatter (Gravity-Projected, accepted only)
    fig2, axes2 = plt.subplots(1, 2, figsize=(14, 6))
    
    # Magnitude
    ax = axes2[0]
    trues = [r["true_dh"] for r in results_mag]
    ests = [r["est_dh"] for r in results_mag]
    ax.scatter(trues, ests, c='red', alpha=0.7, s=50)
    lim = max(abs(min(min(trues), min(ests)))+5, abs(max(max(trues), max(ests)))+5)
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    ax.set_xlabel("True dh (m)"); ax.set_ylabel("Estimated dh (m)")
    ax.set_title(f"Magnitude ZUPT (MAE={np.mean(mag_errors):.2f}m)")
    ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    
    # Gravity-projected (accepted)
    ax = axes2[1]
    trues_a = [r["true_dh"] for r in gp_accepted]
    ests_a = [r["est_dh"] for r in gp_accepted]
    ax.scatter(trues_a, ests_a, c='green', alpha=0.7, s=50, label='Accepted')
    trues_r = [r["true_dh"] for r in gp_rejected]
    ests_r = [r["est_dh"] for r in gp_rejected]
    if trues_r:
        ax.scatter(trues_r, ests_r, c='red', alpha=0.5, s=50, marker='x', label='Rejected')
    ax.plot([-lim, lim], [-lim, lim], 'k--', alpha=0.5)
    ax.set_xlabel("True dh (m)"); ax.set_ylabel("Estimated dh (m)")
    mae_a = np.mean(gp_errors_accepted) if gp_errors_accepted else 0
    ax.set_title(f"Gravity ZUPT Accepted (MAE={mae_a:.2f}m, {len(gp_accepted)}/{len(results_gp)})")
    ax.legend(); ax.grid(True, alpha=0.3); ax.set_aspect('equal')
    ax.set_xlim(-lim, lim); ax.set_ylim(-lim, lim)
    
    fig2.tight_layout(); fig2.savefig(os.path.join(FIGS, "fig2_scatter_comparison.png"), dpi=150); plt.close(fig2)

    # FIG 3: Per-ride error comparison bars
    fig3, ax3 = plt.subplots(figsize=(14, 5))
    x = range(len(gt_rides))
    ax3.bar([i-0.2 for i in x], mag_errors, width=0.4, color='red', alpha=0.6, label='Magnitude ZUPT')
    gp_bar_colors = ['green' if not r["rejected"] else 'gray' for r in results_gp]
    ax3.bar([i+0.2 for i in x], [min(e, 30) for e in gp_errors_all], width=0.4, color=gp_bar_colors, alpha=0.6, label='Gravity ZUPT')
    ax3.axhline(1.0, color='blue', ls='--', alpha=0.5, label='1m target')
    ax3.axhline(3.0, color='orange', ls='--', alpha=0.5, label='1-floor (3m)')
    ax3.set_xlabel("Ride Index"); ax3.set_ylabel("Abs Error (m)")
    ax3.set_title("Per-Ride Error Comparison (gray = rejected by quality filter)")
    ax3.legend(); ax3.grid(axis='y', alpha=0.3)
    ax3.set_ylim(0, 30)
    fig3.tight_layout(); fig3.savefig(os.path.join(FIGS, "fig3_per_ride_errors.png"), dpi=150); plt.close(fig3)

    # FIG 4: Gravity-Projected overlay on GT (accepted only)
    fig4, ax4 = plt.subplots(figsize=(14, 5))
    ax4.plot(gt_t, gt_h, 'k--', lw=1, alpha=0.5, label='Ground Truth')
    for d in gp_accepted:
        si, ei = d["s_idx"], d["e_idx"]
        ax4.plot(t[si:ei], d["h_start"] + d["pos_curve"], 'g-', lw=1.5, alpha=0.8)
    for d in gp_rejected:
        si, ei = d["s_idx"], d["e_idx"]
        ax4.plot(t[si:ei], d["h_start"] + d["pos_curve"], 'r-', lw=0.8, alpha=0.3)
    ax4.set_xlabel("Time (s)"); ax4.set_ylabel("Height (m)")
    ax4.set_title("Gravity-Projected ZUPT: Per-Ride Estimates vs GT (green=accepted, red=rejected)")
    custom = [Line2D([0],[0],color='k',ls='--'), Line2D([0],[0],color='g',lw=2), Line2D([0],[0],color='r',lw=1,alpha=0.3)]
    ax4.legend(custom, ['GT', 'Accepted', 'Rejected']); ax4.grid(True, alpha=0.3)
    fig4.tight_layout(); fig4.savefig(os.path.join(FIGS, "fig4_gp_overlay.png"), dpi=150); plt.close(fig4)

    # FIG 5: Quality score vs error
    fig5, ax5 = plt.subplots(figsize=(8, 6))
    qual = [r["quality"] for r in results_gp if not r["rejected"]]
    errs = [r["err"] for r in results_gp if not r["rejected"]]
    ax5.scatter(qual, errs, c='green', alpha=0.7, s=50, label='Accepted')
    qual_r = [r["quality"] for r in results_gp if r["rejected"]]
    errs_r = [r["err"] for r in results_gp if r["rejected"]]
    if qual_r:
        ax5.scatter(qual_r, errs_r, c='red', alpha=0.5, s=50, marker='x', label='Rejected')
    ax5.axhline(1.0, color='blue', ls='--', alpha=0.5)
    ax5.set_xlabel("Quality Score (lower = better)"); ax5.set_ylabel("Abs Error (m)")
    ax5.set_title("Quality Score vs Height Error")
    ax5.legend(); ax5.grid(True, alpha=0.3)
    fig5.tight_layout(); fig5.savefig(os.path.join(FIGS, "fig5_quality_vs_error.png"), dpi=150); plt.close(fig5)

    # FIG 6: Conformal coverage
    fig6, ax6 = plt.subplots(figsize=(10, 5))
    if test_gp:
        x6 = range(len(test_gp))
        test_ests = [r["est_dh"] for r in test_gp]
        test_trues = [r["true_dh"] for r in test_gp]
        test_cis = [analyzer.get_confidence_interval(r["e_idx"]-r["s_idx"], "Pixel") for r in test_gp]
        ax6.errorbar(x6, test_ests, yerr=test_cis, fmt='o', color='blue', ecolor='lightblue', capsize=3, label='Est +/- 90% CI')
        ax6.scatter(x6, test_trues, color='red', marker='x', s=50, zorder=5, label='True dh')
    ax6.set_xlabel("Test Ride Index"); ax6.set_ylabel("Height Change (m)")
    ax6.set_title(f"Conformal: {cov:.0f}% Coverage on Test Rides (Accepted Only)")
    ax6.legend(); ax6.grid(True, alpha=0.3)
    fig6.tight_layout(); fig6.savefig(os.path.join(FIGS, "fig6_conformal.png"), dpi=150); plt.close(fig6)

    # FIG 7: Per-ride detailed examples (4 best, 2 worst accepted)
    gp_sorted = sorted(gp_accepted, key=lambda r: r["err"])
    examples = gp_sorted[:4] + gp_sorted[-2:]  # 4 best + 2 worst accepted
    fig7, axes7 = plt.subplots(len(examples), 1, figsize=(12, 3*len(examples)))
    for idx, d in enumerate(examples):
        ax = axes7[idx]
        si, ei = d["s_idx"], d["e_idx"]
        ride_t = t[si:ei] - t[si]
        ax.plot(ride_t, d["h_start"] + d["pos_curve"], 'b-', lw=1.5, label='Estimated')
        # GT line
        gt_line = np.linspace(d["h_start"], d["h_end"], len(ride_t))
        ax.plot(ride_t, gt_line, 'r--', lw=1, label='GT (linear)')
        ax.set_ylabel("Height (m)")
        ax.set_title(f"Ride {d['id']}: True={d['true_dh']:+.1f}m, Est={d['est_dh']:+.2f}m, Err={d['err']:.2f}m [{d['phone']}]")
        ax.legend(loc='upper right'); ax.grid(True, alpha=0.3)
    axes7[-1].set_xlabel("Time within ride (s)")
    fig7.tight_layout(); fig7.savefig(os.path.join(FIGS, "fig7_examples.png"), dpi=150); plt.close(fig7)

    # FIG 8: Phone position
    fig8, ax8 = plt.subplots(figsize=(14, 2))
    pp = df_gt["phone_position"].values
    pp_bin = np.array([1 if p=="pocket" else 0 for p in pp])
    ax8.fill_between(gt_t, pp_bin, step='mid', color='purple', alpha=0.4, label='Pocket')
    ax8.fill_between(gt_t, 1-pp_bin, step='mid', color='orange', alpha=0.4, label='Hand')
    ax8.set_yticks([0,1]); ax8.set_yticklabels(['Hand','Pocket'])
    ax8.set_xlabel("Time (s)"); ax8.set_title("Phone Position Over Time")
    ax8.legend(loc='upper right')
    fig8.tight_layout(); fig8.savefig(os.path.join(FIGS, "fig8_phone_position.png"), dpi=150); plt.close(fig8)

    print(f"\n8 figures saved to {FIGS}/")
    print("Pipeline v3 complete.")

if __name__ == "__main__":
    main()
