import os
import sys
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import cv2

sys.path.append(r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist")
from src.algorithms.algo1_direct import estimate_height_direct
from src.algorithms.algo2_zupt import estimate_height_zupt
from src.algorithms.algo3_kalman import estimate_height_kalman

BASE_DIR = r"c:\Users\uriya\PycharmProjects\ElevatorVerticalDist"
ADVIO_DIR = os.path.join(BASE_DIR, "ADVIO")
META_JSON = os.path.join(BASE_DIR, "metadata", "elevator_segments.json")
VID_OUT_DIR = os.path.join(BASE_DIR, "metadata", "videos")

def generate_dataset_video(ds, runs):
    print(f"\n==============================================")
    print(f"Generating Multi-Pane Video for {ds}...")
    print(f"==============================================")
    accel_path = os.path.join(ADVIO_DIR, ds, "iphone", "accelerometer.csv")
    baro_path = os.path.join(ADVIO_DIR, ds, "iphone", "barometer.csv")
    video_path = os.path.join(ADVIO_DIR, ds, "iphone", "frames.mov")
    pose_path = os.path.join(ADVIO_DIR, ds, "ground-truth", "pose.csv")
    out_video_path = os.path.join(VID_OUT_DIR, f"{ds}.mp4")
    
    if not os.path.exists(video_path):
        print(f"Video {video_path} not found. Skipping.")
        return
        
    accel_df = pd.read_csv(accel_path, header=None)
    t_acc = accel_df[0].values
    a_mag = np.sqrt(accel_df[1].values**2 + accel_df[2].values**2 + accel_df[3].values**2)
    
    baro_df = pd.read_csv(baro_path, header=None)
    tb = baro_df[0].values
    hab = baro_df[2].values if baro_df.shape[1] > 2 else np.zeros_like(tb)

    pose_df = pd.read_csv(pose_path, header=None)
    tp = pose_df[0].values
    xp = pose_df[1].values
    yp = pose_df[3].values # Horizontal Z to Y
    zp = pose_df[2].values # Vertical Y to Z
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Failed to open video for {ds}")
        return
        
    fps = cap.get(cv2.CAP_PROP_FPS)
    target_fps = 30.0 # Standardize to 30fps
    frame_skip = max(1, int(round(fps / target_fps))) if fps > 0 else 1
    
    writer = None
    
    # ── INITIALIZE GLOBAL 3D FIGURE FOR THE DATASET ──
    fig3d = plt.figure(figsize=(5, 5), dpi=120)
    ax3d = fig3d.add_subplot(111, projection='3d')
    # Plot the entire building path in light grey
    ax3d.plot(xp, yp, zp, color='silver', alpha=0.3, linewidth=1.0, label='Full Path')
    ax3d.set_xlabel('X (m)')
    ax3d.set_ylabel('Y (m)')
    ax3d.set_zlabel('Height Z (m)')
    ax3d.set_title(f"3D Live Localization ({ds})", fontweight='bold')
    ax3d.view_init(elev=25, azim=-40)
    
    # Add persistent empty placeholders for dynamics
    highlight_line, = ax3d.plot([], [], [], color='#e74c3c', linewidth=3)
    current_dot, = ax3d.plot([], [], [], marker='o', color='red', markersize=10)
    ax3d.legend()
    fig3d.tight_layout()
    # ── END 3D INITIALIZATION ──

    for idx, run in enumerate(runs):
        print(f"  [{ds}] Processing segment {idx+1}/{len(runs)}...")
        s_t, e_t, gt_h = run["start_time"], run["end_time"], run["height_diff"]
        
        # Calculate algorithms
        mask = (t_acc >= (s_t - 2.0)) & (t_acc <= (e_t + 2.0))
        t_sub = t_acc[mask]
        a_sub = a_mag[mask]
        
        mask_rest = (t_sub < s_t)
        g_est = np.mean(a_sub[mask_rest]) if np.any(mask_rest) else np.mean(a_sub[:10])
        a_clean = a_sub - g_est
        
        h_direct = estimate_height_direct(t_sub, a_clean)
        h_zupt = estimate_height_zupt(t_sub, a_clean, accel_threshold=0.2)
        h_kalman = estimate_height_kalman(t_sub, a_clean, accel_threshold=0.2)
        
        idx_sb = np.argmin(np.abs(tb - s_t))
        idx_eb = np.argmin(np.abs(tb - e_t))
        baro_h = abs(hab[idx_eb] - hab[idx_sb])

        # ── Update 3D Segment Highlight ──
        pmask = (tp >= (s_t - 2.0)) & (tp <= (e_t + 2.0))
        highlight_line.set_data(xp[pmask], yp[pmask])
        highlight_line.set_3d_properties(zp[pmask])

        # ── INITIALIZE 2D FIGURE FOR THIS SEGMENT ──
        fig2d, ax2d = plt.subplots(1, 1, figsize=(6, 5), dpi=120)
        ax2d.plot(t_sub, h_direct, label=f'Direct ({h_direct[-1]:.2f}m)', linestyle=':', color='black', alpha=0.8)
        ax2d.plot(t_sub, h_kalman, label=f'Kalman ({h_kalman[-1]:.2f}m)', linestyle='-.', color='#ff7f0e', linewidth=2)
        ax2d.plot(t_sub, h_zupt, label=f'ZUPT ({h_zupt[-1]:.2f}m)', color='#1f77b4', linewidth=3)
        ax2d.axhline(gt_h, color='#2ca02c', linewidth=2, label=f'GT ({gt_h:.2f}m)')
        if not np.isnan(baro_h):
            ax2d.axhline(baro_h, color='#9467bd', linestyle='--', linewidth=2, label=f'Baro ({baro_h:.2f}m)')
        
        y_max = max(gt_h, h_zupt[-1], h_kalman[-1], baro_h if not np.isnan(baro_h) else 0) * 1.5
        ax2d.set_ylim(min(0, min(h_direct[-1], h_zupt[-1], h_kalman[-1], gt_h))*1.5 - 1, y_max)
        
        ax2d.set_title(f"Vertical Altitude Estimators (Ride #{idx})", fontweight='bold')
        ax2d.set_xlabel("Time (s)")
        ax2d.set_ylabel("Vertical Height (m)")
        ax2d.legend(loc='upper left')
        ax2d.grid(True, linestyle='--', alpha=0.5)
        
        time_marker = ax2d.axvline(x=t_sub[0], color='red', linewidth=3)
        fig2d.tight_layout()
        # ── END 2D INITIALIZATION ──
        
        start_frame_idx = int((s_t - 2.0) * fps)
        end_frame_idx = int((e_t + 2.0) * fps)
        
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame_idx)
        current_frame = start_frame_idx
        
        while current_frame <= end_frame_idx:
            ret, frame = cap.read()
            if not ret: break
            
            if current_frame % frame_skip == 0:
                current_t = current_frame / fps
                
                # Update 3D dot
                curr_p_idx = np.argmin(np.abs(tp - current_t))
                current_dot.set_data(np.array([xp[curr_p_idx]]), np.array([yp[curr_p_idx]]))
                current_dot.set_3d_properties(np.array([zp[curr_p_idx]]))
                
                # Render 3D frame
                fig3d.canvas.draw()
                img3d_rgba = np.asarray(fig3d.canvas.buffer_rgba())
                img3d = cv2.cvtColor(img3d_rgba[..., :3], cv2.COLOR_RGB2BGR)

                # Update 2D Vertical Line
                t_clamped = max(t_sub[0], min(current_t, t_sub[-1]))
                time_marker.set_xdata([t_clamped, t_clamped])
                
                # Render 2D frame
                fig2d.canvas.draw()
                img2d_rgba = np.asarray(fig2d.canvas.buffer_rgba())
                img2d = cv2.cvtColor(img2d_rgba[..., :3], cv2.COLOR_RGB2BGR)
                
                # Equalize heights for hconcat
                target_h = img2d.shape[0]
                
                # Resize 3D frame strictly by height maintaining aspect
                aspect3d = img3d.shape[1] / img3d.shape[0]
                img3d_resized = cv2.resize(img3d, (int(target_h * aspect3d), target_h))

                # Resize Camera frame strictly by height maintaining aspect
                aspect_cam = frame.shape[1] / frame.shape[0]
                frame_resized = cv2.resize(frame, (int(target_h * aspect_cam), target_h))
                
                # Stitch: Camera | 3D Map | 2D Graph
                combined = cv2.hconcat([frame_resized, img3d_resized, img2d])
                
                if writer is None:
                    final_h, final_w = combined.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    writer = cv2.VideoWriter(out_video_path, fourcc, target_fps, (final_w, final_h))
                    print(f"  Initialized VideoWriter: {final_w}x{final_h} @ {target_fps}fps")
                    
                writer.write(combined)
            
            current_frame += 1
            
        plt.close(fig2d) # Clear the 2D plot for the next segment!

    if writer:
        writer.release()
    cap.release()
    plt.close(fig3d)
    print(f"Successfully finalized {out_video_path}")

def main():
    os.makedirs(VID_OUT_DIR, exist_ok=True)
    with open(META_JSON, 'r') as f:
        segments = json.load(f)
        
    for ds, runs in segments.items():
        generate_dataset_video(ds, runs)

if __name__ == "__main__":
    main()
