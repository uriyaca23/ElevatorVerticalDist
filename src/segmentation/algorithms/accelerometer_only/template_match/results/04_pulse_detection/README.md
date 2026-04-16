# 04 — Matched-filter pulse detection

Try to locate elevator rides in an entire session by sliding template shapes across the session velocity signal.

## Pipeline

1. K-means cluster visually-labeled trapezoid rides on `(a_max, v_max, W)` → pick 5 representatives. Same for parabola on `(v_peak, W, p)` → 5 representatives. Total **10 templates**.
2. Render each representative as a velocity template at 100 Hz.
3. Compute session velocity: `cumsum(|a| − mean|a|)` → 4th-order Butterworth LPF @ 0.3 Hz (matches main_acc.py).
4. Sliding normalized cross-correlation (NCC) per template; take `max(|NCC|)` across all 5 templates of that shape as the per-shape confidence.
5. Gate into detections: NCC > 0.55 **AND** local peak |v| > 0.25 m/s **AND** local σ|a| < 1.2 m/s² **AND** duration ≥ 3 s. Merge runs closer than 2 s.

## Files

- `pulse_detect_templates.png` / `.md` — the 10 chosen template shapes.
- `pulse_detect_trapezoid_{name}.png` — smoothed `max|NCC|` from 5 trap templates + session velocity + GT overlay.
- `pulse_detect_parabola_{name}.png` — same for parabola templates.
- `pulse_detections_{name}.png` — final gated detections (green) vs GT (yellow), with TP/FP/FN counts.

## Current performance (both experimenters)

| Experimenter | Detections | TP | FP | Recall |
|---|---|---|---|---|
| oria | 75 | 37 | 38 | 37/44 |
| roy_turgman | 88 | 33 | 55 | 33/39 |

## Known issue

`|NCC|` saturates near 1 far too often outside GT windows. Shape similarity alone isn't enough — walking / sitting / phone reorientation integrate over 30 s into hump-shaped signals that match the templates. Precision is ~45%.

See the root `results/README.md` for proposed next-step approaches.

**Script:** `../../scripts/pulse_detect.py`
