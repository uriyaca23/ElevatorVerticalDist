# Detector mistakes — diagnostic dump

Rendered at config = `DetectConfig(r2_peak_thresh=0.75, min_peak_abs_a=0.5, nms_radius_s=0.5, same_sign_min_gap_s=5.0, min_ride_s=10.0, max_ride_s=120.0, joint_r2_thresh=0.75, min_pair_abs_a=0.3)` over 22 train experiments.

Each subfolder is one experiment. Each PNG is one mistake. File name format: ``<kind>_<index>_t<start-second>.png``. Kinds:

| kind | meaning |
|---|---|
| `gt_missed` | GT ride not covered by any prediction. |
| `gt_merged` | GT ride is one of several covered by a single pred. |
| `gt_split` | GT ride covered by ≥2 predictions. |
| `pred_fp` | Pred lands on outside (no overlapping GT). |
| `pred_merged` | One pred swallows several GTs. |
| `pred_split_part` | Pred is one of several sharing a single GT. |

## Layout per figure

Three rows per PNG (mirror the editor's detail panel):

1. **(W, f) R² heatmaps at the two lobe centres.** Positive lobe on the left, negative on the right. The red × marks the joint-fit template (when both lobes exist); for GT-side mistakes that's the shared-shape fit over the best ± samples inside the GT window.
2. **Signal zoom.** `a_vert` (dark blue) and the smoothed trace (orange) over the mistake plus context pad. All GT spans in view are faintly shaded; the focal GT or pred is highlighted. Fitted trapezoid pair drawn in red when available.
3. **Signed-R² trace.** The editor's colour-coded peak status panel: green = accepted, orange = unpaired (greedy), purple / darker purple = NMS-suppressed, slate = lost to the opposite sign, grey shades = below-threshold.

## Totals

| kind | count |
|---|---|
| `gt_missed` | 133 |
| `gt_merged` | 106 |
| `gt_split` | 95 |
| `pred_fp` | 59 |
| `pred_merged` | 136 |
| `pred_split_part` | 18 |
| **total** | **547** |
