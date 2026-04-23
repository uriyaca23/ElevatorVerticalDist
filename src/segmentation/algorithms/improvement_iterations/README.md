# Improvement iterations — acc_template_match segmenter

Systematic iteration loop for reducing segmentation mistakes against GT on
all 26 experiments. Each iteration is a git branch sourced from the
previous iteration's branch:

```
development → iter_00_baseline → iter_01_<slug> → iter_02_<slug> → …
```

Each iteration folder `iter_NN_<slug>/` contains:

- `metrics.json` — full `IntervalPredictionMetrics` per-exp + aggregate + IoU.
- `per_gt.csv` — one row per GT with `status` (clean / missed / gt_merged /
  gt_split) and `pair_*` diagnostic fields (which threshold rejected it).
- `mistakes/<exp>__gt<idx>__<status>.png` — diagnostic figure per non-clean
  GT: signal + fitted trapezoid + signed-R² + (W,f) heatmaps + experiment
  timeline.
- `per_exp_summary.png` — per-experiment stacked bar of failure modes.
- `notes.md` — what changed this iter, patterns observed, next hypothesis.

**Target:** ≥70 % reduction in total mistakes (baseline = 260 → ≤ 78).

Run a new iteration:
```bash
venv/bin/python -m src.segmentation.algorithms.improvement_iterations._iter_runner \
    --iter 01 --slug <slug> --what "<short description>" --kind all
```

## Progress

| iter | branch | clean | mistakes | Δmistakes | f1_like | iou_f1@0.5 | note |
|---|---|---|---|---|---|---|---|
| 00 | `iter_00_baseline` | 264 / 498 | 260 | — | 0.669 | 0.406 | production code as of c4d2a9f |
| 01 | `iter_01_loosen_filters` | 352 / 498 | 292 | +32 | 0.703 | 0.454 | recall+++, FPs exploded (26→146) |
| 02 | `iter_02_tighten_pair_A` | 344 / 498 | 299 | +39 | 0.694 | 0.442 | min_pair_abs_a 0.30→0.40: only killed 1 FP — not the driver |
| 03 | `iter_03_middle_heatmap` | 336 / 498 | 261 | +1 | 0.717 | 0.470 | heatmap_energy 0.20→0.40: FPs 145→99, mistakes back near baseline |
| 04 | `iter_04_quiet_middle` | 334 / 498 | **214** | **−46** | **0.750** | **0.510** | NEW `quiet_middle_ratio=0.5` filter: FPs 99→50, revert `min_pair_abs_a` 0.40→0.30 |
| 05 | `iter_05_narrow_nms` | 334 / 498 | 218 | +4 | 0.747 | 0.508 | `nms_radius_s` 2.0→1.0 — no effect (wrong hypothesis) |
| 06 | `iter_06_per_sign_pick` | 334 / 498 | 218 | 0 | 0.747 | 0.508 | Per-sign peak-pick — no effect; root cause is upstream gates |
| 07 | `iter_07_lower_peak_gates` | 403 / 498 | **154** | **−64** | **0.835** | **0.601** | `r2_peak_thresh` 0.55→0.40, `min_peak_abs_a` 0.4→0.25 |
| 13 | `iter_13_triangle_shape_row` | 403 / 498 | 155 | +1 | 0.834 | **0.703** | Prepend f=0 (triangle) to grid_f(); big IoU win (+0.102) but mistake count flat. Misses are noisy-phone rides, not short rides. |
| 14 | `iter_14_widen_w_and_pair_a` | 394 / 498 | 155 | 0 | 0.817 | 0.691 | `w_min_s` 0.4→0.3 + `min_pair_abs_a` 0.30→0.22. Mistakes flat but gt_split 1→10 (beitMansour1 over-segmented). min_pair_abs_a=0.22 too permissive — will bisect. |
| 15 | `iter_15_w_min_only` | 400 / 498 | 147 | −7 | 0.839 | 0.702 | Keep triangle row + `w_min_s=0.3`; revert `min_pair_abs_a` to 0.30. Beats iter_07. FPs 59→49 (narrower W rejects walking artifacts). |
| 16 | `iter_16_lower_peak_a` | 400 / 498 | **144** | **−10** | **0.841** | 0.705 | `min_peak_abs_a` 0.25→0.20. New best. −3 FPs. Missed unchanged — 50 of 87 have peaks with R²~0.95 but A~0.14, still below gate. |
