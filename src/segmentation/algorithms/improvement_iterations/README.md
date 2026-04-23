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
