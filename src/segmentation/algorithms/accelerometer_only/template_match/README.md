# `template_match/` — trapezoid-based elevator-ride work

Everything that matches a **trapezoid pulse** against the accelerometer
signal lives under this folder. Three sub-packages share the same
`(W, f)` trapezoid-template grid and the same matched-filter primitive,
then branch on how they use it:

| sub-package | input | output | GT used at runtime? |
|---|---|---|---|
| `fit_elevator_parameters/` | sensors + GT | per-ride trapezoid params | **yes** — fits inside GT windows |
| `check_grid_across_signal/` | sensors | predicted ride intervals | **no** — detects rides from scratch |
| `matcher.py` / `templates.py` / `build_pulse_labels.py` | sensors + trained templates | legacy NCC segment candidates | no |

`scripts/` contains one-shot analysis utilities on top. `labels/` is
the on-disk artefact tree (generated PNGs, JSONs, CSVs, iteration logs,
mistake dumps).

---

## 1. The algorithm (conceptually)

The production detector is in `check_grid_across_signal/`. It turns a
continuous accelerometer stream into a list of `(t_start, t_end, type)`
ride intervals in **six stages**, which split cleanly across two
modules: the detection stage finds candidate lobes, and the clearing
stage (the pair filter) decides which pairs of lobes are real rides.

**Stage 1 — Preprocess.** Raw 3-axis accelerometer projected onto the
per-session gravity direction → signed vertical component
`a_vert`. Rolling-mean smoothing with `SMOOTH_SEC = 0.4`s → `a_smooth`.

**Stage 2 — Full-signal (W, f) template sweep.** 30 half-widths
× 15 flat-fractions = 450 trapezoid templates. For every template, run
a matched filter (normalized cross-correlation with LS-optimal
amplitude + coefficient of determination) across the whole session.
Reduce to per-sample argmax arrays:
- `best_r2[i]`, `best_A[i]` (unsigned argmax)
- `best_pos_r2 / best_pos_A` (restricted to $\hat A > 0$)
- `best_neg_r2 / best_neg_A` (restricted to $\hat A < 0$)

**Stage 3 — Peak pick.** Strict interior local maxima of `best_r2`
above `r2_peak_thresh` gated by `|best_A| ≥ min_peak_abs_a`, followed
by a small-radius NMS of `nms_radius_s` (dedup duplicate samples of the
same pulse). Sign of each survivor from `sign(best_A[i])`.

**Stage 4 — Same-sign NMS.** Within each sign separately, two peaks
must be at least `same_sign_min_gap_s` apart — a lift never places two
take-offs or two landings that close together. Highest-R² candidate
wins its window.

**Output of detection (stages 1–4):** `final_peaks` — signed candidate
lobes.

**Stage 5 — Pair filter.** For every admissible `(+, −)` pair with
`min_ride_s ≤ Δt ≤ max_ride_s`, refit a **shared-shape** trapezoid
pair: a single `(W, f)` template and a single `|A|` explain both lobes
jointly. Closed-form LS gives:
$$
A^\star(W, f) = \frac{u_1 + u_2}{2 \lVert \tau \rVert^2}, \qquad
\text{score} = \tfrac{1}{2}(R^2_1 + R^2_2)
$$
where $u_k = s_k \langle a_k, \tau \rangle$. The pair is accepted iff
the grid-max score ≥ `joint_r2_thresh` and $A^\star$ ≥
`min_pair_abs_a`.

**Stage 6 — Greedy conflict resolution.** Accepted pairs are ranked by
$\text{score} - 0.01 \cdot \Delta t$ (a duration-penalty term — see
§5 below) and walked in that order. A pair is committed only if
(a) neither of its lobes is already in a committed pair, and (b) its
time interval doesn't intersect any already-committed interval.

**Output:** A list of `{t_start_s, t_end_s, ride_type, duration_s,
lobe1, lobe2, joint_r2_mean}` dicts. `t_start_s = t(i_1)`,
`t_end_s = t(i_2)` — the ride endpoints are the lobe centres (not
ramp-edge thresholds), because that's the natural quantity the
kinematic model predicts is proportional to floor-to-floor height.

---

## 2. Code structure

### `check_grid_across_signal/` — the production detector
| file | role |
|---|---|
| `detect.py` | Detection (stages 1–4) + the top-level `predict_intervals(acc, config)` wrapper + `diagnose_window(state, t_lo, t_hi)` UI helper + the `DetectConfig` dataclass bundling all 8 tunables. Also re-exported UI helpers (`heatmap_at`, `find_local_maxima`, `classify_peak` + `PEAK_STATUS_*` constants) that the editor and the mistakes dump consume. |
| `pair_filter.py` | The clearing algorithm (stages 5–6): `joint_pair_score(a, t, i1, i2, s1, s2)` and `predict_pairs(state, config)`. |
| `editor.py` | Tkinter inspection UI. Pure display — every algorithmic helper lives in `detect.py`. |
| `evaluate.py` | Sweep harness: single-run evaluation + grid-search over `DetectConfig` + `--save-best` to persist the winner as JSON. Uses `IntervalPredictionMetrics.from_intervals` for the four-mode composite and `IntervalPredictionMetrics.iou_f1` for classical F1 @ IoU ≥ 0.5. |
| `dump_mistakes.py` | Diagnostic dump: runs the detector on every train exp, matches GT vs. pred, and renders the editor's 3-row figure (heatmaps + signal zoom + signed-R² trace) for every non-clean match. |

### `fit_elevator_parameters/` — trapezoid fit **inside** GT windows
This is a separate tool: given GT spans, fit the best trapezoid pair to
each one (useful for template design / parameter calibration).
- `basic_grid.py` — independent per-lobe fit.
- `constrained_grid.py` — shared-shape per-ride fit (same math as the
  pair-filter's joint fit).
- `common.py` — **this is where the shared primitives live**:
  `trapezoid_kernel`, `_vertical_accel`, `_smooth`,
  `match_one_template`, and the global `GRID_W_S` / `GRID_F` arrays and
  their bounds `W_MIN_S=0.4`, `W_MAX_S=3.0`, `F_MIN=0.05`,
  `F_MAX=0.80`. Any change to the grid must happen here.

### `metrics` package
Lives one package up at `src/segmentation/algorithms/metrics/`.
Exports:
- `SegmentationMetrics` + `DetectionResult` — upstream segmentation
  metrics (IoU matching, ECE, Brier, CI coverage).
- **`IntervalPredictionMetrics`** — the detector's evaluator.
  `from_intervals(gt_rides, predictions)` returns counts across the
  four failure modes (see §4 below). `score()` returns the composite;
  `iou_f1(...)` returns classical F1 @ IoU for comparability with
  external work.

### `matcher.py` / `templates.py` / `build_pulse_labels.py`
Legacy NCC pipeline — not related to the trapezoid detector above.
Kept in place because the public
`detect_elevator_segments_from_template_match` entry in `__init__.py`
still ships from here.

---

## 3. Configuration

One dataclass, `DetectConfig`, holds all eight tunables and is the
single object every caller passes. Defaults below are the "safe" ones
in code; the sweep-winning values are in `labels/check_grid_across_signal/best_detect_config.json`.

| field | default | meaning |
|---|---|---|
| `r2_peak_thresh` | 0.80 | minimum R² at peak pick |
| `min_peak_abs_a` | 0.5 | amplitude floor for a peak candidate (m/s²) |
| `nms_radius_s` | 0.5 | small NMS dedup radius (s) |
| `same_sign_min_gap_s` | 20 | min gap between same-sign candidates (s) |
| `min_ride_s` | 0 | min (take-off → landing) gap of a ride (s) |
| `max_ride_s` | 120 | max gap before we reject the pair (s) |
| `joint_r2_thresh` | 0.75 | min shared-shape joint R² |
| `min_pair_abs_a` | 0.5 | min shared \|A\| of the accepted pair (m/s²) |

The eight fields split cleanly into "detection-stage" (first four) and
"pair-filter-stage" (last four). The sweep pins `min_ride_s = 10` and
`max_ride_s = 120` by user constraint and grid-searches the other six.

---

## 4. Evaluation

Two complementary metrics are reported for every configuration.

### Four-mode composite (`IntervalPredictionMetrics`)
Given a GT set and a prediction set, build a bipartite overlap graph.
Intervals count as matching if their absolute overlap is ≥ 1 s **or**
their overlap covers ≥ 30% of the shorter interval. Every GT and every
pred is then categorised:

| category | meaning |
|---|---|
| `clean` | 1-to-1 match (good) |
| `missed` | GT with zero overlapping preds |
| `gt_merged` | GT that shares its single overlapping pred with other GTs |
| `gt_split` | GT covered by ≥ 2 preds |
| `fp` | Pred with zero overlapping GTs |
| `pred_merged` | Pred covering ≥ 2 GTs |
| `pred_split_part` | Pred that's one of several sharing a single GT |

The aggregate score is:
$$
\mathrm{F1}^\ast = \frac{2 \cdot \mathrm{clean}}{2 \cdot \mathrm{clean} + \mathrm{bad}_{\mathrm{GT}} + \mathrm{bad}_{\mathrm{pred}}}
$$
Every failure mode counts once. A merge-heavy and a miss-heavy
configuration score very differently — this is what makes the metric
useful for guiding pair-filter changes.

### IoU-F1 @ 0.5
Standard temporal-detection F1. Greedy best-IoU-per-GT matching;
match if IoU ≥ 0.5; pooled TP/FP/FN counts; F1. Comparable to
external segmentation / temporal-detection baselines (PASCAL VOC,
COCO temporal tracks, ActivityNet).

Both are reported together in `evaluate.py` output and in the
per-iteration JSONs under `pair_filter_iterations/`.

---

## 5. Current state of the detector (what the sweep + iterations found)

**The detector as shipped works, but recall is low.** The best
`DetectConfig` from the sweep with the best pair-filter variant scores
**`f1_like = 0.228`** and **IoU-F1 @ 0.5 = 0.169** across 22 training
experiments (415 GT rides total; 81 cleanly matched, 106 merged, 133
missed).

**Threshold sweep** (144 combos, see `interval_sweep.csv`) found that
the picked values sit at a "recall cliff" on each axis — lowering them
adds FPs faster than recall; raising them tips the mix into misses.

**Pair-filter iteration** (10 variants, see
`pair_filter_iterations/README.md`) identified the dominant failure
mode as **super-pair merges** (a take-off from ride 1 pairing with a
landing from ride 5 or later, swallowing every intermediate GT). The
winning change was a **duration-penalty greedy**:
$$
\text{ranking key} = \text{score} - 0.01 \cdot \Delta t_{\text{sec}}
$$
already applied in `pair_filter.py`. It more than doubled the
composite score (0.097 → 0.228) and halved the merge count.

### Residual failure mode (open)

With duration penalty active, a second-order problem shows up in the
mistakes dump: **dwell pairs** — the landing of ride $n$ gets paired
with the take-off of ride $n+1$ because their gap is shorter than the
real ride durations, so the monotonic short-gap preference now picks
them. This is visible in nearly every `pred_merged` figure in
`labels/check_grid_across_signal/mistakes/` where the super-pair
structure has been broken up into smaller wrong pairs.

### Candidate fixes for the next iteration

1. **Band penalty** — replace `λ · Δt` with `λ · |Δt − T*|`, where
   `T* ≈ 15 s` is a prior on typical short-ride duration. Penalises
   both super-pairs (large Δt) and dwell pairs (small Δt).
2. **Tighter min-gap floor** — bump `min_ride_s` from 10 s to
   ≥ 12 s. Rejects back-to-back dwell windows at the cost of very
   short real rides.
3. **Time-sorted greedy** — walk the peak list left-to-right in time
   and commit each peak's pair with its next admissible opposite
   immediately. Structurally prevents a landing from being stolen by a
   later take-off.
4. **Same-sign-in-middle guard** — if the candidate pair `(i1, i2)` has
   another same-sign-as-`i1` peak between them, the candidate is
   probably a dwell fake; reject.

### Outright recall problems (different animal)

Some experiments have **zero clean matches** despite dozens of GT
rides — e.g. `eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1`
has 10 / 10 missed and 0 predictions. Looking at the mistake figures
for those exps (`gt_missed_*.png`) shows the signed-R² traces never
clear `r2_peak_thresh`, usually because the phone's accelerometer
scaling on that device is systematically smaller. A **per-phone
amplitude calibration** would recover most of them.

---

## 6. Artefact tree (everything auto-generated)

```
labels/
├── experiment_overview/              # per-experiment overview PNGs (legacy)
├── ride_segments/                    # GT-ride-per-PNG (legacy)
├── fit_elevator_paramater/
│   ├── basicTreepzeGrid/             # basic_grid.py outputs
│   └── basicTreepzeGridWithConstraint/
└── check_grid_across_signal/         # the detector's artefacts
    ├── best_detect_config.json       # sweep winner
    ├── interval_sweep.csv            # full 144-row sweep table
    ├── mistakes/                     # dump_mistakes.py output
    │   ├── README.md
    │   └── <exp>/<kind>_<idx>_t<start>.png   # 547 figures total
    └── pair_filter_iterations/       # 10 pair-filter variants
        ├── README.md                 # summary + progress chart
        ├── orchestrator.py           # rerunnable driver
        ├── progress.png
        └── iter_NN_<slug>/
            ├── notes.md
            ├── metrics.json
            ├── errors_bar.png
            └── timeline_<exp>.png
```

The `check_grid_across_signal/` sub-tree is the full handoff for the
detector's empirical state — every number in §5 is backed by a JSON
file you can re-load.

---

## 7. Running everything

```bash
# 1. Single detector run, default config, all train exps.
PYTHONPATH=. venv/bin/python \
  -m src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.evaluate

# 2. Grid-search sweep (144 combos × 22 exps, ~75 min).
PYTHONPATH=. venv/bin/python \
  -m src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.evaluate \
  --sweep --min-ride-s 10 --max-ride-s 120 \
  --out .../labels/check_grid_across_signal/interval_sweep.csv \
  --save-best .../labels/check_grid_across_signal/best_detect_config.json

# 3. Pair-filter iteration exploration (10 variants, ~5 min).
PYTHONPATH=. venv/bin/python \
  src/segmentation/algorithms/accelerometer_only/template_match/labels/check_grid_across_signal/pair_filter_iterations/orchestrator.py

# 4. Dump diagnostic figures for every mistake (547 PNGs, ~2 min).
PYTHONPATH=. venv/bin/python \
  -m src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal.dump_mistakes

# 5. Interactive editor UI for one experiment.
venv/bin/python \
  src/segmentation/algorithms/accelerometer_only/template_match/check_grid_across_signal/editor.py \
  <exp_folder_name>

# 6. `fit_elevator_parameters/` — per-ride trapezoid fits inside GT windows.
PYTHONPATH=. venv/bin/python \
  -m src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.basic_grid
PYTHONPATH=. venv/bin/python \
  -m src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.constrained_grid
```

All CLIs emit to paths inside `labels/` — nothing writes outside this
tree.

---

## 8. Handoff checklist for the next LLM

### What's already done
- Detection stage + pair filter split into `detect.py` + `pair_filter.py`.
- `DetectConfig` dataclass is the single knob bag — no module globals.
- 144-combo sweep found the best threshold values; saved as JSON.
- 10-iteration pair-filter exploration identified the duration-penalty
  greedy rule; that rule is applied in production `pair_filter.py`.
- Evaluation uses both the four-mode composite and classical IoU-F1.
- Editor UI is display-only (logic moved to `detect.py` helpers).
- Every mistake in the training set has an editor-style diagnostic
  figure on disk.
- Docs reflect the current state
  (`docs/latex/main.tex §sec:trapezoid-detector`).

### What's open
- `f1_like ≈ 0.228` is low. Recall is ~20%. The detector is not
  production-quality yet.
- **Biggest open issue**: dwell pairs (described in §5). Band penalty
  is the most promising fix; implement as an extension of the
  orchestrator's `predict_pairs_variant` and re-run.
- **Second open issue**: some phones have 0 detections (systematic
  amplitude miscalibration). A per-phone (or per-experimenter) scaling
  factor on `a_smooth` before the sweep could recover those.
- The "next steps" list in `pair_filter_iterations/README.md §Next
  steps` enumerates 4 concrete ideas in priority order.

### Things not to change without understanding
- The `(W, f)` grid bounds in `fit_elevator_parameters/common.py` are
  used by every downstream caller (detector, fitters, pair filter).
  Changing them invalidates the cached sweep and the iteration log.
- The `metrics/metrics.py` file is shared with the rest of
  `src/segmentation/algorithms/` — its `SegmentationMetrics`,
  `DetectionResult`, `iou`, `ci_center` exports are consumed
  elsewhere. Add to it, don't restructure it.
- The editor's signed-R² colour legend is paired with the
  `PEAK_STATUS_*` constants in `detect.py` — they must stay in sync.

### Useful entry points when debugging a missed GT
1. Open the editor on the exp, click the missed GT row. The bottom
   "signed R²" panel tells you exactly which pipeline stage killed it
   (colour-coded: grey = below thresh, slate = lost to opposite sign,
   purple = NMS-suppressed, orange = unpaired greedy, green =
   accepted).
2. Look in `labels/check_grid_across_signal/mistakes/<exp>/gt_missed_<i>_t<t>.png`
   for the same view, but saved.
3. Call `detect.diagnose_window(state, gt.t_start_s, gt.t_end_s,
   gt.type)` from a REPL — returns a dict with `verdict_lines`
   explaining each stage's verdict in plain English.
