# `check_grid_across_signal/` — whole-signal trapezoid detector + inspection UI

Two files:
* `detect.py` — detector algorithm (pure library — no I/O).
* `editor.py` — Tkinter UI that runs the detector live against a loaded
  experiment and lets you click around to understand what it did and why.

---

## 1. How the detector works (`detect.py`)

Six-stage pipeline, all in-memory. The entire (W, f) grid = 30 × 15 = 450
trapezoid templates, reused at every stage from
`../fit_elevator_parameters/common.py`.

### Stage 1 — Preprocess
* `a_vert` = gravity-projected vertical accel (uses the session's estimated
  gravity vector).
* `a_smooth` = 0.4 s rolling mean of `a_vert` (`SMOOTH_SEC`).

### Stage 2 — Whole-signal matched-filter sweep
For every `(W, f)` on the grid run `match_one_template` over the full
`a_smooth`. It returns per-sample:
* `r2_local[i]`  — how well that template explains the signal around
  sample `i`,
* `A_hat[i]`     — LS-optimal signed amplitude for the fit.

Reduce across the grid to per-sample arrays. Four pairs of arrays:

| array | meaning |
|-------|---------|
| `best_r2`, `best_A`, `best_W_idx`, `best_f_idx` | unsigned argmax across the grid |
| `best_pos_r2`, `best_pos_A` | argmax restricted to templates with `A_hat > 0` |
| `best_neg_r2`, `best_neg_A` | argmax restricted to templates with `A_hat < 0` |

The per-sign arrays are what the UI's signed-R² panel plots (stage-4-and-5
diagnostic).

### Stage 3 — Peak-pick candidates
On `best_r2` gated by `|best_A| >= MIN_PEAK_ABS_A`:
* strict local maxima above `R2_PEAK_THRESH`,
* NMS of `NMS_RADIUS_S` (dedup duplicate peaks from the same physical
  pulse).

Each survivor carries its sign = `np.sign(best_A[i])`.

### Stage 4 — Same-sign NMS
Within each sign separately, two candidates must be at least
`SAME_SIGN_MIN_GAP_S` apart. Greedy by R² desc — highest-R² same-sign
candidate wins its window. Real physics doesn't put two same-sign
elevator pulses closer together.

Output: `final_peaks` — the candidate lobes handed to the pair stage.

### Stage 5 — Pair filter (shared-shape joint fit)
For every `(+candidate, −candidate)` pair with gap in
`[MIN_RIDE_S, MAX_RIDE_S]`:

For each `(W, f)` on the grid, closed-form LS:
```
|A|*   = (s₁·inner₁ + s₂·inner₂) / (2·norm_t)
R²ₖ   = 1 − (Pₖ − 2·|A|*·sₖ·innerₖ + |A|*²·norm_t) / Pₖ
score = mean(R²₁, R²₂)
```
Take the argmax `(W*, f*)` across the grid for that pair.

Accept the pair if `score ≥ JOINT_R2_THRESH` **and**
`|A|* ≥ MIN_PAIR_ABS_A`.

### Stage 6 — Greedy conflict resolution
Sort accepted pairs by `score` desc. Walk the list, accepting a pair
only if both are true:
* neither lobe is already committed,
* the pair's time interval `[t_start, t_end]` does not intersect any
  already-accepted interval.

Emit one dict per accepted pair (schema mirrors
`fit_elevator_parameters.common.RideFit` — same `lobe1` / `lobe2` /
`joint_r2_mean` fields) so existing tooling works unchanged.

### Tunables (top of `detect.py`)

| constant | default | role |
|----------|---------|------|
| `MIN_PEAK_ABS_A` | 0.5 | amplitude floor for a peak candidate |
| `R2_PEAK_THRESH` | 0.80 | min unsigned R² at peak-pick |
| `NMS_RADIUS_S`   | 0.5 | small NMS dedup radius |
| `SAME_SIGN_MIN_GAP_S` | 20.0 | min gap between same-sign candidates |
| `MIN_RIDE_S`     | 0.0  | min pair gap (set 0 to accept back-to-back) |
| `MAX_RIDE_S`     | 30.0 | max pair gap |
| `JOINT_R2_THRESH`| 0.85 | pair-stage joint R² threshold |
| `MIN_PAIR_ABS_A` | 0.5 | pair-stage shared |A| threshold |

Change a constant and re-Load the experiment in the editor — no caches to
bust.

### Public API
* `compute_predictions(acc_df) -> list[dict]` — pure function.
* `preprocess_and_sweep(acc_df) -> dict` — stage 1+2 output (every array
  the UI needs).
* `diagnose_window(state, t_lo, t_hi, ride_type)` — for a time window,
  walks every threshold and returns a human-readable verdict.

---

## 2. The editor UI (`editor.py`)

Live tool — runs the detector on the loaded experiment, no
`predictions.json`, no caches. Read-only; use `gt_editor.py` for GT edits.

### Layout

* **Left** — sensor panels (`altitude`, `velocity`, `a_vert`,
  `acc_velocity`, `gyr`, `mag`). GT spans shaded faintly, accepted
  prediction spans hatched on top in navy / purple.
* **Right top** — two tables:
  * **Predictions** — every pair accepted by the detector.
  * **GT rides** — every GT `up` / `down` with a `matched` / `unmatched`
    status against the predictions.
* **Right bottom — detail panel** (three rows):
  1. Two `(W, f)` R² heatmaps at the lobe centers (or at the best ± samples
     inside a GT window).
  2. Zoomed `a_vert` + `a_smooth` with the fitted trapezoid pair overlaid
     in red.
  3. **Signed-R² over time** (the diagnostic added for missed-peak
     investigations — see next section).

Above the detail figure: `Window ± (s)` spinbox + `×2` / `÷2` buttons.
Controls how many seconds of padding the detail view renders on each side
of the selection. Default 5 s. Widen it to see same-sign suppressors that
fall outside the default view.

### Signed-R² over time panel

The bottom row of the detail panel answers the question
*"why didn't the detector pick up this other pulse?"*.

**Two lines:**

| legend | color | definition |
|--------|-------|------------|
| `max R² (+)` | blue | `best_pos_r2(t)` — highest R² across the 450-cell grid among templates whose `A_hat > 0` at this sample |
| `max R² (−)` | red  | `best_neg_r2(t)` — highest R² across the grid among templates whose `A_hat < 0` at this sample |

Both are **max** operations — the `(+)` / `(−)` is a sign filter on the
amplitude, not "max vs min". A sample that looks like an upward pulse
lights up the blue line; downward, the red line. The dashed horizontal
line is `R2_PEAK_THRESH`.

**Colored dots = local maxima of the signed-R² traces.** The color tells
you at which pipeline stage the peak was kept or dropped:

| color       | tag                  | meaning |
|-------------|----------------------|---------|
| green       | `accepted`           | survived every stage and is one of the two lobes of an accepted prediction pair |
| orange      | `unpaired (greedy)`  | made `final_peaks` but no pair involving it was accepted: either nothing cleared `JOINT_R2_THRESH`, or the stage-6 greedy gave its partner / interval to a higher-scoring pair |
| purple      | `same-sign NMS`      | passed R² and \|A\| thresholds but another same-sign peak within `SAME_SIGN_MIN_GAP_S` (20 s) had higher R² and suppressed it at stage 4 |
| dark purple | `NMS (local)`        | suppressed by the stage-3 small NMS (`NMS_RADIUS_S = 0.5 s`) — a duplicate of a stronger peak less than half a second away |
| slate       | `lost to opp sign`   | signed R² is high at this sample, but the **unsigned** argmax picked the opposite sign — so this sign never entered stage-3 peak-pick. This is a common cause of missed pulses when the two lobes fight at the same sample |
| grey        | `R²<thr`             | signed R² is below `R2_PEAK_THRESH` (0.80) |
| light grey  | \|A\|<thr          | fitted \|amplitude\| is below `MIN_PEAK_ABS_A` (0.5 m/s²) |

The legend in the panel lists **only the statuses present in the current
window** to keep it compact.

> **Gotcha.** The dot-finder only draws peaks inside the zoomed window,
> but `same-sign NMS` runs over the whole session. A purple peak's
> suppressor can live outside the view. Widen `Window ± (s)` until the
> green/orange same-sign neighbor shows up.

### Clicking

* Click any left-panel interval → selects the prediction (top) or GT
  band (behind) under the cursor; detail pane re-renders.
* Click a **Predictions** row → heatmaps at the pair's lobe centers,
  signal zoom with fitted trapezoids, signed-R² panel, verdict text
  (accepted-pair summary).
* Click a **GT Rides** row → heatmaps at the best +/− samples inside the
  GT window (if any), signal zoom with those peaks marked, signed-R²
  panel, verdict text that names the specific threshold each stage
  failed on (`R²=0.78 < 0.85`, `|A|=0.34 < 0.50`,
  `gap=45 s outside [0, 30]`, …).

### Zoom / pan

Same controls as `gt_editor.py`:
* mouse wheel at cursor → zoom X (Shift+wheel → zoom Y)
* `+` / `=` / `-` → zoom X around center; `0` → fit X to full signal
* `Shift+←/→/↑/↓` → pan
* top-bar `Fit` / `± Zoom X/Y` / `◀ X`, `X ▶`, `▼ Y`, `Y ▲` buttons
  mirror the keyboard shortcuts

### Running the editor

```
venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/check_grid_across_signal/editor.py [exp_folder_name]
```

If `exp_folder_name` is provided, the editor auto-loads that experiment
~50 ms after opening.
