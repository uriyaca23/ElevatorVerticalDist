# `template_match/` — trapezoid-based elevator-ride work

Everything that matches a **trapezoid pulse** against the accelerometer
signal lives under this folder. Three independent pieces share the same
30×15 ``(W, f)`` grid and the same matched-filter primitive:

| sub-package | input | output | GT used at runtime? |
|-------------|-------|--------|---------------------|
| `fit_elevator_parameters/` | sensors + gt | per-ride trapezoid params | **yes** — fits inside GT windows |
| `check_grid_across_signal/` | sensors | predicted ride pairs | **no** — detects rides from scratch |
| `matcher.py` / `templates.py` / `build_pulse_labels.py` | sensors + trained templates | segment candidates (legacy NCC pipeline) | no |

Scripts under `scripts/` are one-shot analysis utilities that read the
outputs of the above. The `labels/` directory is the on-disk artefact
tree (PNGs + JSONs). Nothing in this folder depends on `labels/` at
runtime — the editor UI runs the detector live.

---

## 1. `fit_elevator_parameters/` — trapezoid fit INSIDE GT windows

Given a ground-truth ride span, find the best-matching trapezoid for
each of its two acceleration lobes (take-off + landing).

Two algorithms, one folder each under `labels/fit_elevator_paramater/`:

### `basic_grid.py` → `labels/fit_elevator_paramater/basicTreepzeGrid/`
Fits each lobe **independently**. For every `(W, f)` on the grid runs
the matched filter over the lobe's half of the GT window, restricts to
the sign expected from ride type (`up`: +A / −A, `down`: −A / +A) and
keeps the ``(t_c, A, W, f)`` with the highest local R².

### `constrained_grid.py` → `labels/fit_elevator_paramater/basicTreepzeGridWithConstraint/`
Forces both lobes of a ride to **share** ``(|A|, W, f)`` — only `t_c`
differs. For each ``(W, f)`` evaluates an ``N_pos × N_neg`` matrix of
(i1, i2) pairs; closed-form LS gives the shared ``|A|*`` and per-lobe
R² so the argmax is one broadcast per grid cell. Maximises the **mean**
of the two per-lobe R².

Both produce `_all_rides.png` + `parameters.json` per experiment.

### `common.py` — shared primitives
Kernel (`trapezoid_kernel`), preprocessing (`_vertical_accel`,
`_smooth`), the matched-filter sweep (`match_one_template` returning
`(A_hat, r2_local, inner, local_power, norm_t)`), ride slicing
(`build_ride_slices`), dataclasses (`LobeFit`, `RideFit`), the plot
helpers (`save_combined`, `save_parameters`) and the experiment driver
(`run_fitter`). Every other module in this sub-package plugs a `fit_ride`
callable into `run_fitter`.

### Running
```
venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/fit_elevator_parameters/basic_grid.py
venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/fit_elevator_parameters/constrained_grid.py
```

### Analysis scripts (under `scripts/`)
* `plot_trapezoid_similarity.py` — for **each** variant folder writes
  two PNGs: within-experiment trapezoid consistency and per-ride
  `|lobe1|` vs `|lobe2|` symmetry (the constrained variant's symmetry
  RMS is 0 by construction).
* `plot_failed_fits.py` — per-ride PNGs for GT rides whose fit has a
  `t_c is None` lobe, dropped into `<variant>/_failed_fits/<exp>/`.

---

## 2. `check_grid_across_signal/` — detect rides without GT

Whole-signal detector. GT is only used for plotting / diagnostics; the
algorithm never reads it.

### `detect.py` — the algorithm
Pipeline per experiment (all in-memory, no file output):

1. **Preprocess.** `a_vert` (gravity-projected) and smoothed
   `a_smooth` on the whole session.
2. **Sweep.** For every `(W, f)` on the 30×15 grid, run the same
   `match_one_template` over the full signal; at each sample keep a
   running argmax of R² across the grid — produces `best_r2`, signed
   `best_A`, `best_W_idx`, `best_f_idx`.
3. **Peak-pick + amplitude gate.** Local maxima of `best_r2` above
   `R2_PEAK_THRESH` with a small `NMS_RADIUS_S` dedup. Samples whose
   `|best_A|` is below `MIN_PEAK_ABS_A` are ignored — sub-noise
   trapezoid-shaped wiggles on flat stretches would otherwise produce
   r²≈1.0 with essentially random sign.
4. **Per-sign NMS.** Within each sign, two candidates must be at least
   `SAME_SIGN_MIN_GAP_S` apart — physics doesn't put two take-offs
   that close together.
5. **Pair filter.** For every remaining `(+, −)` pair with gap in
   `[MIN_RIDE_S, MAX_RIDE_S]`, refit a **shared-shape** trapezoid by
   re-searching the full `(W, f)` grid (closed form from
   `constrained_grid`). Accept if mean per-lobe R² ≥ `JOINT_R2_THRESH`
   and shared `|A|` ≥ `MIN_PAIR_ABS_A`.
6. **Greedy conflict resolution.** Accept pairs in descending joint-R²
   order, rejecting any pair whose lobes are already used **or** whose
   time interval `[t_start, t_end]` intersects a previously accepted
   pair — rides are non-overlapping by construction.

Result schema mirrors `RideFit` from `fit_elevator_parameters`:
```
{
  "index": 0, "ride_type": "up",
  "t_start_s": 12.3, "t_end_s": 18.1, "duration_s": 5.8,
  "lobe1": {"t_c": 12.3, "a_peak": +3.5, ...},
  "lobe2": {"t_c": 18.1, "a_peak": -3.5, ...},
  "joint_r2_mean": 0.905
}
```

Public API:
* `compute_predictions(acc_df) -> list[dict]` — pure function, no I/O.
* `preprocess_and_sweep(acc_df) -> dict` — exposes intermediate state
  (every array the pair stage and the UI diagnostics need).
* `diagnose_window(state, t_lo, t_hi, ride_type)` — given a time window,
  walks every threshold check and returns a human-readable verdict
  (used by the editor for GT-click diagnostics).

Tunables live at the top of the file; change them and re-Load in the
UI — no other caches.

### `editor.py` — Tkinter inspection UI
Sister tool to `src/data/gt_editor.py`. Loads an experiment, runs
`preprocess_and_sweep` + `compute_predictions` live, and shows:

* **Left.** Same sensor panels as the GT editor (`altitude`,
  `velocity`, `a_vert`, `acc_velocity`, `gyr`, `mag`). GT spans shaded
  faintly, predicted rides hatched on top in navy / purple. Zoom and
  pan buttons + mouse-wheel + keyboard shortcuts (`+` / `-` / `0` /
  `Shift+Arrows`) mirror the GT editor. **Click an interval** on any
  panel to select it (predictions on top, GT behind).
* **Right top.** Two tables — Predictions (the accepted pairs) and GT
  rides (every GT `up` / `down` with a `matched` / `unmatched` status).
* **Right bottom.** Detail panel for the selected row:
  * Prediction selected → two ``(W, f)`` R² heatmaps at `lobe1.t_c` and
    `lobe2.t_c` with the fitted `(W*, f*)` marked, plus a zoomed
    `a_vert` with the fitted trapezoids overlaid, plus a short verdict
    block.
  * GT selected → heatmaps at the best `+` and `−` samples **inside**
    the GT window (if any), the zoomed signal, and a verdict text box
    explaining which specific threshold rejected the interval
    (`R²=0.78 < 0.85`, `|A|=0.34 < 0.50`, `gap=45.2s outside
    [0, 30]`…).

Run:
```
venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/check_grid_across_signal/editor.py [exp_name]
```

---

## 3. Legacy NCC pipeline (top-level modules)

`matcher.py`, `templates.py`, `build_pulse_labels.py`,
`plot_experiment_overviews.py` are the older template-match pipeline
(normalized cross-correlation against trained reference pulses). They
are **not** related to the trapezoid fitters / detector above — kept
here because the public `detect_elevator_segments_from_template_match`
entry point still ships from this folder's `__init__.py`.

---

## Output directory tree

```
labels/
├── experiment_overview/         # per-experiment overview PNGs
├── ride_segments/               # GT-ride-per-PNG  (scripts/plot_gt_ride_segments.py)
└── fit_elevator_paramater/
    ├── basicTreepzeGrid/
    │   ├── <exp>/{_all_rides.png, parameters.json}
    │   ├── _trapezoid_similarity_grid.png
    │   ├── _trapezoid_lobe_symmetry_grid.png
    │   └── _failed_fits/<exp>/ride_NN_<type>.png
    └── basicTreepzeGridWithConstraint/
        └── same layout as above
```

`check_grid_across_signal/` produces no files — the detector is a
library, and the editor UI is its only consumer.

---

## Quick reference — which file does what

| path | role |
|------|------|
| `fit_elevator_parameters/common.py` | shared primitives (kernel, sweep, slicing, plot, driver) |
| `fit_elevator_parameters/basic_grid.py` | independent per-lobe fit |
| `fit_elevator_parameters/constrained_grid.py` | shared-shape per-ride fit |
| `check_grid_across_signal/detect.py` | whole-signal detector + pair filter |
| `check_grid_across_signal/editor.py` | Tkinter inspection UI |
| `scripts/plot_gt_ride_segments.py` | per-GT-segment PNGs |
| `scripts/plot_trapezoid_similarity.py` | fit-variant similarity + symmetry grids |
| `scripts/plot_failed_fits.py` | per-ride PNGs for failed fits |
| `matcher.py`, `templates.py`, `build_pulse_labels.py` | legacy NCC segmentation |
