# `basicTreepzeGrid/` — per-lobe trapezoid fit via matched-filter grid search

Outputs of
`src/segmentation/algorithms/accelerometer_only/template_match/fit_trapezoid_pulses.py`.
One folder per TRAIN experiment; `_all_rides.png` + `parameters.json`
inside each.

## What the fitter does

For every ground-truth `up` / `down` ride, we want a trapezoid model of
both acceleration lobes (see [`../../../../../../../README.md`](../../../../../../../README.md)
for the elevator kinematics).

1. **Grid of templates.** A 2-D grid of `(W, f)` shapes — 30 half-widths
   linearly spaced in `[0.4s, 3.0s]`, 15 plateau fractions in `[0.0, 0.8]`
   → **450 unit trapezoids** in total.
2. **Matched filter per lobe, per template.** For each `(W, f)` we slide
   the unit kernel across the gravity-projected vertical accelerometer
   `a_vert` and compute at every center:
   - `A_hat = ⟨a_local, tpl⟩ / ⟨tpl, tpl⟩`  (closed-form least-squares amplitude)
   - `R²_local = 1 − SS_res / SS_tot` on the ±W window
3. **Two-peak pick.** Lobe 1 searches `[0.0, 0.6] × duration`, lobe 2
   searches `[0.4, 1.0] × duration` (small overlap near the midpoint).
   Each lobe is restricted to the sign expected from ride type:

   | ride type | lobe 1 (take-off) | lobe 2 (landing) |
   |-----------|-------------------|------------------|
   | `up`      | +A                | −A               |
   | `down`    | −A                | +A               |

   Across the `(t_c, W, f)` candidates that satisfy the sign constraint,
   we pick the one with the highest local R². The two lobes are fit
   **independently** — different amplitudes, widths, and plateau
   fractions are allowed, so a noisy landing lobe no longer drags the
   take-off fit (the failure mode we saw with the old shared-parameter
   `curve_fit` approach).

## Plot layout (`_all_rides.png`)

Each ride panel has three stacked axes sharing the same ride-local time:

1. `a_vert` — both fitted trapezoids overlaid in red; annotation shows
   `L1` and `L2` amplitudes, widths, plateau fractions, and local R².
2. `vz` — vertical velocity integrated from the accelerometer.
3. Barometer-derived altitude (purple). Falls back to a grey "no
   barometer" placeholder for phones that don't have a PRS sensor, so
   the layout stays consistent across experiments.

GT `up` / `down` intervals are shaded (green / red) on all three axes.

## `parameters.json` schema

One entry per ride; each lobe is its own nested object:

```json
{
  "index": 0,
  "ride_type": "up",
  "duration_s": 13.58,
  "lobe1": {
    "t_c": 4.74, "a_peak": 0.34, "half_width_s": 0.49,
    "frac_flat": 0.69, "r2_local": 0.98
  },
  "lobe2": {
    "t_c": 7.78, "a_peak": -2.24, "half_width_s": 2.01,
    "frac_flat": 0.80, "r2_local": 0.95
  },
  "lobe_centroid_spacing_s": 3.04
}
```

`a_peak` is signed; `lobe_centroid_spacing_s` is the gap between the two
lobe centers, proportional to floor-to-floor height `H` under the
kinematic model.

## Reproducing

```
venv/bin/python -m src.segmentation.algorithms.accelerometer_only.\
template_match.fit_trapezoid_pulses
```

Run time ≈ a minute per experiment on a 2020s laptop (450 templates × ~20
rides). Writes land here under the ride's experiment-named folder.
