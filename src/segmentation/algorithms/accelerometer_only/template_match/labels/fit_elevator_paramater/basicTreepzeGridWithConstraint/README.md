# `basicTreepzeGridWithConstraint/` — shared-shape per-ride trapezoid fit

Outputs of
`src/segmentation/algorithms/accelerometer_only/template_match/fit_elevator_parameters/constrained_grid.py`.
One folder per TRAIN experiment; `_all_rides.png` + `parameters.json`
inside each, plus the usual similarity / symmetry grids at the root.

## Why another variant?

The sibling `basicTreepzeGrid/` fits each of the two lobes of a ride
**independently**, so lobe 1 (take-off) and lobe 2 (landing) can come
out with different amplitudes, widths, or plateau fractions. Physically
the two pulses are the same pulse up to sign — a noisy landing lobe
should not drag away from the take-off's shape.

This variant enforces that directly:

    lobe 1 :  a ≈  +s · |A| · trapezoid(t − t_c1; W, f)
    lobe 2 :  a ≈  −s · |A| · trapezoid(t − t_c2; W, f)

where `s = +1` for up rides and `−1` for down rides. The shape `(W, f)`
and magnitude `|A|` are shared across the two lobes; only `t_c` is
allowed to differ.

## Selection rule

For every `(W, f)` on the same 30 × 15 grid used by `basicTreepzeGrid`:

1. Slide the unit kernel and read `inner[i] = ⟨a, tpl⟩` and
   `power[i] = ⟨a, a⟩` on the ±W window around each center.
2. Restrict lobe 1 to `LOBE1_REGION` with the correct sign, lobe 2 to
   `LOBE2_REGION`.
3. For every remaining `(i1, i2)` pair the least-squares optimal shared
   magnitude is `|A|* = (s₁·inner[i1] + s₂·inner[i2]) / (2·norm_t)`,
   and the per-lobe local R² under that constraint is
   `R²_k = 1 − (P_k − 2·|A|*·s_k·inner[i_k] + |A|*²·norm_t) / P_k`.
4. Score each pair by `mean(R²_1, R²_2)` and keep the argmax across
   all `(W, f, i1, i2)`.

The grid search is fully exhaustive (outer product per `(W, f)`), so
we don't rely on any top-K heuristic.

## Trade-off vs. `basicTreepzeGrid`

Forcing shape sharing reduces per-lobe R² slightly (constrained median
≈ 0.96 vs. basic ≈ 1.00 on the TRAIN set), in exchange for a
physically-coherent per-ride trapezoid. The within-segment `avg pair
RMS` (see `_trapezoid_similarity_grid.png`) drops from ~0.47 m/s² to
~0.43 m/s², and the per-ride `|lobe1|` vs. `|lobe2|` RMS is exactly 0.00
m/s² by construction (see `_trapezoid_lobe_symmetry_grid.png`).

## `parameters.json` schema

Same as `basicTreepzeGrid/parameters.json`. By construction
`|lobe1.a_peak| == |lobe2.a_peak|`, `lobe1.half_width_s ==
lobe2.half_width_s`, and `lobe1.frac_flat == lobe2.frac_flat`; signs of
`a_peak` differ per the ride-type convention above. `r2_local` is the
constrained per-lobe R² (not the unconstrained one).

## Reproducing

```
venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/fit_elevator_parameters/constrained_grid.py
```

Run time ≈ 15 s on a 2020s laptop (exhaustive pair search is
vectorised per `(W, f)`).
