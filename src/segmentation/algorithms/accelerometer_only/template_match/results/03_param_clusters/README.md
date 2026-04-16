# 03 — Fit parameter clusters

For each shape family, scatter plots across all rides labeled with that family. Goal: see whether rides share a canonical template (tight cluster) or spread across parameter space.

## Trapezoid model
`v(t) = clip(a_max · min(t−t_start, t_end−t), 0, v_max)`
- **a_max** — ramp slope (m/s²)
- **v_max** — plateau velocity (m/s)
- **W**     — total duration (s)
- Derived: `plateau = W − 2·v_max/a_max`

## Generalized parabola model
`v(t) = v_peak · max(0, 1 − ((t−t_c)/W)²)^p`
- **v_peak** — peak velocity (m/s)
- **W**      — full width root-to-root (s)
- **p**      — shape exponent (p=1 pure parabola, p<1 flatter, p>1 sharper)

## Files

- `param_clusters_trapezoid.png` — 32 trapezoid-labeled rides; 3D scatter over (a_max, v_max, W) plus three 2D projections.
- `param_clusters_parabola.png` — 51 parabola-labeled rides; 3D scatter over (v_peak, W, p) plus three 2D projections.

## Takeaways
- `a_max` is tightly clustered (mean 0.40 m/s², std 0.13 for trapezoids). Elevators accelerate in a narrow range.
- `W` spans 3 s to 44 s — rides cluster by **floor count**, not by machine kinematics.
- `plateau vs W` separates rides into triangle-like (plateau≈0) and true trapezoid (plateau up to 30 s).
- Parabola rides are mostly short (W≈5-10 s) with p slightly above 1 (sharper than pure parabola).

**Script:** `../../scripts/param_clusters.py`
