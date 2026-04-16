# 01 — Fit strategies

Each ride's LPF velocity curve is fit with two parametric models:
- **Trapezoid / triangle** — `v(t) = clip(a_max · min(t−t_s, t_e−t), 0, v_max)` (4 params).
- **Generalized parabola** — `v(t) = v_peak · max(0, 1 − ((t−t_c)/W)²)^p` (4 params with shape exponent `p`).

Both fits minimize MAE (Nelder-Mead). Six scoring strategies were then tried to decide which shape wins per ride:

| Strategy | Idea |
|---|---|
| A | MAE on active region only (|v| > 10% peak) |
| B | MAE + AIC parsimony penalty (k/N) |
| C | Force trapezoid if plateau ≥ threshold |
| D | MAE on shape-normalized curves (divide both by peak) |
| E | Preprocess: detect plateau → force trapezoid |
| F | Penalize parabola with p<0.8 (plateau-mimic) |

## Files

- `fit_strategy_A_active_mask_oria.png`
- `fit_strategy_B_aic_oria.png`
- `fit_strategy_C_triangle_vs_par_oria.png`
- `fit_strategy_D_normalized_shape_oria.png`
- `fit_strategy_E_plateau_gate_oria.png`
- `fit_strategy_F_bounded_p_oria.png`

Each PNG shows the 44 oria rides as a grid; black = trapezoid fit, purple dashed = parabola fit, yellow = GT window. The annotation box reports the scoring-strategy winner per ride and its parameters.

**Script:** `../../scripts/compare_fit_strategies.py`
