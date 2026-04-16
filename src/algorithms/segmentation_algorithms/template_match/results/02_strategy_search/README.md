# 02 — Strategy search vs visual labels

Used visual ground truth: every GT ride's velocity curve was viewed and labeled `trapezoid` or `parabola` (see `../../labels/labels.txt`, 83 rides). Then searched over a grid of scoring-strategy thresholds and 2-/3-way OR/AND combos to find the configuration with highest agreement against the visual labels.

## Best result

- **95.2% agreement (79/83)** with hybrid rule:
  > If `plateau detected (peak_tol=0.10, min_sec=0.3)` → **trapezoid**. Else use `B_aic (α=2.0)`.

The hybrid combines a plateau preprocessing gate with an AIC-penalized MAE comparison — the gate catches trapezoids with real plateaus even when the flexible parabola (p<1) would otherwise mimic them, while AIC keeps parsimony honest elsewhere.

## Files

- `strategy_search_results.md` — top-15 table of (strategy, thresholds, agreement %) and full disagreements list for the best configuration.
- `best_strategy.png` — 83-tile grid of all rides with both fits overlaid; green border = matches label, red border = disagrees.

**Script:** `../../scripts/strategy_search.py`  
**Labels:** `../../labels/labels.txt`
