# Segmentation metrics

`SegmentationMetrics` in `metrics.py` evaluates a predicted segment DataFrame
against a ground-truth DataFrame. Both use the common CI schema:

```
start_ci       : (low, high)   tuple; point estimate is midpoint
end_ci         : (low, high)   tuple; point estimate is midpoint
probability_ci : (p_lo, p_hi)  tuple; point estimate is midpoint
duration, type
```

---

## Detection — `match_segments(pred, gt)`

**What it measures.** Did the detector find the right number of rides at the
right times?

**Success criterion (containment).** A predicted segment is a **true positive**
iff **exactly one GT segment lies fully inside it**:

```
pred.start_ci.lo <= gt.start_midpoint
gt.end_midpoint  <= pred.end_ci.hi
```

- If a predicted segment contains **two or more** GT segments, it's wrong —
  the detector merged multiple rides into one. Counts as **false positive**.
- If a predicted segment contains **zero** GT segments, it's a hallucination.
  Also **false positive**.
- A GT segment not contained by any prediction is a **false negative**
  (missed ride).

Why containment instead of IoU? IoU rewards "kind of close" matches even
when a single prediction smears across many real rides. For elevator
segmentation we want a strict 1-to-1 correspondence — one predicted segment
per actual ride. Containment enforces that.

**Reports:**

| field | meaning |
|---|---|
| `tp` | predictions that each cover exactly one GT |
| `fp` | predictions that cover 0 or ≥ 2 GTs |
| `fn` | GT segments not covered by any prediction |
| `precision` | tp / (tp + fp) — fraction of predictions that are correct |
| `recall` | tp / (tp + fn) — fraction of real rides we caught |
| `f1` | harmonic mean of precision and recall |
| `mean_iou` | informational: mean IoU over matched pairs |

---

## Calibration — `ece`, `brier`, `reliability_bins`

**What they measure.** When the model says "0.8", is it actually right 80%
of the time?

### `ece(probs, labels, n_bins=10)` — Expected Calibration Error

Reference: Guo, Pleiss, Sun, Weinberger (2017), *On Calibration of Modern
Neural Networks*.

Bucket predictions into `n_bins` confidence bins. For each bin, compare the
bin's mean predicted probability (`conf`) with its empirical accuracy
(`acc`). ECE is the weighted average of `|acc − conf|`:

```
ECE = Σ_b (n_b / N) · |acc_b − conf_b|
```

- **0** = perfectly calibrated (the model's confidence matches its accuracy).
- **≤ 0.05** = well-calibrated for practical purposes.
- **≥ 0.15** = significant miscalibration.

### `brier(probs, labels)` — Brier Score

Reference: Brier (1950), *Verification of forecasts expressed in terms of
probability*.

```
Brier = (1/N) · Σ (p_i − y_i)^2
```

A proper scoring rule — penalizes both bad calibration and bad
discrimination in one number. Range `[0, 1]`; lower is better. Unlike ECE,
Brier doesn't need binning.

### `reliability_bins(...)` — data for a reliability diagram

Returns per-bin `{mean_conf, empirical_acc, count}` so you can plot a
reliability diagram (Niculescu-Mizil & Caruana 2005). Perfect calibration
is the diagonal; bars below the diagonal mean overconfident, above means
underconfident.

---

## Interval coverage

The segmenter outputs intervals (Venn–Abers probability CI, conformal edge
CIs). Coverage measures whether these CIs actually contain the truth at the
promised rate.

### `prob_ci_coverage(labels, p_lo, p_hi)`

Fraction of segments where the binary label (0 or 1) falls inside the
probability CI `[p_lo, p_hi]`. A well-calibrated 90% CI should have
coverage ≥ 0.90 on held-out data.

### `time_ci_coverage(residuals_sec, half_width_sec)`

Fraction of predictions where the absolute timing error is within the
reported half-width. For a 90% conformal CI (`α=0.1`), this should be
≥ 0.90 by construction on the calibration set, and close to that on new
data under exchangeability (Vovk, Gammerman, Shafer 2005).

---

## Putting it together — `summary(...)`

Returns a single dict:

```python
{
    "detection": {
        "tp": int, "fp": int, "fn": int,
        "precision": float, "recall": float, "f1": float,
        "mean_iou": float, "iou_threshold": float,   # iou_threshold kept for back-compat
    },
    "calibration": {
        "ece": float, "brier": float, "n": int,
        "prob_ci_coverage": float,        # when p_lo/p_hi given
    },
    "edge_start_ci_coverage": float,      # when start residuals + start_q given
    "edge_end_ci_coverage":   float,
}
```

---

## Small glossary

- **TP / FP / FN** — true positive, false positive, false negative.
- **IoU** — Intersection over Union for two intervals.
- **Calibration** — the agreement between predicted probability and observed
  frequency.
- **Coverage** — the frequency with which a CI contains the true value.
- **Proper scoring rule** — a scoring rule minimized in expectation by
  reporting the true probability (Brier and log-loss are proper).

## References

- Brier, G. W. (1950). *Verification of forecasts expressed in terms of
  probability*. Monthly Weather Review.
- Guo, Pleiss, Sun, Weinberger (2017). *On Calibration of Modern Neural
  Networks*. ICML.
- Niculescu-Mizil, Caruana (2005). *Predicting good probabilities with
  supervised learning*. ICML.
- Vovk, Gammerman, Shafer (2005). *Algorithmic Learning in a Random World*.
- Vovk, Petej, Fedorova (2014). *Large-scale probabilistic predictors with
  and without guarantees of validity*. NeurIPS.
