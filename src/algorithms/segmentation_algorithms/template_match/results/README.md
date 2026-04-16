# Template-match experiments — index

Chronological order of the experiments:

1. **`01_fit_strategies/`** — fit each GT ride with trapezoid and parabola models; try 6 scoring strategies to decide which shape wins per ride.
2. **`02_strategy_search/`** — use visual labels as ground truth; grid-search and combo strategies until one reaches 95% agreement with the labels.
3. **`03_param_clusters/`** — visualize the fitted parameters for each shape family. Are rides clustered (shared template) or spread?
4. **`04_pulse_detection/`** — pick representative templates from the clusters and try to locate rides in full sessions with a matched filter + gated detection.
5. **`05_acc_matched_filter/`** — try an amplitude-preserving (template-energy-only) matched filter to fix the NCC "everything-correlates" problem.

Each subfolder has its own `README.md` with context, files, and takeaways.

## Overall finding

Shape fitting per ride works very well (95% agreement). **Detection in full sessions is still unsolved with matched filtering alone** — see `04_pulse_detection/README.md`.
