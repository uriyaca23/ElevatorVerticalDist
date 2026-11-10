# Writing guide — `phd_imwut_main.tex`

How this paper is written, the rules to follow, how the appendix works, and
a log of what has been done. **Read this before editing the `.tex` file.**

---

## 1. What this paper is

- **Title:** *Smartphone-Only Estimation of Elevator Vertical Displacement
  with Calibrated Confidence Intervals*
- **Author:** Eyal Yakir
- **Venue:** PACM IMWUT (Proceedings of the ACM on Interactive, Mobile,
  Wearable and Ubiquitous Technologies).
- **Class:** `\documentclass[manuscript,review,screen]{acmart}` — single
  column, double-spaced, line numbers for reviewers. After acceptance,
  switch to `\documentclass[acmlarge,screen]{acmart}`.
- **Core claim:** the mechanical limits of an elevator cabin force the
  gravity-removed acceleration of a ride into a small parametric family of
  basis functions; the pipeline fits a candidate ride against that family
  to estimate signed vertical displacement Δh with a calibrated 90 % CI,
  using the phone's accelerometer alone.

---

## 2. Process — use the `paper-writing` skill

Every edit to a `.tex` file in this directory **must** go through the
`paper-writing` skill (it triggers automatically on `.tex` work). It encodes
the SNL/Arpit-Gupta editorial methodology. The non-negotiable parts:

- **Mandatory style audit (GATE).** Before presenting or committing *any*
  new/modified `.tex` prose, audit every changed sentence and fix:
  negation-first phrasing, throat-clearing, hedging, generic adjectives,
  sentences > 40 words, passive voice, missing citations.
- **Introduction-twice principle.** The introduction is written twice — a
  disposable Draft 0 first (framing scaffold), then rewritten from scratch
  *after* the evaluation exists so it promises exactly what the evidence
  supports.
- **Section drafting order:** Draft 0 Intro → Evaluation → Design/Method →
  Background → Related Work → Final Intro → Abstract.
- **Compression last:** target 30–50 % reduction from first draft. Do not
  pad to fill page limits.

---

## 3. Voice and editorial rules

These govern every sentence. (Full detail lives in the skill's
`author_profile/`; this is the working subset.)

- Mean sentence length ~21 words; hard max ~40 (contribution lists only).
- **Topic sentences assert claims.** Never open a paragraph with background.
- **Zero hedging.** "We show", not "we believe". State numbers, not "may".
- **Active voice everywhere.** No exceptions.
- **No filler adjectives:** never "novel", "significant", "state-of-the-art",
  "comprehensive", "robust", "substantial", "promising". Use a number or cut.
- **Section/subsection openers** state what the section concludes — never
  "In this section we describe…".
- Paragraphs are 4–6 sentences; each one makes a claim, gives evidence, or
  synthesises a takeaway.
- **Named over vague:** every mechanism/metric/baseline gets a proper name.
- **Interpret figures, don't just cite them.**
- British spelling is used throughout (`parameterised`, `normalise`,
  `colour`) — stay consistent.
- No exclamation marks; no rhetorical questions outside the introduction.

---

## 4. Document structure

| § | Title | Label | State |
|---|-------|-------|-------|
| 1 | Introduction | `sec:intro` | **Written** (Draft 0 — will be rewritten after Evaluation) |
| 2 | Background: Elevator Kinematics | `sec:background` | **Written** |
| 2.1 | How a passenger elevator works | `sec:bg-mechanics` | Written |
| 2.2 | Physical limitations and what they bound | `sec:bg-bounds` | Written |
| 2.3 | Why the ride is an S-curve | `sec:bg-scurve` | Written |
| 2.4 | From machine parameters to ride-time constants | `sec:bg-time-constants` | Written |
| 2.5 | The trapezoid basis function and the parametric family | `sec:bg-family` | Written |
| 2.6 | The smartphone accelerometer | `sec:bg-imu` | Written |
| 3 | Algorithm | `sec:algorithm` | **Written** |
| 3.1 | Segmentation | `sec:algo-seg` | Written (Detecting candidate lobes `sec:algo-detect` / Pairing candidate lobes into rides `sec:algo-pair`) |
| 3.2 | Prediction | `sec:algo-pred` | Written (ZUPT double integration `sec:algo-zupt` / Trapezoid pulse-pair fit `sec:algo-trapezoid` / Theoretical error analysis / Confidence interval / Quality filter) |
| 4 | Dataset and Ground Truth | `sec:dataset` | **Written** (counts are open `\todo`s) |
| 5 | Evaluation | `sec:evaluation` | **STUB (empty)** |
| 6 | Discussion | `sec:discussion` | **STUB (empty)** |
| 7 | Conclusion | `sec:conclusion` | **STUB (empty)** |
| A | Theoretical Elevator Pulse Analysis | `app:derivations` | Written (subsections `app:bangbang`, `app:time-params`) |
| B | Dataset collection and ground-truth pipeline | `app:dataset` | Written |
| C | Segmentation: matched-filter & pair-filter details | `app:segmentation` | Written |
| D | Prediction: error budget & conformal calibration | `app:prediction` | Written |

Section narrative arc: Background derives the **family** → §3 Algorithm
segments hours-long streams with it and predicts each ride's Δh with it →
§4 presents the dataset and ground truth → §5 evaluates → §6/§7 close.

---

## 5. How the appendix is used

**Principle: the main text states results and intuition; the appendix
carries the heavy algebra.** A reader who trusts the result never has to
visit the appendix; a reader who wants the derivation finds it complete
there.

- **Main text** = the *end formula*, the physical intuition, and a figure.
  Then one sentence: "Appendix A derives this."
- **Appendix** = the full step-by-step derivation.
- **Equation labels live in the main text.** The appendix `\eqref{}`s back
  to them (e.g. `app:derivations` references `eq:Tj`, `eq:Ta`, `eq:Tv`,
  `eq:elevator-durations`). So: never move a *labelled* equation into the
  appendix if the main text or another section cites it — keep the labelled
  equation in the body and derive it in the appendix.
- **Appendix A** (`app:derivations`, "Theoretical Elevator Pulse Analysis")
  — two subsections. **A.1** (`app:bangbang`) tightens the
  Pontryagin-maximum-principle argument that time-optimal jerk is
  bang-bang (`eq:bang-bang`) to three moves: Hamiltonian, switching law,
  constrained arcs. **A.2** (`app:time-params`) does the phase-by-phase
  integration for the ride-time constants of §2.4 with displayed math
  and a colour-keyed TikZ figure (`fig:trap-geometry`, red = $T_j$
  contributions, orange = $T_a$ plateau, teal = $T_v$ cruise — colours
  match `fig:ride-time-constants`); ends with the centroid-spacing
  identity used in §2.5.
- **Appendix B** (`app:dataset`) — the field campaign and ground-truth
  pipeline behind §4: buildings/sensor tables, barometer interval detector,
  cross-device time sync, gramushka snapping, GT editor, plus four raster
  data figures.
- **Appendix C** (`app:segmentation`) — the matched-filter least-squares
  solution, the `(W,f)` template grid, the shared-shape joint score
  (`eq:joint-fit`), and the calibrated detection/pair-filter thresholds
  (`tab:seg-thresholds`). Body §3.1 carries only the result + intuition.
- **Appendix D** (`app:prediction`) — five subsections behind §3.2's two
  prediction algorithms. **D.1** the trapezoid pulse-pair integral
  (`eq:trap-area` → `eq:dh-closed`, the joined-pulse variant `eq:dh-joined`,
  the S-curve area agreement `eq:scurve-trap-area`, TikZ `fig:trap-scurve`).
  **D.2** the ZUPT error budget — the four terms of `eq:sigma-zupt-total`.
  **D.3** the trapezoid error budget — Cramér–Rao bound (`eq:crb`),
  delta-method gradients (`eq:dh-grad`) of `eq:sigma-dh`, data-adaptive
  effective noise, velocity-anchored amplitude variance, overlap inflation
  (`eq:overlap-inflation`). **D.4** split-conformal coverage and the
  per-algorithm calibration protocol for `eq:conformal-coverage`. **D.5**
  the quality-filter checks — two tables (`tab:quality-checks-zupt`,
  `tab:quality-checks`) plus a per-feature paragraph and a real-recording
  good/bad figure (`fig:qf-*`, nine PNGs from
  `scripts/figs/plot_quality_features.py`).

When a section gets equation-heavy, push the derivation to a new appendix
paragraph and leave the body with the result + a pointer.

---

## 6. LaTeX conventions in this paper

- **Placeholders:** `\todo{...}` renders bold red. There are ~12 open
  `\todo`s (experiment counts, building counts, CI numbers). Resolve them
  with real numbers from the evaluation; never ship a `\todo`.
- **Schematics are TikZ.** Every schematic/concept figure is hand-drawn
  TikZ — keep new schematics as TikZ. Raster assets in `figures/` are the
  body schematic (`elevator_schematic.png`) and the Appendix C data
  figures (`building_heights*.png`, `forbarometer_alignment.png`,
  `gramushka_drawing.png`, `gt_editor.png` — real plots, photos, and
  screenshots, which belong as raster).
- Loaded TikZ libraries: `decorations.pathreplacing`, `arrows.meta`.
  **`positioning` is NOT loaded** — do not use `below=of …`; place nodes at
  explicit coordinates.
- Loaded packages of note: `graphicx`, `tikz`, `subcaption`, `wrapfig`,
  `xcolor`. `graphicspath` is `{figures/}`.
- **Notation (keep consistent):**
  - `\theta = (j_max, a_max, v_max, H)` — ride parameters (`eq:elevator-params`).
  - `j_max, a_max, v_max` — building constants; `H` — floor-to-floor height
    (the only ride-specific quantity).
  - `T_j, T_a, T_v, T_ride` — ride-time constants (`eq:Tj`…`eq:elevator-durations`).
  - `\tau_{W,f}` — trapezoid basis function (`eq:trap-template`), half-width
    `W`, flat fraction `f`.
  - `\mathcal{F}_ride` — the parametric ride family (`eq:ride-template`).
  - `a(t)` — gravity-removed *vertical* acceleration; `Δh` — signed
    displacement; `R` — body→world rotation; body axes `x,y,z` vs world
    axes `x_w,y_w,z_w`, world vertical `\hat z_w`.
  - `v(t)` — integrated vertical velocity; `T` — ride duration; `N` —
    ZUPT active-sample count (the error-model scale).
  - `W,f,A,s,\Delta t_c` — the fitted trapezoid pair: half-width, flat
    fraction, amplitude, direction, centroid spacing.
  - `\sigma` — per-ride std; `\sigma_white,\sigma_mech,\sigma_rel,\sigma_proj`
    — the four ZUPT error terms; `k` — conformal multiplier.
- Key equation labels: `eq:elevator-bounds` (1), `eq:elevator-opt` (2),
  `eq:bang-bang` (3), `eq:Tj`/`eq:Ta`/`eq:Tv` (4–6),
  `eq:elevator-durations` (7), `eq:elevator-params` (8),
  `eq:trap-template` (9), `eq:W-from-kinematics`/`eq:f-from-kinematics`
  (10–11), `eq:ride-template` (12). §3.1 adds `eq:matched-filter` (13),
  `eq:signed-r2` (14), `eq:detect-gate` (15), `eq:partner-set` (16),
  `eq:joint-fit` (17), `eq:pair-gate` (18), `eq:grid-energy` (19),
  `eq:pair-rank` (20). §3.2 adds `eq:zupt-integral` (21),
  `eq:dh-closed` (22), `eq:sigma-zupt` (23), `eq:sigma-zupt-total` (24),
  `eq:sigma-dh` (25), `eq:nonconformity` (26), `eq:conformal-coverage` (27).
  Appendix E adds `eq:trap-area`, `eq:dh-joined`, `eq:scurve-trap-area`,
  `eq:crb`, `eq:dh-grad`, `eq:overlap-inflation` (appendix equation
  numbers run continuously after Appendices A–D, so they are not
  hand-tracked here; every cross-reference uses `\eqref`).
- `\scurvepanel` — a `\newcommand` defined in §2.3 that draws one
  acceleration S-curve panel; reused by the three-regime figure.

---

## 7. Figure rules (learned the hard way)

Figure placement in this double-spaced, equation-dense manuscript is
fragile. Follow these:

- **Never put a `wrapfigure` next to display equations.** Display math
  ignores the wrap and overruns the figure. An equation-dense section
  (e.g. §2.5) must use a normal `figure` float, or a `wrapfigure` placed
  beside a stretch of *pure prose* with no `equation`/`align` in its span.
- **A `wrapfigure` must be no taller than its host paragraph(s).** If it is
  taller, `wrapfig` pads with blank space — the "gaps between paragraphs"
  problem. Either lengthen the host text, shorten the figure, or switch to a
  normal float.
- **Keep captions short (1–2 lines).** In a narrow `wrapfigure` column a
  long caption balloons the figure's height.
- **Side-by-side or comparison figures** → one `figure` with two
  `subfigure` panels (`subcaption` is loaded). This also dodges all
  wrap-gap issues.
- **Make sibling figures visually distinct.** Two near-identical diagrams
  read as a mistake (e.g. §2.6 pairs a *geometric* phone diagram with a
  *flow/block* diagram).
- TikZ sizing: prefer `scale=` on the `tikzpicture` (coordinates scale,
  font does not) over `\resizebox` (which scales the font too, often too
  small). Match the chosen scale to the target width.

User preferences observed: figures should be **small and on the side**
where layout allows; **prose concise**; recompile and open the PDF after
each change.

---

## 8. Building and checking

No `latexmk`. From `paper_phd/`:

```bash
pdflatex -interaction=nonstopmode -halt-on-error phd_imwut_main.tex   # ×2
# add bibtex between passes if citations changed:
#   pdflatex … ; bibtex phd_imwut_main ; pdflatex … ; pdflatex …
```

After compiling, check:
- `grep -i "undefined" phd_imwut_main.log` — no undefined refs/citations.
- `grep -i "overfull" phd_imwut_main.log` — no new overfull boxes.
- Open `phd_imwut_main.pdf` and read the changed pages.

Bibliography: `\bibliography{references}` → `references.bib` in this folder.

---

## 9. Work log

Sessions so far have rewritten the Background (§2):

- **§2.3** — redesigned the time-optimal control problem `eq:elevator-opt`
  (aligned `minimize`/`subject to` layout). Replaced the verbose
  seven-segment prose + the standalone phase formula + the bulleted
  three-regime list with one compact figure (`fig:scurve-durations`):
  three rows, each pairing an acceleration-profile panel with its
  `(T_v,T_a)` condition and a one-line explanation.
- **§2.4** — stripped the inline algebra (it already lived in Appendix A).
  The body now states only the end formulas (`T_j,T_a,T_v` in an `align`,
  `T_ride` in `eq:elevator-durations`) plus a small `wrapfigure`
  (`fig:ride-time-constants`) that labels the four durations on one
  acceleration trace.
- **§2.5** — rewrote concisely (~35 % shorter); folded the redundant
  "three regimes in (W,f)" paragraph into one clause. The trapezoid
  figure (`fig:trap-template`) is a small `wrapfigure` placed beside the
  pure-prose "three facts" paragraph (the only equation-free stretch).
- **§2.6** — renamed "Smartphone accelerometer hardware" → "Smartphone
  sensors". Three paragraphs: *the accelerometer* (3-axis body-frame
  stream), *the unknown vertical* (the body-vs-world rotation problem),
  *the orientation sensor* (how the gyro/accel/magnetometer fusion
  produces `R`). One `figure` float, `fig:phone-sensors`, with two
  distinct panels: `fig:accel-frame` (geometric) and `fig:orient-fusion`
  (flow diagram).

- **§5 Dataset and Ground Truth** — wrote the half-page data summary as
  its own top-level section (promoted from a subsection of Evaluation;
  Evaluation is now §6): the field-campaign size (counts left as
  `\todo`), barometer-derived ground truth (ISA inversion,
  `up`/`down`/`outside` labelling), and the *gramushka* floor-snapping
  that turns drifting barometric altitude into discrete floor-to-floor
  `\Delta h`. One small TikZ `wrapfigure` (`fig:gt-snap`) shows the
  snap. The heavy detail — buildings/sensor tables, the barometer
  detector, the boot-time/cross-correlation time-sync fix, snapping
  internals, the GT editor — went to **Appendix C** (`app:dataset`).
  Source material: `docs/latex/main.tex` §Data Collection and
  `src/data/loader/gramushka.py`.
- **Appendix C figures** — added four raster figures so a reader fully
  sees the data: `fig:app-buildings` (height range + per-building floor
  profiles, two subfigures), `fig:app-alignment` (a recording session
  after cross-device time-sync), `fig:app-gramushka` (a real gramushka
  architectural drawing), `fig:app-gt-editor` (the GT-editor UI). Files
  copied into `paper_phd/figures/` from `docs/latex/figures/` and
  `src/data/gramushka/`. NOTE: the appendix prose is short, so each
  large float lands one-per-page — generous whitespace on the appendix
  pages. Compact later if needed (group floats into subfigures).

- **§3 Algorithm (new)** — removed the two empty stub sections
  (Building-Level Inventory, Orientation-Sensor Addon) and replaced
  them with a real §3 Algorithm distilled from `docs/latex/main.tex`
  and `docs/latex/prediction_sections.tex`. §3.1 Segmentation
  (Idea / Detection / Pairing algorithm) and §3.2 Prediction
  (Idea / Prediction algorithm / Theoretical error analysis /
  Confidence interval / Quality filter). Idea-first, final algorithm
  only; the heavy maths went to new Appendices D and E. Four §3
  figures: a TikZ pipeline overview (`fig:pipeline`) plus three raster
  assets copied from `docs/latex/figures/` (`fig:signed-r2`,
  `fig:pair-filter` two subfigures, `fig:triangle-trapezoid`).
  Repaired all prose the stub removal
  broke: abstract orientation sentence, intro two-capabilities
  paragraph (now one per-ride goal), contributions (5→3), roadmap,
  §2.5 fact (i), §2.6 (renamed "The smartphone accelerometer", dropped
  the orientation-sensor paragraph and `fig:orient-fusion`,
  `fig:accel-frame` now a small `wrapfigure`), Appendix C sensor-table
  caption, keyword list. ZUPT appears only as the internal
  velocity-anchor integration step. Added one citation
  (`lei2014split`, split conformal). Building-inventory and
  orientation-addon were removed cleanly, not archived in the `.tex`;
  re-add from `docs/latex/main.tex` if ever needed.

- **§3.1 Segmentation rewrite** — restructured for a linear read.
  Dropped the *Idea* subsubsection (its motivation folded into the
  §3.1 opener); §3.1 has exactly two subsubsections,
  *Detecting candidate lobes* (`sec:algo-detect`) and *Pairing
  candidate lobes into rides* (`sec:algo-pair`). Added
  `\usepackage{float}` and changed `fig:pipeline` to `[H]` so the
  pipeline overview sits in-flow inside §3 rather than floating to the
  page top. Appendix D's joint-score paragraph now derives and
  `\eqref`s the body `eq:joint-fit`. NOTE: heatmap/grid energy is a
  pair-level gate, never per-point — confirmed against
  `src/segmentation/.../pair_filter.py`.

- **§3.1 condensed (math-forward)** — rewrote §3.1 much shorter and
  equation-driven for the journal: removed the `fig:r2-heatmap`
  figure, dropped all `\paragraph` subheads, and cut the prose ~50 %.
  §3.1 now reads as seven displays — matched filter `eq:matched-filter`,
  signed maxima `eq:signed-r2`, detection floors `eq:detect-gate`,
  partner set `eq:partner-set`, shared-shape fit `eq:joint-fit`,
  admissibility gates `eq:pair-gate`, greedy rank `eq:pair-rank` —
  with terse connective prose. `eq:quiet-middle` is gone (folded into
  `eq:pair-gate`); the pair score is now written `S(i,j)`, matched in
  Appendix D's gate paragraph and threshold table. §3.1 shrank from
  ~3 pages to ~1.3; the paper is now 20 pages.

- **§2--3 figure cleanup** — removed `fig:pair-filter` (the two
  `pair_filter_timeline_*.png` subfigures, whose PNGs carried raw
  matplotlib paths and read as buggy) and its `\ref` in the §3.1
  greedy-resolver paragraph. Shrank `fig:signed-r2` to
  `0.7\linewidth`. Kept `fig:accel-frame` (§2.6) a small side
  `wrapfigure`, anchored at the §2.6 opening — the highest anchor in
  the section and the only one that starts far enough up page 6 for
  the figure to fit without spilling its caption into the bottom
  margin; caption tightened to one line. Added one sentence to §3.1
  noting
  that `(W*,f*,A*)` is the *average* lobe — the shared-shape fit
  assumes take-off and landing share a shape.
- **§3.1 grid energy made explicit** — replaced the verbal
  definition of the pair-filter grid energy `E` with the formula
  `eq:grid-energy` (mean of `max(0, joint R^2)` over the `(W,f)`
  grid `G`, confirmed against `pair_filter.py`). Added an empty
  Appendix D placeholder (`app:thresholds`, a `\todo`) for
  per-threshold explanations and examples, linked from §3.1.

- **§3.2 Prediction condensed (math-forward)** — rewrote §3.2 to
  match the condensed §3.1. Dropped the *Idea* subsubsection and
  folded its motivation into the §3.2 opener (now two paragraphs —
  the dispatcher contract + roadmap, then the why-trapezoid
  argument). Replaced the five `\textbf{(1)–(5)}` steps of
  *Prediction algorithm* with three paragraphs of equation-driven
  prose around `eq:dh-closed`. Trimmed the error-analysis
  "two corrections" paragraph (it duplicated Appendix E) and the
  quality-filter prose. The quality-filter checks moved to a new
  Appendix E table `tab:quality-checks` (10 rows, thresholds taken
  from `trapezoid_accel/quality.py` + `configTypes.py`) with a short
  explanatory paragraph; body §3.2.4 now states the filter compactly
  and `\ref`s the table. Added `\label{sec:algo-quality}` to the
  Quality-filter subsubsection (referenced from the opener). Kept
  `fig:triangle-trapezoid` in the body; dropped its redundant caption
  sentence. §3.2 shrank ~1 page; the paper is now 19 pages.

- **Appendix D `app:thresholds` filled** — replaced the placeholder
  `\todo` with a per-threshold walk-through: a lead-in plus seven
  `\paragraph`s covering all ten rows of `tab:seg-thresholds`
  (detection floors, peak dedup + same-sign gap, then Gates 1–6).
  Corrected four stale Table 3 values to the deployed `DetectConfig`
  defaults (shape floor 0.40, NMS 1.0 s, ride window [0,30] s, joint
  score 0.90) and trimmed the now-impossible 120 s super-pair
  anecdote from the *Pair-filter gates* paragraph. The quiet-middle
  gate is documented as the implemented RMS-ratio test with a clearly
  informal angle aside (no degree threshold). Three data figures
  added (`fig:thresh-energy`, `fig:thresh-cruise`, `fig:thresh-rank`)
  from a new `scripts/figs/plot_threshold_examples.py` that scans the
  dataset for example recordings (`--scan-only` prints shortlists,
  `OVERRIDES` pins the chosen ones). Cited the existing
  `strakosch2010` and `cibseguidd2020` for ride durations — a bibtex
  pass is needed because `strakosch2010` was newly cited.

- **Appendix D restructured (table-first, condensed, in-context
  figures)** — moved `tab:seg-thresholds` to the top of Appendix D
  (right under the intro) and added `\clearpage` before the section so
  the appendix starts on a fresh page with the table at the top; the
  pending Appendix C raster floats now flush cleanly before it. Cut
  the per-threshold prose ~25\,\% (dropped the Gate 5 angle aside, the
  redundant *Pair-filter gates* paragraph, and the inline j/a/v
  numbers). Pinned the three threshold figures to `[H]` so each sits
  under its gate rather than floating to a page top. **Gate 6 figure
  redesigned** (`scripts/figs/plot_threshold_examples.py`,
  `render_duration_penalty`): a 2-panel `threshold_duration_penalty.png`
  --- acceleration trace with the four lobes and ride/super-pair spans,
  plus the matched-filter signed-$R^2$ score trace --- with a box
  comparing $S$ and rank $S-\lambda\Delta t$ for the three candidate
  pairs. Lobe signs now read from each lobe's `a_peak` (not hard-coded),
  and the picker prefers up-going multi-stop trips; the pinned example
  is `milleniumHotel … exp2`, where all three pairs score $S\approx1$
  so only the duration penalty separates them. Gate 6 prose rewritten
  to match; the stale "consume the lobes / ride-by-ride partition"
  sentence is gone.

- **§3.2 Prediction re-expanded as two co-equal algorithms** — the
  condensed single-pipeline §3.2 was rewritten to present ZUPT and the
  trapezoid fit honestly side by side, matching the code (the
  `PredictAlgorithm` enum carries `ZUPT_ACCEL` and `TRAPEZOID_ACCEL` as
  independent estimators). §3.2 is now an opener plus five
  subsubsections: *ZUPT double integration* (`sec:algo-zupt`,
  `eq:zupt-integral`), *Trapezoid pulse-pair fit* (`sec:algo-trapezoid`,
  `eq:dh-closed`), *Theoretical error analysis* (both budgets —
  `eq:sigma-zupt`, `eq:sigma-zupt-total`, `eq:sigma-dh`), *Confidence
  interval* (per-algorithm conformal calibration), and *Quality filter*
  (both filters). The trapezoid fit is stated to borrow only ZUPT's
  cruise-velocity scale, not its Δh, so §6's comparison stays meaningful.
- **Appendix E restructured into E.1–E.5** — E.1 derives `eq:dh-closed`
  from the trapezoid area `eq:trap-area`, keeps the joined-pulse variant
  `eq:dh-joined`, adds the S-curve area-agreement `eq:scurve-trap-area`
  and a TikZ `fig:trap-scurve`. E.2 is a new ZUPT error budget; E.3 the
  trapezoid budget (now with overlap inflation `eq:overlap-inflation`);
  E.4 the per-algorithm conformal calibration; E.5 two quality tables
  (`tab:quality-checks-zupt` new, `tab:quality-checks` kept) plus
  per-feature paragraphs and nine `fig:qf-*` good/bad figures.
- **`scripts/figs/plot_quality_features.py` (new)** — scans every
  labelled segment, runs both estimators, and renders nine 1×2 good/bad
  quality-feature figures. The `active_fraction` and `end_vel_ratio`
  checks get no figure: no labelled ride trips the active-motion gate,
  and the end-velocity ratio is ≈0 by construction. `--scan-only` prints
  shortlists; `OVERRIDES` pins picks.

- **Appendix A merged and renamed → "Theoretical Elevator Pulse
  Analysis"**. The standalone Appendix B ("Time-optimal jerk is
  bang-bang") was folded in as subsection A.1 *The optimal elevator
  pulse*, compressed from four paragraphs to three (Hamiltonian /
  switching law / constrained arcs); the singular-arc and "resulting
  profile" paragraphs were dropped — the athans1966 citation absorbs
  the former, `fig:scurve-durations` already shows the latter.
  Subsection A.2 *Time parameters formula development* keeps the
  phase-by-phase integration but converts every inline `\(...\)` to a
  centred display and colour-keys the velocity-gain terms to a new
  hand-drawn TikZ figure `fig:trap-geometry`: two red wedges
  ($\tfrac12 a_{\max}T_j$ each) flank an orange rectangle
  ($a_{\max}T_a$); the three slabs sum to $v_{\max}$ and that area
  equation is what gives $T_a$ (\eqref{eq:Ta}). $T_v$ then drops out
  in teal, matching the cruise brace already in
  `fig:ride-time-constants`. Labels `app:derivations` and
  `app:bangbang` were preserved on the new section/subsection so the
  §2.3 and §2.4 `\ref` calls still resolve (printing as "A" and
  "A.1"). The downstream appendices renumber automatically: Dataset
  → B, Segmentation → C, Prediction → D. Paper is now 31 pages.

State after these sessions: 31 pages, compiles clean (no overfull
boxes, no undefined refs). `tab:quality-checks-zupt` now exists in
Appendix D, resolving the previously-open §3.2 dangling reference.
Appendix D carries ten figures; the per-feature `fig:qf-*` floats
leave some appendix whitespace — compact into subfigure grids if a
page budget is later imposed.

---

## 10. What is left to do

1. **§5 Evaluation** — write first (constrains the final introduction).
   Resolve the `\todo`s: experiment count (~150), elevator/building counts,
   CI half-width numbers, prior-best comparison. Follow the evaluation
   rhetorical moves; end every subsection with a Takeaway paragraph.
2. **§6 Discussion** and **§7 Conclusion**.
3. **Rewrite §1 Introduction** from scratch once the evaluation exists.
4. **Abstract** — finalise; clear its `\todo`s.
5. **Related Work** — not yet present; add it (systems-venue style:
   post-evaluation, category-clustered).
6. Run the pre-submission mechanical checklist (page count, fonts, broken
   refs, figure formats).
