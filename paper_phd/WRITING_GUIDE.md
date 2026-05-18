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
  using the phone's accelerometer alone (optional orientation-sensor addon).

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
| 2.6 | Smartphone sensors | `sec:bg-imu` | Written |
| 3 | Building-Level Inventory | `sec:building-inventory` | **STUB** |
| 4 | Orientation-Sensor Addon | `sec:orientation-addon` | **STUB** |
| 5 | Evaluation | `sec:evaluation` | **STUB** (§5.1 written; results pending) |
| 5.1 | Dataset and ground truth | `sec:eval-data` | Written (counts are open `\todo`s) |
| 6 | Discussion | `sec:discussion` | **STUB (empty)** |
| 7 | Conclusion | `sec:conclusion` | **STUB (empty)** |
| A | Derivations referenced in §2.4 | `app:derivations` | Written |
| B | Time-optimal jerk is bang-bang | `app:bangbang` | Written |
| C | Dataset collection and ground-truth pipeline | `app:dataset` | Written |

Section narrative arc: Background derives the **family** → §3 reuses the
family for a building-level inventory → §4 adds the orientation addon → §5
evaluates → §6/§7 close.

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
- **Appendix A** (`app:derivations`) — phase-by-phase integration for the
  ride-time constants of §2.4, plus the centroid-spacing identity used in
  §2.5.
- **Appendix B** (`app:bangbang`) — the Pontryagin-maximum-principle argument
  that time-optimal jerk is bang-bang (`eq:bang-bang`). The body only needs
  the *result*, not the proof.

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
- Key equation labels: `eq:elevator-bounds` (1), `eq:elevator-opt` (2),
  `eq:bang-bang` (3), `eq:Tj`/`eq:Ta`/`eq:Tv` (4–6),
  `eq:elevator-durations` (7), `eq:elevator-params` (8),
  `eq:trap-template` (9), `eq:W-from-kinematics`/`eq:f-from-kinematics`
  (10–11), `eq:ride-template` (12).
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

- **§5.1 Dataset and ground truth** — wrote the half-page data summary
  that opens the Evaluation: the field-campaign size (counts left as
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

State after these sessions: 15 pages, compiles clean (no undefined refs,
no overfull boxes).

---

## 10. What is left to do

1. **§5 Evaluation** — write first (constrains the final introduction).
   Resolve the `\todo`s: experiment count (~150), elevator/building counts,
   CI half-width numbers, prior-best comparison. Follow the evaluation
   rhetorical moves; end every subsection with a Takeaway paragraph.
2. **§3 Building-Level Inventory** — develop the elevator-presence and
   elevator-type classifier that reuses the family-fit machinery.
3. **§4 Orientation-Sensor Addon** — describe the orientation-sensor fusion
   and quantify its accuracy gain on the blind test.
4. **§6 Discussion** and **§7 Conclusion**.
5. **Rewrite §1 Introduction** from scratch once the evaluation exists.
6. **Abstract** — finalise; clear its `\todo`s.
7. **Related Work** — not yet present; add it (systems-venue style:
   post-evaluation, category-clustered).
8. Run the pre-submission mechanical checklist (page count, fonts, broken
   refs, figure formats).
