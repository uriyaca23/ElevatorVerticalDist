# `src/data/` — Sensor data, ground truth, and tooling

This folder holds everything related to experiment data: the raw sensor logs,
the processed CSV artifacts, the loader that hydrates them, and a small GUI
for editing ground-truth (GT) intervals.

---

## Folder layout

```
src/data/
├── rawData/                        # inputs — one folder per experiment
│   └── <exp_name>/
│       ├── sensorLog_*.txt         # tab-separated, no header
│       ├── metadata.txt
│       └── forBarometer/           # optional: secondary device PRS source
│           └── sensorLog_*.txt
│
├── structuredData/                 # processed CSV artifacts
│   ├── metadata.csv                # top-level index of all experiments
│   └── data/<exp_name>/
│       ├── ACC.csv, GYR.csv, MAG.csv, PRS.csv, ORI.csv, ...  (one per sensor)
│       ├── gt.csv                  # ground-truth intervals
│       ├── metadata.csv            # single row, same schema as the index
│       ├── baramoshka.csv          # floor → height map (user-filled later)
│       └── forBarometer_alignment.png  # diagnostic (if applicable)
│
├── loader/                         # pipeline + legacy loaders (Python package)
│   ├── __init__.py                 # public API re-exports
│   ├── __main__.py                 # `python -m src.data.loader` CLI
│   ├── constants.py                # paths, filenames, CSV schemas
│   ├── parsing.py                  # sensorLog + metadata parsers
│   ├── alignment.py                # secondary-barometer time alignment
│   ├── pipeline.py                 # CSV-based flow (main entry points)
│   └── legacy.py                   # old Excel-cached flow
│
├── gt_editor.py                    # Tkinter GUI for editing gt.csv
└── (archive)/                       # legacy experimenter folders, untouched
```

Experiment naming convention (by existing convention, not enforced):
`<experimenter>_<location>_<phone>_<date>[_expN]`.

---

## The loader

Two entry-point families live under `src/data/loader/`:

### 1. Pipeline flow (CSV-based, current)

Use this for all new code. `getExperimentData` lazily materialises the
`structuredData/data/<name>/` CSVs on first access and reuses them thereafter.

```python
from src.data.loader import (
    ExperimentPipeline,
    getExperimentData,
    saveExperimentData,
    list_experiments,
    RAW_DATA_ROOT,
)

# Discover experiments that have a raw sensor log.
names = list_experiments()                         # list[str]

# Load one. Accepts either a bare name or a full path under rawData/.
sensors, gt, metadata = getExperimentData(names[0])
# sensors   — dict[str, pd.DataFrame], keyed by sensor ("ACC", "PRS", ...)
# gt        — DataFrame [start_ms, end_ms, type, description,
#             signalClearRecording] covering the full timeline with 'outside'
#             filler; type ∈ {up, down, outside}; signalClearRecording is a
#             bool (default True) — flip to False when the interval's signal
#             is unclear / unreliable
# metadata  — dict[str, str] with keys: exp_name, experimenter, phone,
#             location, description, date, time

# Force a rebuild (re-parse the raw sensorLog, re-derive GT, overwrite CSVs).
sensors, gt, metadata = getExperimentData(names[0], use_cache=False)

# Wrap for per-interval iteration.
pipeline = ExperimentPipeline(sensors, gt, metadata)
for data_slice, gt_row, meta in pipeline:
    # data_slice is {sensor: df_clipped_to_[start_ms, end_ms)}
    ...

# Persist edits back to structuredData/ (rebuilds the top-level index).
saveExperimentData(name, sensors, gt, metadata)
```

Caching rule: if `structuredData/data/<name>/` already contains `gt.csv`,
`metadata.csv`, and every sensor CSV the raw log would produce, it is loaded
directly. Otherwise the raw sensorLog is parsed, GT is derived from the
barometer (`PRS.GT_height_m` → pressure-filter segmenter), and the CSVs are
written.

Timebase: `timestamp_ms` (sensor CSVs) and `start_ms` / `end_ms` (gt.csv) are
**wall-clock Unix epoch milliseconds**, not phone uptime. The reference is the
ISO timestamp embedded in the raw `sensorLog_YYYYMMDDTHHMMSS.txt` filename
(interpreted as local time on the recording device): the smallest boot
timestamp in the raw log is mapped to that wall-clock instant, and every
sample / GT edge is offset by the same delta so relative durations are
preserved. The forBarometer secondary device is start-aligned in boot time
first, then receives the primary's wall-clock shift.

Migrating older CSVs (one-off): when an existing experiment was processed
before this change, its CSVs are still in boot time. Run

```bash
venv/bin/python -m src.data.migrate_to_wallclock          # all experiments
venv/bin/python -m src.data.migrate_to_wallclock <name>   # one
```

to shift them in place. The script is idempotent (skips any CSV whose first
`timestamp_ms` already looks like Unix epoch ms) and never re-derives GT —
your edits in `gt.csv` are preserved, only their time axis moves.

Other useful entry points:

- `getExperimentRawParsed(exp)` — parse a raw sensorLog into per-sensor frames
  without touching `structuredData/`. Handles the optional `forBarometer/`
  secondary device by aligning and swapping in its PRS frame.
- `rebuild_metadata_index()` — regenerate `structuredData/metadata.csv` from
  every per-experiment `metadata.csv`. Called automatically by
  `getExperimentData` and `saveExperimentData`.

### 2. Legacy flow (Excel-cached)

Kept for older scripts under `(archive)/`. Prefer the pipeline flow in new code.

```python
from src.data.loader import loadBasicData, loadDataWithGT, load_experimenter

data = load_experimenter("eyal")                   # first available experiment
data = loadBasicData("eyal", exp=1)                # per-sensor DataFrames
data = loadDataWithGT("eyal", exp=1)               # + gt_label on PRS, cached
```

### Quick CLI

```bash
venv/bin/python -m src.data.loader [name] [exp]
# Prints row counts and columns per sensor for a legacy experiment.
```

---

## The GT editor

`gt_editor.py` is a Tkinter GUI for inspecting sensor signals and editing the
rows of `gt.csv` for a single experiment.

### Launch

```bash
venv/bin/python -m src.data.gt_editor [exp_folder_name]
```

If `exp_folder_name` is passed it is pre-selected and loaded; otherwise pick
one from the dropdown at the top and click **Load**. The list comes from
`list_experiments(RAW_DATA_ROOT)`.

### Layout

- **Left pane** — stacked matplotlib panels (altitude, velocity, |acc|, |gyr|,
  |mag|). GT intervals are rendered as coloured spans on every panel
  (`up`=green, `down`=red, `outside`=grey). Any non-empty interval `description`
  is annotated above the top panel.
- **Right pane** — a Treeview of intervals plus an edit form and action
  buttons.

### Editing intervals

- **Click** anywhere inside an interval to select it in the tree.
- **Drag an edge** (near the start or end of a coloured span) to resize the
  interval live. The cursor switches to a horizontal-arrow icon when you're
  within the hit tolerance.
- **Edit form** — change `start_ms`, `end_ms`, `type` (`up`/`down`/`outside`),
  and `note`, then click **Apply**.
- **+ Add** — inserts a new 10-second `outside` interval after the current
  selection (or at the end of the timeline).
- **✕ Delete** — removes the selected interval.
- **Auto-fix** — fills gaps between adjacent intervals with `outside` filler
  so the timeline stays contiguous. Warns instead of merging if intervals
  overlap.

### Navigation

- **+ / = / − / 0** — zoom x in, zoom x in, zoom x out, fit to full range.
- **Shift+←  /  Shift+→** — pan left / right by 25% of the visible width.
- **Mouse wheel** over the plot — zoom around the cursor.
- Toolbar buttons at the top mirror the keyboard shortcuts.

### Saving

- **Ctrl+S** or the **Save** button writes back via `saveExperimentData`,
  which rewrites every sensor CSV, `gt.csv`, and `metadata.csv` under
  `structuredData/data/<name>/`, then rebuilds the top-level index.
- `baramoshka.csv` is left untouched.
- The status bar shows unsaved-changes state; closing the window with unsaved
  edits prompts before quitting.

---

## CSV schemas

Defined in `loader/constants.py`.

| File              | Columns                                                                 |
|-------------------|-------------------------------------------------------------------------|
| `<SENSOR>.csv`    | `timestamp_ms` (wall-clock Unix epoch ms) + sensor-specific (e.g. `ACC`: `x, y, z`; `PRS`: `pressure`, `GT_height_m`). All frames get an `exp_name` column stamped on load. |
| `gt.csv`          | `start_ms, end_ms` (wall-clock Unix epoch ms), `type, description, signalClearRecording` (bool, defaults to `True` — set to `False` to mark intervals where the recording is unclear/unreliable) |
| `metadata.csv`    | `exp_name, experimenter, phone, location, description, date, time`      |
| `baramoshka.csv`  | `floor, height`                                                         |

### Gramushka → baramoshka → `height_diff_m`

Each building has a reference table at `src/data/gramushka/<building>/gramushka.csv`
mapping floor names to absolute elevations. The `populate_baramoshka`
script copies that table into every experiment's `baramoshka.csv` (with
the columns renamed `Floor Name`→`floor`, `Elevation (m)`→`height`).

When `baramoshka.csv` is populated and `metadata.csv` has a `start_floor`
row, `addGTtoSegment` runs in **snap mode**: temperature-aware barometer
altitudes are integrated from the known start-floor height and every
segment endpoint is snapped to the nearest baramoshka floor. This makes
`gt.csv`'s `height_diff_m` the difference of successive snapped floor
heights — robust to barometer drift. Segments whose estimated endpoint
is more than 1.5 m from any floor are listed in
`structuredData/data/<exp>/gramushka_flags.csv` for manual review;
`SNAP_AMBIGUITY_THRESHOLD_M` in `loader/pipeline.py` controls the cutoff.

If the baramoshka is empty (e.g. exp3 at Millenium) or the start_floor
doesn't resolve, the function falls back to pure temperature-aware
barometer Δh (standard ISA at 15 °C when no `temperature_c` is set in
metadata).

### `dataset_cleanup/` package

The full cleanup pipeline — gramushka snap, Pixel-10-anchored time
calibration, per-phone-PRS reference consolidation, noisy-segment
tagging, and verification plots — lives under
[`dataset_cleanup/`](dataset_cleanup/README.md). See that package's
README for the run order. Everything there is additive to the loader
above; no public API was removed or renamed.

Sensor column schemas for the raw parser:

| Sensor   | Columns                                           |
|----------|---------------------------------------------------|
| `ACC`    | `x, y, z`                                         |
| `GYR`    | `x, y, z`                                         |
| `MAG`    | `x, y, z`                                         |
| `RAWGYR` | `x, y, z, bias_x, bias_y, bias_z`                 |
| `RAWMAG` | `x, y, z, bias_x, bias_y, bias_z`                 |
| `ORI`    | `w, x, y, z`                                      |
| `PRS`    | `pressure`                                        |
| `GPS`    | `lat, lon, alt`                                   |

Raw `sensorLog_*.txt` is tab-separated with no header, one sample per line:

```
<timestamp_ms>\t<SENSOR>\t<value1>\t<value2>...
```

Rows whose column count doesn't match the sensor schema are dropped silently.
