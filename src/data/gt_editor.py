"""Tkinter GUI for inspecting and editing the GT intervals stored in
`pipeline_data.pkl` files under `structuredData/`.

Usage:
    venv/bin/python -m src.data.gt_editor [exp_folder_name]

If a folder name is passed it's pre-selected; otherwise pick one from the
dropdown and click *Load*.

Keyboard shortcuts:
    Ctrl+S     save to pipeline_data.pkl
    Delete     remove the selected interval
    Ctrl+N     add a new interval after the selected one

Mouse:
    Scroll         zoom the (shared) x-axis around the cursor
    Shift+Scroll   zoom the y-axis of the hovered panel only
"""

from __future__ import annotations

import shutil
import sys
import tkinter as tk
from datetime import datetime, timedelta
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure
from matplotlib.ticker import FuncFormatter

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import (
    RAW_DATA_ROOT,
    SOURCE_EXPERIMENT,
    STRUCTURED_DATA_DIR,
    VALID_SOURCES,
    ExperimentPipeline,
    getExperimentData,
    list_experiments,
    list_structured_experiments,
    saveExperimentData,
)
from src.data.loader.alignment import _smoothed_velocity
from src.data.loader.pipeline import (
    _coerce_bool,
    _derive_gt_from_prs,
    _segments_to_full_gt,
    _structured_dir_for,
    addGTtoSegment,
    load_baramoshka,
    rebuild_metadata_index,
)
from src.data.loadFromDB import LoadedSignal, PhoneType, loadDataFromS3
from src.physics import calculate_velocity_from_accelerometer, pressure_to_altitude
from src.utils.accelerometer_utils import vertical_accel_magnitude
from src.segmentation.algorithms import (
    SEGMENT_ALGORITHM_CONFIG,
    SegmentAlgorithm,
    Segmenter,
)


VALID_TYPES = ["outside", "up", "down"]
TYPE_COLORS = {"up": "#2ca02c", "down": "#d62728", "outside": "#b8b8b8"}
HIGHLIGHT_COLOR = "#1f77b4"


class _AddExperimentValidationError(ValueError):
    """Raised by the add-experiment orchestrator when a form field is
    malformed. Surfaces as `messagebox.showerror` rather than a crash."""


# ----------------------------------------------------------------------
# Upload-mode helpers
# ----------------------------------------------------------------------

# Per-sensor schema for the manual upload path. `data_cols` are the
# non-time columns the user maps; `required` flags ACC as the only
# non-optional sensor (matches the user's spec).
_UPLOAD_SENSORS: list[tuple[str, list[str], bool]] = [
    ("ACC", ["x", "y", "z"], True),
    ("PRS", ["pressure"], False),
    ("GYR", ["x", "y", "z"], False),
    ("MAG", ["x", "y", "z"], False),
    ("ORI", ["w", "x", "y", "z"], False),
]


def _read_tabular(path: Path) -> pd.DataFrame:
    """Read a CSV or Excel file into a DataFrame, picking the reader by
    extension. Mirrors :func:`src.pipelines.streamlit.step2_data._read_tabular_upload`.
    """
    suf = path.suffix.lower()
    if suf == ".csv":
        return pd.read_csv(path)
    if suf in (".xlsx", ".xls"):
        return pd.read_excel(path)
    raise ValueError(
        f"Unsupported file type: {path.suffix!r}. Use .csv, .xlsx, or .xls."
    )


def _guess_column(columns: list[str], candidates: list[str]) -> str:
    """Return the first item in `columns` whose lowercased name matches any
    of `candidates` (case-insensitive substring match), else the first
    column. Used to pre-fill mapping comboboxes."""
    if not columns:
        return ""
    lc = [c.lower() for c in columns]
    for cand in candidates:
        cand_lc = cand.lower()
        for i, c in enumerate(lc):
            if c == cand_lc:
                return columns[i]
        for i, c in enumerate(lc):
            if cand_lc in c:
                return columns[i]
    return columns[0]


def _parse_time_to_epoch_ms(series: pd.Series, day_first: bool) -> np.ndarray:
    """Convert a user-supplied time column to int64 epoch ms.

    The series **must not contain NaN** — the caller is expected to have
    already dropped rows where any mapped column is missing (mirrors the
    streamlit data form's pattern).

    Numeric columns: same heuristic as
    :func:`src.data.csv_ingest.detect_time_unit` —
    ``> 1e12`` → epoch ms, ``> 1e9`` → epoch s, span ``< 1e4`` → relative
    seconds, else relative ms.

    String columns: pandas ``to_datetime`` with ``dayfirst=day_first``.
    Raises ValueError if any value fails to parse.
    """
    if series.empty:
        raise ValueError("Time column is empty.")

    nums = pd.to_numeric(series, errors="coerce")
    numeric_share = nums.notna().mean()
    if numeric_share > 0.95:
        ts = nums.to_numpy(dtype=float)
        n_bad = int(np.isnan(ts).sum())
        if n_bad:
            raise ValueError(
                f"{n_bad} numeric time values failed to parse — drop empty rows first."
            )
        mx = float(np.nanmax(ts))
        span = mx - float(np.nanmin(ts))
        if mx > 1e12:
            return ts.astype("int64")
        if mx > 1e9:
            return (ts * 1000.0).astype("int64")
        if span < 1e4:
            return (ts * 1000.0).astype("int64")
        return ts.astype("int64")

    dts = pd.to_datetime(series, dayfirst=day_first, errors="coerce", utc=False)
    n_bad = int(dts.isna().sum())
    if n_bad:
        raise ValueError(
            f"Could not parse {n_bad} time values as datetime "
            f"(try toggling 'Day-first?')."
        )
    # If the parsed series is tz-aware (e.g. 'Z' or '+HH:MM' suffix),
    # convert to UTC and drop the tz so the int64 cast works — pandas
    # rejects astype on tz-aware → tz-naive directly. Epoch ms is
    # timezone-agnostic, so converting through UTC is lossless.
    if getattr(dts.dt, "tz", None) is not None:
        dts = dts.dt.tz_convert("UTC").dt.tz_localize(None)
    # Cast to a fixed ms resolution before .astype('int64') — pandas 2.x
    # parses to datetime64[us] by default, so a naive int64 cast would be
    # microseconds (then //1e6 → seconds, not ms). The explicit step makes
    # the unit unambiguous regardless of source precision.
    return dts.astype("datetime64[ms]").astype("int64").to_numpy(dtype="int64")


def _cell_str(v) -> str:
    """Cast a DataFrame cell to a safe display string, treating NaN as empty."""
    if v is None:
        return ""
    try:
        if pd.isna(v):
            return ""
    except (TypeError, ValueError):
        pass
    return str(v)


# --------------------------------------------------------------------------
# Panel drawers
# --------------------------------------------------------------------------

def _draw_altitude(ax, data, t0):
    df = data["PRS"]
    t = (df["timestamp_ms"].to_numpy(dtype=float) - t0) / 1000.0
    ax.plot(t, df["GT_height_m"].to_numpy(), color="tab:green", lw=0.9)
    ax.set_ylabel("altitude (m)")


def _draw_velocity(ax, data, t0):
    df = data["PRS"]
    t = (df["timestamp_ms"].to_numpy(dtype=float) - t0) / 1000.0
    h = df["GT_height_m"].to_numpy(dtype=float)
    h_smooth = pd.Series(h).rolling(51, center=True, min_periods=1).median().to_numpy()
    vz = _smoothed_velocity(t, h_smooth)
    ax.plot(t, vz, color="black", lw=0.8)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("vz (m/s)")


def _draw_acc(ax, data, t0):
    df = data["ACC"]
    t = (df["timestamp_ms"].to_numpy(dtype=float) - t0) / 1000.0
    # Plot the same |a|−g residual the matched filter scores against
    # (rotation-invariant; positive = take-off lobe, negative = landing).
    sig = vertical_accel_magnitude(
        df["x"].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        df["z"].to_numpy(dtype=float),
    )
    ax.plot(t, sig, color="tab:blue", lw=0.5)
    ax.axhline(0.0, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax.set_ylabel("|a|−g (m/s²)")


def _draw_acc_velocity(ax, data, t0):
    df = data["ACC"]
    ts_ms = df["timestamp_ms"].to_numpy(dtype=float)
    t = (ts_ms - t0) / 1000.0
    if len(ts_ms) < 2:
        ax.set_ylabel("vz_acc (m/s)")
        return
    dt_ms = float(np.median(np.diff(ts_ms)))
    fs = 1000.0 / dt_ms if dt_ms > 0 else 100.0
    v = calculate_velocity_from_accelerometer(
        df["x"].to_numpy(dtype=float),
        df["y"].to_numpy(dtype=float),
        df["z"].to_numpy(dtype=float),
        fs,
    )
    ax.plot(t, v, color="tab:blue", lw=0.8)
    ax.axhline(0, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("vz_acc (m/s)")


def _draw_gyr(ax, data, t0):
    df = data["GYR"]
    t = (df["timestamp_ms"].to_numpy(dtype=float) - t0) / 1000.0
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2).to_numpy()
    ax.plot(t, mag, color="tab:purple", lw=0.5)
    ax.set_ylabel("|ω| (rad/s)")


def _draw_mag(ax, data, t0):
    df = data["MAG"]
    t = (df["timestamp_ms"].to_numpy(dtype=float) - t0) / 1000.0
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2).to_numpy()
    ax.plot(t, mag, color="tab:orange", lw=0.5)
    ax.set_ylabel("|m| (µT)")


PANEL_DRAWERS = [
    ("altitude",     "PRS", _draw_altitude),
    ("velocity",     "PRS", _draw_velocity),
    ("acc",          "ACC", _draw_acc),
    ("acc_velocity", "ACC", _draw_acc_velocity),
    ("gyr",          "GYR", _draw_gyr),
    ("mag",          "MAG", _draw_mag),
]


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class GtEditor(tk.Tk):
    def __init__(self, preselect: str | None = None):
        super().__init__()
        self.title("GT Editor")
        self.geometry("1700x950")

        self.pipeline: ExperimentPipeline | None = None
        self.exp_path: Path | None = None
        self._t0_ms: int = 0
        self._axes: list = []
        self._hl_spans: list = []
        self._dirty: bool = False
        # Drag state — for edge-drag editing of GT intervals.
        self._drag_active: bool = False
        self._drag_edge: str | None = None   # "start" | "end"
        self._drag_idx: int | None = None

        self._build_ui()
        self._populate_experiments()

        if preselect:
            self.exp_var.set(preselect)
            self.after(50, self.load_experiment)

        self.protocol("WM_DELETE_WINDOW", self._on_close)

    # ---------- UI construction ----------

    def _build_ui(self):
        # Top bar
        top = ttk.Frame(self, padding=6)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Experiment:").pack(side=tk.LEFT)
        self.exp_var = tk.StringVar()
        self.exp_combo = ttk.Combobox(
            top, textvariable=self.exp_var, width=50, state="readonly",
        )
        self.exp_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load", command=self.load_experiment).pack(side=tk.LEFT)
        ttk.Button(top, text="Add experiment…",
                   command=self._open_add_experiment_dialog)\
            .pack(side=tk.LEFT, padx=(4, 0))
        ttk.Button(top, text="Save (Ctrl+S)", command=self.save_gt).pack(side=tk.RIGHT)
        ttk.Button(top, text="🗑 Delete experiment",
                   command=self._on_delete_experiment)\
            .pack(side=tk.RIGHT, padx=(4, 4))

        # Navigation controls live in a compact 2-row grid so the top bar
        # still fits on narrow screens. Row 0 = zoom, row 1 = pan, columns
        # aligned so each pan button sits directly under its zoom button.
        nav = ttk.Frame(top)
        nav.pack(side=tk.LEFT, padx=(12, 6))
        ttk.Button(nav, text="Fit (0)", width=9,
                   command=lambda: self._fit_x())\
            .grid(row=0, column=0, padx=2, pady=1)
        ttk.Button(nav, text="− Zoom X", width=9,
                   command=lambda: self._zoom_x_around_center(zoom_in=False))\
            .grid(row=0, column=1, padx=(8, 2), pady=1)
        ttk.Button(nav, text="+ Zoom X", width=9,
                   command=lambda: self._zoom_x_around_center(zoom_in=True))\
            .grid(row=0, column=2, padx=2, pady=1)
        ttk.Button(nav, text="− Zoom Y", width=9,
                   command=lambda: self._zoom_y_around_center(zoom_in=False))\
            .grid(row=0, column=3, padx=(8, 2), pady=1)
        ttk.Button(nav, text="+ Zoom Y", width=9,
                   command=lambda: self._zoom_y_around_center(zoom_in=True))\
            .grid(row=0, column=4, padx=2, pady=1)
        ttk.Button(nav, text="◀ X", width=9,
                   command=lambda: self._pan_x(-0.25))\
            .grid(row=1, column=1, padx=(8, 2), pady=1)
        ttk.Button(nav, text="X ▶", width=9,
                   command=lambda: self._pan_x(+0.25))\
            .grid(row=1, column=2, padx=2, pady=1)
        ttk.Button(nav, text="▼ Y", width=9,
                   command=lambda: self._pan_y(-0.25))\
            .grid(row=1, column=3, padx=(8, 2), pady=1)
        ttk.Button(nav, text="Y ▲", width=9,
                   command=lambda: self._pan_y(+0.25))\
            .grid(row=1, column=4, padx=2, pady=1)
        self.status_var = tk.StringVar(value="Pick an experiment, then Load.")
        ttk.Label(top, textvariable=self.status_var, foreground="#555")\
            .pack(side=tk.RIGHT, padx=10)
        self.hover_var = tk.StringVar(value="")
        ttk.Label(top, textvariable=self.hover_var, foreground="#222",
                  font=("Menlo", 10))\
            .pack(side=tk.RIGHT, padx=10)

        # Main paned: plot | right panel
        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Left: matplotlib canvas ---
        left = ttk.Frame(main)
        main.add(left, weight=4)
        self.fig = Figure(figsize=(11, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, left).update()
        self.canvas.mpl_connect("button_press_event",   self._on_press)
        self.canvas.mpl_connect("motion_notify_event",  self._on_motion)
        self.canvas.mpl_connect("button_release_event", self._on_release)
        self.canvas.mpl_connect("scroll_event",         self._on_scroll)

        # Tk-level wheel binding as fallback — mpl `scroll_event` is unreliable
        # on macOS TkAgg. Binds both the Aqua `<MouseWheel>` and X11 buttons.
        widget = self.canvas.get_tk_widget()
        widget.bind("<MouseWheel>", self._on_tk_wheel)
        widget.bind("<Button-4>", lambda e: self._on_tk_wheel(e, delta=+120))
        widget.bind("<Button-5>", lambda e: self._on_tk_wheel(e, delta=-120))

        # --- Right: treeview + edit form ---
        right = ttk.Frame(main, padding=6)
        main.add(right, weight=1)

        ttk.Label(right, text="GT Intervals", font=("", 10, "bold")).pack(anchor=tk.W)

        tree_frame = ttk.Frame(right)
        tree_frame.pack(fill=tk.BOTH, expand=True)
        self.tree = ttk.Treeview(
            tree_frame,
            columns=("idx", "start_s", "dur_s", "type", "dh", "clear", "desc"),
            show="headings", selectmode="browse",
        )
        for col, text, w, anchor in [
            ("idx",     "#",         40,  tk.E),
            ("start_s", "start (s)", 80,  tk.E),
            ("dur_s",   "dur (s)",   70,  tk.E),
            ("type",    "type",      80,  tk.CENTER),
            ("dh",      "Δh (m)",    75,  tk.E),
            ("clear",   "clear?",    55,  tk.CENTER),
            ("desc",    "note",      180, tk.W),
        ]:
            self.tree.heading(col, text=text)
            self.tree.column(col, width=w, anchor=anchor)
        self.tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(tree_frame, orient=tk.VERTICAL,
                               command=self.tree.yview)
        self.tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        self.tree.bind("<<TreeviewSelect>>", self._on_tree_select)

        # Color tags for types in the tree
        self.tree.tag_configure("up",      foreground="#2ca02c")
        self.tree.tag_configure("down",    foreground="#d62728")
        self.tree.tag_configure("outside", foreground="#666666")

        # Edit form
        edit = ttk.LabelFrame(right, text="Edit selected", padding=6)
        edit.pack(fill=tk.X, pady=6)

        ttk.Label(edit, text="start (ms):").grid(row=0, column=0, sticky=tk.W)
        self.start_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.start_var, width=14)\
            .grid(row=0, column=1, padx=4, pady=2)

        ttk.Label(edit, text="end (ms):").grid(row=1, column=0, sticky=tk.W)
        self.end_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.end_var, width=14)\
            .grid(row=1, column=1, padx=4, pady=2)

        ttk.Label(edit, text="type:").grid(row=2, column=0, sticky=tk.W)
        self.type_var = tk.StringVar()
        ttk.Combobox(edit, textvariable=self.type_var, values=VALID_TYPES,
                     state="readonly", width=12)\
            .grid(row=2, column=1, padx=4, pady=2, sticky=tk.W)

        ttk.Label(edit, text="note:").grid(row=3, column=0, sticky=tk.W)
        self.desc_var = tk.StringVar()
        ttk.Entry(edit, textvariable=self.desc_var, width=28)\
            .grid(row=3, column=1, padx=4, pady=2, sticky=tk.EW)

        self.clear_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(edit, text="signal clear recording",
                        variable=self.clear_var)\
            .grid(row=4, column=0, columnspan=2, padx=4, pady=2, sticky=tk.W)

        ttk.Label(edit, text="Δh (m):").grid(row=5, column=0, sticky=tk.W)
        self.height_diff_var = tk.StringVar(value="—")
        ttk.Label(edit, textvariable=self.height_diff_var,
                  font=("Menlo", 10), foreground="#1f77b4")\
            .grid(row=5, column=1, padx=4, pady=2, sticky=tk.W)

        ttk.Button(edit, text="Apply", command=self._on_apply)\
            .grid(row=6, column=0, columnspan=2, pady=4, sticky=tk.EW)

        # Actions
        actions = ttk.Frame(right)
        actions.pack(fill=tk.X, pady=4)
        ttk.Button(actions, text="+ Add (Ctrl+N)", command=self._on_add)\
            .pack(side=tk.LEFT)
        ttk.Button(actions, text="✕ Delete (Del)", command=self._on_delete)\
            .pack(side=tk.LEFT, padx=4)
        ttk.Button(actions, text="Auto-fix", command=self._on_autofix)\
            .pack(side=tk.LEFT, padx=4)

        # Keyboard shortcuts
        self.bind("<Control-s>", lambda e: self.save_gt())
        self.bind("<Delete>",    lambda e: self._on_delete())
        self.bind("<Control-n>", lambda e: self._on_add())
        # X-axis zoom shortcuts
        self.bind("<plus>",   lambda e: self._zoom_x_around_center(zoom_in=True))
        self.bind("<equal>",  lambda e: self._zoom_x_around_center(zoom_in=True))
        self.bind("<minus>",  lambda e: self._zoom_x_around_center(zoom_in=False))
        self.bind("<Key-0>",  lambda e: self._fit_x())
        # X-axis pan shortcuts (Shift+Arrow to avoid clashing with interval nav).
        self.bind("<Shift-Left>",  lambda e: self._pan_x(-0.25))
        self.bind("<Shift-Right>", lambda e: self._pan_x(+0.25))
        # Y-axis pan (Shift+Up / Shift+Down) — mirror of the X-axis shortcut.
        self.bind("<Shift-Up>",    lambda e: self._pan_y(+0.25))
        self.bind("<Shift-Down>",  lambda e: self._pan_y(-0.25))

    def _populate_experiments(self):
        # Union of rawData experiments (have a sensorLog) and experiments
        # that exist only under structuredData/ (e.g. S3-ingested ones).
        raw = list_experiments(RAW_DATA_ROOT)
        structured = list_structured_experiments(STRUCTURED_DATA_DIR)
        seen = set(raw)
        merged = list(raw) + [n for n in structured if n not in seen]
        self.exp_combo["values"] = merged

    # ---------- Load / save ----------

    def load_experiment(self):
        name = self.exp_var.get()
        if not name:
            self.status_var.set("Pick an experiment first.")
            return
        if self._dirty and not messagebox.askyesno(
            "Unsaved changes", "Discard unsaved GT edits?",
        ):
            return

        self.exp_path = RAW_DATA_ROOT / name
        try:
            sensors, gt, metadata = getExperimentData(self.exp_path, use_cache=True)
        except Exception as e:
            messagebox.showerror("Load failed", f"{type(e).__name__}: {e}")
            return
        self.pipeline = ExperimentPipeline(
            sensors, gt, metadata,
            valid_intervals=gt.attrs.get("valid_intervals_per_sensor", {}),
        )

        prs = self.pipeline.data.get("PRS")
        acc = self.pipeline.data.get("ACC")
        if acc is not None and not acc.empty:
            self._t0_ms = int(acc["timestamp_ms"].iloc[0])
        elif prs is not None and not prs.empty:
            self._t0_ms = int(prs["timestamp_ms"].iloc[0])
        else:
            self._t0_ms = 0

        self._dirty = False
        self._refresh_plot()
        self._refresh_tree()
        self.status_var.set(
            f"Loaded {len(self.pipeline.gt)} intervals from "
            f"structuredData/data/{name}/"
        )

    def save_gt(self):
        if self.pipeline is None or self.exp_path is None:
            return
        name = self.exp_path.name
        try:
            out_dir = saveExperimentData(
                name,
                self.pipeline.data, self.pipeline.gt, self.pipeline.metaData,
            )
        except Exception as e:
            messagebox.showerror("Save failed", f"{type(e).__name__}: {e}")
            return
        self._dirty = False
        self.status_var.set(f"Saved ✓  ({out_dir})")

    def _on_delete_experiment(self):
        """Delete the currently-loaded experiment's structuredData folder
        and remove its row from `structuredData/metadata.csv`.

        The top-level metadata.csv is rebuilt from per-experiment
        metadata.csv files, so removing the folder + calling
        `rebuild_metadata_index` is enough to drop the row.
        """
        name = self.exp_var.get().strip()
        if not name:
            messagebox.showinfo("Nothing to delete", "Pick an experiment first.")
            return
        out_dir = _structured_dir_for(name)
        if not out_dir.exists():
            messagebox.showinfo(
                "Nothing to delete",
                f"No structuredData folder for '{name}'.",
            )
            return
        if not messagebox.askyesno(
            "Delete experiment?",
            f"This will permanently delete:\n\n  {out_dir}\n\n"
            f"and remove '{name}' from structuredData/metadata.csv.\n\n"
            f"Continue?",
            icon="warning",
        ):
            return
        try:
            shutil.rmtree(out_dir)
            rebuild_metadata_index()
        except Exception as e:
            messagebox.showerror("Delete failed", f"{type(e).__name__}: {e}")
            return

        # Drop the in-memory pipeline + clear the plot/tree so we don't
        # operate on a now-deleted experiment.
        self.pipeline = None
        self.exp_path = None
        self._dirty = False
        self.exp_var.set("")
        self._populate_experiments()
        self._refresh_plot()
        self._refresh_tree()
        self.status_var.set(f"Deleted '{name}' ✓")

    # ---------- Plot ----------

    def _refresh_plot(self, preserve_xlim: bool = False):
        # Save the current x view so edits don't reset the user's zoom.
        prev_xlim = None
        if preserve_xlim and self._axes:
            try:
                prev_xlim = self._axes[0].get_xlim()
            except Exception:
                prev_xlim = None

        self.fig.clear()
        self._axes = []
        self._hl_spans = []
        if self.pipeline is None:
            self.canvas.draw()
            return

        data = self.pipeline.data
        panels = [
            (name, drawer)
            for name, sensor, drawer in PANEL_DRAWERS
            if sensor in data and not data[sensor].empty
        ]
        if not panels:
            self.canvas.draw()
            return

        axes = self.fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
        for ax, (name, drawer) in zip(axes, panels):
            drawer(ax, data, self._t0_ms)
            ax.grid(True, alpha=0.3)
            ax.set_title(name, fontsize=9, loc="left")

        # Bottom panel x-axis: each tick shows the offset in seconds AND
        # the wall-clock equivalent (HH:MM:SS) on a second line. With
        # sharex=True, only the bottom panel renders tick labels.
        axes[-1].xaxis.set_major_formatter(
            FuncFormatter(self._format_x_tick),
        )
        axes[-1].set_xlabel("time  (s since start · wall clock)")
        if prev_xlim is not None:
            axes[0].set_xlim(prev_xlim)  # sharex propagates to all panels

        # GT spans on all panels + note annotation on the top panel.
        for _, row in self.pipeline.gt.iterrows():
            s = (int(row["start_ms"]) - self._t0_ms) / 1000.0
            e = (int(row["end_ms"]) - self._t0_ms) / 1000.0
            c = TYPE_COLORS.get(str(row["type"]), "#cccccc")
            for ax in axes:
                ax.axvspan(s, e, color=c, alpha=0.22, zorder=0)
            note = _cell_str(row.get("description", ""))
            if note:
                axes[0].annotate(
                    note, xy=((s + e) / 2.0, 1.0),
                    xycoords=("data", "axes fraction"),
                    ha="center", va="bottom", fontsize=7, color="#333",
                    clip_on=True,
                )

        # Blue "no data" overlays for sub-threshold-frequency / missing
        # regions. The loader marks valid intervals per sensor; we shade
        # the complement so the user knows not to label GT inside them.
        # We intersect across the dense motion sensors (skipping GPS,
        # which is naturally sparse) so a gap shown here means at least
        # one core IMU lost signal — that's the user-actionable case.
        valid_per_sensor = self.pipeline.valid_intervals or {}
        if not valid_per_sensor:
            valid_per_sensor = (
                self.pipeline.gt.attrs.get("valid_intervals_per_sensor", {})
            )
        core_intervals = [
            sorted((int(a), int(b)) for a, b in ivs)
            for s, ivs in valid_per_sensor.items()
            if s in ("ACC", "GYR", "MAG", "PRS", "ORI") and ivs
        ]
        if core_intervals:
            # Two-pointer intersection of interval lists. ``current``
            # holds the running intersection; we fold each sensor's list
            # into it.
            def _intersect(
                a: list[tuple[int, int]], b: list[tuple[int, int]],
            ) -> list[tuple[int, int]]:
                out: list[tuple[int, int]] = []
                i = j = 0
                while i < len(a) and j < len(b):
                    lo = max(a[i][0], b[j][0])
                    hi = min(a[i][1], b[j][1])
                    if lo <= hi:
                        out.append((lo, hi))
                    if a[i][1] < b[j][1]:
                        i += 1
                    else:
                        j += 1
                return out

            current = core_intervals[0]
            for nxt in core_intervals[1:]:
                current = _intersect(current, nxt)
                if not current:
                    break

            # Compute the gap regions = complement of ``current`` over
            # the union span.
            t_lo_ms = min(ivs[0][0] for ivs in core_intervals)
            t_hi_ms = max(ivs[-1][1] for ivs in core_intervals)
            cursor = t_lo_ms
            gap_spans_ms: list[tuple[int, int]] = []
            for a, b in current:
                if a > cursor:
                    gap_spans_ms.append((cursor, a))
                cursor = max(cursor, b + 1)
            if cursor < t_hi_ms:
                gap_spans_ms.append((cursor, t_hi_ms))

            for gs_ms, ge_ms in gap_spans_ms:
                if ge_ms <= gs_ms:
                    continue
                gs = (gs_ms - self._t0_ms) / 1000.0
                ge = (ge_ms - self._t0_ms) / 1000.0
                for ax in axes:
                    ax.axvspan(gs, ge, color="#3498db", alpha=0.28,
                               zorder=3, lw=0)
                y_lo, y_hi = axes[0].get_ylim()
                axes[0].text(
                    (gs + ge) / 2.0, y_lo + 0.92 * (y_hi - y_lo),
                    "✕", color="#1b5b8e", ha="center", va="center",
                    fontsize=14, fontweight="bold", zorder=5,
                )

        self._axes = list(axes)
        self.fig.tight_layout()
        self.canvas.draw()

    def _wall_time_str(self, s_offset: float, *, sub_second: bool = False) -> str:
        """Convert ``s_offset`` (seconds since the experiment start) to a
        wall-clock string ``HH:MM:SS`` (or ``HH:MM:SS.mmm`` when
        ``sub_second``). Returns ``""`` if no experiment is loaded or the
        timestamp is out of range."""
        if self._t0_ms <= 0:
            return ""
        try:
            wall = datetime.fromtimestamp(self._t0_ms / 1000.0 + s_offset)
        except (OverflowError, ValueError, OSError):
            return ""
        if sub_second:
            return wall.strftime("%H:%M:%S.") + f"{wall.microsecond // 1000:03d}"
        return wall.strftime("%H:%M:%S")

    def _format_x_tick(self, s_offset: float, _pos: int | None = None) -> str:
        """X-axis tick formatter: render the offset in seconds and (when
        an experiment is loaded) the wall-clock equivalent on a second
        line. Wired up via :class:`matplotlib.ticker.FuncFormatter` from
        :meth:`_refresh_plot`."""
        head = f"{s_offset:.0f}s"
        wall = self._wall_time_str(s_offset)
        return f"{head}\n{wall}" if wall else head

    def _highlight_on_plot(self, start_ms, end_ms):
        for h in self._hl_spans:
            try:
                h.remove()
            except Exception:
                pass
        self._hl_spans = []
        s = (int(start_ms) - self._t0_ms) / 1000.0
        e = (int(end_ms) - self._t0_ms) / 1000.0
        for ax in self._axes:
            h = ax.axvspan(s, e, edgecolor=HIGHLIGHT_COLOR, facecolor="none",
                           lw=2.0, zorder=10)
            self._hl_spans.append(h)
        self.canvas.draw_idle()

    def _edge_tolerance_seconds(self, ax) -> float:
        """Drag-hit tolerance in data-seconds, scaled to current zoom level."""
        xlim = ax.get_xlim()
        return max(0.05, (xlim[1] - xlim[0]) * 0.01)

    def _closest_edge(self, ax, x_sec: float):
        """Return (idx, edge, distance) of the closest interval edge to x_sec,
        or None if nothing is within tolerance.
        """
        if self.pipeline is None:
            return None
        tol = self._edge_tolerance_seconds(ax)
        best = None
        for i in self.pipeline.gt.index:
            s = (int(self.pipeline.gt.loc[i, "start_ms"]) - self._t0_ms) / 1000.0
            e = (int(self.pipeline.gt.loc[i, "end_ms"]) - self._t0_ms) / 1000.0
            for edge, pos in (("start", s), ("end", e)):
                d = abs(x_sec - pos)
                if d < tol and (best is None or d < best[2]):
                    best = (int(i), edge, d)
        return best

    def _on_press(self, event):
        if (event.inaxes not in self._axes or self.pipeline is None
                or event.xdata is None or event.button != 1):
            return
        # Suppress drag if matplotlib toolbar zoom/pan is active.
        tb_mode = getattr(self.canvas.manager, "toolbar", None)
        if tb_mode and getattr(tb_mode, "mode", ""):
            return

        # Edge drag if clicking near any edge.
        edge_hit = self._closest_edge(event.inaxes, event.xdata)
        if edge_hit is not None:
            idx, edge, _ = edge_hit
            self.tree.selection_set(str(idx))
            self.tree.see(str(idx))
            self._drag_active = True
            self._drag_edge = edge
            self._drag_idx = idx
            return

        # Otherwise, click-to-select the interval containing the cursor.
        click_ms = int(self._t0_ms + event.xdata * 1000)
        gt = self.pipeline.gt
        mask = (
            (gt["start_ms"].astype(int) <= click_ms)
            & (gt["end_ms"].astype(int) > click_ms)
        )
        matches = gt[mask]
        if len(matches):
            idx = int(matches.index[0])
            self.tree.selection_set(str(idx))
            self.tree.see(str(idx))

    def _on_motion(self, event):
        # Hover coord readout (always updated).
        if (event.inaxes in self._axes and event.xdata is not None
                and event.ydata is not None):
            ylab = event.inaxes.get_ylabel() or "y"
            wall = self._wall_time_str(event.xdata, sub_second=True)
            wall_part = f"  ({wall})" if wall else ""
            self.hover_var.set(
                f"t = {event.xdata:8.2f} s{wall_part}    "
                f"{ylab} = {event.ydata:.3f}"
            )
        else:
            self.hover_var.set("")

        # Cursor feedback when hovering near a draggable edge.
        if (not self._drag_active and event.inaxes in self._axes
                and event.xdata is not None):
            near = self._closest_edge(event.inaxes, event.xdata) is not None
            try:
                self.canvas.get_tk_widget().configure(
                    cursor="sb_h_double_arrow" if near else ""
                )
            except tk.TclError:
                pass

        # Live drag update: move the selected interval's edge.
        if (not self._drag_active or event.xdata is None
                or self._drag_idx is None or self.pipeline is None):
            return
        if self._drag_idx not in self.pipeline.gt.index:
            return

        new_ms = int(self._t0_ms + event.xdata * 1000)
        row = self.pipeline.gt.loc[self._drag_idx]
        if self._drag_edge == "start":
            end_ms = int(row["end_ms"])
            if new_ms < end_ms - 10:
                self.pipeline.gt.loc[self._drag_idx, "start_ms"] = new_ms
                self.start_var.set(str(new_ms))
        else:  # "end"
            start_ms = int(row["start_ms"])
            if new_ms > start_ms + 10:
                self.pipeline.gt.loc[self._drag_idx, "end_ms"] = new_ms
                self.end_var.set(str(new_ms))

        # Fast path: move only the blue highlight outline, not the colored spans.
        row = self.pipeline.gt.loc[self._drag_idx]
        self._highlight_on_plot(row["start_ms"], row["end_ms"])

    def _on_release(self, event):
        if not self._drag_active:
            return
        self._drag_active = False
        idx = self._drag_idx
        self._drag_idx = None
        self._drag_edge = None
        self._recompute_height_diffs()
        self._mark_dirty(f"Dragged interval {idx} — unsaved")
        # Full refresh so the colored GT span catches up with the final bounds.
        self._refresh_plot(preserve_xlim=True)
        self._refresh_tree()
        if idx is not None and str(idx) in self.tree.get_children():
            self.tree.selection_set(str(idx))
            self.tree.see(str(idx))

    def _on_scroll(self, event):
        # mpl scroll_event — unreliable on macOS TkAgg, but wired up as backup.
        if event.inaxes not in self._axes:
            return
        zoom_in = (event.button == "up")
        # Shift+scroll zooms the y-axis of the hovered panel only.
        if event.key and "shift" in event.key:
            if event.ydata is None:
                return
            self._zoom_y_at(event.inaxes, event.ydata, zoom_in=zoom_in)
            return
        if event.xdata is None:
            return
        self._zoom_x_at(event.inaxes, event.xdata, zoom_in=zoom_in)

    # ---------- X-axis zoom helpers ----------

    def _zoom_x_at(self, ax, center_x: float, zoom_in: bool) -> None:
        """Zoom the x-axis around `center_x` (data coords)."""
        xlim = ax.get_xlim()
        width = xlim[1] - xlim[0]
        if width <= 0:
            return
        factor = 1 / 1.5 if zoom_in else 1.5
        new_width = max(width * factor, 0.01)  # cap zoom-in so we can still pan
        left_frac = (center_x - xlim[0]) / width
        new_left = center_x - new_width * left_frac
        ax.set_xlim(new_left, new_left + new_width)
        self.canvas.draw_idle()

    def _zoom_x_around_center(self, zoom_in: bool) -> None:
        if not self._axes:
            return
        ax = self._axes[0]
        xlim = ax.get_xlim()
        self._zoom_x_at(ax, (xlim[0] + xlim[1]) / 2.0, zoom_in=zoom_in)

    # ---------- Y-axis zoom (per-panel) ----------

    def _zoom_y_at(self, ax, center_y: float, zoom_in: bool) -> None:
        """Zoom the y-axis of `ax` around `center_y` (data coords)."""
        ylim = ax.get_ylim()
        height = ylim[1] - ylim[0]
        if height == 0:
            return
        factor = 1 / 1.5 if zoom_in else 1.5
        new_height = height * factor
        bot_frac = (center_y - ylim[0]) / height
        new_bot = center_y - new_height * bot_frac
        ax.set_ylim(new_bot, new_bot + new_height)
        self.canvas.draw_idle()

    def _fit_y(self, ax) -> None:
        """Autoscale just this panel's y-axis to its visible x range."""
        ax.relim()
        ax.autoscale(axis="y")
        self.canvas.draw_idle()

    def _zoom_y_around_center(self, zoom_in: bool) -> None:
        """Zoom every panel's y-axis around its current midpoint."""
        for ax in self._axes:
            ylim = ax.get_ylim()
            self._zoom_y_at(ax, (ylim[0] + ylim[1]) / 2.0, zoom_in=zoom_in)

    def _pan_x(self, frac: float) -> None:
        """Pan the x-axis by `frac` of the current visible width.

        Positive `frac` pans to the right (later in time), negative to the left.
        """
        if not self._axes:
            return
        ax = self._axes[0]
        xlim = ax.get_xlim()
        shift = (xlim[1] - xlim[0]) * frac
        ax.set_xlim(xlim[0] + shift, xlim[1] + shift)
        self.canvas.draw_idle()

    def _pan_y(self, frac: float) -> None:
        """Pan every panel's y-axis by `frac` of its visible height.

        Positive `frac` pans up (toward larger values), negative down.
        """
        for ax in self._axes:
            ylim = ax.get_ylim()
            shift = (ylim[1] - ylim[0]) * frac
            ax.set_ylim(ylim[0] + shift, ylim[1] + shift)
        self.canvas.draw_idle()

    def _fit_x(self) -> None:
        """Reset the x-axis to the full data range."""
        if self.pipeline is None or not self._axes:
            return
        prs = self.pipeline.data.get("PRS")
        acc = self.pipeline.data.get("ACC")
        if acc is not None and not acc.empty:
            t0 = int(acc["timestamp_ms"].iloc[0])
            t1 = int(acc["timestamp_ms"].iloc[-1])
        elif prs is not None and not prs.empty:
            t0 = int(prs["timestamp_ms"].iloc[0])
            t1 = int(prs["timestamp_ms"].iloc[-1])
        else:
            return
        self._axes[0].set_xlim(
            (t0 - self._t0_ms) / 1000.0, (t1 - self._t0_ms) / 1000.0,
        )
        self.canvas.draw_idle()

    def _on_tk_wheel(self, event, delta: int | None = None) -> None:
        """Tk-level wheel handler (primary path on macOS)."""
        if self.pipeline is None or not self._axes:
            return
        if delta is None:
            delta = getattr(event, "delta", 0)
        if delta == 0:
            return

        # Tk event y is measured from the top of the canvas widget;
        # matplotlib display coords use bottom-up y.
        widget = self.canvas.get_tk_widget()
        h = widget.winfo_height()
        mpl_y = h - event.y

        # Find which axis contains the cursor, else default to a shared-x axis.
        ax = None
        for a in self._axes:
            if a.bbox.contains(event.x, mpl_y):
                ax = a
                break
        if ax is None:
            ax = self._axes[0]

        data_x, data_y = ax.transData.inverted().transform((event.x, mpl_y))
        # Shift modifier → zoom this panel's y-axis only; else zoom shared x-axis.
        shift_held = bool(getattr(event, "state", 0) & 0x0001)
        if shift_held:
            self._zoom_y_at(ax, data_y, zoom_in=(delta > 0))
        else:
            self._zoom_x_at(ax, data_x, zoom_in=(delta > 0))

    # ---------- Tree ----------

    def _refresh_tree(self):
        for item in self.tree.get_children():
            self.tree.delete(item)
        if self.pipeline is None:
            return
        # Ensure schema columns exist (older gt.csv may lack newer ones).
        if "description" not in self.pipeline.gt.columns:
            self.pipeline.gt["description"] = ""
        if "signalClearRecording" not in self.pipeline.gt.columns:
            self.pipeline.gt["signalClearRecording"] = True
        for i, row in self.pipeline.gt.iterrows():
            s_s = (int(row["start_ms"]) - self._t0_ms) / 1000.0
            e_s = (int(row["end_ms"]) - self._t0_ms) / 1000.0
            dur = e_s - s_s
            typ = str(row["type"])
            note = _cell_str(row.get("description", ""))
            clear = "✓" if _coerce_bool(row.get("signalClearRecording", True)) else "✗"
            dh = row.get("height_diff_m", float("nan"))
            dh_s = f"{dh:+.2f}" if pd.notna(dh) else "—"
            self.tree.insert(
                "", tk.END, iid=str(i),
                values=(i, f"{s_s:.1f}", f"{dur:.1f}", typ, dh_s, clear, note),
                tags=(typ,),
            )

    def _on_tree_select(self, _event=None):
        sel = self.tree.selection()
        if not sel or self.pipeline is None:
            return
        idx = int(sel[0])
        if idx not in self.pipeline.gt.index:
            return
        row = self.pipeline.gt.loc[idx]
        self.start_var.set(str(int(row["start_ms"])))
        self.end_var.set(str(int(row["end_ms"])))
        self.type_var.set(str(row["type"]))
        self.desc_var.set(_cell_str(row.get("description", "")))
        self.clear_var.set(_coerce_bool(row.get("signalClearRecording", True)))
        dh = row.get("height_diff_m", float("nan"))
        self.height_diff_var.set(f"{dh:+.3f}" if pd.notna(dh) else "—")
        self._highlight_on_plot(row["start_ms"], row["end_ms"])

    # ---------- Edit actions ----------

    def _recompute_height_diffs(self):
        """Refresh ``height_diff_m`` for every row.

        Uses gramushka snap mode when a populated ``baramoshka.csv`` +
        ``start_floor`` are available for this experiment; otherwise falls
        back to temperature-aware pure-barometer Δh. See
        :func:`src.data.loader.pipeline.addGTtoSegment`.
        """
        if self.pipeline is None:
            return
        exp_name = self.pipeline.metaData.get("exp_name", "") if self.pipeline.metaData else ""
        baramoshka = load_baramoshka(exp_name) if exp_name else None
        self.pipeline.gt = addGTtoSegment(
            self.pipeline.data,
            self.pipeline.gt,
            metadata=self.pipeline.metaData,
            baramoshka=baramoshka,
        )

    def _on_apply(self):
        sel = self.tree.selection()
        if not sel or self.pipeline is None:
            return
        idx = int(sel[0])
        try:
            start = int(self.start_var.get())
            end = int(self.end_var.get())
        except ValueError:
            messagebox.showerror("Invalid", "start/end must be integer ms.")
            return
        typ = self.type_var.get()
        if typ not in VALID_TYPES:
            messagebox.showerror("Invalid", f"type must be one of {VALID_TYPES}.")
            return
        if end <= start:
            messagebox.showerror("Invalid", "end must be > start.")
            return
        note = self.desc_var.get().strip()
        clear = bool(self.clear_var.get())
        self.pipeline.gt.loc[idx, "start_ms"] = start
        self.pipeline.gt.loc[idx, "end_ms"] = end
        self.pipeline.gt.loc[idx, "type"] = typ
        self.pipeline.gt.loc[idx, "description"] = note
        self.pipeline.gt.loc[idx, "signalClearRecording"] = clear
        self._recompute_height_diffs()
        self._mark_dirty(f"Updated interval {idx}")
        self._refresh_plot(preserve_xlim=True)
        self._refresh_tree()
        self.tree.selection_set(str(idx))
        self.tree.see(str(idx))

    def _on_add(self):
        if self.pipeline is None:
            return
        gt = self.pipeline.gt
        sel = self.tree.selection()
        if sel:
            idx = int(sel[0])
            base_end = int(gt.loc[idx, "end_ms"])
            new_start = base_end
        elif len(gt):
            new_start = int(gt["end_ms"].max())
        else:
            new_start = self._t0_ms
        new_end = new_start + 10_000

        new_row = pd.DataFrame(
            [{"start_ms": new_start, "end_ms": new_end,
              "type": "outside", "description": "",
              "signalClearRecording": True}],
            columns=["start_ms", "end_ms", "type", "description",
                     "signalClearRecording"],
        )
        self.pipeline.gt = pd.concat([self.pipeline.gt, new_row], ignore_index=True)
        self._recompute_height_diffs()
        self._mark_dirty("Added interval")
        self._refresh_plot(preserve_xlim=True)
        self._refresh_tree()
        new_idx = str(len(self.pipeline.gt) - 1)
        self.tree.selection_set(new_idx)
        self.tree.see(new_idx)

    def _on_delete(self):
        sel = self.tree.selection()
        if not sel or self.pipeline is None:
            return
        idx = int(sel[0])
        self.pipeline.gt = self.pipeline.gt.drop(index=idx).reset_index(drop=True)
        self._recompute_height_diffs()
        self._mark_dirty(f"Deleted interval {idx}")
        self._refresh_plot(preserve_xlim=True)
        self._refresh_tree()

    def _on_autofix(self):
        if self.pipeline is None:
            return
        gt = self.pipeline.gt.sort_values("start_ms").reset_index(drop=True)
        for i in range(1, len(gt)):
            if int(gt.loc[i, "start_ms"]) < int(gt.loc[i - 1, "end_ms"]):
                messagebox.showwarning(
                    "Overlap",
                    f"Intervals {i-1} and {i} overlap. "
                    f"Fix manually, then try Auto-fix again.",
                )
                self.pipeline.gt = gt
                self._refresh_plot(preserve_xlim=True)
                self._refresh_tree()
                return
        fixed: list[dict] = []
        for i in range(len(gt)):
            if fixed:
                prev_end = fixed[-1]["end_ms"]
                cur_start = int(gt.loc[i, "start_ms"])
                if cur_start > prev_end:
                    fixed.append({
                        "start_ms": prev_end, "end_ms": cur_start,
                        "type": "outside", "description": "",
                        "signalClearRecording": True,
                    })
            fixed.append({
                "start_ms":    int(gt.loc[i, "start_ms"]),
                "end_ms":      int(gt.loc[i, "end_ms"]),
                "type":        str(gt.loc[i, "type"]),
                "description": _cell_str(gt.loc[i].get("description", "")),
                "signalClearRecording": _coerce_bool(
                    gt.loc[i].get("signalClearRecording", True)
                ),
            })
        self.pipeline.gt = pd.DataFrame(
            fixed, columns=["start_ms", "end_ms", "type", "description",
                            "signalClearRecording"],
        )
        self._recompute_height_diffs()
        self._mark_dirty(f"Auto-fixed → {len(self.pipeline.gt)} intervals")
        self._refresh_plot(preserve_xlim=True)
        self._refresh_tree()

    # ---------- Add experiment ----------

    def _open_add_experiment_dialog(self):
        """Pop up a modal form for adding a new experiment, either by
        fetching ACC from S3 or by uploading per-sensor CSV/Excel files.
        On success the new experiment is auto-loaded into the editor."""
        dlg = tk.Toplevel(self)
        dlg.title("Add experiment")
        dlg.transient(self)
        dlg.grab_set()
        dlg.geometry("780x820")

        now = datetime.now().replace(microsecond=0, second=0)

        # Mode toggle (S3 vs Upload).
        mode_var = tk.StringVar(value="s3")
        mode_bar = ttk.Frame(dlg, padding=(8, 8, 8, 0))
        mode_bar.pack(fill=tk.X)
        ttk.Label(mode_bar, text="Source:", font=("", 10, "bold"))\
            .pack(side=tk.LEFT)
        ttk.Radiobutton(mode_bar, text="Fetch from S3", value="s3",
                        variable=mode_var,
                        command=lambda: _on_mode_change())\
            .pack(side=tk.LEFT, padx=8)
        ttk.Radiobutton(mode_bar, text="Upload CSV/Excel files",
                        value="upload", variable=mode_var,
                        command=lambda: _on_mode_change())\
            .pack(side=tk.LEFT)

        # ---- S3 fetch group (mode == "s3") ----
        s3_group = ttk.LabelFrame(dlg, text="S3 fetch", padding=8)
        s3_group.columnconfigure(1, weight=1)

        ttk.Label(s3_group, text="Phone type:")\
            .grid(row=0, column=0, sticky=tk.W, pady=2)
        pt_var = tk.StringVar(value=PhoneType.A.value)
        ttk.Combobox(s3_group, textvariable=pt_var,
                     values=[p.value for p in PhoneType],
                     state="readonly", width=22)\
            .grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(s3_group, text="Phone ID:")\
            .grid(row=1, column=0, sticky=tk.W, pady=2)
        pid_var = tk.StringVar()
        ttk.Entry(s3_group, textvariable=pid_var)\
            .grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(s3_group, text="Start (YYYY-MM-DD HH:MM):")\
            .grid(row=2, column=0, sticky=tk.W, pady=2)
        ts_var = tk.StringVar(
            value=(now - timedelta(minutes=5)).strftime("%Y-%m-%d %H:%M"),
        )
        ttk.Entry(s3_group, textvariable=ts_var)\
            .grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(s3_group, text="End (YYYY-MM-DD HH:MM):")\
            .grid(row=3, column=0, sticky=tk.W, pady=2)
        te_var = tk.StringVar(value=now.strftime("%Y-%m-%d %H:%M"))
        ttk.Entry(s3_group, textvariable=te_var)\
            .grid(row=3, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(s3_group, text="Experiment override:")\
            .grid(row=4, column=0, sticky=tk.W, pady=2)
        eo_var = tk.StringVar()
        ttk.Entry(s3_group, textvariable=eo_var)\
            .grid(row=4, column=1, sticky=tk.EW, padx=4, pady=2)
        ttk.Label(s3_group, text="(optional — local-stub only)",
                  foreground="#888", font=("", 8))\
            .grid(row=5, column=1, sticky=tk.W, padx=4)

        # ---- Upload group (mode == "upload") ----
        up_group = ttk.LabelFrame(
            dlg,
            text="Upload sensor files  (ACC required, others optional)",
            padding=4,
        )
        up_notebook = ttk.Notebook(up_group)
        up_notebook.pack(fill=tk.BOTH, expand=True)
        upload_states: dict[str, dict] = {}
        for sensor_name, data_cols, required in _UPLOAD_SENSORS:
            tab, state = self._build_upload_tab(
                up_notebook, sensor_name, data_cols, required,
            )
            label = f"{sensor_name}*" if required else sensor_name
            up_notebook.add(tab, text=label)
            upload_states[sensor_name] = state

        # ---- Group 2: name parts ----
        g2 = ttk.LabelFrame(
            dlg, text="New experiment name  ({who}_{where}_{phone}_{date})",
            padding=8,
        )
        g2.columnconfigure(1, weight=1)

        ttk.Label(g2, text="Who:").grid(row=0, column=0, sticky=tk.W, pady=2)
        who_var = tk.StringVar()
        ttk.Entry(g2, textvariable=who_var)\
            .grid(row=0, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(g2, text="Where:").grid(row=1, column=0, sticky=tk.W, pady=2)
        where_var = tk.StringVar()
        ttk.Entry(g2, textvariable=where_var)\
            .grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(g2, text="Phone (folder):")\
            .grid(row=2, column=0, sticky=tk.W, pady=2)
        pfolder_var = tk.StringVar()
        ttk.Entry(g2, textvariable=pfolder_var)\
            .grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)

        ttk.Label(g2, text="Date (DD-MM-YYYY):")\
            .grid(row=3, column=0, sticky=tk.W, pady=2)
        date_var = tk.StringVar(value=now.strftime("%d-%m-%Y"))
        ttk.Entry(g2, textvariable=date_var)\
            .grid(row=3, column=1, sticky=tk.EW, padx=4, pady=2)

        name_preview = tk.StringVar(value="(fill in all four fields)")
        ttk.Label(g2, textvariable=name_preview, foreground="#1f77b4",
                  font=("Menlo", 10))\
            .grid(row=4, column=0, columnspan=2, sticky=tk.W,
                  padx=4, pady=(4, 0))

        def _refresh_preview(*_):
            parts = [who_var.get().strip(), where_var.get().strip(),
                     pfolder_var.get().strip(), date_var.get().strip()]
            name_preview.set(
                f"→ {'_'.join(parts)}" if all(parts)
                else "(fill in all four fields)"
            )

        for v in (who_var, where_var, pfolder_var, date_var):
            v.trace_add("write", _refresh_preview)

        # ---- Group 3: metadata fields ----
        g3 = ttk.LabelFrame(dlg, text="Metadata", padding=8)
        g3.columnconfigure(1, weight=1)

        md_vars: dict[str, tk.StringVar] = {}

        # `source` is an enum (experiment / ido / realWorld) — readonly combobox.
        ttk.Label(g3, text="Source:").grid(row=0, column=0, sticky=tk.W, pady=2)
        source_var = tk.StringVar(value=SOURCE_EXPERIMENT)
        ttk.Combobox(g3, textvariable=source_var,
                     values=list(VALID_SOURCES),
                     state="readonly", width=14)\
            .grid(row=0, column=1, sticky=tk.W, padx=4, pady=2)
        md_vars["source"] = source_var

        md_fields = [
            ("description", "Description:", ""),
            ("time", "Time (HH:MM):", now.strftime("%H:%M")),
            ("experiment_type", "Experiment type:", "train"),
            ("temperature_c", "Temperature (°C):", ""),
            ("start_floor", "Start floor:", ""),
        ]
        for i, (key, label, default) in enumerate(md_fields, start=1):
            ttk.Label(g3, text=label).grid(row=i, column=0, sticky=tk.W, pady=2)
            v = tk.StringVar(value=default)
            ttk.Entry(g3, textvariable=v)\
                .grid(row=i, column=1, sticky=tk.EW, padx=4, pady=2)
            md_vars[key] = v

        # ---- Buttons ----
        btns = ttk.Frame(dlg, padding=8)
        ttk.Button(btns, text="Cancel", command=dlg.destroy)\
            .pack(side=tk.RIGHT, padx=4)

        def _on_submit():
            try:
                self._finalize_add_experiment(
                    mode=mode_var.get(),
                    s3_inputs={
                        "phone_type": pt_var.get(),
                        "phone_id": pid_var.get().strip(),
                        "t_start_str": ts_var.get().strip(),
                        "t_end_str": te_var.get().strip(),
                        "experiment_override": eo_var.get().strip(),
                    },
                    upload_states=upload_states,
                    name_parts={
                        "who": who_var.get().strip(),
                        "where": where_var.get().strip(),
                        "phone_folder": pfolder_var.get().strip(),
                        "date_str": date_var.get().strip(),
                    },
                    metadata_extra={k: v.get() for k, v in md_vars.items()},
                    dialog=dlg,
                )
            except _AddExperimentValidationError as e:
                messagebox.showerror("Invalid input", str(e), parent=dlg)
            except Exception as e:
                messagebox.showerror(
                    "Add experiment failed",
                    f"{type(e).__name__}: {e}", parent=dlg,
                )

        ttk.Button(btns, text="Add experiment", command=_on_submit)\
            .pack(side=tk.RIGHT)

        # ---- Layout: re-pack the source-mode group when the toggle flips ----
        def _on_mode_change():
            for w in (s3_group, up_group, g2, g3, btns):
                w.pack_forget()
            mode = mode_var.get()
            if mode == "s3":
                s3_group.pack(fill=tk.X, padx=8, pady=4)
            else:
                up_group.pack(fill=tk.BOTH, expand=True, padx=8, pady=4)
            g2.pack(fill=tk.X, padx=8, pady=4)
            g3.pack(fill=tk.X, padx=8, pady=4)
            btns.pack(fill=tk.X)

        _on_mode_change()

    def _build_upload_tab(
        self, parent: ttk.Notebook, sensor_name: str,
        data_cols: list[str], required: bool,
    ) -> tuple[ttk.Frame, dict]:
        """Build one tab of the manual-upload notebook.

        Returns ``(frame, state)`` where ``state`` is a mutable dict the
        orchestrator inspects on submit:

        * ``df`` — the raw uploaded DataFrame (or None)
        * ``col_vars`` — ``{"time": StringVar, **{col: StringVar for col in data_cols}}``
        * ``day_first`` — BooleanVar for ambiguous DD/MM dates
        * ``required`` / ``data_cols`` / ``sensor`` — passthrough metadata
        """
        frame = ttk.Frame(parent, padding=10)
        frame.columnconfigure(1, weight=1)

        state: dict = {
            "sensor": sensor_name,
            "data_cols": list(data_cols),
            "required": required,
            "path_var": tk.StringVar(value=""),
            "df": None,
            "col_vars": {"time": tk.StringVar()},
            "day_first": tk.BooleanVar(value=True),
            "status_var": tk.StringVar(value="No file selected."),
            "_combos": [],   # filled below; populated when a file loads
        }
        for c in data_cols:
            state["col_vars"][c] = tk.StringVar()

        header = (f"{sensor_name} — required" if required
                  else f"{sensor_name} — optional")
        ttk.Label(frame, text=header, font=("", 10, "bold"))\
            .grid(row=0, column=0, columnspan=3, sticky=tk.W, pady=(0, 6))

        # File picker.
        ttk.Label(frame, text="File:").grid(row=1, column=0, sticky=tk.W, pady=2)
        ttk.Entry(frame, textvariable=state["path_var"], state="readonly")\
            .grid(row=1, column=1, sticky=tk.EW, padx=4, pady=2)

        def _on_pick():
            path = filedialog.askopenfilename(
                title=f"Select {sensor_name} CSV/Excel",
                filetypes=[
                    ("CSV / Excel", "*.csv *.xlsx *.xls"),
                    ("All files", "*.*"),
                ],
                parent=frame.winfo_toplevel(),
            )
            if not path:
                return
            try:
                df = _read_tabular(Path(path))
            except Exception as e:
                state["df"] = None
                state["status_var"].set(f"Read error: {e}")
                state["path_var"].set(path)
                return
            state["df"] = df
            state["path_var"].set(path)
            cols = list(df.columns)

            # Pre-fill mappings with sensible guesses; populate combobox values.
            state["col_vars"]["time"].set(_guess_column(
                cols, ["timestamp_ms", "time", "ts", "t"],
            ))
            for c in data_cols:
                state["col_vars"][c].set(_guess_column(cols, [c]))
            for cb in state["_combos"]:
                cb["values"] = cols
            state["status_var"].set(
                f"Loaded {len(df):,} rows × {len(cols)} columns. "
                f"Confirm column mapping below."
            )

        ttk.Button(frame, text="Browse…", command=_on_pick)\
            .grid(row=1, column=2, padx=4, pady=2)

        # Time column + day-first toggle.
        ttk.Label(frame, text="Time column:")\
            .grid(row=2, column=0, sticky=tk.W, pady=2)
        cb_time = ttk.Combobox(frame, textvariable=state["col_vars"]["time"],
                               values=[], state="readonly")
        cb_time.grid(row=2, column=1, sticky=tk.EW, padx=4, pady=2)
        state["_combos"].append(cb_time)

        ttk.Checkbutton(frame, text="Day-first (DD/MM)?",
                        variable=state["day_first"])\
            .grid(row=2, column=2, sticky=tk.W, padx=4, pady=2)

        # Data columns.
        for i, col in enumerate(data_cols, start=3):
            ttk.Label(frame, text=f"{col} column:")\
                .grid(row=i, column=0, sticky=tk.W, pady=2)
            cb = ttk.Combobox(frame, textvariable=state["col_vars"][col],
                              values=[], state="readonly")
            cb.grid(row=i, column=1, sticky=tk.EW, padx=4, pady=2)
            state["_combos"].append(cb)

        # Status footer.
        ttk.Label(frame, textvariable=state["status_var"], foreground="#444",
                  wraplength=620, justify=tk.LEFT)\
            .grid(row=20, column=0, columnspan=3, sticky=tk.W, pady=(8, 0))
        if not required:
            ttk.Label(frame,
                      text="(leave blank to skip — segmenter will fall "
                           "back to ACC.)",
                      foreground="#888", font=("", 8))\
                .grid(row=21, column=0, columnspan=3, sticky=tk.W)

        return frame, state

    def _finalize_add_experiment(
        self, *,
        mode: str,
        s3_inputs: dict,
        upload_states: dict[str, dict],
        name_parts: dict,
        metadata_extra: dict[str, str],
        dialog: tk.Toplevel,
    ) -> None:
        """Validate inputs, gather sensors (S3 or upload), auto-segment, and
        persist as a new experiment under structuredData/. On success the
        new experiment is auto-loaded into the editor."""
        # --- Shared validation: name parts ---
        who = name_parts["who"]
        where = name_parts["where"]
        phone_folder = name_parts["phone_folder"]
        date_str = name_parts["date_str"]
        parts = [who, where, phone_folder, date_str]
        if not all(parts):
            raise _AddExperimentValidationError(
                "All four name parts (who, where, phone, date) are required."
            )
        name = "_".join(parts)
        bad = set(name) & set("/\\:")
        if bad:
            raise _AddExperimentValidationError(
                f"Name parts must not contain {sorted(bad)!r}."
            )

        if (STRUCTURED_DATA_DIR / name).exists():
            if not messagebox.askyesno(
                "Overwrite?",
                f"{name} already exists under structuredData/data/. Overwrite?",
                parent=dialog,
            ):
                return

        # --- Mode-specific sensor loading ---
        if mode == "s3":
            sensors = self._fetch_sensors_from_s3(s3_inputs, name=name)
        else:
            sensors = self._sensors_from_uploads(upload_states)

        if "ACC" not in sensors or sensors["ACC"].empty:
            raise _AddExperimentValidationError(
                "ACC data is required (S3 returned no ACC or no ACC file uploaded)."
            )

        # --- Auto-segment: PRS path if available, else ACC template-match. ---
        self.status_var.set(f"Segmenting {name}…")
        self.update_idletasks()
        gt = self._auto_segment_for_new_experiment(sensors, phone_folder)

        # --- Stamp exp_name (saveExperimentData does not auto-inject —
        #     mirrors the first-time-build path in pipeline.py that calls
        #     _inject_exp_name explicitly before _write_csvs). ---
        for df in sensors.values():
            df["exp_name"] = name
        gt["exp_name"] = name

        metadata = {
            "exp_name":        name,
            "experimenter":    who,
            "phone":           phone_folder,
            "location":        where,
            "description":     metadata_extra.get("description", ""),
            "date":            date_str,
            "time":            metadata_extra.get("time", ""),
            "experiment_type": metadata_extra.get("experiment_type", "train"),
            "temperature_c":   metadata_extra.get("temperature_c", ""),
            "start_floor":     metadata_extra.get("start_floor", ""),
            "source":          metadata_extra.get("source", SOURCE_EXPERIMENT),
        }

        saved_dir = saveExperimentData(name, sensors, gt, metadata)

        # Refresh the dropdown so the new experiment shows up, then load it.
        self._populate_experiments()
        self.exp_var.set(name)
        dialog.destroy()
        n_rides = int((gt["type"] != "outside").sum())
        sensor_summary = ", ".join(
            f"{k}:{len(v):,}" for k, v in sensors.items()
        )
        self.status_var.set(
            f"Loaded [{sensor_summary}] → segmented {n_rides} rides "
            f"→ saved to {saved_dir}"
        )
        self.load_experiment()

    def _fetch_sensors_from_s3(
        self, inp: dict, *, name: str,
    ) -> dict[str, pd.DataFrame]:
        """Validate the S3 form fields, call loadDataFromS3, return the
        full ``{sensor: df}`` bundle the backend produced.

        ACC is always present; PRS / GYR / MAG / ORI come through
        whenever S3 has them. PRS is augmented with ``GT_height_m``
        (ISA inversion of ``pressure``) so the auto-segmenter can take
        the barometer-first path — mirrors what
        :meth:`_build_canonical_sensor_df` does on the upload route.
        """
        phone_id = inp["phone_id"]
        if not (phone_id and phone_id.isdigit()):
            raise _AddExperimentValidationError("Phone ID must be digits only.")
        try:
            t_start = datetime.strptime(inp["t_start_str"], "%Y-%m-%d %H:%M")
            t_end = datetime.strptime(inp["t_end_str"], "%Y-%m-%d %H:%M")
        except ValueError as e:
            raise _AddExperimentValidationError(
                f"Datetime must be 'YYYY-MM-DD HH:MM' ({e})."
            ) from None
        if t_end <= t_start:
            raise _AddExperimentValidationError("End must be after start.")

        self.status_var.set(f"Fetching {name} from S3…")
        self.update_idletasks()
        loaded: LoadedSignal = loadDataFromS3(
            PhoneType(inp["phone_type"]), phone_id, t_start, t_end,
            experiment=inp["experiment_override"] or None,
        )
        out: dict[str, pd.DataFrame] = {"ACC": loaded.acc.copy()}
        for key, df in (
            ("PRS", loaded.prs), ("GYR", loaded.gyr),
            ("MAG", loaded.mag), ("ORI", loaded.ori),
        ):
            if df is not None and not df.empty:
                out[key] = df.copy()
        prs = out.get("PRS")
        if prs is not None and "pressure" in prs.columns:
            prs["GT_height_m"] = pressure_to_altitude(prs["pressure"].to_numpy())
        return out

    def _sensors_from_uploads(
        self, upload_states: dict[str, dict],
    ) -> dict[str, pd.DataFrame]:
        """Read each uploaded file's column mapping into a canonical-schema
        DataFrame. ACC is required; the rest are skipped if no file was
        picked. PRS additionally derives ``GT_height_m`` via ISA inversion."""
        out: dict[str, pd.DataFrame] = {}
        for sensor_name, _, required in _UPLOAD_SENSORS:
            state = upload_states[sensor_name]
            if state["df"] is None:
                if required:
                    raise _AddExperimentValidationError(
                        f"{sensor_name} file is required."
                    )
                continue
            df = self._build_canonical_sensor_df(state)
            if df.empty:
                if required:
                    raise _AddExperimentValidationError(
                        f"{sensor_name} file produced no rows after parsing."
                    )
                continue
            out[sensor_name] = df
        return out

    def _build_canonical_sensor_df(self, state: dict) -> pd.DataFrame:
        """Apply the column mapping from one upload-tab state to its raw
        DataFrame and return a canonical-schema DataFrame
        (`timestamp_ms` + ``data_cols``). For PRS, also derives
        `GT_height_m` from `pressure` via :func:`pressure_to_altitude`."""
        sensor = state["sensor"]
        data_cols = state["data_cols"]
        time_col = state["col_vars"]["time"].get()
        if not time_col:
            raise _AddExperimentValidationError(
                f"{sensor}: no time column selected."
            )
        col_map = {c: state["col_vars"][c].get() for c in data_cols}
        missing = [c for c, v in col_map.items() if not v]
        if missing:
            raise _AddExperimentValidationError(
                f"{sensor}: data columns not selected: {missing}."
            )

        used = [time_col, *col_map.values()]
        if len(set(used)) != len(used):
            raise _AddExperimentValidationError(
                f"{sensor}: each canonical column must map to a different "
                f"source column (got {used})."
            )

        raw = state["df"]
        for c in used:
            if c not in raw.columns:
                raise _AddExperimentValidationError(
                    f"{sensor}: column {c!r} not found in uploaded file."
                )
        sub = raw[used].dropna()
        if sub.empty:
            return pd.DataFrame(columns=["timestamp_ms", *data_cols])

        try:
            ts_ms = _parse_time_to_epoch_ms(
                sub[time_col], day_first=bool(state["day_first"].get()),
            )
        except ValueError as e:
            raise _AddExperimentValidationError(f"{sensor}: {e}") from None

        df = pd.DataFrame({"timestamp_ms": ts_ms.astype("int64")})
        for canon, src in col_map.items():
            df[canon] = pd.to_numeric(sub[src], errors="coerce").to_numpy(dtype=float)
        df = (df.dropna()
                .sort_values("timestamp_ms")
                .drop_duplicates("timestamp_ms", keep="first")
                .reset_index(drop=True))

        if sensor == "PRS" and "pressure" in df.columns:
            # ISA inversion — the same conversion the raw-log loader uses
            # (parsing.py:81). Temperature defaults to 15 °C; the metadata
            # `temperature_c` field is consulted later when GT Δh is recomputed.
            df["GT_height_m"] = pressure_to_altitude(df["pressure"].to_numpy())
        return df

    def _auto_segment_for_new_experiment(
        self, sensors: dict[str, pd.DataFrame], phone_folder: str,
    ) -> pd.DataFrame:
        """Pick the segmenter based on which sensors are present.

        Barometer-first: when PRS is present and has ``GT_height_m``, run
        the pressure-filter segmenter via the existing
        :func:`_derive_gt_from_prs` helper (used by the raw-log loader).
        Else fall back to the accelerometer-only template-match detector.
        """
        prs = sensors.get("PRS")
        if prs is not None and not prs.empty and "GT_height_m" in prs.columns:
            return _derive_gt_from_prs(prs)

        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            raise _AddExperimentValidationError(
                "Cannot segment: neither PRS nor ACC data was provided."
            )
        seg_cfg = SEGMENT_ALGORITHM_CONFIG(
            algorithm=SegmentAlgorithm.ACC_TEMPLATE_MATCH,
        )
        segments = Segmenter(seg_cfg).detect(acc, phone_model=phone_folder)
        t0_ms = int(acc["timestamp_ms"].iloc[0])
        t_end_ms = int(acc["timestamp_ms"].iloc[-1])
        return _segments_to_full_gt(segments, t0_ms, t_end_ms)

    # ---------- Housekeeping ----------

    def _mark_dirty(self, msg: str):
        self._dirty = True
        self.status_var.set(f"{msg} — unsaved (Ctrl+S)")

    def _on_close(self):
        if self._dirty and not messagebox.askyesno(
            "Unsaved changes", "Quit without saving?",
        ):
            return
        self.destroy()


def main():
    pre = sys.argv[1] if len(sys.argv) > 1 else None
    app = GtEditor(pre)
    app.mainloop()


if __name__ == "__main__":
    main()
