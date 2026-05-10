"""Tkinter GUI for inspecting whole-signal trapezoid detections.

Sister tool to ``src/data/gt_editor.py``. Loads an experiment's sensor
data (via ``getExperimentData``) and then runs the detector in
:mod:`detect` **live** each time an experiment is loaded. No JSON on
disk, no stale PNG. Edit the tunables at the top of ``detect.py`` and
re-load an experiment to iterate.

Two tables on the right:

* **Predictions** — pairs accepted by the detector. Clicking a row
  highlights its span on the main plot and fills the detail panel with:
    - Heatmap of the matched-filter R² across the full ``(W, f)`` grid
      at ``lobe1.t_c``.
    - Same at ``lobe2.t_c``.
    - Zoomed ``a_vert`` with the fitted trapezoids overlaid.
    - Threshold verdict text.

* **GT rides** — every ``up`` / ``down`` interval from ``gt.csv``,
  regardless of whether the detector accepted it. Clicking a GT row
  still populates the detail panel (heatmaps at the best ``+`` / ``−``
  samples found **inside** the GT window) together with a diagnostic
  block explaining which specific threshold rejected the interval
  (``R²_PEAK_THRESH``, ``MIN_PEAK_ABS_A``, ``JOINT_R2_THRESH``, …).

Read-only. Use ``gt_editor.py`` for GT edits.

Usage:
    venv/bin/python src/segmentation/algorithms/editor.py [exp_folder_name]

Predict button
--------------
Each selected interval (prediction or GT) can be run through every
:class:`src.prediction.algorithms.Predictor` algorithm by clicking
**Predict** in the detail header. The verdict pane then shows Δh + CI
per algorithm alongside the barometer-derived GT Δh.
"""

from __future__ import annotations

import math
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import numpy as np  # noqa: E402
from matplotlib.backends.backend_tkagg import (  # noqa: E402
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from matplotlib.figure import Figure  # noqa: E402

#changes
# Make absolute ``src.*`` imports resolve when the editor is launched as
# a standalone script (``python .../editor.py``) as well as via ``-m``.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.gt_editor import PANEL_DRAWERS as _GT_PANEL_DRAWERS  # noqa: E402
from src.data.gt_editor import TYPE_COLORS  # noqa: E402
from src.data.loader import (  # noqa: E402
    RAW_DATA_ROOT,
    STRUCTURED_DATA_DIR,
    getExperimentData,
    list_experiments,
    list_structured_experiments,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E402
    SMOOTH_SEC, trapezoid_kernel,
    _estimate_fs_hz, _vertical_accel, _smooth,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect,
)
from src.prediction.algorithms import (  # noqa: E402
    Predictor,
    PREDICT_ALGORITHM_CONFIG,
    PredictAlgorithm,
)
from src.physics.barometric import pressure_to_altitude  # noqa: E402

predict_intervals = _detect.predict_intervals
diagnose_window = _detect.diagnose_window

PRED_COLORS = {"up": "#1f3a5f", "down": "#7d3c98"}
HIGHLIGHT_COLOR = "#e67e22"

# Seconds of pre/post stationary window passed to accelerometer predictors
# for gravity calibration. 2 s at typical phone sampling rates (50–200 Hz)
# gives 100–400 samples — comfortably above the grav_window_sec · fs floor
# inside ``estimate_gravity_stationary``.
PREDICT_PRE_POST_SEC = 2.0


# --------------------------------------------------------------------------
# Custom a_vert panel — same signed, gravity-projected channel the
# detector works on. Replaces the GT editor's |a| magnitude panel.
# --------------------------------------------------------------------------

def _draw_a_vert(ax, data, t0_ms):
    df = data["ACC"]
    ts_ms = df["timestamp_ms"].to_numpy(dtype=float)
    if ts_ms.size == 0:
        ax.set_ylabel("$a_\\mathrm{vert}$ (m/s²)")
        return
    t = (ts_ms - t0_ms) / 1000.0
    fs = _estimate_fs_hz(ts_ms)
    ax_arr = df["x"].to_numpy(dtype=float)
    ay_arr = df["y"].to_numpy(dtype=float)
    az_arr = df["z"].to_numpy(dtype=float)
    a_vert = _vertical_accel(ax_arr, ay_arr, az_arr, fs)
    a_smooth = _smooth(a_vert, fs, SMOOTH_SEC)
    ax.plot(t, a_vert, color="#2c3e50", lw=0.5, label="$a_\\mathrm{vert}$")
    ax.plot(t, a_smooth, color="#e67e22", lw=0.9, alpha=0.9, label="smoothed")
    ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
    ax.set_ylabel("$a_\\mathrm{vert}$ (m/s²)")
    ax.legend(loc="upper right", fontsize=7, frameon=False)


PANEL_DRAWERS = [
    ("altitude",     "PRS", _GT_PANEL_DRAWERS[0][2]),
    ("velocity",     "PRS", _GT_PANEL_DRAWERS[1][2]),
    ("a_vert",       "ACC", _draw_a_vert),
    ("acc_velocity", "ACC", _GT_PANEL_DRAWERS[3][2]),
    ("gyr",          "GYR", _GT_PANEL_DRAWERS[4][2]),
    ("mag",          "MAG", _GT_PANEL_DRAWERS[5][2]),
]


# --------------------------------------------------------------------------
# Main window
# --------------------------------------------------------------------------

class PredictionEditor(tk.Tk):
    def __init__(self, preselect: str | None = None):
        super().__init__()
        self.title("Prediction Editor")
        self.geometry("1700x950")

        self.exp_name: str | None = None
        self.sensors: dict | None = None
        self.gt = None
        self._t0_ms: int = 0
        self._acc_t0_ms: float = 0.0

        # Cached detector state (arrays; shared across tree selections).
        self._state: dict | None = None
        self.predictions: list[dict] = []

        # Per-experiment ACC valid intervals (Unix epoch ms). Populated by
        # ``load_experiment``; drives the red "no data" overlays and the
        # per-interval detector loop.
        self._acc_valid_intervals: list[tuple[int, int]] = []

        self._axes: list = []
        self._hl_spans: list = []

        # Last selection so the detail pane can re-render when the user
        # widens / narrows the viewing window.
        self._last_sel: tuple[str, tuple] | None = None
        self.detail_pad_var = tk.DoubleVar(value=5.0)

        self._build_ui()
        self._populate_experiments()

        if preselect:
            self.exp_var.set(preselect)
            self.after(50, self.load_experiment)

        # Set the trees ↔ detail divider once the window has a real size,
        # so the GT-rides table gets a usable share of vertical space even
        # though the detail pane now hosts a tabbed notebook.
        self.after(150, self._fit_right_pane_sash)

        self.protocol("WM_DELETE_WINDOW", self.destroy)

    def _fit_right_pane_sash(self) -> None:
        try:
            self.update_idletasks()
            h = self._right_pane.winfo_height()
            if h > 200:
                self._right_pane.sashpos(0, int(h * 0.50))
        except (tk.TclError, AttributeError):
            pass

    # ---------- UI construction ----------

    def _build_ui(self):
        top = ttk.Frame(self, padding=6)
        top.pack(fill=tk.X)
        ttk.Label(top, text="Experiment:").pack(side=tk.LEFT)
        self.exp_var = tk.StringVar()
        self.exp_combo = ttk.Combobox(
            top, textvariable=self.exp_var, width=50, state="readonly",
        )
        self.exp_combo.pack(side=tk.LEFT, padx=6)
        ttk.Button(top, text="Load", command=self.load_experiment)\
            .pack(side=tk.LEFT)
        ttk.Button(top, text="Copy name", command=self._copy_exp_name)\
            .pack(side=tk.LEFT, padx=(4, 0))

        # Zoom / pan nav (same buttons as gt_editor.py).
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
                  font=("Menlo", 10)).pack(side=tk.RIGHT, padx=10)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        # --- Left: sensor panels ---
        left = ttk.Frame(main)
        main.add(left, weight=4)
        self.fig = Figure(figsize=(11, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=left)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, left).update()

        # Click on the plot → select the prediction / GT interval under
        # the cursor. Motion handler updates the hover readout + cursor.
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("scroll_event",        self._on_scroll)

        # Tk-level wheel binding — mpl scroll_event is unreliable on macOS
        # TkAgg, this is the primary path there.
        widget = self.canvas.get_tk_widget()
        widget.bind("<MouseWheel>", self._on_tk_wheel)
        widget.bind("<Button-4>", lambda e: self._on_tk_wheel(e, delta=+120))
        widget.bind("<Button-5>", lambda e: self._on_tk_wheel(e, delta=-120))

        # Keyboard shortcuts, mirror of gt_editor.py.
        self.bind("<plus>",        lambda e: self._zoom_x_around_center(zoom_in=True))
        self.bind("<equal>",       lambda e: self._zoom_x_around_center(zoom_in=True))
        self.bind("<minus>",       lambda e: self._zoom_x_around_center(zoom_in=False))
        self.bind("<Key-0>",       lambda e: self._fit_x())
        self.bind("<Shift-Left>",  lambda e: self._pan_x(-0.25))
        self.bind("<Shift-Right>", lambda e: self._pan_x(+0.25))
        self.bind("<Shift-Up>",    lambda e: self._pan_y(+0.25))
        self.bind("<Shift-Down>",  lambda e: self._pan_y(-0.25))

        # --- Right: (predictions + GT) stacked over detail pane ---
        right_pane = ttk.Panedwindow(main, orient=tk.VERTICAL)
        main.add(right_pane, weight=2)
        self._right_pane = right_pane

        # Trees frame with two treeviews stacked vertically.
        trees_frame = ttk.Frame(right_pane, padding=4)
        right_pane.add(trees_frame, weight=1)

        pred_block = ttk.Frame(trees_frame)
        pred_block.pack(fill=tk.BOTH, expand=True)
        ttk.Label(pred_block, text="Predictions", font=("", 10, "bold"))\
            .pack(anchor=tk.W)
        self.tree_pred = self._make_tree(
            pred_block,
            columns=[
                ("idx",     "#",         36, tk.E),
                ("start_s", "start (s)", 74, tk.E),
                ("end_s",   "end (s)",   74, tk.E),
                ("dur_s",   "dur (s)",   60, tk.E),
                ("type",    "type",      56, tk.CENTER),
                ("r2",      "joint R²",  62, tk.E),
                ("abs_A",   "|A|",       50, tk.E),
                ("W",       "W (s)",     54, tk.E),
                ("f",       "f",         50, tk.E),
            ],
            on_select=self._on_pred_select,
        )
        self.tree_pred.tag_configure("up",   foreground=PRED_COLORS["up"])
        self.tree_pred.tag_configure("down", foreground=PRED_COLORS["down"])

        gt_block = ttk.Frame(trees_frame)
        gt_block.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(gt_block, text="GT rides (click to diagnose)",
                  font=("", 10, "bold")).pack(anchor=tk.W)
        self.tree_gt = self._make_tree(
            gt_block,
            columns=[
                ("idx",     "#",         36, tk.E),
                ("start_s", "start (s)", 74, tk.E),
                ("end_s",   "end (s)",   74, tk.E),
                ("dur_s",   "dur (s)",   60, tk.E),
                ("type",    "type",      56, tk.CENTER),
                ("status",  "detector",  140, tk.W),
            ],
            on_select=self._on_gt_select,
        )
        self.tree_gt.tag_configure("up",   foreground=TYPE_COLORS["up"])
        self.tree_gt.tag_configure("down", foreground=TYPE_COLORS["down"])

        # Detail pane (bottom-right) — matplotlib figure on top + text box.
        detail_frame = ttk.Frame(right_pane, padding=4)
        right_pane.add(detail_frame, weight=2)
        detail_header = ttk.Frame(detail_frame)
        detail_header.pack(fill=tk.X)
        ttk.Label(detail_header, text="Detail — heatmaps + signal + verdict",
                  font=("", 10, "bold")).pack(side=tk.LEFT, anchor=tk.W)
        ttk.Label(detail_header, text="  Window ± (s):").pack(side=tk.LEFT)
        self.detail_pad_spin = ttk.Spinbox(
            detail_header, from_=1.0, to=600.0, increment=2.0, width=6,
            textvariable=self.detail_pad_var,
            command=self._on_pad_changed,
        )
        self.detail_pad_spin.pack(side=tk.LEFT, padx=(2, 4))
        self.detail_pad_spin.bind("<Return>", lambda _e: self._on_pad_changed())
        self.detail_pad_spin.bind("<FocusOut>", lambda _e: self._on_pad_changed())
        ttk.Button(detail_header, text="×2", width=3,
                   command=lambda: self._scale_pad(2.0))\
            .pack(side=tk.LEFT, padx=1)
        ttk.Button(detail_header, text="÷2", width=3,
                   command=lambda: self._scale_pad(0.5))\
            .pack(side=tk.LEFT, padx=1)
        ttk.Separator(detail_header, orient=tk.VERTICAL)\
            .pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Button(detail_header, text="▶ Predict all algorithms",
                   command=self._on_predict_clicked)\
            .pack(side=tk.LEFT, padx=(2, 0))
        # Tabbed detail area: Segmentation (default — heatmaps, signal,
        # signed-R²) and Prediction (per-algorithm CI, quality, trapezoid
        # template, ZUPT position). verdict_text stays shared below.
        self.detail_nb = ttk.Notebook(detail_frame)
        self.detail_nb.pack(fill=tk.BOTH, expand=True)

        seg_tab = ttk.Frame(self.detail_nb)
        self.detail_nb.add(seg_tab, text="Segmentation")
        self.detail_fig = Figure(figsize=(6, 5))
        self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=seg_tab)
        self.detail_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        pred_tab = ttk.Frame(self.detail_nb)
        self.detail_nb.add(pred_tab, text="Prediction")
        pred_split = ttk.Panedwindow(pred_tab, orient=tk.VERTICAL)
        pred_split.pack(fill=tk.BOTH, expand=True)
        pred_table_frame = ttk.Frame(pred_split, padding=2)
        pred_split.add(pred_table_frame, weight=1)
        self.pred_table = self._make_tree(
            pred_table_frame,
            columns=[
                ("algo",    "algorithm",  170, tk.W),
                ("dh",      "Δh (m)",      80, tk.E),
                ("ci",      "CI ± (m)",    80, tk.E),
                ("sigma",   "σ (m)",       70, tk.E),
                ("q",       "q",           60, tk.E),
                ("verdict", "verdict",     90, tk.W),
                ("reason",  "reason",     220, tk.W),
            ],
            on_select=lambda _e: None,
        )
        self.pred_table.configure(height=4)
        self.pred_table.tag_configure("ok",     foreground="#1e7a3a")
        self.pred_table.tag_configure("reject", foreground="#a04000")
        self.pred_table.tag_configure("error",  foreground="#7f0000")
        self.pred_table.tag_configure("gt",     foreground="#555")

        pred_canvas_frame = ttk.Frame(pred_split, padding=2)
        pred_split.add(pred_canvas_frame, weight=3)
        self.pred_fig = Figure(figsize=(6, 7))
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=pred_canvas_frame)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        self.verdict_text = tk.Text(
            detail_frame, height=7, wrap="word", font=("Menlo", 9),
            background="#fafafa", foreground="#222", borderwidth=1, relief="solid",
        )
        self.verdict_text.pack(fill=tk.X, pady=(6, 0))
        self.verdict_text.configure(state="disabled")
        self._detail_placeholder()
        self._pred_placeholder()

    def _make_tree(self, parent, columns, on_select):
        inner = ttk.Frame(parent)
        inner.pack(fill=tk.BOTH, expand=True)
        col_ids = [c[0] for c in columns]
        tree = ttk.Treeview(
            inner, columns=col_ids, show="headings", selectmode="browse",
            height=8,
        )
        for cid, text, w, anchor in columns:
            tree.heading(cid, text=text)
            tree.column(cid, width=w, anchor=anchor)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
        scroll = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=tree.yview)
        tree.configure(yscrollcommand=scroll.set)
        scroll.pack(side=tk.RIGHT, fill=tk.Y)
        tree.bind("<<TreeviewSelect>>", on_select)
        return tree

    # ---------- Experiments + loading ----------

    def _populate_experiments(self):
        # Merge raw and structured experiments so ones materialised directly
        # under structuredData/ (e.g. via the GT-editor S3 ingest / upload)
        # are also pickable here. Mirrors gt_editor._populate_experiments.
        raw = list_experiments(RAW_DATA_ROOT)
        structured = list_structured_experiments(STRUCTURED_DATA_DIR)
        seen = set(raw)
        self.exp_combo["values"] = list(raw) + [n for n in structured if n not in seen]

    def _copy_exp_name(self):
        name = self.exp_var.get()
        if not name:
            self.status_var.set("No experiment name to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(name)
        self.update()
        self.status_var.set(f"Copied: {name}")

    def load_experiment(self):
        name = self.exp_var.get()
        if not name:
            self.status_var.set("Pick an experiment first.")
            return
        try:
            sensors, gt, _meta = getExperimentData(RAW_DATA_ROOT / name, use_cache=True)
        except Exception as e:
            messagebox.showerror("Load failed", f"{type(e).__name__}: {e}")
            return

        self.exp_name = name
        self.sensors = sensors
        self.gt = gt

        prs = sensors.get("PRS")
        acc = sensors.get("ACC")
        if prs is not None and not prs.empty:
            self._t0_ms = int(prs["timestamp_ms"].iloc[0])
        elif acc is not None and not acc.empty:
            self._t0_ms = int(acc["timestamp_ms"].iloc[0])
        else:
            self._t0_ms = 0

        self._acc_t0_ms = 0.0
        if acc is not None and not acc.empty:
            self._acc_t0_ms = float(acc["timestamp_ms"].iloc[0])

        # Per-sensor valid intervals were stashed by the loader; ACC drives
        # this UI's detection and plotting. Anything outside is a "no data"
        # gap that the user must not label.
        self._acc_valid_intervals: list[tuple[int, int]] = (
            gt.attrs.get("valid_intervals_per_sensor", {}).get("ACC", [])
            if gt is not None else []
        )

        self.status_var.set(f"Running detector on {name}…")
        self.update_idletasks()
        try:
            # Run the detector once on the full ACC for a canonical state
            # (full-timeline arrays drive the heatmap / correlation panels).
            # Then re-run per valid interval and replace the predictions —
            # per-interval calls cannot produce false matches that span a
            # gap, since the detector never sees both sides at once.
            self.predictions, state = predict_intervals(acc)
            self._state = state if state else None
            if (acc is not None and not acc.empty
                    and len(self._acc_valid_intervals) > 1):
                ts = acc["timestamp_ms"].astype("int64").to_numpy()
                acc_t0_ms = float(acc["timestamp_ms"].iloc[0])
                preds: list[dict] = []
                for s_ms, e_ms in self._acc_valid_intervals:
                    mask = (ts >= s_ms) & (ts <= e_ms)
                    chunk = acc.loc[mask].reset_index(drop=True)
                    if len(chunk) < 2:
                        continue
                    try:
                        chunk_preds, _chunk_state = predict_intervals(chunk)
                    except Exception:  # noqa: BLE001
                        continue
                    chunk_t0_ms = float(chunk["timestamp_ms"].iloc[0])
                    shift_s = (chunk_t0_ms - acc_t0_ms) / 1000.0
                    for p in chunk_preds:
                        q = dict(p)
                        q["t_start_s"] = float(p["t_start_s"]) + shift_s
                        q["t_end_s"] = float(p["t_end_s"]) + shift_s
                        if "lobe1" in q and isinstance(q["lobe1"], dict) and "t_c" in q["lobe1"]:
                            q["lobe1"] = {**q["lobe1"], "t_c": float(q["lobe1"]["t_c"]) + shift_s}
                        if "lobe2" in q and isinstance(q["lobe2"], dict) and "t_c" in q["lobe2"]:
                            q["lobe2"] = {**q["lobe2"], "t_c": float(q["lobe2"]["t_c"]) + shift_s}
                        preds.append(q)
                # Renumber so prediction "index" tags stay unique post-merge
                # (the detail tables key off these indices).
                for i, p in enumerate(preds):
                    p["index"] = i
                self.predictions = preds
        except Exception as e:
            messagebox.showerror("Detector failed", f"{type(e).__name__}: {e}")
            self._state = None
            self.predictions = []

        self._refresh_plot()
        self._refresh_pred_tree()
        self._refresh_gt_tree()
        self._detail_placeholder()
        pred_n = len(self.predictions)
        gt_n = 0 if gt is None else int(
            gt["type"].isin(("up", "down")).sum() if "type" in gt.columns else 0
        )
        self.status_var.set(
            f"Loaded {name} — {pred_n} predictions (live), {gt_n} GT rides."
        )

    # ---------- Main plot ----------

    def _refresh_plot(self):
        self.fig.clear()
        self._axes = []
        self._hl_spans = []
        if self.sensors is None:
            self.canvas.draw()
            return
        panels = [
            (name, drawer)
            for name, sensor, drawer in PANEL_DRAWERS
            if sensor in self.sensors and not self.sensors[sensor].empty
        ]
        if not panels:
            self.canvas.draw()
            return
        axes = self.fig.subplots(len(panels), 1, sharex=True, squeeze=False)[:, 0]
        for ax, (name, drawer) in zip(axes, panels):
            drawer(ax, self.sensors, self._t0_ms)
            ax.grid(True, alpha=0.3)
            ax.set_title(name, fontsize=9, loc="left")
        axes[-1].set_xlabel("time (s)")

        if self.gt is not None and not self.gt.empty:
            for _, row in self.gt.iterrows():
                rt = str(row.get("type", ""))
                if rt not in ("up", "down"):
                    continue
                s = (int(row["start_ms"]) - self._t0_ms) / 1000.0
                e = (int(row["end_ms"]) - self._t0_ms) / 1000.0
                color = TYPE_COLORS.get(rt, "#cccccc")
                for ax in axes:
                    ax.axvspan(s, e, color=color, alpha=0.15, zorder=0)

        offset = self._acc_offset_seconds()
        for p in self.predictions:
            s = p["t_start_s"] + offset
            e = p["t_end_s"] + offset
            col = PRED_COLORS[p["ride_type"]]
            for ax in axes:
                ax.axvspan(s, e, color=col, alpha=0.18, hatch="//", zorder=1)
                ax.axvline(s, color=col, lw=0.9, ls="--", alpha=0.7, zorder=2)
                ax.axvline(e, color=col, lw=0.9, ls="--", alpha=0.7, zorder=2)

        # Red "no data" overlays for ACC gap regions. These tell the user
        # not to mark GT inside the highlighted spans (the segmenter and
        # predictor skip them too — see load_experiment / streamlit step3).
        if self._acc_valid_intervals:
            acc_s = self.sensors.get("ACC") if self.sensors else None
            if acc_s is not None and not acc_s.empty:
                ts0_ms = float(acc_s["timestamp_ms"].iloc[0])
                ts1_ms = float(acc_s["timestamp_ms"].iloc[-1])
                lo_s = (ts0_ms - self._t0_ms) / 1000.0
                hi_s = (ts1_ms - self._t0_ms) / 1000.0
                valid_s = sorted(
                    ((float(a) - self._t0_ms) / 1000.0,
                     (float(b) - self._t0_ms) / 1000.0)
                    for a, b in self._acc_valid_intervals
                )
                gaps: list[tuple[float, float]] = []
                cursor = lo_s
                for vs, ve in valid_s:
                    if ve <= cursor:
                        continue
                    if vs > cursor:
                        gaps.append((cursor, min(vs, hi_s)))
                    cursor = max(cursor, ve)
                    if cursor >= hi_s:
                        break
                if cursor < hi_s:
                    gaps.append((cursor, hi_s))
                for gs, ge in gaps:
                    if ge <= gs:
                        continue
                    for ax in axes:
                        ax.axvspan(gs, ge, color="#3498db", alpha=0.28,
                                   zorder=3, lw=0)
                    # Single ✕ marker on the topmost panel.
                    ax_top = axes[0]
                    y_lo, y_hi = ax_top.get_ylim()
                    ax_top.text(
                        (gs + ge) / 2.0, y_lo + 0.92 * (y_hi - y_lo),
                        "✕", color="#1b5b8e", ha="center", va="center",
                        fontsize=14, fontweight="bold", zorder=5,
                    )

        self._axes = list(axes)
        self.fig.tight_layout()
        self.canvas.draw()

    def _acc_offset_seconds(self) -> float:
        if self._state is None:
            return 0.0
        return (self._acc_t0_ms - self._t0_ms) / 1000.0

    def _highlight_on_plot(self, t_start: float, t_end: float):
        for h in self._hl_spans:
            try:
                h.remove()
            except Exception:
                pass
        self._hl_spans = []
        for ax in self._axes:
            h = ax.axvspan(t_start, t_end, edgecolor=HIGHLIGHT_COLOR,
                           facecolor="none", lw=2.2, zorder=12)
            self._hl_spans.append(h)
        self.canvas.draw_idle()

    # ---------- Prediction tree ----------

    def _refresh_pred_tree(self):
        for item in self.tree_pred.get_children():
            self.tree_pred.delete(item)
        for p in self.predictions:
            l1 = p.get("lobe1") or {}
            abs_A = abs(float(l1.get("a_peak") or 0.0))
            W = float(l1.get("half_width_s") or 0.0)
            f = float(l1.get("frac_flat") or 0.0)
            self.tree_pred.insert(
                "", tk.END, iid=str(p["index"]),
                values=(
                    p["index"],
                    f"{p['t_start_s']:.1f}",
                    f"{p['t_end_s']:.1f}",
                    f"{p['duration_s']:.1f}",
                    p["ride_type"],
                    f"{p['joint_r2_mean']:.3f}",
                    f"{abs_A:.2f}",
                    f"{W:.2f}",
                    f"{f:.2f}",
                ),
                tags=(p["ride_type"],),
            )

    def _on_pred_select(self, _event=None):
        sel = self.tree_pred.selection()
        if not sel:
            return
        idx = int(sel[0])
        match = next((p for p in self.predictions if int(p["index"]) == idx), None)
        if match is None:
            return
        offset = self._acc_offset_seconds()
        self._highlight_on_plot(
            match["t_start_s"] + offset, match["t_end_s"] + offset,
        )
        self._last_sel = ("pred", (idx,))
        self._render_detail_for_prediction(match)

    # ---------- GT tree ----------

    def _gt_rows(self) -> list[tuple[int, float, float, str]]:
        """Return ``(gt_index, t_start_acc_s, t_end_acc_s, ride_type)`` for
        every ``up``/``down`` row. Times are on the ACC-local axis (same as
        the detector state)."""
        if self.gt is None or self.gt.empty:
            return []
        rows: list[tuple[int, float, float, str]] = []
        for i, row in self.gt.iterrows():
            rt = str(row.get("type", ""))
            if rt not in ("up", "down"):
                continue
            t_s = (float(row["start_ms"]) - self._acc_t0_ms) / 1000.0
            t_e = (float(row["end_ms"]) - self._acc_t0_ms) / 1000.0
            if t_e > t_s:
                rows.append((int(i), t_s, t_e, rt))
        return rows

    def _gt_matched_by_prediction(self, t_lo: float, t_hi: float) -> bool:
        for p in self.predictions:
            if p["t_start_s"] <= t_hi and p["t_end_s"] >= t_lo:
                return True
        return False

    def _refresh_gt_tree(self):
        for item in self.tree_gt.get_children():
            self.tree_gt.delete(item)
        for gi, t_s, t_e, rt in self._gt_rows():
            matched = self._gt_matched_by_prediction(t_s, t_e)
            status = "matched" if matched else "unmatched"
            self.tree_gt.insert(
                "", tk.END, iid=f"gt:{gi}",
                values=(
                    gi,
                    f"{t_s:.1f}",
                    f"{t_e:.1f}",
                    f"{t_e - t_s:.1f}",
                    rt,
                    status,
                ),
                tags=(rt,),
            )

    def _on_gt_select(self, _event=None):
        sel = self.tree_gt.selection()
        if not sel or self._state is None:
            return
        if not sel[0].startswith("gt:"):
            return
        gi = int(sel[0].split(":", 1)[1])
        match = next(
            (row for row in self._gt_rows() if row[0] == gi), None,
        )
        if match is None:
            return
        _, t_s_acc, t_e_acc, rt = match
        offset = self._acc_offset_seconds()
        self._highlight_on_plot(t_s_acc + offset, t_e_acc + offset)
        self._last_sel = ("gt", (t_s_acc, t_e_acc, rt, gi))
        self._render_detail_for_gt(t_s_acc, t_e_acc, rt, gi)

    # ---------- "Predict all algorithms" button ----------

    def _interval_for_sel(self) -> tuple[float, float, str] | None:
        """Resolve ``self._last_sel`` to ``(t_start_acc, t_end_acc, label)``.

        Both time values are on the ACC-local seconds axis (same frame
        the detector state uses). Returns ``None`` if no usable selection
        is active.
        """
        if self._last_sel is None:
            return None
        kind, payload = self._last_sel
        if kind == "pred":
            (idx,) = payload
            match = next(
                (p for p in self.predictions if int(p["index"]) == idx), None,
            )
            if match is None:
                return None
            return (
                float(match["t_start_s"]),
                float(match["t_end_s"]),
                f"pred #{idx:02d} ({match['ride_type']})",
            )
        if kind == "gt":
            t_s, t_e, rt, gi = payload
            return float(t_s), float(t_e), f"gt #{gi:02d} ({rt})"
        return None

    def _slice_sensor_ms(self, df, t_lo_ms: float, t_hi_ms: float):
        """Return the rows of ``df`` whose ``timestamp_ms`` falls in the
        absolute-ms window ``[t_lo_ms, t_hi_ms]``.

        ``df`` may be ``None`` or empty — callers should treat the return
        value as possibly-empty.
        """
        if df is None or df.empty:
            return df
        ts = df["timestamp_ms"].to_numpy(dtype=float)
        mask = (ts >= t_lo_ms) & (ts <= t_hi_ms)
        return df.loc[mask].reset_index(drop=True)

    def _run_all_predictors(
        self, t_start_acc: float, t_end_acc: float,
    ) -> dict:
        """Run every :class:`PredictAlgorithm` on the selected interval.

        Also derives a GT Δh by converting the ride's PRS pressure
        samples to altitude and taking the endpoint difference. Keys are
        each algorithm's ``.value`` plus ``"gt_dh"`` (``None`` if the
        experiment has no PRS data).
        """
        if self.sensors is None:
            return {}
        t_lo_ms = self._acc_t0_ms + t_start_acc * 1000.0
        t_hi_ms = self._acc_t0_ms + t_end_acc * 1000.0
        pad_ms = PREDICT_PRE_POST_SEC * 1000.0

        acc = self.sensors.get("ACC")
        prs = self.sensors.get("PRS")
        ride_acc = self._slice_sensor_ms(acc, t_lo_ms, t_hi_ms)
        pre_acc = self._slice_sensor_ms(acc, t_lo_ms - pad_ms, t_lo_ms)
        post_acc = self._slice_sensor_ms(acc, t_hi_ms, t_hi_ms + pad_ms)
        ride_prs = self._slice_sensor_ms(prs, t_lo_ms, t_hi_ms)

        results: dict = {"gt_dh": None}
        if ride_prs is not None and not ride_prs.empty and len(ride_prs) >= 2:
            alt = pressure_to_altitude(ride_prs["pressure"].to_numpy(dtype=float))
            results["gt_dh"] = float(alt[-1] - alt[0])

        for algo in PredictAlgorithm:
            try:
                predictor = Predictor(PREDICT_ALGORITHM_CONFIG(algorithm=algo))
                if algo is PredictAlgorithm.BAROMETER_HEIGHT_DIFF:
                    if ride_prs is None or ride_prs.empty:
                        results[algo.value] = {"error": "no PRS data"}
                        continue
                    out = predictor.predict(ride_prs)
                else:
                    if ride_acc is None or ride_acc.empty:
                        results[algo.value] = {"error": "no ACC data"}
                        continue
                    out = predictor.predict(
                        ride_acc, phone_model="",
                        pre=pre_acc, post=post_acc,
                    )
                results[algo.value] = {
                    "dh":       float(out.height_diff),
                    "ci":       float(out.ci_half_width),
                    "sigma":    float(out.theoretical_sigma),
                    "accepted": bool(out.accepted),
                    "reason":   str(out.reject_reason),
                    "q":        float(out.quality_score),
                    "meta":     dict(out.meta) if out.meta else {},
                }
            except Exception as exc:
                results[algo.value] = {
                    "error": f"{type(exc).__name__}: {exc}",
                }
        return results

    def _format_predictions(self, results: dict, label: str) -> str:
        lines = [f"Predict all algorithms — {label}:"]
        for algo in PredictAlgorithm:
            row = results.get(algo.value, {})
            name = f"{algo.value}"
            if "error" in row:
                lines.append(f"  {name:26s} error: {row['error']}")
                continue
            ci_str = (
                f"±{row['ci']:.2f}m" if math.isfinite(row["ci"]) else "±inf"
            )
            verdict = (
                "OK" if row["accepted"]
                else f"REJECT ({row['reason'] or 'no reason'})"
            )
            lines.append(
                f"  {name:26s} Δh = {row['dh']:+7.2f} m   CI {ci_str}   "
                f"q={row['q']:.2f}   [{verdict}]"
            )
        gt = results.get("gt_dh")
        if gt is None:
            lines.append(f"  {'GT (barometer altitude)':26s} — no PRS data")
        else:
            lines.append(
                f"  {'GT (barometer altitude)':26s} Δh = {gt:+7.2f} m"
            )
        return "\n".join(lines)

    def _annotate_detail_with_predictions(self, summary: str) -> None:
        """Overlay the prediction summary as a small text box on the detail
        signal axis, so the comparison is visible in the overview panel.

        Silently skips if there is no signal subplot (i.e., the placeholder
        is showing) or the axis layout changed.
        """
        if not self.detail_fig.axes:
            return
        axes = [
            ax for ax in self.detail_fig.axes
            if ax.get_xlabel() == "t (s, ACC-local)"
            and ax.get_ylabel() == "a (m/s²)"
        ]
        if not axes:
            return
        ax = axes[0]
        ax.text(
            0.01, 0.98, summary, transform=ax.transAxes,
            ha="left", va="top", fontsize=7.5, family="monospace",
            bbox=dict(facecolor="#ffffff", alpha=0.85,
                      edgecolor="#888", boxstyle="round,pad=0.3"),
            zorder=20,
        )
        self.detail_canvas.draw_idle()

    def _on_predict_clicked(self) -> None:
        """Handler for the "▶ Predict all algorithms" button.

        Runs every :class:`PredictAlgorithm` on the currently-selected
        prediction or GT interval, appends the results + GT Δh to the
        verdict text, overlays a summary on the segmentation signal panel,
        renders the per-algorithm Prediction tab, and auto-switches to it.
        """
        if self.sensors is None:
            self.status_var.set("Load an experiment first.")
            return
        interval = self._interval_for_sel()
        if interval is None:
            self.status_var.set(
                "Select a prediction or a GT row before predicting."
            )
            return
        t_start, t_end, label = interval
        self.status_var.set(f"Running all predictors on {label}…")
        self.update_idletasks()

        results = self._run_all_predictors(t_start, t_end)
        summary = self._format_predictions(results, label)

        current = self.verdict_text.get("1.0", "end-1c").rstrip()
        combined = (current + "\n\n" + summary) if current else summary
        self._set_verdict(combined)
        self._annotate_detail_with_predictions(summary)
        self._render_prediction_tab(results, label, t_start)
        try:
            self.detail_nb.select(1)  # switch to Prediction tab
        except tk.TclError:
            pass
        self.status_var.set(f"Predicted Δh for {label}.")

    # ---------- Prediction tab rendering ----------

    def _pred_placeholder(
        self,
        msg: str = "Click ▶ Predict all algorithms to populate.",
    ) -> None:
        if hasattr(self, "pred_table"):
            for iid in self.pred_table.get_children():
                self.pred_table.delete(iid)
        self.pred_fig.clear()
        ax = self.pred_fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#666",
                style="italic")
        ax.set_axis_off()
        self.pred_canvas.draw_idle()

    def _render_prediction_tab(
        self, results: dict, label: str, t_start_acc: float,
    ) -> None:
        """Populate the Prediction tab table + 2-panel figure.

        All scalar metrics (Δh, CI, σ, q, verdict, reason) live in the
        table. The figure is reserved for the two visualisations that
        actually need a plot: the trapezoid template overlay and the
        ZUPT integrated position trace.

        ``t_start_acc`` is the ride start on the editor's ACC-local
        seconds axis — passed through so the trapezoid panel plots on
        the same axis as the top signal panel rather than a ride-local
        offset starting at 0.
        """
        self._populate_pred_table(results, label)

        self.pred_fig.clear()
        gs = self.pred_fig.add_gridspec(
            2, 1,
            height_ratios=[2.0, 1.3],
            hspace=0.55,
            left=0.12, right=0.97, top=0.94, bottom=0.09,
        )
        ax_trap = self.pred_fig.add_subplot(gs[0, 0])
        ax_zupt = self.pred_fig.add_subplot(gs[1, 0])

        self._render_pred_trapezoid(ax_trap, results, t_start_acc)
        self._render_pred_zupt(ax_zupt, results)

        self.pred_canvas.draw_idle()

    def _populate_pred_table(self, results: dict, label: str) -> None:
        for iid in self.pred_table.get_children():
            self.pred_table.delete(iid)

        def _fmt_float(v: float, fmt: str) -> str:
            if not math.isfinite(v):
                return "—"
            return format(v, fmt)

        for algo in PredictAlgorithm:
            row = results.get(algo.value, {})
            if "error" in row:
                self.pred_table.insert(
                    "", "end", iid=algo.value,
                    values=(algo.value, "—", "—", "—", "—",
                            "ERROR", str(row["error"])[:240]),
                    tags=("error",),
                )
                continue
            ci_str = (
                f"±{row['ci']:.2f}" if math.isfinite(row["ci"]) else "±inf"
            )
            verdict = "OK" if row["accepted"] else "REJECT"
            tag = "ok" if row["accepted"] else "reject"
            self.pred_table.insert(
                "", "end", iid=algo.value,
                values=(
                    algo.value,
                    _fmt_float(row["dh"], "+.2f"),
                    ci_str,
                    _fmt_float(row["sigma"], ".2f"),
                    _fmt_float(row["q"], ".2f"),
                    verdict,
                    (row["reason"] or "")[:240],
                ),
                tags=(tag,),
            )

        gt_dh = results.get("gt_dh")
        gt_str = "—" if gt_dh is None else f"{gt_dh:+.2f}"
        gt_reason = (
            f"PRS-derived Δh for {label}" if gt_dh is not None
            else "no PRS data"
        )
        self.pred_table.insert(
            "", "end", iid="gt_dh",
            values=("GT (PRS Δh)", gt_str, "—", "—", "—", "—", gt_reason),
            tags=("gt",),
        )

    # Algorithm draw order kept consistent across all panels.
    _PRED_ALGO_ORDER = (
        PredictAlgorithm.BAROMETER_HEIGHT_DIFF,
        PredictAlgorithm.ZUPT_ACCEL,
        PredictAlgorithm.TRAPEZOID_ACCEL,
    )
    _PRED_ALGO_COLORS = {
        PredictAlgorithm.BAROMETER_HEIGHT_DIFF: "#2980b9",
        PredictAlgorithm.ZUPT_ACCEL:            "#27ae60",
        PredictAlgorithm.TRAPEZOID_ACCEL:       "#c0392b",
    }

    def _render_pred_trapezoid(
        self, ax, results: dict, t_start_acc: float = 0.0,
    ) -> None:
        """Plot the fitted trapezoid template overlaid on the smoothed
        accel signal, with the fitted parameters annotated.

        The x-axis is ACC-local seconds (matching the top signal panel)
        so this diagnostic and the segmentation panel show the same
        time positions for the same data.
        """
        ax.set_title("Trapezoid fit on accelerometer signal",
                     fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("t (s, ACC-local)", fontsize=8)
        ax.set_ylabel("a (m/s²)", fontsize=8)

        row = results.get(PredictAlgorithm.TRAPEZOID_ACCEL.value, {})
        if "error" in row:
            ax.text(0.5, 0.5, f"trapezoid_accel error: {row['error']}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#7f0000", style="italic")
            return
        meta = row.get("meta") or {}
        t_sec = meta.get("t_sec")
        a_smooth = meta.get("a_smooth")
        a_template = meta.get("a_template")
        params = meta.get("params") or {}

        if t_sec is None or a_smooth is None or a_template is None:
            ax.text(0.5, 0.5,
                    "no trapezoid fit (algorithm did not return template)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")
            return

        # t_sec is ride-local; shift onto the editor's ACC-local axis.
        t_arr = np.asarray(t_sec, dtype=float) + float(t_start_acc)
        a_smooth_arr = np.asarray(a_smooth, dtype=float)
        a_tpl_arr = np.asarray(a_template, dtype=float)
        ax.plot(t_arr, a_smooth_arr, color="#2c3e50", lw=0.9,
                label="a_smooth")
        ax.plot(t_arr, a_tpl_arr, color="#c0392b", lw=1.6, alpha=0.9,
                label="trapezoid template")
        ax.axhline(0.0, color="gray", lw=0.4, ls="--", alpha=0.5)

        t_c1 = params.get("t_c1")
        t_c2 = params.get("t_c2")
        if t_c1 is not None:
            ax.axvline(float(t_c1) + float(t_start_acc),
                       color="#c0392b", lw=0.6, ls=":", alpha=0.7)
        if t_c2 is not None:
            ax.axvline(float(t_c2) + float(t_start_acc),
                       color="#c0392b", lw=0.6, ls=":", alpha=0.7)

        if params:
            sign = int(params.get("sign", 0))
            txt = (
                f"A_used={params.get('A_used', float('nan')):.2f} m/s²  "
                f"W={params.get('W', float('nan')):.2f}s  "
                f"f={params.get('f', float('nan')):.2f}  "
                f"sign={sign:+d}\n"
                f"t_c1={params.get('t_c1', float('nan')):.2f}s  "
                f"t_c2={params.get('t_c2', float('nan')):.2f}s  "
                f"joint_R²={params.get('joint_r2', float('nan')):.3f}  "
                f"v_peak={params.get('v_peak_measured', float('nan')):+.2f}m/s"
            )
            ax.text(
                0.01, 0.98, txt, transform=ax.transAxes,
                ha="left", va="top", fontsize=7, family="monospace",
                bbox=dict(facecolor="#ffffff", alpha=0.85,
                          edgecolor="#888", boxstyle="round,pad=0.3"),
                zorder=20,
            )
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    def _render_pred_zupt(self, ax, results: dict) -> None:
        """ZUPT integrated position curve with motion-window shading."""
        ax.set_title("ZUPT integrated position", fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("sample index", fontsize=8)
        ax.set_ylabel("pos (m)", fontsize=8)

        row = results.get(PredictAlgorithm.ZUPT_ACCEL.value, {})
        if "error" in row:
            ax.text(0.5, 0.5, f"zupt_accel error: {row['error']}",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#7f0000", style="italic")
            return
        meta = row.get("meta") or {}
        pos = meta.get("pos_curve")
        if pos is None:
            ax.text(0.5, 0.5,
                    "no ZUPT trajectory (algorithm did not return pos_curve)",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")
            return
        pos_arr = np.asarray(pos, dtype=float)
        idx = np.arange(pos_arr.size)
        ax.plot(idx, pos_arr, color="#27ae60", lw=1.1, label="pos(t)")

        start = meta.get("start_idx")
        end = meta.get("end_idx")
        if start is not None and end is not None and end > start:
            ax.axvspan(int(start), int(end), color="#27ae60", alpha=0.15,
                       label="motion window")
        ax.axhline(0.0, color="gray", lw=0.4, ls="--", alpha=0.5)

        n_active = meta.get("n_active")
        method = meta.get("method", "")
        active_frac = meta.get("active_fraction")
        info_bits = []
        if math.isfinite(row.get("dh", float("nan"))):
            info_bits.append(f"final Δh={row['dh']:+.2f} m")
        if n_active is not None:
            info_bits.append(f"n_active={int(n_active)}")
        if active_frac is not None:
            info_bits.append(f"active_frac={float(active_frac):.2f}")
        if method:
            info_bits.append(f"method={method}")
        if info_bits:
            ax.text(
                0.01, 0.98, "  ".join(info_bits), transform=ax.transAxes,
                ha="left", va="top", fontsize=7, family="monospace",
                bbox=dict(facecolor="#ffffff", alpha=0.85,
                          edgecolor="#888", boxstyle="round,pad=0.3"),
                zorder=20,
            )
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    # ---------- Detail window pad ----------

    def _current_pad_s(self) -> float:
        try:
            v = float(self.detail_pad_var.get())
        except (tk.TclError, ValueError):
            v = 5.0
        return max(0.5, v)

    def _on_pad_changed(self):
        """User typed a new pad — re-render whichever selection is active."""
        if self._last_sel is None:
            return
        kind, payload = self._last_sel
        if kind == "pred":
            (idx,) = payload
            match = next(
                (p for p in self.predictions if int(p["index"]) == idx), None,
            )
            if match is not None:
                self._render_detail_for_prediction(match)
        elif kind == "gt":
            t_s_acc, t_e_acc, rt, gi = payload
            self._render_detail_for_gt(t_s_acc, t_e_acc, rt, gi)

    def _scale_pad(self, factor: float):
        self.detail_pad_var.set(round(self._current_pad_s() * factor, 2))
        self._on_pad_changed()

    # ---------- Detail panel ----------

    def _detail_placeholder(self, msg: str = "Select a prediction or a GT ride."):
        self.detail_fig.clear()
        ax = self.detail_fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes,
                ha="center", va="center", fontsize=11, color="#666", style="italic")
        ax.set_axis_off()
        self.detail_canvas.draw_idle()
        self._set_verdict("")

    def _set_verdict(self, text: str):
        self.verdict_text.configure(state="normal")
        self.verdict_text.delete("1.0", "end")
        if text:
            self.verdict_text.insert("1.0", text)
        self.verdict_text.configure(state="disabled")

    # ---------- Signed-R² over time panel ----------

    # Status → color mapping for signed-R² peak markers. Keys mirror the
    # ``PEAK_STATUS_*`` tags :func:`detect.classify_peak` returns.
    _STATUS_COLORS: dict[str, str] = {
        "accepted":          "#27ae60",  # green
        "unpaired (greedy)": "#f39c12",  # orange
        "same-sign NMS":     "#9b59b6",  # purple
        "NMS (local)":       "#8e44ad",  # darker purple
        "lost to opp sign":  "#34495e",  # slate
        "R²<thr":            "#7f8c8d",  # grey
        "|A|<thr":           "#95a5a6",  # light grey
    }

    def _render_signed_r2_panel(self, ax, t_lo: float, t_hi: float) -> None:
        """Bottom-row panel: per-sign best R² over time + color-coded peaks.

        Each local max is drawn as a scatter dot colored by its
        classification status (accepted / NMS / below-thr / …). The axis
        legend maps both the traces and every status color that appears
        in the current window.
        """
        from matplotlib.lines import Line2D  # noqa: WPS433 — local import OK

        state = self._state
        if state is None:
            ax.set_axis_off()
            return
        t = state["t"]
        pos_r2 = state["best_pos_r2"]
        neg_r2 = state["best_neg_r2"]
        mask = (t >= t_lo) & (t <= t_hi)
        pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
        neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)
        line_pos, = ax.plot(
            t[mask], pos_plot[mask], color="#2980b9", lw=0.9, label="max R² (+)",
        )
        line_neg, = ax.plot(
            t[mask], neg_plot[mask], color="#c0392b", lw=0.9, label="max R² (−)",
        )
        ax.axhline(state["config"].r2_peak_thresh, color="gray",
                   lw=0.5, ls="--", alpha=0.6)
        ax.set_ylim(0, 1.05)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylabel("R² (per sign)")
        ax.set_xlabel("t (s, ACC-local)")
        ax.grid(True, alpha=0.25)

        peaks_pos = _detect.find_local_maxima(pos_r2, t, t_lo, t_hi)
        peaks_neg = _detect.find_local_maxima(neg_r2, t, t_lo, t_hi)
        statuses_seen: set[str] = set()
        for sign, peaks, arr in (
            (+1, peaks_pos, pos_r2),
            (-1, peaks_neg, neg_r2),
        ):
            for i in peaks:
                tag = _detect.classify_peak(state, i, sign, self.predictions)
                color = self._STATUS_COLORS.get(tag, "#000000")
                ax.scatter([t[i]], [arr[i]], color=color, s=36, zorder=5,
                           edgecolor="black", linewidth=0.4)
                statuses_seen.add(tag)

        handles = [line_pos, line_neg]
        labels = ["max R² (+)", "max R² (−)"]
        for tag in (
            "accepted", "unpaired (greedy)", "same-sign NMS",
            "NMS (local)", "lost to opp sign", "R²<thr", "|A|<thr",
        ):
            if tag not in statuses_seen:
                continue
            handles.append(Line2D(
                [0], [0], marker="o", linestyle="",
                markerfacecolor=self._STATUS_COLORS[tag],
                markeredgecolor="black", markeredgewidth=0.4, markersize=6,
            ))
            labels.append(tag)
        ax.legend(handles, labels, fontsize=6, loc="lower right", ncol=2,
                  framealpha=0.85)

    def _draw_heatmap(self, ax, heat: np.ndarray, title: str,
                      mark_W: float | None = None, mark_f: float | None = None):
        grid_w_s = self._state["grid_w_s"] if self._state else _detect.DEFAULT_CONFIG.grid_w_s()
        grid_f   = self._state["grid_f"]   if self._state else _detect.DEFAULT_CONFIG.grid_f()
        im = ax.imshow(
            heat, origin="lower", aspect="auto",
            extent=(grid_f[0], grid_f[-1], grid_w_s[0], grid_w_s[-1]),
            cmap="viridis", vmin=0.0, vmax=1.0,
        )
        if mark_W is not None and mark_f is not None:
            ax.plot([mark_f], [mark_W], marker="x", color="#e74c3c",
                    markersize=9, markeredgewidth=2.0)
        ax.set_xlabel("plateau f")
        ax.set_ylabel("half-width W (s)")
        ax.set_title(title, fontsize=9)
        self.detail_fig.colorbar(im, ax=ax, fraction=0.05, pad=0.04)

    def _render_detail_for_prediction(self, pred: dict):
        if self._state is None:
            self._detail_placeholder("No detector state.")
            return
        t = self._state["t"]
        a_smooth = self._state["a_smooth"]
        a_vert = self._state["a_vert"]

        t_c1 = float(pred["lobe1"]["t_c"])
        t_c2 = float(pred["lobe2"]["t_c"])
        i1 = int(np.argmin(np.abs(t - t_c1)))
        i2 = int(np.argmin(np.abs(t - t_c2)))

        grid_w_s = self._state["grid_w_s"]
        grid_f = self._state["grid_f"]
        heat1 = _detect.heatmap_at(a_smooth, t, i1, grid_w_s, grid_f)
        heat2 = _detect.heatmap_at(a_smooth, t, i2, grid_w_s, grid_f)

        self.detail_fig.clear()
        gs = self.detail_fig.add_gridspec(
            3, 2, height_ratios=[1.0, 0.9, 0.7], hspace=0.65, wspace=0.28,
        )
        ax_h1 = self.detail_fig.add_subplot(gs[0, 0])
        ax_h2 = self.detail_fig.add_subplot(gs[0, 1])
        ax_sig = self.detail_fig.add_subplot(gs[1, :])
        ax_rt = self.detail_fig.add_subplot(gs[2, :])

        W_star = float(pred["lobe1"]["half_width_s"])
        f_star = float(pred["lobe1"]["frac_flat"])
        self._draw_heatmap(ax_h1, heat1, f"lobe1 @ t={t_c1:.1f}s",
                           mark_W=W_star, mark_f=f_star)
        self._draw_heatmap(ax_h2, heat2, f"lobe2 @ t={t_c2:.1f}s",
                           mark_W=W_star, mark_f=f_star)

        pad = self._current_pad_s()
        mask = (t >= pred["t_start_s"] - pad) & (t <= pred["t_end_s"] + pad)
        ax_sig.plot(t[mask], a_vert[mask], color="#2c3e50", lw=0.7, label="a_vert")
        ax_sig.plot(t[mask], a_smooth[mask], color="#e67e22", lw=1.1, label="a_smooth")
        ax_sig.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
        for lobe_key in ("lobe1", "lobe2"):
            L = pred[lobe_key]
            W = float(L["half_width_s"])
            f = float(L["frac_flat"])
            A = float(L["a_peak"])
            t_c = float(L["t_c"])
            tt = np.linspace(t_c - W, t_c + W, 200)
            yy = A * trapezoid_kernel(tt, t_c, W, f)
            ax_sig.plot(tt, yy, color="#c0392b", lw=1.6)
            ax_sig.scatter([t_c], [A], color="#c0392b", s=24, zorder=5)
        ax_sig.axvline(pred["t_start_s"], color=PRED_COLORS[pred["ride_type"]],
                       lw=1.0, ls="--", alpha=0.8)
        ax_sig.axvline(pred["t_end_s"],   color=PRED_COLORS[pred["ride_type"]],
                       lw=1.0, ls="--", alpha=0.8)
        ax_sig.set_xlabel("t (s, ACC-local)")
        ax_sig.set_ylabel("a (m/s²)")
        ax_sig.grid(True, alpha=0.25)
        ax_sig.legend(fontsize=8, loc="upper right")
        ax_sig.set_title(
            f"pred #{pred['index']:02d} {pred['ride_type']} — "
            f"joint R²={pred['joint_r2_mean']:.3f}  |A|={abs(pred['lobe1']['a_peak']):.2f}  "
            f"W={W_star:.2f}s  f={f_star:.2f}",
            fontsize=9,
        )

        t_lo_win = float(pred["t_start_s"] - pad)
        t_hi_win = float(pred["t_end_s"] + pad)
        self._render_signed_r2_panel(ax_rt, t_lo_win, t_hi_win)
        self.detail_canvas.draw_idle()
        self._set_verdict(
            f"prediction #{pred['index']:02d} — accepted pair.\n"
            f"  lobe1 t={t_c1:.1f}s  A={pred['lobe1']['a_peak']:+.2f}  "
            f"R²={pred['lobe1']['r2_local']:.3f}\n"
            f"  lobe2 t={t_c2:.1f}s  A={pred['lobe2']['a_peak']:+.2f}  "
            f"R²={pred['lobe2']['r2_local']:.3f}\n"
            f"  shared W={W_star:.2f}s  f={f_star:.2f}  "
            f"|A|={abs(pred['lobe1']['a_peak']):.2f}\n"
            f"  joint mean R²={pred['joint_r2_mean']:.3f}  "
            f"heatmap_energy={pred.get('heatmap_energy', float('nan')):.3f}"
        )

    def _render_detail_for_gt(self, t_lo: float, t_hi: float,
                              ride_type: str, gt_index: int):
        if self._state is None:
            self._detail_placeholder("No detector state.")
            return
        t = self._state["t"]
        a_smooth = self._state["a_smooth"]
        a_vert = self._state["a_vert"]

        diag = diagnose_window(self._state, t_lo, t_hi, ride_type=ride_type)

        # Heatmaps at the best + and − samples inside the GT window (if any).
        pos = diag.get("pos_peak")
        neg = diag.get("neg_peak")
        self.detail_fig.clear()
        gs = self.detail_fig.add_gridspec(
            3, 2, height_ratios=[1.0, 0.9, 0.7], hspace=0.65, wspace=0.28,
        )
        ax_h1 = self.detail_fig.add_subplot(gs[0, 0])
        ax_h2 = self.detail_fig.add_subplot(gs[0, 1])
        ax_sig = self.detail_fig.add_subplot(gs[1, :])
        ax_rt = self.detail_fig.add_subplot(gs[2, :])

        def _plot_for(ax, peak, label):
            if peak is None:
                ax.text(0.5, 0.5, f"no {label} sample in window",
                        transform=ax.transAxes, ha="center", va="center",
                        fontsize=9, color="#888", style="italic")
                ax.set_xticks([]); ax.set_yticks([])
                ax.set_title(f"{label} lobe — n/a", fontsize=9)
                return
            i, A, _r2 = peak
            heat = _detect.heatmap_at(
                a_smooth, t, i,
                self._state["grid_w_s"], self._state["grid_f"],
            )
            self._draw_heatmap(ax, heat, f"{label} @ t={t[i]:.1f}s  A={A:+.2f}")

        _plot_for(ax_h1, pos, "+")
        _plot_for(ax_h2, neg, "−")

        pad = self._current_pad_s()
        mask = (t >= t_lo - pad) & (t <= t_hi + pad)
        ax_sig.plot(t[mask], a_vert[mask], color="#2c3e50", lw=0.7, label="a_vert")
        ax_sig.plot(t[mask], a_smooth[mask], color="#e67e22", lw=1.1, label="a_smooth")
        ax_sig.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
        gt_color = TYPE_COLORS[ride_type]
        ax_sig.axvspan(t_lo, t_hi, color=gt_color, alpha=0.22, zorder=0)
        for peak, color, tag in ((pos, "#2980b9", "+"), (neg, "#c0392b", "−")):
            if peak is None:
                continue
            i, A, _ = peak
            ax_sig.axvline(t[i], color=color, lw=0.9, ls=":", alpha=0.85)
            ax_sig.scatter([t[i]], [A], color=color, s=26, zorder=5)
        ax_sig.set_xlabel("t (s, ACC-local)")
        ax_sig.set_ylabel("a (m/s²)")
        ax_sig.grid(True, alpha=0.25)
        ax_sig.legend(fontsize=8, loc="upper right")

        matched = self._gt_matched_by_prediction(t_lo, t_hi)
        ax_sig.set_title(
            f"GT #{gt_index:02d} {ride_type}  t=[{t_lo:.1f}, {t_hi:.1f}]s  "
            f"({'matched by detector' if matched else 'NOT matched'})",
            fontsize=9,
        )

        self._render_signed_r2_panel(ax_rt, t_lo - pad, t_hi + pad)
        self.detail_canvas.draw_idle()

        verdict = "\n".join(diag["verdict_lines"]) if diag.get("verdict_lines") else ""
        header = (
            f"GT #{gt_index:02d} {ride_type}  window=[{t_lo:.1f}, {t_hi:.1f}]s  "
            f"(duration={t_hi - t_lo:.1f}s)\n"
        )
        header += "matched by detector.\n" if matched else "NOT matched — diagnosis:\n"
        self._set_verdict(header + verdict)

    # ---------- Canvas click / hover ----------

    def _on_canvas_click(self, event):
        """Select the prediction or GT interval under the cursor.

        Predictions are hatched on top of the GT bands, so they win if
        both overlap the click. Delegates to the tree selection so
        ``_on_pred_select`` / ``_on_gt_select`` handle the visuals.
        """
        if event.inaxes not in self._axes or event.xdata is None or event.button != 1:
            return
        # Respect matplotlib's zoom/pan modes.
        tb = getattr(self.canvas.manager, "toolbar", None)
        if tb and getattr(tb, "mode", ""):
            return
        x = float(event.xdata)
        offset = self._acc_offset_seconds()

        # 1) Predictions (on top).
        for p in self.predictions:
            s = p["t_start_s"] + offset
            e = p["t_end_s"] + offset
            if s <= x <= e:
                iid = str(p["index"])
                if iid in self.tree_pred.get_children():
                    self.tree_pred.selection_set(iid)
                    self.tree_pred.see(iid)
                return

        # 2) GT (behind).
        for gi, t_s, t_e, _rt in self._gt_rows():
            if (t_s + offset) <= x <= (t_e + offset):
                iid = f"gt:{gi}"
                if iid in self.tree_gt.get_children():
                    self.tree_gt.selection_set(iid)
                    self.tree_gt.see(iid)
                return

    def _on_canvas_motion(self, event):
        if event.inaxes in self._axes and event.xdata is not None and event.ydata is not None:
            ylab = event.inaxes.get_ylabel() or "y"
            self.hover_var.set(
                f"t = {event.xdata:8.2f} s    {ylab} = {event.ydata:.3f}"
            )
        else:
            self.hover_var.set("")

    # ---------- Zoom / pan helpers (mirrored from gt_editor.py) ----------

    def _zoom_x_at(self, ax, center_x: float, zoom_in: bool) -> None:
        xlim = ax.get_xlim()
        width = xlim[1] - xlim[0]
        if width <= 0:
            return
        factor = 1 / 1.5 if zoom_in else 1.5
        new_width = max(width * factor, 0.01)
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

    def _zoom_y_at(self, ax, center_y: float, zoom_in: bool) -> None:
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

    def _zoom_y_around_center(self, zoom_in: bool) -> None:
        for ax in self._axes:
            ylim = ax.get_ylim()
            self._zoom_y_at(ax, (ylim[0] + ylim[1]) / 2.0, zoom_in=zoom_in)

    def _pan_x(self, frac: float) -> None:
        if not self._axes:
            return
        ax = self._axes[0]
        xlim = ax.get_xlim()
        shift = (xlim[1] - xlim[0]) * frac
        ax.set_xlim(xlim[0] + shift, xlim[1] + shift)
        self.canvas.draw_idle()

    def _pan_y(self, frac: float) -> None:
        for ax in self._axes:
            ylim = ax.get_ylim()
            shift = (ylim[1] - ylim[0]) * frac
            ax.set_ylim(ylim[0] + shift, ylim[1] + shift)
        self.canvas.draw_idle()

    def _fit_x(self) -> None:
        if self.sensors is None or not self._axes:
            return
        prs = self.sensors.get("PRS")
        acc = self.sensors.get("ACC")
        if prs is not None and not prs.empty:
            t0 = int(prs["timestamp_ms"].iloc[0])
            t1 = int(prs["timestamp_ms"].iloc[-1])
        elif acc is not None and not acc.empty:
            t0 = int(acc["timestamp_ms"].iloc[0])
            t1 = int(acc["timestamp_ms"].iloc[-1])
        else:
            return
        self._axes[0].set_xlim(
            (t0 - self._t0_ms) / 1000.0, (t1 - self._t0_ms) / 1000.0,
        )
        self.canvas.draw_idle()

    def _on_scroll(self, event):
        if event.inaxes not in self._axes:
            return
        zoom_in = (event.button == "up")
        if event.key and "shift" in event.key:
            if event.ydata is None:
                return
            self._zoom_y_at(event.inaxes, event.ydata, zoom_in=zoom_in)
            return
        if event.xdata is None:
            return
        self._zoom_x_at(event.inaxes, event.xdata, zoom_in=zoom_in)

    def _on_tk_wheel(self, event, delta: int | None = None) -> None:
        if not self._axes:
            return
        if delta is None:
            delta = getattr(event, "delta", 0)
        if delta == 0:
            return
        widget = self.canvas.get_tk_widget()
        h = widget.winfo_height()
        mpl_y = h - event.y
        ax = None
        for a in self._axes:
            if a.bbox.contains(event.x, mpl_y):
                ax = a
                break
        if ax is None:
            ax = self._axes[0]
        data_x, data_y = ax.transData.inverted().transform((event.x, mpl_y))
        shift_held = bool(getattr(event, "state", 0) & 0x0001)
        if shift_held:
            self._zoom_y_at(ax, data_y, zoom_in=(delta > 0))
        else:
            self._zoom_x_at(ax, data_x, zoom_in=(delta > 0))


def main() -> int:
    preselect = sys.argv[1] if len(sys.argv) > 1 else None
    app = PredictionEditor(preselect=preselect)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
