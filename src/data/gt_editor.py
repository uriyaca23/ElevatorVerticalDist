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

import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk

import matplotlib
matplotlib.use("TkAgg")

import numpy as np
import pandas as pd
from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg,
    NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.data.loader import (
    RAW_DATA_ROOT,
    ExperimentPipeline,
    getExperimentData,
    list_experiments,
    saveExperimentData,
)
from src.data.loader.alignment import _smoothed_velocity
from src.data.loader.pipeline import _coerce_bool, addGTtoSegment, load_baramoshka
from src.physics import calculate_velocity_from_accelerometer


VALID_TYPES = ["outside", "up", "down"]
TYPE_COLORS = {"up": "#2ca02c", "down": "#d62728", "outside": "#b8b8b8"}
HIGHLIGHT_COLOR = "#1f77b4"


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
    mag = np.sqrt(df["x"] ** 2 + df["y"] ** 2 + df["z"] ** 2).to_numpy()
    ax.plot(t, mag, color="tab:blue", lw=0.5)
    ax.set_ylabel("|a| (m/s²)")


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
        ttk.Button(top, text="Save (Ctrl+S)", command=self.save_gt).pack(side=tk.RIGHT)

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
        self.exp_combo["values"] = list_experiments(RAW_DATA_ROOT)

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
        self.pipeline = ExperimentPipeline(sensors, gt, metadata)

        prs = self.pipeline.data.get("PRS")
        acc = self.pipeline.data.get("ACC")
        if prs is not None and not prs.empty:
            self._t0_ms = int(prs["timestamp_ms"].iloc[0])
        elif acc is not None and not acc.empty:
            self._t0_ms = int(acc["timestamp_ms"].iloc[0])
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
        axes[-1].set_xlabel("time (s)")
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

        self._axes = list(axes)
        self.fig.tight_layout()
        self.canvas.draw()

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
            self.hover_var.set(
                f"t = {event.xdata:8.2f} s    {ylab} = {event.ydata:.3f}"
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
