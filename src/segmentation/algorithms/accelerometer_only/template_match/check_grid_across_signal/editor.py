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
    venv/bin/python src/segmentation/algorithms/accelerometer_only/\
template_match/check_grid_across_signal/editor.py [exp_folder_name]
"""

from __future__ import annotations

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

# Make absolute ``src.*`` imports resolve when the editor is launched as
# a standalone script (``python .../editor.py``) as well as via ``-m``.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.gt_editor import PANEL_DRAWERS as _GT_PANEL_DRAWERS  # noqa: E402
from src.data.gt_editor import TYPE_COLORS  # noqa: E402
from src.data.loader import (  # noqa: E402
    RAW_DATA_ROOT,
    getExperimentData,
    list_experiments,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E402
    GRID_W_S, GRID_F, SMOOTH_SEC, trapezoid_kernel,
    _estimate_fs_hz, _vertical_accel, _smooth,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect,
)

predict_intervals = _detect.predict_intervals
diagnose_window = _detect.diagnose_window

PRED_COLORS = {"up": "#1f3a5f", "down": "#7d3c98"}
HIGHLIGHT_COLOR = "#e67e22"


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

        self.protocol("WM_DELETE_WINDOW", self.destroy)

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
        self.detail_fig = Figure(figsize=(6, 5))
        self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=detail_frame)
        self.detail_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.verdict_text = tk.Text(
            detail_frame, height=7, wrap="word", font=("Menlo", 9),
            background="#fafafa", foreground="#222", borderwidth=1, relief="solid",
        )
        self.verdict_text.pack(fill=tk.X, pady=(6, 0))
        self.verdict_text.configure(state="disabled")
        self._detail_placeholder()

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
        self.exp_combo["values"] = list_experiments(RAW_DATA_ROOT)

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

        self.status_var.set(f"Running detector on {name}…")
        self.update_idletasks()
        try:
            self.predictions, state = predict_intervals(acc)
            self._state = state if state else None
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
        im = ax.imshow(
            heat, origin="lower", aspect="auto",
            extent=(GRID_F[0], GRID_F[-1], GRID_W_S[0], GRID_W_S[-1]),
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

        heat1 = _detect.heatmap_at(a_smooth, t, i1)
        heat2 = _detect.heatmap_at(a_smooth, t, i2)

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
            f"  joint mean R²={pred['joint_r2_mean']:.3f}"
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
            heat = _detect.heatmap_at(a_smooth, t, i)
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
