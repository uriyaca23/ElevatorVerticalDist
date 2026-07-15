"""Prediction editor — presentation-only Tk diagnostic UI.

A read-only desktop viewer for inspecting whole-signal trapezoid ride
detections and their Δh predictions. It loads an experiment's raw sensor
data (:mod:`src.data.loader`) and delegates *all* computation to the
``pyramidElevatorDist`` package:

* ride detection            -> :func:`pyramidElevatorDist.findSegments`
* per-interval trapezoid fit -> :func:`pyramidElevatorDist.findSegmentParameters`
* Δh + CI per ride           -> :func:`pyramidElevatorDist.predictSegment`
                                / :func:`pyramidElevatorDist.predictByParameters`
* display signal             -> :func:`pyramidElevatorDist.reconstructedSignal`

The window is assembled from one mixin per screen area (top bar, overview
timeline, interaction, predictions table, GT table, segmentation detail,
prediction detail); each lives in its own module and touches only Tk /
matplotlib / numpy + package calls + data loading.

Resample policy: raw experiment data is passed to the package as-is with
``resample=True`` everywhere — the package normalizes onto its canonical
50 Hz grid internally, anchored on the first raw ACC timestamp, so ride
coordinates from detection, fitting and prediction all line up.

Usage::

    python -m src.pipelines.prediction_editor [exp_folder_name]
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import ttk

import matplotlib
matplotlib.use("TkAgg")

# Make absolute ``src.*`` imports resolve when launched via ``-m`` from the
# repo root (the canonical invocation) as well as directly.
_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipelines.prediction_editor.panel_detail_prediction import (  # noqa: E402
    DetailPredictionMixin,
)
from src.pipelines.prediction_editor.panel_detail_segmentation import (  # noqa: E402
    DetailSegmentationMixin,
)
from src.pipelines.prediction_editor.panel_experiment_bar import (  # noqa: E402
    ExperimentBarMixin,
)
from src.pipelines.prediction_editor.panel_gt_table import GtTableMixin  # noqa: E402
from src.pipelines.prediction_editor.panel_overview_plot import (  # noqa: E402
    OverviewPlotMixin,
)
from src.pipelines.prediction_editor.panel_predictions_table import (  # noqa: E402
    PredictionsTableMixin,
)
from src.pipelines.prediction_editor.plot_interaction import (  # noqa: E402
    PlotInteractionMixin,
)


class PredictionEditor(
    ExperimentBarMixin,
    OverviewPlotMixin,
    PlotInteractionMixin,
    PredictionsTableMixin,
    GtTableMixin,
    DetailSegmentationMixin,
    DetailPredictionMixin,
    tk.Tk,
):
    """The assembled editor window."""

    def __init__(self, preselect: str | None = None):
        super().__init__()
        self.title("Prediction Editor")
        self.geometry("1700x950")

        # --- Loaded experiment state ---
        self.exp_name: str | None = None
        self.sensors: dict | None = None
        self.gt = None
        self.acc = None
        self.gyro = None
        self.prs = None
        self._t0_ms: int = 0
        self._acc_t0_ms: float = 0.0
        self._baro_alt = None
        self.predictions: list[dict] = []

        # --- Performance caches (avoid re-running the detector / reconstruction
        # on every click) ---
        self._detail_cache: dict = {}   # pred index -> findSegmentParameters dict
        self._param_cache: dict = {}    # (t_lo, t_hi, ride_type) -> params (GT)
        self._sig_cache: dict = {}      # reconstruct method -> reconstructedSignal df

        # --- Plot / selection state ---
        self._axes: list = []
        self._hl_spans: list = []
        self._last_sel: tuple[str, tuple] | None = None
        self.detail_pad_var = tk.DoubleVar(value=5.0)

        # --- Trapezoid override state (Prediction tab) ---
        self._override_enabled = tk.BooleanVar(value=False)
        self._override_W = tk.DoubleVar(value=1.0)
        self._override_f = tk.DoubleVar(value=0.5)
        self._override_A = tk.DoubleVar(value=1.5)

        self._build_ui()
        self._populate_experiments()

        if preselect:
            self.exp_var.set(preselect)
            self.after(50, self.load_experiment)

        self.after(150, self._fit_right_pane_sash)
        self.protocol("WM_DELETE_WINDOW", self.destroy)

    # ---------- Layout scaffold ----------

    def _build_ui(self) -> None:
        self._build_experiment_bar(self)

        main = ttk.Panedwindow(self, orient=tk.HORIZONTAL)
        main.pack(fill=tk.BOTH, expand=True)

        left = ttk.Frame(main)
        main.add(left, weight=4)
        self._build_overview(left)

        right_pane = ttk.Panedwindow(main, orient=tk.VERTICAL)
        main.add(right_pane, weight=2)
        self._right_pane = right_pane

        trees_frame = ttk.Frame(right_pane, padding=4)
        right_pane.add(trees_frame, weight=1)
        self._build_pred_tree(trees_frame)
        self._build_gt_tree(trees_frame)

        detail_frame = ttk.Frame(right_pane, padding=4)
        right_pane.add(detail_frame, weight=2)
        self._build_detail_pane(detail_frame)

    def _build_detail_pane(self, parent) -> None:
        header = ttk.Frame(parent)
        header.pack(fill=tk.X)
        ttk.Label(header, text="Detail — heatmaps + signal + verdict",
                  font=("", 10, "bold")).pack(side=tk.LEFT, anchor=tk.W)
        ttk.Label(header, text="  Window ± (s):").pack(side=tk.LEFT)
        pad_spin = ttk.Spinbox(
            header, from_=1.0, to=600.0, increment=2.0, width=6,
            textvariable=self.detail_pad_var, command=self._on_pad_changed,
        )
        pad_spin.pack(side=tk.LEFT, padx=(2, 4))
        pad_spin.bind("<Return>", lambda _e: self._on_pad_changed())
        pad_spin.bind("<FocusOut>", lambda _e: self._on_pad_changed())
        ttk.Button(header, text="×2", width=3,
                   command=lambda: self._scale_pad(2.0))\
            .pack(side=tk.LEFT, padx=1)
        ttk.Button(header, text="÷2", width=3,
                   command=lambda: self._scale_pad(0.5))\
            .pack(side=tk.LEFT, padx=1)
        ttk.Separator(header, orient=tk.VERTICAL)\
            .pack(side=tk.LEFT, padx=8, fill=tk.Y)
        ttk.Button(header, text="▶ Predict Δh",
                   command=self._on_predict_clicked)\
            .pack(side=tk.LEFT, padx=(2, 0))

        self.detail_nb = ttk.Notebook(parent)
        self.detail_nb.pack(fill=tk.BOTH, expand=True)

        seg_tab = ttk.Frame(self.detail_nb)
        self.detail_nb.add(seg_tab, text="Segmentation")
        self._build_detail_segmentation_tab(seg_tab)

        pred_tab = ttk.Frame(self.detail_nb)
        self.detail_nb.add(pred_tab, text="Prediction")
        self._build_detail_prediction_tab(pred_tab)

        self.verdict_text = tk.Text(
            parent, height=7, wrap="word", font=("Menlo", 9),
            background="#fafafa", foreground="#222",
            borderwidth=1, relief="solid",
        )
        self.verdict_text.pack(fill=tk.X, pady=(6, 0))
        self.verdict_text.configure(state="disabled")

        self._detail_placeholder()
        self._pred_placeholder()

    def _fit_right_pane_sash(self) -> None:
        try:
            self.update_idletasks()
            h = self._right_pane.winfo_height()
            if h > 200:
                self._right_pane.sashpos(0, int(h * 0.50))
        except (tk.TclError, AttributeError):
            pass


def main() -> int:
    preselect = sys.argv[1] if len(sys.argv) > 1 else None
    app = PredictionEditor(preselect=preselect)
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
