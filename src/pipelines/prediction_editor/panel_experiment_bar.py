"""Top bar: experiment picker, Load, reconstruct-method selector, nav.

Presentation + data loading only. The slimmed :meth:`load_experiment`
reads raw sensor data through :mod:`src.data.loader` and then delegates
*all* ride detection to :func:`pyramidElevatorDist.findSegments`. No
detector configuration or per-interval logic lives here anymore.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import messagebox, ttk

from pyramidElevatorDist import (
    findSegmentsDetailed, reconstructedSignal, barometricAltitude,
    RECONSTRUCT_CHOICES,
)

from src.data.loader import (
    RAW_DATA_ROOT,
    STRUCTURED_DATA_DIR,
    getExperimentData,
    list_experiments,
    list_structured_experiments,
)


class ExperimentBarMixin:
    """Owns the top control bar and the experiment load pipeline."""

    def _build_experiment_bar(self, parent) -> None:
        top = ttk.Frame(parent, padding=6)
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

        # Reconstruction-method selector. Feeds the display signal and the
        # Δh estimators (magnitude-invariant → intentionally not fed to the
        # segmentation calls).
        recon = ttk.Frame(top)
        recon.pack(side=tk.LEFT, padx=(12, 6))
        ttk.Label(recon, text="reconstruct:").pack(side=tk.LEFT)
        self.reconstruct_var = tk.StringVar(value="none")
        self.reconstruct_combo = ttk.Combobox(
            recon, textvariable=self.reconstruct_var, width=14,
            state="readonly", values=list(RECONSTRUCT_CHOICES),
        )
        self.reconstruct_combo.pack(side=tk.LEFT, padx=4)
        self.reconstruct_combo.bind(
            "<<ComboboxSelected>>", self._on_reconstruct_changed,
        )

        # Zoom / pan nav — commands resolve to the interaction + overview
        # mixins through the assembled class's MRO.
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

    # ---------- Experiments ----------

    def _populate_experiments(self) -> None:
        raw = list_experiments(RAW_DATA_ROOT)
        structured = list_structured_experiments(STRUCTURED_DATA_DIR)
        seen = set(raw)
        self.exp_combo["values"] = (
            list(raw) + [n for n in structured if n not in seen]
        )

    def _copy_exp_name(self) -> None:
        name = self.exp_var.get()
        if not name:
            self.status_var.set("No experiment name to copy.")
            return
        self.clipboard_clear()
        self.clipboard_append(name)
        self.update()
        self.status_var.set(f"Copied: {name}")

    def _on_reconstruct_changed(self, _event=None) -> None:
        """Re-draw the overview (and the current segmentation detail) with
        the newly picked reconstruction method. Δh predictions are not
        recomputed until the user clicks Predict again."""
        if self.acc is None:
            return
        self._refresh_plot()
        self._on_pad_changed()

    # ---------- Loading ----------

    def load_experiment(self) -> None:
        name = self.exp_var.get()
        if not name:
            self.status_var.set("Pick an experiment first.")
            return
        try:
            sensors, gt, _meta = getExperimentData(
                RAW_DATA_ROOT / name, use_cache=True,
            )
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Load failed", f"{type(e).__name__}: {e}")
            return

        acc = sensors.get("ACC")
        if acc is None or acc.empty:
            messagebox.showerror(
                "No ACC",
                f"Experiment {name!r} has no ACC samples — the editor "
                "needs ACC to anchor the time axis.",
            )
            return

        self.exp_name = name
        self.sensors = sensors
        self.gt = gt
        self.acc = acc
        self.gyro = sensors.get("GYR")
        self.prs = sensors.get("PRS")

        # Raw data is NOT pre-resampled here: the package resamples onto its
        # canonical 50 Hz grid internally (resample=True everywhere). The
        # grid is anchored on the first raw ACC timestamp, so this session
        # anchor lines up with every returned ride coordinate.
        self._t0_ms = int(acc["timestamp_ms"].iloc[0])
        self._acc_t0_ms = float(self._t0_ms)

        # Fresh caches for the new experiment.
        self._detail_cache = {}
        self._param_cache = {}
        self._sig_cache = {}

        self.status_var.set(f"Detecting rides in {name}…")
        self.update_idletasks()
        try:
            # One detector pass returns every ride WITH its detail (heatmaps +
            # correlation), so segment clicks are instant instead of re-running
            # the detector each time.
            self.predictions = findSegmentsDetailed(acc, resample=True)
            self._detail_cache = {
                int(p["index"]): p.get("detail") for p in self.predictions
            }
        except Exception as e:  # noqa: BLE001
            messagebox.showerror("Detector failed", f"{type(e).__name__}: {e}")
            self.predictions = []

        # Reconstruction-invariant display series cached once for the detail
        # panels (|a|-g does not change with the orientation reconstruction).
        try:
            self._sig_base = reconstructedSignal(acc, resample=True)
        except Exception:  # noqa: BLE001
            self._sig_base = None

        # Barometric altitude (ground-truth reference) for the overview graph.
        # Empty frame when the experiment has no pressure stream.
        try:
            self._baro_alt = barometricAltitude(self.prs)
        except Exception:  # noqa: BLE001
            self._baro_alt = None

        self._last_sel = None
        self._refresh_plot()
        self._refresh_pred_tree()
        self._refresh_gt_tree()
        self._detail_placeholder()
        self._pred_placeholder()

        pred_n = len(self.predictions)
        gt_n = 0 if gt is None else int(
            gt["type"].isin(("up", "down")).sum() if "type" in gt.columns else 0
        )
        self.status_var.set(
            f"Loaded {name} — {pred_n} predictions (live), {gt_n} GT rides."
        )
