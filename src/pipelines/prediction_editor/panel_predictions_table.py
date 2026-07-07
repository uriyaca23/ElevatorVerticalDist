"""Predictions table: one row per detector-accepted ride.

Rows come verbatim from :func:`pyramidElevatorDist.findSegments`. Clicking a
row highlights its span and renders the segmentation detail; double-click
zooms the overview to the ride.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .widgets_common import PRED_COLORS, make_tree


class PredictionsTableMixin:
    """Owns ``tree_pred`` and its selection behaviour."""

    def _build_pred_tree(self, parent) -> None:
        block = ttk.Frame(parent)
        block.pack(fill=tk.BOTH, expand=True)
        ttk.Label(block, text="Predictions", font=("", 10, "bold"))\
            .pack(anchor=tk.W)
        self.tree_pred = make_tree(
            block,
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
        self.tree_pred.tag_configure("up", foreground=PRED_COLORS["up"])
        self.tree_pred.tag_configure("down", foreground=PRED_COLORS["down"])
        self.tree_pred.bind("<Double-Button-1>", self._on_pred_double_click)

    def _refresh_pred_tree(self) -> None:
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

    def _on_pred_select(self, _event=None) -> None:
        sel = self.tree_pred.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except ValueError:
            return
        match = next(
            (p for p in self.predictions if int(p["index"]) == idx), None,
        )
        if match is None:
            return
        self._highlight_on_plot(match["t_start_s"], match["t_end_s"])
        self._last_sel = ("pred", (idx,))
        self._render_detail_for_prediction(match)

    def _on_pred_double_click(self, _event=None) -> None:
        sel = self.tree_pred.selection()
        if not sel:
            return
        try:
            idx = int(sel[0])
        except ValueError:
            return
        match = next(
            (p for p in self.predictions if int(p["index"]) == idx), None,
        )
        if match is None:
            return
        self._focus_zoom_to_seconds(
            float(match["t_start_s"]), float(match["t_end_s"]), pad_s=15.0,
        )
