"""GT rides table: every ``up`` / ``down`` interval from ``gt.csv``.

Each row is shown whether or not the detector accepted it. Clicking one
renders the same segmentation detail a prediction gets — the trapezoid the
package fits inside the marked interval — so the user can see why a GT ride
was or was not matched.
"""
from __future__ import annotations

import tkinter as tk
from tkinter import ttk

from .widgets_common import TYPE_COLORS, make_tree


class GtTableMixin:
    """Owns ``tree_gt`` and its selection behaviour."""

    def _build_gt_tree(self, parent) -> None:
        block = ttk.Frame(parent)
        block.pack(fill=tk.BOTH, expand=True, pady=(10, 0))
        ttk.Label(block, text="GT rides (click to inspect)",
                  font=("", 10, "bold")).pack(anchor=tk.W)
        self.tree_gt = make_tree(
            block,
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
        self.tree_gt.tag_configure("up", foreground=TYPE_COLORS["up"])
        self.tree_gt.tag_configure("down", foreground=TYPE_COLORS["down"])
        self.tree_gt.bind("<Double-Button-1>", self._on_gt_double_click)

    def _gt_rows(self) -> list[tuple[int, float, float, str]]:
        """Return ``(gt_index, t_start_s, t_end_s, ride_type)`` for every
        ``up`` / ``down`` row, in seconds relative to the session anchor."""
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
            if p.t_start_s <= t_hi and p.t_end_s >= t_lo:
                return True
        return False

    def _refresh_gt_tree(self) -> None:
        for item in self.tree_gt.get_children():
            self.tree_gt.delete(item)
        for gi, t_s, t_e, rt in self._gt_rows():
            matched = self._gt_matched_by_prediction(t_s, t_e)
            self.tree_gt.insert(
                "", tk.END, iid=f"gt:{gi}",
                values=(
                    gi,
                    f"{t_s:.1f}",
                    f"{t_e:.1f}",
                    f"{t_e - t_s:.1f}",
                    rt,
                    "matched" if matched else "unmatched",
                ),
                tags=(rt,),
            )

    def _on_gt_select(self, _event=None) -> None:
        sel = self.tree_gt.selection()
        if not sel or not sel[0].startswith("gt:"):
            return
        gi = int(sel[0].split(":", 1)[1])
        match = next(
            (row for row in self._gt_rows() if row[0] == gi), None,
        )
        if match is None:
            return
        _, t_s, t_e, rt = match
        self._highlight_on_plot(t_s, t_e)
        self._last_sel = ("gt", (t_s, t_e, rt, gi))
        self._render_detail_for_gt(t_s, t_e, rt, gi)

    def _on_gt_double_click(self, _event=None) -> None:
        sel = self.tree_gt.selection()
        if not sel or not sel[0].startswith("gt:"):
            return
        try:
            gi = int(sel[0].split(":", 1)[1])
        except ValueError:
            return
        match = next(
            (row for row in self._gt_rows() if row[0] == gi), None,
        )
        if match is None:
            return
        _, t_s, t_e, _rt = match
        self._focus_zoom_to_seconds(float(t_s), float(t_e), pad_s=15.0)
