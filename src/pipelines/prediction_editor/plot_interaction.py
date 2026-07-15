"""Overview-plot interaction: zoom, pan, scroll, click-to-select, hover.

Pure UI. Click/hover map cursor position onto the detector predictions and
GT rows (both already computed by the package) and drive tree selection;
nothing here computes signals.
"""
from __future__ import annotations

from .widgets_common import autoscale_y_to_visible, wall_time_str


class PlotInteractionMixin:
    """Zoom / pan / select behaviour for the overview canvas."""

    # ---------- Zoom / pan ----------

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

    def _focus_zoom_to_seconds(
        self, t_start_s: float, t_end_s: float, *, pad_s: float = 15.0,
    ) -> None:
        """Pan X to ``[t_start - pad, t_end + pad]`` and tighten each
        panel's Y to the data visible inside that window."""
        if not self._axes:
            return
        x_lo, x_hi = float(t_start_s) - pad_s, float(t_end_s) + pad_s
        self._axes[0].set_xlim(x_lo, x_hi)
        for ax in self._axes:
            autoscale_y_to_visible(ax, x_lo, x_hi)
        self.canvas.draw_idle()

    # ---------- Click / hover ----------

    def _on_canvas_click(self, event) -> None:
        """Select the prediction or GT interval under the cursor.

        Predictions are hatched on top of the GT bands, so they win if both
        overlap the click. Delegates the visuals to tree selection."""
        if (event.inaxes not in self._axes or event.xdata is None
                or event.button != 1):
            return
        tb = getattr(self.canvas.manager, "toolbar", None)
        if tb and getattr(tb, "mode", ""):
            return
        x = float(event.xdata)
        is_dbl = bool(getattr(event, "dblclick", False))

        for p in self.predictions:
            if p.t_start_s <= x <= p.t_end_s:
                if is_dbl:
                    self._focus_zoom_to_seconds(
                        float(p.t_start_s), float(p.t_end_s), pad_s=15.0,
                    )
                    return
                iid = str(p.index)
                if iid in self.tree_pred.get_children():
                    self.tree_pred.selection_set(iid)
                    self.tree_pred.see(iid)
                return

        for gi, t_s, t_e, _rt in self._gt_rows():
            if t_s <= x <= t_e:
                if is_dbl:
                    self._focus_zoom_to_seconds(
                        float(t_s), float(t_e), pad_s=15.0,
                    )
                    return
                iid = f"gt:{gi}"
                if iid in self.tree_gt.get_children():
                    self.tree_gt.selection_set(iid)
                    self.tree_gt.see(iid)
                return

    def _on_canvas_motion(self, event) -> None:
        if (event.inaxes in self._axes and event.xdata is not None
                and event.ydata is not None):
            ylab = event.inaxes.get_ylabel() or "y"
            wall = wall_time_str(
                float(self._t0_ms), float(event.xdata), sub_second=True,
            )
            wall_part = f"  ({wall})" if wall else ""
            self.hover_var.set(
                f"t = {event.xdata:8.2f} s{wall_part}    "
                f"{ylab} = {event.ydata:.3f}"
            )
        else:
            self.hover_var.set("")

    def _on_scroll(self, event) -> None:
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
