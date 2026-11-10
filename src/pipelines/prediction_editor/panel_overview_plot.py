"""Left timeline: the whole-trace vertical-accel overview plot.

The signal comes straight from :func:`pyramidElevatorDist.reconstructedSignal`;
:func:`pyramidElevatorDist.displaySeries` then picks the trace to draw — the
gyro-reconstructed a_z when a gyro + method are active, else the ``|a|-g``
fallback. GT bands and detector prediction spans are overlaid; this file owns
no signal processing.
"""
from __future__ import annotations

import tkinter as tk

from matplotlib.backends.backend_tkagg import (
    FigureCanvasTkAgg, NavigationToolbar2Tk,
)
from matplotlib.figure import Figure

from pyramidElevatorDist import displaySeries, reconstructedSignal

from .widgets_common import (
    HIGHLIGHT_COLOR, PRED_COLORS, TYPE_COLORS, make_tick_formatter,
)


class OverviewPlotMixin:
    """Owns the main matplotlib canvas and its redraw."""

    def _build_overview(self, parent) -> None:
        self.fig = Figure(figsize=(11, 8))
        self.canvas = FigureCanvasTkAgg(self.fig, master=parent)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        NavigationToolbar2Tk(self.canvas, parent).update()

        # Mouse interaction (handlers live in the interaction mixin).
        self.canvas.mpl_connect("button_press_event", self._on_canvas_click)
        self.canvas.mpl_connect("motion_notify_event", self._on_canvas_motion)
        self.canvas.mpl_connect("scroll_event", self._on_scroll)

        widget = self.canvas.get_tk_widget()
        widget.bind("<MouseWheel>", self._on_tk_wheel)
        widget.bind("<Button-4>", lambda e: self._on_tk_wheel(e, delta=+120))
        widget.bind("<Button-5>", lambda e: self._on_tk_wheel(e, delta=-120))

        # Keyboard shortcuts (bound on the root window).
        self.bind("<plus>", lambda e: self._zoom_x_around_center(zoom_in=True))
        self.bind("<equal>", lambda e: self._zoom_x_around_center(zoom_in=True))
        self.bind("<minus>", lambda e: self._zoom_x_around_center(zoom_in=False))
        self.bind("<Key-0>", lambda e: self._fit_x())
        self.bind("<Shift-Left>", lambda e: self._pan_x(-0.25))
        self.bind("<Shift-Right>", lambda e: self._pan_x(+0.25))
        self.bind("<Shift-Up>", lambda e: self._pan_y(+0.25))
        self.bind("<Shift-Down>", lambda e: self._pan_y(-0.25))

    def _apply_session_time_axis(self, ax) -> None:
        ax.xaxis.set_major_formatter(make_tick_formatter(float(self._t0_ms)))
        ax.set_xlabel("time")

    def _display_signal(self, method: str):
        """Cached ``reconstructedSignal`` per method. Reconstruction over the
        whole trace is expensive (gyro fusion), so compute it once per method
        and reuse — switching back to a method is then instant."""
        cache = getattr(self, "_sig_cache", None)
        if cache is None:
            self._sig_cache = cache = {}
        if method in cache:
            return cache[method]
        if method != "none":
            self.status_var.set(f"Reconstructing signal ({method})…")
            self.update_idletasks()
        try:
            sig = reconstructedSignal(
                self.acc, gyro=self.gyro, reconstruct=method, resample=True,
            )
        except Exception:  # noqa: BLE001
            sig = None
        cache[method] = sig
        return sig

    @staticmethod
    def _decimate(x, y, max_pts: int = 6000):
        """Uniformly thin a whole-trace series for display. ~90k points make
        every redraw (highlight/zoom/pan) sluggish; a few thousand is visually
        identical at overview zoom. Detail panels use the full-res signal."""
        n = len(x)
        if n <= max_pts:
            return x, y
        step = n // max_pts + 1
        return x[::step], y[::step]

    def _refresh_plot(self) -> None:
        self.fig.clear()
        self._axes = []
        self._hl_spans = []
        if self.acc is None or self.acc.empty:
            self.canvas.draw()
            return

        method = (
            self.reconstruct_var.get()
            if hasattr(self, "reconstruct_var") else "none"
        )
        sig = self._display_signal(method)

        # Barometric altitude (ground truth) shares the x-axis when available,
        # so the elevator's true height change sits under the acceleration.
        baro = getattr(self, "_baro_alt", None)
        has_baro = baro is not None and not baro.empty
        if has_baro:
            ax = self.fig.add_subplot(211)
            ax_alt = self.fig.add_subplot(212, sharex=ax)
        else:
            ax = self.fig.add_subplot(111)
            ax_alt = None

        # --- acceleration panel ---
        has_gyro = self.gyro is not None and not self.gyro.empty
        y_vals, y_label, _ = displaySeries(sig, has_gyro, method)
        if sig is not None and not sig.empty and y_vals.size:
            ts = (sig["timestamp_ms"].to_numpy(dtype=float) - self._t0_ms) / 1000.0
            tx, av = self._decimate(ts, y_vals)
            ax.plot(tx, av, color="#2c3e50", lw=0.6, label=y_label)
            ax.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
        ax.set_ylabel(f"{y_label} (m/s²)")
        ax.legend(loc="upper right", fontsize=7, frameon=False)
        ax.grid(True, alpha=0.3)
        ax.set_title(f"{y_label}  (reconstruct = {method})", fontsize=9, loc="left")

        # --- barometric altitude (ground-truth) panel ---
        if ax_alt is not None:
            at = (baro["timestamp_ms"].to_numpy(dtype=float) - self._t0_ms) / 1000.0
            atx, altd = self._decimate(
                at, baro["altitude_m"].to_numpy(dtype=float))
            ax_alt.plot(atx, altd, color="#16a085", lw=0.9,
                        label="barometer altitude (GT)")
            ax_alt.set_ylabel("altitude (m)")
            ax_alt.legend(loc="upper right", fontsize=7, frameon=False)
            ax_alt.grid(True, alpha=0.3)
            ax_alt.set_title("barometric altitude — ground truth",
                             fontsize=9, loc="left")

        axes = [ax] + ([ax_alt] if ax_alt is not None else [])

        # GT bands + detector prediction spans on every panel.
        for a in axes:
            if self.gt is not None and not self.gt.empty:
                for _, row in self.gt.iterrows():
                    rt = str(row.get("type", ""))
                    if rt not in ("up", "down"):
                        continue
                    s = (int(row["start_ms"]) - self._t0_ms) / 1000.0
                    e = (int(row["end_ms"]) - self._t0_ms) / 1000.0
                    a.axvspan(s, e, color=TYPE_COLORS.get(rt, "#cccccc"),
                              alpha=0.15, zorder=0)
            for p in self.predictions:
                col = PRED_COLORS[p.ride_type]
                a.axvspan(p.t_start_s, p.t_end_s, color=col,
                          alpha=0.18, hatch="//", zorder=1)
                a.axvline(p.t_start_s, color=col, lw=0.9, ls="--",
                          alpha=0.7, zorder=2)
                a.axvline(p.t_end_s, color=col, lw=0.9, ls="--",
                          alpha=0.7, zorder=2)

        # Time axis label only on the bottom panel (they share x).
        self._apply_session_time_axis(axes[-1])

        self._axes = axes
        self.fig.tight_layout()
        self.canvas.draw()

    def _highlight_on_plot(self, t_start: float, t_end: float) -> None:
        for h in self._hl_spans:
            try:
                h.remove()
            except Exception:  # noqa: BLE001
                pass
        self._hl_spans = []
        for ax in self._axes:
            h = ax.axvspan(t_start, t_end, edgecolor=HIGHLIGHT_COLOR,
                           facecolor="none", lw=2.2, zorder=12)
            self._hl_spans.append(h)
        self.canvas.draw_idle()

    def _fit_x(self) -> None:
        if self.acc is None or self.acc.empty or not self._axes:
            return
        t0 = int(self.acc["timestamp_ms"].iloc[0])
        t1 = int(self.acc["timestamp_ms"].iloc[-1])
        self._axes[0].set_xlim(
            (t0 - self._t0_ms) / 1000.0, (t1 - self._t0_ms) / 1000.0,
        )
        self.canvas.draw_idle()
