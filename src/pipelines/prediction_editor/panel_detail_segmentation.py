"""Segmentation detail tab: heatmaps + correlation + signal for one ride.

Everything shown here is read from
:func:`pyramidElevatorDist.findSegmentParameters` — the per-lobe R² heatmaps,
the per-sign correlation curves and the fitted lobe centres. The signal
strip plots the ``|a|-g`` residual from
:func:`pyramidElevatorDist.reconstructedSignal`. No detector internals.
"""
from __future__ import annotations

import tkinter as tk

import numpy as np
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from pyramidElevatorDist import findSegmentParameters

from .widgets_common import (
    PRED_COLORS, TYPE_COLORS, make_tick_formatter, set_verdict,
)


class DetailSegmentationMixin:
    """Owns the Segmentation tab figure and the detail-window pad."""

    def _build_detail_segmentation_tab(self, parent) -> None:
        self.detail_fig = Figure(figsize=(6, 5))
        self.detail_canvas = FigureCanvasTkAgg(self.detail_fig, master=parent)
        self.detail_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Detail-window pad ----------

    def _current_pad_s(self) -> float:
        try:
            v = float(self.detail_pad_var.get())
        except (tk.TclError, ValueError):
            v = 5.0
        return max(0.5, v)

    def _on_pad_changed(self) -> None:
        """Re-render whichever selection is active with the current pad."""
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
            t_s, t_e, rt, gi = payload
            self._render_detail_for_gt(t_s, t_e, rt, gi)

    def _scale_pad(self, factor: float) -> None:
        self.detail_pad_var.set(round(self._current_pad_s() * factor, 2))
        self._on_pad_changed()

    # ---------- Placeholder / time axis ----------

    def _detail_placeholder(
        self, msg: str = "Select a prediction or a GT ride.",
    ) -> None:
        self.detail_fig.clear()
        ax = self.detail_fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color="#666", style="italic")
        ax.set_axis_off()
        self.detail_canvas.draw_idle()
        set_verdict(self.verdict_text, "")

    def _apply_acc_time_axis(self, ax, *, label: str = "t (s, ACC-local)") -> None:
        ax.xaxis.set_major_formatter(make_tick_formatter(float(self._acc_t0_ms)))
        ax.set_xlabel(label)

    # ---------- Drawing helpers ----------

    def _draw_heatmap(self, ax, heat, grid_w_s, grid_f, title: str,
                      mark_W: float | None = None,
                      mark_f: float | None = None) -> None:
        grid_w_s = np.asarray(grid_w_s, dtype=float)
        grid_f = np.asarray(grid_f, dtype=float)
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

    def _draw_correlation(self, ax, correlation, t_lo: float, t_hi: float) -> None:
        """Per-sign best-R² traces over time (no threshold line, no dots)."""
        t = np.asarray(correlation["t"], dtype=float)
        pos = np.asarray(correlation["best_pos_r2"], dtype=float)
        neg = np.asarray(correlation["best_neg_r2"], dtype=float)
        mask = (t >= t_lo) & (t <= t_hi)
        ax.plot(t[mask], np.where(np.isfinite(pos), pos, np.nan)[mask],
                color="#2980b9", lw=0.9, label="max R² (+)")
        ax.plot(t[mask], np.where(np.isfinite(neg), neg, np.nan)[mask],
                color="#c0392b", lw=0.9, label="max R² (−)")
        ax.set_ylim(0, 1.05)
        ax.set_xlim(t_lo, t_hi)
        ax.set_ylabel("R² (per sign)")
        self._apply_acc_time_axis(ax)
        ax.grid(True, alpha=0.25)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    def _slice_base_signal(self, t_lo: float, t_hi: float):
        """Return ``(t_s, a_mag_g)`` for the cached display series inside the
        window, both as numpy arrays (empty when unavailable)."""
        sig = self._sig_base
        if sig is None or sig.empty:
            return np.array([]), np.array([])
        ts = (sig["timestamp_ms"].to_numpy(dtype=float) - self._t0_ms) / 1000.0
        a = sig["a_mag_g"].to_numpy(dtype=float)
        m = (ts >= t_lo) & (ts <= t_hi)
        return ts[m], a[m]

    # ---------- Shared render ----------

    def _render_segmentation_detail(
        self, params: dict, title: str, verdict: str,
        *, gt_span: tuple[float, float] | None = None,
    ) -> None:
        heatmaps = params["heatmaps"]
        correlation = params["correlation"]
        lobe1 = params["lobe1"]
        lobe2 = params["lobe2"]
        ride_type = str(params.get("ride_type", "up"))
        grid_w_s = heatmaps["grid_w_s"]
        grid_f = heatmaps["grid_f"]

        t_lo = float(params["t_start_s"])
        t_hi = float(params["t_end_s"])
        if gt_span is not None:
            t_lo = min(t_lo, float(gt_span[0]))
            t_hi = max(t_hi, float(gt_span[1]))

        self.detail_fig.clear()
        gs = self.detail_fig.add_gridspec(
            3, 2, height_ratios=[1.0, 0.9, 0.7], hspace=0.65, wspace=0.28,
        )
        ax_h1 = self.detail_fig.add_subplot(gs[0, 0])
        ax_h2 = self.detail_fig.add_subplot(gs[0, 1])
        ax_sig = self.detail_fig.add_subplot(gs[1, :])
        ax_rt = self.detail_fig.add_subplot(gs[2, :])

        W_star = float(lobe1["half_width_s"])
        f_star = float(lobe1["frac_flat"])
        self._draw_heatmap(ax_h1, heatmaps["lobe1"], grid_w_s, grid_f,
                           f"lobe1 @ t={float(lobe1['t_c']):.1f}s",
                           mark_W=W_star, mark_f=f_star)
        self._draw_heatmap(ax_h2, heatmaps["lobe2"], grid_w_s, grid_f,
                           f"lobe2 @ t={float(lobe2['t_c']):.1f}s",
                           mark_W=W_star, mark_f=f_star)

        pad = self._current_pad_s()
        t_arr, a_arr = self._slice_base_signal(t_lo - pad, t_hi + pad)
        ax_sig.plot(t_arr, a_arr, color="#2c3e50", lw=0.7, label="|a|-g")
        ax_sig.axhline(0, color="gray", lw=0.4, ls="--", alpha=0.5)
        if gt_span is not None:
            ax_sig.axvspan(gt_span[0], gt_span[1],
                           color=TYPE_COLORS.get(ride_type, "#cccccc"),
                           alpha=0.22, zorder=0)
        for L in (lobe1, lobe2):
            t_c = float(L["t_c"])
            A = float(L["a_peak"])
            ax_sig.scatter([t_c], [A], color="#c0392b", s=26, zorder=5)
            ax_sig.axvline(t_c, color="#c0392b", lw=0.6, ls=":", alpha=0.7)
        pred_col = PRED_COLORS.get(ride_type, "#888888")
        ax_sig.axvline(float(params["t_start_s"]), color=pred_col,
                       lw=1.0, ls="--", alpha=0.8)
        ax_sig.axvline(float(params["t_end_s"]), color=pred_col,
                       lw=1.0, ls="--", alpha=0.8)
        self._apply_acc_time_axis(ax_sig)
        ax_sig.set_ylabel("|a|-g (m/s²)")
        ax_sig.grid(True, alpha=0.25)
        ax_sig.legend(fontsize=8, loc="upper right")
        ax_sig.set_title(title, fontsize=9)

        self._draw_correlation(ax_rt, correlation, t_lo - pad, t_hi + pad)
        self.detail_canvas.draw_idle()
        set_verdict(self.verdict_text, verdict)

    # ---------- Entry points ----------

    def _render_detail_for_prediction(self, pred: dict) -> None:
        if self.acc is None:
            self._detail_placeholder("Load an experiment first.")
            return
        # Detail was precomputed at load (one detector pass) — instant lookup.
        idx = int(pred["index"])
        params = self._detail_cache.get(idx)
        if params is None and idx not in self._detail_cache:
            # Rare fallback: compute on demand and memoize.
            params = findSegmentParameters(
                self.acc, float(pred["t_start_s"]), float(pred["t_end_s"]),
                str(pred["ride_type"]), resample=True,
            )
            self._detail_cache[idx] = params
        if params is None:
            self._detail_placeholder("No trapezoid fit for this interval.")
            return
        lobe1 = params["lobe1"]
        lobe2 = params["lobe2"]
        W_star = float(lobe1["half_width_s"])
        f_star = float(lobe1["frac_flat"])
        title = (
            f"pred #{int(pred['index']):02d} {params['ride_type']} — "
            f"joint R²={float(params['joint_r2_mean']):.3f}  "
            f"|A|={abs(float(lobe1['a_peak'])):.2f}  "
            f"W={W_star:.2f}s  f={f_star:.2f}"
        )
        verdict = (
            f"prediction #{int(pred['index']):02d} — accepted pair.\n"
            f"  lobe1 t={float(lobe1['t_c']):.1f}s  "
            f"A={float(lobe1['a_peak']):+.2f}  "
            f"R²={float(lobe1['r2_local']):.3f}\n"
            f"  lobe2 t={float(lobe2['t_c']):.1f}s  "
            f"A={float(lobe2['a_peak']):+.2f}  "
            f"R²={float(lobe2['r2_local']):.3f}\n"
            f"  shared W={W_star:.2f}s  f={f_star:.2f}  "
            f"|A|={abs(float(lobe1['a_peak'])):.2f}\n"
            f"  joint mean R²={float(params['joint_r2_mean']):.3f}  "
            f"heatmap_energy={float(params.get('heatmap_energy', float('nan'))):.3f}"
        )
        self._render_segmentation_detail(params, title, verdict)

    def _render_detail_for_gt(self, t_lo: float, t_hi: float,
                              ride_type: str, gt_index: int) -> None:
        if self.acc is None:
            self._detail_placeholder("Load an experiment first.")
            return
        # GT windows are arbitrary (not in the detected set), so they're fit on
        # demand — but memoized per window so re-clicks / pad changes are instant.
        key = (round(float(t_lo), 2), round(float(t_hi), 2), str(ride_type))
        if key in self._param_cache:
            params = self._param_cache[key]
        else:
            params = findSegmentParameters(
                self.acc, float(t_lo), float(t_hi), ride_type, resample=True,
            )
            self._param_cache[key] = params
        matched = self._gt_matched_by_prediction(t_lo, t_hi)
        if params is None:
            self._detail_placeholder(
                f"GT #{gt_index:02d} {ride_type}: no usable +/− pair in "
                f"window [{t_lo:.1f}, {t_hi:.1f}]s."
            )
            return
        lobe1 = params["lobe1"]
        lobe2 = params["lobe2"]
        W_star = float(lobe1["half_width_s"])
        f_star = float(lobe1["frac_flat"])
        title = (
            f"GT #{gt_index:02d} {ride_type}  t=[{t_lo:.1f}, {t_hi:.1f}]s  "
            f"({'matched by detector' if matched else 'NOT matched'})"
        )
        verdict = (
            f"GT #{gt_index:02d} {ride_type}  window=[{t_lo:.1f}, {t_hi:.1f}]s  "
            f"(duration={t_hi - t_lo:.1f}s)\n"
            f"{'matched by detector.' if matched else 'NOT matched by detector.'}\n"
            f"  fitted trapezoid: joint R²={float(params['joint_r2_mean']):.3f}  "
            f"|A|={abs(float(lobe1['a_peak'])):.2f}  "
            f"W={W_star:.2f}s  f={f_star:.2f}\n"
            f"  lobe1 t={float(lobe1['t_c']):.1f}s  A={float(lobe1['a_peak']):+.2f}  "
            f"R²={float(lobe1['r2_local']):.3f}\n"
            f"  lobe2 t={float(lobe2['t_c']):.1f}s  A={float(lobe2['a_peak']):+.2f}  "
            f"R²={float(lobe2['r2_local']):.3f}"
        )
        self._render_segmentation_detail(
            params, title, verdict, gt_span=(float(t_lo), float(t_hi)),
        )
