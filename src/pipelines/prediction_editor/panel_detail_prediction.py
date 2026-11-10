"""Prediction detail tab: per-algorithm Δh table + trapezoid / ZUPT panels.

All numbers come from :func:`pyramidElevatorDist.predictSegment` (or
:func:`pyramidElevatorDist.predictByParameters` when the trapezoid override
is active). This file only lays the results out; it does no estimation.
"""
from __future__ import annotations

import math
import tkinter as tk
from tkinter import ttk

import numpy as np

from pyramidElevatorDist import predictByParameters, predictSegment

from .widgets_common import make_tree, set_verdict

# Short id -> display label, in a fixed draw order across every panel. The
# package exposes exactly these two Δh estimators.
_ALGO_ROWS = (("trap", "trapezoid_accel"), ("zupt", "zupt_accel"))


class DetailPredictionMixin:
    """Owns the Prediction tab (override controls + table + figure)."""

    def _build_detail_prediction_tab(self, parent) -> None:
        self._build_override_controls(parent)

        from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
        from matplotlib.figure import Figure

        split = ttk.Panedwindow(parent, orient=tk.VERTICAL)
        split.pack(fill=tk.BOTH, expand=True)

        table_frame = ttk.Frame(split, padding=2)
        split.add(table_frame, weight=1)
        self.pred_table = make_tree(
            table_frame,
            columns=[
                ("algo",    "algorithm",  170, tk.W),
                ("dh",      "Δh (m)",      80, tk.E),
                ("ci",      "CI ± (m)",    80, tk.E),
                ("q",       "q",           60, tk.E),
                ("verdict", "verdict",     90, tk.W),
                ("reason",  "reason",     240, tk.W),
            ],
            on_select=lambda _e: None,
        )
        self.pred_table.configure(height=4)
        self.pred_table.tag_configure("ok", foreground="#1e7a3a")
        self.pred_table.tag_configure("reject", foreground="#a04000")
        self.pred_table.tag_configure("error", foreground="#7f0000")
        self.pred_table.tag_configure("gt", foreground="#2060a0")

        canvas_frame = ttk.Frame(split, padding=2)
        split.add(canvas_frame, weight=3)
        self.pred_fig = Figure(figsize=(6, 7))
        self.pred_canvas = FigureCanvasTkAgg(self.pred_fig, master=canvas_frame)
        self.pred_canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

    # ---------- Override controls ----------

    def _build_override_controls(self, parent) -> None:
        frame = ttk.LabelFrame(
            parent, text="Trapezoid override (predictor)", padding=4,
        )
        frame.pack(fill=tk.X, padx=2, pady=(2, 4))

        ttk.Checkbutton(
            frame, text="Use override", variable=self._override_enabled,
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(frame, text="W (s):").pack(side=tk.LEFT)
        ttk.Spinbox(
            frame, from_=0.01, to=30.0, increment=0.1, width=7,
            textvariable=self._override_W, format="%.3f",
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(frame, text="f:").pack(side=tk.LEFT)
        ttk.Spinbox(
            frame, from_=0.0, to=1.0, increment=0.05, width=6,
            textvariable=self._override_f, format="%.3f",
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Label(frame, text="|A| (m/s²):").pack(side=tk.LEFT)
        ttk.Spinbox(
            frame, from_=0.0, to=30.0, increment=0.05, width=7,
            textvariable=self._override_A, format="%.3f",
        ).pack(side=tk.LEFT, padx=(2, 8))

        ttk.Button(frame, text="Reset",
                   command=self._reset_override_to_estimated)\
            .pack(side=tk.LEFT, padx=(8, 2))
        ttk.Button(frame, text="Re-predict",
                   command=self._on_predict_clicked)\
            .pack(side=tk.LEFT, padx=2)

    def _reset_override_to_estimated(self) -> None:
        self._override_enabled.set(False)
        if self._last_sel is None or self._last_sel[0] != "pred":
            self.status_var.set("Override cleared.")
            return
        (idx,) = self._last_sel[1]
        match = next(
            (p for p in self.predictions if p.index == idx), None,
        )
        if match is None:
            self.status_var.set("Override cleared.")
            return
        l1 = match.lobe1
        W = float(l1.half_width_s)
        f = float(l1.frac_flat)
        A = abs(float(l1.a_peak))
        self._override_W.set(round(W, 3))
        self._override_f.set(round(f, 3))
        self._override_A.set(round(A, 3))
        self.status_var.set(
            f"Reset override to estimated shape for pred #{idx:02d} "
            f"(W={W:.2f}s, f={f:.2f}, |A|={A:.2f})."
        )

    def _current_override(self) -> dict | None:
        if not self._override_enabled.get():
            return None
        try:
            return {
                "W":     float(self._override_W.get()),
                "f":     float(self._override_f.get()),
                "abs_A": abs(float(self._override_A.get())),
            }
        except (tk.TclError, TypeError, ValueError):
            return None

    # ---------- Selection resolution ----------

    def _interval_for_sel(self) -> tuple[float, float, str, str] | None:
        """Resolve ``self._last_sel`` to ``(start_s, end_s, ride_type, label)``."""
        if self._last_sel is None:
            return None
        kind, payload = self._last_sel
        if kind == "pred":
            (idx,) = payload
            match = next(
                (p for p in self.predictions if p.index == idx), None,
            )
            if match is None:
                return None
            return (
                float(match.t_start_s),
                float(match.t_end_s),
                str(match.ride_type),
                f"pred #{idx:02d} ({match.ride_type})",
            )
        if kind == "gt":
            t_s, t_e, rt, gi = payload
            return float(t_s), float(t_e), str(rt), f"gt #{gi:02d} ({rt})"
        return None

    # ---------- Run + render ----------

    def _run_predictors(self, t_start: float, t_end: float,
                        ride_type: str) -> dict:
        """Run the Δh estimators over the selected interval via the package."""
        segment = {"type": ride_type, "start_s": t_start, "end_s": t_end}
        method = (
            self.reconstruct_var.get()
            if hasattr(self, "reconstruct_var") else "none"
        )
        override = self._current_override()
        if override is not None:
            return predictByParameters(
                self.acc, segment, override, resample=True,
                gyro=self.gyro, reconstruct=method, prs=self.prs,
            )
        return predictSegment(
            self.acc, segment, resample=True,
            gyro=self.gyro, reconstruct=method, prs=self.prs,
        )

    def _on_predict_clicked(self) -> None:
        if self.acc is None:
            self.status_var.set("Load an experiment first.")
            return
        interval = self._interval_for_sel()
        if interval is None:
            self.status_var.set(
                "Select a prediction or a GT row before predicting."
            )
            return
        t_start, t_end, ride_type, label = interval
        self.status_var.set(f"Running Δh estimators on {label}…")
        self.update_idletasks()

        result = self._run_predictors(t_start, t_end, ride_type)
        summary = self._format_predictions(result, label)
        current = self.verdict_text.get("1.0", "end-1c").rstrip()
        combined = (current + "\n\n" + summary) if current else summary
        set_verdict(self.verdict_text, combined)

        self._render_prediction_tab(result, label, t_start)
        try:
            self.detail_nb.select(1)
        except tk.TclError:
            pass
        self.status_var.set(f"Predicted Δh for {label}.")

    def _format_predictions(self, result: dict, label: str) -> str:
        # Ground-truth (barometer) Δh, when a pressure stream was supplied.
        baro = result.baro
        gt = None
        if baro is not None:
            g = float(baro.delta_height_m)
            gt = g if math.isfinite(g) else None

        lines = [f"Δh estimators — {label}:"]
        for aid, name in _ALGO_ROWS:
            row = result.row(aid)
            if row is None:
                lines.append(f"  {name:22s} — not run")
                continue
            dh = float(row.delta_height_m)
            ci = float(row.ci_half_width)
            q = float(row.quality_score)
            ci_str = f"±{ci:.2f}m" if math.isfinite(ci) else "±inf"
            verdict = (
                "OK" if row.accepted
                else f"REJECT ({row.reject_reason or 'no reason'})"
            )
            err_str = (f"   err={dh - gt:+.2f}m"
                       if gt is not None and math.isfinite(dh) else "")
            lines.append(
                f"  {name:22s} Δh = {dh:+7.2f} m   CI {ci_str}   "
                f"q={q:.2f}   [{verdict}]{err_str}"
            )
        if gt is not None:
            lines.append(f"  {'barometer (GT)':22s} Δh = {gt:+7.2f} m   "
                         f"[ground truth]")
        elif baro is not None:
            lines.append(f"  {'barometer (GT)':22s} — "
                         f"{baro.reject_reason or 'no pressure'}")
        return "\n".join(lines)

    # ---------- Table ----------

    def _pred_placeholder(
        self, msg: str = "Click ▶ Predict Δh to populate.",
    ) -> None:
        if hasattr(self, "pred_table"):
            for iid in self.pred_table.get_children():
                self.pred_table.delete(iid)
        self.pred_fig.clear()
        ax = self.pred_fig.add_subplot(111)
        ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                va="center", fontsize=11, color="#666", style="italic")
        ax.set_axis_off()
        self.pred_canvas.draw_idle()

    def _populate_pred_table(self, result: dict, _label: str) -> None:
        for iid in self.pred_table.get_children():
            self.pred_table.delete(iid)

        def _fmt(v: float, fmt: str) -> str:
            return format(v, fmt) if math.isfinite(v) else "—"

        for aid, name in _ALGO_ROWS:
            row = result.row(aid)
            if row is None:
                self.pred_table.insert(
                    "", "end", iid=aid,
                    values=(name, "—", "—", "—", "—", "not run"),
                    tags=("error",),
                )
                continue
            dh = float(row.delta_height_m)
            ci = float(row.ci_half_width)
            q = float(row.quality_score)
            accepted = bool(row.accepted)
            reason = str(row.reject_reason or "")
            ov_meta = row.meta.get("trapezoid_override")
            verdict = "OK" if accepted else "REJECT"
            if ov_meta:
                verdict = f"{verdict} (override)"
            label = f"{name}  ✱" if ov_meta else name
            ci_str = f"±{ci:.2f}" if math.isfinite(ci) else "±inf"
            self.pred_table.insert(
                "", "end", iid=aid,
                values=(label, _fmt(dh, "+.2f"), ci_str, _fmt(q, ".2f"),
                        verdict, reason[:240]),
                tags=("ok" if accepted else "reject",),
            )

        # Ground-truth (barometer) reference row, when pressure is available.
        baro = result.baro
        if baro is not None:
            dh = float(baro.delta_height_m)
            reason = str(baro.reject_reason or "")
            self.pred_table.insert(
                "", "end", iid="baro",
                values=("barometer (GT)", _fmt(dh, "+.2f"), "—", "—",
                        "GT" if not reason else "n/a", reason[:240]),
                tags=("gt",),
            )

    # ---------- Figure ----------

    def _render_prediction_tab(self, result: dict, label: str,
                               t_start_acc: float) -> None:
        self._populate_pred_table(result, label)
        self.pred_fig.clear()
        gs = self.pred_fig.add_gridspec(
            2, 1, height_ratios=[2.0, 1.3], hspace=0.55,
            left=0.12, right=0.97, top=0.94, bottom=0.09,
        )
        self._render_pred_trapezoid(self.pred_fig.add_subplot(gs[0, 0]),
                                    result, t_start_acc)
        self._render_pred_zupt(self.pred_fig.add_subplot(gs[1, 0]), result)
        self.pred_canvas.draw_idle()

    def _render_pred_trapezoid(self, ax, result: dict,
                               t_start_acc: float = 0.0) -> None:
        ax.set_title("Trapezoid fit on accelerometer signal",
                     fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
        self._apply_acc_time_axis(ax)
        ax.xaxis.label.set_fontsize(8)
        ax.set_ylabel("a (m/s²)", fontsize=8)

        row = result.trap
        if row is None:
            ax.text(0.5, 0.5, "trapezoid estimator not run",
                    transform=ax.transAxes, ha="center", va="center",
                    fontsize=9, color="#888", style="italic")
            return
        meta = row.meta
        t_sec = meta.get("t_sec")
        a_smooth = meta.get("a_smooth")
        a_template = meta.get("a_template")
        params = meta.get("params") or {}
        if t_sec is None or a_smooth is None or a_template is None:
            reason = str(row.reject_reason or "")
            msg = ("no trapezoid template returned"
                   if not reason else f"no template — {reason}")
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color="#888", style="italic")
            return

        t_arr = np.asarray(t_sec, dtype=float) + float(t_start_acc)
        ax.plot(t_arr, np.asarray(a_smooth, dtype=float),
                color="#2c3e50", lw=0.9, label="a_smooth")
        ax.plot(t_arr, np.asarray(a_template, dtype=float),
                color="#c0392b", lw=1.6, alpha=0.9, label="trapezoid template")
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
            ax.text(0.01, 0.98, txt, transform=ax.transAxes, ha="left",
                    va="top", fontsize=7, family="monospace",
                    bbox=dict(facecolor="#ffffff", alpha=0.85,
                              edgecolor="#888", boxstyle="round,pad=0.3"),
                    zorder=20)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)

    def _render_pred_zupt(self, ax, result: dict) -> None:
        ax.set_title("ZUPT integrated position", fontsize=9, loc="left")
        ax.grid(True, alpha=0.25)
        ax.set_xlabel("sample index", fontsize=8)
        ax.set_ylabel("pos (m)", fontsize=8)

        row = result.zupt
        if row is None:
            ax.text(0.5, 0.5, "ZUPT estimator not run", transform=ax.transAxes,
                    ha="center", va="center", fontsize=9, color="#888",
                    style="italic")
            return
        meta = row.meta
        pos = meta.get("pos_curve")
        if pos is None:
            reason = str(row.reject_reason or "")
            msg = ("no ZUPT trajectory returned"
                   if not reason else f"no trajectory — {reason}")
            ax.text(0.5, 0.5, msg, transform=ax.transAxes, ha="center",
                    va="center", fontsize=9, color="#888", style="italic")
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

        info_bits = []
        dh = float(row.delta_height_m)
        if math.isfinite(dh):
            info_bits.append(f"final Δh={dh:+.2f} m")
        n_active = meta.get("n_active")
        if n_active is not None:
            info_bits.append(f"n_active={int(n_active)}")
        active_frac = meta.get("active_fraction")
        if active_frac is not None:
            info_bits.append(f"active_frac={float(active_frac):.2f}")
        method = meta.get("method", "")
        if method:
            info_bits.append(f"method={method}")
        if info_bits:
            ax.text(0.01, 0.98, "  ".join(info_bits), transform=ax.transAxes,
                    ha="left", va="top", fontsize=7, family="monospace",
                    bbox=dict(facecolor="#ffffff", alpha=0.85,
                              edgecolor="#888", boxstyle="round,pad=0.3"),
                    zorder=20)
        ax.legend(fontsize=7, loc="lower right", framealpha=0.85)
