"""Step 3 — Interactive segmentation.

Runs the trapezoid-template detector on the loaded signal and shows an
editor-style detail panel: signal + segments, per-lobe heatmaps, the
correlation panel with peak-status dots, the fitted-trapezoid overlay,
and a card summarising the trapezoid parameters.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from .common import (
    LoadedSignal,
    PEAK_STATUS_COLORS,
    RIDE_COLORS,
    SELECTED_COLOR,
    STEP_DATA,
    STEP_PREDICT,
    classify_peak,
    find_local_maxima,
    find_matching_prediction,
    goto,
    heatmap_at,
    peak_status_legend_html,
    predict_intervals,
    render_segment_sidebar,
    render_trapezoid_params,
    trapezoid_kernel,
    valid_segments,
)


def _run_detector(loaded: LoadedSignal) -> None:
    with st.spinner("Running trapezoid-template detector…"):
        preds, state = predict_intervals(loaded.acc)
    st.session_state["detector_state"] = state if state else None
    st.session_state["predictions"] = preds
    rows = [
        {"type":     p["ride_type"],
         "start_s":  round(float(p["t_start_s"]), 2),
         "end_s":    round(float(p["t_end_s"]), 2),
         "joint_r2": round(float(p.get("joint_r2_mean", 0.0)), 3)}
        for p in preds
    ]
    st.session_state["segments_df"] = pd.DataFrame(
        rows, columns=["type", "start_s", "end_s", "joint_r2"],
    )
    st.session_state["selected_segment"] = 0 if rows else None


def _main_signal_figure(
    state: dict, segments_df: pd.DataFrame, selected_idx: int | None,
) -> go.Figure:
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=a_vert, mode="lines", name="a_vert",
        line=dict(color="#233044", width=1),
        hovertemplate="t=%{x:.2f}s<br>a=%{y:.2f} m/s²<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t, y=a_smooth, mode="lines", name="smoothed",
        line=dict(color="#e67e22", width=1.6),
    ))
    for i, row in segments_df.iterrows():
        try:
            s = float(row["start_s"]); e = float(row["end_s"])
        except (TypeError, ValueError):
            continue
        if not np.isfinite(s) or not np.isfinite(e) or e <= s:
            continue
        rt = str(row.get("type", "up")).lower()
        is_selected = (selected_idx is not None and i == selected_idx)
        base = RIDE_COLORS.get(rt, RIDE_COLORS["up"])
        fig.add_vrect(
            x0=s, x1=e,
            fillcolor=SELECTED_COLOR if is_selected else base,
            opacity=0.32 if is_selected else 0.16,
            line_width=2 if is_selected else 0,
            line_color=SELECTED_COLOR if is_selected else base,
            annotation_text=f"#{i} {rt}",
            annotation_position="top left",
            annotation_font_color=(SELECTED_COLOR if is_selected else base),
            annotation_font_size=11,
        )
    fig.update_layout(
        height=360, margin=dict(l=10, r=10, t=30, b=30),
        xaxis_title="time (s)", yaxis_title="a_vert (m/s²)",
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="#fafbfc",
    )
    return fig


def _heatmap_figure(state: dict, i_center: int, title: str,
                    mark_W: float | None = None,
                    mark_f: float | None = None) -> go.Figure:
    heat = heatmap_at(
        state["a_smooth"], state["t"], i_center,
        state.get("grid_w_s"), state.get("grid_f"),
    )
    grid_w_s = state["grid_w_s"]; grid_f = state["grid_f"]
    fig = go.Figure(go.Heatmap(
        z=heat, x=grid_f, y=grid_w_s,
        colorscale="Viridis", zmin=0.0, zmax=1.0,
        colorbar=dict(title="R²"),
    ))
    if mark_W is not None and mark_f is not None:
        fig.add_trace(go.Scatter(
            x=[mark_f], y=[mark_W], mode="markers",
            marker=dict(symbol="x", size=14, color="#e74c3c", line=dict(width=2)),
            name="best W,f", showlegend=False,
        ))
    fig.update_layout(
        title=title, height=260, margin=dict(l=10, r=10, t=40, b=30),
        xaxis_title="plateau f", yaxis_title="half-width W (s)",
    )
    return fig


def _correlation_figure_with_peaks(
    state: dict, predictions: list[dict], t_lo: float, t_hi: float,
) -> go.Figure:
    """Per-sign best-R² with peak-status dots, matching the editor's legend."""
    t = np.asarray(state["t"])
    pos_r2 = np.asarray(state["best_pos_r2"])
    neg_r2 = np.asarray(state["best_neg_r2"])
    pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
    neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)
    mask = (t >= t_lo) & (t <= t_hi)

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[mask], y=pos_plot[mask], mode="lines", name="max R² (+)",
        line=dict(color="#2980b9", width=1.2),
    ))
    fig.add_trace(go.Scatter(
        x=t[mask], y=neg_plot[mask], mode="lines", name="max R² (−)",
        line=dict(color="#c0392b", width=1.2),
    ))
    cfg = state.get("config")
    if cfg is not None:
        fig.add_hline(y=cfg.r2_peak_thresh, line_dash="dash",
                      line_color="#888", opacity=0.5)

    # Colored peak dots — mirror of editor._render_signed_r2_panel.
    peaks_pos = find_local_maxima(pos_r2, t, t_lo, t_hi)
    peaks_neg = find_local_maxima(neg_r2, t, t_lo, t_hi)
    for sign, peaks, arr in ((+1, peaks_pos, pos_r2), (-1, peaks_neg, neg_r2)):
        groups: dict[str, tuple[list, list]] = {}
        for i in peaks:
            tag = classify_peak(state, i, sign, predictions)
            groups.setdefault(tag, ([], []))
            groups[tag][0].append(float(t[i]))
            groups[tag][1].append(float(arr[i]))
        for tag, (xs, ys) in groups.items():
            fig.add_trace(go.Scatter(
                x=xs, y=ys, mode="markers",
                name=f"{tag} ({'+' if sign > 0 else '−'})",
                marker=dict(color=PEAK_STATUS_COLORS.get(tag, "#000"),
                            size=9, line=dict(color="#000", width=0.5)),
                hovertemplate="t=%{x:.2f}s<br>R²=%{y:.3f}<br>"
                              f"<b>{tag}</b><extra></extra>",
                showlegend=False,
            ))

    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(title="t (s)", range=[t_lo, t_hi]),
        yaxis=dict(title="R² (per sign)", range=[0, 1.05]),
        hovermode="closest", plot_bgcolor="#fafbfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _signal_with_trapezoid_figure(
    state: dict, prediction: dict | None,
    t_lo: float, t_hi: float, pad_s: float = 4.0,
) -> go.Figure:
    """Zoomed a_vert with the fitted trapezoid overlay (the 'match')."""
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    mask = (t >= t_lo - pad_s) & (t <= t_hi + pad_s)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t[mask], y=a_vert[mask], mode="lines", name="a_vert",
        line=dict(color="#233044", width=1),
    ))
    fig.add_trace(go.Scatter(
        x=t[mask], y=a_smooth[mask], mode="lines", name="smoothed",
        line=dict(color="#e67e22", width=1.4),
    ))
    fig.add_hline(y=0, line_dash="dash", line_color="#bbb", opacity=0.6)
    fig.add_vrect(x0=t_lo, x1=t_hi, fillcolor=SELECTED_COLOR,
                  opacity=0.08, line_width=0)

    if prediction is not None:
        for lobe_key, lobe_color in (("lobe1", "#c0392b"), ("lobe2", "#8e44ad")):
            L = prediction.get(lobe_key) or {}
            try:
                W = float(L["half_width_s"]); f = float(L["frac_flat"])
                A = float(L["a_peak"]); t_c = float(L["t_c"])
            except (KeyError, TypeError, ValueError):
                continue
            tt = np.linspace(t_c - W, t_c + W, 200)
            yy = A * trapezoid_kernel(tt, t_c, W, f)
            fig.add_trace(go.Scatter(
                x=tt, y=yy, mode="lines", name=f"{lobe_key} template",
                line=dict(color=lobe_color, width=2.2),
            ))
            fig.add_trace(go.Scatter(
                x=[t_c], y=[A], mode="markers", showlegend=False,
                marker=dict(color=lobe_color, size=8,
                            line=dict(color="#000", width=0.5)),
            ))

    fig.update_layout(
        height=280, margin=dict(l=10, r=10, t=20, b=30),
        xaxis_title="t (s)", yaxis_title="a (m/s²)",
        hovermode="x unified", plot_bgcolor="#fafbfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def render() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    if loaded is None:
        st.warning("Load a signal first.")
        if st.button("← Back to data"):
            goto(STEP_DATA)
        return

    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 3</span>'
        '<h1>Interactive segmentation</h1>'
        f'<p>Source: {loaded.source} · {loaded.meta.get("samples", "?")} samples · '
        f'{loaded.meta.get("sample_rate", "?")}</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    if st.session_state["detector_state"] is None:
        _run_detector(loaded)
    state = st.session_state["detector_state"]
    if state is None:
        st.error("Detector produced no state — the signal may be too short. "
                 "Go back and load another trace.")
        if st.button("← Back"):
            goto(STEP_DATA)
        return

    predictions = st.session_state["predictions"]
    segments_df = valid_segments(st.session_state["segments_df"])

    sel = st.session_state.get("selected_segment")
    if sel is None or (len(segments_df) and sel >= len(segments_df)):
        sel = 0 if len(segments_df) else None
        st.session_state["selected_segment"] = sel

    # Segment list lives in the sidebar. Main area shows signal + detail.
    render_segment_sidebar(segments_df, sel)

    st.markdown("### Signal + segments")
    st.plotly_chart(
        _main_signal_figure(state, segments_df, sel),
        use_container_width=True, key="seg_main_fig",
    )

    if len(segments_df) == 0 or sel is None:
        st.info("No segments yet. Use the **+ Add** button in the sidebar to "
                "create one, or hit **Reset to detector output** below to "
                "re-seed the list from the detector's proposal.")
    else:
        t_arr = np.asarray(state["t"])
        t_min = float(t_arr[0]); t_max = float(t_arr[-1])

        seg = segments_df.iloc[sel]
        t_lo = float(seg["start_s"]); t_hi = float(seg["end_s"])
        matching = find_matching_prediction(predictions, t_lo, t_hi)

        st.markdown(
            f"#### Detail — segment #{sel}  ({seg['type']},  "
            f"{t_lo:.1f}–{t_hi:.1f}s, duration {t_hi - t_lo:.1f}s)"
        )

        h1, h2 = st.columns(2)
        if matching is not None:
            t_c1 = float(matching["lobe1"]["t_c"])
            t_c2 = float(matching["lobe2"]["t_c"])
            W_star = float(matching["lobe1"]["half_width_s"])
            f_star = float(matching["lobe1"]["frac_flat"])
        else:
            t_c1 = t_lo + 0.25 * (t_hi - t_lo)
            t_c2 = t_lo + 0.75 * (t_hi - t_lo)
            W_star = None; f_star = None
        i1 = int(np.clip(np.argmin(np.abs(t_arr - t_c1)), 0, t_arr.size - 1))
        i2 = int(np.clip(np.argmin(np.abs(t_arr - t_c2)), 0, t_arr.size - 1))
        with h1:
            st.plotly_chart(
                _heatmap_figure(state, i1, f"lobe1 @ t={t_arr[i1]:.1f}s",
                                W_star, f_star),
                use_container_width=True, key=f"heat1_{sel}",
            )
        with h2:
            st.plotly_chart(
                _heatmap_figure(state, i2, f"lobe2 @ t={t_arr[i2]:.1f}s",
                                W_star, f_star),
                use_container_width=True, key=f"heat2_{sel}",
            )

        st.markdown("**Correlation score with peak status**")
        st.markdown(peak_status_legend_html(), unsafe_allow_html=True)
        pad = max(3.0, 0.4 * (t_hi - t_lo))
        st.plotly_chart(
            _correlation_figure_with_peaks(
                state, predictions, max(t_min, t_lo - pad),
                min(t_max, t_hi + pad),
            ),
            use_container_width=True, key=f"corr_{sel}",
        )

        st.markdown("**Signal with fitted trapezoid**")
        st.plotly_chart(
            _signal_with_trapezoid_figure(state, matching, t_lo, t_hi, pad_s=4.0),
            use_container_width=True, key=f"trap_{sel}",
        )

        st.markdown("**Fitted trapezoid parameters**")
        render_trapezoid_params(matching)

    # Spreadsheet-style fallback — the sidebar list covers most flows,
    # but power users can still bulk-edit from this table.
    with st.expander("Spreadsheet editor (advanced)", expanded=False):
        st.caption(
            "All segments in a grid. Edit any cell, add rows at the "
            "bottom, or tick the left-edge checkboxes to delete rows in "
            "bulk. Changes here are committed as soon as you click "
            "outside the edited cell."
        )
        edited = st.data_editor(
            segments_df,
            num_rows="dynamic",
            use_container_width=True, height=240,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "type", options=["up", "down"], required=True,
                ),
                "start_s": st.column_config.NumberColumn(
                    "start (s)", min_value=0.0, step=0.1, format="%.2f",
                ),
                "end_s": st.column_config.NumberColumn(
                    "end (s)", min_value=0.0, step=0.1, format="%.2f",
                ),
                "joint_r2": st.column_config.NumberColumn(
                    "joint R² (detector)", disabled=True, format="%.3f",
                ),
            },
            key="seg_editor_table",
        )
        if not edited.equals(segments_df):
            st.session_state["segments_df"] = edited.reset_index(drop=True)
            st.rerun()

    st.divider()
    c1, c2, c3 = st.columns([1, 1.1, 1])
    with c1:
        if st.button("← Back"):
            goto(STEP_DATA)
    with c2:
        if st.button("Reset to detector output",
                     help="Discards all manual edits and re-runs the detector "
                          "on this signal. Config is unchanged, so the result "
                          "matches the initial proposal."):
            _run_detector(loaded)
            st.rerun()
    with c3:
        valid = valid_segments(st.session_state["segments_df"])
        if st.button(f"Predict Δh → ({len(valid)} segments)",
                     type="primary", disabled=len(valid) == 0):
            st.session_state["prediction_rows"] = None
            st.session_state["prediction_rows_by_algo"] = None
            goto(STEP_PREDICT)
