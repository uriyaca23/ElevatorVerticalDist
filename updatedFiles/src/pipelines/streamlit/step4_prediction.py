"""Step 4 — Per-segment Δh prediction.

Runs every accelerometer-only algorithm in :data:`ACCEL_ALGOS` over the
finalised segment list and renders the bar chart, per-algo metrics, and
a wide comparison table side-by-side. The trapezoid-pulse-pair algorithm
is the "primary" — its rows feed the sidebar list and the PDF report so
existing downstream paths keep working unchanged.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from ui import api_client

from .common import (
    ACCEL_ALGOS,
    HOVER_DATETIME_FMT,
    LoadedSignal,
    PRIMARY_ALGO_ID,
    RIDE_COLORS,
    SELECTED_COLOR,
    STEP_REPORT,
    STEP_SEGMENT,
    find_matching_prediction,
    goto,
    render_predict_segment_sidebar,
    render_trapezoid_params,
    to_datetime,
    to_datetime_array,
    valid_segments,
)
from .step3_segmentation import _add_gap_overlays


# Algorithm short ids the diagnostic charts know how to render.
_TRAP_ALGO_ID = "trap"
_ZUPT_ALGO_ID = "zupt"


def _segment_inside_valid_interval(
    seg_start_s: float, seg_end_s: float,
    valid_intervals: list[tuple[int, int]] | None,
    t0_ms: float | None,
) -> bool:
    """True iff [seg_start_s, seg_end_s] lies entirely inside one valid
    interval. With no intervals defined (legacy / clean signal), returns
    True so existing flows are unaffected."""
    if not valid_intervals or t0_ms is None:
        return True
    seg_lo_ms = float(t0_ms) + seg_start_s * 1000.0
    seg_hi_ms = float(t0_ms) + seg_end_s * 1000.0
    for s_ms, e_ms in valid_intervals:
        if s_ms <= seg_lo_ms and seg_hi_ms <= e_ms:
            return True
    return False


def _run_predictions(
    loaded: LoadedSignal, segments: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Run every accelerometer-only algorithm via the prediction API and
    return one row-list per algorithm, keyed by short id.

    The /predict endpoint owns the per-segment slicing and the gravity
    pre/post window logic — the UI just hands over the full session ACC
    and the finalised segment list. Defensive: any segment whose span
    crosses a "no data" gap is dropped before the predict call (the user
    is shown the red gap overlays in step 3 and should not have placed
    such segments, but we filter here too).
    """
    t0_ms = (
        float(loaded.acc["timestamp_ms"].iloc[0])
        if not loaded.acc.empty else None
    )
    seg_dicts: list[dict] = []
    skipped = 0
    for _, row in segments.iterrows():
        s = float(row["start_s"]); e = float(row["end_s"])
        if not _segment_inside_valid_interval(s, e, loaded.valid_intervals, t0_ms):
            skipped += 1
            continue
        seg_dicts.append({
            "type":    str(row["type"]).lower(),
            "start_s": s,
            "end_s":   e,
        })
    if skipped:
        st.warning(
            f"Skipped {skipped} segment(s) that overlap a no-data gap. "
            "Move them inside a valid interval (or delete them) on the "
            "previous step."
        )
    rows_by_algo, _primary = api_client.predict(loaded.acc, seg_dicts)
    return rows_by_algo


def _prediction_main_figure(
    state: dict, rows: list[dict], selected: int | None,
    valid_intervals: list[tuple[int, int]] | None = None,
) -> go.Figure:
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    t0_ms = state.get("t0_ms")
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=to_datetime_array(t, t0_ms), y=a_vert,
        mode="lines", name="a_vert",
        line=dict(color="#233044", width=1),
        hovertemplate=f"%{{x|{HOVER_DATETIME_FMT}}}<br>"
                      "a=%{y:.2f} m/s²<extra></extra>",
    ))
    if t.size:
        _add_gap_overlays(fig, valid_intervals, t0_ms, float(t[0]), float(t[-1]))
    for r in rows:
        s = float(r["start_s"]); e = float(r["end_s"])
        rt = r["type"]
        is_sel = (selected is not None and int(r["segment"]) == selected)
        base = RIDE_COLORS.get(rt, "#777")
        fig.add_vrect(
            x0=to_datetime(s, t0_ms), x1=to_datetime(e, t0_ms),
            fillcolor=SELECTED_COLOR if is_sel else base,
            opacity=0.35 if is_sel else 0.15,
            line_width=2 if is_sel else 0,
            line_color=SELECTED_COLOR if is_sel else base,
            annotation_text=f"#{r['segment']} Δh={r['delta_height_m']:+.1f}m"
                            if np.isfinite(r["delta_height_m"]) else f"#{r['segment']}",
            annotation_position="top left",
            annotation_font_color=SELECTED_COLOR if is_sel else base,
            annotation_font_size=10,
        )
    fig.update_layout(
        height=330, margin=dict(l=10, r=10, t=20, b=30),
        xaxis=dict(title="time", type="date"),
        yaxis_title="a_vert (m/s²)",
        hovermode="x unified", plot_bgcolor="#fafbfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )

    # Zoom to the currently-selected segment so the ride is centred.
    # Double-click on the chart reverts to the full range.
    if selected is not None:
        sel_row = next((r for r in rows if int(r["segment"]) == selected), None)
        if sel_row is not None:
            s_sel = float(sel_row["start_s"]); e_sel = float(sel_row["end_s"])
            if np.isfinite(s_sel) and np.isfinite(e_sel) and e_sel > s_sel:
                pad = max(5.0, 0.5 * (e_sel - s_sel))
                x_lo = s_sel - pad; x_hi = e_sel + pad
                mask = (t >= x_lo) & (t <= x_hi)
                if mask.any():
                    local_lo = float(np.nanmin(a_vert[mask]))
                    local_hi = float(np.nanmax(a_vert[mask]))
                    y_pad = max(0.5, 0.10 * (local_hi - local_lo))
                    fig.update_layout(
                        xaxis=dict(title="time", type="date",
                                   range=[to_datetime(x_lo, t0_ms),
                                          to_datetime(x_hi, t0_ms)]),
                        yaxis=dict(range=[local_lo - y_pad, local_hi + y_pad]),
                    )
    return fig


def _prediction_bar_figure(
    rows_by_algo: dict[str, list[dict]], selected: int | None,
    t0_ms: float | None,
) -> go.Figure:
    """Grouped bar chart — one bar per (segment, algorithm) pair.

    Bars are labelled by the segment's start time (``HH:MM:SS``) instead
    of the segment number, so the x-axis matches the time-series charts
    above. The selected segment is emphasised by dimming the opacity of
    the others rather than recolouring, so the per-algorithm colour
    legend stays meaningful.
    """
    fig = go.Figure()
    primary_id = ACCEL_ALGOS[0][0]
    rows_template = rows_by_algo.get(primary_id, [])
    seg_ids = [int(r["segment"]) for r in rows_template]
    starts_dt = [to_datetime(float(r["start_s"]), t0_ms) for r in rows_template]
    ends_dt = [to_datetime(float(r["end_s"]), t0_ms) for r in rows_template]
    # Categorical x labels so plotly groups bars cleanly per segment;
    # the time stays first-class via customdata for the hover.
    xs = [dt.strftime("%H:%M:%S") for dt in starts_dt]
    customdata = [
        [seg_ids[i],
         starts_dt[i].strftime("%Y-%m-%d %H:%M:%S"),
         ends_dt[i].strftime("%H:%M:%S")]
        for i in range(len(rows_template))
    ]

    for algo_id, label, color in ACCEL_ALGOS:
        rows = rows_by_algo.get(algo_id, [])
        if not rows:
            continue
        ys = [0.0 if not np.isfinite(r["delta_height_m"])
              else float(r["delta_height_m"]) for r in rows]
        opacities = [
            1.0 if (selected is None or sid == selected) else 0.35
            for sid in seg_ids
        ]
        fig.add_trace(go.Bar(
            x=xs, y=ys, name=label, customdata=customdata,
            marker_color=color, marker_opacity=opacities,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "seg #%{customdata[0]}<br>"
                "%{customdata[1]} → %{customdata[2]}<br>"
                "Δh=%{y:+.2f} m<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=30),
        xaxis_title="segment start time",
        yaxis_title="Δh (m)", plot_bgcolor="#fafbfc",
        barmode="group", bargap=0.25, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.add_hline(y=0, line_color="#333", line_width=0.5)
    return fig


def _trapezoid_fit_figure(
    meta: dict, start_s: float, t0_ms: float | None,
) -> go.Figure | None:
    """Plotly version of editor.py's "trapezoid fit on accel signal" panel.

    Returns ``None`` when the algorithm did not return a fitted template
    (e.g. the segment was rejected before the fit stage). The x-axis is
    the same canonical wall-clock datetime axis the top plot uses, so a
    user comparing the two charts sees the same data at the same time.
    """
    t_sec = meta.get("t_sec")
    a_smooth = meta.get("a_smooth")
    a_template = meta.get("a_template")
    if t_sec is None or a_smooth is None or a_template is None:
        return None
    t = np.asarray(t_sec, dtype=float)
    s = np.asarray(a_smooth, dtype=float)
    tpl = np.asarray(a_template, dtype=float)
    if t.size == 0:
        return None
    # t_sec is ride-local (0 at first sample of the slice); shift by
    # the segment's canonical start_s so it lines up with the top plot.
    t_dt = to_datetime_array(t + float(start_s), t0_ms)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t_dt, y=s, mode="lines", name="a_smooth",
        line=dict(color="#2c3e50", width=1.0),
        hovertemplate=f"%{{x|{HOVER_DATETIME_FMT}}}<br>"
                      "a=%{y:.2f} m/s²<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=t_dt, y=tpl, mode="lines", name="trapezoid template",
        line=dict(color="#c0392b", width=2.0),
        hovertemplate=f"%{{x|{HOVER_DATETIME_FMT}}}<br>"
                      "tpl=%{y:.2f} m/s²<extra></extra>",
    ))
    params = meta.get("params") or {}
    for key in ("t_c1", "t_c2"):
        v = params.get(key)
        if v is not None and np.isfinite(v):
            fig.add_vline(x=to_datetime(float(v) + float(start_s), t0_ms),
                          line_color="#c0392b",
                          line_width=1, line_dash="dot", opacity=0.7)
    fig.add_hline(y=0.0, line_color="#888", line_width=0.5, line_dash="dash")
    if params:
        sign = int(params.get("sign", 0))
        annotation = (
            f"A_used={params.get('A_used', float('nan')):.2f} m/s² · "
            f"W={params.get('W', float('nan')):.2f}s · "
            f"f={params.get('f', float('nan')):.2f} · "
            f"sign={sign:+d}<br>"
            f"t_c1={params.get('t_c1', float('nan')):.2f}s · "
            f"t_c2={params.get('t_c2', float('nan')):.2f}s · "
            f"R²={params.get('joint_r2', float('nan')):.3f} · "
            f"v_peak={params.get('v_peak_measured', float('nan')):+.2f} m/s"
        )
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            text=annotation, showarrow=False,
            font=dict(size=10, family="monospace", color="#1a2436"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#888", borderwidth=0.5, borderpad=4,
        )
    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=30, b=30),
        title=dict(text="Trapezoid fit on accelerometer signal",
                   x=0.0, font=dict(size=12)),
        xaxis=dict(title="time", type="date"),
        yaxis_title="a (m/s²)",
        plot_bgcolor="#fafbfc", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _zupt_position_figure(meta: dict) -> go.Figure | None:
    """Plotly version of editor.py's "ZUPT integrated position" panel."""
    pos = meta.get("pos_curve")
    if pos is None:
        return None
    pos_arr = np.asarray(pos, dtype=float)
    if pos_arr.size == 0:
        return None
    idx = np.arange(pos_arr.size)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=idx, y=pos_arr, mode="lines", name="pos(t)",
        line=dict(color="#27ae60", width=1.4),
        hovertemplate="i=%{x}<br>pos=%{y:.2f} m<extra></extra>",
    ))
    start = meta.get("start_idx")
    end = meta.get("end_idx")
    if (start is not None and end is not None
            and np.isfinite(start) and np.isfinite(end)
            and int(end) > int(start)):
        fig.add_vrect(
            x0=int(start), x1=int(end),
            fillcolor="#27ae60", opacity=0.15, line_width=0,
            annotation_text="motion window", annotation_position="top left",
            annotation_font_color="#1e7a3a", annotation_font_size=10,
        )
    fig.add_hline(y=0.0, line_color="#888", line_width=0.5, line_dash="dash")
    info_bits: list[str] = []
    n_active = meta.get("n_active")
    active_frac = meta.get("active_fraction")
    method = meta.get("method", "")
    if n_active is not None:
        info_bits.append(f"n_active={int(n_active)}")
    if active_frac is not None and np.isfinite(active_frac):
        info_bits.append(f"active_frac={float(active_frac):.2f}")
    if method:
        info_bits.append(f"method={method}")
    if info_bits:
        fig.add_annotation(
            xref="paper", yref="paper", x=0.01, y=0.99,
            xanchor="left", yanchor="top",
            text=" · ".join(info_bits), showarrow=False,
            font=dict(size=10, family="monospace", color="#1a2436"),
            bgcolor="rgba(255,255,255,0.85)",
            bordercolor="#888", borderwidth=0.5, borderpad=4,
        )
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=30),
        title=dict(text="ZUPT integrated position",
                   x=0.0, font=dict(size=12)),
        xaxis_title="sample index", yaxis_title="pos (m)",
        plot_bgcolor="#fafbfc", hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _build_comparison_table(rows_by_algo: dict[str, list[dict]]) -> pd.DataFrame:
    """Wide-format comparison table: one row per segment, one column
    block per algorithm. The shared columns (segment, type, start/end,
    duration) come from the primary algorithm's row list.
    """
    primary = rows_by_algo.get(PRIMARY_ALGO_ID, [])
    if not primary:
        return pd.DataFrame()
    out: list[dict] = []
    for i, base in enumerate(primary):
        rec = {
            "segment":    int(base["segment"]),
            "type":       base["type"],
            "start_s":    float(base["start_s"]),
            "end_s":      float(base["end_s"]),
            "duration_s": float(base["duration_s"]),
        }
        for algo_id, _label, _color in ACCEL_ALGOS:
            rows_a = rows_by_algo.get(algo_id, [])
            r = rows_a[i] if i < len(rows_a) else None
            if r is None:
                rec[f"{algo_id}_dh"]      = float("nan")
                rec[f"{algo_id}_ci"]      = float("nan")
                rec[f"{algo_id}_quality"] = float("nan")
                rec[f"{algo_id}_accepted"] = False
                rec[f"{algo_id}_reject"]   = ""
            else:
                rec[f"{algo_id}_dh"]      = r["delta_height_m"]
                rec[f"{algo_id}_ci"]      = r["ci_half_width"]
                rec[f"{algo_id}_quality"] = r["quality_score"]
                rec[f"{algo_id}_accepted"] = bool(r["accepted"])
                rec[f"{algo_id}_reject"]   = r.get("reject_reason", "")
        out.append(rec)
    return pd.DataFrame(out)


def render() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    segments: pd.DataFrame | None = st.session_state["segments_df"]
    state = st.session_state.get("detector_state")
    if loaded is None or segments is None or state is None:
        st.warning("Complete earlier steps first.")
        if st.button("← Back"):
            goto(STEP_SEGMENT)
        return

    algo_pill_html = " ".join(
        f'<span style="display:inline-block;padding:0.1rem 0.55rem;'
        f'border-radius:999px;background:rgba(255,255,255,0.18);'
        f'border:1px solid rgba(255,255,255,0.35);margin-right:0.35rem;'
        f'font-size:0.74rem;font-weight:600;">'
        f'<span style="display:inline-block;width:0.55rem;height:0.55rem;'
        f'border-radius:50%;background:{color};margin-right:0.35rem;'
        f'vertical-align:middle;"></span>{label}</span>'
        for _aid, label, color in ACCEL_ALGOS
    )
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 4</span>'
        '<h1>Per-segment height predictions</h1>'
        '<p>Both accelerometer-only algorithms run on every segment. '
        'Bars and the table show them side-by-side; the primary '
        '(<b>Trapezoid pulse-pair</b>) drives the sidebar list and the '
        'PDF report.</p>'
        f'<div style="margin-top:0.55rem;">{algo_pill_html}</div>'
        '</div>',
        unsafe_allow_html=True,
    )

    valid = valid_segments(segments)
    if st.session_state["prediction_rows_by_algo"] is None:
        with st.spinner(
            f"Running {len(ACCEL_ALGOS)} algorithms on {len(valid)} segments…"
        ):
            by_algo = _run_predictions(loaded, valid)
            st.session_state["prediction_rows_by_algo"] = by_algo
            st.session_state["prediction_rows"] = by_algo.get(PRIMARY_ALGO_ID, [])
    rows_by_algo: dict[str, list[dict]] = (
        st.session_state["prediction_rows_by_algo"] or {}
    )
    rows: list[dict] = st.session_state["prediction_rows"] or []
    if not rows:
        st.info("No predictable segments.")
        if st.button("← Back"):
            goto(STEP_SEGMENT)
        return

    valid_segment_ids = [int(r["segment"]) for r in rows]
    if st.session_state["predict_selected"] not in valid_segment_ids:
        st.session_state["predict_selected"] = valid_segment_ids[0]
    selected: int = int(st.session_state["predict_selected"])

    # Sidebar list (read-only — predictions don't mutate segments).
    render_predict_segment_sidebar(rows, selected)

    st.plotly_chart(
        _prediction_main_figure(
            state, rows, selected, valid_intervals=loaded.valid_intervals,
        ),
        use_container_width=True, key="pred_main_fig",
    )
    st.plotly_chart(
        _prediction_bar_figure(rows_by_algo, selected, state.get("t0_ms")),
        use_container_width=True, key="pred_bar_fig",
    )

    # Per-algorithm metrics for the selected segment.
    st.markdown(f"#### Detail — segment #{selected}")
    metric_cols = st.columns(len(ACCEL_ALGOS))
    for col, (algo_id, label, color) in zip(metric_cols, ACCEL_ALGOS):
        rows_a = rows_by_algo.get(algo_id, [])
        sel = next((r for r in rows_a if int(r["segment"]) == selected), None)
        with col:
            st.markdown(
                f'<div style="font-weight:600;color:{color};'
                f'font-size:0.85rem;margin-bottom:0.25rem;">'
                f'<span style="display:inline-block;width:0.55rem;'
                f'height:0.55rem;border-radius:50%;background:{color};'
                f'margin-right:0.35rem;"></span>{label}</div>',
                unsafe_allow_html=True,
            )
            if sel is None:
                st.caption("no result")
                continue
            mc = st.columns(4)
            mc[0].metric(
                "Δh",
                f"{sel['delta_height_m']:+.2f} m"
                if np.isfinite(sel['delta_height_m']) else "—",
            )
            mc[1].metric(
                "±CI 90%",
                f"{sel['ci_half_width']:.2f} m"
                if np.isfinite(sel['ci_half_width']) else "—",
            )
            mc[2].metric(
                "Quality",
                f"{sel['quality_score']:.1f}"
                if np.isfinite(sel['quality_score']) else "—",
            )
            mc[3].metric(
                "Accepted", "yes" if sel["accepted"] else "no",
            )
            if sel.get("reject_reason"):
                st.caption(f"reject_reason: `{sel['reject_reason']}`")

    # Per-algorithm diagnostic charts for the selected segment:
    # the trapezoid template overlay (trap algo) and the ZUPT integrated
    # position trace (zupt algo). Mirrors the editor's Prediction tab.
    trap_rows = rows_by_algo.get(_TRAP_ALGO_ID, [])
    zupt_rows = rows_by_algo.get(_ZUPT_ALGO_ID, [])
    trap_sel = next(
        (r for r in trap_rows if int(r["segment"]) == selected), None,
    )
    zupt_sel = next(
        (r for r in zupt_rows if int(r["segment"]) == selected), None,
    )
    diag_cols = st.columns(2)
    with diag_cols[0]:
        if trap_sel:
            trap_fig = _trapezoid_fit_figure(
                (trap_sel.get("meta") or {}),
                start_s=float(trap_sel["start_s"]),
                t0_ms=state.get("t0_ms"),
            )
        else:
            trap_fig = None
        if trap_fig is None:
            st.caption("No trapezoid template for this segment.")
        else:
            st.plotly_chart(
                trap_fig, use_container_width=True,
                key=f"pred_trap_fig_{selected}",
            )
    with diag_cols[1]:
        zupt_fig = _zupt_position_figure(
            (zupt_sel or {}).get("meta") or {}
        ) if zupt_sel else None
        if zupt_fig is None:
            st.caption("No ZUPT trajectory for this segment.")
        else:
            st.plotly_chart(
                zupt_fig, use_container_width=True,
                key=f"pred_zupt_fig_{selected}",
            )

    # Trapezoid template parameters for the selected segment (from the
    # segmentation-step fits — same as before).
    sel_primary = next((r for r in rows if int(r["segment"]) == selected), None)
    if sel_primary is not None:
        predictions = st.session_state.get("predictions") or []
        matching = find_matching_prediction(
            predictions,
            float(sel_primary["start_s"]), float(sel_primary["end_s"]),
        )
        st.markdown("**Fitted trapezoid parameters (from segmentation)**")
        render_trapezoid_params(matching)

    st.markdown("### All segments — both algorithms")
    df = _build_comparison_table(rows_by_algo)
    if not df.empty:
        cfg: dict = {
            "start_s":    st.column_config.NumberColumn("start (s)",    format="%.1f"),
            "end_s":      st.column_config.NumberColumn("end (s)",      format="%.1f"),
            "duration_s": st.column_config.NumberColumn("duration (s)", format="%.1f"),
        }
        for algo_id, label, _color in ACCEL_ALGOS:
            cfg[f"{algo_id}_dh"]      = st.column_config.NumberColumn(
                f"{label} · Δh (m)",        format="%+.2f")
            cfg[f"{algo_id}_ci"]      = st.column_config.NumberColumn(
                f"{label} · ±CI 90% (m)",   format="%.2f")
            cfg[f"{algo_id}_quality"] = st.column_config.NumberColumn(
                f"{label} · quality",       format="%.1f")
        st.dataframe(df, use_container_width=True, height=300, column_config=cfg)

    st.divider()
    c1, _, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Back to segmentation"):
            st.session_state["prediction_rows"] = None
            st.session_state["prediction_rows_by_algo"] = None
            goto(STEP_SEGMENT)
    with c3:
        if st.button("Generate report →", type="primary"):
            goto(STEP_REPORT)
