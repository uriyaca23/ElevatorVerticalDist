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
    valid_segments,
)


def _run_predictions(
    loaded: LoadedSignal, segments: pd.DataFrame,
) -> dict[str, list[dict]]:
    """Run every accelerometer-only algorithm via the prediction API and
    return one row-list per algorithm, keyed by short id.

    The /predict endpoint owns the per-segment slicing and the gravity
    pre/post window logic — the UI just hands over the full session ACC
    and the finalised segment list.
    """
    seg_dicts = [
        {"type":    str(row["type"]).lower(),
         "start_s": float(row["start_s"]),
         "end_s":   float(row["end_s"])}
        for _, row in segments.iterrows()
    ]
    rows_by_algo, _primary = api_client.predict(loaded.acc, seg_dicts)
    return rows_by_algo


def _prediction_main_figure(
    state: dict, rows: list[dict], selected: int | None,
) -> go.Figure:
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=t, y=a_vert, mode="lines", name="a_vert",
        line=dict(color="#233044", width=1),
    ))
    for r in rows:
        s = float(r["start_s"]); e = float(r["end_s"])
        rt = r["type"]
        is_sel = (selected is not None and int(r["segment"]) == selected)
        base = RIDE_COLORS.get(rt, "#777")
        fig.add_vrect(
            x0=s, x1=e,
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
        xaxis_title="time (s)", yaxis_title="a_vert (m/s²)",
        hovermode="x unified", plot_bgcolor="#fafbfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _prediction_bar_figure(
    rows_by_algo: dict[str, list[dict]], selected: int | None,
) -> go.Figure:
    """Grouped bar chart — one bar per (segment, algorithm) pair.

    The selected segment is emphasised by dimming the opacity of the
    other segments rather than recolouring, so the per-algorithm colour
    legend stays meaningful.
    """
    fig = go.Figure()
    primary_id = ACCEL_ALGOS[0][0]
    rows_template = rows_by_algo.get(primary_id, [])
    xs = [f"#{r['segment']}" for r in rows_template]
    seg_ids = [int(r["segment"]) for r in rows_template]

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
            x=xs, y=ys, name=label,
            marker_color=color, marker_opacity=opacities,
            hovertemplate=(
                f"<b>{label}</b><br>"
                "seg %{x}<br>Δh=%{y:+.2f} m<extra></extra>"
            ),
        ))

    fig.update_layout(
        height=300, margin=dict(l=10, r=10, t=10, b=30),
        yaxis_title="Δh (m)", plot_bgcolor="#fafbfc",
        barmode="group", bargap=0.25, bargroupgap=0.08,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    fig.add_hline(y=0, line_color="#333", line_width=0.5)
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
        _prediction_main_figure(state, rows, selected),
        use_container_width=True, key="pred_main_fig",
    )
    st.plotly_chart(
        _prediction_bar_figure(rows_by_algo, selected),
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
