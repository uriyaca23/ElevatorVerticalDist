"""Step 3 — Interactive segmentation.

Runs the trapezoid-template detector on the loaded signal (via the
``pyramidElevatorDist`` package) and shows an editor-style detail panel:
the reconstructed vertical-acceleration overview with the detected
segments overlaid, per-lobe R² heatmaps, the correlation panel, and a
card summarising the fitted trapezoid parameters.

All segmentation / signal computation comes exclusively from the package;
this module is presentation-only.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from pyramidElevatorDist import (
    RECONSTRUCT_CHOICES,
    CorrelationCurves,
    RideSegment,
    displaySeries,
    findSegmentParameters,
    findSegments,
    reconstructedSignal,
)

from .common import (
    LoadedSignal,
    RIDE_COLORS,
    SELECTED_COLOR,
    STEP_DATA,
    STEP_PREDICT,
    _current_t0_ms,
    goto,
    hover_time_template,
    render_segment_sidebar,
    render_trapezoid_params,
    seconds_to_clock,
    time_customdata,
    to_datetime,
    to_datetime_array,
    valid_segments,
)


def _seg_key(t_lo: float, t_hi: float, ride_type: str) -> tuple:
    """Bounds signature used to cache ``findSegmentParameters`` results.

    Keyed by the rounded window + ride type (rather than list position) so
    the cache stays correct across segment edits, adds and deletes — a
    stale entry simply never matches and is recomputed.
    """
    return (round(float(t_lo), 3), round(float(t_hi), 3), str(ride_type).lower())


def _shift_prediction(p: RideSegment, shift_s: float) -> RideSegment:
    """Return a copy of ``p`` with every time-domain field offset by
    ``shift_s`` seconds. Per-part calls return predictions in chunk-local
    seconds; downstream code wants canonical full-signal seconds, and we
    must shift *all* time fields — including ``lobe{1,2}.t_c`` —
    otherwise the trapezoid template overlay renders at the wrong
    horizontal position (a previous bug surfaced as a tiny trapezoid
    parked at the start of the recording while the actual signal slice
    was minutes later).
    """
    return p.model_copy(update={
        "t_start_s": p.t_start_s + shift_s,
        "t_end_s":   p.t_end_s + shift_s,
        "lobe1": p.lobe1.model_copy(update={"t_c": p.lobe1.t_c + shift_s}),
        "lobe2": p.lobe2.model_copy(update={"t_c": p.lobe2.t_c + shift_s}),
    })


def _run_detector(loaded: LoadedSignal) -> None:
    """Detect ride segments once per gap-free part, via ``findSegments``.

    The pipeline splits the raw signal into ``loaded.acc_parts`` — one
    DataFrame per valid interval — at load time. We run the detector on
    each part in isolation, then concatenate predictions onto a single
    canonical time axis. This keeps the detector blind to the gaps (no
    false matches that span a dropout) and keeps every downstream plot
    in absolute wall-clock time.

    Segmentation matches on the rotation-invariant ``|a|-g`` residual, so
    it is magnitude-invariant and unaffected by the gyro reconstruction
    method — the detector is called acc-only.
    """
    parts = loaded.acc_parts or ([loaded.acc] if not loaded.acc.empty else [])
    if not parts:
        st.session_state["predictions"] = []
        st.session_state["segments_df"] = pd.DataFrame(
            columns=["type", "start_s", "end_s", "joint_r2"],
        )
        st.session_state["selected_segment"] = None
        st.session_state["lobe_overrides"] = {}
        st.session_state["segment_params"] = {}
        return

    canonical_t0_ms = float(loaded.acc["timestamp_ms"].iloc[0])
    preds: list[RideSegment] = []
    with st.spinner(
        f"Running trapezoid-template detector on {len(parts)} part(s)…"
    ):
        # Per-part detection — one call per gap-free DataFrame. The trace is
        # already 50 Hz-resampled at load, so resample=False (passing the
        # default would double-resample).
        for i, part in enumerate(parts):
            if part is None or len(part) < 2:
                continue
            try:
                part_preds = findSegments(part, resample=False)
            except Exception as e:  # noqa: BLE001
                st.warning(
                    f"Detection failed on part #{i} "
                    f"({type(e).__name__}: {e}); skipping."
                )
                continue
            shift_s = (
                float(part["timestamp_ms"].iloc[0]) - canonical_t0_ms
            ) / 1000.0
            for p in part_preds:
                preds.append(_shift_prediction(p, shift_s))

    st.session_state["predictions"] = preds
    rows = [
        {"type":     p.ride_type,
         "start_s":  round(float(p.t_start_s), 2),
         "end_s":    round(float(p.t_end_s), 2),
         "joint_r2": round(float(p.joint_r2_mean), 3)}
        for p in preds
    ]
    st.session_state["segments_df"] = pd.DataFrame(
        rows, columns=["type", "start_s", "end_s", "joint_r2"],
    )
    st.session_state["selected_segment"] = 0 if rows else None
    # Detector re-run invalidates any manual trapezoid overrides and the
    # per-segment parameter cache — the segment layout was rebuilt.
    st.session_state["lobe_overrides"] = {}
    st.session_state["segment_params"] = {}


def _gap_spans_seconds(
    valid_intervals: list[tuple[int, int]] | None,
    t0_ms: float | None,
    t_lo: float, t_hi: float,
) -> list[tuple[float, float]]:
    """Return the complement of valid_intervals over [t_lo, t_hi] in seconds.

    Inputs are absolute Unix-ms intervals; outputs are relative seconds
    on the same time axis the figure uses. Empty / missing inputs → empty
    list (caller skips rendering the red overlays).
    """
    if not valid_intervals or t0_ms is None:
        return []
    epoch = float(t0_ms)
    valid_s = sorted(
        ((float(a) - epoch) / 1000.0, (float(b) - epoch) / 1000.0)
        for a, b in valid_intervals
    )
    gaps: list[tuple[float, float]] = []
    cursor = float(t_lo)
    for s, e in valid_s:
        if e <= cursor:
            continue
        if s > cursor:
            gaps.append((cursor, min(s, t_hi)))
        cursor = max(cursor, e)
        if cursor >= t_hi:
            break
    if cursor < t_hi:
        gaps.append((cursor, t_hi))
    return [(a, b) for a, b in gaps if b > a]


def _add_gap_overlays(
    fig: go.Figure,
    valid_intervals: list[tuple[int, int]] | None,
    t0_ms: float | None,
    t_lo: float, t_hi: float,
) -> None:
    """Shade gap regions in red and stamp a ✕ on each."""
    gaps = _gap_spans_seconds(valid_intervals, t0_ms, t_lo, t_hi)
    if not gaps:
        return
    for s, e in gaps:
        fig.add_vrect(
            x0=to_datetime(s, t0_ms), x1=to_datetime(e, t0_ms),
            fillcolor="#3498db", opacity=0.28, line_width=0,
            annotation_text="✕ no data",
            annotation_position="top right",
            annotation_font_color="#1b5b8e",
            annotation_font_size=11,
            layer="below",
        )


def _main_signal_figure(
    disp: pd.DataFrame, t0_ms: float,
    segments_df: pd.DataFrame, selected_idx: int | None,
    valid_intervals: list[tuple[int, int]] | None = None,
    has_gyro: bool = False, method: str = "none",
) -> go.Figure:
    """Overview: reconstructed a_z with the segments overlaid.

    Shows the gyro-reconstructed vertical acceleration when a gyro is
    present, else falls back to ``|a| − g`` (see :func:`displaySeries`).
    """
    ts = np.asarray(disp["timestamp_ms"], dtype=float)
    a_vert, y_label, _ = displaySeries(disp, has_gyro, method)
    t = (ts - t0_ms) / 1000.0 if ts.size else ts
    dt = to_datetime_array(t, t0_ms)

    cd_full = time_customdata(t)
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dt, y=a_vert, mode="lines", name=y_label,
        line=dict(color="#233044", width=1),
        customdata=cd_full,
        hovertemplate=hover_time_template("a=%{y:.2f} m/s²"),
    ))
    if t.size:
        _add_gap_overlays(fig, valid_intervals, t0_ms, float(t[0]), float(t[-1]))
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
            x0=to_datetime(s, t0_ms), x1=to_datetime(e, t0_ms),
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
        xaxis_title="time", yaxis_title=f"{y_label} (m/s²)",
        xaxis=dict(type="date"),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
        plot_bgcolor="#fafbfc",
    )

    # Zoom the initial view to the currently-selected segment so the
    # ride is centred. The user can double-click the plot to revert to
    # the full-signal range at any time.
    if selected_idx is not None and 0 <= selected_idx < len(segments_df):
        try:
            seg_sel = segments_df.iloc[selected_idx]
            s_sel = float(seg_sel["start_s"]); e_sel = float(seg_sel["end_s"])
        except (TypeError, ValueError, IndexError):
            s_sel = e_sel = float("nan")
        if np.isfinite(s_sel) and np.isfinite(e_sel) and e_sel > s_sel:
            pad = max(5.0, 0.5 * (e_sel - s_sel))
            x_lo = s_sel - pad; x_hi = e_sel + pad
            mask = (t >= x_lo) & (t <= x_hi)
            if mask.any():
                local_lo = float(np.nanmin(a_vert[mask]))
                local_hi = float(np.nanmax(a_vert[mask]))
                y_pad = max(0.5, 0.10 * (local_hi - local_lo))
                fig.update_layout(
                    xaxis=dict(type="date",
                               range=[to_datetime(x_lo, t0_ms),
                                      to_datetime(x_hi, t0_ms)]),
                    yaxis=dict(range=[local_lo - y_pad, local_hi + y_pad]),
                )
    return fig


def _heatmap_figure(heat, grid_w_s, grid_f, title: str,
                    mark_W: float | None = None,
                    mark_f: float | None = None) -> go.Figure:
    """Per-lobe R² heatmap over the (W, f) template grid.

    ``heat`` / ``grid_w_s`` / ``grid_f`` come straight from
    ``findSegmentParameters(...).heatmaps``.
    """
    fig = go.Figure(go.Heatmap(
        z=np.asarray(heat), x=np.asarray(grid_f), y=np.asarray(grid_w_s),
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


def _correlation_figure(
    correlation: CorrelationCurves, t_lo: float, t_hi: float, t0_ms: float,
) -> go.Figure:
    """Per-sign best-R² correlation curves over a window.

    Plots only the two curves from
    ``findSegmentParameters(...).correlation`` — no threshold line and
    no peak-status dots (those needed detector internals).
    """
    t = np.asarray(correlation.t, dtype=float)
    pos_r2 = np.asarray(correlation.best_pos_r2, dtype=float)
    neg_r2 = np.asarray(correlation.best_neg_r2, dtype=float)
    pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
    neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)
    mask = (t >= t_lo) & (t <= t_hi)
    dt_window = to_datetime_array(t[mask], t0_ms)
    cd_window = time_customdata(t[mask])

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dt_window, y=pos_plot[mask], mode="lines", name="max R² (+)",
        line=dict(color="#2980b9", width=1.2),
        customdata=cd_window,
        hovertemplate=hover_time_template("R²(+)=%{y:.3f}"),
    ))
    fig.add_trace(go.Scatter(
        x=dt_window, y=neg_plot[mask], mode="lines", name="max R² (−)",
        line=dict(color="#c0392b", width=1.2),
        customdata=cd_window,
        hovertemplate=hover_time_template("R²(−)=%{y:.3f}"),
    ))
    fig.update_layout(
        height=260, margin=dict(l=10, r=10, t=30, b=30),
        xaxis=dict(title="time", type="date",
                   range=[to_datetime(t_lo, t0_ms),
                          to_datetime(t_hi, t0_ms)]),
        yaxis=dict(title="R² (per sign)", range=[0, 1.05]),
        hovermode="closest", plot_bgcolor="#fafbfc",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, x=0),
    )
    return fig


def _render_reconstruction_selector(loaded: LoadedSignal) -> str:
    """Sidebar picker for the gyro orientation-reconstruction method.

    Segmentation is magnitude-invariant, so the choice only affects the
    displayed vertical accel and the ZUPT Δh downstream — the detected
    segments are identical for every method.
    """
    choice = st.sidebar.selectbox(
        "Reconstruction (gyro)", RECONSTRUCT_CHOICES,
        key="reconstruct",
        help="Gyro orientation reconstruction for the displayed vertical "
             "acceleration and the ZUPT Δh. With a gyro and a method "
             "selected the plots show the reconstructed a_z; otherwise they "
             "fall back to |a|−g. Segmentation matches on |a|−g and is "
             "unaffected, so the segments never change.",
    )
    if getattr(loaded, "gyr", None) is None and choice != "none":
        st.sidebar.caption(
            ":orange[No gyroscope in this recording — reconstruction has no "
            "effect; plots show |a|−g.]"
        )
    return choice


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

    method = _render_reconstruction_selector(loaded)

    # Run the detector once (segments_df is None only before the first run
    # or after a reset / downstream invalidation).
    if st.session_state.get("segments_df") is None:
        _run_detector(loaded)

    t0_ms = (
        float(loaded.acc["timestamp_ms"].iloc[0])
        if not loaded.acc.empty else 0.0
    )

    # Whole-trace display signal (the acceleration the pipeline operates on),
    # reconstructed with the selected gyro method. The trace is already at the
    # canonical 50 Hz cadence, so resample=False. Cached for the report.
    disp = reconstructedSignal(
        loaded.acc, gyro=loaded.gyr, reconstruct=method, resample=False,
    )
    st.session_state["display_signal"] = disp

    if disp is None or disp.empty:
        st.error("The loaded signal is empty — go back and load another trace.")
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
        _main_signal_figure(
            disp, t0_ms, segments_df, sel,
            valid_intervals=loaded.valid_intervals,
            has_gyro=loaded.gyr is not None and not loaded.gyr.empty,
            method=method,
        ),
        use_container_width=True, key="seg_main_fig",
    )

    if len(segments_df) == 0 or sel is None:
        st.info("No segments yet. Use the **+ Add** button in the sidebar to "
                "create one, or hit **Reset to detector output** below to "
                "re-seed the list from the detector's proposal.")
    else:
        seg = segments_df.iloc[sel]
        t_lo = float(seg["start_s"]); t_hi = float(seg["end_s"])
        rt = str(seg["type"]).lower()

        st.markdown(
            f"#### Detail — segment #{sel}  ({seg['type']},  "
            f"{t_lo:.1f}–{t_hi:.1f}s, duration {t_hi - t_lo:.1f}s)"
        )

        # Fit the trapezoid + heatmaps + correlation for this interval.
        # Segmentation is magnitude-invariant, so this call is acc-only
        # (no gyro / reconstruct). Cached by bounds signature for step 5.
        params = findSegmentParameters(
            loaded.acc, t_lo, t_hi, ride_type=rt, resample=False,
        )
        cache = st.session_state.get("segment_params")
        if not isinstance(cache, dict):
            cache = {}
        cache[_seg_key(t_lo, t_hi, rt)] = params
        st.session_state["segment_params"] = cache

        if params is None:
            st.caption(
                "No trapezoid fit available for this interval — the window "
                "has no usable +peak / −peak pair."
            )
        else:
            heat = params.heatmaps
            grid_w_s = heat.grid_w_s; grid_f = heat.grid_f
            W_star = float(params.lobe1.half_width_s)
            f_star = float(params.lobe1.frac_flat)
            tc1 = float(params.lobe1.t_c)
            tc2 = float(params.lobe2.t_c)

            h1, h2 = st.columns(2)
            with h1:
                st.plotly_chart(
                    _heatmap_figure(heat.lobe1, grid_w_s, grid_f,
                                    f"lobe1 @ t={tc1:.1f}s", W_star, f_star),
                    use_container_width=True, key=f"heat1_{sel}",
                )
            with h2:
                st.plotly_chart(
                    _heatmap_figure(heat.lobe2, grid_w_s, grid_f,
                                    f"lobe2 @ t={tc2:.1f}s", W_star, f_star),
                    use_container_width=True, key=f"heat2_{sel}",
                )

            st.markdown("**Correlation score**")
            corr = params.correlation
            t_arr = np.asarray(corr.t, dtype=float)
            t_min = float(t_arr[0]); t_max = float(t_arr[-1])
            pad = max(3.0, 0.4 * (t_hi - t_lo))
            st.plotly_chart(
                _correlation_figure(
                    corr, max(t_min, t_lo - pad), min(t_max, t_hi + pad), t0_ms,
                ),
                use_container_width=True, key=f"corr_{sel}",
            )

            st.markdown("**Fitted trapezoid parameters (detector fit)**")
            render_trapezoid_params(params)

    # Spreadsheet-style fallback — the sidebar list covers most flows,
    # but power users can still bulk-edit from this table.
    with st.expander("Spreadsheet editor (advanced)", expanded=False):
        st.caption(
            "All segments in a grid. Edit the second-offset start/end cells, "
            "add rows at the bottom, or tick the left-edge checkboxes to "
            "delete rows in bulk. The read-only clock columns mirror the "
            "wall-clock time and refresh after each commit. Changes are "
            "committed as soon as you click outside the edited cell."
        )
        # Read-only wall-clock mirror columns sit next to the editable
        # second-offset cells so power users keep the seconds they bulk-edit
        # in while still reading the actual time. They are display-only and
        # dropped before the change check / write-back below.
        t0 = _current_t0_ms()
        display_df = segments_df.copy()
        display_df.insert(
            display_df.columns.get_loc("start_s") + 1, "start_clock",
            [seconds_to_clock(float(v), t0) for v in display_df["start_s"]],
        )
        display_df.insert(
            display_df.columns.get_loc("end_s") + 1, "end_clock",
            [seconds_to_clock(float(v), t0) for v in display_df["end_s"]],
        )
        edited = st.data_editor(
            display_df,
            num_rows="dynamic",
            use_container_width=True, height=240,
            column_config={
                "type": st.column_config.SelectboxColumn(
                    "type", options=["up", "down"], required=True,
                ),
                "start_s": st.column_config.NumberColumn(
                    "start (s)", min_value=0.0, step=0.1, format="%.2f",
                ),
                "start_clock": st.column_config.TextColumn(
                    "start (clock)", disabled=True,
                ),
                "end_s": st.column_config.NumberColumn(
                    "end (s)", min_value=0.0, step=0.1, format="%.2f",
                ),
                "end_clock": st.column_config.TextColumn(
                    "end (clock)", disabled=True,
                ),
                "joint_r2": st.column_config.NumberColumn(
                    "joint R² (detector)", disabled=True, format="%.3f",
                ),
            },
            key="seg_editor_table",
        )
        # Drop the display-only mirror columns before comparing / persisting
        # so the stored frame keeps its canonical relative-seconds schema.
        edited = edited.drop(columns=["start_clock", "end_clock"],
                             errors="ignore")
        if not edited.equals(segments_df):
            st.session_state["segments_df"] = edited.reset_index(drop=True)
            # Bulk-edits invalidate the index-keyed overrides (rows may
            # have been added/deleted/reordered). Easiest correct answer
            # is to drop them all; the user can re-tick the affected
            # segments after the table settles. The parameter cache is
            # keyed by bounds signature, so stale entries simply miss.
            st.session_state["lobe_overrides"] = {}
            st.session_state["segment_params"] = {}
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
