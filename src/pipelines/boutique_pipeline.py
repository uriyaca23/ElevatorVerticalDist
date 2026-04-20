"""Boutique Pipeline — Streamlit front-end.

Five-step wizard wrapping the project's core modules:

    Step 1: How to use        (description + limitations)
    Step 2: Data              (DB pull or Excel)
    Step 3: Segmentation      (editor-style interactive detail panel)
    Step 4: Prediction        (per-segment Δh)
    Step 5: PDF report        (Hebrew, per-segment pages)

Run:
    venv/bin/python -m streamlit run src/pipelines/boutique_pipeline.py
"""
from __future__ import annotations

import base64
import io
import sys
from datetime import datetime, time as dtime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.data.loadFromDB import LoadedSignal, PhoneType, loadDataFromS3  # noqa: E402
from src.prediction.algorithms import (  # noqa: E402
    PREDICT_ALGORITHM_CONFIG, PredictAlgorithm, Predictor,
)
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (  # noqa: E402
    detect as _detect,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (  # noqa: E402
    trapezoid_kernel,
)

predict_intervals = _detect.predict_intervals
heatmap_at = _detect.heatmap_at
classify_peak = _detect.classify_peak
find_local_maxima = _detect.find_local_maxima


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Elevator Vertical Distance — Boutique Pipeline"
GRAVITY_MS2 = 9.80665
RIDE_COLORS = {"up": "#1f6feb", "down": "#b54a9b", "outside": "#9aa0a6"}
SELECTED_COLOR = "#e67e22"

# Peak-status palette — matches the editor's legend so the visuals the
# user sees in the desktop tool and in the app are 1-to-1.
PEAK_STATUS_COLORS: dict[str, str] = {
    "accepted":          "#27ae60",
    "unpaired (greedy)": "#f39c12",
    "same-sign NMS":     "#9b59b6",
    "NMS (local)":       "#8e44ad",
    "lost to opp sign":  "#34495e",
    "R²<thr":            "#7f8c8d",
    "|A|<thr":           "#95a5a6",
}

STEP_HOWTO = 1
STEP_DATA = 2
STEP_SEGMENT = 3
STEP_PREDICT = 4
STEP_REPORT = 5
STEP_LABELS = {
    STEP_HOWTO:   "1 · How to use",
    STEP_DATA:    "2 · Data",
    STEP_SEGMENT: "3 · Segmentation",
    STEP_PREDICT: "4 · Prediction",
    STEP_REPORT:  "5 · Report",
}


# ---------------------------------------------------------------------------
# Styling
# ---------------------------------------------------------------------------

_CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, kbd { font-family: 'JetBrains Mono', monospace; }

section.main > div.block-container {
    padding-top: 2.2rem; padding-bottom: 4rem; max-width: 1200px;
}

h1, h2, h3 { letter-spacing: -0.01em; }
h1 { font-weight: 700; }
h2 { font-weight: 600; margin-top: 1.6rem; }
h3 { font-weight: 600; color: #2c3e50; }

.hero {
    background: linear-gradient(135deg, #0f2a4a 0%, #1f6feb 100%);
    color: #fff; padding: 1.6rem 1.8rem; border-radius: 14px;
    margin-bottom: 1.4rem; box-shadow: 0 10px 30px rgba(31,111,235,0.18);
}
.hero h1 { color: #fff; margin: 0; font-size: 1.55rem; }
.hero p  { margin: 0.35rem 0 0 0; opacity: 0.9; font-size: 0.95rem; }

.step-pill {
    display: inline-block; padding: 0.15rem 0.6rem; border-radius: 999px;
    font-size: 0.72rem; font-weight: 600;
    background: rgba(255,255,255,0.2); margin-right: 0.4rem;
}

.limitation {
    background: #fff8e6; border-left: 3px solid #f0b429;
    padding: 0.7rem 1rem; border-radius: 6px; margin: 0.4rem 0;
    font-size: 0.93rem;
}
.info-block {
    background: #eef4ff; border-left: 3px solid #1f6feb;
    padding: 0.75rem 1rem; border-radius: 6px; margin: 0.4rem 0;
    font-size: 0.93rem;
}

.status-legend {
    display: flex; flex-wrap: wrap; gap: 0.6rem; margin: 0.5rem 0 0.8rem 0;
    font-size: 0.78rem;
}
.status-legend .chip {
    display: inline-flex; align-items: center;
    padding: 0.12rem 0.55rem; border-radius: 999px;
    background: #fafbfc; border: 1px solid #e6e9ef; color: #233044;
}
.status-legend .dot {
    width: 0.6rem; height: 0.6rem; border-radius: 50%;
    margin-right: 0.35rem; border: 1px solid rgba(0,0,0,0.2);
}

div[data-testid="stSidebar"] { background: #0f1b2d; }
div[data-testid="stSidebar"] * { color: #eef2f7; }
section[data-testid="stSidebar"] {
    width: 360px !important;
    min-width: 360px !important;
}
section[data-testid="stSidebar"] > div:first-child { width: 360px !important; }

.stButton > button {
    border-radius: 8px; font-weight: 500; padding: 0.45rem 1.1rem;
}
.stDownloadButton > button {
    border-radius: 8px; font-weight: 600;
    background: #1f6feb; color: #fff; border: none;
    padding: 0.6rem 1.3rem;
}
.stDownloadButton > button:hover { background: #1559c7; }

/* Fitted-trapezoid parameter card */
.trap-params {
    display: grid; grid-template-columns: 1fr 1fr; gap: 0.7rem;
    margin: 0.3rem 0 0.4rem 0;
}
.trap-card {
    border: 1px solid #e6e9ef; border-radius: 12px; padding: 0.85rem 1rem;
    background: #ffffff; box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.trap-card.lobe1 { border-top: 3px solid #c0392b; }
.trap-card.lobe2 { border-top: 3px solid #8e44ad; }
.trap-card.joint {
    grid-column: 1 / -1;
    border-top: 3px solid #1f6feb;
    background: linear-gradient(135deg, #f7faff 0%, #ffffff 100%);
}
.trap-card .card-title {
    font-size: 0.72rem; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.09em; font-weight: 700; margin-bottom: 0.55rem;
}
.trap-card.lobe1 .card-title { color: #c0392b; }
.trap-card.lobe2 .card-title { color: #8e44ad; }
.trap-card.joint .card-title { color: #1f6feb; }
.trap-grid {
    display: grid; grid-template-columns: repeat(auto-fit, minmax(70px, 1fr));
    gap: 0.55rem 0.8rem;
}
.trap-item .k {
    font-size: 0.65rem; color: #6b7280; text-transform: uppercase;
    letter-spacing: 0.06em; font-weight: 600;
}
.trap-item .v {
    font-size: 1rem; font-weight: 700; color: #0f2a4a;
    font-family: 'JetBrains Mono', monospace;
}

/* Sidebar segment list */
.sb-seg-label {
    font-size: 0.78rem; color: #eef2f7;
    font-family: 'JetBrains Mono', monospace;
}
</style>
"""


# ---------------------------------------------------------------------------
# Session helpers
# ---------------------------------------------------------------------------

def _init_state() -> None:
    defaults = {
        "step":              STEP_HOWTO,
        "loaded":             None,
        "detector_state":     None,
        "predictions":        [],
        "segments_df":        None,
        "selected_segment":   0,
        "prediction_rows":    None,
        "predict_selected":   None,
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def _goto(step: int) -> None:
    st.session_state["step"] = step
    st.rerun()


def _utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Step 1 — How to use (dedicated page)
# ---------------------------------------------------------------------------

def _render_howto() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 1</span>'
        f'<h1>{APP_TITLE}</h1>'
        '<p>An end-to-end workbench for turning a raw accelerometer trace into '
        'a per-ride height report.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### What this app does")
    st.markdown(
        "This pipeline takes an accelerometer signal recorded during elevator "
        "rides and turns it into a vetted list of rides with per-ride height "
        "differences. The workflow is:"
    )
    st.markdown(
        "1. **Ingest** — pull a trace from the experiment DB by phone + time "
        "window (default), or upload an Excel file with a time and "
        "acceleration column.\n"
        "2. **Segment** — a trapezoid-template detector proposes ride "
        "intervals. You can inspect each candidate against its matched-filter "
        "heatmap, correlation score, and the fitted template overlay, then "
        "accept, edit, remove, or add segments.\n"
        "3. **Predict** — each final segment is fed to the S-curve "
        "accelerometer predictor, which returns a signed Δh (meters), a 90% "
        "confidence interval, and a quality verdict.\n"
        "4. **Export** — download a Hebrew PDF report with one page per segment."
    )


    st.divider()
    _, c = st.columns([3, 1])
    with c:
        if st.button("Start →", type="primary"):
            _goto(STEP_DATA)


# ---------------------------------------------------------------------------
# Step 2 — Data
# ---------------------------------------------------------------------------

def _tabular_to_signal(
    df: pd.DataFrame, time_col: str, acc_col: str, filename: str,
) -> LoadedSignal:
    """Reduce a user-provided Excel/CSV to the ACC schema the detector wants."""
    t_raw = pd.to_numeric(df[time_col], errors="coerce").to_numpy(dtype=float)
    a = pd.to_numeric(df[acc_col], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(t_raw) & np.isfinite(a)
    t_raw = t_raw[good]
    a = a[good]
    if t_raw.size < 10:
        raise ValueError("Not enough numeric samples after cleaning (need ≥10).")

    span = float(t_raw[-1] - t_raw[0])
    if span > 0 and span < 1e4:
        ts_ms = (t_raw * 1000.0).astype("int64")
    else:
        ts_ms = t_raw.astype("int64")
    n = t_raw.size
    acc = pd.DataFrame({
        "timestamp_ms": ts_ms,
        "x": np.zeros(n),
        "y": np.zeros(n),
        "z": GRAVITY_MS2 + a,
    })
    dt = float(np.median(np.diff(ts_ms))) / 1000.0 if n > 1 else 0.02
    fs_est = 1.0 / dt if dt > 0 else float("nan")
    return LoadedSignal(
        acc=acc, source=f"File · {filename}",
        meta={
            "filename":    filename,
            "time_column": time_col,
            "acc_column":  acc_col,
            "samples":     int(n),
            "sample_rate": f"{fs_est:.1f} Hz",
        },
    )


def _read_tabular_upload(uploaded) -> pd.DataFrame:
    """Pick the right reader based on the uploaded file's extension."""
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    return pd.read_excel(uploaded)


def _render_data() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 2</span>'
        '<h1>Load a signal</h1>'
        '<p>Fill in either the phone info section or the file upload — '
        'pick one and use its fetch button at the bottom.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    # --- Phone Info --------------------------------------------------------
    st.subheader("Phone info")
    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    with c1:
        phone = st.selectbox("Phone type",
                             [p.value for p in PhoneType], key="db_phone")
        phone_id = st.text_input(
            "Phone ID",
            placeholder="e.g. IMEI, serial, or device label",
            key="db_phone_id",
            help="Identifier your team uses to disambiguate between multiple "
                 "phones of the same type. Free text — it's stamped into the "
                 "report metadata verbatim.",
        )
    with c2:
        d_start = st.date_input("Start date", key="db_d_start")
        t_start_time = st.time_input("Start time", value=dtime(9, 0),
                                     key="db_t_start")
    with c3:
        d_end = st.date_input("End date", key="db_d_end")
        t_end_time = st.time_input("End time", value=dtime(9, 5),
                                   key="db_t_end")

    t_start = datetime.combine(d_start, t_start_time)
    t_end = datetime.combine(d_end, t_end_time)
    valid_window = t_end > t_start
    if not valid_window:
        st.caption(":orange[End time must be after start time.]")
    else:
        st.caption(f"Window: {(t_end - t_start).total_seconds():.0f} s")

    st.divider()

    # --- File upload -------------------------------------------------------
    st.subheader("Or upload a file")
    st.markdown(
        '<div class="info-block"><b>Expected file structure</b><br>'
        'A spreadsheet (.xlsx / .xls) or CSV with at least two columns:'
        '<ul style="margin:0.3rem 0 0 1.2rem">'
        '<li><b>time</b> — numeric timestamps. Seconds (e.g. <code>0.00, '
        '0.02, …</code>) <i>or</i> milliseconds (Unix epoch ms). The app '
        'auto-detects the unit from the value range.</li>'
        '<li><b>vertical acceleration</b> — numeric, in m/s². This is '
        'acceleration along the gravity axis (gravity removed <i>or</i> '
        'included — the detector only needs the relative signal).</li>'
        '</ul>'
        'Other columns are ignored. The first row must contain the '
        'column headers. Pick which header is which in the dropdowns '
        'below after uploading.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload .xlsx, .xls, or .csv", type=["xlsx", "xls", "csv"],
        key="file_uploader",
    )

    file_df: pd.DataFrame | None = None
    time_col = acc_col = None
    if uploaded is not None:
        try:
            file_df = _read_tabular_upload(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {type(e).__name__}: {e}")
            file_df = None

    if file_df is not None:
        st.markdown("**Preview (first 12 rows)**")
        st.dataframe(file_df.head(12), use_container_width=True, height=220)
        cols = list(file_df.columns)
        cc1, cc2 = st.columns(2)
        with cc1:
            time_col = st.selectbox("Time column", cols, key="xl_time_col")
        with cc2:
            acc_col = st.selectbox(
                "Accelerometer column",
                [c for c in cols if c != time_col], key="xl_acc_col",
            )

    st.divider()

    # --- Bottom-left action row: one button, smart source selection -------
    c_back, c_next, _ = st.columns([1.0, 1.2, 3.0])
    with c_back:
        if st.button("← Back"):
            _goto(STEP_HOWTO)
    with c_next:
        file_ready = (
            file_df is not None
            and time_col is not None
            and acc_col is not None
        )
        phone_ready = valid_window
        if st.button("Next →", type="primary", key="btn_next"):
            if file_ready:
                try:
                    loaded = _tabular_to_signal(
                        file_df, time_col, acc_col, uploaded.name,
                    )
                except Exception as e:
                    st.error(f"Load failed: {type(e).__name__}: {e}")
                    return
                st.session_state["loaded"] = loaded
                _reset_downstream_state()
                _goto(STEP_SEGMENT)
            elif phone_ready:
                with st.spinner("Fetching…"):
                    loaded = loadDataFromS3(
                        PhoneType(phone), phone_id.strip(), t_start, t_end,
                    )
                st.session_state["loaded"] = loaded
                _reset_downstream_state()
                _goto(STEP_SEGMENT)
            else:
                st.error(
                    "Nothing to load — upload a file (with both columns "
                    "picked) or fill in a valid phone-info time window."
                )


def _reset_downstream_state() -> None:
    st.session_state["detector_state"] = None
    st.session_state["predictions"] = []
    st.session_state["segments_df"] = None
    st.session_state["selected_segment"] = 0
    st.session_state["prediction_rows"] = None
    st.session_state["predict_selected"] = None


# ---------------------------------------------------------------------------
# Step 3 — Segmentation
# ---------------------------------------------------------------------------

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


def _valid_segments(df: pd.DataFrame | None) -> pd.DataFrame:
    cols = ["type", "start_s", "end_s", "joint_r2"]
    if df is None or df.empty:
        return pd.DataFrame(columns=cols)
    out = df.copy()
    out["start_s"] = pd.to_numeric(out["start_s"], errors="coerce")
    out["end_s"] = pd.to_numeric(out["end_s"], errors="coerce")
    out = out.dropna(subset=["start_s", "end_s"])
    out = out[out["end_s"] > out["start_s"]]
    out["type"] = out["type"].astype(str).str.lower().where(
        out["type"].isin(["up", "down"]), "up",
    )
    for c in cols:
        if c not in out.columns:
            out[c] = np.nan
    return out[cols].reset_index(drop=True)


def _find_matching_prediction(
    predictions: list[dict], t_lo: float, t_hi: float,
) -> dict | None:
    """Best-overlap detector prediction for a user-edited segment.

    Returns ``None`` if there's no overlap — happens when the user
    adds a segment that the detector never proposed. The UI then
    falls back to a raw-signal view (no trapezoid overlay).
    """
    best = None
    best_overlap = 0.0
    for p in predictions:
        s = float(p["t_start_s"])
        e = float(p["t_end_s"])
        overlap = max(0.0, min(e, t_hi) - max(s, t_lo))
        if overlap > best_overlap:
            best_overlap = overlap
            best = p
    return best


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


def _peak_status_legend_html() -> str:
    chips = []
    for tag, color in PEAK_STATUS_COLORS.items():
        chips.append(
            f'<span class="chip"><span class="dot" '
            f'style="background:{color}"></span>{tag}</span>'
        )
    return '<div class="status-legend">' + "".join(chips) + "</div>"


def _trap_item(key: str, val: str) -> str:
    return (
        f'<div class="trap-item">'
        f'<div class="k">{key}</div>'
        f'<div class="v">{val}</div>'
        f'</div>'
    )


def _render_trapezoid_params(prediction: dict | None) -> None:
    """Three cards: lobe 1, lobe 2, and the joint fit.

    Per-lobe (``half_width_s`` ``W``, ``frac_flat`` ``f``, ``a_peak``
    ``A``, ``r2_local``, ``t_c``) — shown per lobe as requested. The
    joint-fit values (``W*``, ``f*``, ``|A*|``, ``joint_r2_mean``,
    ``heatmap_energy``) are the shared-shape parameters from the pair
    filter; for a confirmed detector match the per-lobe W/f collapse
    onto these shared values.
    """
    if prediction is None:
        st.caption("No detector-matched prediction for this segment — "
                   "parameters unavailable.")
        return

    def _vals(lobe: dict) -> tuple[str, str, str, str, str, str]:
        t_c = float(lobe.get("t_c", float("nan")))
        A = float(lobe.get("a_peak", float("nan")))
        W = float(lobe.get("half_width_s", float("nan")))
        f = float(lobe.get("frac_flat", float("nan")))
        r2 = float(lobe.get("r2_local", float("nan")))
        return (
            f"{t_c:.2f} s"      if np.isfinite(t_c) else "—",
            f"{A:+.2f}"          if np.isfinite(A)  else "—",
            f"{W:.2f} s"         if np.isfinite(W)  else "—",
            f"{f:.2f}"           if np.isfinite(f)  else "—",
            f"{r2:.3f}"          if np.isfinite(r2) else "—",
            f"{abs(A):.2f}"      if np.isfinite(A)  else "—",
        )

    l1 = prediction.get("lobe1") or {}
    l2 = prediction.get("lobe2") or {}
    tc1, A1, W1, f1, r2_1, _ = _vals(l1)
    tc2, A2, W2, f2, r2_2, _ = _vals(l2)

    W_star = float(l1.get("half_width_s", float("nan")))
    f_star = float(l1.get("frac_flat",    float("nan")))
    a1_val = float(l1.get("a_peak",       float("nan")))
    A_star = abs(a1_val) if np.isfinite(a1_val) else float("nan")
    joint_r2 = float(prediction.get("joint_r2_mean",   float("nan")))
    heat_e   = float(prediction.get("heatmap_energy", float("nan")))

    def _card(cls: str, title: str, items: list[tuple[str, str]]) -> str:
        inner = "".join(_trap_item(k, v) for k, v in items)
        return (
            f'<div class="trap-card {cls}">'
            f'<div class="card-title">{title}</div>'
            f'<div class="trap-grid">{inner}</div>'
            f'</div>'
        )

    html = (
        '<div class="trap-params">'
        + _card("lobe1", "Lobe 1 · take-off", [
            ("t_c", tc1), ("A", f"{A1} m/s²"), ("W", W1),
            ("f", f1), ("R²", r2_1),
        ])
        + _card("lobe2", "Lobe 2 · landing", [
            ("t_c", tc2), ("A", f"{A2} m/s²"), ("W", W2),
            ("f", f2), ("R²", r2_2),
        ])
        + _card("joint", "Joint fit · shared shape", [
            ("W*", f"{W_star:.2f} s" if np.isfinite(W_star) else "—"),
            ("f*", f"{f_star:.2f}"   if np.isfinite(f_star) else "—"),
            ("|A*|", f"{A_star:.2f}" if np.isfinite(A_star) else "—"),
            ("R² joint", f"{joint_r2:.3f}" if np.isfinite(joint_r2) else "—"),
            ("heat E",   f"{heat_e:.3f}"   if np.isfinite(heat_e)   else "—"),
        ])
        + '</div>'
    )
    st.markdown(html, unsafe_allow_html=True)




def _render_segmentation() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    if loaded is None:
        st.warning("Load a signal first.")
        if st.button("← Back to data"):
            _goto(STEP_DATA)
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
            _goto(STEP_DATA)
        return

    predictions = st.session_state["predictions"]
    segments_df = _valid_segments(st.session_state["segments_df"])

    sel = st.session_state.get("selected_segment")
    if sel is None or (len(segments_df) and sel >= len(segments_df)):
        sel = 0 if len(segments_df) else None
        st.session_state["selected_segment"] = sel

    # Segment list lives in the sidebar. Main area shows signal + detail.
    _render_segment_sidebar(segments_df, sel)

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
        matching = _find_matching_prediction(predictions, t_lo, t_hi)

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
        st.markdown(_peak_status_legend_html(), unsafe_allow_html=True)
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
        _render_trapezoid_params(matching)

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
            _goto(STEP_DATA)
    with c2:
        if st.button("Reset to detector output",
                     help="Discards all manual edits and re-runs the detector "
                          "on this signal. Config is unchanged, so the result "
                          "matches the initial proposal."):
            _run_detector(loaded)
            st.rerun()
    with c3:
        valid = _valid_segments(st.session_state["segments_df"])
        if st.button(f"Predict Δh → ({len(valid)} segments)",
                     type="primary", disabled=len(valid) == 0):
            st.session_state["prediction_rows"] = None
            _goto(STEP_PREDICT)


def _segment_label(i: int, row: pd.Series) -> str:
    rt = str(row.get("type", "up"))
    s = float(row["start_s"]); e = float(row["end_s"])
    return f"#{i:<2} {rt:<4}  {s:6.1f}–{e:6.1f} s"


def _render_segment_sidebar(segments_df: pd.DataFrame, sel: int | None) -> None:
    """Segmentation-step sidebar panel.

    Uses :func:`st.radio` for selection — the widget gives arrow-key
    navigation (↑ / ↓) for free once the list has keyboard focus.
    Edit / delete / add controls for the selected segment sit below the
    radio so they don't steal focus away from the arrow-key nav.
    """
    st.sidebar.markdown("### Segments")
    st.sidebar.caption(
        "Click a row or focus the list and use ↑ / ↓ arrow keys. "
        "Edit / delete apply to the currently selected segment."
    )

    if len(segments_df) == 0:
        st.sidebar.info("No segments yet.")
    else:
        options = list(range(len(segments_df)))
        cur_index = int(sel) if (sel is not None and sel < len(options)) else 0
        new_sel = st.sidebar.radio(
            "Segments", options=options,
            index=cur_index,
            format_func=lambda i: _segment_label(i, segments_df.iloc[i]),
            label_visibility="collapsed",
            key="sb_seg_radio",
        )
        if int(new_sel) != int(sel or 0):
            st.session_state["selected_segment"] = int(new_sel)
            st.rerun()
        sel = int(new_sel)

        # Edit / delete act on the currently selected segment.
        row = segments_df.iloc[sel]
        col_edit, col_del = st.sidebar.columns([2, 1])
        with col_edit:
            with st.popover("✎  Edit selected", use_container_width=True):
                new_lo = st.number_input(
                    "Start (s)", value=float(row["start_s"]), step=0.05,
                    format="%.2f", key=f"sb_edit_lo_{sel}",
                )
                new_hi = st.number_input(
                    "End (s)", value=float(row["end_s"]), step=0.05,
                    format="%.2f", key=f"sb_edit_hi_{sel}",
                )
                new_type = st.selectbox(
                    "Type", options=["up", "down"],
                    index=0 if str(row["type"]) == "up" else 1,
                    key=f"sb_edit_type_{sel}",
                )
                if st.button("Save", key=f"sb_edit_save_{sel}",
                             use_container_width=True, type="primary"):
                    if new_hi <= new_lo:
                        st.error("End must be greater than start.")
                    else:
                        df = st.session_state["segments_df"]
                        df.loc[df.index[sel], "start_s"] = float(new_lo)
                        df.loc[df.index[sel], "end_s"] = float(new_hi)
                        df.loc[df.index[sel], "type"] = new_type
                        df.loc[df.index[sel], "joint_r2"] = np.nan
                        st.rerun()
        with col_del:
            if st.button("🗑", key=f"sb_del_{sel}",
                         use_container_width=True,
                         help="Delete the currently selected segment"):
                df = st.session_state["segments_df"]
                df = df.drop(df.index[sel]).reset_index(drop=True)
                st.session_state["segments_df"] = df
                new = max(0, sel - 1) if len(df) else None
                st.session_state["selected_segment"] = new
                st.rerun()

    st.sidebar.markdown("")
    if st.sidebar.button("+  Add segment", use_container_width=True,
                         key="sb_add_segment"):
        df = st.session_state.get("segments_df")
        if df is None:
            df = pd.DataFrame(columns=["type", "start_s", "end_s", "joint_r2"])
        # Default: 5-second segment at the current selection's tail, or at
        # the start of the signal when the list is empty.
        if len(df):
            last = df.iloc[-1]
            new_lo = float(last["end_s"]) + 0.5
        else:
            new_lo = 0.0
        new_row = {"type": "up", "start_s": new_lo,
                   "end_s": new_lo + 5.0, "joint_r2": np.nan}
        df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
        st.session_state["segments_df"] = df
        st.session_state["selected_segment"] = len(df) - 1
        st.rerun()

    # Footer — source info + a reset button, since the main sidebar's
    # step indicator was collapsed to make room for the segment list.
    st.sidebar.markdown("---")
    loaded = st.session_state.get("loaded")
    if loaded is not None:
        st.sidebar.caption(f"Loaded: {loaded.source}")
    if st.sidebar.button("Reset session", key="sb_reset_session",
                         use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


def _render_predict_segment_sidebar(rows: list[dict], selected: int) -> None:
    """Prediction-step sidebar — same arrow-key-navigable list as the
    segmentation step, minus the edit / delete / add controls.
    Prediction is read-only w.r.t. the segment layout.
    """
    st.sidebar.markdown("### Segments")
    st.sidebar.caption(
        "Click a row or focus the list and use ↑ / ↓ arrow keys."
    )

    options = [int(r["segment"]) for r in rows]
    by_id = {int(r["segment"]): r for r in rows}

    def _label(seg_id: int) -> str:
        r = by_id[seg_id]
        dh = r["delta_height_m"]
        dh_str = f"Δh={dh:+.2f}m" if np.isfinite(dh) else "Δh=—"
        return (f"#{r['segment']:<2} {r['type']:<4}  "
                f"{r['start_s']:5.1f}–{r['end_s']:5.1f}s  {dh_str}")

    new_sel = st.sidebar.radio(
        "Segments", options=options,
        index=options.index(selected) if selected in options else 0,
        format_func=_label,
        label_visibility="collapsed",
        key="sb_pred_radio",
    )
    if int(new_sel) != selected:
        st.session_state["predict_selected"] = int(new_sel)
        st.rerun()

    # Footer mirroring the segment-step sidebar.
    st.sidebar.markdown("---")
    loaded = st.session_state.get("loaded")
    if loaded is not None:
        st.sidebar.caption(f"Loaded: {loaded.source}")
    if st.sidebar.button("Reset session", key="sb_pred_reset",
                         use_container_width=True):
        for k in list(st.session_state.keys()):
            del st.session_state[k]
        st.rerun()


# ---------------------------------------------------------------------------
# Step 4 — Prediction
# ---------------------------------------------------------------------------

def _build_predictor() -> Predictor:
    cfg = PREDICT_ALGORITHM_CONFIG(algorithm=PredictAlgorithm.SCURVE_ACCEL)
    return Predictor(cfg)


def _slice_acc(acc: pd.DataFrame, t0_ms: float, t_lo: float, t_hi: float) -> pd.DataFrame:
    ts = acc["timestamp_ms"].astype(float).to_numpy()
    lo_ms = t0_ms + t_lo * 1000.0
    hi_ms = t0_ms + t_hi * 1000.0
    mask = (ts >= lo_ms) & (ts < hi_ms)
    return acc.loc[mask].reset_index(drop=True)


def _run_predictions(loaded: LoadedSignal, segments: pd.DataFrame) -> list[dict]:
    state = st.session_state["detector_state"]
    t0_ms = float(state["t0_ms"])
    predictor = _build_predictor()
    rows: list[dict] = []
    for i, row in segments.iterrows():
        t_lo = float(row["start_s"]); t_hi = float(row["end_s"])
        rt = str(row["type"]).lower()
        seg = _slice_acc(loaded.acc, t0_ms, t_lo, t_hi)
        base = {
            "segment":        int(i), "type": rt,
            "start_s":        t_lo, "end_s": t_hi,
            "duration_s":     t_hi - t_lo,
        }
        if seg.empty:
            rows.append({**base,
                "delta_height_m": float("nan"),
                "abs_height_m":   float("nan"),
                "accepted":       False,
                "quality_score":  float("nan"),
                "reject_reason":  "empty_slice",
                "ci_half_width":  float("nan")})
            continue
        try:
            out = predictor.predict(seg, phone_model="")
            dh = float(out.height_diff)
            ci = float(out.ci_half_width) if np.isfinite(out.ci_half_width) else float("nan")
            signed = abs(dh) if rt == "up" else -abs(dh)
            rows.append({**base,
                "delta_height_m": signed,
                "abs_height_m":   abs(dh),
                "accepted":       bool(out.accepted),
                "quality_score":  float(out.quality_score),
                "reject_reason":  str(out.reject_reason or ""),
                "ci_half_width":  ci})
        except Exception as e:
            rows.append({**base,
                "delta_height_m": float("nan"),
                "abs_height_m":   float("nan"),
                "accepted":       False,
                "quality_score":  float("nan"),
                "reject_reason":  f"{type(e).__name__}: {e}",
                "ci_half_width":  float("nan")})
    return rows


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


def _prediction_bar_figure(rows: list[dict], selected: int | None) -> go.Figure:
    xs = [f"#{r['segment']}" for r in rows]
    ys = [0.0 if not np.isfinite(r["delta_height_m"]) else r["delta_height_m"]
          for r in rows]
    colors = []
    for r in rows:
        if selected is not None and int(r["segment"]) == selected:
            colors.append(SELECTED_COLOR)
        else:
            colors.append(RIDE_COLORS.get(r["type"], "#777"))
    fig = go.Figure(go.Bar(
        x=xs, y=ys, marker_color=colors,
        hovertemplate="seg %{x}<br>Δh=%{y:+.2f} m<extra></extra>",
    ))
    fig.update_layout(
        height=270, margin=dict(l=10, r=10, t=10, b=30),
        yaxis_title="Δh (m)", plot_bgcolor="#fafbfc",
    )
    fig.add_hline(y=0, line_color="#333", line_width=0.5)
    return fig


def _render_prediction() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    segments: pd.DataFrame | None = st.session_state["segments_df"]
    state = st.session_state.get("detector_state")
    if loaded is None or segments is None or state is None:
        st.warning("Complete earlier steps first.")
        if st.button("← Back"):
            _goto(STEP_SEGMENT)
        return

    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 4</span>'
        '<h1>Per-segment height predictions</h1>'
        '<p>S-curve accelerometer fitter. Click a segment to highlight it '
        'in the signal and the bar chart.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    valid = _valid_segments(segments)
    if st.session_state["prediction_rows"] is None:
        with st.spinner(f"Running Predictor on {len(valid)} segments…"):
            st.session_state["prediction_rows"] = _run_predictions(loaded, valid)
    rows: list[dict] = st.session_state["prediction_rows"] or []
    if not rows:
        st.info("No predictable segments.")
        if st.button("← Back"):
            _goto(STEP_SEGMENT)
        return

    valid_segment_ids = [int(r["segment"]) for r in rows]
    if st.session_state["predict_selected"] not in valid_segment_ids:
        st.session_state["predict_selected"] = valid_segment_ids[0]
    selected: int = int(st.session_state["predict_selected"])

    # Sidebar list (read-only — predictions don't mutate segments).
    _render_predict_segment_sidebar(rows, selected)

    st.plotly_chart(
        _prediction_main_figure(state, rows, selected),
        use_container_width=True, key="pred_main_fig",
    )
    st.plotly_chart(
        _prediction_bar_figure(rows, selected),
        use_container_width=True, key="pred_bar_fig",
    )

    sel_row = next((r for r in rows if int(r["segment"]) == selected), None)
    if sel_row is not None:
        st.markdown(f"#### Detail — segment #{sel_row['segment']}")
        info_cols = st.columns(4)
        info_cols[0].metric(
            "Δh",
            f"{sel_row['delta_height_m']:+.2f} m"
            if np.isfinite(sel_row['delta_height_m']) else "—",
        )
        info_cols[1].metric(
            "±CI 90%",
            f"{sel_row['ci_half_width']:.2f} m"
            if np.isfinite(sel_row['ci_half_width']) else "—",
        )
        info_cols[2].metric(
            "Quality",
            f"{sel_row['quality_score']:.1f}"
            if np.isfinite(sel_row['quality_score']) else "—",
        )
        info_cols[3].metric(
            "Accepted", "yes" if sel_row["accepted"] else "no",
        )
        if sel_row.get("reject_reason"):
            st.caption(f"reject_reason: `{sel_row['reject_reason']}`")

        # Trapezoid parameters for the selected segment.
        predictions = st.session_state.get("predictions") or []
        matching = _find_matching_prediction(
            predictions, float(sel_row["start_s"]), float(sel_row["end_s"]),
        )
        st.markdown("**Fitted trapezoid parameters**")
        _render_trapezoid_params(matching)

    st.markdown("### All segments")
    df = pd.DataFrame(rows)
    display_cols = ["segment", "type", "start_s", "end_s", "duration_s",
                    "delta_height_m", "ci_half_width", "accepted",
                    "quality_score", "reject_reason"]
    st.dataframe(
        df[display_cols], use_container_width=True, height=260,
        column_config={
            "start_s":        st.column_config.NumberColumn("start (s)", format="%.1f"),
            "end_s":          st.column_config.NumberColumn("end (s)", format="%.1f"),
            "duration_s":     st.column_config.NumberColumn("duration (s)", format="%.1f"),
            "delta_height_m": st.column_config.NumberColumn("Δh (m)", format="%+.2f"),
            "ci_half_width":  st.column_config.NumberColumn("±CI 90% (m)", format="%.2f"),
            "quality_score":  st.column_config.NumberColumn("quality", format="%.1f"),
        },
    )

    st.divider()
    c1, _, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Back to segmentation"):
            st.session_state["prediction_rows"] = None
            _goto(STEP_SEGMENT)
    with c3:
        if st.button("Generate report →", type="primary"):
            _goto(STEP_REPORT)


# ---------------------------------------------------------------------------
# Step 5 — PDF report (Hebrew, per-segment pages)
# ---------------------------------------------------------------------------

# Small translation table for static strings. Kept here so the PDF writer
# and the on-screen preview read from the same place.
HEB = {
    "report_title":   "דוח פייפליין מעלית",
    "source":         "מקור",
    "generated":      "נוצר בתאריך",
    "samples":        "מספר דגימות",
    "sample_rate":    "קצב דגימה",
    "phone_type":     "סוג מכשיר",
    "window":         "חלון זמן",
    "experiment":     "ניסוי",
    "summary":        "סיכום",
    "n_segments":     "מספר מקטעים",
    "n_accepted":     "התקבלו בבקרה",
    "net_dh":         "סך הפרשי גובה (מטר)",
    "per_seg_table":  "טבלת הפרשי גובה",
    "col_idx":        "מקטע",
    "col_type":       "סוג",
    "col_start":      "התחלה (ש')",
    "col_end":        "סיום (ש')",
    "col_dur":        "משך (ש')",
    "col_dh":         "הפרש גובה (מ')",
    "col_ci":         "רווח סמך (מ')",
    "col_quality":    "ציון איכות",
    "col_accepted":   "התקבל",
    "yes":            "כן",
    "no":             "לא",
    "up":             "עלייה",
    "down":           "ירידה",
    "segment_page":   "מקטע",
    "trap_heading":   "אות האצה עם תבנית הטרפז",
    "corr_heading":   "ציון מתאם סביב המקטע (±30 ש')",
    "status_legend":  "מקרא צבעי פסגות",
    "reject_reason":  "סיבת דחייה",
    "how_to_read":    "איך לקרוא את הדוח",
    "how_to_read_body": (
        "כל מקטע נסיעה הותאם בפרופיל תנועה מסוג S (S-curve) "
        "על ציר ההאצה האנכי. הפרש הגובה הוא המרחק המוערך, עם סימן, "
        "שהמעלית עברה במקטע. עמודת רווח הסמך מציגה את חצי הרוחב של "
        "רווח סמך של 90% המבוסס על חסם קרמר־ראו (CRB) לאחר כיול "
        "קונפורמלי. מקטעים שמסומנים כ'לא התקבלו' נכשלו במסנן האיכות "
        "הפנימי (התאמת χ², לוג־הסתברות פריור, יחס רווח סמך, אי־הסכמה "
        "בין ZUPT ו־NLS). הערכי ה־Δh שלהם מדווחים, אך יש להתייחס "
        "אליהם כבעלי ביטחון נמוך."
    ),
}


def _maybe_register_hebrew_font() -> str:
    """Register a Hebrew-capable TTF with reportlab. Returns the font name
    to use; falls back to ``Helvetica`` if no suitable font is found.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return "Helvetica"

    candidates = []
    try:
        import matplotlib
        mpl_fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates.extend([
            mpl_fonts / "DejaVuSans.ttf",
            mpl_fonts / "DejaVuSans-Bold.ttf",
        ])
    except Exception:
        pass
    # Common macOS / Linux locations.
    candidates.extend([
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ])
    for cand in candidates:
        if cand.exists():
            try:
                pdfmetrics.registerFont(TTFont("HebrewSans", str(cand)))
                return "HebrewSans"
            except Exception:
                continue
    return "Helvetica"


import re as _re
_RL_TAG_RE = _re.compile(r"<[^<>]+>")


def _rtl(s: str) -> str:
    """Bidi-reorder Hebrew strings while preserving ReportLab inline tags.

    python-bidi's ``get_display`` reverses character order for RTL-dominant
    text. Running it over a string that embeds ReportLab paraparser markup
    (``<b>...</b>``, ``<font>...</font>``, etc.) also reverses the tag
    bytes themselves, producing ``</b>...<b>`` — which paraparser then
    rejects with "parse ended with 1 unclosed tags para". We therefore
    apply the bidi reorder only to the plain-text chunks between tags and
    leave the tags in their original LTR ASCII order.
    """
    try:
        from bidi.algorithm import get_display
    except ImportError:
        return s
    if "<" not in s:
        return get_display(s)
    parts: list[str] = []
    idx = 0
    for m in _RL_TAG_RE.finditer(s):
        if m.start() > idx:
            parts.append(get_display(s[idx:m.start()]))
        parts.append(m.group(0))
        idx = m.end()
    if idx < len(s):
        parts.append(get_display(s[idx:]))
    return "".join(parts)


def _esc(s) -> str:
    """Escape `<`, `>`, `&` so dynamic values don't confuse paraparser."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


def _build_main_signal_png(state: dict, segments: pd.DataFrame,
                           selected: int | None = None) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=160)
    ax.plot(t, a_vert, color="#233044", lw=0.7)
    ax.plot(t, a_smooth, color="#e67e22", lw=1.1)
    for i, row in segments.iterrows():
        try:
            s = float(row["start_s"]); e = float(row["end_s"])
        except (TypeError, ValueError):
            continue
        color = RIDE_COLORS.get(str(row.get("type", "up")), "#777")
        is_sel = (selected is not None and int(i) == selected)
        ax.axvspan(s, e,
                   color=SELECTED_COLOR if is_sel else color,
                   alpha=0.32 if is_sel else 0.18, lw=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("a_vert (m/s²)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
    plt.close(fig)
    return buf.getvalue()


def _build_segment_page_pngs(
    state: dict, predictions: list[dict],
    t_lo: float, t_hi: float,
) -> tuple[bytes, bytes]:
    """Two PNGs for a segment page:

    1) Zoomed a_vert with the fitted trapezoid overlay (the 'match').
    2) Correlation panel ±30 s around the segment, with peak-status dots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b"", b""

    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    match = _find_matching_prediction(predictions, t_lo, t_hi)

    # --- (1) trapezoid overlay ---
    pad_s = 4.0
    mask = (t >= t_lo - pad_s) & (t <= t_hi + pad_s)
    fig1, ax1 = plt.subplots(figsize=(7.6, 2.7), dpi=160)
    ax1.plot(t[mask], a_vert[mask], color="#233044", lw=0.7, label="a_vert")
    ax1.plot(t[mask], a_smooth[mask], color="#e67e22", lw=1.0, label="smoothed")
    ax1.axhline(0, color="#aaa", lw=0.4, ls="--")
    ax1.axvspan(t_lo, t_hi, color=SELECTED_COLOR, alpha=0.08, lw=0)
    if match is not None:
        for lobe_key, color in (("lobe1", "#c0392b"), ("lobe2", "#8e44ad")):
            L = match.get(lobe_key) or {}
            try:
                W = float(L["half_width_s"]); f = float(L["frac_flat"])
                A = float(L["a_peak"]); t_c = float(L["t_c"])
            except (KeyError, TypeError, ValueError):
                continue
            tt = np.linspace(t_c - W, t_c + W, 200)
            yy = A * trapezoid_kernel(tt, t_c, W, f)
            ax1.plot(tt, yy, color=color, lw=1.8)
            ax1.scatter([t_c], [A], color=color, s=26, zorder=5,
                        edgecolor="#000", linewidth=0.4)
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("a (m/s²)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    fig1.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=160)
    plt.close(fig1)

    # --- (2) correlation panel ±30 s with peak colors ---
    t_min = float(t[0]); t_max = float(t[-1])
    wlo = max(t_min, t_lo - 30.0); whi = min(t_max, t_hi + 30.0)
    mask2 = (t >= wlo) & (t <= whi)
    pos_r2 = np.asarray(state["best_pos_r2"])
    neg_r2 = np.asarray(state["best_neg_r2"])
    pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
    neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)

    fig2, ax2 = plt.subplots(figsize=(7.6, 2.7), dpi=160)
    ax2.plot(t[mask2], pos_plot[mask2], color="#2980b9", lw=0.9, label="max R² (+)")
    ax2.plot(t[mask2], neg_plot[mask2], color="#c0392b", lw=0.9, label="max R² (−)")
    cfg = state.get("config")
    if cfg is not None:
        ax2.axhline(cfg.r2_peak_thresh, color="#888", lw=0.5, ls="--")
    ax2.axvspan(t_lo, t_hi, color=SELECTED_COLOR, alpha=0.08, lw=0)
    peaks_pos = find_local_maxima(pos_r2, t, wlo, whi)
    peaks_neg = find_local_maxima(neg_r2, t, wlo, whi)
    for sign, peaks, arr in ((+1, peaks_pos, pos_r2), (-1, peaks_neg, neg_r2)):
        for i in peaks:
            tag = classify_peak(state, i, sign, predictions)
            ax2.scatter([t[i]], [arr[i]],
                        color=PEAK_STATUS_COLORS.get(tag, "#000"),
                        s=30, zorder=5, edgecolor="#000", linewidth=0.3)
    ax2.set_ylim(0, 1.05); ax2.set_xlim(wlo, whi)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("R² (per sign)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=160)
    plt.close(fig2)

    return buf1.getvalue(), buf2.getvalue()


def _build_pdf(
    loaded: LoadedSignal,
    state: dict,
    predictions: list[dict],
    segments: pd.DataFrame,
    rows: list[dict],
) -> bytes:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    heb_font = _maybe_register_hebrew_font()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="Boutique Pipeline — Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="HebTitle", parent=styles["Title"],
        fontName=heb_font, fontSize=22, alignment=TA_RIGHT,
        textColor=rl_colors.HexColor("#0f2a4a"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        name="HebH2", parent=styles["Heading2"],
        fontName=heb_font, fontSize=14, alignment=TA_RIGHT,
        textColor=rl_colors.HexColor("#1f6feb"),
    )
    body_style = ParagraphStyle(
        name="HebBody", parent=styles["BodyText"],
        fontName=heb_font, fontSize=10, alignment=TA_RIGHT, leading=14,
    )

    def P(text: str, style=body_style) -> Paragraph:
        return Paragraph(_rtl(text), style)

    story: list = []

    # --- Cover page ---
    story.append(P(HEB["report_title"], title_style))
    story.append(Spacer(1, 0.2 * cm))
    meta_rows = [
        (HEB["source"],      loaded.source),
        (HEB["generated"],   _utcnow_iso()),
        (HEB["samples"],     str(loaded.meta.get("samples", "?"))),
        (HEB["sample_rate"], str(loaded.meta.get("sample_rate", "?"))),
    ]
    for k, v in loaded.meta.items():
        if k in ("samples", "sample_rate"):
            continue
        label = HEB.get(k, k)
        meta_rows.append((label, str(v)))
    for label, val in meta_rows:
        story.append(P(f"<b>{_esc(label)}:</b> {_esc(val)}"))

    # Summary metrics + overall signal.
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    total_dh = float(df["delta_height_m"].dropna().sum()) if not df.empty else 0.0
    n_accepted = int(df["accepted"].sum()) if not df.empty else 0

    story.append(Spacer(1, 0.4 * cm))
    story.append(P(HEB["summary"], h2_style))
    story.append(P(f"{HEB['n_segments']}: <b>{len(df)}</b>"))
    story.append(P(f"{HEB['n_accepted']}: <b>{n_accepted}</b>"))
    story.append(P(f"{HEB['net_dh']}: <b>{total_dh:+.2f}</b>"))

    main_png = _build_main_signal_png(state, segments, selected=None)
    if main_png:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Image(io.BytesIO(main_png), width=17 * cm, height=6.6 * cm))

    # Overview table (kept on cover — per-segment pages follow).
    story.append(Spacer(1, 0.4 * cm))
    story.append(P(HEB["per_seg_table"], h2_style))
    header = [HEB["col_accepted"], HEB["col_quality"], HEB["col_ci"],
              HEB["col_dh"], HEB["col_dur"], HEB["col_end"],
              HEB["col_start"], HEB["col_type"], HEB["col_idx"]]
    data = [[_rtl(h) for h in header]]
    for r in rows:
        yn = HEB["yes"] if r["accepted"] else HEB["no"]
        rt_heb = HEB["up"] if r["type"] == "up" else HEB["down"]
        data.append([
            _rtl(yn),
            f"{r['quality_score']:.1f}" if np.isfinite(r['quality_score']) else "—",
            f"{r['ci_half_width']:.2f}" if np.isfinite(r['ci_half_width']) else "—",
            f"{r['delta_height_m']:+.2f}" if np.isfinite(r['delta_height_m']) else "—",
            f"{r['duration_s']:.1f}",
            f"{r['end_s']:.1f}",
            f"{r['start_s']:.1f}",
            _rtl(rt_heb),
            str(r["segment"]),
        ])
    tbl = Table(data, repeatRows=1, hAlign="RIGHT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0f2a4a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), heb_font),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#f5f7fb"), rl_colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.25, rl_colors.HexColor("#cfd4dc")),
    ]))
    story.append(tbl)

    # --- Per-segment pages ---
    for r in rows:
        story.append(PageBreak())
        rt_heb = HEB["up"] if r["type"] == "up" else HEB["down"]
        header_txt = (
            f"{HEB['segment_page']} #{r['segment']} — {rt_heb}  "
            f"({r['start_s']:.1f}–{r['end_s']:.1f} ש')"
        )
        story.append(P(header_txt, title_style))
        story.append(Spacer(1, 0.2 * cm))

        # Key metrics as a right-aligned list.
        dh_str = (f"{r['delta_height_m']:+.2f}"
                  if np.isfinite(r['delta_height_m']) else "—")
        ci_str = (f"{r['ci_half_width']:.2f}"
                  if np.isfinite(r['ci_half_width']) else "—")
        q_str = (f"{r['quality_score']:.1f}"
                 if np.isfinite(r['quality_score']) else "—")
        yn = HEB["yes"] if r["accepted"] else HEB["no"]
        story.append(P(f"<b>{HEB['col_dh']}:</b> {dh_str}"))
        story.append(P(f"<b>{HEB['col_ci']}:</b> {ci_str}"))
        story.append(P(f"<b>{HEB['col_quality']}:</b> {q_str}"))
        story.append(P(f"<b>{HEB['col_accepted']}:</b> {yn}"))
        if r.get("reject_reason"):
            story.append(
                P(f"<b>{HEB['reject_reason']}:</b> {_esc(r['reject_reason'])}")
            )

        story.append(Spacer(1, 0.3 * cm))
        trap_png, corr_png = _build_segment_page_pngs(
            state, predictions, float(r["start_s"]), float(r["end_s"]),
        )
        if trap_png:
            story.append(P(HEB["trap_heading"], h2_style))
            story.append(Image(io.BytesIO(trap_png),
                               width=17 * cm, height=6.0 * cm))
        if corr_png:
            story.append(Spacer(1, 0.2 * cm))
            story.append(P(HEB["corr_heading"], h2_style))
            story.append(Image(io.BytesIO(corr_png),
                               width=17 * cm, height=6.0 * cm))
            story.append(Spacer(1, 0.1 * cm))
            # Status legend as a plain line (the colors show in the PNG).
            chips = "  ".join(f"● {t}" for t in PEAK_STATUS_COLORS)
            story.append(P(f"{HEB['status_legend']}: {chips}"))

    # --- How to read (last page) ---
    story.append(PageBreak())
    story.append(P(HEB["how_to_read"], h2_style))
    story.append(P(HEB["how_to_read_body"]))

    doc.build(story)
    return buf.getvalue()


def _render_report() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    rows = st.session_state.get("prediction_rows") or []
    segments = _valid_segments(st.session_state.get("segments_df"))
    state = st.session_state.get("detector_state")
    predictions = st.session_state.get("predictions") or []
    if loaded is None or not rows or state is None:
        st.warning("Complete earlier steps first.")
        if st.button("← Back"):
            _goto(STEP_PREDICT)
        return

    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 5</span>'
        '<h1>Export (PDF, Hebrew)</h1>'
        '<p>One page per segment: signal with trapezoid template, '
        '±30 s correlation window with color-coded peak status, and a '
        'segment summary.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        pdf_bytes = _build_pdf(loaded, state, predictions, segments, rows)
    except Exception as e:
        st.error(f"PDF build failed: {type(e).__name__}: {e}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    st.download_button(
        "Download Summary PDF",
        data=pdf_bytes,
        file_name=f"boutique_report_{stamp}.pdf",
        mime="application/pdf",
    )

    with st.expander("Preview PDF inline", expanded=True):
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="720" style="border:1px solid #e6e9ef; '
            f'border-radius:10px;"></iframe>',
            unsafe_allow_html=True,
        )

    st.divider()
    c1, _, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Back"):
            _goto(STEP_PREDICT)
    with c3:
        if st.button("Start over"):
            for k in ("loaded", "detector_state", "predictions",
                      "segments_df", "prediction_rows",
                      "selected_segment", "predict_selected"):
                st.session_state[k] = None
            _goto(STEP_HOWTO)


# ---------------------------------------------------------------------------
# Sidebar + entry point
# ---------------------------------------------------------------------------

def _sidebar() -> None:
    """Minimal always-visible header. On the segmentation step the main
    content of the sidebar is the per-segment list, drawn from inside
    :func:`_render_segmentation` — this header just shows the title and
    a compact step indicator, and a reset button lives at the bottom.
    """
    current = st.session_state["step"]
    # Both segmentation and prediction host their own segment list in
    # the sidebar, so collapse the main step indicator on those steps
    # to give the list the full vertical space.
    list_owns_sidebar = current in (STEP_SEGMENT, STEP_PREDICT)

    st.sidebar.markdown("## Boutique Pipeline")
    st.sidebar.caption("Elevator Vertical Distance")
    st.sidebar.markdown("---")

    if list_owns_sidebar:
        st.sidebar.caption(f"**{STEP_LABELS[current]}**")
    else:
        for step in (STEP_HOWTO, STEP_DATA, STEP_SEGMENT, STEP_PREDICT, STEP_REPORT):
            prefix = "●" if step == current else "○"
            st.sidebar.markdown(f"**{prefix}  {STEP_LABELS[step]}**")
        st.sidebar.markdown("---")
        loaded = st.session_state.get("loaded")
        if loaded is not None:
            st.sidebar.caption(f"Loaded: {loaded.source}")
        if st.sidebar.button("Reset session"):
            for k in list(st.session_state.keys()):
                del st.session_state[k]
            st.rerun()


def main() -> None:
    st.set_page_config(
        page_title=APP_TITLE, page_icon="🏙️", layout="wide",
        initial_sidebar_state="expanded",
    )
    st.markdown(_CUSTOM_CSS, unsafe_allow_html=True)
    _init_state()
    _sidebar()
    step = st.session_state["step"]
    if step == STEP_HOWTO:
        _render_howto()
    elif step == STEP_DATA:
        _render_data()
    elif step == STEP_SEGMENT:
        _render_segmentation()
    elif step == STEP_PREDICT:
        _render_prediction()
    elif step == STEP_REPORT:
        _render_report()
    else:
        st.session_state["step"] = STEP_HOWTO
        st.rerun()


if __name__ == "__main__":
    main()
