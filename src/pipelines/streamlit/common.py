"""Shared scaffolding for the Boutique-Pipeline Streamlit wizard.

Holds everything that more than one step touches: imports of the core
project modules, palette / step constants, the custom CSS, session-state
helpers, and the cross-step UI fragments (segment sidebar, trapezoid
parameter cards, peak-status legend, helpers for picking and validating
detector predictions).

Single-step helpers stay in their own ``stepN_*.py`` module to keep this
file focused on truly shared scaffolding.
"""
from __future__ import annotations

from datetime import datetime, timezone

import numpy as np
import pandas as pd
import streamlit as st

# Unified loader entry point lives in src/data/load_data.py so non-UI
# tools (gt_editor, segmentation/editor) can import the same helpers
# without depending on the streamlit package.
from src.data.load_data import (
    LoadedSignal,
    enrich_loaded,
    load_data,
    split_acc_into_parts,
)
from src.data.loadFromDB import PhoneType, loadDataFromS3
from src.segmentation.algorithms.accelerometer_only.template_match.check_grid_across_signal import (
    detect as _detect,
)
from src.segmentation.algorithms.accelerometer_only.template_match.fit_elevator_parameters.common import (
    trapezoid_kernel,
)

# Display-only helpers that operate on the detector state dict (heatmap
# rasters, signed-R² peak classification, simple local-maxima finder).
# These are pure-numpy and do not run any detection — the UI is allowed
# to call them client-side. The actual detection / prediction logic now
# lives behind the HTTP API in ``api/``.
heatmap_at = _detect.heatmap_at
classify_peak = _detect.classify_peak
find_local_maxima = _detect.find_local_maxima

# Algorithm short ids accepted by /predict. Keep in sync with
# ``api/main.py::_ACCEL_ALGO_MAP`` — the boutique UI labels them and
# their colours below.
ALGO_TRAP = "trap"
ALGO_ZUPT = "zupt"


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

APP_TITLE = "Elevator Vertical Distance — Boutique Pipeline"
GRAVITY_MS2 = 9.80665
RIDE_COLORS = {"up": "#2ca02c", "down": "#b54a9b", "outside": "#9aa0a6"}
SELECTED_COLOR = "#e67e22"

# Accelerometer-only prediction algorithms run side-by-side on the Predict
# step. The first entry is the "primary" — its rows feed the sidebar, the
# single-Δh column, and the PDF report (so existing paths keep working).
# Short ids must match those accepted by the API's /predict endpoint.
ACCEL_ALGOS: list[tuple[str, str, str]] = [
    # (short id, human label, colour)
    (ALGO_TRAP, "Trapezoid pulse-pair",        "#1f6feb"),
    (ALGO_ZUPT, "ZUPT (zero-velocity update)", "#27ae60"),
]
PRIMARY_ALGO_ID = ACCEL_ALGOS[0][0]

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

CUSTOM_CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

html, body, [class*="css"]  {
    font-family: 'Inter', -apple-system, BlinkMacSystemFont, sans-serif;
}
code, pre, kbd { font-family: 'JetBrains Mono', monospace; }

/* Browser full-screen / Docker-served iframes can leave the app root
   with a fixed height + overflow:hidden, which traps the page and
   blocks scrolling. Force the scroll containers back to their natural
   "grow with content + scroll the overflow" behaviour. */
html, body { height: auto !important; overflow-y: auto !important; }
[data-testid="stAppViewContainer"],
[data-testid="stApp"] {
    height: auto !important;
    min-height: 100vh;
    overflow-y: auto !important;
}
section.main, section[data-testid="stMain"] {
    overflow-y: auto !important;
    height: auto !important;
}

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
    background: #ffffff; border: 1px solid #d6e1f5;
    border-left: 3px solid #1f6feb;
    padding: 0.85rem 1.1rem; border-radius: 8px; margin: 0.4rem 0;
    font-size: 0.93rem; color: #1a2436; line-height: 1.55;
    box-shadow: 0 1px 2px rgba(16,24,40,0.04);
}
.info-block b   { color: #0f2a4a; }
.info-block i   { color: #233044; }
.info-block code {
    background: #eef2f7; color: #0f2a4a;
    padding: 0.05rem 0.32rem; border-radius: 4px;
    font-size: 0.85em;
}
.info-block ul  { color: #1a2436; }
.info-block li  { margin: 0.18rem 0; }

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

def init_state() -> None:
    defaults = {
        "step":              STEP_HOWTO,
        "loaded":             None,
        "detector_state":     None,
        "predictions":        [],
        "segments_df":        None,
        "selected_segment":   0,
        "prediction_rows":    None,   # primary-algo rows (Trapezoid)
        "prediction_rows_by_algo": None,  # {algo_id: list[dict]}
        "predict_selected":   None,
        "data_input_mode":    None,   # "phone" | "file" | None (= picker)
        "pending_new_segment": False,
        # Per-segment manual trapezoid overrides — keyed by segment index
        # in segments_df. See effective_trapezoid_params() and the
        # step-3 override UI. Cleared whenever the detector is re-run or
        # the segments table is bulk-edited.
        "lobe_overrides":     {},
    }
    for k, v in defaults.items():
        st.session_state.setdefault(k, v)


def goto(step: int) -> None:
    st.session_state["step"] = step
    st.rerun()


def utcnow_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


# ---------------------------------------------------------------------------
# Time-axis helpers
# ---------------------------------------------------------------------------
#
# The detector's ``state['t']`` is in seconds from the start of the
# recording. Every chart in the wizard is more useful with the wall-clock
# time on the x-axis instead, so these helpers turn relative seconds into
# naive ``datetime64`` (in the system's local timezone) using
# ``state['t0_ms']`` as the epoch origin.

_LOCAL_TZ = datetime.now().astimezone().tzinfo

# Plotly d3-format hover string for a datetime x-axis. Shown in the
# tooltip when the user hovers any time-series chart.
HOVER_DATETIME_FMT = "%Y-%m-%d %H:%M:%S.%L"


def hover_time_template(value_template: str = "") -> str:
    """Plotly hovertemplate showing wall-clock + seconds-from-start.

    Mirrors the gt_editor's desktop readout (HH:MM:SS.mmm + offset
    seconds) so every chart in the wizard exposes both representations
    of time on hover. ``value_template`` is a d3-format snippet for the
    trace's y value, e.g. ``"a=%{y:.2f} m/s²"``. Callers must pass
    ``customdata`` shaped ``(N, 1)`` (relative seconds) when adding the
    trace; ``time_customdata`` builds that array.
    """
    body = (
        f"%{{x|{HOVER_DATETIME_FMT}}}<br>"
        "t=%{customdata[0]:.2f} s"
    )
    if value_template:
        body += "<br>" + value_template
    return body + "<extra></extra>"


def time_customdata(t_seconds) -> np.ndarray:
    """Reshape a 1-D seconds-from-start array into the ``(N, 1)`` shape
    Plotly expects when a hovertemplate references ``customdata[0]``.
    """
    arr = np.asarray(t_seconds, dtype=float)
    if arr.ndim == 0:
        arr = arr.reshape(1)
    return arr.reshape(-1, 1)


def _t0_ms(t0_ms: float | None) -> float:
    """Coerce ``state['t0_ms']`` to a finite number, falling back to 0.

    A bad / missing origin still produces sensible plots — the dates
    just sit at the start of the Unix epoch instead of the recording's
    real wall-clock time. The user-visible message is "no date info" via
    the 1970 ticks; nothing crashes.
    """
    if t0_ms is None:
        return 0.0
    try:
        v = float(t0_ms)
    except (TypeError, ValueError):
        return 0.0
    return v if np.isfinite(v) else 0.0


def to_datetime_array(t_s, t0_ms: float | None) -> np.ndarray:
    """Convert a relative-seconds array to local-time ``datetime64[ns]``.

    Suitable for direct use as a Plotly x-axis or a matplotlib datetime
    axis.
    """
    epoch_ms = _t0_ms(t0_ms) + np.asarray(t_s, dtype=float) * 1000.0
    dt_utc = pd.to_datetime(epoch_ms, unit="ms", utc=True)
    if _LOCAL_TZ is not None:
        dt_utc = dt_utc.tz_convert(_LOCAL_TZ)
    return dt_utc.tz_localize(None).values


def to_datetime(t_s: float, t0_ms: float | None) -> pd.Timestamp:
    """Scalar version of :func:`to_datetime_array` — single relative
    second to a ``pandas.Timestamp`` in local time. Plotly accepts the
    Timestamp directly anywhere a datetime is allowed.
    """
    arr = to_datetime_array(np.asarray([float(t_s)]), t0_ms)
    return pd.Timestamp(arr[0])


def reset_downstream_state() -> None:
    st.session_state["detector_state"] = None
    st.session_state["predictions"] = []
    st.session_state["segments_df"] = None
    st.session_state["selected_segment"] = 0
    st.session_state["prediction_rows"] = None
    st.session_state["prediction_rows_by_algo"] = None
    st.session_state["predict_selected"] = None


# ---------------------------------------------------------------------------
# Cross-step segment / prediction helpers
# ---------------------------------------------------------------------------

def valid_segments(df: pd.DataFrame | None) -> pd.DataFrame:
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


def find_matching_prediction(
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


# ---------------------------------------------------------------------------
# Cross-step UI fragments (sidebar lists, legend, trapezoid cards)
# ---------------------------------------------------------------------------

def peak_status_legend_html() -> str:
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


def render_trapezoid_params(prediction: dict | None) -> None:
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


def _segment_label(i: int, row: pd.Series, has_override: bool = False) -> str:
    rt = str(row.get("type", "up"))
    s = float(row["start_s"]); e = float(row["end_s"])
    # ✱ marks a segment whose trapezoid shape has been manually overridden
    # in step 3 — see ``lobe_overrides`` in session state. Two-space pad
    # keeps column widths stable for un-overridden rows.
    star = "✱ " if has_override else "  "
    return f"#{i:<2} {star}{rt:<4}  {s:6.1f}–{e:6.1f} s"


def effective_trapezoid_params(
    prediction: dict | None, override: dict | None,
) -> dict | None:
    """Return the per-lobe trapezoid params with an optional shape override.

    The returned dict matches the detector's ``prediction`` structure
    (``lobe1`` / ``lobe2`` with ``t_c``, ``half_width_s``, ``frac_flat``,
    ``a_peak``, ``r2_local``) so existing renderers consume it unchanged.

    With ``override`` ``None`` this is a thin copy of ``prediction``. With
    an override the shared ``(W, f, |A|)`` are taken from the override;
    each lobe keeps its own ``t_c`` (so inter-lobe duration — the main
    Δh driver — is preserved) and its own sign on ``a_peak``.
    """
    if prediction is None:
        return None
    l1 = dict(prediction.get("lobe1") or {})
    l2 = dict(prediction.get("lobe2") or {})
    if not override:
        return {"lobe1": l1, "lobe2": l2, "override_mode": None}
    try:
        W = float(override["W"])
        f = float(override["f"])
        abs_A = float(override["abs_A"])
    except (KeyError, TypeError, ValueError):
        return {"lobe1": l1, "lobe2": l2, "override_mode": None}

    def _sign(lobe: dict, default: float) -> float:
        a = lobe.get("a_peak")
        try:
            v = float(a)
        except (TypeError, ValueError):
            v = default
        return 1.0 if v >= 0 else -1.0

    sign1 = _sign(l1, 1.0)
    sign2 = _sign(l2, -1.0)
    l1["half_width_s"] = W; l1["frac_flat"] = f
    l1["a_peak"] = sign1 * abs_A
    l2["half_width_s"] = W; l2["frac_flat"] = f
    l2["a_peak"] = sign2 * abs_A
    return {
        "lobe1": l1, "lobe2": l2,
        "override_mode": str(override.get("mode", "manual")),
    }


def _default_new_segment_bounds(
    segments_df: pd.DataFrame,
) -> tuple[float, float]:
    """Default Start/End the add-segment form opens with.

    Picks "just after the last segment" so consecutive additions don't
    pile up on top of each other, falling back to 0–5 s when the list is
    still empty. The form is editable, so this is only a starting point.
    """
    if len(segments_df):
        last = segments_df.iloc[-1]
        new_lo = float(last["end_s"]) + 0.5
    else:
        new_lo = 0.0
    return new_lo, new_lo + 5.0


_NEW_SEG_KEYS = ("sb_new_lo", "sb_new_hi", "sb_new_type")


def _clear_new_segment_state() -> None:
    st.session_state["pending_new_segment"] = False
    for k in _NEW_SEG_KEYS:
        if k in st.session_state:
            del st.session_state[k]


def _render_add_segment_controls(segments_df: pd.DataFrame) -> None:
    """Sidebar control for adding a new segment.

    Opens an inline Start/End/Type form when the user clicks
    "+ Add segment", and only commits to ``segments_df`` after the
    Accept button is pressed. This replaces the earlier behaviour of
    inserting a default 5-second window at "last end + 0.5 s" the moment
    the button was clicked, which placed the new segment at a random
    spot and forced the user to immediately Edit it.
    """
    if st.session_state.get("pending_new_segment"):
        default_lo, default_hi = _default_new_segment_bounds(segments_df)
        st.sidebar.markdown("**New segment**")
        new_lo = st.sidebar.number_input(
            "Start (s)", value=default_lo, step=0.05, format="%.2f",
            key="sb_new_lo",
        )
        new_hi = st.sidebar.number_input(
            "End (s)", value=default_hi, step=0.05, format="%.2f",
            key="sb_new_hi",
        )
        new_type = st.sidebar.selectbox(
            "Type", options=["up", "down"], key="sb_new_type",
        )
        col_a, col_b = st.sidebar.columns(2)
        accept = col_a.button(
            "Accept", type="primary", use_container_width=True,
            key="sb_new_accept",
        )
        cancel = col_b.button(
            "Cancel", use_container_width=True, key="sb_new_cancel",
        )
        if accept:
            if new_hi <= new_lo:
                st.sidebar.error("End must be greater than start.")
            else:
                df = st.session_state.get("segments_df")
                if df is None:
                    df = pd.DataFrame(
                        columns=["type", "start_s", "end_s", "joint_r2"],
                    )
                new_row = {
                    "type":     new_type,
                    "start_s":  float(new_lo),
                    "end_s":    float(new_hi),
                    "joint_r2": np.nan,
                }
                df = pd.concat(
                    [df, pd.DataFrame([new_row])], ignore_index=True,
                )
                st.session_state["segments_df"] = df
                st.session_state["selected_segment"] = len(df) - 1
                _clear_new_segment_state()
                st.rerun()
        if cancel:
            _clear_new_segment_state()
            st.rerun()
        return

    if st.sidebar.button(
        "+  Add segment", use_container_width=True, key="sb_add_segment",
    ):
        st.session_state["pending_new_segment"] = True
        st.rerun()


def render_segment_sidebar(segments_df: pd.DataFrame, sel: int | None) -> None:
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
                # Remap the index-keyed overrides: drop the deleted row's
                # entry, and shift every later row down by 1 so the
                # remaining segments keep their overrides.
                ovs: dict[int, dict] = (
                    st.session_state.get("lobe_overrides") or {}
                )
                if ovs:
                    st.session_state["lobe_overrides"] = {
                        (k if k < sel else k - 1): v
                        for k, v in ovs.items() if k != sel
                    }
                st.rerun()

    st.sidebar.markdown("")
    _render_add_segment_controls(segments_df)

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


def render_predict_segment_sidebar(rows: list[dict], selected: int) -> None:
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
    overrides = st.session_state.get("lobe_overrides", {}) or {}

    def _label(seg_id: int) -> str:
        r = by_id[seg_id]
        dh = r["delta_height_m"]
        dh_str = f"Δh={dh:+.2f}m" if np.isfinite(dh) else "Δh=—"
        # ✱ marks segments with a manual trapezoid override staged for
        # this prediction. Two-space pad keeps un-overridden rows aligned.
        star = "✱ " if seg_id in overrides else "  "
        return (f"#{r['segment']:<2} {star}{r['type']:<4}  "
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
