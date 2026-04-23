"""Step 2 — Data input.

Two-stage page: a chooser ("phone DB" / "upload file") and then the
selected form. The phone-DB form requires a Phone ID that passes
:func:`validate_phone_id`.
"""
from __future__ import annotations

from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import streamlit as st

from .common import (
    GRAVITY_MS2,
    LoadedSignal,
    PhoneType,
    STEP_HOWTO,
    STEP_SEGMENT,
    goto,
    loadDataFromS3,
    reset_downstream_state,
)


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


def validate_phone_id(value: str) -> tuple[bool, str]:
    """Validate a Phone ID. Returns ``(is_valid, error_message)``.

    TODO(later): replace this stub with the real validator (e.g. a
    regex against the team's device-id scheme, or a DB lookup). For
    now we only accept a non-empty digit string so the field is at
    least sanity-checked before we hit S3.
    """
    s = (value or "").strip()
    if not s:
        return False, "Phone ID is required."
    if not s.isdigit():
        return False, "Phone ID must be an integer (digits only)."
    return True, ""


def _render_mode_picker() -> None:
    """Two big chooser buttons — sets ``data_input_mode`` then reruns."""
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 2</span>'
        '<h1>Load a signal</h1>'
        '<p>Pick how you want to bring data in. You can switch back '
        'to this picker at any time.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    st.markdown("### How do you want to enter the data?")
    c1, c2 = st.columns(2, gap="large")
    with c1:
        st.markdown(
            '<div class="info-block">'
            '<b>📱 Pull from phone (S3)</b><br>'
            'Fetch a recording from the experiment DB by phone type, '
            'phone ID and a time window.'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Use phone DB", type="primary",
                     use_container_width=True, key="pick_mode_phone"):
            st.session_state["data_input_mode"] = "phone"
            st.rerun()
    with c2:
        st.markdown(
            '<div class="info-block">'
            '<b>📂 Upload an Excel / CSV file</b><br>'
            'Bring your own recording. Needs a time column and a '
            'vertical-acceleration column.'
            '</div>',
            unsafe_allow_html=True,
        )
        if st.button("Upload a file", type="primary",
                     use_container_width=True, key="pick_mode_file"):
            st.session_state["data_input_mode"] = "file"
            st.rerun()

    st.divider()
    if st.button("← Back"):
        goto(STEP_HOWTO)


def _render_change_mode_link(current: str) -> None:
    """Small 'change input source' affordance shown above the form."""
    other = "file" if current == "phone" else "phone"
    other_label = ("Upload a file" if other == "file"
                   else "Use phone DB")
    cols = st.columns([4, 1])
    with cols[1]:
        if st.button(f"↺ {other_label}", key="change_input_mode"):
            st.session_state["data_input_mode"] = other
            st.rerun()


def _render_phone_form() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 2 · Phone DB</span>'
        '<h1>Pull a signal from the phone DB</h1>'
        '<p>Enter the phone, the phone ID and a time window, then press '
        'Fetch.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_change_mode_link("phone")

    c1, c2, c3 = st.columns([1.2, 1.4, 1.4])
    with c1:
        phone = st.selectbox("Phone type",
                             [p.value for p in PhoneType], key="db_phone")
        phone_id_raw = st.text_input(
            "Phone ID *",
            placeholder="digits only — e.g. 123456",
            key="db_phone_id",
            help="Required. Currently must be an integer (digits only). "
                 "TODO: replace with the team's real device-id validator.",
        )
        id_ok, id_err = validate_phone_id(phone_id_raw)
        if phone_id_raw and not id_ok:
            st.caption(f":red[{id_err}]")
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
    c_back, c_next, _ = st.columns([1.0, 1.2, 3.0])
    with c_back:
        if st.button("← Back"):
            goto(STEP_HOWTO)
    with c_next:
        ready = bool(id_ok and valid_window)
        if st.button("Fetch →", type="primary",
                     key="btn_phone_fetch", disabled=not ready):
            with st.spinner("Fetching…"):
                loaded = loadDataFromS3(
                    PhoneType(phone), phone_id_raw.strip(), t_start, t_end,
                )
            st.session_state["loaded"] = loaded
            reset_downstream_state()
            goto(STEP_SEGMENT)


def _render_file_form() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 2 · Upload</span>'
        '<h1>Upload an Excel / CSV file</h1>'
        '<p>Pick a tabular file with a time column and a vertical-'
        'acceleration column.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_change_mode_link("file")

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
    c_back, c_next, _ = st.columns([1.0, 1.2, 3.0])
    with c_back:
        if st.button("← Back"):
            goto(STEP_HOWTO)
    with c_next:
        ready = (file_df is not None
                 and time_col is not None and acc_col is not None)
        if st.button("Next →", type="primary",
                     key="btn_file_next", disabled=not ready):
            try:
                loaded = _tabular_to_signal(
                    file_df, time_col, acc_col, uploaded.name,
                )
            except Exception as e:
                st.error(f"Load failed: {type(e).__name__}: {e}")
                return
            st.session_state["loaded"] = loaded
            reset_downstream_state()
            goto(STEP_SEGMENT)


def render() -> None:
    mode = st.session_state.get("data_input_mode")
    if mode == "phone":
        _render_phone_form()
    elif mode == "file":
        _render_file_form()
    else:
        _render_mode_picker()
