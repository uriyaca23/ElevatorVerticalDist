"""Step 2 — Data input.

Two-stage page: a chooser ("phone DB" / "upload file") and then the
selected form. The phone-DB form requires a Phone ID that passes
:func:`validate_phone_id`. The file form takes a CSV or Excel file in
the canonical ACC schema (``timestamp_ms``, ``x``, ``y``, ``z`` — same
shape as the files under ``src/data/structuredData/data/<exp>/ACC.csv``);
when the uploaded file uses different column names the user maps them,
and the time column is auto-normalised to int64 milliseconds.
"""
from __future__ import annotations

from datetime import datetime, time as dtime

import numpy as np
import pandas as pd
import streamlit as st

from .common import (
    LoadedSignal,
    PhoneType,
    STEP_HOWTO,
    STEP_SEGMENT,
    enrich_loaded,
    goto,
    loadDataFromS3,
    reset_downstream_state,
    split_acc_into_parts,
)
from src.data.loader import detect_valid_intervals


# Canonical ACC schema, matching src/data/structuredData/data/<exp>/ACC.csv.
CANONICAL_COLUMNS = ("timestamp_ms", "x", "y", "z")
_MIN_SAMPLES = 10


def _detect_time_unit(ts: np.ndarray) -> tuple[str, np.ndarray]:
    """Identify how the user encoded time and convert to int64 ms.

    Heuristic, based on the magnitude of the largest timestamp:

    * ``> 1e12`` — Unix epoch milliseconds (a ms past year 2001 is
      already 13 digits, so anything this large is unambiguously ms).
    * ``> 1e9``  — Unix epoch seconds (10-digit values like
      ``1774373973``).
    * otherwise — relative time from the start of the recording. Span
      ``< 1e4`` is treated as seconds (a 5-minute capture spans ~300
      units); anything wider is already in ms.
    """
    if ts.size == 0:
        raise ValueError("Time column is empty after dropping non-numeric rows.")
    mx = float(np.nanmax(ts))
    mn = float(np.nanmin(ts))
    span = mx - mn
    if mx > 1e12:
        return "Unix epoch milliseconds", ts.astype("int64")
    if mx > 1e9:
        return "Unix epoch seconds", (ts * 1000.0).astype("int64")
    if span < 1e4:
        return "relative seconds (from t=0)", (ts * 1000.0).astype("int64")
    return "relative milliseconds (from t=0)", ts.astype("int64")


def _csv_to_signal(
    df: pd.DataFrame, mapping: dict[str, str], filename: str,
) -> tuple[LoadedSignal, str]:
    """Reduce a user-provided CSV to the canonical ACC schema.

    ``mapping`` maps each canonical column (``timestamp_ms``, ``x``,
    ``y``, ``z``) to the source column the user picked. Returns the
    finished :class:`LoadedSignal` together with the detected time-unit
    label so the caller can echo it back to the user.
    """
    cols = {k: mapping[k] for k in CANONICAL_COLUMNS}
    if len(set(cols.values())) != len(cols):
        raise ValueError(
            "The same source column was mapped to more than one canonical "
            "column. Pick a different column for each of time / x / y / z."
        )
    missing = [c for c in cols.values() if c not in df.columns]
    if missing:
        raise ValueError(f"Columns not found in the CSV: {missing}")

    t_raw = pd.to_numeric(df[cols["timestamp_ms"]], errors="coerce").to_numpy(dtype=float)
    x_raw = pd.to_numeric(df[cols["x"]],            errors="coerce").to_numpy(dtype=float)
    y_raw = pd.to_numeric(df[cols["y"]],            errors="coerce").to_numpy(dtype=float)
    z_raw = pd.to_numeric(df[cols["z"]],            errors="coerce").to_numpy(dtype=float)

    good = (np.isfinite(t_raw) & np.isfinite(x_raw)
            & np.isfinite(y_raw) & np.isfinite(z_raw))
    n_dropped = int((~good).sum())
    t_raw = t_raw[good]; x_raw = x_raw[good]
    y_raw = y_raw[good]; z_raw = z_raw[good]

    if t_raw.size < _MIN_SAMPLES:
        raise ValueError(
            f"Not enough numeric rows after cleaning "
            f"(need ≥{_MIN_SAMPLES}, got {t_raw.size}). Check that the "
            "columns you mapped actually hold numbers and that the file "
            "isn't mostly blank."
        )

    time_label, ts_ms = _detect_time_unit(t_raw)

    order = np.argsort(ts_ms, kind="stable")
    out_of_order = int(np.sum(np.diff(ts_ms) < 0))
    ts_ms = ts_ms[order]
    x_raw = x_raw[order]; y_raw = y_raw[order]; z_raw = z_raw[order]

    dup_mask = np.concatenate([[False], np.diff(ts_ms) == 0])
    n_dups = int(dup_mask.sum())
    if n_dups:
        keep = ~dup_mask
        ts_ms = ts_ms[keep]; x_raw = x_raw[keep]
        y_raw = y_raw[keep]; z_raw = z_raw[keep]

    if ts_ms.size < _MIN_SAMPLES:
        raise ValueError(
            f"Only {ts_ms.size} samples left after deduping repeated "
            f"timestamps (need ≥{_MIN_SAMPLES}). The time column may have "
            "many repeated values — check that you mapped the right column."
        )

    n = ts_ms.size
    span_s = float(ts_ms[-1] - ts_ms[0]) / 1000.0
    if span_s <= 0:
        raise ValueError(
            "Timestamps span zero seconds — every sample carries the same "
            "time value. Map the time column to a non-constant column."
        )
    fs_hz = (n - 1) / span_s

    acc = pd.DataFrame({
        "timestamp_ms": ts_ms,
        "x": x_raw, "y": y_raw, "z": z_raw,
    })
    notes: list[str] = []
    if n_dropped:
        notes.append(f"dropped {n_dropped} non-numeric rows")
    if out_of_order:
        notes.append(f"sorted {out_of_order} out-of-order samples")
    if n_dups:
        notes.append(f"removed {n_dups} duplicate timestamps")

    # Gap detection: any consecutive-sample gap > GAP_THRESHOLD_S splits the
    # signal into separate valid intervals; downstream UI flags the gaps as
    # "no data" and segmenter/predictor only run inside valid intervals.
    valid_intervals = detect_valid_intervals(ts_ms)
    if len(valid_intervals) > 1:
        notes.append(f"detected {len(valid_intervals)} valid intervals (gaps present)")
    acc_parts = split_acc_into_parts(acc, valid_intervals)

    return LoadedSignal(
        acc=acc, source=f"File · {filename}",
        meta={
            "filename":    filename,
            "samples":     int(n),
            "sample_rate": f"{fs_hz:.1f} Hz",
            "time_format": time_label,
            "time_column": cols["timestamp_ms"],
            "x_column":    cols["x"],
            "y_column":    cols["y"],
            "z_column":    cols["z"],
            "notes":       "; ".join(notes) if notes else "",
            "valid_intervals_count": len(valid_intervals),
        },
        valid_intervals=valid_intervals,
        acc_parts=acc_parts,
    ), time_label


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
            '<b>📂 Upload a CSV / Excel file</b><br>'
            'Bring your own recording in the canonical ACC schema '
            '(<code>timestamp_ms</code>, <code>x</code>, <code>y</code>, '
            '<code>z</code>) — or any 4-column file and map the columns '
            'on the next page.'
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
                # Post-load: detect valid intervals and split into gap-free
                # parts. Kept out of loadDataFromS3 itself so the DB loader
                # stays a pure fetcher.
                loaded = enrich_loaded(loaded)
            st.session_state["loaded"] = loaded
            reset_downstream_state()
            goto(STEP_SEGMENT)


def _read_tabular_upload(uploaded) -> pd.DataFrame:
    """Pick the right pandas reader based on the upload's extension.

    Streamlit hands us an :class:`UploadedFile` (a buffer with a
    ``.name`` attribute). The reader is chosen by suffix — pandas needs
    different functions for CSV vs Excel, and Excel additionally needs
    ``openpyxl`` (xlsx) or ``xlrd`` (xls) installed.
    """
    name = (uploaded.name or "").lower()
    if name.endswith(".csv"):
        return pd.read_csv(uploaded)
    if name.endswith(".xlsx") or name.endswith(".xls"):
        return pd.read_excel(uploaded)
    raise ValueError(
        f"Unsupported file type: {uploaded.name!r}. "
        "Use a .csv, .xlsx, or .xls file."
    )


def _render_file_form() -> None:
    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 2 · Upload</span>'
        '<h1>Upload a CSV / Excel file (ACC schema)</h1>'
        '<p>Provide a file in the canonical accelerometer schema, or '
        'any 4-column file and map the columns below.</p>'
        '</div>',
        unsafe_allow_html=True,
    )
    _render_change_mode_link("file")

    st.markdown(
        '<div class="info-block"><b>Required file format</b><br>'
        'A CSV (.csv) or Excel sheet (.xlsx / .xls) with a header row '
        'and four numeric columns:'
        '<ul style="margin:0.3rem 0 0 1.2rem">'
        '<li><code>timestamp_ms</code> — Unix epoch in milliseconds. '
        'Epoch seconds and relative time (seconds or milliseconds from '
        'the start of the recording) are also accepted; the form '
        'auto-detects the unit and converts on the fly.</li>'
        '<li><code>x</code>, <code>y</code>, <code>z</code> — '
        'accelerometer axes in m/s², gravity included (the canonical '
        'files store the raw IMU reading — gravity sits on whichever '
        'axis the phone was held along, usually z ≈ 9.8).</li>'
        '</ul>'
        'Reference file: <code>src/data/structuredData/data/&lt;exp&gt;/ACC.csv</code>. '
        'If your file uses different column names, map them with the '
        'dropdowns that appear after upload.'
        '</div>',
        unsafe_allow_html=True,
    )

    uploaded = st.file_uploader(
        "Upload .csv, .xlsx, or .xls",
        type=["csv", "xlsx", "xls"], key="file_uploader",
    )

    file_df: pd.DataFrame | None = None
    if uploaded is not None:
        try:
            file_df = _read_tabular_upload(uploaded)
        except Exception as e:
            st.error(f"Could not read file: {type(e).__name__}: {e}")
            file_df = None
        if file_df is not None and file_df.empty:
            st.error("The uploaded file has no rows.")
            file_df = None
        if file_df is not None and len(file_df.columns) < 4:
            st.error(
                f"The file needs at least 4 columns (time, x, y, z) — "
                f"got {len(file_df.columns)}: {list(file_df.columns)}"
            )
            file_df = None

    mapping: dict[str, str] | None = None
    if file_df is not None:
        cols = list(file_df.columns)
        canonical = all(c in cols for c in CANONICAL_COLUMNS)
        if canonical:
            st.success(
                "Canonical ACC schema detected — using "
                "`timestamp_ms`, `x`, `y`, `z` directly. You can override "
                "the mapping below if needed."
            )

        st.markdown("**Map source columns → canonical schema**")

        def _default(canon: str) -> int:
            return cols.index(canon) if canon in cols else 0

        cc1, cc2, cc3, cc4 = st.columns(4)
        with cc1:
            t_col = st.selectbox("Time column", cols,
                                 index=_default("timestamp_ms"),
                                 key="map_time_col")
        with cc2:
            x_col = st.selectbox("X column", cols,
                                 index=_default("x"), key="map_x_col")
        with cc3:
            y_col = st.selectbox("Y column", cols,
                                 index=_default("y"), key="map_y_col")
        with cc4:
            z_col = st.selectbox("Z column", cols,
                                 index=_default("z"), key="map_z_col")

        picked = [t_col, x_col, y_col, z_col]
        if len(set(picked)) != len(picked):
            st.error(
                "Each canonical column must map to a different source "
                f"column — currently: {picked}"
            )
        else:
            mapping = {
                "timestamp_ms": t_col, "x": x_col, "y": y_col, "z": z_col,
            }

        st.markdown("**Preview (first 12 rows)**")
        st.dataframe(file_df.head(12), use_container_width=True, height=220)

    st.divider()
    c_back, c_next, _ = st.columns([1.0, 1.2, 3.0])
    with c_back:
        if st.button("← Back"):
            goto(STEP_HOWTO)
    with c_next:
        ready = file_df is not None and mapping is not None
        if st.button("Next →", type="primary",
                     key="btn_file_next", disabled=not ready):
            try:
                loaded, time_label = _csv_to_signal(
                    file_df, mapping, uploaded.name,
                )
            except Exception as e:
                st.error(f"Load failed: {type(e).__name__}: {e}")
                return
            st.session_state["loaded"] = loaded
            reset_downstream_state()
            st.success(
                f"Loaded {loaded.meta['samples']} samples at "
                f"{loaded.meta['sample_rate']} (time format: {time_label})."
            )
            goto(STEP_SEGMENT)


def render() -> None:
    mode = st.session_state.get("data_input_mode")
    if mode == "phone":
        _render_phone_form()
    elif mode == "file":
        _render_file_form()
    else:
        _render_mode_picker()
