"""Step 1 — How to use (intro / landing page)."""
from __future__ import annotations

import streamlit as st

from .common import APP_TITLE, STEP_DATA, goto


def render() -> None:
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
            goto(STEP_DATA)
