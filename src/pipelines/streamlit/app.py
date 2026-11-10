"""Wizard dispatcher for the Boutique-Pipeline Streamlit front-end."""
from __future__ import annotations

import streamlit as st

from . import step1_howto, step2_data, step3_segmentation, step4_prediction, step5_report
from .common import (
    APP_TITLE,
    CUSTOM_CSS,
    STEP_DATA,
    STEP_HOWTO,
    STEP_LABELS,
    STEP_PREDICT,
    STEP_REPORT,
    STEP_SEGMENT,
    init_state,
)

_STEP_RENDERERS = {
    STEP_HOWTO:   step1_howto.render,
    STEP_DATA:    step2_data.render,
    STEP_SEGMENT: step3_segmentation.render,
    STEP_PREDICT: step4_prediction.render,
    STEP_REPORT:  step5_report.render,
}


def _sidebar() -> None:
    """Minimal always-visible header. On the segmentation and prediction
    steps the sidebar's main content is the per-segment list, drawn from
    inside those step renderers — this header just shows the title and a
    compact step indicator, with a reset button at the bottom.
    """
    current = st.session_state["step"]
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
    st.markdown(CUSTOM_CSS, unsafe_allow_html=True)
    init_state()
    _sidebar()
    step = st.session_state["step"]
    renderer = _STEP_RENDERERS.get(step)
    if renderer is None:
        st.session_state["step"] = STEP_HOWTO
        st.rerun()
    else:
        renderer()
