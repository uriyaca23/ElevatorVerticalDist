"""Capture-only entry point for the Boutique Pipeline.

Reads a 4-column CSV (``time_s, ax_ms2, ay_ms2, az_ms2``) referenced by
the URL query param ``?csv=<scenario_name>``, packs it into a
``LoadedSignal`` with the real 3-axis accelerometer data, seeds the
wizard's session state at Step 3 (Segmentation), then hands off to the
production wizard's ``main()`` so the rendered UI looks identical to a
normal session.

Only used by ``tmp_boutique_capture/capture.py`` for screenshotting —
production users still go through Step 2's upload form.

Run:
    streamlit run tmp_boutique_capture/seed_runner.py -- --csv-dir <path>
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from src.data.loadFromDB import LoadedSignal  # noqa: E402
from src.pipelines.streamlit import main as wizard_main  # noqa: E402
from src.pipelines.streamlit.common import STEP_SEGMENT, init_state  # noqa: E402


CSV_DIR = REPO / "tmp_boutique_capture" / "csvs"


def _seed_from_csv(scenario: str) -> None:
    """Build a LoadedSignal from the 4-column capture CSV and stash it
    in session state under the key the wizard expects.
    """
    csv = CSV_DIR / f"{scenario}.csv"
    if not csv.exists():
        st.error(f"Capture CSV not found: {csv}")
        st.stop()

    df = pd.read_csv(csv)
    cols = {"time_s", "ax_ms2", "ay_ms2", "az_ms2"}
    missing = cols - set(df.columns)
    if missing:
        st.error(f"CSV missing columns: {missing}")
        st.stop()

    t_s = pd.to_numeric(df["time_s"], errors="coerce").to_numpy(dtype=float)
    ax = pd.to_numeric(df["ax_ms2"], errors="coerce").to_numpy(dtype=float)
    ay = pd.to_numeric(df["ay_ms2"], errors="coerce").to_numpy(dtype=float)
    az = pd.to_numeric(df["az_ms2"], errors="coerce").to_numpy(dtype=float)
    good = np.isfinite(t_s) & np.isfinite(ax) & np.isfinite(ay) & np.isfinite(az)
    t_s = t_s[good]; ax = ax[good]; ay = ay[good]; az = az[good]
    n = t_s.size
    ts_ms = (t_s * 1000.0).astype("int64")

    acc = pd.DataFrame({
        "timestamp_ms": ts_ms, "x": ax, "y": ay, "z": az,
    })
    dt = float(np.median(np.diff(ts_ms))) / 1000.0 if n > 1 else 0.02
    fs_est = 1.0 / dt if dt > 0 else float("nan")

    st.session_state["loaded"] = LoadedSignal(
        acc=acc, source=f"File · {scenario}.csv",
        meta={
            "filename":    f"{scenario}.csv",
            "samples":     int(n),
            "sample_rate": f"{fs_est:.1f} Hz",
            "scenario":    scenario,
        },
    )
    st.session_state["data_input_mode"] = "file"
    st.session_state["step"] = STEP_SEGMENT


def _query_param_scenario() -> str | None:
    qp = st.query_params
    return qp.get("csv", None)


# Seed-then-render: must run BEFORE wizard_main initialises state.
init_state()  # set defaults if missing
scenario = _query_param_scenario()
if scenario and st.session_state.get("loaded") is None:
    _seed_from_csv(scenario)

wizard_main()
