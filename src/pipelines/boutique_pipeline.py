"""Boutique Pipeline — Streamlit front-end (entry point).

The implementation lives in :mod:`src.pipelines.streamlit`, split into a
shared ``common`` module and one ``stepN_*.py`` module per wizard step.
This file just sets up ``sys.path`` so the package's ``src.*`` imports
resolve when Streamlit invokes the script directly, then hands off to
:func:`src.pipelines.streamlit.main`.

Run:
    venv/bin/python -m streamlit run src/pipelines/boutique_pipeline.py
"""
from __future__ import annotations

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipelines.streamlit import main  # noqa: E402


if __name__ == "__main__":
    main()
