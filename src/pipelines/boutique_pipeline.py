"""Boutique Pipeline — Streamlit front-end (entry point).

The implementation lives in :mod:`src.pipelines.streamlit`, split into a
shared ``common`` module and one ``stepN_*.py`` module per wizard step.
This file makes every invocation work:

    venv/bin/python src/pipelines/boutique_pipeline.py
    venv/bin/python -m src.pipelines.boutique_pipeline
    venv/bin/python -m streamlit run src/pipelines/boutique_pipeline.py

When executed under plain Python (no Streamlit runtime), it re-execs
itself through ``python -m streamlit run`` so the server actually starts.
"""
from __future__ import annotations

import sys
from pathlib import Path

# ``src/pipelines/`` contains a package literally named ``streamlit`` (the
# wizard implementation). When this file is run directly, Python puts that
# directory first on sys.path, so ``import streamlit`` would resolve to the
# wizard package instead of the real library — drop it before importing
# anything, then make repo-root ``src.*`` imports resolve.
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[1]
sys.path[:] = [
    p for p in sys.path
    if p and Path(p).resolve() != _THIS_DIR
]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.pipelines.streamlit import main  # noqa: E402


def _under_streamlit_runtime() -> bool:
    """True when this script is being executed by the Streamlit runner."""
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx
    except Exception:  # pragma: no cover — very old/absent streamlit
        return False
    return get_script_run_ctx(suppress_warning=True) is not None


if __name__ == "__main__":
    if _under_streamlit_runtime():
        main()
    else:
        # Bare-python launch: hand off to the Streamlit CLI so a real
        # server starts (st.* calls are no-ops without the runtime).
        import subprocess

        raise SystemExit(subprocess.run(
            [sys.executable, "-m", "streamlit", "run", str(Path(__file__).resolve()),
             *sys.argv[1:]],
            cwd=str(_REPO_ROOT),
        ).returncode)
