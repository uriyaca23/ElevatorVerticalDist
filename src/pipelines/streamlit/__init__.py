"""Boutique-Pipeline Streamlit front-end, split per step.

The ``main()`` entry point lives in :mod:`.app`; the individual step
renderers live in ``stepN_*.py`` modules; cross-step scaffolding (state
helpers, palette, sidebar fragments, CSS, …) lives in :mod:`.common`.
"""
from __future__ import annotations

from .app import main

__all__ = ["main"]
