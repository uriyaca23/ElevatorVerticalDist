"""End-to-end pipeline (segmentation → prediction → Δh) evaluation.

Entry point: :mod:`src.pipelines.evaluate.evaluateOnData`. The runner
helpers (per-experiment driver, three-views builder, figure renderers)
live in :mod:`.runner` so callers can compose them outside the CLI.
"""

from .runner import (
    PipelineConfig,
    build_views,
    render_view_figures,
    run_experiment,
)

__all__ = [
    "PipelineConfig",
    "build_views",
    "render_view_figures",
    "run_experiment",
]
