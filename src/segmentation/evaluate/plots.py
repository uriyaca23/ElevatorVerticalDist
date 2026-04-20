"""Diagnostic plots for :func:`evaluate_algorithm`.

Kept separate from the numeric logic so the evaluator can run on
headless machines when ``out_dir`` is omitted.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.segmentation.algorithms.metrics import IntervalPredictionMetrics


def _cdf_plot(
    values: list[float], title: str, xlabel: str, out_path: Path,
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(6, 4))
    if values:
        xs = np.sort(np.asarray(values, dtype=float))
        ys = np.arange(1, len(xs) + 1) / len(xs)
        ax.plot(xs, ys, lw=1.5, color="tab:blue")
        ax.axhline(0.5, color="gray", lw=0.5, ls="--")
        ax.set_xlim(xs.min(), xs.max())
    else:
        ax.text(0.5, 0.5, "no matched pairs", ha="center", va="center",
                transform=ax.transAxes)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("empirical CDF")
    ax.set_xlabel(xlabel)
    ax.set_title(f"{title}  (n={len(values)})")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def _failure_mode_bar(m: IntervalPredictionMetrics, out_path: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = ["clean", "missed", "pred_merged", "gt_split", "fp"]
    values = [m.clean, m.missed, m.pred_merged, m.gt_split, m.fp]
    colors = ["tab:green", "tab:red", "tab:orange", "tab:purple", "tab:gray"]

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.bar(labels, values, color=colors)
    for i, v in enumerate(values):
        ax.text(i, v, str(v), ha="center", va="bottom", fontsize=9)
    ax.set_ylabel("count")
    ax.set_title(
        f"Failure modes  (gt={m.n_gt}, pred={m.n_pred}, "
        f"f1_like={m.score():.3f})"
    )
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=120)
    plt.close(fig)


def render_all(
    matched_pairs: list[dict],
    total: IntervalPredictionMetrics,
    out_dir: Path,
) -> dict[str, Path]:
    """Write the standard diagnostic set and return ``{name: path}``."""
    paths: dict[str, Path] = {}

    paths["iou"] = out_dir / "cdf_iou.png"
    _cdf_plot(
        [r["iou"] for r in matched_pairs],
        title="CDF of IoU over matched pairs",
        xlabel="IoU",
        out_path=paths["iou"],
    )

    paths["start_residual"] = out_dir / "cdf_start_residual.png"
    _cdf_plot(
        [r["start_residual_s"] for r in matched_pairs],
        title="CDF of start-edge residual  (pred - gt)",
        xlabel="start residual (s)",
        out_path=paths["start_residual"],
    )

    paths["end_residual"] = out_dir / "cdf_end_residual.png"
    _cdf_plot(
        [r["end_residual_s"] for r in matched_pairs],
        title="CDF of end-edge residual  (pred - gt)",
        xlabel="end residual (s)",
        out_path=paths["end_residual"],
    )

    paths["duration_error"] = out_dir / "cdf_duration_error.png"
    _cdf_plot(
        [r["duration_error_s"] for r in matched_pairs],
        title="CDF of duration error  (pred - gt)",
        xlabel="duration error (s)",
        out_path=paths["duration_error"],
    )

    paths["failure_modes"] = out_dir / "failure_modes.png"
    _failure_mode_bar(total, paths["failure_modes"])

    return paths
