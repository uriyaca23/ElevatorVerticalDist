"""Experiment selection — the standard ``--kind`` / ``--source`` /
``--include`` / ``--exclude`` filter shared by every ``evaluateOnData`` CLI.

One place decides *which experiments feed a run*. Candidates come from
``structuredData/`` (:func:`list_structured_experiments`); an experiment's
train/test split and source are read from the ``structuredData/metadata.csv``
index (:func:`load_experiment_index`) — the metadata it was ingested with,
not a guess from its folder name.
"""

from __future__ import annotations

import argparse

from .constants import EXPERIMENT_TYPES, VALID_SOURCES
from .pipeline import list_structured_experiments, load_experiment_index


def add_selection_args(parser: argparse.ArgumentParser) -> None:
    """Add the standard experiment-selection flags to ``parser``.

    Registers ``--kind`` / ``--source`` / ``--include`` / ``--exclude`` with
    the schema :func:`resolve_experiments` expects. Call it from every CLI's
    argument parser so the four filters stay identical everywhere.
    """
    parser.add_argument(
        "--kind", default="all", choices=("all", *EXPERIMENT_TYPES),
        help="Restrict to train, test, or all experiments (default: all).",
    )
    parser.add_argument(
        "--source", action="append", default=None,
        choices=[*VALID_SOURCES, "all"],
        help="Filter by metadata.source — repeatable. Pass 'all' (or omit "
             "the flag) to keep every source.",
    )
    parser.add_argument(
        "--include", nargs="*", default=None,
        help="Whitelist of experiment names (still subject to the other "
             "filters).",
    )
    parser.add_argument(
        "--exclude", nargs="*", default=None,
        help="Drop these experiment names from the run.",
    )


def resolve_experiments(
    kind: str = "all",
    sources: list[str] | None = None,
    include: list[str] | None = None,
    exclude: list[str] | None = None,
) -> list[str]:
    """Return the sorted experiment names surviving every filter.

    * Candidates are ``include`` verbatim when given, else every experiment
      under ``structuredData/`` (:func:`list_structured_experiments`).
    * ``kind`` (``train`` / ``test`` / ``all``) and ``sources`` are matched
      against the ``structuredData/metadata.csv`` index
      (:func:`load_experiment_index`).
    * ``sources`` may contain the alias ``'all'`` — treated as no filter.
    """
    if sources and "all" in sources:
        sources = None
    candidates = list(include) if include else list_structured_experiments()
    index = load_experiment_index()
    excluded = set(exclude or [])
    out: list[str] = []
    for name in candidates:
        if name in excluded:
            continue
        row = index.get(name, {})
        if kind != "all" and row.get("experiment_type") != kind:
            continue
        if sources and row.get("source") not in sources:
            continue
        out.append(name)
    return sorted(out)
