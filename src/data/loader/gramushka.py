"""Gramushka resolver: map experiments to per-building floor-elevation tables.

Each building whose experiments appear in this project has a central reference
file at ``src/data/gramushka/<building_folder>/gramushka.csv`` with columns
``Floor Name, Elevation (m)``. The per-experiment
``structuredData/data/<exp>/baramoshka.csv`` is the project-local view of that
table: same rows, columns renamed to the architecture's ``floor, height``.

Two public helpers:

- :func:`gramushka_building_for_exp` — experiment name → gramushka folder name
  (or ``None`` when the experiment has no associated building — e.g. archive
  experiments, or exp3 which by project convention has no gramushka
  observations and should fall back to pure-barometer Δh).

- :func:`load_baramoshka_for_exp` — experiment name → DataFrame with the
  ``floor, height`` schema loaded from the central gramushka file. Returns
  ``None`` when no building is resolvable.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from .constants import BAROMOSHKA_COLUMNS

_GRAMUSHKA_ROOT = Path(__file__).resolve().parents[1] / "gramushka"

# Slug extracted from experiment folder names (the segment between
# experimenter and phone) → folder name under src/data/gramushka/.
# Keys are lowercased for case-insensitive lookup.
_BUILDING_SLUG_TO_FOLDER: dict[str, str] = {
    "milleniumhotel":        "פרימה מילניום",
    "milleniumoutside":      "פרימה מילניום",
    "acrobuilding":          "אקרו נדלן",
    "beitmansour1":          "בית_מנצור_1",
    "beityitzchakiraanana":  "בית יצחקי ב",
    "barilan2herzelia":      "בר אילן 2",
    "haari3":                "haari",
}


# Experiments that, by project convention, should NOT use gramushka snapping
# even if their building has a central gramushka file. For these we leave
# baramoshka.csv empty and the corrector falls back to pure barometer.
# (User statement 2026-04-19: "exp3 we don't actually have gramushka for so the
# height differences should be just purely [barometer]".)
_EXPERIMENTS_WITHOUT_GRAMUSHKA: set[str] = set()


def _building_slug_from_exp_name(exp_name: str) -> str | None:
    """Return the lowercased building slug embedded in an experiment folder
    name, or ``None`` if no known slug is present.

    Experiment names follow ``<experimenter>_<buildingSlug>_<phone>_<date>...``
    so we just scan the underscore-separated tokens against the known slugs.
    """
    tokens = [t.lower() for t in exp_name.split("_")]
    for tok in tokens:
        if tok in _BUILDING_SLUG_TO_FOLDER:
            return tok
    return None


def _is_experiment_without_gramushka(exp_name: str) -> bool:
    """True when the experiment should skip gramushka snapping entirely
    (leaves baramoshka.csv empty, corrector falls back to pure barometer).

    Rule: any experiment whose name ends in ``_exp3`` (the April 2026
    ``milleniumOutside`` runs — the external elevator attached to the
    Millenium hotel, where we didn't record floor visits).
    Extend :data:`_EXPERIMENTS_WITHOUT_GRAMUSHKA` for explicit overrides.
    """
    if exp_name in _EXPERIMENTS_WITHOUT_GRAMUSHKA:
        return True
    return exp_name.lower().endswith("_exp3")


def gramushka_building_for_exp(exp_name: str) -> str | None:
    """Return the central gramushka folder name for an experiment, or ``None``.

    Returns ``None`` when:
      * the experiment name contains no known building slug (archive, Haari, etc.), or
      * the experiment is on the :data:`_EXPERIMENTS_WITHOUT_GRAMUSHKA` list,
        or matches the exp3-style override (see :func:`_is_experiment_without_gramushka`).
    """
    if _is_experiment_without_gramushka(exp_name):
        return None
    slug = _building_slug_from_exp_name(exp_name)
    if slug is None:
        return None
    return _BUILDING_SLUG_TO_FOLDER[slug]


def _load_central_gramushka(folder_name: str) -> pd.DataFrame | None:
    """Read ``src/data/gramushka/<folder_name>/gramushka.csv`` and return it
    with columns renamed to the architecture's ``floor, height`` schema.

    Returns ``None`` if the file is missing or unparseable.
    """
    path = _GRAMUSHKA_ROOT / folder_name / "gramushka.csv"
    if not path.exists():
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    if df.empty:
        return None
    # Accept either the central ("Floor Name", "Elevation (m)") or local
    # ("floor", "height") schema.
    rename = {}
    if "Floor Name" in df.columns:
        rename["Floor Name"] = "floor"
    if "Elevation (m)" in df.columns:
        rename["Elevation (m)"] = "height"
    if rename:
        df = df.rename(columns=rename)
    if "floor" not in df.columns or "height" not in df.columns:
        return None
    # Elevations may be prefixed with '+' or '±' — coerce to float.
    df["height"] = (
        df["height"].astype(str).str.replace("±", "", regex=False)
                                .str.replace("+", "", regex=False)
                                .str.strip()
    )
    df["height"] = pd.to_numeric(df["height"], errors="coerce")
    df = df.dropna(subset=["height"]).reset_index(drop=True)
    return df[BAROMOSHKA_COLUMNS].copy()


def load_baramoshka_for_exp(exp_name: str) -> pd.DataFrame | None:
    """Load the floor→height table an experiment should use for GT snapping.

    Returns a DataFrame with columns ``[floor, height]`` (the
    :data:`BAROMOSHKA_COLUMNS` schema), or ``None`` when the experiment has
    no gramushka reference (archive, exp3, unmapped building).
    """
    folder = gramushka_building_for_exp(exp_name)
    if folder is None:
        return None
    return _load_central_gramushka(folder)


def resolve_start_floor_default(exp_name: str) -> str:
    """Project-default start floor for an experiment when none is explicitly set.

    Rules (from Uriya + verified via cumulative barometer on 2026-04-19):

    * ``milleniumHotel ... _exp1`` → ``Floor 12`` (stated by Uriya)
    * ``milleniumHotel ... _exp2`` → ``Floor 12`` (not stated by Uriya, but
      the cumulative barometer across every phone's recording drops 50-80 m
      from the recording's first sample; only Floor 12 matches. The full
      table of per-phone end cumulatives lives in
      `memory/experiment_calibration_facts.md`.)
    * ``milleniumOutside`` (the external hotel elevator, exp3) →
      ``Ground Floor`` via the default branch below.
    * everything else → ``Ground Floor``.

    Returns the floor-name string — the corrector looks it up in the
    experiment's baramoshka.csv to get the numeric start altitude.
    """
    lower = exp_name.lower()
    if "milleniumhotel" in lower and (lower.endswith("_exp1") or lower.endswith("_exp2")):
        return "Floor 12"
    return "Ground Floor"
