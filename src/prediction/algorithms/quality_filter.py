"""Backwards-compatible shim.

The real implementation lives at ``uriya_shit.quality_filter`` (the
folder was previously named ``oria_shit`` and got renamed on this
branch). Two call sites still import from this path
(``src/segmentation/.../acc_segmentation.py`` and ``fit_trapezoid_pulses.py``),
so we re-export the names they need here. Remove this shim once both
call sites have been updated to import from the new location directly.
"""

from .uriya_shit.quality_filter import *  # noqa: F401,F403
from .uriya_shit.quality_filter import (  # noqa: F401
    estimate_gravity_vector,
)
