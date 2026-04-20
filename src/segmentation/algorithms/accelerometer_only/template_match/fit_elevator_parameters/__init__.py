"""Trapezoid-pulse fitters for GT elevator rides.

Algorithms (one per module):

* ``basic_grid`` — independent per-lobe matched-filter grid search. Each
  lobe gets its own ``(t_c, A, W, f)``. Output dir:
  ``labels/fit_elevator_paramater/basicTreepzeGrid/``.

* ``constrained_grid`` — shared-shape fit. Both lobes of a ride are
  constrained to the same ``(|A|, W, f)`` and only differ in sign and
  ``t_c``. The chosen ``(W, f, t_c1, t_c2)`` maximises the *mean* of the
  two per-lobe local R². Output dir:
  ``labels/fit_elevator_paramater/basicTreepzeGridWithConstraint/``.

``common`` holds everything both algorithms share: trapezoid kernel,
signal preprocessing, ride slicing, plotting, dataclasses, and the
experiment-level driver ``run_fitter(out_dir_name, fit_ride, ...)``.
"""
