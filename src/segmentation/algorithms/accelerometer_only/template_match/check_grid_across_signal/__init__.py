"""Whole-signal trapezoid detector + (+lobe, −lobe) pair filter.

Sibling to ``fit_elevator_parameters/``:

* ``fit_elevator_parameters`` fits trapezoids **inside** each GT ride
  window.
* ``check_grid_across_signal`` takes the same 30×15 ``(W, f)`` trapezoid
  grid and sweeps it across the **whole session**, peak-picks the
  matched-filter R², then keeps only sign-opposite candidate pairs whose
  shared-shape joint R² clears a threshold. Result is a ride-like
  prediction list written to
  ``labels/check_grid_across_signal/<exp>/predictions.json`` together
  with a whole-signal overview PNG.
"""
