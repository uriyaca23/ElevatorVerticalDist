"""Dataset-cleanup tooling for the ``uriya-working-on-data`` pass.

Everything under this package is additive to Eyal's loader architecture —
no public API of :mod:`src.data.loader` was removed. The scripts here
just orchestrate that loader plus a few new helpers (gramushka snap,
phone-time xcorr, Pixel-PRS reference sharing) to clean up the dataset.

Run order that reproduces the current state from a fresh checkout::

    python -m src.data.dataset_cleanup.populate_baramoshka
    python -m src.data.dataset_cleanup.gramushka_apply
    python -m src.data.dataset_cleanup.phone_time_calibration --apply
    python -m src.data.dataset_cleanup.residual_calibration --apply
    python -m src.data.dataset_cleanup.mae_residual_sweep --apply
    python -m src.data.dataset_cleanup.apply_pixel_reference --apply
    python -m src.data.dataset_cleanup.tag_noisy_segments
    python -m src.data.dataset_cleanup.verify_calibration
    python -m src.data.dataset_cleanup.save_test_results
    python -m src.data.dataset_cleanup.zupt_sanity_check

See :mod:`src.data.dataset_cleanup.README` (a sibling Markdown file, not
an importable module) for the full motivation.
"""
