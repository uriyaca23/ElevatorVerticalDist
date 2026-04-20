# Active detector lives in ``check_grid_across_signal.detect`` and is
# wired through ``Segmenter``. The legacy sliding-NCC ``matcher.py`` +
# ``templates.py`` / ``build_pulse_labels.py`` files are kept on disk for
# reference but no longer re-exported — they depend on an older
# ``TemplateMatchConfig`` shape that has been repurposed for the grid
# detector.

__all__: list[str] = []
