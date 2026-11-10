# `dataset_cleanup/` — scripts for the `uriya-working-on-data` pass

Everything under this package is **additive** to Eyal's loader
architecture — no public API of `src/data/loader/` was removed. These
scripts use that loader plus three new helpers to clean up the dataset:

1. **Gramushka snap** — every up/down segment's `height_diff_m` in
   `gt.csv` is the difference of successive snapped gramushka-floor
   elevations, anchored to a known `start_floor` in metadata. Uses
   temperature-aware barometric inversion.
2. **Pixel-10-anchored time calibration** — every non-Pixel phone's
   sensor + gt timestamps are shifted so that tagged elevator rides
   align with the Pixel's wall-clock inside ±tens-of-ms.
3. **Pixel-PRS reference pass** — every non-Pixel phone in an experiment
   that has a Pixel reference gets its `gt.csv` + `height_diff_m`
   replaced by values derived from the Pixel's gt + Pixel's PRS. All
   phones in the same experiment end up with identical segment bounds
   and identical Δh (as physics demands).

## Run order (reproduces the current state from a fresh checkout)

```bash
python -m src.data.dataset_cleanup.populate_baramoshka       # fill baramoshka.csv + start_floor + temperature_c
python -m src.data.dataset_cleanup.gramushka_apply           # rewrite gt.csv Δh with gramushka snap
python -m src.data.dataset_cleanup.phone_time_calibration --apply  # coarse xcorr-based per-phone time shifts
python -m src.data.dataset_cleanup.residual_calibration --apply    # per-segment-xcorr high-pass residual pass
python -m src.data.dataset_cleanup.mae_residual_sweep --apply      # brute-force MAE sweep (catches xcorr spurious peaks)
python -m src.data.dataset_cleanup.apply_pixel_reference --apply   # share Pixel's gt + Pixel's PRS across phones
python -m src.data.dataset_cleanup.tag_noisy_segments        # flip signalClearRecording on the 2 dirty 2nd-halves
python -m src.data.dataset_cleanup.verify_calibration        # per-segment ACC overlay plots (post-apply)
python -m src.data.dataset_cleanup.save_test_results         # collect plots into structuredData/test_results/
python -m src.data.dataset_cleanup.zupt_sanity_check         # naive |Δh| cross-phone agreement check
```

Running these in order on an already-clean dataset is a no-op —
`--apply` passes only modify what's off by more than their threshold,
and all the destructive steps store a backup (e.g.
`gt_pre_pixel_ref_backup.csv`) next to the file they replace.

## Outputs

Everything lives under `structuredData/test_results/`:

| Artifact | What it is |
|---|---|
| `phone_time_verify/` | Per-experiment ACC-overlay plots (blue Pixel, green phone) for manual review |
| `phone_time_verify/INDEX.md` | Table of offsets + methods + residuals |
| `noisy_segments/` | Altitude + |a| + SNR plots for the 2 "dirty second-half" experiments |
| `gramushka_dry_run_summary.csv` | Per-experiment Δh-correction RMSE old-vs-new |
| `gramushka_flags_summary.csv` | Per-experiment flagged-segment counts |
| `phone_time_calibration.csv` | Offsets + methods + scores |
| `phone_time_verify_summary.csv` | Per-experiment residual + MAE |
| `pixel_reference_summary.csv` | Which phones had their gt+Δh replaced |
| `residual_calibration.csv` | Output of the xcorr residual pass |
| `mae_residual_sweep.csv` | Output of the MAE-sweep pass |
| `zupt_segments.csv` | Per-ride naive-ZUPT |Δh| on every phone |
| `zupt_experiment_summary.csv` | Cross-phone agreement medians |

Per-experiment auxiliary files written under each
`structuredData/data/<exp>/`:

| File | Purpose |
|---|---|
| `gramushka_flags.csv` | Rows of segments whose gramushka snap distance > 1.5 m |
| `phone_time_calibration.png` / `phone_time_verify.png` | Diagnostic plots (also copied to `test_results/phone_time_verify/`) |
| `phone_time_verify.csv` | Per-segment ACC residual table |
| `gt_pre_pixel_ref_backup.csv` | gt.csv as it was before `apply_pixel_reference` overwrote it |

The loader's `_load_structured_triplet` only picks up CSVs whose stem
matches a known sensor name (`ACC`, `GYR`, `MAG`, `ORI`, `PRS`, `RAWGYR`,
`RAWMAG`, `GPS`), so adding these aux CSVs doesn't confuse downstream
code that iterates by `timestamp_ms`.
