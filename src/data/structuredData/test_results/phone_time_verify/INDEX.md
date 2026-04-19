# Phone-time-calibration verification plots

Per-experiment pulse-alignment plots for manual review. Each non-Pixel experiment has two images:

* `<exp>__verify.png` — final ACC overlay (Pixel blue, phone green) inside six Pixel-tagged elevator segments. If green pulses sit inside the red window and on top of the blue curve, alignment is good.
* `<exp>__calibration.png` — the calibration-step diagnostic (before-shift orange vs after-shift green).

Scan in order; any plot where green pulses clearly sit outside the red window is a misaligned phone to flag.

## Offsets applied

| Experiment | offset (ms) | method | median residual (ms) |
|---|---|---|---|
| `eyalyakir_acroBuilding_SamsungSM-S911B_15-04-2026_exp4` | -1080 | per_segment_prs_xcorr | -260 |
| `eyalyakir_acroBuilding_Xiaomi22101320I_15-04-2026_exp4` | 980 | per_segment_prs_xcorr | -260 |
| `eyalyakir_beitMansour1_SamsungSM-S911B_15-04-2026_exp5` | -980 | per_segment_prs_xcorr | -390 |
| `eyalyakir_beitMansour1_Xiaomi22101320I_15-04-2026_exp5` | -2020 | per_segment_prs_xcorr | -360 |
| `eyalyakir_beitYitzchakiRaanana_SamsungSM-S911B_15-04-2026_exp6` | 0 | per_segment_acc_xcorr | -20 |
| `eyalyakir_beitYitzchakiRaanana_Xiaomi22101320I_15-04-2026_exp6` | 0 | per_segment_acc(majority)_xcorr | 20 |
| `eyalyakir_milleniumHotel_SamsungSM-A235F_15-04-2026_exp1` | 14300 | skipped_low_confidence (acc(degraded)) | 20 |
| `eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp1` | 16680 | skipped_low_confidence (acc(degraded)) | 0 |
| `eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp2` | 0 | per_segment_acc_xcorr | 0 |
| `eyalyakir_milleniumHotel_SamsungSM-S911B_15-04-2026_exp3` | 0 | per_segment_acc_xcorr | 0 |
| `eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp2` | 0 | per_segment_acc_xcorr | 0 |
| `eyalyakir_milleniumHotel_Xiaomi22101320I_15-04-2026_exp3` | 0 | per_segment_acc_xcorr | -20 |
| `UriyaCohenEliya_acroBuilding_SamsungSM-A235F_15-04-2026_exp4` | -720 | per_segment_prs_xcorr | 30 |
| `UriyaCohenEliya_beitMansour1_SamsungSM-A235F_15-04-2026_exp5` | -1420 | per_segment_prs_xcorr | -410 |
| `UriyaCohenEliya_beitYitzchakiRaanana_SamsungSM-A235F_15-04-2026_exp6` | 0 | per_segment_acc_xcorr | 0 |
| `UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp2` | 0 | per_segment_acc_xcorr | 20 |
| `UriyaCohenEliya_milleniumHotel_SamsungSM-A235F_15-04-2026_exp3` | 0 | per_segment_acc_xcorr | 20 |
| `UriyaCohenEliya_milleniumHotel_Xiaomi22101320I_15-04-2026_exp1` | 10820 | skipped_low_confidence (acc(degraded)) | 0 |
