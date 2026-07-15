# CLAUDE.md — data location (moved off the repo)

The heavy sensor data that used to live under `src/data/` has been **moved to
iCloud Drive** to free up local disk and keep it out of git. The code here
(loader, ingest, GT editor) is unchanged; only the data payloads are gone.

## Where the data now lives

**iCloud Drive → `data research/ElevatorVerticalDist/`**

Full local path:

```
/Users/eyal/Library/Mobile Documents/com~apple~CloudDocs/data research/ElevatorVerticalDist/
```

Moved on 2026-07-15. Baseline: 1562 files, 3.64 GiB, checksum-verified before
deletion. A `BACKUP_INFO.txt` sits alongside the data with the same details.

## What was moved (mirror of `src/data/`)

| Folder | Size | What it is |
|---|---|---|
| `rawData/` | 1.1 G | raw per-experiment sensor logs |
| `structuredData/` | 1.7 G | processed per-experiment CSVs + `metadata.csv`, `gt_edits.csv`, `test_results/` |
| `(archive)/` | 703 M | older raw experiment sessions (eyal / uriya / roy_turgeman) |
| `gramushka/` | 91 M | barometer calibration reference (by building) |
| `USC-HAD/` | 44 M | external public HAD dataset (re-downloadable) |

## To restore

Copy the folders back into `src/data/` from the iCloud location above, e.g.:

```bash
SRC="/Users/eyal/Library/Mobile Documents/com~apple~CloudDocs/data research/ElevatorVerticalDist"
rsync -a "$SRC/rawData" "$SRC/structuredData" "$SRC/(archive)" "$SRC/gramushka" "$SRC/USC-HAD" src/data/
```

These folders are git-ignored, so restoring them locally will not re-add them
to the repo.
