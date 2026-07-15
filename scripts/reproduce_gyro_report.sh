#!/usr/bin/env bash
#
# One-shot reproduction of the gyroscope-reconstruction results and the paper.
#
#   bash scripts/reproduce_gyro_report.sh
#
# End to end, from a clean checkout, this:
#   1. sets up the venv (if missing) and installs requirements;
#   2. (re)fits the baseline conformal calibration on the TRAIN split and
#      commits it as the canonical calibration.json for each predictor;
#   3. runs both stages across {none + 5 reconstruct_az filters} and writes the
#      comparison figures/tables into docs/latex/figures/reconstruct_az/;
#   4. refits the calibration with Mahony reconstruction, scores the held-out
#      TEST split, then restores the baseline calibration (so the committed
#      default pipeline is left on reconstruct=none);
#   5. regenerates docs/latex/figures/gyro/*.tex from the run metrics;
#   6. rebuilds docs/latex/main.pdf and docs/latex/algorithm_report.pdf.
#
# Runtime is dominated by step 3 (six full passes through both stages) and can
# take tens of minutes on a laptop. Nothing here needs network access.

set -euo pipefail

REPO="$(cd "$(dirname "$0")/.." && pwd)"
cd "$REPO"

# --- 0. environment -------------------------------------------------------
if [ ! -d venv ]; then
  echo ">> creating venv + installing requirements"
  python3 -m venv venv
  ./venv/bin/pip install -r requirements.txt
fi
PY=./venv/bin/python

Z=src/prediction/algorithms/accelerometer_only/zupt_accel/calibration.json
T=src/prediction/algorithms/accelerometer_only/trapezoid_accel/calibration.json
WORK="$(mktemp -d)"
trap 'echo ">> scratch was $WORK"' EXIT

# --- 1. baseline (none) calibration --------------------------------------
echo ">> [1/6] baseline calibration: refit on TRAIN, score TEST"
$PY -m src.evaluation.prediction.evaluateOnData --kind train --calibrate \
    --out-root "$WORK" --run-name base_train
$PY -m src.evaluation.prediction.evaluateOnData --kind test \
    --out-root "$WORK" --run-name base_test
cp "$Z" "$WORK/z.none"; cp "$T" "$WORK/t.none"   # canonical none-calibration

# --- 2. reconstruction comparison (all six variants, both stages) --------
echo ">> [2/6] reconstruction comparison report (this is the slow step)"
$PY scripts/reconstruction_comparison_report.py --kind all

# --- 3. Mahony calibrated, held-out test ---------------------------------
echo ">> [3/6] Mahony: refit calibration on TRAIN, score held-out TEST"
$PY -m src.evaluation.prediction.evaluateOnData --kind train --reconstruct Mahony \
    --calibrate --out-root "$WORK" --run-name mah_train
$PY -m src.evaluation.prediction.evaluateOnData --kind test --reconstruct Mahony \
    --out-root "$WORK" --run-name mah_test
cp "$WORK/z.none" "$Z"; cp "$WORK/t.none" "$T"   # restore baseline calibration

# --- 4/5. regenerate the section's table fragments -----------------------
echo ">> [4/6] regenerate docs/latex/figures/gyro/*.tex"
$PY scripts/reproduce_gyro_report_tables.py \
    --comparison docs/latex/figures/reconstruct_az/comparison_metrics.json \
    --none-test  "$WORK/base_test/metrics.json" \
    --mahony-test "$WORK/mah_test/metrics.json" \
    --out docs/latex/figures/gyro

# --- 6. build the PDFs ----------------------------------------------------
echo ">> [5/6] build main.pdf"
( cd docs/latex && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null \
                && pdflatex -interaction=nonstopmode -halt-on-error main.tex >/dev/null )
echo ">> [6/6] build algorithm_report.pdf"
( cd docs/latex && pdflatex -interaction=nonstopmode -halt-on-error algorithm_report.tex >/dev/null \
                && pdflatex -interaction=nonstopmode -halt-on-error algorithm_report.tex >/dev/null )

echo ">> DONE"
echo "   docs/latex/main.pdf"
echo "   docs/latex/algorithm_report.pdf"
