"""Stage evaluation outputs + generate LaTeX fragments for main.tex.

Reads from ``src/data/structuredData/test_results/prediction/`` and
writes into ``docs/latex/figures/prediction/`` the ``results_*.tex``
fragments and a flat copy of every figure under predictable names
(``<split>_<algo>_<key>.png``).

Run whenever you regenerate the evaluation outputs::

    python scripts/build_prediction_report_assets.py
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[1]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.prediction.evaluation.report import build_report_assets


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument(
        "--eval-root", type=Path,
        default=_REPO_ROOT / "src" / "data" / "structuredData" / "test_results" / "prediction",
    )
    ap.add_argument(
        "--latex-figures", type=Path,
        default=_REPO_ROOT / "docs" / "latex" / "figures" / "prediction",
    )
    ap.add_argument("--skip-test", action="store_true",
                    help="Skip test outputs (train-only report).")
    args = ap.parse_args()

    train_root = args.eval_root / "train"
    test_root = None if args.skip_test else args.eval_root / "test"
    if test_root is not None and not (test_root / "metrics_test.json").exists():
        print(f"[report] no test metrics at {test_root}; train-only")
        test_root = None

    build_report_assets(train_root=train_root,
                        test_root=test_root,
                        latex_assets_dir=args.latex_figures)
    print(f"[report] wrote assets to {args.latex_figures}")


if __name__ == "__main__":
    main()
