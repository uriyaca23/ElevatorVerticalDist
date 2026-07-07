"""``python -m src.pipelines.prediction_editor [exp]`` entry point."""
from __future__ import annotations

import sys

from src.pipelines.prediction_editor.app import main

if __name__ == "__main__":
    sys.exit(main())
