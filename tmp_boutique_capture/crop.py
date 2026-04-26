"""Post-process the captured Streamlit full-page screenshots into the
panel-zoom crops the LaTeX guide actually places side-by-side.

Each entry in `JOBS` says: take this source PNG, take that pixel
rectangle (relative to the source), save it under that name. Coordinates
were measured by sampling the canonical `clean_step3_seg5_full.png`
which is 3200x4910 (1600×scrollHeight × DPR=2). Most other Step 3
captures share the same layout because the seed runner pre-loads the
same wizard.
"""
from __future__ import annotations
from pathlib import Path
from PIL import Image

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "docs" / "latex" / "figures" / "boutique"


def crop_from(src: str, x: int, y: int, w: int, h: int, out: str) -> None:
    src_path = SRC / src
    if not src_path.exists():
        print(f"  MISSING SOURCE: {src}")
        return
    im = Image.open(src_path)
    box = (x, y, x + w, y + h)
    im2 = im.crop(box)
    out_path = SRC / out
    im2.save(out_path, optimize=True)
    print(f"  {src} {im.size}  ->  {out}  {im2.size}")


# Source PNGs are 3200x{viewport_h*2}. Layout from top to bottom:
#   ~0    – 220   : top app strip / sidebar header (scaled-up)
#   ~220  – 440   : hero banner ("Interactive segmentation")
#   ~440  – 1100  : "Signal + segments" title + main timeline plot
#   ~1100 – 1280  : "Detail — segment #i ..." title strip
#   ~1280 – 1880  : Two heatmaps side-by-side
#   ~1880 – 2080  : "Correlation score with peak status" title + legend
#   ~2080 – 2540  : Correlation panel
#   ~2540 – 2680  : "Signal with fitted trapezoid" title
#   ~2680 – 3140  : Trapezoid overlay panel
#   ~3140 – 3260  : "Fitted trapezoid parameters" title
#   ~3260 – 3700  : Three trapezoid parameter cards
#   ~3700 – ...   : Spreadsheet-editor expander + bottom buttons
#
# Numbers are approximate and were checked against
# clean_step3_seg5_full.png. If the source files come out slightly
# different (e.g. resolution change), tweak the band offsets globally.

# ---- top: signal+segments timeline strip  -------------------------------
TIMELINE = (160, 340, 2950, 760)   # x, y, w, h on 3200×... image

# ---- two heatmaps row ---------------------------------------------------
HEATMAPS = (160, 1180, 2950, 720)

# ---- correlation panel  -------------------------------------------------
CORRPANEL = (160, 1900, 2950, 700)

# ---- trapezoid overlay -------------------------------------------------
TRAP = (160, 2620, 2950, 540)

# ---- parameter cards ---------------------------------------------------
PARAMS = (160, 3220, 2950, 480)


JOBS = [
    # Easy-Path showcase — clean ride
    ("clean_step3_seg5_full.png", *TIMELINE,  "panel_clean_timeline.png"),
    ("clean_step3_seg5_full.png", *HEATMAPS,  "panel_clean_heatmaps.png"),
    ("clean_step3_seg5_full.png", *CORRPANEL, "panel_clean_corr.png"),
    ("clean_step3_seg5_full.png", *TRAP,      "panel_clean_trapezoid.png"),
    ("clean_step3_seg5_full.png", *PARAMS,    "panel_clean_params.png"),

    # False-positive contrast (segment 0 of milleniumA23)
    ("fp_step3_seg0_FP.png",      *TIMELINE,  "panel_fp_timeline.png"),
    ("fp_step3_seg0_FP.png",      *HEATMAPS,  "panel_fp_heatmaps.png"),
    ("fp_step3_seg0_FP.png",      *CORRPANEL, "panel_fp_corr.png"),
    ("fp_step3_seg0_FP.png",      *TRAP,      "panel_fp_trapezoid.png"),

    # Damped-trace (BarIlan Pixel10) — many missed rides
    ("damped_step3_overview.png", *TIMELINE,  "panel_damped_timeline.png"),
    ("damped_step3_seg0_full.png", *HEATMAPS, "panel_damped_heatmaps.png"),
    ("damped_step3_seg0_full.png", *CORRPANEL,"panel_damped_corr.png"),
    ("damped_step3_seg0_full.png", *TRAP,     "panel_damped_trapezoid.png"),

    # Haari — borderline FP/merge, useful as a "real-world messy" example
    ("haari_step3_seg10_full.png", *HEATMAPS, "panel_haari_heatmaps.png"),
    ("haari_step3_seg10_full.png", *CORRPANEL,"panel_haari_corr.png"),
    ("haari_step3_seg10_full.png", *TRAP,     "panel_haari_trapezoid.png"),
]


def main() -> None:
    for src, x, y, w, h, out in JOBS:
        crop_from(src, x, y, w, h, out)


if __name__ == "__main__":
    main()
