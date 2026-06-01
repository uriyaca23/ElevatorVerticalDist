"""Generate the four-stage pipeline flow diagram for the Nehori 2 operating guide.

Boxes are laid out right-to-left (Hebrew reading order):
    שליפת נתונים  ->  סגמנטציה  ->  חיזוי  ->  דו"ח
The two middle stages (segmentation, prediction) carry a callout marking them as
steps where the operator may have to intervene (the pipeline is not fully autonomous).

Run with: venv/bin/python docs/latex/make_pipeline_diagram.py
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

OUT = "docs/latex/figures/nehori2/pipeline_flow.png"


def rtl(s: str) -> str:
    """Reverse a Hebrew string so matplotlib (no bidi) renders it right-to-left."""
    return s[::-1]


# Stage labels in reading order, rightmost first.
STAGES = [
    ("data", rtl("שליפת נתונים"), "#2c6fbb"),
    ("seg", rtl("סגמנטציה"), "#e85d4e"),
    ("pred", rtl("חיזוי"), "#e85d4e"),
    ("report", rtl("דו״ח"), "#2c6fbb"),
]

INTERACTIVE = {"seg", "pred"}

fig, ax = plt.subplots(figsize=(11, 4.2))
ax.set_xlim(0, 100)
ax.set_ylim(0, 42)
ax.axis("off")

box_w, box_h = 18.0, 11.0
gap = 6.0
y = 24.0
# Right-to-left: first stage at the right edge.
n = len(STAGES)
total_w = n * box_w + (n - 1) * gap
x_right = 96.0
centers = []
for i, (key, label, color) in enumerate(STAGES):
    x = x_right - box_w - i * (box_w + gap)
    cx = x + box_w / 2
    centers.append((key, cx))
    interactive = key in INTERACTIVE
    box = FancyBboxPatch(
        (x, y), box_w, box_h,
        boxstyle="round,pad=0.5,rounding_size=2.2",
        linewidth=2.4,
        edgecolor=color,
        facecolor=color,
        alpha=0.16 if not interactive else 0.22,
        zorder=2,
    )
    ax.add_patch(box)
    # colored top border accent
    ax.add_patch(FancyBboxPatch(
        (x, y), box_w, box_h,
        boxstyle="round,pad=0.5,rounding_size=2.2",
        linewidth=2.6, edgecolor=color, facecolor="none", zorder=3))
    ax.text(cx, y + box_h / 2, label, ha="center", va="center",
            fontsize=21, fontweight="bold", color="#15324f", zorder=4)
    ax.text(cx, y + box_h + 2.0, str(i + 1) if False else "", ha="center")

# Arrows between consecutive stages (point in processing direction: right -> left).
for i in range(n - 1):
    _, cx_from = centers[i]      # right box
    _, cx_to = centers[i + 1]    # left box
    start = cx_from - box_w / 2
    end = cx_to + box_w / 2
    arr = FancyArrowPatch(
        (start, y + box_h / 2), (end, y + box_h / 2),
        arrowstyle="-|>", mutation_scale=26,
        linewidth=2.6, color="#5b6b7a", zorder=1,
    )
    ax.add_patch(arr)

# Human-in-the-loop callout under the two interactive stages.
callout_y = 10.0
note = rtl("ייתכן שתידרש התערבות ידנית של המשתמש")
for key, cx in centers:
    if key in INTERACTIVE:
        arr = FancyArrowPatch(
            (cx, y - 0.5), (cx, callout_y + 4.0),
            arrowstyle="-|>", mutation_scale=20,
            linewidth=2.2, color="#e85d4e", linestyle=(0, (4, 2)), zorder=1,
        )
        ax.add_patch(arr)
        ax.plot(cx, callout_y + 3.6, marker="o", color="#e85d4e", markersize=6, zorder=2)

# Single shared callout banner spanning the two interactive boxes.
seg_cx = dict(centers)["seg"]
pred_cx = dict(centers)["pred"]
band_left = min(seg_cx, pred_cx) - box_w / 2 - 1
band_right = max(seg_cx, pred_cx) + box_w / 2 + 1
band = FancyBboxPatch(
    (band_left, callout_y - 4.5), band_right - band_left, 7.5,
    boxstyle="round,pad=0.3,rounding_size=2.0",
    linewidth=2.0, edgecolor="#e85d4e", facecolor="#fdeeec", zorder=2,
)
ax.add_patch(band)
ax.text((band_left + band_right) / 2, callout_y - 0.8, note,
        ha="center", va="center", fontsize=15.5, color="#b03a2e",
        fontweight="bold", zorder=3)

# Title
ax.text(50, 40, rtl("שלבי הפעלת נהורי 2"), ha="center", va="center",
        fontsize=20, fontweight="bold", color="#15324f")

plt.tight_layout()
fig.savefig(OUT, dpi=200, bbox_inches="tight", facecolor="white")
print("wrote", OUT)
