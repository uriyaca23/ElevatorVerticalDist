"""Shared Tk / matplotlib factories, colors and tiny helpers.

Presentation-only: nothing in here does signal processing. Every screen
area of the editor pulls its colors, tree factory, y-autoscale, time-axis
tick formatter and verdict-box writer from this one module so the panels
stay visually consistent and free of duplication.
"""
from __future__ import annotations

from datetime import datetime
import tkinter as tk
from tkinter import ttk

import numpy as np
from matplotlib.ticker import FuncFormatter

# Ride-type colors. ``TYPE_COLORS`` shades the GT bands, ``PRED_COLORS`` the
# detector prediction spans, ``HIGHLIGHT_COLOR`` the current selection outline.
TYPE_COLORS = {"up": "#2ca02c", "down": "#d62728", "outside": "#b8b8b8"}
PRED_COLORS = {"up": "#1f3a5f", "down": "#7d3c98"}
HIGHLIGHT_COLOR = "#e67e22"


def make_tree(parent, columns, on_select):
    """Build a scrolled ``ttk.Treeview`` in ``parent``.

    ``columns`` is a list of ``(id, heading, width, anchor)`` tuples.
    ``on_select`` is bound to ``<<TreeviewSelect>>``. Returns the tree.
    """
    inner = ttk.Frame(parent)
    inner.pack(fill=tk.BOTH, expand=True)
    col_ids = [c[0] for c in columns]
    tree = ttk.Treeview(
        inner, columns=col_ids, show="headings", selectmode="browse",
        height=8,
    )
    for cid, text, w, anchor in columns:
        tree.heading(cid, text=text)
        tree.column(cid, width=w, anchor=anchor)
    tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)
    scroll = ttk.Scrollbar(inner, orient=tk.VERTICAL, command=tree.yview)
    tree.configure(yscrollcommand=scroll.set)
    scroll.pack(side=tk.RIGHT, fill=tk.Y)
    tree.bind("<<TreeviewSelect>>", on_select)
    return tree


def autoscale_y_to_visible(ax, x_lo: float, x_hi: float) -> None:
    """Tighten ``ax.set_ylim`` to the data of every line on this axis that
    falls inside ``[x_lo, x_hi]``. No-op if nothing is in-window."""
    y_chunks: list[np.ndarray] = []
    for line in ax.get_lines():
        xd = np.asarray(line.get_xdata(), dtype=float)
        yd = np.asarray(line.get_ydata(), dtype=float)
        if xd.size == 0:
            continue
        m = (xd >= x_lo) & (xd <= x_hi)
        if m.any():
            y_chunks.append(yd[m])
    if not y_chunks:
        return
    arr = np.concatenate(y_chunks)
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return
    lo, hi = float(arr.min()), float(arr.max())
    pad = (hi - lo) * 0.08 if hi > lo else max(abs(hi) * 0.1, 1.0)
    ax.set_ylim(lo - pad, hi + pad)


def wall_time_str(epoch_ms: float, s_offset: float,
                  *, sub_second: bool = False) -> str:
    """Convert seconds-since-``epoch_ms`` to a ``HH:MM:SS`` wall-clock
    string (or ``HH:MM:SS.mmm`` when ``sub_second``). ``""`` when the
    epoch is unset or the timestamp is out of range."""
    if epoch_ms <= 0:
        return ""
    try:
        wall = datetime.fromtimestamp(epoch_ms / 1000.0 + s_offset)
    except (OverflowError, ValueError, OSError):
        return ""
    if sub_second:
        return wall.strftime("%H:%M:%S.") + f"{wall.microsecond // 1000:03d}"
    return wall.strftime("%H:%M:%S")


def make_tick_formatter(epoch_ms: float) -> FuncFormatter:
    """Two-line x tick formatter: ``"<seconds>s"`` on the first line and
    ``HH:MM:SS`` on the second. ``epoch_ms`` is the wall-clock origin of
    the seconds-from-start axis."""
    def fmt(s_offset, _pos=None):
        head = f"{s_offset:.0f}s"
        wall = wall_time_str(epoch_ms, float(s_offset))
        return f"{head}\n{wall}" if wall else head
    return FuncFormatter(fmt)


def set_verdict(text_widget: tk.Text, text: str) -> None:
    """Replace the contents of a read-only verdict ``tk.Text`` box."""
    text_widget.configure(state="normal")
    text_widget.delete("1.0", "end")
    if text:
        text_widget.insert("1.0", text)
    text_widget.configure(state="disabled")
