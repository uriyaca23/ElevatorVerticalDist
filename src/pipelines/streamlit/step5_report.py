"""Step 5 — Hebrew per-segment PDF report.

Builds a multi-page PDF (cover + per-segment pages + how-to-read) using
ReportLab, with matplotlib-rendered PNG figures for each segment. The
Hebrew text is bidi-reordered through :func:`_rtl` while leaving inline
``<b>``/``<font>`` tags untouched, and any user-supplied dynamic text is
escaped through :func:`_esc` so it can't break the paraparser.

Layout choices that are easy to overlook:

* Hebrew + bold actually requires a registered *font family* (regular +
  bold) — without ``registerFontFamily`` the ``<b>`` tag is silently
  ignored and bold text renders identically to body weight.
* Label / value pairs go into :class:`Table` cells rather than a single
  ``<b>label</b> value`` paragraph, so we never have to mix RTL Hebrew
  with LTR numerics inside one bidi-reordered string.
* The page footer (page number + run timestamp) is drawn through an
  ``onPage`` callback on the Canvas, not as a Flowable.
"""
from __future__ import annotations

import base64
import io
import re as _re
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from .common import (
    LoadedSignal,
    PEAK_STATUS_COLORS,
    RIDE_COLORS,
    SELECTED_COLOR,
    STEP_HOWTO,
    STEP_PREDICT,
    classify_peak,
    find_local_maxima,
    find_matching_prediction,
    goto,
    to_datetime,
    to_datetime_array,
    trapezoid_kernel,
    utcnow_iso,
    valid_segments,
)


# Small translation table for static strings. Kept here so the PDF writer
# and the on-screen preview read from the same place.
HEB = {
    "report_title":   "דוח פייפליין מעלית",
    "subtitle":       "ניתוח מקטעי נסיעה והפרשי גובה",
    "source":         "מקור",
    "generated":      "נוצר בתאריך",
    "samples":        "מספר דגימות",
    "sample_rate":    "קצב דגימה",
    "phone_type":     "סוג מכשיר",
    "phone_id":       "מזהה מכשיר",
    "window":         "חלון זמן",
    "experiment":     "ניסוי",
    "t_start":        "התחלה",
    "t_end":          "סיום",
    "backend":        "מקור נתונים",
    "summary":        "סיכום",
    "metadata":       "נתוני קלט",
    "n_segments":     "מספר מקטעים",
    "n_accepted":     "התקבלו בבקרה",
    "net_dh":         "סך הפרשי גובה",
    "meters":         "מטר",
    "per_seg_table":  "טבלת הפרשי גובה",
    "col_idx":        "מקטע",
    "col_type":       "סוג",
    "col_start":      "התחלה (ש')",
    "col_end":        "סיום (ש')",
    "col_dur":        "משך (ש')",
    "col_dh":         "הפרש גובה (מ')",
    "col_ci":         "רווח סמך (מ')",
    "col_quality":    "ציון איכות",
    "col_accepted":   "התקבל",
    "yes":            "כן",
    "no":             "לא",
    "up":             "עלייה",
    "down":           "ירידה",
    "segment_page":   "מקטע",
    "trap_heading":   "אות האצה עם תבנית הטרפז",
    "corr_heading":   "ציון מתאם סביב המקטע (±30 ש')",
    "status_legend":  "מקרא צבעי פסגות",
    "reject_reason":  "סיבת דחייה",
    "duration":       "משך",
    "page":           "עמוד",
    "of":             "מתוך",
    "trap_fit_heading":  "התאמת טרפז לאות התאוצה",
    "zupt_pos_heading":  "מסלול אינטגרציה (ZUPT) — מיקום משולב",
    "no_trap_fit":       "אין התאמת טרפז זמינה למקטע זה",
    "no_zupt_curve":     "אין מסלול ZUPT זמין למקטע זה",
    "per_algo_heading":  "מדדים לכל אלגוריתם",
    "col_algo":          "אלגוריתם",
    "trap_label":        "טרפז (זוג פולסים)",
    "zupt_label":        "ZUPT (אינטגרציה כפולה)",
    "how_to_read":    "איך לקרוא את הדוח",
    "how_to_read_body": (
        "כל מקטע נסיעה הותאם בפרופיל תנועה מסוג S (S-curve) "
        "על ציר ההאצה האנכי. הפרש הגובה הוא המרחק המוערך, עם סימן, "
        "שהמעלית עברה במקטע. עמודת רווח הסמך מציגה את חצי הרוחב של "
        "רווח סמך של 90% המבוסס על חסם קרמר־ראו (CRB) לאחר כיול "
        "קונפורמלי. מקטעים שמסומנים כ'לא התקבלו' נכשלו במסנן האיכות "
        "הפנימי (התאמת χ², לוג־הסתברות פריור, יחס רווח סמך, אי־הסכמה "
        "בין ZUPT ו־NLS). הערכי ה־Δh שלהם מדווחים, אך יש להתייחס "
        "אליהם כבעלי ביטחון נמוך."
    ),
}


# ---------------------------------------------------------------------------
# Palette
# ---------------------------------------------------------------------------

_PRIMARY      = "#0f2a4a"   # navy — titles, header bars
_ACCENT       = "#1f6feb"   # blue — section headings, primary chips
_UP_COLOR     = "#1f6feb"   # ride-up pill
_DOWN_COLOR   = "#b54a9b"   # ride-down pill
_OK_COLOR     = "#27ae60"   # accepted
_WARN_COLOR   = "#c0392b"   # rejected
_TEXT_MUTED   = "#5b6677"
_BG_SOFT      = "#f5f7fb"
_BORDER_SOFT  = "#cfd4dc"


# ---------------------------------------------------------------------------
# Font handling
# ---------------------------------------------------------------------------

# Ordered (regular_path, bold_path | None, family_name) candidates.
# The first one that exists *and* registers cleanly wins. Bold path is
# optional: when missing we map the regular face to the bold slot too,
# so `<b>` falls through to the regular weight rather than breaking.
_FONT_CANDIDATES: list[tuple[str | None, str | None, str]] = [
    # macOS — Arial ships with a real bold variant and full Hebrew coverage.
    ("/System/Library/Fonts/Supplemental/Arial.ttf",
     "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
     "ArialHeb"),
    ("/Library/Fonts/Arial Unicode.ttf", None, "ArialUniHeb"),
    # macOS Tahoma — also a solid Hebrew face.
    ("/System/Library/Fonts/Supplemental/Tahoma.ttf",
     "/System/Library/Fonts/Supplemental/Tahoma Bold.ttf",
     "TahomaHeb"),
    # LibreOffice / Android Studio bundle Noto Sans Hebrew.
    ("/Applications/LibreOffice.app/Contents/Resources/fonts/truetype/NotoSansHebrew-Regular.ttf",
     "/Applications/LibreOffice.app/Contents/Resources/fonts/truetype/NotoSansHebrew-Bold.ttf",
     "NotoHeb"),
    # Common Linux paths.
    ("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
     "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
     "DejaVuHeb"),
]


def _maybe_register_hebrew_font() -> tuple[str, str]:
    """Register a Hebrew-capable family. Returns ``(regular, bold)``
    ReportLab font names. Falls back to Helvetica when nothing usable
    is on disk; Helvetica has no Hebrew glyphs but at least keeps the
    PDF building.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return "Helvetica", "Helvetica-Bold"

    # matplotlib bundles DejaVuSans as a last resort — append it after
    # the system candidates so we still find *something* on a barebones box.
    candidates = list(_FONT_CANDIDATES)
    try:
        import matplotlib
        mpl_ttf = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates.append((
            str(mpl_ttf / "DejaVuSans.ttf"),
            str(mpl_ttf / "DejaVuSans-Bold.ttf"),
            "DejaVuMpl",
        ))
    except Exception:
        pass

    for reg_path, bold_path, family in candidates:
        if not reg_path or not Path(reg_path).exists():
            continue
        try:
            pdfmetrics.registerFont(TTFont(family, reg_path))
        except Exception:
            continue
        bold_name = family
        if bold_path and Path(bold_path).exists():
            bold_name = f"{family}-Bold"
            try:
                pdfmetrics.registerFont(TTFont(bold_name, bold_path))
            except Exception:
                bold_name = family  # fall back to regular weight
        try:
            pdfmetrics.registerFontFamily(
                family, normal=family, bold=bold_name,
                italic=family, boldItalic=bold_name,
            )
        except Exception:
            pass
        return family, bold_name
    return "Helvetica", "Helvetica-Bold"


# ---------------------------------------------------------------------------
# RTL helpers
# ---------------------------------------------------------------------------

_RL_TAG_RE = _re.compile(r"<[^<>]+>")


def _rtl(s: str) -> str:
    """Bidi-reorder Hebrew strings while preserving ReportLab inline tags.

    python-bidi's ``get_display`` reverses character order for RTL-dominant
    text. Running it over a string that embeds ReportLab paraparser markup
    (``<b>...</b>``, ``<font>...</font>``, etc.) also reverses the tag
    bytes themselves, producing ``</b>...<b>`` — which paraparser then
    rejects with "parse ended with 1 unclosed tags para". We therefore
    apply the bidi reorder only to the plain-text chunks between tags and
    leave the tags in their original LTR ASCII order.
    """
    try:
        from bidi.algorithm import get_display
    except ImportError:
        return s
    if "<" not in s:
        return get_display(s)
    parts: list[str] = []
    idx = 0
    for m in _RL_TAG_RE.finditer(s):
        if m.start() > idx:
            parts.append(get_display(s[idx:m.start()]))
        parts.append(m.group(0))
        idx = m.end()
    if idx < len(s):
        parts.append(get_display(s[idx:]))
    return "".join(parts)


def _esc(s) -> str:
    """Escape `<`, `>`, `&` so dynamic values don't confuse paraparser."""
    return (
        str(s)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
    )


# ---------------------------------------------------------------------------
# Matplotlib helpers
# ---------------------------------------------------------------------------

def _style_axes(ax) -> None:
    """Tight, low-chrome axes shared by every figure."""
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color("#9aa5b1")
        ax.spines[spine].set_linewidth(0.6)
    ax.tick_params(colors="#5b6677", labelsize=8, length=3, width=0.6)
    ax.xaxis.label.set_color("#233044")
    ax.yaxis.label.set_color("#233044")
    ax.grid(True, alpha=0.18, lw=0.5)
    ax.set_facecolor("#fbfcfe")


def _build_main_signal_png(state: dict, segments: pd.DataFrame,
                           selected: int | None = None) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    except Exception:
        return b""
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    t0_ms = state.get("t0_ms")
    dt = to_datetime_array(t, t0_ms)
    fig, ax = plt.subplots(figsize=(8.0, 2.9), dpi=180)
    fig.patch.set_facecolor("white")
    ax.plot(dt, a_vert, color="#233044", lw=0.6, alpha=0.85, label="a_vert")
    ax.plot(dt, a_smooth, color="#e67e22", lw=1.0, label="smoothed")
    for i, row in segments.iterrows():
        try:
            s = float(row["start_s"]); e = float(row["end_s"])
        except (TypeError, ValueError):
            continue
        color = RIDE_COLORS.get(str(row.get("type", "up")), "#777")
        is_sel = (selected is not None and int(i) == selected)
        ax.axvspan(to_datetime(s, t0_ms), to_datetime(e, t0_ms),
                   color=SELECTED_COLOR if is_sel else color,
                   alpha=0.30 if is_sel else 0.16, lw=0)
    locator = AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(AutoDateFormatter(locator))
    ax.set_xlabel("time"); ax.set_ylabel("a_vert (m/s²)")
    ax.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    _style_axes(ax)
    fig.autofmt_xdate()
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _build_segment_page_pngs(
    state: dict, predictions: list[dict],
    t_lo: float, t_hi: float,
) -> tuple[bytes, bytes]:
    """Two PNGs for a segment page:

    1) Zoomed a_vert with the fitted trapezoid overlay (the 'match').
    2) Correlation panel ±30 s around the segment, with peak-status dots.
    """
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.dates import AutoDateFormatter, AutoDateLocator
    except Exception:
        return b"", b""

    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    t0_ms = state.get("t0_ms")
    match = find_matching_prediction(predictions, t_lo, t_hi)

    def _date_axis(ax) -> None:
        loc = AutoDateLocator()
        ax.xaxis.set_major_locator(loc)
        ax.xaxis.set_major_formatter(AutoDateFormatter(loc))

    # --- (1) trapezoid overlay ---
    pad_s = 4.0
    mask = (t >= t_lo - pad_s) & (t <= t_hi + pad_s)
    dt_zoom = to_datetime_array(t[mask], t0_ms)
    fig1, ax1 = plt.subplots(figsize=(7.6, 2.7), dpi=180)
    fig1.patch.set_facecolor("white")
    ax1.plot(dt_zoom, a_vert[mask], color="#233044", lw=0.6, alpha=0.85,
             label="a_vert")
    ax1.plot(dt_zoom, a_smooth[mask], color="#e67e22", lw=1.0,
             label="smoothed")
    ax1.axhline(0, color="#aaa", lw=0.4, ls="--")
    ax1.axvspan(to_datetime(t_lo, t0_ms), to_datetime(t_hi, t0_ms),
                color=SELECTED_COLOR, alpha=0.10, lw=0)
    if match is not None:
        for lobe_key, color in (("lobe1", "#c0392b"), ("lobe2", "#8e44ad")):
            L = match.get(lobe_key) or {}
            try:
                W = float(L["half_width_s"]); f = float(L["frac_flat"])
                A = float(L["a_peak"]); t_c = float(L["t_c"])
            except (KeyError, TypeError, ValueError):
                continue
            tt = np.linspace(t_c - W, t_c + W, 200)
            yy = A * trapezoid_kernel(tt, t_c, W, f)
            ax1.plot(to_datetime_array(tt, t0_ms), yy, color=color, lw=1.8)
            ax1.scatter([to_datetime(t_c, t0_ms)], [A],
                        color=color, s=26, zorder=5,
                        edgecolor="#000", linewidth=0.4)
    _date_axis(ax1)
    ax1.set_xlabel("time"); ax1.set_ylabel("a (m/s²)")
    ax1.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    _style_axes(ax1)
    fig1.autofmt_xdate()
    fig1.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=180, facecolor="white")
    plt.close(fig1)

    # --- (2) correlation panel ±30 s with peak colors ---
    t_min = float(t[0]); t_max = float(t[-1])
    wlo = max(t_min, t_lo - 30.0); whi = min(t_max, t_hi + 30.0)
    mask2 = (t >= wlo) & (t <= whi)
    pos_r2 = np.asarray(state["best_pos_r2"])
    neg_r2 = np.asarray(state["best_neg_r2"])
    pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
    neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)
    dt_corr = to_datetime_array(t[mask2], t0_ms)

    fig2, ax2 = plt.subplots(figsize=(7.6, 2.7), dpi=180)
    fig2.patch.set_facecolor("white")
    ax2.plot(dt_corr, pos_plot[mask2], color="#2980b9", lw=0.9,
             label="max R² (+)")
    ax2.plot(dt_corr, neg_plot[mask2], color="#c0392b", lw=0.9,
             label="max R² (−)")
    cfg = state.get("config")
    if cfg is not None:
        ax2.axhline(cfg.r2_peak_thresh, color="#888", lw=0.5, ls="--")
    ax2.axvspan(to_datetime(t_lo, t0_ms), to_datetime(t_hi, t0_ms),
                color=SELECTED_COLOR, alpha=0.10, lw=0)
    peaks_pos = find_local_maxima(pos_r2, t, wlo, whi)
    peaks_neg = find_local_maxima(neg_r2, t, wlo, whi)
    for sign, peaks, arr in ((+1, peaks_pos, pos_r2), (-1, peaks_neg, neg_r2)):
        for i in peaks:
            tag = classify_peak(state, i, sign, predictions)
            ax2.scatter([to_datetime(t[i], t0_ms)], [arr[i]],
                        color=PEAK_STATUS_COLORS.get(tag, "#000"),
                        s=30, zorder=5, edgecolor="#000", linewidth=0.3)
    ax2.set_ylim(0, 1.05)
    ax2.set_xlim(to_datetime(wlo, t0_ms), to_datetime(whi, t0_ms))
    _date_axis(ax2)
    ax2.set_xlabel("time"); ax2.set_ylabel("R² (per sign)")
    ax2.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    _style_axes(ax2)
    fig2.autofmt_xdate()
    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=180, facecolor="white")
    plt.close(fig2)

    return buf1.getvalue(), buf2.getvalue()


def _build_trap_fit_png(meta: dict) -> bytes:
    """PNG of the fitted trapezoid template overlaid on the smoothed accel
    signal (matching editor.py's Prediction-tab top panel).
    """
    if not meta:
        return b""
    t_sec = meta.get("t_sec")
    a_smooth = meta.get("a_smooth")
    a_template = meta.get("a_template")
    if t_sec is None or a_smooth is None or a_template is None:
        return b""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    t = np.asarray(t_sec, dtype=float)
    s = np.asarray(a_smooth, dtype=float)
    tpl = np.asarray(a_template, dtype=float)
    if t.size == 0:
        return b""

    fig, ax = plt.subplots(figsize=(7.6, 2.7), dpi=180)
    fig.patch.set_facecolor("white")
    ax.plot(t, s, color="#2c3e50", lw=0.9, label="a_smooth")
    ax.plot(t, tpl, color="#c0392b", lw=1.6, alpha=0.9,
            label="trapezoid template")
    ax.axhline(0.0, color="#888", lw=0.4, ls="--", alpha=0.6)

    params = meta.get("params") or {}
    for key in ("t_c1", "t_c2"):
        v = params.get(key)
        if v is not None and np.isfinite(v):
            ax.axvline(float(v), color="#c0392b", lw=0.6, ls=":", alpha=0.7)
    if params:
        sign = int(params.get("sign", 0))
        annot = (
            f"A_used={params.get('A_used', float('nan')):.2f} m/s²  "
            f"W={params.get('W', float('nan')):.2f}s  "
            f"f={params.get('f', float('nan')):.2f}  "
            f"sign={sign:+d}\n"
            f"t_c1={params.get('t_c1', float('nan')):.2f}s  "
            f"t_c2={params.get('t_c2', float('nan')):.2f}s  "
            f"R²={params.get('joint_r2', float('nan')):.3f}  "
            f"v_peak={params.get('v_peak_measured', float('nan')):+.2f} m/s"
        )
        ax.text(
            0.01, 0.98, annot, transform=ax.transAxes,
            ha="left", va="top", fontsize=7, family="monospace",
            bbox=dict(facecolor="#ffffff", alpha=0.85,
                      edgecolor="#888", boxstyle="round,pad=0.3"),
            zorder=20,
        )
    ax.set_xlabel("t (s, ride-local)"); ax.set_ylabel("a (m/s²)")
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)
    _style_axes(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


def _build_zupt_pos_png(meta: dict) -> bytes:
    """PNG of the ZUPT integrated position curve with the motion window
    shaded (matching editor.py's Prediction-tab bottom panel).
    """
    if not meta:
        return b""
    pos = meta.get("pos_curve")
    if pos is None:
        return b""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    pos_arr = np.asarray(pos, dtype=float)
    if pos_arr.size == 0:
        return b""

    fig, ax = plt.subplots(figsize=(7.6, 2.4), dpi=180)
    fig.patch.set_facecolor("white")
    idx = np.arange(pos_arr.size)
    ax.plot(idx, pos_arr, color="#27ae60", lw=1.1, label="pos(t)")
    start = meta.get("start_idx"); end = meta.get("end_idx")
    if (start is not None and end is not None
            and np.isfinite(start) and np.isfinite(end)
            and int(end) > int(start)):
        ax.axvspan(int(start), int(end), color="#27ae60", alpha=0.15,
                   label="motion window")
    ax.axhline(0.0, color="#888", lw=0.4, ls="--", alpha=0.6)
    info_bits: list[str] = []
    n_active = meta.get("n_active")
    active_frac = meta.get("active_fraction")
    method = meta.get("method", "")
    if n_active is not None:
        info_bits.append(f"n_active={int(n_active)}")
    if active_frac is not None and np.isfinite(active_frac):
        info_bits.append(f"active_frac={float(active_frac):.2f}")
    if method:
        info_bits.append(f"method={method}")
    if info_bits:
        ax.text(
            0.01, 0.98, "  ".join(info_bits), transform=ax.transAxes,
            ha="left", va="top", fontsize=7, family="monospace",
            bbox=dict(facecolor="#ffffff", alpha=0.85,
                      edgecolor="#888", boxstyle="round,pad=0.3"),
            zorder=20,
        )
    ax.set_xlabel("sample index"); ax.set_ylabel("pos (m)")
    ax.legend(loc="lower right", fontsize=7, frameon=False, ncol=2)
    _style_axes(ax)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=180, facecolor="white")
    plt.close(fig)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# PDF flowable factories
# ---------------------------------------------------------------------------

def _hero_band(title: str, subtitle: str, font: str, bold_font: str,
               width_cm: float):
    """Coloured title band. Built as a 1-cell Table so we can paint a
    background and round-corner-ish padding without dropping to Canvas."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    title_p = Paragraph(
        _rtl(title),
        ParagraphStyle(
            "HeroTitle", fontName=bold_font, fontSize=22,
            alignment=TA_RIGHT, textColor=rl_colors.white, leading=26,
        ),
    )
    sub_p = Paragraph(
        _rtl(subtitle),
        ParagraphStyle(
            "HeroSub", fontName=font, fontSize=11,
            alignment=TA_RIGHT,
            textColor=rl_colors.HexColor("#cfe0ff"), leading=14,
        ),
    )
    tbl = Table(
        [[title_p], [sub_p]],
        colWidths=[width_cm * cm],
    )
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, -1), rl_colors.HexColor(_PRIMARY)),
        ("LEFTPADDING",   (0, 0), (-1, -1), 18),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 18),
        ("TOPPADDING",    (0, 0), (0, 0), 14),
        ("BOTTOMPADDING", (0, 0), (0, 0), 0),
        ("TOPPADDING",    (0, 1), (0, 1), 0),
        ("BOTTOMPADDING", (0, 1), (0, 1), 14),
        ("LINEBELOW",     (0, -1), (-1, -1), 3,
                          rl_colors.HexColor(_ACCENT)),
    ]))
    return tbl


def _section_heading(text: str, bold_font: str):
    """Small underlined section header used between blocks."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.platypus import Paragraph

    return Paragraph(
        _rtl(text),
        ParagraphStyle(
            "SectionH", fontName=bold_font, fontSize=12.5,
            alignment=TA_RIGHT,
            textColor=rl_colors.HexColor(_PRIMARY),
            spaceBefore=8, spaceAfter=4,
            borderPadding=0,
        ),
    )


def _metadata_table(meta_rows: list[tuple[str, str]],
                    font: str, bold_font: str, width_cm: float):
    """Two-column Hebrew metadata table.

    Right column = bold Hebrew label. Left column = LTR value. Built as
    separate cells so we never have to bidi-reorder a label/value mix.
    """
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    data = [[_rtl(v), _rtl(label)] for label, v in meta_rows]
    tbl = Table(
        data,
        colWidths=[(width_cm - 5.5) * cm, 5.5 * cm],
        hAlign="RIGHT",
    )
    tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (0, -1), font),       # values
        ("FONTNAME",     (1, 0), (1, -1), bold_font),  # labels
        ("FONTSIZE",     (0, 0), (-1, -1), 10),
        ("TEXTCOLOR",    (1, 0), (1, -1),
                         rl_colors.HexColor(_PRIMARY)),
        ("TEXTCOLOR",    (0, 0), (0, -1),
                         rl_colors.HexColor("#1a2436")),
        ("ALIGN",        (0, 0), (0, -1), "LEFT"),
        ("ALIGN",        (1, 0), (1, -1), "RIGHT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",  (0, 0), (-1, -1), 8),
        ("RIGHTPADDING", (0, 0), (-1, -1), 8),
        ("TOPPADDING",   (0, 0), (-1, -1), 5),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 5),
        ("ROWBACKGROUNDS", (0, 0), (-1, -1),
            [rl_colors.HexColor(_BG_SOFT), rl_colors.white]),
        ("LINEBELOW",    (0, 0), (-1, -2), 0.25,
                         rl_colors.HexColor(_BORDER_SOFT)),
        ("BOX",          (0, 0), (-1, -1), 0.4,
                         rl_colors.HexColor(_BORDER_SOFT)),
    ]))
    return tbl


def _summary_tiles(n_segments: int, n_accepted: int, total_dh: float,
                   font: str, bold_font: str, width_cm: float):
    """Three coloured KPI tiles, side-by-side."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    val_style = ParagraphStyle(
        "TileVal", fontName=bold_font, fontSize=22, alignment=TA_CENTER,
        textColor=rl_colors.white, leading=26,
    )
    lbl_style = ParagraphStyle(
        "TileLbl", fontName=font, fontSize=10, alignment=TA_CENTER,
        textColor=rl_colors.HexColor("#eaf2ff"), leading=12,
    )

    def _cell(value: str, label: str):
        return [
            Paragraph(_rtl(value), val_style),
            Paragraph(_rtl(label), lbl_style),
        ]

    tile_w = (width_cm - 0.6) / 3.0  # 0.3 cm gap on each side
    n_rej = max(n_segments - n_accepted, 0)

    cells = [
        _cell(str(n_segments), HEB["n_segments"]),
        _cell(f"{n_accepted} / {n_segments}", HEB["n_accepted"]),
        _cell(f"{total_dh:+.2f} {HEB['meters']}", HEB["net_dh"]),
    ]

    # Lay cells out as one row of 3 sub-tables so each can have its own
    # background colour without bleeding into the others.
    tile_colors = [
        _ACCENT, _OK_COLOR if n_rej == 0 else _PRIMARY,
        _UP_COLOR if total_dh >= 0 else _DOWN_COLOR,
    ]
    tile_tables = []
    for (val_p, lbl_p), bg in zip(cells, tile_colors):
        t = Table([[val_p], [lbl_p]], colWidths=[tile_w * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), rl_colors.HexColor(bg)),
            ("LEFTPADDING",   (0, 0), (-1, -1), 6),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 6),
            ("TOPPADDING",    (0, 0), (-1, -1), 10),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 10),
            ("LINEABOVE",     (0, 0), (-1, 0), 3,
                              rl_colors.HexColor(_PRIMARY)),
        ]))
        tile_tables.append(t)

    outer = Table(
        [tile_tables],
        colWidths=[tile_w * cm] * 3,
    )
    outer.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    # Inject column gaps via a wrapping spacer-equivalent
    outer._argW = [tile_w * cm, tile_w * cm, tile_w * cm]
    return outer


def _overview_table(rows: list[dict], font: str, bold_font: str,
                    width_cm: float):
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    # Right-to-left column order so the visual leftmost column is the
    # rightmost-meaning one in Hebrew reading direction. Final column
    # (rightmost) is the segment index — what the eye lands on first.
    header = [HEB["col_accepted"], HEB["col_quality"], HEB["col_ci"],
              HEB["col_dh"], HEB["col_dur"], HEB["col_end"],
              HEB["col_start"], HEB["col_type"], HEB["col_idx"]]
    data = [[_rtl(h) for h in header]]
    type_col_styles: list[tuple] = []
    accept_col_styles: list[tuple] = []
    for ridx, r in enumerate(rows, start=1):
        yn = HEB["yes"] if r["accepted"] else HEB["no"]
        rt_heb = HEB["up"] if r["type"] == "up" else HEB["down"]
        data.append([
            _rtl(yn),
            f"{r['quality_score']:.1f}" if np.isfinite(r['quality_score']) else "—",
            f"{r['ci_half_width']:.2f}" if np.isfinite(r['ci_half_width']) else "—",
            f"{r['delta_height_m']:+.2f}" if np.isfinite(r['delta_height_m']) else "—",
            f"{r['duration_s']:.1f}",
            f"{r['end_s']:.1f}",
            f"{r['start_s']:.1f}",
            _rtl(rt_heb),
            str(r["segment"]),
        ])
        # Per-row colouring for the type & accepted cells.
        type_color = _UP_COLOR if r["type"] == "up" else _DOWN_COLOR
        type_col_styles.append(("TEXTCOLOR", (7, ridx), (7, ridx),
                                rl_colors.HexColor(type_color)))
        accept_color = _OK_COLOR if r["accepted"] else _WARN_COLOR
        accept_col_styles.append(("TEXTCOLOR", (0, ridx), (0, ridx),
                                  rl_colors.HexColor(accept_color)))

    # Distribute width: index narrow, type narrow, the rest equal.
    n_cols = len(header)
    narrow = 1.4 * cm
    wide_total = (width_cm * cm) - narrow * 3
    wide = wide_total / (n_cols - 3)
    col_widths = [wide] * (n_cols - 3) + [narrow, narrow, narrow]

    tbl = Table(data, repeatRows=1, hAlign="RIGHT", colWidths=col_widths)
    tbl.setStyle(TableStyle([
        # Header
        ("BACKGROUND",   (0, 0), (-1, 0), rl_colors.HexColor(_PRIMARY)),
        ("TEXTCOLOR",    (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), bold_font),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 8),
        ("TOPPADDING",   (0, 0), (-1, 0), 8),
        # Body
        ("FONTNAME",     (0, 1), (-1, -1), font),
        ("FONTSIZE",     (0, 1), (-1, -1), 8.5),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [rl_colors.white, rl_colors.HexColor(_BG_SOFT)]),
        ("LINEBELOW",    (0, 0), (-1, 0), 1,
                         rl_colors.HexColor(_ACCENT)),
        ("INNERGRID",    (0, 1), (-1, -1), 0.25,
                         rl_colors.HexColor(_BORDER_SOFT)),
        ("BOX",          (0, 0), (-1, -1), 0.4,
                         rl_colors.HexColor(_BORDER_SOFT)),
        # Type & accepted highlights
        ("FONTNAME",     (7, 1), (7, -1), bold_font),
        ("FONTNAME",     (0, 1), (0, -1), bold_font),
        *type_col_styles,
        *accept_col_styles,
    ]))
    return tbl


def _segment_header(r: dict, font: str, bold_font: str, width_cm: float):
    """Coloured header strip for a per-segment page.

    Right-most cell is a type pill (עלייה / ירידה in matching colour).
    """
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER, TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    rt_heb = HEB["up"] if r["type"] == "up" else HEB["down"]
    type_color = _UP_COLOR if r["type"] == "up" else _DOWN_COLOR

    title_text = (
        f"{HEB['segment_page']} #{r['segment']}"
    )
    sub_text = f"{r['start_s']:.1f}–{r['end_s']:.1f} ש'   ·   "
    sub_text += f"{HEB['duration']}: {r['duration_s']:.1f} ש'"

    title_p = Paragraph(
        _rtl(title_text),
        ParagraphStyle(
            "SegTitle", fontName=bold_font, fontSize=16, alignment=TA_RIGHT,
            textColor=rl_colors.HexColor(_PRIMARY), leading=20,
        ),
    )
    sub_p = Paragraph(
        _rtl(sub_text),
        ParagraphStyle(
            "SegSub", fontName=font, fontSize=10, alignment=TA_RIGHT,
            textColor=rl_colors.HexColor(_TEXT_MUTED), leading=14,
        ),
    )
    pill_p = Paragraph(
        _rtl(rt_heb),
        ParagraphStyle(
            "Pill", fontName=bold_font, fontSize=12, alignment=TA_CENTER,
            textColor=rl_colors.white, leading=14,
        ),
    )
    pill_w = 2.6 * cm
    text_w = (width_cm - 2.6) * cm

    title_block = Table([[title_p], [sub_p]], colWidths=[text_w])
    title_block.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 12),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    pill_cell = Table([[pill_p]], colWidths=[pill_w], rowHeights=[1.0 * cm])
    pill_cell.setStyle(TableStyle([
        ("BACKGROUND",    (0, 0), (-1, -1), rl_colors.HexColor(type_color)),
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 4),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
        ("TOPPADDING",    (0, 0), (-1, -1), 4),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 4),
    ]))

    outer = Table([[pill_cell, title_block]],
                  colWidths=[pill_w, text_w])
    outer.setStyle(TableStyle([
        ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
        ("LINEBELOW",     (0, 0), (-1, -1), 1.2,
                          rl_colors.HexColor(_BORDER_SOFT)),
    ]))
    return outer


def _segment_metric_cards(r: dict, font: str, bold_font: str,
                          width_cm: float):
    """Four metric tiles: Δh, CI half-width, quality, accepted."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_CENTER
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    dh = r["delta_height_m"]
    ci = r["ci_half_width"]
    q = r["quality_score"]
    accepted = bool(r["accepted"])

    dh_str = f"{dh:+.2f} מ'" if np.isfinite(dh) else "—"
    ci_str = f"±{ci:.2f} מ'" if np.isfinite(ci) else "—"
    q_str  = f"{q:.1f}" if np.isfinite(q) else "—"
    yn     = HEB["yes"] if accepted else HEB["no"]

    cards = [
        (HEB["col_dh"],       dh_str,
         _UP_COLOR if (np.isfinite(dh) and dh >= 0) else _DOWN_COLOR),
        (HEB["col_ci"],       ci_str,  _ACCENT),
        (HEB["col_quality"],  q_str,   _PRIMARY),
        (HEB["col_accepted"], yn,
         _OK_COLOR if accepted else _WARN_COLOR),
    ]

    val_style = ParagraphStyle(
        "MetricVal", fontName=bold_font, fontSize=15, alignment=TA_CENTER,
        textColor=rl_colors.HexColor(_PRIMARY), leading=18,
    )
    lbl_style = ParagraphStyle(
        "MetricLbl", fontName=font, fontSize=8.5, alignment=TA_CENTER,
        textColor=rl_colors.HexColor(_TEXT_MUTED), leading=10,
    )

    sub_tables = []
    for label, value, accent in cards:
        val_p = Paragraph(_rtl(value), val_style)
        lbl_p = Paragraph(_rtl(label), lbl_style)
        t = Table([[val_p], [lbl_p]],
                  colWidths=[((width_cm - 0.6) / 4.0) * cm])
        t.setStyle(TableStyle([
            ("BACKGROUND",    (0, 0), (-1, -1), rl_colors.white),
            ("BOX",           (0, 0), (-1, -1), 0.6,
                              rl_colors.HexColor(_BORDER_SOFT)),
            ("LINEABOVE",     (0, 0), (-1, 0), 2.5,
                              rl_colors.HexColor(accent)),
            ("LEFTPADDING",   (0, 0), (-1, -1), 4),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 4),
            ("TOPPADDING",    (0, 0), (0, 0), 8),
            ("BOTTOMPADDING", (0, 0), (0, 0), 2),
            ("TOPPADDING",    (0, 1), (0, 1), 0),
            ("BOTTOMPADDING", (0, 1), (0, 1), 8),
        ]))
        sub_tables.append(t)

    cw = (width_cm - 0.6) / 4.0
    outer = Table([sub_tables], colWidths=[cw * cm] * 4)
    outer.setStyle(TableStyle([
        ("LEFTPADDING",   (0, 0), (-1, -1), 0),
        ("RIGHTPADDING",  (0, 0), (-1, -1), 0),
        ("TOPPADDING",    (0, 0), (-1, -1), 0),
        ("BOTTOMPADDING", (0, 0), (-1, -1), 0),
    ]))
    return outer


def _per_algo_metrics_table(rows_for_seg: list[tuple[str, dict | None]],
                            font: str, bold_font: str, width_cm: float):
    """Per-algorithm Δh / CI / quality / accepted table for one segment.

    Argument is a list of ``(label, row | None)`` pairs in display order
    (typically trap then zupt). ``None`` means the algorithm did not
    return a row for this segment.
    """
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.units import cm
    from reportlab.platypus import Table, TableStyle

    header = [HEB["col_accepted"], HEB["col_quality"], HEB["col_ci"],
              HEB["col_dh"], HEB["col_algo"]]
    data = [[_rtl(h) for h in header]]
    accept_col_styles: list[tuple] = []
    for ridx, (label, r) in enumerate(rows_for_seg, start=1):
        if r is None:
            data.append([
                "—", "—", "—", "—", _rtl(label),
            ])
            continue
        dh = r.get("delta_height_m", float("nan"))
        ci = r.get("ci_half_width", float("nan"))
        q = r.get("quality_score", float("nan"))
        accepted = bool(r.get("accepted"))
        yn = HEB["yes"] if accepted else HEB["no"]
        data.append([
            _rtl(yn),
            f"{q:.2f}" if np.isfinite(q) else "—",
            f"±{ci:.2f}" if np.isfinite(ci) else "—",
            f"{dh:+.2f}" if np.isfinite(dh) else "—",
            _rtl(label),
        ])
        accept_color = _OK_COLOR if accepted else _WARN_COLOR
        accept_col_styles.append(("TEXTCOLOR", (0, ridx), (0, ridx),
                                  rl_colors.HexColor(accept_color)))

    n_cols = len(header)
    label_w = 4.6 * cm
    metric_w = ((width_cm * cm) - label_w) / (n_cols - 1)
    col_widths = [metric_w] * (n_cols - 1) + [label_w]

    tbl = Table(data, repeatRows=1, hAlign="RIGHT", colWidths=col_widths)
    tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0, 0), (-1, 0), rl_colors.HexColor(_PRIMARY)),
        ("TEXTCOLOR",    (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",     (0, 0), (-1, 0), bold_font),
        ("FONTSIZE",     (0, 0), (-1, 0), 9),
        ("BOTTOMPADDING",(0, 0), (-1, 0), 6),
        ("TOPPADDING",   (0, 0), (-1, 0), 6),
        ("FONTNAME",     (0, 1), (-1, -1), font),
        ("FONTNAME",     (-1, 1), (-1, -1), bold_font),
        ("FONTSIZE",     (0, 1), (-1, -1), 9),
        ("ALIGN",        (0, 0), (-1, -1), "CENTER"),
        ("ALIGN",        (-1, 0), (-1, -1), "RIGHT"),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
            [rl_colors.white, rl_colors.HexColor(_BG_SOFT)]),
        ("LINEBELOW",    (0, 0), (-1, 0), 1,
                         rl_colors.HexColor(_ACCENT)),
        ("INNERGRID",    (0, 1), (-1, -1), 0.25,
                         rl_colors.HexColor(_BORDER_SOFT)),
        ("BOX",          (0, 0), (-1, -1), 0.4,
                         rl_colors.HexColor(_BORDER_SOFT)),
        *accept_col_styles,
    ]))
    return tbl


def _peak_legend(font: str, bold_font: str, width_cm: float):
    """Coloured-dot legend for peak status, laid out as a wide table."""
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import Paragraph, Table, TableStyle

    items = list(PEAK_STATUS_COLORS.items())
    cells = []
    chip_style = ParagraphStyle(
        "Chip", fontName=font, fontSize=8, alignment=TA_RIGHT,
        textColor=rl_colors.HexColor("#1a2436"), leading=10,
    )
    for tag, color in items:
        text = f'<font color="{color}">●</font>  {_esc(tag)}'
        cells.append(Paragraph(text, chip_style))

    # Lay as 4-per-row grid.
    rows = [cells[i:i + 4] for i in range(0, len(cells), 4)]
    while len(rows[-1]) < 4:
        rows[-1].append("")
    cw = width_cm / 4.0
    tbl = Table(rows, colWidths=[cw * cm] * 4)
    tbl.setStyle(TableStyle([
        ("FONTNAME",     (0, 0), (-1, -1), font),
        ("FONTSIZE",     (0, 0), (-1, -1), 8),
        ("BOTTOMPADDING",(0, 0), (-1, -1), 3),
        ("TOPPADDING",   (0, 0), (-1, -1), 3),
        ("VALIGN",       (0, 0), (-1, -1), "MIDDLE"),
    ]))
    legend_caption = Paragraph(
        _rtl(f'<b>{_esc(HEB["status_legend"])}</b>'),
        ParagraphStyle(
            "LegCap", fontName=bold_font, fontSize=9, alignment=TA_RIGHT,
            textColor=rl_colors.HexColor(_PRIMARY), leading=12,
            spaceBefore=4, spaceAfter=2,
        ),
    )
    return [legend_caption, tbl]


# ---------------------------------------------------------------------------
# Page footer (drawn via canvas onPage)
# ---------------------------------------------------------------------------

class _FooterCanvas:
    """Closure builder so the onPage handler can see font + timestamp."""

    @staticmethod
    def make(font: str, stamp: str):
        from reportlab.lib import colors as rl_colors

        def _draw(canvas, doc):
            canvas.saveState()
            page_w = doc.pagesize[0]
            page_n = canvas.getPageNumber()
            try:
                canvas.setFont(font, 8)
            except Exception:
                canvas.setFont("Helvetica", 8)
            canvas.setFillColor(rl_colors.HexColor(_TEXT_MUTED))
            # Page number — left side (LTR is fine for numbers).
            canvas.drawString(
                doc.leftMargin, 0.8 * 28,
                f"{page_n}",
            )
            # Run timestamp — right side.
            canvas.drawRightString(
                page_w - doc.rightMargin, 0.8 * 28,
                stamp,
            )
            # Thin separator line above footer.
            canvas.setStrokeColor(rl_colors.HexColor(_BORDER_SOFT))
            canvas.setLineWidth(0.4)
            canvas.line(
                doc.leftMargin, 1.4 * 28,
                page_w - doc.rightMargin, 1.4 * 28,
            )
            canvas.restoreState()
        return _draw


# ---------------------------------------------------------------------------
# Top-level builder
# ---------------------------------------------------------------------------

def _build_pdf(
    loaded: LoadedSignal,
    state: dict,
    predictions: list[dict],
    segments: pd.DataFrame,
    rows: list[dict],
    rows_by_algo: dict[str, list[dict]] | None = None,
) -> bytes:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer,
    )

    font, bold_font = _maybe_register_hebrew_font()

    buf = io.BytesIO()
    page_w, page_h = A4
    left_margin = right_margin = 1.6 * cm
    top_margin = 1.4 * cm
    bottom_margin = 1.6 * cm
    content_w_cm = (page_w - left_margin - right_margin) / cm

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=left_margin, rightMargin=right_margin,
        topMargin=top_margin, bottomMargin=bottom_margin,
        title="Boutique Pipeline — Hebrew Report",
        author="ElevatorVerticalDist",
    )

    body_style = ParagraphStyle(
        name="Body", fontName=font, fontSize=10,
        alignment=TA_RIGHT, leading=15,
        textColor=rl_colors.HexColor("#1a2436"),
    )

    story: list = []

    # ---------------- Cover ----------------
    story.append(_hero_band(
        HEB["report_title"], HEB["subtitle"],
        font, bold_font, content_w_cm,
    ))
    story.append(Spacer(1, 0.5 * cm))

    # Metadata block (right column = label, left column = value).
    meta_items: list[tuple[str, str]] = [
        (HEB["source"],      str(loaded.source)),
        (HEB["generated"],   utcnow_iso()),
        (HEB["samples"],     str(loaded.meta.get("samples", "?"))),
        (HEB["sample_rate"], str(loaded.meta.get("sample_rate", "?"))),
    ]
    for k, v in loaded.meta.items():
        if k in ("samples", "sample_rate"):
            continue
        label = HEB.get(k, k)
        meta_items.append((label, str(v)))

    story.append(_section_heading(HEB["metadata"], bold_font))
    story.append(_metadata_table(meta_items, font, bold_font, content_w_cm))

    # Summary KPIs.
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    total_dh = float(df["delta_height_m"].dropna().sum()) if not df.empty else 0.0
    n_accepted = int(df["accepted"].sum()) if not df.empty else 0

    story.append(Spacer(1, 0.4 * cm))
    story.append(_section_heading(HEB["summary"], bold_font))
    story.append(_summary_tiles(
        len(df), n_accepted, total_dh, font, bold_font, content_w_cm,
    ))

    # Overview signal.
    main_png = _build_main_signal_png(state, segments, selected=None)
    if main_png:
        story.append(Spacer(1, 0.4 * cm))
        story.append(Image(io.BytesIO(main_png),
                           width=content_w_cm * cm,
                           height=content_w_cm * cm * (2.9 / 8.0)))

    # Overview table.
    if rows:
        story.append(Spacer(1, 0.4 * cm))
        story.append(_section_heading(HEB["per_seg_table"], bold_font))
        story.append(_overview_table(rows, font, bold_font, content_w_cm))

    # ---------------- Per-segment pages ----------------
    rows_by_algo = rows_by_algo or {}
    trap_rows = rows_by_algo.get("trap", [])
    zupt_rows = rows_by_algo.get("zupt", [])

    def _row_for(rs: list[dict], seg_id: int) -> dict | None:
        return next(
            (x for x in rs if int(x.get("segment", -1)) == seg_id), None,
        )

    for r in rows:
        seg_id = int(r["segment"])
        trap_r = _row_for(trap_rows, seg_id) or r
        zupt_r = _row_for(zupt_rows, seg_id)

        story.append(PageBreak())
        story.append(_segment_header(r, font, bold_font, content_w_cm))

        if r.get("reject_reason"):
            story.append(Spacer(1, 0.25 * cm))
            story.append(Paragraph(
                _rtl(f"<b>{_esc(HEB['reject_reason'])}:</b> "
                     f"{_esc(r['reject_reason'])}"),
                ParagraphStyle(
                    "Reject", parent=body_style,
                    textColor=rl_colors.HexColor(_WARN_COLOR),
                    fontName=font, alignment=TA_RIGHT,
                ),
            ))

        # Per-algorithm metrics — Δh / CI / quality / accepted side-by-side
        # so the report shows both estimators' verdicts, not just the
        # primary's.
        per_algo_pairs: list[tuple[str, dict | None]] = [
            (HEB["trap_label"], trap_r),
            (HEB["zupt_label"], zupt_r),
        ]
        story.append(Spacer(1, 0.35 * cm))
        story.append(_section_heading(HEB["per_algo_heading"], bold_font))
        story.append(_per_algo_metrics_table(
            per_algo_pairs, font, bold_font, content_w_cm,
        ))

        story.append(Spacer(1, 0.4 * cm))
        trap_png, corr_png = _build_segment_page_pngs(
            state, predictions, float(r["start_s"]), float(r["end_s"]),
        )
        if trap_png:
            story.append(_section_heading(HEB["trap_heading"], bold_font))
            story.append(Image(io.BytesIO(trap_png),
                               width=content_w_cm * cm,
                               height=content_w_cm * cm * (2.7 / 7.6)))
        if corr_png:
            story.append(Spacer(1, 0.25 * cm))
            story.append(_section_heading(HEB["corr_heading"], bold_font))
            story.append(Image(io.BytesIO(corr_png),
                               width=content_w_cm * cm,
                               height=content_w_cm * cm * (2.7 / 7.6)))
            story.append(Spacer(1, 0.2 * cm))
            story.extend(_peak_legend(font, bold_font, content_w_cm))

        # Trapezoid template fit + ZUPT integrated position — same panels
        # editor.py exposes in its Prediction tab. Both come straight from
        # the prediction stage's `meta`, so they show what each estimator
        # actually fitted, not the segmentation-stage match.
        trap_meta = (trap_r or {}).get("meta") or {}
        trap_fit_png = _build_trap_fit_png(trap_meta)
        if trap_fit_png:
            story.append(Spacer(1, 0.3 * cm))
            story.append(_section_heading(
                HEB["trap_fit_heading"], bold_font,
            ))
            story.append(Image(io.BytesIO(trap_fit_png),
                               width=content_w_cm * cm,
                               height=content_w_cm * cm * (2.7 / 7.6)))
        else:
            story.append(Spacer(1, 0.2 * cm))
            story.append(_section_heading(
                HEB["trap_fit_heading"], bold_font,
            ))
            story.append(Paragraph(
                _rtl(_esc(HEB["no_trap_fit"])), body_style,
            ))

        zupt_meta = (zupt_r or {}).get("meta") or {}
        zupt_pos_png = _build_zupt_pos_png(zupt_meta)
        if zupt_pos_png:
            story.append(Spacer(1, 0.25 * cm))
            story.append(_section_heading(
                HEB["zupt_pos_heading"], bold_font,
            ))
            story.append(Image(io.BytesIO(zupt_pos_png),
                               width=content_w_cm * cm,
                               height=content_w_cm * cm * (2.4 / 7.6)))
        else:
            story.append(Spacer(1, 0.2 * cm))
            story.append(_section_heading(
                HEB["zupt_pos_heading"], bold_font,
            ))
            story.append(Paragraph(
                _rtl(_esc(HEB["no_zupt_curve"])), body_style,
            ))

    # ---------------- How to read ----------------
    story.append(PageBreak())
    story.append(_section_heading(HEB["how_to_read"], bold_font))
    story.append(Paragraph(_rtl(HEB["how_to_read_body"]), body_style))

    stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    on_page = _FooterCanvas.make(font, stamp)
    doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Streamlit entry point
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Signatures CSV — long-form record of every segment's trapezoid shape
# plus the session metadata, suitable for cross-session analysis later.
# ---------------------------------------------------------------------------

_FS_UNSAFE_RE = _re.compile(r"[^A-Za-z0-9._-]+")


def _fs_safe(s: str) -> str:
    """Reduce ``s`` to characters legal in a filename on every common OS.
    Collapses runs of unsafe characters into a single underscore and
    trims leading / trailing underscores.
    """
    cleaned = _FS_UNSAFE_RE.sub("_", str(s or "")).strip("_")
    return cleaned or "session"


def _signatures_filename(loaded: LoadedSignal) -> str:
    """``{tool_id}_{start}_{end}.csv`` for phone-DB sessions, falling
    back to ``{filename_stem}_{utc_stamp}.csv`` for uploaded files.
    """
    meta = loaded.meta or {}
    mode = str(meta.get("input_mode", "")).lower()
    if mode == "phone":
        tool_id = _fs_safe(meta.get("phone_id", "tool"))
        start = _fs_safe(meta.get("session_start_iso", ""))
        end = _fs_safe(meta.get("session_end_iso", ""))
        if start and end:
            return f"{tool_id}_{start}_{end}.csv"
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        return f"{tool_id}_{stamp}.csv"
    # File-upload mode or unknown: stem the filename + utc timestamp.
    filename = str(meta.get("filename", "session"))
    stem = _fs_safe(Path(filename).stem)
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    return f"{stem}_{stamp}.csv"


def _row_for_segment(rs: list[dict], seg_id: int) -> dict | None:
    return next(
        (x for x in rs if int(x.get("segment", -1)) == seg_id), None,
    )


def _build_signatures_csv(
    loaded: LoadedSignal,
    segments: pd.DataFrame,
    predictions: list[dict],
    rows_by_algo: dict[str, list[dict]],
    overrides: dict[int, dict],
) -> bytes:
    """One row per segment, wide-format, with the session metadata
    replicated on every row. Goal: accumulate across many sessions into
    a single dataframe with ``pd.concat`` for shape (W, f, A) analysis.
    """
    meta = loaded.meta or {}
    base_metadata = {
        "tool_id":             meta.get("phone_id", ""),
        "phone_type":          meta.get("phone_type", ""),
        "input_mode":          meta.get("input_mode", ""),
        "session_start_iso":   meta.get("session_start_iso", ""),
        "session_end_iso":     meta.get("session_end_iso", ""),
        "session_start_date":  meta.get("session_start_date", ""),
        "session_end_date":    meta.get("session_end_date", ""),
        "session_start_time":  meta.get("session_start_time", ""),
        "session_end_time":    meta.get("session_end_time", ""),
        "data_source":         str(loaded.source),
        "filename":            meta.get("filename", ""),
        "time_format":         meta.get("time_format", ""),
        "samples":             meta.get("samples", ""),
        "sample_rate":         meta.get("sample_rate", ""),
        "resampled_hz":        meta.get("resampled_hz", ""),
        "valid_intervals":     meta.get("valid_intervals_count", ""),
        "report_generated_utc": utcnow_iso(),
    }

    trap_rows = rows_by_algo.get("trap", [])
    zupt_rows = rows_by_algo.get("zupt", [])

    out_rows: list[dict] = []
    for pos in range(len(segments)):
        row = segments.iloc[pos]
        try:
            t_lo = float(row["start_s"]); t_hi = float(row["end_s"])
        except (TypeError, ValueError):
            continue
        seg_type = str(row.get("type", "")).lower()
        matching = find_matching_prediction(predictions, t_lo, t_hi)
        l1 = (matching or {}).get("lobe1") or {}
        l2 = (matching or {}).get("lobe2") or {}

        ov = overrides.get(pos) if overrides else None
        trap_r = _row_for_segment(trap_rows, pos) or {}
        zupt_r = _row_for_segment(zupt_rows, pos) or {}
        # Joint-fit shape comes from the predictor's actual params dict
        # so it reflects the override when one is active. Falls back to
        # the detector's lobe1 (the shared-shape detector matches lobe1
        # === lobe2 by construction).
        trap_params = (trap_r.get("meta") or {}).get("params") or {}
        joint_W = trap_params.get("W", l1.get("half_width_s"))
        joint_f = trap_params.get("f", l1.get("frac_flat"))
        joint_A_used = trap_params.get("A_used", abs(l1.get("a_peak", float("nan"))))

        row_dict = {
            **base_metadata,
            "segment_idx":          int(pos),
            "segment_type":         seg_type,
            "segment_start_s":      t_lo,
            "segment_end_s":        t_hi,
            "segment_duration_s":   t_hi - t_lo,
            # Detector's per-lobe fit (what the segmenter found).
            "lobe1_t_c":            l1.get("t_c", ""),
            "lobe1_W":              l1.get("half_width_s", ""),
            "lobe1_f":              l1.get("frac_flat", ""),
            "lobe1_A":              l1.get("a_peak", ""),
            "lobe1_r2":             l1.get("r2_local", ""),
            "lobe2_t_c":            l2.get("t_c", ""),
            "lobe2_W":              l2.get("half_width_s", ""),
            "lobe2_f":              l2.get("frac_flat", ""),
            "lobe2_A":              l2.get("a_peak", ""),
            "lobe2_r2":             l2.get("r2_local", ""),
            # Joint shape the predictor actually used (reflects override).
            "joint_W":              joint_W if joint_W is not None else "",
            "joint_f":              joint_f if joint_f is not None else "",
            "joint_abs_A":          joint_A_used if joint_A_used is not None else "",
            "joint_r2":             (matching or {}).get("joint_r2_mean", ""),
            # Override status (empty if no override was applied).
            "override_active":      bool(ov),
            "override_W":           (ov or {}).get("W", ""),
            "override_f":           (ov or {}).get("f", ""),
            "override_abs_A":       (ov or {}).get("abs_A", ""),
            # Predictor outputs per algorithm.
            "trap_delta_height_m":  trap_r.get("delta_height_m", ""),
            "trap_ci_m":            trap_r.get("ci_half_width", ""),
            "trap_quality":         trap_r.get("quality_score", ""),
            "trap_accepted":        trap_r.get("accepted", ""),
            "trap_reject_reason":   trap_r.get("reject_reason", ""),
            "zupt_delta_height_m":  zupt_r.get("delta_height_m", ""),
            "zupt_ci_m":            zupt_r.get("ci_half_width", ""),
            "zupt_quality":         zupt_r.get("quality_score", ""),
            "zupt_accepted":        zupt_r.get("accepted", ""),
            "zupt_reject_reason":   zupt_r.get("reject_reason", ""),
        }
        out_rows.append(row_dict)

    df_out = pd.DataFrame(out_rows)
    return df_out.to_csv(index=False).encode("utf-8")


def render() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    rows = st.session_state.get("prediction_rows") or []
    rows_by_algo = st.session_state.get("prediction_rows_by_algo") or {}
    segments = valid_segments(st.session_state.get("segments_df"))
    state = st.session_state.get("detector_state")
    predictions = st.session_state.get("predictions") or []
    if loaded is None or not rows or state is None:
        st.warning("Complete earlier steps first.")
        if st.button("← Back"):
            goto(STEP_PREDICT)
        return

    st.markdown(
        '<div class="hero">'
        '<span class="step-pill">Step 5</span>'
        '<h1>Export (PDF, Hebrew)</h1>'
        '<p>One page per segment: signal with trapezoid template, '
        '±30 s correlation window with color-coded peak status, and a '
        'segment summary.</p>'
        '</div>',
        unsafe_allow_html=True,
    )

    try:
        pdf_bytes = _build_pdf(
            loaded, state, predictions, segments, rows, rows_by_algo,
        )
    except Exception as e:
        st.error(f"PDF build failed: {type(e).__name__}: {e}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    overrides = st.session_state.get("lobe_overrides") or {}

    # Two side-by-side download buttons: the PDF (human-readable report)
    # and a signatures CSV (one row per segment, full session metadata
    # replicated on each row, suitable for cross-session W/f/A analysis
    # via ``pd.concat``).
    dl_pdf, dl_csv = st.columns(2)
    with dl_pdf:
        st.download_button(
            "Download Summary PDF",
            data=pdf_bytes,
            file_name=f"boutique_report_{stamp}.pdf",
            mime="application/pdf",
            use_container_width=True,
        )
    with dl_csv:
        try:
            csv_bytes = _build_signatures_csv(
                loaded, segments, predictions, rows_by_algo, overrides,
            )
            csv_name = _signatures_filename(loaded)
        except Exception as exc:  # noqa: BLE001
            st.error(f"CSV build failed: {type(exc).__name__}: {exc}")
        else:
            st.download_button(
                "Download Signatures CSV",
                data=csv_bytes,
                file_name=csv_name,
                mime="text/csv",
                use_container_width=True,
                help=(
                    "One row per segment with the session metadata "
                    "(tool_id, dates, etc.) replicated on every row and "
                    "the trapezoid shape parameters (W, f, |A|) the "
                    "predictor used. Suitable for collecting W/f/A "
                    "across many sessions for offline analysis."
                ),
            )

    with st.expander("Preview PDF inline", expanded=True):
        b64 = base64.b64encode(pdf_bytes).decode()
        st.markdown(
            f'<iframe src="data:application/pdf;base64,{b64}" '
            f'width="100%" height="720" style="border:1px solid #e6e9ef; '
            f'border-radius:10px;"></iframe>',
            unsafe_allow_html=True,
        )

    st.divider()
    c1, _, c3 = st.columns([1, 1, 1])
    with c1:
        if st.button("← Back"):
            goto(STEP_PREDICT)
    with c3:
        if st.button("Start over"):
            for k in ("loaded", "detector_state", "predictions",
                      "segments_df", "prediction_rows",
                      "prediction_rows_by_algo",
                      "selected_segment", "predict_selected"):
                st.session_state[k] = None
            goto(STEP_HOWTO)
