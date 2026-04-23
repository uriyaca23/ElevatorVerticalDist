"""Step 5 — Hebrew per-segment PDF report.

Builds a multi-page PDF (cover + per-segment pages + how-to-read) using
ReportLab, with matplotlib-rendered PNG figures for each segment. The
Hebrew text is bidi-reordered through :func:`_rtl` while leaving inline
``<b>``/``<font>`` tags untouched, and any user-supplied dynamic text is
escaped through :func:`_esc` so it can't break the paraparser.
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
    trapezoid_kernel,
    utcnow_iso,
    valid_segments,
)


# Small translation table for static strings. Kept here so the PDF writer
# and the on-screen preview read from the same place.
HEB = {
    "report_title":   "דוח פייפליין מעלית",
    "source":         "מקור",
    "generated":      "נוצר בתאריך",
    "samples":        "מספר דגימות",
    "sample_rate":    "קצב דגימה",
    "phone_type":     "סוג מכשיר",
    "window":         "חלון זמן",
    "experiment":     "ניסוי",
    "summary":        "סיכום",
    "n_segments":     "מספר מקטעים",
    "n_accepted":     "התקבלו בבקרה",
    "net_dh":         "סך הפרשי גובה (מטר)",
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


def _maybe_register_hebrew_font() -> str:
    """Register a Hebrew-capable TTF with reportlab. Returns the font name
    to use; falls back to ``Helvetica`` if no suitable font is found.
    """
    try:
        from reportlab.pdfbase import pdfmetrics
        from reportlab.pdfbase.ttfonts import TTFont
    except ImportError:
        return "Helvetica"

    candidates = []
    try:
        import matplotlib
        mpl_fonts = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
        candidates.extend([
            mpl_fonts / "DejaVuSans.ttf",
            mpl_fonts / "DejaVuSans-Bold.ttf",
        ])
    except Exception:
        pass
    # Common macOS / Linux locations.
    candidates.extend([
        Path("/Library/Fonts/Arial Unicode.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"),
    ])
    for cand in candidates:
        if cand.exists():
            try:
                pdfmetrics.registerFont(TTFont("HebrewSans", str(cand)))
                return "HebrewSans"
            except Exception:
                continue
    return "Helvetica"


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


def _build_main_signal_png(state: dict, segments: pd.DataFrame,
                           selected: int | None = None) -> bytes:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception:
        return b""
    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    fig, ax = plt.subplots(figsize=(8.0, 3.2), dpi=160)
    ax.plot(t, a_vert, color="#233044", lw=0.7)
    ax.plot(t, a_smooth, color="#e67e22", lw=1.1)
    for i, row in segments.iterrows():
        try:
            s = float(row["start_s"]); e = float(row["end_s"])
        except (TypeError, ValueError):
            continue
        color = RIDE_COLORS.get(str(row.get("type", "up")), "#777")
        is_sel = (selected is not None and int(i) == selected)
        ax.axvspan(s, e,
                   color=SELECTED_COLOR if is_sel else color,
                   alpha=0.32 if is_sel else 0.18, lw=0)
    ax.set_xlabel("time (s)"); ax.set_ylabel("a_vert (m/s²)")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=160)
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
    except Exception:
        return b"", b""

    t = np.asarray(state["t"])
    a_vert = np.asarray(state["a_vert"])
    a_smooth = np.asarray(state["a_smooth"])
    match = find_matching_prediction(predictions, t_lo, t_hi)

    # --- (1) trapezoid overlay ---
    pad_s = 4.0
    mask = (t >= t_lo - pad_s) & (t <= t_hi + pad_s)
    fig1, ax1 = plt.subplots(figsize=(7.6, 2.7), dpi=160)
    ax1.plot(t[mask], a_vert[mask], color="#233044", lw=0.7, label="a_vert")
    ax1.plot(t[mask], a_smooth[mask], color="#e67e22", lw=1.0, label="smoothed")
    ax1.axhline(0, color="#aaa", lw=0.4, ls="--")
    ax1.axvspan(t_lo, t_hi, color=SELECTED_COLOR, alpha=0.08, lw=0)
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
            ax1.plot(tt, yy, color=color, lw=1.8)
            ax1.scatter([t_c], [A], color=color, s=26, zorder=5,
                        edgecolor="#000", linewidth=0.4)
    ax1.set_xlabel("t (s)"); ax1.set_ylabel("a (m/s²)")
    ax1.grid(True, alpha=0.25)
    ax1.legend(loc="upper right", fontsize=8, frameon=False)
    fig1.tight_layout()
    buf1 = io.BytesIO()
    fig1.savefig(buf1, format="png", dpi=160)
    plt.close(fig1)

    # --- (2) correlation panel ±30 s with peak colors ---
    t_min = float(t[0]); t_max = float(t[-1])
    wlo = max(t_min, t_lo - 30.0); whi = min(t_max, t_hi + 30.0)
    mask2 = (t >= wlo) & (t <= whi)
    pos_r2 = np.asarray(state["best_pos_r2"])
    neg_r2 = np.asarray(state["best_neg_r2"])
    pos_plot = np.where(np.isfinite(pos_r2), pos_r2, np.nan)
    neg_plot = np.where(np.isfinite(neg_r2), neg_r2, np.nan)

    fig2, ax2 = plt.subplots(figsize=(7.6, 2.7), dpi=160)
    ax2.plot(t[mask2], pos_plot[mask2], color="#2980b9", lw=0.9, label="max R² (+)")
    ax2.plot(t[mask2], neg_plot[mask2], color="#c0392b", lw=0.9, label="max R² (−)")
    cfg = state.get("config")
    if cfg is not None:
        ax2.axhline(cfg.r2_peak_thresh, color="#888", lw=0.5, ls="--")
    ax2.axvspan(t_lo, t_hi, color=SELECTED_COLOR, alpha=0.08, lw=0)
    peaks_pos = find_local_maxima(pos_r2, t, wlo, whi)
    peaks_neg = find_local_maxima(neg_r2, t, wlo, whi)
    for sign, peaks, arr in ((+1, peaks_pos, pos_r2), (-1, peaks_neg, neg_r2)):
        for i in peaks:
            tag = classify_peak(state, i, sign, predictions)
            ax2.scatter([t[i]], [arr[i]],
                        color=PEAK_STATUS_COLORS.get(tag, "#000"),
                        s=30, zorder=5, edgecolor="#000", linewidth=0.3)
    ax2.set_ylim(0, 1.05); ax2.set_xlim(wlo, whi)
    ax2.set_xlabel("t (s)"); ax2.set_ylabel("R² (per sign)")
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc="upper right", fontsize=7, frameon=False, ncol=2)
    fig2.tight_layout()
    buf2 = io.BytesIO()
    fig2.savefig(buf2, format="png", dpi=160)
    plt.close(fig2)

    return buf1.getvalue(), buf2.getvalue()


def _build_pdf(
    loaded: LoadedSignal,
    state: dict,
    predictions: list[dict],
    segments: pd.DataFrame,
    rows: list[dict],
) -> bytes:
    from reportlab.lib import colors as rl_colors
    from reportlab.lib.enums import TA_RIGHT
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import cm
    from reportlab.platypus import (
        Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle,
    )

    heb_font = _maybe_register_hebrew_font()

    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=1.5 * cm, rightMargin=1.5 * cm,
        topMargin=1.4 * cm, bottomMargin=1.4 * cm,
        title="Boutique Pipeline — Report",
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        name="HebTitle", parent=styles["Title"],
        fontName=heb_font, fontSize=22, alignment=TA_RIGHT,
        textColor=rl_colors.HexColor("#0f2a4a"),
        spaceAfter=6,
    )
    h2_style = ParagraphStyle(
        name="HebH2", parent=styles["Heading2"],
        fontName=heb_font, fontSize=14, alignment=TA_RIGHT,
        textColor=rl_colors.HexColor("#1f6feb"),
    )
    body_style = ParagraphStyle(
        name="HebBody", parent=styles["BodyText"],
        fontName=heb_font, fontSize=10, alignment=TA_RIGHT, leading=14,
    )

    def P(text: str, style=body_style) -> Paragraph:
        return Paragraph(_rtl(text), style)

    story: list = []

    # --- Cover page ---
    story.append(P(HEB["report_title"], title_style))
    story.append(Spacer(1, 0.2 * cm))
    meta_rows = [
        (HEB["source"],      loaded.source),
        (HEB["generated"],   utcnow_iso()),
        (HEB["samples"],     str(loaded.meta.get("samples", "?"))),
        (HEB["sample_rate"], str(loaded.meta.get("sample_rate", "?"))),
    ]
    for k, v in loaded.meta.items():
        if k in ("samples", "sample_rate"):
            continue
        label = HEB.get(k, k)
        meta_rows.append((label, str(v)))
    for label, val in meta_rows:
        story.append(P(f"<b>{_esc(label)}:</b> {_esc(val)}"))

    # Summary metrics + overall signal.
    df = pd.DataFrame(rows) if rows else pd.DataFrame()
    total_dh = float(df["delta_height_m"].dropna().sum()) if not df.empty else 0.0
    n_accepted = int(df["accepted"].sum()) if not df.empty else 0

    story.append(Spacer(1, 0.4 * cm))
    story.append(P(HEB["summary"], h2_style))
    story.append(P(f"{HEB['n_segments']}: <b>{len(df)}</b>"))
    story.append(P(f"{HEB['n_accepted']}: <b>{n_accepted}</b>"))
    story.append(P(f"{HEB['net_dh']}: <b>{total_dh:+.2f}</b>"))

    main_png = _build_main_signal_png(state, segments, selected=None)
    if main_png:
        story.append(Spacer(1, 0.3 * cm))
        story.append(Image(io.BytesIO(main_png), width=17 * cm, height=6.6 * cm))

    # Overview table (kept on cover — per-segment pages follow).
    story.append(Spacer(1, 0.4 * cm))
    story.append(P(HEB["per_seg_table"], h2_style))
    header = [HEB["col_accepted"], HEB["col_quality"], HEB["col_ci"],
              HEB["col_dh"], HEB["col_dur"], HEB["col_end"],
              HEB["col_start"], HEB["col_type"], HEB["col_idx"]]
    data = [[_rtl(h) for h in header]]
    for r in rows:
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
    tbl = Table(data, repeatRows=1, hAlign="RIGHT")
    tbl.setStyle(TableStyle([
        ("BACKGROUND", (0, 0), (-1, 0), rl_colors.HexColor("#0f2a4a")),
        ("TEXTCOLOR",  (0, 0), (-1, 0), rl_colors.white),
        ("FONTNAME",   (0, 0), (-1, -1), heb_font),
        ("FONTSIZE",   (0, 0), (-1, -1), 9),
        ("ALIGN",      (0, 0), (-1, -1), "CENTER"),
        ("ROWBACKGROUNDS", (0, 1), (-1, -1),
         [rl_colors.HexColor("#f5f7fb"), rl_colors.white]),
        ("GRID",       (0, 0), (-1, -1), 0.25, rl_colors.HexColor("#cfd4dc")),
    ]))
    story.append(tbl)

    # --- Per-segment pages ---
    for r in rows:
        story.append(PageBreak())
        rt_heb = HEB["up"] if r["type"] == "up" else HEB["down"]
        header_txt = (
            f"{HEB['segment_page']} #{r['segment']} — {rt_heb}  "
            f"({r['start_s']:.1f}–{r['end_s']:.1f} ש')"
        )
        story.append(P(header_txt, title_style))
        story.append(Spacer(1, 0.2 * cm))

        # Key metrics as a right-aligned list.
        dh_str = (f"{r['delta_height_m']:+.2f}"
                  if np.isfinite(r['delta_height_m']) else "—")
        ci_str = (f"{r['ci_half_width']:.2f}"
                  if np.isfinite(r['ci_half_width']) else "—")
        q_str = (f"{r['quality_score']:.1f}"
                 if np.isfinite(r['quality_score']) else "—")
        yn = HEB["yes"] if r["accepted"] else HEB["no"]
        story.append(P(f"<b>{HEB['col_dh']}:</b> {dh_str}"))
        story.append(P(f"<b>{HEB['col_ci']}:</b> {ci_str}"))
        story.append(P(f"<b>{HEB['col_quality']}:</b> {q_str}"))
        story.append(P(f"<b>{HEB['col_accepted']}:</b> {yn}"))
        if r.get("reject_reason"):
            story.append(
                P(f"<b>{HEB['reject_reason']}:</b> {_esc(r['reject_reason'])}")
            )

        story.append(Spacer(1, 0.3 * cm))
        trap_png, corr_png = _build_segment_page_pngs(
            state, predictions, float(r["start_s"]), float(r["end_s"]),
        )
        if trap_png:
            story.append(P(HEB["trap_heading"], h2_style))
            story.append(Image(io.BytesIO(trap_png),
                               width=17 * cm, height=6.0 * cm))
        if corr_png:
            story.append(Spacer(1, 0.2 * cm))
            story.append(P(HEB["corr_heading"], h2_style))
            story.append(Image(io.BytesIO(corr_png),
                               width=17 * cm, height=6.0 * cm))
            story.append(Spacer(1, 0.1 * cm))
            # Status legend as a plain line (the colors show in the PNG).
            # Keys may contain `<` (e.g. "|A|<thr") — escape so the
            # paraparser doesn't read them as unclosed tags.
            chips = "  ".join(f"● {_esc(t)}" for t in PEAK_STATUS_COLORS)
            story.append(P(f"{_esc(HEB['status_legend'])}: {chips}"))

    # --- How to read (last page) ---
    story.append(PageBreak())
    story.append(P(HEB["how_to_read"], h2_style))
    story.append(P(HEB["how_to_read_body"]))

    doc.build(story)
    return buf.getvalue()


def render() -> None:
    loaded: LoadedSignal | None = st.session_state["loaded"]
    rows = st.session_state.get("prediction_rows") or []
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
        pdf_bytes = _build_pdf(loaded, state, predictions, segments, rows)
    except Exception as e:
        st.error(f"PDF build failed: {type(e).__name__}: {e}")
        return

    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    st.download_button(
        "Download Summary PDF",
        data=pdf_bytes,
        file_name=f"boutique_report_{stamp}.pdf",
        mime="application/pdf",
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
