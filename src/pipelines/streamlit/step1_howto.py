"""Step 1 — How to use (intro / landing page)."""
from __future__ import annotations

from pathlib import Path

import streamlit as st

from .common import APP_TITLE, STEP_DATA, goto

_FIG_ROOT = (
    Path(__file__).resolve().parents[3]
    / "docs" / "latex" / "figures" / "boutique"
)


def _heb(html: str) -> None:
    st.markdown(
        '<div dir="rtl" style="text-align: right; line-height: 1.75; '
        f'font-size: 0.97rem;">{html}</div>',
        unsafe_allow_html=True,
    )


def _heb_caption(html: str) -> None:
    st.markdown(
        '<div dir="rtl" style="text-align: right; line-height: 1.5; '
        'font-size: 0.85rem; color: #4a5568; margin: 0.2rem 0 1rem 0;">'
        f'{html}</div>',
        unsafe_allow_html=True,
    )


def _img(rel: str) -> None:
    st.image(str(_FIG_ROOT / rel), use_container_width=True)


def render() -> None:
   

    # ------------------------------------------------------------------
    # Hebrew walkthrough — mirrors docs/latex/boutique_pipeline_guide_hebrew.tex
    # ------------------------------------------------------------------
    _heb('<h2 style="margin-top:0;">מדריך שימוש בעברית</h2>')
    _heb(
        '<p>מדריך מעשי, מבוסס תמונות, לאפליקציה זו. עוברים על חמשת השלבים '
        'של האשף עם צילומי מסך, ולאחר מכן מובא דף בדיקה ויזואלי של '
        '<b>טוב מול גרוע</b> לסגמנטציה (שלב 3) — השלב היחיד שמצריך שיקול '
        'דעת. אם רק רוצים להפיק דו"ח מהר — קוראים את חלק '
        '"התחלה מהירה" ולוחצים <b>Start ←</b>.</p>'
    )

    _heb(
        '<h3>הפעלה</h3>'
        '<p>משורש הפרויקט, עם הסביבה הוירטואלית פעילה:</p>'
    )
    st.code(
        "venv/bin/python -m streamlit run src/pipelines/boutique_pipeline.py",
        language="bash",
    )
    _heb(
        '<p>Streamlit ידפיס כתובת מקומית (ברירת מחדל '
        '<code>http://localhost:8501</code>). האשף מתקדם דרך הכפתור הראשי '
        'שבפינה הימנית-תחתונה של כל עמוד '
        '(<b>Next ←</b>, <b>Predict ←</b>, <b>Generate report ←</b>).</p>'
    )

    # ---------- Step 1 ----------
    _heb('<h3>שלב 1 — דף פתיחה</h3>')
    _heb('<p>קוראים את הרקע הקצר, ולוחצים <b>Start ←</b>.</p>')
    _img("ui_step1_landing.png")

    # ---------- Step 2 ----------
    _heb('<h3>שלב 2 — נתונים</h3>')
    _heb(
        '<p>שני מצבי קלט. ניתן לחזור ולהחליף ביניהם בכל עת מהקישור '
        '<i>change input source</i> בפינה הימנית-עליונה.</p>'
    )
    c1, c2 = st.columns(2)
    with c1:
        _img("ui_step2_picker.png")
        _heb_caption(
            '<b>2א.</b> בוחרים: <b>Use phone DB</b> (משתמשים פנימיים) '
            'או <b>Upload a file</b> (כל השאר).'
        )
    with c2:
        _img("ui_step2_upload_form.png")
        _heb_caption(
            '<b>2ב.</b> להעלאת קובץ: CSV/XLSX עם עמודת זמן (שניות או '
            'מילישניות, מזוהה אוטומטית) ועמודת תאוצה אנכית (m/s²). '
            'שורה ראשונה חייבת להיות כותרת.'
        )

    # ---------- Step 3 ----------
    _heb('<h3>שלב 3 — סגמנטציה</h3>')
    _heb(
        '<p>הגלאי רץ אוטומטית. כל נסיעה שזוהתה מסומנת ברצועה צבעונית '
        '(<span style="color:#1f6feb;"><b>כחול</b></span> = עליה, '
        '<span style="color:#b54a9b;"><b>ורוד</b></span> = ירידה); '
        'הסגמנט הנבחר ב<span style="color:#e67e22;"><b>כתום</b></span>. '
        'אם המקטעים נראים בסדר — לוחצים <b>Predict ←</b>. אחרת, מציצים '
        'בדף הבדיקה למטה ובדפוסי התיקון.</p>'
    )
    _img("panels/clean_step3_overview__signal.png")
    _heb_caption(
        'דוגמה לאות נקי: 12 נסיעות זוהו (הקלטת S23). כל רצועה יושבת על '
        'צורת שתי-אונות ברורה.'
    )

    # ---------- Step 4 ----------
    _heb('<h3>שלב 4 — חיזוי</h3>')
    _heb(
        '<p>שני האלגוריתמים — <span style="color:#1f6feb;"><b>Trapezoid '
        'pulse-pair</b></span> ו-<span style="color:#27ae60;"><b>ZUPT</b>'
        '</span> — רצים אוטומטית; עמודה אחת לכל אלגוריתם לכל מקטע. '
        'עמודה חיובית ⇐ נסיעה כלפי מעלה. שני העמודים אמורים להיות דומים '
        'בגובה. בסיום — לוחצים <b>Generate report ←</b>.</p>'
    )
    _img("crops/clean_step4_metrics.png")
    _heb_caption(
        'תרשים העמודות: Trapezoid (כחול) ו-ZUPT (ירוק) זה לצד זה. '
        'הסכמה ביניהם = ביטחון גבוה.'
    )

    # ---------- Step 5 ----------
    _heb('<h3>שלב 5 — דו"ח</h3>')
    _heb(
        '<p>תצוגה מקדימה של ה-PDF מוטמעת בעמוד. לחיצה על '
        '<b>Download Summary PDF</b> שומרת את הקובץ העברי הסופי. שם '
        'הקובץ: <code>boutique_report_YYYYMMDDTHHMMSSZ.pdf</code>.</p>'
    )
    _img("crops/clean_step5_top.png")

    _heb(
        '<h3>משפט אחד לכל שלב</h3>'
        '<ol>'
        '<li>קוראים את הרקע הקצר, לוחצים <b>Start</b>.</li>'
        '<li>בוחרים מקור, לוחצים <b>Next</b>.</li>'
        '<li>מציצים על המקטעים; אם תקינים — <b>Predict</b>.</li>'
        '<li>מציצים על תרשים העמודות; אם שני האלגוריתמים מסכימים — '
        '<b>Generate report</b>.</li>'
        '<li><b>Download</b>. סיימנו.</li>'
        '</ol>'
        '<p><i>השלב היחיד שמצריך שיקול דעת הוא שלב 3.</i> דף הבדיקה '
        'הויזואלי שלמטה הוא הכלי לכך.</p>'
    )

    # ---------------- Cheat sheet: Good vs Bad ----------------
    with st.expander("דף בדיקה לשלב 3 — האם הסגמנט תקין? (טוב מול גרוע)"):

        # Panel A — Signal overview good vs bad
        _heb(
            '<h4>חלונית א — אות + סגמנטים: האם הנסיעות מכוסות?</h4>'
            '<p>מבט-על על כל ההקלטה. כל נסיעה שזוהתה היא רצועה צבועה.</p>'
        )
        c1, c2 = st.columns(2)
        with c1:
            _img("panels/clean_step3_overview__signal.png")
            _heb_caption(
                '<span style="color:#27ae60;"><b>טוב.</b></span> 12 נסיעות '
                'נקיות. כל רצועה יושבת על צורת שתי-אונות ברורה; בין '
                'הרצועות — אות שקט. אין מה לתקן.'
            )
        with c2:
            _img("panels/fp_step3_overview__signal.png")
            _heb_caption(
                '<span style="color:#c0392b;"><b>גרוע — חלוקה עודפת.</b>'
                '</span> מקטעים צרים רבים; מקטע 0 הוא קוץ בודד גבוה '
                '(תקיעה של הטלפון), לא נסיעה. רובם צריכים להימחק.'
            )

        _img("panels/damped_step3_overview__signal.png")
        _heb_caption(
            '<span style="color:#c0392b;"><b>גרוע — הקלטה מוחלשת '
            '(טלפון בכיס).</b></span> הגלאי מצא 10 מקטעים אבל בהקלטה '
            'יש בפועל הרבה יותר נסיעות — כל קוץ שחור גדול שלא עוטף '
            'ברצועה הוא נסיעה שפוספסה כי המשרעת היתה מתחת לסף. מוסיפים '
            'סגמנטים ידנית.'
        )

        # Panel B — Trapezoid fits
        _heb(
            '<h4>חלונית ב — הטרפז המותאם: האם התבנית מתאימה לאות?</h4>'
            '<p>החלונית מתמקדת במקטע הנבחר ומציגה את שתי תבניות הטרפז '
            'שהותאמו (אונה 1 באדום, אונה 2 בסגול) על האות הגולמי. <i>זוהי '
            'החלונית האינפורמטיבית ביותר להחלטה האם המקטע אמין.</i></p>'
        )
        _img("panels/clean_step3_overview__trap.png")
        _heb_caption(
            '<span style="color:#27ae60;"><b>טוב.</b></span> התבניות '
            'יושבות על המעטפת המוחלקת; רמות plateau של ±0.5 m/s² '
            'אופייניות למעלית בעיר; אונה 1 חיובית, אונה 2 שלילית '
            '⇐ נסיעת ירידה; שתי האונות בעלות משרעת דומה.'
        )

        c1, c2 = st.columns(2)
        with c1:
            _img("panels/fp_step3_seg0_FP__trap.png")
            _heb_caption(
                '<span style="color:#c0392b;"><b>גרוע — משרעת גבוהה '
                'מאוד (FP).</b></span> ה-plateau הוא ~5 m/s², וסביב '
                'המקטע יש קוצים של ±50. זו תקיעה/נפילה של הטלפון, '
                'לא נסיעה. מוחקים.'
            )
        with c2:
            _img("panels/damped_step3_seg0_full__trap.png")
            _heb_caption(
                '<span style="color:#c0392b;"><b>גרוע — משרעת זעירה '
                '(מוחלש).</b></span> plateau ב-±0.4 m/s²; האונות '
                'בקושי מעל לרעש. רווח הסמך יהיה רחב.'
            )

        _img("panels/haari_step3_seg20_full__trap.png")
        _heb_caption(
            '<span style="color:#c0392b;"><b>גרוע — מקטע ארוך מדי '
            '(שני אירועים אוחדו).</b></span> הרצועה הכתומה משתרעת '
            'על 22 שניות, עם 17 שניות שקטות באמצע. יש לחתוך לאירוע '
            'הראשון בלבד ולהוסיף סגמנט שני.'
        )

        # Panel C — Correlation
        _heb(
            '<h4>חלונית ג — מתאם עם סטטוס שיא</h4>'
            '<p>שני קווים: R² הטוב ביותר על תבניות חיוביות (כחול) '
            'ועל תבניות שליליות (אדום). <i>שתי נקודות ירוקות</i> '
            'בסביבת המקטע הנבחר = זוג מאומת אחד.</p>'
        )
        c1, c2 = st.columns(2)
        with c1:
            _img("panels/clean_step3_overview__corr.png")
            _heb_caption(
                '<span style="color:#27ae60;"><b>טוב.</b></span> '
                'נקודה ירוקה על פסגת הקו הכחול ונקודה ירוקה על פסגת '
                'הקו האדום. הנקודות האפורות מסביב הן שיאים מתחת לסף.'
            )
        with c2:
            _img("panels/damped_step3_seg0_full__corr.png")
            _heb_caption(
                '<span style="color:#e67e22;"><b>גבולי.</b></span> '
                'שתי נקודות ירוקות קיימות אבל מוקפות בכתומות '
                '(unpaired) ובאפורות (|A|<thr) — סימן קלאסי של '
                'הקלטה מוחלשת.'
            )

        # Panel D — Heatmaps
        _heb(
            '<h4>חלונית ד — מפות חום: האם צורת הטרפז ממוקמת היטב?</h4>'
            '<p>R² של ההתאמה על פני <i>גריד התבניות</i>: חצי-רוחב '
            'W בציר ה-y ושבר plateau f בציר ה-x. ה-<b>X</b> האדום '
            'מסמן את הטוב ביותר. נסיעה נקיה = אגן בהיר וקומפקטי, '
            'וה-X לא צמוד לקצה.</p>'
        )
        c1, c2 = st.columns(2)
        with c1:
            _img("panels/clean_step3_overview__heatmaps.png")
            _heb_caption(
                '<span style="color:#27ae60;"><b>טוב.</b></span> '
                'שתי האונות מגיעות לשיא ב-(W*, f*) ≈ (1.3 s, 0.4) '
                '— בתוך הגריד. האגן הבהיר קומפקטי.'
            )
        with c2:
            _img("panels/damped_step3_seg0_full__heatmaps.png")
            _heb_caption(
                '<span style="color:#c0392b;"><b>גרוע — ארגמקס בקצה.'
                '</b></span> ה-X צמוד ל-f ≈ 0.83. הנסיעה האמיתית '
                '<i>מחוץ</i> לגריד התבניות.</span>'
            )

        # Fixing patterns
        _heb(
            '<h4>תיקונים בשלב 3 — ארבעת הדפוסים</h4>'
            '<ul>'
            '<li><b>דפוס א — נסיעה שפוספסה.</b> צורת שתי-אונות '
            'ברורה ללא רצועה צבועה מעליה. תיקון: <b>+ Add segment</b>, '
            'ואז <b>Edit selected</b> לקביעת זמני התחלה/סיום.</li>'
            '<li><b>דפוס ב — חיוב שגוי (FP).</b> רצועה עם שכבת טרפז '
            'בקנה מידה מטורף, או ארגמקס צמוד לקצה, או שהאות בתוך '
            'הרצועה הוא קוץ בודד. תיקון: בוחרים את השורה ולוחצים '
            'על אייקון הפח.</li>'
            '<li><b>דפוס ג — שתי נסיעות אוחדו.</b> רצועה אחת ארוכה '
            'משמעותית, עם קטע שקט באמצע. תיקון: <b>Edit selected</b>, '
            'חיתוך זמן הסיום, ואז <b>+ Add segment</b> לאירוע השני.</li>'
            '<li><b>דפוס ד — גבולות שגויים ב-1–2 שניות.</b> תיקון: '
            '<b>Edit selected</b> והזזת הגבול. שכבת הטרפז מתעדכנת '
            'מיידית.</li>'
            '</ul>'
            '<p><b>בספק — אל תיגעו.</b> הגלאי מכויל להיות שמרני; '
            'הטעות הנפוצה ביותר היא עריכת-יתר על מקטעים גבוליים.</p>'
        )

        # Step 4 cheat sheet
        _heb(
            '<h4>דף בדיקה לשלב 4 — האם שני האלגוריתמים מסכימים?</h4>'
        )
        c1, c2 = st.columns(2)
        with c1:
            _img("crops/clean_step4_metrics.png")
            _heb_caption(
                '<span style="color:#27ae60;"><b>טוב.</b></span> '
                'העמודות לכל מקטע כמעט זהות בגובה לשני האלגוריתמים.'
            )
        with c2:
            _img("haari_step4_overview.png")
            _heb_caption(
                '<span style="color:#c0392b;"><b>לחקור.</b></span> '
                'פערים גדולים בין הכחול לירוק. כל פער > 1.5 m הוא '
                'רמז לרעש או לגבולות שגויים — חוזרים לשלב 3.'
            )

        _heb(
            '<h4>רשימת בדיקה לפני שליחת דו"ח</h4>'
            '<ol>'
            '<li>כל רצועה בגרף העליון מכסה צורת שתי-אונות נראית-לעין.</li>'
            '<li>גרף המתאם של כל מקטע נבחר מציג שתי נקודות ירוקות.</li>'
            '<li>ארגמקס מפת החום של כל מקטע נבחר נמצא בתוך הגריד.</li>'
            '<li>שכבת הטרפז של כל מקטע נבחר יושבת על המעטפת המוחלקת.</li>'
            '<li>שני האלגוריתמים מסכימים על Δh עד כדי ~1 m, '
            'וה-Trapezoid מסמן <b>accepted</b>.</li>'
            '<li>רווח סמך מתחת ל-~1.5 m בנסיעות עד ~10 s ומתחת '
            'ל-~3 m בארוכות ביותר.</li>'
            '</ol>'
        )

    st.divider()
    _, c = st.columns([3, 1])
    with c:
        if st.button("Start →", type="primary"):
            goto(STEP_DATA)
