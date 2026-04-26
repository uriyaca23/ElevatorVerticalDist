"""Playwright-driven screenshot capture for the Boutique-Pipeline guide.

Run AFTER:
  - tmp_boutique_capture/survey.py has been run (4-col CSVs in csvs/)
  - the seed-runner streamlit is listening on http://localhost:8511

Output: docs/latex/figures/boutique/*.png
"""
from __future__ import annotations

from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
CSV_DIR = REPO / "tmp_boutique_capture" / "csvs"
SHOTS_DIR = REPO / "docs" / "latex" / "figures" / "boutique"
SHOTS_DIR.mkdir(parents=True, exist_ok=True)

BASE_URL = "http://localhost:8511"
VIEWPORT = {"width": 1600, "height": 1000}


def _wait_idle(page, ms: int = 800) -> None:
    page.wait_for_load_state("networkidle", timeout=15_000)
    page.wait_for_timeout(ms)


def _click_text_button(page, text: str) -> None:
    page.get_by_role("button", name=text, exact=False).first.click()
    _wait_idle(page)


def _expand_viewport_to_main(page, max_h: int = 5500) -> int:
    """Resize the browser viewport so the main scroll container fits in
    one screenshot. We first reset to the baseline so the measurement
    isn't contaminated by a previously-expanded viewport.
    """
    page.set_viewport_size({"width": 1600, "height": 1000})
    page.wait_for_timeout(400)
    needed = page.evaluate("""
() => {
  const m = document.querySelector('section[data-testid="stMain"]');
  return m ? m.scrollHeight : window.innerHeight;
}
""")
    target = min(int(needed) + 60, max_h)
    if target < 1000:
        target = 1000
    page.set_viewport_size({"width": 1600, "height": target})
    page.wait_for_timeout(800)
    return target


def _save(page, name: str, full: bool = True) -> Path:
    out = SHOTS_DIR / f"{name}.png"
    if full:
        h = _expand_viewport_to_main(page)
    else:
        h = None
    page.screenshot(path=str(out), full_page=full)
    suffix = f"vp_h={h}" if h else "viewport"
    print(f"  saved {out.name} ({suffix})")
    return out


def _save_clip(page, name: str, x: int, y: int, w: int, h: int) -> Path:
    """Screenshot a viewport rectangle. Useful for cropping a specific
    panel after a full screenshot is too big.
    """
    out = SHOTS_DIR / f"{name}.png"
    page.screenshot(path=str(out), clip={"x": x, "y": y, "width": w, "height": h})
    print(f"  saved {out.name} (clip)")
    return out


def goto_seed(page, scenario: str) -> None:
    """Navigate the seed-runner to a given scenario, ending up on Step 3
    with the detector already finished."""
    page.goto(f"{BASE_URL}/?csv={scenario}", wait_until="networkidle")
    _wait_idle(page, 1500)
    page.wait_for_selector(
        "text=Signal + segments", state="visible", timeout=60_000,
    )
    _wait_idle(page, 2500)


def goto_clean(page) -> None:
    """Land on Step 1 fresh — used for landing-page screenshots and the
    Step 2 chooser/upload-form mockups (the seed runner skips these for
    scenario captures)."""
    page.goto(BASE_URL, wait_until="networkidle")
    _wait_idle(page, 1200)


def select_segment(page, idx: int) -> None:
    """Click the i-th segment in the sidebar's radio list."""
    sidebar = page.locator("section[data-testid='stSidebar']")
    radios = sidebar.locator("div[role='radiogroup'] label")
    count = radios.count()
    if idx >= count:
        raise IndexError(f"sidebar has {count} radio options; idx={idx}")
    target = radios.nth(idx)
    target.scroll_into_view_if_needed()
    target.click(force=True)
    _wait_idle(page, 1500)


def go_predict(page) -> None:
    _click_text_button(page, "Predict")
    page.wait_for_selector(
        "text=Per-segment height predictions", state="visible", timeout=60_000,
    )
    # Wait until the spinner ('Running N algorithms...') is gone.
    try:
        page.wait_for_selector(
            "text=Running", state="hidden", timeout=60_000,
        )
    except Exception:
        pass
    # Wait for either the bar chart or the segments table to appear.
    page.wait_for_selector(
        "text=All segments", state="visible", timeout=60_000,
    )
    _wait_idle(page, 3000)


def go_report(page) -> None:
    _click_text_button(page, "Generate report")
    page.wait_for_selector(
        "text=Export (PDF, Hebrew)", state="visible", timeout=60_000,
    )
    _wait_idle(page, 4000)


# ---------------------------------------------------------------------------

def capture_landing_pages(page) -> None:
    """Step 1, Step 2 chooser, Step 2 upload form (no data loaded)."""
    # Need a clean session — the seed runner doesn't seed if there's no
    # ?csv=... param.
    goto_clean(page)
    _save(page, "ui_step1_landing")

    _click_text_button(page, "Start")
    _save(page, "ui_step2_picker")

    _click_text_button(page, "Upload a file")
    _save(page, "ui_step2_upload_form")


def capture_clean_pipeline(page) -> None:
    """Use clean_S23_milleniumOutside (12/12, F1=1.00, short trace)."""
    goto_seed(page, "clean_S23_milleniumOutside")
    _save(page, "clean_step3_overview")

    select_segment(page, 0)
    _save(page, "clean_step3_seg0_full")

    select_segment(page, 5)
    _save(page, "clean_step3_seg5_full")

    go_predict(page)
    _save(page, "clean_step4_overview")

    select_segment(page, 5)
    _save(page, "clean_step4_seg5")

    go_report(page)
    _save(page, "clean_step5_report")


def capture_fp_scenario(page) -> None:
    """clean_milleniumA23: pred 0 (up, 457.3-465.1) and 5, 33 are FPs."""
    goto_seed(page, "clean_milleniumA23")
    _save(page, "fp_step3_overview")
    select_segment(page, 0)
    _save(page, "fp_step3_seg0_FP")
    select_segment(page, 5)
    _save(page, "fp_step3_seg5_FP")


def capture_damped_scenario(page) -> None:
    """damped_BarIlanPix10: 44 GT but 10 detected, many missed rides."""
    goto_seed(page, "damped_BarIlanPix10")
    _save(page, "damped_step3_overview")
    select_segment(page, 0)
    _save(page, "damped_step3_seg0_full")


def capture_haari_scenario(page) -> None:
    """merged_split_Haari3: many borderline segments — mid-list ones tend
    to expose merge/split / endpoint issues."""
    goto_seed(page, "merged_split_Haari3")
    _save(page, "haari_step3_overview")
    select_segment(page, 0)
    _save(page, "haari_step3_seg0_full")
    select_segment(page, 10)
    _save(page, "haari_step3_seg10_full")
    select_segment(page, 20)
    _save(page, "haari_step3_seg20_full")
    go_predict(page)
    _save(page, "haari_step4_overview")


def main() -> None:
    from playwright.sync_api import sync_playwright

    targets = [
        CSV_DIR / "clean_S23_milleniumOutside.csv",
        CSV_DIR / "clean_milleniumA23.csv",
        CSV_DIR / "damped_BarIlanPix10.csv",
        CSV_DIR / "merged_split_Haari3.csv",
    ]
    for p in targets:
        assert p.exists(), f"missing: {p}"

    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        ctx = browser.new_context(viewport=VIEWPORT, device_scale_factor=2)
        page = ctx.new_page()
        page.set_default_timeout(60_000)

        capture_landing_pages(page)
        capture_clean_pipeline(page)
        capture_fp_scenario(page)
        capture_damped_scenario(page)
        capture_haari_scenario(page)

        browser.close()
    print("\nAll screenshots saved.")


if __name__ == "__main__":
    main()
