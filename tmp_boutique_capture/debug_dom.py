"""Probe the seed-runner DOM to find a working radio-button selector."""
from __future__ import annotations
from playwright.sync_api import sync_playwright

URL = "http://localhost:8511/?csv=clean_S23_milleniumOutside"

with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page()
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Signal + segments", timeout=60_000)
    page.wait_for_timeout(2500)

    # Find the sidebar via several known testid candidates.
    for sb_sel in [
        "section[data-testid='stSidebar']",
        "aside[data-testid='stSidebar']",
        "div[data-testid='stSidebar']",
        "section[role='complementary']",
        "[data-testid*='Sidebar']",
    ]:
        cnt = page.locator(sb_sel).count()
        print(f"sidebar candidate {sb_sel!r:50s} -> {cnt}")
    sb = page.locator(
        "section[data-testid='stSidebar'], aside[data-testid='stSidebar'], "
        "div[data-testid='stSidebar'], section[role='complementary']"
    ).first
    for sel in [
        "div[role='radiogroup'] label",
        "div[role='radiogroup'] [role='radio']",
        "[role='radiogroup'] [role='radio']",
        "[role='radio']",
        "input[type='radio']",
        "div[data-baseweb='radio']",
        "label[data-baseweb='radio']",
        "div[role='radiogroup'] > label",
        "div[role='radiogroup']",
    ]:
        n = sb.locator(sel).count()
        print(f"  {sel!r:50s} -> {n}")

    # Dump first ~40 lines of sidebar HTML to inspect structure
    html = sb.first.inner_html()
    snippet = html[:5000]
    print("\n--- sidebar.inner_html (first 5000 chars) ---")
    print(snippet)
    b.close()
