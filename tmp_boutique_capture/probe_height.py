"""Check page scroll height & body overflow on the seed runner."""
from playwright.sync_api import sync_playwright

URL = "http://localhost:8511/?csv=clean_S23_milleniumOutside"
with sync_playwright() as p:
    b = p.chromium.launch(headless=True)
    page = b.new_page(viewport={"width": 1600, "height": 1000})
    page.goto(URL, wait_until="networkidle")
    page.wait_for_selector("text=Signal + segments", timeout=60_000)
    page.wait_for_timeout(2500)
    # Click segment 0 to make sure the detail section is rendered.
    sb = page.locator("section[data-testid='stSidebar']")
    sb.locator("div[role='radiogroup'] label").first.click(force=True)
    page.wait_for_timeout(2000)
    info = page.evaluate("""
() => {
  const cands = [
    'section.main',
    'div[data-testid="stAppViewContainer"]',
    'div[data-testid="stMain"]',
    'main',
    'section[data-testid="stMain"]',
  ];
  const out = {};
  for (const sel of cands) {
    const el = document.querySelector(sel);
    if (el) {
      const cs = getComputedStyle(el);
      out[sel] = {
        scrollH: el.scrollHeight,
        clientH: el.clientHeight,
        overflowY: cs.overflowY,
      };
    }
  }
  return out;
}
""")
    import json; print(json.dumps(info, indent=2))
    b.close()
