"""End-to-end UI check: starts the web app on a throwaway database and clicks
through every page and control with Playwright (Chromium).

    python scripts/e2e_ui.py                    # PyTorch backend, headless
    python scripts/e2e_ui.py --backend matlab   # the MATLAB engine instead
    python scripts/e2e_ui.py --headed           # watch it

Every automatic recall is sent to the audit queue (DRISHTI_AUDIT_FRACTION=1)
so the doctor portal has several cases to work through. Prints PASS / FAIL per
feature, saves a screenshot for each failure, and exits non-zero on any
failure. Your own app/drishti.db is never touched.
"""
from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import tempfile
import time
import traceback
import urllib.request
from pathlib import Path

from playwright.sync_api import expect, sync_playwright

REPO = Path(__file__).resolve().parent.parent
SAMPLES = REPO / "app" / "samples"
BANNED = ["Aravind", "Sankara", "AIIMS", "510(k)", "ResNet", "DenseNet", "Tri-Ensemble", "92.5%", "64%",
          "Zero Retention", "ABDM", "Quantized", "98.4%", "90.5%", "94.1", "Anita", "Rajesh", "800-DRISHTI"]

results: list[tuple[str, str, str]] = []
console_errors: list[str] = []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, db: Path, backend: str) -> subprocess.Popen:
    env = {**os.environ, "DRISHTI_DB": str(db), "DRISHTI_AUDIT_FRACTION": "1", "DRISHTI_BACKEND": backend}
    log = open(db.with_suffix(".log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                         cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(120):
        try:
            if urllib.request.urlopen(f"http://127.0.0.1:{port}/", timeout=2).status == 200:
                return p
        except Exception:
            time.sleep(0.5)
    p.kill()
    raise RuntimeError(f"server did not start; see {db.with_suffix('.log')}")


def run(name: str, fn, page, shots: Path):
    t0 = time.time()
    try:
        detail = fn() or ""
        results.append((name, "PASS", f"{detail} ({time.time() - t0:.1f} s)".strip()))
    except Exception as e:
        msg = f"{type(e).__name__}: {e}".splitlines()[0][:300]
        results.append((name, "FAIL", msg))
        try:
            page.screenshot(path=str(shots / (re.sub(r"\W+", "_", name) + ".png")), full_page=True)
        except Exception:
            pass
        if os.environ.get("E2E_TRACE"):
            traceback.print_exc()


def watch(page, label):
    page.on("pageerror", lambda e: console_errors.append(f"{label}: page error: {e}"))
    page.on("console", lambda m: m.type == "error" and "favicon" not in m.text and "Failed to load resource" not in m.text
            and console_errors.append(f"{label}: console: {m.text[:200]}"))


def main():
    sys.stdout.reconfigure(encoding="utf-8")  # results quote Hindi and Tamil text
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="python", choices=["python", "matlab", "auto"])
    ap.add_argument("--headed", action="store_true")
    a = ap.parse_args()
    work = Path(tempfile.mkdtemp(prefix="drishti_e2e_"))
    shots = work / "failures"
    shots.mkdir()
    port = free_port()
    base = f"http://127.0.0.1:{port}"
    server = start_server(port, work / "e2e.db", a.backend)
    slow = 300_000 if a.backend == "matlab" else 60_000  # MATLAB imports its networks on the first photo
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not a.headed)
            ctx = browser.new_context(viewport={"width": 1440, "height": 900})
            ctx.set_default_timeout(slow)
            page = ctx.new_page()
            watch(page, "main")
            state = {}

            def S(name, fn):
                run(name, fn, page, shots)

            # ------------------------------------------------------------ landing
            def landing_loads():
                page.goto(base + "/")
                txt = page.inner_text("body")
                bad = [b for b in BANNED if b in txt or b in page.content()]
                assert not bad, f"banned claims still on the page: {bad}"
                link = page.locator("a[href='https://doi.org/10.1371/journal.pone.0270562']")
                assert link.count() == 1, "SMART India citation link missing"
                return "no false claims found; citation present"
            S("Landing: loads, no removed claims, citation", landing_loads)

            def landing_roc():
                page.locator("#benchmarks").scroll_into_view_if_needed()
                page.wait_for_function("() => typeof chartInstance !== 'undefined' && chartInstance")
                page.click("#btn-chart-idrid")
                lab = page.evaluate("chartInstance.data.datasets[0].label")
                assert "IDRiD" in lab and "0.976" in lab, lab
                page.click("#btn-chart-messidor")
                lab2 = page.evaluate("chartInstance.data.datasets[0].label")
                n = page.evaluate("chartInstance.data.datasets[0].data.length")
                assert "0.982" in lab2 and n > 20, (lab2, n)
                return f"{lab2}; {n} ROC points"
            S("Landing: ROC toggle Messidor-2 / IDRiD", landing_roc)

            def landing_cta():
                page.locator("a:has-text('Enter Field Screener Portal')").first.click()
                page.wait_for_url(re.compile(r"/portal/screener$"))
                return page.url
            S("Landing: 'Enter Field Screener Portal' button", landing_cta)

            # ------------------------------------------------------------ screener
            def wait_done():
                page.wait_for_function("() => !document.getElementById('btn-run-ai').disabled && "
                                       "!document.getElementById('quality-status-badge').textContent.includes('Checking')")

            def case_id():
                return page.evaluate("activeCaseId")

            def scr_loads():
                page.goto(base + "/screen")
                eng = page.inner_text("header")
                assert "AI engine:" in eng, "engine label missing"
                assert page.inner_text("#camp-screened-num").strip() == "0"
                assert "No patients screened today" in page.inner_text("#roster-list-desktop")
                return re.search(r"AI engine: \w+", eng).group(0)
            S("Screener: loads with live (empty) roster and engine label", scr_loads)

            def scr_refer():
                page.fill("#patient-id", "E2E-REFER")
                page.click("[data-sample='idrid_test_grade2_009.jpg']")
                wait_done()
                head = page.inner_text("#triage-headline")
                assert "REFERRAL" in head, head
                assert "P(REFERABLE DR)" in page.inner_text("#triage-kicker")
                vals = [page.inner_text(f"#val-{k}") for k in ("fov", "completeness", "median", "clipped", "evenness", "focus")]
                assert all(v.strip() not in ("", "–") for v in vals), vals
                assert page.inner_text("#conformal-set-text").startswith("[")
                assert page.is_visible("#referral-booking-card")
                state["refer_case"] = case_id()
                return f"{head}; metrics {vals}"
            S("Screener: public sample -> real result, metrics, referral card", scr_refer)

            def scr_layers():
                srcs = {}
                for layer in ("unet", "gradcam", "redfree", "fundus"):
                    page.click(f"#layer-btn-{layer}")
                    srcs[layer] = page.get_attribute("#preview-img", "src") or ""
                    assert page.get_attribute(f"#layer-btn-{layer}", "class").count("active"), layer
                assert len(set(srcs.values())) == 4, "layers did not change the image"
                page.click("#layer-btn-unet")
                assert page.is_visible("#lesion-telemetry-row")
                counts = {k: page.inner_text(f"#cnt-{k}") for k in ("MA", "HE", "EX", "SE")}
                assert all(c.isdigit() for c in counts.values()), counts
                page.click("#layer-btn-fundus")
                return f"4 distinct images; lesion counts {counts}"
            S("Screener: Colour / U-Net / Grad-CAM / Red-free layers", scr_layers)

            def scr_loupe():
                box = page.locator("#viewport-box").bounding_box()
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.wait_for_function("() => getComputedStyle(document.getElementById('magnifier-loupe')).display === 'block'")
                page.mouse.move(5, 5)
                return "magnifier appears on hover"
            S("Screener: hover magnifier", scr_loupe)

            def scr_languages():
                out = {}
                for lang, rx in (("Tamil", r"[\u0B80-\u0BFF]"), ("Hindi", r"[\u0900-\u097F]"), ("English", r"^[\x00-\x7F\"]+$")):
                    page.locator(".cockpit-chip", has_text=lang).first.click()
                    page.wait_for_function(f"() => /{rx}/.test(document.getElementById('counseling-script-text').textContent)")
                    out[lang] = page.inner_text("#counseling-script-text")[:40]
                return str(out)
            S("Screener: counselling text in Tamil / Hindi / English", scr_languages)

            def scr_speak():
                n = len(console_errors)
                page.locator("button", has_text="Speak to Patient").click()
                time.sleep(0.5)
                assert len(console_errors) == n, console_errors[n:]
                return "no errors (audio not checked headless)"
            S("Screener: 'Speak to Patient' button", scr_speak)

            def scr_book():
                page.fill("#referral-appointment-date", time.strftime("%Y-%m-%d", time.localtime(time.time() + 5 * 86400)))
                page.click("#facility-card-2")
                page.click("#btn-confirm-referral")
                expect(page.locator("#booking-confirmation-msg")).to_be_visible()
                ref = page.inner_text("#booking-ref-num")
                assert re.fullmatch(r"REF-\d+", ref), ref
                state["ref_date"] = page.input_value("#referral-appointment-date")
                return f"{ref}: {page.inner_text('#booking-confirmed-text')}"
            S("Screener: record referral & appointment", scr_book)

            def scr_slip():
                with ctx.expect_page() as pinfo:
                    page.click("#btn-print-slip")
                slip = pinfo.value
                slip.wait_for_load_state()
                t = slip.inner_text("body")
                slip.close()
                assert "/slip/" in slip.url and ("referral" in t.lower() or "Referral" in t), t[:200]
                return slip.url.split(base)[1]
            S("Screener: print slip opens the slip", scr_slip)

            def scr_status():
                with ctx.expect_page() as pinfo:
                    page.locator("button", has_text="Patient status link").click()
                st = pinfo.value
                st.wait_for_load_state()
                ok = "/s/" in st.url and len(st.inner_text("body")) > 50
                u = st.url
                st.close()
                assert ok, u
                return u.split(base)[1]
            S("Screener: patient status link opens /s/<token>", scr_status)

            def scr_left_eye():
                pid = page.input_value("#patient-id")
                page.locator("#next-action-label").click()
                assert "active" in page.get_attribute("#eye-tab-left", "class")
                assert page.is_visible("#drop-prompt")
                page.click("[data-sample='idrid_test_grade0_030.jpg']")
                wait_done()
                head = page.inner_text("#triage-headline")
                assert "ROUTINE" in head, head
                assert "audit" in page.inner_text("#action-detail-text").lower(), "audit note missing"
                assert page.input_value("#patient-id") == pid
                eye = page.evaluate("currentExaminedEye")
                page.locator("#next-action-label").click()  # complete & next patient
                assert page.input_value("#patient-id") != pid, "no new patient ID"
                assert "active" in page.get_attribute("#eye-tab-right", "class")
                return f"left eye screened ({eye}): {head}; then new patient"
            S("Screener: left eye, audit note, then next patient", scr_left_eye)

            def scr_upload():
                page.set_input_files("#file-input", str(SAMPLES / "hrf_02_h.jpg"))
                wait_done()
                head = page.inner_text("#triage-headline")
                assert page.inner_text("#preview-filename-text") == "hrf_02_h.jpg"
                return head
            S("Screener: upload your own photo", scr_upload)

            def scr_bad_upload():
                page.set_input_files("#file-input", files=[{"name": "notes.txt", "mimeType": "text/plain", "buffer": b"not an image"}])
                wait_done()
                badge = page.inner_text("#quality-status-badge")
                assert badge == "Error", badge
                assert page.is_hidden("#result-card"), "a result was shown for a non-image"
                return page.inner_text("#quality-alert-body")[:80]
            S("Screener: non-image upload shows an error, no fake result", scr_bad_upload)

            def scr_blur():
                page.keyboard.press("Alt+n")
                page.click("[data-sample='blur']")
                wait_done()
                steps = [page.inner_text("#attempt-indicator-text")]
                for _ in range(2):
                    page.click("#btn-run-ai")
                    wait_done()
                    steps.append(page.inner_text("#attempt-indicator-text"))
                head = page.inner_text("#triage-headline")
                assert "UNGRADABLE" in head, (head, steps)
                assert page.input_value("#referral-doctor-select").startswith("Cataract")
                assert "Blurry" in page.inner_text("#quality-alert-body")
                return f"{steps} -> {head}"
            S("Screener: out-of-focus retake x3 -> ungradable referral", scr_blur)

            def scr_alt_keys():
                page.keyboard.press("Alt+n")
                assert page.is_visible("#drop-prompt")
                assert page.inner_text("#attempt-indicator-text") == "Attempt 1 of 3"
                page.click("[data-sample='hrf_05_dr.jpg']")
                wait_done()
                first = case_id()
                page.keyboard.press("Alt+r")
                page.wait_for_function(f"() => activeCaseId && activeCaseId !== {first}")
                wait_done()
                return f"Alt+N reset; Alt+R re-ran case {first} as {case_id()}"
            S("Screener: Alt+N new patient, Alt+R re-run", scr_alt_keys)

            def scr_review():
                page.keyboard.press("Alt+n")
                page.fill("#patient-id", "E2E-REVIEW")
                page.click("[data-sample='idrid_test_grade4_002.jpg']")
                wait_done()
                assert page.is_visible("#telereview-panel")
                page.wait_for_function("() => /Queue position: \\d+ of \\d+/.test(document.getElementById('telereview-position').textContent)")
                state["review_case"] = case_id()
                return page.inner_text("#telereview-position")
            S("Screener: uncertain photo -> tele-review panel with queue position", scr_review)

            def scr_roster():
                page.goto(base + "/screen")
                n = int(page.inner_text("#camp-screened-num"))
                items = page.locator("#roster-list-desktop .roster-item")
                assert items.count() == n and n >= 8, (n, items.count())
                page.fill("#roster-search-input", "E2E-REVIEW")
                page.keyboard.up("W")  # keyup triggers the filter
                vis = page.locator("#roster-list-desktop .roster-item:visible").count()
                assert vis >= 1 and vis < n, vis
                page.fill("#roster-search-input", "")
                page.locator("#roster-search-input").press("Backspace")
                page.locator("button", has_text=re.compile(r"^Recall \(")).click()
                cats = page.eval_on_selector_all("#roster-list-desktop .roster-item", "els => els.filter(e => e.style.display !== 'none').map(e => e.dataset.cat)")
                assert cats and set(cats) == {"recall"}, cats
                page.click("#sidebar-tab-referrals")
                refs = page.inner_text("#sidebar-panel-referrals")
                assert "State:" in refs and "booked" in refs, refs[:200]
                return f"{n} cases today; search, Recall filter and referrals tab work"
            S("Screener: roster count, search, filter, referrals tab", scr_roster)

            def scr_mobile():
                m = ctx.new_page()
                watch(m, "mobile")
                m.set_viewport_size({"width": 390, "height": 844})
                m.goto(base + "/screen")
                sw = m.evaluate("document.documentElement.scrollWidth")
                m.click("button[title=\"Open Today's Patient Roster\"]")
                expect(m.locator("#cockpit-roster-offcanvas")).to_be_visible()
                n = m.locator("#cockpit-roster-offcanvas a[href^='/case/']").count()
                m.close()
                assert sw <= 392, f"horizontal scroll: page is {sw}px wide"
                assert n >= 1
                return f"no sideways scroll at 390 px; drawer lists {n} cases"
            S("Screener: phone width layout and roster drawer", scr_mobile)

            # ------------------------------------------------------------ doctor
            def doc_loads():
                page.goto(base + "/portal/doctor")
                cards = page.locator(".queue-card")
                assert cards.count() >= 4, cards.count()
                page.wait_for_function("() => caseData && caseData.images")
                assert page.get_attribute("#cockpit-fundus-img", "src").startswith("data:image")
                assert "Audit sample" in page.inner_text(".queue-col")
                return f"{cards.count()} queued cases incl. audit samples; first case loaded"
            S("Doctor: real queue and first case", doc_loads)

            def doc_layers():
                srcs = set()
                for layer in ("raw", "gradcam", "lesion"):
                    page.click(f"#btn-layer-{layer}")
                    srcs.add(page.get_attribute("#cockpit-fundus-img", "src"))
                assert len(srcs) == 3
                assert "P(referable DR)" in page.inner_text("#ai-summary")
                assert page.locator("#quadrant-grid > div").count() == 4
                return "3 distinct layers; summary and quadrants filled"
            S("Doctor: image layers, AI summary, quadrant counts", doc_layers)

            def doc_needs_grade():
                page.keyboard.press("Enter")
                assert "Choose a grade" in page.inner_text("#sign-status")
                return "sign-off blocked until a grade is chosen"
            S("Doctor: cannot sign without a grade", doc_needs_grade)

            def doc_keys():
                first = page.evaluate("caseId")
                page.keyboard.press("u")
                assert "selected" in page.get_attribute("#g-u", "class")
                before = page.is_checked("#chk-dme")
                page.keyboard.press("d")
                assert page.is_checked("#chk-dme") != before
                page.keyboard.press("Enter")
                page.wait_for_function("() => document.getElementById('sign-status').textContent.startsWith('Saved')")
                msg = page.inner_text("#sign-status")
                assert "referral created" in msg, msg
                page.wait_for_function(f"() => caseId !== {first} && caseData")
                return f"U + D + Enter on case {first}: {msg}; advanced to {page.evaluate('caseId')}"
            S("Doctor: keys U / D / Enter, auto-advance", doc_keys)

            def doc_recall():
                page.click("#g-1")
                page.uncheck("#chk-dme")
                page.click("#btn-sign")
                page.wait_for_function("() => document.getElementById('sign-status').textContent.startsWith('Saved')")
                msg = page.inner_text("#sign-status")
                assert "routine recall" in msg, msg
                page.wait_for_function("() => caseData && !document.getElementById('sign-status').textContent")
                return msg
            S("Doctor: grade 1 -> routine recall", doc_recall)

            def doc_refer_with_date():
                cid = page.evaluate("caseId")
                page.click("#g-3")
                date = time.strftime("%Y-%m-%d", time.localtime(time.time() + 3 * 86400))
                page.fill("#ref-date", date)
                page.click("#btn-sign")
                page.wait_for_function("() => document.getElementById('sign-status').textContent.startsWith('Saved')")
                r = page.request.get(f"{base}/referrals").text()
                assert date in r, "appointment date not on the referral tracker"
                state["doc_ref_case"] = cid
                return f"case {cid}: grade 3, booked for {date}"
            S("Doctor: grade 3 with appointment date -> booked referral", doc_refer_with_date)

            def doc_reviewer_name():
                page.fill("#reviewer-name", "Dr E2E")
                page.dispatch_event("#reviewer-name", "change")
                page.reload()
                v = page.input_value("#reviewer-name")
                assert v == "Dr E2E", v
                return "remembered after reload"
            S("Doctor: reviewer name remembered", doc_reviewer_name)

            def screener_sees_decision():
                cid = state.get("review_case")
                j = page.request.get(f"{base}/api/case/{cid}").json()
                if not j["final"]:  # sign it now if the queue order left it untouched
                    page.request.post(f"{base}/api/review/{cid}", data={"grade": 4, "ms": 9000, "reviewer": "e2e"})
                p2 = ctx.new_page()
                p2.goto(base + "/screen")
                p2.evaluate(f"activeCaseId = {cid}; document.getElementById('telereview-panel').classList.remove('hidden'); startPolling();")
                p2.wait_for_function("() => document.getElementById('telereview-position').textContent === 'Reviewed'")
                t = p2.inner_text("#telereview-text")
                p2.close()
                return t
            S("Screener: picks up the specialist's decision", screener_sees_decision)

            # ------------------------------------------------------------ classic pages
            def classic_review():
                page.goto(base + "/review")
                links = page.locator("a[href^='/review/']")
                if links.count() == 0:
                    return "queue empty (all signed in the doctor portal)"
                links.first.click()
                page.wait_for_url(re.compile(r"/review/\d+"))
                for tab in ("photo", "gradcam", "lesions"):
                    if page.locator(f"#tab-{tab}").count():
                        page.click(f"#tab-{tab}")
                page.click("#btn-grade-2")
                page.wait_for_function("() => document.getElementById('done').textContent.includes('Saved')")
                return page.inner_text("#done")
            S("Classic review page: tabs and grading", classic_review)

            def referral_tracker():
                page.goto(base + "/referrals")
                row = page.locator("tr", has=page.locator("button:has-text('Mark Explained')")).first
                rid = row.locator("button:has-text('Mark Explained')").get_attribute("onclick")
                rid = re.search(r"adv\((\d+)", rid).group(1)
                steps = []
                for label in ("Explained", "Booked", "Missed", "Escalated"):
                    if label == "Booked":
                        page.fill(f"#d{rid}", time.strftime("%Y-%m-%d", time.localtime(time.time() + 7 * 86400)))
                    page.locator(f"button[onclick=\"adv({rid}, '{label.lower()}')\"]").click()
                    page.wait_for_load_state()
                    page.wait_for_timeout(300)
                    steps.append(label)
                dash = page.request.get(f"{base}/dashboard").text()
                assert "SMS (mock)" in dash and "family" in dash, "escalation SMS to family not in the outbox"
                return f"referral {rid}: " + " -> ".join(steps) + "; outbox shows mock SMS to family"
            S("Referral tracker: explained -> booked -> missed -> escalated", referral_tracker)

            def booked_needs_date():
                page.goto(base + "/referrals")
                btn = page.locator("button:has-text('Mark Booked')").first
                if btn.count() == 0:
                    return "no referral at the booking step"
                msgs = []
                page.once("dialog", lambda d: (msgs.append(d.message), d.dismiss()))
                btn.click()
                page.wait_for_timeout(300)
                assert msgs and "date" in msgs[0].lower(), msgs
                return msgs[0]
            S("Referral tracker: booking requires a date", booked_needs_date)

            def dashboard():
                page.goto(base + "/dashboard")
                t = page.inner_text("body")
                assert "recall" in t.lower() and "refer" in t.lower()
                return "loads with outcome counts"
            S("Dashboard", dashboard)

            def result_page():
                cid = state["refer_case"]
                out = []
                for lang, rx in (("en", r"[A-Za-z]"), ("hi", r"[\u0900-\u097F]"), ("ta", r"[\u0B80-\u0BFF]")):
                    page.goto(f"{base}/case/{cid}?lang={lang}")
                    assert re.search(rx, page.inner_text("body")), lang
                    out.append(lang)
                return f"case {cid} in {out}"
            S("Result page in three languages", result_page)

            def study():
                page.goto(base + "/study")
                if page.locator("#f-reader").count() == 0:
                    raise AssertionError("no study set (app/study_set.csv missing)")
                page.fill("#f-reader", "E2E")
                page.fill("#f-n", "2")
                page.locator("button:has-text('Start Reading Session')").click()
                page.wait_for_url(re.compile(r"/study/\d+/0"))
                page.click("#btn-grade-0")
                page.wait_for_url(re.compile(r"/study/\d+/1"))
                page.click("#btn-grade-2")
                page.wait_for_url(re.compile(r"/study-results/\d+"))
                t = page.inner_text("body")
                assert re.search(r"\d+(\.\d+)?\s*s", t), t[:300]
                return "2-case session read and scored"
            S("Reader study: start, read 2 cases, results", study)

            browser.close()
    finally:
        server.terminate()

    width = max(len(n) for n, _, _ in results)
    print()
    for n, st, d in results:
        print(f"{st:4}  {n:<{width}}  {d}")
    fails = sum(st == "FAIL" for _, st, _ in results)
    print(f"\n{len(results) - fails} passed, {fails} failed; backend {a.backend}")
    if console_errors:
        print(f"\nbrowser errors ({len(console_errors)}):")
        for e in dict.fromkeys(console_errors):
            print("  ", e)
    print(f"server log and failure screenshots: {work}")
    sys.exit(1 if fails or console_errors else 0)


if __name__ == "__main__":
    main()
