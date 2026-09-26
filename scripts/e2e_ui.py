"""End-to-end UI check: starts the web app on a throwaway database and clicks
through every screen and state with Playwright (Chromium).

    python scripts/e2e_ui.py                    # PyTorch backend, headless
    python scripts/e2e_ui.py --backend matlab   # the MATLAB engine instead
    python scripts/e2e_ui.py --headed           # watch it

Every automatic recall is sent to the audit queue (DRISHTI_AUDIT_FRACTION=1)
so the specialist has several cases to work through. A second server with an
empty model folder checks the "grading is off" states. Prints PASS / FAIL
per feature, saves a screenshot for each failure, and exits non-zero on any
failure or browser error. Your own app/drishti.db is never touched.
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
          "Zero Retention", "ABDM", "Quantized", "98.4%", "90.5%", "Anita", "Rajesh", "800-DRISHTI"]
DEVANAGARI, TAMIL = r"[ऀ-ॿ]", r"[஀-௿]"

results: list[tuple[str, str, str]] = []
console_errors: list[str] = []


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def start_server(port: int, db: Path, backend: str, extra_env: dict | None = None) -> subprocess.Popen:
    env = {**os.environ, "DRISHTI_DB": str(db), "DRISHTI_AUDIT_FRACTION": "1", "DRISHTI_BACKEND": backend, **(extra_env or {})}
    log = open(db.with_suffix(".log"), "w")
    p = subprocess.Popen([sys.executable, "-m", "uvicorn", "app.main:app", "--port", str(port)],
                         cwd=REPO, env=env, stdout=log, stderr=subprocess.STDOUT)
    for _ in range(240):
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
    port, port2 = free_port(), free_port()
    base, base2 = f"http://127.0.0.1:{port}", f"http://127.0.0.1:{port2}"
    server = start_server(port, work / "e2e.db", a.backend)
    empty_runs = work / "no_models"
    empty_runs.mkdir()
    server2 = start_server(port2, work / "nomodel.db", "python", {"DRISHTI_RUNS": str(empty_runs)})
    slow = 300_000 if a.backend == "matlab" else 30_000  # MATLAB imports its networks on the first photo
    try:
        with sync_playwright() as pw:
            browser = pw.chromium.launch(headless=not a.headed)
            desk = browser.new_context(viewport={"width": 1440, "height": 900}, accept_downloads=True)
            phone = browser.new_context(viewport={"width": 390, "height": 844}, is_mobile=True, has_touch=True)
            for c in (desk, phone):
                c.set_default_timeout(slow)
            page = desk.new_page()
            watch(page, "desktop")
            hw = phone.new_page()
            watch(hw, "phone")
            st = {}

            def S(name, fn, pg=None):
                run(name, fn, pg or page, shots)

            # ================================================================ entry
            def entry():
                page.goto(base + "/")
                txt = page.content()
                bad = [b for b in BANNED if b in txt]
                assert not bad, f"removed claims are back: {bad}"
                roles = {a_.get_attribute("data-role"): a_.get_attribute("href") for a_ in page.locator(".rolelist__item").all()}
                assert roles == {"health-worker": "/screen", "specialist": "/review", "coordinator": "/referrals",
                                 "programme": "/dashboard", "researcher": "/study"}, roles
                page.click("text=About and evidence")
                page.wait_for_url(re.compile(r"/about$"))
                page.click("text=Validation results and simulated workload")
                page.wait_for_url(re.compile(r"/evidence$"))
                return "5 roles, about, evidence link"
            S("Entry: role picker, about and evidence links", entry)

            def entry_phone():
                hw.goto(base + "/")
                assert hw.locator(".rolelist__item span").first.is_hidden(), "descriptions should hide on phones"
                sw = hw.evaluate("document.documentElement.scrollWidth")
                assert sw <= 392, f"sideways scroll {sw}px"
                return "compact list, no sideways scroll"
            S("Entry: phone layout", entry_phone, hw)

            # ================================================================ health worker (phone)
            def hw_wait_result():
                hw.wait_for_function("() => ['result','retake','error'].includes(document.body.dataset.view)")

            def hw_first_use():
                hw.goto(base + "/screen")
                expect(hw.locator("#intro")).to_be_visible()
                expect(hw.locator("#today-list")).to_contain_text("No patients screened today")
                assert hw.inner_text("#btn-start") == "Start screening"
                return "intro, empty day, Start screening"
            S("Health worker: first use, empty day (A1)", hw_first_use, hw)

            def hw_patient_form():
                hw.click("#btn-start")
                expect(hw.locator("[data-screen=patient]")).to_be_visible()
                hw.click("#btn-continue")
                expect(hw.locator("#f-patient-err")).to_be_visible()
                hw.fill("#f-patient", "e2e-001")
                hw.select_option("#f-site", "PHC-2")
                hw.click("#f-lang [data-lang=hi]")
                hw.click("#btn-continue")
                expect(hw.locator("[data-screen=capture]")).to_be_visible()
                assert hw.inner_text("[data-screen=capture] [data-bind=patient]") == "E2E-001"
                assert "Right eye" in hw.inner_text("#eye-stepper")
                return "code required; code upper-cased; site and language kept"
            S("Health worker: new patient form and validation (A3)", hw_patient_form, hw)

            def hw_capture():
                expect(hw.locator("#capture-empty")).to_be_visible()
                hw.click("[data-act=samples]")
                expect(hw.locator("#sample-sheet")).to_be_visible()
                hw.click("[data-sample='idrid_test_grade2_009.jpg']:not([data-simulate])")
                expect(hw.locator("#capture-img")).to_be_visible()
                assert "Attempt 1 of 3" in hw.inner_text("#capture-attempt")
                hw.click("[data-act=take-again]")
                expect(hw.locator("#capture-empty")).to_be_visible()
                hw.click("[data-act=samples]")
                hw.click("[data-sample='idrid_test_grade2_009.jpg']:not([data-simulate])")
                return "sample sheet, photo preview, take again"
            S("Health worker: capture, sample photo, take again (A4/A5)", hw_capture, hw)

            def hw_result_urgent():
                hw.click("#btn-check")
                expect(hw.locator("[data-screen=checking]")).to_be_visible(timeout=5000)
                hw_wait_result()
                kind = hw.get_attribute("#r-banner", "data-kind")
                assert kind == "urgent", kind
                assert re.search(DEVANAGARI, hw.inner_text("#r-say")), "message should start in Hindi"
                assert re.search(DEVANAGARI, hw.inner_text("#disclaimer")), "Hindi disclaimer line missing"
                hw.click("#r-lang [data-lang=en]")
                hw.wait_for_function("() => /Serious signs/.test(document.getElementById('r-say').textContent)")
                hw.click("#r-lang [data-lang=ta]")
                hw.wait_for_function(f"() => /{TAMIL}/.test(document.getElementById('r-say').textContent)")
                hw.click("#r-lang [data-lang=en]")
                assert hw.inner_text("#btn-result-main") == "Book urgent visit"
                assert "btn--danger" in hw.get_attribute("#btn-result-main", "class")
                return f"{hw.inner_text('#r-title')}; Hindi, Tamil, English messages"
            S("Health worker: result, urgent refer, three languages (C6/F1/F2)", hw_result_urgent, hw)

            def hw_details():
                hw.click("#btn-details")
                expect(hw.locator("[data-screen=details]")).to_be_visible()
                srcs = set()
                for l in ("photo", "lesions", "gradcam"):
                    hw.click(f"#d-layers [data-layer={l}]")
                    srcs.add(hw.get_attribute("#d-img", "src"))
                assert len(srcs) == 3, "layers did not change the image"
                counts = hw.inner_text("#d-counts")
                assert re.search(r"\d", counts), counts
                pref = hw.inner_text("#d-pref")
                hw.click("[data-screen=details] [data-act=back]")
                expect(hw.locator("[data-screen=result]")).to_be_visible()
                return f"3 layers; P(ref) {pref}; counts {counts.split()}"
            S("Health worker: details with layers (C7)", hw_details, hw)

            def hw_book():
                hw.click("#btn-result-main")
                expect(hw.locator("[data-screen=book]")).to_be_visible()
                assert "Urgent refer" in hw.inner_text("#b-badge")
                hw.click("#b-facilities .choice >> nth=1")
                hw.click("[data-act=other-date]")
                hw.fill("#b-date-other", time.strftime("%Y-%m-%d", time.localtime(time.time() + 4 * 86400)))
                hw.check("#b-transit")
                hw.click("#btn-book")
                expect(hw.locator("[data-screen=booked]")).to_be_visible()
                st["token_url"] = hw.get_attribute("#k-link", "href")
                st["slip_url"] = hw.get_attribute("#btn-slip", "href")
                assert "Transport requested" in hw.inner_text("#k-help")
                assert hw.inner_text("#btn-after-booking") == "Next: left eye"
                return f"{hw.inner_text('#k-where').splitlines()[0]}, {hw.inner_text('#k-when')}"
            S("Health worker: book hospital and date (D1/D2)", hw_book, hw)

            def hw_slip_and_status():
                p = phone.new_page()
                p.goto(base + st["slip_url"])
                assert p.locator(".slip__qr svg").count() == 1, "no QR code"
                txt = p.inner_text(".slip")
                assert "URGENT" in txt and re.search(DEVANAGARI, txt) and re.search(TAMIL, txt), "slip should be trilingual"
                p.goto(base + st["token_url"])
                assert re.search(DEVANAGARI, p.inner_text(".banner__title")), "opens in the language chosen at screening (Hindi)"
                p.goto(base + st["token_url"] + "?lang=en")
                assert p.get_attribute(".banner", "data-kind") == "urgent"
                assert "Your appointment" in p.inner_text("main")
                p.goto(base + st["token_url"] + "?lang=ta")
                assert re.search(TAMIL, p.inner_text(".banner__title"))
                p.close()
                return "QR slip in 3 languages; status page with appointment; Tamil"
            S("Patient: slip and status page after booking (P1/P3/P7)", hw_slip_and_status, hw)

            def hw_left_review():
                hw.click("#btn-after-booking")
                expect(hw.locator("[data-screen=capture]")).to_be_visible()
                assert "Right eye · done" in hw.inner_text("#eye-stepper")
                hw.click("[data-act=samples]")
                hw.click("[data-sample='idrid_test_grade4_002.jpg']")
                hw.click("#btn-check")
                hw_wait_result()
                assert hw.get_attribute("#r-banner", "data-kind") == "review"
                expect(hw.locator("#r-wait")).to_be_visible()
                assert re.search(r"in line|in the queue", hw.inner_text("#r-wait-text"))
                assert hw.inner_text("#btn-result-main") == "Finish patient"
                expect(hw.locator("#result-hint")).to_be_visible()
                st["review_case"] = hw.evaluate("S.current.id")
                hw.click("#btn-result-main")
                expect(hw.locator("[data-screen=today]")).to_be_visible()
                hw.wait_for_function("() => /E2E-001/.test(document.getElementById('today-list').textContent)")
                txt = hw.inner_text("#today-list")
                assert "Waiting for eye doctor" in txt and "booked" in txt, txt
                return "left eye sent to review with its queue position; today list shows both eyes"
            S("Health worker: left eye, review pending, finish (C3/A2)", hw_left_review, hw)

            def hw_retakes():
                hw.click("#btn-start")
                hw.fill("#f-patient", "E2E-002")
                hw.click("#btn-continue")
                hw.click("[data-act=samples]")
                hw.click("[data-sample][data-simulate=blur]")
                seen = []
                for i in range(3):
                    hw.click("#btn-check")
                    hw_wait_result()
                    scr = hw.evaluate("document.body.dataset.view")
                    if scr == "retake":
                        seen.append(hw.inner_text("#retake-attempt"))
                        assert "Blurry" in hw.inner_text("#problems")
                        hw.click("#btn-retake")
                        hw.click("[data-act=samples]")
                        hw.click("[data-sample][data-simulate=blur]")
                assert seen == ["1 of 3", "2 of 3"], seen
                assert hw.get_attribute("#r-banner", "data-kind") == "ungradable"
                assert hw.inner_text("#btn-result-main") == "Choose hospital and date"
                return f"retake {seen} then {hw.inner_text('#r-title')}"
            S("Health worker: retake 1 and 2 of 3, then ungradable (B3/B4/B5)", hw_retakes, hw)

            def hw_recall_audit():
                hw.click("#btn-result-main")
                hw.click("#btn-book")  # default hospital and date
                expect(hw.locator("[data-screen=booked]")).to_be_visible()
                hw.click("#btn-after-booking")
                hw.click("[data-act=samples]")
                hw.click("[data-sample='idrid_test_grade0_030.jpg']")
                hw.click("#btn-check")
                hw_wait_result()
                assert hw.get_attribute("#r-banner", "data-kind") == "recall"
                expect(hw.locator("#r-audit")).to_be_visible()
                return "ungradable booked; recall with the audit note"
            S("Health worker: recall picked for audit (C1/C2)", hw_recall_audit, hw)

            def hw_upload_and_bad_file():
                hw.click("#btn-result-main")  # finish
                hw.click("#btn-start")
                hw.fill("#f-patient", "E2E-003")
                hw.click("#btn-continue")
                hw.set_input_files("#in-files", files=[{"name": "notes.txt", "mimeType": "text/plain", "buffer": b"not an image"}])
                hw.click("#btn-check")
                hw_wait_result()
                assert hw.evaluate("document.body.dataset.view") == "error"
                assert "not a photo" in hw.inner_text("#e-title")
                hw.click("[data-screen=error] [data-act=check]")  # try again: same file, same answer
                hw_wait_result()
                assert hw.evaluate("document.body.dataset.view") == "error"
                hw.evaluate("S.stack = ['capture']")
                hw.evaluate("back()")
                hw.set_input_files("#in-files", str(SAMPLES / "hrf_02_h.jpg"))
                hw.click("#btn-check")
                hw_wait_result()
                return f"non-image -> error, no outcome; own photo -> {hw.inner_text('#r-title')}"
            S("Health worker: bad file error, then own photo (E3)", hw_upload_and_bad_file, hw)

            def hw_network_error():
                hw.click("#btn-result-main")
                hw.click("[data-act=samples]")
                hw.click("[data-sample='hrf_05_dr.jpg']")
                hw.route("**/api/screen", lambda r: r.abort())
                hw.click("#btn-check")
                hw_wait_result()
                txt = hw.inner_text("[data-screen=error]")
                hw.unroute("**/api/screen")
                assert "No result received" in txt and "Do not tell the patient a result" in txt
                assert hw.locator("[data-screen=error] .thumb img").get_attribute("src"), "photo should be kept"
                hw.click("[data-screen=error] [data-act=check]")
                hw_wait_result()
                return f"offline -> no outcome shown; retry -> {hw.inner_text('#r-title')}"
            S("Health worker: network error keeps the photo, retry works (E2)", hw_network_error, hw)

            def hw_warmup_and_cancel():
                hw.click("#btn-result-main")
                hw.click("#btn-start")
                hw.fill("#f-patient", "E2E-004")
                hw.click("#btn-continue")
                hw.click("[data-act=samples]")
                hw.click("[data-sample='hrf_02_h.jpg']")

                def slow_route(r):
                    time.sleep(9.5)
                    r.continue_()
                hw.route("**/api/screen", slow_route)
                hw.click("#btn-check")
                expect(hw.locator("#warmup")).to_be_visible(timeout=15000)
                el = hw.inner_text("#warm-elapsed")
                hw.click("[data-act=cancel-check]")
                expect(hw.locator("[data-screen=capture]")).to_be_visible(timeout=15000)
                hw.unroute("**/api/screen")
                return f"slow check shows the start-up timer ({el}); cancel returns to capture"
            S("Health worker: warm-up timer and cancel (B1/B2)", hw_warmup_and_cancel, hw)

            def hw_decision_arrives():
                cid = st["review_case"]
                hw.evaluate("show('today', false)")
                page.request.post(f"{base}/api/review/{cid}", data={"grade": 2, "dme": 0, "ms": 12000, "reviewer": "e2e"})
                expect(hw.locator(".toast")).to_contain_text("answer", timeout=20000)
                hw.locator(".toast button").first.click()
                expect(hw.locator("[data-screen=result]")).to_be_visible()
                expect(hw.locator("#r-decision")).to_be_visible()
                txt = hw.inner_text("#r-decision-text")
                assert hw.get_attribute("#r-banner", "data-kind") == "refer"
                return f"toast, then result shows the decision: {txt}"
            S("Health worker: specialist decision arrives as a toast (C4)", hw_decision_arrives, hw)

            def hw_desktop():
                p = desk.new_page()
                watch(p, "hw-desktop")
                p.goto(base + "/screen")
                expect(p.locator(".hw__aside")).to_be_visible()
                p.wait_for_function("() => document.querySelectorAll('#today-aside .patientcard').length >= 3")
                n = p.locator("#today-aside .patientcard").count()
                p.locator("#today-aside .eyerow[data-case]").first.click()
                expect(p.locator("[data-screen=result]")).to_be_visible()
                p.close()
                return f"today list beside the screen ({n} patients); opens a case"
            S("Health worker: desktop layout with today list (G1)", hw_desktop)

            # ================================================================ grading off (second server)
            def grading_off():
                p = phone.new_page()
                p.goto(base2 + "/screen")
                p.click("#btn-start")
                p.fill("#f-patient", "NM-1")
                p.click("#btn-continue")
                expect(p.locator(".strip--review")).to_contain_text("Automatic grading is off")
                p.click("[data-act=samples]")
                p.click("[data-sample='idrid_test_grade2_009.jpg']:not([data-simulate])")
                p.click("#btn-check")
                p.wait_for_function("() => document.body.dataset.view === 'result'")
                kind = p.get_attribute("#r-banner", "data-kind")
                cid = p.evaluate("S.current.id")
                d = desk.new_page()
                d.goto(f"{base2}/review/{cid}")
                expect(d.locator("#no-ai")).to_be_visible()
                p.close(); d.close()
                assert kind == "review", kind
                return "banner on capture; photo goes to review; specialist sees 'No AI grade'"
            S("Models missing: banner and 'No AI grade' (E1)", grading_off)

            # ================================================================ specialist
            def spec_queue():
                page.goto(base + "/review")
                rows = page.locator("#q-rows tr")
                n = rows.count()
                assert n >= 3, n
                assert page.locator("#q-cards").is_hidden(), "phone cards should not show on desktop"
                txt = page.inner_text("#q-rows")
                assert "Audit" in txt, "audit samples missing"
                page.fill("#reviewer", "Dr E2E")
                page.dispatch_event("#reviewer", "change")
                page.locator("h1").click()
                page.keyboard.press("Enter")
                page.wait_for_url(re.compile(r"/review/\d+"))
                page.wait_for_function("() => typeof C !== 'undefined' && C && C.id")
                return f"{n} queued cases incl. audit; Enter opens the first"
            S("Specialist: queue, Enter starts reviewing (S1)", spec_queue)

            def spec_layers_zoom():
                srcs = set()
                for k in ("l", "h", "p"):
                    page.keyboard.press(k)
                    srcs.add(page.get_attribute("#img", "src"))
                assert len(srcs) == 3
                box = page.locator("#viewer").bounding_box()
                page.mouse.move(box["x"] + box["width"] / 2, box["y"] + box["height"] / 2)
                page.mouse.wheel(0, -400)
                tf = page.evaluate("document.getElementById('img').style.transform")
                assert "scale(1)" not in tf, tf
                page.dblclick("#viewer")
                assert "scale(1)" in page.evaluate("document.getElementById('img').style.transform")
                return "P/L/H switch layers; wheel zooms; double-click resets"
            S("Specialist: layers by key, zoom and reset (S2)", spec_layers_zoom)

            def spec_preview_and_save():
                assert page.is_disabled("#confirm")
                page.keyboard.press("Enter")  # nothing to save yet
                prev = {}
                for k, want in (("2", "Refer"), ("3", "Urgent refer"), ("u", "Ungradable"), ("1", "Recall")):
                    page.keyboard.press(k)
                    prev[k] = page.inner_text("#preview")
                    assert prev[k].startswith(want), (k, prev[k])
                page.keyboard.press("d")
                assert page.inner_text("#preview").startswith("Urgent refer"), "oedema should make it urgent"
                page.keyboard.press("d")
                page.keyboard.press("3")
                page.fill("#appt", time.strftime("%Y-%m-%d", time.localtime(time.time() + 6 * 86400)))
                first = page.evaluate("C.id")
                page.locator("#img").click()
                page.keyboard.press("Enter")
                expect(page.locator(".toast").last).to_contain_text(f"Saved case {first}")
                page.wait_for_function(f"() => C && C.id !== {first}")
                refs = page.request.get(base + "/api/referrals").json()["referrals"]
                r = next(x for x in refs if x["case_id"] == first)
                assert r["appointment"], "appointment not booked"
                return f"previews {list(prev.values())}; case {first} saved, booked {r['appointment']}, next case opened"
            S("Specialist: preview, confirm with appointment, auto-advance", spec_preview_and_save)

            def spec_escape():
                page.keyboard.press("Escape")
                page.wait_for_url(re.compile(r"/review$"))
                return "Esc returns to the queue"
            S("Specialist: Esc back to queue", spec_escape)

            def spec_load_error():
                cid = page.request.get(base + "/api/queue").json()["cases"][0]["id"]
                page.route("**/api/case/**", lambda r: r.abort())
                page.goto(f"{base}/review/{cid}")
                expect(page.locator("#state-error")).to_be_visible()
                page.unroute("**/api/case/**")
                page.click("#btn-retry")
                expect(page.locator("#work")).to_be_visible()
                return "error state with retry; retry loads the case"
            S("Specialist: load error and retry", spec_load_error)

            def spec_phone():
                p = phone.new_page()
                watch(p, "spec-phone")
                p.goto(base + "/review")
                expect(p.locator("#q-cards")).to_be_visible()
                p.locator("#q-cards .qcard").first.click()
                p.wait_for_function("() => typeof C !== 'undefined' && C && C.id")
                expect(p.locator("#phone-sum")).to_be_visible()
                sw = p.evaluate("document.documentElement.scrollWidth")
                assert sw <= 392, f"case screen is {sw}px wide on a 390px phone"
                p.click("#btn-evidence")
                expect(p.locator("#aside")).to_be_visible()
                p.click(".grade[data-grade='0']")
                p.click("#confirm")
                expect(p.locator(".toast").last).to_contain_text("Saved case")
                p.close()
                return "cards, summary, all evidence, touch grading"
            S("Specialist: phone queue and touch grading (S4/S5)", spec_phone)

            def spec_clear():
                page.goto(base + "/review")
                n = page.locator("#q-rows tr").count()
                if n:
                    page.locator("#q-rows tr").first.click()
                    for _ in range(n + 2):
                        page.wait_for_function("() => (typeof C !== 'undefined' && C && C.id) || !document.getElementById('state-clear').hidden")
                        if page.is_visible("#state-clear"):
                            break
                        page.keyboard.press("1")
                        cid = page.evaluate("C.id")
                        page.keyboard.press("Enter")
                        page.wait_for_function(f"() => !document.getElementById('state-clear').hidden || (C && C.id !== {cid})")
                expect(page.locator("#state-clear")).to_be_visible()
                summary = page.inner_text("#clear-summary")
                page.goto(base + "/review")
                expect(page.locator("#q-empty")).to_be_visible()
                return f"queue clear: {summary.split('.')[0]}; empty queue shows the keys"
            S("Specialist: queue clear and empty queue (S3)", spec_clear)

            # ================================================================ coordinator
            def coord():
                page.goto(base + "/referrals")
                assert page.locator("#cards").is_hidden()
                n = page.locator("#rows tr").count()
                assert n >= 3, n
                row = page.locator("#rows tr", has_text="Flagged").first
                rid = row.get_attribute("data-id")
                row.click()
                steps = []
                page.click("#p-moves [data-move=explained]"); steps.append("explained")
                page.wait_for_function(f"() => D.referrals.find(r => r.id == {rid}).state === 'explained'")
                page.click("#p-moves [data-move=booked]")  # no date: refused
                expect(page.locator(".toast").last).to_contain_text("appointment date")
                page.fill("#p-date", time.strftime("%Y-%m-%d", time.localtime(time.time() + 8 * 86400)))
                page.click("#p-moves [data-move=booked]"); steps.append("booked")
                page.wait_for_function(f"() => D.referrals.find(r => r.id == {rid}).state === 'booked'")
                page.click("#p-moves [data-move=missed]"); steps.append("missed")
                page.wait_for_function(f"() => D.referrals.find(r => r.id == {rid}).state === 'missed'")
                page.click("#p-moves [data-move=escalated]"); steps.append("escalated")
                page.wait_for_function(f"() => D.referrals.find(r => r.id == {rid}).tier === 'family'")
                assert "Family" in page.inner_text(".tier--on")
                hist = page.locator("#p-hist li").count()
                page.click("#filters [data-filter='Missed / escalated']")
                assert page.locator("#rows tr").count() >= 1
                return f"R-{rid}: {' -> '.join(steps)}; contact now family; {hist} history entries; filters"
            S("Coordinator: tracker, moves, booking needs a date, escalation (K1)", coord)

            def coord_close():
                page.click("#filters [data-filter='All open']")
                row = page.locator("#rows tr", has_text="Booked").first
                rid = row.get_attribute("data-id")
                row.click()
                page.click("#p-moves [data-move=attended]")
                page.wait_for_function(f"() => D.referrals.find(r => r.id == {rid}).state === 'attended'")
                page.click("#p-moves [data-move=closed]")
                page.wait_for_function(f"() => D.referrals.find(r => r.id == {rid}).state === 'closed'")
                page.click("#filters [data-filter=Closed]")
                assert page.locator(f"#rows tr[data-id='{rid}']").count() == 1
                return f"R-{rid} attended then closed; shows under Closed"
            S("Coordinator: attend and close", coord_close)

            def outbox():
                page.goto(base + "/outbox")
                t = page.inner_text("body")
                assert "MOCK" in t and "No SMS gateway is connected" in t
                assert "Family" in page.inner_text("#outbox"), "escalation message to family missing"
                return f"{page.locator('#outbox tbody tr').count()} mock messages, labelled"
            S("Coordinator: SMS outbox labelled mock (K2)", outbox)

            def coord_phone():
                p = phone.new_page()
                p.goto(base + "/referrals")
                expect(p.locator("#cards")).to_be_visible()
                b = p.locator("#cards [data-card-move]").first
                label = b.inner_text()
                b.click()
                expect(p.locator(".toast").last).to_be_visible()
                p.close()
                return f"cards with the next move ({label})"
            S("Coordinator: phone cards (K4)", coord_phone)

            # ================================================================ patient
            def patient_states():
                p = phone.new_page()
                r = p.goto(base + "/s/not-a-real-token")
                assert r.status == 404
                t = p.inner_text("main")
                assert "did not open" in t and re.search(DEVANAGARI, t) and re.search(TAMIL, t)
                today = page.request.get(base + "/api/today").json()["patients"]
                eyes = [e for pt in today for e in pt["eyes"]]
                rec = next(e for e in eyes if e["view"]["kind"] == "recall")
                p.goto(f"{base}/s/{rec['token']}?lang=en")
                assert p.get_attribute(".banner", "data-kind") == "recall"
                assert "Next check" in p.inner_text("main")
                ung = next(e for e in eyes if e["view"]["kind"] == "ungradable")
                p.goto(f"{base}/s/{ung['token']}?lang=hi")
                assert re.search(DEVANAGARI, p.inner_text(".banner__title"))
                p.close()
                return "bad link in 3 languages; recall with next check; ungradable in Hindi"
            S("Patient: bad link, recall and ungradable pages (P3/P4/P5)", patient_states)

            # ================================================================ programme
            def dashboard():
                page.goto(base + "/dashboard")
                total = int(page.inner_text("#k-total"))
                assert total >= 6, total
                assert page.locator(".stackbar__seg").count() >= 3
                assert page.locator(".funnel__track").count() == 5
                return f"{total} eyes, outcome bar, referral funnel"
            S("Programme: dashboard (M1)", dashboard)

            def evidence():
                page.goto(base + "/evidence")
                t = page.inner_text("#m2")
                assert "96.3%" in t and "87.4%" in t and "0.982" in t and "96.8%" in t
                assert "SIMULATED" in page.inner_text("main")
                assert page.locator(".forest__pt").count() == 2
                return "Messidor-2 vs IDx-DR with CIs, IDRiD, simulated workload"
            S("Programme: evidence page (M2)", evidence)

            # ================================================================ research
            def study():
                page.goto(base + "/study")
                page.fill("#f-reader", "E2E reader")
                page.evaluate("document.getElementById('f-n-val').value = 3")  # a short session for the test
                page.click("button:has-text('Start session')")
                page.wait_for_url(re.compile(r"/study/\d+/0"))
                sid = re.search(r"/study/(\d+)/", page.url).group(1)
                page.click("#btn-pause")
                expect(page.locator("#paused")).to_be_visible()
                page.click("#btn-pause")
                page.keyboard.press("0")
                page.keyboard.press("Enter")
                page.wait_for_url(re.compile(r"/study/\d+/1"))
                page.goto(base + "/study")
                page.click(f"a[href='/study/{sid}/resume']")
                page.wait_for_url(re.compile(rf"/study/{sid}/1"))
                page.keyboard.press("2")
                page.keyboard.press("Enter")
                page.wait_for_url(re.compile(rf"/study/{sid}/2"))
                page.keyboard.press("u")
                page.keyboard.press("Enter")
                page.wait_for_url(re.compile(r"/study-results/\d+"))
                med = page.inner_text("#r-median")
                with page.expect_download() as dl:
                    page.click("text=Download CSV")
                rows = dl.value.path().read_text().strip().splitlines()
                assert len(rows) == 4, rows
                return f"pause, resume after 1 of 3, results (median {med}), CSV with 3 answers"
            S("Research: session, pause, resume, results, CSV (R1/R2/R3)", study)

            def practice():
                page.goto(base + "/study")
                before = page.locator("#sessions tbody tr").count()
                page.click("#btn-practice")
                page.wait_for_url(re.compile(r"/study/\d+/0"))
                assert "practice" in page.inner_text(".rv__bar")
                page.keyboard.press("1")
                page.keyboard.press("Enter")
                page.wait_for_url(re.compile(r"/study/\d+/1"))
                page.goto(base + "/study")
                after = page.locator("#sessions tbody tr").count()
                assert after == before, "practice session should not be listed"
                return "5-image practice runs; answers not saved or listed"
            S("Research: practice session", practice)

            def phone_widths():
                p = phone.new_page()
                tok = page.request.get(base + "/api/today").json()["patients"][0]["eyes"][0]["token"]
                wide = {}
                for u in ["/", "/about", "/screen", "/review", "/referrals", "/outbox", "/dashboard", "/evidence", "/study",
                          f"/s/{tok}", f"/s/{tok}?lang=ta", f"/slip/{tok}", "/s/nope"]:
                    p.goto(base + u)
                    p.wait_for_timeout(300)
                    w = p.evaluate("document.documentElement.scrollWidth")
                    if w > 392:
                        wide[u] = w
                p.close()
                assert not wide, f"wider than a 390px phone: {wide}"
                return "13 pages fit a 390 px phone with no sideways scroll"
            S("Every page fits a phone", phone_widths)

            browser.close()
    finally:
        server.terminate()
        server2.terminate()

    width = max(len(n) for n, _, _ in results)
    print()
    for n, st_, d in results:
        print(f"{st_:4}  {n:<{width}}  {d}")
    fails = sum(s == "FAIL" for _, s, _ in results)
    print(f"\n{len(results) - fails} passed, {fails} failed; backend {a.backend}")
    if console_errors:
        print(f"\nbrowser errors ({len(console_errors)}):")
        for e in dict.fromkeys(console_errors):
            print("  ", e)
    print(f"server logs and failure screenshots: {work}")
    sys.exit(1 if fails or console_errors else 0)


if __name__ == "__main__":
    main()
