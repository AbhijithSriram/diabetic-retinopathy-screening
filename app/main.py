"""DrishtiTrust web app: field screening, 30-second tele-review, referral
tracking, a programme dashboard with the validation evidence, and a
reader-study mode for timing real ophthalmologists.

    python -m uvicorn app.main:app --port 8026

Pages are server-rendered Jinja2 (templates/) on one stylesheet
(static/drishti.css, from the Claude Design handoff); live screens fetch the
JSON endpoints below with small vanilla scripts.
"""
from __future__ import annotations

import csv
import io
import json
import os
import random
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.concurrency import run_in_threadpool
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from app import db, present  # noqa: E402
from drishti.config import ICDR_NAMES  # noqa: E402
from drishti.report.messages import referral_cfg  # noqa: E402

from contextlib import asynccontextmanager  # noqa: E402


@asynccontextmanager
async def lifespan(_app):
    db.init()
    s = screener()
    if hasattr(s, "warm_up"):
        s.warm_up()  # MATLAB starts and imports its networks while the pages already serve
    yield


app = FastAPI(title="DrishtiTrust", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
if (ROOT / "samples").exists():
    app.mount("/samples", StaticFiles(directory=ROOT / "samples"), name="samples")
tpl = Jinja2Templates(directory=ROOT / "templates")
tpl.env.globals.update(ICDR=ICDR_NAMES, DISCLAIMER=present.P["en"]["disclaimer"])
tpl.env.filters["fromjson"] = json.loads
tpl.env.filters["hm"] = lambda t: time.strftime("%H:%M", time.localtime(t)) if t else ""
tpl.env.filters["dmy"] = lambda t: time.strftime("%d %b %Y", time.localtime(t)).lstrip("0") if t else ""

_screener = None
AUDIT_FRACTION = float(os.environ.get("DRISHTI_AUDIT_FRACTION", 0.02))  # share of auto-recalls sent for review
STUDY_SET = ROOT / "study_set.csv"
SITES = ["PHC-1", "PHC-2", "CHC-Central", "Mobile-Van"]
FACILITIES = [{"name": "District Eye Hospital (example)", "specialty": "Vitreo-Retina & Laser Clinic"},
              {"name": "Regional Eye Institute (example)", "specialty": "Retina Services"},
              {"name": "CHC Ophthalmic Unit (example)", "specialty": "Cataract & General Ophthalmology"}]


def _torch_checkpoints() -> bool:
    """Whether the PyTorch grader named in configs/models.json is on disk
    (checked without importing torch)."""
    from drishti.config import RUNS
    cfg = ROOT.parent / "configs" / "models.json"
    name = json.loads(cfg.read_text()).get("grader") if cfg.exists() else None
    if not name:
        return False
    args = RUNS / name / "args.json"
    members = json.loads(args.read_text()).get("ensemble_of", [name]) if args.exists() else [name]
    return all((RUNS / m / "best.pt").exists() for m in members)


def screener():
    """DRISHTI_BACKEND=python | matlab | auto (default). auto serves the
    PyTorch checkpoints when they are on disk, else the MATLAB export."""
    global _screener
    if _screener is None:
        mode = os.environ.get("DRISHTI_BACKEND", "auto").lower()
        from drishti import matlab_backend
        if mode == "matlab" or (mode == "auto" and not _torch_checkpoints() and matlab_backend.available()):
            _screener = matlab_backend.MatlabScreener()
        else:
            from drishti.pipeline import Screener
            _screener = Screener()
    return _screener


def _decode(data: bytes) -> np.ndarray:
    img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    if img is None:
        raise HTTPException(400, "not an image")
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def _samples():
    p = ROOT / "samples"
    return sorted(x.name for x in p.glob("*.jpg")) if p.exists() else []


def _sample_label(name: str) -> str:
    """Reference label of a bundled public-dataset photo, from its file name."""
    stem = Path(name).stem
    if stem.startswith("idrid_") and "_grade" in stem:
        g = int(stem.split("_grade")[1][0])
        return f"IDRiD · graded {g} ({ICDR_NAMES[g]})"
    if stem.startswith("hrf_"):
        return "HRF · diabetic retinopathy" if stem.endswith("_dr") else "HRF · healthy" if stem.endswith("_h") else "HRF"
    return stem


def _today_start() -> float:
    t = time.localtime()
    return time.mktime((t.tm_year, t.tm_mon, t.tm_mday, 0, 0, 0, 0, 0, -1))


def _grading_off(s) -> bool:
    return "grader" in s.missing or "triage policy" in s.missing


def _case_payload(k, c, lang: str | None = None, full: bool = False) -> dict:
    """Everything the screens show about one case (one eye)."""
    res = json.loads(c["result_json"])
    rv = present.latest_review(k, c["id"])
    kind = present.case_kind(c, res, rv)
    ref = present.referral_of(k, c["id"])
    lang = lang or c["lang"]
    out = {"id": c["id"], "token": c["token"], "site": c["site"], "patient": c["patient"], "eye": c["eye"],
           "created": c["created"], "attempt": c["attempt"], "outcome": c["outcome"], "final": c["final"],
           "urgent": bool(c["urgent"]), "audit": bool(c["audit"]), "reason": res.get("reason"), "lang": c["lang"],
           "view": present.view(kind, reviewed=bool(rv)), "review": rv,
           "referral": {x: ref[x] for x in ("id", "state", "appointment", "facility", "specialty", "transit", "escort", "urgent")} if ref else None,
           "message": present.patient_message(res, c, kind, lang),
           "grade": res.get("grade"), "grade_set": res.get("grade_set"), "reasons": res.get("reasons") or [],
           "quality": res.get("quality"), "missing_models": res.get("missing_models") or []}
    if full:
        out.update({x: res.get(x) for x in ("images", "rules", "policy", "backend")})
        out["grade_set_idx"] = present.grade_set_idx(res.get("grade_set"))
    return out


# ------------------------------------------------------------------ entry
@app.get("/", response_class=HTMLResponse)
def entry(request: Request):
    return tpl.TemplateResponse(request, "entry.html", {})


@app.get("/about", response_class=HTMLResponse)
def about(request: Request):
    return tpl.TemplateResponse(request, "about.html", {"cfg": referral_cfg()})


for _old, _new in (("/portal/screener", "/screen"), ("/portal/doctor", "/review"), ("/login", "/")):
    app.add_api_route(_old, lambda _n=_new: RedirectResponse(_n), methods=["GET"], include_in_schema=False)


@app.get("/api/status")
def api_status():
    s = screener()
    ready = s._ready.is_set() if hasattr(s, "_ready") else True
    return {"backend": getattr(s, "backend", "python"), "ready": ready, "status": getattr(s, "status", "ready"),
            "missing": s.missing, "grading_off": _grading_off(s)}


# ------------------------------------------------------------------ health worker
@app.get("/screen", response_class=HTMLResponse)
def health_worker(request: Request, case: int | None = None):
    s = screener()
    from drishti.quality.gate import load_cfg
    return tpl.TemplateResponse(request, "health_worker.html", {
        "cfg": referral_cfg(), "qcfg": load_cfg(), "sites": SITES, "facilities": FACILITIES,
        "samples": [{"file": f, "label": _sample_label(f)} for f in _samples()],
        "backend": getattr(s, "backend", "python"), "grading_off": _grading_off(s), "open_case": case})


@app.get("/case/{case_id}")
def case_redirect(case_id: int):
    return RedirectResponse(f"/screen?case={case_id}")


@app.get("/api/today")
def api_today(site: str | None = None):
    """Today's screenings grouped by patient, latest case per eye."""
    q = "SELECT * FROM cases WHERE study=0 AND created>=?" + (" AND site=?" if site else "") + " ORDER BY id"
    args = (_today_start(), site) if site else (_today_start(),)
    with db.conn() as k:
        rows = k.execute(q, args).fetchall()
        patients: dict[str, dict] = {}
        for c in rows:
            key = c["patient"] or f"#{c['id']}"
            p = patients.setdefault(key, {"patient": key, "site": c["site"], "first": c["created"], "last": c["created"], "eyes": {}})
            p["last"] = c["created"]
            p["eyes"][c["eye"]] = _case_payload(k, c)
    out = sorted(patients.values(), key=lambda p: -p["last"])
    for p in out:
        p["eyes"] = [p["eyes"][e] for e in ("right", "left") if e in p["eyes"]]
    return {"patients": out, "eyes": sum(len(p["eyes"]) for p in out)}


@app.get("/api/case/{case_id}")
def api_case(case_id: int, lang: str | None = None, full: int = 0):
    c = db.get_case(case_id)
    if c is None:
        raise HTTPException(404)
    q = [x["id"] for x in db.review_queue()]
    with db.conn() as k:
        out = _case_payload(k, c, lang, bool(full))
    out.update({"queue_position": (q.index(case_id) + 1) if case_id in q else None, "queue_n": len(q)})
    return out


@app.post("/api/screen")
async def api_screen(file: UploadFile | None = File(None), sample: str = Form(""), site: str = Form("PHC-1"), patient: str = Form(""),
                     eye: str = Form("right"), attempt: int = Form(1), lang: str = Form("en"), simulate: str = Form("")):
    if sample:
        p = ROOT / "samples" / Path(sample).name
        if not p.exists():
            raise HTTPException(404, f"no sample {Path(sample).name}")
        data = p.read_bytes()
    elif file is not None:
        data = await file.read()
    else:
        raise HTTPException(400, "no image")
    rgb = _decode(data)
    if simulate == "blur":  # demo of the recapture path: an out-of-focus capture of the same eye
        rgb = cv2.GaussianBlur(rgb, (0, 0), rgb.shape[1] / 120)
    res = await run_in_threadpool(screener().screen, rgb, attempt=attempt, site=site, lang=lang)
    audit = res.get("outcome") == "recall" and random.random() < AUDIT_FRACTION
    case_id, tok = db.insert_case(res, site, patient, eye, attempt, lang, audit=audit)
    ref_id = None
    if res.get("outcome") == "refer":
        ref_id = db.create_referral(case_id, urgent=bool(res.get("urgent")))
    with db.conn() as k:
        payload = _case_payload(k, db.get_case(case_id), lang, full=True)
    return {"case_id": case_id, "token": tok, "outcome": res["outcome"], "referral_id": ref_id, "audit": audit,
            "result": res, "case": payload}


@app.post("/api/referral/book")
async def api_book_referral(request: Request):
    b = await request.json()
    case_id = int(b.get("case_id", 0))
    if db.get_case(case_id) is None:
        raise HTTPException(404)
    res = db.book_referral_direct(case_id, b.get("facility") or FACILITIES[0]["name"], b.get("specialty") or FACILITIES[0]["specialty"],
                                  b.get("appointment_date", ""), transit=bool(b.get("transit")), escort=bool(b.get("escort")),
                                  notes=b.get("notes", ""))
    return {"ok": True, "booking": res, "slip_url": f"/slip/{res['token']}", "status_url": f"/s/{res['token']}"}


# ------------------------------------------------------------------ specialist
def _queue_payload():
    with db.conn() as k:
        cases = []
        for c in db.review_queue():
            res = json.loads(c["result_json"])
            why = ("Random audit of an automatic recall" if c["audit"] and c["outcome"] == "recall"
                   else (res.get("reasons") or ["Sent for review"])[0])
            cases.append({"id": c["id"], "site": c["site"], "patient": c["patient"], "eye": c["eye"], "created": c["created"],
                          "urgent": bool(c["urgent"]), "audit": bool(c["audit"]) and c["outcome"] == "recall", "why": why,
                          "most_likely": (res.get("grade") or {}).get("most_likely"), "p_ref": (res.get("grade") or {}).get("p_referable"),
                          "waiting": present.waiting(c["created"])})
        since = _today_start()
        ms = sorted(r["ms"] for r in k.execute("SELECT ms FROM reviews WHERE created>=? AND ms>0", (since,)))
    return {"cases": cases, "reviewed_today": len(ms), "median_today_s": round(statistics.median(ms) / 1000) if ms else None,
            "oldest_h": round(max((c["waiting"]["hours"] for c in cases), default=0)), "target_h": referral_cfg()["review_hours"]}


@app.get("/review", response_class=HTMLResponse)
def review_queue(request: Request):
    return tpl.TemplateResponse(request, "review_queue.html", {"q": _queue_payload()})


@app.get("/api/queue")
def api_queue():
    return _queue_payload()


@app.get("/review/{case_id}", response_class=HTMLResponse)
def review_case(request: Request, case_id: int):
    if db.get_case(case_id) is None:
        raise HTTPException(404)
    return tpl.TemplateResponse(request, "review.html", {"case_id": case_id, "study": None, "cfg": referral_cfg()})


@app.post("/api/review/{case_id}")
async def api_review(case_id: int, request: Request):
    b = await request.json()
    if db.get_case(case_id) is None:
        raise HTTPException(404)
    final = db.add_review(case_id, b.get("reviewer", "reviewer"), int(b["grade"]), int(b.get("dme", 0)), int(b.get("ungradable", 0)), int(b.get("ms", 0)))
    if final == "refer" and b.get("appointment_date"):
        db.book_referral_direct(case_id, b.get("facility") or FACILITIES[0]["name"], b.get("specialty") or FACILITIES[0]["specialty"],
                                b["appointment_date"], notes=f"Booked at tele-review by {b.get('reviewer', 'reviewer')}")
    return {"final": final}


# ------------------------------------------------------------------ coordinator
def _referrals_payload():
    cfg = referral_cfg()
    rows = []
    with db.conn() as k:
        for r in db.referrals():
            c = db.get_case(r["case_id"])
            res = json.loads(c["result_json"])
            kind = present.case_kind(c, res, present.latest_review(k, c["id"]))
            kind = kind if kind in ("urgent", "ungradable") else "refer"
            days = cfg["urgent_days"] if kind == "urgent" else cfg["routine_days"]
            hist = json.loads(r["history_json"])
            rows.append({"id": r["id"], "case_id": r["case_id"], "patient": r["patient"], "site": r["site"], "eye": r["eye"],
                         "kind": kind, "state": r["state"], "tier": r["contact_tier"], "appointment": r["appointment"],
                         "due": r["created"] + days * 86400, "created": r["created"], "token": r["token"],
                         "history": hist, "moves": db.NEXT[r["state"]],
                         "facility": next((h.get("facility") for h in reversed(hist) if h.get("facility")), None)})
    order = {"urgent": 0, "ungradable": 1, "refer": 1}
    rows.sort(key=lambda x: (x["state"] == "closed", order[x["kind"]], x["due"]))
    return {"referrals": rows, "now": time.time()}


@app.get("/referrals", response_class=HTMLResponse)
def referrals_page(request: Request):
    return tpl.TemplateResponse(request, "coordinator.html", {"data": _referrals_payload(), "facilities": FACILITIES})


@app.get("/api/referrals")
def api_referrals():
    return _referrals_payload()


@app.post("/api/referral/{ref_id}")
async def api_referral(ref_id: int, request: Request):
    b = await request.json()
    try:
        db.advance_referral(ref_id, b["to"], b.get("note", ""), b.get("appointment"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/outbox", response_class=HTMLResponse)
def outbox(request: Request):
    with db.conn() as k:
        rows = [dict(r) for r in k.execute("SELECT * FROM outbox ORDER BY id DESC LIMIT 200")]
    return tpl.TemplateResponse(request, "outbox.html", {"rows": rows})


# ------------------------------------------------------------------ patient
def _eye_card(k, c, lang):
    p = _case_payload(k, c, lang)
    lab = present.patient_labels(lang)
    p["plabel"] = lab[p["view"]["kind"]]
    return p


@app.get("/s/{token}", response_class=HTMLResponse)
def patient_status(request: Request, token: str, lang: str | None = None):
    """No-login status page behind an unguessable link (sent by SMS / printed)."""
    c = db.get_case_by_token(token)
    if c is None:
        return tpl.TemplateResponse(request, "patient_bad_link.html", {"L": present.P}, status_code=404)
    lang = lang if lang in present.P else c["lang"] if c["lang"] in present.P else "en"
    with db.conn() as k:
        main = _eye_card(k, c, lang)
        other = None
        if c["patient"]:  # the other eye of the same patient, same site, same day
            day0 = time.mktime(time.localtime(c["created"])[:3] + (0, 0, 0, 0, 0, -1))
            o = k.execute("SELECT * FROM cases WHERE patient=? AND site=? AND eye!=? AND study=0 AND created BETWEEN ? AND ? "
                          "ORDER BY id DESC LIMIT 1", (c["patient"], c["site"], c["eye"], day0, day0 + 86400)).fetchone()
            other = _eye_card(k, o, lang) if o else None
    cfg = referral_cfg()
    next_check = time.strftime("%B %Y", time.localtime(c["created"] + cfg["recall_months"] * 30.44 * 86400))
    return tpl.TemplateResponse(request, "patient.html", {"c": main, "other": other, "lang": lang, "L": present.patient_labels(lang),
                                                          "next_check": next_check, "token": token})


@app.get("/slip/{token}", response_class=HTMLResponse)
def slip(request: Request, token: str):
    c = db.get_case_by_token(token)
    if c is None:
        return tpl.TemplateResponse(request, "patient_bad_link.html", {"L": present.P}, status_code=404)
    import segno
    url = str(request.base_url).rstrip("/") + f"/s/{token}"
    with db.conn() as k:
        p = _case_payload(k, c, "en", full=True)
    p.pop("images", None)
    qr = segno.make(url, error="m").svg_inline(scale=3, border=0, dark="#000")
    return tpl.TemplateResponse(request, "slip.html", {"c": p, "line": present.slip_line(p["view"]["kind"]), "qr": qr, "url": url})


# ------------------------------------------------------------------ programme
def _dashboard_payload():
    with db.conn() as k:
        cases = k.execute("SELECT * FROM cases WHERE study=0").fetchall()
        counts = {"recall": 0, "review": 0, "refer": 0, "ungradable": 0}
        for c in cases:
            if c["outcome"] == "recapture":
                continue  # retake requests are not outcomes
            res = json.loads(c["result_json"])
            kind = "ungradable" if c["outcome"] == "refer" and res.get("reason") == "ungradable" else c["outcome"]
            counts[kind] = counts.get(kind, 0) + 1
        ms = sorted(r["ms"] for r in k.execute("SELECT ms FROM reviews WHERE ms>0"))
        funnel = {s: 0 for s in ("flagged", "explained", "booked", "attended", "closed", "missed", "escalated")}
        tiers = {"family": 0, "asha": 0}
        for r in k.execute("SELECT history_json, contact_tier FROM referrals"):
            seen = {h["state"] for h in json.loads(r["history_json"])}
            for s in funnel:
                funnel[s] += s in seen
            if "escalated" in seen and r["contact_tier"] in tiers:
                tiers[r["contact_tier"]] += 1
        audits = k.execute("SELECT COUNT(*) FROM cases WHERE audit=1").fetchone()[0]
    total = sum(counts.values())
    p90 = ms[min(len(ms) - 1, int(len(ms) * 0.9))] if ms else None
    return {"counts": counts, "total": total, "reviews": len(ms), "median_s": round(statistics.median(ms) / 1000) if ms else None,
            "p90_s": round(p90 / 1000) if p90 else None, "audits": audits, "funnel": funnel, "tiers": tiers,
            "updated": time.strftime("%H:%M")}


@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    return tpl.TemplateResponse(request, "dashboard.html", {"d": _dashboard_payload()})


def _evidence():
    """Validation and simulation figures, read from results/ so the page cannot
    drift from what the evaluation produced."""
    rdir = ROOT.parent / "results"
    ev = {}
    try:
        e = json.loads((rdir / "eval_ens_d3.json").read_text())
        m = e["messidor2_exam_level"]
        ev["m2"] = {"n": m["n_exams"], "sens": m["cnn_only"]["sensitivity"], "spec": m["cnn_only"]["specificity"],
                    "auc": m["auc_cnn"], "ref": m["reference"]}
        i = next(v for k, v in e["sets"].items() if k.startswith("IDRiD"))
        ev["idrid"] = {"n": 516, "sens": i["at_operating_point"]["sensitivity"], "spec": i["at_operating_point"]["specificity"],
                       "auc": i["auc_referable"]}
    except Exception:
        pass
    try:
        s = json.loads((rdir / "simulation" / "sim_summary.json").read_text())
        grid = [g for g in s["grid"] if g.get("online_fraction") == 0.6 and g["completion"] == 0.15 and g["reviewers"] == 2]
        ev["sim"] = {"all": next(g["reviews_per_year"] for g in grid if g["review_all"]),
                     "ai": next(g["reviews_per_year"] for g in grid if not g["review_all"]),
                     "need_all": s["min_reviewers_for_95pct_sla"]["all images reviewed | online 0.6"],
                     "need_ai": s["min_reviewers_for_95pct_sla"]["AI triage | online 0.6"],
                     "screenings": s["params"]["district"]["annual_screenings"], "sites": s["params"]["district"]["sites"],
                     "days": s["params"]["run"]["days"], "online": s["params"]["network"]["online_fraction"]}
    except Exception:
        pass
    return ev


@app.get("/evidence", response_class=HTMLResponse)
def evidence(request: Request):
    return tpl.TemplateResponse(request, "evidence.html", {"ev": _evidence()})


# ------------------------------------------------------------------ reader study
def _study_items():
    if not STUDY_SET.exists():
        return []
    with open(STUDY_SET, newline="") as f:
        return list(csv.DictReader(f))


def _session(sid: int):
    with db.conn() as c:
        s = c.execute("SELECT * FROM study_sessions WHERE id=?", (sid,)).fetchone()
    if s is None:
        raise HTTPException(404)
    return s


@app.get("/study", response_class=HTMLResponse)
def study_home(request: Request):
    with db.conn() as c:
        sessions = [dict(r) for r in c.execute(
            "SELECT s.*, (SELECT COUNT(*) FROM study_answers a WHERE a.session_id=s.id) n FROM study_sessions s "
            "WHERE COALESCE(s.practice, 0)=0 ORDER BY id DESC")]
    for s in sessions:
        s["total"] = len(json.loads(s["items_json"]))
    return tpl.TemplateResponse(request, "study.html", {"n_items": len(_study_items()), "sessions": sessions})


@app.post("/study/start")
async def study_start(reader: str = Form(...), n: int = Form(40), practice: int = Form(0)):
    items = _study_items()
    if not items:
        raise HTTPException(400, "no study set (app/study_set.csv)")
    random.shuffle(items)
    items = items[:5 if practice else max(1, n)]
    with db.conn() as c:
        cur = c.execute("INSERT INTO study_sessions(reader, created, items_json, practice) VALUES(?,?,?,?)",
                        (reader, time.time(), json.dumps(items), int(bool(practice))))
    return RedirectResponse(f"/study/{cur.lastrowid}/0", status_code=303)


@app.get("/study/{sid}/resume")
def study_resume(sid: int):
    with db.conn() as c:
        n = c.execute("SELECT COUNT(*) FROM study_answers WHERE session_id=?", (sid,)).fetchone()[0]
    return RedirectResponse(f"/study/{sid}/{n}", status_code=303)


@app.get("/study/{sid}/{k}", response_class=HTMLResponse)
def study_case(request: Request, sid: int, k: int):
    s = _session(sid)
    items = json.loads(s["items_json"])
    if k >= len(items):
        return RedirectResponse("/study" if s["practice"] else f"/study-results/{sid}", status_code=303)
    it = items[k]
    rgb = cv2.cvtColor(cv2.imdecode(np.fromfile(it["path"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB)
    res = screener().screen(rgb)
    case = {"id": None, "study": True, "images": res.get("images"), "grade": res.get("grade"), "grade_set": res.get("grade_set"),
            "grade_set_idx": present.grade_set_idx(res.get("grade_set")), "rules": res.get("rules"), "reasons": res.get("reasons") or [],
            "quality": res.get("quality"), "outcome": res.get("outcome"), "missing_models": res.get("missing_models") or []}
    return tpl.TemplateResponse(request, "review.html", {"case_id": None, "cfg": referral_cfg(), "case_json": case,
                                                         "study": {"sid": sid, "k": k, "n": len(items), "image_id": it["image_id"],
                                                                   "reader": s["reader"], "practice": bool(s["practice"])}})


@app.post("/api/study/{sid}/{k}")
async def study_answer(sid: int, k: int, request: Request):
    b = await request.json()
    s = _session(sid)
    if not s["practice"]:  # practice answers are not saved
        it = json.loads(s["items_json"])[k]
        with db.conn() as c:
            c.execute("INSERT INTO study_answers(session_id, image_id, ref_grade, ref_dme, grade, dme, ungradable, ms, ai_p_ref, ai_outcome) "
                      "VALUES(?,?,?,?,?,?,?,?,?,?)",
                      (sid, it["image_id"], int(it["grade"]), int(it.get("dme") or 0), int(b["grade"]), int(b.get("dme", 0)),
                       int(b.get("ungradable", 0)), int(b.get("ms", 0)), b.get("ai_p_ref"), b.get("ai_outcome")))
    return {"next": f"/study/{sid}/{k + 1}"}


def _study_results(sid: int):
    from drishti.eval.metrics import binary_report, qwk
    with db.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM study_answers WHERE session_id=? ORDER BY id", (sid,))]
    out = {"n": len(rows)}
    if rows:
        ms = sorted(r["ms"] for r in rows)
        out["median_s"] = round(ms[len(ms) // 2] / 1000, 1)
        out["p90_s"] = round(ms[min(len(ms) - 1, int(len(ms) * 0.9))] / 1000, 1)
        out["under_30s_pct"] = round(100 * sum(m < 30000 for m in ms) / len(ms), 1)
        g = [r for r in rows if not r["ungradable"]]
        if len(g) > 1:
            out["qwk_vs_reference"] = round(qwk([r["ref_grade"] for r in g], [r["grade"] for r in g]), 3)
            rep = binary_report([r["ref_grade"] >= 2 for r in g], [r["grade"] >= 2 for r in g])
            out["referable_sens"] = [round(v, 3) for v in rep["sensitivity"]]
            out["referable_spec"] = [round(v, 3) for v in rep["specificity"]]
    return out, rows


@app.get("/study-results/{sid}", response_class=HTMLResponse)
def study_results(request: Request, sid: int):
    s = _session(sid)
    out, rows = _study_results(sid)
    return tpl.TemplateResponse(request, "study_results.html", {"s": s, "out": out, "rows": rows})


@app.get("/study-results/{sid}/csv")
def study_results_csv(sid: int):
    _session(sid)
    _, rows = _study_results(sid)
    buf = io.StringIO()
    cols = ["image_id", "ref_grade", "grade", "ref_dme", "dme", "ungradable", "ms", "ai_outcome", "ai_p_ref"]
    w = csv.DictWriter(buf, fieldnames=cols, extrasaction="ignore")
    w.writeheader()
    w.writerows(rows)
    return Response(buf.getvalue(), media_type="text/csv", headers={"Content-Disposition": f'attachment; filename="study_{sid}.csv"'})
