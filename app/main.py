"""DrishtiTrust web app: field screening, 30-second tele-review, referral
tracking, and a reader-study mode for timing real ophthalmologists.

    python -m uvicorn app.main:app --port 8026
"""
from __future__ import annotations

import csv
import json
import os
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.concurrency import run_in_threadpool
from fastapi.templating import Jinja2Templates

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))

from app import db  # noqa: E402
from drishti.config import ICDR_NAMES  # noqa: E402
from drishti.report.messages import referral_cfg, worker_message  # noqa: E402

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
tpl.env.globals.update(ICDR=ICDR_NAMES)
tpl.env.filters["fromjson"] = json.loads
tpl.env.filters["ago"] = lambda t: f"{int((time.time() - t) // 60)} min ago" if time.time() - t < 3600 else f"{(time.time() - t) / 3600:.1f} h ago"

_screener = None
AUDIT_FRACTION = float(os.environ.get("DRISHTI_AUDIT_FRACTION", 0.02))  # share of auto-recalls sent for review
STUDY_SET = ROOT / "study_set.csv"


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


# ------------------------------------------------------------------ landing & portals
@app.get("/", response_class=HTMLResponse)
def landing(request: Request):
    return tpl.TemplateResponse(request, "landing.html", {})


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


@app.get("/portal/screener", response_class=HTMLResponse)
@app.get("/screen", response_class=HTMLResponse)
def portal_screener(request: Request):
    s = screener()
    from drishti.quality.gate import load_cfg
    since = _today_start()
    with db.conn() as c:
        today = [dict(r) for r in c.execute("SELECT id, token, created, site, patient, eye, outcome, final, urgent, p_ref "
                                            "FROM cases WHERE study=0 AND created>=? ORDER BY id DESC LIMIT 40", (since,))]
    refs = [dict(r) for r in db.referrals()][-12:][::-1]
    d = db.dashboard()
    return tpl.TemplateResponse(request, "portal_screener.html", {
        "missing": s.missing, "backend": getattr(s, "backend", "python"),
        "samples": [{"file": f, "label": _sample_label(f)} for f in _samples()],
        "qcfg": load_cfg(), "cfg": referral_cfg(), "today": today, "refs": refs,
        "queue_n": len(db.review_queue()), "review_median_s": d["review_seconds_median"], "reviews_n": d["reviews"]})


@app.get("/portal/doctor", response_class=HTMLResponse)
def portal_doctor(request: Request):
    return tpl.TemplateResponse(request, "portal_doctor.html", {"cases": [dict(c) for c in db.review_queue()], "cfg": referral_cfg()})


@app.get("/api/case/{case_id}")
def api_case(case_id: int, lang: str | None = None, full: int = 0):
    """Case status for the portals: the outcome, the specialist's decision once
    made, the patient message in the chosen language and, with full=1, the
    images and model outputs the review screen needs."""
    c = db.get_case(case_id)
    if c is None:
        raise HTTPException(404)
    res = json.loads(c["result_json"])
    lang = lang or c["lang"]
    q = [x["id"] for x in db.review_queue()]
    with db.conn() as k:
        rv = k.execute("SELECT reviewer, grade, dme, ungradable, ms, final FROM reviews WHERE case_id=? ORDER BY id DESC LIMIT 1",
                       (case_id,)).fetchone()
    out = {"id": case_id, "token": c["token"], "outcome": c["outcome"], "final": c["final"], "urgent": bool(c["urgent"]),
           "audit": bool(c["audit"]), "queue_position": (q.index(case_id) + 1) if case_id in q else None, "queue_n": len(q),
           "review": dict(rv) if rv else None,
           "message": worker_message({**res, "outcome": c["final"] or res["outcome"]}, lang)}
    if full:
        out.update({k: res.get(k) for k in ("images", "grade", "grade_set", "rules", "reasons", "quality", "policy", "backend")})
        out.update({"site": c["site"], "patient": c["patient"], "eye": c["eye"], "created": c["created"]})
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
    return {"case_id": case_id, "token": tok, "outcome": res["outcome"], "referral_id": ref_id, "audit": audit, "result": res}


@app.post("/api/referral/book")
async def api_book_referral(request: Request):
    b = await request.json()
    case_id = int(b.get("case_id", 0))
    facility = b.get("facility", "District Eye Hospital (Ananthapur)")
    specialty = b.get("specialty", "Vitreo-Retina & Laser Clinic")
    appointment_date = b.get("appointment_date", "")
    transit = bool(b.get("transit", False))
    escort = bool(b.get("escort", False))
    notes = b.get("notes", "")
    res = db.book_referral_direct(case_id, facility, specialty, appointment_date, transit=transit, escort=escort, notes=notes)
    return {"ok": True, "booking": res, "slip_url": f"/slip/{res['token']}"}


@app.get("/case/{case_id}", response_class=HTMLResponse)
def case_view(request: Request, case_id: int, lang: str | None = None):
    c = db.get_case(case_id)
    if c is None:
        raise HTTPException(404)
    res = json.loads(c["result_json"])
    lang = lang or c["lang"]
    res["message"] = worker_message(res, lang)
    return tpl.TemplateResponse(request, "result.html", {"c": c, "r": res, "lang": lang, "cfg": referral_cfg()})


# ----------------------------------------------------------------- review
@app.get("/review", response_class=HTMLResponse)
def review_list(request: Request):
    return tpl.TemplateResponse(request, "review_list.html", {"cases": db.review_queue()})


@app.get("/review/{case_id}", response_class=HTMLResponse)
def review_case(request: Request, case_id: int):
    c = db.get_case(case_id)
    if c is None:
        raise HTTPException(404)
    q = [x["id"] for x in db.review_queue()]
    nxt = next((i for i in q if i != case_id), None)
    return tpl.TemplateResponse(request, "review.html", {"c": c, "r": json.loads(c["result_json"]), "next_id": nxt, "study": None})


@app.post("/api/review/{case_id}")
async def api_review(case_id: int, request: Request):
    b = await request.json()
    final = db.add_review(case_id, b.get("reviewer", "reviewer"), int(b["grade"]), int(b.get("dme", 0)), int(b.get("ungradable", 0)), int(b.get("ms", 0)))
    return {"final": final}


# -------------------------------------------------------------- referrals
@app.get("/referrals", response_class=HTMLResponse)
def referral_list(request: Request):
    return tpl.TemplateResponse(request, "referrals.html", {"refs": db.referrals(), "next": db.NEXT})


@app.post("/api/referral/{ref_id}")
async def api_referral(ref_id: int, request: Request):
    b = await request.json()
    try:
        db.advance_referral(ref_id, b["to"], b.get("note", ""), b.get("appointment"))
    except ValueError as e:
        raise HTTPException(400, str(e))
    return {"ok": True}


@app.get("/slip/{token}", response_class=HTMLResponse)
def slip(request: Request, token: str):
    c = db.get_case_by_token(token)
    if c is None:
        raise HTTPException(404)
    res = json.loads(c["result_json"])
    return tpl.TemplateResponse(request, "slip.html", {"c": c, "r": res, "cfg": referral_cfg(), "base": str(request.base_url)})


@app.get("/s/{token}", response_class=HTMLResponse)
def patient_status(request: Request, token: str, lang: str = "en"):
    """No-login status page behind an unguessable link (sent by SMS / printed)."""
    c = db.get_case_by_token(token)
    if c is None:
        raise HTTPException(404)
    res = json.loads(c["result_json"])
    res["message"] = worker_message({**res, "outcome": c["final"] or res["outcome"]}, lang)
    with db.conn() as k:
        ref = k.execute("SELECT * FROM referrals WHERE case_id=?", (c["id"],)).fetchone()
    return tpl.TemplateResponse(request, "status.html", {"c": c, "r": res, "ref": ref, "lang": lang})


# -------------------------------------------------------------- dashboard
@app.get("/dashboard", response_class=HTMLResponse)
def dashboard(request: Request):
    d = db.dashboard()
    results = {}
    rdir = ROOT.parent / "results"
    for f in sorted(rdir.glob("eval_*.json")):
        results[f.stem] = json.loads(f.read_text())
    return tpl.TemplateResponse(request, "dashboard.html", {"d": d, "results": results})


# ------------------------------------------------------------ reader study
def _study_items():
    if not STUDY_SET.exists():
        return []
    with open(STUDY_SET, newline="") as f:
        return list(csv.DictReader(f))


@app.get("/study", response_class=HTMLResponse)
def study_home(request: Request):
    with db.conn() as c:
        sessions = c.execute("SELECT s.*, (SELECT COUNT(*) FROM study_answers a WHERE a.session_id=s.id) n FROM study_sessions s ORDER BY id DESC").fetchall()
    return tpl.TemplateResponse(request, "study.html", {"items": _study_items(), "sessions": sessions})


@app.post("/study/start")
async def study_start(reader: str = Form(...), n: int = Form(40)):
    items = _study_items()
    random.shuffle(items)
    items = items[:n]
    with db.conn() as c:
        cur = c.execute("INSERT INTO study_sessions(reader, created, items_json) VALUES(?,?,?)", (reader, time.time(), json.dumps(items)))
    return RedirectResponse(f"/study/{cur.lastrowid}/0", status_code=303)


@app.get("/study/{sid}/{k}", response_class=HTMLResponse)
def study_case(request: Request, sid: int, k: int):
    with db.conn() as c:
        s = c.execute("SELECT * FROM study_sessions WHERE id=?", (sid,)).fetchone()
    items = json.loads(s["items_json"])
    if k >= len(items):
        return RedirectResponse(f"/study-results/{sid}", status_code=303)
    it = items[k]
    res = screener().screen(cv2.cvtColor(cv2.imdecode(np.fromfile(it["path"], np.uint8), cv2.IMREAD_COLOR), cv2.COLOR_BGR2RGB))
    fake = {"id": 0, "site": "study", "patient": f"{k + 1}/{len(items)}", "eye": "", "created": time.time(), "urgent": 0}
    return tpl.TemplateResponse(request, "review.html", {"c": fake, "r": res, "next_id": None,
                                                         "study": {"sid": sid, "k": k, "n": len(items), "image_id": it["image_id"]}})


@app.post("/api/study/{sid}/{k}")
async def study_answer(sid: int, k: int, request: Request):
    b = await request.json()
    with db.conn() as c:
        s = c.execute("SELECT * FROM study_sessions WHERE id=?", (sid,)).fetchone()
        it = json.loads(s["items_json"])[k]
        c.execute("INSERT INTO study_answers(session_id, image_id, ref_grade, ref_dme, grade, dme, ungradable, ms, ai_p_ref, ai_outcome) VALUES(?,?,?,?,?,?,?,?,?,?)",
                  (sid, it["image_id"], int(it["grade"]), int(it.get("dme") or 0), int(b["grade"]), int(b.get("dme", 0)), int(b.get("ungradable", 0)),
                   int(b.get("ms", 0)), b.get("ai_p_ref"), b.get("ai_outcome")))
    return {"next": f"/study/{sid}/{k + 1}"}


@app.get("/study-results/{sid}", response_class=HTMLResponse)
def study_results(request: Request, sid: int):
    from drishti.eval.metrics import binary_report, qwk
    with db.conn() as c:
        rows = [dict(r) for r in c.execute("SELECT * FROM study_answers WHERE session_id=?", (sid,))]
        s = c.execute("SELECT * FROM study_sessions WHERE id=?", (sid,)).fetchone()
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
    return tpl.TemplateResponse(request, "study_results.html", {"s": s, "out": out, "rows": rows})
