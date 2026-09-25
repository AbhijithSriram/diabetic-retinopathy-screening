"""DrishtiTrust web app: field screening, 30-second tele-review, referral
tracking, and a reader-study mode for timing real ophthalmologists.

    python -m uvicorn app.main:app --port 8026
"""
from __future__ import annotations

import csv
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
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
    yield


app = FastAPI(title="DrishtiTrust", lifespan=lifespan)
app.mount("/static", StaticFiles(directory=ROOT / "static"), name="static")
tpl = Jinja2Templates(directory=ROOT / "templates")
tpl.env.globals.update(ICDR=ICDR_NAMES)
tpl.env.filters["fromjson"] = json.loads
tpl.env.filters["ago"] = lambda t: f"{int((time.time() - t) // 60)} min ago" if time.time() - t < 3600 else f"{(time.time() - t) / 3600:.1f} h ago"

_screener = None
AUDIT_FRACTION = 0.02
STUDY_SET = ROOT / "study_set.csv"


def screener():
    global _screener
    if _screener is None:
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


@app.get("/portal/screener", response_class=HTMLResponse)
@app.get("/screen", response_class=HTMLResponse)
def portal_screener(request: Request):
    s = screener()
    return tpl.TemplateResponse(request, "portal_screener.html", {"missing": s.missing, "samples": _samples()})


@app.get("/portal/doctor", response_class=HTMLResponse)
def portal_doctor(request: Request):
    return tpl.TemplateResponse(request, "portal_doctor.html", {"cases": db.review_queue()})


def _samples():
    p = ROOT / "samples"
    return sorted(x.name for x in p.glob("*.jpg")) if p.exists() else []


@app.post("/api/screen")
async def api_screen(file: UploadFile | None = File(None), sample: str = Form(""), site: str = Form("PHC-1"), patient: str = Form(""),
                     eye: str = Form("right"), attempt: int = Form(1), lang: str = Form("en")):
    if sample:
        data = (ROOT / "samples" / Path(sample).name).read_bytes()
    elif file is not None:
        data = await file.read()
    else:
        raise HTTPException(400, "no image")
    rgb = _decode(data)
    res = screener().screen(rgb, attempt=attempt, site=site, lang=lang)
    audit = res.get("outcome") == "recall" and random.random() < AUDIT_FRACTION
    case_id, tok = db.insert_case(res, site, patient, eye, attempt, lang, audit=audit)
    ref_id = None
    if res.get("outcome") == "refer":
        ref_id = db.create_referral(case_id, urgent=bool(res.get("urgent")))
    return {"case_id": case_id, "token": tok, "outcome": res["outcome"], "referral_id": ref_id, "result": res}


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
