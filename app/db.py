"""SQLite storage for cases, reviews, referrals and the reader study."""
from __future__ import annotations

import json
import secrets
import sqlite3
import time
from pathlib import Path

DB = Path(__file__).resolve().parent / "drishti.db"

SCHEMA = """
CREATE TABLE IF NOT EXISTS cases(
  id INTEGER PRIMARY KEY, token TEXT UNIQUE, created REAL, site TEXT, patient TEXT, eye TEXT,
  attempt INT, outcome TEXT, urgent INT, p_ref REAL, grade_set TEXT, audit INT DEFAULT 0,
  final TEXT, lang TEXT, study INT DEFAULT 0, result_json TEXT);
CREATE TABLE IF NOT EXISTS reviews(
  id INTEGER PRIMARY KEY, case_id INT, created REAL, reviewer TEXT, grade INT, dme INT,
  ungradable INT, ms INT, ai_outcome TEXT, final TEXT);
CREATE TABLE IF NOT EXISTS referrals(
  id INTEGER PRIMARY KEY, case_id INT, token TEXT, state TEXT, created REAL, updated REAL,
  urgent INT, contact_tier TEXT, appointment TEXT, history_json TEXT);
CREATE TABLE IF NOT EXISTS outbox(
  id INTEGER PRIMARY KEY, created REAL, referral_id INT, channel TEXT, recipient TEXT, text TEXT);
CREATE TABLE IF NOT EXISTS study_sessions(
  id INTEGER PRIMARY KEY, reader TEXT, created REAL, items_json TEXT);
CREATE TABLE IF NOT EXISTS study_answers(
  id INTEGER PRIMARY KEY, session_id INT, image_id TEXT, ref_grade INT, ref_dme INT,
  grade INT, dme INT, ungradable INT, ms INT, ai_p_ref REAL, ai_outcome TEXT);
"""

REFERRAL_STATES = ["flagged", "explained", "booked", "reminded", "attended", "missed", "escalated", "closed"]
NEXT = {
    "flagged": ["explained"],
    "explained": ["booked"],
    "booked": ["reminded", "attended", "missed"],
    "reminded": ["attended", "missed"],
    "missed": ["escalated"],
    "escalated": ["booked", "closed"],
    "attended": ["closed"],
    "closed": [],
}


def conn() -> sqlite3.Connection:
    c = sqlite3.connect(DB, timeout=10)
    c.row_factory = sqlite3.Row
    return c


def init():
    with conn() as c:
        c.executescript(SCHEMA)


def new_token() -> str:
    return secrets.token_urlsafe(12)


def insert_case(result: dict, site, patient, eye, attempt, lang, audit=False, study=False) -> tuple[int, str]:
    tok = new_token()
    with conn() as c:
        cur = c.execute(
            "INSERT INTO cases(token, created, site, patient, eye, attempt, outcome, urgent, p_ref, grade_set, audit, lang, study, result_json)"
            " VALUES(?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
            (tok, time.time(), site, patient, eye, attempt, result.get("outcome"), int(bool(result.get("urgent"))),
             (result.get("grade") or {}).get("p_referable"), json.dumps(result.get("grade_set")), int(audit), lang, int(study),
             json.dumps(result)))
        return cur.lastrowid, tok


def get_case(case_id: int):
    with conn() as c:
        r = c.execute("SELECT * FROM cases WHERE id=?", (case_id,)).fetchone()
    return r


def get_case_by_token(tok: str):
    with conn() as c:
        return c.execute("SELECT * FROM cases WHERE token=?", (tok,)).fetchone()


def review_queue():
    with conn() as c:
        return c.execute(
            "SELECT * FROM cases WHERE (outcome='review' OR audit=1) AND study=0 AND final IS NULL "
            "ORDER BY urgent DESC, created ASC").fetchall()


def add_review(case_id, reviewer, grade, dme, ungradable, ms):
    case = get_case(case_id)
    final = "refer" if (ungradable or grade >= 2 or dme) else "recall"
    with conn() as c:
        c.execute("INSERT INTO reviews(case_id, created, reviewer, grade, dme, ungradable, ms, ai_outcome, final) VALUES(?,?,?,?,?,?,?,?,?)",
                  (case_id, time.time(), reviewer, grade, dme, ungradable, ms, case["outcome"], final))
        c.execute("UPDATE cases SET final=? WHERE id=?", (final, case_id))
    if final == "refer":
        create_referral(case_id, urgent=bool(grade >= 3 or dme))
    return final


def create_referral(case_id, urgent=False) -> int:
    with conn() as c:
        row = c.execute("SELECT id FROM referrals WHERE case_id=?", (case_id,)).fetchone()
        if row:
            return row["id"]
        case = get_case(case_id)
        tok = case["token"] if case else new_token()
        hist = [{"state": "flagged", "t": time.time()}]
        cur = c.execute("INSERT INTO referrals(case_id, token, state, created, updated, urgent, contact_tier, history_json) VALUES(?,?,?,?,?,?,?,?)",
                  (case_id, tok, "flagged", time.time(), time.time(), int(urgent), "patient", json.dumps(hist)))
        return cur.lastrowid


def book_referral_direct(case_id: int, facility: str, specialty: str, appointment_date: str, transit: bool = False, escort: bool = False, notes: str = ""):
    with conn() as c:
        ref = c.execute("SELECT * FROM referrals WHERE case_id=?", (case_id,)).fetchone()
        if ref is None:
            case = get_case(case_id)
            urgent = bool(case["urgent"]) if case else False
            ref_id = create_referral(case_id, urgent=urgent)
            ref = c.execute("SELECT * FROM referrals WHERE id=?", (ref_id,)).fetchone()
        else:
            ref_id = ref["id"]

        hist = json.loads(ref["history_json"])
        booking_entry = {
            "state": "booked",
            "t": time.time(),
            "facility": facility,
            "specialty": specialty,
            "appointment": appointment_date,
            "transit": transit,
            "escort": escort,
            "notes": notes
        }
        hist.append(booking_entry)

        c.execute("UPDATE referrals SET state='booked', updated=?, history_json=?, appointment=? WHERE id=?",
                  (time.time(), json.dumps(hist), appointment_date, ref_id))

        sms_text = (f"DrishtiTrust Referral Booked: Appointment confirmed at {facility} ({specialty}) on {appointment_date}. "
                    f"Transit: {'Reserved (Bus Route 4B)' if transit else 'Self'}. Slip token: {ref['token']}. Bring your printed slip.")
        c.execute("INSERT INTO outbox(created, referral_id, channel, recipient, text) VALUES(?,?,?,?,?)",
                  (time.time(), ref_id, "SMS", "patient", sms_text))

        return {
            "referral_id": ref_id,
            "token": ref["token"],
            "state": "booked",
            "facility": facility,
            "specialty": specialty,
            "appointment": appointment_date,
            "transit": transit,
            "sms_sent": True
        }


def advance_referral(ref_id: int, to_state: str, note: str = "", appointment: str | None = None):
    with conn() as c:
        r = c.execute("SELECT * FROM referrals WHERE id=?", (ref_id,)).fetchone()
        if r is None or to_state not in NEXT[r["state"]]:
            raise ValueError(f"cannot go from {r['state'] if r else '?'} to {to_state}")
        hist = json.loads(r["history_json"])
        hist.append({"state": to_state, "t": time.time(), "note": note})
        tier = r["contact_tier"]
        if to_state == "escalated":  # three-tier contact plan: patient -> family -> ASHA worker
            tier = {"patient": "family", "family": "asha"}.get(tier, "asha")
        c.execute("UPDATE referrals SET state=?, updated=?, history_json=?, contact_tier=?, appointment=COALESCE(?, appointment) WHERE id=?",
                  (to_state, time.time(), json.dumps(hist), tier, appointment, ref_id))
        msg = {
            "explained": "Your eye photo showed signs that need an eye specialist. Your health worker has explained why.",
            "booked": f"Your eye hospital appointment is booked{(' for ' + appointment) if appointment else ''}. Bring this message and your slip.",
            "reminded": "Reminder: your eye hospital appointment is coming up. Treatment can protect your sight.",
            "escalated": "We could not reach the patient about a missed eye appointment. Please help them rebook.",
        }.get(to_state)
        if msg:
            c.execute("INSERT INTO outbox(created, referral_id, channel, recipient, text) VALUES(?,?,?,?,?)",
                      (time.time(), ref_id, "SMS (mock)", tier, msg))


def referrals():
    with conn() as c:
        return c.execute("SELECT r.*, c.site, c.patient, c.eye FROM referrals r JOIN cases c ON c.id=r.case_id "
                         "ORDER BY r.urgent DESC, r.created ASC").fetchall()


def dashboard():
    with conn() as c:
        out = {}
        out["outcomes"] = {r["outcome"]: r["n"] for r in c.execute("SELECT outcome, COUNT(*) n FROM cases WHERE study=0 GROUP BY outcome")}
        rv = c.execute("SELECT ms, ai_outcome, final FROM reviews").fetchall()
        out["reviews"] = len(rv)
        ms = sorted(r["ms"] for r in rv if r["ms"])
        out["review_seconds_median"] = round(ms[len(ms) // 2] / 1000, 1) if ms else None
        out["review_seconds_p90"] = round(ms[int(len(ms) * 0.9)] / 1000, 1) if ms else None
        out["referral_states"] = {r["state"]: r["n"] for r in c.execute("SELECT state, COUNT(*) n FROM referrals GROUP BY state")}
        out["audits"] = c.execute("SELECT COUNT(*) FROM cases WHERE audit=1").fetchone()[0]
        out["outbox"] = [dict(r) for r in c.execute("SELECT * FROM outbox ORDER BY id DESC LIMIT 12")]
        sa = c.execute("SELECT * FROM study_answers").fetchall()
        out["study_answers"] = len(sa)
    return out
