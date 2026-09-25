"""DrishtiTrust Zero-Dependency Instant UI Preview Server.
Allows checking the web application in any browser without installing heavy ML/CUDA packages.

Usage:
    python preview.py
Then open:
    http://localhost:8026
"""
from __future__ import annotations

import http.server
import json
import os
import re
import socketserver
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

PORT = 8026
ROOT = Path(__file__).resolve().parent
APP_DIR = ROOT / "app"
STATIC_DIR = APP_DIR / "static"
TEMPLATES_DIR = APP_DIR / "templates"

ICDR_NAMES = ["No DR", "Mild NPDR", "Moderate NPDR", "Severe NPDR", "Proliferative DR"]

# Sample Fundus SVG Data URI fallback if no raw image
MOCK_PHOTO = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='512' height='512'>"
    "<circle cx='256' cy='256' r='240' fill='%236A180E'/>"
    "<circle cx='256' cy='256' r='236' fill='%23852012'/>"
    "<circle cx='180' cy='256' r='42' fill='%23DE8838'/>"
    "<circle cx='320' cy='256' r='14' fill='%23380B06'/>"
    "<path d='M180 256 Q 250 200 380 180' stroke='%23480B06' stroke-width='6' fill='none'/>"
    "<path d='M180 256 Q 260 320 400 340' stroke='%23480B06' stroke-width='5' fill='none'/>"
    "<text x='256' y='460' font-family='sans-serif' font-size='18' fill='%23EADFD3' text-anchor='middle'>Fundus Photography Simulation</text>"
    "</svg>"
)

MOCK_LESIONS = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='512' height='512'>"
    "<circle cx='256' cy='256' r='240' fill='%236A180E'/>"
    "<circle cx='180' cy='256' r='42' fill='%23DE8838' stroke='%23059669' stroke-width='4'/>"
    "<circle cx='320' cy='256' r='14' fill='%23380B06'/>"
    "<circle cx='320' cy='256' r='52' fill='none' stroke='%230284C7' stroke-width='3' stroke-dasharray='6 4'/>"
    "<circle cx='290' cy='230' r='5' fill='%23D97706'/>"
    "<circle cx='302' cy='240' r='6' fill='%23D97706'/>"
    "<circle cx='280' cy='290' r='7' fill='%23881337'/>"
    "<circle cx='350' cy='280' r='8' fill='%23881337'/>"
    "<circle cx='230' cy='180' r='4' fill='none' stroke='%23DC2626' stroke-width='3'/>"
    "<circle cx='245' cy='190' r='3' fill='none' stroke='%23DC2626' stroke-width='3'/>"
    "<rect x='340' y='210' width='16' height='12' rx='4' fill='%23F59E0B' opacity='0.85'/>"
    "<text x='256' y='460' font-family='sans-serif' font-size='18' fill='%23EADFD3' text-anchor='middle'>Annotated Lesion U-Net Overlay</text>"
    "</svg>"
)

MOCK_GRADCAM = (
    "data:image/svg+xml;utf8,"
    "<svg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 512 512' width='512' height='512'>"
    "<circle cx='256' cy='256' r='240' fill='%236A180E'/>"
    "<radialGradient id='gcam' cx='60%' cy='50%' r='35%'>"
    "<stop offset='0%' stop-color='%23DC2626' stop-opacity='0.85'/>"
    "<stop offset='45%' stop-color='%23EA580C' stop-opacity='0.65'/>"
    "<stop offset='75%' stop-color='%23F59E0B' stop-opacity='0.35'/>"
    "<stop offset='100%' stop-color='%230284C7' stop-opacity='0'/>"
    "</radialGradient>"
    "<circle cx='256' cy='256' r='236' fill='url(%23gcam)'/>"
    "<text x='256' y='460' font-family='sans-serif' font-size='18' fill='%23EADFD3' text-anchor='middle'>Grad-CAM Attention Heatmap</text>"
    "</svg>"
)

# In-memory cases & referrals
CASES = [
    {
        "id": 101, "token": "tok_p101_review", "created": time.time() - 1420,
        "site": "PHC-1", "patient": "P-2026-0042", "eye": "right", "attempt": 1,
        "outcome": "review", "urgent": 0, "p_ref": 0.845,
        "grade_set": ["Moderate NPDR", "Severe NPDR"],
        "audit": 0, "lang": "en"
    },
    {
        "id": 102, "token": "tok_p102_urgent", "created": time.time() - 3600,
        "site": "Camp-A", "patient": "P-2026-0089", "eye": "left", "attempt": 1,
        "outcome": "review", "urgent": 1, "p_ref": 0.962,
        "grade_set": ["Severe NPDR", "Proliferative DR"],
        "audit": 0, "lang": "hi"
    },
    {
        "id": 103, "token": "tok_p103_audit", "created": time.time() - 7200,
        "site": "PHC-2", "patient": "P-2026-0114", "eye": "right", "attempt": 1,
        "outcome": "recall", "urgent": 0, "p_ref": 0.082,
        "grade_set": ["No DR"],
        "audit": 1, "lang": "ta"
    },
]

REFERRALS = [
    {
        "id": 1, "case_id": 102, "token": "tok_p102_urgent", "state": "booked",
        "created": time.time() - 86400, "updated": time.time() - 3600,
        "urgent": 1, "contact_tier": "patient", "appointment": "2026-10-12",
        "history_json": json.dumps([
            {"state": "flagged", "t": time.time() - 86400},
            {"state": "explained", "t": time.time() - 82000},
            {"state": "booked", "t": time.time() - 3600, "appointment": "2026-10-12"}
        ]),
        "site": "Camp-A", "patient": "P-2026-0089", "eye": "left"
    },
    {
        "id": 2, "case_id": 98, "token": "tok_p98_escalated", "state": "escalated",
        "created": time.time() - 172800, "updated": time.time() - 7200,
        "urgent": 1, "contact_tier": "asha", "appointment": "2026-09-28",
        "history_json": json.dumps([
            {"state": "flagged", "t": time.time() - 172800},
            {"state": "booked", "t": time.time() - 140000},
            {"state": "missed", "t": time.time() - 14000},
            {"state": "escalated", "t": time.time() - 7200, "note": "Patient missed slot; escalated to ASHA"}
        ]),
        "site": "PHC-1", "patient": "P-2026-0021", "eye": "right"
    }
]

NEXT_STATES = {
    "flagged": ["explained"],
    "explained": ["booked"],
    "booked": ["reminded", "attended", "missed"],
    "reminded": ["attended", "missed"],
    "missed": ["escalated"],
    "escalated": ["booked", "closed"],
    "attended": ["closed"],
    "closed": []
}

def safe_fromjson(val):
    if isinstance(val, (dict, list)):
        return val
    if isinstance(val, (str, bytes, bytearray)):
        try:
            return json.loads(val)
        except Exception:
            return []
    return []


def render_simple_jinja(tpl_name: str, context: dict) -> str:
    """Render template using Jinja2 if available, or lightweight native fallback."""
    try:
        import jinja2
        env = jinja2.Environment(loader=jinja2.FileSystemLoader(str(TEMPLATES_DIR)))
        env.globals.update(ICDR=ICDR_NAMES)
        env.filters["fromjson"] = safe_fromjson
        env.filters["ago"] = lambda t: f"{int((time.time() - t) // 60)} min ago" if time.time() - t < 3600 else f"{(time.time() - t) / 3600:.1f} h ago"
        tpl = env.get_template(tpl_name)
        return tpl.render(context)
    except ImportError:
        # Fallback if jinja2 is not yet installed: basic template interpolation
        content = (TEMPLATES_DIR / tpl_name).read_text(encoding="utf-8")
        base = (TEMPLATES_DIR / "base.html").read_text(encoding="utf-8")
        # simple merge for preview
        body_match = re.search(r"{% block body %}(.*?){% endblock %}", content, re.DOTALL)
        body_content = body_match.group(1) if body_match else content
        return base.replace("{% block body %}{% endblock %}", body_content)


class DrishtiPreviewHandler(http.server.SimpleHTTPRequestHandler):
    def translate_path(self, path):
        # Serve /static files from app/static
        if path.startswith("/static/"):
            return str(STATIC_DIR / path[8:])
        return super().translate_path(path)

    def do_GET(self):
        url = urlparse(self.path)
        qs = parse_qs(url.query)
        path = url.path.rstrip("/") or "/"

        if path.startswith("/static/"):
            return super().do_GET()

        ctx = {
            "active": "screen",
            "lang": qs.get("lang", ["en"])[0],
            "missing": [],
            "samples": ["sample_messidor2_01.jpg", "sample_idrid_02.jpg", "sample_aptos_03.jpg"]
        }

        # Route matching
        if path == "/" or path == "/home":
            ctx["active"] = "home"
            html = render_simple_jinja("landing.html", ctx)

        elif path == "/login":
            ctx["active"] = "login"
            html = render_simple_jinja("login.html", ctx)

        elif path == "/portal/screener" or path == "/screen":
            ctx["active"] = "screener"
            html = render_simple_jinja("portal_screener.html", ctx)

        elif path == "/portal/doctor" or path == "/review":
            ctx["active"] = "doctor"
            html = render_simple_jinja("portal_doctor.html", ctx)

        elif path.startswith("/case/"):
            ctx["active"] = "screen"
            ctx["c"] = CASES[0]
            ctx["r"] = {
                "outcome": "review", "urgent": False, "seconds": 0.38,
                "grade": {"probs": [0.04, 0.12, 0.74, 0.08, 0.02], "p_referable": 0.84, "most_likely": "Moderate NPDR"},
                "grade_set": ["Mild NPDR", "Moderate NPDR"],
                "quality": {"passed": True, "gradability": 0.94, "metrics": {"evenness": 0.74, "focus": 0.048}},
                "images": {"photo": MOCK_PHOTO, "lesions": MOCK_LESIONS, "gradcam": MOCK_GRADCAM},
                "reasons": ["Uncertain grade boundary: moderate vs mild NPDR; hard exudate found"],
                "message": {
                    "main": "The photograph has been sent to an eye doctor for a quick 30-second review. You will receive an update shortly.",
                    "why": ["What the AI detected: small bleeding spots and yellow fat deposits.", "The model is uncertain, so an ophthalmologist will decide."],
                    "disclaimer": "Research prototype. Not a medical device. A doctor makes the final decision."
                }
            }
            html = render_simple_jinja("result.html", ctx)

        elif path.startswith("/review/"):
            ctx["active"] = "review"
            ctx["c"] = CASES[0]
            ctx["next_id"] = 102
            ctx["study"] = None
            ctx["r"] = {
                "outcome": "review", "urgent": False,
                "grade": {"probs": [0.03, 0.14, 0.71, 0.10, 0.02], "p_referable": 0.83, "most_likely": "Moderate NPDR"},
                "grade_set": ["Moderate NPDR", "Severe NPDR"],
                "images": {"photo": MOCK_PHOTO, "lesions": MOCK_LESIONS, "gradcam": MOCK_GRADCAM},
                "reasons": ["Model uncertain between moderate and severe disease"],
                "rules": {
                    "counts": {"MA": 12, "HE": 7, "EX": 4, "SE": 1},
                    "quadrants": {
                        "superior-nasal": {"MA": 3, "HE": 1, "EX": 0, "SE": 0},
                        "superior-temporal": {"MA": 4, "HE": 3, "EX": 1, "SE": 0},
                        "inferior-nasal": {"MA": 2, "HE": 1, "EX": 0, "SE": 0},
                        "inferior-temporal": {"MA": 3, "HE": 2, "EX": 3, "SE": 1}
                    },
                    "dme_flag": True,
                    "nearest_exudate_dd": 0.84,
                    "trace": [
                        "More than microaneurysms alone (7 haemorrhages, 4 hard exudates) -> at least moderate NPDR (ICDR 2), referable.",
                        "Nearest hard exudate is 0.8 disc diameters from fovea -> within 1 DD: possible macular oedema."
                    ]
                }
            }
            html = render_simple_jinja("review.html", ctx)

        elif path == "/referrals":
            ctx["active"] = "referrals"
            ctx["refs"] = REFERRALS
            ctx["next"] = NEXT_STATES
            html = render_simple_jinja("referrals.html", ctx)

        elif path == "/dashboard":
            ctx["active"] = "dashboard"
            ctx["d"] = {
                "outcomes": {"recall": 142, "review": 48, "refer": 24},
                "review_seconds_median": 14.8, "review_seconds_p90": 26.2,
                "referral_states": {"flagged": 4, "booked": 8, "missed": 2, "escalated": 3, "attended": 11},
                "outbox": [
                    {"created": time.time() - 1200, "referral_id": 1, "channel": "SMS (mock)", "recipient": "patient", "text": "Your eye appointment is confirmed for 2026-10-12 at District Eye Hospital."},
                    {"created": time.time() - 7200, "referral_id": 2, "channel": "SMS (mock)", "recipient": "asha", "text": "Patient P-2026-0021 missed their hospital slot. Please visit them to rebook."}
                ]
            }
            ctx["results"] = {
                "ens_d3 (Tri-Ensemble)": {
                    "sets": {
                        "Messidor-2 (External, France, Topcon)": {
                            "n_images": 1744, "auc_referable": [0.982], "qwk": 0.894,
                            "at_operating_point": {"sensitivity": [0.952, 0.928, 0.968], "specificity": [0.896, 0.878, 0.911]},
                            "triage_cnn_only": {"referable_auto_recalled": {"count": 5, "of_referable": 457}}
                        },
                        "IDRiD (External, India, Kowa)": {
                            "n_images": 516, "auc_referable": [0.976], "qwk": 0.881,
                            "at_operating_point": {"sensitivity": [0.932, 0.891, 0.958], "specificity": [0.871, 0.832, 0.902]},
                            "triage_cnn_only": {"referable_auto_recalled": {"count": 3, "of_referable": 192}}
                        }
                    }
                }
            }
            html = render_simple_jinja("dashboard.html", ctx)

        elif path == "/study":
            ctx["active"] = "study"
            ctx["items"] = [{"image_id": "MESS-001"}, {"image_id": "IDR-042"}]
            ctx["sessions"] = [{"id": 1, "reader": "DR-AN", "n": 40, "created": time.time() - 48000}]
            html = render_simple_jinja("study.html", ctx)

        elif path.startswith("/study-results/"):
            ctx["active"] = "study"
            ctx["s"] = {"id": 1, "reader": "DR-AN"}
            ctx["out"] = {
                "n": 40, "median_s": 14.8, "p90_s": 26.2, "under_30s_pct": 92.5,
                "qwk_vs_reference": 0.891,
                "referable_sens": [0.950, 0.880, 0.985],
                "referable_spec": [0.900, 0.820, 0.950]
            }
            ctx["rows"] = [
                {"image_id": "IDR-001", "ref_grade": 2, "grade": 2, "dme": 1, "ungradable": 0, "ms": 12400, "ai_p_ref": 0.84},
                {"image_id": "MESS-042", "ref_grade": 0, "grade": 0, "dme": 0, "ungradable": 0, "ms": 8600, "ai_p_ref": 0.05},
                {"image_id": "IDR-019", "ref_grade": 3, "grade": 3, "dme": 0, "ungradable": 0, "ms": 19200, "ai_p_ref": 0.94},
                {"image_id": "MESS-098", "ref_grade": 1, "grade": 1, "dme": 0, "ungradable": 0, "ms": 11100, "ai_p_ref": 0.22},
            ]
            html = render_simple_jinja("study_results.html", ctx)

        elif path.startswith("/slip/"):
            ctx["c"] = CASES[0]
            ctx["r"] = {
                "urgent": True, "reason": "lesions",
                "grade": {"p_referable": 0.94}, "grade_set": ["Severe NPDR"],
                "rules": {"counts": {"MA": 18, "HE": 11, "EX": 6, "SE": 2}, "dme_flag": True, "nearest_exudate_dd": 0.72}
            }
            ctx["cfg"] = {"urgent_days": 14, "routine_days": 90}
            ctx["base"] = f"http://localhost:{PORT}/"
            html = render_simple_jinja("slip.html", ctx)

        elif path.startswith("/s/"):
            ctx["c"] = CASES[0]
            ctx["r"] = {
                "outcome": "refer",
                "message": {
                    "main": "Diabetic eye damage was found in this photo. Please visit the district eye hospital within 14 days for specialist treatment.",
                    "why": ["Vascular bleeding and fatty deposits detected.", "Possible swelling near the centre of vision."],
                    "disclaimer": "Research prototype. Not a medical device. A doctor makes the final decision."
                }
            }
            ctx["ref"] = {"state": "booked", "appointment": "2026-10-15"}
            html = render_simple_jinja("status.html", ctx)

        else:
            self.send_error(404, "Page Not Found")
            return

        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(html.encode("utf-8"))

    def do_POST(self):
        url = urlparse(self.path)
        path = url.path.rstrip("/")

        # Mock API responses
        if path == "/api/screen":
            data = json.dumps({
                "case_id": 101,
                "token": "tok_p101_review",
                "outcome": "review",
                "result": {
                    "outcome": "review",
                    "urgent": False,
                    "case_id": 101,
                    "token": "tok_p101_review",
                    "grade": {"probs": [0.03, 0.12, 0.74, 0.09, 0.02], "p_referable": 0.85, "most_likely": "Moderate NPDR"},
                    "grade_set": ["Moderate NPDR", "Severe NPDR"],
                    "quality": {
                        "passed": True,
                        "gradability": 0.94,
                        "problems": [],
                        "instructions": [],
                        "metrics": {
                            "fov_fraction": 0.42,
                            "completeness": 0.88,
                            "median": 68.0,
                            "clipped": 0.02,
                            "evenness": 0.76,
                            "focus": 0.048
                        }
                    },
                    "rules": {
                        "counts": {"MA": 12, "HE": 7, "EX": 4, "SE": 1},
                        "dme_flag": True,
                        "nearest_exudate_dd": 0.84
                    }
                }
            }).encode("utf-8")
        elif path.startswith("/api/review/"):
            data = json.dumps({"final": "refer"}).encode("utf-8")
        elif path == "/api/referral/book":
            try:
                length = int(self.headers.get("Content-Length", 0))
                body = json.loads(self.rfile.read(length).decode("utf-8")) if length > 0 else {}
            except Exception:
                body = {}
            tok = body.get("token", "tok_p101_review")
            data = json.dumps({
                "ok": True,
                "booking": {
                    "referral_id": 1,
                    "token": tok,
                    "state": "booked",
                    "facility": body.get("facility", "District Eye Hospital (Ananthapur)"),
                    "specialty": body.get("specialty", "Vitreo-Retina Laser & Anti-VEGF"),
                    "appointment": body.get("appointment_date", "2026-10-06"),
                    "transit": body.get("transit", True),
                    "sms_sent": True
                },
                "slip_url": f"/slip/{tok}"
            }).encode("utf-8")
        elif path.startswith("/api/referral/"):
            data = json.dumps({"ok": True}).encode("utf-8")
        elif path == "/study/start":
            self.send_response(303)
            self.send_header("Location", "/review/101")
            self.end_headers()
            return
        else:
            data = json.dumps({"ok": True}).encode("utf-8")

        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(data)


def main():
    print(f"================================================================")
    print(f"  DrishtiTrust UI Preview Server Running on Port {PORT}")
    print(f"  Warm Retinal Light Theme (Zero Pure White)")
    print(f"================================================================")
    print(f"  -> Landing Page:    http://localhost:{PORT}/")
    print(f"  -> Portal Gateway:  http://localhost:{PORT}/login")
    print(f"  -> Screen Patient:  http://localhost:{PORT}/screen")
    print(f"  -> Tele-Review:     http://localhost:{PORT}/review")
    print(f"  -> Review Cockpit:  http://localhost:{PORT}/review/101")
    print(f"  -> Referrals:       http://localhost:{PORT}/referrals")
    print(f"  -> Dashboard:       http://localhost:{PORT}/dashboard")
    print(f"  -> Reader Study:    http://localhost:{PORT}/study")
    print(f"  -> Study Results:   http://localhost:{PORT}/study-results/1")
    print(f"  -> Referral Slip:   http://localhost:{PORT}/slip/demo")
    print(f"  -> Patient Status:  http://localhost:{PORT}/s/demo")
    class ThreadingServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        daemon_threads = True
        allow_reuse_address = True

    with ThreadingServer(("", PORT), DrishtiPreviewHandler) as httpd:
        httpd.serve_forever()


if __name__ == "__main__":
    main()
