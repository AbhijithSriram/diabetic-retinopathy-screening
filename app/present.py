"""How a case is presented: one outcome -> colour/shape/words mapping used by
every screen (docs: the design handoff's "Outcome -> presentation" table).

Durations always come from configs/referral.json (referral_cfg), never
hard-coded. Patient-facing labels exist in English, Hindi and Tamil; the Hindi
and Tamil ones are new and, like drishti/report/messages.py, need a native
speaker's review before real use.
"""
from __future__ import annotations

import json
import time

from drishti.config import ICDR_NAMES
from drishti.report.messages import referral_cfg, worker_message

KINDS = ("recall", "review", "refer", "urgent", "ungradable", "recapture")


def case_kind(case, res: dict, review: dict | None) -> str:
    """recall | review | refer | urgent | ungradable | recapture, after the
    specialist's decision when there is one."""
    if review:
        if review["ungradable"]:
            return "ungradable"
        if case["final"] == "refer":
            return "urgent" if (review["grade"] >= 3 or review["dme"]) else "refer"
        return "recall"
    o = case["outcome"]
    if o == "refer" and res.get("reason") == "ungradable":
        return "ungradable"
    if o == "refer" and case["urgent"]:
        return "urgent"
    return o


def view(kind: str, reviewed: bool = False) -> dict:
    """Worker-facing words for an outcome kind."""
    c = referral_cfg()
    v = {
        "recall": ("No signs of eye damage", f"Screen this eye again in {c['recall_months']} months",
                   f"Recall · {c['recall_months']} months"),
        "review": ("Sent to an eye doctor", f"Answer within {c['review_hours']} hours", "Waiting for eye doctor"),
        "refer": ("Refer to eye hospital", f"Within {c['routine_days']} days", f"Refer · {c['routine_days']} days"),
        "urgent": ("Urgent: refer to eye hospital", f"Within {c['urgent_days']} days. Arrange the visit today.",
                   f"Urgent refer · {c['urgent_days']} days"),
        "ungradable": ("No clear photo after 3 tries", f"Not a normal result. In-person eye check within {c['routine_days']} days.",
                       "No clear photo · see in person"),
        "recapture": ("Take the photo again", "", "Retake needed"),
    }[kind]
    if kind == "ungradable" and reviewed:
        v = ("Eye doctor could not read the photo", f"Not a normal result. In-person eye check within {c['routine_days']} days.",
             "Doctor: no clear photo · see in person")
    return {"kind": kind, "title": v[0], "sub": v[1], "short": v[2],
            "books": kind in ("refer", "urgent", "ungradable"),
            "days": {"refer": c["routine_days"], "urgent": c["urgent_days"], "ungradable": c["routine_days"]}.get(kind)}


def latest_review(k, case_id: int):
    r = k.execute("SELECT reviewer, grade, dme, ungradable, ms, final, created FROM reviews WHERE case_id=? ORDER BY id DESC LIMIT 1",
                  (case_id,)).fetchone()
    return dict(r) if r else None


def referral_of(k, case_id: int):
    r = k.execute("SELECT * FROM referrals WHERE case_id=?", (case_id,)).fetchone()
    if not r:
        return None
    r = dict(r)
    hist = json.loads(r["history_json"])
    booked = next((h for h in reversed(hist) if h.get("state") == "booked"), {})
    r["facility"] = booked.get("facility")
    r["specialty"] = booked.get("specialty")
    r["transit"] = booked.get("transit")
    r["escort"] = booked.get("escort")
    r["history"] = hist
    return r


def patient_message(res: dict, case, kind: str, lang: str) -> dict:
    """The message for the outcome the patient should act on now."""
    outcome = {"urgent": "refer", "ungradable": "refer"}.get(kind, kind)
    r = {**res, "outcome": outcome, "urgent": kind == "urgent"}
    if kind != "ungradable":
        r.pop("reason", None)
    return worker_message(r, lang)


def review_preview(grade, dme: bool) -> dict:
    """What the health worker receives for a specialist's choice (mirrors db.add_review)."""
    if grade is None:
        return {"kind": None, "text": "Choose a grade"}
    if grade == "U":
        k = "ungradable"
    elif grade >= 3 or dme:
        k = "urgent"
    elif grade >= 2:
        k = "refer"
    else:
        k = "recall"
    return {"kind": k, "text": view(k)["short"]}


def grade_set_idx(names) -> list[int]:
    if not names:
        return []
    ix = [ICDR_NAMES.index(n) for n in names if n in ICDR_NAMES]
    return list(range(min(ix), max(ix) + 1)) if ix else []


def waiting(t: float) -> dict:
    h = (time.time() - t) / 3600
    label = f"{int(h * 60)} min" if h < 1 else f"{h:.0f} h"
    return {"label": label, "hours": round(h, 2), "level": "late" if h >= 48 else "amber" if h >= 24 else ""}


# patient-facing labels (status page and slip)
P = {
    "en": {
        "your_check": "Your eye check", "checked_at": "Checked at {site} on {date}", "right": "Right eye", "left": "Left eye",
        "appointment": "Your appointment", "bring_slip": "Bring your slip.", "transport": "Transport has been arranged.",
        "not_booked": "Appointment not booked yet.", "not_booked_2": "Your health worker will contact you. This page will show the date when it is booked.",
        "updates": "This page updates by itself.", "next_check": "Next check", "print": "Print the referral slip",
        "other_eye": "Other eye", "not_screened": "not screened yet",
        "recall": ("No signs of eye damage", "Screen again in {recall_months} months"),
        "review": ("Eye doctor checking", "Answer within {review_hours} hours"),
        "refer": ("See an eye specialist", "Within {routine_days} days"),
        "urgent": ("See an eye specialist urgently", "Within {urgent_days} days"),
        "ungradable": ("Eye check in person", "Not a normal result · within {routine_days} days"),
        "recapture": ("Photo being retaken", "Your health worker is taking the photo again"),
        "bad_title": "This link did not open a result",
        "bad_body": "Check that the whole link was copied from your SMS or slip. If it still does not work, ask your health worker. Your result is safe.",
        "disclaimer": "Research prototype. Not a medical device. A doctor makes the final decision.",
    },
    "hi": {
        "your_check": "आपकी आंख की जांच", "checked_at": "{site} · {date}", "right": "दाईं आंख", "left": "बाईं आंख",
        "appointment": "आपका अपॉइंटमेंट", "bring_slip": "अपनी पर्ची साथ लाएं।", "transport": "आने-जाने की व्यवस्था की गई है।",
        "not_booked": "अभी अपॉइंटमेंट तय नहीं हुआ है।", "not_booked_2": "आपके स्वास्थ्य कार्यकर्ता आपसे संपर्क करेंगे। तय होने पर तारीख इस पेज पर दिखेगी।",
        "updates": "यह पेज अपने आप अपडेट होगा", "next_check": "अगली जांच", "print": "रेफ़रल पर्ची प्रिंट करें",
        "other_eye": "दूसरी आंख", "not_screened": "अभी जांच नहीं हुई",
        "recall": ("नुकसान के कोई लक्षण नहीं", "{recall_months} महीने बाद फिर से जांच"),
        "review": ("डॉक्टर जांच कर रहे हैं", "{review_hours} घंटे के अंदर जवाब"),
        "refer": ("आंखों के विशेषज्ञ को दिखाएं", "{routine_days} दिनों के अंदर"),
        "urgent": ("जल्द आंखों के विशेषज्ञ को दिखाएं", "{urgent_days} दिनों के अंदर"),
        "ungradable": ("आमने-सामने आंख की जांच", "सामान्य नतीजा नहीं · {routine_days} दिनों के अंदर"),
        "recapture": ("फोटो फिर से ली जा रही है", "स्वास्थ्य कार्यकर्ता फोटो दोबारा ले रहे हैं"),
        "bad_title": "यह लिंक नहीं खुला",
        "bad_body": "अपने स्वास्थ्य कार्यकर्ता से पूछें।",
        "disclaimer": "शोध प्रोटोटाइप। यह मेडिकल उपकरण नहीं है। अंतिम निर्णय डॉक्टर लेंगे।",
    },
    "ta": {
        "your_check": "உங்கள் கண் பரிசோதனை", "checked_at": "{site} · {date}", "right": "வலது கண்", "left": "இடது கண்",
        "appointment": "உங்கள் சந்திப்பு", "bring_slip": "உங்கள் சீட்டைக் கொண்டு வாருங்கள்.", "transport": "போக்குவரத்து ஏற்பாடு செய்யப்பட்டுள்ளது.",
        "not_booked": "சந்திப்பு இன்னும் பதிவு செய்யப்படவில்லை.", "not_booked_2": "உங்கள் சுகாதாரப் பணியாளர் உங்களைத் தொடர்பு கொள்வார். பதிவு செய்ததும் தேதி இந்தப் பக்கத்தில் தெரியும்.",
        "updates": "இந்தப் பக்கம் தானாகப் புதுப்பிக்கப்படும்", "next_check": "அடுத்த பரிசோதனை", "print": "பரிந்துரைச் சீட்டை அச்சிடுங்கள்",
        "other_eye": "மற்ற கண்", "not_screened": "இன்னும் பரிசோதிக்கப்படவில்லை",
        "recall": ("பாதிப்பு அறிகுறிகள் இல்லை", "{recall_months} மாதங்களில் மீண்டும்"),
        "review": ("மருத்துவர் பரிசோதிக்கிறார்", "{review_hours} மணி நேரத்திற்குள் பதில்"),
        "refer": ("கண் நிபுணரைப் பாருங்கள்", "{routine_days} நாட்களுக்குள்"),
        "urgent": ("விரைவில் கண் நிபுணரைப் பாருங்கள்", "{urgent_days} நாட்களுக்குள்"),
        "ungradable": ("நேரடி கண் பரிசோதனை", "சாதாரண முடிவு அல்ல · {routine_days} நாட்களுக்குள்"),
        "recapture": ("படம் மீண்டும் எடுக்கப்படுகிறது", "சுகாதாரப் பணியாளர் படத்தை மீண்டும் எடுக்கிறார்"),
        "bad_title": "இந்த இணைப்பு திறக்கவில்லை",
        "bad_body": "உங்கள் சுகாதாரப் பணியாளரிடம் கேளுங்கள்.",
        "disclaimer": "ஆராய்ச்சி முன்மாதிரி. மருத்துவ சாதனம் அல்ல. இறுதி முடிவு மருத்துவருடையது.",
    },
}

# referral slip: the outcome line in all three languages
SLIP = {
    "refer": ("Refer: see an eye specialist within {routine_days} days",
              "{routine_days} दिनों के अंदर आंखों के विशेषज्ञ को दिखाएं",
              "{routine_days} நாட்களுக்குள் கண் நிபுணரைப் பாருங்கள்"),
    "urgent": ("URGENT: see an eye specialist within {urgent_days} days",
               "ज़रूरी: {urgent_days} दिनों के अंदर आंखों के विशेषज्ञ को दिखाएं",
               "அவசரம்: {urgent_days} நாட்களுக்குள் கண் நிபுணரைப் பாருங்கள்"),
    "ungradable": ("No clear photo: in-person eye check within {routine_days} days",
                   "साफ़ फोटो नहीं: {routine_days} दिनों के अंदर आमने-सामने आंख की जांच",
                   "தெளிவான படம் இல்லை: {routine_days} நாட்களுக்குள் நேரடி கண் பரிசோதனை"),
    "review": ("Sent to an eye doctor: answer within {review_hours} hours",
               "आंखों के डॉक्टर को भेजा गया: {review_hours} घंटे के अंदर जवाब",
               "கண் மருத்துவருக்கு அனுப்பப்பட்டது: {review_hours} மணி நேரத்திற்குள் பதில்"),
    "recall": ("No referral needed: screen again in {recall_months} months",
               "रेफ़रल की ज़रूरत नहीं: {recall_months} महीने बाद फिर से जांच",
               "பரிந்துரை தேவையில்லை: {recall_months} மாதங்களில் மீண்டும் பரிசோதனை"),
}


def patient_labels(lang: str) -> dict:
    c = referral_cfg()
    base = P.get(lang, P["en"])
    out = dict(base)
    for k in KINDS:
        t, s = base[k]
        out[k] = (t.format(**c), s.format(**c))
    return out


def slip_line(kind: str) -> tuple[str, str, str]:
    c = referral_cfg()
    return tuple(x.format(**c) for x in SLIP.get(kind, SLIP["review"]))
