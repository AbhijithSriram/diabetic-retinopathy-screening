"""Plain-language messages for the health worker and patient.

Every message is a fixed template filled with the system's actual outputs;
nothing is generated freely, so every sentence a patient hears can be traced
to a rule. The Hindi and Tamil text needs review by a native speaker who
trains field staff before any real use (see docs/LIMITATIONS.md).

Referral time frames are programme settings (configs/referral.json), not
medical advice baked into code; the defaults must be confirmed by the
partner ophthalmologist.
"""
from __future__ import annotations

import json

from ..config import REPO

REFERRAL_CFG = REPO / "configs" / "referral.json"
DEFAULT_REFERRAL = {"urgent_days": 14, "routine_days": 90, "recall_months": 12, "review_hours": 48}


def referral_cfg() -> dict:
    cfg = dict(DEFAULT_REFERRAL)
    if REFERRAL_CFG.exists():
        cfg.update(json.loads(REFERRAL_CFG.read_text()))
    return cfg


T = {
    "en": {
        "recall": "No signs of diabetic eye damage were found in this photo. Screen this eye again in {recall_months} months. Keeping blood sugar and blood pressure under control protects the eyes.",
        "review": "The photo has been sent to an eye doctor for a quick check. The answer will come back to this app within {review_hours} hours. The patient does not need to travel now. Tell the patient you will contact them.",
        "refer": "Signs of diabetic eye damage were found. The patient should see an eye specialist within {routine_days} days. Treatment at this stage can prevent loss of vision.",
        "refer_urgent": "Serious signs of diabetic eye damage were found. The patient should see an eye specialist as soon as possible, within {urgent_days} days. Help the patient arrange the visit today.",
        "recapture": "The photo could not be read. {instruction}",
        "ungradable": "A clear photo could not be taken after {attempts} tries. This is NOT a normal result: the patient needs an in-person eye check (a common cause is cataract). Refer within {routine_days} days.",
        "why_lesions": "What the system saw: {what}.",
        "why_none": "The system did not find bleeding spots or leaking fat in the retina.",
        "why_uncertain": "The system is not sure, so a doctor will decide.",
        "dme": "Leaking fat is close to the centre of vision.",
        "disclaimer": "Research prototype. Not a medical device. A doctor makes the final decision.",
        "lesion_words": {"MA": "tiny bulges in blood vessels", "HE": "small bleeding spots", "EX": "yellow fat deposits", "SE": "fluffy white patches"},
    },
    "hi": {
        "recall": "इस फोटो में डायबिटीज़ से आंख को नुकसान के कोई लक्षण नहीं मिले। {recall_months} महीने बाद इस आंख की फिर से जांच करें। ब्लड शुगर और ब्लड प्रेशर नियंत्रण में रखने से आंखें सुरक्षित रहती हैं।",
        "review": "फोटो आंखों के डॉक्टर को जांच के लिए भेजी गई है। जवाब {review_hours} घंटे के अंदर इसी ऐप में आएगा। मरीज़ को अभी कहीं जाने की ज़रूरत नहीं है। मरीज़ को बताएं कि आप उनसे संपर्क करेंगे।",
        "refer": "डायबिटीज़ से आंख को नुकसान के लक्षण मिले हैं। मरीज़ को {routine_days} दिनों के अंदर आंखों के विशेषज्ञ को दिखाना चाहिए। इस समय इलाज से नज़र बचाई जा सकती है।",
        "refer_urgent": "डायबिटीज़ से आंख को गंभीर नुकसान के लक्षण मिले हैं। मरीज़ को जल्द से जल्द, {urgent_days} दिनों के अंदर, आंखों के विशेषज्ञ को दिखाना चाहिए। आज ही मरीज़ की मदद करें।",
        "recapture": "फोटो पढ़ी नहीं जा सकी। {instruction}",
        "ungradable": "{attempts} बार कोशिश के बाद भी साफ़ फोटो नहीं मिली। यह सामान्य नतीजा नहीं है: मरीज़ की आंख की जांच आमने-सामने करानी होगी (अक्सर मोतियाबिंद के कारण)। {routine_days} दिनों के अंदर रेफर करें।",
        "why_lesions": "सिस्टम ने यह देखा: {what}।",
        "why_none": "सिस्टम को रेटिना में खून के धब्बे या चर्बी का रिसाव नहीं मिला।",
        "why_uncertain": "सिस्टम पक्का नहीं है, इसलिए डॉक्टर तय करेंगे।",
        "dme": "चर्बी का रिसाव नज़र के केंद्र के पास है।",
        "disclaimer": "शोध प्रोटोटाइप। यह मेडिकल उपकरण नहीं है। अंतिम निर्णय डॉक्टर लेंगे।",
        "lesion_words": {"MA": "खून की नलियों में छोटे उभार", "HE": "खून के छोटे धब्बे", "EX": "पीली चर्बी के जमाव", "SE": "सफ़ेद रुई जैसे धब्बे"},
    },
    "ta": {
        "recall": "இந்த படத்தில் சர்க்கரை நோயால் கண் பாதிப்பு அறிகுறிகள் எதுவும் இல்லை. {recall_months} மாதங்களுக்குப் பிறகு மீண்டும் பரிசோதிக்கவும். இரத்த சர்க்கரை மற்றும் இரத்த அழுத்தத்தைக் கட்டுப்பாட்டில் வைத்தால் கண்கள் பாதுகாக்கப்படும்.",
        "review": "படம் கண் மருத்துவரின் பரிசோதனைக்கு அனுப்பப்பட்டுள்ளது. {review_hours} மணி நேரத்திற்குள் பதில் இந்த செயலியில் வரும். நோயாளி இப்போது பயணம் செய்ய வேண்டியதில்லை. நீங்கள் அவர்களைத் தொடர்பு கொள்வீர்கள் என்று நோயாளியிடம் சொல்லுங்கள்.",
        "refer": "சர்க்கரை நோயால் கண் பாதிப்பு அறிகுறிகள் கண்டறியப்பட்டுள்ளன. நோயாளி {routine_days} நாட்களுக்குள் கண் நிபுணரைப் பார்க்க வேண்டும். இந்த நிலையில் சிகிச்சை பார்வை இழப்பைத் தடுக்கும்.",
        "refer_urgent": "சர்க்கரை நோயால் கடுமையான கண் பாதிப்பு அறிகுறிகள் கண்டறியப்பட்டுள்ளன. நோயாளி விரைவில், {urgent_days} நாட்களுக்குள், கண் நிபுணரைப் பார்க்க வேண்டும். இன்றே நோயாளிக்கு உதவுங்கள்.",
        "recapture": "படத்தைப் படிக்க முடியவில்லை. {instruction}",
        "ungradable": "{attempts} முறை முயன்றும் தெளிவான படம் கிடைக்கவில்லை. இது சாதாரண முடிவு அல்ல: நோயாளிக்கு நேரடி கண் பரிசோதனை தேவை (பெரும்பாலும் கண்புரை காரணம்). {routine_days} நாட்களுக்குள் பரிந்துரைக்கவும்.",
        "why_lesions": "அமைப்பு கண்டது: {what}.",
        "why_none": "விழித்திரையில் இரத்தப் புள்ளிகளோ கொழுப்புக் கசிவோ கண்டறியப்படவில்லை.",
        "why_uncertain": "அமைப்புக்கு உறுதியில்லை, எனவே மருத்துவர் முடிவு செய்வார்.",
        "dme": "கொழுப்புக் கசிவு பார்வை மையத்திற்கு அருகில் உள்ளது.",
        "disclaimer": "ஆராய்ச்சி முன்மாதிரி. மருத்துவ சாதனம் அல்ல. இறுதி முடிவு மருத்துவருடையது.",
        "lesion_words": {"MA": "இரத்த நாளங்களில் சிறிய வீக்கங்கள்", "HE": "சிறிய இரத்தப் புள்ளிகள்", "EX": "மஞ்சள் கொழுப்புப் படிவுகள்", "SE": "வெள்ளை பஞ்சு போன்ற திட்டுகள்"},
    },
}


def worker_message(result: dict, lang: str = "en") -> dict:
    t = T.get(lang, T["en"])
    cfg = referral_cfg()
    outcome = result["outcome"]
    rules = result.get("rules") or {}
    counts = rules.get("counts", {})
    if outcome == "recapture":
        main = t["recapture"].format(instruction=" ".join(result["quality"]["instructions"]))
    elif outcome == "refer" and result.get("reason") == "ungradable":
        main = t["ungradable"].format(attempts=result.get("attempt", 3), **cfg)
    elif outcome == "refer":
        main = t["refer_urgent" if result.get("urgent") else "refer"].format(**cfg)
    else:
        main = t[outcome].format(**cfg)
    why = []
    # lesions are described only when the patient is being sent onwards; on a
    # routine recall a single faint detection would contradict the result
    if outcome in ("refer", "review") and result.get("reason") != "ungradable":
        if rules:  # only describe lesions when the lesion model actually ran
            seen = [t["lesion_words"][k] for k in ("HE", "EX", "SE", "MA") if counts.get(k)]
            why.append(t["why_lesions"].format(what=", ".join(seen)) if seen else t["why_none"])
        if rules.get("dme_flag"):
            why.append(t["dme"])
        if outcome == "review":
            why.append(t["why_uncertain"])
    return {"main": main, "why": why, "disclaimer": t["disclaimer"], "lang": lang}
