"""App smoke tests: the full flow works with whatever models exist
(including none), using a synthetic fundus-like image."""
import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient


@pytest.fixture(scope="module")
def client(tmp_path_factory):
    from app import db
    db.DB = tmp_path_factory.mktemp("db") / "t.db"
    from app.main import app
    with TestClient(app) as c:
        yield c


def fundus_png(dark=False):
    img = np.zeros((900, 1100, 3), np.uint8)
    cv2.circle(img, (550, 450), 420, (40, 70, 170) if not dark else (5, 8, 20), -1)  # BGR orange-red disc
    rng = np.random.default_rng(0)
    for _ in range(40):  # vessel-like texture so focus is not zero
        p = rng.integers(200, 800, size=4)
        cv2.line(img, (int(p[0]), int(p[1])), (int(p[2]), int(p[3])), (20, 30, 110), 3)
    cv2.circle(img, (760, 430), 60, (150, 200, 230), -1)  # optic disc
    return cv2.imencode(".png", img)[1].tobytes()


def test_screen_flow(client):
    r = client.post("/api/screen", files={"file": ("x.png", fundus_png(), "image/png")}, data={"site": "PHC-1", "patient": "T1"})
    assert r.status_code == 200, r.text
    j = r.json()
    assert j["outcome"] in {"recall", "review", "refer", "recapture"}
    page = client.get(f"/case/{j['case_id']}?lang=hi")
    assert page.status_code == 200
    assert "शोध प्रोटोटाइप" in page.text  # Hindi disclaimer
    for url in ("/review", "/referrals", "/dashboard", "/study"):
        assert client.get(url).status_code == 200


def test_dark_image_asks_for_retake(client):
    r = client.post("/api/screen", files={"file": ("d.png", fundus_png(dark=True), "image/png")}, data={"attempt": "1"})
    assert r.json()["outcome"] == "recapture"


def test_review_and_referral_state_machine(client):
    r = client.post("/api/screen", files={"file": ("x.png", fundus_png(), "image/png")}).json()
    cid = r["case_id"]
    res = client.post(f"/api/review/{cid}", json={"grade": 3, "dme": 0, "ms": 21000})
    assert res.json()["final"] == "refer"
    from app import db
    ref = [x for x in db.referrals() if x["case_id"] == cid][0]
    assert client.post(f"/api/referral/{ref['id']}", json={"to": "attended"}).status_code == 400  # must be explained first
    for s in ("explained", "booked", "missed", "escalated"):
        assert client.post(f"/api/referral/{ref['id']}", json={"to": s}).status_code == 200
    ref = [x for x in db.referrals() if x["case_id"] == cid][0]
    assert ref["contact_tier"] == "family"  # escalation moves down the contact plan
    tok = ref["token"]
    assert client.get(f"/s/{tok}?lang=ta").status_code == 200
    assert client.get(f"/slip/{tok}").status_code == 200
