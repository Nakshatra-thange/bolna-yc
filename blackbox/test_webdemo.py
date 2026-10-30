"""Web demo API: typed input drives the real pipeline; script triggers the alerts we promise in the pitch."""
import argparse

import pytest
from fastapi.testclient import TestClient

from webdemo.app import PHRASES, create_app


@pytest.fixture()
def client(tmp_path):
    args = argparse.Namespace(port=0, host="x", model="small", language="en", no_asr=True, llm_model=None,
                              llm_backend="llamacpp", keys=str(tmp_path / "k"), out=str(tmp_path),
                              incident="T1", server_url="http://127.0.0.1:1")
    with TestClient(create_app(args)) as c:
        yield c


def say(c, t):
    r = c.post("/api/text", json={"text": t})
    assert r.status_code == 200, r.text
    return r.json()


def test_demo_script_end_to_end(client):
    rules, actions = [], []
    for line in PHRASES["script"]:
        u = say(client, line)["utterance"]
        rules += [a["rule_id"] for a in u["alerts"]]
        actions += [r["action"] for r in (u["alert_response"] or [])] + [(u["confirmation"] or {}).get("action")]
    assert {"CONTRAINDICATED", "DUPLICATE", "DOSE_ABOVE_MAX"} <= set(rules)
    assert {"overridden", "acknowledged", "cancelled_dose", "confirmed"} <= set(actions)
    st = client.get("/api/state").json()
    assert st["chain"]["ok"] and st["patient"]["weight_kg"] == 70 and st["patient"]["allergies"] == ["aspirin"]
    assert any(x["event"].get("vital") == {"sbp": 120, "dbp": 80, "hr": 90} for x in st["record"])


def test_every_chip_phrase_is_accepted(client):
    for g in PHRASES["groups"]:
        for line in g["items"]:
            say(client, line)
    assert client.get("/api/state").json()["chain"]["ok"]


def test_readback_is_returned_for_the_browser_to_speak(client):
    u = say(client, "Gave morphine ten milligram IV")["utterance"]
    assert u["say"] and "morphine" in u["say"][0].lower()
    assert say(client, "confirm")["utterance"]["confirmation"]["action"] == "confirmed"


def test_tamper_demo_detects_and_leaves_real_chain_intact(client):
    assert not client.post("/api/tamper").json()["ok"]            # nothing to forge yet
    say(client, "Gave adrenaline one milligram IV")
    t = client.post("/api/tamper").json()
    assert t["detected"] and t["new"] == 10 * t["old"] and t["first_bad"] == t["seq"]
    assert client.get("/api/state").json()["chain"]["ok"]


def test_voice_without_speech_model_is_a_clear_error(client):
    r = client.post("/api/voice", content=b"\0" * 64000)
    assert r.status_code == 503 and "no-asr" in r.json()["detail"]


def test_new_incident_and_sync_failure_is_graceful(client):
    say(client, "Gave adrenaline one milligram IV")
    st = client.post("/api/incident", json={"name": "INC-2"}).json()
    assert st["incident"] == "INC-2" and st["record"] == []
    s = client.post("/api/sync", json={}).json()
    assert s["ok"] is False and "safe on the device" in s["message"]