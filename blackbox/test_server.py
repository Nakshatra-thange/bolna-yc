"""Tests for Phase 5: sync, pinning, forks, anchors, server-side tamper detection, FHIR, dashboard.  python -m pytest -q"""
import base64
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest
from fastapi.testclient import TestClient

from audit.chain import canonical
from audit.keys import KeyStore, load_public_key
from server.app import create_app
from sync.client import SyncClient

T0 = datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)
LINES = ["patient weight seventy kilos, allergic to morphine", "epi one milligram IV", "epi one milligram IV",
         "override, rhythm changed", "fentanyl fifty micrograms IV", "confirm", "morphine four milligrams IV",
         "cancel dose"]


def build_device(tmp_path, lines=LINES, incident="INC-S"):
    from asr.engine import Transcript
    from asr.live import Pipeline
    from asr.vad import Clip

    class Engine:
        it = iter(lines)

        def transcribe(self, audio):
            return Transcript(text=next(self.it), audio_s=1.0, engine="fake")

    pipe = Pipeline(Engine(), incident, tmp_path / "runs", extract=True, keys_dir=tmp_path / "keys")
    for i in range(len(lines)):
        pipe.handle(Clip((np.random.default_rng(i).standard_normal(8000) * 0.1).astype(np.float32), i, i + .5),
                    T0 + timedelta(minutes=i))
    return pipe


@pytest.fixture
def world(tmp_path):
    pipe = build_device(tmp_path)
    app = create_app(tmp_path / "server_data")
    http = TestClient(app)

    def opener(method, path, body):
        r = http.request(method, path, json=body)
        if r.status_code >= 400:
            raise RuntimeError(f"server said {r.status_code}: {r.json().get('detail')}")
        return r.json()

    client = SyncClient("http://test", tmp_path / "runs" / "blackbox.db", pipe.keys, opener=opener)
    return pipe, app, http, client


def test_sync_anchor_and_verified_view(world):
    pipe, app, http, client = world
    r = client.push()
    assert r["accepted"] > 0 and r["clips"] > 0
    rc = r["receipt"]
    load_public_key(rc["server_public_key"]).verify(
        base64.b64decode(rc["sig"]), canonical({k: rc[k] for k in ("device_id", "seq", "entry_hash", "server_time")}))
    v = http.get(f"/api/devices/{pipe.keys.device_id}/view").json()
    assert v["integrity"]["ok"] and v["stats"]["alerts"] == 2 and v["stats"]["overrides"] == 1
    assert {x["status"] for x in v["record"] if x["event"].get("drug") == "morphine"} == {"cancelled"}
    assert v["patient"]["allergies"] == ["morphine"]


def test_resync_is_idempotent_and_device_anchor_entry_syncs(world):
    pipe, app, http, client = world
    client.push()
    r2 = client.push()                      # sends the device's own 'anchor' entry
    r3 = client.push()                      # nothing new
    assert r2["accepted"] == 1 and r3["accepted"] == 0
    kinds = [json.loads(e.body)["kind"] for e in pipe.chain.entries()]
    assert kinds.count("anchor") == 1


def test_audio_playback_matches_chain(world):
    pipe, app, http, client = world
    client.push()
    v = http.get(f"/api/devices/{pipe.keys.device_id}/view").json()
    seq = next(c["seq"] for c in v["chain"] if c["has_audio"])
    r = http.get(f"/api/devices/{pipe.keys.device_id}/audio/{seq}")
    assert r.status_code == 200 and r.headers["x-audio-matches-chain"] == "true" and r.content[:4] == b"RIFF"


def test_server_side_tampering_is_detected_and_restorable(world):
    pipe, app, http, client = world
    client.push()
    dev = pipe.keys.device_id
    v = http.get(f"/api/devices/{dev}/view").json()
    seq = next(c["seq"] for c in v["chain"] if c["kind"] == "event" and "fentanyl" in c["summary"])
    http.post("/api/demo/tamper", json={"device_id": dev, "kind": "edit_dose", "seq": seq, "value": 500})
    v = http.get(f"/api/devices/{dev}/view").json()
    assert not v["integrity"]["ok"] and v["integrity"]["bad"] == [seq]
    fent = next(x for x in v["record"] if x["event"].get("drug") == "fentanyl")
    assert fent["event"]["dose"] == 500 and not fent["integrity_ok"]          # UI shows the lie AND flags it
    http.post("/api/demo/tamper", json={"device_id": dev, "kind": "delete", "seq": seq + 2})
    probs = {c["seq"]: c["problems"] for c in http.get(f"/api/devices/{dev}/view").json()["chain"] if c["problems"]}
    assert any("ENTRIES DELETED" in p for ps in probs.values() for p in ps)
    http.post("/api/demo/restore", json={"device_id": dev})
    assert http.get(f"/api/devices/{dev}/view").json()["integrity"]["ok"]


def test_truncation_after_anchor_is_detected(world):
    pipe, app, http, client = world
    client.push()
    dev = pipe.keys.device_id
    head = http.get(f"/api/devices/{dev}/head").json()["seq"]
    http.post("/api/demo/tamper", json={"device_id": dev, "kind": "delete", "seq": head})
    v = http.get(f"/api/devices/{dev}/view").json()
    assert not v["integrity"]["ok"] and "TAIL TRUNCATED" in v["integrity"]["chain_problems"][0]


def test_forged_entry_rejected_and_fork_recorded(world, tmp_path):
    pipe, app, http, client = world
    client.push()
    dev = pipe.keys.device_id
    con = sqlite3.connect(pipe.chain.path)
    seq, h, prev, body, sig = con.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain WHERE seq=3").fetchone()
    con.close()
    forged = body.replace('"dose":1.0', '"dose":10.0')
    r = http.post("/api/sync", json={"device_id": dev, "entries": [
        {"seq": seq, "entry_hash": "f" * 64, "prev_hash": prev, "body": forged, "sig": sig}], "clips": {}})
    assert r.status_code == 409 and "FORK" in r.json()["detail"]
    assert http.get(f"/api/devices/{dev}/view").json()["forks"]


def test_bad_signature_batch_rejected(world):
    pipe, app, http, client = world
    dev = pipe.keys.device_id
    client.register()
    con = sqlite3.connect(pipe.chain.path)
    rows = con.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain ORDER BY seq").fetchall()
    con.close()
    entries = [dict(zip(("seq", "entry_hash", "prev_hash", "body", "sig"), r)) for r in rows]
    entries[2]["sig"] = entries[1]["sig"]
    r = http.post("/api/sync", json={"device_id": dev, "entries": entries, "clips": {}})
    assert r.status_code == 422 and "SIGNATURE INVALID" in r.json()["detail"]


def test_device_key_is_pinned(world, tmp_path):
    pipe, app, http, client = world
    client.register()
    other = KeyStore(tmp_path / "evil")
    r = http.post("/api/devices/register", json={"device_id": pipe.keys.device_id, "public_key": other.public_key_hex})
    assert r.status_code == 409


def test_fhir_bundle_has_provenance_signatures(world):
    pipe, app, http, client = world
    client.push()
    b = http.get(f"/api/devices/{pipe.keys.device_id}/incidents/INC-S/fhir").json()
    types = [e["resource"]["resourceType"] for e in b["entry"]]
    assert b["resourceType"] == "Bundle" and "MedicationAdministration" in types and "Provenance" in types
    morphine = next(e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "MedicationAdministration"
                    and e["resource"]["medicationCodeableConcept"]["coding"][0]["code"] == "7052")
    assert morphine["status"] == "not-done" and "CONTRAINDICATED" in morphine["note"][0]["text"]
    prov = [e["resource"] for e in b["entry"] if e["resource"]["resourceType"] == "Provenance"]
    assert all(p["signature"][0]["data"] for p in prov) and b["meta"]["security"][0]["code"] == "intact"


def test_dashboard_and_handover_render(world):
    pipe, app, http, client = world
    assert "No synced device yet" in http.get("/").text
    client.push()
    h = http.get(f"/api/devices/{pipe.keys.device_id}/incidents/INC-S/handover").text
    assert "INTACT" in h and "morphine" in h and "overridden" in h