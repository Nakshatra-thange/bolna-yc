"""
Black Box server - the hospital side.

  python -m server.app                      # http://127.0.0.1:8000   (dashboard at /)
  python -m server.app --port 8080 --data server_data

API
  POST /api/devices/register   {device_id, public_key, audio_key}     pin the device key (first time only)
  GET  /api/devices/{id}/head                                          last seq the server has
  POST /api/sync               {device_id, entries:[...], clips:{path: b64}}  -> anchor receipt (server-signed)
  GET  /api/devices/{id}/view?incident=...                              everything the dashboard shows
  GET  /api/devices/{id}/audio/{seq}                                    decrypted original voice clip (wav)
  GET  /api/devices/{id}/incidents/{inc}/fhir                           FHIR R4 Bundle (+ Provenance signatures)
  GET  /api/devices/{id}/incidents/{inc}/handover                       printable handover sheet (HTML -> PDF)
  POST /api/demo/tamper  {device_id, kind, seq, value}                  insider edits the SERVER copy (demo)
  POST /api/demo/restore {device_id}                                    put the synced originals back
"""
from __future__ import annotations

import argparse
import hashlib
import html
import json
import sys
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import Body, FastAPI, HTTPException  # noqa: E402
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse, Response  # noqa: E402

from audit.keys import load_public_key  # noqa: E402
from audit.vault import decrypt_file  # noqa: E402
from audit.verify import materialize, patient_context, verify_chain  # noqa: E402
from server.fhir import build_bundle  # noqa: E402
from server.store import ServerStore, SyncError  # noqa: E402

STATIC = Path(__file__).with_name("static")


def summary(d: dict) -> str:
    p, k = d.get("payload", {}), d.get("kind")
    try:
        if k == "genesis":
            return f"device {p['device_id']} created"
        if k == "utterance":
            return f"“{p.get('text', '')}”"
        if k == "event":
            e = p["event"]
            if e["event_type"] == "medication":
                return f"{(e.get('drug') or '?').replace('_', ' ')} {e.get('dose', '?'):g} {e.get('unit', '')} {e.get('route', '?')}" \
                    if isinstance(e.get("dose"), (int, float)) else f"{e.get('drug')} (dose ?) {e.get('route', '?')}"
            if e["event_type"] == "procedure":
                return f"procedure: {(e.get('procedure') or '').replace('_', ' ')}"
            if e["event_type"] == "vital":
                return "vitals " + ", ".join(f"{a} {b:g}" for a, b in e.get("vital", {}).items())
            return "note"
        if k == "confirmation":
            return f"{p['action']} by voice" + (f" (“{p.get('said')}”)" if p.get("said") else "")
        if k == "alert":
            return f"Guardian {p['rule_id']}: {p['message']}"
        if k == "alert_response":
            return f"alert {p['action'].replace('_', ' ')}" + (f" — {p['reason']}" if p.get("reason") else "")
        if k == "patient":
            return "patient: " + ", ".join(f"{a} {b}" for a, b in p.items() if a not in ("utterance_ref", "current"))
        if k == "correction":
            return f"correction {p['changes']} — {p.get('reason')} (by {p.get('by')})"
        if k == "anchor":
            return f"anchored at server: #{p.get('seq')}"
    except Exception:  # noqa: BLE001 - a tampered body may not have the expected shape
        pass
    return k or "?"


def create_app(data_dir: str | Path = ROOT / "server_data") -> FastAPI:
    store = ServerStore(data_dir)
    app = FastAPI(title="Black Box server")
    app.state.store = store

    def guard(fn):
        try:
            return fn()
        except SyncError as e:
            raise HTTPException(e.status, e.detail)

    def dev_or_404(device_id: str) -> dict:
        d = store.device(device_id)
        if not d:
            raise HTTPException(404, "unknown device")
        return d

    # ------------------------------------------------------------- sync API
    @app.get("/api/server")
    def server_info():
        return {"server_public_key": store.keys.public_key_hex,
                "devices": [{"device_id": d["device_id"], "registered_at": d["registered_at"],
                             "head": store.head(d["device_id"])[0]} for d in store.devices()]}

    @app.post("/api/devices/register")
    def register(body: dict = Body(...)):
        return guard(lambda: store.register(body["device_id"], body["public_key"], body.get("audio_key")))

    @app.get("/api/devices/{device_id}/head")
    def head(device_id: str):
        dev_or_404(device_id)
        s, h = store.head(device_id)
        return {"seq": s, "entry_hash": h, "anchor": store.last_anchor(device_id)}

    @app.post("/api/sync")
    def sync(body: dict = Body(...)):
        return guard(lambda: store.sync(body["device_id"], body.get("entries", []), body.get("clips", {})))

    # ------------------------------------------------------------- dashboard data
    def _verify(device_id: str):
        dev = dev_or_404(device_id)
        entries = store.entries(device_id)
        akey = bytes.fromhex(dev["audio_key"]) if dev.get("audio_key") else None
        anchor = store.last_anchor(device_id)
        rep = verify_chain(entries, load_public_key(dev["public_key"]), audio_root=store.clips / device_id,
                           audio_key=akey, expected_head=(anchor["seq"], anchor["entry_hash"]) if anchor else None)
        return dev, entries, rep, anchor

    @app.get("/api/devices/{device_id}/view")
    def view(device_id: str, incident: Optional[str] = None):
        dev, entries, rep, anchor = _verify(device_id)
        datas = {}
        for e in entries:
            try:
                datas[e.seq] = json.loads(e.body)
            except json.JSONDecodeError:
                datas[e.seq] = {"kind": "?", "payload": {}}
        incs = []
        for d in datas.values():
            i = d.get("incident_id")
            if i and i not in incs:
                incs.append(i)
        inc = incident if incident in incs else (incs[-1] if incs else None)
        res_by_seq = {r.seq: r for r in rep.results}
        hash_to_utt = {e.entry_hash: (e.seq, datas[e.seq].get("payload", {})) for e in entries
                       if datas[e.seq].get("kind") == "utterance"}
        good_entries = [e for e in entries if datas[e.seq].get("kind") != "?"]
        record = materialize(good_entries, inc) if inc else []
        ev_entry = {datas[e.seq]["payload"].get("event_id"): e for e in entries if datas[e.seq].get("kind") == "event"}
        for x in record:
            ent = ev_entry.get(x["event_id"])
            ref = datas[ent.seq]["payload"].get("utterance_ref") if ent else None
            if ref in hash_to_utt:
                x["utterance_seq"], up = hash_to_utt[ref]
                x["utterance_text"] = up.get("text")
                x["raw_text"] = up.get("raw_text")
            x["integrity_ok"] = bool(ent) and res_by_seq.get(ent.seq) is not None and res_by_seq[ent.seq].ok
            x["entry_seq"] = ent.seq if ent else None
        chain = []
        for e in entries:
            d, r = datas[e.seq], res_by_seq.get(e.seq)
            if inc and d.get("incident_id") not in (inc, None):
                continue
            chain.append({"seq": e.seq, "kind": d.get("kind"), "ts": d.get("ts_wall"), "summary": summary(d),
                          "hash": e.entry_hash, "ok": bool(r and r.ok), "problems": r.problems if r else [],
                          "warnings": r.warnings if r else [],
                          "has_audio": d.get("kind") == "utterance" and bool(d.get("payload", {}).get("clip_path"))})
        meds = [x for x in record if x["event"]["event_type"] == "medication"]
        alerts = [a for x in record for a in x.get("alerts", [])]
        stats = {"entries": len(chain), "medications": len(meds),
                 "procedures": sum(x["event"]["event_type"] == "procedure" for x in record),
                 "vitals": sum(x["event"]["event_type"] == "vital" for x in record),
                 "alerts": len(alerts),
                 "overrides": sum(1 for a in alerts if (a.get("response") or {}).get("action") == "overridden"),
                 "cancelled": sum(x["status"] == "cancelled" for x in meds),
                 "confirmed": sum(x["status"] == "confirmed" for x in meds)}
        bad = [r.seq for r in rep.results if not r.ok]
        return {"device": {"device_id": device_id, "registered_at": dev["registered_at"],
                           "public_key": dev["public_key"]},
                "server_public_key": store.keys.public_key_hex,
                "incidents": incs, "incident": inc,
                "patient": patient_context(good_entries, inc) if inc else {},
                "integrity": {"ok": rep.ok, "bad": bad, "chain_problems": rep.chain_problems,
                              "head_seq": rep.head_seq, "head_hash": rep.head_hash, "total": len(entries)},
                "anchor": anchor, "forks": store.forks(device_id),
                "record": record, "chain": chain, "stats": stats}

    @app.get("/api/devices/{device_id}/audio/{seq}")
    def audio(device_id: str, seq: int):
        dev = dev_or_404(device_id)
        e = next((x for x in store.entries(device_id) if x.seq == seq), None)
        if not e:
            raise HTTPException(404, "no such entry")
        p = json.loads(e.body).get("payload", {})
        clip = store.clips / device_id / p.get("clip_path", "")
        if not p.get("clip_path") or not clip.exists() or not dev.get("audio_key"):
            raise HTTPException(404, "no voice clip for this entry")
        try:
            wav = decrypt_file(clip, bytes.fromhex(dev["audio_key"]), aad=p["clip_path"])
        except Exception:  # noqa: BLE001
            raise HTTPException(409, "voice clip fails authenticated decryption (altered)")
        match = hashlib.sha256(wav).hexdigest() == p.get("audio_sha256")
        return Response(wav, media_type="audio/wav", headers={"X-Audio-Matches-Chain": str(match).lower()})

    @app.get("/api/devices/{device_id}/incidents/{incident}/fhir")
    def fhir(device_id: str, incident: str):
        dev, entries, rep, anchor = _verify(device_id)
        index = {}
        for e in entries:
            try:
                index[e.seq] = {"entry_hash": e.entry_hash, "sig": e.sig, "ts_wall": json.loads(e.body)["ts_wall"]}
            except Exception:  # noqa: BLE001
                pass
        b = build_bundle(device_id, incident, materialize(entries, incident), patient_context(entries, incident),
                         index, anchor)
        b.setdefault("meta", {})["security"] = [{"system": "urn:blackbox:integrity",
                                                  "code": "intact" if rep.ok else "TAMPERED"}]
        return JSONResponse(b, headers={"Content-Disposition": f'attachment; filename="{incident}-fhir.json"'})

    @app.get("/api/devices/{device_id}/incidents/{incident}/handover", response_class=HTMLResponse)
    def handover(device_id: str, incident: str):
        v = view(device_id, incident)
        return HTMLResponse(render_handover(v))

    # ------------------------------------------------------------- demo controls
    @app.post("/api/demo/tamper")
    def tamper(body: dict = Body(...)):
        return {"done": guard(lambda: store.tamper(body["device_id"], body["kind"], int(body["seq"]), body.get("value")))}

    @app.post("/api/demo/restore")
    def restore(body: dict = Body(...)):
        return {"restored": store.restore(body["device_id"])}

    @app.get("/", response_class=HTMLResponse)
    def index():
        return FileResponse(STATIC / "index.html")

    return app


def render_handover(v: dict) -> str:
    esc = html.escape
    p = v["patient"] or {}
    integ = v["integrity"]
    rows = []
    for x in v["record"]:
        ev = x["event"]
        t = (ev.get("time_said") or "")[11:16]
        if ev["event_type"] == "medication":
            what = f"<b>{esc((ev.get('drug') or '?').replace('_', ' '))}</b> {ev.get('dose', '?')} {esc(ev.get('unit', ''))} {esc(ev.get('route', '?'))}"
        elif ev["event_type"] == "procedure":
            what = esc((ev.get("procedure") or "").replace("_", " ")) + (
                f" {ev['params']['joules']} J" if ev.get("params", {}).get("joules") else "")
        elif ev["event_type"] == "vital":
            what = ", ".join(f"{esc(k)} {vv}" for k, vv in ev.get("vital", {}).items())
        else:
            continue
        al = "".join(f"<div class='al'>Guardian {esc(a['rule_id'])}: {esc(a['message'])} &rarr; "
                     f"<b>{esc(((a.get('response') or {}).get('action') or 'no answer').replace('_', ' '))}</b>"
                     f"{(' &mdash; ' + esc(a['response']['reason'])) if a.get('response') and a['response'].get('reason') else ''}</div>"
                     for a in x.get("alerts", []))
        rows.append(f"<tr class='{esc(x['status'])}'><td>{t}</td><td>{esc(ev['event_type'])}</td><td>{what}{al}</td>"
                    f"<td>{esc(x['status'])}</td></tr>")
    anchor = v.get("anchor") or {}
    return f"""<!doctype html><html><head><meta charset="utf-8"><title>Handover {esc(v['incident'] or '')}</title>
<style>body{{font:14px/1.45 -apple-system,Segoe UI,Roboto,sans-serif;margin:32px;color:#111}}
h1{{font-size:20px;margin:0}}.sub{{color:#555;margin:4px 0 18px}}table{{width:100%;border-collapse:collapse}}
th,td{{text-align:left;border-bottom:1px solid #ddd;padding:6px 8px;vertical-align:top}}th{{background:#f3f4f6}}
.cancelled td{{color:#888;text-decoration:line-through}}.al{{color:#9a3412;font-size:12px;margin-top:3px}}
.box{{border:1px solid #ddd;border-radius:8px;padding:10px 14px;margin:14px 0}}
.ok{{color:#15803d;font-weight:700}}.bad{{color:#b91c1c;font-weight:700}}code{{font-size:11px;word-break:break-all}}
@media print{{button{{display:none}}}}</style></head><body>
<button onclick="print()" style="float:right">Print / Save PDF</button>
<h1>Ambulance handover &mdash; {esc(v['incident'] or '')}</h1>
<div class="sub">Device {esc(v['device']['device_id'])} &middot; generated from the signed Black Box record</div>
<div class="box"><b>Patient</b>: age {esc(str(p.get('age_years') or '?'))} &middot; weight {esc(str(p.get('weight_kg') or '?'))} kg
&middot; allergies {esc(', '.join(p.get('allergies') or []) or 'none recorded')}
{(' &middot; flags ' + esc(', '.join(p.get('flags')))) if p.get('flags') else ''}</div>
<table><tr><th>Time</th><th>Type</th><th>Detail</th><th>Status</th></tr>{''.join(rows)}</table>
<div class="box">Record integrity: <span class="{'ok' if integ['ok'] else 'bad'}">{'INTACT' if integ['ok'] else 'TAMPERED (entries ' + esc(str(integ['bad'])) + ')'}</span>
&middot; {integ['total']} signed entries<br>Chain head: <code>#{integ['head_seq']} {esc(integ['head_hash'] or '')}</code><br>
Server anchor: <code>#{anchor.get('seq', '-')} {esc(anchor.get('entry_hash', ''))} at {esc(anchor.get('server_time', ''))}</code></div>
<p style="color:#777;font-size:12px">Decision-support record. Not for clinical use (hackathon prototype).</p>
</body></html>"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=8000)
    ap.add_argument("--data", default=str(ROOT / "server_data"))
    a = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(a.data), host=a.host, port=a.port, log_level="warning")


if __name__ == "__main__":
    main()