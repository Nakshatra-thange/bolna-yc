"""
Black Box - device sync client (run when the ambulance has signal; safe to run any time, repeatedly).

  python -m sync.client push                               # -> http://127.0.0.1:8000
  python -m sync.client push --server http://192.168.1.20:8000
  python -m sync.client status

What it does:
  1. registers the device (public key pinned on first contact; server key pinned locally on first contact)
  2. sends every chain entry the server doesn't have yet + the ENCRYPTED voice clips they reference
  3. receives an anchor receipt signed by the server, checks the signature, saves it to runs/anchors.jsonl
     and appends it to the chain as an 'anchor' entry (so the device's own record proves it synced)
Offline queue = "entries after the server's head". Nothing to manage: if it fails, run it again later.
"""
from __future__ import annotations

import argparse
import base64
import json
import sqlite3
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from cryptography.exceptions import InvalidSignature  # noqa: E402

from audit.chain import AuditChain, canonical  # noqa: E402
from audit.keys import KeyStore, load_public_key  # noqa: E402


class SyncClient:
    def __init__(self, server: str, db: Path, keys: KeyStore, opener=None):
        self.server = server.rstrip("/")
        self.db, self.keys = Path(db), keys
        self.root = self.db.parent
        self._open = opener   # tests inject a fake transport

    # ------------------------------------------------------------- http
    def _req(self, method: str, path: str, body: dict | None = None) -> dict:
        if self._open:
            return self._open(method, path, body)
        data = json.dumps(body).encode() if body is not None else None
        req = urllib.request.Request(self.server + path, data=data, method=method,
                                     headers={"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(req, timeout=60) as r:
                return json.loads(r.read())
        except urllib.error.HTTPError as e:
            raise RuntimeError(f"server said {e.code}: {json.loads(e.read() or b'{}').get('detail', e.reason)}")

    # ------------------------------------------------------------- steps
    def register(self) -> str:
        r = self._req("POST", "/api/devices/register", {"device_id": self.keys.device_id,
                                                        "public_key": self.keys.public_key_hex,
                                                        "audio_key": self.keys.audio_key.hex()})
        pin = self.root / "server_pubkey.txt"
        if pin.exists() and pin.read_text().strip() != r["server_public_key"]:
            raise RuntimeError("SERVER KEY CHANGED since last sync - refusing (possible impostor server)")
        pin.write_text(r["server_public_key"] + "\n")
        return r["server_public_key"]

    def push(self) -> dict:
        server_key = self.register()
        head = self._req("GET", f"/api/devices/{self.keys.device_id}/head")["seq"]
        con = sqlite3.connect(self.db)
        rows = con.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain WHERE seq>? ORDER BY seq",
                           (head,)).fetchall()
        con.close()
        entries = [dict(zip(("seq", "entry_hash", "prev_hash", "body", "sig"), r)) for r in rows]
        clips = {}
        for e in entries:
            p = json.loads(e["body"]).get("payload", {})
            f = self.root / p.get("clip_path", "")
            if p.get("clip_path") and f.exists():
                clips[p["clip_path"]] = base64.b64encode(f.read_bytes()).decode()   # still AES-GCM encrypted
        res = self._req("POST", "/api/sync", {"device_id": self.keys.device_id, "entries": entries, "clips": clips})
        receipt = res.get("receipt")
        if receipt:
            body = {k: receipt[k] for k in ("device_id", "seq", "entry_hash", "server_time")}
            try:
                load_public_key(server_key).verify(base64.b64decode(receipt["sig"]), canonical(body))
            except InvalidSignature:
                raise RuntimeError("anchor receipt signature INVALID - not from the pinned server")
            with open(self.root / "anchors.jsonl", "a") as f:
                f.write(json.dumps(receipt) + "\n")
            if any(json.loads(e["body"]).get("kind") != "anchor" for e in entries):   # don't anchor anchors forever
                AuditChain(self.db, self.keys).append("anchor", body | {"server_sig": receipt["sig"]}, incident_id=None)
        return {"sent": len(entries), "accepted": res["accepted"], "clips": len(clips), "receipt": receipt}

    def last_receipt(self) -> dict | None:
        p = self.root / "anchors.jsonl"
        if not p.exists():
            return None
        lines = [l for l in p.read_text().splitlines() if l.strip()]
        return json.loads(lines[-1]) if lines else None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["push", "status"])
    ap.add_argument("--server", default="http://127.0.0.1:8000")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "blackbox.db")
    ap.add_argument("--keys", default=None)
    a = ap.parse_args()
    c = SyncClient(a.server, a.db, KeyStore(a.keys))
    if a.cmd == "status":
        r = c.last_receipt()
        print("never synced" if not r else f"last anchor: #{r['seq']} {r['entry_hash'][:16]}... at {r['server_time']}\n"
              f"verify locally against it:  python -m audit.cli verify --expect-head {r['seq']}:{r['entry_hash']}")
        return
    try:
        r = c.push()
    except (RuntimeError, urllib.error.URLError) as exc:
        sys.exit(f"sync failed (data is safe on the device, try again later): {exc}")
    rc = r["receipt"]
    print(f"synced {r['accepted']} new entries ({r['sent']} sent, {r['clips']} encrypted clips)")
    if rc:
        print(f"server anchor receipt: #{rc['seq']} {rc['entry_hash'][:16]}... at {rc['server_time']}  (signature OK)")


if __name__ == "__main__":
    main()