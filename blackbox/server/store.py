"""
Black Box server - storage + sync acceptance + anchor receipts.

Sync rules (each one is a sentence for the judges):
  * a device's public key is pinned the first time it registers; a different key later is refused
  * a batch must continue the stored chain exactly (seq = head+1, prev_hash = stored head hash)
  * re-sending entries the server already has is fine (idempotent); the SAME seq with a DIFFERENT
    hash is a FORK - refused and kept as evidence
  * every new entry is verified (hash, Ed25519 signature with the pinned key, links) before storing
  * after a good sync the server signs an ANCHOR RECEIPT {device, seq, hash, server_time} with its own key.
    Once anchored, deleting or rewriting those entries on the device is detectable forever.
"""
from __future__ import annotations

import base64
import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from audit.chain import Entry, canonical
from audit.keys import KeyStore, load_public_key
from audit.verify import verify_chain

SCHEMA = """
CREATE TABLE IF NOT EXISTS devices (device_id TEXT PRIMARY KEY, public_key TEXT NOT NULL,
                                    audio_key TEXT, registered_at TEXT NOT NULL);
CREATE TABLE IF NOT EXISTS entries (device_id TEXT, seq INTEGER, entry_hash TEXT, prev_hash TEXT, body TEXT,
                                    sig TEXT, received_at TEXT, PRIMARY KEY (device_id, seq));
CREATE TABLE IF NOT EXISTS originals (device_id TEXT, seq INTEGER, entry_hash TEXT, prev_hash TEXT, body TEXT,
                                      sig TEXT, PRIMARY KEY (device_id, seq));
CREATE TABLE IF NOT EXISTS anchors (device_id TEXT, seq INTEGER, entry_hash TEXT, server_time TEXT, sig TEXT);
CREATE TABLE IF NOT EXISTS forks (device_id TEXT, seq INTEGER, stored_hash TEXT, offered_hash TEXT,
                                  body TEXT, seen_at TEXT);
"""


class SyncError(Exception):
    def __init__(self, status: int, detail: str):
        super().__init__(detail)
        self.status, self.detail = status, detail


def now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="milliseconds")


class ServerStore:
    def __init__(self, data_dir: str | Path):
        self.dir = Path(data_dir)
        self.dir.mkdir(parents=True, exist_ok=True)
        self.clips = self.dir / "clips"
        self.db = sqlite3.connect(self.dir / "server.db", check_same_thread=False, isolation_level=None)
        self.db.executescript(SCHEMA)
        self.keys = KeyStore(self.dir / "server_keys")   # the server's OWN signing key for anchor receipts

    # ------------------------------------------------------------- devices
    def register(self, device_id: str, public_key: str, audio_key: Optional[str]) -> dict:
        row = self.db.execute("SELECT public_key FROM devices WHERE device_id=?", (device_id,)).fetchone()
        if row and row[0] != public_key:
            raise SyncError(409, "device already registered with a DIFFERENT public key - refusing")
        if not row:
            self.db.execute("INSERT INTO devices VALUES (?,?,?,?)", (device_id, public_key, audio_key, now()))
        return {"device_id": device_id, "pinned": True, "server_public_key": self.keys.public_key_hex}

    def device(self, device_id: str) -> Optional[dict]:
        r = self.db.execute("SELECT device_id, public_key, audio_key, registered_at FROM devices WHERE device_id=?",
                            (device_id,)).fetchone()
        return dict(zip(("device_id", "public_key", "audio_key", "registered_at"), r)) if r else None

    def devices(self) -> list[dict]:
        return [self.device(r[0]) for r in self.db.execute("SELECT device_id FROM devices ORDER BY registered_at")]

    # ------------------------------------------------------------- entries
    def entries(self, device_id: str) -> list[Entry]:
        return [Entry(*r) for r in self.db.execute(
            "SELECT seq, entry_hash, prev_hash, body, sig FROM entries WHERE device_id=? ORDER BY seq", (device_id,))]

    def head(self, device_id: str) -> tuple[int, Optional[str]]:
        r = self.db.execute("SELECT seq, entry_hash FROM entries WHERE device_id=? ORDER BY seq DESC LIMIT 1",
                            (device_id,)).fetchone()
        return (r[0], r[1]) if r else (-1, None)

    def sync(self, device_id: str, batch: list[dict], clips: dict[str, str]) -> dict:
        dev = self.device(device_id)
        if not dev:
            raise SyncError(404, "unknown device - register first")
        stored = {e.seq: e for e in self.entries(device_id)}
        new: list[Entry] = []
        for b in sorted(batch, key=lambda x: x["seq"]):
            e = Entry(b["seq"], b["entry_hash"], b["prev_hash"], b["body"], b["sig"])
            old = stored.get(e.seq)
            if old is not None:
                if old.entry_hash != e.entry_hash:
                    self.db.execute("INSERT INTO forks VALUES (?,?,?,?,?,?)",
                                    (device_id, e.seq, old.entry_hash, e.entry_hash, e.body, now()))
                    raise SyncError(409, f"FORK at seq {e.seq}: device now has a different entry than it "
                                         f"previously synced - history was rewritten on the device")
                continue  # idempotent re-send
            new.append(e)
        if new:
            head_seq, head_hash = self.head(device_id)
            if new[0].seq != head_seq + 1 or (head_hash and new[0].prev_hash != head_hash):
                raise SyncError(409, f"batch does not continue the stored chain (server head #{head_seq})")
            pub = load_public_key(dev["public_key"])
            rep = verify_chain(list(stored.values()) + new, pub)
            bad = [r for r in rep.results if r.seq >= new[0].seq and not r.ok]
            if bad:
                raise SyncError(422, f"rejected: entry #{bad[0].seq}: {'; '.join(bad[0].problems)}")
            self.db.execute("BEGIN")
            for e in new:
                self.db.execute("INSERT INTO entries VALUES (?,?,?,?,?,?,?)",
                                (device_id, e.seq, e.entry_hash, e.prev_hash, e.body, e.sig, now()))
                self.db.execute("INSERT OR REPLACE INTO originals VALUES (?,?,?,?,?,?)",
                                (device_id, e.seq, e.entry_hash, e.prev_hash, e.body, e.sig))
            self.db.execute("COMMIT")
        for rel, b64 in clips.items():   # encrypted clips only; path is checked to stay inside our folder
            p = (self.clips / device_id / rel).resolve()
            if str(p).startswith(str((self.clips / device_id).resolve())):
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_bytes(base64.b64decode(b64))
        return {"accepted": len(new), "receipt": self.anchor(device_id)}

    # ------------------------------------------------------------- anchors
    def anchor(self, device_id: str) -> Optional[dict]:
        seq, h = self.head(device_id)
        if h is None:
            return None
        last = self.last_anchor(device_id)
        if last and last["seq"] == seq and last["entry_hash"] == h:
            return last
        body = {"device_id": device_id, "seq": seq, "entry_hash": h, "server_time": now()}
        sig = base64.b64encode(self.keys.sign(canonical(body))).decode()
        self.db.execute("INSERT INTO anchors VALUES (?,?,?,?,?)", (device_id, seq, h, body["server_time"], sig))
        return dict(body, sig=sig, server_public_key=self.keys.public_key_hex)

    def last_anchor(self, device_id: str) -> Optional[dict]:
        r = self.db.execute("SELECT seq, entry_hash, server_time, sig FROM anchors WHERE device_id=? "
                            "ORDER BY rowid DESC LIMIT 1", (device_id,)).fetchone()
        if not r:
            return None
        return {"device_id": device_id, "seq": r[0], "entry_hash": r[1], "server_time": r[2], "sig": r[3],
                "server_public_key": self.keys.public_key_hex}

    def forks(self, device_id: str) -> list[dict]:
        return [dict(zip(("seq", "stored_hash", "offered_hash", "seen_at"), r)) for r in self.db.execute(
            "SELECT seq, stored_hash, offered_hash, seen_at FROM forks WHERE device_id=?", (device_id,))]

    # ------------------------------------------------------------- demo: insider tampering on the SERVER copy
    def tamper(self, device_id: str, kind: str, seq: int, value=None) -> str:
        row = self.db.execute("SELECT body FROM entries WHERE device_id=? AND seq=?", (device_id, seq)).fetchone()
        if not row:
            raise SyncError(404, f"no entry #{seq}")
        d = json.loads(row[0])
        if kind == "edit_dose":
            d["payload"]["event"]["dose"] = float(value)
            self.db.execute("UPDATE entries SET body=? WHERE device_id=? AND seq=?",
                            (canonical(d).decode(), device_id, seq))
            return f"changed dose in #{seq} to {value}"
        if kind == "backdate":
            from datetime import timedelta
            t = datetime.fromisoformat(d["ts_wall"]) - timedelta(minutes=float(value or 15))
            d["ts_wall"] = t.isoformat(timespec="milliseconds")
            ev = d["payload"].get("event")
            if ev and ev.get("time_said"):
                ev["time_said"] = (datetime.fromisoformat(ev["time_said"]) - timedelta(minutes=float(value or 15))).isoformat()
            self.db.execute("UPDATE entries SET body=? WHERE device_id=? AND seq=?",
                            (canonical(d).decode(), device_id, seq))
            return f"moved #{seq} back {value or 15} minutes"
        if kind == "delete":
            self.db.execute("DELETE FROM entries WHERE device_id=? AND seq=?", (device_id, seq))
            return f"deleted #{seq}"
        if kind == "edit_text":
            d["payload"]["text"] = str(value)
            self.db.execute("UPDATE entries SET body=? WHERE device_id=? AND seq=?",
                            (canonical(d).decode(), device_id, seq))
            return f"rewrote transcript of #{seq}"
        raise SyncError(400, f"unknown attack {kind}")

    def restore(self, device_id: str) -> int:
        self.db.execute("DELETE FROM entries WHERE device_id=?", (device_id,))
        n = self.db.execute("INSERT INTO entries SELECT device_id, seq, entry_hash, prev_hash, body, sig, ? "
                            "FROM originals WHERE device_id=?", (now(), device_id)).rowcount
        return n