"""
Black Box - the audit chain.

Every entry:
    body = canonical JSON {v, seq, prev_hash, device_id, boot_id, ts_wall, ts_mono_ns, incident_id, kind, payload}
    entry_hash = SHA-256(body)          -> changing ANY byte of the body changes the hash
    sig = Ed25519(entry_hash)           -> only the device key can produce it
    prev_hash links to the entry before -> deleting / reordering breaks the link

Kinds:
    genesis       first entry: device public key
    utterance     raw ASR text, corrected text, every auto-correction, SHA-256 of the voice clip
    event         one extracted medication / procedure / vital, pointing at its utterance
    confirmation  medic said confirm / cancel (or it timed out: unconfirmed), pointing at the event
    correction    an amendment to an event (new values + reason). The original is never changed.
    alert, alert_response  (Phase 4 Guardian)

Storage: SQLite with triggers that refuse UPDATE / DELETE. The triggers are a speed bump, not the
security: anyone can drop them. The hashes and signatures are the security - see audit/verify.py.
"""
from __future__ import annotations

import hashlib
import json
import sqlite3
import time
import uuid
from base64 import b64encode
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Optional

from audit.keys import KeyStore

VERSION = 1
GENESIS_PREV = "0" * 64

SCHEMA = """
CREATE TABLE IF NOT EXISTS chain (
    seq        INTEGER PRIMARY KEY,
    entry_hash TEXT NOT NULL UNIQUE,
    prev_hash  TEXT NOT NULL,
    body       TEXT NOT NULL,
    sig        TEXT NOT NULL
);
CREATE TRIGGER IF NOT EXISTS chain_no_update BEFORE UPDATE ON chain
BEGIN SELECT RAISE(ABORT, 'audit chain is append-only'); END;
CREATE TRIGGER IF NOT EXISTS chain_no_delete BEFORE DELETE ON chain
BEGIN SELECT RAISE(ABORT, 'audit chain is append-only'); END;
"""


def canonical(obj) -> bytes:
    """Deterministic JSON bytes: sorted keys, no whitespace, UTF-8, no NaN.
    (RFC 8785-style; Python's float repr is the shortest round-trip form.)"""
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=False,
                      allow_nan=False).encode("utf-8")


def sha256_hex(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


def _plain(obj):
    """Make a payload JSON-native (datetimes -> ISO strings) BEFORE hashing, so it hashes the same forever."""
    return json.loads(json.dumps(obj, default=lambda o: o.isoformat() if hasattr(o, "isoformat") else str(o)))


@dataclass
class Entry:
    seq: int
    entry_hash: str
    prev_hash: str
    body: str
    sig: str

    @property
    def data(self) -> dict:
        return json.loads(self.body)


class AuditChain:
    def __init__(self, db_path: str | Path, keys: KeyStore):
        self.path = Path(db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.keys = keys
        self.db = sqlite3.connect(self.path, isolation_level=None)   # autocommit; we use explicit txns
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA synchronous=FULL")                   # survive power loss in the ambulance
        self.db.executescript(SCHEMA)
        self.boot_id = uuid.uuid4().hex[:12]                         # monotonic clock is only comparable within a boot
        if self.head() is None:
            self.append("genesis", {"public_key": keys.public_key_hex, "device_id": keys.device_id,
                                    "schema": VERSION}, incident_id=None)

    # -------------------------------------------------------------
    def head(self) -> Optional[Entry]:
        r = self.db.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain ORDER BY seq DESC LIMIT 1").fetchone()
        return Entry(*r) if r else None

    def entries(self, incident_id: Optional[str] = None) -> Iterator[Entry]:
        for r in self.db.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain ORDER BY seq"):
            e = Entry(*r)
            if incident_id is None or e.data.get("incident_id") in (incident_id, None):
                yield e

    def append(self, kind: str, payload: dict, incident_id: Optional[str]) -> Entry:
        self.db.execute("BEGIN IMMEDIATE")          # one writer at a time; seq and prev_hash can't race
        try:
            h = self.head()
            seq = 0 if h is None else h.seq + 1
            prev = GENESIS_PREV if h is None else h.entry_hash
            body = {
                "v": VERSION, "seq": seq, "prev_hash": prev,
                "device_id": self.keys.device_id, "boot_id": self.boot_id,
                "ts_wall": datetime.now(timezone.utc).isoformat(timespec="milliseconds"),
                "ts_mono_ns": time.monotonic_ns(),
                "incident_id": incident_id, "kind": kind, "payload": _plain(payload),
            }
            raw = canonical(body)
            eh = sha256_hex(raw)
            sig = b64encode(self.keys.sign(bytes.fromhex(eh))).decode()
            self.db.execute("INSERT INTO chain (seq, entry_hash, prev_hash, body, sig) VALUES (?,?,?,?,?)",
                            (seq, eh, prev, raw.decode("utf-8"), sig))
            self.db.execute("COMMIT")
        except Exception:
            self.db.execute("ROLLBACK")
            raise
        return Entry(seq, eh, prev, raw.decode("utf-8"), sig)

    # -------------------------------------------------------------
    def find_event(self, event_id: str) -> Optional[Entry]:
        for e in self.db.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain "
                                 "WHERE body LIKE ? ORDER BY seq DESC", (f'%"event_id":"{event_id}"%',)):
            ent = Entry(*e)
            d = ent.data
            if d["kind"] == "event" and d["payload"].get("event_id") == event_id:
                return ent
        return None

    def correct(self, event_id: str, changes: dict, reason: str, by: str) -> Entry:
        ev = self.find_event(event_id)
        if ev is None:
            raise KeyError(f"no event {event_id}")
        return self.append("correction", {"event_id": event_id, "event_ref": ev.entry_hash,
                                          "changes": changes, "reason": reason, "by": by},
                           incident_id=ev.data["incident_id"])

    def close(self):
        self.db.close()