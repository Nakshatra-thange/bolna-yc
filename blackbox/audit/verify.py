"""
Black Box - chain verifier + "effective record" view.

Checks, per entry:
  hash      SHA-256(body) == stored entry_hash            -> catches edited content (dose, time, text)
  sig       Ed25519(entry_hash) valid for PINNED key      -> catches re-hashed forgeries without the device key
  link      body.prev_hash == previous entry's hash       -> catches deleted / inserted / reordered entries
  seq       no gaps                                       -> says HOW MANY entries are missing
  clock     wall clock never runs backwards, and agrees with the monotonic clock inside one boot
  audio     decrypt clip, SHA-256 == hash in the chain    -> catches swapped / edited voice clips
  head      optional expected (seq, hash) from the server -> catches deleted entries at the END (truncation)

The pinned public key must come from OUTSIDE the database (server registry / file). Trusting the
key inside the genesis entry would let an attacker replace it and re-sign everything.
"""
from __future__ import annotations

import base64
import hashlib
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

from cryptography.exceptions import InvalidSignature
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

from audit.chain import GENESIS_PREV, Entry, sha256_hex
from audit.vault import decrypt_file

CLOCK_JUMP_S = 120


@dataclass
class EntryResult:
    seq: int
    kind: str
    problems: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


@dataclass
class Report:
    results: list[EntryResult]
    chain_problems: list[str]
    head_seq: Optional[int]
    head_hash: Optional[str]

    @property
    def ok(self) -> bool:
        return not self.chain_problems and all(r.ok for r in self.results)

    @property
    def first_bad_seq(self) -> Optional[int]:
        return next((r.seq for r in self.results if not r.ok), None)


def verify_chain(entries: Iterable[Entry], pubkey: Ed25519PublicKey,
                 audio_root: Optional[Path] = None, audio_key: Optional[bytes] = None,
                 expected_head: Optional[tuple[int, str]] = None) -> Report:
    entries = list(entries)
    results: list[EntryResult] = []
    chain_problems: list[str] = []
    prev: Optional[Entry] = None
    prev_body: Optional[dict] = None
    pinned_hex = pubkey.public_bytes_raw().hex()

    for e in entries:
        try:
            body = json.loads(e.body)
        except json.JSONDecodeError:
            results.append(EntryResult(e.seq, "?", ["body is not valid JSON"]))
            prev, prev_body = e, None
            continue
        r = EntryResult(e.seq, body.get("kind", "?"))

        # 1. content
        if sha256_hex(e.body.encode("utf-8")) != e.entry_hash:
            r.problems.append("CONTENT ALTERED: body does not match its hash")
        if body.get("seq") != e.seq:
            r.problems.append(f"sequence number altered (body says {body.get('seq')})")
        # 2. signature
        try:
            pubkey.verify(base64.b64decode(e.sig), bytes.fromhex(e.entry_hash))
        except (InvalidSignature, ValueError):
            r.problems.append("SIGNATURE INVALID: not signed by this device's key")
        # 3. link + gaps
        expected_prev = GENESIS_PREV if prev is None else prev.entry_hash
        if body.get("prev_hash") != expected_prev or e.prev_hash != body.get("prev_hash"):
            if prev is not None and e.seq > prev.seq + 1:
                r.problems.append(f"ENTRIES DELETED: {e.seq - prev.seq - 1} missing before this one "
                                  f"(seq {prev.seq + 1}..{e.seq - 1})")
            elif prev is None and e.seq != 0:
                r.problems.append(f"ENTRIES DELETED: chain starts at seq {e.seq}, genesis missing")
            else:
                r.problems.append("CHAIN BROKEN: previous entry was removed, inserted or reordered")
        # 4. genesis key pinning
        if r.kind == "genesis" and body.get("payload", {}).get("public_key") != pinned_hex:
            r.problems.append("GENESIS KEY MISMATCH: chain claims a different device key")
        # 5. clocks
        if prev_body and body.get("ts_wall") and prev_body.get("ts_wall"):
            dw = (datetime.fromisoformat(body["ts_wall"]) - datetime.fromisoformat(prev_body["ts_wall"])).total_seconds()
            if dw < -1:
                r.warnings.append(f"device clock went BACKWARDS by {-dw:.0f}s")
            if body.get("boot_id") == prev_body.get("boot_id"):
                dm = (body["ts_mono_ns"] - prev_body["ts_mono_ns"]) / 1e9
                if abs(dw - dm) > CLOCK_JUMP_S:
                    r.warnings.append(f"wall clock jumped {dw - dm:+.0f}s vs monotonic clock (clock changed?)")
        # 6. audio
        p = body.get("payload", {})
        if r.kind == "utterance" and audio_root is not None and p.get("clip_path") and p.get("audio_sha256"):
            clip = audio_root / p["clip_path"]
            if not clip.exists():
                r.warnings.append("voice clip missing")
            elif audio_key is not None:
                try:
                    plain = decrypt_file(clip, audio_key, aad=p["clip_path"])
                    if hashlib.sha256(plain).hexdigest() != p["audio_sha256"]:
                        r.problems.append("AUDIO ALTERED: clip does not match hash in chain")
                except Exception:  # noqa: BLE001 - wrong key / tag failure / garbage
                    r.problems.append("AUDIO ALTERED: clip fails authenticated decryption")
        results.append(r)
        prev, prev_body = e, body

    head_seq, head_hash = (prev.seq, prev.entry_hash) if prev else (None, None)
    if expected_head:
        problem = check_anchor(entries, expected_head)
        if problem:
            chain_problems.append(problem)
    return Report(results, chain_problems, head_seq, head_hash)


def check_anchor(entries: list[Entry], expected_head: tuple[int, str]) -> Optional[str]:
    """The entry the server anchored must still be there with the same hash."""
    s, h = expected_head
    e = next((x for x in entries if x.seq == s), None)
    if e is None:
        return f"TAIL TRUNCATED: anchored seq {s} is gone"
    if e.entry_hash != h:
        return f"HISTORY REWRITTEN: anchored seq {s} now has a different hash"
    return None


def materialize(entries: Iterable[Entry], incident_id: Optional[str] = None) -> list[dict]:
    """What the chain says NOW: each event with its corrections applied and its confirmation status.
    The original values stay visible in 'history'."""
    events: dict[str, dict] = {}
    order: list[str] = []
    for e in entries:
        d = e.data
        if incident_id and d.get("incident_id") != incident_id:
            continue
        p = d["payload"]
        if d["kind"] == "event":
            ev = dict(p["event"])
            needs = ev.get("needs_confirmation", False)
            events[p["event_id"]] = {"event_id": p["event_id"], "seq": e.seq, "event": ev,
                                     "status": "pending" if needs else "recorded",
                                     "reasons": ev.get("confirm_reasons", []), "history": [], "alerts": []}
            order.append(p["event_id"])
        elif d["kind"] == "confirmation" and p.get("event_id") in events:
            x = events[p["event_id"]]
            x["status"] = p["action"]
            x["history"].append({"seq": e.seq, "what": p["action"], "said": p.get("said", ""), "at": d["ts_wall"]})
        elif d["kind"] == "alert" and p.get("event_id") in events:
            events[p["event_id"]]["alerts"].append({"alert_id": p["alert_id"], "seq": e.seq, "rule_id": p["rule_id"],
                                                    "severity": p["severity"], "message": p["message"],
                                                    "response": None})
        elif d["kind"] == "alert_response" and p.get("event_id") in events:
            for al in events[p["event_id"]]["alerts"]:
                if al["alert_id"] == p.get("alert_id"):
                    al["response"] = {"action": p["action"], "reason": p.get("reason"), "said": p.get("said"),
                                      "seq": e.seq, "at": d["ts_wall"]}
        elif d["kind"] == "correction" and p.get("event_id") in events:
            x = events[p["event_id"]]
            before = {k: x["event"].get(k) for k in p["changes"]}
            x["event"].update(p["changes"])
            x["history"].append({"seq": e.seq, "what": "corrected", "before": before, "after": p["changes"],
                                 "reason": p.get("reason"), "by": p.get("by"), "at": d["ts_wall"]})
    return [events[i] for i in order]


def patient_context(entries: Iterable[Entry], incident_id: str) -> dict:
    """Latest patient context (weight, age, allergies, flags) recorded for an incident."""
    cur: dict = {}
    for e in entries:
        d = e.data
        if d.get("incident_id") == incident_id and d["kind"] == "patient":
            cur = d["payload"].get("current", cur)
    return cur