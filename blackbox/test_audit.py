"""Tests for Phase 3: signed hash chain, encrypted audio, verifier, effective record.  python -m pytest -q"""
import hashlib
import json
import sqlite3
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from audit.chain import AuditChain, Entry, canonical, sha256_hex
from audit.keys import KeyStore
from audit.vault import decrypt_file, encrypt_file
from audit.verify import check_anchor, materialize, verify_chain


@pytest.fixture
def chain(tmp_path):
    c = AuditChain(tmp_path / "bb.db", KeyStore(tmp_path / "keys"))
    u = c.append("utterance", {"text": "fentanyl 50 mcg IV"}, "INC-1")
    c.append("event", {"event_id": "E1", "utterance_ref": u.entry_hash,
                       "event": {"event_type": "medication", "drug": "fentanyl", "dose": 50.0, "unit": "mcg",
                                 "route": "IV", "needs_confirmation": True}}, "INC-1")
    c.append("confirmation", {"event_id": "E1", "action": "confirmed", "said": "confirm"}, "INC-1")
    c.append("event", {"event_id": "E2", "event": {"event_type": "medication", "drug": "morphine", "dose": 4.0,
                                                   "unit": "mg", "route": "IV"}}, "INC-1")
    return c


def rows(c):
    return list(c.entries())


def tampered(c, sql, *args):
    """Run raw SQL on the DB like an attacker would (dropping the append-only triggers first)."""
    con = sqlite3.connect(c.path)
    con.executescript("DROP TRIGGER IF EXISTS chain_no_update; DROP TRIGGER IF EXISTS chain_no_delete;")
    con.execute(sql, args)
    con.commit()
    con.close()
    con = sqlite3.connect(c.path)
    out = [Entry(*r) for r in con.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain ORDER BY seq")]
    con.close()
    return out


def problems(rep):
    return {r.seq: " ".join(r.problems) for r in rep.results if r.problems}


# ------------------------------------------------------------------ basics
def test_canonical_is_deterministic_and_key_order_free():
    assert canonical({"b": 1, "a": [0.4, "é"]}) == canonical({"a": [0.4, "é"], "b": 1}) == '{"a":[0.4,"é"],"b":1}'.encode()


def test_fresh_chain_verifies(chain):
    rep = verify_chain(rows(chain), chain.keys.public_key)
    assert rep.ok and rep.head_seq == 4 and [r.kind for r in rep.results][0] == "genesis"


def test_triggers_block_update_and_delete(chain):
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        chain.db.execute("UPDATE chain SET body='x' WHERE seq=1")
    with pytest.raises(sqlite3.IntegrityError, match="append-only"):
        chain.db.execute("DELETE FROM chain WHERE seq=1")


def test_reopen_continues_same_chain(tmp_path):
    keys = KeyStore(tmp_path / "keys")
    AuditChain(tmp_path / "bb.db", keys).append("utterance", {"text": "a"}, "I")
    c2 = AuditChain(tmp_path / "bb.db", KeyStore(tmp_path / "keys"))
    c2.append("utterance", {"text": "b"}, "I")
    rep = verify_chain(rows(c2), keys.public_key)
    assert rep.ok and rep.head_seq == 2      # one genesis only


def test_keys_are_private_files(tmp_path):
    k = KeyStore(tmp_path / "keys")
    assert oct((tmp_path / "keys" / "signing.key").stat().st_mode)[-3:] == "600"
    assert k.device_id.startswith("BB-") and len(k.public_key_hex) == 64


# ------------------------------------------------------------------ attacks
def test_edited_dose_detected(chain):
    body = rows(chain)[2].body.replace('"dose":50.0', '"dose":500.0')
    rep = verify_chain(tampered(chain, "UPDATE chain SET body=? WHERE seq=2", body), chain.keys.public_key)
    assert problems(rep) == {2: "CONTENT ALTERED: body does not match its hash"}


def test_rehashed_forgery_fails_signature(chain):
    body = rows(chain)[2].body.replace('"dose":50.0', '"dose":500.0')
    rep = verify_chain(tampered(chain, "UPDATE chain SET body=?, entry_hash=? WHERE seq=2", body,
                                sha256_hex(body.encode())), chain.keys.public_key)
    assert "SIGNATURE INVALID" in problems(rep)[2] and "CHAIN BROKEN" in problems(rep)[3]


def test_deleted_entry_detected_with_count(chain):
    rep = verify_chain(tampered(chain, "DELETE FROM chain WHERE seq=3"), chain.keys.public_key)
    assert problems(rep) == {4: "ENTRIES DELETED: 1 missing before this one (seq 3..3)"}


def test_reordered_entries_detected(chain):
    r = rows(chain)
    con_rows = tampered(chain, "UPDATE chain SET body=?, entry_hash=?, prev_hash=?, sig=? WHERE seq=3",
                        r[4].body, r[4].entry_hash + "x", r[4].prev_hash, r[4].sig)
    rep = verify_chain(con_rows, chain.keys.public_key)
    assert 3 in problems(rep)


def test_truncated_tail_needs_anchor(chain):
    head = (chain.head().seq, chain.head().entry_hash)
    cut = tampered(chain, "DELETE FROM chain WHERE seq>2")
    assert verify_chain(cut, chain.keys.public_key).ok            # invisible on its own ...
    rep = verify_chain(cut, chain.keys.public_key, expected_head=head)
    assert not rep.ok and "TAIL TRUNCATED" in rep.chain_problems[0]   # ... caught by the server anchor


def test_anchor_detects_rewritten_history(chain):
    assert check_anchor(rows(chain), (2, "f" * 64)).startswith("HISTORY REWRITTEN")


def test_attacker_key_cannot_pass(chain, tmp_path):
    other = KeyStore(tmp_path / "attacker")
    rep = verify_chain(rows(chain), other.public_key)
    assert "GENESIS KEY MISMATCH" in problems(rep)[0] and all(not r.ok for r in rep.results)


def test_backwards_clock_is_flagged(tmp_path):
    c = AuditChain(tmp_path / "bb.db", KeyStore(tmp_path / "keys"))
    c.append("utterance", {"text": "a"}, "I")
    r = rows(c)
    d = json.loads(r[1].body)
    d["ts_wall"] = (datetime.fromisoformat(d["ts_wall"]) - timedelta(minutes=30)).isoformat(timespec="milliseconds")
    rep = verify_chain(tampered(c, "UPDATE chain SET body=? WHERE seq=1", canonical(d).decode()), c.keys.public_key)
    assert "CONTENT ALTERED" in problems(rep)[1]
    assert any("BACKWARDS" in w for w in rep.results[1].warnings)


# ------------------------------------------------------------------ audio
def test_audio_encrypted_and_verified(tmp_path):
    keys = KeyStore(tmp_path / "keys")
    c = AuditChain(tmp_path / "bb.db", keys)
    clips = tmp_path / "I" / "clips"
    clips.mkdir(parents=True)
    for n, content in (("0001.wav", b"RIFF-voice-one"), ("0002.wav", b"RIFF-voice-two")):
        p = clips / n
        p.write_bytes(content)
        rel = f"I/clips/{n}.enc"
        encrypt_file(p, keys.audio_key, aad=rel)
        c.append("utterance", {"clip_path": rel, "audio_sha256": hashlib.sha256(content).hexdigest()}, "I")
    assert not (clips / "0001.wav").exists()                                   # no plaintext left
    assert decrypt_file(clips / "0001.wav.enc", keys.audio_key, "I/clips/0001.wav.enc") == b"RIFF-voice-one"
    assert verify_chain(rows(c), keys.public_key, tmp_path, keys.audio_key).ok
    (clips / "0001.wav.enc").write_bytes((clips / "0002.wav.enc").read_bytes())  # swap clips
    rep = verify_chain(rows(c), keys.public_key, tmp_path, keys.audio_key)
    assert "AUDIO ALTERED" in problems(rep)[1]


# ------------------------------------------------------------------ effective record
def test_materialize_confirmation_and_correction(chain):
    chain.append("confirmation", {"event_id": "E2", "action": "cancelled", "said": "galat"}, "INC-1")
    chain.correct("E1", {"dose": 25.0}, reason="said twenty five", by="P1")
    rec = {x["event_id"]: x for x in materialize(rows(chain), "INC-1")}
    assert rec["E1"]["status"] == "confirmed" and rec["E1"]["event"]["dose"] == 25.0
    assert rec["E1"]["history"][-1]["before"] == {"dose": 50.0}
    assert rec["E2"]["status"] == "cancelled"
    assert verify_chain(rows(chain), chain.keys.public_key).ok        # corrections never break the chain


def test_live_pipeline_writes_verifiable_chain(tmp_path):
    from asr.engine import Transcript
    from asr.live import Pipeline
    from asr.vad import Clip

    class Engine:
        said = iter(["fentanyl 50 mcg IV", "confirm", "epi 1 mg IV"])

        def transcribe(self, audio):
            return Transcript(text=next(self.said), audio_s=1.0, engine="fake")

    pipe = Pipeline(Engine(), "INC-L", tmp_path, extract=True, keys_dir=tmp_path / "keys")
    t = datetime.now(timezone.utc)
    for i in range(3):
        pipe.handle(Clip((np.random.default_rng(i).standard_normal(16000) * 0.1).astype(np.float32), i, i + 1),
                    t + timedelta(seconds=10 * i))
    entries = list(pipe.chain.entries())
    assert [json.loads(e.body)["kind"] for e in entries] == \
        ["genesis", "utterance", "event", "utterance", "confirmation", "utterance", "event"]
    rep = verify_chain(entries, pipe.keys.public_key, tmp_path, pipe.keys.audio_key)
    assert rep.ok, problems(rep)
    rec = materialize(entries, "INC-L")
    assert [(x["event"]["drug"], x["status"]) for x in rec] == [("fentanyl", "confirmed"), ("adrenaline", "recorded")]