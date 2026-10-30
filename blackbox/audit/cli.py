"""
Black Box - audit CLI.

  python -m audit.cli pubkey                                  # export device public key (pin it at the server)
  python -m audit.cli verify                                  # green/red verdict for every entry
  python -m audit.cli verify --pubkey device_pubkey.txt --expect-head 42:ab12...
  python -m audit.cli show --incident INC-...                 # effective record: events, confirmations, corrections
  python -m audit.cli correct --event INC-...-0003-e1 --set dose=0.5 --reason "said 0.5, heard 5" --by P1

  # DEMO: play the attacker on a COPY of the database (original untouched)
  python -m audit.cli tamper edit   --seq 7 --path event.dose --value 10    -> verify --db runs/blackbox.tampered.db
  python -m audit.cli tamper edit   --seq 7 --path event.dose --value 10 --rehash   (forger recomputes the hash)
  python -m audit.cli tamper delete --seq 5
  python -m audit.cli tamper swap   --seq 4 --with 6
  python -m audit.cli tamper clock  --seq 7 --minutes -15
  python -m audit.cli tamper audio  --seq 3 --from-seq 9         (replace one voice clip with another)
  python -m audit.cli tamper truncate --keep 10                    (verify with --expect-head to catch it)
"""
from __future__ import annotations

import argparse
import json
import shutil
import sqlite3
import sys
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from audit.chain import AuditChain, Entry, canonical, sha256_hex  # noqa: E402
from audit.keys import KeyStore, load_public_key  # noqa: E402
from audit.verify import materialize, patient_context, verify_chain  # noqa: E402

G, R, Y, DIM, B, X = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[1m", "\033[0m"


def read_entries(db: Path) -> list[Entry]:
    con = sqlite3.connect(db)
    rows = con.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain ORDER BY seq").fetchall()
    con.close()
    return [Entry(*r) for r in rows]


def summary(d: dict) -> str:
    p, k = d["payload"], d["kind"]
    if k == "genesis":
        return f"device {p['device_id']} key {p['public_key'][:16]}..."
    if k == "utterance":
        return f"\"{p.get('text', '')[:70]}\""
    if k == "event":
        e = p["event"]
        if e["event_type"] == "medication":
            s = f"{e.get('drug')} {e.get('dose', '?')} {e.get('unit', '')} {e.get('route', '?')}"
        elif e["event_type"] == "procedure":
            s = f"procedure {e.get('procedure')}"
        elif e["event_type"] == "vital":
            s = "vitals " + " ".join(f"{a}={b}" for a, b in e.get("vital", {}).items())
        else:
            s = "note"
        return s + (f"  {Y}[confirm: {'; '.join(e.get('confirm_reasons', []))}]{X}" if e.get("needs_confirmation") else "")
    if k == "confirmation":
        return f"{p['action'].upper()} {p['event_id']}" + (f"  (\"{p.get('said')}\")" if p.get("said") else "")
    if k == "alert":
        return f"{R if p['severity'] == 'critical' else Y}ALERT {p['severity'].upper()} {p['rule_id']}{X}: {p['message']}"
    if k == "alert_response":
        why = f" - {p['reason']}" if p.get("reason") else ""
        return f"{p['action'].upper()} alert {p['alert_id']} ({p.get('rule_id')}){why}"
    if k == "patient":
        return "patient context " + ", ".join(f"{a}={b}" for a, b in p.items() if a not in ("utterance_ref", "current"))
    if k == "correction":
        return f"CORRECTION {p['event_id']}: {p['changes']}  reason: {p.get('reason')}  by {p.get('by')}"
    return json.dumps(p)[:80]


# ------------------------------------------------------------------ commands
def cmd_verify(a):
    entries = read_entries(a.db)
    keys = KeyStore(a.keys)
    if a.pubkey:
        pub, src = load_public_key(a.pubkey), a.pubkey
    else:
        pub, src = keys.public_key, "local keystore (pin a copy with `pubkey` for real verification)"
    head = None
    if a.expect_head:
        s, h = a.expect_head.split(":", 1)
        head = (int(s), h)
    rep = verify_chain(entries, pub, audio_root=a.audio_root or a.db.parent, audio_key=keys.audio_key,
                       expected_head=head)
    by_seq = {e.seq: e for e in entries}
    print(f"{B}Black Box chain verification{X}  {a.db}\n  public key: {src}\n")
    for r in rep.results:
        e = by_seq[r.seq]
        try:
            d = json.loads(e.body)
            line = f"{d.get('ts_wall', '')[11:19]}  {r.kind:<12} {summary(d)}"
        except Exception:  # noqa: BLE001
            line = f"{'':8}  {r.kind:<12} <unreadable>"
        if a.incident and '"incident_id":"%s"' % a.incident not in e.body and r.kind != "genesis" and r.ok:
            continue
        mark = f"{G}  OK {X}" if r.ok else f"{R}{B} BAD {X}"
        print(f"{mark} #{r.seq:<4} {line if r.ok else R + line + X}")
        for p in r.problems:
            print(f"{R}         -> {p}{X}")
        for w in r.warnings:
            print(f"{Y}         !  {w}{X}")
    for p in rep.chain_problems:
        print(f"\n{R}{B}CHAIN: {p}{X}")
    print()
    if rep.ok:
        print(f"{G}{B}RECORD INTACT{X}  {len(rep.results)} entries, head #{rep.head_seq} {rep.head_hash[:16]}...")
        print(f"{DIM}anchor this head at the server:  --expect-head {rep.head_seq}:{rep.head_hash}{X}")
    else:
        bad = [r.seq for r in rep.results if not r.ok]
        print(f"{R}{B}TAMPERING DETECTED{X}  bad entries: {bad[:20]}{'...' if len(bad) > 20 else ''}"
              f"   first bad: #{rep.first_bad_seq}")
    sys.exit(0 if rep.ok else 2)


def cmd_show(a):
    entries = read_entries(a.db)
    incs = sorted({json.loads(e.body).get("incident_id") for e in entries} - {None})
    inc = a.incident or (incs[-1] if incs else None)
    print(f"{B}Effective record{X}  incident {inc}   (incidents on device: {', '.join(incs)})")
    ctx = patient_context(entries, inc)
    if ctx:
        print(f"  patient: " + ", ".join(f"{k}={v}" for k, v in ctx.items() if v not in (None, [])))
    print()
    icon = {"recorded": f"{G}recorded {X}", "confirmed": f"{G}CONFIRMED{X}", "cancelled": f"{DIM}CANCELLED{X}",
            "pending": f"{Y}PENDING  {X}", "unconfirmed": f"{R}UNCONFIRMED{X}"}
    for x in materialize(entries, inc):
        e = x["event"]
        d = {"kind": "event", "payload": {"event": dict(e, needs_confirmation=False)}}
        print(f"  {icon.get(x['status'], x['status'])}  {e.get('time_said', '')[11:19]}  {summary(d)}   {DIM}{x['event_id']}{X}")
        for al in x.get("alerts", []):
            col = R if al["severity"] == "critical" else Y
            r = al["response"]
            ans = f"{r['action'].upper()}" + (f" ({r['reason']})" if r and r.get("reason") else "") if r else "no answer"
            print(f"{col}             GUARDIAN {al['rule_id']}: {al['message']}  -> {ans}{X}")
        for h in x["history"]:
            if h["what"] == "corrected":
                print(f"{Y}             corrected {h['before']} -> {h['after']} ({h['reason']}, by {h['by']}){X}")


def cmd_correct(a):
    changes = {}
    for kv in a.set:
        k, v = kv.split("=", 1)
        try:
            v = json.loads(v)
        except json.JSONDecodeError:
            pass
        changes[k] = v
    chain = AuditChain(a.db, KeyStore(a.keys))
    e = chain.correct(a.event, changes, a.reason, a.by)
    print(f"correction appended as #{e.seq} {e.entry_hash[:16]}... (original entry untouched)")


def cmd_pubkey(a):
    p = KeyStore(a.keys).export_public(a.out)
    print(f"device {KeyStore(a.keys).device_id}  public key -> {p}")


def _set_path(obj: dict, path: str, value):
    parts = path.split(".")
    for k in parts[:-1]:
        obj = obj[k]
    obj[parts[-1]] = value


def cmd_tamper(a):
    out = a.out or a.db.with_name(a.db.stem + ".tampered.db")
    shutil.copy(a.db, out)
    for ext in ("-wal", "-shm"):
        if Path(str(a.db) + ext).exists():
            shutil.copy(str(a.db) + ext, str(out) + ext)
    con = sqlite3.connect(out)
    con.executescript("DROP TRIGGER IF EXISTS chain_no_update; DROP TRIGGER IF EXISTS chain_no_delete;")
    print(f"{DIM}(attacker drops the append-only triggers - they are not the security){X}")
    rows = {r[0]: list(r) for r in con.execute("SELECT seq, entry_hash, prev_hash, body, sig FROM chain")}

    if a.action == "edit":
        seq, h, prev, body, sig = rows[a.seq]
        d = json.loads(body)
        try:
            val = json.loads(a.value)
        except json.JSONDecodeError:
            val = a.value
        _set_path(d["payload"], a.path, val)
        new_body = canonical(d).decode()
        new_hash = sha256_hex(new_body.encode()) if a.rehash else h
        con.execute("UPDATE chain SET body=?, entry_hash=? WHERE seq=?", (new_body, new_hash, seq))
        if a.rehash:  # a careful forger also re-links the next entry ... and still has no key to sign
            nxt = rows.get(seq + 1)
            if nxt:
                nd = json.loads(nxt[3]); nd["prev_hash"] = new_hash
                nb = canonical(nd).decode()
                con.execute("UPDATE chain SET body=?, prev_hash=?, entry_hash=? WHERE seq=?",
                            (nb, new_hash, sha256_hex(nb.encode()), seq + 1))
        print(f"edited #{a.seq} payload.{a.path} = {val}" + (" and recomputed hashes" if a.rehash else ""))
    elif a.action == "delete":
        con.execute("DELETE FROM chain WHERE seq=?", (a.seq,))
        print(f"deleted #{a.seq}")
    elif a.action == "swap":
        r1, r2 = rows[a.seq], rows[a.with_]
        con.execute("UPDATE chain SET entry_hash='tmp', prev_hash=?, body=?, sig=? WHERE seq=?", (r2[2], r2[3], r2[4], a.seq))
        con.execute("UPDATE chain SET entry_hash=?, prev_hash=?, body=?, sig=? WHERE seq=?", (r1[1], r1[2], r1[3], r1[4], a.with_))
        con.execute("UPDATE chain SET entry_hash=? WHERE seq=?", (r2[1], a.seq))
        print(f"swapped #{a.seq} and #{a.with_}")
    elif a.action == "clock":
        seq, h, prev, body, sig = rows[a.seq]
        d = json.loads(body)
        d["ts_wall"] = (datetime.fromisoformat(d["ts_wall"]) + timedelta(minutes=a.minutes)).isoformat(timespec="milliseconds")
        if d["payload"].get("event", {}).get("time_said"):
            t = datetime.fromisoformat(d["payload"]["event"]["time_said"]) + timedelta(minutes=a.minutes)
            d["payload"]["event"]["time_said"] = t.isoformat()
        con.execute("UPDATE chain SET body=? WHERE seq=?", (canonical(d).decode(), seq))
        print(f"moved #{a.seq} timestamps by {a.minutes:+} min")
    elif a.action == "audio":
        dst = json.loads(rows[a.seq][3])["payload"]["clip_path"]
        src = json.loads(rows[a.from_seq][3])["payload"]["clip_path"]
        tdir = out.parent / (out.stem + "_audio")
        shutil.copytree(a.db.parent, tdir, dirs_exist_ok=True, ignore=shutil.ignore_patterns("*.db*", "*_audio"))
        shutil.copy(a.db.parent / src, tdir / dst)
        print(f"replaced clip of #{a.seq} with clip of #{a.from_seq} (in copy {tdir})\n"
              f"verify with:  python -m audit.cli verify --db {out} --audio-root {tdir}")
    elif a.action == "truncate":
        con.execute("DELETE FROM chain WHERE seq>?", (a.keep,))
        print(f"deleted every entry after #{a.keep} (no gap, no broken link - only the server anchor catches this)")
    con.commit()
    con.close()
    print(f"tampered copy: {out}\nnow run:  python -m audit.cli verify --db {out}")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
        # --db / --keys work before OR after the subcommand
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--db", type=Path, default=argparse.SUPPRESS)
    common.add_argument("--keys", default=argparse.SUPPRESS, help="keystore folder (default ~/.blackbox)")
    ap.add_argument("--db", type=Path, default=ROOT / "runs" / "blackbox.db")
    ap.add_argument("--keys", default=None, help="keystore folder (default ~/.blackbox)")
    sub = ap.add_subparsers(dest="cmd", required=True)
    _add = sub.add_parser
    sub.add_parser = lambda *x, **k: _add(*x, parents=[common], **k)  # noqa: E731


    v = sub.add_parser("verify")
    v.add_argument("--pubkey", help="pinned public key (hex or file)")
    v.add_argument("--expect-head", help="SEQ:HASH last anchored at the server")
    v.add_argument("--incident")
    v.add_argument("--audio-root", type=Path)

    s = sub.add_parser("show")
    s.add_argument("--incident")

    c = sub.add_parser("correct")
    c.add_argument("--event", required=True)
    c.add_argument("--set", action="append", required=True, help="field=value, e.g. dose=0.5")
    c.add_argument("--reason", required=True)
    c.add_argument("--by", required=True)

    p = sub.add_parser("pubkey")
    p.add_argument("--out", default=str(ROOT / "device_pubkey.txt"))

    t = sub.add_parser("tamper")
    t.add_argument("action", choices=["edit", "delete", "swap", "clock", "audio", "truncate"])
    t.add_argument("--seq", type=int)
    t.add_argument("--with", dest="with_", type=int)
    t.add_argument("--from-seq", type=int)
    t.add_argument("--path", default="event.dose")
    t.add_argument("--value", default="10")
    t.add_argument("--rehash", action="store_true")
    t.add_argument("--minutes", type=float, default=-15)
    t.add_argument("--keep", type=int)
    t.add_argument("--out", type=Path)

    a = ap.parse_args()
    {"verify": cmd_verify, "show": cmd_show, "correct": cmd_correct, "pubkey": cmd_pubkey, "tamper": cmd_tamper}[a.cmd](a)


if __name__ == "__main__":
    main()