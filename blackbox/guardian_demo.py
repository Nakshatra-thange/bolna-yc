
from __future__ import annotations

import argparse
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from asr.engine import SR, Transcript  # noqa: E402
from asr.live import Pipeline  # noqa: E402
from asr.vad import Clip  # noqa: E402
from audit.verify import materialize, verify_chain  # noqa: E402

G, A, R, D, B, X = "\033[92m", "\033[93m", "\033[91m", "\033[2m", "\033[1m", "\033[0m"

SCRIPT = [
    ("context", "Patient weight seventy kilos"),
    ("unsafe",  "Administer five milligram adrenaline IV"),            # protocol max single dose is 1 mg
    ("answer",  "override, patient is in cardiac arrest"),             # spoken reason is mandatory
]


class TextEngine:
    text = ""

    def transcribe(self, audio):
        return Transcript(text=self.text, avg_logprob=-0.05, engine="typed")


def say(pipe, eng, t0, line):
    eng.text = line
    now = time.monotonic() - t0
    return pipe.handle(Clip(np.zeros(SR // 10, dtype=np.float32), now, now + 0.1), datetime.now(timezone.utc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--type", action="store_true", help="type the lines yourself")
    ap.add_argument("--fast", action="store_true")
    a = ap.parse_args()
    pause = (lambda s: None) if a.fast else time.sleep

    tmp = tempfile.mkdtemp(prefix="bb-guardian-")
    eng, t0 = TextEngine(), time.monotonic()
    pipe = Pipeline(eng, "GUARDIAN-DEMO", Path(tmp), extract=True, keys_dir=tmp + "/keys", guardian=True)

    print(f"\n{B}BLACK BOX · GUARDIAN{X}  {D}(second pair of ears){X}\n")
    lines = iter(SCRIPT)
    while True:
        if a.type:
            try:
                line = input(f"{B}Paramedic says >{X} ").strip()
            except (EOFError, KeyboardInterrupt):
                break
            if not line:
                continue
        else:
            nxt = next(lines, None)
            if nxt is None:
                break
            line = nxt[1]
            pause(1.2)
            print(f"{B}Paramedic says >{X} {line}")
            pause(0.6)
        say(pipe, eng, t0, line)
        pause(1.0)
        print()

    entries = list(pipe.chain.entries())
    rep = verify_chain(entries, pipe.keys.public_key)
    print(f"{B}── What is now in the permanent, signed record ──{X}")
    for x in materialize(entries, "GUARDIAN-DEMO"):
        e = x["event"]
        if e.get("event_type") != "medication":
            continue
        print(f"  {e.get('drug')} {e.get('dose')} {e.get('unit')} {e.get('route') or ''}   status: {x['status']}")
        for al in x["alerts"]:
            col = R if al["severity"] == "critical" else A
            print(f"    {col}⚠ {al['rule_id']}{X}: {al['message']}")
            r = al["response"]
            if r:
                why = f' — reason: "{r["reason"]}"' if r.get("reason") else ""
                print(f"    {G}↳ {r['action'].upper()}{why}   (signed in chain entry #{r['seq']}){X}")
            else:
                print(f"    {R}↳ NO ANSWER recorded{X}")
    print(f"\n  {G if rep.ok else R}Chain integrity: {'VERIFIED' if rep.ok else 'FAILED'}{X}  {D}({len(entries)} signed entries){X}\n")


if __name__ == "__main__":
    main()