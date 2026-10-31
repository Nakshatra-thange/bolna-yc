
from __future__ import annotations

import argparse
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asr.engine import Transcript  # noqa: E402
from asr.live import Pipeline  # noqa: E402
from asr.vad import Clip  # noqa: E402

SCENE = [
    "patient weight seventy kilos, allergic to morphine",
    "started CPR",
    "shocked at two hundred joules",
    "epi one milligram IV",
    "epi one milligram IV",                    # 1 min later -> Guardian: REPEAT_TOO_SOON
    "override, rhythm changed back to VF",      # override WITH a reason -> chained
    "fentanyl fifty micrograms IV",             # controlled drug -> read-back
    "confirm",
    "amiodarone three hundred milligrams IV",
    "morphine four milligrams IV",              # allergy -> Guardian: CONTRAINDICATED (critical)
    "cancel dose",                              # dose cancelled via the alert
    "BP ninety over sixty, pulse one ten",
    "naloxone point four milligrams IM",
]


class ScriptEngine:
    """Stands in for Whisper: returns the scripted line for each clip."""
    name = "script"

    def __init__(self, lines):
        self.lines = iter(lines)

    def transcribe(self, audio):
        return Transcript(text=next(self.lines), audio_s=len(audio) / 16000, engine="script")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--incident", default="DEMO-ARREST")
    ap.add_argument("--keys", default=None)
    a = ap.parse_args()

    try:
        sys.path.insert(0, str(ROOT / "eval"))
        from synth_tts import ESpeak
        tts = ESpeak()
        voice = tts.say
    except Exception:  # noqa: BLE001
        rng = np.random.default_rng(0)
        voice = lambda t: (rng.standard_normal(16000) * 0.05).astype(np.float32)  # noqa: E731

    pipe = Pipeline(ScriptEngine(SCENE), a.incident, Path(a.out), extract=True, keys_dir=a.keys)
    t0 = datetime.now(timezone.utc) - timedelta(minutes=len(SCENE))
    for i, line in enumerate(SCENE):
        audio = voice(line)
        pipe.handle(Clip(audio, i * 60.0, i * 60.0 + len(audio) / 16000), t0 + timedelta(minutes=i))
    h = pipe.chain.head()
    print(f"\nchain {pipe.chain.path}: head #{h.seq} {h.entry_hash[:16]}...  device {pipe.keys.device_id}")


if __name__ == "__main__":
    main()