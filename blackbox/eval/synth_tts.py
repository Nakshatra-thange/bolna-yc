"""
Black Box - synthesize every gold dictation with offline eSpeak-NG (no recording needed).

  pip install espeakng-loader soundfile numpy
  python eval/synth_tts.py                 # -> eval/data/audio/tts/BB-001.wav ... (all 119 cases)
  python eval/synth_tts.py --gap 2.0       # also writes eval/data/scene_tts.wav: all lines with pauses,
                                           #   for testing live VAD segmentation with --file

Robotic voice, so it's an easy ASR condition - treat it as a smoke test, not as real accuracy.
Your own recordings (eval/record.py) are the numbers that count.
"""
from __future__ import annotations

import argparse
import ctypes
import json
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
SR_OUT = 16000


class ESpeak:
    def __init__(self, voice: str = "en-us", rate_wpm: int = 165):
        import espeakng_loader as el
        self.lib = ctypes.cdll.LoadLibrary(el.get_library_path())
        self.lib.espeak_Initialize.restype = ctypes.c_int
        sr = self.lib.espeak_Initialize(2, 0, el.get_data_path().encode(), 0)  # 2 = AUDIO_OUTPUT_SYNCHRONOUS
        if sr <= 0:
            raise RuntimeError("espeak_Initialize failed")
        self.sr = sr
        self._buf: list[np.ndarray] = []
        CB = ctypes.CFUNCTYPE(ctypes.c_int, ctypes.POINTER(ctypes.c_short), ctypes.c_int, ctypes.c_void_p)

        def _cb(wav, n, _events):
            if wav and n > 0:
                self._buf.append(np.ctypeslib.as_array(wav, shape=(n,)).copy())
            return 0

        self._cb = CB(_cb)  # keep a reference so it isn't garbage-collected
        self.lib.espeak_SetSynthCallback(self._cb)
        self.lib.espeak_SetVoiceByName(voice.encode())
        self.lib.espeak_SetParameter(1, rate_wpm, 0)  # 1 = espeakRATE

    def say(self, text: str) -> np.ndarray:
        """Return float32 mono audio at 16 kHz."""
        self._buf = []
        b = text.encode("utf-8") + b"\0"
        self.lib.espeak_Synth(ctypes.c_char_p(b), ctypes.c_size_t(len(b)), 0, 1, 0, 0, None, None)
        self.lib.espeak_Synchronize()
        x = (np.concatenate(self._buf) if self._buf else np.zeros(1, np.int16)).astype(np.float32) / 32768
        n_out = int(len(x) * SR_OUT / self.sr)
        return np.interp(np.linspace(0, len(x) - 1, n_out), np.arange(len(x)), x).astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voice", default="en-us")
    ap.add_argument("--gap", type=float, default=0, help="also write a single scene file with this many seconds between lines")
    ap.add_argument("--limit", type=int, default=0)
    a = ap.parse_args()

    gold = [json.loads(l) for l in open(ROOT / "eval" / "data" / "gold.jsonl", encoding="utf-8")]
    if a.limit:
        gold = gold[: a.limit]
    out = ROOT / "eval" / "data" / "audio" / "tts"
    out.mkdir(parents=True, exist_ok=True)
    tts = ESpeak(a.voice)
    scene = []
    pad = np.zeros(int(0.3 * SR_OUT), np.float32)
    for c in gold:
        x = np.concatenate([pad, tts.say(c["text"]), pad])
        sf.write(out / f"{c['id']}.wav", x, SR_OUT, subtype="PCM_16")
        scene += [x, np.zeros(int(a.gap * SR_OUT), np.float32)]
    print(f"wrote {len(gold)} files -> {out}")
    if a.gap:
        p = ROOT / "eval" / "data" / "scene_tts.wav"
        sf.write(p, np.concatenate(scene), SR_OUT, subtype="PCM_16")
        print(f"wrote scene ({len(gold)} lines, {a.gap}s gaps) -> {p}")


if __name__ == "__main__":
    main()