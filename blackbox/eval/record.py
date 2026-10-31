
from __future__ import annotations

import argparse
import csv
import queue
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

try:
    import sounddevice as sd
except OSError:
    sys.exit("PortAudio missing. macOS: brew install portaudio | Ubuntu: sudo apt install libportaudio2")

ROOT = Path(__file__).resolve().parents[1]
SR = 16000


def record_once() -> np.ndarray:
    q: queue.Queue = queue.Queue()
    chunks = []
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", callback=lambda d, *_: q.put(d.copy())):
        input("  recording... press Enter to stop")
        while not q.empty():
            chunks.append(q.get())
    return np.concatenate(chunks)[:, 0] if chunks else np.zeros(0, dtype="float32")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--speaker", required=True, help="P1 / P2 / P3 (or 'all')")
    ap.add_argument("--condition", default="clean", help="clean | real_siren | real_road | in_vehicle ...")
    ap.add_argument("--redo", action="store_true", help="re-record files that already exist")
    a = ap.parse_args()

    out = ROOT / "eval" / "data" / "audio" / a.condition
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / "eval" / "data" / "recording_script.csv", encoding="utf-8")))
    rows = [r for r in rows if a.speaker == "all" or r["speaker"] == a.speaker]
    print(f"{len(rows)} lines for {a.speaker} -> {out}\n")

    for n, r in enumerate(rows, 1):
        path = out / f"{r['id']}.wav"
        if path.exists() and not a.redo:
            continue
        while True:
            print(f"[{n}/{len(rows)}] {r['id']}   ({r['tags']})\n\n    >>  {r['text']}\n")
            input("  press Enter to start")
            audio = record_once()
            peak = float(np.abs(audio).max()) if audio.size else 0.0
            print(f"  {audio.size / SR:.1f}s, peak {peak:.2f}" + ("  <- too quiet, move closer" if peak < 0.05 else "")
                  + ("  <- clipping, move back" if peak > 0.98 else ""))
            choice = input("  [k]eep / [r]edo / [s]kip / [q]uit: ").strip().lower() or "k"
            if choice == "q":
                return
            if choice == "s":
                break
            if choice == "k" and audio.size:
                sf.write(path, audio, SR, subtype="PCM_16")
                print(f"  saved {path.name}\n")
                break


if __name__ == "__main__":
    main()