"""
Black Box - run offline ASR over every audio condition, write hypotheses for score.py.

  pip install faster-whisper
  python eval/run_asr.py --model small                     # plain
  python eval/run_asr.py --model small --vocab-prompt      # biased toward drug names (compare WER!)

First run downloads the model once (~460 MB for 'small'); after that it works with Wi-Fi off.
Output: eval/preds/asr_<model>[_vocab].jsonl  ->  {id, condition, text, time_logged, audio_s, latency_s}
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from protocol import ProtocolBook  # noqa: E402


def vocab_prompt() -> str:
    """Whisper's initial_prompt nudges decoding toward these spellings. Keep it short (<~200 tokens)."""
    book = ProtocolBook.load()
    names = sorted({a for d in book.p.drugs.values() for a in d.aliases if len(a) > 3})
    return ("Paramedic dictation. Drugs: " + ", ".join(names) +
            ". Routes: IV, IO, IM, intranasal, sublingual, nebulised. Units: mg, mcg, mL, grams.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="small", help="tiny | base | small | medium | large-v3 | distil-small.en ...")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--compute-type", default="int8")
    ap.add_argument("--vocab-prompt", action="store_true")
    ap.add_argument("--language", default=None, help="force 'en' or 'hi'; default auto-detect (better for Hinglish? test it)")
    ap.add_argument("--conditions", nargs="*", help="subset of folders under eval/data/audio")
    a = ap.parse_args()

    from faster_whisper import WhisperModel  # imported here so --help works without it

    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "eval" / "data" / "gold.jsonl", encoding="utf-8")}
    audio_root = ROOT / "eval" / "data" / "audio"
    conds = a.conditions or sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    model = WhisperModel(a.model, device=a.device, compute_type=a.compute_type)
    prompt = vocab_prompt() if a.vocab_prompt else None

    out = ROOT / "eval" / "preds" / f"asr_{a.model}{'_vocab' if a.vocab_prompt else ''}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for cond in conds:
            for wav in sorted((audio_root / cond).glob("*.wav")):
                cid = wav.stem
                if cid not in gold:
                    continue
                dur = sf.info(wav).duration
                t0 = time.perf_counter()
                segs, _ = model.transcribe(str(wav), beam_size=5, language=a.language, initial_prompt=prompt,
                                           vad_filter=True, condition_on_previous_text=False)
                text = " ".join(s.text.strip() for s in segs)
                lat = time.perf_counter() - t0
                f.write(json.dumps({"id": cid, "condition": cond, "text": text,
                                    "time_logged": gold[cid]["time_logged"],
                                    "audio_s": round(dur, 3), "latency_s": round(lat, 3)}, ensure_ascii=False) + "\n")
                n += 1
                print(f"{cond:<12} {cid}  {lat:5.2f}s  {text}")
    print(f"\n{n} transcripts -> {out}")


if __name__ == "__main__":
    main()