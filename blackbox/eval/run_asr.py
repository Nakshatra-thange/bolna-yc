"""
Black Box - run offline ASR over every audio condition, write hypotheses for score.py.
Each line stores raw_text (straight from ASR) AND text (after lexicon correction), so one run
gives you both numbers.

  # A/B the biasing modes (faster-whisper):
  python eval/run_asr.py --model small --bias none
  python eval/run_asr.py --model small --bias prompt
  python eval/run_asr.py --model small --bias hotwords
  python eval/score.py --asr eval/preds/asr_fw-small_none.jsonl --asr-field raw_text   # no biasing at all
  python eval/score.py --asr eval/preds/asr_fw-small_prompt.jsonl                      # prompt + lexicon fix

  # sherpa-onnx (Android-portable engine family):
  python eval/run_asr.py --backend sherpa-whisper --model-dir models/sherpa-onnx-whisper-small.en

First faster-whisper run downloads the model once (~460 MB for 'small'); after that it's offline.
Output: eval/preds/asr_<engine>_<bias>.jsonl
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path

import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asr.engine import make_engine  # noqa: E402
from asr.vocab import LexiconCorrector  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="faster-whisper", choices=["faster-whisper", "sherpa-whisper"])
    ap.add_argument("--model", default="small", help="faster-whisper: tiny | base | small | medium | distil-small.en ...")
    ap.add_argument("--model-dir", help="sherpa-onnx model folder")
    ap.add_argument("--bias", default="prompt", choices=["none", "prompt", "hotwords", "both"])
    ap.add_argument("--language", default="en", help="'en', 'hi', or 'auto' (test which suits Hinglish)")
    ap.add_argument("--conditions", nargs="*", help="subset of folders under eval/data/audio")
    a = ap.parse_args()

    lang = None if a.language == "auto" else a.language
    engine = make_engine(a.backend, model=a.model, model_dir=a.model_dir, bias=a.bias, language=lang)
    tag = (f"fw-{a.model}_{a.bias}" if a.backend == "faster-whisper"
           else f"sherpa-{Path(a.model_dir).name.replace('sherpa-onnx-whisper-', '')}_none")
    fixer = LexiconCorrector()

    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "eval" / "data" / "gold.jsonl", encoding="utf-8")}
    audio_root = ROOT / "eval" / "data" / "audio"
    conds = a.conditions or sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    out = ROOT / "eval" / "preds" / f"asr_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)

    n = 0
    with open(out, "w", encoding="utf-8") as f:
        for cond in conds:
            for wav in sorted((audio_root / cond).glob("*.wav")):
                cid = wav.stem
                if cid not in gold:
                    continue
                x, sr = sf.read(wav, dtype="float32")
                if x.ndim > 1:
                    x = x.mean(axis=1)
                assert sr == 16000, f"{wav} is {sr} Hz"
                tr = engine.transcribe(x)
                fixed, fixes = fixer.correct(tr.text)
                f.write(json.dumps({
                    "id": cid, "condition": cond, "text": fixed, "raw_text": tr.text,
                    "corrections": [asdict(c) for c in fixes], "asr_confidence": round(tr.confidence, 3),
                    "time_logged": gold[cid]["time_logged"], "engine": tr.engine,
                    "audio_s": round(tr.audio_s, 3), "latency_s": round(tr.latency_s, 3),
                }, ensure_ascii=False) + "\n")
                n += 1
                print(f"{cond:<12} {cid}  {tr.latency_s:5.2f}s  {fixed}")
    print(f"\n{n} transcripts -> {out}")


if __name__ == "__main__":
    main()