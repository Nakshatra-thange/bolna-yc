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
Output: eval/preds/asr_<engine>_<bias>.jsonl, written via a .partial file and only renamed into place
once every planned (condition, id) is present - an interrupted run can't pass for a finished one.
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


def plan_conditions(audio_root: Path, gold: dict, conditions: list[str] | None) -> dict[str, list[Path]]:
    """{condition: [wav, ...]} for every requested condition. Raises if a condition is unknown,
    empty, or holds wavs whose id isn't in gold - nothing is silently skipped."""
    available = sorted(p.name for p in audio_root.iterdir() if p.is_dir())
    conds = conditions or available
    unknown = [c for c in conds if c not in available]
    if unknown:
        raise SystemExit(f"unknown condition(s) {unknown}; available: {available}")
    plan, problems = {}, []
    for cond in conds:
        wavs = sorted((audio_root / cond).glob("*.wav"))
        stray = [w.stem for w in wavs if w.stem not in gold]
        if not wavs:
            problems.append(f"{cond}: no .wav files")
        if stray:
            problems.append(f"{cond}: ids not in gold.jsonl: {stray}")
        plan[cond] = wavs
    if problems:
        raise SystemExit("audio problems:\n  " + "\n  ".join(problems))
    return plan


def check_counts(out: Path, plan: dict[str, list[Path]]) -> list[str]:
    """Compare the written file against the plan; returns a list of mismatches (empty = OK)."""
    rows = [json.loads(l) for l in open(out, encoding="utf-8") if l.strip()]
    got: dict[str, set] = {}
    for r in rows:
        got.setdefault(r["condition"], set()).add(r["id"])
    bad = []
    for cond, wavs in plan.items():
        want = {w.stem for w in wavs}
        if got.get(cond, set()) != want:
            bad.append(f"{cond}: expected {len(want)}, got {len(got.get(cond, set()))}, "
                       f"missing {sorted(want - got.get(cond, set()))}")
    if len(rows) != sum(len(w) for w in plan.values()):
        bad.append(f"{len(rows)} rows for {sum(len(w) for w in plan.values())} planned utterances")
    return bad


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--backend", default="faster-whisper", choices=["faster-whisper", "sherpa-whisper"])
    ap.add_argument("--model", default="small", help="faster-whisper: tiny | base | small | medium | distil-small.en ...")
    ap.add_argument("--model-dir", help="sherpa-onnx model folder")
    ap.add_argument("--bias", default="prompt", choices=["none", "prompt", "hotwords", "both"])
    ap.add_argument("--language", default="en", help="'en', 'hi', or 'auto' (test which suits Hinglish)")
    ap.add_argument("--conditions", nargs="*", help="subset of folders under eval/data/audio")
    ap.add_argument("--out", help="output jsonl (default eval/preds/asr_<engine>_<bias>[__<conditions>].jsonl)")
    a = ap.parse_args()

    gold = {json.loads(l)["id"]: json.loads(l) for l in open(ROOT / "eval" / "data" / "gold.jsonl", encoding="utf-8")}
    audio_root = ROOT / "eval" / "data" / "audio"
    plan = plan_conditions(audio_root, gold, a.conditions)
    print("plan: " + ", ".join(f"{c}={len(w)}" for c, w in plan.items()))

    lang = None if a.language == "auto" else a.language
    engine = make_engine(a.backend, model=a.model, model_dir=a.model_dir, bias=a.bias, language=lang)
    tag = (f"fw-{a.model}_{a.bias}" if a.backend == "faster-whisper"
           else f"sherpa-{Path(a.model_dir).name.replace('sherpa-onnx-whisper-', '')}_none")
    if a.conditions:  # a subset run must never overwrite the full-run file
        tag += "__" + "+".join(a.conditions)
    fixer = LexiconCorrector()

    out = Path(a.out) if a.out else ROOT / "eval" / "preds" / f"asr_{tag}.jsonl"
    out.parent.mkdir(parents=True, exist_ok=True)
    tmp = out.with_suffix(out.suffix + ".partial")  # renamed to `out` only after counts check out

    n = 0
    with open(tmp, "w", encoding="utf-8") as f:
        for cond, wavs in plan.items():
            for wav in wavs:
                cid = wav.stem
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
                f.flush()
                n += 1
                print(f"{cond:<12} {cid}  {tr.latency_s:5.2f}s  {fixed}")
    bad = check_counts(tmp, plan)
    if bad:
        raise SystemExit(f"count check FAILED, left incomplete output at {tmp}:\n  " + "\n  ".join(bad))
    tmp.replace(out)
    print(f"\n{n} transcripts -> {out}")
    print("counts OK: " + ", ".join(f"{c}={len(w)}" for c, w in plan.items()))


if __name__ == "__main__":
    main()
