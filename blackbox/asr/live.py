"""
Black Box - live on-scene pipeline:  mic/file -> VAD clips -> offline ASR (biased) -> lexicon fix -> log
Works with Wi-Fi off once the model is on disk.

  python -m asr.live --mic --model small --bias prompt --extract
  python -m asr.live --mic --extract --llm-model models/qwen2.5-1.5b-instruct-q4_k_m.gguf --speak
  python -m asr.live --mic --extract --llm-backend ollama --llm-model qwen2.5:1.5b --speak
  python -m asr.live --file eval/data/scene_tts.wav --extract                       # replay a recording
  python -m asr.live --file scene.wav --backend sherpa-whisper --model-dir models/sherpa-onnx-whisper-small.en

Low-confidence events are read back ("Confirm fentanyl 50 micrograms I V?"); the medic answers
"confirm" / "haan" or "cancel" / "galat". Answers are logged as their own records.

Writes runs/<incident>/clips/0001.wav ... and runs/<incident>/utterances.jsonl.
Each line keeps raw_text AND the corrected text, every correction, and the clip's SHA-256:
this is exactly what Phase 3 will sign into the audit chain.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import queue
import sys
import time
from dataclasses import asdict
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import soundfile as sf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from asr.engine import SR, make_engine  # noqa: E402
from asr.vad import Clip, Segmenter  # noqa: E402
from asr.vocab import LexiconCorrector  # noqa: E402
from extract.confirm import ConfirmationQueue, parse_command  # noqa: E402

CONFIRM_BELOW = 0.60   # ASR confidence under this -> ask the medic to confirm


def speak(text: str):
    """Offline text-to-speech for read-backs: macOS 'say', else eSpeak. Never blocks the pipeline on failure."""
    import shutil
    import subprocess
    try:
        if shutil.which("say"):
            subprocess.Popen(["say", "-r", "210", text])
            return
        sys.path.insert(0, str(ROOT / "eval"))
        import sounddevice as sd
        from synth_tts import ESpeak
        speak._tts = getattr(speak, "_tts", None) or ESpeak()
        sd.play(speak._tts.say(text), SR)
    except Exception as exc:  # noqa: BLE001
        print(f"     (speech output unavailable: {exc})")


class Pipeline:
    def __init__(self, engine, incident: str, out_root: Path, lexfix: bool = True, extract: bool = False,
                 extractor=None, speak_prompts: bool = False):
        self.engine, self.incident = engine, incident
        self.dir = out_root / incident
        (self.dir / "clips").mkdir(parents=True, exist_ok=True)
        self.log = open(self.dir / "utterances.jsonl", "a", encoding="utf-8")
        self.fixer = LexiconCorrector() if lexfix else None
        self.extract = extract or extractor is not None
        if self.extract and extractor is None:
            from extract.hybrid import HybridExtractor
            extractor = HybridExtractor(llm=None)
        self.extractor = extractor
        self.queue = ConfirmationQueue()
        self.speak = speak_prompts
        self.seq = sum(1 for _ in open(self.dir / "utterances.jsonl", encoding="utf-8"))

    def handle(self, clip: Clip, time_logged: datetime) -> dict:
        self.seq += 1
        path = self.dir / "clips" / f"{self.seq:04d}.wav"
        sf.write(path, clip.audio, SR, subtype="PCM_16")
        sha = hashlib.sha256(path.read_bytes()).hexdigest()   # hash of the exact bytes on disk

        tr = self.engine.transcribe(clip.audio)
        text, fixes = (self.fixer.correct(tr.text) if self.fixer else (tr.text, []))
        rec = {
            "id": f"{self.incident}-{self.seq:04d}", "seq": self.seq, "incident_id": self.incident,
            "text": text, "raw_text": tr.text, "corrections": [asdict(f) for f in fixes],
            "time_logged": time_logged.isoformat(),
            "clip_start_s": round(clip.start_s, 3), "clip_end_s": round(clip.end_s, 3),
            "audio_sha256": sha, "clip_path": str(path.relative_to(self.dir)),
            "asr_confidence": round(tr.confidence, 3), "engine": tr.engine,
            "audio_s": round(tr.audio_s, 3), "latency_s": round(tr.latency_s, 3),
        }
        for exp in self.queue.expire(time_logged):
            self._write({"type": "confirmation", **exp})
            print(f"     ... {exp['event_id']} left UNCONFIRMED")

        cmd = parse_command(text) if self.extract else None
        if cmd and self.queue.pending:
            res = self.queue.handle(cmd, time_logged, said=text)
            rec.update(type="command", command=cmd, confirmation=res)
            self._write(rec)
            self._print(rec, False)
            print(f"     >> {res['event_id']} {res['action'].upper()}")
            return rec

        rec["type"] = "dictation"
        if self.extract:
            events, meta = self.extractor.extract(text, time_logged, asr=rec)
            for k, e in enumerate(events, 1):
                e["event_id"] = f"{rec['id']}-e{k}"
            rec["events"], rec["extract_meta"] = events, {k: v for k, v in meta.items() if k != "llm_raw"}
        self._write(rec)
        self._print(rec, tr.words != [] or tr.avg_logprob is not None)
        for e in rec.get("events", []):
            if e.get("needs_confirmation"):
                prompt = self.queue.add(e["event_id"], e, time_logged)
                print(f"     ?? {prompt}   ({'; '.join(e['confirm_reasons'])})")
                if self.speak:
                    speak(prompt)
        return rec

    def _write(self, rec: dict):
        self.log.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        self.log.flush()

    def _print(self, r: dict, has_conf: bool):
        flag = "  [CONFIRM?]" if has_conf and r["asr_confidence"] < CONFIRM_BELOW else ""
        print(f"\n#{r['seq']:<3} {r['time_logged'][11:19]}  ({r['audio_s']:.1f}s audio, {r['latency_s']:.2f}s ASR){flag}")
        if r["corrections"]:
            print(f"     heard: {r['raw_text']}")
        print(f"     text : {r['text']}")
        for f in r["corrections"]:
            print(f"     fix  : '{f['heard']}' -> '{f['fixed']}' ({f['kind']}, {f['score']})")
        for e in r.get("events", []):
            mark = "?" if e.get("needs_confirmation") else " "
            if e["event_type"] == "medication":
                print(f"   {mark} EVENT: {e.get('drug')} {e.get('dose', '?')} {e.get('unit', '')} {e.get('route', '?')} @ {e['time_said'][11:19]}")
            elif e["event_type"] != "note":
                print(f"   {mark} EVENT: {e['event_type']} {e.get('procedure') or e.get('vital') or ''}")
        for h in r.get("extract_meta", {}).get("held", []):
            print(f"     (not logged as given: {h.get('drug_as_spoken')} - {h.get('status')})")


def run_file(pipe: Pipeline, path: str, start: datetime, realtime: bool):
    x, sr = sf.read(path, dtype="float32", always_2d=True)
    x = x.mean(axis=1)
    if sr != SR:
        x = np.interp(np.arange(0, len(x), sr / SR), np.arange(len(x)), x).astype(np.float32)
    seg = Segmenter()
    block = 1600  # 100 ms, like a mic callback
    for i in range(0, len(x), block):
        for clip in seg.feed(x[i:i + block]):
            pipe.handle(clip, start + timedelta(seconds=clip.end_s))
        if realtime:
            time.sleep(block / SR)
    for clip in seg.flush():
        pipe.handle(clip, start + timedelta(seconds=clip.end_s))


def run_mic(pipe: Pipeline, device=None):
    try:
        import sounddevice as sd
    except OSError:
        sys.exit("PortAudio missing. macOS: brew install portaudio")
    q: queue.Queue = queue.Queue()
    seg = Segmenter()
    print("listening... (Ctrl+C to stop)")
    with sd.InputStream(samplerate=SR, channels=1, dtype="float32", blocksize=1600, device=device,
                        callback=lambda d, *_: q.put(d[:, 0].copy())):
        try:
            while True:
                for clip in seg.feed(q.get()):
                    pipe.handle(clip, datetime.now(timezone.utc))
        except KeyboardInterrupt:
            for clip in seg.flush():
                pipe.handle(clip, datetime.now(timezone.utc))
            print("\nstopped.")


def main():
    ap = argparse.ArgumentParser()
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--mic", action="store_true")
    src.add_argument("--file")
    ap.add_argument("--device", help="sounddevice input device index/name")
    ap.add_argument("--backend", default="faster-whisper", choices=["faster-whisper", "sherpa-whisper"])
    ap.add_argument("--model", default="small")
    ap.add_argument("--model-dir", help="sherpa-onnx model folder")
    ap.add_argument("--bias", default="prompt", choices=["none", "prompt", "hotwords", "both"])
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-lexfix", action="store_true")
    ap.add_argument("--extract", action="store_true", help="extract events (rules; add --llm-model for hybrid)")
    ap.add_argument("--llm-backend", default="llamacpp", choices=["llamacpp", "ollama"])
    ap.add_argument("--llm-model", help="GGUF path (llamacpp) or tag (ollama) -> hybrid LLM+rules extraction")
    ap.add_argument("--speak", action="store_true", help="read confirmation prompts aloud (offline TTS)")
    ap.add_argument("--incident", default=datetime.now().strftime("INC-%Y%m%d-%H%M%S"))
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--realtime", action="store_true", help="with --file: replay at real speed")
    a = ap.parse_args()

    print(f"loading {a.backend} ...")
    engine = make_engine(a.backend, model=a.model, model_dir=a.model_dir, bias=a.bias, language=a.language)
    extractor = None
    if a.llm_model:
        from extract.hybrid import HybridExtractor
        from extract.llm import LLMExtractor, make_backend
        print(f"loading LLM {a.llm_backend}:{a.llm_model} ...")
        extractor = HybridExtractor(llm=LLMExtractor(make_backend(a.llm_backend, a.llm_model)))
    pipe = Pipeline(engine, a.incident, Path(a.out), lexfix=not a.no_lexfix, extract=a.extract,
                    extractor=extractor, speak_prompts=a.speak)
    print(f"incident {a.incident} -> {pipe.dir}")
    if a.mic:
        run_mic(pipe, a.device)
    else:
        run_file(pipe, a.file, datetime.now(timezone.utc), a.realtime)


if __name__ == "__main__":
    main()