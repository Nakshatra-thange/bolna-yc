"""
Black Box - live web demo (runs on the ambulance laptop, i.e. localhost; nothing leaves the machine).

  python -m webdemo.app                                   # rules extractor, small.en-style ASR
  python -m webdemo.app --model base                      # faster on slow laptops
  python -m webdemo.app --llm-model models/qwen2.5-1.5b-instruct-q4_k_m.gguf
  python -m webdemo.app --no-asr                          # typed / tap-a-phrase only (no speech model needed)
  then open http://localhost:8001   (use localhost, NOT a LAN IP: browsers only allow the mic on localhost/https)

It drives the SAME Pipeline as `python -m asr.live`: ASR -> lexicon fix -> extraction -> Guardian -> confirmations
-> signed audit chain. The page only records audio, sends it here, and draws what comes back.
All pipeline work runs on ONE worker thread (SQLite connections are not shared across threads).
"""
from __future__ import annotations

import argparse
import asyncio
import json
import sys
import threading
from contextlib import asynccontextmanager
import time
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from fastapi import FastAPI, HTTPException, Request  # noqa: E402
from fastapi.responses import HTMLResponse, JSONResponse  # noqa: E402

from asr.engine import SR, Transcript  # noqa: E402
from asr.live import Pipeline  # noqa: E402
from asr.vad import Clip  # noqa: E402
from audit.chain import Entry  # noqa: E402
from audit.verify import materialize, verify_chain  # noqa: E402

STATIC = Path(__file__).parent / "static"

PHRASES = {
    "groups": [
        {"label": "English", "items": [
            "Patient weight seventy kilos, allergic to aspirin",
            "Gave adrenaline one milligram IV",
            "Gave morphine ten milligram IV",
            "BP one twenty over eighty, heart rate ninety",
            "Defibrillate at two hundred joules",
            "Gave aspirin three hundred milligram",
            "confirm", "cancel", "acknowledged", "override, patient has chest pain", "cancel dose"]},
        {"label": "हिंदी · Hinglish", "items": [
            "wazan pachaas kilo, umar das saal",
            "patient ko penicillin ki allergy hai",
            "adrenaline ek milligram IV diya",
            "morphine do milligram IV diya",
            "fentanyl pachaas mcg IV diya",
            "BP ek sau bees by assi",
            "haan", "galat"]},
    ],
    # played by the "Play demo" button - a safe fallback if the mic or room is a problem
    "script": [
        "Patient weight seventy kilos, allergic to aspirin",
        "Gave adrenaline one milligram IV",
        "Gave aspirin three hundred milligram",
        "override, patient has chest pain",
        "Gave morphine ten milligram IV",
        "confirm",
        "Gave adrenaline one milligram IV",
        "acknowledged",
        "BP ek sau bees by assi, heart rate ninety",
        "fentanyl pachaas mcg IV diya",
        "haan",
        "Gave adrenaline five milligram IV",
        "cancel dose",
    ],
}


class SwitchEngine:
    """Real ASR for voice clips; exact text for typed input (marked engine='typed', confidence high)."""

    def __init__(self, real=None):
        self.real, self.typed = real, None

    def transcribe(self, audio):
        if self.typed is not None:
            t, self.typed = self.typed, None
            return Transcript(text=t, avg_logprob=-0.05, engine="typed")
        if self.real is None:
            raise RuntimeError("speech model not loaded (started with --no-asr or model missing)")
        return self.real.transcribe(audio)


def _plain(x):
    return json.loads(json.dumps(x, default=str))


class Demo:
    def __init__(self, args):
        self.args = args
        self.pool = ThreadPoolExecutor(max_workers=1)      # every pipeline / sqlite call happens here
        self.lock = threading.Lock()
        self.pipe: Pipeline | None = None
        self.sw: SwitchEngine | None = None
        self.asr_error: str | None = None
        self.t0 = time.monotonic()
        self.say: list[str] = []

    # ---------------------------------------------------------------- setup (runs on the worker thread)
    def boot(self):
        a = self.args
        real = None
        if not a.no_asr:
            try:
                from asr.engine import make_engine
                print(f"loading speech model '{a.model}' ...")
                real = make_engine("faster-whisper", model=a.model, bias="prompt", language=a.language)
            except Exception as exc:  # noqa: BLE001
                self.asr_error = f"{type(exc).__name__}: {exc}"
                print(f"!! speech model unavailable ({self.asr_error}). Typed input still works.")
        else:
            self.asr_error = "started with --no-asr"
        self.sw = SwitchEngine(real)
        extractor = None
        if a.llm_model:
            from extract.hybrid import HybridExtractor
            from extract.llm import LLMExtractor, make_backend
            print(f"loading LLM {a.llm_backend}:{a.llm_model} ...")
            extractor = HybridExtractor(llm=LLMExtractor(make_backend(a.llm_backend, a.llm_model)))
        self.pipe = Pipeline(self.sw, a.incident, Path(a.out), extract=True, extractor=extractor,
                             keys_dir=a.keys, guardian=True)
        self.pipe.on_say = lambda lines: self.say.extend(lines)
        print("ready.")

    def new_incident(self, name: str | None = None):
        a = self.args
        old = self.pipe
        incident = name or datetime.now().strftime("WEB-%Y%m%d-%H%M%S")
        self.pipe = Pipeline(self.sw, incident, Path(a.out), extract=True, extractor=old.extractor,
                             keys_dir=a.keys, guardian=True)
        self.pipe.on_say = lambda lines: self.say.extend(lines)
        self.t0 = time.monotonic()
        return incident

    # ---------------------------------------------------------------- work
    def state(self) -> dict:
        p = self.pipe
        entries = list(p.chain.entries())
        rep = verify_chain(entries, p.keys.public_key)
        head = p.chain.head()
        g = p.guardian
        pat = g.patient
        rec = materialize(entries, p.incident)
        return _plain({
            "incident": p.incident, "device_id": p.keys.device_id,
            "asr": {"ready": bool(self.sw.real), "error": self.asr_error, "model": self.args.model,
                    "language": self.args.language, "llm": bool(self.args.llm_model)},
            "chain": {"entries": len(entries), "head_seq": head.seq, "head_hash": head.entry_hash,
                      "ok": rep.ok, "first_bad": rep.first_bad_seq},
            "patient": {"weight_kg": pat.weight_kg, "age_years": pat.age_years, "allergies": pat.allergies,
                        "flags": pat.flags},
            "pending_confirm": list(p.queue.pending.keys()),
            "pending_alerts": list(g.pending.keys()),
            "record": [{"event_id": x["event_id"], "seq": x["seq"], "status": x["status"], "event": x["event"],
                        "alerts": x["alerts"], "history": x["history"]}
                       for x in rec if x["event"].get("event_type") != "note"],
        })

    def _pack(self, rec: dict, source: str) -> dict:
        say, self.say = self.say, []
        out = {
            "seq": rec["seq"], "type": rec.get("type"), "source": source, "text": rec["text"],
            "raw_text": rec["raw_text"], "fixes": rec.get("corrections", []),
            "confidence": rec.get("asr_confidence"), "audio_s": rec.get("audio_s"), "latency_s": rec.get("latency_s"),
            "engine": rec.get("engine"), "clip_sha256": rec.get("audio_sha256") if source == "voice" else None,
            "chain_seq": rec.get("chain_seq"), "events": [], "alerts": rec.get("alerts", []),
            "command": rec.get("command"), "confirmation": rec.get("confirmation"),
            "alert_response": rec.get("alert_response"), "held": rec.get("extract_meta", {}).get("held", []),
            "extract_source": rec.get("extract_meta", {}).get("source"), "say": say,
        }
        for e in rec.get("events", []):
            out["events"].append({k: e.get(k) for k in ("event_id", "event_type", "drug", "drug_as_spoken", "dose",
                                                        "unit", "route", "vital", "procedure", "params",
                                                        "needs_confirmation", "confirm_reasons", "time_said")})
        return _plain(out)

    def handle_text(self, text: str) -> dict:
        text = " ".join(text.split())
        self.sw.typed = text
        now = time.monotonic() - self.t0
        rec = self.pipe.handle(Clip(np.zeros(SR // 10, dtype=np.float32), now, now + 0.1), datetime.now(timezone.utc))
        return {"utterance": self._pack(rec, "typed"), "state": self.state()}

    def handle_voice(self, pcm16: bytes) -> dict:
        if not self.sw.real:
            raise RuntimeError(self.asr_error or "speech model not loaded")
        audio = np.frombuffer(pcm16[: len(pcm16) // 2 * 2], dtype="<i2").astype(np.float32) / 32768.0
        now = time.monotonic() - self.t0
        rec = self.pipe.handle(Clip(audio, now - len(audio) / SR, now), datetime.now(timezone.utc))
        if not rec["text"].strip():
            return {"utterance": None, "state": self.state()}
        return {"utterance": self._pack(rec, "voice"), "state": self.state()}

    def tamper_demo(self) -> dict:
        """Forge the latest dose in a COPY of the chain (in memory) and verify it. The real chain is untouched."""
        entries = list(self.pipe.chain.entries())
        target = None
        for e in reversed(entries):
            d = e.data
            ev = d.get("payload", {}).get("event") if d.get("kind") == "event" else None
            if ev and ev.get("event_type") == "medication" and ev.get("dose") is not None:
                target = e
                break
        if target is None:
            return {"ok": False, "message": "No medication recorded yet - dictate one first."}
        d = target.data
        old = d["payload"]["event"]["dose"]
        new = old * 10 if old else 1
        d["payload"]["event"]["dose"] = new
        forged = Entry(target.seq, target.entry_hash, target.prev_hash, json.dumps(d, separators=(",", ":")), target.sig)
        rep = verify_chain([forged if e.seq == target.seq else e for e in entries], self.pipe.keys.public_key)
        bad = next((r for r in rep.results if not r.ok), None)
        return {"ok": True, "detected": not rep.ok, "seq": target.seq, "drug": d["payload"]["event"].get("drug"),
                "old": old, "new": new, "first_bad": rep.first_bad_seq,
                "problems": bad.problems if bad else []}

    def sync(self, server: str) -> dict:
        from sync.client import SyncClient
        c = SyncClient(server, Path(self.args.out) / "blackbox.db", self.pipe.keys)
        try:
            r = c.push()
        except Exception as exc:  # noqa: BLE001
            return {"ok": False, "message": f"Hospital server not reachable ({exc}). Data is safe on the device."}
        rc = r["receipt"] or {}
        return {"ok": True, "accepted": r["accepted"], "clips": r["clips"], "anchor_seq": rc.get("seq"),
                "anchor_hash": rc.get("entry_hash"), "server": server, "dashboard": server.rstrip("/") + "/"}


def create_app(args) -> FastAPI:
    demo = Demo(args)
    loop_run = lambda fn, *a: asyncio.get_running_loop().run_in_executor(demo.pool, fn, *a)  # noqa: E731

    @asynccontextmanager
    async def lifespan(_app):
        await loop_run(demo.boot)
        yield

    app = FastAPI(title="Black Box live demo", lifespan=lifespan)

    @app.get("/", response_class=HTMLResponse)
    async def index():
        return (STATIC / "index.html").read_text(encoding="utf-8")

    @app.get("/api/phrases")
    async def phrases():
        return PHRASES

    @app.get("/api/state")
    async def state():
        return await loop_run(demo.state)

    @app.post("/api/text")
    async def text(req: Request):
        body = await req.json()
        t = (body.get("text") or "").strip()
        if not t:
            raise HTTPException(400, "empty text")
        return await loop_run(demo.handle_text, t)

    @app.post("/api/voice")
    async def voice(req: Request):
        pcm = await req.body()
        if len(pcm) < SR * 2 * 0.25:
            return {"utterance": None, "state": await loop_run(demo.state)}
        try:
            return await loop_run(demo.handle_voice, pcm)
        except RuntimeError as exc:
            return JSONResponse({"detail": str(exc)}, status_code=503)

    @app.post("/api/incident")
    async def incident(req: Request):
        body = await req.json() if (await req.body()) else {}
        await loop_run(demo.new_incident, body.get("name"))
        return await loop_run(demo.state)

    @app.post("/api/tamper")
    async def tamper():
        return await loop_run(demo.tamper_demo)

    @app.post("/api/sync")
    async def sync(req: Request):
        body = await req.json() if (await req.body()) else {}
        return await loop_run(demo.sync, body.get("server") or args.server_url)

    return app


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8001)
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--model", default="small", help="faster-whisper size: tiny/base/small (base is faster)")
    ap.add_argument("--language", default="en")
    ap.add_argument("--no-asr", action="store_true", help="skip loading the speech model (typed input only)")
    ap.add_argument("--llm-model", help="GGUF path / ollama tag for hybrid extraction")
    ap.add_argument("--llm-backend", default="llamacpp", choices=["llamacpp", "ollama"])
    ap.add_argument("--keys", help="keystore folder (default ~/.blackbox)")
    ap.add_argument("--out", default=str(ROOT / "runs"))
    ap.add_argument("--incident", default=datetime.now().strftime("WEB-%Y%m%d-%H%M%S"))
    ap.add_argument("--server-url", default="http://127.0.0.1:8000", help="hospital server for the Sync button")
    args = ap.parse_args()
    import uvicorn
    uvicorn.run(create_app(args), host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    main()