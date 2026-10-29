"""
Black Box - local LLM extractor with grammar-constrained JSON output.

Division of labour (this is the safety argument):
  the LLM only READS:  copies drug-as-spoken, dose number, unit, route, time phrase, evidence span,
                       and decides given / planned / not_given (negation, plans, self-corrections)
  the CODE decides:    drug identity (protocol aliases), unit defaults, time arithmetic, and checks
                       every drug name and dose number really occurs in the transcript.
                       Anything inferred or unverifiable -> needs_confirmation.

Backends (both fully offline):
  llamacpp  pip install llama-cpp-python ; a GGUF file, e.g. qwen2.5-1.5b-instruct-q4_k_m.gguf
  ollama    brew install ollama ; ollama pull qwen2.5:1.5b   (server on localhost)
Both enforce the JSON schema below at decode time, so output is always parseable.
"""
from __future__ import annotations

import json
import re
import sys
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Optional, Protocol

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalize import resolve_time, words_to_numbers  # noqa: E402
from protocol import ProtocolBook  # noqa: E402

BOOK = ProtocolBook.load()
ROUTES = ["IV", "IO", "IM", "IN", "PO", "SL", "NEB", "SC", "ODT", "ET"]
UNITS = ["mg", "mcg", "g", "mL"]
PROCEDURES = ["defibrillation", "cpr", "intubation", "iv_access", "io_access", "ecg_12_lead", "other"]
VITAL_KEYS = {"sbp": "sbp", "dbp": "dbp", "hr": "hr", "spo2": "spo2", "rr": "rr", "gcs": "gcs", "bgl": "bgl_mg_dl"}


def _nullable(schema: dict) -> dict:
    return {"anyOf": [schema, {"type": "null"}]}


# Property order matters for llama.cpp grammars: 'evidence' first makes the model quote the
# words it is using BEFORE it fills in values (cheap grounding for small models).
SCHEMA = {
    "type": "object",
    "properties": {
        "events": {
            "type": "array",
            "maxItems": 6,
            "items": {
                "type": "object",
                "properties": {
                    "evidence": {"type": "string"},
                    "type": {"enum": ["medication", "procedure", "vital", "note"]},
                    "status": {"enum": ["given", "planned", "not_given"]},
                    "drug_spoken": _nullable({"type": "string"}),
                    "dose": _nullable({"type": "number"}),
                    "unit": _nullable({"enum": UNITS}),
                    "route": _nullable({"enum": ROUTES}),
                    "time_phrase": _nullable({"type": "string"}),
                    "procedure": _nullable({"enum": PROCEDURES}),
                    "joules": _nullable({"type": "number"}),
                    "vitals": _nullable({
                        "type": "object",
                        "properties": {k: _nullable({"type": "number"}) for k in VITAL_KEYS},
                    }),
                },
                "required": ["evidence", "type", "status", "drug_spoken", "dose", "unit", "route", "time_phrase"],
            },
        }
    },
    "required": ["events"],
}

SYSTEM = """You turn one paramedic dictation (English or Hinglish, maybe with speech-recognition errors) into JSON.
Output only actions that the crew says happened. Rules:
- evidence: copy the exact words of the dictation this event comes from.
- type: medication (a drug), procedure, vital (measurements), note (anything else).
- status: given = already done. planned = drawing up / ready / about to / will give / hold / next dose if.
  not_given = negated, refused, allergic, or the patient took it himself before we arrived.
- drug_spoken: the drug name exactly as said (brand names too: Ondem, Ecosprin, Asthalin, Sorbitrate, Mezolam).
- dose: the number only. If the speaker corrects himself ("two mg, sorry, four") use the final number.
  Pain scores like 8/10 and blood sugar readings are NOT doses.
- unit: mg, mcg, g or mL as said; null if no unit was said. ("mics" = mcg, "migs" = mg, "cc" = mL)
- route: IV (also ivp, push, line, line mein), IO, IM (in the thigh), IN (intranasal, up the nose, naak),
  SL (under the tongue, jeebh ke neeche), PO (chewed, orally, munh se), NEB (neb, nebulised), ODT. null if not said.
- time_phrase: copy the words saying WHEN ("5 minutes ago", "at 10:42", "teen minute pehle"), else null. Never compute times.
- procedure: defibrillation (shock), cpr (compressions), intubation, iv_access, io_access, ecg_12_lead, other.
- vitals: sbp, dbp, hr, spo2, rr, gcs, bgl (blood sugar / GRBS). "over palp" means only sbp is known.
- Hinglish: diya / de diya / lagaya / chadhaya = given; nahi diya = not_given.
- If nothing was done, return one note event."""

FEWSHOT = [
    ("okay gave naloxone point four IM, about three minutes ago",
     {"events": [{"evidence": "gave naloxone point four IM, about three minutes ago", "type": "medication", "status": "given",
                  "drug_spoken": "naloxone", "dose": 0.4, "unit": None, "route": "IM", "time_phrase": "about three minutes ago"}]}),
    ("versed is drawn up, ten mg IM, don't give yet",
     {"events": [{"evidence": "versed is drawn up, ten mg IM, don't give yet", "type": "medication", "status": "planned",
                  "drug_spoken": "versed", "dose": 10, "unit": "mg", "route": "IM", "time_phrase": None}]}),
    ("pulse one ten, sats ninety one, then Asthalin 5 mg neb chalu kiya",
     {"events": [{"evidence": "pulse one ten, sats ninety one", "type": "vital", "status": "given", "drug_spoken": None,
                  "dose": None, "unit": None, "route": None, "time_phrase": None,
                  "vitals": {"sbp": None, "dbp": None, "hr": 110, "spo2": 91, "rr": None, "gcs": None, "bgl": None}},
                 {"evidence": "Asthalin 5 mg neb chalu kiya", "type": "medication", "status": "given",
                  "drug_spoken": "Asthalin", "dose": 5, "unit": "mg", "route": "NEB", "time_phrase": None}]}),
    ("he says no morphine, allergic. shocked at one fifty joules",
     {"events": [{"evidence": "no morphine, allergic", "type": "medication", "status": "not_given",
                  "drug_spoken": "morphine", "dose": None, "unit": None, "route": None, "time_phrase": None},
                 {"evidence": "shocked at one fifty joules", "type": "procedure", "status": "given", "drug_spoken": None,
                  "dose": None, "unit": None, "route": None, "time_phrase": None, "procedure": "defibrillation", "joules": 150}]}),
]


def build_messages(text: str) -> list[dict]:
    msgs = [{"role": "system", "content": SYSTEM}]
    for ex_in, ex_out in FEWSHOT:
        msgs.append({"role": "user", "content": ex_in})
        msgs.append({"role": "assistant", "content": json.dumps(ex_out, ensure_ascii=False)})
    msgs.append({"role": "user", "content": text})
    return msgs


# ------------------------------------------------------------------ backends
class Backend(Protocol):
    name: str
    def generate(self, messages: list[dict], schema: dict) -> str: ...


class LlamaCppBackend:
    def __init__(self, model_path: str, n_ctx: int = 4096, n_threads: Optional[int] = None,
                 n_gpu_layers: int = -1, max_tokens: int = 600):
        from llama_cpp import Llama
        self.llm = Llama(model_path=model_path, n_ctx=n_ctx, n_threads=n_threads,
                         n_gpu_layers=n_gpu_layers, verbose=False)   # -1 = all layers on Metal/GPU if available
        self.max_tokens = max_tokens
        self.name = f"llamacpp:{Path(model_path).name}"

    def generate(self, messages, schema):
        out = self.llm.create_chat_completion(
            messages=messages, temperature=0.0, max_tokens=self.max_tokens,
            response_format={"type": "json_object", "schema": schema},   # compiled to a GBNF grammar
        )
        return out["choices"][0]["message"]["content"]


class OllamaBackend:
    def __init__(self, model: str = "qwen2.5:1.5b", host: str = "http://127.0.0.1:11434", timeout: float = 30):
        self.model, self.host, self.timeout = model, host.rstrip("/"), timeout
        self.name = f"ollama:{model}"

    def generate(self, messages, schema):
        body = json.dumps({"model": self.model, "messages": messages, "format": schema, "stream": False,
                           "keep_alive": "30m", "options": {"temperature": 0}}).encode()
        req = urllib.request.Request(f"{self.host}/api/chat", data=body, headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=self.timeout) as r:
            return json.loads(r.read())["message"]["content"]


def make_backend(kind: str, model: str, n_ctx: int = 4096) -> Backend:
    if kind == "llamacpp":
        return LlamaCppBackend(model, n_ctx=n_ctx)
    if kind == "ollama":
        return OllamaBackend(model)
    raise ValueError(kind)


# ------------------------------------------------------------------ post-processing
def _num_str(x: float) -> str:
    return str(int(x)) if float(x) == int(x) else repr(float(x))


def _resolve_drug(spoken: Optional[str]) -> Optional[str]:
    if not spoken:
        return None
    s = re.sub(r"^(?:inj\.?|injection|tab\.?|tablet)\s+", "", spoken.strip(), flags=re.I)
    hit = BOOK.resolve_drug(s)
    if hit:
        return hit
    words = re.findall(r"[a-z0-9]+", s.lower())
    for n in (2, 1):
        for i in range(len(words) - n + 1):
            hit = BOOK.resolve_drug(" ".join(words[i:i + n]))
            if hit:
                return hit
    return None


def _drug_in_text(drug: str, norm_text: str) -> bool:
    names = [a.lower() for a in BOOK.p.drugs[drug].aliases] + [drug.replace("_", " ")]
    return any(re.search(r"(?<![a-z0-9])" + re.escape(n) + r"(?![a-z0-9])", norm_text) for n in names)


def postprocess(data: dict, text: str, time_logged: datetime) -> tuple[list[dict], list[dict]]:
    """LLM JSON -> (events in gold format with needs_confirmation, held = planned/not_given)."""
    norm = words_to_numbers(text)
    numbers = set(re.findall(r"\d+(?:\.\d+)?", norm))
    numbers |= {_num_str(float(n)) for n in numbers}
    whole_t, whole_src, _ = resolve_time(norm, time_logged)
    events, held = [], []

    for e in data.get("events", []):
        reasons: list[str] = []
        if e.get("time_phrase"):
            t, src, span = resolve_time(words_to_numbers(e["time_phrase"]), time_logged)
            if span is None:
                t, src = whole_t, whole_src
        else:
            t, src = whole_t, whole_src
        etype = e.get("type")

        if etype == "medication":
            drug = _resolve_drug(e.get("drug_spoken"))
            if e.get("status") != "given":
                held.append({"status": e.get("status"), "drug": drug, "drug_as_spoken": e.get("drug_spoken"),
                             "evidence": e.get("evidence")})
                continue
            ev = {"event_type": "medication", "drug": drug, "drug_as_spoken": e.get("drug_spoken")}
            if drug is None:
                reasons.append(f"unknown drug '{e.get('drug_spoken')}'")
            elif not _drug_in_text(drug, norm):
                reasons.append("drug name not in transcript")
            dose = e.get("dose")
            if dose is not None and dose > 0:
                ev["dose"] = float(dose)
                if _num_str(dose) not in numbers:
                    reasons.append(f"dose {_num_str(dose)} not spoken (inferred)")
                unit = e.get("unit") or (BOOK.p.drugs[drug].canonical_unit if drug else None)
                if unit:
                    ev["unit"] = unit
            else:
                reasons.append("no dose")
            if e.get("route"):
                ev["route"] = e["route"]
                if drug and e["route"] not in {r.value for ind in BOOK.p.drugs[drug].indications.values() for r in ind.routes}:
                    reasons.append(f"route {e['route']} not in protocol for {drug}")
            else:
                reasons.append("no route")
        elif etype == "procedure":
            ev = {"event_type": "procedure", "procedure": e.get("procedure") or "other", "params": {}}
            if e.get("joules"):
                ev["params"]["joules"] = int(e["joules"]) if float(e["joules"]).is_integer() else e["joules"]
        elif etype == "vital":
            vit = {VITAL_KEYS[k]: v for k, v in (e.get("vitals") or {}).items() if v is not None and k in VITAL_KEYS}
            vit = {k: (float(v) if k == "bgl_mg_dl" else int(v)) for k, v in vit.items()}
            if not vit:
                continue
            ev = {"event_type": "vital", "vital": vit}
        else:
            continue
        ev.update({"time_said": t.isoformat(), "time_source": src, "evidence": e.get("evidence"),
                   "needs_confirmation": bool(reasons), "confirm_reasons": reasons})
        events.append(ev)

    # merge vitals into one event (gold format), keep notes only if nothing else happened
    vit_events = [e for e in events if e["event_type"] == "vital"]
    if len(vit_events) > 1:
        merged = dict(vit_events[0])
        merged["vital"] = {k: v for e in vit_events for k, v in e["vital"].items()}
        events = [e for e in events if e["event_type"] != "vital"] + [merged]
    if not events and not held:
        events = [{"event_type": "note", "time_said": whole_t.isoformat(), "time_source": whole_src,
                   "needs_confirmation": False, "confirm_reasons": []}]
    return events, held


class LLMExtractor:
    def __init__(self, backend: Backend):
        self.backend = backend

    def extract(self, text: str, time_logged: datetime) -> tuple[list[dict], list[dict], dict]:
        t0 = time.perf_counter()
        raw = self.backend.generate(build_messages(text), SCHEMA)
        data = json.loads(raw)
        events, held = postprocess(data, text, time_logged)
        return events, held, {"llm_raw": data, "llm_latency_s": round(time.perf_counter() - t0, 3),
                              "llm": self.backend.name}