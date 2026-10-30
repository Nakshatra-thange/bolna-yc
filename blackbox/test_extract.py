"""Tests for Phase 2 extraction: LLM post-processing, hybrid safety policy, confirmation.
No model needed: a scripted fake LLM returns fixed JSON.   python -m pytest -q"""
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from extract.confirm import ConfirmationQueue, parse_command, speakable
from extract.hybrid import HybridExtractor
from extract.llm import SCHEMA, LLMExtractor, build_messages, postprocess

ROOT = Path(__file__).parent
TL = datetime(2026, 10, 25, 9, 0, tzinfo=timezone(timedelta(hours=5, minutes=30)))


def ev(**kw):
    base = {"evidence": "", "type": "medication", "status": "given", "drug_spoken": None, "dose": None,
            "unit": None, "route": None, "time_phrase": None}
    base.update(kw)
    return base


class Scripted:
    """Fake LLM backend: answers from a dict {dictation text: llm json}."""
    name = "scripted"

    def __init__(self, answers: dict):
        self.answers = answers

    def generate(self, messages, schema):
        return json.dumps(self.answers[messages[-1]["content"]])


class Broken:
    name = "broken"

    def generate(self, messages, schema):
        raise TimeoutError("model not loaded")


def hybrid(answers):
    return HybridExtractor(llm=LLMExtractor(Scripted(answers)))


def meds(events):
    return [e for e in events if e["event_type"] == "medication"]


def test_hybrid_imports_as_package_and_module():
    for code in ("import extract.hybrid",):
        result = subprocess.run([sys.executable, "-c", code], cwd=ROOT, capture_output=True, text=True)
        assert result.returncode == 0, result.stderr
    result = subprocess.run([sys.executable, "-m", "extract.hybrid"], cwd=ROOT, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr


# ------------------------------------------------------------------ schema / prompt
def test_schema_compiles_to_llama_grammar():
    llama_grammar = pytest.importorskip("llama_cpp.llama_grammar")
    g = llama_grammar.LlamaGrammar.from_json_schema(json.dumps(SCHEMA), verbose=False)
    assert g is not None


def test_prompt_has_fewshot_and_ends_with_dictation():
    m = build_messages("epi 1 mg IV")
    assert m[0]["role"] == "system" and m[-1] == {"role": "user", "content": "epi 1 mg IV"}
    for a in m[2::2]:
        json.loads(a["content"])  # few-shot answers are valid JSON


def test_fewshot_does_not_leak_heldout_cases():
    gold = {json.loads(l)["text"].lower() for l in open(ROOT / "eval" / "data" / "gold.jsonl", encoding="utf-8")}
    for m in build_messages("x")[1:-1]:
        assert m["content"].lower() not in gold


# ------------------------------------------------------------------ post-processing
def test_given_med_with_relative_time():
    events, held = postprocess({"events": [ev(drug_spoken="Ondem", dose=4, unit="mg", route="IV",
                                              time_phrase="teen minute pehle")]},
                               "teen minute pehle Ondem 4 mg IV diya", TL)
    (e,) = events
    assert (e["drug"], e["dose"], e["route"], e["time_source"]) == ("ondansetron", 4.0, "IV", "relative")
    assert e["time_said"] == (TL - timedelta(minutes=3)).isoformat() and not e["needs_confirmation"]


@pytest.mark.parametrize("status", ["not_given", "planned"])
def test_not_given_and_planned_are_held_not_recorded(status):
    events, held = postprocess({"events": [ev(drug_spoken="aspirin", status=status)]}, "no aspirin, he's allergic", TL)
    assert meds(events) == [] and held[0]["drug"] == "aspirin"


def test_inferred_dose_needs_confirmation():
    (e,), _ = postprocess({"events": [ev(drug_spoken="D50", dose=25, unit="g", route="IV")]}, "gave him an amp of D50", TL)
    assert e["dose"] == 25 and e["needs_confirmation"] and "not spoken" in e["confirm_reasons"][0]


def test_spoken_am_pm_time_is_preserved_and_displayed_as_event_time(capsys):
    logged = TL.replace(hour=18, minute=0)
    (event,), _ = postprocess({"events": [ev(drug_spoken="aspirin", dose=100, unit="mg", route="PO",
                                                time_phrase="at 5 PM")]},
                             "give aspirin 100 milligrams orally at 5 PM", logged)
    assert event["time_source"] == "explicit"
    assert event["time_said"] == logged.replace(hour=17).isoformat()
    from asr.live import Pipeline
    Pipeline._print(object.__new__(Pipeline), {
        "seq": 1, "time_logged": logged.isoformat(), "audio_s": 1.0, "latency_s": 0.1,
        "corrections": [], "text": "aspirin 100 mg PO at 5 PM", "asr_confidence": 1.0,
        "events": [event], "extract_meta": {},
    }, False)
    assert "@ 17:00:00" in capsys.readouterr().out


def test_ambiguous_bare_hour_does_not_override_logged_time():
    (event,), _ = postprocess({"events": [ev(drug_spoken="aspirin", dose=100, unit="mg", route="PO",
                                                time_phrase="at 5")]},
                             "give aspirin 100 milligrams orally at 5", TL)
    assert event["time_source"] == "logged" and event["time_said"] == TL.isoformat()


def test_hallucinated_drug_needs_confirmation():
    (e,), _ = postprocess({"events": [ev(drug_spoken="morphine", dose=4, unit="mg", route="IV")]}, "gave 4 mg IV", TL)
    assert "drug name not in transcript" in e["confirm_reasons"]


def test_unknown_drug_and_off_protocol_route():
    (a, b), _ = postprocess({"events": [ev(drug_spoken="ketorolac", dose=30, unit="mg", route="IV"),
                                         ev(drug_spoken="aspirin", dose=325, unit="mg", route="IV")]},
                            "ketorolac 30 mg IV and aspirin 325 mg IV", TL)
    assert a["event_type"] == "clarification" and "drug" in a["resolution_required"]
    assert a["drug"] is None and a["needs_confirmation"]
    assert any("route IV not in protocol" in r for r in b["confirm_reasons"])


def test_live_unknown_drug_candidate_requires_explicit_name_resolution(tmp_path):
    from asr.engine import Transcript
    from asr.live import Pipeline
    from asr.vad import Clip

    class Engine:
        said = iter(["Give the patient", "aspirin"])

        def transcribe(self, audio):
            return Transcript(text=next(self.said), audio_s=1.0, engine="fake")

    class Extractor:
        def extract(self, text, time_logged, asr=None):
            return ([{"event_type": "clarification", "candidate_event_type": "medication", "drug": None,
                      "resolution_required": ["drug"], "time_said": time_logged.isoformat(),
                      "time_source": "logged", "needs_confirmation": True,
                      "confirm_reasons": ["unknown drug 'None'", "no dose", "no route"]}],
                    {"source": "hybrid", "held": []})

    pipe = Pipeline(Engine(), "INC-UNKNOWN", tmp_path, keys_dir=tmp_path / "keys", extractor=Extractor())
    clip = Clip(np.zeros(16000, np.float32), 0, 1)
    first = pipe.handle(clip, TL)
    assert first["events"][0]["event_type"] == "clarification"
    assert not any(json.loads(e.body)["kind"] == "event" for e in pipe.chain.entries())

    resolved = pipe.handle(clip, TL + timedelta(seconds=2))
    med = resolved["events"][0]
    assert resolved["type"] == "resolution" and med["event_type"] == "medication"
    assert med["drug"] == "aspirin" and med["needs_confirmation"]
    assert any(json.loads(e.body)["kind"] == "event" for e in pipe.chain.entries())


def test_implied_unit_and_spoken_number_words():
    (e,), _ = postprocess({"events": [ev(drug_spoken="aspirin", dose=324, route="PO")]}, "aspirin three twenty four chewed", TL)
    assert e["unit"] == "mg" and not e["needs_confirmation"]


def test_oracle_llm_scores_perfectly():
    """A perfect LLM answer must survive post-processing unchanged -> 100% on the whole gold set."""
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from score import load_jsonl, score_extraction
    inv = {"bgl_mg_dl": "bgl"}
    rows = load_jsonl(ROOT / "eval" / "data" / "gold.jsonl")
    preds = []
    for c in rows:
        out = []
        for g in c["events"]:
            e = ev(type=g["event_type"], evidence=c["text"])
            if g["event_type"] == "medication":
                e.update(drug_spoken=g["drug_as_spoken"], dose=g["dose"], unit=g["unit"], route=g["route"])
            elif g["event_type"] == "procedure":
                e.update(procedure=g["procedure"], joules=g["params"].get("joules"))
            elif g["event_type"] == "vital":
                e["vitals"] = {inv.get(k, k): v for k, v in g["vital"].items()}
            out.append(e)
        if not out:  # gold has no event: the model would report the drug as planned
            out = [ev(drug_spoken="x", status="planned")]
        events, _ = postprocess({"events": out}, c["text"], datetime.fromisoformat(c["time_logged"]))
        preds.append({"id": c["id"], "events": events})
    res = score_extraction(rows, preds)
    assert res["headline"].f1 == 1.0 and res["exact_match"] == 1.0


# ------------------------------------------------------------------ hybrid safety policy
def test_self_correction_disagreement_is_flagged():
    t = "narcan two mg intranasal, uh, sorry, four milligrams, four"
    events, _ = hybrid({t: {"events": [ev(drug_spoken="narcan", dose=4, unit="mg", route="IN")]}}).extract(t, TL)
    (e,) = meds(events)
    assert e["dose"] == 4 and any("rules heard dose 2" in r for r in e["confirm_reasons"])


def test_llm_negation_beats_rules():
    t = "no aspirin, he's allergic"
    events, meta = hybrid({t: {"events": [ev(drug_spoken="aspirin", status="not_given")]}}).extract(t, TL)
    assert meds(events) == [] and meta["held"][0]["drug"] == "aspirin"


def test_rules_only_med_is_kept_but_flagged():
    t = "epi 1 mg IV and BP 90 over 60"
    events, _ = hybrid({t: {"events": [ev(type="vital", vitals={"sbp": 90, "dbp": 60})]}}).extract(t, TL)
    (e,) = meds(events)
    assert e["drug"] == "adrenaline" and "only the rule extractor saw this" in e["confirm_reasons"]


def test_agreement_is_not_flagged():
    t = "epi 1 mg IV"
    events, _ = hybrid({t: {"events": [ev(drug_spoken="epi", dose=1, unit="mg", route="IV")]}}).extract(t, TL)
    (e,) = meds(events)
    assert not e["needs_confirmation"]


def test_controlled_drug_always_confirmed():
    t = "fentanyl 50 mcg IV"
    events, _ = hybrid({t: {"events": [ev(drug_spoken="fentanyl", dose=50, unit="mcg", route="IV")]}}).extract(t, TL)
    assert meds(events)[0]["confirm_reasons"] == ["controlled drug"]


def test_incomplete_rule_med_is_flagged_even_without_llm():
    events, _ = HybridExtractor(llm=None).extract("No aspirin, he can't learn", TL)
    assert set(meds(events)[0]["confirm_reasons"]) == {"no dose", "no route"}


def test_llm_failure_falls_back_to_rules_all_flagged():
    events, meta = HybridExtractor(llm=LLMExtractor(Broken())).extract("epi 1 mg IV", TL)
    (e,) = meds(events)
    assert meta["source"] == "rules-fallback" and "LLM unavailable - rules only" in e["confirm_reasons"]


def test_asr_autocorrected_drug_is_flagged():
    t = "naloxone 0.4 mg IM"
    asr = {"corrections": [{"heard": "now look so", "fixed": "naloxone", "kind": "drug", "score": 0.72}]}
    events, _ = hybrid({t: {"events": [ev(drug_spoken="naloxone", dose=0.4, unit="mg", route="IM")]}}).extract(t, TL, asr)
    assert any("auto-corrected from 'now look so'" in r for r in meds(events)[0]["confirm_reasons"])


# ------------------------------------------------------------------ confirmation
@pytest.mark.parametrize("said,cmd", [
    ("Confirm.", "confirm"), ("yes", "confirm"), ("haan ji", "confirm"), ("theek hai", "confirm"),
    ("that's right", "confirm"), ("Cancel that.", "cancel"), ("no", "cancel"), ("galat hai", "cancel"),
    ("scratch that", "cancel"), ("cancel karo", "cancel"),
    ("confirmed confirmed confirmed confirmed confirmed", "confirm"),
    ("no aspirin, he's allergic", None), ("yes we gave epi 1 mg IV", None), ("no response to pain", None),
])
def test_parse_command(said, cmd):
    assert parse_command(said) == cmd


def test_speakable_prompt():
    assert speakable({"event_type": "medication", "drug": "fentanyl", "dose": 50.0, "unit": "mcg", "route": "IV"}) \
        == "Confirm fentanyl 50 micrograms I V?"


def test_queue_latest_first_and_expiry():
    q = ConfirmationQueue(timeout_s=60)
    q.add("e1", {"event_type": "medication", "drug": "morphine"}, TL)
    q.add("e2", {"event_type": "medication", "drug": "fentanyl"}, TL + timedelta(seconds=30))
    assert q.handle("confirm", TL + timedelta(seconds=35))["event_id"] == "e2"
    (exp,) = q.expire(TL + timedelta(seconds=70))
    assert exp == {"action": "unconfirmed", "event_id": "e1", "at": (TL + timedelta(seconds=70)).isoformat(), "said": ""}


def test_unknown_medication_cannot_be_confirmed_before_name_resolution():
    q = ConfirmationQueue()
    q.add("e1", {"event_type": "clarification", "resolution_required": ["drug"]}, TL)
    result = q.handle("confirm", TL + timedelta(seconds=1), said="confirm")
    assert result["action"] == "needs_resolution" and "e1" in q.pending


def test_live_pipeline_dictation_then_spoken_confirm(tmp_path):
    from asr.engine import Transcript
    from asr.live import Pipeline
    from asr.vad import Clip

    class Engine:
        name = "fake"
        said = iter(["fentanyl 50 mcg IV", "confirm"])

        def transcribe(self, audio):
            return Transcript(text=next(self.said), audio_s=1.0, engine="fake")

    pipe = Pipeline(Engine(), "INC-C", tmp_path, keys_dir=tmp_path / "keys", extractor=hybrid(
        {"fentanyl 50 mcg IV": {"events": [ev(drug_spoken="fentanyl", dose=50, unit="mcg", route="IV")]}}))
    clip = Clip(np.zeros(16000, np.float32), 0, 1)
    r1 = pipe.handle(clip, TL)
    r2 = pipe.handle(clip, TL + timedelta(seconds=5))
    assert r1["events"][0]["needs_confirmation"] and r2["type"] == "command"
    assert r2["confirmation"] == {"action": "confirmed", "event_id": r1["events"][0]["event_id"],
                                  "at": (TL + timedelta(seconds=5)).isoformat(), "said": "confirm"}


def test_silent_llm_never_worse_than_rules_and_nothing_unflagged():
    """Worst-case model (answers nothing): accuracy must equal rules-only, and every wrong dose is flagged."""
    import sys
    sys.path.insert(0, str(ROOT / "eval"))
    from rule_extractor import extract as rule_extract
    from score import load_jsonl, score_extraction

    class Silent:
        name = "silent"

        def generate(self, messages, schema):
            return '{"events": []}'

    rows = load_jsonl(ROOT / "eval" / "data" / "gold.jsonl")
    hx = HybridExtractor(llm=LLMExtractor(Silent()))
    tl = lambda c: datetime.fromisoformat(c["time_logged"])  # noqa: E731
    hyb = score_extraction(rows, [{"id": c["id"], "events": hx.extract(c["text"], tl(c))[0]} for c in rows])
    rul = score_extraction(rows, [{"id": c["id"], "events": rule_extract(c["text"], tl(c))} for c in rows])
    assert hyb["headline"].f1 == rul["headline"].f1
    assert hyb["safety"]["wrong"] == hyb["safety"]["wrong_flagged"]

def test_unit_disagreement_is_flagged():
    t = "patient received ondansetron four migs through the line"
    events, _ = hybrid({t: {"events": [ev(drug_spoken="ondansetron", dose=4, unit="mL", route="IV")]}}).extract(t, TL)
    assert "rules heard unit mg" in meds(events)[0]["confirm_reasons"]
