"""Tests for Phase 4: Guardian rules, patient context, alert answers, chain integration.  python -m pytest -q"""
import json
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

import numpy as np
import pytest

from guardian.context import parse_context
from guardian.guardian import Guardian, parse_alert_response

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "eval"))
from guardian_scenes import SCENES  # noqa: E402
from score_guardian import run_scene  # noqa: E402

T0 = datetime(2026, 10, 30, 9, 0, tzinfo=timezone.utc)


def med(eid, drug, dose, unit, route, minute):
    return {"event_type": "medication", "event_id": eid, "drug": drug, "dose": dose, "unit": unit, "route": route,
            "time_said": (T0 + timedelta(minutes=minute)).isoformat()}


# ------------------------------------------------------------------ every scripted scene
@pytest.mark.parametrize("scene", SCENES, ids=[s["name"] for s in SCENES])
def test_scene_alerts_exactly_as_expected(scene):
    got = {(i, a.rule_id) for i, a in run_scene(scene)}
    assert got == scene["expected"]


# ------------------------------------------------------------------ context
@pytest.mark.parametrize("text,expected", [
    ("patient weight twenty kilos, six years old", {"weight_kg": 20.0, "age_years": 6.0}),
    ("wazan 30 kilo, umar 10 saal", {"weight_kg": 30.0, "age_years": 10.0}),
    ("eighteen month old", {"age_years": 1.5}),
    ("known allergy to penicillin and NSAIDs", {"allergies": ["nsaid", "penicillin"]}),
    ("allergic to morphine and he took viagra", {"allergies": ["morphine"], "flags": ["pde5_inhibitor_48h"]}),
    ("morphine se allergy hai", {"allergies": ["morphine"]}),
    ("no known allergies", {}),
    ("BP eighty five over fifty", {}),
])
def test_parse_context(text, expected):
    assert parse_context(text) == expected


def test_weight_is_never_a_dose():
    from rule_extractor import extract
    evs = extract("patient weight seventy kilos, giving morphine", T0)
    m = [e for e in evs if e["event_type"] == "medication"][0]
    assert m["drug"] == "morphine" and "dose" not in m


def test_allergy_mention_is_not_an_administration():
    from extract.hybrid import HybridExtractor
    g = Guardian()
    events, _ = HybridExtractor(llm=None).extract("allergic to morphine", T0)
    kept, alerts = g.assess(events, "allergic to morphine", g.observe_context("allergic to morphine"))
    assert [e for e in kept if e["event_type"] == "medication"] == [] and alerts == []


# ------------------------------------------------------------------ answers
@pytest.mark.parametrize("said,action,reason", [
    ("Acknowledged.", "acknowledged", None), ("theek hai", "acknowledged", None),
    ("override, rhythm changed", "overridden", "rhythm changed"),
    ("override because patient still in VF", "overridden", "patient still in vf"),
    ("Override.", "need_reason", None), ("cancel dose", "cancelled_dose", None), ("mat do", "cancelled_dose", None),
])
def test_parse_alert_response(said, action, reason):
    r = parse_alert_response(said)
    assert r["action"] == action and r["reason"] == reason


def test_dictation_is_not_an_answer():
    assert parse_alert_response("no aspirin he is allergic") is None
    assert parse_alert_response("epi one milligram IV") is None


def test_one_answer_covers_all_alerts_on_the_dose_and_cancel_unregisters():
    g = Guardian()
    g.observe_context("allergic to morphine")
    _, alerts = g.assess([med("m1", "morphine", 30.0, "mg", "IV", 0)])
    assert {a.rule_id for a in alerts} == {"CONTRAINDICATED", "DOSE_ABOVE_MAX", "TOTAL_ABOVE_MAX"}
    for a in alerts:
        g.add_pending(a, T0)
    resp = g.respond("cancel dose", T0)
    assert len(resp) == 3 and {r["action"] for r in resp} == {"cancelled_dose"} and not g.pending
    _, alerts2 = g.assess([med("m2", "morphine", 4.0, "mg", "IV", 10)])
    assert "TOTAL_ABOVE_MAX" not in {a.rule_id for a in alerts2}       # the cancelled 30 mg no longer counts


def test_unanswered_alert_expires():
    g = Guardian(alert_timeout_s=60)
    _, alerts = g.assess([med("a1", "amiodarone", 50.0, "mg", "IV", 0)])
    g.add_pending(alerts[0], T0)
    (exp,) = g.expire(T0 + timedelta(seconds=61))
    assert exp["action"] == "unanswered" and exp["rule_id"] == "DOSE_BELOW_MIN"


def test_info_alerts_do_not_wait_for_an_answer():
    g = Guardian()
    g.observe_context("eight year old")
    _, alerts = g.assess([med("x", "midazolam", 5.0, "mg", "IM", 0)])
    g.add_pending(alerts[0], T0)
    assert alerts[0].rule_id == "NEEDS_WEIGHT" and not g.pending


# ------------------------------------------------------------------ live pipeline + chain
class Engine:
    def __init__(self, lines):
        self.lines = iter(lines)

    def transcribe(self, audio):
        from asr.engine import Transcript
        return Transcript(text=next(self.lines), audio_s=1.0, engine="fake")


def run_pipeline(tmp_path, lines, incident="INC-G", start_min=0):
    from asr.live import Pipeline
    from asr.vad import Clip
    pipe = Pipeline(Engine(lines), incident, tmp_path, extract=True, keys_dir=tmp_path / "keys")
    for i, _ in enumerate(lines):
        pipe.handle(Clip((np.random.default_rng(i).standard_normal(8000) * 0.1).astype(np.float32), i, i + 0.5),
                    T0 + timedelta(minutes=start_min + i))
    return pipe


def test_alert_and_override_are_chained_and_verifiable(tmp_path):
    from audit.verify import materialize, verify_chain
    pipe = run_pipeline(tmp_path, ["epi one milligram IV", "epi one milligram IV", "override, rhythm changed"])
    entries = list(pipe.chain.entries())
    kinds = [json.loads(e.body)["kind"] for e in entries]
    assert kinds.count("alert") == 1 and kinds.count("alert_response") == 1
    resp = [json.loads(e.body)["payload"] for e in entries if json.loads(e.body)["kind"] == "alert_response"][0]
    alert_entry = [e for e in entries if json.loads(e.body)["kind"] == "alert"][0]
    assert resp["action"] == "overridden" and resp["reason"] == "rhythm changed"
    assert resp["alert_ref"] == alert_entry.entry_hash                     # answer points at the exact alert
    assert verify_chain(entries, pipe.keys.public_key, tmp_path, pipe.keys.audio_key).ok
    rec = materialize(entries, "INC-G")
    assert rec[1]["alerts"][0]["response"]["action"] == "overridden"


def test_cancel_dose_via_alert_cancels_event(tmp_path):
    from audit.verify import materialize
    pipe = run_pipeline(tmp_path, ["allergic to morphine", "morphine four milligrams IV", "cancel dose"])
    rec = materialize(pipe.chain.entries(), "INC-G")
    (m,) = [x for x in rec if x["event"].get("drug") == "morphine"]
    assert m["status"] == "cancelled" and m["alerts"][0]["rule_id"] == "CONTRAINDICATED"
    assert not pipe.queue.pending                                        # read-back for it was withdrawn too


def test_guardian_state_survives_restart(tmp_path):
    run_pipeline(tmp_path, ["patient weight twenty kilos, six years old", "epi zero point two milligrams IV"])
    pipe2 = run_pipeline(tmp_path, ["epi zero point two milligrams IV"], start_min=2)  # new process, same incident
    alerts = [json.loads(e.body)["payload"]["rule_id"] for e in pipe2.chain.entries()
              if json.loads(e.body)["kind"] == "alert"]
    assert alerts == ["REPEAT_TOO_SOON"]   # knew about the earlier dose AND still used the child's weight
    assert pipe2.guardian.patient.weight_kg == 20.0