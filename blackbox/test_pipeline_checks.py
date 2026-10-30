"""Regression tests for the eval pipeline's input checks: ASR runner -> rule_extractor -> score.py.
An incomplete or mismatched input must be an error, never a quiet 0.0.   python -m pytest -q"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).parent
sys.path.insert(0, str(ROOT / "eval"))

import run_asr  # noqa: E402
import score  # noqa: E402
from rule_extractor import select_rows  # noqa: E402

GOLD = score.load_jsonl(ROOT / "eval" / "data" / "gold.jsonl")
G = {g["id"]: g for g in GOLD}
TL = G["BB-030"]["time_logged"]


def asr_row(cid, cond, text="naloxone 4 mg up the nose", **kw):
    return {"id": cid, "condition": cond, "text": text, "raw_text": text, "time_logged": TL, **kw}


def write_jsonl(path, rows):
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))
    return path


# ------------------------------------------------------------------ run_asr: discovery + counts
@pytest.fixture
def audio(tmp_path):
    for cond, ids in {"clean": ["BB-030", "BB-044"], "siren_5db": ["BB-030", "BB-044"]}.items():
        (tmp_path / cond).mkdir()
        for i in ids:
            (tmp_path / cond / f"{i}.wav").touch()
    return tmp_path


def test_plan_covers_every_condition(audio):
    plan = run_asr.plan_conditions(audio, G, None)
    assert {c: len(w) for c, w in plan.items()} == {"clean": 2, "siren_5db": 2}


def test_plan_rejects_unknown_condition(audio):
    with pytest.raises(SystemExit, match="unknown condition"):
        run_asr.plan_conditions(audio, G, ["siren_50db"])


def test_plan_rejects_empty_condition(audio):
    (audio / "road_0db").mkdir()
    with pytest.raises(SystemExit, match="road_0db: no .wav"):
        run_asr.plan_conditions(audio, G, None)


def test_plan_rejects_wav_not_in_gold(audio):
    (audio / "clean" / "XX-999.wav").touch()
    with pytest.raises(SystemExit, match="XX-999"):
        run_asr.plan_conditions(audio, G, None)


def test_count_check_catches_truncated_run(audio, tmp_path):
    """The original bug: run stopped partway, file looked complete."""
    plan = run_asr.plan_conditions(audio, G, None)
    out = write_jsonl(tmp_path / "o.jsonl", [asr_row("BB-030", "clean"), asr_row("BB-044", "clean"),
                                             asr_row("BB-030", "siren_5db")])
    bad = run_asr.check_counts(out, plan)
    assert any("siren_5db: expected 2, got 1" in b and "BB-044" in b for b in bad)
    write_jsonl(out, [asr_row(i, c) for c in ("clean", "siren_5db") for i in ("BB-030", "BB-044")])
    assert run_asr.check_counts(out, plan) == []


# ------------------------------------------------------------------ rule_extractor input selection
def test_extractor_missing_condition_fails():
    with pytest.raises(SystemExit, match="no rows with condition='siren_5db'.*clean"):
        select_rows([asr_row("BB-030", "clean")], "siren_5db", "text")


def test_extractor_missing_text_field_fails():
    r = asr_row("BB-030", "clean")
    del r["raw_text"]
    with pytest.raises(SystemExit, match="lacks \\['raw_text'\\]"):
        select_rows([r], "clean", "raw_text")


def test_extractor_refuses_mixed_conditions():
    with pytest.raises(SystemExit, match="mixes conditions"):
        select_rows([asr_row("BB-030", "clean"), asr_row("BB-030", "siren_5db")], None, "text")


def test_extractor_uses_requested_field():
    rows = select_rows([asr_row("BB-030", "clean"), asr_row("BB-030", "tts")], "clean", "raw_text")
    assert [r["condition"] for r in rows] == ["clean"]


# ------------------------------------------------------------------ score.py extraction
def test_empty_predictions_fail():
    with pytest.raises(SystemExit, match="empty"):
        score.check_preds(GOLD, [])


def test_mismatched_prediction_ids_fail():
    with pytest.raises(SystemExit, match="not in gold.*XX-1"):
        score.check_preds(GOLD, [{"id": "XX-1", "events": []}])


def test_duplicate_prediction_ids_fail():
    with pytest.raises(SystemExit, match="duplicate"):
        score.check_preds(GOLD, [{"id": "BB-030", "events": []}] * 2)


def test_zero_matched_cases_is_an_error_not_a_score():
    with pytest.raises(ValueError, match="0 gold cases"):
        score.score_extraction([G["BB-030"]], [{"id": "BB-044", "events": []}])


# ------------------------------------------------------------------ score.py ASR
def test_asr_missing_field_fails_instead_of_falling_back():
    r = asr_row("BB-030", "clean")
    del r["raw_text"]
    with pytest.raises(SystemExit, match="lack 'raw_text'"):
        score.check_asr(GOLD, [r], "raw_text")


def test_asr_unknown_ids_fail():
    with pytest.raises(SystemExit, match="not in gold"):
        score.check_asr(GOLD, [asr_row("XX-1", "clean")])


def test_asr_safety_fields_scored_separately():
    """BB-030 is naloxone 4 mg IN. A transcript with the right drug+dose but 'IV' must miss route only."""
    stats, detail = score.score_asr(GOLD, [asr_row("BB-030", "clean", "administered naloxone 4 mg IV")])
    assert detail[0]["missed"] == ["route:IN"]
    s = stats["clean"]
    assert (s["drug_hit"], s["dose_hit"], s["unit_hit"], s["route_hit"]) == (1, 1, 1, 0)


def _score(*args):
    return subprocess.run([sys.executable, str(ROOT / "eval" / "score.py"), *map(str, args)],
                          capture_output=True, text=True)


def test_cli_scopes_asr_to_prediction_condition(tmp_path):
    asr = write_jsonl(tmp_path / "asr.jsonl", [asr_row("BB-030", "siren_5db"), asr_row("BB-030", "tts", "x")])
    pred = write_jsonl(tmp_path / "p.jsonl", [{"id": "BB-030", "condition": "siren_5db", "events": []}])
    r = _score("--pred", pred, "--asr", asr)
    assert r.returncode == 0, r.stderr
    asr_part = r.stdout.split("=== ASR")[1]
    assert "[siren_5db]" in asr_part and "[tts]" not in asr_part and "\ntts " not in asr_part


def test_cli_empty_pred_exits_nonzero(tmp_path):
    r = _score("--pred", write_jsonl(tmp_path / "p.jsonl", []))
    assert r.returncode != 0 and "empty" in r.stderr


def test_cli_same_utts_check(tmp_path):
    a = write_jsonl(tmp_path / "a.jsonl", [asr_row("BB-030", "clean"), asr_row("BB-044", "clean")])
    b = write_jsonl(tmp_path / "b.jsonl", [asr_row("BB-030", "clean")])
    r = _score("--asr", a, "--same-utts-as", b)
    assert r.returncode != 0 and "differ" in r.stderr
    assert _score("--asr", a, "--same-utts-as", a).returncode == 0
