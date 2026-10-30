"""
Black Box - rule-based extractor (baseline + Phase 2 safety-net fallback).

text -> list of compact events (same format as eval/data/gold.jsonl).
Deterministic, no model, runs anywhere. The LLM extractor in Phase 2 has to beat this.

CLI:
  python rule_extractor.py --input eval/data/gold.jsonl --out eval/preds/rules.jsonl
  python rule_extractor.py --input eval/preds/asr_small.jsonl --condition siren_5db --out eval/preds/rules_on_siren5.jsonl
  python rule_extractor.py --text "epi one milligram IV, two minutes ago"
"""
from __future__ import annotations

import argparse
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from normalize import resolve_time, words_to_numbers
from protocol import ProtocolBook

BOOK = ProtocolBook.load()

# --- drug aliases, longest first so "normal saline" beats "saline"
_ALIASES = sorted({a.lower() for d in BOOK.p.drugs.values() for a in d.aliases} | set(BOOK.p.drugs),
                  key=len, reverse=True)
DRUG_RE = re.compile(r"(?<![a-z0-9])(" + "|".join(re.escape(a.replace("_", " ")) for a in _ALIASES)
                     + r"|normal_saline)(?![a-z0-9])")

UNIT_PATTERNS = [
    ("mcg", r"micrograms?|mcgs?|mics?|mikes?|ug"),
    ("mg", r"milligrams?|mgs?|migs?"),
    ("mL", r"millilit(?:er|re)s?|mls?|cc"),
    ("g", r"grams?|gms?|g"),
]
UNIT_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)\s*(" + "|".join(p for _, p in UNIT_PATTERNS) + r")(?![a-z])")
NUM_RE = re.compile(r"(?<![\w.])(\d+(?:\.\d+)?)(?![\w.]|\.\d)")

ROUTE_PATTERNS = [
    ("IO", r"\bio\b|intraosseous\w*"),
    ("IV", r"\biv\b|intravenous\w*|through the line|line (?:mein|me)\b"),
    ("IM", r"\bim\b|intramuscular\w*|in the thigh|thigh (?:mein|me)\b"),
    ("IN", r"intranasal\w*|up the nose|\bnasal\w*|naak (?:mein|me)\b"),
    ("SL", r"sublingual\w*|under the tongue|\bsl\b|jeebh ke neeche"),
    ("ODT", r"\bodt\b|dissolving tablet|\bwafer\b"),
    ("NEB", r"nebuli[sz]\w*"),
    ("PO", r"\bchew\w*|orally|\bpo\b|by mouth|munh se"),
]

PROC_PATTERNS = [
    ("defibrillation", r"\bshock(?:ed)?\b|\bdefib\w*|\bdefibrillat\w*"),
    ("cpr", r"\bcpr\b|\bcompressions?\b"),
    ("intubation", r"\bintubat\w*|\btubed\b|\bet tube\b|\bett\b"),
    ("io_access", r"\bio (?:access|needle)\b|intraosseous access"),
    ("iv_access", r"\biv access\b|\bcannula\w*|\b\d+ gauge\b|\bline laga\w*|\bgot a line\b"),
    ("ecg_12_lead", r"\b12 lead\b"),
]
JOULES_RE = re.compile(r"(\d+)\s*(?:joules?|j)\b")

CANCEL_RE = re.compile(r"not (?:yet )?given|not given yet|drawing up|drawn up|prepping|preparing|"
                       r"\bholding\b|\bhold\b|nahi diya|mat do|about to give|going to give|ready to give|"
                       r"wait for my call|ready hai")

_V = r"\s*(?:is|of|was|hai|at|reading|level)?\s*"
VITAL_PATTERNS = [
    ("bp", re.compile(r"\b(?:bp|blood pressure|pressure)" + _V + r"(\d{2,3})\s*(?:over|/|by)\s*(\d{2,3})")),
    ("hr", re.compile(r"\b(?:hr|heart rate|pulse)" + _V + r"(\d{2,3})")),
    ("spo2", re.compile(r"\b(?:spo2|sats?|saturation|o2 sats?)" + _V + r"(\d{2,3})")),
    ("rr", re.compile(r"\b(?:rr|resp(?:iratory)? rate|resps)" + _V + r"(\d{1,2})")),
    ("gcs", re.compile(r"\bgcs" + _V + r"(\d{1,2})")),
    ("bgl_mg_dl", re.compile(r"\b(?:blood sugar|sugar|glucose|bgl|cbg|grbs)" + _V + r"(\d{2,3})(?!\s*(?:g|grams?)\b)")),
]
NOT_DOSE_RE = re.compile(r"(?<![\w.])\d+(?:\.\d+)?\s*(?:kg|kgs|kilos?|kilograms?|years?|yrs?|saal|months?|"
                         r"minutes?|mins?|hours?|joules?|j|gauge|%|percent|over|by|/)(?![a-z])"
                         r"|(?:weight|wazan|age|umar|aged)\s*(?:is|of|hai)?\s*\d+(?:\.\d+)?")
CLAUSE_SPLIT = re.compile(r"\s+(?:and|then|phir|aur)\s+|;")


def _num(s: str) -> float:
    return float(s)


def _extract_vitals(text: str) -> tuple[dict, str]:
    vit: dict = {}
    for name, rx in VITAL_PATTERNS:
        m = rx.search(text)
        if not m:
            continue
        if name == "bp":
            vit["sbp"], vit["dbp"] = int(m.group(1)), int(m.group(2))
        elif name == "bgl_mg_dl":
            vit[name] = float(m.group(1))
        else:
            vit[name] = int(m.group(1))
        text = text[:m.start()] + " " + text[m.end():]
    return vit, text


def _extract_med(clause: str) -> dict | None:
    m = DRUG_RE.search(clause)
    if not m:
        return None
    spoken = m.group(1)
    drug = BOOK.resolve_drug(spoken) or BOOK.resolve_drug(spoken.replace(" ", "_"))
    # remove the drug token so "d50" etc. never leaks a number
    rest = clause[:m.start()] + " " + clause[m.end():]
    dose = unit = None
    um = UNIT_RE.search(rest)
    if um:
        dose = _num(um.group(1))
        raw = um.group(2)
        unit = next(u for u, p in UNIT_PATTERNS if re.fullmatch(p, raw))
    else:
                # numbers that are weights / ages / times / energies are never doses ("weight seventy kilos")
        rest_nd = NOT_DOSE_RE.sub(" ", rest)
        nums = [_num(x) for x in NUM_RE.findall(rest_nd) if _num(x) > 0]
        if nums:
            dose = nums[0]
            unit = BOOK.p.drugs[drug].canonical_unit  # implied unit ("aspirin three twenty four")
    if dose == 0:  # "zero five" after ASR dropped "point" - never accept a zero dose
        dose = unit = None
    route = next((r for r, p in ROUTE_PATTERNS if re.search(p, rest)), None)
    ev = {"event_type": "medication", "drug": drug, "drug_as_spoken": spoken}
    if dose is not None:
        ev["dose"], ev["unit"] = dose, unit
    if route:
        ev["route"] = route
    return ev


def extract(text: str, time_logged: datetime) -> list[dict]:
    norm = words_to_numbers(text)
    t_said, t_src, span = resolve_time(norm, time_logged)
    if span:
        norm = norm[:span[0]] + " " + norm[span[1]:]

    events: list[dict] = []
    vitals: dict = {}
    for clause in CLAUSE_SPLIT.split(norm):
        clause = clause.strip(" ,.")
        if not clause:
            continue
        v, clause_wo_vitals = _extract_vitals(clause)
        vitals.update(v)
        for name, pat in PROC_PATTERNS:
            if re.search(pat, clause):
                ev = {"event_type": "procedure", "procedure": name, "params": {}}
                jm = JOULES_RE.search(clause)
                if name == "defibrillation" and jm:
                    ev["params"]["joules"] = int(jm.group(1))
                events.append(ev)
                break
        if not CANCEL_RE.search(clause):
            med = _extract_med(clause_wo_vitals)
            if med:
                events.append(med)
    if vitals:
        events.append({"event_type": "vital", "vital": vitals})
    if not events and not CANCEL_RE.search(norm):
        events.append({"event_type": "note"})
    for ev in events:
        ev["time_said"], ev["time_source"] = t_said.isoformat(), t_src
    return events


def select_rows(rows: list[dict], condition: str | None, text_field: str) -> list[dict]:
    """Select a single ASR condition and fail clearly on incomplete input."""
    selected = rows
    if condition is not None:
        selected = [r for r in rows if r.get("condition") == condition]
        if not selected:
            conditions = sorted({str(r.get("condition")) for r in rows})
            raise SystemExit(f"no rows with condition={condition!r}; available: {', '.join(conditions) or '(none)'}")
    else:
        conditions = {r.get("condition") for r in selected if r.get("condition") is not None}
        if len(conditions) > 1:
            raise SystemExit(f"input mixes conditions: {', '.join(sorted(map(str, conditions)))}; specify --condition")
    for row in selected:
        if text_field not in row:
            raise SystemExit(f"row {row.get('id', '(unknown)')} lacks [{text_field!r}]")
    return selected


# ------------------------------------------------------------------ CLI
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", help="jsonl with id, text, time_logged (gold or ASR output)")
    ap.add_argument("--condition", help="only lines with this ASR condition (e.g. clean, siren_5db)")
    ap.add_argument("--out", help="output jsonl")
    ap.add_argument("--text", help="extract a single utterance and print it")
    a = ap.parse_args()

    if a.text:
        print(json.dumps(extract(a.text, datetime.now(timezone.utc)), indent=2))
        return
    rows = [json.loads(l) for l in open(a.input, encoding="utf-8") if l.strip()]
    if a.condition:
        rows = [r for r in rows if r.get("condition") == a.condition]
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    with open(a.out, "w", encoding="utf-8") as f:
        for r in rows:
            evs = extract(r["text"], datetime.fromisoformat(r["time_logged"]))
            f.write(json.dumps({"id": r["id"], "events": evs}, ensure_ascii=False) + "\n")
    print(f"extracted {len(rows)} utterances -> {a.out}")


if __name__ == "__main__":
    main()
