"""
Black Box - synthetic dictation generator (Phase 1 eval set).

Writes:
  eval/data/gold.jsonl            one dictation per line + ground-truth events
  eval/data/recording_script.csv  the subset your team reads aloud (split across P1/P2/P3)

Every gold event is validated against schema.ClinicalEvent, so the eval set can
never drift away from the real schema.

Run:  python eval/generate_dataset.py            (deterministic, seed=7)
"""
from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from protocol import ProtocolBook  # noqa: E402
from schema import ClinicalEvent  # noqa: E402

OUT = ROOT / "eval" / "data"
IST = timezone(timedelta(hours=5, minutes=30))
BASE = datetime(2026, 10, 25, 9, 0, 0, tzinfo=IST)
FAKE_AUDIO = hashlib.sha256(b"placeholder").hexdigest()
BOOK = ProtocolBook.load()
rng = random.Random(7)

# ------------------------------------------------------------------ vocab
#        drug            indication                 route  doses (spoken unit)   unit
MEDS = [
    ("adrenaline",    "cardiac_arrest",          "IV",  [1],             "mg"),
    ("adrenaline",    "cardiac_arrest",          "IO",  [1],             "mg"),
    ("adrenaline",    "anaphylaxis",             "IM",  [0.3, 0.5],      "mg"),
    ("amiodarone",    "cardiac_arrest",          "IV",  [300, 150],      "mg"),
    ("aspirin",       "acute_coronary_syndrome", "PO",  [324, 325, 162], "mg"),
    ("nitroglycerin", "acute_coronary_syndrome", "SL",  [0.4],           "mg"),
    ("naloxone",      "opioid_overdose",         "IN",  [4, 2],          "mg"),
    ("naloxone",      "opioid_overdose",         "IV",  [0.4],           "mg"),
    ("naloxone",      "opioid_overdose",         "IM",  [0.4],           "mg"),
    ("midazolam",     "seizure",                 "IM",  [10, 5],         "mg"),
    ("midazolam",     "seizure",                 "IV",  [2.5, 5],        "mg"),
    ("fentanyl",      "pain",                    "IN",  [50, 100],       "mcg"),
    ("fentanyl",      "pain",                    "IV",  [25, 50],        "mcg"),
    ("morphine",      "pain",                    "IV",  [4, 5, 2],       "mg"),
    ("ondansetron",   "nausea_vomiting",         "IV",  [4],             "mg"),
    ("ondansetron",   "nausea_vomiting",         "ODT", [4],             "mg"),
    ("salbutamol",    "bronchospasm",            "NEB", [5, 2.5],        "mg"),
    ("dextrose",      "hypoglycemia",            "IV",  [25, 10],        "g"),
    ("glucagon",      "hypoglycemia",            "IM",  [1],             "mg"),
    ("normal_saline", "hypoperfusion",           "IV",  [500, 250, 1000], "mL"),
]

SPOKEN_EN = {
    "adrenaline": ["adrenaline", "epi", "epinephrine"],
    "amiodarone": ["amiodarone", "amio"],
    "aspirin": ["aspirin", "ASA"],
    "nitroglycerin": ["nitro", "GTN", "nitroglycerin"],
    "naloxone": ["naloxone", "narcan"],
    "midazolam": ["midazolam", "versed", "midaz"],
    "fentanyl": ["fentanyl", "fent"],
    "morphine": ["morphine"],
    "ondansetron": ["ondansetron", "zofran"],
    "salbutamol": ["salbutamol", "albuterol", "ventolin"],
    "dextrose": ["dextrose", "D50"],
    "glucagon": ["glucagon"],
    "normal_saline": ["normal saline", "saline", "NS"],
}
# Indian brand names (all present as aliases in protocol.json)
SPOKEN_BRAND = {
    "aspirin": ["Ecosprin", "Disprin"],
    "ondansetron": ["Ondem", "Emeset"],
    "salbutamol": ["Asthalin"],
    "nitroglycerin": ["Sorbitrate"],
    "midazolam": ["Mezolam"],
}
UNIT_WORDS = {
    "mg": ["milligrams", "mg", "migs"],
    "mcg": ["micrograms", "mics", "mcg"],
    "g": ["grams", "g"],
    "mL": ["mls", "milliliters", "cc"],
}
ROUTE_WORDS = {
    "IV": ["IV", "intravenous", "through the line"],
    "IO": ["IO", "intraosseous"],
    "IM": ["IM", "intramuscular", "in the thigh"],
    "IN": ["intranasal", "up the nose", "nasal"],
    "PO": ["chewed", "orally", "PO"],
    "SL": ["sublingual", "under the tongue", "SL"],
    "NEB": ["nebulised", "via nebuliser"],
    "ODT": ["ODT", "as a dissolving tablet"],
}
ROUTE_WORDS_HI = {**ROUTE_WORDS, "IV": ["IV", "line mein"], "IN": ["naak mein", "nasal"],
                  "PO": ["munh se", "chewed"], "SL": ["jeebh ke neeche", "SL"]}
HINDI_MIN = {2: "do", 3: "teen", 5: "paanch", 10: "das", 15: "pandrah"}

EN_TEMPLATES = [
    "{d} {n} {u} {r} given{t}",
    "gave {n} {u} of {d} {r}{t}",
    "pushing {n} {u} {d} {r}",
    "{n} {u} {d} {r} in{t}",
    "okay {d} {n} {u} {r}{t}",
    "patient received {d} {n} {u} {r}{t}",
    "administered {d} {n} {u} {r}{t}",
]
HI_TEMPLATES = [
    "patient ko {d} {n} {u} {r} diya{t}",
    "{d} {n} {u} {r} de diya hai{t}",
    "abhi {d} {n} {u} {r} lagaya",
    "{n} {u} {d} {r} diya{t}",
]

# ------------------------------------------------------------------ numbers -> speech
_ONES = "zero one two three four five six seven eight nine".split()
_TEENS = "ten eleven twelve thirteen fourteen fifteen sixteen seventeen eighteen nineteen".split()
_TENS = "_ _ twenty thirty forty fifty sixty seventy eighty ninety".split()


def int_words(n: int) -> str:
    if n < 10:
        return _ONES[n]
    if n < 20:
        return _TEENS[n - 10]
    if n < 100:
        return _TENS[n // 10] + ("" if n % 10 == 0 else " " + _ONES[n % 10])
    if n < 1000:
        rest = n % 100
        return _ONES[n // 100] + " hundred" + ("" if rest == 0 else " " + int_words(rest))
    rest = n % 1000
    return int_words(n // 1000) + " thousand" + ("" if rest == 0 else " " + int_words(rest))


def speak_number(x: float, style: str) -> str:
    """style: 'digits' (typed note), 'words', or 'colloquial' ('three twenty four', 'point four')."""
    if style == "digits":
        return str(int(x)) if x == int(x) else str(x)
    if x == int(x):
        n = int(x)
        if style == "colloquial" and 100 < n < 1000 and n % 100 >= 10:
            return _ONES[n // 100] + " " + int_words(n % 100)
        return int_words(n)
    whole, frac = str(x).split(".")
    lead = "" if (whole == "0" and style == "colloquial") else int_words(int(whole)) + " "
    return lead + "point " + " ".join(_ONES[int(c)] for c in frac)


def unit_word(unit: str, dose: float) -> str:
    w = rng.choice(UNIT_WORDS[unit])
    if dose == 1 and w.endswith("s"):
        w = w[:-1]  # "one milligram", "one mig"
    return w


# ------------------------------------------------------------------ cases
cases: list[dict] = []


def logged_at(i: int) -> datetime:
    return BASE + timedelta(seconds=45 * i + rng.randint(0, 20))


def time_phrase(tl: datetime, hinglish: bool) -> tuple[str, datetime, str]:
    r = rng.random()
    if r < 0.55:
        return "", tl, "logged"
    if r < 0.85:
        mins = rng.choice([2, 3, 5, 10, 15])
        if hinglish:
            num = HINDI_MIN[mins] if rng.random() < 0.6 else str(mins)
            return f", {num} minute pehle", tl - timedelta(minutes=mins), "relative"
        lead = rng.choice(["", "about ", "like "])
        tail = rng.choice(["minutes ago", "minutes back"])
        return f", {lead}{int_words(mins)} {tail}", tl - timedelta(minutes=mins), "relative"
    t = (tl - timedelta(minutes=rng.choice([4, 7, 12]))).replace(second=0, microsecond=0)
    return f" at {t:%H:%M}", t, "explicit"


def med_event(drug, dose, unit, route, spoken, t_said, t_src, indication):
    return {"event_type": "medication", "drug": drug, "drug_as_spoken": spoken, "dose": float(dose),
            "unit": unit, "route": route, "indication": indication,
            "time_said": t_said.isoformat(), "time_source": t_src}


def add(text: str, tl: datetime, events: list[dict], tags: list[str]):
    cases.append({"id": f"BB-{len(cases) + 1:03d}", "text": text, "time_logged": tl.isoformat(),
                  "events": events, "tags": sorted(set(tags))})


def make_med(i: int, hinglish: bool = False, no_unit: bool = False):
    drug, ind, route, doses, unit = rng.choice(
        [m for m in MEDS if not no_unit or m[0] in ("aspirin", "ondansetron", "amiodarone", "morphine", "adrenaline")])
    dose = rng.choice(doses)
    brand = hinglish and drug in SPOKEN_BRAND and rng.random() < 0.7
    spoken = rng.choice(SPOKEN_BRAND[drug] if brand else SPOKEN_EN[drug])
    if drug == "dextrose" and spoken == "D50" and dose != 25:
        spoken = "D10"
    style = rng.choices(["digits", "words", "colloquial"], [0.3, 0.45, 0.25])[0]
    tl = logged_at(i)
    t, t_said, t_src = time_phrase(tl, hinglish)
    tmpl = rng.choice(HI_TEMPLATES if hinglish else EN_TEMPLATES)
    if "{t}" not in tmpl:
        t, t_said, t_src = "", tl, "logged"
    u = "" if no_unit else unit_word(unit, dose)
    r = rng.choice((ROUTE_WORDS_HI if hinglish else ROUTE_WORDS)[route])
    text = " ".join(tmpl.format(d=spoken, n=speak_number(dose, style), u=u, r=r, t=t).split())
    tags = ["medication", "hinglish" if hinglish else "english", f"time_{t_src}", f"numbers_{style}"]
    if brand:
        tags.append("brand_name")
    if no_unit:
        tags.append("no_unit")
    add(text, tl, [med_event(drug, dose, unit, route, spoken, t_said, t_src, ind)], tags)


i = 0
for _ in range(32):
    make_med(i); i += 1
for _ in range(14):
    make_med(i, hinglish=True); i += 1
for _ in range(6):
    make_med(i, no_unit=True); i += 1

# --- cancelled / not-yet-given: gold has NO medication event (classic false-positive trap)
CANCEL = [
    "drawing up {d} {n} {u}, not given yet",
    "{d} {n} {u} ready, holding for now",
    "about to give {d} {n} {u} {r}",
    "{d} ready hai, abhi nahi diya",
    "prepping {n} {u} {d}, wait for my call",
]
for tmpl in CANCEL:
    drug, ind, route, doses, unit = rng.choice(MEDS)
    dose = rng.choice(doses)
    text = tmpl.format(d=rng.choice(SPOKEN_EN[drug]), n=speak_number(dose, "words"),
                       u=unit_word(unit, dose), r=rng.choice(ROUTE_WORDS[route]))
    add(text, logged_at(i), [], ["cancelled", "hinglish" if "hai" in tmpl else "english"]); i += 1

# --- multi-event utterances
tl = logged_at(i); i += 1
add("aspirin three twenty four chewed and nitro point four under the tongue", tl,
    [med_event("aspirin", 324, "mg", "PO", "aspirin", tl, "logged", "acute_coronary_syndrome"),
     med_event("nitroglycerin", 0.4, "mg", "SL", "nitro", tl, "logged", "acute_coronary_syndrome")],
    ["multi_event", "english", "numbers_colloquial"])
tl = logged_at(i); i += 1
add("zofran four milligrams IV then fentanyl fifty mics IV", tl,
    [med_event("ondansetron", 4, "mg", "IV", "zofran", tl, "logged", "nausea_vomiting"),
     med_event("fentanyl", 50, "mcg", "IV", "fentanyl", tl, "logged", "pain")],
    ["multi_event", "english", "numbers_words"])
tl = logged_at(i); i += 1
add("Ecosprin 325 mg munh se diya aur Sorbitrate 0.4 mg jeebh ke neeche", tl,
    [med_event("aspirin", 325, "mg", "PO", "Ecosprin", tl, "logged", "acute_coronary_syndrome"),
     med_event("nitroglycerin", 0.4, "mg", "SL", "Sorbitrate", tl, "logged", "acute_coronary_syndrome")],
    ["multi_event", "hinglish", "brand_name", "numbers_digits"])
tl = logged_at(i); i += 1
add("IO access right tibia, then epi one milligram IO", tl,
    [{"event_type": "procedure", "procedure": "io_access", "params": {},
      "time_said": tl.isoformat(), "time_source": "logged"},
     med_event("adrenaline", 1, "mg", "IO", "epi", tl, "logged", "cardiac_arrest")],
    ["multi_event", "procedure", "english", "numbers_words"])
tl = logged_at(i); i += 1
add("shocked at two hundred joules and amio three hundred milligrams IV, about two minutes ago", tl,
    [{"event_type": "procedure", "procedure": "defibrillation", "params": {"joules": 200},
      "time_said": (tl - timedelta(minutes=2)).isoformat(), "time_source": "relative"},
     med_event("amiodarone", 300, "mg", "IV", "amio", tl - timedelta(minutes=2), "relative", "cardiac_arrest")],
    ["multi_event", "procedure", "english", "time_relative", "numbers_words"])
tl = logged_at(i); i += 1
add("Ondem 4 mg IV diya phir normal saline five hundred mls IV", tl,
    [med_event("ondansetron", 4, "mg", "IV", "Ondem", tl, "logged", "nausea_vomiting"),
     med_event("normal_saline", 500, "mL", "IV", "normal saline", tl, "logged", "hypoperfusion")],
    ["multi_event", "hinglish", "brand_name", "numbers_words"])

# --- procedures
PROCS = [
    ("shocked at two hundred joules", "defibrillation", {"joules": 200}, "english"),
    ("defib, one fifty joules delivered", "defibrillation", {"joules": 150}, "english"),
    ("second shock three sixty joules", "defibrillation", {"joules": 360}, "english"),
    ("shock diya 200 joules", "defibrillation", {"joules": 200}, "hinglish"),
    ("started CPR", "cpr", {}, "english"),
    ("compressions started, two minutes ago", "cpr", {}, "english"),
    ("CPR shuru kiya", "cpr", {}, "hinglish"),
    ("patient intubated, seven point five tube, first attempt", "intubation", {}, "english"),
    ("IV access left AC, eighteen gauge", "iv_access", {}, "english"),
    ("line laga di, 18 gauge", "iv_access", {}, "hinglish"),
    ("twelve lead done, sinus tach", "ecg_12_lead", {}, "english"),
]
for text, name, params, lang in PROCS:
    tl = logged_at(i); i += 1
    t_said, src = (tl - timedelta(minutes=2), "relative") if "minutes ago" in text else (tl, "logged")
    add(text, tl, [{"event_type": "procedure", "procedure": name, "params": params,
                    "time_said": t_said.isoformat(), "time_source": src}],
        ["procedure", lang] + (["time_relative"] if src == "relative" else []))

# --- vitals
VITALS = [
    ("BP one twenty over eighty, heart rate one ten, sats ninety four", {"sbp": 120, "dbp": 80, "hr": 110, "spo2": 94}, "english"),
    ("pulse forty eight, BP ninety over sixty", {"hr": 48, "sbp": 90, "dbp": 60}, "english"),
    ("sugar is forty", {"bgl_mg_dl": 40}, "english"),
    ("GRBS 45 hai", {"bgl_mg_dl": 45}, "hinglish"),
    ("GCS fourteen, resp rate twenty two", {"gcs": 14, "rr": 22}, "english"),
    ("sats eighty eight on room air", {"spo2": 88}, "english"),
    ("BP 100 by 70 hai, pulse 120", {"sbp": 100, "dbp": 70, "hr": 120}, "hinglish"),
    ("blood pressure eighty five over fifty, heart rate one thirty", {"sbp": 85, "dbp": 50, "hr": 130}, "english"),
    ("glucose reading sixty two", {"bgl_mg_dl": 62}, "english"),
    ("SpO2 97, RR 18", {"spo2": 97, "rr": 18}, "english"),
]
for text, vit, lang in VITALS:
    tl = logged_at(i); i += 1
    add(text, tl, [{"event_type": "vital", "vital": vit, "time_said": tl.isoformat(), "time_source": "logged"}],
        ["vital", lang])

# --- free-text notes (no structured clinical action)
NOTES = [
    ("patient found unresponsive in the bathroom, wife says he is diabetic", "english"),
    ("complains of crushing chest pain radiating to the left arm", "english"),
    ("mareez ko saans lene mein dikkat hai", "hinglish"),
    ("known allergy to penicillin", "english"),
    ("moving patient to the stretcher now", "english"),
]
for text, lang in NOTES:
    tl = logged_at(i); i += 1
    add(text, tl, [{"event_type": "note", "time_said": tl.isoformat(), "time_source": "logged"}], ["note", lang])


# --- HELD-OUT HARD SET: hand-written, messy, real-world style.
# RULE FOR THE TEAM: never tune the rule extractor or prompts by looking at these.
# Add new ones blind (one person writes, the others don't read them) to keep the score honest.
def M(drug, dose, unit, route, spoken, ind, mins_ago=0):
    return ("medication", dict(drug=drug, dose=float(dose), unit=unit, route=route, drug_as_spoken=spoken,
                               indication=ind), mins_ago)


def P(name, params=None):
    return ("procedure", dict(procedure=name, params=params or {}), 0)


def V(**vit):
    return ("vital", dict(vital=vit), 0)


NOTE = ("note", {}, 0)

HARD = [
    ("uh okay so epi, one milligram, IV push", [M("adrenaline", 1, "mg", "IV", "epi", "cardiac_arrest")], "english"),
    ("epi 1mg ivp", [M("adrenaline", 1, "mg", "IV", "epi", "cardiac_arrest")], "english"),
    ("ASA 324 PO", [M("aspirin", 324, "mg", "PO", "ASA", "acute_coronary_syndrome")], "english"),
    ("ntg 0.4 SL x1", [M("nitroglycerin", 0.4, "mg", "SL", "ntg", "acute_coronary_syndrome")], "english"),
    ("NS 500cc bolus wide open", [M("normal_saline", 500, "mL", "IV", "NS", "hypoperfusion")], "english"),
    ("gave him an amp of D50", [M("dextrose", 25, "g", "IV", "D50", "hypoglycemia")], "english"),
    ("versed five IM for the seizure", [M("midazolam", 5, "mg", "IM", "versed", "seizure")], "english"),
    ("fent 50mcg IN", [M("fentanyl", 50, "mcg", "IN", "fent", "pain")], "english"),
    ("zofran 4 ivp", [M("ondansetron", 4, "mg", "IV", "zofran", "nausea_vomiting")], "english"),
    ("no aspirin, he's allergic", [NOTE], "english"),
    ("narcan two mg intranasal, uh, sorry, four milligrams, four", [M("naloxone", 4, "mg", "IN", "narcan", "opioid_overdose")], "english"),
    ("pushed point five of epi, one in a thousand, IM", [M("adrenaline", 0.5, "mg", "IM", "epi", "anaphylaxis")], "english"),
    ("amiodarone three hundred IV, second dose one fifty in three minutes if still in VF",
     [M("amiodarone", 300, "mg", "IV", "amiodarone", "cardiac_arrest")], "english"),
    ("pressure's ninety systolic", [V(sbp=90)], "english"),
    ("BP 140/90 HR 88 RR 16", [V(sbp=140, dbp=90, hr=88, rr=16)], "english"),
    ("pt c/o 8/10 chest pain, gave ASA 324 chewed", [M("aspirin", 324, "mg", "PO", "ASA", "acute_coronary_syndrome")], "english"),
    ("starting compressions, epi is drawn up but hold it", [P("cpr")], "english"),
    ("glucagon one milligram IM, couldn't get a line", [M("glucagon", 1, "mg", "IM", "glucagon", "hypoglycemia")], "english"),
    ("uh we gave the morphine, like, four of it IV maybe five minutes ago",
     [M("morphine", 4, "mg", "IV", "morphine", "pain", mins_ago=5)], "english"),
    ("shocked again, same energy", [P("defibrillation")], "english"),
    ("one of epi IO, round three", [M("adrenaline", 1, "mg", "IO", "epi", "cardiac_arrest")], "english"),
    ("midazolam two and a half milligrams IV", [M("midazolam", 2.5, "mg", "IV", "midazolam", "seizure")], "english"),
    ("BP is 90 over palp", [V(sbp=90)], "english"),
    ("epinephrine 1 milligram I V given", [M("adrenaline", 1, "mg", "IV", "epinephrine", "cardiac_arrest")], "english"),
    ("Ondem ka injection 4 mg diya IV mein", [M("ondansetron", 4, "mg", "IV", "Ondem", "nausea_vomiting")], "hinglish"),
    ("Asthalin neb chalu kiya, 5 mg", [M("salbutamol", 5, "mg", "NEB", "Asthalin", "bronchospasm")], "hinglish"),
    ("Inj. Fentanyl 50 mcg IV stat", [M("fentanyl", 50, "mcg", "IV", "fentanyl", "pain")], "hinglish"),
    ("teen minute pehle Emeset 4 mg IV diya tha", [M("ondansetron", 4, "mg", "IV", "Emeset", "nausea_vomiting", mins_ago=3)], "hinglish"),
    ("patient ne Ecosprin khud li thi ghar pe, humne nahi di", [NOTE], "hinglish"),
    ("sugar bahut low hai, 38", [V(bgl_mg_dl=38)], "hinglish"),
]
for text, specs, lang in HARD:
    tl = logged_at(i); i += 1
    evs, tags = [], ["hard_heldout", lang]
    for etype, body, mins in specs:
        t_said = tl - timedelta(minutes=mins)
        evs.append({"event_type": etype, **body, "time_said": t_said.isoformat(),
                    "time_source": "relative" if mins else "logged"})
        tags.append(etype)
    add(text, tl, evs, tags)


# ------------------------------------------------------------------ validate against schema
def validate(case: dict):
    for ev in case["events"]:
        common = dict(incident_id="EVAL", event_type=ev["event_type"], time_said=ev["time_said"],
                      time_logged=case["time_logged"], time_source=ev["time_source"],
                      speaker={"id": "P1"}, audio_sha256=FAKE_AUDIO, confidence=1.0, transcript=case["text"])
        t = ev["event_type"]
        if t == "medication":
            assert BOOK.resolve_drug(ev["drug_as_spoken"]) == ev["drug"], (case["id"], ev["drug_as_spoken"])
            ClinicalEvent(**common, medication={k: ev[k] for k in ("drug", "drug_as_spoken", "dose", "unit", "route", "indication")})
        elif t == "procedure":
            ClinicalEvent(**common, procedure={"name": ev["procedure"], "parameters": ev["params"]})
        elif t == "vital":
            ClinicalEvent(**common, vital=ev["vital"])
        else:
            ClinicalEvent(**common, note=case["text"])


for c in cases:
    validate(c)

# ------------------------------------------------------------------ recording subset
# ~30 cases, stratified so every tag shows up; split round-robin across 3 teammates.
by_tag: dict[str, list[dict]] = {}
for c in cases:
    for t in c["tags"]:
        by_tag.setdefault(t, []).append(c)
chosen: dict[str, dict] = {}
for t in sorted(by_tag):
    for c in rng.sample(by_tag[t], min(2, len(by_tag[t]))):
        chosen[c["id"]] = c
rest = [c for c in cases if c["id"] not in chosen]
for c in rng.sample(rest, max(0, 30 - len(chosen))):
    chosen[c["id"]] = c
for n, cid in enumerate(sorted(chosen)):
    chosen[cid]["record"] = True
    chosen[cid]["speaker"] = f"P{n % 3 + 1}"

# ------------------------------------------------------------------ write
OUT.mkdir(parents=True, exist_ok=True)
with open(OUT / "gold.jsonl", "w", encoding="utf-8") as f:
    for c in cases:
        f.write(json.dumps(c, ensure_ascii=False) + "\n")
with open(OUT / "recording_script.csv", "w", newline="", encoding="utf-8") as f:
    w = csv.writer(f)
    w.writerow(["id", "speaker", "text", "tags"])
    for c in cases:
        if c.get("record"):
            w.writerow([c["id"], c["speaker"], c["text"], " ".join(c["tags"])])

n_ev = sum(len(c["events"]) for c in cases)
print(f"wrote {len(cases)} dictations / {n_ev} gold events -> {OUT / 'gold.jsonl'}")
print(f"wrote {len(chosen)} lines to record -> {OUT / 'recording_script.csv'}")
tag_counts = {t: len(v) for t, v in sorted(by_tag.items())}
print("tags:", tag_counts)