"""
Black Box Guardian - patient context from speech.

  "patient weight twenty kilos, six years old"  -> weight_kg=20, age_years=6
  "allergic to morphine" / "morphine se allergy"  -> allergies += [morphine]
  "took viagra last night"                         -> flags += [pde5_inhibitor_48h]
  "wazan 30 kilo, umar 10 saal"                    -> weight 30, age 10   (Hinglish)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalize import words_to_numbers  # noqa: E402
from protocol import ProtocolBook  # noqa: E402

BOOK = ProtocolBook.load()

WEIGHT_RE = re.compile(r"(?:weigh(?:t|s|ing)?|wt|wazan|vajan)\D{0,12}?(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilos|kilogram|kilograms)?"
                       r"|(\d+(?:\.\d+)?)\s*(?:kg|kgs|kilo|kilos|kilograms?)\b")
AGE_RE = re.compile(r"(\d+(?:\.\d+)?)\s*(?:-\s*)?(?:years?|yrs?|yo|saal|sal|varsh)(?:\s*-?\s*old)?\b"
                    r"|(?:age|aged|umar|umr)\D{0,6}?(\d+(?:\.\d+)?)")
MONTHS_RE = re.compile(r"(\d+)\s*(?:months?|mahine)\s*(?:-\s*)?old")
ALLERGY_RE = re.compile(r"allerg(?:ic|y|ies)\s*(?:to|for|with)?\s*([a-z0-9 ,]+)"
                        r"|([a-z0-9]+)\s*(?:se|ka|ki|ko)\s*allerg")
FLAG_PATTERNS = {
    "pde5_inhibitor_48h": r"viagra|sildenafil|tadalafil|cialis|vardenafil|levitra|erectile",
    "pregnant": r"pregnan|garbhvati",
    "long_qt": r"long qt|prolonged qt|qt prolong",
    "active_gi_bleed": r"gi bleed|gastro\w* bleed|blood in (?:vomit|stool)|melena|haematemesis|hematemesis",
    "pulmonary_edema": r"pulmonary (?:o|e)dema|crackles|crepitations|fluid in (?:the )?lungs",
    "rv_infarct": r"right ventric\w* infarct|rv infarct|inferior stemi",
}
ALLERGY_CLASSES = {"nsaid": "nsaid", "nsaids": "nsaid", "opioid": "opioid", "opioids": "opioid",
                   "benzo": "benzodiazepine", "benzos": "benzodiazepine", "benzodiazepine": "benzodiazepine",
                   "iodine": "iodine", "penicillin": "penicillin", "sulfa": "sulfa"}


STOP = {"the", "patient", "known", "is", "has", "he", "she", "took", "also", "hai", "to", "a", "an",
        "nothing", "none", "no", "nkda", "any", "nahi"}


def _allergen(word: str, strict: bool) -> str | None:
    """strict=True: only protocol drugs / known classes (used after 'and', so 'and he took…' isn't an allergy)."""
    w = word.strip().lower().rstrip("s") if word.lower().endswith("ids") else word.strip().lower()
    if not w or w in STOP:
        return None
    hit = BOOK.resolve_drug(w) or ALLERGY_CLASSES.get(w) or ALLERGY_CLASSES.get(w + "s")
    if hit or strict:
        return hit
    return w if len(w) > 3 else None


def parse_context(text: str) -> dict:
    """Return only what the utterance states: {weight_kg, age_years, allergies:[...], flags:[...]}."""
    t = words_to_numbers(text)
    out: dict = {}
    m = WEIGHT_RE.search(t)
    if m and (m.group(1) or m.group(2)):
        w = float(m.group(1) or m.group(2))
        if 0.5 <= w <= 300:
            out["weight_kg"] = w
    m = MONTHS_RE.search(t)
    if m:
        out["age_years"] = round(int(m.group(1)) / 12, 2)
    else:
        m = AGE_RE.search(t)
        if m and (m.group(1) or m.group(2)):
            a = float(m.group(1) or m.group(2))
            if 0 <= a <= 120:
                out["age_years"] = a
    allergies = []
    for m in ALLERGY_RE.finditer(t):
        chunk = m.group(1) or m.group(2) or ""
        for i, part in enumerate(re.split(r",|\band\b|\bor\b|\baur\b", chunk)):
            for word in part.split()[:2]:
                a = _allergen(word, strict=i > 0)
                if a:
                    allergies.append(a)
                    break
    if allergies:
        out["allergies"] = sorted(set(allergies))
    flags = [f for f, p in FLAG_PATTERNS.items() if re.search(p, t)]
    if flags:
        out["flags"] = flags
    return out