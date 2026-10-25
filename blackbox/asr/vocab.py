"""
Black Box - drug-vocabulary biasing.

Layer 1 (before decoding, faster-whisper only):
    initial_prompt()  an example-style "dictation" prompt built from protocol.json, so Whisper
                      expects 'epi 1 mg IV' instead of 'happy one mg ivy'
    hotwords()        the distinctive drug/brand names as faster-whisper hotwords

Layer 2 (after decoding, any engine):
    LexiconCorrector  repairs mis-heard clinical terms ("blue cannon" -> glucagon, "in trimester" ->
                      intramuscular) using spelling + sound-alike similarity, ONLY where the context
                      looks like a dose (a number/unit/route nearby). Every change is returned so the
                      raw transcript and each correction can go into the audit chain.
"""
from __future__ import annotations

import re
import sys
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from protocol import ProtocolBook  # noqa: E402

BOOK = ProtocolBook.load()

# Short / everyday aliases that are too risky to fuzzy-match onto (exact matches still pass through).
NO_FUZZY = {"ms", "ns", "asa", "epi", "ntg", "gtn", "neb", "sugar", "glucose", "fluids",
            "fluid bolus", "d10", "d25", "d50", "amio", "fent", "mso4", "epipen"}

# Everyday words medics say constantly. Never rewritten into a drug/route, whatever they sound like.
PROTECTED = set("""
line lines pain chest open wide bolus tube tubes push pushed access gauge shock shocked joules lead
left right arm leg thigh tibia nose mouth tongue neck head back side blood pressure pulse sats rate
sugar breathing airway oxygen mask cpr compressions rhythm patient pt male female years old found
floor home wife husband allergic allergy allergies says said given gave give giving going ready hold
holding drawn drawing prepping again same energy round dose doses second first third repeat another
start started starting stop stopped minutes minute hours ago back now then before after about like
maybe okay sorry still check checked done moving stretcher bed room air high low normal stable
unresponsive responsive conscious alert pale sweaty vomiting nausea seizure seizing fitting cardiac
arrest bradycardic tachy sinus nebs neb wafer tablet chewed orally under over by of the and
""".split())

ROUTE_TERMS = {
    "intravenous": "intravenous", "intramuscular": "intramuscular", "intraosseous": "intraosseous",
    "intranasal": "intranasal", "sublingual": "sublingual", "nebulised": "nebulised",
}
UNIT_TERMS = {"milligram": "milligram", "milligrams": "milligrams", "micrograms": "micrograms",
              "microgram": "microgram", "milliliters": "milliliters", "grams": "grams"}

# Exact ASR confusions seen in eval runs. Add to this from score.py's "worst utterances" list.
KNOWN_CONFUSIONS = [
    (r"\broman (?:4|iv|four)\b", "IV"),
    (r"\bi(?:\. ?| )v\b\.?", "IV"),
    (r"\bi(?:\. ?| )m\b\.?", "IM"),
    (r"\bi(?:\. ?| )o\b\.?", "IO"),
    # number homophones right before a unit: "for milligrams" -> "4 milligrams"
    (r"\bfor(?=\s+(?:milligrams?|mg|micrograms?|mcg|mics?|grams?|mls?|milliliters?)\b)", "4"),
    (r"\bto{1,2}(?=\s+(?:milligrams?|mg|micrograms?|mcg|mics?|grams?|mls?|milliliters?)\b)", "2"),
    (r"\bwon(?=\s+(?:milligrams?|mg|micrograms?|mcg|grams?)\b)", "1"),
    (r"\bate(?=\s+(?:milligrams?|mg|micrograms?|mcg|grams?)\b)", "8"),
    (r"\b(?:m g|m\. ?g\.)(?=\s|$|,)", "mg"),
    (r"(?<=\d)\s?mb\b", " mg"),
    (r"\bmic (grams?)\b", r"micro\1"),
    (r"\bmilli (grams?)\b", r"milli\1"),
]


# ------------------------------------------------------------------ layer 1
@lru_cache
def initial_prompt() -> str:
    """Example-style prompt, kept well under Whisper's ~224-token prompt limit."""
    parts = []
    for drug_id, d in BOOK.p.drugs.items():
        ind = next(iter(d.indications.values()))
        spoken = d.aliases[0]
        dose = ind.adult.dose_max
        dose_s = str(int(dose)) if dose == int(dose) else str(dose)
        parts.append(f"{spoken} {dose_s} {d.canonical_unit} {ind.routes[0].value}")
    brands = ["Ecosprin", "Ondem", "Emeset", "Asthalin", "Sorbitrate", "Mezolam", "Narcan", "Versed", "Zofran"]
    return ("Paramedic dictation. " + ". ".join(parts) + ". Also: " + ", ".join(brands)
            + ". Routes IV, IO, IM, intranasal, sublingual, nebulised.")


@lru_cache
def hotwords() -> str:
    names = sorted({a for d in BOOK.p.drugs.values() for a in d.aliases
                    if len(a) > 3 and a.lower() not in NO_FUZZY})
    return " ".join(names)


# ------------------------------------------------------------------ layer 2
def phonetic(s: str) -> str:
    """Tiny sound-alike key (Metaphone-ish). 'morfeen' and 'morphine' -> 'mrfn'."""
    s = re.sub(r"[^a-z]", "", s.lower())
    if not s:
        return ""
    for a, b in [("ph", "f"), ("gh", "g"), ("ck", "k"), ("sch", "sk"), ("th", "t"), ("qu", "kw"),
                 ("x", "ks"), ("dg", "j")]:
        s = s.replace(a, b)
    s = re.sub(r"c(?=[eiy])", "s", s)
    s = s.replace("c", "k").replace("q", "k").replace("z", "s").replace("v", "f")
    first, rest = s[0], re.sub(r"[aeiouyhw]", "", s[1:])
    key = (first if first not in "aeiouyhw" else "a") + rest
    return re.sub(r"(.)\1+", r"\1", key)


def _sim(a: str, b: str) -> float:
    a2, b2 = a.replace(" ", ""), b.replace(" ", "")
    char = SequenceMatcher(None, a2, b2).ratio()
    pa, pb = phonetic(a2), phonetic(b2)
    phon = SequenceMatcher(None, pa, pb).ratio() if pa and pb else 0.0
    return 0.4 * char + 0.6 * phon


@dataclass
class Correction:
    heard: str
    fixed: str
    kind: str      # drug | route | unit | known
    score: float


class LexiconCorrector:
    def __init__(self, drug_threshold: float = 0.72, route_threshold: float = 0.74, window: int = 4):
        self.drug_t, self.route_t, self.window = drug_threshold, route_threshold, window
        self.drug_terms = {a.lower() for d in BOOK.p.drugs.values() for a in d.aliases
                           if a.lower() not in NO_FUZZY and len(a) >= 5}
        self.exact = {a.lower() for d in BOOK.p.drugs.values() for a in d.aliases} | set(ROUTE_TERMS) | set(UNIT_TERMS)
        self.exact |= {"iv", "im", "io", "sl", "po", "mg", "mcg", "ml", "and", "then", "the", "of", "in", "at",
                       "given", "gave", "minutes", "ago", "back", "okay", "patient", "pushing", "push", "about"}

    def correct(self, text: str) -> tuple[str, list[Correction]]:
        """Return (corrected_text, corrections). Only the matched spans change; everything else is untouched."""
        fixes: list[Correction] = []
        for pat, rep in KNOWN_CONFUSIONS:
            def _sub(m, rep=rep):
                new = m.expand(rep)
                if m.group(0) != new:
                    fixes.append(Correction(m.group(0), new.strip(), "known", 1.0))
                return new
            text = re.sub(pat, _sub, text, flags=re.I)

        toks = list(re.finditer(r"[A-Za-z]+|\d+(?:\.\d+)?", text))
        low = [t.group(0).lower() for t in toks]
        is_num = [t[0].isdigit() for t in low]
        edits: list[tuple[int, int, str]] = []
        i = 0
        while i < len(toks):
            best = None  # (score, n, term, kind)
            if not is_num[i] and low[i] not in self.exact:
                for n in (3, 2, 1):
                    if i + n > len(toks) or any(is_num[i:i + n]) or any(t in self.exact for t in low[i:i + n]):
                        continue
                    # protected everyday words may be PART of a mishearing ("now look so" -> naloxone),
                    # but a window made only of protected words ("chest pain", "wide open") is never touched
                    if all(t in PROTECTED for t in low[i:i + n]):
                        continue
                    if n > 1 and toks[i + n - 1].start() - toks[i].end() > 3 * (n - 1):
                        continue  # words must be adjacent in the text
                    heard = " ".join(low[i:i + n])
                    flat = heard.replace(" ", "")
                    if len(flat) < 4:
                        continue
                    near_num = any(is_num[max(0, i - self.window):i + n + self.window])
                    near_drug = any(t in self.drug_terms for t in low[max(0, i - 6):i + n + 6])
                    cand: list[tuple[str, str, float]] = []
                    if near_num and len(flat) >= 5:
                        cand += [(t, "drug", self.drug_t) for t in self.drug_terms]
                    if near_num or near_drug:
                        cand += [(t, "route", self.route_t) for t in ROUTE_TERMS]
                    if i > 0 and is_num[i - 1]:
                        cand += [(t, "unit", 0.70) for t in UNIT_TERMS]
                    for term, kind, thr in cand:
                        if not 0.6 <= len(flat) / len(term.replace(" ", "")) <= 1.6:
                            continue
                        sc = _sim(heard, term)
                        if sc >= thr and (best is None or sc > best[0]):
                            best = (sc, n, term, kind)
            if best:
                sc, n, term, kind = best
                a, b = toks[i].start(), toks[i + n - 1].end()
                fixes.append(Correction(text[a:b], term, kind, round(sc, 3)))
                edits.append((a, b, term))
                i += n
            else:
                i += 1
        for a, b, term in reversed(edits):
            text = text[:a] + term + text[b:]
        return text, fixes