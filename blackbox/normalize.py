"""
Black Box - spoken-text normaliser.

Turns how medics *say* things into values code can check:
  "zero point five milligrams"   -> "0.5 milligrams"
  "three twenty four"            -> "324"          (colloquial hundreds)
  "BP one twenty over eighty"    -> "bp 120 over 80"
  "paanch minute pehle"          -> time_said = logged - 5 min

Used by the rule extractor now and by the LLM extractor's post-processing in Phase 2.
"""
from __future__ import annotations

import re
from datetime import datetime, timedelta
from typing import Optional

ONES = {"zero": 0, "oh": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
        "seven": 7, "eight": 8, "nine": 9}
TEENS = {"ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13, "fourteen": 14, "fifteen": 15,
         "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19}
TENS = {"twenty": 20, "thirty": 30, "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90}
SCALES = {"hundred": 100, "thousand": 1000}
NUM_WORDS = set(ONES) | set(TEENS) | set(TENS) | set(SCALES) | {"point", "a", "half"}

# Hindi number words, only trusted right before "minute"/"min" (since "do" is also English)
HINDI_NUM = {"ek": 1, "do": 2, "teen": 3, "char": 4, "chaar": 4, "paanch": 5, "panch": 5,
             "chhe": 6, "saat": 7, "aath": 8, "nau": 9, "das": 10, "pandrah": 15, "bees": 20,
             "tees": 30}


def _int_words(words: list[str]) -> Optional[int]:
    """Parse an integer from number words. Handles 'one twenty' / 'three twenty four' too."""
    if not words:
        return None
    # colloquial hundreds: <ones> <tens|teens> [<ones>]   e.g. "one twenty", "three twenty four"
    if (len(words) in (2, 3) and words[0] in ONES and words[0] not in ("zero", "oh")
            and (words[1] in TENS or words[1] in TEENS)):
        rest = _int_words(words[1:])
        if rest is not None and rest < 100:
            return ONES[words[0]] * 100 + rest
    total, current = 0, 0
    prev = None
    for w in words:
        if w == "a":
            current = 1
        elif w in ONES:
            if prev in ONES or prev in TEENS:
                return None  # "two three" is not a number we trust
            current += ONES[w]
        elif w in TEENS:
            current += TEENS[w]
        elif w in TENS:
            current += TENS[w]
        elif w == "hundred":
            current = (current or 1) * 100
        elif w == "thousand":
            total += (current or 1) * 1000
            current = 0
        else:
            return None
        prev = w
    return total + current


def _parse_number(words: list[str]) -> Optional[float]:
    if words == ["half"] or words == ["a", "half"]:
        return 0.5
    if "point" in words:
        i = words.index("point")
        whole = _int_words(words[:i]) if words[:i] else 0
        frac_words = words[i + 1:]
        if whole is None or not frac_words or not all(w in ONES for w in frac_words):
            return None
        return float(f"{whole}." + "".join(str(ONES[w]) for w in frac_words))
    if words and words[-1] == "half":
        head = words[:-1]
        if head and head[-1] == "a":
            head = head[:-1]  # "two and a half" -> ["two"]
        base = _int_words(head) if head else 0
        return None if base is None else base + 0.5
    v = _int_words(words)
    return None if v is None else float(v)


def _fmt(x: float) -> str:
    return str(int(x)) if x == int(x) else repr(x)


_HI_UNITS = ("ek do teen chaar char paanch panch chhe chheh che chhah saat aath nau das gyarah barah terah chaudah "
             "pandrah solah satrah atharah unnis bees ikkis baais teis chaubis pachchis pachis chhabbis sattais "
             "atthais unattis tees iktis battis taintis chauntis paintis chhattis saintis adtis unchalis chalis "
             "iktalis bayalis taintalis chauvalis paintalis chhiyalis saintalis adtalis unchas pachaas pachas "
             "ikyavan baavan tirepan chauvan pachpan chhappan sattavan atthavan unsath saath iksath baasath "
             "tirsath chausath painsath chhiyasath sadsath adsath unhattar sattar ikhattar bahattar tihattar "
             "chauhattar pachattar chhihattar satattar athattar unnasi assi ikyasi bayasi tirasi chaurasi pachasi "
             "chhiyasi sattasi atthasi navasi nabbe ikyanve baanve tiranve chauranve pachanve chhiyanve sattanve "
             "atthanve ninyanve").split()
_HI_VAL = {w: v for w, v in zip(_HI_UNITS, [
    1, 2, 3, 4, 4, 5, 5, 6, 6, 6, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 25,
    26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 45, 46, 47, 48, 49, 50, 50, 51,
    52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 66, 67, 68, 69, 70, 71, 72, 73, 74, 75, 76, 77, 78,
    79, 80, 81, 82, 83, 84, 85, 86, 87, 88, 89, 90, 91, 92, 93, 94, 95, 96, 97, 98, 99])}
_HI_VAL.update({"dedh": 1.5, "dhai": 2.5, "dhaai": 2.5, "aadha": 0.5, "adha": 0.5})
_HI_MULT = {"sau": 100, "hazaar": 1000, "hazar": 1000}
_HI_AMBIGUOUS = {"ek", "do", "teen", "char", "che", "saat", "nau", "das", "bees", "tees", "saath"}   # also real words
_HI_NEXT_OK = {"milligram", "milligrams", "mg", "mcg", "microgram", "micrograms", "gram", "grams", "kilo", "kilos",
               "kg", "kilogram", "ml", "millilitre", "milliliter", "joule", "joules", "j", "saal", "percent", "bpm",
               "by", "over", "minute", "minutes", "mins", "min", "unit", "units", "litre", "liter", "lpm"}


def _hindi_numbers(text: str) -> str:
    """Hinglish: 'ek sau bees by assi' -> '120 by 80'. A run converts only if it is unambiguous
    (contains a non-ambiguous word, or 'sau'/'hazaar'), or sits right before a unit / 'by' or right after 'by'."""
    toks = re.findall(r"\d+(?:\.\d+)?|[a-z]+\d*|[^\sa-z\d]", text)
    words = set(_HI_VAL) | set(_HI_MULT) | {"saade", "sade"}
    out, i = [], 0
    while i < len(toks):
        if toks[i] not in words:
            out.append(toks[i])
            i += 1
            continue
        j = i
        while j < len(toks) and toks[j] in words:
            j += 1
        run = toks[i:j]
        nxt = toks[j] if j < len(toks) else ""
        prv = out[-1] if out else ""
        trusted = (len(run) > 1 or run[0] in _HI_MULT or run[0] not in _HI_AMBIGUOUS
                   or nxt in _HI_NEXT_OK or prv in ("by", "over"))
        if run == ["saade"] or run == ["sade"] or not trusted:
            out.extend(run)
            i = j
            continue
        total = cur = 0.0
        half = False
        for t in run:
            if t in ("saade", "sade"):
                half = True
            elif t in _HI_MULT:
                if _HI_MULT[t] == 1000:
                    total += (cur or 1) * 1000
                    cur = 0
                else:
                    cur = (cur or 1) * 100
            else:
                cur += _HI_VAL[t] + (0.5 if half else 0)
                half = False
        out.append(_fmt(total + cur))
        i = j
    return " ".join(out)


def words_to_numbers(text: str) -> str:
    """Lowercase + replace runs of English (and Hinglish) number words with digits."""
    text = text.lower().replace("i.v.", "iv").replace("i.m.", "im").replace("i.o.", "io")
    text = _hindi_numbers(text)
    tokens = re.findall(r"[a-z]+\d+|[a-z]+|\d+(?:\.\d+)?|[^\sa-z\d]", text)
    out, i = [], 0
    while i < len(tokens):
        starts_a = (tokens[i] == "a" and i + 1 < len(tokens) and tokens[i + 1] in ("hundred", "thousand"))
        if starts_a or (tokens[i] in NUM_WORDS and tokens[i] not in ("a", "half", "point")):
            j = i
            while j < len(tokens) and (tokens[j] in NUM_WORDS or
                                       (tokens[j] == "and" and j + 1 < len(tokens) and tokens[j + 1] in NUM_WORDS)):
                j += 1
            run = [t for t in tokens[i:j] if t != "and"]
            # "one" before a non-number word that isn't a unit is often a pronoun; keep numeric anyway
            val = _parse_number(run)
            if val is None and len(run) > 1:
                # fall back: split "one twenty over eighty" style runs greedily
                val = _parse_number(run[:1])
                j = i + 1
            if val is not None:
                out.append(_fmt(val))
                i = j
                continue
        elif tokens[i] == "point" and i + 1 < len(tokens) and tokens[i + 1] in ONES:
            j = i + 1
            while j < len(tokens) and tokens[j] in ONES:
                j += 1
            out.append(_fmt(_parse_number(tokens[i:j])))
            i = j
            continue
        out.append(tokens[i])
        i += 1
    s = " ".join(out)
    s = re.sub(r"\s+([,.;:!?])", r"\1", s)
    s = re.sub(r"(\d) : (\d)", r"\1:\2", s)
    s = re.sub(r"(\d):\s(\d)", r"\1:\2", s)
    return s


# ------------------------------------------------------------------ time
_REL_EN = re.compile(r"(?:about |around |like |roughly |approx(?:imately)? )?"
                     r"(\d+(?:\.\d+)?)\s*(?:minutes?|mins?)\s*(?:ago|back|before)")
_REL_HI = re.compile(r"(\d+|" + "|".join(HINDI_NUM) + r")\s*(?:minute|minutes|min|mins)\s*(?:pehle|pahle|pehele)")
_HALF_HOUR = re.compile(r"half an hour (?:ago|back)|aadha ghanta pehle")
_CLOCK = re.compile(r"\bat (\d{1,2}):(\d{2})\b")
_CLOCK_AMPM = re.compile(r"\bat (\d{1,2})(?::(\d{2}))?\s*(a\.?m\.?|p\.?m\.?)\b", re.I)


def resolve_time(text: str, time_logged: datetime) -> tuple[datetime, str, Optional[tuple[int, int]]]:
    """Return (time_said, time_source, span) where span is the matched char range or None.
    text should already be passed through words_to_numbers()."""
    m = _CLOCK_AMPM.search(text)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2) or 0)
        if not 1 <= hh <= 12 or not 0 <= mm <= 59:
            return time_logged, "logged", None
        hh = hh % 12 + (12 if m.group(3).lower().startswith("p") else 0)
        t = time_logged.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t > time_logged:
            t -= timedelta(days=1)
        return t, "explicit", m.span()
    m = _CLOCK.search(text)
    if m:
        hh, mm = int(m.group(1)), int(m.group(2))
        t = time_logged.replace(hour=hh, minute=mm, second=0, microsecond=0)
        if t > time_logged:  # said "at 23:50" just after midnight
            t -= timedelta(days=1)
        return t, "explicit", m.span()
    m = _REL_HI.search(text)
    if m:
        g = m.group(1)
        n = float(g) if g[0].isdigit() else HINDI_NUM[g]
        return time_logged - timedelta(minutes=n), "relative", m.span()
    m = _REL_EN.search(text)
    if m:
        return time_logged - timedelta(minutes=float(m.group(1))), "relative", m.span()
    m = _HALF_HOUR.search(text)
    if m:
        return time_logged - timedelta(minutes=30), "relative", m.span()
    return time_logged, "logged", None
