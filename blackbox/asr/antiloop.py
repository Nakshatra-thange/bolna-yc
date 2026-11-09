"""
Whisper sometimes gets stuck repeating a phrase ("I'm sorry I'm sorry I'm sorry ...") on short, padded or hard clips.
Those loops wreck word-error-rate and take seconds to generate. This is the safety net behind the decoder settings:
cut the transcript where a phrase starts repeating.
"""
from __future__ import annotations

import re

_MAX_PHRASE = 6


def _norm(w: str) -> str:
    return re.sub(r"[^\w']", "", w.lower())


def strip_repetition_loops(text: str) -> tuple[str, bool]:
    """Return (text, stripped). Cuts at the first place a 1-6 word phrase repeats >=3 times in a row
    (>=4 for a single word, so natural 'one one' / 'ten ten' is left alone)."""
    words = text.split()
    norm = [_norm(w) for w in words]
    for i in range(len(words)):
        for k in range(1, _MAX_PHRASE + 1):
            if i + k > len(words):
                break
            unit = norm[i:i + k]
            if not any(unit):
                continue
            reps, j = 1, i + k
            while j + k <= len(words) and norm[j:j + k] == unit:
                reps += 1
                j += k
            if reps >= (4 if k == 1 else 3):
                return " ".join(words[:i]).strip(), True
    return text.strip(), False