"""
Black Box - spoken confirm / cancel for low-confidence events.

The medic hears "Confirm: fentanyl 50 micrograms IV?" and just says "confirm" / "haan" or "cancel" / "galat".
Only a SHORT utterance made entirely of command words counts as a command, so
"no aspirin, he's allergic" is dictation, not a cancel.
Nothing is ever deleted: confirmed / cancelled / unconfirmed are new records that point at the event
(in Phase 3 they become chain entries).
"""
from __future__ import annotations

import re
from collections import OrderedDict
from datetime import datetime, timedelta
from typing import Optional

_FILLER = r"(?:\s+(?:that|it|please|karo|kar do|do|ji|this|one|dose|the dose))*"
CONFIRM_RE = re.compile(r"^(?:yes|yeah|yep|confirm(?:ed)?|correct|affirmative|that'?s right|right|"
                        r"haan|han|ha|haan ji|theek hai|thik hai|sahi hai|sahi|ok(?:ay)? confirm)" + _FILLER + r"$")
CANCEL_RE = re.compile(r"^(?:no|nope|cancel(?:led)?|wrong|scratch that|delete|negative|"
                       r"nahi|nahin|galat|galat hai|cancel karo|mat likho)" + _FILLER + r"$")

UNIT_WORDS = {"mg": "milligrams", "mcg": "micrograms", "g": "grams", "mL": "millilitres"}
ROUTE_WORDS = {"IV": "I V", "IO": "I O", "IM": "I M", "IN": "intranasal", "PO": "oral", "SL": "sublingual",
               "NEB": "nebulised", "ODT": "dissolving tablet", "SC": "subcutaneous", "ET": "via tube"}


def parse_command(text: str) -> Optional[str]:
    t = re.sub(r"[^\w\s']", " ", text.lower())
    words = t.split()
    words = [w for i, w in enumerate(words) if i == 0 or w != words[i - 1]]  # Whisper loops: "confirmed confirmed ..."
    t = " ".join(words)
    if not t or len(words) > 4:
        return None
    if CONFIRM_RE.match(t):
        return "confirm"
    if CANCEL_RE.match(t):
        return "cancel"
    return None


def speakable(ev: dict) -> str:
    if ev.get("event_type") == "clarification" and "drug" in ev.get("resolution_required", []):
        return "Medication name unknown. Say the medication name, or say cancel."
    if ev["event_type"] == "medication":
        dose = ev.get("dose")
        d = "" if dose is None else (f"{dose:g} " + UNIT_WORDS.get(ev.get("unit", ""), ev.get("unit", "")))
        r = ROUTE_WORDS.get(ev.get("route", ""), "route unknown")
        return f"Confirm {ev.get('drug') or 'unknown drug'} {d} {r}?".replace("  ", " ").replace("_", " ")
    if ev["event_type"] == "procedure":
        return f"Confirm {ev.get('procedure', '').replace('_', ' ')}?"
    return "Confirm last entry?"


class ConfirmationQueue:
    """Pending events, newest first. A command applies to the most recent pending event."""

    def __init__(self, timeout_s: float = 90):
        self.timeout = timedelta(seconds=timeout_s)
        self.pending: "OrderedDict[str, tuple[dict, datetime]]" = OrderedDict()

    def add(self, event_id: str, ev: dict, now: datetime) -> str:
        self.pending[event_id] = (ev, now)
        return speakable(ev)

    def handle(self, command: str, now: datetime, said: str = "") -> Optional[dict]:
        if not self.pending:
            return None
        event_id, (ev, _) = next(reversed(self.pending.items()))
        if command == "confirm" and "drug" in ev.get("resolution_required", []):
            return {"action": "needs_resolution", "event_id": event_id, "at": now.isoformat(), "said": said}
        self.pending.popitem(last=True)
        return {"action": "confirmed" if command == "confirm" else "cancelled", "event_id": event_id,
                "at": now.isoformat(), "said": said}

    def resolve_latest_drug(self, text: str) -> Optional[tuple[dict, str]]:
        """Resolve the newest missing-drug candidate from an explicit drug-name reply."""
        if not self.pending:
            return None
        event_id, (ev, _) = next(reversed(self.pending.items()))
        if "drug" not in ev.get("resolution_required", []):
            return None
        phrase = re.sub(r"^(?:(?:the )?(?:drug|medicine) is)\s+", "", text.strip(), flags=re.I)
        from protocol import ProtocolBook
        drug = ProtocolBook.load().resolve_drug(phrase)
        if not drug:
            return None
        del self.pending[event_id]
        return dict(ev), drug

    def expire(self, now: datetime) -> list[dict]:
        out = []
        for eid, (ev, t) in list(self.pending.items()):
            if now - t > self.timeout:
                del self.pending[eid]
                out.append({"action": "unconfirmed", "event_id": eid, "at": now.isoformat(), "said": ""})
        return out
