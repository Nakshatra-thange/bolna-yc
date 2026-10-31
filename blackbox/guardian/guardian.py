
from __future__ import annotations

import re
import sys
from collections import OrderedDict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from guardian.context import parse_context  # noqa: E402
from guardian.rules import Alert, GivenDose, check_medication  # noqa: E402
from protocol import ProtocolBook, to_canonical  # noqa: E402
from schema import PatientContext  # noqa: E402

ACK_RE = re.compile(r"^(?:ack|acknowledged?|noted|understood|copy(?: that)?|okay noted|ok noted|"
                    r"theek hai|thik hai|samajh gaya|samjha|got it)$")
OVERRIDE_RE = re.compile(r"^(?:override|overriding|over ride)\b[\s,:-]*(?:because|reason|kyunki|as|since)?[\s,:-]*(.*)$")
CANCEL_DOSE_RE = re.compile(r"^(?:cancel (?:the )?dose|dose cancel|don'?t give(?: it)?|do not give(?: it)?|"
                            r"hold (?:the )?dose|mat do|mat dena|nahi dena)$")


def parse_alert_response(text: str) -> Optional[dict]:
    t = re.sub(r"[^\w\s',:-]", " ", text.lower())
    t = re.sub(r"\s+", " ", t).strip(" ,.")
    if len(t.split()) <= 4 and ACK_RE.match(t):
        return {"action": "acknowledged", "reason": None}
    m = OVERRIDE_RE.match(t)
    if m:
        reason = m.group(1).strip(" ,.")
        return {"action": "overridden", "reason": reason} if len(reason) >= 3 else {"action": "need_reason", "reason": None}
    if len(t.split()) <= 4 and CANCEL_DOSE_RE.match(t):
        return {"action": "cancelled_dose", "reason": None}
    return None


class Guardian:
    def __init__(self, book: Optional[ProtocolBook] = None, alert_timeout_s: float = 120):
        self.book = book or ProtocolBook.load()
        self.patient = PatientContext()
        self.vitals: dict = {}
        self.given: list[GivenDose] = []
        self.pending: "OrderedDict[str, tuple[Alert, datetime]]" = OrderedDict()
        self.timeout = timedelta(seconds=alert_timeout_s)
        self._n = 0

    # ------------------------------------------------------------- context
    def observe_context(self, text: str) -> dict:
        upd = parse_context(text)
        if "weight_kg" in upd:
            self.patient.weight_kg = upd["weight_kg"]
        if "age_years" in upd:
            self.patient.age_years = upd["age_years"]
        for a in upd.get("allergies", []):
            if a not in self.patient.allergies:
                self.patient.allergies.append(a)
        for f in upd.get("flags", []):
            if f not in self.patient.flags:
                self.patient.flags.append(f)
        return upd

    # ------------------------------------------------------------- events
    def assess(self, events: list[dict], text: str = "", context_update: Optional[dict] = None
               ) -> tuple[list[dict], list[Alert]]:
        """Drop meds that were only MENTIONED as allergies, check the rest, remember them. Returns (events, alerts)."""
        allergic_now = set((context_update or {}).get("allergies", []))
        # "allergic to morphine" MENTIONS morphine - it is not an administration
        kept = [e for e in events if not (e["event_type"] == "medication" and e.get("drug") in allergic_now)]
        for e in kept:  # vitals first: "BP 85 over 50, giving nitro" must check nitro against SBP 85
            if e["event_type"] == "vital":
                self.vitals.update(e.get("vital", {}))
        alerts: list[Alert] = []
        for e in kept:
            if e["event_type"] != "medication" or not e.get("drug"):
                continue
            new = check_medication(e, self.given, self.patient, self.vitals, self.book, text)
            alerts.extend(new)
            self.register(e)
        for a in alerts:
            self._n += 1
            a.detail["alert_id"] = f"A{self._n:03d}"
        return kept, alerts

    def register(self, e: dict):
        drug = e.get("drug")
        if drug not in self.book.p.drugs:
            return
        d = self.book.p.drugs[drug]
        dose = None
        if e.get("dose") is not None:
            try:
                dose = to_canonical(float(e["dose"]), e.get("unit") or d.canonical_unit, d.canonical_unit)
            except ValueError:
                pass
        self.given.append(GivenDose(e.get("event_id", ""), drug, dose, e.get("route"),
                                    datetime.fromisoformat(e["time_said"])))

    def unregister(self, event_id: str):
        """A cancelled dose no longer counts towards totals / repeat intervals."""
        self.given = [g for g in self.given if g.event_id != event_id]

    # ------------------------------------------------------------- alert answers
    def add_pending(self, alert: Alert, now: datetime):
        if alert.severity in ("critical", "warning"):
            self.pending[alert.detail["alert_id"]] = (alert, now)

    def respond(self, text: str, now: datetime) -> Optional[list[dict]]:
        """Answer the latest alert - and every other pending alert on the SAME dose (one answer per dose).
        Returns a list of responses, [{'action': 'need_reason'}] for a bare 'override', or None."""
        if not self.pending:
            return None
        r = parse_alert_response(text)
        if r is None:
            return None
        if r["action"] == "need_reason":
            return [{"action": "need_reason"}]
        last_id, (last, _) = next(reversed(self.pending.items()))
        ids = [aid for aid, (a, _) in self.pending.items()
               if aid == last_id or (last.event_id and a.event_id == last.event_id)]
        out = []
        for aid in ids:
            alert, _ = self.pending.pop(aid)
            out.append({"action": r["action"], "reason": r["reason"], "alert_id": aid, "event_id": alert.event_id,
                        "rule_id": alert.rule_id, "said": text, "at": now.isoformat()})
        if r["action"] == "cancelled_dose" and last.event_id:
            self.unregister(last.event_id)
        return out

    def expire(self, now: datetime) -> list[dict]:
        out = []
        for aid, (a, t) in list(self.pending.items()):
            if now - t > self.timeout:
                del self.pending[aid]
                out.append({"action": "unanswered", "reason": None, "alert_id": aid, "event_id": a.event_id,
                            "rule_id": a.rule_id, "said": "", "at": now.isoformat()})
        return out

    # ------------------------------------------------------------- restart
    @classmethod
    def from_chain(cls, entries, incident_id: str, book: Optional[ProtocolBook] = None) -> "Guardian":
        """Rebuild state after a restart: patient context, vitals, and every non-cancelled dose (corrections applied)."""
        from audit.verify import materialize
        g = cls(book)
        entries = list(entries)
        for e in entries:
            d = e.data
            if d.get("incident_id") == incident_id and d["kind"] == "patient":
                p = d["payload"]
                for k in ("weight_kg", "age_years"):
                    if p.get(k) is not None:
                        setattr(g.patient, k, p[k])
                g.patient.allergies = sorted(set(g.patient.allergies) | set(p.get("allergies", [])))
                g.patient.flags = sorted(set(g.patient.flags) | set(p.get("flags", [])))
        for x in materialize(entries, incident_id):
            ev = x["event"]
            if ev["event_type"] == "vital":
                g.vitals.update(ev.get("vital", {}))
            elif ev["event_type"] == "medication" and x["status"] != "cancelled":
                g.register(dict(ev, event_id=x["event_id"]))
        return g