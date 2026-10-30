"""
Black Box Guardian - protocol rules. Each check is plain, auditable code reading protocol.json.

rule_id               severity   what
DOSE_ABOVE_MAX        critical   single dose > protocol max (per-kg for children, route overrides)
TOTAL_ABOVE_MAX       critical   cumulative dose (this one included) > max total
CONTRAINDICATED       critical   protocol contraindication: vital (SBP<90...), flag (PDE-5...), allergy, age;
                                 or a child with no pediatric dosing in the protocol
DANGEROUS_MISMATCH    critical   adrenaline 1:1000 (1 mg/mL) given IV/IO
REPEAT_TOO_SOON       warning    same drug again before the minimum repeat interval
TOO_MANY_DOSES        warning    more doses than protocol allows
ROUTE_NOT_ALLOWED     warning    route not in the protocol for this drug
DOSE_BELOW_MIN        warning    adult dose below protocol minimum
DUPLICATE             warning    same drug + same dose logged again within 45 s (double entry / two crew)
NEEDS_WEIGHT          info       child dosing needs a weight that hasn't been said
"""
from __future__ import annotations

import operator
import re
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Optional

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from normalize import words_to_numbers  # noqa: E402
from protocol import ProtocolBook, to_canonical  # noqa: E402
from schema import PatientContext  # noqa: E402

DUPLICATE_WINDOW_S = 45      # same drug + dose this close = double entry, not a deliberate repeat
SEVERITY_RANK = {"critical": 0, "warning": 1, "info": 2}
OPS = {"<": operator.lt, "<=": operator.le, ">": operator.gt, ">=": operator.ge}
UNIT_SPOKEN = {"mg": "milligrams", "mcg": "micrograms", "g": "grams", "mL": "millilitres"}
VITAL_SPOKEN = {"sbp": "systolic", "spo2": "oxygen saturation", "bgl_mg_dl": "blood sugar"}
EPI_1_IN_1000 = re.compile(r"1\s*(?:in|:|/)\s*(?:a\s*)?1,?000\b|1 mg per ml|1 mg/ml|one milligram per ml")


@dataclass
class Alert:
    rule_id: str
    severity: str
    message: str           # full sentence for the record / screen
    spoken: str            # short sentence for the speaker
    event_id: Optional[str] = None
    detail: dict = field(default_factory=dict)


@dataclass
class GivenDose:
    event_id: str
    drug: str
    dose: Optional[float]      # canonical unit
    route: Optional[str]
    time: datetime


def _fmt(x: float) -> str:
    return f"{x:g}"


def _name(drug: str) -> str:
    return drug.replace("_", " ")


def infer_indication(book: ProtocolBook, drug: str, route: Optional[str]) -> str:
    inds = book.p.drugs[drug].indications
    for name, ind in inds.items():
        if route in [r.value for r in ind.routes]:
            return name
    return next(iter(inds))


def check_medication(ev: dict, history: list[GivenDose], patient: PatientContext, vitals: dict,
                     book: ProtocolBook, text: str = "") -> list[Alert]:
    drug, route = ev.get("drug"), ev.get("route")
    if drug not in book.p.drugs:
        return []
    d = book.p.drugs[drug]
    eid = ev.get("event_id")
    t = datetime.fromisoformat(ev["time_said"])
    unit = ev.get("unit") or d.canonical_unit
    dose = None
    if ev.get("dose") is not None:
        try:
            dose = to_canonical(float(ev["dose"]), unit, d.canonical_unit)
        except ValueError:
            dose = None  # e.g. mL for a mass-dosed drug: can't compare, skip dose maths
    cu_plural = UNIT_SPOKEN.get(d.canonical_unit, d.canonical_unit)
    cu = cu_plural[:-1] if dose == 1 and cu_plural.endswith("s") else cu_plural
    alerts: list[Alert] = []
    A = lambda rid, sev, msg, spoken, **det: alerts.append(Alert(rid, sev, msg, spoken, eid, det))  # noqa: E731

    indication = infer_indication(book, drug, route)
    ind = d.indications[indication]
    prior = [h for h in history if h.drug == drug and h.event_id != eid and h.time <= t]

    # --- contraindications from the protocol file
    for c in d.contraindications:
        hit = None
        if c.type == "vital" and c.field in vitals and OPS[c.op](vitals[c.field], c.value):
            hit = f"{VITAL_SPOKEN.get(c.field, c.field)} {_fmt(vitals[c.field])}"
        elif c.type == "flag" and c.flag in patient.flags:
            hit = c.text
        elif c.type == "allergy" and any(a in c.match or a == drug for a in patient.allergies):
            hit = "allergy"
        elif c.type == "age" and patient.age_years is not None and OPS[c.op](patient.age_years, c.value):
            hit = f"age {_fmt(patient.age_years)}"
        if hit:
            A("CONTRAINDICATED", "critical", f"{_name(drug)} contraindicated: {c.text} ({hit}).",
              f"Caution. {_name(drug)} with {hit}. Contraindicated.", contraindication=c.id)

    # --- dangerous concentration / route mismatch
    if drug == "adrenaline" and route in ("IV", "IO") and EPI_1_IN_1000.search(words_to_numbers(text)):
        A("DANGEROUS_MISMATCH", "critical", "Adrenaline 1:1000 (1 mg/mL) given IV/IO - use 1:10,000 for IV.",
          "Stop. One in one thousand adrenaline must not go I V.")

    # --- route
    if route and route not in [r.value for r in ind.routes]:
        all_routes = {r.value for i in d.indications.values() for r in i.routes}
        if route not in all_routes:
            A("ROUTE_NOT_ALLOWED", "warning", f"{_name(drug)} {route} is not a protocol route "
              f"(allowed: {', '.join(sorted(all_routes))}).", f"{_name(drug)} {route} is not a protocol route.")

    # --- duplicate entry
    dup = next((h for h in reversed(prior) if dose is not None and h.dose == dose
                and 0 <= (t - h.time).total_seconds() < DUPLICATE_WINDOW_S), None)
    if dup:
        secs = int((t - dup.time).total_seconds())
        A("DUPLICATE", "warning", f"{_name(drug)} {_fmt(dose)} {d.canonical_unit} already logged {secs} s ago - duplicate?",
          f"{_name(drug)} {_fmt(dose)} {cu} already logged {secs} seconds ago. Duplicate?", duplicate_of=dup.event_id)

    # --- dose limits (adult or pediatric)
    try:
        lim = book.dose_limits(drug, indication, route or ind.routes[0].value, patient)
    except ValueError as exc:
        msg = str(exc)
        if "needs patient weight" in msg:
            A("NEEDS_WEIGHT", "info", f"Child dosing for {_name(drug)} needs the patient's weight.",
              "Child dosing needs weight. Say patient weight.")
        elif "no pediatric dosing" in msg and not any(a.rule_id == "CONTRAINDICATED" for a in alerts):
            A("CONTRAINDICATED", "critical", f"No pediatric dose for {_name(drug)} in protocol.",
              f"Caution. No child dose for {_name(drug)} in protocol.")
        lim = None

    if lim and dose is not None:
        if dose > lim.max_single + 1e-9:
            who = "this child" if lim.pediatric else "protocol"
            A("DOSE_ABOVE_MAX", "critical",
              f"{_name(drug)} {_fmt(dose)} {lim.unit} exceeds max single dose {_fmt(lim.max_single)} {lim.unit} ({who}).",
              f"Warning. {_name(drug)} {_fmt(dose)} {cu} is above the maximum of {_fmt(lim.max_single)} for {who}.",
              max_single=lim.max_single)
        elif lim.min_single is not None and dose < lim.min_single - 1e-9:
            A("DOSE_BELOW_MIN", "warning",
              f"{_name(drug)} {_fmt(dose)} {lim.unit} is below protocol minimum {_fmt(lim.min_single)} {lim.unit}.",
              f"{_name(drug)} {_fmt(dose)} {cu} is below the protocol minimum.")
        prior_total = sum(h.dose for h in prior if h.dose is not None)
        if lim.max_total is not None and prior_total + dose > lim.max_total + 1e-9:
            A("TOTAL_ABOVE_MAX", "critical",
              f"{_name(drug)} total {_fmt(prior_total + dose)} {lim.unit} exceeds max {_fmt(lim.max_total)} {lim.unit}.",
              f"Warning. {_name(drug)} total {_fmt(prior_total + dose)} {cu_plural} exceeds the maximum of {_fmt(lim.max_total)}.",
              total=prior_total + dose)
    if lim:
        if lim.max_doses is not None and len(prior) + 1 > lim.max_doses:
            A("TOO_MANY_DOSES", "warning", f"{_name(drug)} dose {len(prior) + 1} exceeds max {lim.max_doses} doses.",
              f"{_name(drug)} dose number {len(prior) + 1}. Maximum is {lim.max_doses}.")
        if lim.min_repeat_interval_min and prior and not dup:
            gap = (t - prior[-1].time).total_seconds() / 60
            if gap < lim.min_repeat_interval_min:
                A("REPEAT_TOO_SOON", "warning",
                  f"{_name(drug)} repeated after {gap:.1f} min; minimum interval {_fmt(lim.min_repeat_interval_min)} min.",
                  f"{_name(drug)} repeated after {max(1, round(gap))} minute{'s' if round(gap) != 1 else ''}. "
                  f"Minimum is {_fmt(lim.min_repeat_interval_min)}.", gap_min=round(gap, 2))

    # one alert per rule, most severe first
    seen, out = set(), []
    for a in sorted(alerts, key=lambda a: SEVERITY_RANK[a.severity]):
        if a.rule_id not in seen:
            seen.add(a.rule_id)
            out.append(a)
    return out